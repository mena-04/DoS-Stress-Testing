"""A vLLM stand-in for developing and testing the gateway without a GPU.

Calibrated against the first real measurements on a Tesla T4 running
Qwen2.5-0.5B-Instruct: ~35 ms/token at batch 1 (25 tokens in 0.887 s) rising
to ~44 ms/token at batch 100 (128 tokens in 5.6 s). Batching is cheap, which
is precisely why request rate is the wrong thing to limit and slot occupancy
is the right thing.

It reproduces the three behaviours that matter for the defense:

  * ``--max-num-seqs`` admission width, so a real queue forms and
    ``num_requests_waiting`` becomes a number that moves
  * optional prefix caching, so identical prompts skip prefill the way the
    real server does by default
  * optional ``/metrics`` stalling under load, which is what made the first
    overload run look idle

    python -m gateway.fake_vllm --port 8000 --max-num-seqs 16
"""

from __future__ import annotations

import argparse
import asyncio
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


class Engine:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.slots = asyncio.Semaphore(args.max_num_seqs)
        self.running = 0
        self.waiting = 0
        self.prefix_cache: set[int] = set()
        self.success_by_reason = {"stop": 0, "length": 0, "abort": 0, "error": 0}
        self.prompt_tokens_total = 0
        self.generation_tokens_total = 0
        self.queue_time_total = 0.0
        self.e2e_total = 0.0
        self.finished = 0

    def decode_seconds_per_token(self) -> float:
        # Mild per-token degradation with batch size, matching the measured
        # 35 ms -> 44 ms move between batch 1 and batch 100.
        return self.args.decode_s_per_token * (
            1.0 + self.args.batch_penalty * max(0, self.running - 1)
        )

    async def generate(self, prompt_tokens: int, max_tokens: int, prompt_hash: int):
        queued_at = time.perf_counter()
        self.waiting += 1
        async with self.slots:
            self.waiting -= 1
            self.running += 1
            queue_time = time.perf_counter() - queued_at
            try:
                cached = (
                    self.args.enable_prefix_caching and prompt_hash in self.prefix_cache
                )
                if self.args.enable_prefix_caching:
                    self.prefix_cache.add(prompt_hash)
                prefill = 0.0 if cached else prompt_tokens * self.args.prefill_s_per_token
                await asyncio.sleep(prefill + max_tokens * self.decode_seconds_per_token())
            finally:
                self.running -= 1

        self.prompt_tokens_total += prompt_tokens
        self.generation_tokens_total += max_tokens
        self.success_by_reason["length"] += 1
        self.queue_time_total += queue_time
        self.e2e_total += time.perf_counter() - queued_at
        self.finished += 1
        return queue_time, cached

    def render_metrics(self) -> str:
        labels = f'engine="0",model_name="{MODEL}"'
        lines = [
            "# HELP vllm:num_requests_running Number of requests currently running.",
            "# TYPE vllm:num_requests_running gauge",
            f"vllm:num_requests_running{{{labels}}} {float(self.running)}",
            "# HELP vllm:num_requests_waiting Number of requests waiting to be processed.",
            "# TYPE vllm:num_requests_waiting gauge",
            f"vllm:num_requests_waiting{{{labels}}} {float(self.waiting)}",
            "# TYPE vllm:num_requests_waiting_by_reason gauge",
            f'vllm:num_requests_waiting_by_reason{{{labels},reason="capacity"}} {float(self.waiting)}',
            f'vllm:num_requests_waiting_by_reason{{{labels},reason="deferred"}} 0.0',
            "# TYPE vllm:request_success_total counter",
        ]
        for reason, count in self.success_by_reason.items():
            lines.append(
                f'vllm:request_success_total{{{labels},finished_reason="{reason}"}} {float(count)}'
            )
        lines += [
            "# TYPE vllm:prompt_tokens_total counter",
            f"vllm:prompt_tokens_total{{{labels}}} {float(self.prompt_tokens_total)}",
            "# TYPE vllm:generation_tokens_total counter",
            f"vllm:generation_tokens_total{{{labels}}} {float(self.generation_tokens_total)}",
            "# TYPE vllm:e2e_request_latency_seconds histogram",
            f"vllm:e2e_request_latency_seconds_sum{{{labels}}} {self.e2e_total}",
            f"vllm:e2e_request_latency_seconds_count{{{labels}}} {float(self.finished)}",
            "# TYPE vllm:request_queue_time_seconds histogram",
            f"vllm:request_queue_time_seconds_sum{{{labels}}} {self.queue_time_total}",
            f"vllm:request_queue_time_seconds_count{{{labels}}} {float(self.finished)}",
        ]
        return "\n".join(lines) + "\n"


def create_app(args: argparse.Namespace) -> FastAPI:
    app = FastAPI(title="fake vLLM")
    engine = Engine(args)
    app.state.engine = engine

    @app.get("/health")
    async def health() -> Response:
        return Response(status_code=200)

    @app.get("/v1/models")
    async def models() -> dict:
        return {"object": "list", "data": [{"id": MODEL, "object": "model"}]}

    @app.get("/metrics")
    async def metrics() -> Response:
        if args.metrics_stall_ms and engine.running >= args.metrics_stall_above:
            # The behaviour that made the first real overload run record
            # num_requests_running = 0 for its entire duration.
            await asyncio.sleep(args.metrics_stall_ms / 1000.0)
        return Response(content=engine.render_metrics(), media_type="text/plain")

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        body = await request.json()
        messages = body.get("messages") or []
        text = " ".join(str(m.get("content", "")) for m in messages if isinstance(m, dict))
        prompt_tokens = max(1, len(text) // 4)
        max_tokens = int(body.get("max_tokens") or 16)
        if prompt_tokens + max_tokens > args.max_model_len:
            engine.success_by_reason["error"] += 1
            return JSONResponse(
                {
                    "error": {
                        "message": (
                            f"This model's maximum context length is {args.max_model_len} "
                            f"tokens, however you requested {prompt_tokens + max_tokens}."
                        ),
                        "type": "BadRequestError",
                        "code": 400,
                    }
                },
                status_code=400,
            )

        await engine.generate(prompt_tokens, max_tokens, hash(text))
        return JSONResponse(
            {
                "id": f"chatcmpl-{int(time.time() * 1e6)}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": body.get("model", MODEL),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "x " * max_tokens},
                        "finish_reason": "length",
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": max_tokens,
                    "total_tokens": prompt_tokens + max_tokens,
                },
            }
        )

    return app


def build_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--decode-s-per-token", type=float, default=0.035)
    parser.add_argument("--prefill-s-per-token", type=float, default=0.0002)
    parser.add_argument("--batch-penalty", type=float, default=0.0026)
    parser.add_argument(
        "--enable-prefix-caching", action="store_true", default=False,
        help="mirror vLLM's default-on prefix caching, which makes repeated prompts free",
    )
    parser.add_argument("--metrics-stall-ms", type=int, default=0)
    parser.add_argument("--metrics-stall-above", type=int, default=8)
    return parser.parse_args(argv)


def main() -> None:
    import uvicorn

    args = build_args()
    print(f"fake vLLM on http://{args.host}:{args.port} max_num_seqs={args.max_num_seqs}")
    uvicorn.run(
        create_app(args), host=args.host, port=args.port,
        log_level="warning", access_log=False,
    )


if __name__ == "__main__":
    main()
