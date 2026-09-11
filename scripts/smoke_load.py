"""Two-client smoke probe for the gateway.

Not the load generator. The real traffic profiles (normal, spike, flood,
low-and-slow) live on the load-generation side; this exists so the gateway
can be verified over real sockets before that lands, and as a fast check
that a config change did not break admission.

    python -m gateway.fake_vllm --port 8000 --max-num-seqs 8 &
    python -m gateway --config configs/ratelimit_queue.yaml --run-id smoke &
    python scripts/smoke_load.py --target http://127.0.0.1:8080
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time
import uuid
from collections import Counter

import httpx

CHEAP = "what is an inference server?"
EXPENSIVE = "explain the inference server architecture in exhaustive detail " * 60


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(q * (len(ordered) + 1)) - 1))
    return ordered[index]


async def send(client, target, client_id, label, prompt, max_tokens, results):
    body = {
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    headers = {
        "X-Client-ID": client_id,
        "X-Traffic-Label": label,
        "X-Request-ID": uuid.uuid4().hex,
    }
    started = time.perf_counter()
    try:
        response = await client.post(
            f"{target}/v1/chat/completions", json=body, headers=headers
        )
        results.append(
            {
                "label": label,
                "status": response.status_code,
                "reason": response.headers.get("X-Gateway-Reason"),
                "latency": time.perf_counter() - started,
            }
        )
    except Exception as exc:
        results.append(
            {
                "label": label,
                "status": type(exc).__name__,
                "reason": None,
                "latency": time.perf_counter() - started,
            }
        )


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="http://127.0.0.1:8080")
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--attacker-concurrency", type=int, default=8)
    parser.add_argument("--attacker-max-tokens", type=int, default=1024)
    # More than one attacker identity defeats the per-client caps, leaving
    # the global admission width and fair-share as the only defense.
    parser.add_argument("--attacker-clients", type=int, default=1)
    parser.add_argument("--legit-interval", type=float, default=0.4)
    args = parser.parse_args()

    results: list[dict] = []
    deadline = time.perf_counter() + args.seconds
    # No retries on 429/503: a retrying client turns rejections into
    # amplification and makes the rejection rate meaningless.
    limits = httpx.Limits(max_connections=200, max_keepalive_connections=200)

    async with httpx.AsyncClient(timeout=120.0, limits=limits) as client:

        async def attacker():
            wave = 0
            while time.perf_counter() < deadline:
                await asyncio.gather(
                    *[
                        send(
                            client, args.target,
                            f"attacker-{(wave * args.attacker_concurrency + i) % args.attacker_clients + 1}",
                            "attacker", EXPENSIVE, args.attacker_max_tokens, results,
                        )
                        for i in range(args.attacker_concurrency)
                    ]
                )
                wave += 1

        async def legit():
            await asyncio.sleep(1.0)
            while time.perf_counter() < deadline:
                await send(client, args.target, "legit-1", "legit", CHEAP, 32, results)
                await asyncio.sleep(args.legit_interval)

        await asyncio.gather(attacker(), legit())

    for label in ("legit", "attacker"):
        rows = [r for r in results if r["label"] == label]
        if not rows:
            continue
        ok = [r for r in rows if r["status"] == 200]
        latencies = [r["latency"] for r in ok]
        rejected = [r for r in rows if r["status"] in (429, 503)]
        print(f"\n=== {label} ({len(rows)} requests) ===")
        print(f"  success rate   {len(ok) / len(rows):.1%}")
        print(f"  rejection rate {len(rejected) / len(rows):.1%}")
        if latencies:
            print(f"  p50 {statistics.median(latencies):.3f}s  "
                  f"p95 {percentile(latencies, 0.95):.3f}s  "
                  f"max {max(latencies):.3f}s")
        reasons = Counter(r["reason"] for r in rows if r["reason"])
        if reasons:
            print(f"  reasons        {dict(reasons)}")
        statuses = Counter(str(r["status"]) for r in rows)
        print(f"  statuses       {dict(statuses)}")


if __name__ == "__main__":
    asyncio.run(main())
