"""The mitigation gateway.

Sits between the load generator and vLLM. Allowed requests are forwarded
untouched; rejected ones get a 429 or a 503 with a machine-readable reason,
and every outcome is written to ``gateway.jsonl`` keyed on the request id the
generator supplies.

The admission logic never reads the traffic label. Branching on ground truth
would make "attacker rejection rate" a measurement of the label rather than
of the defense.
"""

from __future__ import annotations

import json
import time
import uuid
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from .admission import AdmissionController, Decision
from .anomaly.scorer import build_scorer
from .backpressure import PressureController, UpstreamMetricsPoller
from .config import GatewayConfig, load_config
from .features import FeatureStore
from .obs.decision_log import DecisionLog, StateSampler
from .obs.metrics import GatewayMetrics
from .tokens import TokenCounter, TokenEstimate
from .upstream import UpstreamClient, filter_headers

_REJECT_MESSAGES = {
    "rate_limit_requests": "per-client request rate exceeded",
    "rate_limit_cost": "per-client token cost budget exceeded",
    "client_concurrency": "too many concurrent requests for this client",
    "cost_exceeds_budget": "request cost exceeds this client's entire token budget",
    "anomaly_shed": "client traffic flagged anomalous while backend is under pressure",
    "queue_pressure": "backend under pressure and this client is above its fair share",
    "queue_timeout": "no admission slot available",
    "global_capacity": "gateway at capacity",
    "body_too_large": "request body too large",
}


def create_app(
    config: GatewayConfig | None = None,
    upstream_transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    config = config or load_config()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state = app.state
        state.log.open()
        await state.log.start_flusher()
        await state.upstream.start()

        poller = None
        if config.backpressure.enabled:
            poller = UpstreamMetricsPoller(
                config.upstream.base_url,
                config.backpressure,
                # Under test the upstream is an in-process ASGI app, so the
                # poller has to reuse that transport rather than open a socket.
                client=state.upstream.client if upstream_transport is not None else None,
            )
            await poller.start()
        state.poller = poller
        state.pressure = PressureController(config.backpressure, poller)
        state.scorer = build_scorer(config.anomaly, state.features)
        await state.scorer.start()
        state.controller = AdmissionController(
            config,
            state.pressure,
            features=state.features,
            scorer=state.scorer if state.scorer.available else None,
        )
        state.sampler = StateSampler(state.log, state.controller, poller, config.logging)
        await state.sampler.start()
        state.log.write_config(
            {
                "config": config.as_dict(),
                "tokenizer_method": state.tokens.method,
                "anomaly_active": state.scorer.available,
                "started_at": time.time(),
            }
        )
        try:
            yield
        finally:
            await state.sampler.stop()
            await state.scorer.stop()
            if state.poller is not None:
                await state.poller.stop()
            await state.upstream.stop()
            await state.log.stop()

    app = FastAPI(title="DoS mitigation gateway", version="0.1.0", lifespan=lifespan)

    state = app.state
    state.config = config
    state.metrics = GatewayMetrics()
    state.tokens = TokenCounter(config.tokenizer)
    state.features = FeatureStore(config.fairshare.window_s)
    state.upstream = UpstreamClient(config.upstream, transport=upstream_transport)
    state.log = DecisionLog(config.logging)

    @app.get("/health")
    async def health() -> dict:
        """Never admission-controlled: a health probe must not be shed."""
        return {
            "status": "ok",
            "mode": config.mode,
            "inflight": state.controller.inflight,
        }

    @app.get("/metrics")
    async def metrics() -> Response:
        state.metrics.inflight.set(state.controller.inflight)
        state.metrics.pressure_level.set(int(state.pressure.level))
        state.metrics.active_clients.set(state.controller.limiters.active_count())
        if state.poller is not None:
            stale = state.poller.age_ms() > config.backpressure.metrics_stale_after_ms
            state.metrics.upstream_stale.set(1 if stale else 0)
        return Response(
            content=state.metrics.render(), media_type="text/plain; version=0.0.4"
        )

    @app.get("/gateway/state")
    async def gateway_state() -> dict:
        return {
            "mode": config.mode,
            "anomaly_active": state.scorer.available,
            "tokenizer_method": state.tokens.method,
            **state.sampler.snapshot(),
        }

    @app.get("/v1/models")
    async def models() -> Response:
        response = await state.upstream.request("GET", "/v1/models")
        return Response(
            content=response.content,
            status_code=response.status_code,
            media_type=response.headers.get("content-type"),
        )

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        return await _handle(request, "/v1/chat/completions")

    @app.post("/v1/completions")
    async def completions(request: Request) -> Response:
        return await _handle(request, "/v1/completions")

    async def _handle(request: Request, route: str) -> Response:
        started = time.perf_counter()
        raw = await request.body()
        client_id = request.headers.get(config.client_id_header) or (
            request.client.host if request.client else "unknown"
        )
        request_id = request.headers.get(config.request_id_header) or uuid.uuid4().hex
        label = request.headers.get(config.label_header)
        empty = TokenEstimate(0, 0, state.tokens.method)

        if len(raw) > config.max_body_bytes:
            decision = Decision(allowed=False, reason="body_too_large", status_code=413)
            return _reject_response(
                route, decision, client_id, request_id, label, started, empty
            )

        try:
            body = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            return JSONResponse(_error_body("bad_request", "invalid JSON body"), 400)
        if not isinstance(body, dict):
            return JSONResponse(_error_body("bad_request", "body must be an object"), 400)

        estimate = state.tokens.estimate(
            body,
            ceiling=config.rate_limit.max_tokens_ceiling,
            default_max=config.rate_limit.default_max_tokens,
        )
        prompt_hash = _prompt_hash(body)

        decision = await state.controller.acquire(client_id, estimate, prompt_hash)
        if not decision.allowed:
            return _reject_response(
                route, decision, client_id, request_id, label, started, estimate
            )

        # Clamp the forwarded max_tokens to the ceiling that was charged, so
        # the backend cannot be asked for more work than the limiter accounted.
        if config.rate_limit.enabled and isinstance(body.get("max_tokens"), int):
            if body["max_tokens"] > config.rate_limit.max_tokens_ceiling:
                body["max_tokens"] = config.rate_limit.max_tokens_ceiling
                raw = json.dumps(body).encode()

        headers = filter_headers(request.headers)
        headers[config.request_id_header] = request_id
        streaming = bool(body.get("stream"))

        if streaming:
            return await _forward_streaming(
                route, decision, headers, raw, estimate, client_id,
                request_id, label, started,
            )

        failed = False
        upstream_status = 0
        try:
            response = await state.upstream.request(
                "POST", route, headers=headers, content=raw,
                max_tokens=estimate.max_tokens,
            )
            upstream_status = response.status_code
            failed = response.status_code >= 500
            out: Response = Response(
                content=response.content,
                status_code=response.status_code,
                media_type=response.headers.get("content-type"),
            )
        except httpx.TimeoutException:
            failed, upstream_status = True, 504
            out = JSONResponse(
                _error_body("upstream_timeout", "upstream did not respond in time"), 504
            )
        except Exception as exc:
            # Broad on purpose: an unexpected error must still reach the
            # release below, or the admission slot leaks and gateway capacity
            # shrinks permanently under exactly the load that triggered it.
            failed, upstream_status = True, 502
            out = JSONResponse(_error_body("upstream_error", str(exc)), 502)
        finally:
            latency_ms = (time.perf_counter() - started) * 1000.0
            state.controller.release(decision, latency_ms, failed)
            _record(
                route, decision, client_id, request_id, label, estimate,
                upstream_status, latency_ms, streaming=False,
            )

        out.headers[config.request_id_header] = request_id
        out.headers["X-Gateway-Decision"] = "allow"
        return out

    async def _forward_streaming(
        route, decision, headers, raw, estimate, client_id, request_id, label, started
    ) -> Response:
        """Pass a streamed body through without buffering it."""
        generator = state.upstream.stream(
            "POST", route, headers=headers, content=raw, max_tokens=estimate.max_tokens
        )
        try:
            response, chunks = await generator.__anext__()
        except Exception as exc:
            latency_ms = (time.perf_counter() - started) * 1000.0
            state.controller.release(decision, latency_ms, True)
            _record(
                route, decision, client_id, request_id, label, estimate,
                502, latency_ms, streaming=True,
            )
            return JSONResponse(_error_body("upstream_error", str(exc)), 502)

        async def body_iter():
            failed = response.status_code >= 500
            try:
                async for chunk in chunks:
                    yield chunk
            except Exception:
                failed = True
                raise
            finally:
                await response.aclose()
                await generator.aclose()
                latency_ms = (time.perf_counter() - started) * 1000.0
                state.controller.release(decision, latency_ms, failed)
                _record(
                    route, decision, client_id, request_id, label, estimate,
                    response.status_code, latency_ms, streaming=True,
                )

        out = StreamingResponse(
            body_iter(),
            status_code=response.status_code,
            media_type=response.headers.get("content-type"),
        )
        out.headers[config.request_id_header] = request_id
        out.headers["X-Gateway-Decision"] = "allow"
        return out

    def _reject_response(
        route, decision, client_id, request_id, label, started, estimate
    ) -> Response:
        state.controller.release(decision, 0.0, False)
        latency_ms = (time.perf_counter() - started) * 1000.0
        _record(
            route, decision, client_id, request_id, label, estimate,
            decision.status_code, latency_ms, streaming=False,
        )
        headers = {
            config.request_id_header: request_id,
            "X-Gateway-Decision": "reject",
            "X-Gateway-Reason": decision.reason or "unknown",
        }
        if decision.retry_after_s > 0:
            headers["Retry-After"] = str(max(1, int(round(decision.retry_after_s))))
        return JSONResponse(
            _error_body(
                decision.reason or "rejected",
                _REJECT_MESSAGES.get(decision.reason or "", "request rejected by gateway"),
            ),
            status_code=decision.status_code,
            headers=headers,
        )

    def _record(
        route, decision, client_id, request_id, label, estimate,
        upstream_status, latency_ms, streaming,
    ) -> None:
        outcome = "allowed" if decision.allowed else "rejected"
        state.metrics.requests.labels(route=route, outcome=outcome).inc()
        if decision.reason:
            state.metrics.rejections.labels(reason=decision.reason).inc()
        if decision.allowed:
            state.metrics.upstream_status.labels(status=str(upstream_status)).inc()
            state.metrics.latency.observe(latency_ms / 1000.0)
        state.metrics.queue_wait.observe(decision.queue_wait_ms / 1000.0)

        record = {
            "ts": time.time(),
            "run_id": config.logging.run_id,
            "mode": config.mode,
            "request_id": request_id,
            "client_id": client_id,
            # Recorded for the analysis join only; see the module docstring.
            "traffic_label": label,
            "route": route,
            "decision": outcome,
            "reason": decision.reason,
            "http_status": upstream_status,
            "prompt_tokens_est": estimate.prompt_tokens,
            "max_tokens": estimate.max_tokens,
            "cost": estimate.prompt_tokens + estimate.max_tokens,
            "token_method": estimate.method,
            "streaming": streaming,
            "queue_wait_ms": round(decision.queue_wait_ms, 3),
            "gateway_latency_ms": round(latency_ms, 3),
            "gateway_inflight": decision.gateway_inflight,
            "slot_pool": decision.slot_pool,
            "pressure_level": decision.pressure_level,
            "pressure_triggers": list(decision.pressure_triggers),
            "upstream_running": decision.upstream_running,
            "upstream_waiting": decision.upstream_waiting,
            "upstream_stale": decision.upstream_stale,
            "cost_share": round(decision.cost_share, 4),
            "share_threshold": round(decision.share_threshold, 4),
            "active_clients": decision.active_clients,
            "tier": decision.tier,
            "anomaly_score": decision.anomaly_score,
        }
        if config.logging.log_features:
            record["features"] = [round(v, 6) for v in state.features.vector(client_id)]
        state.log.record(record)

    return app


def _prompt_hash(body: dict) -> int:
    payload = body.get("messages")
    if payload is None:
        payload = body.get("prompt", "")
    return hash(json.dumps(payload, sort_keys=True, default=str))


def _error_body(code: str, message: str) -> dict:
    return {"error": {"code": code, "message": message, "type": "gateway_rejection"}}
