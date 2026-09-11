"""Prometheus metrics for the gateway itself, served at ``/metrics``.

Deliberately on a private registry: the gateway must not shadow or mix with
vLLM's ``vllm:*`` series, which the analysis side scrapes separately.
"""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest


class GatewayMetrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "gateway_requests_total",
            "Requests seen by the gateway.",
            ["route", "outcome"],
            registry=self.registry,
        )
        self.rejections = Counter(
            "gateway_rejections_total",
            "Rejections by reason.",
            ["reason"],
            registry=self.registry,
        )
        self.upstream_status = Counter(
            "gateway_upstream_status_total",
            "Upstream response status classes.",
            ["status"],
            registry=self.registry,
        )
        self.queue_wait = Histogram(
            "gateway_queue_wait_seconds",
            "Time spent waiting for an admission slot.",
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 1.0),
            registry=self.registry,
        )
        self.latency = Histogram(
            "gateway_request_latency_seconds",
            "End-to-end latency observed by the gateway.",
            buckets=(0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 30, 60, 120),
            registry=self.registry,
        )
        self.inflight = Gauge(
            "gateway_requests_inflight",
            "Requests currently forwarded upstream.",
            registry=self.registry,
        )
        self.pressure_level = Gauge(
            "gateway_pressure_level",
            "0 normal, 1 elevated, 2 critical.",
            registry=self.registry,
        )
        self.active_clients = Gauge(
            "gateway_active_clients",
            "Clients with live rate-limit state.",
            registry=self.registry,
        )
        self.upstream_stale = Gauge(
            "gateway_upstream_metrics_stale",
            "1 when the upstream metrics snapshot is older than the staleness bound.",
            registry=self.registry,
        )
        self.metrics_failures = Counter(
            "gateway_upstream_metrics_failures_total",
            "Failed or unparseable upstream metrics polls.",
            registry=self.registry,
        )

    def render(self) -> bytes:
        return generate_latest(self.registry)
