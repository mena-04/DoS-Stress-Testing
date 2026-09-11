"""Queue-pressure detection and fast rejection.

Signal priority is deliberate. The gateway's own in-flight count is primary:
it is instant, free, and cannot go stale. vLLM's ``/metrics`` is secondary,
because the first observed overload run showed that endpoint timing out under
exactly the load it is supposed to report (180 samples, all recording
``num_requests_running = 0`` while 100 requests were demonstrably in flight).

Hence the staleness guard: when the upstream snapshot ages out, its signals
are dropped rather than read as zero. Treating absent metrics as "queue
empty" would switch off shedding at the precise moment it is needed.
"""

from __future__ import annotations

import asyncio
import enum
import re
import time
from dataclasses import dataclass, field

import httpx

from .config import BackpressureConfig

_METRIC_LINE = r"^{name}(?:\{{[^}}]*\}})?\s+([0-9.eE+-]+)\s*$"


class PressureLevel(enum.IntEnum):
    NORMAL = 0
    ELEVATED = 1  # fair-share shedding only
    CRITICAL = 2  # fair-share shedding plus global fast rejection


@dataclass
class UpstreamSnapshot:
    ts: float = 0.0
    running: float | None = None
    waiting: float | None = None
    waiting_capacity: float | None = None
    ok: bool = False
    parse_failures: int = 0

    @property
    def valid(self) -> bool:
        return self.ok and self.running is not None


@dataclass
class PressureReport:
    level: PressureLevel
    signals: dict[str, float] = field(default_factory=dict)
    triggers: tuple[str, ...] = ()
    upstream_stale: bool = False
    snapshot: UpstreamSnapshot = field(default_factory=UpstreamSnapshot)


def parse_metric(text: str, name: str) -> float | None:
    """Sum a Prometheus metric across its label sets.

    Anchored on the full metric name so ``vllm:num_requests_waiting`` does not
    also match ``vllm:num_requests_waiting_by_reason``, and tolerant of
    unlabelled lines so a missing ``model_name`` label does not read as zero.
    """
    pattern = _METRIC_LINE.format(name=re.escape(name))
    matches = re.findall(pattern, text, re.MULTILINE)
    if not matches:
        return None
    total = 0.0
    for value in matches:
        try:
            total += float(value)
        except ValueError:
            continue
    return total


def parse_labelled_metric(text: str, name: str, label: str, value: str) -> float | None:
    pattern = (
        rf"^{re.escape(name)}\{{[^}}]*{re.escape(label)}=\"{re.escape(value)}\""
        r"[^}]*\}\s+([0-9.eE+-]+)\s*$"
    )
    match = re.search(pattern, text, re.MULTILINE)
    return float(match.group(1)) if match else None


class UpstreamMetricsPoller:
    """Polls vLLM ``/metrics`` on a fixed interval, never on the request path."""

    def __init__(
        self,
        base_url: str,
        config: BackpressureConfig,
        client: httpx.AsyncClient | None = None,
        clock=time.monotonic,
    ) -> None:
        self._url = base_url.rstrip("/") + "/metrics"
        self._config = config
        self._client = client
        self._owns_client = client is None
        self._clock = clock
        self._snapshot = UpstreamSnapshot()
        self._task: asyncio.Task | None = None
        self.poll_count = 0
        self.failure_count = 0

    @property
    def snapshot(self) -> UpstreamSnapshot:
        return self._snapshot

    def age_ms(self) -> float:
        # Keyed on ``ok`` rather than a zero timestamp: a monotonic clock can
        # legitimately read 0.0 just after process start.
        if not self._snapshot.ok:
            return float("inf")
        return (self._clock() - self._snapshot.ts) * 1000.0

    async def start(self) -> None:
        if self._client is None:
            timeout = httpx.Timeout(self._config.metrics_timeout_ms / 1000.0)
            self._client = httpx.AsyncClient(timeout=timeout)
        self._task = asyncio.create_task(self._loop(), name="upstream-metrics-poller")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def poll_once(self) -> UpstreamSnapshot:
        assert self._client is not None
        self.poll_count += 1
        try:
            response = await self._client.get(
                self._url, timeout=self._config.metrics_timeout_ms / 1000.0
            )
            text = response.text
        except Exception:
            # Counted, not swallowed. A silent drop here is what made the
            # first overload run look idle.
            self.failure_count += 1
            return self._snapshot

        running = parse_metric(text, "vllm:num_requests_running")
        waiting = parse_metric(text, "vllm:num_requests_waiting")
        capacity = parse_labelled_metric(
            text, "vllm:num_requests_waiting_by_reason", "reason", "capacity"
        )
        snapshot = UpstreamSnapshot(
            ts=self._clock(),
            running=running,
            waiting=waiting,
            waiting_capacity=capacity,
            ok=running is not None or waiting is not None,
            parse_failures=sum(1 for v in (running, waiting) if v is None),
        )
        if not snapshot.ok:
            self.failure_count += 1
            return self._snapshot
        self._snapshot = snapshot
        return snapshot

    async def _loop(self) -> None:
        interval = self._config.metrics_poll_interval_ms / 1000.0
        while True:
            await self.poll_once()
            await asyncio.sleep(interval)


class PressureController:
    """Fuses local and upstream signals into a level, with hysteresis."""

    def __init__(
        self,
        config: BackpressureConfig,
        poller: UpstreamMetricsPoller | None = None,
        clock=time.monotonic,
    ) -> None:
        self._config = config
        self._poller = poller
        self._clock = clock
        self._level = PressureLevel.NORMAL
        self._latency_ewma_ms = 0.0
        self.inflight = 0

    @property
    def level(self) -> PressureLevel:
        return self._level

    @property
    def latency_ewma_ms(self) -> float:
        return self._latency_ewma_ms

    def observe_latency(self, latency_ms: float) -> None:
        alpha = self._config.latency_ewma_alpha
        if self._latency_ewma_ms == 0.0:
            self._latency_ewma_ms = latency_ms
        else:
            self._latency_ewma_ms = alpha * latency_ms + (1 - alpha) * self._latency_ewma_ms

    def evaluate(self) -> PressureReport:
        config = self._config
        snapshot = self._poller.snapshot if self._poller else UpstreamSnapshot()
        stale = True
        if self._poller is not None and snapshot.valid:
            stale = self._poller.age_ms() > config.metrics_stale_after_ms

        local_occupancy = self.inflight / max(1, config.global_max_inflight)
        signals: dict[str, float] = {
            "local_occupancy": local_occupancy,
            "latency_ewma_ms": self._latency_ewma_ms,
        }
        # (value, high, low) per signal. Local signals are always present.
        checks: list[tuple[str, float, float, float]] = [
            ("local_occupancy", local_occupancy, config.occupancy_high, config.occupancy_low),
            (
                "latency_ewma_ms",
                self._latency_ewma_ms,
                config.latency_ewma_high_ms,
                config.latency_ewma_low_ms,
            ),
        ]
        if not stale:
            waiting = snapshot.waiting or 0.0
            occupancy = (snapshot.running or 0.0) / max(1, config.upstream_max_num_seqs)
            signals["upstream_waiting"] = waiting
            signals["upstream_occupancy"] = occupancy
            signals["upstream_running"] = snapshot.running or 0.0
            checks.append(
                ("upstream_waiting", waiting, config.upstream_waiting_high, config.upstream_waiting_low)
            )
            checks.append(
                ("upstream_occupancy", occupancy, config.occupancy_high, config.occupancy_low)
            )

        triggers = tuple(name for name, value, high, _ in checks if value >= high)
        all_clear = all(value < low for _, value, _, low in checks)

        if triggers:
            self._level = PressureLevel.CRITICAL
        elif self._level == PressureLevel.CRITICAL and not all_clear:
            # Hold until every signal clears its low mark, otherwise the
            # controller oscillates between shedding and admitting each tick.
            self._level = PressureLevel.CRITICAL
        elif all_clear:
            self._level = PressureLevel.NORMAL
        else:
            self._level = PressureLevel.ELEVATED

        return PressureReport(
            level=self._level,
            signals=signals,
            triggers=triggers,
            upstream_stale=stale,
            snapshot=snapshot,
        )
