"""Rolling per-client behavioural windows.

Feeds two consumers: fair-share shedding needs recent cost per client right
now, and the VAE needs the same window turned into a fixed-width feature
vector. Keeping one window implementation means the anomaly detector scores
exactly the traffic the limiter acted on.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field

# Order is the VAE input layout. Appending is safe; reordering invalidates
# any trained model, so treat this tuple as the model's ABI.
FEATURE_NAMES = (
    "request_rate",
    "cost_rate",
    "mean_prompt_tokens",
    "p95_prompt_tokens",
    "mean_max_tokens",
    "token_ratio",
    "interarrival_mean",
    "interarrival_stdev",
    "concurrency",
    "repeat_ratio",
    "error_rate",
    "mean_latency_ms",
)


@dataclass
class RequestEvent:
    ts: float
    prompt_tokens: int
    max_tokens: int
    prompt_hash: int
    latency_ms: float = 0.0
    failed: bool = False
    client_id: str = ""
    open: bool = False
    expired: bool = False

    @property
    def cost(self) -> float:
        return float(self.prompt_tokens + self.max_tokens)


@dataclass
class ClientWindow:
    client_id: str
    events: deque[RequestEvent] = field(default_factory=lambda: deque(maxlen=512))
    concurrency: int = 0
    last_seen: float = 0.0
    # Cost of requests admitted before the window opened that are still
    # running. Tracked separately because they have left ``events``.
    expired_open_cost: float = 0.0

    def add(self, event: RequestEvent) -> None:
        self.events.append(event)
        self.last_seen = event.ts

    def trim(self, cutoff: float) -> None:
        while self.events and self.events[0].ts < cutoff:
            event = self.events.popleft()
            if event.open:
                event.expired = True
                self.expired_open_cost += event.cost
        if self.concurrency <= 0:
            # Nothing is running, so no expired request can still be holding
            # a slot. Also self-heals if a completion was never recorded,
            # which would otherwise charge the client forever.
            self.expired_open_cost = 0.0

    def recent_cost(self) -> float:
        """Cost attributable to this client right now.

        A generation longer than the window would otherwise age out of
        ``events`` while still occupying a slot, dropping the client to zero
        recent cost. Requests that are still running keep being charged.
        """
        return sum(e.cost for e in self.events) + self.expired_open_cost


class FeatureStore:
    def __init__(self, window_s: float = 10.0, clock=time.monotonic) -> None:
        self.window_s = window_s
        self._clock = clock
        self._windows: dict[str, ClientWindow] = {}

    def window(self, client_id: str) -> ClientWindow:
        entry = self._windows.get(client_id)
        if entry is None:
            entry = ClientWindow(client_id)
            self._windows[client_id] = entry
        return entry

    def record_admitted(
        self, client_id: str, prompt_tokens: int, max_tokens: int, prompt_hash: int
    ) -> RequestEvent:
        now = self._clock()
        event = RequestEvent(
            now, prompt_tokens, max_tokens, prompt_hash, client_id=client_id, open=True
        )
        self.window(client_id).add(event)
        return event

    def complete(self, event: RequestEvent, latency_ms: float, failed: bool) -> None:
        event.latency_ms = latency_ms
        event.failed = failed
        if not event.open:
            return
        event.open = False
        if event.expired:
            entry = self._windows.get(event.client_id)
            if entry is not None:
                entry.expired_open_cost = max(
                    0.0, entry.expired_open_cost - event.cost
                )

    def evict(self, ttl_s: float) -> None:
        now = self._clock()
        stale = [
            key
            for key, entry in self._windows.items()
            if entry.concurrency == 0 and now - entry.last_seen > ttl_s
        ]
        for key in stale:
            del self._windows[key]

    def active_clients(self, now: float | None = None) -> list[str]:
        now = self._clock() if now is None else now
        cutoff = now - self.window_s
        return [
            key
            for key, entry in self._windows.items()
            if entry.concurrency > 0 or (entry.events and entry.events[-1].ts >= cutoff)
        ]

    def cost_shares(self, now: float | None = None) -> tuple[dict[str, float], float]:
        """Recent cost per client and the window total.

        Cost is ``prompt_tokens + max_tokens``, matching what the limiter
        charges, so shares and budgets are denominated in the same unit.
        """
        now = self._clock() if now is None else now
        cutoff = now - self.window_s
        costs: dict[str, float] = {}
        total = 0.0
        for key, entry in self._windows.items():
            entry.trim(cutoff)
            cost = entry.recent_cost()
            if cost <= 0 and entry.concurrency == 0:
                continue
            costs[key] = cost
            total += cost
        return costs, total

    def vector(self, client_id: str, now: float | None = None) -> list[float]:
        now = self._clock() if now is None else now
        cutoff = now - self.window_s
        entry = self.window(client_id)
        entry.trim(cutoff)
        events = list(entry.events)
        span = max(self.window_s, 1e-6)

        if not events:
            return [0.0] * len(FEATURE_NAMES)

        prompts = [float(e.prompt_tokens) for e in events]
        maxes = [float(e.max_tokens) for e in events]
        gaps = [b.ts - a.ts for a, b in zip(events, events[1:])]
        finished = [e for e in events if e.latency_ms > 0.0]
        distinct = len({e.prompt_hash for e in events})

        return [
            len(events) / span,
            sum(p + m for p, m in zip(prompts, maxes)) / span,
            _mean(prompts),
            _percentile(prompts, 0.95),
            _mean(maxes),
            _mean(maxes) / max(1.0, _mean(prompts)),
            _mean(gaps),
            _stdev(gaps),
            float(entry.concurrency),
            1.0 - (distinct / len(events)),
            sum(1.0 for e in events if e.failed) / len(events),
            _mean([e.latency_ms for e in finished]) if finished else 0.0,
        ]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return ordered[index]
