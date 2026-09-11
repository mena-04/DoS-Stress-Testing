"""Admission control pipeline.

Checks run cheapest-first and nothing awaits the upstream before a rejection,
so a denial costs microseconds instead of a scheduler slot:

  1. per-client budget  (dual token bucket, per-client in-flight cap)  -> 429
  2. fair-share         (only while under pressure)                    -> 503
  3. global admission   (bounded wait for a slot)                      -> 503

Rate limiting precedes slot acquisition deliberately: a client that is over
budget must not occupy admission capacity on its way to being rejected.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from .backpressure import PressureController, PressureLevel
from .config import GatewayConfig
from .fairshare import FairShareController
from .features import FeatureStore, RequestEvent
from .ratelimit import ClientLimiter, LimiterRegistry
from .slots import AdmissionSlots
from .tokens import TokenEstimate

# 429 means "you, specifically, are over budget"; 503 means "the system is
# saturated". Person 2 needs attacker rejection separable from global shedding.
_STATUS_BY_REASON = {
    "rate_limit_requests": 429,
    "rate_limit_cost": 429,
    "client_concurrency": 429,
    "cost_exceeds_budget": 429,
    "anomaly_shed": 429,
    "queue_pressure": 503,
    "queue_timeout": 503,
    "global_capacity": 503,
    "body_too_large": 413,
}


@dataclass
class Decision:
    allowed: bool
    reason: str | None = None
    status_code: int = 200
    retry_after_s: float = 0.0
    tier: str = "normal"
    anomaly_score: float | None = None
    pressure_level: int = 0
    pressure_triggers: tuple[str, ...] = ()
    upstream_stale: bool = True
    upstream_running: float | None = None
    upstream_waiting: float | None = None
    cost_share: float = 0.0
    share_threshold: float = 1.0
    active_clients: int = 0
    queue_wait_ms: float = 0.0
    gateway_inflight: int = 0
    slot_pool: str | None = None
    signals: dict[str, float] = field(default_factory=dict)
    _limiter: ClientLimiter | None = None
    _event: RequestEvent | None = None
    _holds_slot: bool = False


class AdmissionController:
    def __init__(
        self,
        config: GatewayConfig,
        pressure: PressureController,
        features: FeatureStore | None = None,
        scorer=None,
        clock=time.monotonic,
    ) -> None:
        self._config = config
        self._clock = clock
        self.limiters = LimiterRegistry(config.rate_limit, clock=clock)
        self.features = features or FeatureStore(config.fairshare.window_s, clock=clock)
        self.pressure = pressure
        self.fairshare = FairShareController(config.fairshare, self.features)
        self.scorer = scorer
        self.slots = AdmissionSlots(
            config.backpressure.global_max_inflight,
            config.backpressure.reserved_cheap_slots,
            config.backpressure.cheap_cost_threshold,
        )
        self.inflight = 0

    async def acquire(
        self, client_id: str, estimate: TokenEstimate, prompt_hash: int
    ) -> Decision:
        config = self._config
        now = self._clock()
        decision = Decision(allowed=True, gateway_inflight=self.inflight)

        limiter = self.limiters.get(client_id)
        decision._limiter = limiter

        if self.scorer is not None:
            tier, multiplier, score = self.scorer.tier(client_id)
            decision.tier = tier
            decision.anomaly_score = score
            if multiplier < 1.0:
                limiter.apply_multiplier(multiplier)
            elif multiplier >= 1.0:
                limiter.apply_multiplier(1.0)
            if tier == "hostile" and self.pressure.level >= PressureLevel.ELEVATED:
                # Only shed on the anomaly signal when the backend is actually
                # under strain; a high score on an idle server harms nobody.
                return self._reject(decision, "anomaly_shed", retry_after_s=2.0)

        if config.rate_limit.enabled:
            verdict = limiter.check(
                estimate.cost, config.rate_limit.client_max_inflight, now
            )
            if not verdict.allowed:
                return self._reject(
                    decision, verdict.reason or "rate_limit_requests", verdict.retry_after_s
                )

        if config.backpressure.enabled:
            report = self.pressure.evaluate()
            decision.pressure_level = int(report.level)
            decision.pressure_triggers = report.triggers
            decision.upstream_stale = report.upstream_stale
            decision.upstream_running = report.snapshot.running
            decision.upstream_waiting = report.snapshot.waiting
            decision.signals = report.signals

            if report.level >= PressureLevel.ELEVATED:
                share = self.fairshare.evaluate(client_id)
                decision.cost_share = share.share
                decision.share_threshold = share.threshold
                decision.active_clients = share.active_clients
                if share.over_share:
                    self._refund(limiter, estimate)
                    return self._reject(decision, "queue_pressure", retry_after_s=1.0)

            started = self._clock()
            pool = await self.slots.acquire(
                estimate.cost, config.backpressure.queue_wait_ms / 1000.0
            )
            decision.queue_wait_ms = (self._clock() - started) * 1000.0
            if pool is None:
                self._refund(limiter, estimate)
                return self._reject(decision, "queue_timeout", retry_after_s=1.0)
            decision.slot_pool = pool
            decision._holds_slot = True

        limiter.inflight += 1
        self.inflight += 1
        self.pressure.inflight = self.inflight
        window = self.features.window(client_id)
        window.concurrency += 1
        decision._event = self.features.record_admitted(
            client_id, estimate.prompt_tokens, estimate.max_tokens, prompt_hash
        )
        decision.gateway_inflight = self.inflight
        return decision

    def release(self, decision: Decision, latency_ms: float, failed: bool) -> None:
        if decision._holds_slot and decision.slot_pool is not None:
            # slot_pool itself is kept for the log record, which is written
            # after release.
            self.slots.release(decision.slot_pool)
            decision._holds_slot = False
        if not decision.allowed:
            return
        limiter = decision._limiter
        if limiter is not None:
            limiter.inflight = max(0, limiter.inflight - 1)
            window = self.features.window(limiter.client_id)
            window.concurrency = max(0, window.concurrency - 1)
        self.inflight = max(0, self.inflight - 1)
        self.pressure.inflight = self.inflight
        if decision._event is not None:
            self.features.complete(decision._event, latency_ms, failed)
        if not failed:
            self.pressure.observe_latency(latency_ms)

    def _refund(self, limiter: ClientLimiter, estimate: TokenEstimate) -> None:
        """Return budget taken by a check that a later stage then rejected.

        Without this, a client shed for global pressure would also lose the
        rate budget it never spent, compounding one rejection into several.
        """
        if not self._config.rate_limit.enabled:
            return
        now = self._clock()
        limiter.requests._tokens = min(
            limiter.requests.capacity, limiter.requests.available(now) + 1.0
        )
        limiter.cost._tokens = min(
            limiter.cost.capacity, limiter.cost.available(now) + estimate.cost
        )

    def _reject(self, decision: Decision, reason: str, retry_after_s: float = 0.0) -> Decision:
        decision.allowed = False
        decision.reason = reason
        decision.status_code = _STATUS_BY_REASON.get(reason, 503)
        decision.retry_after_s = retry_after_s
        return decision
