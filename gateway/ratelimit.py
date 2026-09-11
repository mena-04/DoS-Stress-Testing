"""Per-client dual token bucket.

Two buckets per client, both refilled lazily from a monotonic clock:

  * a request bucket, which bounds arrival rate (spike, flood)
  * a cost bucket charged ``prompt_tokens + max_tokens``, which bounds work

The cost bucket is the one that catches low-and-slow. A client at 0.5 req/s
with ``max_tokens=1024`` passes any request-rate limit while occupying a
scheduler slot for tens of seconds.

Refill is lazy rather than timer-driven, so idle clients cost nothing and
there is no background task per client.
"""

from __future__ import annotations

import time
from dataclasses import dataclass


class TokenBucket:
    __slots__ = ("capacity", "rate", "_tokens", "_updated")

    def __init__(self, rate: float, capacity: float, now: float | None = None) -> None:
        self.rate = float(rate)
        self.capacity = float(capacity)
        self._tokens = float(capacity)
        self._updated = now if now is not None else time.monotonic()

    def _refill(self, now: float) -> None:
        elapsed = now - self._updated
        if elapsed <= 0:
            return
        self._updated = now
        self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)

    def available(self, now: float) -> float:
        self._refill(now)
        return self._tokens

    def try_consume(self, amount: float, now: float) -> bool:
        self._refill(now)
        if self._tokens < amount:
            return False
        self._tokens -= amount
        return True

    def retry_after(self, amount: float, now: float) -> float:
        """Seconds until ``amount`` would be available, for the Retry-After hint."""
        self._refill(now)
        if self._tokens >= amount:
            return 0.0
        if self.rate <= 0:
            return float("inf")
        return (amount - self._tokens) / self.rate

    def set_rate(self, rate: float, capacity: float | None = None) -> None:
        """Retune in place, used by anomaly-adaptive tightening."""
        self.rate = max(0.0, float(rate))
        if capacity is not None:
            self.capacity = max(0.0, float(capacity))
            self._tokens = min(self._tokens, self.capacity)


@dataclass
class LimitVerdict:
    allowed: bool
    reason: str | None = None
    retry_after_s: float = 0.0


class ClientLimiter:
    """Buckets plus in-flight accounting for one client."""

    __slots__ = ("client_id", "requests", "cost", "inflight", "last_seen", "_base")

    def __init__(self, client_id: str, config, now: float) -> None:
        self.client_id = client_id
        self.requests = TokenBucket(config.requests_per_sec, config.requests_burst, now)
        self.cost = TokenBucket(config.tokens_per_sec, config.tokens_burst, now)
        self.inflight = 0
        self.last_seen = now
        self._base = (
            config.requests_per_sec,
            config.requests_burst,
            config.tokens_per_sec,
            config.tokens_burst,
        )

    def apply_multiplier(self, multiplier: float) -> None:
        """Scale both buckets relative to their configured baseline.

        Relative to the baseline rather than the current value, so repeated
        application does not ratchet a client's limits to zero.
        """
        req_rate, req_burst, cost_rate, cost_burst = self._base
        self.requests.set_rate(req_rate * multiplier, req_burst * multiplier)
        self.cost.set_rate(cost_rate * multiplier, cost_burst * multiplier)

    def check(self, cost: float, max_inflight: int, now: float) -> LimitVerdict:
        self.last_seen = now
        if cost > self.cost.capacity:
            # A request costing more than the whole bucket could never be
            # admitted, however long the client waits. Say so distinctly
            # instead of looking like a transient rate limit forever: it means
            # tokens_burst and max_tokens_ceiling disagree.
            return LimitVerdict(False, "cost_exceeds_budget")
        if self.inflight >= max_inflight:
            return LimitVerdict(False, "client_concurrency", retry_after_s=1.0)
        # Request bucket first: it is the cheaper rejection and does not
        # consume cost budget on the way to a denial.
        if not self.requests.try_consume(1.0, now):
            return LimitVerdict(
                False, "rate_limit_requests", self.requests.retry_after(1.0, now)
            )
        if not self.cost.try_consume(cost, now):
            # Refund the request token so a cost rejection does not also
            # burn rate budget the client never got to use.
            self.requests._tokens = min(
                self.requests.capacity, self.requests._tokens + 1.0
            )
            return LimitVerdict(False, "rate_limit_cost", self.cost.retry_after(cost, now))
        return LimitVerdict(True)


class LimiterRegistry:
    """Per-client limiters with TTL eviction.

    Eviction matters under identity rotation: an attacker cycling client IDs
    would otherwise grow this map without bound.
    """

    def __init__(self, config, clock=time.monotonic) -> None:
        self._config = config
        self._clock = clock
        self._clients: dict[str, ClientLimiter] = {}
        self._since_sweep = 0

    def get(self, client_id: str) -> ClientLimiter:
        now = self._clock()
        limiter = self._clients.get(client_id)
        if limiter is None:
            limiter = ClientLimiter(client_id, self._config, now)
            self._clients[client_id] = limiter
        self._since_sweep += 1
        if self._since_sweep >= 512:
            self._since_sweep = 0
            self._sweep(now)
        return limiter

    def _sweep(self, now: float) -> None:
        ttl = self._config.client_ttl_s
        stale = [
            key
            for key, limiter in self._clients.items()
            if limiter.inflight == 0 and now - limiter.last_seen > ttl
        ]
        for key in stale:
            del self._clients[key]

    def active_count(self) -> int:
        return len(self._clients)

    def inflight_total(self) -> int:
        return sum(limiter.inflight for limiter in self._clients.values())

    def __len__(self) -> int:
        return len(self._clients)
