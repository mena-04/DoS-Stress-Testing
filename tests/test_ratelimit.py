from gateway.config import RateLimitConfig
from gateway.ratelimit import ClientLimiter, LimiterRegistry, TokenBucket


def test_bucket_starts_full_and_drains():
    bucket = TokenBucket(rate=1.0, capacity=5.0, now=0.0)
    assert all(bucket.try_consume(1.0, 0.0) for _ in range(5))
    assert not bucket.try_consume(1.0, 0.0)


def test_bucket_refills_at_rate_and_caps_at_capacity():
    bucket = TokenBucket(rate=2.0, capacity=4.0, now=0.0)
    assert bucket.try_consume(4.0, 0.0)
    assert bucket.available(1.0) == 2.0
    assert bucket.available(100.0) == 4.0


def test_retry_after_reflects_deficit():
    bucket = TokenBucket(rate=2.0, capacity=2.0, now=0.0)
    bucket.try_consume(2.0, 0.0)
    assert bucket.retry_after(1.0, 0.0) == 0.5


def test_cost_bucket_catches_low_and_slow():
    """One request per second is trivially under the rate limit.

    With max_tokens=1024 it is not under the cost limit, which is the whole
    point: the low-and-slow profile is invisible to request-rate limiting.
    """
    config = RateLimitConfig(
        requests_per_sec=5.0, requests_burst=10.0,
        tokens_per_sec=100.0, tokens_burst=1200.0,
    )
    limiter = ClientLimiter("slow", config, now=0.0)
    assert limiter.check(cost=1024, max_inflight=8, now=0.0).allowed
    verdict = limiter.check(cost=1024, max_inflight=8, now=1.0)
    assert not verdict.allowed
    assert verdict.reason == "rate_limit_cost"


def test_request_costing_more_than_the_whole_bucket_is_named_distinctly():
    """Otherwise a misconfigured ceiling looks like a transient rate limit
    that never clears, however long the client backs off."""
    config = RateLimitConfig(tokens_per_sec=100.0, tokens_burst=500.0)
    limiter = ClientLimiter("c", config, now=0.0)
    verdict = limiter.check(cost=2048, max_inflight=8, now=0.0)
    assert not verdict.allowed
    assert verdict.reason == "cost_exceeds_budget"


def test_cost_rejection_does_not_burn_request_budget():
    config = RateLimitConfig(
        requests_per_sec=1.0, requests_burst=2.0,
        tokens_per_sec=10.0, tokens_burst=10.0,
    )
    limiter = ClientLimiter("c", config, now=0.0)
    before = limiter.requests.available(0.0)
    assert not limiter.check(cost=1000, max_inflight=8, now=0.0).allowed
    assert limiter.requests.available(0.0) == before


def test_per_client_concurrency_cap():
    config = RateLimitConfig(client_max_inflight=2)
    limiter = ClientLimiter("c", config, now=0.0)
    limiter.inflight = 2
    verdict = limiter.check(cost=10, max_inflight=2, now=0.0)
    assert not verdict.allowed
    assert verdict.reason == "client_concurrency"


def test_multiplier_is_relative_to_baseline_not_cumulative():
    config = RateLimitConfig(requests_per_sec=10.0, tokens_per_sec=1000.0)
    limiter = ClientLimiter("c", config, now=0.0)
    for _ in range(3):
        limiter.apply_multiplier(0.5)
    assert limiter.requests.rate == 5.0
    limiter.apply_multiplier(1.0)
    assert limiter.requests.rate == 10.0


def test_registry_evicts_idle_clients():
    """Identity rotation must not grow the limiter map without bound."""
    now = [0.0]
    config = RateLimitConfig(client_ttl_s=1.0)
    registry = LimiterRegistry(config, clock=lambda: now[0])
    for i in range(600):
        registry.get(f"client-{i}")
    now[0] = 10.0
    registry.get("trigger-sweep")
    for i in range(600):
        registry.get(f"later-{i}")
    assert len(registry) < 700


def test_registry_keeps_clients_with_inflight_requests():
    now = [0.0]
    config = RateLimitConfig(client_ttl_s=1.0)
    registry = LimiterRegistry(config, clock=lambda: now[0])
    registry.get("busy").inflight = 1
    now[0] = 100.0
    registry._sweep(now[0])
    assert "busy" in registry._clients
