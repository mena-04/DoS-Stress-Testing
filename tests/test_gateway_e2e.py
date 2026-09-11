"""End-to-end behaviour against the vLLM stand-in."""

import asyncio
import json

import httpx
import pytest

from .conftest import fake_vllm_app

CHEAP = "hello there, short question" * 2          # ~13 heuristic tokens
EXPENSIVE = "explain the inference server in detail " * 100  # ~975 tokens


def payload(prompt: str, max_tokens: int = 64) -> dict:
    return {
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }


def headers(client_id: str, label: str = "legit", request_id: str | None = None) -> dict:
    out = {"X-Client-ID": client_id, "X-Traffic-Label": label}
    if request_id:
        out["X-Request-ID"] = request_id
    return out


def read_log(app) -> list[dict]:
    app.state.log.decisions.flush()
    with open(app.state.log.decisions.path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


@pytest.mark.asyncio
async def test_off_mode_is_a_transparent_proxy(tuned_config, gateway):
    client, app = await gateway(tuned_config(mode="off"))
    response = await client.post(
        "/v1/chat/completions", json=payload(CHEAP), headers=headers("c1")
    )
    assert response.status_code == 200
    assert response.json()["usage"]["completion_tokens"] == 64
    assert response.headers["X-Gateway-Decision"] == "allow"

    # The baseline still produces a decision log, so the comparison between
    # mitigation off and on uses identically shaped data.
    records = read_log(app)
    assert len(records) == 1
    assert records[0]["decision"] == "allowed"
    assert records[0]["mode"] == "off"
    assert records[0]["traffic_label"] == "legit"


@pytest.mark.asyncio
async def test_off_mode_applies_no_limits(tuned_config, gateway):
    client, _ = await gateway(tuned_config(mode="off"))
    responses = await asyncio.gather(
        *[
            client.post("/v1/chat/completions", json=payload(CHEAP, 8), headers=headers("c1"))
            for _ in range(20)
        ]
    )
    assert all(r.status_code == 200 for r in responses)


@pytest.mark.asyncio
async def test_request_rate_limit_returns_429_with_reason(tuned_config, gateway):
    config = tuned_config(
        mode="ratelimit",
        **{"rate_limit.requests_per_sec": 2.0, "rate_limit.requests_burst": 3.0,
           "rate_limit.client_max_inflight": 32, "rate_limit.tokens_per_sec": 1e9,
           "rate_limit.tokens_burst": 1e9},
    )
    client, app = await gateway(config)
    responses = await asyncio.gather(
        *[
            client.post("/v1/chat/completions", json=payload(CHEAP, 8), headers=headers("burst"))
            for _ in range(12)
        ]
    )
    codes = [r.status_code for r in responses]
    assert 200 in codes and 429 in codes
    rejected = [r for r in responses if r.status_code == 429]
    assert all(r.headers["X-Gateway-Reason"] == "rate_limit_requests" for r in rejected)
    assert all("Retry-After" in r.headers for r in rejected)
    assert all(r.json()["error"]["code"] == "rate_limit_requests" for r in rejected)

    reasons = {r["reason"] for r in read_log(app) if r["decision"] == "rejected"}
    assert reasons == {"rate_limit_requests"}


@pytest.mark.asyncio
async def test_cost_limit_catches_low_and_slow(tuned_config, gateway):
    """Two requests per second is trivially under any rate limit.

    At max_tokens=1024 each it is not under the cost limit, which is the only
    reason the low-and-slow profile is stoppable at all.
    """
    config = tuned_config(
        mode="ratelimit",
        **{"rate_limit.requests_per_sec": 100.0, "rate_limit.requests_burst": 100.0,
           "rate_limit.tokens_per_sec": 100.0, "rate_limit.tokens_burst": 1500.0,
           "rate_limit.client_max_inflight": 32},
    )
    client, _ = await gateway(config)
    codes = []
    for _ in range(6):
        response = await client.post(
            "/v1/chat/completions", json=payload(CHEAP, 1024), headers=headers("slow", "attacker")
        )
        codes.append(response.status_code)
        if response.status_code == 429:
            assert response.headers["X-Gateway-Reason"] == "rate_limit_cost"
    assert 429 in codes


@pytest.mark.asyncio
async def test_max_tokens_is_clamped_to_the_charged_ceiling(tuned_config, gateway):
    config = tuned_config(
        mode="ratelimit",
        **{"rate_limit.max_tokens_ceiling": 32, "rate_limit.tokens_per_sec": 1e9,
           "rate_limit.tokens_burst": 1e9},
    )
    client, _ = await gateway(config)
    response = await client.post(
        "/v1/chat/completions", json=payload(CHEAP, 5000), headers=headers("greedy")
    )
    assert response.status_code == 200
    # The backend must not be asked for more work than the limiter accounted.
    assert response.json()["usage"]["completion_tokens"] == 32


@pytest.mark.asyncio
async def test_per_client_concurrency_cap(tuned_config, gateway):
    config = tuned_config(
        mode="ratelimit",
        **{"rate_limit.client_max_inflight": 2, "rate_limit.requests_per_sec": 1e6,
           "rate_limit.requests_burst": 1e6, "rate_limit.tokens_per_sec": 1e9,
           "rate_limit.tokens_burst": 1e9},
    )
    client, _ = await gateway(config)
    responses = await asyncio.gather(
        *[
            client.post("/v1/chat/completions", json=payload(CHEAP, 128), headers=headers("hog"))
            for _ in range(8)
        ]
    )
    reasons = {r.headers.get("X-Gateway-Reason") for r in responses if r.status_code == 429}
    assert "client_concurrency" in reasons


@pytest.mark.asyncio
async def test_legitimate_traffic_survives_a_flood(tuned_config, gateway):
    """The definition of done.

    One attacker floods with expensive requests while a legitimate client
    sends cheap ones. Legitimate requests must keep succeeding and the
    attacker must absorb the rejections.
    """
    config = tuned_config(
        mode="ratelimit_queue",
        **{
            "rate_limit.requests_per_sec": 8.0,
            "rate_limit.requests_burst": 10.0,
            "rate_limit.tokens_per_sec": 3000.0,
            "rate_limit.tokens_burst": 6000.0,
            "rate_limit.client_max_inflight": 4,
            "backpressure.global_max_inflight": 6,
            "backpressure.queue_wait_ms": 300,
            "backpressure.upstream_max_num_seqs": 4,
        },
    )
    client, app = await gateway(
        config, upstream_app=fake_vllm_app(max_num_seqs=4, decode_s_per_token=0.002)
    )

    legit_results: list[int] = []
    attacker_results: list[int] = []

    async def attacker() -> None:
        for _ in range(15):
            tasks = [
                client.post(
                    "/v1/chat/completions",
                    json=payload(EXPENSIVE, 1024),
                    headers=headers("attacker-1", "attacker"),
                )
                for _ in range(6)
            ]
            for response in await asyncio.gather(*tasks, return_exceptions=True):
                if isinstance(response, httpx.Response):
                    attacker_results.append(response.status_code)
            await asyncio.sleep(0.02)

    async def legit() -> None:
        await asyncio.sleep(0.15)  # let the attacker build pressure first
        for _ in range(25):
            response = await client.post(
                "/v1/chat/completions",
                json=payload(CHEAP, 32),
                headers=headers("legit-1", "legit"),
            )
            legit_results.append(response.status_code)
            await asyncio.sleep(0.05)

    await asyncio.gather(attacker(), legit())

    legit_success = sum(1 for code in legit_results if code == 200) / len(legit_results)
    attacker_rejected = sum(1 for code in attacker_results if code in (429, 503)) / len(
        attacker_results
    )

    assert legit_success >= 0.9, f"legitimate success rate {legit_success:.2f}"
    assert attacker_rejected >= 0.5, f"attacker rejection rate {attacker_rejected:.2f}"

    records = read_log(app)
    assert {r["client_id"] for r in records} == {"attacker-1", "legit-1"}
    # Rejections must be attributable to a mechanism, not unexplained.
    assert all(
        r["reason"] in {"rate_limit_requests", "rate_limit_cost", "client_concurrency",
                        "queue_pressure", "queue_timeout"}
        for r in records
        if r["decision"] == "rejected"
    )


@pytest.mark.asyncio
async def test_legitimate_traffic_survives_identity_rotation(tuned_config, gateway):
    """Eight attacker identities defeat every per-client limit.

    With eight IDs each one looks almost fair (a 0.125 cost share against an
    0.111 equal share), so neither the token buckets nor fair-share ranking
    engages. What keeps legitimate traffic flowing is the reserved lane,
    which partitions capacity by request cost and so is indifferent to how
    many identities the expensive traffic arrives under.
    """
    config = tuned_config(
        mode="ratelimit_queue",
        **{
            "rate_limit.requests_per_sec": 20.0,
            "rate_limit.requests_burst": 20.0,
            "rate_limit.tokens_per_sec": 1e9,
            "rate_limit.tokens_burst": 1e9,
            "rate_limit.client_max_inflight": 4,
            "backpressure.global_max_inflight": 8,
            "backpressure.reserved_cheap_slots": 3,
            "backpressure.cheap_cost_threshold": 256,
            "backpressure.queue_wait_ms": 150,
            "backpressure.upstream_max_num_seqs": 4,
        },
    )
    client, app = await gateway(
        config, upstream_app=fake_vllm_app(max_num_seqs=4, decode_s_per_token=0.002)
    )

    legit_results: list[int] = []

    async def attacker(index: int) -> None:
        for _ in range(6):
            await client.post(
                "/v1/chat/completions",
                json=payload(EXPENSIVE, 1024),
                headers=headers(f"attacker-{index}", "attacker"),
            )

    async def legit() -> None:
        await asyncio.sleep(0.2)
        for _ in range(20):
            response = await client.post(
                "/v1/chat/completions",
                json=payload(CHEAP, 32),
                headers=headers("legit-1", "legit"),
            )
            legit_results.append(response.status_code)
            await asyncio.sleep(0.05)

    await asyncio.gather(
        *[attacker(i) for i in range(8)], legit(), return_exceptions=True
    )

    success = sum(1 for code in legit_results if code == 200) / len(legit_results)
    assert success >= 0.9, f"legitimate success rate {success:.2f}"

    records = read_log(app)
    reserved = [r for r in records if r.get("slot_pool") == "reserved"]
    assert all(r["client_id"] == "legit-1" for r in reserved)


@pytest.mark.asyncio
async def test_health_is_never_shed(tuned_config, gateway):
    config = tuned_config(**{"backpressure.global_max_inflight": 1})
    client, _ = await gateway(config)
    background = asyncio.gather(
        *[
            client.post("/v1/chat/completions", json=payload(EXPENSIVE, 256), headers=headers("f"))
            for _ in range(6)
        ],
        return_exceptions=True,
    )
    await asyncio.sleep(0.05)
    response = await client.get("/health")
    assert response.status_code == 200
    await background


@pytest.mark.asyncio
async def test_upstream_failure_does_not_leak_an_admission_slot(tuned_config, gateway):
    """A leaked slot would shrink capacity permanently under the exact load
    that caused the failure."""

    def explode(request):
        raise httpx.ConnectError("upstream down")

    config = tuned_config(**{"backpressure.global_max_inflight": 2})
    app_client, app = await gateway(config)
    app.state.upstream._client = httpx.AsyncClient(
        transport=httpx.MockTransport(explode), base_url="http://upstream"
    )

    for _ in range(5):
        response = await app_client.post(
            "/v1/chat/completions", json=payload(CHEAP), headers=headers("c")
        )
        assert response.status_code == 502
    assert app.state.controller.inflight == 0


@pytest.mark.asyncio
async def test_request_id_is_echoed_for_the_analysis_join(tuned_config, gateway):
    client, app = await gateway(tuned_config(mode="off"))
    await client.post(
        "/v1/chat/completions",
        json=payload(CHEAP),
        headers=headers("c1", request_id="req-abc"),
    )
    assert read_log(app)[0]["request_id"] == "req-abc"


@pytest.mark.asyncio
async def test_gateway_exposes_its_own_metrics_and_state(tuned_config, gateway):
    client, _ = await gateway(tuned_config())
    await client.post("/v1/chat/completions", json=payload(CHEAP), headers=headers("c1"))

    metrics = await client.get("/metrics")
    assert metrics.status_code == 200
    body = metrics.text
    assert "gateway_requests_total" in body
    # The gateway must not shadow vLLM's own series.
    assert "vllm:" not in body

    state = await client.get("/gateway/state")
    assert state.status_code == 200
    assert state.json()["mode"] == "ratelimit_queue"


@pytest.mark.asyncio
async def test_state_samples_record_upstream_staleness(tuned_config, gateway):
    client, app = await gateway(tuned_config())
    await client.post("/v1/chat/completions", json=payload(CHEAP), headers=headers("c1"))
    await asyncio.sleep(0.2)
    app.state.log.samples.flush()
    with open(app.state.log.samples.path) as handle:
        samples = [json.loads(line) for line in handle if line.strip()]
    assert samples
    # Staleness is recorded alongside the values, so a run where /metrics went
    # dark is visible in the data rather than looking like an idle backend.
    assert "upstream_stale" in samples[0]
    assert "metrics_failures" in samples[0]


@pytest.mark.asyncio
async def test_baseline_run_still_records_upstream_queue_depth(tuned_config, gateway):
    """The mitigation-off baseline needs its own queue-depth series, or the
    mitigated run's chart has nothing to be compared against."""
    client, app = await gateway(tuned_config(mode="off"))
    await client.post("/v1/chat/completions", json=payload(CHEAP), headers=headers("c1"))
    await asyncio.sleep(0.3)
    app.state.log.samples.flush()
    with open(app.state.log.samples.path) as handle:
        samples = [json.loads(line) for line in handle if line.strip()]
    fresh = [s for s in samples if not s["upstream_stale"]]
    assert fresh, "no upstream metrics were sampled in off mode"
    assert fresh[-1]["upstream_running"] is not None


@pytest.mark.asyncio
async def test_context_length_errors_pass_through_as_400(tuned_config, gateway):
    """A 400 from the backend is a client error, not mitigation and not
    overload. It has to stay distinguishable in the logs."""
    config = tuned_config(mode="off")
    client, _ = await gateway(
        config, upstream_app=fake_vllm_app(max_num_seqs=4, decode_s_per_token=0.001, max_model_len=256)
    )
    response = await client.post(
        "/v1/chat/completions", json=payload(EXPENSIVE, 128), headers=headers("c1")
    )
    assert response.status_code == 400
    assert "maximum context length" in response.text
