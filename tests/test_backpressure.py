import pytest

from gateway.backpressure import (
    PressureController,
    PressureLevel,
    UpstreamMetricsPoller,
    UpstreamSnapshot,
    parse_labelled_metric,
    parse_metric,
)
from gateway.config import BackpressureConfig

# Copied verbatim from a real vLLM 0.29.0 /metrics response on a T4, including
# the _by_reason series that a prefix match would wrongly pick up.
VLLM_METRICS = """
# HELP vllm:num_requests_running Number of requests currently running.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{engine="0",model_name="Qwen/Qwen2.5-0.5B-Instruct"} 37.0
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{engine="0",model_name="Qwen/Qwen2.5-0.5B-Instruct"} 12.0
vllm:num_requests_waiting_by_reason{engine="0",model_name="Qwen/Qwen2.5-0.5B-Instruct",reason="capacity"} 9.0
vllm:num_requests_waiting_by_reason{engine="0",model_name="Qwen/Qwen2.5-0.5B-Instruct",reason="deferred"} 3.0
vllm:request_success_total{engine="0",finished_reason="stop",model_name="Qwen/Qwen2.5-0.5B-Instruct"} 1.0
"""


def test_parse_metric_does_not_match_longer_metric_names():
    assert parse_metric(VLLM_METRICS, "vllm:num_requests_waiting") == 12.0
    assert parse_metric(VLLM_METRICS, "vllm:num_requests_running") == 37.0


def test_parse_metric_returns_none_when_absent():
    """None must stay distinct from 0.0.

    Collapsing a parse failure into zero is what made the first real overload
    run report an idle backend for its entire duration.
    """
    assert parse_metric(VLLM_METRICS, "vllm:nonexistent") is None


def test_parse_metric_tolerates_missing_labels():
    assert parse_metric("vllm:num_requests_running 4.0\n", "vllm:num_requests_running") == 4.0


def test_parse_metric_sums_label_sets():
    text = (
        'vllm:num_requests_running{engine="0"} 3.0\n'
        'vllm:num_requests_running{engine="1"} 4.0\n'
    )
    assert parse_metric(text, "vllm:num_requests_running") == 7.0


def test_parse_labelled_metric_selects_by_reason():
    value = parse_labelled_metric(
        VLLM_METRICS, "vllm:num_requests_waiting_by_reason", "reason", "capacity"
    )
    assert value == 9.0


def _controller(**overrides):
    config = BackpressureConfig(
        global_max_inflight=10,
        occupancy_high=0.9,
        occupancy_low=0.6,
        latency_ewma_high_ms=10_000,
        latency_ewma_low_ms=5_000,
        **overrides,
    )
    return PressureController(config, poller=None)


def test_local_occupancy_escalates_to_critical():
    controller = _controller()
    controller.inflight = 9
    assert controller.evaluate().level == PressureLevel.CRITICAL


def test_hysteresis_holds_critical_until_low_mark():
    controller = _controller()
    controller.inflight = 9
    assert controller.evaluate().level == PressureLevel.CRITICAL
    # Between the low and high marks the level must not drop back, or the
    # controller flaps between shedding and admitting on every tick.
    controller.inflight = 7
    assert controller.evaluate().level == PressureLevel.CRITICAL
    controller.inflight = 5
    assert controller.evaluate().level == PressureLevel.NORMAL


def test_stale_upstream_snapshot_does_not_read_as_idle():
    """The core fail-safe.

    A stale snapshot's signals are dropped, and local in-flight still reports
    pressure. If absent metrics were read as "queue empty", shedding would
    switch off exactly when /metrics is too loaded to answer.
    """
    clock = [1000.0]
    config = BackpressureConfig(global_max_inflight=10, metrics_stale_after_ms=1000)
    poller = UpstreamMetricsPoller("http://x", config, clock=lambda: clock[0])
    poller._snapshot = UpstreamSnapshot(ts=1000.0, running=0.0, waiting=0.0, ok=True)
    controller = PressureController(config, poller, clock=lambda: clock[0])

    controller.inflight = 9
    report = controller.evaluate()
    assert report.upstream_stale is False
    assert report.level == PressureLevel.CRITICAL

    clock[0] = 1010.0
    report = controller.evaluate()
    assert report.upstream_stale is True
    assert "upstream_waiting" not in report.signals
    assert report.level == PressureLevel.CRITICAL


def test_upstream_waiting_triggers_pressure_independently():
    config = BackpressureConfig(
        global_max_inflight=1000, upstream_waiting_high=20.0, upstream_waiting_low=5.0
    )
    clock = [1000.0]
    poller = UpstreamMetricsPoller("http://x", config, clock=lambda: clock[0])
    poller._snapshot = UpstreamSnapshot(ts=1000.0, running=1.0, waiting=25.0, ok=True)
    controller = PressureController(config, poller, clock=lambda: clock[0])
    report = controller.evaluate()
    assert report.level == PressureLevel.CRITICAL
    assert "upstream_waiting" in report.triggers


def test_latency_ewma_feeds_pressure():
    controller = _controller()
    for _ in range(40):
        controller.observe_latency(30_000)
    assert controller.evaluate().level == PressureLevel.CRITICAL


@pytest.mark.asyncio
async def test_poller_counts_failures_instead_of_swallowing_them():
    import httpx

    config = BackpressureConfig(metrics_timeout_ms=50)

    def handler(request):
        raise httpx.ConnectError("refused")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    poller = UpstreamMetricsPoller("http://upstream", config, client=client)
    await poller.poll_once()
    assert poller.failure_count == 1
    assert poller.snapshot.valid is False
    await client.aclose()


@pytest.mark.asyncio
async def test_poller_parses_live_response():
    import httpx

    config = BackpressureConfig()
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, text=VLLM_METRICS))
    )
    poller = UpstreamMetricsPoller("http://upstream", config, client=client)
    snapshot = await poller.poll_once()
    assert snapshot.running == 37.0
    assert snapshot.waiting == 12.0
    assert snapshot.waiting_capacity == 9.0
    await client.aclose()
