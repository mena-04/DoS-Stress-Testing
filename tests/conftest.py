import httpx
import pytest
import pytest_asyncio

from gateway.app import create_app
from gateway.config import GatewayConfig
from gateway.fake_vllm import build_args
from gateway.fake_vllm import create_app as create_fake_vllm


def fake_vllm_app(**overrides):
    """Fast-running vLLM stand-in: 2 ms/token instead of 35 ms/token."""
    argv = []
    for key, value in overrides.items():
        flag = "--" + key.replace("_", "-")
        if value is True:
            argv.append(flag)
        elif value is not False:
            argv += [flag, str(value)]
    args = build_args(argv)
    return create_fake_vllm(args)


@pytest.fixture
def tuned_config(tmp_path):
    """Scaled-down thresholds so tests run in seconds, not minutes."""

    def build(**overrides) -> GatewayConfig:
        config = GatewayConfig(mode=overrides.pop("mode", "ratelimit_queue"))
        config.tokenizer.use_transformers = False  # deterministic, no network
        config.logging.dir = str(tmp_path)
        config.logging.buffer_lines = 1
        config.logging.flush_interval_ms = 50
        config.logging.sample_interval_ms = 50
        config.upstream.base_url = "http://upstream"
        config.upstream.read_timeout_base_s = 5.0
        config.upstream.read_timeout_per_token_s = 0.02
        config.backpressure.metrics_poll_interval_ms = 50
        for key, value in overrides.items():
            section, _, field = key.partition(".")
            if field:
                setattr(getattr(config, section), field, value)
            else:
                setattr(config, section, value)
        config.__post_init__()
        return config

    return build


@pytest_asyncio.fixture
async def gateway():
    """Yields a factory producing an httpx client bound to a live gateway."""
    stack = []

    async def build(config, upstream_app=None):
        upstream_app = upstream_app or fake_vllm_app(
            max_num_seqs=4, decode_s_per_token=0.002
        )
        app = create_app(config, upstream_transport=httpx.ASGITransport(app=upstream_app))
        context = app.router.lifespan_context(app)
        await context.__aenter__()
        client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://gateway",
            timeout=30.0,
        )
        stack.append((context, client, app))
        return client, app

    yield build

    for context, client, _ in reversed(stack):
        await client.aclose()
        await context.__aexit__(None, None, None)
