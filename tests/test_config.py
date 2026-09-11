import pytest

from gateway.config import GatewayConfig, load_config


def test_mode_off_disables_every_mechanism():
    config = GatewayConfig(mode="off")
    assert not config.rate_limit.enabled
    assert not config.backpressure.enabled
    assert not config.fairshare.enabled
    assert not config.anomaly.enabled


def test_mode_ratelimit_enables_only_the_buckets():
    config = GatewayConfig(mode="ratelimit")
    assert config.rate_limit.enabled
    assert not config.backpressure.enabled


def test_mode_full_enables_the_anomaly_layer():
    config = GatewayConfig(mode="full")
    assert config.anomaly.enabled
    assert config.backpressure.enabled


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        GatewayConfig(mode="paranoid")


def test_unquoted_yaml_off_is_not_read_as_false(tmp_path):
    """YAML 1.1 parses a bare `off` as the boolean false."""
    path = tmp_path / "c.yaml"
    path.write_text("mode: off\n")
    assert load_config(str(path)).mode == "off"


def test_unknown_config_key_is_rejected(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("rate_limit:\n  requests_per_secc: 5\n")
    with pytest.raises(ValueError, match="unknown config key: rate_limit.requests_per_secc"):
        load_config(str(path))


def test_shipped_configs_all_load():
    for name in ("off", "ratelimit", "ratelimit_queue", "full"):
        config = load_config(f"configs/{name}.yaml")
        assert config.mode in ("off", "ratelimit", "ratelimit_queue", "full")


def test_cli_overrides_win_over_file(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("mode: ratelimit\nupstream:\n  base_url: http://from-file:8000\n")
    config = load_config(str(path), **{"upstream.base_url": "http://override:9000"})
    assert config.upstream.base_url == "http://override:9000"


def test_none_overrides_are_ignored(tmp_path):
    """Unset argparse flags must not blank out file values."""
    path = tmp_path / "c.yaml"
    path.write_text("upstream:\n  base_url: http://from-file:8000\n")
    config = load_config(str(path), **{"upstream.base_url": None})
    assert config.upstream.base_url == "http://from-file:8000"


def test_env_var_sets_upstream(monkeypatch):
    monkeypatch.setenv("GATEWAY_UPSTREAM", "http://env:8000")
    assert load_config().upstream.base_url == "http://env:8000"


def test_tokens_estimate_defaults_to_a_charge_not_zero():
    from gateway.tokens import TokenCounter
    from gateway.config import TokenizerConfig

    counter = TokenCounter(TokenizerConfig(use_transformers=False))
    # An omitted max_tokens is not a cheap request: the server generates until
    # the context limit, so charging zero would be a free pass.
    estimate = counter.estimate({"messages": [{"role": "user", "content": "hi"}]}, 1024, 128)
    assert estimate.max_tokens == 128


def test_tokens_estimate_accounts_for_n():
    from gateway.tokens import TokenCounter
    from gateway.config import TokenizerConfig

    counter = TokenCounter(TokenizerConfig(use_transformers=False))
    estimate = counter.estimate(
        {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 64, "n": 4},
        1024,
        128,
    )
    assert estimate.max_tokens == 256
