"""Gateway configuration.

Four named modes exist so that the experiment ablation is a single file swap:

    off              pure passthrough proxy, logging only
    ratelimit        per-client request + cost buckets
    ratelimit_queue  adds queue-pressure shedding and fair-share
    full             adds anomaly-adaptive tightening

``off`` still traverses the gateway. The mitigation-off baseline has to pay the
same proxy hop as the mitigation-on run or the latency comparison is measuring
the hop instead of the mitigation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any

import yaml

MODES = ("off", "ratelimit", "ratelimit_queue", "full")


@dataclass
class UpstreamConfig:
    base_url: str = "http://127.0.0.1:8000"
    connect_timeout_s: float = 2.0
    # Read timeout scales with the requested generation length; a flat timeout
    # makes the gateway manufacture failures on legitimately expensive prompts.
    read_timeout_base_s: float = 20.0
    read_timeout_per_token_s: float = 0.25
    read_timeout_max_s: float = 600.0
    max_connections: int = 256


@dataclass
class RateLimitConfig:
    enabled: bool = True
    # Request-rate bucket: catches spike and flood.
    requests_per_sec: float = 5.0
    requests_burst: float = 10.0
    # Cost bucket, charged prompt_tokens + max_tokens: catches low-and-slow,
    # which by construction stays under any request-rate limit.
    tokens_per_sec: float = 2000.0
    tokens_burst: float = 8000.0
    # Per-client in-flight cap. Bounds slot occupancy independently of arrival
    # rate, which is the actual scarce resource on a batching inference server.
    client_max_inflight: int = 4
    # Hard ceiling applied to max_tokens before cost accounting.
    max_tokens_ceiling: int = 1024
    default_max_tokens: int = 128
    client_ttl_s: float = 300.0


@dataclass
class BackpressureConfig:
    enabled: bool = True
    # Local admission width. Primary signal: instant, always correct, and
    # unlike upstream /metrics it cannot go stale under load.
    global_max_inflight: int = 48
    # Bounded wait for an admission slot. Waiting longer than this is worse
    # than a fast 503 because the client has usually given up already.
    queue_wait_ms: int = 200
    # Slots only cheap requests may use. This is what keeps legitimate
    # traffic flowing when per-client limits dilute under identity rotation,
    # because it partitions capacity by request cost rather than by identity.
    reserved_cheap_slots: int = 4
    cheap_cost_threshold: int = 512
    # Upstream signals, polled. High/low pairs give hysteresis so the
    # controller does not flap between shedding and admitting every tick.
    upstream_waiting_high: float = 24.0
    upstream_waiting_low: float = 8.0
    occupancy_high: float = 0.9
    occupancy_low: float = 0.6
    upstream_max_num_seqs: int = 32
    latency_ewma_high_ms: float = 20000.0
    latency_ewma_low_ms: float = 8000.0
    latency_ewma_alpha: float = 0.2
    metrics_poll_interval_ms: int = 250
    metrics_timeout_ms: int = 500
    # Past this age the upstream snapshot is ignored and local signals decide.
    # Reading absent metrics as "queue empty" would disable shedding exactly
    # when the endpoint is too loaded to answer.
    metrics_stale_after_ms: int = 3000


@dataclass
class FairShareConfig:
    enabled: bool = True
    window_s: float = 10.0
    # A client may hold this multiple of an equal share of recent cost before
    # becoming a shedding candidate under pressure.
    share_multiplier: float = 1.25
    min_active_clients: int = 2
    # Never shed a client below this cost share, however few clients there are.
    grace_cost_share: float = 0.15
    # Ceiling on the threshold, so that with only two active clients a
    # dominant one is still sheddable (1.25 * 1/2 would otherwise exceed 1.0).
    max_share_threshold: float = 0.5


@dataclass
class AnomalyConfig:
    enabled: bool = False
    model_path: str = "models/vae.npz"
    window_s: float = 10.0
    score_interval_ms: int = 200
    # Reconstruction error in baseline standard deviations. Calibrate against
    # the holdout percentiles printed by gateway.anomaly.train; a threshold
    # under the baseline p99 throttles normal users.
    suspect_threshold: float = 3.0
    hostile_threshold: float = 6.0
    suspect_rate_multiplier: float = 0.5
    hostile_rate_multiplier: float = 0.15
    # Tiers decay rather than latch, so a client that bursts once recovers.
    tier_decay_s: float = 30.0


@dataclass
class LoggingConfig:
    run_id: str = "dev"
    dir: str = "runs"
    decisions_filename: str = "gateway.jsonl"
    samples_filename: str = "gateway_samples.jsonl"
    sample_interval_ms: int = 250
    flush_interval_ms: int = 500
    # Buffered writes: per-request fsync on a 2-vCPU box costs more than the
    # admission logic it is recording.
    buffer_lines: int = 256
    # Emit the per-request feature vector. Needed on baseline runs to train
    # the VAE; adds a dozen floats per line otherwise.
    log_features: bool = False


@dataclass
class TokenizerConfig:
    # Exact counts make the cost bucket honest. Falls back to chars/4 when
    # transformers is unavailable, which is fine for relative accounting.
    name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    use_transformers: bool = True
    chars_per_token: float = 4.0
    per_message_overhead: int = 4


@dataclass
class GatewayConfig:
    mode: str = "ratelimit_queue"
    host: str = "127.0.0.1"
    port: int = 8080
    client_id_header: str = "X-Client-ID"
    request_id_header: str = "X-Request-ID"
    # Logged for Person 2's join, never read by admission. Deciding on a
    # ground-truth attacker label would make the rejection rate meaningless.
    label_header: str = "X-Traffic-Label"
    max_body_bytes: int = 1_000_000
    upstream: UpstreamConfig = field(default_factory=UpstreamConfig)
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)
    backpressure: BackpressureConfig = field(default_factory=BackpressureConfig)
    fairshare: FairShareConfig = field(default_factory=FairShareConfig)
    anomaly: AnomalyConfig = field(default_factory=AnomalyConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        # The mode is the single switch; sub-flags follow from it so a config
        # file cannot describe a mode that disagrees with its own components.
        if self.mode == "off":
            self.rate_limit.enabled = False
            self.backpressure.enabled = False
            self.fairshare.enabled = False
            self.anomaly.enabled = False
        elif self.mode == "ratelimit":
            self.rate_limit.enabled = True
            self.backpressure.enabled = False
            self.fairshare.enabled = False
            self.anomaly.enabled = False
        elif self.mode == "ratelimit_queue":
            self.rate_limit.enabled = True
            self.backpressure.enabled = True
            self.anomaly.enabled = False
        elif self.mode == "full":
            self.rate_limit.enabled = True
            self.backpressure.enabled = True
            self.anomaly.enabled = True

    def as_dict(self) -> dict[str, Any]:
        return _to_dict(self)


def _to_dict(obj: Any) -> Any:
    if is_dataclass(obj):
        return {f.name: _to_dict(getattr(obj, f.name)) for f in fields(obj)}
    return obj


def _merge(target: Any, values: dict[str, Any], path: str = "") -> None:
    known = {f.name: f for f in fields(target)}
    for key, value in values.items():
        if key not in known:
            raise ValueError(f"unknown config key: {path}{key}")
        current = getattr(target, key)
        if is_dataclass(current) and isinstance(value, dict):
            _merge(current, value, path=f"{path}{key}.")
        else:
            setattr(target, key, value)


def load_config(path: str | None = None, **overrides: Any) -> GatewayConfig:
    """Build a config from an optional YAML file, then env vars, then kwargs."""
    raw: dict[str, Any] = {}
    if path:
        with open(path) as handle:
            raw = yaml.safe_load(handle) or {}

    config = GatewayConfig()
    # Mode first: it resets sub-flags, so it must not clobber later overrides.
    if "mode" in raw:
        mode = raw.pop("mode")
        # An unquoted `mode: off` is the boolean false under YAML 1.1, which
        # is the one config typo everybody makes exactly once.
        config.mode = "off" if mode is False else mode
    _merge(config, raw)

    if env_upstream := os.environ.get("GATEWAY_UPSTREAM"):
        config.upstream.base_url = env_upstream
    if env_run := os.environ.get("GATEWAY_RUN_ID"):
        config.logging.run_id = env_run
    if env_mode := os.environ.get("GATEWAY_MODE"):
        config.mode = env_mode

    for key, value in overrides.items():
        if value is None:
            continue
        if "." in key:
            head, tail = key.split(".", 1)
            _merge(getattr(config, head), {tail: value})
        else:
            setattr(config, key, value)

    config.__post_init__()
    return config
