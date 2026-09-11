import numpy as np

from gateway.anomaly.scorer import NullScorer, VAEScorer, build_scorer
from gateway.anomaly.vae import anomaly_score, train
from gateway.config import AnomalyConfig
from gateway.features import FEATURE_NAMES, FeatureStore


def _normal_traffic(n=600, seed=0):
    """Baseline: modest rates, short prompts, small generations."""
    rng = np.random.default_rng(seed)
    data = np.zeros((n, len(FEATURE_NAMES)))
    data[:, 0] = rng.normal(1.0, 0.2, n)        # request_rate
    data[:, 1] = rng.normal(200.0, 40.0, n)     # cost_rate
    data[:, 2] = rng.normal(60.0, 10.0, n)      # mean_prompt_tokens
    data[:, 3] = rng.normal(80.0, 12.0, n)      # p95_prompt_tokens
    data[:, 4] = rng.normal(64.0, 8.0, n)       # mean_max_tokens
    data[:, 5] = rng.normal(1.1, 0.2, n)        # token_ratio
    data[:, 6] = rng.normal(1.0, 0.2, n)        # interarrival_mean
    data[:, 7] = rng.normal(0.3, 0.1, n)        # interarrival_stdev
    data[:, 8] = rng.normal(1.0, 0.4, n)        # concurrency
    data[:, 9] = rng.uniform(0.0, 0.1, n)       # repeat_ratio
    data[:, 10] = rng.uniform(0.0, 0.02, n)     # error_rate
    data[:, 11] = rng.normal(900.0, 120.0, n)   # mean_latency_ms
    return data


def _trained():
    return train(_normal_traffic(), FEATURE_NAMES, epochs=120, seed=0)


def test_baseline_scores_sit_near_zero():
    weights = _trained()
    scores = [anomaly_score(row, weights) for row in _normal_traffic(100, seed=99)]
    assert float(np.percentile(scores, 95)) < 6.0


def test_low_and_slow_profile_scores_above_baseline():
    """Low request rate, huge generations: invisible to a rate limiter,
    and structurally unlike anything in the baseline."""
    weights = _trained()
    baseline = float(np.percentile([anomaly_score(r, weights) for r in _normal_traffic(100, 7)], 95))

    attacker = _normal_traffic(1, seed=5)[0].copy()
    attacker[0] = 0.4      # request_rate below normal
    attacker[1] = 4000.0   # cost_rate far above normal
    attacker[4] = 1024.0   # mean_max_tokens
    attacker[5] = 12.0     # token_ratio
    attacker[9] = 1.0      # identical prompts every time
    assert anomaly_score(attacker, weights) > baseline


def test_flood_profile_scores_above_baseline():
    weights = _trained()
    baseline = float(np.percentile([anomaly_score(r, weights) for r in _normal_traffic(100, 7)], 95))
    attacker = _normal_traffic(1, seed=3)[0].copy()
    attacker[0] = 60.0
    attacker[1] = 20000.0
    attacker[8] = 40.0
    assert anomaly_score(attacker, weights) > baseline


def test_scoring_is_deterministic():
    """A sampled latent would score the same client differently on
    consecutive ticks, which is unacceptable in a rate limiter."""
    weights = _trained()
    row = _normal_traffic(1, seed=11)[0]
    assert anomaly_score(row, weights) == anomaly_score(row, weights)


def test_save_and_load_round_trip(tmp_path):
    from gateway.anomaly.vae import VAEWeights

    weights = _trained()
    path = tmp_path / "vae.npz"
    weights.save(str(path))
    loaded = VAEWeights.load(str(path))
    row = _normal_traffic(1, seed=21)[0]
    assert abs(anomaly_score(row, weights) - anomaly_score(row, loaded)) < 1e-9
    assert tuple(loaded.feature_names) == FEATURE_NAMES


def test_missing_model_fails_open():
    scorer = build_scorer(
        AnomalyConfig(enabled=True, model_path="/nonexistent/vae.npz"),
        FeatureStore(),
    )
    assert isinstance(scorer, NullScorer)
    assert scorer.tier("anyone") == ("normal", 1.0, None)


def test_feature_layout_mismatch_fails_open(tmp_path):
    weights = _trained()
    weights.feature_names = ("a", "b")
    path = tmp_path / "stale.npz"
    weights.save(str(path))
    scorer = build_scorer(
        AnomalyConfig(enabled=True, model_path=str(path)), FeatureStore()
    )
    assert isinstance(scorer, NullScorer)


def test_tier_decays_rather_than_latching(monkeypatch):
    """Decay and tiering are tested with the score driven directly.

    Whether a given traffic shape scores high is the subject of the tests
    above; this one is about what happens to a client's tier once it does.
    """
    clock = [100.0]
    scores = iter([9.0, 0.0, 0.0])
    monkeypatch.setattr(
        "gateway.anomaly.scorer.anomaly_score", lambda vector, weights: next(scores)
    )

    store = FeatureStore(window_s=10.0, clock=lambda: clock[0])
    config = AnomalyConfig(
        enabled=True, suspect_threshold=3.0, hostile_threshold=6.0, tier_decay_s=30.0
    )
    scorer = VAEScorer(config, store, _trained(), clock=lambda: clock[0])

    store.record_admitted("attacker", 1200, 1024, prompt_hash=1)
    scorer.score_once()
    assert scorer.tier("attacker")[0] == "hostile"
    assert scorer.tier("attacker")[1] == config.hostile_rate_multiplier

    # Scoring clean again, but still inside the decay window: the penalty
    # holds, so a client cannot escape by going quiet for a single tick.
    clock[0] = 110.0
    store.record_admitted("attacker", 40, 32, prompt_hash=2)
    scorer.score_once()
    assert scorer.tier("attacker")[0] == "hostile"

    # Past the decay window it is allowed to recover.
    clock[0] = 140.0
    store.record_admitted("attacker", 40, 32, prompt_hash=3)
    scorer.score_once()
    assert scorer.tier("attacker")[0] == "normal"
    assert scorer.tier("attacker")[1] == 1.0
