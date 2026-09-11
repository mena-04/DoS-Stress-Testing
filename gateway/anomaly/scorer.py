"""Anomaly scoring loop and tier assignment.

Scores are computed in a background task and cached per client. The request
path only reads a dict entry, because a mitigation layer that adds tail
latency under load defeats its own purpose.

The whole layer is optional and fails open: a missing or unreadable model
leaves every client in the ``normal`` tier, and the static limits carry the
defense on their own.
"""

from __future__ import annotations

import asyncio
import os
import time

from ..config import AnomalyConfig
from ..features import FEATURE_NAMES, FeatureStore
from .vae import VAEWeights, anomaly_score

TIERS = ("normal", "suspect", "hostile")


class NullScorer:
    """Used whenever anomaly detection is disabled or unavailable."""

    available = False

    def tier(self, client_id: str) -> tuple[str, float, float | None]:
        return "normal", 1.0, None

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None


class VAEScorer:
    available = True

    def __init__(
        self,
        config: AnomalyConfig,
        features: FeatureStore,
        weights: VAEWeights,
        clock=time.monotonic,
    ) -> None:
        self._config = config
        self._features = features
        self._weights = weights
        self._clock = clock
        # client_id -> (tier, score, tier_expires_at)
        self._cache: dict[str, tuple[str, float, float]] = {}
        self._task: asyncio.Task | None = None
        self.scored_clients = 0

    def tier(self, client_id: str) -> tuple[str, float, float | None]:
        entry = self._cache.get(client_id)
        if entry is None:
            return "normal", 1.0, None
        name, score, _ = entry
        return name, self._multiplier(name), score

    def _multiplier(self, tier: str) -> float:
        if tier == "hostile":
            return self._config.hostile_rate_multiplier
        if tier == "suspect":
            return self._config.suspect_rate_multiplier
        return 1.0

    def _tier_for(self, score: float) -> str:
        if score >= self._config.hostile_threshold:
            return "hostile"
        if score >= self._config.suspect_threshold:
            return "suspect"
        return "normal"

    def score_once(self) -> dict[str, tuple[str, float, float]]:
        now = self._clock()
        for client_id in self._features.active_clients(now):
            vector = self._features.vector(client_id, now)
            score = anomaly_score(vector, self._weights)
            observed = self._tier_for(score)
            previous = self._cache.get(client_id)
            if previous is not None:
                prev_tier, _, expires = previous
                # Tiers decay instead of latching: a client that bursts once
                # recovers, but cannot escape a penalty by going quiet for a
                # single scoring tick.
                if TIERS.index(observed) < TIERS.index(prev_tier) and now < expires:
                    self._cache[client_id] = (prev_tier, score, expires)
                    continue
            self._cache[client_id] = (
                observed,
                score,
                now + self._config.tier_decay_s if observed != "normal" else now,
            )
            self.scored_clients += 1
        self._evict(now)
        return dict(self._cache)

    def _evict(self, now: float) -> None:
        active = set(self._features.active_clients(now))
        for client_id in [k for k in self._cache if k not in active]:
            _, _, expires = self._cache[client_id]
            if now >= expires:
                del self._cache[client_id]

    async def start(self) -> None:
        interval = self._config.score_interval_ms / 1000.0

        async def loop() -> None:
            while True:
                try:
                    self.score_once()
                except Exception:
                    # Never let the scorer take the gateway down; static
                    # limits remain in force regardless.
                    pass
                await asyncio.sleep(interval)

        self._task = asyncio.create_task(loop(), name="anomaly-scorer")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None


def build_scorer(config: AnomalyConfig, features: FeatureStore, clock=time.monotonic):
    if not config.enabled:
        return NullScorer()
    if not os.path.exists(config.model_path):
        return NullScorer()
    try:
        weights = VAEWeights.load(config.model_path)
    except Exception:
        return NullScorer()
    if tuple(weights.feature_names) != FEATURE_NAMES:
        # A model trained on a different feature layout would score garbage.
        return NullScorer()
    return VAEScorer(config, features, weights, clock=clock)
