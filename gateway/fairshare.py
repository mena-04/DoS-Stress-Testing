"""Fair-share shedding.

Under pressure, uniform rejection hurts the quiet client as much as the noisy
one. This ranks active clients by their share of recent token cost and sheds
only those above an equal share, which is what lets legitimate traffic keep
flowing while a flooder absorbs the rejections.

Known limitation: shares are per observed client ID. An attacker rotating IDs
dilutes its own share below the threshold, and only the global admission cap
constrains it. That trade-off is deliberate and recorded rather than hidden.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import FairShareConfig
from .features import FeatureStore


@dataclass
class ShareVerdict:
    over_share: bool
    share: float
    threshold: float
    active_clients: int


class FairShareController:
    def __init__(self, config: FairShareConfig, features: FeatureStore) -> None:
        self._config = config
        self._features = features

    def evaluate(self, client_id: str) -> ShareVerdict:
        config = self._config
        costs, total = self._features.cost_shares()
        active = len(costs)
        if not config.enabled or active < config.min_active_clients or total <= 0:
            return ShareVerdict(False, 0.0, 1.0, active)

        share = costs.get(client_id, 0.0) / total
        equal_share = 1.0 / active
        threshold = max(
            config.grace_cost_share,
            min(config.max_share_threshold, config.share_multiplier * equal_share),
        )
        # Strict inequality: clients splitting capacity evenly all sit exactly
        # at the threshold and none of them should be singled out.
        return ShareVerdict(share > threshold, share, threshold, active)
