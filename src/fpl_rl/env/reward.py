"""Reward calculation for the FPL environment."""

from __future__ import annotations

from fpl_rl.data.loader import SeasonDataLoader
from fpl_rl.engine.state import GameState, StepResult


class RewardCalculator:
    """Computes reward for a gameweek step.

    reward = net_points + 0.1 * (net_points - gw_average) + 0.05 * team_value_change

    Components:
    - Primary: net gameweek points (already includes hit penalties)
    - Auxiliary 1: performance relative to GW average (beat the crowd)
    - Auxiliary 2: team value appreciation (smart transfers)
    """

    def __init__(
        self,
        loader: SeasonDataLoader,
        relative_weight: float = 0.1,
        value_weight: float = 0.05,
    ) -> None:
        self.loader = loader
        self.relative_weight = relative_weight
        self.value_weight = value_weight

    def calculate(
        self,
        result: StepResult,
        state_before: GameState,
        state_after: GameState,
        gw: int,
    ) -> float:
        """Calculate the reward for a single GW step."""
        # Primary: net points
        net_points = float(result.net_points)

        # Auxiliary 1: relative to GW average
        gw_avg = self.loader.get_gw_average_points(gw)
        relative = net_points - gw_avg

        # Auxiliary 2: team value change
        value_before = sum(p.selling_price for p in state_before.squad.players)
        value_after = sum(p.selling_price for p in state_after.squad.players)
        value_change = float(value_after - value_before) / 10.0  # in £0.1m units

        reward = (
            net_points
            + self.relative_weight * relative
            + self.value_weight * value_change
        )

        return reward
