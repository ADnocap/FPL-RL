"""Hybrid action space: RL outputs strategy, MILP optimizer handles player selection."""

from __future__ import annotations

import logging

import numpy as np
from gymnasium import spaces

from fpl_rl.data.loader import SeasonDataLoader
from fpl_rl.engine.chips import validate_chip
from fpl_rl.engine.state import EngineAction, GameState
from fpl_rl.optimizer.transfer_optimizer import optimize_transfers
from fpl_rl.optimizer.types import (
    OptimizerResult,
    build_candidate_pool,
    to_engine_action,
)

logger = logging.getLogger(__name__)

# Hybrid action space: just 2 dimensions
HYBRID_TRANSFER_DIM = 6  # 0-5 transfers (upper bound)
HYBRID_CHIP_DIM = 6  # 0=none, 1=WC, 2=FH, 3=BB, 4=TC, 5=reserved
HYBRID_ACTION_DIMS = [HYBRID_TRANSFER_DIM, HYBRID_CHIP_DIM]
HYBRID_MASK_LENGTH = sum(HYBRID_ACTION_DIMS)  # 12

CHIP_INDEX_MAP = {
    0: None,
    1: "wildcard",
    2: "free_hit",
    3: "bench_boost",
    4: "triple_captain",
    5: None,  # reserved
}


def create_hybrid_action_space() -> spaces.MultiDiscrete:
    """Create the 2-dimensional hybrid action space."""
    return spaces.MultiDiscrete(HYBRID_ACTION_DIMS)


class HybridActionEncoder:
    """Encodes RL strategy decisions into MILP-optimized EngineActions.

    The RL agent outputs (transfer_count, chip). This encoder calls the
    MILP optimizer to find the best specific player selections, lineup,
    captain, and bench order — guaranteed valid.
    """

    def __init__(
        self,
        loader: SeasonDataLoader,
        prediction_integrator: object | None = None,
    ) -> None:
        self.loader = loader
        self._integrator = prediction_integrator
        self.last_result: OptimizerResult | None = None

    def _get_predicted_points(self, gw: int) -> dict[int, float]:
        """Build predicted points dict for optimizer candidate pool."""
        if self._integrator is not None:
            # Use LightGBM predictions
            all_eids = self.loader.get_all_element_ids(gw)
            return {
                eid: self._integrator.get_predicted_points(eid, gw)
                for eid in all_eids
            }
        # Fallback: use rolling form as a proxy for predicted points
        all_eids = self.loader.get_all_element_ids(gw)
        return {
            eid: self.loader.get_player_form(eid, gw, window=5)
            for eid in all_eids
        }

    def decode(self, action: np.ndarray, state: GameState) -> EngineAction:
        """Convert a 2-dim hybrid action into an optimized EngineAction.

        Calls the MILP optimizer internally to find the best transfers,
        lineup, captain, and bench order.
        """
        transfer_count = int(action[0])
        chip_idx = int(action[1])

        # Map chip index to name
        chip = CHIP_INDEX_MAP.get(chip_idx)
        if chip is not None:
            error = validate_chip(state, chip)
            if error:
                chip = None

        # Determine max_transfers for optimizer
        # WC/FH: unlimited transfers (optimizer decides optimally)
        # Otherwise: upper-bound from RL agent's decision
        if chip in ("wildcard", "free_hit"):
            max_transfers = None
        else:
            max_transfers = transfer_count

        # Build candidate pool with predicted points
        gw = state.current_gw
        predicted_points = self._get_predicted_points(gw)
        candidates = build_candidate_pool(self.loader, gw, predicted_points)

        if not candidates:
            logger.warning("No candidates for GW%d, returning no-op", gw)
            self.last_result = None
            return EngineAction()

        # Call MILP optimizer
        try:
            result = optimize_transfers(
                state,
                candidates,
                chip=chip,
                max_transfers=max_transfers,
            )
            self.last_result = result
            return to_engine_action(result)
        except RuntimeError as e:
            logger.warning("Optimizer failed at GW%d: %s — returning no-op", gw, e)
            self.last_result = None
            return EngineAction()

    def get_action_mask(
        self, state: GameState, **kwargs,
    ) -> np.ndarray:
        """Build a 12-element boolean mask.

        - transfer_count (0-5): always all valid (optimizer handles feasibility)
        - chip (0-5): mask unavailable chips
        """
        mask = np.ones(HYBRID_MASK_LENGTH, dtype=bool)

        # Chip dimension (indices 6-11)
        chip_offset = HYBRID_TRANSFER_DIM
        for i in range(HYBRID_CHIP_DIM):
            chip_name = CHIP_INDEX_MAP.get(i)
            if chip_name is None:
                if i == 5:  # reserved
                    mask[chip_offset + i] = False
            else:
                if not state.chips.is_available(chip_name, state.current_gw):
                    mask[chip_offset + i] = False

        return mask
