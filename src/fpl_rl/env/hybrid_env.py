"""HybridFPLEnv: RL agent picks strategy, MILP optimizer picks players."""

from __future__ import annotations

import logging
from pathlib import Path

import gymnasium
import numpy as np

from fpl_rl.data.downloader import DEFAULT_DATA_DIR
from fpl_rl.data.loader import SeasonDataLoader
from fpl_rl.engine.engine import FPLGameEngine
from fpl_rl.engine.state import (
    ChipState,
    EngineAction,
    GameState,
    PlayerSlot,
    Squad,
)
from fpl_rl.env.hybrid_action_space import (
    HYBRID_MASK_LENGTH,
    HybridActionEncoder,
    create_hybrid_action_space,
)
from fpl_rl.env.observation_space import ObservationBuilder, create_observation_space
from fpl_rl.env.reward import RewardCalculator
from fpl_rl.optimizer.squad_selection import select_squad
from fpl_rl.optimizer.types import build_candidate_pool
from fpl_rl.utils.constants import (
    INITIAL_FREE_TRANSFERS,
    STARTING_BUDGET,
    TOTAL_GAMEWEEKS,
)

logger = logging.getLogger(__name__)


class HybridFPLEnv(gymnasium.Env):
    """Hybrid RL+MILP environment for Fantasy Premier League.

    The RL agent outputs high-level strategy (transfer count + chip).
    The MILP optimizer handles all player selection, lineup, captain,
    and bench decisions — guaranteed valid.

    Episode structure: 38 GW steps (no preseason; optimizer handles GW1 squad).
    """

    metadata = {"render_modes": ["human"], "name": "HybridFPLEnv-v0"}

    def __init__(
        self,
        season: str = "2023-24",
        data_dir: Path = DEFAULT_DATA_DIR,
        render_mode: str | None = None,
        predictor_model_dir: Path | None = None,
        prediction_integrator=None,
    ) -> None:
        super().__init__()
        self.season = season
        self.render_mode = render_mode

        # Load data
        self.loader = SeasonDataLoader(season, data_dir)

        # Engine
        self.engine = FPLGameEngine(self.loader)

        # Optionally load point prediction model
        integrator = prediction_integrator
        if integrator is None and predictor_model_dir is not None:
            from fpl_rl.prediction.integration import PredictionIntegrator
            pred_data_dir = data_dir.parent if data_dir.name == "raw" else data_dir
            integrator = PredictionIntegrator.from_model(
                predictor_model_dir, pred_data_dir, season,
            )

        # Env components
        self.action_encoder = HybridActionEncoder(self.loader, prediction_integrator=integrator)
        self.obs_builder = ObservationBuilder(self.loader, prediction_integrator=integrator)
        self.reward_calc = RewardCalculator(self.loader)

        # Spaces
        self.action_space = create_hybrid_action_space()
        self.observation_space = create_observation_space()

        # State
        self.state: GameState | None = None
        self._num_gws = min(self.loader.get_num_gameweeks(), TOTAL_GAMEWEEKS)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Reset environment: use MILP optimizer for initial squad selection.

        Adds Gaussian noise to predicted points for training diversity.
        """
        super().reset(seed=seed)

        # Build GW1 candidate pool with noisy predictions for diversity
        predicted_points = self.action_encoder._get_predicted_points(1)

        # Add noise for training diversity (different seeds → different squads)
        noisy_pp: dict[int, float] = {}
        for eid, xp in predicted_points.items():
            noise = float(self.np_random.normal(0, 0.3))
            noisy_pp[eid] = max(0.0, xp * (1.0 + noise))

        candidates = build_candidate_pool(self.loader, 1, noisy_pp)

        if not candidates:
            raise ValueError(f"No candidates for GW1 of {self.season}")

        # Use MILP optimizer for initial squad selection
        try:
            result = select_squad(candidates, budget=STARTING_BUDGET)
        except RuntimeError as e:
            raise ValueError(f"Squad optimizer failed for {self.season}: {e}") from e

        # Convert OptimizerResult to GameState
        self.state = self._result_to_game_state(result)

        # Build observation (pool from action encoder for obs builder)
        pool_eids = [c.element_id for c in candidates[:50]]
        obs = self.obs_builder.build(self.state, pool_eids)

        info: dict = {"season": self.season, "gw": 1}
        return obs, info

    def step(
        self, action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one gameweek with RL strategy + MILP player selection."""
        if self.state is None:
            raise RuntimeError("Must call reset() before step()")

        gw = self.state.current_gw
        state_before = self.state.copy()

        # Decode: RL strategy → MILP optimizer → EngineAction
        engine_action = self.action_encoder.decode(action, self.state)

        # Execute via engine (optimizer guarantees validity, but keep safety net)
        try:
            new_state, result = self.engine.step(self.state, engine_action)
        except ValueError as e:
            logger.warning("Engine error at GW%d despite optimizer: %s", gw, e)
            engine_action = EngineAction()
            new_state, result = self.engine.step(self.state, engine_action)

        self.state = new_state

        # Build observation for next GW
        pool_eids: list[int] = []
        if self.state.current_gw <= self._num_gws:
            pp = self.action_encoder._get_predicted_points(self.state.current_gw)
            next_candidates = build_candidate_pool(
                self.loader, self.state.current_gw, pp,
            )
            pool_eids = [c.element_id for c in next_candidates[:50]]

        obs = self.obs_builder.build(self.state, pool_eids)

        # Reward
        reward = self.reward_calc.calculate(result, state_before, self.state, gw)

        # Termination
        terminated = self.state.current_gw > self._num_gws
        truncated = False

        info = {
            "gw": gw,
            "preseason": False,
            "gw_points": result.gw_points,
            "net_points": result.net_points,
            "hit_cost": result.hit_cost,
            "total_points": self.state.total_points,
            "num_transfers": len(engine_action.transfers_out),
            "auto_subs": result.auto_subs,
            "captain_failover": result.captain_failover,
            "active_chip": engine_action.chip,
            "captain_points": result.captain_points,
            "bench_points": result.bench_points,
        }

        if self.render_mode == "human":
            self._render_gw(gw, result, engine_action)

        return obs, reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        """Return 12-element boolean mask for MaskablePPO."""
        if self.state is None:
            return np.ones(HYBRID_MASK_LENGTH, dtype=bool)
        return self.action_encoder.get_action_mask(self.state)

    def _result_to_game_state(self, result) -> GameState:
        """Convert an OptimizerResult into an initial GameState."""
        players: list[PlayerSlot] = []
        for eid in result.squad_element_ids:
            pos = self.loader.get_player_position(eid)
            price = self.loader.get_player_price(eid, 1)
            if pos is None or price <= 0:
                continue
            players.append(
                PlayerSlot(
                    element_id=eid, position=pos,
                    purchase_price=price, selling_price=price,
                )
            )

        eid_to_idx = {p.element_id: i for i, p in enumerate(players)}
        lineup = [eid_to_idx[eid] for eid in result.lineup_element_ids if eid in eid_to_idx]
        bench = [eid_to_idx[eid] for eid in result.bench_element_ids if eid in eid_to_idx]
        captain_idx = eid_to_idx.get(result.captain_id, 0)
        vice_idx = eid_to_idx.get(result.vice_captain_id, 1)

        squad = Squad(
            players=players, lineup=lineup, bench=bench,
            captain_idx=captain_idx, vice_captain_idx=vice_idx,
        )

        return GameState(
            squad=squad,
            bank=STARTING_BUDGET - result.total_cost,
            free_transfers=INITIAL_FREE_TRANSFERS,
            chips=ChipState(),
            current_gw=1,
            total_points=0,
        )

    def _render_gw(self, gw: int, result, action: EngineAction) -> None:
        """Print a human-readable summary of the GW."""
        print(f"\n--- GW{gw} ---")
        n_xfers = len(action.transfers_out)
        chip = action.chip or "none"
        print(f"Transfers: {n_xfers} | Chip: {chip}")
        print(f"Points: {result.gw_points} (hits: -{result.hit_cost})")
        print(f"Net: {result.net_points} | Total: {self.state.total_points}")
