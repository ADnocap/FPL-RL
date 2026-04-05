"""FPLEnv: Gymnasium-compatible FPL environment."""

from __future__ import annotations

import logging
from pathlib import Path

import gymnasium
import numpy as np

from fpl_rl.data.downloader import DEFAULT_DATA_DIR
from fpl_rl.data.loader import SeasonDataLoader
from fpl_rl.engine.constraints import is_valid_formation, is_valid_squad_composition
from fpl_rl.engine.engine import FPLGameEngine
from fpl_rl.engine.state import (
    ChipState,
    EngineAction,
    GameState,
    PlayerSlot,
    Squad,
)
from fpl_rl.env.action_space import ActionEncoder, create_action_space
from fpl_rl.env.observation_space import ObservationBuilder, create_observation_space
from fpl_rl.env.reward import RewardCalculator
from fpl_rl.utils.constants import (
    INITIAL_FREE_TRANSFERS,
    POSITION_LIMITS,
    SQUAD_SIZE,
    STARTING_BUDGET,
    TOTAL_GAMEWEEKS,
    Position,
)

logger = logging.getLogger(__name__)

# Number of pre-season steps before GW1 where the model can reshape
# the random initial squad via transfers (no points scored, no GW advance).
PRESEASON_STEPS = 2


class FPLEnv(gymnasium.Env):
    """Gymnasium environment for Fantasy Premier League.

    Replays historical seasons using data from vaastav/Fantasy-Premier-League.
    Supports MaskablePPO via action_masks() method.

    Episode structure: PRESEASON_STEPS pre-season steps (squad building) +
    num_gws real gameweek steps. Total episode length = PRESEASON_STEPS + num_gws.
    """

    metadata = {"render_modes": ["human"], "name": "FPLEnv-v0"}

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

        # Engine (no Gym dependency)
        self.engine = FPLGameEngine(self.loader)

        # Optionally load point prediction model
        integrator = prediction_integrator
        if integrator is None and predictor_model_dir is not None:
            from fpl_rl.prediction.integration import PredictionIntegrator
            # Prediction pipeline expects the parent data dir (with id_maps/,
            # understat/, etc.), not the vaastav-only data/raw/ subdirectory.
            pred_data_dir = data_dir.parent if data_dir.name == "raw" else data_dir
            integrator = PredictionIntegrator.from_model(
                predictor_model_dir, pred_data_dir, season,
            )

        # Env components
        self.action_encoder = ActionEncoder(self.loader)
        self.obs_builder = ObservationBuilder(self.loader, prediction_integrator=integrator)
        self.reward_calc = RewardCalculator(self.loader)

        # Spaces
        self.action_space = create_action_space()
        self.observation_space = create_observation_space()

        # State (initialized in reset)
        self.state: GameState | None = None
        self._num_gws = min(self.loader.get_num_gameweeks(), TOTAL_GAMEWEEKS)
        self._preseason_steps_remaining = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Reset environment to start of season with a random valid squad."""
        super().reset(seed=seed)

        # Build initial squad
        squad = self._build_initial_squad()
        budget_used = sum(p.purchase_price for p in squad.players)

        self.state = GameState(
            squad=squad,
            bank=STARTING_BUDGET - budget_used,
            # GW1 = initial squad selection. In real FPL you pick freely
            # before the deadline. We give max FTs so the model can reshape
            # the random initial squad at no cost across pre-season + GW1.
            free_transfers=SQUAD_SIZE,
            chips=ChipState(),
            current_gw=1,
            total_points=0,
            active_chip=None,
            free_hit_stash=None,
        )

        # Pre-season: model gets extra steps to rebuild squad before GW1
        self._preseason_steps_remaining = PRESEASON_STEPS

        # Build candidate pool for GW1
        self.action_encoder.build_candidate_pool(self.state, 1)

        obs = self.obs_builder.build(
            self.state, self.action_encoder._candidate_pool
        )
        info: dict = {"season": self.season, "gw": 0, "preseason": True}
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one step: pre-season (squad building) or gameweek.

        Pre-season steps apply transfers/lineup changes but score no points
        and don't advance the gameweek. After PRESEASON_STEPS pre-season
        steps, the real GW1 step begins.

        Returns (obs, reward, terminated, truncated, info).
        """
        if self.state is None:
            raise RuntimeError("Must call reset() before step()")

        is_preseason = self._preseason_steps_remaining > 0

        gw = self.state.current_gw
        state_before = self.state.copy()

        # Decode action
        engine_action = self.action_encoder.decode(action, self.state)

        # Execute via engine
        try:
            new_state, result = self.engine.step(
                self.state, engine_action, preseason=is_preseason,
            )
        except ValueError as e:
            logger.debug("Invalid action at GW%d: %s — treating as no-op", gw, e)
            engine_action = EngineAction()  # replace so info dict reflects reality
            new_state, result = self.engine.step(
                self.state, engine_action, preseason=is_preseason,
            )

        self.state = new_state

        # --- Pre-season path: no scoring, no GW advance ---
        if is_preseason:
            # Consume free transfers used (engine doesn't bank during preseason)
            num_transfers = len(engine_action.transfers_out)
            self.state.free_transfers = max(
                0, self.state.free_transfers - num_transfers
            )
            self._preseason_steps_remaining -= 1

            # Rebuild candidate pool (squad may have changed via transfers)
            self.action_encoder.build_candidate_pool(self.state, 1)

            obs = self.obs_builder.build(
                self.state, self.action_encoder._candidate_pool
            )

            info = {
                "gw": 0,
                "preseason": True,
                "preseason_steps_remaining": self._preseason_steps_remaining,
                "gw_points": 0,
                "net_points": 0,
                "hit_cost": 0,
                "total_points": 0,
                "auto_subs": [],
                "captain_failover": False,
                "active_chip": None,
                "captain_points": 0,
                "bench_points": 0,
            }
            return obs, 0.0, False, False, info

        # --- Regular GW step ---

        # GW1 is initial squad selection — everyone starts GW2 with 1 FT
        if gw == 1:
            self.state.free_transfers = INITIAL_FREE_TRANSFERS

        # Rebuild candidate pool for next GW
        if self.state.current_gw <= self._num_gws:
            self.action_encoder.build_candidate_pool(
                self.state, self.state.current_gw
            )

        # Build observation
        obs = self.obs_builder.build(
            self.state, self.action_encoder._candidate_pool
        )

        # Calculate reward
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
            "auto_subs": result.auto_subs,
            "captain_failover": result.captain_failover,
            "active_chip": engine_action.chip,  # chip requested this step
            "num_transfers": len(engine_action.transfers_out),
            "captain_points": result.captain_points,
            "bench_points": result.bench_points,
        }

        if self.render_mode == "human":
            self._render_gw(gw, result)

        return obs, reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        """Return flat boolean mask for MaskablePPO."""
        if self.state is None:
            return np.ones(sum(self.action_space.nvec), dtype=bool)
        preseason = self._preseason_steps_remaining > 0
        return self.action_encoder.get_action_mask(
            self.state, preseason=preseason,
        )

    def _build_initial_squad(self) -> Squad:
        """Build a random valid initial squad from GW1 data.

        Selects players semi-randomly while respecting position limits,
        budget, and club limits. Randomization provides training diversity.
        """
        gw1_data = self.loader.get_gameweek_data(1)
        if gw1_data.empty:
            raise ValueError(f"No GW1 data for season {self.season}")

        # Deduplicate
        available = gw1_data.drop_duplicates(subset="element")

        # Group by position
        by_position: dict[Position, list[dict]] = {p: [] for p in Position}
        for _, row in available.iterrows():
            eid = int(row["element"])
            pos = self.loader.get_player_position(eid)
            if pos is not None:
                by_position[pos].append({
                    "element_id": eid,
                    "price": int(row.get("value", 50)),
                    "team": self.loader.get_player_team(eid),
                })

        # Precompute minimum cost per position for budget headroom checks
        min_cost: dict[Position, int] = {}
        for pos in Position:
            prices = [p["price"] for p in by_position[pos]]
            min_cost[pos] = min(prices) if prices else 0

        rng = self.np_random
        max_attempts = 10
        squad_players: list[PlayerSlot] | None = None

        for attempt in range(max_attempts):
            # Shuffle each position group for randomness
            for pos in by_position:
                rng.shuffle(by_position[pos])

            # Greedy selection respecting constraints
            candidates: list[PlayerSlot] = []
            team_counts: dict[int, int] = {}
            budget_remaining = STARTING_BUDGET
            remaining_slots: dict[Position, int] = dict(POSITION_LIMITS)

            for pos, needed in POSITION_LIMITS.items():
                selected = 0
                for p in by_position[pos]:
                    if selected >= needed:
                        break
                    # Check budget
                    if p["price"] > budget_remaining:
                        continue
                    # Check club limit
                    team_id = p.get("team")
                    if team_id is not None:
                        current = team_counts.get(team_id, 0)
                        if current >= 3:
                            continue

                    # Budget headroom: ensure remaining positions can still be filled
                    future_budget = budget_remaining - p["price"]
                    future_slots = dict(remaining_slots)
                    future_slots[pos] = needed - selected - 1
                    future_min = sum(
                        min_cost[fp] * fc for fp, fc in future_slots.items() if fc > 0
                    )
                    if future_budget < future_min:
                        continue

                    if team_id is not None:
                        team_counts[team_id] = team_counts.get(team_id, 0) + 1

                    candidates.append(
                        PlayerSlot(
                            element_id=p["element_id"],
                            position=pos,
                            purchase_price=p["price"],
                            selling_price=p["price"],
                        )
                    )
                    budget_remaining -= p["price"]
                    selected += 1

                remaining_slots[pos] = needed - selected

                if selected < needed:
                    # Fill with cheapest available if needed
                    by_price = sorted(by_position[pos], key=lambda x: x["price"])
                    for p in by_price:
                        if selected >= needed:
                            break
                        if any(sp.element_id == p["element_id"] for sp in candidates):
                            continue
                        if p["price"] > budget_remaining:
                            continue
                        team_id = p.get("team")
                        if team_id is not None:
                            current = team_counts.get(team_id, 0)
                            if current >= 3:
                                continue

                        # Budget headroom check for fallback path too
                        future_budget = budget_remaining - p["price"]
                        future_slots = dict(remaining_slots)
                        future_slots[pos] = needed - selected - 1
                        future_min = sum(
                            min_cost[fp] * fc
                            for fp, fc in future_slots.items()
                            if fc > 0
                        )
                        if future_budget < future_min:
                            continue

                        if team_id is not None:
                            team_counts[team_id] = team_counts.get(team_id, 0) + 1

                        candidates.append(
                            PlayerSlot(
                                element_id=p["element_id"],
                                position=pos,
                                purchase_price=p["price"],
                                selling_price=p["price"],
                            )
                        )
                        budget_remaining -= p["price"]
                        selected += 1

                    remaining_slots[pos] = needed - selected

            if len(candidates) == 15:
                squad_players = candidates
                break
            # else retry with a new shuffle

        if squad_players is None or len(squad_players) != 15:
            raise ValueError(
                f"Could only select {len(squad_players) if squad_players else 0}/15 "
                f"players for initial squad after {max_attempts} attempts"
            )

        # Assign lineup (first valid formation) and bench
        lineup_indices: list[int] = []
        bench_indices: list[int] = []

        # Start with 1 GK, then fill outfield by position
        gk_indices = [i for i, p in enumerate(squad_players) if p.position == Position.GK]
        lineup_indices.append(gk_indices[0])
        bench_indices.append(gk_indices[1])

        # Use 4-4-2 as default formation
        target = {Position.DEF: 4, Position.MID: 4, Position.FWD: 2}
        for pos, count in target.items():
            pos_indices = [i for i, p in enumerate(squad_players) if p.position == pos]
            lineup_indices.extend(pos_indices[:count])
            bench_indices.extend(pos_indices[count:])

        return Squad(
            players=squad_players,
            lineup=lineup_indices,
            bench=bench_indices,
            captain_idx=lineup_indices[1],  # First DEF as captain (arbitrary)
            vice_captain_idx=lineup_indices[5],  # First MID as vice
        )

    def _render_gw(self, gw: int, result) -> None:
        """Print a human-readable summary of the GW."""
        print(f"\n--- GW{gw} ---")
        print(f"Points: {result.gw_points} (hits: -{result.hit_cost})")
        print(f"Net: {result.net_points} | Total: {self.state.total_points}")
        if result.auto_subs:
            for out_id, in_id in result.auto_subs:
                print(f"  Auto-sub: {out_id} -> {in_id}")
        if result.captain_failover:
            print("  Captain failover to vice-captain")
