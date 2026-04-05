"""Observation space builder for the FPL environment."""

from __future__ import annotations

import numpy as np
from gymnasium import spaces

from fpl_rl.data.loader import SeasonDataLoader
from fpl_rl.engine.state import GameState
from fpl_rl.utils.constants import (
    TOTAL_GAMEWEEKS,
    Position,
)

# Feature dimensions
PLAYER_FEATURES = 24  # features per squad player
POOL_FEATURES = 19  # features per candidate pool player
SQUAD_SIZE = 15
POOL_SIZE = 50
NUM_CHIPS = 8  # 4 chips x 2 halves
NUM_DGW_FLAGS = 20  # one per team
NUM_BGW_FLAGS = 20

SQUAD_BLOCK = SQUAD_SIZE * PLAYER_FEATURES  # 360
POOL_BLOCK = POOL_SIZE * POOL_FEATURES  # 950
GLOBAL_BLOCK = 5 + NUM_CHIPS + NUM_DGW_FLAGS + NUM_BGW_FLAGS  # 53

OBS_DIM = SQUAD_BLOCK + POOL_BLOCK + GLOBAL_BLOCK  # 1363


def create_observation_space() -> spaces.Box:
    """Create the flat Box observation space."""
    return spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(OBS_DIM,),
        dtype=np.float32,
    )


class ObservationBuilder:
    """Builds observation vectors from game state."""

    def __init__(
        self,
        loader: SeasonDataLoader,
        prediction_integrator: object | None = None,
    ) -> None:
        self.loader = loader
        self._integrator = prediction_integrator

    def build(
        self,
        state: GameState,
        candidate_pool: list[int],
    ) -> np.ndarray:
        """Build a flat observation vector.

        Layout:
        - Squad block (15 x 24 = 360): per-player features + squad context
        - Pool block (50 x 19 = 950): per-player features for candidates
        - Global block (53): game state features
        """
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        gw = state.current_gw
        offset = 0

        # Squad block
        for i, player in enumerate(state.squad.players):
            features = self._player_features(player.element_id, gw)
            # Add squad-specific features
            is_starter = 1.0 if i in state.squad.lineup else 0.0
            bench_order = 0.0
            if i in state.squad.bench:
                bench_order = float(state.squad.bench.index(i) + 1) / 4.0
            is_captain = 1.0 if i == state.squad.captain_idx else 0.0
            is_vice = 1.0 if i == state.squad.vice_captain_idx else 0.0
            purchase_price = float(player.purchase_price) / 1000.0

            squad_features = np.concatenate([
                features,  # 19 base features
                [is_starter, bench_order, is_captain, is_vice, purchase_price],
            ])
            end = offset + PLAYER_FEATURES
            obs[offset:end] = squad_features[:PLAYER_FEATURES]
            offset = end

        # Pool block
        for i in range(POOL_SIZE):
            if i < len(candidate_pool):
                features = self._player_features(candidate_pool[i], gw)
            else:
                features = np.zeros(POOL_FEATURES, dtype=np.float32)
            end = offset + POOL_FEATURES
            obs[offset:end] = features[:POOL_FEATURES]
            offset = end

        # Global block
        global_features = self._global_features(state)
        end = offset + GLOBAL_BLOCK
        obs[offset:end] = global_features[:GLOBAL_BLOCK]

        return obs

    def _player_features(self, element_id: int, gw: int) -> np.ndarray:
        """Build feature vector for a single player (19 features).

        Uses two data sources to avoid lookahead bias:
        - Pre-match data (price, ownership, was_home): from current GW
        - Post-match data (xG, xA, ICT, BPS, minutes, points): from previous GW
        This ensures the agent only sees information available before the deadline.
        """
        features = np.zeros(19, dtype=np.float32)

        # Pre-match data from current GW (available before deadline)
        pre_data = self.loader.get_player_gw(element_id, gw)
        if pre_data is None and gw > 1:
            pre_data = self.loader.get_player_gw(element_id, gw - 1)

        # Post-match data from PREVIOUS GW (no lookahead)
        prev_gw = gw - 1
        post_data = self.loader.get_player_gw(element_id, prev_gw) if prev_gw >= 1 else None

        pos = self.loader.get_player_position(element_id)

        # Position one-hot (4 features)
        if pos is not None:
            features[pos.value - 1] = 1.0

        # Pre-match features from current GW
        if pre_data is not None:
            features[4] = float(pre_data.get("value", 0)) / 1000.0
            features[13] = float(pre_data.get("selected", 0)) / 1e7
            features[17] = float(pre_data.get("was_home", 0))

        # Form (already uses past GWs only — safe)
        features[5] = self.loader.get_player_form(element_id, gw, window=5) / 10.0

        # Post-match features from PREVIOUS GW (avoids lookahead)
        if post_data is not None:
            features[6] = float(post_data.get("expected_goals", 0))
            features[7] = float(post_data.get("expected_assists", 0))
            features[8] = float(post_data.get("influence", 0)) / 100.0
            features[9] = float(post_data.get("creativity", 0)) / 100.0
            features[10] = float(post_data.get("threat", 0)) / 100.0
            features[11] = float(post_data.get("ict_index", 0)) / 100.0
            features[12] = float(post_data.get("bps", 0)) / 100.0
            features[14] = float(post_data.get("minutes", 0)) / 90.0
            features[15] = float(post_data.get("total_points", 0)) / 15.0
            features[16] = float(post_data.get("transfers_balance", 0)) / 1e5

        # Predicted points (from LightGBM model, or 0.0 if no integrator)
        if self._integrator is not None:
            features[18] = self._integrator.get_predicted_points(element_id, gw) / 15.0
        else:
            features[18] = 0.0

        return features

    def _global_features(self, state: GameState) -> np.ndarray:
        """Build global feature vector (53 features)."""
        features = np.zeros(GLOBAL_BLOCK, dtype=np.float32)
        idx = 0

        # GW number (normalized)
        features[idx] = float(state.current_gw) / TOTAL_GAMEWEEKS
        idx += 1

        # Bank (normalized)
        features[idx] = float(state.bank) / 1000.0
        idx += 1

        # Free transfers (normalized)
        features[idx] = float(state.free_transfers) / 5.0
        idx += 1

        # Team value (sum of selling prices, normalized)
        team_value = sum(p.selling_price for p in state.squad.players)
        features[idx] = float(team_value) / 1000.0
        idx += 1

        # Total points (normalized)
        features[idx] = float(state.total_points) / 2500.0
        idx += 1

        # Chip availability (8 booleans: 4 chips x 2 halves)
        for chip_name in ["wildcard", "free_hit", "bench_boost", "triple_captain"]:
            chip_list = state.chips._get_chip_list(chip_name)
            features[idx] = float(chip_list[0])
            idx += 1
            features[idx] = float(chip_list[1])
            idx += 1

        # DGW/BGW flags (20 teams each)
        # Detect which teams have double or blank gameweeks from fixture data
        teams_playing = self.loader.get_teams_playing(state.current_gw)
        for team_id in range(1, 21):
            if teams_playing and self.loader.is_dgw(team_id, state.current_gw):
                features[idx] = 1.0
            idx += 1

        for team_id in range(1, 21):
            if teams_playing and team_id not in teams_playing:
                features[idx] = 1.0  # blank gameweek for this team
            idx += 1

        return features
