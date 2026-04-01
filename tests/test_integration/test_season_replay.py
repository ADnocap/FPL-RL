"""Integration tests: full season replay and random actions."""

import numpy as np
import pytest

from fpl_rl.env.action_space import ACTION_DIMS, MAX_TRANSFERS_PER_STEP


def _noop_action() -> np.ndarray:
    """Build a no-op action (0 transfers, keep defaults)."""
    a = np.zeros(len(ACTION_DIMS), dtype=int)
    base = 1 + MAX_TRANSFERS_PER_STEP * 2  # 11
    a[base] = 0       # captain
    a[base + 1] = 1   # vice
    a[base + 2] = 0   # formation
    a[base + 3] = 3   # bench_1
    a[base + 4] = 4   # bench_2
    a[base + 5] = 5   # bench_3
    a[base + 6] = 0   # chip (none)
    return a


@pytest.fixture
def env(test_data_dir):
    """Create an FPLEnv with test data for integration tests."""
    from fpl_rl.data.loader import SeasonDataLoader
    from fpl_rl.engine.engine import FPLGameEngine
    from fpl_rl.env.action_space import ActionEncoder, create_action_space
    from fpl_rl.env.fpl_env import FPLEnv
    from fpl_rl.env.observation_space import ObservationBuilder, create_observation_space
    from fpl_rl.env.reward import RewardCalculator

    env = FPLEnv.__new__(FPLEnv)
    env.season = "test-season"
    env.render_mode = None

    original_init = SeasonDataLoader.__init__

    def patched_init(self, season, data_dir):
        self.season = season
        self.data_dir = data_dir
        self._season_dir = data_dir / season
        self._merged_gw = self._load_merged_gw()
        self._player_info = self._load_player_info()
        self._fixtures = self._load_fixtures()
        self._teams = self._load_teams()
        self._gw_index = {}
        for idx, row in self._merged_gw.iterrows():
            key = (int(row["element"]), int(row["GW"]))
            self._gw_index.setdefault(key, []).append(idx)
        self._position_map = self._build_position_map()
        self._team_map = self._build_team_map()

    SeasonDataLoader.__init__ = patched_init
    try:
        env.loader = SeasonDataLoader("test-season", test_data_dir)
    finally:
        SeasonDataLoader.__init__ = original_init

    env.engine = FPLGameEngine(env.loader)
    env.action_encoder = ActionEncoder(env.loader)
    env.obs_builder = ObservationBuilder(env.loader)
    env.reward_calc = RewardCalculator(env.loader)
    env.action_space = create_action_space()
    env.observation_space = create_observation_space()
    env.state = None
    env._num_gws = min(env.loader.get_num_gameweeks(), 38)
    env.np_random = np.random.default_rng(42)
    env.metadata = {"render_modes": ["human"], "name": "FPLEnv-v0"}

    return env


class TestNoOpReplay:
    """Full season replay with no-op agent (no transfers)."""

    def test_noop_full_season(self, env):
        obs, info = env.reset(seed=42)

        total_points = 0
        gws_played = 0

        while True:
            action = _noop_action()
            obs, reward, terminated, truncated, info = env.step(action)
            gws_played += 1

            assert not np.isnan(obs).any(), f"NaN in obs at GW{gws_played}"
            assert not np.isinf(obs).any(), f"Inf in obs at GW{gws_played}"
            assert env.observation_space.contains(obs)

            if terminated or truncated:
                break

        assert gws_played == env._num_gws
        assert env.state.total_points != 0


class TestRandomActions:
    """Random valid actions for full season without crashes."""

    def test_random_actions_full_season(self, env):
        obs, info = env.reset(seed=42)
        rng = np.random.default_rng(123)

        gws_played = 0

        while True:
            # Sample random action
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            gws_played += 1

            assert not np.isnan(obs).any(), f"NaN in obs at GW{gws_played}"
            assert env.observation_space.contains(obs)

            if terminated or truncated:
                break

        assert gws_played == env._num_gws

    def test_masked_random_actions(self, env):
        """Use action masks to filter random actions."""
        obs, info = env.reset(seed=42)

        gws_played = 0

        while True:
            masks = env.action_masks()
            # Sample from masked action space
            action = np.zeros(len(env.action_space.nvec), dtype=int)
            offset = 0
            for i, dim_size in enumerate(env.action_space.nvec):
                dim_mask = masks[offset : offset + dim_size]
                valid_indices = np.where(dim_mask)[0]
                if len(valid_indices) > 0:
                    action[i] = np.random.choice(valid_indices)
                offset += dim_size

            obs, reward, terminated, truncated, info = env.step(action)
            gws_played += 1

            if terminated or truncated:
                break

        assert gws_played == env._num_gws


class TestMultipleResets:
    """Test resetting the environment multiple times."""

    def test_multiple_resets(self, env):
        for seed in [42, 123, 456]:
            obs1, _ = env.reset(seed=seed)
            assert env.state.current_gw == 1
            assert env.state.total_points == 0

            # Take one step
            action = _noop_action()
            env.step(action)
            assert env.state.current_gw == 2

            # Reset again
            obs2, _ = env.reset(seed=seed)
            assert env.state.current_gw == 1
