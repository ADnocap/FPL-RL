"""Tests for the FPLEnv Gymnasium environment."""

import numpy as np
import pytest

from fpl_rl.env.fpl_env import FPLEnv


@pytest.fixture
def env(test_data_dir):
    """Create an FPLEnv with test data."""
    from fpl_rl.data.loader import SeasonDataLoader

    # Patch the env to use test data
    env = FPLEnv.__new__(FPLEnv)
    env.season = "test-season"
    env.render_mode = None

    # Create loader from test data (bypass download)
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

    from fpl_rl.engine.engine import FPLGameEngine
    from fpl_rl.env.action_space import ActionEncoder, create_action_space
    from fpl_rl.env.observation_space import ObservationBuilder, create_observation_space
    from fpl_rl.env.reward import RewardCalculator

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


class TestFPLEnvReset:
    def test_reset_returns_obs_info(self, env):
        obs, info = env.reset(seed=42)
        assert obs.shape == env.observation_space.shape
        assert isinstance(info, dict)
        assert "season" in info

    def test_reset_initializes_state(self, env):
        env.reset(seed=42)
        assert env.state is not None
        assert env.state.current_gw == 1
        assert env.state.total_points == 0
        assert len(env.state.squad.players) == 15

    def test_reset_obs_no_nan(self, env):
        obs, _ = env.reset(seed=42)
        assert not np.isnan(obs).any()

    def test_reset_obs_in_space(self, env):
        obs, _ = env.reset(seed=42)
        assert env.observation_space.contains(obs)


class TestFPLEnvStep:
    def test_step_returns_correct_tuple(self, env):
        env.reset(seed=42)
        action = env.action_space.sample()
        result = env.step(action)

        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert obs.shape == env.observation_space.shape
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_step_advances_gw(self, env):
        env.reset(seed=42)
        action = env.action_space.sample()
        env.step(action)

        assert env.state.current_gw == 2

    def test_noop_step(self, env):
        env.reset(seed=42)
        # No-op action: 0 transfers, captain/vice stay, no chip
        action = np.array([0, 0, 0, 0, 0, 0, 1, 0])
        obs, reward, terminated, truncated, info = env.step(action)

        assert not terminated
        assert info["gw"] == 1
        assert info["hit_cost"] == 0

    def test_full_season_no_crash(self, env):
        """Run through all available GWs without crashing."""
        env.reset(seed=42)
        total_reward = 0.0
        gw = 0

        while True:
            action = np.array([0, 0, 0, 0, 0, 0, 1, 0])  # no-op
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            gw += 1

            if terminated or truncated:
                break

        assert gw == env._num_gws
        assert env.state.total_points != 0


class TestFPLEnvMasks:
    def test_action_masks_shape(self, env):
        env.reset(seed=42)
        masks = env.action_masks()
        assert masks.shape == (sum(env.action_space.nvec),)
        assert masks.dtype == bool

    def test_action_masks_have_valid_actions(self, env):
        env.reset(seed=42)
        masks = env.action_masks()

        # Each dimension should have at least one valid action
        offset = 0
        for dim_size in env.action_space.nvec:
            dim_mask = masks[offset : offset + dim_size]
            assert dim_mask.any()
            offset += dim_size

    def test_step_without_reset_raises(self, env):
        with pytest.raises(RuntimeError, match="reset"):
            env.step(env.action_space.sample())
