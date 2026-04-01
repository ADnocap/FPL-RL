"""Tests for the FPLEnv Gymnasium environment."""

import numpy as np
import pytest

from fpl_rl.env.action_space import ACTION_DIMS, CHIP_DIM, MASK_LENGTH, MAX_TRANSFERS_PER_STEP
from fpl_rl.env.fpl_env import FPLEnv, PRESEASON_STEPS


def _noop_action() -> np.ndarray:
    """Build a no-op action (0 transfers, keep captain/vice/formation)."""
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
    """Create an FPLEnv with test data."""
    from fpl_rl.data.loader import SeasonDataLoader

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
    env._preseason_steps_remaining = 0
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

    def test_reset_gw1_has_many_free_transfers(self, env):
        env.reset(seed=42)
        assert env.state.free_transfers >= 5

    def test_reset_initializes_preseason(self, env):
        env.reset(seed=42)
        assert env._preseason_steps_remaining == PRESEASON_STEPS


class TestFPLEnvPreseason:
    """Pre-season steps: squad building before GW1."""

    def test_preseason_no_gw_advance(self, env):
        env.reset(seed=42)
        action = _noop_action()

        # Both preseason steps should keep current_gw = 1
        for i in range(PRESEASON_STEPS):
            obs, reward, term, trunc, info = env.step(action)
            assert env.state.current_gw == 1, f"GW advanced during preseason step {i+1}"
            assert info["preseason"] is True
            assert info["gw"] == 0
            assert reward == 0.0
            assert not term

    def test_preseason_then_real_gw1(self, env):
        env.reset(seed=42)
        action = _noop_action()

        # Skip preseason steps
        for _ in range(PRESEASON_STEPS):
            env.step(action)

        # Next step should be real GW1
        obs, reward, term, trunc, info = env.step(action)
        assert info["preseason"] is False
        assert info["gw"] == 1
        assert env.state.current_gw == 2  # advanced to GW2

    def test_preseason_ft_consumed(self, env):
        """Free transfers consumed during preseason but not banked."""
        env.reset(seed=42)
        initial_ft = env.state.free_transfers  # 15

        # No-op = 0 transfers → FT unchanged
        action = _noop_action()
        env.step(action)
        assert env.state.free_transfers == initial_ft

    def test_preseason_chips_masked(self, env):
        env.reset(seed=42)
        masks = env.action_masks()

        # Chip dimension is the last CHIP_DIM values
        chip_offset = MASK_LENGTH - CHIP_DIM
        assert masks[chip_offset] == True  # chip=none always allowed
        for i in range(1, CHIP_DIM):
            assert masks[chip_offset + i] == False, (
                f"Chip index {i} should be masked during preseason"
            )

    def test_gw1_resets_ft_to_one(self, env):
        """After all preseason + GW1 step, FTs reset to 1."""
        env.reset(seed=42)
        action = _noop_action()

        # Preseason steps
        for _ in range(PRESEASON_STEPS):
            env.step(action)

        # Real GW1
        env.step(action)

        from fpl_rl.utils.constants import INITIAL_FREE_TRANSFERS
        assert env.state.free_transfers == INITIAL_FREE_TRANSFERS

    def test_episode_length(self, env):
        """Episode = PRESEASON_STEPS + num_gws steps."""
        env.reset(seed=42)
        steps = 0
        while True:
            obs, reward, term, trunc, info = env.step(_noop_action())
            steps += 1
            if term or trunc:
                break

        assert steps == PRESEASON_STEPS + env._num_gws


class TestFPLEnvStep:
    def test_step_returns_correct_tuple(self, env):
        env.reset(seed=42)
        action = env.action_space.sample()
        result = env.step(action)

        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert obs.shape == env.observation_space.shape
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_step_advances_gw_after_preseason(self, env):
        env.reset(seed=42)
        action = env.action_space.sample()

        # Skip preseason
        for _ in range(PRESEASON_STEPS):
            env.step(action)

        # Real GW1 step should advance
        env.step(action)
        assert env.state.current_gw == 2

    def test_noop_step(self, env):
        env.reset(seed=42)
        action = _noop_action()

        # Skip preseason
        for _ in range(PRESEASON_STEPS):
            env.step(action)

        # Real GW1
        obs, reward, terminated, truncated, info = env.step(action)
        assert not terminated
        assert info["gw"] == 1
        assert info["hit_cost"] == 0

    def test_full_season_no_crash(self, env):
        """Run through preseason + all available GWs without crashing."""
        env.reset(seed=42)
        total_reward = 0.0
        steps = 0

        while True:
            action = _noop_action()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

            if terminated or truncated:
                break

        assert steps == PRESEASON_STEPS + env._num_gws
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

        offset = 0
        for dim_size in env.action_space.nvec:
            dim_mask = masks[offset : offset + dim_size]
            assert dim_mask.any()
            offset += dim_size

    def test_step_without_reset_raises(self, env):
        with pytest.raises(RuntimeError, match="reset"):
            env.step(env.action_space.sample())
