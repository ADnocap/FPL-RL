"""SB3 smoke test: train MaskablePPO for a few steps.

This test requires sb3-contrib to be installed.
Run with: pytest tests/test_integration/test_sb3_smoke.py -v
"""

import numpy as np
import pytest


def _sb3_available() -> bool:
    try:
        import sb3_contrib  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.fixture
def env(test_data_dir):
    """Create an FPLEnv with test data for SB3 smoke test."""
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


@pytest.mark.skipif(not _sb3_available(), reason="sb3-contrib not installed")
class TestSB3Smoke:
    def test_maskable_ppo_train(self, env):
        """Train MaskablePPO for a small number of steps."""
        from sb3_contrib import MaskablePPO

        model = MaskablePPO("MlpPolicy", env, verbose=0, n_steps=4, batch_size=2)
        model.learn(total_timesteps=8)

        # Verify model can predict
        obs, _ = env.reset(seed=42)
        action, _ = model.predict(obs, action_masks=env.action_masks())
        assert env.action_space.contains(action)
