"""Tests for FPL training callbacks."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest


def _sb3_available() -> bool:
    try:
        import sb3_contrib  # noqa: F401
        return True
    except ImportError:
        return False


def _make_patched_env(test_data_dir):
    """Create a monkey-patched FPLEnv using test data."""
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
class TestFPLEvalCallback:
    def test_eval_fires_and_saves_best(self, test_data_dir, tmp_path):
        """Eval callback should fire at the right frequency and save best model."""
        from sb3_contrib import MaskablePPO
        from fpl_rl.training.callbacks import FPLEvalCallback

        train_env = _make_patched_env(test_data_dir)
        eval_env = _make_patched_env(test_data_dir)

        model = MaskablePPO("MlpPolicy", train_env, verbose=0,
                            n_steps=4, batch_size=2, seed=42)

        save_path = tmp_path / "best"
        save_path.mkdir()

        eval_cb = FPLEvalCallback(
            eval_env=eval_env,
            eval_freq=4,
            n_eval_episodes=1,
            best_model_save_path=save_path,
        )

        model.learn(total_timesteps=8, callback=[eval_cb])

        # Best model should have been saved
        assert (save_path / "best_model.zip").exists()

        train_env.close()
        eval_env.close()

    def test_eval_records_metrics(self, test_data_dir, tmp_path):
        """Eval callback should record eval/mean_total_points."""
        from sb3_contrib import MaskablePPO
        from fpl_rl.training.callbacks import FPLEvalCallback

        train_env = _make_patched_env(test_data_dir)
        eval_env = _make_patched_env(test_data_dir)

        model = MaskablePPO("MlpPolicy", train_env, verbose=0,
                            n_steps=4, batch_size=2, seed=42)

        save_path = tmp_path / "best"
        save_path.mkdir()

        # Use eval_freq=1 to guarantee the callback fires
        eval_cb = FPLEvalCallback(
            eval_env=eval_env,
            eval_freq=1,
            n_eval_episodes=1,
            best_model_save_path=save_path,
        )

        model.learn(total_timesteps=8, callback=[eval_cb])

        # The callback should have fired and updated best_mean_points
        assert eval_cb.best_mean_points > -np.inf

        train_env.close()
        eval_env.close()


@pytest.mark.skipif(not _sb3_available(), reason="sb3-contrib not installed")
class TestFPLEpisodeLogCallback:
    def test_episode_count_increments(self, test_data_dir):
        """Episode log callback should count completed episodes."""
        from sb3_contrib import MaskablePPO
        from fpl_rl.training.callbacks import FPLEpisodeLogCallback

        env = _make_patched_env(test_data_dir)

        model = MaskablePPO("MlpPolicy", env, verbose=0,
                            n_steps=4, batch_size=2, seed=42)

        episode_cb = FPLEpisodeLogCallback()
        model.learn(total_timesteps=8, callback=[episode_cb])

        # With 2 GWs in test data, episodes are short — should complete at least 1
        assert episode_cb._episode_count >= 1

        env.close()
