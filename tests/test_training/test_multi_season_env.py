"""Tests for MultiSeasonFPLEnv."""

from __future__ import annotations

import numpy as np
import pytest

from fpl_rl.data.loader import SeasonDataLoader
from fpl_rl.env.action_space import create_action_space
from fpl_rl.env.observation_space import create_observation_space


def _patch_loader_init():
    """Return (patched_init, original_init) for SeasonDataLoader."""
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

    return patched_init, original_init


@pytest.fixture
def multi_season_env(test_data_dir):
    """Create a MultiSeasonFPLEnv with test data using two 'seasons'."""
    import shutil
    from pathlib import Path
    from fpl_rl.training.multi_season_env import MultiSeasonFPLEnv

    # Create a second "season" by copying the same test data
    test_data_src = Path(__file__).parent.parent / "test_data"
    for season_name in ["season-a", "season-b"]:
        season_dir = test_data_dir / season_name
        gws_dir = season_dir / "gws"
        gws_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(test_data_src / "sample_merged_gw.csv", gws_dir / "merged_gw.csv")
        shutil.copy(test_data_src / "sample_cleaned_players.csv", season_dir / "cleaned_players.csv")
        shutil.copy(test_data_src / "sample_fixtures.csv", season_dir / "fixtures.csv")
        shutil.copy(test_data_src / "sample_teams.csv", season_dir / "teams.csv")

    patched_init, original_init = _patch_loader_init()
    SeasonDataLoader.__init__ = patched_init
    try:
        env = MultiSeasonFPLEnv(
            seasons=["season-a", "season-b"],
            data_dir=test_data_dir,
            shuffle=False,
        )
    finally:
        SeasonDataLoader.__init__ = original_init

    # Keep the patch active for reset/step calls
    env._original_loader_init = original_init
    env._patched_loader_init = patched_init
    yield env
    # Restore
    SeasonDataLoader.__init__ = original_init
    env.close()


class TestMultiSeasonEnv:
    def test_spaces_match_inner_env(self, multi_season_env):
        """Spaces should match the standard FPLEnv spaces."""
        expected_action = create_action_space()
        expected_obs = create_observation_space()

        np.testing.assert_array_equal(
            multi_season_env.action_space.nvec, expected_action.nvec,
        )
        np.testing.assert_array_equal(
            multi_season_env.observation_space.shape, expected_obs.shape,
        )

    def test_round_robin_cycling(self, multi_season_env):
        """Seasons cycle in order: a, b, a, b, ..."""
        env = multi_season_env
        # Patch loader for each reset call
        SeasonDataLoader.__init__ = env._patched_loader_init
        try:
            obs1, info1 = env.reset(seed=42)
            assert info1["season"] == "season-a"

            obs2, info2 = env.reset(seed=43)
            assert info2["season"] == "season-b"

            obs3, info3 = env.reset(seed=44)
            assert info3["season"] == "season-a"
        finally:
            SeasonDataLoader.__init__ = env._original_loader_init

    def test_shuffle_mode(self, test_data_dir):
        """With shuffle=True, seasons are sampled randomly."""
        import shutil
        from pathlib import Path
        from fpl_rl.training.multi_season_env import MultiSeasonFPLEnv

        test_data_src = Path(__file__).parent.parent / "test_data"
        for season_name in ["season-x", "season-y"]:
            season_dir = test_data_dir / season_name
            gws_dir = season_dir / "gws"
            gws_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(test_data_src / "sample_merged_gw.csv", gws_dir / "merged_gw.csv")
            shutil.copy(test_data_src / "sample_cleaned_players.csv", season_dir / "cleaned_players.csv")
            shutil.copy(test_data_src / "sample_fixtures.csv", season_dir / "fixtures.csv")
            shutil.copy(test_data_src / "sample_teams.csv", season_dir / "teams.csv")

        patched_init, original_init = _patch_loader_init()
        SeasonDataLoader.__init__ = patched_init
        try:
            env = MultiSeasonFPLEnv(
                seasons=["season-x", "season-y"],
                data_dir=test_data_dir,
                shuffle=True,
            )
            # With shuffle, we should get a mix over many resets
            seasons_seen = set()
            for i in range(10):
                _, info = env.reset(seed=i)
                seasons_seen.add(info["season"])
            assert len(seasons_seen) == 2, f"Expected both seasons, got {seasons_seen}"
            env.close()
        finally:
            SeasonDataLoader.__init__ = original_init

    def test_action_masks_delegate(self, multi_season_env):
        """action_masks() should delegate to the inner env."""
        env = multi_season_env
        SeasonDataLoader.__init__ = env._patched_loader_init
        try:
            env.reset(seed=42)
            masks = env.action_masks()
            expected_len = sum(create_action_space().nvec)
            assert masks.shape == (expected_len,)
            assert masks.dtype == bool
        finally:
            SeasonDataLoader.__init__ = env._original_loader_init

    def test_full_episode_completion(self, multi_season_env):
        """Run a full episode and check it terminates."""
        env = multi_season_env
        SeasonDataLoader.__init__ = env._patched_loader_init
        try:
            obs, info = env.reset(seed=42)
            done = False
            steps = 0
            while not done:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                steps += 1
                assert steps <= 40, "Episode should terminate within 38 GWs"

            assert steps >= 1
            assert "total_points" in info
        finally:
            SeasonDataLoader.__init__ = env._original_loader_init

    def test_action_masks_before_reset(self, multi_season_env):
        """action_masks() before reset should return all-True."""
        masks = multi_season_env.action_masks()
        assert masks.all()

    def test_empty_seasons_raises(self):
        """Passing empty seasons list should raise ValueError."""
        from fpl_rl.training.multi_season_env import MultiSeasonFPLEnv
        with pytest.raises(ValueError, match="at least one season"):
            MultiSeasonFPLEnv(seasons=[])

    def test_memory_cleanup_on_reset(self, multi_season_env):
        """Old inner env should be cleaned up on each reset."""
        env = multi_season_env
        SeasonDataLoader.__init__ = env._patched_loader_init
        try:
            env.reset(seed=42)
            first_inner = env._inner_env
            assert first_inner is not None

            env.reset(seed=43)
            second_inner = env._inner_env
            assert second_inner is not None
            assert second_inner is not first_inner
        finally:
            SeasonDataLoader.__init__ = env._original_loader_init
