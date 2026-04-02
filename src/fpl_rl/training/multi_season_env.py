"""Multi-season cycling Gymnasium wrapper for FPL RL training."""

from __future__ import annotations

import logging
from pathlib import Path

import gymnasium
import numpy as np

from fpl_rl.data.downloader import DEFAULT_DATA_DIR
from fpl_rl.env.action_space import create_action_space
from fpl_rl.env.fpl_env import FPLEnv
from fpl_rl.env.observation_space import create_observation_space

# Lazy imports for hybrid mode (avoid importing optimizer at module level)
_HybridFPLEnv = None
_create_hybrid_action_space = None

logger = logging.getLogger(__name__)


class MultiSeasonFPLEnv(gymnasium.Env):
    """Gymnasium env that cycles through multiple seasons on each reset().

    Inner ``FPLEnv`` instances are cached per season so that the expensive
    ``SeasonDataLoader`` construction (CSV reads, index building, position/team
    backfill) happens only once per season rather than every episode.

    Parameters
    ----------
    seasons : list[str]
        Seasons to cycle through (e.g. ``["2016-17", ..., "2022-23"]``).
    data_dir : Path
        Root data directory for ``SeasonDataLoader`` (e.g. ``data/raw``).
    predictor_model_dir : Path | None
        If provided, pre-compute ``PredictionIntegrator`` for every season at
        init time and pass the cached integrator to each inner ``FPLEnv``.
    prediction_data_dir : Path | None
        Root data directory for the prediction pipeline (``IDResolver``,
        ``FeaturePipeline``).  Defaults to *data_dir*'s parent when
        *data_dir* ends in ``raw``, otherwise falls back to *data_dir*.
    shuffle : bool
        If ``True``, sample a random season on each reset.  Otherwise cycle
        round-robin.
    hybrid : bool
        If ``True``, use ``HybridFPLEnv`` (RL+MILP) instead of ``FPLEnv``.
    """

    metadata = {"render_modes": ["human"], "name": "MultiSeasonFPLEnv-v0"}

    def __init__(
        self,
        seasons: list[str],
        data_dir: Path = DEFAULT_DATA_DIR,
        predictor_model_dir: Path | None = None,
        prediction_data_dir: Path | None = None,
        shuffle: bool = False,
        hybrid: bool = False,
    ) -> None:
        super().__init__()

        if not seasons:
            raise ValueError("Must provide at least one season")

        self.seasons = list(seasons)
        self.data_dir = data_dir
        self.predictor_model_dir = predictor_model_dir
        self.shuffle = shuffle
        self.hybrid = hybrid

        # Resolve the prediction data dir (parent of raw/ by convention)
        if prediction_data_dir is not None:
            pred_data_dir = prediction_data_dir
        elif data_dir.name == "raw":
            pred_data_dir = data_dir.parent
        else:
            pred_data_dir = data_dir

        # Pre-cache prediction integrators per season
        self._integrators: dict = {}
        if predictor_model_dir is not None:
            from tqdm.auto import tqdm
            from fpl_rl.prediction.integration import PredictionIntegrator
            for season in tqdm(self.seasons, desc="Caching predictions", unit="season"):
                self._integrators[season] = PredictionIntegrator.from_model(
                    predictor_model_dir, pred_data_dir, season,
                )

        # Spaces (same for all seasons)
        if hybrid:
            from fpl_rl.env.hybrid_action_space import create_hybrid_action_space
            self.action_space = create_hybrid_action_space()
        else:
            self.action_space = create_action_space()
        self.observation_space = create_observation_space()

        # Cycling state
        self._season_idx = 0
        self._inner_env: FPLEnv | None = None
        self._env_cache: dict[str, FPLEnv] = {}

    @property
    def current_season(self) -> str:
        """The season that will be used on the next (or current) episode."""
        return self.seasons[self._season_idx % len(self.seasons)]

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        # Pick season
        if self.shuffle:
            idx = self.np_random.integers(0, len(self.seasons))
            season = self.seasons[int(idx)]
        else:
            season = self.seasons[self._season_idx % len(self.seasons)]
            self._season_idx += 1

        # Reuse cached env or create once
        if season not in self._env_cache:
            integrator = self._integrators.get(season)
            if self.hybrid:
                from fpl_rl.env.hybrid_env import HybridFPLEnv
                self._env_cache[season] = HybridFPLEnv(
                    season=season,
                    data_dir=self.data_dir,
                    prediction_integrator=integrator,
                )
            else:
                self._env_cache[season] = FPLEnv(
                    season=season,
                    data_dir=self.data_dir,
                    prediction_integrator=integrator,
                )

        self._inner_env = self._env_cache[season]
        # Share the rng so the inner env's squad randomisation is seeded
        self._inner_env.np_random = self.np_random

        obs, info = self._inner_env.reset(seed=None)
        info["season"] = season
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self._inner_env is None:
            raise RuntimeError("Must call reset() before step()")
        return self._inner_env.step(action)

    def action_masks(self) -> np.ndarray:
        if self._inner_env is None:
            return np.ones(sum(self.action_space.nvec), dtype=bool)
        return self._inner_env.action_masks()

    def close(self) -> None:
        for env in self._env_cache.values():
            env.close()
        self._env_cache.clear()
        self._inner_env = None
        super().close()
