"""SB3 callbacks for FPL RL training with action-mask support."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

logger = logging.getLogger(__name__)


class FPLEvalCallback(BaseCallback):
    """Evaluate a MaskablePPO agent on a held-out season.

    SB3's built-in ``EvalCallback`` doesn't pass ``action_masks`` to
    ``model.predict``, so we need a custom callback.

    Parameters
    ----------
    eval_env : gymnasium.Env
        Environment to evaluate on (must expose ``action_masks()``).
    eval_freq : int
        Evaluate every *eval_freq* training timesteps.
    n_eval_episodes : int
        Number of full episodes to run per evaluation.
    best_model_save_path : Path | None
        If set, save the best model (by mean total points) here.
    """

    def __init__(
        self,
        eval_env,
        eval_freq: int = 10_000,
        n_eval_episodes: int = 3,
        best_model_save_path: Path | None = None,
        verbose: int = 0,
        show_progress: bool = False,
    ) -> None:
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.best_model_save_path = best_model_save_path
        self.best_mean_points = -np.inf
        self.show_progress = show_progress

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True

        total_points_list: list[float] = []

        episodes = range(self.n_eval_episodes)
        if self.show_progress and self.n_eval_episodes > 1:
            from tqdm.auto import tqdm
            episodes = tqdm(
                episodes, desc="Eval episodes", unit="ep", leave=False,
            )

        for _ in episodes:
            obs, info = self.eval_env.reset()
            done = False
            episode_total_points = 0.0

            while not done:
                masks = self.eval_env.action_masks()
                action, _ = self.model.predict(
                    obs, deterministic=True, action_masks=masks,
                )
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                done = terminated or truncated
                episode_total_points = info.get("total_points", episode_total_points)

            total_points_list.append(episode_total_points)

        mean_pts = float(np.mean(total_points_list))
        std_pts = float(np.std(total_points_list))

        self.logger.record("eval/mean_total_points", mean_pts)
        self.logger.record("eval/std_total_points", std_pts)

        if self.verbose >= 1:
            logger.info(
                "Eval at step %d: %.1f +/- %.1f pts (%d episodes)",
                self.num_timesteps, mean_pts, std_pts, self.n_eval_episodes,
            )

        # Save best model
        if self.best_model_save_path is not None and mean_pts > self.best_mean_points:
            self.best_mean_points = mean_pts
            save_path = Path(self.best_model_save_path) / "best_model"
            self.model.save(str(save_path))
            if self.verbose >= 1:
                logger.info("New best model: %.1f pts -> %s", mean_pts, save_path)

        return True


class FPLEpisodeLogCallback(BaseCallback):
    """Log training episode total points to TensorBoard.

    Detects episode completion via ``dones`` in ``_on_step`` and reads
    ``total_points`` from the info dict.
    """

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)
        self._episode_count = 0

    def _on_step(self) -> bool:
        # SB3 stores infos from the last step in self.locals
        infos = self.locals.get("infos")
        dones = self.locals.get("dones")

        if infos is None or dones is None:
            return True

        for i, done in enumerate(dones):
            if done:
                info = infos[i]
                # SB3 auto-reset wrappers store the terminal info under
                # "terminal_info" or "terminal_observation"; but for
                # non-vectorized envs, info is the terminal info directly.
                total_pts = info.get("total_points", None)
                if total_pts is not None:
                    self._episode_count += 1
                    self.logger.record("train/episode_total_points", total_pts)
                    self.logger.record("train/episodes", self._episode_count)

        return True
