#!/usr/bin/env python3
"""Standalone RL training script for cluster (SLURM) execution.

Designed for long-running batch jobs with periodic checkpointing.
Saves model checkpoints, best model, and TensorBoard metrics.

Usage:
    python scripts/train_rl_cluster.py
    python scripts/train_rl_cluster.py --total-steps 10000000 --n-envs 30
"""

from __future__ import annotations

import argparse
import datetime
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from sb3_contrib import MaskablePPO

from fpl_rl.env.fpl_env import FPLEnv
from fpl_rl.prediction.integration import PredictionIntegrator
from fpl_rl.training.callbacks import FPLEvalCallback, FPLEpisodeLogCallback


class RollingCheckpointCallback(BaseCallback):
    """Save a single rolling checkpoint, deleting the previous one."""

    def __init__(self, save_freq: int, save_path: str, verbose: int = 1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        self._prev_path: Path | None = None

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            path = self.save_path / f"checkpoint_{self.num_timesteps}_steps"
            self.model.save(str(path))
            if self.verbose:
                print(f"Checkpoint saved: {path.name}")
            # Delete previous checkpoint
            if self._prev_path and self._prev_path.exists():
                self._prev_path.unlink(missing_ok=True)
            self._prev_path = Path(str(path) + ".zip")
        return True


class PPOMetricsCallback(BaseCallback):
    """Log PPO training metrics to stderr after each rollout.

    SB3 stores metrics in model.logger after each train() call. This
    callback reads them after each rollout and logs a parseable line.
    """

    def __init__(self, log_freq: int = 1):
        super().__init__(verbose=0)
        self.log_freq = log_freq
        self._rollout_count = 0
        self._log = logging.getLogger("train_rl")

    def _on_rollout_end(self) -> None:
        self._rollout_count += 1
        if self._rollout_count % self.log_freq != 0:
            return

        logger = self.model.logger
        kvs = logger.name_to_value if hasattr(logger, "name_to_value") else {}

        metrics = {
            "loss": kvs.get("train/loss", float("nan")),
            "policy_loss": kvs.get("train/policy_gradient_loss", float("nan")),
            "value_loss": kvs.get("train/value_loss", float("nan")),
            "entropy": kvs.get("train/entropy_loss", float("nan")),
            "approx_kl": kvs.get("train/approx_kl", float("nan")),
            "clip_frac": kvs.get("train/clip_fraction", float("nan")),
            "expl_var": kvs.get("train/explained_variance", float("nan")),
            "mean_reward": kvs.get("rollout/ep_rew_mean", float("nan")),
            "mean_ep_len": kvs.get("rollout/ep_len_mean", float("nan")),
        }

        parts = " | ".join(f"{k}={v:.4f}" for k, v in metrics.items() if not np.isnan(v))
        if parts:
            self._log.info(f"PPO step {self.num_timesteps}: {parts}")

    def _on_step(self) -> bool:
        return True


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train MaskablePPO on FPL env")
    p.add_argument("--data-dir", type=str, default="data/raw")
    p.add_argument("--predictor-dir", type=str, default="models/point_predictor")
    p.add_argument("--run-dir", type=str, default="runs/fpl_ppo")
    p.add_argument("--total-steps", type=int, default=5_000_000)
    p.add_argument("--n-envs", type=int, default=30)
    p.add_argument("--n-steps", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--ent-coef", type=float, default=0.02)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--eval-freq", type=int, default=50_000)
    p.add_argument("--checkpoint-freq", type=int, default=100_000)
    p.add_argument("--n-eval-episodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", type=str, default=None,
                   help="Path to model .zip to resume from")
    p.add_argument("--hybrid", action="store_true",
                   help="Use hybrid RL+MILP env (2-dim action space)")
    return p.parse_args()


def make_train_env(rank, seasons, data_dir, predictor_dir, pred_data_dir, hybrid=False):
    def _init():
        from fpl_rl.training.multi_season_env import MultiSeasonFPLEnv
        return MultiSeasonFPLEnv(
            seasons=seasons,
            data_dir=Path(data_dir),
            predictor_model_dir=Path(predictor_dir) if predictor_dir else None,
            prediction_data_dir=Path(pred_data_dir) if predictor_dir else None,
            shuffle=True,
            hybrid=hybrid,
        )
    return _init


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("train_rl")

    data_dir = Path(args.data_dir)
    pred_data_dir = data_dir.parent  # data/ root for id_maps, understat, etc.
    predictor_dir = Path(args.predictor_dir) if Path(args.predictor_dir).exists() else None
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    train_seasons = [
        "2016-17", "2017-18", "2018-19", "2019-20",
        "2020-21", "2021-22", "2022-23",
    ]
    eval_season = "2023-24"

    log.info("=" * 60)
    log.info("FPL RL Training — Cluster Job")
    log.info("=" * 60)
    log.info(f"Total timesteps:  {args.total_steps:,}")
    log.info(f"Parallel envs:    {args.n_envs}")
    log.info(f"Batch size:       {args.batch_size}")
    log.info(f"Checkpoint every: {args.checkpoint_freq:,} steps")
    log.info(f"Eval every:       {args.eval_freq:,} steps")
    log.info(f"Predictor:        {predictor_dir or 'disabled'}")
    log.info(f"Resume from:      {args.resume or 'scratch'}")
    log.info(f"Run dir:          {run_dir.resolve()}")
    log.info(f"Start time:       {datetime.datetime.now()}")
    log.info("=" * 60)

    # --- Build environments ---
    log.info(f"Building {args.n_envs} parallel training envs (hybrid={args.hybrid})...")
    train_env = SubprocVecEnv([
        make_train_env(i, train_seasons, data_dir, predictor_dir, pred_data_dir,
                       hybrid=args.hybrid)
        for i in range(args.n_envs)
    ])

    log.info(f"Building eval env ({eval_season})...")
    eval_kwargs = dict(season=eval_season, data_dir=data_dir)
    if predictor_dir:
        eval_kwargs["prediction_integrator"] = PredictionIntegrator.from_model(
            predictor_dir, pred_data_dir, eval_season,
        )
    if args.hybrid:
        from fpl_rl.env.hybrid_env import HybridFPLEnv
        eval_env = HybridFPLEnv(**eval_kwargs)
    else:
        eval_env = FPLEnv(**eval_kwargs)

    log.info(f"Action space:      {train_env.action_space}")
    log.info(f"Observation space: {train_env.observation_space.shape}")

    # --- Create or resume model ---
    tb_log_path = str(run_dir / "tb_logs")

    if args.resume and Path(args.resume).exists():
        model = MaskablePPO.load(
            args.resume,
            env=train_env,
            tensorboard_log=tb_log_path,
            learning_rate=args.lr,
            ent_coef=args.ent_coef,
            device="cpu",
        )
        log.info(f"Resumed from {args.resume}")
        reset_timesteps = False
    else:
        net_arch = [64, 64] if args.hybrid else [256, 256]
        model = MaskablePPO(
            "MlpPolicy",
            train_env,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=10,
            learning_rate=args.lr,
            gamma=args.gamma,
            ent_coef=args.ent_coef,
            policy_kwargs=dict(net_arch=net_arch),
            tensorboard_log=tb_log_path,
            seed=args.seed,
            verbose=0,
            device="cpu",
        )
        log.info("Created new model from scratch")
        reset_timesteps = True

    n_params = sum(p.numel() for p in model.policy.parameters())
    log.info(f"Policy parameters: {n_params:,}")

    # --- Callbacks ---
    best_path = run_dir / "best_model"
    best_path.mkdir(parents=True, exist_ok=True)

    checkpoint_path = run_dir / "checkpoints"
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    eval_cb = FPLEvalCallback(
        eval_env=eval_env,
        eval_freq=max(args.eval_freq // args.n_envs, 1),
        n_eval_episodes=args.n_eval_episodes,
        best_model_save_path=best_path,
        verbose=1,
        show_progress=False,  # no tqdm in batch jobs
    )
    episode_cb = FPLEpisodeLogCallback(verbose=0)
    checkpoint_cb = RollingCheckpointCallback(
        save_freq=max(args.checkpoint_freq // args.n_envs, 1),
        save_path=str(checkpoint_path),
        verbose=1,
    )
    metrics_cb = PPOMetricsCallback(log_freq=1)  # log every rollout

    # --- Train ---
    log.info("Starting training...")
    model.learn(
        total_timesteps=args.total_steps,
        callback=[eval_cb, episode_cb, checkpoint_cb, metrics_cb],
        progress_bar=False,  # no tqdm in batch
        reset_num_timesteps=reset_timesteps,
    )

    # --- Save final model ---
    final_path = run_dir / "final_model"
    final_path.mkdir(parents=True, exist_ok=True)
    model.save(str(final_path / "final_model"))

    log.info("=" * 60)
    log.info("Training complete!")
    log.info(f"End time:          {datetime.datetime.now()}")
    log.info(f"Best eval points:  {eval_cb.best_mean_points:.1f}")
    log.info(f"Training episodes: {episode_cb._episode_count}")
    log.info(f"Final model:       {final_path.resolve()}")
    log.info(f"Best model:        {best_path.resolve()}")
    log.info(f"Checkpoints:       {checkpoint_path.resolve()}")
    log.info("=" * 60)

    # --- Quick holdout eval ---
    log.info("Running holdout evaluation (2024-25)...")
    holdout_kwargs = dict(season="2024-25", data_dir=data_dir)
    if predictor_dir:
        holdout_kwargs["prediction_integrator"] = PredictionIntegrator.from_model(
            predictor_dir, pred_data_dir, "2024-25",
        )
    if args.hybrid:
        from fpl_rl.env.hybrid_env import HybridFPLEnv
        holdout_env = HybridFPLEnv(**holdout_kwargs)
    else:
        holdout_env = FPLEnv(**holdout_kwargs)

    results = []
    for ep in range(5):
        obs, _ = holdout_env.reset()
        done = False
        pts = 0.0
        while not done:
            masks = holdout_env.action_masks()
            action, _ = model.predict(obs, deterministic=True, action_masks=masks)
            obs, _, terminated, truncated, info = holdout_env.step(action)
            done = terminated or truncated
            pts = info.get("total_points", pts)
        results.append(pts)
        log.info(f"  Holdout ep {ep+1}: {pts:.0f} pts")

    log.info(f"Holdout mean: {np.mean(results):.1f} +/- {np.std(results):.1f}")

    train_env.close()
    eval_env.close()
    holdout_env.close()


if __name__ == "__main__":
    main()
