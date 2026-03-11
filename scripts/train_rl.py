#!/usr/bin/env python
"""Train MaskablePPO on FPL across multiple historical seasons.

Usage:
    python scripts/train_rl.py                                        # defaults
    python scripts/train_rl.py --total-timesteps 500000               # longer
    python scripts/train_rl.py --total-timesteps 200 --eval-freq 100  # smoke test
    python scripts/train_rl.py --no-predictor                         # no LightGBM
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure the project root is on sys.path when running as a script
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train MaskablePPO on FPL")

    # Seasons
    p.add_argument(
        "--train-seasons", nargs="+",
        default=["2016-17", "2017-18", "2018-19", "2019-20",
                 "2020-21", "2021-22", "2022-23"],
        help="Training seasons (default: 2016-17 to 2022-23)",
    )
    p.add_argument("--eval-season", default="2023-24",
                   help="Evaluation season for model selection")
    p.add_argument("--holdout-season", default="2024-25",
                   help="Holdout season for final metric")

    # Data / model
    p.add_argument("--data-dir", type=Path, default=None,
                   help="Data directory (default: package default)")
    p.add_argument("--predictor-model-dir", type=Path, default=None,
                   help="LightGBM model directory for point predictions")
    p.add_argument("--no-predictor", action="store_true",
                   help="Skip loading the LightGBM predictor")

    # PPO hyperparameters
    p.add_argument("--total-timesteps", type=int, default=200_000)
    p.add_argument("--n-steps", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--n-epochs", type=int, default=10)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--ent-coef", type=float, default=0.02)
    p.add_argument("--net-arch", nargs="+", type=int, default=[256, 256])

    # Eval / logging
    p.add_argument("--eval-freq", type=int, default=10_000,
                   help="Evaluate every N timesteps")
    p.add_argument("--n-eval-episodes", type=int, default=3)
    p.add_argument("--holdout-episodes", type=int, default=5)
    p.add_argument("--shuffle-seasons", action="store_true",
                   help="Randomly sample training seasons instead of cycling")

    # Resume
    p.add_argument("--resume", type=Path, default=None,
                   help="Path to saved model to resume training from")

    # Output
    p.add_argument("--run-dir", type=Path, default=Path("runs/fpl_ppo"),
                   help="Directory for logs and saved models")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verbose", type=int, default=1)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    log = logging.getLogger("train_rl")

    from sb3_contrib import MaskablePPO
    from fpl_rl.data.downloader import DEFAULT_DATA_DIR
    from fpl_rl.env.fpl_env import FPLEnv
    from fpl_rl.training.callbacks import FPLEpisodeLogCallback, FPLEvalCallback
    from fpl_rl.training.multi_season_env import MultiSeasonFPLEnv

    data_dir = args.data_dir or DEFAULT_DATA_DIR
    predictor_dir = None if args.no_predictor else args.predictor_model_dir
    # Prediction pipeline expects parent dir (with id_maps/, understat/, etc.)
    pred_data_dir = data_dir.parent if data_dir.name == "raw" else data_dir

    # --- Build environments ---
    log.info("Building training env with seasons: %s", args.train_seasons)
    train_env = MultiSeasonFPLEnv(
        seasons=args.train_seasons,
        data_dir=data_dir,
        predictor_model_dir=predictor_dir,
        shuffle=args.shuffle_seasons,
    )

    log.info("Building eval env for season: %s", args.eval_season)
    eval_kwargs = dict(season=args.eval_season, data_dir=data_dir)
    if predictor_dir is not None:
        from fpl_rl.prediction.integration import PredictionIntegrator
        eval_kwargs["prediction_integrator"] = PredictionIntegrator.from_model(
            predictor_dir, pred_data_dir, args.eval_season,
        )
    eval_env = FPLEnv(**eval_kwargs)

    # --- Build model ---
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    tb_log_path = str(run_dir / "tb_logs")

    if args.resume:
        log.info("Resuming from %s ...", args.resume)
        model = MaskablePPO.load(
            str(args.resume),
            env=train_env,
            tensorboard_log=tb_log_path,
            learning_rate=args.learning_rate,
            ent_coef=args.ent_coef,
            device="cpu",
        )
    else:
        log.info("Creating MaskablePPO ...")
        model = MaskablePPO(
            "MlpPolicy",
            train_env,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            ent_coef=args.ent_coef,
            policy_kwargs=dict(net_arch=args.net_arch),
            tensorboard_log=tb_log_path,
            seed=args.seed,
            verbose=args.verbose,
        )

    # --- Callbacks ---
    best_model_path = run_dir / "best_model"
    best_model_path.mkdir(parents=True, exist_ok=True)

    eval_cb = FPLEvalCallback(
        eval_env=eval_env,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        best_model_save_path=best_model_path,
        verbose=args.verbose,
    )
    episode_cb = FPLEpisodeLogCallback(verbose=args.verbose)

    # --- Train ---
    log.info("Starting training for %d timesteps ...", args.total_timesteps)
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[eval_cb, episode_cb],
        reset_num_timesteps=args.resume is None,
    )

    # Save final model
    final_path = run_dir / "final_model"
    final_path.mkdir(parents=True, exist_ok=True)
    model.save(str(final_path / "final_model"))
    log.info("Final model saved to %s", final_path)

    # --- Holdout evaluation ---
    log.info("Running holdout evaluation on %s (%d episodes) ...",
             args.holdout_season, args.holdout_episodes)

    holdout_kwargs = dict(season=args.holdout_season, data_dir=data_dir)
    if predictor_dir is not None:
        holdout_kwargs["prediction_integrator"] = PredictionIntegrator.from_model(
            predictor_dir, pred_data_dir, args.holdout_season,
        )
    holdout_env = FPLEnv(**holdout_kwargs)

    holdout_points: list[float] = []
    for ep in range(args.holdout_episodes):
        obs, info = holdout_env.reset()
        done = False
        total_pts = 0.0
        while not done:
            masks = holdout_env.action_masks()
            action, _ = model.predict(obs, deterministic=True, action_masks=masks)
            obs, reward, terminated, truncated, info = holdout_env.step(action)
            done = terminated or truncated
            total_pts = info.get("total_points", total_pts)
        holdout_points.append(total_pts)
        log.info("  Holdout ep %d: %.0f pts", ep + 1, total_pts)

    mean_pts = sum(holdout_points) / len(holdout_points)
    log.info("Holdout mean: %.1f pts (std: %.1f)",
             mean_pts,
             (sum((p - mean_pts) ** 2 for p in holdout_points) / len(holdout_points)) ** 0.5)

    train_env.close()
    eval_env.close()
    holdout_env.close()


if __name__ == "__main__":
    main()
