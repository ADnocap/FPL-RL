#!/usr/bin/env python3
"""Evaluate shifted models with UNSHIFTED features (matching training distribution).

During RL training, PredictionIntegrator.from_model() was called without
shifting fpl_xp, even for the shifted LightGBM model. So the RL agent
was trained on mismatched predictions. This script evaluates them with
the same mismatch to see their true training-distribution performance.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("holdout_v2")

HOLDOUT = "2024-25"
N_EPISODES = 5


def run_episodes(model, env, n_episodes):
    results = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        pts = 0.0
        while not done:
            masks = env.action_masks()
            action, _ = model.predict(obs, deterministic=True, action_masks=masks)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            pts = info.get("total_points", pts)
        results.append(pts)
        log.info("  Episode %d: %.0f pts", ep + 1, pts)
    return results


def fmt(pts):
    return f"{np.mean(pts):7.1f} +/- {np.std(pts):5.1f}  (min={min(pts):.0f}, max={max(pts):.0f})"


def main():
    from sb3_contrib import MaskablePPO
    from fpl_rl.data.downloader import DEFAULT_DATA_DIR
    from fpl_rl.env.hybrid_env import HybridFPLEnv
    from fpl_rl.prediction.integration import PredictionIntegrator

    data_dir = DEFAULT_DATA_DIR
    pred_data_dir = data_dir.parent if data_dir.name == "raw" else data_dir
    runs_dir = Path("runs/holdout_eval")

    # Shifted models with UNSHIFTED features (matches training distribution)
    models = [
        ("Shifted BEST (as-trained)",   runs_dir / "shifted_best.zip",   "models/ab_test_shifted"),
        ("Shifted LATEST (as-trained)", runs_dir / "shifted_latest.zip", "models/ab_test_shifted"),
    ]

    all_results = {}
    for label, model_path, predictor_dir in models:
        log.info("=" * 65)
        log.info("Evaluating: %s", label)

        # Use standard from_model (UNSHIFTED features — same as during training)
        integrator = PredictionIntegrator.from_model(
            Path(predictor_dir), pred_data_dir, HOLDOUT,
        )
        env = HybridFPLEnv(
            season=HOLDOUT, data_dir=data_dir,
            prediction_integrator=integrator,
        )
        model = MaskablePPO.load(str(model_path), env=env)
        results = run_episodes(model, env, N_EPISODES)
        all_results[label] = results
        env.close()

    print("\n" + "=" * 75)
    print(f"  Holdout (as-trained) on {HOLDOUT} ({N_EPISODES} episodes)")
    print("=" * 75)
    for label, pts in all_results.items():
        print(f"  {label:<35} {fmt(pts)}")
    print("=" * 75)


if __name__ == "__main__":
    main()
