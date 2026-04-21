#!/usr/bin/env python3
"""Evaluate shifted vs unshifted hybrid RL models on 2024-25 holdout season.

For each variant (shifted/unshifted), evaluates both the best and latest model.
The shifted variant requires fpl_xp to be shifted by 1 GW before prediction.

Usage:
    python scripts/evaluate_holdout_ab.py
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
log = logging.getLogger("holdout_eval")

HOLDOUT = "2024-25"
N_EPISODES = 5


def build_shifted_integrator(model_dir: Path, data_dir: Path, season: str):
    """Build a PredictionIntegrator with fpl_xp shifted by 1 GW.

    The shifted predictor was trained on features where fpl_xp was
    shifted: groupby(season, element).shift(1). We must reproduce
    this at inference time.
    """
    from fpl_rl.prediction.model import PointPredictor
    from fpl_rl.prediction.id_resolver import IDResolver
    from fpl_rl.prediction.feature_pipeline import FeaturePipeline
    from fpl_rl.prediction.integration import PredictionIntegrator

    predictor = PointPredictor.load(model_dir)
    id_resolver = IDResolver(data_dir)

    pipeline = FeaturePipeline(data_dir, id_resolver, [season])
    df = pipeline.build()

    if df.empty:
        log.warning("No feature data for season %s", season)
        return PredictionIntegrator({})

    # Apply the same shift that was used during training
    df = df.sort_values(["season", "element", "GW"])
    df["fpl_xp"] = df.groupby(["season", "element"])["fpl_xp"].shift(1)

    preds = predictor.predict(df)

    predictions: dict[tuple[int, int], float] = {}
    for i, (_, row) in enumerate(df.iterrows()):
        eid = id_resolver.element_id_from_code(int(row["code"]), season)
        if eid is not None:
            predictions[(eid, int(row["GW"]))] = float(preds[i])

    log.info("Shifted integrator: %d predictions for %s", len(predictions), season)
    return PredictionIntegrator(predictions)


def run_episodes(model, env, n_episodes: int) -> list[float]:
    """Run agent for n full episodes, return total points per episode."""
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


def run_noop_episodes(env, n_episodes: int) -> list[float]:
    """No-op baseline."""
    results = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        pts = 0.0
        while not done:
            action = np.zeros(env.action_space.shape, dtype=int)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            pts = info.get("total_points", pts)
        results.append(pts)
    return results


def fmt(pts: list[float]) -> str:
    return f"{np.mean(pts):7.1f} +/- {np.std(pts):5.1f}  (min={min(pts):.0f}, max={max(pts):.0f})"


def main():
    from sb3_contrib import MaskablePPO
    from fpl_rl.data.downloader import DEFAULT_DATA_DIR
    from fpl_rl.env.hybrid_env import HybridFPLEnv
    from fpl_rl.prediction.integration import PredictionIntegrator

    data_dir = DEFAULT_DATA_DIR
    pred_data_dir = data_dir.parent if data_dir.name == "raw" else data_dir

    runs_dir = Path("runs/holdout_eval")

    # Model configs: (label, model_path, predictor_dir, shifted)
    models = [
        ("Unshifted BEST",   runs_dir / "unshifted_best.zip",   "models/ab_test_unshifted", False),
        ("Unshifted LATEST", runs_dir / "unshifted_latest.zip",  "models/ab_test_unshifted", False),
        ("Shifted BEST",     runs_dir / "shifted_best.zip",      "models/ab_test_shifted",   True),
        ("Shifted LATEST",   runs_dir / "shifted_latest.zip",    "models/ab_test_shifted",   True),
    ]

    all_results: dict[str, list[float]] = {}

    # Run no-op baseline once (doesn't depend on model)
    log.info("=" * 65)
    log.info("Building no-op baseline env...")
    noop_integrator = PredictionIntegrator.from_model(
        Path("models/ab_test_unshifted"), pred_data_dir, HOLDOUT,
    )
    noop_env = HybridFPLEnv(
        season=HOLDOUT, data_dir=data_dir,
        prediction_integrator=noop_integrator,
    )
    log.info("Running no-op baseline...")
    noop_pts = run_noop_episodes(noop_env, N_EPISODES)
    all_results["No-op baseline"] = noop_pts
    noop_env.close()

    # Evaluate each model
    for label, model_path, predictor_dir, shifted in models:
        log.info("=" * 65)
        log.info("Evaluating: %s", label)
        log.info("  Model: %s", model_path)
        log.info("  Predictor: %s (shifted=%s)", predictor_dir, shifted)

        if not model_path.exists():
            log.error("  Model file not found: %s", model_path)
            continue

        # Build integrator
        if shifted:
            integrator = build_shifted_integrator(
                Path(predictor_dir), pred_data_dir, HOLDOUT,
            )
        else:
            integrator = PredictionIntegrator.from_model(
                Path(predictor_dir), pred_data_dir, HOLDOUT,
            )

        # Build env
        env = HybridFPLEnv(
            season=HOLDOUT, data_dir=data_dir,
            prediction_integrator=integrator,
        )

        # Load model
        model = MaskablePPO.load(str(model_path), env=env)

        # Run episodes
        results = run_episodes(model, env, N_EPISODES)
        all_results[label] = results
        env.close()

    # Print comparison table
    print("\n" + "=" * 75)
    print(f"  Holdout Evaluation on {HOLDOUT} ({N_EPISODES} episodes each)")
    print("=" * 75)
    for label, pts in all_results.items():
        print(f"  {label:<25} {fmt(pts)}")
    print("-" * 75)
    print("  Reference: Best human ~2810, Oracle 4756")
    print("=" * 75)


if __name__ == "__main__":
    main()
