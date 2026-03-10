#!/usr/bin/env python
"""Load a saved MaskablePPO model and evaluate on a given season.

Compares against no-op and random baselines.

Usage:
    python scripts/evaluate_rl.py runs/fpl_ppo/final_model/final_model
    python scripts/evaluate_rl.py runs/fpl_ppo/best_model/best_model --season 2024-25
    python scripts/evaluate_rl.py runs/fpl_ppo/final_model/final_model -n 10
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a saved MaskablePPO model")
    p.add_argument("model_path", type=Path,
                   help="Path to saved MaskablePPO model (without .zip)")
    p.add_argument("--season", default="2024-25",
                   help="Season to evaluate on")
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--predictor-model-dir", type=Path, default=None)
    p.add_argument("--no-predictor", action="store_true")
    p.add_argument("-n", "--episodes", type=int, default=5,
                   help="Number of evaluation episodes")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def run_agent_episodes(model, env, n_episodes: int) -> list[float]:
    """Run the trained agent for n full episodes, return total points."""
    results = []
    for _ in range(n_episodes):
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
    return results


def run_noop_episodes(env, n_episodes: int) -> list[float]:
    """Run a no-op baseline (always action 0 = no transfer, no chip)."""
    import numpy as np
    results = []
    for _ in range(n_episodes):
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


def run_random_episodes(env, n_episodes: int, seed: int = 42) -> list[float]:
    """Run a random masked baseline."""
    import numpy as np
    rng = np.random.default_rng(seed)
    results = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        pts = 0.0
        while not done:
            masks = env.action_masks()
            # Sample random valid action per dimension
            action = np.zeros(env.action_space.shape, dtype=int)
            offset = 0
            for dim_i, n_choices in enumerate(env.action_space.nvec):
                dim_mask = masks[offset:offset + n_choices]
                valid = np.where(dim_mask)[0]
                if len(valid) > 0:
                    action[dim_i] = rng.choice(valid)
                offset += n_choices
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            pts = info.get("total_points", pts)
        results.append(pts)
    return results


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s")
    log = logging.getLogger("evaluate_rl")

    import numpy as np
    from sb3_contrib import MaskablePPO
    from fpl_rl.data.downloader import DEFAULT_DATA_DIR
    from fpl_rl.env.fpl_env import FPLEnv

    data_dir = args.data_dir or DEFAULT_DATA_DIR
    predictor_dir = None if args.no_predictor else args.predictor_model_dir
    pred_data_dir = data_dir.parent if data_dir.name == "raw" else data_dir

    # Build env
    env_kwargs = dict(season=args.season, data_dir=data_dir)
    if predictor_dir is not None:
        from fpl_rl.prediction.integration import PredictionIntegrator
        env_kwargs["prediction_integrator"] = PredictionIntegrator.from_model(
            predictor_dir, pred_data_dir, args.season,
        )
    env = FPLEnv(**env_kwargs)

    # Load model
    log.info("Loading model from %s", args.model_path)
    model = MaskablePPO.load(str(args.model_path), env=env)

    # Run evaluations
    log.info("Running %d episodes for each baseline ...", args.episodes)

    agent_pts = run_agent_episodes(model, env, args.episodes)
    noop_pts = run_noop_episodes(env, args.episodes)
    random_pts = run_random_episodes(env, args.episodes, seed=args.seed)

    env.close()

    # Print comparison table
    def fmt(pts: list[float]) -> str:
        return f"{np.mean(pts):7.1f} +/- {np.std(pts):5.1f}  (min={min(pts):.0f}, max={max(pts):.0f})"

    print("\n" + "=" * 65)
    print(f"  Evaluation on season {args.season} ({args.episodes} episodes)")
    print("=" * 65)
    print(f"  {'Agent (PPO):':<20} {fmt(agent_pts)}")
    print(f"  {'No-op baseline:':<20} {fmt(noop_pts)}")
    print(f"  {'Random baseline:':<20} {fmt(random_pts)}")
    print("=" * 65)


if __name__ == "__main__":
    main()
