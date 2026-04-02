#!/usr/bin/env python3
"""Evaluate a trained MaskablePPO model with detailed per-GW breakdown.

Usage:
    python scripts/eval_model.py --model runs/best_model.zip --season 2023-24
    python scripts/eval_model.py --model runs/best_model.zip --season 2024-25 --episodes 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate FPL RL model")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--season", type=str, default="2023-24")
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--predictor-dir", type=str, default=None)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def run_episode(model, env, deterministic=True):
    """Run one full episode, collecting per-GW stats."""
    obs, info = env.reset()
    done = False
    gw_log = []

    while not done:
        masks = env.action_masks()
        action, _ = model.predict(obs, deterministic=deterministic, action_masks=masks)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if not info.get("preseason", False):
            gw_log.append({
                "gw": info["gw"],
                "gw_points": info["gw_points"],
                "net_points": info["net_points"],
                "hit_cost": info["hit_cost"],
                "num_transfers": info.get("num_transfers", 0),
                "captain_points": info["captain_points"],
                "bench_points": info["bench_points"],
                "auto_subs": len(info.get("auto_subs", [])),
                "captain_failover": info.get("captain_failover", False),
                "chip": info.get("active_chip"),
                "total_points": info["total_points"],
            })

    return gw_log


def print_episode_summary(ep_num, gw_log):
    """Print detailed breakdown for one episode."""
    total = gw_log[-1]["total_points"]
    total_hits = sum(g["hit_cost"] for g in gw_log)
    total_transfers = sum(g["num_transfers"] for g in gw_log)
    total_captain = sum(g["captain_points"] for g in gw_log)
    total_bench = sum(g["bench_points"] for g in gw_log)
    total_auto_subs = sum(g["auto_subs"] for g in gw_log)
    failovers = sum(1 for g in gw_log if g["captain_failover"])
    chips_used = [(g["gw"], g["chip"]) for g in gw_log if g["chip"]]
    gw_points = [g["gw_points"] for g in gw_log]
    net_points = [g["net_points"] for g in gw_log]

    print(f"\n{'='*60}")
    print(f"Episode {ep_num}: {total} total points")
    print(f"{'='*60}")
    print(f"  Gross points:     {sum(gw_points):>6}")
    print(f"  Total transfers:  {total_transfers:>6} (hits: {total_hits}, extra: {total_hits//4})")
    print(f"  Net points:       {sum(net_points):>6}")
    print(f"  Captain bonus:    {total_captain:>6}")
    print(f"  Bench points:     {total_bench:>6}")
    print(f"  Auto-subs:        {total_auto_subs:>6}")
    print(f"  Captain failovers:{failovers:>6}")
    print(f"  Chips used:       {chips_used if chips_used else 'none'}")
    print(f"  Avg GW points:    {np.mean(gw_points):>6.1f}")
    print(f"  Best GW:          {max(gw_points):>6} (GW{gw_log[gw_points.index(max(gw_points))]['gw']})")
    print(f"  Worst GW:         {min(gw_points):>6} (GW{gw_log[gw_points.index(min(gw_points))]['gw']})")

    # Per-GW table
    print(f"\n  {'GW':>3} {'Gross':>6} {'Hits':>5} {'Net':>6} {'Capt':>5} {'Xfer':>5} {'Total':>6}")
    print(f"  {'---':>3} {'-----':>6} {'----':>5} {'-----':>6} {'----':>5} {'----':>5} {'-----':>6}")
    for g in gw_log:
        chip_flag = f" [{g['chip'][:2].upper()}]" if g["chip"] else ""
        print(f"  {g['gw']:>3} {g['gw_points']:>6} {g['hit_cost']:>5} "
              f"{g['net_points']:>6} {g['captain_points']:>5} {g['num_transfers']:>5} "
              f"{g['total_points']:>6}{chip_flag}")


def main():
    args = parse_args()

    from sb3_contrib import MaskablePPO
    from fpl_rl.data.downloader import DEFAULT_DATA_DIR
    from fpl_rl.env.fpl_env import FPLEnv

    data_dir = Path(args.data_dir) if args.data_dir else DEFAULT_DATA_DIR
    pred_data_dir = data_dir.parent if data_dir.name == "raw" else data_dir

    # Load model
    print(f"Loading model: {args.model}")
    model = MaskablePPO.load(args.model, device="cpu")

    # Build env
    env_kwargs = dict(season=args.season, data_dir=data_dir)
    if args.predictor_dir and Path(args.predictor_dir).exists():
        from fpl_rl.prediction.integration import PredictionIntegrator
        env_kwargs["prediction_integrator"] = PredictionIntegrator.from_model(
            Path(args.predictor_dir), pred_data_dir, args.season,
        )
    env = FPLEnv(**env_kwargs)

    print(f"Season: {args.season}")
    print(f"Episodes: {args.episodes}")
    print(f"Action space: {env.action_space}")

    # Run episodes
    all_totals = []
    all_gw_logs = []

    for ep in range(args.episodes):
        gw_log = run_episode(model, env, deterministic=True)
        total = gw_log[-1]["total_points"]
        all_totals.append(total)
        all_gw_logs.append(gw_log)
        print_episode_summary(ep + 1, gw_log)

    # Overall summary
    print(f"\n{'='*60}")
    print(f"OVERALL SUMMARY ({args.episodes} episodes, {args.season})")
    print(f"{'='*60}")
    print(f"  Mean:   {np.mean(all_totals):.1f} pts")
    print(f"  Std:    {np.std(all_totals):.1f} pts")
    print(f"  Best:   {max(all_totals):.0f} pts")
    print(f"  Worst:  {min(all_totals):.0f} pts")
    print(f"  All:    {[int(t) for t in all_totals]}")

    # Average per-GW curve
    if all_gw_logs:
        max_gws = max(len(log) for log in all_gw_logs)
        avg_cumulative = []
        for gw_idx in range(max_gws):
            vals = [log[gw_idx]["total_points"] for log in all_gw_logs if gw_idx < len(log)]
            avg_cumulative.append(np.mean(vals))
        print(f"\n  Avg cumulative points by GW10/GW20/GW30/GW38:")
        for gw_check in [9, 19, 29, min(max_gws - 1, 37)]:
            if gw_check < len(avg_cumulative):
                print(f"    GW{gw_check+1}: {avg_cumulative[gw_check]:.0f}")

    env.close()


if __name__ == "__main__":
    main()
