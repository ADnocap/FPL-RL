#!/usr/bin/env python3
"""Evaluate best vs latest unshifted model on 2024-25 holdout.

Runs both models for N episodes, collects per-GW points, and generates
comparison charts.

Usage:
    python scripts/evaluate_best_vs_latest.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("best_vs_latest")

HOLDOUT = "2024-25"
N_EPISODES = 10


def run_episodes_detailed(model, env, n_episodes):
    """Run episodes and collect per-GW breakdown."""
    episodes = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        gw_data = []
        while not done:
            masks = env.action_masks()
            action, _ = model.predict(obs, deterministic=True, action_masks=masks)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            gw_data.append({
                "gw": info.get("gw", 0),
                "gw_points": info.get("gw_points", 0),
                "net_points": info.get("net_points", 0),
                "hit_cost": info.get("hit_cost", 0),
                "num_transfers": info.get("num_transfers", 0),
                "total_points": info.get("total_points", 0),
                "chip": str(info.get("active_chip", "none")),
            })
        total = info.get("total_points", 0)
        episodes.append({"total": total, "gws": gw_data})
        log.info("  Episode %d: %.0f pts", ep + 1, total)
    return episodes


def run_noop_detailed(env, n_episodes):
    """No-op baseline with per-GW data."""
    episodes = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        gw_data = []
        while not done:
            action = np.zeros(env.action_space.shape, dtype=int)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            gw_data.append({
                "gw": info.get("gw", 0),
                "gw_points": info.get("gw_points", 0),
                "net_points": info.get("net_points", 0),
                "hit_cost": info.get("hit_cost", 0),
                "total_points": info.get("total_points", 0),
            })
        total = info.get("total_points", 0)
        episodes.append({"total": total, "gws": gw_data})
    return episodes


def plot_comparison(results, output_path):
    """Generate comparison charts."""
    fig = plt.figure(figsize=(14, 14))
    gs = gridspec.GridSpec(3, 2, hspace=0.35, wspace=0.3)
    fig.suptitle(f"Holdout Evaluation ({HOLDOUT}): Best vs Latest Model",
                 fontsize=14, fontweight="bold")

    colors = {"Best (step 1.5M)": "blue", "Latest (step 2.5M)": "red", "No-op": "gray"}

    # 1. Total points bar chart
    ax1 = fig.add_subplot(gs[0, 0])
    labels = list(results.keys())
    means = [np.mean([e["total"] for e in results[l]]) for l in labels]
    stds = [np.std([e["total"] for e in results[l]]) for l in labels]
    bars = ax1.bar(labels, means, yerr=stds, capsize=5,
                   color=[colors.get(l, "blue") for l in labels], alpha=0.8)
    ax1.axhline(y=2810, color="green", ls=":", alpha=0.7, label="Best human")
    for bar, m, s in zip(bars, means, stds):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + s + 20,
                 f"{m:.0f}", ha="center", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Total Points")
    ax1.legend(fontsize=8)
    ax1.set_title("Total Points Comparison")
    ax1.grid(True, alpha=0.3, axis="y")

    # 2. Episode distribution (box plot)
    ax2 = fig.add_subplot(gs[0, 1])
    all_totals = [[e["total"] for e in results[l]] for l in labels]
    bp = ax2.boxplot(all_totals, labels=labels, patch_artist=True)
    for patch, label in zip(bp["boxes"], labels):
        patch.set_facecolor(colors.get(label, "blue"))
        patch.set_alpha(0.5)
    ax2.axhline(y=2810, color="green", ls=":", alpha=0.7, label="Best human")
    ax2.set_ylabel("Total Points")
    ax2.legend(fontsize=8)
    ax2.set_title("Episode Distribution")
    ax2.grid(True, alpha=0.3, axis="y")

    # 3. Cumulative points over GWs (mean across episodes)
    ax3 = fig.add_subplot(gs[1, :])
    for label in labels:
        eps = results[label]
        # Build cumulative per-GW matrix
        n_gws = max(len(e["gws"]) for e in eps)
        cum_matrix = np.zeros((len(eps), n_gws))
        for i, ep in enumerate(eps):
            cum = 0
            for j, gw in enumerate(ep["gws"]):
                cum += gw["net_points"]
                cum_matrix[i, j] = cum
        gws = np.arange(1, n_gws + 1)
        mean_cum = cum_matrix.mean(axis=0)
        std_cum = cum_matrix.std(axis=0)
        c = colors.get(label, "blue")
        ax3.plot(gws, mean_cum, color=c, lw=2, label=label)
        ax3.fill_between(gws, mean_cum - std_cum, mean_cum + std_cum,
                         color=c, alpha=0.1)
    ax3.set_xlabel("Gameweek")
    ax3.set_ylabel("Cumulative Net Points")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_title("Cumulative Points Over Season")

    # 4. Per-GW net points (mean)
    ax4 = fig.add_subplot(gs[2, 0])
    for label in ["Best (step 1.5M)", "Latest (step 2.5M)"]:
        if label not in results:
            continue
        eps = results[label]
        n_gws = max(len(e["gws"]) for e in eps)
        gw_pts = np.zeros((len(eps), n_gws))
        for i, ep in enumerate(eps):
            for j, gw in enumerate(ep["gws"]):
                gw_pts[i, j] = gw["net_points"]
        gws = np.arange(1, n_gws + 1)
        c = colors.get(label, "blue")
        ax4.plot(gws, gw_pts.mean(axis=0), color=c, lw=1.5, alpha=0.8, label=label)
    ax4.set_xlabel("Gameweek")
    ax4.set_ylabel("Net Points")
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    ax4.set_title("Per-GW Net Points (Mean)")

    # 5. Transfer hits comparison
    ax5 = fig.add_subplot(gs[2, 1])
    for label in ["Best (step 1.5M)", "Latest (step 2.5M)"]:
        if label not in results:
            continue
        eps = results[label]
        n_gws = max(len(e["gws"]) for e in eps)
        hits = np.zeros((len(eps), n_gws))
        for i, ep in enumerate(eps):
            for j, gw in enumerate(ep["gws"]):
                hits[i, j] = gw.get("hit_cost", 0)
        gws = np.arange(1, n_gws + 1)
        c = colors.get(label, "blue")
        ax5.plot(gws, hits.mean(axis=0), color=c, lw=1.5, alpha=0.8, label=label)
    total_hits = {}
    for label in ["Best (step 1.5M)", "Latest (step 2.5M)"]:
        if label not in results:
            continue
        total = np.mean([sum(gw.get("hit_cost", 0) for gw in e["gws"]) for e in results[label]])
        total_hits[label] = total
    hit_text = "  |  ".join(f"{l}: {h:.0f}" for l, h in total_hits.items())
    ax5.set_xlabel("Gameweek")
    ax5.set_ylabel("Hit Cost")
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)
    ax5.set_title(f"Transfer Hits (total: {hit_text})")

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")


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

    # Build integrator once (same predictor for both)
    log.info("Building prediction integrator...")
    integrator = PredictionIntegrator.from_model(
        Path("models/ab_test_unshifted"), pred_data_dir, HOLDOUT,
    )

    results = {}

    # No-op baseline
    log.info("=" * 65)
    log.info("Running no-op baseline...")
    env = HybridFPLEnv(season=HOLDOUT, data_dir=data_dir, prediction_integrator=integrator)
    results["No-op"] = run_noop_detailed(env, N_EPISODES)
    env.close()

    # Best model (step 1.5M)
    log.info("=" * 65)
    log.info("Evaluating: Best (step 1.5M)")
    env = HybridFPLEnv(season=HOLDOUT, data_dir=data_dir, prediction_integrator=integrator)
    model = MaskablePPO.load(str(runs_dir / "unshifted_best.zip"), env=env)
    results["Best (step 1.5M)"] = run_episodes_detailed(model, env, N_EPISODES)
    env.close()

    # Latest model (step 2.5M)
    log.info("=" * 65)
    log.info("Evaluating: Latest (step 2.5M)")
    env = HybridFPLEnv(season=HOLDOUT, data_dir=data_dir, prediction_integrator=integrator)
    model = MaskablePPO.load(str(runs_dir / "unshifted_latest.zip"), env=env)
    results["Latest (step 2.5M)"] = run_episodes_detailed(model, env, N_EPISODES)
    env.close()

    # Print table
    print("\n" + "=" * 75)
    print(f"  Holdout Evaluation on {HOLDOUT} ({N_EPISODES} episodes each)")
    print("=" * 75)
    for label in results:
        totals = [e["total"] for e in results[label]]
        print(f"  {label:<25} {fmt(totals)}")
    print("-" * 75)
    print("  Reference: Best human ~2810, Oracle 4756")
    print("=" * 75)

    # Plot
    plot_comparison(results, "runs/best_vs_latest_holdout.png")


if __name__ == "__main__":
    main()
