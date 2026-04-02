#!/usr/bin/env python3
"""Plot training progress dashboard from cluster log file.

Usage:
    ssh ... "grep 'Eval at step' log.err" | python scripts/plot_training.py
    python scripts/plot_training.py --log path/to/log.err
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


def parse_eval_lines(log_text: str) -> dict:
    pattern = r"(\d{2}:\d{2}:\d{2}) \[INFO\] Eval at step (\d+): ([\d.]+) \+/- ([\d.]+) pts \((\d+) episodes\)"
    matches = re.findall(pattern, log_text)
    return {
        "times": [m[0] for m in matches],
        "steps": [int(m[1]) for m in matches],
        "means": [float(m[2]) for m in matches],
        "stds": [float(m[3]) for m in matches],
    }


def parse_ppo_metrics(log_text: str) -> dict:
    """Parse PPO step metrics from PPOMetricsCallback output."""
    pattern = r"PPO step (\d+): (.+)"
    matches = re.findall(pattern, log_text)

    data: dict[str, list] = {"steps": []}
    metric_keys = [
        "loss", "policy_loss", "value_loss", "entropy",
        "approx_kl", "clip_frac", "expl_var", "mean_reward", "mean_ep_len",
    ]
    for k in metric_keys:
        data[k] = []

    for step_str, metrics_str in matches:
        data["steps"].append(int(step_str))
        kvs = {}
        for part in metrics_str.split(" | "):
            k, v = part.split("=")
            kvs[k.strip()] = float(v.strip())
        for k in metric_keys:
            data[k].append(kvs.get(k, float("nan")))

    return data


def time_to_minutes(times: list[str], base: str | None = None) -> list[float]:
    """Convert HH:MM:SS strings to minutes from first timestamp."""
    def to_sec(t):
        h, m, s = t.split(":")
        return int(h) * 3600 + int(m) * 60 + int(s)
    base_sec = to_sec(base or times[0])
    return [(to_sec(t) - base_sec) / 60 for t in times]


def plot_dashboard(data: dict, output_path: Path, total_target: int = 20_000_000,
                   ppo: dict | None = None):
    steps = np.array(data["steps"])
    means = np.array(data["means"])
    stds = np.array(data["stds"])
    steps_k = steps / 1000
    minutes = time_to_minutes(data["times"])

    has_ppo = ppo is not None and len(ppo.get("steps", [])) > 0
    n_rows = 5 if has_ppo else 3

    fig = plt.figure(figsize=(14, 4 * n_rows))
    fig.suptitle("FPL RL Training Dashboard (v3 — rules audit fixes)", fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(n_rows, 2, hspace=0.35, wspace=0.3)

    # --- 1. Mean eval points with std band ---
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(steps_k, means, "b-o", linewidth=2, markersize=6, label="Mean eval points", zorder=3)
    ax1.fill_between(steps_k, means - stds, means + stds, alpha=0.2, color="blue")
    rolling_best = np.maximum.accumulate(means)
    ax1.plot(steps_k, rolling_best, "r--", linewidth=1.5, alpha=0.7, label=f"Running best: {rolling_best[-1]:.0f}")
    ax1.axhline(y=means.mean(), color="gray", linestyle=":", alpha=0.5, label=f"Avg: {means.mean():.0f}")
    ax1.set_ylabel("Total Season Points")
    ax1.set_xlabel("Training Steps (K)")
    ax1.legend(loc="lower right", fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Eval Performance (2023-24 Season)", fontsize=11)

    # --- 2. Std deviation over time ---
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(steps_k, stds, "orange", linewidth=2, marker="s", markersize=5)
    ax2.fill_between(steps_k, 0, stds, alpha=0.15, color="orange")
    ax2.set_ylabel("Std Dev (points)")
    ax2.set_xlabel("Training Steps (K)")
    ax2.grid(True, alpha=0.3)
    ax2.set_title("Eval Variance", fontsize=11)

    # --- 3. Throughput ---
    ax3 = fig.add_subplot(gs[1, 1])
    if len(minutes) > 1:
        step_diffs = np.diff(steps)
        time_diffs = np.diff(minutes)
        throughput = step_diffs / time_diffs  # steps per minute
        mid_steps = (steps_k[:-1] + steps_k[1:]) / 2
        ax3.bar(mid_steps, throughput, width=(steps_k[1] - steps_k[0]) * 0.7,
                color="green", alpha=0.7, edgecolor="darkgreen")
        ax3.axhline(y=np.mean(throughput), color="red", linestyle="--",
                     label=f"Avg: {np.mean(throughput):.0f} steps/min")
        ax3.legend(fontsize=9)
    ax3.set_ylabel("Steps / minute")
    ax3.set_xlabel("Training Steps (K)")
    ax3.grid(True, alpha=0.3)
    ax3.set_title("Throughput", fontsize=11)

    # --- 4. Min/Max/Mean breakdown ---
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(steps_k, means + stds, "g--", alpha=0.6, label="Mean + 1σ (best runs)")
    ax4.plot(steps_k, means, "b-", linewidth=2, label="Mean")
    ax4.plot(steps_k, means - stds, "r--", alpha=0.6, label="Mean - 1σ (worst runs)")
    ax4.fill_between(steps_k, means - stds, means + stds, alpha=0.1, color="blue")
    ax4.set_ylabel("Points")
    ax4.set_xlabel("Training Steps (K)")
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    ax4.set_title("Performance Range", fontsize=11)

    # --- 5. Progress & ETA ---
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis("off")

    total_elapsed_min = minutes[-1] - minutes[0] if len(minutes) > 1 else 0
    latest_step = steps[-1]
    pct_done = latest_step / total_target * 100
    avg_rate = latest_step / max(total_elapsed_min, 1)
    remaining_steps = total_target - latest_step
    eta_min = remaining_steps / max(avg_rate, 1)
    eta_hrs = eta_min / 60

    stats_text = (
        f"Training Progress\n"
        f"{'─' * 32}\n"
        f"Steps completed:  {latest_step:>12,}\n"
        f"Target:           {total_target:>12,}\n"
        f"Progress:         {pct_done:>11.1f}%\n"
        f"{'─' * 32}\n"
        f"Elapsed:          {total_elapsed_min:>8.0f} min\n"
        f"Avg throughput:   {avg_rate:>8.0f} steps/min\n"
        f"ETA:              {eta_hrs:>8.1f} hours\n"
        f"{'─' * 32}\n"
        f"Best eval:        {max(means):>8.0f} pts\n"
        f"Latest eval:      {means[-1]:>8.0f} ± {stds[-1]:.0f} pts\n"
        f"Avg eval:         {means.mean():>8.0f} pts\n"
        f"{'─' * 32}\n"
        f"Eval episodes:    {len(means):>8d}\n"
        f"Parallel envs:           38\n"
        f"Max CPUs:                40\n"
    )
    ax5.text(0.05, 0.95, stats_text, transform=ax5.transAxes,
             fontsize=10, fontfamily="monospace", verticalalignment="top",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))

    # --- PPO metric plots (if available) ---
    if has_ppo:
        ppo_steps_k = np.array(ppo["steps"]) / 1000

        # Row 4: Loss plots
        ax6 = fig.add_subplot(gs[3, 0])
        policy_loss = np.array(ppo["policy_loss"])
        value_loss = np.array(ppo["value_loss"])
        valid = ~np.isnan(policy_loss)
        if valid.any():
            ax6.plot(ppo_steps_k[valid], policy_loss[valid], "b-", linewidth=1.5, alpha=0.8, label="Policy loss")
        valid_v = ~np.isnan(value_loss)
        if valid_v.any():
            ax6_r = ax6.twinx()
            ax6_r.plot(ppo_steps_k[valid_v], value_loss[valid_v], "r-", linewidth=1.5, alpha=0.8, label="Value loss")
            ax6_r.set_ylabel("Value Loss", color="red", fontsize=9)
            ax6_r.tick_params(axis="y", labelcolor="red")
        ax6.set_ylabel("Policy Loss", color="blue", fontsize=9)
        ax6.tick_params(axis="y", labelcolor="blue")
        ax6.set_xlabel("Training Steps (K)")
        ax6.grid(True, alpha=0.3)
        ax6.set_title("Policy & Value Loss", fontsize=11)
        lines1, labels1 = ax6.get_legend_handles_labels()
        if valid_v.any():
            lines2, labels2 = ax6_r.get_legend_handles_labels()
            ax6.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)
        else:
            ax6.legend(fontsize=8)

        ax7 = fig.add_subplot(gs[3, 1])
        entropy = np.array(ppo["entropy"])
        valid_e = ~np.isnan(entropy)
        if valid_e.any():
            ax7.plot(ppo_steps_k[valid_e], np.abs(entropy[valid_e]), "purple", linewidth=1.5)
        ax7.set_ylabel("Entropy (abs)")
        ax7.set_xlabel("Training Steps (K)")
        ax7.grid(True, alpha=0.3)
        ax7.set_title("Policy Entropy", fontsize=11)

        # Row 5: KL divergence and explained variance
        ax8 = fig.add_subplot(gs[4, 0])
        approx_kl = np.array(ppo["approx_kl"])
        valid_kl = ~np.isnan(approx_kl)
        if valid_kl.any():
            ax8.plot(ppo_steps_k[valid_kl], approx_kl[valid_kl], "teal", linewidth=1.5)
            ax8.axhline(y=0.02, color="red", linestyle="--", alpha=0.5, label="Target KL (0.02)")
            ax8.legend(fontsize=8)
        ax8.set_ylabel("Approx KL")
        ax8.set_xlabel("Training Steps (K)")
        ax8.grid(True, alpha=0.3)
        ax8.set_title("KL Divergence", fontsize=11)

        ax9 = fig.add_subplot(gs[4, 1])
        expl_var = np.array(ppo["expl_var"])
        valid_ev = ~np.isnan(expl_var)
        if valid_ev.any():
            ax9.plot(ppo_steps_k[valid_ev], expl_var[valid_ev], "green", linewidth=1.5)
            ax9.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5, label="Perfect (1.0)")
            ax9.axhline(y=0.0, color="red", linestyle=":", alpha=0.5, label="Random (0.0)")
            ax9.legend(fontsize=8)
            ax9.set_ylim(-0.5, 1.1)
        ax9.set_ylabel("Explained Variance")
        ax9.set_xlabel("Training Steps (K)")
        ax9.grid(True, alpha=0.3)
        ax9.set_title("Value Function Quality", fontsize=11)

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=str, default=None)
    parser.add_argument("--output", type=str, default="runs/training_progress.png")
    parser.add_argument("--total-steps", type=int, default=20_000_000)
    args = parser.parse_args()

    if args.log:
        log_text = Path(args.log).read_text()
    else:
        log_text = sys.stdin.read()

    data = parse_eval_lines(log_text)

    if not data["steps"]:
        print("No eval data found in log.")
        sys.exit(1)

    print(f"Found {len(data['steps'])} eval points")
    best_idx = data["means"].index(max(data["means"]))
    print(f"Best: {max(data['means']):.1f} pts at step {data['steps'][best_idx]:,}")
    print(f"Latest: {data['means'][-1]:.1f} ± {data['stds'][-1]:.1f} pts")

    ppo = parse_ppo_metrics(log_text)
    if ppo["steps"]:
        print(f"Found {len(ppo['steps'])} PPO rollout metrics")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    plot_dashboard(data, output, total_target=args.total_steps, ppo=ppo if ppo["steps"] else None)


if __name__ == "__main__":
    main()
