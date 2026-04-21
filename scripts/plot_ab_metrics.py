#!/usr/bin/env python3
"""Plot training metrics for the hybrid RL model."""
import sys
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def parse_all(text):
    # Eval
    eval_pattern = r"Eval at step (\d+): ([\d.]+) \+/- ([\d.]+)"
    evals = re.findall(eval_pattern, text)
    e_steps = np.array([int(m[0]) for m in evals])
    e_means = np.array([float(m[1]) for m in evals])
    e_stds = np.array([float(m[2]) for m in evals])

    # PPO
    ppo_pattern = r"PPO step (\d+): (.+)"
    ppo_matches = re.findall(ppo_pattern, text)
    keys = ["loss", "policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac", "expl_var"]
    ppo = {k: [] for k in ["steps"] + keys}
    for step_str, metrics_str in ppo_matches:
        ppo["steps"].append(int(step_str))
        kvs = {}
        for part in metrics_str.split(" | "):
            k, v = part.split("=")
            kvs[k.strip()] = float(v.strip())
        for k in keys:
            ppo[k].append(kvs.get(k, float("nan")))
    ppo = {k: np.array(v) for k, v in ppo.items()}

    # Handle resumed runs: if steps go backwards, keep only the latest
    # contiguous segment at each point (trim earlier data at resume boundary)
    def dedup_monotonic(steps, *arrays):
        if len(steps) == 0:
            return (steps,) + arrays
        breaks = []
        for i in range(1, len(steps)):
            if steps[i] <= steps[i - 1]:
                breaks.append(i)
        if not breaks:
            return (steps,) + arrays
        result_mask = np.ones(len(steps), dtype=bool)
        for brk in breaks:
            resume_start = steps[brk]
            for j in range(brk):
                if steps[j] >= resume_start:
                    result_mask[j] = False
        return tuple(a[result_mask] for a in (steps,) + arrays)

    e_steps, e_means, e_stds = dedup_monotonic(e_steps, e_means, e_stds)
    if len(ppo["steps"]) > 0:
        arrays = [ppo[k] for k in ["steps"] + keys]
        deduped = dedup_monotonic(*arrays)
        for i, k in enumerate(["steps"] + keys):
            ppo[k] = deduped[i]

    return {"eval_steps": e_steps, "eval_means": e_means, "eval_stds": e_stds, "ppo": ppo}


def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_ab_metrics.py <log_file> [shifted.log]")
        sys.exit(1)

    with open(sys.argv[1], encoding="utf-8", errors="replace") as f:
        u_text = f.read()

    u = parse_all(u_text)

    # Find the best eval step (for star marker)
    best_idx = int(np.argmax(u["eval_means"]))
    best_step_k = u["eval_steps"][best_idx] / 1000
    best_pts = u["eval_means"][best_idx]

    # Find the closest PPO step to the best eval
    best_ppo_idx = None
    if len(u["ppo"]["steps"]) > 0:
        best_ppo_idx = int(np.argmin(np.abs(u["ppo"]["steps"] - u["eval_steps"][best_idx])))

    fig = plt.figure(figsize=(14, 16))
    gs = gridspec.GridSpec(4, 2, hspace=0.35, wspace=0.3)
    fig.suptitle("Hybrid RL Training: Unshifted xP", fontsize=14, fontweight="bold")

    star_kw = dict(marker="*", color="gold", s=300, zorder=10, edgecolors="black", linewidths=0.8)

    # 1. Eval points
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(u["eval_steps"]/1000, u["eval_means"], "b-o", lw=2, ms=4, label="Eval (2023-24)")
    ax1.fill_between(u["eval_steps"]/1000, u["eval_means"]-u["eval_stds"], u["eval_means"]+u["eval_stds"], alpha=0.15, color="blue")
    ax1.axhline(y=4756, color="gold", ls="--", alpha=0.7, label="Oracle: 4,756")
    ax1.axhline(y=2810, color="green", ls=":", alpha=0.7, label="Best human: 2,810")
    ax1.scatter(best_step_k, best_pts, **star_kw, label=f"Best: {best_pts:.0f} @ {best_step_k:.0f}K")
    ax1.set_ylabel("Eval Points (2023-24)")
    ax1.legend(loc="lower left", fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Eval Performance")

    # 2. Policy Loss
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(u["ppo"]["steps"]/1000, u["ppo"]["policy_loss"], "b-", lw=1.5)
    if best_ppo_idx is not None:
        ax2.scatter(u["ppo"]["steps"][best_ppo_idx]/1000, u["ppo"]["policy_loss"][best_ppo_idx], **star_kw)
    ax2.set_ylabel("Policy Loss"); ax2.set_xlabel("Steps (K)")
    ax2.grid(True, alpha=0.3); ax2.set_title("Policy Loss")

    # 3. Value Loss
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(u["ppo"]["steps"]/1000, u["ppo"]["value_loss"], "b-", lw=1.5)
    if best_ppo_idx is not None:
        ax3.scatter(u["ppo"]["steps"][best_ppo_idx]/1000, u["ppo"]["value_loss"][best_ppo_idx], **star_kw)
    ax3.set_ylabel("Value Loss"); ax3.set_xlabel("Steps (K)")
    ax3.grid(True, alpha=0.3); ax3.set_title("Value Loss")

    # 4. Entropy
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(u["ppo"]["steps"]/1000, np.abs(u["ppo"]["entropy"]), "b-", lw=1.5)
    if best_ppo_idx is not None:
        ax4.scatter(u["ppo"]["steps"][best_ppo_idx]/1000, np.abs(u["ppo"]["entropy"][best_ppo_idx]), **star_kw)
    ax4.set_ylabel("|Entropy|"); ax4.set_xlabel("Steps (K)")
    ax4.grid(True, alpha=0.3); ax4.set_title("Policy Entropy")

    # 5. Explained Variance
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.plot(u["ppo"]["steps"]/1000, u["ppo"]["expl_var"], "b-", lw=1.5)
    ax5.axhline(y=1.0, color="gray", ls=":", alpha=0.5, label="Perfect (1.0)")
    if best_ppo_idx is not None:
        ax5.scatter(u["ppo"]["steps"][best_ppo_idx]/1000, u["ppo"]["expl_var"][best_ppo_idx], **star_kw)
    ax5.set_ylabel("Explained Variance"); ax5.set_xlabel("Steps (K)")
    ax5.legend(fontsize=8); ax5.grid(True, alpha=0.3); ax5.set_title("Value Function Quality")

    # 6. Approx KL
    ax6 = fig.add_subplot(gs[3, 0])
    ax6.plot(u["ppo"]["steps"]/1000, u["ppo"]["approx_kl"], "b-", lw=1.5)
    ax6.axhline(y=0.02, color="orange", ls="--", alpha=0.5, label="Target KL (0.02)")
    if best_ppo_idx is not None:
        ax6.scatter(u["ppo"]["steps"][best_ppo_idx]/1000, u["ppo"]["approx_kl"][best_ppo_idx], **star_kw)
    ax6.set_ylabel("Approx KL"); ax6.set_xlabel("Steps (K)")
    ax6.legend(fontsize=8); ax6.grid(True, alpha=0.3); ax6.set_title("KL Divergence")

    # 7. Eval Variance
    ax7 = fig.add_subplot(gs[3, 1])
    ax7.plot(u["eval_steps"]/1000, u["eval_stds"], "b-o", lw=1.5, ms=4)
    ax7.scatter(best_step_k, u["eval_stds"][best_idx], **star_kw)
    ax7.set_ylabel("Eval Std Dev"); ax7.set_xlabel("Steps (K)")
    ax7.grid(True, alpha=0.3); ax7.set_title("Eval Variance")

    plt.savefig("runs/ab_full_metrics.png", dpi=150, bbox_inches="tight")
    print("Saved: runs/ab_full_metrics.png")


if __name__ == "__main__":
    main()
