#!/usr/bin/env python3
"""Plot full A/B training metrics from log data piped via stdin."""
import sys
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def parse_all(text, label):
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

    return {"eval_steps": e_steps, "eval_means": e_means, "eval_stds": e_stds, "ppo": ppo}


def main():
    # Read both log files passed as args
    if len(sys.argv) < 3:
        print("Usage: python plot_ab_metrics.py <unshifted.log> <shifted.log>")
        sys.exit(1)

    with open(sys.argv[1], encoding="utf-8", errors="replace") as f:
        u_text = f.read()
    with open(sys.argv[2], encoding="utf-8", errors="replace") as f:
        s_text = f.read()

    u = parse_all(u_text, "unshifted")
    s = parse_all(s_text, "shifted")

    fig = plt.figure(figsize=(14, 16))
    gs = gridspec.GridSpec(4, 2, hspace=0.35, wspace=0.3)
    fig.suptitle("Hybrid RL A/B: Unshifted vs Shifted xP", fontsize=14, fontweight="bold")

    # 1. Eval points
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(u["eval_steps"]/1000, u["eval_means"], "b-o", lw=2, ms=5, label="Unshifted xP")
    ax1.fill_between(u["eval_steps"]/1000, u["eval_means"]-u["eval_stds"], u["eval_means"]+u["eval_stds"], alpha=0.15, color="blue")
    ax1.plot(s["eval_steps"]/1000, s["eval_means"], "r-s", lw=2, ms=5, label="Shifted xP")
    ax1.fill_between(s["eval_steps"]/1000, s["eval_means"]-s["eval_stds"], s["eval_means"]+s["eval_stds"], alpha=0.15, color="red")
    ax1.axhline(y=4756, color="gold", ls="--", alpha=0.7, label="Oracle: 4,756")
    ax1.axhline(y=2810, color="green", ls=":", alpha=0.7, label="Best human: 2,810")
    ax1.set_ylabel("Eval Points (2023-24)")
    ax1.legend(loc="lower left", fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Eval Performance")

    # 2. Policy Loss
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(u["ppo"]["steps"]/1000, u["ppo"]["policy_loss"], "b-", lw=1.5, label="Unshifted")
    ax2.plot(s["ppo"]["steps"]/1000, s["ppo"]["policy_loss"], "r-", lw=1.5, label="Shifted")
    ax2.set_ylabel("Policy Loss"); ax2.set_xlabel("Steps (K)")
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3); ax2.set_title("Policy Loss")

    # 3. Value Loss
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(u["ppo"]["steps"]/1000, u["ppo"]["value_loss"], "b-", lw=1.5, label="Unshifted")
    ax3.plot(s["ppo"]["steps"]/1000, s["ppo"]["value_loss"], "r-", lw=1.5, label="Shifted")
    ax3.set_ylabel("Value Loss"); ax3.set_xlabel("Steps (K)")
    ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3); ax3.set_title("Value Loss")

    # 4. Entropy
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(u["ppo"]["steps"]/1000, np.abs(u["ppo"]["entropy"]), "b-", lw=1.5, label="Unshifted")
    ax4.plot(s["ppo"]["steps"]/1000, np.abs(s["ppo"]["entropy"]), "r-", lw=1.5, label="Shifted")
    ax4.set_ylabel("|Entropy|"); ax4.set_xlabel("Steps (K)")
    ax4.legend(fontsize=8); ax4.grid(True, alpha=0.3); ax4.set_title("Policy Entropy")

    # 5. Explained Variance
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.plot(u["ppo"]["steps"]/1000, u["ppo"]["expl_var"], "b-", lw=1.5, label="Unshifted")
    ax5.plot(s["ppo"]["steps"]/1000, s["ppo"]["expl_var"], "r-", lw=1.5, label="Shifted")
    ax5.axhline(y=1.0, color="gray", ls=":", alpha=0.5, label="Perfect (1.0)")
    ax5.set_ylabel("Explained Variance"); ax5.set_xlabel("Steps (K)")
    ax5.legend(fontsize=8); ax5.grid(True, alpha=0.3); ax5.set_title("Value Function Quality")

    # 6. Approx KL
    ax6 = fig.add_subplot(gs[3, 0])
    ax6.plot(u["ppo"]["steps"]/1000, u["ppo"]["approx_kl"], "b-", lw=1.5, label="Unshifted")
    ax6.plot(s["ppo"]["steps"]/1000, s["ppo"]["approx_kl"], "r-", lw=1.5, label="Shifted")
    ax6.axhline(y=0.02, color="orange", ls="--", alpha=0.5, label="Target KL (0.02)")
    ax6.set_ylabel("Approx KL"); ax6.set_xlabel("Steps (K)")
    ax6.legend(fontsize=8); ax6.grid(True, alpha=0.3); ax6.set_title("KL Divergence")

    # 7. Eval Variance
    ax7 = fig.add_subplot(gs[3, 1])
    ax7.plot(u["eval_steps"]/1000, u["eval_stds"], "b-o", lw=1.5, ms=5, label="Unshifted")
    ax7.plot(s["eval_steps"]/1000, s["eval_stds"], "r-s", lw=1.5, ms=5, label="Shifted")
    ax7.set_ylabel("Eval Std Dev"); ax7.set_xlabel("Steps (K)")
    ax7.legend(fontsize=8); ax7.grid(True, alpha=0.3); ax7.set_title("Eval Variance")

    plt.savefig("runs/ab_full_metrics.png", dpi=150, bbox_inches="tight")
    print("Saved: runs/ab_full_metrics.png")


if __name__ == "__main__":
    main()
