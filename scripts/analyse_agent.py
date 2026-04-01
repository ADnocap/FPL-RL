#!/usr/bin/env python3
"""Analyse a trained RL agent's FPL decisions in detail.

Runs episodes on eval/holdout seasons and tracks every decision:
transfers, chips, captain choices, formation, points breakdown.

Usage:
    python scripts/analyse_agent.py
    python scripts/analyse_agent.py --season 2024-25 --episodes 5
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sb3_contrib import MaskablePPO

from fpl_rl.env.fpl_env import FPLEnv
from fpl_rl.env.action_space import CHIP_INDEX_MAP
from fpl_rl.prediction.integration import PredictionIntegrator
from fpl_rl.utils.constants import ALL_CHIPS


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="runs/fpl_ppo/best_model/best_model")
    p.add_argument("--season", default="2024-25")
    p.add_argument("--data-dir", default="data/raw")
    p.add_argument("--predictor-dir", default="models/point_predictor")
    p.add_argument("--episodes", type=int, default=5)
    return p.parse_args()


def run_detailed_episode(model, env):
    """Run one episode tracking every decision."""
    obs, _ = env.reset()
    done = False
    gw_log = []

    state = env.state

    while not done:
        gw = state.current_gw
        masks = env.action_masks()
        action, _ = model.predict(obs, deterministic=True, action_masks=masks)

        # Decode action to understand what was chosen
        engine_action = env.action_encoder.decode(action, state)

        # Record pre-step state
        pre_state = state.copy()
        squad_before = [p.element_id for p in pre_state.squad.players]
        captain_eid = pre_state.squad.players[pre_state.squad.captain_idx].element_id
        bank_before = pre_state.bank
        ft_before = pre_state.free_transfers

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        state = env.state

        # Post-step analysis
        squad_after = [p.element_id for p in state.squad.players]
        transfers_out = set(squad_before) - set(squad_after)
        transfers_in = set(squad_after) - set(squad_before)

        # Captain info
        new_captain_eid = state.squad.players[state.squad.captain_idx].element_id if not done else captain_eid

        # Formation
        lineup_players = pre_state.squad.get_lineup_players()
        formation = Counter(p.position.name for p in lineup_players)
        formation_str = f"{formation.get('DEF',0)}-{formation.get('MID',0)}-{formation.get('FWD',0)}"

        gw_log.append({
            "gw": info["gw"],
            "gw_points": info["gw_points"],
            "net_points": info["net_points"],
            "hit_cost": info["hit_cost"],
            "total_points": info["total_points"],
            "chip": info["active_chip"],
            "captain_points": info.get("captain_points", 0),
            "bench_points": info.get("bench_points", 0),
            "auto_subs": info.get("auto_subs", 0),
            "captain_failover": info.get("captain_failover", False),
            "n_transfers": len(transfers_in),
            "transfers_in": transfers_in,
            "transfers_out": transfers_out,
            "formation": formation_str,
            "bank": bank_before,
            "free_transfers": ft_before,
            "reward": reward,
        })

    return gw_log


def print_episode_summary(ep_num, gw_log, loader):
    """Print detailed summary of one episode."""
    total = gw_log[-1]["total_points"]
    print(f"\n{'='*80}")
    print(f"  EPISODE {ep_num} — {total} total points")
    print(f"{'='*80}")

    # GW-by-GW table
    print(f"\n{'GW':>3} {'Pts':>5} {'Net':>5} {'Hit':>4} {'Chip':<8} "
          f"{'Capt':>5} {'Bench':>5} {'Xfers':>5} {'Formation':<8} {'FT':>3} {'Bank':>6} {'Total':>6}")
    print("-" * 80)

    for g in gw_log:
        chip_str = g["chip"] or ""
        print(f"{g['gw']:>3} {g['gw_points']:>5} {g['net_points']:>5} "
              f"{g['hit_cost']:>4} {chip_str:<8} "
              f"{g['captain_points']:>5} {g['bench_points']:>5} "
              f"{g['n_transfers']:>5} {g['formation']:<8} "
              f"{g['free_transfers']:>3} {g['bank']/10:>6.1f} "
              f"{g['total_points']:>6}")

    # Summary stats
    chips_used = [g["chip"] for g in gw_log if g["chip"]]
    total_hits = sum(g["hit_cost"] for g in gw_log)
    total_transfers = sum(g["n_transfers"] for g in gw_log)
    total_captain_pts = sum(g["captain_points"] for g in gw_log)
    total_bench_pts = sum(g["bench_points"] for g in gw_log)
    auto_subs = sum(len(g["auto_subs"]) if isinstance(g["auto_subs"], list) else g["auto_subs"] for g in gw_log)
    captain_failovers = sum(1 for g in gw_log if g["captain_failover"])
    formations = Counter(g["formation"] for g in gw_log)

    print(f"\n--- Summary ---")
    print(f"  Total points:     {total}")
    print(f"  Total hits:       {total_hits} pts ({total_hits//4} extra transfers)")
    print(f"  Total transfers:  {total_transfers}")
    print(f"  Captain points:   {total_captain_pts} (avg {total_captain_pts/38:.1f}/GW)")
    print(f"  Bench points:     {total_bench_pts}")
    print(f"  Auto-subs:        {auto_subs}")
    print(f"  Captain failover: {captain_failovers}")

    print(f"\n  Chips used ({len(chips_used)}/8):")
    if chips_used:
        for g in gw_log:
            if g["chip"]:
                print(f"    GW{g['gw']}: {g['chip']} ({g['gw_points']} pts)")
    else:
        print(f"    NONE — agent is not using chips!")

    chips_not_used = 8 - len(chips_used)
    if chips_not_used > 0:
        print(f"    ({chips_not_used} chips wasted)")

    print(f"\n  Formations used:")
    for f, count in formations.most_common():
        print(f"    {f}: {count} GWs")


def print_cross_episode_analysis(all_logs):
    """Print analysis across multiple episodes."""
    print(f"\n{'='*80}")
    print(f"  CROSS-EPISODE ANALYSIS ({len(all_logs)} episodes)")
    print(f"{'='*80}")

    points = [log[-1]["total_points"] for log in all_logs]
    print(f"\n  Points: {np.mean(points):.1f} +/- {np.std(points):.1f} "
          f"(min={min(points)}, max={max(points)})")

    # Chip usage analysis
    chip_usage = defaultdict(list)
    for ep, log in enumerate(all_logs):
        ep_chips = {}
        for g in log:
            if g["chip"]:
                ep_chips[g["chip"]] = g["gw"]
        for chip in ALL_CHIPS:
            # Check both halves
            gws = [g["gw"] for g in log if g["chip"] == chip]
            chip_usage[chip].extend(gws)

    print(f"\n  Chip usage across episodes:")
    for chip in ALL_CHIPS:
        gws = chip_usage[chip]
        if gws:
            print(f"    {chip:<18} used {len(gws)}/{len(all_logs)} eps, "
                  f"avg GW={np.mean(gws):.0f} (GWs: {sorted(gws)})")
        else:
            print(f"    {chip:<18} NEVER USED")

    # Transfer hit analysis
    total_hits_per_ep = [sum(g["hit_cost"] for g in log) for log in all_logs]
    print(f"\n  Hit cost per season: {np.mean(total_hits_per_ep):.1f} +/- {np.std(total_hits_per_ep):.1f}")

    # Captain analysis
    capt_pts_per_ep = [sum(g["captain_points"] for g in log) for log in all_logs]
    print(f"  Captain pts per season: {np.mean(capt_pts_per_ep):.1f} +/- {np.std(capt_pts_per_ep):.1f}")

    # GW-by-GW avg
    print(f"\n  Per-GW average points:")
    for gw in range(1, 39):
        gw_pts = [log[gw-1]["gw_points"] for log in all_logs if gw <= len(log)]
        if gw_pts:
            if gw % 5 == 1 or gw == 38:
                print(f"    GW{gw:>2}: {np.mean(gw_pts):>6.1f} pts")


def main():
    args = parse_args()

    data_dir = Path(args.data_dir)
    pred_data_dir = data_dir.parent

    print(f"Loading model from {args.model}...")
    model = MaskablePPO.load(args.model, device="cpu")

    print(f"Building env for {args.season}...")
    env_kwargs = dict(season=args.season, data_dir=data_dir)
    predictor_dir = Path(args.predictor_dir)
    if predictor_dir.exists():
        env_kwargs["prediction_integrator"] = PredictionIntegrator.from_model(
            predictor_dir, pred_data_dir, args.season,
        )
    env = FPLEnv(**env_kwargs)

    all_logs = []
    for ep in range(args.episodes):
        print(f"\nRunning episode {ep+1}/{args.episodes}...")
        gw_log = run_detailed_episode(model, env)
        all_logs.append(gw_log)
        print_episode_summary(ep + 1, gw_log, env.loader)

    if len(all_logs) > 1:
        print_cross_episode_analysis(all_logs)

    env.close()


if __name__ == "__main__":
    main()
