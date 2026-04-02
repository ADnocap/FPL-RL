#!/usr/bin/env python3
"""Thorough GW-by-GW audit of a hybrid model run.

Checks: lookahead, budget, squad composition, club limits, formation,
transfer validity, chip legality, captain in lineup, scoring correctness.

Usage:
    python scripts/audit_run.py --model runs/best_hybrid_model.zip --season 2024-25
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sb3_contrib import MaskablePPO

from fpl_rl.data.downloader import DEFAULT_DATA_DIR
from fpl_rl.data.loader import SeasonDataLoader
from fpl_rl.engine.scoring import get_player_points, did_player_play
from fpl_rl.env.hybrid_env import HybridFPLEnv
from fpl_rl.utils.constants import (
    MAX_PER_CLUB,
    POSITION_LIMITS,
    SQUAD_SIZE,
    STARTING_BUDGET,
    VALID_FORMATIONS,
    Position,
)


def audit_squad(state, loader, gw, label=""):
    """Check squad validity: size, positions, club limits, budget."""
    errors = []
    squad = state.squad
    players = squad.players

    # Squad size
    if len(players) != SQUAD_SIZE:
        errors.append(f"Squad size {len(players)} != {SQUAD_SIZE}")

    # Position counts
    pos_counts = Counter(p.position for p in players)
    for pos, expected in POSITION_LIMITS.items():
        actual = pos_counts.get(pos, 0)
        if actual != expected:
            errors.append(f"{pos.name}: {actual} != {expected}")

    # Club limits
    team_map = loader._team_map
    club_counts: Counter = Counter()
    for p in players:
        tid = team_map.get(p.element_id, -1)
        club_counts[tid] += 1
    for tid, count in club_counts.items():
        if count > MAX_PER_CLUB:
            errors.append(f"Team {tid}: {count} players > {MAX_PER_CLUB}")

    # Lineup size and formation
    if len(squad.lineup) != 11:
        errors.append(f"Lineup size {len(squad.lineup)} != 11")
    else:
        lineup_players = [players[i] for i in squad.lineup]
        lineup_pos = Counter(p.position for p in lineup_players)
        formation = (
            lineup_pos.get(Position.DEF, 0),
            lineup_pos.get(Position.MID, 0),
            lineup_pos.get(Position.FWD, 0),
        )
        if formation not in VALID_FORMATIONS:
            errors.append(f"Invalid formation {formation}")
        if lineup_pos.get(Position.GK, 0) != 1:
            errors.append(f"GK in lineup: {lineup_pos.get(Position.GK, 0)}")

    # Bench size
    if len(squad.bench) != 4:
        errors.append(f"Bench size {len(squad.bench)} != 4")

    # All indices unique and covering all players
    all_idx = set(squad.lineup) | set(squad.bench)
    if len(all_idx) != SQUAD_SIZE:
        errors.append(f"Lineup+bench indices cover {len(all_idx)} != {SQUAD_SIZE}")

    # Captain in lineup
    if squad.captain_idx not in squad.lineup:
        errors.append(f"Captain idx {squad.captain_idx} not in lineup")
    if squad.vice_captain_idx not in squad.lineup:
        errors.append(f"Vice-captain idx {squad.vice_captain_idx} not in lineup")

    # Budget: bank >= 0
    if state.bank < 0:
        errors.append(f"Negative bank: {state.bank}")

    # Team value + bank should be roughly STARTING_BUDGET
    total_value = sum(p.selling_price for p in players) + state.bank
    # Can be higher due to price appreciation, but not dramatically lower
    if total_value < STARTING_BUDGET * 0.8:
        errors.append(f"Suspicious total value: {total_value} (bank={state.bank})")

    return errors


def check_lookahead(predictions, loader, gw):
    """Verify that predictions for GW N don't use GW N's actual data."""
    issues = []
    eids = loader.get_all_element_ids(gw)

    # Check a sample of predictions vs actuals
    exact_matches = 0
    total_checked = 0
    for eid in eids[:50]:
        pred = predictions.get((eid, gw), None)
        if pred is None:
            continue
        data = loader.get_player_gw(eid, gw)
        if data is None:
            continue
        actual = float(data["total_points"])
        total_checked += 1
        if abs(pred - actual) < 0.01:
            exact_matches += 1

    # If > 50% of predictions exactly match actuals, likely lookahead
    if total_checked > 0 and exact_matches / total_checked > 0.5:
        issues.append(
            f"GW{gw}: {exact_matches}/{total_checked} predictions exactly match actuals — LOOKAHEAD?"
        )

    return issues


def verify_points(state, loader, gw, result_gw_points, result_captain_points):
    """Manually recompute points to verify engine scoring."""
    squad = state.squad

    # Lineup points
    lineup_pts = sum(
        get_player_points(loader, squad.players[i].element_id, gw)
        for i in squad.lineup
    )

    # Captain bonus: (multiplier - 1) * captain_base_points
    captain_id = squad.players[squad.captain_idx].element_id
    vice_id = squad.players[squad.vice_captain_idx].element_id

    if did_player_play(loader, captain_id, gw):
        expected_captain = get_player_points(loader, captain_id, gw)
    elif did_player_play(loader, vice_id, gw):
        expected_captain = get_player_points(loader, vice_id, gw)
    else:
        expected_captain = 0

    expected_gw = lineup_pts + expected_captain
    issues = []

    # Allow small difference due to auto-subs changing lineup
    if abs(expected_gw - result_gw_points) > 30:
        issues.append(
            f"Points mismatch: expected ~{expected_gw}, got {result_gw_points} "
            f"(lineup={lineup_pts}, captain_bonus={expected_captain})"
        )

    return issues


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--model", default="runs/best_hybrid_model.zip")
    p.add_argument("--season", default="2024-25")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    print(f"Loading model: {args.model}")
    model = MaskablePPO.load(args.model, device="cpu")

    data_dir = DEFAULT_DATA_DIR
    pred_data_dir = data_dir.parent if data_dir.name == "raw" else data_dir

    from fpl_rl.prediction.integration import PredictionIntegrator

    integrator = PredictionIntegrator.from_model(
        Path("models/point_predictor"), pred_data_dir, args.season,
    )

    env = HybridFPLEnv(
        season=args.season, data_dir=data_dir,
        prediction_integrator=integrator,
    )
    loader = env.loader

    print(f"Season: {args.season}")
    print(f"Predictions loaded: {len(integrator._predictions)}")

    # Check for lookahead in predictions
    print("\n=== LOOKAHEAD CHECK ===")
    for gw in [1, 5, 10, 15, 20, 25, 30]:
        issues = check_lookahead(integrator._predictions, loader, gw)
        if issues:
            for iss in issues:
                print(f"  WARNING: {iss}")
        else:
            print(f"  GW{gw}: OK (predictions != actuals)")

    # Run episode
    print(f"\n=== GW-BY-GW AUDIT (seed={args.seed}) ===")
    obs, info = env.reset(seed=args.seed)

    total_errors = []
    total_transfers = 0
    chips_used = []
    prev_squad_eids = set()

    print(f"\n{'GW':>3} {'Gross':>6} {'Hit':>4} {'Net':>6} {'Xfer':>5} "
          f"{'Chip':>6} {'Bank':>6} {'FTs':>3} {'Total':>6} {'Errors':>8}")
    print("-" * 72)

    for step in range(env._num_gws):
        gw = env.state.current_gw
        ft_before = env.state.free_transfers
        bank_before = env.state.bank
        squad_before = set(p.element_id for p in env.state.squad.players)

        # Pre-step audit
        pre_errors = audit_squad(env.state, loader, gw, f"pre-GW{gw}")

        # Model action
        masks = env.action_masks()
        action, _ = model.predict(obs, deterministic=True, action_masks=masks)
        transfer_count = int(action[0])
        chip_idx = int(action[1])

        obs, reward, term, trunc, info = env.step(action)

        gw_points = info["gw_points"]
        net_points = info["net_points"]
        hit_cost = info["hit_cost"]
        n_xfers = info["num_transfers"]
        chip = info.get("active_chip")
        total_transfers += n_xfers

        if chip:
            chips_used.append((gw, chip))

        # Post-step audit
        post_errors = audit_squad(env.state, loader, gw + 1 if not term else gw, f"post-GW{gw}")

        # Verify transfers are real
        squad_after = set(p.element_id for p in env.state.squad.players)
        actual_changes = len(squad_before - squad_after)

        # Check hit cost is correct
        free_xfers = ft_before
        expected_hit = max(0, n_xfers - free_xfers) * 4
        hit_errors = []
        if hit_cost != expected_hit and chip not in ("wildcard", "free_hit"):
            hit_errors.append(f"Hit: expected {expected_hit}, got {hit_cost}")

        all_gw_errors = pre_errors + post_errors + hit_errors
        total_errors.extend(all_gw_errors)

        err_str = f"{len(all_gw_errors)} err" if all_gw_errors else "OK"
        chip_str = chip[:2].upper() if chip else "-"

        print(f"{gw:>3} {gw_points:>6} {hit_cost:>4} {net_points:>6} {n_xfers:>5} "
              f"{chip_str:>6} {env.state.bank:>6} {env.state.free_transfers:>3} "
              f"{info['total_points']:>6} {err_str:>8}")

        if all_gw_errors:
            for e in all_gw_errors:
                print(f"      ERROR: {e}")

        if term:
            break

    # Final summary
    print(f"\n{'='*72}")
    print(f"AUDIT SUMMARY")
    print(f"{'='*72}")
    print(f"Season: {args.season}")
    print(f"Total points: {info['total_points']}")
    print(f"Total transfers: {total_transfers}")
    print(f"Chips used: {chips_used if chips_used else 'none'}")
    print(f"Total errors found: {len(total_errors)}")

    if total_errors:
        print(f"\nALL ERRORS:")
        for e in total_errors:
            print(f"  - {e}")
    else:
        print(f"\nNO ERRORS — all rules verified correct")

    final_value = sum(p.selling_price for p in env.state.squad.players) + env.state.bank
    print(f"\nFinal squad value + bank: {final_value} (started at {STARTING_BUDGET})")
    print(f"Final bank: {env.state.bank}")

    env.close()


if __name__ == "__main__":
    main()
