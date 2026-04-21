#!/usr/bin/env python3
"""Infer chance_of_playing from historical data and test EP reconstruction."""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("data/raw")


def round_half(x):
    return np.round(x * 2) / 2


def load_season(season):
    """Load merged_gw and teams for a season."""
    mgw = pd.read_csv(DATA_DIR / season / "gws" / "merged_gw.csv",
                       encoding="utf-8", on_bad_lines="skip")
    for c in ["xP", "total_points", "minutes", "element", "GW",
              "opponent_team", "value", "selected", "transfers_in",
              "transfers_out", "transfers_balance"]:
        if c in mgw.columns:
            mgw[c] = pd.to_numeric(mgw[c], errors="coerce")
    mgw["kickoff_dt"] = pd.to_datetime(mgw["kickoff_time"], utc=True, errors="coerce")
    mgw["was_home"] = mgw["was_home"].map(
        {True: True, False: False, "True": True, "False": False}
    )

    teams = pd.read_csv(DATA_DIR / season / "teams.csv")
    name_to_id = dict(zip(teams["name"], teams["id"]))
    team_str = dict(zip(teams["id"], teams["strength"].astype(int)))

    # Map team names to IDs
    team_numeric = pd.to_numeric(mgw["team"], errors="coerce")
    if team_numeric.notna().mean() < 0.5:
        mgw["team_id"] = mgw["team"].map(name_to_id)
    else:
        mgw["team_id"] = team_numeric

    return mgw, team_str


def compute_form(mgw, gw, deadline):
    """30-day form: mean pts for played games (mins>0), rounded to 0.5."""
    cutoff = deadline - pd.Timedelta(days=30)
    mask = (
        (mgw["GW"] < gw) &
        (mgw["kickoff_dt"] >= cutoff) &
        (mgw["kickoff_dt"] < deadline) &
        (mgw["minutes"] > 0)
    )
    prev = mgw[mask]
    if prev.empty:
        return pd.Series(dtype=float)
    return round_half(prev.groupby("element")["total_points"].mean())


def infer_cop_strategies(mgw, gw, all_elements):
    """Build multiple cop inference strategies for each player.

    Returns dict: strategy_name -> {element_id: cop_value}
    """
    prev = mgw[mgw["GW"] < gw].copy()
    if prev.empty:
        return {}

    strategies = {}

    # --- Strategy 1: Consecutive 0-minute GWs ---
    # If last N appearances had 0 mins, likely injured
    last_appearances = prev.sort_values("GW").groupby("element").tail(3)
    consec_zero = {}
    for eid in all_elements:
        player_recent = last_appearances[last_appearances["element"] == eid].sort_values("GW")
        if player_recent.empty:
            consec_zero[eid] = 0  # never appeared = cop 0
            continue
        # Count consecutive 0-min from most recent backwards
        mins_list = player_recent["minutes"].values[::-1]  # most recent first
        count = 0
        for m in mins_list:
            if m == 0:
                count += 1
            else:
                break
        consec_zero[eid] = count

    # Map consecutive zeros to cop
    cop_consec = {}
    for eid, cz in consec_zero.items():
        if cz >= 3:
            cop_consec[eid] = 0
        elif cz >= 2:
            cop_consec[eid] = 25
        elif cz >= 1:
            cop_consec[eid] = 75
        else:
            cop_consec[eid] = 100
    strategies["consec_zero"] = cop_consec

    # --- Strategy 2: Last GW minutes binary ---
    # Did they play in the most recent GW?
    max_prev_gw = prev["GW"].max()
    last_gw = prev[prev["GW"] == max_prev_gw]
    last_gw_mins = dict(last_gw.groupby("element")["minutes"].sum())

    cop_last_gw = {}
    for eid in all_elements:
        if eid in last_gw_mins:
            cop_last_gw[eid] = 100 if last_gw_mins[eid] > 0 else 25
        else:
            # Not in last GW data at all
            # Check if appeared in any recent GW
            recent = prev[prev["GW"] >= max_prev_gw - 2]
            recent_player = recent[recent["element"] == eid]
            if recent_player.empty:
                cop_last_gw[eid] = 0
            elif (recent_player["minutes"] > 0).any():
                cop_last_gw[eid] = 75  # played recently but missed last GW
            else:
                cop_last_gw[eid] = 0
    strategies["last_gw"] = cop_last_gw

    # --- Strategy 3: Transfer out spike ---
    # If transfers_out in last GW is way above average, injury news
    if "transfers_out" in prev.columns:
        # Per-element last GW transfers_out
        last_gw_tfers = dict(last_gw.groupby("element")["transfers_out"].sum())
        # Per-element average transfers_out over season
        avg_tfers = prev.groupby("element")["transfers_out"].mean().to_dict()

        cop_tfer = {}
        for eid in all_elements:
            recent_t = last_gw_tfers.get(eid, 0)
            avg_t = avg_tfers.get(eid, 0)
            if avg_t > 0 and recent_t > avg_t * 3:
                # Massive sell-off = injury news
                cop_tfer[eid] = 25
            elif eid in last_gw_mins and last_gw_mins[eid] == 0:
                cop_tfer[eid] = 50
            else:
                cop_tfer[eid] = 100
        strategies["tfer_spike"] = cop_tfer

    # --- Strategy 4: Combined heuristic ---
    cop_combined = {}
    for eid in all_elements:
        cz = consec_zero.get(eid, 0)
        last_mins = last_gw_mins.get(eid, -1)  # -1 = not in data

        if last_mins == -1:
            # Not in last GW data at all — check further back
            any_recent = prev[
                (prev["element"] == eid) & (prev["GW"] >= max_prev_gw - 3)
            ]
            if any_recent.empty:
                cop_combined[eid] = 0
            elif (any_recent["minutes"] > 0).any():
                cop_combined[eid] = 50  # was playing but disappeared
            else:
                cop_combined[eid] = 0
        elif cz >= 3:
            cop_combined[eid] = 0
        elif cz >= 2:
            cop_combined[eid] = 25
        elif cz >= 1:
            # Missed last game but played before that
            cop_combined[eid] = 50
        else:
            cop_combined[eid] = 100
    strategies["combined"] = cop_combined

    # --- Strategy 5: Always 100 (baseline) ---
    strategies["always_100"] = {eid: 100 for eid in all_elements}

    return strategies


def verify_season(season):
    """Run all strategies on one season."""
    mgw, team_str = load_season(season)
    if "xP" not in mgw.columns:
        return None

    gw_deadlines = mgw.groupby("GW")["kickoff_dt"].min().sort_index()

    # All unique elements across the season
    all_elements = mgw["element"].unique()

    strategy_results = {}  # strategy -> list of (ep_computed, xP, actual_pts) rows

    for gw in sorted(mgw["GW"].unique()):
        if gw < 3:  # need at least 2 prior GWs for cop inference
            continue
        if gw not in gw_deadlines.index:
            continue

        deadline = gw_deadlines[gw]
        form = compute_form(mgw, gw, deadline)

        gw_data = mgw[mgw["GW"] == gw]
        gw_elements = gw_data["element"].unique()

        # Infer cop with different strategies
        cops = infer_cop_strategies(mgw, gw, gw_elements)

        for eid in gw_elements:
            player_fixtures = gw_data[gw_data["element"] == eid]
            xp_actual = player_fixtures["xP"].mean()
            actual_pts = player_fixtures["total_points"].sum()

            if pd.isna(xp_actual):
                continue

            player_form = form.get(eid, 0.0)
            if pd.isna(player_form):
                player_form = 0.0

            # Compute EP per fixture
            for strat_name, cop_dict in cops.items():
                cop = cop_dict.get(eid, 100)

                ep_sum = 0.0
                for _, frow in player_fixtures.drop_duplicates("fixture").iterrows():
                    own_id = frow.get("team_id")
                    opp_id = frow.get("opponent_team")
                    if pd.notna(own_id) and pd.notna(opp_id):
                        own_s = team_str.get(int(own_id), 3)
                        opp_s = team_str.get(int(opp_id), 3)
                        offset = (own_s - opp_s) * 0.5
                    else:
                        offset = 0.0

                    ep_fix = round((player_form + offset) * cop / 100, 1)
                    ep_sum += ep_fix

                if strat_name not in strategy_results:
                    strategy_results[strat_name] = []
                strategy_results[strat_name].append({
                    "ep": round(ep_sum, 1),
                    "xP": round(xp_actual, 1),
                    "actual": actual_pts,
                    "gw": gw,
                })

    return strategy_results


def evaluate(rows):
    """Compute metrics from list of {ep, xP, actual, gw} dicts."""
    df = pd.DataFrame(rows)

    pos = df[df["xP"] > 0]
    diff = (pos["ep"] - pos["xP"]).abs()

    # Per-GW correlation with actual points (using ALL rows including xP=0)
    gw_corrs_ep = []
    gw_corrs_xp = []
    for gw in df["gw"].unique():
        gd = df[df["gw"] == gw]
        if len(gd) > 20:
            c1 = gd["ep"].corr(gd["actual"])
            c2 = gd["xP"].corr(gd["actual"])
            if not np.isnan(c1):
                gw_corrs_ep.append(c1)
            if not np.isnan(c2):
                gw_corrs_xp.append(c2)

    # xP=0 accuracy
    xp_zero = df[df["xP"] == 0]
    our_zero_correct = ((df["xP"] == 0) & (df["ep"] == 0)).sum()
    false_pos = ((df["xP"] == 0) & (df["ep"] > 0)).sum()

    return {
        "corr_xp": pos["ep"].corr(pos["xP"]),
        "mae_xp": diff.mean(),
        "exact": (diff < 0.01).mean(),
        "within_05": (diff <= 0.5).mean(),
        "within_10": (diff <= 1.0).mean(),
        "gw_corr_ep": np.mean(gw_corrs_ep) if gw_corrs_ep else np.nan,
        "gw_corr_xp": np.mean(gw_corrs_xp) if gw_corrs_xp else np.nan,
        "n_xp0": len(xp_zero),
        "xp0_correct": our_zero_correct,
        "false_pos": false_pos,
        "fp_rate": false_pos / max(len(xp_zero), 1),
    }


def main():
    seasons = ["2020-21", "2021-22", "2022-23", "2023-24", "2024-25"]

    print("=" * 100)
    print("CHANCE_OF_PLAYING INFERENCE: Testing strategies to close the xP gap")
    print("=" * 100)

    # Collect all seasons
    all_strategy_rows = {}
    for season in seasons:
        print(f"\nProcessing {season}...")
        result = verify_season(season)
        if not result:
            continue
        for strat, rows in result.items():
            if strat not in all_strategy_rows:
                all_strategy_rows[strat] = []
            all_strategy_rows[strat].extend(rows)

    if not all_strategy_rows:
        print("No results!")
        return

    # Evaluate each strategy
    print("\n" + "=" * 100)
    print("RESULTS ACROSS ALL SEASONS")
    print("=" * 100)

    header = (
        f"{'Strategy':<16} {'r(xP)':>7} {'MAE':>7} {'Exact':>7} {'<0.5':>7} {'<1.0':>7} "
        f"{'GW_r(pts)':>10} {'xP_GW_r':>8} {'Gap':>7} {'FP rate':>8}"
    )
    print(header)
    print("-" * len(header))

    strat_metrics = {}
    for strat in ["always_100", "last_gw", "consec_zero", "tfer_spike", "combined"]:
        if strat not in all_strategy_rows:
            continue
        m = evaluate(all_strategy_rows[strat])
        strat_metrics[strat] = m
        gap = m["gw_corr_xp"] - m["gw_corr_ep"]
        print(
            f"  {strat:<14} {m['corr_xp']:>7.4f} {m['mae_xp']:>7.3f} {m['exact']:>6.1%} "
            f"{m['within_05']:>6.1%} {m['within_10']:>6.1%} {m['gw_corr_ep']:>10.4f} "
            f"{m['gw_corr_xp']:>8.4f} {gap:>+7.4f} {m['fp_rate']:>7.1%}"
        )

    # Breakdown: how much did cop inference help?
    print("\n" + "=" * 100)
    print("IMPACT ANALYSIS")
    print("=" * 100)

    baseline = strat_metrics.get("always_100", {})
    best_strat = max(strat_metrics.items(), key=lambda x: x[1].get("gw_corr_ep", 0))

    if baseline and best_strat:
        bl_gap = baseline["gw_corr_xp"] - baseline["gw_corr_ep"]
        best_gap = best_strat[1]["gw_corr_xp"] - best_strat[1]["gw_corr_ep"]
        closed = bl_gap - best_gap

        print(f"Baseline (cop=100):     per-GW corr = {baseline['gw_corr_ep']:.4f}, gap to xP = {bl_gap:+.4f}")
        print(f"Best ({best_strat[0]:<14}): per-GW corr = {best_strat[1]['gw_corr_ep']:.4f}, gap to xP = {best_gap:+.4f}")
        print(f"Gap closed: {closed:.4f} ({100*closed/bl_gap:.0f}% of total gap)")
        print(f"Remaining gap: {best_gap:+.4f}")

        print(f"\nFalse positive reduction:")
        print(f"  Baseline: {baseline['fp_rate']:.1%} of xP=0 rows we incorrectly predict >0")
        print(f"  Best:     {best_strat[1]['fp_rate']:.1%} of xP=0 rows we incorrectly predict >0")

    # Per-season breakdown for best strategy
    print(f"\n" + "=" * 100)
    print(f"PER-SEASON BREAKDOWN: {best_strat[0]} strategy")
    print("=" * 100)
    for season in seasons:
        # Filter rows for this season (approximate by GW ranges)
        # Actually we need to re-run per season. Let's just run verify_season again
        pass

    print(f"\nBest strategy: {best_strat[0]}")


if __name__ == "__main__":
    main()
