#!/usr/bin/env python3
"""Verify the EP formula against vaastav's xP data across all seasons."""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("data/raw")


def round_half(x):
    """Round to nearest 0.5 like FPL does for form."""
    return np.round(x * 2) / 2


def compute_form_30d(merged_gw, gw, deadline):
    """Compute 30-day form for all players before a given GW.

    form = round_to_0.5(mean(total_points for games with minutes > 0 in last 30 days))
    """
    cutoff = deadline - pd.Timedelta(days=30)

    # Only previous GWs, within 30-day window, only played (minutes > 0)
    mask = (
        (merged_gw["GW"] < gw) &
        (merged_gw["kickoff_dt"] >= cutoff) &
        (merged_gw["kickoff_dt"] < deadline) &
        (merged_gw["minutes"] > 0)
    )
    prev = merged_gw[mask]

    if prev.empty:
        return pd.Series(dtype=float)

    form = prev.groupby("element")["total_points"].mean()
    return round_half(form)


def load_team_strength(season):
    """Load simple integer strength (1-5) per team from teams.csv."""
    teams_path = DATA_DIR / season / "teams.csv"
    if not teams_path.exists():
        return {}

    teams = pd.read_csv(teams_path)
    if "strength" not in teams.columns or "id" not in teams.columns:
        return {}

    return dict(zip(teams["id"], teams["strength"].astype(int)))


def verify_season(season):
    """Verify EP formula for one season. Returns stats dict."""
    mgw_path = DATA_DIR / season / "gws" / "merged_gw.csv"
    if not mgw_path.exists():
        return None

    try:
        mgw = pd.read_csv(mgw_path, encoding="utf-8", on_bad_lines="skip")
    except UnicodeDecodeError:
        mgw = pd.read_csv(mgw_path, encoding="latin-1", on_bad_lines="skip")

    # Check xP exists
    if "xP" not in mgw.columns:
        print(f"  {season}: no xP column")
        return None

    for c in ["xP", "total_points", "minutes", "element", "GW", "opponent_team"]:
        if c in mgw.columns:
            mgw[c] = pd.to_numeric(mgw[c], errors="coerce")

    if "kickoff_time" not in mgw.columns:
        print(f"  {season}: no kickoff_time")
        return None

    mgw["kickoff_dt"] = pd.to_datetime(mgw["kickoff_time"], utc=True, errors="coerce")
    mgw["was_home"] = mgw["was_home"].map(
        {True: True, False: False, "True": True, "False": False, 1: True, 0: False}
    )

    # Load team strengths
    team_str = load_team_strength(season)
    if not team_str:
        print(f"  {season}: no teams.csv or no strength column")
        return None

    # Map team names to IDs
    teams_path = DATA_DIR / season / "teams.csv"
    teams_df = pd.read_csv(teams_path)
    name_to_id = dict(zip(teams_df["name"], teams_df["id"]))

    # Need team column — ensure it maps to IDs
    if "team" in mgw.columns:
        # Try numeric first
        team_numeric = pd.to_numeric(mgw["team"], errors="coerce")
        if team_numeric.notna().mean() < 0.5:
            # String names — map to IDs
            mgw["team_id"] = mgw["team"].map(name_to_id)
        else:
            mgw["team_id"] = team_numeric
    else:
        mgw["team_id"] = np.nan

    gw_deadlines = mgw.groupby("GW")["kickoff_dt"].min().sort_index()

    # Per-fixture info for strength diff
    fixture_info = mgw[["element", "GW", "opponent_team", "team_id"]].drop_duplicates()

    results = []
    for gw in sorted(mgw["GW"].unique()):
        if gw < 2:
            continue
        if gw not in gw_deadlines.index:
            continue

        deadline = gw_deadlines[gw]
        form = compute_form_30d(mgw, gw, deadline)

        # Get xP per (element, GW) — mean for DGW
        gw_data = mgw[mgw["GW"] == gw]

        # Per-fixture EP computation for DGW support
        for eid in gw_data["element"].unique():
            player_fixtures = gw_data[gw_data["element"] == eid]
            xp_actual = player_fixtures["xP"].mean()  # vaastav stores same xP per fixture
            actual_pts = player_fixtures["total_points"].sum()
            actual_mins = player_fixtures["minutes"].sum()

            if pd.isna(xp_actual):
                continue

            player_form = form.get(eid, 0.0)
            if pd.isna(player_form):
                player_form = 0.0

            # Compute EP per fixture and sum
            ep_sum = 0.0
            n_fix = 0
            for _, frow in player_fixtures.drop_duplicates("fixture").iterrows():
                opp_id = frow["opponent_team"]
                own_id = frow["team_id"]

                if pd.notna(own_id) and pd.notna(opp_id):
                    own_s = team_str.get(int(own_id), 3)
                    opp_s = team_str.get(int(opp_id), 3)
                    offset = (own_s - opp_s) * 0.5
                else:
                    offset = 0.0

                # cop = 100 (we don't have per-GW cop)
                ep_fix = round((player_form + offset) * 100 / 100, 1)
                ep_sum += ep_fix
                n_fix += 1

            if n_fix == 0:
                continue

            results.append({
                "element": eid, "GW": gw,
                "form": player_form, "n_fix": n_fix,
                "ep_computed": round(ep_sum, 1),
                "xP": round(xp_actual, 1),
                "actual_pts": actual_pts,
                "actual_mins": actual_mins,
            })

    if not results:
        return None

    df = pd.DataFrame(results)

    # Stats
    # Filter to xP > 0 (xP=0 could be cop=0 which we can't reconstruct)
    pos = df[df["xP"] > 0].copy()
    all_rows = df.copy()

    diff = (pos["ep_computed"] - pos["xP"]).abs()
    exact = (diff < 0.01).mean()
    within_05 = (diff <= 0.5).mean()
    within_10 = (diff <= 1.0).mean()
    corr_xp = pos["ep_computed"].corr(pos["xP"])
    mae_xp = diff.mean()

    # Correlation with actual points
    corr_pts_ep = pos["ep_computed"].corr(pos["actual_pts"])
    corr_pts_xp = pos["xP"].corr(pos["actual_pts"])

    # Per-GW correlation
    gw_corrs_ep = []
    gw_corrs_xp = []
    for gw in sorted(all_rows["GW"].unique()):
        gd = all_rows[all_rows["GW"] == gw]
        if len(gd) > 20:
            c1 = gd["ep_computed"].corr(gd["actual_pts"])
            c2 = gd["xP"].corr(gd["actual_pts"])
            if not np.isnan(c1):
                gw_corrs_ep.append(c1)
            if not np.isnan(c2):
                gw_corrs_xp.append(c2)

    # How many xP=0 do we correctly get?
    zeros = all_rows[all_rows["xP"] == 0]
    our_zeros = all_rows[all_rows["ep_computed"] == 0]
    # xP=0 and we say 0
    both_zero = ((all_rows["xP"] == 0) & (all_rows["ep_computed"] == 0)).sum()
    # xP>0 but we say 0 (false negative — we miss that they'll play)
    false_neg = ((all_rows["xP"] > 0) & (all_rows["ep_computed"] == 0)).sum()
    # xP=0 but we say >0 (false positive — we think they'll play but cop=0)
    false_pos = ((all_rows["xP"] == 0) & (all_rows["ep_computed"] > 0)).sum()

    return {
        "season": season,
        "total_rows": len(all_rows),
        "xp_pos_rows": len(pos),
        "exact": exact,
        "within_05": within_05,
        "within_10": within_10,
        "corr_xp": corr_xp,
        "mae_xp": mae_xp,
        "corr_pts_ep": corr_pts_ep,
        "corr_pts_xp": corr_pts_xp,
        "gw_corr_ep": np.mean(gw_corrs_ep) if gw_corrs_ep else np.nan,
        "gw_corr_xp": np.mean(gw_corrs_xp) if gw_corrs_xp else np.nan,
        "n_xp_zero": len(zeros),
        "both_zero": both_zero,
        "false_neg": false_neg,
        "false_pos": false_pos,
    }


def main():
    seasons = ["2020-21", "2021-22", "2022-23", "2023-24", "2024-25"]

    print("=" * 90)
    print("EP FORMULA VERIFICATION: ep = round((form + (team_str - opp_str)*0.5) * cop/100, 1)")
    print("form = round_half(mean(pts for games with mins>0 in last 30 days))")
    print("cop = 100 (assumed — we don't have per-GW chance_of_playing)")
    print("=" * 90)

    all_results = []
    for season in seasons:
        print(f"\nProcessing {season}...")
        r = verify_season(season)
        if r:
            all_results.append(r)

    if not all_results:
        print("No results!")
        return

    # Print results
    print("\n" + "=" * 90)
    print("RESULTS: HOW WELL DOES OUR FORMULA MATCH xP? (among xP > 0 rows)")
    print("=" * 90)
    print(f"{'Season':<10} {'Rows':>6} {'Exact':>7} {'<0.5':>7} {'<1.0':>7} {'r(xP)':>7} {'MAE':>7}")
    print("-" * 55)
    for r in all_results:
        print(f"{r['season']:<10} {r['xp_pos_rows']:>6} {r['exact']:>6.1%} {r['within_05']:>6.1%} {r['within_10']:>6.1%} {r['corr_xp']:>7.4f} {r['mae_xp']:>7.3f}")

    # Averages
    avg_exact = np.mean([r["exact"] for r in all_results])
    avg_05 = np.mean([r["within_05"] for r in all_results])
    avg_10 = np.mean([r["within_10"] for r in all_results])
    avg_corr = np.mean([r["corr_xp"] for r in all_results])
    avg_mae = np.mean([r["mae_xp"] for r in all_results])
    print(f"{'AVERAGE':<10} {'':>6} {avg_exact:>6.1%} {avg_05:>6.1%} {avg_10:>6.1%} {avg_corr:>7.4f} {avg_mae:>7.3f}")

    print("\n" + "=" * 90)
    print("PREDICTION QUALITY: PER-GW CORRELATION WITH ACTUAL POINTS")
    print("=" * 90)
    print(f"{'Season':<10} {'Our EP':>10} {'vaastav xP':>12} {'Gap':>8}")
    print("-" * 42)
    for r in all_results:
        gap = r["gw_corr_xp"] - r["gw_corr_ep"]
        print(f"{r['season']:<10} {r['gw_corr_ep']:>10.4f} {r['gw_corr_xp']:>12.4f} {gap:>+8.4f}")

    avg_ep = np.mean([r["gw_corr_ep"] for r in all_results])
    avg_xp = np.mean([r["gw_corr_xp"] for r in all_results])
    print(f"{'AVERAGE':<10} {avg_ep:>10.4f} {avg_xp:>12.4f} {avg_xp-avg_ep:>+8.4f}")

    print("\n" + "=" * 90)
    print("CHANCE_OF_PLAYING GAP: xP=0 vs our EP")
    print("=" * 90)
    print(f"{'Season':<10} {'xP=0':>7} {'Both=0':>8} {'FalseNeg':>9} {'FalsePos':>9}")
    print(f"{'':>10} {'':>7} {'(correct)':>8} {'(we=0,xP>0)':>9} {'(we>0,xP=0)':>9}")
    print("-" * 50)
    for r in all_results:
        print(f"{r['season']:<10} {r['n_xp_zero']:>7} {r['both_zero']:>8} {r['false_neg']:>9} {r['false_pos']:>9}")

    total_fp = sum(r["false_pos"] for r in all_results)
    total_xp0 = sum(r["n_xp_zero"] for r in all_results)
    print(f"\nFalse positives = we predict EP>0 but xP=0 (player was marked unavailable)")
    print(f"Total across all seasons: {total_fp}/{total_xp0} ({100*total_fp/total_xp0:.1f}% of xP=0 rows)")
    print(f"This is the chance_of_playing gap — we assume cop=100 for everyone.")


if __name__ == "__main__":
    main()
