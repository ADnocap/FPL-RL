#!/usr/bin/env python3
"""Try many approaches to reconstruct FPL's ep_this and match xP."""

import pandas as pd
import numpy as np
from pathlib import Path

def round_half(x):
    return np.round(x * 2) / 2

def main():
    mgw = pd.read_csv('data/raw/2024-25/gws/merged_gw.csv', encoding='utf-8')
    for c in ['xP','total_points','minutes','element','GW','opponent_team','value','selected']:
        mgw[c] = pd.to_numeric(mgw[c], errors='coerce')
    mgw['kickoff_dt'] = pd.to_datetime(mgw['kickoff_time'], utc=True)
    mgw['was_home'] = mgw['was_home'].map({True:True,False:False,'True':True,'False':False})

    teams = pd.read_csv('data/raw/2024-25/teams.csv')
    team_name_to_id = dict(zip(teams['name'], teams['id']))
    team_strength = teams.set_index('id').to_dict('index')

    # Load prior season PPG for form=0 fallback
    try:
        prev_mgw = pd.read_csv('data/raw/2023-24/gws/merged_gw.csv', encoding='utf-8')
        prev_mgw['total_points'] = pd.to_numeric(prev_mgw['total_points'], errors='coerce')
        prev_mgw['element'] = pd.to_numeric(prev_mgw['element'], errors='coerce')
        # Map 2023-24 element IDs to 2024-25 via cleaned_players
        # For now, use code-based mapping
        prev_ppg = prev_mgw.groupby('element')['total_points'].mean()
    except Exception:
        prev_ppg = pd.Series(dtype=float)

    # Players_raw for price-based form proxy
    try:
        pr = pd.read_csv('data/raw/2024-25/players_raw.csv')
        pr_ppg = dict(zip(pr['id'], pd.to_numeric(pr['points_per_game'], errors='coerce')))
        pr_form = dict(zip(pr['id'], pd.to_numeric(pr['form'], errors='coerce')))
    except Exception:
        pr_ppg = {}
        pr_form = {}

    gw_deadlines = mgw.groupby('GW')['kickoff_dt'].min().sort_index()
    fixture_counts = mgw.groupby(['element','GW'])['fixture'].nunique().reset_index()
    fixture_counts.columns = ['element','GW','n_fixtures']
    fixture_info = mgw[['element','GW','fixture','opponent_team','was_home','team']].drop_duplicates()

    # ---- Build per (element, GW) features ----
    results = []
    for gw in sorted(mgw['GW'].unique()):
        deadline = gw_deadlines[gw]
        cutoff30 = deadline - pd.Timedelta(days=30)

        prev_all = mgw[mgw['GW'] < gw]
        prev_30d = prev_all[prev_all['kickoff_dt'] >= cutoff30]

        gw_data = mgw[mgw['GW'] == gw]
        gw_agg = gw_data.groupby('element').agg({
            'xP': 'mean', 'total_points': 'sum', 'minutes': 'sum'
        }).reset_index()

        fc = fixture_counts[fixture_counts['GW'] == gw].set_index('element')['n_fixtures']

        for _, row in gw_agg.iterrows():
            eid = row['element']
            xp_val = row['xP']
            actual_pts = row['total_points']
            actual_mins = row['minutes']
            nfix = fc.get(eid, 1)

            # --- Form variants ---
            # 1. 30-day form (all entries including 0-min)
            p30 = prev_30d[prev_30d['element'] == eid]
            form_30d_all = round_half(p30['total_points'].mean()) if len(p30) > 0 else 0.0

            # 2. 30-day form (played only, mins>0)
            p30_played = p30[p30['minutes'] > 0]
            form_30d_played = round_half(p30_played['total_points'].mean()) if len(p30_played) > 0 else 0.0

            # 3. Rolling 5 GW form (our current approach)
            p_prev = prev_all[prev_all['element'] == eid].sort_values('GW')
            last5 = p_prev.tail(5)
            form_r5 = round_half(last5['total_points'].mean()) if len(last5) > 0 else 0.0

            # 4. Season PPG so far
            season_ppg = round_half(p_prev['total_points'].mean()) if len(p_prev) > 0 else 0.0

            # --- Playing probability variants ---
            last3 = p_prev.tail(3)
            # A: from mins (our current)
            pp_mins = min(last3['minutes'].mean() / 90.0, 1.0) if len(last3) > 0 else 0.0
            # B: binary - did they play ANY of last 3?
            pp_any3 = float((last3['minutes'] > 0).any()) if len(last3) > 0 else 0.0
            # C: appeared in last GW at all? (even 0 mins = was in squad data)
            last1 = p_prev.tail(1)
            pp_recent = 1.0 if len(last1) > 0 and last1.iloc[-1]['GW'] >= gw - 2 else 0.0
            # D: "ideal" - use current GW minutes as oracle
            pp_oracle = 1.0 if actual_mins > 0 else 0.0

            # --- Form=0 fallback ---
            # Use end-of-season PPG from players_raw as proxy for pre-season estimate
            ppg_fallback = round_half(pr_ppg.get(eid, 0.0))

            # --- Strength diff ---
            fix_rows = fixture_info[(fixture_info['element']==eid) & (fixture_info['GW']==gw)]
            sdiffs = []
            for _, fr in fix_rows.iterrows():
                tid = team_name_to_id.get(fr['team'])
                oid = int(fr['opponent_team']) if pd.notna(fr['opponent_team']) else None
                ih = fr['was_home']
                if tid and oid and tid in team_strength and oid in team_strength:
                    own = team_strength[tid]
                    opp = team_strength[oid]
                    o_s = own.get('strength_overall_home' if ih else 'strength_overall_away', 1100)
                    p_s = opp.get('strength_overall_away' if ih else 'strength_overall_home', 1100)
                    sdiffs.append(o_s - p_s)
            sdiff = np.mean(sdiffs) if sdiffs else 0

            results.append({
                'element': eid, 'GW': gw, 'xP': xp_val,
                'actual_pts': actual_pts, 'actual_mins': actual_mins,
                'n_fix': nfix, 'sdiff': sdiff,
                'form_30d': form_30d_all, 'form_30d_played': form_30d_played,
                'form_r5': form_r5, 'season_ppg': season_ppg,
                'pp_mins': pp_mins, 'pp_any3': pp_any3,
                'pp_recent': pp_recent, 'pp_oracle': pp_oracle,
                'ppg_fallback': ppg_fallback,
            })

    df = pd.DataFrame(results)
    pos = df.copy()  # use ALL rows including xP=0

    # Fixture factor (additive offset, quantized)
    q_offset = (round_half(pos['sdiff'] / 200)).clip(-1, 1)

    # ---- Build many synthetic formulas ----
    formulas = {}

    # Baseline: our current best
    formulas['A: form_30d * n_fix'] = round_half(pos['form_30d']) * pos['n_fix']

    # With fixture offset
    formulas['B: (form_30d + offset) * n_fix'] = round_half(pos['form_30d'] + q_offset) * pos['n_fix']

    # With fallback for form=0
    form_fb = pos['form_30d'].copy()
    form_fb[form_fb == 0] = pos.loc[form_fb == 0, 'ppg_fallback']
    formulas['C: (form|fallback + offset) * n_fix'] = round_half(form_fb + q_offset) * pos['n_fix']

    # With playing probability from minutes
    formulas['D: (form|fb + offset) * pp_mins * n_fix'] = (
        round_half(form_fb + q_offset) * pos['pp_mins'] * pos['n_fix']
    )

    # With binary playing (any of last 3)
    formulas['E: (form|fb + offset) * pp_any3 * n_fix'] = (
        round_half(form_fb + q_offset) * pos['pp_any3'] * pos['n_fix']
    )

    # With recent appearance
    formulas['F: (form|fb + offset) * pp_recent * n_fix'] = (
        round_half(form_fb + q_offset) * pos['pp_recent'] * pos['n_fix']
    )

    # ORACLE: use actual minutes to determine if played (THIS IS LOOKAHEAD)
    formulas['G*: (form|fb + offset) * pp_ORACLE * n_fix'] = (
        round_half(form_fb + q_offset) * pos['pp_oracle'] * pos['n_fix']
    )

    # Form_played variant with oracle
    form_pl_fb = pos['form_30d_played'].copy()
    form_pl_fb[form_pl_fb == 0] = pos.loc[form_pl_fb == 0, 'ppg_fallback']
    formulas['H*: (form_played|fb + offset) * pp_ORACLE * n_fix'] = (
        round_half(form_pl_fb + q_offset) * pos['pp_oracle'] * pos['n_fix']
    )

    # Rolling 5 with oracle
    form_r5_fb = pos['form_r5'].copy()
    form_r5_fb[form_r5_fb == 0] = pos.loc[form_r5_fb == 0, 'ppg_fallback']
    formulas['I*: (form_r5|fb + offset) * pp_ORACLE * n_fix'] = (
        round_half(form_r5_fb + q_offset) * pos['pp_oracle'] * pos['n_fix']
    )

    # What if we use season PPG as form with oracle?
    ppg_fb = pos['season_ppg'].copy()
    ppg_fb[ppg_fb == 0] = pos.loc[ppg_fb == 0, 'ppg_fallback']
    formulas['J*: (season_ppg|fb + offset) * pp_ORACLE * n_fix'] = (
        round_half(ppg_fb + q_offset) * pos['pp_oracle'] * pos['n_fix']
    )

    # ---- Evaluate ----
    print("=" * 100)
    print("MATCHING xP: FORMULA COMPARISON (* = uses oracle/lookahead)")
    print("=" * 100)
    print(f"{'Formula':<52} {'r(xP)':>7} {'MAE(xP)':>8} {'r(pts)':>7} {'GW_r(pts)':>10} {'<0.5':>6} {'<1.0':>6}")
    print("-" * 100)

    for name, synth in formulas.items():
        corr_xp = synth.corr(pos['xP'])
        mae_xp = (synth - pos['xP']).abs().mean()
        corr_pts = synth.corr(pos['actual_pts'])

        # Per-GW correlation with actual
        gw_corrs = []
        for gw in sorted(pos['GW'].unique()):
            gd = pos[pos['GW'] == gw]
            if len(gd) > 20:
                c = pd.Series(synth.values[pos['GW'] == gw]).reset_index(drop=True).corr(
                    gd['actual_pts'].reset_index(drop=True)
                )
                if not np.isnan(c):
                    gw_corrs.append(c)
        mean_gw = np.mean(gw_corrs) if gw_corrs else 0

        diff = (synth - pos['xP']).abs()
        w05 = (diff <= 0.5).mean()
        w10 = (diff <= 1.0).mean()

        print(f"  {name:<50} {corr_xp:>7.4f} {mae_xp:>8.3f} {corr_pts:>7.4f} {mean_gw:>10.4f} {w05:>5.1%} {w10:>5.1%}")

    # Reference
    xp_corr_pts = pos['xP'].corr(pos['actual_pts'])
    gw_corrs_xp = []
    for gw in sorted(pos['GW'].unique()):
        gd = pos[pos['GW'] == gw]
        if len(gd) > 20:
            c = gd['xP'].corr(gd['actual_pts'])
            if not np.isnan(c):
                gw_corrs_xp.append(c)
    xp_gw = np.mean(gw_corrs_xp)
    print(f"\n  {'xP (reference)':<50} {'1.000':>7} {'0.000':>8} {xp_corr_pts:>7.4f} {xp_gw:>10.4f}")

    # ---- Breakdown: what fraction of xP's advantage comes from chance_of_playing? ----
    print("\n" + "=" * 100)
    print("BREAKDOWN: WHERE DOES xP's ADVANTAGE COME FROM?")
    print("=" * 100)

    # Split by whether player actually played
    played = pos[pos['actual_mins'] > 0]
    sat = pos[pos['actual_mins'] == 0]
    print(f"Played: {len(played)} rows, Sat out: {len(sat)} rows")

    # Among players who PLAYED, how does our form compare to xP?
    synth_best = round_half(form_fb + q_offset) * pos['n_fix']  # formula C
    print(f"\nAmong PLAYED players:")
    print(f"  xP corr w/ actual:    {played['xP'].corr(played['actual_pts']):.4f}")
    c_played = synth_best[played.index].corr(played['actual_pts'])
    print(f"  Synth C corr w/ pts:  {c_played:.4f}")

    print(f"\nAmong SAT OUT players (mins=0):")
    print(f"  xP mean:    {sat['xP'].mean():.3f}")
    print(f"  Synth mean: {synth_best[sat.index].mean():.3f}")
    xp_zero_pct = (sat['xP'] == 0).mean()
    synth_zero_pct = (synth_best[sat.index] == 0).mean()
    print(f"  xP=0 rate:    {xp_zero_pct:.1%}")
    print(f"  Synth=0 rate: {synth_zero_pct:.1%}")


if __name__ == "__main__":
    main()
