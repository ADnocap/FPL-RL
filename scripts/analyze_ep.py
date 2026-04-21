#!/usr/bin/env python3
"""Reverse-engineer FPL's ep_this formula and test synthetic versions."""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression

def round_half(x):
    """Round to nearest 0.5 like FPL does."""
    return np.round(x * 2) / 2

def main():
    mgw = pd.read_csv('data/raw/2024-25/gws/merged_gw.csv', encoding='utf-8')
    for c in ['xP','total_points','minutes','element','GW','opponent_team']:
        mgw[c] = pd.to_numeric(mgw[c], errors='coerce')
    mgw['kickoff_dt'] = pd.to_datetime(mgw['kickoff_time'], utc=True)
    mgw['was_home'] = mgw['was_home'].map(
        {True:True, False:False, 'True':True, 'False':False}
    )

    teams = pd.read_csv('data/raw/2024-25/teams.csv')
    team_strength = teams.set_index('id').to_dict('index')
    team_name_to_id = dict(zip(teams['name'], teams['id']))

    fixture_counts = mgw.groupby(['element','GW'])['fixture'].nunique().reset_index()
    fixture_counts.columns = ['element','GW','n_fixtures']

    fixture_info = mgw[['element','GW','fixture','opponent_team','was_home','team']].drop_duplicates()

    gw_deadlines = mgw.groupby('GW')['kickoff_dt'].min().sort_index()

    # ---- Compute form and strength diff per (element, GW) ----
    results = []
    for gw in sorted(mgw['GW'].unique()):
        if gw < 2:
            continue
        deadline = gw_deadlines[gw]
        cutoff = deadline - pd.Timedelta(days=30)
        prev_data = mgw[(mgw['GW'] < gw) & (mgw['kickoff_dt'] >= cutoff)]

        if prev_data.empty:
            continue

        player_form = prev_data.groupby('element')['total_points'].mean()
        player_form_rounded = round_half(player_form)

        # Also compute form from only PLAYED games (mins > 0)
        played = prev_data[prev_data['minutes'] > 0]
        player_form_played = played.groupby('element')['total_points'].mean() if not played.empty else pd.Series(dtype=float)
        player_form_played_rounded = round_half(player_form_played)

        gw_data = mgw[mgw['GW'] == gw]
        gw_xp = gw_data.groupby('element')['xP'].mean()
        fc = fixture_counts[fixture_counts['GW'] == gw].set_index('element')['n_fixtures']

        for eid in gw_xp.index:
            if pd.isna(gw_xp[eid]):
                continue

            xp_val = gw_xp[eid]
            form_all = player_form_rounded.get(eid, 0.0)
            form_played = player_form_played_rounded.get(eid, 0.0)
            nfix = fc.get(eid, 1)

            # Strength diff per fixture
            fix_rows = fixture_info[(fixture_info['element']==eid) & (fixture_info['GW']==gw)]
            strength_diffs = []
            for _, frow in fix_rows.iterrows():
                team_name = frow['team']
                opp_id = int(frow['opponent_team']) if pd.notna(frow['opponent_team']) else None
                is_home = frow['was_home']
                team_id = team_name_to_id.get(team_name)
                if team_id and opp_id and team_id in team_strength and opp_id in team_strength:
                    own = team_strength[team_id]
                    opp = team_strength[opp_id]
                    if is_home:
                        own_str = own.get('strength_overall_home', 1100)
                        opp_str = opp.get('strength_overall_away', 1100)
                    else:
                        own_str = own.get('strength_overall_away', 1100)
                        opp_str = opp.get('strength_overall_home', 1100)
                    strength_diffs.append(own_str - opp_str)

            # Minutes-based playing prob
            recent = prev_data[prev_data['element'] == eid].sort_values('GW').tail(3)
            avg_mins = recent['minutes'].mean() if not recent.empty else 0
            play_prob = min(avg_mins / 90.0, 1.0)

            results.append({
                'element': eid, 'GW': gw,
                'form_all': form_all,
                'form_played': form_played if not pd.isna(form_played) else 0.0,
                'xP': xp_val, 'n_fixtures': nfix,
                'strength_diff': np.mean(strength_diffs) if strength_diffs else 0,
                'play_prob': play_prob,
            })

    df = pd.DataFrame(results)

    # ---- Test different formulas ----
    # Filter to xP > 0 for comparison
    pos = df[df['xP'] > 0].copy()

    print("=" * 70)
    print("REVERSE-ENGINEERING FPL ep_this")
    print("=" * 70)
    print(f"Total rows with xP>0: {len(pos)}")

    # What form variant matches xP best?
    print("\n=== FORM VARIANT CORRELATION WITH xP ===")
    for form_col in ['form_all', 'form_played']:
        c = pos[form_col].corr(pos['xP'])
        print(f"  {form_col}: r = {c:.4f}")

    # For SGW healthy players (1 fixture, play_prob > 0.8)
    sgw = pos[(pos['n_fixtures'] == 1) & (pos['play_prob'] > 0.8)].copy()
    print(f"\nSGW healthy players: {len(sgw)}")

    # Implied fixture factor
    sgw['implied_factor'] = sgw['xP'] / sgw['form_all'].replace(0, np.nan)
    valid_factor = sgw.dropna(subset=['implied_factor'])
    valid_factor = valid_factor[np.isfinite(valid_factor['implied_factor'])]

    print(f"\nImplied fixture factor stats:")
    print(f"  mean: {valid_factor['implied_factor'].mean():.4f}")
    print(f"  median: {valid_factor['implied_factor'].median():.4f}")
    print(f"  std: {valid_factor['implied_factor'].std():.4f}")

    # Regression: factor = a + b * strength_diff
    X = valid_factor[['strength_diff']].values
    y = valid_factor['implied_factor'].values
    lr = LinearRegression().fit(X, y)
    print(f"\n  Regression: factor = {lr.intercept_:.4f} + {lr.coef_[0]:.6f} * strength_diff")
    print(f"  R² = {lr.score(X, y):.4f}")

    # ---- Test synthetic formulas ----
    print("\n" + "=" * 70)
    print("SYNTHETIC EP FORMULAS")
    print("=" * 70)

    # Formula 1: form_all * n_fixtures
    pos['s1'] = round_half(pos['form_all']) * pos['n_fixtures']
    # Formula 2: form_all * learned_factor * n_fixtures
    pos['learned_f'] = lr.predict(pos[['strength_diff']].values).clip(0.7, 1.3)
    pos['s2'] = round_half(pos['form_all'] * pos['learned_f']) * pos['n_fixtures']
    # Formula 3: with floor and play_prob
    pos['s3'] = round_half(pos['form_all'].clip(lower=0.5) * pos['learned_f']) * pos['n_fixtures'] * pos['play_prob']
    # Formula 4: form_played instead of form_all
    pos['s4'] = round_half(pos['form_played'].clip(lower=0.5) * pos['learned_f']) * pos['n_fixtures']
    # Formula 5: form_all + fixture_offset (additive not multiplicative)
    offset = (pos['strength_diff'] / 200).clip(-1, 1)
    pos['s5'] = round_half(pos['form_all'] + offset) * pos['n_fixtures']
    # Formula 6: (form_all + quantized_offset) * play_prob * n_fix
    q_offset = (round_half(pos['strength_diff'] / 200)).clip(-1, 1)
    pos['s6'] = round_half(pos['form_all'] + q_offset) * pos['n_fixtures']

    formulas = {
        'S1: form * n_fix': 's1',
        'S2: form * learned_f * n_fix': 's2',
        'S3: floored * learned_f * play_prob * n_fix': 's3',
        'S4: form_played * learned_f * n_fix': 's4',
        'S5: (form + offset) * n_fix': 's5',
        'S6: round(form + q_offset) * n_fix': 's6',
    }

    # Get actual points for per-GW correlation
    gw_actual = mgw.groupby(['element','GW']).agg({'total_points':'sum'}).reset_index()
    pos = pos.merge(gw_actual, on=['element','GW'], how='left')

    print(f"\n{'Formula':<45} {'corr_xP':>8} {'MAE_xP':>8} {'corr_pts':>9} {'exact':>7} {'<0.5':>7} {'<1.0':>7}")
    print("-" * 95)

    for name, col in formulas.items():
        corr_xp = pos[col].corr(pos['xP'])
        mae_xp = (pos[col] - pos['xP']).abs().mean()
        corr_pts = pos[col].corr(pos['total_points'])

        diff = (pos[col] - pos['xP']).abs()
        exact = (diff < 0.01).mean()
        within_half = (diff <= 0.5).mean()
        within_1 = (diff <= 1.0).mean()

        print(f"  {name:<43} {corr_xp:>8.4f} {mae_xp:>8.3f} {corr_pts:>9.4f} {exact:>6.1%} {within_half:>6.1%} {within_1:>6.1%}")

    # Also show xP's correlation with actual points
    corr_xp_pts = pos['xP'].corr(pos['total_points'])
    print(f"\n  {'xP (reference)':<43} {'1.0000':>8} {'0.000':>8} {corr_xp_pts:>9.4f}")

    # ---- Per-GW correlation with actual points ----
    print("\n" + "=" * 70)
    print("PER-GW CORRELATION WITH ACTUAL POINTS")
    print("=" * 70)

    gw_results = {}
    for gw in sorted(pos['GW'].unique()):
        gd = pos[pos['GW'] == gw]
        if len(gd) < 20:
            continue
        row = {'GW': gw}
        row['xP'] = gd['xP'].corr(gd['total_points'])
        for name, col in formulas.items():
            short = name.split(':')[0]
            row[short] = gd[col].corr(gd['total_points'])
        gw_results[gw] = row

    gw_df = pd.DataFrame(gw_results.values())
    means = gw_df.drop(columns='GW').mean()
    print(f"\nMean per-GW correlation:")
    for col, val in means.items():
        print(f"  {col:<10} {val:.4f}")

    # ---- What's missing? Check form=0 players ----
    print("\n" + "=" * 70)
    print("PLAYERS WITH form=0 BUT xP > 0")
    print("=" * 70)
    zero_form = pos[pos['form_all'] == 0]
    print(f"Count: {len(zero_form)} ({len(zero_form)/len(pos):.1%} of total)")
    if len(zero_form) > 0:
        print(f"xP distribution: mean={zero_form['xP'].mean():.2f}, max={zero_form['xP'].max():.1f}")
        print(f"These are likely GW1/returning/new players where FPL uses a pre-season estimate")
        print(f"\nTop 10:")
        print(zero_form.nlargest(10, 'xP')[['element','GW','form_all','form_played','xP','n_fixtures','play_prob']].to_string(index=False))


if __name__ == "__main__":
    main()
