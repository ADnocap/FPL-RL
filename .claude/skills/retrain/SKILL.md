---
name: retrain
description: Retrain the FPL point-prediction model of record with the latest data, evaluate it honestly, and promote it for live use. Use mid-season (e.g. after ~GW8, at the January window, or when prediction quality degrades) or when the user asks to retrain/improve the model.
---

# Retrain the point predictor

## Steps

1. **Refresh current-season data** (needs all completed GWs on disk):
   ```
   python scripts/gameweek.py --fresh-squad   # cheap way to rebuild 2026-27 files
   python scripts/collect_data.py --sources understat --seasons 2026-27 --per-match
   ```
   Understat per-match data for 2026-27 feeds the rolling xG features.

2. **Retrain** (features built once; eval + production fit):
   ```
   python scripts/train_predictor.py --out models/prod_2026-27
   ```
   - Eval phase holds out 2025-26 → compare MAE/corr against the previous
     training_report.json before promoting.
   - Mid-season, consider adding completed 2026-27 GWs to the training set by
     extending PROD_SEASONS in the script.

3. **Promote**: `scripts/gameweek.py` automatically prefers `models/prod_2026-27/`.
   Keep the old model dir until the new one has beaten it on a full holdout.

## Quality bars (history)

- Old leaky-xP model: inflated (don't compare against it)
- no_xp (103 feats, ≤2023-24 train): MAE 0.993, per-GW corr 0.591 on 2024-25
- full_pregame (107 feats incl. legit fpl_xp): per-GW corr ~0.787 on 2024-25
- Anything materially worse than the current training_report.json = don't promote.

## Rules that bite

- fpl_xp is point-in-time SAFE (EP_FORMULA.md) — keep it; live snapshots capture it.
- Per-group rolling only (shift+groupby per element) — never global rolling.
- 2025-26+ includes DEFCON scoring — the target distribution shifted for
  DEF/MID; that's why 2025-26 must stay in the training set.
- FBref parquets don't exist for 2025-26 (Cloudflare) — FotMob fallback covers it;
  the NaN prior features are expected, not a bug.
