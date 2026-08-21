# 2026-27 Season Operations Guide

The repo runs live this season. This is the operational handbook.

## Weekly loop (before every deadline — calendar events are already set)

```bash
# The one command (computes transfers + lineup + captain for your real team):
python scripts/gameweek.py --team-id <YOUR_ENTRY_ID>

# Then either apply manually on the site, or fully via API:
python scripts/gameweek.py --team-id <YOUR_ENTRY_ID> --apply          # dry-run validation
python scripts/gameweek.py --team-id <YOUR_ENTRY_ID> --apply --yes    # commit for real
```

Timing: run it the evening before the deadline (post-press-conference news is in
`chance_of_playing`), sanity-check again 1-2h before. If very close to the
deadline, add `--skip-refresh` (saves ~10 min of element-summary downloads).

Chip evaluation: re-run with `--chip wildcard|free_hit|bench_boost|triple_captain`
and compare objective values. Sanity check vs FPL's own EP: `--ep`.

## One-time setup remaining (you)

1. **Your FPL team ID**: after GW1, from the Points page URL
   (`/entry/<ID>/event/1`). Put it in `.env` as `FPL_TEAM_ID=<id>` for reference.
2. **API write access** (so you never open the app): log in at
   fantasy.premierleague.com, DevTools Console:
   ```js
   copy(JSON.parse(localStorage.getItem(Object.keys(localStorage).find(k=>k.startsWith('oidc.user:')))).refresh_token)
   ```
   Paste into `.env` as `FPL_REFRESH_TOKEN=...`. Token rotation is handled
   automatically (`src/fpl_rl/live/auth.py`). If refresh ever 400/401s,
   re-extract from a fresh browser login.
   *Caveat: PL competition T&Cs void "script-generated entries" (prize
   eligibility, not bans — no documented ban for automating your own account).
   Keep volumes tiny; the site UI always shows what was submitted.*
3. **(Optional) Odds**: free account at the-odds-api.com (Starter, 500
   credits/mo — a live EPL h2h snapshot costs 1 credit/GW). Key → `.env`
   `ODDS_API_KEY`. Your own ablation says odds add ~nothing; skip guilt-free.

## Data & retraining

- **2025-26 backfill: DONE** (vaastav complete season, understat league +
  per-match, FotMob 27110, football-data odds 380/380).
- **Model of record**: `models/prod_2026-27` — produced by
  `python scripts/train_predictor.py` (10 seasons, includes fpl_xp; eval report
  in `training_report.json`). `gameweek.py` picks it up automatically.
- **Mid-season retrain** (~GW8, and at the January window): `/retrain` skill or
  `scripts/train_predictor.py` after collecting 2026-27 understat per-match:
  `python scripts/collect_data.py --sources understat --seasons 2026-27 --per-match`
- **Lockdown rule**: GW scores are final 09:00 UK the morning after the GW's
  last match — never rebuild training data before that.

## 2026/27 rules (verified against bootstrap + official articles)

- Scoring **unchanged** from 2025/26, incl. DEFCON (+2: DEF ≥10 CBIT; MID/FWD
  ≥12 CBIRT; GK never; capped at 2/match).
- **BPS internals changed** (bonus only): tackled −1 removed, CBI 1-per-3,
  GK saves 2 + inside-box +1 + big-chance +1, pen save 8→7. Bonus-sensitive
  historical patterns from ≤2025/26 are slightly off — retraining absorbs most.
- Chips 4×2: **WC/FH not playable GW1** (start GW2); BB/TC from GW1; one chip
  per GW; first set **expires at GW19 deadline (Sat 2 Jan 2027, 13:30 GMT)**;
  FH in GW19 blocks FH in GW20.
- FTs: bank to 5; on a WC/FH week the banked count carries **unchanged**;
  hits −4; max 20 transfers/GW (no cap on WC/FH). Sell = purchase +
  floor(appreciation/2). No AFCON FT top-up this season.
- Prices: locked until GW1 deadline, then ±£0.1m/day at 00:00 UK; NEW official
  Price Change Predictor page (updates every 15 min) — useful before buying.

## Chip plan (community consensus + our model)

| Chip | Window | Plan |
|------|--------|------|
| Bench Boost 1 | GW1-19 | GW1-2 only if bench is built to start; otherwise after WC1 |
| Triple Captain 1 | GW1-19 | Haaland home vs promoted side: GW3 (COV), GW7 (IPS), GW16 (HUL) |
| Wildcard 1 | GW2-19 | GW5-6 international break (5 GWs of minutes data) |
| Free Hit 1 | GW2-19 | Reactive — bad fixture week / injury pile-up |
| Second set | GW20-38 | Save for spring DGWs (BB on the biggest DGW, TC on a Haaland DGW) |

Don't hoard the first set — it dies at GW19.

## Meta notes for 2026/27

- **Salah has left Liverpool.** Haaland (£15.5m, ~70% owned) is the one
  near-mandatory premium; skipping him is a large rank-risk. Our model agrees:
  he's its highest-rated player (8.66 xPts GW1).
- Template GW1: Haaland + Bruno Fernandes + João Pedro; promoted teams
  (Coventry, Hull, Ipswich): attacking picks only (Leif Davis £4.0m the
  standout), never trust their clean sheets.
- Early season: no hits GW1-4, bank FTs, act after press conferences.

## Known gaps (next build targets)

1. **Multi-GW planning**: the MILP maximizes a single GW — no FT-banking value,
   no fixture-swing planning, no chip scheduler. Biggest optimizer upgrade.
2. **Bonus/BPS module**: 2026-27 BPS overhaul not modeled explicitly (only via
   realized totals in training data).
3. **Price-change modeling**: selling-price management is reactive, not planned.
4. RL agent: research path only; retrain only if it beats MILP on holdout.
