---
name: gameweek
description: Run the weekly FPL pre-deadline routine — refresh live data, predict the upcoming GW, optimize transfers/lineup/captain for the user's real team, and present the recommendation. Use before each GW deadline or when the user asks "what should I do this gameweek".
---

# Weekly gameweek routine (live 2026-27 season)

## Steps

1. **Check the deadline first.** `GET https://fantasy.premierleague.com/api/bootstrap-static/`
   → events → the `is_next` event's `deadline_time`. Tell the user how long is left.
   If < 15 minutes, run with `--skip-refresh` (saves ~10 min of element downloads).

2. **Run the driver** (team id: ask the user once, then store it in `.env` as
   `FPL_TEAM_ID` and reuse):
   ```
   python scripts/gameweek.py --team-id <FPL_TEAM_ID>
   ```
   - GW1 or wildcard-from-scratch: add `--fresh-squad`
   - Chip evaluation: re-run with `--chip wildcard` / `free_hit` / `bench_boost` /
     `triple_captain` and compare objective values against the no-chip run
   - Sanity check: re-run with `--ep` and compare — large disagreements between the
     model and FPL's EP deserve a look (usually injury/rotation news the model
     can't see).

3. **Review before presenting**:
   - Flag any recommended player with `status` d/i/s in bootstrap or
     `chance_of_playing_next_round < 100` (news field says why).
   - Check `transfers made > free_transfers` → mention the hit cost explicitly.
   - DGW/BGW: check fixture counts for GW in `data/raw/2026-27/fixtures.csv`.

4. **Present**: transfers (out→in with prices), XI + formation, captain/vice,
   bench order, expected points, hit cost if any. Remind the user to apply on
   fantasy.premierleague.com before the deadline (or via the API automation in
   scripts/apply_team.py if it exists and they've set their session cookie).

## Failure modes

- Model predictions empty → season files not rebuilt; run without `--skip-build`.
- `Entry has no completed gameweek` → GW1 hasn't finished; use `--fresh-squad`.
- Element summaries stale mid-GW → fine; rolling features only use completed GWs.
- After the GW completes, scores are final 09:00 UK the NEXT day (2026-27
  lockdown rule) — don't rebuild training data before that.

## Weekly cadence (already in the user's Google Calendar, green events)

Snapshot + recommendation ideally the evening before the deadline, final check
1-2h before. Deadlines vary: Fri 19:30, Sat 12:00-15:30, some midweek Wed —
always read `deadline_time` from the API, never assume.
