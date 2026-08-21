#!/usr/bin/env python3
"""Backfill historical player-prop odds for a season from The Odds API.

Anti-lookahead: every odds snapshot is taken at (kickoff - 2h), matching the
convention of the existing h2h odds data. Resume-safe: already-downloaded
events are skipped, so the script can be re-run after interruption without
re-spending credits. Cost: ~40 credits per event (4 markets x 1 region x 10)
+ 1 credit per GW events-list call — ~15.3K credits for a full season.

Usage:
    python scripts/collect_props.py --season 2025-26 [--markets ...]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

REPO_ROOT = Path(__file__).resolve().parent.parent
BASE = "https://api.the-odds-api.com/v4/historical/sports/soccer_epl"

DEFAULT_MARKETS = (
    "player_goal_scorer_anytime,player_shots_on_target,"
    "player_assists,player_to_receive_card"
)
CREDIT_FLOOR = 500  # stop before draining the plan completely


def _load_env() -> None:
    env = REPO_ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", default="2025-26")
    parser.add_argument("--markets", default=DEFAULT_MARKETS)
    parser.add_argument("--regions", default="us")
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    args = parser.parse_args()

    _load_env()
    key = os.environ.get("ODDS_API_KEY", "")
    if not key:
        raise SystemExit("ODDS_API_KEY missing from .env")

    fixtures = pd.read_csv(args.data_dir / "raw" / args.season / "fixtures.csv")
    fixtures = fixtures.dropna(subset=["event", "kickoff_time"])
    out_dir = args.data_dir / "props" / args.season
    out_dir.mkdir(parents=True, exist_ok=True)

    remaining: float = float("inf")

    def _get(url: str, params: dict) -> dict | None:
        nonlocal remaining
        resp = requests.get(url, params={"apiKey": key, **params}, timeout=30)
        rem = resp.headers.get("x-requests-remaining")
        if rem is not None:
            remaining = float(rem)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()

    # --- Per GW: one events-list snapshot at (first kickoff - 2h) ---
    total_saved = 0
    total_skipped = 0
    for gw, group in fixtures.groupby("event"):
        gw = int(gw)
        kickoffs = pd.to_datetime(group["kickoff_time"], utc=True).sort_values()
        list_ts = kickoffs.iloc[0].to_pydatetime() - timedelta(hours=2)

        events_index_path = out_dir / f"gw{gw}_events.json"
        if events_index_path.exists():
            events = json.loads(events_index_path.read_text(encoding="utf-8"))
        else:
            payload = _get(f"{BASE}/events", {"date": _iso(list_ts)})
            events = (payload or {}).get("data", [])
            events_index_path.write_text(
                json.dumps(events, indent=1), encoding="utf-8"
            )
            time.sleep(0.6)

        # Keep only events kicking off within this GW's window
        gw_start = kickoffs.iloc[0] - pd.Timedelta(hours=3)
        gw_end = kickoffs.iloc[-1] + pd.Timedelta(hours=3)
        gw_events = [
            e for e in events
            if gw_start <= pd.Timestamp(e["commence_time"]) <= gw_end
        ]
        print(f"GW{gw}: {len(gw_events)} events "
              f"(credits remaining ~{remaining:.0f})")

        for e in gw_events:
            if remaining < CREDIT_FLOOR:
                print(f"STOP: credit floor reached ({remaining:.0f} left). "
                      "Re-run after reset to resume.")
                print(f"Saved {total_saved}, skipped {total_skipped}")
                return
            dest = out_dir / f"gw{gw}_{e['id']}.json"
            if dest.exists():
                total_skipped += 1
                continue
            snap_ts = (
                pd.Timestamp(e["commence_time"]).to_pydatetime()
                - timedelta(hours=2)
            )
            odds = _get(
                f"{BASE}/events/{e['id']}/odds",
                {
                    "date": _iso(snap_ts),
                    "regions": args.regions,
                    "markets": args.markets,
                    "oddsFormat": "decimal",
                },
            )
            record = {
                "gw": gw,
                "event": e,
                "snapshot_requested": _iso(snap_ts),
                "odds": odds,
            }
            dest.write_text(json.dumps(record, indent=1), encoding="utf-8")
            total_saved += 1
            time.sleep(0.6)

    print(f"\nDone: {total_saved} events saved, {total_skipped} already present, "
          f"~{remaining:.0f} credits remaining")


if __name__ == "__main__":
    main()
