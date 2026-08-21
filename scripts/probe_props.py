#!/usr/bin/env python3
"""Probe The Odds API for EPL player-prop market availability.

Runs three escalating probes against the historical endpoints:

1. h2h odds snapshot at a known-good date  -> confirms historical access
2. historical events list for a 2024-25 matchday -> gets real event ids
3. per-market player-prop odds for one event at a pre-kickoff snapshot
   -> which prop markets exist and how many bookmakers quote them

Prints a GO/NO-GO summary with the projected credit cost of a full
2025-26 backfill, and saves a JSON report under runs/.

Requires ODDS_API_KEY in .env or environment.  Historical endpoints are
only available on PAID plans -- an invalid key or a free-plan key gets a
clear diagnostic, not a traceback.

Credit costs (historical): odds/event-odds = 10 x markets x regions per
request; events list = 1 per request.

Usage:
    python scripts/probe_props.py
    python scripts/probe_props.py --regions uk --skip-h2h
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

API_BASE = "https://api.the-odds-api.com/v4"
SPORT = "soccer_epl"

# Player-prop market keys to probe (candidate features for the predictor).
PROP_MARKETS = [
    "player_goal_scorer_anytime",
    "player_shots_on_target",
    "player_assists",
    "player_to_receive_card",
]

# Known-good historical snapshot (mid-season 2023-24, well inside coverage).
H2H_PROBE_DATE = "2024-01-15T12:00:00Z"
# 2024-25 GW4 Saturday, before the early kickoff (11:30 UTC).
EVENTS_PROBE_DATE = "2024-09-14T10:00:00Z"

# Full 2025-26 backfill assumptions.
BACKFILL_EVENTS = 380
BACKFILL_GWS = 38


def _load_dotenv() -> None:
    """Load .env file from project root if it exists."""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip()
            if key and key not in os.environ:
                os.environ[key] = value


class ProbeAuthError(Exception):
    """API rejected the key (invalid, expired, or plan lacks access)."""


def _get(url: str, params: dict, report: dict) -> dict | list:
    """GET with clear failure modes.  Tracks credit headers in `report`."""
    resp = requests.get(url, params=params, timeout=30)
    used = resp.headers.get("x-requests-used")
    remaining = resp.headers.get("x-requests-remaining")
    if used is not None:
        report["credits_used"] = used
        report["credits_remaining"] = remaining
    if resp.status_code == 401:
        try:
            message = resp.json().get("message", resp.text)
        except ValueError:
            message = resp.text
        raise ProbeAuthError(message)
    if resp.status_code != 200:
        try:
            message = resp.json().get("message", resp.text)
        except ValueError:
            message = resp.text
        raise RuntimeError(f"HTTP {resp.status_code}: {message}")
    return resp.json()


def probe_h2h_historical(api_key: str, report: dict) -> bool:
    """Step 1: confirm the key can read historical odds at all (10 credits)."""
    print(f"[1/3] Historical h2h snapshot at {H2H_PROBE_DATE} ...")
    data = _get(
        f"{API_BASE}/historical/sports/{SPORT}/odds",
        {
            "apiKey": api_key,
            "regions": "eu",
            "markets": "h2h",
            "date": H2H_PROBE_DATE,
        },
        report,
    )
    events = data.get("data", [])
    print(
        f"      OK: snapshot {data.get('timestamp')} with {len(events)} events "
        f"(credits used so far: {report.get('credits_used', '?')})"
    )
    report["h2h_historical"] = {
        "snapshot": data.get("timestamp"),
        "n_events": len(events),
    }
    return True


def probe_events(api_key: str, report: dict) -> list[dict]:
    """Step 2: historical events list for a 2024-25 matchday (1 credit)."""
    print(f"[2/3] Historical events list at {EVENTS_PROBE_DATE} ...")
    data = _get(
        f"{API_BASE}/historical/sports/{SPORT}/events",
        {"apiKey": api_key, "date": EVENTS_PROBE_DATE},
        report,
    )
    events = data.get("data", [])
    print(f"      OK: {len(events)} upcoming events at that snapshot")
    for ev in events[:5]:
        print(
            f"        {ev['commence_time']}  {ev['home_team']} vs "
            f"{ev['away_team']}  ({ev['id']})"
        )
    report["events"] = [
        {k: ev[k] for k in ("id", "commence_time", "home_team", "away_team")}
        for ev in events
    ]
    return events


def probe_prop_markets(
    api_key: str, event: dict, regions: str, report: dict
) -> dict[str, dict]:
    """Step 3: request each prop market for one event pre-kickoff.

    One request per market (same total cost as combined, but a market the
    API rejects with 422 INVALID_MARKET doesn't sink the others).
    """
    kickoff = datetime.fromisoformat(event["commence_time"].replace("Z", "+00:00"))
    snap = (kickoff - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(
        f"[3/3] Player props for {event['home_team']} vs {event['away_team']} "
        f"at {snap} (1h pre-kickoff), regions={regions} ..."
    )
    results: dict[str, dict] = {}
    for market in PROP_MARKETS:
        try:
            data = _get(
                f"{API_BASE}/historical/sports/{SPORT}/events/{event['id']}/odds",
                {
                    "apiKey": api_key,
                    "regions": regions,
                    "markets": market,
                    "date": snap,
                },
                report,
            )
        except ProbeAuthError:
            raise
        except RuntimeError as exc:
            print(f"      {market}: FAILED ({exc})")
            results[market] = {"available": False, "error": str(exc)}
            continue
        payload = data.get("data") or {}
        bookmakers = [
            bm["title"]
            for bm in payload.get("bookmakers", [])
            if any(m.get("key") == market for m in bm.get("markets", []))
        ]
        n_outcomes = sum(
            len(m.get("outcomes", []))
            for bm in payload.get("bookmakers", [])
            for m in bm.get("markets", [])
            if m.get("key") == market
        )
        available = len(bookmakers) > 0
        status = (
            f"{len(bookmakers)} bookmaker(s), {n_outcomes} outcomes "
            f"[{', '.join(bookmakers[:5])}]"
            if available
            else "no bookmakers quoted it at this snapshot"
        )
        print(f"      {market}: {'OK' if available else 'EMPTY'} -- {status}")
        results[market] = {
            "available": available,
            "n_bookmakers": len(bookmakers),
            "bookmakers": bookmakers,
            "n_outcomes": n_outcomes,
        }
    report["prop_markets"] = results
    return results


def backfill_cost_table(n_markets: int) -> list[str]:
    """Projected credit cost of a full 2025-26 backfill (one snapshot/event)."""
    lines = [
        f"Backfill cost, 2025-26 ({BACKFILL_EVENTS} events, {n_markets} markets, "
        "1 pre-kickoff snapshot per event):",
    ]
    for n_regions in (1, 2, 3):
        event_cost = BACKFILL_EVENTS * n_markets * n_regions * 10
        total = event_cost + BACKFILL_GWS  # + events-list snapshots (1 each)
        lines.append(
            f"  {n_regions} region(s): {event_cost:,} event-odds credits "
            f"+ {BACKFILL_GWS} events-list = {total:,} credits"
        )
    lines.append(
        "  (docs: soccer props are US-bookmakers-only, so 1 region [us] "
        "captures full coverage; fits in one 20K/month plan cycle)"
    )
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Probe The Odds API for EPL player-prop availability."
    )
    parser.add_argument(
        "--regions",
        type=str,
        default="us",
        help=(
            "Regions for the prop probe (default: us -- per the-odds-api.com "
            "docs, soccer player props are US bookmakers only; each extra "
            "region multiplies cost)"
        ),
    )
    parser.add_argument(
        "--skip-h2h",
        action="store_true",
        help="Skip the h2h historical-access probe (saves 10 credits)",
    )
    args = parser.parse_args()

    _load_dotenv()
    api_key = os.environ.get("ODDS_API_KEY", "")
    if not api_key:
        print("Error: ODDS_API_KEY not found in environment or .env file")
        return 1

    report: dict = {
        "probed_utc": datetime.now(timezone.utc).isoformat(),
        "regions": args.regions,
        "markets": PROP_MARKETS,
    }
    verdict = "NO-GO"
    reason = ""
    try:
        if not args.skip_h2h:
            probe_h2h_historical(api_key, report)
        events = probe_events(api_key, report)
        if not events:
            reason = "historical access works but no events at probe snapshot"
        else:
            results = probe_prop_markets(api_key, events[0], args.regions, report)
            available = [m for m, r in results.items() if r.get("available")]
            if available:
                verdict = "GO"
                reason = (
                    f"{len(available)}/{len(PROP_MARKETS)} prop markets have "
                    f"bookmaker quotes: {', '.join(available)}"
                )
            else:
                reason = (
                    "historical access works but no bookmaker quoted any "
                    "requested prop market for this event/snapshot"
                )
    except ProbeAuthError as exc:
        reason = (
            f"authentication failed (HTTP 401): {exc}\n"
            "  -> The key is invalid/expired, or the plan does not include\n"
            "     historical endpoints.  Historical odds (and therefore any\n"
            "     player-prop backfill) require a PAID plan on\n"
            "     https://the-odds-api.com -- free keys get 401 here.\n"
            "     Renew/upgrade the key in .env (ODDS_API_KEY) and re-run."
        )
        report["auth_error"] = str(exc)
    except requests.RequestException as exc:
        reason = f"network error: {exc}"
    except RuntimeError as exc:
        reason = f"API error: {exc}"

    report["verdict"] = verdict
    report["reason"] = reason

    print()
    print("=" * 72)
    print(f"VERDICT: {verdict}")
    print(f"  {reason}")
    if verdict == "GO":
        print()
        for line in backfill_cost_table(len(PROP_MARKETS)):
            print(f"  {line}")
        print(
            f"  Credits remaining on this key: "
            f"{report.get('credits_remaining', 'unknown')}"
        )
    print("=" * 72)

    runs_dir = Path(__file__).resolve().parent.parent / "runs"
    runs_dir.mkdir(exist_ok=True)
    out = runs_dir / (
        f"props_probe_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    )
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report saved: {out}")
    return 0 if verdict == "GO" else 1


if __name__ == "__main__":
    sys.exit(main())
