"""Download free Pinnacle closing odds from football-data.co.uk for a season.

Converts E0.csv into the same JSON layout as data/odds/{season}.json
(dict keyed by GW -> list of {event_id, commence_time, home/away_team,
home/draw/away odds}).  Uses Pinnacle closing odds (PSCH/PSCD/PSCA) with
opening odds (PSH/PSD/PSA) as fallback.  GW mapping comes from the season's
fixtures.csv kickoff dates.

Usage:
    python scripts/collect_football_data_odds.py --season 2025-26
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

REPO_ROOT = Path(__file__).resolve().parent.parent
URL_TEMPLATE = "https://www.football-data.co.uk/mmz4281/{yy}/E0.csv"


def season_to_code(season: str) -> str:
    """'2025-26' -> '2526'"""
    start, end = season.split("-")
    return start[2:] + end


def build_date_to_gw(fixtures_path: Path) -> dict[str, int]:
    """Map 'YYYY-MM-DD' -> most common GW among fixtures that day."""
    fx = pd.read_csv(fixtures_path)
    fx = fx.dropna(subset=["event", "kickoff_time"])
    date_events: dict[str, Counter] = {}
    for _, row in fx.iterrows():
        date = str(row["kickoff_time"])[:10]
        date_events.setdefault(date, Counter())[int(row["event"])] += 1
    return {d: c.most_common(1)[0][0] for d, c in date_events.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", required=True, help="e.g. 2025-26")
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    args = parser.parse_args()

    url = URL_TEMPLATE.format(yy=season_to_code(args.season))
    print(f"Downloading {url}")
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.content.decode("utf-8", errors="replace")))
    print(f"{len(df)} matches in E0.csv")

    fixtures_path = args.data_dir / "raw" / args.season / "fixtures.csv"
    if not fixtures_path.exists():
        raise SystemExit(f"fixtures.csv missing for {args.season} — collect it first")
    date_to_gw = build_date_to_gw(fixtures_path)

    by_gw: dict[str, list[dict]] = {}
    unmapped = 0
    for _, row in df.iterrows():
        # football-data dates are DD/MM/YYYY (or DD/MM/YY in old files)
        try:
            date = pd.to_datetime(row["Date"], dayfirst=True).strftime("%Y-%m-%d")
        except Exception:
            unmapped += 1
            continue
        gw = date_to_gw.get(date)
        if gw is None:
            # fall back to nearest known fixture date within 3 days
            candidates = [
                (abs((pd.Timestamp(date) - pd.Timestamp(d)).days), g)
                for d, g in date_to_gw.items()
                if abs((pd.Timestamp(date) - pd.Timestamp(d)).days) <= 3
            ]
            if candidates:
                gw = min(candidates)[1]
            else:
                unmapped += 1
                continue

        def _odds(*cols: str) -> float | None:
            # Pinnacle closing > Pinnacle opening > market-avg closing > avg
            for col in cols:
                v = row.get(col)
                if pd.notna(v):
                    return float(v)
            return None

        home = _odds("PSCH", "PSH", "AvgCH", "AvgH")
        draw = _odds("PSCD", "PSD", "AvgCD", "AvgD")
        away = _odds("PSCA", "PSA", "AvgCA", "AvgA")
        if home is None or draw is None or away is None:
            continue
        by_gw.setdefault(str(gw), []).append(
            {
                "event_id": f"fd_{args.season}_{row['HomeTeam']}_{row['AwayTeam']}",
                "commence_time": f"{date}T00:00:00Z",
                "home_team": str(row["HomeTeam"]),
                "away_team": str(row["AwayTeam"]),
                "home_odds": home,
                "draw_odds": draw,
                "away_odds": away,
                "last_update": "",
            }
        )

    out_path = args.data_dir / "odds" / f"{args.season}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(by_gw, indent=1), encoding="utf-8")
    n = sum(len(v) for v in by_gw.values())
    print(f"Saved {n} matches across {len(by_gw)} GWs to {out_path}"
          f" ({unmapped} unmapped)")


if __name__ == "__main__":
    main()
