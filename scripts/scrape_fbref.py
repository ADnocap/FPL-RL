#!/usr/bin/env python3
"""Download FBref player season stats using undetected-chromedriver.

Bypasses Cloudflare automatically — no manual intervention needed.

Usage:
    python scripts/scrape_fbref.py
    python scripts/scrape_fbref.py --seasons 2023-24,2024-25
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fpl_rl.utils.constants import AVAILABLE_SEASONS

logger = logging.getLogger(__name__)

STAT_TYPES = ["stats", "passing", "defense", "shooting"]
STAT_FILE_NAMES = {
    "stats": "standard",
    "passing": "passing",
    "defense": "defense",
    "shooting": "shooting",
}

FBREF_COMP_ID = 9

SEASON_TO_FBREF = {
    s: f"20{s.split('-')[0][2:]}-20{s.split('-')[1]}" for s in AVAILABLE_SEASONS
}

MIN_DELAY = 8


def build_url(fbref_season: str, stat_type: str) -> str:
    return (
        f"https://fbref.com/en/comps/{FBREF_COMP_ID}"
        f"/{fbref_season}/{stat_type}/{fbref_season}-Premier-League-Stats"
    )


def wait_for_page(driver, timeout: int = 60) -> bool:
    """Wait for Cloudflare to auto-solve and page to load."""
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        title = driver.title
        if "Just a moment" not in title:
            return True
        time.sleep(2)
    return False


def scrape_table(driver, url: str) -> pd.DataFrame | None:
    """Navigate to a FBref page and extract the player stats table."""
    driver.get(url)

    if not wait_for_page(driver):
        logger.warning("Cloudflare did not resolve for %s", url)
        return None

    # Extra wait for table rendering
    time.sleep(3)

    # Find the player stats table (has a 'Player' header)
    try:
        from selenium.webdriver.common.by import By

        tables = driver.find_elements(By.CSS_SELECTOR, "table.stats_table")
        target_table = None
        for table in tables:
            headers = table.find_elements(By.TAG_NAME, "th")
            for h in headers:
                if h.text.strip() == "Player":
                    target_table = table
                    break
            if target_table:
                break

        if not target_table:
            if tables:
                target_table = tables[-1]
            else:
                logger.warning("No stats table found at %s", url)
                return None

        table_html = target_table.get_attribute("outerHTML")
    except Exception as exc:
        logger.warning("Failed to extract table from %s: %s", url, exc)
        return None

    try:
        dfs = pd.read_html(io.StringIO(table_html))
        if not dfs:
            return None
        df = dfs[0]
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ["_".join(str(c) for c in col).strip("_") for col in df.columns]
        rk_col = [c for c in df.columns if "Rk" in str(c)]
        if rk_col:
            df = df[df[rk_col[0]] != "Rk"]
            df = df.dropna(subset=[rk_col[0]])
        return df
    except Exception as exc:
        logger.error("Failed to parse table: %s", exc)
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scrape FBref stats")
    parser.add_argument(
        "--seasons", type=str, default=None,
        help="Comma-separated seasons. Default: all",
    )
    parser.add_argument(
        "--data-dir", type=str, default="data",
        help="Base data directory (default: data/)",
    )
    parser.add_argument(
        "--delay", type=int, default=MIN_DELAY,
        help=f"Seconds between requests (default: {MIN_DELAY})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    seasons = args.seasons.split(",") if args.seasons else list(AVAILABLE_SEASONS)
    fbref_dir = Path(args.data_dir) / "fbref"
    fbref_dir.mkdir(parents=True, exist_ok=True)

    tasks: list[tuple[str, str, str, Path]] = []
    for season in seasons:
        fb_season = SEASON_TO_FBREF.get(season)
        if not fb_season:
            logger.error("Unknown season: %s", season)
            continue
        for stat_url_type, stat_file_name in STAT_FILE_NAMES.items():
            dest = fbref_dir / f"{season}_{stat_file_name}.parquet"
            if dest.exists() and dest.stat().st_size > 0:
                logger.info("Cached: %s", dest.name)
                continue
            tasks.append((season, fb_season, stat_url_type, dest))

    if not tasks:
        print("All FBref data already cached!")
        return

    print(f"{len(tasks)} tables to download")
    print("Launching browser...\n")

    import undetected_chromedriver as uc

    options = uc.ChromeOptions()
    driver = uc.Chrome(options=options, version_main=145)

    try:
        ok = 0
        failed = 0
        for season, fb_season, stat_type, dest in tqdm(
            tasks, desc="FBref tables", unit="table"
        ):
            url = build_url(fb_season, stat_type)
            stat_name = STAT_FILE_NAMES[stat_type]
            logger.info("Fetching %s %s ...", season, stat_name)

            try:
                df = scrape_table(driver, url)
                if df is not None and not df.empty:
                    df.to_parquet(dest)
                    logger.info("Saved %s (%d rows)", dest.name, len(df))
                    ok += 1
                else:
                    logger.warning("Empty table for %s %s", season, stat_name)
                    failed += 1
            except Exception as exc:
                logger.error("Failed %s %s: %s", season, stat_name, exc)
                failed += 1

            time.sleep(args.delay)

        print(f"\nDone: {ok} saved, {failed} failed out of {len(tasks)} total")

    finally:
        try:
            driver.quit()
        except Exception:
            pass


if __name__ == "__main__":
    main()
