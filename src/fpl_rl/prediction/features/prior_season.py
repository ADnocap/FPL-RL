"""Prior-season features from Understat league and FBref full-season aggregates.

These features are constant per player per season (no per-GW variation).
They use **full-season** aggregates from season S-1, which means they are
safe from lookahead bias when used as features for season S.

Produces 12 features per player:
  Understat (6): prev_xg_per90, prev_xa_per90, prev_npxg_per90,
                 prev_shots_per90, prev_key_passes_per90, prev_minutes
  FBref    (6): prev_sot_per90, prev_pass_cmp_pct, prev_prog_dist_per90,
                prev_tkl_int_per90, prev_blocks_per90, prev_gls_per90
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from fpl_rl.data.collectors.id_mapping import _normalize_name
from fpl_rl.prediction.id_resolver import IDResolver

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------
# Season mapping
# -------------------------------------------------------------------------

PREV_SEASON: dict[str, str] = {
    "2017-18": "2016-17",
    "2018-19": "2017-18",
    "2019-20": "2018-19",
    "2020-21": "2019-20",
    "2021-22": "2020-21",
    "2022-23": "2021-22",
    "2023-24": "2022-23",
    "2024-25": "2023-24",
}

# All 12 output feature column names
PRIOR_FEATURE_COLUMNS: list[str] = [
    "prev_xg_per90",
    "prev_xa_per90",
    "prev_npxg_per90",
    "prev_shots_per90",
    "prev_key_passes_per90",
    "prev_minutes",
    "prev_sot_per90",
    "prev_pass_cmp_pct",
    "prev_prog_dist_per90",
    "prev_tkl_int_per90",
    "prev_blocks_per90",
    "prev_gls_per90",
]

# Minimum minutes threshold for per-90 calculations
_MIN_MINUTES_THRESHOLD = 90


# -------------------------------------------------------------------------
# Column-finding helper for FBref parquets
# -------------------------------------------------------------------------


def _find_col(columns: list[str], *fragments: str) -> str | None:
    """Find first column whose name contains ALL given fragments (case-insensitive).

    Parameters
    ----------
    columns : list[str]
        Available column names.
    *fragments : str
        Substrings that must all appear in the column name.

    Returns
    -------
    str | None
        The matching column name, or None if not found.
    """
    for col in columns:
        col_lower = col.lower()
        if all(f.lower() in col_lower for f in fragments):
            return col
    return None


# -------------------------------------------------------------------------
# Understat league features
# -------------------------------------------------------------------------


def _load_understat_features(
    data_dir: Path,
    prev_season: str,
    id_resolver: IDResolver,
    codes: list[int],
) -> pd.DataFrame:
    """Load understat league JSON for *prev_season* and compute 6 features.

    Returns a DataFrame with columns (code, prev_xg_per90, prev_xa_per90,
    prev_npxg_per90, prev_shots_per90, prev_key_passes_per90, prev_minutes).
    """
    understat_cols = [
        "prev_xg_per90",
        "prev_xa_per90",
        "prev_npxg_per90",
        "prev_shots_per90",
        "prev_key_passes_per90",
        "prev_minutes",
    ]

    json_path = data_dir / "understat" / "league" / f"{prev_season}.json"

    if not json_path.exists():
        logger.warning("Understat league file not found: %s", json_path)
        return _empty_df(codes, understat_cols)

    try:
        with open(json_path, encoding="utf-8") as f:
            players = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to read understat league data: %s", exc)
        return _empty_df(codes, understat_cols)

    # Build understat_id -> player dict
    us_by_id: dict[int, dict] = {}
    for p in players:
        try:
            us_by_id[int(p["id"])] = p
        except (ValueError, KeyError):
            continue

    # Map each code to its understat data
    rows: list[dict] = []
    for code in codes:
        us_id = id_resolver.understat_id(code)
        if us_id is None or us_id not in us_by_id:
            rows.append({"code": code})
            continue

        p = us_by_id[us_id]
        try:
            minutes = float(p.get("time", 0))
            xg = float(p.get("xG", 0))
            xa = float(p.get("xA", 0))
            npxg = float(p.get("npxG", 0))
            shots = float(p.get("shots", 0))
            key_passes = float(p.get("key_passes", 0))
        except (ValueError, TypeError):
            rows.append({"code": code})
            continue

        nineties = minutes / 90.0

        row: dict = {"code": code, "prev_minutes": minutes}

        if minutes >= _MIN_MINUTES_THRESHOLD:
            row["prev_xg_per90"] = xg / nineties
            row["prev_xa_per90"] = xa / nineties
            row["prev_npxg_per90"] = npxg / nineties
            row["prev_shots_per90"] = shots / nineties
            row["prev_key_passes_per90"] = key_passes / nineties
        # else: per-90 values stay NaN (not set)

        rows.append(row)

    df = pd.DataFrame(rows)
    # Ensure all columns exist
    for col in understat_cols:
        if col not in df.columns:
            df[col] = np.nan
    return df[["code"] + understat_cols]


# -------------------------------------------------------------------------
# FBref features
# -------------------------------------------------------------------------


def _load_fbref_features(
    data_dir: Path,
    prev_season: str,
    id_resolver: IDResolver,
    codes: list[int],
) -> pd.DataFrame:
    """Load FBref parquets for *prev_season* and compute 6 features.

    Returns a DataFrame with columns (code, prev_sot_per90, prev_pass_cmp_pct,
    prev_prog_dist_per90, prev_tkl_int_per90, prev_blocks_per90, prev_gls_per90).
    """
    fbref_cols = [
        "prev_sot_per90",
        "prev_pass_cmp_pct",
        "prev_prog_dist_per90",
        "prev_tkl_int_per90",
        "prev_blocks_per90",
        "prev_gls_per90",
    ]

    fbref_dir = data_dir / "fbref"

    # Build reverse mappings: fbref_id -> code and normalized name -> code
    fbref_id_to_code: dict[str, int] = {}
    name_to_code: dict[str, int] = {}
    for code in codes:
        fb_id = id_resolver.fbref_id(code)
        if fb_id is not None:
            fbref_id_to_code[fb_id] = code

        pname = id_resolver.player_name(code)
        if pname and pname != "Unknown":
            name_to_code[_normalize_name(pname)] = code

    # Initialize result dict: code -> feature dict
    result: dict[int, dict] = {code: {"code": code} for code in codes}

    # --- Shooting: prev_sot_per90 ---
    _extract_shooting(fbref_dir, prev_season, fbref_id_to_code, name_to_code, result)

    # --- Passing: prev_pass_cmp_pct, prev_prog_dist_per90 ---
    _extract_passing(fbref_dir, prev_season, fbref_id_to_code, name_to_code, result)

    # --- Defense: prev_tkl_int_per90, prev_blocks_per90 ---
    _extract_defense(fbref_dir, prev_season, fbref_id_to_code, name_to_code, result)

    # --- Standard: prev_gls_per90 ---
    _extract_standard(fbref_dir, prev_season, fbref_id_to_code, name_to_code, result)

    df = pd.DataFrame(list(result.values()))
    for col in fbref_cols:
        if col not in df.columns:
            df[col] = np.nan
    return df[["code"] + fbref_cols]


def _match_fbref_row_to_code(
    row: pd.Series,
    player_col: str,
    fbref_id_to_code: dict[str, int],
    name_to_code: dict[str, int],
) -> int | None:
    """Try to match a FBref row to a stable code.

    Strategy: name matching via _normalize_name.
    """
    # Match by player name
    raw_name = row.get(player_col)
    if pd.notna(raw_name) and isinstance(raw_name, str):
        norm = _normalize_name(raw_name)
        code = name_to_code.get(norm)
        if code is not None:
            return code

    return None


def _read_parquet_safe(path: Path) -> pd.DataFrame | None:
    """Read a parquet file, returning None on any error."""
    if not path.exists():
        logger.warning("FBref parquet not found: %s", path)
        return None
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        logger.warning("Failed to read FBref parquet %s: %s", path, exc)
        return None


def _get_player_col(df: pd.DataFrame) -> str | None:
    """Find the player name column in a FBref parquet."""
    return _find_col(list(df.columns), "player")


def _get_90s_col(df: pd.DataFrame) -> str | None:
    """Find the 90s column in a FBref parquet."""
    return _find_col(list(df.columns), "90s")


def _extract_shooting(
    fbref_dir: Path,
    prev_season: str,
    fbref_id_to_code: dict[str, int],
    name_to_code: dict[str, int],
    result: dict[int, dict],
) -> None:
    """Extract prev_sot_per90 from shooting parquet."""
    df = _read_parquet_safe(fbref_dir / f"{prev_season}_shooting.parquet")
    if df is None:
        return

    cols = list(df.columns)
    player_col = _get_player_col(df)
    # SoT/90 is provided directly in the data
    sot_per90_col = _find_col(cols, "sot", "/90") or _find_col(cols, "sot", "90")
    nineties_col = _get_90s_col(df)
    sot_col = _find_col(cols, "sot") if sot_per90_col is None else None

    if player_col is None:
        return

    for _, row in df.iterrows():
        code = _match_fbref_row_to_code(row, player_col, fbref_id_to_code, name_to_code)
        if code is None or code not in result:
            continue

        if sot_per90_col is not None:
            val = pd.to_numeric(row.get(sot_per90_col), errors="coerce")
            if pd.notna(val):
                result[code]["prev_sot_per90"] = float(val)
        elif sot_col is not None and nineties_col is not None:
            sot_val = pd.to_numeric(row.get(sot_col), errors="coerce")
            nineties_val = pd.to_numeric(row.get(nineties_col), errors="coerce")
            if pd.notna(sot_val) and pd.notna(nineties_val) and nineties_val >= 1.0:
                result[code]["prev_sot_per90"] = float(sot_val / nineties_val)


def _extract_passing(
    fbref_dir: Path,
    prev_season: str,
    fbref_id_to_code: dict[str, int],
    name_to_code: dict[str, int],
    result: dict[int, dict],
) -> None:
    """Extract prev_pass_cmp_pct and prev_prog_dist_per90 from passing parquet."""
    df = _read_parquet_safe(fbref_dir / f"{prev_season}_passing.parquet")
    if df is None:
        return

    cols = list(df.columns)
    player_col = _get_player_col(df)
    nineties_col = _get_90s_col(df)
    cmp_pct_col = _find_col(cols, "total", "cmp%") or _find_col(cols, "cmp%")
    prog_dist_col = _find_col(cols, "prgdist") or _find_col(cols, "prog", "dist")

    if player_col is None:
        return

    for _, row in df.iterrows():
        code = _match_fbref_row_to_code(row, player_col, fbref_id_to_code, name_to_code)
        if code is None or code not in result:
            continue

        # Pass completion %
        if cmp_pct_col is not None:
            val = pd.to_numeric(row.get(cmp_pct_col), errors="coerce")
            if pd.notna(val):
                result[code]["prev_pass_cmp_pct"] = float(val)

        # Progressive passing distance per 90
        if prog_dist_col is not None and nineties_col is not None:
            dist_val = pd.to_numeric(row.get(prog_dist_col), errors="coerce")
            nineties_val = pd.to_numeric(row.get(nineties_col), errors="coerce")
            if pd.notna(dist_val) and pd.notna(nineties_val) and nineties_val >= 1.0:
                result[code]["prev_prog_dist_per90"] = float(dist_val / nineties_val)


def _extract_defense(
    fbref_dir: Path,
    prev_season: str,
    fbref_id_to_code: dict[str, int],
    name_to_code: dict[str, int],
    result: dict[int, dict],
) -> None:
    """Extract prev_tkl_int_per90 and prev_blocks_per90 from defense parquet."""
    df = _read_parquet_safe(fbref_dir / f"{prev_season}_defense.parquet")
    if df is None:
        return

    cols = list(df.columns)
    player_col = _get_player_col(df)
    nineties_col = _get_90s_col(df)

    # Tkl+Int is often a single column
    tkl_int_col = _find_col(cols, "tkl+int")
    # Fallback: separate Tackles_Tkl and Int columns
    tkl_col = _find_col(cols, "tackles", "tkl") if tkl_int_col is None else None
    int_col = _find_col(cols, "int") if tkl_int_col is None else None
    # Be careful: "int" might also match "Challenges_Tkl%" etc. — use the standalone Int col
    if int_col is not None and "challenge" in int_col.lower():
        int_col = None
    # Try to find the standalone Int column more precisely
    if tkl_int_col is None and int_col is None:
        for c in cols:
            c_lower = c.lower()
            if c_lower.endswith("_int") or c_lower.endswith("int"):
                if "tkl" not in c_lower and "challenge" not in c_lower:
                    int_col = c
                    break

    blocks_col = _find_col(cols, "blocks", "blocks") or _find_col(cols, "blocks")
    # Avoid the sub-columns Blocks_Sh, Blocks_Pass — prefer Blocks_Blocks
    if blocks_col is not None and ("_sh" in blocks_col.lower() or "_pass" in blocks_col.lower()):
        # Try to find the main blocks column
        for c in cols:
            if c.lower().endswith("blocks_blocks") or c.lower() == "blocks_blocks":
                blocks_col = c
                break

    if player_col is None:
        return

    for _, row in df.iterrows():
        code = _match_fbref_row_to_code(row, player_col, fbref_id_to_code, name_to_code)
        if code is None or code not in result:
            continue

        nineties_val = pd.to_numeric(row.get(nineties_col), errors="coerce") if nineties_col else np.nan

        # Tackles + Interceptions per 90
        if tkl_int_col is not None and nineties_col is not None:
            val = pd.to_numeric(row.get(tkl_int_col), errors="coerce")
            if pd.notna(val) and pd.notna(nineties_val) and nineties_val >= 1.0:
                result[code]["prev_tkl_int_per90"] = float(val / nineties_val)
        elif tkl_col is not None and int_col is not None and nineties_col is not None:
            tkl_val = pd.to_numeric(row.get(tkl_col), errors="coerce")
            int_val = pd.to_numeric(row.get(int_col), errors="coerce")
            if (
                pd.notna(tkl_val)
                and pd.notna(int_val)
                and pd.notna(nineties_val)
                and nineties_val >= 1.0
            ):
                result[code]["prev_tkl_int_per90"] = float(
                    (tkl_val + int_val) / nineties_val
                )

        # Blocks per 90
        if blocks_col is not None and nineties_col is not None:
            val = pd.to_numeric(row.get(blocks_col), errors="coerce")
            if pd.notna(val) and pd.notna(nineties_val) and nineties_val >= 1.0:
                result[code]["prev_blocks_per90"] = float(val / nineties_val)


def _extract_standard(
    fbref_dir: Path,
    prev_season: str,
    fbref_id_to_code: dict[str, int],
    name_to_code: dict[str, int],
    result: dict[int, dict],
) -> None:
    """Extract prev_gls_per90 from standard parquet."""
    df = _read_parquet_safe(fbref_dir / f"{prev_season}_standard.parquet")
    if df is None:
        return

    cols = list(df.columns)
    player_col = _get_player_col(df)
    # Per 90 Minutes_Gls is the goals per 90 directly
    gls_per90_col = _find_col(cols, "per 90", "gls") or _find_col(cols, "90", "gls")
    nineties_col = _get_90s_col(df)
    # Fallback: Performance_Gls / 90s
    gls_col = _find_col(cols, "performance", "gls") if gls_per90_col is None else None

    if player_col is None:
        return

    for _, row in df.iterrows():
        code = _match_fbref_row_to_code(row, player_col, fbref_id_to_code, name_to_code)
        if code is None or code not in result:
            continue

        if gls_per90_col is not None:
            val = pd.to_numeric(row.get(gls_per90_col), errors="coerce")
            if pd.notna(val):
                result[code]["prev_gls_per90"] = float(val)
        elif gls_col is not None and nineties_col is not None:
            gls_val = pd.to_numeric(row.get(gls_col), errors="coerce")
            nineties_val = pd.to_numeric(row.get(nineties_col), errors="coerce")
            if pd.notna(gls_val) and pd.notna(nineties_val) and nineties_val >= 1.0:
                result[code]["prev_gls_per90"] = float(gls_val / nineties_val)


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------


def _empty_df(codes: list[int], feature_cols: list[str]) -> pd.DataFrame:
    """Return a DataFrame of NaNs for *codes* with the given feature columns."""
    df = pd.DataFrame({"code": codes})
    for col in feature_cols:
        df[col] = np.nan
    return df


# -------------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------------


def compute_prior_season_features(
    data_dir: Path,
    season: str,
    id_resolver: IDResolver,
) -> pd.DataFrame:
    """Compute 12 prior-season features for all players in *season*.

    Parameters
    ----------
    data_dir : Path
        Root data directory containing ``understat/league/`` and ``fbref/``
        sub-directories.
    season : str
        The target season (e.g. ``"2023-24"``). Features are drawn from
        the previous season ``S-1``.
    id_resolver : IDResolver
        Resolver providing code <-> understat/fbref ID mappings.

    Returns
    -------
    pd.DataFrame
        One row per player code, with columns ``code`` and 12 feature
        columns. Players without prior-season data get NaN.
    """
    codes = id_resolver.all_codes_for_season(season)

    prev = PREV_SEASON.get(season)
    if prev is None:
        logger.info("No prior season for %s — returning all NaN.", season)
        return _empty_df(codes, PRIOR_FEATURE_COLUMNS)

    # Load features from both sources
    us_df = _load_understat_features(data_dir, prev, id_resolver, codes)
    fb_df = _load_fbref_features(data_dir, prev, id_resolver, codes)

    # Merge on code
    merged = us_df.merge(fb_df, on="code", how="outer")

    # Ensure all expected columns exist
    for col in PRIOR_FEATURE_COLUMNS:
        if col not in merged.columns:
            merged[col] = np.nan

    return merged[["code"] + PRIOR_FEATURE_COLUMNS].reset_index(drop=True)
