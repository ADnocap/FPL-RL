"""Features derived from FPL expected points (xP).

The ``xP`` column in merged_gw.csv contains FPL's predicted points per
player per GW, published BEFORE the GW deadline — fully point-in-time
safe.  Available from 2020-21 onward.

Features (per element per GW)
-----------------------------
fpl_xp : FPL's own predicted points for this GW (2020-21+)
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "fpl_xp",
]


def compute_players_raw_features(
    data_dir: Path,
    season: str,
    merged_gw: pd.DataFrame,
) -> pd.DataFrame:
    """Extract FPL expected points from merged_gw.

    Uses the ``xP`` column directly from merged_gw.csv, which is
    available for seasons 2020-21 onward.  For older seasons the
    feature is NaN.

    Parameters
    ----------
    data_dir : Path
        Root data directory (unused, kept for API consistency).
    season : str
        Season string, e.g. ``"2023-24"``.
    merged_gw : pd.DataFrame
        Per-player-per-fixture data with columns ``element``, ``GW``,
        and optionally ``xP``.

    Returns
    -------
    pd.DataFrame
        One row per (element, GW) with columns: element, GW, fpl_xp.
    """
    empty = pd.DataFrame(columns=["element", "GW"] + FEATURE_COLS)
    if merged_gw.empty:
        return empty

    base = merged_gw[["element", "GW"]].copy()

    if "xP" in merged_gw.columns:
        base["fpl_xp"] = pd.to_numeric(merged_gw["xP"], errors="coerce")
    else:
        base["fpl_xp"] = float("nan")

    # Aggregate DGW rows (average xP across fixtures)
    result = base.groupby(["element", "GW"], as_index=False).agg(
        {"fpl_xp": "mean"}
    )

    n_xp = result["fpl_xp"].notna().sum()
    logger.info(
        "fpl_xp %s: %d/%d rows have xP (%.1f%%)",
        season, n_xp, len(result),
        100.0 * n_xp / len(result) if len(result) > 0 else 0,
    )

    return result[["element", "GW"] + FEATURE_COLS]
