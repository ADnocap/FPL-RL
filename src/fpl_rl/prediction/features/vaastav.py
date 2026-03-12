"""Rolling features computed from vaastav merged_gw.csv data.

All rolling windows use `.shift(1)` before `.rolling()` to prevent lookahead:
the feature for GW=k uses only data from GW < k.

DGW rows (multiple fixtures in one gameweek) are pre-aggregated by summing
numeric columns per (element, GW) before any rolling computation.
"""

from __future__ import annotations

import pandas as pd

# ---------------------------------------------------------------------------
# Rolling feature specifications
# ---------------------------------------------------------------------------
# Each entry: (output_col, source_col, window, agg)
# agg is one of: "mean", "sum", "std"

_ROLLING_SPECS: list[tuple[str, str, int, str]] = [
    # Points
    ("pts_rolling_3", "total_points", 3, "mean"),
    ("pts_rolling_5", "total_points", 5, "mean"),
    ("pts_rolling_10", "total_points", 10, "mean"),
    # Minutes
    ("mins_rolling_3", "minutes", 3, "mean"),
    ("mins_rolling_5", "minutes", 5, "mean"),
    ("mins_std_5", "minutes", 5, "std"),
    # Goals
    ("goals_rolling_3", "goals_scored", 3, "sum"),
    ("goals_rolling_5", "goals_scored", 5, "sum"),
    # Assists
    ("assists_rolling_3", "assists", 3, "sum"),
    ("assists_rolling_5", "assists", 5, "sum"),
    # Clean sheets
    ("cs_rolling_5", "clean_sheets", 5, "mean"),
    ("cs_rolling_10", "clean_sheets", 10, "mean"),
    # Bonus
    ("bonus_rolling_5", "bonus", 5, "mean"),
    ("bonus_rolling_10", "bonus", 10, "mean"),
    # BPS
    ("bps_rolling_5", "bps", 5, "mean"),
    ("bps_rolling_10", "bps", 10, "mean"),
    # ICT
    ("ict_rolling_3", "ict_index", 3, "mean"),
    ("ict_rolling_5", "ict_index", 5, "mean"),
    ("ict_rolling_10", "ict_index", 10, "mean"),
    # ICT sub-components
    ("influence_rolling_5", "influence", 5, "mean"),
    ("creativity_rolling_5", "creativity", 5, "mean"),
    ("threat_rolling_5", "threat", 5, "mean"),
    # Saves (GK points source)
    ("saves_rolling_5", "saves", 5, "mean"),
    # Goals conceded (DEF/GK clean sheet proxy)
    ("goals_conceded_rolling_5", "goals_conceded", 5, "mean"),
    # Transfers balance (crowd wisdom)
    ("transfers_balance_rolling_3", "transfers_balance", 3, "mean"),
    # Discipline
    ("yellows_rolling_5", "yellow_cards", 5, "sum"),
    ("reds_rolling_10", "red_cards", 10, "sum"),
    # Starts (rotation signal, 2022-23+ only)
    ("starts_rolling_5", "starts", 5, "mean"),
    # FPL expected stats (2022-23+ only)
    ("fpl_xg_rolling_5", "expected_goals", 5, "mean"),
    ("fpl_xa_rolling_5", "expected_assists", 5, "mean"),
    ("fpl_xgi_rolling_5", "expected_goal_involvements", 5, "mean"),
    ("fpl_xgc_rolling_5", "expected_goals_conceded", 5, "mean"),
]

# Columns that get summed during DGW aggregation
_NUMERIC_SUM_COLS = [
    "total_points", "minutes", "goals_scored", "assists",
    "clean_sheets", "bonus", "bps",
    "saves", "goals_conceded", "yellow_cards", "red_cards",
    "starts", "expected_goals", "expected_assists",
    "expected_goal_involvements", "expected_goals_conceded",
    "transfers_balance",
]

# Columns that are floats in the raw data and get summed during DGW agg
_FLOAT_SUM_COLS = ["influence", "creativity", "threat", "ict_index"]

# Columns taken from the last row in a DGW group (known before deadline)
_LAST_COLS = ["value", "selected"]

# All output feature columns (rolling specs + expanding + non-rolling)
FEATURE_COLUMNS: list[str] = (
    [spec[0] for spec in _ROLLING_SPECS]
    + ["season_avg_pts", "season_total_mins", "games_played",
       "value", "selected_norm"]
)


def compute_vaastav_features(merged_gw: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling features from merged_gw data.

    Parameters
    ----------
    merged_gw : pd.DataFrame
        Raw merged_gw data. Must contain at minimum the columns:
        element, GW, total_points, minutes, goals_scored, assists,
        clean_sheets, bonus, bps, influence, creativity, threat,
        ict_index, value, selected.
        May contain multiple rows per (element, GW) for double gameweeks.

    Returns
    -------
    pd.DataFrame
        One row per (element, GW) with columns ``element``, ``GW``,
        and the feature columns listed in :data:`FEATURE_COLUMNS`.
    """
    df = merged_gw.copy()

    # Ensure float types for ICT columns (they come as strings in some seasons)
    for col in _FLOAT_SUM_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # ------------------------------------------------------------------
    # 1. Aggregate DGW rows: sum numeric columns, take last for non-summed
    # ------------------------------------------------------------------
    sum_cols = [c for c in _NUMERIC_SUM_COLS + _FLOAT_SUM_COLS if c in df.columns]
    last_cols = [c for c in _LAST_COLS if c in df.columns]

    agg_dict: dict[str, str] = {}
    for c in sum_cols:
        agg_dict[c] = "sum"
    for c in last_cols:
        agg_dict[c] = "last"

    df = (
        df.sort_values(["element", "GW"])
        .groupby(["element", "GW"], as_index=False)
        .agg(agg_dict)
    )

    # ------------------------------------------------------------------
    # 2. Sort and compute rolling features per player
    # ------------------------------------------------------------------
    df = df.sort_values(["element", "GW"]).reset_index(drop=True)

    grouped = df.groupby("element")

    # Shifted series cache: {source_col: shifted Series}
    _shifted: dict[str, pd.Series] = {}

    def _get_shifted(col: str) -> pd.Series:
        if col not in _shifted:
            _shifted[col] = grouped[col].shift(1)
        return _shifted[col]

    # Compute each rolling feature (per-group to avoid cross-player contamination)
    for out_col, src_col, window, agg in _ROLLING_SPECS:
        # Guard: if source column doesn't exist (e.g. newer-season-only columns),
        # set output to NaN and skip
        if src_col not in df.columns:
            df[out_col] = float("nan")
            continue

        shifted = _get_shifted(src_col)
        if agg == "mean":
            df[out_col] = shifted.groupby(df["element"]).transform(
                lambda s: s.rolling(window=window, min_periods=1).mean()
            )
        elif agg == "sum":
            df[out_col] = shifted.groupby(df["element"]).transform(
                lambda s: s.rolling(window=window, min_periods=1).sum()
            )
        elif agg == "std":
            df[out_col] = shifted.groupby(df["element"]).transform(
                lambda s: s.rolling(window=window, min_periods=1).std()
            )
        else:
            raise ValueError(f"Unknown aggregation: {agg}")

    # ------------------------------------------------------------------
    # 3. Expanding (season-level) features — also shifted to avoid lookahead
    # ------------------------------------------------------------------
    shifted_pts = _get_shifted("total_points")
    shifted_mins = _get_shifted("minutes")

    df["season_avg_pts"] = shifted_pts.groupby(df["element"]).transform(
        lambda s: s.expanding(min_periods=1).mean()
    )
    df["season_total_mins"] = shifted_mins.groupby(df["element"]).transform(
        lambda s: s.expanding(min_periods=1).sum()
    )

    # games_played: count of prior GWs where the player had minutes > 0
    # shift(1) already done, so we count non-NaN values where minutes > 0
    played_flag = (shifted_mins > 0).astype(float)
    # Replace NaN (from the shift) with 0 for the flag
    played_flag = played_flag.fillna(0.0)
    df["games_played"] = played_flag.groupby(df["element"]).transform(
        lambda s: s.expanding(min_periods=1).sum()
    )

    # ------------------------------------------------------------------
    # 4. Non-rolling features (known at prediction time, from current row)
    # ------------------------------------------------------------------
    # value is already in the df from the aggregation step
    df["selected_norm"] = df["selected"] / 1e7

    # ------------------------------------------------------------------
    # 5. Select output columns
    # ------------------------------------------------------------------
    output_cols = ["element", "GW"] + FEATURE_COLUMNS
    # Only keep columns that exist (value is guaranteed)
    output_cols = [c for c in output_cols if c in df.columns]

    return df[output_cols].copy()
