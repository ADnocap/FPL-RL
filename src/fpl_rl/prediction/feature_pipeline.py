"""Orchestrates all feature modules into a single training-ready DataFrame.

Builds one row per (code, season, gw) with all features + position + target.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from fpl_rl.prediction.id_resolver import IDResolver, SEASON_TO_COL
from fpl_rl.prediction.features.vaastav import compute_vaastav_features
from fpl_rl.prediction.features.understat import compute_understat_features
from fpl_rl.prediction.features.prior_season import compute_prior_season_features
from fpl_rl.prediction.features.opponent import compute_opponent_features

logger = logging.getLogger(__name__)

# Position mapping (element_type int -> string label)
_ELEMENT_TYPE_TO_POS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


class FeaturePipeline:
    """Orchestrates feature engineering across all seasons.

    Parameters
    ----------
    data_dir : Path
        Root data directory containing ``raw/``, ``understat/``, ``fbref/``,
        and ``id_maps/`` sub-directories.
    id_resolver : IDResolver
        Cross-source ID resolver.
    seasons : list[str]
        Seasons to process, e.g. ``["2016-17", "2017-18", ...]``.
    """

    def __init__(
        self,
        data_dir: Path,
        id_resolver: IDResolver,
        seasons: list[str],
    ) -> None:
        self.data_dir = data_dir
        self.id_resolver = id_resolver
        self.seasons = seasons

    def build(self) -> pd.DataFrame:
        """Build the full feature DataFrame across all seasons.

        Returns
        -------
        pd.DataFrame
            One row per (code, season, gw) with columns:
            - ``code``, ``season``, ``GW``
            - All feature columns from vaastav, understat, prior_season, opponent
            - ``position`` (str: GK/DEF/MID/FWD)
            - ``total_points`` (target variable)
        """
        all_season_dfs: list[pd.DataFrame] = []

        try:
            from tqdm.auto import tqdm
            season_iter = tqdm(self.seasons, desc="Seasons", unit="season")
        except ImportError:
            season_iter = self.seasons

        for season in season_iter:
            logger.info("Processing season %s...", season)
            try:
                season_df = self._build_season(season)
                if not season_df.empty:
                    all_season_dfs.append(season_df)
                    logger.info(
                        "  %s: %d rows, %d columns",
                        season, len(season_df), len(season_df.columns),
                    )
            except Exception as exc:
                logger.error("Failed to process season %s: %s", season, exc)
                continue

        if not all_season_dfs:
            logger.warning("No data produced for any season")
            return pd.DataFrame()

        result = pd.concat(all_season_dfs, ignore_index=True)
        logger.info(
            "Feature pipeline complete: %d rows x %d columns",
            len(result), len(result.columns),
        )
        return result

    def _build_season(self, season: str) -> pd.DataFrame:
        """Build features for a single season."""
        raw_dir = self.data_dir / "raw" / season

        # 1. Load merged_gw
        merged_path = raw_dir / "gws" / "merged_gw.csv"
        if not merged_path.exists():
            logger.warning("No merged_gw.csv for %s", season)
            return pd.DataFrame()

        try:
            merged_gw = pd.read_csv(merged_path, encoding="utf-8", on_bad_lines="skip")
        except UnicodeDecodeError:
            merged_gw = pd.read_csv(merged_path, encoding="latin-1", on_bad_lines="skip")

        # Ensure numeric types
        for col in ["element", "GW", "total_points", "minutes", "value",
                     "goals_scored", "assists", "clean_sheets", "bonus", "bps",
                     "goals_conceded", "selected"]:
            if col in merged_gw.columns:
                merged_gw[col] = pd.to_numeric(merged_gw[col], errors="coerce")

        for col in ["influence", "creativity", "threat", "ict_index"]:
            if col in merged_gw.columns:
                merged_gw[col] = pd.to_numeric(merged_gw[col], errors="coerce").fillna(0.0)

        # Ensure was_home is boolean
        if "was_home" in merged_gw.columns:
            merged_gw["was_home"] = merged_gw["was_home"].map(
                {True: True, False: False, "True": True, "False": False, 1: True, 0: False}
            ).fillna(False)

        # Ensure 'team' is a numeric column (some seasons have team names, some lack it)
        merged_gw = self._ensure_numeric_team(merged_gw, raw_dir)

        # Ensure opponent_team is numeric
        if "opponent_team" in merged_gw.columns:
            merged_gw["opponent_team"] = pd.to_numeric(
                merged_gw["opponent_team"], errors="coerce"
            )

        # 2. Map element_id -> code
        merged_gw["code"] = merged_gw["element"].apply(
            lambda eid: self.id_resolver.code_from_element_id(season, int(eid))
            if pd.notna(eid) else None
        )
        # Drop rows without a code mapping
        n_before = len(merged_gw)
        merged_gw = merged_gw.dropna(subset=["code"])
        merged_gw["code"] = merged_gw["code"].astype(int)
        n_dropped = n_before - len(merged_gw)
        if n_dropped > 0:
            logger.debug("  %s: dropped %d rows without code mapping", season, n_dropped)

        if merged_gw.empty:
            return pd.DataFrame()

        # 3. Compute vaastav rolling features (uses element column)
        vaastav_df = compute_vaastav_features(merged_gw)

        # 4. Compute understat features (needs GW dates)
        gw_dates = self._extract_gw_dates(merged_gw)
        understat_df = compute_understat_features(
            self.data_dir, season, self.id_resolver, gw_dates,
        )

        # 5. Compute prior-season features
        prior_df = compute_prior_season_features(
            self.data_dir, season, self.id_resolver,
        )

        # 6. Compute opponent features
        fixtures_df = self._load_csv_safe(raw_dir / "fixtures.csv")
        teams_df = self._load_csv_safe(raw_dir / "teams.csv")
        opponent_df = compute_opponent_features(merged_gw, fixtures_df, teams_df)

        # 7. Get position mapping from cleaned_players.csv
        position_df = self._load_positions(raw_dir, merged_gw)

        # 8. Get target (total_points) — aggregate DGW rows
        target_df = (
            merged_gw.groupby(["element", "GW"], as_index=False)
            .agg({"total_points": "sum", "code": "first"})
        )

        # 9. Merge everything together
        # Start with vaastav (has element, GW)
        result = vaastav_df.copy()

        # Add code from merged_gw mapping (element -> code)
        eid_to_code = merged_gw[["element", "code"]].drop_duplicates("element")
        result = result.merge(eid_to_code, on="element", how="left")

        # Add understat features (on code, GW)
        if not understat_df.empty:
            result = result.merge(understat_df, on=["code", "GW"], how="left")

        # Add prior-season features (on code, broadcast to all GWs)
        if not prior_df.empty:
            result = result.merge(prior_df, on="code", how="left")

        # Add opponent features (on element, GW)
        if not opponent_df.empty:
            result = result.merge(opponent_df, on=["element", "GW"], how="left")

        # Add position
        if not position_df.empty:
            result = result.merge(position_df, on="element", how="left")
        else:
            result["position"] = None

        # Add target
        result = result.merge(
            target_df[["element", "GW", "total_points"]].rename(
                columns={"total_points": "target"}
            ),
            on=["element", "GW"],
            how="left",
        )

        # Add season column
        result["season"] = season

        return result

    def _extract_gw_dates(self, merged_gw: pd.DataFrame) -> pd.Series:
        """Extract earliest kickoff_time per GW as a Series indexed by GW."""
        if "kickoff_time" not in merged_gw.columns:
            # Fallback: create synthetic dates (1 week apart)
            gws = sorted(merged_gw["GW"].unique())
            base = pd.Timestamp("2020-01-01")
            return pd.Series(
                {gw: base + pd.Timedelta(weeks=int(gw) - 1) for gw in gws}
            )

        merged_gw = merged_gw.copy()
        merged_gw["kickoff_dt"] = pd.to_datetime(
            merged_gw["kickoff_time"], format="mixed", errors="coerce", utc=True,
        )
        gw_dates = merged_gw.groupby("GW")["kickoff_dt"].min()
        # Strip timezone to avoid comparison issues with naive timestamps
        gw_dates = gw_dates.dt.tz_localize(None)
        return gw_dates

    def _load_positions(
        self, raw_dir: Path, merged_gw: pd.DataFrame
    ) -> pd.DataFrame:
        """Load position mapping from cleaned_players.csv or merged_gw."""
        # Try cleaned_players.csv first
        cp_path = raw_dir / "cleaned_players.csv"
        if cp_path.exists():
            try:
                cp = pd.read_csv(cp_path, encoding="utf-8", on_bad_lines="skip")
            except UnicodeDecodeError:
                cp = pd.read_csv(cp_path, encoding="latin-1", on_bad_lines="skip")
            if "element_type" in cp.columns and "id" in cp.columns:
                cp["position"] = cp["element_type"].map(_ELEMENT_TYPE_TO_POS)
                return cp[["id", "position"]].rename(columns={"id": "element"})

        # Fallback: position column in merged_gw
        if "position" in merged_gw.columns:
            pos_df = merged_gw[["element", "position"]].drop_duplicates("element")
            return pos_df

        return pd.DataFrame(columns=["element", "position"])

    @staticmethod
    def _ensure_numeric_team(merged_gw: pd.DataFrame, raw_dir: Path) -> pd.DataFrame:
        """Ensure 'team' column is numeric (int team IDs).

        Some seasons have string team names or lack the column entirely.
        Falls back to cleaned_players.csv for the element -> team mapping.
        """
        has_team = "team" in merged_gw.columns
        if has_team:
            # Check if already numeric
            numeric_team = pd.to_numeric(merged_gw["team"], errors="coerce")
            if numeric_team.notna().mean() > 0.5:
                merged_gw["team"] = numeric_team
                return merged_gw
            # Non-numeric (string names) — drop and rebuild from cleaned_players
            merged_gw = merged_gw.drop(columns=["team"])

        # Look up team from cleaned_players.csv
        cp_path = raw_dir / "cleaned_players.csv"
        if cp_path.exists():
            try:
                cp = pd.read_csv(cp_path, encoding="utf-8", on_bad_lines="skip")
            except UnicodeDecodeError:
                cp = pd.read_csv(cp_path, encoding="latin-1", on_bad_lines="skip")
            if "id" in cp.columns and "team" in cp.columns:
                team_map = cp[["id", "team"]].drop_duplicates("id")
                team_map["team"] = pd.to_numeric(team_map["team"], errors="coerce")
                merged_gw = merged_gw.merge(
                    team_map.rename(columns={"id": "element"}),
                    on="element", how="left",
                )
                return merged_gw

        # Last resort: set team to NaN
        merged_gw["team"] = pd.NA
        return merged_gw

    @staticmethod
    def _load_csv_safe(path: Path) -> pd.DataFrame:
        """Load a CSV, returning empty DataFrame if missing."""
        if not path.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
        except UnicodeDecodeError:
            try:
                return pd.read_csv(path, encoding="latin-1", on_bad_lines="skip")
            except Exception:
                return pd.DataFrame()
        except Exception:
            return pd.DataFrame()
