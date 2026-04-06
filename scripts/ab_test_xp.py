#!/usr/bin/env python3
"""A/B test: shifted vs unshifted xP. Same data, same hyperparams, only xP differs."""

from __future__ import annotations
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

HOLDOUT = "2024-25"
TRAIN_SEASONS = [
    "2016-17", "2017-18", "2018-19", "2019-20",
    "2020-21", "2021-22", "2022-23", "2023-24",
]

PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "num_leaves": 127,
    "learning_rate": 0.01,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 10,
    "n_estimators": 2000,
    "verbose": -1,
    "lambda_l1": 1.0,
    "lambda_l2": 1.0,
}


def main():
    from fpl_rl.data.downloader import DEFAULT_DATA_DIR
    from fpl_rl.prediction.id_resolver import IDResolver
    from fpl_rl.prediction.feature_pipeline import FeaturePipeline
    from fpl_rl.prediction.model import PointPredictor
    from fpl_rl.data.loader import SeasonDataLoader
    from fpl_rl.engine.engine import FPLGameEngine
    from fpl_rl.engine.state import ChipState, EngineAction, GameState, PlayerSlot, Squad
    from fpl_rl.optimizer.squad_selection import select_squad
    from fpl_rl.optimizer.transfer_optimizer import optimize_transfers
    from fpl_rl.optimizer.types import build_candidate_pool, to_engine_action
    from fpl_rl.prediction.integration import PredictionIntegrator
    from fpl_rl.utils.constants import INITIAL_FREE_TRANSFERS, STARTING_BUDGET

    data_dir = DEFAULT_DATA_DIR.parent if DEFAULT_DATA_DIR.name == "raw" else DEFAULT_DATA_DIR
    all_seasons = TRAIN_SEASONS + [HOLDOUT]

    # ================================================================
    # STEP 1: Build features ONCE (both variants share the same base)
    # ================================================================
    print("=" * 70)
    print("A/B TEST: UNSHIFTED xP vs SHIFTED xP")
    print("=" * 70)
    print(f"Train: {TRAIN_SEASONS}")
    print(f"Holdout: {HOLDOUT}")
    print()

    print("Building feature pipeline...")
    t0 = time.time()
    resolver = IDResolver(data_dir)
    pipeline = FeaturePipeline(data_dir, resolver, all_seasons)
    df = pipeline.build()
    print(f"Built {len(df)} rows x {len(df.columns)} cols in {time.time()-t0:.0f}s")

    # Check if fpl_xp exists
    has_xp = "fpl_xp" in df.columns
    print(f"fpl_xp column present: {has_xp}")
    if not has_xp:
        print("ERROR: fpl_xp not in features. Cannot compare.")
        return

    # ================================================================
    # STEP 2: Create shifted variant
    # ================================================================
    df_unshifted = df.copy()

    df_shifted = df.copy()
    df_shifted = df_shifted.sort_values(["season", "element", "GW"])
    df_shifted["fpl_xp"] = df_shifted.groupby(["season", "element"])["fpl_xp"].shift(1)
    print(f"Unshifted fpl_xp non-null: {df_unshifted['fpl_xp'].notna().sum()}")
    print(f"Shifted fpl_xp non-null:   {df_shifted['fpl_xp'].notna().sum()}")

    # ================================================================
    # STEP 3: Train both models with SAME splits
    # ================================================================
    for label, data in [("UNSHIFTED", df_unshifted), ("SHIFTED", df_shifted)]:
        print(f"\n{'='*70}")
        print(f"  MODEL: {label} xP")
        print(f"{'='*70}")

        train_full = data[data["season"].isin(TRAIN_SEASONS)].copy()
        holdout_df = data[data["season"] == HOLDOUT].copy()

        # Val: last 8 GWs of last training season
        last_season = TRAIN_SEASONS[-1]
        last_data = train_full[train_full["season"] == last_season]
        max_gw = int(last_data["GW"].max())
        val_mask = (train_full["season"] == last_season) & (train_full["GW"] > max_gw - 8)
        val_df = train_full[val_mask].copy()
        train_df = train_full[~val_mask].copy()

        print(f"  Train: {len(train_df)}, Val: {len(val_df)}, Holdout: {len(holdout_df)}")

        # Train
        predictor = PointPredictor(params=PARAMS, early_stopping_rounds=50)
        metrics = predictor.train(train_df, val_df)
        print(f"  Train MAE: {metrics}")

        # Evaluate on holdout
        preds = predictor.predict(holdout_df)
        actual = holdout_df["target"].values
        valid = ~np.isnan(actual)

        mae = np.mean(np.abs(preds[valid] - actual[valid]))
        rmse = np.sqrt(np.mean((preds[valid] - actual[valid]) ** 2))
        corr = np.corrcoef(preds[valid], actual[valid])[0, 1]

        print(f"  Holdout MAE:  {mae:.4f}")
        print(f"  Holdout RMSE: {rmse:.4f}")
        print(f"  Holdout Corr: {corr:.4f}")

        for pos in ["GK", "DEF", "MID", "FWD"]:
            pos_mask = (holdout_df["position"] == pos).values & valid
            if pos_mask.sum() > 0:
                pos_mae = np.mean(np.abs(preds[pos_mask] - actual[pos_mask]))
                print(f"    {pos}: MAE={pos_mae:.4f} (n={pos_mask.sum()})")

        # Save model
        model_dir = Path(f"models/ab_test_{label.lower()}")
        model_dir.mkdir(parents=True, exist_ok=True)
        predictor.save(model_dir)

        # Feature importance for fpl_xp
        for pos in ["MID"]:  # just check one
            if pos in predictor._models:
                model = predictor._models[pos]
                feat_names = predictor._feature_names
                importances = model.feature_importance(importance_type="gain")
                xp_idx = feat_names.index("fpl_xp") if "fpl_xp" in feat_names else -1
                if xp_idx >= 0:
                    xp_imp = importances[xp_idx]
                    total_imp = sum(importances)
                    rank = sorted(range(len(importances)), key=lambda i: importances[i], reverse=True).index(xp_idx) + 1
                    print(f"  fpl_xp importance (MID): {xp_imp:.0f} ({100*xp_imp/total_imp:.1f}%), rank #{rank}")

        # Run MILP optimizer on holdout
        print(f"\n  MILP Optimizer on {HOLDOUT}:")
        integrator = PredictionIntegrator.from_model(model_dir, data_dir, HOLDOUT)
        loader = SeasonDataLoader(HOLDOUT, DEFAULT_DATA_DIR)
        engine = FPLGameEngine(loader)

        def pred_fn(gw):
            eids = loader.get_all_element_ids(gw)
            return {eid: integrator.get_predicted_points(eid, gw) for eid in eids}

        for max_xfers in [0, 1, 5]:
            pp1 = pred_fn(1)
            cands = build_candidate_pool(loader, 1, pp1)
            result = select_squad(cands, budget=STARTING_BUDGET)

            players = []
            for eid in result.squad_element_ids:
                pos = loader.get_player_position(eid)
                price = loader.get_player_price(eid, 1)
                if pos and price > 0:
                    players.append(PlayerSlot(
                        element_id=eid, position=pos,
                        purchase_price=price, selling_price=price,
                    ))
            eid_to_idx = {p.element_id: i for i, p in enumerate(players)}
            squad = Squad(
                players=players,
                lineup=[eid_to_idx[e] for e in result.lineup_element_ids if e in eid_to_idx],
                bench=[eid_to_idx[e] for e in result.bench_element_ids if e in eid_to_idx],
                captain_idx=eid_to_idx.get(result.captain_id, 0),
                vice_captain_idx=eid_to_idx.get(result.vice_captain_id, 1),
            )
            state = GameState(
                squad=squad, bank=STARTING_BUDGET - result.total_cost,
                free_transfers=INITIAL_FREE_TRANSFERS, current_gw=1,
            )

            for gw in range(1, 39):
                pp = pred_fn(gw)
                c = build_candidate_pool(loader, gw, pp)
                try:
                    opt = optimize_transfers(state, c, max_transfers=max_xfers)
                    action = to_engine_action(opt)
                except RuntimeError:
                    action = EngineAction()
                try:
                    state, res = engine.step(state, action)
                except ValueError:
                    state, res = engine.step(state, EngineAction())

            print(f"    {max_xfers} xfer/GW: {state.total_points} net pts")

    # Reference
    print(f"\n{'='*70}")
    print("REFERENCE")
    print(f"{'='*70}")
    print("Oracle (5 xfer/GW, 2024-25): 3806")
    print("Best human 2024-25:          ~2810")


if __name__ == "__main__":
    main()
