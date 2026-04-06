#!/usr/bin/env python3
"""Compare shifted vs unshifted xP predictor models."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fpl_rl.data.downloader import DEFAULT_DATA_DIR
from fpl_rl.data.loader import SeasonDataLoader
from fpl_rl.engine.engine import FPLGameEngine
from fpl_rl.engine.state import ChipState, EngineAction, GameState, PlayerSlot, Squad
from fpl_rl.optimizer.squad_selection import select_squad
from fpl_rl.optimizer.transfer_optimizer import optimize_transfers
from fpl_rl.optimizer.types import build_candidate_pool, to_engine_action
from fpl_rl.prediction.integration import PredictionIntegrator
from fpl_rl.prediction.model import PointPredictor
from fpl_rl.prediction.feature_pipeline import FeaturePipeline
from fpl_rl.prediction.id_resolver import IDResolver
from fpl_rl.utils.constants import INITIAL_FREE_TRANSFERS, STARTING_BUDGET

data_dir = DEFAULT_DATA_DIR.parent
V1 = "models/point_predictor"
V2 = "models/point_predictor_v2"


def run_optimizer(model_dir, season, max_xfers):
    loader = SeasonDataLoader(season, DEFAULT_DATA_DIR)
    engine = FPLGameEngine(loader)
    integrator = PredictionIntegrator.from_model(Path(model_dir), data_dir, season)

    def pred_fn(gw):
        eids = loader.get_all_element_ids(gw)
        return {eid: integrator.get_predicted_points(eid, gw) for eid in eids}

    pp1 = pred_fn(1)
    candidates = build_candidate_pool(loader, 1, pp1)
    result = select_squad(candidates, budget=STARTING_BUDGET)

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
        cands = build_candidate_pool(loader, gw, pp)
        try:
            opt = optimize_transfers(state, cands, max_transfers=max_xfers)
            action = to_engine_action(opt)
        except RuntimeError:
            action = EngineAction()
        try:
            state, res = engine.step(state, action)
        except ValueError:
            state, res = engine.step(state, EngineAction())
    return state.total_points


def get_mae_corr(model_dir, season):
    resolver = IDResolver(data_dir)
    pipeline = FeaturePipeline(data_dir, resolver, [season])
    df = pipeline.build()
    predictor = PointPredictor.load(Path(model_dir))
    preds = predictor.predict(df)
    actual = df["target"].values
    valid = ~np.isnan(actual)
    mae = np.mean(np.abs(preds[valid] - actual[valid]))
    corr = np.corrcoef(preds[valid], actual[valid])[0, 1]
    return mae, corr


print("=" * 70)
print("COMPREHENSIVE COMPARISON: UNSHIFTED xP vs SHIFTED xP")
print("=" * 70)

for season in ["2023-24", "2024-25"]:
    label = "(IN-SAMPLE)" if season == "2023-24" else "(HOLDOUT)"
    print(f"\n--- {season} {label} ---")

    mae_v1, corr_v1 = get_mae_corr(V1, season)
    mae_v2, corr_v2 = get_mae_corr(V2, season)

    print(f"  Prediction MAE:  unshifted={mae_v1:.4f}  shifted={mae_v2:.4f}  diff={mae_v2-mae_v1:+.4f}")
    print(f"  Prediction Corr: unshifted={corr_v1:.4f}  shifted={corr_v2:.4f}  diff={corr_v2-corr_v1:+.4f}")

    for xfers in [0, 1, 5]:
        pts_v1 = run_optimizer(V1, season, xfers)
        pts_v2 = run_optimizer(V2, season, xfers)
        print(f"  MILP {xfers} xfer/GW:  unshifted={pts_v1:>5}  shifted={pts_v2:>5}  diff={pts_v2-pts_v1:>+5}")

print()
print("Oracle scores (perfect foresight, 5 xfer/GW):")
print("  2023-24: 4139")
print("  2024-25: 3806")
print()
print("Best human 2024-25: ~2810")
print("Best human ever:    ~2844")
