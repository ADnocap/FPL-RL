"""Simulate a season using a trained RL model and collect per-GW frames."""
from __future__ import annotations

import logging
from pathlib import Path

from fpl_rl.data.downloader import DEFAULT_DATA_DIR
from fpl_rl.data.loader import SeasonDataLoader
from fpl_rl.engine.state import EngineAction, GameState, Squad
from fpl_rl.env.fpl_env import FPLEnv
from fpl_rl.utils.constants import Position

from .schemas import (
    AutoSub,
    GameweekFrame,
    PlayerInfo,
    SimulationResponse,
    TransferInfo,
)

logger = logging.getLogger(__name__)


def _build_name_map(loader: SeasonDataLoader) -> dict[int, str]:
    """Build element_id -> player name mapping from merged_gw data."""
    df = loader._merged_gw[["element", "name"]].drop_duplicates("element")
    return dict(zip(df["element"].astype(int), df["name"].astype(str)))


def _build_team_name_map(loader: SeasonDataLoader) -> dict[int, str]:
    """Build team_id -> team name mapping."""
    team_map: dict[int, str] = {}
    if not loader._teams.empty and "id" in loader._teams.columns and "name" in loader._teams.columns:
        for _, row in loader._teams.iterrows():
            team_map[int(row["id"])] = str(row["name"])
    # Hardcoded fallback for seasons without teams.csv
    if not team_map:
        team_map = {
            1: "Arsenal", 2: "Bournemouth", 3: "Burnley", 4: "Chelsea",
            5: "Crystal Palace", 6: "Everton", 7: "Hull", 8: "Leicester",
            9: "Liverpool", 10: "Man City", 11: "Man Utd", 12: "Middlesbrough",
            13: "Southampton", 14: "Stoke", 15: "Sunderland", 16: "Swansea",
            17: "Spurs", 18: "Watford", 19: "West Brom", 20: "West Ham",
        }
    return team_map


def _get_formation(squad: Squad, lineup_indices: list[int]) -> str:
    """Determine formation string from lineup."""
    counts = {Position.DEF: 0, Position.MID: 0, Position.FWD: 0}
    for idx in lineup_indices:
        pos = squad.players[idx].position
        if pos in counts:
            counts[pos] += 1
    return f"{counts[Position.DEF]}-{counts[Position.MID]}-{counts[Position.FWD]}"


def _make_player_info(
    slot, squad: Squad, name_map: dict[int, str],
    team_name_map: dict[int, str], loader: SeasonDataLoader,
    gw: int, is_captain: bool, is_vice: bool,
) -> PlayerInfo:
    """Build PlayerInfo from a PlayerSlot."""
    eid = slot.element_id
    gw_data = loader.get_player_gw(eid, gw)
    points = int(gw_data.get("total_points", 0)) if gw_data else 0
    team_id = loader.get_player_team(eid)
    team_name = team_name_map.get(team_id, f"Team {team_id}") if team_id else "Unknown"
    return PlayerInfo(
        element_id=eid,
        name=name_map.get(eid, f"Player {eid}"),
        position=slot.position.name,
        team=team_name,
        points=points,
        is_captain=is_captain,
        is_vice_captain=is_vice,
        purchase_price=slot.purchase_price,
        selling_price=slot.selling_price,
    )


def _build_gw_frame(
    gw: int,
    state_before: GameState,
    state_after: GameState,
    info: dict,
    engine_action: EngineAction,
    loader: SeasonDataLoader,
    name_map: dict[int, str],
    team_name_map: dict[int, str],
) -> GameweekFrame:
    """Build a GameweekFrame from before/after state and step info."""
    # Use state_after squad for display (post-transfer, post-auto-sub)
    # Exception: Free Hit reverts squad, so reconstruct from state_before + transfers
    display_squad = state_after.squad
    if state_before.active_chip == "free_hit" and state_after.free_hit_stash is None:
        # Free Hit GW: the state_after has reverted squad
        # Reconstruct the during-GW squad from state_before + transfers
        display_squad = state_before.squad

    captain_eid = display_squad.players[display_squad.captain_idx].element_id
    vice_eid = display_squad.players[display_squad.vice_captain_idx].element_id

    # Build lineup player info
    lineup = []
    for idx in display_squad.lineup:
        slot = display_squad.players[idx]
        lineup.append(_make_player_info(
            slot, display_squad, name_map, team_name_map, loader, gw,
            is_captain=(slot.element_id == captain_eid),
            is_vice=(slot.element_id == vice_eid),
        ))

    # Build bench player info (in priority order)
    bench = []
    for idx in display_squad.bench:
        slot = display_squad.players[idx]
        bench.append(_make_player_info(
            slot, display_squad, name_map, team_name_map, loader, gw,
            is_captain=(slot.element_id == captain_eid),
            is_vice=(slot.element_id == vice_eid),
        ))

    formation = _get_formation(display_squad, display_squad.lineup)

    # Transfers
    transfers_in = []
    for eid in engine_action.transfers_in:
        pos = loader.get_player_position(eid)
        team_id = loader.get_player_team(eid)
        price = loader.get_player_price(eid, gw)
        transfers_in.append(TransferInfo(
            element_id=eid,
            name=name_map.get(eid, f"Player {eid}"),
            position=pos.name if pos else "UNK",
            team=team_name_map.get(team_id, f"Team {team_id}") if team_id else "Unknown",
            price=price,
        ))

    transfers_out = []
    for eid in engine_action.transfers_out:
        pos = loader.get_player_position(eid)
        team_id = loader.get_player_team(eid)
        price = loader.get_player_price(eid, gw)
        transfers_out.append(TransferInfo(
            element_id=eid,
            name=name_map.get(eid, f"Player {eid}"),
            position=pos.name if pos else "UNK",
            team=team_name_map.get(team_id, f"Team {team_id}") if team_id else "Unknown",
            price=price,
        ))

    # Auto-subs
    auto_subs = []
    for out_eid, in_eid in info.get("auto_subs", []):
        auto_subs.append(AutoSub(
            out_name=name_map.get(out_eid, f"Player {out_eid}"),
            in_name=name_map.get(in_eid, f"Player {in_eid}"),
        ))

    # Chips available
    chips = state_after.chips
    chips_available = {
        "wildcard": list(chips.wildcard),
        "free_hit": list(chips.free_hit),
        "bench_boost": list(chips.bench_boost),
        "triple_captain": list(chips.triple_captain),
    }

    return GameweekFrame(
        gw=gw,
        lineup=lineup,
        bench=bench,
        formation=formation,
        transfers_in=transfers_in,
        transfers_out=transfers_out,
        chip_used=info.get("active_chip"),
        gw_points=info.get("gw_points", 0),
        hit_cost=info.get("hit_cost", 0),
        net_points=info.get("net_points", 0),
        total_points=info.get("total_points", 0),
        bank=state_after.bank,
        free_transfers=state_after.free_transfers,
        auto_subs=auto_subs,
        captain_failover=info.get("captain_failover", False),
        chips_available=chips_available,
    )


def simulate_season(
    season: str,
    model_path: str | Path,
    data_dir: Path = DEFAULT_DATA_DIR,
    predictor_model_dir: Path | None = None,
) -> SimulationResponse:
    """Run a trained model through a full season and collect per-GW frames."""
    from sb3_contrib import MaskablePPO

    env = FPLEnv(
        season=season,
        data_dir=data_dir,
        predictor_model_dir=predictor_model_dir,
    )
    model = MaskablePPO.load(str(model_path), env=env, device="cpu")

    name_map = _build_name_map(env.loader)
    team_name_map = _build_team_name_map(env.loader)

    obs, _ = env.reset()
    frames: list[GameweekFrame] = []

    while True:
        gw = env.state.current_gw
        state_before = env.state.copy()

        masks = env.action_masks()
        action, _ = model.predict(obs, deterministic=True, action_masks=masks)
        engine_action = env.action_encoder.decode(action, env.state)

        obs, reward, terminated, truncated, info = env.step(action)

        frame = _build_gw_frame(
            gw, state_before, env.state, info, engine_action,
            env.loader, name_map, team_name_map,
        )
        frames.append(frame)

        if terminated or truncated:
            break

    return SimulationResponse(season=season, gameweeks=frames)
