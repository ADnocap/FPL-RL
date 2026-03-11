"""Pydantic response models for the FPL visualizer API."""
from __future__ import annotations
from pydantic import BaseModel


class PlayerInfo(BaseModel):
    element_id: int
    name: str
    position: str  # "GK", "DEF", "MID", "FWD"
    team: str
    points: int
    is_captain: bool
    is_vice_captain: bool
    purchase_price: int
    selling_price: int


class TransferInfo(BaseModel):
    element_id: int
    name: str
    position: str
    team: str
    price: int


class AutoSub(BaseModel):
    out_name: str
    in_name: str


class GameweekFrame(BaseModel):
    gw: int
    lineup: list[PlayerInfo]
    bench: list[PlayerInfo]
    formation: str  # e.g. "4-4-2"
    transfers_in: list[TransferInfo]
    transfers_out: list[TransferInfo]
    chip_used: str | None
    gw_points: int
    hit_cost: int
    net_points: int
    total_points: int
    bank: int
    free_transfers: int
    auto_subs: list[AutoSub]
    captain_failover: bool
    chips_available: dict[str, list[bool]]


class SimulationResponse(BaseModel):
    season: str
    gameweeks: list[GameweekFrame]
