"""Action encoding/decoding and masking for the FPL environment."""

from __future__ import annotations

import numpy as np
from gymnasium import spaces

from fpl_rl.data.loader import SeasonDataLoader
from fpl_rl.engine.chips import validate_chip
from fpl_rl.engine.constraints import is_valid_formation
from fpl_rl.engine.state import EngineAction, GameState
from fpl_rl.engine.transfers import calculate_selling_price
from fpl_rl.utils.constants import ALL_CHIPS, VALID_FORMATIONS, Position

# Action space dimensions
MAX_TRANSFERS_PER_STEP = 5  # matches MAX_FREE_TRANSFERS
NUM_TRANSFERS_DIM = MAX_TRANSFERS_PER_STEP + 1  # 0..5 transfers
SQUAD_DIM = 15  # squad index for transfer out / captain / vice captain / bench
CANDIDATE_POOL_DIM = 50  # candidate pool index for transfer in
FORMATION_DIM = 8  # one per valid formation
CHIP_DIM = 6  # 0=none, 1=WC, 2=FH, 3=BB, 4=TC, 5=reserved

# Transfer pairs: 5 out/in slots (unused slots ignored based on num_transfers)
_TRANSFER_DIMS: list[int] = []
for _i in range(MAX_TRANSFERS_PER_STEP):
    _TRANSFER_DIMS.extend([SQUAD_DIM, CANDIDATE_POOL_DIM])

ACTION_DIMS = [
    NUM_TRANSFERS_DIM,  # num_transfers (0-5)
    *_TRANSFER_DIMS,    # 5 × (transfer_out, transfer_in) pairs
    SQUAD_DIM,          # captain_idx
    SQUAD_DIM,          # vice_captain_idx
    FORMATION_DIM,      # formation
    SQUAD_DIM,          # bench_1 (highest auto-sub priority)
    SQUAD_DIM,          # bench_2
    SQUAD_DIM,          # bench_3 (lowest outfield sub priority)
    CHIP_DIM,           # chip
]

MASK_LENGTH = sum(ACTION_DIMS)

CHIP_INDEX_MAP = {
    0: None,
    1: "wildcard",
    2: "free_hit",
    3: "bench_boost",
    4: "triple_captain",
    5: None,  # reserved
}
CHIP_TO_INDEX = {v: k for k, v in CHIP_INDEX_MAP.items() if v is not None}

# Formation index → (DEF, MID, FWD) tuple (same order as VALID_FORMATIONS)
FORMATION_INDEX_MAP = {i: f for i, f in enumerate(VALID_FORMATIONS)}


def create_action_space() -> spaces.MultiDiscrete:
    """Create the MultiDiscrete action space."""
    return spaces.MultiDiscrete(ACTION_DIMS)


class ActionEncoder:
    """Encodes/decodes actions and builds candidate pools."""

    def __init__(self, loader: SeasonDataLoader) -> None:
        self.loader = loader
        self._candidate_pool: list[int] = []  # element_ids
        # Cache: gw -> {Position -> [element_ids ranked by total_points]}
        self._gw_base_pool: dict[int, dict[Position, list[int]]] = {}

    def _get_base_pool(
        self, gw: int, pool_size: int = CANDIDATE_POOL_DIM
    ) -> dict[Position, list[int]]:
        """Compute and cache the per-position ranked player lists for a GW.

        Players are ranked by rolling form (average of previous GWs) to avoid
        lookahead bias — we never use current-GW total_points for ranking.
        """
        if gw in self._gw_base_pool:
            return self._gw_base_pool[gw]

        gw_data = self.loader.get_gameweek_data(gw)
        if gw_data.empty:
            gw_data = self.loader.get_gameweek_data(max(1, gw - 1))

        base: dict[Position, list[int]] = {}
        if gw_data.empty:
            for pos in Position:
                base[pos] = []
            self._gw_base_pool[gw] = base
            return base

        # Deduplicate
        unique = gw_data.drop_duplicates(subset="element")

        # Build position lookup for all unique elements
        element_ids = unique["element"].astype(int).tolist()
        pos_map = {eid: self.loader.get_player_position(eid) for eid in element_ids}

        # Rank by rolling form (past GWs only) to avoid lookahead
        form_scores = {
            eid: self.loader.get_player_form(eid, gw, window=5)
            for eid in element_ids
        }

        for pos in Position:
            pos_eids = [eid for eid in element_ids if pos_map.get(eid) == pos]
            pos_eids.sort(key=lambda eid: form_scores.get(eid, 0.0), reverse=True)
            base[pos] = pos_eids

        self._gw_base_pool[gw] = base
        return base

    def build_candidate_pool(
        self, state: GameState, gw: int, pool_size: int = CANDIDATE_POOL_DIM
    ) -> list[int]:
        """Build a pool of ~50 candidate players for transfers.

        Uses a cached per-GW base pool and cheaply excludes current squad.
        """
        base = self._get_base_pool(gw, pool_size)

        # Check if base pool is empty
        if all(len(v) == 0 for v in base.values()):
            self._candidate_pool = []
            return self._candidate_pool

        squad_ids = {p.element_id for p in state.squad.players}
        per_position = pool_size // 4  # ~12 per position

        candidates: list[int] = []
        for pos in Position:
            count = 0
            for eid in base[pos]:
                if eid not in squad_ids:
                    candidates.append(eid)
                    count += 1
                    if count >= per_position:
                        break

        # Pad to pool_size from remaining players across all positions
        if len(candidates) < pool_size:
            used = set(candidates) | squad_ids
            needed = pool_size - len(candidates)
            for pos in Position:
                for eid in base[pos]:
                    if eid not in used:
                        candidates.append(eid)
                        used.add(eid)
                        needed -= 1
                        if needed <= 0:
                            break
                if needed <= 0:
                    break

        self._candidate_pool = candidates[:pool_size]
        return self._candidate_pool

    def decode(self, action: np.ndarray, state: GameState) -> EngineAction:
        """Decode a MultiDiscrete action array into an EngineAction.

        Invalid combinations fall back to no-op.
        """
        idx = 0
        num_transfers = int(action[idx]); idx += 1

        # Read 5 transfer pairs
        transfer_pairs: list[tuple[int, int]] = []
        for _ in range(MAX_TRANSFERS_PER_STEP):
            out_idx = int(action[idx]); idx += 1
            in_idx = int(action[idx]); idx += 1
            transfer_pairs.append((out_idx, in_idx))

        captain_idx = int(action[idx]); idx += 1
        vice_captain_idx = int(action[idx]); idx += 1
        formation_idx = int(action[idx]); idx += 1
        bench1_idx = int(action[idx]); idx += 1
        bench2_idx = int(action[idx]); idx += 1
        bench3_idx = int(action[idx]); idx += 1
        chip_idx = int(action[idx]); idx += 1

        transfers_out: list[int] = []
        transfers_in: list[int] = []

        for t in range(min(num_transfers, MAX_TRANSFERS_PER_STEP)):
            if not self._candidate_pool:
                break
            out_raw, in_raw = transfer_pairs[t]
            out_id = self._safe_squad_id(state, out_raw)
            in_id = self._safe_pool_id(in_raw)
            if (
                out_id is not None
                and in_id is not None
                and out_id not in transfers_out
                and in_id not in transfers_in
            ):
                transfers_out.append(out_id)
                transfers_in.append(in_id)

        # Captain / vice-captain
        captain = self._safe_squad_id(state, captain_idx)
        vice = self._safe_squad_id(state, vice_captain_idx)
        if captain is not None and vice is not None and captain == vice:
            vice = None

        # Chip
        chip = CHIP_INDEX_MAP.get(chip_idx)
        if chip is not None:
            error = validate_chip(state, chip)
            if error:
                chip = None

        # Formation + lineup/bench
        lineup_eids, bench_eids = self._decode_lineup_bench(
            state, formation_idx, bench1_idx, bench2_idx, bench3_idx,
        )

        return EngineAction(
            transfers_out=transfers_out,
            transfers_in=transfers_in,
            captain=captain,
            vice_captain=vice,
            chip=chip,
            lineup=lineup_eids,
            bench=bench_eids,
        )

    def _decode_lineup_bench(
        self,
        state: GameState,
        formation_idx: int,
        bench1_idx: int,
        bench2_idx: int,
        bench3_idx: int,
    ) -> tuple[list[int] | None, list[int] | None]:
        """Decode formation + bench indices into lineup/bench element_id lists.

        Returns (lineup_eids, bench_eids) or (None, None) on invalid combo.
        """
        players = state.squad.players
        n_players = len(players)

        formation = FORMATION_INDEX_MAP.get(formation_idx)
        if formation is None:
            return None, None

        target_def, target_mid, target_fwd = formation

        # Clamp bench indices to valid range
        bench1_idx = min(bench1_idx, n_players - 1)
        bench2_idx = min(bench2_idx, n_players - 1)
        bench3_idx = min(bench3_idx, n_players - 1)

        # Deduplicate bench selections — keep first occurrence, fill gaps
        # from current bench
        bench_outfield_indices: list[int] = []
        seen: set[int] = set()
        for idx in [bench1_idx, bench2_idx, bench3_idx]:
            if idx not in seen and players[idx].position != Position.GK:
                bench_outfield_indices.append(idx)
                seen.add(idx)

        # Fill missing bench slots from the current bench (outfield only)
        if len(bench_outfield_indices) < 3:
            for idx in state.squad.bench:
                if (
                    idx not in seen
                    and idx < n_players
                    and players[idx].position != Position.GK
                ):
                    bench_outfield_indices.append(idx)
                    seen.add(idx)
                    if len(bench_outfield_indices) >= 3:
                        break

        if len(bench_outfield_indices) < 3:
            return None, None

        bench_outfield_indices = bench_outfield_indices[:3]

        # Find the 2 GK indices
        gk_indices = [
            i for i in range(n_players) if players[i].position == Position.GK
        ]
        if len(gk_indices) != 2:
            return None, None

        # Bench set (outfield)
        bench_set = set(bench_outfield_indices)

        # Starting XI = all players NOT on bench (outfield) + 1 GK
        outfield_starters = [
            i for i in range(n_players)
            if players[i].position != Position.GK and i not in bench_set
        ]

        # Count positions in outfield starters
        from collections import Counter
        pos_counts = Counter(players[i].position for i in outfield_starters)
        actual = (
            pos_counts.get(Position.DEF, 0),
            pos_counts.get(Position.MID, 0),
            pos_counts.get(Position.FWD, 0),
        )

        if actual != (target_def, target_mid, target_fwd):
            # Formation doesn't match — fall back to current lineup
            return None, None

        # Pick starting GK: prefer the one currently in the lineup
        starting_gk = gk_indices[0]
        for gk_i in gk_indices:
            if gk_i in state.squad.lineup:
                starting_gk = gk_i
                break
        bench_gk = [g for g in gk_indices if g != starting_gk][0]

        # Build lineup and bench as element_ids
        lineup_indices = [starting_gk] + outfield_starters
        bench_indices = bench_outfield_indices + [bench_gk]

        lineup_eids = [players[i].element_id for i in lineup_indices]
        bench_eids = [players[i].element_id for i in bench_indices]

        return lineup_eids, bench_eids

    def get_action_mask(
        self, state: GameState, *, preseason: bool = False,
    ) -> np.ndarray:
        """Build a flat boolean mask for MaskablePPO.

        Conservative per-dimension masks. Combined validation happens in decode().
        When preseason=True, all chips are masked (no chip use before season).
        """
        mask = np.ones(MASK_LENGTH, dtype=bool)
        offset = 0

        # num_transfers: allow 0..MAX_TRANSFERS_PER_STEP
        offset += NUM_TRANSFERS_DIM

        # 5 transfer pairs: (transfer_out, transfer_in)
        pool_len = len(self._candidate_pool)
        for _ in range(MAX_TRANSFERS_PER_STEP):
            # transfer_out: all 15 squad positions valid
            offset += SQUAD_DIM
            # transfer_in: mask out empty pool slots
            for i in range(CANDIDATE_POOL_DIM):
                if i >= pool_len:
                    mask[offset + i] = False
            offset += CANDIDATE_POOL_DIM

        # captain_idx: restrict to current lineup players only
        lineup_set = set(state.squad.lineup)
        for i in range(SQUAD_DIM):
            if i not in lineup_set:
                mask[offset + i] = False
        offset += SQUAD_DIM

        # vice_captain_idx: restrict to current lineup players only
        for i in range(SQUAD_DIM):
            if i not in lineup_set:
                mask[offset + i] = False
        offset += SQUAD_DIM

        # formation: mask based on achievable formations given squad
        achievable = self._get_achievable_formations(state)
        for i in range(FORMATION_DIM):
            if i not in achievable:
                mask[offset + i] = False
        offset += FORMATION_DIM

        # bench_1, bench_2, bench_3: mask GK positions (only outfield on bench)
        for _bench_dim in range(3):
            for i in range(SQUAD_DIM):
                if i < len(state.squad.players):
                    if state.squad.players[i].position == Position.GK:
                        mask[offset + i] = False
                else:
                    mask[offset + i] = False
            offset += SQUAD_DIM

        # chip: mask unavailable chips (all masked during pre-season)
        for i in range(CHIP_DIM):
            if preseason:
                if i != 0:  # only allow "no chip" during pre-season
                    mask[offset + i] = False
            else:
                chip_name = CHIP_INDEX_MAP.get(i)
                if chip_name is None:
                    if i == 5:  # reserved slot
                        mask[offset + i] = False
                else:
                    if not state.chips.is_available(chip_name, state.current_gw):
                        mask[offset + i] = False
        # Always allow chip=0 (none)

        return mask

    @staticmethod
    def _get_achievable_formations(state: GameState) -> set[int]:
        """Return formation indices achievable with current squad composition.

        A formation is achievable if benching 3 outfield players can produce
        exactly that DEF/MID/FWD split in the remaining 10 outfield starters.
        """
        from collections import Counter
        players = state.squad.players
        pos_counts = Counter(
            p.position for p in players if p.position != Position.GK
        )
        n_def = pos_counts.get(Position.DEF, 0)
        n_mid = pos_counts.get(Position.MID, 0)
        n_fwd = pos_counts.get(Position.FWD, 0)

        achievable: set[int] = set()
        for idx, (t_def, t_mid, t_fwd) in FORMATION_INDEX_MAP.items():
            # Need to bench exactly (n_def - t_def) DEFs, etc.
            bench_def = n_def - t_def
            bench_mid = n_mid - t_mid
            bench_fwd = n_fwd - t_fwd
            if bench_def >= 0 and bench_mid >= 0 and bench_fwd >= 0:
                if bench_def + bench_mid + bench_fwd == 3:
                    achievable.add(idx)

        return achievable

    def _safe_squad_id(self, state: GameState, idx: int) -> int | None:
        """Safely get element_id from squad by index."""
        if 0 <= idx < len(state.squad.players):
            return state.squad.players[idx].element_id
        return None

    def _safe_pool_id(self, idx: int) -> int | None:
        """Safely get element_id from candidate pool by index."""
        if 0 <= idx < len(self._candidate_pool):
            return self._candidate_pool[idx]
        return None
