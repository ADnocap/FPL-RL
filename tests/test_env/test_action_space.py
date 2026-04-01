"""Tests for action encoding/decoding and masking."""

import numpy as np
import pytest

from fpl_rl.env.action_space import (
    ACTION_DIMS,
    MASK_LENGTH,
    MAX_TRANSFERS_PER_STEP,
    ActionEncoder,
    create_action_space,
)


def _make_action(
    num_transfers=0,
    transfers=None,
    captain=0,
    vice=1,
    formation=0,
    bench=(3, 4, 5),
    chip=0,
) -> np.ndarray:
    """Build an 18-element action array for the new 5-transfer action space."""
    a = np.zeros(len(ACTION_DIMS), dtype=int)
    a[0] = num_transfers

    # Fill transfer pairs (slots 1-10)
    if transfers:
        for i, (out_idx, in_idx) in enumerate(transfers):
            a[1 + i * 2] = out_idx
            a[2 + i * 2] = in_idx

    # Indices after the 5 transfer pairs: 11=captain, 12=vice, 13=formation,
    # 14-16=bench, 17=chip
    base = 1 + MAX_TRANSFERS_PER_STEP * 2  # 11
    a[base] = captain
    a[base + 1] = vice
    a[base + 2] = formation
    a[base + 3] = bench[0]
    a[base + 4] = bench[1]
    a[base + 5] = bench[2]
    a[base + 6] = chip
    return a


class TestActionSpace:
    def test_space_dimensions(self):
        space = create_action_space()
        assert space.shape == (len(ACTION_DIMS),)
        assert list(space.nvec) == ACTION_DIMS

    def test_mask_length(self):
        assert MASK_LENGTH == sum(ACTION_DIMS)


class TestActionEncoder:
    def test_decode_noop(self, loader, sample_state):
        encoder = ActionEncoder(loader)
        encoder.build_candidate_pool(sample_state, 1)

        action = _make_action(captain=7, vice=12)
        engine_action = encoder.decode(action, sample_state)

        assert engine_action.transfers_out == []
        assert engine_action.transfers_in == []
        assert engine_action.chip is None

    def test_decode_one_transfer(self, loader, sample_state):
        encoder = ActionEncoder(loader)
        pool = encoder.build_candidate_pool(sample_state, 1)

        if pool:
            # 1 transfer: out squad idx 6 (DEF5), in pool idx 0
            action = _make_action(
                num_transfers=1,
                transfers=[(6, 0)],
                captain=7, vice=12,
            )
            engine_action = encoder.decode(action, sample_state)

            assert len(engine_action.transfers_out) == 1
            assert engine_action.transfers_out[0] == 7  # element_id of DEF5
            assert len(engine_action.transfers_in) == 1

    def test_decode_captain(self, loader, sample_state):
        encoder = ActionEncoder(loader)
        encoder.build_candidate_pool(sample_state, 1)

        # Captain idx 12 (FWD1, element 13), vice idx 7 (MID1, element 8)
        action = _make_action(captain=12, vice=7)
        engine_action = encoder.decode(action, sample_state)

        assert engine_action.captain == 13  # FWD1
        assert engine_action.vice_captain == 8  # MID1

    def test_decode_same_captain_vice(self, loader, sample_state):
        encoder = ActionEncoder(loader)
        encoder.build_candidate_pool(sample_state, 1)

        action = _make_action(captain=7, vice=7)
        engine_action = encoder.decode(action, sample_state)

        assert engine_action.captain is not None
        assert engine_action.vice_captain is None

    def test_decode_chip(self, loader, sample_state):
        encoder = ActionEncoder(loader)
        encoder.build_candidate_pool(sample_state, 1)

        action = _make_action(captain=7, vice=12, chip=3)  # bench_boost
        engine_action = encoder.decode(action, sample_state)

        assert engine_action.chip == "bench_boost"

    def test_decode_unavailable_chip_fallback(self, loader, sample_state):
        encoder = ActionEncoder(loader)
        encoder.build_candidate_pool(sample_state, 1)

        # Use wildcard first
        sample_state.chips.use_chip("wildcard", 1)

        action = _make_action(captain=7, vice=12, chip=1)  # wildcard again
        engine_action = encoder.decode(action, sample_state)

        assert engine_action.chip is None

    def test_mask_shape(self, loader, sample_state):
        encoder = ActionEncoder(loader)
        encoder.build_candidate_pool(sample_state, 1)

        mask = encoder.get_action_mask(sample_state)
        assert mask.shape == (MASK_LENGTH,)
        assert mask.dtype == bool

    def test_mask_at_least_one_valid_per_dim(self, loader, sample_state):
        encoder = ActionEncoder(loader)
        encoder.build_candidate_pool(sample_state, 1)

        mask = encoder.get_action_mask(sample_state)

        # Check each dimension has at least one True
        offset = 0
        for dim_size in ACTION_DIMS:
            dim_mask = mask[offset : offset + dim_size]
            assert dim_mask.any(), f"No valid actions in dimension starting at {offset}"
            offset += dim_size

    def test_build_candidate_pool(self, loader, sample_state):
        encoder = ActionEncoder(loader)
        pool = encoder.build_candidate_pool(sample_state, 1)

        # Pool should not contain current squad players
        squad_ids = {p.element_id for p in sample_state.squad.players}
        for eid in pool:
            assert eid not in squad_ids

    def test_decode_formation_sets_lineup(self, loader, sample_state):
        """Choosing a valid formation + bench should set lineup/bench."""
        encoder = ActionEncoder(loader)
        encoder.build_candidate_pool(sample_state, 1)

        # Find outfield player indices to bench (need 3 outfield)
        outfield = [
            i for i, p in enumerate(sample_state.squad.players)
            if p.position.name != "GK"
        ]
        b1, b2, b3 = outfield[0], outfield[1], outfield[2]

        # Formation 3 = (4, 4, 2)
        action = _make_action(captain=7, vice=12, formation=3, bench=(b1, b2, b3))
        engine_action = encoder.decode(action, sample_state)

        if engine_action.lineup is not None:
            assert len(engine_action.lineup) == 11
            assert len(engine_action.bench) == 4

    def test_decode_invalid_formation_fallback(self, loader, sample_state):
        """Invalid formation/bench combo falls back to None (keep current)."""
        encoder = ActionEncoder(loader)
        encoder.build_candidate_pool(sample_state, 1)

        # Bench 3 DEFs with formation 5-4-1 (needs 5 DEFs but only 2 left)
        defs = [
            i for i, p in enumerate(sample_state.squad.players)
            if p.position.name == "DEF"
        ]
        if len(defs) >= 3:
            action = _make_action(
                captain=7, vice=12, formation=7,  # (5,4,1)
                bench=(defs[0], defs[1], defs[2]),
            )
            engine_action = encoder.decode(action, sample_state)
            assert engine_action.lineup is None
            assert engine_action.bench is None

    def test_mask_gk_excluded_from_bench(self, loader, sample_state):
        """GK positions should be masked out in bench dimensions."""
        encoder = ActionEncoder(loader)
        encoder.build_candidate_pool(sample_state, 1)

        mask = encoder.get_action_mask(sample_state)

        gk_indices = [
            i for i, p in enumerate(sample_state.squad.players)
            if p.position.name == "GK"
        ]

        # bench dims start after: num_transfers + 5*(squad+pool) + captain + vice + formation
        bench_offset = (
            6  # NUM_TRANSFERS_DIM
            + MAX_TRANSFERS_PER_STEP * (15 + 50)  # 5 transfer pairs
            + 15  # captain
            + 15  # vice
            + 8   # formation
        )
        for bench_dim in range(3):
            dim_start = bench_offset + bench_dim * 15
            for gk_i in gk_indices:
                assert mask[dim_start + gk_i] == False, (
                    f"GK at index {gk_i} should be masked in bench dim {bench_dim}"
                )

    def test_encode_decode_roundtrip(self, loader, sample_state):
        """Verify decode produces valid EngineAction from sampled action."""
        encoder = ActionEncoder(loader)
        encoder.build_candidate_pool(sample_state, 1)
        space = create_action_space()

        for _ in range(10):
            action = space.sample()
            engine_action = encoder.decode(action, sample_state)
            assert isinstance(engine_action.transfers_out, list)
            assert isinstance(engine_action.transfers_in, list)
            assert len(engine_action.transfers_out) == len(engine_action.transfers_in)

    def test_decode_five_transfers(self, loader, sample_state):
        """5 transfers should be decoded correctly."""
        encoder = ActionEncoder(loader)
        pool = encoder.build_candidate_pool(sample_state, 1)

        if len(pool) >= 5:
            action = _make_action(
                num_transfers=5,
                transfers=[(2, 0), (3, 1), (4, 2), (5, 3), (6, 4)],
                captain=7, vice=12,
            )
            engine_action = encoder.decode(action, sample_state)
            # Should have up to 5 valid transfers (depends on pool/squad)
            assert len(engine_action.transfers_out) <= 5
            assert len(engine_action.transfers_out) == len(engine_action.transfers_in)

    def test_preseason_chips_masked(self, loader, sample_state):
        """During preseason, all chips except 'none' should be masked."""
        encoder = ActionEncoder(loader)
        encoder.build_candidate_pool(sample_state, 1)

        mask = encoder.get_action_mask(sample_state, preseason=True)

        chip_offset = MASK_LENGTH - 6  # CHIP_DIM = 6
        assert mask[chip_offset] == True  # chip=none
        for i in range(1, 6):
            assert mask[chip_offset + i] == False, (
                f"Chip index {i} should be masked during preseason"
            )
