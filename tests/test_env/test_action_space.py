"""Tests for action encoding/decoding and masking."""

import numpy as np
import pytest

from fpl_rl.env.action_space import (
    ACTION_DIMS,
    MASK_LENGTH,
    ActionEncoder,
    create_action_space,
)


class TestActionSpace:
    def test_space_dimensions(self):
        space = create_action_space()
        assert space.shape == (12,)
        assert list(space.nvec) == ACTION_DIMS

    def test_mask_length(self):
        assert MASK_LENGTH == sum(ACTION_DIMS)
        assert MASK_LENGTH == 222


class TestActionEncoder:
    def test_decode_noop(self, loader, sample_state):
        encoder = ActionEncoder(loader)
        encoder.build_candidate_pool(sample_state, 1)

        # action: 0 transfers, everything else ignored
        action = np.array([0, 0, 0, 0, 0, 7, 12, 0, 3, 4, 5, 0])
        engine_action = encoder.decode(action, sample_state)

        assert engine_action.transfers_out == []
        assert engine_action.transfers_in == []
        assert engine_action.chip is None

    def test_decode_one_transfer(self, loader, sample_state):
        encoder = ActionEncoder(loader)
        pool = encoder.build_candidate_pool(sample_state, 1)

        if pool:
            # 1 transfer: out squad idx 6 (DEF5), in pool idx 0
            action = np.array([1, 6, 0, 0, 0, 7, 12, 0, 3, 4, 5, 0])
            engine_action = encoder.decode(action, sample_state)

            assert len(engine_action.transfers_out) == 1
            assert engine_action.transfers_out[0] == 7  # element_id of DEF5
            assert len(engine_action.transfers_in) == 1

    def test_decode_captain(self, loader, sample_state):
        encoder = ActionEncoder(loader)
        encoder.build_candidate_pool(sample_state, 1)

        # Captain idx 12 (FWD1, element 13), vice idx 7 (MID1, element 8)
        action = np.array([0, 0, 0, 0, 0, 12, 7, 0, 3, 4, 5, 0])
        engine_action = encoder.decode(action, sample_state)

        assert engine_action.captain == 13  # FWD1
        assert engine_action.vice_captain == 8  # MID1

    def test_decode_same_captain_vice(self, loader, sample_state):
        encoder = ActionEncoder(loader)
        encoder.build_candidate_pool(sample_state, 1)

        # Same player for captain and vice — vice should be None
        action = np.array([0, 0, 0, 0, 0, 7, 7, 0, 3, 4, 5, 0])
        engine_action = encoder.decode(action, sample_state)

        assert engine_action.captain is not None
        assert engine_action.vice_captain is None

    def test_decode_chip(self, loader, sample_state):
        encoder = ActionEncoder(loader)
        encoder.build_candidate_pool(sample_state, 1)

        # Chip index 3 = bench_boost (now at position 11)
        action = np.array([0, 0, 0, 0, 0, 7, 12, 0, 3, 4, 5, 3])
        engine_action = encoder.decode(action, sample_state)

        assert engine_action.chip == "bench_boost"

    def test_decode_unavailable_chip_fallback(self, loader, sample_state):
        encoder = ActionEncoder(loader)
        encoder.build_candidate_pool(sample_state, 1)

        # Use wildcard first
        sample_state.chips.use_chip("wildcard", 1)

        # Try to use wildcard again — should fall back to None
        action = np.array([0, 0, 0, 0, 0, 7, 12, 0, 3, 4, 5, 1])
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

        # Formation 3 = (4, 4, 2) — check if achievable with these bench picks
        action = np.array([0, 0, 0, 0, 0, 7, 12, 3, b1, b2, b3, 0])
        engine_action = encoder.decode(action, sample_state)

        # If the formation matched, lineup/bench should be set
        if engine_action.lineup is not None:
            assert len(engine_action.lineup) == 11
            assert len(engine_action.bench) == 4

    def test_decode_invalid_formation_fallback(self, loader, sample_state):
        """Invalid formation/bench combo falls back to None (keep current)."""
        encoder = ActionEncoder(loader)
        encoder.build_candidate_pool(sample_state, 1)

        # Bench 3 DEFs with formation 5-3-2 (needs 5 DEFs but only 2 left)
        defs = [
            i for i, p in enumerate(sample_state.squad.players)
            if p.position.name == "DEF"
        ]
        if len(defs) >= 3:
            action = np.array([0, 0, 0, 0, 0, 7, 12, 7, defs[0], defs[1], defs[2], 0])
            # formation 7 = (5,4,1) needs 5 DEF, but we benched 3 of 5 → only 2 left
            engine_action = encoder.decode(action, sample_state)
            assert engine_action.lineup is None
            assert engine_action.bench is None

    def test_mask_gk_excluded_from_bench(self, loader, sample_state):
        """GK positions should be masked out in bench dimensions."""
        encoder = ActionEncoder(loader)
        encoder.build_candidate_pool(sample_state, 1)

        mask = encoder.get_action_mask(sample_state)

        # Find GK indices in squad
        gk_indices = [
            i for i, p in enumerate(sample_state.squad.players)
            if p.position.name == "GK"
        ]

        # Check bench dims (offset: 3+15+50+15+50+15+15+8 = 171)
        bench_offset = 3 + 15 + 50 + 15 + 50 + 15 + 15 + 8
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
            # Should always produce a valid EngineAction
            assert isinstance(engine_action.transfers_out, list)
            assert isinstance(engine_action.transfers_in, list)
            assert len(engine_action.transfers_out) == len(engine_action.transfers_in)
