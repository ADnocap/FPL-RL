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
        assert space.shape == (8,)
        assert list(space.nvec) == ACTION_DIMS

    def test_mask_length(self):
        assert MASK_LENGTH == sum(ACTION_DIMS)
        assert MASK_LENGTH == 169


class TestActionEncoder:
    def test_decode_noop(self, loader, sample_state):
        encoder = ActionEncoder(loader)
        encoder.build_candidate_pool(sample_state, 1)

        # action: 0 transfers, everything else ignored
        action = np.array([0, 0, 0, 0, 0, 7, 12, 0])
        engine_action = encoder.decode(action, sample_state)

        assert engine_action.transfers_out == []
        assert engine_action.transfers_in == []
        assert engine_action.chip is None

    def test_decode_one_transfer(self, loader, sample_state):
        encoder = ActionEncoder(loader)
        pool = encoder.build_candidate_pool(sample_state, 1)

        if pool:
            # 1 transfer: out squad idx 6 (DEF5), in pool idx 0
            action = np.array([1, 6, 0, 0, 0, 7, 12, 0])
            engine_action = encoder.decode(action, sample_state)

            assert len(engine_action.transfers_out) == 1
            assert engine_action.transfers_out[0] == 7  # element_id of DEF5
            assert len(engine_action.transfers_in) == 1

    def test_decode_captain(self, loader, sample_state):
        encoder = ActionEncoder(loader)
        encoder.build_candidate_pool(sample_state, 1)

        # Captain idx 12 (FWD1, element 13), vice idx 7 (MID1, element 8)
        action = np.array([0, 0, 0, 0, 0, 12, 7, 0])
        engine_action = encoder.decode(action, sample_state)

        assert engine_action.captain == 13  # FWD1
        assert engine_action.vice_captain == 8  # MID1

    def test_decode_same_captain_vice(self, loader, sample_state):
        encoder = ActionEncoder(loader)
        encoder.build_candidate_pool(sample_state, 1)

        # Same player for captain and vice — vice should be None
        action = np.array([0, 0, 0, 0, 0, 7, 7, 0])
        engine_action = encoder.decode(action, sample_state)

        assert engine_action.captain is not None
        assert engine_action.vice_captain is None

    def test_decode_chip(self, loader, sample_state):
        encoder = ActionEncoder(loader)
        encoder.build_candidate_pool(sample_state, 1)

        # Chip index 3 = bench_boost
        action = np.array([0, 0, 0, 0, 0, 7, 12, 3])
        engine_action = encoder.decode(action, sample_state)

        assert engine_action.chip == "bench_boost"

    def test_decode_unavailable_chip_fallback(self, loader, sample_state):
        encoder = ActionEncoder(loader)
        encoder.build_candidate_pool(sample_state, 1)

        # Use wildcard first
        sample_state.chips.use_chip("wildcard", 1)

        # Try to use wildcard again — should fall back to None
        action = np.array([0, 0, 0, 0, 0, 7, 12, 1])
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
