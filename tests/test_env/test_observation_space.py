"""Tests for observation space builder."""

import numpy as np
import pytest

from fpl_rl.env.observation_space import (
    OBS_DIM,
    ObservationBuilder,
    create_observation_space,
)


class TestObservationSpace:
    def test_space_shape(self):
        space = create_observation_space()
        assert space.shape == (OBS_DIM,)
        assert OBS_DIM == 1363

    def test_space_dtype(self):
        space = create_observation_space()
        assert space.dtype == np.float32


class TestObservationBuilder:
    def test_build_shape(self, loader, sample_state):
        builder = ObservationBuilder(loader)
        obs = builder.build(sample_state, [16, 17, 18])

        assert obs.shape == (OBS_DIM,)
        assert obs.dtype == np.float32

    def test_no_nan(self, loader, sample_state):
        builder = ObservationBuilder(loader)
        obs = builder.build(sample_state, [16, 17, 18])

        assert not np.isnan(obs).any(), "Observation contains NaN"

    def test_no_inf(self, loader, sample_state):
        builder = ObservationBuilder(loader)
        obs = builder.build(sample_state, [16, 17, 18])

        assert not np.isinf(obs).any(), "Observation contains Inf"

    def test_obs_within_space(self, loader, sample_state):
        builder = ObservationBuilder(loader)
        space = create_observation_space()
        obs = builder.build(sample_state, [16, 17, 18])

        assert space.contains(obs)

    def test_empty_pool(self, loader, sample_state):
        builder = ObservationBuilder(loader)
        obs = builder.build(sample_state, [])

        assert obs.shape == (OBS_DIM,)
        assert not np.isnan(obs).any()

    def test_full_pool(self, loader, sample_state):
        builder = ObservationBuilder(loader)
        pool = [16, 17, 18] + [16] * 47  # Pad to 50
        obs = builder.build(sample_state, pool)

        assert obs.shape == (OBS_DIM,)
