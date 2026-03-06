"""Tests for PredictionIntegrator and ObservationBuilder integration."""

from __future__ import annotations

import numpy as np
import pytest

from fpl_rl.prediction.integration import PredictionIntegrator


class TestPredictionIntegrator:
    def test_lookup_existing_key(self) -> None:
        predictions = {(10, 1): 5.5, (10, 2): 3.2, (20, 1): 2.1}
        integrator = PredictionIntegrator(predictions)

        assert integrator.get_predicted_points(10, 1) == pytest.approx(5.5)
        assert integrator.get_predicted_points(20, 1) == pytest.approx(2.1)

    def test_lookup_missing_key_returns_zero(self) -> None:
        predictions = {(10, 1): 5.5}
        integrator = PredictionIntegrator(predictions)

        assert integrator.get_predicted_points(99, 1) == 0.0
        assert integrator.get_predicted_points(10, 99) == 0.0

    def test_len(self) -> None:
        predictions = {(10, 1): 5.5, (10, 2): 3.2}
        integrator = PredictionIntegrator(predictions)
        assert len(integrator) == 2

    def test_empty_predictions(self) -> None:
        integrator = PredictionIntegrator({})
        assert integrator.get_predicted_points(1, 1) == 0.0
        assert len(integrator) == 0


class TestObservationBuilderIntegration:
    """Test that ObservationBuilder correctly uses the integrator."""

    def test_without_integrator_features18_is_zero(self, loader, sample_state) -> None:
        from fpl_rl.env.observation_space import ObservationBuilder

        obs_builder = ObservationBuilder(loader)
        obs = obs_builder.build(sample_state, [16, 17, 18])

        # features[18] for first squad player should be 0.0
        # First player starts at offset 0, feature 18 is at index 18
        assert obs[18] == 0.0

    def test_with_integrator_features18_is_populated(self, loader, sample_state) -> None:
        from fpl_rl.env.observation_space import ObservationBuilder

        # Create integrator with predictions for squad players
        predictions = {}
        for player in sample_state.squad.players:
            predictions[(player.element_id, sample_state.current_gw)] = 7.5

        integrator = PredictionIntegrator(predictions)
        obs_builder = ObservationBuilder(loader, prediction_integrator=integrator)
        obs = obs_builder.build(sample_state, [16, 17, 18])

        # features[18] for first squad player should be 7.5 / 15.0 = 0.5
        assert obs[18] == pytest.approx(0.5)

    def test_integrator_partial_predictions(self, loader, sample_state) -> None:
        from fpl_rl.env.observation_space import ObservationBuilder

        # Only provide prediction for first player
        first_eid = sample_state.squad.players[0].element_id
        predictions = {(first_eid, sample_state.current_gw): 6.0}

        integrator = PredictionIntegrator(predictions)
        obs_builder = ObservationBuilder(loader, prediction_integrator=integrator)
        obs = obs_builder.build(sample_state, [16, 17, 18])

        # First player: 6.0 / 15.0 = 0.4
        assert obs[18] == pytest.approx(0.4)
        # Second player (no prediction): 0.0 / 15.0 = 0.0
        assert obs[24 + 18] == pytest.approx(0.0)
