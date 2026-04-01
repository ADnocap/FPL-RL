# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test Commands

```bash
# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run only engine unit tests
pytest tests/test_engine/ -v

# Run only env unit tests
pytest tests/test_env/ -v

# Run integration tests (includes SB3 smoke test)
pytest tests/test_integration/ -v

# Run a single test file
pytest tests/test_engine/test_chips.py -v

# Run a single test class or method
pytest tests/test_engine/test_chips.py::TestActivateChip::test_one_chip_per_gw -v
```

## Architecture

The codebase has a strict two-layer separation:

**`engine/`** — Pure game logic with **zero Gymnasium dependency**. Stateless: `step(GameState, EngineAction) → (GameState, StepResult)`. Never mutates input state — all copies are explicit. This module can be used standalone by the MILP optimizer (Stage 2) without any RL infrastructure.

**`env/`** — Thin Gymnasium wrapper. Handles action encoding/decoding (`MultiDiscrete → EngineAction`), observation building (`GameState → flat numpy array`), reward computation, and masking for MaskablePPO.

**`data/`** — Historical data loading from vaastav/Fantasy-Premier-League GitHub repo. `SeasonDataLoader` pre-indexes by `(element_id, gw)` for O(1) lookups and handles DGW aggregation (sums points across multiple fixtures).

### Key Data Flow

```
MultiDiscrete action → ActionEncoder.decode() → EngineAction
                                                      ↓
GameState + EngineAction → FPLGameEngine.step() → (new GameState, StepResult)
                                                      ↓
new GameState → ObservationBuilder.build() → flat numpy obs (1363,)
StepResult → RewardCalculator.calculate() → scalar reward
```

### Spaces

- **Action**: `MultiDiscrete([3, 15, 50, 15, 50, 15, 15, 8, 15, 15, 15, 6])` = 222 mask length
- **Observation**: `Box(1363,)` = 15×24 squad + 50×19 pool + 53 global features

### Important Conventions

- **Prices are in tenths**: `100 = £10.0m`. All price math uses integers to avoid floating-point issues.
- **Lineup/bench are indices into `Squad.players`**, not element_ids. `Squad.lineup = [0, 2, 3, ...]` means players[0], players[2], etc.
- **Invalid actions don't crash**: `FPLEnv.step()` catches ValueError from the engine and falls back to no-op.
- **Candidate pool is rebuilt every GW** in both `reset()` and `step()`.

## Test Data Pattern

Tests use hand-crafted CSVs in `tests/test_data/` (18 players, 2 GWs). The `SeasonDataLoader.__init__` is monkey-patched in `conftest.py` to skip downloads and load from a temp directory instead. **Critical constraint**: sample test data must have max 3 players per team (club limit rule), and the `team` column in `merged_gw.csv` and `cleaned_players.csv` must be consistent.

When adding new test scenarios: add rows to the sample CSVs or create new fixture CSV files rather than mocking the loader methods.

## FPL Rules Encoded

Key 2025/26 rules implemented in the engine:
- 8 valid formations (3-4-3 through 5-4-1), always 1 GK in starting XI
- 4 chips × 2 halves (GW1-19, GW20-38), one chip per GW, unused first-half chips expire after GW19
- Free transfer banking up to 5 (Wildcard/Free Hit do NOT reset banked transfers)
- Selling price = purchase + floor(appreciation / 2)
- Transfer hit = 4 points per extra transfer beyond free allowance
- Auto-substitution walks bench in priority order, respects formation validity
- Captain failover: if captain has 0 minutes, vice-captain gets the multiplier
- "Played" = 1+ minutes OR received a card (for auto-sub purposes)
