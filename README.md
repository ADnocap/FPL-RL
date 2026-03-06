# FPL-RL

Reinforcement learning environment for Fantasy Premier League. Replays historical seasons using real player data from [vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League), with full FPL game rules encoded as a Gymnasium environment compatible with [MaskablePPO](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html).

## Architecture

```
src/fpl_rl/
├── data/       # Historical data loading (vaastav GitHub dataset)
├── engine/     # Pure game logic — zero Gymnasium dependency
├── env/        # Thin Gymnasium wrapper (obs, actions, reward, masking)
└── utils/      # Shared constants and helpers
```

**Two-layer separation:**

- **Engine** — Stateless: `step(GameState, EngineAction) → (GameState, StepResult)`. Never mutates input state. Can be used standalone by an MILP optimizer without any RL infrastructure.
- **Env** — Translates between RL primitives (numpy arrays, scalars) and engine dataclasses. Handles action encoding/decoding, observation building, reward computation, and action masking.

### Data Flow

```
MultiDiscrete action → ActionEncoder.decode() → EngineAction
                                                      ↓
GameState + EngineAction → FPLGameEngine.step() → (new GameState, StepResult)
                                                      ↓
new GameState → ObservationBuilder.build() → flat numpy obs (1363,)
StepResult → RewardCalculator.calculate() → scalar reward
```

## Installation

```bash
# Clone and install in editable mode with dev dependencies
git clone <repo-url>
cd FPL-RL
pip install -e ".[dev]"
```

Requires Python 3.11+.

**Core dependencies:** gymnasium, numpy, pandas, requests

**Dev dependencies:** pytest, pytest-cov, sb3-contrib, stable-baselines3

## Quick Start

### Create and step through the environment

```python
from fpl_rl.env.fpl_env import FPLEnv

env = FPLEnv(season="2023-24")
obs, info = env.reset(seed=42)

# Take a random valid action
action = env.action_space.sample(mask=env.action_masks())
obs, reward, terminated, truncated, info = env.step(action)

print(f"GW{info['gw']}: {info['gw_points']} pts (net {info['net_points']})")
```

### Train with MaskablePPO

```python
from sb3_contrib import MaskablePPO
from fpl_rl.env.fpl_env import FPLEnv

env = FPLEnv(season="2023-24")
model = MaskablePPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10_000)

# Predict with action masking
obs, _ = env.reset(seed=42)
action, _ = model.predict(obs, action_masks=env.action_masks())
```

## Spaces

### Action Space

`MultiDiscrete([3, 15, 50, 15, 50, 15, 15, 6])` — 169 total mask length

| Index | Dimension      | Range | Meaning                                                         |
| ----- | -------------- | ----- | --------------------------------------------------------------- |
| 0     | num_transfers  | 0-2   | How many transfers to make                                      |
| 1     | transfer_out_1 | 0-14  | Squad index of player to sell                                   |
| 2     | transfer_in_1  | 0-49  | Candidate pool index of player to buy                           |
| 3     | transfer_out_2 | 0-14  | Squad index of second player to sell                            |
| 4     | transfer_in_2  | 0-49  | Candidate pool index of second player to buy                    |
| 5     | captain        | 0-14  | Squad index of captain                                          |
| 6     | vice_captain   | 0-14  | Squad index of vice-captain                                     |
| 7     | chip           | 0-5   | 0=none, 1=wildcard, 2=free_hit, 3=bench_boost, 4=triple_captain |

The candidate pool (50 players) is rebuilt each gameweek, ranked by recent form across all four positions.

### Observation Space

`Box(1363,)` — flat float32 vector

| Block  | Size          | Content                                                                                                                               |
| ------ | ------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| Squad  | 15 x 24 = 360 | Per-player: position (one-hot), price, form, xG, xA, ICT stats, minutes, is_starter, bench_order, is_captain, is_vice, purchase_price |
| Pool   | 50 x 19 = 950 | Per-candidate: position (one-hot), price, form, xG, xA, ICT stats, minutes, points                                                    |
| Global | 53            | GW number, bank, free transfers, team value, total points, 8 chip booleans, 20 DGW flags, 20 BGW flags                                |

### Reward

```
reward = net_points + 0.1 * (net_points - gw_average) + 0.05 * team_value_change
```

- **Primary:** net points (after transfer hit deductions)
- **Relative:** bonus for beating the gameweek average
- **Value:** incentive for growing team value through smart transfers

## FPL Rules Encoded

The engine implements 2025/26 FPL rules:

- **8 valid formations** (3-4-3 through 5-4-1), always 1 GK in starting XI
- **4 chips x 2 halves** (GW1-19, GW20-38) — one chip per GW, unused first-half chips expire after GW19
- **Free transfer banking** up to 5 (Wildcard/Free Hit do NOT reset banked transfers)
- **Selling price** = purchase_price + floor(appreciation / 2)
- **Transfer hit** = 4 points per extra transfer beyond free allowance
- **Auto-substitution** walks bench in priority order, respects formation validity
- **Captain failover** — if captain has 0 minutes, vice-captain gets the multiplier
- **Triple Captain** — 3x multiplier instead of 2x
- **Bench Boost** — all bench players' points count
- **Free Hit** — unlimited transfers for one GW, squad reverts next GW
- **"Played"** = 1+ minutes OR received a card (for auto-sub purposes)

All prices are stored as integers in tenths (100 = £10.0m) to avoid floating-point issues.

## Testing

```bash
# Run all tests (130 tests)
pytest

# Run by layer
pytest tests/test_engine/ -v      # Engine unit tests
pytest tests/test_env/ -v         # Env unit tests
pytest tests/test_integration/ -v # SB3 smoke test

# Single test
pytest tests/test_engine/test_chips.py::TestActivateChip::test_one_chip_per_gw -v
```

Tests use hand-crafted CSVs in `tests/test_data/` (18 players, 2 GWs). The `SeasonDataLoader` is monkey-patched in test fixtures to skip GitHub downloads and load from local test data instead.

## Roadmap

- [x] **Stage 0:** Gymnasium environment with full FPL rules, action masking, historical replay
- [ ] **Stage 1:** XGBoost/LightGBM point prediction model (predicted points as observation feature)
- [ ] **Stage 2:** PuLP/HiGHS MILP optimizer for team selection (uses engine directly, no RL)
- [ ] **Stage 3:** Full MaskablePPO training across multiple historical seasons
