# FPL-RL

Predict-and-optimize system for Fantasy Premier League that combines LightGBM point predictions with MILP squad optimization, achieving backtest scores above the all-time human record.

Replays historical seasons (2016-17 to 2024-25) using real player data from [vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League), with full FPL game rules encoded as a Gymnasium environment compatible with [MaskablePPO](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html).

## Results (2024-25 holdout season)

| Strategy | Net Points | Context |
|----------|-----------|---------|
| No transfers (optimized GW1 squad) | 1,950 | Passive baseline |
| MILP optimizer, 1 transfer/GW | **2,918** | Human-realistic |
| MILP optimizer, 5 transfers/GW | **3,171** | Aggressive |
| Oracle (perfect foresight, 5 xfers) | 3,713 | Theoretical ceiling |
| Best human 2024-25 | ~2,810 | Lovro Budisin |
| Best human ever | ~2,844 | Jamie Pigott, 2021-22 |

Prediction model: 0.787 per-GW correlation with actuals (86 features, LightGBM). All features verified point-in-time safe (no lookahead bias).

## Architecture

```
src/fpl_rl/
├── data/          # Historical data loading (vaastav, Understat, FBref, FotMob, odds)
├── engine/        # Pure game logic — zero Gymnasium dependency, stateless
├── env/           # Gymnasium wrappers (standard + hybrid RL/MILP)
├── prediction/    # LightGBM point prediction (86 features, 4 position-specific models)
├── optimizer/     # PuLP/CBC MILP optimizer (squad selection, transfers, lineup)
├── training/      # Multi-season RL training infrastructure
└── utils/         # Shared constants and helpers
```

**Two-layer separation:**

- **Engine** -- Stateless: `step(GameState, EngineAction) -> (GameState, StepResult)`. Never mutates input state. Used by both the RL environment and the MILP optimizer directly.
- **Env** -- Translates between RL primitives (numpy arrays, scalars) and engine dataclasses. Two variants: `FPLEnv` (full action space for RL) and `HybridFPLEnv` (RL picks strategy, MILP picks players).

### Data Flow

```
MultiDiscrete action --> ActionEncoder.decode() --> EngineAction
                                                          |
GameState + EngineAction --> FPLGameEngine.step() --> (new GameState, StepResult)
                                                          |
new GameState --> ObservationBuilder.build() --> flat numpy obs (1363,)
StepResult --> RewardCalculator.calculate() --> scalar reward
```

### Prediction Pipeline

```
merged_gw.csv --> FeaturePipeline.build() --> 86 features per (player, GW)
                                                     |
                                              LightGBM (4 models: GK/DEF/MID/FWD)
                                                     |
                                              predicted_points per player
                                                     |
                                              MILP optimizer --> optimal squad/transfers
```

All rolling features use `.shift(1)` before `.rolling()` to prevent lookahead. Point-in-time safety verified for every feature source (vaastav, Understat, FBref, FotMob, odds, FPL xP).

## Installation

```bash
git clone <repo-url>
cd FPL-RL
pip install -e ".[dev]"
```

Requires Python 3.11+.

**Extras:**
- `pip install -e ".[prediction]"` -- adds lightgbm, scikit-learn
- `pip install -e ".[optimizer]"` -- adds pulp
- `pip install -e ".[dev]"` -- adds pytest, sb3-contrib, tensorboard

## Quick Start

### Evaluate the MILP optimizer

```python
from pathlib import Path
from fpl_rl.data.downloader import DEFAULT_DATA_DIR
from fpl_rl.data.loader import SeasonDataLoader
from fpl_rl.engine.engine import FPLGameEngine
from fpl_rl.prediction.integration import PredictionIntegrator
from fpl_rl.optimizer.squad_selection import select_squad
from fpl_rl.optimizer.transfer_optimizer import optimize_transfers
from fpl_rl.optimizer.types import build_candidate_pool, to_engine_action

# Load prediction model and run on a season
integrator = PredictionIntegrator.from_model(
    Path("models/point_predictor"), DEFAULT_DATA_DIR.parent, "2024-25"
)
```

### Train with MaskablePPO

```python
from sb3_contrib import MaskablePPO
from fpl_rl.env.fpl_env import FPLEnv

env = FPLEnv(season="2023-24")
model = MaskablePPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)
```

### Use the hybrid RL+MILP environment

```python
from fpl_rl.env.hybrid_env import HybridFPLEnv

# RL picks strategy (transfer count + chip), MILP optimizes player selection
env = HybridFPLEnv(season="2023-24", prediction_integrator=integrator)
obs, info = env.reset()
action = env.action_space.sample(mask=env.action_masks())
obs, reward, terminated, truncated, info = env.step(action)
```

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/eval_model.py` | Evaluate a trained RL model with per-GW breakdown |
| `scripts/audit_run.py` | Verify rules, squad validity, and scoring correctness |
| `scripts/oracle_comparison.py` | Compare model predictions vs oracle vs baselines |
| `scripts/retrain_and_test.py` | Retrain the LightGBM predictor and run full evaluation |
| `scripts/plot_training.py` | Visualize training progress from logs |
| `scripts/train_rl.py` | Multi-season RL training with MaskablePPO |
| `scripts/collect_data.py` | Download all historical data sources |

## Spaces

### Action Space (Hybrid)

`MultiDiscrete([6, 6])` -- 12 total mask length

| Index | Dimension | Range | Meaning |
|-------|-----------|-------|---------|
| 0 | transfer_count | 0-5 | Upper bound on transfers (MILP optimizes within this) |
| 1 | chip | 0-5 | 0=none, 1=WC, 2=FH, 3=BB, 4=TC, 5=reserved |

The MILP optimizer handles all player selection, lineup, captain, and bench decisions.

### Action Space (Full RL)

`MultiDiscrete([6, 15,50, 15,50, 15,50, 15,50, 15,50, 15, 15, 8, 15, 15, 15, 6])` -- 18 dims, 5 transfer pairs

### Observation Space

`Box(1363,)` -- flat float32 vector

| Block | Size | Content |
|-------|------|---------|
| Squad | 15 x 24 = 360 | Per-player: position (one-hot), price, form, xG, xA, ICT, minutes, predicted points, squad context |
| Pool | 50 x 19 = 950 | Per-candidate: same features without squad context |
| Global | 53 | GW number, bank, free transfers, team value, total points, 8 chip flags, 20 DGW/BGW flags |

### Reward

```
reward = net_points + 0.1 * (net_points - gw_average) + 0.05 * team_value_change
```

## FPL Rules Encoded

The engine implements 2025/26 FPL rules with 338 passing tests:

- **8 valid formations** (3-4-3 through 5-4-1), always 1 GK in starting XI
- **4 chips x 2 halves** (GW1-19, GW20-38) -- one chip per GW, unused first-half chips expire after GW19
- **Free transfer banking** up to 5 (Wildcard/Free Hit do NOT reset banked transfers)
- **Selling price** = purchase_price + floor(appreciation / 2)
- **Transfer hit** = 4 points per extra transfer beyond free allowance
- **Auto-substitution** walks bench in priority order, respects formation validity
- **Captain failover** -- if captain has 0 minutes, vice-captain gets the multiplier
- **Triple Captain** -- 3x multiplier instead of 2x
- **Bench Boost** -- all bench players' points count
- **Free Hit** -- unlimited transfers for one GW, squad reverts next GW

## Data Sources

| Source | Coverage | Features |
|--------|----------|----------|
| [vaastav](https://github.com/vaastav/Fantasy-Premier-League) | 2016-17 to 2024-25 | Points, minutes, goals, assists, xG/xA, ICT, prices, ownership, xP |
| [Understat](https://understat.com) | 2016-17 to 2024-25 | Per-match xG, xA, npxG, shots, key passes, xGChain, xGBuildup |
| [FBref](https://fbref.com) | 2016-17 to 2024-25 | Season-level passing, shooting, defense stats (prior-season features) |
| [FotMob](https://data.fotmob.com) | 2016-17 to 2024-25 | Pass completion, blocks, long balls (fills FBref gaps) |
| [The Odds API](https://the-odds-api.com) | 2020-21 to 2024-25 | Pinnacle pre-match odds (win/draw/loss implied probabilities) |
| [football-data.co.uk](https://football-data.co.uk) | 2016-17 to 2019-20 | Historical Pinnacle closing odds (fills Odds API gap) |

## Testing

```bash
pytest                            # All 338 tests
pytest tests/test_engine/ -v      # Engine unit tests
pytest tests/test_env/ -v         # Env unit tests
pytest tests/test_integration/ -v # SB3 smoke test
```

Tests use hand-crafted CSVs in `tests/test_data/` (18 players, 2 GWs). The `SeasonDataLoader` is monkey-patched in test fixtures to skip downloads.

## Models

Pre-trained models are available in [GitHub Releases](../../releases):

- **point_predictor/** -- LightGBM point prediction (4 position-specific models, 86 features)
- **best_hybrid_model.zip** -- Trained MaskablePPO for the hybrid RL+MILP environment
- **best_model.zip** -- Trained MaskablePPO for the full RL environment

Download and place in the project root:
```bash
# Models go in models/ and runs/ directories
models/point_predictor/  # LightGBM .lgb files + metadata
runs/best_hybrid_model.zip
```

## Roadmap

- [x] **Stage 0:** Gymnasium environment with full FPL rules, action masking, historical replay
- [x] **Stage 1:** LightGBM point prediction model (86 features, 0.787 per-GW correlation)
- [x] **Stage 2:** PuLP/CBC MILP optimizer for squad selection and transfers
- [x] **Stage 3:** Multi-season MaskablePPO training + hybrid RL/MILP environment
- [ ] **Stage 4:** Multi-season holdout backtesting with confidence intervals
- [ ] **Stage 5:** Live deployment for 2025-26 season
