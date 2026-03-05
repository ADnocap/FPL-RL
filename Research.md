# Building an RL agent to dominate Fantasy Premier League

**No production-ready FPL reinforcement learning environment exists today — you'll need to build one yourself, but the payoff is substantial.** The most successful automated FPL system in the literature (Matthews et al., 2012) achieved **top 1% among 2.5 million human managers** using Bayesian Q-learning combined with integer programming. The optimal architecture isn't pure RL but a **hybrid pipeline**: supervised ML for point prediction, mathematical optimization for team selection, and RL exclusively for sequential strategic decisions (transfers, chip timing, hit decisions). This report provides everything needed to build and deploy that system against the real FPL API.

---

## The FPL scoring engine you need to replicate

FPL's points system is position-dependent and more nuanced than most simulators account for. Here is the complete scoring table your environment must implement:

| Action | GK | DEF | MID | FWD |
|--------|-----|------|------|------|
| Goal scored | **10** | **6** | **5** | **4** |
| Assist | 3 | 3 | 3 | 3 |
| Clean sheet (60+ min) | **4** | **4** | 1 | 0 |
| Every 2 goals conceded | -1 | -1 | — | — |
| Every 3 saves | 1 | — | — | — |
| Penalty save | 5 | — | — | — |
| Penalty miss | -2 | -2 | -2 | -2 |
| Playing 1–59 min | 1 | 1 | 1 | 1 |
| Playing 60+ min | 2 | 2 | 2 | 2 |
| Yellow card | -1 | -1 | -1 | -1 |
| Red card | -3 | -3 | -3 | -3 |
| Own goal | -2 | -2 | -2 | -2 |
| Bonus (top 3 BPS in match) | 1–3 | 1–3 | 1–3 | 1–3 |

New for 2025/26: **defensive contribution points** — defenders earn 2 extra points for 10+ combined clearances, blocks, interceptions, and tackles (CBIT) per match; midfielders and forwards earn 2 points for 12+ CBIRT (adding recoveries).

The **Bonus Points System** is the hardest component to simulate. Each player accumulates a BPS score from 30+ Opta statistics during a match — tackles won (2 BPS each), goals (12–24 BPS depending on position and penalty status), assists (9), clean sheets for GK/DEF (12), saves from inside the box (3), successful crosses (1), big chances created (3), and many others. The top three BPS scorers per match receive 3, 2, and 1 bonus FPL points respectively, with tie-breaking rules that can award bonus to more than three players.

**BPS changes for 2025/26**: goalline clearances increased from 3 to 9 BPS; penalty goals now award 12 BPS regardless of position (previously varied by position); saves from inside the box award 3 BPS; tackles are counted individually at 2 BPS each (no longer grouped in threes).

**Assist rule simplification (2025/26)**: assists are now awarded when the ball is received inside the box with only 1 defensive touch between the passer and scorer (previously required no touch or was more restrictive).

**Valid formations**: exactly 8 formations are permitted — 3-4-3, 3-5-2, 4-3-3, 4-4-2, 4-5-1, 5-2-3, 5-3-2, 5-4-1. All require exactly 1 GK in the starting 11.

Captain mechanics double points (or triple with the Triple Captain chip). If your captain gets zero minutes, the vice-captain's points are doubled instead. Auto-substitution happens end-of-gameweek: bench players replace starters who played zero minutes, following bench priority order and respecting formation validity (minimum 3 DEF, 2 MID, 1 FWD). **Playing time clarification**: a player counts as "having played" (and thus is not auto-substituted) if they play 1+ minutes OR receive a yellow/red card without appearing on the pitch.

The **chip system changed significantly for 2025/26**: all four chips (Wildcard, Free Hit, Bench Boost, Triple Captain) are available **twice** — once per half-season (GW1–19, GW20–38). Unused first-half chips are lost at the GW19 deadline. **Only one chip may be activated per gameweek.** The **Assistant Manager chip** (introduced in 2024/25) has been **removed** for 2025/26. Transfer banking now allows accumulating up to **5 free transfers** (previously capped at 2). Wildcard and Free Hit no longer reset your banked transfers.

**Double & Blank Gameweek mechanics**: In a Double Gameweek (DGW), a player's team plays two fixtures — the player earns points from both matches (summed). In a Blank Gameweek (BGW), a player's team has no fixture — the player scores 0 points. Auto-substitution still applies per normal rules.

**AFCON GW16 extra transfers**: During AFCON windows (typically around GW16), managers receive extra free transfers to reach a maximum of 5, helping cope with player absences.

**Price changes** follow a proprietary algorithm driven by net transfer activity scaled by ownership percentage. Reverse-engineering efforts reveal that falls are approximately **6× easier to trigger** than rises, cumulative transfer counters reset after each change, and Wildcard/Free Hit transfers are discounted. Players sell at purchase price plus only **50% of appreciation** (rounded down to £0.1m).

---

## No Gym environment exists, but here's what to build on

The most important finding from surveying the ecosystem: **no Gymnasium-compatible FPL environment exists**. The RL-for-FPL space is remarkably underdeveloped. Here's what does exist, ranked by utility:

**`alan-turing-institute/AIrsenal`** is the most complete FPL game logic implementation available. Built by the Alan Turing Institute, it includes Bayesian match prediction (NumPyro), player-level models, multi-gameweek transfer optimization, and actual FPL API integration for automated transfers and lineups. It implements squad constraints, captaincy, all four chips, budget management, and price tracking. It does not use RL and has no Gym interface, but its game engine is the strongest foundation for building one.

**`sertalpbilal/FPL-Optimization-Tools`** provides production-grade MILP optimization using HiGHS/sasoptpy with full constraint modeling. This serves as both an excellent oracle baseline to benchmark your RL agent against and a reference implementation for the optimization component of the hybrid pipeline.

**`evenmn/fantasy-pl-ai`** is the only repository that combines an FPL game engine with an actual RL agent (tabular Q-learning). However, it has only 17 commits, 1 star, and implements a fraction of FPL rules. It's useful as a conceptual starting point but not as a foundation for serious work.

**`vaastav/Fantasy-Premier-League`** is the canonical dataset — CSV files with gameweek-level stats for every player from 2016/17 through the current season, updated automatically via GitHub Actions. Nearly every FPL ML paper uses this dataset.

**`DanialRamezani/Data-Driven-FPL`** accompanies a 2025 arXiv paper implementing deterministic and robust MILP with hybrid predictive metrics. It's a clean academic reference for the optimization formulation.

Your environment will need to wrap historical data from vaastav's dataset into a Gymnasium `step/reset/reward` interface. The core game logic from AIrsenal provides the most reliable reference for constraint validation, auto-substitution, and chip mechanics.

---

## Data pipeline from four complementary sources

**The FPL Official API** (`https://fantasy.premierleague.com/api/`) is your primary live data source. The `bootstrap-static/` endpoint returns everything in a single call: all ~700 players with **60+ fields each** (price, form, xG, xA, ICT index, ownership, status, expected points), all 20 teams with home/away strength ratings, all 38 gameweek objects with deadlines and chip usage stats, and game settings (budget rules, transfer costs). No authentication needed for read operations.

Key endpoints for a deployed agent:

- **`bootstrap-static/`** — master data dump, call once and cache
- **`element-summary/{player_id}/`** — individual player's fixture list and gameweek history
- **`event/{gw}/live/`** — live points for all players during a gameweek
- **`fixtures/`** — all fixtures with difficulty ratings (FDR, 1–5 scale) and post-match stats
- **`entry/{manager_id}/event/{gw}/picks/`** — any manager's team for a given week
- **`my-team/{manager_id}/`** — your current squad (authenticated)

Write operations (making transfers, setting lineup) require session-cookie authentication via `https://users.premierleague.com/accounts/login/`. The **`fpl` library by amosbastian** (`pip install fpl`) handles this with `await fpl.login(email, password)` and provides async methods for transfers, lineup changes, and data retrieval. Note that FPL has added CAPTCHA challenges in recent seasons that can break automated auth.

**`vaastav/Fantasy-Premier-League`** on GitHub provides 8+ seasons of historical gameweek data in CSV format. The `merged_gw.csv` file per season contains every player's every-gameweek performance — points, goals, assists, minutes, BPS, ICT index, xG/xA (from 2022/23 onward), price, ownership, and transfers. Load it directly:

```python
import pandas as pd
df = pd.read_csv("https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2024-25/gws/merged_gw.csv")
```

**Understat** (`understat.com`) fills the xG/xA gap for pre-2022 seasons. The `understatapi` Python package provides per-player expected goals, expected assists, xG chain, xG buildup, non-penalty xG, and shot-level data going back to 2014/15. **FBref** (`fbref.com`) offers StatsBomb-derived advanced metrics including progressive passes, shot-creating actions, and defensive actions, accessible via the `soccerdata` Python package.

For the RL state representation, encode each player as a **30–50 feature vector** including: rolling form (last 3/5/10 GW averages), xG and xA from Understat, ICT index components (influence, creativity, threat), fixture difficulty of next 3–5 opponents, home/away indicator, minutes consistency (standard deviation of recent minutes), price and price change trend, ownership percentage and transfer momentum, BPS rolling average, set-piece responsibility flags (penalty order, corner order), and availability status. The full game state adds: current 15-player squad with features, bank balance, free transfers available (0–5), chips remaining (up to 4 booleans), gameweek number, and team value.

---

## The hybrid architecture that actually wins

Pure RL is not the optimal approach for FPL. The research unanimously favors a **three-stage hybrid pipeline** that plays to each method's strength:

**Stage 1 — Point prediction (supervised ML).** Train an XGBoost or LightGBM model to predict each player's expected points for upcoming gameweeks. Use the vaastav dataset (2016–present) with features described above. This is a tabular regression problem with ~2,800 gameweeks of training data across 8+ seasons — far more data than the ~7 full seasons available for RL episodes. Target variable is `total_points` for the next gameweek. Multi-horizon prediction (1, 3, and 5 GWs ahead) enables longer-term planning.

**Stage 2 — Team selection (MILP optimization).** Given predicted points, solve the constrained knapsack problem exactly using PuLP with the HiGHS solver. This handles budget (£100m), position constraints (2 GK, 5 DEF, 5 MID, 3 FWD), club limits (max 3 per team), formation validity, and captaincy optimally in milliseconds. This is a solved problem — integer programming guarantees the best possible squad given your predictions.

**Stage 3 — Strategic planning (RL with MaskablePPO).** The RL agent handles only the decisions MILP cannot: when to take point hits for extra transfers, when to activate each chip (wildcard timing relative to fixture swings, bench boost on double gameweeks, free hit during blanks, triple captain on premium double-gameweek players), whether to build team value by targeting rising players, and multi-week transfer planning. This dramatically reduces the action space from combinatorial player selection to a manageable set of strategic choices.

```
┌──────────────────────────────────────────────────────┐
│  XGBoost/LightGBM  →  PuLP/HiGHS MILP  →  MaskablePPO  │
│  (predict xP)         (select squad)       (strategy)     │
└──────────────────────────────────────────────────────┘
```

This decomposition is why Matthews et al. (2012) succeeded: Bayesian prediction + knapsack optimization + MDP for transfers achieved top 1%. Bonello et al. (2019) reached **top 0.5%** (rank ~30,000 of 6.5M) with multi-source prediction and optimization alone.

For the RL component, **MaskablePPO from `sb3-contrib`** is the clear algorithm choice. PPO's stability and sample efficiency are well-suited to the 38-step episodes. Invalid action masking is critical — at any gameweek, many actions are illegal (can't play a used chip, can't exceed budget, can't have more than 3 from one club). The `MaskablePPO` implementation in sb3-contrib handles this natively. The factored action space should be:

```python
action_space = spaces.MultiDiscrete([
    3,   # n_transfers: 0, 1, or 2
    15,  # player_out index (from current squad)
    80,  # player_in index (top-20 per position, pre-filtered)
    15,  # captain choice (from squad)
    5,   # chip: none/wildcard/free_hit/bench_boost/triple_captain
])
```

DQN fails here because it requires enumerating Q-values over the full action space — infeasible for combinatorial actions. SAC is designed for continuous spaces. A3C is superseded by PPO in practice. The Decision Transformer architecture is promising for offline RL but requires large expert-play datasets that don't exist for FPL.

For temporal player form, use a lightweight **Transformer encoder** (2 layers, 4 attention heads, 128-dim) over a context window of the last 6–10 gameweeks per player. Self-attention naturally captures which recent performances are most informative and handles the 38-step episode length better than LSTMs, which suffer from vanishing gradients over long sequences.

---

## Training pipeline and code architecture

**Episode structure:** One episode equals one 38-gameweek season. Training data spans 2016/17 through 2022/23 (7 seasons), validation on 2023/24, testing on 2024/25. This gives only ~7 training episodes for the RL agent — sample efficiency matters enormously. Data augmentation helps: randomize starting squads, vary initial budget allocations, add Gaussian noise to point predictions to simulate forecast uncertainty.

**Handling non-stationarity:** Encode players by statistical features, never by identity. A "midfielder averaging 6.2 points, priced at £7.5m, with 0.45 xG/90" is the same state representation whether it's Salah in 2019 or Palmer in 2024. This lets the agent generalize across seasons despite roster turnover.

**Framework choice:** Stable-Baselines3 with sb3-contrib is the recommended stack. Empirical comparisons show SB3 achieves equivalent or better PPO performance than RLlib with far less complexity. RLlib's distributed training is overkill — your environment runs in microseconds (historical data lookup), not real-time simulation. CleanRL is excellent for research prototyping but lacks MaskablePPO out of the box.

**Reward design:** FPL's weekly scoring provides semi-dense rewards (38 per episode), which is manageable. Use a composite reward:

- **Primary:** Raw gameweek points (already penalizes transfer hits per FPL rules)
- **Auxiliary 1:** Points relative to the gameweek average (×0.1 weight) — teaches the agent to beat the crowd
- **Auxiliary 2:** Team value appreciation (×0.05 weight) — incentivizes smart early transfers
- **Potential-based shaping:** Φ(s) = estimated remaining-season value of current squad, using the xP model — accelerates learning while preserving optimal policy invariance per Ng et al. (1999)

Normalize rewards with `VecNormalize` to stabilize PPO training. Set γ = 0.99 for season-long optimization.

**Core training code:**

```python
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

class FPLEnv(gymnasium.Env):
    def __init__(self, season_data, xp_model):
        self.season_data = season_data          # vaastav CSV loaded
        self.xp_model = xp_model                # trained XGBoost
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(STATE_DIM,), dtype=np.float32
        )
        self.action_space = spaces.MultiDiscrete([3, 15, 80, 15, 5])

    def action_masks(self):
        # Enforce: budget, position validity, chip availability, club limits
        return np.concatenate([...])  # boolean mask per action dimension

    def step(self, action):
        transfers, captain, chip = self.decode(action)
        self.apply_transfers(transfers)       # update squad, deduct hits
        self.apply_chip(chip)                 # activate if valid
        points = self.calculate_gw_points()   # from historical data
        self.current_gw += 1
        return self.get_obs(), self.compute_reward(points), self.current_gw > 38, False, {}

    def reset(self, seed=None):
        self.current_gw = 1
        self.squad = self.solve_initial_squad()  # MILP for GW1 squad
        self.budget = 1000  # £100.0m in tenths
        self.free_transfers = 1
        self.chips = {"wildcard_h1": True, "wildcard_h2": True,
                      "freehit_h1": True, "freehit_h2": True,
                      "benchboost_h1": True, "benchboost_h2": True,
                      "triplecaptain_h1": True, "triplecaptain_h2": True}
        return self.get_obs(), {}

# Parallel envs across seasons
envs = SubprocVecEnv([make_env(s) for s in TRAIN_SEASONS])
envs = VecNormalize(envs, norm_obs=True, norm_reward=True)

model = MaskablePPO(
    "MlpPolicy", envs,
    learning_rate=3e-4, n_steps=2048, batch_size=256,
    n_epochs=10, gamma=0.99, gae_lambda=0.95,
    ent_coef=0.01, verbose=1, tensorboard_log="./fpl_tb/"
)
model.learn(total_timesteps=5_000_000)
```

**Deployment integration** pulls live data from the FPL API, runs the xP model, solves MILP for the optimal squad, then asks the RL agent for strategic decisions:

```python
class FPLBot:
    def __init__(self):
        self.xp_model = xgboost.Booster(model_file="xp_model.json")
        self.rl_agent = MaskablePPO.load("fpl_strategic_agent")
        self.optimizer = FPLOptimizer()  # PuLP wrapper

    async def execute_gameweek(self):
        async with aiohttp.ClientSession() as session:
            fpl = FPL(session)
            await fpl.login(EMAIL, PASSWORD)
            data = await fpl.get_players()            # bootstrap-static
            xp = self.xp_model.predict(features)      # predict all players
            squad = self.optimizer.solve(xp, budget, constraints)
            strategy = self.rl_agent.predict(state, action_masks=masks)
            await user.transfer(out_ids, in_ids)       # execute via API
```

---

## What the research says works (and doesn't)

The academic literature on RL for FPL is strikingly thin. Multiple papers explicitly state the field is "extremely scarce." Here are the key works:

**Matthews, Ramchurn & Chalkiadakis (AAAI 2012)** remains the foundational and most successful RL paper. They modeled FPL as a belief-state MDP with Bayesian Q-learning, used Monte Carlo match simulations with generative priors from previous seasons, and solved weekly team selection as a multi-dimensional knapsack problem. Result: **top ~1% of 2.5M players** in the 2010/11 season, with a best-case score of 2,222 points. Discount factor γ ≈ 0.5 proved optimal, suggesting moderate myopia is better than full long-horizon planning.

**"Optimizing Fantasy Sports Team Selection with Deep RL" (ACM CODS-COMAD 2024, arXiv 2412.19215)** is the most recent deep RL work, applying DQN and PPO to fantasy cricket. PPO consistently outperformed DQN, placing teams above the 60th percentile. The architecture used a shared network with actor/critic heads and 3 fully-connected layers of 1024 units. While cricket-specific, the methodology transfers directly to FPL.

**Bonello et al. (2019)** achieved **top 0.5%** (rank ~30,000 of 6.5M) without RL, instead combining multi-stream data (statistics, Twitter sentiment, betting odds, expert opinions, fixture difficulty) with gradient boosting prediction and optimization. This demonstrates that prediction quality matters more than the decision-making algorithm.

**Groos (2025, arXiv 2508.09992)** released **OpenFPL**, an open-source position-specific ensemble ML forecasting system trained on FPL + Understat data. It matched the accuracy of FPL Review (the leading commercial forecasting service) and surpassed it for high-return players above 2 predicted points. The code is available at `github.com/daniegr/OpenFPL`.

**Beal, Norman & Ramchurn (2020)** extended the Matthews framework to NFL Daily Fantasy Sports, achieving profitability in **81.3% of gameweeks** over four seasons using LSTM prediction + mixed-integer programming.

The consistent finding across all research: **the two-stage predict-then-optimize approach dominates pure RL**. Prediction quality is the binding constraint — a perfect optimizer with mediocre predictions loses to a simple greedy algorithm with excellent predictions.

---

## Concrete development roadmap

**Weeks 1–2:** Build the xP prediction model. Load vaastav data (2016–present), engineer features (rolling form, xG/xA from Understat, fixture difficulty, ICT index, ownership trends, set-piece flags). Train XGBoost with 5-fold cross-validation using season-based splits. Target: mean absolute error under 2.0 points per player per gameweek.

**Weeks 3–4:** Implement MILP team selection with PuLP. Model all constraints: budget, positions (2-5-5-3), max 3 per club, formation validity, selling price rules. Reference `sertalpbilal/FPL-Optimization-Tools` for constraint formulations. Validate on historical seasons — solve the "hindsight optimal" squad each week using actual points to establish an upper bound.

**Weeks 5–7:** Build the custom Gymnasium environment. Implement the full game engine: transfer logic (free transfers banking up to 5, -4 point hits), all 8 chips with half-season restrictions, auto-substitution with formation validation, captain/vice-captain failover, and price change simulation (approximate with historical price data from vaastav). Use historical points as ground truth rewards.

**Weeks 8–10:** Train MaskablePPO. Start with the strategic-only action space (transfer count, chip activation). Train across 7 season environments in parallel with `SubprocVecEnv`. Iterate on reward shaping — add team value auxiliary reward if the agent neglects price rises. Monitor with TensorBoard/Weights & Biases.

**Weeks 11–12:** Deploy. Build the API integration layer using `amosbastian/fpl` for authenticated operations. Set up a scheduled job (AWS Lambda or cron) to run before each gameweek deadline. Implement a recommendation mode that suggests actions for human approval before a fully autonomous mode.

The key insight for beating your friends: **chip timing and transfer planning are where RL adds the most value over conventional optimization.** Most FPL managers waste chips or take unnecessary hits. An RL agent trained on 7+ seasons of strategic patterns — when double gameweeks cluster, when fixture swings favor wildcards, when bench boost maximizes bench player minutes — will exploit these timing decisions far better than human intuition or static optimization.