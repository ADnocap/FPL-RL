"""Lineup selector: given a fixed 15-player squad, pick optimal XI + captain."""

from __future__ import annotations

from fpl_rl.optimizer.types import OptimizerResult, PlayerCandidate
from fpl_rl.utils.constants import Position


def select_lineup(
    squad: list[PlayerCandidate],
) -> OptimizerResult:
    """Select optimal starting XI, captain, vice-captain, and bench order.

    Solves a binary MILP:
        max  sum(xP_i * y_i) + sum(xP_i * c_i)
    where y_i=1 means player i starts, c_i=1 means captain.

    Parameters
    ----------
    squad : list[PlayerCandidate]
        Exactly 15 players (2 GK, 5 DEF, 5 MID, 3 FWD).

    Returns
    -------
    OptimizerResult
    """
    import pulp

    n = len(squad)
    if n != 15:
        raise ValueError(f"Squad must have exactly 15 players, got {n}")

    # Decision variables
    y = [pulp.LpVariable(f"y_{i}", cat="Binary") for i in range(n)]
    c = [pulp.LpVariable(f"c_{i}", cat="Binary") for i in range(n)]
    v = [pulp.LpVariable(f"v_{i}", cat="Binary") for i in range(n)]

    prob = pulp.LpProblem("lineup_selection", pulp.LpMaximize)

    # Objective: maximise expected points + captain bonus
    xp = [p.predicted_points for p in squad]
    prob += pulp.lpSum(xp[i] * y[i] for i in range(n)) + pulp.lpSum(
        xp[i] * c[i] for i in range(n)
    )

    # --- constraints ---
    # Exactly 11 starters
    prob += pulp.lpSum(y) == 11

    # Position constraints (on starters)
    gk_idx = [i for i, p in enumerate(squad) if p.position == Position.GK]
    def_idx = [i for i, p in enumerate(squad) if p.position == Position.DEF]
    mid_idx = [i for i, p in enumerate(squad) if p.position == Position.MID]
    fwd_idx = [i for i, p in enumerate(squad) if p.position == Position.FWD]

    prob += pulp.lpSum(y[i] for i in gk_idx) == 1
    prob += pulp.lpSum(y[i] for i in def_idx) >= 3
    prob += pulp.lpSum(y[i] for i in def_idx) <= 5
    prob += pulp.lpSum(y[i] for i in mid_idx) >= 2
    prob += pulp.lpSum(y[i] for i in mid_idx) <= 5
    prob += pulp.lpSum(y[i] for i in fwd_idx) >= 1
    prob += pulp.lpSum(y[i] for i in fwd_idx) <= 3

    # Captain / vice-captain must be in starting XI and distinct
    for i in range(n):
        prob += c[i] <= y[i]
        prob += v[i] <= y[i]
        prob += c[i] + v[i] <= 1
    prob += pulp.lpSum(c) == 1
    prob += pulp.lpSum(v) == 1

    # Solve (use default available solver)
    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    if prob.status != pulp.constants.LpStatusOptimal:
        raise RuntimeError(f"Lineup LP not optimal, status={prob.status}")

    # Extract solution
    lineup_ids: list[int] = []
    bench_candidates: list[tuple[float, int, int]] = []  # (xp, position, eid)
    captain_id = squad[0].element_id
    vice_captain_id = squad[1].element_id

    for i in range(n):
        if pulp.value(y[i]) > 0.5:
            lineup_ids.append(squad[i].element_id)
        else:
            bench_candidates.append(
                (squad[i].predicted_points, squad[i].position, squad[i].element_id)
            )
        if pulp.value(c[i]) > 0.5:
            captain_id = squad[i].element_id
        if pulp.value(v[i]) > 0.5:
            vice_captain_id = squad[i].element_id

    # Bench order: backup GK first, then by predicted points descending
    bench_gk = [eid for xp, pos, eid in bench_candidates if pos == Position.GK]
    bench_outfield = sorted(
        [(xp, eid) for xp, pos, eid in bench_candidates if pos != Position.GK],
        key=lambda t: t[0],
        reverse=True,
    )
    bench_ids = bench_gk + [eid for _, eid in bench_outfield]

    all_ids = lineup_ids + bench_ids
    return OptimizerResult(
        squad_element_ids=all_ids,
        lineup_element_ids=lineup_ids,
        bench_element_ids=bench_ids,
        captain_id=captain_id,
        vice_captain_id=vice_captain_id,
        objective_value=pulp.value(prob.objective),
    )
