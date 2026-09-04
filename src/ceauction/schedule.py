"""Regular-season schedule.

12 teams give an 11-week single round robin (circle method), which satisfies
"every team plays every other team at least once".  Weeks 12-14 repeat rounds
1-3.

The *mapping from teams to schedule slots* is permuted once per simulated
season from its own RNG stream, so no team is structurally advantaged by the
three repeated opponents.  Under common random numbers the same permutation is
drawn for both scenarios of a paired comparison, so schedule luck cancels.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Tuple

import numpy as np

from . import rng
from .league import DEFAULT_LEAGUE, LeagueSettings

__all__ = ["round_robin", "base_schedule", "opponents_for_batch"]


@lru_cache(maxsize=8)
def round_robin(n_teams: int) -> np.ndarray:
    """``(n_teams - 1, n_teams)`` array; entry ``[r, i]`` is i's opponent in round r."""
    if n_teams % 2:
        raise ValueError("round_robin needs an even number of teams")
    rounds = n_teams - 1
    out = np.zeros((rounds, n_teams), dtype=np.int64)
    order = list(range(n_teams))
    for r in range(rounds):
        for i in range(n_teams // 2):
            a, b = order[i], order[n_teams - 1 - i]
            out[r, a] = b
            out[r, b] = a
        order = [order[0], order[-1]] + order[1:-1]
    return out


@lru_cache(maxsize=8)
def base_schedule(n_teams: int, n_weeks: int) -> np.ndarray:
    """``(n_weeks, n_teams)`` opponent-slot table, rounds recycled as needed."""
    rr = round_robin(n_teams)
    idx = np.arange(n_weeks) % rr.shape[0]
    return rr[idx]


def opponents_for_batch(
    seed: int,
    sim_start: int,
    n_sims: int,
    settings: LeagueSettings = DEFAULT_LEAGUE,
) -> np.ndarray:
    """``(n_sims, n_teams, regular_season_weeks)`` opponent **team** indices."""
    n = settings.n_teams
    weeks = settings.regular_season_weeks
    base = base_schedule(n, weeks)                      # (W, n) slot -> opp slot
    perm = rng.permutation_batch(n, n_sims, seed, rng.Kind.SCHEDULE_PERM, sim_start)
    inv = np.argsort(perm, axis=1)                      # team -> slot
    opp_slot = base.T[inv]                              # (S, n, W) opponent slots
    return perm[np.arange(n_sims)[:, None, None], opp_slot]
