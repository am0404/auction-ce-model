"""The fixed six-team, three-week playoff bracket.

::

    Week 15   seeds 1 and 2 bye
              QF-A: 4 vs 5
              QF-B: 3 vs 6
    Week 16   SF-1: 1 vs winner(QF-A)
              SF-2: 2 vs winner(QF-B)
    Week 17   final: winner(SF-1) vs winner(SF-2)

No reseeding.  A tie is won by the higher seed.  **There is no randomness in
this module at all** -- given the weekly score array the entire postseason is
deterministic, exactly as the specification requires.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .league import DEFAULT_LEAGUE, LeagueSettings
from .standings import RegularSeason

__all__ = ["PlayoffResult", "run_bracket"]


@dataclass(frozen=True)
class PlayoffResult:
    """Bracket outcomes for a batch of seasons."""

    champion: np.ndarray        # (S,) team index
    made_playoffs: np.ndarray   # (S, T) bool
    has_bye: np.ndarray         # (S, T) bool
    made_final: np.ndarray      # (S, T) bool
    qualifiers: np.ndarray      # (S, 6) team index by seed


def _advance(
    team_a: np.ndarray,
    seed_a: np.ndarray,
    team_b: np.ndarray,
    seed_b: np.ndarray,
    scores: np.ndarray,
    week: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Winner of one matchup; ties go to the higher (numerically lower) seed."""
    rows = np.arange(team_a.shape[0])
    sa = scores[rows, team_a, week]
    sb = scores[rows, team_b, week]
    a_wins = (sa > sb) | ((sa == sb) & (seed_a < seed_b))
    return np.where(a_wins, team_a, team_b), np.where(a_wins, seed_a, seed_b)


def run_bracket(
    scores: np.ndarray,
    standings: RegularSeason,
    settings: LeagueSettings = DEFAULT_LEAGUE,
) -> PlayoffResult:
    """Run weeks 15-17 from ``scores`` ``(S, T, total_weeks)``."""
    s, n_teams = scores.shape[0], scores.shape[1]
    q = standings.seed_to_team[:, : settings.n_playoff_teams]  # (S, 6)
    w15, w16, w17 = (wk - 1 for wk in settings.playoff_weeks)

    seed1, seed2 = q[:, 0], q[:, 1]
    seed3, seed4, seed5, seed6 = q[:, 2], q[:, 3], q[:, 4], q[:, 5]
    z = np.zeros(s, dtype=np.int64)

    qa_team, qa_seed = _advance(seed4, z + 3, seed5, z + 4, scores, w15)
    qb_team, qb_seed = _advance(seed3, z + 2, seed6, z + 5, scores, w15)
    sf1_team, sf1_seed = _advance(seed1, z + 0, qa_team, qa_seed, scores, w16)
    sf2_team, sf2_seed = _advance(seed2, z + 1, qb_team, qb_seed, scores, w16)
    champion, _ = _advance(sf1_team, sf1_seed, sf2_team, sf2_seed, scores, w17)

    rows = np.arange(s)[:, None]
    made_playoffs = np.zeros((s, n_teams), dtype=bool)
    made_playoffs[rows, q] = True
    has_bye = np.zeros((s, n_teams), dtype=bool)
    has_bye[rows, q[:, : settings.n_byes]] = True
    made_final = np.zeros((s, n_teams), dtype=bool)
    made_final[np.arange(s), sf1_team] = True
    made_final[np.arange(s), sf2_team] = True

    return PlayoffResult(
        champion=champion,
        made_playoffs=made_playoffs,
        has_bye=has_bye,
        made_final=made_final,
        qualifiers=q,
    )
