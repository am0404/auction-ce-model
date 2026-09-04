"""Weekly results, records, total-points tiebreaker and seeding.

Each regular-season week produces **two** results per team: one against the
scheduled opponent and one against the league median.  Win = 1.0, tie = 0.5,
loss = 0.0, so a team's weekly haul is 2-0, 1-1 (or a tie variant) or 0-2.

The league median of 12 scores is the mean of the 6th and 7th ranked scores.
A score exactly equal to it is a tie.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .league import DEFAULT_LEAGUE, LeagueSettings

__all__ = ["RegularSeason", "weekly_median", "compare", "regular_season"]


@dataclass(frozen=True)
class RegularSeason:
    """Per-season, per-team regular-season outcomes."""

    wins: np.ndarray          # (S, T) total, out of 2 * weeks
    points: np.ndarray        # (S, T) total points scored
    h2h_wins: np.ndarray      # (S, T) head-to-head result total
    median_wins: np.ndarray   # (S, T) vs-median result total
    seed_to_team: np.ndarray  # (S, T) seed_to_team[:, 0] is the 1 seed
    team_to_seed: np.ndarray  # (S, T) 0-indexed seed of each team
    median: np.ndarray        # (S, W) the weekly league median


def weekly_median(scores: np.ndarray) -> np.ndarray:
    """``(S, W)`` league median from ``(S, T, W)`` scores.

    For an even team count this is the mean of the two central order
    statistics, which is what Sleeper uses.
    """
    n = scores.shape[1]
    part = np.sort(scores, axis=1)
    k = n // 2
    if n % 2:
        return part[:, k, :]
    return 0.5 * (part[:, k - 1, :] + part[:, k, :])


def compare(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """1.0 where ``a > b``, 0.0 where ``a < b``, 0.5 on an exact tie."""
    return np.where(a > b, 1.0, np.where(a < b, 0.0, 0.5))


def regular_season(
    scores: np.ndarray,
    opponents: np.ndarray,
    settings: LeagueSettings = DEFAULT_LEAGUE,
) -> RegularSeason:
    """Compute records and seeding.

    Parameters
    ----------
    scores:
        ``(S, T, total_weeks)`` realized team scores.  Only the first
        ``regular_season_weeks`` are used here.
    opponents:
        ``(S, T, regular_season_weeks)`` opponent team indices.
    """
    w = settings.regular_season_weeks
    reg = scores[:, :, :w]
    opp_scores = np.take_along_axis(reg, opponents, axis=1)

    h2h = compare(reg, opp_scores)
    med = weekly_median(reg)                      # (S, W)
    med_res = compare(reg, med[:, None, :])

    h2h_wins = h2h.sum(axis=2)
    median_wins = med_res.sum(axis=2)
    wins = h2h_wins + median_wins
    points = reg.sum(axis=2)

    # Sort key priority (lexsort reads keys last-first): wins desc, then total
    # points desc, then team index asc as a deterministic final tiebreak.
    n_teams = scores.shape[1]
    team_idx = np.broadcast_to(np.arange(n_teams), wins.shape)
    seed_to_team = np.lexsort((team_idx, -points, -wins), axis=1)
    team_to_seed = np.argsort(seed_to_team, axis=1)

    return RegularSeason(
        wins=wins,
        points=points,
        h2h_wins=h2h_wins,
        median_wins=median_wins,
        seed_to_team=seed_to_team,
        team_to_seed=team_to_seed,
        median=med,
    )
