"""Season simulation: rosters in, per-season outcomes out.

Runs the full pipeline of ``SPEC.md`` section 2 in batches of seasons::

    latent state -> availability -> realized scores -> pregame information
        -> lineup decision -> team score -> standings -> playoffs -> champion

Batching exists purely for memory.  Because the RNG is counter-based, results
are identical for any batch size, which ``tests/test_reproducibility.py``
asserts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .league import DEFAULT_LEAGUE, LeagueSettings
from .lineup_vec import select_lineups_mask
from .playoffs import run_bracket
from .roster import RosterSet
from .schedule import opponents_for_batch
from .standings import regular_season
from .worlds import PoolArrays, WorldBatch, build_pool_arrays, generate_world

__all__ = ["SeasonOutcomes", "team_scores", "simulate_seasons", "DEFAULT_CHUNK"]

DEFAULT_CHUNK = 256


@dataclass
class SeasonOutcomes:
    """Per-season, per-team results for a whole CE run.

    Stored per season (not just aggregated) so that paired comparisons can
    difference matched seasons rather than differencing two noisy means.
    """

    n_sims: int
    team_names: Sequence[str]
    champion: np.ndarray       # (S,) int16 team index
    made_playoffs: np.ndarray  # (S, T) bool
    has_bye: np.ndarray        # (S, T) bool
    made_final: np.ndarray     # (S, T) bool
    wins: np.ndarray           # (S, T) float32
    points: np.ndarray         # (S, T) float32
    h2h_wins: np.ndarray       # (S, T) float32
    median_wins: np.ndarray    # (S, T) float32
    seed: np.ndarray           # (S, T) int8, 0-indexed
    starters_filled: np.ndarray  # (S, T) float32, mean slots filled per week

    def championship_equity(self) -> np.ndarray:
        """``(T,)`` probability each team wins the league."""
        n_teams = self.made_playoffs.shape[1]
        counts = np.bincount(self.champion, minlength=n_teams)
        return counts / float(self.n_sims)

    def champion_indicator(self, team: int) -> np.ndarray:
        """``(S,)`` float 0/1 -- the per-season quantity a paired test needs."""
        return (self.champion == team).astype(np.float64)


def team_scores(world: WorldBatch, roster_matrix: np.ndarray):
    """``(scores, slots_filled)``, both ``(S, T, W)``, from legally chosen lineups.

    This is the one place where the information barrier could be violated, so
    it is written to make a violation obvious: ``select_lineups_mask`` is
    handed ``projection`` and ``available`` only, and ``realized`` is touched
    only *after* the mask exists.
    """
    n_teams, roster_size = roster_matrix.shape
    proj = world.pregame.projection[:, roster_matrix, :]       # (S, T, R, W)
    avail = world.availability.available[:, roster_matrix, :]  # (S, T, R, W)
    positions = world.pool.position[roster_matrix]             # (T, R)

    # Move the roster axis last so the optimiser sees (..., roster_size).
    proj_t = np.moveaxis(proj, 2, -1)                          # (S, T, W, R)
    avail_t = np.moveaxis(avail, 2, -1)
    pos_t = np.broadcast_to(positions[None, :, None, :], proj_t.shape)

    mask = select_lineups_mask(proj_t, avail_t, pos_t)         # (S, T, W, R)

    realized = np.moveaxis(world.realized.points[:, roster_matrix, :], 2, -1)
    scores = np.einsum("stwr,stwr->stw", mask, realized)
    filled = mask.sum(axis=-1).astype(np.float32)
    return scores, filled


def simulate_seasons(
    rosters: RosterSet,
    n_sims: int,
    seed: int,
    chunk: int = DEFAULT_CHUNK,
    settings: Optional[LeagueSettings] = None,
    pool: Optional[PoolArrays] = None,
) -> SeasonOutcomes:
    """Simulate ``n_sims`` complete seasons and return per-season outcomes."""
    settings = settings or rosters.settings
    pool = pool if pool is not None else build_pool_arrays(rosters.pool, settings)
    roster_matrix = rosters.roster_matrix()
    n_teams = settings.n_teams

    champion = np.empty(n_sims, dtype=np.int16)
    made_playoffs = np.empty((n_sims, n_teams), dtype=bool)
    has_bye = np.empty((n_sims, n_teams), dtype=bool)
    made_final = np.empty((n_sims, n_teams), dtype=bool)
    wins = np.empty((n_sims, n_teams), dtype=np.float32)
    points = np.empty((n_sims, n_teams), dtype=np.float32)
    h2h_wins = np.empty((n_sims, n_teams), dtype=np.float32)
    median_wins = np.empty((n_sims, n_teams), dtype=np.float32)
    seed_arr = np.empty((n_sims, n_teams), dtype=np.int8)
    filled_arr = np.empty((n_sims, n_teams), dtype=np.float32)

    for start in range(0, n_sims, chunk):
        size = min(chunk, n_sims - start)
        world = generate_world(pool, seed, start, size)
        scores, filled = team_scores(world, roster_matrix)
        opponents = opponents_for_batch(seed, start, size, settings)
        rs = regular_season(scores, opponents, settings)
        po = run_bracket(scores, rs, settings)

        sl = slice(start, start + size)
        champion[sl] = po.champion
        made_playoffs[sl] = po.made_playoffs
        has_bye[sl] = po.has_bye
        made_final[sl] = po.made_final
        wins[sl] = rs.wins
        points[sl] = rs.points
        h2h_wins[sl] = rs.h2h_wins
        median_wins[sl] = rs.median_wins
        seed_arr[sl] = rs.team_to_seed
        filled_arr[sl] = filled[:, :, : settings.regular_season_weeks].mean(axis=2)

    return SeasonOutcomes(
        n_sims=n_sims,
        team_names=rosters.team_names,
        champion=champion,
        made_playoffs=made_playoffs,
        has_bye=has_bye,
        made_final=made_final,
        wins=wins,
        points=points,
        h2h_wins=h2h_wins,
        median_wins=median_wins,
        seed=seed_arr,
        starters_filled=filled_arr,
    )
