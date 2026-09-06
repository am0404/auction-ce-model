"""A cheap stand-in for roster strength, used only to choose finalists.

Championship equity is expensive: a full season simulation per candidate
roster, thousands of seasons deep. A bounded search has to look at far more
rosters than that allows, so it needs something cheaper to rank them by.

This is that something, and it is **not value**. It is the expected weekly
starting-lineup projection under the roster's own availability process:

* availability is drawn from the engine's real bye and injury process, so a
  roster that covers a bye week scores better than one that does not;
* the lineup is chosen by the real slot-eligibility optimiser, so a fourth
  running back who can never start is correctly worth almost nothing and a
  tight end who competes for a WR/TE slot is correctly worth something;
* the contingency uplift is included, so a backup who inherits work when his
  starter is out is credited for exactly that and nothing more;
* scores are the static projection, not a realized draw, so this measures
  construction rather than luck.

Nothing is added by hand. There is no positional premium, no stacking bonus,
no floor or ceiling adjustment and no handcuff bonus, because none of those is
a quantity this model has. What is here is what the modelled fields already
imply.

What it therefore misses, and why the second stage exists: it uses a static
projection rather than the weekly one, so it cannot see a player whose value is
that his *projection* moves; it ignores scoring variance entirely, so it cannot
see that a volatile roster wins more championships than its mean suggests; and
it says nothing about opponents. Ranking by this alone would be ranking by
expected points, which is exactly the mistake the whole project exists to
avoid. Its only job is to pick a finalist set small enough to simulate.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from ..league import DEFAULT_LEAGUE, LeagueSettings, Position
from ..lineup_vec import select_lineups_mask
from ..players import PlayerSpec
from ..worlds import PoolArrays, build_pool_arrays, _contingency_bonus, _draw_availability

__all__ = ["ProxyEvaluator", "proxy_strength"]


class ProxyEvaluator:
    """Availability-only replicates, reused across many candidate rosters.

    The expensive part -- drawing byes and injuries for the whole pool -- is
    done once and shared, because every candidate roster is a subset of the
    same pool and the RNG is coordinate-addressed by player. Two rosters
    sharing a player see that player's *identical* availability, which makes
    proxy comparisons paired for free.
    """

    def __init__(self, pool: Sequence[PlayerSpec],
                 settings: LeagueSettings = DEFAULT_LEAGUE,
                 n_reps: int = 96, seed: int = 20260904):
        if n_reps < 1:
            raise ValueError("n_reps must be positive")
        self.settings = settings
        self.n_reps = int(n_reps)
        self.seed = int(seed)
        self.specs = tuple(pool)
        self.index = {s.player_id: i for i, s in enumerate(self.specs)}
        self._arrays: PoolArrays = build_pool_arrays(self.specs, settings)

        # Same coordinate shapes generate_world uses, so a player's byes and
        # injuries here are the ones he gets in the real simulation at this
        # seed. The proxy is therefore looking at the same seasons the CE
        # evaluation will, rather than at an independent world.
        sims = np.arange(self.n_reps, dtype=np.int64).reshape(self.n_reps, 1, 1)
        keys = self._arrays.stream_key.reshape(1, len(self.specs), 1)
        weeks = settings.total_weeks
        avail = _draw_availability(self._arrays, self.seed, sims, keys, weeks)
        self._available = avail.available                      # (R, P, W)
        bonus = _contingency_bonus(self._arrays, self._available)
        # The projection a manager would see: the player's level plus whatever
        # depth-chart uplift is already visible from who is out this week.
        self._projection = (self._arrays.base_mean.reshape(1, -1, 1) + bonus)
        self._weeks = weeks
        self._score_weeks = settings.regular_season_weeks

    @property
    def n_players(self) -> int:
        return len(self.specs)

    def strength(self, player_ids: Sequence[int]) -> float:
        """Mean weekly starting-lineup projection for this roster.

        Regular-season weeks only: the playoff weeks are a different question
        and averaging them in would quietly reward depth that only matters in
        weeks most teams never play.
        """
        idx = np.array([self.index[p] for p in player_ids], dtype=np.int64)
        proj = np.moveaxis(self._projection[:, idx, :], 1, -1)   # (R, W, n)
        avail = np.moveaxis(self._available[:, idx, :], 1, -1)
        pos = np.broadcast_to(
            self._arrays.position[idx][None, None, :], proj.shape)
        mask = select_lineups_mask(proj, avail, pos)
        started = (mask * proj).sum(axis=-1)                      # (R, W)
        return float(started[:, : self._score_weeks].mean())

    def strength_many(self, rosters: Sequence[Sequence[int]]) -> np.ndarray:
        """Strength for several rosters of the same size, in one pass."""
        if not rosters:
            return np.zeros(0, dtype=np.float64)
        sizes = {len(r) for r in rosters}
        if len(sizes) != 1:
            return np.array([self.strength(r) for r in rosters], dtype=np.float64)
        idx = np.array([[self.index[p] for p in r] for r in rosters],
                       dtype=np.int64)                            # (C, n)
        proj = np.moveaxis(self._projection[:, idx, :], 2, -1)    # (R, C, W, n)
        avail = np.moveaxis(self._available[:, idx, :], 2, -1)
        pos = np.broadcast_to(
            self._arrays.position[idx][None, :, None, :], proj.shape)
        mask = select_lineups_mask(proj, avail, pos)
        started = (mask * proj).sum(axis=-1)                      # (R, C, W)
        return started[:, :, : self._score_weeks].mean(axis=(0, 2))

    def solo_value(self, player_id: int) -> float:
        """A player's static projection. Used only to order a search, never as value."""
        return float(self._arrays.base_mean[self.index[player_id]])


def proxy_strength(specs: Sequence[PlayerSpec], player_ids: Sequence[int],
                   settings: LeagueSettings = DEFAULT_LEAGUE,
                   n_reps: int = 96, seed: int = 20260904) -> float:
    """One-shot convenience wrapper. Builds an evaluator and throws it away."""
    return ProxyEvaluator(specs, settings, n_reps, seed).strength(player_ids)
