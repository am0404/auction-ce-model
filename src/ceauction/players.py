"""Player specifications and player-state interfaces.

``PlayerSpec`` is the seam between "where the numbers came from" and "what the
CE engine does with them".  Tonight every spec is produced by
:mod:`ceauction.synthetic` and carries ``data_source="SYNTHETIC"``.  When real
data arrives, the same fields are populated from real medians, real injury
probabilities, real boom/bust widths, real weekly projections and real
correlation relationships — **the engine does not change**.

The five concepts the specification requires be kept apart live in five
different places:

===  ===================================  ====================================
 #   Concept                              Home
===  ===================================  ====================================
 1   Persistent latent season state       ``LatentState`` (worlds.py)
 2   Health and availability              ``Availability`` (worlds.py)
 3   Pregame-observable role/projection   ``Pregame`` (worlds.py) / pregame.py
 4   Realized weekly performance          ``Realized`` (worlds.py)
 5   Observable information channel       ``SignalBatch`` (worlds.py)
===  ===================================  ====================================

Concept 5 is a *separate observable process*, not a re-reading of concept 4.
Beliefs update from simulated observable signals -- usage, role, underlying
performance indicators -- and never from realized fantasy-point noise, so an
unforecastable scoring spike cannot become future projection.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional, Sequence, Tuple

import numpy as np

from .league import Position

__all__ = ["ShockLoading", "Contingency", "PlayerSpec", "positions_array", "with_overrides"]


@dataclass(frozen=True)
class ShockLoading:
    """Membership in a shared weekly shock group.

    One draw per ``(group_id, week)`` is added to every member's realized score
    scaled by ``beta``.  This is the single mechanism behind every correlation
    the model supports:

    * ``team:BUF`` with positive betas -> shared team environment;
    * ``stack:BUF-pass`` on a QB and his WR -> QB / pass-catcher correlation;
    * opposite-signed betas on one group -> negative correlation (e.g. a
      backfield timeshare);
    * any user-defined group -> arbitrary correlation structure.

    Correlation enters *realized* scores only.  It is invisible pregame, which
    is what makes correlated bench upside a genuinely different asset from
    independent bench upside.
    """

    group_id: str
    beta: float = 1.0


@dataclass(frozen=True)
class Contingency:
    """Next-man-up structure (a handcuff).

    In any week where ``on_player_id`` is unavailable, this player's true mean
    *and* his projection rise by ``bonus``.  Availability is pregame-observable,
    so the contingency is observable too: the engine correctly projects a backup
    up in the weeks his starter is out, and back down when the starter returns.
    """

    on_player_id: int
    bonus: float


@dataclass(frozen=True)
class PlayerSpec:
    """Everything the CE engine needs to know about one player.

    All rate parameters are per-week.  All point parameters are half-PPR
    fantasy points.
    """

    player_id: int
    name: str
    position: Position
    nfl_team: str

    # --- level and dispersion -------------------------------------------------
    base_mean: float
    """Consensus per-week mean when active in the player's base role.
    Real data: the player's projected weekly median/mean."""

    week_sd: float
    """Idiosyncratic weekly standard deviation.  Real data: boom/bust width."""

    season_sd: float = 0.0
    """SD of the persistent, unobserved season-level deviation from
    ``base_mean``.  Zero means consensus is exactly right about true talent."""

    # --- availability ---------------------------------------------------------
    bye_week: int = 0
    """1-indexed bye week; 0 means no bye."""

    weekly_injury_hazard: float = 0.0
    """Probability a new injury starts in a given healthy week."""

    injury_mean_weeks: float = 2.5
    """Mean length of an absence, in weeks (geometric-ish, minimum 1)."""

    # --- unforecastable upside -----------------------------------------------
    spike_rate: float = 0.0
    """Probability of a spike week.  Spikes are *not* in the projection."""

    spike_scale: float = 0.0
    """Mean size of a spike, in points, conditional on one happening."""

    spike_mean_removed: bool = True
    """When ``True`` (the default) the unconditional spike mean is subtracted,
    so ``spike_rate`` changes the *shape* of the weekly distribution without
    changing its mean -- the right setting for asking "does volatility help?".

    When ``False`` the spikes add ``spike_rate * spike_scale`` to the player's
    true mean **without** appearing in his projection.  That is the right
    setting for asking "is a point of forecastable production worth more than a
    point of unforecastable production?", which is the fourth CE laboratory
    experiment."""

    # --- observable role change ----------------------------------------------
    role_change_prob: float = 0.0
    role_change_mean: float = 0.0
    role_change_sd: float = 0.0
    role_reveal_lag: int = 1
    """Weeks between a role change taking effect and becoming observable."""

    # --- the observable-information channel -----------------------------------
    signal_noise_sd: Optional[float] = None
    """Noise on the weekly **observable signal** about this player's persistent
    season level.

    This is the *only* channel through which past weeks change future
    projections.  It is deliberately not the realized fantasy score: a manager
    updating on scoring noise would turn a random touchdown into a permanent
    projection increase, which is exactly the error this field exists to
    prevent.

    Read it as the precision of whatever a real manager actually watches --
    snap share, route participation, target share, carry share, depth-chart
    reporting, or the week-to-week movement of a published projection.  Small
    values mean usage tells you the truth quickly; large values mean it barely
    tells you anything.  ``None`` (the default) falls back to ``week_sd``, so
    one week of observed usage is about as informative as one observed score
    would have been -- a deliberately conservative placeholder that should be
    replaced by a real calibration (see ``OPEN_QUESTIONS.md`` A5).

    Note what this does *not* carry: the signal observes the persistent latent
    level only.  Role changes reach the projection through
    ``observed_role_delta``, exactly once, from their reveal week."""

    # --- forecastable weekly state --------------------------------------------
    weekly_state_sd: float = 0.0
    """SD of a per-week, per-player component that is **known before lineup
    lock**.

    This is the model's representation of knowable weekly conditions: matchup,
    expected volume, announced usage, weather, pace, a banged-up offensive line.
    It appears in the pregame projection *and* in the realized score's
    conditional mean, which is precisely what makes it different from
    ``proj_noise_sd`` (projection error, which moves the projection but not the
    score) and from ``week_sd`` (scoring noise, which moves the score but not
    the projection).

    Without it most players have an effectively static pregame level, so two
    candidates for one lineup spot can never rotate on knowable conditions and
    the model cannot represent building a spot in the aggregate.  Draws are
    independent across players; use ``weekly_state_pattern`` for correlated or
    deliberately offset weekly structure."""

    weekly_state_pattern: Optional[Tuple[float, ...]] = None
    """Deterministic forecastable weekly offsets, one per simulated week.

    Added to ``weekly_state_sd``'s stochastic draw.  Because the pattern is
    supplied per player, any cross-player structure is expressible: identical
    patterns give perfectly correlated good weeks, negated or rotated patterns
    give offset ones, and omitting it leaves the players independent."""

    hidden_weekly_pattern: Optional[Tuple[float, ...]] = None
    """Deterministic weekly offsets that are **not** forecastable.

    Identical in shape to ``weekly_state_pattern`` but applied to the realized
    score only, never to the projection.  Its purpose is control arms: giving
    one arm ``weekly_state_pattern`` and the other the same numbers as
    ``hidden_weekly_pattern`` produces two worlds with byte-identical realized
    production that differ *only* in whether the good weeks were identifiable
    before kickoff."""

    # --- structure ------------------------------------------------------------
    shock_loadings: Tuple[ShockLoading, ...] = ()
    contingency: Optional[Contingency] = None

    # --- projection quality ---------------------------------------------------
    proj_noise_sd: float = 0.0
    """How wrong the published projection is, week to week, beyond what the
    filter already knows.  Zero means the projection is the exact posterior
    mean."""

    weekly_projection_override: Optional[Tuple[float, ...]] = None
    """Real published weekly projections, one per simulated week.  When set,
    it replaces the entire modelled projection for this player."""

    # --- provenance -----------------------------------------------------------
    data_source: str = "SYNTHETIC"
    """``"SYNTHETIC"`` tonight.  Real specs should use ``"REAL:<vendor>"`` so
    that mixed pools are obvious in any report."""

    crn_key: Optional[int] = None
    """Common-random-number key.  Defaults to ``player_id``.  Two *alternative*
    players competing for the same roster slot in a paired comparison may share
    a ``crn_key`` so they receive identical uniform draws and differ only
    through their parameters — the strongest available variance reduction."""

    notes: str = ""

    def __post_init__(self) -> None:
        if self.week_sd < 0 or self.season_sd < 0:
            raise ValueError("standard deviations must be non-negative")
        if not (0.0 <= self.weekly_injury_hazard <= 1.0):
            raise ValueError("weekly_injury_hazard must be a probability")
        if not (0.0 <= self.spike_rate <= 1.0):
            raise ValueError("spike_rate must be a probability")
        if not (0.0 <= self.role_change_prob <= 1.0):
            raise ValueError("role_change_prob must be a probability")
        if self.injury_mean_weeks <= 0:
            raise ValueError("injury_mean_weeks must be positive")
        if self.role_reveal_lag < 0:
            raise ValueError("role_reveal_lag must be non-negative")
        if self.signal_noise_sd is not None and self.signal_noise_sd < 0:
            raise ValueError("signal_noise_sd must be non-negative")
        if self.weekly_state_sd < 0:
            raise ValueError("weekly_state_sd must be non-negative")

    @property
    def stream_key(self) -> int:
        """Coordinate used for this player's random streams."""
        return self.player_id if self.crn_key is None else self.crn_key

    @property
    def is_synthetic(self) -> bool:
        return self.data_source.upper().startswith("SYNTHETIC")


def positions_array(specs: Sequence[PlayerSpec]) -> np.ndarray:
    """``int8`` array of position codes, indexed like ``specs``."""
    return np.array([int(s.position) for s in specs], dtype=np.int8)


def with_overrides(spec: PlayerSpec, **changes) -> PlayerSpec:
    """Return a copy of ``spec`` with fields replaced.

    Used heavily by the CE laboratory to build A/B variants that differ in
    exactly one parameter.
    """
    return replace(spec, **changes)
