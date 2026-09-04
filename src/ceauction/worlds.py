"""Simulation worlds: latent state, availability, realized scores, pregame info.

This module is where the **information barrier** is built.  A world is
generated in five layers, kept in five separate objects so that it is
structurally obvious which of them a lineup decision may read:

======================  =========================  ==================
Layer                   Object                     Lineup may read?
======================  =========================  ==================
persistent latent       :class:`LatentState`       no
health / availability   :class:`AvailabilityBatch` **yes**
observable signals      :class:`SignalBatch`       **yes** (weeks < w)
realized performance    :class:`RealizedBatch`     no
pregame information     :class:`PregameBatch`      **yes**
======================  =========================  ==================

Two barriers, not one
---------------------

**No same-week clairvoyance.** ``_build_pregame`` forms week *w*'s belief from
a cumulative sum over weeks strictly less than *w*, so week *w*'s own outcome
is arithmetically absent from week *w*'s projection.

**No learning from unforecastable noise.**  ``_build_pregame`` does not receive
the realized array *at all*.  Beliefs update from :class:`SignalBatch` -- a
distinct observable process standing in for usage, snaps, routes, targets and
depth-chart reporting -- which observes the persistent latent level and nothing
else.  A random touchdown, a heavy-tailed spike week and a hidden team shock
therefore move that week's score and *only* that week's score.  They cannot
reach a future projection, because there is no argument through which they
could arrive.

The four things a week's score is made of
-----------------------------------------

They are distinct quantities and each is added exactly once:

============================  ============================  ==================
Component                     Realized score                Projection
============================  ============================  ==================
persistent player level       ``base_mean + season_shift``   ``base_mean`` plus
                                                             the posterior from
                                                             observed signals
observable role change        ``true_role_delta``            ``observed_role_delta``
                              (from the change week)         (from the reveal week)
forecastable weekly state     ``contingency_bonus``          the same value
unforecastable realized       group shock + idiosyncratic    absent
noise                         noise + spikes
============================  ============================  ==================

The pregame side never re-derives one component from another, which is what
keeps a role change from being counted once as "unexplained good play" before
it is revealed and again as an explicit delta afterwards.

All arrays are shaped ``(n_sims, n_players, n_weeks)`` unless noted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import rng
from .league import DEFAULT_LEAGUE, LeagueSettings
from .players import PlayerSpec

__all__ = [
    "PoolArrays",
    "LatentState",
    "AvailabilityBatch",
    "SignalBatch",
    "RealizedBatch",
    "PregameBatch",
    "WorldBatch",
    "build_pool_arrays",
    "generate_world",
    "stable_group_id",
]

_EPS = 1e-9


def stable_group_id(name: str) -> int:
    """Deterministic 63-bit id for a correlation-group name.

    Python's built-in ``hash`` is salted per process, which would break
    reproducibility across runs, so this uses an explicit FNV-1a.
    """
    h = 0xCBF29CE484222325
    for b in name.encode("utf-8"):
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h >> 1


@dataclass(frozen=True)
class PoolArrays:
    """Static, per-player arrays derived once from a list of ``PlayerSpec``.

    Building these is pure bookkeeping; it happens once per CE run, not once
    per simulated season.
    """

    n_players: int
    n_weeks: int
    position: np.ndarray          # (P,) int8
    stream_key: np.ndarray        # (P,) int64
    base_mean: np.ndarray         # (P,)
    week_sd: np.ndarray           # (P,)
    season_sd: np.ndarray         # (P,)
    bye_index: np.ndarray         # (P,) 0-indexed week, -1 for none
    injury_hazard: np.ndarray     # (P,)
    injury_mean_weeks: np.ndarray # (P,)
    spike_rate: np.ndarray        # (P,)
    spike_scale: np.ndarray       # (P,)
    spike_demean: np.ndarray      # (P,) 1.0 if the spike mean is subtracted
    role_prob: np.ndarray         # (P,)
    role_mean: np.ndarray         # (P,)
    role_sd: np.ndarray           # (P,)
    role_lag: np.ndarray          # (P,) int
    signal_noise_sd: np.ndarray   # (P,) observable-signal precision
    proj_noise_sd: np.ndarray     # (P,)
    contingency_on: np.ndarray    # (P,) pool index or -1
    contingency_bonus: np.ndarray # (P,)
    beta: np.ndarray              # (P, G) loading matrix
    group_key: np.ndarray         # (G,) int64 stream keys
    proj_override: Optional[np.ndarray]      # (P, W) or None
    proj_override_mask: Optional[np.ndarray] # (P,) bool or None
    group_names: Tuple[str, ...] = ()


def build_pool_arrays(
    specs: Sequence[PlayerSpec], settings: LeagueSettings = DEFAULT_LEAGUE
) -> PoolArrays:
    """Vectorise a player pool.  ``specs`` order defines pool indices."""
    p = len(specs)
    w = settings.total_weeks
    id_to_index = {s.player_id: i for i, s in enumerate(specs)}

    group_names: List[str] = []
    group_pos: Dict[str, int] = {}
    for s in specs:
        for load in s.shock_loadings:
            if load.group_id not in group_pos:
                group_pos[load.group_id] = len(group_names)
                group_names.append(load.group_id)
    g = max(len(group_names), 1)
    beta = np.zeros((p, g), dtype=np.float64)
    for i, s in enumerate(specs):
        for load in s.shock_loadings:
            beta[i, group_pos[load.group_id]] += load.beta

    override_rows = [s.weekly_projection_override for s in specs]
    has_override = any(o is not None for o in override_rows)
    proj_override = None
    proj_override_mask = None
    if has_override:
        proj_override = np.zeros((p, w), dtype=np.float64)
        proj_override_mask = np.zeros(p, dtype=bool)
        for i, o in enumerate(override_rows):
            if o is None:
                continue
            arr = np.asarray(o, dtype=np.float64)
            if arr.shape != (w,):
                raise ValueError(
                    f"weekly_projection_override for {specs[i].name} must have "
                    f"length {w}, got {arr.shape}"
                )
            proj_override[i] = arr
            proj_override_mask[i] = True

    cont_on = np.full(p, -1, dtype=np.int64)
    cont_bonus = np.zeros(p, dtype=np.float64)
    for i, s in enumerate(specs):
        if s.contingency is not None:
            if s.contingency.on_player_id not in id_to_index:
                raise KeyError(
                    f"{s.name} is contingent on player "
                    f"{s.contingency.on_player_id}, who is not in the pool"
                )
            cont_on[i] = id_to_index[s.contingency.on_player_id]
            cont_bonus[i] = s.contingency.bonus

    def col(attr, dtype=np.float64):
        return np.array([getattr(s, attr) for s in specs], dtype=dtype)

    bye = col("bye_week", np.int64) - 1
    bye[bye < 0] = -1

    return PoolArrays(
        n_players=p,
        n_weeks=w,
        position=np.array([int(s.position) for s in specs], dtype=np.int8),
        stream_key=np.array([s.stream_key for s in specs], dtype=np.int64),
        base_mean=col("base_mean"),
        week_sd=np.maximum(col("week_sd"), _EPS),
        season_sd=col("season_sd"),
        bye_index=bye,
        injury_hazard=col("weekly_injury_hazard"),
        injury_mean_weeks=col("injury_mean_weeks"),
        spike_rate=col("spike_rate"),
        spike_scale=col("spike_scale"),
        spike_demean=np.array(
            [1.0 if s.spike_mean_removed else 0.0 for s in specs], dtype=np.float64
        ),
        role_prob=col("role_change_prob"),
        role_mean=col("role_change_mean"),
        role_sd=col("role_change_sd"),
        role_lag=col("role_reveal_lag", np.int64),
        # `signal_noise_sd=None` means "usage tells you about as much per week
        # as one observed score would have"; see PlayerSpec.signal_noise_sd.
        signal_noise_sd=np.maximum(
            np.array(
                [s.week_sd if s.signal_noise_sd is None else s.signal_noise_sd
                 for s in specs],
                dtype=np.float64,
            ),
            _EPS,
        ),
        proj_noise_sd=col("proj_noise_sd"),
        contingency_on=cont_on,
        contingency_bonus=cont_bonus,
        beta=beta,
        group_key=np.array(
            [stable_group_id(n) for n in group_names] or [0], dtype=np.int64
        ),
        proj_override=proj_override,
        proj_override_mask=proj_override_mask,
        group_names=tuple(group_names),
    )


# --------------------------------------------------------------------------
# Layer 1 -- persistent latent season state (never observed directly)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class LatentState:
    """Persistent, unobservable season state."""

    season_shift: np.ndarray       # (S, P) true deviation from consensus mean
    role_happens: np.ndarray       # (S, P) bool
    role_week: np.ndarray          # (S, P) 0-indexed week the change takes effect
    role_size: np.ndarray          # (S, P) size of the change in points
    true_role_delta: np.ndarray    # (S, P, W) in effect from role_week onward
    observed_role_delta: np.ndarray  # (S, P, W) revealed from role_week + lag


# --------------------------------------------------------------------------
# Layer 2 -- health and availability (observable)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class AvailabilityBatch:
    """Who can play, and why not.  Fully observable before kickoff."""

    available: np.ndarray  # (S, P, W) bool
    on_bye: np.ndarray     # (S, P, W) bool
    injured: np.ndarray    # (S, P, W) bool


# --------------------------------------------------------------------------
# Layer 3 -- the observable-information channel
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SignalBatch:
    """Weekly observable signals about a player's persistent latent level.

    This is the seam a real information source plugs into.  ``level_signal``
    stands for whatever a manager actually watches week to week -- snap share,
    route participation, target or carry share, depth-chart reporting, the
    drift of a published projection -- expressed on the fantasy-points scale
    as a noisy reading of the player's persistent deviation from consensus:

    .. code-block:: none

        level_signal[p, w] = season_shift[p] + signal_noise_sd[p] * xi[p, w]

    Three properties are deliberate.

    * It observes **only** the persistent latent level.  Idiosyncratic weekly
      noise, spike weeks and shared team shocks are absent by construction, so
      no amount of unforecastable scoring can move a future projection.
    * It is drawn from its own RNG stream, independent of every realized-score
      stream.  Perturbing realized noise leaves the signals bit-identical,
      which is what the adversarial tests exploit.
    * A signal exists only for weeks the player actually played (``observed``),
      because usage is not observed for a player who did not appear.

    Calibrating this against real data replaces ``signal_noise_sd`` and nothing
    else: a small value is a league where usage tells you the truth in two
    weeks, a large value one where it never quite does.
    """

    level_signal: np.ndarray  # (S, P, W)
    observed: np.ndarray      # (S, P, W) bool


# --------------------------------------------------------------------------
# Layer 4 -- realized performance (never visible to a lineup decision)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RealizedBatch:
    """What actually happened.  Zero for weeks a player was unavailable."""

    points: np.ndarray      # (S, P, W)
    spike: np.ndarray       # (S, P, W) the unforecastable component
    group_effect: np.ndarray  # (S, P, W) the correlated component


# --------------------------------------------------------------------------
# Layer 5 -- pregame information (the only thing a lineup may read)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PregameBatch:
    """Everything observable before kickoff, and nothing else."""

    projection: np.ndarray           # (S, P, W)
    observed_role_delta: np.ndarray  # (S, P, W)
    contingency_bonus: np.ndarray    # (S, P, W)
    posterior_mean: np.ndarray       # (S, P, W) belief about season_shift,
                                     # formed from observable signals only
    n_observed: np.ndarray           # (S, P, W) signals used to form that belief


@dataclass(frozen=True)
class WorldBatch:
    """One contiguous block of simulated seasons."""

    sim_start: int
    n_sims: int
    pool: PoolArrays
    latent: LatentState
    availability: AvailabilityBatch
    signals: SignalBatch
    realized: RealizedBatch
    pregame: PregameBatch

    @property
    def n_weeks(self) -> int:
        return self.pool.n_weeks


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------


def _draw_availability(
    pool: PoolArrays, seed: int, sims: np.ndarray, keys: np.ndarray, n_weeks: int
) -> AvailabilityBatch:
    """Byes plus a hazard/duration injury process.

    Injuries are generated by a forward scan over weeks: in any week a healthy
    player may start an absence, whose length is drawn at onset.  The scan is
    vectorised across (season, player); only the 17 week-steps are sequential.
    """
    s, p = sims.shape[0], keys.shape[1]
    weeks = np.arange(n_weeks, dtype=np.int64).reshape(1, 1, n_weeks)

    on_bye = np.broadcast_to(
        pool.bye_index.reshape(1, p, 1) == weeks, (s, p, n_weeks)
    ).copy()

    injured = np.zeros((s, p, n_weeks), dtype=bool)
    if pool.injury_hazard.max() > 0.0:
        hazard = pool.injury_hazard.reshape(1, p)
        mean_len = pool.injury_mean_weeks.reshape(1, p)
        onset_u = rng.uniform(seed, rng.Kind.INJURY_ONSET, sims, keys, weeks)
        dur_e = rng.exponential(seed, rng.Kind.INJURY_DURATION, sims, keys, weeks)
        out_until = np.full((s, p), -1, dtype=np.int64)
        for w in range(n_weeks):
            healthy = out_until < w
            onset = healthy & (onset_u[:, :, w] < hazard)
            duration = np.maximum(1, np.rint(dur_e[:, :, w] * mean_len).astype(np.int64))
            out_until = np.where(onset, w + duration - 1, out_until)
            injured[:, :, w] = out_until >= w

    available = ~(on_bye | injured)
    return AvailabilityBatch(available=available, on_bye=on_bye, injured=injured)


def _draw_latent(
    pool: PoolArrays, seed: int, sims: np.ndarray, keys: np.ndarray, n_weeks: int
) -> LatentState:
    s, p = sims.shape[0], keys.shape[1]
    weeks = np.arange(n_weeks, dtype=np.int64).reshape(1, 1, n_weeks)

    if pool.season_sd.max() > 0.0:
        season_shift = (
            rng.normal(seed, rng.Kind.SEASON_SHIFT, sims, keys)
            * pool.season_sd.reshape(1, p)
        )
    else:
        season_shift = np.zeros((s, p), dtype=np.float64)

    if pool.role_prob.max() <= 0.0:
        zeros3 = np.zeros((s, p, n_weeks), dtype=np.float64)
        return LatentState(
            season_shift=season_shift,
            role_happens=np.zeros((s, p), dtype=bool),
            role_week=np.zeros((s, p), dtype=np.int64),
            role_size=np.zeros((s, p), dtype=np.float64),
            true_role_delta=zeros3,
            observed_role_delta=zeros3,
        )

    role_happens = rng.uniform(seed, rng.Kind.ROLE_HAPPENS, sims, keys) < pool.role_prob.reshape(1, p)
    # Role changes land in weeks 2..13 (0-indexed 1..12) so that there is both
    # a "before" and an "after" inside the regular season.
    span = max(1, min(12, n_weeks - 2))
    role_week = 1 + rng.randint(span, seed, rng.Kind.ROLE_WEEK, sims, keys)
    role_size = (
        pool.role_mean.reshape(1, p)
        + rng.normal(seed, rng.Kind.ROLE_SIZE, sims, keys) * pool.role_sd.reshape(1, p)
    )
    magnitude = np.where(role_happens, role_size, 0.0).reshape(s, p, 1)

    active = weeks >= role_week.reshape(s, p, 1)
    revealed = weeks >= (role_week.reshape(s, p, 1) + pool.role_lag.reshape(1, p, 1))
    return LatentState(
        season_shift=season_shift,
        role_happens=role_happens,
        role_week=role_week,
        role_size=role_size,
        true_role_delta=magnitude * active,
        observed_role_delta=magnitude * revealed,
    )


def _contingency_bonus(
    pool: PoolArrays, available: np.ndarray
) -> np.ndarray:
    """(S, P, W) uplift for players whose depth-chart superior is out.

    Availability is pregame-observable, so this is observable too.
    """
    s, p, w = available.shape
    bonus = np.zeros((s, p, w), dtype=np.float64)
    has = np.flatnonzero(pool.contingency_on >= 0)
    if has.size:
        src = pool.contingency_on[has]
        bonus[:, has, :] = (~available[:, src, :]) * pool.contingency_bonus[has].reshape(1, -1, 1)
    return bonus


def _draw_signals(
    pool: PoolArrays,
    seed: int,
    sims: np.ndarray,
    keys: np.ndarray,
    n_weeks: int,
    latent: LatentState,
    avail: AvailabilityBatch,
) -> SignalBatch:
    """Draw the observable-information process.  See :class:`SignalBatch`.

    Its own RNG stream (``Kind.SIGNAL_NOISE``) is the whole point: nothing here
    shares a draw with a realized score, so realized noise and these signals
    are independent by construction rather than by careful bookkeeping.
    """
    s, p = sims.shape[0], keys.shape[1]
    weeks = np.arange(n_weeks, dtype=np.int64).reshape(1, 1, n_weeks)

    if pool.season_sd.max() <= 0.0:
        # Nothing persistent to learn, so the signal carries no information and
        # the posterior is identically zero.  Skip the draw entirely.
        return SignalBatch(
            level_signal=np.zeros((s, p, n_weeks), dtype=np.float64),
            observed=avail.available,
        )

    noise = rng.normal(seed, rng.Kind.SIGNAL_NOISE, sims, keys, weeks)
    noise *= pool.signal_noise_sd.reshape(1, p, 1)
    level_signal = latent.season_shift.reshape(s, p, 1) + noise
    return SignalBatch(level_signal=level_signal, observed=avail.available)


def _draw_realized(
    pool: PoolArrays,
    seed: int,
    sims: np.ndarray,
    keys: np.ndarray,
    n_weeks: int,
    latent: LatentState,
    avail: AvailabilityBatch,
    contingency: np.ndarray,
) -> RealizedBatch:
    s, p = sims.shape[0], keys.shape[1]
    weeks = np.arange(n_weeks, dtype=np.int64).reshape(1, 1, n_weeks)

    # Correlated component: one shock per (group, week), loaded per player.
    if np.any(pool.beta):
        n_groups = pool.group_key.shape[0]
        shock = rng.normal(
            seed,
            rng.Kind.GROUP_SHOCK,
            sims.reshape(s, 1, 1),
            pool.group_key.reshape(1, n_groups, 1),
            weeks,
        )  # (S, G, W)
        group_effect = np.tensordot(pool.beta, shock, axes=([1], [1])).transpose(1, 0, 2)
    else:
        group_effect = np.zeros((s, p, n_weeks), dtype=np.float64)

    idio = rng.normal(seed, rng.Kind.WEEK_NOISE, sims, keys, weeks)
    idio *= pool.week_sd.reshape(1, p, 1)

    # Unforecastable spike weeks.  The unconditional mean is removed so that
    # raising `spike_rate` changes the shape of the distribution without
    # changing its mean -- which is what makes the "predictable upside vs
    # unforecastable spikes" experiment a clean comparison.
    if np.any(pool.spike_rate > 0.0):
        hit = rng.uniform(seed, rng.Kind.SPIKE_HIT, sims, keys, weeks) < pool.spike_rate.reshape(1, p, 1)
        size = rng.exponential(seed, rng.Kind.SPIKE_SIZE, sims, keys, weeks) * pool.spike_scale.reshape(1, p, 1)
        spike = hit * size - (
            pool.spike_rate * pool.spike_scale * pool.spike_demean
        ).reshape(1, p, 1)
    else:
        spike = np.zeros((s, p, n_weeks), dtype=np.float64)

    # Accumulate in place: each of these terms is an (S, P, W) array, and a
    # naive sum would allocate six full-size temporaries per batch.
    #
    # There is deliberately **no** `maximum(raw, 0)` here.  Interceptions and
    # lost fumbles are both -2 in this league's rules and nothing floors an
    # individual player's weekly total, so a bad week really can be negative.
    # Flooring would have made `base_mean` a latent parameter rather than the
    # player's expected points, which is not what any projection source means.
    raw = np.empty((s, p, n_weeks), dtype=np.float64)
    np.add(pool.base_mean.reshape(1, p, 1), latent.season_shift.reshape(s, p, 1), out=raw)
    raw += latent.true_role_delta
    raw += contingency
    raw += group_effect
    raw += idio
    raw += spike
    # Players who do not play score exactly zero -- that is an availability
    # rule, not a floor on performance.
    points = np.where(avail.available, raw, 0.0)
    return RealizedBatch(points=points, spike=spike, group_effect=group_effect)


def _build_pregame(
    pool: PoolArrays,
    seed: int,
    sims: np.ndarray,
    keys: np.ndarray,
    n_weeks: int,
    latent: LatentState,
    avail: AvailabilityBatch,
    signals: SignalBatch,
    contingency: np.ndarray,
) -> PregameBatch:
    """Project week *w* from information available before week *w*.

    Note the signature: there is **no realized argument**.  Beliefs are formed
    from :class:`SignalBatch` -- a separate observable process that reads the
    persistent latent level and nothing else -- so realized idiosyncratic
    noise, spike weeks and shared team shocks have no path into a projection.
    That is a structural guarantee, not a property of the arithmetic below.

    Week *w*'s projection sums three quantities, each contributed by exactly
    one source:

    ``base_mean + posterior``
        the persistent player level.  ``posterior`` is a Gaussian conjugate
        update on the signals of weeks strictly before *w*, with prior mean 0
        and prior SD ``season_sd``.
    ``observed_role_delta``
        the revealed part of a role change, in full, from its reveal week.
        Because the signal channel cannot see role changes, an unrevealed one
        does **not** first appear as an inflated persistent level and then get
        added a second time on reveal.
    ``contingency_bonus``
        the forecastable uplift from a depth-chart superior being out.
    """
    s, p = sims.shape[0], keys.shape[1]
    weeks = np.arange(n_weeks, dtype=np.int64).reshape(1, 1, n_weeks)

    observed = signals.observed
    sig = np.where(observed, signals.level_signal, 0.0)

    # Shift by one week: week w sees weeks 0..w-1 and nothing else.
    cum_sig = np.zeros_like(sig)
    cum_n = np.zeros(sig.shape, dtype=np.float64)
    np.cumsum(sig[:, :, :-1], axis=2, out=cum_sig[:, :, 1:])
    np.cumsum(observed[:, :, :-1].astype(np.float64), axis=2, out=cum_n[:, :, 1:])

    # posterior = tau_sig * S / (tau_0 + n * tau_sig), prior mean 0.
    # Written as S / (signal_noise_sd^2 / season_sd^2 + n) so season_sd == 0
    # gives exactly 0 -- consensus is then known to be right and nothing is
    # learnable.
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(
            pool.season_sd > 0.0,
            (pool.signal_noise_sd ** 2) / np.maximum(pool.season_sd ** 2, _EPS),
            np.inf,
        ).reshape(1, p, 1)
        posterior = np.where(np.isinf(ratio), 0.0, cum_sig / (ratio + cum_n))

    # The conditional mean of the week's score given everything observable.
    # Scores are not floored, so no distributional correction belongs here.
    projection = (
        pool.base_mean.reshape(1, p, 1)
        + posterior
        + latent.observed_role_delta
        + contingency
    )
    if pool.proj_noise_sd.max() > 0.0:
        projection = projection + (
            rng.normal(seed, rng.Kind.PROJ_NOISE, sims, keys, weeks)
            * pool.proj_noise_sd.reshape(1, p, 1)
        )

    if pool.proj_override is not None:
        mask = pool.proj_override_mask.reshape(1, p, 1)
        projection = np.where(mask, pool.proj_override.reshape(1, p, n_weeks), projection)

    return PregameBatch(
        projection=projection,
        observed_role_delta=latent.observed_role_delta,
        contingency_bonus=contingency,
        posterior_mean=posterior,
        n_observed=cum_n,
    )


def generate_world(
    pool: PoolArrays, seed: int, sim_start: int, n_sims: int
) -> WorldBatch:
    """Generate seasons ``[sim_start, sim_start + n_sims)``.

    Because the RNG is counter-based, the world for simulation index *i* is
    the same no matter which batch it is generated in, so chunking never
    changes results.
    """
    sims = (np.arange(n_sims, dtype=np.int64) + sim_start).reshape(n_sims, 1)
    keys = pool.stream_key.reshape(1, pool.n_players)
    w = pool.n_weeks

    sims3 = sims.reshape(n_sims, 1, 1)
    keys3 = keys.reshape(1, pool.n_players, 1)

    latent = _draw_latent(pool, seed, sims, keys, w)
    avail = _draw_availability(pool, seed, sims3, keys3, w)
    contingency = _contingency_bonus(pool, avail.available)
    signals = _draw_signals(pool, seed, sims3, keys3, w, latent, avail)
    realized = _draw_realized(pool, seed, sims3, keys3, w, latent, avail, contingency)
    # `realized` is deliberately not passed to `_build_pregame`.
    pregame = _build_pregame(
        pool, seed, sims3, keys3, w, latent, avail, signals, contingency
    )
    return WorldBatch(
        sim_start=sim_start,
        n_sims=n_sims,
        pool=pool,
        latent=latent,
        availability=avail,
        signals=signals,
        realized=realized,
        pregame=pregame,
    )
