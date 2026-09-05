"""Mapping the normalized contract into ``PlayerSpec``, with the assumptions visible.

Nothing here is a settled estimate. Every quantity this module produces is
either (a) calibrated numerically against a stated target, with its error
reported, or (b) a labelled **sensitivity scenario**. The configuration object
exists so that no assumption can hide in a constant.

Four things are worth reading before the code.

**Identity is a pure function of the player.** ``player_id`` and ``crn_key``
come from the canonical player key, never from a row index. They are RNG
coordinates: an id that moves when the pool is reordered gives the same player
a different simulated career in every scenario, and every paired comparison
built on that is comparing two different people.

**The projection target is two questions, not one.** *What statistic* the
source's season total reports (median or mean) and *what health state* it
describes (full health, or already net of expected absence) are independent,
and both are swept. Under ``full_health`` the level is calibrated with no
injury process and absences are applied afterwards, so unconditional season
output lands **below** the source total. Under ``availability_adjusted`` the
injury process is inside the solve, so the simulated full-season total matches
the source total after absences and the active-game level is correspondingly
higher.

**Injury parameters are solved over the season they describe.** The vendor's
injury probability and projected games missed span the full NFL season: 18
calendar weeks, 17 scheduled games, one bye. That is the horizon the solve
uses, and both targets are matched there directly. The fitted parameters are
then carried into the fantasy Weeks 1-17 simulation, which contains 16
scheduled games, and the expected absence over *that* window is reported as a
separate number rather than substituted for the target.

**Signal quality is explicit.** ``PlayerSpec.signal_noise_sd`` defaults to
``None``, which the engine reads as ``week_sd``. A real spec must never leave
it there: the implicit value ties how fast managers learn to how noisy scoring
is, and that coupling silently contaminates any sweep over ``season_sd``. The
mapping always writes an explicit value chosen by ``signal_quality``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from .. import rng
from ..league import DEFAULT_LEAGUE, LeagueSettings, Position
from ..players import PlayerSpec
from .identity import assign_stable_ids, canonical_player_key

__all__ = [
    "PlayerSpecMappingConfig",
    "LevelCalibration",
    "InjuryCalibration",
    "MappedPlayer",
    "MappingResult",
    "calibrate_level",
    "calibrate_injury",
    "map_contract_to_playerspecs",
    "FORECASTABLE_SHARE_SCENARIOS",
    "SEASON_SD_SCENARIOS",
    "AVAILABILITY_INTERPRETATIONS",
    "SIGNAL_QUALITY_SCENARIOS",
    "resolve_signal_noise_sd",
]

#: Share of total weekly dispersion assumed knowable before lineup lock.
#: **Scenarios, not estimates.** Nothing in the inventory measures this.
FORECASTABLE_SHARE_SCENARIOS: Tuple[float, ...] = (0.00, 0.25, 0.50)

#: Season-level uncertainty as a fraction of the active mean. Also scenarios.
SEASON_SD_SCENARIOS: Tuple[float, ...] = (0.00, 0.10, 0.20)

#: What health state the source's season total describes. **Neither is
#: preferred**; the source's own documentation says its projections do not
#: fully capture current health, which places it between the two rather than
#: at either end.
AVAILABILITY_INTERPRETATIONS: Tuple[str, ...] = ("full_health",
                                                 "availability_adjusted")

#: How informative one week of observed usage is about a player's persistent
#: level. ``"none"`` is a league where managers never learn; the other two set
#: ``signal_noise_sd`` to a stated multiple of ``week_sd``. All three are
#: scenarios -- nothing in the inventory measures learning speed.
SIGNAL_QUALITY_SCENARIOS: Tuple[str, ...] = ("none", "week_sd", "2x_week_sd")


def resolve_signal_noise_sd(signal_quality: str, week_sd: float) -> float:
    """The explicit ``signal_noise_sd`` a signal-quality scenario implies.

    ``"none"`` returns ``inf``: an infinitely noisy reading of the latent level
    carries no information, so the posterior stays at the prior and the
    projection never updates. That is the honest encoding of "managers do not
    learn" -- distinct from ``season_sd = 0``, which says there is nothing to
    learn in the first place. Keeping the two separate is the whole point of
    this axis.
    """
    if signal_quality == "none":
        return math.inf
    if signal_quality == "week_sd":
        return float(week_sd)
    if signal_quality == "2x_week_sd":
        return 2.0 * float(week_sd)
    raise ValueError(f"unknown signal_quality {signal_quality!r}")


@dataclass(frozen=True)
class PlayerSpecMappingConfig:
    """Every assumption the mapping makes, in one place.

    No default here is a finding. Each is either a documented fact about the
    source (``games_basis``), a provisional reading being carried alongside its
    alternative (``target``, ``projection_availability_interpretation``), or
    the first point of a sensitivity sweep.
    """

    target: str = "median_target"
    """``"median_target"`` or ``"mean_target"``.

    Which *statistic* of the season total the source reports. The published
    methodology makes the total a hybrid -- continuous prop categories are
    market medians, discrete ones are probability-weighted expectations -- so
    neither reading is proven and both are swept."""

    projection_availability_interpretation: str = "full_health"
    """``"full_health"`` or ``"availability_adjusted"``.

    Which *health state* the season total describes; independent of ``target``.
    Under ``full_health`` the level is solved with no injury process and
    absences are applied on top, so the unconditional season output falls below
    the source total by roughly the projected missed games. Under
    ``availability_adjusted`` the injury process is inside the solve, so the
    simulated full-season total reproduces the source total *after* absences
    and the per-active-game level is higher for anyone projected to miss time.

    **Neither is preferred.** The source states its projections do not fully
    capture current health and that injury designations are applied manually,
    which puts it somewhere between the two. Both enter sensitivity."""

    signal_quality: str = "week_sd"
    """``"none"``, ``"week_sd"`` or ``"2x_week_sd"``; see
    :func:`resolve_signal_noise_sd`. A scenario, and one that must be stated:
    leaving ``signal_noise_sd`` at ``None`` makes learning speed a silent
    function of scoring noise, which contaminates the ``season_sd`` sweep."""

    fumble_interpretation: str = "exclude"
    """``"exclude"`` (default), ``"lost"`` or ``"total"``. The source's
    published category list does not include fumbles, so the column's meaning
    is unresolved; excluded from the primary mapping and swept."""

    forecastable_share: float = 0.0
    """Fraction of total weekly variance treated as knowable before lock.
    A **scenario**. ``week_sd`` and ``weekly_state_sd`` are split as
    ``sqrt(1-f)`` and ``sqrt(f)`` of the total, which preserves total variance
    exactly."""

    season_sd_fraction: float = 0.0
    """Season-level uncertainty as a fraction of the active mean. A scenario."""

    injury_model: str = "individual"
    """``"individual"`` calibrates per-player hazard and duration against the
    supplied season injury probability and games missed. ``"positional"`` uses
    the fitted all-cause rate for everyone. ``"none"`` leaves availability
    unmodelled, which is not the same as healthy and is labelled as such."""

    missing_injury_fallback: str = "positional_all_cause"
    """What to do for a player with no individual profile.
    ``"positional_all_cause"`` uses the fitted positional weekly-miss rate,
    which includes benching, rest and trades and is **not** an injury-only
    hazard. ``"none"`` leaves him unmodelled. Silently treating him as
    perfectly healthy is not an option this config offers."""

    calibration_sims: int = 200_000
    """Simulations behind a healthy-level calibration. Large because the solve
    is cached per dispersion shape rather than run per player, so the cost is
    paid a handful of times and finite-sample error stops mattering."""

    availability_calibration_sims: int = 25_000
    """Simulations behind an ``availability_adjusted`` level calibration.

    Smaller than ``calibration_sims`` because that solve cannot be cached per
    dispersion shape: it depends on the player's own fitted hazard, duration
    and bye, so it runs once per distinct availability profile. The Monte Carlo
    standard error it leaves on the level is computed and reported per player
    rather than assumed negligible."""

    injury_calibration_sims: int = 4_000
    """Simulations behind an injury solve. Runs per distinct (probability,
    games-missed, bye) triple and drives a 2-D search, so it is deliberately
    cheaper than the level solve. At 4,000 draws the standard error on a rate
    near 0.35 is about 0.0075, comfortably inside the feasibility tolerance."""

    calibration_seed: int = 20260904
    games_basis: float = 17.0
    """Scheduled games the source's season total spans: the full NFL regular
    season. The calendar containing them is one week longer, because of the
    bye; see :attr:`full_season_weeks`."""

    settings: LeagueSettings = DEFAULT_LEAGUE

    def __post_init__(self) -> None:
        if self.target not in ("median_target", "mean_target"):
            raise ValueError(f"unknown target {self.target!r}")
        if self.projection_availability_interpretation not in AVAILABILITY_INTERPRETATIONS:
            raise ValueError(
                "unknown projection_availability_interpretation "
                f"{self.projection_availability_interpretation!r}")
        if self.signal_quality not in SIGNAL_QUALITY_SCENARIOS:
            raise ValueError(f"unknown signal_quality {self.signal_quality!r}")
        if self.fumble_interpretation not in ("exclude", "lost", "total"):
            raise ValueError(
                f"unknown fumble_interpretation {self.fumble_interpretation!r}")
        if not 0.0 <= self.forecastable_share < 1.0:
            raise ValueError("forecastable_share must be in [0, 1)")
        if self.season_sd_fraction < 0.0:
            raise ValueError("season_sd_fraction must be non-negative")
        if self.injury_model not in ("individual", "positional", "none"):
            raise ValueError(f"unknown injury_model {self.injury_model!r}")
        if self.missing_injury_fallback not in ("positional_all_cause", "none"):
            raise ValueError(
                f"unknown missing_injury_fallback {self.missing_injury_fallback!r}")
        if self.calibration_sims < 100:
            raise ValueError("calibration_sims must be at least 100")
        if self.availability_calibration_sims < 100:
            raise ValueError("availability_calibration_sims must be at least 100")

    # --- horizons ----------------------------------------------------------
    #
    # Three different spans are in play and conflating any two of them was the
    # defect this pass exists to fix.

    @property
    def full_season_games(self) -> int:
        """Scheduled NFL games the vendor's season figures describe: 17."""
        return int(round(self.games_basis))

    @property
    def full_season_weeks(self) -> int:
        """Calendar weeks containing those games: 18, because of the bye.

        Injury hazard is a *per-week* rate, so it must be fitted over the
        calendar the targets describe. Fitting it over 17 weeks and then
        comparing to a 17-*game* target mixes the two units."""
        return self.full_season_games + 1

    @property
    def fantasy_weeks(self) -> int:
        """Simulated fantasy weeks: 17 (Weeks 1-17)."""
        return self.settings.total_weeks

    @property
    def fantasy_scheduled_games(self) -> int:
        """Scheduled games inside the fantasy window: 16, because of the bye."""
        return self.fantasy_weeks - 1

    def label(self) -> str:
        """Short, stable identifier for a scenario cell."""
        avail = ("fh" if self.projection_availability_interpretation == "full_health"
                 else "aa")
        return (f"{self.target}|avail={avail}|f={self.forecastable_share:.2f}"
                f"|ssd={self.season_sd_fraction:.2f}"
                f"|sig={self.signal_quality}"
                f"|inj={self.injury_model}|fum={self.fumble_interpretation}")


# ---------------------------------------------------------------------------
# Injury calibration
# ---------------------------------------------------------------------------
#
# Solved first, because an ``availability_adjusted`` level solve needs the
# fitted availability process before it can run.


@dataclass(frozen=True)
class InjuryCalibration:
    """Hazard and duration solved against two supplied full-season targets.

    Every reported quantity names the horizon it belongs to. The two targets
    and their achieved values are **full-season** (18 calendar weeks, 17
    scheduled games, one bye), because that is the span the vendor's figures
    describe. ``expected_fantasy_games_missed`` is a *consequence* of the
    fitted parameters over the shorter fantasy window (17 weeks, 16 games), not
    a target and not a rescaled target.
    """

    weekly_injury_hazard: float
    injury_mean_weeks: float
    source: str
    """``"individual"``, ``"positional_all_cause"`` or ``"unmodelled"``."""

    target_injury_prob: Optional[float] = None
    achieved_injury_prob: Optional[float] = None
    target_games_missed: Optional[float] = None
    achieved_games_missed: Optional[float] = None
    expected_fantasy_games_missed: Optional[float] = None
    """Expected scheduled games missed inside fantasy Weeks 1-17, at the fitted
    parameters. Reported, never fitted."""
    fantasy_injury_prob: Optional[float] = None
    full_season_weeks: int = 18
    full_season_games: int = 17
    fantasy_weeks: int = 17
    fantasy_scheduled_games: int = 16
    feasible: bool = True
    note: str = ""

    @property
    def injury_prob_error(self) -> Optional[float]:
        if self.target_injury_prob is None or self.achieved_injury_prob is None:
            return None
        return self.achieved_injury_prob - self.target_injury_prob

    @property
    def games_missed_error(self) -> Optional[float]:
        if self.target_games_missed is None or self.achieved_games_missed is None:
            return None
        return self.achieved_games_missed - self.target_games_missed

    @property
    def expected_games_played_full_season(self) -> Optional[float]:
        if self.achieved_games_missed is None:
            return None
        return self.full_season_games - self.achieved_games_missed


def _simulate_availability(hazard, mean_weeks, bye_index: int,
                           weeks: int, sims: int, seed: int,
                           return_played: bool = False):
    """``(P(any injury), E[games missed])`` under the engine's own process.

    ``hazard`` and ``mean_weeks`` may be arrays, in which case the whole grid is
    simulated at once and the returns have the grid's shape. That matters: the
    calibration inverts this function, and inverting it one player at a time
    was the difference between seconds and minutes.

    This mirrors ``worlds._draw_availability`` exactly, including that a week
    which is both a bye and an absence costs no *game* -- the player was not
    playing that week anyway. Conflating calendar weeks with scheduled games
    here would bias every duration estimate.

    With ``return_played`` the per-simulation ``(…, sims, weeks)`` boolean
    "played a scheduled game this week" array is returned as a third element,
    which is what an availability-adjusted level solve consumes.
    """
    hazard = np.atleast_1d(np.asarray(hazard, dtype=np.float64))
    mean_weeks = np.atleast_1d(np.asarray(mean_weeks, dtype=np.float64))
    grid_shape = np.broadcast(hazard, mean_weeks).shape
    h = np.broadcast_to(hazard, grid_shape)[..., None]
    d = np.broadcast_to(mean_weeks, grid_shape)[..., None]

    s_idx = np.arange(sims, dtype=np.int64).reshape(1, sims)
    out_until = np.full(grid_shape + (sims,), -1, dtype=np.int64)
    ever = np.zeros(grid_shape + (sims,), dtype=bool)
    missed = np.zeros(grid_shape + (sims,), dtype=np.float64)
    played = (np.zeros(grid_shape + (sims, weeks), dtype=bool)
              if return_played else None)

    for week in range(weeks):
        wk = np.full((1, 1), week, dtype=np.int64)
        onset_u = rng.uniform(seed, rng.Kind.INJURY_ONSET, s_idx, wk).reshape(1, sims)
        dur_e = rng.exponential(seed, rng.Kind.INJURY_DURATION, s_idx, wk).reshape(1, sims)
        healthy = out_until < week
        onset = healthy & (onset_u < h)
        ever |= onset
        duration = np.maximum(1, np.rint(dur_e * d).astype(np.int64))
        out_until = np.where(onset, week + duration - 1, out_until)
        absent = out_until >= week
        if week != bye_index:          # a bye week costs no scheduled game
            missed += absent
            if played is not None:
                played[..., week] = ~absent

    prob = ever.mean(axis=-1)
    miss = missed.mean(axis=-1)
    if grid_shape == (1,):
        out = (float(prob[0]), float(miss[0]))
        return out + (played[0],) if return_played else out
    return (prob, miss, played) if return_played else (prob, miss)


#: Search grid for the injury inverse. Log-spaced in both axes because the
#: targets are far more sensitive at the low end of each. Deliberately coarse:
#: its only job is to localise the solution so the refinement below starts in
#: the right basin, and a fine grid would cost far more than the refinement it
#: replaces.
_HAZARD_GRID = np.geomspace(1e-4, 0.60, 24)
_DURATION_GRID = np.geomspace(0.5, 30.0, 20)
_GRID_CACHE: Dict[Tuple, Tuple[np.ndarray, np.ndarray]] = {}


#: Bye week used when building the localisation grid. The grid's only job is to
#: put the refinement in the right basin, and moving the bye shifts expected
#: games missed by well under one game -- far less than the grid's own spacing.
#: The refinement rounds below use each player's ACTUAL bye, so accuracy is not
#: affected; this just avoids rebuilding the grid ten times over.
_GRID_REFERENCE_BYE = 8


def _availability_grid(weeks: int, sims: int, seed: int):
    """``(prob, missed)`` over the whole (hazard, duration) grid, cached."""
    key = (_GRID_REFERENCE_BYE, weeks, sims, seed)
    hit = _GRID_CACHE.get(key)
    if hit is not None:
        return hit
    h = _HAZARD_GRID[:, None]
    d = _DURATION_GRID[None, :]
    prob, miss = _simulate_availability(h, d, _GRID_REFERENCE_BYE, weeks, sims, seed)
    _GRID_CACHE[key] = (prob, miss)
    return prob, miss


def _invert_grid(prob: np.ndarray, miss: np.ndarray, target_prob: float,
                 target_missed: float) -> Tuple[float, float, float, float]:
    """Grid point minimising relative error on both targets simultaneously.

    Relative rather than absolute, so a 0.05 probability and a 4-game absence
    are weighted comparably instead of the larger number dominating.
    """
    pe = (prob - target_prob) / max(target_prob, 1e-6)
    me = (miss - target_missed) / max(target_missed, 1e-6)
    cost = pe ** 2 + me ** 2
    i, j = np.unravel_index(int(np.argmin(cost)), cost.shape)
    return (float(_HAZARD_GRID[i]), float(_DURATION_GRID[j]),
            float(prob[i, j]), float(miss[i, j]))


_INJURY_CACHE: Dict[Tuple, "InjuryCalibration"] = {}


def calibrate_injury(injury_prob: Optional[float],
                     proj_games_missed: Optional[float],
                     config: PlayerSpecMappingConfig,
                     positional_miss_rate: Optional[float] = None,
                     position: str = "",
                     bye_index: int = 8) -> InjuryCalibration:
    """Solve ``(hazard, mean_weeks)`` against the two supplied season targets.

    **Horizon.** Both vendor figures -- season injury probability and projected
    games missed -- describe the full NFL season. The solve therefore runs over
    18 calendar weeks containing 17 scheduled games and one bye, and matches
    both targets there *directly*. An earlier pass scaled projected games
    missed by 16/17 onto the fantasy window while leaving the injury
    probability untouched, which asked the fit to reproduce two targets defined
    on different spans; that inconsistency is removed.

    The fitted parameters are per-week rates, so they carry into the fantasy
    Weeks 1-17 simulation unchanged. The expected absence over *that* window --
    17 weeks, 16 scheduled games -- is computed and reported as
    ``expected_fantasy_games_missed``, a consequence rather than a target.

    Returns an ``unmodelled`` or ``positional_all_cause`` calibration when an
    individual profile is unavailable, per the configured fallback. It never
    silently returns perfect health.
    """
    weeks = config.full_season_weeks             # 18 calendar weeks
    games = config.full_season_games             # 17 scheduled games
    f_weeks = config.fantasy_weeks               # 17 fantasy weeks
    f_games = config.fantasy_scheduled_games     # 16 scheduled games
    sims = config.injury_calibration_sims
    seed = config.calibration_seed

    horizons = dict(full_season_weeks=weeks, full_season_games=games,
                    fantasy_weeks=f_weeks, fantasy_scheduled_games=f_games)

    cache_key = (
        None if injury_prob is None else round(float(injury_prob), 4),
        None if proj_games_missed is None else round(float(proj_games_missed), 3),
        int(bye_index), weeks, f_weeks, sims, seed, config.injury_model,
        config.missing_injury_fallback,
        None if positional_miss_rate is None else round(float(positional_miss_rate), 6),
        position,
    )
    cached = _INJURY_CACHE.get(cache_key)
    if cached is not None:
        return cached

    def _cache(result: "InjuryCalibration") -> "InjuryCalibration":
        _INJURY_CACHE[cache_key] = result
        return result

    def _fantasy(hazard: float, mean_weeks: float) -> Tuple[float, float]:
        """What the fitted parameters imply over fantasy Weeks 1-17."""
        prob, missed = _simulate_availability(
            hazard, mean_weeks, bye_index, f_weeks, sims, seed)
        return float(prob), float(missed)

    if config.injury_model == "none":
        return _cache(InjuryCalibration(
            0.0, 2.5, "unmodelled", expected_fantasy_games_missed=0.0,
            fantasy_injury_prob=0.0, feasible=True, **horizons,
            note="availability unmodelled by configuration; this is NOT a claim "
                 "of perfect health"))

    have_individual = (injury_prob is not None and proj_games_missed is not None
                       and injury_prob > 0 and proj_games_missed > 0)

    if config.injury_model == "positional" or not have_individual:
        if config.missing_injury_fallback == "none" and not have_individual:
            return _cache(InjuryCalibration(
                0.0, 2.5, "unmodelled", expected_fantasy_games_missed=0.0,
                fantasy_injury_prob=0.0, feasible=True, **horizons,
                note="no individual profile and fallback disabled; availability "
                     "unmodelled, NOT healthy"))
        rate = positional_miss_rate
        if rate is None:
            return _cache(InjuryCalibration(
                0.0, 2.5, "unmodelled", expected_fantasy_games_missed=0.0,
                fantasy_injury_prob=0.0, feasible=False, **horizons,
                note=f"no individual profile and no positional rate for "
                     f"{position!r}; availability unmodelled, NOT healthy"))
        prob, missed = _simulate_availability(rate, 2.5, bye_index, weeks,
                                              sims, seed)
        f_prob, f_missed = _fantasy(rate, 2.5)
        return _cache(InjuryCalibration(
            weekly_injury_hazard=rate, injury_mean_weeks=2.5,
            source="positional_all_cause",
            achieved_injury_prob=prob, achieved_games_missed=missed,
            expected_fantasy_games_missed=f_missed, fantasy_injury_prob=f_prob,
            feasible=True, **horizons,
            note="ALL-CAUSE availability rate fitted from historical weekly "
                 "results: it counts benching, rest and trades as well as "
                 "injury, and is not an injury-only hazard. Duration is the "
                 "engine default, not fitted."))

    # Both targets are full-season quantities and are matched as supplied.
    target_missed = float(proj_games_missed)
    target_prob = float(injury_prob)

    # Invert the engine's own availability process over a precomputed grid.
    # Hazard drives P(any onset) and duration drives games missed, but the two
    # interact -- being absent blocks new onsets -- so both targets are matched
    # jointly rather than one after the other.
    grid_prob, grid_miss = _availability_grid(weeks, sims, seed)
    hazard, mean_weeks, prob, missed = _invert_grid(
        grid_prob, grid_miss, target_prob, target_missed)

    # Polish off the grid. The coarse grid localises the solution; two rounds
    # of a tighter local grid then remove the discretisation error, which would
    # otherwise be mistaken for the two targets being incompatible. Both rounds
    # are vectorised -- refining by scalar bisection was the difference between
    # milliseconds and seconds per player.
    for span in (3.0, 1.35):
        h_axis = np.geomspace(max(hazard / span, 1e-5),
                              min(hazard * span, 0.95), 12)
        d_axis = np.geomspace(max(mean_weeks / span, 0.05),
                              mean_weeks * span, 12)
        prob_g, miss_g = _simulate_availability(
            h_axis[:, None], d_axis[None, :], bye_index, weeks, sims, seed)
        pe = (prob_g - target_prob) / max(target_prob, 1e-6)
        me = (miss_g - target_missed) / max(target_missed, 1e-6)
        i, j = np.unravel_index(int(np.argmin(pe ** 2 + me ** 2)), pe.shape)
        hazard, mean_weeks = float(h_axis[i]), float(d_axis[j])
        prob, missed = float(prob_g[i, j]), float(miss_g[i, j])

    f_prob, f_missed = _fantasy(hazard, mean_weeks)

    prob_err = abs(prob - target_prob)
    missed_err = abs(missed - target_missed)
    # Tolerances are set against the calibration's own Monte Carlo noise: at
    # 4,000 draws the standard error on a rate near 0.35 is about 0.0075, so
    # 0.02 is under three of them. Anything outside that is structure, not
    # sampling.
    feasible = prob_err <= 0.02 and missed_err <= 0.15
    note = ""
    if not feasible:
        note = (f"targets not jointly reproduced within tolerance over the "
                f"{weeks}-week / {games}-game NFL season: P(injury) "
                f"{prob:.3f} vs {target_prob:.3f} (err {prob_err:.3f}), games "
                f"missed {missed:.2f} vs {target_missed:.2f} (err "
                f"{missed_err:.3f}). The engine's process ties frequency and "
                f"duration together -- an absence blocks new onsets -- so a low "
                f"injury probability combined with many games missed, or the "
                f"reverse, has no solution. The fitted parameters are the "
                f"closest available and both errors are reported.")
    return _cache(InjuryCalibration(
        weekly_injury_hazard=hazard, injury_mean_weeks=mean_weeks,
        source="individual",
        target_injury_prob=target_prob, achieved_injury_prob=prob,
        target_games_missed=target_missed, achieved_games_missed=missed,
        expected_fantasy_games_missed=f_missed, fantasy_injury_prob=f_prob,
        feasible=feasible, **horizons, note=note))


# ---------------------------------------------------------------------------
# Level calibration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LevelCalibration:
    """The per-active-game level, and how well it hit its target."""

    level: float
    target: str
    availability_interpretation: str
    target_value: float
    achieved: float
    abs_error: float
    naive_level: float
    """``season_total / games_basis`` -- what a direct assignment would give."""
    unit_statistic: float
    """The chosen statistic of a full-season total at unit level."""
    unit_statistic_se: float = 0.0
    """Monte Carlo standard error of ``unit_statistic``. Zero only when the
    statistic is available in closed form."""
    expected_unconditional_season_output: Optional[float] = None
    """**Expected** (mean) full-season output once absences are applied.

    Under ``full_health`` this sits **below** ``target_value`` by construction:
    the level was solved as if the player played all 17 games and the injury
    process then removes some of them.

    Under ``availability_adjusted`` it is the *calibrated statistic* that
    reproduces the target, not necessarily this number. When the target is the
    median the two differ, because absences truncate the lower tail only and
    the unconditional total stops being symmetric. That gap is a result, not an
    error: it is the concrete reason mean and median cannot be treated as
    interchangeable once availability is in the model."""

    @property
    def relative_error(self) -> float:
        return self.abs_error / self.target_value if self.target_value else 0.0

    @property
    def divergence_from_naive(self) -> float:
        return self.level - self.naive_level

    @property
    def level_se(self) -> float:
        """MC standard error propagated onto the level."""
        if self.unit_statistic <= 0:
            return 0.0
        return self.level * self.unit_statistic_se / self.unit_statistic

    @property
    def unconditional_shortfall(self) -> Optional[float]:
        """How far *expected* unconditional season output falls below the target."""
        if self.expected_unconditional_season_output is None:
            return None
        return self.target_value - self.expected_unconditional_season_output


def _healthy_season_totals(level: float, week_sd: float, season_sd: float,
                           games: int, sims: int, seed: int) -> np.ndarray:
    """Simulated **full-health** season totals at a given per-game level.

    Uses the engine's own counter-based RNG, so a calibration is reproducible
    and two calls at the same seed compare like for like.
    """
    s = np.arange(sims, dtype=np.int64).reshape(sims, 1)
    w = np.arange(games, dtype=np.int64).reshape(1, games)
    weekly = np.full((sims, games), float(level))
    if week_sd > 0:
        weekly = weekly + rng.normal(seed, rng.Kind.WEEK_NOISE, s, w) * week_sd
    if season_sd > 0:
        shift = rng.normal(seed, rng.Kind.SEASON_SHIFT, s,
                           np.zeros((1, 1), dtype=np.int64))
        weekly = weekly + shift
    return weekly.sum(axis=1)


def _statistic(totals: np.ndarray, target: str) -> Tuple[float, float]:
    """``(statistic, its Monte Carlo standard error)``.

    The median's standard error uses the standard large-sample result
    ``1.253 * sd / sqrt(n)``, valid for a smooth unimodal density, which a sum
    of seventeen weekly draws comfortably is.
    """
    n = totals.shape[0]
    sd = float(np.std(totals, ddof=1))
    if target == "median_target":
        return float(np.median(totals)), 1.2533 * sd / math.sqrt(n)
    return float(np.mean(totals)), sd / math.sqrt(n)


_UNIT_STAT_CACHE: Dict[Tuple, Tuple[float, float]] = {}


def _unit_healthy_statistic(week_sd_fraction: float, season_sd_fraction: float,
                            games: int, sims: int, seed: int, target: str
                            ) -> Tuple[float, float]:
    """The chosen statistic of a full-health season total at **unit** level.

    The whole simulated season scales exactly linearly in the level: every
    dispersion term is expressed as a fraction of it, so a season total at
    level ``L`` is ``L`` times the same total at level 1. Both the mean and the
    median are positively homogeneous, so

        stat(total at level L) = L * stat(total at level 1)

    and the calibration is an exact division rather than a search. That is not
    a shortcut around the solve -- it is the solve, done in closed form once the
    simulated sample is fixed, and it lets the sample be large enough that
    finite-sample error stops mattering. Cached because it depends only on the
    dispersion shape, not on the player.
    """
    key = (round(week_sd_fraction, 10), round(season_sd_fraction, 10), games,
           sims, seed, target)
    hit = _UNIT_STAT_CACHE.get(key)
    if hit is not None:
        return hit
    totals = _healthy_season_totals(1.0, week_sd_fraction, season_sd_fraction,
                                    games, sims, seed)
    stat = _statistic(totals, target)
    _UNIT_STAT_CACHE[key] = stat
    return stat


_AVAIL_STAT_CACHE: Dict[Tuple, Tuple[float, float]] = {}


def _unit_available_statistic(week_sd_fraction: float, season_sd_fraction: float,
                              hazard: float, mean_weeks: float, bye_index: int,
                              weeks: int, sims: int, seed: int, target: str
                              ) -> Tuple[float, float]:
    """The chosen statistic of an **unconditional** season total at unit level.

    Identical homogeneity argument to the healthy case -- availability is drawn
    independently of the level, so the total is still exactly ``L`` times the
    unit-level total and the solve is still an exact division. What changes is
    that the distribution is no longer symmetric: a missed game removes a whole
    week's production from the lower tail only. Mean and median therefore
    **separate**, and the two targets stop agreeing.

    This cannot be cached by dispersion shape alone -- it depends on the
    player's own fitted hazard, duration and bye -- so it runs with fewer
    simulations and its Monte Carlo error is reported rather than assumed away.
    """
    key = (round(week_sd_fraction, 8), round(season_sd_fraction, 8),
           round(hazard, 8), round(mean_weeks, 6), int(bye_index), weeks,
           sims, seed, target)
    hit = _AVAIL_STAT_CACHE.get(key)
    if hit is not None:
        return hit

    _, _, played = _simulate_availability(
        hazard, mean_weeks, bye_index, weeks, sims, seed, return_played=True)
    played = played.astype(np.float64)          # (sims, weeks)

    s = np.arange(sims, dtype=np.int64).reshape(sims, 1)
    w = np.arange(weeks, dtype=np.int64).reshape(1, weeks)
    weekly = np.ones((sims, weeks), dtype=np.float64)
    if week_sd_fraction > 0:
        weekly = weekly + rng.normal(seed, rng.Kind.WEEK_NOISE, s, w) * week_sd_fraction
    if season_sd_fraction > 0:
        shift = rng.normal(seed, rng.Kind.SEASON_SHIFT, s,
                           np.zeros((1, 1), dtype=np.int64))
        weekly = weekly + shift
    totals = (weekly * played).sum(axis=1)
    stat = _statistic(totals, target)
    _AVAIL_STAT_CACHE[key] = stat
    return stat


def calibrate_level(season_total: float, week_sd_fraction: float,
                    season_sd_fraction: float, config: PlayerSpecMappingConfig,
                    injury: Optional[InjuryCalibration] = None,
                    bye_index: int = 8) -> LevelCalibration:
    """Solve for the per-active-game level that hits the chosen target.

    ``week_sd_fraction`` and ``season_sd_fraction`` are expressed relative to
    the level, so dispersion scales with it -- which is what makes the season
    total exactly homogeneous in the level and the calibration exact.

    **Under ``full_health``** the solve sees no injury process at all: the
    target is the total a player who plays all 17 games would produce. Absences
    are then applied downstream, so the *unconditional* season output lands
    below the source total by roughly the projected missed games. That shortfall
    is computed and reported rather than left implicit.

    With the fields this source can populate, every full-health weekly component
    is symmetric -- the idiosyncratic draw and the season shift are both normal,
    and the skewed component (spikes) has no source and stays unpopulated -- so
    the mean and median of a full-health season total coincide and both targets
    return ``season_total / games_basis``. That equivalence is a property of
    *this* configuration, not a general fact, and it does **not** survive
    ``availability_adjusted``.

    **Under ``availability_adjusted``** the fitted availability process is
    inside the solve, so the simulated unconditional full-season total matches
    the source total. Missed games truncate the lower tail only, the total stops
    being symmetric, and mean and median separate. The resulting per-active-game
    level is higher than the naive ``total / 17`` for any player projected to
    miss time, which is the intended difference between the two readings.
    """
    games = config.full_season_games
    naive = season_total / config.games_basis
    interp = config.projection_availability_interpretation

    if interp == "availability_adjusted":
        if injury is None:
            raise ValueError(
                "availability_adjusted level calibration needs the fitted "
                "InjuryCalibration; solve availability before the level")
        unit, unit_se = _unit_available_statistic(
            week_sd_fraction, season_sd_fraction,
            injury.weekly_injury_hazard, injury.injury_mean_weeks, bye_index,
            config.full_season_weeks, config.availability_calibration_sims,
            config.calibration_seed, config.target)
    else:
        unit, unit_se = _unit_healthy_statistic(
            week_sd_fraction, season_sd_fraction, games,
            config.calibration_sims, config.calibration_seed, config.target)

    if unit <= 0:
        raise ValueError(
            "the simulated unit-level season statistic is non-positive; the "
            "dispersion fractions are too large for this calibration")
    level = season_total / unit
    achieved = level * unit

    # What the player actually produces over a full season once absences bite.
    unconditional: Optional[float] = None
    if injury is not None and injury.achieved_games_missed is not None:
        unconditional = level * (games - injury.achieved_games_missed)
    elif injury is not None:
        unconditional = level * games

    return LevelCalibration(
        level=level, target=config.target, availability_interpretation=interp,
        target_value=season_total, achieved=achieved,
        abs_error=abs(achieved - season_total), naive_level=naive,
        unit_statistic=unit, unit_statistic_se=unit_se,
        expected_unconditional_season_output=unconditional)


# ---------------------------------------------------------------------------
# The mapping
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MappedPlayer:
    """One ``PlayerSpec`` plus the provenance of every number in it."""

    spec: PlayerSpec
    canonical_key: str
    level: LevelCalibration
    injury: InjuryCalibration
    total_week_sd: float
    forecastable_share: float
    season_sd_fraction: float
    signal_quality: str
    signal_noise_sd: float
    unresolved_placeholders: Tuple[str, ...]
    """Fields left at zero because no source supports them. **Placeholders,
    not estimates** -- a zero here means "unmodelled", never "measured zero"."""


@dataclass
class MappingResult:
    """Every mapped player, plus what the mapping could not do."""

    players: List[MappedPlayer]
    config: PlayerSpecMappingConfig
    warnings: List[str] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)

    @property
    def specs(self) -> List[PlayerSpec]:
        return [m.spec for m in self.players]

    @property
    def by_key(self) -> Dict[str, MappedPlayer]:
        return {m.canonical_key: m for m in self.players}

    def calibration_summary(self) -> Dict[str, object]:
        """Aggregate calibration error. Counts and ranges only."""
        lvl = [m.level.abs_error for m in self.players]
        lvl_se = [m.level.level_se for m in self.players]
        ind = [m for m in self.players if m.injury.source == "individual"]
        prob_err = [abs(m.injury.injury_prob_error) for m in ind
                    if m.injury.injury_prob_error is not None]
        miss_err = [abs(m.injury.games_missed_error) for m in ind
                    if m.injury.games_missed_error is not None]
        fantasy_missed = [m.injury.expected_fantasy_games_missed for m in self.players
                          if m.injury.expected_fantasy_games_missed is not None]
        full_missed = [m.injury.achieved_games_missed for m in self.players
                       if m.injury.achieved_games_missed is not None]
        shortfall = [m.level.unconditional_shortfall for m in self.players
                     if m.level.unconditional_shortfall is not None]
        by_source: Dict[str, int] = {}
        for m in self.players:
            by_source[m.injury.source] = by_source.get(m.injury.source, 0) + 1

        def stat(xs):
            if not xs:
                return None
            xs = sorted(xs)
            return {"n": len(xs), "max": round(xs[-1], 5),
                    "median": round(xs[len(xs) // 2], 5),
                    "mean": round(sum(xs) / len(xs), 5)}

        cfg = self.config
        return {
            "players": len(self.players),
            "target": cfg.target,
            "availability_interpretation": cfg.projection_availability_interpretation,
            "signal_quality": cfg.signal_quality,
            "horizons": {
                "calibration_calendar_weeks": cfg.full_season_weeks,
                "calibration_scheduled_games": cfg.full_season_games,
                "fantasy_weeks": cfg.fantasy_weeks,
                "fantasy_scheduled_games": cfg.fantasy_scheduled_games,
            },
            "level_abs_error": stat(lvl),
            "level_mc_se": stat(lvl_se),
            "unconditional_season_shortfall": stat(shortfall),
            "injury_source_counts": by_source,
            "full_season_injury_prob_abs_error": stat(prob_err),
            "full_season_games_missed_abs_error": stat(miss_err),
            "full_season_games_missed_achieved": stat(full_missed),
            "fantasy_games_missed_expected": stat(fantasy_missed),
            "infeasible_injury_calibrations": sum(
                1 for m in ind if not m.injury.feasible),
        }


#: Fields no source in the inventory can support. Zero here is an unresolved
#: placeholder, and every mapped player carries this list so a downstream run
#: can say which parts of its answer rest on data.
#:
#: ``signal_noise_sd`` was on this list and is not any more: it is now set
#: explicitly from the ``signal_quality`` scenario. It remains uncalibrated,
#: but it is no longer *implicit*, which is a different failure.
UNRESOLVED_PLACEHOLDER_FIELDS: Tuple[str, ...] = (
    "spike_rate", "spike_scale", "role_change_prob", "role_change_mean",
    "role_change_sd", "shock_loadings", "contingency", "proj_noise_sd",
    "weekly_projection_override",
)

_POS = {"QB": Position.QB, "RB": Position.RB, "WR": Position.WR,
        "TE": Position.TE}


def positional_fits_from_contract(payload: Dict
                                  ) -> Tuple[Dict[str, float], Dict[str, float]]:
    """``(weekly_cv, weekly_miss_rate)`` per position, read back off the contract.

    Ingestion already attaches each player's cohort fit to his row, and the
    cohort is positional, so the positional fallbacks used for players with no
    individual profile can be recovered from the contract itself rather than
    from a second local file. That makes a sensitivity run reproducible from
    the committed contract alone.

    A position whose rows disagree is reported by raising: silently averaging
    two different fits would hide a contract that had been assembled from
    mismatched sources.
    """
    cv: Dict[str, float] = {}
    miss: Dict[str, float] = {}
    for row in payload.get("players", []):
        pos = str(row.get("position") or "")
        disp = row.get("cohort_dispersion") or {}
        for store, key in ((cv, "weekly_cv"), (miss, "weekly_miss_rate")):
            val = disp.get(key)
            if val is None:
                continue
            val = round(float(val), 6)
            prior = store.get(pos)
            if prior is not None and abs(prior - val) > 1e-6:
                raise ValueError(
                    f"contract carries two different {key} values for {pos}: "
                    f"{prior} and {val}; the cohort fit is supposed to be "
                    f"positional")
            store[pos] = val
    return cv, miss


def map_contract_to_playerspecs(
    payload: Dict,
    config: PlayerSpecMappingConfig,
    positional_miss: Optional[Dict[str, float]] = None,
    positional_cv: Optional[Dict[str, float]] = None,
    limit: Optional[int] = None,
    only_keys: Optional[Iterable[str]] = None,
) -> MappingResult:
    """Turn a validated contract into ``PlayerSpec`` objects.

    ``limit`` maps only the top *n* players by recomputed season points, which
    is what a draftable-pool analysis wants: the deep tail is not a decision
    anyone makes and calibrating it costs real time.

    ``only_keys`` restricts the mapping to a fixed set of canonical keys and is
    what keeps a sensitivity sweep honest: every scenario maps *the same
    people*, so a paired comparison differs only in the assumption under test
    and not in who happens to have made the cut under this scenario's ordering.

    ``player_id`` and ``crn_key`` are derived from the canonical key, so a
    player keeps the same identity and the same random streams across every
    fumble interpretation, projection interpretation, ranking, pool limit and
    scenario. Collisions raise rather than silently sharing a stream.
    """
    positional_miss = positional_miss or {}
    positional_cv = positional_cv or {}
    warnings: List[str] = []
    skipped: List[str] = []
    wanted: Optional[Set[str]] = (
        {canonical_player_key(k) for k in only_keys} if only_keys is not None
        else None)

    rows = list(payload.get("players", []))
    rows.sort(key=lambda p: -((p.get("season_points") or {}).get("points") or 0.0))

    keyed: List[Tuple[str, Dict]] = []
    for row in rows:
        try:
            ckey = canonical_player_key(row.get("player_key"), row.get("name"))
        except ValueError as exc:
            skipped.append(f"{row.get('player_key')}: {exc}")
            continue
        if wanted is not None and ckey not in wanted:
            continue
        keyed.append((ckey, row))
    if wanted is None and limit is not None:
        keyed = keyed[:limit]

    # One id per canonical key, checked for collisions before anything is
    # simulated. `assign_stable_ids` raises rather than letting two players
    # share a random stream.
    ids = assign_stable_ids(ckey for ckey, _ in keyed)

    out: List[MappedPlayer] = []
    for ckey, row in keyed:
        pos_name = str(row.get("position") or "")
        if pos_name not in _POS:
            skipped.append(f"{ckey}: position {pos_name!r}")
            continue
        points = (row.get("season_points") or {}).get("points")
        if points is None:
            skipped.append(f"{ckey}: no recomputed points")
            continue
        if float(points) <= 0.0:
            # A non-positive season total would produce a non-positive level
            # and therefore a negative standard deviation. Skipping is honest;
            # clamping would invent a player.
            skipped.append(
                f"{ckey}: season points {float(points):.2f} is not positive")
            continue

        cv = (row.get("cohort_dispersion") or {}).get("weekly_cv")
        if cv is None:
            cv = positional_cv.get(pos_name)
        if cv is None:
            skipped.append(f"{ckey}: no dispersion for {pos_name}")
            continue

        bye = row.get("bye_week")
        bye_index = int(bye) - 1 if bye else 8

        # Availability first: an availability-adjusted level solve consumes the
        # fitted process, and the full-health solve ignores it. Solving in this
        # order means neither reading needs a second pass.
        avail = row.get("availability") or {}
        injury = calibrate_injury(
            avail.get("injury_prob"), avail.get("proj_games_missed"), config,
            positional_miss_rate=positional_miss.get(pos_name),
            position=pos_name, bye_index=bye_index)
        if injury.note and not injury.feasible:
            warnings.append(f"{ckey}: {injury.note}")

        # The level solve needs dispersion expressed relative to the level, so
        # that both move together as the solve searches.
        level_cal = calibrate_level(
            float(points), week_sd_fraction=float(cv),
            season_sd_fraction=config.season_sd_fraction, config=config,
            injury=injury, bye_index=bye_index)
        active_mean = level_cal.level

        total_week_sd = float(cv) * active_mean
        f = config.forecastable_share
        week_sd = total_week_sd * math.sqrt(1.0 - f)
        weekly_state_sd = total_week_sd * math.sqrt(f)
        season_sd = config.season_sd_fraction * active_mean
        signal_noise_sd = resolve_signal_noise_sd(config.signal_quality, week_sd)

        pid = ids[ckey]
        spec = PlayerSpec(
            player_id=pid,
            name=str(row.get("name")),
            position=_POS[pos_name],
            nfl_team=str(row.get("nfl_team") or "UNK"),
            base_mean=active_mean,
            week_sd=week_sd,
            season_sd=season_sd,
            bye_week=int(bye) if bye else 0,
            weekly_injury_hazard=injury.weekly_injury_hazard,
            injury_mean_weeks=max(injury.injury_mean_weeks, 1e-6),
            weekly_state_sd=weekly_state_sd,
            # Explicit, never None: see PlayerSpecMappingConfig.signal_quality.
            signal_noise_sd=signal_noise_sd,
            # Everything below is an unresolved placeholder, not an estimate.
            spike_rate=0.0, spike_scale=0.0,
            role_change_prob=0.0, proj_noise_sd=0.0,
            shock_loadings=(), contingency=None,
            data_source="REAL:winwithodds+draftsharks+nflverse-fits",
            crn_key=pid,
            notes=(f"key={ckey}; level={config.target}; "
                   f"avail={config.projection_availability_interpretation}; "
                   f"f={f:.2f}; ssd={config.season_sd_fraction:.2f}; "
                   f"sig={config.signal_quality}; injury={injury.source}"),
        )
        out.append(MappedPlayer(
            spec=spec, canonical_key=ckey, level=level_cal, injury=injury,
            total_week_sd=total_week_sd, forecastable_share=f,
            season_sd_fraction=config.season_sd_fraction,
            signal_quality=config.signal_quality,
            signal_noise_sd=signal_noise_sd,
            unresolved_placeholders=UNRESOLVED_PLACEHOLDER_FIELDS))

    if wanted is not None:
        missing = sorted(wanted - {m.canonical_key for m in out})
        if missing:
            warnings.append(
                f"{len(missing)} requested player(s) could not be mapped under "
                f"this scenario; a paired comparison over a fixed roster needs "
                f"every one of them")

    n_fallback = sum(1 for m in out if m.injury.source == "positional_all_cause")
    if n_fallback:
        warnings.append(
            f"{n_fallback} of {len(out)} players have no individual injury "
            f"profile and fall back to the fitted positional ALL-CAUSE "
            f"availability rate, which counts benching, rest and trades as well "
            f"as injury. It is not an injury-only hazard and must not be "
            f"reported as one.")
    n_unmodelled = sum(1 for m in out if m.injury.source == "unmodelled")
    if n_unmodelled:
        warnings.append(
            f"{n_unmodelled} of {len(out)} players have availability "
            f"UNMODELLED. That is not a claim of perfect health.")
    return MappingResult(players=out, config=config, warnings=warnings,
                         skipped=skipped)
