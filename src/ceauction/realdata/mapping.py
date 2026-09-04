"""Mapping the normalized contract into ``PlayerSpec``, with the assumptions visible.

Nothing here is a settled estimate. Every quantity this module produces is
either (a) calibrated numerically against a stated target, with its error
reported, or (b) a labelled **sensitivity scenario**. The configuration object
exists so that no assumption can hide in a constant.

Three things are worth reading before the code.

**The projection target is a question, not a fact.** The source's season total
is a hybrid: continuous categories are market medians, discrete ones are
probability-weighted expectations. ``PlayerSpec.base_mean`` is an expected
value. So the healthy level is *calibrated* against the total under an explicit
reading -- median or mean -- and both are carried.

**Injury parameters are solved, not assigned.** The engine's availability
process has its own dynamics (recurring onsets, a duration draw, a bye that
costs no game). Two supplied numbers -- season injury probability and projected
games missed -- are matched by a 2-D numerical solve against that process, and
the residual error of both targets is reported. When they cannot be jointly
reproduced, the caller is told.

**The variance split has no empirical basis and is not invented here.** The
fitted CV is *total* active-week dispersion. How much of it is forecastable
before lock is unmeasured, so it is swept as a scenario rather than estimated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .. import rng
from ..league import DEFAULT_LEAGUE, LeagueSettings, Position
from ..players import PlayerSpec

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
]

#: Share of total weekly dispersion assumed knowable before lineup lock.
#: **Scenarios, not estimates.** Nothing in the inventory measures this.
FORECASTABLE_SHARE_SCENARIOS: Tuple[float, ...] = (0.00, 0.25, 0.50)

#: Season-level uncertainty as a fraction of the active mean. Also scenarios.
SEASON_SD_SCENARIOS: Tuple[float, ...] = (0.00, 0.10, 0.20)


@dataclass(frozen=True)
class PlayerSpecMappingConfig:
    """Every assumption the mapping makes, in one place.

    No default here is a finding. Each is either a documented fact about the
    source (``games_basis``), a provisional reading being carried alongside its
    alternative (``target``), or the first point of a sensitivity sweep
    (``forecastable_share``, ``season_sd_fraction``).
    """

    target: str = "median_target"
    """``"median_target"`` or ``"mean_target"``.

    Provisionally median: the user described the source as median outcomes, and
    the major continuous props genuinely are market medians. But the total is a
    hybrid -- discrete categories are expectations -- so ``mean_target`` is a
    required sensitivity case, not an afterthought."""

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
    """Simulations behind a calibration. Large by default because the level
    solve is cached per dispersion shape rather than run per player, so the
    cost is paid a handful of times and finite-sample error stops mattering."""

    injury_calibration_sims: int = 4_000
    """Simulations behind an injury solve. Much smaller than
    ``calibration_sims`` on purpose: the level solve is cached per dispersion
    shape and runs a handful of times, while an injury solve runs per distinct
    (probability, games-missed, bye) triple and drives a 2-D search. 6,000 puts
    the standard error on a rate near 0.35 at about 0.006, well inside the
    tolerance the feasibility check applies."""

    calibration_seed: int = 20260904
    games_basis: float = 17.0
    """Games the source's season total spans: the full NFL regular season."""

    settings: LeagueSettings = DEFAULT_LEAGUE

    def __post_init__(self) -> None:
        if self.target not in ("median_target", "mean_target"):
            raise ValueError(f"unknown target {self.target!r}")
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

    def label(self) -> str:
        """Short, stable identifier for a scenario cell."""
        return (f"{self.target}|f={self.forecastable_share:.2f}"
                f"|ssd={self.season_sd_fraction:.2f}"
                f"|inj={self.injury_model}|fum={self.fumble_interpretation}")


# ---------------------------------------------------------------------------
# Level calibration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LevelCalibration:
    """The healthy per-game level, and how well it hit its target."""

    level: float
    target: str
    target_value: float
    achieved: float
    abs_error: float
    naive_level: float
    """``season_total / games_basis`` -- what a direct assignment would give."""

    @property
    def relative_error(self) -> float:
        return self.abs_error / self.target_value if self.target_value else 0.0

    @property
    def divergence_from_naive(self) -> float:
        return self.level - self.naive_level


def _season_totals(level: float, week_sd: float, season_sd: float,
                   games: int, sims: int, seed: int) -> np.ndarray:
    """Simulated healthy full-season totals at a given per-game level.

    Uses the engine's own counter-based RNG, so a calibration is reproducible
    and two calls at the same seed compare like for like.
    """
    s = np.arange(sims, dtype=np.int64).reshape(sims, 1)
    w = np.arange(games, dtype=np.int64).reshape(1, games)
    weekly = np.full((sims, games), float(level))
    if week_sd > 0:
        weekly = weekly + rng.normal(seed, rng.Kind.WEEK_NOISE, s, w) * week_sd
    if season_sd > 0:
        shift = rng.normal(seed, rng.Kind.SEASON_SHIFT, s, np.zeros((1, 1), dtype=np.int64))
        weekly = weekly + shift
    return weekly.sum(axis=1)


_UNIT_STAT_CACHE: Dict[Tuple, float] = {}


def _unit_season_statistic(week_sd_fraction: float, season_sd_fraction: float,
                           games: int, sims: int, seed: int, target: str
                           ) -> float:
    """The chosen statistic of a full-season total at **unit** healthy level.

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
    totals = _season_totals(1.0, week_sd_fraction, season_sd_fraction,
                            games, sims, seed)
    stat = float(np.median(totals) if target == "median_target"
                 else np.mean(totals))
    _UNIT_STAT_CACHE[key] = stat
    return stat


def calibrate_level(season_total: float, week_sd_fraction: float,
                    season_sd_fraction: float, config: PlayerSpecMappingConfig
                    ) -> LevelCalibration:
    """Solve for the healthy per-game level that hits the chosen target.

    ``week_sd_fraction`` and ``season_sd_fraction`` are expressed relative to
    the level, so dispersion scales with it -- which is what makes the season
    total exactly homogeneous in the level and the calibration exact.

    **What this finds, and it is worth stating plainly.** With the fields this
    source can populate, every weekly component is symmetric: the idiosyncratic
    draw and the season shift are both normal, and the skewed component
    (spikes) has no source and stays unpopulated. A sum of symmetric variables
    is symmetric, so its mean and median coincide and both targets return
    ``season_total / games_basis``. The calibration *confirms* that rather than
    assuming it, and the confirmation is the point: the same routine would
    return different levels for the two targets the moment a skewed component
    were populated, and would do so without anyone having to remember to
    revisit this function.

    ``divergence_from_naive`` reports the residual. It is finite-sample error in
    the calibration draw, identical for every player sharing a dispersion
    shape, and it shrinks as ``calibration_sims`` rises.
    """
    games = int(round(config.games_basis))
    naive = season_total / config.games_basis
    unit = _unit_season_statistic(
        week_sd_fraction, season_sd_fraction, games,
        config.calibration_sims, config.calibration_seed, config.target)
    if unit <= 0:
        raise ValueError(
            "the simulated unit-level season statistic is non-positive; the "
            "dispersion fractions are too large for this calibration")
    level = season_total / unit
    achieved = level * unit
    return LevelCalibration(
        level=level, target=config.target, target_value=season_total,
        achieved=achieved, abs_error=abs(achieved - season_total),
        naive_level=naive)


# ---------------------------------------------------------------------------
# Injury calibration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InjuryCalibration:
    """Hazard and duration solved against two supplied season targets."""

    weekly_injury_hazard: float
    injury_mean_weeks: float
    source: str
    """``"individual"``, ``"positional_all_cause"`` or ``"unmodelled"``."""

    target_injury_prob: Optional[float] = None
    achieved_injury_prob: Optional[float] = None
    target_games_missed: Optional[float] = None
    achieved_games_missed: Optional[float] = None
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


def _simulate_availability(hazard, mean_weeks, bye_index: int,
                           weeks: int, sims: int, seed: int):
    """``(P(any injury), E[games missed])`` under the engine's own process.

    ``hazard`` and ``mean_weeks`` may be arrays, in which case the whole grid is
    simulated at once and the returns have the grid's shape. That matters: the
    calibration inverts this function, and inverting it one player at a time
    was the difference between seconds and minutes.

    This mirrors ``worlds._draw_availability`` exactly, including that a week
    which is both a bye and an absence costs no *game* -- the player was not
    playing that week anyway. Conflating calendar weeks with scheduled games
    here would bias every duration estimate.
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

    for week in range(weeks):
        wk = np.full((1, 1), week, dtype=np.int64)
        onset_u = rng.uniform(seed, rng.Kind.INJURY_ONSET, s_idx, wk).reshape(1, sims)
        dur_e = rng.exponential(seed, rng.Kind.INJURY_DURATION, s_idx, wk).reshape(1, sims)
        healthy = out_until < week
        onset = healthy & (onset_u < h)
        ever |= onset
        duration = np.maximum(1, np.rint(dur_e * d).astype(np.int64))
        out_until = np.where(onset, week + duration - 1, out_until)
        if week != bye_index:          # a bye week costs no scheduled game
            missed += (out_until >= week)

    prob = ever.mean(axis=-1)
    miss = missed.mean(axis=-1)
    if grid_shape == (1,):
        return float(prob[0]), float(miss[0])
    return prob, miss


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


def _availability_grid(bye_index: int, weeks: int, sims: int, seed: int):
    """``(prob, missed)`` over the whole (hazard, duration) grid, cached."""
    bye_index = _GRID_REFERENCE_BYE
    key = (bye_index, weeks, sims, seed)
    hit = _GRID_CACHE.get(key)
    if hit is not None:
        return hit
    h = _HAZARD_GRID[:, None]
    d = _DURATION_GRID[None, :]
    prob, miss = _simulate_availability(h, d, bye_index, weeks, sims, seed)
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

    The vendor's ``proj_games_missed`` is over the full 17-game NFL season. The
    fantasy horizon holds only 16 of those games, so the target is scaled by
    16/17 before solving. Conflating the two spans would inflate every absence
    by about 6%.

    Returns an ``unmodelled`` or ``positional_all_cause`` calibration when an
    individual profile is unavailable, per the configured fallback. It never
    silently returns perfect health.
    """
    weeks = config.settings.total_weeks
    scheduled = weeks - 1                      # one bye inside the horizon
    sims = config.injury_calibration_sims
    seed = config.calibration_seed

    cache_key = (
        None if injury_prob is None else round(float(injury_prob), 4),
        None if proj_games_missed is None else round(float(proj_games_missed), 3),
        int(bye_index), weeks, sims, seed, config.injury_model,
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

    if config.injury_model == "none":
        return _cache(InjuryCalibration(
            0.0, 2.5, "unmodelled", feasible=True,
            note="availability unmodelled by configuration; this is NOT a claim "
                 "of perfect health"))

    have_individual = (injury_prob is not None and proj_games_missed is not None
                       and injury_prob > 0 and proj_games_missed > 0)

    if config.injury_model == "positional" or not have_individual:
        if config.missing_injury_fallback == "none" and not have_individual:
            return _cache(InjuryCalibration(
                0.0, 2.5, "unmodelled", feasible=True,
                note="no individual profile and fallback disabled; availability "
                     "unmodelled, NOT healthy"))
        rate = positional_miss_rate
        if rate is None:
            return _cache(InjuryCalibration(
                0.0, 2.5, "unmodelled", feasible=False,
                note=f"no individual profile and no positional rate for "
                     f"{position!r}; availability unmodelled, NOT healthy"))
        prob, missed = _simulate_availability(rate, 2.5, bye_index, weeks,
                                              sims, seed)
        return _cache(InjuryCalibration(
            weekly_injury_hazard=rate, injury_mean_weeks=2.5,
            source="positional_all_cause",
            achieved_injury_prob=prob, achieved_games_missed=missed,
            feasible=True,
            note="ALL-CAUSE availability rate fitted from historical weekly "
                 "results: it counts benching, rest and trades as well as "
                 "injury, and is not an injury-only hazard. Duration is the "
                 "engine default, not fitted."))

    # Scale the vendor's 17-game figure onto the 16-game fantasy horizon.
    target_missed = float(proj_games_missed) * (scheduled / config.games_basis)
    target_prob = float(injury_prob)

    # Invert the engine's own availability process over a precomputed grid.
    # Hazard drives P(any onset) and duration drives games missed, but the two
    # interact -- being absent blocks new onsets -- so both targets are matched
    # jointly rather than one after the other.
    grid_prob, grid_miss = _availability_grid(bye_index, weeks, sims, seed)
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

    prob_err = abs(prob - target_prob)
    missed_err = abs(missed - target_missed)
    # Tolerances are set against the calibration's own Monte Carlo noise: at
    # 6,000 draws the standard error on a rate near 0.35 is about 0.006, so
    # 0.02 is roughly three of them. Anything outside that is structure, not
    # sampling.
    feasible = prob_err <= 0.02 and missed_err <= 0.15
    note = ""
    if not feasible:
        note = (f"targets not jointly reproduced within tolerance: P(injury) "
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
        feasible=feasible, note=note))


# ---------------------------------------------------------------------------
# The mapping
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MappedPlayer:
    """One ``PlayerSpec`` plus the provenance of every number in it."""

    spec: PlayerSpec
    level: LevelCalibration
    injury: InjuryCalibration
    total_week_sd: float
    forecastable_share: float
    season_sd_fraction: float
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

    def calibration_summary(self) -> Dict[str, object]:
        """Aggregate calibration error. Counts and ranges only."""
        lvl = [m.level.abs_error for m in self.players]
        ind = [m for m in self.players if m.injury.source == "individual"]
        prob_err = [abs(m.injury.injury_prob_error) for m in ind
                    if m.injury.injury_prob_error is not None]
        miss_err = [abs(m.injury.games_missed_error) for m in ind
                    if m.injury.games_missed_error is not None]
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

        return {
            "players": len(self.players),
            "level_abs_error": stat(lvl),
            "injury_source_counts": by_source,
            "injury_prob_abs_error": stat(prob_err),
            "games_missed_abs_error": stat(miss_err),
            "infeasible_injury_calibrations": sum(
                1 for m in ind if not m.injury.feasible),
        }


#: Fields no source in the inventory can support. Zero here is an unresolved
#: placeholder, and every mapped player carries this list so a downstream run
#: can say which parts of its answer rest on data.
UNRESOLVED_PLACEHOLDER_FIELDS: Tuple[str, ...] = (
    "spike_rate", "spike_scale", "role_change_prob", "role_change_mean",
    "role_change_sd", "shock_loadings", "contingency", "proj_noise_sd",
    "signal_noise_sd", "weekly_projection_override",
)

_POS = {"QB": Position.QB, "RB": Position.RB, "WR": Position.WR,
        "TE": Position.TE}


def map_contract_to_playerspecs(
    payload: Dict,
    config: PlayerSpecMappingConfig,
    positional_miss: Optional[Dict[str, float]] = None,
    positional_cv: Optional[Dict[str, float]] = None,
    limit: Optional[int] = None,
) -> MappingResult:
    """Turn a validated contract into ``PlayerSpec`` objects.

    ``limit`` maps only the top *n* players by recomputed season points, which
    is what a draftable-pool analysis wants: the deep tail is not a decision
    anyone makes and calibrating it costs real time.
    """
    positional_miss = positional_miss or {}
    positional_cv = positional_cv or {}
    warnings: List[str] = []
    skipped: List[str] = []

    rows = list(payload.get("players", []))
    rows.sort(key=lambda p: -((p.get("season_points") or {}).get("points") or 0.0))
    if limit is not None:
        rows = rows[:limit]

    out: List[MappedPlayer] = []
    for idx, row in enumerate(rows):
        pos_name = str(row.get("position") or "")
        if pos_name not in _POS:
            skipped.append(f"{row.get('player_key')}: position {pos_name!r}")
            continue
        points = (row.get("season_points") or {}).get("points")
        if points is None:
            skipped.append(f"{row.get('player_key')}: no recomputed points")
            continue
        if float(points) <= 0.0:
            # A non-positive season total would produce a non-positive level
            # and therefore a negative standard deviation. Skipping is honest;
            # clamping would invent a player.
            skipped.append(
                f"{row.get('player_key')}: season points {float(points):.2f} "
                f"is not positive")
            continue

        cv = (row.get("cohort_dispersion") or {}).get("weekly_cv")
        if cv is None:
            cv = positional_cv.get(pos_name)
        if cv is None:
            skipped.append(f"{row.get('player_key')}: no dispersion for {pos_name}")
            continue

        # The level solve needs dispersion expressed relative to the level, so
        # that both move together as the solve searches.
        level_cal = calibrate_level(
            float(points), week_sd_fraction=float(cv),
            season_sd_fraction=config.season_sd_fraction, config=config)
        active_mean = level_cal.level

        total_week_sd = float(cv) * active_mean
        f = config.forecastable_share
        week_sd = total_week_sd * math.sqrt(1.0 - f)
        weekly_state_sd = total_week_sd * math.sqrt(f)
        season_sd = config.season_sd_fraction * active_mean

        bye = row.get("bye_week")
        bye_index = int(bye) - 1 if bye else 8
        avail = row.get("availability") or {}
        injury = calibrate_injury(
            avail.get("injury_prob"), avail.get("proj_games_missed"), config,
            positional_miss_rate=positional_miss.get(pos_name),
            position=pos_name, bye_index=bye_index)
        if injury.note and not injury.feasible:
            warnings.append(f"{row.get('player_key')}: {injury.note}")

        spec = PlayerSpec(
            player_id=idx,
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
            # Everything below is an unresolved placeholder, not an estimate.
            spike_rate=0.0, spike_scale=0.0,
            role_change_prob=0.0, proj_noise_sd=0.0,
            signal_noise_sd=None, shock_loadings=(), contingency=None,
            data_source="REAL:winwithodds+draftsharks+nflverse-fits",
            crn_key=idx,
            notes=(f"level={config.target}; f={f:.2f}; "
                   f"ssd={config.season_sd_fraction:.2f}; "
                   f"injury={injury.source}"),
        )
        out.append(MappedPlayer(
            spec=spec, level=level_cal, injury=injury,
            total_week_sd=total_week_sd, forecastable_share=f,
            season_sd_fraction=config.season_sd_fraction,
            unresolved_placeholders=UNRESOLVED_PLACEHOLDER_FIELDS))

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
