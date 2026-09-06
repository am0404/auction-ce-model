"""The model-scenario cross-product, and paired CE bands over it.

Four assumptions in this model are unresolved, and the calibration audit
measured each of them one axis at a time. That answers "does this assumption
move CE?" but not "how wide is the range of answers the model can produce at
all?", which is the question a valuation has to carry. This module enumerates
the full cross-product of the four and runs an arbitrary paired roster
comparison in every cell.

    availability_interpretation   2   what health state the season total describes
    forecastable_share            3   share of weekly variance knowable pre-lock
    season_sd                     3   persistent season-level uncertainty
    signal_quality                3   how fast a manager learns a latent change
                                 --
                                 54

**No cell is correct.** The grid deliberately has no centre and no default
weighting. A point estimate may be quoted only alongside the name of the
scenario it came from, and the honest summary of a comparison is the band
across all 54 cells widened by each cell's own Monte Carlo interval.

Two properties make the band meaningful rather than decorative.

**The cast is fixed.** Both arms of a comparison hold the same twelve teams
made of the same people; only the change under test differs. Player ids derive
from the canonical player key, so a player untouched by the comparison draws
bit-identical random numbers in both arms of every cell.

**Every cell is paired.** Within a cell the two arms are simulated over the
same seasons at the same seed, and the reported standard error comes from the
per-season difference. Across cells the specs genuinely differ, so cells are
not paired with each other and are never differenced.
"""

from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from ..league import DEFAULT_LEAGUE, LeagueSettings
from ..players import PlayerSpec
from ..simulate import SeasonOutcomes, simulate_seasons
from .mapping import (
    AVAILABILITY_INTERPRETATIONS,
    FORECASTABLE_SHARE_SCENARIOS,
    SEASON_SD_SCENARIOS,
    SIGNAL_QUALITY_SCENARIOS,
    PlayerSpecMappingConfig,
    map_contract_to_playerspecs,
)
from .sensitivity import MIN_COMMITTED_SIMS, paired_team_deltas
from .smoke import rosters_from_assignment

__all__ = [
    "ModelScenario",
    "enumerate_scenarios",
    "SCENARIO_GRID",
    "GRID_SIZE",
    "BASELINE_SCENARIO",
    "ArmSpec",
    "Pairing",
    "ScenarioCell",
    "ScenarioBand",
    "run_scenario_band",
    "run_scenario_arms",
    "format_band",
]

#: Short codes so a scenario id stays readable in a filename or a table cell.
_AVAIL_CODE = {"full_health": "fh", "availability_adjusted": "aa"}
_AVAIL_FROM_CODE = {v: k for k, v in _AVAIL_CODE.items()}
_SIGNAL_CODE = {"none": "n0", "week_sd": "w1", "2x_week_sd": "w2"}
_SIGNAL_FROM_CODE = {v: k for k, v in _SIGNAL_CODE.items()}


@dataclass(frozen=True)
class ModelScenario:
    """One cell of the cross-product: a complete set of the four assumptions.

    This is a *scenario*, not an estimate. Constructing one commits to nothing;
    reporting a number from one without naming it does.
    """

    availability_interpretation: str
    forecastable_share: float
    season_sd_fraction: float
    signal_quality: str

    def __post_init__(self) -> None:
        if self.availability_interpretation not in AVAILABILITY_INTERPRETATIONS:
            raise ValueError(
                f"unknown availability_interpretation "
                f"{self.availability_interpretation!r}")
        if self.signal_quality not in SIGNAL_QUALITY_SCENARIOS:
            raise ValueError(f"unknown signal_quality {self.signal_quality!r}")
        if not 0.0 <= self.forecastable_share < 1.0:
            raise ValueError("forecastable_share must be in [0, 1)")
        if self.season_sd_fraction < 0.0:
            raise ValueError("season_sd_fraction must be non-negative")

    @property
    def scenario_id(self) -> str:
        """Stable, sortable, filename-safe identifier.

        Round-trips through :meth:`from_id`. Kept stable on purpose: it is the
        cache key that stops one scenario's CE being served for another.
        """
        return (f"{_AVAIL_CODE[self.availability_interpretation]}"
                f"-f{int(round(self.forecastable_share * 100)):03d}"
                f"-s{int(round(self.season_sd_fraction * 100)):03d}"
                f"-{_SIGNAL_CODE[self.signal_quality]}")

    @classmethod
    def from_id(cls, scenario_id: str) -> "ModelScenario":
        try:
            avail, fshare, ssd, sig = scenario_id.split("-")
            return cls(
                availability_interpretation=_AVAIL_FROM_CODE[avail],
                forecastable_share=int(fshare[1:]) / 100.0,
                season_sd_fraction=int(ssd[1:]) / 100.0,
                signal_quality=_SIGNAL_FROM_CODE[sig])
        except (ValueError, KeyError) as exc:
            raise ValueError(f"not a scenario id: {scenario_id!r}") from exc

    def describe(self) -> str:
        return (f"availability={self.availability_interpretation}, "
                f"forecastable_share={self.forecastable_share:.2f}, "
                f"season_sd={self.season_sd_fraction:.2f}, "
                f"signal={self.signal_quality}")

    def to_dict(self) -> Dict[str, object]:
        return {
            "scenario_id": self.scenario_id,
            "availability_interpretation": self.availability_interpretation,
            "forecastable_share": self.forecastable_share,
            "season_sd_fraction": self.season_sd_fraction,
            "signal_quality": self.signal_quality,
        }

    def to_mapping_config(self, **overrides) -> PlayerSpecMappingConfig:
        """The calibration config this scenario implies.

        ``overrides`` passes through anything the scenario does not govern --
        ``target``, ``fumble_interpretation``, ``calibration_sims`` and so on.
        Passing one of the four axes here is refused rather than silently
        winning, because a config that disagreed with its own scenario id would
        poison every cache keyed on that id.
        """
        governed = {"projection_availability_interpretation", "forecastable_share",
                    "season_sd_fraction", "signal_quality"}
        clash = governed & set(overrides)
        if clash:
            raise ValueError(
                f"{sorted(clash)} is governed by the scenario and cannot be "
                f"overridden; build a different ModelScenario instead")
        return PlayerSpecMappingConfig(
            projection_availability_interpretation=self.availability_interpretation,
            forecastable_share=self.forecastable_share,
            season_sd_fraction=self.season_sd_fraction,
            signal_quality=self.signal_quality,
            **overrides)


def enumerate_scenarios(
    availability: Sequence[str] = AVAILABILITY_INTERPRETATIONS,
    forecastable: Sequence[float] = FORECASTABLE_SHARE_SCENARIOS,
    season_sd: Sequence[float] = SEASON_SD_SCENARIOS,
    signal: Sequence[str] = SIGNAL_QUALITY_SCENARIOS,
) -> Tuple[ModelScenario, ...]:
    """Every cell, in a deterministic order that never depends on a set.

    The nesting order is fixed (availability outermost, signal innermost) so
    that a cell's position in the sequence is reproducible across runs and
    processes -- which matters when a long run is resumed or compared.
    """
    return tuple(
        ModelScenario(a, float(f), float(s), g)
        for a, f, s, g in itertools.product(availability, forecastable,
                                            season_sd, signal))


#: The committed 54-cell grid.
SCENARIO_GRID: Tuple[ModelScenario, ...] = enumerate_scenarios()
GRID_SIZE: int = len(SCENARIO_GRID)

#: The cell whose four settings match ``PlayerSpecMappingConfig``'s defaults.
#: It is the reference used to *choose* a fixed cast, and nothing else -- it is
#: not a preferred scenario and carries no more evidence than any other cell.
BASELINE_SCENARIO = ModelScenario("full_health", 0.0, 0.0, "week_sd")


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArmSpec:
    """One side of a comparison: a name and a fixed 12x15 roster assignment.

    An arm is a *roster assignment by player id*, not a set of specs. The specs
    are re-derived per scenario; the people and the teams they are on stay
    identical, which is what makes a cell's difference attributable to the
    change under test rather than to a reshuffle.
    """

    name: str
    assignment: Tuple[Tuple[int, ...], ...]

    def __post_init__(self) -> None:
        flat = [pid for team in self.assignment for pid in team]
        if len(set(flat)) != len(flat):
            raise ValueError(f"arm {self.name!r} gives one player to two teams")

    @property
    def player_ids(self) -> Tuple[int, ...]:
        return tuple(pid for team in self.assignment for pid in team)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScenarioCell:
    """One scenario's paired result for one arm pair."""

    scenario: ModelScenario
    arm_a: str
    arm_b: str
    team_index: int
    team_name: str
    ce_a: float
    ce_b: float
    delta_ce: float
    delta_ce_se: float
    discordance: float
    n_sims: int
    runtime_s: float

    @property
    def scenario_id(self) -> str:
        return self.scenario.scenario_id

    @property
    def ci95(self) -> Tuple[float, float]:
        if math.isnan(self.delta_ce_se):
            return (float("nan"), float("nan"))
        half = 1.96 * self.delta_ce_se
        return (self.delta_ce - half, self.delta_ce + half)

    @property
    def resolved(self) -> bool:
        lo, hi = self.ci95
        if math.isnan(lo):
            return False
        return lo > 0.0 or hi < 0.0

    def to_dict(self) -> Dict[str, object]:
        lo, hi = self.ci95
        return {
            "scenario_id": self.scenario_id,
            **self.scenario.to_dict(),
            "arm_a": self.arm_a, "arm_b": self.arm_b,
            "team_index": self.team_index, "team_name": self.team_name,
            "ce_a": round(self.ce_a, 6), "ce_b": round(self.ce_b, 6),
            "delta_ce": round(self.delta_ce, 6),
            "delta_ce_se": round(self.delta_ce_se, 6),
            "ci95": [round(lo, 6), round(hi, 6)],
            "discordance": round(self.discordance, 5),
            "resolved": self.resolved,
            "n_sims": self.n_sims,
            "runtime_s": round(self.runtime_s, 2),
        }


@dataclass
class ScenarioBand:
    """The whole cross-product for one arm pair, and what it does and does not say."""

    label: str
    arm_a: str
    arm_b: str
    cells: Tuple[ScenarioCell, ...]
    n_sims: int
    seed: int
    total_runtime_s: float
    grid_size: int = GRID_SIZE
    notes: str = ""

    # --- point-estimate band ------------------------------------------------

    @property
    def min_delta(self) -> float:
        return min(c.delta_ce for c in self.cells)

    @property
    def max_delta(self) -> float:
        return max(c.delta_ce for c in self.cells)

    @property
    def band_width(self) -> float:
        """Spread of the point estimates alone, ignoring sampling error."""
        return self.max_delta - self.min_delta

    @property
    def argmin_cell(self) -> ScenarioCell:
        return min(self.cells, key=lambda c: (c.delta_ce, c.scenario_id))

    @property
    def argmax_cell(self) -> ScenarioCell:
        return max(self.cells, key=lambda c: (c.delta_ce, c.scenario_id))

    # --- band widened by Monte Carlo uncertainty ----------------------------

    @property
    def mc_low(self) -> float:
        """Lowest value any cell's own 95% interval reaches."""
        return min(c.ci95[0] for c in self.cells)

    @property
    def mc_high(self) -> float:
        return max(c.ci95[1] for c in self.cells)

    @property
    def mc_band_width(self) -> float:
        return self.mc_high - self.mc_low

    @property
    def mc_inflation(self) -> float:
        """How much of the reported width is sampling rather than assumptions.

        1.0 means Monte Carlo added nothing; a large value means the run is too
        small to see the assumption band at all.
        """
        return (self.mc_band_width / self.band_width) if self.band_width > 0 else float("inf")

    # --- what the band supports --------------------------------------------

    @property
    def sign_stable(self) -> bool:
        """Do all 54 point estimates share a sign (or sit exactly at zero)?"""
        signs = {(c.delta_ce > 0) - (c.delta_ce < 0) for c in self.cells}
        return len(signs - {0}) <= 1

    @property
    def all_resolved(self) -> bool:
        return all(c.resolved for c in self.cells)

    @property
    def resolved_cells(self) -> Tuple[ScenarioCell, ...]:
        return tuple(c for c in self.cells if c.resolved)

    @property
    def rank_stability(self) -> str:
        """One of ``"strong"``, ``"weak"``, ``"none"``.

        ``strong``  every cell resolves and they all agree on the sign.
        ``weak``    the point estimates agree on a sign but not every cell
                    resolves, so the ordering is suggested rather than shown.
        ``none``    cells disagree on the sign: the model's own assumption
                    range contains both answers.
        """
        if not self.sign_stable:
            return "none"
        if self.all_resolved and self.mc_low > 0.0:
            return "strong"
        if self.all_resolved and self.mc_high < 0.0:
            return "strong"
        return "weak"

    @property
    def dominant_axis(self) -> Optional[str]:
        """Which of the four assumptions spreads the band most.

        Computed as the largest range of within-axis mean delta. It names the
        assumption worth resolving first; it is not a claim that the other three
        do not matter.
        """
        best, best_range = None, -1.0
        for axis in ("availability_interpretation", "forecastable_share",
                     "season_sd_fraction", "signal_quality"):
            by_value: Dict[object, List[float]] = {}
            for c in self.cells:
                by_value.setdefault(getattr(c.scenario, axis), []).append(c.delta_ce)
            if len(by_value) < 2:
                continue
            means = [sum(v) / len(v) for v in by_value.values()]
            spread = max(means) - min(means)
            if spread > best_range:
                best, best_range = axis, spread
        return best

    def by_scenario(self) -> Dict[str, ScenarioCell]:
        return {c.scenario_id: c for c in self.cells}

    def summary(self) -> Dict[str, object]:
        return {
            "label": self.label,
            "arm_a": self.arm_a, "arm_b": self.arm_b,
            "grid_size": self.grid_size,
            "cells_run": len(self.cells),
            "full_grid": len(self.cells) == self.grid_size,
            "n_sims_per_arm": self.n_sims,
            "seed": self.seed,
            "min_delta_ce": round(self.min_delta, 6),
            "max_delta_ce": round(self.max_delta, 6),
            "band_width": round(self.band_width, 6),
            "mc_low": round(self.mc_low, 6),
            "mc_high": round(self.mc_high, 6),
            "mc_band_width": round(self.mc_band_width, 6),
            "mc_inflation": round(self.mc_inflation, 3),
            "argmin_scenario": self.argmin_cell.scenario_id,
            "argmax_scenario": self.argmax_cell.scenario_id,
            "sign_stable": self.sign_stable,
            "cells_resolved": len(self.resolved_cells),
            "rank_stability": self.rank_stability,
            "dominant_axis": self.dominant_axis,
            "total_runtime_s": round(self.total_runtime_s, 1),
            "notes": self.notes,
            "cells": [c.to_dict() for c in self.cells],
        }


# ---------------------------------------------------------------------------
# Running a band
# ---------------------------------------------------------------------------


def _map_for_scenario(payload: Dict, scenario: ModelScenario,
                      only_keys: Sequence[str],
                      positional_miss: Dict[str, float],
                      positional_cv: Dict[str, float],
                      config_overrides: Dict) -> List[PlayerSpec]:
    cfg = scenario.to_mapping_config(**config_overrides)
    mapped = map_contract_to_playerspecs(
        payload, cfg, positional_miss=positional_miss,
        positional_cv=positional_cv, only_keys=only_keys)
    got = {m.canonical_key for m in mapped.players}
    missing = sorted(set(only_keys) - got)
    if missing:
        raise ValueError(
            f"scenario {scenario.scenario_id} could not map "
            f"{len(missing)} of the {len(only_keys)} players the arms need; a "
            f"band needs the same people in every cell")
    return mapped.specs


@dataclass(frozen=True)
class Pairing:
    """A comparison to extract from a set of arms simulated in the same cell."""

    label: str
    arm_a: str
    arm_b: str
    notes: str = ""


def run_scenario_arms(
    payload: Dict,
    arms: Sequence[ArmSpec],
    pairings: Sequence[Pairing],
    only_keys: Sequence[str],
    team_index: int,
    *,
    positional_miss: Optional[Dict[str, float]] = None,
    positional_cv: Optional[Dict[str, float]] = None,
    scenarios: Sequence[ModelScenario] = SCENARIO_GRID,
    n_sims: int = MIN_COMMITTED_SIMS,
    seed: int = 20260904,
    settings: LeagueSettings = DEFAULT_LEAGUE,
    team_names: Optional[Sequence[str]] = None,
    config_overrides: Optional[Dict] = None,
    progress=None,
) -> Dict[str, ScenarioBand]:
    """Simulate several arms per scenario and return one band per pairing.

    Every arm in a cell is mapped from the *same* spec list and simulated at the
    *same* seed, so any two of them are properly paired and a player common to
    both contributes an exact zero to their difference. Extracting several
    comparisons from one set of arms is therefore free of any pairing
    compromise, and it avoids re-running a shared arm -- the baseline is
    simulated once no matter how many comparisons reference it.

    ``only_keys`` must cover every player any arm rosters.

    ``progress`` is an optional callable ``(i, n, scenario)`` invoked after each
    cell, so a long run can report without this module knowing what a terminal
    is.
    """
    positional_miss = positional_miss or {}
    positional_cv = positional_cv or {}
    config_overrides = config_overrides or {}
    names = list(team_names) if team_names else [
        f"Real{i + 1:02d}" for i in range(settings.n_teams)]
    only_keys = tuple(only_keys)

    by_name = {a.name: a for a in arms}
    if len(by_name) != len(arms):
        raise ValueError("arm names must be unique within a run")
    for p in pairings:
        missing = {p.arm_a, p.arm_b} - set(by_name)
        if missing:
            raise ValueError(f"pairing {p.label!r} names unknown arm(s) {sorted(missing)}")

    cells: Dict[str, List[ScenarioCell]] = {p.label: [] for p in pairings}
    t_start = time.perf_counter()

    for i, scenario in enumerate(scenarios):
        t0 = time.perf_counter()
        specs = _map_for_scenario(payload, scenario, only_keys, positional_miss,
                                  positional_cv, config_overrides)
        outcomes: Dict[str, SeasonOutcomes] = {}
        for arm in arms:
            rs = rosters_from_assignment(arm.assignment, specs, settings, names)
            outcomes[arm.name] = simulate_seasons(rs, n_sims, seed)
        elapsed = time.perf_counter() - t0

        for p in pairings:
            deltas, _ = paired_team_deltas(outcomes[p.arm_a], outcomes[p.arm_b],
                                           names)
            d = deltas[team_index]
            cells[p.label].append(ScenarioCell(
                scenario=scenario, arm_a=p.arm_a, arm_b=p.arm_b,
                team_index=team_index, team_name=names[team_index],
                ce_a=d.ce_baseline, ce_b=d.ce_scenario,
                delta_ce=d.delta_ce, delta_ce_se=d.delta_ce_se,
                discordance=d.discordance, n_sims=n_sims,
                runtime_s=elapsed / max(len(pairings), 1)))
        if progress is not None:
            progress(i + 1, len(scenarios), scenario)

    total = time.perf_counter() - t_start
    return {
        p.label: ScenarioBand(
            label=p.label, arm_a=p.arm_a, arm_b=p.arm_b,
            cells=tuple(cells[p.label]), n_sims=n_sims, seed=seed,
            total_runtime_s=total, notes=p.notes)
        for p in pairings
    }


def run_scenario_band(
    payload: Dict,
    arm_a: ArmSpec,
    arm_b: ArmSpec,
    only_keys: Sequence[str],
    team_index: int,
    *,
    label: str = "scenario band",
    notes: str = "",
    **kwargs,
) -> ScenarioBand:
    """Convenience wrapper: one arm pair, one band."""
    bands = run_scenario_arms(
        payload, [arm_a, arm_b],
        [Pairing(label=label, arm_a=arm_a.name, arm_b=arm_b.name, notes=notes)],
        only_keys, team_index, **kwargs)
    return bands[label]


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def format_band(band: ScenarioBand, width: int = 96, show_cells: bool = True) -> str:
    """Sanitized rendering: scenario ids and CE deltas, no player rows."""
    bar = "=" * width
    out = [bar, f"SCENARIO BAND -- {band.label}", bar,
           f"{band.arm_b} minus {band.arm_a}, focus team "
           f"{band.cells[0].team_name if band.cells else '?'}",
           f"{len(band.cells)} of {band.grid_size} scenarios, "
           f"{band.n_sims:,} seasons per arm, seed {band.seed}",
           ""]
    if len(band.cells) != band.grid_size:
        out += [f"*** REDUCED GRID: {len(band.cells)} of {band.grid_size} cells "
                f"were run. This is NOT the full cross-product. ***", ""]
    if band.notes:
        out += [band.notes, ""]

    if show_cells:
        head = (f"  {'scenario':<20}{'dCE':>10}{'se':>9}"
                f"{'95% CI':>22}{'discord':>9}{'':>4}")
        out += ["PER-SCENARIO PAIRED DELTA CE", head, "  " + "-" * (len(head) - 2)]
        for c in sorted(band.cells, key=lambda c: c.delta_ce):
            lo, hi = c.ci95
            out.append(f"  {c.scenario_id:<20}{c.delta_ce:>+10.5f}"
                       f"{c.delta_ce_se:>9.5f}  [{lo:+.5f}, {hi:+.5f}]"
                       f"{c.discordance:>9.4f}{('**' if c.resolved else ''):>4}")
        out += ["  " + "-" * (head and len(head) - 2 or 0),
                "  ** = this cell's own 95% interval excludes zero.", ""]

    out += ["BAND",
            f"  point estimates          [{band.min_delta:+.5f}, "
            f"{band.max_delta:+.5f}]   width {band.band_width:.5f}",
            f"  widened by Monte Carlo   [{band.mc_low:+.5f}, "
            f"{band.mc_high:+.5f}]   width {band.mc_band_width:.5f}",
            f"  MC inflation             {band.mc_inflation:.2f}x"
            f"   (1.00 = sampling adds nothing)",
            f"  lowest  cell             {band.argmin_cell.scenario_id}"
            f"  ({band.argmin_cell.scenario.describe()})",
            f"  highest cell             {band.argmax_cell.scenario_id}"
            f"  ({band.argmax_cell.scenario.describe()})",
            f"  dominant assumption      {band.dominant_axis}",
            f"  cells resolved           {len(band.resolved_cells)} of {len(band.cells)}",
            f"  sign stable              {band.sign_stable}",
            f"  rank stability           {band.rank_stability.upper()}",
            f"  total runtime            {band.total_runtime_s:.1f}s",
            "",
            "No cell in this grid is the correct one. Quoting a point estimate",
            "requires naming its scenario. The band is the model's own range of",
            "answers, and it is not a confidence interval.",
            "",
            "RANK STABILITY HERE IS ABOUT ONE COMPARISON ONLY. A stable sign for",
            "this pair of players says nothing about whether every other pair on",
            "the board keeps its order; that would need the same test run over",
            "the pairs in question.",
            bar]
    return "\n".join(out)
