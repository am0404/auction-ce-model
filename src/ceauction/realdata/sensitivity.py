"""How much does a roster's championship equity depend on what we assumed?

Every axis swept here is an assumption the data does not settle. The point is
not to find a best cell -- there is no evidence for one -- but to measure how
far CE moves when an unresolved choice is made differently, **and whether that
movement is resolvable at all** at the sample size in hand.

Three design rules, each of which an earlier pass got wrong.

**One set of people, one set of teams.** The twelve integration rosters are
built once, from the baseline mapping, and every scenario re-simulates *those
exact players on those exact teams*. Player ids are derived from the canonical
player key, so a player keeps his identity and his random streams across every
scenario. Rebuilding the snake per scenario would have let the allocation move
whenever an assumption moved ``base_mean``, and the "effect of the assumption"
would silently include an effect of reshuffling the league.

**Every comparison is paired, season by season.** Baseline and scenario are
simulated over the same worlds at the same seed. The reported quantity is the
mean of a per-season difference, and its standard error comes from *that*
difference -- never from a separate experiment at a different sample size, and
never from one arm's marginal variance.

**An effect is real only if its own interval excludes zero.** Nothing here
borrows a standard error from anywhere else, and a shift larger than another
shift is not thereby significant.

Output is aggregate: per-scenario, per-team paired deltas for the twelve
integration rosters. It carries no player rows and no proprietary values.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..ce import paired_se
from ..roster import RosterSet
from ..simulate import SeasonOutcomes, simulate_seasons
from .mapping import (
    AVAILABILITY_INTERPRETATIONS,
    FORECASTABLE_SHARE_SCENARIOS,
    SEASON_SD_SCENARIOS,
    SIGNAL_QUALITY_SCENARIOS,
    PlayerSpecMappingConfig,
    map_contract_to_playerspecs,
)
from .smoke import build_test_rosters, roster_assignment, rosters_from_assignment

__all__ = [
    "TeamDelta",
    "Contrast",
    "ScenarioResult",
    "SensitivityGrid",
    "MIN_COMMITTED_SIMS",
    "run_sensitivity",
    "format_sensitivity",
]

#: The committed report may not be run below this. A paired CE standard error
#: at twelve teams is a few times 1e-3 here, and anything smaller than a few
#: thousand seasons cannot separate the axes at all -- which is precisely the
#: mistake of quoting a 2,000-season point estimate beside a 16,000-season
#: standard error.
MIN_COMMITTED_SIMS = 16_000


@dataclass(frozen=True)
class TeamDelta:
    """One team's paired difference between a scenario and the baseline.

    Every field is computed from the *same* seasons in both arms.
    """

    team_index: int
    team_name: str
    ce_baseline: float
    ce_scenario: float
    delta_ce: float
    delta_ce_se: float
    discordance: float
    """Fraction of seasons in which this team is champion in exactly one arm.

    The ceiling on ``|delta_ce|``: a scenario that never changes who wins
    cannot move CE at all, and one that changes the winner often but
    symmetrically moves CE very little while still being a large change to the
    world. Reporting it stops a near-zero delta being read as "the assumption
    does not matter"."""

    @property
    def ci95(self) -> Tuple[float, float]:
        if math.isnan(self.delta_ce_se):
            return (float("nan"), float("nan"))
        half = 1.96 * self.delta_ce_se
        return (self.delta_ce - half, self.delta_ce + half)

    @property
    def z(self) -> float:
        if self.delta_ce_se == 0.0 or math.isnan(self.delta_ce_se):
            return 0.0
        return self.delta_ce / self.delta_ce_se

    @property
    def resolved(self) -> bool:
        """Does this team's own 95% interval exclude zero?"""
        lo, hi = self.ci95
        if math.isnan(lo):
            return False
        return lo > 0.0 or hi < 0.0


@dataclass(frozen=True)
class Contrast:
    """A paired comparison between two *scenarios*, neither of them baseline.

    Every headline delta in this module is measured against the baseline, which
    answers "does this assumption move CE?" but not "does the answer to one
    question depend on the answer to another?". A season_sd effect measured at
    one learning speed and a season_sd effect measured at another are two
    numbers with overlapping intervals; whether they actually differ needs its
    own paired comparison, which is what this is.
    """

    name: str
    label_a: str
    label_b: str
    question: str
    team_deltas: Tuple[TeamDelta, ...]
    discordance: float

    @property
    def largest_delta(self) -> Optional[TeamDelta]:
        if not self.team_deltas:
            return None
        return max(self.team_deltas, key=lambda d: abs(d.delta_ce))

    @property
    def resolved_teams(self) -> Tuple[TeamDelta, ...]:
        return tuple(d for d in self.team_deltas if d.resolved)

    @property
    def any_resolved(self) -> bool:
        return bool(self.resolved_teams)

    @property
    def max_resolved_delta(self) -> float:
        return max((abs(d.delta_ce) for d in self.resolved_teams), default=0.0)


@dataclass(frozen=True)
class ScenarioResult:
    """One cell of the grid, with its paired comparison against the baseline."""

    label: str
    axis: str
    target: str
    availability_interpretation: str
    signal_quality: str
    forecastable_share: float
    season_sd_fraction: float
    injury_model: str
    fumble_interpretation: str
    ce: Tuple[float, ...]
    mean_points_per_week: float
    players_mapped: int
    infeasible_injuries: int
    team_deltas: Tuple[TeamDelta, ...] = ()
    champion_discordance: float = 0.0
    """Fraction of seasons in which the two arms crown different champions.

    A league-level measure of how much the scenario changed the world,
    independent of whether any particular team gained."""

    @property
    def is_baseline(self) -> bool:
        return self.axis == "baseline"

    @property
    def ce_max(self) -> float:
        return max(self.ce) if self.ce else 0.0

    @property
    def ce_min(self) -> float:
        return min(self.ce) if self.ce else 0.0

    @property
    def ce_spread(self) -> float:
        """Max minus min CE across the twelve rosters.

        A measure of how much this allocation separates teams at all -- not of
        how good any team is.
        """
        return self.ce_max - self.ce_min

    @property
    def largest_delta(self) -> Optional[TeamDelta]:
        """The team whose paired delta is largest in absolute value."""
        if not self.team_deltas:
            return None
        return max(self.team_deltas, key=lambda d: abs(d.delta_ce))

    @property
    def resolved_teams(self) -> Tuple[TeamDelta, ...]:
        return tuple(d for d in self.team_deltas if d.resolved)

    @property
    def any_resolved(self) -> bool:
        return bool(self.resolved_teams)

    @property
    def max_resolved_delta(self) -> float:
        """Largest |delta| among teams whose own interval excludes zero.

        Zero when nothing on this axis resolved. This -- not the largest raw
        shift -- is what an axis has actually demonstrated.
        """
        res = self.resolved_teams
        return max((abs(d.delta_ce) for d in res), default=0.0)


@dataclass
class SensitivityGrid:
    """Every scenario, and what moved between them."""

    scenarios: List[ScenarioResult]
    baseline_label: str
    n_sims: int
    seed: int
    team_names: Tuple[str, ...] = ()
    roster_source: str = "deterministic snake over the mapped pool"
    contrasts: List[Contrast] = field(default_factory=list)

    def by_axis(self, axis: str) -> List[ScenarioResult]:
        return [s for s in self.scenarios if s.axis in (axis, "baseline")]

    def baseline(self) -> Optional[ScenarioResult]:
        return next((s for s in self.scenarios if s.label == self.baseline_label),
                    None)

    def axes(self) -> List[str]:
        return sorted({s.axis for s in self.scenarios} - {"baseline"})

    def max_ce_shift(self, axis: str) -> float:
        """Largest absolute paired CE change for any roster along one axis."""
        worst = 0.0
        for s in self.scenarios:
            if s.axis != axis:
                continue
            d = s.largest_delta
            if d is not None:
                worst = max(worst, abs(d.delta_ce))
        return worst

    def max_resolved_shift(self, axis: str) -> float:
        """Largest *statistically resolved* CE change along one axis."""
        return max((s.max_resolved_delta for s in self.scenarios
                    if s.axis == axis), default=0.0)

    def axis_is_resolved(self, axis: str) -> bool:
        return any(s.any_resolved for s in self.scenarios if s.axis == axis)

    def summary(self) -> Dict[str, object]:
        base = self.baseline()
        axes = self.axes()
        return {
            "n_sims": self.n_sims, "seed": self.seed,
            "meets_committed_minimum": self.n_sims >= MIN_COMMITTED_SIMS,
            "baseline": self.baseline_label,
            "baseline_ce_range": [round(base.ce_min, 4), round(base.ce_max, 4)]
            if base else None,
            "scenarios": len(self.scenarios),
            "roster_source": self.roster_source,
            "statistics": "paired common random numbers, per season, vs baseline",
            "max_ce_shift_by_axis": {a: round(self.max_ce_shift(a), 5)
                                     for a in axes},
            "max_resolved_ce_shift_by_axis": {
                a: round(self.max_resolved_shift(a), 5) for a in axes},
            "axis_resolved": {a: self.axis_is_resolved(a) for a in axes},
            "contrasts": [
                {"name": c.name, "question": c.question,
                 "a": c.label_a, "b": c.label_b,
                 "discordance": round(c.discordance, 4),
                 "resolved_teams": len(c.resolved_teams),
                 "max_abs_delta_ce": round(
                     abs(c.largest_delta.delta_ce), 5) if c.largest_delta else 0.0,
                 "max_resolved_delta_ce": round(c.max_resolved_delta, 5)}
                for c in self.contrasts],
            "cells": [
                {"label": s.label, "axis": s.axis,
                 "ce_min": round(s.ce_min, 4), "ce_max": round(s.ce_max, 4),
                 "ce_spread": round(s.ce_spread, 4),
                 "mean_points_per_week": round(s.mean_points_per_week, 3),
                 "players_mapped": s.players_mapped,
                 "infeasible_injuries": s.infeasible_injuries,
                 "champion_discordance": round(s.champion_discordance, 4),
                 "resolved_teams": len(s.resolved_teams),
                 "max_abs_delta_ce": round(
                     abs(s.largest_delta.delta_ce), 5) if s.largest_delta else 0.0,
                 "max_resolved_delta_ce": round(s.max_resolved_delta, 5),
                 "team_deltas": [
                     {"team": d.team_name,
                      "ce_baseline": round(d.ce_baseline, 5),
                      "ce_scenario": round(d.ce_scenario, 5),
                      "delta_ce": round(d.delta_ce, 5),
                      "se": round(d.delta_ce_se, 5),
                      "ci95": [round(d.ci95[0], 5), round(d.ci95[1], 5)],
                      "discordance": round(d.discordance, 4),
                      "resolved": d.resolved}
                     for d in s.team_deltas],
                 }
                for s in self.scenarios
            ],
        }


# ---------------------------------------------------------------------------
# Running the grid
# ---------------------------------------------------------------------------


def _paired_deltas(base: SeasonOutcomes, scen: SeasonOutcomes,
                   team_names: Sequence[str]) -> Tuple[Tuple[TeamDelta, ...], float]:
    """Per-team paired deltas plus the league-level champion discordance.

    ``d_i = 1{team wins in the scenario} - 1{team wins in the baseline}`` for
    season *i*. Its mean is the CE difference and its own standard deviation
    gives the standard error, so seasons where the scenario changed nothing
    contribute an exact zero and shrink the interval -- which is the entire
    value of running both arms over the same worlds.
    """
    n = base.champion.shape[0]
    if scen.champion.shape[0] != n:
        raise ValueError("paired arms must have the same number of seasons")
    deltas: List[TeamDelta] = []
    for t, name in enumerate(team_names):
        a = (scen.champion == t).astype(np.float64)
        b = (base.champion == t).astype(np.float64)
        d = a - b
        deltas.append(TeamDelta(
            team_index=t, team_name=name,
            ce_baseline=float(b.mean()), ce_scenario=float(a.mean()),
            delta_ce=float(d.mean()), delta_ce_se=paired_se(d),
            discordance=float(np.mean(d != 0.0))))
    champion_discordance = float(np.mean(base.champion != scen.champion))
    return tuple(deltas), champion_discordance


def _run_one(payload: Dict, cfg: PlayerSpecMappingConfig, axis: str,
             positional_miss: Dict[str, float], positional_cv: Dict[str, float],
             assignment: Sequence[Sequence[int]], only_keys: Iterable[str],
             n_sims: int, seed: int,
             baseline_out: Optional[SeasonOutcomes],
             team_names: Sequence[str]) -> Tuple[ScenarioResult, SeasonOutcomes]:
    """Map, rebuild the fixed rosters, simulate, and pair against the baseline."""
    mapped = map_contract_to_playerspecs(
        payload, cfg, positional_miss=positional_miss,
        positional_cv=positional_cv, only_keys=only_keys)
    rosters = rosters_from_assignment(assignment, mapped.specs,
                                      settings=cfg.settings,
                                      team_names=list(team_names))
    out = simulate_seasons(rosters, n_sims, seed)
    weeks = cfg.settings.regular_season_weeks

    deltas: Tuple[TeamDelta, ...] = ()
    discord = 0.0
    if baseline_out is not None:
        deltas, discord = _paired_deltas(baseline_out, out, team_names)

    result = ScenarioResult(
        label=cfg.label(), axis=axis, target=cfg.target,
        availability_interpretation=cfg.projection_availability_interpretation,
        signal_quality=cfg.signal_quality,
        forecastable_share=cfg.forecastable_share,
        season_sd_fraction=cfg.season_sd_fraction,
        injury_model=cfg.injury_model,
        fumble_interpretation=cfg.fumble_interpretation,
        ce=tuple(float(x) for x in out.championship_equity()),
        mean_points_per_week=float(out.points.mean() / weeks),
        players_mapped=len(mapped.players),
        infeasible_injuries=sum(1 for m in mapped.players
                                if not m.injury.feasible),
        team_deltas=deltas, champion_discordance=discord)
    return result, out


#: The off-corner cells of a deliberately small season_sd x signal-quality
#: grid. `season_sd` says how much there is to learn; `signal_quality` says how
#: fast anyone learns it, and neither is interpretable alone -- a large latent
#: shift nobody can detect is a different world from one everybody detects by
#: week three.
#:
#: Together with the two single-axis sweeps below, the cells actually run form
#: the complete 2 x 3 grid with no cell run twice:
#:
#:            sig=none   sig=week_sd    sig=2x_week_sd
#:   ssd=0.00     -        BASELINE           -           (nothing to learn)
#:   ssd=0.10  signal_q.   season_sd      signal_q.
#:   ssd=0.20  ssd x sig   season_sd      ssd x sig
#:
#: At ssd = 0 there is no latent level to learn, so signal quality cannot bite
#: and that row is not worth running.
SEASON_SD_SIGNAL_GRID: Tuple[Tuple[float, str], ...] = (
    (0.20, "none"), (0.20, "2x_week_sd"),
)


def run_sensitivity(payload_by_fumble: Dict[str, Dict],
                    positional_miss: Dict[str, float],
                    positional_cv: Dict[str, float],
                    limit: int = 300, n_sims: int = MIN_COMMITTED_SIMS,
                    seed: int = 20260904,
                    calibration_sims: int = 200_000,
                    require_minimum: bool = False) -> SensitivityGrid:
    """Sweep every unresolved assumption, one axis at a time from a baseline.

    ``payload_by_fumble`` maps a fumble interpretation to the contract built
    under it, because that choice changes the recomputed season totals and so
    has to happen upstream of the mapping.

    One axis moves at a time, plus a small ``season_sd`` x ``signal_quality``
    interaction grid. A full cross-product would be more scenarios and less
    information; that one interaction is run because neither of its axes means
    anything without the other.

    The twelve rosters are built **once** from the baseline mapping and reused
    verbatim, and every scenario is restricted to exactly the baseline's
    players, so the only thing that changes between arms is the assumption
    under test.
    """
    if require_minimum and n_sims < MIN_COMMITTED_SIMS:
        raise ValueError(
            f"a committed sensitivity report needs at least "
            f"{MIN_COMMITTED_SIMS:,} seasons; got {n_sims:,}")

    base_payload = payload_by_fumble["exclude"]
    common = dict(calibration_sims=calibration_sims)
    baseline_cfg = PlayerSpecMappingConfig(**common)

    # --- the fixed cast ----------------------------------------------------
    baseline_mapped = map_contract_to_playerspecs(
        base_payload, baseline_cfg, positional_miss=positional_miss,
        positional_cv=positional_cv, limit=limit)
    baseline_rosters = build_test_rosters(baseline_mapped.specs,
                                          settings=baseline_cfg.settings)
    assignment = roster_assignment(baseline_rosters)
    team_names = baseline_rosters.team_names
    rostered = {pid for team in assignment for pid in team}
    only_keys = tuple(m.canonical_key for m in baseline_mapped.players
                      if m.spec.player_id in rostered)

    scenarios: List[ScenarioResult] = []
    outcomes: Dict[str, SeasonOutcomes] = {}

    baseline, baseline_out = _run_one(
        base_payload, baseline_cfg, "baseline", positional_miss, positional_cv,
        assignment, only_keys, n_sims, seed, None, team_names)
    scenarios.append(baseline)
    outcomes[baseline.label] = baseline_out

    def add(cfg: PlayerSpecMappingConfig, axis: str, payload: Dict = None) -> None:
        res, out = _run_one(payload if payload is not None else base_payload,
                            cfg, axis, positional_miss, positional_cv,
                            assignment, only_keys, n_sims, seed, baseline_out,
                            team_names)
        scenarios.append(res)
        outcomes[res.label] = out

    # 1. which statistic the season total reports
    add(PlayerSpecMappingConfig(target="mean_target", **common), "target")

    # 2. which health state the season total describes
    for interp in AVAILABILITY_INTERPRETATIONS:
        if interp == baseline_cfg.projection_availability_interpretation:
            continue
        add(PlayerSpecMappingConfig(
            projection_availability_interpretation=interp, **common),
            "availability_interpretation")

    # 3. forecastable variance share
    for f in FORECASTABLE_SHARE_SCENARIOS:
        if f == baseline_cfg.forecastable_share:
            continue
        add(PlayerSpecMappingConfig(forecastable_share=f, **common),
            "forecastable_share")

    # 4. season-level uncertainty, at the baseline signal quality
    for ssd in SEASON_SD_SCENARIOS:
        if ssd == baseline_cfg.season_sd_fraction:
            continue
        add(PlayerSpecMappingConfig(season_sd_fraction=ssd, **common), "season_sd")

    # 5. signal quality on its own. At ssd = 0 there is nothing to learn, so
    #    this is run at a non-zero season_sd where the axis can bite at all.
    probe_ssd = SEASON_SD_SCENARIOS[1] if len(SEASON_SD_SCENARIOS) > 1 else 0.10
    for sq in SIGNAL_QUALITY_SCENARIOS:
        if sq == baseline_cfg.signal_quality:
            continue
        add(PlayerSpecMappingConfig(signal_quality=sq,
                                    season_sd_fraction=probe_ssd, **common),
            "signal_quality")

    # 6. the interaction: how much is learnable x how fast anyone learns it
    for ssd, sq in SEASON_SD_SIGNAL_GRID:
        add(PlayerSpecMappingConfig(season_sd_fraction=ssd, signal_quality=sq,
                                    **common),
            "season_sd_x_signal")

    # 7. individual injury calibration vs the positional all-cause fallback
    add(PlayerSpecMappingConfig(injury_model="positional", **common),
        "injury_model")

    # 8. fumbles excluded vs treated as lost
    if "lost" in payload_by_fumble:
        add(PlayerSpecMappingConfig(fumble_interpretation="lost", **common),
            "fumbles", payload_by_fumble["lost"])

    # Two cells with the same label would be the same run counted twice, and
    # would make an axis look corroborated by its own duplicate.
    labels = [s.label for s in scenarios]
    if len(set(labels)) != len(labels):
        dupes = sorted({lab for lab in labels if labels.count(lab) > 1})
        raise AssertionError(f"duplicate scenario cells: {dupes}")

    # --- does the season_sd answer depend on how fast anyone learns? -------
    # Each of these cells is already measured against the baseline. Comparing
    # two of those measurements by eye would be reading a difference off two
    # overlapping intervals, so the difference gets its own paired run.
    contrasts: List[Contrast] = []
    for ssd in (probe_ssd, SEASON_SD_SCENARIOS[-1]):
        ref = PlayerSpecMappingConfig(season_sd_fraction=ssd, **common).label()
        if ref not in outcomes:
            continue
        for sq in SIGNAL_QUALITY_SCENARIOS:
            if sq == baseline_cfg.signal_quality:
                continue
            other = PlayerSpecMappingConfig(
                season_sd_fraction=ssd, signal_quality=sq, **common).label()
            if other not in outcomes:
                continue
            deltas, disc = _paired_deltas(outcomes[ref], outcomes[other],
                                          team_names)
            contrasts.append(Contrast(
                name=f"ssd={ssd:.2f}: sig={sq} vs sig={baseline_cfg.signal_quality}",
                label_a=ref, label_b=other,
                question=("at this much latent uncertainty, does changing how "
                          "fast managers learn it change championship equity?"),
                team_deltas=deltas, discordance=disc))

    return SensitivityGrid(scenarios=scenarios, baseline_label=baseline.label,
                           n_sims=n_sims, seed=seed, team_names=team_names,
                           contrasts=contrasts)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def format_sensitivity(grid: SensitivityGrid, width: int = 100) -> str:
    """Sanitized rendering: aggregates and paired intervals, no player rows."""
    bar = "=" * width
    base = grid.baseline()
    out = [bar, "REAL-DATA CE SENSITIVITY (sanitized, paired)", bar,
           f"{grid.n_sims:,} seasons per scenario, seed {grid.seed}, "
           f"12 integration rosters",
           "",
           "MODEL SENSITIVITY DIAGNOSTIC ONLY. The twelve rosters are a",
           "deterministic snake over the mapped pool, built once from the",
           "baseline mapping and reused verbatim in every scenario so that the",
           "arms differ only in the assumption under test. This is NOT a",
           "player-value analysis, NOT a price band, and the CE levels are not",
           "advice about anyone.",
           "",
           "Every delta below is a PAIRED difference, computed season by season",
           "against the baseline over identical simulated worlds. Every standard",
           "error comes from that same paired difference. An effect is called",
           "resolved only when its own 95% interval excludes zero.",
           ""]
    if grid.n_sims < MIN_COMMITTED_SIMS:
        out += [f"*** {grid.n_sims:,} seasons is below the {MIN_COMMITTED_SIMS:,} "
                f"minimum for a committed report; treat every",
                "    interval below as a preview rather than a result. ***", ""]

    head = (f"{'scenario':<62} {'CE min':>7} {'CE max':>7} {'pts/wk':>7} "
            f"{'discord':>8} {'infeas':>6}")
    out += ["PER-SCENARIO LEVELS", head, "-" * len(head)]
    for s in grid.scenarios:
        out.append(f"{s.label:<62} {s.ce_min:>7.4f} {s.ce_max:>7.4f} "
                   f"{s.mean_points_per_week:>7.2f} "
                   f"{s.champion_discordance:>8.4f} {s.infeasible_injuries:>6}")
    out += ["-" * len(head), "",
            "  discord = fraction of seasons whose champion differs from the",
            "  baseline's. A large discordance with a near-zero delta means the",
            "  assumption changed the world a great deal without favouring anyone.",
            ""]

    out += ["PAIRED DELTA CE VS BASELINE -- LARGEST-MOVING TEAM PER SCENARIO"]
    head2 = (f"{'scenario':<62} {'team':>7} {'dCE':>9} {'se':>8} "
             f"{'95% CI':>19} {'':>4}")
    out += [head2, "-" * len(head2)]
    for s in grid.scenarios:
        if s.is_baseline:
            continue
        d = s.largest_delta
        if d is None:
            continue
        lo, hi = d.ci95
        flag = "**" if d.resolved else ""
        out.append(f"{s.label:<62} {d.team_name[-2:]:>7} {d.delta_ce:>+9.5f} "
                   f"{d.delta_ce_se:>8.5f} "
                   f"[{lo:+.5f}, {hi:+.5f}] {flag:>4}")
    out += ["-" * len(head2),
            "  ** = this team's own paired 95% interval excludes zero.", ""]

    if grid.contrasts:
        out += ["SCENARIO-VS-SCENARIO CONTRASTS (not against the baseline)",
                "  Does the season_sd answer depend on how fast anyone learns?"]
        head4 = (f"  {'contrast':<44}{'team':>6}{'dCE':>10}{'se':>9}"
                 f"{'discord':>9}{'':>4}")
        out += [head4, "  " + "-" * (len(head4) - 2)]
        for c in grid.contrasts:
            d = c.largest_delta
            if d is None:
                continue
            out.append(f"  {c.name:<44}{d.team_name[-2:]:>6}{d.delta_ce:>+10.5f}"
                       f"{d.delta_ce_se:>9.5f}{c.discordance:>9.4f}"
                       f"{('**' if d.resolved else ''):>4}")
        out += ["  " + "-" * (len(head4) - 2),
                "  A resolved row means learning speed changes the season_sd",
                "  conclusion and the two axes cannot be reported separately.",
                "  An unresolved row means this run cannot separate them.", ""]

    out += ["WHAT EACH AXIS HAS ACTUALLY DEMONSTRATED"]
    head3 = f"  {'axis':<28}{'max |dCE|':>11}{'resolved':>10}{'max resolved':>14}"
    out += [head3, "  " + "-" * (len(head3) - 2)]
    for axis in grid.axes():
        resolved = grid.axis_is_resolved(axis)
        out.append(f"  {axis:<28}{grid.max_ce_shift(axis):>11.5f}"
                   f"{('yes' if resolved else 'no'):>10}"
                   f"{grid.max_resolved_shift(axis):>14.5f}")
    out += ["",
            "Read a resolved shift as: this unresolved assumption has to be",
            "settled before any number downstream of it can be trusted. Read an",
            "unresolved one as: this run cannot tell whether it matters -- which",
            "is not the same as showing that it does not.", bar]
    return "\n".join(out)
