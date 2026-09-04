"""How much does a roster's championship equity depend on what we assumed?

Every axis swept here is an assumption the data does not settle. The point is
not to find a best cell -- there is no evidence for one -- but to measure how
far CE moves when an unresolved choice is made differently. A scenario that
barely moves CE is one nobody needs to resolve before pricing; a scenario that
moves it a lot is a blocker.

Output is aggregate: per-scenario CE for the twelve integration rosters, plus
spreads. It carries no player rows and no proprietary values.
"""

from __future__ import annotations

import statistics as st
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..simulate import simulate_seasons
from .mapping import (
    FORECASTABLE_SHARE_SCENARIOS,
    SEASON_SD_SCENARIOS,
    PlayerSpecMappingConfig,
    map_contract_to_playerspecs,
)
from .smoke import build_test_rosters

__all__ = ["ScenarioResult", "SensitivityGrid", "run_sensitivity", "format_sensitivity"]


@dataclass(frozen=True)
class ScenarioResult:
    """One cell of the grid."""

    label: str
    axis: str
    target: str
    forecastable_share: float
    season_sd_fraction: float
    injury_model: str
    fumble_interpretation: str
    ce: Tuple[float, ...]
    mean_points_per_week: float
    players_mapped: int
    infeasible_injuries: int

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


@dataclass
class SensitivityGrid:
    """Every scenario, and what moved between them."""

    scenarios: List[ScenarioResult]
    baseline_label: str
    n_sims: int
    seed: int

    def by_axis(self, axis: str) -> List[ScenarioResult]:
        return [s for s in self.scenarios if s.axis in (axis, "baseline")]

    def baseline(self) -> Optional[ScenarioResult]:
        return next((s for s in self.scenarios if s.label == self.baseline_label),
                    None)

    def max_ce_shift(self, axis: str) -> float:
        """Largest absolute CE change for any roster along one axis."""
        base = self.baseline()
        if base is None:
            return 0.0
        worst = 0.0
        for s in self.scenarios:
            if s.axis != axis:
                continue
            for a, b in zip(s.ce, base.ce):
                worst = max(worst, abs(a - b))
        return worst

    def summary(self) -> Dict[str, object]:
        base = self.baseline()
        axes = sorted({s.axis for s in self.scenarios} - {"baseline"})
        return {
            "n_sims": self.n_sims, "seed": self.seed,
            "baseline": self.baseline_label,
            "baseline_ce_range": [round(base.ce_min, 4), round(base.ce_max, 4)]
            if base else None,
            "scenarios": len(self.scenarios),
            "max_ce_shift_by_axis": {a: round(self.max_ce_shift(a), 5)
                                     for a in axes},
            "cells": [
                {"label": s.label, "axis": s.axis,
                 "ce_min": round(s.ce_min, 4), "ce_max": round(s.ce_max, 4),
                 "ce_spread": round(s.ce_spread, 4),
                 "mean_points_per_week": round(s.mean_points_per_week, 3),
                 "players_mapped": s.players_mapped,
                 "infeasible_injuries": s.infeasible_injuries}
                for s in self.scenarios
            ],
        }


def _run_one(payload: Dict, cfg: PlayerSpecMappingConfig, axis: str,
             positional_miss: Dict[str, float], positional_cv: Dict[str, float],
             limit: int, n_sims: int, seed: int) -> ScenarioResult:
    mapped = map_contract_to_playerspecs(
        payload, cfg, positional_miss=positional_miss,
        positional_cv=positional_cv, limit=limit)
    rosters = build_test_rosters(mapped.specs, settings=cfg.settings)
    out = simulate_seasons(rosters, n_sims, seed)
    weeks = cfg.settings.regular_season_weeks
    return ScenarioResult(
        label=cfg.label(), axis=axis, target=cfg.target,
        forecastable_share=cfg.forecastable_share,
        season_sd_fraction=cfg.season_sd_fraction,
        injury_model=cfg.injury_model,
        fumble_interpretation=cfg.fumble_interpretation,
        ce=tuple(float(x) for x in out.championship_equity()),
        mean_points_per_week=float(out.points.mean() / weeks),
        players_mapped=len(mapped.players),
        infeasible_injuries=sum(1 for m in mapped.players
                                if not m.injury.feasible),
    )


def run_sensitivity(payload_by_fumble: Dict[str, Dict],
                    positional_miss: Dict[str, float],
                    positional_cv: Dict[str, float],
                    limit: int = 300, n_sims: int = 2000,
                    seed: int = 20260904,
                    calibration_sims: int = 200_000) -> SensitivityGrid:
    """Sweep every unresolved assumption, one axis at a time from a baseline.

    ``payload_by_fumble`` maps a fumble interpretation to the contract built
    under it, because that choice changes the recomputed season totals and so
    has to happen upstream of the mapping.

    One axis moves at a time. A full cross-product would be more scenarios and
    less information: what a reader needs is how far each individual unresolved
    choice can move the answer.
    """
    base_payload = payload_by_fumble["exclude"]
    common = dict(calibration_sims=calibration_sims)
    scenarios: List[ScenarioResult] = []

    baseline_cfg = PlayerSpecMappingConfig(**common)
    baseline = _run_one(base_payload, baseline_cfg, "baseline", positional_miss,
                        positional_cv, limit, n_sims, seed)
    scenarios.append(baseline)

    # 1. projection target
    scenarios.append(_run_one(
        base_payload, PlayerSpecMappingConfig(target="mean_target", **common),
        "target", positional_miss, positional_cv, limit, n_sims, seed))

    # 2. forecastable variance share
    for f in FORECASTABLE_SHARE_SCENARIOS:
        if f == baseline_cfg.forecastable_share:
            continue
        scenarios.append(_run_one(
            base_payload, PlayerSpecMappingConfig(forecastable_share=f, **common),
            "forecastable_share", positional_miss, positional_cv, limit,
            n_sims, seed))

    # 3. season-level uncertainty
    for ssd in SEASON_SD_SCENARIOS:
        if ssd == baseline_cfg.season_sd_fraction:
            continue
        scenarios.append(_run_one(
            base_payload, PlayerSpecMappingConfig(season_sd_fraction=ssd, **common),
            "season_sd", positional_miss, positional_cv, limit, n_sims, seed))

    # 4. individual injury calibration vs the positional all-cause fallback
    scenarios.append(_run_one(
        base_payload, PlayerSpecMappingConfig(injury_model="positional", **common),
        "injury_model", positional_miss, positional_cv, limit, n_sims, seed))

    # 5. fumbles excluded vs treated as lost
    if "lost" in payload_by_fumble:
        scenarios.append(_run_one(
            payload_by_fumble["lost"],
            PlayerSpecMappingConfig(fumble_interpretation="lost", **common),
            "fumbles", positional_miss, positional_cv, limit, n_sims, seed))

    return SensitivityGrid(scenarios=scenarios, baseline_label=baseline.label,
                           n_sims=n_sims, seed=seed)


def format_sensitivity(grid: SensitivityGrid, width: int = 96) -> str:
    """Sanitized rendering: aggregates and ranges, no player rows."""
    bar = "=" * width
    base = grid.baseline()
    out = [bar, "REAL-DATA CE SENSITIVITY (sanitized)", bar,
           f"{grid.n_sims:,} seasons per scenario, seed {grid.seed}, "
           f"12 integration rosters",
           "",
           "These rosters are a deterministic snake over the real draftable pool,",
           "built solely so the engine has twelve legal disjoint teams to simulate.",
           "They are not an auction allocation and the CE levels are not advice.",
           "What the table measures is how far CE MOVES when an unresolved",
           "assumption is made differently.",
           ""]
    head = (f"{'scenario':<46} {'CE min':>8} {'CE max':>8} {'spread':>8} "
            f"{'pts/wk':>8} {'infeas':>7}")
    out += [head, "-" * len(head)]
    for s in grid.scenarios:
        out.append(f"{s.label:<46} {s.ce_min:>8.4f} {s.ce_max:>8.4f} "
                   f"{s.ce_spread:>8.4f} {s.mean_points_per_week:>8.2f} "
                   f"{s.infeasible_injuries:>7}")
    out += ["-" * len(head), "",
            "LARGEST CE SHIFT FOR ANY SINGLE ROSTER, VERSUS THE BASELINE"]
    for axis, shift in grid.summary()["max_ce_shift_by_axis"].items():
        out.append(f"  {axis:<24} {shift:+.5f}")
    out += ["",
            "Read a large shift as: this unresolved assumption has to be settled",
            "before any number downstream of it can be trusted. Read a small one",
            "as: it can wait.", bar]
    return "\n".join(out)
