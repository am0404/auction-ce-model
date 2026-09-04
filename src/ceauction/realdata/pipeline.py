"""End-to-end ingestion: sources in, contract and sanitized report out.

Separated from the CLI so the whole path is testable without argument parsing,
and so a caller can drive it from a config file instead of flags.

**Every path is a parameter.** Nothing in this package knows where the data
lives. The sources are subscriber-gated vendor exports that are not
redistributable and must never enter this public repository.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from ..league import DEFAULT_LEAGUE, LeagueSettings
from ..scoring import HALF_PPR, ScoringRules
from .contract import GAMES_BASIS, TARGET_LEAGUE_CONFIG_ID, BuildResult, build_contract
from .report import build_report, format_report
from .sources import (
    load_dispersion_fits,
    load_fantasypros,
    load_injury_profiles,
    load_projections,
)
from .validate import ValidationResult, validate_contract

__all__ = ["IngestionPaths", "IngestionOutcome", "ingest"]


@dataclass(frozen=True)
class IngestionPaths:
    """Where the sources are. Supplied by the caller, never defaulted."""

    projections: Path
    fantasypros: Optional[Path] = None
    injuries: Optional[Path] = None
    fits: Optional[Path] = None


@dataclass
class IngestionOutcome:
    """The contract, its validation, and the report that may be committed."""

    payload: Dict
    validation: ValidationResult
    report: Dict
    build: BuildResult

    @property
    def ok(self) -> bool:
        return self.validation.ok

    @property
    def player_count(self) -> int:
        return len(self.payload.get("players", []))

    def format_report(self) -> str:
        return format_report(self.report)


def ingest(
    paths: IngestionPaths,
    settings: LeagueSettings = DEFAULT_LEAGUE,
    scoring: ScoringRules = HALF_PPR,
    games_basis: float = GAMES_BASIS,
    fumble_interpretation: str = "exclude",
    league_config_id: str = TARGET_LEAGUE_CONFIG_ID,
    generated_at: Optional[str] = None,
) -> IngestionOutcome:
    """Load, join, validate and summarise.

    Raises :class:`~.sources.SyntheticSourceRefused` if any CSV supplied has the
    column signature of the generated provisional player pool.
    """
    projections, proj_src = load_projections(paths.projections)

    fantasypros, fp_src = ((), None)
    if paths.fantasypros is not None:
        fantasypros, fp_src = load_fantasypros(paths.fantasypros)

    injuries, inj_src = ((), None)
    if paths.injuries is not None:
        injuries, inj_src = load_injury_profiles(paths.injuries)

    fits, fits_src = (None, None)
    if paths.fits is not None:
        fits, fits_src = load_dispersion_fits(paths.fits)

    build = build_contract(
        projections, proj_src, fantasypros, fp_src, injuries, inj_src,
        fits, fits_src, settings=settings, scoring=scoring,
        games_basis=games_basis, fumble_interpretation=fumble_interpretation,
        league_config_id=league_config_id, generated_at=generated_at,
    )
    validation = validate_contract(build.payload)
    report = build_report(build.payload, build.reports, validation,
                          build.warnings)
    return IngestionOutcome(payload=build.payload, validation=validation,
                            report=report, build=build)


def write_outputs(outcome: IngestionOutcome, contract_path: Optional[Path],
                  report_path: Optional[Path]) -> Dict[str, str]:
    """Write the contract and/or the sanitized report.

    The contract contains real player rows and belongs only in an ignored
    location; the caller is responsible for that and the CLI enforces it.
    """
    written: Dict[str, str] = {}
    if contract_path is not None:
        contract_path = Path(contract_path)
        contract_path.parent.mkdir(parents=True, exist_ok=True)
        contract_path.write_text(json.dumps(outcome.payload, indent=2) + "\n",
                                 encoding="utf-8")
        written["contract"] = str(contract_path)
    if report_path is not None:
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(outcome.report, indent=2) + "\n",
                               encoding="utf-8")
        written["report"] = str(report_path)
    return written
