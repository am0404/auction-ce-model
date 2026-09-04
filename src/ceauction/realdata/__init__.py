"""Strict ingestion of real player data.

This package turns vendor files into the versioned contract in
``schemas/real_player_input_v1.schema.json``.  It stops there.  It does not
produce dollar values, opening bids, live bids or auction-room behaviour, and it
does not populate ``PlayerSpec`` fields that the sources cannot support.

The design rule throughout: **an unresolved question travels with the data.**
Where the inventory found a meaning that could not be proven, this package
records the ambiguity in the output rather than resolving it silently.  Three
places where that shows:

* the source's availability treatment is unknown, so **both** readings of it are
  computed and neither is marked preferred;
* the ``Fumbles`` column's meaning is unknown, so its points are excluded from
  the primary total and the omitted contribution is reported;
* the projections' central tendency is recorded as ``median`` on the user's
  assertion, with the provenance of that assertion attached.

See ``docs/PLAYER_DATA_INVENTORY.md`` for what each source actually contains and
``docs/PLAYER_MAPPING_GAPS.md`` for which engine parameters remain uncalibrated.
"""

from __future__ import annotations

from .contract import (
    FANTASY_SCHEDULED_GAMES,
    FUMBLE_INTERPRETATIONS,
    GAMES_BASIS,
    UNSUPPORTED_SCORING_CATEGORIES,
    build_contract,
)
from .coverage import AliasBook, coverage_by_band, load_alias_book
from .identity import IdentityIndex, MatchReport, normalize_name
from .mapping import (
    PlayerSpecMappingConfig,
    calibrate_injury,
    calibrate_level,
    map_contract_to_playerspecs,
)
from .smoke import build_test_rosters, run_smoke_checks
from .report import build_report, format_report, numeric_summary
from .scoring import ScoringBreakdown, season_points_from_components
from .sources import (
    DispersionFits,
    SyntheticSourceRefused,
    FantasyProsRow,
    InjuryProfileRow,
    ProjectionRow,
    SourceFile,
    load_dispersion_fits,
    load_fantasypros,
    load_injury_profiles,
    load_projections,
)
from .validate import ValidationError, ValidationResult, validate_contract

__all__ = [
    "FUMBLE_INTERPRETATIONS",
    "GAMES_BASIS",
    "FANTASY_SCHEDULED_GAMES",
    "PlayerSpecMappingConfig",
    "calibrate_level",
    "calibrate_injury",
    "map_contract_to_playerspecs",
    "build_test_rosters",
    "run_smoke_checks",
    "coverage_by_band",
    "load_alias_book",
    "AliasBook",
    "UNSUPPORTED_SCORING_CATEGORIES",
    "build_contract",
    "IdentityIndex",
    "MatchReport",
    "normalize_name",
    "build_report",
    "format_report",
    "numeric_summary",
    "SyntheticSourceRefused",
    "ScoringBreakdown",
    "season_points_from_components",
    "DispersionFits",
    "FantasyProsRow",
    "InjuryProfileRow",
    "ProjectionRow",
    "SourceFile",
    "load_dispersion_fits",
    "load_fantasypros",
    "load_injury_profiles",
    "load_projections",
    "ValidationError",
    "ValidationResult",
    "validate_contract",
]
