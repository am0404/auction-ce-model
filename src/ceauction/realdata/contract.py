"""Build the normalized real-player contract from the loaded sources.

This is where the settled modelling decisions are applied, and each one is
applied *visibly*:

* **Target league.** The 12-team superflex league in ``SPEC.md``. The previous
  model's 10-team non-superflex configuration never enters.
* **Central tendency.** Recorded as ``median`` with the provenance
  "user asserted; vendor documentation not located". ``PlayerSpec.base_mean``
  is an expected value and a median is not one, so the claim travels attached
  to its source rather than silently.
* **Points.** Always recomputed from components under the target league's
  scoring. The vendor's own total is never read.
* **Expert grades.** Carried as optional metadata. Never mapped to any
  distribution parameter.
* **Injury fields.** ``injury_prob`` (season-level risk) and
  ``proj_games_missed`` (projected games missed) are preserved separately. No
  weekly injury process is derived.
* **Availability.** Both interpretations are computed and **neither is
  preferred**, because the source's treatment is unresolved.
* **Fumbles.** Excluded from the primary total by default, with the omitted
  contribution reported.
* **Missing categories.** Two-point conversions and individual special-teams
  touchdowns are absent, not zero.

What this module deliberately does **not** do: populate any ``PlayerSpec`` field
the sources cannot support. Those are listed in ``uncalibrated_parameters`` so a
downstream run can always say which parts of its answer rest on data.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

from ..league import DEFAULT_LEAGUE, LeagueSettings
from ..scoring import HALF_PPR, ScoringRules
from .identity import IdentityIndex, MatchReport, join_report, normalize_name
from .scoring import (
    FUMBLE_INTERPRETATIONS,
    UNSUPPORTED_CATEGORIES,
    ScoringBreakdown,
    season_points_from_components,
)
from .sources import (
    DispersionFits,
    FantasyProsRow,
    InjuryProfileRow,
    ProjectionRow,
    SourceFile,
)

__all__ = [
    "GAMES_BASIS",
    "FUMBLE_INTERPRETATIONS",
    "UNSUPPORTED_SCORING_CATEGORIES",
    "TARGET_LEAGUE_CONFIG_ID",
    "CENTRAL_TENDENCY",
    "CENTRAL_TENDENCY_PROVENANCE",
    "BuildResult",
    "build_contract",
    "UNCALIBRATED_PARAMETERS",
    "OPEN_QUESTIONS",
]

#: Games a season total is spread over under interpretation A. Seventeen, per
#: the target league's scoring window (14 regular-season weeks plus a 3-week
#: bracket). Explicit and overridable rather than buried in an expression.
GAMES_BASIS = 17.0

UNSUPPORTED_SCORING_CATEGORIES = UNSUPPORTED_CATEGORIES

#: Names the target league. Any payload whose id looks like a 10-team
#: non-superflex configuration is rejected by ``validate_semantics``.
TARGET_LEAGUE_CONFIG_ID = "spec-md-12team-superflex-half-ppr"

CENTRAL_TENDENCY = "median"
CENTRAL_TENDENCY_PROVENANCE = (
    "user asserted; vendor documentation not located"
)

#: Engine parameters this source cannot calibrate, carried with the payload.
UNCALIBRATED_PARAMETERS: Tuple[Dict[str, str], ...] = (
    {"parameter": "season_sd",
     "reason": "no measure of true-talent deviation from consensus",
     "would_be_settled_by": "preseason projections joined to realised season means"},
    {"parameter": "signal_noise_sd",
     "reason": "no usage-vs-outcome relationship measured",
     "would_be_settled_by": "rest-of-season scoring regressed on usage through week k"},
    {"parameter": "weekly_state_sd",
     "reason": "no forecastable/unforecastable split; the fitted CV is total dispersion",
     "would_be_settled_by": "R-squared of archived weekly projections against weekly outcomes"},
    {"parameter": "proj_noise_sd",
     "reason": "no archived weekly projections",
     "would_be_settled_by": "RMSE of weekly projections against outcomes"},
    {"parameter": "spike_rate",
     "reason": "no tail-shape measurement; expert grades are not a distribution",
     "would_be_settled_by": "fitting the upper tail of weekly score distributions"},
    {"parameter": "spike_scale",
     "reason": "no tail-shape measurement; expert grades are not a distribution",
     "would_be_settled_by": "fitting the upper tail of weekly score distributions"},
    {"parameter": "role_change_prob",
     "reason": "no role-transition rates fitted",
     "would_be_settled_by": "hand-tagged dated depth-chart changes"},
    {"parameter": "shock_loadings",
     "reason": "no correlation structure available",
     "would_be_settled_by": "residual correlation matrix of weekly scores"},
    {"parameter": "contingency",
     "reason": "no real depth chart available",
     "would_be_settled_by": "a real depth chart plus backup-usage-on-absence estimates"},
    {"parameter": "weekly_injury_hazard",
     "reason": "injury_prob is season-level risk, not a weekly hazard; no weekly "
               "injury process has been derived",
     "would_be_settled_by": "a stated model converting season risk and projected "
                            "games missed into a weekly process"},
)

#: Questions that must be answered before this data can populate the engine.
OPEN_QUESTIONS: Tuple[Dict[str, object], ...] = (
    {"id": "Q1",
     "question": "What week range does the season projection span, and is 17 the "
                 "right games basis?",
     "affects": ["base_mean", "active_rate"], "blocking": True},
    {"id": "Q2",
     "question": "Is the Fumbles column total fumbles or fumbles lost?",
     "affects": ["season_points"], "blocking": True},
    {"id": "Q3",
     "question": "Is the projection already availability-adjusted, or full health?",
     "affects": ["active_rate"], "blocking": True},
    {"id": "Q5",
     "question": "What does the injury vendor mean by a significant injury, and "
                 "over what horizon?",
     "affects": ["injury_prob", "proj_games_missed"], "blocking": False},
    {"id": "Q7",
     "question": "How stale is the injury capture? It carries no update timestamp.",
     "affects": ["injury_prob", "proj_games_missed"], "blocking": False},
)


@dataclass
class BuildResult:
    """The contract payload plus everything the join could not do."""

    payload: Dict[str, object]
    reports: Dict[str, MatchReport]
    warnings: List[str]

    @property
    def player_count(self) -> int:
        return len(self.payload.get("players", []))


def _active_rates(points: float, games_basis: float,
                  games_missed: Optional[float]) -> Dict[str, object]:
    """Both readings of what the projection already accounts for.

    **A, full health**: the total describes a player who plays every game, so
    the per-game rate is ``points / games_basis`` and availability must be
    applied separately downstream.

    **B, availability-adjusted**: the total already reflects expected absence,
    so it is spread over only the games he is expected to play and the rate per
    game actually played is ``points / (games_basis - games_missed)`` -- higher
    than A whenever any games are projected missed.

    Neither is marked preferred. Choosing between them is a modelling decision
    that has not been made, and picking one silently would either double-count
    injuries or ignore them entirely.
    """
    a = points / games_basis if games_basis else None
    b: Optional[float] = None
    if games_missed is not None:
        available = games_basis - games_missed
        # A player projected to miss the whole season has no per-active-game
        # rate; reporting one would be a division artefact, not a quantity.
        if available > 0:
            b = points / available
    return {
        "games_basis": games_basis,
        "interpretation_a_full_health": a,
        "interpretation_b_availability_adjusted": b,
        "preferred": None,
    }


def build_contract(
    projections: Sequence[ProjectionRow],
    projection_source: SourceFile,
    fantasypros: Sequence[FantasyProsRow] = (),
    fantasypros_source: Optional[SourceFile] = None,
    injuries: Sequence[InjuryProfileRow] = (),
    injury_source: Optional[SourceFile] = None,
    fits: Optional[DispersionFits] = None,
    fits_source: Optional[SourceFile] = None,
    settings: LeagueSettings = DEFAULT_LEAGUE,
    scoring: ScoringRules = HALF_PPR,
    games_basis: float = GAMES_BASIS,
    fumble_interpretation: str = "exclude",
    league_config_id: str = TARGET_LEAGUE_CONFIG_ID,
    generated_at: Optional[str] = None,
    include_expert_labels: bool = True,
) -> BuildResult:
    """Join the sources into one normalized, self-describing payload.

    The projection file is the spine: a player who is not projected has no
    row, because the other sources describe players rather than produce them.
    """
    if fumble_interpretation not in FUMBLE_INTERPRETATIONS:
        raise ValueError(
            f"fumble_interpretation must be one of {FUMBLE_INTERPRETATIONS}")
    if settings.n_teams != 12:
        raise ValueError(
            f"target league must be the 12-team superflex league in SPEC.md; "
            f"got n_teams={settings.n_teams}. The previous model's 10-team "
            f"non-superflex configuration must not enter the CE engine.")

    warnings: List[str] = []
    reports: Dict[str, MatchReport] = {}

    proj_index = IdentityIndex(projections, lambda r: r.name,
                               lambda r: r.position, source="projections")
    fp_index = IdentityIndex(fantasypros, lambda r: r.name,
                             lambda r: r.position, source="fantasypros")
    inj_index = IdentityIndex(injuries, lambda r: r.name,
                              lambda r: r.position, source="injury")

    if fantasypros:
        reports["fantasypros"] = join_report(proj_index, fp_index, "fantasypros")
    if injuries:
        reports["injury"] = join_report(proj_index, inj_index, "injury")

    for key, count in proj_index.duplicates():
        warnings.append(
            f"projections: {count} rows share the normalised name {key!r}; "
            f"neither row is used for the ambiguous join.")
    for key, positions in proj_index.conflicts():
        warnings.append(
            f"projections: {key!r} appears at conflicting positions {positions}.")

    players: List[Dict[str, object]] = []
    seen_keys: set = set()
    for row in projections:
        key = normalize_name(row.name)
        if not key:
            continue
        if key in seen_keys:
            # The duplicate is already reported above; emitting it twice would
            # put two rows under one player_key in the contract.
            continue
        seen_keys.add(key)

        breakdown = season_points_from_components(
            row.stat_dict(), rules=scoring,
            fumble_interpretation=fumble_interpretation)

        meta = None if proj_index.is_ambiguous(key) else fp_index.get(key)
        prof = None if proj_index.is_ambiguous(key) else inj_index.get(key)
        if fp_index.is_ambiguous(key):
            meta = None
        if inj_index.is_ambiguous(key):
            prof = None

        games_missed = prof.proj_games_missed if prof else None

        player: Dict[str, object] = {
            "player_key": key.replace(" ", "_"),
            "name": row.name,
            "position": row.position,
            "nfl_team": (meta.team if meta else None),
            "bye_week": (meta.bye_week if meta and meta.bye_week else None),
            "stat_line": {
                "pass_yards": row.pass_yards,
                "pass_tds": row.pass_tds,
                "interceptions": row.interceptions,
                "rush_yards": row.rush_yards,
                "rush_tds": row.rush_tds,
                "rec_yards": row.rec_yards,
                "rec_tds": row.rec_tds,
                "receptions": row.receptions,
                "fumbles": row.fumbles,
                # Unresolved: this league scores fumbles LOST and the column is
                # named Fumbles. Null until Q2 is answered.
                "fumbles_are_lost_fumbles": None,
            },
            "stat_line_horizon": {
                "basis": "season_total",
                # Unresolved (Q1): the source does not state its window.
                "games_assumed": None,
                # Unresolved (Q3): both readings are given in active_rate.
                "conditional_on_playing": None,
                "central_tendency": CENTRAL_TENDENCY,
                "central_tendency_provenance": CENTRAL_TENDENCY_PROVENANCE,
            },
            "season_points": {
                "points": breakdown.points,
                "scoring_source": "recomputed_from_components",
                "fumble_interpretation": breakdown.fumble_interpretation,
                "omitted_fumble_points": breakdown.omitted_fumble_points,
            },
            "active_rate": _active_rates(breakdown.points, games_basis,
                                         games_missed),
            "availability": {
                "injury_prob": (prof.injury_prob if prof else None),
                "injury_prob_definition": None,
                "proj_games_missed": games_missed,
                "proj_games_missed_definition": None,
                "games_in_horizon": games_basis if prof else None,
            },
            "raw_fields": dict(row.raw),
        }

        if fits is not None:
            cv = fits.weekly_cv.get(row.position)
            miss = fits.weekly_miss.get(row.position)
            player["cohort_dispersion"] = {
                "weekly_cv": cv,
                "weekly_cv_is_total_dispersion": True,
                "weekly_miss_rate": miss,
                "fit_provenance": (
                    f"seasons {list(fits.seasons)}; cv n={fits.cv_sample}, "
                    f"miss n={fits.miss_sample}; cohort: {fits.cv_cohort}"),
            }

        if include_expert_labels and meta is not None:
            player["expert_labels"] = {
                "upside": meta.upside,
                "bust": meta.bust,
                "scale": "ordinal_1_to_5",
                # Pinned. These are ordinal expert labels, not a distribution.
                "may_derive_dispersion": False,
            }

        players.append(player)

    sources = [projection_source.to_dict()]
    for src in (fantasypros_source, injury_source, fits_source):
        if src is not None:
            sources.append(src.to_dict())

    supported = ["pass_yard", "pass_td", "interception", "rush_yard",
                 "rush_td", "rec_yard", "rec_td", "reception"]
    unsupported = [
        {"category": c,
         "reason": "no source column in the component projection export",
         "treated_as": "absent"}
        for c in UNSUPPORTED_SCORING_CATEGORIES
    ]
    if fumble_interpretation == "exclude":
        unsupported.append({
            "category": "fumble_lost",
            "reason": ("the source column is named 'Fumbles' and whether it means "
                       "fumbles lost is unresolved (Q2); its contribution is "
                       "excluded and reported per player in omitted_fumble_points"),
            "treated_as": "absent"})
    else:
        supported.append("fumble_lost")

    payload = {
        "schema_version": "1.0.0",
        "provenance": {
            "generated_at": generated_at or datetime.now(timezone.utc)
                                                    .isoformat(timespec="seconds"),
            "league_config_id": league_config_id,
            "sources": sources,
        },
        "scoring_support": {
            "supported_categories": supported,
            "unsupported_categories": unsupported,
        },
        "uncalibrated_parameters": [dict(p) for p in UNCALIBRATED_PARAMETERS],
        "open_questions": [dict(q) for q in OPEN_QUESTIONS],
        "players": players,
    }

    for name, rep in reports.items():
        if rep.match_rate < 0.5:
            warnings.append(
                f"{name}: only {rep.matched}/{rep.left_rows} projected players "
                f"matched ({rep.match_rate:.1%}). A half-matched join applies "
                f"information to some players and not others, which looks like "
                f"a signal.")
    return BuildResult(payload=payload, reports=reports, warnings=warnings)
