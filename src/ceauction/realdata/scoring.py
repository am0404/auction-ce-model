"""Component stat lines to league points, with the omissions made visible.

Two rules govern this module, and both are the result of the inventory.

**The vendor's own fantasy total is never used.**  It solves to full PPR -- a
0.98 reception coefficient -- and taking it would add double-digit phantom
points to every pass catcher.  Points are always recomputed from components
using the *target* league's scoring.

**A category with no source column is unmodelled, not zero.**  This league
scores three kinds of two-point conversion and an individual special-teams
touchdown; the source carries none of them.  They are reported as absent, and
:class:`ScoringBreakdown` records them so that a total is never mistaken for
complete.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..scoring import HALF_PPR, ScoringRules

__all__ = [
    "ScoringBreakdown",
    "season_points_from_components",
    "FUMBLE_INTERPRETATIONS",
    "SUPPORTED_CATEGORIES",
    "UNSUPPORTED_CATEGORIES",
]

#: How the source's ``Fumbles`` column may be read.  ``exclude`` is the only
#: default: this league scores -2 for a fumble **lost**, the column is named
#: ``Fumbles``, and roughly half of NFL fumbles are recovered by the offence, so
#: reading total-as-lost would roughly double the penalty.  The other two exist
#: so the question can be answered later without a code change.
FUMBLE_INTERPRETATIONS = ("exclude", "lost", "total")

#: Component fields this league scores that the source supplies.
SUPPORTED_CATEGORIES: Tuple[str, ...] = (
    "pass_yard", "pass_td", "interception",
    "rush_yard", "rush_td", "rec_yard", "rec_td", "reception",
)

#: League scoring categories with no source column at all.  Absent, not zero.
UNSUPPORTED_CATEGORIES: Tuple[str, ...] = (
    "pass_2pt", "rush_2pt", "rec_2pt", "special_teams_td",
)


@dataclass(frozen=True)
class ScoringBreakdown:
    """Season points, plus everything deliberately left out of them."""

    points: float
    """Season points from the components this league scores.  Excludes the
    fumble contribution unless ``fumble_interpretation`` says otherwise."""

    per_category: Dict[str, float]
    """Contribution of each scored category, for auditing."""

    fumble_interpretation: str
    omitted_fumble_points: Optional[float]
    """Points NOT included because of the fumble interpretation.  ``None`` when
    the source carried no fumble value for this player.  Reported so the size of
    the omission is visible rather than silent."""

    missing_categories: Tuple[str, ...] = UNSUPPORTED_CATEGORIES
    """League categories with no source column.  These are unmodelled."""

    null_components: Tuple[str, ...] = ()
    """Component fields that were absent for this player.  A blank cell means
    *not projected*; it is scored as no contribution, but which fields were
    blank is recorded rather than lost."""

    @property
    def points_including_fumbles(self) -> float:
        """What the total would be if the omitted fumble points were included.

        Never the primary number.  Provided so a report can state the size of
        the open question rather than describing it in words alone.
        """
        return self.points + (self.omitted_fumble_points or 0.0)


def _num(value) -> Optional[float]:
    """Parse a source cell.  Blank / ``-`` / ``nan`` mean *not projected*."""
    if value is None:
        return None
    text = str(value).strip().strip('"')
    if text in ("", "-", "nan", "None", "NA", "N/A"):
        return None
    try:
        return float(text.replace(",", ""))
    except ValueError:
        return None


def season_points_from_components(
    stats: Dict[str, object],
    rules: ScoringRules = HALF_PPR,
    fumble_interpretation: str = "exclude",
) -> ScoringBreakdown:
    """Recompute season fantasy points from component statistics.

    Parameters
    ----------
    stats:
        Component values keyed by this project's canonical stat names
        (``pass_yards``, ``pass_tds``, ``interceptions``, ``rush_yards``,
        ``rush_tds``, ``rec_yards``, ``rec_tds``, ``receptions``, ``fumbles``).
        ``None`` and blank strings mean *not projected*.
    rules:
        The **target** league's scoring.  Never a vendor's.
    fumble_interpretation:
        ``"exclude"`` (default) leaves the fumble contribution out of ``points``
        and reports it in ``omitted_fumble_points``.  ``"lost"`` treats the
        column as fumbles lost and includes it.  ``"total"`` also includes it,
        and is offered only so the choice is explicit; it is the reading that
        makes the source column and the league rule agree least.
    """
    if fumble_interpretation not in FUMBLE_INTERPRETATIONS:
        raise ValueError(
            f"fumble_interpretation must be one of {FUMBLE_INTERPRETATIONS}, "
            f"got {fumble_interpretation!r}"
        )

    fields: Tuple[Tuple[str, str, float], ...] = (
        ("pass_yards", "pass_yard", rules.pass_yard),
        ("pass_tds", "pass_td", rules.pass_td),
        ("interceptions", "interception", rules.interception),
        ("rush_yards", "rush_yard", rules.rush_yard),
        ("rush_tds", "rush_td", rules.rush_td),
        ("rec_yards", "rec_yard", rules.rec_yard),
        ("rec_tds", "rec_td", rules.rec_td),
        ("receptions", "reception", rules.reception),
    )

    per_category: Dict[str, float] = {}
    nulls: List[str] = []
    total = 0.0
    for stat_key, category, coeff in fields:
        value = _num(stats.get(stat_key))
        if value is None:
            nulls.append(stat_key)
            continue
        contribution = value * coeff
        per_category[category] = contribution
        total += contribution

    fumbles = _num(stats.get("fumbles"))
    if fumbles is None:
        nulls.append("fumbles")
        fumble_points = None
    else:
        fumble_points = fumbles * rules.fumble_lost

    omitted = None
    if fumble_interpretation == "exclude":
        omitted = fumble_points
    elif fumble_points is not None:
        per_category["fumble_lost"] = fumble_points
        total += fumble_points

    return ScoringBreakdown(
        points=total,
        per_category=per_category,
        fumble_interpretation=fumble_interpretation,
        omitted_fumble_points=omitted,
        null_components=tuple(nulls),
    )
