"""Half-PPR scoring rules for the league.

The CE engine models fantasy *points* directly, so this module is not on the
hot path.  It exists because it is the documented, tested seam for the moment
real data arrives as projected **stat lines** rather than projected points:
call :func:`score_statline` to convert, then feed the result into
``PlayerSpec.base_mean`` / ``weekly_projection_override``.

The seam represents the *complete* league scoring system, including the events
the engine never needs to reason about on its own (two-point conversions,
individual special-teams touchdowns).  An incomplete seam would silently drop
points the moment real stat lines arrive, so completeness here is cheap
insurance rather than decoration.

Note that several rules are negative -- an interception is -2 and a lost fumble
is -2 -- so a stat line, and therefore a weekly fantasy score, can be negative.
There is no rule flooring an individual player's weekly total at zero, and the
engine does not impose one (see ``SPEC.md`` §5.5).
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["ScoringRules", "StatLine", "HALF_PPR", "score_statline"]


@dataclass(frozen=True)
class ScoringRules:
    """Points awarded per scoring event."""

    pass_yard: float = 0.04
    pass_td: float = 4.0
    interception: float = -2.0
    rush_yard: float = 0.1
    rush_td: float = 6.0
    rec_yard: float = 0.1
    rec_td: float = 6.0
    reception: float = 0.5
    fumble_lost: float = -2.0
    pass_2pt: float = 2.0
    """Two-point conversion *thrown* by this player."""
    rush_2pt: float = 2.0
    """Two-point conversion *run in* by this player."""
    rec_2pt: float = 2.0
    """Two-point conversion *caught* by this player."""
    special_teams_td: float = 6.0
    """Kick- or punt-return touchdown credited to this individual player.
    (Team/DST scoring does not exist in this league.)"""


HALF_PPR = ScoringRules()


@dataclass(frozen=True)
class StatLine:
    """A single player's box score for one week."""

    pass_yards: float = 0.0
    pass_tds: float = 0.0
    interceptions: float = 0.0
    rush_yards: float = 0.0
    rush_tds: float = 0.0
    rec_yards: float = 0.0
    rec_tds: float = 0.0
    receptions: float = 0.0
    fumbles_lost: float = 0.0
    pass_2pt_conversions: float = 0.0
    rush_2pt_conversions: float = 0.0
    rec_2pt_conversions: float = 0.0
    special_teams_tds: float = 0.0


def score_statline(stats: StatLine, rules: ScoringRules = HALF_PPR) -> float:
    """Fantasy points for ``stats`` under ``rules``.

    The result may be negative: interceptions and lost fumbles both score -2,
    so a quarterback who throws three picks for 90 yards has a negative week.
    """
    return (
        stats.pass_yards * rules.pass_yard
        + stats.pass_tds * rules.pass_td
        + stats.interceptions * rules.interception
        + stats.rush_yards * rules.rush_yard
        + stats.rush_tds * rules.rush_td
        + stats.rec_yards * rules.rec_yard
        + stats.rec_tds * rules.rec_td
        + stats.receptions * rules.reception
        + stats.fumbles_lost * rules.fumble_lost
        + stats.pass_2pt_conversions * rules.pass_2pt
        + stats.rush_2pt_conversions * rules.rush_2pt
        + stats.rec_2pt_conversions * rules.rec_2pt
        + stats.special_teams_tds * rules.special_teams_td
    )
