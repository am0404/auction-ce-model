"""League settings: roster construction, slots, schedule shape, playoff shape.

Everything here is a *rule* of the league, not a modelling choice.  The
defaults encode the target league (12-team, $200, half-PPR, 15-man rosters,
1QB/2RB/3W-T/FLEX/SUPERFLEX, median-plus-head-to-head standings, 6-team
bracket over weeks 15-17).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, FrozenSet, Tuple

from .scoring import HALF_PPR, ScoringRules

__all__ = [
    "Position",
    "Slot",
    "SLOT_ELIGIBILITY",
    "SLOT_ORDER",
    "LeagueSettings",
    "DEFAULT_LEAGUE",
]


class Position(IntEnum):
    """Fantasy positions in play.  No K, no DST in this league."""

    QB = 0
    RB = 1
    WR = 2
    TE = 3

    @property
    def label(self) -> str:
        return self.name


class Slot(IntEnum):
    """The eight starting lineup slots, in canonical order."""

    QB = 0
    RB1 = 1
    RB2 = 2
    WT1 = 3
    WT2 = 4
    WT3 = 5
    FLEX = 6
    SUPERFLEX = 7

    @property
    def label(self) -> str:
        return {
            Slot.QB: "QB",
            Slot.RB1: "RB1",
            Slot.RB2: "RB2",
            Slot.WT1: "WR/TE-1",
            Slot.WT2: "WR/TE-2",
            Slot.WT3: "WR/TE-3",
            Slot.FLEX: "FLEX (RB/WR/TE)",
            Slot.SUPERFLEX: "SUPERFLEX (QB/RB/WR/TE)",
        }[self]


#: Which positions may legally occupy each slot.
SLOT_ELIGIBILITY: Dict[Slot, FrozenSet[Position]] = {
    Slot.QB: frozenset({Position.QB}),
    Slot.RB1: frozenset({Position.RB}),
    Slot.RB2: frozenset({Position.RB}),
    Slot.WT1: frozenset({Position.WR, Position.TE}),
    Slot.WT2: frozenset({Position.WR, Position.TE}),
    Slot.WT3: frozenset({Position.WR, Position.TE}),
    Slot.FLEX: frozenset({Position.RB, Position.WR, Position.TE}),
    Slot.SUPERFLEX: frozenset({Position.QB, Position.RB, Position.WR, Position.TE}),
}

#: Slots ordered most-restrictive-first.  Because the eligibility sets form a
#: laminar family, filling in this order never blocks a later slot.
SLOT_ORDER: Tuple[Slot, ...] = (
    Slot.QB,
    Slot.RB1,
    Slot.RB2,
    Slot.WT1,
    Slot.WT2,
    Slot.WT3,
    Slot.FLEX,
    Slot.SUPERFLEX,
)


@dataclass(frozen=True)
class LeagueSettings:
    """Immutable league configuration."""

    n_teams: int = 12
    budget: int = 200
    min_bid: int = 1
    roster_size: int = 15
    n_starters: int = 8
    regular_season_weeks: int = 14
    playoff_weeks: Tuple[int, ...] = (15, 16, 17)
    n_playoff_teams: int = 6
    n_byes: int = 2
    median_result_per_week: bool = True
    scoring: ScoringRules = field(default_factory=lambda: HALF_PPR)

    # NFL bye weeks are drawn from this inclusive range by the synthetic
    # generator.  Replace with the real bye schedule when real data lands.
    bye_week_range: Tuple[int, int] = (5, 14)

    def __post_init__(self) -> None:
        if self.roster_size <= self.n_starters:
            raise ValueError("roster_size must exceed n_starters")
        if self.n_teams % 2 != 0:
            raise ValueError("n_teams must be even for a round-robin schedule")
        if self.n_playoff_teams != 6 or self.n_byes != 2:
            raise ValueError(
                "playoffs.run_bracket implements the fixed 6-team / 2-bye bracket only"
            )
        if len(self.playoff_weeks) != 3:
            raise ValueError("this bracket spans exactly three weeks")

    @property
    def bench_size(self) -> int:
        return self.roster_size - self.n_starters

    @property
    def total_weeks(self) -> int:
        """Weeks that must be simulated, regular season plus playoffs."""
        return self.regular_season_weeks + len(self.playoff_weeks)

    @property
    def pool_size(self) -> int:
        """Players consumed by a full league."""
        return self.n_teams * self.roster_size

    @property
    def max_regular_season_wins(self) -> float:
        return 2.0 * self.regular_season_weeks


DEFAULT_LEAGUE = LeagueSettings()
