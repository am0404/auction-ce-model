"""Pregame-observable information.

These types carry **everything a manager may legally use to set a lineup** and
**nothing else**.  Note in particular that :class:`PregameEntry` has no field
capable of holding a realized score.  That is deliberate: the information
barrier is enforced at the type level in the scalar API, not merely by
convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Sequence, Tuple

from .league import Position

__all__ = ["Availability", "PregameEntry", "PregameWeek"]


class Availability(str, Enum):
    """Whether a player can be started this week, and why not if he cannot."""

    ACTIVE = "active"
    BYE = "bye"
    INJURED = "injured"

    @property
    def can_start(self) -> bool:
        return self is Availability.ACTIVE


@dataclass(frozen=True)
class PregameEntry:
    """One rostered player, as seen before kickoff.

    Attributes
    ----------
    projection:
        The manager's point estimate for this week.  Posterior mean of the
        player's level given weeks strictly before this one, plus the observed
        role state and contingency status, plus published-projection noise.
    observed_role_delta:
        The part of the player's role change that has been *revealed*.  A role
        change that has happened but not yet been reported is not here.
    contingency_bonus:
        Uplift because a player ahead of him on the depth chart is out.  It is
        part of ``weekly_state`` and is broken out separately only because it
        deserves its own line in an explanation.
    weekly_state:
        Knowable conditions specific to this week -- matchup, expected volume,
        announced usage, weather -- already included in ``projection``.  This
        is what lets two candidates for one lineup spot rotate on information
        available before kickoff rather than after it.
    """

    player_id: int
    name: str
    position: Position
    projection: float
    availability: Availability = Availability.ACTIVE
    observed_role_delta: float = 0.0
    contingency_bonus: float = 0.0
    weekly_state: float = 0.0

    @property
    def startable(self) -> bool:
        return self.availability.can_start


@dataclass(frozen=True)
class PregameWeek:
    """A team's full pregame picture for one week."""

    week: int
    entries: Tuple[PregameEntry, ...]

    def startable(self) -> List[PregameEntry]:
        return [e for e in self.entries if e.startable]

    def by_id(self, player_id: int) -> PregameEntry:
        for e in self.entries:
            if e.player_id == player_id:
                return e
        raise KeyError(player_id)
