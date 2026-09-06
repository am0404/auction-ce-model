"""Can this roster still field a legal starting lineup?

The question a roster-position quota cannot answer. This league's eight slots
are

    QB        QB
    RB1 RB2   RB
    WT1..WT3  WR or TE
    FLEX      RB, WR or TE
    SUPERFLEX QB, RB, WR or TE

and a "you need two quarterbacks" rule is nowhere in that list. A team may
start a running back in the superflex and never draft a second quarterback; a
team may roster five quarterbacks. What is *not* legal is a roster that cannot
fill the eight slots at all, and that is a matching question on the eligibility
graph, not a quota.

Because the eligibility sets form a laminar family, Hall's condition collapses
to a handful of counting constraints -- the same argument
:func:`ceauction.lineup.is_feasible` rests on. Here it is applied to a
*complete* roster rather than to a week's available players, so the constraints
are the saturating ones: every slot must be fillable simultaneously.

Nothing here says a roster is *good*. A roster of one quarterback, two running
backs, three receivers and nine more receivers is feasible and probably
terrible. Feasibility is the floor the auction may not go below, not an
objective.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

from ..league import DEFAULT_LEAGUE, LeagueSettings, Position

__all__ = [
    "PositionCounts",
    "lineup_deficit",
    "can_fill_lineup",
    "min_additions_for_lineup",
    "can_complete",
    "completion_shortfall",
]


@dataclass(frozen=True)
class PositionCounts:
    """How many of each position a roster holds.

    ``wt`` deliberately merges WR and TE. This league has no dedicated TE slot,
    so for every lineup question the two are the same position; keeping them
    apart here would invite a tight-end quota that the rules do not contain.
    """

    qb: int = 0
    rb: int = 0
    wr: int = 0
    te: int = 0

    @property
    def wt(self) -> int:
        """WR and TE together: the positions that share the three WT slots."""
        return self.wr + self.te

    @property
    def total(self) -> int:
        return self.qb + self.rb + self.wr + self.te

    @property
    def flex_eligible(self) -> int:
        """Players who may occupy RB1/RB2/WT1-3/FLEX: everyone but a QB."""
        return self.rb + self.wt

    def plus(self, position: Position, n: int = 1) -> "PositionCounts":
        if position is Position.QB:
            return PositionCounts(self.qb + n, self.rb, self.wr, self.te)
        if position is Position.RB:
            return PositionCounts(self.qb, self.rb + n, self.wr, self.te)
        if position is Position.WR:
            return PositionCounts(self.qb, self.rb, self.wr + n, self.te)
        return PositionCounts(self.qb, self.rb, self.wr, self.te + n)

    def to_dict(self) -> Dict[str, int]:
        return {"QB": self.qb, "RB": self.rb, "WR": self.wr, "TE": self.te,
                "total": self.total}

    @classmethod
    def from_positions(cls, positions: Iterable[Position]) -> "PositionCounts":
        c = cls()
        for p in positions:
            c = c.plus(Position(int(p)))
        return c


#: The saturating Hall constraints for filling all eight slots at once, as
#: ``(name, minimum, accessor)``. Each is a slot subset paired with the set of
#: positions that can serve it:
#:
#:   {QB}                      -> 1 QB
#:   {RB1, RB2}                -> 2 RB
#:   {WT1, WT2, WT3}           -> 3 WR/TE
#:   {RB1, RB2, WT1-3, FLEX}   -> 6 RB/WR/TE
#:   all eight                 -> 8 players
#:
#: The intermediate subsets ({RB1,RB2,FLEX} needing 3, and {WT1-3,FLEX}
#: needing 4) are implied by the sixth constraint and add nothing.
_CONSTRAINTS: Tuple[Tuple[str, int, str], ...] = (
    ("QB slot", 1, "qb"),
    ("RB1 and RB2", 2, "rb"),
    ("WT1 through WT3", 3, "wt"),
    ("the six non-QB starting slots", 6, "flex_eligible"),
    ("all eight starting slots", 8, "total"),
)


def lineup_deficit(counts: PositionCounts) -> Dict[str, int]:
    """How far each saturating constraint falls short. Empty means feasible."""
    out: Dict[str, int] = {}
    for name, need, attr in _CONSTRAINTS:
        have = getattr(counts, attr)
        if have < need:
            out[name] = need - have
    return out


def can_fill_lineup(counts: PositionCounts) -> bool:
    """Can these players fill all eight slots simultaneously, all healthy?"""
    return not lineup_deficit(counts)


def min_additions_for_lineup(counts: PositionCounts) -> Tuple[int, PositionCounts]:
    """Fewest players that must still be added, and a witness of what they are.

    The witness is *a* minimal completion, not the only one: any position works
    for the final "all eight" shortfall because the superflex accepts every
    position. It exists so a caller can say which positions a roster still
    needs rather than only that it needs something.
    """
    need_qb = max(0, 1 - counts.qb)
    need_rb = max(0, 2 - counts.rb)
    need_wt = max(0, 3 - counts.wt)

    after = counts.plus(Position.QB, need_qb).plus(Position.RB, need_rb) \
                  .plus(Position.WR, need_wt)
    # Six non-QB starters must exist; any of RB/WR/TE serves, so add receivers.
    extra_flex = max(0, 6 - after.flex_eligible)
    after = after.plus(Position.WR, extra_flex)
    # The superflex takes anyone, so the last shortfall is position-free.
    extra_any = max(0, 8 - after.total)
    after = after.plus(Position.WR, extra_any)

    added = need_qb + need_rb + need_wt + extra_flex + extra_any
    return added, after


def completion_shortfall(
    counts: PositionCounts,
    open_slots: int,
    available: Optional[Mapping[Position, int]] = None,
) -> Optional[str]:
    """Why this roster cannot be completed legally, or ``None`` if it can.

    ``available`` counts what the pool still holds by position. Supplying it
    catches the case a counting argument alone misses: a roster that needs a
    quarterback when none is left is infeasible however many slots are open.
    Omitting it assumes the pool is unconstrained, which is the right default
    mid-auction and the wrong one at the very end.
    """
    if open_slots < 0:
        return f"negative open slots ({open_slots})"
    need, witness = min_additions_for_lineup(counts)
    if need > open_slots:
        missing = lineup_deficit(counts)
        return (f"needs {need} more player(s) to field a legal lineup but only "
                f"{open_slots} roster slot(s) remain; short on "
                + ", ".join(f"{k} (by {v})" for k, v in missing.items()))
    if available is None:
        return None

    # Pool-aware: the minimal witness names concrete positions, so check them.
    want_qb = max(0, 1 - counts.qb)
    want_rb = max(0, 2 - counts.rb)
    want_wt = max(0, 3 - counts.wt) + max(
        0, 6 - counts.plus(Position.RB, want_rb)
                     .plus(Position.WR, max(0, 3 - counts.wt)).flex_eligible)
    have_qb = int(available.get(Position.QB, 0))
    have_rb = int(available.get(Position.RB, 0))
    have_wt = int(available.get(Position.WR, 0)) + int(available.get(Position.TE, 0))
    if want_qb > have_qb:
        return f"needs {want_qb} QB but the pool holds {have_qb}"
    if want_rb > have_rb:
        return f"needs {want_rb} RB but the pool holds {have_rb}"
    if want_wt > have_wt:
        return f"needs {want_wt} WR/TE but the pool holds {have_wt}"
    total_available = have_qb + have_rb + have_wt
    if need > total_available:
        return (f"needs {need} more player(s) but the pool holds "
                f"{total_available}")
    return None


def can_complete(
    counts: PositionCounts,
    open_slots: int,
    available: Optional[Mapping[Position, int]] = None,
) -> bool:
    """Does at least one legal completion of this roster exist?"""
    return completion_shortfall(counts, open_slots, available) is None
