"""Exact legal lineup selection from pregame projections only.

Why greedy is exact here
------------------------
The slot eligibility sets are

    QB        -> {QB}
    RB1, RB2  -> {RB}
    WT1..WT3  -> {WR, TE}
    FLEX      -> {RB, WR, TE}
    SUPERFLEX -> {QB, RB, WR, TE}

Any two of these sets are either nested or disjoint — a *laminar* family.  The
collection of player sets that can be simultaneously assigned to distinct slots
is a transversal matroid, and for a laminar family its independence test
collapses to seven counting constraints on ``(q, r, t)`` = the number of
selected QBs / RBs / (WRs or TEs)::

    q <= 2                  QB slot + SUPERFLEX
    r <= 4                  RB1 + RB2 + FLEX + SUPERFLEX
    t <= 5                  WT1..3 + FLEX + SUPERFLEX
    q + r <= 5              union of the above neighbourhoods
    q + t <= 6
    r + t <= 7
    q + r + t <= 8          eight slots

Each bound is the size of the neighbourhood of that position group in the
slot graph, so this is exactly Hall's condition.  Because independent sets
form a matroid, the greedy algorithm — sort by projection descending, take each
player whose addition keeps the counts feasible, stop at eight — maximises
total projection **and** simultaneously maximises the number of filled slots.
No LP, no Hungarian algorithm, no heuristic.

``tests/test_lineup.py`` cross-checks this against brute-force enumeration of
all C(15, 8) subsets on randomised inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .league import SLOT_ELIGIBILITY, SLOT_ORDER, Position, Slot
from .pregame import Availability, PregameEntry, PregameWeek

__all__ = [
    "LineupChoice",
    "BenchNote",
    "Lineup",
    "is_feasible",
    "select_lineup",
]

_ORDINAL = ("best", "2nd-best", "3rd-best", "4th-best", "5th-best")


def _ordinal(i: int) -> str:
    return _ORDINAL[i] if i < len(_ORDINAL) else f"{i + 1}th-best"


@dataclass(frozen=True)
class LineupChoice:
    """A started player, the slot he fills, and why he was chosen."""

    slot: Slot
    player_id: int
    name: str
    position: Position
    projection: float
    reason: str


@dataclass(frozen=True)
class BenchNote:
    """A rostered player who did not start, and why not."""

    player_id: int
    name: str
    position: Position
    projection: float
    reason: str


@dataclass(frozen=True)
class Lineup:
    """The chosen lineup for one team in one week."""

    week: int
    choices: Tuple[LineupChoice, ...]
    bench: Tuple[BenchNote, ...]

    @property
    def projected_points(self) -> float:
        return sum(c.projection for c in self.choices)

    @property
    def started_ids(self) -> Tuple[int, ...]:
        return tuple(c.player_id for c in self.choices)

    @property
    def filled_slots(self) -> int:
        return len(self.choices)

    def slot_of(self, player_id: int) -> Optional[Slot]:
        for c in self.choices:
            if c.player_id == player_id:
                return c.slot
        return None

    def explain(self) -> List[str]:
        lines = [f"Week {self.week}: {self.filled_slots}/8 slots filled, "
                 f"{self.projected_points:.2f} projected"]
        for c in self.choices:
            lines.append(
                f"  {c.slot.label:<24} {c.position.label:<3} {c.name:<22} "
                f"{c.projection:6.2f}   {c.reason}"
            )
        for b in self.bench:
            lines.append(
                f"  {'BENCH':<24} {b.position.label:<3} {b.name:<22} "
                f"{b.projection:6.2f}   {b.reason}"
            )
        return lines


def is_feasible(q: int, r: int, t: int) -> bool:
    """Hall's condition for the laminar slot family.  See the module docstring."""
    return (
        q <= 2
        and r <= 4
        and t <= 5
        and q + r <= 5
        and q + t <= 6
        and r + t <= 7
        and q + r + t <= 8
    )


def _counts_delta(position: Position) -> Tuple[int, int, int]:
    if position is Position.QB:
        return (1, 0, 0)
    if position is Position.RB:
        return (0, 1, 0)
    return (0, 0, 1)


def select_lineup(pregame: PregameWeek) -> Lineup:
    """Choose the highest-projected legal lineup.

    Uses ``pregame`` and nothing else.  There is no argument through which a
    realized score could reach this function.

    Ties in projection are broken by roster order, which is stable and
    deterministic, so repeated runs return identical lineups.
    """
    entries = list(pregame.entries)
    # Stable sort on projection descending; the enumerate index keeps ties
    # resolving in roster order.
    ranked = sorted(
        ((e, i) for i, e in enumerate(entries)),
        key=lambda pair: (-pair[0].projection, pair[1]),
    )

    q = r = t = 0
    chosen: List[PregameEntry] = []
    bench_reasons: Dict[int, str] = {}
    seen_by_position: Dict[Position, int] = {p: 0 for p in Position}

    for entry, _ in ranked:
        if not entry.startable:
            bench_reasons[entry.player_id] = (
                "on bye" if entry.availability is Availability.BYE else "injured / unavailable"
            )
            continue
        dq, dr, dt = _counts_delta(entry.position)
        if len(chosen) >= 8:
            bench_reasons[entry.player_id] = (
                f"out-projected: 8 slots already filled by higher projections"
            )
            continue
        if is_feasible(q + dq, r + dr, t + dt):
            q, r, t = q + dq, r + dr, t + dt
            chosen.append(entry)
            seen_by_position[entry.position] += 1
        else:
            bench_reasons[entry.player_id] = (
                f"slot-blocked: no legal slot left for a {entry.position.label}"
            )

    choices = _assign_slots(chosen)
    started = {c.player_id for c in choices}
    for entry in entries:
        if entry.player_id in started:
            continue
        bench_reasons.setdefault(entry.player_id, "out-projected")

    bench = tuple(
        BenchNote(
            player_id=e.player_id,
            name=e.name,
            position=e.position,
            projection=e.projection,
            reason=bench_reasons[e.player_id],
        )
        for e in entries
        if e.player_id not in started
    )
    return Lineup(week=pregame.week, choices=tuple(choices), bench=bench)


def _assign_slots(chosen: Sequence[PregameEntry]) -> List[LineupChoice]:
    """Assign a feasible player set to concrete slots, most-restrictive first.

    Valid because the eligibility family is laminar: a slot with the smallest
    eligible pool can always be filled first without stranding a later slot.
    """
    remaining = sorted(chosen, key=lambda e: -e.projection)
    used: set = set()
    out: List[LineupChoice] = []
    rank_within_pool: Dict[str, int] = {}

    for slot in SLOT_ORDER:
        eligible = SLOT_ELIGIBILITY[slot]
        pick = None
        for e in remaining:
            if e.player_id in used:
                continue
            if e.position in eligible:
                pick = e
                break
        if pick is None:
            continue
        used.add(pick.player_id)
        pool_key = slot.name.rstrip("0123456789")
        k = rank_within_pool.get(pool_key, 0)
        rank_within_pool[pool_key] = k + 1
        out.append(
            LineupChoice(
                slot=slot,
                player_id=pick.player_id,
                name=pick.name,
                position=pick.position,
                projection=pick.projection,
                reason=_reason(slot, pick, k),
            )
        )
    return out


def _reason(slot: Slot, entry: PregameEntry, rank_in_pool: int) -> str:
    bits: List[str] = []
    if slot is Slot.QB:
        bits.append("best available QB")
    elif slot in (Slot.RB1, Slot.RB2):
        bits.append(f"{_ordinal(rank_in_pool)} available RB")
    elif slot in (Slot.WT1, Slot.WT2, Slot.WT3):
        bits.append(f"{_ordinal(rank_in_pool)} available WR/TE")
    elif slot is Slot.FLEX:
        bits.append("best remaining RB/WR/TE")
    elif slot is Slot.SUPERFLEX:
        bits.append("best remaining player of any position")
        bits.append("non-QB in superflex" if entry.position is not Position.QB
                    else "2nd QB in superflex")
    if entry.observed_role_delta:
        bits.append(f"observed role change {entry.observed_role_delta:+.1f}")
    if entry.contingency_bonus:
        bits.append(f"contingency start {entry.contingency_bonus:+.1f}")
    # The contingency uplift is itself part of weekly_state, so report only the
    # rest of it -- otherwise a handcuff week would read its bonus twice.
    rest = entry.weekly_state - entry.contingency_bonus
    if abs(rest) > 5e-2:
        bits.append(f"weekly conditions {rest:+.1f}")
    return "; ".join(bits) + f" ({entry.projection:.2f} proj)"
