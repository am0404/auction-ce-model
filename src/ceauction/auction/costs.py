"""What the rest of the board will cost, and where that number came from.

A completion search cannot run without an assumed price for every player it
might buy. Those prices are *not* in this repository, and the ones that exist
nearby are for a different league: a 10-team half-PPR market file prices a
different scarcity, a different roster shape and a different superflex rule
than a 12-team superflex league. Treating them as this league's prices would
be the single easiest way to produce a confident wrong answer.

So every cost carries its own provenance, and the provenance is a required
field rather than a comment. A cost is one of:

``REAL``          observed in this league, this format, this season.
``TRANSFORMED``   derived from real data for a different format, with the
                  transform named. Still not this league's prices.
``PROVISIONAL``   a placeholder a human chose, for exploring the machinery.
``FABRICATED``    invented for a test. May never leave a test.

Anything a ``PROVISIONAL`` or ``FABRICATED`` book produces is labelled as such
all the way to the output, and :meth:`CostBook.disclaimer` is the sentence a
report has to print.

**Missing costs are refused, not defaulted.** Filling a gap with $1 so the
solver runs would quietly tell the search that an unpriced star is free, and
the search would take every one of them. A caller who genuinely wants a
fallback has to name it.

Acquisition cost and on-field performance are kept in separate objects on
purpose. They are different quantities from different sources with different
error, and a single record holding both invites the assumption that a player
who costs more is projected for more.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Dict, Iterable, Mapping, Optional, Tuple

__all__ = [
    "CostProvenance",
    "PROVENANCE_LEVELS",
    "CostEntry",
    "CostBook",
    "MissingCost",
    "flat_cost_book",
]


class MissingCost(KeyError):
    """A player the search may buy has no assumed cost.

    Raised rather than defaulted. See the module docstring.
    """


#: Ordered weakest-to-strongest claim about where a number came from. A book's
#: overall standing is its *weakest* entry, because one fabricated price in a
#: real book makes the whole result fabricated.
PROVENANCE_LEVELS: Tuple[str, ...] = ("FABRICATED", "PROVISIONAL",
                                      "TRANSFORMED", "REAL")


@dataclass(frozen=True)
class CostProvenance:
    """Where a set of costs came from, and what may be claimed for them."""

    level: str
    source: str
    """Free text naming the file, market or person. Required."""
    scenario_id: Optional[str] = None
    """Model scenario these costs were derived under, when they were."""
    room_state: Optional[str] = None
    """Auction-state fingerprint they were derived under, when applicable.

    Real auction prices drift with the room -- money left, positions gone -- so
    a cost book fitted mid-auction is only valid for the state it was fitted
    to. Recording that state is what lets a cache refuse to reuse it."""
    version: Optional[str] = None
    generated_at: Optional[str] = None
    notes: str = ""

    def __post_init__(self) -> None:
        if self.level not in PROVENANCE_LEVELS:
            raise ValueError(
                f"unknown provenance level {self.level!r}; "
                f"expected one of {PROVENANCE_LEVELS}")
        if not self.source or not self.source.strip():
            raise ValueError(
                "a cost source must be named; an unattributed price is "
                "indistinguishable from an invented one")

    @property
    def is_real(self) -> bool:
        return self.level == "REAL"

    @property
    def may_be_reported_as_a_value(self) -> bool:
        """Only genuinely real costs can support a claim about real value."""
        return self.level == "REAL"

    def to_dict(self) -> Dict[str, object]:
        return {"level": self.level, "source": self.source,
                "scenario_id": self.scenario_id, "room_state": self.room_state,
                "version": self.version, "generated_at": self.generated_at,
                "notes": self.notes}


@dataclass(frozen=True)
class CostEntry:
    """One player's assumed acquisition cost.

    ``cost`` is whole auction dollars, because bids are. ``low``/``high`` carry
    a range when the source has one; they are reported, never averaged into a
    point behind the caller's back.
    """

    player_id: int
    cost: int
    low: Optional[int] = None
    high: Optional[int] = None

    def __post_init__(self) -> None:
        for name, v in (("cost", self.cost), ("low", self.low), ("high", self.high)):
            if v is None:
                continue
            if not isinstance(v, int) or isinstance(v, bool):
                raise ValueError(f"{name} must be whole auction dollars; got {v!r}")
        if self.cost < 0:
            raise ValueError(f"negative cost {self.cost}")
        if self.low is not None and self.high is not None and self.low > self.high:
            raise ValueError(f"low {self.low} exceeds high {self.high}")

    @property
    def has_range(self) -> bool:
        return self.low is not None or self.high is not None

    def to_dict(self) -> Dict[str, object]:
        return {"player_id": self.player_id, "cost": self.cost,
                "low": self.low, "high": self.high}


@dataclass(frozen=True)
class CostBook:
    """Assumed acquisition costs for a set of players, with their provenance."""

    entries: Tuple[CostEntry, ...]
    provenance: CostProvenance
    minimum_cost: int = 1
    """The league's minimum winning bid. A cost below it is raised to it, since
    no player can be bought for less; that is a rule, not an assumption."""

    def __post_init__(self) -> None:
        ids = [e.player_id for e in self.entries]
        if len(set(ids)) != len(ids):
            dupes = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(f"duplicate cost entries for player(s) {dupes[:5]}")

    # --- lookup ------------------------------------------------------------

    @property
    def by_id(self) -> Dict[int, CostEntry]:
        return {e.player_id: e for e in self.entries}

    def __len__(self) -> int:
        return len(self.entries)

    def __contains__(self, player_id: int) -> bool:
        return player_id in self.by_id

    def cost_of(self, player_id: int, default: Optional[int] = None) -> int:
        """This player's assumed cost.

        ``default`` is the only way to get a fallback, and passing it is the
        caller stating on the record that an unpriced player is worth that
        much. There is no implicit $1.
        """
        entry = self.by_id.get(player_id)
        if entry is None:
            if default is None:
                raise MissingCost(
                    f"player {player_id} has no assumed acquisition cost in "
                    f"cost book {self.provenance.source!r}; supply one, or pass "
                    f"an explicit default to state what an unpriced player is "
                    f"assumed to cost")
            if not isinstance(default, int) or isinstance(default, bool):
                raise ValueError(f"default cost must be an integer; got {default!r}")
            return max(default, self.minimum_cost)
        return max(entry.cost, self.minimum_cost)

    def costs_for(self, player_ids: Iterable[int],
                  default: Optional[int] = None) -> Dict[int, int]:
        """Costs for many players, reporting *all* gaps rather than the first."""
        ids = list(player_ids)
        if default is None:
            missing = sorted(p for p in ids if p not in self.by_id)
            if missing:
                raise MissingCost(
                    f"{len(missing)} of {len(ids)} players have no assumed cost "
                    f"in {self.provenance.source!r} (first few: {missing[:5]}); "
                    f"supply them, or pass an explicit default")
        return {p: self.cost_of(p, default) for p in ids}

    def covers(self, player_ids: Iterable[int]) -> bool:
        return all(p in self.by_id for p in player_ids)

    def missing_from(self, player_ids: Iterable[int]) -> Tuple[int, ...]:
        return tuple(sorted(p for p in player_ids if p not in self.by_id))

    # --- editing -----------------------------------------------------------

    def restricted_to(self, player_ids: Iterable[int]) -> "CostBook":
        keep = set(player_ids)
        return replace(self, entries=tuple(e for e in self.entries
                                           if e.player_id in keep))

    def with_costs(self, costs: Mapping[int, int],
                   provenance: Optional[CostProvenance] = None) -> "CostBook":
        """Override or add costs.

        A provenance must be supplied whenever the new costs are weaker than
        the book's own, and the result takes the weaker of the two: adding one
        fabricated price to a real book does not leave a real book.
        """
        merged = {e.player_id: e for e in self.entries}
        for pid, c in costs.items():
            merged[pid] = CostEntry(pid, int(c))
        prov = provenance or self.provenance
        weaker = min(prov, self.provenance,
                     key=lambda p: PROVENANCE_LEVELS.index(p.level))
        if weaker is not prov and provenance is not None:
            prov = replace(prov, level=weaker.level,
                           notes=(prov.notes + " | downgraded to "
                                  f"{weaker.level} by merge").strip(" |"))
        elif provenance is None and prov.level != self.provenance.level:
            prov = self.provenance
        return replace(self, entries=tuple(
            merged[k] for k in sorted(merged)), provenance=prov)

    # --- reporting ---------------------------------------------------------

    @property
    def level(self) -> str:
        return self.provenance.level

    @property
    def total_assumed_cost(self) -> int:
        return sum(self.cost_of(e.player_id) for e in self.entries)

    def disclaimer(self) -> str:
        """The sentence any report using this book has to print."""
        if self.provenance.level == "REAL":
            return (f"Acquisition costs are REAL, from {self.provenance.source}. "
                    f"They are still assumptions about a future auction.")
        if self.provenance.level == "TRANSFORMED":
            return (f"Acquisition costs are TRANSFORMED from "
                    f"{self.provenance.source} and describe a DIFFERENT format. "
                    f"Nothing derived from them is this league's market price.")
        if self.provenance.level == "PROVISIONAL":
            return (f"Acquisition costs are PROVISIONAL placeholders from "
                    f"{self.provenance.source}. Nothing derived from them is a "
                    f"real player value or a market price.")
        return (f"Acquisition costs are FABRICATED ({self.provenance.source}) "
                f"and exist only to exercise the machinery. Nothing derived "
                f"from them describes any real player.")

    def summary(self) -> Dict[str, object]:
        costs = [self.cost_of(e.player_id) for e in self.entries]
        costs.sort()
        return {
            "n_entries": len(self.entries),
            "provenance": self.provenance.to_dict(),
            "level": self.level,
            "may_be_reported_as_a_value": self.provenance.may_be_reported_as_a_value,
            "minimum_cost": self.minimum_cost,
            "total_assumed_cost": self.total_assumed_cost,
            "cost_min": costs[0] if costs else None,
            "cost_median": costs[len(costs) // 2] if costs else None,
            "cost_max": costs[-1] if costs else None,
            "entries_with_a_range": sum(1 for e in self.entries if e.has_range),
            "disclaimer": self.disclaimer(),
        }

    def to_dict(self) -> Dict[str, object]:
        return {"provenance": self.provenance.to_dict(),
                "minimum_cost": self.minimum_cost,
                "entries": [e.to_dict() for e in self.entries]}

    @classmethod
    def from_dict(cls, blob: Mapping[str, object]) -> "CostBook":
        prov = dict(blob["provenance"])  # type: ignore[index]
        return cls(
            entries=tuple(CostEntry(**e) for e in blob["entries"]),  # type: ignore[index]
            provenance=CostProvenance(**prov),
            minimum_cost=int(blob.get("minimum_cost", 1)))  # type: ignore[arg-type]

    def write_json(self, path) -> None:
        from pathlib import Path
        Path(path).write_text(json.dumps(self.to_dict(), indent=2),
                              encoding="utf-8")

    @classmethod
    def read_json(cls, path) -> "CostBook":
        from pathlib import Path
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def flat_cost_book(player_ids: Iterable[int], cost: int, source: str,
                   level: str = "FABRICATED", **provenance) -> CostBook:
    """Every player at the same price. For tests and for machinery smoke runs.

    Defaults to ``FABRICATED`` deliberately: a flat price is not a market and
    nothing derived from one may be presented as a value.
    """
    return CostBook(
        entries=tuple(CostEntry(int(p), int(cost)) for p in sorted(set(player_ids))),
        provenance=CostProvenance(
            level=level, source=source,
            generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **provenance))
