"""The live auction room, as an immutable value.

Everything here is a *fact about the room*, never a prediction about it. The
state knows what every owner has spent, what each of them may still legally
bid, and which of them could take the next dollar. It contains no model of who
*will* bid, and adding one here would be a category error: a behavioural guess
stored beside a rule would eventually be read as a rule.

Three things this deliberately does not do.

**No quarterback maximum.** The league has none. A roster of five quarterbacks
is legal and a roster with one is legal, because the superflex accepts any
position. Feasibility is a matching question on the eligibility graph -- see
:mod:`ceauction.auction.feasibility` -- and never a positional quota.

**No mutation.** Every transition returns a new state. An auction is a
sequence of counterfactuals ("what if I buy him, what if he goes to Real07")
and a mutable room would make each of those a chance to corrupt the real one.

**No behavioural model.** ``owners_who_can_bid`` answers who is *permitted* to
bid, from budget, open slots and feasibility. Who *would* is not modelled and
is not guessed.

Opponent budgets are load-bearing, not decoration. The interface is built so
later logic can tell apart an owner who wants a player but cannot legally bid,
one who can bid exactly one more dollar, one with enough left to control the
endgame, and one whose remaining roster capacity makes the player unattractive.
Reducing the room to "dollars left in the league" would erase all four.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import (Dict, FrozenSet, Iterable, List, Mapping, Optional,
                    Sequence, Tuple)

from ..league import DEFAULT_LEAGUE, LeagueSettings, Position
from ..players import PlayerSpec
from .feasibility import (PositionCounts, can_complete, can_fill_lineup,
                          completion_shortfall, min_additions_for_lineup)

__all__ = [
    "AuctionError",
    "AuctionRuleError",
    "AuctionStateInvalid",
    "RosterSlotFilled",
    "Transaction",
    "Nomination",
    "OwnerAuctionState",
    "AuctionState",
    "new_auction",
]


class AuctionError(Exception):
    """Base for everything this module refuses to do."""


class AuctionRuleError(AuctionError):
    """A transition the league's rules do not permit."""


class AuctionStateInvalid(AuctionError):
    """A state that violates an invariant. Carries every violation found."""

    def __init__(self, problems: Sequence[str]):
        self.problems = list(problems)
        super().__init__(
            f"{len(self.problems)} invariant violation(s): "
            + "; ".join(self.problems))


# ---------------------------------------------------------------------------
# Pieces
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RosterSlotFilled:
    """One player an owner holds, and what he cost."""

    player_id: int
    position: Position
    price: int

    def __post_init__(self) -> None:
        if not isinstance(self.price, int) or isinstance(self.price, bool):
            raise AuctionRuleError(
                f"auction dollars are integers; got {self.price!r}")
        if self.price < 0:
            raise AuctionRuleError(f"negative price {self.price}")


@dataclass(frozen=True)
class Transaction:
    """A completed purchase, in room order."""

    sequence: int
    player_id: int
    owner_id: str
    price: int
    nominated_by: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return {"sequence": self.sequence, "player_id": self.player_id,
                "owner_id": self.owner_id, "price": self.price,
                "nominated_by": self.nominated_by}


@dataclass(frozen=True)
class Nomination:
    """A player on the block, and where the bidding stands."""

    player_id: int
    nominated_by: str
    current_bid: Optional[int] = None
    high_bidder: Optional[str] = None

    def __post_init__(self) -> None:
        if (self.current_bid is None) != (self.high_bidder is None):
            raise AuctionRuleError(
                "a nomination has either both a current bid and a high bidder, "
                "or neither")
        if self.current_bid is not None and self.current_bid < 0:
            raise AuctionRuleError(f"negative bid {self.current_bid}")

    def to_dict(self) -> Dict[str, object]:
        return {"player_id": self.player_id, "nominated_by": self.nominated_by,
                "current_bid": self.current_bid, "high_bidder": self.high_bidder}


@dataclass(frozen=True)
class OwnerAuctionState:
    """One owner's money, roster and legal room, at a moment.

    Every derived quantity is a property rather than a stored field, so a state
    cannot be constructed whose summary disagrees with its own contents.
    """

    owner_id: str
    team_name: str
    budget_start: int
    roster_capacity: int
    filled: Tuple[RosterSlotFilled, ...] = ()
    min_bid: int = 1

    def __post_init__(self) -> None:
        if not self.owner_id:
            raise AuctionRuleError("an owner needs an id")
        if self.budget_start < 0:
            raise AuctionRuleError(f"{self.owner_id}: negative starting budget")
        if self.roster_capacity <= 0:
            raise AuctionRuleError(f"{self.owner_id}: non-positive roster capacity")
        ids = [f.player_id for f in self.filled]
        if len(set(ids)) != len(ids):
            raise AuctionRuleError(f"{self.owner_id} holds a player twice")

    # --- roster ------------------------------------------------------------

    @property
    def player_ids(self) -> Tuple[int, ...]:
        return tuple(f.player_id for f in self.filled)

    @property
    def n_players(self) -> int:
        return len(self.filled)

    @property
    def open_slots(self) -> int:
        return self.roster_capacity - self.n_players

    @property
    def is_full(self) -> bool:
        return self.open_slots == 0

    @property
    def counts(self) -> PositionCounts:
        """Position counts. **Reporting only** -- no rule reads them as a quota."""
        return PositionCounts.from_positions(f.position for f in self.filled)

    # --- money -------------------------------------------------------------

    @property
    def spent(self) -> int:
        return sum(f.price for f in self.filled)

    @property
    def budget_remaining(self) -> int:
        return self.budget_start - self.spent

    @property
    def reserve_for_open_slots(self) -> int:
        """Dollars that must survive to fill every open slot at the minimum.

        This is the reserve when *not* bidding. Bidding on one of those slots
        releases exactly one minimum bid from it, which is why
        :attr:`max_bid` is a dollar above :attr:`discretionary`.
        """
        return self.open_slots * self.min_bid

    @property
    def discretionary(self) -> int:
        """Money above the floor: what can be spent in excess of the minimum.

        The honest measure of an owner's remaining power. Two owners with the
        same budget and different roster capacity do not have the same room.
        """
        return self.budget_remaining - self.reserve_for_open_slots

    @property
    def max_bid(self) -> int:
        """The greatest legal bid: ``budget_remaining - (open_slots - 1)``.

        Zero when the roster is full -- an owner with nowhere to put a player
        cannot bid at all, whatever his budget says.
        """
        if self.open_slots <= 0:
            return 0
        return self.budget_remaining - (self.open_slots - 1) * self.min_bid

    def can_bid(self, amount: int) -> bool:
        return (self.open_slots > 0 and amount >= self.min_bid
                and amount <= self.max_bid)

    # --- feasibility -------------------------------------------------------

    def can_still_field_lineup(
        self, available: Optional[Mapping[Position, int]] = None) -> bool:
        """Does at least one legal completion of this roster still exist?"""
        return can_complete(self.counts, self.open_slots, available)

    def lineup_shortfall(
        self, available: Optional[Mapping[Position, int]] = None) -> Optional[str]:
        return completion_shortfall(self.counts, self.open_slots, available)

    @property
    def fields_a_full_lineup(self) -> bool:
        """Can the roster *as it stands* fill all eight slots?

        False mid-auction is normal and not a defect; it only matters once the
        roster is full.
        """
        return can_fill_lineup(self.counts)

    # --- transitions -------------------------------------------------------

    def with_player(self, player_id: int, position: Position,
                    price: int) -> "OwnerAuctionState":
        return replace(self, filled=self.filled + (
            RosterSlotFilled(player_id, Position(int(position)), int(price)),))

    def without_player(self, player_id: int) -> "OwnerAuctionState":
        if player_id not in self.player_ids:
            raise AuctionRuleError(f"{self.owner_id} does not hold {player_id}")
        return replace(self, filled=tuple(
            f for f in self.filled if f.player_id != player_id))

    # --- reporting ---------------------------------------------------------

    def to_dict(self) -> Dict[str, object]:
        return {
            "owner_id": self.owner_id, "team_name": self.team_name,
            "budget_start": self.budget_start,
            "budget_remaining": self.budget_remaining,
            "spent": self.spent,
            "n_players": self.n_players, "open_slots": self.open_slots,
            "reserve_for_open_slots": self.reserve_for_open_slots,
            "discretionary": self.discretionary,
            "max_bid": self.max_bid,
            "position_counts": self.counts.to_dict(),
            "can_still_field_lineup": self.can_still_field_lineup(),
            "player_ids": list(self.player_ids),
        }


# ---------------------------------------------------------------------------
# The room
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AuctionState:
    """The whole room. Immutable; every transition returns a new one."""

    settings: LeagueSettings
    pool: Tuple[PlayerSpec, ...]
    """Every player the auction may allocate -- drafted and undrafted alike.

    Kept whole rather than shrinking as players are bought, so a state always
    knows the position and identity of a player any owner holds.
    """

    owners: Tuple[OwnerAuctionState, ...]
    focus_owner_id: str
    transactions: Tuple[Transaction, ...] = ()
    withdrawn: FrozenSet[int] = frozenset()
    """Players removed from consideration without being awarded."""
    nomination: Optional[Nomination] = None

    # --- lookups -----------------------------------------------------------

    def __post_init__(self) -> None:
        if not self.owners:
            raise AuctionRuleError("an auction needs owners")
        ids = [o.owner_id for o in self.owners]
        if len(set(ids)) != len(ids):
            raise AuctionRuleError("owner ids must be unique")
        if self.focus_owner_id not in ids:
            raise AuctionRuleError(
                f"focus owner {self.focus_owner_id!r} is not in the room")

    @property
    def spec_by_id(self) -> Dict[int, PlayerSpec]:
        return {s.player_id: s for s in self.pool}

    @property
    def owner_by_id(self) -> Dict[str, OwnerAuctionState]:
        return {o.owner_id: o for o in self.owners}

    def owner(self, owner_id: str) -> OwnerAuctionState:
        try:
            return self.owner_by_id[owner_id]
        except KeyError:
            raise AuctionRuleError(f"no owner {owner_id!r} in the room") from None

    @property
    def focus(self) -> OwnerAuctionState:
        return self.owner(self.focus_owner_id)

    def spec(self, player_id: int) -> PlayerSpec:
        try:
            return self.spec_by_id[player_id]
        except KeyError:
            raise AuctionRuleError(
                f"player {player_id} is not in this auction's pool") from None

    @property
    def rostered_ids(self) -> FrozenSet[int]:
        return frozenset(pid for o in self.owners for pid in o.player_ids)

    @property
    def owner_of(self) -> Dict[int, str]:
        return {pid: o.owner_id for o in self.owners for pid in o.player_ids}

    @property
    def available_ids(self) -> Tuple[int, ...]:
        """Pool order, minus everyone rostered or withdrawn."""
        gone = self.rostered_ids | self.withdrawn
        return tuple(s.player_id for s in self.pool if s.player_id not in gone)

    @property
    def available_specs(self) -> Tuple[PlayerSpec, ...]:
        gone = self.rostered_ids | self.withdrawn
        return tuple(s for s in self.pool if s.player_id not in gone)

    def available_by_position(self) -> Dict[Position, int]:
        out: Dict[Position, int] = {p: 0 for p in Position}
        for s in self.available_specs:
            out[Position(int(s.position))] += 1
        return out

    def is_available(self, player_id: int) -> bool:
        return player_id in set(self.available_ids)

    # --- money and legality ------------------------------------------------

    def max_bid(self, owner_id: str) -> int:
        """The greatest dollar this owner may legally bid right now."""
        return self.owner(owner_id).max_bid

    def owners_who_can_bid(self, amount: int,
                           exclude: Sequence[str] = ()) -> Tuple[OwnerAuctionState, ...]:
        """Everyone *permitted* to bid ``amount``. Not everyone who would.

        Permission is budget, an open roster slot, and a roster that still has
        a legal completion after the purchase. Nothing here models desire.
        """
        skip = set(exclude)
        out = []
        for o in self.owners:
            if o.owner_id in skip or not o.can_bid(amount):
                continue
            out.append(o)
        return tuple(out)

    def bid_capacity(self, amount: int) -> Dict[str, object]:
        """Who could take ``amount``, and why each of the others could not.

        Room pressure is not "dollars left in the league": an owner with $80
        and a full roster exerts none, and an owner with $12 and one slot can
        still take a player to $12.
        """
        able, blocked = [], {}
        for o in self.owners:
            if o.can_bid(amount):
                able.append(o.owner_id)
            elif o.open_slots <= 0:
                blocked[o.owner_id] = "roster full"
            elif amount > o.max_bid:
                blocked[o.owner_id] = (
                    f"max bid {o.max_bid} (${o.budget_remaining} less "
                    f"${o.reserve_for_open_slots - o.min_bid} reserved for "
                    f"{o.open_slots - 1} other open slot(s))")
            else:
                blocked[o.owner_id] = f"bid below the {o.min_bid} minimum"
        return {"amount": amount, "able": able, "blocked": blocked,
                "n_able": len(able)}

    def purchase_shortfall(self, player_id: int, owner_id: str,
                           price: int) -> Optional[str]:
        """Why this purchase is illegal, or ``None`` if it is legal.

        Checked in the order a human would object: does the player exist, is he
        available, has the owner room, is the money legal, and does the roster
        still have a legal completion afterwards.
        """
        if not isinstance(price, int) or isinstance(price, bool):
            return f"auction dollars are integers; got {price!r}"
        spec = self.spec_by_id.get(player_id)
        if spec is None:
            return f"player {player_id} is not in this auction's pool"
        if player_id in self.withdrawn:
            return f"player {player_id} was withdrawn from the board"
        held_by = self.owner_of.get(player_id)
        if held_by is not None:
            return f"player {player_id} already belongs to {held_by}"
        o = self.owner_by_id.get(owner_id)
        if o is None:
            return f"no owner {owner_id!r} in the room"
        if o.open_slots <= 0:
            return f"{owner_id} has no open roster slot"
        if price < o.min_bid:
            return f"price {price} is below the {o.min_bid} minimum bid"
        if price > o.max_bid:
            return (f"price {price} exceeds {owner_id}'s legal maximum "
                    f"{o.max_bid} (${o.budget_remaining} remaining, "
                    f"{o.open_slots - 1} other slot(s) to reserve for)")
        after = o.with_player(player_id, spec.position, price)
        # Pool-aware: what would still be on the board once this player is gone.
        remaining = self.available_by_position()
        remaining[Position(int(spec.position))] -= 1
        short = after.lineup_shortfall(remaining)
        if short is not None:
            return f"{owner_id} could not complete a legal roster afterwards: {short}"
        return None

    def purchase_is_legal(self, player_id: int, owner_id: str,
                          price: int) -> bool:
        return self.purchase_shortfall(player_id, owner_id, price) is None

    def purchase_leaves_feasible_completion(self, player_id: int, owner_id: str,
                                            price: int) -> bool:
        """Does a legal completion survive this purchase?

        Narrower than :meth:`purchase_is_legal`: it asks only the feasibility
        question, so a caller exploring prices is told *why* a price fails
        rather than only that it does.
        """
        spec = self.spec(player_id)
        o = self.owner(owner_id)
        if o.open_slots <= 0:
            return False
        after = o.with_player(player_id, spec.position, max(price, 0))
        remaining = self.available_by_position()
        remaining[Position(int(spec.position))] -= 1
        return after.can_still_field_lineup(remaining)

    # --- transitions -------------------------------------------------------

    def _replace_owner(self, owner: OwnerAuctionState) -> "AuctionState":
        return replace(self, owners=tuple(
            owner if o.owner_id == owner.owner_id else o for o in self.owners))

    def apply_purchase(self, player_id: int, owner_id: str, price: int,
                       nominated_by: Optional[str] = None,
                       validate: bool = True) -> "AuctionState":
        """Award a player. Returns a new state; this one is untouched."""
        if validate:
            problem = self.purchase_shortfall(player_id, owner_id, price)
            if problem is not None:
                raise AuctionRuleError(problem)
        spec = self.spec(player_id)
        owner = self.owner(owner_id).with_player(player_id, spec.position,
                                                 int(price))
        txn = Transaction(
            sequence=len(self.transactions), player_id=player_id,
            owner_id=owner_id, price=int(price),
            nominated_by=nominated_by or (
                self.nomination.nominated_by
                if self.nomination and self.nomination.player_id == player_id
                else None))
        state = self._replace_owner(owner)
        nomination = (None if self.nomination
                      and self.nomination.player_id == player_id
                      else self.nomination)
        return replace(state, transactions=self.transactions + (txn,),
                       nomination=nomination)

    def award_to_rival(self, player_id: int, owner_id: str,
                       price: int) -> "AuctionState":
        """Force a player to a named rival, for a counterfactual branch.

        Identical to :meth:`apply_purchase` and separately named because a
        counterfactual is not a record of the room. Calling this on the state
        you believe in is a mistake the name is meant to make visible.
        """
        if owner_id == self.focus_owner_id:
            raise AuctionRuleError(
                "award_to_rival is for opponents; use apply_purchase for the "
                "focus owner")
        return self.apply_purchase(player_id, owner_id, price)

    def withdraw(self, player_id: int) -> "AuctionState":
        """Remove a player from consideration without awarding him.

        The pass branch's ``unavailable`` destination: nobody gets him, and
        crucially he stops being one of *our* alternatives.
        """
        if player_id in self.rostered_ids:
            raise AuctionRuleError(
                f"player {player_id} is already rostered and cannot be withdrawn")
        self.spec(player_id)
        nomination = (None if self.nomination
                      and self.nomination.player_id == player_id
                      else self.nomination)
        return replace(self, withdrawn=self.withdrawn | {player_id},
                       nomination=nomination)

    def nominate(self, player_id: int, by_owner: str,
                 opening_bid: Optional[int] = None) -> "AuctionState":
        """Put a player on the block."""
        if not self.is_available(player_id):
            raise AuctionRuleError(
                f"player {player_id} is not available to nominate")
        self.owner(by_owner)
        nom = Nomination(player_id=player_id, nominated_by=by_owner)
        state = replace(self, nomination=nom)
        if opening_bid is not None:
            state = state.place_bid(by_owner, opening_bid)
        return state

    def place_bid(self, owner_id: str, amount: int) -> "AuctionState":
        """Record a bid on the current nomination."""
        if self.nomination is None:
            raise AuctionRuleError("no player is nominated")
        if not isinstance(amount, int) or isinstance(amount, bool):
            raise AuctionRuleError(f"auction dollars are integers; got {amount!r}")
        o = self.owner(owner_id)
        cur = self.nomination.current_bid
        if cur is not None and amount <= cur:
            raise AuctionRuleError(
                f"bid {amount} does not beat the standing bid {cur}")
        if not o.can_bid(amount):
            raise AuctionRuleError(
                f"{owner_id} cannot legally bid {amount} "
                f"(max {o.max_bid}, {o.open_slots} open slot(s))")
        return replace(self, nomination=replace(
            self.nomination, current_bid=amount, high_bidder=owner_id))

    def clear_nomination(self) -> "AuctionState":
        return replace(self, nomination=None)

    def award_nomination(self) -> "AuctionState":
        """Sell the nominated player to the standing high bidder."""
        if self.nomination is None or self.nomination.high_bidder is None:
            raise AuctionRuleError("no standing bid to award")
        return self.apply_purchase(
            self.nomination.player_id, self.nomination.high_bidder,
            int(self.nomination.current_bid),
            nominated_by=self.nomination.nominated_by)

    # --- validation --------------------------------------------------------

    def problems(self) -> List[str]:
        """Every invariant violation, rather than the first one found."""
        out: List[str] = []
        pool_ids = {s.player_id for s in self.pool}

        seen: Dict[int, str] = {}
        for o in self.owners:
            if o.budget_remaining < 0:
                out.append(f"{o.owner_id}: budget {o.budget_remaining} is negative")
            if o.n_players > o.roster_capacity:
                out.append(f"{o.owner_id}: {o.n_players} players exceeds "
                           f"capacity {o.roster_capacity}")
            if o.spent + o.budget_remaining != o.budget_start:
                out.append(f"{o.owner_id}: spent {o.spent} plus remaining "
                           f"{o.budget_remaining} does not reconcile to "
                           f"{o.budget_start}")
            if o.budget_remaining < o.reserve_for_open_slots:
                out.append(
                    f"{o.owner_id}: ${o.budget_remaining} cannot cover "
                    f"{o.open_slots} open slot(s) at the ${o.min_bid} minimum")
            for pid in o.player_ids:
                if pid not in pool_ids:
                    out.append(f"{o.owner_id} holds {pid}, which is not in the pool")
                if pid in seen:
                    out.append(f"player {pid} is held by both {seen[pid]} "
                               f"and {o.owner_id}")
                seen[pid] = o.owner_id
            if not o.can_still_field_lineup():
                out.append(f"{o.owner_id}: {o.lineup_shortfall()}")

        for pid in self.withdrawn:
            if pid in seen:
                out.append(f"player {pid} is both withdrawn and rostered "
                           f"by {seen[pid]}")
            if pid not in pool_ids:
                out.append(f"withdrawn player {pid} is not in the pool")

        by_owner: Dict[str, int] = {}
        for i, t in enumerate(self.transactions):
            if t.sequence != i:
                out.append(f"transaction {i} carries sequence {t.sequence}")
            if t.owner_id not in self.owner_by_id:
                out.append(f"transaction {t.sequence} names unknown owner "
                           f"{t.owner_id!r}")
                continue
            if seen.get(t.player_id) != t.owner_id:
                out.append(f"transaction {t.sequence} awards {t.player_id} to "
                           f"{t.owner_id} but that owner does not hold him")
            by_owner[t.owner_id] = by_owner.get(t.owner_id, 0) + t.price
        for oid, total in by_owner.items():
            o = self.owner_by_id[oid]
            if total > o.spent:
                out.append(f"{oid}: transactions total {total} exceeds roster "
                           f"spend {o.spent}")

        if self.nomination is not None:
            n = self.nomination
            if n.player_id not in pool_ids:
                out.append(f"nominated player {n.player_id} is not in the pool")
            elif not self.is_available(n.player_id):
                out.append(f"nominated player {n.player_id} is not available")
            if n.nominated_by not in self.owner_by_id:
                out.append(f"nomination names unknown owner {n.nominated_by!r}")
            if n.high_bidder is not None:
                hb = self.owner_by_id.get(n.high_bidder)
                if hb is None:
                    out.append(f"high bidder {n.high_bidder!r} is not in the room")
                elif not hb.can_bid(int(n.current_bid)):
                    out.append(f"high bidder {n.high_bidder} cannot legally bid "
                               f"{n.current_bid} (max {hb.max_bid})")

        if len(self.owners) != self.settings.n_teams:
            out.append(f"{len(self.owners)} owners but the league has "
                       f"{self.settings.n_teams} teams")
        return out

    def validate(self) -> "AuctionState":
        """Raise on any violation; return self so it can be chained."""
        problems = self.problems()
        if problems:
            raise AuctionStateInvalid(problems)
        return self

    @property
    def is_valid(self) -> bool:
        return not self.problems()

    @property
    def is_complete(self) -> bool:
        return all(o.is_full for o in self.owners)

    # --- reporting ---------------------------------------------------------

    def fingerprint(self) -> str:
        """A stable digest of everything that can change a valuation.

        Rosters, prices, budgets, withdrawals and the focus owner. Used as a
        cache key: two states with the same fingerprint are interchangeable for
        any question this package answers, and two that differ anywhere must
        never share a cached result.
        """
        import hashlib
        parts = [f"focus={self.focus_owner_id}",
                 f"teams={self.settings.n_teams}",
                 f"cap={self.settings.roster_size}"]
        for o in sorted(self.owners, key=lambda x: x.owner_id):
            roster = ",".join(f"{f.player_id}@{f.price}"
                              for f in sorted(o.filled, key=lambda f: f.player_id))
            parts.append(f"{o.owner_id}|{o.budget_start}|{o.roster_capacity}|{roster}")
        parts.append("withdrawn=" + ",".join(str(p) for p in sorted(self.withdrawn)))
        parts.append("pool=" + ",".join(str(s.player_id) for s in self.pool))
        return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> Dict[str, object]:
        return {
            "focus_owner_id": self.focus_owner_id,
            "fingerprint": self.fingerprint(),
            "n_owners": len(self.owners),
            "pool_size": len(self.pool),
            "available": len(self.available_ids),
            "withdrawn": sorted(self.withdrawn),
            "complete": self.is_complete,
            "valid": self.is_valid,
            "problems": self.problems(),
            "owners": [o.to_dict() for o in self.owners],
            "transactions": [t.to_dict() for t in self.transactions],
            "nomination": self.nomination.to_dict() if self.nomination else None,
        }

    def room_summary(self, width: int = 88, next_bid: Optional[int] = None) -> str:
        """A sanitized table of the room. No player names, no projections."""
        bar = "=" * width
        out = [bar, "AUCTION ROOM", bar,
               f"focus owner   {self.focus_owner_id}",
               f"fingerprint   {self.fingerprint()}",
               f"pool          {len(self.pool)} players, "
               f"{len(self.available_ids)} still available, "
               f"{len(self.withdrawn)} withdrawn",
               f"transactions  {len(self.transactions)}",
               ""]
        head = (f"  {'owner':<10}{'budget':>8}{'spent':>7}{'slots':>7}"
                f"{'reserve':>9}{'discret':>9}{'max bid':>9}"
                f"  {'QB/RB/WR/TE':<13}{'feasible':>9}")
        out += [head, "  " + "-" * (len(head) - 2)]
        for o in self.owners:
            c = o.counts
            mark = "*" if o.owner_id == self.focus_owner_id else " "
            out.append(
                f" {mark}{o.owner_id:<10}{o.budget_remaining:>8}{o.spent:>7}"
                f"{o.open_slots:>7}{o.reserve_for_open_slots:>9}"
                f"{o.discretionary:>9}{o.max_bid:>9}"
                f"  {c.qb}/{c.rb}/{c.wr}/{c.te:<9}"
                f"{('yes' if o.can_still_field_lineup() else 'NO'):>9}")
        out += ["  " + "-" * (len(head) - 2),
                "  reserve = $1 held for each open slot; discretionary = budget",
                "  above that floor; max bid = discretionary + $1.",
                "  feasible = a legal 8-slot lineup can still be completed. It is",
                "  a matching question on the eligibility graph, not a positional",
                "  quota -- this league has no quarterback maximum.", ""]

        if self.nomination is not None:
            n = self.nomination
            out += [f"ON THE BLOCK  player {n.player_id}, nominated by "
                    f"{n.nominated_by}",
                    f"              standing bid "
                    + (f"${n.current_bid} from {n.high_bidder}"
                       if n.current_bid is not None else "none"), ""]
            if next_bid is None and n.current_bid is not None:
                next_bid = n.current_bid + 1

        if next_bid is not None:
            cap = self.bid_capacity(next_bid)
            out += [f"WHO MAY LEGALLY BID ${next_bid}",
                    f"  able ({cap['n_able']}): "
                    + (", ".join(cap["able"]) if cap["able"] else "nobody")]
            for oid, why in cap["blocked"].items():
                out.append(f"  blocked  {oid:<10} {why}")
            out += ["",
                    "  This is who MAY bid, from budget, roster room and",
                    "  feasibility. Who WOULD bid is not modelled here.", ""]

        problems = self.problems()
        if problems:
            out += ["INVARIANT VIOLATIONS"]
            out += [f"  - {p}" for p in problems]
            out.append("")
        out.append(bar)
        return "\n".join(out)


def new_auction(pool: Sequence[PlayerSpec], owner_ids: Sequence[str],
                focus_owner_id: str,
                settings: LeagueSettings = DEFAULT_LEAGUE,
                team_names: Optional[Sequence[str]] = None,
                budget: Optional[int] = None,
                roster_capacity: Optional[int] = None,
                min_bid: Optional[int] = None) -> AuctionState:
    """An empty room: every owner at full budget with an empty roster.

    Budget and capacity default to the league settings ($200, 15) and are
    overridable only explicitly, so a test fixture cannot quietly disagree with
    the league this project is for.
    """
    budget = settings.budget if budget is None else budget
    roster_capacity = settings.roster_size if roster_capacity is None else roster_capacity
    min_bid = settings.min_bid if min_bid is None else min_bid
    names = list(team_names) if team_names else list(owner_ids)
    if len(names) != len(owner_ids):
        raise AuctionRuleError("team_names must match owner_ids in length")
    owners = tuple(
        OwnerAuctionState(owner_id=oid, team_name=nm, budget_start=budget,
                          roster_capacity=roster_capacity, min_bid=min_bid)
        for oid, nm in zip(owner_ids, names))
    return AuctionState(settings=settings, pool=tuple(pool), owners=owners,
                        focus_owner_id=focus_owner_id)
