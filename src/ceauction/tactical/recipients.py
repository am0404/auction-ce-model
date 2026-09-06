"""Who gets the player if we stop bidding -- by name, one branch at a time.

This is the centre of the tactical layer. "The player costs $30" is not a
decision input; "if we stop at $29 the player joins Owner07, who already has
two good backs and $84 left" is. The same player going to two different owners
can move our championship equity by different amounts and in principle in
different directions, and a model that averages the recipients before
evaluating them destroys exactly that information.

So the rule here is absolute: **every named recipient is evaluated on its own
branch first.** A weighted summary may be produced afterwards, from branches
that already exist, and the weights are labelled as scenario assumptions rather
than as estimated probabilities.

An illegal named recipient is refused loudly. If the caller tells us Owner07 is
the current high bidder and Owner07 cannot legally hold this player at this
price, the branch is returned with ``legal=False`` and the shortfall text; it is
never quietly replaced by a different owner, because that would answer a
question nobody asked.

**Clearing price is an approximation, and named as one.** A recipient pays the
second-highest scenario willingness plus one increment, floored at the next
legal bid over the current price and capped by his own willingness and his
candidate-specific legal maximum. That is the standard ascending-auction
approximation. It is *not* a proven description of Sleeper's mechanism, and the
field ``price_rule`` says so in every serialization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from ..auction.costs import CostBook
from ..auction.state import AuctionState
from ..market.live import MarketState
from .bidders import (BIDDER_SCENARIOS, DEFAULT_BIDDER_SCENARIO,
                      BidderWillingness, assess_room_bidders)
from .endgame import candidate_legal_max

__all__ = [
    "RecipientBranch",
    "RecipientSet",
    "PRICE_RULE",
    "enumerate_recipients",
    "format_recipients",
]

PRICE_RULE = (
    "APPROXIMATION: second-highest scenario willingness + one increment, "
    "floored at the next legal bid and capped by the winner's own willingness "
    "and his candidate-specific legal maximum. Not Sleeper's proven mechanism.")


@dataclass(frozen=True)
class RecipientBranch:
    """One named owner taking the candidate at one price. A branch, not a guess."""

    kind: str
    """``current_leader`` | ``strongest_bidder`` | ``plausible`` | ``unavailable``"""

    owner_id: Optional[str]
    price: Optional[int]
    legal: bool
    refusal: Optional[str]
    rationale: str
    willingness_low: Optional[int] = None
    willingness_base: Optional[int] = None
    willingness_high: Optional[int] = None
    candidate_legal_max: Optional[int] = None
    scenario_weight: float = 0.0
    """A STATED assumption about how the branches divide, not an estimate."""

    @property
    def label(self) -> str:
        if self.kind == "unavailable":
            return "unavailable (withdrawn from our board, no owner assigned)"
        return f"{self.owner_id} at ${self.price}"

    def to_dict(self) -> Dict[str, object]:
        return {
            "kind": self.kind, "owner_id": self.owner_id, "price": self.price,
            "legal": self.legal, "refusal": self.refusal,
            "rationale": self.rationale, "label": self.label,
            "willingness": {"low": self.willingness_low,
                            "base": self.willingness_base,
                            "high": self.willingness_high},
            "candidate_legal_max": self.candidate_legal_max,
            "scenario_weight": round(self.scenario_weight, 4),
            "weight_kind": "stated scenario assumption, not an estimated probability",
        }


@dataclass(frozen=True)
class RecipientSet:
    """Every named branch for one candidate at one moment."""

    candidate_id: int
    current_price: Optional[int]
    increment: int
    current_leader: Optional[str]
    focus_owner_id: str
    scenario: str
    auction_fingerprint: str
    market_fingerprint: Optional[str]
    branches: Tuple[RecipientBranch, ...]
    price_rule: str = PRICE_RULE

    @property
    def legal_branches(self) -> Tuple[RecipientBranch, ...]:
        return tuple(b for b in self.branches if b.legal)

    @property
    def refused(self) -> Tuple[RecipientBranch, ...]:
        return tuple(b for b in self.branches if not b.legal)

    @property
    def named_owners(self) -> Tuple[str, ...]:
        return tuple(b.owner_id for b in self.legal_branches
                     if b.owner_id is not None)

    def branch_for(self, owner_id: str) -> Optional[RecipientBranch]:
        for b in self.branches:
            if b.owner_id == owner_id:
                return b
        return None

    def to_dict(self) -> Dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "current_price": self.current_price,
            "increment": self.increment,
            "current_leader": self.current_leader,
            "focus_owner_id": self.focus_owner_id,
            "bidder_scenario": self.scenario,
            "auction_fingerprint": self.auction_fingerprint,
            "market_fingerprint": self.market_fingerprint,
            "price_rule": self.price_rule,
            "branches": [b.to_dict() for b in self.branches],
            "label": ("named branches. Each is evaluated separately; any "
                      "weighted summary is built AFTER the branches exist."),
        }


def _clearing_price(winner: BidderWillingness,
                    runner_up: Optional[BidderWillingness],
                    floor_price: int, increment: int) -> int:
    """The documented ascending-auction approximation. See :data:`PRICE_RULE`."""
    second = runner_up.base if runner_up is not None else 0
    price = max(floor_price, second + increment)
    price = min(price, winner.base if winner.base > 0 else winner.candidate_legal_max)
    price = min(price, winner.candidate_legal_max)
    return max(price, 1)


def enumerate_recipients(
    state: AuctionState,
    candidate_id: int,
    *,
    current_price: Optional[int] = None,
    increment: int = 1,
    current_leader: Optional[str] = None,
    scenario: str = DEFAULT_BIDDER_SCENARIO,
    market: Optional[MarketState] = None,
    candidate_key: Optional[str] = None,
    costs: Optional[CostBook] = None,
    eligible: Optional[Sequence[str]] = None,
    max_named: int = 3,
    include_unavailable: bool = True,
) -> RecipientSet:
    """Name every materially plausible recipient. Never average them.

    ``current_leader``, when supplied, is *always* represented -- as a legal
    branch if he can legally hold the player, and as an explicitly refused
    branch with the shortfall text if he cannot. Silently substituting another
    owner would be answering a different question.
    """
    if increment < 1:
        raise ValueError("the bid increment must be at least $1")
    if scenario not in BIDDER_SCENARIOS:
        raise ValueError(f"unknown bidder scenario {scenario!r}")
    focus = state.focus_owner_id
    if current_leader is not None and current_leader not in state.owner_by_id:
        raise ValueError(f"no owner {current_leader!r} in the room")
    if not state.is_available(candidate_id):
        raise ValueError(f"player {candidate_id} is not available")

    floor = (current_price + increment) if current_price is not None else 1
    allowed = set(eligible) if eligible is not None else None

    rows = assess_room_bidders(state, candidate_id, scenario=scenario,
                               market=market, candidate_key=candidate_key,
                               costs=costs, exclude=(focus,))
    if allowed is not None:
        rows = tuple(r for r in rows if r.owner_id in allowed
                     or r.owner_id == current_leader)
    by_owner = {r.owner_id: r for r in rows}
    competitive = [r for r in rows if r.candidate_legal_max > 0]

    branches: List[RecipientBranch] = []
    seen: set = set()

    def add(owner_id: str, kind: str, rationale: str) -> None:
        if owner_id in seen:
            return
        seen.add(owner_id)
        w = by_owner.get(owner_id)
        legal_max = (w.candidate_legal_max if w is not None
                     else candidate_legal_max(state, owner_id, candidate_id))
        if legal_max <= 0 or legal_max < floor:
            why = state.purchase_shortfall(state.spec(candidate_id).player_id,
                                           owner_id, max(floor, 1))
            branches.append(RecipientBranch(
                kind=kind, owner_id=owner_id, price=None, legal=False,
                refusal=why or (f"{owner_id}'s candidate-specific legal maximum "
                                f"is ${legal_max}, below the ${floor} needed"),
                rationale=rationale, candidate_legal_max=legal_max))
            return
        others = [r for r in competitive if r.owner_id != owner_id]
        runner = max(others, key=lambda r: (r.base, r.owner_id), default=None)
        price = _clearing_price(w, runner, floor, increment) if w else floor
        problem = state.purchase_shortfall(candidate_id, owner_id, price)
        if problem is not None:
            branches.append(RecipientBranch(
                kind=kind, owner_id=owner_id, price=price, legal=False,
                refusal=problem, rationale=rationale,
                candidate_legal_max=legal_max))
            return
        branches.append(RecipientBranch(
            kind=kind, owner_id=owner_id, price=price, legal=True, refusal=None,
            rationale=rationale,
            willingness_low=w.low if w else None,
            willingness_base=w.base if w else None,
            willingness_high=w.high if w else None,
            candidate_legal_max=legal_max))

    if current_leader is not None and current_leader != focus:
        add(current_leader, "current_leader",
            "the owner currently holding the high bid; supplied by the caller "
            "and therefore represented whether or not the model would pick him")

    strongest = max(competitive, key=lambda r: (r.base, r.pursuit_score,
                                                r.owner_id), default=None)
    if strongest is not None:
        add(strongest.owner_id, "strongest_bidder",
            "highest scenario willingness among owners who may legally hold him")

    for r in competitive:
        if len([b for b in branches if b.owner_id]) >= max_named:
            break
        if r.owner_id in seen:
            continue
        if r.base < floor:
            continue
        add(r.owner_id, "plausible",
            f"willing to ${r.base} under {scenario}; {r.roster_fit}")

    if include_unavailable:
        branches.append(RecipientBranch(
            kind="unavailable", owner_id=None, price=None, legal=True,
            refusal=None,
            rationale=("the player leaves our board without our assigning him "
                       "to a fabricated owner. Correct when no named owner is "
                       "credible, and honest about what we do not know.")))

    # Weights are attached only now, after every branch already exists, and are
    # a stated assumption: proportional to base willingness among legal named
    # branches, with a fixed residual on the unassigned branch.
    named = [b for b in branches if b.legal and b.owner_id is not None]
    total = float(sum(max(1, b.willingness_base or 1) for b in named))
    weighted: List[RecipientBranch] = []
    residual = 0.15 if (include_unavailable and named) else (
        1.0 if include_unavailable else 0.0)
    import dataclasses
    for b in branches:
        if b.kind == "unavailable":
            weighted.append(dataclasses.replace(b, scenario_weight=residual))
        elif b.legal and b.owner_id is not None:
            share = (1.0 - residual) * max(1, b.willingness_base or 1) / total
            weighted.append(dataclasses.replace(b, scenario_weight=share))
        else:
            weighted.append(b)

    return RecipientSet(
        candidate_id=candidate_id, current_price=current_price,
        increment=int(increment), current_leader=current_leader,
        focus_owner_id=focus, scenario=scenario,
        auction_fingerprint=state.fingerprint(),
        market_fingerprint=None if market is None else market.fingerprint(),
        branches=tuple(weighted))


def format_recipients(rs: RecipientSet, width: int = 96) -> str:
    bar = "=" * width
    out = [bar, "NAMED PASS RECIPIENTS", bar,
           f"candidate        player {rs.candidate_id}",
           f"current price    "
           f"{'-' if rs.current_price is None else '$%d' % rs.current_price}"
           f"   increment ${rs.increment}",
           f"current leader   {rs.current_leader or '(none supplied)'}",
           f"bidder scenario  {rs.scenario}",
           f"auction / market {rs.auction_fingerprint} / "
           f"{rs.market_fingerprint or 'no market state'}", ""]
    head = (f"  {'kind':<18}{'owner':<10}{'price':>7}{'legal max':>11}"
            f"{'willing':>14}{'weight':>8}  note")
    out += [head, "  " + "-" * (width - 2)]
    for b in rs.branches:
        rng = ("-" if b.willingness_base is None else
               f"{b.willingness_low}/{b.willingness_base}/{b.willingness_high}")
        note = b.refusal or b.rationale
        out.append(f"  {b.kind:<18}{(b.owner_id or '-'):<10}"
                   f"{('-' if b.price is None else '$%d' % b.price):>7}"
                   f"{('-' if b.candidate_legal_max is None else b.candidate_legal_max):>11}"
                   f"{rng:>14}{b.scenario_weight:>8.3f}  {note[:width - 70]}")
    out += ["", f"  price rule: {rs.price_rule}", "",
            "  Each branch is evaluated SEPARATELY. Weights are stated scenario",
            "  assumptions and are applied only after the branches exist.", bar]
    return "\n".join(out)
