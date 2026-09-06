"""Who can actually compete for this player, and on what evidence.

Four different questions get four different answers here, and the whole point is
that they are not the same question:

**Financial ability** -- can this owner afford the price at all, from budget and
the dollar he must reserve for every other open slot.

**Roster legality** -- may he *legally* buy this particular player. An owner one
slot from the end who still needs a receiver cannot bid on a quarterback at any
price, however much money he has.

**Estimated interest** -- would this player improve his roster. A weak signal
here, derived from lineup structure, and labelled as weak.

**Predicted clearing pressure** -- what the market model expects the player to go
for, and how much of that is evidence rather than assumption.

Nothing in this module predicts who wins. There is no historical bidder data for
this league, so estimated interest is structural and the price is a scenario
range. An auction winner is determined by preferences this repository has never
observed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from ..auction.state import AuctionState, OwnerAuctionState
from ..league import Position
from .live import AdjustedPrice, MarketState

__all__ = ["OwnerPressure", "RoomPressure", "assess_room_pressure",
           "format_room_pressure"]


@dataclass(frozen=True)
class OwnerPressure:
    """One owner's ability, legality, fit and evidence for one candidate."""

    owner_id: str
    budget_remaining: int
    open_slots: int
    financial_max_bid: int
    """What the rules let him spend: budget less a dollar per other open slot."""
    candidate_legal_max_bid: int
    """The same, but zero when buying this player would strand his roster."""
    can_bid_at_price: bool
    blocked_reason: Optional[str]
    roster_fit: str
    """Structural signal only. Not a preference, and not a prediction."""
    fit_score: float
    own_observations: int
    """Sales by this owner the model has learned from. Usually very few."""
    buyer_multiplier: Optional[float]

    @property
    def is_competitor(self) -> bool:
        return self.can_bid_at_price

    def to_dict(self) -> Dict[str, object]:
        return {
            "owner_id": self.owner_id,
            "budget_remaining": self.budget_remaining,
            "open_slots": self.open_slots,
            "financial_max_bid": self.financial_max_bid,
            "candidate_legal_max_bid": self.candidate_legal_max_bid,
            "can_bid_at_price": self.can_bid_at_price,
            "blocked_reason": self.blocked_reason,
            "roster_fit": self.roster_fit,
            "fit_score": round(self.fit_score, 3),
            "own_observations": self.own_observations,
            "buyer_multiplier": (None if self.buyer_multiplier is None
                                 else round(self.buyer_multiplier, 4)),
        }


@dataclass(frozen=True)
class RoomPressure:
    """The whole room's capacity to compete for one candidate at one price."""

    candidate_id: int
    candidate_key: Optional[str]
    position: str
    price: int
    auction_fingerprint: str
    market_fingerprint: Optional[str]
    owners: Tuple[OwnerPressure, ...]
    expected_clearing: Optional[AdjustedPrice]

    @property
    def competitors(self) -> Tuple[OwnerPressure, ...]:
        return tuple(o for o in self.owners if o.is_competitor)

    @property
    def blocked(self) -> Tuple[OwnerPressure, ...]:
        return tuple(o for o in self.owners if not o.is_competitor)

    @property
    def headroom(self) -> int:
        """The highest any legal competitor could go. A ceiling, not a forecast."""
        return max((o.candidate_legal_max_bid for o in self.competitors), default=0)

    def to_dict(self) -> Dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "position": self.position,
            "price": self.price,
            "auction_fingerprint": self.auction_fingerprint,
            "market_fingerprint": self.market_fingerprint,
            "n_competitors": len(self.competitors),
            "n_blocked": len(self.blocked),
            "max_legal_headroom": self.headroom,
            "expected_clearing_price": (self.expected_clearing.to_dict()
                                        if self.expected_clearing else None),
            "owners": [o.to_dict() for o in self.owners],
            "label": ("capacity and legality, plus a scenario clearing price. "
                      "This does NOT predict the winner: no bidder preferences "
                      "have ever been observed for this league."),
        }


def _roster_fit(state: AuctionState, owner: OwnerAuctionState,
                position: Position) -> Tuple[str, float]:
    """A structural read on whether this player would reach the owner's lineup.

    Counts only, against the actual slot rules -- one QB seat plus a flexible
    one, two RB seats, three WR/TE seats, a flex open to RB/WR/TE. It says
    whether a starting seat is still unfilled, never whether the owner wants
    the player.
    """
    c = owner.counts
    if position is Position.QB:
        if c.qb == 0:
            return ("fills the dedicated QB slot", 1.0)
        if c.qb == 1:
            return ("superflex candidate; an RB/WR/TE may take that seat instead",
                    0.55)
        return ("third or later QB; bench or injury cover only", 0.2)
    if position is Position.RB:
        if c.rb < 2:
            return ("fills a dedicated RB slot", 1.0)
        if c.rb + c.wt < 6:
            return ("competes for flex or a remaining non-QB seat", 0.6)
        return ("bench depth", 0.25)
    # WR and TE share the three WT seats; neither has a dedicated slot here.
    if c.wt < 3:
        return ("fills a WR/TE slot", 1.0)
    if c.rb + c.wt < 6:
        return ("competes for flex or a remaining non-QB seat", 0.6)
    return ("bench depth", 0.25)


def assess_room_pressure(
    state: AuctionState,
    candidate_id: int,
    price: int,
    *,
    market: Optional[MarketState] = None,
    candidate_key: Optional[str] = None,
) -> RoomPressure:
    """Every owner's ability, legality, fit and learned tendency, for one price.

    Reuses the auction state's own candidate-aware legality rather than
    re-deriving it: an owner who cannot legally hold this player is excluded for
    that reason and told so, separately from being unable to afford him.
    """
    spec = state.spec(candidate_id)
    position = Position(int(spec.position))

    owners: List[OwnerPressure] = []
    for o in state.owners:
        shortfall = state.purchase_shortfall(candidate_id, o.owner_id, price)
        legal_max = 0
        if o.open_slots > 0:
            # Walk down from the financial ceiling to the highest price that is
            # also legal for THIS player. Feasibility does not depend on price,
            # so one probe settles it.
            probe = state.purchase_shortfall(candidate_id, o.owner_id,
                                             max(o.min_bid, 1))
            legal_max = o.max_bid if probe is None else 0
        fit, score = _roster_fit(state, o, position)
        level = None
        if market is not None:
            lv = market.buyer_levels.get(o.owner_id)
            level = lv
        owners.append(OwnerPressure(
            owner_id=o.owner_id, budget_remaining=o.budget_remaining,
            open_slots=o.open_slots, financial_max_bid=o.max_bid,
            candidate_legal_max_bid=legal_max,
            can_bid_at_price=shortfall is None,
            blocked_reason=shortfall, roster_fit=fit, fit_score=score,
            own_observations=0 if level is None else level.n,
            buyer_multiplier=(None if level is None
                              or level.n < (market.config.min_buyer_observations
                                            if market else 99)
                              else level.multiplier)))

    expected = None
    if market is not None and candidate_key:
        expected = market.adjusted(candidate_key)

    return RoomPressure(
        candidate_id=candidate_id, candidate_key=candidate_key,
        position=position.name, price=price,
        auction_fingerprint=state.fingerprint(),
        market_fingerprint=None if market is None else market.fingerprint(),
        owners=tuple(owners), expected_clearing=expected)


def format_room_pressure(pressure: RoomPressure, width: int = 100) -> str:
    """Sanitized rendering. Ids and dollars only; no player names."""
    bar = "=" * width
    out = [bar, "ROOM PRESSURE", bar,
           f"candidate       id {pressure.candidate_id} ({pressure.position})",
           f"price probed    ${pressure.price}",
           f"auction state   {pressure.auction_fingerprint}",
           f"market state    {pressure.market_fingerprint or 'none supplied'}"]
    if pressure.expected_clearing:
        e = pressure.expected_clearing
        out += [f"expected clear  ${e.low}-${e.high} (base ${e.base})  "
                f"anchor credibility {e.anchor_credibility:.2f}",
                f"                evidence: room {e.observations_used['room']}, "
                f"position {e.observations_used['position']}, "
                f"tier {e.observations_used['tier']}"]
    else:
        out.append("expected clear  not supplied (no market state)")
    out.append("")

    head = (f"  {'owner':<10}{'budget':>8}{'slots':>7}{'fin max':>9}"
            f"{'legal max':>11}{'bids?':>7}{'obs':>5}  {'fit':<44}")
    out += [head, "  " + "-" * (len(head) - 2)]
    for o in pressure.owners:
        mark = "yes" if o.can_bid_at_price else "NO"
        out.append(f"  {o.owner_id:<10}{o.budget_remaining:>8}{o.open_slots:>7}"
                   f"{o.financial_max_bid:>9}{o.candidate_legal_max_bid:>11}"
                   f"{mark:>7}{o.own_observations:>5}  {o.roster_fit:<44}")
    out += ["  " + "-" * (len(head) - 2), ""]

    if pressure.blocked:
        out.append("WHY AN OWNER IS OUT")
        for o in pressure.blocked:
            out.append(f"  {o.owner_id:<10} {o.blocked_reason}")
        out.append("")
    out += [f"  competitors at ${pressure.price}: {len(pressure.competitors)} of "
            f"{len(pressure.owners)}",
            f"  highest legal headroom: ${pressure.headroom}",
            "",
            "  'fin max' is the financial ceiling; 'legal max' is zero when",
            "  buying THIS player would leave the roster unable to field a",
            "  lineup. The two are different constraints and are not merged.",
            "",
            "  Fit is STRUCTURAL -- an unfilled seat, not a preference. No",
            "  bidder preferences have ever been observed for this league, so",
            "  nothing here predicts who wins. The clearing price is a scenario",
            "  range, and a final sale would be evidence about that range, not",
            "  about anyone's maximum.", bar]
    return "\n".join(out)
