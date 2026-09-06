"""Room arithmetic for an auction endgame. No simulation, no behaviour, no CE.

Everything in this module is a *fact about the room* derived from budgets,
open slots and the candidate's own legality. None of it is a value, a
prediction or a recommendation, and the separation is deliberate: knowing that
no rival can legally bid $41 tells you what you *can* win, and says nothing
whatever about what you *should* pay. A model that confuses those two is a
model that overpays for control.

The one derived quantity that looks like advice is the **financial-control
threshold**: one increment above the highest rival candidate-specific legal
maximum. It means exactly one thing -- above it, no other owner may legally
bid. It is not a recommended bid and the formatter says so on every line.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from ..auction.state import AuctionState
from ..league import Position

__all__ = [
    "OwnerCapacity",
    "EndgameReport",
    "candidate_legal_max",
    "assess_endgame",
    "format_endgame",
]


def candidate_legal_max(state: AuctionState, owner_id: str,
                        candidate_id: int) -> int:
    """The highest price this owner may legally pay for *this* player.

    Zero means he cannot buy the player at any price -- a full roster, or a
    roster whose remaining slots have a lineup requirement this player cannot
    satisfy. That is a different fact from being unable to afford him, and the
    caller is told which.

    Legality is monotone in price: raising the bid can only break the budget
    test, never fix it, and the feasibility test does not read the price except
    through the money it leaves behind. So the legal prices form one contiguous
    run from ``min_bid`` upward and a binary search finds its top exactly.
    """
    owner = state.owner(owner_id)
    if owner.open_slots <= 0:
        return 0
    lo = owner.min_bid
    hi = owner.max_bid
    if hi < lo:
        return 0
    if state.purchase_shortfall(candidate_id, owner_id, lo) is not None:
        return 0
    if state.purchase_shortfall(candidate_id, owner_id, hi) is None:
        return hi
    # Invariant: lo is legal, hi is not.
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if state.purchase_shortfall(candidate_id, owner_id, mid) is None:
            lo = mid
        else:
            hi = mid
    return lo


@dataclass(frozen=True)
class OwnerCapacity:
    """One owner's arithmetic position on one candidate."""

    owner_id: str
    budget_remaining: int
    open_slots: int
    reserved_for_open_slots: int
    """Dollars that must survive to put at least $1 in every unfilled slot."""
    financial_max_bid: int
    candidate_legal_max: int
    can_compete: bool
    exclusion_reason: Optional[str]
    positionally_blocked: bool
    """True when money exists but the candidate cannot legally join the roster."""

    def to_dict(self) -> Dict[str, object]:
        return {
            "owner_id": self.owner_id,
            "budget_remaining": self.budget_remaining,
            "open_slots": self.open_slots,
            "reserved_for_open_slots": self.reserved_for_open_slots,
            "financial_max_bid": self.financial_max_bid,
            "candidate_legal_max": self.candidate_legal_max,
            "can_compete": self.can_compete,
            "exclusion_reason": self.exclusion_reason,
            "positionally_blocked": self.positionally_blocked,
        }


@dataclass(frozen=True)
class EndgameReport:
    """Pure room arithmetic for one candidate at one moment."""

    candidate_id: int
    position: str
    focus_owner_id: str
    current_price: Optional[int]
    increment: int
    auction_fingerprint: str
    owners: Tuple[OwnerCapacity, ...]

    @property
    def by_id(self) -> Dict[str, OwnerCapacity]:
        return {o.owner_id: o for o in self.owners}

    @property
    def us(self) -> OwnerCapacity:
        return self.by_id[self.focus_owner_id]

    @property
    def rivals(self) -> Tuple[OwnerCapacity, ...]:
        return tuple(o for o in self.owners if o.owner_id != self.focus_owner_id)

    @property
    def our_legal_max(self) -> int:
        return self.us.candidate_legal_max

    @property
    def highest_rival_legal_max(self) -> int:
        return max((o.candidate_legal_max for o in self.rivals), default=0)

    @property
    def top_rival(self) -> Optional[str]:
        live = [o for o in self.rivals if o.candidate_legal_max > 0]
        if not live:
            return None
        return max(live, key=lambda o: (o.candidate_legal_max,
                                        # deterministic tiebreak
                                        [-ord(c) for c in o.owner_id])).owner_id

    @property
    def competitors(self) -> Tuple[str, ...]:
        return tuple(o.owner_id for o in self.rivals if o.candidate_legal_max > 0)

    @property
    def financial_control_threshold(self) -> Optional[int]:
        """One increment above the highest rival legal maximum.

        ``None`` when we could not reach it legally ourselves. Reaching it means
        *no rival may legally outbid us*, and nothing more than that. It is not
        a price at which buying is a good idea.
        """
        threshold = self.highest_rival_legal_max + self.increment
        if self.our_legal_max <= 0:
            return None
        return threshold

    @property
    def we_can_guarantee_winning(self) -> bool:
        """Can we legally reach a price no rival may legally match?"""
        t = self.financial_control_threshold
        return t is not None and self.our_legal_max >= t

    @property
    def rival_can_guarantee_outbidding_us(self) -> Optional[str]:
        """A rival who may legally exceed our own legal maximum, if any."""
        for o in sorted(self.rivals, key=lambda x: -x.candidate_legal_max):
            if o.candidate_legal_max > self.our_legal_max:
                return o.owner_id
        return None

    def fall_out_at(self, price: int) -> Tuple[str, ...]:
        """Rivals who may bid below ``price`` but not at it."""
        return tuple(o.owner_id for o in self.rivals
                     if 0 < o.candidate_legal_max < price)

    @property
    def n_fall_out_at_next_bid(self) -> int:
        if self.current_price is None:
            return 0
        nxt = self.current_price + self.increment
        return len([o for o in self.rivals
                    if 0 < o.candidate_legal_max < nxt])

    @property
    def dollar_nomination_uncontested(self) -> bool:
        """Would a $1 nomination draw no legal rival bid at $2?

        Uncontested here means *no rival may legally raise it*, which is a
        capacity fact. A rival who may bid and chooses not to is behaviour and
        is not modelled here.
        """
        return not any(o.candidate_legal_max >= 1 + self.increment
                       for o in self.rivals)

    @property
    def our_leverage_is_illusory(self) -> bool:
        """We have the most money and cannot legally use it on this player."""
        return (self.us.positionally_blocked
                and self.us.financial_max_bid >= max(
                    (o.financial_max_bid for o in self.rivals), default=0))

    @property
    def money_reserved_league_wide(self) -> int:
        return sum(o.reserved_for_open_slots for o in self.owners)

    def to_dict(self) -> Dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "position": self.position,
            "focus_owner_id": self.focus_owner_id,
            "current_price": self.current_price,
            "increment": self.increment,
            "auction_fingerprint": self.auction_fingerprint,
            "owners": [o.to_dict() for o in self.owners],
            "our_legal_max": self.our_legal_max,
            "highest_rival_legal_max": self.highest_rival_legal_max,
            "top_rival": self.top_rival,
            "competitors": list(self.competitors),
            "financial_control_threshold": self.financial_control_threshold,
            "we_can_guarantee_winning": self.we_can_guarantee_winning,
            "rival_can_guarantee_outbidding_us":
                self.rival_can_guarantee_outbidding_us,
            "n_fall_out_at_next_bid": self.n_fall_out_at_next_bid,
            "money_reserved_league_wide": self.money_reserved_league_wide,
            "dollar_nomination_uncontested": self.dollar_nomination_uncontested,
            "our_leverage_is_illusory": self.our_leverage_is_illusory,
            "label": ("room arithmetic only. A control threshold says rivals "
                      "CANNOT legally outbid; it never says we SHOULD bid."),
        }


def assess_endgame(state: AuctionState, candidate_id: int, *,
                   current_price: Optional[int] = None,
                   increment: int = 1) -> EndgameReport:
    """Every owner's legal capacity on one candidate. Arithmetic, not value."""
    if increment < 1:
        raise ValueError("the bid increment must be at least $1")
    spec = state.spec(candidate_id)
    caps: List[OwnerCapacity] = []
    for o in state.owners:
        legal = candidate_legal_max(state, o.owner_id, candidate_id)
        reason: Optional[str] = None
        blocked = False
        if legal <= 0:
            if o.open_slots <= 0:
                reason = "roster full"
            elif o.max_bid < o.min_bid:
                reason = (f"no legal dollar left ({o.budget_remaining} "
                          f"remaining, {o.open_slots} slot(s) to fill)")
            else:
                reason = state.purchase_shortfall(candidate_id, o.owner_id,
                                                  o.min_bid)
                blocked = True
        caps.append(OwnerCapacity(
            owner_id=o.owner_id, budget_remaining=o.budget_remaining,
            open_slots=o.open_slots,
            reserved_for_open_slots=o.reserve_for_open_slots,
            financial_max_bid=o.max_bid, candidate_legal_max=legal,
            can_compete=legal > 0, exclusion_reason=reason,
            positionally_blocked=blocked))
    return EndgameReport(
        candidate_id=candidate_id, position=Position(int(spec.position)).name,
        focus_owner_id=state.focus_owner_id, current_price=current_price,
        increment=int(increment), auction_fingerprint=state.fingerprint(),
        owners=tuple(caps))


def format_endgame(report: EndgameReport, width: int = 92) -> str:
    """Sanitized rendering: owner ids, dollars and reasons. No player names."""
    bar = "=" * width
    out = [bar, "ENDGAME ROOM ARITHMETIC", bar,
           f"candidate            player {report.candidate_id} "
           f"({report.position})",
           f"our owner            {report.focus_owner_id}",
           f"current price        "
           f"{'-' if report.current_price is None else '$%d' % report.current_price}"
           f"   increment ${report.increment}",
           f"auction fingerprint  {report.auction_fingerprint}", ""]
    head = (f"  {'owner':<10}{'budget':>8}{'slots':>7}{'reserve':>9}"
            f"{'fin max':>9}{'legal max':>11}  reason")
    out += [head, "  " + "-" * (width - 2)]
    for o in report.owners:
        mark = "*" if o.owner_id == report.focus_owner_id else " "
        out.append(f" {mark}{o.owner_id:<10}{o.budget_remaining:>8}"
                   f"{o.open_slots:>7}{o.reserved_for_open_slots:>9}"
                   f"{o.financial_max_bid:>9}{o.candidate_legal_max:>11}  "
                   f"{o.exclusion_reason or ''}")
    out += ["", f"our legal maximum            ${report.our_legal_max}",
            f"highest rival legal maximum  ${report.highest_rival_legal_max}"
            f"  ({report.top_rival or 'nobody can compete'})",
            f"owners who can still compete {len(report.competitors)}  "
            f"{', '.join(report.competitors) or '(none)'}",
            f"financial-control threshold  "
            f"{'n/a' if report.financial_control_threshold is None else '$%d' % report.financial_control_threshold}",
            f"we can guarantee winning     {report.we_can_guarantee_winning}",
            f"a rival can outbid us        "
            f"{report.rival_can_guarantee_outbidding_us or 'no'}",
            f"fall out at the next bid     {report.n_fall_out_at_next_bid}",
            f"league-wide $1 reserve       ${report.money_reserved_league_wide}",
            f"$1 nomination uncontested    {report.dollar_nomination_uncontested}",
            f"our leverage is illusory     {report.our_leverage_is_illusory}",
            "", "  The control threshold is a LEGALITY fact: above it no rival",
            "  may bid. It is NOT a recommended price and carries no CE claim.",
            bar]
    return "\n".join(out)
