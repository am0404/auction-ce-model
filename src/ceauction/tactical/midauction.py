"""SIMULATED mid-auction states, and explicit pass-price semantics.

**No league sale has ever been recorded.** Every partial auction history this
module produces is invented from the market prior for the purpose of testing
whether a reservation frontier can be found at all. Nothing here is observed,
recorded, historical or calibrated, and :class:`SimulatedState` carries a
``provenance`` string saying so that every report prints. When the real auction
happens the same machinery consumes actual recorded sales instead; until then,
calling any of this "observed" would be inventing evidence.

Why a mid-auction state is needed. At the empty room every owner holds $200 and
fifteen slots, so paying $6 or $32 for a candidate leaves the same completions
affordable and the price ladder never binds -- the opening experiment reported
``FRONTIER_NOT_REACHED`` for exactly that reason, and the five-branch
decomposition showed our payment effect sitting near -0.09 regardless of price.
Money has to be scarce before a reservation price can exist.

**Three pass-price semantics, never mixed.** "What is he worth?" is not a
question until you say what happens when you stop bidding, and the three
sensible answers give different numbers:

``STOP_NOW``        the rival leads at ``q``; we consider the next bid
                    ``p = q + increment``. The live in-auction question.
``RIVAL_OUTBIDS``   we bid ``p``; the rival answers at ``q = p + increment``,
                    subject to his willingness and legal maximum. The
                    "what if I push him" question.
``FIXED_MARKET``    ``q`` is supplied from a market scenario and does not move
                    with ``p``. Ex-ante analysis only.

:class:`PassPrice` refuses to compute ``q`` without being told which rule is in
force, and refuses to accept an explicit ``q`` under a rule that derives it.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..auction.state import AuctionState
from ..league import Position

__all__ = [
    "SIMULATED_WATERMARK",
    "PassPriceMode",
    "PassPrice",
    "PassPriceMisuse",
    "SimulatedSale",
    "SimulatedState",
    "StateSpec",
    "STATE_SPECS",
    "build_simulated_state",
    "StateValidation",
    "validate_state",
    "StateRefused",
]

SIMULATED_WATERMARK = (
    "SIMULATED MID-AUCTION STATE -- invented from the market prior for frontier "
    "testing. NOT observed, NOT recorded, NOT historical, NOT calibrated. No "
    "league sale has ever been recorded.")


class PassPriceMisuse(RuntimeError):
    """A pass price was requested without, or against, its declared rule."""


class StateRefused(RuntimeError):
    """A simulated state is unfit for a frontier experiment."""


class PassPriceMode(Enum):
    STOP_NOW = "stop_now"
    RIVAL_OUTBIDS = "rival_outbids"
    FIXED_MARKET = "fixed_market"


@dataclass(frozen=True)
class PassPrice:
    """What the named rival pays if we stop, under ONE declared rule."""

    mode: PassPriceMode
    increment: int = 1
    standing_price: Optional[int] = None
    """``STOP_NOW`` only: the rival's current high bid."""
    fixed_q: Optional[int] = None
    """``FIXED_MARKET`` only: the explicitly supplied rival price."""

    def __post_init__(self) -> None:
        if self.increment < 1:
            raise PassPriceMisuse("the bid increment must be at least $1")
        if self.mode is PassPriceMode.STOP_NOW:
            if self.standing_price is None:
                raise PassPriceMisuse(
                    "STOP_NOW needs the rival's standing price: the whole rule "
                    "is that he already leads at q and we consider q+increment")
            if self.fixed_q is not None:
                raise PassPriceMisuse(
                    "STOP_NOW derives q from the standing bid; supplying "
                    "fixed_q would mix two pass-price semantics")
        elif self.mode is PassPriceMode.RIVAL_OUTBIDS:
            if self.fixed_q is not None or self.standing_price is not None:
                raise PassPriceMisuse(
                    "RIVAL_OUTBIDS derives q from OUR price as p+increment; "
                    "supplying a standing or fixed price would mix semantics")
        else:
            if self.fixed_q is None:
                raise PassPriceMisuse(
                    "FIXED_MARKET requires an explicit q: that is what makes it "
                    "a stated counterfactual rather than a derived one")
            if self.standing_price is not None:
                raise PassPriceMisuse(
                    "FIXED_MARKET does not use a standing bid")

    def our_prices(self, legal_max: int) -> Optional[int]:
        """Under STOP_NOW our price is pinned to the standing bid."""
        if self.mode is PassPriceMode.STOP_NOW:
            p = int(self.standing_price) + self.increment
            return p if p <= legal_max else None
        return None

    def rival_price(self, p: int, *, rival_legal_max: int,
                    rival_willingness: Optional[int] = None) -> Optional[int]:
        """``q`` for our price ``p``. ``None`` when the rival cannot pay it."""
        if self.mode is PassPriceMode.STOP_NOW:
            q = int(self.standing_price)
        elif self.mode is PassPriceMode.RIVAL_OUTBIDS:
            q = int(p) + self.increment
        else:
            q = int(self.fixed_q)
        if q > rival_legal_max:
            return None
        if rival_willingness is not None and q > rival_willingness:
            return None
        return max(1, q)

    def to_dict(self) -> Dict[str, object]:
        return {"mode": self.mode.value, "increment": self.increment,
                "standing_price": self.standing_price,
                "fixed_q": self.fixed_q,
                "note": ("a reservation price is conditional on this rule; a "
                         "result under one rule is not a universal max bid")}


@dataclass(frozen=True)
class SimulatedSale:
    """One invented sale. LOCAL OUTPUT ONLY when it names a real player."""

    sequence: int
    player_id: int
    position: str
    owner_id: str
    price: int
    market_base: int

    def to_dict(self, *, include_identity: bool = False) -> Dict[str, object]:
        out: Dict[str, object] = {
            "sequence": self.sequence, "position": self.position,
            "owner_id": self.owner_id, "price": self.price,
            "market_base": self.market_base, "simulated": True}
        if include_identity:
            out["player_id"] = self.player_id
        return out


@dataclass(frozen=True)
class StateSpec:
    """A stated recipe for one simulated partial auction. Not a prediction."""

    name: str
    description: str
    n_sales: int
    market_scenario: str
    position_inflation: Dict[str, float] = field(default_factory=dict)
    """Multiplier on the market base price for these positions. A SCENARIO
    knob, labelled everywhere; it is not a measured market behaviour."""
    seed: int = 20260906

    def to_dict(self) -> Dict[str, object]:
        return dataclasses.asdict(self)


STATE_SPECS: Dict[str, StateSpec] = {s.name: s for s in (
    StateSpec("balanced",
              "SIMULATED: ~66 sales at base market prices, no positional tilt",
              n_sales=66, market_scenario="base"),
    StateSpec("qb_inflation",
              "SIMULATED: same stage, quarterbacks clearing 40% over the base "
              "prior. A stated scenario, not observed behaviour.",
              n_sales=66, market_scenario="base",
              position_inflation={"QB": 1.40}),
    StateSpec("skill_inflation",
              "SIMULATED: same stage, RB and WR clearing 35% over the base "
              "prior. A stated scenario, not observed behaviour.",
              n_sales=66, market_scenario="base",
              position_inflation={"RB": 1.35, "WR": 1.35}),
)}


@dataclass
class SimulatedState:
    """A partial auction that never happened, and says so."""

    spec: StateSpec
    state: AuctionState
    sales: Tuple[SimulatedSale, ...]
    protected: Tuple[int, ...]
    provenance: str = SIMULATED_WATERMARK

    @property
    def is_simulated(self) -> bool:
        return True

    def fingerprint(self) -> str:
        h = hashlib.sha256()
        h.update(json.dumps(self.spec.to_dict(), sort_keys=True).encode())
        for s in self.sales:
            h.update(f"{s.sequence}|{s.player_id}|{s.owner_id}|{s.price}\n"
                     .encode())
        return h.hexdigest()[:16]

    def to_dict(self, *, include_identity: bool = False) -> Dict[str, object]:
        return {
            "provenance": self.provenance,
            "simulated": True,
            "spec": self.spec.to_dict(),
            "fingerprint": self.fingerprint(),
            "n_sales": len(self.sales),
            "sales": [s.to_dict(include_identity=include_identity)
                      for s in self.sales],
            "protected_count": len(self.protected),
        }


def build_simulated_state(board, spec: StateSpec, *, protect: Sequence[int],
                          focus_min_players: int = 3) -> SimulatedState:
    """Invent a legal partial auction. Deterministic for a fixed seed.

    Owners are chosen round-robin from a seeded shuffle so budgets and slots
    diverge without any positional quota; the price paid is the scenario's
    market base times any stated positional inflation, jittered by a seeded
    draw so the room is not uniform. Protected candidates are never sold.
    """
    rng = np.random.default_rng(spec.seed)
    st = board.state
    focus = st.focus_owner_id
    costs = board.costs[spec.market_scenario]
    protected = set(protect)

    order = sorted((s for s in st.available_specs
                    if s.player_id not in protected),
                   key=lambda s: (-costs.cost_of(s.player_id, 1),
                                  -s.base_mean, s.player_id))
    owners = [o.owner_id for o in st.owners]
    sales: List[SimulatedSale] = []
    cur = st
    idx = 0
    while len(sales) < spec.n_sales and idx < len(order):
        spec_p = order[idx]
        idx += 1
        pos = Position(int(spec_p.position)).name
        base = costs.cost_of(spec_p.player_id, 1)
        infl = spec.position_inflation.get(pos, 1.0)
        jitter = 1.0 + float(rng.normal(0.0, 0.18))
        price = max(1, int(round(base * infl * max(0.4, jitter))))
        # Rotate buyers so budgets and open slots diverge. The focus team buys
        # less often than a rival, which is what leaves it a plausible partial
        # roster rather than a full one.
        pick = owners[int(rng.integers(0, len(owners)))]
        if pick == focus and cur.owner(focus).n_players >= focus_min_players:
            pick = owners[(owners.index(pick) + 1) % len(owners)]
        placed = False
        for cand_owner in [pick] + [o for o in owners if o != pick]:
            for pay in (price, max(1, price // 2), 1):
                if cur.purchase_shortfall(spec_p.player_id, cand_owner,
                                          pay) is None:
                    cur = cur.apply_purchase(spec_p.player_id, cand_owner, pay)
                    sales.append(SimulatedSale(
                        sequence=len(sales), player_id=spec_p.player_id,
                        position=pos, owner_id=cand_owner, price=pay,
                        market_base=base))
                    placed = True
                    break
            if placed:
                break
    cur.validate()
    return SimulatedState(spec=spec, state=cur, sales=tuple(sales),
                          protected=tuple(sorted(protected)))


@dataclass
class StateValidation:
    """Every room fact a frontier experiment needs before it may run."""

    n_sales: int
    total_spent: int
    dollars_reconcile: bool
    duplicate_ownership: bool
    reserve_ok: bool
    owners: Dict[str, Dict[str, object]]
    pool_remaining: Dict[str, int]
    candidates: Dict[str, Dict[str, object]]
    focus_can_bind: bool
    refusals: Tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.refusals

    def to_dict(self) -> Dict[str, object]:
        return {
            "simulated": True,
            "n_sales": self.n_sales, "total_spent": self.total_spent,
            "dollars_reconcile": self.dollars_reconcile,
            "duplicate_ownership": self.duplicate_ownership,
            "reserve_ok": self.reserve_ok,
            "owners": self.owners, "pool_remaining": self.pool_remaining,
            "candidates": self.candidates,
            "focus_budget_can_bind": self.focus_can_bind,
            "refusals": list(self.refusals), "ok": self.ok}


def validate_state(sim: SimulatedState, board, candidates: Dict[str, int],
                   *, market_scenario: str = "base") -> StateValidation:
    """Refuse a state that cannot support a frontier, and say why."""
    from .endgame import candidate_legal_max

    st = sim.state
    focus = st.focus_owner_id
    refusals: List[str] = []

    seen: set = set()
    dupes = False
    for o in st.owners:
        for pid in o.player_ids:
            if pid in seen:
                dupes = True
            seen.add(pid)
    total_spent = sum(o.spent for o in st.owners)
    started = sum(o.budget_start for o in st.owners)
    remaining = sum(o.budget_remaining for o in st.owners)
    reconcile = (total_spent + remaining) == started
    reserve_ok = all(o.budget_remaining >= o.open_slots * o.min_bid
                     for o in st.owners)

    owners = {}
    for o in st.owners:
        counts: Dict[str, int] = {}
        for f in o.filled:
            n = Position(int(f.position)).name
            counts[n] = counts.get(n, 0) + 1
        owners[o.owner_id] = {
            "budget_remaining": o.budget_remaining, "spent": o.spent,
            "open_slots": o.open_slots, "n_players": o.n_players,
            "positions": counts, "financial_max_bid": o.max_bid}

    pool = {}
    for p, n in st.available_by_position().items():
        pool[Position(int(p)).name] = int(n)

    cands: Dict[str, Dict[str, object]] = {}
    for label, cid in candidates.items():
        if not st.is_available(cid):
            refusals.append(f"{label}: candidate already sold in the "
                            f"simulated history")
            continue
        our_max = candidate_legal_max(st, focus, cid)
        rivals = {o.owner_id: candidate_legal_max(st, o.owner_id, cid)
                  for o in st.owners if o.owner_id != focus}
        top = max(rivals.values(), default=0)
        able = sum(1 for v in rivals.values() if v > 0)
        cands[label] = {
            "position": Position(int(st.spec(cid).position)).name,
            "our_legal_max": our_max,
            "highest_rival_legal_max": top,
            "owners_able_to_bid": able,
            "top_rival": max(rivals, key=lambda k: rivals[k]) if rivals else None}
        if our_max <= 0:
            refusals.append(f"{label}: we cannot legally bid at any price")

    if st.owner(focus).open_slots <= 0:
        refusals.append("focus roster is full")
    if not st.owner(focus).can_still_field_lineup(st.available_by_position()):
        refusals.append("no legal completion exists for the focus roster")
    if dupes:
        refusals.append("duplicate ownership")
    if not reconcile:
        refusals.append("dollars do not reconcile")
    if not reserve_ok:
        refusals.append("an owner has less than $1 per open slot")

    o = st.owner(focus)
    can_bind = o.budget_remaining < 200 and o.open_slots > 0
    if not can_bind:
        refusals.append("the focus budget cannot bind: this is an empty-room "
                        "state in disguise and no frontier can exist")

    return StateValidation(
        n_sales=len(sim.sales), total_spent=total_spent,
        dollars_reconcile=reconcile, duplicate_ownership=dupes,
        reserve_ok=reserve_ok, owners=owners, pool_remaining=pool,
        candidates=cands, focus_can_bind=can_bind, refusals=tuple(refusals))
