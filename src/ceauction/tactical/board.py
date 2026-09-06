"""Shared-board continuation: everyone finishes drafting out of one pool.

The largest simplification in the existing counterfactual layer is the frozen
:class:`ComparisonCast` -- eleven complete rival rosters, held fixed while our
completion is searched. It answers "which roster is best against *this*
league", and quietly assumes the rest of the room stops drafting the moment we
start thinking. Two consequences matter: a player can be simultaneously "on the
board" for us and already on a rival's assumed roster, and the money a rival
spends beating us to a player never leaves his budget.

This module replaces that with a bounded continuation in which everyone drafts
from **one common pool**:

* no player reaches two rosters,
* every unfinished roster is completed to fifteen legally, via the existing
  :mod:`ceauction.auction.feasibility` matching rules,
* every owner keeps $1 for each of his still-open slots at every step,
* budgets and open slots update after each allocation,
* the result is deterministic under a fixed seed and changes with the seed.

**Opponents are allocated by a named proxy, not by championship equity.**
Running the CE engine inside each of ~150 allocations is not affordable, and
pretending an opponent's beam-searched roster is CE-optimal when it was chosen
by scenario willingness would be a false claim. What opponents use is stated in
``method`` and repeated in every serialization.

The clearing rule is the same documented approximation the recipient layer
uses: second-highest willingness plus one increment, capped by the winner's own
willingness and his candidate-specific legal maximum.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np

from ..auction.completion import ComparisonCast
from ..auction.costs import CostBook
from ..auction.state import AuctionState
from ..league import Position
from ..market.live import MarketState
from ..auction.feasibility import PositionCounts, can_complete
from ..market.pressure import _roster_fit
from .bidders import BIDDER_SCENARIOS, DEFAULT_BIDDER_SCENARIO
from .endgame import candidate_legal_max

__all__ = [
    "BoardSettings",
    "Allocation",
    "BoardResult",
    "continue_shared_board",
    "cast_from_board",
    "format_board",
]

OPPONENT_METHOD = (
    "opponents allocated by SCENARIO WILLINGNESS over a shared pool, ordered by "
    "fabricated projection. This is a named proxy for behaviour and is NOT a "
    "championship-equity optimisation of any opponent's roster.")


@dataclass(frozen=True)
class BoardSettings:
    """Every knob the continuation has, and the bound it accepts."""

    seed: int = 20260906
    bidder_scenario: str = DEFAULT_BIDDER_SCENARIO
    market_scenario: str = "base"
    """Which point of the clearing band prices the pool: low/base/high."""
    pool_depth: int = 200
    """How far down the board the continuation looks. A real bound: a player
    below this cut cannot be allocated, and the result says whether it bit."""
    max_allocations: int = 200
    jitter: float = 0.06
    """Relative tie-breaking noise on willingness. Two owners with identical
    scenario willingness are not identical bidders in a real room, and a
    deterministic tie would hand every such player to the alphabetically first
    owner. Seeded, so a fixed seed reproduces exactly."""
    focus_bids: bool = True
    """Whether the focus team competes in the continuation.

    It must, and the default says so. With the focus team silent, eleven rivals
    draft the whole top of the board against an empty seat: they never have to
    outbid us, they get better players for less money, and our CE-backed search
    is handed the leftovers. Measured on the fabricated demo that put our proxy
    strength at 86.9 against rivals' 105-109 and drove our championship equity
    to exactly zero in BOTH branches, which made every audited buy/pass
    comparison a difference of two zeroes.

    What the focus team does NOT do here is take delivery. Players it wins are
    *held* -- kept out of rival hands, but left on our board and unpaid for --
    because which of them we actually end up with is the question the CE
    completion search exists to answer, and letting a willingness proxy decide
    it would replace the real search with the cheap one."""

    def cache_key(self) -> Tuple:
        import dataclasses
        return tuple((f.name, getattr(self, f.name))
                     for f in dataclasses.fields(self))

    def to_dict(self) -> Dict[str, object]:
        import dataclasses
        return {f.name: getattr(self, f.name)
                for f in dataclasses.fields(self)}


@dataclass(frozen=True)
class Allocation:
    """One player going to one owner at one price, inside the continuation."""

    sequence: int
    player_id: int
    position: str
    owner_id: str
    price: int
    runner_up: Optional[str]
    runner_up_willingness: int

    def to_dict(self) -> Dict[str, object]:
        return {"sequence": self.sequence, "player_id": self.player_id,
                "position": self.position, "owner_id": self.owner_id,
                "price": self.price, "runner_up": self.runner_up,
                "runner_up_willingness": self.runner_up_willingness}


@dataclass(frozen=True)
class BoardResult:
    """The completed room, and an honest label for how complete it is."""

    state: AuctionState
    """The auction state after every allocation. Legal by construction: each
    step goes through :meth:`AuctionState.apply_purchase`, which validates."""

    allocations: Tuple[Allocation, ...]
    settings: BoardSettings
    exactness: str
    """``exact`` | ``bounded`` | ``truncated`` | ``heuristic``"""
    unfilled: Dict[str, int]
    pool_considered: int
    pool_exhausted: bool
    held_for_focus: FrozenSet[int] = frozenset()
    """Players the focus team outbid the room for, kept out of rival rosters and
    left on our board for the CE-backed search to choose among. Not owned, not
    paid for, and never counted against our budget twice."""
    method: str = OPPONENT_METHOD

    @property
    def allocated_ids(self) -> FrozenSet[int]:
        return frozenset(a.player_id for a in self.allocations)

    @property
    def all_complete(self) -> bool:
        return not any(v > 0 for v in self.unfilled.values())

    @property
    def has_duplicates(self) -> bool:
        seen: set = set()
        for o in self.state.owners:
            for pid in o.player_ids:
                if pid in seen:
                    return True
                seen.add(pid)
        return False

    def reserved_ids(self, exclude_owner: Optional[str] = None) -> FrozenSet[int]:
        """Everyone another team now holds. What our own search may not touch."""
        return frozenset(
            pid for o in self.state.owners if o.owner_id != exclude_owner
            for pid in o.player_ids)

    def fingerprint(self) -> str:
        h = hashlib.sha256()
        h.update(f"settings={self.settings.cache_key()}\n".encode())
        for a in self.allocations:
            h.update(f"{a.sequence}|{a.player_id}|{a.owner_id}|{a.price}\n".encode())
        h.update(("held=" + ",".join(str(p) for p in sorted(self.held_for_focus))
                  + "\n").encode())
        return h.hexdigest()[:16]

    def to_dict(self, *, include_allocations: bool = True) -> Dict[str, object]:
        out: Dict[str, object] = {
            "fingerprint": self.fingerprint(),
            "settings": self.settings.to_dict(),
            "exactness": self.exactness,
            "method": self.method,
            "n_allocations": len(self.allocations),
            "pool_considered": self.pool_considered,
            "pool_exhausted": self.pool_exhausted,
            "unfilled_slots": self.unfilled,
            "held_for_focus": sorted(self.held_for_focus),
            "all_rosters_complete": self.all_complete,
            "duplicate_ownership": self.has_duplicates,
            "spend": {o.owner_id: o.spent for o in self.state.owners},
        }
        if include_allocations:
            out["allocations"] = [a.to_dict() for a in self.allocations]
        return out


def _pool_prices(state: AuctionState, costs: Optional[CostBook],
                 market: Optional[MarketState],
                 key_by_id: Optional[Dict[int, str]],
                 point: str) -> Dict[int, int]:
    """One expected clearing dollar per available player. Never zero."""
    out: Dict[int, int] = {}
    for spec in state.available_specs:
        price: Optional[int] = None
        key = key_by_id.get(spec.player_id) if key_by_id else None
        if market is not None and key:
            adj = market.adjusted(key)
            if adj is not None:
                price = {"low": adj.low, "base": adj.base, "high": adj.high}[point]
        if price is None and costs is not None and spec.player_id in costs:
            price = costs.cost_of(spec.player_id)
        out[spec.player_id] = max(1, int(price if price is not None else 1))
    return out


def _remaining_after(state: AuctionState,
                     taken: Position) -> Dict[Position, int]:
    """Board positions left once one player of ``taken`` is gone."""
    remaining = state.available_by_position()
    remaining[taken] = remaining.get(taken, 0) - 1
    return remaining


def _roster_fit_counts(counts: PositionCounts,
                       position: Position) -> Tuple[str, float]:
    """:func:`_roster_fit` against bare counts, for the focus shadow ledger.

    The focus team's continuation roster exists only in local variables -- it is
    never written into the auction state -- so it has no ``OwnerAuctionState``
    to hand the shared helper. Same rules, same numbers, different container.
    """
    class _Shim:
        pass

    shim = _Shim()
    shim.counts = counts                                  # type: ignore[attr-defined]
    return _roster_fit(None, shim, position)              # type: ignore[arg-type]


def continue_shared_board(
    state: AuctionState,
    *,
    settings: BoardSettings = BoardSettings(),
    costs: Optional[CostBook] = None,
    market: Optional[MarketState] = None,
    key_by_id: Optional[Dict[int, str]] = None,
    protect: Sequence[int] = (),
) -> BoardResult:
    """Finish every unfinished roster out of one shared pool. Deterministic.

    ``protect`` names players the continuation may not allocate -- typically
    the candidate under evaluation, whose destination is the whole question and
    must not be decided by a proxy behind the caller's back.
    """
    if settings.bidder_scenario not in BIDDER_SCENARIOS:
        raise ValueError(f"unknown bidder scenario {settings.bidder_scenario!r}")
    if settings.market_scenario not in ("low", "base", "high"):
        raise ValueError(
            f"market scenario must be low/base/high, got "
            f"{settings.market_scenario!r}")
    sc = BIDDER_SCENARIOS[settings.bidder_scenario]
    rng = np.random.default_rng(settings.seed)
    focus = state.focus_owner_id
    protected = set(protect)

    prices = _pool_prices(state, costs, market, key_by_id,
                          settings.market_scenario)
    order = sorted((s for s in state.available_specs
                    if s.player_id not in protected),
                   key=lambda s: (-prices[s.player_id], -s.base_mean,
                                  s.player_id))
    pool_exhausted = len(order) <= settings.pool_depth
    order = order[:settings.pool_depth]
    # One jitter draw per (player, owner) pair, taken up front so the sequence
    # does not depend on how many owners happen to be live at each step.
    n_owners = len(state.owners)
    noise = rng.normal(0.0, settings.jitter, size=(len(order), n_owners))
    owner_index = {o.owner_id: i for i, o in enumerate(state.owners)}

    allocations: List[Allocation] = []
    held: List[int] = []
    cur = state
    truncated = False
    # The focus team's shadow money and slots. It bids for real -- rivals must
    # outbid it or pay more -- but it never takes delivery, so its purchases
    # are tracked here instead of in the auction state. Without this ledger it
    # would win the entire board for free.
    focus_owner = state.owner(focus)
    focus_budget = focus_owner.budget_remaining
    focus_slots = focus_owner.open_slots
    focus_counts = focus_owner.counts

    for i, spec in enumerate(order):
        if len(allocations) + len(held) >= settings.max_allocations:
            truncated = True
            break
        rivals_done = all(o.open_slots <= 0 for o in cur.owners
                          if o.owner_id != focus)
        if rivals_done and (focus_slots <= 0 or not settings.focus_bids):
            break
        position = Position(int(spec.position))
        market_price = prices[spec.player_id]
        bids: List[Tuple[float, int, str]] = []
        for o in cur.owners:
            if o.owner_id == focus:
                if not settings.focus_bids or focus_slots <= 0:
                    continue
                # Same arithmetic the state applies to everyone: keep $1 for
                # every other open slot, and refuse a purchase that would leave
                # no legal completion.
                legal_max = focus_budget - (focus_slots - 1) * o.min_bid
                if legal_max < o.min_bid:
                    continue
                after = focus_counts.plus(position)
                if not can_complete(after, focus_slots - 1,
                                    _remaining_after(cur, position)):
                    continue
                fit_counts = after
                _, fit = _roster_fit_counts(focus_counts, position)
                discretionary = focus_budget - focus_slots * o.min_bid
            else:
                legal_max = candidate_legal_max(cur, o.owner_id, spec.player_id)
                if legal_max <= 0:
                    continue
                _, fit = _roster_fit(cur, o, position)
                discretionary = o.discretionary
            want = market_price * sc.aggression * (1.0 + sc.fit_weight * (fit - 0.5))
            want = min(want, sc.budget_share * max(0, discretionary) + o.min_bid)
            want *= (1.0 + float(noise[i, owner_index[o.owner_id]]))
            w = int(max(0, min(legal_max, round(want))))
            if w >= o.min_bid:
                bids.append((want, w, o.owner_id))
        if not bids:
            continue
        bids.sort(key=lambda b: (-b[0], b[2]))
        winner_want, winner_w, winner = bids[0]
        runner = bids[1] if len(bids) > 1 else None
        second = runner[1] if runner else 0
        price = max(cur.owner(winner).min_bid, second + 1)
        price = min(price, winner_w)
        if winner == focus:
            # Held, not bought. The player leaves the rivals' reach and stays
            # on our board, unpaid for; the CE completion search decides which
            # of these we actually take, inside our real budget.
            held.append(spec.player_id)
            focus_budget -= price
            focus_slots -= 1
            focus_counts = focus_counts.plus(position)
            allocations.append(Allocation(
                sequence=len(allocations), player_id=spec.player_id,
                position=position.name, owner_id=focus, price=price,
                runner_up=runner[2] if runner else None,
                runner_up_willingness=second))
            continue
        problem = cur.purchase_shortfall(spec.player_id, winner, price)
        if problem is not None:
            # Legality is the authority, not the willingness model. Skipping is
            # correct: this owner cannot hold this player, so the room simply
            # does not produce that sale.
            continue
        cur = cur.apply_purchase(spec.player_id, winner, price)
        allocations.append(Allocation(
            sequence=len(allocations), player_id=spec.player_id,
            position=position.name, owner_id=winner, price=price,
            runner_up=runner[2] if runner else None,
            runner_up_willingness=second))

    unfilled = {o.owner_id: o.open_slots for o in cur.owners
                if o.owner_id != focus}
    if truncated:
        exactness = "truncated"
    elif any(v > 0 for v in unfilled.values()):
        exactness = "truncated"
    elif not pool_exhausted:
        exactness = "bounded"
    else:
        exactness = "bounded"
    return BoardResult(state=cur, allocations=tuple(allocations),
                       settings=settings, exactness=exactness,
                       unfilled=unfilled, pool_considered=len(order),
                       pool_exhausted=pool_exhausted,
                       held_for_focus=frozenset(held))


def cast_from_board(board: BoardResult, cast: ComparisonCast,
                    state: AuctionState) -> ComparisonCast:
    """Turn a finished continuation into a comparison cast, keeping our slot free.

    Every rival team comes from the continuation, so no two of them hold the
    same player and none of them holds anyone we could still buy. The focus
    slot keeps whatever placeholder the incoming cast had; the completion
    search overwrites it.
    """
    rosters: List[Tuple[int, ...]] = []
    for i, name in enumerate(cast.team_names):
        if i == cast.focus_team_index:
            rosters.append(cast.rosters[i])
            continue
        owner = board.state.owner_by_id.get(name)
        rosters.append(tuple(owner.player_ids) if owner is not None
                       else cast.rosters[i])
    return ComparisonCast(cast.focus_team_index, tuple(rosters), cast.team_names)


def format_board(board: BoardResult, width: int = 92, show: int = 12) -> str:
    bar = "=" * width
    out = [bar, "SHARED-BOARD CONTINUATION", bar,
           f"seed                 {board.settings.seed}",
           f"bidder scenario      {board.settings.bidder_scenario}",
           f"market scenario      {board.settings.market_scenario}",
           f"pool depth / seen    {board.settings.pool_depth} / "
           f"{board.pool_considered}"
           f"{'' if board.pool_exhausted else '  (BOUND BIT: board deeper than the cut)'}",
           f"allocations          {len(board.allocations)}",
           f"exactness            {board.exactness}",
           f"all rosters complete {board.all_complete}",
           f"duplicate ownership  {board.has_duplicates}",
           f"fingerprint          {board.fingerprint()}", "",
           f"  first {show} allocations:",
           f"  {'#':<4}{'player':>8}{'pos':>5}{'owner':>10}{'price':>7}"
           f"{'runner-up':>12}{'2nd want':>10}"]
    for a in board.allocations[:show]:
        out.append(f"  {a.sequence:<4}{a.player_id:>8}{a.position:>5}"
                   f"{a.owner_id:>10}{a.price:>7}{a.runner_up or '-':>12}"
                   f"{a.runner_up_willingness:>10}")
    out += ["", "  unfilled slots after the continuation:",
            "  " + ", ".join(f"{k}:{v}" for k, v in sorted(board.unfilled.items())),
            "", f"  {board.method}", bar]
    return "\n".join(out)
