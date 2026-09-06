"""Nested opportunity sets across a price ladder.

The repaired 4,000-season sweep reported a player as worth *less* bought at $1
than bought at $5. That cannot be true. Every roster affordable with $105 is
affordable with $109; the extra $4 may simply go unspent. So the result was an
artefact, and this module removes the artefact rather than smoothing the curve.

**The cause, proven rather than guessed.** The best construction found at $5
cost $95, we hold $109 at $1, and no rival had taken any of its players -- it
was affordable and legal at $1 and the search simply never found it. What
differed was the *board the beam was handed*: the shadow continuation depends on
our own remaining money, so at $1 we shadow-won {.., 271} and at $5 we
shadow-won {.., 93}, and a two-player difference in the input was enough to send
a heuristic beam down a worse path. Market state, rival budgets, seeds and
player scoring were all identical across prices and are ruled out by test.

**The fix.** Generate focus completions at every tested price, take the union,
and then offer that whole union back to every price, keeping the entries that
are legal and affordable there. Feasibility nesting then holds by construction:
a construction that survives the filter at $5 also survives it at $1, because
the filter is a budget comparison and our budget at $1 is strictly larger.

What this does *not* claim: that the union is the complete set of legal
constructions. It is the union of what a bounded beam found, which is a
heuristic set, and :attr:`PriceOpportunities.exactness` says so. Nesting is a
property of the set we actually evaluate, and that is the property the economics
requires.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

from ..auction.completion import (ComparisonCast, Completion,
                                  CompletionSettings, complete_roster)
from ..auction.costs import CostBook
from ..auction.proxy import ProxyEvaluator
from ..auction.state import AuctionState
from ..market.live import MarketState
from .board import BoardSettings, cast_from_board, continue_shared_board

__all__ = [
    "PriceOpportunities",
    "NestedLadder",
    "roster_fingerprint",
    "build_nested_ladder",
    "format_nested_ladder",
]


def roster_fingerprint(roster: Sequence[int],
                       prices: Optional[Dict[int, int]] = None) -> str:
    """Identity of a construction: who is on it and, optionally, at what cost."""
    parts = []
    for pid in sorted(roster):
        if prices is not None and pid in prices:
            parts.append(f"{pid}:{prices[pid]}")
        else:
            parts.append(str(pid))
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class PriceOpportunities:
    """The choice set actually offered at one candidate price."""

    price: int
    generated: Tuple[Completion, ...]
    """What the beam found with this price's own shadow board."""
    inherited: Tuple[Completion, ...]
    """Constructions first found at some OTHER price and imported here."""
    feasible: Tuple[Completion, ...]
    """Everything legal and affordable at this price. Generated plus inherited,
    deduplicated, minus whatever this price cannot pay for."""
    rejected_unaffordable: Tuple[str, ...]
    rejected_illegal: Tuple[str, ...]
    budget_for_completion: int
    open_slots: int

    @property
    def n_generated(self) -> int:
        return len(self.generated)

    @property
    def n_inherited(self) -> int:
        return len(self.inherited)

    @property
    def n_feasible(self) -> int:
        return len(self.feasible)

    @property
    def feasible_keys(self) -> FrozenSet[str]:
        return frozenset(roster_fingerprint(c.roster) for c in self.feasible)

    def to_dict(self) -> Dict[str, object]:
        return {
            "price": self.price,
            "n_generated": self.n_generated,
            "n_inherited": self.n_inherited,
            "n_feasible": self.n_feasible,
            "budget_for_completion": self.budget_for_completion,
            "open_slots": self.open_slots,
            "rejected_unaffordable": len(self.rejected_unaffordable),
            "rejected_illegal": len(self.rejected_illegal),
            "feasible_fingerprints": sorted(self.feasible_keys),
        }


@dataclass(frozen=True)
class NestedLadder:
    """One choice set per price, provably nested in the money direction."""

    candidate_id: int
    prices: Tuple[int, ...]
    by_price: Dict[int, PriceOpportunities]
    union_size: int
    exactness: str = ("heuristic union of bounded beam searches; nesting is "
                      "exact over the evaluated set, not over all legal rosters")

    def nesting_violations(self) -> Tuple[Tuple[int, int, str], ...]:
        """``(lower_price, higher_price, roster)`` a lower price failed to offer.

        Empty is the invariant. Non-empty is a bug, and the tuple names it
        rather than leaving a reader to infer it from a curve.
        """
        out: List[Tuple[int, int, str]] = []
        ordered = sorted(self.prices)
        for i, lo in enumerate(ordered):
            lo_keys = self.by_price[lo].feasible_keys
            for hi in ordered[i + 1:]:
                for key in self.by_price[hi].feasible_keys:
                    if key not in lo_keys:
                        out.append((lo, hi, key))
        return tuple(out)

    @property
    def is_nested(self) -> bool:
        return not self.nesting_violations()

    def fingerprint(self) -> str:
        h = hashlib.sha256()
        h.update(f"candidate={self.candidate_id}\n".encode())
        for p in sorted(self.prices):
            h.update((f"{p}=" + ",".join(sorted(self.by_price[p].feasible_keys))
                      + "\n").encode())
        return h.hexdigest()[:16]

    def to_dict(self) -> Dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "prices": list(self.prices),
            "union_size": self.union_size,
            "fingerprint": self.fingerprint(),
            "is_nested": self.is_nested,
            "nesting_violations": [list(v) for v in self.nesting_violations()],
            "exactness": self.exactness,
            "by_price": {str(p): o.to_dict() for p, o in
                         sorted(self.by_price.items())},
        }


def _cost_of(costs: CostBook, roster: Sequence[int], owned: FrozenSet[int],
             default_cost: int) -> int:
    return sum(costs.cost_of(pid, default_cost) for pid in roster
               if pid not in owned)


def build_nested_ladder(
    state: AuctionState,
    cast: ComparisonCast,
    costs: CostBook,
    candidate_id: int,
    prices: Sequence[int],
    *,
    board_settings: BoardSettings = BoardSettings(),
    completion: CompletionSettings = CompletionSettings(),
    market: Optional[MarketState] = None,
    key_by_id: Optional[Dict[int, str]] = None,
    proxy: Optional[ProxyEvaluator] = None,
    default_cost: int = 1,
) -> NestedLadder:
    """Generate at every price, union, then re-offer the union to every price.

    ``state`` is the room *before* the candidate is bought. Each price gets its
    own post-purchase room, its own shadow board and its own beam; the union of
    everything found is then filtered back through each price's budget and
    roster legality, which is what makes the surviving sets nested.
    """
    focus = state.focus_owner_id
    if proxy is None:
        proxy = ProxyEvaluator(state.pool, state.settings,
                               completion.proxy_reps, completion.proxy_seed)
    tested = tuple(sorted({int(p) for p in prices}))

    generated: Dict[int, Tuple[Completion, ...]] = {}
    post: Dict[int, AuctionState] = {}
    for price in tested:
        if state.purchase_shortfall(candidate_id, focus, price) is not None:
            continue
        bought = state.apply_purchase(candidate_id, focus, price)
        post[price] = bought
        shadow = continue_shared_board(bought, settings=board_settings,
                                       costs=costs, market=market,
                                       key_by_id=key_by_id)
        shadow_cast = cast_from_board(shadow, cast, bought)
        res = complete_roster(
            shadow.state, shadow_cast, costs, settings=completion,
            owner_id=focus, evaluate_ce=False, default_cost=default_cost,
            proxy=proxy,
            reserved_ids=shadow.reserved_ids(exclude_owner=focus),
            notes=f"nested ladder generation at ${price}")
        found = list(res.finalists) or ([res.best] if res.best else [])
        generated[price] = tuple(found)

    # The union, deduplicated on roster identity.
    union: Dict[str, Completion] = {}
    origin: Dict[str, int] = {}
    for price in tested:
        for c in generated.get(price, ()):
            key = roster_fingerprint(c.roster)
            if key not in union:
                union[key] = c
                origin[key] = price

    by_price: Dict[int, PriceOpportunities] = {}
    for price in tested:
        if price not in post:
            continue
        bought = post[price]
        owner = bought.owner(focus)
        owned = frozenset(owner.player_ids)
        budget = owner.budget_remaining
        feasible: List[Completion] = []
        unaffordable: List[str] = []
        illegal: List[str] = []
        for key, c in union.items():
            # Legality first: the construction must contain what we already
            # hold at this price (the candidate included) and nothing owned by
            # anyone else.
            roster = set(c.roster)
            if not owned <= roster:
                illegal.append(key)
                continue
            if any(bought.owner_of.get(pid) not in (None, focus)
                   for pid in roster):
                illegal.append(key)
                continue
            if len(roster) != owner.roster_capacity:
                illegal.append(key)
                continue
            cost = _cost_of(costs, c.roster, owned, default_cost)
            if cost > budget:
                unaffordable.append(key)
                continue
            feasible.append(c)
        gen_keys = {roster_fingerprint(c.roster)
                    for c in generated.get(price, ())}
        by_price[price] = PriceOpportunities(
            price=price, generated=generated.get(price, ()),
            inherited=tuple(c for c in feasible
                            if roster_fingerprint(c.roster) not in gen_keys),
            feasible=tuple(feasible),
            rejected_unaffordable=tuple(sorted(unaffordable)),
            rejected_illegal=tuple(sorted(illegal)),
            budget_for_completion=budget, open_slots=owner.open_slots)

    return NestedLadder(candidate_id=candidate_id,
                        prices=tuple(sorted(by_price)), by_price=by_price,
                        union_size=len(union))


def format_nested_ladder(ladder: NestedLadder, width: int = 92) -> str:
    bar = "=" * width
    out = [bar, "NESTED OPPORTUNITY SETS", bar,
           f"candidate     player {ladder.candidate_id}",
           f"union size    {ladder.union_size}",
           f"fingerprint   {ladder.fingerprint()}",
           f"nested        {ladder.is_nested}", "",
           f"  {'$':>5}{'budget':>9}{'slots':>7}{'generated':>11}"
           f"{'inherited':>11}{'feasible':>10}{'unafford':>10}{'illegal':>9}"]
    for p in sorted(ladder.prices):
        o = ladder.by_price[p]
        out.append(f"  {p:>5}{o.budget_for_completion:>9}{o.open_slots:>7}"
                   f"{o.n_generated:>11}{o.n_inherited:>11}{o.n_feasible:>10}"
                   f"{len(o.rejected_unaffordable):>10}"
                   f"{len(o.rejected_illegal):>9}")
    viol = ladder.nesting_violations()
    out += ["", f"  nesting violations: {len(viol)}"]
    for lo, hi, key in viol[:5]:
        out.append(f"    ${lo} does not offer a construction feasible at ${hi} "
                   f"({key})")
    out += ["", f"  {ladder.exactness}", bar]
    return "\n".join(out)
