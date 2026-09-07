"""The full-budget completion union, built from $1 upward.

The reconciled run produced a **negative** possession effect for both real
candidates (RB `-0.09480`, QB `-0.03277`) where the earlier decomposition had
it positive. The suspicion this module tests: the ``with_candidate`` union was
generated over the tested bid prices only (`$65..$100` for the RB), so it never
contained constructions affordable only when little or nothing is spent on the
candidate. ``UF`` pays nothing and must see those; if it cannot, possession is
measured against an offer that was already cut by a price ``UF`` never pays.

**Three prices, three names, never conflated.** Merging them is how a search
artifact becomes a quoted bid:

``search_support_price``  a price used only to DISCOVER roster constructions.
                          $1, $5, $10, $20 are support prices. With a $30
                          standing bid they are **not legal live bids** and
                          must never be printed as prices we could pay.
``evaluated_bid_price``   a price we actually evaluate as a purchase: the legal
                          next bid `q+1` and above.
``standing_pass_price``   the named rival's current high bid `q`.

:class:`PriceRole` tags every price with which it is, and
:func:`assert_live_biddable` refuses to let a support price below the standing
bid be reported as a live purchase option.

The union accumulates **monotonically**: a later, wider search may add
constructions but may never remove one an earlier search found. That is what
makes "more effort" safe, and it is checked rung by rung rather than assumed.
"""

from __future__ import annotations

import dataclasses
import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

from ..auction.completion import (Completion, CompletionSettings,
                                  complete_roster)
from ..auction.costs import CostBook
from ..auction.proxy import ProxyEvaluator
from ..auction.state import AuctionState

__all__ = [
    "SUPPORT_PRICES",
    "PriceRole",
    "PricedValue",
    "LiveBidMisuse",
    "assert_live_biddable",
    "completion_fingerprint",
    "UnionRung",
    "CompletionUnionReport",
    "build_full_budget_union",
    "UNION_NOT_CONVERGED",
    "UNION_CONVERGED",
    "UnionAdequacy",
    "check_union_adequacy",
]

#: Support prices, lowest first. The $1 rung must come first: it is the widest
#: budget any branch can have, so it discovers the constructions ``UF`` needs.
SUPPORT_PRICES: Tuple[int, ...] = (1, 5, 10, 20, 40, 65, 80, 100)

UNION_CONVERGED = "UNION_CONVERGED"
UNION_NOT_CONVERGED = "UNION_NOT_CONVERGED"


class LiveBidMisuse(RuntimeError):
    """A search-support price was presented as a live purchase price."""


class PriceRole(Enum):
    SEARCH_SUPPORT = "search_support_price"
    EVALUATED_BID = "evaluated_bid_price"
    STANDING_PASS = "standing_pass_price"


@dataclass(frozen=True)
class PricedValue:
    """A dollar figure that knows what kind of price it is."""

    amount: int
    role: PriceRole

    @property
    def is_live_bid(self) -> bool:
        return self.role is PriceRole.EVALUATED_BID

    def to_dict(self) -> Dict[str, object]:
        return {"amount": self.amount, "role": self.role.value,
                "is_live_bid": self.is_live_bid}


def assert_live_biddable(price: PricedValue, standing_pass_price: int,
                         *, increment: int = 1) -> None:
    """Refuse to treat a support price -- or anything at/below `q` -- as a bid.

    With a $30 standing bid, $1/$5/$10/$20 discover roster constructions and
    nothing more. Printing one as a purchase option would invite a bid that is
    not even legal, which is a worse failure than any modelling error here.
    """
    if price.role is not PriceRole.EVALUATED_BID:
        raise LiveBidMisuse(
            f"${price.amount} is a {price.role.value}, not a live bid. Support "
            f"prices exist only to discover roster constructions and must "
            f"never be reported as prices we could pay.")
    floor = standing_pass_price + increment
    if price.amount < floor:
        raise LiveBidMisuse(
            f"${price.amount} is below the legal next bid of ${floor} "
            f"(standing pass price ${standing_pass_price} + ${increment}); it "
            f"cannot be a live purchase price.")


def completion_fingerprint(c: Completion) -> str:
    """Complete economic/roster identity: who is on it and what it cost."""
    body = ",".join(str(p) for p in sorted(c.roster)) + f"|cost={c.added_cost}"
    return hashlib.sha256(body.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class UnionRung:
    """What one support-price search contributed to the accumulating union."""

    search_support_price: int
    generated: int
    newly_unique: int
    cumulative_unique: int
    min_cost: int
    max_cost: int
    roster_fingerprints: Tuple[str, ...]
    search_effort_nested: bool
    cross_price_nested: bool

    def to_dict(self) -> Dict[str, object]:
        return {
            "search_support_price": self.search_support_price,
            "role": PriceRole.SEARCH_SUPPORT.value,
            "generated": self.generated, "newly_unique": self.newly_unique,
            "cumulative_unique": self.cumulative_unique,
            "min_completion_cost": self.min_cost,
            "max_completion_cost": self.max_cost,
            "n_roster_fingerprints": len(self.roster_fingerprints),
            "search_effort_nested": self.search_effort_nested,
            "cross_price_nested": self.cross_price_nested,
        }


@dataclass
class CompletionUnionReport:
    """The accumulated union and the rung-by-rung evidence for it."""

    candidate_id: int
    rungs: Tuple[UnionRung, ...]
    completions: Tuple[Completion, ...]
    focus_budget: int

    @property
    def size(self) -> int:
        return len(self.completions)

    @property
    def keys(self) -> FrozenSet[str]:
        return frozenset(completion_fingerprint(c) for c in self.completions)

    @property
    def accumulated_monotonically(self) -> bool:
        sizes = [r.cumulative_unique for r in self.rungs]
        return sizes == sorted(sizes)

    @property
    def started_at_one(self) -> bool:
        return bool(self.rungs) and self.rungs[0].search_support_price == 1

    def fingerprint(self) -> str:
        h = hashlib.sha256()
        for k in sorted(self.keys):
            h.update(k.encode())
        return h.hexdigest()[:16]

    def to_dict(self) -> Dict[str, object]:
        return {
            "union_fingerprint": self.fingerprint(),
            "size": self.size,
            "focus_budget": self.focus_budget,
            "started_at_one_dollar": self.started_at_one,
            "accumulated_monotonically": self.accumulated_monotonically,
            "rungs": [r.to_dict() for r in self.rungs],
            "note": ("every price above is a SEARCH SUPPORT price used only to "
                     "discover roster constructions; none is a live bid"),
        }


def _affordable_keys(seen: Dict[str, Completion], costs: CostBook,
                     budget: int, support_price: int,
                     owned_after: FrozenSet[int], default_cost: int
                     ) -> FrozenSet[str]:
    """Union members payable once ``support_price`` is spent on the candidate."""
    room = budget - support_price
    out = set()
    for k, c in seen.items():
        spend = sum(costs.cost_of(pid, default_cost) for pid in c.roster
                    if pid not in owned_after)
        if spend <= room:
            out.add(k)
    return frozenset(out)


def build_full_budget_union(
    state: AuctionState, cast, costs: CostBook, candidate_id: int, *,
    completion: CompletionSettings, proxy: ProxyEvaluator,
    support_prices: Sequence[int] = SUPPORT_PRICES,
    legal_max: Optional[int] = None, default_cost: int = 1,
) -> CompletionUnionReport:
    """Search at each support price; keep everything ever found.

    Starts at $1 deliberately. That rung leaves the largest budget for the other
    fourteen slots, so it is where the constructions ``UF`` can afford are
    found; starting higher is exactly the restriction that made possession
    negative.
    """
    focus = state.focus_owner_id
    budget = state.owner(focus).budget_remaining
    prices = sorted({int(p) for p in support_prices
                     if state.purchase_shortfall(candidate_id, focus,
                                                 int(p)) is None})
    if legal_max is not None and legal_max >= 1 and \
            state.purchase_shortfall(candidate_id, focus, legal_max) is None:
        prices = sorted(set(prices) | {int(legal_max)})
    if not prices:
        raise ValueError("no legal support price for this candidate")
    if prices[0] != 1:
        # Not fatal, but it is the exact restriction under investigation.
        pass

    owned_after = frozenset(state.owner(focus).player_ids) | {candidate_id}
    seen: Dict[str, Completion] = {}
    rungs: List[UnionRung] = []
    prev_keys: FrozenSet[str] = frozenset()
    prev_price_keys: Optional[FrozenSet[str]] = None
    prev_p = prices[0]
    for p in prices:
        bought = state.apply_purchase(candidate_id, focus, p)
        res = complete_roster(bought, cast, costs, settings=completion,
                              owner_id=focus, evaluate_ce=False,
                              default_cost=default_cost, proxy=proxy,
                              reserved_ids=frozenset(),
                              notes=f"union support search at ${p}")
        found = list(res.finalists) or ([res.best] if res.best else [])
        this_keys = set()
        new = 0
        for c in found:
            k = completion_fingerprint(c)
            this_keys.add(k)
            if k not in seen:
                seen[k] = c
                new += 1
        keys_now = frozenset(seen)
        # Search-effort nesting: nothing an earlier rung found may vanish.
        # Structural under a union (we only ever add), so this VERIFIES the
        # implementation rather than discovering a fact about the search.
        effort_ok = prev_keys <= keys_now
        # Cross-price nesting, verified the way it is actually consumed: the
        # affordable slice of the union at a dearer support price must be a
        # subset of the slice at every cheaper one. Also structural under a
        # single cost threshold over one union -- which is precisely why the
        # union fixes the price-dependence the restricted build suffered.
        cross_ok = True
        if prev_price_keys is not None:
            cross_ok = (_affordable_keys(seen, costs, budget, p, owned_after,
                                         default_cost)
                        <= _affordable_keys(seen, costs, budget, prev_p,
                                            owned_after, default_cost))
        costs_seen = [c.added_cost for c in seen.values()]
        rungs.append(UnionRung(
            search_support_price=p, generated=len(found), newly_unique=new,
            cumulative_unique=len(seen),
            min_cost=min(costs_seen) if costs_seen else 0,
            max_cost=max(costs_seen) if costs_seen else 0,
            roster_fingerprints=tuple(sorted(this_keys)),
            search_effort_nested=effort_ok, cross_price_nested=cross_ok))
        prev_keys = keys_now
        prev_price_keys = frozenset(this_keys)
        prev_p = p

    ordered = tuple(seen[k] for k in sorted(seen))
    return CompletionUnionReport(candidate_id=candidate_id, rungs=tuple(rungs),
                                 completions=ordered, focus_budget=budget)


@dataclass
class UnionAdequacy:
    """Restricted vs full union, and whether more effort would change it."""

    restricted_size: int
    full_size: int
    overlap: int
    added_by_low_prices: int
    restricted_uf_fingerprint: str
    full_uf_fingerprint: str
    restricted_uf_proxy: float
    full_uf_proxy: float
    extra_effort_size: int
    extra_effort_uf_fingerprint: str
    extra_effort_uf_proxy: float
    tolerance: float = 0.25

    @property
    def uf_selection_changed(self) -> bool:
        return self.restricted_uf_fingerprint != self.full_uf_fingerprint

    @property
    def extra_effort_changed_selection(self) -> bool:
        return self.extra_effort_uf_fingerprint != self.full_uf_fingerprint

    @property
    def extra_effort_proxy_delta(self) -> float:
        return self.extra_effort_uf_proxy - self.full_uf_proxy

    @property
    def status(self) -> str:
        """Converged over the EVALUATED ACCUMULATED UNION, not exhaustively.

        The claim is bounded on purpose: more search effort did not change the
        chosen construction or move its strength beyond tolerance. It is not a
        claim that no better roster exists anywhere in the combinatorial space.
        """
        if self.extra_effort_changed_selection and \
                abs(self.extra_effort_proxy_delta) > self.tolerance:
            return UNION_NOT_CONVERGED
        return UNION_CONVERGED

    @property
    def may_quote_bracket(self) -> bool:
        return self.status == UNION_CONVERGED

    def to_dict(self) -> Dict[str, object]:
        return {
            "restricted_union_size": self.restricted_size,
            "full_union_size": self.full_size,
            "overlap": self.overlap,
            "added_by_low_price_searches": self.added_by_low_prices,
            "uf_selection_changed": self.uf_selection_changed,
            "restricted_uf_proxy": round(self.restricted_uf_proxy, 4),
            "full_uf_proxy": round(self.full_uf_proxy, 4),
            "uf_proxy_gain": round(self.full_uf_proxy
                                   - self.restricted_uf_proxy, 4),
            "extra_effort_union_size": self.extra_effort_size,
            "extra_effort_changed_selection":
                self.extra_effort_changed_selection,
            "extra_effort_proxy_delta": round(self.extra_effort_proxy_delta, 4),
            "tolerance": self.tolerance,
            "status": self.status,
            "may_quote_bracket": self.may_quote_bracket,
            "convergence_scope": (
                "convergence is over the EVALUATED ACCUMULATED UNION -- more "
                "search effort did not change the chosen construction. It is "
                "NOT a claim that no better roster exists in the full space."),
        }


def _best(completions: Sequence[Completion]
          ) -> Tuple[Optional[Completion], float]:
    """Strongest affordable construction, by the search's own proxy strength.

    Ties are broken by fingerprint so the "chosen completion" is reproducible;
    an unbroken tie would make the adequacy comparison report a spurious change
    of selection between two rosters the model cannot tell apart.
    """
    if not completions:
        return None, float("-inf")
    best = max(completions,
               key=lambda c: (c.proxy, completion_fingerprint(c)))
    return best, best.proxy


def check_union_adequacy(
    state: AuctionState, cast, costs: CostBook, candidate_id: int, *,
    restricted: CompletionUnionReport, full: CompletionUnionReport,
    completion: CompletionSettings, proxy: ProxyEvaluator,
    extra_effort: Optional[CompletionSettings] = None,
    default_cost: int = 1, tolerance: float = 0.25,
) -> UnionAdequacy:
    """Compare the two unions, then spend more effort and see if it matters."""
    focus = state.focus_owner_id
    budget = state.owner(focus).budget_remaining
    owned = frozenset(state.owner(focus).player_ids)
    extra_owned = owned | frozenset({candidate_id})

    def affordable(rep: CompletionUnionReport) -> List[Completion]:
        out = []
        for c in rep.completions:
            cost = sum(costs.cost_of(pid, default_cost) for pid in c.roster
                       if pid not in extra_owned)
            if cost <= budget:
                out.append(c)
        return out

    r_aff, f_aff = affordable(restricted), affordable(full)
    r_best, r_proxy = _best(r_aff)
    f_best, f_proxy = _best(f_aff)

    if extra_effort is None:
        extra_effort = dataclasses.replace(
            completion, beam_width=completion.beam_width * 3,
            candidate_pool=int(completion.candidate_pool * 1.8),
            max_candidates=completion.max_candidates * 3,
            finalists=max(completion.finalists, 6))
    boosted = build_full_budget_union(
        state, cast, costs, candidate_id, completion=extra_effort, proxy=proxy,
        support_prices=SUPPORT_PRICES, default_cost=default_cost)
    merged = dict((completion_fingerprint(c), c) for c in full.completions)
    for c in boosted.completions:
        merged.setdefault(completion_fingerprint(c), c)
    e_aff = affordable(CompletionUnionReport(
        candidate_id=candidate_id, rungs=(), focus_budget=budget,
        completions=tuple(merged.values())))
    e_best, e_proxy = _best(e_aff)

    return UnionAdequacy(
        restricted_size=restricted.size, full_size=full.size,
        overlap=len(restricted.keys & full.keys),
        added_by_low_prices=len(full.keys - restricted.keys),
        restricted_uf_fingerprint=(completion_fingerprint(r_best)
                                   if r_best else "none"),
        full_uf_fingerprint=(completion_fingerprint(f_best)
                             if f_best else "none"),
        restricted_uf_proxy=r_proxy, full_uf_proxy=f_proxy,
        extra_effort_size=len(merged),
        extra_effort_uf_fingerprint=(completion_fingerprint(e_best)
                                     if e_best else "none"),
        extra_effort_uf_proxy=e_proxy, tolerance=tolerance)
