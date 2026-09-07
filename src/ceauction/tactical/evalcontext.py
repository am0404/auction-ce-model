"""One immutable evaluation context, consumed by the frontier AND the decomposition.

The RB disagreed with itself: the ladder said ``+0.01298`` (favorable) at
``p=$80, q=$30`` while the five-branch decomposition said ``-0.00366``. The
cause was not a modelling choice; it was three different completion searches
being run for what was supposed to be one world:

    frontier buy arm   completions=<nested pre-pass set>
    frontier pass arm  (none) -> independent beam
    decompose, all 5   (none) -> independent beam

So the ladder's own buy and pass arms already disagreed with each other, and
the decomposition disagreed with both. Whichever number you preferred, at least
two of the three were describing a different auction.

This module makes that structurally impossible. :class:`EvalContext` carries
every input needed to evaluate one candidate at one price -- state, board,
candidate, ``p``, ``q``, the pass rule, the recipient, the cost book, the
scenario, completion settings, the accumulated nested opportunity set, the
allocation rotation, and both sample seeds -- and hands the SAME opportunity
set to every branch that should see it. A consumer that is given a context and
searches anyway raises :class:`IndependentSearchRefused`.

Two opportunity sets, not one, because the branches differ in who holds the
candidate:

``with_candidate``     every completion containing him, at the WIDEST budget
                       any branch could have -- i.e. the union across the whole
                       price ladder, not the set already narrowed to price
                       ``p``. Serves ``UF`` and ``UP``, each filtered by the
                       budget that branch actually has. Handing in a
                       price-filtered set instead makes ``UF`` (which pays
                       nothing) inherit ``UP``'s affordability cut, and the
                       possession effect then moves with a price it does not
                       depend on -- a bug this module hit and now guards.
``without_candidate``  every completion not containing him. Serves ``W``,
                       ``RF`` and ``RP`` -- our budget is untouched in all three,
                       so one set covers them.

Filtering by budget rather than re-searching is what preserves cross-price
nesting: a construction affordable at a higher price is still in the union at
every lower one.
"""

from __future__ import annotations

import dataclasses
import hashlib
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

from ..auction.completion import ComparisonCast, Completion, CompletionSettings
from ..auction.costs import CostBook
from ..auction.state import AuctionState
from ..market.live import MarketState
from .board import BoardSettings
from .ensemble import AllocationDraw
from .midauction import PassPrice

__all__ = [
    "EvalContext",
    "ContextMismatch",
    "IndependentSearchRefused",
    "BRANCH_HOLDS_CANDIDATE",
    "build_eval_context",
]


class ContextMismatch(RuntimeError):
    """Two evaluations were asked to share a world they do not agree on."""


class IndependentSearchRefused(RuntimeError):
    """A consumer holding a shared context tried to run its own beam."""


#: Which decomposition branches put the candidate on OUR roster.
BRANCH_HOLDS_CANDIDATE: Dict[str, bool] = {
    "W": False, "UF": True, "UP": True, "RF": False, "RP": False}


def _cost_of(costs: CostBook, roster: Sequence[int], owned: FrozenSet[int],
             default_cost: int) -> int:
    return sum(costs.cost_of(pid, default_cost) for pid in roster
               if pid not in owned)


@dataclass(frozen=True)
class EvalContext:
    """Everything needed to evaluate one candidate at one price. Immutable."""

    state: AuctionState
    cast: ComparisonCast
    costs: CostBook
    candidate_id: int
    p: int
    q: int
    pass_price: PassPrice
    recipient: str
    market: Optional[MarketState]
    key_by_id: Optional[Dict[int, str]]
    scenario: str
    completion: CompletionSettings
    board: BoardSettings
    draw: AllocationDraw
    selection_sims: int
    selection_seed: int
    holdout_sims: int
    holdout_seed: int
    with_candidate: Tuple[Completion, ...]
    """The UNION over the price ladder, never a price-``p`` slice. See the
    module docstring: ``UF`` pays nothing and must not inherit ``UP``'s cut."""
    without_candidate: Tuple[Completion, ...]
    default_cost: int = 1
    max_worlds: int = 2

    # --- identity ---------------------------------------------------------

    def opportunity_fingerprint(self) -> str:
        """Digest of the accumulated opportunity set itself.

        Rosters only, order-insensitive: two contexts offering the same
        constructions are the same offer however they were discovered.
        """
        h = hashlib.sha256()
        for label, group in (("with", self.with_candidate),
                             ("without", self.without_candidate)):
            h.update(label.encode())
            for key in sorted(",".join(str(x) for x in sorted(c.roster))
                              for c in group):
                h.update(key.encode())
                h.update(b"\n")
        return h.hexdigest()[:16]

    def fingerprint(self) -> str:
        """Digest of every input that could change an answer."""
        parts = [
            f"auction={self.state.fingerprint()}",
            f"pool={self.state.pool_fingerprint()}",
            f"cast={self.cast.fingerprint()}",
            f"costs={self.costs.fingerprint()}",
            f"market={'none' if self.market is None else self.market.fingerprint()}",
            f"candidate={self.candidate_id}",
            f"p={self.p}", f"q={self.q}",
            f"pass_rule={self.pass_price.mode.value}",
            f"increment={self.pass_price.increment}",
            f"standing={self.pass_price.standing_price}",
            f"fixed_q={self.pass_price.fixed_q}",
            f"recipient={self.recipient}",
            f"scenario={self.scenario}",
            f"completion={self.completion.cache_key()}",
            f"board={self.board.cache_key()}",
            f"draw={self.draw.key()}",
            f"selection={self.selection_sims}:{self.selection_seed}",
            f"holdout={self.holdout_sims}:{self.holdout_seed}",
            f"opportunity={self.opportunity_fingerprint()}",
            f"default_cost={self.default_cost}",
            f"max_worlds={self.max_worlds}",
        ]
        return "ctx-" + hashlib.sha256(
            "\n".join(parts).encode("utf-8")).hexdigest()[:24]

    def assert_matches(self, other: "EvalContext", *, what: str = "") -> None:
        """Refuse two contexts that are not the same world."""
        if self.fingerprint() == other.fingerprint():
            return
        diffs: List[str] = []
        for name in ("candidate_id", "p", "q", "recipient", "scenario",
                     "selection_sims", "selection_seed", "holdout_sims",
                     "holdout_seed", "default_cost", "max_worlds"):
            a, b = getattr(self, name), getattr(other, name)
            if a != b:
                diffs.append(f"{name}: {a!r} != {b!r}")
        if self.pass_price.mode is not other.pass_price.mode:
            diffs.append(f"pass_rule: {self.pass_price.mode.value} != "
                         f"{other.pass_price.mode.value}")
        if self.opportunity_fingerprint() != other.opportunity_fingerprint():
            diffs.append(
                f"opportunity set: {self.opportunity_fingerprint()} != "
                f"{other.opportunity_fingerprint()} "
                f"({len(self.with_candidate)}/{len(self.without_candidate)} vs "
                f"{len(other.with_candidate)}/{len(other.without_candidate)} "
                f"completions)")
        if self.draw.key() != other.draw.key():
            diffs.append(f"allocation draw: {self.draw.key()} != "
                         f"{other.draw.key()}")
        if not diffs:
            diffs.append("a fingerprinted input differs (state, cast, costs, "
                         "market, completion or board settings)")
        raise ContextMismatch(
            (f"{what}: " if what else "")
            + "contexts describe different worlds and may not be compared:\n  - "
            + "\n  - ".join(diffs))

    # --- the opportunity set, per branch ----------------------------------

    def focus_budget_for(self, branch: str) -> int:
        """Our remaining money in that branch, before completing the roster."""
        focus = self.state.focus_owner_id
        base = self.state.owner(focus).budget_remaining
        if branch == "UP":
            return base - self.p
        if branch == "UF":
            return base
        return base                      # W, RF, RP: we pay nothing

    def completions_for(self, branch: str) -> Tuple[Completion, ...]:
        """The offer this branch sees. Filtered by budget, never re-searched.

        Branches that put the candidate on our roster draw from
        ``with_candidate``; the rest from ``without_candidate``. Both are then
        cut to what that branch's focus budget can pay for, which is the same
        filter :mod:`.nested` applies across prices and is what keeps the sets
        nested rather than independently discovered.
        """
        if branch not in BRANCH_HOLDS_CANDIDATE:
            raise ContextMismatch(f"unknown branch {branch!r}")
        pool = (self.with_candidate if BRANCH_HOLDS_CANDIDATE[branch]
                else self.without_candidate)
        budget = self.focus_budget_for(branch)
        focus = self.state.focus_owner_id
        owned = frozenset(self.state.owner(focus).player_ids)
        # In a branch where we hold the candidate he is already ours, so his
        # price is not part of the completion cost.
        extra = frozenset({self.candidate_id}) if \
            BRANCH_HOLDS_CANDIDATE[branch] else frozenset()
        out = [c for c in pool
               if _cost_of(self.costs, c.roster, owned | extra,
                           self.default_cost) <= budget]
        if not out:
            raise ContextMismatch(
                f"branch {branch}: no completion in the shared opportunity set "
                f"is affordable with ${budget}. Re-searching here would be "
                f"exactly the independent search this context exists to "
                f"prevent; widen the ladder or lower the price instead.")
        return tuple(out)

    def possession_offer_is_price_independent(self) -> bool:
        """``UF``'s offer must not change with ``p``. Cheap structural check.

        ``UF`` and ``W`` are the two branches whose CE difference is the
        possession effect, and neither pays anything, so a possession effect
        that moves with ``p`` means the offer was pre-filtered by price.
        """
        focus = self.state.focus_owner_id
        owned = frozenset(self.state.owner(focus).player_ids)
        extra = frozenset({self.candidate_id})
        budget = self.state.owner(focus).budget_remaining
        return all(_cost_of(self.costs, c.roster, owned | extra,
                            self.default_cost) <= budget
                   for c in self.with_candidate)

    def to_dict(self) -> Dict[str, object]:
        return {
            "fingerprint": self.fingerprint(),
            "possession_offer_price_independent":
                self.possession_offer_is_price_independent(),
            "opportunity_fingerprint": self.opportunity_fingerprint(),
            "candidate_price_p": self.p, "pass_price_q": self.q,
            "pass_rule": self.pass_price.mode.value,
            "recipient": self.recipient, "scenario": self.scenario,
            "draw": self.draw.to_dict(),
            "selection": {"sims": self.selection_sims,
                          "seed": self.selection_seed},
            "holdout": {"sims": self.holdout_sims, "seed": self.holdout_seed},
            "with_candidate_completions": len(self.with_candidate),
            "without_candidate_completions": len(self.without_candidate),
            "note": ("the frontier and the five-branch decomposition consume "
                     "this same object; neither may search independently"),
        }


def build_eval_context(
    state: AuctionState, cast: ComparisonCast, costs: CostBook,
    candidate_id: int, *, p: int, pass_price: PassPrice, recipient: str,
    q: int, draw: AllocationDraw, board: BoardSettings,
    completion: CompletionSettings, market: Optional[MarketState],
    key_by_id, proxy, scenario: str = "base",
    with_candidate: Optional[Sequence[Completion]] = None,
    without_candidate: Optional[Sequence[Completion]] = None,
    default_cost: int = 1, max_worlds: int = 2,
) -> EvalContext:
    """Assemble a context, searching ONCE for each opportunity set if needed.

    The search happens here and nowhere else. Callers that already hold the
    accumulated sets (the ladder's nested pre-pass) pass them in; callers that
    do not get one search, whose results then serve every branch.
    """
    from ..auction.completion import complete_roster

    focus = state.focus_owner_id
    if with_candidate is None:
        bought = state.apply_purchase(candidate_id, focus, max(1, p)) \
            if state.purchase_shortfall(candidate_id, focus,
                                        max(1, p)) is None else None
        if bought is None:
            raise ContextMismatch(
                f"cannot build a with-candidate opportunity set: buying "
                f"{candidate_id} at ${max(1, p)} is illegal here")
        res = complete_roster(bought, cast, costs, settings=completion,
                              owner_id=focus, evaluate_ce=False,
                              default_cost=default_cost, proxy=proxy,
                              reserved_ids=frozenset(),
                              notes="shared context: with-candidate set")
        with_candidate = list(res.finalists) or (
            [res.best] if res.best else [])
    if without_candidate is None:
        res = complete_roster(state, cast, costs, settings=completion,
                              owner_id=focus, evaluate_ce=False,
                              default_cost=default_cost, proxy=proxy,
                              reserved_ids=frozenset({candidate_id}),
                              notes="shared context: without-candidate set")
        without_candidate = list(res.finalists) or (
            [res.best] if res.best else [])
    return EvalContext(
        state=state, cast=cast, costs=costs, candidate_id=candidate_id, p=int(p),
        q=int(q), pass_price=pass_price, recipient=recipient, market=market,
        key_by_id=key_by_id, scenario=scenario, completion=completion,
        board=board, draw=draw, selection_sims=completion.selection_sims,
        selection_seed=completion.selection_seed,
        holdout_sims=completion.evaluation_sims,
        holdout_seed=completion.evaluation_seed,
        with_candidate=tuple(with_candidate),
        without_candidate=tuple(without_candidate),
        default_cost=default_cost, max_worlds=max_worlds)
