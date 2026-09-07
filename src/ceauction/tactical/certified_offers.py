"""Certified completions, adapted into the opportunity sets CE already consumes.

The solver in :mod:`ceauction.auction.certified` answers "what is the best legal
completion, and how do you know". The championship-equity layer does not consume
answers, it consumes :class:`~ceauction.auction.completion.Completion` objects
held in an :class:`~ceauction.tactical.evalcontext.EvalContext`. This module is
the join, and it is deliberately narrow: it builds ``Completion`` objects and
nothing else.

**Nothing downstream is rewritten.** ``build_eval_context`` already accepts
``with_candidate`` and ``without_candidate``; supplying both is what makes it
search zero times. The five-branch evaluator, the frontier, the decomposition
and the joint world builder are untouched and unaware that the completions in
front of them came from a MILP rather than from a beam. That is the point: the
reconciliation guarantees in :mod:`.reconciled` hold because every branch reads
one immutable set, and swapping how that set was *discovered* cannot disturb
them.

Two sets, because the branches differ in who ends up holding the candidate, and
they are built at different budgets for the reason ``evalcontext`` documents:

``with_candidate``     serves ``UF`` and ``UP``. Generated at the **widest**
                       budget any branch could have -- our whole remaining
                       budget, because ``UF`` pays nothing for a candidate it
                       already holds -- and then unioned with a certified set at
                       every evaluated price's own post-purchase budget, so each
                       price's own optimum is present rather than inherited.
                       Handing in a price-filtered set would make ``UF`` inherit
                       ``UP``'s affordability cut and the possession effect would
                       then move with a price it does not depend on.
``without_candidate``  serves ``W``, ``RF`` and ``RP``. The candidate is reserved
                       off the board and our budget is untouched, so one set at
                       the full budget covers all three.

Every adapted completion is re-validated from scratch against the auction state
-- ids, prices, size, cost, budget, candidate ownership, lineup feasibility, and
the objective recomputed through ``ProxyEvaluator`` -- and any mismatch is
refused rather than repaired. A completion that reached CE with a cost the cost
book disagrees with would corrupt every conservation check downstream, and those
checks are the reason to trust the frontier at all.
"""

from __future__ import annotations

import dataclasses
import hashlib
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

from ..auction.certified import (CERTIFIED_ABS_TOLERANCE, OBJECTIVE_TOLERANCE,
                                 CompletionProblem, NearOptimalSet,
                                 ProxyObjective, enumerate_near_optimal,
                                 problem_from_state, solve_completion_exact)
from ..auction.completion import Completion, PositionCounts
from ..auction.costs import CostBook
from ..auction.feasibility import can_fill_lineup
from ..auction.proxy import ProxyEvaluator
from ..auction.state import AuctionState
from ..league import Position
from .union import completion_fingerprint

__all__ = [
    "OfferAdaptationError",
    "CertifiedOfferSet",
    "adapt_completion",
    "build_certified_offers",
]


class OfferAdaptationError(RuntimeError):
    """A certified roster did not survive re-validation against the state.

    Its own type because the correct response is to stop, not to drop the
    offending completion and carry on with a quietly smaller opportunity set.
    """


def adapt_completion(
    roster: Sequence[int],
    objective: float,
    *,
    state: AuctionState,
    costs: CostBook,
    proxy: ProxyEvaluator,
    candidate_id: Optional[int],
    holds_candidate: bool,
    budget: int,
    default_cost: int = 1,
    owner_id: Optional[str] = None,
    tolerance: float = OBJECTIVE_TOLERANCE,
) -> Completion:
    """One certified roster to one validated ``Completion``. Refuses mismatches.

    ``state`` is the **pre-purchase** state, the same one the ``EvalContext``
    carries, because that is what ``completions_for`` measures cost against.
    Every field is recomputed here rather than carried over from the solver: the
    point of the adapter is to be a checkpoint, and a checkpoint that trusts its
    input is not one.
    """
    owner_id = owner_id or state.focus_owner_id
    owner = state.owner(owner_id)
    pre_owned = tuple(owner.player_ids)
    ids = [int(p) for p in roster]

    if len(set(ids)) != len(ids):
        raise OfferAdaptationError(f"duplicate player in roster {sorted(ids)}")
    got = set(ids)
    if not set(pre_owned) <= got:
        raise OfferAdaptationError(
            f"roster drops already-owned players: "
            f"{sorted(set(pre_owned) - got)}")
    if holds_candidate:
        if candidate_id is None or candidate_id not in got:
            raise OfferAdaptationError(
                f"a with-candidate completion must contain {candidate_id}")
    elif candidate_id is not None and candidate_id in got:
        raise OfferAdaptationError(
            f"a without-candidate completion must not contain {candidate_id}")

    size = int(state.settings.roster_size)
    if len(ids) != size:
        raise OfferAdaptationError(
            f"roster size {len(ids)} != league roster size {size}")

    exclude = set(pre_owned) | ({candidate_id} if holds_candidate
                                and candidate_id is not None else set())
    added_set = [p for p in ids if p not in exclude]
    for pid in added_set:
        if not state.is_available(pid):
            raise OfferAdaptationError(
                f"completion adds player {pid}, who is not available in this "
                f"state")
    # Board order, so two discoveries of the same roster produce byte-identical
    # Completion objects and therefore one fingerprint.
    added = tuple(sorted(added_set,
                         key=lambda p: (-float(state.spec(p).base_mean), p)))
    added_cost = sum(costs.cost_of(p, default_cost) for p in added)
    if added_cost > budget:
        raise OfferAdaptationError(
            f"completion costs ${added_cost}, above the ${budget} this branch "
            f"has")

    counts = PositionCounts.from_positions(
        Position(int(state.spec(p).position)) for p in ids)
    if counts.total != size:
        raise OfferAdaptationError("position counts do not sum to roster size")
    if not can_fill_lineup(counts):
        raise OfferAdaptationError(
            f"roster cannot field a legal lineup: {counts.to_dict()}")

    recomputed = float(proxy.strength(sorted(ids)))
    if abs(recomputed - float(objective)) > tolerance:
        raise OfferAdaptationError(
            f"objective disagrees with ProxyEvaluator: solver said "
            f"{objective!r}, ProxyEvaluator says {recomputed!r} "
            f"(tolerance {tolerance})")

    ordered = tuple(pre_owned)
    if holds_candidate and candidate_id is not None:
        ordered = ordered + (int(candidate_id),)
    ordered = ordered + added
    if sorted(ordered) != sorted(ids):
        raise OfferAdaptationError("reordered roster is not the same roster")

    return Completion(added=added, roster=ordered, added_cost=int(added_cost),
                      proxy=recomputed, counts=counts)


@dataclass(frozen=True)
class CertifiedOfferSet:
    """Both opportunity sets, plus the certificate behind each."""

    with_candidate: Tuple[Completion, ...]
    without_candidate: Tuple[Completion, ...]
    provenance: Dict[str, object] = field(default_factory=dict)

    def opportunity_fingerprint(self) -> str:
        """The digest ``EvalContext`` will compute, available before building one."""
        h = hashlib.sha256()
        for label, group in (("with", self.with_candidate),
                             ("without", self.without_candidate)):
            h.update(label.encode())
            for key in sorted(",".join(str(x) for x in sorted(c.roster))
                              for c in group):
                h.update(key.encode())
                h.update(b"\n")
        return h.hexdigest()[:16]

    def to_dict(self) -> Dict[str, object]:
        return {
            "with_candidate": len(self.with_candidate),
            "without_candidate": len(self.without_candidate),
            "opportunity_fingerprint": self.opportunity_fingerprint(),
            "with_candidate_proxy_range": [
                round(min((c.proxy for c in self.with_candidate), default=0), 6),
                round(max((c.proxy for c in self.with_candidate), default=0), 6)],
            "with_candidate_cost_range": [
                min((c.added_cost for c in self.with_candidate), default=0),
                max((c.added_cost for c in self.with_candidate), default=0)],
            "provenance": self.provenance,
        }


def build_certified_offers(
    state: AuctionState,
    costs: CostBook,
    proxy: ProxyEvaluator,
    *,
    candidate_id: int,
    prices: Sequence[int],
    pool_depth: int = 40,
    target: int = 12,
    band: float = 0.50,
    tolerance: float = CERTIFIED_ABS_TOLERANCE,
    default_cost: int = 1,
    time_limit_s: Optional[float] = None,
    owner_id: Optional[str] = None,
    objective: Optional[ProxyObjective] = None,
    verbose: bool = False,
) -> CertifiedOfferSet:
    """Certified near-optimal sets for both halves of the five-branch lattice."""
    owner_id = owner_id or state.focus_owner_id
    owner = state.owner(owner_id)
    B = int(owner.budget_remaining)
    obj = objective if objective is not None else ProxyObjective(proxy)

    if state.purchase_shortfall(candidate_id, owner_id, 1) is not None:
        raise OfferAdaptationError(
            f"cannot hold {candidate_id} in this state at any price")
    post = state.apply_purchase(candidate_id, owner_id, 1)
    base = problem_from_state(post, costs, owner_id=owner_id,
                              candidate_pool=pool_depth,
                              default_cost=default_cost)

    with_by_key: Dict[str, Completion] = {}
    rows: List[Dict[str, object]] = []

    # The widest budget first: UF holds the candidate and pays nothing for him.
    budgets: List[Tuple[str, int]] = [("UF_widest", B)]
    budgets += [(f"p={p}", B - int(p)) for p in prices]
    for label, bud in budgets:
        if bud < 0:
            continue
        problem = dataclasses.replace(base, budget=int(bud))
        nos = enumerate_near_optimal(problem, proxy, target=target, band=band,
                                     tolerance=tolerance,
                                     time_limit_s=time_limit_s, objective=obj)
        n_new = 0
        for roster, value, gap in nos.entries:
            c = adapt_completion(
                roster, value, state=state, costs=costs, proxy=proxy,
                candidate_id=candidate_id, holds_candidate=True, budget=bud,
                default_cost=default_cost, owner_id=owner_id)
            k = completion_fingerprint(c)
            if k not in with_by_key:
                with_by_key[k] = c
                n_new += 1
        rows.append({"set": "with_candidate", "label": label, "budget": bud,
                     "emitted": len(nos.entries), "new": n_new,
                     "optimum": round(nos.optimum, 6),
                     "exhausted": nos.exhausted,
                     "max_gap": round(max((g for _, _, g in nos.entries),
                                          default=0.0), 6)})
        if verbose:
            print(f"    with_candidate {label:>10} ${bud:>4}  "
                  f"emitted={len(nos.entries)} new={n_new} "
                  f"opt={nos.optimum:.4f}", flush=True)

    without_problem = problem_from_state(
        state, costs, owner_id=owner_id, candidate_pool=pool_depth,
        default_cost=default_cost, reserved_ids=frozenset({candidate_id}))
    nos_w = enumerate_near_optimal(without_problem, proxy, target=target,
                                   band=band, tolerance=tolerance,
                                   time_limit_s=time_limit_s, objective=obj)
    without_by_key: Dict[str, Completion] = {}
    for roster, value, gap in nos_w.entries:
        c = adapt_completion(
            roster, value, state=state, costs=costs, proxy=proxy,
            candidate_id=candidate_id, holds_candidate=False, budget=B,
            default_cost=default_cost, owner_id=owner_id)
        without_by_key.setdefault(completion_fingerprint(c), c)
    rows.append({"set": "without_candidate", "label": "full_budget",
                 "budget": B, "emitted": len(nos_w.entries),
                 "new": len(without_by_key),
                 "optimum": round(nos_w.optimum, 6),
                 "exhausted": nos_w.exhausted,
                 "max_gap": round(max((g for _, _, g in nos_w.entries),
                                      default=0.0), 6)})
    if verbose:
        print(f"    without_candidate      ${B:>4}  "
              f"emitted={len(nos_w.entries)} opt={nos_w.optimum:.4f}",
              flush=True)

    if not with_by_key or not without_by_key:
        raise OfferAdaptationError(
            "a certified opportunity set came back empty; CE may not be run "
            "against a set the solver could not populate")

    def order(d: Dict[str, Completion]) -> Tuple[Completion, ...]:
        return tuple(sorted(d.values(),
                            key=lambda c: (-c.proxy, completion_fingerprint(c))))

    return CertifiedOfferSet(
        with_candidate=order(with_by_key),
        without_candidate=order(without_by_key),
        provenance={"pool_depth": pool_depth, "target": target, "band": band,
                    "tolerance": tolerance, "full_budget": B,
                    "prices": list(map(int, prices)), "rows": rows})
