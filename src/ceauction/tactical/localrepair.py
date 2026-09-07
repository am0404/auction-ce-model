"""Repair completions by deterministic local search instead of a wider beam.

`QB_UNION_CONVERGENCE.md` found the completion search's objective is
**non-monotone in beam width** -- 32:95.27, 64:94.17, 96:97.99, 384:95.15 -- so
widening the beam is not a convergence strategy. This module implements the
repair the objective's actual shape calls for.

**The objective.** ``ProxyEvaluator.strength`` is

    f(R) = mean over (rep r, scoring week w) of
           sum of proj[r,w,p] for p in Greedy(R, r, w)

where ``Greedy`` is :func:`ceauction.lineup_vec.select_lineups_mask`, an exact
matroid greedy over the league's eligibility graph restricted to the players
available that week. Per scenario this is the max-weight basis of a matroid, so
each scenario's value is **submodular** in the roster set, and an average of
submodular functions is submodular. Therefore ``f`` is:

- **not additive by player** -- a player's marginal value depends on who else is
  rostered and on which availability draw occurs;
- **not best-lineup-plus-additive-bench** -- bench value exists only through
  conditional substitution when a starter is out;
- **nonlinear through availability/conditional effects** -- this is the box it
  belongs in;
- **not dependent on other completed rosters** -- ``f`` reads only ``R``, so
  there is no coupling to rivals at the proxy stage.

That diagnosis explains the beam's behaviour rather than merely describing it.
A beam ranks partial rosters by a prefix value, but under a submodular objective
a prefix's value is not a valid bound on what the finished roster is worth --
diminishing returns mean a strong prefix can finish weakly and vice versa. So
carrying more prefixes does not monotonically improve the result, which is
exactly what the sweep measured.

Local search is the principled repair for submodular maximisation under
matroid/knapsack constraints: one-swap local optima carry standard
approximation guarantees, and, more to the point here, **improvement is monotone
by construction** -- a swap is accepted only if it raises ``f``. A bad beam draw
gets repaired rather than discarded.

**Nothing is replaced.** A repaired completion is *added* to the accumulated
union alongside the original, so the union stays monotone: no search, however
much better, ever removes something an earlier one found.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

from ..auction.completion import Completion
from ..auction.costs import CostBook
from ..auction.feasibility import can_fill_lineup
from ..auction.proxy import ProxyEvaluator
from ..auction.state import AuctionState
from ..league import Position
from .union import completion_fingerprint

__all__ = [
    "SwapCandidate", "RepairResult", "RepairReport",
    "repair_completion", "repair_union", "IMPROVEMENT_EPSILON",
]

#: A swap must beat the incumbent by more than this to be accepted. Guards
#: against an infinite loop on floating-point ties between equivalent rosters.
IMPROVEMENT_EPSILON = 1e-9


def _counts(roster: Sequence[int], state: AuctionState):
    from ..auction.completion import PositionCounts
    qb = rb = wr = te = 0
    for pid in roster:
        p = Position(int(state.spec(pid).position))
        if p is Position.QB:
            qb += 1
        elif p is Position.RB:
            rb += 1
        elif p is Position.WR:
            wr += 1
        else:
            te += 1
    return PositionCounts(qb=qb, rb=rb, wr=wr, te=te)


@dataclass(frozen=True)
class SwapCandidate:
    """One accepted one-for-one exchange."""

    out_id: int
    in_id: int
    gain: float
    new_cost: int

    def to_dict(self) -> Dict[str, object]:
        return {"gain": round(self.gain, 6), "new_cost": self.new_cost}


@dataclass
class RepairResult:
    """What local search did to one completion."""

    original: Completion
    repaired: Completion
    swaps: Tuple[SwapCandidate, ...]

    @property
    def improved(self) -> bool:
        return bool(self.swaps)

    @property
    def gain(self) -> float:
        return self.repaired.proxy - self.original.proxy

    @property
    def n_swaps(self) -> int:
        return len(self.swaps)


def repair_completion(
    completion: Completion, *, state: AuctionState, costs: CostBook,
    proxy: ProxyEvaluator, candidate_id: Optional[int],
    owned: FrozenSet[int], budget: int, pool: Sequence[int],
    default_cost: int = 1, max_swaps: int = 64,
) -> RepairResult:
    """Improve one completion by exhaustive legal one-player swaps.

    Removes one rostered non-candidate player and adds one available player,
    keeping the roster size, the candidate, the budget, the ``$1``-per-open-slot
    reserve, lineup feasibility and distinct ownership. Repeats until no single
    swap improves, i.e. a one-swap local optimum.

    Every improving swap is evaluated and the **best** is taken, not the first
    found, so the walk does not depend on iteration order. Ties are broken on a
    stable roster fingerprint rather than on position in any list -- an owner
    tuple index or an incidental ordering would make the result depend on how
    the pool happened to be assembled.
    """
    roster = list(completion.roster)
    cur_proxy = completion.proxy
    cur_cost = completion.added_cost
    swaps: List[SwapCandidate] = []
    protected = set(owned)
    if candidate_id is not None:
        protected.add(candidate_id)

    def spend(rs: Sequence[int]) -> int:
        return sum(costs.cost_of(p, default_cost) for p in rs
                   if p not in owned)

    for _ in range(max_swaps):
        cur_set = set(roster)
        trials: List[List[int]] = []
        meta: List[Tuple[int, int, int]] = []
        for out_id in roster:
            # The candidate, and anything already owned, cannot be sold.
            if out_id in protected:
                continue
            base = [p for p in roster if p != out_id]
            base_spend = spend(base)
            for in_id in pool:
                if in_id in cur_set:
                    continue
                new_spend = base_spend + costs.cost_of(in_id, default_cost)
                if new_spend > budget:
                    continue
                trial = base + [in_id]
                if len(set(trial)) != len(completion.roster):
                    continue
                if not can_fill_lineup(_counts(trial, state)):
                    continue
                trials.append(trial)
                meta.append((out_id, in_id, new_spend))
        if not trials:
            break
        # One batched pass rather than a strength() call per trial: the swap
        # neighbourhood is thousands of rosters and they are all the same size.
        vals = proxy.strength_many(trials)
        best = None
        for i, val in enumerate(vals):
            v = float(val)
            if v <= cur_proxy + IMPROVEMENT_EPSILON:
                continue
            # Stable tie-break on the roster fingerprint, never on the order
            # the pool or the roster tuple happened to be assembled in.
            fp = ",".join(str(x) for x in sorted(trials[i]))
            if best is None or (v, fp) > (best[0], best[1]):
                best = (v, fp, i)
        if best is None:
            break
        val, _fp, i = best
        out_id, in_id, new_spend = meta[i]
        swaps.append(SwapCandidate(out_id=out_id, in_id=in_id,
                                   gain=val - cur_proxy, new_cost=new_spend))
        roster = sorted(trials[i])
        cur_proxy = val
        cur_cost = new_spend

    if not swaps:
        return RepairResult(completion, completion, ())
    added = tuple(sorted(p for p in roster if p not in owned))
    repaired = Completion(added=added, roster=tuple(roster),
                          added_cost=int(cur_cost), proxy=float(cur_proxy))
    return RepairResult(completion, repaired, tuple(swaps))


@dataclass
class RepairReport:
    """Aggregate evidence that the repair pass did something and did it legally."""

    n_input: int
    n_improved: int
    n_already_optimal: int
    gains: Tuple[float, ...]
    swap_counts: Tuple[int, ...]
    union_size_before: int
    union_size_after: int
    runtime_s: float

    @property
    def mean_gain(self) -> float:
        g = [x for x in self.gains if x > 0]
        return sum(g) / len(g) if g else 0.0

    @property
    def max_gain(self) -> float:
        return max(self.gains, default=0.0)

    @property
    def mean_swaps(self) -> float:
        s = [x for x in self.swap_counts if x > 0]
        return sum(s) / len(s) if s else 0.0

    def to_dict(self) -> Dict[str, object]:
        return {
            "completions_examined": self.n_input,
            "completions_improved": self.n_improved,
            "already_locally_optimal": self.n_already_optimal,
            "mean_improvement": round(self.mean_gain, 4),
            "max_improvement": round(self.max_gain, 4),
            "mean_swaps_per_repaired": round(self.mean_swaps, 2),
            "max_swaps": max(self.swap_counts, default=0),
            "union_size_before": self.union_size_before,
            "union_size_after": self.union_size_after,
            "union_growth": self.union_size_after - self.union_size_before,
            "runtime_s": round(self.runtime_s, 1),
            "note": ("repaired completions are ADDED to the union, never "
                     "substituted for their originals, so the union stays "
                     "monotone"),
        }


def repair_union(
    union: Dict[str, Completion], *, state: AuctionState, costs: CostBook,
    proxy: ProxyEvaluator, candidate_id: Optional[int],
    owned: FrozenSet[int], budget: int, pool: Sequence[int],
    default_cost: int = 1, limit: Optional[int] = None,
) -> Tuple[Dict[str, Completion], RepairReport]:
    """Repair the strongest completions and fold the results back in.

    ``limit`` caps how many are repaired, strongest first -- the weak tail of a
    1,700-member union cannot become the selected roster, so spending the pass
    on it buys nothing. The cap is a declared budget, not hidden pruning: every
    original stays in the union either way.
    """
    t0 = time.perf_counter()
    ordered = sorted(union.values(), key=lambda c: (-c.proxy,
                                                    completion_fingerprint(c)))
    targets = ordered if limit is None else ordered[:limit]
    before = len(union)
    out = dict(union)
    gains: List[float] = []
    swap_counts: List[int] = []
    improved = 0
    for c in targets:
        r = repair_completion(
            c, state=state, costs=costs, proxy=proxy,
            candidate_id=candidate_id, owned=owned, budget=budget, pool=pool,
            default_cost=default_cost)
        gains.append(r.gain)
        swap_counts.append(r.n_swaps)
        if r.improved:
            improved += 1
            out.setdefault(completion_fingerprint(r.repaired), r.repaired)
    return out, RepairReport(
        n_input=len(targets), n_improved=improved,
        n_already_optimal=len(targets) - improved, gains=tuple(gains),
        swap_counts=tuple(swap_counts), union_size_before=before,
        union_size_after=len(out), runtime_s=time.perf_counter() - t0)
