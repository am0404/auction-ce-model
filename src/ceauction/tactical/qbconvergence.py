"""A convergence stopping rule for the QB's full-budget completion union.

`FULL_BUDGET_UNION.md` withheld the QB's `PASS_AT_NEXT_BID` reading because
tripling the search effort grew the union from 26 to 72 constructions and moved
`UF`'s selected roster by `+0.690` proxy points, above the `0.25` tolerance.
This module decides which of three things is true: the union converges at
practical effort, the search needs an algorithmic change, or the maximum effort
is reached without convergence (:data:`QB_UNION_UNDERCONVERGED`).

**The diagnosis that shaped this design.** A knob sweep on the real QB found:

- ``finalists`` sets how many completions come out and **never** moves the
  objective (3, 12, 40 finalists all peak at 95.2731). It controlled union
  *size*, which is why raw size looked like progress when it was not.
- ``beam_width`` is the only primary knob that moves the objective, and it moves
  it **non-monotonically**: 32→95.27, 64→94.17, 96→97.99, 128→97.45, 192→97.56,
  256→97.56, 384→95.15, 512→95.89, 768→97.32. A spread of ~3.8 points, fifteen
  times the tolerance, with more effort frequently doing *worse*.
- ``spend_buckets`` also moves it (4→96.47, 8→97.56, 16→98.14, 32→97.85), also
  non-monotonically.
- ``candidate_pool``, ``proxy_candidates`` and ``proxy_seed`` move it **not at
  all**. The search is deterministic given (beam, buckets); the proxy seed is
  not a source of independent generation paths, so using it as one would fake
  diversity that does not exist.
- Exact enumeration is infeasible and correctly refuses: 1.5e11 combinations at
  the smallest pool depth, against a 400,000 limit.

So "run it again with triple the effort" cannot converge this search: the thing
being increased does not monotonically improve the answer. The fix is to stop
treating one setting's output as the answer. Every configuration's completions
accumulate into **one monotone union**, the whole union is rescored by one
evaluator, and the reported objective is the **best over everything seen so
far** -- which is non-decreasing by construction. Convergence then means the
best-so-far stopped moving, which is a statement about the accumulated
opportunity set rather than about one lucky beam.

That is a real claim but a bounded one, and the wording matters: it says more
search effort no longer finds a better construction. It does **not** say no
better roster exists. See :attr:`ConvergenceReport.scope`.
"""

from __future__ import annotations

import dataclasses
import hashlib
import statistics
import time
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

from ..auction.completion import (Completion, CompletionSettings,
                                  complete_roster)
from ..auction.costs import CostBook
from ..auction.proxy import ProxyEvaluator
from ..auction.state import AuctionState
from .union import SUPPORT_PRICES, completion_fingerprint

__all__ = [
    "EffortRung", "EFFORT_LADDER", "LADDER_DECLARED_AT",
    "PROXY_TOLERANCE", "EconomicProfile", "economic_profile",
    "economically_equivalent", "SeedMisuse", "assert_independent_seeds",
    "RungResult", "ConvergenceReport", "run_convergence_ladder",
    "evaluate_stopping_rule",
    "QB_UNION_CONVERGED", "QB_UNION_UNDERCONVERGED",
]

QB_UNION_CONVERGED = "QB_UNION_CONVERGED"
QB_UNION_UNDERCONVERGED = "QB_UNION_UNDERCONVERGED"

#: Weekly starting-projection points. Unchanged from the tolerance the previous
#: run failed against; loosening it to obtain convergence would be cheating.
PROXY_TOLERANCE = 0.25


@dataclass(frozen=True)
class EffortRung:
    """One rung of the predeclared ladder.

    ``paths`` are independent deterministic generation paths -- distinct
    ``(beam_width, spend_buckets)`` pairs. Those are the two settings the sweep
    showed actually change what the search finds; ``proxy_seed`` does not, so it
    is deliberately not used here.
    """

    name: str
    multiple: int
    paths: Tuple[Tuple[int, int], ...]
    finalists: int
    max_candidates: int

    @property
    def n_paths(self) -> int:
        return len(self.paths)

    def settings(self, base: CompletionSettings) -> List[CompletionSettings]:
        return [dataclasses.replace(base, beam_width=b, spend_buckets=sb,
                                    finalists=self.finalists,
                                    max_candidates=self.max_candidates)
                for b, sb in self.paths]

    def to_dict(self) -> Dict[str, object]:
        return {"name": self.name, "effort_multiple": self.multiple,
                "generation_paths": [list(p) for p in self.paths],
                "n_paths": self.n_paths, "finalists": self.finalists,
                "max_candidates": self.max_candidates}


#: **Predeclared.** Fixed before any result of this experiment was seen; the
#: hash below is asserted by the test suite so a later edit cannot quietly
#: retrofit the ladder to whatever happened to converge.
EFFORT_LADDER: Tuple[EffortRung, ...] = (
    EffortRung("1x", 1, ((32, 8), (32, 16)), finalists=8, max_candidates=400),
    EffortRung("2x", 2, ((64, 8), (64, 16), (96, 8)), finalists=16,
               max_candidates=800),
    EffortRung("4x", 4, ((128, 8), (128, 16), (192, 8), (192, 16)),
               finalists=24, max_candidates=1600),
    EffortRung("8x", 8, ((256, 8), (256, 16), (384, 16), (512, 16), (768, 32)),
               finalists=32, max_candidates=3200),
)

LADDER_DECLARED_AT = "2026-09-07, before any rung of this experiment was run"


def ladder_fingerprint(ladder: Sequence[EffortRung] = EFFORT_LADDER) -> str:
    h = hashlib.sha256()
    for r in ladder:
        h.update(f"{r.name}|{r.multiple}|{r.paths}|{r.finalists}|"
                 f"{r.max_candidates}".encode())
    return h.hexdigest()[:16]


class SeedMisuse(RuntimeError):
    """Selection and holdout seeds are equal or overlapping."""


def assert_independent_seeds(selection_seed: int, holdout_seed: int) -> None:
    """The holdout must be independent of the sample that chose the winner.

    Sharing a seed would let the roster that won on a kind sample be graded by
    that same sample, which is how a selection artifact becomes a reported
    edge.
    """
    if selection_seed == holdout_seed:
        raise SeedMisuse(
            f"selection and holdout seeds are both {selection_seed}; the "
            f"holdout would re-use the sample that chose the winner")
    if abs(selection_seed - holdout_seed) < 1000:
        raise SeedMisuse(
            f"selection seed {selection_seed} and holdout seed {holdout_seed} "
            f"are within 1000 of each other; adjacent seeds can produce "
            f"overlapping streams and are refused as not independent")


@dataclass(frozen=True)
class EconomicProfile:
    """What must match for two completions to be *economically equivalent*.

    Two rosters need not be the same fifteen players to be the same decision.
    They are the same decision when every quantity the decision depends on is
    the same: what it costs, how strong a legal lineup it fields, that it is
    feasible at all, that it settles the candidate the same way, and what money
    and slots it leaves behind. Anything looser -- matching on cost alone, say
    -- would let a materially worse roster pass as "equivalent".
    """

    total_cost: int
    strength_bucket: int
    feasible: bool
    holds_candidate: bool
    budget_left: int
    slots_left: int

    def to_dict(self) -> Dict[str, object]:
        return dataclasses.asdict(self)


def economic_profile(c: Completion, *, costs: CostBook, budget: int,
                     owned: FrozenSet[int], candidate_id: int,
                     roster_size: int, default_cost: int = 1,
                     tolerance: float = PROXY_TOLERANCE) -> EconomicProfile:
    spend = sum(costs.cost_of(pid, default_cost) for pid in c.roster
                if pid not in owned)
    return EconomicProfile(
        total_cost=int(c.added_cost),
        strength_bucket=int(round(c.proxy / tolerance)),
        feasible=(spend <= budget and len(set(c.roster)) == roster_size),
        holds_candidate=candidate_id in c.roster,
        budget_left=int(budget - spend),
        slots_left=int(roster_size - len(set(c.roster))))


def economically_equivalent(a: Optional[Completion], b: Optional[Completion],
                            **kw) -> bool:
    """Identical fingerprint, or identical on every economic quantity."""
    if a is None or b is None:
        return a is b
    if completion_fingerprint(a) == completion_fingerprint(b):
        return True
    return economic_profile(a, **kw) == economic_profile(b, **kw)


def _affordable(union: Sequence[Completion], *, costs: CostBook, budget: int,
                owned: FrozenSet[int], price: int, default_cost: int = 1
                ) -> List[Completion]:
    room = budget - price
    out = []
    for c in union:
        spend = sum(costs.cost_of(pid, default_cost) for pid in c.roster
                    if pid not in owned)
        if spend <= room:
            out.append(c)
    return out


def _best(cs: Sequence[Completion]) -> Optional[Completion]:
    if not cs:
        return None
    return max(cs, key=lambda c: (c.proxy, completion_fingerprint(c)))


@dataclass
class RungResult:
    """One rung: what it added, what it selected, and what moved."""

    rung: EffortRung
    union_size: int
    newly_unique: int
    best_proxy: float
    uf_fingerprint: str
    uf_completion: Optional[Completion]
    up_fingerprints: Dict[int, str]
    up_best_proxy: Dict[int, float]
    affordable_counts: Dict[int, int]
    per_path_best: Dict[str, float]
    per_path_new: Dict[str, int]
    runtime_s: float

    @property
    def seed_disagreement(self) -> float:
        """Spread of the best objective across this rung's generation paths.

        Large means the paths are exploring genuinely different regions and no
        single one of them can be trusted as the answer.
        """
        vals = list(self.per_path_best.values())
        return (max(vals) - min(vals)) if len(vals) > 1 else 0.0

    def to_dict(self) -> Dict[str, object]:
        return {
            **self.rung.to_dict(),
            "union_size": self.union_size, "newly_unique": self.newly_unique,
            "best_proxy": round(self.best_proxy, 4),
            "uf_fingerprint": self.uf_fingerprint,
            "up_fingerprints": {str(k): v
                                for k, v in self.up_fingerprints.items()},
            "up_best_proxy": {str(k): round(v, 4)
                              for k, v in self.up_best_proxy.items()},
            "affordable_counts": {str(k): v
                                  for k, v in self.affordable_counts.items()},
            "per_path_best": {k: round(v, 4)
                              for k, v in self.per_path_best.items()},
            "per_path_new": dict(self.per_path_new),
            "generation_path_disagreement": round(self.seed_disagreement, 4),
            "runtime_s": round(self.runtime_s, 1),
        }


@dataclass
class ConvergenceReport:
    """The ladder, the stopping-rule verdict, and the reasons behind it."""

    rungs: Tuple[RungResult, ...]
    checks: Dict[str, bool]
    failures: Tuple[str, ...]
    union: Tuple[Completion, ...]
    tolerance: float = PROXY_TOLERANCE

    @property
    def status(self) -> str:
        if len(self.rungs) < 2:
            return QB_UNION_UNDERCONVERGED
        return (QB_UNION_CONVERGED if not self.failures
                else QB_UNION_UNDERCONVERGED)

    @property
    def converged(self) -> bool:
        return self.status == QB_UNION_CONVERGED

    @property
    def may_evaluate_ce(self) -> bool:
        """Full CE spend is gated on search convergence, never the reverse."""
        return self.converged

    @property
    def may_quote_bid(self) -> bool:
        return self.converged

    @property
    def scope(self) -> str:
        return ("convergence is over the accumulated union under the "
                "predeclared effort ladder: more search effort no longer finds "
                "a better construction. It is NOT a claim that no better "
                "roster exists in the full combinatorial space, which exact "
                "enumeration refuses to visit (1.5e11 combinations).")

    def to_dict(self) -> Dict[str, object]:
        return {
            "status": self.status,
            "ladder_declared_at": LADDER_DECLARED_AT,
            "ladder_fingerprint": ladder_fingerprint(),
            "tolerance_weekly_points": self.tolerance,
            "n_rungs_completed": len(self.rungs),
            "rungs": [r.to_dict() for r in self.rungs],
            "stopping_rule_checks": dict(self.checks),
            "failures": list(self.failures),
            "may_evaluate_ce": self.may_evaluate_ce,
            "may_quote_bid": self.may_quote_bid,
            "convergence_scope": self.scope,
        }



def evaluate_stopping_rule(results: Sequence[RungResult], *,
                           evaluated_prices: Sequence[int],
                           eq_kw: Dict[str, object],
                           tolerance: float = PROXY_TOLERANCE
                           ) -> Tuple[Dict[str, bool], List[str]]:
    """Apply the stopping rule to the final two completed rungs.

    Separated from the search so the rule can be exercised against constructed
    rungs. Every clause must pass; a single failure means underconverged.
    """
    checks: Dict[str, bool] = {}
    failures: List[str] = []
    if len(results) < 2:
        failures.append("fewer than two completed rungs; convergence cannot be "
                        "declared from a single effort level")
        return checks, failures

    prev, last = results[-2], results[-1]

    ok = economically_equivalent(prev.uf_completion, last.uf_completion,
                                 **eq_kw)
    checks["uf_selection_stable"] = ok
    if not ok:
        failures.append(
            f"UF's selected completion changed between {prev.rung.name} and "
            f"{last.rung.name} and the two are not economically equivalent")

    ok = abs(last.best_proxy - prev.best_proxy) <= tolerance
    checks["uf_objective_within_tolerance"] = ok
    if not ok:
        failures.append(
            f"UF objective moved {last.best_proxy - prev.best_proxy:+.4f}, "
            f"beyond the {tolerance} tolerance")

    bad = [ep for ep in evaluated_prices
           if last.up_best_proxy[ep] - prev.up_best_proxy[ep] > tolerance]
    checks["no_evaluated_price_improved_beyond_tolerance"] = not bad
    if bad:
        failures.append(
            f"the objective at evaluated price(s) {bad} improved by more than "
            f"{tolerance}; a better buy-side roster was still being found")

    bad = [ep for ep in evaluated_prices
           if prev.up_fingerprints[ep] != last.up_fingerprints[ep]]
    checks["up_selection_stable"] = not bad
    if bad:
        failures.append(
            f"the selected UP completion changed at evaluated price(s) {bad}")

    bad = [ep for ep in evaluated_prices
           if (prev.affordable_counts[ep] == 0)
           != (last.affordable_counts[ep] == 0)]
    checks["affordability_unchanged_at_tested_prices"] = not bad
    if bad:
        failures.append(
            f"added completions changed feasibility at evaluated price(s) "
            f"{bad}")

    ok = last.seed_disagreement <= tolerance
    checks["generation_paths_agree"] = ok
    if not ok:
        failures.append(
            f"the final rung's generation paths disagree by "
            f"{last.seed_disagreement:.4f}, beyond {tolerance}; independent "
            f"paths are still finding materially different constructions")

    return checks, failures


def run_convergence_ladder(
    state: AuctionState, cast, costs: CostBook, candidate_id: int, *,
    base: CompletionSettings, proxy: ProxyEvaluator,
    evaluated_prices: Sequence[int],
    support_prices: Sequence[int] = SUPPORT_PRICES,
    ladder: Sequence[EffortRung] = EFFORT_LADDER,
    legal_max: Optional[int] = None, default_cost: int = 1,
    tolerance: float = PROXY_TOLERANCE, progress=None,
) -> ConvergenceReport:
    """Walk the predeclared ladder, accumulating one monotone union.

    Every rung's every path searches at every support price, and everything
    found is kept. The objective reported is the best over the whole
    accumulated union, so it cannot fall -- which is what makes "it stopped
    moving" mean something.
    """
    focus = state.focus_owner_id
    budget = state.owner(focus).budget_remaining
    owned_after = frozenset(state.owner(focus).player_ids) | {candidate_id}
    roster_size = state.settings.roster_size

    prices = sorted({int(p) for p in support_prices
                     if state.purchase_shortfall(candidate_id, focus,
                                                 int(p)) is None})
    if legal_max is not None and \
            state.purchase_shortfall(candidate_id, focus, legal_max) is None:
        prices = sorted(set(prices) | {int(legal_max)})

    eq_kw = dict(costs=costs, budget=budget, owned=owned_after,
                 candidate_id=candidate_id, roster_size=roster_size,
                 default_cost=default_cost, tolerance=tolerance)

    seen: Dict[str, Completion] = {}
    results: List[RungResult] = []
    for rung in ladder:
        t0 = time.perf_counter()
        before = len(seen)
        per_path_best: Dict[str, float] = {}
        per_path_new: Dict[str, int] = {}
        for cs, (beam, buckets) in zip(rung.settings(base), rung.paths):
            label = f"beam{beam}_buckets{buckets}"
            path_before = len(seen)
            path_best = float("-inf")
            for p in prices:
                bought = state.apply_purchase(candidate_id, focus, p)
                res = complete_roster(
                    bought, cast, costs, settings=cs, owner_id=focus,
                    evaluate_ce=False, default_cost=default_cost, proxy=proxy,
                    reserved_ids=frozenset(),
                    notes=f"{rung.name} {label} support ${p}")
                found = list(res.finalists) or ([res.best] if res.best else [])
                for c in found:
                    seen.setdefault(completion_fingerprint(c), c)
                    path_best = max(path_best, c.proxy)
            per_path_best[label] = path_best
            per_path_new[label] = len(seen) - path_before

        union = list(seen.values())
        uf_aff = _affordable(union, costs=costs, budget=budget,
                             owned=owned_after, price=0,
                             default_cost=default_cost)
        uf = _best(uf_aff)
        up_fps: Dict[int, str] = {}
        up_best: Dict[int, float] = {}
        aff_counts: Dict[int, int] = {}
        for ep in evaluated_prices:
            aff = _affordable(union, costs=costs, budget=budget,
                              owned=owned_after, price=ep,
                              default_cost=default_cost)
            b = _best(aff)
            aff_counts[ep] = len(aff)
            up_fps[ep] = completion_fingerprint(b) if b else "none"
            up_best[ep] = b.proxy if b else float("-inf")

        rr = RungResult(
            rung=rung, union_size=len(seen), newly_unique=len(seen) - before,
            best_proxy=(uf.proxy if uf else float("-inf")),
            uf_fingerprint=completion_fingerprint(uf) if uf else "none",
            uf_completion=uf, up_fingerprints=up_fps, up_best_proxy=up_best,
            affordable_counts=aff_counts, per_path_best=per_path_best,
            per_path_new=per_path_new, runtime_s=time.perf_counter() - t0)
        results.append(rr)
        if progress is not None:
            progress(rr)

    checks, failures = evaluate_stopping_rule(
        results, evaluated_prices=evaluated_prices, eq_kw=eq_kw,
        tolerance=tolerance)
    return ConvergenceReport(tuple(results), checks, tuple(failures),
                             tuple(seen.values()), tolerance)
