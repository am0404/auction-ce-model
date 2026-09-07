"""Run the certified completion solver on the real QB blocker.

``QB_UNION_UNDERCONVERGED`` was never a statement that the roster search had
found a bad answer. It was a statement that nobody could tell: the beam moved
3.8 weekly points with its width, the one-swap and two-swap passes each found
further improvements, and no bound existed against which to call any of them
finished.

This module answers the question the ladder could not, on the same inputs: the
same real QB candidate, the same **SIMULATED** balanced mid-auction state, the
same evaluated prices, the same declared candidate-pool depth. For each price
it reports the best roster every previous method found, the certified optimum,
the proven upper bound, and the gap between them.

Player-level output stays under ``local_data/``; the committed artifact carries
objectives, gaps, runtimes and fingerprints only.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from ..auction.certified import (CERTIFIED_ABS_TOLERANCE, CertifiedResult,
                                 CompletionProblem, NearOptimalSet,
                                 ProxyObjective, enumerate_near_optimal,
                                 problem_from_state, roster_fingerprint,
                                 solve_completion_exact, solver_available,
                                 solver_version)
from ..auction.completion import (Completion, CompletionSettings,
                                  complete_roster)
from ..auction.proxy import ProxyEvaluator
from .localrepair import (TWO_SWAP_COMPLETION_LIMIT, TWO_SWAP_SHORTLIST,
                          repair_union, repair_union_two_swap)
from .midauction import SIMULATED_WATERMARK
from .union import completion_fingerprint

__all__ = [
    "CERTIFIED_SOLVER_NO_GO", "SOLVER_GAP_TOO_WIDE",
    "CE_CANDIDATE_SET_UNDERCONVERGED", "RUNTIME_NO_GO",
    "CANDIDATE_POOL_DEPTH", "PriceCertification", "certify_price",
]

CERTIFIED_SOLVER_NO_GO = "CERTIFIED_SOLVER_NO_GO"
SOLVER_GAP_TOO_WIDE = "SOLVER_GAP_TOO_WIDE"
CE_CANDIDATE_SET_UNDERCONVERGED = "CE_CANDIDATE_SET_UNDERCONVERGED"
RUNTIME_NO_GO = "RUNTIME_NO_GO"

#: Board depth the certified solve uses, matched to the control run's
#: ``candidate_pool``. A deeper pool would find a roster the beam could not
#: have seen, and the comparison would stop being like-for-like. Reported with
#: every result, because it is a bound on the answer.
CANDIDATE_POOL_DEPTH = 40


@dataclass
class PriceCertification:
    """Everything known about one evaluated price."""

    price: int
    label: str
    open_slots: int
    budget: int
    pool_depth: int
    board_size: int

    beam: Optional[float] = None
    union_best: Optional[float] = None
    one_swap: Optional[float] = None
    two_swap: Optional[float] = None

    certified: Optional[float] = None
    upper_bound: Optional[float] = None
    abs_gap: Optional[float] = None
    rel_gap: Optional[float] = None
    iterations: int = 0
    solver_runtime_s: float = 0.0
    heuristic_runtime_s: float = 0.0
    roster_fp: str = ""
    status: str = ""
    near_optimal: Optional[Dict[str, object]] = None
    defect: Optional[str] = None

    @property
    def best_heuristic(self) -> Optional[float]:
        vals = [v for v in (self.beam, self.union_best, self.one_swap,
                            self.two_swap) if v is not None]
        return max(vals) if vals else None

    @property
    def certified_gain(self) -> Optional[float]:
        bh = self.best_heuristic
        if bh is None or self.certified is None:
            return None
        return self.certified - bh

    def to_dict(self) -> Dict[str, object]:
        def r(x):
            return None if x is None else round(float(x), 6)
        return {
            "price": self.price, "label": self.label,
            "open_slots": self.open_slots, "budget": self.budget,
            "pool_depth": self.pool_depth, "board_size": self.board_size,
            "beam": r(self.beam), "union_best": r(self.union_best),
            "one_swap": r(self.one_swap), "two_swap": r(self.two_swap),
            "best_heuristic": r(self.best_heuristic),
            "certified": r(self.certified),
            "upper_bound": r(self.upper_bound),
            "abs_gap": r(self.abs_gap), "rel_gap": r(self.rel_gap),
            "certified_gain_over_heuristic": r(self.certified_gain),
            "iterations": self.iterations,
            "solver_runtime_s": round(self.solver_runtime_s, 3),
            "heuristic_runtime_s": round(self.heuristic_runtime_s, 3),
            "roster_fingerprint": self.roster_fp,
            "status": self.status,
            "near_optimal": self.near_optimal,
            "defect": self.defect,
        }


def _beam_union(state, cast, costs, proxy, *, candidate_id, settings
                ) -> Tuple[Dict[str, Completion], Optional[float]]:
    """The control search, run exactly as the ladder ran it."""
    res = complete_roster(state, cast, costs, settings=settings,
                          evaluate_ce=False, default_cost=1, proxy=proxy)
    union: Dict[str, Completion] = {}
    for c in tuple(res.finalists) + ((res.best,) if res.best else ()):
        union.setdefault(completion_fingerprint(c), c)
    best = max((c.proxy for c in union.values()), default=None)
    return union, best


def certify_price(
    state,
    cast,
    costs,
    proxy: ProxyEvaluator,
    *,
    price: int,
    label: str,
    candidate_id: Optional[int],
    pre_purchase_owned: frozenset,
    settings: CompletionSettings,
    pool_depth: int = CANDIDATE_POOL_DEPTH,
    tolerance: float = CERTIFIED_ABS_TOLERANCE,
    near_optimal_target: int = 12,
    near_optimal_band: float = 0.50,
    time_limit_s: Optional[float] = None,
    run_heuristics: bool = True,
    objective: Optional[ProxyObjective] = None,
    verbose: bool = True,
) -> PriceCertification:
    """Heuristics and the certified solver on one post-purchase state.

    The heuristics run first and their rosters are handed to the solver as
    warm starts. That is not a courtesy: a cut at a strong roster tightens the
    bound exactly where the optimum lives, so seeding makes the certificate
    cheaper without changing what is being certified. It also makes the
    comparison honest -- the solver is never allowed to "win" by having looked
    somewhere the heuristics were never pointed.
    """
    owner = state.focus
    problem = problem_from_state(state, costs, candidate_pool=pool_depth,
                                 default_cost=1)
    out = PriceCertification(
        price=price, label=label, open_slots=owner.open_slots,
        budget=owner.budget_remaining, pool_depth=pool_depth,
        board_size=len(problem.board))

    obj = objective if objective is not None else ProxyObjective(proxy)
    seeds: List[Sequence[int]] = []

    if run_heuristics:
        th = time.perf_counter()
        union, beam_best = _beam_union(state, cast, costs, proxy,
                                       candidate_id=candidate_id,
                                       settings=settings)
        out.beam = beam_best
        pool = list(problem.board)
        if union:
            union1, _ = repair_union(
                union, state=state, costs=costs, proxy=proxy,
                candidate_id=candidate_id, owned=pre_purchase_owned,
                budget=owner.budget_remaining, pool=pool, default_cost=1,
                limit=12)
            out.one_swap = max(c.proxy for c in union1.values())
            union2, _ = repair_union_two_swap(
                union1, state=state, costs=costs, proxy=proxy,
                candidate_id=candidate_id, owned=pre_purchase_owned,
                budget=owner.budget_remaining, pool=pool, default_cost=1,
                limit=TWO_SWAP_COMPLETION_LIMIT,
                shortlist_size=TWO_SWAP_SHORTLIST)
            out.two_swap = max(c.proxy for c in union2.values())
            out.union_best = out.two_swap
            best_first = sorted(union2.values(), key=lambda c: -c.proxy)[:8]
            seeds = [list(c.roster) for c in best_first]
        out.heuristic_runtime_s = time.perf_counter() - th
        if verbose:
            print(f"  heuristics  beam={out.beam} one={out.one_swap} "
                  f"two={out.two_swap}  ({out.heuristic_runtime_s:.1f}s)")

    res = solve_completion_exact(
        problem, proxy, tolerance=tolerance, time_limit_s=time_limit_s,
        objective=obj)
    out.certified = res.objective
    out.upper_bound = res.upper_bound
    out.abs_gap = res.abs_gap
    out.rel_gap = res.rel_gap
    out.iterations = res.iterations
    out.solver_runtime_s = res.runtime_s
    out.roster_fp = res.fingerprint
    out.status = res.status
    if verbose:
        print(f"  certified   {res.objective:.4f}  ub={res.upper_bound:.4f} "
              f"gap={res.abs_gap:.4f}  iters={res.iterations} "
              f"({res.runtime_s:.1f}s)  {res.status}")

    bh = out.best_heuristic
    if bh is not None and out.certified is not None and out.certified < bh - 1e-6:
        # The brief's rule: a certified optimum below a roster a heuristic
        # actually found is not a tie-break, it is a bug in the formulation or
        # in the cut, and it is recorded as one rather than reported as a
        # result.
        out.defect = (f"certified {out.certified:.6f} is below the best "
                      f"heuristic roster {bh:.6f}; formulation defect")

    if res.status == "certified" and out.defect is None:
        nos = enumerate_near_optimal(
            problem, proxy, target=near_optimal_target,
            band=near_optimal_band, tolerance=tolerance,
            time_limit_s=time_limit_s, objective=obj)
        out.near_optimal = nos.to_dict()
        if verbose:
            print(f"  near-opt    n={len(nos.entries)} "
                  f"within={nos.n_within_band} exhausted={nos.exhausted} "
                  f"({nos.runtime_s:.1f}s)")
    return out
