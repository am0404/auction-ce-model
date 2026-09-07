"""Evaluated-price generation plus local repair, under the unchanged gate.

The control (`qbconvergence.EFFORT_LADDER`, fingerprint ``0fddb000228e1276``) is
left exactly as it was. This module is the *treatment*: same rungs, same
stopping rule, same 0.25 tolerance, but the union is built differently.

Two changes, both aimed at the diagnosed failure -- ``UP`` unstable while ``UF``
had settled, because the union was thinnest exactly where the budget binds
(1,244 affordable constructions at ``$36``, only 101 at ``$128``):

1. **Generate at the evaluated prices.** The old search generated only at
   support prices and then filtered by affordability, so expensive prices
   inherited whatever happened to survive. Here each evaluated price gets its
   own search under its own real post-purchase budget.
2. **Repair by local search.** Every generated completion can be improved by
   exhaustive legal one-player swaps (:mod:`.localrepair`), which is monotone by
   construction and therefore cannot make a rung worse.

**Every bound below is declared here, before results, and fingerprinted.**
"""

from __future__ import annotations

import dataclasses
import hashlib
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from ..auction.completion import (Completion, CompletionSettings,
                                  complete_roster)
from ..auction.costs import CostBook
from ..auction.proxy import ProxyEvaluator
from ..auction.state import AuctionState
from .localrepair import RepairReport, repair_union
from .qbconvergence import (EFFORT_LADDER, EffortRung, PROXY_TOLERANCE,
                            ConvergenceReport, RungResult,
                            evaluate_stopping_rule)
from .union import SUPPORT_PRICES, completion_fingerprint

__all__ = [
    "REPAIR_SETTINGS_DECLARED_AT", "SWAP_POOL_DEPTH", "REPAIR_TOP_K",
    "REPAIR_TOP_K_PER_PRICE",
    "price_effort_multiplier", "repair_settings_fingerprint",
    "PriceGenerationRow", "RepairedLadderResult", "run_repaired_ladder",
]

#: How deep the one-swap shortlist reaches, by projection order. A declared
#: bound, not hidden pruning: a swap-in below this depth is not considered, and
#: that fact is reported rather than left implicit.
SWAP_POOL_DEPTH = 120

#: How many completions each rung repairs at the FULL budget, strongest first.
#: The weak tail of a large union cannot become the selected roster, so
#: repairing it buys nothing; every original stays in the union regardless.
REPAIR_TOP_K = 12

#: How many are repaired under EACH evaluated price's own post-purchase budget.
#:
#: The first treatment run repaired only at the full budget, and its ``UP``
#: numbers came back byte-identical to the control. The reason is structural
#: rather than incidental: local search maximises the objective subject to the
#: budget it is given, so repairing at $139 produces rosters costing near $139,
#: none of which is affordable once $36 or more has gone to the candidate. The
#: repair could not reach the side that was failing. Repairing under each
#: evaluated price's real remaining budget is what lets it.
REPAIR_TOP_K_PER_PRICE = 6

REPAIR_SETTINGS_DECLARED_AT = (
    "2026-09-07, before any rung of the repaired experiment was run")


def price_effort_multiplier(rank: int, n_prices: int) -> float:
    """Generation effort by how tight the budget is. Declared before results.

    The cheapest evaluated price gets 1x and the dearest 3x, interpolated
    linearly. This is the rule the brief asks be fixed in advance: effort goes
    where the affordable set is thinnest, and "thinnest" is known before any
    search runs because it follows from the price, not from the outcome.
    """
    if n_prices <= 1:
        return 1.0
    return 1.0 + 2.0 * (rank / (n_prices - 1))


def repair_settings_fingerprint() -> str:
    h = hashlib.sha256()
    h.update(f"swap_pool={SWAP_POOL_DEPTH}|top_k={REPAIR_TOP_K}|"
             f"per_price_k={REPAIR_TOP_K_PER_PRICE}|"
             f"mult=1..3linear|epsilon=1e-9".encode())
    return h.hexdigest()[:16]


@dataclass
class PriceGenerationRow:
    """What generating at one evaluated price contributed."""

    evaluated_bid_price: int
    remaining_budget: int
    effort_multiplier: float
    generated: int
    newly_unique: int
    affordable_accumulated: int
    best_proxy_before_repair: float
    selected_before_repair: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "evaluated_bid_price": self.evaluated_bid_price,
            "remaining_budget_after_purchase": self.remaining_budget,
            "effort_multiplier": round(self.effort_multiplier, 2),
            "generated": self.generated, "newly_unique": self.newly_unique,
            "affordable_accumulated": self.affordable_accumulated,
            "best_proxy_before_repair": round(self.best_proxy_before_repair, 4),
            "selected_before_repair": self.selected_before_repair,
        }


@dataclass
class RepairedLadderResult:
    """The treatment run: convergence report plus the repair evidence."""

    convergence: ConvergenceReport
    price_rows: Dict[str, Tuple[PriceGenerationRow, ...]]
    repair_reports: Dict[str, RepairReport]
    generation_s: float
    repair_s: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "repair_settings_fingerprint": repair_settings_fingerprint(),
            "repair_settings_declared_at": REPAIR_SETTINGS_DECLARED_AT,
            "swap_pool_depth": SWAP_POOL_DEPTH,
            "repair_top_k": REPAIR_TOP_K,
            "repair_top_k_per_price": REPAIR_TOP_K_PER_PRICE,
            "effort_rule": "linear 1x (cheapest) to 3x (dearest evaluated price)",
            "control_ladder_unchanged": True,
            "price_generation": {
                k: [r.to_dict() for r in v]
                for k, v in self.price_rows.items()},
            "repair": {k: v.to_dict() for k, v in self.repair_reports.items()},
            "generation_s": round(self.generation_s, 1),
            "repair_s": round(self.repair_s, 1),
            **self.convergence.to_dict(),
        }


def _affordable(union: Sequence[Completion], *, costs: CostBook, budget: int,
                owned, price: int, default_cost: int = 1) -> List[Completion]:
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


def run_repaired_ladder(
    state: AuctionState, cast, costs: CostBook, candidate_id: int, *,
    base: CompletionSettings, proxy: ProxyEvaluator,
    evaluated_prices: Sequence[int],
    support_prices: Sequence[int] = SUPPORT_PRICES,
    ladder: Sequence[EffortRung] = EFFORT_LADDER,
    default_cost: int = 1, tolerance: float = PROXY_TOLERANCE,
    repair_top_k: int = REPAIR_TOP_K, swap_pool_depth: int = SWAP_POOL_DEPTH,
    per_price_k: int = REPAIR_TOP_K_PER_PRICE,
    progress=None,
) -> RepairedLadderResult:
    """The same ladder and the same stopping rule, over a repaired union."""
    focus = state.focus_owner_id
    budget = state.owner(focus).budget_remaining
    owned_after = frozenset(state.owner(focus).player_ids) | {candidate_id}
    roster_size = state.settings.roster_size
    eq_kw = dict(costs=costs, budget=budget, owned=owned_after,
                 candidate_id=candidate_id, roster_size=roster_size,
                 default_cost=default_cost, tolerance=tolerance)

    ev_prices = sorted({int(p) for p in evaluated_prices})
    sup_prices = sorted({int(p) for p in support_prices
                         if state.purchase_shortfall(candidate_id, focus,
                                                     int(p)) is None})

    seen: Dict[str, Completion] = {}
    results: List[RungResult] = []
    price_rows: Dict[str, Tuple[PriceGenerationRow, ...]] = {}
    repair_reports: Dict[str, RepairReport] = {}
    gen_s = rep_s = 0.0

    for rung in ladder:
        t_rung = time.perf_counter()
        before = len(seen)
        per_path_best: Dict[str, float] = {}
        per_path_new: Dict[str, int] = {}
        t = time.perf_counter()

        for cs, (beam, buckets) in zip(rung.settings(base), rung.paths):
            label = f"beam{beam}_buckets{buckets}"
            path_before = len(seen)
            path_best = float("-inf")
            # (a) support-price searches: these serve UF, which pays nothing
            #     and must see the widest budget any branch can have.
            for p in sup_prices:
                bought = state.apply_purchase(candidate_id, focus, p)
                res = complete_roster(
                    bought, cast, costs, settings=cs, owner_id=focus,
                    evaluate_ce=False, default_cost=default_cost, proxy=proxy,
                    reserved_ids=frozenset(),
                    notes=f"{rung.name} {label} support ${p}")
                for c in (list(res.finalists) or
                          ([res.best] if res.best else [])):
                    seen.setdefault(completion_fingerprint(c), c)
                    path_best = max(path_best, c.proxy)
            # (b) evaluated-price searches, under the real post-purchase
            #     budget, with effort scaled by how tight that budget is.
            for rank, p in enumerate(ev_prices):
                if state.purchase_shortfall(candidate_id, focus, p) is not None:
                    continue
                mult = price_effort_multiplier(rank, len(ev_prices))
                pcs = dataclasses.replace(
                    cs, finalists=max(1, int(round(cs.finalists * mult))),
                    max_candidates=max(1, int(round(cs.max_candidates * mult))))
                bought = state.apply_purchase(candidate_id, focus, p)
                res = complete_roster(
                    bought, cast, costs, settings=pcs, owner_id=focus,
                    evaluate_ce=False, default_cost=default_cost, proxy=proxy,
                    reserved_ids=frozenset(),
                    notes=f"{rung.name} {label} evaluated ${p}")
                for c in (list(res.finalists) or
                          ([res.best] if res.best else [])):
                    seen.setdefault(completion_fingerprint(c), c)
                    path_best = max(path_best, c.proxy)
            per_path_best[label] = path_best
            per_path_new[label] = len(seen) - path_before
        gen_s += time.perf_counter() - t

        # Per-evaluated-price generation evidence, BEFORE any repair.
        rows: List[PriceGenerationRow] = []
        union_now = list(seen.values())
        for rank, p in enumerate(ev_prices):
            aff = _affordable(union_now, costs=costs, budget=budget,
                              owned=owned_after, price=p,
                              default_cost=default_cost)
            b = _best(aff)
            rows.append(PriceGenerationRow(
                evaluated_bid_price=p, remaining_budget=budget - p,
                effort_multiplier=price_effort_multiplier(rank, len(ev_prices)),
                generated=len(seen) - before, newly_unique=len(seen) - before,
                affordable_accumulated=len(aff),
                best_proxy_before_repair=(b.proxy if b else float("-inf")),
                selected_before_repair=(completion_fingerprint(b) if b
                                        else "none")))
        price_rows[rung.name] = tuple(rows)

        # (c) local repair. Repaired completions are ADDED, never substituted.
        t = time.perf_counter()
        pool = [s.player_id for s in state.available_specs
                if s.player_id != candidate_id][:swap_pool_depth]
        # (c1) at the full budget: this is what UF can afford.
        seen, rr = repair_union(
            seen, state=state, costs=costs, proxy=proxy,
            candidate_id=candidate_id, owned=owned_after, budget=budget,
            pool=pool, default_cost=default_cost, limit=repair_top_k)
        # (c2) at each evaluated price's own remaining budget. Without this the
        #      repair only ever produces rosters costing near the full budget,
        #      none of which survives the affordability filter at a real
        #      purchase price -- which is exactly how the first treatment run
        #      left UP untouched.
        merged = dict(seen)
        for p in ev_prices:
            if state.purchase_shortfall(candidate_id, focus, p) is not None:
                continue
            room = budget - p
            seed = {k: c for k, c in merged.items()
                    if sum(costs.cost_of(pid, default_cost) for pid in c.roster
                           if pid not in owned_after) <= room}
            if not seed:
                continue
            repaired, sub = repair_union(
                seed, state=state, costs=costs, proxy=proxy,
                candidate_id=candidate_id, owned=owned_after, budget=room,
                pool=pool, default_cost=default_cost,
                limit=per_price_k)
            for k, c in repaired.items():
                merged.setdefault(k, c)
            rr = RepairReport(
                n_input=rr.n_input + sub.n_input,
                n_improved=rr.n_improved + sub.n_improved,
                n_already_optimal=rr.n_already_optimal + sub.n_already_optimal,
                gains=rr.gains + sub.gains,
                swap_counts=rr.swap_counts + sub.swap_counts,
                union_size_before=rr.union_size_before,
                union_size_after=len(merged),
                runtime_s=rr.runtime_s + sub.runtime_s)
        seen = merged
        rep_s += time.perf_counter() - t
        repair_reports[rung.name] = rr

        union = list(seen.values())
        uf = _best(_affordable(union, costs=costs, budget=budget,
                               owned=owned_after, price=0,
                               default_cost=default_cost))
        up_fps: Dict[int, str] = {}
        up_best: Dict[int, float] = {}
        aff_counts: Dict[int, int] = {}
        for p in ev_prices:
            aff = _affordable(union, costs=costs, budget=budget,
                              owned=owned_after, price=p,
                              default_cost=default_cost)
            b = _best(aff)
            aff_counts[p] = len(aff)
            up_fps[p] = completion_fingerprint(b) if b else "none"
            up_best[p] = b.proxy if b else float("-inf")

        rr_res = RungResult(
            rung=rung, union_size=len(seen), newly_unique=len(seen) - before,
            best_proxy=(uf.proxy if uf else float("-inf")),
            uf_fingerprint=completion_fingerprint(uf) if uf else "none",
            uf_completion=uf, up_fingerprints=up_fps, up_best_proxy=up_best,
            affordable_counts=aff_counts, per_path_best=per_path_best,
            per_path_new=per_path_new,
            runtime_s=time.perf_counter() - t_rung)
        results.append(rr_res)
        if progress is not None:
            progress(rr_res, rr)

    checks, failures = evaluate_stopping_rule(
        results, evaluated_prices=ev_prices, eq_kw=eq_kw, tolerance=tolerance)
    conv = ConvergenceReport(tuple(results), checks, tuple(failures),
                             tuple(seen.values()), tolerance)
    return RepairedLadderResult(convergence=conv, price_rows=price_rows,
                                repair_reports=repair_reports,
                                generation_s=gen_s, repair_s=rep_s)
