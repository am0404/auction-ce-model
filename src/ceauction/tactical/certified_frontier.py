"""The CE frontier, run against certified completion sets.

Structurally identical to the frontier in :mod:`.qbconvergence_experiment` --
same ladder, same pass-price rules, same ``check_agreement`` per price, same
classification. Exactly one thing differs: the opportunity sets handed to
``build_eval_context`` come from :mod:`.certified_offers` rather than from an
accumulated beam union.

That substitution is the whole point of the exercise. ``QB_UNION_UNDERCONVERGED``
withheld the frontier because the opportunity set was still improving under more
search effort, which made any bid quoted from it a statement about where the
beam stopped. A certified set is not a stopping point; every completion in it
carries a proven distance from the optimum. The gate below re-checks that
property immediately before spending any CE budget, so a set that failed to
certify cannot reach the simulator by accident.

Nothing here re-implements the five-branch lattice, the joint world builder, the
conservation checks or the decomposition. Those are consumed unchanged.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from ..auction.certified import CERTIFIED_ABS_TOLERANCE
from .board import BoardSettings
from .certified_offers import CertifiedOfferSet
from .ensemble import balanced_schedule
from .evalcontext import build_eval_context
from .midauction import PassPrice, PassPriceMode
from .reconciled import check_agreement
from .union import PriceRole, PricedValue, assert_live_biddable
from .union_experiment import _classify, _uncertainty

__all__ = [
    "CE_CANDIDATE_SET_UNDERCONVERGED", "SOLVER_GAP_TOO_WIDE",
    "FrontierGate", "gate_certified_offers", "run_certified_frontier",
]

CE_CANDIDATE_SET_UNDERCONVERGED = "CE_CANDIDATE_SET_UNDERCONVERGED"
SOLVER_GAP_TOO_WIDE = "SOLVER_GAP_TOO_WIDE"


@dataclass(frozen=True)
class FrontierGate:
    """Whether the certified sets are fit to spend CE budget on."""

    ok: bool
    failures: Tuple[str, ...]
    checks: Dict[str, object]

    def to_dict(self) -> Dict[str, object]:
        return {"ok": self.ok, "failures": list(self.failures),
                "checks": self.checks}


def gate_certified_offers(
    offers: CertifiedOfferSet, *, min_per_set: int = 12,
    tolerance: float = CERTIFIED_ABS_TOLERANCE, band: float = 0.50,
) -> FrontierGate:
    """Every precondition, checked before the frontier is allowed to run.

    Cheap, and deliberately so: the expensive checks (formulation against
    exhaustive enumeration, per-price optimality gaps) are made elsewhere and
    recorded, and this is the last structural gate between a set and the
    simulator.
    """
    rows = list(offers.provenance.get("rows", []))
    failures: List[str] = []
    checks: Dict[str, object] = {}

    worst_gap = max((float(r.get("max_gap", 0.0)) for r in rows), default=0.0)
    checks["worst_reported_gap"] = round(worst_gap, 6)
    checks["tolerance"] = tolerance
    if worst_gap > band + 1e-9:
        failures.append(
            f"{SOLVER_GAP_TOO_WIDE}: a completion is {worst_gap:.4f} from its "
            f"certified optimum, outside the {band} band it was emitted under")

    checks["with_candidate"] = len(offers.with_candidate)
    checks["without_candidate"] = len(offers.without_candidate)
    if len(offers.with_candidate) < min_per_set:
        failures.append(
            f"{CE_CANDIDATE_SET_UNDERCONVERGED}: only "
            f"{len(offers.with_candidate)} with-candidate completions, below "
            f"the declared minimum of {min_per_set}")
    if len(offers.without_candidate) < min_per_set:
        failures.append(
            f"{CE_CANDIDATE_SET_UNDERCONVERGED}: only "
            f"{len(offers.without_candidate)} without-candidate completions, "
            f"below the declared minimum of {min_per_set}")

    fps = {tuple(sorted(c.roster)) for c in offers.with_candidate}
    checks["with_candidate_distinct"] = len(fps)
    if len(fps) != len(offers.with_candidate):
        failures.append("with-candidate set contains duplicate rosters")

    costs = sorted({c.added_cost for c in offers.with_candidate})
    checks["with_candidate_spend_levels"] = len(costs)
    if len(costs) < 2:
        failures.append(
            "every with-candidate completion spends the same amount; the set "
            "cannot express a trade-off CE could prefer")

    checks["opportunity_fingerprint"] = offers.opportunity_fingerprint()
    return FrontierGate(not failures, tuple(failures), checks)


def run_certified_frontier(
    state,
    cast,
    costs,
    proxy,
    offers: CertifiedOfferSet,
    *,
    candidate_id: int,
    recipient: str,
    q: int,
    prices: Sequence[int],
    legal_max: int,
    next_bid: int,
    completion_settings,
    market=None,
    key_by_id=None,
    k: int = 11,
    increment: int = 1,
    board_settings: Optional[BoardSettings] = None,
    verbose: bool = True,
) -> Dict[str, object]:
    """One candidate's frontier across the price ladder, on certified offers."""
    t0 = time.perf_counter()
    board_settings = board_settings or BoardSettings(pool_depth=400,
                                                     max_allocations=200)
    draws = balanced_schedule(len(state.owners), k)
    points: List[Dict[str, object]] = []

    for p in prices:
        tp = time.perf_counter()
        pp = (PassPrice(mode=PassPriceMode.STOP_NOW, increment=increment,
                        standing_price=q) if p == next_bid
              else PassPrice(mode=PassPriceMode.FIXED_MARKET,
                             increment=increment, fixed_q=q))
        assert_live_biddable(PricedValue(p, PriceRole.EVALUATED_BID), q,
                             increment=increment)
        ctxs = [build_eval_context(
            state, cast, costs, candidate_id, p=p, pass_price=pp,
            recipient=recipient, q=q, draw=d, board=board_settings,
            completion=completion_settings, market=market,
            key_by_id=key_by_id, proxy=proxy,
            with_candidate=offers.with_candidate,
            without_candidate=offers.without_candidate) for d in draws]
        for ctx in ctxs:
            if not ctx.possession_offer_is_price_independent():
                raise RuntimeError(
                    f"p=${p}: UF's offer moves with the purchase price, so the "
                    f"possession effect would be measured against an offer "
                    f"already cut by a price UF never pays")
        rep = check_agreement(ctxs, proxy=proxy)
        pt = {"evaluated_bid_price": p, "p": p,
              "pass_rule": pp.mode.value, "standing_pass_price": q,
              "verdict": rep.verdict,
              "components": {kk: round(vv, 8)
                             for kk, vv in rep.component_means().items()},
              "max_per_draw_residual": rep.max_residual,
              "all_draws_agree": rep.all_agree,
              "allocation_fingerprints": sorted(
                  {d.branches["UP"].alloc_fingerprint for d in rep.draws}),
              "conservation_ok": all(b.conservation_ok for d in rep.draws
                                     for b in d.branches.values()),
              "league_ce_sums_ok": all(
                  abs(b.league_ce_sum - 1.0) < 1e-9 for d in rep.draws
                  for b in d.branches.values()),
              "completions_offered": {
                  b: rep.draws[0].branches[b].n_offered
                  for b in rep.draws[0].branches},
              "runtime_s": round(time.perf_counter() - tp, 1),
              **_uncertainty(rep)}
        points.append(pt)
        if verbose:
            ci = pt.get("t_interval_95")
            # A single draw has no between-allocation interval; that is a
            # property of k=1, not a failure, and the smoke test runs there.
            ci_txt = ("ci=[n/a] (k=1)" if not ci
                      else f"ci=[{ci[0]:+.5f},{ci[1]:+.5f}]")
            print(f"    p=${p:<4} {pp.mode.value:<13} "
                  f"delta={pt['mean_delta']:+.5f} {ci_txt} "
                  f"[{pt['verdict']}] "
                  f"resid={rep.max_residual:.1e} {pt['runtime_s']}s",
                  flush=True)

    cls = _classify(points, legal_max, next_bid)
    if cls["classification"] == "BRACKET":
        cls["classification"] = "RESERVATION_BRACKET"
    return {**cls, "points": points, "k": k,
            "opportunity_fingerprint": offers.opportunity_fingerprint(),
            "runtime_s": round(time.perf_counter() - t0, 1)}
