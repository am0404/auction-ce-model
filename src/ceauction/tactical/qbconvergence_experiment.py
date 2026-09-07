"""Drive the QB union convergence ladder, and gate the frontier on it.

``ce-lab tactical qb-convergence``. The full CE budget is spent only if the
predeclared ladder converges; otherwise the run reports
:data:`QB_UNION_UNDERCONVERGED` and quotes nothing. Player-level output stays
under ``local_data/``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from ..auction.completion import CompletionSettings, complete_roster
from ..auction.proxy import ProxyEvaluator
from .board import BoardSettings
from .convergence import DEFAULT_LADDER, converged_diagnose
from .ensemble import balanced_schedule
from .evalcontext import build_eval_context
from .midauction import (SIMULATED_WATERMARK, PassPrice, PassPriceMode,
                         STATE_SPECS, build_simulated_state, validate_state)
from .qbconvergence import (EFFORT_LADDER, LADDER_DECLARED_AT,
                            QB_UNION_UNDERCONVERGED, assert_independent_seeds,
                            ladder_fingerprint, run_convergence_ladder)
from .realpilot import PilotInputs, load_real_board, select_candidates
from .recipients import enumerate_recipients
from .reconciled import check_agreement
from .union import (PriceRole, PricedValue, SUPPORT_PRICES,
                    assert_live_biddable)
from .union_experiment import _classify, _uncertainty

SELECTION_SEED = 555_000_111
HOLDOUT_SEED = 917_324_011


def _base_settings(sel: int, hold: int) -> CompletionSettings:
    return CompletionSettings(
        beam_width=32, candidate_pool=40, proxy_candidates=600, finalists=8,
        max_candidates=400, proxy_reps=16, selection_sims=sel,
        evaluation_sims=hold, selection_seed=SELECTION_SEED,
        evaluation_seed=HOLDOUT_SEED, rival_selection="proxy")


def _pick_qb(board, px, reuse: Optional[Path]) -> int:
    """The QB from the converged diagnostic.

    A prior run's local output may supply it, which skips ~25 minutes of
    re-deriving the same answer. The candidate is only accepted from the cache
    if it is still on the board.
    """
    if reuse is not None and reuse.exists():
        try:
            blob = json.loads(reuse.read_text(encoding="utf-8"))
            cid = int(blob["positions"]["QB"]["candidate_id"])
        except (KeyError, ValueError, TypeError, json.JSONDecodeError):
            cid = None
        if cid is not None and cid in board.state.available_ids:
            return cid
    cands, _ = select_candidates(board)
    best = None
    for c in (x for x in cands if x.position == "QB"):
        d, _ = converged_diagnose(board, c, proxy=px, ladder=DEFAULT_LADDER)
        if d.policy != "4000-season audit":
            continue
        if best is None or d.lineup_improvement > best[1].lineup_improvement:
            best = (c, d)
    if best is None:
        raise RuntimeError("no converged audited QB candidate")
    return best[0].player_id


def run(inputs: PilotInputs, *, frontier_state: str = "balanced", k: int = 11,
        holdout_sims: int = 4000, selection_sims: int = 800,
        increment: int = 1, reuse_candidates: Optional[Path] = None,
        verbose: bool = True) -> Tuple[Dict[str, object], Dict[str, object]]:
    t0 = time.perf_counter()
    assert_independent_seeds(SELECTION_SEED, HOLDOUT_SEED)
    board = load_real_board(inputs)
    load_s = time.perf_counter() - t0
    px = ProxyEvaluator(board.state.pool, board.state.settings, 16, 7)

    t = time.perf_counter()
    qb_id = _pick_qb(board, px, reuse_candidates)
    pick_s = time.perf_counter() - t
    rb_id = None
    if reuse_candidates is not None and reuse_candidates.exists():
        try:
            rb_id = int(json.loads(reuse_candidates.read_text(
                encoding="utf-8"))["positions"]["RB"]["candidate_id"])
        except (KeyError, ValueError, TypeError, json.JSONDecodeError):
            rb_id = None
    protect = [qb_id] + ([rb_id] if rb_id is not None else [])
    sim = build_simulated_state(board, STATE_SPECS[frontier_state],
                                protect=protect)
    val = validate_state(sim, board, {"QB": qb_id})
    if not val.ok:
        raise RuntimeError(f"simulated state refused: {val.refusals}")
    st = sim.state
    legal_max = int(val.candidates["QB"]["our_legal_max"])

    rs = enumerate_recipients(st, qb_id, increment=increment,
                              market=board.market,
                              candidate_key=board.key_for(qb_id),
                              costs=board.costs["base"])
    named = [b for b in rs.branches if b.legal and b.owner_id]
    if not named:
        raise RuntimeError("no legal recipient for the QB")
    recipient = named[0].owner_id
    q = int(named[0].price)
    next_bid = q + increment
    prices = [next_bid] + [p for p in (40, 65, 80, 100) if p > next_bid]
    if legal_max > max(prices):
        prices.append(legal_max)

    base = _base_settings(selection_sims, holdout_sims)
    if verbose:
        print(SIMULATED_WATERMARK)
        print()
        print(f"state {sim.fingerprint()}  recipient {recipient}  standing "
              f"pass price ${q}  legal max ${legal_max}")
        print(f"ladder {ladder_fingerprint()} declared {LADDER_DECLARED_AT}")
        for r in EFFORT_LADDER:
            print(f"  rung {r.name:<3} x{r.multiple:<2} paths={list(r.paths)} "
                  f"finalists={r.finalists} max_candidates={r.max_candidates}")
        print()
        print(f"{'rung':<5}{'union':>7}{'new':>6}{'best':>11}"
              f"{'spread':>9}{'secs':>7}")

    def prog(rr):
        if verbose:
            print(f"{rr.rung.name:<5}{rr.union_size:>7}{rr.newly_unique:>6}"
                  f"{rr.best_proxy:>11.4f}{rr.seed_disagreement:>9.4f}"
                  f"{rr.runtime_s:>7.1f}")

    t = time.perf_counter()
    conv = run_convergence_ladder(
        st, board.cast, board.costs["base"], qb_id, base=base, proxy=px,
        evaluated_prices=prices, support_prices=SUPPORT_PRICES,
        legal_max=legal_max, progress=prog)
    union_s = time.perf_counter() - t

    out: Dict[str, object] = {
        "provenance": SIMULATED_WATERMARK, "simulated": True,
        "state": {"name": frontier_state, "fingerprint": sim.fingerprint(),
                  "validation": val.to_dict()},
        "recipient": recipient, "standing_pass_price": q,
        "next_legal_bid": next_bid, "legal_max": legal_max,
        "evaluated_bid_prices": prices,
        "support_prices": list(SUPPORT_PRICES),
        "convergence": conv.to_dict(),
        "performance": {},
    }
    local: Dict[str, object] = {"provenance": SIMULATED_WATERMARK,
                                "simulated": True, "candidate_id": qb_id,
                                "name": board.name_by_id.get(qb_id)}

    frontier_s = 0.0
    if not conv.may_evaluate_ce:
        out["frontier"] = {
            "classification": QB_UNION_UNDERCONVERGED,
            "reason": ("the predeclared effort ladder reached its maximum rung "
                       "without satisfying the stopping rule; no bid, bracket "
                       "or PASS_AT_NEXT_BID may be quoted from an "
                       "opportunity set that more effort still improves"),
            "failures": list(conv.failures),
        }
        if verbose:
            print()
            print(f"  => {QB_UNION_UNDERCONVERGED}")
            for f in conv.failures:
                print(f"     - {f}")
            print("  full CE evaluation SKIPPED: it is gated on convergence.")
    else:
        t = time.perf_counter()
        draws = balanced_schedule(len(st.owners), k)
        cs = base
        wo = complete_roster(st, board.cast, board.costs["base"], settings=cs,
                             owner_id=st.focus_owner_id, evaluate_ce=False,
                             default_cost=1, proxy=px,
                             reserved_ids=frozenset({qb_id}))
        wo_set = list(wo.finalists) or ([wo.best] if wo.best else [])
        points: List[Dict[str, object]] = []
        for p in prices:
            pp = (PassPrice(mode=PassPriceMode.STOP_NOW, increment=increment,
                            standing_price=q) if p == next_bid
                  else PassPrice(mode=PassPriceMode.FIXED_MARKET,
                                 increment=increment, fixed_q=q))
            assert_live_biddable(PricedValue(p, PriceRole.EVALUATED_BID), q,
                                 increment=increment)
            ctxs = [build_eval_context(
                st, board.cast, board.costs["base"], qb_id, p=p, pass_price=pp,
                recipient=recipient, q=q, draw=d, board=BoardSettings(
                    pool_depth=400, max_allocations=200),
                completion=cs, market=board.market,
                key_by_id=board.key_by_id, proxy=px,
                with_candidate=conv.union, without_candidate=wo_set)
                for d in draws]
            for ctx in ctxs:
                if not ctx.possession_offer_is_price_independent():
                    raise RuntimeError(
                        f"p=${p}: UF's offer moves with the purchase price")
            rep = check_agreement(ctxs, proxy=px)
            pt = {"evaluated_bid_price": p, "p": p,
                  "pass_rule": pp.mode.value, "standing_pass_price": q,
                  "verdict": rep.verdict,
                  "components": {kk: round(vv, 8)
                                 for kk, vv in rep.component_means().items()},
                  "max_per_draw_residual": rep.max_residual,
                  "all_draws_agree": rep.all_agree,
                  "allocation_fingerprints": sorted(
                      {d.branches["UP"].alloc_fingerprint for d in rep.draws}),
                  **_uncertainty(rep)}
            points.append(pt)
            if verbose:
                ci = pt["t_interval_95"]
                print(f"    p=${p:<4} {pp.mode.value:<13} "
                      f"delta={pt['mean_delta']:+.5f} "
                      f"ci=[{ci[0]:+.5f},{ci[1]:+.5f}] [{pt['verdict']}] "
                      f"resid={rep.max_residual:.1e}")
        cls = _classify(points, legal_max, next_bid)
        if cls["classification"] == "BRACKET":
            cls["classification"] = "RESERVATION_BRACKET"
        out["frontier"] = {**cls, "points": points}
        frontier_s = time.perf_counter() - t

    total = time.perf_counter() - t0
    per_rung = {r.rung.name: round(r.runtime_s, 1) for r in conv.rungs}
    out["performance"] = {
        "board_load_s": round(load_s, 1),
        "candidate_selection_s": round(pick_s, 1),
        "per_rung_s": per_rung,
        "total_union_generation_s": round(union_s, 1),
        "frontier_ce_s": round(frontier_s, 1),
        "total_candidate_runtime_s": round(total, 1),
        "cache_hit_candidate_selection": pick_s < 5.0,
        "note": ("union generation is a small fraction of the cost; the CE "
                 "evaluation of K draws x 5 branches x the price ladder "
                 "dominates. See PERFORMANCE in the documentation."),
    }
    local["convergence"] = conv.to_dict()
    local["frontier"] = out["frontier"]
    local["performance"] = out["performance"]
    return out, local
