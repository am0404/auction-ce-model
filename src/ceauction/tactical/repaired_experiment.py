"""Drive the repaired search and gate the QB frontier on the unchanged rule."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Optional, Tuple

from ..auction.proxy import ProxyEvaluator
from .midauction import (SIMULATED_WATERMARK, STATE_SPECS,
                         build_simulated_state, validate_state)
from .qbconvergence import (QB_UNION_UNDERCONVERGED, assert_independent_seeds,
                            ladder_fingerprint)
from .qbconvergence_experiment import (HOLDOUT_SEED, SELECTION_SEED,
                                       _base_settings, _pick_qb)
from .realpilot import PilotInputs, load_real_board
from .recipients import enumerate_recipients
from .repairedsearch import REPAIR_TOP_K, run_repaired_ladder


def run(inputs: PilotInputs, *, frontier_state: str = "balanced", k: int = 11,
        holdout_sims: int = 4000, selection_sims: int = 800,
        increment: int = 1, reuse_candidates: Optional[Path] = None,
        repair_top_k: int = REPAIR_TOP_K, verbose: bool = True
        ) -> Tuple[Dict[str, object], Dict[str, object]]:
    t0 = time.perf_counter()
    assert_independent_seeds(SELECTION_SEED, HOLDOUT_SEED)
    board = load_real_board(inputs)
    load_s = time.perf_counter() - t0
    px = ProxyEvaluator(board.state.pool, board.state.settings, 16, 7)

    t = time.perf_counter()
    qb_id = _pick_qb(board, px, reuse_candidates)
    pick_s = time.perf_counter() - t

    # Both candidates are protected, exactly as the control run protected
    # them: the simulated state is a function of the protected set, so
    # dropping the RB here would silently build a DIFFERENT state and destroy
    # comparability with the control.
    rb_id = None
    if reuse_candidates is not None and reuse_candidates.exists():
        import json as _json
        try:
            rb_id = int(_json.loads(reuse_candidates.read_text(
                encoding="utf-8"))["positions"]["RB"]["candidate_id"])
        except (KeyError, ValueError, TypeError, _json.JSONDecodeError):
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
    recipient, q = named[0].owner_id, int(named[0].price)
    next_bid = q + increment
    prices = [next_bid] + [p for p in (40, 65, 80, 100) if p > next_bid]
    if legal_max > max(prices):
        prices.append(legal_max)

    if verbose:
        print(SIMULATED_WATERMARK)
        print()
        print(f"state {sim.fingerprint()}  recipient {recipient}  q=${q}  "
              f"legal max ${legal_max}")
        print(f"control ladder {ladder_fingerprint()} (UNCHANGED)")
        print(f"{'rung':<5}{'union':>7}{'new':>6}{'best':>11}{'spread':>9}"
              f"{'repaired':>10}{'maxgain':>9}{'secs':>8}")

    def prog(rr, rep):
        if verbose:
            print(f"{rr.rung.name:<5}{rr.union_size:>7}{rr.newly_unique:>6}"
                  f"{rr.best_proxy:>11.4f}{rr.seed_disagreement:>9.4f}"
                  f"{rep.n_improved:>10}{rep.max_gain:>9.4f}"
                  f"{rr.runtime_s:>8.1f}")

    t = time.perf_counter()
    res = run_repaired_ladder(
        st, board.cast, board.costs["base"], qb_id,
        base=_base_settings(selection_sims, holdout_sims), proxy=px,
        evaluated_prices=prices, repair_top_k=repair_top_k, progress=prog)
    ladder_s = time.perf_counter() - t
    conv = res.convergence

    out: Dict[str, object] = {
        "provenance": SIMULATED_WATERMARK, "simulated": True,
        "state": {"name": frontier_state, "fingerprint": sim.fingerprint()},
        "recipient": recipient, "standing_pass_price": q,
        "next_legal_bid": next_bid, "legal_max": legal_max,
        "evaluated_bid_prices": prices,
        "control_ladder_fingerprint": ladder_fingerprint(),
        "repaired": res.to_dict(),
    }
    if not conv.may_evaluate_ce:
        out["frontier"] = {
            "classification": QB_UNION_UNDERCONVERGED,
            "reason": ("the repaired search still fails the unchanged "
                       "convergence gate; the CE frontier is not run after a "
                       "failed gate"),
            "failures": list(conv.failures)}
        if verbose:
            print()
            print(f"  => {QB_UNION_UNDERCONVERGED}")
            for f in conv.failures:
                print(f"     - {f}")
            print("  CE frontier SKIPPED: gated on convergence.")
    else:  # pragma: no cover - not reached while the gate fails
        raise NotImplementedError(
            "the repaired search passed the gate; wire the frontier run here")

    total = time.perf_counter() - t0
    out["performance"] = {
        "board_load_s": round(load_s, 1),
        "candidate_selection_s": round(pick_s, 1),
        "cache_hit_candidate_selection": pick_s < 5.0,
        "evaluated_price_generation_s": round(res.generation_s, 1),
        "local_improvement_s": round(res.repair_s, 1),
        "convergence_ladder_s": round(ladder_s, 1),
        "frontier_ce_s": 0.0,
        "total_candidate_runtime_s": round(total, 1),
    }
    local = {"provenance": SIMULATED_WATERMARK, "simulated": True,
             "candidate_id": qb_id, "name": board.name_by_id.get(qb_id),
             **out}
    return out, local
