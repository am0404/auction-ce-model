"""Does a $1-upward with-candidate union restore a positive possession effect?

``ce-lab tactical full-budget-union``. The reconciled run built the
with-candidate opportunity set only at the tested bid prices, so ``UF`` -- which
pays nothing for the candidate -- could only choose among constructions found
under a budget already cut by a price it never pays. Possession came out
negative for both candidates. This experiment rebuilds that set from $1 upward,
feeds the one union to all five branches, and reports whether the sign moves.

Every state here is SIMULATED. See
:data:`ceauction.tactical.midauction.SIMULATED_WATERMARK`.

Player-level output stays under ``local_data/``; the committed blob carries
positions, prices and CE only.
"""

from __future__ import annotations

import math
import statistics
import time
from typing import Dict, List, Optional, Sequence, Tuple

from ..auction.completion import CompletionSettings, complete_roster
from ..auction.proxy import ProxyEvaluator
from .board import BoardSettings
from .convergence import DEFAULT_LADDER, converged_diagnose
from .decompose import COMPONENT_ORDER
from .ensemble import balanced_schedule
from .evalcontext import build_eval_context
from .midauction import (SIMULATED_WATERMARK, PassPrice, PassPriceMode,
                         STATE_SPECS, build_simulated_state, validate_state)
from .realpilot import PilotInputs, load_real_board, select_candidates
from .recipients import enumerate_recipients
from .reconciled import check_agreement
from .union import (PriceRole, PricedValue, SUPPORT_PRICES, UNION_CONVERGED,
                    assert_live_biddable, build_full_budget_union,
                    check_union_adequacy, completion_fingerprint)

SELECTION_SEED = 555_000_111
HOLDOUT_SEED = 917_324_011

#: The high-only support prices the reconciled run used. Kept verbatim so the
#: comparison is against what actually ran, not a reconstruction of it.
RESTRICTED_SUPPORT_PRICES: Tuple[int, ...] = (65, 80, 100)

#: Bid prices above ``q+increment`` that each candidate is evaluated at.
EVALUATED_ABOVE: Tuple[int, ...] = (40, 65, 80, 100)

PASS_AT_NEXT_BID = "PASS_AT_NEXT_BID"
FRONTIER_NOT_REACHED = "FRONTIER_NOT_REACHED"
BRACKET = "BRACKET"
UNDERPOWERED = "UNDERPOWERED"


def _completion(sel: int, hold: int) -> CompletionSettings:
    return CompletionSettings(
        beam_width=32, candidate_pool=40, proxy_candidates=32, finalists=3,
        max_candidates=160, proxy_reps=16, selection_sims=sel,
        evaluation_sims=hold, selection_seed=SELECTION_SEED,
        evaluation_seed=HOLDOUT_SEED, rival_selection="proxy")


def _t95(df: int) -> float:
    from .ensemble import _t95 as t
    return t(df)


def _uncertainty(report) -> Dict[str, object]:
    """Allocation uncertainty and season noise, reported separately.

    The two are never pooled. The between-allocation SD asks "how much does the
    answer depend on which auction future occurs"; the within-allocation SE
    asks "how well do 4,000 seasons pin one future down". Averaging 11 x 4,000
    into 44,000 independent seasons would answer neither.
    """
    deltas = [d.frontier_delta for d in report.draws]
    k = len(deltas)
    mean = statistics.fmean(deltas)
    sd = statistics.stdev(deltas) if k > 1 else 0.0
    se_mean = sd / math.sqrt(k) if k else float("nan")
    half = _t95(k - 1) * se_mean if k > 1 else float("nan")

    # Paired within-allocation SE: difference the UP and RP season indicators
    # season by season. They share the holdout seed, so pairing is the point.
    within: List[float] = []
    for d in report.draws:
        up = d.branches["UP"].holdout_indicator
        rp = d.branches["RP"].holdout_indicator
        if up is None or rp is None:
            continue
        try:
            diff = [float(a) - float(b) for a, b in zip(up, rp)]
        except TypeError:
            continue
        n = len(diff)
        if n > 1:
            within.append(statistics.stdev(diff) / math.sqrt(n))
    rms_within = (math.sqrt(statistics.fmean([w * w for w in within]))
                  if within else None)

    fps = [d.branches["UP"].joint_fingerprint for d in report.draws]
    distinct = len(set(fps))
    pos = sum(1 for x in deltas if x > 0)
    return {
        "k": k,
        "mean_delta": round(mean, 8),
        "median_delta": round(statistics.median(deltas), 8),
        "between_allocation_sd": round(sd, 8),
        "se_of_ensemble_mean": round(se_mean, 8),
        "t_interval_95": [round(mean - half, 8), round(mean + half, 8)]
                         if k > 1 else None,
        "t_df": k - 1,
        "rms_within_allocation_se": (None if rms_within is None
                                     else round(rms_within, 8)),
        "sign_frequency_positive": round(pos / k, 4) if k else None,
        "sign_frequency_negative": round((k - pos) / k, 4) if k else None,
        "distinct_allocation_worlds": distinct,
        "redundant_draws": k - distinct,
        "effective_k": distinct,
        "pooling_note": (
            "between-allocation SD and within-allocation season SE are "
            "reported separately and never pooled; the interval above is a "
            f"cluster t-interval on {k} allocation draws (df={k - 1}), not on "
            "44,000 seasons treated as independent futures"),
    }


def _classify(points: Sequence[Dict[str, object]], legal_max: int,
              next_bid: int) -> Dict[str, object]:
    """Reservation classification. No integer refinement inside a bracket."""
    fav = [p for p in points if p["verdict"] == "favorable"]
    unf = [p for p in points if p["verdict"] == "unfavorable"]
    unresolved = [p for p in points if p["verdict"] == "unresolved"]
    first = points[0] if points else None

    if first is not None and first["p"] == next_bid and \
            first["verdict"] == "unfavorable":
        cls = PASS_AT_NEXT_BID
    elif fav and not unf and any(p["p"] == legal_max for p in fav):
        cls = FRONTIER_NOT_REACHED
    elif fav and unf:
        cls = BRACKET
    elif unresolved and not fav and not unf:
        cls = UNDERPOWERED
    else:
        cls = UNDERPOWERED if unresolved else BRACKET

    highest_fav = max((p["p"] for p in fav), default=None)
    lowest_unf = min((p["p"] for p in unf), default=None)
    bracket = ([highest_fav, lowest_unf]
               if cls == BRACKET and highest_fav is not None
               and lowest_unf is not None else None)
    return {
        "classification": cls,
        "highest_favorable_price": highest_fav,
        "lowest_unfavorable_price": lowest_unf,
        "bracket": bracket,
        "legal_max": legal_max,
        "next_legal_bid": next_bid,
        "refinement": ("none -- no integer walk inside a bracket; the endpoints "
                       "are the audited prices and nothing between them was "
                       "evaluated"),
        "meaning": {
            PASS_AT_NEXT_BID: ("unfavorable already at the legal next bid: "
                               "there is no price at which buying is worth it"),
            FRONTIER_NOT_REACHED: ("favorable through the legal maximum: the "
                                   "ladder never bounds the reservation price, "
                                   "so no maximum bid may be quoted from it"),
            BRACKET: ("the reservation price lies between the bracket "
                      "endpoints; it was not resolved to a dollar"),
            UNDERPOWERED: ("no price separated from zero at K draws; the "
                           "ensemble cannot answer this question"),
        }[cls],
    }


def run(inputs: PilotInputs, *, frontier_state: str = "balanced", k: int = 11,
        holdout_sims: int = 4000, selection_sims: int = 800,
        increment: int = 1, verbose: bool = True
        ) -> Tuple[Dict[str, object], Dict[str, object]]:
    t0 = time.perf_counter()
    board = load_real_board(inputs)
    px = ProxyEvaluator(board.state.pool, board.state.settings, 16, 7)
    cs = _completion(selection_sims, holdout_sims)
    bs = BoardSettings(pool_depth=400, max_allocations=200)
    draws = balanced_schedule(len(board.state.owners), k)
    cands, _ = select_candidates(board)

    if verbose:
        print(SIMULATED_WATERMARK)
        print()
        print("Selecting the CONVERGED QB and RB (same rule as the frontier "
              "run, so the candidates are the ones already reported on).")
    chosen: Dict[str, object] = {}
    for pos in ("QB", "RB"):
        best = None
        for c in (x for x in cands if x.position == pos):
            d, _ = converged_diagnose(board, c, proxy=px, ladder=DEFAULT_LADDER)
            if d.policy != "4000-season audit":
                continue
            if best is None or d.lineup_improvement > best[1].lineup_improvement:
                best = (c, d)
        if best is None:
            raise RuntimeError(f"no converged audited {pos} candidate")
        chosen[pos] = best
        if verbose:
            c, d = best
            print(f"  {pos}: tier={c.tier} band={c.price_low}/{c.price_base}/"
                  f"{c.price_high} anchor={c.anchor_display} "
                  f"improvement={d.lineup_improvement:+.2f}")

    protect = [chosen["QB"][0].player_id, chosen["RB"][0].player_id]
    spec = STATE_SPECS[frontier_state]
    sim = build_simulated_state(board, spec, protect=protect)
    val = validate_state(sim, board, {"QB": protect[0], "RB": protect[1]})
    if not val.ok:
        raise RuntimeError(f"simulated state refused: {val.refusals}")
    st = sim.state

    out: Dict[str, object] = {
        "provenance": SIMULATED_WATERMARK, "simulated": True,
        "state": {"name": frontier_state, "fingerprint": sim.fingerprint(),
                  "validation": val.to_dict()},
        "support_prices": list(SUPPORT_PRICES),
        "restricted_support_prices": list(RESTRICTED_SUPPORT_PRICES),
        "price_vocabulary": {
            r.value: {
                "search_support_price":
                    "generates roster constructions only; NEVER a live bid",
                "evaluated_bid_price":
                    "a purchase actually evaluated; legal only at q+increment "
                    "or above",
                "standing_pass_price":
                    "the named rival's current high bid q",
            }[r.value] for r in PriceRole},
        "positions": {},
    }
    local: Dict[str, object] = {"provenance": SIMULATED_WATERMARK,
                                "simulated": True, "positions": {}}

    for pos in ("QB", "RB"):
        c, diag = chosen[pos]
        cid = c.player_id
        rs = enumerate_recipients(st, cid, increment=increment,
                                  market=board.market,
                                  candidate_key=board.key_for(cid),
                                  costs=board.costs["base"])
        named = [b for b in rs.branches if b.legal and b.owner_id]
        if not named:
            out["positions"][pos] = {"status": "no legal recipient"}
            continue
        recipient = named[0].owner_id
        q = int(named[0].price)
        legal_max = int(val.candidates[pos]["our_legal_max"])
        next_bid = q + increment

        # --- the two unions -------------------------------------------------
        if verbose:
            print(f"\n[SIMULATED] {pos}: building unions "
                  f"(recipient {recipient}, standing pass price ${q})")
        full = build_full_budget_union(
            st, board.cast, board.costs["base"], cid, completion=cs, proxy=px,
            support_prices=SUPPORT_PRICES, legal_max=legal_max)
        restricted = build_full_budget_union(
            st, board.cast, board.costs["base"], cid, completion=cs, proxy=px,
            support_prices=RESTRICTED_SUPPORT_PRICES)
        adequacy = check_union_adequacy(
            st, board.cast, board.costs["base"], cid, restricted=restricted,
            full=full, completion=cs, proxy=px)
        if verbose:
            for r in full.rungs:
                print(f"    support ${r.search_support_price:<4} generated "
                      f"{r.generated:>3}  new {r.newly_unique:>3}  "
                      f"cumulative {r.cumulative_unique:>3}  cost "
                      f"{r.min_cost}-{r.max_cost}  effort_nested "
                      f"{r.search_effort_nested}  cross_nested "
                      f"{r.cross_price_nested}")
            print(f"    restricted {restricted.size} -> full {full.size} "
                  f"(+{adequacy.added_by_low_prices} from low support prices); "
                  f"UF proxy {adequacy.restricted_uf_proxy:.3f} -> "
                  f"{adequacy.full_uf_proxy:.3f}; {adequacy.status}")

        wo = complete_roster(st, board.cast, board.costs["base"], settings=cs,
                             owner_id=st.focus_owner_id, evaluate_ce=False,
                             default_cost=1, proxy=px,
                             reserved_ids=frozenset({cid}),
                             notes="full-budget union: without-candidate set")
        wo_set = list(wo.finalists) or ([wo.best] if wo.best else [])

        prices = [next_bid] + [p for p in EVALUATED_ABOVE if p > next_bid]
        if legal_max > max(prices):
            prices.append(legal_max)

        points: List[Dict[str, object]] = []
        for p in prices:
            # p=q+increment is the live STOP_NOW question; above it we are
            # stating an explicit counterfactual market price instead.
            if p == next_bid:
                pp = PassPrice(mode=PassPriceMode.STOP_NOW, increment=increment,
                               standing_price=q)
            else:
                pp = PassPrice(mode=PassPriceMode.FIXED_MARKET,
                               increment=increment, fixed_q=q)
            assert_live_biddable(PricedValue(p, PriceRole.EVALUATED_BID), q,
                                 increment=increment)
            if st.purchase_shortfall(cid, st.focus_owner_id, p) is not None:
                continue
            ctxs = [build_eval_context(
                st, board.cast, board.costs["base"], cid, p=p, pass_price=pp,
                recipient=recipient, q=q, draw=d, board=bs, completion=cs,
                market=board.market, key_by_id=board.key_by_id, proxy=px,
                with_candidate=full.completions, without_candidate=wo_set)
                for d in draws]
            for ctx in ctxs:
                if not ctx.possession_offer_is_price_independent():
                    raise RuntimeError(
                        f"{pos} p=${p}: UF's offer still moves with the "
                        f"purchase price; the union did not fix it")
            rep = check_agreement(ctxs, proxy=px)
            unc = _uncertainty(rep)
            pt = {
                "evaluated_bid_price": p,
                "pass_rule": pp.mode.value,
                "standing_pass_price": q,
                "verdict": rep.verdict,
                "components": {kk: round(vv, 8)
                               for kk, vv in rep.component_means().items()},
                "max_per_draw_residual": rep.max_residual,
                "all_draws_agree": rep.all_agree,
                **unc,
            }
            pt["p"] = p
            points.append(pt)
            if verbose:
                ci = pt["t_interval_95"]
                cm = pt["components"]
                print(f"    p=${p:<4} {pp.mode.value:<13} "
                      f"delta={pt['mean_delta']:+.5f} "
                      f"ci=[{ci[0]:+.5f},{ci[1]:+.5f}] "
                      f"poss={cm['our_possession']:+.5f} "
                      f"[{pt['verdict']}] resid={rep.max_residual:.1e}")

        cls = _classify(points, legal_max, next_bid)
        possession = ({p["components"]["our_possession"] for p in points}
                      if points else set())
        blob = {
            "recipient": recipient, "standing_pass_price": q,
            "next_legal_bid": next_bid, "legal_max": legal_max,
            "market_band": [c.price_low, c.price_base, c.price_high],
            "anchor": c.anchor_display,
            "lineup_improvement": diag.lineup_improvement,
            "union": full.to_dict(),
            "union_adequacy": adequacy.to_dict(),
            "possession_is_price_invariant": len(possession) <= 1,
            "possession_effect": (round(sorted(possession)[0], 8)
                                  if possession else None),
            "points": points,
            **cls,
            "bracket_quotable": (adequacy.may_quote_bracket
                                 and cls["classification"] == BRACKET),
        }
        out["positions"][pos] = blob
        local["positions"][pos] = {"candidate_id": cid,
                                   "name": board.name_by_id.get(cid), **blob}
        if verbose:
            print(f"  => {cls['classification']}  bracket {cls['bracket']}  "
                  f"union {adequacy.status}")

    out["runtime_s"] = round(time.perf_counter() - t0, 1)
    local["runtime_s"] = out["runtime_s"]
    return out, local
