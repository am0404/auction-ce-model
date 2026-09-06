"""Drive the real-board tactical pilot. Twelve players, one auction state.

``ce-lab tactical real-pilot``. Player-level output goes only to ``local_data/``;
the returned :class:`SanitizedReport` is the only thing fit to commit.
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from ..auction.completion import CompletionSettings
from ..auction.proxy import ProxyEvaluator
from .board import BoardSettings
from .joint import JointComparison, build_joint_worlds, evaluate_joint_arm
from .nested import build_nested_ladder
from .realpilot import (Candidate, PilotInputs, RealBoard, SanitizedReport,
                        diagnose, load_real_board, select_candidates)
from .recipients import enumerate_recipients
from .signal import allocation_spread

SELECTION_SEED = 555_000_111
HOLDOUT_SEED = 917_324_011
ALLOC_SEEDS = (20260906, 424242, 987654321)


def _completion(selection_sims: int, holdout_sims: int) -> CompletionSettings:
    return CompletionSettings(
        beam_width=32, candidate_pool=40, proxy_candidates=32, finalists=3,
        max_candidates=160, proxy_reps=16, selection_sims=selection_sims,
        evaluation_sims=holdout_sims, selection_seed=SELECTION_SEED,
        evaluation_seed=HOLDOUT_SEED, rival_selection="proxy")


def _board_settings(seed: int) -> BoardSettings:
    return BoardSettings(seed=seed, pool_depth=400, max_allocations=200)


def _arm(board: RealBoard, cid: int, state, cs, px, seed, sims, *,
         completions=None, branch=frozenset()):
    worlds = build_joint_worlds(
        state, board.cast, board.costs["base"],
        board_settings=_board_settings(seed), completion=cs,
        market=board.market, key_by_id=board.key_by_id, proxy=px,
        default_cost=1, branch_acquired=branch, completions=completions,
        max_worlds=2)
    return evaluate_joint_arm(
        worlds, focus_team_index=0, proxy=px,
        selection_sims=cs.selection_sims, selection_seed=SELECTION_SEED,
        holdout_sims=sims, holdout_seed=HOLDOUT_SEED)


def run(inputs: PilotInputs, *, holdout_sims: int = 4000,
        selection_sims: int = 800, per_position: int = 3,
        positions: Sequence[str] = ("QB", "RB", "WR", "TE"),
        alloc_seeds: Sequence[int] = ALLOC_SEEDS,
        base_price_only: bool = False, ladders: bool = True,
        runtime_budget_s: float = 1800.0,
        verbose: bool = True) -> Tuple[SanitizedReport, Dict[str, object]]:
    """Returns ``(sanitized_report, local_payload)``. Only the first is committable."""
    t0 = time.perf_counter()
    board = load_real_board(inputs)
    load_s = time.perf_counter() - t0
    cands, notes = select_candidates(board, per_position=per_position,
                                     positions=positions)
    px = ProxyEvaluator(board.state.pool, board.state.settings, 16, 7)
    cs = _completion(selection_sims, holdout_sims)
    st = board.state
    focus = board.focus_owner_id

    if verbose:
        print(f"real board loaded in {load_s:.1f}s: "
              f"{board.coverage['mapped_playerspecs']} PlayerSpecs, "
              f"{board.coverage['with_market_anchor']} anchored")
        print(f"selected {len(cands)} candidates; {len(notes)} substitutions\n")
        print(f"  {'pos':<4}{'tier':<11}{'mean':>7}{'anchor':>8}{'band':>12}"
              f"{'improve':>9}  role / policy")

    diags, local_rows = [], []
    for c in cands:
        d = diagnose(board, c, proxy=px)
        diags.append(d)
        local_rows.append({
            "player_id": c.player_id,
            "name": board.name_by_id.get(c.player_id),
            **c.to_dict(include_identity=True),
            **d.to_dict(include_identity=True)})
        if verbose:
            band = f"{c.price_low}/{c.price_base}/{c.price_high}"
            print(f"  {c.position:<4}{c.tier:<11}{c.base_mean:>7.2f}"
                  f"{str(c.anchor_display):>8}{band:>12}"
                  f"{d.lineup_improvement:>9.2f}  {d.role} / {d.policy}")

    # --- base-price CE audit ----------------------------------------------
    audited: List[Dict[str, object]] = []
    proxy_only: List[Dict[str, object]] = []
    if verbose:
        print("\nBASE-PRICE AUDIT\n")
    for c, d in zip(cands, diags):
        if d.policy == "proxy only":
            proxy_only.append({"position": c.position, "tier": c.tier,
                               "reason": d.reason,
                               "lineup_improvement": d.lineup_improvement,
                               "market_band": [c.price_low, c.price_base,
                                               c.price_high]})
            if verbose:
                print(f"  {c.position:<4}{c.tier:<11} PROXY ONLY -- "
                      f"{d.reason[:70]}")
            continue
        if time.perf_counter() - t0 > runtime_budget_s:
            notes.append(f"{c.position}/{c.tier}: skipped, runtime budget spent")
            continue
        price = c.price_base or 1
        if st.purchase_shortfall(c.player_id, focus, price) is not None:
            notes.append(f"{c.position}/{c.tier}: base price illegal; skipped")
            continue
        rs = enumerate_recipients(st, c.player_id, increment=1,
                                  market=board.market,
                                  candidate_key=board.key_for(c.player_id),
                                  costs=board.costs["base"])
        named = [b for b in rs.branches if b.legal and b.owner_id]
        if not named:
            notes.append(f"{c.position}/{c.tier}: no legal recipient; skipped")
            continue
        rec, rec_price = named[0].owner_id, int(named[0].price)
        ts = time.perf_counter()
        buy = _arm(board, c.player_id,
                   st.apply_purchase(c.player_id, focus, price), cs, px,
                   alloc_seeds[0], holdout_sims,
                   branch=frozenset({c.player_id}))
        pas = _arm(board, c.player_id,
                   st.award_to_rival(c.player_id, rec, rec_price), cs, px,
                   alloc_seeds[0], holdout_sims,
                   branch=frozenset({c.player_id}))
        cmp_ = JointComparison(buy=buy, pass_arm=pas, price=price,
                               recipient=rec, recipient_price=rec_price)
        alt = sorted(set(pas.world.focus_roster) - set(buy.world.focus_roster))
        row = {
            "position": c.position, "tier": c.tier, "price": price,
            "market_band": [c.price_low, c.price_base, c.price_high],
            "anchor_display": c.anchor_display,
            "lineup_improvement": d.lineup_improvement,
            "ce_buy": round(buy.ce, 5), "ce_pass": round(pas.ce, 5),
            "delta_ce": round(cmp_.delta_ce, 5),
            "delta_se": round(cmp_.delta_se, 6),
            "ci95": [round(x, 5) for x in cmp_.ci95],
            "verdict": cmp_.verdict,
            "buy_rank": buy.rank, "pass_rank": pas.rank,
            "league_ce_sum": round(buy.league_ce_sum, 6),
            "conservation_ok": (buy.world.conservation.ok
                                and pas.world.conservation.ok),
            "joint_fp_buy": buy.world.fingerprint(),
            "joint_fp_pass": pas.world.fingerprint(),
            "recipient": rec, "recipient_price": rec_price,
            "n_alternatives_lost": len(alt),
            "runtime_s": round(time.perf_counter() - ts, 2)}
        audited.append(row)
        local_rows.append({"player_id": c.player_id,
                           "name": board.name_by_id.get(c.player_id),
                           "audit": row,
                           "best_alternative_ids": alt[:6]})
        if verbose:
            hw = (cmp_.ci95[1] - cmp_.ci95[0]) / 2
            print(f"  {c.position:<4}{c.tier:<11}${price:>3} "
                  f"delta={cmp_.delta_ce:+.5f} +/-{hw:.5f} [{cmp_.verdict}] "
                  f"rank {pas.rank}->{buy.rank}  {row['runtime_s']:.1f}s")

    # --- allocation-seed variation for the strongest audited per position --
    spreads: Dict[str, object] = {}
    if verbose and not base_price_only:
        print("\nALLOCATION-SEED VARIATION\n")
    if not base_price_only:
        for pos in positions:
            same = [(c, d) for c, d in zip(cands, diags)
                    if c.position == pos and d.policy != "proxy only"]
            if not same:
                continue
            c, d = max(same, key=lambda t: t[1].lineup_improvement)
            price = c.price_base or 1
            rs = enumerate_recipients(st, c.player_id, increment=1,
                                      market=board.market,
                                      candidate_key=board.key_for(c.player_id),
                                      costs=board.costs["base"])
            named = [b for b in rs.branches if b.legal and b.owner_id]
            if not named:
                continue
            rec, rec_price = named[0].owner_id, int(named[0].price)
            ds, ses, fps = [], [], []
            for seed in alloc_seeds:
                if time.perf_counter() - t0 > runtime_budget_s:
                    break
                buy = _arm(board, c.player_id,
                           st.apply_purchase(c.player_id, focus, price), cs,
                           px, seed, holdout_sims,
                           branch=frozenset({c.player_id}))
                pas = _arm(board, c.player_id,
                           st.award_to_rival(c.player_id, rec, rec_price), cs,
                           px, seed, holdout_sims,
                           branch=frozenset({c.player_id}))
                cmp_ = JointComparison(buy=buy, pass_arm=pas, price=price,
                                       recipient=rec, recipient_price=rec_price)
                ds.append(cmp_.delta_ce)
                ses.append(cmp_.delta_se)
                fps.append(buy.world.rival_board.fingerprint())
            if len(ds) >= 2:
                sp = allocation_spread(alloc_seeds[:len(ds)], ds, ses, fps)
                spreads[pos] = {"tier": c.tier, **sp.to_dict()}
                if verbose:
                    print(f"  {pos:<4}{c.tier:<11} deltas="
                          f"{[round(x, 5) for x in ds]} "
                          f"between-SD={sp.between_sd:.5f} "
                          f"within-SE={sp.mean_within_se:.5f} "
                          f"sign-stable={sp.sign_stable}")

    # --- priority ladders --------------------------------------------------
    ladder_out: Dict[str, object] = {}
    if ladders and not base_price_only:
        if verbose:
            print("\nPRIORITY LADDERS (sparse; brackets, not frontiers)\n")
        for pos in positions:
            same = [(c, d) for c, d in zip(cands, diags) if c.position == pos]
            if not same:
                continue
            c, d = max(same, key=lambda t: t[1].lineup_improvement)
            lo = c.price_low or 1
            hi = c.price_high or 1
            grid = sorted({max(1, lo - 10), lo, c.price_base or lo, hi,
                           hi + 10})
            legal = [p for p in grid
                     if st.purchase_shortfall(c.player_id, focus, p) is None]
            if len(legal) < 2 or time.perf_counter() - t0 > runtime_budget_s:
                notes.append(f"{pos}: ladder skipped (budget or legality)")
                continue
            nested = build_nested_ladder(
                st, board.cast, board.costs["base"], c.player_id, legal,
                board_settings=_board_settings(alloc_seeds[0]), completion=cs,
                market=board.market, key_by_id=board.key_by_id, proxy=px)
            rs = enumerate_recipients(st, c.player_id, increment=1,
                                      market=board.market,
                                      candidate_key=board.key_for(c.player_id),
                                      costs=board.costs["base"])
            named = [b for b in rs.branches if b.legal and b.owner_id]
            rec, rec_price = named[0].owner_id, int(named[0].price)
            pas = _arm(board, c.player_id,
                       st.award_to_rival(c.player_id, rec, rec_price), cs, px,
                       alloc_seeds[0], holdout_sims,
                       branch=frozenset({c.player_id}))
            rows = []
            for price in legal:
                buy = _arm(board, c.player_id,
                           st.apply_purchase(c.player_id, focus, price), cs,
                           px, alloc_seeds[0], holdout_sims,
                           completions=nested.by_price[price].feasible,
                           branch=frozenset({c.player_id}))
                cmp_ = JointComparison(buy=buy, pass_arm=pas, price=price,
                                       recipient=rec, recipient_price=rec_price)
                rows.append({"price": price,
                             "delta_ce": round(cmp_.delta_ce, 5),
                             "delta_se": round(cmp_.delta_se, 6),
                             "ci95": [round(x, 5) for x in cmp_.ci95],
                             "verdict": cmp_.verdict,
                             "n_feasible": nested.by_price[price].n_feasible})
                if verbose:
                    print(f"  {pos:<4}${price:>3} {cmp_.delta_ce:+.5f} "
                          f"+/-{1.96 * cmp_.delta_se:.5f}  {cmp_.verdict}")
            fav = [r["price"] for r in rows if r["verdict"] == "favorable"]
            nonfav = [r["price"] for r in rows if r["verdict"] != "favorable"]
            ladder_out[pos] = {
                "tier": c.tier, "policy": d.policy,
                "market_band": [c.price_low, c.price_base, c.price_high],
                "anchor_display": c.anchor_display,
                "tested_prices": legal, "rows": rows,
                "nested": nested.is_nested,
                "robust_max": max(fav) if fav else None,
                "bracket": [max(fav), min(p for p in nonfav if p > max(fav))]
                if fav and any(p > max(fav) for p in nonfav) else None,
                "untested_gaps": [[a + 1, b - 1] for a, b in
                                  zip(legal, legal[1:]) if b - a > 1],
                "legal_max": st.owner(focus).max_bid,
                "exactness": "sparse ladder; the highest favorable price is a "
                             "BRACKET, not an exact frontier"}

    # --- opening-state symmetry -------------------------------------------
    symmetry: Dict[str, object] = {}
    audited_cands = [(c, d) for c, d in zip(cands, diags)
                     if d.policy != "proxy only"]
    if audited_cands and time.perf_counter() - t0 <= runtime_budget_s:
        c, _ = max(audited_cands, key=lambda t: t[1].lineup_improvement)
        price = c.price_base or 1
        res = []
        for rec in ("Team02", "Team03"):
            pas = _arm(board, c.player_id,
                       st.award_to_rival(c.player_id, rec, price), cs, px,
                       alloc_seeds[0], holdout_sims,
                       branch=frozenset({c.player_id}))
            res.append({"recipient": rec, "ce": round(pas.ce, 5),
                        "rank": pas.rank})
        gap = abs(res[0]["ce"] - res[1]["ce"])
        se = math_sqrt_pq(res[0]["ce"], holdout_sims)
        symmetry = {"branches": res, "abs_gap": round(gap, 5),
                    "approx_1se": round(se, 5),
                    "within_noise": gap <= 2.5 * se,
                    "note": ("in an empty room two rivals are structurally "
                             "identical; a material gap here would mean "
                             "opponent identity was invented before any sale")}
        if verbose:
            print(f"\nOPENING SYMMETRY: {res} gap={gap:.5f} "
                  f"~1SE={se:.5f} within_noise={symmetry['within_noise']}")

    runtime = {"load_s": round(load_s, 1),
               "total_s": round(time.perf_counter() - t0, 1),
               "audited_candidates": len(audited),
               "proxy_only_candidates": len(proxy_only),
               "mean_audit_s": (round(sum(r["runtime_s"] for r in audited)
                                      / len(audited), 2) if audited else None)}
    resolved = [r for r in audited if r["verdict"] != "unresolved"]
    by_pos: Dict[str, Dict[str, object]] = {}
    for r in audited:
        b = by_pos.setdefault(r["position"], {"audited": 0, "resolved": 0,
                                              "deltas": [], "anchors": [],
                                              "bands": []})
        b["audited"] += 1
        b["resolved"] += int(r["verdict"] != "unresolved")
        b["deltas"].append(r["delta_ce"])
        b["anchors"].append(r["anchor_display"])
        b["bands"].append(r["market_band"][1])

    report = SanitizedReport(
        coverage=board.coverage,
        selection={"candidates": len(cands),
                   "by_position": {p: sum(1 for c in cands if c.position == p)
                                   for p in positions},
                   "by_tier": {t: sum(1 for c in cands if c.tier == t)
                               for t in ("expensive", "mid", "cheap")},
                   "substitutions": notes},
        diagnostics=[d.to_dict() for d in diags],
        results={"audited": [{k: v for k, v in r.items()
                              if k not in ("recipient",)} for r in audited],
                 "proxy_only": proxy_only,
                 "n_resolved": len(resolved),
                 "n_unresolved": len(audited) - len(resolved),
                 "by_position": by_pos,
                 "allocation_spread": spreads,
                 "ladders": ladder_out,
                 "symmetry": symmetry},
        runtime=runtime,
        warnings=notes,
        assumptions=[
            "performance scenario: " + inputs.performance_scenario,
            "market scenario: " + inputs.market_scenario,
            "unanchored mapped players priced at the $1 league minimum, which "
            "is a stated floor and not a valuation",
            "lineup-improvement diagnostic is measured against ONE reference "
            "roster (1 QB / 4 RB / 6 WR / 3 TE); a position already carried "
            "three deep in that reference will read low",
            "no real conditional-backfield (handcuff) mapping exists; "
            "standalone projections only",
            "empty-room state only: this prices nothing mid-auction",
        ])
    local = {"candidates": local_rows, "report": report.to_dict(),
             "owner_ids": list(board.state.owner_by_id)}
    return report, local


def math_sqrt_pq(p: float, n: int) -> float:
    import math
    p = min(max(p, 0.0), 1.0)
    return math.sqrt(max(p * (1.0 - p), 1e-9) / max(n, 1))
