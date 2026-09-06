"""SIMULATED mid-auction frontier experiment for one QB and one RB.

``ce-lab tactical midauction-frontier``. Player-level output stays under
``local_data/``. Every state used here is invented; see
:data:`ceauction.tactical.midauction.SIMULATED_WATERMARK`.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from ..auction.completion import CompletionSettings
from ..auction.proxy import ProxyEvaluator
from .board import BoardSettings
from .convergence import DEFAULT_LADDER, converged_diagnose
from .decompose import COMPONENT_ORDER, decompose
from .ensemble import balanced_schedule
from .frontier import coarse_ladder, run_frontier
from .midauction import (SIMULATED_WATERMARK, PassPrice, PassPriceMode,
                         STATE_SPECS, build_simulated_state, validate_state)
from .realpilot import PilotInputs, load_real_board, select_candidates
from .recipients import enumerate_recipients

SELECTION_SEED = 555_000_111
HOLDOUT_SEED = 917_324_011


def _completion(sel: int, hold: int) -> CompletionSettings:
    return CompletionSettings(
        beam_width=32, candidate_pool=40, proxy_candidates=32, finalists=3,
        max_candidates=160, proxy_reps=16, selection_sims=sel,
        evaluation_sims=hold, selection_seed=SELECTION_SEED,
        evaluation_seed=HOLDOUT_SEED, rival_selection="proxy")


def run(inputs: PilotInputs, *, states: Sequence[str] = ("balanced",
                                                         "qb_inflation",
                                                         "skill_inflation"),
        frontier_state: str = "balanced", k: int = 11,
        holdout_sims: int = 4000, selection_sims: int = 800,
        increment: int = 1, audit_budget: int = 7,
        runtime_budget_s: float = 4200.0, verbose: bool = True
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
        print("Selecting the CONVERGED QB and RB from the ladder diagnostic.")
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
                  f"improvement={d.lineup_improvement:+.2f} role={d.role}")
    protect = [chosen["QB"][0].player_id, chosen["RB"][0].player_id]

    out: Dict[str, object] = {"provenance": SIMULATED_WATERMARK,
                              "simulated": True, "states": {},
                              "frontiers": {}, "decompositions": {},
                              "by_recipient": {}}
    local: Dict[str, object] = {"provenance": SIMULATED_WATERMARK,
                                "simulated": True, "states": {},
                                "frontiers": {}}

    sims: Dict[str, object] = {}
    for name in states:
        spec = STATE_SPECS[name]
        sim = build_simulated_state(board, spec, protect=protect)
        val = validate_state(sim, board,
                             {"QB": protect[0], "RB": protect[1]})
        sims[name] = (sim, val)
        out["states"][name] = {"spec": spec.to_dict(),
                               "fingerprint": sim.fingerprint(),
                               "validation": val.to_dict()}
        local["states"][name] = {**sim.to_dict(include_identity=True),
                                 "validation": val.to_dict()}
        if verbose:
            f = sim.state.owner(sim.state.focus_owner_id)
            print(f"\n[SIMULATED] {name}: {val.n_sales} sales, ${val.total_spent} "
                  f"spent, reconcile={val.dollars_reconcile}, ok={val.ok}")
            print(f"  focus: ${f.budget_remaining} left, {f.n_players} players, "
                  f"{f.open_slots} open slots")
            for lbl, cd in val.candidates.items():
                print(f"  {lbl}: our max ${cd['our_legal_max']}, top rival "
                      f"${cd['highest_rival_legal_max']} ({cd['top_rival']}), "
                      f"{cd['owners_able_to_bid']} able to bid")
            if val.refusals:
                print(f"  REFUSED: {val.refusals}")

    sim, val = sims[frontier_state]
    st = sim.state
    if not val.ok:
        raise RuntimeError(f"frontier state {frontier_state} refused: "
                           f"{val.refusals}")

    for pos in ("QB", "RB"):
        c, diag = chosen[pos]
        cid = c.player_id
        rs = enumerate_recipients(st, cid, increment=increment,
                                  market=board.market,
                                  candidate_key=board.key_for(cid),
                                  costs=board.costs["base"])
        named = [b for b in rs.branches if b.legal and b.owner_id]
        if not named:
            out["frontiers"][pos] = {"status": "no legal recipient"}
            continue
        leader = named[0].owner_id
        standing = int(named[0].price)

        # PRIMARY: STOP_NOW. The live in-auction question -- the rival already
        # leads at q and we decide whether to say q+increment. Sweeping q gives
        # the price above which we should stop.
        primary = PassPrice(mode=PassPriceMode.STOP_NOW, increment=increment,
                            standing_price=standing)
        our_max = val.candidates[pos]["our_legal_max"]
        ladder = coarse_ladder(our_max,
                               extra=[c.price_low or 1, c.price_base or 1,
                                      c.price_high or 1])

        def progress(pt):
            if verbose:
                ci = "" if pt.ci95 is None else \
                    f" ci=[{pt.ci95[0]:+.5f},{pt.ci95[1]:+.5f}]"
                print(f"    p=${pt.p:<4} q=${pt.q}  delta={pt.delta:+.5f}{ci} "
                      f"sd={pt.between_sd:.5f} [{pt.verdict}] "
                      f"{'REUSED ' if pt.reused_ce else ''}{pt.runtime_s:.0f}s")

        if verbose:
            print(f"\n[SIMULATED] {pos} FRONTIER -- pass rule STOP_NOW "
                  f"(rival {leader} leads at ${standing}; our p = q+{increment})")
            print(f"  ladder: {list(ladder)}  our legal max ${our_max}")
        fr = run_frontier(
            st, board.cast, board.costs["base"], cid, recipient=leader,
            pass_price=primary, position=pos,
            state_fingerprint=sim.fingerprint(), prices=ladder, board=bs,
            completion=cs, market=board.market, key_by_id=board.key_by_id,
            proxy=px, draws=draws, holdout_sims=holdout_sims,
            holdout_seed=HOLDOUT_SEED, audit_budget=audit_budget,
            runtime_budget_s=runtime_budget_s - (time.perf_counter() - t0),
            progress=progress)
        out["frontiers"][pos] = {
            "state": frontier_state, "leader": leader,
            "standing_price": standing,
            "market_band": [c.price_low, c.price_base, c.price_high],
            "anchor": c.anchor_display,
            "lineup_improvement": diag.lineup_improvement,
            "role": diag.role, **fr.to_dict(include_points=True)}
        local["frontiers"][pos] = {"candidate_id": cid,
                                   "name": board.name_by_id.get(cid),
                                   **out["frontiers"][pos]}
        if verbose:
            print(f"  => {fr.classification} | highest favorable "
                  f"{fr.highest_favorable} | next unfavorable "
                  f"{fr.next_unfavorable} | bracket {fr.bracket} | "
                  f"cache hits {fr.cache_hits}")
    out["runtime_s"] = round(time.perf_counter() - t0, 1)
    local["runtime_s"] = out["runtime_s"]
    out["chosen"] = {p: {"tier": chosen[p][0].tier,
                         "band": [chosen[p][0].price_low,
                                  chosen[p][0].price_base,
                                  chosen[p][0].price_high],
                         "anchor": chosen[p][0].anchor_display,
                         "improvement": chosen[p][1].lineup_improvement,
                         "role": chosen[p][1].role} for p in ("QB", "RB")}
    return out, local
