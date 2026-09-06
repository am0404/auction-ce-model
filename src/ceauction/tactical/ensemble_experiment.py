"""Allocation-ensemble experiment on the real board. Fabricated-safe outputs.

``ce-lab tactical allocation-ensemble``. Player-level detail goes only to
``local_data/``; the returned aggregate carries position-level numbers only.
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
from .ensemble import (FRONTIER_NOT_REACHED, balanced_schedule,
                       run_ensemble, symmetry_check)
from .convergence import CONVERGED, DEFAULT_LADDER, converged_diagnose
from .realpilot import PilotInputs, load_real_board, select_candidates
from .recipients import enumerate_recipients

CONVERGENCE_KS = (1, 3, 6, 12, 15, 24)
SELECTION_SEED = 555_000_111
HOLDOUT_SEED = 917_324_011


def _completion(sel: int, hold: int) -> CompletionSettings:
    return CompletionSettings(
        beam_width=32, candidate_pool=40, proxy_candidates=32, finalists=3,
        max_candidates=160, proxy_reps=16, selection_sims=sel,
        evaluation_sims=hold, selection_seed=SELECTION_SEED,
        evaluation_seed=HOLDOUT_SEED, rival_selection="proxy")


def _board(shock: float, scenario: str) -> BoardSettings:
    return BoardSettings(pool_depth=400, max_allocations=200,
                         tie_break="mechanical", tie_tolerance=0.5,
                         preference_shock=shock, shock_scenario=scenario,
                         jitter=0.0)


def run(inputs: PilotInputs, *, k: int = 15, holdout_sims: int = 4000,
        selection_sims: int = 800, shock: float = 0.0,
        shock_scenario: str = "none",
        positions: Sequence[str] = ("RB", "QB", "WR", "TE"),
        secondary_k: int = 6, runtime_budget_s: float = 3000.0,
        verbose: bool = True) -> Tuple[Dict[str, object], Dict[str, object]]:
    t0 = time.perf_counter()
    board = load_real_board(inputs)
    st = board.state
    focus = board.focus_owner_id
    px = ProxyEvaluator(st.pool, st.settings, 16, 7)
    cs = _completion(selection_sims, holdout_sims)
    bs = _board(shock, shock_scenario)
    cands, _ = select_candidates(board)
    # Selection comes from the CONVERGED ladder diagnostic, not a single beam.
    # The previous branch chose ensemble candidates with an unconverged
    # beam-32 diagnostic, so its CE numbers described players the corrected
    # diagnostic does not nominate.
    diags, convergence = {}, {}
    for c in cands:
        d, rep = converged_diagnose(board, c, proxy=px, ladder=DEFAULT_LADDER)
        diags[(c.position, c.tier)] = d
        convergence[(c.position, c.tier)] = rep
    schedule = balanced_schedule(len(st.owners), k)

    def recipient_for(cid: int) -> Optional[Tuple[str, int]]:
        rs = enumerate_recipients(st, cid, increment=1, market=board.market,
                                  candidate_key=board.key_for(cid),
                                  costs=board.costs["base"])
        named = [b for b in rs.branches if b.legal and b.owner_id]
        return (named[0].owner_id, int(named[0].price)) if named else None

    out: Dict[str, object] = {"positions": {}}
    local: Dict[str, object] = {"positions": {}}

    # --- primary: the RB that previously flipped sign ----------------------
    primary_pos = positions[0]
    # The candidate a quota-free diagnostic says is worth auditing, not the
    # most expensive one. Price does not decide who earns CE seasons.
    audited_here = [c for c in cands if c.position == primary_pos
                    and diags[(c.position, c.tier)].policy == "4000-season audit"]
    if not audited_here:
        raise RuntimeError(
            f"no {primary_pos} qualifies for a CE audit under the quota-free "
            f"diagnostic; nothing to run")
    prim = max(audited_here,
               key=lambda c: diags[(c.position, c.tier)].lineup_improvement)
    rec = recipient_for(prim.player_id)
    if rec is None:
        raise RuntimeError("no legal recipient for the primary candidate")
    if verbose:
        print(f"primary: {primary_pos} {prim.tier}, price ${prim.price_base}, "
              f"recipient {rec[0]} @${rec[1]}, k={k}\n")

    def progress(i, total, dr):
        if verbose:
            print(f"  draw {i:>2}/{total} rot={dr.draw.rotation:>2} "
                  f"delta={dr.delta:+.5f} se={dr.paired_se:.5f} "
                  f"alloc={dr.alloc_fingerprint} {dr.runtime_s:.1f}s")

    ens = run_ensemble(
        st, board.cast, board.costs["base"], prim.player_id,
        prim.price_base or 1, rec[0], rec[1], draws=schedule, board=bs,
        completion=cs, market=board.market, key_by_id=board.key_by_id,
        proxy=px, holdout_sims=holdout_sims, holdout_seed=HOLDOUT_SEED,
        performance_scenario=inputs.performance_scenario,
        runtime_budget_s=runtime_budget_s, progress=progress)
    out["primary"] = {"position": primary_pos, "tier": prim.tier,
                      "role": diags[(prim.position, prim.tier)].role,
                      "lineup_improvement":
                          diags[(prim.position, prim.tier)].lineup_improvement,
                      "sampling_basis":
                          diags[(prim.position, prim.tier)].reason,
                      "convergence": convergence[
                          (prim.position, prim.tier)].to_dict()["status"],
                      **ens.to_dict(include_draws=False),
                      "convergence": ens.running(CONVERGENCE_KS)}
    local["primary"] = ens.to_dict(include_draws=True)
    if verbose:
        lo, hi = ens.ci95
        print(f"\n  ENSEMBLE k={ens.k} mean={ens.mean:+.5f} "
              f"between-SD={ens.between_sd:.5f} within-SE={ens.rms_within_se:.5f} "
              f"CI95=[{lo:+.5f},{hi:+.5f}] sign+={ens.sign_frequency['positive']:.2f} "
              f"distinct-allocs={ens.distinct_allocations} -> {ens.verdict}")

    # --- symmetry ----------------------------------------------------------
    sym = symmetry_check(
        st, board.cast, board.costs["base"], prim.player_id,
        prim.price_base or 1, "Team02", "Team03", draws=schedule, board=bs,
        completion=cs, market=board.market, key_by_id=board.key_by_id,
        proxy=px, holdout_sims=holdout_sims, holdout_seed=HOLDOUT_SEED,
        runtime_budget_s=max(60.0, runtime_budget_s - (time.perf_counter() - t0)))
    out["symmetry"] = sym.to_dict()
    if verbose:
        lo, hi = sym.ci95
        print(f"\n  SYMMETRY Team02 vs Team03: single-seed gap "
              f"{sym.single_seed_gap:+.5f}, ensemble mean {sym.mean:+.5f}, "
              f"CI95=[{lo:+.5f},{hi:+.5f}] contains_zero={sym.contains_zero} "
              f"max|diff|={sym.max_abs:.5f}")

    # --- secondary positions ----------------------------------------------
    small = balanced_schedule(len(st.owners), secondary_k)
    for pos in positions[1:]:
        eligible = [x for x in cands if x.position == pos
                    and diags[(x.position, x.tier)].policy == "4000-season audit"]
        if eligible:
            c = max(eligible,
                    key=lambda x: diags[(x.position, x.tier)].lineup_improvement)
            d = diags[(c.position, c.tier)]
        else:
            c = next((x for x in cands if x.position == pos), None)
            d = diags.get((c.position, c.tier)) if c else None
        if c is None or d is None:
            continue
        if d.policy == "proxy only":
            out["positions"][pos] = {
                "status": "PROXY ONLY -- retained; no CE seasons spent",
                "tier": c.tier,
                "lineup_improvement": d.lineup_improvement,
                "improvement_at_min": d.improvement_at_min,
                "role": d.role, "reason": d.reason}
            if verbose:
                print(f"\n  {pos}: PROXY ONLY (improvement "
                      f"{d.lineup_improvement:.2f}); no ensemble run")
            continue
        if time.perf_counter() - t0 > runtime_budget_s:
            out["positions"][pos] = {"status": "skipped: runtime budget"}
            continue
        r = recipient_for(c.player_id)
        if r is None:
            continue
        e = run_ensemble(
            st, board.cast, board.costs["base"], c.player_id,
            c.price_base or 1, r[0], r[1], draws=small, board=bs,
            completion=cs, market=board.market, key_by_id=board.key_by_id,
            proxy=px, holdout_sims=holdout_sims, holdout_seed=HOLDOUT_SEED,
            performance_scenario=inputs.performance_scenario,
            runtime_budget_s=runtime_budget_s - (time.perf_counter() - t0))
        out["positions"][pos] = {
            "tier": c.tier, "role": d.role,
            "lineup_improvement": d.lineup_improvement,
            "sampling_basis": d.reason,
            "convergence": convergence[(c.position, c.tier)].to_dict()["status"],
            **e.to_dict(include_draws=False)}
        local["positions"][pos] = e.to_dict(include_draws=True)
        if verbose:
            lo, hi = e.ci95
            print(f"\n  {pos} k={e.k} mean={e.mean:+.5f} "
                  f"between-SD={e.between_sd:.5f} "
                  f"CI95=[{lo:+.5f},{hi:+.5f}] -> {e.verdict}")

    out["schedule"] = {"k": k, "secondary_k": secondary_k,
                       "draws": [d.to_dict() for d in schedule],
                       "balanced": True,
                       "note": ("predetermined rotation schedule; every rival "
                                "occupies every priority slot equally often")}
    out["settings"] = {"tie_break": bs.tie_break,
                       "tie_tolerance": bs.tie_tolerance,
                       "preference_shock": bs.preference_shock,
                       "shock_scenario": bs.shock_scenario,
                       "holdout_sims": holdout_sims,
                       "selection_sims": selection_sims,
                       "selection_seed": SELECTION_SEED,
                       "holdout_seed": HOLDOUT_SEED,
                       "performance_scenario": inputs.performance_scenario}
    out["coverage"] = board.coverage
    out["runtime_s"] = round(time.perf_counter() - t0, 1)
    out["price_ladder_policy"] = {
        "status": FRONTIER_NOT_REACHED,
        "note": ("no ladder is run here. The pilot's ladders never changed "
                 "the allocation over their tested range, so "
                 "no maximum bid may be quoted "
                 "from them; that is a statement about an empty room where "
                 "money is not scarce, not about price insensitivity.")}
    return out, local
