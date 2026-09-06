"""Real-board CE decomposition for the converged audited candidates.

``ce-lab tactical decompose``. Player-level output stays under ``local_data/``.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from ..auction.completion import CompletionSettings
from ..auction.proxy import ProxyEvaluator
from .board import BoardSettings
from .convergence import DEFAULT_LADDER, converged_diagnose
from .decompose import BRANCHES, COMPONENT_ORDER, Decomposition, decompose
from .ensemble import balanced_schedule
from .realpilot import PilotInputs, load_real_board, select_candidates
from .recipients import enumerate_recipients

SELECTION_SEED = 555_000_111
HOLDOUT_SEED = 917_324_011


def run(inputs: PilotInputs, *, k: int = 11, holdout_sims: int = 4000,
        selection_sims: int = 800, positions: Sequence[str] = ("RB", "TE",
                                                               "QB", "WR"),
        extra_recipients: Sequence[str] = ("RB", "TE"),
        runtime_budget_s: float = 3600.0,
        verbose: bool = True) -> Tuple[Dict[str, object], Dict[str, object]]:
    t0 = time.perf_counter()
    board = load_real_board(inputs)
    st = board.state
    px = ProxyEvaluator(st.pool, st.settings, 16, 7)
    cs = CompletionSettings(
        beam_width=32, candidate_pool=40, proxy_candidates=32, finalists=3,
        max_candidates=160, proxy_reps=16, selection_sims=selection_sims,
        evaluation_sims=holdout_sims, selection_seed=SELECTION_SEED,
        evaluation_seed=HOLDOUT_SEED, rival_selection="proxy")
    bs = BoardSettings(pool_depth=400, max_allocations=200)
    schedule = balanced_schedule(len(st.owners), k)
    cands, _ = select_candidates(board)

    if verbose:
        print("Selecting candidates with the CONVERGED ladder diagnostic.")
    chosen: Dict[str, object] = {}
    for pos in positions:
        best = None
        for c in (x for x in cands if x.position == pos):
            d, rep = converged_diagnose(board, c, proxy=px,
                                        ladder=DEFAULT_LADDER)
            if d.policy != "4000-season audit":
                continue
            if best is None or d.lineup_improvement > best[1].lineup_improvement:
                best = (c, d)
        if best is None:
            chosen[pos] = None
            if verbose:
                print(f"  {pos}: no audited candidate; skipped")
            continue
        chosen[pos] = best
        if verbose:
            print(f"  {pos}: {best[0].tier} @${best[0].price_base}, "
                  f"improvement {best[1].lineup_improvement:+.2f}, "
                  f"{best[1].role}")

    out: Dict[str, object] = {"positions": {}, "by_recipient": {}}
    local: Dict[str, object] = {"positions": {}, "by_recipient": {}}

    for pos in positions:
        entry = chosen.get(pos)
        if entry is None:
            out["positions"][pos] = {"status": "no audited candidate"}
            continue
        if time.perf_counter() - t0 > runtime_budget_s:
            out["positions"][pos] = {"status": "skipped: runtime budget"}
            continue
        c, diag = entry
        rs = enumerate_recipients(st, c.player_id, increment=1,
                                  market=board.market,
                                  candidate_key=board.key_for(c.player_id),
                                  costs=board.costs["base"])
        named = [b for b in rs.branches if b.legal and b.owner_id]
        if not named:
            out["positions"][pos] = {"status": "no legal recipient"}
            continue
        rec = named[0]

        def progress(i, total, dd):
            if verbose:
                print(f"    {pos} draw {i:>2}/{total} total={dd.total:+.5f} "
                      f"pay={dd.components['our_payment']:+.5f} "
                      f"poss={dd.components['our_possession']:+.5f} "
                      f"deny={dd.components['rival_denial']:+.5f} "
                      f"rpay={dd.components['rival_payment']:+.5f}")

        if verbose:
            print(f"\n{pos}: candidate at ${c.price_base}, rival "
                  f"{rec.owner_id} @${rec.price}, K={k}")
        dec = decompose(
            st, board.cast, board.costs["base"], c.player_id,
            focus_price=c.price_base or 1, rival=rec.owner_id,
            rival_price=int(rec.price), position=pos,
            lineup_improvement=diag.lineup_improvement, role=diag.role,
            draws=schedule, board=bs, completion=cs, market=board.market,
            key_by_id=board.key_by_id, proxy=px, holdout_sims=holdout_sims,
            holdout_seed=HOLDOUT_SEED,
            runtime_budget_s=runtime_budget_s - (time.perf_counter() - t0),
            progress=progress)
        out["positions"][pos] = dec.to_dict(include_draws=False)
        local["positions"][pos] = dec.to_dict(include_draws=True,
                                              include_identity=True)
        if verbose:
            blob = out["positions"][pos]
            print(f"  {pos} MEAN total={blob['total']['mean']:+.5f} "
                  f"pay={blob['components']['our_payment']['mean']:+.5f} "
                  f"poss={blob['components']['our_possession']['mean']:+.5f} "
                  f"deny={blob['components']['rival_denial']['mean']:+.5f} "
                  f"rpay={blob['components']['rival_payment']['mean']:+.5f} "
                  f"resid={blob['ensemble_residual']:.2e} "
                  f"-> {blob['classification']}")

        if pos in extra_recipients and len(named) > 1:
            alts = [b for b in named[1:]
                    if b.owner_id != rec.owner_id][:1]
            for alt in alts:
                if time.perf_counter() - t0 > runtime_budget_s:
                    break
                if verbose:
                    print(f"  {pos} alternate recipient {alt.owner_id} "
                          f"@${alt.price}")
                d2 = decompose(
                    st, board.cast, board.costs["base"], c.player_id,
                    focus_price=c.price_base or 1, rival=alt.owner_id,
                    rival_price=int(alt.price), position=pos,
                    lineup_improvement=diag.lineup_improvement,
                    role=diag.role, draws=schedule, board=bs, completion=cs,
                    market=board.market, key_by_id=board.key_by_id, proxy=px,
                    holdout_sims=holdout_sims, holdout_seed=HOLDOUT_SEED,
                    runtime_budget_s=runtime_budget_s
                    - (time.perf_counter() - t0))
                out["by_recipient"].setdefault(pos, []).append(
                    d2.to_dict(include_draws=False))
                local["by_recipient"].setdefault(pos, []).append(
                    d2.to_dict(include_draws=True, include_identity=True))
                if verbose:
                    b2 = out["by_recipient"][pos][-1]
                    print(f"    total={b2['total']['mean']:+.5f} "
                          f"deny={b2['components']['rival_denial']['mean']:+.5f} "
                          f"rpay={b2['components']['rival_payment']['mean']:+.5f}")

    out["settings"] = {"k": k, "holdout_sims": holdout_sims,
                       "selection_sims": selection_sims,
                       "selection_seed": SELECTION_SEED,
                       "holdout_seed": HOLDOUT_SEED,
                       "branches": {b: s.to_dict()
                                    for b, s in BRANCHES.items()},
                       "component_order": [list(t) for t in COMPONENT_ORDER]}
    out["coverage"] = board.coverage
    out["runtime_s"] = round(time.perf_counter() - t0, 1)
    return out, local


def main(argv=None) -> int:
    argv = list(sys.argv if argv is None else argv)
    sims = int(argv[1]) if len(argv) > 1 else 4000
    out_path = argv[2] if len(argv) > 2 else None
    inputs = PilotInputs(
        contract=Path("local_data/real_player_contract_v1.json"),
        sleeper_csv=Path("local_data/sleeper_2qb_values_2026_clean.csv"),
        out_dir=Path("local_data/tactical"))
    blob, local = run(inputs, holdout_sims=sims)
    Path("local_data/tactical").mkdir(parents=True, exist_ok=True)
    Path("local_data/tactical/decomposition.json").write_text(
        json.dumps(local, indent=2, default=str), encoding="utf-8")
    if out_path:
        Path(out_path).write_text(json.dumps(blob, indent=2, default=str),
                                  encoding="utf-8")
    print("total %.0fs" % blob["runtime_s"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
