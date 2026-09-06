"""Controlled one-factor roster-context experiment. Fabricated data only.

``python -m ceauction.tactical.context_experiment [HOLDOUT_SIMS] [OUT.json]``
or ``ce-lab tactical context``.

Four regimes differing in exactly one thing: the projected scoring of the
players the focus team already owns. Everything else -- budget, spend, roster
size, open slots, positions, the remaining board, its costs, every rival, the
market, the candidate, its price, the recipient, and every seed -- is identical
and is asserted so before a single season is simulated.

Committed output lives in ``docs/TACTICAL_CONTEXT.md``.
"""

from __future__ import annotations

import json
import sys
import time
from typing import Dict, List, Optional

from ..auction.completion import CompletionSettings
from ..auction.proxy import ProxyEvaluator
from .board import BoardSettings
from .context import (CONTEXT_REGIMES, build_all_regimes,
                      check_structural_equality, expected_differences,
                      measure_regime)
from .joint import JointComparison, build_joint_worlds, evaluate_joint_arm
from .nested import build_nested_ladder
from .recipients import enumerate_recipients

PRICES = (1, 13, 20, 30)
SELECTION_SEED = 20260904
HOLDOUT_SEED = 917_324_011


class InvalidExperiment(RuntimeError):
    """Structural state moved. The comparison is not controlled and is refused."""


def run(holdout_sims: int = 4000, selection_sims: int = 1000,
        verbose: bool = True) -> Dict[str, object]:
    t0 = time.perf_counter()
    regimes = build_all_regimes()

    # Candidate, price, recipient and leader come from ONE regime and are then
    # reused verbatim; taking them per-regime would reintroduce a second factor.
    ref = regimes["playoff_bubble"]
    cid = ref.candidate().player_id
    rs = enumerate_recipients(ref.state, cid, increment=1, market=ref.market,
                              candidate_key=ref.key_for(cid), costs=ref.costs)
    named = [b for b in rs.branches if b.legal and b.owner_id]
    recipient, rec_price = named[0].owner_id, int(named[0].price)

    diffs = check_structural_equality(
        regimes, candidate_id=cid, price=rec_price, recipient=recipient,
        leader=None, seeds=(BoardSettings().seed, SELECTION_SEED, HOLDOUT_SEED))
    if diffs:
        raise InvalidExperiment(
            "the regimes are not structurally identical, so the comparison is "
            "not controlled:\n  - " + "\n  - ".join(str(d) for d in diffs[:8]))
    expected = expected_differences(regimes)
    if verbose:
        print(f"structural equality: PASS ({len(diffs)} differences over "
              f"{len(regimes)} regimes, all pairs)")
        print(f"expected difference: focus-owned projections differ "
              f"{expected['focus_owned_projection_total']}")
        print(f"every other projection identical: "
              f"{expected['non_focus_projections_identical']}")
        print(f"candidate {cid}, recipient {recipient} at ${rec_price}\n")

    cs = CompletionSettings(beam_width=32, candidate_pool=30,
                            proxy_candidates=32, finalists=3,
                            max_candidates=160, proxy_reps=16,
                            selection_sims=selection_sims,
                            evaluation_sims=holdout_sims,
                            selection_seed=SELECTION_SEED,
                            evaluation_seed=HOLDOUT_SEED,
                            rival_selection="proxy")
    out: Dict[str, object] = {}
    for name in ("underdog", "playoff_bubble", "bye_bubble", "favorite"):
        d = regimes[name]
        st = d.state
        focus = st.focus_owner_id
        px = ProxyEvaluator(st.pool, st.settings, 16, 7)
        common = dict(board_settings=BoardSettings(), completion=cs,
                      market=d.market, key_by_id=d.key_by_id, proxy=px,
                      default_cost=1, max_worlds=3)
        ev = dict(focus_team_index=d.cast.focus_team_index, proxy=px,
                  selection_sims=selection_sims, selection_seed=SELECTION_SEED,
                  holdout_sims=holdout_sims, holdout_seed=HOLDOUT_SEED)

        pre = measure_regime(name, d, cs, d.costs, proxy=px,
                             sims=holdout_sims, seed=HOLDOUT_SEED,
                             board_settings=BoardSettings(), market=d.market,
                             key_by_id=d.key_by_id)
        if verbose:
            print(f"{name:<16} scale={pre.strength_scale:.2f} "
                  f"proxy={pre.focus_proxy:6.1f} wins={pre.mean_wins:5.2f} "
                  f"playoff={pre.playoff_probability:.3f} "
                  f"bye={pre.bye_probability:.3f} "
                  f"ce={pre.championship_equity:.4f} rank={pre.ce_rank}")

        legal = [p for p in PRICES
                 if st.purchase_shortfall(cid, focus, p) is None]
        ladder = build_nested_ladder(st, d.cast, d.costs, cid, legal,
                                     board_settings=BoardSettings(),
                                     completion=cs, market=d.market,
                                     key_by_id=d.key_by_id, proxy=px)
        pass_arm = evaluate_joint_arm(build_joint_worlds(
            st.award_to_rival(cid, recipient, rec_price), d.cast, d.costs,
            branch_acquired=frozenset({cid}), **common), **ev)
        ridx = list(d.cast.team_names).index(recipient)

        rows: List[Dict[str, object]] = []
        prev_focus = prev_rivals = None
        for price in legal:
            buy = evaluate_joint_arm(build_joint_worlds(
                st.apply_purchase(cid, focus, price), d.cast, d.costs,
                branch_acquired=frozenset({cid}),
                completions=ladder.by_price[price].feasible, **common), **ev)
            cmp_ = JointComparison(buy=buy, pass_arm=pass_arm, price=price,
                                   recipient=recipient,
                                   recipient_price=rec_price)
            froster = tuple(sorted(buy.world.focus_roster))
            rivals = tuple(sorted(
                (o.owner_id, o.player_ids) for o in buy.world.state.owners
                if o.owner_id != focus))
            rows.append({
                "price": price,
                "focus_completion": list(froster),
                "alloc_fp": buy.world.fingerprint(),
                "ce_buy": round(buy.ce, 5), "ce_pass": round(pass_arm.ce, 5),
                "delta_ce": round(cmp_.delta_ce, 5),
                "delta_se": round(cmp_.delta_se, 6),
                "ci95": [round(x, 5) for x in cmp_.ci95],
                "verdict": cmp_.verdict,
                "buy_rank": buy.rank, "pass_rank": pass_arm.rank,
                "focus_proxy_buy": round(buy.focus_proxy, 2),
                "focus_completion_changed":
                    prev_focus is not None and froster != prev_focus,
                "rival_allocation_changed":
                    prev_rivals is not None and rivals != prev_rivals,
                "n_feasible": ladder.by_price[price].n_feasible,
                "conservation_ok": buy.world.conservation.ok})
            prev_focus, prev_rivals = froster, rivals
            if verbose:
                print(f"    ${price:>2} ce_buy={buy.ce:.5f} "
                      f"ce_pass={pass_arm.ce:.5f} "
                      f"delta={cmp_.delta_ce:+.5f}"
                      f"+/-{1.96 * cmp_.delta_se:.5f} [{cmp_.verdict}] "
                      f"{time.perf_counter() - t0:.0f}s")

        out[name] = {
            "regime": CONTEXT_REGIMES[name].to_dict(),
            "pre_acquisition": pre.to_dict(),
            "pass_arm": {"ce": round(pass_arm.ce, 5), "rank": pass_arm.rank,
                         "focus_proxy": round(pass_arm.focus_proxy, 2),
                         "league_ce_sum": round(pass_arm.league_ce_sum, 6),
                         "recipient_ce_after": round(
                             pass_arm.league_ce[ridx], 5)},
            "nested": {"is_nested": ladder.is_nested,
                       "union": ladder.union_size,
                       "feasible_by_price": {str(p): ladder.by_price[p].n_feasible
                                             for p in ladder.prices}},
            "rows": rows}

    return {"holdout_sims": holdout_sims, "selection_sims": selection_sims,
            "selection_seed": SELECTION_SEED, "holdout_seed": HOLDOUT_SEED,
            "board_seed": BoardSettings().seed,
            "candidate_id": cid, "recipient": recipient,
            "recipient_price": rec_price,
            "structural_differences": [d.to_dict() for d in diffs],
            "expected_differences": expected,
            "regimes": out,
            "runtime_s": round(time.perf_counter() - t0, 1)}


def main(argv=None) -> int:
    argv = list(sys.argv if argv is None else argv)
    sims = int(argv[1]) if len(argv) > 1 else 4000
    out_path = argv[2] if len(argv) > 2 else None
    blob = run(holdout_sims=sims, selection_sims=max(400, sims // 4))
    if out_path:
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(blob, fh, indent=2)
        print("wrote", out_path)
    print("total %.0fs" % blob["runtime_s"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
