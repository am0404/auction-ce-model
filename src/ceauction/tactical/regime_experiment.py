"""Audited candidate value across four fabricated roster-strength regimes.

``python -m ceauction.tactical.regime_experiment [HOLDOUT_SIMS] [OUT.json]``

Committed output lives in ``docs/TACTICAL_REGIMES.md``. Slow: four regimes x
four prices x two arms x reconciled joint worlds. Fabricated data only.
"""

from __future__ import annotations

import json
import sys
import time

from ..auction.completion import CompletionSettings
from ..auction.proxy import ProxyEvaluator
from .board import BoardSettings
from .joint import (JointComparison, build_joint_worlds, evaluate_joint_arm)
from .nested import build_nested_ladder
from .recipients import enumerate_recipients
from .regimes import REGIMES, build_regime

PRICES = (1, 13, 20, 30)


def main(argv=None) -> int:
    argv = list(sys.argv if argv is None else argv)
    sims = int(argv[1]) if len(argv) > 1 else 4000
    sel = max(400, sims // 4)
    out_path = argv[2] if len(argv) > 2 else None

    results = {}
    t0 = time.perf_counter()
    for name in ("underdog", "bubble", "bye_contender", "favorite"):
        d = build_regime(name)
        st = d.state
        focus = st.focus_owner_id
        cid = d.candidate().player_id
        cs = CompletionSettings(beam_width=32, candidate_pool=30,
                                proxy_candidates=32, finalists=3,
                                max_candidates=160, proxy_reps=16,
                                selection_sims=sel, evaluation_sims=sims)
        px = ProxyEvaluator(st.pool, st.settings, 16, 7)
        common = dict(board_settings=BoardSettings(), completion=cs,
                      market=d.market, key_by_id=d.key_by_id, proxy=px,
                      default_cost=1, max_worlds=3)
        ev = dict(focus_team_index=d.cast.focus_team_index, proxy=px,
                  selection_sims=sel, selection_seed=cs.selection_seed,
                  holdout_sims=sims, holdout_seed=cs.evaluation_seed)

        legal = [p for p in PRICES
                 if st.purchase_shortfall(cid, focus, p) is None]
        ladder = build_nested_ladder(st, d.cast, d.costs, cid, legal,
                                     board_settings=BoardSettings(),
                                     completion=cs, market=d.market,
                                     key_by_id=d.key_by_id, proxy=px)
        rs = enumerate_recipients(st, cid, increment=1, market=d.market,
                                  candidate_key=d.key_for(cid), costs=d.costs)
        named = [b for b in rs.branches if b.legal and b.owner_id][:2]
        rec = named[0]

        base_arm = evaluate_joint_arm(build_joint_worlds(
            st.award_to_rival(cid, rec.owner_id, int(rec.price)), d.cast,
            d.costs, branch_acquired=frozenset({cid}), **common), **ev)

        rows = []
        for price in legal:
            buy = evaluate_joint_arm(build_joint_worlds(
                st.apply_purchase(cid, focus, price), d.cast, d.costs,
                branch_acquired=frozenset({cid}),
                completions=ladder.by_price[price].feasible, **common), **ev)
            cmp_ = JointComparison(buy=buy, pass_arm=base_arm, price=price,
                                   recipient=rec.owner_id,
                                   recipient_price=rec.price)
            rows.append({
                "price": price, "ce_buy": round(buy.ce, 5),
                "ce_pass": round(base_arm.ce, 5),
                "delta_ce": round(cmp_.delta_ce, 5),
                "delta_se": round(cmp_.delta_se, 6),
                "ci95": [round(x, 5) for x in cmp_.ci95],
                "verdict": cmp_.verdict, "buy_rank": buy.rank,
                "focus_proxy": round(buy.focus_proxy, 2),
                "alloc_fp": buy.world.fingerprint(),
                "n_feasible": ladder.by_price[price].n_feasible})
            print(f"  {name:<14} ${price:>2} ce_buy={buy.ce:.5f} "
                  f"ce_pass={base_arm.ce:.5f} delta={cmp_.delta_ce:+.5f}"
                  f"+/-{1.96 * cmp_.delta_se:.5f} [{cmp_.verdict}] "
                  f"rank={buy.rank} proxy={buy.focus_proxy:.1f} "
                  f"{time.perf_counter() - t0:.0f}s")

        by_recipient = []
        if name == "bubble":
            for b in named:
                arm = evaluate_joint_arm(build_joint_worlds(
                    st.award_to_rival(cid, b.owner_id, int(b.price)), d.cast,
                    d.costs, branch_acquired=frozenset({cid}), **common), **ev)
                ridx = list(d.cast.team_names).index(b.owner_id)
                by_recipient.append({
                    "recipient": b.owner_id, "pays": b.price,
                    "our_ce_if_he_gets_him": round(arm.ce, 5),
                    "our_rank": arm.rank,
                    "recipient_ce_after": round(arm.league_ce[ridx], 5),
                    "recipient_rank_after": sorted(
                        range(12), key=lambda i: -arm.league_ce[i]
                    ).index(ridx) + 1,
                    "alloc_fp": arm.world.fingerprint()})
                print(f"    recipient {b.owner_id}: our ce={arm.ce:.5f} "
                      f"his ce={arm.league_ce[ridx]:.5f}")

        results[name] = {
            "regime": REGIMES[name].to_dict(),
            "budget_remaining": st.owner(focus).budget_remaining,
            "pass_arm": {"ce": round(base_arm.ce, 5), "rank": base_arm.rank,
                         "focus_proxy": round(base_arm.focus_proxy, 2),
                         "field_mean": round(base_arm.field_mean, 2),
                         "field_min": round(min(base_arm.field_proxy), 2),
                         "field_max": round(max(base_arm.field_proxy), 2),
                         "league_ce_sum": round(base_arm.league_ce_sum, 6)},
            "nested": {"is_nested": ladder.is_nested,
                       "union": ladder.union_size,
                       "fingerprint": ladder.fingerprint(),
                       "feasible_by_price": {
                           str(p): ladder.by_price[p].n_feasible
                           for p in ladder.prices}},
            "rows": rows, "by_recipient": by_recipient,
            "recipient": rec.owner_id, "recipient_price": rec.price}

    blob = {"holdout_sims": sims, "selection_sims": sel,
            "regimes": results, "runtime_s": round(time.perf_counter() - t0, 1)}
    if out_path:
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(blob, fh, indent=2)
        print("wrote", out_path)
    print("total %.0fs" % (time.perf_counter() - t0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
