"""Bounded price sweep over reconciled joint worlds. Fabricated data only.

Run as ``python -m ceauction.tactical.sweep_experiment [SIMS] [OUT.json]``.
Committed output lives in ``docs/TACTICAL_JOINT_SWEEP.md``.

Slow by construction: six prices x two arms x three finalists x two independent
CE samples. It is an experiment, not a library call, and nothing imports it.
No real player appears anywhere in it.
"""

from __future__ import annotations

import json
import sys
import time

from ..auction.completion import CompletionSettings
from ..auction.proxy import ProxyEvaluator
from .board import BoardSettings
from .demo import build_tactical_demo, demo_sales
from .joint import (JointComparison, build_joint_worlds, evaluate_joint_arm)
from .recipients import enumerate_recipients

PRICES = (1, 5, 10, 13, 20, 30)


def main(argv=None) -> int:
    argv = list(sys.argv if argv is None else argv)
    sims = int(argv[1]) if len(argv) > 1 else 4000
    out_path = argv[2] if len(argv) > 2 else None

    d = build_tactical_demo()
    d = d.with_market(d.market.observe_all(demo_sales(d, 12)))
    cid = d.candidate().player_id
    st = d.state
    focus = st.focus_owner_id

    cs = CompletionSettings(beam_width=32, candidate_pool=30,
                            proxy_candidates=32, finalists=3,
                            max_candidates=160, proxy_reps=16,
                            selection_sims=sims, evaluation_sims=sims)
    px = ProxyEvaluator(st.pool, st.settings, 16, 7)
    common = dict(board_settings=BoardSettings(), completion=cs,
                  market=d.market, key_by_id=d.key_by_id, proxy=px,
                  default_cost=1, max_worlds=3)
    ev = dict(focus_team_index=d.cast.focus_team_index, proxy=px,
              selection_sims=sims, selection_seed=cs.selection_seed,
              holdout_sims=sims, holdout_seed=cs.evaluation_seed)

    rs = enumerate_recipients(st, cid, increment=1, market=d.market,
                              candidate_key=d.key_for(cid), costs=d.costs)
    named = [b for b in rs.branches if b.legal and b.owner_id][:2]
    print("named recipients:", [(b.owner_id, b.price) for b in named])

    rows, prev = [], None
    t0 = time.perf_counter()
    for price in PRICES:
        bad = st.purchase_shortfall(cid, focus, price)
        if bad is not None:
            rows.append({"price": price, "skipped": bad})
            continue
        buy = evaluate_joint_arm(build_joint_worlds(
            st.apply_purchase(cid, focus, price), d.cast, d.costs,
            branch_acquired=frozenset({cid}), **common), **ev)
        rec = named[0]
        psd = evaluate_joint_arm(build_joint_worlds(
            st.award_to_rival(cid, rec.owner_id, int(rec.price)), d.cast,
            d.costs, branch_acquired=frozenset({cid}), **common), **ev)
        cmp_ = JointComparison(buy=buy, pass_arm=psd, price=price,
                               recipient=rec.owner_id,
                               recipient_price=rec.price)
        roster = tuple(sorted(buy.world.focus_roster))
        rows.append({
            "price": price, "focus_roster": list(roster),
            "candidate_included": cid in roster,
            "roster_cost": buy.world.focus_cost + price,
            "best_alternative_if_passed": sorted(
                set(psd.world.focus_roster) - set(roster))[:4],
            "focus_proxy": round(buy.focus_proxy, 3),
            "field_min": round(min(buy.field_proxy), 2),
            "field_max": round(max(buy.field_proxy), 2),
            "field_mean": round(buy.field_mean, 2),
            "ce_buy": round(buy.ce, 5), "ce_pass": round(psd.ce, 5),
            "buy_rank": buy.rank, "pass_rank": psd.rank,
            "league_sum_buy": round(buy.league_ce_sum, 6),
            "league_ce_buy": [round(x, 5) for x in buy.league_ce],
            "roster_changed": prev is not None and roster != prev,
            "worlds_compared": buy.n_worlds_compared,
            "conservation_ok": (buy.world.conservation.ok
                                and psd.world.conservation.ok),
            **cmp_.to_dict()})
        prev = roster
        print(f"  ${price:>2}  ce_buy={buy.ce:.5f} ce_pass={psd.ce:.5f} "
              f"delta={cmp_.delta_ce:+.5f}+/-{1.96 * cmp_.delta_se:.5f} "
              f"[{cmp_.verdict}] proxy={buy.focus_proxy:.1f} "
              f"fp={buy.world.fingerprint()}  {time.perf_counter() - t0:.0f}s")

    by_recipient = []
    for rec in named:
        arm = evaluate_joint_arm(build_joint_worlds(
            st.award_to_rival(cid, rec.owner_id, int(rec.price)), d.cast,
            d.costs, branch_acquired=frozenset({cid}), **common), **ev)
        by_recipient.append({
            "recipient": rec.owner_id, "price": rec.price,
            "ce_pass": round(arm.ce, 5), "rank": arm.rank,
            "alloc_fp": arm.world.fingerprint(),
            "field_mean": round(arm.field_mean, 2),
            "league": [round(x, 5) for x in arm.league_ce]})
        print(f"  recipient {rec.owner_id} @${rec.price}: "
              f"ce_pass={arm.ce:.5f} fp={arm.world.fingerprint()}")

    wa = evaluate_joint_arm(build_joint_worlds(
        st.withdraw(cid), d.cast, d.costs,
        declared_withdrawn=frozenset({cid}), **common), **ev)
    print(f"  withdrawn (declared): ce={wa.ce:.5f} rank={wa.rank}")

    ref = 13
    bw = evaluate_joint_arm(build_joint_worlds(
        st.apply_purchase(cid, focus, ref), d.cast, d.costs,
        branch_acquired=frozenset({cid}), **common), **ev)
    pw = evaluate_joint_arm(build_joint_worlds(
        st.award_to_rival(cid, named[0].owner_id, int(named[0].price)),
        d.cast, d.costs, branch_acquired=frozenset({cid}), **common), **ev)
    next_best = {
        "reference_price": ref,
        "only_in_buy": sorted(set(bw.world.focus_roster)
                              - set(pw.world.focus_roster)),
        "only_in_pass": sorted(set(pw.world.focus_roster)
                               - set(bw.world.focus_roster)),
        "buy_cost": bw.world.focus_cost, "pass_cost": pw.world.focus_cost,
        "buy_proxy": round(bw.focus_proxy, 3),
        "pass_proxy": round(pw.focus_proxy, 3),
        "buy_field_mean": round(bw.field_mean, 3),
        "pass_field_mean": round(pw.field_mean, 3),
        "buy_rank": bw.rank, "pass_rank": pw.rank,
        "buy_reclaimed": list(bw.world.reclaimed_by_rivals),
        "pass_reclaimed": list(pw.world.reclaimed_by_rivals)}
    print("next-best: only_in_buy", next_best["only_in_buy"],
          " only_in_pass", next_best["only_in_pass"])

    blob = {"sims": sims, "candidate_id": cid, "rows": rows,
            "by_recipient": by_recipient, "next_best": next_best,
            "withdrawn": {"ce": round(wa.ce, 5), "rank": wa.rank,
                          "alloc_fp": wa.world.fingerprint(),
                          "field_mean": round(wa.field_mean, 3),
                          "league": [round(x, 5) for x in wa.league_ce]},
            "runtime_s": round(time.perf_counter() - t0, 1)}
    if out_path:
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(blob, fh, indent=2)
        print("wrote", out_path)
    print("total %.0fs" % (time.perf_counter() - t0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
