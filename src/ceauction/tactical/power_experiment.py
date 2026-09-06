"""Estimator power by candidate tier and roster context. Fabricated data only.

``python -m ceauction.tactical.power_experiment [SIMS] [OUT.json]`` or
``ce-lab tactical signal-power``. Committed output: ``docs/TACTICAL_POWER.md``.

Experiment A holds price fixed and crosses four candidate tiers with four
controlled roster contexts, so scoring strength and roster context are
separated from price. Experiment B walks a tier-appropriate price ladder.
Both run over nested completion sets and reconciled joint worlds.
"""

from __future__ import annotations

import json
import math
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

from ..auction.completion import CompletionSettings
from ..auction.proxy import ProxyEvaluator
from .board import BoardSettings
from .joint import JointComparison, build_joint_worlds, evaluate_joint_arm
from .nested import build_nested_ladder
from .recipients import enumerate_recipients
from .signal import (CANDIDATE_TIERS, HALF_WIDTH_TARGETS, TIER_PRICE_LADDERS,
                     allocation_spread, build_tiered_world, calibrate_tier,
                     plan_confirmatory)

CONTEXTS = ("underdog", "playoff_bubble", "bye_bubble", "favorite")
TIERS = ("bench", "marginal_starter", "strong_starter", "elite")
COMMON_PRICE = 13
PILOT_SEED = 20260904
CONFIRM_SEED = 917_324_011
SELECTION_SEED = 555_000_111
ALLOC_SEEDS = (20260906, 424242, 987654321)


def _settings(pilot: int, confirm: int) -> CompletionSettings:
    return CompletionSettings(
        beam_width=32, candidate_pool=30, proxy_candidates=32, finalists=3,
        max_candidates=160, proxy_reps=16, selection_sims=max(300, pilot // 2),
        evaluation_sims=confirm, selection_seed=SELECTION_SEED,
        evaluation_seed=CONFIRM_SEED, rival_selection="proxy")


def _arms(d, cid, price, recipient, rec_price, cs, px, board_seed, sims, seed):
    """One paired buy/pass comparison inside one shared-board allocation."""
    st = d.state
    focus = st.focus_owner_id
    bs = BoardSettings(seed=board_seed)
    common = dict(board_settings=bs, completion=cs, market=d.market,
                  key_by_id=d.key_by_id, proxy=px, default_cost=1,
                  max_worlds=3)
    ev = dict(focus_team_index=d.cast.focus_team_index, proxy=px,
              selection_sims=cs.selection_sims, selection_seed=SELECTION_SEED,
              holdout_sims=sims, holdout_seed=seed)
    ladder = build_nested_ladder(st, d.cast, d.costs, cid, [price],
                                 board_settings=bs, completion=cs,
                                 market=d.market, key_by_id=d.key_by_id,
                                 proxy=px)
    buy = evaluate_joint_arm(build_joint_worlds(
        st.apply_purchase(cid, focus, price), d.cast, d.costs,
        branch_acquired=frozenset({cid}),
        completions=ladder.by_price[price].feasible, **common), **ev)
    pas = evaluate_joint_arm(build_joint_worlds(
        st.award_to_rival(cid, recipient, rec_price), d.cast, d.costs,
        branch_acquired=frozenset({cid}), **common), **ev)
    return buy, pas, JointComparison(buy=buy, pass_arm=pas, price=price,
                                     recipient=recipient,
                                     recipient_price=rec_price)


def run(pilot_sims: int = 1000, confirm_sims: int = 4000,
        cap_sims: int = 40_000, target_half_width: float = 0.005,
        alloc_seeds: Sequence[int] = ALLOC_SEEDS,
        verbose: bool = True) -> Dict[str, object]:
    t0 = time.perf_counter()
    cs = _settings(pilot_sims, confirm_sims)
    power: List[Dict[str, object]] = []
    calib: Dict[str, object] = {}

    if verbose:
        print("EXPERIMENT A -- estimator power (price fixed at $%d)\n" % COMMON_PRICE)
        print(f"  {'tier':<17}{'context':<16}{'improve':>9}{'delta':>10}"
              f"{'+/-95%':>9}{'|d|/SE':>8}  verdict      req@0.005")

    for tier in TIERS:
        for context in CONTEXTS:
            ts = time.perf_counter()
            d, cid = build_tiered_world(context, tier)
            st = d.state
            focus = st.focus_owner_id
            px = ProxyEvaluator(st.pool, st.settings, 16, 7)
            cal = calibrate_tier(d, cid, tier, context, proxy=px)
            calib.setdefault(tier, {})[context] = cal.to_dict()

            rs = enumerate_recipients(st, cid, increment=1, market=d.market,
                                      candidate_key=d.key_for(cid),
                                      costs=d.costs)
            named = [b for b in rs.branches if b.legal and b.owner_id]
            recipient, rec_price = named[0].owner_id, int(named[0].price)
            price = COMMON_PRICE
            if st.purchase_shortfall(cid, focus, price) is not None:
                power.append({"tier": tier, "context": context,
                              "skipped": st.purchase_shortfall(cid, focus, price)})
                continue

            # Pilot: independent sample, used ONLY to estimate variance.
            _, _, pilot = _arms(d, cid, price, recipient, rec_price, cs, px,
                                ALLOC_SEEDS[0], pilot_sims, PILOT_SEED)
            plan = plan_confirmatory(
                pilot.delta_ce, pilot.delta_se, pilot_sims,
                target_half_width=target_half_width, cap_n=cap_sims,
                pilot_seed=PILOT_SEED, confirmatory_seed=CONFIRM_SEED,
                targets=HALF_WIDTH_TARGETS)

            # Confirmatory: an independent sample at a different seed. Its
            # interval reuses no pilot season.
            buy, pas, conf = _arms(d, cid, price, recipient, rec_price, cs, px,
                                   ALLOC_SEEDS[0], confirm_sims, CONFIRM_SEED)
            row = {
                "tier": tier, "context": context, "price": price,
                "candidate_id": cid,
                "lineup_improvement": round(cal.lineup_improvement, 4),
                "candidate_weekly": round(cal.candidate_weekly, 3),
                "replacement_weekly": round(cal.replacement_weekly, 3),
                "displaced_player_id": cal.displaced_player_id,
                "displaced_weekly": round(cal.displaced_weekly, 3),
                "starts": cal.starts,
                "pre_ce": round(pas.ce, 5), "pre_rank": pas.rank,
                "ce_buy": round(buy.ce, 5), "ce_pass": round(pas.ce, 5),
                "delta_ce": round(conf.delta_ce, 5),
                "delta_se": round(conf.delta_se, 6),
                "ci95": [round(x, 5) for x in conf.ci95],
                "abs_t": (round(abs(conf.delta_ce) / conf.delta_se, 3)
                          if conf.delta_se else None),
                "verdict": conf.verdict,
                "focus_completion_fp": buy.world.fingerprint(),
                "alloc_fp": buy.world.rival_board.fingerprint(),
                "conservation_ok": buy.world.conservation.ok,
                "league_ce_sum": round(buy.league_ce_sum, 6),
                "plan": plan.to_dict(),
                "runtime_s": round(time.perf_counter() - ts, 2)}
            power.append(row)
            if verbose:
                hw = (conf.ci95[1] - conf.ci95[0]) / 2
                print(f"  {tier:<17}{context:<16}"
                      f"{cal.lineup_improvement:>9.2f}{conf.delta_ce:>+10.5f}"
                      f"{hw:>9.5f}"
                      f"{(abs(conf.delta_ce) / conf.delta_se if conf.delta_se else 0):>8.2f}"
                      f"  {conf.verdict:<12} "
                      f"{plan.requirements[str(target_half_width)]:>9,}")

    # --- Experiment B: tier-appropriate price ladders ----------------------
    econ: List[Dict[str, object]] = []
    if verbose:
        print("\nEXPERIMENT B -- price sensitivity (EXPERIMENTAL ladders, "
              "NOT market predictions)\n")
    context = "playoff_bubble"
    for tier in TIERS:
        d, cid = build_tiered_world(context, tier)
        st = d.state
        focus = st.focus_owner_id
        px = ProxyEvaluator(st.pool, st.settings, 16, 7)
        rs = enumerate_recipients(st, cid, increment=1, market=d.market,
                                  candidate_key=d.key_for(cid), costs=d.costs)
        named = [b for b in rs.branches if b.legal and b.owner_id]
        recipient, rec_price = named[0].owner_id, int(named[0].price)
        legal = [p for p in TIER_PRICE_LADDERS[tier]
                 if st.purchase_shortfall(cid, focus, p) is None]
        bs = BoardSettings(seed=ALLOC_SEEDS[0])
        ladder = build_nested_ladder(st, d.cast, d.costs, cid, legal,
                                     board_settings=bs, completion=cs,
                                     market=d.market, key_by_id=d.key_by_id,
                                     proxy=px)
        for price in legal:
            _, _, cmp_ = _arms(d, cid, price, recipient, rec_price, cs, px,
                               ALLOC_SEEDS[0], confirm_sims, CONFIRM_SEED)
            econ.append({
                "tier": tier, "context": context, "price": price,
                "delta_ce": round(cmp_.delta_ce, 5),
                "delta_se": round(cmp_.delta_se, 6),
                "ci95": [round(x, 5) for x in cmp_.ci95],
                "verdict": cmp_.verdict,
                "n_feasible": ladder.by_price[price].n_feasible,
                "alloc_fp": cmp_.buy.world.fingerprint()})
            if verbose:
                print(f"  {tier:<17}${price:>3}  {cmp_.delta_ce:+.5f} "
                      f"+/-{1.96 * cmp_.delta_se:.5f}  {cmp_.verdict}")
        econ.append({"tier": tier, "nested": ladder.is_nested,
                     "feasible_by_price": {str(p): ladder.by_price[p].n_feasible
                                           for p in ladder.prices}})

    # --- allocation-seed variation ----------------------------------------
    spreads: Dict[str, object] = {}
    if verbose:
        print("\nALLOCATION-SEED VARIATION (marginal / strong / elite)\n")
    for tier in ("marginal_starter", "strong_starter", "elite"):
        d, cid = build_tiered_world(context, tier)
        st = d.state
        focus = st.focus_owner_id
        px = ProxyEvaluator(st.pool, st.settings, 16, 7)
        rs = enumerate_recipients(st, cid, increment=1, market=d.market,
                                  candidate_key=d.key_for(cid), costs=d.costs)
        named = [b for b in rs.branches if b.legal and b.owner_id]
        recipient, rec_price = named[0].owner_id, int(named[0].price)
        ds, ses, fps = [], [], []
        for seed in alloc_seeds:
            _, _, cmp_ = _arms(d, cid, COMMON_PRICE, recipient, rec_price, cs,
                               px, seed, confirm_sims, CONFIRM_SEED)
            ds.append(cmp_.delta_ce)
            ses.append(cmp_.delta_se)
            fps.append(cmp_.buy.world.rival_board.fingerprint())
        sp = allocation_spread(alloc_seeds, ds, ses, fps)
        spreads[tier] = sp.to_dict()
        if verbose:
            print(f"  {tier:<17} deltas={[round(x, 5) for x in ds]} "
                  f"between-SD={sp.between_sd:.5f} "
                  f"within-SE={sp.mean_within_se:.5f} "
                  f"alloc-dominated={sp.dominated_by_allocation}")

    return {"pilot_sims": pilot_sims, "confirm_sims": confirm_sims,
            "cap_sims": cap_sims, "target_half_width": target_half_width,
            "pilot_seed": PILOT_SEED, "confirm_seed": CONFIRM_SEED,
            "selection_seed": SELECTION_SEED,
            "allocation_seeds": list(alloc_seeds),
            "common_price": COMMON_PRICE,
            "tiers": {t: CANDIDATE_TIERS[t].to_dict() for t in TIERS},
            "calibration": calib, "power": power, "economics": econ,
            "allocation_spread": spreads,
            "runtime_s": round(time.perf_counter() - t0, 1)}


def verdict(blob: Dict[str, object]) -> Tuple[str, str]:
    """GO / PARTIAL GO / NO-GO, derived from the measured rows only."""
    rows = [r for r in blob["power"] if "verdict" in r]

    def resolved(tier: str) -> int:
        return sum(1 for r in rows
                   if r["tier"] == tier and r["verdict"] != "unresolved")

    strong, elite = resolved("strong_starter"), resolved("elite")
    weak = resolved("bench") + resolved("marginal_starter")
    spreads = blob["allocation_spread"]
    # Instability that matters is spread relative to the EFFECT, not relative
    # to season noise. A 0.5 effect with a 0.03 allocation spread is stable.
    big_unstable = [t for t in ("strong_starter", "elite")
                    if spreads.get(t, {}).get("dominates_effect")
                    or not spreads.get(t, {}).get("sign_stable", True)]
    if strong + elite == 0:
        return ("NO-GO", "neither strong-starter nor elite effects resolved at "
                         "the confirmatory sample size")
    if big_unstable:
        return ("NO-GO", f"allocation instability dominates the CE effect for "
                         f"{', '.join(big_unstable)}")
    if weak == 0:
        return ("PARTIAL GO",
                "strong and elite effects resolve; bench and marginal do not. "
                "CE-audit high-impact players, proxy-price the rest, and label "
                "the fringe unresolved.")
    return ("GO", "effects resolve across tiers at a practical sample size")


def main(argv=None) -> int:
    argv = list(sys.argv if argv is None else argv)
    sims = int(argv[1]) if len(argv) > 1 else 4000
    out = argv[2] if len(argv) > 2 else None
    blob = run(pilot_sims=max(500, sims // 4), confirm_sims=sims)
    v, why = verdict(blob)
    blob["verdict"], blob["verdict_reason"] = v, why
    print(f"\nVERDICT: {v} -- {why}")
    if out:
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(blob, fh, indent=2)
        print("wrote", out)
    print("total %.0fs" % blob["runtime_s"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
