"""Shared evaluation context: the frontier and decomposition cannot diverge.

The defect: at ``p=$80, q=$30`` the RB ladder reported ``+0.01298`` while the
five-branch decomposition reported ``-0.00366``. Three different completion
searches were running for one world — the ladder's buy arm used the nested
pre-pass set, the ladder's pass arm ran its own beam, and all five
decomposition branches ran another. These tests make that structurally
impossible rather than merely fixed once.
"""

from __future__ import annotations

import dataclasses
import subprocess
from pathlib import Path

import pytest

from ceauction.auction.completion import (ComparisonCast, Completion,
                                          CompletionSettings, complete_roster)
from ceauction.auction.proxy import ProxyEvaluator
from ceauction.auction.state import new_auction
from ceauction.league import DEFAULT_LEAGUE, Position
from ceauction.market.costbook import cost_book_from_prior
from ceauction.tactical.board import BoardSettings
from ceauction.tactical.decompose import BRANCHES, COMPONENT_ORDER
from ceauction.tactical.demo import build_tactical_demo
from ceauction.tactical.ensemble import balanced_schedule
from ceauction.tactical.evalcontext import (BRANCH_HOLDS_CANDIDATE,
                                            ContextMismatch, EvalContext,
                                            IndependentSearchRefused,
                                            build_eval_context)
from ceauction.tactical.midauction import (PassPrice, PassPriceMode,
                                           STATE_SPECS, build_simulated_state)
from ceauction.tactical.nested import build_nested_ladder
from ceauction.tactical.realpilot import OWNER_IDS, RealBoard
from ceauction.tactical.reconciled import (AGREEMENT_TOLERANCE,
                                           check_agreement, evaluate_branch,
                                           evaluate_draw)

_CS = CompletionSettings(beam_width=24, candidate_pool=30, proxy_candidates=24,
                         finalists=3, max_candidates=120, proxy_reps=16,
                         selection_sims=300, evaluation_sims=400,
                         selection_seed=555_000_111,
                         evaluation_seed=917_324_011, rival_selection="proxy")
_BS = BoardSettings(pool_depth=200, max_allocations=200)


@pytest.fixture(scope="module")
def board():
    d = build_tactical_demo()
    state = new_auction(list(d.state.pool), OWNER_IDS, OWNER_IDS[0],
                        settings=DEFAULT_LEAGUE)
    state.validate()
    costs = {}
    for sc in ("low", "base", "high"):
        cb = cost_book_from_prior(d.prior, sc)
        gaps = {s.player_id: 1 for s in state.pool if s.player_id not in cb}
        costs[sc] = cb.with_costs(gaps) if gaps else cb
    return RealBoard(
        state=state,
        cast=ComparisonCast(0, tuple(() for _ in OWNER_IDS), OWNER_IDS),
        prior=d.prior, market=d.market, costs=costs,
        key_by_id=dict(d.key_by_id),
        name_by_id={s.player_id: s.name for s in state.pool}, coverage={})


@pytest.fixture(scope="module")
def sim(board):
    by = {}
    for s in board.state.available_specs:
        by.setdefault(Position(int(s.position)).name, []).append(s)
    for g in by.values():
        g.sort(key=lambda s: -s.base_mean)
    prot = [by["RB"][0].player_id]
    return build_simulated_state(board, STATE_SPECS["balanced"],
                                 protect=prot), prot[0]


@pytest.fixture(scope="module")
def px(sim):
    s, _ = sim
    return ProxyEvaluator(s.state.pool, s.state.settings, 16, 7)


@pytest.fixture(scope="module")
def ctxs(board, sim, px):
    s, cid = sim
    st = s.state
    prices = [1, 20]
    lad = build_nested_ladder(st, board.cast, board.costs["base"], cid, prices,
                              board_settings=_BS, completion=_CS,
                              market=board.market, key_by_id=board.key_by_id,
                              proxy=px)
    wo = complete_roster(st, board.cast, board.costs["base"], settings=_CS,
                         owner_id=st.focus_owner_id, evaluate_ce=False,
                         default_cost=1, proxy=px,
                         reserved_ids=frozenset({cid}))
    wo_set = list(wo.finalists) or ([wo.best] if wo.best else [])
    pp = PassPrice(mode=PassPriceMode.FIXED_MARKET, increment=1, fixed_q=15)
    draws = balanced_schedule(12, 3)
    return [build_eval_context(
        st, board.cast, board.costs["base"], cid, p=20, pass_price=pp,
        recipient="Team05", q=15, draw=d, board=_BS, completion=_CS,
        market=board.market, key_by_id=board.key_by_id, proxy=px,
        with_candidate=lad.by_price[20].feasible,
        without_candidate=wo_set) for d in draws]


# ---------------------------------------------------------------------------
# Independent search is refused
# ---------------------------------------------------------------------------


def test_a_branch_may_not_search_independently(ctxs, px):
    with pytest.raises(IndependentSearchRefused, match="own completion search"):
        evaluate_branch(ctxs[0], "UP", proxy=px,
                        allow_independent_search=True)


def test_the_refusal_names_the_defect_it_prevents(ctxs, px):
    with pytest.raises(IndependentSearchRefused, match=r"p=\$80"):
        evaluate_branch(ctxs[0], "UF", proxy=px,
                        allow_independent_search=True)


def test_every_branch_draws_from_the_shared_offer(ctxs, px):
    ctx = ctxs[0]
    for branch in BRANCHES:
        offers = ctx.completions_for(branch)
        assert offers, branch
        pool = (ctx.with_candidate if BRANCH_HOLDS_CANDIDATE[branch]
                else ctx.without_candidate)
        keys = {frozenset(c.roster) for c in pool}
        for c in offers:
            assert frozenset(c.roster) in keys, \
                f"{branch} was offered a completion not in the shared set"




@pytest.mark.parametrize("branch,holds", sorted(BRANCH_HOLDS_CANDIDATE.items()))
def test_the_offer_matches_who_holds_the_candidate(ctxs, branch, holds):
    ctx = ctxs[0]
    for c in ctx.completions_for(branch):
        assert (ctx.candidate_id in c.roster) is holds, (
            f"branch {branch} offered a completion whose candidate membership "
            f"contradicts who holds him")


# ---------------------------------------------------------------------------
# Mismatched contexts are refused
# ---------------------------------------------------------------------------


def _mismatch(ctx, **kw):
    return dataclasses.replace(ctx, **kw)


def test_a_mismatched_opportunity_set_is_refused(ctxs):
    a = ctxs[0]
    b = _mismatch(a, with_candidate=a.with_candidate[:1])
    assert a.opportunity_fingerprint() != b.opportunity_fingerprint()
    with pytest.raises(ContextMismatch, match="opportunity set"):
        a.assert_matches(b)


def test_a_deliberately_altered_completion_is_detected(ctxs, board, sim):
    """Swap one roster for another; the fingerprint must move."""
    a = ctxs[0]
    if len(a.with_candidate) < 2:
        pytest.skip("need two completions to swap")
    swapped = (a.with_candidate[1],) + a.with_candidate[1:]
    b = _mismatch(a, with_candidate=swapped)
    assert a.opportunity_fingerprint() != b.opportunity_fingerprint()
    with pytest.raises(ContextMismatch):
        a.assert_matches(b)


def test_order_alone_does_not_change_the_opportunity_fingerprint(ctxs):
    a = ctxs[0]
    if len(a.with_candidate) < 2:
        pytest.skip("need two completions to reorder")
    b = _mismatch(a, with_candidate=tuple(reversed(a.with_candidate)))
    assert a.opportunity_fingerprint() == b.opportunity_fingerprint(), \
        "the same offer discovered in a different order is the same offer"


@pytest.mark.parametrize("field,value", [
    ("p", 21), ("q", 16), ("recipient", "Team06"), ("scenario", "high"),
    ("selection_seed", 1), ("holdout_seed", 2), ("selection_sims", 1),
    ("holdout_sims", 3), ("default_cost", 2), ("max_worlds", 3),
])
def test_every_material_input_is_fingerprinted(ctxs, field, value):
    a = ctxs[0]
    b = _mismatch(a, **{field: value})
    assert a.fingerprint() != b.fingerprint(), field
    with pytest.raises(ContextMismatch):
        a.assert_matches(b)


def test_a_different_pass_rule_is_refused(ctxs):
    a = ctxs[0]
    b = _mismatch(a, pass_price=PassPrice(mode=PassPriceMode.RIVAL_OUTBIDS,
                                          increment=1))
    assert a.fingerprint() != b.fingerprint()
    with pytest.raises(ContextMismatch, match="pass_rule"):
        a.assert_matches(b)


def test_a_different_allocation_draw_is_refused(ctxs):
    with pytest.raises(ContextMismatch, match="allocation draw"):
        ctxs[0].assert_matches(ctxs[1])


def test_an_ensemble_with_mixed_prices_is_refused(ctxs, px):
    bad = list(ctxs[:2]) + [_mismatch(ctxs[2], p=99)]
    with pytest.raises(ContextMismatch, match="disagree on p"):
        check_agreement(bad, proxy=px)


def test_an_ensemble_with_mixed_offers_is_refused(ctxs, px):
    bad = list(ctxs[:2]) + [_mismatch(ctxs[2],
                                      with_candidate=ctxs[2].with_candidate[:1])]
    with pytest.raises(ContextMismatch, match="different opportunity sets"):
        check_agreement(bad, proxy=px)


def test_an_unknown_branch_is_refused(ctxs):
    with pytest.raises(ContextMismatch, match="unknown branch"):
        ctxs[0].completions_for("XX")


# ---------------------------------------------------------------------------
# Per-draw agreement and telescoping
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def draw_eval(ctxs, px):
    return evaluate_draw(ctxs[0], proxy=px)


def test_the_two_paths_agree_per_draw_not_after_averaging(draw_eval):
    assert abs(draw_eval.residual) <= AGREEMENT_TOLERANCE
    assert draw_eval.agrees
    assert draw_eval.frontier_delta == pytest.approx(
        draw_eval.decomposition_total, abs=AGREEMENT_TOLERANCE)


def test_the_telescoping_identity_holds_exactly(draw_eval):
    b = draw_eval.branches
    comps = draw_eval.components
    assert comps["our_payment"] == pytest.approx(b["UP"].ce - b["UF"].ce,
                                                 abs=1e-15)
    assert comps["our_possession"] == pytest.approx(b["UF"].ce - b["W"].ce,
                                                    abs=1e-15)
    assert comps["rival_denial"] == pytest.approx(b["W"].ce - b["RF"].ce,
                                                  abs=1e-15)
    assert comps["rival_payment"] == pytest.approx(b["RF"].ce - b["RP"].ce,
                                                   abs=1e-15)
    assert sum(comps.values()) == pytest.approx(
        b["UP"].ce - b["RP"].ce, abs=AGREEMENT_TOLERANCE)


def test_the_frontier_delta_is_exactly_up_minus_rp(draw_eval):
    b = draw_eval.branches
    assert draw_eval.frontier_delta == b["UP"].ce - b["RP"].ce


def test_every_draw_agrees_across_the_ensemble(ctxs, px):
    rep = check_agreement(ctxs, proxy=px)
    assert rep.all_agree
    assert rep.max_residual <= AGREEMENT_TOLERANCE
    blob = rep.to_dict(include_draws=False)
    assert blob["all_draws_agree"] is True
    assert blob["agreement_tolerance"] == AGREEMENT_TOLERANCE
    assert "never only after averaging" in blob["note"]


def test_per_draw_world_identity_is_recorded(draw_eval):
    for name, ev in draw_eval.branches.items():
        assert ev.focus_fingerprint and ev.joint_fingerprint
        assert ev.alloc_fingerprint
        assert ev.rival_rosters
        assert ev.n_offered > 0
        blob = ev.to_dict()
        for k in ("focus_fingerprint", "joint_fingerprint",
                  "alloc_fingerprint", "dollars_spent", "pool_remaining"):
            assert k in blob


# ---------------------------------------------------------------------------
# Invariants preserved
# ---------------------------------------------------------------------------


def test_conservation_and_league_sums_hold_in_every_branch(draw_eval):
    for name, ev in draw_eval.branches.items():
        assert ev.conservation_ok, name
        assert ev.league_ce_sum == pytest.approx(1.0, abs=1e-9), name


def test_no_duplicate_ownership_in_any_branch(draw_eval):
    for name, ev in draw_eval.branches.items():
        seen = set(ev.focus_roster)
        for owner, roster in ev.rival_rosters:
            for pid in roster:
                assert pid not in seen, f"{name}: {pid} on two rosters"
                seen.add(pid)


def test_budget_and_reserve_stay_legal(ctxs):
    """Every branch's focus budget is non-negative and leaves the $1 reserve."""
    ctx = ctxs[0]
    for branch in BRANCHES:
        budget = ctx.focus_budget_for(branch)
        assert budget >= 0, branch
        offers = ctx.completions_for(branch)
        assert offers, branch


def test_paying_more_never_widens_the_offer(ctxs, board, sim, px):
    """Cross-price nesting, enforced through the shared context."""
    s, cid = sim
    st = s.state
    lad = build_nested_ladder(st, board.cast, board.costs["base"], cid,
                              [1, 20], board_settings=_BS, completion=_CS,
                              market=board.market, key_by_id=board.key_by_id,
                              proxy=px)
    assert lad.is_nested
    cheap = lad.by_price[1].feasible_keys
    dear = lad.by_price[20].feasible_keys
    assert dear <= cheap, "a dearer price offered something a cheaper one did not"


def test_an_unaffordable_branch_refuses_rather_than_researching(ctxs):
    """The failure mode must be a refusal, never a silent new beam."""
    ctx = dataclasses.replace(ctxs[0], p=10_000)
    with pytest.raises(ContextMismatch, match="independent search"):
        ctx.completions_for("UP")


# ---------------------------------------------------------------------------
# Hygiene
# ---------------------------------------------------------------------------


def test_the_context_serialization_states_its_purpose(ctxs):
    blob = ctxs[0].to_dict()
    assert "neither may search independently" in blob["note"]
    assert blob["pass_rule"] == "fixed_market"
    assert blob["candidate_price_p"] == 20 and blob["pass_price_q"] == 15


def test_no_local_data_file_is_tracked():
    out = subprocess.run(["git", "ls-files", "local_data"],
                         capture_output=True, text=True, check=True)
    assert out.stdout.strip() == ""
