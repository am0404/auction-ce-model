"""Joint-allocation conservation. Fabricated data only; no real player.

The layer these tests guard exists because the previous fix traded one defect
for another: giving the focus team a shadow ledger stopped rivals drafting
against an empty seat, and started letting us hold twelve players, buy six, pay
for six, and deny the other six to eleven opponents for nothing.

Everything here is fast except the two CE tests, which are marked ``slow``.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from ceauction.auction.completion import CompletionSettings, complete_roster
from ceauction.auction.proxy import ProxyEvaluator
from ceauction.tactical import (BoardSettings, build_tactical_demo,
                                cast_from_board, continue_shared_board,
                                demo_sales)
from ceauction.tactical.joint import (ConservationError, JointComparison,
                                      audit_shadow_holds, build_joint_world,
                                      build_joint_worlds, evaluate_joint_arm,
                                      format_conservation, format_shadow_audit,
                                      validate_joint_world)

_CS = CompletionSettings(beam_width=32, candidate_pool=30, proxy_candidates=32,
                         finalists=3, max_candidates=160, proxy_reps=16)


@pytest.fixture(scope="module")
def demo():
    d = build_tactical_demo()
    return d.with_market(d.market.observe_all(demo_sales(d, 12)))


@pytest.fixture(scope="module")
def px(demo):
    return ProxyEvaluator(demo.state.pool, demo.state.settings, 16, 7)


def _cid(d):
    return d.candidate().player_id


def _worlds(d, px, state=None, **kw):
    return build_joint_worlds(
        state if state is not None else d.state, d.cast, d.costs,
        completion=_CS, market=d.market, key_by_id=d.key_by_id, proxy=px,
        default_cost=1, **kw)


# ---------------------------------------------------------------------------
# The defect, as an explicit inventory
# ---------------------------------------------------------------------------


def test_the_shadow_ledger_inventory_is_explicit(demo, px):
    a = audit_shadow_holds(demo.state, demo.cast, demo.costs, completion=_CS,
                           market=demo.market, key_by_id=demo.key_by_id,
                           proxy=px, protect=(_cid(demo),))
    assert a.shadow_bid, "the audit must name every shadow-won player"
    assert set(a.selected_held) | set(a.unselected_held) == set(a.shadow_bid)
    assert not (set(a.selected_held) & set(a.unselected_held))
    assert all(p in a.shadow_prices for p in a.shadow_bid)
    assert a.shadow_budget_after == a.shadow_budget_before - a.shadow_cost_total
    assert a.shadow_slots_after == a.shadow_slots_before - len(a.shadow_bid)
    assert "SHADOW-HOLD AUDIT" in format_shadow_audit(a)


def test_free_blocking_existed_and_the_audit_names_it(demo, px):
    """The defect this branch exists to remove, pinned as a fact."""
    a = audit_shadow_holds(demo.state, demo.cast, demo.costs, completion=_CS,
                           market=demo.market, key_by_id=demo.key_by_id,
                           proxy=px, protect=(_cid(demo),),
                           rivals_recompleted=False)
    assert a.unselected_held, "some holds went unbought"
    assert a.free_blocks == a.unselected_held
    assert not a.clean
    assert a.free_block_cost >= 0


def test_recompletion_removes_every_free_block(demo, px):
    a = audit_shadow_holds(demo.state, demo.cast, demo.costs, completion=_CS,
                           market=demo.market, key_by_id=demo.key_by_id,
                           proxy=px, protect=(_cid(demo),),
                           rivals_recompleted=True)
    assert a.free_blocks == ()
    assert a.clean


# ---------------------------------------------------------------------------
# Conservation
# ---------------------------------------------------------------------------


def test_every_held_player_gets_a_final_disposition(demo, px):
    cid = _cid(demo)
    worlds = _worlds(demo, px, protect=(cid,))
    for w in worlds:
        final = w.state
        bought = set(w.focus_prices)
        for pid in w.returned_to_board:
            owned = final.owner_of.get(pid)
            assert (pid in bought or owned is not None
                    or final.is_available(pid)), \
                f"held player {pid} has no final disposition"


def test_unselected_holds_return_and_rivals_can_take_them(demo, px):
    cid = _cid(demo)
    worlds = _worlds(demo, px, protect=(cid,))
    assert any(w.returned_to_board for w in worlds), \
        "the fabricated board must exercise the return path"
    assert any(w.reclaimed_by_rivals for w in worlds), \
        "a returned hold must be reachable by a rival, not merely 'available'"
    for w in worlds:
        focus = w.state.focus_owner_id
        for pid in w.reclaimed_by_rivals:
            owner = w.state.owner_of[pid]
            assert owner != focus
            assert pid not in w.focus_prices, "we did not pay for it"


def test_we_cannot_block_a_player_without_paying_and_rostering_him(demo, px):
    cid = _cid(demo)
    for w in _worlds(demo, px, protect=(cid,)):
        assert w.conservation.unpaid_reservations == ()
        assert w.conservation.ok


def test_shadow_cost_and_actual_cost_cannot_silently_diverge(demo, px):
    """Divergence is allowed; hiding it is not."""
    a = audit_shadow_holds(demo.state, demo.cast, demo.costs, completion=_CS,
                           market=demo.market, key_by_id=demo.key_by_id,
                           proxy=px, protect=(_cid(demo),))
    assert a.cost_divergence == a.shadow_cost_total - a.actual_focus_cost
    assert "cost_divergence" in a.to_dict()
    for w in _worlds(demo, px, protect=(_cid(demo),)):
        assert w.focus_cost == sum(w.focus_prices.values())


def test_player_dollar_and_slot_conservation(demo, px):
    cid = _cid(demo)
    for w in _worlds(demo, px, protect=(cid,)):
        r = w.conservation
        assert r.pool_balances
        assert r.dollars_balance
        assert r.dollars_paid + r.dollars_remaining == r.dollars_started == 2400
        assert r.reserve_violations == ()
        assert r.over_budget == ()
        assert "CONSERVATION REPORT" in format_conservation(r)


def test_no_duplicate_ownership_and_all_rosters_legal(demo, px):
    for w in _worlds(demo, px, protect=(_cid(demo),)):
        assert w.conservation.duplicate_owners == ()
        assert w.conservation.illegal_rosters == ()
        assert w.conservation.wrong_size == ()
        w.state.validate()


def test_the_branch_candidate_stays_owned_by_the_branch_owner(demo, px):
    cid = _cid(demo)
    focus = demo.state.focus_owner_id
    buy = demo.state.apply_purchase(cid, focus, 13)
    for w in _worlds(demo, px, state=buy, branch_acquired=frozenset({cid})):
        assert w.state.owner_of[cid] == focus
        assert w.conservation.n_branch_acquired == 1
        assert w.conservation.pool_balances

    rival = "Owner04"
    pas = demo.state.award_to_rival(cid, rival, 22)
    for w in _worlds(demo, px, state=pas, branch_acquired=frozenset({cid})):
        assert w.state.owner_of[cid] == rival
        assert cid not in w.focus_roster


def test_a_declared_withdrawal_is_labelled_not_hidden(demo, px):
    """The `unavailable` branch is not conservation-neutral and must say so."""
    cid = _cid(demo)
    worlds = _worlds(demo, px, state=demo.state.withdraw(cid),
                     declared_withdrawn=frozenset({cid}))
    for w in worlds:
        assert w.conservation.declared_withdrawn == (cid,)
        assert w.conservation.ok, "declared, so accepted"
        assert w.conservation.unpaid_reservations == ()
        assert "NOT conservation-neutral" in format_conservation(w.conservation)
    # Undeclared, the same withdrawal is refused as an unpaid reservation.
    with pytest.raises(ConservationError, match="no finalist"):
        _worlds(demo, px, state=demo.state.withdraw(cid))


def test_the_validator_refuses_an_unreconciled_world(demo, px):
    cid = _cid(demo)
    w = _worlds(demo, px, protect=(cid,))[0]
    from dataclasses import replace as dc_replace
    broken = dc_replace(w, conservation=dc_replace(
        w.conservation, unpaid_reservations=(999,)))
    assert not broken.conservation.ok
    with pytest.raises(ConservationError, match="UNPAID RESERVATIONS"):
        validate_joint_world(broken)


# ---------------------------------------------------------------------------
# Conditional reallocation
# ---------------------------------------------------------------------------


def test_rival_completion_depends_on_the_selected_focus_roster(demo, px):
    cid = _cid(demo)
    worlds = _worlds(demo, px, protect=(cid,))
    assert len(worlds) >= 2, "need two finalists to compare"
    rosters = {tuple(sorted(w.focus_roster)) for w in worlds}
    assert len(rosters) >= 2, "finalists must differ"
    rival_sets = {tuple(sorted(pid for o in w.state.owners
                               if o.owner_id != w.state.focus_owner_id
                               for pid in o.player_ids)) for w in worlds}
    assert len(rival_sets) >= 2, \
        "different focus rosters must leave the rivals different boards"


def test_different_focus_completions_give_different_joint_fingerprints(demo, px):
    worlds = _worlds(demo, px, protect=(_cid(demo),))
    fps = {w.fingerprint() for w in worlds}
    assert len(fps) == len(worlds)


def test_an_unaffordable_finalist_is_refused_not_evaluated(demo, px):
    """A roster we cannot pay for is not a roster."""
    cid = _cid(demo)
    board = continue_shared_board(demo.state, costs=demo.costs,
                                  market=demo.market,
                                  key_by_id=demo.key_by_id, protect=(cid,))
    cast = cast_from_board(board, demo.cast, demo.state)
    res = complete_roster(board.state, cast, demo.costs, settings=_CS,
                          owner_id=demo.state.focus_owner_id,
                          evaluate_ce=False, default_cost=1, proxy=px,
                          reserved_ids=board.reserved_ids(
                              exclude_owner=demo.state.focus_owner_id))
    poor = demo.costs.with_costs({pid: 199 for pid in res.best.roster})
    with pytest.raises(ConservationError, match="cannot actually be bought"):
        build_joint_world(demo.state, demo.cast, poor, res.best,
                          market=demo.market, key_by_id=demo.key_by_id)


def test_the_focus_silent_regression_stays_fixed(demo, px):
    """Rivals must still have to outbid us in the shadow pass."""
    assert BoardSettings().focus_bids is True
    b = continue_shared_board(demo.state, costs=demo.costs, market=demo.market,
                              key_by_id=demo.key_by_id)
    assert b.held_for_focus


# ---------------------------------------------------------------------------
# CE over reconciled worlds
# ---------------------------------------------------------------------------


_SIMS = 400


def _arm(d, px, state=None, **kw):
    return evaluate_joint_arm(
        _worlds(d, px, state=state, **kw),
        focus_team_index=d.cast.focus_team_index, proxy=px,
        selection_sims=_SIMS, selection_seed=_CS.selection_seed,
        holdout_sims=_SIMS, holdout_seed=_CS.evaluation_seed)


@pytest.mark.slow
def test_league_equity_sums_to_one_and_selection_uses_a_holdout(demo, px):
    cid = _cid(demo)
    focus = demo.state.focus_owner_id
    arm = _arm(demo, px, state=demo.state.apply_purchase(cid, focus, 13),
               branch_acquired=frozenset({cid}))
    assert abs(arm.league_ce_sum - 1.0) < 1e-9
    assert len(arm.league_ce) == 12
    assert arm.n_worlds_compared >= 2
    # Selection and reporting must not be the same sample.
    assert _CS.selection_seed != _CS.evaluation_seed
    assert arm.selection_ce is not None and arm.ce is not None
    assert 1 <= arm.rank <= 12


@pytest.mark.slow
def test_identical_allocation_gives_identical_ce_and_price_alone_cannot_move_it(
        demo, px):
    """CE may only move when the joint allocation moves."""
    cid = _cid(demo)
    focus = demo.state.focus_owner_id
    a = _arm(demo, px, state=demo.state.apply_purchase(cid, focus, 13),
             branch_acquired=frozenset({cid}))
    b = _arm(demo, px, state=demo.state.apply_purchase(cid, focus, 13),
             branch_acquired=frozenset({cid}))
    assert a.world.fingerprint() == b.world.fingerprint()
    assert a.ce == b.ce, "same allocation, same seed, same equity"

    rival = "Owner04"
    p1 = _arm(demo, px, state=demo.state.award_to_rival(cid, rival, 22),
              branch_acquired=frozenset({cid}))
    cmp_ = JointComparison(buy=a, pass_arm=p1, price=13, recipient=rival,
                           recipient_price=22)
    assert not cmp_.allocations_identical
    assert cmp_.delta_se >= 0.0
    lo, hi = cmp_.ci95
    assert lo <= cmp_.delta_ce <= hi
    assert cmp_.verdict in ("favorable", "unfavorable", "unresolved")
    # No premium is injected anywhere: a difference of zero is allowed to occur
    # and is reported as unresolved rather than nudged.
    assert cmp_.delta_ce == a.ce - p1.ce
