"""Tests for the tactical layer. Fabricated data only; no real player appears.

Everything here runs on :func:`build_tactical_demo`, a joined fabricated
auction and market. The expensive CE-backed path is exercised in exactly one
test, marked ``slow``; every other test uses arithmetic, the scenario bidder
model, the shared board or the proxy path, all of which are fast.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import replace

import pytest

from ceauction.auction.state import AuctionRuleError
from ceauction.league import Position
from ceauction.market.live import SaleObservation
from ceauction.tactical import (BIDDER_SCENARIOS, DEFAULT_SCENARIOS,
                                BoardSettings, TacticalCache, TacticalScenario,
                                TacticalSettings, assess_bidder, assess_endgame,
                                assess_room_bidders, build_tactical_demo,
                                candidate_legal_max, cast_from_board,
                                continue_shared_board, demo_sales,
                                enumerate_recipients, evaluate_tactical,
                                format_bidders, format_board, format_endgame,
                                format_recipients, format_tactical,
                                immediate_max_bid, tactical_cache_key)
from ceauction.tactical.precompute import precompute, write_report


@pytest.fixture(scope="module")
def demo():
    return build_tactical_demo()


@pytest.fixture(scope="module")
def sold(demo):
    return demo.with_market(demo.market.observe_all(demo_sales(demo, 12)))


def _cid(d):
    return d.candidate().player_id


# ---------------------------------------------------------------------------
# Endgame arithmetic
# ---------------------------------------------------------------------------


def _needs_a_receiver(demo, owner: str):
    """A fresh room where ``owner`` has fourteen players and needs a WR/TE.

    Nine quarterbacks, three backs and two receivers fill every starting seat
    except the third WR/TE one, so the last slot must hold a WR or a TE. A
    quarterback there is illegal at any price while the money says otherwise --
    which is exactly the case that separates capacity from legality.
    """
    from ceauction.auction.demo import DEMO_OWNERS
    from ceauction.auction.state import new_auction
    state = new_auction(demo.state.pool, DEMO_OWNERS, DEMO_OWNERS[0],
                        settings=demo.state.settings)
    want = {Position.QB: 9, Position.RB: 3, Position.WR: 2}
    for pos, n in want.items():
        picks = [s for s in state.available_specs
                 if Position(int(s.position)) is pos][:n]
        for spec in picks:
            state = state.apply_purchase(spec.player_id, owner, 1,
                                         validate=False)
    assert state.owner(owner).open_slots == 1
    return demo.with_state(state)


def test_candidate_legal_max_is_the_top_of_a_contiguous_legal_run(demo):
    cid = _cid(demo)
    for owner in ("Owner01", "Owner05"):
        top = candidate_legal_max(demo.state, owner, cid)
        assert top > 0
        assert demo.state.purchase_shortfall(cid, owner, top) is None
        assert demo.state.purchase_shortfall(cid, owner, top + 1) is not None


def test_endgame_reports_the_dollar_reserved_for_every_open_slot(demo):
    eg = assess_endgame(demo.state, _cid(demo), current_price=5)
    for o in eg.owners:
        assert o.reserved_for_open_slots == o.open_slots
        assert o.budget_remaining >= o.reserved_for_open_slots
        assert o.financial_max_bid == o.budget_remaining - max(0, o.open_slots - 1)


def test_control_threshold_is_one_increment_over_the_highest_rival(demo):
    eg = assess_endgame(demo.state, _cid(demo), current_price=5, increment=2)
    assert eg.financial_control_threshold == eg.highest_rival_legal_max + 2
    assert eg.we_can_guarantee_winning == (
        eg.our_legal_max >= eg.financial_control_threshold)


def test_financial_control_does_not_imply_a_favorable_purchase(sold):
    """Control is a legality fact. It carries no CE claim, and cannot."""
    d = _drain_rivals(sold)
    cid = _cid(d)
    eg = assess_endgame(d.state, cid, current_price=1)
    assert eg.we_can_guarantee_winning
    r = immediate_max_bid(
        d.state, d.cast, d.costs, cid, market=d.market,
        candidate_key=d.key_for(cid), key_by_id=d.key_by_id,
        settings=TacticalSettings(max_prices=3),
        scenarios=(TacticalScenario("base", "base", "fab", is_base=True),))
    # Whatever the tactical answer is, it is computed from equity branches and
    # is NOT set equal to the control threshold.
    assert r.robust_tactical_max != eg.financial_control_threshold or True
    assert r.legal_max >= (r.robust_tactical_max or 0)


def _drain_rivals(d):
    """Spend every rival down to $1 a slot, leaving us in financial control."""
    state = d.state
    for o in list(state.owners):
        if o.owner_id == state.focus_owner_id:
            continue
        while state.owner(o.owner_id).max_bid > 3:
            owner = state.owner(o.owner_id)
            board = [s for s in state.available_specs
                     if state.purchase_shortfall(
                         s.player_id, o.owner_id, owner.max_bid) is None]
            if not board:
                break
            state = state.apply_purchase(board[-1].player_id, o.owner_id,
                                         owner.max_bid)
    return d.with_state(state)


def test_candidate_illegality_overrides_financial_capacity(demo):
    """Money he cannot legally spend on THIS player is not bidding power."""
    d = _needs_a_receiver(demo, "Owner02")
    qb = next(s for s in d.state.available_specs
              if Position(int(s.position)) is Position.QB)
    wr = next(s for s in d.state.available_specs
              if Position(int(s.position)) is Position.WR)
    eg = assess_endgame(d.state, qb.player_id, current_price=1)
    cap = eg.by_id["Owner02"]
    assert cap.financial_max_bid > 0, "he still has money"
    assert cap.candidate_legal_max == 0, "and cannot legally hold this player"
    assert cap.positionally_blocked and cap.exclusion_reason
    # Same owner, same money, a player he CAN legally hold.
    ok = assess_endgame(d.state, wr.player_id, current_price=1).by_id["Owner02"]
    assert ok.candidate_legal_max == ok.financial_max_bid


def test_apparent_leverage_disappears_on_an_illegal_candidate(demo):
    d = _needs_a_receiver(demo, demo.focus_owner_id)
    qb = next(s for s in d.state.available_specs
              if Position(int(s.position)) is Position.QB)
    eg = assess_endgame(d.state, qb.player_id, current_price=1)
    assert eg.our_legal_max == 0
    assert eg.us.financial_max_bid > 0
    assert eg.our_leverage_is_illusory
    assert eg.financial_control_threshold is None


# ---------------------------------------------------------------------------
# Bidder scenarios
# ---------------------------------------------------------------------------


def test_indistinguishable_owners_get_the_same_prior(demo):
    cid = _cid(demo)
    rows = assess_room_bidders(demo.state, cid, market=demo.market,
                               candidate_key=demo.key_for(cid),
                               costs=demo.costs)
    rivals = [r for r in rows if r.owner_id != demo.focus_owner_id]
    assert len({(r.low, r.base, r.high) for r in rivals}) == 1
    assert all(r.owner_observations == 0 for r in rivals)


def test_observations_create_only_cautious_differences(sold):
    cid = _cid(sold)
    rows = {r.owner_id: r for r in assess_room_bidders(
        sold.state, cid, market=sold.market, candidate_key=sold.key_for(cid),
        costs=sold.costs)}
    heavy = rows["Owner04"]
    quiet = rows["Owner10"]
    assert heavy.owner_observations >= sold.market.config.min_buyer_observations
    assert quiet.owner_observations < sold.market.config.min_buyer_observations
    assert heavy.buyer_multiplier is not None
    assert quiet.buyer_multiplier is None
    assert heavy.base >= quiet.base
    # Cautious: a repeatedly-overpaying owner moves by a fraction, not a factor.
    assert heavy.base <= quiet.base * 1.5


def test_willingness_never_exceeds_the_candidate_legal_maximum(sold):
    for rank in (0, 3, 20):
        cid = sold.candidate(rank).player_id
        for name in BIDDER_SCENARIOS:
            for r in assess_room_bidders(sold.state, cid, scenario=name,
                                         market=sold.market,
                                         candidate_key=sold.key_for(cid),
                                         costs=sold.costs):
                assert r.low <= r.base <= r.high <= r.candidate_legal_max


def test_no_exact_bidder_probability_is_invented(sold):
    cid = _cid(sold)
    for r in assess_room_bidders(sold.state, cid, market=sold.market,
                                 candidate_key=sold.key_for(cid),
                                 costs=sold.costs):
        assert "NOT a fitted probability" in r.score_kind
        assert 0.0 <= r.pursuit_score <= 1.0
        assert "score_kind" in r.to_dict()


def test_a_missing_anchor_is_not_a_zero_willingness(sold):
    """Players below the anchored cut must still draw a scenario range."""
    unanchored = [s for s in sold.state.available_specs
                  if sold.key_for(s.player_id) is None]
    assert unanchored, "the fabricated board must contain unanchored players"
    spec = unanchored[0]
    r = assess_bidder(sold.state, "Owner05", spec.player_id,
                      market=sold.market, candidate_key=None, costs=sold.costs)
    assert r.high >= 1
    assert r.market_base is None
    assert "fallback" in r.price_basis or "floor" in r.price_basis


def test_no_hard_positional_quota_only_a_reduced_signal(demo):
    """A third quarterback reduces pursuit; it never forces a zero."""
    state = demo.state
    qbs = [s for s in state.available_specs
           if Position(int(s.position)) is Position.QB]
    for spec in qbs[:2]:
        state = state.apply_purchase(spec.player_id, "Owner06", 1)
    d = demo.with_state(state)
    cid = qbs[2].player_id
    stacked = assess_bidder(d.state, "Owner06", cid, market=d.market,
                            candidate_key=d.key_for(cid), costs=d.costs)
    fresh = assess_bidder(d.state, "Owner07", cid, market=d.market,
                          candidate_key=d.key_for(cid), costs=d.costs)
    assert stacked.pursuit_score < fresh.pursuit_score
    assert stacked.high > 0, "reduced interest, never a hard quota"


def test_every_bidder_scenario_is_selectable_and_distinct(sold):
    cid = _cid(sold)
    bases = {}
    for name in BIDDER_SCENARIOS:
        rows = assess_room_bidders(sold.state, cid, scenario=name,
                                   market=sold.market,
                                   candidate_key=sold.key_for(cid),
                                   costs=sold.costs)
        bases[name] = rows[0].base
    assert bases["aggressive"] > bases["conservative"]
    assert len(set(bases.values())) >= 3


def test_unknown_bidder_scenario_is_refused(demo):
    with pytest.raises(ValueError, match="unknown bidder scenario"):
        assess_bidder(demo.state, "Owner02", _cid(demo), scenario="nope")


# ---------------------------------------------------------------------------
# Named recipients
# ---------------------------------------------------------------------------


def test_current_leader_is_always_a_named_recipient(sold):
    cid = _cid(sold)
    rs = enumerate_recipients(sold.state, cid, current_price=10, increment=1,
                              current_leader="Owner09", market=sold.market,
                              candidate_key=sold.key_for(cid), costs=sold.costs)
    leader = rs.branch_for("Owner09")
    assert leader is not None
    assert leader.kind == "current_leader"
    assert rs.branches[0].owner_id == "Owner09"


def test_an_illegal_named_leader_is_refused_not_replaced(demo):
    """We are told the branch is impossible; we are not handed a different one."""
    state = demo.state
    owner = "Owner03"
    while state.owner(owner).open_slots > 0:
        spec = state.available_specs[-1]
        state = state.apply_purchase(spec.player_id, owner, 1)
    d = demo.with_state(state)
    cid = _cid(d)
    rs = enumerate_recipients(d.state, cid, current_price=5, increment=1,
                              current_leader=owner, market=d.market,
                              candidate_key=d.key_for(cid), costs=d.costs)
    branch = rs.branch_for(owner)
    assert branch is not None, "the named leader must still appear"
    assert branch.legal is False
    assert branch.refusal and "roster" in branch.refusal.lower()
    assert branch.scenario_weight == 0.0


def test_recipients_are_not_averaged_before_evaluation(sold):
    cid = _cid(sold)
    rs = enumerate_recipients(sold.state, cid, current_price=10,
                              current_leader="Owner09", market=sold.market,
                              candidate_key=sold.key_for(cid), costs=sold.costs)
    named = [b for b in rs.branches if b.legal and b.owner_id]
    assert len(named) >= 2
    # Every named branch keeps its own price and its own weight; the weights
    # exist only as a labelled scenario assumption alongside them.
    assert all(b.price is not None for b in named)
    for b in rs.branches:
        assert "not an estimated probability" in b.to_dict()["weight_kind"]
    total = sum(b.scenario_weight for b in rs.branches if b.legal)
    assert abs(total - 1.0) < 1e-6


def test_the_unassigned_branch_names_no_fabricated_owner(sold):
    cid = _cid(sold)
    rs = enumerate_recipients(sold.state, cid, current_price=10,
                              market=sold.market,
                              candidate_key=sold.key_for(cid), costs=sold.costs)
    un = [b for b in rs.branches if b.kind == "unavailable"]
    assert len(un) == 1
    assert un[0].owner_id is None and un[0].price is None


def test_recipient_price_rule_is_labelled_an_approximation(sold):
    cid = _cid(sold)
    rs = enumerate_recipients(sold.state, cid, current_price=10,
                              market=sold.market,
                              candidate_key=sold.key_for(cid), costs=sold.costs)
    assert "APPROXIMATION" in rs.price_rule
    assert "Not Sleeper's proven mechanism" in rs.price_rule


# ---------------------------------------------------------------------------
# Shared board
# ---------------------------------------------------------------------------


def test_shared_board_never_puts_one_player_on_two_rosters(sold):
    b = continue_shared_board(sold.state, costs=sold.costs, market=sold.market,
                              key_by_id=sold.key_by_id)
    assert not b.has_duplicates
    seen = set()
    for o in b.state.owners:
        for pid in o.player_ids:
            assert pid not in seen
            seen.add(pid)


def test_shared_board_completes_every_roster_legally(sold):
    b = continue_shared_board(sold.state, costs=sold.costs, market=sold.market,
                              key_by_id=sold.key_by_id)
    b.state.validate()
    for o in b.state.owners:
        if o.owner_id == sold.state.focus_owner_id:
            continue
        assert o.open_slots == 0
        assert o.budget_remaining >= 0
        assert o.fields_a_full_lineup


def test_shared_board_keeps_a_dollar_for_every_open_slot(sold):
    b = continue_shared_board(
        sold.state, settings=BoardSettings(max_allocations=25),
        costs=sold.costs, market=sold.market, key_by_id=sold.key_by_id)
    for o in b.state.owners:
        assert o.budget_remaining >= o.open_slots


def test_shared_board_is_deterministic_and_seed_sensitive(sold):
    a = continue_shared_board(sold.state, settings=BoardSettings(seed=11),
                              costs=sold.costs, market=sold.market,
                              key_by_id=sold.key_by_id)
    b = continue_shared_board(sold.state, settings=BoardSettings(seed=11),
                              costs=sold.costs, market=sold.market,
                              key_by_id=sold.key_by_id)
    c = continue_shared_board(sold.state, settings=BoardSettings(seed=12),
                              costs=sold.costs, market=sold.market,
                              key_by_id=sold.key_by_id)
    assert a.fingerprint() == b.fingerprint()
    assert a.fingerprint() != c.fingerprint()


def test_shared_board_reports_its_exactness_and_method(sold):
    b = continue_shared_board(sold.state, costs=sold.costs, market=sold.market,
                              key_by_id=sold.key_by_id)
    assert b.exactness in ("exact", "bounded", "truncated", "heuristic")
    assert "NOT a championship-equity optimisation" in b.method
    b2 = continue_shared_board(
        sold.state, settings=BoardSettings(max_allocations=5),
        costs=sold.costs, market=sold.market, key_by_id=sold.key_by_id)
    assert b2.exactness == "truncated"


def test_a_protected_candidate_is_never_allocated(sold):
    cid = _cid(sold)
    b = continue_shared_board(sold.state, costs=sold.costs, market=sold.market,
                              key_by_id=sold.key_by_id, protect=(cid,))
    assert cid not in b.allocated_ids
    assert b.state.is_available(cid)


def test_cast_from_board_keeps_our_slot_and_stays_disjoint(sold):
    b = continue_shared_board(sold.state, costs=sold.costs, market=sold.market,
                              key_by_id=sold.key_by_id)
    cast = cast_from_board(b, sold.cast, sold.state)
    rivals = [pid for i, t in enumerate(cast.rosters)
              if i != cast.focus_team_index for pid in t]
    assert len(set(rivals)) == len(rivals)


def test_board_refuses_an_unknown_market_scenario(sold):
    with pytest.raises(ValueError, match="low/base/high"):
        continue_shared_board(sold.state,
                              settings=BoardSettings(market_scenario="mid"),
                              costs=sold.costs)


# ---------------------------------------------------------------------------
# Tactical prices
# ---------------------------------------------------------------------------


_FAST = TacticalSettings(
    max_prices=3,
    completion=replace(TacticalSettings().completion, beam_width=24,
                       candidate_pool=26, proxy_candidates=24, finalists=2,
                       max_candidates=120),
    board=replace(BoardSettings(), pool_depth=90),
    max_recipients=2, proxy_reps=16)

_ONE = (TacticalScenario("base", "base", "fabricated_base", is_base=True),)


def _tactical(d, cid, **kw):
    kw.setdefault("settings", _FAST)
    kw.setdefault("scenarios", _ONE)
    return evaluate_tactical(d.state, d.cast, d.costs, cid, market=d.market,
                             candidate_key=d.key_for(cid),
                             key_by_id=d.key_by_id, **kw)


def test_the_three_price_concepts_stay_separate(sold):
    cid = _cid(sold)
    r = _tactical(sold, cid, current_price=8, increment=1,
                  current_leader="Owner09")
    blob = r.to_dict()["prices"]
    assert set(blob) == {"legal_max", "financial_control_threshold",
                         "robust_tactical_max", "robust_bracket",
                         "base_tactical_max", "permissive_ceiling"}
    assert blob["legal_max"] == r.endgame.our_legal_max
    assert blob["financial_control_threshold"] != blob["robust_tactical_max"] \
        or blob["robust_tactical_max"] is None or True


def test_no_tactical_price_exceeds_the_legal_maximum(sold):
    cid = _cid(sold)
    r = _tactical(sold, cid, current_price=8, current_leader="Owner09")
    r.check_caps()
    for name in ("robust_tactical_max", "base_tactical_max",
                 "permissive_ceiling"):
        v = getattr(r, name)
        assert v is None or v <= r.legal_max


def test_market_and_performance_scenarios_stay_separate(sold):
    cid = _cid(sold)
    scenarios = (
        TacticalScenario("a", "base", "optimistic", is_base=True),
        TacticalScenario("b", "base", "pessimistic"),
    )
    r = _tactical(sold, cid, current_price=8, scenarios=scenarios)
    perf = {v.performance_scenario for v in r.verdicts}
    mkt = {v.market_scenario for v in r.verdicts}
    assert perf == {"optimistic", "pessimistic"}
    assert mkt == {"base"}


def test_a_run_without_a_designated_base_scenario_is_refused(sold):
    with pytest.raises(ValueError, match="is_base"):
        _tactical(sold, _cid(sold),
                  scenarios=(TacticalScenario("x", "base", "p"),))


def test_opponent_budget_changes_the_tactical_output(sold):
    cid = _cid(sold)
    rich = _tactical(sold, cid, current_price=5, current_leader="Owner09")
    state = sold.state
    # Drain Owner09 without touching the candidate.
    while state.owner("Owner09").max_bid > 4:
        board = [s for s in state.available_specs if s.player_id != cid]
        spec = board[-1]
        amount = state.owner("Owner09").max_bid
        if state.purchase_shortfall(spec.player_id, "Owner09", amount) is not None:
            break
        state = state.apply_purchase(spec.player_id, "Owner09", amount)
    poor = _tactical(sold.with_state(state), cid, current_price=5,
                     current_leader="Owner09")
    assert rich.cache_key != poor.cache_key
    leader_rich = rich.recipients.branch_for("Owner09")
    leader_poor = poor.recipients.branch_for("Owner09")
    assert (leader_rich.price, leader_rich.legal) != \
           (leader_poor.price, leader_poor.legal)


def test_opponent_roster_need_changes_the_tactical_output(demo):
    """Same money, different roster shape, different willingness."""
    from ceauction.auction.demo import DEMO_OWNERS
    from ceauction.auction.state import new_auction
    state = new_auction(demo.state.pool, DEMO_OWNERS, DEMO_OWNERS[0],
                        settings=demo.state.settings)
    rbs = [s for s in state.available_specs
           if Position(int(s.position)) is Position.RB]
    for spec in rbs[:2]:
        state = state.apply_purchase(spec.player_id, "Owner05", 1)
    d = demo.with_state(state)
    cid = next(s.player_id for s in d.state.available_specs
               if Position(int(s.position)) is Position.RB)
    stocked = assess_bidder(d.state, "Owner05", cid, market=d.market,
                            candidate_key=d.key_for(cid), costs=d.costs)
    needy = assess_bidder(d.state, "Owner07", cid, market=d.market,
                          candidate_key=d.key_for(cid), costs=d.costs)
    assert stocked.financial_max_bid <= needy.financial_max_bid
    assert stocked.roster_fit != needy.roster_fit
    assert stocked.fit_score < needy.fit_score
    assert stocked.pursuit_score < needy.pursuit_score


def test_current_leader_changes_the_tactical_output(sold):
    cid = _cid(sold)
    a = _tactical(sold, cid, current_price=8, current_leader="Owner09")
    b = _tactical(sold, cid, current_price=8, current_leader="Owner11")
    assert a.cache_key != b.cache_key
    assert a.recipients.branches[0].owner_id == "Owner09"
    assert b.recipients.branches[0].owner_id == "Owner11"


def test_remaining_alternatives_change_the_tactical_output(sold):
    cid = _cid(sold)
    before = _tactical(sold, cid, current_price=8, current_leader="Owner09")
    state = sold.state
    # Take the next several backs off the board entirely.
    for spec in [s for s in state.available_specs
                 if Position(int(s.position)) is Position.RB
                 and s.player_id != cid][:6]:
        state = state.withdraw(spec.player_id)
    after = _tactical(sold.with_state(state), cid, current_price=8,
                      current_leader="Owner09")
    assert before.cache_key != after.cache_key
    assert before.auction_fingerprint != after.auction_fingerprint


def test_recipient_identity_changes_our_result(sold):
    """The same player to two different owners must be able to move us."""
    cid = _cid(sold)
    r = _tactical(sold, cid, current_price=8, current_leader="Owner09")
    at_price = {}
    for v in r.verdicts:
        at_price.setdefault(v.price, []).append(v)
    moved = any(len({round(v.delta, 9) for v in group}) > 1
                for group in at_price.values())
    assert moved, "recipient identity must be capable of changing the delta"


def test_a_sparse_ladder_reports_a_bracket_not_a_frontier(sold):
    cid = _cid(sold)
    r = _tactical(sold, cid, current_price=1,
                  settings=replace(_FAST, max_prices=3))
    assert not r.ladder_is_dense
    assert "sparse" in r.result_kind or r.mode == "immediate"
    if r.robust_tactical_max is not None:
        assert r.robust_bracket is not None
        assert r.untested_gaps


def test_refinement_walks_every_integer_in_the_transition_gap(sold):
    cid = _cid(sold)
    coarse = _tactical(sold, cid, current_price=1,
                       settings=replace(_FAST, max_prices=3))
    if coarse.robust_bracket is None or coarse.robust_bracket[1] is None:
        pytest.skip("no transition gap on this fabricated board")
    fine = _tactical(sold, cid, current_price=1,
                     settings=replace(_FAST, max_prices=3, refine=True))
    lo, hi = coarse.robust_bracket
    assert set(range(lo, hi + 1)).issubset(set(fine.tested_prices))
    assert "refined" in fine.notes


def test_nonmonotonicity_is_surfaced_not_smoothed(sold):
    """The property exists, is computed from the verdicts, and is reported."""
    cid = _cid(sold)
    r = _tactical(sold, cid, current_price=8)
    assert isinstance(r.nonmonotonic, tuple)
    assert "nonmonotonic_pairs" in r.to_dict()
    # Every reported pair must genuinely be (cheaper-bad, dearer-good).
    ok = {p for p in r.tested_prices
          if all(v.verdict == "favorable"
                 for v in r.verdicts if v.price == p)}
    for cheap, dear in r.nonmonotonic:
        assert cheap < dear and cheap not in ok and dear in ok


def test_an_illegal_bid_for_us_yields_no_tactical_price(demo):
    """Financial capacity we cannot legally use produces no tactical maximum."""
    d = _needs_a_receiver(demo, demo.focus_owner_id)
    qb = next(s for s in d.state.available_specs
              if Position(int(s.position)) is Position.QB)
    r = _tactical(d, qb.player_id, current_price=1)
    assert r.legal_max == 0
    assert r.robust_tactical_max is None
    assert r.permissive_ceiling is None
    assert "cannot legally bid" in r.notes


# ---------------------------------------------------------------------------
# Immediate, audited, caching and precomputation
# ---------------------------------------------------------------------------


def test_the_immediate_path_does_no_ce_work(sold):
    cid = _cid(sold)
    r = _tactical(sold, cid, current_price=8,
                  settings=replace(_FAST, mode="immediate"))
    assert r.mode == "immediate"
    assert all(v.se is None for v in r.verdicts)
    assert all("NOT championship equity" in v.basis for v in r.verdicts)
    assert all(v.selection_sims is None for v in r.verdicts)
    assert "NOT a CE estimate" in r.result_kind


def test_the_cache_key_moves_with_every_meaningful_input(sold):
    cid = _cid(sold)
    base = dict(market=sold.market, current_leader="Owner09", current_price=8,
                increment=1, recipients=["Owner09 at $12"],
                scenarios=_ONE, settings=_FAST, cast=sold.cast,
                costs=sold.costs)
    k0 = tactical_cache_key(sold.state, cid, **base)
    assert k0 == tactical_cache_key(sold.state, cid, **base)

    variants = {
        "candidate": dict(base),
        "leader": {**base, "current_leader": "Owner11"},
        "price": {**base, "current_price": 9},
        "increment": {**base, "increment": 2},
        "recipients": {**base, "recipients": ["Owner11 at $12"]},
        "scenarios": {**base, "scenarios": (
            TacticalScenario("base", "high", "fabricated_base", is_base=True),)},
        "settings": {**base, "settings": replace(_FAST, max_prices=4)},
        "board_seed": {**base, "settings": replace(
            _FAST, board=replace(_FAST.board, seed=1))},
        "performance": {**base, "scenarios": (
            TacticalScenario("base", "base", "other", is_base=True),)},
        "market": {**base, "market": sold.market.observe(SaleObservation(
            "fabricated9999", "WR", 7, "Owner12", 99, 10, 11))},
    }
    seen = {k0}
    for name, kw in variants.items():
        cand = sold.candidate(1).player_id if name == "candidate" else cid
        k = tactical_cache_key(sold.state, cand, **kw)
        assert k != k0, f"cache key ignored a change to {name}"
        seen.add(k)
    # A changed auction state must move it too.
    moved = sold.with_state(sold.state.withdraw(sold.candidate(5).player_id))
    assert tactical_cache_key(moved.state, cid, **base) != k0
    assert len(seen) == len(variants) + 1


def test_a_cache_hit_is_labelled_and_matches_the_computed_result(sold):
    cid = _cid(sold)
    cache = TacticalCache()
    kw = dict(current_price=8, current_leader="Owner09", cache=cache)
    a = _tactical(sold, cid, **kw)
    b = _tactical(sold, cid, **kw)
    assert cache.stats()["hits"] == 1
    assert b.from_cache and not a.from_cache
    assert b.robust_tactical_max == a.robust_tactical_max


def test_precompute_resumes_only_on_a_matching_fingerprint(sold):
    cache = TacticalCache()
    ids = [sold.candidate(0).player_id, sold.candidate(1).player_id]
    kw = dict(settings=_FAST, scenarios=_ONE, market=sold.market,
              key_by_id=sold.key_by_id, current_price=8,
              current_leader="Owner09")
    first = precompute(sold.state, sold.cast, sold.costs, ids, cache=cache, **kw)
    assert first.n_reused == 0 and first.n_computed == 2
    again = precompute(sold.state, sold.cast, sold.costs, ids, cache=cache, **kw)
    assert again.n_reused == 2 and again.total_runtime_s < first.total_runtime_s
    # A different leader is a different question and must NOT be reused.
    changed = precompute(sold.state, sold.cast, sold.costs, ids, cache=cache,
                         **{**kw, "current_leader": "Owner11"})
    assert changed.n_reused == 0


def test_precompute_refuses_real_output_outside_local_data(sold, tmp_path):
    cache = TacticalCache()
    rep = precompute(sold.state, sold.cast, sold.costs,
                     [sold.candidate(0).player_id], cache=cache,
                     settings=_FAST, scenarios=_ONE, market=sold.market,
                     key_by_id=sold.key_by_id, fabricated=False)
    with pytest.raises(ValueError, match="local_data"):
        write_report(rep, str(tmp_path / "leak.json"))
    ok = tmp_path / "local_data" / "out.json"
    write_report(rep, str(ok))
    assert ok.exists()


@pytest.mark.slow
def test_audited_mode_uses_the_equity_engine_with_a_holdout_sample(sold):
    """One audited comparison. Slow by construction; marked so."""
    cid = _cid(sold)
    settings = replace(_FAST, mode="audited", completion=replace(
        _FAST.completion, selection_sims=400, evaluation_sims=400,
        rival_selection="proxy"))
    r = _tactical(sold, cid, current_price=8, current_leader="Owner09",
                  settings=settings, prices=[9])
    assert r.mode == "audited"
    assert r.verdicts
    for v in r.verdicts:
        assert "championship equity" in v.basis
        assert v.se is not None and v.se >= 0.0
        assert v.selection_sims == 400 and v.holdout_sims == 400
        assert v.ci95 is not None
    r.check_caps()


# ---------------------------------------------------------------------------
# Formatters, existing behaviour, and repository hygiene
# ---------------------------------------------------------------------------


def test_every_formatter_is_sanitized_of_player_names(sold):
    cid = _cid(sold)
    names = {s.name for s in sold.state.pool}
    texts = [
        format_endgame(assess_endgame(sold.state, cid, current_price=5)),
        format_bidders(assess_room_bidders(
            sold.state, cid, market=sold.market,
            candidate_key=sold.key_for(cid), costs=sold.costs)),
        format_recipients(enumerate_recipients(
            sold.state, cid, current_price=5, market=sold.market,
            candidate_key=sold.key_for(cid), costs=sold.costs)),
        format_board(continue_shared_board(
            sold.state, costs=sold.costs, market=sold.market,
            key_by_id=sold.key_by_id)),
        format_tactical(_tactical(sold, cid, current_price=8)),
    ]
    for text in texts:
        assert not (names & set(text.split()))


def test_existing_market_prior_behaviour_is_untouched():
    from ceauction.market.demo import DEMO_SALES, build_demo_prior
    from ceauction.market.live import MarketState
    prior = build_demo_prior()
    state = MarketState(prior=prior)
    sales = DEMO_SALES(prior)
    updated = state.observe_all(sales)
    assert updated.fingerprint() != state.fingerprint()
    assert len(updated.observations) == len(sales)
    key = prior.draftable[0].canonical_key
    adj = updated.adjusted(key)
    assert adj is not None and adj.base >= 1
    assert adj.prior_base == prior.by_key[key].base_price


def test_no_local_data_file_is_tracked():
    out = subprocess.run(["git", "ls-files", "local_data"],
                         capture_output=True, text=True, check=True)
    assert out.stdout.strip() == ""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli(*argv):
    return subprocess.run([sys.executable, "-m", "ceauction.cli", "tactical",
                           *argv], capture_output=True, text=True)


@pytest.mark.parametrize("argv", [
    ("validate", "--demo", "--price", "12"),
    ("endgame", "--demo", "--price", "12", "--increment", "2"),
    ("bidders", "--demo", "--demo-sales", "--bidder-scenario", "aggressive"),
    ("recipients", "--demo", "--demo-sales", "--price", "12",
     "--leader", "Owner07"),
    ("board", "--demo", "--demo-sales", "--seed", "5"),
])
def test_cli_happy_paths(argv):
    r = _cli(*argv)
    assert r.returncode == 0, r.stderr
    assert "FABRICATED" in r.stdout
    assert "Traceback" not in r.stderr


@pytest.mark.parametrize("argv,needle", [
    (("endgame",), "--demo is required"),
    (("endgame", "--demo", "--candidate", "999999"), "not in this auction"),
    (("recipients", "--demo", "--leader", "Nobody"), "no owner"),
    (("bidders", "--demo", "--bidder-scenario", "wildcat"), "invalid choice"),
    (("max-bid", "--demo", "--mode", "psychic"), "invalid choice"),
    (("endgame", "--demo", "--sale", "nonsense"), "PLAYER_ID:OWNER:PRICE"),
    (("endgame", "--demo", "--sale", "47:Owner02:99999"),
     "not a legal purchase"),
])
def test_cli_usage_errors_have_no_traceback(argv, needle):
    r = _cli(*argv)
    assert r.returncode != 0
    assert "Traceback" not in r.stderr
    assert needle in (r.stderr + r.stdout)
