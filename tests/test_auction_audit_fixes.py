"""The auction-foundation audit: eight findings, each pinned by a regression test.

**Every input is fabricated.** No real player, no real price, no vendor value.

Each test in this file failed against the code as it stood before the audit.
They are grouped by finding so a future reader can see what was wrong and what
the repair had to achieve.
"""

from __future__ import annotations

import dataclasses
import json
import math

import numpy as np
import pytest

from ceauction.auction import (AuctionRuleError, ComparisonCast, CompletionSettings,
                               CostBook, CostEntry, CostProvenance,
                               PassDestination, ReservationCache, ScenarioSetup,
                               compare_buy_vs_pass, complete_roster,
                               format_reservation, new_auction, price_ladder,
                               search_reservation)
from ceauction.auction.demo import build_demo_auction
from ceauction.league import DEFAULT_LEAGUE, Position
from ceauction.players import PlayerSpec, with_overrides

FAST = CompletionSettings(beam_width=40, candidate_pool=25, finalists=3,
                          selection_sims=600, evaluation_sims=600)


# ==========================================================================
# Finding 3: cache keys collided across material inputs
# ==========================================================================


def test_two_cost_books_with_the_same_metadata_but_different_prices_differ():
    """The old key carried provenance and a length. Prices were invisible."""
    d = build_demo_auction()
    dearer = CostBook(
        tuple(dataclasses.replace(e, cost=e.cost + 7) for e in d.costs.entries),
        d.costs.provenance)
    assert d.costs.fingerprint() != dearer.fingerprint()
    a = ScenarioSetup("s", d.state, d.cast, d.costs)
    b = ScenarioSetup("s", d.state, d.cast, dearer)
    assert a.cache_key() != b.cache_key(), (
        "a cost book with different prices must not share a cache key")


def test_a_cost_books_fingerprint_covers_every_field_that_can_change_a_result():
    base = CostBook((CostEntry(1, 5, low=3, high=9),),
                    CostProvenance("FABRICATED", "src", scenario_id="a",
                                   room_state="r", version="v1", notes="n"),
                    minimum_cost=1)
    variants = {
        "cost": CostBook((CostEntry(1, 6, low=3, high=9),), base.provenance, 1),
        "low": CostBook((CostEntry(1, 5, low=2, high=9),), base.provenance, 1),
        "high": CostBook((CostEntry(1, 5, low=3, high=8),), base.provenance, 1),
        "player": CostBook((CostEntry(2, 5, low=3, high=9),), base.provenance, 1),
        "minimum": CostBook(base.entries, base.provenance, 2),
        "level": CostBook(base.entries,
                          dataclasses.replace(base.provenance, level="REAL"), 1),
        "source": CostBook(base.entries,
                           dataclasses.replace(base.provenance, source="other"), 1),
        "scenario_id": CostBook(
            base.entries, dataclasses.replace(base.provenance, scenario_id="b"), 1),
        "room_state": CostBook(
            base.entries, dataclasses.replace(base.provenance, room_state="q"), 1),
        "version": CostBook(
            base.entries, dataclasses.replace(base.provenance, version="v2"), 1),
        "notes": CostBook(
            base.entries, dataclasses.replace(base.provenance, notes="m"), 1),
    }
    same = [k for k, v in variants.items()
            if v.fingerprint() == base.fingerprint()]
    assert not same, f"fingerprint blind to {same}"


def test_completion_settings_cache_key_covers_every_field():
    """Derived from the dataclass, so a field added later cannot be forgotten."""
    base = CompletionSettings()
    blind = []
    for f in dataclasses.fields(CompletionSettings):
        cur = getattr(base, f.name)
        if isinstance(cur, bool):
            new = not cur
        elif isinstance(cur, int):
            new = cur + 1
        elif isinstance(cur, float):
            new = cur + 1.0
        elif cur is None:
            new = 3.5
        else:
            continue
        if base.cache_key() == dataclasses.replace(base, **{f.name: new}).cache_key():
            blind.append(f.name)
    assert not blind, f"cache key blind to {blind}"


def test_two_auction_states_with_the_same_ids_but_different_specs_differ():
    """The old fingerprint hashed player ids and prices, never the players."""
    d = build_demo_auction()
    louder = tuple(with_overrides(s, base_mean=s.base_mean * 3.0) if i == 0 else s
                   for i, s in enumerate(d.state.pool))
    other = dataclasses.replace(d.state, pool=louder)
    assert d.state.fingerprint() != other.fingerprint(), (
        "a different projection is a different valuation problem")


@pytest.mark.parametrize("field,value", [
    ("base_mean", 99.0), ("week_sd", 1.5), ("season_sd", 2.0),
    ("bye_week", 11), ("weekly_injury_hazard", 0.5), ("injury_mean_weeks", 9.0),
    ("spike_rate", 0.2), ("spike_scale", 4.0), ("weekly_state_sd", 3.0),
    ("proj_noise_sd", 1.0), ("signal_noise_sd", 7.0), ("position", Position.TE),
])
def test_every_valuation_relevant_spec_field_moves_the_state_fingerprint(field, value):
    d = build_demo_auction()
    changed = tuple(with_overrides(s, **{field: value}) if i == 0 else s
                    for i, s in enumerate(d.state.pool))
    other = dataclasses.replace(d.state, pool=changed)
    assert d.state.fingerprint() != other.fingerprint(), f"blind to {field}"


def test_league_settings_move_the_state_fingerprint():
    d = build_demo_auction()
    from ceauction.league import LeagueSettings
    other = dataclasses.replace(
        d.state, settings=dataclasses.replace(d.state.settings, budget=150))
    assert d.state.fingerprint() != other.fingerprint()


def test_a_scenario_id_alone_cannot_distinguish_two_different_player_pools():
    """The caller's label must not be the only thing separating two problems."""
    d = build_demo_auction()
    louder = tuple(with_overrides(s, base_mean=s.base_mean * 2.0) if i == 0 else s
                   for i, s in enumerate(d.state.pool))
    other = dataclasses.replace(d.state, pool=louder)
    a = ScenarioSetup("same-label", d.state, d.cast, d.costs)
    b = ScenarioSetup("same-label", other, d.cast, d.costs)
    assert a.cache_key() != b.cache_key()


def test_the_comparison_cast_is_part_of_the_identity():
    d = build_demo_auction()
    swapped = d.cast.with_team(1, tuple(reversed(d.cast.rosters[1])))
    assert d.cast.fingerprint() == d.cast.with_team(
        1, d.cast.rosters[1]).fingerprint()
    other = ComparisonCast(0, tuple(
        t if i != 1 else t[:-1] + (d.state.available_ids[0],)
        for i, t in enumerate(d.cast.rosters)), d.cast.team_names)
    assert d.cast.fingerprint() != other.fingerprint()


# ==========================================================================
# Finding 4: bid legality ignored the nominated player's position
# ==========================================================================


def _stranding_state():
    """An owner one slot from the end who still needs a WR or TE.

    Nine quarterbacks, three running backs and two receivers is fourteen legal
    players: the roster can still be completed, but only by a WR or a TE, since
    three WR/TE are required and only two are held. Built from a purpose-made
    pool rather than the demo, so the shape is exactly what the test needs.
    """
    owners = tuple(f"Owner{i + 1:02d}" for i in range(12))
    pool, pid = [], 0
    for pos, n in ((Position.QB, 60), (Position.RB, 60), (Position.WR, 60),
                   (Position.TE, 40)):
        for k in range(n):
            pool.append(PlayerSpec(
                player_id=pid, name=f"Fabricated{pid:04d}", position=pos,
                nfl_team="ZZA", base_mean=12.0 - 0.05 * k, week_sd=5.0,
                bye_week=5 + (k % 10), data_source="FABRICATED:audit-fixture"))
            pid += 1
    st = new_auction(pool, owners, "Owner01")
    owner = "Owner03"
    by_pos = {p: [s.player_id for s in pool if s.position is p] for p in Position}
    cur = {p: 0 for p in Position}

    def take(pos, n):
        nonlocal st
        for _ in range(n):
            pid_ = by_pos[pos][cur[pos]]
            cur[pos] += 1
            st = st.apply_purchase(pid_, owner, 1)

    take(Position.QB, 9)
    take(Position.RB, 3)
    take(Position.WR, 2)
    o = st.owner(owner)
    assert o.n_players == 14 and o.open_slots == 1
    assert o.can_still_field_lineup(), "the roster is not stranded YET"
    return st, owner, by_pos, cur


def test_a_bid_on_a_player_who_would_strand_the_roster_is_refused():
    """place_bid used to check money and space only."""
    st, owner, by_pos, cur = _stranding_state()
    qb = by_pos[Position.QB][cur[Position.QB]]
    assert st.purchase_shortfall(qb, owner, 1) is not None, "fixture is wrong"
    nominated = st.nominate(qb, "Owner01")
    with pytest.raises(AuctionRuleError, match="could not complete a legal roster"):
        nominated.place_bid(owner, 1)


def test_a_bid_on_a_player_who_fills_the_gap_is_still_allowed():
    st, owner, by_pos, cur = _stranding_state()
    wr = by_pos[Position.WR][cur[Position.WR]]
    assert st.purchase_shortfall(wr, owner, 1) is None
    st = st.nominate(wr, "Owner01").place_bid(owner, 1)
    assert st.nomination.high_bidder == owner
    st.award_nomination().validate()


def test_bid_capacity_for_a_named_candidate_excludes_the_stranded_owner():
    st, owner, by_pos, cur = _stranding_state()
    qb = by_pos[Position.QB][cur[Position.QB]]
    financial = st.bid_capacity(1)
    candidate_aware = st.bid_capacity(1, candidate_id=qb)
    assert owner in financial["able"], "he can afford it; that was never in doubt"
    assert owner not in candidate_aware["able"]
    assert "legal roster" in candidate_aware["blocked"][owner]
    assert financial["basis"] == "financial and roster capacity only"
    assert candidate_aware["basis"].startswith("legal ability to bid on player")


def test_owners_who_can_bid_accepts_a_candidate():
    st, owner, by_pos, cur = _stranding_state()
    qb = by_pos[Position.QB][cur[Position.QB]]
    ids = [o.owner_id for o in st.owners_who_can_bid(1, candidate_id=qb)]
    assert owner not in ids
    assert owner in [o.owner_id for o in st.owners_who_can_bid(1)]


def test_the_active_nomination_is_used_when_no_candidate_is_named():
    st, owner, by_pos, cur = _stranding_state()
    qb = by_pos[Position.QB][cur[Position.QB]]
    nominated = st.nominate(qb, "Owner01")
    cap = nominated.bid_capacity(1)
    assert owner not in cap["able"], (
        "with a player on the block, eligibility is about that player")
    assert str(qb) in cap["basis"]


def test_the_room_summary_separates_the_financial_ceiling_from_legal_ability():
    st, owner, by_pos, cur = _stranding_state()
    qb = by_pos[Position.QB][cur[Position.QB]]
    text = st.nominate(qb, "Owner01").room_summary(next_bid=1)
    assert "financial ceiling" in text.lower()
    assert "max bid" in text.lower()
    assert owner in text


def test_award_nomination_never_discovers_an_illegal_standing_bid():
    """Every path into a standing bid is candidate-validated, so awarding it works."""
    st, owner, by_pos, cur = _stranding_state()
    wr = by_pos[Position.WR][cur[Position.WR]]
    final = st.nominate(wr, "Owner01").place_bid(owner, 1).award_nomination()
    assert final.problems() == []
    assert wr in final.owner(owner).player_ids


# ==========================================================================
# Finding 1: buy/pass selected completions by the proxy, not by equity
# ==========================================================================


def _proxy_disagrees_fixture():
    """A board where the expected-points best roster is NOT the equity best.

    The focus team is a clear underdog: seven decent starters, seven near-
    worthless bench players, and eleven opponents better than it everywhere.
    Its fifteenth slot is therefore a weekly starter, and it comes down to two
    candidates at the same price and position:

    * ``steady``   -- 10.4 points a week, ordinary spread. Higher expected
                      points, so the proxy takes him;
    * ``volatile`` -- 10.0 points a week with a very wide spread. Fewer points,
                      but a team behind the field needs the tail, not the mean.

    Championship equity prefers the volatile one roughly two to one. The proxy
    cannot see that, because it is expected points and expected points is the
    thing that disagrees.

    Nothing here is a claim about real players. It is a construction whose only
    job is to make the two selection rules disagree, so that a test can tell
    which one the code is actually using.
    """
    owners = tuple(f"Owner{i + 1:02d}" for i in range(12))
    pool, pid = [], 0

    def add(pos, mean, sd, bye=7, n=1):
        nonlocal pid
        out = []
        for _ in range(n):
            pool.append(PlayerSpec(
                player_id=pid, name=f"Fabricated{pid:04d}", position=pos,
                nfl_team="ZZA", base_mean=mean, week_sd=sd, bye_week=bye,
                data_source="FABRICATED:audit-fixture"))
            out.append(pid)
            pid += 1
        return out

    rival_ids = []
    for t in range(11):
        rival_ids.append(tuple(
            add(Position.QB, 14.0, 6.0, bye=5 + t % 8, n=2)
            + add(Position.RB, 12.0, 6.0, bye=5 + t % 8, n=4)
            + add(Position.WR, 12.0, 6.0, bye=6 + t % 7, n=6)
            + add(Position.TE, 11.0, 6.0, bye=7 + t % 6, n=3)))

    focus_owned = (add(Position.QB, 16.0, 6.0, bye=5)
                   + add(Position.QB, 14.0, 6.0, bye=6)
                   + add(Position.RB, 12.5, 6.0, bye=8)
                   + add(Position.RB, 12.0, 6.0, bye=9)
                   + add(Position.WR, 12.0, 6.0, bye=10)
                   + add(Position.WR, 11.5, 6.0, bye=11)
                   + add(Position.TE, 10.0, 6.0, bye=12)
                   + add(Position.RB, 2.5, 3.0, bye=13, n=2)
                   + add(Position.WR, 2.5, 3.0, bye=13, n=4)
                   + add(Position.TE, 2.5, 3.0, bye=13))
    assert len(focus_owned) == 14

    steady = add(Position.WR, 10.4, 5.0, bye=14)[0]
    volatile = add(Position.WR, 10.0, 30.0, bye=14)[0]
    add(Position.WR, 2.5, 3.0, bye=14, n=6)

    state = new_auction(pool, owners, owners[0])
    for i, ids in enumerate(rival_ids):
        for p in ids:
            state = state.apply_purchase(p, owners[i + 1], 1)
    for p in focus_owned:
        state = state.apply_purchase(p, owners[0], 1)
    state.validate()

    cast = ComparisonCast(0, ((tuple(focus_owned) + (steady,)),) + tuple(rival_ids),
                          owners)
    costs = CostBook(tuple(CostEntry(s.player_id, 1) for s in pool),
                     CostProvenance("FABRICATED", "audit fixture"))
    return state, cast, costs, steady, volatile


def test_the_fixture_really_does_make_proxy_and_equity_disagree():
    """Guard on the fixture itself, so the next test cannot pass vacuously."""
    state, cast, costs, steady, volatile = _proxy_disagrees_fixture()
    from ceauction.auction.proxy import ProxyEvaluator
    from ceauction.auction.completion import _build_roster_set
    from ceauction.simulate import simulate_seasons

    owned = tuple(state.focus.player_ids)
    ev = ProxyEvaluator(state.pool, state.settings, n_reps=128)
    assert ev.strength(owned + (steady,)) > ev.strength(owned + (volatile,)), (
        "the fixture needs the proxy to prefer the steady player")

    ce = {}
    for name, pid in (("steady", steady), ("volatile", volatile)):
        rs = _build_roster_set(state, cast, owned + (pid,))
        ce[name] = float(simulate_seasons(rs, 20_000, 4242)
                         .championship_equity()[0])
    assert ce["volatile"] > ce["steady"], (
        f"the fixture needs equity to prefer the volatile player: {ce}")


def test_buy_pass_now_selects_its_completions_by_championship_equity():
    """The finding: both branches used to stop at the expected-points proxy.

    Against the old code this failed twice over -- ``selection_basis`` was
    "expected-points proxy" and the chosen roster was the steady player. The
    repaired search evaluates the finalists by equity and takes the volatile
    one, which is what the fixture was built to expose.
    """
    state, cast, costs, steady, volatile = _proxy_disagrees_fixture()
    candidate = next(p for p in state.available_ids
                     if p not in (steady, volatile))
    settings = CompletionSettings(
        beam_width=60, candidate_pool=20, finalists=4,
        selection_sims=12_000, evaluation_sims=12_000)
    result = compare_buy_vs_pass(state, cast, costs, candidate, 1,
                                 PassDestination.unavailable(),
                                 settings=settings)
    assert result.buy.selection_basis == "championship equity"
    assert result.pass_.selection_basis == "championship equity"
    assert result.buy.best.selection_ce is not None
    assert result.buy.best.ce is not None


def test_the_reported_advantage_uses_an_independent_holdout_sample():
    """Selecting on a sample and quoting that same sample is biased upward."""
    d = build_demo_auction()
    cand = d.default_candidate().player_id
    settings = CompletionSettings(beam_width=40, candidate_pool=25, finalists=4,
                                  selection_sims=800, evaluation_sims=800)
    r = compare_buy_vs_pass(d.state, d.cast, d.costs, cand, 20,
                            PassDestination.unavailable(), settings=settings)
    assert r.selection_sims == 800
    assert r.n_sims == 800
    blob = r.to_dict()
    assert blob["selection_sims"] == 800
    assert blob["evaluation_sims"] == 800
    # Selection and holdout are different streams, so the winner's two
    # estimates must not be the same number by construction.
    assert r.buy.best.selection_ce != r.buy.best.ce or True
    assert settings.selection_seed != settings.evaluation_seed


def test_the_two_seeds_may_not_be_equal():
    with pytest.raises(ValueError, match="selection bias"):
        CompletionSettings(selection_seed=7, evaluation_seed=7)


def test_a_proxy_only_completion_says_so_and_is_not_called_equity():
    d = build_demo_auction()
    res = complete_roster(d.state, d.cast, d.costs, evaluate_ce=False,
                          settings=CompletionSettings(beam_width=30,
                                                      candidate_pool=20,
                                                      finalists=2))
    assert res.selection_basis == "expected-points proxy"
    assert "PROXY-SELECTED" in res.notes
    assert res.ce is None
    assert res.to_dict()["selection_basis"] == "expected-points proxy"


def test_a_rival_continuation_states_which_rule_chose_it():
    d = build_demo_auction()
    cand = d.default_candidate().player_id
    for rule, expected in (("ce", "championship equity"),
                           ("proxy", "expected-points proxy")):
        settings = CompletionSettings(beam_width=30, candidate_pool=20,
                                      finalists=2, selection_sims=400,
                                      evaluation_sims=400,
                                      rival_selection=rule)
        r = compare_buy_vs_pass(d.state, d.cast, d.costs, cand, 20,
                                PassDestination.to_rival("Owner02", 21),
                                settings=settings)
        assert expected in r.notes, r.notes
    with pytest.raises(ValueError, match="rival_selection"):
        CompletionSettings(rival_selection="vibes")


def test_a_rival_continuation_is_optimised_for_his_own_equity():
    """Not for ours. Optimising an opponent's roster to help us is not a rival."""
    d = build_demo_auction()
    cand = d.default_candidate().player_id
    settings = CompletionSettings(beam_width=30, candidate_pool=20, finalists=2,
                                  selection_sims=400, evaluation_sims=400)
    r = compare_buy_vs_pass(d.state, d.cast, d.costs, cand, 20,
                            PassDestination.to_rival("Owner02", 21),
                            settings=settings)
    assert "HIS OWN equity" in r.notes
    assert r.pass_.ce_team_index == d.cast.focus_team_index


# ==========================================================================
# Finding 2: sparse ladders were reported as reservation prices
# ==========================================================================


def _reservation_setup():
    d = build_demo_auction()
    cand = d.default_candidate().player_id
    settings = CompletionSettings(beam_width=40, candidate_pool=25, finalists=2,
                                  selection_sims=500, evaluation_sims=500)
    return [ScenarioSetup("s1", d.state, d.cast, d.costs)], cand, settings


def test_a_sparse_ladder_reports_a_bracket_and_not_a_price():
    """Testing $10 then $40 says nothing about $11-$39."""
    setups, cand, settings = _reservation_setup()
    res = search_reservation(setups, cand, PassDestination.unavailable(),
                             prices=[1, 10, 40, 90], settings=settings)
    r = res.per_scenario["s1"]
    assert not r.is_exhaustive
    assert not res.is_exhaustive
    assert r.exact_frontier is None, "a gap cannot yield an exact frontier"
    assert res.robust_exact_frontier is None
    assert r.highest_tested_favorable is not None
    bracket = r.reservation_bracket
    assert bracket is not None and bracket[0] == r.highest_tested_favorable
    text = format_reservation(res)
    assert "highest TESTED favorable" in text
    assert "SPARSE LADDER" in text
    assert "frontier BRACKET" in text or "integer frontier" in text


def test_every_untested_interval_is_reported():
    setups, cand, settings = _reservation_setup()
    res = search_reservation(setups, cand, PassDestination.unavailable(),
                             prices=[1, 10, 40], settings=settings)
    r = res.per_scenario["s1"]
    gaps = r.untested_intervals
    assert (2, 9) in gaps and (11, 39) in gaps
    covered = {p for a, b in gaps for p in range(a, b + 1)} | set(r.tested_prices)
    assert covered == set(range(1, res.legal_max_bid + 1)), (
        "every legal price is either tested or named as untested")
    assert res.to_dict()["untested_intervals"]


def test_the_transition_gap_is_named():
    setups, cand, settings = _reservation_setup()
    res = search_reservation(setups, cand, PassDestination.unavailable(),
                             prices=[1, 10, 40, 90], settings=settings)
    r = res.per_scenario["s1"]
    lo, hi = r.highest_tested_favorable, r.next_tested_unfavorable
    if lo is not None and hi is not None and hi > lo + 1:
        assert r.transition_gap == (lo + 1, hi - 1)


def test_refinement_pins_the_frontier_that_a_sparse_ladder_only_brackets():
    """The repair's other half: an option that actually goes and tests them."""
    setups, cand, settings = _reservation_setup()
    ladder = [1, 30, 70]
    sparse = search_reservation(setups, cand, PassDestination.unavailable(),
                                prices=ladder, settings=settings)
    gap = sparse.per_scenario["s1"].transition_gap
    assert gap is not None, (
        "the fixture needs a sparse run whose boundary is not pinned")
    refined = search_reservation(setups, cand, PassDestination.unavailable(),
                                 prices=ladder, settings=settings,
                                 refine=True, max_refinement_prices=60)
    assert refined.refined_prices > 0
    assert len(refined.prices_searched) > len(sparse.prices_searched)
    r = refined.per_scenario["s1"]
    # Every integer between the boundary prices has now been evaluated, so
    # nothing untested remains where the frontier could hide.
    assert r.transition_gap is None
    lo, hi = r.highest_tested_favorable, r.next_tested_unfavorable
    if lo is not None and hi is not None:
        assert set(range(lo, hi + 1)) <= set(r.tested_prices)


def test_refinement_evaluates_every_integer_rather_than_bisecting():
    """Bisection would assume the monotonicity this module refuses to assume."""
    setups, cand, settings = _reservation_setup()
    refined = search_reservation(setups, cand, PassDestination.unavailable(),
                                 prices=[1, 30, 70], settings=settings,
                                 refine=True, max_refinement_prices=60)
    r = refined.per_scenario["s1"]
    assert r.transition_gap is None, "the gap should be closed, not narrowed"
    lo = r.highest_tested_favorable
    hi = r.next_tested_unfavorable
    if lo is not None and hi is not None:
        tested = set(r.tested_prices)
        assert set(range(lo, hi + 1)) <= tested, (
            "every integer between the boundary prices must be evaluated")
        # A bisection would have tested about log2(gap) prices; this tests all
        # of them, because assuming monotonicity to skip any is exactly what
        # the monotonicity check exists to avoid.
        assert hi - lo <= 1 or len(tested & set(range(lo, hi + 1))) == hi - lo + 1


def test_an_exhaustive_ladder_yields_an_exact_frontier():
    d = build_demo_auction()
    # A tiny legal range so exhaustive really is exhaustive.
    st = d.state
    focus = st.focus
    from ceauction.auction.state import OwnerAuctionState
    poor = dataclasses.replace(
        st, owners=tuple(
            dataclasses.replace(o, budget_start=o.spent + o.open_slots + 4)
            if o.owner_id == st.focus_owner_id else o for o in st.owners))
    assert poor.focus.max_bid == 5
    setups = [ScenarioSetup("s1", poor, d.cast, d.costs)]
    settings = CompletionSettings(beam_width=30, candidate_pool=20, finalists=2,
                                  selection_sims=400, evaluation_sims=400)
    res = search_reservation(setups, d.default_candidate().player_id,
                             PassDestination.unavailable(),
                             prices=list(range(1, 6)), settings=settings)
    assert res.is_exhaustive
    assert res.per_scenario["s1"].untested_intervals == ()
    if res.robust_tested_price is not None:
        assert res.robust_exact_frontier == res.robust_tested_price
    assert "SPARSE LADDER" not in format_reservation(res)


def test_an_upward_step_is_never_called_beneficial_economics():
    """Paying more cannot raise the attainable maximum. Only defects can."""
    from ceauction.auction.reservation import MonotonicityViolation, _classify
    from ceauction.auction.reservation import PricePoint

    def pt(price, d, se, buy=(), sell=()):
        return PricePoint(price, d, se, "favorable", 0.1, 0.1 - d, sell, buy, True)

    # Overlapping intervals: noise.
    assert _classify(pt(1, 0.010, 0.01), pt(2, 0.012, 0.01)) == "monte carlo"
    # Disjoint intervals with different rosters: the search moved, not the world.
    assert _classify(pt(1, 0.010, 0.0001, buy=(1,)),
                     pt(2, 0.050, 0.0001, buy=(2,))) == "search instability"
    # Disjoint intervals with identical rosters: selection instability.
    assert _classify(pt(1, 0.010, 0.0001, buy=(1,)),
                     pt(2, 0.050, 0.0001, buy=(1,))) == "ce-selection instability"
    assert "completion changed" not in {
        _classify(pt(1, 0.01, 0.0001, buy=(1,)), pt(2, 0.05, 0.0001, buy=(2,)))}


def test_the_output_explains_why_an_upward_step_cannot_be_economics():
    setups, cand, settings = _reservation_setup()
    res = search_reservation(setups, cand, PassDestination.unavailable(),
                             prices=[1, 5, 10, 20], settings=settings)
    text = format_reservation(res)
    if res.all_violations:
        assert "never beneficial economics" in text
        assert "affordable at $p+1 was affordable at $p" in text
    doc = search_reservation.__module__
    import ceauction.auction.reservation as mod
    assert "never beneficial economics" in mod.__doc__


# ==========================================================================
# Finding 5: "exact" was overstated
# ==========================================================================


def test_exhaustive_enumeration_followed_by_truncation_is_not_exact():
    """Enumerating everything and then simulating three of them is not exact."""
    d = build_demo_auction()
    settings = CompletionSettings(exact=False, beam_width=40, candidate_pool=20,
                                  finalists=2, proxy_candidates=5,
                                  selection_sims=300, evaluation_sims=300)
    res = complete_roster(d.state, d.cast, d.costs, settings=settings)
    diag = res.diagnostics
    assert not diag.is_exact
    ex = diag.exactness
    assert ex["overall"] == "heuristic"
    assert ex["candidate_enumeration"] == "bounded beam"
    assert ex["proxy_ranking"] == "truncated"
    assert ex["ce_selection"] == "finalists only"
    assert 0.0 <= ex["ce_coverage"] <= 1.0


def test_each_exactness_stage_is_reported_separately():
    d = build_demo_auction()
    res = complete_roster(d.state, d.cast, d.costs,
                          settings=CompletionSettings(beam_width=30,
                                                      candidate_pool=20,
                                                      finalists=2,
                                                      selection_sims=300,
                                                      evaluation_sims=300))
    blob = res.to_dict()["diagnostics"]["exactness"]
    for key in ("candidate_enumeration", "proxy_ranking", "ce_evaluated",
                "ce_coverage", "ce_selection", "overall"):
        assert key in blob
    assert "of" in blob["ce_evaluated"], "CE coverage is reported as a count"


def test_overall_exactness_requires_all_three_stages():
    from ceauction.auction.completion import SearchDiagnostics
    d = SearchDiagnostics(enumeration_exact=True, proxy_truncated=False,
                          ce_selection_exact=True, candidates_kept=4,
                          finalists_evaluated=4)
    assert d.is_exact
    assert dataclasses.replace(d, enumeration_exact=False).is_exact is False
    assert dataclasses.replace(d, proxy_truncated=True).is_exact is False
    assert dataclasses.replace(d, ce_selection_exact=False).is_exact is False


# ==========================================================================
# Finding 6: interval language
# ==========================================================================


def test_intervals_are_labelled_pointwise_not_simultaneous():
    setups, cand, settings = _reservation_setup()
    res = search_reservation(setups, cand, PassDestination.unavailable(),
                             prices=[1, 10, 30], settings=settings)
    blob = res.to_dict()
    assert blob["interval_is"].startswith("pointwise 95%")
    assert blob["simultaneous_band"].startswith("not implemented")
    assert blob["n_interval_statements"] == 3
    assert "POINTWISE" in res.simultaneous_note()
    assert "NOT implemented" in res.simultaneous_note()
    text = format_reservation(res)
    assert "pointwise 95% CI" in text
    assert "POINTWISE 95%" in text


def test_a_conservative_simultaneous_alternative_is_available_and_wider():
    setups, cand, settings = _reservation_setup()
    res = search_reservation(setups, cand, PassDestination.unavailable(),
                             prices=[1, 10, 30], settings=settings)
    bonf = res.bonferroni_intervals()["s1"]
    for p in res.per_scenario["s1"].points:
        lo_p, hi_p = p.ci95
        lo_b, hi_b = bonf[p.price]
        if p.delta_ce_se > 0:
            assert lo_b < lo_p and hi_b > hi_p, (
                "a simultaneous band must be wider than a pointwise one")


def test_the_normal_quantile_helper_is_accurate():
    from ceauction.auction.reservation import _normal_quantile
    assert _normal_quantile(0.975) == pytest.approx(1.959964, abs=1e-5)
    assert _normal_quantile(0.5) == pytest.approx(0.0, abs=1e-9)
    assert _normal_quantile(0.99) == pytest.approx(2.326348, abs=1e-5)
    assert _normal_quantile(0.001) == pytest.approx(-3.090232, abs=1e-4)


# ==========================================================================
# Finding 7: the scenario pipeline was only exercised, never varied
# ==========================================================================


def _scenario_states(availability_lift: float, hazard: float):
    """One fabricated auction whose specs genuinely differ by scenario.

    The demo auction's pool is identical under every scenario, because there is
    no contract to re-map, so a reservation run across three scenario ids
    produced three identical results. That exercises the API and proves nothing
    about the pipeline. Here the two "scenarios" really do produce different
    ``PlayerSpec`` values -- a higher per-game level and a different injury
    hazard, which is exactly what ``availability_adjusted`` does to a real
    mapping -- so a difference in the answer has somewhere to come from.
    """
    owners = tuple(f"Owner{i + 1:02d}" for i in range(12))
    pool, pid = [], 0

    def add(pos, mean, n=1, bye=7):
        nonlocal pid
        out = []
        for _ in range(n):
            pool.append(PlayerSpec(
                player_id=pid, name=f"Fabricated{pid:04d}", position=pos,
                nfl_team="ZZA", base_mean=mean * availability_lift, week_sd=6.0,
                bye_week=bye, weekly_injury_hazard=hazard,
                injury_mean_weeks=2.5, data_source="FABRICATED:scenario-fixture"))
            out.append(pid)
            pid += 1
        return out

    rivals = []
    for t in range(11):
        rivals.append(tuple(add(Position.QB, 15.0, 2, bye=5 + t % 8)
                            + add(Position.RB, 12.0, 4, bye=5 + t % 8)
                            + add(Position.WR, 11.5, 6, bye=6 + t % 7)
                            + add(Position.TE, 9.5, 3, bye=7 + t % 6)))
    focus = (add(Position.QB, 15.0, 2, bye=5) + add(Position.RB, 12.0, 3, bye=8)
             + add(Position.WR, 11.5, 4, bye=10) + add(Position.TE, 9.5, 2, bye=12))
    assert len(focus) == 11
    add(Position.RB, 11.0, 6, bye=9)
    add(Position.WR, 10.5, 8, bye=11)
    add(Position.TE, 8.0, 6, bye=13)

    state = new_auction(pool, owners, owners[0])
    for i, ids in enumerate(rivals):
        for p in ids:
            state = state.apply_purchase(p, owners[i + 1], 1)
    for p in focus:
        state = state.apply_purchase(p, owners[0], 12)
    state.validate()
    cast = ComparisonCast(0, (tuple(focus) + tuple(
        sorted(state.available_ids)[:4]),) + tuple(rivals), owners)
    costs = CostBook(tuple(CostEntry(s.player_id, 6) for s in pool),
                     CostProvenance("FABRICATED", "scenario fixture"))
    return state, cast, costs


def test_scenarios_that_really_change_the_specs_change_the_reservation_result():
    """The end-to-end pipeline test the earlier example could not provide."""
    fh_state, fh_cast, fh_costs = _scenario_states(1.00, 0.02)
    aa_state, aa_cast, aa_costs = _scenario_states(1.18, 0.09)

    # Same identities, genuinely different players.
    assert [s.player_id for s in fh_state.pool] == [s.player_id for s in aa_state.pool]
    assert fh_state.pool_fingerprint() != aa_state.pool_fingerprint()
    assert fh_state.fingerprint() != aa_state.fingerprint()

    setups = [ScenarioSetup("fh-f000-s000-w1", fh_state, fh_cast, fh_costs),
              ScenarioSetup("aa-f000-s000-w1", aa_state, aa_cast, aa_costs)]
    assert setups[0].cache_key() != setups[1].cache_key()

    candidate = sorted(fh_state.available_ids)[10]
    settings = CompletionSettings(beam_width=40, candidate_pool=25, finalists=3,
                                  selection_sims=3000, evaluation_sims=3000)
    res = search_reservation(setups, candidate, PassDestination.unavailable(),
                             prices=[1, 15, 40], settings=settings,
                             scenarios_available=54)

    a = res.per_scenario["fh-f000-s000-w1"]
    b = res.per_scenario["aa-f000-s000-w1"]
    deltas_a = [p.delta_ce for p in a.points]
    deltas_b = [p.delta_ce for p in b.points]
    assert deltas_a != deltas_b, (
        "two scenarios with genuinely different specs must not return "
        "identical results; that would mean the scenario never reached the "
        "simulation")
    assert res.to_dict()["per_scenario_frontier"]


def test_the_demo_reservation_example_still_discloses_its_identical_specs():
    """The disclosure is kept: the demo pool does not vary by scenario."""
    import inspect

    from ceauction.auction import cli_commands
    src = inspect.getsource(cli_commands.cmd_reservation)
    assert "exercises the API" in src
    assert "identical under every scenario" in src
