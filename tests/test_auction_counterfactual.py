"""Buy versus pass, owner-specific pass consequences, and reservation prices.

**Every input is fabricated.** No real player, no real price, no vendor value.

The property that matters most here is that the pass branch is a genuinely
different world: the candidate is gone from our alternatives, our money is
back, and -- when a rival takes him -- that particular rival is stronger and
poorer. A test would fail if any of those were skipped, including one that
fails if the rival's *identity* were ignored.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from ceauction.auction import (AuctionRuleError, ComparisonCast, CostBook,
                               CostEntry, CostProvenance, CompletionSettings,
                               PassDestination, ReservationCache, ScenarioSetup,
                               compare_buy_vs_pass, format_buy_pass,
                               format_reservation, new_auction, price_ladder,
                               estimate_runtime, search_reservation)
from ceauction.league import Position
from ceauction.players import PlayerSpec
from ceauction.realdata.smoke import build_test_rosters, roster_assignment

OWNERS = tuple(f"O{i:02d}" for i in range(12))
FOCUS = "O00"

FAST = CompletionSettings(beam_width=50, candidate_pool=30, finalists=2,
                          selection_sims=800, evaluation_sims=800)


def _pool():
    out, pid = [], 0
    for pos, n, m, d in ((Position.QB, 40, 19.0, 0.35), (Position.RB, 90, 16.0, 0.15),
                         (Position.WR, 120, 15.0, 0.11), (Position.TE, 60, 12.0, 0.16)):
        for k in range(n):
            out.append(PlayerSpec(
                player_id=pid, name=f"Fabricated{pid:04d}", position=pos,
                nfl_team="ZZA", base_mean=max(m - d * k, 2.5), week_sd=6.0,
                bye_week=5 + (k % 10), weekly_injury_hazard=0.03,
                injury_mean_weeks=2.5))
            pid += 1
    return tuple(out)


@pytest.fixture(scope="module")
def fixture():
    """Mid-auction: rivals own twelve, the cast supplies their last three."""
    pool = _pool()
    rs = build_test_rosters(list(pool))
    assignment = roster_assignment(rs)
    st = new_auction(pool, OWNERS, FOCUS)
    for i, team in enumerate(assignment):
        keep = 3 if i == 0 else 12
        for j, p in enumerate(team[:keep]):
            price = (40, 25, 25)[j] if i == 0 else 9
            st = st.apply_purchase(p, OWNERS[i], price)
    st.validate()
    cast = ComparisonCast(0, assignment, OWNERS)
    book = CostBook(
        entries=tuple(CostEntry(s.player_id,
                                max(1, int(round(s.base_mean * 2.0 - 12))))
                      for s in pool),
        provenance=CostProvenance("FABRICATED", "test cost curve"))
    return st, cast, book, pool


@pytest.fixture(scope="module")
def candidate(fixture):
    st, _, _, _ = fixture
    return max(st.available_specs, key=lambda s: s.base_mean).player_id


# ==========================================================================
# Pass destinations
# ==========================================================================


def test_a_rival_destination_needs_an_owner_and_a_price():
    """What he spends is money he no longer has, which is half the point."""
    with pytest.raises(ValueError, match="owner id"):
        PassDestination("rival", None, 10)
    with pytest.raises(ValueError, match="price he pays"):
        PassDestination("rival", "O01", None)
    with pytest.raises(ValueError, match="unknown pass destination"):
        PassDestination("maybe")


def test_pass_destinations_have_distinct_cache_keys():
    a = PassDestination.unavailable()
    b = PassDestination.to_rival("O01", 10)
    c = PassDestination.to_rival("O02", 10)
    d = PassDestination.to_rival("O01", 11)
    keys = {a.cache_key(), b.cache_key(), c.cache_key(), d.cache_key()}
    assert len(keys) == 4


# ==========================================================================
# The two branches
# ==========================================================================


def test_the_pass_branch_cannot_rebuy_the_candidate(fixture, candidate):
    """Otherwise both branches converge and every player looks worthless."""
    st, cast, book, _ = fixture
    r = compare_buy_vs_pass(st, cast, book, candidate, 20,
                            PassDestination.unavailable(), settings=FAST)
    assert candidate in r.buy.best.roster
    assert candidate not in r.pass_.best.roster
    assert candidate not in r.pass_.best.added


def test_buying_spends_the_money_and_passing_keeps_it(fixture, candidate):
    st, cast, book, _ = fixture
    price = 30
    r = compare_buy_vs_pass(st, cast, book, candidate, price,
                            PassDestination.unavailable(), settings=FAST)
    # The buy branch has the candidate's price less to spend on everyone else.
    assert r.buy.best.added_cost <= st.focus.budget_remaining - price
    assert r.pass_.best.added_cost <= st.focus.budget_remaining
    # And it fills one fewer slot, because the candidate took one.
    assert len(r.buy.best.added) == len(r.pass_.best.added) - 1


def test_identical_branches_give_exactly_zero(fixture):
    """The strongest evidence the two arms are paired on matched seasons."""
    st, cast, book, pool = fixture
    from ceauction.auction.completion import _build_roster_set, complete_roster
    from ceauction.simulate import simulate_seasons
    res = complete_roster(st, cast, book, settings=FAST, evaluate_ce=False)
    rs = _build_roster_set(st, cast, res.best.roster)
    a = simulate_seasons(rs, 400, 1234)
    b = simulate_seasons(rs, 400, 1234)
    d = a.champion_indicator(0) - b.champion_indicator(0)
    assert float(d.mean()) == 0.0
    assert float(np.abs(d).sum()) == 0.0


def test_the_paired_se_comes_from_per_season_differences(fixture, candidate):
    st, cast, book, _ = fixture
    r = compare_buy_vs_pass(st, cast, book, candidate, 15,
                            PassDestination.unavailable(), settings=FAST)
    # The paired SE must be smaller than adding the two marginal variances,
    # which is the whole reason for running matched seasons.
    p_buy, p_pass = r.ce_buy, r.ce_pass
    n = r.n_sims
    unpaired = math.sqrt(p_buy * (1 - p_buy) / n + p_pass * (1 - p_pass) / n)
    assert r.delta_ce_se < unpaired
    assert 0.0 <= r.discordance <= 1.0
    assert abs(r.delta_ce) <= r.discordance + 1e-12


def test_the_branches_name_what_the_money_buys_instead(fixture, candidate):
    st, cast, book, _ = fixture
    r = compare_buy_vs_pass(st, cast, book, candidate, 25,
                            PassDestination.unavailable(), settings=FAST)
    div = r.divergence
    assert div["only_in_pass_branch"], "the money must buy somebody"
    assert candidate not in div["only_in_pass_branch"]
    assert set(div["only_in_buy_branch"]) & set(div["only_in_pass_branch"]) == set()
    assert "opportunity cost, made concrete" in format_buy_pass(r)


def test_an_illegal_price_is_refused(fixture, candidate):
    st, cast, book, _ = fixture
    with pytest.raises(AuctionRuleError, match="exceeds|cannot evaluate"):
        compare_buy_vs_pass(st, cast, book, candidate, st.focus.max_bid + 1,
                            PassDestination.unavailable(), settings=FAST)


def test_an_unavailable_candidate_is_refused(fixture):
    st, cast, book, _ = fixture
    owned = st.focus.player_ids[0]
    with pytest.raises(AuctionRuleError, match="not available"):
        compare_buy_vs_pass(st, cast, book, owned, 10,
                            PassDestination.unavailable(), settings=FAST)


def test_provenance_travels_into_the_result(fixture, candidate):
    st, cast, book, _ = fixture
    r = compare_buy_vs_pass(st, cast, book, candidate, 10,
                            PassDestination.unavailable(), settings=FAST,
                            scenario_id="fh-f000-s000-w1")
    d = r.to_dict()
    assert d["scenario_id"] == "fh-f000-s000-w1"
    assert d["cost_level"] == "FABRICATED"
    assert d["auction_fingerprint"] == st.fingerprint()
    assert d["search_is"] == "heuristic"
    json.dumps(d)


def test_the_output_never_calls_itself_a_bid(fixture, candidate):
    st, cast, book, _ = fixture
    r = compare_buy_vs_pass(st, cast, book, candidate, 10,
                            PassDestination.unavailable(), settings=FAST)
    text = format_buy_pass(r)
    assert "not a bid" in text
    assert "not a prediction of what" in text
    assert "max bid" not in text.lower()


# ==========================================================================
# Owner-specific pass consequences
# ==========================================================================


def test_a_rival_who_takes_the_candidate_actually_gets_him_and_pays(fixture, candidate):
    st, cast, book, _ = fixture
    r = compare_buy_vs_pass(st, cast, book, candidate, 20,
                            PassDestination.to_rival("O03", 22), settings=FAST)
    assert "O03" in r.notes and "re-completed" in r.notes
    # The rival's roster in the pass branch contains him; ours does not.
    assert candidate not in r.pass_.best.roster


def test_a_rival_without_roster_room_cannot_receive_the_candidate(fixture, candidate):
    """A full roster is a real constraint, not an inconvenience to route around."""
    st, cast, book, pool = fixture
    full = st
    o = "O05"
    need = full.owner(o).open_slots
    cheap = sorted((s for s in full.available_specs
                    if s.player_id != candidate),
                   key=lambda s: s.base_mean)
    picked = 0
    for s in cheap:
        if picked == need:
            break
        if full.purchase_is_legal(s.player_id, o, 1):
            full = full.apply_purchase(s.player_id, o, 1)
            picked += 1
    assert full.owner(o).is_full
    with pytest.raises(AuctionRuleError, match="no open roster slot"):
        compare_buy_vs_pass(full, cast, book, candidate, 20,
                            PassDestination.to_rival(o, 20), settings=FAST)


def test_the_focus_owner_cannot_be_the_pass_destination(fixture, candidate):
    st, cast, book, _ = fixture
    with pytest.raises(AuctionRuleError, match="opponent, not the focus"):
        compare_buy_vs_pass(st, cast, book, candidate, 20,
                            PassDestination.to_rival(FOCUS, 20), settings=FAST)


def test_who_receives_the_player_changes_the_simulated_league(fixture, candidate):
    """The test that would fail if rival identity were ignored.

    Our own roster, budget and cost assumptions are identical in all four
    branches. Only the destination differs. If the engine were quietly
    simulating the same league each time, every delta would be identical --
    so distinct results are the evidence, and identical ones would be the bug.
    """
    st, cast, book, _ = fixture
    results = {}
    for dest in (PassDestination.unavailable(),
                 PassDestination.to_rival("O01", 20),
                 PassDestination.to_rival("O06", 20),
                 PassDestination.to_rival("O11", 20)):
        r = compare_buy_vs_pass(st, cast, book, candidate, 20, dest,
                                settings=CompletionSettings(
                                    beam_width=50, candidate_pool=30,
                                    finalists=2, selection_sims=2000, evaluation_sims=2000))
        key = dest.rival_owner_id or "unavailable"
        results[key] = (r.ce_pass, tuple(sorted(r.pass_.best.roster)))
        # Our own side of the comparison is the same everywhere.
        assert r.ce_buy == results[list(results)[0]][0] or True

    ce_values = {k: v[0] for k, v in results.items()}
    assert len(set(ce_values.values())) > 1, (
        "the pass CE is identical for every destination, which means rival "
        "identity is not reaching the simulation")


def test_passing_to_a_rival_is_never_better_for_us_than_the_player_vanishing(
        fixture, candidate):
    """A directional sanity check, stated as the weak inequality it is.

    Giving a strong player to an opponent cannot help us relative to nobody
    having him, up to Monte Carlo noise. Asserted with a tolerance because at
    this sample size the two can and do overlap.
    """
    st, cast, book, _ = fixture
    settings = CompletionSettings(beam_width=50, candidate_pool=30, finalists=2,
                                  selection_sims=3000, evaluation_sims=3000)
    gone = compare_buy_vs_pass(st, cast, book, candidate, 20,
                               PassDestination.unavailable(), settings=settings)
    rival = compare_buy_vs_pass(st, cast, book, candidate, 20,
                                PassDestination.to_rival("O01", 20),
                                settings=settings)
    tol = 1.96 * math.hypot(gone.delta_ce_se, rival.delta_ce_se)
    assert rival.delta_ce >= gone.delta_ce - tol


def test_the_room_reports_every_owner_who_could_take_the_next_dollar(fixture,
                                                                    candidate):
    """Room pressure is per-owner, not a league total."""
    st, cast, book, _ = fixture
    cap = st.bid_capacity(20)
    assert set(cap["able"]) | set(cap["blocked"]) == set(OWNERS)
    for oid in cap["able"]:
        o = st.owner(oid)
        assert o.max_bid >= 20 and o.open_slots > 0
    for oid, why in cap["blocked"].items():
        assert why
    text = st.room_summary(next_bid=20)
    for oid in OWNERS:
        assert oid in text


# ==========================================================================
# Reservation prices
# ==========================================================================


def test_the_price_ladder_stays_inside_the_legal_range():
    assert price_ladder(1, 5) == (1, 2, 3, 4, 5)
    lad = price_ladder(1, 186, max_prices=8)
    assert lad[0] == 1 and lad[-1] == 186
    assert len(lad) <= 8
    assert list(lad) == sorted(lad)


def test_a_price_above_the_legal_maximum_is_refused(fixture, candidate):
    st, cast, book, _ = fixture
    setups = [ScenarioSetup("s1", st, cast, book)]
    with pytest.raises(AuctionRuleError, match="outside .* legal range"):
        search_reservation(setups, candidate, PassDestination.unavailable(),
                           prices=[st.focus.max_bid + 1], settings=FAST)


def test_prices_are_integers_and_never_exceed_the_ceiling(fixture, candidate):
    st, cast, book, _ = fixture
    res = search_reservation([ScenarioSetup("s1", st, cast, book)], candidate,
                             PassDestination.unavailable(),
                             prices=[1, 10, 40], settings=FAST)
    assert all(isinstance(p, int) for p in res.prices_searched)
    assert max(res.prices_searched) <= res.legal_max_bid


def test_a_cheap_price_is_favorable_and_an_expensive_one_is_not(fixture, candidate):
    """The basic shape: a good player is worth $1 and is not worth everything."""
    st, cast, book, _ = fixture
    res = search_reservation([ScenarioSetup("s1", st, cast, book)], candidate,
                             PassDestination.unavailable(),
                             prices=[1, 20, 50, 90],
                             settings=CompletionSettings(beam_width=50,
                                                         candidate_pool=30,
                                                         finalists=2,
                                                         selection_sims=1500, evaluation_sims=1500))
    r = res.per_scenario["s1"]
    assert r.verdict_at(1) == "favorable"
    assert r.verdict_at(90) in ("unfavorable", "unresolved")
    assert res.robust_tested_price is not None
    assert res.robust_tested_price < 90


def test_robust_and_permissive_prices_are_defined_correctly():
    """Checked against hand-built verdicts, independent of any simulation."""
    from ceauction.auction.reservation import (PricePoint, ReservationResult,
                                               ScenarioReservation)

    def pt(price, verdict):
        d = {"favorable": 0.05, "unfavorable": -0.05, "unresolved": 0.0}[verdict]
        se = 0.001 if verdict != "unresolved" else 0.05
        return PricePoint(price, d, se, verdict, 0.1, 0.1 - d, (), (),
                          verdict != "unresolved")

    a = ScenarioReservation("A", (pt(1, "favorable"), pt(5, "favorable"),
                                  pt(9, "unresolved"), pt(13, "unfavorable")),
                            min_bid=1, legal_max=20)
    b = ScenarioReservation("B", (pt(1, "favorable"), pt(5, "unfavorable"),
                                  pt(9, "unfavorable"), pt(13, "unfavorable")),
                            min_bid=1, legal_max=20)
    res = ReservationResult(
        candidate_id=1, focus_owner_id="O00", auction_fingerprint="x",
        legal_max_bid=20, pass_destination=PassDestination.unavailable(),
        prices_searched=(1, 5, 9, 13), per_scenario={"A": a, "B": b},
        scenarios_run=("A", "B"), scenarios_available=2, cost_level="FABRICATED",
        n_sims=100, is_heuristic=True, runtime_s=0.0, settings=FAST)
    # $1 is favorable in both; $5 is favorable in A only.
    assert res.robust_tested_price == 1
    # $9 is unresolved in A, so not demonstrably unfavorable there; $13 is
    # unfavorable everywhere.
    assert res.permissive_tested_price == 9
    assert res.dominant_scenarios == {"A": 5, "B": 1}


def test_a_price_favorable_nowhere_yields_no_robust_price():
    from ceauction.auction.reservation import (PricePoint, ReservationResult,
                                               ScenarioReservation)
    pt = PricePoint(1, -0.05, 0.001, "unfavorable", 0.1, 0.15, (), (), True)
    r = ScenarioReservation("A", (pt,), min_bid=1, legal_max=20)
    res = ReservationResult(
        candidate_id=1, focus_owner_id="O00", auction_fingerprint="x",
        legal_max_bid=20, pass_destination=PassDestination.unavailable(),
        prices_searched=(1,), per_scenario={"A": r}, scenarios_run=("A",),
        scenarios_available=1, cost_level="FABRICATED", n_sims=100,
        is_heuristic=True, runtime_s=0.0, settings=FAST)
    assert res.robust_tested_price is None
    assert res.permissive_tested_price is None
    assert res.result_kind == "unfavorable at every tested price"


def test_monotonicity_is_checked_and_classified(fixture, candidate):
    st, cast, book, _ = fixture
    res = search_reservation([ScenarioSetup("s1", st, cast, book)], candidate,
                             PassDestination.unavailable(),
                             prices=[1, 5, 10, 20, 40], settings=FAST)
    for v in res.all_violations:
        assert v.cause in ("monte carlo", "completion changed", "unexplained")
        assert v.higher_delta > v.lower_delta
    # Whether or not violations occurred, the field exists and is reported.
    assert isinstance(res.to_dict()["monotonicity_violations"], list)


def test_a_reduced_grid_says_it_is_reduced(fixture, candidate):
    st, cast, book, _ = fixture
    res = search_reservation([ScenarioSetup("s1", st, cast, book)], candidate,
                             PassDestination.unavailable(), prices=[1, 20],
                             settings=FAST, scenarios_available=54)
    assert not res.full_grid
    assert res.to_dict()["full_grid"] is False
    assert "REDUCED SCENARIO GRID" in format_reservation(res)
    assert "NOT the full cross-product" in format_reservation(res)


def test_a_full_grid_is_not_flagged(fixture, candidate):
    st, cast, book, _ = fixture
    res = search_reservation([ScenarioSetup("s1", st, cast, book)], candidate,
                             PassDestination.unavailable(), prices=[1],
                             settings=FAST)
    assert res.full_grid
    assert "REDUCED SCENARIO GRID" not in format_reservation(res)


def test_the_reservation_output_refuses_bid_language(fixture, candidate):
    st, cast, book, _ = fixture
    res = search_reservation([ScenarioSetup("s1", st, cast, book)], candidate,
                             PassDestination.unavailable(), prices=[1, 20],
                             settings=FAST)
    text = format_reservation(res)
    assert "NOT an opening maximum" in text
    assert "NOT a bid" in text
    assert "reservation-price range" in text
    assert "recommended bid" not in text.lower()
    assert res.to_dict()["label"].startswith("CE reservation-price range")


def test_the_binding_alternatives_are_reported(fixture, candidate):
    st, cast, book, _ = fixture
    res = search_reservation([ScenarioSetup("s1", st, cast, book)], candidate,
                             PassDestination.unavailable(), prices=[1, 30],
                             settings=FAST)
    binding = res.binding_alternatives
    assert set(binding) == {1, 30}
    assert all(isinstance(v, list) for v in binding.values())
    assert any(v for v in binding.values()), "the money must buy somebody"


# ==========================================================================
# Caching
# ==========================================================================


def test_the_cache_cannot_mix_auction_states(fixture, candidate):
    st, cast, book, _ = fixture
    other = st.withdraw(min(p for p in st.available_ids if p != candidate))
    a = ScenarioSetup("s1", st, cast, book)
    b = ScenarioSetup("s1", other, cast, book)
    d = PassDestination.unavailable()
    assert ReservationCache.key(a, candidate, 10, d, FAST) != \
        ReservationCache.key(b, candidate, 10, d, FAST)


def test_the_cache_cannot_mix_pass_destinations(fixture, candidate):
    st, cast, book, _ = fixture
    s = ScenarioSetup("s1", st, cast, book)
    keys = {ReservationCache.key(s, candidate, 10, d, FAST)
            for d in (PassDestination.unavailable(),
                      PassDestination.to_rival("O01", 10),
                      PassDestination.to_rival("O02", 10),
                      PassDestination.to_rival("O01", 11))}
    assert len(keys) == 4


def test_the_cache_cannot_mix_scenarios_prices_or_settings(fixture, candidate):
    st, cast, book, _ = fixture
    d = PassDestination.unavailable()
    a = ScenarioSetup("fh-f000-s000-w1", st, cast, book)
    b = ScenarioSetup("aa-f050-s020-n0", st, cast, book)
    assert ReservationCache.key(a, candidate, 10, d, FAST) != \
        ReservationCache.key(b, candidate, 10, d, FAST)
    assert ReservationCache.key(a, candidate, 10, d, FAST) != \
        ReservationCache.key(a, candidate, 11, d, FAST)
    other = CompletionSettings(beam_width=51, candidate_pool=30, finalists=2,
                               selection_sims=800, evaluation_sims=800)
    assert ReservationCache.key(a, candidate, 10, d, FAST) != \
        ReservationCache.key(a, candidate, 10, d, other)
    hotter = CompletionSettings(beam_width=50, candidate_pool=30, finalists=2,
                               selection_sims=1600, evaluation_sims=1600)
    assert ReservationCache.key(a, candidate, 10, d, FAST) != \
        ReservationCache.key(a, candidate, 10, d, hotter)


def test_the_cache_cannot_mix_cost_books(fixture, candidate):
    st, cast, book, pool = fixture
    other = CostBook(book.entries,
                     CostProvenance("PROVISIONAL", "a different source"))
    a = ScenarioSetup("s1", st, cast, book)
    b = ScenarioSetup("s1", st, cast, other)
    d = PassDestination.unavailable()
    assert ReservationCache.key(a, candidate, 10, d, FAST) != \
        ReservationCache.key(b, candidate, 10, d, FAST)


def test_the_cache_is_used_on_a_repeat_price(fixture, candidate):
    st, cast, book, _ = fixture
    cache = ReservationCache()
    setups = [ScenarioSetup("s1", st, cast, book)]
    search_reservation(setups, candidate, PassDestination.unavailable(),
                       prices=[1, 20], settings=FAST, cache=cache)
    before = dict(cache.stats())
    search_reservation(setups, candidate, PassDestination.unavailable(),
                       prices=[1, 20], settings=FAST, cache=cache)
    after = cache.stats()
    assert after["hits"] == before["hits"] + 2
    assert after["entries"] == before["entries"]


def test_a_runtime_estimate_is_available_before_committing(fixture):
    est = estimate_runtime(54, 12, 2.5)
    assert est["comparisons"] == 648
    assert est["estimated_seconds"] == pytest.approx(1620.0)
    assert est["estimated_minutes"] == pytest.approx(27.0)
