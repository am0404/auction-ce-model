"""Simulated mid-auction states and reservation frontiers. Fabricated only.

Two things these guard above all. First: **no league sale has ever been
recorded**, so every partial auction history in this project is invented and
must say so — a simulated state presented as observed would be inventing
evidence. Second: a reservation price is conditional on what happens when we
stop bidding, so the three pass-price rules are kept strictly apart.
"""

from __future__ import annotations

import math
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from ceauction.auction.completion import ComparisonCast, CompletionSettings
from ceauction.auction.proxy import ProxyEvaluator
from ceauction.auction.state import new_auction
from ceauction.league import DEFAULT_LEAGUE, Position
from ceauction.market.costbook import cost_book_from_prior
from ceauction.tactical.board import BoardSettings
from ceauction.tactical.demo import build_tactical_demo
from ceauction.tactical.frontier import (BRACKET, COARSE_LADDER,
                                         EXACT_FRONTIER, FRONTIER_NOT_REACHED,
                                         UNDERPOWERED, FrontierResult,
                                         PricePoint, coarse_ladder,
                                         proxy_prepass)
from ceauction.tactical.midauction import (SIMULATED_WATERMARK, PassPrice,
                                           PassPriceMisuse, PassPriceMode,
                                           STATE_SPECS, SimulatedSale,
                                           build_simulated_state,
                                           validate_state)
from ceauction.tactical.realpilot import OWNER_IDS, RealBoard


@pytest.fixture(scope="module")
def fake_board():
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
def protect(fake_board):
    by = {}
    for s in fake_board.state.available_specs:
        by.setdefault(Position(int(s.position)).name, []).append(s)
    for g in by.values():
        g.sort(key=lambda s: -s.base_mean)
    return [by["QB"][0].player_id, by["RB"][0].player_id]


@pytest.fixture(scope="module")
def sim(fake_board, protect):
    return build_simulated_state(fake_board, STATE_SPECS["balanced"],
                                 protect=protect)


# ---------------------------------------------------------------------------
# Simulated states are labelled as such
# ---------------------------------------------------------------------------


def test_the_watermark_denies_every_observed_word():
    w = SIMULATED_WATERMARK.lower()
    assert "simulated" in w
    for word in ("not observed", "not recorded", "not historical",
                 "not calibrated"):
        assert word in w
    assert "no league sale has ever been recorded" in w


def test_a_simulated_state_carries_its_provenance(sim):
    assert sim.is_simulated
    assert sim.provenance == SIMULATED_WATERMARK
    blob = sim.to_dict()
    assert blob["simulated"] is True
    assert blob["provenance"] == SIMULATED_WATERMARK


def test_no_simulated_sale_can_be_labelled_recorded(sim):
    for s in sim.sales:
        d = s.to_dict()
        assert d["simulated"] is True
        for banned in ("recorded", "observed", "historical", "actual"):
            assert banned not in {k.lower() for k in d}


def test_every_state_spec_says_simulated():
    for spec in STATE_SPECS.values():
        assert spec.description.upper().startswith("SIMULATED")


# ---------------------------------------------------------------------------
# State generation is legal, deterministic and conserving
# ---------------------------------------------------------------------------


def test_generation_is_deterministic(fake_board, protect):
    a = build_simulated_state(fake_board, STATE_SPECS["balanced"],
                              protect=protect)
    b = build_simulated_state(fake_board, STATE_SPECS["balanced"],
                              protect=protect)
    assert a.fingerprint() == b.fingerprint()
    assert [(s.player_id, s.owner_id, s.price) for s in a.sales] == \
           [(s.player_id, s.owner_id, s.price) for s in b.sales]


def test_a_different_seed_changes_the_history(fake_board, protect):
    other = replace(STATE_SPECS["balanced"], seed=987654321)
    b = build_simulated_state(fake_board, other, protect=protect)
    base = build_simulated_state(fake_board, STATE_SPECS["balanced"],
                                 protect=protect)
    assert b.fingerprint() != base.fingerprint()


def test_protected_candidates_are_never_sold(sim, protect):
    sold = {s.player_id for s in sim.sales}
    for pid in protect:
        assert pid not in sold
        assert sim.state.is_available(pid)


def test_every_simulated_purchase_is_legal(sim):
    sim.state.validate()
    seen = set()
    for o in sim.state.owners:
        for pid in o.player_ids:
            assert pid not in seen, "duplicate ownership"
            seen.add(pid)
        assert o.budget_remaining >= o.open_slots * o.min_bid
        assert o.budget_remaining >= 0


def test_dollars_and_slots_reconcile(sim):
    st = sim.state
    started = sum(o.budget_start for o in st.owners)
    spent = sum(o.spent for o in st.owners)
    left = sum(o.budget_remaining for o in st.owners)
    assert spent + left == started == 2400
    assert sum(o.n_players for o in st.owners) == len(sim.sales)
    assert sum(o.open_slots for o in st.owners) == 180 - len(sim.sales)


def test_the_sale_count_is_respected(sim):
    assert len(sim.sales) <= STATE_SPECS["balanced"].n_sales


def test_the_remaining_pool_is_correct(sim):
    st = sim.state
    sold = {s.player_id for s in sim.sales}
    for pid in sold:
        assert not st.is_available(pid)
    assert len(st.available_ids) == len(st.pool) - len(sold)


# ---------------------------------------------------------------------------
# Validation refuses unfit states
# ---------------------------------------------------------------------------


def test_validation_passes_a_fit_state(fake_board, sim, protect):
    v = validate_state(sim, fake_board, {"QB": protect[0], "RB": protect[1]})
    assert v.dollars_reconcile and not v.duplicate_ownership and v.reserve_ok
    assert set(v.candidates) == {"QB", "RB"}
    assert v.to_dict()["simulated"] is True


def test_validation_refuses_a_sold_candidate(fake_board, sim):
    sold = sim.sales[0].player_id
    v = validate_state(sim, fake_board, {"X": sold})
    assert not v.ok
    assert any("already sold" in r for r in v.refusals)


def test_validation_refuses_an_unbinding_budget(fake_board, protect):
    """An empty room in disguise cannot host a frontier."""
    empty = replace(STATE_SPECS["balanced"], n_sales=0)
    sim0 = build_simulated_state(fake_board, empty, protect=protect)
    v = validate_state(sim0, fake_board, {"QB": protect[0]})
    assert not v.ok
    assert any("cannot bind" in r for r in v.refusals)


# ---------------------------------------------------------------------------
# Pass-price semantics
# ---------------------------------------------------------------------------


def test_stop_now_uses_p_equals_q_plus_increment():
    pp = PassPrice(mode=PassPriceMode.STOP_NOW, increment=1, standing_price=30)
    assert pp.our_prices(legal_max=100) == 31
    assert pp.rival_price(31, rival_legal_max=140) == 30
    assert pp.rival_price(99, rival_legal_max=140) == 30, \
        "under STOP_NOW the rival's price is the standing bid, not ours"


def test_rival_outbids_uses_q_equals_p_plus_increment():
    pp = PassPrice(mode=PassPriceMode.RIVAL_OUTBIDS, increment=1)
    assert pp.rival_price(31, rival_legal_max=140) == 32
    assert pp.rival_price(50, rival_legal_max=140) == 51
    assert pp.our_prices(legal_max=100) is None


def test_fixed_market_requires_an_explicit_q():
    with pytest.raises(PassPriceMisuse, match="requires an explicit q"):
        PassPrice(mode=PassPriceMode.FIXED_MARKET, increment=1)
    pp = PassPrice(mode=PassPriceMode.FIXED_MARKET, increment=1, fixed_q=44)
    assert pp.rival_price(10, rival_legal_max=140) == 44
    assert pp.rival_price(90, rival_legal_max=140) == 44


def test_the_three_modes_cannot_be_mixed():
    with pytest.raises(PassPriceMisuse, match="mix two pass-price semantics"):
        PassPrice(mode=PassPriceMode.STOP_NOW, standing_price=30, fixed_q=5)
    with pytest.raises(PassPriceMisuse, match="mix semantics"):
        PassPrice(mode=PassPriceMode.RIVAL_OUTBIDS, standing_price=30)
    with pytest.raises(PassPriceMisuse, match="does not use a standing bid"):
        PassPrice(mode=PassPriceMode.FIXED_MARKET, fixed_q=10,
                  standing_price=5)
    with pytest.raises(PassPriceMisuse, match="needs the rival's standing"):
        PassPrice(mode=PassPriceMode.STOP_NOW)


def test_a_rival_who_cannot_pay_q_yields_none():
    pp = PassPrice(mode=PassPriceMode.RIVAL_OUTBIDS, increment=1)
    assert pp.rival_price(200, rival_legal_max=50) is None
    assert pp.rival_price(10, rival_legal_max=50, rival_willingness=5) is None


def test_the_pass_price_serialization_states_its_conditionality():
    pp = PassPrice(mode=PassPriceMode.STOP_NOW, standing_price=30)
    assert "not a universal max bid" in pp.to_dict()["note"]


# ---------------------------------------------------------------------------
# Ladder
# ---------------------------------------------------------------------------


def test_the_coarse_ladder_always_reaches_the_legal_maximum():
    lad = coarse_ladder(128)
    assert lad[0] == 1 and lad[-1] == 128
    assert lad == tuple(sorted(set(lad)))
    assert all(1 <= p <= 128 for p in lad)
    assert max(lad) > max(x for x in COARSE_LADDER if x <= 128) or 128 in lad


def test_the_ladder_drops_illegal_prices_and_dedups():
    lad = coarse_ladder(30, extra=[5, 5, 30, 999, 0, -4])
    assert 999 not in lad and 0 not in lad and -4 not in lad
    assert len(lad) == len(set(lad))
    assert lad[-1] == 30


def test_an_impossible_legal_max_yields_an_empty_ladder():
    assert coarse_ladder(0) == ()


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def _pt(p, verdict, audited=True):
    return PricePoint(p=p, q=p - 1, audited=audited, feasible_count=5,
                      focus_fingerprint=f"f{p}", alloc_fingerprint=f"a{p}",
                      roster_changed=False, delta=0.01,
                      ci95=(0.005, 0.015), verdict=verdict)


def _fr(points, legal_max=128, **kw):
    base = dict(candidate_position="RB", legal_max=legal_max, recipient="Team12",
                pass_price=PassPrice(mode=PassPriceMode.STOP_NOW,
                                     standing_price=30),
                points=tuple(points), k=11, holdout_sims=4000,
                selection_sims=800, cache_hits=0, state_fingerprint="s")
    base.update(kw)
    return FrontierResult(**base)


def test_a_sparse_favorable_unfavorable_pair_is_a_bracket():
    fr = _fr([_pt(10, "favorable"), _pt(50, "unfavorable")])
    assert fr.highest_favorable == 10
    assert fr.next_unfavorable == 50
    assert fr.bracket == (10, 50)
    assert fr.untested_gap == (11, 49)
    assert fr.classification == BRACKET


def test_adjacent_prices_make_an_exact_frontier():
    fr = _fr([_pt(30, "favorable"), _pt(31, "unfavorable")])
    assert fr.bracket == (30, 31)
    assert fr.untested_gap is None
    assert fr.classification == EXACT_FRONTIER


def test_favorable_to_the_legal_maximum_is_frontier_not_reached():
    fr = _fr([_pt(1, "favorable"), _pt(128, "favorable")], legal_max=128)
    assert fr.next_unfavorable is None
    assert fr.classification == FRONTIER_NOT_REACHED


def test_all_unresolved_is_underpowered():
    fr = _fr([_pt(10, "unresolved"), _pt(50, "unresolved")])
    assert fr.classification == UNDERPOWERED


def test_no_audited_points_is_underpowered():
    fr = _fr([_pt(10, "not audited", audited=False)])
    assert fr.classification == UNDERPOWERED


def test_nonmonotonic_results_are_surfaced_not_smoothed():
    fr = _fr([_pt(10, "unfavorable"), _pt(30, "favorable"),
              _pt(60, "unfavorable")])
    assert (10, 30) in fr.nonmonotonic
    assert "nonmonotonic" in fr.to_dict(include_points=False)


def test_the_serialization_refuses_to_call_a_bracket_a_max_bid():
    fr = _fr([_pt(10, "favorable"), _pt(50, "unfavorable")])
    blob = fr.to_dict(include_points=False)
    assert blob["simulated_state"] is True
    assert "NOT an exact max bid" in blob["note"]
    assert blob["pass_price"]["mode"] == "stop_now"


# ---------------------------------------------------------------------------
# Proxy pre-pass and nesting
# ---------------------------------------------------------------------------


def test_the_prepass_keeps_opportunity_sets_nested(fake_board, sim, protect):
    px = ProxyEvaluator(sim.state.pool, sim.state.settings, 16, 7)
    cs = CompletionSettings(beam_width=24, candidate_pool=30,
                            proxy_candidates=24, finalists=3,
                            max_candidates=120, proxy_reps=16)
    pre, ladder = proxy_prepass(
        sim.state, fake_board.cast, fake_board.costs["base"], protect[1],
        [1, 10, 30], board=BoardSettings(pool_depth=200, max_allocations=200),
        completion=cs, market=fake_board.market,
        key_by_id=fake_board.key_by_id, proxy=px)
    assert ladder.is_nested
    counts = [pre[p]["feasible"] for p in sorted(pre)]
    assert counts == sorted(counts, reverse=True), \
        f"feasible set must not grow with price: {counts}"


def test_no_local_real_data_is_required(sim):
    """Every test in this file runs off the fabricated demo board."""
    assert sim.state.pool
    out = subprocess.run(["git", "ls-files", "local_data"],
                         capture_output=True, text=True, check=True)
    assert out.stdout.strip() == ""
