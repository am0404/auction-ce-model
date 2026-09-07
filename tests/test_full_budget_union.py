"""The with-candidate union must be built from $1 upward, not from high prices.

The defect under test: the reconciled run generated the with-candidate
opportunity set only at the *tested* bid prices (`$65..$100` for the RB). ``UF``
pays nothing for the candidate, so it should see every construction its full
budget allows -- but it could only ever choose from constructions discovered
under a budget already reduced by a price it never pays. Possession therefore
came out negative (RB `-0.09480`, QB `-0.03277`) where it had been positive.

These tests also police the vocabulary. A price used to DISCOVER rosters is not
a price we may bid; with a $30 standing bid, $1 is a search-support price and
quoting it as a purchase option would invite an illegal bid.
"""

from __future__ import annotations

import dataclasses

import pytest

from ceauction.auction.completion import (ComparisonCast, CompletionSettings,
                                          complete_roster)
from ceauction.auction.proxy import ProxyEvaluator
from ceauction.auction.state import new_auction
from ceauction.league import DEFAULT_LEAGUE, Position
from ceauction.market.costbook import cost_book_from_prior
from ceauction.tactical.board import BoardSettings
from ceauction.tactical.demo import build_tactical_demo
from ceauction.tactical.ensemble import balanced_schedule
from ceauction.tactical.evalcontext import build_eval_context
from ceauction.tactical.midauction import (PassPrice, PassPriceMode,
                                           STATE_SPECS, build_simulated_state)
from ceauction.tactical.realpilot import OWNER_IDS, RealBoard
from ceauction.tactical.union import (LiveBidMisuse, PriceRole, PricedValue,
                                      SUPPORT_PRICES, UNION_CONVERGED,
                                      UNION_NOT_CONVERGED, UnionAdequacy,
                                      assert_live_biddable,
                                      build_full_budget_union,
                                      check_union_adequacy,
                                      completion_fingerprint)

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
def full(board, sim, px):
    s, cid = sim
    return build_full_budget_union(s.state, board.cast, board.costs["base"],
                                   cid, completion=_CS, proxy=px)


@pytest.fixture(scope="module")
def restricted(board, sim, px):
    """The old build: high tested prices only. The defect, reproduced."""
    s, cid = sim
    return build_full_budget_union(s.state, board.cast, board.costs["base"],
                                   cid, completion=_CS, proxy=px,
                                   support_prices=(65, 80, 100))


# ---------------------------------------------------------------------------
# The three prices are distinct, and only one of them is biddable
# ---------------------------------------------------------------------------


def test_a_support_price_is_not_a_live_bid():
    with pytest.raises(LiveBidMisuse, match="not a live bid"):
        assert_live_biddable(PricedValue(1, PriceRole.SEARCH_SUPPORT), 30)


def test_the_standing_pass_price_is_not_a_live_bid():
    with pytest.raises(LiveBidMisuse, match="not a live bid"):
        assert_live_biddable(PricedValue(30, PriceRole.STANDING_PASS), 30)


def test_an_evaluated_bid_at_or_below_the_standing_bid_is_refused():
    for amount in (29, 30):
        with pytest.raises(LiveBidMisuse, match="below the legal next bid"):
            assert_live_biddable(PricedValue(amount, PriceRole.EVALUATED_BID),
                                 30)


def test_the_legal_next_bid_is_biddable():
    assert_live_biddable(PricedValue(31, PriceRole.EVALUATED_BID), 30)


def test_only_the_evaluated_bid_role_reports_as_a_live_bid():
    assert PricedValue(31, PriceRole.EVALUATED_BID).is_live_bid
    assert not PricedValue(1, PriceRole.SEARCH_SUPPORT).is_live_bid
    assert not PricedValue(30, PriceRole.STANDING_PASS).is_live_bid


def test_rung_dicts_label_every_price_as_search_support(full):
    for rung in full.rungs:
        assert rung.to_dict()["role"] == PriceRole.SEARCH_SUPPORT.value
    assert "none is a live bid" in full.to_dict()["note"]


# ---------------------------------------------------------------------------
# The union starts at $1 and accumulates monotonically
# ---------------------------------------------------------------------------


def test_the_union_starts_at_one_dollar(full):
    assert full.started_at_one
    assert full.rungs[0].search_support_price == 1


def test_support_prices_are_ordered_cheapest_first():
    assert list(SUPPORT_PRICES) == sorted(SUPPORT_PRICES)
    assert SUPPORT_PRICES[0] == 1


def test_the_union_accumulates_monotonically(full):
    assert full.accumulated_monotonically
    sizes = [r.cumulative_unique for r in full.rungs]
    assert sizes == sorted(sizes)


def test_no_rung_ever_removes_an_earlier_completion(full):
    assert all(r.search_effort_nested for r in full.rungs)


def test_cross_price_nesting_holds_at_every_rung(full):
    assert all(r.cross_price_nested for r in full.rungs)


def test_dearer_support_prices_afford_a_subset_of_cheaper_ones(board, sim,
                                                              full):
    """Nesting stated the way the branches consume it."""
    s, cid = sim
    focus = s.state.focus_owner_id
    budget = s.state.owner(focus).budget_remaining
    owned = frozenset(s.state.owner(focus).player_ids) | {cid}
    costs = board.costs["base"]

    def afford(price):
        out = set()
        for c in full.completions:
            spend = sum(costs.cost_of(pid, 1) for pid in c.roster
                        if pid not in owned)
            if spend <= budget - price:
                out.add(completion_fingerprint(c))
        return out

    prices = [r.search_support_price for r in full.rungs]
    for lo, hi in zip(prices, prices[1:]):
        assert afford(hi) <= afford(lo)


def test_the_fingerprint_distinguishes_cost_not_just_roster(full):
    c = full.completions[0]
    cheaper = dataclasses.replace(c, added_cost=c.added_cost + 7)
    assert completion_fingerprint(cheaper) != completion_fingerprint(c)


def test_the_fingerprint_is_order_insensitive_over_the_roster(full):
    c = full.completions[0]
    shuffled = dataclasses.replace(c, roster=tuple(reversed(c.roster)))
    assert completion_fingerprint(shuffled) == completion_fingerprint(c)


# ---------------------------------------------------------------------------
# The full union strictly contains the restricted one
# ---------------------------------------------------------------------------


def test_low_price_searches_add_completions_the_old_build_never_saw(
        full, restricted):
    added = full.keys - restricted.keys
    assert added, ("the $1-upward rungs added nothing; the restricted build "
                   "cannot then be the cause of negative possession")


def test_the_full_union_is_at_least_as_large_as_the_restricted_one(
        full, restricted):
    assert full.size >= restricted.size


def test_the_restricted_union_never_starts_at_one_dollar(restricted):
    assert not restricted.started_at_one


def test_the_full_union_reaches_cheaper_completions(full, restricted):
    """A $1 support search leaves the most room, so it must reach further."""
    assert min(c.added_cost for c in full.completions) <= \
        min(c.added_cost for c in restricted.completions)


# ---------------------------------------------------------------------------
# The union feeds all five branches, and UF's offer is price-independent
# ---------------------------------------------------------------------------


def test_the_union_makes_the_possession_offer_price_independent(
        board, sim, px, full):
    s, cid = sim
    st = s.state
    wo = complete_roster(st, board.cast, board.costs["base"], settings=_CS,
                         owner_id=st.focus_owner_id, evaluate_ce=False,
                         default_cost=1, proxy=px,
                         reserved_ids=frozenset({cid}))
    wo_set = list(wo.finalists) or ([wo.best] if wo.best else [])
    pp = PassPrice(mode=PassPriceMode.FIXED_MARKET, increment=1, fixed_q=15)
    draw = balanced_schedule(12, 3)[0]
    seen = set()
    for p in (20, 40, 65):
        ctx = build_eval_context(
            st, board.cast, board.costs["base"], cid, p=p, pass_price=pp,
            recipient="Team05", q=15, draw=draw, board=_BS, completion=_CS,
            market=board.market, key_by_id=board.key_by_id, proxy=px,
            with_candidate=full.completions, without_candidate=wo_set)
        assert ctx.possession_offer_is_price_independent()
        seen.add(tuple(sorted(completion_fingerprint(c)
                              for c in ctx.completions_for("UF"))))
    assert len(seen) == 1, "UF's offer moved with a price UF never pays"


def test_up_sees_a_subset_of_what_uf_sees(board, sim, px, full):
    s, cid = sim
    st = s.state
    wo = complete_roster(st, board.cast, board.costs["base"], settings=_CS,
                         owner_id=st.focus_owner_id, evaluate_ce=False,
                         default_cost=1, proxy=px,
                         reserved_ids=frozenset({cid}))
    wo_set = list(wo.finalists) or ([wo.best] if wo.best else [])
    pp = PassPrice(mode=PassPriceMode.FIXED_MARKET, increment=1, fixed_q=15)
    ctx = build_eval_context(
        st, board.cast, board.costs["base"], cid, p=40, pass_price=pp,
        recipient="Team05", q=15, draw=balanced_schedule(12, 3)[0], board=_BS,
        completion=_CS, market=board.market, key_by_id=board.key_by_id,
        proxy=px, with_candidate=full.completions, without_candidate=wo_set)
    uf = {completion_fingerprint(c) for c in ctx.completions_for("UF")}
    up = {completion_fingerprint(c) for c in ctx.completions_for("UP")}
    assert up <= uf


# ---------------------------------------------------------------------------
# Adequacy and the refusal to quote a bracket
# ---------------------------------------------------------------------------


def _adequacy(**kw):
    base = dict(restricted_size=10, full_size=14, overlap=10,
                added_by_low_prices=4, restricted_uf_fingerprint="aaa",
                full_uf_fingerprint="bbb", restricted_uf_proxy=100.0,
                full_uf_proxy=103.0, extra_effort_size=14,
                extra_effort_uf_fingerprint="bbb", extra_effort_uf_proxy=103.0)
    base.update(kw)
    return UnionAdequacy(**base)


def test_a_stable_selection_under_more_effort_is_converged():
    assert _adequacy().status == UNION_CONVERGED
    assert _adequacy().may_quote_bracket


def test_a_changed_selection_beyond_tolerance_is_not_converged():
    a = _adequacy(extra_effort_uf_fingerprint="ccc",
                  extra_effort_uf_proxy=106.0)
    assert a.status == UNION_NOT_CONVERGED
    assert not a.may_quote_bracket


def test_a_changed_selection_within_tolerance_is_still_converged():
    """Two near-identical rosters swapping places is not non-convergence."""
    a = _adequacy(extra_effort_uf_fingerprint="ccc",
                  extra_effort_uf_proxy=103.05)
    assert a.status == UNION_CONVERGED


def test_convergence_is_scoped_to_the_evaluated_union():
    text = _adequacy().to_dict()["convergence_scope"]
    assert "NOT a claim" in text


def test_adequacy_reports_what_the_low_price_rungs_added():
    d = _adequacy().to_dict()
    assert d["added_by_low_price_searches"] == 4
    assert d["uf_selection_changed"] is True


def test_real_adequacy_check_runs_and_reports(board, sim, px, full,
                                              restricted):
    s, cid = sim
    a = check_union_adequacy(s.state, board.cast, board.costs["base"], cid,
                             restricted=restricted, full=full, completion=_CS,
                             proxy=px)
    assert a.full_size >= a.restricted_size
    assert a.overlap <= min(a.full_size, a.restricted_size)
    assert a.status in (UNION_CONVERGED, UNION_NOT_CONVERGED)
    assert a.extra_effort_size >= a.full_size
