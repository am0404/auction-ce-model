"""The acquisition-cost contract and the roster-completion search.

**Every input is fabricated.** No real player, no real price, no vendor value.

The completion search is the "best alternative" half of an opportunity-cost
calculation, so what these tests defend is that it is a real search rather than
a greedy walk: that it respects budget, uniqueness and the lineup graph, that
it keeps several constructions alive, that a bounded run agrees with exhaustive
enumeration where enumeration is possible, and that it never calls a heuristic
answer optimal or an unresolved pair ordered.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from ceauction.auction import new_auction
from ceauction.auction.completion import (ComparisonCast, Completion,
                                          CompletionSettings, _best_eight,
                                          complete_roster,
                                          enumerate_completions_exactly,
                                          format_completion)
from ceauction.auction.costs import (PROVENANCE_LEVELS, CostBook, CostEntry,
                                     CostProvenance, MissingCost,
                                     flat_cost_book)
from ceauction.auction.feasibility import PositionCounts
from ceauction.auction.proxy import ProxyEvaluator
from ceauction.league import DEFAULT_LEAGUE, Position
from ceauction.players import PlayerSpec
from ceauction.realdata.smoke import build_test_rosters, roster_assignment

OWNERS = tuple(f"O{i:02d}" for i in range(12))
FOCUS = "O00"


def fab(pid, pos, mean, bye=7, hazard=0.03):
    return PlayerSpec(player_id=pid, name=f"Fabricated{pid:04d}", position=pos,
                      nfl_team="ZZA", base_mean=mean, week_sd=6.0, bye_week=bye,
                      weekly_injury_hazard=hazard, injury_mean_weeks=2.5)


@pytest.fixture(scope="module")
def big_pool():
    out, pid = [], 0
    for pos, n, m, d in ((Position.QB, 40, 19.0, 0.35), (Position.RB, 90, 16.0, 0.15),
                         (Position.WR, 120, 15.0, 0.11), (Position.TE, 60, 12.0, 0.16)):
        for k in range(n):
            out.append(fab(pid, pos, max(m - d * k, 2.5), bye=5 + (k % 10)))
            pid += 1
    return tuple(out)


@pytest.fixture(scope="module")
def midauction(big_pool):
    """A snake cast with the focus owner stripped back to three players."""
    rs = build_test_rosters(list(big_pool))
    assignment = roster_assignment(rs)
    st = new_auction(big_pool, OWNERS, FOCUS)
    for i, team in enumerate(assignment):
        take = team if i != 0 else team[:3]
        for j, p in enumerate(take):
            price = 40 if (i == 0 and j == 0) else (25 if i == 0 else 12)
            st = st.apply_purchase(p, OWNERS[i], price)
    st.validate()
    cast = ComparisonCast(
        focus_team_index=0,
        rosters=tuple(tuple(st.owner(o).player_ids) for o in OWNERS),
        team_names=OWNERS)
    book = CostBook(
        entries=tuple(CostEntry(s.player_id, max(1, int(round(s.base_mean * 2.0 - 12))))
                      for s in big_pool),
        provenance=CostProvenance("FABRICATED", "test cost curve"))
    return st, cast, book


# ==========================================================================
# The cost contract
# ==========================================================================


def test_a_cost_needs_a_named_source():
    with pytest.raises(ValueError, match="must be named"):
        CostProvenance("FABRICATED", "")


def test_provenance_levels_are_ordered_weakest_first():
    assert PROVENANCE_LEVELS == ("FABRICATED", "PROVISIONAL", "TRANSFORMED", "REAL")


def test_only_real_costs_may_support_a_claim_about_value():
    for level in PROVENANCE_LEVELS:
        p = CostProvenance(level, "src")
        assert p.may_be_reported_as_a_value == (level == "REAL")


def test_every_provenance_level_has_a_disclaimer_that_names_it():
    for level in PROVENANCE_LEVELS:
        book = CostBook((CostEntry(1, 5),), CostProvenance(level, "a source"))
        text = book.disclaimer()
        assert level in text
        if level != "REAL":
            assert ("not" in text.lower())


def test_a_transformed_book_says_it_describes_a_different_format():
    """The 10-team market file is the trap this label exists for."""
    book = CostBook((CostEntry(1, 5),),
                    CostProvenance("TRANSFORMED", "10-team half-PPR market"))
    assert "DIFFERENT format" in book.disclaimer()
    assert "is this league's market price" in book.disclaimer()
    assert book.disclaimer().startswith("Acquisition costs are TRANSFORMED")


def test_a_missing_cost_is_refused_not_defaulted():
    book = flat_cost_book([1, 2, 3], 4, "unit test")
    with pytest.raises(MissingCost, match="no assumed acquisition cost"):
        book.cost_of(99)
    with pytest.raises(MissingCost, match="explicit default"):
        book.costs_for([1, 99])


def test_a_fallback_must_be_stated_explicitly():
    book = flat_cost_book([1], 4, "unit test")
    assert book.cost_of(99, default=7) == 7
    assert book.costs_for([1, 99], default=2) == {1: 4, 99: 2}


def test_all_gaps_are_reported_not_only_the_first():
    book = flat_cost_book([1], 4, "unit test")
    with pytest.raises(MissingCost) as exc:
        book.costs_for([1, 50, 51, 52])
    assert "3 of 4" in str(exc.value)


def test_a_cost_below_the_minimum_bid_is_raised_to_it():
    """No player can be bought for less; that is a rule, not an assumption."""
    book = CostBook((CostEntry(1, 0),), CostProvenance("FABRICATED", "s"),
                    minimum_cost=1)
    assert book.cost_of(1) == 1


def test_costs_are_whole_dollars():
    with pytest.raises(ValueError, match="whole auction dollars"):
        CostEntry(1, 3.5)
    with pytest.raises(ValueError, match="negative"):
        CostEntry(1, -2)


def test_duplicate_cost_entries_are_refused():
    with pytest.raises(ValueError, match="duplicate cost entries"):
        CostBook((CostEntry(1, 2), CostEntry(1, 3)),
                 CostProvenance("FABRICATED", "s"))


def test_merging_a_weaker_book_downgrades_the_whole_book():
    """One fabricated price in a real book does not leave a real book."""
    real = CostBook((CostEntry(1, 10),), CostProvenance("REAL", "this league"))
    merged = real.with_costs({2: 5}, CostProvenance("FABRICATED", "invented"))
    assert merged.level == "FABRICATED"
    assert not merged.provenance.may_be_reported_as_a_value


def test_a_cost_book_round_trips_through_json(tmp_path):
    book = flat_cost_book([1, 2], 3, "unit test", scenario_id="fh-f000-s000-w1",
                          room_state="abc123", version="v1")
    path = tmp_path / "costs.json"
    book.write_json(path)
    back = CostBook.read_json(path)
    assert back.entries == book.entries
    assert back.provenance.scenario_id == "fh-f000-s000-w1"
    assert back.provenance.room_state == "abc123"
    json.dumps(book.summary())


def test_a_flat_book_defaults_to_fabricated():
    """A flat price is not a market."""
    assert flat_cost_book([1], 5, "anything").level == "FABRICATED"


# ==========================================================================
# The proxy
# ==========================================================================


def test_the_proxy_prefers_a_stronger_roster(big_pool):
    ev = ProxyEvaluator(big_pool, n_reps=32)
    ids = {p: [s.player_id for s in big_pool if s.position is p] for p in Position}
    strong = ([ids[Position.QB][0]] + ids[Position.RB][:4]
              + ids[Position.WR][:7] + ids[Position.TE][:3])
    weak = ([ids[Position.QB][30]] + ids[Position.RB][60:64]
            + ids[Position.WR][80:87] + ids[Position.TE][40:43])
    assert ev.strength(strong) > ev.strength(weak)


def test_the_proxy_will_not_start_a_player_who_cannot_start(big_pool):
    """At equal projection, a player who can reach a slot beats one who cannot.

    Stated at *equal* projection on purpose. A third quarterback is often worth
    more than a seventh receiver in this league, because the superflex gives
    him a second slot and injury coverage -- so comparing an 18-point QB3 with
    a 14-point WR7 would test the projections, not the slot rules. Here the two
    candidates are identical except for position, and the roster already holds
    four quarterbacks, so the fifth can essentially never start.
    """
    ids = {p: [s.player_id for s in big_pool if s.position is p] for p in Position}
    extra_qb = fab(9101, Position.QB, 9.0, bye=6)
    extra_wr = fab(9102, Position.WR, 9.0, bye=6)
    ev = ProxyEvaluator(list(big_pool) + [extra_qb, extra_wr], n_reps=96)
    base = (ids[Position.QB][:4] + ids[Position.RB][:4]
            + ids[Position.WR][:4] + ids[Position.TE][:2])
    assert len(base) == 14
    with_qb = ev.strength(base + [9101])
    with_wr = ev.strength(base + [9102])
    assert with_wr > with_qb


def test_the_proxy_rewards_covering_a_bye(big_pool):
    """Depth that is available when a starter is not has to score better."""
    ev = ProxyEvaluator(big_pool, n_reps=64)
    ids = {p: [s.player_id for s in big_pool if s.position is p] for p in Position}
    core = ([ids[Position.QB][0]] + ids[Position.RB][:4]
            + ids[Position.WR][:6] + ids[Position.TE][:2])
    # Two candidates of near-identical projection, one sharing a starter's bye.
    pool = list(big_pool)
    same_bye = fab(9001, Position.WR, 9.0, bye=pool[ids[Position.WR][0]].bye_week)
    other_bye = fab(9002, Position.WR, 9.0,
                    bye=pool[ids[Position.WR][0]].bye_week % 10 + 5)
    ev2 = ProxyEvaluator(list(big_pool) + [same_bye, other_bye], n_reps=96)
    assert ev2.strength(core + [9002]) >= ev2.strength(core + [9001])


def test_the_proxy_is_batched_consistently(big_pool):
    ev = ProxyEvaluator(big_pool, n_reps=24)
    ids = {p: [s.player_id for s in big_pool if s.position is p] for p in Position}
    a = ([ids[Position.QB][0]] + ids[Position.RB][:4] + ids[Position.WR][:7]
         + ids[Position.TE][:3])
    b = ([ids[Position.QB][1]] + ids[Position.RB][4:8] + ids[Position.WR][7:14]
         + ids[Position.TE][3:6])
    batched = ev.strength_many([a, b])
    assert batched[0] == pytest.approx(ev.strength(a))
    assert batched[1] == pytest.approx(ev.strength(b))


def test_best_eight_respects_the_slot_rules():
    """Three quarterbacks cannot all start, however good they are."""
    qbs = [(30.0, int(Position.QB))] * 3
    rbs = [(10.0, int(Position.RB))] * 4
    wts = [(10.0, int(Position.WR))] * 5
    got = _best_eight(qbs + rbs + wts)
    # At most two QBs (QB slot + superflex), then six non-QB starters.
    assert got == pytest.approx(2 * 30.0 + 6 * 10.0)


# ==========================================================================
# The exact oracle
# ==========================================================================


def _tiny():
    pool = (fab(0, Position.QB, 18.0), fab(1, Position.QB, 12.0),
            fab(2, Position.RB, 15.0), fab(3, Position.RB, 13.0),
            fab(4, Position.RB, 9.0), fab(5, Position.WR, 14.0),
            fab(6, Position.WR, 11.0), fab(7, Position.WR, 8.0),
            fab(8, Position.TE, 10.0), fab(9, Position.TE, 6.0))
    return pool


def test_exact_enumeration_returns_only_legal_completions():
    pool = _tiny()
    costs = {s.player_id: 2 for s in pool}
    out = enumerate_completions_exactly(pool, costs, PositionCounts(), 8, budget=16)
    assert out
    for ids, spend in out:
        assert len(set(ids)) == 8
        assert spend == 16
        counts = PositionCounts.from_positions(
            next(s.position for s in pool if s.player_id == p) for p in ids)
        assert counts.qb >= 1 and counts.rb >= 2 and counts.wt >= 3
        assert counts.flex_eligible >= 6


def test_exact_enumeration_honours_the_budget():
    pool = _tiny()
    costs = {s.player_id: 3 for s in pool}
    assert enumerate_completions_exactly(pool, costs, PositionCounts(), 8,
                                         budget=20) == []


def test_exact_enumeration_refuses_an_absurd_combination_count():
    pool = _tiny() * 6
    pool = tuple(fab(i, s.position, s.base_mean) for i, s in enumerate(pool))
    with pytest.raises(ValueError, match="above the .* limit"):
        enumerate_completions_exactly(pool, {s.player_id: 1 for s in pool},
                                      PositionCounts(), 20, 999,
                                      max_combinations=100)


def test_the_beam_finds_the_same_best_roster_as_exhaustive_enumeration(big_pool):
    """On a board small enough to enumerate, the bounded search must not lose."""
    pool = _tiny()
    costs = {s.player_id: max(1, int(s.base_mean // 4)) for s in pool}
    exact = enumerate_completions_exactly(pool, costs, PositionCounts(), 8,
                                          budget=22)
    assert len(exact) > 5
    ev = ProxyEvaluator(pool, DEFAULT_LEAGUE, n_reps=64)
    best_exact = max(exact, key=lambda t: (ev.strength(t[0]), -t[1]))

    from ceauction.auction.completion import _beam_search, SearchDiagnostics
    diag = SearchDiagnostics()
    settings = CompletionSettings(beam_width=5000, candidate_pool=10,
                                  max_candidates=5000, proxy_candidates=5000)
    partials = _beam_search(list(pool), costs, PositionCounts(), [], 8, 22, 1,
                            settings, diag, None)
    beam_best = max(partials, key=lambda p: (ev.strength(p.taken), -p.spend))
    assert ev.strength(beam_best.taken) == pytest.approx(ev.strength(best_exact[0]))


# ==========================================================================
# The search
# ==========================================================================


def test_a_completion_fills_the_roster_legally(midauction):
    st, cast, book = midauction
    res = complete_roster(st, cast, book,
                          settings=CompletionSettings(beam_width=80, candidate_pool=40,
                                                      finalists=3, selection_sims=400, evaluation_sims=400))
    assert res.best is not None
    for c in res.finalists:
        assert len(c.roster) == DEFAULT_LEAGUE.roster_size
        assert len(set(c.roster)) == DEFAULT_LEAGUE.roster_size
        assert c.counts.qb >= 1 and c.counts.rb >= 2 and c.counts.wt >= 3
        assert c.counts.flex_eligible >= 6
        assert set(st.focus.player_ids) <= set(c.roster)


def test_a_completion_never_exceeds_the_budget(midauction):
    st, cast, book = midauction
    res = complete_roster(st, cast, book,
                          settings=CompletionSettings(beam_width=80, candidate_pool=40,
                                                      finalists=3, selection_sims=400, evaluation_sims=400))
    for c in res.finalists:
        assert c.added_cost <= st.focus.budget_remaining


def test_a_completion_never_takes_an_unavailable_player(midauction):
    st, cast, book = midauction
    res = complete_roster(st, cast, book,
                          settings=CompletionSettings(beam_width=80, candidate_pool=40,
                                                      finalists=3, selection_sims=400, evaluation_sims=400))
    rivals = cast.rival_ids
    for c in res.finalists:
        assert not (set(c.added) & rivals), "a rival's player is not available"
        for pid in c.added:
            assert st.is_available(pid)


def test_the_search_is_deterministic(midauction):
    st, cast, book = midauction
    settings = CompletionSettings(beam_width=80, candidate_pool=40, finalists=3,
                                  selection_sims=400, evaluation_sims=400)
    a = complete_roster(st, cast, book, settings=settings)
    b = complete_roster(st, cast, book, settings=settings)
    assert [c.roster for c in a.finalists] == [c.roster for c in b.finalists]
    assert [c.ce for c in a.finalists] == [c.ce for c in b.finalists]


def test_the_search_retains_several_distinct_constructions(midauction):
    """A greedy walk returns one roster; this must return several."""
    st, cast, book = midauction
    res = complete_roster(st, cast, book,
                          settings=CompletionSettings(beam_width=150,
                                                      candidate_pool=50,
                                                      finalists=6, selection_sims=300, evaluation_sims=300))
    keys = {c.key for c in res.finalists}
    assert len(keys) == len(res.finalists) >= 3
    assert res.diagnostics.candidates_kept > len(res.finalists)


def test_the_search_spans_more_than_one_spending_level(midauction):
    """The beam keeps a slice of each spend bucket, so depth stays comparable."""
    st, cast, book = midauction
    res = complete_roster(st, cast, book, evaluate_ce=False,
                          settings=CompletionSettings(beam_width=200,
                                                      candidate_pool=60,
                                                      finalists=6,
                                                      spend_buckets=8))
    assert res.diagnostics.candidate_spend_levels > 1, (
        "every candidate spending the same is a collapsed beam")
    assert res.diagnostics.finalist_spend_levels > 1, (
        "runner-ups that spend the same as the leader are near-clones and "
        "comparing them answers nothing")


def test_the_leader_is_never_dropped_for_diversity(midauction):
    st, cast, book = midauction
    settings = CompletionSettings(beam_width=150, candidate_pool=50, finalists=5,
                                  spend_buckets=8)
    res = complete_roster(st, cast, book, evaluate_ce=False, settings=settings)
    assert res.finalists[0].proxy == max(c.proxy for c in res.finalists)


def test_more_budget_never_shrinks_what_an_exhaustive_search_can_reach(midauction):
    """Stated for the exact search, where it is guaranteed; a beam may not."""
    pool = _tiny()
    costs = {s.player_id: max(1, int(s.base_mean // 4)) for s in pool}
    poor = enumerate_completions_exactly(pool, costs, PositionCounts(), 8, 20)
    rich = enumerate_completions_exactly(pool, costs, PositionCounts(), 8, 30)
    assert {frozenset(i) for i, _ in poor} <= {frozenset(i) for i, _ in rich}


def test_a_better_affordable_player_weakly_improves_the_attainable_proxy(big_pool):
    """Swapping in a strictly better player at the same price cannot hurt."""
    ev = ProxyEvaluator(big_pool, n_reps=48)
    ids = {p: [s.player_id for s in big_pool if s.position is p] for p in Position}
    core = ([ids[Position.QB][0]] + ids[Position.RB][:4]
            + ids[Position.WR][:6] + ids[Position.TE][:2])
    worse = ev.strength(core + [ids[Position.WR][40]])
    better = ev.strength(core + [ids[Position.WR][6]])
    assert better >= worse


def test_a_bounded_search_labels_itself_heuristic(midauction):
    st, cast, book = midauction
    res = complete_roster(st, cast, book,
                          settings=CompletionSettings(beam_width=40, candidate_pool=30,
                                                      finalists=2, selection_sims=200, evaluation_sims=200))
    assert not res.diagnostics.is_exact
    assert "heuristic" in res.result_kind
    assert res.to_dict()["diagnostics"]["result_is"] == "heuristic"
    text = format_completion(res)
    assert "BOUNDED SEARCH" in text
    assert "not a proof of optimality" in text
    assert "optimal completion" not in text.lower()


def test_unresolved_finalists_are_returned_as_co_best(midauction):
    """Near-identical rosters at a small sample must not be ordered on noise."""
    st, cast, book = midauction
    res = complete_roster(st, cast, book,
                          settings=CompletionSettings(beam_width=120,
                                                      candidate_pool=45,
                                                      finalists=5, selection_sims=300, evaluation_sims=300))
    assert res.unresolved, "five near-identical rosters at 300 seasons must tie"
    assert not res.is_resolved
    assert "unresolved" in res.result_kind
    assert "co-best" in format_completion(res)


def test_finalists_are_compared_on_matched_seasons(midauction):
    """Paired CRN: a shared player draws identically in both finalists."""
    st, cast, book = midauction
    settings = CompletionSettings(beam_width=100, candidate_pool=40, finalists=4,
                                  selection_sims=500, evaluation_sims=500)
    res = complete_roster(st, cast, book, settings=settings)
    # Same seed, same cast: re-running gives byte-identical CE for each roster.
    again = complete_roster(st, cast, book, settings=settings)
    assert [c.ce for c in res.finalists] == [c.ce for c in again.finalists]
    # Every finalist carries the selection-sample estimate that ranked it.
    assert all(c.selection_ce is not None for c in res.finalists)
    # The winner and any co-best finalist additionally carry a HOLDOUT
    # estimate on an independent seed. The rest do not, because the holdout
    # exists to de-bias the reported winner rather than to re-rank the field.
    with_holdout = [c for c in res.finalists if c.ce is not None]
    assert res.best.ce is not None
    assert len(with_holdout) == 1 + len(res.unresolved)


def test_the_proxy_is_never_called_championship_equity(midauction):
    st, cast, book = midauction
    res = complete_roster(st, cast, book, evaluate_ce=False,
                          settings=CompletionSettings(beam_width=50,
                                                      candidate_pool=30,
                                                      finalists=2))
    assert all(c.ce is None for c in res.finalists)
    assert res.selection_basis == "expected-points proxy"
    assert "PROXY-SELECTED" in res.notes
    assert "may not be the best by equity" in res.notes
    text = format_completion(res)
    assert "EXPECTED POINTS, not equity" in text
    assert "selected by   expected-points proxy" in text
    assert "PROXY-SELECTED" in text


def test_an_impossible_completion_is_reported_not_faked(big_pool):
    """No legal completion must come back as None with a reason, not a guess."""
    st = new_auction(big_pool, OWNERS, FOCUS)
    ids = {p: [s.player_id for s in big_pool if s.position is p] for p in Position}
    # Nine quarterbacks: legal so far, but with $6 left and six slots there is
    # no completion that both fits the lineup graph and the money.
    for pid in ids[Position.QB][:9]:
        st = st.apply_purchase(pid, FOCUS, 21)
    cast = ComparisonCast(0, tuple(tuple(st.owner(o).player_ids) for o in OWNERS),
                          OWNERS)
    book = flat_cost_book([s.player_id for s in big_pool], 40, "too expensive")
    res = complete_roster(st, cast, book,
                          settings=CompletionSettings(beam_width=40,
                                                      candidate_pool=30,
                                                      finalists=2, selection_sims=100, evaluation_sims=100))
    assert res.best is None
    assert res.result_kind == "no legal completion"
    assert res.diagnostics.stop_reason
    assert "No legal completion was found" in format_completion(res)


def test_diagnostics_report_what_the_search_actually_did(midauction):
    st, cast, book = midauction
    res = complete_roster(st, cast, book,
                          settings=CompletionSettings(beam_width=100,
                                                      candidate_pool=45,
                                                      finalists=3, selection_sims=300, evaluation_sims=300))
    d = res.diagnostics.to_dict()
    for key in ("states_expanded", "states_pruned_by_beam",
                "states_rejected_infeasible", "states_rejected_unaffordable",
                "candidates_found", "candidates_kept", "retained_per_depth",
                "finalists_evaluated", "stop_reason", "total_seconds",
                "board_size", "candidate_pool_size", "result_is"):
        assert key in d
    assert d["states_expanded"] > 0
    assert d["finalists_evaluated"] == 3
    assert len(d["retained_per_depth"]) > 0
    json.dumps(res.to_dict())


def test_the_candidate_pool_bound_is_reported(midauction):
    """A player below the cut cannot be chosen, so the cut must be visible."""
    st, cast, book = midauction
    res = complete_roster(st, cast, book, evaluate_ce=False,
                          settings=CompletionSettings(beam_width=30,
                                                      candidate_pool=25,
                                                      finalists=2))
    d = res.diagnostics
    assert d.board_size > d.candidate_pool_size
    # The pool is the projection cut PLUS cheap fillers at every position,
    # without which the most expensive players on the board cannot fill twelve
    # slots on any budget and the search finds no legal completion at all.
    assert d.cheap_fillers_added > 0
    assert d.candidate_pool_size == 25 + d.cheap_fillers_added
    assert "considered" in format_completion(res)


def test_the_pool_always_contains_an_affordable_completion(midauction):
    """The cheap tail is what makes the search complete, not an optimisation."""
    st, cast, book = midauction
    tight = CompletionSettings(beam_width=40, candidate_pool=12, finalists=2,
                               selection_sims=200, evaluation_sims=200)
    res = complete_roster(st, cast, book, evaluate_ce=False, settings=tight)
    assert res.best is not None, (
        "a projection-ranked cut alone leaves only expensive players and no "
        "legal completion")
    assert res.best.added_cost <= st.focus.budget_remaining


def test_a_missing_cost_stops_the_search_rather_than_defaulting(midauction):
    st, cast, book = midauction
    short = book.restricted_to(list(book.by_id)[:5])
    with pytest.raises(MissingCost):
        complete_roster(st, cast, short,
                        settings=CompletionSettings(beam_width=20,
                                                    candidate_pool=30,
                                                    finalists=2, selection_sims=100, evaluation_sims=100))


def test_the_cost_level_travels_into_the_result(midauction):
    st, cast, book = midauction
    res = complete_roster(st, cast, book, evaluate_ce=False,
                          settings=CompletionSettings(beam_width=20,
                                                      candidate_pool=25,
                                                      finalists=2))
    assert res.cost_level == "FABRICATED"
    assert res.to_dict()["cost_level"] == "FABRICATED"
    assert "FABRICATED" in format_completion(res, cost_disclaimer=book.disclaimer())


def test_a_comparison_cast_refuses_a_duplicated_rival_player():
    with pytest.raises(ValueError, match="two teams"):
        ComparisonCast(0, ((1,), (2,), (2,)), ("A", "B", "C"))
