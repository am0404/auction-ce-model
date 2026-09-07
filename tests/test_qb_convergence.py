"""A convergence stopping rule for the QB union, and its refusals.

`FULL_BUDGET_UNION.md` withheld the QB's `PASS_AT_NEXT_BID` because tripling the
search effort moved `UF`'s selected roster by `+0.690` proxy points. These tests
pin the rule that decides when that has stopped -- and, just as importantly, the
things that must NOT count as convergence: a union whose raw size settled, a
single effort rung, or a ladder retrofitted after seeing results.
"""

from __future__ import annotations

import dataclasses

import pytest

from ceauction.auction.completion import (ComparisonCast, Completion,
                                          CompletionSettings)
from ceauction.auction.proxy import ProxyEvaluator
from ceauction.auction.state import new_auction
from ceauction.league import DEFAULT_LEAGUE, Position
from ceauction.market.costbook import cost_book_from_prior
from ceauction.tactical.demo import build_tactical_demo
from ceauction.tactical.midauction import (STATE_SPECS, build_simulated_state)
from ceauction.tactical.qbconvergence import (
    EFFORT_LADDER, LADDER_DECLARED_AT, PROXY_TOLERANCE, QB_UNION_CONVERGED,
    QB_UNION_UNDERCONVERGED, ConvergenceReport, EffortRung, RungResult,
    SeedMisuse, assert_independent_seeds, economic_profile,
    economically_equivalent, evaluate_stopping_rule, ladder_fingerprint,
    run_convergence_ladder)
from ceauction.tactical.realpilot import OWNER_IDS, RealBoard
from ceauction.tactical.union import completion_fingerprint

_PRICES = (10, 20, 40)


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
    prot = [by["QB"][0].player_id]
    return build_simulated_state(board, STATE_SPECS["balanced"],
                                 protect=prot), prot[0]


@pytest.fixture(scope="module")
def eq_kw(board, sim):
    s, cid = sim
    st = s.state
    focus = st.focus_owner_id
    return dict(costs=board.costs["base"],
                budget=st.owner(focus).budget_remaining,
                owned=frozenset(st.owner(focus).player_ids) | {cid},
                candidate_id=cid, roster_size=st.settings.roster_size,
                default_cost=1, tolerance=PROXY_TOLERANCE)


def _rung(name="8x", *, best=98.0, uf=None, up_proxy=98.0, up_fp="a",
          spread=0.0, afford=10, size=100):
    return RungResult(
        rung=EffortRung(name, 8, ((32, 8), (64, 8)), 8, 400),
        union_size=size, newly_unique=0, best_proxy=best, uf_fingerprint="uf",
        uf_completion=uf,
        up_fingerprints={p: up_fp for p in _PRICES},
        up_best_proxy={p: up_proxy for p in _PRICES},
        affordable_counts={p: afford for p in _PRICES},
        per_path_best={"a": best, "b": best - spread},
        per_path_new={"a": 0, "b": 0}, runtime_s=1.0)


# ---------------------------------------------------------------------------
# The ladder is declared before results
# ---------------------------------------------------------------------------


def test_the_effort_ladder_is_predeclared_and_pinned():
    """A later edit must break this test, not quietly retrofit the ladder."""
    assert ladder_fingerprint() == "0fddb000228e1276"
    assert "before any rung of this experiment was run" in LADDER_DECLARED_AT


def test_the_ladder_is_bounded_and_ordered():
    mults = [r.multiple for r in EFFORT_LADDER]
    assert mults == sorted(mults)
    assert mults == [1, 2, 4, 8]
    assert len(EFFORT_LADDER) == 4, "the ladder must be bounded"


def test_every_rung_runs_at_least_two_independent_generation_paths():
    for r in EFFORT_LADDER:
        assert r.n_paths >= 2
        assert len(set(r.paths)) == r.n_paths


def test_generation_paths_vary_the_settings_that_actually_change_the_search():
    """`proxy_seed` provably does not move this search, so it is not used."""
    for r in EFFORT_LADDER:
        beams = {b for b, _ in r.paths}
        buckets = {sb for _, sb in r.paths}
        assert len(beams) > 1 or len(buckets) > 1


# ---------------------------------------------------------------------------
# Independent seeds
# ---------------------------------------------------------------------------


def test_identical_selection_and_holdout_seeds_are_refused():
    with pytest.raises(SeedMisuse, match="chose the winner"):
        assert_independent_seeds(555_000_111, 555_000_111)


def test_adjacent_seeds_are_refused_as_overlapping():
    with pytest.raises(SeedMisuse, match="not independent"):
        assert_independent_seeds(555_000_111, 555_000_600)


def test_the_production_seeds_are_independent():
    assert_independent_seeds(555_000_111, 917_324_011)


# ---------------------------------------------------------------------------
# Economic equivalence is defined, and is not a loophole
# ---------------------------------------------------------------------------


def test_economic_profile_covers_every_documented_quantity(board, sim, eq_kw):
    c = Completion(added=(1,), roster=tuple(range(15)), added_cost=40,
                   proxy=100.0)
    p = economic_profile(c, **eq_kw)
    for field in ("total_cost", "strength_bucket", "feasible",
                  "holds_candidate", "budget_left", "slots_left"):
        assert hasattr(p, field)


def test_identical_completions_are_equivalent(eq_kw):
    c = Completion(added=(1,), roster=tuple(range(15)), added_cost=40,
                   proxy=100.0)
    assert economically_equivalent(c, c, **eq_kw)


def test_a_different_cost_is_not_economically_equivalent(eq_kw):
    a = Completion(added=(1,), roster=tuple(range(15)), added_cost=40,
                   proxy=100.0)
    b = dataclasses.replace(a, added_cost=55)
    assert not economically_equivalent(a, b, **eq_kw)


def test_a_materially_weaker_roster_is_not_economically_equivalent(eq_kw):
    """Different players, same cost, three points weaker. Not the same decision."""
    a = Completion(added=(1,), roster=tuple(range(15)), added_cost=40,
                   proxy=100.0)
    b = Completion(added=(2,), roster=tuple(range(2, 17)), added_cost=40,
                   proxy=97.0)
    assert not economically_equivalent(a, b, **eq_kw)


def test_the_same_roster_at_the_same_cost_is_equivalent_by_fingerprint(eq_kw):
    """Strength is a function of the roster, so identity settles it early."""
    a = Completion(added=(1,), roster=tuple(range(15)), added_cost=40,
                   proxy=100.0)
    b = Completion(added=(1,), roster=tuple(reversed(range(15))),
                   added_cost=40, proxy=100.0)
    assert economically_equivalent(a, b, **eq_kw)


def test_equivalence_is_not_satisfied_by_matching_cost_alone(eq_kw):
    """Same price, different strength, must NOT pass."""
    a = Completion(added=(1,), roster=tuple(range(15)), added_cost=40,
                   proxy=100.0)
    b = Completion(added=(2,), roster=tuple(range(1, 16)), added_cost=40,
                   proxy=90.0)
    assert a.added_cost == b.added_cost
    assert not economically_equivalent(a, b, **eq_kw)


# ---------------------------------------------------------------------------
# The stopping rule
# ---------------------------------------------------------------------------


def test_convergence_cannot_be_declared_from_one_rung(eq_kw):
    checks, failures = evaluate_stopping_rule(
        [_rung("1x")], evaluated_prices=_PRICES, eq_kw=eq_kw)
    assert failures
    assert "single effort level" in failures[0]


def test_a_report_with_one_rung_is_underconverged(eq_kw):
    rep = ConvergenceReport((_rung("1x"),), {}, (), ())
    assert rep.status == QB_UNION_UNDERCONVERGED


def test_two_stable_rungs_converge(eq_kw):
    c = Completion(added=(1,), roster=tuple(range(15)), added_cost=40,
                   proxy=98.0)
    a, b = _rung("4x", uf=c), _rung("8x", uf=c)
    checks, failures = evaluate_stopping_rule(
        [a, b], evaluated_prices=_PRICES, eq_kw=eq_kw)
    assert not failures
    assert all(checks.values())
    assert ConvergenceReport((a, b), checks, tuple(failures), ()).converged


def test_the_observed_0690_movement_fails_convergence(eq_kw):
    """The exact defect this branch exists to resolve."""
    c = Completion(added=(1,), roster=tuple(range(15)), added_cost=40,
                   proxy=98.0)
    a = _rung("4x", best=97.4508, uf=c)
    b = _rung("8x", best=98.1406, uf=c)
    assert round(b.best_proxy - a.best_proxy, 4) == 0.6898
    checks, failures = evaluate_stopping_rule(
        [a, b], evaluated_prices=_PRICES, eq_kw=eq_kw)
    assert checks["uf_objective_within_tolerance"] is False
    assert any("beyond the 0.25 tolerance" in f for f in failures)


def test_raw_union_size_stability_alone_is_not_convergence(eq_kw):
    """Same size, still improving. Size is not the criterion."""
    c = Completion(added=(1,), roster=tuple(range(15)), added_cost=40,
                   proxy=98.0)
    a = _rung("4x", size=1749, best=97.0, uf=c)
    b = _rung("8x", size=1749, best=98.0, uf=c)
    assert a.union_size == b.union_size
    checks, failures = evaluate_stopping_rule(
        [a, b], evaluated_prices=_PRICES, eq_kw=eq_kw)
    assert failures, "identical union size must not imply convergence"


def test_material_generation_path_disagreement_fails_convergence(eq_kw):
    c = Completion(added=(1,), roster=tuple(range(15)), added_cost=40,
                   proxy=98.0)
    a, b = _rung("4x", uf=c), _rung("8x", uf=c, spread=0.5276)
    checks, failures = evaluate_stopping_rule(
        [a, b], evaluated_prices=_PRICES, eq_kw=eq_kw)
    assert checks["generation_paths_agree"] is False
    assert any("generation paths disagree" in f for f in failures)


def test_an_improving_evaluated_price_fails_convergence(eq_kw):
    c = Completion(added=(1,), roster=tuple(range(15)), added_cost=40,
                   proxy=98.0)
    a = _rung("4x", uf=c, up_proxy=93.80)
    b = _rung("8x", uf=c, up_proxy=94.69)
    checks, failures = evaluate_stopping_rule(
        [a, b], evaluated_prices=_PRICES, eq_kw=eq_kw)
    assert checks["no_evaluated_price_improved_beyond_tolerance"] is False


def test_a_changed_up_selection_fails_convergence(eq_kw):
    c = Completion(added=(1,), roster=tuple(range(15)), added_cost=40,
                   proxy=98.0)
    a = _rung("4x", uf=c, up_fp="aaa")
    b = _rung("8x", uf=c, up_fp="bbb")
    checks, failures = evaluate_stopping_rule(
        [a, b], evaluated_prices=_PRICES, eq_kw=eq_kw)
    assert checks["up_selection_stable"] is False


def test_a_feasibility_change_at_a_tested_price_fails_convergence(eq_kw):
    c = Completion(added=(1,), roster=tuple(range(15)), added_cost=40,
                   proxy=98.0)
    a = _rung("4x", uf=c, afford=0)
    b = _rung("8x", uf=c, afford=7)
    checks, failures = evaluate_stopping_rule(
        [a, b], evaluated_prices=_PRICES, eq_kw=eq_kw)
    assert checks["affordability_unchanged_at_tested_prices"] is False


def test_every_clause_must_pass_for_convergence(eq_kw):
    """One failure is enough to withhold the verdict."""
    c = Completion(added=(1,), roster=tuple(range(15)), added_cost=40,
                   proxy=98.0)
    a, b = _rung("4x", uf=c), _rung("8x", uf=c, up_fp="different")
    checks, failures = evaluate_stopping_rule(
        [a, b], evaluated_prices=_PRICES, eq_kw=eq_kw)
    rep = ConvergenceReport((a, b), checks, tuple(failures), ())
    assert not rep.converged
    assert rep.status == QB_UNION_UNDERCONVERGED


# ---------------------------------------------------------------------------
# An underconverged union may not spend CE or quote a bid
# ---------------------------------------------------------------------------


def test_an_underconverged_union_may_not_quote_a_bid():
    rep = ConvergenceReport((_rung("4x"), _rung("8x")), {},
                            ("still improving",), ())
    assert rep.status == QB_UNION_UNDERCONVERGED
    assert not rep.may_quote_bid


def test_full_ce_evaluation_is_gated_on_convergence():
    bad = ConvergenceReport((_rung("4x"), _rung("8x")), {}, ("nope",), ())
    good = ConvergenceReport((_rung("4x"), _rung("8x")), {}, (), ())
    assert not bad.may_evaluate_ce
    assert good.may_evaluate_ce


def test_the_convergence_scope_is_stated_and_bounded():
    rep = ConvergenceReport((_rung("4x"), _rung("8x")), {}, (), ())
    assert "NOT a claim that no better roster exists" in rep.scope


def test_the_report_dict_carries_the_ladder_identity():
    d = ConvergenceReport((_rung("4x"), _rung("8x")), {}, (), ()).to_dict()
    assert d["ladder_fingerprint"] == ladder_fingerprint()
    assert d["tolerance_weekly_points"] == PROXY_TOLERANCE


# ---------------------------------------------------------------------------
# The real ladder, on the demo board
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_run(board, sim):
    s, cid = sim
    st = s.state
    px = ProxyEvaluator(st.pool, st.settings, 16, 7)
    base = CompletionSettings(
        beam_width=32, candidate_pool=30, proxy_candidates=120, finalists=4,
        max_candidates=200, proxy_reps=16, selection_sims=300,
        evaluation_sims=400, selection_seed=555_000_111,
        evaluation_seed=917_324_011, rival_selection="proxy")
    ladder = EFFORT_LADDER[:2]
    return run_convergence_ladder(
        st, board.cast, board.costs["base"], cid, base=base, proxy=px,
        evaluated_prices=(10, 20), support_prices=(1, 10, 20), ladder=ladder)


def test_the_union_accumulates_monotonically(real_run):
    sizes = [r.union_size for r in real_run.rungs]
    assert sizes == sorted(sizes)


def test_no_rung_ever_shrinks_the_union(real_run):
    for a, b in zip(real_run.rungs, real_run.rungs[1:]):
        assert b.union_size >= a.union_size
        assert b.newly_unique >= 0


def test_the_accumulated_objective_never_falls(real_run):
    """Best-so-far over the union is non-decreasing by construction."""
    best = [r.best_proxy for r in real_run.rungs]
    assert best == sorted(best)


def test_independent_paths_all_feed_one_union(real_run):
    for r in real_run.rungs:
        assert len(r.per_path_best) == r.rung.n_paths
        assert sum(r.per_path_new.values()) == r.newly_unique


def test_every_union_member_is_unique_by_fingerprint(real_run):
    keys = [completion_fingerprint(c) for c in real_run.union]
    assert len(keys) == len(set(keys))


def test_the_run_reports_a_status_and_a_scope(real_run):
    assert real_run.status in (QB_UNION_CONVERGED, QB_UNION_UNDERCONVERGED)
    assert real_run.to_dict()["convergence_scope"]
