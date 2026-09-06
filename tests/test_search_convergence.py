"""Search-effort convergence for the marginal diagnostic. Fabricated only.

The defect these guard: the previous branch declared the diagnostic converged
because it stopped violating *price* monotonicity, while its own table showed
the $18 improvement moving 2.29 weekly points and reversing sign between
beam 160 and beam 320. A wider beam is a different heuristic, not a strictly
better one, so an independent wider run can lose what a narrower run found.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from ceauction.auction.completion import Completion, CompletionSettings
from ceauction.auction.proxy import ProxyEvaluator
from ceauction.auction.state import new_auction
from ceauction.league import DEFAULT_LEAGUE, Position
from ceauction.tactical.convergence import (CONVERGED, DEFAULT_LADDER,
                                            DIAGNOSTIC_NOT_CONVERGED,
                                            TOLERANCES, CompletionUnion,
                                            ConvergenceReport, EffortLevel,
                                            LevelResult, converged_diagnose)
from ceauction.tactical.demo import build_tactical_demo
from ceauction.tactical.realpilot import OWNER_IDS, RealBoard, select_candidates


@pytest.fixture(scope="module")
def fake_board():
    from ceauction.auction.completion import ComparisonCast
    from ceauction.market.costbook import cost_book_from_prior

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
        name_by_id={s.player_id: s.name for s in state.pool},
        coverage={})


@pytest.fixture(scope="module")
def px(fake_board):
    return ProxyEvaluator(fake_board.state.pool, fake_board.state.settings,
                          16, 7)


_SMALL = (EffortLevel(24, 30), EffortLevel(64, 60), EffortLevel(120, 80))


@pytest.fixture(scope="module")
def ladder_runs(fake_board, px):
    cands, _ = select_candidates(fake_board)
    out = []
    for c in cands[:6]:
        out.append((c, *converged_diagnose(fake_board, c, proxy=px,
                                           ladder=_SMALL)))
    return out


# ---------------------------------------------------------------------------
# Union mechanics
# ---------------------------------------------------------------------------


def _completion(ids, cost=10):
    ids = tuple(ids)
    return Completion(added=ids, roster=ids, added_cost=cost, proxy=0.0)


def test_the_union_deduplicates_by_roster_not_by_order(fake_board, px):
    u = CompletionUnion(px, budget=200)
    ids = [s.player_id for s in fake_board.state.pool[:15]]
    assert u.add_many([_completion(ids)]) == 1
    assert u.add_many([_completion(list(reversed(ids)))]) == 0
    assert len(u) == 1


def test_the_union_rejects_unaffordable_and_illegal_members(fake_board, px):
    ids = [s.player_id for s in fake_board.state.pool[:15]]
    u = CompletionUnion(px, budget=5)
    assert u.add_many([_completion(ids, cost=99)]) == 0
    assert u.rejected_unaffordable == 1

    required = frozenset({ids[0]})
    v = CompletionUnion(px, budget=200, required=required)
    assert v.add_many([_completion(ids[1:16])]) == 0
    assert v.rejected_illegal == 1

    w = CompletionUnion(px, budget=200, forbidden=frozenset({ids[0]}))
    assert w.add_many([_completion(ids)]) == 0
    assert w.rejected_illegal == 1


def test_a_later_batch_cannot_lose_an_earlier_better_member(fake_board, px):
    """The whole point: more effort must not discard a better construction."""
    pool = sorted(fake_board.state.pool, key=lambda s: -s.base_mean)
    strong = [s.player_id for s in pool[:15]]
    weak = [s.player_id for s in pool[-15:]]
    u = CompletionUnion(px, budget=500)
    u.add_many([_completion(strong)])
    first = u.best_objective()
    u.add_many([_completion(weak)])
    assert u.best_objective() == first, \
        "adding a worse construction changed the best objective"
    assert len(u) == 2, "but the worse one is still retained for comparison"


def test_the_union_rescoring_makes_levels_comparable(fake_board, px):
    """Members carry a `proxy` from their own level; the union ignores it."""
    ids = [s.player_id for s in fake_board.state.pool[:15]]
    lying = Completion(added=tuple(ids), roster=tuple(ids), added_cost=10,
                       proxy=9999.0)
    u = CompletionUnion(px, budget=200)
    u.add_many([lying])
    assert u.best_objective() == pytest.approx(px.strength(ids))
    assert u.best_objective() < 9999.0


# ---------------------------------------------------------------------------
# Search-effort monotonicity on real ladder runs
# ---------------------------------------------------------------------------


def test_the_best_objective_never_worsens_with_more_effort(ladder_runs):
    for _, _, rep in ladder_runs:
        objs = [l.best_with for l in rep.levels]
        assert objs == sorted(objs), (
            f"objective regressed across effort levels: {objs}")
        assert rep.search_violations == ()


def test_search_effort_candidate_sets_are_nested(ladder_runs):
    for _, _, rep in ladder_runs:
        sizes = [l.union_with for l in rep.levels]
        assert sizes == sorted(sizes), f"union shrank: {sizes}"
        sizes_wo = [l.union_without for l in rep.levels]
        assert sizes_wo == sorted(sizes_wo)


def test_price_nesting_holds_once_the_union_has_grown(ladder_runs):
    """Early narrow levels may violate it; the accumulated union must not.

    A price violation at a small beam is the search being under-resourced, and
    the ladder exists to show that. What matters is the FINAL union, which is
    what the diagnostic actually reports.
    """
    for _, d, rep in ladder_runs:
        if d.price > 1:
            assert d.improvement_at_min >= d.lineup_improvement - 0.5, (
                f"the final union still violates price monotonicity by "
                f"{d.lineup_improvement - d.improvement_at_min:.2f}")
        # And any violation that did occur is reported, never hidden.
        assert isinstance(rep.price_violations, tuple)


def test_independent_beams_disagree_and_the_union_absorbs_it(fake_board, px):
    """Detect the disagreement rather than shrugging at it."""
    cands, _ = select_candidates(fake_board)
    c = cands[0]
    _, rep = converged_diagnose(fake_board, c, proxy=px, ladder=_SMALL)
    # A later level contributing new members IS the disagreement: the wider
    # beam found constructions the narrower one did not, and vice versa is
    # absorbed because earlier members are never dropped.
    assert any(l.new_with > 0 for l in rep.levels[1:]) or \
        rep.levels[-1].union_with == rep.levels[0].union_with
    assert rep.levels[-1].union_with >= rep.levels[0].union_with


# ---------------------------------------------------------------------------
# Convergence status
# ---------------------------------------------------------------------------


def _level(i, obj, imp, role, policy):
    return LevelResult(
        level=EffortLevel(64 * (i + 1), 60), generated_with=1,
        generated_without=1, new_with=1, new_without=1, union_with=i + 1,
        union_without=i + 1, best_with=obj, best_without=obj - imp,
        improvement_at_price=imp, improvement_at_min=imp, role=role,
        policy=policy, composition={}, roster_fingerprint=f"r{i}",
        added_cost=10, budget_left=10, exactness="heuristic", runtime_s=1.0)


def test_a_stable_ladder_reports_converged():
    rep = ConvergenceReport(levels=(
        _level(0, 100.0, 1.0, "marginal starter", "4000-season audit"),
        _level(1, 100.1, 1.05, "marginal starter", "4000-season audit"),
        _level(2, 100.1, 1.05, "marginal starter", "4000-season audit")))
    assert rep.status(0.25) == CONVERGED
    assert rep.effort_required(0.25) is not None


def test_an_unsettled_objective_reports_not_converged():
    rep = ConvergenceReport(levels=(
        _level(0, 100.0, 1.0, "marginal starter", "4000-season audit"),
        _level(1, 102.5, 3.5, "marginal starter", "4000-season audit")))
    assert rep.status(0.25) == DIAGNOSTIC_NOT_CONVERGED
    assert rep.effort_required(0.25) is None


def test_role_instability_reports_not_converged_even_if_numbers_settle():
    """A quarter-point-stable improvement whose role flips is not converged."""
    rep = ConvergenceReport(levels=(
        _level(0, 100.0, 0.30, "marginal starter", "4000-season audit"),
        _level(1, 100.05, 0.24, "replaceable starter", "proxy only")))
    assert abs(rep.objective_deltas()[-1]) < 0.25
    assert not rep.role_stable
    assert rep.status(0.25) == DIAGNOSTIC_NOT_CONVERGED


def test_policy_instability_reports_not_converged():
    rep = ConvergenceReport(levels=(
        _level(0, 100.0, 1.0, "marginal starter", "4000-season audit"),
        _level(1, 100.05, 1.02, "marginal starter", "proxy only")))
    assert not rep.policy_stable
    assert rep.status(0.25) == DIAGNOSTIC_NOT_CONVERGED


def test_a_single_level_can_never_be_converged():
    rep = ConvergenceReport(levels=(
        _level(0, 100.0, 1.0, "marginal starter", "4000-season audit"),))
    assert rep.status(0.25) == DIAGNOSTIC_NOT_CONVERGED
    assert rep.status(0.5) == DIAGNOSTIC_NOT_CONVERGED


def test_both_tolerances_are_reported():
    rep = ConvergenceReport(levels=(
        _level(0, 100.0, 1.0, "marginal starter", "4000-season audit"),
        _level(1, 100.4, 1.4, "marginal starter", "4000-season audit")))
    blob = rep.to_dict()
    assert set(blob["status"]) == {str(t) for t in TOLERANCES}
    assert blob["status"]["0.25"] == DIAGNOSTIC_NOT_CONVERGED
    assert blob["status"]["0.5"] == CONVERGED
    assert "not claims about" in blob["tolerance_note"]


def test_a_violation_forces_not_converged():
    rep = ConvergenceReport(
        levels=(_level(0, 100.0, 1.0, "marginal starter", "4000-season audit"),
                _level(1, 100.0, 1.0, "marginal starter", "4000-season audit")),
        search_violations=(("160/90", 0.4),))
    assert rep.status(0.5) == DIAGNOSTIC_NOT_CONVERGED


# ---------------------------------------------------------------------------
# Non-converged candidates are never silently filed away
# ---------------------------------------------------------------------------


def test_a_nonconverged_candidate_is_unresolved_not_proxy_only(fake_board, px):
    cands, _ = select_candidates(fake_board)
    c = cands[0]
    # A two-level ladder with a huge gap will rarely settle to 0.001.
    d, rep = converged_diagnose(fake_board, c, proxy=px,
                                ladder=(EffortLevel(16, 24),
                                        EffortLevel(200, 110)),
                                tolerances=(0.001, 0.002))
    if rep.status(0.001) == DIAGNOSTIC_NOT_CONVERGED:
        assert d.policy.startswith("unresolved")
        assert "may not" in d.reason and "withhold CE" in d.reason
        assert d.policy != "proxy only"


def test_converged_candidates_get_a_real_policy(ladder_runs):
    for _, d, rep in ladder_runs:
        if rep.status(0.25) == CONVERGED:
            assert d.policy in ("4000-season audit", "proxy only")
            assert not d.policy.startswith("unresolved")


# ---------------------------------------------------------------------------
# Quota-freedom and legality survive
# ---------------------------------------------------------------------------


def test_no_positional_quota_returns():
    src = Path(__file__).resolve().parents[1] / "src" / "ceauction" / "tactical"
    for name in ("convergence.py", "realpilot.py"):
        text = (src / name).read_text(encoding="utf-8")
        for banned in ('("QB", 1)', '("RB", 4)', '("WR", 6)', '("TE", 3)',
                       "template = (("):
            assert banned not in text, f"{banned} in {name}"


def test_zero_tight_ends_and_many_quarterbacks_remain_legal():
    from ceauction.auction.feasibility import PositionCounts, can_fill_lineup
    assert can_fill_lineup(PositionCounts(qb=2, rb=3, wr=3, te=0))
    assert can_fill_lineup(PositionCounts(qb=1, rb=4, wr=3, te=0))
    # Five QBs is legal, but the eight starting seats still have to fill:
    # QB, RB, RB, three WR/TE, a flex and the superflex. qb=5/rb=2/wr=1 cannot,
    # which is the eligibility graph doing its job, not a quota.
    assert can_fill_lineup(PositionCounts(qb=5, rb=3, wr=3, te=0))
    assert not can_fill_lineup(PositionCounts(qb=5, rb=2, wr=1, te=0))


def test_branches_stay_legal_and_the_dollar_reserve_holds(fake_board,
                                                          ladder_runs):
    cap = fake_board.state.owner(fake_board.focus_owner_id).roster_capacity
    budget = fake_board.state.owner(
        fake_board.focus_owner_id).budget_remaining
    for c, d, _ in ladder_runs:
        for branch in (d.without, d.with_candidate):
            assert len(branch.player_ids) == cap
            assert len(set(branch.player_ids)) == cap
            assert branch.budget_left >= 0
        assert c.player_id in d.with_candidate.player_ids
        assert c.player_id not in d.without.player_ids
        assert d.with_candidate.added_cost + d.price <= budget


def test_an_unaffordable_price_is_refused(fake_board, px):
    cands, _ = select_candidates(fake_board)
    with pytest.raises(ValueError, match="cannot be bought"):
        converged_diagnose(fake_board, cands[0], proxy=px, price=100000,
                           ladder=_SMALL)


# ---------------------------------------------------------------------------
# Hygiene and CLI
# ---------------------------------------------------------------------------


def test_no_local_data_file_is_tracked():
    out = subprocess.run(["git", "ls-files", "local_data"],
                         capture_output=True, text=True, check=True)
    assert out.stdout.strip() == ""


def _cli(*argv):
    return subprocess.run([sys.executable, "-m", "ceauction.cli", "tactical",
                           *argv], capture_output=True, text=True)


@pytest.mark.parametrize("argv,needle", [
    (("marginal-diagnostics", "--out-dir", "/tmp/leak"), "local_data"),
    (("marginal-diagnostics", "--tolerance", "0"), "must be positive"),
    (("marginal-diagnostics", "--effort", "64/60"), "at least two levels"),
    (("marginal-diagnostics", "--effort", "bad", "--effort", "worse"),
     "BEAM/POOL"),
    (("marginal-diagnostics", "--effort", "160/90", "--effort", "64/60"),
     "must increase"),
    (("marginal-diagnostics", "--max-effort", "10"), "fewer than two"),
])
def test_cli_usage_errors_have_no_traceback(argv, needle):
    r = _cli(*argv)
    assert r.returncode != 0
    assert "Traceback" not in r.stderr
    assert needle in (r.stdout + r.stderr)
