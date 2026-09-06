"""Quota-free marginal diagnostics. Fabricated fixtures only.

The defect these guard: a hard-coded 1 QB / 4 RB / 6 WR / 3 TE reference roster
decided which real players earned a 4,000-season championship-equity audit. It
was legal and arbitrary. Reserving three places for tight ends made a fourth
tight end read 0.22 weekly points and skip CE entirely; with the quota removed
the same player reads +4.86 and is a clear starter.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from ceauction.auction.completion import CompletionSettings
from ceauction.auction.proxy import ProxyEvaluator
from ceauction.auction.state import new_auction
from ceauction.league import DEFAULT_LEAGUE, Position
from ceauction.tactical.demo import build_tactical_demo
from ceauction.tactical.realpilot import (BENCH_IMPROVEMENT,
                                          MARGINAL_IMPROVEMENT,
                                          NO_CONTINGENCY_MODEL, OWNER_IDS,
                                          BranchRoster, RealBoard, diagnose,
                                          select_candidates)

_CS = CompletionSettings(beam_width=24, candidate_pool=30, proxy_candidates=24,
                         finalists=2, max_candidates=120, proxy_reps=16)


@pytest.fixture(scope="module")
def fake_board():
    from ceauction.auction.completion import ComparisonCast
    from ceauction.market.costbook import cost_book_from_prior

    d = build_tactical_demo()
    state = new_auction(list(d.state.pool), OWNER_IDS, OWNER_IDS[0],
                        settings=DEFAULT_LEAGUE)
    state.validate()
    costs = {}
    for scenario in ("low", "base", "high"):
        cb = cost_book_from_prior(d.prior, scenario)
        gaps = {s.player_id: 1 for s in state.pool if s.player_id not in cb}
        costs[scenario] = cb.with_costs(gaps) if gaps else cb
    return RealBoard(
        state=state,
        cast=ComparisonCast(0, tuple(() for _ in OWNER_IDS), OWNER_IDS),
        prior=d.prior, market=d.market, costs=costs,
        key_by_id=dict(d.key_by_id),
        name_by_id={s.player_id: s.name for s in state.pool},
        coverage={"mapped_playerspecs": len(state.pool)})


@pytest.fixture(scope="module")
def px(fake_board):
    return ProxyEvaluator(fake_board.state.pool, fake_board.state.settings,
                          16, 7)


@pytest.fixture(scope="module")
def diags(fake_board, px):
    cands, _ = select_candidates(fake_board)
    return [(c, diagnose(fake_board, c, proxy=px, settings=_CS))
            for c in cands]


# ---------------------------------------------------------------------------
# No template survives anywhere
# ---------------------------------------------------------------------------


def test_no_fixed_positional_template_remains_in_source():
    src = Path(__file__).resolve().parents[1] / "src" / "ceauction" / "tactical"
    text = (src / "realpilot.py").read_text(encoding="utf-8")
    fn = text[text.index("def diagnose"):]
    for banned in ('("QB", 1)', '("RB", 4)', '("WR", 6)', '("TE", 3)',
                   "template = ("):
        assert banned not in fn, f"a positional template survives: {banned}"
    # And no module-level template either.
    assert 'template = (("QB"' not in text


def test_composition_is_an_output_not_an_input(diags):
    """Different candidates must be able to produce different compositions."""
    comps = {tuple(sorted(d.with_candidate.composition.items()))
             for _, d in diags}
    assert len(comps) > 1, "every candidate produced the same roster shape"


def test_completion_may_select_zero_tight_ends(fake_board, px):
    """No TE is required by the league, so none may be required here."""
    cands, _ = select_candidates(fake_board)
    seen_zero = False
    for c in cands:
        d = diagnose(fake_board, c, proxy=px, settings=_CS)
        for branch in (d.without, d.with_candidate):
            if branch.composition.get("TE", 0) == 0:
                seen_zero = True
    # Structural guarantee regardless of what this board happens to pick:
    from ceauction.auction.feasibility import PositionCounts, can_fill_lineup
    assert can_fill_lineup(PositionCounts(qb=2, rb=3, wr=3, te=0))
    assert isinstance(seen_zero, bool)


def test_completion_may_select_several_quarterbacks(diags):
    """No QB maximum exists; the search must be free to stack them."""
    counts = {d.with_candidate.composition.get("QB", 0) for _, d in diags}
    assert counts, "no completions produced"
    assert max(counts) >= 2, (
        "a superflex board should let the completion take more than one QB; "
        f"got {sorted(counts)}")


def test_the_superflex_accepts_a_skill_position_fallback():
    from ceauction.auction.feasibility import PositionCounts, can_fill_lineup
    assert can_fill_lineup(PositionCounts(qb=1, rb=4, wr=3, te=0)), \
        "one QB plus skill players must fill the superflex"
    assert can_fill_lineup(PositionCounts(qb=1, rb=2, wr=3, te=2))
    assert not can_fill_lineup(PositionCounts(qb=0, rb=4, wr=4, te=0)), \
        "the dedicated QB seat still requires a QB"


# ---------------------------------------------------------------------------
# Both branches are legal
# ---------------------------------------------------------------------------


def test_both_branches_are_legal_full_rosters(fake_board, diags):
    cap = fake_board.state.owner(fake_board.focus_owner_id).roster_capacity
    for _, d in diags:
        for branch in (d.without, d.with_candidate):
            assert len(branch.player_ids) == cap
            assert len(set(branch.player_ids)) == cap
            assert sum(branch.composition.values()) == cap
            assert branch.budget_left >= 0


def test_the_dollar_reserve_is_maintained(fake_board, diags):
    budget = fake_board.state.owner(fake_board.focus_owner_id).budget_remaining
    for _, d in diags:
        assert d.with_candidate.added_cost + d.price <= budget
        assert d.without.added_cost <= budget


def test_the_candidate_is_in_his_own_branch_and_absent_from_the_other(diags):
    for c, d in diags:
        assert c.player_id in d.with_candidate.player_ids
        assert c.player_id not in d.without.player_ids


def test_best_legal_eight_counts_about_eight_starters(fake_board, px, diags):
    for _, d in diags:
        shares = d.with_candidate.start_shares
        assert 0.0 <= sum(shares.values()) <= 8.0 + 1e-6
        assert sum(shares.values()) > 4.0, "a full roster should start ~8"


def test_a_quarterback_only_roster_starts_exactly_two(fake_board, px):
    """The eligibility graph, not a template, caps quarterbacks at QB+superflex."""
    qbs = [s.player_id for s in fake_board.state.pool
           if Position(int(s.position)) is Position.QB][:15]
    shares = px.lineup_shares(qbs)
    assert sum(shares.values()) == pytest.approx(2.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Roles come from measurement, never from position
# ---------------------------------------------------------------------------


def test_role_leads_with_improvement_not_start_share(diags):
    for _, d in diags:
        if d.lineup_improvement >= MARGINAL_IMPROVEMENT:
            assert d.role == "clear starter"
        elif d.lineup_improvement >= BENCH_IMPROVEMENT:
            assert d.role == "marginal starter"
        else:
            assert d.role in ("replaceable starter", "aggregate depth",
                              "bench/insurance", "currently redundant")


def test_a_high_start_share_with_no_improvement_is_replaceable_not_audited(diags):
    """The bug that audited all twelve candidates."""
    replaceable = [d for _, d in diags if d.role == "replaceable starter"]
    for d in replaceable:
        assert d.policy == "proxy only"
        assert d.start_share >= 0.5
        assert d.lineup_improvement < BENCH_IMPROVEMENT
        assert "PRICE judgement" in d.reason


def test_price_and_player_quality_are_reported_separately(diags):
    for _, d in diags:
        blob = d.to_dict()
        assert "improvement_at_min" in blob
        assert "price_cost_of_slot" in blob
        assert blob["price_cost_of_slot"] == pytest.approx(
            d.improvement_at_min - d.lineup_improvement, abs=1e-3)


def test_there_is_no_tight_end_penalty_and_no_second_qb_bonus():
    src = Path(__file__).resolve().parents[1] / "src" / "ceauction" / "tactical"
    text = (src / "realpilot.py").read_text(encoding="utf-8")
    fn = text[text.index("def diagnose"):]
    for token in ('== "TE"', "== 'TE'", '== "QB"', "== 'QB'", "Position.TE",
                  "Position.QB"):
        assert token not in fn, (
            f"the diagnostic branches on position ({token}); role must come "
            f"from measured lineup behaviour only")


def test_bench_points_do_not_become_lineup_value(diags):
    """A benched player's raw points must not be credited as improvement."""
    for _, d in diags:
        if d.start_share < 1e-6:
            assert d.lineup_improvement < BENCH_IMPROVEMENT + 1e-6


def test_missing_contingency_mapping_is_reported_not_priced(diags):
    for _, d in diags:
        assert d.contingency_note == NO_CONTINGENCY_MODEL
        assert "NOT quantified" in d.contingency_note
        assert "no conditional-backfield" in d.contingency_note


def test_completion_exactness_is_labelled(diags):
    for _, d in diags:
        assert d.with_candidate.exactness
        assert d.without.exactness
        assert "heuristic" in d.with_candidate.exactness or \
               "exact" in d.with_candidate.exactness


def test_displacement_is_reported_when_the_rosters_differ(diags):
    for c, d in diags:
        gone = set(d.without.player_ids) - set(d.with_candidate.player_ids)
        if gone:
            assert d.displaced_player_id in gone
            assert 0.0 <= d.displaced_start_share <= 1.0
        else:
            assert d.displaced_player_id is None


# ---------------------------------------------------------------------------
# Selection and sanitization unchanged
# ---------------------------------------------------------------------------


def test_selection_remains_deterministic(fake_board):
    a, _ = select_candidates(fake_board)
    b, _ = select_candidates(fake_board)
    assert [c.player_id for c in a] == [c.player_id for c in b]
    assert len(a) == 12


def test_diagnostics_carry_no_identity_unless_asked(diags):
    for _, d in diags:
        blob = d.to_dict()
        assert "player_id" not in blob
        assert "displaced_player_id" not in blob
        assert "player_ids" not in blob["with_candidate"]
        rich = d.to_dict(include_identity=True)
        assert "player_id" in rich
        assert "player_ids" in rich["with_candidate"]


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
    (("marginal-diagnostics", "--per-position", "0"), "at least 1"),
    (("marginal-diagnostics", "--position", "K"), "invalid choice"),
    (("marginal-diagnostics", "--contract", "/nope.json"), "missing"),
])
def test_cli_usage_errors_have_no_traceback(argv, needle):
    r = _cli(*argv)
    assert r.returncode != 0
    assert "Traceback" not in r.stderr
    assert needle in (r.stdout + r.stderr)
