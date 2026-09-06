"""Possession / payment / denial decomposition. Fabricated fixtures only.

The question this answers: the converged diagnostic found positive weekly
lineup improvement for all four audited real candidates, and two of them had
negative tactical CE. Improving our lineup is not the same as being worth
buying at the market price, above all when passing may make an opponent
overpay. These tests guard the instrument that separates those.
"""

from __future__ import annotations

import math
import statistics
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from ceauction.auction.completion import CompletionSettings
from ceauction.auction.proxy import ProxyEvaluator
from ceauction.auction.state import new_auction
from ceauction.league import DEFAULT_LEAGUE
from ceauction.tactical.board import BoardSettings
from ceauction.tactical.decompose import (ALTERNATE_ORDER, BRANCHES,
                                          COMPONENT_ORDER,
                                          AnalyticalBranchMisuse,
                                          Decomposition, DrawDecomposition,
                                          assert_not_a_recipient,
                                          build_branch_state, decompose)
from ceauction.tactical.demo import build_tactical_demo
from ceauction.tactical.ensemble import balanced_schedule
from ceauction.tactical.realpilot import OWNER_IDS, RealBoard, select_candidates

_CS = CompletionSettings(beam_width=24, candidate_pool=30, proxy_candidates=24,
                         finalists=2, max_candidates=120, proxy_reps=16,
                         selection_sims=300, evaluation_sims=400,
                         rival_selection="proxy")
_BS = BoardSettings(pool_depth=200, max_allocations=200)


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
        name_by_id={s.player_id: s.name for s in state.pool}, coverage={})


@pytest.fixture(scope="module")
def cid(fake_board):
    return select_candidates(fake_board)[0][0].player_id


@pytest.fixture(scope="module")
def run(fake_board, cid):
    px = ProxyEvaluator(fake_board.state.pool, fake_board.state.settings, 16, 7)
    return decompose(
        fake_board.state, fake_board.cast, fake_board.costs["base"], cid,
        focus_price=12, rival="Team05", rival_price=15, position="QB",
        lineup_improvement=1.5, role="marginal starter",
        draws=balanced_schedule(12, 3), board=_BS, completion=_CS,
        market=fake_board.market, key_by_id=fake_board.key_by_id, proxy=px,
        holdout_sims=400)


# ---------------------------------------------------------------------------
# Branch construction
# ---------------------------------------------------------------------------


def test_every_branch_places_the_candidate_correctly(fake_board, cid):
    st = fake_board.state
    focus = st.focus_owner_id
    for key, spec in BRANCHES.items():
        new, book = build_branch_state(st, cid, key, focus_price=12,
                                       rival="Team05", rival_price=15)
        if spec.holder == "none":
            assert new.owner_of.get(cid) is None
            assert cid in new.withdrawn
            assert not new.is_available(cid)
            assert book["declared_withdrawn"] == frozenset({cid})
        elif spec.holder == "focus":
            assert new.owner_of[cid] == focus
        else:
            assert new.owner_of[cid] == "Team05"
        # Removed from the remaining board in every branch.
        assert not new.is_available(cid)


def test_free_acquisition_takes_a_slot_but_no_money(fake_board, cid):
    st = fake_board.state
    focus = st.focus_owner_id
    before = st.owner(focus)
    new, _ = build_branch_state(st, cid, "UF", focus_price=12, rival="Team05",
                                rival_price=15)
    after = new.owner(focus)
    assert after.budget_remaining == before.budget_remaining
    assert after.open_slots == before.open_slots - 1
    assert after.n_players == before.n_players + 1


def test_paid_acquisition_spends_exactly_the_price(fake_board, cid):
    st = fake_board.state
    focus = st.focus_owner_id
    before = st.owner(focus).budget_remaining
    new, _ = build_branch_state(st, cid, "UP", focus_price=12, rival="Team05",
                                rival_price=15)
    assert new.owner(focus).budget_remaining == before - 12
    rival_state, _ = build_branch_state(st, cid, "RP", focus_price=12,
                                        rival="Team05", rival_price=15)
    assert rival_state.owner("Team05").budget_remaining == \
        st.owner("Team05").budget_remaining - 15


def test_free_rival_branch_costs_the_rival_nothing(fake_board, cid):
    st = fake_board.state
    new, _ = build_branch_state(st, cid, "RF", focus_price=12, rival="Team05",
                                rival_price=15)
    assert new.owner("Team05").budget_remaining == \
        st.owner("Team05").budget_remaining
    assert new.owner("Team05").open_slots == \
        st.owner("Team05").open_slots - 1


def test_the_withdrawn_branch_needs_an_explicit_declaration(fake_board, cid):
    _, book = build_branch_state(fake_board.state, cid, "W", focus_price=12,
                                 rival="Team05", rival_price=15)
    assert book["declared_withdrawn"] == frozenset({cid})
    assert book["branch_acquired"] == frozenset()


def test_analytical_branches_cannot_become_pass_probabilities():
    for key in ("W", "UF", "RF"):
        assert BRANCHES[key].analytical
        with pytest.raises(AnalyticalBranchMisuse, match="ANALYTICAL"):
            assert_not_a_recipient(key)
    for key in ("UP", "RP"):
        assert not BRANCHES[key].analytical
        assert_not_a_recipient(key) is None


def test_a_rival_equal_to_the_focus_owner_is_refused(fake_board, cid):
    px = ProxyEvaluator(fake_board.state.pool, fake_board.state.settings, 8, 7)
    with pytest.raises(ValueError, match="must not be the focus owner"):
        decompose(fake_board.state, fake_board.cast, fake_board.costs["base"],
                  cid, focus_price=5, rival=fake_board.state.focus_owner_id,
                  rival_price=5, draws=balanced_schedule(12, 2), board=_BS,
                  completion=_CS, proxy=px, holdout_sims=200)


# ---------------------------------------------------------------------------
# Telescoping
# ---------------------------------------------------------------------------


def test_the_four_components_telescope_exactly_per_draw(run):
    for d in run.draws:
        assert d.residual == pytest.approx(0.0, abs=1e-12)
        assert sum(d.components.values()) == pytest.approx(d.total, abs=1e-12)
        assert d.total == pytest.approx(d.ce["UP"] - d.ce["RP"], abs=1e-12)


def test_the_components_telescope_at_the_ensemble_mean(run):
    assert run.ensemble_residual == pytest.approx(0.0, abs=1e-9)
    means = {n: statistics.fmean(run.component_values(n))
             for n, _, _ in COMPONENT_ORDER}
    assert sum(means.values()) == pytest.approx(
        statistics.fmean(run.totals()), abs=1e-9)


def test_the_declared_path_covers_every_branch_once(run):
    used = []
    for _, a, b in COMPONENT_ORDER:
        used += [a, b]
    # UP and RP appear once each (the endpoints); the interior branches twice.
    assert used.count("UP") == 1 and used.count("RP") == 1
    for interior in ("UF", "W", "RF"):
        assert used.count(interior) == 2


def test_serialized_output_reconciles_too(run):
    blob = run.to_dict(include_draws=True)
    assert blob["ensemble_residual"] == pytest.approx(0.0, abs=1e-9)
    assert blob["max_per_draw_residual"] == pytest.approx(0.0, abs=1e-12)
    s = sum(blob["components"][n]["mean"] for n, _, _ in COMPONENT_ORDER)
    assert s == pytest.approx(blob["total"]["mean"], abs=1e-5)
    assert "declared telescoping path" in blob["path_note"]
    assert "not possible auction outcomes" in blob["analytical_note"]


# ---------------------------------------------------------------------------
# Coupling and conservation
# ---------------------------------------------------------------------------


def test_every_branch_uses_the_same_rotation_schedule(run):
    schedule = balanced_schedule(12, 3)
    assert [d.draw.key() for d in run.draws] == [s.key() for s in schedule]
    for d in run.draws:
        assert set(d.ce) == set(BRANCHES)
        assert set(d.fingerprints) == set(BRANCHES)


def test_branches_produce_distinct_joint_allocations(run):
    for d in run.draws:
        assert len(set(d.fingerprints.values())) >= 4, \
            "branches must not collapse to one world"


def test_conservation_and_league_sums_hold_in_every_branch(run):
    for d in run.draws:
        assert d.conservation_ok
        for branch, total in d.league_ce_sums.items():
            assert total == pytest.approx(1.0, abs=1e-9), branch


def test_allocation_and_season_uncertainty_stay_separate(run):
    blob = run.to_dict(include_draws=False)
    for name, _, _ in COMPONENT_ORDER:
        c = blob["components"][name]
        assert "between_allocation_sd" in c
        assert "ci95" in c and "sign_positive" in c
    assert set(blob["rms_within_allocation_se"]) == set(BRANCHES)


# ---------------------------------------------------------------------------
# The components respond to what they name
# ---------------------------------------------------------------------------


def test_rival_price_moves_the_rival_payment_component(fake_board, cid):
    px = ProxyEvaluator(fake_board.state.pool, fake_board.state.settings, 16, 7)
    kw = dict(position="QB", draws=balanced_schedule(12, 2), board=_BS,
              completion=_CS, market=fake_board.market,
              key_by_id=fake_board.key_by_id, proxy=px, holdout_sims=400)
    cheap = decompose(fake_board.state, fake_board.cast,
                      fake_board.costs["base"], cid, focus_price=12,
                      rival="Team05", rival_price=2, **kw)
    dear = decompose(fake_board.state, fake_board.cast,
                     fake_board.costs["base"], cid, focus_price=12,
                     rival="Team05", rival_price=60, **kw)
    # RF is identical in both (rival pays nothing); only RP differs, so any
    # change in the total must land in the rival-payment term.
    assert [d.ce["RF"] for d in cheap.draws] == [d.ce["RF"] for d in dear.draws]
    assert [d.ce["UP"] for d in cheap.draws] == [d.ce["UP"] for d in dear.draws]
    if [d.ce["RP"] for d in cheap.draws] != [d.ce["RP"] for d in dear.draws]:
        assert cheap.component_values("rival_payment") != \
            dear.component_values("rival_payment")


def test_recipient_identity_can_move_the_denial_component(fake_board, cid):
    px = ProxyEvaluator(fake_board.state.pool, fake_board.state.settings, 16, 7)
    kw = dict(position="QB", focus_price=12, rival_price=15,
              draws=balanced_schedule(12, 2), board=_BS, completion=_CS,
              market=fake_board.market, key_by_id=fake_board.key_by_id,
              proxy=px, holdout_sims=400)
    a = decompose(fake_board.state, fake_board.cast, fake_board.costs["base"],
                  cid, rival="Team05", **kw)
    b = decompose(fake_board.state, fake_board.cast, fake_board.costs["base"],
                  cid, rival="Team09", **kw)
    # W and UF are recipient-independent; RF is not, so denial is where
    # recipient identity is allowed to show up.
    assert [d.ce["W"] for d in a.draws] == [d.ce["W"] for d in b.draws]
    assert [d.ce["UF"] for d in a.draws] == [d.ce["UF"] for d in b.draws]
    assert isinstance(a.component_values("rival_denial"), tuple)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def _fake(components, totals, improvement=2.0):
    draws = balanced_schedule(12, len(totals))
    dd = []
    for d, comps, tot in zip(draws, components, totals):
        dd.append(DrawDecomposition(
            draw=d, ce={b: 0.1 for b in BRANCHES}, components=comps,
            total=tot, residual=0.0, conservation_ok=True,
            league_ce_sums={b: 1.0 for b in BRANCHES},
            fingerprints={b: b for b in BRANCHES}))
    return Decomposition(
        candidate_id=1, position="RB", focus_price=25, rival="Team12",
        rival_price=30, draws=tuple(dd), lineup_improvement=improvement,
        role="marginal starter", within_se={b: 0.001 for b in BRANCHES},
        runtime_s=1.0)


def test_helps_us_but_costs_too_much_is_recognised():
    comps = [{"our_payment": -0.10, "our_possession": 0.06,
              "rival_denial": 0.001, "rival_payment": 0.0}] * 5
    d = _fake(comps, [-0.039] * 5)
    label, why = d.classify()
    assert label == "helps us but costs too much"
    assert "own payment" in why


def test_a_beneficial_rival_overpay_is_recognised():
    comps = [{"our_payment": -0.01, "our_possession": 0.005,
              "rival_denial": 0.0, "rival_payment": 0.04}] * 5
    d = _fake(comps, [0.035] * 5)
    label, _ = d.classify()
    assert label in ("passing induces a beneficial rival overpay",
                     "denial value matters", "mixed")


def test_an_unresolved_total_is_not_given_a_cause():
    comps = [{"our_payment": -0.05, "our_possession": 0.05,
              "rival_denial": 0.0, "rival_payment": 0.0}] * 4
    d = _fake([dict(c) for c in comps], [0.02, -0.02, 0.03, -0.03])
    label, why = d.classify()
    assert label == "unresolved"
    assert "contains zero" in why


def test_the_proxy_is_not_automatically_blamed():
    """Positive improvement with a negative total is not selector failure."""
    comps = [{"our_payment": -0.12, "our_possession": 0.08,
              "rival_denial": 0.0, "rival_payment": 0.0}] * 5
    d = _fake(comps, [-0.04] * 5, improvement=2.5)
    label, _ = d.classify()
    assert label != "proxy is misleading"


def test_the_proxy_is_blamed_only_when_own_possession_disagrees():
    comps = [{"our_payment": -0.001, "our_possession": -0.03,
              "rival_denial": 0.0, "rival_payment": 0.0}] * 5
    d = _fake(comps, [-0.031] * 5, improvement=2.5)
    label, why = d.classify()
    assert label == "proxy is misleading"
    assert "own possession" in why


def test_positive_improvement_and_negative_total_coexist_legitimately():
    comps = [{"our_payment": -0.09, "our_possession": 0.05,
              "rival_denial": 0.0, "rival_payment": 0.0}] * 5
    d = _fake(comps, [-0.04] * 5, improvement=2.5)
    assert d.lineup_improvement > 0
    assert statistics.fmean(d.totals()) < 0
    assert d.classify()[0] == "helps us but costs too much"


def test_no_hard_coded_denial_premium():
    src = Path(__file__).resolve().parents[1] / "src" / "ceauction" / "tactical"
    for path in src.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for banned in ("denial_premium", "denial_bonus", "denial_coefficient",
                       "denial_multiplier", "DENIAL_WEIGHT"):
            assert banned not in text, f"{banned} in {path.name}"


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
    (("decompose", "--out-dir", "/tmp/leak"), "local_data"),
    (("decompose", "--draws", "1"), "at least 2"),
    (("decompose", "--sims", "10"), "at least 100"),
    (("decompose", "--position", "K"), "invalid choice"),
    (("decompose", "--contract", "/nope.json"), "missing"),
])
def test_cli_usage_errors_have_no_traceback(argv, needle):
    r = _cli(*argv)
    assert r.returncode != 0
    assert "Traceback" not in r.stderr
    assert needle in (r.stdout + r.stderr)
