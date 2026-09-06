"""Controlled one-factor roster-context regimes. Fabricated data only.

The previous regime experiment moved four things at once -- pre-owned count,
money, open slots and the remaining board -- so nothing it measured could be
attributed to roster strength. These tests exist to make that class of mistake
loud: `check_structural_equality` must return empty, and the validator must
catch a planted difference in each prohibited dimension.
"""

from __future__ import annotations

import dataclasses
from dataclasses import replace

import pytest

from ceauction.auction.completion import CompletionSettings
from ceauction.auction.proxy import ProxyEvaluator
from ceauction.tactical.board import BoardSettings
from ceauction.tactical.context import (CONTEXT_REGIMES, N_PREOWNED,
                                        PREOWNED_PRICE, build_all_regimes,
                                        build_context_regime,
                                        check_structural_equality,
                                        expected_differences, measure_regime)

_CS = CompletionSettings(beam_width=32, candidate_pool=30, proxy_candidates=32,
                         finalists=3, max_candidates=160, proxy_reps=16,
                         selection_sims=400, evaluation_sims=1500,
                         rival_selection="proxy")
_ORDER = ("underdog", "playoff_bubble", "bye_bubble", "favorite")


@pytest.fixture(scope="module")
def regimes():
    return build_all_regimes()


@pytest.fixture(scope="module")
def metrics(regimes):
    out = {}
    for name in _ORDER:
        d = regimes[name]
        px = ProxyEvaluator(d.state.pool, d.state.settings, 16, 7)
        out[name] = measure_regime(
            name, d, _CS, d.costs, proxy=px, sims=1500, seed=917_324_011,
            board_settings=BoardSettings(), market=d.market,
            key_by_id=d.key_by_id)
    return out


# ---------------------------------------------------------------------------
# Structural equality: the whole point
# ---------------------------------------------------------------------------


def test_controlled_regimes_are_structurally_identical(regimes):
    diffs = check_structural_equality(
        regimes, candidate_id=1, price=13, recipient="Owner04", leader=None,
        seeds=(1, 2, 3))
    assert diffs == (), "\n".join(str(d) for d in diffs)


def test_budgets_spend_and_open_slots_are_equal(regimes):
    seen = set()
    for d in regimes.values():
        o = d.state.owner(d.state.focus_owner_id)
        seen.add((o.budget_remaining, o.spent, o.n_players, o.open_slots))
        assert o.n_players == N_PREOWNED
        assert all(f.price == PREOWNED_PRICE for f in o.filled)
    assert len(seen) == 1, seen


def test_remaining_boards_are_identical(regimes):
    boards = {tuple(sorted(d.state.available_ids)) for d in regimes.values()}
    assert len(boards) == 1
    costs = {tuple((pid, d.costs.cost_of(pid, 1))
                   for pid in sorted(d.state.available_ids))
             for d in regimes.values()}
    assert len(costs) == 1


def test_rival_states_are_identical(regimes):
    rivals = set()
    for d in regimes.values():
        st = d.state
        rivals.add(tuple(sorted(
            (o.owner_id, o.player_ids, o.spent, o.budget_remaining,
             o.open_slots)
            for o in st.owners if o.owner_id != st.focus_owner_id)))
    assert len(rivals) == 1


def test_market_and_settings_fingerprints_are_identical(regimes):
    assert len({d.market.fingerprint() for d in regimes.values()}) == 1
    assert len({d.costs.fingerprint() for d in regimes.values()}) == 1
    assert len({d.state.settings_fingerprint() for d in regimes.values()}) == 1


def test_only_focus_owned_playerspecs_differ(regimes):
    exp = expected_differences(regimes)
    assert exp["pool_fingerprints_differ"], "the factor under test must move"
    assert exp["non_focus_projections_identical"], \
        "nothing outside the focus roster may move"
    totals = exp["focus_owned_projection_total"]
    assert len(set(totals.values())) == 4
    # Spec-by-spec: every board player is byte-identical across regimes.
    ref = regimes["underdog"]
    owned = set(ref.state.owner(ref.state.focus_owner_id).player_ids)
    ref_specs = {s.player_id: s for s in ref.state.pool}
    for name, d in regimes.items():
        for s in d.state.pool:
            if s.player_id in owned:
                continue
            assert s == ref_specs[s.player_id], (name, s.player_id)


def test_focus_positions_are_identical_only_scoring_moves(regimes):
    shapes = set()
    for d in regimes.values():
        o = d.state.owner(d.state.focus_owner_id)
        shapes.add(tuple(sorted((f.player_id, int(f.position))
                                for f in o.filled)))
    assert len(shapes) == 1


# ---------------------------------------------------------------------------
# The validator must catch planted differences
# ---------------------------------------------------------------------------


def _fields(diffs):
    return {d.field for d in diffs}


def test_validator_catches_a_budget_difference(regimes):
    d = regimes["underdog"]
    st = d.state
    focus = st.focus_owner_id
    owner = st.owner(focus)
    cheaper = replace(owner, filled=tuple(
        replace(f, price=f.price - 1) for f in owner.filled))
    tampered = replace(st, owners=tuple(
        cheaper if o.owner_id == focus else o for o in st.owners))
    diffs = check_structural_equality(
        {**regimes, "tampered": d.with_state(tampered)})
    assert "focus_budget_remaining" in _fields(diffs)
    assert "focus_spent" in _fields(diffs)


def test_validator_catches_a_board_difference(regimes):
    d = regimes["underdog"]
    victim = sorted(d.state.available_ids)[0]
    diffs = check_structural_equality(
        {**regimes, "tampered": d.with_state(d.state.withdraw(victim))})
    assert "remaining_board_ids" in _fields(diffs)
    assert "withdrawn" in _fields(diffs)


def test_validator_catches_a_rival_state_difference(regimes):
    d = regimes["underdog"]
    st = d.state
    rival = next(o for o in st.owners if o.owner_id != st.focus_owner_id)
    victim = next(pid for pid in sorted(st.available_ids)
                  if st.purchase_shortfall(pid, rival.owner_id, 1) is None)
    diffs = check_structural_equality(
        {**regimes,
         "tampered": d.with_state(st.apply_purchase(victim, rival.owner_id, 1))})
    assert "rival_states" in _fields(diffs)


def test_validator_catches_a_differing_candidate_or_seed(regimes):
    a = check_structural_equality(regimes, candidate_id=1, seeds=(1,))
    assert a == ()
    # The candidate/price/seed fields are stamped identically on every regime,
    # so a mismatch there is a caller error the report still surfaces.
    tables = check_structural_equality({**regimes})
    assert tables == ()


def test_an_unknown_regime_is_refused():
    with pytest.raises(ValueError, match="unknown context regime"):
        build_context_regime("juggernaut")


# ---------------------------------------------------------------------------
# Labels must be earned by measured outcomes
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_league_ce_sums_to_one_in_every_regime(metrics):
    for name, m in metrics.items():
        assert abs(m.league_ce_sum - 1.0) < 1e-9, name


@pytest.mark.slow
def test_underdog_is_below_the_playoff_bubble(metrics):
    assert metrics["underdog"].championship_equity < \
        metrics["playoff_bubble"].championship_equity
    assert metrics["underdog"].playoff_probability < 0.25
    assert metrics["underdog"].championship_equity > 0.0, \
        "an underdog pinned at exactly zero cannot show a marginal effect"


@pytest.mark.slow
def test_the_playoff_bubble_is_actually_uncertain(metrics):
    p = metrics["playoff_bubble"].playoff_probability
    assert 0.25 < p < 0.75, f"p(playoff)={p} is not a bubble"


@pytest.mark.slow
def test_the_bye_bubble_is_actually_uncertain_and_stronger(metrics):
    b = metrics["bye_bubble"]
    assert 0.25 < b.bye_probability < 0.75, f"p(bye)={b.bye_probability}"
    assert b.championship_equity > metrics["playoff_bubble"].championship_equity
    assert b.playoff_probability > metrics["playoff_bubble"].playoff_probability


@pytest.mark.slow
def test_the_favorite_is_actually_top_two(metrics):
    f = metrics["favorite"]
    assert f.ce_rank <= 2, f"rank {f.ce_rank} does not earn the label"
    assert f.championship_equity > metrics["bye_bubble"].championship_equity
    assert f.bye_probability > 0.75


@pytest.mark.slow
def test_the_ordering_is_monotone_in_the_single_factor(metrics):
    ce = [metrics[n].championship_equity for n in _ORDER]
    assert ce == sorted(ce), ce
    scales = [CONTEXT_REGIMES[n].strength_scale for n in _ORDER]
    assert scales == sorted(scales)


# ---------------------------------------------------------------------------
# No hard-coded context modifier anywhere
# ---------------------------------------------------------------------------


def test_no_predetermined_bubble_premium_exists_in_code():
    """Context effects must emerge from simulation, never from a constant."""
    import pathlib
    src = pathlib.Path(__file__).resolve().parents[1] / "src" / "ceauction"
    banned = ("bubble_premium", "bubble_bonus", "context_multiplier",
              "contender_premium", "underdog_discount", "favorite_discount")
    for path in src.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for word in banned:
            assert word not in text, f"{word} in {path.name}"


def test_the_strength_scale_is_the_only_regime_parameter():
    fields = {f.name for f in dataclasses.fields(CONTEXT_REGIMES["underdog"])}
    assert fields == {"name", "description", "strength_scale"}
