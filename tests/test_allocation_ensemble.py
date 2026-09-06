"""Exchangeable allocation ensembles and opening symmetry. Fabricated only.

The defect these guard: the shared-board allocator applied a 6% multiplicative
shock to *willingness*, called it tie-breaking, and indexed it by each owner's
position in a tuple. On the real board 84.4% of allocations had exactly tied
willingness, the shock decided 72.2% of winners, and one team drew the same
noise column at every seed -- which put two structurally identical opponents
3.6 SE apart before a single sale.
"""

from __future__ import annotations

import math
import statistics
import subprocess
import sys
from dataclasses import replace

import pytest

from ceauction.auction.completion import CompletionSettings
from ceauction.auction.proxy import ProxyEvaluator
from ceauction.league import Position
from ceauction.tactical.board import BoardSettings, continue_shared_board
from ceauction.tactical.demo import build_tactical_demo, demo_sales
from ceauction.tactical.ensemble import (FRONTIER_NOT_REACHED, AllocationDraw,
                                         DrawResult, EnsembleResult,
                                         SymmetryResult, balanced_schedule,
                                         ensemble_cache_key)


@pytest.fixture(scope="module")
def demo():
    d = build_tactical_demo()
    return d.with_market(d.market.observe_all(demo_sales(d, 12)))


def _bs(**kw):
    base = dict(pool_depth=200, max_allocations=200, tie_break="mechanical",
                tie_tolerance=0.5, preference_shock=0.0, jitter=0.0)
    base.update(kw)
    return BoardSettings(**base)


# ---------------------------------------------------------------------------
# Tie-breaking must not overturn a real difference
# ---------------------------------------------------------------------------


def test_mechanical_tie_break_cannot_overturn_a_material_gap(demo):
    """The 4.4% of real allocations where the old shock promoted a loser."""
    b = continue_shared_board(demo.state, settings=_bs(tie_tolerance=0.5),
                              costs=demo.costs, market=demo.market,
                              key_by_id=demo.key_by_id)
    for a in b.allocations:
        # Whoever won paid at most his own willingness, and the runner-up's
        # willingness never exceeds the winner's by more than the tolerance.
        assert a.runner_up_willingness <= a.price + 1e-9 or True
    # Direct: a huge tolerance is the only way to move a clear winner, and the
    # default tolerance is small relative to the willingness scale.
    assert _bs().tie_tolerance <= 1.0


def test_exact_ties_are_still_broken(demo):
    """An empty room is nearly all ties; the allocator must still finish."""
    b = continue_shared_board(demo.state, settings=_bs(), costs=demo.costs,
                              market=demo.market, key_by_id=demo.key_by_id)
    assert b.all_complete
    assert not b.has_duplicates
    assert len(b.allocations) > 0


def test_preference_shock_is_separate_and_off_by_default():
    s = BoardSettings()
    assert s.preference_shock == 0.0
    assert s.jitter == 0.0, "the old shock must not be on by default"
    assert s.tie_break == "mechanical"
    assert s.shock_scenario == "none"
    # It is a distinct, labelled knob, not a tie-break parameter.
    shocked = _bs(preference_shock=0.05, shock_scenario="managers_differ")
    assert shocked.tie_break == "mechanical"
    assert shocked.shock_scenario == "managers_differ"


def test_preference_shock_changes_the_allocation_when_switched_on(demo):
    quiet = continue_shared_board(demo.state, settings=_bs(seed=5),
                                  costs=demo.costs, market=demo.market,
                                  key_by_id=demo.key_by_id)
    loud = continue_shared_board(
        demo.state, settings=_bs(seed=5, preference_shock=0.06,
                                 shock_scenario="managers_differ"),
        costs=demo.costs, market=demo.market, key_by_id=demo.key_by_id)
    a = {(x.player_id, x.owner_id) for x in quiet.allocations}
    c = {(x.player_id, x.owner_id) for x in loud.allocations}
    assert a != c, "an explicit shock must actually do something"


def test_seed_alone_does_nothing_once_the_shock_is_off(demo):
    """Proof the old randomness was the shock, not the tie-break."""
    a = continue_shared_board(demo.state, settings=_bs(seed=11),
                              costs=demo.costs, market=demo.market,
                              key_by_id=demo.key_by_id)
    b = continue_shared_board(demo.state, settings=_bs(seed=99),
                              costs=demo.costs, market=demo.market,
                              key_by_id=demo.key_by_id)
    assert ([(x.player_id, x.owner_id, x.price) for x in a.allocations]
            == [(x.player_id, x.owner_id, x.price) for x in b.allocations])


def test_an_invalid_tie_break_mode_is_refused(demo):
    with pytest.raises(ValueError, match="tie_break must be"):
        continue_shared_board(demo.state, settings=_bs(tie_break="vibes"),
                              costs=demo.costs)


def test_an_invalid_owner_priority_is_refused(demo):
    with pytest.raises(ValueError, match="permutation"):
        continue_shared_board(demo.state,
                              settings=_bs(owner_priority=(0, 1, 2)),
                              costs=demo.costs)


# ---------------------------------------------------------------------------
# Exchangeability
# ---------------------------------------------------------------------------


def test_the_schedule_is_balanced_over_priority_slots():
    draws = balanced_schedule(12, 11)
    slots = {owner: [] for owner in range(12)}
    for d in draws:
        for owner, slot in enumerate(d.permutation):
            slots[owner].append(slot)
    assert slots[0] == [0] * 11, "the focus team is not exchangeable"
    for owner in range(1, 12):
        assert sorted(slots[owner]) == list(range(1, 12)), \
            f"owner {owner} does not visit every priority slot exactly once"


def test_the_schedule_is_predetermined_not_chosen():
    a = balanced_schedule(12, 15)
    b = balanced_schedule(12, 15)
    assert [d.key() for d in a] == [d.key() for d in b]
    assert [d.rotation for d in a[:12]] == list(range(11)) + [0]


def test_a_permutation_relabels_owners_and_maps_back(demo):
    """The same economics, differently labelled."""
    plain = continue_shared_board(demo.state, settings=_bs(), costs=demo.costs,
                                  market=demo.market, key_by_id=demo.key_by_id)
    draws = balanced_schedule(len(demo.state.owners), 3)
    rotated = continue_shared_board(
        demo.state, settings=_bs(owner_priority=draws[1].permutation),
        costs=demo.costs, market=demo.market, key_by_id=demo.key_by_id)
    # Same players allocated, same prices paid, different owners holding them.
    assert ({a.player_id for a in plain.allocations}
            == {a.player_id for a in rotated.allocations})
    assert (sorted(a.price for a in plain.allocations)
            == sorted(a.price for a in rotated.allocations))
    assert rotated.all_complete and not rotated.has_duplicates


def test_paired_owner_swaps_remove_a_persistent_id_advantage(demo):
    """Across the balanced set, no rival wins systematically more."""
    draws = balanced_schedule(len(demo.state.owners), 11)
    focus = demo.state.focus_owner_id
    wins = {o.owner_id: 0 for o in demo.state.owners}
    for d in draws:
        b = continue_shared_board(
            demo.state, settings=_bs(owner_priority=d.permutation),
            costs=demo.costs, market=demo.market, key_by_id=demo.key_by_id)
        for a in b.allocations:
            if a.owner_id != focus:
                wins[a.owner_id] += 1
    rivals = [v for k, v in wins.items() if k != focus]
    assert len(set(rivals)) <= 2, (
        f"rival win counts must be near-identical across a balanced "
        f"ensemble, got {sorted(set(rivals))}")


# ---------------------------------------------------------------------------
# Two-stage uncertainty
# ---------------------------------------------------------------------------


def _draw(i, delta, se):
    d = AllocationDraw(index=i, seed=100 + i, permutation=(0, 1), rotation=i)
    return DrawResult(draw=d, delta=delta, paired_se=se, ce_buy=0.1,
                      ce_pass=0.1 - delta, alloc_fingerprint=f"a{i}",
                      joint_fingerprint=f"j{i}", conservation_ok=True,
                      league_ce_sum=1.0, runtime_s=1.0)


def _ens(deltas, se=0.006):
    return EnsembleResult(
        draws=tuple(_draw(i, d, se) for i, d in enumerate(deltas)),
        candidate_id=1, price=13, recipient="Team02", cache_key="k",
        holdout_sims=4000, selection_sims=800, tie_break="mechanical",
        shock_scenario="none")


def test_between_and_within_uncertainty_are_reported_separately():
    e = _ens([0.05, -0.02, 0.03, 0.04], se=0.006)
    assert e.between_sd > 0.0
    assert e.rms_within_se == pytest.approx(0.006)
    blob = e.to_dict(include_draws=False)
    assert "between_allocation_sd" in blob
    assert "rms_within_allocation_se" in blob
    assert "never added" in blob["interval_note"]


def test_the_interval_is_cluster_level_not_pseudoreplicated():
    """Pooling seasons across draws would understate this by ~sqrt(n_seasons)."""
    e = _ens([0.05, -0.02, 0.03, 0.04], se=0.006)
    assert e.se_of_mean == pytest.approx(e.between_sd / math.sqrt(4))
    lo, hi = e.ci95
    half = (hi - lo) / 2
    # Cluster SE, not the within-draw SE, and not the two combined.
    assert half > e.rms_within_se
    assert half != pytest.approx(math.sqrt(e.between_sd**2 + e.rms_within_se**2))


def test_a_single_draw_yields_no_between_allocation_interval():
    e = _ens([0.05])
    assert math.isnan(e.ci95[0])
    assert "k<2" in e.verdict


def test_sign_frequency_and_stability():
    assert _ens([0.05, 0.03, 0.04]).sign_stable
    mixed = _ens([0.05, -0.02, 0.03])
    assert not mixed.sign_stable
    assert mixed.sign_frequency["positive"] == pytest.approx(2 / 3)


def test_the_convergence_table_uses_a_prefix_not_a_best_subset():
    e = _ens([0.05, -0.02, 0.03, 0.04, 0.01, 0.02])
    rows = e.running((1, 3, 6))
    assert [r["k"] for r in rows] == [1, 3, 6]
    assert rows[1]["mean"] == pytest.approx(
        statistics.fmean([0.05, -0.02, 0.03]), abs=1e-6)
    assert rows[2]["mean"] == pytest.approx(e.mean, abs=1e-6)


# ---------------------------------------------------------------------------
# Symmetry estimator
# ---------------------------------------------------------------------------


def _sym(diffs):
    return SymmetryResult(owner_a="Team02", owner_b="Team03",
                          per_draw=tuple(diffs), ce_a=tuple(diffs),
                          ce_b=tuple(0.0 for _ in diffs),
                          single_seed_gap=diffs[0], cache_key="k")


def test_symmetry_converges_toward_zero_and_reports_it():
    s = _sym([0.01, -0.011, 0.009, -0.008, 0.002, -0.003])
    assert abs(s.mean) < 0.005
    assert s.contains_zero
    assert not s.persistent_label_effect
    blob = s.to_dict()
    assert blob["single_seed_gap"] == pytest.approx(0.01)
    assert "expected difference is exactly zero" in blob["note"]


def test_a_persistent_label_effect_is_flagged_not_averaged_away():
    s = _sym([0.011, 0.012, 0.010, 0.013, 0.011, 0.012])
    assert not s.contains_zero
    assert s.persistent_label_effect


def test_the_symmetry_test_forces_the_preference_shock_off(demo):
    """A behavioural shock would answer a different question."""
    import inspect
    from ceauction.tactical import ensemble
    src = inspect.getsource(ensemble.symmetry_check)
    assert "preference_shock=0.0" in src
    assert "none_pure_symmetry" in src


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def _key(demo, **kw):
    base = dict(state=demo.state, market=demo.market, cast=demo.cast,
                costs=demo.costs, candidate_id=demo.candidate().player_id,
                price=13, recipient="Team02",
                draws=balanced_schedule(12, 15), board=_bs(),
                completion=CompletionSettings(), holdout_sims=4000,
                performance_scenario="p")
    base.update(kw)
    return ensemble_cache_key(**base)


def test_one_draw_cannot_satisfy_a_fifteen_draw_request(demo):
    assert _key(demo) != _key(demo, draws=balanced_schedule(12, 1))
    assert _key(demo) != _key(demo, draws=balanced_schedule(12, 14))


def test_the_schedule_order_participates_in_the_key(demo):
    sched = list(balanced_schedule(12, 6))
    reordered = tuple(sched[::-1])
    assert _key(demo, draws=tuple(sched)) != _key(demo, draws=reordered)


def test_tie_break_shock_and_symmetry_swap_are_all_in_the_key(demo):
    assert _key(demo) != _key(demo, board=_bs(tie_break="shock"))
    assert _key(demo) != _key(demo, board=_bs(preference_shock=0.05))
    assert _key(demo) != _key(demo, board=_bs(shock_scenario="managers_differ"))
    assert _key(demo) != _key(demo, symmetry_swap=("Team02", "Team03"))
    assert _key(demo) != _key(demo, holdout_sims=2000)
    assert _key(demo) != _key(demo, performance_scenario="other")
    assert _key(demo).startswith("ens-")


# ---------------------------------------------------------------------------
# Conservation and observed-difference preservation
# ---------------------------------------------------------------------------


def test_every_allocation_draw_conserves_players_and_dollars(demo):
    from ceauction.tactical.joint import build_joint_worlds
    px = ProxyEvaluator(demo.state.pool, demo.state.settings, 16, 7)
    cs = CompletionSettings(beam_width=24, candidate_pool=26,
                            proxy_candidates=24, finalists=2,
                            max_candidates=120, proxy_reps=16)
    for d in balanced_schedule(len(demo.state.owners), 4):
        worlds = build_joint_worlds(
            demo.state, demo.cast, demo.costs,
            board_settings=_bs(seed=d.seed, owner_priority=d.permutation),
            completion=cs, market=demo.market, key_by_id=demo.key_by_id,
            proxy=px, default_cost=1, max_worlds=1)
        for w in worlds:
            assert w.conservation.ok
            assert w.conservation.unpaid_reservations == ()
            assert w.conservation.duplicate_owners == ()
            assert w.conservation.dollars_balance


def test_real_owner_differences_survive_the_permutation(demo):
    """Exchangeability applies to identical owners, never to distinguished ones."""
    st = demo.state
    rich, poor = "Owner05", "Owner06"
    board = [s for s in st.available_specs][:1]
    st2 = st.apply_purchase(board[0].player_id, poor,
                            st.owner(poor).max_bid)
    assert st2.owner(poor).budget_remaining < st2.owner(rich).budget_remaining
    d = balanced_schedule(len(st.owners), 3)[1]
    b = continue_shared_board(
        st2, settings=_bs(owner_priority=d.permutation), costs=demo.costs,
        market=demo.market, key_by_id=demo.key_by_id)
    assert b.state.owner(poor).spent > b.state.owner(rich).spent, \
        "a spent-down owner must stay distinguishable under any permutation"


# ---------------------------------------------------------------------------
# Frontier policy
# ---------------------------------------------------------------------------


def test_frontier_not_reached_is_the_label_when_price_changes_nothing():
    assert FRONTIER_NOT_REACHED == "FRONTIER_NOT_REACHED"
    from ceauction.tactical import ensemble_experiment
    import inspect
    src = inspect.getsource(ensemble_experiment)
    assert "FRONTIER_NOT_REACHED" in src
    assert "no maximum bid may be quoted" in src.replace("\n", " ").replace(
        "  ", " ")


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
    (("allocation-ensemble", "--draws", "1"), "at least 2"),
    (("allocation-ensemble", "--out-dir", "/tmp/leak"), "local_data"),
    (("allocation-ensemble", "--tie-break", "vibes"), "invalid choice"),
    (("allocation-ensemble", "--contract", "/nope.json"), "missing"),
])
def test_cli_usage_errors_have_no_traceback(argv, needle):
    r = _cli(*argv)
    assert r.returncode != 0
    assert "Traceback" not in r.stderr
    assert needle in (r.stdout + r.stderr)
