"""Championship equity, common random numbers, and paired comparison."""

from __future__ import annotations

import math

import numpy as np
import pytest

from ceauction.ce import compare_scenarios, team_report, wilson_interval
from ceauction.league import DEFAULT_LEAGUE
from ceauction.players import with_overrides
from ceauction.roster import RosterSet
from ceauction.simulate import simulate_seasons, team_scores
from ceauction.synthetic import make_identical_league, make_synthetic_league
from ceauction.worlds import build_pool_arrays, generate_world

SEED = 606
T = DEFAULT_LEAGUE.n_teams
IDENTICAL_SIMS = 6000


def test_wilson_interval_brackets_the_estimate():
    lo, hi = wilson_interval(80, 1000)
    assert lo < 0.08 < hi
    assert 0.0 <= lo and hi <= 1.0
    assert wilson_interval(0, 100)[0] == 0.0


def test_twelve_identical_teams_have_approximately_equal_ce(identical_league):
    """The strongest global check on the whole pipeline.

    Twelve teams with identical parameters (but independent players) must each
    win about 1/12 of the time.  Anything that silently favours a team index --
    a schedule that is not exchangeable, a tiebreak that leaks, a bracket that
    is misaligned -- shows up here.
    """
    out = simulate_seasons(identical_league, IDENTICAL_SIMS, SEED)
    ce = out.championship_equity()
    expected = 1.0 / T
    se = math.sqrt(expected * (1 - expected) / IDENTICAL_SIMS)

    assert ce.sum() == pytest.approx(1.0)
    assert np.abs(ce - expected).max() < 4.0 * se, (
        f"CE spread too wide: {np.round(ce, 4).tolist()}"
    )
    # Chi-square goodness of fit, 11 df, 0.1% critical value 31.26.
    counts = np.bincount(out.champion, minlength=T)
    chi2 = float(((counts - IDENTICAL_SIMS * expected) ** 2
                  / (IDENTICAL_SIMS * expected)).sum())
    assert chi2 < 31.26, f"chi2 = {chi2:.2f}"


def test_identical_teams_also_share_playoff_and_bye_probabilities(identical_league):
    out = simulate_seasons(identical_league, IDENTICAL_SIMS, SEED)
    playoff = out.made_playoffs.mean(axis=0)
    bye = out.has_bye.mean(axis=0)
    assert playoff.mean() == pytest.approx(0.5)
    assert bye.mean() == pytest.approx(2.0 / T)
    assert playoff.max() - playoff.min() < 0.05
    assert bye.max() - bye.min() < 0.035


def test_no_team_index_advantage_in_seeding(identical_league):
    out = simulate_seasons(identical_league, IDENTICAL_SIMS, SEED)
    mean_seed = out.seed.mean(axis=0)
    assert abs(mean_seed.max() - mean_seed.min()) < 0.35


# --------------------------------------------------------------------------
# Common random numbers
# --------------------------------------------------------------------------


def _swap_one_player(league, factor=1.4):
    pid = league.rosters[0].player_ids[0]
    spec = league.spec(pid)
    return league.with_pool_player(
        with_overrides(spec, base_mean=spec.base_mean * factor, crn_key=spec.player_id)
    )


def test_changing_one_roster_leaves_every_other_teams_scores_untouched(league):
    """The defining property of common random numbers in this design."""
    variant = _swap_one_player(league)
    pool_a = build_pool_arrays(league.pool, league.settings)
    pool_b = build_pool_arrays(variant.pool, variant.settings)
    wa = generate_world(pool_a, SEED, 0, 24)
    wb = generate_world(pool_b, SEED, 0, 24)
    rm = league.roster_matrix()
    sa, _ = team_scores(wa, rm)
    sb, _ = team_scores(wb, rm)
    assert not np.allclose(sa[:, 0, :], sb[:, 0, :]), "the changed team must move"
    assert np.array_equal(sa[:, 1:, :], sb[:, 1:, :]), (
        "an unrelated team's scores changed -- CRN is broken"
    )


def test_paired_scenarios_share_injuries_environments_and_schedules(league):
    variant = _swap_one_player(league)
    pool_a = build_pool_arrays(league.pool, league.settings)
    pool_b = build_pool_arrays(variant.pool, variant.settings)
    wa = generate_world(pool_a, SEED, 0, 16)
    wb = generate_world(pool_b, SEED, 0, 16)
    assert np.array_equal(wa.availability.available, wb.availability.available)
    assert np.array_equal(wa.realized.group_effect, wb.realized.group_effect)
    assert np.array_equal(wa.latent.role_week, wb.latent.role_week)

    from ceauction.schedule import opponents_for_batch
    oa = opponents_for_batch(SEED, 0, 16, league.settings)
    ob = opponents_for_batch(SEED, 0, 16, variant.settings)
    assert np.array_equal(oa, ob)


def test_a_null_comparison_has_exactly_zero_delta(league):
    """Comparing a league with itself must produce an identical world twice."""
    c = compare_scenarios(league, league, 0, 200, SEED, label="null")
    assert c.delta_ce == 0.0
    assert c.delta_points_per_week == 0.0
    assert c.seasons_differing == 0
    assert c.report_a.championship_equity == c.report_b.championship_equity


def test_pairing_beats_independent_sampling(league):
    """CRN must actually reduce variance, or the design has no point."""
    variant = _swap_one_player(league, factor=1.15)
    c = compare_scenarios(variant, league, 0, 1500, SEED, label="paired")
    assert c.paired_efficiency > 2.0
    unpaired_se = math.sqrt(c.report_a.ce_se ** 2 + c.report_b.ce_se ** 2)
    assert c.delta_ce_se < unpaired_se


def test_a_real_improvement_is_detected_with_the_right_sign(league):
    variant = _swap_one_player(league, factor=1.5)
    c = compare_scenarios(variant, league, 0, 1500, SEED, label="stronger roster")
    assert c.delta_ce > 0
    assert c.delta_ce_z > 3.0
    assert c.delta_playoff > 0
    assert c.delta_points_per_week > 0
    assert c.significant_95


def test_team_report_metrics_are_self_consistent(league):
    out = simulate_seasons(league, 400, SEED)
    r = team_report(out, 0, league.settings)
    assert 0.0 <= r.championship_equity <= r.final_probability
    assert r.final_probability <= r.playoff_probability <= 1.0
    assert r.bye_probability <= r.playoff_probability
    assert 0.0 <= r.above_median_rate <= 1.0
    assert 0.0 <= r.head_to_head_win_rate <= 1.0
    assert r.mean_points_per_week == pytest.approx(
        r.mean_regular_season_points / DEFAULT_LEAGUE.regular_season_weeks
    )
    assert r.ce_ci95[0] <= r.championship_equity <= r.ce_ci95[1]


def test_comparison_is_reproducible(league):
    variant = _swap_one_player(league)
    a = compare_scenarios(variant, league, 0, 400, SEED, label="x")
    b = compare_scenarios(variant, league, 0, 400, SEED, label="x")
    assert a.delta_ce == b.delta_ce
    assert a.delta_ce_se == b.delta_ce_se
    assert a.seasons_differing == b.seasons_differing


def test_no_extra_randomness_enters_the_playoffs(league):
    """Given the score array, the champion is a pure function of it."""
    from ceauction.playoffs import run_bracket
    from ceauction.schedule import opponents_for_batch
    from ceauction.standings import regular_season
    pool = build_pool_arrays(league.pool, league.settings)
    w = generate_world(pool, SEED, 0, 32)
    scores, _ = team_scores(w, league.roster_matrix())
    opp = opponents_for_batch(SEED, 0, 32, league.settings)
    rs = regular_season(scores, opp, league.settings)
    champs = [run_bracket(scores, rs, league.settings).champion for _ in range(3)]
    assert all(np.array_equal(champs[0], c) for c in champs[1:])
