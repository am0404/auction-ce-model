"""The marginal championship-equity curve.

The curve is the object every auction pricing scheme would be a transformation
of, so the properties that matter are less about the numbers it produces than
about the guarantees behind them:

* two levels that are the same player must differ by exactly zero;
* the answer must not depend on the order the caller asked for levels;
* common random numbers must survive the sweep, or the whole design is
  pointless;
* the sweep must agree exactly with a directly computed paired comparison;
* the adjacent slopes must be genuinely paired, not a difference of two
  separately estimated baseline deltas.

Sim counts here are small on purpose. These tests check identities and
invariances, which hold exactly at any sample size; the one test about the
*shape* of the curve is explicit about its Monte Carlo tolerance.
"""

from __future__ import annotations

import csv
import io
import math

import numpy as np
import pytest

from ceauction.ce import compare_scenarios, paired_se
from ceauction.curve import (
    DEFAULT_RESOLUTION_TARGETS,
    MarginalCurve,
    isotonic_fit,
    sweep_marginal_curve,
    weakest_flex_slot,
)
from ceauction.league import DEFAULT_LEAGUE, Position
from ceauction.players import with_overrides
from ceauction.simulate import simulate_seasons

SEED = 606
SIMS = 400
TEAM = 0


@pytest.fixture(scope="module")
def target(league):
    return weakest_flex_slot(league, TEAM)


@pytest.fixture(scope="module")
def curve(league, target):
    return sweep_marginal_curve(
        league, TEAM, target.player_id, baseline_level=4.0,
        levels=[4.0, 7.0, 10.0, 13.0, 16.0], n_sims=SIMS, seed=SEED,
    )


def _sweep(league, target, levels, baseline=4.0, **kw):
    return sweep_marginal_curve(
        league, TEAM, target.player_id, baseline_level=baseline,
        levels=levels, n_sims=SIMS, seed=SEED, **kw,
    )


# --------------------------------------------------------------------------
# 1. An identical candidate produces exactly zero.
# --------------------------------------------------------------------------


def test_the_baseline_level_has_exactly_zero_paired_delta(curve):
    """Not "approximately zero": the two arms are the same league."""
    base = next(p for p in curve.points if p.is_baseline)
    assert base.level == 4.0
    assert base.delta_ce == 0.0
    assert base.delta_ce_se == 0.0
    assert base.delta_ce_z == 0.0
    assert base.delta_points_per_week == 0.0
    assert base.delta_playoff == 0.0
    assert base.delta_bye == 0.0
    assert base.seasons_ce_differs == 0


def test_a_candidate_equal_to_the_baseline_is_simulated_once_and_reads_zero(league, target):
    """Requesting the baseline as a candidate must not double-count it."""
    c = _sweep(league, target, [4.0, 4.0, 9.0], baseline=4.0)
    assert [p.level for p in c.points] == [4.0, 9.0]
    zero = c.points[0]
    assert zero.is_baseline and zero.delta_ce == 0.0 and zero.seasons_ce_differs == 0


def test_a_zero_step_slope_cannot_arise_from_deduplication(league, target):
    """Duplicate levels are removed, so no adjacent step is ever zero."""
    c = _sweep(league, target, [10.0, 6.0, 10.0, 6.0, 8.0])
    steps = [p.slope_step for p in c.points if p.slope_step is not None]
    assert all(s > 0 for s in steps)
    assert [p.level for p in c.points] == [4.0, 6.0, 8.0, 10.0]


# --------------------------------------------------------------------------
# 2. Request order cannot change the answer.
# --------------------------------------------------------------------------


def test_results_do_not_depend_on_the_order_levels_are_requested(league, target):
    ascending = _sweep(league, target, [6.0, 9.0, 12.0, 15.0])
    shuffled = _sweep(league, target, [15.0, 6.0, 12.0, 9.0])
    reversed_ = _sweep(league, target, [15.0, 12.0, 9.0, 6.0])
    assert ascending.rows() == shuffled.rows() == reversed_.rows()
    assert ascending.to_csv() == shuffled.to_csv() == reversed_.to_csv()


def test_the_curve_is_reported_in_ascending_level_order(curve):
    levels = [p.level for p in curve.points]
    assert levels == sorted(levels)
    assert len(set(levels)) == len(levels)


# --------------------------------------------------------------------------
# 3. Common random numbers survive the sweep.
# --------------------------------------------------------------------------


def test_only_the_swept_player_changes_across_levels(league, target):
    """Every other player's realized scores must be byte-identical."""
    from ceauction.worlds import build_pool_arrays, generate_world

    idx = league.id_to_index[target.player_id]
    worlds = []
    for level in (4.0, 12.0, 20.0):
        variant = league.with_pool_player(
            with_overrides(target, base_mean=level, crn_key=target.stream_key)
        )
        pool = build_pool_arrays(variant.pool, variant.settings)
        worlds.append(generate_world(pool, SEED, 0, 24))

    others = [i for i in range(worlds[0].pool.n_players) if i != idx]
    for w in worlds[1:]:
        assert np.array_equal(w.realized.points[:, others, :],
                              worlds[0].realized.points[:, others, :])
        # The swept player keeps every draw except the level he sits on.
        assert np.array_equal(w.availability.available, worlds[0].availability.available)
        assert np.array_equal(w.signals.level_signal, worlds[0].signals.level_signal)
        assert np.array_equal(w.pregame.weekly_state, worlds[0].pregame.weekly_state)
        assert np.array_equal(w.realized.group_effect, worlds[0].realized.group_effect)
        assert np.array_equal(w.realized.spike, worlds[0].realized.spike)
        assert np.array_equal(w.latent.role_week, worlds[0].latent.role_week)
    # ...and his own scoring does move, or nothing is being measured.
    assert not np.allclose(worlds[0].realized.points[:, idx, :],
                           worlds[2].realized.points[:, idx, :])


def test_the_swept_player_keeps_his_crn_key_and_every_other_parameter(league, target):
    from ceauction.curve import _candidate

    hot = _candidate(target, 21.0)
    assert hot.base_mean == 21.0
    assert hot.stream_key == target.stream_key
    assert hot.player_id == target.player_id
    for field in ("position", "week_sd", "season_sd", "weekly_injury_hazard",
                  "injury_mean_weeks", "spike_rate", "spike_scale", "bye_week",
                  "shock_loadings", "contingency", "proj_noise_sd",
                  "weekly_state_sd", "signal_noise_sd", "role_change_prob"):
        assert getattr(hot, field) == getattr(target, field), field


def test_the_schedule_is_shared_across_every_level(league):
    from ceauction.schedule import opponents_for_batch

    a = opponents_for_batch(SEED, 0, 32, league.settings)
    b = opponents_for_batch(SEED, 0, 32, league.settings)
    assert np.array_equal(a, b)


# --------------------------------------------------------------------------
# 4. The sweep agrees with a direct comparison.
# --------------------------------------------------------------------------


def test_a_direct_paired_comparison_reproduces_the_sweep_exactly(league, target, curve):
    """Same arms, same seed, same estimator -- so the same number, bit for bit."""
    level = 13.0
    point = next(p for p in curve.points if p.level == level)

    base_rs = league.with_pool_player(
        with_overrides(target, base_mean=4.0, crn_key=target.stream_key))
    cand_rs = league.with_pool_player(
        with_overrides(target, base_mean=level, crn_key=target.stream_key))
    direct = compare_scenarios(cand_rs, base_rs, TEAM, SIMS, SEED, label="direct")

    assert direct.delta_ce == point.delta_ce
    assert direct.delta_ce_se == point.delta_ce_se
    assert direct.report_a.championship_equity == point.championship_equity
    assert direct.report_a.playoff_probability == point.playoff_probability
    assert direct.report_a.bye_probability == point.bye_probability
    assert direct.report_a.mean_points_per_week == pytest.approx(point.points_per_week)
    assert direct.delta_points_per_week == pytest.approx(point.delta_points_per_week)

    # `seasons_ce_differs` is deliberately NOT PairedComparison's
    # `seasons_differing`: the latter counts seasons where the league champion
    # changed at all, which includes titles moving between two rivals.
    assert point.seasons_ce_differs <= direct.seasons_differing
    assert point.seasons_ce_differs > 0


def test_raw_ce_matches_an_independent_simulation_of_the_same_arm(league, target, curve):
    point = next(p for p in curve.points if p.level == 10.0)
    rs = league.with_pool_player(
        with_overrides(target, base_mean=10.0, crn_key=target.stream_key))
    out = simulate_seasons(rs, SIMS, SEED)
    assert float(out.championship_equity()[TEAM]) == point.championship_equity


# --------------------------------------------------------------------------
# 5. Adjacent slopes are genuinely paired.
# --------------------------------------------------------------------------


def test_adjacent_slopes_are_paired_against_the_previous_level(league, target, curve):
    """Recompute one slope from matched per-season indicators and compare.

    This is the property the audit asked for by name: the slope's uncertainty
    must come from a paired adjacent difference, not from combining two
    separately estimated baseline deltas.
    """
    lo, hi = 10.0, 13.0
    point = next(p for p in curve.points if p.level == hi)
    assert point.slope_from_level == lo
    assert point.slope_step == pytest.approx(hi - lo)

    runs = {}
    for level in (lo, hi):
        rs = league.with_pool_player(
            with_overrides(target, base_mean=level, crn_key=target.stream_key))
        runs[level] = simulate_seasons(rs, SIMS, SEED).champion_indicator(TEAM)

    per_point = (runs[hi] - runs[lo]) / (hi - lo)
    assert point.slope_dce_per_point == pytest.approx(float(per_point.mean()))
    assert point.slope_dce_per_point_se == pytest.approx(paired_se(per_point))
    assert point.slope_seasons_ce_differs == int(np.count_nonzero(runs[hi] != runs[lo]))


def test_the_paired_slope_se_is_not_the_unpaired_combination(league, target, curve):
    """The two estimators must not coincide, or pairing bought nothing.

    Treating the two baseline deltas as independent gives
    sqrt(se_hi^2 + se_lo^2)/step. The baseline deltas share the baseline arm
    and are strongly positively correlated, so that overstates the slope's
    uncertainty; the paired estimate must come out smaller.
    """
    lo, hi = 10.0, 13.0
    p_lo = next(p for p in curve.points if p.level == lo)
    p_hi = next(p for p in curve.points if p.level == hi)
    naive = math.sqrt(p_hi.delta_ce_se ** 2 + p_lo.delta_ce_se ** 2) / (hi - lo)
    assert p_hi.slope_dce_per_point_se < naive
    # The point estimates do agree -- only the uncertainty differs.
    assert p_hi.slope_dce_per_point == pytest.approx(
        (p_hi.delta_ce - p_lo.delta_ce) / (hi - lo)
    )


def test_the_first_level_has_no_slope(curve):
    assert curve.points[0].slope_dce_per_point is None
    assert curve.points[0].slope_from_level is None
    assert all(p.slope_dce_per_point is not None for p in curve.points[1:])


def test_slopes_are_expressed_per_projected_point(league, target):
    """A 4-point step and two 2-point steps must be on the same scale."""
    coarse = _sweep(league, target, [8.0, 12.0])
    fine = _sweep(league, target, [8.0, 10.0, 12.0])
    c = next(p for p in coarse.points if p.level == 12.0)
    f = next(p for p in fine.points if p.level == 12.0)
    assert c.slope_step == 4.0 and f.slope_step == 2.0
    # Both are dCE per point, so both are small numbers of the same order --
    # not one four times the other.
    assert abs(c.slope_dce_per_point) < 0.05
    assert abs(f.slope_dce_per_point) < 0.05


# --------------------------------------------------------------------------
# 5b. Published weekly projections move with the level.
# --------------------------------------------------------------------------


def _override_spec(base_mean=10.0, **kw):
    """A player carrying real published weekly projections."""
    from ceauction.players import PlayerSpec

    weeks = DEFAULT_LEAGUE.total_weeks
    # A deliberately lumpy shape: a bye-ish trough and a late ramp, so a pure
    # level shift is distinguishable from any rescaling.
    override = tuple(base_mean + (-4.0 if w == 6 else 0.35 * w - 2.0)
                     for w in range(weeks))
    params = dict(week_sd=6.0, season_sd=0.0, weekly_injury_hazard=0.0,
                  spike_rate=0.0, role_change_prob=0.0, proj_noise_sd=0.0,
                  shock_loadings=(), contingency=None, bye_week=0,
                  nfl_team="TST", weekly_projection_override=override)
    params.update(kw)
    return PlayerSpec(player_id=4242, name="OVR", position=Position.WR,
                      base_mean=base_mean, **params)


def test_an_override_shifts_by_exactly_the_level_delta():
    """The published shape is preserved; only its level moves."""
    from ceauction.curve import _candidate

    spec = _override_spec(base_mean=10.0)
    for level in (4.0, 10.0, 17.5, 22.0):
        cand = _candidate(spec, level)
        delta = level - spec.base_mean
        assert cand.base_mean == level
        assert len(cand.weekly_projection_override) == DEFAULT_LEAGUE.total_weeks
        for before, after in zip(spec.weekly_projection_override,
                                 cand.weekly_projection_override):
            assert after == pytest.approx(before + delta, abs=1e-12)
        # Shape preserved exactly: every week-to-week difference is unchanged.
        d_before = np.diff(np.array(spec.weekly_projection_override))
        d_after = np.diff(np.array(cand.weekly_projection_override))
        assert np.allclose(d_before, d_after, atol=1e-12)


def test_the_shift_is_measured_from_the_original_base_mean_not_chained():
    """Each candidate is built from the original spec, so deltas never compound."""
    from ceauction.curve import _candidate

    spec = _override_spec(base_mean=10.0)
    a = _candidate(spec, 15.0)
    b = _candidate(_candidate(spec, 12.0), 15.0)
    assert a.weekly_projection_override == pytest.approx(
        b.weekly_projection_override
    )
    assert a.base_mean == b.base_mean == 15.0


def test_pregame_projections_shift_by_the_level_delta():
    """The point of the fix: the manager's view moves with the player.

    Without it, `base_mean` would move realized scoring while the override kept
    every level's pregame projection identical -- so the lineup decision would
    be the same at every level and the sweep would measure the value of
    unforecastable production instead of projected production.
    """
    from ceauction.curve import _candidate
    from ceauction.worlds import build_pool_arrays, generate_world

    spec = _override_spec(base_mean=10.0)
    proj = {}
    for level in (6.0, 10.0, 18.0):
        pool = build_pool_arrays([_candidate(spec, level)], DEFAULT_LEAGUE)
        proj[level] = generate_world(pool, SEED, 0, 4).pregame.projection[:, 0, :]

    for level in (6.0, 18.0):
        shift = proj[level] - proj[10.0]
        assert np.allclose(shift, level - 10.0, atol=1e-9), (
            f"projections did not shift by {level - 10.0}"
        )
    # And the realized conditional mean moved by the same amount, so projection
    # and scoring stay consistent with each other.
    means = {}
    for level in (6.0, 18.0):
        pool = build_pool_arrays([_candidate(spec, level)], DEFAULT_LEAGUE)
        w = generate_world(pool, SEED, 0, 3000)
        means[level] = float(w.realized.points[:, 0, :].mean())
    assert means[18.0] - means[6.0] == pytest.approx(12.0, abs=0.3)


def test_the_override_shift_changes_nothing_else_about_the_player():
    """CRN key and every non-level characteristic survive the shift."""
    from ceauction.curve import _candidate

    spec = _override_spec(base_mean=10.0, crn_key=99)
    cand = _candidate(spec, 19.0)
    assert cand.stream_key == spec.stream_key == 99
    assert cand.player_id == spec.player_id
    for field in ("position", "week_sd", "season_sd", "weekly_injury_hazard",
                  "injury_mean_weeks", "spike_rate", "spike_scale", "bye_week",
                  "shock_loadings", "contingency", "proj_noise_sd",
                  "weekly_state_sd", "weekly_state_pattern",
                  "hidden_weekly_pattern", "signal_noise_sd",
                  "role_change_prob", "role_reveal_lag", "data_source"):
        assert getattr(cand, field) == getattr(spec, field), field


def test_a_player_without_an_override_is_unaffected_by_the_fix(league, target):
    """The synthetic pool carries no overrides, so nothing there changes."""
    from ceauction.curve import _candidate

    assert target.weekly_projection_override is None
    cand = _candidate(target, 17.0)
    assert cand.weekly_projection_override is None
    assert cand.base_mean == 17.0
    assert cand.stream_key == target.stream_key


def test_an_override_sweep_produces_a_rising_curve(league, target):
    """End to end: give the swept slot an override and the curve still rises.

    Before the fix this curve would have been flat in CE, because every level
    shared one pregame projection and therefore one set of lineup decisions.
    """
    from ceauction.players import with_overrides

    weeks = DEFAULT_LEAGUE.total_weeks
    override = tuple(target.base_mean + 0.2 * w for w in range(weeks))
    with_ovr = league.with_pool_player(
        with_overrides(target, weekly_projection_override=override,
                       crn_key=target.stream_key)
    )
    c = sweep_marginal_curve(with_ovr, TEAM, target.player_id,
                             baseline_level=4.0, levels=[4.0, 12.0, 20.0],
                             n_sims=SIMS, seed=SEED)
    ce = [p.championship_equity for p in c.points]
    assert ce[-1] > ce[0], f"override curve did not rise: {ce}"
    assert c.points[-1].delta_ce > 0.0
    assert c.points[-1].points_per_week > c.points[0].points_per_week


# --------------------------------------------------------------------------
# 5c. The level grid never overshoots the requested maximum.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("lo,hi,step,expected", [
    # Exactly divisible.
    (4.0, 10.0, 2.0, [4.0, 6.0, 8.0, 10.0]),
    (4.0, 22.0, 1.0, [4.0 + k for k in range(19)]),
    # Not divisible: the FINAL step shortens, and the maximum is never exceeded.
    (4.0, 10.0, 4.0, [4.0, 8.0, 10.0]),
    (4.0, 9.0, 2.0, [4.0, 6.0, 8.0, 9.0]),
    (0.0, 5.0, 3.0, [0.0, 3.0, 5.0]),
    # Step larger than the whole range.
    (4.0, 4.5, 1.0, [4.0, 4.5]),
    # Decimal steps.
    (4.0, 5.0, 0.25, [4.0, 4.25, 4.5, 4.75, 5.0]),
    (4.0, 5.0, 0.3, [4.0, 4.3, 4.6, 4.9, 5.0]),
    # A single level.
    (7.0, 7.0, 1.0, [7.0]),
])
def test_level_grid_covers_the_range_without_overshooting(lo, hi, step, expected):
    from ceauction.curve import level_grid

    grid = level_grid(lo, hi, step)
    assert grid == pytest.approx(expected, abs=1e-9)
    assert grid[0] == pytest.approx(lo)
    assert grid[-1] == hi, "the endpoint must be exact, not accumulated"
    assert max(grid) <= hi + 1e-12, f"grid exceeded max_level: {grid}"
    assert grid == sorted(grid)
    assert len(set(grid)) == len(grid)


def test_level_grid_rejects_a_bad_range():
    from ceauction.curve import level_grid

    with pytest.raises(ValueError, match="step must be positive"):
        level_grid(4.0, 10.0, 0.0)
    with pytest.raises(ValueError, match="step must be positive"):
        level_grid(4.0, 10.0, -1.0)
    with pytest.raises(ValueError, match=">= min_level"):
        level_grid(10.0, 4.0, 1.0)


def test_level_grid_endpoint_is_not_a_floating_point_accumulation():
    """0.1 twenty times is not 2.0; the endpoint must still be exact."""
    from ceauction.curve import level_grid

    grid = level_grid(0.0, 2.0, 0.1)
    assert grid[-1] == 2.0
    assert len(grid) == 21


# --------------------------------------------------------------------------
# 6. Shape: scoring rises, CE broadly rises.
# --------------------------------------------------------------------------


def test_average_points_per_week_increases_with_level(curve):
    """Nearly deterministic: the swept player scores more, so the team does."""
    pts = [p.points_per_week for p in curve.points]
    assert pts == sorted(pts), pts
    assert pts[-1] - pts[0] > 5.0
    deltas = [p.delta_points_per_week for p in curve.points]
    assert deltas == sorted(deltas)


def test_championship_equity_is_broadly_monotonic_within_uncertainty(curve):
    """The curve should rise, but monotonicity is not guaranteed a priori.

    Raising a player's level changes which players the manager starts, which
    changes team scores, the league median, records and the bracket, so a local
    decline is not automatically noise. What this test asserts is the weaker,
    honest thing: the curve rises strongly end to end, and no individual
    adjacent decrease on *this* synthetic roster is statistically significant.
    A significant decrease would be a finding to investigate, not a failure of
    the estimator.
    """
    ce = [p.championship_equity for p in curve.points]
    assert ce[-1] > ce[0] + 0.05, f"curve did not rise: {ce}"

    for p in curve.points[1:]:
        if p.slope_dce_per_point < 0:
            assert p.slope_dce_per_point_z > -2.0, (
                f"CE fell significantly from {p.slope_from_level} to {p.level} "
                f"(z = {p.slope_dce_per_point_z:+.2f})"
            )
    # And the baseline delta must be significant by the top of the range.
    assert curve.points[-1].delta_ce_z > 3.0


def test_playoff_and_bye_probabilities_also_rise(curve):
    assert curve.points[-1].playoff_probability > curve.points[0].playoff_probability
    assert curve.points[-1].bye_probability > curve.points[0].bye_probability


# --------------------------------------------------------------------------
# 7. Determinism.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("chunk", [1, 7, 64, 512])
def test_the_sweep_is_identical_across_chunk_sizes(league, target, chunk):
    ref = _sweep(league, target, [6.0, 11.0], chunk=256)
    out = _sweep(league, target, [6.0, 11.0], chunk=chunk)
    assert ref.rows() == out.rows()


def test_repeating_the_sweep_gives_identical_numbers(league, target):
    a = _sweep(league, target, [5.0, 9.0])
    b = _sweep(league, target, [5.0, 9.0])
    assert a.rows() == b.rows()


# --------------------------------------------------------------------------
# 8. The resolution report.
# --------------------------------------------------------------------------


def test_the_resolution_report_scales_as_one_over_root_n(curve):
    r = curve.resolution
    assert r.n_sims == SIMS
    assert r.seconds_per_season > 0.0
    assert r.seconds_per_paired_comparison == pytest.approx(
        2.0 * SIMS * r.seconds_per_season
    )
    assert [t.delta_ce for t in r.targets] == list(DEFAULT_RESOLUTION_TARGETS)

    # se(n) = se_obs * sqrt(n_obs/n); |z| = 2 requires se(n) <= target/2.
    for t in r.targets:
        implied = r.observed_adjacent_se * math.sqrt(SIMS / t.required_sims)
        assert implied <= t.delta_ce / 2.0 + 1e-12
        assert t.required_seconds == pytest.approx(
            2.0 * t.required_sims * r.seconds_per_season
        )
    # Smaller targets cost strictly more.
    needed = [t.required_sims for t in r.targets]
    assert needed == sorted(needed)


def test_a_target_beyond_the_live_budget_is_called_over_budget(league, target):
    """The report must say so in words, not leave the reader to compare numbers."""
    c = _sweep(league, target, [8.0, 12.0], live_auction_budget_seconds=0.001)
    verdicts = [t.verdict(0.001) for t in c.resolution.targets]
    assert all(not t.live_auction_feasible for t in c.resolution.targets)
    assert all("over budget" in v for v in verdicts)

    generous = _sweep(league, target, [8.0, 12.0],
                      live_auction_budget_seconds=1e9)
    assert all(t.live_auction_feasible for t in generous.resolution.targets)
    assert all("within budget" in t.verdict(1e9)
               for t in generous.resolution.targets)


def test_every_verdict_is_scoped_to_this_pilot(league, target):
    """No verdict may read as a general capability claim about the engine.

    Paired variance tracks the discordance rate, which differs between
    comparisons, so "0.005 is resolvable live" is not something any single
    sweep can establish. Every row must say "in this pilot".
    """
    for budget in (0.001, 1e9):
        c = _sweep(league, target, [8.0, 12.0], live_auction_budget_seconds=budget)
        for t in c.resolution.targets:
            assert "in this pilot" in t.verdict(budget), t.verdict(budget)
            assert "IMPRACTICAL" not in t.verdict(budget)
            assert "feasible live" not in t.verdict(budget)


def test_the_resolution_section_is_rendered_and_honestly_scoped(curve):
    text = curve.format()
    assert "MONTE CARLO RESOLUTION" in text
    for t in DEFAULT_RESOLUTION_TARGETS:
        assert f"{t:.4f}" in text

    # It must present itself as a scenario-specific pilot...
    assert "PILOT ESTIMATE, SCOPED TO THIS COMPARISON" in text
    assert "DISCORDANCE RATE" in text
    assert "own pilot run" in text and "adaptive rule" in text
    # ...and must NOT claim the extrapolation is a bound or conservative.
    assert "neither a bound nor conservative" in text
    lowered = text.lower()
    assert "are therefore conservative" not in lowered
    assert "somewhat fewer simulations" not in lowered


# --------------------------------------------------------------------------
# 9. Output: CSV schema and the terminal table.
# --------------------------------------------------------------------------


def test_csv_schema_is_stable_and_complete(curve, tmp_path):
    path = tmp_path / "curve.csv"
    text = curve.to_csv(str(path))
    # Written with an explicit "\n" terminator, so the file round-trips
    # byte-for-byte rather than depending on the platform's csv dialect.
    assert path.read_text(encoding="utf-8") == text
    assert "\r" not in text

    rows = list(csv.DictReader(io.StringIO(text)))
    assert len(rows) == len(curve.points)
    assert list(rows[0]) == list(MarginalCurve.CSV_COLUMNS)

    required = {
        "level", "championship_equity", "ce_se", "delta_ce", "delta_ce_se",
        "delta_ce_z", "points_per_week", "playoff_probability",
        "bye_probability", "slope_dce_per_point", "slope_dce_per_point_se",
        "seasons_ce_differs",
    }
    assert required <= set(MarginalCurve.CSV_COLUMNS)

    # Everything except the first row's slope columns parses as a float.
    for i, row in enumerate(rows):
        assert float(row["level"]) == curve.points[i].level
        for col in ("championship_equity", "delta_ce", "points_per_week"):
            float(row[col])
        if i == 0:
            assert row["slope_dce_per_point"] == ""
        else:
            float(row["slope_dce_per_point"])


def test_csv_is_not_a_dataclass_field():
    """CSV_COLUMNS is a ClassVar, so it must not appear in the constructor."""
    import dataclasses

    names = {f.name for f in dataclasses.fields(MarginalCurve)}
    assert "CSV_COLUMNS" not in names


def test_the_terminal_table_reports_every_level_and_names_its_estimator(curve):
    text = curve.format()
    for p in curve.points:
        assert f"{p.level:>6.2f}" in text
    assert "ADJACENT paired slope" in text
    assert curve.player_name in text
    assert "not a difference of two baseline" in text


# --------------------------------------------------------------------------
# 10. The optional isotonic display column.
# --------------------------------------------------------------------------


def test_isotonic_fit_is_monotone_and_preserves_the_mean():
    raw = [0.10, 0.09, 0.14, 0.13, 0.20]
    fit = isotonic_fit(raw)
    assert fit == sorted(fit)
    assert sum(fit) == pytest.approx(sum(raw))
    # Already-monotone input is returned untouched.
    ok = [0.1, 0.2, 0.3]
    assert isotonic_fit(ok) == pytest.approx(ok)
    assert isotonic_fit([]) == []


def test_isotonic_weights_pull_noisy_points_further():
    raw = [0.10, 0.30, 0.20]
    tight = isotonic_fit(raw, weights=[1.0, 100.0, 1.0])
    loose = isotonic_fit(raw, weights=[1.0, 1.0, 100.0])
    # The heavily weighted point moves least.
    assert abs(tight[1] - 0.30) < abs(loose[1] - 0.30)


def test_isotonic_display_never_overwrites_a_raw_estimate(league, target):
    levels = [6.0, 9.0, 12.0, 15.0]
    plain = _sweep(league, target, levels)
    fitted = _sweep(league, target, levels, isotonic=True)

    assert all(p.ce_isotonic is None for p in plain.points)
    assert all(p.ce_isotonic is not None for p in fitted.points)

    for a, b in zip(plain.points, fitted.points):
        assert a.championship_equity == b.championship_equity
        assert a.ce_se == b.ce_se
        assert a.ce_ci95 == b.ce_ci95
        assert a.delta_ce == b.delta_ce
        assert a.delta_ce_se == b.delta_ce_se
        assert a.slope_dce_per_point == b.slope_dce_per_point
        assert a.slope_dce_per_point_se == b.slope_dce_per_point_se

    iso = [p.ce_isotonic for p in fitted.points]
    assert iso == sorted(iso)
    text = fitted.format()
    assert "CE(iso)" in text
    assert "raw CE" in text and "unchanged by it" in text


def test_the_isotonic_column_is_described_as_an_imposed_assumption(league, target):
    """It must not be presented as noise removal.

    The manager sets lineups from noisy projections, so raising a player's
    level changes start decisions and therefore team scores, the league median,
    records and the bracket. A local decline in raw CE may be a real pathwise
    effect, so the fit imposes an assumption rather than revealing one.
    """
    from ceauction.curve import isotonic_fit

    fitted = _sweep(league, target, [6.0, 9.0, 12.0], isotonic=True)
    text = fitted.format()
    assert "IMPOSES monotonicity" in text
    assert "does not test for it" in text
    assert "may be a real pathwise effect rather than noise" in text

    doc = isotonic_fit.__doc__
    assert "imposes an assumption" in doc
    assert "not** automatically Monte" in doc
    # The old, wrong justification must not come back.
    assert "as a matter of theory" not in doc
    assert "cannot reduce championship equity" not in doc


# --------------------------------------------------------------------------
# 11. Argument validation and slot selection.
# --------------------------------------------------------------------------


def test_the_default_target_is_the_weakest_flex_eligible_player(league):
    spec = weakest_flex_slot(league, TEAM)
    assert spec.position in (Position.RB, Position.WR, Position.TE)
    on_roster = [league.spec(p) for p in league.rosters[TEAM].player_ids]
    flex = [s for s in on_roster if s.position in (Position.RB, Position.WR, Position.TE)]
    assert spec.base_mean == min(s.base_mean for s in flex)
    assert spec.player_id in league.rosters[TEAM].player_ids


def test_sweeping_a_player_who_is_not_on_the_focus_team_is_rejected(league):
    other = league.rosters[1].player_ids[0]
    with pytest.raises(KeyError, match="is not on"):
        sweep_marginal_curve(league, TEAM, other, 4.0, [8.0], 50, SEED)


def test_a_single_season_cannot_produce_a_standard_error(league, target):
    with pytest.raises(ValueError, match="at least 2"):
        sweep_marginal_curve(league, TEAM, target.player_id, 4.0, [8.0], 1, SEED)


# --------------------------------------------------------------------------
# 12. CLI.
# --------------------------------------------------------------------------


def test_cli_curve_runs_and_writes_csv(tmp_path, capsys):
    from ceauction.cli import main

    path = tmp_path / "cli.csv"
    argv = ["curve", "--sims", "60", "--min-level", "4", "--max-level", "10",
            "--step", "2", "--csv", str(path)]
    assert main(argv) == 0
    out = capsys.readouterr().out
    assert "SYNTHETIC" in out
    assert "MARGINAL CHAMPIONSHIP-EQUITY CURVE" in out
    assert "MONTE CARLO RESOLUTION" in out
    assert str(path) in out

    rows = list(csv.DictReader(io.StringIO(path.read_text(encoding="utf-8"))))
    assert [float(r["level"]) for r in rows] == [4.0, 6.0, 8.0, 10.0]
    assert list(rows[0]) == list(MarginalCurve.CSV_COLUMNS)


def test_cli_curve_accepts_an_explicit_player_and_isotonic_flag(league, capsys):
    from ceauction.cli import main

    pid = weakest_flex_slot(league, 0).player_id
    argv = ["curve", "--sims", "60", "--min-level", "5", "--max-level", "9",
            "--step", "2", "--player-id", str(pid), "--isotonic"]
    assert main(argv) == 0
    out = capsys.readouterr().out
    assert "requested explicitly" in out
    assert "CE(iso)" in out


@pytest.mark.parametrize("lo,hi,step,expected", [
    ("4", "10", "2", [4.0, 6.0, 8.0, 10.0]),      # divisible
    ("4", "10", "4", [4.0, 8.0, 10.0]),           # NOT divisible -> short last step
    ("4", "5", "0.5", [4.0, 4.5, 5.0]),           # decimal
    ("6", "6", "1", [6.0]),                       # single level
])
def test_cli_curve_grid_never_exceeds_max_level(lo, hi, step, expected,
                                                tmp_path, capsys):
    """`--min-level 4 --max-level 10 --step 4` must be 4, 8, 10 -- not 4, 8, 12."""
    from ceauction.cli import main

    path = tmp_path / f"g{lo}_{hi}_{step}.csv"
    argv = ["curve", "--sims", "40", "--min-level", lo, "--max-level", hi,
            "--step", step, "--csv", str(path)]
    assert main(argv) == 0
    out = capsys.readouterr().out

    rows = list(csv.DictReader(io.StringIO(path.read_text(encoding="utf-8"))))
    levels = [float(r["level"]) for r in rows]
    assert levels == pytest.approx(expected)
    assert max(levels) <= float(hi) + 1e-12

    # The printed header must describe the grid that was actually run.
    assert f"{len(expected)} levels from {float(lo):g} to {float(hi):g}" in out


def test_cli_reports_a_shortened_final_step(capsys):
    """When the range is not a whole number of steps, say so rather than lie."""
    from ceauction.cli import main

    assert main(["curve", "--sims", "40", "--min-level", "4",
                 "--max-level", "10", "--step", "4"]) == 0
    out = capsys.readouterr().out
    assert "3 levels from 4 to 10" in out
    assert "final step of 2" in out
    assert "not a whole number of steps" in out


def test_cli_curve_rejects_a_bad_range(capsys):
    from ceauction.cli import main

    assert main(["curve", "--sims", "20", "--step", "0"]) == 2
    assert "--step must be positive" in capsys.readouterr().err
    assert main(["curve", "--sims", "20", "--min-level", "10",
                 "--max-level", "4"]) == 2
    assert ">= --min-level" in capsys.readouterr().err


def test_cli_curve_does_not_claim_to_produce_a_price(capsys):
    """The command must not be mistakable for the pricing layer."""
    from ceauction.cli import main

    assert main(["curve", "--sims", "40", "--min-level", "4",
                 "--max-level", "6", "--step", "2"]) == 0
    out = capsys.readouterr().out
    assert "not a price" in out
