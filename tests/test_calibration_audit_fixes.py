"""The defects the player-spec calibration audit found, each pinned by a test.

**Every input is fabricated.** No real player, no vendor value.

Six things are defended here, and every one of them is a behaviour an earlier
pass got wrong rather than a feature nobody had got to yet:

* a player's identity, and therefore his random streams, is a function of the
  player and not of where his row landed in a sorted list;
* the twelve integration rosters are one fixed cast that every scenario shares;
* both readings of what the projection's health state means exist, differ in
  the right direction, and each reproduces its own target;
* injury parameters are fitted over the season the vendor's figures describe,
  and the shorter fantasy window is reported rather than substituted;
* signal quality is stated rather than inherited from scoring noise;
* the paired statistics are arithmetically what they claim to be.

The last section reads the committed documentation and fails if a superseded
claim has been left standing in it.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pytest

from ceauction.ce import paired_se
from ceauction.league import DEFAULT_LEAGUE, Position
from ceauction.players import PlayerSpec
from ceauction.realdata.identity import (
    IdentityCollision,
    assign_stable_ids,
    canonical_player_key,
    stable_player_id,
)
from ceauction.realdata.mapping import (
    AVAILABILITY_INTERPRETATIONS,
    SIGNAL_QUALITY_SCENARIOS,
    PlayerSpecMappingConfig,
    calibrate_injury,
    calibrate_level,
    map_contract_to_playerspecs,
    positional_fits_from_contract,
    resolve_signal_noise_sd,
)
from ceauction.realdata.sensitivity import (
    MIN_COMMITTED_SIMS,
    Contrast,
    ScenarioResult,
    TeamDelta,
    _paired_deltas,
)
from ceauction.realdata.smoke import (
    ROSTER_TEMPLATE,
    build_test_rosters,
    roster_assignment,
    rosters_from_assignment,
)
from ceauction.simulate import simulate_seasons

REPO = Path(__file__).resolve().parents[1]

FAST = dict(calibration_sims=20_000, injury_calibration_sims=2_000,
            availability_calibration_sims=8_000)


def _player(key, name, pos, points, *, bye=7, team="ZZA",
            injury_prob=None, games_missed=None, cv=0.6, miss=0.07):
    """One fabricated contract row."""
    return {
        "player_key": key, "name": name, "position": pos,
        "nfl_team": team, "bye_week": bye,
        "stat_line": {"rec_yards": 100.0, "receptions": 10.0,
                      "fumbles": None, "fumbles_are_lost_fumbles": None},
        "season_points": {"points": points,
                          "scoring_source": "recomputed_from_components",
                          "fumble_interpretation": "exclude",
                          "omitted_fumble_points": None},
        "active_rate": {"games_basis": 17.0,
                        "interpretation_a_full_health": points / 17.0,
                        "interpretation_b_availability_adjusted": None,
                        "preferred": None},
        "availability": {"injury_prob": injury_prob,
                         "injury_prob_definition": None,
                         "proj_games_missed": games_missed,
                         "proj_games_missed_definition": None,
                         "games_in_horizon": 17.0},
        "cohort_dispersion": {"weekly_cv": cv,
                              "weekly_cv_is_total_dispersion": True,
                              "weekly_miss_rate": miss,
                              "fit_provenance": "FIXTURE"},
        "raw_fields": {"Name": name, "Projections": "999.9"},
    }


@pytest.fixture(scope="module")
def payload():
    """A fabricated pool big enough to fill twelve legal rosters."""
    players = []
    n = 0
    for pos, count in (("QB", 40), ("RB", 55), ("WR", 80), ("TE", 30)):
        for i in range(count):
            n += 1
            players.append(_player(
                f"fab_{pos.lower()}_{i}", f"Fabricated {pos}{i:03d}", pos,
                points=float(340 - 1.4 * n),
                bye=5 + (i % 10),
                injury_prob=0.30 if i % 4 else None,
                games_missed=1.5 if i % 4 else None,
                cv={"QB": 0.44, "RB": 0.62, "WR": 0.65, "TE": 0.74}[pos],
                miss={"QB": 0.05, "RB": 0.07, "WR": 0.07, "TE": 0.11}[pos]))
    return {"schema_version": "1.0.0", "players": players}


@pytest.fixture(scope="module")
def fits():
    return ({"QB": 0.44, "RB": 0.62, "WR": 0.65, "TE": 0.74},
            {"QB": 0.05, "RB": 0.07, "WR": 0.07, "TE": 0.11})


# ==========================================================================
# 1. Stable identity
# ==========================================================================


def test_the_canonical_key_is_one_key_per_person():
    """A slug and a display name must not become two different people."""
    assert canonical_player_key("josh_allen") == canonical_player_key(name="Josh Allen")
    assert canonical_player_key("aj_brown") == canonical_player_key(name="A.J. Brown Jr.")
    assert canonical_player_key("fab_wr_1") != canonical_player_key("fab_wr_2")


def test_the_id_is_a_pure_function_of_the_key():
    a = stable_player_id("fabricated_person")
    b = stable_player_id("fabricated_person")
    assert a == b
    assert a > 0
    assert a < (1 << 62), "ids stay clear of the int64 sign bit"


def test_a_collision_is_raised_not_absorbed(monkeypatch):
    """Two players sharing one random stream would look like a correlation."""
    import ceauction.realdata.identity as ident
    monkeypatch.setattr(ident, "stable_player_id", lambda key: 7)
    with pytest.raises(IdentityCollision, match="two players one random stream|two players"):
        ident.assign_stable_ids(["fab_a", "fab_b"])


def test_the_same_key_twice_is_not_a_collision():
    """Only *distinct* keys colliding is a defect."""
    out = assign_stable_ids(["fab_a", "fab_a", "fab_b"])
    assert set(out) == {"fab_a", "fab_b"}


def test_ids_survive_a_reordered_input(payload):
    """The defect: ids used to be the row index of a points-sorted list."""
    cfg = PlayerSpecMappingConfig(**FAST)
    forward = map_contract_to_playerspecs(payload, cfg, limit=30)
    shuffled = {"schema_version": "1.0.0",
                "players": list(reversed(payload["players"]))}
    backward = map_contract_to_playerspecs(shuffled, cfg, limit=30)
    assert {m.canonical_key: m.spec.player_id for m in forward.players} == \
           {m.canonical_key: m.spec.player_id for m in backward.players}


def test_ids_survive_a_pool_limit(payload):
    """A player in the top 10 keeps his id when 60 players are mapped."""
    cfg = PlayerSpecMappingConfig(**FAST)
    small = map_contract_to_playerspecs(payload, cfg, limit=10)
    large = map_contract_to_playerspecs(payload, cfg, limit=60)
    big = {m.canonical_key: m.spec.player_id for m in large.players}
    for m in small.players:
        assert big[m.canonical_key] == m.spec.player_id


def test_ids_survive_a_changed_ranking(payload):
    """Rewriting the points column reorders every row and moves no id."""
    cfg = PlayerSpecMappingConfig(**FAST)
    before = map_contract_to_playerspecs(payload, cfg, limit=40)
    rescored = {"schema_version": "1.0.0", "players": [
        {**p, "season_points": {**p["season_points"],
                                "points": 500.0 - p["season_points"]["points"]}}
        for p in payload["players"]]}
    after = map_contract_to_playerspecs(rescored, cfg, limit=None)
    seen = {m.canonical_key: m.spec.player_id for m in after.players}
    for m in before.players:
        assert seen[m.canonical_key] == m.spec.player_id


def test_ids_and_crn_survive_every_scenario_axis(payload):
    """The property the whole paired design rests on."""
    keys = [f"fab_wr_{i}" for i in range(20)]
    reference = None
    scenarios = [
        PlayerSpecMappingConfig(**FAST),
        PlayerSpecMappingConfig(target="mean_target", **FAST),
        PlayerSpecMappingConfig(fumble_interpretation="lost", **FAST),
        PlayerSpecMappingConfig(forecastable_share=0.5, **FAST),
        PlayerSpecMappingConfig(season_sd_fraction=0.2, **FAST),
        PlayerSpecMappingConfig(signal_quality="none", **FAST),
        PlayerSpecMappingConfig(injury_model="positional", **FAST),
        PlayerSpecMappingConfig(
            projection_availability_interpretation="availability_adjusted",
            **FAST),
    ]
    for cfg in scenarios:
        res = map_contract_to_playerspecs(payload, cfg, only_keys=keys,
                                          positional_miss={"WR": 0.07},
                                          positional_cv={"WR": 0.65})
        got = {m.canonical_key: (m.spec.player_id, m.spec.crn_key,
                                 m.spec.stream_key)
               for m in res.players}
        assert got, "every scenario must map the requested players"
        if reference is None:
            reference = got
        else:
            assert got == reference, f"identity moved under {cfg.label()}"


def test_crn_key_equals_player_id_so_streams_follow_identity(payload):
    cfg = PlayerSpecMappingConfig(**FAST)
    for m in map_contract_to_playerspecs(payload, cfg, limit=10).players:
        assert m.spec.crn_key == m.spec.player_id
        assert m.spec.stream_key == m.spec.player_id


# ==========================================================================
# 2. One fixed cast of twelve rosters
# ==========================================================================


def test_the_roster_assignment_is_reusable_across_scenarios(payload, fits):
    """Rebuilding the snake per scenario would let the league reshuffle."""
    cv, miss = fits
    base_cfg = PlayerSpecMappingConfig(**FAST)
    base = map_contract_to_playerspecs(payload, base_cfg, positional_cv=cv,
                                       positional_miss=miss, limit=250)
    rosters = build_test_rosters(base.specs)
    assignment = roster_assignment(rosters)
    rostered = {pid for team in assignment for pid in team}
    keys = [m.canonical_key for m in base.players
            if m.spec.player_id in rostered]

    other = PlayerSpecMappingConfig(
        projection_availability_interpretation="availability_adjusted", **FAST)
    scen = map_contract_to_playerspecs(payload, other, positional_cv=cv,
                                       positional_miss=miss, only_keys=keys)
    rebuilt = rosters_from_assignment(assignment, scen.specs)

    assert roster_assignment(rebuilt) == assignment
    assert rebuilt.team_names == rosters.team_names


def test_a_rebuilt_scenario_holds_exactly_the_same_people(payload, fits):
    cv, miss = fits
    base = map_contract_to_playerspecs(
        payload, PlayerSpecMappingConfig(**FAST), positional_cv=cv,
        positional_miss=miss, limit=250)
    rosters = build_test_rosters(base.specs)
    assignment = roster_assignment(rosters)
    rostered = {pid for team in assignment for pid in team}
    keys = [m.canonical_key for m in base.players if m.spec.player_id in rostered]

    scen = map_contract_to_playerspecs(
        payload, PlayerSpecMappingConfig(season_sd_fraction=0.2, **FAST),
        positional_cv=cv, positional_miss=miss, only_keys=keys)
    rebuilt = rosters_from_assignment(assignment, scen.specs)

    for team in range(DEFAULT_LEAGUE.n_teams):
        a = [rosters.spec(p).name for p in rosters.rosters[team].player_ids]
        b = [rebuilt.spec(p).name for p in rebuilt.rosters[team].player_ids]
        assert a == b


def test_rebuilding_against_a_missing_player_raises(payload, fits):
    """Silently substituting anyone would break the pairing."""
    cv, miss = fits
    base = map_contract_to_playerspecs(
        payload, PlayerSpecMappingConfig(**FAST), positional_cv=cv,
        positional_miss=miss, limit=250)
    rosters = build_test_rosters(base.specs)
    assignment = roster_assignment(rosters)
    short = [s for s in base.specs if s.player_id != assignment[0][0]]
    with pytest.raises(KeyError, match="absent from this scenario"):
        rosters_from_assignment(assignment, short)


def test_only_keys_restricts_the_mapping_to_the_named_cast(payload):
    keys = ["fab_rb_3", "fab_wr_9", "fab_qb_1"]
    res = map_contract_to_playerspecs(payload, PlayerSpecMappingConfig(**FAST),
                                      only_keys=keys)
    assert sorted(m.canonical_key for m in res.players) == sorted(keys)


# ==========================================================================
# 3. The availability-treatment axis
# ==========================================================================


def test_both_availability_interpretations_exist_and_neither_is_preferred():
    assert AVAILABILITY_INTERPRETATIONS == ("full_health", "availability_adjusted")
    for interp in AVAILABILITY_INTERPRETATIONS:
        PlayerSpecMappingConfig(projection_availability_interpretation=interp)
    with pytest.raises(ValueError, match="projection_availability_interpretation"):
        PlayerSpecMappingConfig(projection_availability_interpretation="preferred")


def test_the_two_interpretations_produce_different_levels_in_the_right_direction():
    """Availability-adjusted must be HIGHER for a player who misses games."""
    injured = calibrate_injury(0.40, 3.0, PlayerSpecMappingConfig(**FAST),
                               position="RB", bye_index=6)
    fh = calibrate_level(
        238.0, 0.62, 0.0,
        PlayerSpecMappingConfig(projection_availability_interpretation="full_health",
                                **FAST),
        injury=injured, bye_index=6)
    aa = calibrate_level(
        238.0, 0.62, 0.0,
        PlayerSpecMappingConfig(
            projection_availability_interpretation="availability_adjusted", **FAST),
        injury=injured, bye_index=6)
    assert aa.level > fh.level
    assert fh.level == pytest.approx(238.0 / 17.0, rel=0.01)


def test_a_player_who_misses_nothing_sees_almost_no_gap():
    """The gap between the two readings is the projected absence, nothing else."""
    healthy = calibrate_injury(None, None, PlayerSpecMappingConfig(
        injury_model="none", **FAST))
    assert healthy.weekly_injury_hazard == 0.0
    fh = calibrate_level(238.0, 0.62, 0.0, PlayerSpecMappingConfig(
        projection_availability_interpretation="full_health", **FAST),
        injury=healthy, bye_index=6)
    aa = calibrate_level(238.0, 0.62, 0.0, PlayerSpecMappingConfig(
        projection_availability_interpretation="availability_adjusted", **FAST),
        injury=healthy, bye_index=6)
    assert aa.level == pytest.approx(fh.level, rel=0.02)


def test_full_health_leaves_unconditional_output_below_the_source_total():
    """Stated in the audit as the expected consequence, so it is asserted."""
    injured = calibrate_injury(0.40, 3.0, PlayerSpecMappingConfig(**FAST),
                               position="RB", bye_index=6)
    fh = calibrate_level(238.0, 0.62, 0.0, PlayerSpecMappingConfig(
        projection_availability_interpretation="full_health", **FAST),
        injury=injured, bye_index=6)
    assert fh.expected_unconditional_season_output < fh.target_value
    assert fh.unconditional_shortfall > 0.0
    # Expected output is exactly the level times the games actually played.
    assert fh.expected_unconditional_season_output == pytest.approx(
        fh.level * (17 - injured.achieved_games_missed), rel=1e-12)
    # So the shortfall is the absence, plus whatever Monte Carlo error the
    # unit statistic left on the level. The second term is small but it is
    # not zero, and pretending it were would be the kind of quiet rounding
    # this whole pass exists to remove.
    mc = fh.target_value - fh.level * 17
    assert fh.unconditional_shortfall == pytest.approx(
        mc + fh.level * injured.achieved_games_missed, rel=1e-9)
    assert abs(mc) < 0.02 * fh.target_value


def test_availability_adjusted_reproduces_its_own_target_after_absence():
    """Simulate the fitted process at the fitted level and land on the total."""
    cfg = PlayerSpecMappingConfig(
        projection_availability_interpretation="availability_adjusted",
        target="mean_target", **FAST)
    injured = calibrate_injury(0.40, 3.0, cfg, position="RB", bye_index=6)
    cal = calibrate_level(238.0, 0.62, 0.0, cfg, injury=injured, bye_index=6)

    from ceauction.realdata.mapping import _simulate_availability
    from ceauction import rng
    sims, weeks = 40_000, cfg.full_season_weeks
    _, _, played = _simulate_availability(
        injured.weekly_injury_hazard, injured.injury_mean_weeks, 6, weeks,
        sims, 4242, return_played=True)
    s = np.arange(sims, dtype=np.int64).reshape(sims, 1)
    w = np.arange(weeks, dtype=np.int64).reshape(1, weeks)
    weekly = cal.level + rng.normal(4242, rng.Kind.WEEK_NOISE, s, w) * 0.62 * cal.level
    total = (weekly * played).mean(axis=0).sum()
    # An independent seed and 40k draws: agreement to ~1% is the calibration
    # working, not the same random numbers being reused.
    assert total == pytest.approx(238.0, rel=0.02)


def test_median_and_mean_separate_once_availability_is_in_the_model():
    """The claim the audit forbids until absences are included.

    Under full health every component is symmetric and the two targets return
    the same level. Absences truncate the lower tail only, so under
    availability_adjusted they must not.
    """
    inj = calibrate_injury(0.45, 3.5, PlayerSpecMappingConfig(**FAST),
                           position="RB", bye_index=6)
    levels = {}
    for interp in AVAILABILITY_INTERPRETATIONS:
        for target in ("median_target", "mean_target"):
            cfg = PlayerSpecMappingConfig(
                target=target, projection_availability_interpretation=interp,
                **FAST)
            levels[(interp, target)] = calibrate_level(
                238.0, 0.62, 0.0, cfg, injury=inj, bye_index=6).level

    fh_gap = abs(levels[("full_health", "median_target")]
                 - levels[("full_health", "mean_target")])
    aa_gap = abs(levels[("availability_adjusted", "median_target")]
                 - levels[("availability_adjusted", "mean_target")])
    fh_level = levels[("full_health", "mean_target")]
    assert fh_gap / fh_level < 0.01, "full health: symmetric, so they agree"
    assert aa_gap / fh_level > 0.02, "with absences they must separate"
    assert aa_gap > 5 * fh_gap


def test_the_mapping_carries_the_interpretation_through_to_the_spec(payload):
    for interp in AVAILABILITY_INTERPRETATIONS:
        cfg = PlayerSpecMappingConfig(
            projection_availability_interpretation=interp, **FAST)
        res = map_contract_to_playerspecs(payload, cfg, limit=5)
        assert all(m.level.availability_interpretation == interp
                   for m in res.players)
        assert all(interp in m.spec.notes for m in res.players)
        assert interp[:2] in cfg.label()


# ==========================================================================
# 4. The injury horizon
# ==========================================================================


def test_the_three_horizons_are_distinct_and_named():
    cfg = PlayerSpecMappingConfig()
    assert cfg.full_season_weeks == 18, "18 calendar weeks"
    assert cfg.full_season_games == 17, "17 scheduled NFL games"
    assert cfg.fantasy_weeks == 17, "fantasy Weeks 1-17"
    assert cfg.fantasy_scheduled_games == 16, "16 of them are games"
    assert cfg.full_season_weeks - cfg.full_season_games == 1, "one bye"
    assert cfg.fantasy_weeks - cfg.fantasy_scheduled_games == 1, "one bye"


def test_fantasy_weeks_one_to_seventeen_contain_sixteen_scheduled_games():
    assert DEFAULT_LEAGUE.total_weeks == 17
    assert DEFAULT_LEAGUE.total_weeks - 1 == 16


def test_neither_injury_target_is_rescaled():
    """The defect: games missed was scaled 16/17, injury probability was not."""
    cfg = PlayerSpecMappingConfig(**FAST)
    cal = calibrate_injury(0.35, 1.7, cfg, position="RB", bye_index=8)
    assert cal.target_games_missed == pytest.approx(1.7)
    assert cal.target_injury_prob == pytest.approx(0.35)


def test_both_full_season_targets_are_matched_on_the_eighteen_week_season():
    cfg = PlayerSpecMappingConfig(**FAST)
    for prob, missed in ((0.35, 1.7), (0.20, 1.0), (0.55, 4.0)):
        cal = calibrate_injury(prob, missed, cfg, position="RB", bye_index=6)
        assert cal.full_season_weeks == 18 and cal.full_season_games == 17
        if cal.feasible:
            assert abs(cal.injury_prob_error) <= 0.02
            assert abs(cal.games_missed_error) <= 0.15


def test_the_fantasy_window_is_reported_separately_never_substituted():
    cfg = PlayerSpecMappingConfig(**FAST)
    cal = calibrate_injury(0.35, 2.0, cfg, position="RB", bye_index=6)
    assert cal.expected_fantasy_games_missed is not None
    assert cal.expected_fantasy_games_missed != cal.target_games_missed
    # 16 of the 17 scheduled games sit inside the fantasy window, and the
    # window is one week shorter, so fewer games are lost inside it.
    assert cal.expected_fantasy_games_missed < cal.achieved_games_missed
    assert cal.expected_fantasy_games_missed > 0.6 * cal.achieved_games_missed


def test_the_fitted_parameters_are_per_week_and_carry_across_horizons():
    """Nothing is refitted for the fantasy window; the same rates are reused."""
    cfg = PlayerSpecMappingConfig(**FAST)
    cal = calibrate_injury(0.35, 2.0, cfg, position="RB", bye_index=6)
    from ceauction.realdata.mapping import _simulate_availability
    _, missed = _simulate_availability(
        cal.weekly_injury_hazard, cal.injury_mean_weeks, 6,
        cfg.fantasy_weeks, cfg.injury_calibration_sims, cfg.calibration_seed)
    assert missed == pytest.approx(cal.expected_fantasy_games_missed, abs=1e-9)


def test_a_longer_horizon_loses_more_games_than_a_shorter_one():
    """Sanity on the direction, so a horizon swap could not pass unnoticed."""
    from ceauction.realdata.mapping import _simulate_availability
    _, m17 = _simulate_availability(0.06, 3.0, 6, 17, 6000, 11)
    _, m18 = _simulate_availability(0.06, 3.0, 6, 18, 6000, 11)
    assert m18 > m17


def test_the_calibration_summary_reports_all_three_horizon_quantities(payload, fits):
    cv, miss = fits
    res = map_contract_to_playerspecs(
        payload, PlayerSpecMappingConfig(**FAST), positional_cv=cv,
        positional_miss=miss, limit=40)
    s = res.calibration_summary()
    assert s["full_season_injury_prob_abs_error"] is not None
    assert s["full_season_games_missed_abs_error"] is not None
    assert s["fantasy_games_missed_expected"] is not None
    assert s["horizons"] == {"calibration_calendar_weeks": 18,
                             "calibration_scheduled_games": 17,
                             "fantasy_weeks": 17,
                             "fantasy_scheduled_games": 16}


# ==========================================================================
# 5. Explicit signal quality
# ==========================================================================


def test_signal_quality_is_a_stated_scenario():
    assert SIGNAL_QUALITY_SCENARIOS == ("none", "week_sd", "2x_week_sd")
    with pytest.raises(ValueError, match="signal_quality"):
        PlayerSpecMappingConfig(signal_quality="whatever")


def test_the_three_signal_qualities_resolve_as_documented():
    assert resolve_signal_noise_sd("week_sd", 6.0) == 6.0
    assert resolve_signal_noise_sd("2x_week_sd", 6.0) == 12.0
    assert math.isinf(resolve_signal_noise_sd("none", 6.0))


def test_no_real_spec_leaves_signal_noise_implicit(payload):
    """The defect: None silently became week_sd, tying learning to scoring noise."""
    for sq in SIGNAL_QUALITY_SCENARIOS:
        cfg = PlayerSpecMappingConfig(signal_quality=sq, season_sd_fraction=0.1,
                                      **FAST)
        res = map_contract_to_playerspecs(payload, cfg, limit=8)
        assert res.players
        for m in res.players:
            assert m.spec.signal_noise_sd is not None
            assert m.spec.data_source.startswith("REAL:")
            assert f"sig={sq}" in m.spec.notes
        if sq == "week_sd":
            assert all(m.spec.signal_noise_sd == pytest.approx(m.spec.week_sd)
                       for m in res.players)
        elif sq == "2x_week_sd":
            assert all(m.spec.signal_noise_sd == pytest.approx(2 * m.spec.week_sd)
                       for m in res.players)
        else:
            assert all(math.isinf(m.spec.signal_noise_sd) for m in res.players)


def test_no_learning_freezes_the_projection_at_consensus():
    """An infinite signal SD must give a posterior of exactly zero, not a small one."""
    from ceauction.worlds import build_pool_arrays, generate_world
    specs = [PlayerSpec(player_id=i, name=f"Fab{i}", position=Position.WR,
                        nfl_team="ZZA", base_mean=12.0, week_sd=6.0,
                        season_sd=4.0,
                        signal_noise_sd=(math.inf if i == 0 else 6.0))
             for i in range(2)]
    world = generate_world(build_pool_arrays(specs, DEFAULT_LEAGUE), 99, 0, 64)
    post = world.pregame.posterior_mean
    assert np.all(post[:, 0, :] == 0.0), "no learning means exactly no learning"
    assert np.abs(post[:, 1, :]).max() > 0.0, "the control still learns"
    assert np.isfinite(world.pregame.projection).all()


def test_no_learning_differs_from_nothing_to_learn():
    """season_sd = 0 and signal_quality = none are different worlds."""
    from ceauction.worlds import build_pool_arrays, generate_world

    def spread(season_sd, signal):
        specs = [PlayerSpec(player_id=0, name="Fab", position=Position.WR,
                            nfl_team="ZZA", base_mean=12.0, week_sd=6.0,
                            season_sd=season_sd, signal_noise_sd=signal)]
        w = generate_world(build_pool_arrays(specs, DEFAULT_LEAGUE), 5, 0, 64)
        return float(np.std(w.realized.points))

    # Nobody learns in either case, but only one of them has a latent shift.
    assert spread(4.0, math.inf) > spread(0.0, math.inf)


def test_signal_quality_moves_the_posterior_in_the_right_direction():
    """Noisier usage must teach less, given the same latent uncertainty."""
    from ceauction.worlds import build_pool_arrays, generate_world

    def learned(signal_sd):
        specs = [PlayerSpec(player_id=0, name="Fab", position=Position.WR,
                            nfl_team="ZZA", base_mean=12.0, week_sd=6.0,
                            season_sd=4.0, signal_noise_sd=signal_sd)]
        w = generate_world(build_pool_arrays(specs, DEFAULT_LEAGUE), 5, 0, 256)
        return float(np.abs(w.pregame.posterior_mean[:, 0, -1]).mean())

    assert learned(6.0) > learned(12.0) > learned(48.0)


# ==========================================================================
# 6. The paired statistics
# ==========================================================================


class _FakeOutcomes:
    """Only the field ``_paired_deltas`` reads."""

    def __init__(self, champion):
        self.champion = np.asarray(champion)


def test_paired_delta_and_se_are_what_they_claim_to_be():
    base = _FakeOutcomes([0, 0, 1, 1, 2, 2, 0, 1])
    scen = _FakeOutcomes([0, 1, 1, 0, 2, 0, 0, 1])
    deltas, discord = _paired_deltas(base, scen, ["T0", "T1", "T2"])

    # Team 0 wins seasons {0, 1, 6} in the baseline and {0, 3, 5, 6} in the
    # scenario, so the paired difference is +1 twice, -1 once, 0 elsewhere.
    b0 = np.array([1., 1, 0, 0, 0, 0, 1, 0])
    a0 = np.array([1., 0, 0, 1, 0, 1, 1, 0])
    d0 = a0 - b0
    assert deltas[0].delta_ce == pytest.approx(d0.mean())
    assert deltas[0].delta_ce_se == pytest.approx(paired_se(d0))
    assert deltas[0].ce_baseline == pytest.approx(3 / 8)
    assert deltas[0].ce_scenario == pytest.approx(4 / 8)
    assert deltas[0].discordance == pytest.approx(3 / 8)
    # Seasons 1, 3 and 5 crown a different champion.
    assert discord == pytest.approx(3 / 8)


def test_the_deltas_sum_to_zero_across_teams():
    """Exactly one champion per season in each arm, so the deltas must cancel."""
    rs = np.random.default_rng(3)
    base = _FakeOutcomes(rs.integers(0, 12, 500))
    scen = _FakeOutcomes(rs.integers(0, 12, 500))
    deltas, _ = _paired_deltas(base, scen, [f"T{i}" for i in range(12)])
    assert sum(d.delta_ce for d in deltas) == pytest.approx(0.0, abs=1e-12)


def test_an_identical_arm_gives_exactly_zero_with_zero_uncertainty():
    """The strongest property common random numbers buy."""
    champ = _FakeOutcomes([0, 3, 7, 3, 11, 0, 2, 2])
    deltas, discord = _paired_deltas(champ, champ, [f"T{i}" for i in range(12)])
    assert discord == 0.0
    for d in deltas:
        assert d.delta_ce == 0.0
        assert d.delta_ce_se == 0.0
        assert d.ci95 == (0.0, 0.0)
        assert d.discordance == 0.0
        assert not d.resolved, "a zero interval touching zero is not a finding"


def test_the_interval_is_the_delta_plus_or_minus_1_96_se():
    d = TeamDelta(0, "T0", 0.10, 0.13, 0.03, 0.01, 0.4)
    lo, hi = d.ci95
    assert lo == pytest.approx(0.03 - 1.96 * 0.01)
    assert hi == pytest.approx(0.03 + 1.96 * 0.01)
    assert d.z == pytest.approx(3.0)
    assert d.resolved


def test_an_interval_straddling_zero_is_not_resolved():
    assert not TeamDelta(0, "T0", 0.1, 0.11, 0.010, 0.008, 0.3).resolved
    assert TeamDelta(0, "T0", 0.1, 0.13, 0.030, 0.008, 0.3).resolved
    assert TeamDelta(0, "T0", 0.1, 0.07, -0.030, 0.008, 0.3).resolved


def test_discordance_bounds_the_delta():
    """|delta CE| can never exceed the rate at which the outcome changed."""
    rs = np.random.default_rng(11)
    base = _FakeOutcomes(rs.integers(0, 12, 2000))
    scen = _FakeOutcomes(rs.integers(0, 12, 2000))
    deltas, _ = _paired_deltas(base, scen, [f"T{i}" for i in range(12)])
    for d in deltas:
        assert abs(d.delta_ce) <= d.discordance + 1e-12


def test_a_scenario_reports_only_its_own_resolved_effects():
    teams = (TeamDelta(0, "T0", 0.1, 0.14, 0.04, 0.010, 0.5),   # resolved
             TeamDelta(1, "T1", 0.1, 0.11, 0.01, 0.020, 0.5),   # not
             TeamDelta(2, "T2", 0.1, 0.05, -0.05, 0.010, 0.5))  # resolved
    s = ScenarioResult(
        label="x", axis="a", target="median_target",
        availability_interpretation="full_health", signal_quality="week_sd",
        forecastable_share=0.0, season_sd_fraction=0.0,
        injury_model="individual", fumble_interpretation="exclude",
        ce=(0.1,), mean_points_per_week=90.0, players_mapped=180,
        infeasible_injuries=0, team_deltas=teams, champion_discordance=0.5)
    assert s.any_resolved
    assert len(s.resolved_teams) == 2
    assert s.max_resolved_delta == pytest.approx(0.05)
    assert abs(s.largest_delta.delta_ce) == pytest.approx(0.05)


def test_an_axis_with_nothing_resolved_reports_nothing_resolved():
    teams = (TeamDelta(0, "T0", 0.1, 0.11, 0.01, 0.02, 0.4),)
    s = ScenarioResult(
        label="x", axis="a", target="median_target",
        availability_interpretation="full_health", signal_quality="week_sd",
        forecastable_share=0.0, season_sd_fraction=0.0,
        injury_model="individual", fumble_interpretation="exclude",
        ce=(0.1,), mean_points_per_week=90.0, players_mapped=180,
        infeasible_injuries=0, team_deltas=teams)
    assert not s.any_resolved
    assert s.max_resolved_delta == 0.0
    assert abs(s.largest_delta.delta_ce) == pytest.approx(0.01)


def test_a_contrast_compares_two_scenarios_not_a_scenario_and_a_baseline():
    """The season_sd answer at two learning speeds needs its own paired run.

    Reading that difference off two overlapping baseline intervals is exactly
    the class of error this pass exists to remove.
    """
    a = _FakeOutcomes([0, 0, 1, 1, 2, 2, 0, 1])
    b = _FakeOutcomes([0, 1, 1, 0, 2, 0, 0, 1])
    deltas, disc = _paired_deltas(a, b, ["T0", "T1", "T2"])
    c = Contrast(name="ssd=0.10: sig=none vs sig=week_sd", label_a="A",
                 label_b="B", question="q", team_deltas=deltas,
                 discordance=disc)
    assert c.largest_delta is not None
    assert abs(c.largest_delta.delta_ce) == max(abs(d.delta_ce) for d in deltas)
    assert c.max_resolved_delta == max(
        (abs(d.delta_ce) for d in deltas if d.resolved), default=0.0)
    assert c.any_resolved == bool(c.resolved_teams)


def test_a_contrast_between_identical_arms_resolves_nothing():
    same = _FakeOutcomes([0, 3, 7, 3, 11, 0, 2, 2])
    deltas, disc = _paired_deltas(same, same, [f"T{i}" for i in range(12)])
    c = Contrast(name="n", label_a="A", label_b="B", question="q",
                 team_deltas=deltas, discordance=disc)
    assert not c.any_resolved
    assert c.max_resolved_delta == 0.0
    assert c.discordance == 0.0


def test_the_committed_report_has_a_documented_sample_floor():
    assert MIN_COMMITTED_SIMS >= 16_000


def test_an_underpowered_committed_run_is_refused():
    from ceauction.realdata.sensitivity import run_sensitivity
    with pytest.raises(ValueError, match="at least 16,000 seasons"):
        run_sensitivity({"exclude": {"players": []}}, {}, {}, n_sims=2000,
                        require_minimum=True)


def test_pairing_actually_pairs_on_real_simulated_worlds(payload, fits):
    """End to end: an unchanged scenario must give a byte-identical arm.

    This is what makes a paired standard error legitimate. If ids or streams
    moved between arms, identical parameters would still give different
    seasons and this would fail.
    """
    cv, miss = fits
    cfg = PlayerSpecMappingConfig(**FAST)
    mapped = map_contract_to_playerspecs(payload, cfg, positional_cv=cv,
                                         positional_miss=miss, limit=250)
    rosters = build_test_rosters(mapped.specs)
    assignment = roster_assignment(rosters)
    rostered = {pid for team in assignment for pid in team}
    keys = [m.canonical_key for m in mapped.players if m.spec.player_id in rostered]

    again = map_contract_to_playerspecs(payload, cfg, positional_cv=cv,
                                        positional_miss=miss, only_keys=keys)
    rebuilt = rosters_from_assignment(assignment, again.specs)

    a = simulate_seasons(rosters, 300, 4242)
    b = simulate_seasons(rebuilt, 300, 4242)
    assert np.array_equal(a.champion, b.champion)
    deltas, discord = _paired_deltas(a, b, rosters.team_names)
    assert discord == 0.0
    assert all(d.delta_ce == 0.0 and d.delta_ce_se == 0.0 for d in deltas)


# ==========================================================================
# 7. The documentation carries one current story
# ==========================================================================

#: Claims that were true of an earlier pass and are not true now. A committed
#: document may discuss them as superseded history, but must not assert them,
#: so each pattern is paired with the markers that make it retrospective.
SUPERSEDED_CLAIMS = (
    (r"vendor documentation (was|could) not (be )?locat",
     "the published methodology is now cited in the contract's provenance"),
    (r"no weekly injury process has been derived",
     "hazard and duration are now solved against both season targets"),
    (r"scaled by 16/17|16/17 before solving",
     "both injury targets are now matched on the full season"),
)

DOCS = ("OPEN_QUESTIONS.md", "HANDOFF.md", "docs/CALIBRATION_AUDIT.md")

#: A line is exempt when it is explicitly flagging the claim as withdrawn.
_RETRACTION = re.compile(
    r"withdraw|supersed|no longer|used to|previously|earlier pass|"
    r"was wrong|corrected|retract|before this pass|historic", re.I)


def _assertive_lines(text: str, pattern: str):
    rx = re.compile(pattern, re.I)
    for i, line in enumerate(text.splitlines(), 1):
        if rx.search(line) and not _RETRACTION.search(line):
            yield i, line.strip()


@pytest.mark.parametrize("doc", DOCS)
def test_no_document_still_asserts_a_superseded_claim(doc):
    path = REPO / doc
    text = path.read_text(encoding="utf-8")
    problems = []
    for pattern, why in SUPERSEDED_CLAIMS:
        for lineno, line in _assertive_lines(text, pattern):
            problems.append(f"{doc}:{lineno}: {line!r} -- {why}")
    assert not problems, "superseded claims left standing:\n" + "\n".join(problems)


@pytest.mark.parametrize("doc", DOCS)
def test_no_document_calls_the_source_simply_a_median(doc):
    """The total is a hybrid, and calling it a median is the claim to avoid."""
    text = (REPO / doc).read_text(encoding="utf-8")
    rx = re.compile(r"(source|projection|total)s? (is|are) (simply |just |)"
                    r"(a |the |)median", re.I)
    bad = [f"{doc}:{i}: {ln.strip()!r}"
           for i, ln in enumerate(text.splitlines(), 1)
           if rx.search(ln) and not _RETRACTION.search(ln)]
    assert not bad, "\n".join(bad)


def test_one_source_of_truth_description_is_stated_and_agrees_everywhere():
    """The five facts that must be findable, and must not contradict."""
    audit = (REPO / "docs/CALIBRATION_AUDIT.md").read_text(encoding="utf-8").lower()
    for phrase in ("hybrid_market_location",
                   "market median",
                   "probability-weighted expectation",
                   "17 nfl games",
                   "partial"):
        assert phrase in audit, f"CALIBRATION_AUDIT.md must state: {phrase}"


def test_the_withdrawn_numbers_are_named_as_withdrawn():
    """Three specific figures the audit required withdrawing."""
    text = (REPO / "docs/CALIBRATION_AUDIT.md").read_text(encoding="utf-8")
    lowered = text.lower()
    assert "withdraw" in lowered
    for figure in ("0.0005", "0.0100"):
        assert figure in text, f"{figure} must be named as withdrawn, not deleted"
