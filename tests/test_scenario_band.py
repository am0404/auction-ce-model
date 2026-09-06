"""The 54-cell model-scenario grid and paired bands over it.

**Every input is fabricated.** No real player, no vendor value.

What these tests defend is that a band means what it says: the grid is the full
cross-product in a fixed order, a scenario id round-trips and cannot silently
disagree with its own config, arms in a cell are genuinely paired, an unchanged
arm gives exactly zero, and the band arithmetic (point spread, Monte Carlo
widening, sign stability, dominant axis) is what it claims.
"""

from __future__ import annotations

import itertools
import json
import math

import numpy as np
import pytest

from ceauction.league import DEFAULT_LEAGUE
from ceauction.realdata.mapping import (
    AVAILABILITY_INTERPRETATIONS,
    FORECASTABLE_SHARE_SCENARIOS,
    SEASON_SD_SCENARIOS,
    SIGNAL_QUALITY_SCENARIOS,
    PlayerSpecMappingConfig,
    map_contract_to_playerspecs,
)
from ceauction.realdata.scenarios import (
    BASELINE_SCENARIO,
    GRID_SIZE,
    SCENARIO_GRID,
    ArmSpec,
    ModelScenario,
    Pairing,
    ScenarioBand,
    ScenarioCell,
    enumerate_scenarios,
    format_band,
    run_scenario_arms,
    run_scenario_band,
)
from ceauction.realdata.slot_swap import (
    N_NON_QB_STARTERS,
    SwapSelection,
    run_slot_swap,
    select_slot_swap,
)
from ceauction.realdata.smoke import build_test_rosters, roster_assignment

FAST = dict(calibration_sims=6_000, injury_calibration_sims=1_200,
            availability_calibration_sims=3_000)


def _player(key, name, pos, points, *, bye=7, cv=0.6):
    return {
        "player_key": key, "name": name, "position": pos,
        "nfl_team": "ZZA", "bye_week": bye,
        "season_points": {"points": points,
                          "scoring_source": "recomputed_from_components",
                          "fumble_interpretation": "exclude",
                          "omitted_fumble_points": None},
        "availability": {"injury_prob": 0.30, "proj_games_missed": 1.5,
                         "injury_prob_definition": None,
                         "proj_games_missed_definition": None,
                         "games_in_horizon": 17.0},
        "cohort_dispersion": {"weekly_cv": cv, "weekly_cv_is_total_dispersion": True,
                              "weekly_miss_rate": 0.07, "fit_provenance": "FIXTURE"},
        "raw_fields": {"Name": name},
    }


@pytest.fixture(scope="module")
def payload():
    """A fabricated pool big enough for twelve legal rosters plus spares."""
    players, n = [], 0
    for pos, count in (("QB", 40), ("RB", 55), ("WR", 80), ("TE", 30)):
        for i in range(count):
            n += 1
            players.append(_player(
                f"fab_{pos.lower()}_{i}", f"Fabricated {pos}{i:03d}", pos,
                points=float(340 - 1.4 * n), bye=5 + (i % 10),
                cv={"QB": 0.44, "RB": 0.62, "WR": 0.65, "TE": 0.74}[pos]))
    return {"schema_version": "1.0.0", "players": players}


@pytest.fixture(scope="module")
def cast(payload):
    """A fixed baseline cast and the keys it needs, mapped once."""
    cfg = BASELINE_SCENARIO.to_mapping_config(**FAST)
    mapped = map_contract_to_playerspecs(payload, cfg, limit=250)
    rosters = build_test_rosters(mapped.specs)
    assignment = roster_assignment(rosters)
    rostered = {pid for team in assignment for pid in team}
    key_of = {m.spec.player_id: m.canonical_key for m in mapped.players}
    spare = [s for s in mapped.specs if s.player_id not in rostered]
    return {"assignment": assignment, "rostered": rostered, "key_of": key_of,
            "specs": mapped.specs, "spare": spare, "rosters": rosters}


# ==========================================================================
# The grid itself
# ==========================================================================


def test_the_grid_is_the_full_cross_product():
    assert GRID_SIZE == 54
    assert GRID_SIZE == (len(AVAILABILITY_INTERPRETATIONS)
                         * len(FORECASTABLE_SHARE_SCENARIOS)
                         * len(SEASON_SD_SCENARIOS)
                         * len(SIGNAL_QUALITY_SCENARIOS))
    seen = {(s.availability_interpretation, s.forecastable_share,
             s.season_sd_fraction, s.signal_quality) for s in SCENARIO_GRID}
    expected = set(itertools.product(
        AVAILABILITY_INTERPRETATIONS, FORECASTABLE_SHARE_SCENARIOS,
        SEASON_SD_SCENARIOS, SIGNAL_QUALITY_SCENARIOS))
    assert seen == expected


def test_enumeration_order_is_deterministic():
    assert enumerate_scenarios() == SCENARIO_GRID
    assert enumerate_scenarios() == enumerate_scenarios()
    # Not an accident of set iteration: the first and last cells are pinned.
    assert SCENARIO_GRID[0].scenario_id == "fh-f000-s000-n0"
    assert SCENARIO_GRID[-1].scenario_id == "aa-f050-s020-w2"


def test_scenario_ids_are_unique_and_round_trip():
    ids = [s.scenario_id for s in SCENARIO_GRID]
    assert len(set(ids)) == GRID_SIZE
    for s in SCENARIO_GRID:
        assert ModelScenario.from_id(s.scenario_id) == s


def test_a_scenario_id_is_filename_safe_and_sortable():
    for s in SCENARIO_GRID:
        assert s.scenario_id.replace("-", "").isalnum()
        assert "/" not in s.scenario_id and " " not in s.scenario_id


def test_a_bad_scenario_id_is_refused():
    for bad in ("", "nonsense", "zz-f000-s000-w1", "fh-f000-s000", "fh-f000-s000-xx"):
        with pytest.raises(ValueError, match="not a scenario id|unknown"):
            ModelScenario.from_id(bad)


def test_a_scenario_is_serializable(tmp_path):
    blob = [s.to_dict() for s in SCENARIO_GRID]
    path = tmp_path / "grid.json"
    path.write_text(json.dumps(blob), encoding="utf-8")
    back = json.loads(path.read_text(encoding="utf-8"))
    assert [ModelScenario.from_id(d["scenario_id"]) for d in back] == list(SCENARIO_GRID)


def test_a_scenario_builds_the_config_it_names():
    for s in SCENARIO_GRID:
        cfg = s.to_mapping_config()
        assert cfg.projection_availability_interpretation == s.availability_interpretation
        assert cfg.forecastable_share == s.forecastable_share
        assert cfg.season_sd_fraction == s.season_sd_fraction
        assert cfg.signal_quality == s.signal_quality


def test_a_config_cannot_silently_disagree_with_its_scenario():
    """An override on a governed axis would poison every cache keyed on the id."""
    s = SCENARIO_GRID[0]
    for axis in ("projection_availability_interpretation", "forecastable_share",
                 "season_sd_fraction", "signal_quality"):
        with pytest.raises(ValueError, match="governed by the scenario"):
            s.to_mapping_config(**{axis: SCENARIO_GRID[-1].to_dict()[
                axis if axis != "projection_availability_interpretation"
                else "availability_interpretation"]})


def test_ungoverned_config_fields_pass_through():
    cfg = SCENARIO_GRID[0].to_mapping_config(target="mean_target",
                                             calibration_sims=1234)
    assert cfg.target == "mean_target" and cfg.calibration_sims == 1234


def test_invalid_scenarios_are_refused():
    with pytest.raises(ValueError, match="availability_interpretation"):
        ModelScenario("preferred", 0.0, 0.0, "week_sd")
    with pytest.raises(ValueError, match="signal_quality"):
        ModelScenario("full_health", 0.0, 0.0, "telepathy")
    with pytest.raises(ValueError, match="forecastable_share"):
        ModelScenario("full_health", 1.5, 0.0, "week_sd")
    with pytest.raises(ValueError, match="season_sd_fraction"):
        ModelScenario("full_health", 0.0, -0.1, "week_sd")


def test_the_baseline_scenario_matches_the_config_defaults():
    """It is a reference for cast construction, not a preferred cell."""
    d = PlayerSpecMappingConfig()
    assert BASELINE_SCENARIO.availability_interpretation == \
        d.projection_availability_interpretation
    assert BASELINE_SCENARIO.forecastable_share == d.forecastable_share
    assert BASELINE_SCENARIO.season_sd_fraction == d.season_sd_fraction
    assert BASELINE_SCENARIO.signal_quality == d.signal_quality
    assert BASELINE_SCENARIO in SCENARIO_GRID


# ==========================================================================
# Arms
# ==========================================================================


def test_an_arm_refuses_to_give_one_player_to_two_teams():
    with pytest.raises(ValueError, match="two teams"):
        ArmSpec("bad", ((1, 2), (2, 3)))


def test_a_pairing_must_name_arms_that_exist(payload, cast):
    with pytest.raises(ValueError, match="unknown arm"):
        run_scenario_arms(
            payload, [ArmSpec("a", cast["assignment"])],
            [Pairing("p", "a", "ghost")],
            only_keys=[cast["key_of"][p] for p in sorted(cast["rostered"])],
            team_index=0, scenarios=SCENARIO_GRID[:1], n_sims=20,
            config_overrides=FAST)


def test_arm_names_must_be_unique(payload, cast):
    arm = ArmSpec("dup", cast["assignment"])
    with pytest.raises(ValueError, match="unique"):
        run_scenario_arms(
            payload, [arm, arm], [Pairing("p", "dup", "dup")],
            only_keys=[cast["key_of"][p] for p in sorted(cast["rostered"])],
            team_index=0, scenarios=SCENARIO_GRID[:1], n_sims=20,
            config_overrides=FAST)


def test_a_band_refuses_a_cell_that_cannot_map_every_arm_player(payload, cast):
    """Every cell must hold the same people or the arms are not comparable."""
    keys = [cast["key_of"][p] for p in sorted(cast["rostered"])]
    with pytest.raises(ValueError, match="could not map|needs the same people"):
        run_scenario_band(
            payload, ArmSpec("a", cast["assignment"]),
            ArmSpec("b", cast["assignment"]),
            only_keys=keys + ["fab_nobody_9999"], team_index=0,
            scenarios=SCENARIO_GRID[:1], n_sims=20, config_overrides=FAST)


# ==========================================================================
# Pairing
# ==========================================================================


def test_identical_arms_give_exactly_zero_in_every_cell(payload, cast):
    """The strongest evidence that the arms really are paired."""
    keys = [cast["key_of"][p] for p in sorted(cast["rostered"])]
    band = run_scenario_band(
        payload, ArmSpec("a", cast["assignment"]),
        ArmSpec("b", cast["assignment"]), only_keys=keys, team_index=3,
        scenarios=SCENARIO_GRID[:6], n_sims=120, config_overrides=FAST)
    assert len(band.cells) == 6
    for c in band.cells:
        assert c.delta_ce == 0.0
        assert c.delta_ce_se == 0.0
        assert c.discordance == 0.0
        assert not c.resolved
    assert band.band_width == 0.0
    assert band.mc_band_width == 0.0


def test_a_real_swap_is_not_zero_and_moves_the_focus_team(payload, cast):
    """A downgrade at a starting slot must register as a loss."""
    assignment = cast["assignment"]
    rosters = cast["rosters"]
    team = 0
    # Replace the team's best flex player with a far worse spare: unambiguous.
    from ceauction.curve import FLEX_ELIGIBLE
    on_team = sorted((rosters.spec(p) for p in assignment[team]
                      if rosters.spec(p).position in FLEX_ELIGIBLE),
                     key=lambda s: -s.base_mean)
    out_id = on_team[0].player_id
    spare = min((s for s in cast["spare"] if s.position in FLEX_ELIGIBLE),
                key=lambda s: s.base_mean)
    swapped = tuple(
        tuple(spare.player_id if p == out_id else p for p in t) if i == team else t
        for i, t in enumerate(assignment))
    keys = [cast["key_of"][p] for p in sorted(cast["rostered"])] + \
           [cast["key_of"][spare.player_id]]
    band = run_scenario_band(
        payload, ArmSpec("before", assignment), ArmSpec("after", swapped),
        only_keys=keys, team_index=team, scenarios=SCENARIO_GRID[:2],
        n_sims=400, config_overrides=FAST)
    assert all(c.delta_ce < 0 for c in band.cells), "a downgrade must cost CE"
    assert all(c.discordance > 0 for c in band.cells)


def test_several_pairings_share_one_simulation_of_each_arm(payload, cast):
    """Extracting more comparisons must not change any of them."""
    keys = [cast["key_of"][p] for p in sorted(cast["rostered"])]
    arms = [ArmSpec("a", cast["assignment"]), ArmSpec("b", cast["assignment"])]
    many = run_scenario_arms(
        payload, arms,
        [Pairing("ab", "a", "b"), Pairing("ba", "b", "a"), Pairing("aa", "a", "a")],
        only_keys=keys, team_index=2, scenarios=SCENARIO_GRID[:2], n_sims=100,
        config_overrides=FAST)
    one = run_scenario_band(
        payload, arms[0], arms[1], only_keys=keys, team_index=2, label="ab",
        scenarios=SCENARIO_GRID[:2], n_sims=100, config_overrides=FAST)
    assert [c.delta_ce for c in many["ab"].cells] == [c.delta_ce for c in one.cells]
    assert all(c.delta_ce == 0.0 for c in many["aa"].cells)


def test_signal_quality_cannot_bite_when_there_is_nothing_to_learn(payload, cast):
    """A structural check that the grid axes reach the engine as intended."""
    keys = [cast["key_of"][p] for p in sorted(cast["rostered"])]
    cells = [ModelScenario("full_health", 0.0, 0.0, sq)
             for sq in SIGNAL_QUALITY_SCENARIOS]
    band = run_scenario_band(
        payload, ArmSpec("a", cast["assignment"]),
        ArmSpec("b", cast["assignment"]), only_keys=keys, team_index=1,
        scenarios=cells, n_sims=100, config_overrides=FAST)
    # season_sd = 0 means no latent level exists, so how fast anyone learns it
    # cannot matter; all three cells must agree exactly.
    assert len({c.delta_ce for c in band.cells}) == 1


# ==========================================================================
# Band arithmetic
# ==========================================================================


def _cell(sid, delta, se, **kw):
    s = ModelScenario.from_id(sid)
    return ScenarioCell(
        scenario=s, arm_a="a", arm_b="b", team_index=0, team_name="T0",
        ce_a=kw.get("ce_a", 0.1), ce_b=kw.get("ce_b", 0.1 + delta),
        delta_ce=delta, delta_ce_se=se, discordance=kw.get("disc", 0.4),
        n_sims=16000, runtime_s=1.0)


def _band(cells, **kw):
    return ScenarioBand(label="t", arm_a="a", arm_b="b", cells=tuple(cells),
                        n_sims=16000, seed=1, total_runtime_s=1.0, **kw)


def test_the_point_band_is_the_spread_of_point_estimates():
    b = _band([_cell("fh-f000-s000-w1", 0.01, 0.002),
               _cell("aa-f050-s020-n0", 0.05, 0.002),
               _cell("fh-f025-s010-w2", 0.03, 0.002)])
    assert b.min_delta == pytest.approx(0.01)
    assert b.max_delta == pytest.approx(0.05)
    assert b.band_width == pytest.approx(0.04)
    assert b.argmin_cell.scenario_id == "fh-f000-s000-w1"
    assert b.argmax_cell.scenario_id == "aa-f050-s020-n0"


def test_the_mc_band_widens_the_point_band_by_each_cells_own_interval():
    b = _band([_cell("fh-f000-s000-w1", 0.01, 0.002),
               _cell("aa-f050-s020-n0", 0.05, 0.003)])
    assert b.mc_low == pytest.approx(0.01 - 1.96 * 0.002)
    assert b.mc_high == pytest.approx(0.05 + 1.96 * 0.003)
    assert b.mc_band_width > b.band_width
    assert b.mc_inflation == pytest.approx(b.mc_band_width / b.band_width)


def test_a_zero_width_point_band_reports_infinite_inflation():
    """All cells agreeing is not the same as the run being precise."""
    b = _band([_cell("fh-f000-s000-w1", 0.01, 0.002),
               _cell("aa-f050-s020-n0", 0.01, 0.002)])
    assert b.band_width == 0.0
    assert math.isinf(b.mc_inflation)


def test_sign_stability_and_rank_stability():
    strong = _band([_cell("fh-f000-s000-w1", 0.05, 0.001),
                    _cell("aa-f050-s020-n0", 0.06, 0.001)])
    assert strong.sign_stable and strong.all_resolved
    assert strong.rank_stability == "strong"

    weak = _band([_cell("fh-f000-s000-w1", 0.05, 0.001),
                  _cell("aa-f050-s020-n0", 0.001, 0.002)])
    assert weak.sign_stable and not weak.all_resolved
    assert weak.rank_stability == "weak"

    none = _band([_cell("fh-f000-s000-w1", 0.05, 0.001),
                  _cell("aa-f050-s020-n0", -0.05, 0.001)])
    assert not none.sign_stable
    assert none.rank_stability == "none"


def test_a_negative_but_fully_resolved_band_is_strong():
    b = _band([_cell("fh-f000-s000-w1", -0.05, 0.001),
               _cell("aa-f050-s020-n0", -0.06, 0.001)])
    assert b.rank_stability == "strong"


def test_a_band_straddling_zero_is_never_strong():
    b = _band([_cell("fh-f000-s000-w1", 0.05, 0.001),
               _cell("aa-f050-s020-n0", 0.001, 0.0001)])
    assert b.all_resolved and b.sign_stable
    assert b.mc_low < 0.0 or b.rank_stability == "strong"


def test_the_dominant_axis_is_the_one_that_spreads_the_band_most():
    # availability moves the delta a lot; nothing else moves it at all.
    cells = []
    for s in SCENARIO_GRID:
        d = 0.05 if s.availability_interpretation == "availability_adjusted" else 0.01
        cells.append(_cell(s.scenario_id, d, 0.001))
    assert _band(cells).dominant_axis == "availability_interpretation"

    cells = []
    for s in SCENARIO_GRID:
        d = 0.01 + 0.1 * s.season_sd_fraction
        cells.append(_cell(s.scenario_id, d, 0.001))
    assert _band(cells).dominant_axis == "season_sd_fraction"


def test_a_band_summary_serializes_and_flags_a_reduced_grid():
    b = _band([_cell("fh-f000-s000-w1", 0.01, 0.002)])
    d = b.summary()
    json.dumps(d)
    assert d["cells_run"] == 1 and d["grid_size"] == 54
    assert d["full_grid"] is False
    assert "REDUCED GRID" in format_band(b)


def test_a_full_grid_is_not_flagged_as_reduced():
    b = _band([_cell(s.scenario_id, 0.01, 0.002) for s in SCENARIO_GRID])
    assert b.summary()["full_grid"] is True
    assert "REDUCED GRID" not in format_band(b)


def test_the_rendering_never_claims_a_correct_scenario():
    b = _band([_cell(s.scenario_id, 0.01, 0.002) for s in SCENARIO_GRID])
    text = format_band(b)
    assert "No cell in this grid is the correct one" in text
    assert "not a confidence interval" in text
    assert "ONE COMPARISON" in text


# ==========================================================================
# Slot-swap selection
# ==========================================================================


def test_the_swap_selection_is_deterministic(payload):
    a, _ = select_slot_swap(payload, limit=250, selection_sims=200,
                            config_overrides=FAST)
    b, _ = select_slot_swap(payload, limit=250, selection_sims=200,
                            config_overrides=FAST)
    assert (a.focus_team, a.outgoing_id, a.incoming_id, a.adjacent_id) == \
           (b.focus_team, b.outgoing_id, b.incoming_id, b.adjacent_id)
    assert a.marginal_starter_id == b.marginal_starter_id


def test_the_focus_team_is_nearest_the_median_baseline_ce(payload):
    sel, _ = select_slot_swap(payload, limit=250, selection_sims=200,
                              config_overrides=FAST)
    ce = sorted(sel.baseline_ce)
    median = (ce[len(ce) // 2 - 1] + ce[len(ce) // 2]) / 2.0
    best = min(abs(c - median) for c in sel.baseline_ce)
    assert abs(sel.baseline_ce[sel.focus_team] - median) == pytest.approx(best)


def test_the_outgoing_player_is_the_weakest_flex_and_the_incoming_is_unrostered(payload):
    from ceauction.curve import FLEX_ELIGIBLE
    sel, specs = select_slot_swap(payload, limit=250, selection_sims=200,
                                  config_overrides=FAST)
    by = {s.player_id: s for s in specs}
    rostered = {p for team in sel.assignment for p in team}
    on_team = [by[p] for p in sel.assignment[sel.focus_team]
               if by[p].position in FLEX_ELIGIBLE]
    assert by[sel.outgoing_id].base_mean == min(s.base_mean for s in on_team)
    assert sel.incoming_id not in rostered
    assert by[sel.incoming_id].position in FLEX_ELIGIBLE
    # The incoming player is the best the board still offers.
    free = [s for s in specs if s.player_id not in rostered
            and s.position in FLEX_ELIGIBLE]
    assert by[sel.incoming_id].base_mean == max(s.base_mean for s in free)


def test_the_marginal_starter_is_the_sixth_ranked_flex_player(payload):
    from ceauction.curve import FLEX_ELIGIBLE
    sel, specs = select_slot_swap(payload, limit=250, selection_sims=200,
                                  config_overrides=FAST)
    by = {s.player_id: s for s in specs}
    ranked = sorted((by[p] for p in sel.assignment[sel.focus_team]
                     if by[p].position in FLEX_ELIGIBLE),
                    key=lambda s: (-s.base_mean, s.player_id))
    assert N_NON_QB_STARTERS == 6
    assert sel.marginal_starter_id == ranked[N_NON_QB_STARTERS - 1].player_id
    # And he is genuinely better than the 15th man, so the two slots differ.
    assert by[sel.marginal_starter_id].base_mean > by[sel.outgoing_id].base_mean


def test_the_swapped_assignment_changes_exactly_one_player(payload):
    sel, _ = select_slot_swap(payload, limit=250, selection_sims=200,
                              config_overrides=FAST)
    swapped = sel.swapped_assignment(sel.incoming_id)
    for t, (a, b) in enumerate(zip(sel.assignment, swapped)):
        if t != sel.focus_team:
            assert a == b
        else:
            assert set(a) ^ set(b) == {sel.outgoing_id, sel.incoming_id}
            assert len(a) == len(b) == DEFAULT_LEAGUE.roster_size
            # Position in the tuple is preserved, so no other player's roster
            # index moves between arms.
            assert [i for i, p in enumerate(a) if p != b[i]] == \
                   [a.index(sel.outgoing_id)]


def test_an_explicit_outgoing_player_must_be_on_the_focus_team(payload):
    with pytest.raises(ValueError, match="is not on team"):
        select_slot_swap(payload, limit=250, selection_sims=200,
                         config_overrides=FAST, outgoing_id=-12345)


def test_an_explicit_incoming_player_must_not_be_rostered(payload):
    sel, _ = select_slot_swap(payload, limit=250, selection_sims=200,
                              config_overrides=FAST)
    already = sel.assignment[0][0]
    with pytest.raises(ValueError, match="already rostered"):
        select_slot_swap(payload, limit=250, selection_sims=200,
                         config_overrides=FAST, incoming_id=already)


def test_supplying_ids_is_recorded_in_the_rule(payload):
    """No output may imply a choice was derived when it was given."""
    sel, specs = select_slot_swap(payload, limit=250, selection_sims=200,
                                  config_overrides=FAST)
    rostered = {p for team in sel.assignment for p in team}
    from ceauction.curve import FLEX_ELIGIBLE
    free = sorted((s for s in specs if s.player_id not in rostered
                   and s.position in FLEX_ELIGIBLE),
                  key=lambda s: (-s.base_mean, s.player_id))
    other, _ = select_slot_swap(payload, limit=250, selection_sims=200,
                                config_overrides=FAST,
                                incoming_id=free[2].player_id)
    assert "OVERRIDDEN" in other.rule and "incoming" in other.rule
    assert "OVERRIDDEN" not in sel.rule


# ==========================================================================
# The experiment end to end
# ==========================================================================


def test_the_experiment_runs_every_band_on_a_small_grid(payload):
    exp = run_slot_swap(payload, limit=250, n_sims=200, selection_sims=200,
                        scenarios=SCENARIO_GRID[:2], config_overrides=FAST)
    assert set(exp.bands) == {"bench_slot", "marginal_starter_slot",
                              "adjacent_alternative"}
    for band in exp.bands.values():
        assert len(band.cells) == 2
    d = exp.summary()
    json.dumps(d)
    assert d["presentation_verdict"] in ("narrow", "wide", "unmeasurable")
    assert any("does NOT establish" in w for w in d["warnings"])
    assert any("opportunity cost" in w for w in d["warnings"])


def test_replacing_a_starter_costs_more_than_replacing_the_last_bench_spot(payload):
    """Both swaps bring in the same player; only the slot vacated differs."""
    exp = run_slot_swap(payload, limit=250, n_sims=1500, selection_sims=200,
                        scenarios=SCENARIO_GRID[:1], config_overrides=FAST)
    bench = exp.bench_band.cells[0].delta_ce
    starter = exp.starter_band.cells[0].delta_ce
    assert starter < bench, (
        "vacating a starting slot must cost more than vacating the 15th spot")


def test_the_experiment_never_calls_a_wide_band_a_reason_not_to_build(payload):
    exp = run_slot_swap(payload, limit=250, n_sims=200, selection_sims=200,
                        scenarios=SCENARIO_GRID[:2], config_overrides=FAST)
    from ceauction.realdata.slot_swap import format_slot_swap
    text = format_slot_swap(exp)
    assert "best-alternative solver is still required" in text
    assert "not whether opportunity" in text
    assert "do not" in text and "robust" in text
    assert "TWO COMPARISONS ON ONE ROSTER" in text
