"""Mapping the real-player contract into PlayerSpec.

**Every input is fabricated.** No real player, no vendor value.

What these tests defend is that the mapping never converts an unresolved
question into a confident number. The level is *calibrated* against a stated
target rather than assigned; injury parameters are *solved* against two
supplied targets with their errors reported; the variance split is a labelled
scenario; and every field the sources cannot support stays at a zero that is
identified as a placeholder rather than an estimate.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from ceauction.league import DEFAULT_LEAGUE, LeagueSettings, Position
from ceauction.realdata.contract import (
    CENTRAL_TENDENCY,
    FANTASY_SCHEDULED_GAMES,
    GAMES_BASIS,
)
from ceauction.realdata.coverage import (
    DEPTH_BANDS,
    AliasBook,
    coverage_by_band,
    load_alias_book,
)
from ceauction.realdata.mapping import (
    FORECASTABLE_SHARE_SCENARIOS,
    SEASON_SD_SCENARIOS,
    UNRESOLVED_PLACEHOLDER_FIELDS,
    PlayerSpecMappingConfig,
    calibrate_injury,
    calibrate_level,
    map_contract_to_playerspecs,
)
from ceauction.realdata.smoke import ROSTER_TEMPLATE, build_test_rosters

FAST = dict(calibration_sims=20_000, injury_calibration_sims=3_000)


def _player(key, name, pos, points, *, bye=7, team="ZZA",
            injury_prob=None, games_missed=None, cv=0.6, miss=0.07):
    return {
        "player_key": key, "name": name, "position": pos,
        "nfl_team": team, "bye_week": bye,
        "stat_line": {"rec_yards": 100.0, "receptions": 10.0,
                      "fumbles": None, "fumbles_are_lost_fumbles": None},
        "stat_line_horizon": {"basis": "season_total", "games_assumed": None,
                              "conditional_on_playing": None,
                              "central_tendency": CENTRAL_TENDENCY,
                              "central_tendency_provenance": "FIXTURE"},
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
    """A pool big enough to fill twelve legal rosters."""
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


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


def test_no_assumption_is_buried_in_a_constant():
    cfg = PlayerSpecMappingConfig()
    for field in ("target", "fumble_interpretation", "forecastable_share",
                  "season_sd_fraction", "injury_model",
                  "missing_injury_fallback", "calibration_sims",
                  "calibration_seed", "games_basis"):
        assert hasattr(cfg, field), f"{field} must be configurable"


@pytest.mark.parametrize("kwargs", [
    dict(target="mode_target"), dict(fumble_interpretation="maybe"),
    dict(forecastable_share=1.0), dict(forecastable_share=-0.1),
    dict(season_sd_fraction=-0.1), dict(injury_model="hope"),
    dict(missing_injury_fallback="assume_healthy"), dict(calibration_sims=10),
])
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        PlayerSpecMappingConfig(**kwargs)


def test_the_config_label_distinguishes_scenarios():
    a = PlayerSpecMappingConfig(target="median_target")
    b = PlayerSpecMappingConfig(target="mean_target")
    c = PlayerSpecMappingConfig(forecastable_share=0.25)
    assert len({a.label(), b.label(), c.label()}) == 3


def test_there_is_no_assume_healthy_fallback():
    """Silently giving an unprofiled player perfect health is not an option."""
    with pytest.raises(ValueError):
        PlayerSpecMappingConfig(missing_injury_fallback="healthy")


# --------------------------------------------------------------------------
# Games basis and the fantasy horizon
# --------------------------------------------------------------------------


def test_the_games_basis_is_the_nfl_season_and_the_horizon_is_shorter():
    assert GAMES_BASIS == 17.0
    assert FANTASY_SCHEDULED_GAMES == 16.0
    assert DEFAULT_LEAGUE.total_weeks == 17
    assert PlayerSpecMappingConfig().games_basis == 17.0


# --------------------------------------------------------------------------
# Level calibration
# --------------------------------------------------------------------------


def test_the_level_is_calibrated_not_assigned():
    """The solve must actually hit its target, exactly."""
    cfg = PlayerSpecMappingConfig(**FAST)
    cal = calibrate_level(280.0, 0.62, 0.0, cfg)
    assert cal.abs_error == pytest.approx(0.0, abs=1e-9)
    assert cal.achieved == pytest.approx(280.0)
    assert cal.naive_level == pytest.approx(280.0 / 17.0)


def test_the_level_is_exactly_proportional_to_the_season_total():
    """Homogeneity is what makes the cached solve exact rather than an approximation."""
    cfg = PlayerSpecMappingConfig(**FAST)
    a = calibrate_level(100.0, 0.62, 0.0, cfg)
    b = calibrate_level(300.0, 0.62, 0.0, cfg)
    assert b.level / a.level == pytest.approx(3.0, rel=1e-12)


@pytest.mark.parametrize("target", ["median_target", "mean_target"])
def test_both_targets_are_supported_and_hit_exactly(target):
    cfg = PlayerSpecMappingConfig(target=target, **FAST)
    cal = calibrate_level(280.0, 0.62, 0.10, cfg)
    assert cal.target == target
    assert cal.abs_error == pytest.approx(0.0, abs=1e-9)


def test_median_and_mean_targets_agree_because_the_model_is_symmetric():
    """A finding, not an assumption.

    Every component this source can populate is normal, so the season total is
    symmetric and its median equals its mean. The two targets therefore return
    the same level. They would diverge if a skewed component were populated --
    which is exactly why the calibration is run rather than assumed.
    """
    med = calibrate_level(280.0, 0.62, 0.10,
                          PlayerSpecMappingConfig(target="median_target", **FAST))
    mean = calibrate_level(280.0, 0.62, 0.10,
                           PlayerSpecMappingConfig(target="mean_target", **FAST))
    assert med.level == pytest.approx(mean.level, rel=0.01)
    # ...and both land on the closed-form answer.
    assert med.level == pytest.approx(280.0 / 17.0, rel=0.01)


def test_the_level_is_recalibrated_when_season_sd_changes():
    """Changing season_sd must re-run the solve, not reuse a stale level."""
    cfg_a = PlayerSpecMappingConfig(season_sd_fraction=0.0, **FAST)
    cfg_b = PlayerSpecMappingConfig(season_sd_fraction=0.20, **FAST)
    a = calibrate_level(280.0, 0.62, 0.0, cfg_a)
    b = calibrate_level(280.0, 0.62, 0.20, cfg_b)
    assert a.abs_error == pytest.approx(0.0, abs=1e-9)
    assert b.abs_error == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------
# Injury calibration
# --------------------------------------------------------------------------


def test_injury_parameters_are_solved_against_both_targets():
    cfg = PlayerSpecMappingConfig(**FAST)
    cal = calibrate_injury(0.35, 1.2, cfg, position="RB", bye_index=8)
    assert cal.source == "individual"
    assert cal.feasible
    assert abs(cal.injury_prob_error) <= 0.02
    assert abs(cal.games_missed_error) <= 0.15
    assert cal.weekly_injury_hazard > 0 and cal.injury_mean_weeks > 0


def test_both_injury_targets_are_matched_on_the_span_they_describe():
    """Both vendor figures span the full NFL season, so both are matched there.

    An earlier pass scaled projected games missed by 16/17 onto the fantasy
    window while leaving the injury probability untouched, asking one fit to
    reproduce two targets defined on different spans. The target is now taken
    as supplied.
    """
    cfg = PlayerSpecMappingConfig(**FAST)
    cal = calibrate_injury(0.35, 1.7, cfg, position="RB", bye_index=8)
    assert cal.target_games_missed == pytest.approx(1.7), "no rescaling"
    assert cal.target_injury_prob == pytest.approx(0.35)
    assert cal.full_season_weeks == 18 and cal.full_season_games == 17


def test_a_bye_week_absence_costs_no_scheduled_game():
    """Weeks and games are different things and must not be conflated."""
    from ceauction.realdata.mapping import _simulate_availability
    _, missed_early = _simulate_availability(0.15, 3.0, 0, 17, 4000, 7)
    _, missed_mid = _simulate_availability(0.15, 3.0, 8, 17, 4000, 7)
    # Whichever week the bye falls in, one absent week is not charged as a game.
    _, missed_none = _simulate_availability(0.15, 3.0, -1, 17, 4000, 7)
    assert missed_none > missed_mid
    assert missed_none > missed_early


def test_incompatible_targets_are_reported_not_silently_fitted():
    """A low injury probability with many games missed has no solution."""
    cfg = PlayerSpecMappingConfig(**FAST)
    cal = calibrate_injury(0.05, 4.5, cfg, position="WR", bye_index=8)
    assert not cal.feasible
    assert "not jointly reproduced" in cal.note
    # It still returns the closest available fit and reports both errors.
    assert cal.injury_prob_error is not None
    assert cal.games_missed_error is not None


def test_a_missing_profile_falls_back_to_a_labelled_all_cause_rate():
    cfg = PlayerSpecMappingConfig(**FAST)
    cal = calibrate_injury(None, None, cfg, positional_miss_rate=0.07,
                           position="WR", bye_index=8)
    assert cal.source == "positional_all_cause"
    assert cal.weekly_injury_hazard == pytest.approx(0.07)
    for phrase in ("ALL-CAUSE", "benching", "not an injury-only hazard"):
        assert phrase in cal.note


def test_a_missing_profile_is_never_silently_healthy():
    cfg = PlayerSpecMappingConfig(missing_injury_fallback="none", **FAST)
    cal = calibrate_injury(None, None, cfg, positional_miss_rate=0.07,
                           position="WR", bye_index=8)
    assert cal.source == "unmodelled"
    assert "NOT healthy" in cal.note


def test_the_none_injury_model_says_what_it_is_not():
    cfg = PlayerSpecMappingConfig(injury_model="none", **FAST)
    cal = calibrate_injury(0.35, 1.2, cfg, position="RB", bye_index=8)
    assert cal.source == "unmodelled"
    assert "NOT a claim" in cal.note


# --------------------------------------------------------------------------
# Weekly dispersion
# --------------------------------------------------------------------------


@pytest.mark.parametrize("share", FORECASTABLE_SHARE_SCENARIOS)
def test_the_variance_split_preserves_total_dispersion(payload, share):
    """week_sd^2 + weekly_state_sd^2 must equal the total, exactly."""
    cfg = PlayerSpecMappingConfig(forecastable_share=share, **FAST)
    res = map_contract_to_playerspecs(payload, cfg, limit=8)
    for m in res.players:
        total = math.hypot(m.spec.week_sd, m.spec.weekly_state_sd)
        assert total == pytest.approx(m.total_week_sd, rel=1e-9)
        assert m.spec.weekly_state_sd == pytest.approx(
            m.total_week_sd * math.sqrt(share), rel=1e-9)
        assert m.spec.week_sd == pytest.approx(
            m.total_week_sd * math.sqrt(1 - share), rel=1e-9)


def test_total_dispersion_is_cv_times_the_active_mean(payload):
    cfg = PlayerSpecMappingConfig(**FAST)
    res = map_contract_to_playerspecs(payload, cfg, limit=6)
    for m in res.players:
        cv = {Position.QB: 0.44, Position.RB: 0.62,
              Position.WR: 0.65, Position.TE: 0.74}[m.spec.position]
        assert m.total_week_sd == pytest.approx(cv * m.spec.base_mean, rel=1e-9)


@pytest.mark.parametrize("ssd", SEASON_SD_SCENARIOS)
def test_season_sd_is_a_fraction_of_the_active_mean(payload, ssd):
    cfg = PlayerSpecMappingConfig(season_sd_fraction=ssd, **FAST)
    res = map_contract_to_playerspecs(payload, cfg, limit=6)
    for m in res.players:
        assert m.spec.season_sd == pytest.approx(ssd * m.spec.base_mean, rel=1e-9)


def test_the_scenarios_are_the_ones_specified():
    assert FORECASTABLE_SHARE_SCENARIOS == (0.00, 0.25, 0.50)
    assert SEASON_SD_SCENARIOS == (0.00, 0.10, 0.20)


def test_no_expert_grade_reaches_any_distribution_parameter(payload):
    """UPSIDE/BUST are absent from the fixture and must change nothing."""
    cfg = PlayerSpecMappingConfig(**FAST)
    base = map_contract_to_playerspecs(payload, cfg, limit=5)
    with_labels = json.loads(json.dumps(payload))
    for p in with_labels["players"]:
        p["expert_labels"] = {"upside": 5, "bust": 1, "scale": "ordinal_1_to_5",
                              "may_derive_dispersion": False}
    labelled = map_contract_to_playerspecs(with_labels, cfg, limit=5)
    for a, b in zip(base.players, labelled.players):
        assert a.spec.week_sd == b.spec.week_sd
        assert a.spec.weekly_state_sd == b.spec.weekly_state_sd
        assert a.spec.season_sd == b.spec.season_sd
        assert a.spec.spike_rate == b.spec.spike_rate == 0.0


# --------------------------------------------------------------------------
# Unresolved placeholders
# --------------------------------------------------------------------------


def test_unsupported_fields_are_placeholders_not_estimates(payload):
    cfg = PlayerSpecMappingConfig(**FAST)
    res = map_contract_to_playerspecs(payload, cfg, limit=5)
    for m in res.players:
        assert m.spec.spike_rate == 0.0 and m.spec.spike_scale == 0.0
        assert m.spec.role_change_prob == 0.0
        assert m.spec.shock_loadings == ()
        assert m.spec.contingency is None
        assert m.spec.proj_noise_sd == 0.0
        # signal_noise_sd is deliberately NOT a placeholder any more: it is
        # still uncalibrated, but it is set explicitly from the signal-quality
        # scenario rather than left to default silently to week_sd.
        assert m.spec.signal_noise_sd is not None
        assert "signal_noise_sd" not in m.unresolved_placeholders
        # ...and every one of them is named as unresolved.
        for field in ("spike_rate", "spike_scale", "role_change_prob",
                      "shock_loadings", "contingency", "proj_noise_sd"):
            assert field in m.unresolved_placeholders
    assert "weekly_projection_override" in UNRESOLVED_PLACEHOLDER_FIELDS


def test_every_spec_is_marked_real_and_none_synthetic(payload):
    cfg = PlayerSpecMappingConfig(**FAST)
    res = map_contract_to_playerspecs(payload, cfg, limit=5)
    for m in res.players:
        assert m.spec.data_source.startswith("REAL:")
        assert not m.spec.is_synthetic


def test_the_vendor_total_never_becomes_base_mean(payload):
    """raw_fields carries 999.9; base_mean must be a per-game level."""
    cfg = PlayerSpecMappingConfig(**FAST)
    res = map_contract_to_playerspecs(payload, cfg, limit=5)
    for m in res.players:
        assert m.spec.base_mean != pytest.approx(999.9)
        assert m.spec.base_mean < 60.0


# --------------------------------------------------------------------------
# Coverage bands and aliases
# --------------------------------------------------------------------------


def test_coverage_is_reported_by_depth_band(payload):
    bands, unresolved = coverage_by_band(payload)
    labels = [b.band for b in bands]
    assert labels == ["top_180", "top_240", "top_300", "full_pool"]
    assert DEPTH_BANDS == (180, 240, 300, None)
    for b in bands:
        d = b.to_dict()
        assert set(d) >= {"band", "players", "team_pct", "bye_pct", "injury_pct"}
        # Counts and rates only -- no names.
        assert "Fabricated" not in json.dumps(d)
    assert set(unresolved) == set(labels)


def test_bands_are_ranked_by_recomputed_points(payload):
    bands, _ = coverage_by_band(payload, bands=(3, None))
    assert bands[0].players == 3
    assert bands[1].players == len(payload["players"])


def test_an_alias_redirects_a_join(tmp_path):
    path = tmp_path / "aliases.json"
    path.write_text(json.dumps({
        "notes": "fixture",
        "aliases": {"Fabricated Nickname": "Fabricated Real"},
        "overrides": {"Fabricated Real": {"bye_week": 11}}}))
    book = load_alias_book(path)
    assert book.resolve("fabricated nickname") == "fabricated real"
    assert book.resolve("someone else") == "someone else"
    assert book.override_for("fabricated nickname")["bye_week"] == 11


def test_an_absent_alias_file_is_an_empty_book(tmp_path):
    book = load_alias_book(tmp_path / "nope.json")
    assert book.aliases == {} and book.overrides == {}
    assert book.resolve("anything") == "anything"


def test_the_committed_alias_book_documents_every_entry():
    """An alias is a human judgement and must carry its reason."""
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "data" / "player_aliases.json"
    payload = json.loads(path.read_text())
    assert payload["notes"]
    for name in payload["aliases"]:
        assert name in payload["alias_reasons"], f"{name} has no recorded reason"
        assert len(payload["alias_reasons"][name]) > 40
    for name, why in payload.get("known_unresolvable", {}).items():
        assert len(why) > 40


# --------------------------------------------------------------------------
# Test rosters
# --------------------------------------------------------------------------


def test_test_rosters_are_legal_disjoint_and_deterministic(payload):
    cfg = PlayerSpecMappingConfig(**FAST)
    res = map_contract_to_playerspecs(payload, cfg, limit=205)
    a = build_test_rosters(res.specs)
    b = build_test_rosters(res.specs)
    assert [r.player_ids for r in a.rosters] == [r.player_ids for r in b.rosters]
    assert len(a.rosters) == DEFAULT_LEAGUE.n_teams
    seen = set()
    for roster in a.rosters:
        assert len(roster) == DEFAULT_LEAGUE.roster_size
        assert not seen & set(roster.player_ids), "rosters must be disjoint"
        seen |= set(roster.player_ids)


def test_test_rosters_can_always_field_eight_legal_starters(payload):
    cfg = PlayerSpecMappingConfig(**FAST)
    res = map_contract_to_playerspecs(payload, cfg, limit=205)
    rosters = build_test_rosters(res.specs)
    for t in range(DEFAULT_LEAGUE.n_teams):
        counts = rosters.position_counts(t)
        assert counts[Position.QB] >= 1
        assert counts[Position.RB] >= 2
        assert counts[Position.WR] + counts[Position.TE] >= 3
    assert sum(ROSTER_TEMPLATE.values()) == DEFAULT_LEAGUE.roster_size


def test_a_pool_too_small_is_refused_rather_than_shrunk(payload):
    cfg = PlayerSpecMappingConfig(**FAST)
    res = map_contract_to_playerspecs(payload, cfg, limit=40)
    with pytest.raises(ValueError, match="refusing to build short rosters"):
        build_test_rosters(res.specs)


# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------


def test_the_mapping_is_deterministic(payload):
    cfg = PlayerSpecMappingConfig(**FAST)
    a = map_contract_to_playerspecs(payload, cfg, limit=12)
    b = map_contract_to_playerspecs(payload, cfg, limit=12)
    for x, y in zip(a.players, b.players):
        assert x.spec == y.spec


def test_players_are_mapped_in_descending_points_order(payload):
    cfg = PlayerSpecMappingConfig(**FAST)
    res = map_contract_to_playerspecs(payload, cfg, limit=20)
    means = [m.spec.base_mean for m in res.players]
    assert means == sorted(means, reverse=True)


def test_the_fallback_count_is_warned_about(payload):
    """With a positional rate supplied, unprofiled players fall back to it."""
    cfg = PlayerSpecMappingConfig(**FAST)
    res = map_contract_to_playerspecs(
        payload, cfg, positional_miss={"QB": 0.05, "RB": 0.07, "WR": 0.07,
                                       "TE": 0.11}, limit=40)
    assert any("ALL-CAUSE" in w for w in res.warnings)
    assert any(m.injury.source == "positional_all_cause" for m in res.players)


def test_without_a_positional_rate_the_player_is_unmodelled_not_healthy(payload):
    cfg = PlayerSpecMappingConfig(**FAST)
    res = map_contract_to_playerspecs(payload, cfg, limit=40)
    unmodelled = [m for m in res.players if m.injury.source == "unmodelled"]
    assert unmodelled, "unprofiled players with no fallback rate must be unmodelled"
    assert any("NOT healthy" in m.injury.note for m in unmodelled)
    assert any("UNMODELLED" in w for w in res.warnings)


def test_a_non_positive_season_total_is_skipped_not_clamped():
    """It would otherwise produce a negative standard deviation."""
    bad = {"schema_version": "1.0.0", "players": [
        _player("zero", "Fabricated Zero", "WR", 0.0),
        _player("neg", "Fabricated Neg", "WR", -5.0),
        _player("ok", "Fabricated Ok", "WR", 100.0)]}
    res = map_contract_to_playerspecs(bad, PlayerSpecMappingConfig(**FAST))
    assert len(res.players) == 1
    assert len(res.skipped) == 2
    assert all("not positive" in s for s in res.skipped)
