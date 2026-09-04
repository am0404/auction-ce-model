"""Real-player ingestion.

**Every input here is fabricated.** No real player, no vendor value, no real
content hash. The sources this package reads are subscriber-gated exports that
are not redistributable, and this repository is public.

What these tests defend is not arithmetic for its own sake but a set of settled
modelling decisions, each of which has a way of silently un-deciding itself:

* a vendor's fantasy total is full PPR and must never become `points`;
* an ordinal expert grade must never become a standard deviation;
* a category with no source column is unmodelled, not observed zero;
* the two availability readings must both be produced and neither preferred;
* the fumble column's meaning is unresolved, so its points stay out;
* the previous model's 10-team non-superflex league must not enter;
* the generated provisional pool must never be read as real data.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ceauction.league import DEFAULT_LEAGUE, LeagueSettings
from ceauction.realdata import (
    SyntheticSourceRefused,
    build_report,
    load_dispersion_fits,
    load_fantasypros,
    load_injury_profiles,
    load_projections,
    normalize_name,
    season_points_from_components,
    validate_contract,
)
from ceauction.realdata.contract import (
    CENTRAL_TENDENCY,
    CENTRAL_TENDENCY_PROVENANCE,
    GAMES_BASIS,
    TARGET_LEAGUE_CONFIG_ID,
    build_contract,
)
from ceauction.realdata.identity import IdentityIndex, join_report
from ceauction.realdata.pipeline import IngestionPaths, ingest
from ceauction.realdata.scoring import FUMBLE_INTERPRETATIONS

FIXED_TIME = "2020-01-01T00:00:00+00:00"


# --------------------------------------------------------------------------
# Fabricated sources
# --------------------------------------------------------------------------

PROJ_HEADER = ("Rank,Name,Pos,Attempts,Comps,Pass TDs,Pass Yards,Ints,Receptions,"
               "Rec Yards,Rec TDs,Rec FD,Rush Attempts,Rush Yards,Rush TDs,"
               "Rush FD,Fumbles,Projections,7-Day Delta")

#: One passer with a complete line, one receiver with blanks, so the
#: "blank means not projected" path is always exercised.
PROJ_ROWS = (
    "1,Fabricated Alpha,QB,500,340,25,4000,10,,,,,50,300,3,20,5,999.9,0",
    "2,Fabricated Beta,WR,,,,,,80,1000,,90,,,,,,222.2,0",
)


def _write(tmp_path: Path, name: str, text: str) -> Path:
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


@pytest.fixture
def projections_csv(tmp_path):
    return _write(tmp_path, "proj.csv",
                  PROJ_HEADER + "\n" + "\n".join(PROJ_ROWS) + "\n")


@pytest.fixture
def fantasypros_csv(tmp_path):
    return _write(tmp_path, "fp.csv",
                  '"RK",TIERS,"PLAYER NAME",TEAM,"POS","BYE WEEK","UPSIDE ","BUST "\n'
                  '1,1,Fabricated Alpha,ZZA,QB1,7,4 out of 5,2 out of 5\n'
                  '2,1,Fabricated Beta,ZZB,WR1,9,-,-\n')


@pytest.fixture
def injuries_json(tmp_path):
    return _write(tmp_path, "inj.json", json.dumps({
        "source": "FIXTURE injury source",
        "last_update_time": None,
        "players": [{"name": "Fabricated Alpha", "pos": "QB",
                     "injury_prob": 0.20, "proj_games_missed": 2.0,
                     "durability": 4.0, "positional_risk_group": 2,
                     "injury_count": 3,
                     "ds_projection_pts_ppr": 333.3}]}))


@pytest.fixture
def fits_json(tmp_path):
    return _write(tmp_path, "fits.json", json.dumps({
        "fitted_at": FIXED_TIME, "seasons": [2018, 2019],
        "fits": {
            "player.weekly_cv": {"value": {"QB": 0.44, "WR": 0.65},
                                 "sample": 100, "cohort": "fixture cohort"},
            "availability.weekly_miss": {"value": {"QB": 0.05, "WR": 0.07},
                                         "sample": 200, "cohort": "fixture cohort"}}}))


@pytest.fixture
def built(projections_csv, fantasypros_csv, injuries_json, fits_json):
    return ingest(IngestionPaths(projections_csv, fantasypros_csv,
                                 injuries_json, fits_json),
                  generated_at=FIXED_TIME)


def _alpha(payload):
    return next(p for p in payload["players"] if p["name"] == "Fabricated Alpha")


def _beta(payload):
    return next(p for p in payload["players"] if p["name"] == "Fabricated Beta")


# --------------------------------------------------------------------------
# 1. Scoring arithmetic
# --------------------------------------------------------------------------


def test_season_points_arithmetic_is_exact():
    """4000*0.04 + 25*4 + 10*(-2) + 300*0.1 + 3*6 = 160+100-20+30+18."""
    b = season_points_from_components(
        {"pass_yards": 4000, "pass_tds": 25, "interceptions": 10,
         "rush_yards": 300, "rush_tds": 3})
    assert b.points == pytest.approx(288.0)
    assert b.per_category["pass_yard"] == pytest.approx(160.0)
    assert b.per_category["interception"] == pytest.approx(-20.0)


def test_half_ppr_receptions_score_at_half_a_point():
    b = season_points_from_components({"receptions": 80, "rec_yards": 1000})
    assert b.points == pytest.approx(80 * 0.5 + 1000 * 0.1)


def test_a_blank_component_is_not_projected_rather_than_zero():
    """It contributes nothing, but which fields were blank is recorded."""
    b = season_points_from_components(
        {"rec_yards": 1000, "receptions": 80, "rec_tds": "", "rush_yards": "-"})
    assert "rec_tds" in b.null_components and "rush_yards" in b.null_components
    assert "rec_td" not in b.per_category
    assert b.points == pytest.approx(140.0)


def test_scoring_uses_the_target_league_rules_not_a_vendors():
    from ceauction.scoring import HALF_PPR
    assert HALF_PPR.reception == 0.5
    b = season_points_from_components({"receptions": 100}, rules=HALF_PPR)
    assert b.points == pytest.approx(50.0)
    assert b.points != pytest.approx(100.0), "full PPR must not be used"


# --------------------------------------------------------------------------
# 2. Median metadata
# --------------------------------------------------------------------------


def test_central_tendency_is_recorded_as_hybrid_not_median(built):
    """The source derives its categories two different ways.

    Continuous categories (yards, receptions) take the sportsbook over/under
    line, which the source calls an accurate median. Discrete categories
    (touchdowns, interceptions) are devigged into implied probabilities and
    turned into a probability-weighted expectation -- a mean. A total summed
    from both is neither, and labelling it "median" would be a claim the
    documentation does not support.
    """
    for p in built.payload["players"]:
        h = p["stat_line_horizon"]
        assert h["central_tendency"] == "hybrid_market_location" == CENTRAL_TENDENCY
        assert h["central_tendency_provenance"] == CENTRAL_TENDENCY_PROVENANCE
        prov = h["central_tendency_provenance"]
        assert "winwithodds.com" in prov, "official documentation must be cited"
        assert "vendor documentation not located" not in prov
        # Both halves of the derivation must be described.
        assert "median" in prov and "expectation" in prov


def test_the_health_treatment_is_not_labelled_full_health(built):
    """The source says projections do not fully capture current health.

    It also applies injury designations manually, so a known injury may already
    have depressed a projection. Neither 'full_health' nor
    'availability_adjusted' is safe.
    """
    for p in built.payload["players"]:
        h = p["stat_line_horizon"]
        assert h["health_treatment"] == "partially_health_agnostic"
        assert "winwithodds.com" in h["health_treatment_provenance"]
        assert h["health_treatment"] != "full_health"


def test_the_games_basis_is_justified_by_the_nfl_season_not_the_fantasy_weeks(built):
    """17 because the NFL regular season is 17 games, not because 14 + 3 = 17.

    The earlier justification arrived at the right number for the wrong reason
    and would have broken if either the league's shape or the NFL's changed.
    """
    from ceauction.realdata import contract as C

    assert C.GAMES_BASIS == 17.0
    doc = C.__dict__["__doc__"] or ""
    src = __import__("inspect").getsource(C)
    marker = src[src.index("#: Games the source's season total spans"):
                 src.index("GAMES_BASIS = 17.0")]
    # The comment is line-wrapped, so compare on collapsed whitespace.
    flat = " ".join(marker.replace("#:", " ").split())
    assert "NFL regular season is 17 games per team" in flat
    assert "14 regular-season weeks plus a 3-week bracket" in flat, (
        "the corrected comment should record the mistake it replaced")
    assert "16 scheduled games inside the fantasy window" in flat
    for p in built.payload["players"]:
        assert p["active_rate"]["games_basis"] == 17.0


def test_the_fantasy_horizon_holds_sixteen_scheduled_games():
    """17 fantasy weeks minus one bye. The 17th NFL game is in week 18."""
    from ceauction.realdata.contract import (FANTASY_SCHEDULED_GAMES,
                                             GAMES_BASIS)
    from ceauction.league import DEFAULT_LEAGUE

    assert DEFAULT_LEAGUE.total_weeks == 17
    assert FANTASY_SCHEDULED_GAMES == 16.0
    assert FANTASY_SCHEDULED_GAMES == DEFAULT_LEAGUE.total_weeks - 1
    assert GAMES_BASIS - FANTASY_SCHEDULED_GAMES == 1.0, (
        "exactly one of the 17 NFL games falls outside the fantasy horizon")
    # Byes are drawn inside the horizon, so the missing game is week 18's.
    lo, hi = DEFAULT_LEAGUE.bye_week_range
    assert 1 <= lo <= hi <= DEFAULT_LEAGUE.total_weeks


def test_a_central_tendency_claim_without_provenance_is_rejected(built):
    payload = json.loads(json.dumps(built.payload))
    payload["players"][0]["stat_line_horizon"]["central_tendency_provenance"] = None
    res = validate_contract(payload)
    assert not res.ok
    assert any("carries no provenance" in e for e in res.errors)


def test_the_unresolved_horizon_fields_stay_null(built):
    """Q1 and Q3 are open; the payload must say so rather than assume."""
    for p in built.payload["players"]:
        assert p["stat_line_horizon"]["games_assumed"] is None
        assert p["stat_line_horizon"]["conditional_on_playing"] is None


# --------------------------------------------------------------------------
# 3. Both availability interpretations
# --------------------------------------------------------------------------


def test_both_availability_interpretations_are_computed(built):
    a = _alpha(built.payload)
    rate = a["active_rate"]
    points = a["season_points"]["points"]
    assert rate["games_basis"] == GAMES_BASIS == 17.0
    assert rate["interpretation_a_full_health"] == pytest.approx(points / 17.0)
    # Two games missed, so B spreads the same total over fifteen.
    assert rate["interpretation_b_availability_adjusted"] == pytest.approx(
        points / 15.0)
    assert rate["interpretation_b_availability_adjusted"] > \
        rate["interpretation_a_full_health"]


def test_neither_interpretation_is_preferred(built):
    """Choosing one is a modelling decision that has not been made."""
    for p in built.payload["players"]:
        assert p["active_rate"]["preferred"] is None


def test_preferring_an_interpretation_is_rejected(built):
    payload = json.loads(json.dumps(built.payload))
    payload["players"][0]["active_rate"]["preferred"] = "a"
    res = validate_contract(payload)
    assert not res.ok
    assert any("preferred" in e for e in res.errors)


def test_interpretation_b_is_absent_without_a_games_missed_figure(built):
    """Beta has no injury profile, so B cannot be computed and is null."""
    b = _beta(built.payload)
    assert b["availability"]["proj_games_missed"] is None
    assert b["active_rate"]["interpretation_b_availability_adjusted"] is None
    assert b["active_rate"]["interpretation_a_full_health"] is not None


def test_a_full_season_absence_produces_no_active_rate_b():
    """Dividing by zero available games would be an artefact, not a rate."""
    from ceauction.realdata.contract import _active_rates
    assert _active_rates(100.0, 17.0, 17.0)[
        "interpretation_b_availability_adjusted"] is None
    assert _active_rates(100.0, 17.0, 20.0)[
        "interpretation_b_availability_adjusted"] is None


# --------------------------------------------------------------------------
# 4. Injury field separation
# --------------------------------------------------------------------------


def test_injury_probability_and_games_missed_are_kept_separate(built):
    a = _alpha(built.payload)["availability"]
    assert a["injury_prob"] == pytest.approx(0.20)
    assert a["proj_games_missed"] == pytest.approx(2.0)
    # Neither has been combined into the other, and neither carries a
    # definition, because the vendor's is not documented.
    assert a["injury_prob_definition"] is None
    assert a["proj_games_missed_definition"] is None


def test_no_weekly_injury_hazard_is_derived(built):
    """injury_prob is season-level risk and must not be used as a weekly rate."""
    blob = json.dumps(built.payload)
    assert "weekly_injury_hazard" not in json.dumps(built.payload["players"])
    listed = {u["parameter"] for u in built.payload["uncalibrated_parameters"]}
    assert "weekly_injury_hazard" in listed


def test_the_availability_hazard_from_fits_is_labelled_as_availability(built):
    a = _alpha(built.payload)["cohort_dispersion"]
    assert a["weekly_miss_rate"] == pytest.approx(0.05)
    assert a["weekly_cv_is_total_dispersion"] is True


# --------------------------------------------------------------------------
# 5. Fumble exclusion and explicit alternatives
# --------------------------------------------------------------------------


def test_fumble_points_are_excluded_by_default_and_reported(built):
    a = _alpha(built.payload)["season_points"]
    assert a["fumble_interpretation"] == "exclude"
    assert a["omitted_fumble_points"] == pytest.approx(-10.0)  # 5 * -2
    assert a["points"] == pytest.approx(288.0)


def test_the_excluded_fumble_category_is_declared_unsupported(built):
    unsup = {u["category"]: u for u in
             built.payload["scoring_support"]["unsupported_categories"]}
    assert "fumble_lost" in unsup
    assert unsup["fumble_lost"]["treated_as"] == "absent"
    assert "fumble_lost" not in built.payload["scoring_support"][
        "supported_categories"]


@pytest.mark.parametrize("interpretation", ["lost", "total"])
def test_the_alternatives_are_explicit_and_include_the_points(interpretation):
    b = season_points_from_components(
        {"rush_yards": 100, "fumbles": 5}, fumble_interpretation=interpretation)
    assert b.points == pytest.approx(10.0 - 10.0)
    assert b.omitted_fumble_points is None
    assert b.fumble_interpretation == interpretation


def test_choosing_an_alternative_moves_fumbles_into_supported(
        projections_csv, fantasypros_csv):
    out = ingest(IngestionPaths(projections_csv, fantasypros_csv),
                 fumble_interpretation="lost", generated_at=FIXED_TIME)
    assert "fumble_lost" in out.payload["scoring_support"]["supported_categories"]
    cats = {u["category"] for u in
            out.payload["scoring_support"]["unsupported_categories"]}
    assert "fumble_lost" not in cats
    assert _alpha(out.payload)["season_points"]["points"] == pytest.approx(278.0)


def test_an_unknown_fumble_interpretation_is_rejected():
    with pytest.raises(ValueError, match="fumble_interpretation"):
        season_points_from_components({"rush_yards": 10},
                                      fumble_interpretation="guess")
    assert set(FUMBLE_INTERPRETATIONS) == {"exclude", "lost", "total"}


def test_the_omitted_contribution_is_reported_not_hidden(built):
    a = _alpha(built.payload)["season_points"]
    from ceauction.realdata.scoring import season_points_from_components as spc
    full = spc({"pass_yards": 4000, "pass_tds": 25, "interceptions": 10,
                "rush_yards": 300, "rush_tds": 3, "fumbles": 5},
               fumble_interpretation="lost")
    assert a["points"] + a["omitted_fumble_points"] == pytest.approx(full.points)


# --------------------------------------------------------------------------
# 6. Missing scoring categories remain absent
# --------------------------------------------------------------------------


def test_two_point_and_special_teams_categories_are_absent_not_zero(built):
    unsup = {u["category"]: u["treated_as"] for u in
             built.payload["scoring_support"]["unsupported_categories"]}
    for category in ("pass_2pt", "rush_2pt", "rec_2pt", "special_teams_td"):
        assert unsup[category] == "absent"
    # And they appear nowhere in any player's stat line.
    for p in built.payload["players"]:
        assert not set(p["stat_line"]) & {
            "pass_2pt_conversions", "rush_2pt_conversions",
            "rec_2pt_conversions", "special_teams_tds"}


def test_treating_a_missing_category_as_zero_is_rejected(built):
    payload = json.loads(json.dumps(built.payload))
    payload["scoring_support"]["unsupported_categories"][0]["treated_as"] = "zero"
    res = validate_contract(payload)
    assert not res.ok
    assert any("absent" in e for e in res.errors)


def test_a_category_cannot_be_both_supported_and_unsupported(built):
    payload = json.loads(json.dumps(built.payload))
    payload["scoring_support"]["supported_categories"].append("pass_2pt")
    res = validate_contract(payload)
    assert any("both" in e for e in res.errors)


# --------------------------------------------------------------------------
# 7. Identity matching
# --------------------------------------------------------------------------


@pytest.mark.parametrize("raw,expected", [
    ("A.J. Brown Jr.", "aj brown"),
    ("AJ Brown", "aj brown"),
    ("  Kenneth  Walker III ", "kenneth walker"),
    ("Amon-Ra St. Brown", "amon ra st brown"),
    ("D'Andre Swift", "dandre swift"),
    ("Jose Alvarez", "jose alvarez"),
    ("", ""),
])
def test_name_normalisation(raw, expected):
    assert normalize_name(raw) == expected


def test_accents_fold_to_the_same_key():
    assert normalize_name("José Álvarez") == normalize_name("Jose Alvarez")


def test_a_suffix_is_only_stripped_when_it_is_a_whole_trailing_token():
    """'Vi' and 'Li' are real name fragments and must survive."""
    assert normalize_name("Alpha Vismara") == "alpha vismara"
    assert normalize_name("Beta Li") == "beta li"
    assert normalize_name("Gamma Delta V") == "gamma delta"


def test_matching_joins_across_sources(built):
    rep = built.build.reports["fantasypros"]
    assert rep.matched == 2 and rep.left_rows == 2
    assert rep.match_rate == pytest.approx(1.0)
    assert rep.unmatched_left == []
    assert _alpha(built.payload)["nfl_team"] == "ZZA"
    assert _alpha(built.payload)["bye_week"] == 7


def test_a_partial_join_is_reported_rather_than_assumed(built):
    """Only one of two players has an injury profile."""
    rep = built.build.reports["injury"]
    assert rep.matched == 1
    assert rep.match_rate == pytest.approx(0.5)
    assert rep.unmatched_left == ["fabricated beta"]
    assert _beta(built.payload)["availability"]["injury_prob"] is None


# --------------------------------------------------------------------------
# 8. Ambiguous, unmatched, duplicate and conflicting identities
# --------------------------------------------------------------------------


def test_an_ambiguous_match_is_reported_and_not_guessed(tmp_path,
                                                        projections_csv):
    """Two right-hand rows share a name; neither may be picked arbitrarily."""
    fp = _write(tmp_path, "fp_dup.csv",
                '"RK",TIERS,"PLAYER NAME",TEAM,"POS","BYE WEEK","UPSIDE ","BUST "\n'
                '1,1,Fabricated Alpha,ZZA,QB1,7,4 out of 5,2 out of 5\n'
                '2,1,Fabricated Alpha,ZZC,QB2,11,3 out of 5,3 out of 5\n')
    out = ingest(IngestionPaths(projections_csv, fp), generated_at=FIXED_TIME)
    rep = out.build.reports["fantasypros"]
    assert [k for k, _ in rep.ambiguous] == ["fabricated alpha"]
    assert len(rep.ambiguous[0][1]) == 2
    assert rep.matched == 0
    # The ambiguous player still exists, with no team and no bye rather than a
    # coin-flipped one.
    assert _alpha(out.payload)["nfl_team"] is None
    assert _alpha(out.payload)["bye_week"] is None


def test_an_unmatched_right_hand_row_is_reported(tmp_path, projections_csv):
    fp = _write(tmp_path, "fp_extra.csv",
                '"RK",TIERS,"PLAYER NAME",TEAM,"POS","BYE WEEK","UPSIDE ","BUST "\n'
                '1,1,Fabricated Alpha,ZZA,QB1,7,-,-\n'
                '2,1,Fabricated Gamma,ZZD,WR2,10,-,-\n')
    out = ingest(IngestionPaths(projections_csv, fp), generated_at=FIXED_TIME)
    rep = out.build.reports["fantasypros"]
    assert "fabricated gamma" in rep.unmatched_right
    assert "fabricated beta" in rep.unmatched_left
    # A source-only player never becomes a contract row: the projection file is
    # the spine.
    assert all(p["name"] != "Fabricated Gamma" for p in out.payload["players"])


def test_duplicates_in_the_projection_file_are_reported_and_emitted_once(
        tmp_path):
    proj = _write(tmp_path, "dup.csv", PROJ_HEADER + "\n" +
                  PROJ_ROWS[0] + "\n" + PROJ_ROWS[0].replace("1,", "3,", 1) + "\n")
    out = ingest(IngestionPaths(proj), generated_at=FIXED_TIME)
    assert out.player_count == 1, "one player_key must not carry two rows"
    assert any("share the normalised name" in w for w in out.build.warnings)
    keys = [p["player_key"] for p in out.payload["players"]]
    assert len(keys) == len(set(keys))


def test_a_position_conflict_between_same_named_rows_is_reported(tmp_path,
                                                                 projections_csv):
    inj = _write(tmp_path, "inj_conflict.json", json.dumps({
        "source": "FIXTURE", "last_update_time": None,
        "players": [
            {"name": "Fabricated Alpha", "pos": "QB", "injury_prob": 0.2,
             "proj_games_missed": 1.0},
            {"name": "Fabricated Alpha", "pos": "RB", "injury_prob": 0.5,
             "proj_games_missed": 3.0}]}))
    out = ingest(IngestionPaths(projections_csv, injuries=inj),
                 generated_at=FIXED_TIME)
    rep = out.build.reports["injury"]
    assert rep.conflicting, "same name at two positions must be reported"
    key, source, positions = rep.conflicting[0]
    assert key == "fabricated alpha" and set(positions) == {"QB", "RB"}
    assert not rep.clean
    assert _alpha(out.payload)["availability"]["injury_prob"] is None


def test_the_match_report_summary_is_counts_only(built):
    """It is committed, so it must carry no names."""
    for rep in built.build.reports.values():
        summary = rep.summary()
        assert all(isinstance(v, (int, float, str)) for v in summary.values())
        assert "Fabricated" not in json.dumps(summary)


# --------------------------------------------------------------------------
# 9. Prohibited synthetic player data
# --------------------------------------------------------------------------


def test_the_generated_provisional_pool_is_refused(tmp_path):
    """Refused by column signature, because it can be renamed."""
    fake = _write(tmp_path, "innocent_name.csv",
                  "player_id,name,pos,nfl_team,depth_rank,ppg_baseline,adp\n"
                  "0,Synthetic One,QB,0,0,18.5,1\n")
    with pytest.raises(SyntheticSourceRefused, match="provisional"):
        load_projections(fake)
    with pytest.raises(SyntheticSourceRefused):
        ingest(IngestionPaths(fake))


def test_a_real_projection_file_is_not_mistaken_for_the_synthetic_pool(
        projections_csv):
    rows, _ = load_projections(projections_csv)
    assert len(rows) == 2


# --------------------------------------------------------------------------
# 10. Schema failures
# --------------------------------------------------------------------------


@pytest.mark.parametrize("drop", [
    "schema_version", "provenance", "scoring_support",
    "uncalibrated_parameters", "open_questions", "players",
])
def test_omitting_a_required_root_member_fails_validation(built, drop):
    payload = json.loads(json.dumps(built.payload))
    payload.pop(drop)
    res = validate_contract(payload)
    assert not res.ok, f"omitting {drop} must fail"


def test_omitting_league_config_id_fails_validation(built):
    payload = json.loads(json.dumps(built.payload))
    payload["provenance"].pop("league_config_id")
    assert not validate_contract(payload).ok


def test_omitting_central_tendency_fails_validation(built):
    payload = json.loads(json.dumps(built.payload))
    payload["players"][0]["stat_line_horizon"].pop("central_tendency")
    assert not validate_contract(payload).ok


def test_omitting_treated_as_fails_validation(built):
    payload = json.loads(json.dumps(built.payload))
    payload["scoring_support"]["unsupported_categories"][0].pop("treated_as")
    assert not validate_contract(payload).ok


def test_an_empty_uncalibrated_parameter_list_fails_validation(built):
    payload = json.loads(json.dumps(built.payload))
    payload["uncalibrated_parameters"] = []
    res = validate_contract(payload)
    assert not res.ok
    assert any("uncalibrated" in e for e in res.errors)


def test_an_unexpected_property_fails_validation(built):
    payload = json.loads(json.dumps(built.payload))
    payload["dollar_values"] = {"Fabricated Alpha": 42}
    assert not validate_contract(payload).ok


def test_dropping_raw_fields_fails_validation(built):
    payload = json.loads(json.dumps(built.payload))
    payload["players"][0]["raw_fields"] = {}
    assert not validate_contract(payload).ok


def test_the_built_payload_validates(built):
    assert built.validation.ok, built.validation.errors


# --------------------------------------------------------------------------
# 11. The vendor fantasy total may never be used
# --------------------------------------------------------------------------


def test_the_vendor_total_is_preserved_raw_but_never_becomes_points(built):
    a = _alpha(built.payload)
    assert a["raw_fields"]["Projections"] == "999.9"
    assert a["season_points"]["points"] == pytest.approx(288.0)
    assert a["season_points"]["scoring_source"] == "recomputed_from_components"


def test_points_equal_to_the_vendor_total_is_warned_about(built):
    """Beta has 80 receptions, so half and full PPR must differ by 40.

    A warning rather than an error: value equality is not proof of provenance.
    The vendor total is full-PPR only on average -- it was recovered by least
    squares across the file -- so an individual row can coincide with a
    correctly recomputed half-PPR figure. On the real export exactly one player
    in 549 does, and blocking on it would reject a sound ingestion. The binding
    guarantee is the structural `scoring_source` check.
    """
    payload = json.loads(json.dumps(built.payload))
    beta = next(p for p in payload["players"] if p["name"] == "Fabricated Beta")
    beta["season_points"]["points"] = 222.2   # the vendor total in raw_fields
    res = validate_contract(payload)
    assert res.ok, "a coincidence must not block ingestion"
    assert any("vendor total" in w for w in res.warnings)
    # Aggregated, never per player: the report is published.
    joined = " ".join(res.warnings)
    assert "222.2" not in joined and "Fabricated" not in joined


def test_a_zero_reception_player_matching_the_vendor_total_is_not_flagged(built):
    """Half and full PPR agree exactly when there are no receptions.

    Flagging that coincidence would reject a legitimate ingestion: on the real
    export it fired on seven players, every one of them with no receptions. The
    check only means something where the two scoring systems would disagree.
    """
    payload = json.loads(json.dumps(built.payload))
    alpha = next(p for p in payload["players"] if p["name"] == "Fabricated Alpha")
    assert alpha["stat_line"]["receptions"] is None
    alpha["raw_fields"]["Projections"] = str(alpha["season_points"]["points"])
    res = validate_contract(payload)
    assert res.ok, res.errors
    assert not any("vendor total" in w for w in res.warnings)


def test_the_vendor_total_check_scales_with_reception_count(built):
    """One reception separates the systems by 0.5, which is enough to check."""
    payload = json.loads(json.dumps(built.payload))
    alpha = next(p for p in payload["players"] if p["name"] == "Fabricated Alpha")
    alpha["stat_line"]["receptions"] = 1.0
    alpha["raw_fields"]["Projections"] = str(alpha["season_points"]["points"])
    res = validate_contract(payload)
    assert any("receptions" in w for w in res.warnings)
    assert "1.0 receptions" not in " ".join(res.warnings)


def test_a_non_recomputed_scoring_source_is_rejected(built):
    payload = json.loads(json.dumps(built.payload))
    payload["players"][0]["season_points"]["scoring_source"] = "vendor_total"
    assert not validate_contract(payload).ok


def test_the_vendor_projection_column_is_not_a_loaded_stat():
    from ceauction.realdata.sources import (FORBIDDEN_PROJECTION_COLUMNS,
                                            PROJECTION_COLUMNS)
    assert "Projections" not in PROJECTION_COLUMNS
    assert "Projections" in FORBIDDEN_PROJECTION_COLUMNS


def test_the_injury_files_own_fantasy_total_is_not_read(built):
    a = _alpha(built.payload)
    assert a["season_points"]["points"] != pytest.approx(333.3)


# --------------------------------------------------------------------------
# 12. Boom/bust must never become a distribution
# --------------------------------------------------------------------------


def test_expert_labels_are_carried_as_metadata_only(built):
    labels = _alpha(built.payload)["expert_labels"]
    assert labels["upside"] == 4 and labels["bust"] == 2
    assert labels["scale"] == "ordinal_1_to_5"
    assert labels["may_derive_dispersion"] is False


def test_an_unlabelled_player_gets_null_not_an_average(built):
    """No tag is not an average tag, but it is no evidence."""
    labels = _beta(built.payload)["expert_labels"]
    assert labels["upside"] is None and labels["bust"] is None


def test_claiming_the_grades_yield_dispersion_is_rejected(built):
    payload = json.loads(json.dumps(built.payload))
    payload["players"][0]["expert_labels"]["may_derive_dispersion"] = True
    res = validate_contract(payload)
    assert not res.ok
    assert any("expert_labels" in e for e in res.errors)


def test_no_distribution_parameter_is_derived_from_the_grades(built):
    """The only dispersion present must come from the fitted cohort."""
    a = _alpha(built.payload)
    assert a["cohort_dispersion"]["weekly_cv"] == pytest.approx(0.44)
    assert "fit_provenance" in a["cohort_dispersion"]
    banned = {"week_sd", "season_sd", "spike_rate", "spike_scale",
              "weekly_state_sd", "ceiling", "floor", "variance"}
    assert not banned & set(json.dumps(a["expert_labels"]).split('"'))


def test_labels_can_be_omitted_entirely(projections_csv, fantasypros_csv):
    from ceauction.realdata.sources import load_fantasypros, load_projections
    rows, src = load_projections(projections_csv)
    fp, fpsrc = load_fantasypros(fantasypros_csv)
    res = build_contract(rows, src, fp, fpsrc, include_expert_labels=False,
                         generated_at=FIXED_TIME)
    assert all("expert_labels" not in p for p in res.payload["players"])
    assert validate_contract(res.payload).ok


# --------------------------------------------------------------------------
# 13. The target league
# --------------------------------------------------------------------------


def test_the_target_league_is_the_twelve_team_superflex_league(built):
    assert built.payload["provenance"]["league_config_id"] == TARGET_LEAGUE_CONFIG_ID
    assert "12team" in TARGET_LEAGUE_CONFIG_ID
    assert "superflex" in TARGET_LEAGUE_CONFIG_ID
    assert DEFAULT_LEAGUE.n_teams == 12


def test_a_ten_team_league_configuration_is_refused(projections_csv):
    from ceauction.realdata.sources import load_projections
    rows, src = load_projections(projections_csv)
    ten = LeagueSettings(n_teams=10)
    with pytest.raises(ValueError, match="12-team superflex"):
        build_contract(rows, src, settings=ten)


@pytest.mark.parametrize("bad_id", [
    "warroom-10team-halfppr", "sleeper_10-team_non-superflex",
    "legacy 10 team board",
])
def test_a_league_id_that_looks_like_the_old_model_is_rejected(built, bad_id):
    payload = json.loads(json.dumps(built.payload))
    payload["provenance"]["league_config_id"] = bad_id
    res = validate_contract(payload)
    assert not res.ok
    assert any("10-team" in e for e in res.errors)


# --------------------------------------------------------------------------
# 14. Deterministic normalized output
# --------------------------------------------------------------------------


def test_ingestion_is_deterministic(projections_csv, fantasypros_csv,
                                    injuries_json, fits_json):
    paths = IngestionPaths(projections_csv, fantasypros_csv, injuries_json,
                           fits_json)
    a = ingest(paths, generated_at=FIXED_TIME)
    b = ingest(paths, generated_at=FIXED_TIME)
    assert json.dumps(a.payload, sort_keys=True) == \
        json.dumps(b.payload, sort_keys=True)
    assert json.dumps(a.report, sort_keys=True) == \
        json.dumps(b.report, sort_keys=True)


def test_player_order_follows_the_projection_file(built):
    assert [p["name"] for p in built.payload["players"]] == [
        "Fabricated Alpha", "Fabricated Beta"]


def test_source_hashes_are_recorded_and_stable(built, projections_csv):
    import hashlib
    src = built.payload["provenance"]["sources"][0]
    expected = hashlib.sha256(projections_csv.read_bytes()).hexdigest()
    assert src["sha256"] == expected
    assert len(src["sha256"]) == 64


def test_a_changed_source_changes_its_hash(tmp_path, projections_csv):
    from ceauction.realdata.sources import load_projections
    _, a = load_projections(projections_csv)
    edited = _write(tmp_path, "proj2.csv",
                    PROJ_HEADER + "\n" + PROJ_ROWS[0].replace("4000", "4001")
                    + "\n" + PROJ_ROWS[1] + "\n")
    _, b = load_projections(edited)
    assert a.sha256 != b.sha256


# --------------------------------------------------------------------------
# 15. The sanitized report
# --------------------------------------------------------------------------


def test_the_report_contains_no_player_rows(built):
    blob = json.dumps(built.report)
    assert "Fabricated" not in blob
    assert "999.9" not in blob
    assert "raw_fields" not in blob


def test_the_report_carries_counts_coverage_and_ranges(built):
    r = built.report
    assert r["players_normalized"] == 2
    assert r["players_by_position"] == {"QB": 1, "WR": 1}
    assert r["coverage"]["injury_profile"]["n"] == 1
    assert r["joins"]["injury"]["match_rate"] == pytest.approx(0.5)
    pts = r["field_summaries"]["season_points"]
    assert pts["n"] == 2 and "median" in pts and "max" in pts
    assert r["validation"]["ok"] is True


def test_the_report_names_the_unsupported_categories(built):
    cats = {u["category"] for u in built.report["scoring_support"]["unsupported"]}
    assert {"pass_2pt", "rush_2pt", "rec_2pt", "special_teams_td",
            "fumble_lost"} <= cats


def test_the_report_renders(built):
    text = built.format_report()
    assert "REAL-PLAYER INGESTION REPORT" in text
    assert "UNCALIBRATED PARAMETERS" in text
    assert "Fabricated" not in text


def test_a_source_without_a_timestamp_is_warned_about(built):
    assert any("cannot be dated" in w for w in built.validation.warnings)


# --------------------------------------------------------------------------
# 16. No unsupported PlayerSpec field is silently populated
# --------------------------------------------------------------------------


def test_no_unsupported_playerspec_field_appears_in_the_contract(built):
    unsupported = {"season_sd", "signal_noise_sd", "weekly_state_sd",
                   "proj_noise_sd", "spike_rate", "spike_scale",
                   "role_change_prob", "role_change_mean", "role_change_sd",
                   "shock_loadings", "contingency",
                   "weekly_projection_override", "weekly_injury_hazard"}
    for p in built.payload["players"]:
        assert not unsupported & set(p)
    listed = {u["parameter"] for u in built.payload["uncalibrated_parameters"]}
    assert unsupported - listed <= {"role_change_mean", "role_change_sd",
                                    "weekly_projection_override"}


def test_every_uncalibrated_parameter_carries_a_reason(built):
    for u in built.payload["uncalibrated_parameters"]:
        assert u["reason"]
        assert u["parameter"]
