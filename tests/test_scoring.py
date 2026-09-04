"""Half-PPR scoring rules, the seam for real stat-level data.

The seam must represent the *complete* league scoring system even though the
engine models total fantasy points directly, because the moment real stat lines
arrive a missing rule becomes silently dropped points.
"""

from __future__ import annotations

import dataclasses

import pytest

from ceauction.scoring import HALF_PPR, ScoringRules, StatLine, score_statline


def test_passing_line():
    # 300 pass yards = 12, 2 pass TD = 8, 1 INT = -2
    assert score_statline(StatLine(pass_yards=300, pass_tds=2, interceptions=1)) == pytest.approx(18.0)


def test_half_ppr_reception_value():
    assert score_statline(StatLine(receptions=8)) == pytest.approx(4.0)


def test_rushing_receiving_line():
    # 60 rush yards = 6, 1 rush TD = 6, 5 rec = 2.5, 70 rec yards = 7, 1 fumble = -2
    line = StatLine(rush_yards=60, rush_tds=1, receptions=5, rec_yards=70, fumbles_lost=1)
    assert score_statline(line) == pytest.approx(19.5)


def test_every_rule_matches_the_league_settings():
    r = HALF_PPR
    assert (r.pass_yard, r.pass_td, r.interception) == (0.04, 4.0, -2.0)
    assert (r.rush_yard, r.rec_yard) == (0.1, 0.1)
    assert (r.rush_td, r.rec_td) == (6.0, 6.0)
    assert (r.reception, r.fumble_lost) == (0.5, -2.0)
    assert (r.pass_2pt, r.rush_2pt, r.rec_2pt) == (2.0, 2.0, 2.0)
    assert r.special_teams_td == 6.0


def test_empty_line_scores_zero():
    assert score_statline(StatLine()) == 0.0


# --------------------------------------------------------------------------
# The rules added by the audit pass.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("field,points", [
    ("pass_2pt_conversions", 2.0),
    ("rush_2pt_conversions", 2.0),
    ("rec_2pt_conversions", 2.0),
    ("special_teams_tds", 6.0),
])
def test_each_added_rule_scores_on_its_own(field, points):
    assert score_statline(StatLine(**{field: 1})) == pytest.approx(points)
    assert score_statline(StatLine(**{field: 3})) == pytest.approx(3 * points)


def test_two_point_conversions_are_counted_separately_by_type():
    line = StatLine(pass_2pt_conversions=1, rush_2pt_conversions=1, rec_2pt_conversions=1)
    assert score_statline(line) == pytest.approx(6.0)


def test_a_complete_line_uses_every_rule_exactly_once():
    """One of every event; the total pins all thirteen coefficients together."""
    line = StatLine(
        pass_yards=100, pass_tds=1, interceptions=1,
        rush_yards=50, rush_tds=1,
        rec_yards=40, rec_tds=1, receptions=4,
        fumbles_lost=1,
        pass_2pt_conversions=1, rush_2pt_conversions=1, rec_2pt_conversions=1,
        special_teams_tds=1,
    )
    # 4 + 4 - 2 + 5 + 6 + 4 + 6 + 2 - 2 + 2 + 2 + 2 + 6
    assert score_statline(line) == pytest.approx(39.0)


def test_every_statline_field_has_a_matching_rule():
    """No stat may exist that the scorer silently ignores."""
    stat_fields = [f.name for f in dataclasses.fields(StatLine)]
    assert len(stat_fields) == len(dataclasses.fields(ScoringRules))
    for name in stat_fields:
        one = score_statline(StatLine(**{name: 1}))
        assert one != 0.0, f"{name} scores nothing"


def test_a_weekly_score_can_be_negative():
    """There is no rule flooring an individual player's week at zero.

    Three interceptions and a lost fumble on 90 passing yards is -4.4, and the
    engine must be able to represent that (see test_worlds.py).
    """
    line = StatLine(pass_yards=90, interceptions=3, fumbles_lost=1)
    assert score_statline(line) == pytest.approx(-4.4)
    assert score_statline(line) < 0.0


def test_custom_rules_override_the_defaults():
    ppr = ScoringRules(reception=1.0, special_teams_td=0.0)
    assert score_statline(StatLine(receptions=6), ppr) == pytest.approx(6.0)
    assert score_statline(StatLine(special_teams_tds=1), ppr) == 0.0
