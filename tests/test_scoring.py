"""Half-PPR scoring rules, the seam for real stat-level data."""

from __future__ import annotations

import pytest

from ceauction.scoring import HALF_PPR, StatLine, score_statline


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


def test_empty_line_scores_zero():
    assert score_statline(StatLine()) == 0.0
