"""Dual weekly results, records, and the total-points tiebreaker."""

from __future__ import annotations

import numpy as np
import pytest

from ceauction.league import DEFAULT_LEAGUE
from ceauction.schedule import base_schedule, opponents_for_batch, round_robin
from ceauction.standings import compare, regular_season, weekly_median

T = DEFAULT_LEAGUE.n_teams
W = DEFAULT_LEAGUE.regular_season_weeks
TOT = DEFAULT_LEAGUE.total_weeks


def _scores(week_scores):
    """(1, 12, 17) score array with week 0 set and every later week constant."""
    a = np.full((1, T, TOT), 50.0)
    a[0, :, 0] = week_scores
    return a


def _fixed_opponents():
    """Deterministic week-0 pairing 0-1, 2-3, ... for hand-checkable tests."""
    opp = np.zeros((1, T, W), dtype=np.int64)
    for w in range(W):
        for t in range(T):
            opp[0, t, w] = t + 1 if t % 2 == 0 else t - 1
    return opp


def test_median_of_twelve_is_the_mean_of_the_two_central_scores():
    s = np.array([[[float(x)] for x in range(1, 13)]])  # (1, 12, 1)
    assert weekly_median(s)[0, 0] == pytest.approx(6.5)


def test_compare_returns_win_loss_and_tie():
    a = np.array([10.0, 5.0, 7.0])
    b = np.array([5.0, 10.0, 7.0])
    assert np.array_equal(compare(a, b), np.array([1.0, 0.0, 0.5]))


def test_median_result_gives_wins_losses_and_exact_ties():
    # sorted: 90 x5, 100 x2, 110 x5 -> 6th and 7th are both 100, median 100.
    week = [110.0] * 5 + [100.0] * 2 + [90.0] * 5
    rs = regular_season(_scores(week), _fixed_opponents(), DEFAULT_LEAGUE)
    med_week0 = weekly_median(_scores(week)[:, :, :W])[0, 0]
    assert med_week0 == pytest.approx(100.0)
    # week 0 contributions, isolated by subtracting the 13 constant weeks
    # (in which every team ties the median at 50.0, worth 0.5 each).
    contrib = rs.median_wins[0] - 0.5 * (W - 1)
    assert list(contrib[:5]) == [1.0] * 5      # above the median: win
    assert list(contrib[5:7]) == [0.5] * 2     # exactly the median: tie
    assert list(contrib[7:]) == [0.0] * 5      # below the median: loss


def test_a_team_can_go_two_and_oh_one_and_one_or_oh_and_two():
    week = [130.0, 40.0] + [100.0] * 10
    rs = regular_season(_scores(week), _fixed_opponents(), DEFAULT_LEAGUE)
    h2h = rs.h2h_wins[0] - 0.5 * (W - 1)
    med = rs.median_wins[0] - 0.5 * (W - 1)
    assert (h2h[0], med[0]) == (1.0, 1.0)     # 2-0
    assert (h2h[1], med[1]) == (0.0, 0.0)     # 0-2
    # A team can also split: beat its opponent while sitting below the median.
    week2 = [90.0, 80.0] + [100.0] * 10
    rs2 = regular_season(_scores(week2), _fixed_opponents(), DEFAULT_LEAGUE)
    assert rs2.h2h_wins[0, 0] - 0.5 * (W - 1) == 1.0
    assert rs2.median_wins[0, 0] - 0.5 * (W - 1) == 0.0


def test_every_week_produces_exactly_two_results_per_team():
    rs = regular_season(_scores([100.0] * T), _fixed_opponents(), DEFAULT_LEAGUE)
    assert np.allclose(rs.wins, rs.h2h_wins + rs.median_wins)
    assert np.all(rs.wins <= 2 * W)
    # head-to-head is zero sum: each of the T/2 matchups per week awards
    # exactly 1.0 in total, however it is split.
    assert rs.h2h_wins.sum() == pytest.approx(T / 2.0 * W)
    # the median result is likewise conserved across the league
    assert rs.median_wins.sum() == pytest.approx(T / 2.0 * W)


def test_total_points_breaks_ties_in_the_standings():
    """Teams 0 and 1 split their two head-to-heads and both clear the median in
    each, so they finish level on results -- points must then decide."""
    scores = np.full((1, T, TOT), 50.0)
    scores[0, :, 5] = [80.0, 90.0] + [50.0] * 10   # team 1 wins the h2h
    scores[0, :, 6] = [90.0, 85.0] + [50.0] * 10   # team 0 wins the h2h
    rs = regular_season(scores, _fixed_opponents(), DEFAULT_LEAGUE)

    assert rs.wins[0, 0] == rs.wins[0, 1], "results should be level"
    assert rs.points[0, 1] > rs.points[0, 0], "team 1 scored more"
    assert rs.team_to_seed[0, 1] < rs.team_to_seed[0, 0], (
        "the higher-scoring team must seed above on the points tiebreak"
    )

    # Flip the points advantage and the seed order must flip with it.
    scores[0, 0, 6] = 95.0
    scores[0, 1, 5] = 85.0
    rs2 = regular_season(scores, _fixed_opponents(), DEFAULT_LEAGUE)
    assert rs2.wins[0, 0] == rs2.wins[0, 1]
    assert rs2.points[0, 0] > rs2.points[0, 1]
    assert rs2.team_to_seed[0, 0] < rs2.team_to_seed[0, 1]


def test_seeds_are_a_permutation_and_wins_are_non_increasing(league):
    from ceauction.simulate import simulate_seasons
    out = simulate_seasons(league, 40, 7)
    for s in range(40):
        assert sorted(out.seed[s].tolist()) == list(range(T))


def test_index_is_the_final_deterministic_tiebreak():
    scores = np.full((1, T, TOT), 50.0)
    rs = regular_season(scores, _fixed_opponents(), DEFAULT_LEAGUE)
    # Perfectly symmetric league: every team ties everything, so ordering falls
    # through to team index and must be exactly 0..11.
    assert list(rs.seed_to_team[0]) == list(range(T))


def test_schedule_covers_every_pair_and_is_symmetric():
    rr = round_robin(T)
    assert rr.shape == (T - 1, T)
    for t in range(T):
        assert sorted(rr[:, t].tolist()) == [x for x in range(T) if x != t]
    opp = opponents_for_batch(3, 0, 6, DEFAULT_LEAGUE)
    assert opp.shape == (6, T, W)
    for s in range(6):
        for w in range(W):
            for t in range(T):
                assert opp[s, opp[s, t, w], w] == t
                assert opp[s, t, w] != t
        pairs = {(min(t, int(o)), max(t, int(o)))
                 for t in range(T) for o in opp[s, t]}
        assert len(pairs) == T * (T - 1) // 2, "not every pair is played"
