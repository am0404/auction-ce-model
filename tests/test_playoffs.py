"""The fixed six-team bracket over weeks 15-17.

The bracket contains no randomness at all: given the score array, every result
is determined.  These tests pin the structure exactly.
"""

from __future__ import annotations

import numpy as np
import pytest

from ceauction.league import DEFAULT_LEAGUE
from ceauction.playoffs import run_bracket
from ceauction.standings import RegularSeason

T = DEFAULT_LEAGUE.n_teams
TOT = DEFAULT_LEAGUE.total_weeks
W15, W16, W17 = (w - 1 for w in DEFAULT_LEAGUE.playoff_weeks)


def _standings(seed_to_team):
    """Build a RegularSeason whose seeding is exactly ``seed_to_team``."""
    stt = np.asarray(seed_to_team, dtype=np.int64).reshape(1, T)
    zeros = np.zeros((1, T))
    return RegularSeason(
        wins=zeros, points=zeros, h2h_wins=zeros, median_wins=zeros,
        seed_to_team=stt, team_to_seed=np.argsort(stt, axis=1),
        median=np.zeros((1, DEFAULT_LEAGUE.regular_season_weeks)),
    )


def _blank_scores():
    return np.zeros((1, T, TOT))


IDENTITY = list(range(T))  # team i is seed i+1


def test_exactly_six_teams_qualify_and_the_top_two_get_byes():
    scores = _blank_scores()
    res = run_bracket(scores, _standings(IDENTITY), DEFAULT_LEAGUE)
    assert res.made_playoffs[0].sum() == 6
    assert list(np.flatnonzero(res.made_playoffs[0])) == [0, 1, 2, 3, 4, 5]
    assert res.has_bye[0].sum() == 2
    assert list(np.flatnonzero(res.has_bye[0])) == [0, 1]
    assert list(res.qualifiers[0]) == [0, 1, 2, 3, 4, 5]


def test_week_fifteen_pairs_three_six_and_four_five():
    """Seeds 3 and 4 win their quarterfinals; check who they then face."""
    scores = _blank_scores()
    scores[0, 2, W15] = 100.0   # seed 3 beats seed 6
    scores[0, 5, W15] = 90.0
    scores[0, 3, W15] = 100.0   # seed 4 beats seed 5
    scores[0, 4, W15] = 90.0
    # Semifinals: the 4/5 winner must meet seed 1, the 3/6 winner must meet seed 2.
    scores[0, 3, W16] = 500.0   # seed 4 upsets seed 1
    scores[0, 0, W16] = 10.0
    scores[0, 1, W16] = 500.0   # seed 2 beats seed 3
    scores[0, 2, W16] = 10.0
    scores[0, 3, W17] = 200.0   # seed 4 wins the title
    scores[0, 1, W17] = 100.0
    res = run_bracket(scores, _standings(IDENTITY), DEFAULT_LEAGUE)
    assert res.champion[0] == 3
    assert list(np.flatnonzero(res.made_final[0])) == [1, 3]


def test_seed_one_never_faces_the_three_six_winner():
    """No reseeding: 1 always plays the 4/5 winner, even if that is seed 5."""
    scores = _blank_scores()
    scores[0, 4, W15] = 100.0   # seed 5 beats seed 4
    scores[0, 3, W15] = 10.0
    scores[0, 5, W15] = 100.0   # seed 6 beats seed 3
    scores[0, 2, W15] = 10.0
    scores[0, 4, W16] = 100.0   # seed 5 beats seed 1
    scores[0, 0, W16] = 10.0
    scores[0, 5, W16] = 100.0   # seed 6 beats seed 2
    scores[0, 1, W16] = 10.0
    scores[0, 4, W17] = 100.0
    scores[0, 5, W17] = 10.0
    res = run_bracket(scores, _standings(IDENTITY), DEFAULT_LEAGUE)
    assert res.champion[0] == 4
    assert list(np.flatnonzero(res.made_final[0])) == [4, 5]


def test_bye_teams_do_not_play_in_week_fifteen():
    """Whatever seeds 1 and 2 'score' in week 15 must be irrelevant."""
    base = _blank_scores()
    base[0, 2, W15], base[0, 5, W15] = 100.0, 10.0
    base[0, 3, W15], base[0, 4, W15] = 100.0, 10.0
    base[0, :, W16] = np.arange(T) * 10.0
    base[0, :, W17] = np.arange(T) * 7.0
    a = run_bracket(base, _standings(IDENTITY), DEFAULT_LEAGUE)
    tampered = base.copy()
    tampered[0, 0, W15] = -9999.0
    tampered[0, 1, W15] = 9999.0
    b = run_bracket(tampered, _standings(IDENTITY), DEFAULT_LEAGUE)
    assert a.champion[0] == b.champion[0]
    assert np.array_equal(a.made_final, b.made_final)


def test_higher_seed_advances_every_tied_matchup():
    """All 17 weeks scoreless: every round ties, so seeds must simply hold."""
    res = run_bracket(_blank_scores(), _standings(IDENTITY), DEFAULT_LEAGUE)
    assert res.champion[0] == 0
    assert list(np.flatnonzero(res.made_final[0])) == [0, 1]


def test_higher_seed_advances_a_tie_in_each_individual_round():
    for week, expected in ((W15, 0), (W16, 0), (W17, 0)):
        scores = _blank_scores()
        # Make every other round decisive for the lower seed, and tie `week`.
        for w in (W15, W16, W17):
            if w == week:
                continue
            scores[0, :, w] = 100.0 - np.arange(T)  # higher seed always wins
        res = run_bracket(scores, _standings(IDENTITY), DEFAULT_LEAGUE)
        assert res.champion[0] == expected


def test_a_tie_is_resolved_by_seed_not_by_team_index():
    """Reverse the seeding so that seed order and team index disagree."""
    reversed_seeds = list(range(T - 1, -1, -1))  # team 11 is the 1 seed
    res = run_bracket(_blank_scores(), _standings(reversed_seeds), DEFAULT_LEAGUE)
    assert res.champion[0] == T - 1
    assert list(np.flatnonzero(res.has_bye[0])) == [T - 2, T - 1]


def test_bracket_is_deterministic_given_scores():
    rs = np.random.default_rng(0)
    scores = rs.uniform(60, 140, (5, T, TOT))
    st = _standings(IDENTITY)
    st = RegularSeason(
        wins=np.zeros((5, T)), points=np.zeros((5, T)), h2h_wins=np.zeros((5, T)),
        median_wins=np.zeros((5, T)),
        seed_to_team=np.tile(np.arange(T), (5, 1)),
        team_to_seed=np.tile(np.arange(T), (5, 1)),
        median=np.zeros((5, DEFAULT_LEAGUE.regular_season_weeks)),
    )
    a = run_bracket(scores, st, DEFAULT_LEAGUE)
    b = run_bracket(scores, st, DEFAULT_LEAGUE)
    assert np.array_equal(a.champion, b.champion)
    assert np.all(a.made_playoffs[np.arange(5), a.champion])
