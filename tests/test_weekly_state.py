"""Forecastable weekly variation: knowable conditions, not projection error.

A per-player, per-week component that is known before lineup lock, appears in
the projection, and moves the realized score's conditional mean by the same
amount. It is what lets several candidates rotate through one lineup spot on
information available before kickoff -- the aggregate-roster question the
15-for-8 format poses.

It must be kept distinct from the two things it superficially resembles:

    proj_noise_sd   moves the projection but *not* the score  (forecast error)
    week_sd         moves the score but *not* the projection  (scoring noise)
    weekly_state    moves both, by the same amount            (knowable conditions)
"""

from __future__ import annotations

import numpy as np
import pytest

from ceauction.experiments import offset_patterns
from ceauction.league import DEFAULT_LEAGUE, Position
from ceauction.lineup import select_lineup
from ceauction.players import PlayerSpec
from ceauction.roster import Roster, RosterSet
from ceauction.simulate import pregame_week
from ceauction.worlds import build_pool_arrays, generate_world

from helpers import flat_spec

SEED = 24601
NW = DEFAULT_LEAGUE.total_weeks


def _world(specs, n_sims=400, seed=SEED):
    pool = build_pool_arrays(specs, DEFAULT_LEAGUE)
    return pool, generate_world(pool, seed, 0, n_sims)


# --------------------------------------------------------------------------
# The defining property: it is in both the projection and the score.
# --------------------------------------------------------------------------


def test_a_weekly_pattern_appears_in_the_projection_exactly_as_supplied():
    pattern = tuple(float(w) - 8.0 for w in range(NW))
    spec = flat_spec(0, Position.WR, 10.0, week_sd=0.0,
                     weekly_state_pattern=pattern)
    _, w = _world([spec], n_sims=5)
    expected = 10.0 + np.array(pattern)
    assert np.allclose(w.pregame.projection[:, 0, :], expected)
    assert np.allclose(w.pregame.weekly_state[:, 0, :], np.array(pattern))


def test_it_moves_the_realized_conditional_mean_by_the_same_amount():
    """Same array on both sides -- not a forecast of the score, but part of it."""
    pattern = tuple(6.0 if w % 2 == 0 else -6.0 for w in range(NW))
    spec = flat_spec(0, Position.WR, 12.0, week_sd=5.0,
                     weekly_state_pattern=pattern)
    _, w = _world([spec], n_sims=6000)
    realized = w.realized.points[:, 0, :].mean(axis=0)
    assert np.allclose(realized, 12.0 + np.array(pattern), atol=0.25)
    # The residual around the pattern is exactly the idiosyncratic noise.
    resid = w.realized.points[:, 0, :] - w.pregame.projection[:, 0, :]
    assert abs(float(resid.std()) - 5.0) < 0.12
    assert abs(float(resid.mean())) < 0.12


def test_the_stochastic_form_is_forecastable_and_the_scoring_noise_is_not():
    """The two dispersions are separated by whether the projection tracks them."""
    knowable = flat_spec(0, Position.WR, 12.0, week_sd=1e-9, weekly_state_sd=5.0)
    unknowable = flat_spec(1, Position.WR, 12.0, week_sd=5.0, weekly_state_sd=0.0)
    _, w = _world([knowable, unknowable], n_sims=3000)

    for i in (0, 1):
        assert abs(float(w.realized.points[:, i, :].std()) - 5.0) < 0.15

    # The knowable arm's projection tracks its score almost exactly...
    r = float(np.corrcoef(w.pregame.projection[:, 0, :].ravel(),
                          w.realized.points[:, 0, :].ravel())[0, 1])
    assert r > 0.99, f"corr(projection, score) = {r:.3f}"
    # ...while the unknowable arm's projection does not move at all, so there
    # is no correlation to compute: every week is projected at the same level.
    assert np.allclose(w.pregame.projection[:, 1, :], 12.0)


def test_it_is_distinct_from_projection_error():
    """`proj_noise_sd` moves the projection *away* from the conditional mean."""
    knowable = flat_spec(0, Position.WR, 12.0, week_sd=1e-9, weekly_state_sd=5.0)
    wrong = flat_spec(1, Position.WR, 12.0, week_sd=1e-9, proj_noise_sd=5.0)
    _, w = _world([knowable, wrong], n_sims=3000)
    assert abs(float(w.pregame.projection[:, 0, :].std()) - 5.0) < 0.15
    assert abs(float(w.pregame.projection[:, 1, :].std()) - 5.0) < 0.15
    # ...but only the knowable arm's *score* moves with it.
    assert abs(float(w.realized.points[:, 0, :].std()) - 5.0) < 0.15
    assert float(w.realized.points[:, 1, :].std()) < 0.1


def test_a_hidden_pattern_reaches_the_score_and_never_the_projection():
    """The control mechanism the aggregate experiment depends on."""
    pattern = tuple(6.0 if w % 2 == 0 else -6.0 for w in range(NW))
    shown = flat_spec(0, Position.WR, 12.0, week_sd=4.0,
                      weekly_state_pattern=pattern)
    hidden = flat_spec(0, Position.WR, 12.0, week_sd=4.0,
                       hidden_weekly_pattern=pattern)
    _, ws = _world([shown], n_sims=200)
    _, wh = _world([hidden], n_sims=200)
    assert np.array_equal(ws.realized.points, wh.realized.points), (
        "the two arms must have byte-identical realized production"
    )
    assert np.allclose(wh.pregame.projection[:, 0, :], 12.0)
    assert not np.allclose(ws.pregame.projection[:, 0, :], 12.0)


def test_pattern_length_is_validated():
    spec = flat_spec(0, Position.WR, 10.0, weekly_state_pattern=(1.0, 2.0))
    with pytest.raises(ValueError, match="weekly_state_pattern.*length 17"):
        build_pool_arrays([spec], DEFAULT_LEAGUE)
    spec = flat_spec(0, Position.WR, 10.0, hidden_weekly_pattern=(1.0,))
    with pytest.raises(ValueError, match="hidden_weekly_pattern.*length 17"):
        build_pool_arrays([spec], DEFAULT_LEAGUE)


def test_a_negative_weekly_state_sd_is_rejected():
    with pytest.raises(ValueError, match="weekly_state_sd"):
        PlayerSpec(player_id=1, name="x", position=Position.WR, nfl_team="T",
                   base_mean=10.0, week_sd=5.0, weekly_state_sd=-1.0)


# --------------------------------------------------------------------------
# Cross-player structure: correlated, independent, offset.
# --------------------------------------------------------------------------


def test_patterns_express_correlated_independent_and_offset_weekly_structure():
    up = tuple(5.0 if w % 2 == 0 else -5.0 for w in range(NW))
    down = tuple(-x for x in up)
    specs = [
        flat_spec(0, Position.WR, 10.0, week_sd=1e-9, weekly_state_pattern=up),
        flat_spec(1, Position.WR, 10.0, week_sd=1e-9, weekly_state_pattern=up),
        flat_spec(2, Position.WR, 10.0, week_sd=1e-9, weekly_state_pattern=down),
        flat_spec(3, Position.WR, 10.0, week_sd=1e-9, weekly_state_sd=5.0),
        flat_spec(4, Position.WR, 10.0, week_sd=1e-9, weekly_state_sd=5.0),
    ]
    _, w = _world(specs, n_sims=600)
    proj = w.pregame.projection

    def corr(i, j):
        return float(np.corrcoef(proj[:, i, :].ravel(), proj[:, j, :].ravel())[0, 1])

    assert corr(0, 1) > 0.99, "identical patterns must move together"
    assert corr(0, 2) < -0.99, "negated patterns must be offset"
    assert abs(corr(3, 4)) < 0.06, "stochastic weekly states must be independent"


def test_offset_patterns_are_mean_zero_and_sum_to_a_constant():
    """Both properties are what keep the aggregate experiment matched.

    17 weeks does not divide by 3, so centring each pattern on the global
    1/3 rather than on its own hot-week frequency would leave the arms with
    different season means.
    """
    for n_players in (2, 3, 4, 5):
        pats = np.array(offset_patterns(NW, n_players, 12.0))
        assert pats.shape == (n_players, NW)
        assert np.allclose(pats.mean(axis=1), 0.0, atol=1e-12)
        assert np.allclose(pats.sum(axis=0), 0.0, atol=1e-12)
        # Exactly one player is hot in each week.
        assert ((pats == pats.max(axis=0)).sum(axis=0) == 1).all()


# --------------------------------------------------------------------------
# The consequence: a lineup spot really does rotate.
# --------------------------------------------------------------------------


def test_offset_candidates_rotate_through_one_lineup_spot():
    """Three interchangeable WRs, offset good weeks, one starting slot's worth.

    Whoever is hot must start, and over a season every one of them must start
    at least once -- which is impossible if pregame levels are static.
    """
    template = ([Position.QB] * 3 + [Position.RB] * 4
                + [Position.WR] * 6 + [Position.TE] * 2)
    pats = offset_patterns(NW, 3, 12.0)
    specs, ids = [], []
    for k, pos in enumerate(template):
        if k in (12, 13, 14):
            # Three rotating WR/TE candidates at the bottom of the roster.
            specs.append(flat_spec(k, Position.WR, 10.0, week_sd=1e-9,
                                   weekly_state_pattern=pats[k - 12]))
        else:
            specs.append(flat_spec(k, pos, 14.0 - k * 0.05, week_sd=1e-9))
        ids.append(k)
    rosters = [Roster("A", tuple(ids))]
    pid = len(specs)
    for t in range(1, DEFAULT_LEAGUE.n_teams):
        team = []
        for pos in template:
            specs.append(flat_spec(pid, pos, 9.0, week_sd=1e-9))
            team.append(pid)
            pid += 1
        rosters.append(Roster(f"B{t}", tuple(team)))
    rs = RosterSet(tuple(specs), tuple(rosters), DEFAULT_LEAGUE)

    pool = build_pool_arrays(rs.pool, rs.settings)
    world = generate_world(pool, SEED, 0, 1)

    starts = {12: 0, 13: 0, 14: 0}
    for week in range(DEFAULT_LEAGUE.regular_season_weeks):
        lu = select_lineup(pregame_week(world, rs, 0, week))
        hot = 12 + (week % 3)
        assert hot in lu.started_ids, (
            f"week {week + 1}: the candidate with the good matchup was benched"
        )
        for pid_ in starts:
            starts[pid_] += pid_ in lu.started_ids
    assert all(v > 0 for v in starts.values()), (
        f"the spot did not rotate: {starts}"
    )


def test_the_reason_names_the_weekly_conditions():
    template = ([Position.QB] * 3 + [Position.RB] * 4
                + [Position.WR] * 6 + [Position.TE] * 2)
    pattern = tuple(9.0 for _ in range(NW))
    specs, ids = [], []
    for k, pos in enumerate(template):
        if k == 14:
            specs.append(flat_spec(k, Position.WR, 6.0, week_sd=1e-9,
                                   weekly_state_pattern=pattern))
        else:
            specs.append(flat_spec(k, pos, 8.0 - k * 0.05, week_sd=1e-9))
        ids.append(k)
    rosters = [Roster("A", tuple(ids))]
    pid = len(specs)
    for t in range(1, DEFAULT_LEAGUE.n_teams):
        team = []
        for pos in template:
            specs.append(flat_spec(pid, pos, 9.0, week_sd=1e-9))
            team.append(pid)
            pid += 1
        rosters.append(Roster(f"B{t}", tuple(team)))
    rs = RosterSet(tuple(specs), tuple(rosters), DEFAULT_LEAGUE)
    world = generate_world(build_pool_arrays(rs.pool, rs.settings), SEED, 0, 1)
    lu = select_lineup(pregame_week(world, rs, 0, 0))
    reason = next(c.reason for c in lu.choices if c.player_id == 14)
    assert "weekly conditions +9.0" in reason


def test_the_synthetic_league_rotates_more_than_a_static_one():
    """The motivation for the whole channel, measured on the baseline league."""
    import dataclasses

    from ceauction.lineup_vec import select_lineups_mask
    from ceauction.synthetic import (
        DEFAULT_PROFILES,
        SyntheticConfig,
        make_synthetic_league,
    )

    def rotation(lg):
        pool = build_pool_arrays(lg.pool, lg.settings)
        w = generate_world(pool, SEED, 0, 120)
        rm = lg.roster_matrix()
        proj = np.moveaxis(w.pregame.projection[:, rm, :], 2, -1)
        avail = np.moveaxis(w.availability.available[:, rm, :], 2, -1)
        pos = np.broadcast_to(w.pool.position[rm][None, :, None, :], proj.shape)
        m = select_lineups_mask(proj, avail, pos)[:, :, :DEFAULT_LEAGUE.regular_season_weeks, :]
        swaps = (m[:, :, 1:, :] != m[:, :, :-1, :]).sum(axis=3) / 2.0
        return float(m.any(axis=2).sum(axis=2).mean()), float(swaps.mean())

    flat = {p: dataclasses.replace(pr, weekly_state_sd=0.0)
            for p, pr in DEFAULT_PROFILES.items()}
    static_players, static_swaps = rotation(
        make_synthetic_league(SyntheticConfig(profiles=flat))
    )
    live_players, live_swaps = rotation(make_synthetic_league())

    assert live_swaps > static_swaps * 1.2, (
        f"forecastable conditions barely changed lineups: "
        f"{static_swaps:.2f} -> {live_swaps:.2f} swaps/week"
    )
    assert live_players > static_players
    assert live_players > 14.0
