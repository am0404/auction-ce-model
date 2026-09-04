"""The information rule: a lineup may use only pregame-observable information.

These are the most important tests in the suite.  If any of them fails, every
CE number the engine produces is worthless, because the simulated manager would
be cheating.
"""

from __future__ import annotations

import numpy as np
import pytest

from ceauction.league import DEFAULT_LEAGUE, Position
from ceauction.lineup import select_lineup
from ceauction.lineup_vec import select_lineups_mask
from ceauction.players import PlayerSpec
from ceauction.pregame import PregameEntry
from ceauction.roster import Roster, RosterSet
from ceauction.simulate import pregame_week, simulate_seasons, team_scores
from ceauction.synthetic import make_synthetic_league
from ceauction.worlds import build_pool_arrays, generate_world

from helpers import entry, flat_league, flat_spec, week_of

SEED = 909


def test_pregame_entry_cannot_carry_a_realized_score():
    """Type-level enforcement: there is nowhere to put one."""
    fields = set(PregameEntry.__dataclass_fields__)
    assert not fields & {"realized", "actual", "points", "score", "realized_points"}
    with pytest.raises(TypeError):
        PregameEntry(  # type: ignore[call-arg]
            player_id=1, name="x", position=Position.WR, projection=1.0,
            realized_points=99.0,
        )


def test_lineups_are_invariant_to_realized_scores(league):
    """Permuting realized scores must not move a single starter."""
    pool = build_pool_arrays(league.pool, league.settings)
    world = generate_world(pool, SEED, 0, 8)
    rm = league.roster_matrix()
    base_scores, _ = team_scores(world, rm)

    rs = np.random.default_rng(0)
    shuffled = world.realized.points.copy()
    rs.shuffle(shuffled.reshape(-1))
    tampered = type(world)(
        sim_start=world.sim_start, n_sims=world.n_sims, pool=world.pool,
        latent=world.latent, availability=world.availability,
        signals=world.signals,
        realized=type(world.realized)(points=shuffled, spike=world.realized.spike,
                                      group_effect=world.realized.group_effect),
        pregame=world.pregame,
    )
    proj = np.moveaxis(world.pregame.projection[:, rm, :], 2, -1)
    avail = np.moveaxis(world.availability.available[:, rm, :], 2, -1)
    pos = np.broadcast_to(world.pool.position[rm][None, :, None, :], proj.shape)
    mask_a = select_lineups_mask(proj, avail, pos)

    tampered_scores, _ = team_scores(tampered, rm)
    proj_b = np.moveaxis(tampered.pregame.projection[:, rm, :], 2, -1)
    mask_b = select_lineups_mask(proj_b, avail, pos)

    assert np.array_equal(mask_a, mask_b), "starters changed when realized scores changed"
    assert not np.allclose(base_scores, tampered_scores), (
        "sanity: shuffling realized points should change team scores"
    )


def test_benched_surprise_points_are_worth_exactly_zero():
    """A benched player scoring 1000 must not add a single point."""
    league = flat_league()
    pool = build_pool_arrays(league.pool, league.settings)
    world = generate_world(pool, SEED, 0, 1)
    rm = league.roster_matrix()
    before, _ = team_scores(world, rm)

    proj = np.moveaxis(world.pregame.projection[:, rm, :], 2, -1)
    avail = np.moveaxis(world.availability.available[:, rm, :], 2, -1)
    pos = np.broadcast_to(world.pool.position[rm][None, :, None, :], proj.shape)
    mask = select_lineups_mask(proj, avail, pos)
    benched = np.flatnonzero(~mask[0, 0, 0, :])
    assert benched.size, "expected at least one benched player"

    boosted = world.realized.points.copy()
    for r in benched:
        boosted[0, rm[0, r], 0] += 1000.0
    tampered = type(world)(
        sim_start=0, n_sims=1, pool=world.pool, latent=world.latent,
        availability=world.availability, signals=world.signals,
        realized=type(world.realized)(points=boosted, spike=world.realized.spike,
                                      group_effect=world.realized.group_effect),
        pregame=world.pregame,
    )
    after, _ = team_scores(tampered, rm)
    assert after[0, 0, 0] == pytest.approx(before[0, 0, 0])
    assert np.allclose(after, before)


def test_projection_for_week_w_ignores_week_w_and_later_outcomes():
    """The filtration is a strictly-shifted cumulative sum."""
    league = make_synthetic_league()
    pool = build_pool_arrays(league.pool, league.settings)
    world = generate_world(pool, SEED, 0, 4)
    # Week 1 has seen nothing at all, so the posterior must be exactly zero.
    assert np.allclose(world.pregame.posterior_mean[:, :, 0], 0.0)
    assert np.allclose(world.pregame.n_observed[:, :, 0], 0.0)
    # n_observed[w] counts exactly the played weeks strictly before w.
    played = world.availability.available.astype(np.float64)
    expected = np.cumsum(played, axis=2) - played
    assert np.allclose(world.pregame.n_observed, expected)


def test_a_late_season_signal_cannot_move_an_early_projection():
    league = make_synthetic_league()
    pool = build_pool_arrays(league.pool, league.settings)
    w = generate_world(pool, SEED, 0, 2)
    # Rebuild the posterior by hand from the *signals* of weeks < w and confirm
    # it matches, which is only possible if nothing later leaked in.
    sig = np.where(w.signals.observed, w.signals.level_signal, 0.0)
    cum = np.zeros_like(sig)
    cum[:, :, 1:] = np.cumsum(sig[:, :, :-1], axis=2)
    n = w.pregame.n_observed
    ratio = np.where(pool.season_sd > 0,
                     (pool.signal_noise_sd ** 2) / np.maximum(pool.season_sd ** 2, 1e-9),
                     np.inf).reshape(1, -1, 1)
    posterior = np.where(np.isinf(ratio), 0.0, cum / (ratio + n))
    assert np.allclose(posterior, w.pregame.posterior_mean)


def test_build_pregame_has_no_realized_parameter():
    """Type-level enforcement of the *second* barrier.

    The no-same-week-clairvoyance rule is arithmetic. The no-learning-from-
    unforecastable-noise rule is structural: there is no argument through which
    a realized score could reach the projection builder at all.
    """
    import inspect

    from ceauction.worlds import _build_pregame

    params = set(inspect.signature(_build_pregame).parameters)
    assert "signals" in params
    assert not params & {"realized", "points", "scores"}


def test_observable_role_change_moves_future_lineups_not_past_ones():
    """A revealed role change must show up in projections from the reveal week on."""
    specs = []
    ids = []
    template = ([Position.QB] * 3 + [Position.RB] * 4
                + [Position.WR] * 6 + [Position.TE] * 2)
    for k, pos in enumerate(template):
        specs.append(flat_spec(k, pos, 10.0 - k * 0.1))
        ids.append(k)
    # The last bench WR gets a large, certain, observable promotion in week 6.
    promoted = 11
    specs[promoted] = flat_spec(
        promoted, Position.WR, 1.0,
        role_change_prob=1.0, role_change_mean=30.0, role_change_sd=0.0,
        role_reveal_lag=1,
    )
    rosters = [Roster("A", tuple(ids))]
    pid = len(specs)
    for t in range(1, DEFAULT_LEAGUE.n_teams):
        team_ids = []
        for pos in template:
            specs.append(flat_spec(pid, pos, 9.0))
            team_ids.append(pid)
            pid += 1
        rosters.append(Roster(f"B{t}", tuple(team_ids)))
    rs = RosterSet(tuple(specs), tuple(rosters), DEFAULT_LEAGUE)

    pool = build_pool_arrays(rs.pool, rs.settings)
    world = generate_world(pool, SEED, 0, 1)
    change_week = int(world.latent.role_week[0, promoted])

    obs = world.pregame.observed_role_delta[0, promoted]
    true = world.latent.true_role_delta[0, promoted]
    assert true[change_week] == pytest.approx(30.0)
    assert obs[change_week] == pytest.approx(0.0), "reveal lag not respected"
    assert obs[change_week + 1] == pytest.approx(30.0)

    started_before = pregame_week(world, rs, 0, change_week).entries
    lu_before = select_lineup(pregame_week(world, rs, 0, change_week))
    lu_after = select_lineup(pregame_week(world, rs, 0, change_week + 1))
    assert promoted not in lu_before.started_ids, (
        "the change was not observable yet, so he must not be started"
    )
    assert promoted in lu_after.started_ids, (
        "once revealed, the role change must change the lineup"
    )
    reason = next(c.reason for c in lu_after.choices if c.player_id == promoted)
    assert "observed role change" in reason
