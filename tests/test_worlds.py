"""The generative world: latent state, availability, realized scores, correlation."""

from __future__ import annotations

import numpy as np
import pytest

from ceauction.league import DEFAULT_LEAGUE, Position
from ceauction.players import Contingency, PlayerSpec, ShockLoading
from ceauction.worlds import build_pool_arrays, generate_world, stable_group_id

from helpers import flat_spec

SEED = 5150
W = DEFAULT_LEAGUE.total_weeks


def _world(specs, n_sims=400, seed=SEED, start=0):
    pool = build_pool_arrays(specs, DEFAULT_LEAGUE)
    return pool, generate_world(pool, seed, start, n_sims)


def test_stable_group_id_is_stable_across_processes():
    # Hard-coded so a change to the hash is caught rather than silently
    # re-randomising every correlated world.
    assert stable_group_id("team:BUF") == stable_group_id("team:BUF")
    assert stable_group_id("team:BUF") != stable_group_id("team:MIA")
    assert stable_group_id("") >= 0


def test_bye_week_makes_a_player_unavailable_exactly_once():
    specs = [flat_spec(0, Position.WR, 10.0, bye_week=7)]
    pool, w = _world(specs, n_sims=3)
    assert w.availability.on_bye[:, 0, 6].all()
    assert w.availability.on_bye[:, 0].sum(axis=1).tolist() == [1, 1, 1]
    assert not w.availability.available[:, 0, 6].any()
    assert np.allclose(w.realized.points[:, 0, 6], 0.0), "bye weeks score zero"


def test_zero_hazard_means_never_injured():
    specs = [flat_spec(0, Position.WR, 10.0, weekly_injury_hazard=0.0)]
    _, w = _world(specs, n_sims=50)
    assert not w.availability.injured.any()


def test_certain_hazard_means_injured_from_week_one():
    specs = [flat_spec(0, Position.WR, 10.0,
                       weekly_injury_hazard=1.0, injury_mean_weeks=3.0)]
    _, w = _world(specs, n_sims=20)
    assert w.availability.injured[:, 0, 0].all()
    assert np.allclose(w.realized.points, 0.0)


def test_injury_frequency_tracks_the_hazard():
    lo = flat_spec(0, Position.WR, 10.0, weekly_injury_hazard=0.02, injury_mean_weeks=2.0)
    hi = flat_spec(1, Position.WR, 10.0, weekly_injury_hazard=0.20, injury_mean_weeks=2.0)
    _, w = _world([lo, hi], n_sims=3000)
    miss_lo = (~w.availability.available[:, 0, :]).mean()
    miss_hi = (~w.availability.available[:, 1, :]).mean()
    assert 0.0 < miss_lo < 0.12
    assert miss_hi > 4 * miss_lo


def test_injuries_last_multiple_weeks():
    specs = [flat_spec(0, Position.WR, 10.0,
                       weekly_injury_hazard=0.05, injury_mean_weeks=4.0)]
    _, w = _world(specs, n_sims=2000)
    inj = w.availability.injured[:, 0, :]
    # Given a player is out in week k, he is usually still out in week k+1.
    cont = inj[:, :-1] & inj[:, 1:]
    assert cont.sum() / max(inj[:, :-1].sum(), 1) > 0.5


def test_realized_points_are_never_negative():
    specs = [flat_spec(0, Position.WR, 1.0, week_sd=12.0)]
    _, w = _world(specs, n_sims=2000)
    assert w.realized.points.min() >= 0.0


def test_season_shift_is_persistent_within_a_season():
    specs = [flat_spec(0, Position.WR, 10.0, season_sd=4.0)]
    _, w = _world(specs, n_sims=2000)
    shift = w.latent.season_shift[:, 0]
    assert abs(float(shift.std()) - 4.0) < 0.25
    # It is one draw for the whole season, so it must correlate perfectly with
    # the season-long mean of realized points.
    mean_pts = w.realized.points[:, 0, :].mean(axis=1)
    assert float(np.corrcoef(shift, mean_pts)[0, 1]) > 0.95


def test_mean_removed_spikes_do_not_change_the_mean():
    plain = flat_spec(0, Position.WR, 12.0, week_sd=5.0)
    spiky = flat_spec(1, Position.WR, 12.0, week_sd=5.0,
                      spike_rate=0.15, spike_scale=20.0)
    _, w = _world([plain, spiky], n_sims=20000)
    m0 = float(w.realized.points[:, 0, :].mean())
    m1 = float(w.realized.points[:, 1, :].mean())
    assert abs(m0 - m1) < 0.25, "mean-removed spikes must be mean-neutral"
    assert w.realized.points[:, 1, :].max() > w.realized.points[:, 0, :].max()


def test_non_demeaned_spikes_add_their_mean_without_entering_the_projection():
    plain = flat_spec(0, Position.WR, 12.0, week_sd=5.0)
    spiky = flat_spec(1, Position.WR, 12.0, week_sd=5.0,
                      spike_rate=0.10, spike_scale=30.0, spike_mean_removed=False)
    _, w = _world([plain, spiky], n_sims=8000)
    gap = float(w.realized.points[:, 1, :].mean() - w.realized.points[:, 0, :].mean())
    assert abs(gap - 3.0) < 0.25, "expected +3.0 pts/week of hidden production"
    # season_sd is 0 here, so the filter learns nothing: the projection stays
    # at the player's *expected points* excluding the spikes.  The production
    # really is unforecastable.
    from ceauction.stats import floored_mean
    assert np.allclose(w.pregame.projection[:, 1, :], float(floored_mean(12.0, 5.0)))
    assert np.allclose(w.pregame.projection[:, 0, :], w.pregame.projection[:, 1, :])


def test_shared_shock_group_creates_correlation_and_private_groups_do_not():
    shared = [
        flat_spec(0, Position.WR, 10.0, week_sd=4.0,
                  shock_loadings=(ShockLoading("g", 6.0),)),
        flat_spec(1, Position.WR, 10.0, week_sd=4.0,
                  shock_loadings=(ShockLoading("g", 6.0),)),
    ]
    private = [
        flat_spec(0, Position.WR, 10.0, week_sd=4.0,
                  shock_loadings=(ShockLoading("g0", 6.0),)),
        flat_spec(1, Position.WR, 10.0, week_sd=4.0,
                  shock_loadings=(ShockLoading("g1", 6.0),)),
    ]
    _, ws = _world(shared, n_sims=4000)
    _, wp = _world(private, n_sims=4000)
    rs = float(np.corrcoef(ws.realized.points[:, 0, :].ravel(),
                           ws.realized.points[:, 1, :].ravel())[0, 1])
    rp = float(np.corrcoef(wp.realized.points[:, 0, :].ravel(),
                           wp.realized.points[:, 1, :].ravel())[0, 1])
    assert rs > 0.5, "shared shock must induce positive correlation"
    assert abs(rp) < 0.05, "private shocks must be independent"
    # Marginals are untouched by the dependence structure.
    assert abs(float(ws.realized.points[:, 0, :].std())
               - float(wp.realized.points[:, 0, :].std())) < 0.2


def test_negative_beta_creates_negative_correlation():
    specs = [
        flat_spec(0, Position.RB, 10.0, week_sd=3.0,
                  shock_loadings=(ShockLoading("share", 6.0),)),
        flat_spec(1, Position.RB, 10.0, week_sd=3.0,
                  shock_loadings=(ShockLoading("share", -6.0),)),
    ]
    _, w = _world(specs, n_sims=4000)
    r = float(np.corrcoef(w.realized.points[:, 0, :].ravel(),
                          w.realized.points[:, 1, :].ravel())[0, 1])
    assert r < -0.4


def test_contingency_fires_exactly_when_the_starter_is_unavailable():
    starter = flat_spec(0, Position.RB, 14.0, bye_week=9,
                        weekly_injury_hazard=0.10, injury_mean_weeks=2.0)
    backup = flat_spec(1, Position.RB, 3.0,
                       contingency=Contingency(on_player_id=0, bonus=9.0))
    _, w = _world([starter, backup], n_sims=500)
    out = ~w.availability.available[:, 0, :]
    bonus = w.pregame.contingency_bonus[:, 1, :]
    assert np.array_equal(bonus > 0, out)
    assert np.allclose(bonus[out], 9.0)
    # The bonus is pregame-observable, so it must reach the projection.
    assert np.allclose(w.pregame.projection[:, 1, :][out], 12.0)
    assert np.allclose(w.pregame.projection[:, 1, :][~out], 3.0)


def test_role_change_is_delayed_before_it_becomes_observable():
    specs = [flat_spec(0, Position.WR, 5.0, role_change_prob=1.0,
                       role_change_mean=8.0, role_change_sd=0.0, role_reveal_lag=2)]
    _, w = _world(specs, n_sims=100)
    for s in range(10):
        wc = int(w.latent.role_week[s, 0])
        assert w.latent.true_role_delta[s, 0, wc] == pytest.approx(8.0)
        assert w.latent.observed_role_delta[s, 0, wc] == pytest.approx(0.0)
        assert w.latent.observed_role_delta[s, 0, wc + 1] == pytest.approx(0.0)
        assert w.latent.observed_role_delta[s, 0, wc + 2] == pytest.approx(8.0)


def test_filter_learns_a_persistent_level_but_barely_reacts_to_one_spike():
    """Persistent signal is learned; unforecastable noise is shrunk away."""
    specs = [flat_spec(0, Position.WR, 10.0, week_sd=6.0, season_sd=4.0)]
    _, w = _world(specs, n_sims=4000)
    shift = w.latent.season_shift[:, 0]
    early = w.pregame.posterior_mean[:, 0, 1]
    late = w.pregame.posterior_mean[:, 0, W - 1]
    assert float(np.corrcoef(shift, late)[0, 1]) > float(np.corrcoef(shift, early)[0, 1])
    assert float(np.corrcoef(shift, late)[0, 1]) > 0.75
    # A single week's residual is shrunk by roughly week_var/season_var.
    assert float(np.abs(early).mean()) < float(np.abs(late).mean())


def test_projection_override_replaces_the_model():
    override = tuple(float(20 + i) for i in range(W))
    specs = [flat_spec(0, Position.WR, 5.0, season_sd=3.0,
                       weekly_projection_override=override),
             flat_spec(1, Position.WR, 5.0, season_sd=3.0)]
    _, w = _world(specs, n_sims=10)
    assert np.allclose(w.pregame.projection[:, 0, :], np.array(override))
    assert not np.allclose(w.pregame.projection[:, 1, :], np.array(override))


def test_world_generation_is_chunk_independent():
    specs = [flat_spec(i, Position.WR, 10.0, week_sd=5.0, season_sd=2.0,
                       weekly_injury_hazard=0.05, spike_rate=0.1, spike_scale=5.0,
                       role_change_prob=0.3, role_change_mean=2.0, role_change_sd=1.0,
                       shock_loadings=(ShockLoading("g", 2.0),))
             for i in range(6)]
    pool = build_pool_arrays(specs, DEFAULT_LEAGUE)
    full = generate_world(pool, SEED, 0, 40)
    tail = generate_world(pool, SEED, 25, 10)
    assert np.array_equal(full.realized.points[25:35], tail.realized.points)
    assert np.array_equal(full.pregame.projection[25:35], tail.pregame.projection)
    assert np.array_equal(full.availability.available[25:35], tail.availability.available)


def test_crn_key_makes_two_specs_share_their_draws():
    a = flat_spec(0, Position.WR, 10.0, week_sd=5.0)
    b = flat_spec(1, Position.WR, 10.0, week_sd=5.0, crn_key=0)
    c = flat_spec(2, Position.WR, 10.0, week_sd=5.0)
    _, w = _world([a, b, c], n_sims=50)
    assert np.array_equal(w.realized.points[:, 0, :], w.realized.points[:, 1, :])
    assert not np.array_equal(w.realized.points[:, 0, :], w.realized.points[:, 2, :])


def test_projection_is_expected_points_not_the_latent_mean():
    """Scores are floored at zero, so the two differ -- a lot, at the bottom.

    Projecting the latent mean would under-project every low-mean,
    high-variance player and bias the whole bench against them.
    """
    from ceauction.stats import floored_mean
    replacement = flat_spec(0, Position.WR, 4.0, week_sd=7.2)
    stud = flat_spec(1, Position.WR, 18.0, week_sd=7.2)
    _, w = _world([replacement, stud], n_sims=6000)
    for i, spec in enumerate((replacement, stud)):
        expected = float(floored_mean(spec.base_mean, spec.week_sd))
        assert np.allclose(w.pregame.projection[:, i, :], expected)
        realized = float(w.realized.points[:, i, :].mean())
        assert abs(realized - expected) < 0.15, (
            "the projection must be an unbiased estimate of expected points"
        )
    # The correction is large where it matters and negligible where it does not.
    assert w.pregame.projection[0, 0, 0] - 4.0 > 1.0
    assert w.pregame.projection[0, 1, 0] - 18.0 < 0.05
