"""The information-learning channel: what may and may not change a projection.

The no-same-week-clairvoyance barrier lives in ``test_information_barrier.py``.
These tests pin the *other* barrier, which is about which past information is
allowed to move a future belief.

The rule: pregame beliefs update from a distinct **observable signal** process
standing in for usage, role and underlying performance indicators.  They never
update from realized fantasy points.  A player who scores 40 on one long
touchdown has had a good week and nothing more; a player whose route share
doubled has told you something about every week that follows.

Every test here is adversarial in the same way: hold the observable signals
fixed, corrupt the realized side as violently as possible, and assert that
projections and lineups do not move at all.
"""

from __future__ import annotations

import numpy as np
import pytest

import ceauction.worlds as W
from ceauction.league import DEFAULT_LEAGUE, Position
from ceauction.lineup_vec import select_lineups_mask
from ceauction.players import ShockLoading
from ceauction.worlds import build_pool_arrays, generate_world

from helpers import flat_spec

SEED = 5309
NW = DEFAULT_LEAGUE.total_weeks


def _world(specs, n_sims=200, seed=SEED, start=0):
    pool = build_pool_arrays(specs, DEFAULT_LEAGUE)
    return pool, generate_world(pool, seed, start, n_sims)


def _rebuild_pregame(pool, world, seed=SEED):
    """Re-run the projection builder against this world's own inputs."""
    n = world.n_sims
    sims3 = (np.arange(n, dtype=np.int64) + world.sim_start).reshape(n, 1, 1)
    keys3 = pool.stream_key.reshape(1, pool.n_players, 1)
    contingency = W._contingency_bonus(pool, world.availability.available)
    forecastable, _ = W._weekly_state(
        pool, seed, sims3, keys3, pool.n_weeks, contingency
    )
    return W._build_pregame(
        pool, seed, sims3, keys3, pool.n_weeks,
        world.latent, world.availability, world.signals, forecastable, contingency,
    )


# --------------------------------------------------------------------------
# 1. A spike changes its own week and nothing else.
# --------------------------------------------------------------------------


def test_a_spike_changes_only_the_week_it_lands_in():
    """The headline behaviour: a random touchdown is not new information.

    Stated exactly rather than statistically. The spiky and quiet arms share a
    CRN key and a signal precision, so their observable channels are the same
    draws; only the spike term differs. Realized scores must separate and
    projections must not move by a single bit.
    """
    common = dict(week_sd=6.0, season_sd=4.0, signal_noise_sd=6.0)
    quiet = flat_spec(0, Position.WR, 10.0, **common)
    spiky = flat_spec(0, Position.WR, 10.0, spike_rate=0.15, spike_scale=25.0,
                      **common)
    _, wq = _world([quiet], n_sims=400)
    _, ws = _world([spiky], n_sims=400)

    hit = ws.realized.spike[:, 0, :] > 10.0
    assert hit.any(), "expected some genuine spike weeks"

    # In a spike week the two arms' realized scores are far apart...
    gap = (ws.realized.points[:, 0, :] - wq.realized.points[:, 0, :])[hit]
    assert float(gap.mean()) > 10.0

    # ...and in every week, before and after, the projections are identical.
    assert np.array_equal(ws.pregame.projection, wq.pregame.projection)

    # The projection is the level plus the posterior and nothing else: there is
    # no term in which a spike could hide.
    assert np.allclose(ws.pregame.projection, 10.0 + ws.pregame.posterior_mean)


def test_adding_a_hundred_points_to_a_past_week_moves_no_future_projection():
    """The audit's adversarial case, stated directly.

    Under the old residual filter this moved the next week's projection by
    roughly half the injected points.  The projection builder no longer has a
    realized parameter, so the injection has nowhere to go.
    """
    spec = flat_spec(0, Position.WR, 10.0, week_sd=6.0, season_sd=4.0)
    pool, w = _world([spec], n_sims=50)

    tampered = w.realized.points.copy()
    tampered[:, 0, 0] += 100.0
    hacked = W.RealizedBatch(points=tampered, spike=w.realized.spike,
                             group_effect=w.realized.group_effect)
    assert not np.allclose(hacked.points, w.realized.points)

    rebuilt = _rebuild_pregame(pool, w)
    assert np.array_equal(rebuilt.projection, w.pregame.projection)
    assert np.array_equal(rebuilt.posterior_mean, w.pregame.posterior_mean)


# --------------------------------------------------------------------------
# 2. Changing the unforecastable side of the process changes no projection.
# --------------------------------------------------------------------------


def _paired(spec_a, spec_b, n_sims=300):
    """Two specs sharing a CRN key, so only the changed parameter can move."""
    pa, wa = _world([spec_a], n_sims=n_sims)
    pb, wb = _world([spec_b], n_sims=n_sims)
    return (pa, wa), (pb, wb)


@pytest.mark.parametrize("label,overrides", [
    ("spike size", dict(spike_rate=0.2, spike_scale=40.0)),
    ("hidden team shock", dict(shock_loadings=(ShockLoading("hidden:team", 9.0),))),
    ("mean-adding hidden production", dict(spike_rate=0.2, spike_scale=15.0,
                                           spike_mean_removed=False)),
])
def test_unforecastable_realized_components_never_reach_a_projection(label, overrides):
    """Idiosyncratic noise, spikes and shared shocks are all invisible pregame.

    ``signal_noise_sd`` is pinned explicitly here so the observable channel is
    identical in both arms; without that, changing ``week_sd`` would also
    change how informative usage is, which is a real effect and not the one
    under test.
    """
    common = dict(week_sd=6.0, season_sd=4.0, signal_noise_sd=6.0)
    plain = flat_spec(0, Position.WR, 12.0, **common)
    loud = flat_spec(0, Position.WR, 12.0, **common, **overrides)
    (_, wa), (_, wb) = _paired(plain, loud)

    assert not np.allclose(wa.realized.points, wb.realized.points), (
        f"{label}: the realized side should have moved"
    )
    assert np.array_equal(wa.signals.level_signal, wb.signals.level_signal)
    assert np.array_equal(wa.pregame.projection, wb.pregame.projection), (
        f"{label} leaked into the projection"
    )


def test_lineup_decisions_are_unchanged_when_only_realized_noise_changes():
    """The consequence that matters: the manager makes the same choices.

    Fifteen players, one league week, one arm given violent unforecastable
    upside.  Every starter mask must be bit-identical.
    """
    template = ([Position.QB] * 3 + [Position.RB] * 4
                + [Position.WR] * 6 + [Position.TE] * 2)
    common = dict(week_sd=6.0, season_sd=3.0, signal_noise_sd=6.0)
    quiet = [flat_spec(i, pos, 6.0 + 0.7 * i, **common)
             for i, pos in enumerate(template)]
    loud = [flat_spec(i, pos, 6.0 + 0.7 * i, spike_rate=0.25, spike_scale=45.0,
                      **common)
            for i, pos in enumerate(template)]

    pa, wa = _world(quiet, n_sims=120)
    pb, wb = _world(loud, n_sims=120)
    assert not np.allclose(wa.realized.points, wb.realized.points)

    pos = pa.position.reshape(1, 1, -1)
    masks = []
    for pool, w in ((pa, wa), (pb, wb)):
        proj = np.moveaxis(w.pregame.projection, 1, -1)      # (S, W, P)
        avail = np.moveaxis(w.availability.available, 1, -1)
        masks.append(select_lineups_mask(proj, avail, pos))
    assert np.array_equal(masks[0], masks[1]), (
        "a lineup changed because of points nobody could have forecast"
    )


# --------------------------------------------------------------------------
# 3. A persistent latent improvement *is* learnable -- through the signal.
# --------------------------------------------------------------------------


def test_a_persistent_level_is_learned_through_the_observable_signal():
    spec = flat_spec(0, Position.WR, 10.0, week_sd=6.0, season_sd=4.0)
    _, w = _world([spec], n_sims=4000)
    shift = w.latent.season_shift[:, 0]
    early = w.pregame.posterior_mean[:, 0, 1]
    late = w.pregame.posterior_mean[:, 0, NW - 1]
    r_early = float(np.corrcoef(shift, early)[0, 1])
    r_late = float(np.corrcoef(shift, late)[0, 1])
    assert r_late > r_early, "the manager must learn as signals accumulate"
    assert r_late > 0.75
    # Week 1 has seen nothing, so the belief is exactly the prior mean.
    assert np.allclose(w.pregame.posterior_mean[:, 0, 0], 0.0)


def test_signal_precision_is_the_dial_that_controls_learning_speed():
    """`signal_noise_sd` is the calibration seam for real usage data."""
    sharp = flat_spec(0, Position.WR, 10.0, week_sd=6.0, season_sd=4.0,
                      signal_noise_sd=0.5)
    vague = flat_spec(0, Position.WR, 10.0, week_sd=6.0, season_sd=4.0,
                      signal_noise_sd=40.0)
    _, ws = _world([sharp], n_sims=3000)
    _, wv = _world([vague], n_sims=3000)
    shift = ws.latent.season_shift[:, 0]
    assert np.array_equal(shift, wv.latent.season_shift[:, 0]), "CRN check"

    r_sharp = float(np.corrcoef(shift, ws.pregame.posterior_mean[:, 0, 3])[0, 1])
    r_vague = float(np.corrcoef(shift, wv.pregame.posterior_mean[:, 0, 3])[0, 1])
    assert r_sharp > 0.95, "clean usage data should reveal the level quickly"
    assert r_vague < 0.35, "useless usage data should reveal almost nothing"


def test_a_zero_season_sd_player_is_never_learned_about():
    """Consensus known to be exactly right: there is nothing to learn."""
    spec = flat_spec(0, Position.WR, 10.0, week_sd=8.0, season_sd=0.0)
    _, w = _world([spec], n_sims=100)
    assert np.allclose(w.pregame.posterior_mean, 0.0)
    assert np.allclose(w.pregame.projection, 10.0)


def test_signals_exist_only_for_weeks_the_player_actually_played():
    """You do not observe usage for a player who did not appear."""
    spec = flat_spec(0, Position.WR, 10.0, week_sd=6.0, season_sd=4.0,
                     bye_week=7, weekly_injury_hazard=0.12, injury_mean_weeks=2.5)
    _, w = _world([spec], n_sims=300)
    assert np.array_equal(w.signals.observed, w.availability.available)
    played = w.availability.available.astype(np.float64)
    assert np.allclose(w.pregame.n_observed, np.cumsum(played, axis=2) - played)


def test_the_signal_stream_is_independent_of_every_realized_stream():
    """Structural independence, not merely an empirically small correlation."""
    spec = flat_spec(0, Position.WR, 10.0, week_sd=6.0, season_sd=4.0,
                     signal_noise_sd=6.0)
    _, w = _world([spec], n_sims=4000)
    resid_signal = (w.signals.level_signal[:, 0, :]
                    - w.latent.season_shift[:, 0][:, None])
    resid_score = (w.realized.points[:, 0, :]
                   - 10.0 - w.latent.season_shift[:, 0][:, None])
    r = float(np.corrcoef(resid_signal.ravel(), resid_score.ravel())[0, 1])
    assert abs(r) < 0.03, f"signal noise correlates with score noise (r = {r:.3f})"


# --------------------------------------------------------------------------
# 4. State decomposition: each component is added exactly once.
# --------------------------------------------------------------------------


def test_a_revealed_role_change_moves_the_projection_by_exactly_its_size():
    """The audit's second finding, as a deterministic identity.

    Baseline 10.0, a certain +20.0 role change revealed one week after it takes
    effect, and no other source of variation anywhere. The projection must be
    exactly 10.0 before the reveal and exactly 30.0 from the reveal onward.

    The old residual filter read the unrevealed change as evidence of a higher
    persistent level and then added the explicit delta again on top, projecting
    roughly 38 instead of 30. The signal channel cannot see role changes at
    all, so no second copy can exist.
    """
    spec = flat_spec(0, Position.WR, 10.0, week_sd=6.0, season_sd=0.0,
                     role_change_prob=1.0, role_change_mean=20.0,
                     role_change_sd=0.0, role_reveal_lag=1)
    _, w = _world([spec], n_sims=200)

    for s in range(200):
        wc = int(w.latent.role_week[s, 0])
        proj = w.pregame.projection[s, 0]
        # Realized production rises in week wc; the projection does not yet.
        assert w.latent.true_role_delta[s, 0, wc] == pytest.approx(20.0)
        assert proj[wc] == pytest.approx(10.0, abs=1e-12), (
            "the pre-reveal week created a second copy of the role change"
        )
        assert proj[wc + 1] == pytest.approx(30.0, abs=1e-12)
        assert proj[NW - 1] == pytest.approx(30.0, abs=1e-12)
        assert proj[wc + 1] - proj[wc] == pytest.approx(20.0, abs=1e-12)
        # Every week before the change is exactly the baseline.
        assert np.allclose(proj[:wc], 10.0)


def test_a_zero_lag_role_change_is_projected_from_the_week_it_takes_effect():
    """An *announced* change: revealed the moment it happens, still once."""
    spec = flat_spec(0, Position.WR, 10.0, week_sd=6.0, season_sd=0.0,
                     role_change_prob=1.0, role_change_mean=20.0,
                     role_change_sd=0.0, role_reveal_lag=0)
    _, w = _world([spec], n_sims=100)
    for s in range(100):
        wc = int(w.latent.role_week[s, 0])
        proj = w.pregame.projection[s, 0]
        assert proj[wc - 1] == pytest.approx(10.0, abs=1e-12)
        assert proj[wc] == pytest.approx(30.0, abs=1e-12)


def test_a_long_unrevealed_role_change_never_leaks_through_the_signal():
    """Four weeks of visibly better play, none of it in the projection.

    This is the strongest form of the double-count test: the player really is
    outscoring his projection for four straight weeks, and the manager still
    projects the baseline, because usage has not yet told him anything.
    """
    spec = flat_spec(0, Position.WR, 10.0, week_sd=6.0, season_sd=4.0,
                     role_change_prob=1.0, role_change_mean=20.0,
                     role_change_sd=0.0, role_reveal_lag=4)
    _, w = _world([spec], n_sims=300)
    for s in range(300):
        wc = int(w.latent.role_week[s, 0])
        proj = w.pregame.projection[s, 0]
        post = w.pregame.posterior_mean[s, 0]
        for k in range(4):
            if wc + k < NW:
                # The projection is the baseline plus the level posterior, with
                # no role term -- the residual outperformance is invisible.
                assert proj[wc + k] == pytest.approx(10.0 + post[wc + k])
        if wc + 4 < NW:
            assert proj[wc + 4] == pytest.approx(30.0 + post[wc + 4])


def test_the_four_components_of_a_realized_score_add_exactly_once():
    """Decomposition identity, checked against the raw arrays.

    persistent level + observable role change + forecastable weekly state
    + unforecastable noise, each appearing exactly one time.
    """
    from ceauction.players import Contingency

    starter = flat_spec(0, Position.RB, 14.0, week_sd=5.0,
                        weekly_injury_hazard=0.15, injury_mean_weeks=2.5)
    backup = flat_spec(1, Position.RB, 6.0, week_sd=4.0, season_sd=3.0,
                       role_change_prob=1.0, role_change_mean=5.0,
                       role_change_sd=0.0, role_reveal_lag=1,
                       spike_rate=0.2, spike_scale=10.0,
                       shock_loadings=(ShockLoading("g", 2.0),),
                       contingency=Contingency(on_player_id=0, bonus=9.0))
    pool, w = _world([starter, backup], n_sims=200)

    persistent = 6.0 + w.latent.season_shift[:, 1][:, None]
    role = w.latent.true_role_delta[:, 1, :]
    forecastable = w.pregame.contingency_bonus[:, 1, :]
    noise = w.realized.group_effect[:, 1, :] + w.realized.spike[:, 1, :]

    total = persistent + role + forecastable + noise
    avail = w.availability.available[:, 1, :]
    idio = np.where(avail, w.realized.points[:, 1, :] - total, 0.0)

    # The only unaccounted-for term is the idiosyncratic draw, whose SD must be
    # exactly week_sd -- if any component were double counted it would not be.
    assert abs(float(idio[avail].std()) - 4.0) < 0.12
    assert abs(float(idio[avail].mean())) < 0.12
