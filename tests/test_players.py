"""PlayerSpec: validation, provenance labelling, and the real-data seam.

``PlayerSpec`` is the only thing the CE engine needs to know about a player, so
it is the interface real data will arrive through.  These tests pin its
contract.
"""

from __future__ import annotations

import numpy as np
import pytest

from ceauction.league import DEFAULT_LEAGUE, Position
from ceauction.players import Contingency, PlayerSpec, ShockLoading, with_overrides
from ceauction.synthetic import SyntheticConfig, make_synthetic_league, make_synthetic_pool
from ceauction.worlds import build_pool_arrays

from helpers import flat_spec


def _spec(**kw):
    base = dict(player_id=1, name="x", position=Position.WR, nfl_team="TST",
                base_mean=10.0, week_sd=5.0)
    base.update(kw)
    return PlayerSpec(**base)


@pytest.mark.parametrize("bad", [
    dict(week_sd=-1.0),
    dict(season_sd=-0.1),
    dict(weekly_injury_hazard=1.5),
    dict(weekly_injury_hazard=-0.01),
    dict(spike_rate=1.2),
    dict(role_change_prob=-0.5),
    dict(injury_mean_weeks=0.0),
    dict(role_reveal_lag=-1),
])
def test_invalid_parameters_are_rejected(bad):
    with pytest.raises(ValueError):
        _spec(**bad)


def test_boundary_values_are_accepted():
    _spec(weekly_injury_hazard=0.0)
    _spec(weekly_injury_hazard=1.0)
    _spec(spike_rate=1.0)
    _spec(week_sd=0.0, season_sd=0.0)
    _spec(role_reveal_lag=0)


def test_stream_key_defaults_to_player_id_and_crn_key_overrides_it():
    assert _spec(player_id=7).stream_key == 7
    assert _spec(player_id=7, crn_key=99).stream_key == 99


def test_specs_are_immutable_and_with_overrides_copies():
    s = _spec()
    with pytest.raises(Exception):
        s.base_mean = 20.0  # frozen dataclass
    t = with_overrides(s, base_mean=20.0)
    assert t.base_mean == 20.0 and s.base_mean == 10.0
    assert t.player_id == s.player_id


def test_provenance_is_carried_and_every_synthetic_spec_is_labelled():
    pool = make_synthetic_pool()
    assert len(pool) == DEFAULT_LEAGUE.pool_size
    assert all(p.data_source == "SYNTHETIC" for p in pool)
    assert all(p.is_synthetic for p in pool)
    assert all(p.name.startswith("SYN-") for p in pool)
    real = _spec(data_source="REAL:vendor")
    assert not real.is_synthetic


def test_projection_override_length_is_validated():
    good = tuple(float(i) for i in range(DEFAULT_LEAGUE.total_weeks))
    build_pool_arrays([_spec(weekly_projection_override=good)], DEFAULT_LEAGUE)
    with pytest.raises(ValueError, match="length 17"):
        build_pool_arrays([_spec(weekly_projection_override=(1.0, 2.0))], DEFAULT_LEAGUE)


def test_shock_loadings_accumulate_into_the_beta_matrix():
    a = _spec(player_id=1, shock_loadings=(ShockLoading("g", 2.0), ShockLoading("g", 1.5)))
    b = _spec(player_id=2, shock_loadings=(ShockLoading("h", -3.0),))
    pool = build_pool_arrays([a, b], DEFAULT_LEAGUE)
    assert pool.group_names == ("g", "h")
    assert pool.beta[0, 0] == pytest.approx(3.5)
    assert pool.beta[0, 1] == 0.0
    assert pool.beta[1, 1] == pytest.approx(-3.0)


def test_bye_week_zero_means_no_bye():
    pool = build_pool_arrays([_spec(bye_week=0), _spec(player_id=2, bye_week=7)],
                             DEFAULT_LEAGUE)
    assert pool.bye_index[0] == -1
    assert pool.bye_index[1] == 6


def test_synthetic_league_is_balanced_and_legal():
    """No team may collect the best player at every position."""
    rs = make_synthetic_league()
    tops = []
    for t in range(DEFAULT_LEAGUE.n_teams):
        means = sorted((rs.spec(p).base_mean for p in rs.rosters[t].player_ids),
                       reverse=True)
        tops.append(sum(means[:8]))
        counts = rs.position_counts(t)
        assert counts[Position.QB] >= 1
        assert counts[Position.RB] >= 2
        assert counts[Position.WR] + counts[Position.TE] >= 3
        assert sum(counts.values()) == DEFAULT_LEAGUE.roster_size
    assert max(tops) - min(tops) < 0.10 * float(np.mean(tops)), (
        f"snake draft is unbalanced: {min(tops):.1f} to {max(tops):.1f}"
    )


def test_synthetic_config_switches_actually_switch_things_off():
    off = SyntheticConfig(enable_team_shocks=False, enable_stacks=False,
                          enable_contingency=False)
    pool = make_synthetic_pool(off)
    assert all(p.shock_loadings == () for p in pool)
    assert all(p.contingency is None for p in pool)
    on = make_synthetic_pool()
    assert any(p.shock_loadings for p in on)
    assert any(p.contingency is not None for p in on)


def test_contingency_points_at_a_real_teammate():
    pool = make_synthetic_pool()
    ids = {p.player_id for p in pool}
    for p in pool:
        if p.contingency is not None:
            assert p.contingency.on_player_id in ids
            assert p.contingency.on_player_id != p.player_id
            assert p.contingency.bonus > 0
