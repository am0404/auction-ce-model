"""The CE laboratory: every experiment must build a legal league and run paired."""

from __future__ import annotations

import numpy as np
import pytest

from ceauction.experiments import (
    EXPERIMENTS,
    FOCUS_TEAM,
    _matched_variance,
    lab_player,
    roster_by_strength,
    run_experiment,
    swap_in,
    tweak,
)
from ceauction.league import DEFAULT_LEAGUE, Position
from ceauction.roster import RosterSet
from ceauction.synthetic import make_synthetic_league

SEED = 8080
SMOKE_SIMS = 120


def test_every_required_experiment_exists():
    required = {
        "marginal-point", "second-qb", "volatility", "spikes", "concentration",
        "injury", "bench-correlation", "stack", "handcuff", "opponent-placement",
    }
    assert required <= set(EXPERIMENTS)


@pytest.mark.parametrize("key", sorted(EXPERIMENTS))
def test_experiment_runs_and_reports_paired_results(key, league):
    out = run_experiment(key, SMOKE_SIMS, SEED, base=league)
    assert out.comparisons, f"{key} produced no comparisons"
    for c in out.comparisons:
        assert c.n_sims == SMOKE_SIMS
        assert c.team_index == FOCUS_TEAM
        assert np.isfinite(c.delta_ce)
        assert np.isfinite(c.delta_ce_se)
        assert c.report_a.championship_equity >= 0.0
        assert c.report_b.championship_equity >= 0.0
        assert c.label and c.scenario_a and c.scenario_b
        assert isinstance(c.format(), str) and len(c.format()) > 200
    assert out.interpretation


def test_swap_in_keeps_the_league_legal(league):
    victim = roster_by_strength(league, FOCUS_TEAM)[-1]
    new = lab_player(900_999, "LAB-X", Position.WR, 12.0)
    rs = swap_in(league, FOCUS_TEAM, victim.player_id, new)
    assert isinstance(rs, RosterSet)  # __post_init__ validated it
    assert 900_999 in rs.rosters[FOCUS_TEAM].player_ids
    assert victim.player_id not in rs.rosters[FOCUS_TEAM].player_ids
    assert len(rs.rosters[FOCUS_TEAM]) == DEFAULT_LEAGUE.roster_size
    for t in range(1, DEFAULT_LEAGUE.n_teams):
        assert rs.rosters[t].player_ids == league.rosters[t].player_ids


def test_tweak_preserves_the_crn_key(league):
    pid = league.rosters[FOCUS_TEAM].player_ids[0]
    before = league.spec(pid)
    after = tweak(league, pid, base_mean=99.0).spec(pid)
    assert after.base_mean == 99.0
    assert after.stream_key == before.stream_key
    assert after.player_id == before.player_id


def test_matched_variance_preserves_total_weekly_sd():
    idio = _matched_variance(8.0, 5.0)
    assert idio ** 2 + 5.0 ** 2 == pytest.approx(64.0)
    with pytest.raises(ValueError):
        _matched_variance(4.0, 5.0)


def test_lab_players_are_inert_by_default():
    p = lab_player(1, "x", Position.WR, 10.0)
    assert p.weekly_injury_hazard == 0.0
    assert p.spike_rate == 0.0
    assert p.role_change_prob == 0.0
    assert p.shock_loadings == ()
    assert p.bye_week == 0
    assert p.data_source == "SYNTHETIC"
    assert p.stream_key == 1


def test_moving_a_stud_between_two_rivals_does_not_move_your_ce(league):
    """The control experiment must come back statistically flat.

    A significant result here would mean the schedule is not exchangeable
    across teams, or that team identity leaks into the standings somewhere.
    """
    out = run_experiment("opponent-placement", 3000, SEED, base=league)
    c = out.comparisons[0]
    assert abs(c.delta_ce_z) < 3.0, (
        f"the focus team's CE moved when a rival's roster was relabelled "
        f"(dCE = {c.delta_ce:+.5f}, z = {c.delta_ce_z:+.2f})"
    )
    assert abs(c.delta_points_per_week) < 1e-9, (
        "the focus team's own scoring must be byte-identical"
    )


def test_focus_team_players_are_identical_across_the_control_arms(league):
    out = run_experiment("opponent-placement", 120, SEED, base=league)
    c = out.comparisons[0]
    assert c.report_a.mean_regular_season_points == pytest.approx(
        c.report_b.mean_regular_season_points, rel=0, abs=1e-9
    )


def test_unknown_experiment_key_is_rejected():
    with pytest.raises(KeyError, match="unknown experiment"):
        run_experiment("does-not-exist", 10, SEED)
