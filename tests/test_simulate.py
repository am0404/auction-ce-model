"""End-to-end season simulation: determinism, chunking, and roster integrity."""

from __future__ import annotations

import numpy as np
import pytest

from ceauction.league import DEFAULT_LEAGUE, Position
from ceauction.lineup import select_lineup
from ceauction.roster import Roster, RosterSet
from ceauction.simulate import pregame_week, simulate_seasons, team_scores
from ceauction.synthetic import make_synthetic_league
from ceauction.worlds import build_pool_arrays, generate_world

from helpers import flat_league, flat_spec

SEED = 31337
T = DEFAULT_LEAGUE.n_teams


def test_identical_seeds_give_identical_results(league):
    a = simulate_seasons(league, 300, SEED)
    b = simulate_seasons(league, 300, SEED)
    assert np.array_equal(a.champion, b.champion)
    assert np.array_equal(a.points, b.points)
    assert np.array_equal(a.seed, b.seed)
    assert np.array_equal(a.made_playoffs, b.made_playoffs)


def test_different_seeds_give_different_results(league):
    a = simulate_seasons(league, 300, SEED)
    b = simulate_seasons(league, 300, SEED + 1)
    assert not np.array_equal(a.champion, b.champion)


@pytest.mark.parametrize("chunk", [1, 7, 64, 300, 1000])
def test_results_do_not_depend_on_chunk_size(league, chunk):
    ref = simulate_seasons(league, 300, SEED, chunk=256)
    out = simulate_seasons(league, 300, SEED, chunk=chunk)
    assert np.array_equal(ref.champion, out.champion)
    assert np.array_equal(ref.points, out.points)
    assert np.array_equal(ref.wins, out.wins)


def test_a_prefix_of_a_longer_run_matches_a_shorter_run(league):
    long_run = simulate_seasons(league, 400, SEED)
    short_run = simulate_seasons(league, 120, SEED)
    assert np.array_equal(long_run.champion[:120], short_run.champion)


def test_championship_equity_is_a_probability_distribution(league):
    out = simulate_seasons(league, 500, SEED)
    ce = out.championship_equity()
    assert ce.shape == (T,)
    assert ce.min() >= 0.0
    assert ce.sum() == pytest.approx(1.0)


def test_exactly_one_champion_per_season_and_champions_made_the_playoffs(league):
    out = simulate_seasons(league, 400, SEED)
    rows = np.arange(out.n_sims)
    assert out.made_playoffs[rows, out.champion].all()
    assert out.made_final[rows, out.champion].all()
    assert out.made_playoffs.sum(axis=1).tolist() == [6] * out.n_sims
    assert out.has_bye.sum(axis=1).tolist() == [2] * out.n_sims


def test_records_and_points_are_in_range(league):
    out = simulate_seasons(league, 200, SEED)
    assert out.wins.min() >= 0.0
    assert out.wins.max() <= 2 * DEFAULT_LEAGUE.regular_season_weeks
    assert out.points.min() > 0.0
    assert out.starters_filled.max() <= 8.0


def test_a_strictly_better_roster_wins_more(league):
    """Doubling one team's projections must raise its CE a lot."""
    from ceauction.players import with_overrides
    boosted = league
    for pid in league.rosters[0].player_ids:
        s = league.spec(pid)
        boosted = boosted.with_pool_player(
            with_overrides(s, base_mean=s.base_mean * 1.5, crn_key=s.player_id)
        )
    base_ce = simulate_seasons(league, 400, SEED).championship_equity()[0]
    up_ce = simulate_seasons(boosted, 400, SEED).championship_equity()[0]
    assert up_ce > base_ce + 0.3


def test_the_roster_is_a_portfolio_not_eight_fixed_starters(league):
    """Over a season, many more than eight players start at least once."""
    pool = build_pool_arrays(league.pool, league.settings)
    world = generate_world(pool, SEED, 0, 1)
    started = set()
    for week in range(DEFAULT_LEAGUE.regular_season_weeks):
        lu = select_lineup(pregame_week(world, league, 0, week))
        started.update(lu.started_ids)
    assert len(started) > 8, "byes, injuries and role changes must rotate the lineup"


def test_bench_depth_has_value_when_a_starter_is_lost(league):
    """Removing a starter costs less when the bench behind him is strong."""
    from ceauction.players import with_overrides
    ranked = sorted(
        (league.spec(p) for p in league.rosters[0].player_ids),
        key=lambda s: -s.base_mean,
    )
    starter = ranked[1]
    weak_bench = league
    for s in ranked[8:]:
        weak_bench = weak_bench.with_pool_player(
            with_overrides(s, base_mean=0.5, crn_key=s.player_id)
        )
    hurt_strong = league.with_pool_player(
        with_overrides(starter, weekly_injury_hazard=0.5, crn_key=starter.player_id)
    )
    hurt_weak = weak_bench.with_pool_player(
        with_overrides(starter, weekly_injury_hazard=0.5, crn_key=starter.player_id)
    )
    loss_strong = (simulate_seasons(league, 600, SEED).points[:, 0].mean()
                   - simulate_seasons(hurt_strong, 600, SEED).points[:, 0].mean())
    loss_weak = (simulate_seasons(weak_bench, 600, SEED).points[:, 0].mean()
                 - simulate_seasons(hurt_weak, 600, SEED).points[:, 0].mean())
    assert loss_weak > loss_strong


# --------------------------------------------------------------------------
# Roster validation
# --------------------------------------------------------------------------


def test_roster_set_rejects_wrong_sizes_duplicates_and_shared_players():
    base = flat_league()
    with pytest.raises(ValueError, match="expected 15"):
        RosterSet(base.pool, (Roster("x", base.rosters[0].player_ids[:14]),)
                  + base.rosters[1:], DEFAULT_LEAGUE)
    dup = base.rosters[0].player_ids[:14] + base.rosters[0].player_ids[:1]
    with pytest.raises(ValueError, match="duplicate"):
        RosterSet(base.pool, (Roster("x", dup),) + base.rosters[1:], DEFAULT_LEAGUE)
    stolen = base.rosters[0].player_ids[:14] + base.rosters[1].player_ids[:1]
    with pytest.raises(ValueError, match="both"):
        RosterSet(base.pool, (Roster("x", stolen),) + base.rosters[1:], DEFAULT_LEAGUE)
    with pytest.raises(ValueError, match="12 rosters"):
        RosterSet(base.pool, base.rosters[:11], DEFAULT_LEAGUE)


def test_roster_replaced_preserves_order():
    r = Roster("x", (1, 2, 3, 4))
    assert r.replaced(3, 99).player_ids == (1, 2, 99, 4)
    with pytest.raises(KeyError):
        r.replaced(77, 99)
    with pytest.raises(ValueError):
        r.replaced(1, 2)


def test_missing_contingency_target_is_rejected():
    from ceauction.players import Contingency
    spec = flat_spec(0, Position.RB, 5.0, contingency=Contingency(999, 5.0))
    with pytest.raises(KeyError, match="not in the pool"):
        build_pool_arrays([spec], DEFAULT_LEAGUE)
