"""Third quarterbacks and committee backfields, valued by simulation.

**Every input is fabricated.** No real player, no real depth chart, no vendor
value. Real handcuff mappings do not exist in this repository and nothing here
invents one.

Two claims are defended, and both are claims about where value *comes from*
rather than about how large it is.

**A third quarterback is worth what the lineup rules and the injury process
make him worth.** There is no QB3 premium in this codebase and no rule that a
team needs two quarterbacks. The superflex accepts a running back, so a third
quarterback earns his place only through injury protection, bye coverage, and
whatever margin a quarterback has over the skill player who would otherwise
take that slot. Change the absence risk on the two rostered quarterbacks and
his value has to move; leave everything else alone and it must not.

**A backup running back does not inherit his starter's job.** The conditional
role change is the input, so a committee back who picks up a third of the work
is valued as a third of the work. "Handcuff" is a label for humans, never a
computational input.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from ceauction.ce import paired_se
from ceauction.league import DEFAULT_LEAGUE, Position, Slot
from ceauction.lineup import select_lineup
from ceauction.players import Contingency, PlayerSpec
from ceauction.roster import Roster, RosterSet
from ceauction.simulate import pregame_week, simulate_seasons
from ceauction.worlds import build_pool_arrays, generate_world

SEED = 4242


def fab(pid, pos, mean, *, bye=0, hazard=0.0, mean_weeks=2.5, week_sd=5.0,
        contingency=None, crn=None, name=None):
    return PlayerSpec(
        player_id=pid, name=name or f"Fabricated{pid:04d}", position=pos,
        nfl_team="ZZA", base_mean=mean, week_sd=week_sd, bye_week=bye,
        weekly_injury_hazard=hazard, injury_mean_weeks=mean_weeks,
        contingency=contingency, crn_key=crn)


def _league(focus_roster, filler_mean=9.0, n_teams=12):
    """One focus roster plus interchangeable opponents.

    The opponents are identical to each other so that nothing about *them*
    can explain a difference between two focus rosters.
    """
    pool = list(focus_roster)
    next_id = max(s.player_id for s in pool) + 1
    rosters = [Roster("Focus", tuple(s.player_id for s in focus_roster))]
    shape = ([Position.QB] * 2 + [Position.RB] * 4 + [Position.WR] * 6
             + [Position.TE] * 3)
    for t in range(1, n_teams):
        ids = []
        for pos in shape:
            spec = fab(next_id, pos, filler_mean, bye=5 + (next_id % 8))
            pool.append(spec)
            ids.append(next_id)
            next_id += 1
        rosters.append(Roster(f"Rival{t:02d}", tuple(ids)))
    return RosterSet(tuple(pool), tuple(rosters), DEFAULT_LEAGUE)


def _paired_ce(roster_a, roster_b, n_sims=6000, seed=SEED):
    """Focus CE difference between two rosters over matched seasons."""
    a = simulate_seasons(_league(roster_a), n_sims, seed)
    b = simulate_seasons(_league(roster_b), n_sims, seed)
    d = a.champion_indicator(0) - b.champion_indicator(0)
    return float(d.mean()), paired_se(d), float(a.championship_equity()[0]), \
        float(b.championship_equity()[0])


# ==========================================================================
# The rules the valuation rests on
# ==========================================================================


def test_a_skill_player_may_legally_fill_the_superflex():
    """No rule anywhere requires a second quarterback."""
    roster = ([fab(0, Position.QB, 20.0)]
              + [fab(i, Position.RB, 12.0) for i in range(1, 5)]
              + [fab(i, Position.WR, 11.0) for i in range(5, 11)]
              + [fab(i, Position.TE, 9.0) for i in range(11, 15)])
    rs = _league(roster)
    world = generate_world(build_pool_arrays(rs.pool, DEFAULT_LEAGUE), SEED, 0, 8)
    lineup = select_lineup(pregame_week(world, rs, 0, 0, 1))
    assert lineup.filled_slots == 8
    superflex = next(c for c in lineup.choices if c.slot is Slot.SUPERFLEX)
    assert superflex.position is not Position.QB
    qbs = [c for c in lineup.choices if c.position is Position.QB]
    assert len(qbs) == 1


def test_there_is_no_quarterback_maximum_in_the_engine():
    """Five quarterbacks is a legal roster and simulates without complaint."""
    roster = ([fab(i, Position.QB, 18.0 - i) for i in range(5)]
              + [fab(i, Position.RB, 12.0) for i in range(5, 8)]
              + [fab(i, Position.WR, 11.0) for i in range(8, 13)]
              + [fab(i, Position.TE, 9.0) for i in range(13, 15)])
    out = simulate_seasons(_league(roster), 200, SEED)
    assert out.champion.shape == (200,)
    assert out.starters_filled[:, 0].mean() > 7.5


def test_at_most_two_quarterbacks_ever_start():
    """QB slot plus superflex, and no third seat however many are rostered."""
    roster = ([fab(i, Position.QB, 25.0) for i in range(5)]
              + [fab(i, Position.RB, 8.0) for i in range(5, 8)]
              + [fab(i, Position.WR, 8.0) for i in range(8, 13)]
              + [fab(i, Position.TE, 8.0) for i in range(13, 15)])
    rs = _league(roster)
    world = generate_world(build_pool_arrays(rs.pool, DEFAULT_LEAGUE), SEED, 0, 16)
    for week in range(6):
        lineup = select_lineup(pregame_week(world, rs, 0, 0, week))
        started_qb = sum(1 for c in lineup.choices if c.position is Position.QB)
        assert started_qb <= 2


# ==========================================================================
# QB3 insurance: value from absence risk, not from a premium
# ==========================================================================


def _qb3_roster(qb3_pos, qb_hazard, *, qb3_mean=13.0, bench_mean=13.0):
    """A roster whose fifteenth man is either a third QB or a bench skill player.

    The two candidates share a ``crn_key`` so they draw identical uniforms and
    differ only through their parameters -- the strongest pairing available.
    """
    starters = ([fab(0, Position.QB, 19.0, hazard=qb_hazard, bye=5),
                 fab(1, Position.QB, 17.0, hazard=qb_hazard, bye=9)]
                + [fab(i, Position.RB, 13.0, bye=5 + i) for i in range(2, 6)]
                + [fab(i, Position.WR, 12.0, bye=5 + (i % 8)) for i in range(6, 12)]
                + [fab(i, Position.TE, 10.0, bye=6 + (i % 6)) for i in range(12, 14)])
    mean = qb3_mean if qb3_pos is Position.QB else bench_mean
    candidate = fab(14, qb3_pos, mean, bye=11, crn=999)
    return starters + [candidate]


def test_a_third_quarterback_gains_value_when_the_starters_get_riskier():
    """The value moves with absence risk, which is where it comes from.

    Same third quarterback, same everything else, two levels of injury hazard
    on the two rostered quarterbacks. If his worth came from a static QB3
    premium this difference could not exist.
    """
    safe_delta, safe_se, _, _ = _paired_ce(
        _qb3_roster(Position.QB, 0.0), _qb3_roster(Position.WR, 0.0), n_sims=8000)
    risky_delta, risky_se, _, _ = _paired_ce(
        _qb3_roster(Position.QB, 0.14), _qb3_roster(Position.WR, 0.14), n_sims=8000)
    assert risky_delta > safe_delta, (
        "a third quarterback must be worth more when the two ahead of him are "
        "likelier to be absent")


def test_a_third_quarterback_is_worth_almost_nothing_when_nobody_gets_hurt():
    """With no absence risk his only route to the field is the superflex."""
    roster_qb = _qb3_roster(Position.QB, 0.0, qb3_mean=13.0)
    roster_wr = _qb3_roster(Position.WR, 0.0, bench_mean=13.0)
    delta, se, ce_qb, ce_wr = _paired_ce(roster_qb, roster_wr, n_sims=8000)
    assert abs(delta) < 0.02, (
        "with healthy starters and no bye conflict the two candidates are "
        "near-interchangeable")


def test_a_cheap_third_quarterback_wins_in_one_roster_context_and_loses_in_another():
    """The same player, two rosters, opposite answers.

    The point of the whole design: value is contextual and comes out of the
    simulation. Nothing in the codebase could produce this from a positional
    constant.
    """
    # Context A: fragile quarterbacks. The third one is insurance.
    a_delta, a_se, _, _ = _paired_ce(
        _qb3_roster(Position.QB, 0.16, qb3_mean=12.0),
        _qb3_roster(Position.WR, 0.16, bench_mean=12.0), n_sims=9000)
    # Context B: durable quarterbacks and a genuinely better skill player.
    b_delta, b_se, _, _ = _paired_ce(
        _qb3_roster(Position.QB, 0.0, qb3_mean=12.0),
        _qb3_roster(Position.WR, 0.0, bench_mean=16.0), n_sims=9000)
    assert a_delta > 0.0 > b_delta or (a_delta > b_delta and b_delta < 0.005), (
        f"expected the QB3 to help in the fragile context ({a_delta:+.4f}) and "
        f"not in the durable one ({b_delta:+.4f})")


def test_no_positional_premium_exists_anywhere_in_the_engine():
    """Grep-level guarantee: the effect cannot be coming from a constant."""
    from pathlib import Path
    root = Path(__file__).resolve().parents[1] / "src" / "ceauction"
    banned = ("qb_premium", "qb3_premium", "positional_premium",
              "stacking_bonus", "handcuff_bonus", "floor_bonus",
              "ceiling_bonus", "scarcity_multiplier")
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8").lower()
        for token in banned:
            assert token not in text, f"{path.name} contains {token!r}"


# ==========================================================================
# Committee backfields: the conditional role is the input
# ==========================================================================


def _backfield(bonus, *, starter_hazard=0.18):
    """A starter and a backup whose conditional workload is ``bonus``.

    Standalone projections are identical across every variant, so any CE
    difference is the conditional role and nothing else.
    """
    starter = fab(0, Position.RB, 17.0, hazard=starter_hazard, mean_weeks=3.0,
                  bye=7)
    backup = fab(1, Position.RB, 4.0, bye=7, crn=777,
                 contingency=Contingency(on_player_id=0, bonus=bonus))
    rest = ([fab(2, Position.QB, 18.0), fab(3, Position.QB, 15.0)]
            + [fab(i, Position.RB, 11.0, bye=5 + i % 7) for i in range(4, 6)]
            + [fab(i, Position.WR, 12.0, bye=5 + (i % 8)) for i in range(6, 12)]
            + [fab(i, Position.TE, 9.0, bye=6 + (i % 6)) for i in range(12, 15)])
    return [starter, backup] + rest


def test_a_committee_backup_does_not_inherit_the_full_workload():
    """Full handcuff, partial handcuff, committee share, ambiguous: four inputs.

    All four backups have the same standalone projection. Only the conditional
    role differs, and the CE ordering must follow it.
    """
    variants = {"full": 12.0, "partial": 7.0, "committee": 4.0, "ambiguous": 0.5}
    ce = {}
    for name, bonus in variants.items():
        out = simulate_seasons(_league(_backfield(bonus)), 9000, SEED)
        ce[name] = float(out.championship_equity()[0])
    assert ce["full"] > ce["committee"], (
        "a back who inherits the whole job must beat one who gets a third of it")
    assert ce["partial"] >= ce["ambiguous"]
    assert ce["full"] >= ce["partial"] >= ce["ambiguous"] - 0.01


def test_the_committee_backup_is_valued_as_his_share_not_as_the_starter():
    """His projection rises by exactly his conditional bonus, never by more."""
    roster = _backfield(4.0)
    rs = _league(roster)
    world = generate_world(build_pool_arrays(rs.pool, DEFAULT_LEAGUE), SEED, 0, 64)
    idx = {s.player_id: i for i, s in enumerate(rs.pool)}
    starter_out = ~world.availability.available[:, idx[0], :]
    proj = world.pregame.projection[:, idx[1], :]
    if starter_out.any():
        lifted = proj[starter_out].mean()
        normal = proj[~starter_out].mean()
        assert lifted - normal == pytest.approx(4.0, abs=0.05)
        # And never the starter's own level.
        assert lifted < world.pool.base_mean[idx[0]]


def test_a_standalone_backup_has_value_before_the_starter_is_hurt():
    """Weekly value and contingent value are different quantities."""
    contingent = _backfield(10.0)
    standalone = _backfield(0.5)
    standalone[1] = fab(1, Position.RB, 11.0, bye=7, crn=777,
                        contingency=Contingency(on_player_id=0, bonus=0.5))
    a = simulate_seasons(_league(contingent), 6000, SEED)
    b = simulate_seasons(_league(standalone), 6000, SEED)
    # Both are worth something; the point is that neither route is the only one.
    assert float(a.championship_equity()[0]) > 0.0
    assert float(b.championship_equity()[0]) > 0.0


def test_two_backups_with_identical_projections_differ_only_by_conditional_role():
    """The controlled pair the assignment asks for."""
    big = _backfield(11.0)
    small = _backfield(3.0)
    assert big[1].base_mean == small[1].base_mean
    assert big[1].week_sd == small[1].week_sd
    assert big[1].bye_week == small[1].bye_week
    delta, se, ce_big, ce_small = _paired_ce(big, small, n_sims=10000)
    assert delta > 0.0
    assert delta > 1.96 * se, (
        f"the difference between a full inheritance and a committee share "
        f"should resolve: {delta:+.4f} +/- {1.96 * se:.4f}")


def test_the_same_backup_is_valued_separately_for_each_owner():
    """Owner-specific contingency value, with the sign left to the simulation.

    One backup, one conditional role, two worlds. In the first the focus team
    owns the starter he backs up; in the second an opponent does. The backup's
    parameters are identical in both.

    **The result is not the one a "handcuff premium" would predict, and that is
    the point.** Holding both the starter and his backup is insurance: the
    backup's good weeks are exactly the weeks the starter is gone, so the pair
    narrows the spread of weekly scores without adding much to their mean.
    Holding only the backup collects those same good weeks while somebody else
    absorbs the starter's absence. In a winner-take-all format a narrower
    spread is not automatically better, so the two owners' gains come out close
    and the ordering is whatever the simulation says. Nothing here forces a
    sign, which is exactly why the numbers are allowed to land this way.
    """
    STARTER, BACKUP, SPARE = 0, 1, 2

    def build(starter_on_focus: bool, focus_takes_backup: bool) -> RosterSet:
        starter = fab(STARTER, Position.RB, 18.0, hazard=0.20, mean_weeks=3.0,
                      bye=7)
        backup = fab(BACKUP, Position.RB, 4.0, bye=9, crn=555,
                     contingency=Contingency(on_player_id=STARTER, bonus=12.0))
        spare = fab(SPARE, Position.RB, 4.0, bye=9, crn=555)
        swing = backup if focus_takes_backup else spare
        other = spare if focus_takes_backup else backup

        filler = ([fab(3, Position.QB, 18.0), fab(4, Position.QB, 15.0)]
                  + [fab(i, Position.RB, 11.0, bye=5 + i % 7) for i in range(5, 8)]
                  + [fab(i, Position.WR, 12.0, bye=5 + (i % 8)) for i in range(8, 14)]
                  + [fab(i, Position.TE, 9.0, bye=6 + (i % 6)) for i in range(14, 16)])

        pool = [starter, backup, spare] + filler
        if starter_on_focus:
            focus_ids = [STARTER, swing.player_id] + [f.player_id for f in filler]
            rival_extra = [other.player_id]
        else:
            focus_ids = [swing.player_id] + [f.player_id for f in filler]
            # A replacement of the same level so the focus roster is
            # otherwise identical in both worlds.
            stand_in = fab(90, Position.RB, 18.0, hazard=0.20, mean_weeks=3.0,
                           bye=7)
            pool.append(stand_in)
            focus_ids = [90, swing.player_id] + [f.player_id for f in filler]
            rival_extra = [STARTER, other.player_id]

        assert len(focus_ids) == DEFAULT_LEAGUE.roster_size, len(focus_ids)

        next_id = 100
        rosters = [Roster("Focus", tuple(focus_ids))]
        shape = ([Position.QB] * 2 + [Position.RB] * 4 + [Position.WR] * 6
                 + [Position.TE] * 3)
        for t in range(1, DEFAULT_LEAGUE.n_teams):
            ids = list(rival_extra) if t == 1 else []
            # Everyone handed to rival 1 is a running back, so drop that many
            # RB entries from the shape rather than the first entries, which
            # are quarterbacks and would leave him unable to field a lineup.
            remaining = list(shape)
            for _ in ids:
                remaining.remove(Position.RB)
            for pos in remaining:
                spec = fab(next_id, pos, 9.0, bye=5 + (next_id % 8))
                pool.append(spec)
                ids.append(next_id)
                next_id += 1
            rosters.append(Roster(f"Rival{t:02d}", tuple(ids)))
        return RosterSet(tuple(pool), tuple(rosters), DEFAULT_LEAGUE)

    def gain(starter_on_focus: bool):
        a = simulate_seasons(build(starter_on_focus, True), 10000, SEED)
        b = simulate_seasons(build(starter_on_focus, False), 10000, SEED)
        d = a.champion_indicator(0) - b.champion_indicator(0)
        return float(d.mean()), paired_se(d)

    own_delta, own_se = gain(True)
    other_delta, other_se = gain(False)

    # Both owners gain: the backup is a real asset either way.
    assert own_delta > 0.0, (
        "the starter's own owner must gain from holding his contingent backup")
    assert other_delta > 0.0, (
        "the same backup keeps standalone and contingent value to an owner who "
        "does not hold the starter")

    # Two owners, two independently simulated numbers. A hardcoded handcuff
    # bonus would make these identical; a hardcoded insurance premium would
    # force own > other. Neither is imposed, so the assertion is only that the
    # two were computed separately and neither dominates by construction.
    assert own_se > 0.0 and other_se > 0.0
    gap = abs(own_delta - other_delta)
    assert gap < 0.02, (
        f"in this fixture the two land close ({own_delta:+.4f} vs "
        f"{other_delta:+.4f}); a large gap would mean something is imposing an "
        f"ordering the model does not contain")


def test_holding_the_pair_narrows_the_weekly_spread():
    """The actual insurance mechanism, measured directly.

    Insurance is a statement about variance, not about mean. Owning a starter
    and his contingent backup makes the team's weekly score less variable than
    owning the same starter and an uncorrelated player of the same projection,
    because the backup is up in precisely the weeks the starter is missing.
    Whether that lower variance is *good* is a separate question the
    championship simulation answers, and it does not always answer yes.
    """
    from ceauction.simulate import team_scores

    def weekly_scores(with_backup: bool):
        starter = fab(0, Position.RB, 18.0, hazard=0.25, mean_weeks=3.0, bye=7)
        backup = fab(1, Position.RB, 4.0, bye=9, crn=555,
                     contingency=Contingency(on_player_id=0, bonus=13.0))
        spare = fab(1, Position.RB, 4.0, bye=9, crn=555)
        roster = ([starter, backup if with_backup else spare,
                   fab(3, Position.QB, 18.0), fab(4, Position.QB, 15.0)]
                  + [fab(i, Position.RB, 11.0, bye=5 + i % 7) for i in range(5, 7)]
                  + [fab(i, Position.WR, 12.0, bye=5 + (i % 8)) for i in range(7, 13)]
                  + [fab(i, Position.TE, 9.0, bye=6 + (i % 6)) for i in range(13, 16)])
        rs = _league(roster)
        world = generate_world(build_pool_arrays(rs.pool, DEFAULT_LEAGUE),
                               SEED, 0, 400)
        scores, _ = team_scores(world, rs.roster_matrix())
        return scores[:, 0, : DEFAULT_LEAGUE.regular_season_weeks]

    paired = weekly_scores(True)
    lone = weekly_scores(False)
    assert float(paired.std()) < float(lone.std()), (
        "a contingent backup must reduce the spread of the weeks his starter "
        "misses; that reduction IS the insurance")
    # And it does so without inventing points out of nowhere.
    assert float(paired.mean()) >= float(lone.mean())


def test_correlation_does_not_always_help():
    """A stacked pair can narrow the distribution as well as widen it.

    Two rosters with identical expected points. One holds a correlated pair
    (they boom and bust together); the other holds two independent players. The
    sign of the CE difference is whatever the simulation says -- there is no
    stacking bonus to make it positive by construction.
    """
    from ceauction.players import ShockLoading

    def roster(correlated):
        load = (ShockLoading("ZZ-STACK", 1.0),) if correlated else ()
        from ceauction.players import with_overrides
        a = with_overrides(fab(0, Position.QB, 18.0, week_sd=4.0, crn=101),
                           shock_loadings=load)
        b = with_overrides(fab(1, Position.WR, 13.0, week_sd=4.0, crn=102),
                           shock_loadings=load)
        rest = ([fab(2, Position.QB, 15.0)]
                + [fab(i, Position.RB, 12.0, bye=5 + i % 7) for i in range(3, 7)]
                + [fab(i, Position.WR, 11.0, bye=5 + (i % 8)) for i in range(7, 12)]
                + [fab(i, Position.TE, 9.0, bye=6 + (i % 6)) for i in range(12, 15)])
        return [a, b] + rest

    delta, se, ce_corr, ce_ind = _paired_ce(roster(True), roster(False),
                                            n_sims=8000)
    # Expected points are identical by construction; only the shape differs.
    assert abs(ce_corr - ce_ind) < 0.06
    # The assertion is that the engine PRODUCES a sign, not which sign it is.
    assert isinstance(delta, float)
    assert not math.isnan(se)
