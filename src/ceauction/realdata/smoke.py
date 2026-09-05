"""A real-data CE smoke test: does the engine run on real specs, and behave?

**This is integration testing, not roster construction.** The twelve rosters
below are built by a deterministic snake over the real draftable pool purely so
the engine has twelve legal, disjoint, position-feasible teams to simulate. It
is not an auction algorithm, it does not price anything, and the CE numbers it
produces are a property of an arbitrary allocation rather than advice.

What it actually checks is that the engine's invariants survive contact with
real inputs: legal eight-player lineups, a non-QB superflex, sixteen scheduled
games inside a seventeen-week horizon, no week-18 contribution, unavailable
players scoring zero, one champion per season, determinism and chunk
invariance -- and that no synthetic player, vendor fantasy total or expert
grade reached the specs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..league import DEFAULT_LEAGUE, LeagueSettings, Position
from ..players import PlayerSpec
from ..roster import Roster, RosterSet

__all__ = ["ROSTER_TEMPLATE", "build_test_rosters", "roster_assignment",
           "rosters_from_assignment", "SmokeChecks", "run_smoke_checks"]

#: Position counts per team. Guarantees eight legal starters are always
#: available: 1 QB + 2 RB + 3 WR/TE + FLEX + SUPERFLEX.
ROSTER_TEMPLATE: Dict[str, int] = {"QB": 3, "RB": 4, "WR": 6, "TE": 2}


def build_test_rosters(specs: Sequence[PlayerSpec],
                       settings: LeagueSettings = DEFAULT_LEAGUE,
                       team_names: Optional[Sequence[str]] = None) -> RosterSet:
    """Twelve deterministic, legal, disjoint rosters from the real pool.

    A per-position snake: players are ranked by ``base_mean`` within position
    and dealt out in alternating order, so the twelve teams end up close in
    strength without any of them being hand-tuned. Ties break on ``player_id``,
    so the allocation is a pure function of the pool.

    Raises if the pool cannot fill the template -- silently shrinking a roster
    would make every downstream invariant check meaningless.
    """
    n_teams = settings.n_teams
    by_pos: Dict[str, List[PlayerSpec]] = {p: [] for p in ROSTER_TEMPLATE}
    for spec in specs:
        name = Position(int(spec.position)).name
        if name in by_pos:
            by_pos[name].append(spec)
    for name in by_pos:
        by_pos[name].sort(key=lambda s: (-s.base_mean, s.player_id))

    for name, per_team in ROSTER_TEMPLATE.items():
        need = per_team * n_teams
        if len(by_pos[name]) < need:
            raise ValueError(
                f"pool has {len(by_pos[name])} {name}s but {need} are needed "
                f"for {n_teams} teams; refusing to build short rosters")

    picks: List[List[int]] = [[] for _ in range(n_teams)]
    used: List[PlayerSpec] = []
    for pos_i, (name, per_team) in enumerate(sorted(ROSTER_TEMPLATE.items())):
        pool = by_pos[name]
        r = 0
        for rnd in range(per_team):
            order = range(n_teams) if rnd % 2 == 0 else range(n_teams - 1, -1, -1)
            for t in order:
                # Offset per position so no team collects every positional best.
                team = (t + pos_i * 5) % n_teams
                spec = pool[r]
                picks[team].append(spec.player_id)
                used.append(spec)
                r += 1

    names = list(team_names) if team_names else [f"Real{i + 1:02d}"
                                                 for i in range(n_teams)]
    rosters = tuple(Roster(names[t], tuple(picks[t])) for t in range(n_teams))
    return RosterSet(tuple(used), rosters, settings)


def roster_assignment(rosters: RosterSet) -> Tuple[Tuple[int, ...], ...]:
    """The twelve teams as tuples of ``player_id``, stripped of everything else.

    This is the object a sensitivity sweep must hold fixed. Rebuilding the
    snake under each scenario would let the allocation itself move whenever an
    assumption changed ``base_mean``, so the "paired" comparison would be
    between two different sets of teams and the difference attributed to the
    assumption would partly be a difference in who was on which roster.
    """
    return tuple(tuple(r.player_ids) for r in rosters.rosters)


def rosters_from_assignment(assignment: Sequence[Sequence[int]],
                            specs: Sequence[PlayerSpec],
                            settings: LeagueSettings = DEFAULT_LEAGUE,
                            team_names: Optional[Sequence[str]] = None
                            ) -> RosterSet:
    """Rebuild a fixed roster assignment against a different set of specs.

    ``specs`` must contain every ``player_id`` the assignment names -- which is
    exactly what stable, key-derived ids guarantee across scenarios. A missing
    id raises: quietly substituting anyone would break the pairing this
    function exists to preserve.
    """
    by_id = {s.player_id: s for s in specs}
    missing = sorted({pid for team in assignment for pid in team} - set(by_id))
    if missing:
        raise KeyError(
            f"{len(missing)} player_id(s) in the fixed assignment are absent "
            f"from this scenario's specs; a paired comparison needs the same "
            f"people in both arms (first few: {missing[:5]})")
    names = list(team_names) if team_names else [
        f"Real{i + 1:02d}" for i in range(len(assignment))]
    used = [by_id[pid] for team in assignment for pid in team]
    rosters = tuple(Roster(names[t], tuple(team))
                    for t, team in enumerate(assignment))
    return RosterSet(tuple(used), rosters, settings)


@dataclass
class SmokeChecks:
    """Every invariant, and whether it held."""

    results: Dict[str, bool]
    details: Dict[str, object]

    @property
    def ok(self) -> bool:
        return all(self.results.values())

    def failures(self) -> List[str]:
        return [k for k, v in self.results.items() if not v]

    def lines(self) -> List[str]:
        out = []
        for name, passed in self.results.items():
            mark = "PASS" if passed else "FAIL"
            extra = self.details.get(name, "")
            out.append(f"  [{mark}] {name}" + (f"  ({extra})" if extra else ""))
        return out


def run_smoke_checks(rosters: RosterSet, specs: Sequence[PlayerSpec],
                     n_sims: int = 400, seed: int = 20260904) -> SmokeChecks:
    """Run the engine on real specs and assert the invariants that matter."""
    from ..lineup_vec import select_lineups_mask
    from ..simulate import simulate_seasons, team_scores
    from ..worlds import build_pool_arrays, generate_world

    settings = rosters.settings
    results: Dict[str, bool] = {}
    details: Dict[str, object] = {}

    pool = build_pool_arrays(rosters.pool, settings)
    world = generate_world(pool, seed, 0, min(n_sims, 64))
    rm = rosters.roster_matrix()
    proj = np.moveaxis(world.pregame.projection[:, rm, :], 2, -1)
    avail = np.moveaxis(world.availability.available[:, rm, :], 2, -1)
    pos = np.broadcast_to(world.pool.position[rm][None, :, None, :], proj.shape)
    mask = select_lineups_mask(proj, avail, pos)

    positions = world.pool.position[rm]
    started_pos = np.where(mask, positions[None, :, None, :], -1)
    q = (started_pos == int(Position.QB)).sum(axis=-1)
    r = (started_pos == int(Position.RB)).sum(axis=-1)
    t = ((started_pos == int(Position.WR)) |
         (started_pos == int(Position.TE))).sum(axis=-1)
    legal = ((q <= 2) & (r <= 4) & (t <= 5) & (q + r <= 5) & (q + t <= 6)
             & (r + t <= 7) & (q + r + t <= 8))
    results["hall_condition_holds"] = bool(legal.all())

    # --- lineups are legal and MAXIMAL --------------------------------------
    # Not "always eight". Real byes cluster: several of a team's players can
    # share a bye week, and a week where fewer than eight are startable is a
    # correct outcome, not a defect. What must hold is that the lineup is as
    # full as availability and the slot rules allow -- no available player
    # could legally have been added.
    filled = mask.sum(axis=-1)
    n_avail = avail.sum(axis=-1)
    results["lineups_never_exceed_eight"] = bool((filled <= 8).all())
    results["only_available_players_start"] = bool(not (mask & ~avail).any())

    maximal = True
    for pos_code, (dq, dr, dt) in ((int(Position.QB), (1, 0, 0)),
                                   (int(Position.RB), (0, 1, 0)),
                                   (int(Position.WR), (0, 0, 1)),
                                   (int(Position.TE), (0, 0, 1))):
        addable = avail & ~mask & (positions[None, :, None, :] == pos_code)
        nq, nr, nt = q + dq, r + dr, t + dt
        fits = ((nq <= 2) & (nr <= 4) & (nt <= 5) & (nq + nr <= 5)
                & (nq + nt <= 6) & (nr + nt <= 7) & (nq + nr + nt <= 8))
        # A benched available player who would still fit means the lineup was
        # not maximal.
        if bool((addable.any(axis=-1) & fits).any()):
            maximal = False
    results["lineups_are_maximal"] = maximal

    full = filled == 8
    results["eight_slots_filled_when_possible"] = bool(
        (filled[n_avail >= 8] >= 8).all() or maximal)
    details["eight_slots_filled_when_possible"] = (
        f"{100.0 * float(full.mean()):.1f}% of team-weeks fill all eight; "
        f"min {int(filled.min())} when byes cluster")

    # --- a non-QB may fill the superflex ------------------------------------
    # With eight filled and at most two QBs, any team-week starting fewer than
    # two QBs has a non-QB in the superflex.
    results["non_qb_superflex_allowed"] = bool((q < 2).any())
    details["non_qb_superflex_allowed"] = (
        f"{100.0 * float((q < 2).mean()):.1f}% of team-weeks")

    # --- byes: 16 scheduled games in a 17-week horizon ----------------------
    on_bye = world.availability.on_bye
    with_bye = [i for i in range(pool.n_players) if pool.bye_index[i] >= 0]
    bye_counts = on_bye[:, with_bye, :].sum(axis=2)
    results["one_bye_per_player_with_a_bye"] = bool((bye_counts == 1).all())
    scheduled = settings.total_weeks - 1
    results["sixteen_scheduled_games_in_horizon"] = scheduled == 16
    details["sixteen_scheduled_games_in_horizon"] = (
        f"{settings.total_weeks} weeks minus one bye = {scheduled}")

    # --- week 18 cannot contribute ------------------------------------------
    results["no_week_18"] = (world.realized.points.shape[2] ==
                             settings.total_weeks == 17)
    details["no_week_18"] = f"array has {world.realized.points.shape[2]} weeks"

    # --- unavailable players score exactly zero -----------------------------
    unavailable = ~world.availability.available
    results["unavailable_players_score_zero"] = bool(
        np.all(world.realized.points[unavailable] == 0.0))

    # --- one champion per season, determinism, chunk invariance -------------
    out_a = simulate_seasons(rosters, n_sims, seed)
    out_b = simulate_seasons(rosters, n_sims, seed)
    out_c = simulate_seasons(rosters, n_sims, seed, chunk=7)
    rows = np.arange(out_a.n_sims)
    results["exactly_one_champion_per_season"] = bool(
        out_a.made_playoffs[rows, out_a.champion].all()
        and out_a.champion.shape == (n_sims,))
    results["deterministic"] = bool(np.array_equal(out_a.champion, out_b.champion)
                                    and np.array_equal(out_a.points, out_b.points))
    results["chunk_invariant"] = bool(np.array_equal(out_a.champion, out_c.champion)
                                      and np.array_equal(out_a.points, out_c.points))
    ce = out_a.championship_equity()
    results["ce_is_a_distribution"] = bool(abs(ce.sum() - 1.0) < 1e-9
                                           and (ce >= 0).all())
    details["ce_is_a_distribution"] = f"sum {ce.sum():.6f}"

    # --- provenance: nothing synthetic, no vendor total, no expert grade -----
    sources = {s.data_source for s in specs}
    results["no_synthetic_players"] = not any(
        s.upper().startswith("SYNTHETIC") for s in sources)
    details["no_synthetic_players"] = f"sources {sorted(sources)}"
    results["all_specs_marked_real"] = all(s.startswith("REAL:") for s in sources)

    # base_mean must be a calibrated level, never a raw vendor season total.
    means = np.array([s.base_mean for s in specs], dtype=float)
    results["base_mean_is_a_per_game_level"] = bool(means.max() < 60.0)
    details["base_mean_is_a_per_game_level"] = f"max {means.max():.2f}"

    # No spec may carry a distribution parameter that only a grade could supply.
    results["no_grade_derived_distribution"] = all(
        s.spike_rate == 0.0 and s.spike_scale == 0.0 and s.role_change_prob == 0.0
        and s.shock_loadings == () and s.contingency is None
        for s in specs)
    return SmokeChecks(results=results, details=details)
