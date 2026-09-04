"""Deterministic builders shared by the test suite."""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from ceauction.league import DEFAULT_LEAGUE, Position
from ceauction.players import PlayerSpec
from ceauction.pregame import Availability, PregameEntry, PregameWeek
from ceauction.roster import Roster, RosterSet

SEED = 4242


def entry(pid, pos, proj, status=Availability.ACTIVE, name=None):
    return PregameEntry(
        player_id=pid,
        name=name or f"P{pid}",
        position=pos,
        projection=proj,
        availability=status,
    )


def week_of(entries, week=1):
    return PregameWeek(week=week, entries=tuple(entries))


def flat_spec(pid: int, pos: Position, mean: float, **kw) -> PlayerSpec:
    """A totally inert player: no variance, no injuries, no correlation.

    Used to build worlds whose arithmetic can be checked by hand.
    """
    params = dict(
        week_sd=0.0,
        season_sd=0.0,
        bye_week=0,
        weekly_injury_hazard=0.0,
        spike_rate=0.0,
        spike_scale=0.0,
        role_change_prob=0.0,
        proj_noise_sd=0.0,
        shock_loadings=(),
        contingency=None,
        nfl_team="TST",
        data_source="SYNTHETIC",
    )
    params.update(kw)
    return PlayerSpec(
        player_id=pid, name=f"T{pid}", position=pos, base_mean=mean, **params
    )


def flat_league(means_per_team: Optional[List[List[float]]] = None) -> RosterSet:
    """A 12-team league of inert players with a legal 3QB/4RB/6WR/2TE shape.

    Every player scores exactly ``base_mean`` every week, so team scores,
    standings and brackets are fully determined and hand-checkable.
    """
    template = (
        [Position.QB] * 3 + [Position.RB] * 4 + [Position.WR] * 6 + [Position.TE] * 2
    )
    specs: List[PlayerSpec] = []
    rosters = []
    pid = 0
    for t in range(DEFAULT_LEAGUE.n_teams):
        ids = []
        for k, pos in enumerate(template):
            mean = (
                means_per_team[t][k]
                if means_per_team is not None
                else 10.0 + t * 0.5 + k * 0.01
            )
            specs.append(flat_spec(pid, pos, mean))
            ids.append(pid)
            pid += 1
        rosters.append(Roster(f"F{t:02d}", tuple(ids)))
    return RosterSet(tuple(specs), tuple(rosters), DEFAULT_LEAGUE)
