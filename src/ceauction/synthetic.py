"""SYNTHETIC player generator.  NOT REAL FOOTBALL DATA.

======================================================================
   EVERY NUMBER IN THIS MODULE IS INVENTED.  It is chosen to be
   plausible in *shape* (QBs score more and vary less than RBs; RBs get
   hurt more often than WRs; replacement level is flat) so that the CE
   engine can be exercised and its behaviour inspected.  None of it is
   an estimate of any real player, and no fantasy-football conclusion
   should be drawn from any output produced from it.

   Every spec emitted here carries ``data_source="SYNTHETIC"``.
======================================================================

This module exists to demonstrate the eight properties the engine must
support, all of which are expressed purely through ``PlayerSpec`` fields:

1. persistent season-level performance states   -> ``season_sd``
2. pregame projections                          -> filtered posterior + noise
3. weekly scoring variance                      -> ``week_sd``
4. injuries and unavailable weeks               -> hazard/duration + byes
5. observable role changes                      -> ``role_change_*``
6. unforecastable spike weeks                   -> ``spike_rate``/``spike_scale``
7. shared team-level shocks                     -> ``shock_loadings`` (``team:*``)
8. hooks for player correlations                -> ``shock_loadings`` (any group)

Replacing this module with real data requires **no change to the engine**:
populate the same ``PlayerSpec`` fields from real medians, real injury
probabilities, real boom/bust widths, real published weekly projections and
real correlation relationships.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .league import DEFAULT_LEAGUE, LeagueSettings, Position
from .players import Contingency, PlayerSpec, ShockLoading
from .roster import Roster, RosterSet

__all__ = [
    "PositionProfile",
    "SyntheticConfig",
    "DEFAULT_PROFILES",
    "make_synthetic_pool",
    "make_synthetic_league",
    "make_identical_league",
    "ROSTER_TEMPLATE",
]

#: Roster construction used by the synthetic league: 3 QB / 4 RB / 6 WR / 2 TE.
#: Any 8 legal starters are always available from this shape.
ROSTER_TEMPLATE: Dict[Position, int] = {
    Position.QB: 3,
    Position.RB: 4,
    Position.WR: 6,
    Position.TE: 2,
}

N_NFL_TEAMS = 32
_NFL_ABBRS = tuple(f"T{i:02d}" for i in range(N_NFL_TEAMS))


@dataclass(frozen=True)
class PositionProfile:
    """SYNTHETIC per-position shape parameters.

    ``base_mean`` for the player ranked *r* (0-indexed) at this position is
    ``floor + (top - floor) * exp(-decay * r)``.
    """

    top: float
    floor: float
    decay: float
    week_sd: float
    season_sd: float
    injury_hazard: float
    injury_mean_weeks: float
    spike_rate: float
    spike_scale: float
    role_change_prob: float
    role_change_mean: float
    role_change_sd: float
    proj_noise_sd: float
    team_beta: float
    stack_beta: float


#: SYNTHETIC. Half-PPR shaped: QB highest mean and lowest relative variance,
#: RB highest injury hazard, TE lowest ceiling.
DEFAULT_PROFILES: Dict[Position, PositionProfile] = {
    Position.QB: PositionProfile(
        top=22.0, floor=9.0, decay=0.055, week_sd=6.0, season_sd=2.2,
        injury_hazard=0.020, injury_mean_weeks=2.0,
        spike_rate=0.08, spike_scale=6.0,
        role_change_prob=0.10, role_change_mean=0.0, role_change_sd=3.0,
        proj_noise_sd=1.0, team_beta=1.3, stack_beta=1.4,
    ),
    Position.RB: PositionProfile(
        top=17.0, floor=4.0, decay=0.070, week_sd=6.8, season_sd=2.6,
        injury_hazard=0.045, injury_mean_weeks=2.6,
        spike_rate=0.10, spike_scale=7.0,
        role_change_prob=0.18, role_change_mean=0.3, role_change_sd=3.2,
        proj_noise_sd=1.2, team_beta=0.7, stack_beta=0.0,
    ),
    Position.WR: PositionProfile(
        top=16.5, floor=4.0, decay=0.048, week_sd=7.2, season_sd=2.5,
        injury_hazard=0.030, injury_mean_weeks=2.2,
        spike_rate=0.12, spike_scale=7.5,
        role_change_prob=0.15, role_change_mean=0.2, role_change_sd=3.0,
        proj_noise_sd=1.3, team_beta=0.9, stack_beta=1.1,
    ),
    Position.TE: PositionProfile(
        top=13.0, floor=3.0, decay=0.090, week_sd=5.8, season_sd=2.0,
        injury_hazard=0.030, injury_mean_weeks=2.2,
        spike_rate=0.10, spike_scale=6.0,
        role_change_prob=0.15, role_change_mean=0.2, role_change_sd=2.6,
        proj_noise_sd=1.1, team_beta=0.7, stack_beta=0.9,
    ),
}


@dataclass(frozen=True)
class SyntheticConfig:
    """Knobs for the SYNTHETIC pool."""

    settings: LeagueSettings = DEFAULT_LEAGUE
    profiles: Dict[Position, PositionProfile] = field(
        default_factory=lambda: dict(DEFAULT_PROFILES)
    )
    template: Dict[Position, int] = field(default_factory=lambda: dict(ROSTER_TEMPLATE))
    enable_team_shocks: bool = True
    enable_stacks: bool = True
    enable_contingency: bool = True
    contingency_bonus: float = 5.0
    first_bye_week: int = 5
    n_bye_weeks: int = 10

    def counts(self) -> Dict[Position, int]:
        return {p: n * self.settings.n_teams for p, n in self.template.items()}


def _base_mean(profile: PositionProfile, rank: int) -> float:
    return profile.floor + (profile.top - profile.floor) * float(np.exp(-profile.decay * rank))


def make_synthetic_pool(config: Optional[SyntheticConfig] = None) -> List[PlayerSpec]:
    """Build the SYNTHETIC player pool: ``n_teams * roster_size`` players.

    Player ids are ``position_code * 1000 + rank`` so that they are stable and
    readable, and so that a laboratory variant can deliberately reuse a
    ``crn_key``.
    """
    config = config or SyntheticConfig()
    counts = config.counts()
    specs: List[PlayerSpec] = []

    # NFL-team assignment is round-robin within position, so an NFL team's
    # players span a range of ranks and team quality is not degenerate.
    for pos in (Position.QB, Position.RB, Position.WR, Position.TE):
        prof = config.profiles[pos]
        for rank in range(counts[pos]):
            nfl_i = rank % N_NFL_TEAMS
            abbr = _NFL_ABBRS[nfl_i]
            loadings: List[ShockLoading] = []
            if config.enable_team_shocks:
                loadings.append(ShockLoading(f"team:{abbr}", prof.team_beta))
            if config.enable_stacks and prof.stack_beta > 0.0:
                # Only the first QB / first WR / first TE on each NFL team join
                # the passing-game stack, which is where the real correlation
                # lives.
                if rank // N_NFL_TEAMS == 0:
                    loadings.append(ShockLoading(f"stack:{abbr}", prof.stack_beta))

            contingency = None
            if (
                config.enable_contingency
                and pos is Position.RB
                and rank >= N_NFL_TEAMS
                and rank - N_NFL_TEAMS < counts[Position.RB] - N_NFL_TEAMS
            ):
                # The second RB on an NFL team is the first one's handcuff.
                contingency = Contingency(
                    on_player_id=int(Position.RB) * 1000 + (rank - N_NFL_TEAMS),
                    bonus=config.contingency_bonus,
                )

            specs.append(
                PlayerSpec(
                    player_id=int(pos) * 1000 + rank,
                    name=f"SYN-{pos.label}{rank + 1:03d}",
                    position=pos,
                    nfl_team=abbr,
                    bye_week=config.first_bye_week + (nfl_i % config.n_bye_weeks),
                    base_mean=_base_mean(prof, rank),
                    week_sd=prof.week_sd,
                    season_sd=prof.season_sd,
                    weekly_injury_hazard=prof.injury_hazard,
                    injury_mean_weeks=prof.injury_mean_weeks,
                    spike_rate=prof.spike_rate,
                    spike_scale=prof.spike_scale,
                    role_change_prob=prof.role_change_prob,
                    role_change_mean=prof.role_change_mean,
                    role_change_sd=prof.role_change_sd,
                    role_reveal_lag=1,
                    shock_loadings=tuple(loadings),
                    contingency=contingency,
                    proj_noise_sd=prof.proj_noise_sd,
                    data_source="SYNTHETIC",
                    notes=f"synthetic {pos.label} rank {rank + 1}",
                )
            )
    return specs


def _snake_assign(n_teams: int, n_per_team: int, offset: int = 0) -> List[List[int]]:
    """Snake order: team lists of pool ranks, so teams are near-equal.

    ``offset`` rotates which team picks first.  Each position uses a different
    offset so that no single team collects the best player at every position.
    """
    out: List[List[int]] = [[] for _ in range(n_teams)]
    r = 0
    for rnd in range(n_per_team):
        base = list(range(n_teams)) if rnd % 2 == 0 else list(range(n_teams - 1, -1, -1))
        for t in base:
            out[(t + offset) % n_teams].append(r)
            r += 1
    return out


def make_synthetic_league(
    config: Optional[SyntheticConfig] = None,
    team_names: Optional[Sequence[str]] = None,
) -> RosterSet:
    """A full 12-team SYNTHETIC league built by a per-position snake draft.

    Teams end up close in strength but not identical, which is the useful
    baseline for laboratory experiments: a change to one team's roster has a
    measurable but not overwhelming effect.
    """
    config = config or SyntheticConfig()
    pool = make_synthetic_pool(config)
    settings = config.settings
    n_teams = settings.n_teams
    by_pos: Dict[Position, List[PlayerSpec]] = {p: [] for p in Position}
    for s in pool:
        by_pos[s.position].append(s)

    picks: List[List[int]] = [[] for _ in range(n_teams)]
    for pos_i, (pos, per_team) in enumerate(sorted(config.template.items())):
        assign = _snake_assign(n_teams, per_team, offset=(pos_i * 5) % n_teams)
        for t in range(n_teams):
            for rank in assign[t]:
                picks[t].append(by_pos[pos][rank].player_id)

    names = list(team_names) if team_names else [f"Team{i + 1:02d}" for i in range(n_teams)]
    rosters = tuple(Roster(names[t], tuple(picks[t])) for t in range(n_teams))
    return RosterSet(tuple(pool), rosters, settings)


def make_identical_league(
    config: Optional[SyntheticConfig] = None,
    template_rank: int = 6,
) -> RosterSet:
    """12 statistically identical teams (distinct players, identical parameters).

    Used by ``tests/test_ce.py`` to assert that CE is ~1/12 for everyone.  The
    players are distinct ids so their draws are independent; only their
    *parameters* match.  Team shocks, stacks and contingencies are switched off
    so that no team can be advantaged by correlation structure.
    """
    config = config or SyntheticConfig(
        enable_team_shocks=False, enable_stacks=False, enable_contingency=False
    )
    settings = config.settings
    n_teams = settings.n_teams
    counts = config.template

    specs: List[PlayerSpec] = []
    picks: List[List[int]] = [[] for _ in range(n_teams)]
    pid = 0
    for pos, per_team in counts.items():
        prof = config.profiles[pos]
        for slot in range(per_team):
            # Every team's k-th player at this position has identical params.
            mean = _base_mean(prof, template_rank + slot * n_teams // 2)
            for t in range(n_teams):
                specs.append(
                    PlayerSpec(
                        player_id=pid,
                        name=f"SYN-IDENT-{pos.label}{slot}-T{t:02d}",
                        position=pos,
                        nfl_team=f"N{pid:03d}",  # a private NFL team each: no shared byes
                        bye_week=config.first_bye_week + (slot % config.n_bye_weeks),
                        base_mean=mean,
                        week_sd=prof.week_sd,
                        season_sd=prof.season_sd,
                        weekly_injury_hazard=prof.injury_hazard,
                        injury_mean_weeks=prof.injury_mean_weeks,
                        spike_rate=prof.spike_rate,
                        spike_scale=prof.spike_scale,
                        role_change_prob=prof.role_change_prob,
                        role_change_mean=prof.role_change_mean,
                        role_change_sd=prof.role_change_sd,
                        proj_noise_sd=prof.proj_noise_sd,
                        data_source="SYNTHETIC",
                        notes="identical-teams control",
                    )
                )
                picks[t].append(pid)
                pid += 1

    rosters = tuple(
        Roster(f"Ident{t + 1:02d}", tuple(picks[t])) for t in range(n_teams)
    )
    return RosterSet(tuple(specs), rosters, settings)
