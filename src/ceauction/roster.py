"""Rosters and the league-wide roster set."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .league import DEFAULT_LEAGUE, LeagueSettings, Position
from .players import PlayerSpec

__all__ = ["Roster", "RosterSet"]


@dataclass(frozen=True)
class Roster:
    """One team's 15 drafted players.

    A roster is a *portfolio supporting eight weekly lineup spots*, not eight
    starters plus seven spares.  Nothing here labels any player a starter; the
    lineup optimiser re-decides every week from that week's information.
    """

    team_name: str
    player_ids: Tuple[int, ...]

    def __len__(self) -> int:
        return len(self.player_ids)

    def replaced(self, old_player_id: int, new_player_id: int) -> "Roster":
        """Copy with one player swapped, preserving order.

        Order preservation matters: it keeps the two scenarios of a paired
        comparison aligned slot-for-slot.
        """
        if old_player_id not in self.player_ids:
            raise KeyError(f"{old_player_id} not on {self.team_name}")
        if new_player_id in self.player_ids and new_player_id != old_player_id:
            raise ValueError(f"{new_player_id} is already on {self.team_name}")
        ids = tuple(new_player_id if p == old_player_id else p for p in self.player_ids)
        return Roster(self.team_name, ids)


@dataclass(frozen=True)
class RosterSet:
    """The full league: 12 rosters drawn from one player pool.

    Holds the pool as an ordered list of :class:`PlayerSpec` plus an index
    matrix ``(n_teams, roster_size)`` of pool positions, which is the form the
    vectorised engine consumes.
    """

    pool: Tuple[PlayerSpec, ...]
    rosters: Tuple[Roster, ...]
    settings: LeagueSettings = DEFAULT_LEAGUE

    def __post_init__(self) -> None:
        s = self.settings
        if len(self.rosters) != s.n_teams:
            raise ValueError(f"expected {s.n_teams} rosters, got {len(self.rosters)}")
        ids = [p.player_id for p in self.pool]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate player_id in pool")
        index = self.id_to_index
        seen: Dict[int, str] = {}
        for r in self.rosters:
            if len(r) != s.roster_size:
                raise ValueError(
                    f"{r.team_name} has {len(r)} players, expected {s.roster_size}"
                )
            if len(set(r.player_ids)) != len(r.player_ids):
                raise ValueError(f"{r.team_name} has a duplicate player")
            for pid in r.player_ids:
                if pid not in index:
                    raise KeyError(f"player {pid} on {r.team_name} is not in the pool")
                if pid in seen:
                    raise ValueError(
                        f"player {pid} is on both {seen[pid]} and {r.team_name}"
                    )
                seen[pid] = r.team_name

    # --- lookups -------------------------------------------------------------

    @property
    def id_to_index(self) -> Dict[int, int]:
        return {p.player_id: i for i, p in enumerate(self.pool)}

    def spec(self, player_id: int) -> PlayerSpec:
        return self.pool[self.id_to_index[player_id]]

    @property
    def team_names(self) -> Tuple[str, ...]:
        return tuple(r.team_name for r in self.rosters)

    def team_index(self, team_name: str) -> int:
        for i, r in enumerate(self.rosters):
            if r.team_name == team_name:
                return i
        raise KeyError(team_name)

    # --- vectorised views ----------------------------------------------------

    def roster_matrix(self) -> np.ndarray:
        """``(n_teams, roster_size)`` pool indices."""
        index = self.id_to_index
        return np.array(
            [[index[p] for p in r.player_ids] for r in self.rosters], dtype=np.int64
        )

    def position_matrix(self) -> np.ndarray:
        """``(n_teams, roster_size)`` position codes."""
        index = self.id_to_index
        return np.array(
            [[int(self.pool[index[p]].position) for p in r.player_ids] for r in self.rosters],
            dtype=np.int8,
        )

    # --- editing -------------------------------------------------------------

    def with_pool_player(self, spec: PlayerSpec) -> "RosterSet":
        """Replace (or append) a pool entry by ``player_id``, keeping rosters."""
        index = self.id_to_index
        pool = list(self.pool)
        if spec.player_id in index:
            pool[index[spec.player_id]] = spec
        else:
            pool.append(spec)
        return RosterSet(tuple(pool), self.rosters, self.settings)

    def with_roster(self, team_index: int, roster: Roster) -> "RosterSet":
        rosters = list(self.rosters)
        rosters[team_index] = roster
        return RosterSet(self.pool, tuple(rosters), self.settings)

    def position_counts(self, team_index: int) -> Dict[Position, int]:
        counts = {p: 0 for p in Position}
        for pid in self.rosters[team_index].player_ids:
            counts[self.spec(pid).position] += 1
        return counts

    def describe(self, team_index: int) -> List[str]:
        lines = []
        for pid in self.rosters[team_index].player_ids:
            s = self.spec(pid)
            lines.append(
                f"  {s.position.label:<3} {s.name:<22} base={s.base_mean:5.1f} "
                f"sd={s.week_sd:4.1f} inj={s.weekly_injury_hazard:.3f} bye={s.bye_week}"
            )
        return lines
