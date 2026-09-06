"""A fabricated auction, so every command has something honest to run on.

**Nothing here describes a real player.** The pool is generated from four
smooth positional curves, the prices come from a straight-line function of
projection, and both are labelled ``FABRICATED`` everywhere they surface. The
point is to exercise the machinery and to give the committed examples something
reproducible to show, not to say anything about anybody.

The shape is a realistic mid-auction: eleven opponents have most of a roster
and still have money and a slot or two left, so a pass branch can genuinely
award a candidate to one of them. The focus team has three players and the rest
of its budget.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from ..league import DEFAULT_LEAGUE, LeagueSettings, Position
from ..players import PlayerSpec
from ..realdata.smoke import build_test_rosters, roster_assignment
from .completion import ComparisonCast
from .costs import CostBook, CostEntry, CostProvenance
from .state import AuctionState, new_auction

__all__ = ["DemoAuction", "build_demo_auction", "DEMO_OWNERS"]

DEMO_OWNERS: Tuple[str, ...] = tuple(f"Owner{i + 1:02d}" for i in range(12))

#: ``(position, count, top projection, decline per rank)``. Four different
#: slopes so no position is uniformly the best thing left on the board, which
#: is the degenerate case a flat curve produces.
_CURVES = ((Position.QB, 40, 19.0, 0.35), (Position.RB, 90, 16.0, 0.15),
           (Position.WR, 120, 15.0, 0.11), (Position.TE, 60, 12.0, 0.16))


@dataclass(frozen=True)
class DemoAuction:
    """Everything a command needs, all of it fabricated."""

    state: AuctionState
    cast: ComparisonCast
    costs: CostBook
    pool: Tuple[PlayerSpec, ...]

    @property
    def focus_owner_id(self) -> str:
        return self.state.focus_owner_id

    def top_available(self, n: int = 1) -> List[PlayerSpec]:
        return sorted(self.state.available_specs,
                      key=lambda s: (-s.base_mean, s.player_id))[:n]

    def default_candidate(self) -> PlayerSpec:
        """The best player still on the board. Deterministic."""
        top = self.top_available(1)
        if not top:
            raise ValueError("the demo board is empty")
        return top[0]


def build_demo_auction(seed_shift: int = 0,
                       focus_keep: int = 3,
                       rival_keep: int = 12,
                       settings: LeagueSettings = DEFAULT_LEAGUE) -> DemoAuction:
    """A deterministic fabricated mid-auction state.

    ``rival_keep`` below the roster size is what leaves opponents able to bid;
    an auction where everyone is full has no counterfactual worth running.
    """
    pool: List[PlayerSpec] = []
    pid = 0
    for pos, n, top, decline in _CURVES:
        for k in range(n):
            pool.append(PlayerSpec(
                player_id=pid + seed_shift, name=f"Fabricated{pid:04d}",
                position=pos, nfl_team="ZZA",
                base_mean=max(top - decline * k, 2.5), week_sd=6.0,
                bye_week=5 + (k % 10), weekly_injury_hazard=0.03,
                injury_mean_weeks=2.5,
                data_source="FABRICATED:demo-curve"))
            pid += 1

    assignment = roster_assignment(build_test_rosters(pool, settings=settings))
    state = new_auction(pool, DEMO_OWNERS, DEMO_OWNERS[0], settings=settings)
    for i, team in enumerate(assignment):
        keep = focus_keep if i == 0 else rival_keep
        for j, player_id in enumerate(team[:keep]):
            price = (40, 25, 25)[j] if i == 0 else 9
            state = state.apply_purchase(player_id, DEMO_OWNERS[i], price)
    state.validate()

    cast = ComparisonCast(0, assignment, DEMO_OWNERS)
    costs = CostBook(
        entries=tuple(CostEntry(s.player_id,
                                max(1, int(round(s.base_mean * 2.0 - 12))))
                      for s in pool),
        provenance=CostProvenance(
            level="FABRICATED",
            source="ceauction.auction.demo straight-line curve",
            notes="cost = 2 * projection - 12, floored at $1. Not a market."))
    return DemoAuction(state=state, cast=cast, costs=costs, pool=tuple(pool))
