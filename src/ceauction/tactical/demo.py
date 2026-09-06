"""One coherent fabricated world: an auction room and a market that agree.

The auction demo and the market demo each generate their own fabricated board,
and they do not share players. Tactical work needs them joined -- a candidate
needs both an auction ``player_id`` and a market ``canonical_key`` -- so this
module builds a single world in which the two line up.

**No real player appears anywhere here.** Names are ``Fabricated0001``, prices
come from a straight-line function of a made-up projection, and every command
that prints them says so.

Only the top ``n_anchored`` players get a Sleeper anchor. That is deliberate:
the remainder exercise the missing-anchor path, which must produce a wide
scenario range from the cost-book fallback and must never produce a zero
willingness.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, List, Optional, Tuple

from ..auction.completion import ComparisonCast
from ..auction.costs import CostBook
from ..auction.demo import DEMO_OWNERS, DemoAuction, build_demo_auction
from ..auction.state import AuctionState
from ..league import DEFAULT_LEAGUE, LeagueSettings, Position
from ..market.anchors import AnchorBook, SleeperAnchor
from ..market.live import MarketState, SaleObservation
from ..market.prior import MarketPrior, build_market_prior
from ..players import PlayerSpec

__all__ = ["TacticalDemo", "build_tactical_demo", "demo_sales"]


@dataclass(frozen=True)
class TacticalDemo:
    """A fabricated auction and a fabricated market that describe one board."""

    auction: DemoAuction
    prior: MarketPrior
    market: MarketState
    key_by_id: Dict[int, str]
    id_by_key: Dict[str, int]

    @property
    def state(self) -> AuctionState:
        return self.auction.state

    @property
    def cast(self) -> ComparisonCast:
        return self.auction.cast

    @property
    def costs(self) -> CostBook:
        return self.auction.costs

    @property
    def focus_owner_id(self) -> str:
        return self.auction.focus_owner_id

    def key_for(self, player_id: int) -> Optional[str]:
        return self.key_by_id.get(player_id)

    def with_state(self, state: AuctionState) -> "TacticalDemo":
        import dataclasses
        return dataclasses.replace(
            self, auction=dataclasses.replace(self.auction, state=state))

    def with_market(self, market: MarketState) -> "TacticalDemo":
        import dataclasses
        return dataclasses.replace(self, market=market)

    def candidate(self, rank: int = 0) -> PlayerSpec:
        """The ``rank``-th best player still on the board. Deterministic."""
        avail = sorted(self.state.available_specs,
                       key=lambda s: (-s.base_mean, s.player_id))
        if rank >= len(avail):
            raise IndexError("the fabricated board is not that deep")
        return avail[rank]

    def candidate_at(self, position: Position, rank: int = 0) -> PlayerSpec:
        avail = sorted((s for s in self.state.available_specs
                        if Position(int(s.position)) is position),
                       key=lambda s: (-s.base_mean, s.player_id))
        if rank >= len(avail):
            raise IndexError(f"not that many {position.name}s left")
        return avail[rank]


def build_tactical_demo(*, focus_keep: int = 3, rival_keep: int = 12,
                        n_anchored: int = 170,
                        settings: LeagueSettings = DEFAULT_LEAGUE
                        ) -> TacticalDemo:
    """A fabricated mid-auction with a market prior built from the same pool."""
    auction = build_demo_auction(focus_keep=focus_keep, rival_keep=rival_keep,
                                 settings=settings)
    ranked = sorted(auction.pool, key=lambda s: (-s.base_mean, s.player_id))
    anchored = ranked[:n_anchored]

    anchors: List[SleeperAnchor] = []
    key_by_id: Dict[int, str] = {}
    for s in anchored:
        pos = Position(int(s.position)).name
        # A straight-line map from the fabricated projection to a fabricated
        # dollar value. Not a market and not a model of one.
        raw = max(0.4, round(s.base_mean * 3.1 - 12.0, 2))
        a = SleeperAnchor(sleeper_player_id=f"8{s.player_id:05d}",
                          player_name=s.name, position=pos, nfl_team="ZZA",
                          active=True, raw_value=Decimal(str(raw)))
        anchors.append(a)
        key_by_id[s.player_id] = a.canonical_key

    book = AnchorBook(tuple(anchors), source_sha256="fabricated-tactical-demo",
                      notes="FABRICATED tactical demo anchors; no real player.")
    prior = build_market_prior(
        book, settings=settings,
        player_ids={k: pid for pid, k in key_by_id.items()},
        notes="FABRICATED tactical demo board.")
    id_by_key = {k: pid for pid, k in key_by_id.items()}
    return TacticalDemo(auction=auction, prior=prior,
                        market=MarketState(prior=prior),
                        key_by_id=key_by_id, id_by_key=id_by_key)


def demo_sales(demo: TacticalDemo, n: int = 9) -> List[SaleObservation]:
    """A scripted fabricated sale sequence over the joined board.

    Tight ends clear well under the anchor, receivers near it, and one owner
    repeatedly pays up -- enough to move every learning channel the market
    layer has, and enough for one owner to cross the buyer-evidence threshold.
    """
    sales: List[SaleObservation] = []
    seq = 0
    plan = (("TE", 0.45, None), ("WR", 1.02, None), ("RB", 1.40, "Owner04"))
    per = max(1, n // len(plan))
    for pos, ratio, buyer in plan:
        group = [p for p in demo.prior.draftable if p.position == pos]
        group.sort(key=lambda p: (-p.base_price, p.canonical_key))
        for p in group[:per]:
            sales.append(SaleObservation(
                p.canonical_key, pos, max(1, int(round(p.base_price * ratio))),
                buyer or f"Owner{(seq % 11) + 2:02d}", seq, p.base_price,
                p.display_anchor,
                notes=f"FABRICATED {pos} sale at {ratio:.2f}x the model"))
            seq += 1
    return sales
