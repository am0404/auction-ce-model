"""A fabricated market, so every command has something honest to run on.

**No real player appears here.** Anchors are generated from four smooth
positional curves and are labelled fabricated everywhere they surface. The point
is to exercise the machinery and give the committed examples something
reproducible, not to say anything about anybody.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, List, Tuple

from ..realdata.identity import stable_player_id
from .anchors import AnchorBook, SleeperAnchor
from .live import MarketState, SaleObservation, ShrinkageConfig
from .prior import MarketPrior, build_market_prior

__all__ = ["build_demo_anchors", "build_demo_prior", "DEMO_SALES", "demo_state"]

#: ``(position, count, top value, decline per rank)``. Four different slopes so
#: no position is uniformly the most expensive thing on the board.
_CURVES = (("QB", 30, 46.0, 1.45), ("RB", 50, 52.0, 0.98),
           ("WR", 62, 49.0, 0.74), ("TE", 20, 31.0, 1.42))


def build_demo_anchors() -> AnchorBook:
    """Deterministic fabricated Sleeper anchors, plus a few unusable rows."""
    anchors: List[SleeperAnchor] = []
    i = 0
    for pos, n, top, decline in _CURVES:
        for k in range(n):
            anchors.append(SleeperAnchor(
                sleeper_player_id=f"9{i:04d}",
                player_name=f"Fabricated {pos}{k:03d}",
                position=pos, nfl_team="ZZA", active=True,
                raw_value=Decimal(str(round(max(top - decline * k, 0.4), 2)))))
            i += 1
    # The three shapes a real export contains and a loader must keep apart.
    anchors.append(SleeperAnchor("99900", "Fabricated Kicker", "K", "ZZA", True,
                                 Decimal("3.10")))
    anchors.append(SleeperAnchor("99901", "Fabricated Retired", "RB", "ZZA",
                                 False, Decimal("12.40")))
    anchors.append(SleeperAnchor("99902", "Fabricated Negative", "WR", "ZZA",
                                 True, Decimal("-4.25")))
    return AnchorBook(tuple(anchors), source_sha256="fabricated-demo",
                      notes="FABRICATED demo anchors; no real player.")


def build_demo_prior() -> MarketPrior:
    book = build_demo_anchors()
    ids = {a.canonical_key: stable_player_id(a.canonical_key)
           for a in book.anchors}
    return build_market_prior(book, player_ids=ids,
                              notes="FABRICATED demo board.")


def _top(prior: MarketPrior, pos: str, n: int):
    group = [p for p in prior.draftable if p.position == pos]
    return sorted(group, key=lambda p: (-p.base_price, p.canonical_key))[:n]


#: A scripted room: tight ends going cheap, receivers near the model, and one
#: owner repeatedly overpaying. Enough to show every learning channel move.
def DEMO_SALES(prior: MarketPrior) -> List[SaleObservation]:
    sales: List[SaleObservation] = []
    seq = 0
    for p in _top(prior, "TE", 5):
        sales.append(SaleObservation(
            p.canonical_key, "TE", max(1, int(round(p.base_price * 0.45))),
            f"Owner{(seq % 12) + 1:02d}", seq, p.base_price, p.display_anchor,
            notes="TE going well under the visible anchor"))
        seq += 1
    for p in _top(prior, "WR", 4):
        sales.append(SaleObservation(
            p.canonical_key, "WR", max(1, int(round(p.base_price * 1.02))),
            f"Owner{(seq % 12) + 1:02d}", seq, p.base_price, p.display_anchor,
            notes="receiver clearing near the model"))
        seq += 1
    for p in _top(prior, "RB", 3):
        sales.append(SaleObservation(
            p.canonical_key, "RB", max(1, int(round(p.base_price * 1.35))),
            "Owner04", seq, p.base_price, p.display_anchor,
            notes="one owner paying up repeatedly"))
        seq += 1
    return sales


def demo_state(config: ShrinkageConfig = ShrinkageConfig()) -> MarketState:
    prior = build_demo_prior()
    return MarketState(prior=prior, config=config)
