"""Fabricated roster-strength regimes, so a player is valued in more than one.

Every earlier measurement put the focus team 12th of 12. A marginal player
looks very different from the bottom of a league than from the top: near the
playoff cutoff a few points of roster strength convert steeply into
championship probability, and well above or below it they barely convert at
all. Valuing a player in one regime and calling the answer his value is the
mistake this module exists to prevent.

**Only our roster moves.** The eleven rival rosters, the pool, the cost book,
the market state and every seed are identical across regimes; what differs is
how many players the focus team already holds and how good they are. Any CE
difference between regimes is a difference in *our* strength, not in the field.
The regime names label stated fabricated states; they are not predictions about
anybody, and which of them values a player most is left to CE to answer.

**Budget alone was tried first and does not work.** Varying only our money --
$28 to $179 with the same three pre-owned players -- moved our finished proxy
strength by 0.2 points (103.8 to 104.0) and left us 12th of 12 in every case.
The binding constraint is the board, not the wallet: by the time we complete,
the players who would lift us are already gone. So the regimes vary how much
roster we start with, which is the lever that actually reaches the top of the
league. The failed budget-only fixture is recorded here because it is a real
finding about this model, not a false start worth hiding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from ..auction.demo import DEMO_OWNERS, build_demo_auction
from ..league import DEFAULT_LEAGUE
from .demo import TacticalDemo, build_tactical_demo, demo_sales

__all__ = ["Regime", "REGIMES", "build_regime", "regime_summary"]


@dataclass(frozen=True)
class Regime:
    """One fabricated starting position for the focus team."""

    name: str
    description: str
    n_preowned: int
    """How many players the focus team already holds."""
    quality_offset: int
    """Where its pre-owned players are drawn from in the board's projection
    order. 0 is the best still available; a larger offset is a worse start."""
    spend_per_player: int
    """What it paid for each. Kept modest so no regime is starved of money --
    money is deliberately NOT the lever here."""

    @property
    def spent(self) -> int:
        return self.n_preowned * self.spend_per_player

    @property
    def budget_left(self) -> int:
        return DEFAULT_LEAGUE.budget - self.spent

    def to_dict(self) -> Dict[str, object]:
        return {"name": self.name, "description": self.description,
                "n_preowned": self.n_preowned,
                "quality_offset": self.quality_offset,
                "spend_per_player": self.spend_per_player,
                "spent": self.spent, "budget_left": self.budget_left}


#: Four stated regimes. The ordering of the *labels* is an intention; which one
#: actually values a player most is left to championship equity to reveal.
REGIMES: Dict[str, Regime] = {r.name: r for r in (
    Regime("underdog",
           "two mediocre players held; furthest below the field",
           n_preowned=2, quality_offset=40, spend_per_player=12),
    Regime("bubble",
           "three good players held; near the middle of the field",
           n_preowned=3, quality_offset=0, spend_per_player=18),
    Regime("bye_contender",
           "six good players held; competing for a top-two finish",
           n_preowned=8, quality_offset=0, spend_per_player=13),
    Regime("favorite",
           "nine good players held; among the strongest rosters in the room",
           n_preowned=11, quality_offset=0, spend_per_player=10),
)}


def build_regime(name: str, *, with_sales: bool = True) -> TacticalDemo:
    """A fabricated world identical to the demo except for our own budget."""
    if name not in REGIMES:
        raise ValueError(
            f"unknown regime {name!r}; known: {', '.join(sorted(REGIMES))}")
    regime = REGIMES[name]
    base = build_tactical_demo()
    # Rebuild the room paying the regime's prices for the focus team's three,
    # leaving every rival transaction exactly as the shared demo made it.
    fresh = build_demo_auction(focus_keep=0)
    state = fresh.state
    focus = state.focus_owner_id
    board = sorted(state.available_specs,
                   key=lambda sp: (-sp.base_mean, sp.player_id))
    taken = 0
    i = regime.quality_offset
    while taken < regime.n_preowned and i < len(board):
        spec = board[i]
        i += 1
        if state.purchase_shortfall(spec.player_id, focus,
                                    regime.spend_per_player) is not None:
            continue
        state = state.apply_purchase(spec.player_id, focus,
                                     regime.spend_per_player)
        taken += 1
    if taken < regime.n_preowned:
        raise ValueError(
            f"regime {name!r} could not legally seat {regime.n_preowned} "
            f"players; only {taken} fit")
    state.validate()
    d = base.with_state(state)
    if with_sales:
        d = d.with_market(d.market.observe_all(demo_sales(d, 12)))
    return d


def regime_summary(name: str, d: TacticalDemo) -> Dict[str, object]:
    focus = d.state.focus_owner_id
    o = d.state.owner(focus)
    return {"regime": name, "budget_remaining": o.budget_remaining,
            "n_preowned": o.n_players,
            "spent": o.spent, "open_slots": o.open_slots,
            "pre_owned": list(o.player_ids)}
