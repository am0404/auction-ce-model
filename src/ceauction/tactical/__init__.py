"""The tactical layer: from a price band to a bid, through named opponents.

The auction layer answers "is buying him at $p better than the best
alternative?" and the market layer answers "what will this room pay?". Neither
answers the question a manager actually has with ten seconds on the clock:

    Owner07 has just bid $28 on this player. What is the highest price I should
    say, given who Owner07 is, what he already has, who else can still legally
    bid, who gets the player if I stop, and what each of those outcomes does to
    my championship equity?

Four prices, never collapsed:

``sleeper_display_anchor``   generic Sleeper number.       :mod:`..market.prior`
``expected_clearing_price``  what this room may pay.       :mod:`..market.live`
``ce_reservation_price``     what our equity can afford.   :mod:`..auction.reservation`
``tactical_max_bid``         what to actually bid.         :mod:`.maxbid`

The tactical maximum uses the other three and overwrites none of them.

What is honest about this layer, stated once here and repeated in every output:

* No manager's behaviour has been fitted. There is no historical auction for
  this league. Every willingness number is a **scenario range** under a named
  assumption, and the assumption travels with the number.
* Recipients are **named and evaluated one at a time**. Any weighted summary is
  built afterwards from branches that already exist, and its weights are stated
  assumptions rather than estimated probabilities.
* Opponent continuations use a **named proxy**, not championship equity. The
  final buy-versus-pass comparison uses the real CE engine when runtime allows,
  and says which of the two produced each number.
* ``immediate`` results are proxy or cached and carry no interval. They are not
  championship-equity estimates and are never labelled as such.
"""

from __future__ import annotations

from .bidders import (BIDDER_SCENARIOS, DEFAULT_BIDDER_SCENARIO,
                      BidderScenario, BidderWillingness, assess_bidder,
                      assess_room_bidders, format_bidders)
from .board import (OPPONENT_METHOD, Allocation, BoardResult, BoardSettings,
                    cast_from_board, continue_shared_board, format_board)
from .demo import TacticalDemo, build_tactical_demo, demo_sales
from .endgame import (EndgameReport, OwnerCapacity, assess_endgame,
                      candidate_legal_max, format_endgame)
from .maxbid import (DEFAULT_SCENARIOS, PriceVerdict, TacticalCache,
                     TacticalResult, TacticalScenario, TacticalSettings,
                     audited_max_bid, evaluate_tactical, format_tactical,
                     immediate_max_bid, tactical_cache_key)
from .recipients import (PRICE_RULE, RecipientBranch, RecipientSet,
                         enumerate_recipients, format_recipients)

__all__ = [
    "BIDDER_SCENARIOS", "DEFAULT_BIDDER_SCENARIO", "BidderScenario",
    "BidderWillingness", "assess_bidder", "assess_room_bidders",
    "format_bidders",
    "OPPONENT_METHOD", "Allocation", "BoardResult", "BoardSettings",
    "cast_from_board", "continue_shared_board", "format_board",
    "TacticalDemo", "build_tactical_demo", "demo_sales",
    "EndgameReport", "OwnerCapacity", "assess_endgame", "candidate_legal_max",
    "format_endgame",
    "DEFAULT_SCENARIOS", "PriceVerdict", "TacticalCache", "TacticalResult",
    "TacticalScenario", "TacticalSettings", "audited_max_bid",
    "evaluate_tactical", "format_tactical", "immediate_max_bid",
    "tactical_cache_key",
    "PRICE_RULE", "RecipientBranch", "RecipientSet", "enumerate_recipients",
    "format_recipients",
]
