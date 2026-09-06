"""Acquisition cost: what this room may pay, kept apart from what we value.

Four quantities, and the whole package exists to stop them collapsing into one:

``sleeper_display_anchor``   what managers see on Sleeper -- a generic 2026
                             `2qb` number for a lineup that is not ours.
``expected_clearing_price``  what this room may actually pay, built from that
                             anchor plus a lineup-format argument, a budget
                             reconciliation, and whatever has already sold.
``ce_reservation_range``     what our championship-equity engine says we can
                             afford. A different question entirely, and it lives
                             in :mod:`ceauction.auction.reservation`.
``tactical_max_bid``         what to actually bid given who else can bid and how
                             their money is committed. **Not built anywhere.**

Nothing here is fitted. No historical auction exists for this league or a
comparable one, so every coefficient is a stated scenario, and the label travels
with the number to every place it surfaces.
"""

from __future__ import annotations

from .anchors import (ROSTERABLE_POSITIONS, SLEEPER_ENDPOINT, SLEEPER_FORMAT,
                      SLEEPER_SEASON, AnchorBook, AnchorError, AnchorMatchReport,
                      SleeperAnchor, display_price, load_sleeper_csv,
                      match_anchors_to_contract)
from .costbook import (MARKET_SCENARIO_NAMES, cost_book_from_market_state,
                       cost_book_from_prior)
from .live import (TIERS, AdjustedPrice, MarketState, PoolLevel,
                   SaleObservation, ShrinkageConfig, tier_of)
from .pressure import (OwnerPressure, RoomPressure, assess_room_pressure,
                       format_room_pressure)
from .prior import (MARKET_SCENARIOS, AnchorCredibility, MarketPrior,
                    MarketScenario, PlayerPrior, RoomBudget,
                    build_market_prior, format_prior_summary)

__all__ = [
    "SLEEPER_ENDPOINT", "SLEEPER_FORMAT", "SLEEPER_SEASON",
    "ROSTERABLE_POSITIONS", "AnchorError", "SleeperAnchor", "AnchorBook",
    "display_price", "load_sleeper_csv", "AnchorMatchReport",
    "match_anchors_to_contract", "RoomBudget", "AnchorCredibility",
    "MarketScenario", "MARKET_SCENARIOS", "PlayerPrior", "MarketPrior",
    "build_market_prior", "format_prior_summary",
    "SaleObservation", "PoolLevel", "ShrinkageConfig", "MarketState",
    "AdjustedPrice", "TIERS", "tier_of",
    "OwnerPressure", "RoomPressure", "assess_room_pressure",
    "format_room_pressure",
    "MARKET_SCENARIO_NAMES", "cost_book_from_prior",
    "cost_book_from_market_state",
]
