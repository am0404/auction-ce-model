"""The auction layer: room state, completion search, and opportunity cost.

What this package answers is one question, and it is not "what is this player
worth". It is:

    Is buying X at $p at least as good, for championship equity, as passing on
    X and spending the $p on the best alternative still available?

That framing is the whole point. A player's value is not a property of the
player; it is the difference between the best roster reachable *with* him and
the best roster reachable *without* him and with his price back in the budget.
Neither of those rosters exists until something searches for it.

Three prices are kept apart, in names, in documentation and in output, because
conflating them is how a model starts lying:

**CE reservation price** -- the greatest price at which acquiring the player is
at least as good as the modelled best alternative. This package computes this
one.

**Expected clearing price** -- what the room will actually pay. Not modelled
here, and not predictable from anything in this package.

**Tactical winning bid** -- what to actually bid, given nomination order, who
else is still able to bid, and how much of their budget is committed. This
package builds the *state* that such logic would need. It does not contain the
logic.

Nothing here may be presented as an opening max bid or a recommended bid.
"""

from __future__ import annotations

from .feasibility import (PositionCounts, can_complete, can_fill_lineup,
                          completion_shortfall, lineup_deficit,
                          min_additions_for_lineup)
from .completion import (ComparisonCast, Completion, CompletionResult,
                         CompletionSettings, SearchDiagnostics,
                         complete_roster, enumerate_completions_exactly,
                         format_completion)
from .costs import (PROVENANCE_LEVELS, CostBook, CostEntry, CostProvenance,
                    MissingCost, flat_cost_book)
from .counterfactual import (BuyPassResult, PassDestination,
                             compare_buy_vs_pass, format_buy_pass)
from .proxy import ProxyEvaluator, proxy_strength
from .reservation import (MonotonicityViolation, PricePoint, ReservationCache,
                          ReservationResult, ScenarioReservation, ScenarioSetup,
                          estimate_runtime, format_reservation, price_ladder,
                          search_reservation)
from .state import (AuctionError, AuctionRuleError, AuctionState,
                    AuctionStateInvalid, Nomination, OwnerAuctionState,
                    RosterSlotFilled, Transaction, new_auction)

__all__ = [
    "PositionCounts",
    "can_complete",
    "can_fill_lineup",
    "completion_shortfall",
    "lineup_deficit",
    "min_additions_for_lineup",
    "AuctionError",
    "AuctionRuleError",
    "AuctionStateInvalid",
    "AuctionState",
    "Nomination",
    "OwnerAuctionState",
    "RosterSlotFilled",
    "Transaction",
    "new_auction",
    "ComparisonCast",
    "Completion",
    "CompletionResult",
    "CompletionSettings",
    "SearchDiagnostics",
    "complete_roster",
    "enumerate_completions_exactly",
    "format_completion",
    "PROVENANCE_LEVELS",
    "CostBook",
    "CostEntry",
    "CostProvenance",
    "MissingCost",
    "flat_cost_book",
    "BuyPassResult",
    "PassDestination",
    "compare_buy_vs_pass",
    "format_buy_pass",
    "ProxyEvaluator",
    "proxy_strength",
    "MonotonicityViolation",
    "PricePoint",
    "ReservationCache",
    "ReservationResult",
    "ScenarioReservation",
    "ScenarioSetup",
    "estimate_runtime",
    "format_reservation",
    "price_ladder",
    "search_reservation",
]
