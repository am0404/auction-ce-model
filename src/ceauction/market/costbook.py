"""Feeding expected clearing prices into the existing acquisition-cost contract.

The completion search needs a price per player. This turns a market scenario
into a :class:`~ceauction.auction.costs.CostBook` without letting the two axes
touch:

* the **market-cost scenario** (low / base / high clearing price) says what the
  room may charge;
* the **model scenario** (availability interpretation, forecastable share,
  season_sd, signal quality) says how good the players are.

They are independent questions with independent answers, and a run has to state
both. Nothing here reads championship equity, and nothing in the CE engine reads
these prices as value.

Provenance is ``PROVISIONAL``, never ``REAL``. These prices come from a generic
Sleeper 2026 `2qb` list transformed by scenarios nobody has fitted, so nothing
derived from them is this league's market price -- and the cost contract's own
disclaimer machinery says so wherever the book surfaces.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Mapping, Optional

from ..auction.costs import CostBook, CostEntry, CostProvenance
from .live import MarketState
from .prior import MarketPrior

__all__ = ["MARKET_SCENARIO_NAMES", "cost_book_from_prior",
           "cost_book_from_market_state"]

MARKET_SCENARIO_NAMES = ("low", "base", "high")


def _provenance(fingerprint: str, scenario: str, model_scenario: Optional[str],
                observations: int, source: str, notes: str) -> CostProvenance:
    return CostProvenance(
        level="PROVISIONAL",
        source=source,
        scenario_id=(f"market={scenario}"
                     + (f"|model={model_scenario}" if model_scenario else "")),
        room_state=fingerprint,
        version=f"obs={observations}",
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        notes=("Expected clearing price from a GENERIC Sleeper 2026 '2qb' "
               "anchor, format-adjusted and budget-reconciled under stated "
               "scenarios. Not fitted; no historical auction exists. Market "
               "cost and championship-equity value are separate axes and this "
               "book carries only the former. " + notes).strip())


def cost_book_from_prior(
    prior: MarketPrior,
    scenario: str = "base",
    *,
    player_ids: Optional[Mapping[str, int]] = None,
    model_scenario: Optional[str] = None,
    minimum_cost: int = 1,
    notes: str = "",
) -> CostBook:
    """A cost book at one market scenario, before any sale has been observed.

    Players without a usable anchor are **omitted**, not priced at the minimum.
    The cost contract already refuses a missing price unless the caller states a
    fallback, and that refusal is the correct behaviour: an unpriced player and
    a player predicted to go for a dollar are different claims.
    """
    if scenario not in MARKET_SCENARIO_NAMES:
        raise ValueError(
            f"unknown market scenario {scenario!r}; expected one of "
            f"{MARKET_SCENARIO_NAMES}")
    player_ids = dict(player_ids or {})
    attr = {"low": "low_price", "base": "base_price", "high": "high_price"}[scenario]

    entries = []
    for p in prior.draftable:
        pid = p.player_id if p.player_id is not None else player_ids.get(p.canonical_key)
        if pid is None:
            continue
        entries.append(CostEntry(int(pid), max(minimum_cost, getattr(p, attr)),
                                 low=max(minimum_cost, p.low_price),
                                 high=max(minimum_cost, p.high_price)))
    return CostBook(
        entries=tuple(sorted(entries, key=lambda e: e.player_id)),
        provenance=_provenance(prior.fingerprint(), scenario, model_scenario, 0,
                               "ceauction.market prior (no sales observed)",
                               notes),
        minimum_cost=minimum_cost)


def cost_book_from_market_state(
    state: MarketState,
    scenario: str = "base",
    *,
    player_ids: Optional[Mapping[str, int]] = None,
    model_scenario: Optional[str] = None,
    buyer: Optional[str] = None,
    minimum_cost: int = 1,
    notes: str = "",
) -> CostBook:
    """A cost book reflecting everything the room has done so far.

    The book's fingerprint moves with the market state's, so a completion search
    cached against three sales ago cannot be served for the position now.
    """
    if scenario not in MARKET_SCENARIO_NAMES:
        raise ValueError(
            f"unknown market scenario {scenario!r}; expected one of "
            f"{MARKET_SCENARIO_NAMES}")
    player_ids = dict(player_ids or {})
    entries = []
    for p in state.prior.draftable:
        pid = p.player_id if p.player_id is not None else player_ids.get(p.canonical_key)
        if pid is None:
            continue
        adj = state.adjusted(p.canonical_key, buyer=buyer)
        if adj is None:
            continue
        value = {"low": adj.low, "base": adj.base, "high": adj.high}[scenario]
        entries.append(CostEntry(int(pid), max(minimum_cost, value),
                                 low=max(minimum_cost, adj.low),
                                 high=max(minimum_cost, adj.high)))
    return CostBook(
        entries=tuple(sorted(entries, key=lambda e: e.player_id)),
        provenance=_provenance(
            state.fingerprint(), scenario, model_scenario,
            len(state.observations),
            f"ceauction.market live state ({len(state.observations)} sale(s) observed)",
            notes),
        minimum_cost=minimum_cost)
