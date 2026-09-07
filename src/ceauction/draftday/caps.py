"""The provisional cap: one visible working number, and what bound it.

The user needs a single dollar figure to bid against on a ten-second clock.
That number must never be able to pass itself off as championship-equity
evidence, so this module keeps the rails apart and reports which one bound the
answer:

1. **exact legal maximum** -- auction arithmetic, not an estimate;
2. **adjusted market low / base / high** -- the existing market prior, moved by
   live sales when any have been observed;
3. **the existing immediate tactical proxy ceiling**, where one is cached;
4. **a resolved cached CE bracket**, only where it passes :func:`ce_is_usable`;
5. **a manual adjustment** the user typed.

The default, when no valid CE bracket exists::

    provisional_cap = min(exact legal maximum,
                          adjusted market high,
                          proxy permissive ceiling if available)

and with no proxy ceiling::

    provisional_cap = min(exact legal maximum, adjusted market high)

There is no second coefficient, no blend and no hidden shrinkage. A cap is a
minimum of rails that already exist, which is why every cap can name the rail
that produced it.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Optional, Tuple

__all__ = [
    "BASES",
    "EXACT",
    "MARKET_PRIOR",
    "MARKET_LIVE",
    "PROXY",
    "CE_AUDITED",
    "CE_UNDERPOWERED",
    "SEARCH_UNDERCONVERGED",
    "MANUAL",
    "MARKET_LED_LABEL",
    "DISAGREEMENT_LABEL",
    "DISAGREEMENT_DOLLARS",
    "DISAGREEMENT_FRACTION",
    "MarketBand",
    "CapRails",
    "ProvisionalCap",
    "ce_is_usable",
    "provisional_cap",
]

# --- the only recommendation bases this product may print -------------------

EXACT = "EXACT FINANCIAL/ROSTER ARITHMETIC"
MARKET_PRIOR = "MARKET PRIOR"
MARKET_LIVE = "MARKET + LIVE SALES"
PROXY = "PROXY/HEURISTIC"
CE_AUDITED = "CE AUDITED"
CE_UNDERPOWERED = "CE UNDERPOWERED"
SEARCH_UNDERCONVERGED = "SEARCH UNDERCONVERGED"
MANUAL = "MANUAL OVERRIDE"

#: Every label a recommendation may carry. Anything else is a bug.
BASES: Tuple[str, ...] = (
    EXACT, MARKET_PRIOR, MARKET_LIVE, PROXY, CE_AUDITED, CE_UNDERPOWERED,
    SEARCH_UNDERCONVERGED, MANUAL,
)

#: What a cap built from rails 1-3 is called. Deliberately not a CE word.
MARKET_LED_LABEL = "MARKET-LED PROVISIONAL CAP"

DISAGREEMENT_LABEL = "MANUAL REVIEW -- MODEL DISAGREEMENT"

#: Market and proxy disagree materially past either of these.
DISAGREEMENT_DOLLARS = 10
DISAGREEMENT_FRACTION = 0.30


@dataclass(frozen=True)
class MarketBand:
    """A player's expected clearing price, low / base / high.

    ``anchored`` is false for a player the Sleeper list never priced. Such a
    player is **unanchored**, which is not the same thing as an observed $1
    sale, and the distinction is carried all the way to the screen.
    """

    low: Optional[int]
    base: Optional[int]
    high: Optional[int]
    anchored: bool
    basis: str = MARKET_PRIOR
    """:data:`MARKET_PRIOR` until a live sale has moved this player's level,
    then :data:`MARKET_LIVE`."""
    note: str = ""

    @property
    def is_priced(self) -> bool:
        return self.anchored and self.high is not None

    def to_dict(self) -> Dict[str, object]:
        return {"low": self.low, "base": self.base, "high": self.high,
                "anchored": self.anchored, "basis": self.basis,
                "note": self.note}


@dataclass(frozen=True)
class CapRails:
    """Every rail that could bind a cap, kept separate and all displayable."""

    legal_max: int
    """Exact auction arithmetic: the largest dollar we may legally bid on this
    player right now, given our budget, open slots and roster feasibility."""

    market: MarketBand
    proxy_ceiling: Optional[int] = None
    """The existing ``TacticalResult.permissive_ceiling`` -- favourable under at
    least one scenario/recipient. A ceiling, not advice, and not CE."""

    proxy_status: str = "absent"
    """``absent`` / ``calculating`` / ``cached`` / ``failed``."""

    ce_bracket: Optional[Tuple[int, Optional[int]]] = None
    ce_status: str = "none"
    """``none`` / ``usable`` / plus a refusal reason from :func:`ce_is_usable`."""

    manual_adjustment: int = 0
    """Signed dollars the user typed. Applied last, clamped to the legal max."""

    manual_note: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {"legal_max": self.legal_max, "market": self.market.to_dict(),
                "proxy_ceiling": self.proxy_ceiling,
                "proxy_status": self.proxy_status,
                "ce_bracket": list(self.ce_bracket) if self.ce_bracket else None,
                "ce_status": self.ce_status,
                "manual_adjustment": self.manual_adjustment,
                "manual_note": self.manual_note}


@dataclass(frozen=True)
class ProvisionalCap:
    """The one working number, plus an account of where it came from."""

    cap: int
    basis: str
    label: str
    bound_by: str
    """Which rail produced the minimum: ``legal_max`` / ``market_high`` /
    ``proxy_ceiling`` / ``ce_bracket`` / ``manual``."""

    rails: CapRails
    disagreement: bool = False
    notes: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.basis not in BASES:
            raise ValueError(f"unknown recommendation basis {self.basis!r}")

    @property
    def is_ce(self) -> bool:
        return self.basis == CE_AUDITED

    def to_dict(self) -> Dict[str, object]:
        return {"cap": self.cap, "basis": self.basis, "label": self.label,
                "bound_by": self.bound_by, "disagreement": self.disagreement,
                "disagreement_label": (DISAGREEMENT_LABEL
                                       if self.disagreement else ""),
                "notes": list(self.notes), "rails": self.rails.to_dict()}


def ce_is_usable(record: Optional[Dict[str, object]],
                 state_fingerprint: str) -> Tuple[bool, str]:
    """May this cached CE result replace the provisional cap?

    Five conditions, all required, none inferred from any other:

    * the completion search converged;
    * the allocation interval resolved to a bracket;
    * the frontier and the decomposition reconciled;
    * the pass semantics recorded match the decision being displayed;
    * the result was produced against *this* auction state.

    The quarterback union work on ``qb-union-convergence`` satisfies none of
    the first three, which is why it produces no CE max bid here.
    """
    if not record:
        return False, "none"
    if not record.get("converged"):
        return False, SEARCH_UNDERCONVERGED
    if not record.get("interval_resolved"):
        return False, CE_UNDERPOWERED
    if not record.get("reconciled"):
        return False, CE_UNDERPOWERED
    if record.get("pass_semantics") != record.get("displayed_decision"):
        return False, CE_UNDERPOWERED
    if record.get("state_fingerprint") != state_fingerprint:
        return False, "stale: produced against a different auction state"
    bracket = record.get("bracket")
    if not bracket:
        return False, CE_UNDERPOWERED
    return True, "usable"


def _disagree(market_base: Optional[int], proxy: Optional[int]) -> bool:
    """Do the market and the proxy differ enough to demand a human look?"""
    if market_base is None or proxy is None:
        return False
    gap = abs(int(proxy) - int(market_base))
    if gap > DISAGREEMENT_DOLLARS:
        return True
    denom = max(1, int(market_base))
    return gap / denom > DISAGREEMENT_FRACTION


def provisional_cap(rails: CapRails) -> ProvisionalCap:
    """Apply the stated policy. No coefficient is introduced here."""
    notes: list = []
    legal = max(0, int(rails.legal_max))
    market = rails.market
    disagreement = _disagree(market.base, rails.proxy_ceiling)

    # --- rail 4: audited CE, only when the gate passed upstream ------------
    if rails.ce_status == "usable" and rails.ce_bracket is not None:
        low, high = rails.ce_bracket
        cap = min(legal, int(high if high is not None else low))
        bound = "ce_bracket" if cap != legal else "legal_max"
        return ProvisionalCap(cap=cap, basis=CE_AUDITED, label="CE AUDITED CAP",
                              bound_by=bound, rails=rails,
                              disagreement=disagreement, notes=tuple(notes))
    if rails.ce_status not in ("none", "usable"):
        notes.append(f"CE result refused: {rails.ce_status}")

    # --- rails 1-3: the market-led default ---------------------------------
    candidates: Dict[str, int] = {"legal_max": legal}
    if market.is_priced:
        candidates["market_high"] = int(market.high)
    else:
        notes.append(
            "UNANCHORED -- the market list never priced this player; this is "
            "not an observed $1 sale")
    if rails.proxy_status == "cached":
        if rails.proxy_ceiling is not None:
            candidates["proxy_ceiling"] = int(rails.proxy_ceiling)
        else:
            # The ladder ran and no price came back favourable under any
            # scenario. That is a result, not a missing one, and it is the
            # opposite of permissive -- so it must be visible rather than
            # silently leaving the market rail to bind alone.
            notes.append(
                "PROXY: no price on the ladder was favourable under any "
                "scenario or recipient. The proxy supports no bid here; the "
                "cap below is market-led only.")

    bound = min(candidates, key=lambda k: (candidates[k], k))
    cap = candidates[bound]

    if "proxy_ceiling" in candidates:
        basis = PROXY
        label = MARKET_LED_LABEL
    elif market.is_priced:
        basis = market.basis
        label = MARKET_LED_LABEL
    else:
        basis = EXACT
        label = "LEGAL MAXIMUM ONLY -- NO MARKET ANCHOR"
    if rails.proxy_status == "calculating":
        notes.append("CALCULATING -- proxy ceiling not yet available")

    # --- rail 5: the manual adjustment, applied last -----------------------
    if rails.manual_adjustment:
        cap = max(0, min(legal, cap + int(rails.manual_adjustment)))
        basis = MANUAL
        label = "MANUAL OVERRIDE CAP"
        bound = "manual" if cap != legal else "legal_max"
        if rails.manual_note:
            notes.append(rails.manual_note)

    return ProvisionalCap(cap=cap, basis=basis, label=label, bound_by=bound,
                          rails=rails, disagreement=disagreement,
                          notes=tuple(notes))


def advice(cap: ProvisionalCap, next_bid: int) -> str:
    """BID / CAUTION / STOP against the next legal bid.

    Thresholds are the cap itself and the market base -- both already on the
    screen. Nothing new is estimated to produce this word.
    """
    if next_bid > cap.rails.legal_max:
        return "STOP"
    if next_bid > cap.cap:
        return "STOP"
    base = cap.rails.market.base
    if base is not None and next_bid > int(base):
        return "CAUTION"
    if cap.disagreement:
        return "CAUTION"
    return "BID"
