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
    "MARKET_ONLY_BASES",
    "NO_CE_NOTE",
    "GUARDRAIL_LABEL",
    "GUARDRAIL_NUMBER_LABEL",
    "MAX_BID_NUMBER_LABEL",
    "BELOW_MARKET",
    "IN_MARKET_RANGE",
    "ABOVE_MARKET",
    "OVER_LEGAL_MAX",
    "CANNOT_BID",
    "Recommendation",
    "has_tactical_support",
    "market_anchor_guardrail",
    "recommend",
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

#: Bases that rest on the market prior alone. A number carrying one of these
#: describes *what the room is likely to pay*. It is not a statement about what
#: winning this player is worth to us, so it can neither authorise a bid nor
#: forbid one -- see :func:`recommend`.
MARKET_ONLY_BASES: Tuple[str, ...] = (MARKET_PRIOR, MARKET_LIVE)

#: Bases that positively record a *failed* championship-equity computation.
#: These must never reach a BID/STOP word: a search that did not converge is
#: not evidence in either direction.
REFUSED_CE_BASES: Tuple[str, ...] = (SEARCH_UNDERCONVERGED, CE_UNDERPOWERED)

#: Printed beside every market-led recommendation, every time.
NO_CE_NOTE = "CE NOT AUDITED"

#: What the market-led dollar figure is called. Deliberately not "max bid":
#: nothing here has computed a maximum worth paying.
GUARDRAIL_NUMBER_LABEL = "MARKET GUARDRAIL"
MAX_BID_NUMBER_LABEL = "MAX BID"

#: The user's stated market-anchor policy (see :func:`market_anchor_guardrail`).
GUARDRAIL_LABEL = "USER MARKET-ANCHOR POLICY"

# The three words a market-only basis may print. None of them is advice.
BELOW_MARKET = "BELOW MARKET"
IN_MARKET_RANGE = "IN MARKET RANGE"
ABOVE_MARKET = "ABOVE MARKET"

#: Not a judgement at all -- auction arithmetic says this bid cannot be made.
OVER_LEGAL_MAX = "OVER LEGAL MAXIMUM"
CANNOT_BID = "CANNOT BID"


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

    sleeper_display_anchor: Optional[int] = None
    """The whole-dollar number Sleeper itself shows, carried through untouched.

    This is **not** a model output and nothing in this module may alter it. It
    is here because the room reads it: the user's stated premise is that
    managers treat the Sleeper figure as a psychological anchor, are reluctant
    to bid far above it and see prices far below it as bargains. That makes it
    a fact about the bidders, which is exactly the sort of thing a guardrail
    may use -- see :func:`market_anchor_guardrail`.
    """

    def to_dict(self) -> Dict[str, object]:
        return {"legal_max": self.legal_max, "market": self.market.to_dict(),
                "proxy_ceiling": self.proxy_ceiling,
                "proxy_status": self.proxy_status,
                "ce_bracket": list(self.ce_bracket) if self.ce_bracket else None,
                "ce_status": self.ce_status,
                "manual_adjustment": self.manual_adjustment,
                "manual_note": self.manual_note,
                "sleeper_display_anchor": self.sleeper_display_anchor}


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

    # The basis names the rail that actually bound the number, not the most
    # interesting rail available. When our own legal maximum is the binding
    # constraint -- because the player is gone, or because we simply cannot
    # reach his market price -- the answer came from auction arithmetic, and
    # calling it a market recommendation would credit an estimate for a number
    # the rules produced.
    if bound == "legal_max":
        basis = EXACT
        label = ("LEGAL MAXIMUM BINDS" if market.is_priced
                 else "LEGAL MAXIMUM ONLY -- NO MARKET ANCHOR")
    elif bound == "proxy_ceiling":
        basis = PROXY
        label = MARKET_LED_LABEL
    else:
        basis = market.basis
        label = MARKET_LED_LABEL
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


# ---------------------------------------------------------------------------
# What the screen is allowed to say
# ---------------------------------------------------------------------------


def has_tactical_support(rails: CapRails) -> bool:
    """Is there a converged tactical or CE result behind this player at all?

    A market prior is a forecast of what *the room* will pay. It contains no
    statement about what winning the player is worth to us, so it cannot
    authorise a bid and it cannot forbid one. Only two things in this
    repository can: an audited CE bracket, or the immediate tactical proxy when
    it actually ran and returned a ceiling.

    ``proxy_status == "cached"`` with a ``None`` ceiling is a *converged*
    result that found no favourable price -- it is real evidence, and it
    supports a STOP. ``calculating``, ``failed`` and ``absent`` are not.
    """
    if rails.ce_status == "usable" and rails.ce_bracket is not None:
        return True
    return rails.proxy_status == "cached"


def market_anchor_guardrail(rails: CapRails) -> Tuple[Optional[int], str]:
    """The USER MARKET-ANCHOR POLICY number, and the rail that bound it.

    The user's premise about this league is explicit: managers see Sleeper's
    projected price, are reluctant to go far above it, read prices far below it
    as bargains, and disregard it where the format is obviously wrong -- most
    of all at tight end, where a number computed for another lineup shape is
    simply about a different game.

    Both of those are true at once, and they disagree. This repository's
    format-adjusted range says what the price *should* be under this league's
    rules; the Sleeper number says what the room is *anchored on*. Taking the
    smaller would have us walk away from players the room will genuinely bid
    up, so the policy takes the larger::

        guardrail = max(raw Sleeper displayed price, adjusted market high)

    clamped only by the exact legal maximum, with the user's manual adjustment
    applied explicitly on top.

    This is a **bidding-behaviour guardrail and nothing more**. It is not
    championship equity, it is not a maximum worth paying, and it must never be
    described as either.

    An unpriced player returns ``None``, not a number. Falling back to the legal
    maximum there would put "$186" beside a player nobody has valued at all,
    which reads as permission to spend it -- the same kind of invented signal
    this policy exists to remove.
    """
    legal = max(0, int(rails.legal_max))
    market_high = int(rails.market.high) if rails.market.is_priced else None
    sleeper = (int(rails.sleeper_display_anchor)
               if rails.sleeper_display_anchor is not None else None)

    if market_high is None and sleeper is None:
        return None, "unpriced"
    if sleeper is None:
        value, bound = market_high, "market_high"
    elif market_high is None:
        value, bound = sleeper, "sleeper_anchor"
    elif sleeper >= market_high:
        value, bound = sleeper, "sleeper_anchor"
    else:
        value, bound = market_high, "market_high"

    if value > legal:
        value, bound = legal, "legal_max"
    if rails.manual_adjustment:
        value = max(0, min(legal, value + int(rails.manual_adjustment)))
        bound = "manual"
    return int(value), bound


@dataclass(frozen=True)
class Recommendation:
    """What the screen prints, and every claim it is entitled to make.

    ``decision`` is the one word the user reads on a ten-second clock.
    ``ce_audited`` says whether any championship-equity result stands behind
    it. When it is false, ``decision`` is a market-position statement --
    :data:`BELOW_MARKET` / :data:`IN_MARKET_RANGE` / :data:`ABOVE_MARKET` --
    and never :data:`BID` or :data:`STOP`.
    """

    decision: str
    number: Optional[int]
    number_label: str
    basis: str
    ce_audited: bool
    detail: str = ""
    notes: Tuple[str, ...] = ()

    @property
    def is_advice(self) -> bool:
        """True only when the word is an instruction rather than a position."""
        return self.decision in ("BID", "CAUTION", "STOP")

    def to_dict(self) -> Dict[str, object]:
        return {"decision": self.decision, "number": self.number,
                "number_label": self.number_label, "basis": self.basis,
                "ce_audited": self.ce_audited, "detail": self.detail,
                "notes": list(self.notes), "is_advice": self.is_advice}


def recommend(cap: ProvisionalCap, next_bid: int, *,
              guardrail: Optional[int] = None) -> Recommendation:
    """The only function permitted to produce the word on the screen.

    The rule this hotfix exists to enforce: **a market prior alone can neither
    authorise a bid nor forbid one.** Josh Allen at $30 against an adjusted
    market high of $29 is a player going ABOVE MARKET -- a fact about where the
    price sits in a forecast band. Printing STOP there claims we computed that
    $30 costs us championship equity, and we did not.

    So a market-only basis prints a *position*, and the dollar figure beside it
    is a :data:`GUARDRAIL_NUMBER_LABEL`, never a :data:`MAX_BID_NUMBER_LABEL`.
    BID/STOP is reserved for a converged tactical proxy or an audited CE
    bracket, and :data:`SEARCH_UNDERCONVERGED` / :data:`CE_UNDERPOWERED` --
    failed computations, not evidence -- can never reach it.
    """
    rails = cap.rails
    legal = max(0, int(rails.legal_max))
    tactical = has_tactical_support(rails)
    number = guardrail if guardrail is not None else int(cap.cap)
    notes: list = []

    # Auction arithmetic, which owes nobody a model. These are facts about the
    # rules, so they are stated as such rather than dressed as advice.
    if legal <= 0:
        return Recommendation(
            decision=CANNOT_BID, number=0, number_label="LEGAL MAXIMUM",
            basis=EXACT, ce_audited=False,
            detail=("we cannot legally buy this player at any price in the "
                    "current room"))
    if next_bid > legal:
        return Recommendation(
            decision=OVER_LEGAL_MAX, number=legal,
            number_label="LEGAL MAXIMUM", basis=EXACT, ce_audited=False,
            detail=f"${next_bid} exceeds our exact legal maximum of ${legal}")

    if cap.basis in REFUSED_CE_BASES or rails.ce_status in REFUSED_CE_BASES:
        notes.append(f"CE result refused: {rails.ce_status}")

    # --- the audited path: the only place BID/STOP may be printed ----------
    if tactical and cap.basis not in REFUSED_CE_BASES:
        audited = cap.basis == CE_AUDITED
        label = MAX_BID_NUMBER_LABEL if audited else "PROXY CEILING"
        if not audited:
            notes.append(NO_CE_NOTE)
        if next_bid > cap.cap:
            decision = "STOP"
        elif cap.disagreement:
            decision = "CAUTION"
        else:
            base = rails.market.base
            decision = ("CAUTION" if base is not None and next_bid > int(base)
                        else "BID")
        return Recommendation(
            decision=decision, number=int(cap.cap), number_label=label,
            basis=cap.basis, ce_audited=audited,
            detail=(f"next bid ${next_bid} against a {label.lower()} of "
                    f"${cap.cap}"),
            notes=tuple(notes))

    # --- the market-only path: a position, not an instruction --------------
    notes.append(NO_CE_NOTE)
    band = rails.market
    if not band.is_priced:
        # No number, not even our legal maximum: "nobody priced him" and "you
        # may spend up to $186 on him" are different claims.
        return Recommendation(
            decision="UNPRICED", number=None,
            number_label=GUARDRAIL_NUMBER_LABEL, basis=cap.basis,
            ce_audited=False,
            detail=("the market list never priced this player. That is not an "
                    "observed $1 sale and no price is shown for him."),
            notes=tuple(notes))

    low, high = int(band.low), int(band.high)
    if next_bid < low:
        decision, detail = BELOW_MARKET, (
            f"${next_bid} is under the adjusted range ${low}-${high}")
    elif next_bid <= high:
        decision, detail = IN_MARKET_RANGE, (
            f"${next_bid} sits inside the adjusted range ${low}-${high}")
    else:
        decision, detail = ABOVE_MARKET, (
            f"${next_bid} is over the adjusted range ${low}-${high}")
    if cap.disagreement:
        notes.append(DISAGREEMENT_LABEL)
    return Recommendation(
        decision=decision, number=number,
        number_label=GUARDRAIL_NUMBER_LABEL, basis=cap.basis,
        ce_audited=False, detail=detail, notes=tuple(notes))
