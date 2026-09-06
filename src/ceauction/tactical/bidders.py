"""Scenario bidder models: what an owner might be willing to pay, and why not.

**There is no historical auction for this league.** Nothing here is fitted,
nothing here is a probability, and nothing here is a claim about any particular
manager. What this module produces is a *scenario range*: "under the
``aggressive`` reading of this room, this owner would go to somewhere between
$18 and $27 on this player, and never above his $31 legal maximum".

Three rules the module enforces rather than merely documents.

**Every owner starts identical.** The prior is a function of the candidate, the
owner's money, his open slots and his roster shape. It contains no per-owner
constant. Two owners with the same money and the same roster get the same
number, and a test asserts it.

**Observation moves an owner only cautiously, and only through the market
layer.** :meth:`MarketState.adjusted` applies a buyer's learned tendency only
once he has cleared ``min_buyer_observations``; one purchase says almost
nothing about a manager and reading a personality into it is exactly the
failure mode this project exists to avoid. A sale price also does not reveal
the winner's maximum (he paid one increment over the runner-up, not his
ceiling) nor any loser's, and the model never pretends otherwise.

**Legality is a hard cap, behaviour is not.** Willingness is clipped by the
candidate-specific legal maximum from :mod:`ceauction.tactical.endgame`. There
is no positional quota anywhere: a manager with no sensible use for a third
quarterback gets a *reduced* pursuit score, never a forced zero.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence, Tuple

from ..auction.costs import CostBook
from ..auction.state import AuctionState, OwnerAuctionState
from ..league import Position
from ..market.live import MarketState
from ..market.pressure import _roster_fit
from .endgame import candidate_legal_max

__all__ = [
    "BidderScenario",
    "BIDDER_SCENARIOS",
    "DEFAULT_BIDDER_SCENARIO",
    "BidderWillingness",
    "assess_bidder",
    "assess_room_bidders",
    "format_bidders",
]


@dataclass(frozen=True)
class BidderScenario:
    """One stated reading of how this room bids. An assumption, not a fit."""

    name: str
    description: str

    anchor_weight: float
    """How far the scenario leans on Sleeper's displayed anchor rather than the
    updated clearing-price band. Multiplied by the market layer's own
    credibility for the position, so a room demonstrably ignoring the anchor
    pulls this down even under ``anchor_heavy``."""

    band_point: str
    """Which point of the clearing band centres the scenario: low/base/high."""

    aggression: float
    """Multiplier on the centred price. 1.0 pays the band."""

    fit_weight: float
    """How much a starting-lineup fit moves willingness. At 0.4 a player who
    fills a starting seat is worth ~20% more than pure bench depth."""

    budget_share: float
    """Ceiling as a share of the owner's discretionary money -- what he has
    above the $1 he must keep for every other open slot. A soft cap, applied
    before the hard legal cap."""

    spread: float
    """Half-width of the reported willingness range, relative to the centre.
    Wider is a more uncertain scenario, never a more informative one."""

    def cache_key(self) -> Tuple:
        import dataclasses
        return tuple((f.name, getattr(self, f.name))
                     for f in dataclasses.fields(self))

    def to_dict(self) -> Dict[str, object]:
        import dataclasses
        return {f.name: getattr(self, f.name)
                for f in dataclasses.fields(self)}


#: The stated scenarios. Names describe a *reading of the room*, never a person.
BIDDER_SCENARIOS: Dict[str, BidderScenario] = {
    s.name: s for s in (
        BidderScenario(
            "anchor_heavy",
            "the room bids Sleeper's displayed number and barely adjusts",
            anchor_weight=0.85, band_point="base", aggression=1.0,
            fit_weight=0.15, budget_share=0.55, spread=0.18),
        BidderScenario(
            "market_base",
            "the room bids the updated clearing band; the designated base case",
            anchor_weight=0.35, band_point="base", aggression=1.0,
            fit_weight=0.30, budget_share=0.60, spread=0.20),
        BidderScenario(
            "format_aware",
            "the room prices the superflex/no-TE-slot format, not the generic list",
            anchor_weight=0.10, band_point="base", aggression=1.02,
            fit_weight=0.55, budget_share=0.65, spread=0.22),
        BidderScenario(
            "aggressive",
            "the room pays up early and runs out of money late",
            anchor_weight=0.25, band_point="high", aggression=1.18,
            fit_weight=0.35, budget_share=0.75, spread=0.24),
        BidderScenario(
            "conservative",
            "the room hoards money and waits for the bargain rounds",
            anchor_weight=0.30, band_point="low", aggression=0.85,
            fit_weight=0.30, budget_share=0.45, spread=0.20),
    )
}

DEFAULT_BIDDER_SCENARIO = "market_base"


@dataclass(frozen=True)
class BidderWillingness:
    """One owner's scenario willingness for one candidate, with its caps."""

    owner_id: str
    candidate_id: int
    position: str
    scenario: str

    financial_max_bid: int
    candidate_legal_max: int
    """Hard caps. Willingness is clipped to these and can never exceed them."""

    market_low: Optional[int]
    market_base: Optional[int]
    market_high: Optional[int]
    display_anchor: Optional[int]
    price_basis: str
    """Where the centring price came from: the updated band, the raw prior, a
    cost-book fallback, or a floor. Never silently zero."""

    roster_fit: str
    fit_score: float
    open_slots: int
    discretionary: int
    owner_observations: int
    buyer_multiplier: Optional[float]

    low: int
    base: int
    high: int
    """The willingness RANGE. Not a point estimate and never reported as one."""

    pursuit_score: float
    score_kind: str = "uncalibrated ordinal score, NOT a fitted probability"
    exclusion_reason: Optional[str] = None
    capped_by: str = ""

    @property
    def can_compete(self) -> bool:
        return self.candidate_legal_max > 0 and self.high > 0

    def to_dict(self) -> Dict[str, object]:
        return {
            "owner_id": self.owner_id,
            "candidate_id": self.candidate_id,
            "position": self.position,
            "scenario": self.scenario,
            "financial_max_bid": self.financial_max_bid,
            "candidate_legal_max": self.candidate_legal_max,
            "market_band": {"low": self.market_low, "base": self.market_base,
                            "high": self.market_high},
            "display_anchor": self.display_anchor,
            "price_basis": self.price_basis,
            "roster_fit": self.roster_fit,
            "fit_score": round(self.fit_score, 4),
            "open_slots": self.open_slots,
            "discretionary": self.discretionary,
            "owner_observations": self.owner_observations,
            "buyer_multiplier": (None if self.buyer_multiplier is None
                                 else round(self.buyer_multiplier, 4)),
            "willingness": {"low": self.low, "base": self.base,
                            "high": self.high},
            "pursuit_score": round(self.pursuit_score, 4),
            "score_kind": self.score_kind,
            "exclusion_reason": self.exclusion_reason,
            "capped_by": self.capped_by,
            "label": ("scenario willingness range under a STATED assumption. "
                      "No manager preference has been estimated from data."),
        }


def _centre_price(scenario: BidderScenario,
                  low: Optional[int], base: Optional[int], high: Optional[int],
                  anchor: Optional[int], credibility: float,
                  fallback: Optional[int]) -> Tuple[float, str]:
    """The price the scenario centres on, and a name for where it came from."""
    band = {"low": low, "base": base, "high": high}[scenario.band_point]
    if band is None:
        band = base
    if band is None and anchor is None:
        if fallback is None:
            # A missing anchor is missing information, not a zero valuation.
            # The floor keeps the owner in the auction with a wide range rather
            # than silently excluding him.
            return 1.0, "floor (no anchor, no prior, no cost book)"
        return float(fallback), "cost-book fallback (no market anchor)"
    if band is None:
        return float(anchor), "displayed anchor only (no clearing prior)"
    if anchor is None:
        return float(band), "updated clearing band (no displayed anchor)"
    w = max(0.0, min(1.0, scenario.anchor_weight * credibility))
    return w * float(anchor) + (1.0 - w) * float(band), (
        f"blend: {w:.2f} displayed anchor + {1 - w:.2f} clearing band")


def assess_bidder(state: AuctionState, owner_id: str, candidate_id: int, *,
                  scenario: str = DEFAULT_BIDDER_SCENARIO,
                  market: Optional[MarketState] = None,
                  candidate_key: Optional[str] = None,
                  costs: Optional[CostBook] = None) -> BidderWillingness:
    """One owner's scenario willingness range for one candidate.

    Deterministic. Depends on the owner only through his money, his open slots
    and his roster shape, plus -- once he has enough observed purchases for the
    market layer to use them -- his own learned tendency.
    """
    if scenario not in BIDDER_SCENARIOS:
        raise ValueError(
            f"unknown bidder scenario {scenario!r}; "
            f"known: {', '.join(sorted(BIDDER_SCENARIOS))}")
    sc = BIDDER_SCENARIOS[scenario]
    spec = state.spec(candidate_id)
    position = Position(int(spec.position))
    owner = state.owner(owner_id)

    legal_max = candidate_legal_max(state, owner_id, candidate_id)
    reason: Optional[str] = None
    if legal_max <= 0:
        reason = (state.purchase_shortfall(candidate_id, owner_id,
                                           max(owner.min_bid, 1))
                  or "no legal price")

    adj = None
    if market is not None and candidate_key:
        adj = market.adjusted(candidate_key, buyer=owner_id)
    prior = (market.prior.by_key.get(candidate_key)
             if market is not None and candidate_key else None)

    low = adj.low if adj else (prior.low_price if prior else None)
    base = adj.base if adj else (prior.base_price if prior else None)
    high = adj.high if adj else (prior.high_price if prior else None)
    anchor = prior.display_anchor if prior else None
    cred = (market.anchor_credibility(position.name)
            if market is not None else 0.0)
    fallback = None
    if costs is not None and candidate_id in costs:
        fallback = costs.cost_of(candidate_id)

    centre, basis = _centre_price(sc, low, base, high, anchor, cred, fallback)

    fit_text, fit = _roster_fit(state, owner, position)
    # A starting seat is worth more than bench depth, by a stated factor. No
    # position gets a premium of its own: a quarterback and a receiver filling
    # the same superflex seat get the same multiplier.
    fit_mult = 1.0 + sc.fit_weight * (fit - 0.5)

    # Scarcity of alternatives: if very few comparable players remain, the same
    # seat is worth chasing harder. Bounded, and derived from the board rather
    # than from a positional preference.
    remaining = state.available_by_position()
    same = remaining.get(position, 0)
    total = sum(remaining.values()) or 1
    share = same / float(total)
    scarcity = 1.0 + 0.20 * (1.0 - min(1.0, share / 0.25))

    centre *= sc.aggression * fit_mult * scarcity

    # Soft ceiling: nobody spends everything he has above the $1 floor on one
    # player. A share, not a rule, and named in ``capped_by`` when it binds.
    soft = sc.budget_share * max(0, owner.discretionary) + owner.min_bid
    capped_by = ""
    if centre > soft:
        centre = soft
        capped_by = f"soft budget share ({sc.budget_share:.0%} of discretionary)"

    lo = centre * (1.0 - sc.spread)
    hi = centre * (1.0 + sc.spread)

    def clip(v: float) -> int:
        return int(max(0, min(legal_max, int(round(v)))))

    w_low, w_base, w_high = clip(lo), clip(centre), clip(hi)
    if legal_max > 0:
        w_low = max(owner.min_bid, w_low)
        w_base = max(w_low, w_base)
        w_high = max(w_base, w_high)
        if int(round(hi)) > legal_max:
            capped_by = (capped_by + "; " if capped_by else "") + \
                "candidate-specific legal maximum"
    else:
        w_low = w_base = w_high = 0

    # Pursuit score: an ordinal, bounded, uncalibrated readiness signal. It is
    # NOT a probability of bidding and the field name says so wherever it goes.
    if legal_max <= 0:
        score = 0.0
    else:
        headroom = min(1.0, w_base / float(legal_max)) if legal_max else 0.0
        slots = min(1.0, owner.open_slots / 4.0)
        score = max(0.0, min(1.0, 0.5 * fit + 0.3 * headroom + 0.2 * slots))

    lv = market.buyer_levels.get(owner_id) if market is not None else None
    n_obs = 0 if lv is None else lv.n
    return BidderWillingness(
        owner_id=owner_id, candidate_id=candidate_id, position=position.name,
        scenario=sc.name, financial_max_bid=owner.max_bid,
        candidate_legal_max=legal_max, market_low=low, market_base=base,
        market_high=high, display_anchor=anchor, price_basis=basis,
        roster_fit=fit_text, fit_score=fit, open_slots=owner.open_slots,
        discretionary=owner.discretionary, owner_observations=n_obs,
        buyer_multiplier=(adj.buyer_multiplier if adj else None),
        low=w_low, base=w_base, high=w_high, pursuit_score=score,
        exclusion_reason=reason, capped_by=capped_by)


def assess_room_bidders(state: AuctionState, candidate_id: int, *,
                        scenario: str = DEFAULT_BIDDER_SCENARIO,
                        market: Optional[MarketState] = None,
                        candidate_key: Optional[str] = None,
                        costs: Optional[CostBook] = None,
                        exclude: Sequence[str] = ()
                        ) -> Tuple[BidderWillingness, ...]:
    """Every owner's willingness, in descending order of the base point."""
    skip = set(exclude)
    out = [assess_bidder(state, o.owner_id, candidate_id, scenario=scenario,
                         market=market, candidate_key=candidate_key,
                         costs=costs)
           for o in state.owners if o.owner_id not in skip]
    out.sort(key=lambda w: (-w.base, -w.pursuit_score, w.owner_id))
    return tuple(out)


def format_bidders(rows: Sequence[BidderWillingness], width: int = 100) -> str:
    bar = "=" * width
    if not rows:
        return "no bidders assessed"
    first = rows[0]
    out = [bar, "SCENARIO BIDDER WILLINGNESS", bar,
           f"candidate  player {first.candidate_id} ({first.position})",
           f"scenario   {first.scenario} -- "
           f"{BIDDER_SCENARIOS[first.scenario].description}", ""]
    head = (f"  {'owner':<10}{'fin max':>9}{'legal max':>11}"
            f"{'willing low/base/high':>24}{'pursuit':>9}{'obs':>5}  fit / why not")
    out += [head, "  " + "-" * (width - 2)]
    for r in rows:
        rng = f"{r.low}/{r.base}/{r.high}"
        note = r.exclusion_reason or r.roster_fit
        if r.capped_by:
            note += f"  [capped by {r.capped_by}]"
        out.append(f"  {r.owner_id:<10}{r.financial_max_bid:>9}"
                   f"{r.candidate_legal_max:>11}{rng:>24}"
                   f"{r.pursuit_score:>9.3f}{r.owner_observations:>5}  {note}")
    out += ["", "  Willingness is a SCENARIO RANGE under a stated assumption.",
            "  'pursuit' is an uncalibrated ordinal score, NOT a probability.",
            "  No manager's preferences have been estimated from data; there is",
            "  no historical auction for this league to estimate them from.", bar]
    return "\n".join(out)
