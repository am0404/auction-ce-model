"""Learning from sales, cautiously, without pretending a sale reveals a bidder.

A completed sale is strong evidence about one thing and weak-to-silent evidence
about several others, and the difference matters enough to state at the top:

**What a final price does tell you.** That the room cleared this player at this
number. That at least two bidders were willing to go to one dollar below it, and
that the winner was willing to go to it. That is a *clearing price*, and it is
what this module learns from.

**What it does not tell you.** Not the winner's maximum: he stopped because
everyone else did, not because he ran out of willingness, so his true ceiling is
somewhere at or above what he paid and nothing here can say where. Not any
losing bidder's maximum either -- only that it was below the final price. Every
report from this module says so, because an updater that quietly treated a sale
as a revealed valuation would be inventing the most valuable data in the auction.

**How it learns.** Partially pooled, in four nested places -- the whole room, the
position, the price tier, and the individual buyer -- each with its own prior
strength. An observation moves a level by an amount proportional to how much
evidence that level already has, so the first tight end sold moves the tight-end
adjustment noticeably and the twentieth barely does. Residuals are capped before
they are pooled, so one absurd sale cannot drag the model with it, and every
level keeps a nonzero uncertainty floor no matter how many sales arrive: a room
that has bought forty players is better understood, not solved.

Deterministic: the same observations in the same order give the same state, and
the state serializes.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from typing import Dict, Iterable, Optional, Tuple

from .prior import MarketPrior

__all__ = [
    "SaleObservation",
    "PoolLevel",
    "ShrinkageConfig",
    "MarketState",
    "AdjustedPrice",
    "TIERS",
    "tier_of",
]

#: Price tiers, in dollars of base prior. Coarse on purpose: a finer partition
#: would spread the few observations an auction produces so thin that no tier
#: ever accumulated evidence.
TIERS: Tuple[Tuple[str, int], ...] = (
    ("elite", 30), ("mid", 12), ("depth", 4), ("dollar", 0),
)


def tier_of(base_price: int) -> str:
    for name, floor in TIERS:
        if base_price >= floor:
            return name
    return TIERS[-1][0]


@dataclass(frozen=True)
class SaleObservation:
    """One completed sale, as observed in the room."""

    player_key: str
    position: str
    final_price: int
    buyer: str
    sequence: int
    prior_base: Optional[int] = None
    """Our base expectation for him before the sale, if he had one."""
    display_anchor: Optional[int] = None
    tier: Optional[str] = None
    room_budget_remaining: Optional[int] = None
    """Pre-sale room state, recorded so a late-auction sale can be read as one."""
    slots_remaining: Optional[int] = None
    notes: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.final_price, int) or isinstance(self.final_price, bool):
            raise ValueError("final_price must be whole auction dollars")
        if self.final_price < 1:
            raise ValueError("a sale price is at least the $1 minimum bid")
        if self.sequence < 0:
            raise ValueError("sequence must be non-negative")

    @property
    def has_usable_prior(self) -> bool:
        return self.prior_base is not None and self.prior_base > 0

    @property
    def additive_residual(self) -> Optional[float]:
        """Dollars paid above our expectation. Defined for every priced player."""
        if self.prior_base is None:
            return None
        return float(self.final_price - self.prior_base)

    @property
    def log_ratio(self) -> Optional[float]:
        """Log price ratio. Only defined when the prior is comfortably positive.

        A ratio against a $1 prior is arithmetic noise -- a $4 sale reads as a
        300% miss -- so the threshold is deliberately above the floor and the
        additive residual carries those players instead.
        """
        if self.prior_base is None or self.prior_base < 3:
            return None
        return math.log(self.final_price / self.prior_base)

    def to_dict(self) -> Dict[str, object]:
        return {"player_key": self.player_key, "position": self.position,
                "final_price": self.final_price, "buyer": self.buyer,
                "sequence": self.sequence, "prior_base": self.prior_base,
                "display_anchor": self.display_anchor, "tier": self.tier,
                "room_budget_remaining": self.room_budget_remaining,
                "slots_remaining": self.slots_remaining, "notes": self.notes}


@dataclass(frozen=True)
class PoolLevel:
    """One partially pooled adjustment: room, position, tier or buyer."""

    name: str
    prior_strength: float
    """Pseudo-observations of evidence the prior is worth. Higher shrinks more."""
    n: int = 0
    mean_log_ratio: float = 0.0
    mean_additive: float = 0.0

    @property
    def weight(self) -> float:
        """How much of this level's estimate comes from data rather than prior."""
        return self.n / (self.n + self.prior_strength) if self.n else 0.0

    @property
    def multiplier(self) -> float:
        """Shrunk multiplicative adjustment. 1.0 means no learned change."""
        return math.exp(self.weight * self.mean_log_ratio)

    @property
    def additive(self) -> float:
        return self.weight * self.mean_additive

    def observe(self, log_ratio: Optional[float], additive: Optional[float],
                cap: float) -> "PoolLevel":
        """Fold in one observation, with the residual capped before pooling.

        Capping first is what stops a single absurd sale dominating: a $60
        overpay on a $4 player enters as its capped value, so it moves the
        level without deciding it.
        """
        n = self.n + 1
        lr = self.mean_log_ratio
        if log_ratio is not None:
            clipped = max(-cap, min(cap, log_ratio))
            lr = self.mean_log_ratio + (clipped - self.mean_log_ratio) / n
        add = self.mean_additive
        if additive is not None:
            add = self.mean_additive + (additive - self.mean_additive) / n
        return replace(self, n=n, mean_log_ratio=lr, mean_additive=add)

    def to_dict(self) -> Dict[str, object]:
        return {"name": self.name, "n": self.n,
                "prior_strength": self.prior_strength,
                "weight": round(self.weight, 4),
                "multiplier": round(self.multiplier, 4),
                "additive": round(self.additive, 3)}


@dataclass(frozen=True)
class ShrinkageConfig:
    """How much evidence each level needs before it moves. All stated, none fitted."""

    room: float = 10.0
    """Deliberately stronger than the position prior: a room-wide claim needs
    evidence from more than one position before it moves far."""
    position: float = 5.0
    tier: float = 6.0
    buyer: float = 8.0
    credibility: float = 4.0
    log_ratio_cap: float = 0.85
    """About a 2.3x miss in either direction, past which a sale is an outlier."""
    min_buyer_observations: int = 2
    """Below this, an owner's own purchases are not used to adjust his prices."""
    uncertainty_floor: float = 0.12
    """Band half-width can never shrink past this. Forty sales is not certainty."""

    def cache_key(self) -> Tuple:
        import dataclasses
        return tuple((f.name, getattr(self, f.name))
                     for f in dataclasses.fields(self))

    def to_dict(self) -> Dict[str, object]:
        import dataclasses
        return {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}


@dataclass(frozen=True)
class AdjustedPrice:
    """One player's expected clearing price after everything observed so far."""

    player_key: str
    position: str
    prior_low: int
    prior_base: int
    prior_high: int
    low: int
    base: int
    high: int
    room_multiplier: float
    position_multiplier: float
    tier_multiplier: float
    buyer_multiplier: Optional[float]
    anchor_credibility: float
    observations_used: Dict[str, int]
    uncertainty: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "position": self.position,
            "prior": {"low": self.prior_low, "base": self.prior_base,
                      "high": self.prior_high},
            "expected_clearing_price": {"low": self.low, "base": self.base,
                                        "high": self.high},
            "adjustments": {
                "room": round(self.room_multiplier, 4),
                "position": round(self.position_multiplier, 4),
                "tier": round(self.tier_multiplier, 4),
                "buyer": (None if self.buyer_multiplier is None
                          else round(self.buyer_multiplier, 4)),
            },
            "anchor_credibility": round(self.anchor_credibility, 4),
            "observations_used": self.observations_used,
            "uncertainty": round(self.uncertainty, 4),
            "label": ("expected clearing price, NOT our value and NOT a bid"),
        }


@dataclass(frozen=True)
class MarketState:
    """The prior plus everything the room has done. Immutable; serializable."""

    prior: MarketPrior
    config: ShrinkageConfig = ShrinkageConfig()
    observations: Tuple[SaleObservation, ...] = ()
    room: PoolLevel = PoolLevel("room", 10.0)
    positions: Tuple[Tuple[str, PoolLevel], ...] = ()
    tiers: Tuple[Tuple[str, PoolLevel], ...] = ()
    buyers: Tuple[Tuple[str, PoolLevel], ...] = ()
    credibility_shift: Tuple[Tuple[str, PoolLevel], ...] = ()
    """Per-position evidence about whether the room is following the anchor."""

    @property
    def room_effect(self) -> PoolLevel:
        """The room level, discounted by how identifiable it actually is.

        A room-wide adjustment and a position adjustment are collinear until
        sales span more than one position: with only tight ends sold, "this
        room is cheap" and "this room is cheap on tight ends" fit the data
        equally well, and attributing it to the room would drag every
        quarterback down on tight-end evidence. So the room level's evidence is
        scaled by how many distinct positions have contributed -- zero with one
        position, and approaching its face value as the sales spread out.
        """
        distinct = len({o.position for o in self.observations})
        if distinct <= 1:
            credit = 0.0
        else:
            credit = (distinct - 1) / float(distinct)
        return replace(self.room, n=int(round(self.room.n * credit)))

    # --- lookups -----------------------------------------------------------

    @property
    def position_levels(self) -> Dict[str, PoolLevel]:
        return dict(self.positions)

    @property
    def tier_levels(self) -> Dict[str, PoolLevel]:
        return dict(self.tiers)

    @property
    def buyer_levels(self) -> Dict[str, PoolLevel]:
        return dict(self.buyers)

    @property
    def credibility_levels(self) -> Dict[str, PoolLevel]:
        return dict(self.credibility_shift)

    def _level(self, levels: Tuple[Tuple[str, PoolLevel], ...], key: str,
               strength: float) -> PoolLevel:
        for name, lv in levels:
            if name == key:
                return lv
        return PoolLevel(key, strength)

    # --- observing ---------------------------------------------------------

    def observe(self, sale: SaleObservation) -> "MarketState":
        """Fold one sale in. Returns a new state; this one is untouched."""
        cfg = self.config
        lr = sale.log_ratio
        add = sale.additive_residual
        cap = cfg.log_ratio_cap

        def upsert(levels, key, strength):
            cur = self._level(levels, key, strength)
            new = cur.observe(lr, add, cap)
            return tuple(sorted(
                [(k, v) for k, v in levels if k != key] + [(key, new)],
                key=lambda kv: kv[0]))

        # The position level is updated with the raw residual; the room level
        # is updated with what is LEFT once that position's current estimate is
        # removed. Without that partialling-out, six tight ends sold at a
        # discount would drag every quarterback down with them -- which was the
        # observed behaviour before this, and is exactly the leak partial
        # pooling exists to prevent. A run of same-position sales now moves that
        # position a lot and the room only by whatever it fails to explain.
        pos_before = self._level(self.positions, sale.position, cfg.position)
        pos_offset = pos_before.weight * pos_before.mean_log_ratio
        pos_add_offset = pos_before.weight * pos_before.mean_additive

        positions = upsert(self.positions, sale.position, cfg.position)
        tier = sale.tier or tier_of(sale.prior_base or 0)
        tiers = upsert(self.tiers, tier, cfg.tier)
        buyers = upsert(self.buyers, sale.buyer, cfg.buyer)

        room_lr = None if lr is None else lr - pos_offset
        room_add = None if add is None else add - pos_add_offset

        # Anchor credibility learns from the anchor, not from our own prior:
        # the question it answers is whether this room is following the number
        # on the screen, which is a different question from whether our
        # transformation of it was right.
        cred = self.credibility_shift
        if sale.display_anchor and sale.display_anchor >= 3:
            anchor_lr = math.log(sale.final_price / sale.display_anchor)
            cur = self._level(cred, sale.position, cfg.credibility)
            cred = tuple(sorted(
                [(k, v) for k, v in cred if k != sale.position]
                + [(sale.position, cur.observe(anchor_lr, None, cap))],
                key=lambda kv: kv[0]))

        return replace(self, observations=self.observations + (sale,),
                       room=self.room.observe(room_lr, room_add, cap),
                       positions=positions, tiers=tiers, buyers=buyers,
                       credibility_shift=cred)

    def observe_all(self, sales: Iterable[SaleObservation]) -> "MarketState":
        state = self
        for sale in sales:
            state = state.observe(sale)
        return state

    # --- reading -----------------------------------------------------------

    def adjusted(self, player_key: str,
                 buyer: Optional[str] = None) -> Optional[AdjustedPrice]:
        """This player's expected clearing price given everything observed.

        ``buyer`` applies that owner's own learned tendency, but only once he
        has cleared ``min_buyer_observations``: one purchase says almost nothing
        about a manager and using it would be reading a personality into a
        single data point.
        """
        p = self.prior.by_key.get(player_key)
        if p is None or not p.draftable:
            return None
        cfg = self.config
        tier = tier_of(p.base_price)
        room = self.room_effect
        pos = self._level(self.positions, p.position, cfg.position)
        tl = self._level(self.tiers, tier, cfg.tier)

        buyer_level = None
        if buyer is not None:
            candidate = self._level(self.buyers, buyer, cfg.buyer)
            if candidate.n >= cfg.min_buyer_observations:
                buyer_level = candidate

        mult = room.multiplier * pos.multiplier * tl.multiplier
        if buyer_level is not None:
            mult *= buyer_level.multiplier
        # Additive residuals reach players a ratio cannot: a $1 anchor has no
        # meaningful ratio, so the pooled dollar miss carries them instead.
        additive = room.additive if p.base_price < 3 else 0.0

        def apply(value: int) -> int:
            return max(1, int(round(value * mult + additive)))

        base = apply(p.base_price)
        low, high = apply(p.low_price), apply(p.high_price)
        low, high = min(low, base), max(high, base)

        evidence = room.n + pos.n + tl.n
        half = max(cfg.uncertainty_floor,
                   (high - low) / (2.0 * base) if base else cfg.uncertainty_floor)
        # More relevant evidence narrows the band, but never past the floor.
        shrunk = max(cfg.uncertainty_floor,
                     half / math.sqrt(1.0 + evidence / 8.0))
        low = max(1, int(round(base * (1.0 - shrunk))))
        high = max(low, int(round(base * (1.0 + shrunk))))

        cred = self.anchor_credibility(p.position)
        return AdjustedPrice(
            player_key=player_key, position=p.position,
            prior_low=p.low_price, prior_base=p.base_price,
            prior_high=p.high_price, low=low, base=base, high=high,
            room_multiplier=room.multiplier, position_multiplier=pos.multiplier,
            tier_multiplier=tl.multiplier,
            buyer_multiplier=None if buyer_level is None else buyer_level.multiplier,
            anchor_credibility=cred,
            observations_used={"room": room.n, "position": pos.n, "tier": tl.n,
                               "buyer": 0 if buyer_level is None else buyer_level.n},
            uncertainty=shrunk)

    def anchor_credibility(self, position: str) -> float:
        """Current credibility of the Sleeper anchor for this position.

        Starts at the format-derived weight and moves with evidence. A room that
        keeps buying tight ends below their displayed number drives this down;
        one that pays list drives it up. **The anchor itself is never deleted or
        rewritten** -- only how far the room is believed to follow it changes.
        """
        base = self.prior.credibility.get(position)
        if base is None:
            return 0.0
        level = self._level(self.credibility_shift, position, self.config.credibility)
        if level.n == 0:
            return base.weight
        # A room paying far from the anchor is a room not using it. The size of
        # the miss reduces credibility; the learn rate says how fast, and the
        # pooled weight says how much evidence has accumulated.
        miss = abs(level.mean_log_ratio)
        pull = level.weight * base.learn_rate * min(1.0, miss / self.config.log_ratio_cap)
        return max(0.0, min(1.0, base.weight * (1.0 - pull)))

    # --- identity and reporting -------------------------------------------

    def fingerprint(self) -> str:
        """Digest of prior, config and every observation, in order."""
        h = hashlib.sha256()
        h.update(f"prior={self.prior.fingerprint()}\n".encode())
        h.update(f"config={self.config.cache_key()}\n".encode())
        for o in self.observations:
            h.update(f"{o.sequence}|{o.player_key}|{o.position}|{o.buyer}|"
                     f"{o.final_price}|{o.prior_base}|{o.display_anchor}\n".encode())
        return h.hexdigest()[:16]

    def to_dict(self) -> Dict[str, object]:
        return {
            "fingerprint": self.fingerprint(),
            "prior_fingerprint": self.prior.fingerprint(),
            "config": self.config.to_dict(),
            "observations": len(self.observations),
            "room": self.room.to_dict(),
            "room_effective": self.room_effect.to_dict(),
            "distinct_positions_observed": len(
                {o.position for o in self.observations}),
            "positions": {k: v.to_dict() for k, v in self.positions},
            "tiers": {k: v.to_dict() for k, v in self.tiers},
            "buyers": {k: v.to_dict() for k, v in self.buyers},
            "anchor_credibility": {
                pos: {"initial": round(self.prior.credibility[pos].weight, 4),
                      "current": round(self.anchor_credibility(pos), 4),
                      "observations": self._level(
                          self.credibility_shift, pos, self.config.credibility).n}
                for pos in self.prior.credibility},
            "what_a_sale_reveals": (
                "A final price is evidence about the CLEARING price and about "
                "buyer behaviour. It does NOT reveal the winner's maximum -- he "
                "stopped because the others did -- and it does NOT reveal any "
                "losing bidder's maximum beyond it being below the final price."),
        }

    def serialize(self) -> str:
        """Reproducible JSON of the observations. The prior is referenced by digest."""
        return json.dumps({
            "prior_fingerprint": self.prior.fingerprint(),
            "config": self.config.to_dict(),
            "observations": [o.to_dict() for o in self.observations],
        }, indent=2, sort_keys=True)

    @classmethod
    def restore(cls, prior: MarketPrior, blob: str) -> "MarketState":
        """Rebuild from serialized observations. Refuses a mismatched prior."""
        data = json.loads(blob)
        if data.get("prior_fingerprint") != prior.fingerprint():
            raise ValueError(
                f"these observations were recorded against prior "
                f"{data.get('prior_fingerprint')}, not {prior.fingerprint()}; "
                f"replaying them onto a different prior would silently change "
                f"every residual they encode")
        cfg = ShrinkageConfig(**data["config"])
        state = cls(prior=prior, config=cfg, room=PoolLevel("room", cfg.room))
        return state.observe_all(
            SaleObservation(**o) for o in sorted(data["observations"],
                                                 key=lambda o: o["sequence"]))
