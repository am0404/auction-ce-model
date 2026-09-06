"""Turning a generic Sleeper anchor into this league's expected clearing price.

The anchor is a number on a screen. Three things stand between it and what this
room will pay, and each is kept separately inspectable rather than folded into a
single adjusted price:

**Format.** The endpoint is `2qb`; this league is *superflex*. A team may start
a running back in the flexible slot and never draft a second quarterback, so
generic two-quarterback demand overstates the floor under quarterbacks. And the
list prices tight ends for a format with a required TE slot, which this league
does not have: a tight end here competes with receivers for three WR/TE slots
and with everyone for the flex. Neither is corrected by a hand-coded discount.
Both are expressed as **anchor credibility** -- how much this room is expected
to follow a number computed for a different lineup -- which is a quantity that
later evidence can move. A fixed "TE penalty" could not be wrong, and therefore
could not be learned from.

**Budget.** The room holds $2,400. It must fill 180 roster spots, and $1 has to
survive for each, so $180 is committed before anyone bids and $2,220 is
discretionary. The 149 priced anchors already total $2,384.26 *before* the other
31 players are bought. Paying list is not merely unlikely here, it is
arithmetically impossible, so a reconciliation step is mandatory rather than a
refinement.

**The room itself.** What has already sold, and for how much. That lives in
:mod:`ceauction.market.live`, which consumes this module's output as its prior.

Nothing here is fitted. There is no historical auction for this league or a
comparable one, so every coefficient below is a **stated scenario**, not an
estimate, and the code says so in every direction a reader might look.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, replace
from decimal import Decimal
from typing import Dict, List, Mapping, Optional, Tuple

from ..league import DEFAULT_LEAGUE, LeagueSettings
from .anchors import ROSTERABLE_POSITIONS, AnchorBook, SleeperAnchor

__all__ = [
    "RoomBudget",
    "AnchorCredibility",
    "MarketScenario",
    "MARKET_SCENARIOS",
    "PlayerPrior",
    "MarketPrior",
    "build_market_prior",
    "format_prior_summary",
]


# ---------------------------------------------------------------------------
# The room's money
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RoomBudget:
    """What the league can actually spend, from the rules alone.

    Every quantity here is arithmetic on the league settings, not a judgement.
    """

    settings: LeagueSettings = DEFAULT_LEAGUE

    @property
    def nominal_total(self) -> int:
        return self.settings.n_teams * self.settings.budget

    @property
    def roster_slots(self) -> int:
        return self.settings.n_teams * self.settings.roster_size

    @property
    def minimum_committed(self) -> int:
        """A dollar for every slot: money that cannot chase anyone."""
        return self.roster_slots * self.settings.min_bid

    @property
    def discretionary(self) -> int:
        return self.nominal_total - self.minimum_committed

    def spendable(self, spend_rate: float) -> float:
        """Discretionary dollars actually spent under a room-spend scenario.

        Rooms leave money on the table. ``spend_rate`` is the fraction of the
        discretionary pool this scenario assumes gets bid, and it is a stated
        assumption -- nothing in this repository measures it.
        """
        if not 0.0 < spend_rate <= 1.0:
            raise ValueError("spend_rate must be in (0, 1]")
        return self.discretionary * spend_rate

    def to_dict(self) -> Dict[str, object]:
        return {"nominal_total": self.nominal_total,
                "roster_slots": self.roster_slots,
                "minimum_committed": self.minimum_committed,
                "discretionary": self.discretionary}


# ---------------------------------------------------------------------------
# How far this room follows a number computed for another format
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnchorCredibility:
    """Per-position weight on the Sleeper anchor, and how movable it is.

    ``weight`` is how much of a player's position-relative standing this room is
    assumed to take from Sleeper rather than from its own reading of the lineup.
    ``learn_rate`` is how fast a sale should move it. A low weight paired with a
    high learn rate says "we do not expect this room to follow the anchor here,
    and the first few sales will tell us quickly" -- which is exactly the tight
    end's situation and is why the anchor is kept rather than deleted.
    """

    weight: float
    learn_rate: float
    reason: str

    def __post_init__(self) -> None:
        if not 0.0 <= self.weight <= 1.0:
            raise ValueError("credibility weight must be in [0, 1]")
        if not 0.0 < self.learn_rate <= 1.0:
            raise ValueError("learn_rate must be in (0, 1]")

    def to_dict(self) -> Dict[str, object]:
        return {"weight": round(self.weight, 4),
                "learn_rate": round(self.learn_rate, 4),
                "reason": self.reason}


def _format_credibility(settings: LeagueSettings,
                        adherence: float) -> Dict[str, AnchorCredibility]:
    """Credibility per position, derived from the lineup graph.

    The reasoning is the eligibility structure, not a table of preferences:

    * **QB** -- one dedicated slot plus a flexible one that a running back,
      receiver or tight end may fill instead. Generic 2QB pricing assumes both
      quarterback seats are effectively required, so its floor under
      quarterbacks is firmer than this league's. Credibility is reduced, but
      only partly: the dedicated slot is real and quarterbacks do score more.
    * **TE** -- *no dedicated slot at all*. A tight end competes with receivers
      for three WR/TE seats and with everyone for the flex, so a list built for
      a required-TE format overprices the position's floor here. Credibility is
      lowest and its learn rate is highest: a room that has noticed will show it
      in the first few tight ends sold.
    * **RB / WR** -- the format matches. Two dedicated running-back slots, three
      WR/TE slots, a flex open to both. Full credibility for the scenario's
      adherence level.

    ``adherence`` scales all four: how much this room follows a visible number
    at all, before any format argument.
    """
    dedicated = {"QB": 1, "RB": 2, "WR": 3, "TE": 3}
    # Fraction of a position's starting seats that are *guaranteed* to it rather
    # than contested. TE shares all three WT seats with WR, so its guaranteed
    # share is the lowest of the four even though the seat count is not.
    contested = {"QB": 0.55, "RB": 1.0, "WR": 0.9, "TE": 0.45}
    reasons = {
        "QB": ("one dedicated QB slot plus a superflex an RB/WR/TE may fill; "
               "generic 2qb pricing assumes firmer two-QB demand than this "
               "league has"),
        "RB": "two dedicated RB slots plus flex eligibility; format matches",
        "WR": "three WR/TE slots plus flex eligibility; format matches",
        "TE": ("NO dedicated TE slot in this league -- tight ends compete with "
               "receivers for the three WR/TE seats. A list priced for a "
               "required-TE format overstates the floor here"),
    }
    learn = {"QB": 0.35, "RB": 0.25, "WR": 0.25, "TE": 0.55}
    return {
        pos: AnchorCredibility(weight=adherence * contested[pos],
                               learn_rate=learn[pos], reason=reasons[pos])
        for pos in ROSTERABLE_POSITIONS
    }


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MarketScenario:
    """One stated set of assumptions about how this room will spend.

    **None of these is fitted.** No historical auction for this league exists,
    so every number is a scenario a human chose and can change, and the label
    says so wherever it surfaces.
    """

    name: str
    spend_rate: float
    """Fraction of the $2,220 discretionary pool assumed to be bid."""
    anchor_adherence: float
    """How closely the room follows a visible number, before format arguments."""
    spread: float
    """Reported uncertainty for this scenario, as a fraction.

    **Descriptive only.** It is not applied to prices: the low/base/high band is
    the three scenarios themselves, and multiplying on top of them produced a
    high board the room could not pay."""
    description: str

    def __post_init__(self) -> None:
        if not 0.0 < self.spend_rate <= 1.0:
            raise ValueError("spend_rate must be in (0, 1]")
        if not 0.0 <= self.anchor_adherence <= 1.0:
            raise ValueError("anchor_adherence must be in [0, 1]")
        if not 0.0 <= self.spread < 1.0:
            raise ValueError("spread must be in [0, 1)")

    def cache_key(self) -> Tuple:
        return (self.name, self.spend_rate, self.anchor_adherence, self.spread)

    def to_dict(self) -> Dict[str, object]:
        return {"name": self.name, "spend_rate": self.spend_rate,
                "anchor_adherence": self.anchor_adherence,
                "spread": self.spread, "description": self.description,
                "basis": "STATED SCENARIO, not fitted; no historical auction exists"}


#: The committed scenario set. ``base`` is not "correct" -- it is the middle of
#: a range nobody has measured.
MARKET_SCENARIOS: Dict[str, MarketScenario] = {
    "low": MarketScenario(
        name="low", spend_rate=0.88, anchor_adherence=0.55, spread=0.30,
        description=("Modest unused money and format-aware managers who "
                     "discount a list built for another lineup.")),
    "base": MarketScenario(
        name="base", spend_rate=0.95, anchor_adherence=0.75, spread=0.25,
        description=("Near-full room spending with moderate anchor "
                     "adherence.")),
    "high": MarketScenario(
        name="high", spend_rate=1.00, anchor_adherence=0.92, spread=0.22,
        description=("The whole discretionary pool is bid and managers follow "
                     "the visible Sleeper number closely.")),
}


# ---------------------------------------------------------------------------
# Per-player output
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlayerPrior:
    """One player's expected clearing price, with every stage kept apart."""

    canonical_key: str
    player_id: Optional[int]
    position: str
    sleeper_player_id: Optional[str]
    raw_value: Optional[Decimal]
    display_anchor: Optional[int]
    credibility: Optional[AnchorCredibility]
    format_adjusted: Optional[float]
    """Anchor after the lineup-format argument, before any budget scaling."""
    reconciliation_factor: float
    base_price: int
    low_price: int
    high_price: int
    draftable: bool
    status: str
    notes: str = ""

    @property
    def band_width(self) -> int:
        return self.high_price - self.low_price

    @property
    def uncertainty(self) -> float:
        """Band width relative to the point estimate. Higher is less certain."""
        return self.band_width / self.base_price if self.base_price else float("inf")

    def to_dict(self, *, include_identity: bool = False) -> Dict[str, object]:
        out: Dict[str, object] = {
            "position": self.position,
            "raw_value": None if self.raw_value is None else str(self.raw_value),
            "display_anchor": self.display_anchor,
            "credibility": self.credibility.to_dict() if self.credibility else None,
            "format_adjusted": (None if self.format_adjusted is None
                                else round(self.format_adjusted, 3)),
            "reconciliation_factor": round(self.reconciliation_factor, 5),
            "expected_clearing_price": {"low": self.low_price,
                                        "base": self.base_price,
                                        "high": self.high_price},
            "band_width": self.band_width,
            "uncertainty": round(self.uncertainty, 4),
            "draftable": self.draftable,
            "status": self.status,
            "notes": self.notes,
        }
        if include_identity:
            out["canonical_key"] = self.canonical_key
            out["player_id"] = self.player_id
            out["sleeper_player_id"] = self.sleeper_player_id
        return out


@dataclass(frozen=True)
class MarketPrior:
    """Expected clearing prices for a whole board, under one scenario set."""

    players: Tuple[PlayerPrior, ...]
    scenarios: Dict[str, MarketScenario]
    budget: RoomBudget
    credibility: Dict[str, AnchorCredibility]
    anchor_fingerprint: str
    anchor_provenance: Dict[str, object]
    reconciliation: Dict[str, Dict[str, object]]
    notes: str = ""

    @property
    def by_key(self) -> Dict[str, PlayerPrior]:
        return {p.canonical_key: p for p in self.players}

    @property
    def draftable(self) -> Tuple[PlayerPrior, ...]:
        return tuple(p for p in self.players if p.draftable)

    @property
    def anchor_display_total(self) -> int:
        """What the board would cost at Sleeper's displayed prices.

        Printed beside the discretionary pool because the comparison is the
        whole argument for a reconciliation step existing.
        """
        return sum(p.display_anchor or 0 for p in self.draftable)

    def scenario_total(self, scenario: str) -> int:
        attr = {"low": "low_price", "base": "base_price", "high": "high_price"}[scenario]
        return sum(getattr(p, attr) for p in self.draftable)

    def positional_summary(self) -> Dict[str, Dict[str, object]]:
        out: Dict[str, Dict[str, object]] = {}
        for pos in ROSTERABLE_POSITIONS:
            group = [p for p in self.draftable if p.position == pos]
            if not group:
                continue
            # Totals, not a mean of per-player ratios: the latter is dominated
            # by cheap players whose $1 anchor becomes $2, which says nothing
            # about whether the position as a whole moved.
            anchored = [p for p in group if p.display_anchor]
            anchor_sum = sum(p.display_anchor for p in anchored)
            base_sum = sum(p.base_price for p in anchored)
            shift = (base_sum / anchor_sum) if anchor_sum else None
            out[pos] = {
                "players": len(group),
                "anchor_display_total": sum(p.display_anchor or 0 for p in group),
                "base_total": sum(p.base_price for p in group),
                "base_over_anchor": None if shift is None else round(shift, 4),
                "credibility_weight": round(self.credibility[pos].weight, 4),
                "credibility_learn_rate": round(self.credibility[pos].learn_rate, 4),
            }
        return out

    def fingerprint(self) -> str:
        """Digest of anchors, scenarios and credibility together.

        Any of the three changing must change every cost book built from this.
        """
        h = hashlib.sha256()
        h.update(f"anchors={self.anchor_fingerprint}\n".encode())
        for name in sorted(self.scenarios):
            h.update(f"{name}={self.scenarios[name].cache_key()}\n".encode())
        for pos in sorted(self.credibility):
            c = self.credibility[pos]
            h.update(f"{pos}={c.weight:.6f}:{c.learn_rate:.6f}\n".encode())
        h.update(f"budget={self.budget.to_dict()}\n".encode())
        return h.hexdigest()[:16]

    def summary(self) -> Dict[str, object]:
        """Aggregates only. Safe to commit; carries no player rows."""
        return {
            "fingerprint": self.fingerprint(),
            "anchor_provenance": self.anchor_provenance,
            "budget": self.budget.to_dict(),
            "scenarios": {k: v.to_dict() for k, v in self.scenarios.items()},
            "credibility": {k: v.to_dict() for k, v in self.credibility.items()},
            "reconciliation": self.reconciliation,
            "priced_players": len(self.draftable),
            "anchor_display_total": self.anchor_display_total,
            "scenario_totals": {s: self.scenario_total(s)
                                for s in ("low", "base", "high")},
            "positional": self.positional_summary(),
            "label": ("EXPECTED CLEARING PRICE under stated scenarios. NOT "
                      "championship-equity value, NOT a reservation price, NOT "
                      "a max bid."),
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Building it
# ---------------------------------------------------------------------------


def _format_adjust(anchor: SleeperAnchor, cred: AnchorCredibility,
                   position_center: float) -> float:
    """Shrink a player toward his position's centre by one minus credibility.

    Full credibility leaves the anchor alone. Zero credibility says the room
    ignores the list entirely and treats everyone at the position alike, which
    is the honest limit of "this number was computed for another lineup". In
    between the anchor's *relative* information is kept and its magnitude is
    discounted, which is what a manager who half-believes a number does.

    **Geometric, not arithmetic.** Shrinking a $1 player toward a $19 positional
    mean triples him, which is not what partial disbelief in a list means -- the
    room does not decide a waiver-wire tight end is worth three times more
    because the list is unreliable. A weighted geometric mean shrinks
    multiplicatively, so cheap players stay cheap and expensive ones come down,
    which is the shape a credibility discount actually has. Strictly increasing
    in the raw value, so it cannot reorder a position.

    Shrinking toward a centre rather than subtracting a fixed penalty is what
    makes this learnable at all: moving credibility later moves every player at
    the position coherently, where a hand-coded discount could never be updated
    by evidence.
    """
    raw = max(float(anchor.raw_value), 1e-6)
    return math.exp(cred.weight * math.log(raw)
                    + (1.0 - cred.weight) * math.log(max(position_center, 1e-6)))


def build_market_prior(
    book: AnchorBook,
    *,
    settings: LeagueSettings = DEFAULT_LEAGUE,
    scenarios: Optional[Mapping[str, MarketScenario]] = None,
    player_ids: Optional[Mapping[str, int]] = None,
    notes: str = "",
) -> MarketPrior:
    """Anchor -> format adjustment -> budget reconciliation -> price band.

    The reconciliation is a single monotone scale per scenario: every
    format-adjusted price is multiplied by the same factor, chosen so the
    draftable board sums to that scenario's spendable dollars once the $1 floor
    is honoured. Monotone by construction, so it cannot reorder players, and one
    number a reader can check rather than a per-player fudge.

    The alternative considered was a replacement-level transformation --
    subtracting a positional baseline before scaling, which spreads the cut onto
    cheap players and steepens the top. It is defensible and it is *not* used
    here, because it needs a replacement level that nothing in this repository
    measures, and inventing one would bury a second unfitted assumption inside a
    step whose whole job is to be inspectable. Recorded in the notes so the
    choice is visible rather than implicit.
    """
    scenarios = dict(scenarios or MARKET_SCENARIOS)
    for required in ("low", "base", "high"):
        if required not in scenarios:
            raise ValueError(f"a market prior needs a {required!r} scenario")
    budget = RoomBudget(settings)
    player_ids = player_ids or {}

    cred = _format_credibility(settings, scenarios["base"].anchor_adherence)
    priced = book.priced
    if not priced:
        raise ValueError("the anchor book has no priced, rosterable players")

    # Positional centres, from the anchors themselves. Geometric, to match the
    # geometric shrinkage: the arithmetic mean of a long-tailed price list sits
    # well above the typical player and would pull the bottom of the board up.
    pos_center: Dict[str, float] = {}
    for pos, group in book.by_position().items():
        if group:
            logs = [math.log(max(float(a.raw_value), 1e-6)) for a in group]
            pos_center[pos] = math.exp(sum(logs) / len(logs))

    adjusted: Dict[str, float] = {}
    for a in priced:
        adjusted[a.canonical_key] = _format_adjust(a, cred[a.position],
                                                   pos_center[a.position])

    # --- budget reconciliation, per scenario -------------------------------
    #
    # The board holds `roster_slots` players. The priced anchors cover only some
    # of them; the rest are minimum-price fills. Those fills consume their $1
    # from the committed floor, so the money available to *separate* the priced
    # players is the scenario's spendable discretionary pool.
    n_priced = len(priced)
    unpriced_slots = budget.roster_slots - n_priced
    if unpriced_slots < 0:
        raise ValueError(
            f"{n_priced} priced anchors exceed the {budget.roster_slots} roster "
            f"slots this league has")

    reconciliation: Dict[str, Dict[str, object]] = {}
    factors: Dict[str, float] = {}
    adj_total = sum(adjusted.values())
    for name, sc in scenarios.items():
        # Each priced player is bought for at least $1, which comes out of the
        # committed floor. What the scenario's discretionary dollars buy is the
        # amount ABOVE $1, so that is what the factor is solved against.
        spendable = budget.spendable(sc.spend_rate)
        headroom = adj_total - n_priced
        factor = 1.0 if headroom <= 0 else (spendable - n_priced) / headroom
        factor = max(factor, 0.0)
        factors[name] = factor
        reconciliation[name] = {
            "spend_rate": sc.spend_rate,
            "spendable_discretionary": round(spendable, 2),
            "anchor_raw_total": str(book.priced_raw_total),
            "format_adjusted_total": round(adj_total, 2),
            "reconciliation_factor": round(factor, 5),
            "priced_players": n_priced,
            "minimum_price_fills": unpriced_slots,
            "floor_dollars_for_fills": unpriced_slots * settings.min_bid,
        }

    # Filled in below, once the integer prices exist. Rounding 149 prices to
    # whole dollars cannot hit a target exactly, so the achieved total is
    # reported beside the target rather than asserted equal to it.
    rounding_slack = max(1, n_priced // 2)

    def scale(key: str, name: str) -> int:
        """Scaled price, never below the legal minimum bid."""
        value = settings.min_bid + (adjusted[key] - 1.0) * factors[name]
        return max(settings.min_bid, int(round(value)))

    players: List[PlayerPrior] = []
    for a in book.anchors:
        key = a.canonical_key
        if a.counts_toward_room_budget:
            players.append(PlayerPrior(
                canonical_key=key, player_id=player_ids.get(key),
                position=a.position, sleeper_player_id=a.sleeper_player_id,
                raw_value=a.raw_value, display_anchor=a.display_anchor,
                credibility=cred[a.position],
                format_adjusted=adjusted[key],
                reconciliation_factor=factors["base"],
                base_price=scale(key, "base"), low_price=scale(key, "low"),
                high_price=scale(key, "high"), draftable=True,
                status="priced"))
        else:
            # Not a predicted $1 sale: a player with no usable anchor. The two
            # are different claims and are kept apart.
            players.append(PlayerPrior(
                canonical_key=key, player_id=player_ids.get(key),
                position=a.position, sleeper_player_id=a.sleeper_player_id,
                raw_value=a.raw_value, display_anchor=a.display_anchor,
                credibility=None, format_adjusted=None,
                reconciliation_factor=1.0, base_price=0, low_price=0,
                high_price=0, draftable=False, status=a.status,
                notes=("no usable anchor; UNPRICED, which is not the same as a "
                       "predicted $1 sale")))

    # The band is the three scenarios themselves, not a widening applied on top
    # of them. An earlier version multiplied each scenario price by a further
    # spread, which pushed the high board to $2,705 against a $2,220 room -- a
    # band that cannot be paid is not a forecast. Only the ordering is enforced
    # here, since rounding can cross two adjacent scenarios by a dollar.
    fixed: List[PlayerPrior] = []
    for p in players:
        if not p.draftable:
            fixed.append(p)
            continue
        lo = max(settings.min_bid, min(p.low_price, p.base_price, p.high_price))
        hi = max(lo, p.low_price, p.base_price, p.high_price)
        base = min(max(p.base_price, lo), hi)
        fixed.append(replace(p, low_price=lo, base_price=base, high_price=hi))

    for name in scenarios:
        attr = {"low": "low_price", "base": "base_price", "high": "high_price"}[name]
        achieved = sum(getattr(p, attr) for p in fixed if p.draftable)
        target = reconciliation[name]["spendable_discretionary"]
        reconciliation[name]["achieved_board_total"] = achieved
        reconciliation[name]["overspend_vs_target"] = round(achieved - target, 2)
        reconciliation[name]["within_integer_rounding"] = (
            achieved - target <= rounding_slack)

    return MarketPrior(
        players=tuple(fixed), scenarios=scenarios, budget=budget,
        credibility=cred, anchor_fingerprint=book.fingerprint(),
        anchor_provenance=book.provenance(), reconciliation=reconciliation,
        notes=(notes + " Reconciliation is a single monotone scale per scenario. "
               "A replacement-level transformation was considered and rejected: "
               "it needs a baseline nothing here measures.").strip())


def format_prior_summary(prior: MarketPrior, width: int = 90) -> str:
    """Sanitized rendering: aggregates only, no player rows."""
    bar = "=" * width
    s = prior.summary()
    b = prior.budget
    out = [bar, "EXPECTED CLEARING-PRICE PRIOR (provisional)", bar,
           f"anchor source   {prior.anchor_provenance['endpoint']}",
           f"                format={prior.anchor_provenance['format']} "
           f"season={prior.anchor_provenance['season']}",
           f"anchor digest   {prior.anchor_fingerprint}",
           f"prior digest    {prior.fingerprint()}",
           "",
           "THE ANCHOR IS NOT A PRICE. It is a generic Sleeper 2qb number for a",
           "different lineup. This table is an EXPECTED CLEARING PRICE under",
           "stated scenarios -- not championship equity, not a reservation",
           "price, and not a bid.",
           "",
           "ROOM BUDGET",
           f"  nominal total          ${b.nominal_total}",
           f"  roster slots           {b.roster_slots}",
           f"  committed at $1/slot   ${b.minimum_committed}",
           f"  discretionary          ${b.discretionary}",
           f"  priced anchors         {len(prior.draftable)} totalling "
           f"${prior.anchor_display_total} at list",
           f"  minimum-price fills    {b.roster_slots - len(prior.draftable)}",
           "",
           "  Paying list is not merely unlikely, it is arithmetically",
           "  impossible: the priced anchors alone exceed the discretionary",
           "  pool before the remaining slots are filled.",
           ""]

    head = f"  {'scenario':<8}{'spend':>7}{'adherence':>11}{'factor':>9}{'board $':>10}"
    out += ["RECONCILIATION", head, "  " + "-" * (len(head) - 2)]
    for name in ("low", "base", "high"):
        r = prior.reconciliation[name]
        sc = prior.scenarios[name]
        out.append(f"  {name:<8}{sc.spend_rate:>7.2f}{sc.anchor_adherence:>11.2f}"
                   f"{r['reconciliation_factor']:>9.4f}"
                   f"{prior.scenario_total(name):>10}")
    out += ["  " + "-" * (len(head) - 2),
            "  Every coefficient is a STATED SCENARIO. No historical auction",
            "  for this league exists, so none of them is fitted.", ""]

    head2 = (f"  {'pos':<5}{'players':>9}{'anchor $':>10}{'base $':>9}"
             f"{'base/anchor':>13}{'credibility':>13}{'learn':>8}")
    out += ["BY POSITION", head2, "  " + "-" * (len(head2) - 2)]
    for pos, v in prior.positional_summary().items():
        ratio = v["base_over_anchor"]
        out.append(f"  {pos:<5}{v['players']:>9}{v['anchor_display_total']:>10}"
                   f"{v['base_total']:>9}"
                   f"{('n/a' if ratio is None else f'{ratio:.3f}'):>13}"
                   f"{v['credibility_weight']:>13.3f}"
                   f"{v['credibility_learn_rate']:>8.2f}")
    out += ["  " + "-" * (len(head2) - 2), ""]
    for pos in ROSTERABLE_POSITIONS:
        out.append(f"  {pos}: {prior.credibility[pos].reason}")
    out += ["",
            "  Credibility is a WEIGHT, not a discount. The tight-end anchor is",
            "  preserved in full; what is low is how far this room is assumed to",
            "  follow it, and its high learn rate means the first few tight ends",
            "  sold will move it quickly in either direction.", bar]
    return "\n".join(out)


