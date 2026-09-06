"""The greatest price at which buying still beats the best alternative.

Not obtained by dividing a CE difference by a dollars-per-CE constant. There is
no such constant. At every price the remaining budget is different, so the best
alternative roster is different, so the comparison is a *different question* at
$18 than at $17 -- and the answer can jump when one more dollar spent here puts
some other player out of reach. That is a real discontinuity in the auction, not
noise, and a global conversion factor would erase it.

So the price domain is searched, and each price gets its own buy-versus-pass
comparison with its own interval. A price is:

``favorable``    the lower 95% bound on the CE difference is at or above zero;
``unfavorable``  the upper bound is below zero;
``unresolved``   the interval contains zero.

Two prices come out, and they answer different questions:

**Robust reservation price** -- the greatest price favorable in *every* modelled
scenario. If one exists, buying at or below it beats the alternative however the
four unresolved assumptions turn out.

**Permissive reservation price** -- the greatest price not demonstrably
unfavorable in *at least one* scenario. Above it, every scenario says no.

Neither is a bid. The robust price is not an opening maximum and the permissive
price is not a recommendation; both are CE reservation prices under stated
completion-cost assumptions and a stated model scenario set. What the room will
actually pay is a different quantity this package does not estimate, and what to
bid given who else is still able to bid is a third.

Monotonicity is checked, never assumed. Violations are reported and classified,
because a jump caused by a player becoming unaffordable is information and a
jump caused by Monte Carlo noise is not.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .completion import ComparisonCast, CompletionSettings
from .costs import CostBook
from .counterfactual import BuyPassResult, PassDestination, compare_buy_vs_pass
from .proxy import ProxyEvaluator
from .state import AuctionRuleError, AuctionState

__all__ = [
    "ScenarioSetup",
    "PricePoint",
    "MonotonicityViolation",
    "ScenarioReservation",
    "ReservationResult",
    "ReservationCache",
    "price_ladder",
    "search_reservation",
    "estimate_runtime",
    "format_reservation",
]


@dataclass(frozen=True)
class ScenarioSetup:
    """One model scenario's view of the same auction.

    Scenarios change the *players* -- calibration produces different specs
    under different assumptions -- so each carries its own state. Building
    those is the caller's job; this module refuses to guess how a scenario
    reaches an auction state.
    """

    scenario_id: str
    state: AuctionState
    cast: ComparisonCast
    costs: CostBook

    def cache_key(self) -> Tuple:
        return (self.scenario_id, self.state.fingerprint(),
                self.costs.provenance.source, self.costs.provenance.version,
                self.costs.level, len(self.costs))


@dataclass(frozen=True)
class PricePoint:
    """One price's verdict in one scenario."""

    price: int
    delta_ce: float
    delta_ce_se: float
    verdict: str
    ce_buy: float
    ce_pass: float
    added_in_pass: Tuple[int, ...]
    """What the money buys instead. Its changing is what makes a jump real."""
    resolved: bool

    @property
    def ci95(self) -> Tuple[float, float]:
        if math.isnan(self.delta_ce_se):
            return (float("nan"), float("nan"))
        half = 1.96 * self.delta_ce_se
        return (self.delta_ce - half, self.delta_ce + half)

    def to_dict(self) -> Dict[str, object]:
        lo, hi = self.ci95
        return {"price": self.price, "delta_ce": round(self.delta_ce, 6),
                "se": round(self.delta_ce_se, 6),
                "ci95": [round(lo, 6), round(hi, 6)],
                "verdict": self.verdict, "resolved": self.resolved,
                "ce_buy": round(self.ce_buy, 6),
                "ce_pass": round(self.ce_pass, 6),
                "alternative_purchases": list(self.added_in_pass)}


@dataclass(frozen=True)
class MonotonicityViolation:
    """A price where paying more looked *better*, and the best guess at why."""

    lower_price: int
    higher_price: int
    lower_delta: float
    higher_delta: float
    cause: str
    """``"monte carlo"``, ``"completion changed"`` or ``"unexplained"``.

    ``completion changed`` means the alternative roster genuinely differs
    between the two prices -- a dollar moved somebody out of reach -- which is
    a real discontinuity. ``monte carlo`` means the two intervals overlap.
    ``unexplained`` means neither, which is the one worth investigating: it
    suggests the bounded search found a better roster at the higher price than
    at the lower one, i.e. search instability rather than economics.
    """

    def to_dict(self) -> Dict[str, object]:
        return {"lower_price": self.lower_price, "higher_price": self.higher_price,
                "lower_delta": round(self.lower_delta, 6),
                "higher_delta": round(self.higher_delta, 6),
                "cause": self.cause}


@dataclass
class ScenarioReservation:
    """Every searched price in one scenario, and where the line falls."""

    scenario_id: str
    points: Tuple[PricePoint, ...]
    violations: Tuple[MonotonicityViolation, ...] = ()

    @property
    def favorable_prices(self) -> Tuple[int, ...]:
        return tuple(p.price for p in self.points if p.verdict == "favorable")

    @property
    def unfavorable_prices(self) -> Tuple[int, ...]:
        return tuple(p.price for p in self.points if p.verdict == "unfavorable")

    @property
    def unresolved_prices(self) -> Tuple[int, ...]:
        return tuple(p.price for p in self.points if p.verdict == "unresolved")

    @property
    def favorable_frontier(self) -> Optional[int]:
        """Greatest searched price whose interval sits at or above zero."""
        fav = self.favorable_prices
        return max(fav) if fav else None

    @property
    def unresolved_region(self) -> Optional[Tuple[int, int]]:
        u = self.unresolved_prices
        return (min(u), max(u)) if u else None

    def verdict_at(self, price: int) -> Optional[str]:
        for p in self.points:
            if p.price == price:
                return p.verdict
        return None

    def to_dict(self) -> Dict[str, object]:
        return {
            "scenario_id": self.scenario_id,
            "favorable_frontier": self.favorable_frontier,
            "unresolved_region": list(self.unresolved_region)
            if self.unresolved_region else None,
            "n_favorable": len(self.favorable_prices),
            "n_unfavorable": len(self.unfavorable_prices),
            "n_unresolved": len(self.unresolved_prices),
            "monotonicity_violations": [v.to_dict() for v in self.violations],
            "points": [p.to_dict() for p in self.points],
        }


@dataclass
class ReservationResult:
    """The scenario-aware reservation range, and what it is not."""

    candidate_id: int
    focus_owner_id: str
    auction_fingerprint: str
    legal_max_bid: int
    pass_destination: PassDestination
    prices_searched: Tuple[int, ...]
    per_scenario: Dict[str, ScenarioReservation]
    scenarios_run: Tuple[str, ...]
    scenarios_available: int
    cost_level: str
    n_sims: int
    is_heuristic: bool
    runtime_s: float
    settings: CompletionSettings
    notes: str = ""

    @property
    def full_grid(self) -> bool:
        return len(self.scenarios_run) == self.scenarios_available

    @property
    def robust_price(self) -> Optional[int]:
        """Greatest searched price favorable in EVERY scenario run."""
        best = None
        for price in self.prices_searched:
            if all(r.verdict_at(price) == "favorable"
                   for r in self.per_scenario.values()):
                best = price if best is None else max(best, price)
        return best

    @property
    def permissive_price(self) -> Optional[int]:
        """Greatest searched price not demonstrably unfavorable in some scenario."""
        best = None
        for price in self.prices_searched:
            if any(r.verdict_at(price) in ("favorable", "unresolved")
                   for r in self.per_scenario.values()):
                best = price if best is None else max(best, price)
        return best

    @property
    def scenario_band(self) -> Optional[Tuple[Optional[int], Optional[int]]]:
        """Lowest and highest per-scenario favorable frontier."""
        fronts = [r.favorable_frontier for r in self.per_scenario.values()]
        present = [f for f in fronts if f is not None]
        if not present:
            return None
        return (min(present) if len(present) == len(fronts) else None, max(present))

    @property
    def dominant_scenarios(self) -> Dict[str, Optional[int]]:
        return {sid: r.favorable_frontier for sid, r in self.per_scenario.items()}

    @property
    def all_violations(self) -> Tuple[MonotonicityViolation, ...]:
        return tuple(v for r in self.per_scenario.values() for v in r.violations)

    @property
    def binding_alternatives(self) -> Dict[int, List[int]]:
        """What the money buys instead, per price, in the first scenario run.

        Reported because "you are giving up these players" is the part of an
        opportunity cost a human can check.
        """
        if not self.scenarios_run:
            return {}
        first = self.per_scenario[self.scenarios_run[0]]
        return {p.price: list(p.added_in_pass) for p in first.points}

    @property
    def result_kind(self) -> str:
        if self.robust_price is not None:
            return "heuristic" if self.is_heuristic else "exact"
        if self.permissive_price is not None:
            return "unresolved (no price is favorable in every scenario)"
        return "unfavorable at every searched price"

    def to_dict(self) -> Dict[str, object]:
        band = self.scenario_band
        return {
            "candidate_id": self.candidate_id,
            "focus_owner_id": self.focus_owner_id,
            "auction_fingerprint": self.auction_fingerprint,
            "legal_max_bid": self.legal_max_bid,
            "pass_destination": self.pass_destination.to_dict(),
            "prices_searched": list(self.prices_searched),
            "scenarios_run": list(self.scenarios_run),
            "scenarios_available": self.scenarios_available,
            "full_grid": self.full_grid,
            "cost_level": self.cost_level,
            "n_sims": self.n_sims,
            "robust_reservation_price": self.robust_price,
            "permissive_reservation_price": self.permissive_price,
            "scenario_band": list(band) if band else None,
            "per_scenario_frontier": self.dominant_scenarios,
            "monotonicity_violations": [v.to_dict() for v in self.all_violations],
            "result_kind": self.result_kind,
            "search_is": "heuristic" if self.is_heuristic else "exact",
            "runtime_s": round(self.runtime_s, 1),
            "settings": self.settings.to_dict(),
            "per_scenario": {k: v.to_dict() for k, v in self.per_scenario.items()},
            "label": ("CE reservation-price range under the stated "
                      "completion-cost and model scenarios. NOT an opening max "
                      "bid, NOT a recommended bid, NOT a clearing-price "
                      "prediction."),
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


class ReservationCache:
    """Buy/pass results, keyed by everything that can change one.

    The key is deliberately paranoid. A cached CE from one scenario served for
    another, or from one pass destination served for another, would be a silent
    wrong answer of exactly the kind this package exists to avoid -- so the key
    carries the auction fingerprint, focus owner, candidate, price, pass
    destination, cost book identity, scenario, search settings, simulation
    count and seed.
    """

    def __init__(self) -> None:
        self._store: Dict[Tuple, BuyPassResult] = {}
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(setup: ScenarioSetup, candidate_id: int, price: int,
            dest: PassDestination, settings: CompletionSettings) -> Tuple:
        return (setup.cache_key(), setup.state.focus_owner_id, candidate_id,
                int(price), dest.cache_key(), settings.cache_key(),
                settings.ce_sims, settings.ce_seed)

    def get(self, k: Tuple) -> Optional[BuyPassResult]:
        hit = self._store.get(k)
        if hit is None:
            self.misses += 1
        else:
            self.hits += 1
        return hit

    def put(self, k: Tuple, value: BuyPassResult) -> None:
        self._store[k] = value

    def __len__(self) -> int:
        return len(self._store)

    def stats(self) -> Dict[str, int]:
        return {"entries": len(self._store), "hits": self.hits,
                "misses": self.misses}


# ---------------------------------------------------------------------------
# The search
# ---------------------------------------------------------------------------


def price_ladder(min_bid: int, legal_max: int, max_prices: int = 12) -> Tuple[int, ...]:
    """A deterministic set of prices spanning the legal range.

    Every price when the range is short enough, otherwise an evenly spaced
    ladder that always includes both endpoints. Evenly spaced rather than
    adaptive on purpose: an adaptive ladder that refined around a sign change
    would assume the sign changes once, and monotonicity is the thing being
    checked rather than assumed.
    """
    if legal_max < min_bid:
        return ()
    span = legal_max - min_bid + 1
    if span <= max_prices:
        return tuple(range(min_bid, legal_max + 1))
    step = (legal_max - min_bid) / (max_prices - 1)
    return tuple(sorted({int(round(min_bid + i * step))
                         for i in range(max_prices)}))


def _classify(lo: PricePoint, hi: PricePoint) -> str:
    if set(lo.added_in_pass) != set(hi.added_in_pass):
        return "completion changed"
    lo_lo, lo_hi = lo.ci95
    hi_lo, hi_hi = hi.ci95
    if not (math.isnan(lo_lo) or math.isnan(hi_lo)) and lo_lo <= hi_hi and hi_lo <= lo_hi:
        return "monte carlo"
    return "unexplained"


def estimate_runtime(n_scenarios: int, n_prices: int,
                     seconds_per_comparison: float) -> Dict[str, float]:
    """What a given grid would cost, before running it."""
    total = n_scenarios * n_prices * seconds_per_comparison
    return {"comparisons": n_scenarios * n_prices,
            "seconds_per_comparison": seconds_per_comparison,
            "estimated_seconds": total,
            "estimated_minutes": total / 60.0,
            "estimated_hours": total / 3600.0}


def search_reservation(
    setups: Sequence[ScenarioSetup],
    candidate_id: int,
    pass_destination: PassDestination,
    *,
    prices: Optional[Sequence[int]] = None,
    max_prices: int = 12,
    settings: CompletionSettings = CompletionSettings(),
    scenarios_available: Optional[int] = None,
    default_cost: Optional[int] = None,
    cache: Optional[ReservationCache] = None,
    notes: str = "",
    progress=None,
) -> ReservationResult:
    """Search the legal price range in every supplied scenario.

    ``scenarios_available`` is the size of the full grid, so a reduced run
    reports itself as reduced. Leaving it unset assumes the supplied scenarios
    *are* the whole grid, which is only true when they are.
    """
    if not setups:
        raise ValueError("a reservation search needs at least one scenario setup")
    t0 = time.perf_counter()
    cache = cache if cache is not None else ReservationCache()
    base = setups[0]
    focus_id = base.state.focus_owner_id
    owner = base.state.owner(focus_id)
    legal_max = owner.max_bid
    if legal_max < owner.min_bid:
        raise AuctionRuleError(
            f"{focus_id} cannot legally bid at all "
            f"({owner.open_slots} open slot(s), ${owner.budget_remaining})")

    ladder = tuple(sorted(set(int(p) for p in prices))) if prices is not None \
        else price_ladder(owner.min_bid, legal_max, max_prices)
    if not ladder:
        raise ValueError("the price ladder is empty")
    over = [p for p in ladder if p > legal_max or p < owner.min_bid]
    if over:
        raise AuctionRuleError(
            f"prices {over} are outside {focus_id}'s legal range "
            f"[{owner.min_bid}, {legal_max}]")

    per_scenario: Dict[str, ScenarioReservation] = {}
    heuristic = False
    total = len(setups) * len(ladder)
    done = 0
    for setup in setups:
        proxy = ProxyEvaluator(setup.state.pool, setup.state.settings,
                               settings.proxy_reps, settings.proxy_seed)
        points: List[PricePoint] = []
        for price in ladder:
            k = ReservationCache.key(setup, candidate_id, price,
                                     pass_destination, settings)
            res = cache.get(k)
            if res is None:
                res = compare_buy_vs_pass(
                    setup.state, setup.cast, setup.costs, candidate_id, price,
                    pass_destination, settings=settings,
                    scenario_id=setup.scenario_id, default_cost=default_cost,
                    proxy=proxy)
                cache.put(k, res)
            heuristic = heuristic or res.is_heuristic
            points.append(PricePoint(
                price=price, delta_ce=res.delta_ce, delta_ce_se=res.delta_ce_se,
                verdict=res.verdict, ce_buy=res.ce_buy, ce_pass=res.ce_pass,
                added_in_pass=tuple(res.pass_.best.added) if res.pass_.best else (),
                resolved=res.resolved))
            done += 1
            if progress is not None:
                progress(done, total, setup.scenario_id, price)

        violations = []
        for a, b in zip(points, points[1:]):
            # Paying more can only be worse, so delta must not rise with price.
            if b.delta_ce > a.delta_ce:
                violations.append(MonotonicityViolation(
                    a.price, b.price, a.delta_ce, b.delta_ce, _classify(a, b)))
        per_scenario[setup.scenario_id] = ScenarioReservation(
            setup.scenario_id, tuple(points), tuple(violations))

    return ReservationResult(
        candidate_id=candidate_id, focus_owner_id=focus_id,
        auction_fingerprint=base.state.fingerprint(), legal_max_bid=legal_max,
        pass_destination=pass_destination, prices_searched=ladder,
        per_scenario=per_scenario,
        scenarios_run=tuple(s.scenario_id for s in setups),
        scenarios_available=scenarios_available if scenarios_available is not None
        else len(setups),
        cost_level=base.costs.level, n_sims=settings.ce_sims,
        is_heuristic=heuristic, runtime_s=time.perf_counter() - t0,
        settings=settings, notes=notes)


def format_reservation(result: ReservationResult, width: int = 92,
                       cost_disclaimer: str = "") -> str:
    """Sanitized rendering. Never prints the words max bid or recommended bid."""
    bar = "=" * width
    out = [bar, "CE RESERVATION-PRICE RANGE", bar,
           f"candidate       id {result.candidate_id}",
           f"focus owner     {result.focus_owner_id}",
           f"auction state   {result.auction_fingerprint}",
           f"legal ceiling   ${result.legal_max_bid} "
           f"(the rules' limit, not a valuation)",
           f"if we pass      {result.pass_destination.label}",
           f"cost source     {result.cost_level}"]
    if cost_disclaimer:
        out.append(f"                {cost_disclaimer}")
    out += [f"seasons         {result.n_sims:,} per branch, matched",
            f"scenarios       {len(result.scenarios_run)} of "
            f"{result.scenarios_available}",
            f"prices          {list(result.prices_searched)}", ""]
    if not result.full_grid:
        out += [f"*** REDUCED SCENARIO GRID: {len(result.scenarios_run)} of "
                f"{result.scenarios_available} cells were run.",
                f"    This is NOT the full cross-product. ***", ""]

    for sid in result.scenarios_run:
        r = result.per_scenario[sid]
        out.append(f"SCENARIO {sid}")
        head = (f"    {'price':>7}{'dCE':>11}{'se':>9}{'95% CI':>22}"
                f"{'verdict':>13}")
        out += [head, "    " + "-" * (len(head) - 4)]
        for p in r.points:
            lo, hi = p.ci95
            out.append(f"    ${p.price:>6}{p.delta_ce:>+11.5f}{p.delta_ce_se:>9.5f}"
                       f"  [{lo:+.5f}, {hi:+.5f}]{p.verdict:>13}")
        out += ["    " + "-" * (len(head) - 4),
                f"    favorable up to   "
                + (f"${r.favorable_frontier}" if r.favorable_frontier is not None
                   else "no searched price is favorable"),
                f"    unresolved band   "
                + (f"${r.unresolved_region[0]}-${r.unresolved_region[1]}"
                   if r.unresolved_region else "none")]
        if r.violations:
            out.append(f"    monotonicity      {len(r.violations)} violation(s):")
            for v in r.violations:
                out.append(f"      ${v.lower_price} -> ${v.higher_price}: "
                           f"{v.lower_delta:+.5f} -> {v.higher_delta:+.5f} "
                           f"[{v.cause}]")
        out.append("")

    out += [bar, "RESERVATION RANGE", bar,
            f"  robust      "
            + (f"${result.robust_price}" if result.robust_price is not None
               else "none -- no searched price is favorable in every scenario"),
            "              the greatest price favorable in EVERY scenario run",
            f"  permissive  "
            + (f"${result.permissive_price}" if result.permissive_price is not None
               else "none -- every scenario says no at every searched price"),
            "              the greatest price not demonstrably unfavorable in",
            "              at least one scenario",
            ""]
    band = result.scenario_band
    if band:
        lo = f"${band[0]}" if band[0] is not None else "none in some scenario"
        out.append(f"  per-scenario frontier spans {lo} to ${band[1]}")
    out += [f"  result is    {result.result_kind}",
            f"  runtime      {result.runtime_s:.1f}s", ""]
    if result.all_violations:
        causes: Dict[str, int] = {}
        for v in result.all_violations:
            causes[v.cause] = causes.get(v.cause, 0) + 1
        out += ["  Monotonicity was checked, not assumed. "
                + ", ".join(f"{n} {c}" for c, n in sorted(causes.items())) + ".",
                "  A 'completion changed' jump is real economics: a dollar here",
                "  put somebody there out of reach.", ""]
    if result.is_heuristic:
        out += ["  Both branches at every price are BOUNDED SEARCHES. The range",
                "  is between the best rosters this search found.", ""]
    out += ["  This is a CE reservation-price range under the stated",
            "  completion-cost assumptions and model scenarios.",
            "  It is NOT an opening maximum, NOT a bid, and NOT a prediction of",
            "  what the room will pay.", bar]
    return "\n".join(out)
