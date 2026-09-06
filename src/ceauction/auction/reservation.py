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

**A sparse ladder cannot name a reservation price.** Testing $35 and then $50
tells you the frontier lies somewhere in $35-$49; it does not tell you it is
$35. Every result here reports the *highest tested favorable price*, the next
tested unfavorable price, the untested gap between them, and whether the ladder
was exhaustive. A definitive integer frontier requires every relevant integer to
have been evaluated, and ``refine=True`` will go and do that inside the
transition gap.

**Delta CE cannot legitimately rise with price.** With fixed acquisition costs
and a correctly optimised completion, every roster affordable after paying
``p + 1`` was also affordable at ``p``, so the buy branch's attainable maximum
is non-increasing in price; and the pass branch does not depend on our price at
all when the destination and the rival's price are fixed. A completion change
can therefore produce a genuine *downward* discontinuity -- a dollar here puts
somebody there out of reach -- but it can never make the true optimum improve
because we paid more. An observed increase is Monte Carlo noise, instability in
the bounded candidate search, instability in the CE selection, or a bug. It is
never beneficial economics, and an earlier version of this module described it
as such.
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
        """Identity by content, never by the caller's label.

        The audit found this keyed on the scenario id, the state fingerprint
        and the cost book's *metadata and length* -- so two books with the same
        provenance and different prices were interchangeable, and a state whose
        players had different projections was too. The label is kept because it
        is useful in output, but it is no longer load-bearing: every component
        below is a digest of actual content.
        """
        return (self.scenario_id, self.state.fingerprint(),
                self.cast.fingerprint(), self.costs.fingerprint())


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
    """What the money buys instead. A change here explains a downward step."""
    added_in_buy: Tuple[int, ...] = ()
    resolved: bool = False

    @property
    def ci95(self) -> Tuple[float, float]:
        """**Pointwise** 95% interval for this price in this scenario.

        Not a simultaneous band. A run covering many prices and many scenarios
        makes many such statements, and the chance that at least one of them is
        wrong grows with their number. See
        :meth:`ReservationResult.simultaneous_note`.
        """
        if math.isnan(self.delta_ce_se):
            return (float("nan"), float("nan"))
        half = 1.96 * self.delta_ce_se
        return (self.delta_ce - half, self.delta_ce + half)

    def to_dict(self) -> Dict[str, object]:
        lo, hi = self.ci95
        return {"price": self.price, "delta_ce": round(self.delta_ce, 6),
                "se": round(self.delta_ce_se, 6),
                "ci95_pointwise": [round(lo, 6), round(hi, 6)],
                "interval_is": "pointwise 95%, not simultaneous",
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
    """Why delta CE rose with price, which it should never truly do.

    ``"monte carlo"``          the two pointwise intervals overlap; the rise is
                               within sampling error.
    ``"search instability"``   the branches found different rosters at the two
                               prices and the rise is outside sampling error.
                               A dollar cannot buy a *better* attainable
                               optimum, so this is the bounded beam finding a
                               better roster at the higher price than it found
                               at the lower one.
    ``"ce-selection instability"``
                               same rosters, rise outside sampling error: the
                               finalist chosen by equity differed between the
                               two runs for reasons the sample cannot justify.
    ``"unexplained"``          none of the above, and worth investigating as a
                               possible bug.

    **None of these is beneficial economics.** With fixed costs and a correctly
    optimised completion the buy branch's attainable maximum is non-increasing
    in price and the pass branch does not depend on our price at all. A
    completion change can produce a genuine downward step; it cannot produce an
    upward one.
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

    min_bid: int = 1
    legal_max: int = 0

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
    def tested_prices(self) -> Tuple[int, ...]:
        return tuple(p.price for p in self.points)

    @property
    def is_exhaustive(self) -> bool:
        """Was every integer in the legal range actually evaluated?

        Only then can a frontier be stated rather than bracketed.
        """
        if self.legal_max < self.min_bid:
            return False
        return set(self.tested_prices) == set(range(self.min_bid,
                                                    self.legal_max + 1))

    @property
    def highest_tested_favorable(self) -> Optional[int]:
        """The greatest price *that was tested* and came back favorable.

        Deliberately not called a frontier. An earlier version reported this
        as the reservation price after testing $35 and then $50, which asserts
        something about $36-$49 that was never measured.
        """
        fav = self.favorable_prices
        return max(fav) if fav else None

    @property
    def next_tested_unfavorable(self) -> Optional[int]:
        """The lowest tested unfavorable price above the favorable ones."""
        base = self.highest_tested_favorable
        above = [p for p in self.unfavorable_prices
                 if base is None or p > base]
        return min(above) if above else None

    @property
    def reservation_bracket(self) -> Optional[Tuple[int, Optional[int]]]:
        """``(lower, upper)`` -- the frontier is at least ``lower``, below ``upper``.

        ``upper`` is ``None`` when nothing above was tested unfavorable, in
        which case the frontier is only known to be at or above ``lower``. When
        the ladder is exhaustive the bracket is tight: ``upper == lower + 1``.
        """
        lo = self.highest_tested_favorable
        if lo is None:
            return None
        return (lo, self.next_tested_unfavorable)

    @property
    def exact_frontier(self) -> Optional[int]:
        """The integer frontier, only when every relevant price was evaluated.

        ``None`` whenever the ladder left a gap, however small. A number here
        is a claim that no untested price could have changed it.
        """
        lo = self.highest_tested_favorable
        if lo is None:
            return None
        nxt = self.next_tested_unfavorable
        if self.is_exhaustive:
            return lo
        if nxt is not None and nxt == lo + 1:
            return lo
        return None

    @property
    def untested_intervals(self) -> Tuple[Tuple[int, int], ...]:
        """Every run of integers inside the legal range that was never tried."""
        if self.legal_max < self.min_bid:
            return ()
        tested = set(self.tested_prices)
        gaps, start = [], None
        for price in range(self.min_bid, self.legal_max + 1):
            if price in tested:
                if start is not None:
                    gaps.append((start, price - 1))
                    start = None
            elif start is None:
                start = price
        if start is not None:
            gaps.append((start, self.legal_max))
        return tuple(gaps)

    @property
    def transition_gap(self) -> Optional[Tuple[int, int]]:
        """The **untested** span between the last favorable and next unfavorable.

        This is what :func:`search_reservation` refines when asked to pin the
        frontier: everywhere else an untested price cannot move the answer as
        much as this one can.

        ``None`` once every integer in that span has been evaluated -- including
        when they came back *unresolved*, which is a measured verdict and not a
        gap. Treating an unresolved price as untested would make refinement
        loop forever trying to test something it had already tested.
        """
        lo = self.highest_tested_favorable
        hi = self.next_tested_unfavorable
        if lo is None or hi is None or hi <= lo + 1:
            return None
        tested = set(self.tested_prices)
        missing = [p for p in range(lo + 1, hi) if p not in tested]
        if not missing:
            return None
        return (min(missing), max(missing))

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
        bracket = self.reservation_bracket
        return {
            "scenario_id": self.scenario_id,
            "highest_tested_favorable": self.highest_tested_favorable,
            "next_tested_unfavorable": self.next_tested_unfavorable,
            "reservation_bracket": list(bracket) if bracket else None,
            "exact_frontier": self.exact_frontier,
            "ladder_is_exhaustive": self.is_exhaustive,
            "untested_intervals": [list(g) for g in self.untested_intervals],
            "transition_gap": list(self.transition_gap)
            if self.transition_gap else None,
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
    refined_prices: int = 0
    """Extra integer prices evaluated to pin the frontier."""
    notes: str = ""

    @property
    def full_grid(self) -> bool:
        return len(self.scenarios_run) == self.scenarios_available

    @property
    def is_exhaustive(self) -> bool:
        """Was every legal integer price evaluated in every scenario run?"""
        return all(r.is_exhaustive for r in self.per_scenario.values())

    @property
    def robust_tested_price(self) -> Optional[int]:
        """Highest **tested** price favorable in EVERY scenario run.

        Not a frontier unless :attr:`is_exhaustive`. With a sparse ladder the
        true frontier lies somewhere in :attr:`robust_bracket`.
        """
        best = None
        for price in self.prices_searched:
            if all(r.verdict_at(price) == "favorable"
                   for r in self.per_scenario.values()):
                best = price if best is None else max(best, price)
        return best

    #: Kept so existing callers keep working; the name is misleading on a
    #: sparse ladder, which is the point of the rename.
    @property
    def robust_price(self) -> Optional[int]:
        return self.robust_tested_price

    @property
    def robust_bracket(self) -> Optional[Tuple[int, Optional[int]]]:
        """Where the robust frontier lies: at least ``lower``, below ``upper``."""
        lo = self.robust_tested_price
        if lo is None:
            return None
        above = [p for p in self.prices_searched
                 if p > lo and any(r.verdict_at(p) == "unfavorable"
                                   for r in self.per_scenario.values())]
        return (lo, min(above) if above else None)

    @property
    def robust_exact_frontier(self) -> Optional[int]:
        """The robust frontier as an integer, only if it was actually pinned."""
        lo, hi = self.robust_bracket or (None, None)
        if lo is None:
            return None
        if self.is_exhaustive or (hi is not None and hi == lo + 1):
            return lo
        return None

    @property
    def permissive_tested_price(self) -> Optional[int]:
        """Highest **tested** price not demonstrably unfavorable somewhere."""
        best = None
        for price in self.prices_searched:
            if any(r.verdict_at(price) in ("favorable", "unresolved")
                   for r in self.per_scenario.values()):
                best = price if best is None else max(best, price)
        return best

    @property
    def permissive_price(self) -> Optional[int]:
        return self.permissive_tested_price

    @property
    def untested_intervals(self) -> Tuple[Tuple[int, int], ...]:
        """Untested integer runs, from the first scenario's ladder."""
        if not self.scenarios_run:
            return ()
        return self.per_scenario[self.scenarios_run[0]].untested_intervals

    @property
    def n_interval_statements(self) -> int:
        """How many pointwise 95% claims this run makes."""
        return sum(len(r.points) for r in self.per_scenario.values())

    def simultaneous_note(self) -> str:
        """Why the reported intervals are not a joint 95% band."""
        n = self.n_interval_statements
        return (f"Every interval here is POINTWISE 95%. This run makes {n} such "
                f"statements across {len(self.prices_searched)} price(s) and "
                f"{len(self.scenarios_run)} scenario(s), so the probability "
                f"that at least one is wrong is far above 5%. A simultaneous "
                f"band over prices and scenarios is NOT implemented; a "
                f"Bonferroni-adjusted alternative is available via "
                f"`bonferroni_intervals()` and is conservative rather than "
                f"exact.")

    def bonferroni_intervals(self) -> Dict[str, Dict[int, Tuple[float, float]]]:
        """Conservative simultaneous intervals, by splitting alpha evenly.

        Valid but wide: it ignores the heavy positive dependence between
        neighbouring prices, which share most of their rosters and all of their
        random numbers. Offered because a conservative honest band is better
        than an exact-sounding one that is not.
        """
        n = max(self.n_interval_statements, 1)
        # Two-sided Bonferroni: alpha/n per statement.
        alpha = 0.05 / n
        z = _normal_quantile(1.0 - alpha / 2.0)
        out: Dict[str, Dict[int, Tuple[float, float]]] = {}
        for sid, r in self.per_scenario.items():
            out[sid] = {p.price: (p.delta_ce - z * p.delta_ce_se,
                                  p.delta_ce + z * p.delta_ce_se)
                        for p in r.points}
        return out

    @property
    def scenario_band(self) -> Optional[Tuple[Optional[int], Optional[int]]]:
        """Lowest and highest per-scenario highest-tested-favorable price."""
        fronts = [r.highest_tested_favorable for r in self.per_scenario.values()]
        present = [f for f in fronts if f is not None]
        if not present:
            return None
        return (min(present) if len(present) == len(fronts) else None, max(present))

    @property
    def dominant_scenarios(self) -> Dict[str, Optional[int]]:
        return {sid: r.highest_tested_favorable
                for sid, r in self.per_scenario.items()}

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
        if self.robust_tested_price is None:
            if self.permissive_tested_price is not None:
                return "unresolved (no tested price is favorable in every scenario)"
            return "unfavorable at every tested price"
        pinned = self.robust_exact_frontier is not None
        search = "heuristic search" if self.is_heuristic else "exact search"
        ladder = "exact integer frontier" if pinned else "bracketed frontier"
        return f"{search}, {ladder}"

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
            "ladder_is_exhaustive": self.is_exhaustive,
            "refined_prices": self.refined_prices,
            "untested_intervals": [list(g) for g in self.untested_intervals],
            "robust_highest_tested_favorable": self.robust_tested_price,
            "robust_bracket": list(self.robust_bracket)
            if self.robust_bracket else None,
            "robust_exact_frontier": self.robust_exact_frontier,
            "permissive_highest_tested_favorable": self.permissive_tested_price,
            "interval_is": "pointwise 95%, not simultaneous",
            "n_interval_statements": self.n_interval_statements,
            "simultaneous_band": "not implemented; see simultaneous_note()",
            "simultaneous_note": self.simultaneous_note(),
            "scenario_band": list(band) if band else None,
            "per_scenario_frontier": self.dominant_scenarios,
            "monotonicity_violations": [v.to_dict() for v in self.all_violations],
            "result_kind": self.result_kind,
            "search_is": "heuristic" if self.is_heuristic else "exact",
            "runtime_s": round(self.runtime_s, 1),
            "settings": self.settings.to_dict(),
            "per_scenario": {k: v.to_dict() for k, v in self.per_scenario.items()},
            "label": ("CE reservation-price range under the stated "
                      "completion-cost and model scenarios, from a "
                      + ("complete" if self.is_exhaustive else "SPARSE")
                      + " integer price ladder. NOT an opening max bid, NOT a "
                      "recommended bid, NOT a clearing-price prediction."),
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


def _normal_quantile(p: float) -> float:
    """Inverse standard normal CDF, Acklam's rational approximation.

    Accurate to about 1.15e-9 in absolute value over the whole range, which is
    far more than a confidence bound needs, and it avoids adding a SciPy
    dependency the rest of this project does not have.
    """
    if not 0.0 < p < 1.0:
        raise ValueError("p must be strictly between 0 and 1")
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    lo, hi = 0.02425, 1 - 0.02425
    if p < lo:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > hi:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


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
    """Why did paying more look better? Never because it was.

    Sampling error is checked first, because most such steps are noise. Only
    when the two intervals are disjoint is the rise large enough to need a
    structural explanation, and every available structural explanation is a
    defect in the search rather than a property of the auction.
    """
    lo_lo, lo_hi = lo.ci95
    hi_lo, hi_hi = hi.ci95
    overlap = (not (math.isnan(lo_lo) or math.isnan(hi_lo))
               and lo_lo <= hi_hi and hi_lo <= lo_hi)
    if overlap:
        return "monte carlo"
    if set(lo.added_in_pass) != set(hi.added_in_pass) or \
            set(lo.added_in_buy) != set(hi.added_in_buy):
        return "search instability"
    return "ce-selection instability"


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
    refine: bool = False,
    max_refinement_prices: int = 40,
    notes: str = "",
    progress=None,
) -> ReservationResult:
    """Search the legal price range in every supplied scenario.

    ``scenarios_available`` is the size of the full grid, so a reduced run
    reports itself as reduced. Leaving it unset assumes the supplied scenarios
    *are* the whole grid, which is only true when they are.

    ``refine`` evaluates **every integer** between the highest tested favorable
    price and the next tested unfavorable one, in every scenario, until the
    frontier is pinned or ``max_refinement_prices`` is spent. Without it a
    sparse ladder can only bracket the frontier, and the result says so rather
    than naming a price it never tested.

    Refinement does **not** assume monotonicity to skip regions. Every integer
    in the gap is evaluated. That is more expensive than a bisection would be,
    and it is the honest cost of not assuming the property the monotonicity
    check exists to test.
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
    proxies = {setup.scenario_id: ProxyEvaluator(
        setup.state.pool, setup.state.settings, settings.proxy_reps,
        settings.proxy_seed) for setup in setups}
    evaluated: Dict[str, Dict[int, PricePoint]] = {s.scenario_id: {} for s in setups}
    total = len(setups) * len(ladder)
    done = 0

    def evaluate(setup: ScenarioSetup, price: int) -> PricePoint:
        nonlocal heuristic
        k = ReservationCache.key(setup, candidate_id, price, pass_destination,
                                 settings)
        res = cache.get(k)
        if res is None:
            res = compare_buy_vs_pass(
                setup.state, setup.cast, setup.costs, candidate_id, price,
                pass_destination, settings=settings,
                scenario_id=setup.scenario_id, default_cost=default_cost,
                proxy=proxies[setup.scenario_id])
            cache.put(k, res)
        heuristic = heuristic or res.is_heuristic
        return PricePoint(
            price=price, delta_ce=res.delta_ce, delta_ce_se=res.delta_ce_se,
            verdict=res.verdict, ce_buy=res.ce_buy, ce_pass=res.ce_pass,
            added_in_pass=tuple(res.pass_.best.added) if res.pass_.best else (),
            added_in_buy=tuple(res.buy.best.added) if res.buy.best else (),
            resolved=res.resolved)

    for setup in setups:
        for price in ladder:
            evaluated[setup.scenario_id][price] = evaluate(setup, price)
            done += 1
            if progress is not None:
                progress(done, total, setup.scenario_id, price)

    def build(sid: str) -> ScenarioReservation:
        pts = tuple(evaluated[sid][p] for p in sorted(evaluated[sid]))
        violations = []
        for a, b in zip(pts, pts[1:]):
            if b.delta_ce > a.delta_ce:
                violations.append(MonotonicityViolation(
                    a.price, b.price, a.delta_ce, b.delta_ce, _classify(a, b)))
        return ScenarioReservation(sid, pts, tuple(violations),
                                   min_bid=owner.min_bid, legal_max=legal_max)

    # --- optional refinement of the transition gap -------------------------
    refined = 0
    if refine:
        while refined < max_refinement_prices:
            gaps = {}
            for setup in setups:
                gap = build(setup.scenario_id).transition_gap
                if gap is not None:
                    gaps[setup.scenario_id] = gap
            if not gaps:
                break
            progressed = False
            for sid, (lo_p, hi_p) in gaps.items():
                setup = next(s for s in setups if s.scenario_id == sid)
                for price in range(lo_p, hi_p + 1):
                    if price in evaluated[sid]:
                        continue
                    evaluated[sid][price] = evaluate(setup, price)
                    refined += 1
                    progressed = True
                    if progress is not None:
                        progress(done + refined, total + refined, sid, price)
                    if refined >= max_refinement_prices:
                        break
                if refined >= max_refinement_prices:
                    break
            if not progressed:
                break

    for setup in setups:
        per_scenario[setup.scenario_id] = build(setup.scenario_id)

    return ReservationResult(
        candidate_id=candidate_id, focus_owner_id=focus_id,
        auction_fingerprint=base.state.fingerprint(), legal_max_bid=legal_max,
        pass_destination=pass_destination,
        prices_searched=tuple(sorted(
            set(ladder) | {price for d in evaluated.values() for price in d})),
        per_scenario=per_scenario,
        scenarios_run=tuple(s.scenario_id for s in setups),
        scenarios_available=scenarios_available if scenarios_available is not None
        else len(setups),
        cost_level=base.costs.level, n_sims=settings.ce_sims,
        is_heuristic=heuristic, runtime_s=time.perf_counter() - t0,
        settings=settings, refined_prices=refined, notes=notes)


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
        head = (f"    {'price':>7}{'dCE':>11}{'se':>9}"
                f"{'pointwise 95% CI':>24}{'verdict':>13}")
        out += [head, "    " + "-" * (len(head) - 4)]
        for p in r.points:
            lo, hi = p.ci95
            out.append(f"    ${p.price:>6}{p.delta_ce:>+11.5f}{p.delta_ce_se:>9.5f}"
                       f"    [{lo:+.5f}, {hi:+.5f}]{p.verdict:>13}")
        out += ["    " + "-" * (len(head) - 4)]
        hi_fav = r.highest_tested_favorable
        nxt = r.next_tested_unfavorable
        out.append("    highest TESTED favorable   "
                   + (f"${hi_fav}" if hi_fav is not None
                      else "none of the tested prices is favorable"))
        out.append("    next tested unfavorable    "
                   + (f"${nxt}" if nxt is not None else "none tested above"))
        if r.exact_frontier is not None:
            out.append(f"    integer frontier           ${r.exact_frontier} "
                       f"(every relevant price was evaluated)")
        elif r.reservation_bracket is not None:
            lo_b, hi_b = r.reservation_bracket
            out.append(f"    frontier BRACKET           at least ${lo_b}"
                       + (f", below ${hi_b}" if hi_b is not None
                          else ", nothing above was tested"))
        gap = r.transition_gap
        if gap is not None:
            out.append(f"    UNTESTED between them      ${gap[0]}-${gap[1]} "
                       f"({gap[1] - gap[0] + 1} price(s) never evaluated)")
        if r.untested_intervals:
            spans = ", ".join(f"${a}-${b}" for a, b in r.untested_intervals[:6])
            more = "" if len(r.untested_intervals) <= 6 else " ..."
            out.append(f"    untested overall           {spans}{more}")
        out.append("    unresolved band            "
                   + (f"${r.unresolved_region[0]}-${r.unresolved_region[1]}"
                      if r.unresolved_region else "none"))
        if r.violations:
            out.append(f"    monotonicity      {len(r.violations)} violation(s) "
                       f"(paying more looked BETTER, which it cannot truly be):")
            for v in r.violations:
                out.append(f"      ${v.lower_price} -> ${v.higher_price}: "
                           f"{v.lower_delta:+.5f} -> {v.higher_delta:+.5f} "
                           f"[{v.cause}]")
        out.append("")

    out += [bar, "RESERVATION RANGE", bar]
    lo_b, hi_b = result.robust_bracket or (None, None)
    if result.robust_exact_frontier is not None:
        out.append(f"  robust frontier      ${result.robust_exact_frontier}  "
                   f"(EXACT: every relevant integer price was evaluated)")
    elif lo_b is not None:
        out.append(f"  robust bracket       at least ${lo_b}"
                   + (f", below ${hi_b}" if hi_b is not None
                      else ", nothing above was tested"))
        out.append("                       the highest TESTED price favorable in")
        out.append("                       every scenario. The true frontier is")
        out.append("                       inside this bracket, not necessarily")
        out.append("                       at its lower end.")
    else:
        out.append("  robust               none -- no tested price is favorable "
                   "in every scenario")
    out.append("  permissive (tested)  "
               + (f"${result.permissive_tested_price}"
                  if result.permissive_tested_price is not None
                  else "none -- every scenario says no at every tested price"))
    out.append("                       highest tested price not demonstrably")
    out.append("                       unfavorable in at least one scenario")
    out.append("")
    if not result.is_exhaustive:
        gaps = result.untested_intervals
        closed = all(r.transition_gap is None
                     for r in result.per_scenario.values())
        out += [f"  *** SPARSE LADDER: {len(gaps)} untested interval(s) remain "
                f"in the legal range.", ""]
        if closed:
            out += ["      The TRANSITION REGION is fully evaluated: every "
                    "integer between",
                    "      the highest favorable and the next unfavorable price "
                    "was tested.",
                    "      What stops this being an exact frontier is that "
                    "prices inside it",
                    "      came back UNRESOLVED, or that untested prices remain "
                    "elsewhere and",
                    "      monotonicity is checked rather than assumed, so a "
                    "favorable price",
                    "      above the bracket cannot be ruled out by argument "
                    "alone. ***", ""]
        else:
            out += ["      No price between the tested ones was evaluated, so "
                    "none of the",
                    "      numbers above is a definitive integer frontier. "
                    "Re-run with",
                    "      --refine to evaluate every integer in the transition "
                    "gap. ***", ""]
    if result.refined_prices:
        out.append(f"  refinement           {result.refined_prices} extra integer "
                   f"price(s) evaluated to pin the frontier")
    band = result.scenario_band
    if band:
        lo = f"${band[0]}" if band[0] is not None else "none in some scenario"
        out.append(f"  per-scenario highest tested favorable spans {lo} "
                   f"to ${band[1]}")
    out += [f"  result is    {result.result_kind}",
            f"  runtime      {result.runtime_s:.1f}s", ""]
    if result.all_violations:
        causes: Dict[str, int] = {}
        for v in result.all_violations:
            causes[v.cause] = causes.get(v.cause, 0) + 1
        out += ["  Monotonicity was checked, not assumed. "
                + ", ".join(f"{n} {c}" for c, n in sorted(causes.items())) + ".",
                "  With fixed costs and a correctly optimised completion, paying",
                "  more can never raise the attainable maximum: every roster",
                "  affordable at $p+1 was affordable at $p, and the pass branch",
                "  does not depend on our price at all. So an upward step is",
                "  noise or instability, never beneficial economics.", ""]
    if result.is_heuristic:
        out += ["  Both branches at every price are BOUNDED SEARCHES. The range",
                "  is between the best rosters this search found.", ""]
    out += ["  " + result.simultaneous_note().replace(". ", ".\n  "), "",
            "  This is a CE reservation-price range under the stated",
            "  completion-cost assumptions and model scenarios.",
            "  It is NOT an opening maximum, NOT a bid, and NOT a prediction of",
            "  what the room will pay.", bar]
    return "\n".join(out)
