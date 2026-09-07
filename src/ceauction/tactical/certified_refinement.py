r"""Integer price refinement, with an interval that survives being asked 14 times.

The coarse frontier brackets the crossing between ``$65`` (favorable) and
``$80`` (unfavorable). Refining that to an integer means evaluating every price
in between and reading off where the sign changes -- and the moment fourteen
prices are each given a 95% interval, the chance that at least one of them is
wrong is nowhere near 5%. Quoting the highest favorable price off a sequence of
pointwise intervals is the specific mistake this module exists to avoid.

So two answers are reported, always together:

**pointwise** each price's own cluster t-interval on ``k`` allocation draws,
              the same statistic the coarse ladder reports. Correct for one
              price examined in isolation, and that is all.

**simultaneous** a Bonferroni band: the same t-interval with ``alpha`` split
                 evenly across every price on the grid. Conservative rather than
                 exact -- it ignores the heavy positive dependence between
                 neighbouring prices, which share most of their rosters and all
                 of their random numbers -- but a wide honest band beats a
                 narrow one that is not a band at all. ``AUCTION_LAYER.md``
                 already made this choice for the reservation search; this is
                 the same choice, adapted from a normal quantile on a per-price
                 SE to a t quantile on ``k - 1`` degrees of freedom, because
                 the frontier's uncertainty is between allocation draws and not
                 between seasons.

Monotonicity is **not** assumed anywhere. The grid is evaluated exhaustively,
every price is reported including unresolved ones, and any sign reversal above
the first crossing is recorded as a nonmonotonicity rather than smoothed away.
A binary search would have assumed exactly the property under test.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = [
    "t_quantile", "bonferroni_t_half_width", "RefinementReading",
    "read_refinement",
]


def _t_pdf(x: float, df: int) -> float:
    return (math.exp(math.lgamma((df + 1) / 2.0) - math.lgamma(df / 2.0))
            / math.sqrt(df * math.pi)
            * (1.0 + x * x / df) ** (-(df + 1) / 2.0))


def _t_cdf(x: float, df: int, steps: int = 4000) -> float:
    """Student-t CDF by Simpson integration of the pdf.

    ``scipy`` is not a dependency of this package and is not going to become
    one for a single quantile, so the integral is done directly. Steps are
    chosen so the result is accurate to far more digits than an interval on
    eleven draws could possibly need.
    """
    if x < 0:
        return 1.0 - _t_cdf(-x, df, steps)
    n = steps if steps % 2 == 0 else steps + 1
    h = x / n
    total = _t_pdf(0.0, df) + _t_pdf(x, df)
    for i in range(1, n):
        total += _t_pdf(i * h, df) * (4 if i % 2 else 2)
    return 0.5 + total * h / 3.0


def t_quantile(p: float, df: int) -> float:
    """Two-sided-usable Student-t quantile, by bisection on :func:`_t_cdf`."""
    if df < 1:
        return float("nan")
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    lo, hi = 0.0, 200.0
    if p < 0.5:
        return -t_quantile(1.0 - p, df)
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if _t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def bonferroni_t_half_width(se: float, k: int, m: int,
                            alpha: float = 0.05) -> float:
    """Half-width of one interval in a Bonferroni band over ``m`` statements."""
    if k < 2:
        return float("nan")
    return t_quantile(1.0 - alpha / (2.0 * m), k - 1) * se


@dataclass(frozen=True)
class RefinementReading:
    """What the grid says, pointwise and simultaneously."""

    prices: Tuple[int, ...]
    pointwise: Dict[int, Tuple[float, float]]
    simultaneous: Dict[int, Tuple[float, float]]
    means: Dict[int, float]
    pointwise_verdict: Dict[int, str]
    simultaneous_verdict: Dict[int, str]

    @property
    def m(self) -> int:
        return len(self.prices)

    def _highest_favorable(self, verdicts: Dict[int, str]) -> Optional[int]:
        fav = [p for p in self.prices if verdicts[p] == "favorable"]
        return max(fav) if fav else None

    def _first_unfavorable(self, verdicts: Dict[int, str]) -> Optional[int]:
        unf = [p for p in self.prices if verdicts[p] == "unfavorable"]
        return min(unf) if unf else None

    @property
    def pointwise_highest_favorable(self) -> Optional[int]:
        return self._highest_favorable(self.pointwise_verdict)

    @property
    def simultaneous_highest_favorable(self) -> Optional[int]:
        return self._highest_favorable(self.simultaneous_verdict)

    @property
    def pointwise_first_unfavorable(self) -> Optional[int]:
        return self._first_unfavorable(self.pointwise_verdict)

    @property
    def simultaneous_first_unfavorable(self) -> Optional[int]:
        return self._first_unfavorable(self.simultaneous_verdict)

    def unresolved_gap(self, verdicts: Dict[int, str]) -> Tuple[Optional[int], Optional[int]]:
        """The open interval between the last favorable and the first unfavorable."""
        hi = self._highest_favorable(verdicts)
        un = self._first_unfavorable(verdicts)
        return (hi, un)

    @property
    def nonmonotonicities(self) -> Tuple[str, ...]:
        """Any favorable price above an unfavorable one, pointwise.

        Recorded rather than corrected. The refinement does not assume the
        delta is monotone in price, so a reversal is a finding about the
        surface and not a bug to be smoothed.
        """
        out: List[str] = []
        first_unf = self.pointwise_first_unfavorable
        if first_unf is None:
            return ()
        for p in self.prices:
            if p > first_unf and self.pointwise_verdict[p] == "favorable":
                out.append(f"${p} favorable above first unfavorable ${first_unf}")
        prev = None
        for p in self.prices:
            cur = self.means[p]
            if prev is not None and cur > prev + 1e-12:
                out.append(f"delta rises from ${p - 1} to ${p}: "
                           f"{prev:+.5f} -> {cur:+.5f}")
            prev = cur
        return tuple(out)

    def to_dict(self) -> Dict[str, object]:
        return {
            "m_statements": self.m,
            "alpha": 0.05,
            "bonferroni_alpha_per_statement": round(0.05 / self.m, 8),
            "pointwise_highest_favorable": self.pointwise_highest_favorable,
            "pointwise_first_unfavorable": self.pointwise_first_unfavorable,
            "simultaneous_highest_favorable":
                self.simultaneous_highest_favorable,
            "simultaneous_first_unfavorable":
                self.simultaneous_first_unfavorable,
            "pointwise_unresolved_gap":
                list(self.unresolved_gap(self.pointwise_verdict)),
            "simultaneous_unresolved_gap":
                list(self.unresolved_gap(self.simultaneous_verdict)),
            "nonmonotonicities": list(self.nonmonotonicities),
            "prices": [
                {"p": p, "mean_delta": round(self.means[p], 8),
                 "pointwise_ci": [round(x, 8) for x in self.pointwise[p]],
                 "pointwise_verdict": self.pointwise_verdict[p],
                 "simultaneous_ci": [round(x, 8) for x in self.simultaneous[p]],
                 "simultaneous_verdict": self.simultaneous_verdict[p]}
                for p in self.prices],
            "note": ("the pointwise sequence is NOT a simultaneous 95% "
                     "frontier; the Bonferroni band is conservative because it "
                     "ignores the positive dependence between neighbouring "
                     "prices"),
        }


def _verdict(lo: float, hi: float) -> str:
    if lo > 0:
        return "favorable"
    if hi < 0:
        return "unfavorable"
    return "unresolved"


def read_refinement(points: Sequence[Dict[str, object]], *, k: int,
                    alpha: float = 0.05) -> RefinementReading:
    """Turn per-price frontier points into both readings.

    ``points`` must each carry ``p``, ``mean_delta`` and
    ``se_of_ensemble_mean`` -- the between-allocation standard error the
    frontier already reports, never a season-level one.
    """
    prices = tuple(int(pt["p"]) for pt in points)
    m = len(prices)
    means: Dict[int, float] = {}
    pw: Dict[int, Tuple[float, float]] = {}
    sim: Dict[int, Tuple[float, float]] = {}
    pwv: Dict[int, str] = {}
    simv: Dict[int, str] = {}
    t_point = t_quantile(1.0 - alpha / 2.0, k - 1)
    t_simul = t_quantile(1.0 - alpha / (2.0 * m), k - 1)
    for pt in points:
        p = int(pt["p"])
        mean = float(pt["mean_delta"])
        se = float(pt["se_of_ensemble_mean"])
        means[p] = mean
        pw[p] = (mean - t_point * se, mean + t_point * se)
        sim[p] = (mean - t_simul * se, mean + t_simul * se)
        pwv[p] = _verdict(*pw[p])
        simv[p] = _verdict(*sim[p])
    return RefinementReading(prices, pw, sim, means, pwv, simv)
