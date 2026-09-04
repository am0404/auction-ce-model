"""Small statistical helpers, dependency-free.

NumPy has no ``erf``, and SciPy is not a dependency, so the normal CDF is
implemented here with Abramowitz & Stegun 7.1.26 (|error| < 1.5e-7 -- far
tighter than anything in this model is calibrated to).
"""

from __future__ import annotations

import math
from typing import Union

import numpy as np

__all__ = ["norm_cdf", "norm_pdf", "floored_mean", "match_floored_mean"]

_A = (0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429)
_P = 0.3275911
_INV_SQRT2 = 1.0 / math.sqrt(2.0)
_INV_SQRT2PI = 1.0 / math.sqrt(2.0 * math.pi)


def _erf(x):
    """Abramowitz & Stegun 7.1.26.  Works elementwise on arrays or scalars."""
    x = np.asarray(x, dtype=np.float64)
    sign = np.sign(x)
    ax = np.abs(x)
    t = 1.0 / (1.0 + _P * ax)
    poly = t * (_A[0] + t * (_A[1] + t * (_A[2] + t * (_A[3] + t * _A[4]))))
    return sign * (1.0 - poly * np.exp(-ax * ax))


def norm_cdf(z):
    """Standard normal CDF."""
    return 0.5 * (1.0 + _erf(np.asarray(z, dtype=np.float64) * _INV_SQRT2))


def norm_pdf(z):
    """Standard normal density."""
    z = np.asarray(z, dtype=np.float64)
    return _INV_SQRT2PI * np.exp(-0.5 * z * z)


def floored_mean(mu, sd):
    """``E[max(0, N(mu, sd))]``.

    Weekly fantasy scores are floored at zero, so a player's *expected points*
    are not his latent mean: the more volatile he is, the more the floor is
    worth to him.  A replacement-tier synthetic WR at 4.0 points with a weekly
    SD of 7.2 actually expects 5.3.

    This is why the engine projects ``floored_mean(level, week_sd)`` rather
    than ``level``.  Projecting the latent mean would systematically
    under-project low-mean, high-variance players and bias every bench and
    flex decision against them.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sd = np.asarray(sd, dtype=np.float64)
    safe = np.maximum(sd, 1e-12)
    z = mu / safe
    out = mu * norm_cdf(z) + safe * norm_pdf(z)
    return np.where(sd <= 0.0, np.maximum(mu, 0.0), out)


def match_floored_mean(target: float, sd: float, tol: float = 1e-10) -> float:
    """Find ``mu`` such that ``floored_mean(mu, sd) == target``.

    ``floored_mean`` is strictly increasing in ``mu``, so bisection is exact
    and needs no derivative.
    """
    if sd <= 0.0:
        return float(target)
    lo, hi = float(target) - 8.0 * sd - 1.0, float(target) + 1.0
    while float(floored_mean(lo, sd)) > target:
        lo -= 8.0 * sd + 1.0
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        if float(floored_mean(mid, sd)) < target:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)
