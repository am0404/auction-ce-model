"""Counter-based deterministic RNG.

Every random value in the simulator is a pure hash of its *coordinates*
``(seed, kind, sim, entity, week, sub)``.  There is no sequential state, which
buys three things that matter for championship equity:

1. **Reproducibility.**  A value depends only on where it sits, never on how
   many draws happened before it, so chunking the simulation into batches of
   any size gives bit-identical results.
2. **Common random numbers.**  Swapping one player on one roster perturbs no
   other player's draws, because the other players' coordinates are unchanged.
3. **Vectorisation.**  Draws are produced by integer arithmetic on whole
   NumPy arrays, so there is no per-stream ``Generator`` construction cost.

The mixing function is the splitmix64 finalizer, which is not
cryptographically strong but has excellent avalanche behaviour and is more
than adequate for Monte Carlo.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np

__all__ = [
    "Kind",
    "mix64",
    "hash_coords",
    "uniform",
    "normal",
    "exponential",
    "bernoulli",
    "randint",
    "permutation_batch",
]

_MASK = np.uint64(0xFFFFFFFFFFFFFFFF)
_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)
_S30 = np.uint64(30)
_S27 = np.uint64(27)
_S31 = np.uint64(31)
_S11 = np.uint64(11)
_S32 = np.uint64(32)
_LOW32 = np.uint64(0xFFFFFFFF)
_ODD = np.uint64(0xD6E8FEB86659FD93)
_TWO53 = np.float64(1.0 / 9007199254740992.0)  # 2**-53
_TWO32 = np.float64(1.0 / 4294967296.0)        # 2**-32


class Kind:
    """Namespace of stream identifiers.

    Each distinct random quantity gets its own ``kind`` so that adding a new
    kind of draw later cannot disturb the values of existing ones.
    """

    SEASON_SHIFT = 101
    INJURY_ONSET = 102
    INJURY_DURATION = 103
    WEEK_NOISE = 104
    SPIKE_HIT = 105
    SPIKE_SIZE = 106
    ROLE_HAPPENS = 107
    ROLE_WEEK = 108
    ROLE_SIZE = 109
    GROUP_SHOCK = 110
    PROJ_NOISE = 111
    SCHEDULE_PERM = 112
    SIGNAL_NOISE = 113
    WEEKLY_STATE = 114


def mix64(x: np.ndarray) -> np.ndarray:
    """splitmix64 finalizer applied elementwise to a uint64 array.

    Integer wraparound is the point of the algorithm, so the overflow warning
    NumPy raises for 0-d/scalar operands is suppressed.
    """
    with np.errstate(over="ignore"):
        z = (x + _GOLDEN) & _MASK
        z = ((z ^ (z >> _S30)) * _M1) & _MASK
        z = ((z ^ (z >> _S27)) * _M2) & _MASK
        return z ^ (z >> _S31)


def hash_coords(seed: int, kind: int, *coords: Iterable) -> np.ndarray:
    """Hash ``(seed, kind, *coords)`` into a uint64 array.

    ``coords`` are broadcast against each other, so passing arrays of shapes
    ``(S, 1, 1)``, ``(1, P, 1)`` and ``(1, 1, W)`` yields an ``(S, P, W)``
    block of independent values in one call.
    """
    seed_arr = np.array(seed, dtype=np.uint64)
    kind_arr = np.array(kind, dtype=np.uint64)
    acc = mix64(seed_arr ^ mix64(kind_arr))
    with np.errstate(over="ignore"):
        for c in coords:
            arr = np.asarray(c)
            if arr.dtype != np.uint64:
                arr = arr.astype(np.int64, copy=False).astype(np.uint64, copy=False)
            # Multiplying by an odd constant is a bijection, so one mixing round
            # per coordinate suffices.  Coordinates should be passed
            # smallest-broadcast-first: only the last one runs at full size.
            acc = mix64(acc ^ ((arr * _ODD) & _MASK))
    return acc


def uniform(seed: int, kind: int, *coords: Iterable) -> np.ndarray:
    """Uniform on the half-open interval ``[0, 1)``."""
    return (hash_coords(seed, kind, *coords) >> _S11).astype(np.float64) * _TWO53


def _uniform_open(seed: int, kind: int, *coords: Iterable) -> np.ndarray:
    """Uniform on ``(0, 1)`` — safe for ``log`` and Box-Muller."""
    u = uniform(seed, kind, *coords)
    return np.where(u <= 0.0, _TWO53, u)


def normal(seed: int, kind: int, *coords: Iterable) -> np.ndarray:
    """Standard normal via Box-Muller.

    Both uniforms come from the two halves of a *single* 64-bit hash rather
    than from two hashes, which roughly quarters the cost of the dominant draw
    in the simulator.  32 bits per uniform supports magnitudes out to about
    6.6 sigma, far beyond anything a fantasy score distribution needs.
    """
    h = hash_coords(seed, kind, *coords)
    u1 = ((h >> _S32).astype(np.float64) + 0.5) * _TWO32
    u2 = ((h & _LOW32).astype(np.float64) + 0.5) * _TWO32
    return np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)


def exponential(seed: int, kind: int, *coords: Iterable) -> np.ndarray:
    """Exponential with rate 1 (mean 1)."""
    return -np.log(_uniform_open(seed, kind, *coords))


def bernoulli(p, seed: int, kind: int, *coords: Iterable) -> np.ndarray:
    """Bernoulli draws; ``p`` may be a scalar or a broadcastable array."""
    return uniform(seed, kind, *coords) < np.asarray(p, dtype=np.float64)


def randint(high, seed: int, kind: int, *coords: Iterable) -> np.ndarray:
    """Uniform integers in ``[0, high)``; ``high`` may be array-like."""
    return np.floor(uniform(seed, kind, *coords) * np.asarray(high)).astype(np.int64)


def permutation_batch(n: int, count: int, seed: int, kind: int, offset: int = 0) -> np.ndarray:
    """``(count, n)`` array whose rows are independent permutations of ``range(n)``.

    Implemented as an argsort of independent uniforms, which is a valid (if not
    the fastest possible) way to generate a uniform permutation and is fully
    vectorised.
    """
    sims = (np.arange(count, dtype=np.int64) + offset).reshape(count, 1)
    keys = uniform(seed, kind, sims, np.arange(n, dtype=np.int64).reshape(1, n))
    return np.argsort(keys, axis=1, kind="stable")
