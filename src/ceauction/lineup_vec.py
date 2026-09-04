"""Vectorised lineup selection.

Identical algorithm to :func:`ceauction.lineup.select_lineup` — the same exact
matroid greedy — but evaluated for many (season, team, week) cells at once so
that a full championship-equity run is a few seconds rather than a few minutes.
``tests/test_lineup.py`` asserts the two implementations agree exactly.

The function signature is itself part of the information barrier: it accepts
projections, availability and positions.  There is no parameter through which
a realized score could be passed.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

__all__ = ["select_lineups_mask", "MAX_STARTERS"]

MAX_STARTERS = 8

_QB, _RB = 0, 1  # position codes; 2 = WR, 3 = TE share the WR/TE pool


def select_lineups_mask(
    projection: np.ndarray,
    available: np.ndarray,
    position: np.ndarray,
) -> np.ndarray:
    """Return a boolean starter mask with the shape of ``projection``.

    Parameters
    ----------
    projection:
        ``(..., roster_size)`` pregame projections.
    available:
        ``(..., roster_size)`` bool; ``False`` for bye or injury.
    position:
        ``(..., roster_size)`` int position codes, broadcastable to the shape
        of ``projection`` (typically ``(n_teams, roster_size)``).

    Notes
    -----
    Ties are broken by roster order via a stable argsort, matching the scalar
    implementation exactly.
    """
    projection = np.asarray(projection, dtype=np.float64)
    available = np.asarray(available, dtype=bool)
    position = np.broadcast_to(np.asarray(position), projection.shape)

    lead = projection.shape[:-1]
    n_slots = projection.shape[-1]
    flat = int(np.prod(lead)) if lead else 1

    proj = projection.reshape(flat, n_slots)
    avail = available.reshape(flat, n_slots)
    pos = position.reshape(flat, n_slots)

    # Unavailable players sort to the bottom and are rejected by `avail` below.
    key = np.where(avail, proj, -np.inf)
    order = np.argsort(-key, axis=1, kind="stable")

    is_qb = pos == _QB
    is_rb = pos == _RB
    is_wt = pos > _RB

    q = np.zeros(flat, dtype=np.int16)
    r = np.zeros(flat, dtype=np.int16)
    t = np.zeros(flat, dtype=np.int16)
    chosen = np.zeros((flat, n_slots), dtype=bool)
    rows = np.arange(flat)

    for k in range(n_slots):
        idx = order[:, k]
        ok_avail = avail[rows, idx]
        dq = is_qb[rows, idx].astype(np.int16)
        dr = is_rb[rows, idx].astype(np.int16)
        dt = is_wt[rows, idx].astype(np.int16)
        nq, nr, nt = q + dq, r + dr, t + dt
        feasible = (
            (nq <= 2)
            & (nr <= 4)
            & (nt <= 5)
            & (nq + nr <= 5)
            & (nq + nt <= 6)
            & (nr + nt <= 7)
            & (nq + nr + nt <= MAX_STARTERS)
        )
        take = ok_avail & feasible
        q = np.where(take, nq, q)
        r = np.where(take, nr, r)
        t = np.where(take, nt, t)
        chosen[rows, idx] = take

    return chosen.reshape(projection.shape)
