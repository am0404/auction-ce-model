"""Runtime benchmarks and Monte Carlo uncertainty at several simulation counts."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from .roster import RosterSet
from .simulate import DEFAULT_CHUNK, simulate_seasons
from .synthetic import make_synthetic_league
from .worlds import build_pool_arrays

__all__ = ["BenchRow", "benchmark", "format_table", "profile_stages"]


@dataclass(frozen=True)
class BenchRow:
    n_sims: int
    chunk: int
    seconds: float
    seasons_per_second: float
    ms_per_season: float
    ce_focus: float
    ce_se: float
    ce_halfwidth95: float
    peak_ce: float

    def as_row(self) -> str:
        return (
            f"{self.n_sims:>9,}  {self.chunk:>6}  {self.seconds:>8.2f}  "
            f"{self.seasons_per_second:>10,.0f}  {self.ms_per_season:>9.3f}  "
            f"{self.ce_focus:>8.4f}  {self.ce_se:>8.5f}  {self.ce_halfwidth95:>9.5f}"
        )


def benchmark(
    counts: Sequence[int] = (250, 1_000, 4_000, 16_000),
    seed: int = 20260904,
    chunk: int = DEFAULT_CHUNK,
    rosters: Optional[RosterSet] = None,
    focus_team: int = 0,
) -> List[BenchRow]:
    """Time a full CE run at each simulation count."""
    rosters = rosters if rosters is not None else make_synthetic_league()
    pool = build_pool_arrays(rosters.pool, rosters.settings)
    rows: List[BenchRow] = []
    # Warm up NumPy / BLAS so the first row is not penalised.
    simulate_seasons(rosters, 32, seed, chunk, pool=pool)
    for n in counts:
        t0 = time.perf_counter()
        out = simulate_seasons(rosters, n, seed, chunk, pool=pool)
        dt = time.perf_counter() - t0
        ce = out.championship_equity()
        p = float(ce[focus_team])
        se = math.sqrt(max(p * (1 - p), 0.0) / n)
        rows.append(
            BenchRow(
                n_sims=n,
                chunk=chunk,
                seconds=dt,
                seasons_per_second=n / dt if dt else float("inf"),
                ms_per_season=1000.0 * dt / n,
                ce_focus=p,
                ce_se=se,
                ce_halfwidth95=1.96 * se,
                peak_ce=float(ce.max()),
            )
        )
    return rows


def format_table(rows: Sequence[BenchRow]) -> str:
    head = (
        f"{'seasons':>9}  {'chunk':>6}  {'seconds':>8}  {'seasons/s':>10}  "
        f"{'ms/season':>9}  {'CE(T1)':>8}  {'SE':>8}  {'+/-95%':>9}"
    )
    return "\n".join([head, "-" * len(head)] + [r.as_row() for r in rows])


def profile_stages(
    n_sims: int = 2_000, seed: int = 20260904, chunk: int = DEFAULT_CHUNK,
    rosters: Optional[RosterSet] = None,
) -> List:
    """Time each pipeline stage separately to locate the bottleneck."""
    from .playoffs import run_bracket
    from .schedule import opponents_for_batch
    from .simulate import team_scores
    from .standings import regular_season
    from .worlds import generate_world

    rosters = rosters if rosters is not None else make_synthetic_league()
    settings = rosters.settings
    pool = build_pool_arrays(rosters.pool, settings)
    rm = rosters.roster_matrix()
    timings = {"world": 0.0, "lineup+score": 0.0, "schedule": 0.0,
               "standings": 0.0, "playoffs": 0.0}

    for start in range(0, n_sims, chunk):
        size = min(chunk, n_sims - start)
        t = time.perf_counter(); world = generate_world(pool, seed, start, size)
        timings["world"] += time.perf_counter() - t
        t = time.perf_counter(); scores, _ = team_scores(world, rm)
        timings["lineup+score"] += time.perf_counter() - t
        t = time.perf_counter(); opp = opponents_for_batch(seed, start, size, settings)
        timings["schedule"] += time.perf_counter() - t
        t = time.perf_counter(); rs = regular_season(scores, opp, settings)
        timings["standings"] += time.perf_counter() - t
        t = time.perf_counter(); run_bracket(scores, rs, settings)
        timings["playoffs"] += time.perf_counter() - t

    total = sum(timings.values())
    return sorted(
        ((k, v, 100.0 * v / total if total else 0.0) for k, v in timings.items()),
        key=lambda r: -r[1],
    )
