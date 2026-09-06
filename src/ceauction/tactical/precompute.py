"""Bounded precomputation: do the slow work between nominations, not during one.

A ten-second timer cannot pay for a CE-backed search, so the search has to have
happened already. This module prepares tactical results for a *named, bounded*
candidate set and stores them under their full cache key, so the live path is a
dictionary lookup.

Two things it deliberately refuses to be. It is not a full board precomputation
-- every player crossed with every price, recipient and scenario is a
combinatorial fiction and is out of scope. And it is not a resume-by-guessing:
a stored entry is reused **only** when its cache key matches exactly, which
means any change to the room, the market, the leader, the price, the recipient
set, the scenarios or the settings invalidates it. A stale bid is worse than a
slow one.

Real-player output must be written only under ``local_data/``, which is
gitignored; the writer refuses any other path for a run that is not fabricated.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from ..auction.completion import ComparisonCast
from ..auction.costs import CostBook
from ..auction.state import AuctionState
from ..market.live import MarketState
from .maxbid import (DEFAULT_SCENARIOS, TacticalCache, TacticalResult,
                     TacticalScenario, TacticalSettings, evaluate_tactical)

__all__ = ["PrecomputeEntry", "PrecomputeReport", "precompute", "write_report"]


@dataclass(frozen=True)
class PrecomputeEntry:
    candidate_id: int
    cache_key: str
    reused: bool
    runtime_s: float
    robust: Optional[int]
    base: Optional[int]
    permissive: Optional[int]
    legal_max: int
    mode: str

    def to_dict(self) -> Dict[str, object]:
        import dataclasses
        return {f.name: getattr(self, f.name)
                for f in dataclasses.fields(self)}


@dataclass
class PrecomputeReport:
    entries: Tuple[PrecomputeEntry, ...]
    total_runtime_s: float
    n_reused: int
    n_computed: int
    mode: str
    scenarios: Tuple[str, ...]
    fabricated: bool

    def to_dict(self, *, include_entries: bool = True) -> Dict[str, object]:
        out: Dict[str, object] = {
            "mode": self.mode,
            "scenarios": list(self.scenarios),
            "candidates": len(self.entries),
            "reused_from_cache": self.n_reused,
            "computed": self.n_computed,
            "total_runtime_s": round(self.total_runtime_s, 3),
            "fabricated": self.fabricated,
            "label": ("bounded precomputation over a NAMED candidate set. "
                      "Not a complete board precomputation."),
        }
        if include_entries:
            out["entries"] = [e.to_dict() for e in self.entries]
        return out


def precompute(
    state: AuctionState,
    cast: ComparisonCast,
    costs: CostBook,
    candidate_ids: Sequence[int],
    *,
    cache: TacticalCache,
    settings: TacticalSettings = TacticalSettings(),
    scenarios: Sequence[TacticalScenario] = DEFAULT_SCENARIOS,
    market: Optional[MarketState] = None,
    key_by_id: Optional[Dict[int, str]] = None,
    current_price: Optional[int] = None,
    increment: int = 1,
    current_leader: Optional[str] = None,
    eligible: Optional[Sequence[str]] = None,
    progress: Optional[Callable[[int, int, int, bool], None]] = None,
    fabricated: bool = True,
) -> PrecomputeReport:
    """Fill ``cache`` for a bounded candidate list. Resumes on matching keys only."""
    ids = list(dict.fromkeys(int(c) for c in candidate_ids))
    if not ids:
        raise ValueError("precompute needs at least one candidate id")
    t0 = time.perf_counter()
    entries: List[PrecomputeEntry] = []
    reused = 0
    for i, cid in enumerate(ids):
        before = cache.hits
        t1 = time.perf_counter()
        r: TacticalResult = evaluate_tactical(
            state, cast, costs, cid, settings=settings, scenarios=scenarios,
            market=market,
            candidate_key=key_by_id.get(cid) if key_by_id else None,
            key_by_id=key_by_id, current_price=current_price,
            increment=increment, current_leader=current_leader,
            eligible=eligible, cache=cache)
        was_reused = cache.hits > before
        reused += int(was_reused)
        entries.append(PrecomputeEntry(
            candidate_id=cid, cache_key=r.cache_key, reused=was_reused,
            runtime_s=time.perf_counter() - t1,
            robust=r.robust_tactical_max, base=r.base_tactical_max,
            permissive=r.permissive_ceiling, legal_max=r.legal_max,
            mode=r.mode))
        if progress is not None:
            progress(i + 1, len(ids), cid, was_reused)
    return PrecomputeReport(
        entries=tuple(entries), total_runtime_s=time.perf_counter() - t0,
        n_reused=reused, n_computed=len(ids) - reused, mode=settings.mode,
        scenarios=tuple(s.scenario_id for s in scenarios),
        fabricated=fabricated)


def write_report(report: PrecomputeReport, path: str) -> None:
    """Write the report. Real-player runs are refused outside ``local_data/``."""
    p = Path(path)
    if not report.fabricated:
        parts = {q.name for q in p.resolve().parents}
        if "local_data" not in parts:
            raise ValueError(
                f"refusing to write real-player tactical output to {path!r}. "
                f"Player-level output from real data must live under "
                f"local_data/, which is gitignored.")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report.to_dict(), indent=2, default=str),
                 encoding="utf-8")
