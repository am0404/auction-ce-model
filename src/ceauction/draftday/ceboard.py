"""ROUGH CE MAX — a CE-informed auction heuristic for draft day.

This is deliberately a heuristic draft tool, not a reservation price. It exists
because the certified path costs minutes per price and a draft board needs
hundreds of numbers before eight o'clock.

The split of labour is the whole idea, and it is what keeps the numbers honest:

**Championship equity decides the ranking.** Every player's score is a real
paired season/playoff simulation -- insert him into a representative roster,
displace the weakest legal alternative, and difference the championship
indicator against the same roster without him on identical cached seasons.
Injuries, weekly variance, byes, lineup choice and the playoff bracket are all
in there because they are in ``PlayerSpec`` and the simulation. Nothing here
converts lineup points into CE by a coefficient.

**The market decides the dollar scale.** The CE score is a probability, not a
price, and no defensible constant turns one into the other. So instead of
inventing a coefficient, the existing reconciled opening base-price
distribution is *rank-mapped* onto the CE ordering: the largest real dollar
value goes to the highest-CE player, and so on down. That preserves the shape
of a real auction -- the total pool, and how many players belong at $50, $20
and $3 -- while letting CE rather than Sleeper's ordering decide who occupies
each rung.

**Completed sales move it, not a new simulation.** Live values apply the market
multipliers the room has actually taught us, reconcile against the dollars
still in the room, and clamp to our exact legal maximum. No CE is re-run
mid-draft, and no second scarcity coefficient is introduced anywhere.

What this is not: it is not a maximum bid, not audited, and not a certified
reservation price. Three representative contexts are three contexts, not the
future auction. The confidence label is about how much those three disagree,
which is the only uncertainty this method actually measures.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..auction.feasibility import can_fill_lineup
from ..auction.state import AuctionState
from ..league import Position
from .roughce import RoughWorldCache

__all__ = [
    "ROUGH_LABEL", "ROUGH_DISCLOSURE", "FORBIDDEN_WORDS", "CONTEXTS",
    "RANK_SEASONS", "RANK_SEED", "PlayerCE", "BoardEntry", "OpeningCEBoard",
    "build_contexts", "score_players", "assign_dollars", "build_opening_board",
    "live_value", "CEBOARD_FILENAME",
]

ROUGH_LABEL = "ROUGH CE MAX"
ROUGH_DISCLOSURE = (
    "ROUGH CE MAX IS A HEURISTIC STARTING POINT.\n"
    "CE determines player ranking; the market price curve supplies the dollar "
    "scale.\n"
    "Live changes reflect completed-sale inflation and exact legal limits, not "
    "a fresh CE simulation.")

#: Claims this board may never make.
FORBIDDEN_WORDS = ("CE AUDITED", "CERTIFIED MAX BID", "CERTIFIED", "OPTIMAL",
                   "GUARANTEED", "RESERVATION PRICE", "EXACT MAX")

CEBOARD_FILENAME = "ce_price_board.json"

#: Seasons for the ranking. A ranking needs far less precision than a
#: reservation price: it only has to order players, and the paired CRN removes
#: the season noise that would otherwise dominate a small sample.
RANK_SEASONS = 1_000
RANK_SEED = 917_324_011

#: Three deterministic, legal 15-man shapes. Each fields all eight starters
#: (1 QB, 2 RB, 3 WR/TE, 1 flex, 1 superflex) with real slack, and they differ
#: in the one dimension this league actually turns on -- how much of the roster
#: is quarterback, given a superflex and no dedicated tight-end slot.
#: The QB counts are 2/3/4, not 1/2/4. This is a SUPERFLEX league: the QB slot
#: and the superflex are both normally quarterbacks, so a team carries two
#: startable ones and a backup. A single-QB roster is legal but not
#: representative -- it cannot fill the superflex with a quarterback at all, so
#: every QB added to it scores an enormous marginal equity and the whole
#: position inflates. The first run of this board did exactly that: QB dollars
#: totalled $730 against $598 for RB, and $26 landed on a QB Sleeper prices at
#: $10. Correcting the roster shape is fixing the setup, not fitting the
#: output.
CONTEXTS: Dict[str, Dict[Position, int]] = {
    "balanced": {Position.QB: 3, Position.RB: 5, Position.WR: 5,
                 Position.TE: 2},
    "qb_light": {Position.QB: 2, Position.RB: 5, Position.WR: 6,
                 Position.TE: 2},
    "qb_heavy": {Position.QB: 4, Position.RB: 4, Position.WR: 5,
                 Position.TE: 2},
}


def _counts_ok(counts: Dict[Position, int]) -> bool:
    """Fifteen players that can actually field the eight starting slots."""
    if sum(counts.values()) != 15:
        return False
    qb, rb = counts.get(Position.QB, 0), counts.get(Position.RB, 0)
    wt = counts.get(Position.WR, 0) + counts.get(Position.TE, 0)
    if qb < 1 or rb < 2 or wt < 3:
        return False
    # six non-QB starters (2 RB + 3 WR/TE + flex) plus a superflex that takes
    # anyone: the roster must supply eight bodies across those shapes.
    return (rb + wt) >= 6 and (qb + rb + wt) >= 8


@dataclass(frozen=True)
class PlayerCE:
    """One player's simulated marginal championship contribution."""

    player_id: int
    name: str
    position: str
    anchor: Optional[int]
    delta_by_context: Dict[str, float]
    displaced_by_context: Dict[str, int]

    @property
    def score(self) -> float:
        """Median of the three contexts. A median, not a mean: one context
        disagreeing should move the ranking, not decide it."""
        return statistics.median(self.delta_by_context.values())

    @property
    def delta_range(self) -> Tuple[float, float]:
        v = list(self.delta_by_context.values())
        return (min(v), max(v))

    @property
    def signs_agree(self) -> bool:
        v = list(self.delta_by_context.values())
        return all(x > 0 for x in v) or all(x < 0 for x in v)

    def to_dict(self) -> Dict[str, object]:
        lo, hi = self.delta_range
        return {"player_id": self.player_id, "name": self.name,
                "position": self.position, "anchor": self.anchor,
                "score": round(self.score, 6),
                "delta_by_context": {k: round(v, 6)
                                     for k, v in self.delta_by_context.items()},
                "delta_low": round(lo, 6), "delta_high": round(hi, 6),
                "signs_agree": self.signs_agree}


@dataclass(frozen=True)
class BoardEntry:
    """One player's rough CE dollars, and how confident the three contexts are."""

    player_id: int
    name: str
    position: str
    anchor: Optional[int]
    ce: PlayerCE
    price_by_context: Dict[str, int]

    @property
    def center(self) -> int:
        return int(statistics.median(self.price_by_context.values()))

    @property
    def low(self) -> int:
        return int(min(self.price_by_context.values()))

    @property
    def high(self) -> int:
        return int(max(self.price_by_context.values()))

    @property
    def confidence(self) -> str:
        """How much the three contexts disagree, in dollars.

        This measures context sensitivity and nothing else. It is not a
        statement about how close the number is to a true maximum bid, because
        this method does not compute one.
        """
        spread = self.high - self.low
        tight = spread <= max(3, int(round(0.25 * max(1, self.center))))
        if tight and self.ce.signs_agree:
            return "MEDIUM"
        return "LOW"

    def to_dict(self) -> Dict[str, object]:
        return {"player_id": self.player_id, "name": self.name,
                "position": self.position, "anchor": self.anchor,
                "rough_ce_max": self.center, "range_low": self.low,
                "range_high": self.high, "confidence": self.confidence,
                "price_by_context": dict(self.price_by_context),
                "ce": self.ce.to_dict()}


@dataclass
class OpeningCEBoard:
    """The precomputed board, and the fingerprint that makes it trustworthy."""

    entries: Dict[int, BoardEntry]
    fingerprint: str
    contexts: Dict[str, List[int]]
    seasons: int
    seed: int
    runtime_s: float
    coverage: Dict[str, int]
    legal_max_opening: int

    def get(self, player_id: int) -> Optional[BoardEntry]:
        return self.entries.get(int(player_id))

    def to_dict(self) -> Dict[str, object]:
        return {"fingerprint": self.fingerprint, "seasons": self.seasons,
                "seed": self.seed, "runtime_s": round(self.runtime_s, 1),
                "coverage": self.coverage,
                "legal_max_opening": self.legal_max_opening,
                "contexts": {k: list(v) for k, v in self.contexts.items()},
                "label": ROUGH_LABEL, "disclosure": ROUGH_DISCLOSURE,
                "entries": [e.to_dict()
                            for e in sorted(self.entries.values(),
                                            key=lambda x: (-x.center,
                                                           x.player_id))],
                "warning": "LOCAL ONLY -- gitignored draft-day board."}

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=1), encoding="utf-8")
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path, expect_fingerprint: str) -> Optional["OpeningCEBoard"]:
        """Load only if the fingerprint matches. A stale board is never shown.

        Returning ``None`` is the point: the dashboard then says CE PRECOMPUTE
        REQUIRED rather than displaying numbers computed for a different pool,
        scenario or league.
        """
        path = Path(path)
        if not path.exists():
            return None
        try:
            blob = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if str(blob.get("fingerprint")) != str(expect_fingerprint):
            return None
        entries: Dict[int, BoardEntry] = {}
        try:
            for e in blob["entries"]:
                ce = e["ce"]
                entries[int(e["player_id"])] = BoardEntry(
                    player_id=int(e["player_id"]), name=e["name"],
                    position=e["position"], anchor=e.get("anchor"),
                    price_by_context={k: int(v) for k, v in
                                      e["price_by_context"].items()},
                    ce=PlayerCE(
                        player_id=int(e["player_id"]), name=e["name"],
                        position=e["position"], anchor=e.get("anchor"),
                        delta_by_context={k: float(v) for k, v in
                                          ce["delta_by_context"].items()},
                        displaced_by_context={}))
            return cls(entries=entries, fingerprint=str(blob["fingerprint"]),
                       contexts={k: list(v) for k, v in
                                 blob.get("contexts", {}).items()},
                       seasons=int(blob.get("seasons", RANK_SEASONS)),
                       seed=int(blob.get("seed", RANK_SEED)),
                       runtime_s=float(blob.get("runtime_s", 0.0)),
                       coverage=dict(blob.get("coverage", {})),
                       legal_max_opening=int(blob.get("legal_max_opening", 186)))
        except (KeyError, TypeError, ValueError):
            return None


# ---------------------------------------------------------------------------
# Representative rosters
# ---------------------------------------------------------------------------


def build_contexts(state: AuctionState, points_by_id: Dict[int, float],
                   *, contexts: Optional[Dict[str, Dict[Position, int]]] = None
                   ) -> Tuple[Dict[str, List[int]], List[List[int]]]:
    """Three focus rosters and the fixed eleven-team field they play against.

    Deterministic throughout: players are taken in descending projected points
    with ``player_id`` breaking ties, so the same board always yields the same
    contexts. The field is representative and held fixed on purpose -- this
    method does not model the future auction, and pretending otherwise by
    randomising the opponents would add noise without adding information.
    """
    contexts = contexts or CONTEXTS
    spec = {s.player_id: s for s in state.pool}
    by_pos: Dict[Position, List[int]] = {}
    for pid, s in spec.items():
        by_pos.setdefault(s.position, []).append(pid)
    for pos in by_pos:
        by_pos[pos].sort(key=lambda p: (-points_by_id.get(p, 0.0), p))

    focus: Dict[str, List[int]] = {}
    used: set = set()
    for name, counts in contexts.items():
        if not _counts_ok(counts):
            raise ValueError(f"context {name!r} is not a legal 15-man roster")
        roster: List[int] = []
        for pos, n in sorted(counts.items(), key=lambda kv: int(kv[0])):
            pool = [p for p in by_pos.get(pos, []) if p not in roster]
            if len(pool) < n:
                raise ValueError(f"not enough {pos.name} for context {name!r}")
            roster.extend(pool[:n])
        focus[name] = sorted(roster)
        used.update(roster)

    # Eleven rivals, filled from what no focus roster used, in the same
    # deterministic order, each a legal fifteen.
    queues: Dict[Position, List[int]] = {}
    for pos, ids in by_pos.items():
        queues[pos] = [p for p in ids if p not in used]
    shape = {Position.QB: 2, Position.RB: 5, Position.WR: 6, Position.TE: 2}
    field: List[List[int]] = []
    for _ in range(11):
        team: List[int] = []
        for pos, n in sorted(shape.items(), key=lambda kv: int(kv[0])):
            q = queues.get(pos, [])
            if len(q) < n:
                raise ValueError(
                    f"pool too small to build the representative field: "
                    f"needed {n} more {pos.name}")
            team.extend(q[:n])
            queues[pos] = q[n:]
        field.append(sorted(team))
    return focus, field


def _displace(roster: Sequence[int], candidate: int,
              spec: Dict[int, object], points_by_id: Dict[int, float]
              ) -> Optional[int]:
    """The weakest roster member we may drop and stay legal.

    Legality is checked on the resulting shape, not assumed: dropping the last
    quarterback from a superflex roster is cheap in points and illegal in fact,
    so the candidate list is filtered by ``_counts_ok`` before the weakest is
    taken.
    """
    cand_pos = spec[candidate].position
    order = sorted(roster, key=lambda p: (points_by_id.get(p, 0.0), -p))
    for victim in order:
        if victim == candidate:
            continue
        counts: Dict[Position, int] = {}
        for p in roster:
            if p == victim:
                continue
            counts[spec[p].position] = counts.get(spec[p].position, 0) + 1
        counts[cand_pos] = counts.get(cand_pos, 0) + 1
        if _counts_ok(counts):
            return victim
    return None


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_players(state: AuctionState, cache: RoughWorldCache,
                  focus: Dict[str, List[int]], field: Sequence[Sequence[int]],
                  players: Sequence[int], *,
                  points_by_id: Dict[int, float],
                  name_by_id: Dict[int, str],
                  anchor_by_id: Dict[int, Optional[int]],
                  progress=None) -> Dict[int, PlayerCE]:
    """Paired ΔCE for every player, in each representative context.

    Both arms read the identical cached seasons, so a player on both rosters
    contributes an exact zero to the difference and the season-to-season noise
    that would otherwise swamp a ranking cancels.
    """
    spec = {s.player_id: s for s in state.pool}
    out: Dict[int, PlayerCE] = {}
    baselines: Dict[Tuple[str, Tuple[int, ...]], float] = {}

    def ce_of(roster: Sequence[int]) -> float:
        rosters = [list(roster)] + [list(t) for t in field]
        _, champion = cache.league_outcomes(rosters)
        return float((champion == 0).mean())

    for i, pid in enumerate(players):
        deltas: Dict[str, float] = {}
        displaced: Dict[str, int] = {}
        for cname, base in focus.items():
            if pid in base:
                # He is already in this shape. The honest comparison is still
                # "with him versus without him": drop him and put back the best
                # player this context does not already hold.
                without = [p for p in base if p != pid]
                filler = next(
                    (p for p in sorted(spec,
                                       key=lambda x: (-points_by_id.get(x, 0.0), x))
                     if p not in base and spec[p].position == spec[pid].position),
                    None)
                if filler is None:
                    continue
                without = sorted(without + [filler])
                with_ = list(base)
                displaced[cname] = filler
            else:
                victim = _displace(base, pid, spec, points_by_id)
                if victim is None:
                    continue
                without = list(base)
                with_ = sorted([p for p in base if p != victim] + [pid])
                displaced[cname] = victim
            key = (cname, tuple(without))
            if key not in baselines:
                baselines[key] = ce_of(without)
            deltas[cname] = ce_of(with_) - baselines[key]
        if len(deltas) < 2:
            continue
        out[pid] = PlayerCE(
            player_id=pid, name=name_by_id.get(pid, str(pid)),
            position=spec[pid].position.name, anchor=anchor_by_id.get(pid),
            delta_by_context=deltas, displaced_by_context=displaced)
        if progress is not None and (i % 20 == 0 or i == len(players) - 1):
            progress(i + 1, len(players))
    return out


# ---------------------------------------------------------------------------
# Rank -> dollars
# ---------------------------------------------------------------------------


def assign_dollars(scores: Dict[int, PlayerCE], base_prices: Dict[int, int],
                   context: str) -> Dict[int, int]:
    """Rank-map the real dollar distribution onto the CE ordering.

    The dollar *multiset* is preserved exactly -- the same prices, the same
    total, the same number of players at each level -- and only the assignment
    changes. That is why no CE-to-dollar coefficient appears anywhere: the
    scale comes from the market, and CE only decides the order.

    Ties in CE break on ``player_id``, so the mapping is deterministic.
    """
    ranked = sorted(scores.values(),
                    key=lambda pc: (-pc.delta_by_context.get(context, float("-inf")),
                                    pc.player_id))
    dollars = sorted((base_prices[p.player_id] for p in ranked), reverse=True)
    return {p.player_id: int(max(1, d)) for p, d in zip(ranked, dollars)}


def build_opening_board(state: AuctionState, cache: RoughWorldCache, *,
                        players: Sequence[int], points_by_id, name_by_id,
                        anchor_by_id, base_prices: Dict[int, int],
                        fingerprint: str, legal_max_opening: int,
                        contexts=None, progress=None) -> OpeningCEBoard:
    t0 = time.perf_counter()
    focus, fieldrosters = build_contexts(state, points_by_id, contexts=contexts)
    scores = score_players(state, cache, focus, fieldrosters, players,
                           points_by_id=points_by_id, name_by_id=name_by_id,
                           anchor_by_id=anchor_by_id, progress=progress)
    per_context = {c: assign_dollars(scores, base_prices, c) for c in focus}
    entries: Dict[int, BoardEntry] = {}
    for pid, pc in scores.items():
        prices = {c: min(int(legal_max_opening), m[pid])
                  for c, m in per_context.items() if pid in m}
        if not prices:
            continue
        entries[pid] = BoardEntry(
            player_id=pid, name=pc.name, position=pc.position,
            anchor=pc.anchor, ce=pc, price_by_context=prices)
    coverage: Dict[str, int] = {}
    for e in entries.values():
        coverage[e.position] = coverage.get(e.position, 0) + 1
    coverage["TOTAL"] = len(entries)
    return OpeningCEBoard(
        entries=entries, fingerprint=fingerprint,
        contexts={k: list(v) for k, v in focus.items()},
        seasons=cache.seasons, seed=cache.seed,
        runtime_s=time.perf_counter() - t0, coverage=coverage,
        legal_max_opening=int(legal_max_opening))


# ---------------------------------------------------------------------------
# Live adjustment
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LiveValue:
    """The opening number carried forward through the room's own evidence."""

    player_id: int
    opening: int
    opening_low: int
    opening_high: int
    room_multiplier: float
    position_multiplier: float
    reconcile_multiplier: float
    legal_max: int
    live: int
    live_low: int
    live_high: int
    clamped: bool
    confidence: str

    def to_dict(self) -> Dict[str, object]:
        return {"player_id": self.player_id, "opening_rough_ce_max": self.opening,
                "opening_low": self.opening_low, "opening_high": self.opening_high,
                "room_multiplier": round(self.room_multiplier, 4),
                "position_multiplier": round(self.position_multiplier, 4),
                "reconcile_multiplier": round(self.reconcile_multiplier, 4),
                "legal_max": self.legal_max, "live_rough_ce": self.live,
                "live_low": self.live_low, "live_high": self.live_high,
                "clamped": self.clamped,
                "clamp_label": "EXACT LEGAL LIMIT" if self.clamped else "",
                "confidence": self.confidence,
                "formula": ("live = opening x room_multiplier x "
                            "position_multiplier, reconciled to remaining "
                            "auction dollars, clamped to our exact legal max")}


def live_value(entry: BoardEntry, *, market, position: str,
               legal_max: int, reconcile_multiplier: float = 1.0) -> LiveValue:
    """Opening rough CE carried through the market the room has revealed.

    Every factor is one the project already computes. The room and position
    multipliers come from :class:`MarketState`, which learned them from
    completed sales; the reconciliation factor comes from the dollars actually
    left in the room; the cap is the exact legal maximum. No new coefficient is
    introduced, and none of these is a fresh CE simulation.
    """
    room = float(market.room_effect.multiplier) if market is not None else 1.0
    levels = market.position_levels if market is not None else {}
    pos = float(levels[position].multiplier) if position in levels else 1.0
    factor = room * pos * float(reconcile_multiplier)

    def carry(v: int) -> int:
        return max(1, int(round(v * factor)))

    live = carry(entry.center)
    lo, hi = carry(entry.low), carry(entry.high)
    clamped = live > legal_max or hi > legal_max
    return LiveValue(
        player_id=entry.player_id, opening=entry.center,
        opening_low=entry.low, opening_high=entry.high,
        room_multiplier=room, position_multiplier=pos,
        reconcile_multiplier=float(reconcile_multiplier),
        legal_max=int(legal_max),
        live=min(live, int(legal_max)), live_low=min(lo, int(legal_max)),
        live_high=min(hi, int(legal_max)), clamped=clamped,
        confidence=entry.confidence)


def board_fingerprint(state: AuctionState, pool_digest: str, *,
                      base_prices: Dict[int, int], seasons: int, seed: int,
                      contexts=None, scenario: str = "base") -> str:
    """Everything that changes the board. A miss rebuilds; it never adapts."""
    contexts = contexts or CONTEXTS
    h = hashlib.sha256()
    h.update(state.fingerprint().encode())
    h.update(pool_digest.encode())
    h.update(repr(state.settings).encode())
    h.update(f"{scenario}|{seasons}|{seed}".encode())
    for name in sorted(contexts):
        h.update(name.encode())
        for pos in sorted(contexts[name], key=int):
            h.update(f"{int(pos)}:{contexts[name][pos]}".encode())
    for pid in sorted(base_prices):
        h.update(f"{pid}:{base_prices[pid]}".encode())
    return h.hexdigest()[:16]
