"""The first targeted tactical pilot on the real board. Twelve players, not a board.

What this is: a bounded check that the machinery which passed every fabricated
test survives contact with real projections, real anchors and a real 549-player
contract. Twelve candidates, chosen by transparent data rather than by name,
across four positions and three market tiers.

**What this is not**: a draft board. It prices twelve players out of ~549 at a
single point in the auction (the empty room), which is the easiest state to
model and the least useful one to bid from. Nothing here is a recommendation.

Every player-level output — names, dollars, per-player CE — is written only
under ``local_data/``, which is gitignored. What this module will hand back for
committing is :class:`SanitizedReport`, which carries counts, rates, positional
aggregates and runtimes, and which *refuses* to serialize a player name or a
player-level dollar figure.
"""

from __future__ import annotations

import dataclasses
import json
import math
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ..auction.completion import ComparisonCast, CompletionSettings
from ..auction.costs import CostBook
from ..auction.proxy import ProxyEvaluator
from ..auction.state import AuctionState, new_auction
from ..league import DEFAULT_LEAGUE, LeagueSettings, Position
from ..market.anchors import load_sleeper_csv
from ..market.costbook import cost_book_from_prior
from ..market.live import MarketState
from ..market.prior import MarketPrior, build_market_prior
from ..players import PlayerSpec

__all__ = [
    "PilotInputs",
    "RealBoard",
    "load_real_board",
    "OWNER_IDS",
    "Candidate",
    "select_candidates",
    "CandidateDiagnostics",
    "diagnose",
    "SanitizedReport",
    "SanitizationError",
    "TIER_LABELS",
]

#: Twelve structurally identical owners. Names carry no information and are not
#: real managers; opponent identity is meaningless in an empty room and this
#: module asserts that rather than assuming it.
OWNER_IDS: Tuple[str, ...] = tuple(f"Team{i + 1:02d}" for i in range(12))

TIER_LABELS: Tuple[str, ...] = ("expensive", "mid", "cheap")


class SanitizationError(RuntimeError):
    """A committed artifact tried to carry real player-level detail."""


@dataclass(frozen=True)
class PilotInputs:
    """Where the real inputs live. All of them are gitignored paths."""

    contract: Path
    sleeper_csv: Path
    out_dir: Path
    pool_limit: int = 260
    market_scenario: str = "base"
    performance_scenario: str = "median_target/full_health/week_sd/exclude"

    def missing(self) -> List[str]:
        return [str(p) for p in (self.contract, self.sleeper_csv)
                if not p.exists()]


@dataclass
class RealBoard:
    """The real opening auction, its market, and everything that did not map."""

    state: AuctionState
    cast: ComparisonCast
    prior: MarketPrior
    market: MarketState
    costs: Dict[str, CostBook]
    """One book per market scenario: ``low`` / ``base`` / ``high``."""
    key_by_id: Dict[int, str]
    name_by_id: Dict[int, str]
    """LOCAL ONLY. Never written to a committed artifact."""
    coverage: Dict[str, object]

    @property
    def focus_owner_id(self) -> str:
        return self.state.focus_owner_id

    def key_for(self, player_id: int) -> Optional[str]:
        return self.key_by_id.get(player_id)


def load_real_board(inputs: PilotInputs, *,
                    settings: LeagueSettings = DEFAULT_LEAGUE) -> RealBoard:
    """Contract -> PlayerSpecs -> anchors -> prior -> the empty 12-team room.

    Reuses the repository's existing ingestion, mapping and market layers
    verbatim. Nothing here re-derives a projection or invents a second pipeline.
    """
    from ..realdata.mapping import (PlayerSpecMappingConfig,
                                    map_contract_to_playerspecs,
                                    positional_fits_from_contract)

    payload = json.loads(inputs.contract.read_text(encoding="utf-8"))
    cv, miss = positional_fits_from_contract(payload)
    cfg = PlayerSpecMappingConfig()
    mapped = map_contract_to_playerspecs(payload, cfg, positional_miss=miss,
                                         positional_cv=cv,
                                         limit=inputs.pool_limit)
    specs: List[PlayerSpec] = list(mapped.specs)
    if not specs:
        raise ValueError("the contract produced no usable PlayerSpecs")

    key_by_id = {m.spec.player_id: m.canonical_key for m in mapped.players}
    name_by_id = {m.spec.player_id: m.spec.name for m in mapped.players}

    book = load_sleeper_csv(inputs.sleeper_csv)
    ids = {k: pid for pid, k in key_by_id.items()}
    # Only anchors that reach a mapped player can price this board; the rest are
    # players the contract does not carry and are counted, not silently dropped.
    prior = build_market_prior(book, settings=settings, player_ids=ids,
                               notes="real 2026 Sleeper 2qb anchors")
    market = MarketState(prior=prior)

    anchored = sum(1 for pid in key_by_id
                   if key_by_id[pid] in prior.by_key
                   and prior.by_key[key_by_id[pid]].draftable)
    with_injury = sum(1 for m in mapped.players
                      if m.injury.source == "individual")
    unresolved = sum(1 for m in mapped.players if m.unresolved_placeholders)

    pos_counts: Dict[str, int] = {}
    for s in specs:
        pos_counts[Position(int(s.position)).name] = pos_counts.get(
            Position(int(s.position)).name, 0) + 1

    costs = {}
    for scenario in ("low", "base", "high"):
        cb = cost_book_from_prior(prior, scenario)
        # Every mapped player needs a price; unanchored players fall to the
        # league minimum, which is a stated floor and not a valuation.
        gaps = {pid: settings.min_bid for pid in key_by_id
                if pid not in cb}
        costs[scenario] = cb.with_costs(gaps) if gaps else cb

    state = new_auction(specs, OWNER_IDS, OWNER_IDS[0], settings=settings)
    state.validate()
    # Placeholder cast: every rival slot is empty in an opening room and the
    # joint-allocation layer fills them from the shared board.
    cast = ComparisonCast(0, tuple(() for _ in OWNER_IDS), OWNER_IDS)

    coverage = {
        "contract_players": len(payload.get("players", [])),
        "mapped_playerspecs": len(specs),
        "pool_limit": inputs.pool_limit,
        "position_counts": pos_counts,
        "with_market_anchor": anchored,
        "without_market_anchor": len(specs) - anchored,
        "with_individual_injury": with_injury,
        "injury_fallback_positional": len(specs) - with_injury,
        "unresolved_placeholder_fields": unresolved,
        "mapping_warnings": len(mapped.warnings),
        "mapping_skipped": len(mapped.skipped),
        "anchor_rows": len(book.anchors),
        "market_scenario": inputs.market_scenario,
        "performance_scenario": inputs.performance_scenario,
    }
    return RealBoard(state=state, cast=cast, prior=prior, market=market,
                     costs=costs, key_by_id=key_by_id, name_by_id=name_by_id,
                     coverage=coverage)


# ---------------------------------------------------------------------------
# Candidate selection -- transparent, deterministic, no hard-coded names
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    """One selected pilot player. The name lives only in local output."""

    player_id: int
    position: str
    tier: str
    position_rank: int
    base_mean: float
    anchor_raw: Optional[str]
    anchor_display: Optional[int]
    price_low: Optional[int]
    price_base: Optional[int]
    price_high: Optional[int]
    canonical_key: Optional[str]

    def to_dict(self, *, include_identity: bool = False) -> Dict[str, object]:
        out = {k: v for k, v in dataclasses.asdict(self).items()
               if k != "canonical_key"}
        if include_identity:
            out["canonical_key"] = self.canonical_key
        else:
            out.pop("player_id", None)
        return out


def select_candidates(board: RealBoard, *,
                      per_position: int = 3,
                      positions: Sequence[str] = ("QB", "RB", "WR", "TE"),
                      ) -> Tuple[List[Candidate], List[str]]:
    """Three tiers per position, chosen by market band and position rank.

    Deterministic and name-free: players are ordered by their base clearing
    price and then by projection, and the expensive/mid/cheap representatives
    are taken at fixed quantiles of that order. A player who fails validation is
    replaced by his immediate neighbour, and the substitution is reported.
    """
    notes: List[str] = []
    out: List[Candidate] = []
    by_pos: Dict[str, List[PlayerSpec]] = {}
    for s in board.state.available_specs:
        by_pos.setdefault(Position(int(s.position)).name, []).append(s)

    for pos in positions:
        group = by_pos.get(pos, [])
        if not group:
            notes.append(f"{pos}: no mapped players; position skipped")
            continue

        def price_of(spec: PlayerSpec) -> int:
            key = board.key_for(spec.player_id)
            p = board.prior.by_key.get(key) if key else None
            return p.base_price if p and p.draftable else 0

        ranked = sorted(group, key=lambda s: (-price_of(s), -s.base_mean,
                                              s.player_id))
        n = len(ranked)
        # Fixed quantiles of the position's own price order. Not thresholds:
        # a shallow position and a deep one both yield three representatives.
        picks = [0, min(n - 1, max(1, n // 6)), min(n - 1, max(2, n // 3))]
        seen: set = set()
        for tier, idx in zip(TIER_LABELS[:per_position], picks):
            chosen = None
            for probe in range(idx, n):
                spec = ranked[probe]
                if spec.player_id in seen:
                    continue
                if board.state.purchase_shortfall(
                        spec.player_id, board.focus_owner_id, 1) is not None:
                    notes.append(
                        f"{pos}/{tier}: rank {probe} refused by the auction "
                        f"validator; advanced one")
                    continue
                chosen = spec
                if probe != idx:
                    notes.append(
                        f"{pos}/{tier}: substituted rank {probe} for {idx}")
                break
            if chosen is None:
                notes.append(f"{pos}/{tier}: no legal candidate found")
                continue
            seen.add(chosen.player_id)
            key = board.key_for(chosen.player_id)
            p = board.prior.by_key.get(key) if key else None
            out.append(Candidate(
                player_id=chosen.player_id, position=pos, tier=tier,
                position_rank=ranked.index(chosen) + 1,
                base_mean=round(float(chosen.base_mean), 4),
                anchor_raw=(str(p.raw_value) if p and p.raw_value is not None
                            else None),
                anchor_display=p.display_anchor if p else None,
                price_low=p.low_price if p and p.draftable else None,
                price_base=p.base_price if p and p.draftable else None,
                price_high=p.high_price if p and p.draftable else None,
                canonical_key=key))
    return out, notes


# ---------------------------------------------------------------------------
# Pre-CE diagnostics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CandidateDiagnostics:
    """What the candidate does to a lineup, before any equity is simulated."""

    player_id: int
    position: str
    tier: str
    lineup_improvement: float
    candidate_weekly: float
    replacement_weekly: float
    displaced_player_id: Optional[int]
    role: str
    """``clear starter`` | ``marginal starter`` | ``bench/insurance``"""
    policy: str
    """``proxy only`` | ``4000-season audit`` | ``larger offline confirmation``"""
    reason: str

    def to_dict(self, *, include_identity: bool = False) -> Dict[str, object]:
        out = dataclasses.asdict(self)
        if not include_identity:
            out.pop("player_id", None)
            out.pop("displaced_player_id", None)
        return out


#: Thresholds derived from `docs/TACTICAL_POWER.md`, not invented here: a
#: candidate whose weekly starting improvement is ~0 produced |d|/SE <= 1.62 in
#: every roster context and never resolved, so buying seasons for him is waste.
BENCH_IMPROVEMENT = 0.25
MARGINAL_IMPROVEMENT = 3.0


def diagnose(board: RealBoard, cand: Candidate, *,
             proxy: ProxyEvaluator, roster_seed_pool: Sequence[int] = ()
             ) -> CandidateDiagnostics:
    """Weekly lineup improvement over a $1 replacement, in a real lineup.

    The fourteen other slots form a legal, realistic roster -- one QB, four RBs,
    six WRs, three TEs -- so the candidate competes against a real starting
    lineup for a real seat. Two failure modes are avoided by construction:
    filling with replacement-level players makes every candidate look like a
    starter, and filling with the top fourteen by projection makes every
    quarterback look worthless because a superflex board is quarterback-heavy
    at the top.
    """
    st = board.state
    focus = board.focus_owner_id
    cid = cand.player_id
    capacity = st.owner(focus).roster_capacity

    # The fourteen other slots must form a REALISTIC, LEGAL roster. Taking the
    # top fourteen by projection does not: in a superflex league quarterbacks
    # dominate the projection order, so that roster is fourteen quarterbacks --
    # illegal, and it makes every quarterback candidate look worthless because
    # the QB and superflex seats are already full. The first run of this pilot
    # produced exactly that, with QB1 improving a lineup by 0.03 points.
    #
    # One quarterback is deliberate: it fills the dedicated QB seat and leaves
    # the SUPERFLEX open, so a quarterback candidate competes against the best
    # RB/WR/TE fallback for that seat, which is the actual question this league
    # poses. It is not an assumption that a second quarterback is required.
    by_pos: Dict[str, List[PlayerSpec]] = {}
    for spec in st.available_specs:
        if spec.player_id == cid:
            continue
        by_pos.setdefault(Position(int(spec.position)).name, []).append(spec)
    for group in by_pos.values():
        group.sort(key=lambda s: (-s.base_mean, s.player_id))
    template = (("QB", 1), ("RB", 4), ("WR", 6), ("TE", 3))
    fillers: List[int] = []
    for pos, want in template:
        fillers.extend(s.player_id for s in by_pos.get(pos, [])[:want])
    if len(fillers) != capacity - 1:
        raise ValueError(
            f"the filler template yields {len(fillers)} players, not "
            f"{capacity - 1}; the real board is too thin at some position")
    cheap = sorted((s for s in st.available_specs if s.player_id != cid),
                   key=lambda s: (s.base_mean, s.player_id))
    replacement = next(s.player_id for s in cheap if s.player_id not in fillers)
    with_cand = fillers + [cid]
    with_repl = fillers + [replacement]
    a = proxy.strength(with_cand)
    b = proxy.strength(with_repl)
    gain = a - b

    displaced = None
    if gain > 1e-9:
        best = None
        for pid in fillers:
            trimmed = [p for p in with_cand if p != pid]
            g = a - proxy.strength(trimmed + [replacement])
            if best is None or g > best:
                best, displaced = g, pid

    if gain >= MARGINAL_IMPROVEMENT:
        role, policy = "clear starter", "4000-season audit"
        reason = (f"weekly starting improvement {gain:.2f} >= "
                  f"{MARGINAL_IMPROVEMENT}; strong-tier effects resolved at "
                  f"|d|/SE 14-33 in the power study")
    elif gain >= BENCH_IMPROVEMENT:
        role, policy = "marginal starter", "4000-season audit"
        reason = (f"weekly starting improvement {gain:.2f}; marginal-tier "
                  f"effects resolved at |d|/SE 2.5-5.4 in the power study")
    else:
        role, policy = "bench/insurance", "proxy only"
        reason = (f"weekly starting improvement {gain:.2f} < "
                  f"{BENCH_IMPROVEMENT}; zero-improvement candidates never "
                  f"resolved in the power study and CE seasons on them are "
                  f"waste, not evidence")
    return CandidateDiagnostics(
        player_id=cid, position=cand.position, tier=cand.tier,
        lineup_improvement=round(gain, 4), candidate_weekly=round(a, 4),
        replacement_weekly=round(b, 4), displaced_player_id=displaced,
        role=role, policy=policy, reason=reason)


# ---------------------------------------------------------------------------
# Sanitization
# ---------------------------------------------------------------------------


@dataclass
class SanitizedReport:
    """The only artifact from this pilot that may be committed.

    It refuses to serialize a player name or a player-level dollar figure. The
    refusal is a check rather than a convention because the cost of getting it
    wrong is committing proprietary projections to a public repository.
    """

    coverage: Dict[str, object]
    selection: Dict[str, object]
    diagnostics: List[Dict[str, object]]
    results: Dict[str, object]
    runtime: Dict[str, object]
    warnings: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)

    def check(self, forbidden_names: Iterable[str]) -> "SanitizedReport":
        blob = json.dumps(self.to_dict(), default=str).lower()
        for name in forbidden_names:
            n = str(name).strip().lower()
            if len(n) < 4:
                continue
            if n in blob:
                raise SanitizationError(
                    f"the sanitized report contains a real player name "
                    f"({n!r}). Player-level identity belongs only under "
                    f"local_data/.")
        for row in self.diagnostics:
            for banned in ("player_id", "displaced_player_id", "name",
                           "canonical_key"):
                if banned in row:
                    raise SanitizationError(
                        f"the sanitized report carries {banned!r}, which is "
                        f"player-level identity")
        return self

    def to_dict(self) -> Dict[str, object]:
        return {
            "label": ("SANITIZED AGGREGATE. Counts, rates and positional "
                      "aggregates only. No player name, no proprietary "
                      "projection row, no player-level dollar figure."),
            "coverage": self.coverage,
            "selection": self.selection,
            "diagnostics": self.diagnostics,
            "results": self.results,
            "runtime": self.runtime,
            "warnings": self.warnings,
            "assumptions": self.assumptions,
        }
