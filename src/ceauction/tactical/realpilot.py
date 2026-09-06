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
    "BranchRoster",
    "CandidateDiagnostics",
    "NO_CONTINGENCY_MODEL",
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
class BranchRoster:
    """One legally completed roster, and what it actually starts."""

    player_ids: Tuple[int, ...]
    composition: Dict[str, int]
    best_eight: float
    """Mean weekly points from the best LEGAL starting eight, chosen by the
    league's eligibility graph rather than by any positional template."""
    added_cost: int
    budget_left: int
    exactness: str
    start_shares: Dict[int, float]

    def to_dict(self, *, include_identity: bool = False) -> Dict[str, object]:
        out: Dict[str, object] = {
            "composition": self.composition,
            "best_eight": round(self.best_eight, 4),
            "added_cost": self.added_cost,
            "budget_left": self.budget_left,
            "exactness": self.exactness,
            "n_starters": sum(1 for v in self.start_shares.values() if v >= 0.5),
        }
        if include_identity:
            out["player_ids"] = list(self.player_ids)
            out["start_shares"] = {str(k): round(v, 4)
                                   for k, v in self.start_shares.items()}
        return out


@dataclass(frozen=True)
class CandidateDiagnostics:
    """What the candidate does to a QUOTA-FREE legal roster, before any equity.

    Both branches are produced by the existing bounded completion search over
    the real shared board, under exact lineup eligibility and the $1-per-slot
    reserve. No position is required, capped, or reserved a number of places.
    The composition of each branch is an output, never an input.
    """

    player_id: int
    position: str
    tier: str
    price: int
    without: BranchRoster
    with_candidate: BranchRoster
    lineup_improvement: float
    """best-legal-eight(with) - best-legal-eight(without) AT THIS PRICE.

    It can be negative, and a negative value is information rather than a bug:
    the money spent on the candidate is money the completion would otherwise
    have spent on players who contribute more. That is opportunity cost showing
    up exactly where it should."""

    improvement_at_min: float
    """The same difference with the candidate acquired at the $1 minimum.

    Separates *how good the player is* from *what he costs*. A player with a
    large value here and a negative value at his market price is a good player
    who is not worth his price on this board."""
    start_share: float
    """Fraction of scoring weeks the candidate is in his own roster's best eight."""
    displaced_player_id: Optional[int]
    displaced_start_share: float
    bench_delta: int
    """Change in the number of non-starting roster places."""
    role: str
    policy: str
    reason: str
    contingency_note: str
    price_monotonicity_violation: float = 0.0
    """``improvement(price) - improvement($1)``, which economics says is <= 0.

    Money not spent stays available, so acquiring the same player for less can
    never be worse. A positive value here is bounded-search error, not a
    finding, and it is reported rather than smoothed: at beam_width=32 it
    reached +8.2 points on the fabricated board, which is larger than the
    effects this diagnostic exists to measure."""

    @property
    def search_is_converged(self) -> bool:
        return self.price_monotonicity_violation <= 0.5

    def to_dict(self, *, include_identity: bool = False) -> Dict[str, object]:
        out: Dict[str, object] = {
            "position": self.position, "tier": self.tier, "price": self.price,
            "without": self.without.to_dict(include_identity=include_identity),
            "with_candidate": self.with_candidate.to_dict(
                include_identity=include_identity),
            "lineup_improvement": round(self.lineup_improvement, 4),
            "improvement_at_min": round(self.improvement_at_min, 4),
            "price_cost_of_slot": round(
                self.improvement_at_min - self.lineup_improvement, 4),
            "start_share": round(self.start_share, 4),
            "displaced_start_share": round(self.displaced_start_share, 4),
            "bench_delta": self.bench_delta,
            "role": self.role, "policy": self.policy, "reason": self.reason,
            "contingency_note": self.contingency_note,
            "price_monotonicity_violation": round(
                self.price_monotonicity_violation, 4),
            "search_is_converged": self.search_is_converged,
        }
        if include_identity:
            out["player_id"] = self.player_id
            out["displaced_player_id"] = self.displaced_player_id
        return out


#: Thresholds from `docs/TACTICAL_POWER.md`, applied to a quota-free
#: measurement. A candidate whose best-legal-eight contribution is ~0 produced
#: |d|/SE <= 1.62 in every roster context and never resolved.
BENCH_IMPROVEMENT = 0.25
MARGINAL_IMPROVEMENT = 3.0
STARTER_SHARE = 0.55
MARGINAL_SHARE = 0.15

#: No conditional-backfield or QB-injury-insurance mapping exists in this
#: repository. Every contingency statement must say so rather than inventing a
#: premium for depth the model cannot actually price.
NO_CONTINGENCY_MODEL = (
    "no conditional-backfield or QB-insurance mapping exists in this "
    "repository, so contingency value is NOT quantified here; a bench player's "
    "raw points are deliberately not converted into lineup value")


def _composition(state: AuctionState, ids: Sequence[int]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for pid in ids:
        name = Position(int(state.spec(pid).position)).name
        out[name] = out.get(name, 0) + 1
    return out


def _branch(board: "RealBoard", state: AuctionState, cast, costs,
            settings: CompletionSettings, proxy: ProxyEvaluator,
            reserved: frozenset, note: str) -> Optional[BranchRoster]:
    from ..auction.completion import complete_roster
    focus = state.focus_owner_id
    res = complete_roster(state, cast, costs, settings=settings,
                          owner_id=focus, evaluate_ce=False, default_cost=1,
                          proxy=proxy, reserved_ids=reserved, notes=note)
    if res.best is None:
        return None
    roster = tuple(res.best.roster)
    return BranchRoster(
        player_ids=roster, composition=_composition(state, roster),
        best_eight=float(proxy.strength(roster)),
        added_cost=res.best.added_cost,
        budget_left=state.owner(focus).budget_remaining - res.best.added_cost,
        exactness=res.result_kind, start_shares=proxy.lineup_shares(roster))


def diagnose(board: "RealBoard", cand: Candidate, *,
             proxy: ProxyEvaluator, price: Optional[int] = None,
             settings: Optional[CompletionSettings] = None
             ) -> CandidateDiagnostics:
    """Complete a legal roster with and without the candidate, and difference them.

    This replaces a hard-coded 1 QB / 4 RB / 6 WR / 3 TE reference roster. That
    template was legal and arbitrary, and arbitrary is fatal here: reserving
    three places for tight ends makes a fourth tight end look worthless, and
    reserving one for a quarterback flatters every second quarterback. It was
    deciding which real players earned a championship-equity audit, which is a
    decision no positional quota may make.

    What replaces it has no quotas at all. Both branches are bounded completion
    searches over the real remaining board under exact lineup eligibility, so
    the number of quarterbacks, backs, receivers and tight ends is whatever the
    projections, prices and the eligibility graph produce.
    """
    st = board.state
    focus = board.focus_owner_id
    cid = cand.player_id
    p = int(price if price is not None else (cand.price_base or 1))
    # Beam width matters more here than anywhere else in the project. The
    # with/without difference is a few points; at beam_width=32 the search was
    # not converged and produced violations of the price monotonicity invariant
    # of +5.9 to +8.2 points -- larger than the effects being measured. At
    # beam_width=160 the same violation collapses to +0.05. Anything narrower
    # is measuring the beam, not the player.
    cs = settings or CompletionSettings(
        beam_width=160, candidate_pool=90, proxy_candidates=48, finalists=3,
        max_candidates=400, proxy_reps=16)

    without = _branch(board, st, board.cast, board.costs["base"], cs, proxy,
                      frozenset({cid}), "quota-free completion WITHOUT candidate")
    shortfall = st.purchase_shortfall(cid, focus, p)
    with_state = st if shortfall is not None else st.apply_purchase(cid, focus, p)
    with_c = (None if shortfall is not None
              else _branch(board, with_state, board.cast, board.costs["base"],
                           cs, proxy, frozenset(),
                           "quota-free completion WITH candidate"))
    if without is None or with_c is None:
        raise ValueError(
            f"no legal completion for candidate {cid} at ${p}"
            + (f": {shortfall}" if shortfall else ""))

    gain = with_c.best_eight - without.best_eight
    share = with_c.start_shares.get(cid, 0.0)

    # The same difference at the $1 minimum, so price and player quality are
    # never conflated. Cheap: one extra bounded completion.
    at_min = gain
    if p > 1 and st.purchase_shortfall(cid, focus, 1) is None:
        floor = _branch(board, st.apply_purchase(cid, focus, 1), board.cast,
                        board.costs["base"], cs, proxy, frozenset(),
                        "quota-free completion WITH candidate at the $1 minimum")
        if floor is not None:
            at_min = floor.best_eight - without.best_eight
    gone = [pid for pid in without.player_ids
            if pid not in set(with_c.player_ids)]
    displaced, disp_share = None, 0.0
    if gone:
        displaced = max(gone, key=lambda x: without.start_shares.get(x, 0.0))
        disp_share = without.start_shares.get(displaced, 0.0)
    bench_without = sum(1 for v in without.start_shares.values() if v < 0.5)
    bench_with = sum(1 for v in with_c.start_shares.values() if v < 0.5)

    # Role from MEASURED behaviour in a quota-free roster, never from position.
    #
    # Improvement leads, not start share. On a fifteen-man roster with eight
    # starting slots almost everyone the completion chose starts most weeks, so
    # a share threshold on its own classifies every candidate as a starter and
    # audits the whole board. What decides whether championship equity is worth
    # simulating is whether the roster is BETTER with him at this price.
    if gain >= MARGINAL_IMPROVEMENT:
        role = "clear starter"
    elif gain >= BENCH_IMPROVEMENT:
        role = "marginal starter"
    elif share >= STARTER_SHARE:
        role = "replaceable starter"
    elif share >= MARGINAL_SHARE:
        role = "aggregate depth"
    elif gain <= 1e-6 and share < 1e-6:
        role = "currently redundant"
    else:
        role = "bench/insurance"

    if role in ("clear starter", "marginal starter"):
        policy = "4000-season audit"
        reason = (f"best-legal-eight improves {gain:+.2f}/wk at ${p} and the "
                  f"candidate holds a starting slot {share:.0%} of scoring "
                  f"weeks in his own quota-free completion")
    elif role == "replaceable starter":
        policy = "proxy only"
        reason = (f"the candidate starts {share:.0%} of weeks but the "
                  f"best-legal-eight improves only {gain:+.2f}/wk at ${p}: the "
                  f"completion is as good without him, because the ${p} buys "
                  f"more elsewhere. He is worth {at_min:+.2f}/wk at the $1 "
                  f"minimum, so this is a PRICE judgement, not a verdict on "
                  f"the player")
    else:
        policy = "proxy only"
        reason = (f"best-legal-eight improves {gain:+.2f}/wk (< "
                  f"{BENCH_IMPROVEMENT}) and the candidate starts {share:.0%} "
                  f"of weeks; zero-improvement candidates never resolved in "
                  f"the power study, so CE seasons on him are waste. This is a "
                  f"measured role, not a positional rule")
    return CandidateDiagnostics(
        player_id=cid, position=cand.position, tier=cand.tier, price=p,
        without=without, with_candidate=with_c, lineup_improvement=gain,
        improvement_at_min=at_min, start_share=share, displaced_player_id=displaced,
        displaced_start_share=disp_share,
        bench_delta=bench_with - bench_without, role=role, policy=policy,
        reason=reason, contingency_note=NO_CONTINGENCY_MODEL,
        price_monotonicity_violation=(gain - at_min if p > 1 else 0.0))


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
