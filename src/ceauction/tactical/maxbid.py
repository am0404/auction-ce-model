"""The tactical maximum bid: the highest price at which buying still wins.

Four prices, kept apart everywhere, because collapsing them is how a model
starts lying:

``sleeper_display_anchor``   what Sleeper shows. Lives in the market layer.
``expected_clearing_price``  what this room may pay. Lives in the market layer.
``ce_reservation_price``     what our equity says we can afford, against a best
                             alternative. Lives in :mod:`..auction.reservation`.
``tactical_max_bid``         what to actually bid, given who else can bid, who
                             gets him if we stop, and what that does to us.
                             **This module, and only this module.**

A tactical maximum may *use* the other three. It never overwrites or relabels
one. And it is never a single number: what comes out is a named set of
thresholds -- robust, base, permissive -- each with the rule that produced it.

Two levels of answer, because a ten-second bid timer cannot wait for a
multi-minute equity search:

``immediate``  proxy-backed or cached. Fast, signed, and **without a confidence
               interval**, because expected starting points is not championship
               equity and pretending otherwise would be the whole failure mode.
``audited``    the real CE engine, on matched seasons, with intervals and the
               existing selection/holdout discipline. Slow. Precomputed between
               nominations, never during one.

Every result says which it is, on every line and in every serialization.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

from ..auction.completion import (ComparisonCast, CompletionSettings,
                                  complete_roster)
from ..auction.costs import CostBook
from ..auction.counterfactual import (BuyPassResult, PassDestination,
                                      compare_buy_vs_pass)
from ..auction.proxy import ProxyEvaluator
from ..auction.reservation import price_ladder
from ..auction.state import AuctionRuleError, AuctionState
from ..market.live import MarketState
from .bidders import DEFAULT_BIDDER_SCENARIO
from .board import (BoardResult, BoardSettings, cast_from_board,
                    continue_shared_board)
from .endgame import EndgameReport, assess_endgame
from .joint import (ConservationError, JointComparison, build_joint_worlds,
                    evaluate_joint_arm)
from .nested import NestedLadder, build_nested_ladder
from .recipients import RecipientBranch, RecipientSet, enumerate_recipients

__all__ = [
    "TacticalScenario",
    "DEFAULT_SCENARIOS",
    "TacticalSettings",
    "PriceVerdict",
    "TacticalResult",
    "TacticalCache",
    "tactical_cache_key",
    "evaluate_tactical",
    "immediate_max_bid",
    "audited_max_bid",
    "format_tactical",
]


@dataclass(frozen=True)
class TacticalScenario:
    """One market reading crossed with one performance reading. Both stated.

    The two axes are kept apart deliberately. "The room pays up" and "these
    projections are optimistic" are different claims about different things,
    and a single combined dial would let one of them silently stand in for the
    other.
    """

    scenario_id: str
    market_scenario: str
    """``low`` | ``base`` | ``high`` -- which point of the clearing band prices
    the shared board and the recipients."""
    performance_scenario: str
    """Names the PlayerSpec/projection reading in force. Recorded, fingerprinted
    and reported; it never doubles as a market dial."""
    bidder_scenario: str = DEFAULT_BIDDER_SCENARIO
    is_base: bool = False
    """Exactly one scenario in a run must be the designated base case."""

    def cache_key(self) -> Tuple:
        import dataclasses
        return tuple((f.name, getattr(self, f.name))
                     for f in dataclasses.fields(self))

    def to_dict(self) -> Dict[str, object]:
        import dataclasses
        return {f.name: getattr(self, f.name)
                for f in dataclasses.fields(self)}


#: A small, explicitly named default set. Not a sweep and not exhaustive.
DEFAULT_SCENARIOS: Tuple[TacticalScenario, ...] = (
    TacticalScenario("cheap_room", "low", "fabricated_base", "conservative"),
    TacticalScenario("base", "base", "fabricated_base", "market_base",
                     is_base=True),
    TacticalScenario("hot_room", "high", "fabricated_base", "aggressive"),
)


@dataclass(frozen=True)
class TacticalSettings:
    """Everything that can change an answer, in one fingerprintable place."""

    mode: str = "immediate"
    """``immediate`` (proxy/cached) or ``audited`` (CE-backed)."""
    board: BoardSettings = BoardSettings()
    completion: CompletionSettings = CompletionSettings(
        beam_width=48, candidate_pool=40, proxy_candidates=48, finalists=3,
        proxy_reps=32, max_candidates=320)
    max_prices: int = 6
    refine: bool = False
    """Walk every integer in the unresolved transition gap. Without it a sparse
    ladder is reported as a BRACKET and never as an exact frontier."""
    max_recipients: int = 3
    max_worlds: int = 3
    """Reconciled joint worlds compared per audited arm. A real bound."""
    include_unavailable: bool = True
    default_cost: int = 1
    proxy_seed: int = 20260906
    proxy_reps: int = 32

    def __post_init__(self) -> None:
        if self.mode not in ("immediate", "audited"):
            raise ValueError(
                f"mode must be 'immediate' or 'audited', got {self.mode!r}")
        if self.max_prices < 2:
            raise ValueError("a ladder needs at least two prices")

    def cache_key(self) -> Tuple:
        return (self.mode, self.board.cache_key(),
                self.completion.cache_key(), self.max_prices, self.refine,
                self.max_recipients, self.max_worlds,
                self.include_unavailable,
                self.default_cost, self.proxy_seed, self.proxy_reps)

    def to_dict(self) -> Dict[str, object]:
        return {"mode": self.mode, "board": self.board.to_dict(),
                "completion": self.completion.to_dict(),
                "max_prices": self.max_prices, "refine": self.refine,
                "max_recipients": self.max_recipients,
                "max_worlds": self.max_worlds,
                "include_unavailable": self.include_unavailable,
                "default_cost": self.default_cost,
                "proxy_seed": self.proxy_seed, "proxy_reps": self.proxy_reps}


@dataclass(frozen=True)
class PriceVerdict:
    """One price, one scenario, one named recipient. Never an average."""

    price: int
    scenario_id: str
    market_scenario: str
    performance_scenario: str
    recipient_label: str
    recipient_owner: Optional[str]
    recipient_price: Optional[int]
    delta: float
    se: Optional[float]
    verdict: str
    """``favorable`` | ``unfavorable`` | ``unresolved``"""
    basis: str
    """``proxy (expected starting points, NO interval)`` or ``championship equity``"""
    selection_sims: Optional[int] = None
    holdout_sims: Optional[int] = None
    board_exactness: str = ""
    completion_kind: str = ""
    runtime_s: float = 0.0

    @property
    def ci95(self) -> Optional[Tuple[float, float]]:
        if self.se is None:
            return None
        return (self.delta - 1.96 * self.se, self.delta + 1.96 * self.se)

    def to_dict(self) -> Dict[str, object]:
        ci = self.ci95
        return {"price": self.price, "scenario_id": self.scenario_id,
                "market_scenario": self.market_scenario,
                "performance_scenario": self.performance_scenario,
                "recipient": self.recipient_label,
                "recipient_owner": self.recipient_owner,
                "recipient_price": self.recipient_price,
                "delta": round(self.delta, 6),
                "se": None if self.se is None else round(self.se, 6),
                "ci95": None if ci is None else [round(ci[0], 6), round(ci[1], 6)],
                "verdict": self.verdict, "basis": self.basis,
                "selection_sims": self.selection_sims,
                "holdout_sims": self.holdout_sims,
                "board_exactness": self.board_exactness,
                "completion_kind": self.completion_kind,
                "runtime_s": round(self.runtime_s, 3)}


@dataclass
class TacticalResult:
    """Named tactical thresholds, and every verdict they were derived from."""

    candidate_id: int
    focus_owner_id: str
    mode: str
    current_price: Optional[int]
    increment: int
    current_leader: Optional[str]
    endgame: EndgameReport
    recipients: RecipientSet
    scenarios: Tuple[TacticalScenario, ...]
    settings: TacticalSettings
    verdicts: Tuple[PriceVerdict, ...]
    tested_prices: Tuple[int, ...]
    ladder_is_dense: bool
    cache_key: str
    auction_fingerprint: str
    market_fingerprint: Optional[str]
    runtime_s: float = 0.0
    from_cache: bool = False
    notes: str = ""
    nested_ladder: Optional["NestedLadder"] = None
    """The cross-price opportunity set. ``None`` in proxy mode, which does not
    build one and therefore cannot claim nesting."""

    # --- the named prices -------------------------------------------------

    @property
    def legal_max(self) -> int:
        """Highest bid our budget, slots and this player's legality permit."""
        return self.endgame.our_legal_max

    @property
    def financial_control_threshold(self) -> Optional[int]:
        return self.endgame.financial_control_threshold

    def _by_price(self) -> Dict[int, List[PriceVerdict]]:
        out: Dict[int, List[PriceVerdict]] = {}
        for v in self.verdicts:
            out.setdefault(v.price, []).append(v)
        return out

    def _highest(self, ok) -> Optional[int]:
        best = None
        for price, group in sorted(self._by_price().items()):
            if ok(group):
                best = price
        return best

    @property
    def robust_tactical_max(self) -> Optional[int]:
        """Favorable against EVERY included recipient and EVERY scenario."""
        return self._highest(
            lambda g: bool(g) and all(v.verdict == "favorable" for v in g))

    @property
    def base_tactical_max(self) -> Optional[int]:
        """Favorable under the designated base scenario, against every recipient."""
        base_ids = {s.scenario_id for s in self.scenarios if s.is_base}
        if not base_ids:
            return None

        def ok(group: Sequence[PriceVerdict]) -> bool:
            rows = [v for v in group if v.scenario_id in base_ids]
            return bool(rows) and all(v.verdict == "favorable" for v in rows)

        return self._highest(ok)

    @property
    def permissive_ceiling(self) -> Optional[int]:
        """Favorable under AT LEAST ONE scenario/recipient. A ceiling, not advice."""
        return self._highest(
            lambda g: any(v.verdict == "favorable" for v in g))

    @property
    def robust_bracket(self) -> Optional[Tuple[int, Optional[int]]]:
        """``(highest favorable, lowest tested price above it)``.

        The upper end is ``None`` when nothing above was tested. On a sparse
        ladder this bracket -- not the lower number alone -- is the answer.
        """
        r = self.robust_tactical_max
        if r is None:
            return None
        above = [p for p in self.tested_prices if p > r]
        return (r, min(above) if above else None)

    @property
    def untested_gaps(self) -> Tuple[Tuple[int, int], ...]:
        out: List[Tuple[int, int]] = []
        ps = sorted(self.tested_prices)
        for a, b in zip(ps, ps[1:]):
            if b - a > 1:
                out.append((a + 1, b - 1))
        return tuple(out)

    @property
    def nonmonotonic(self) -> Tuple[Tuple[int, int], ...]:
        """``(cheaper unfavorable price, dearer favorable price)`` pairs.

        A real possibility, not an error: a shared board reacts to what we
        spend, and a different price can send a different player to a different
        rival. Reported and classified rather than smoothed into a frontier.
        """
        rows: List[Tuple[int, bool]] = []
        for price, group in sorted(self._by_price().items()):
            rows.append((price, bool(group) and
                         all(v.verdict == "favorable" for v in group)))
        out: List[Tuple[int, int]] = []
        for i, (p, ok) in enumerate(rows):
            if ok:
                for q, ok_q in rows[:i]:
                    if not ok_q:
                        out.append((q, p))
        return tuple(out)

    PROXY_BANNER = "PROXY ONLY -- NOT A CE RESERVATION PRICE"

    @property
    def is_audited(self) -> bool:
        return self.mode == "audited"

    @property
    def result_kind(self) -> str:
        """What kind of answer this is. Proxy results never borrow CE words.

        ``CE-audited``, ``resolved`` and ``reservation`` are reserved for the
        audited path. A proxy threshold that borrowed any of them would read as
        a CE claim in a transcript, and the withdrawn $1/$13 numbers from the
        previous branch are exactly what that looks like when it goes wrong.
        """
        if not self.is_audited:
            return f"{self.PROXY_BANNER} (proxy bracket, no interval)"
        if self.ladder_is_dense:
            return "CE-audited over reconciled joint worlds, dense ladder"
        return ("CE-audited over reconciled joint worlds, sparse ladder -- "
                "report the bracket, not a frontier")

    @property
    def nesting_ok(self) -> Optional[bool]:
        return None if self.nested_ladder is None else self.nested_ladder.is_nested

    def check_caps(self) -> None:
        """No tactical price may exceed the legal maximum. Enforced, not hoped."""
        for name in ("robust_tactical_max", "base_tactical_max",
                     "permissive_ceiling"):
            v = getattr(self, name)
            if v is not None and v > self.legal_max:
                raise AssertionError(
                    f"{name} ${v} exceeds the legal maximum ${self.legal_max}")

    def to_dict(self, *, include_verdicts: bool = True) -> Dict[str, object]:
        out: Dict[str, object] = {
            "candidate_id": self.candidate_id,
            "focus_owner_id": self.focus_owner_id,
            "mode": self.mode,
            "is_audited": self.is_audited,
            "result_kind": self.result_kind,
            "proxy_banner": None if self.is_audited else self.PROXY_BANNER,
            "nested_ladder": (None if self.nested_ladder is None
                              else self.nested_ladder.to_dict()),
            "current_price": self.current_price,
            "increment": self.increment,
            "current_leader": self.current_leader,
            "cache_key": self.cache_key,
            "auction_fingerprint": self.auction_fingerprint,
            "market_fingerprint": self.market_fingerprint,
            "from_cache": self.from_cache,
            "runtime_s": round(self.runtime_s, 3),
            "threshold_basis": ("CE-audited" if self.is_audited
                                else "proxy-only, NOT a CE reservation price"),
            "prices": {
                "legal_max": self.legal_max,
                "financial_control_threshold": self.financial_control_threshold,
                "robust_tactical_max": self.robust_tactical_max,
                "robust_bracket": list(self.robust_bracket)
                if self.robust_bracket else None,
                "base_tactical_max": self.base_tactical_max,
                "permissive_ceiling": self.permissive_ceiling,
            },
            "tested_prices": list(self.tested_prices),
            "ladder_is_dense": self.ladder_is_dense,
            "untested_gaps": [list(g) for g in self.untested_gaps],
            "nonmonotonic_pairs": [list(p) for p in self.nonmonotonic],
            "scenarios": [s.to_dict() for s in self.scenarios],
            "settings": self.settings.to_dict(),
            "recipients": self.recipients.to_dict(),
            "endgame": self.endgame.to_dict(),
            "notes": self.notes,
            "label": ("tactical maximum bid. Distinct from the Sleeper anchor, "
                      "from the expected clearing price and from the CE "
                      "reservation price; it uses them and replaces none."),
        }
        if include_verdicts:
            out["verdicts"] = [v.to_dict() for v in self.verdicts]
        return out


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def tactical_cache_key(state: AuctionState, candidate_id: int, *,
               market: Optional[MarketState],
               current_leader: Optional[str],
               current_price: Optional[int],
               increment: int,
               recipients: Sequence[str],
               scenarios: Sequence[TacticalScenario],
               settings: TacticalSettings,
               cast: ComparisonCast,
               costs: CostBook,
               regime: Optional[str] = None,
               nested_fingerprint: Optional[str] = None,
               ladder_prices: Optional[Sequence[int]] = None) -> str:
    h = hashlib.sha256()
    parts = [
        f"auction={state.fingerprint()}",
        f"pool={state.pool_fingerprint()}",
        f"settings_fp={state.settings_fingerprint()}",
        f"cast={cast.fingerprint()}",
        f"costs={costs.fingerprint()}",
        f"market={'none' if market is None else market.fingerprint()}",
        f"candidate={candidate_id}",
        f"leader={current_leader}",
        f"current_price={current_price}",
        f"increment={increment}",
        "recipients=" + "|".join(sorted(recipients)),
        "scenarios=" + "|".join(str(s.cache_key()) for s in scenarios),
        f"settings={settings.cache_key()}",
        # The mode is part of the identity, not a rendering choice. Without it
        # a cheap proxy result could satisfy a request for an audited one, and
        # the caller would read expected starting points as championship
        # equity. The prefix makes that structurally impossible.
        f"MODE={settings.mode}",
        f"selection={settings.completion.selection_sims}:"
        f"{settings.completion.selection_seed}",
        f"holdout={settings.completion.evaluation_sims}:"
        f"{settings.completion.evaluation_seed}",
        f"regime={regime or 'default'}",
        # The nested opportunity set is a deterministic function of the auction
        # state, the cast, the cost book, the completion/board settings and the
        # tested price list -- every one of which is already hashed above, the
        # price list explicitly. So a different nested set implies a different
        # key, and `test_cache_key_moves_with_the_nested_price_ladder` proves
        # it. The fingerprint is carried here too when the caller already has
        # one, so a stored entry names the set it was built from.
        "prices=" + ",".join(str(x) for x in (ladder_prices or ())),
        f"nested={nested_fingerprint or 'derived'}",
    ]
    h.update("\n".join(parts).encode("utf-8"))
    return f"{settings.mode[:3]}-" + h.hexdigest()[:24]


class TacticalCache:
    """Keyed by :func:`tactical_cache_key`. A miss is honest; a wrong hit is not."""

    def __init__(self) -> None:
        self._store: Dict[str, TacticalResult] = {}
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[TacticalResult]:
        hit = self._store.get(key)
        if hit is None:
            self.misses += 1
            return None
        self.hits += 1
        return replace(hit, from_cache=True)

    def put(self, key: str, value: TacticalResult) -> None:
        self._store[key] = value

    def __len__(self) -> int:
        return len(self._store)

    def stats(self) -> Dict[str, int]:
        return {"entries": len(self._store), "hits": self.hits,
                "misses": self.misses}


# ---------------------------------------------------------------------------
# Branch evaluation
# ---------------------------------------------------------------------------


@dataclass
class _Branch:
    """A completed league under one hypothesis, plus how it was completed."""

    value: float
    board: BoardResult
    completion_kind: str


def _complete_against_board(state: AuctionState, cast: ComparisonCast,
                            costs: CostBook, settings: TacticalSettings,
                            proxy: ProxyEvaluator, market: Optional[MarketState],
                            key_by_id: Optional[Dict[int, str]],
                            board_settings: BoardSettings,
                            protect: Sequence[int] = ()) -> _Branch:
    """Continue the shared board, then fill our roster from what is left."""
    board = continue_shared_board(state, settings=board_settings, costs=costs,
                                  market=market, key_by_id=key_by_id,
                                  protect=protect)
    new_cast = cast_from_board(board, cast, state)
    focus = state.focus_owner_id
    res = complete_roster(board.state, new_cast, costs,
                          settings=settings.completion, owner_id=focus,
                          evaluate_ce=False, default_cost=settings.default_cost,
                          proxy=proxy,
                          reserved_ids=board.reserved_ids(exclude_owner=focus),
                          notes="proxy branch over a shared board")
    if res.best is None:
        return _Branch(value=float("-inf"), board=board,
                       completion_kind=res.result_kind)
    # Our strength RELATIVE to the league we would be playing in. Absolute
    # starting points would be blind to who bought the candidate, which is the
    # one thing this layer exists to see: a player strengthening the rival we
    # already trail is worse for us than the same player strengthening the team
    # in last, even though our own roster is identical in both branches.
    rivals = [tuple(o.player_ids) for o in board.state.owners
              if o.owner_id != focus and len(o.player_ids) == len(res.best.roster)]
    field = 0.0
    if rivals:
        field = float(proxy.strength_many(rivals).mean())
    return _Branch(value=res.best.proxy - field, board=board,
                   completion_kind=res.result_kind)


def _proxy_verdict(delta: float) -> str:
    # No interval exists for a proxy comparison, so the only honest rule is the
    # sign, and a dead heat is unresolved rather than favorable.
    if abs(delta) < 1e-9:
        return "unresolved"
    return "favorable" if delta > 0 else "unfavorable"


def evaluate_tactical(
    state: AuctionState,
    cast: ComparisonCast,
    costs: CostBook,
    candidate_id: int,
    *,
    settings: TacticalSettings = TacticalSettings(),
    scenarios: Sequence[TacticalScenario] = DEFAULT_SCENARIOS,
    market: Optional[MarketState] = None,
    candidate_key: Optional[str] = None,
    key_by_id: Optional[Dict[int, str]] = None,
    current_price: Optional[int] = None,
    increment: int = 1,
    current_leader: Optional[str] = None,
    eligible: Optional[Sequence[str]] = None,
    prices: Optional[Sequence[int]] = None,
    cache: Optional[TacticalCache] = None,
    regime: Optional[str] = None,
) -> TacticalResult:
    """Walk a price ladder against every named recipient and every scenario.

    Buy and pass share matched randomness: one :class:`ProxyEvaluator` and one
    board seed serve both branches, so a player on both completed rosters draws
    identical availability and contributes exactly zero to the difference.
    """
    t0 = time.perf_counter()
    if increment < 1:
        raise ValueError("the bid increment must be at least $1")
    if not any(s.is_base for s in scenarios):
        raise ValueError(
            "one scenario must be flagged is_base: a base tactical maximum "
            "with no designated base case would be an unlabelled default")
    focus = state.focus_owner_id
    endgame = assess_endgame(state, candidate_id, current_price=current_price,
                             increment=increment)
    base_sc = next(s for s in scenarios if s.is_base)
    rset = enumerate_recipients(
        state, candidate_id, current_price=current_price, increment=increment,
        current_leader=current_leader, scenario=base_sc.bidder_scenario,
        market=market, candidate_key=candidate_key, costs=costs,
        eligible=eligible, max_named=settings.max_recipients,
        include_unavailable=settings.include_unavailable)

    branches = [b for b in rset.branches if b.legal]
    legal_max = endgame.our_legal_max
    floor = max(1, (current_price + increment) if current_price is not None
                else state.owner(focus).min_bid)
    if prices is not None:
        planned = tuple(sorted({int(p) for p in prices
                                if floor <= int(p) <= legal_max}))
    elif legal_max >= floor:
        planned = price_ladder(floor, legal_max, settings.max_prices)
    else:
        planned = ()
    key = tactical_cache_key(state, candidate_id, market=market,
                     current_leader=current_leader, current_price=current_price,
                     increment=increment,
                     recipients=[b.label for b in rset.branches],
                     scenarios=scenarios, settings=settings, cast=cast,
                     costs=costs, regime=regime, ladder_prices=planned)
    if cache is not None:
        hit = cache.get(key)
        if hit is not None:
            return hit

    if legal_max < floor:
        result = TacticalResult(
            candidate_id=candidate_id, focus_owner_id=focus, mode=settings.mode,
            current_price=current_price, increment=increment,
            current_leader=current_leader, endgame=endgame, recipients=rset,
            scenarios=tuple(scenarios), settings=settings, verdicts=(),
            tested_prices=(), ladder_is_dense=True, cache_key=key,
            auction_fingerprint=state.fingerprint(),
            market_fingerprint=None if market is None else market.fingerprint(),
            runtime_s=time.perf_counter() - t0,
            notes=("we cannot legally bid: "
                   + (endgame.us.exclusion_reason or "no legal price")))
        if cache is not None:
            cache.put(key, result)
        return result

    ladder_prices = planned
    ladder_prices = tuple(ladder_prices)
    dense = (len(ladder_prices) > 1
             and all(b - a == 1 for a, b in zip(ladder_prices, ladder_prices[1:])))

    proxy = ProxyEvaluator(state.pool, state.settings, settings.proxy_reps,
                           settings.proxy_seed)
    verdicts: List[PriceVerdict] = []

    # Audited mode builds ONE nested opportunity set across the whole ladder.
    # Re-running the beam per price offers different prices different choices,
    # which is what made a $1 purchase look worse than the identical $5 one.
    ladder: Optional[NestedLadder] = None
    audited_cache: Dict[Tuple, object] = {}
    if settings.mode == "audited":
        ladder = build_nested_ladder(
            state, cast, costs, candidate_id, ladder_prices,
            board_settings=settings.board, completion=settings.completion,
            market=market, key_by_id=key_by_id, proxy=proxy,
            default_cost=settings.default_cost)
        if not ladder.is_nested:
            raise ConservationError(
                "the opportunity set is not nested across prices: "
                f"{ladder.nesting_violations()[:3]}. A construction affordable "
                "at a higher price must be offered at every lower price, and "
                "evaluating a ladder that violates that would reproduce the "
                "$1-worse-than-$5 artefact this branch exists to remove.")

    for sc in scenarios:
        bset = replace(settings.board, market_scenario=sc.market_scenario,
                       bidder_scenario=sc.bidder_scenario)
        # Pass branches do not depend on OUR price, so each is built once per
        # scenario and reused across the whole ladder. That is not a shortcut:
        # our budget is unchanged when we pass, so the branch genuinely is the
        # same one at every price we might have stopped at.
        pass_cache: Dict[str, object] = {}
        for price in ladder_prices:
            if state.purchase_shortfall(candidate_id, focus, price) is not None:
                continue
            tb = time.perf_counter()
            if settings.mode == "immediate":
                buy_state = state.apply_purchase(candidate_id, focus, price)
                buy = _complete_against_board(buy_state, cast, costs, settings,
                                              proxy, market, key_by_id, bset)
            for b in branches:
                bt = time.perf_counter()
                label = b.label
                if settings.mode == "immediate":
                    if label not in pass_cache:
                        if b.kind == "unavailable":
                            ps = state.withdraw(candidate_id)
                        else:
                            ps = state.award_to_rival(candidate_id, b.owner_id,
                                                      int(b.price))
                        pass_cache[label] = _complete_against_board(
                            ps, cast, costs, settings, proxy, market,
                            key_by_id, bset)
                    pb = pass_cache[label]           # type: ignore[assignment]
                    delta = buy.value - pb.value     # type: ignore[union-attr]
                    verdicts.append(PriceVerdict(
                        price=price, scenario_id=sc.scenario_id,
                        market_scenario=sc.market_scenario,
                        performance_scenario=sc.performance_scenario,
                        recipient_label=label, recipient_owner=b.owner_id,
                        recipient_price=b.price, delta=delta, se=None,
                        verdict=_proxy_verdict(delta),
                        basis="proxy (expected starting points, NO interval; "
                              "NOT championship equity)",
                        board_exactness=buy.board.exactness,
                        completion_kind=buy.completion_kind,
                        runtime_s=time.perf_counter() - bt))
                else:
                    joint = audited_cache.get(("buy", sc.scenario_id, price))
                    if joint is None:
                        try:
                            feasible = ladder.by_price[price].feasible
                        except KeyError:
                            feasible = ()
                        if not feasible:
                            verdicts.append(PriceVerdict(
                                price=price, scenario_id=sc.scenario_id,
                                market_scenario=sc.market_scenario,
                                performance_scenario=sc.performance_scenario,
                                recipient_label=label, recipient_owner=b.owner_id,
                                recipient_price=b.price, delta=0.0, se=None,
                                verdict="unresolved",
                                basis="refused: no legal, affordable focus "
                                      "construction at this price",
                                runtime_s=time.perf_counter() - bt))
                            continue
                        try:
                            joint = evaluate_joint_arm(
                                build_joint_worlds(
                                    state.apply_purchase(candidate_id, focus,
                                                         price),
                                    cast, costs, board_settings=bset,
                                    completion=settings.completion,
                                    market=market, key_by_id=key_by_id,
                                    proxy=proxy,
                                    default_cost=settings.default_cost,
                                    branch_acquired=frozenset({candidate_id}),
                                    completions=feasible,
                                    max_worlds=settings.max_worlds),
                                focus_team_index=cast.focus_team_index,
                                proxy=proxy,
                                selection_sims=settings.completion.selection_sims,
                                selection_seed=settings.completion.selection_seed,
                                holdout_sims=settings.completion.evaluation_sims,
                                holdout_seed=settings.completion.evaluation_seed,
                                chunk=settings.completion.ce_chunk)
                        except (ConservationError, AuctionRuleError) as exc:
                            verdicts.append(PriceVerdict(
                                price=price, scenario_id=sc.scenario_id,
                                market_scenario=sc.market_scenario,
                                performance_scenario=sc.performance_scenario,
                                recipient_label=label, recipient_owner=b.owner_id,
                                recipient_price=b.price, delta=0.0, se=None,
                                verdict="unresolved",
                                basis=f"REFUSED (joint-world validation): {exc}",
                                runtime_s=time.perf_counter() - bt))
                            continue
                        audited_cache[("buy", sc.scenario_id, price)] = joint

                    pkey = ("pass", sc.scenario_id, label)
                    parm = audited_cache.get(pkey)
                    if parm is None:
                        if b.kind == "unavailable":
                            # Withdrawn, not acquired. Counting him as both a
                            # branch acquisition and a declared withdrawal
                            # doubles him and breaks the pool identity by one.
                            ps = state.withdraw(candidate_id)
                            extra = {"declared_withdrawn":
                                     frozenset({candidate_id}),
                                     "branch_acquired": frozenset()}
                        else:
                            ps = state.award_to_rival(candidate_id, b.owner_id,
                                                      int(b.price))
                            extra = {"branch_acquired":
                                     frozenset({candidate_id})}
                        try:
                            parm = evaluate_joint_arm(
                                build_joint_worlds(
                                    ps, cast, costs, board_settings=bset,
                                    completion=settings.completion,
                                    market=market, key_by_id=key_by_id,
                                    proxy=proxy,
                                    default_cost=settings.default_cost,
                                    max_worlds=settings.max_worlds, **extra),
                                focus_team_index=cast.focus_team_index,
                                proxy=proxy,
                                selection_sims=settings.completion.selection_sims,
                                selection_seed=settings.completion.selection_seed,
                                holdout_sims=settings.completion.evaluation_sims,
                                holdout_seed=settings.completion.evaluation_seed,
                                chunk=settings.completion.ce_chunk)
                        except (ConservationError, AuctionRuleError) as exc:
                            verdicts.append(PriceVerdict(
                                price=price, scenario_id=sc.scenario_id,
                                market_scenario=sc.market_scenario,
                                performance_scenario=sc.performance_scenario,
                                recipient_label=label, recipient_owner=b.owner_id,
                                recipient_price=b.price, delta=0.0, se=None,
                                verdict="unresolved",
                                basis=f"REFUSED (joint-world validation): {exc}",
                                runtime_s=time.perf_counter() - bt))
                            continue
                        audited_cache[pkey] = parm

                    cmp_ = JointComparison(
                        buy=joint, pass_arm=parm, price=price,
                        recipient=b.owner_id, recipient_price=b.price)
                    degenerate = (cmp_.delta_ce == 0.0
                                  and cmp_.allocations_identical)
                    verdicts.append(PriceVerdict(
                        price=price, scenario_id=sc.scenario_id,
                        market_scenario=sc.market_scenario,
                        performance_scenario=sc.performance_scenario,
                        recipient_label=label, recipient_owner=b.owner_id,
                        recipient_price=b.price, delta=cmp_.delta_ce,
                        se=cmp_.delta_se,
                        # A zero difference between two IDENTICAL allocations
                        # carries no information; the engine's lo >= 0 rule
                        # would call it favorable. A zero difference between
                        # two different allocations is a real dead heat and is
                        # reported as unresolved by the interval rule itself.
                        verdict="unresolved" if degenerate else cmp_.verdict,
                        basis=("CE-audited over reconciled joint worlds "
                               "(independent selection/holdout samples, paired "
                               "seasons)"
                               + ("; DEGENERATE: buy and pass produced an "
                                  "identical joint allocation"
                                  if degenerate else "")),
                        selection_sims=settings.completion.selection_sims,
                        holdout_sims=settings.completion.evaluation_sims,
                        board_exactness=joint.world.rival_board.exactness,
                        completion_kind=(
                            f"joint world {joint.world.fingerprint()} vs "
                            f"{parm.world.fingerprint()}; "
                            f"{joint.n_worlds_compared} worlds compared; "
                            f"conservation ok"),
                        runtime_s=time.perf_counter() - bt))

    result = TacticalResult(
        candidate_id=candidate_id, focus_owner_id=focus, mode=settings.mode,
        current_price=current_price, increment=increment,
        current_leader=current_leader, endgame=endgame, recipients=rset,
        scenarios=tuple(scenarios), settings=settings,
        verdicts=tuple(verdicts), tested_prices=tuple(ladder_prices),
        nested_ladder=ladder,
        ladder_is_dense=dense, cache_key=key,
        auction_fingerprint=state.fingerprint(),
        market_fingerprint=None if market is None else market.fingerprint(),
        runtime_s=time.perf_counter() - t0)
    result.check_caps()

    if settings.refine and not dense:
        gap = result.robust_bracket
        if gap is not None and gap[1] is not None and gap[1] - gap[0] > 1:
            extra = tuple(range(gap[0], gap[1] + 1))
            merged = tuple(sorted(set(ladder_prices) | set(extra)))
            refined = evaluate_tactical(
                state, cast, costs, candidate_id,
                settings=replace(settings, refine=False), scenarios=scenarios,
                market=market, candidate_key=candidate_key,
                key_by_id=key_by_id, current_price=current_price,
                increment=increment, current_leader=current_leader,
                eligible=eligible, prices=merged, cache=None, regime=regime)
            refined = replace(refined, cache_key=key,
                              runtime_s=time.perf_counter() - t0,
                              notes="refined: every integer in the transition "
                                    "gap was walked")
            refined.check_caps()
            if cache is not None:
                cache.put(key, refined)
            return refined

    if cache is not None:
        cache.put(key, result)
    return result


def immediate_max_bid(*args, **kwargs) -> TacticalResult:
    """The ten-second answer: proxy-backed or served from cache.

    Says so on every line. A proxy comparison has no confidence interval and is
    not a championship-equity estimate; it is a fast ordering of branches that
    the audited path may later contradict.
    """
    settings = kwargs.pop("settings", TacticalSettings())
    return evaluate_tactical(*args, settings=replace(settings, mode="immediate"),
                             **kwargs)


def audited_max_bid(*args, **kwargs) -> TacticalResult:
    """The slow answer: the real CE engine, matched seasons, holdout sample."""
    settings = kwargs.pop("settings", TacticalSettings())
    return evaluate_tactical(*args, settings=replace(settings, mode="audited"),
                             **kwargs)


def format_tactical(r: TacticalResult, width: int = 100,
                    max_rows: int = 24) -> str:
    bar = "=" * width
    mode_note = (f"IMMEDIATE -- {r.PROXY_BANNER}. No interval."
                 if not r.is_audited else
                 "AUDITED -- championship equity over reconciled joint worlds.")
    out = [bar, "TACTICAL MAXIMUM BID", bar,
           f"mode                 {r.mode}   {mode_note}",
           f"candidate            player {r.candidate_id} "
           f"({r.endgame.position})",
           f"our owner            {r.focus_owner_id}",
           f"current price        "
           f"{'-' if r.current_price is None else '$%d' % r.current_price}"
           f"   increment ${r.increment}"
           f"   leader {r.current_leader or '(none)'}",
           f"auction / market     {r.auction_fingerprint} / "
           f"{r.market_fingerprint or 'no market state'}",
           f"cache key            {r.cache_key}"
           f"{'   (SERVED FROM CACHE)' if r.from_cache else ''}",
           f"runtime              {r.runtime_s:.2f}s",
           f"result kind          {r.result_kind}",
           f"threshold basis      "
           f"{'CE-audited' if r.is_audited else 'PROXY ONLY -- not a CE reservation price'}",
           f"nested opportunity   "
           f"{'not built (proxy mode)' if r.nested_ladder is None else ('nested=%s union=%d fp=%s' % (r.nested_ladder.is_nested, r.nested_ladder.union_size, r.nested_ladder.fingerprint()))}",
           f"joint validation     "
           f"{'n/a (proxy mode)' if not r.is_audited else 'every evaluated world passed conservation'}",
           f"selection / holdout  "
           f"{r.settings.completion.selection_sims:,} / "
           f"{r.settings.completion.evaluation_sims:,} seasons", "",
           "PRICES (four different questions; none replaces another)",
           f"  legal maximum               ${r.legal_max}",
           f"  financial-control threshold "
           f"{'n/a' if r.financial_control_threshold is None else '$%d' % r.financial_control_threshold}"
           f"   (rivals CANNOT legally exceed it; not a recommendation)",
           f"  robust tactical maximum     "
           f"{'none' if r.robust_tactical_max is None else '$%d' % r.robust_tactical_max}"
           f"   (favorable vs EVERY recipient and scenario)",
           f"  robust bracket              {r.robust_bracket}",
           f"  base tactical maximum       "
           f"{'none' if r.base_tactical_max is None else '$%d' % r.base_tactical_max}"
           f"   (designated base scenario only)",
           f"  permissive ceiling          "
           f"{'none' if r.permissive_ceiling is None else '$%d' % r.permissive_ceiling}"
           f"   (A CEILING, not the recommended bid)", ""]
    if r.notes:
        out += [f"  note: {r.notes}", ""]
    out.append(f"tested prices        {list(r.tested_prices)}"
               f"   dense={r.ladder_is_dense}")
    if r.untested_gaps:
        out.append(f"untested gaps        "
                   f"{[f'${a}-${b}' for a, b in r.untested_gaps]}")
    if r.nonmonotonic:
        out.append(f"NONMONOTONIC         {list(r.nonmonotonic)}  "
                   f"(cheaper price unfavorable, dearer price favorable)")
    out += ["", f"  {'price':>6}{'scenario':>12}{'recipient':<34}"
                f"{'delta':>11}{'+/-95%':>10}  verdict", "  " + "-" * (width - 2)]
    for v in r.verdicts[:max_rows]:
        ci = v.ci95
        pm = "-" if ci is None else f"{(ci[1] - ci[0]) / 2:.4f}"
        out.append(f"  {v.price:>6}{v.scenario_id:>12}  {v.recipient_label:<32}"
                   f"{v.delta:>11.5f}{pm:>10}  {v.verdict}")
    if len(r.verdicts) > max_rows:
        out.append(f"  ... {len(r.verdicts) - max_rows} more rows")
    out += ["", "  Each row is ONE named recipient. No recipient was averaged",
            "  into another before its branch was evaluated.",
            f"  {r.recipients.price_rule}", bar]
    return "\n".join(out)
