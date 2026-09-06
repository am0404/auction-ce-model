"""Reservation frontiers: where total buy-versus-pass CE actually crosses zero.

A frontier is where

    delta = CE(we buy at p) - CE(named rival buys at q)

changes sign. It is **not** where our payment cost equals our possession value:
the total also carries the rival's possession/denial effect, the rival's payment
effect, and every downstream reallocation of the shared board. The five-branch
decomposition exists precisely to show that those terms are not zero, so a
"frontier" defined on the payment/possession pair alone would be a different
and wrong number.

``p`` and ``q`` are printed together everywhere, because a reservation price is
conditional on the pass-price rule (see :mod:`.midauction`) and a result under
one rule is not a universal maximum bid.

Two cost controls, both honest about what they skip:

**A proxy pre-pass** finds the prices where anything actually changes -- the
feasible set, the selected completion, the joint allocation. Championship
equity is then spent only on those transitions plus the market band and the
bracketing prices, rather than on a dense grid of prices that all produce the
identical world.

**Joint-world reuse.** If two prices produce byte-identical focus rosters and
rival allocations, their CE is identical by construction and the second is
served from cache. Reuse is counted and reported; it is a saving, not an
approximation.
"""

from __future__ import annotations

import dataclasses
import math
import statistics
import time
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

from ..auction.completion import ComparisonCast, CompletionSettings
from ..auction.costs import CostBook
from ..auction.proxy import ProxyEvaluator
from ..auction.state import AuctionState
from ..market.live import MarketState
from .board import BoardSettings
from .ensemble import AllocationDraw, _t95, balanced_schedule
from .endgame import candidate_legal_max
from .joint import JointComparison, build_joint_worlds, evaluate_joint_arm
from .midauction import PassPrice, PassPriceMode
from .nested import build_nested_ladder

__all__ = [
    "COARSE_LADDER",
    "FRONTIER_NOT_REACHED",
    "UNDERPOWERED",
    "EXACT_FRONTIER",
    "BRACKET",
    "PricePoint",
    "FrontierResult",
    "coarse_ladder",
    "proxy_prepass",
    "run_frontier",
]

COARSE_LADDER: Tuple[int, ...] = (
    1, 5, 10, 20, 30, 40, 50, 65, 80, 100, 130, 160)

FRONTIER_NOT_REACHED = "FRONTIER_NOT_REACHED"
UNDERPOWERED = "UNDERPOWERED"
EXACT_FRONTIER = "EXACT_FRONTIER"
BRACKET = "RESERVATION_BRACKET"


def coarse_ladder(legal_max: int, *, extra: Sequence[int] = (),
                  base: Sequence[int] = COARSE_LADDER) -> Tuple[int, ...]:
    """Deduplicated, sorted, legal prices, always reaching the legal maximum.

    The opening experiment's narrow band-shaped ladders never bound, so the
    default reaches far past the market band and always includes ``legal_max``:
    a frontier that exists only above the tested range is indistinguishable
    from no frontier at all.
    """
    if legal_max < 1:
        return ()
    vals = {int(v) for v in list(base) + list(extra) if 1 <= int(v) <= legal_max}
    vals.add(1)
    vals.add(int(legal_max))
    return tuple(sorted(vals))


@dataclass(frozen=True)
class PricePoint:
    """One evaluated rung: our price, the rival's, and what came of it."""

    p: int
    q: Optional[int]
    audited: bool
    feasible_count: int
    focus_fingerprint: str
    alloc_fingerprint: str
    roster_changed: bool
    reused_ce: bool = False
    ce_buy: Optional[float] = None
    ce_pass: Optional[float] = None
    delta: Optional[float] = None
    between_sd: Optional[float] = None
    within_se: Optional[float] = None
    ci95: Optional[Tuple[float, float]] = None
    sign_positive: Optional[float] = None
    verdict: str = "not audited"
    runtime_s: float = 0.0
    note: str = ""

    def to_dict(self) -> Dict[str, object]:
        out = dataclasses.asdict(self)
        if self.ci95 is not None:
            out["ci95"] = [round(self.ci95[0], 6), round(self.ci95[1], 6)]
        for k in ("ce_buy", "ce_pass", "delta", "between_sd", "within_se"):
            if out.get(k) is not None:
                out[k] = round(out[k], 6)
        out["runtime_s"] = round(self.runtime_s, 2)
        return out


@dataclass
class FrontierResult:
    """The ladder, its classification, and the honest limits of both."""

    candidate_position: str
    legal_max: int
    recipient: str
    pass_price: PassPrice
    points: Tuple[PricePoint, ...]
    k: int
    holdout_sims: int
    selection_sims: int
    cache_hits: int
    state_fingerprint: str
    refined: bool = False
    runtime_s: float = 0.0

    @property
    def audited(self) -> Tuple[PricePoint, ...]:
        return tuple(x for x in self.points if x.audited)

    @property
    def highest_favorable(self) -> Optional[int]:
        f = [x.p for x in self.audited if x.verdict == "favorable"]
        return max(f) if f else None

    @property
    def next_unfavorable(self) -> Optional[int]:
        hf = self.highest_favorable
        bad = [x.p for x in self.audited
               if x.verdict in ("unfavorable", "unresolved")
               and (hf is None or x.p > hf)]
        return min(bad) if bad else None

    @property
    def bracket(self) -> Optional[Tuple[int, int]]:
        hf, nu = self.highest_favorable, self.next_unfavorable
        if hf is None or nu is None or nu <= hf:
            return None
        return (hf, nu)

    @property
    def classification(self) -> str:
        aud = self.audited
        if not aud:
            return UNDERPOWERED
        if all(x.verdict == "unresolved" for x in aud):
            return UNDERPOWERED
        hf, nu = self.highest_favorable, self.next_unfavorable
        if hf is not None and nu is None:
            # Favorable everywhere tested, including the legal maximum.
            if hf >= self.legal_max:
                return FRONTIER_NOT_REACHED
            return FRONTIER_NOT_REACHED
        if hf is None:
            # Unfavorable from the first tested price upward.
            return BRACKET
        if nu is not None and nu - hf == 1:
            return EXACT_FRONTIER
        return BRACKET

    @property
    def untested_gap(self) -> Optional[Tuple[int, int]]:
        b = self.bracket
        if b is None or b[1] - b[0] <= 1:
            return None
        return (b[0] + 1, b[1] - 1)

    @property
    def nonmonotonic(self) -> Tuple[Tuple[int, int], ...]:
        """(cheaper unfavorable, dearer favorable) pairs. Reported, not smoothed."""
        aud = sorted(self.audited, key=lambda x: x.p)
        out = []
        for i, x in enumerate(aud):
            if x.verdict != "favorable":
                continue
            for y in aud[:i]:
                if y.verdict == "unfavorable":
                    out.append((y.p, x.p))
        return tuple(out)

    def to_dict(self, *, include_points: bool = True) -> Dict[str, object]:
        out: Dict[str, object] = {
            "simulated_state": True,
            "candidate_position": self.candidate_position,
            "legal_max": self.legal_max,
            "recipient": self.recipient,
            "pass_price": self.pass_price.to_dict(),
            "k": self.k, "holdout_sims": self.holdout_sims,
            "selection_sims": self.selection_sims,
            "cache_hits": self.cache_hits,
            "state_fingerprint": self.state_fingerprint,
            "refined": self.refined,
            "n_tested": len(self.points), "n_audited": len(self.audited),
            "highest_favorable": self.highest_favorable,
            "next_unfavorable": self.next_unfavorable,
            "bracket": list(self.bracket) if self.bracket else None,
            "untested_gap": list(self.untested_gap) if self.untested_gap else None,
            "classification": self.classification,
            "nonmonotonic": [list(t) for t in self.nonmonotonic],
            "runtime_s": round(self.runtime_s, 1),
            "note": ("a reservation price is conditional on the pass-price "
                     "rule above; a sparse bracket is NOT an exact max bid"),
        }
        if include_points:
            out["points"] = [x.to_dict() for x in self.points]
        return out


def proxy_prepass(state: AuctionState, cast: ComparisonCast, costs: CostBook,
                  candidate_id: int, prices: Sequence[int], *,
                  board: BoardSettings, completion: CompletionSettings,
                  market: Optional[MarketState], key_by_id, proxy,
                  ) -> Tuple[Dict[int, Dict[str, object]], object]:
    """Cheap sweep: where does anything actually change?

    Builds one nested ladder over the whole price range -- so the cross-price
    invariant holds for the audit that follows -- and records, per price, the
    feasible-set size and the best completion's identity. Prices at which
    neither moves cannot produce a different joint world and do not earn CE
    seasons.
    """
    from .nested import roster_fingerprint

    focus = state.focus_owner_id
    legal = [p for p in prices
             if state.purchase_shortfall(candidate_id, focus, p) is None]
    ladder = build_nested_ladder(state, cast, costs, candidate_id, legal,
                                 board_settings=board, completion=completion,
                                 market=market, key_by_id=key_by_id,
                                 proxy=proxy)
    out: Dict[int, Dict[str, object]] = {}
    prev_fp = None
    for p in sorted(ladder.prices):
        feas = ladder.by_price[p].feasible
        best = max(feas, key=lambda c: proxy.strength(c.roster)) if feas else None
        fp = roster_fingerprint(best.roster) if best is not None else "none"
        out[p] = {"feasible": len(feas), "focus_fingerprint": fp,
                  "changed": prev_fp is not None and fp != prev_fp,
                  "completions": feas}
        prev_fp = fp
    return out, ladder


def run_frontier(
    state: AuctionState, cast: ComparisonCast, costs: CostBook,
    candidate_id: int, *, recipient: str, pass_price: PassPrice,
    position: str, state_fingerprint: str,
    prices: Optional[Sequence[int]] = None,
    board: Optional[BoardSettings] = None,
    completion: Optional[CompletionSettings] = None,
    market: Optional[MarketState] = None, key_by_id=None,
    proxy: Optional[ProxyEvaluator] = None,
    draws: Optional[Sequence[AllocationDraw]] = None,
    holdout_sims: int = 4000, holdout_seed: int = 917_324_011,
    max_worlds: int = 2, audit_budget: int = 8,
    runtime_budget_s: float = 2400.0, refine: bool = False,
    progress=None,
) -> FrontierResult:
    """Coarse ladder, proxy pre-pass, then K=11 CE at the prices that matter."""
    t0 = time.perf_counter()
    focus = state.focus_owner_id
    if recipient == focus:
        raise ValueError("the named recipient must be an opponent")
    if board is None:
        board = BoardSettings(pool_depth=400, max_allocations=200)
    if completion is None:
        completion = CompletionSettings(
            beam_width=32, candidate_pool=40, proxy_candidates=32, finalists=3,
            max_candidates=160, proxy_reps=16, selection_sims=800,
            evaluation_sims=holdout_sims, rival_selection="proxy")
    if proxy is None:
        proxy = ProxyEvaluator(state.pool, state.settings,
                               completion.proxy_reps, completion.proxy_seed)
    if draws is None:
        draws = balanced_schedule(len(state.owners), 11)

    our_max = candidate_legal_max(state, focus, candidate_id)
    rival_max = candidate_legal_max(state, recipient, candidate_id)
    ladder_prices = (tuple(sorted({int(x) for x in prices}))
                     if prices is not None else coarse_ladder(our_max))
    ladder_prices = tuple(p for p in ladder_prices if p <= our_max)

    pre, nested = proxy_prepass(state, cast, costs, candidate_id,
                                ladder_prices, board=board,
                                completion=completion, market=market,
                                key_by_id=key_by_id, proxy=proxy)

    # Which prices earn CE seasons: every transition, the extremes, and a
    # spread of the rest up to the audit budget.
    transitions = [p for p, v in pre.items() if v["changed"]]
    must = set(transitions)
    if pre:
        must.add(min(pre))
        must.add(max(pre))
    ordered = sorted(pre)
    for p in ordered:
        if len(must) >= audit_budget:
            break
        must.add(p)
    audit_at = sorted(must)[:max(2, audit_budget)]

    points: List[PricePoint] = []
    world_cache: Dict[Tuple[str, str], object] = {}
    hits = 0
    prev_fp = None

    for p in ordered:
        v = pre[p]
        q = pass_price.rival_price(p, rival_legal_max=rival_max)
        do_audit = p in audit_at and q is not None
        if time.perf_counter() - t0 > runtime_budget_s:
            do_audit = False
        pt_note = "" if q is not None else (
            f"the named rival cannot legally pay q under this rule "
            f"(his legal max is ${rival_max})")
        if not do_audit:
            points.append(PricePoint(
                p=p, q=q, audited=False, feasible_count=v["feasible"],
                focus_fingerprint=v["focus_fingerprint"],
                alloc_fingerprint="", roster_changed=bool(v["changed"]),
                note=pt_note or "proxy pre-pass only; no CE spent"))
            prev_fp = v["focus_fingerprint"]
            continue

        ts = time.perf_counter()
        common = dict(board_settings=board, completion=completion,
                      market=market, key_by_id=key_by_id, proxy=proxy,
                      default_cost=1, max_worlds=max_worlds,
                      branch_acquired=frozenset({candidate_id}))
        ev = dict(focus_team_index=cast.focus_team_index, proxy=proxy,
                  selection_sims=completion.selection_sims,
                  selection_seed=completion.selection_seed,
                  holdout_sims=holdout_sims, holdout_seed=holdout_seed)
        deltas, ses, allocs = [], [], []
        reused = False
        for d in draws:
            bs = replace(board, seed=d.seed, owner_priority=d.permutation)
            c2 = dict(common, board_settings=bs)
            buy = evaluate_joint_arm(build_joint_worlds(
                state.apply_purchase(candidate_id, focus, p), cast, costs,
                completions=v["completions"], **c2), **ev)
            key = ("pass", buy.world.rival_board.fingerprint(), str(q))
            cached = world_cache.get(key)
            if cached is not None:
                pas = cached
                reused = True
                hits += 1
            else:
                pas = evaluate_joint_arm(build_joint_worlds(
                    state.award_to_rival(candidate_id, recipient, q), cast,
                    costs, **c2), **ev)
                world_cache[key] = pas
            cmp_ = JointComparison(buy=buy, pass_arm=pas, price=p,
                                   recipient=recipient, recipient_price=q)
            deltas.append(cmp_.delta_ce)
            ses.append(cmp_.delta_se)
            allocs.append(buy.world.fingerprint())
        n = len(deltas)
        mean = statistics.fmean(deltas)
        sd = statistics.stdev(deltas) if n > 1 else 0.0
        se = sd / math.sqrt(n) if n > 1 else float("nan")
        half = _t95(n - 1) * se if n > 1 else float("nan")
        lo, hi = (mean - half, mean + half) if not math.isnan(half) else (
            float("nan"), float("nan"))
        rms = math.sqrt(sum(s * s for s in ses if math.isfinite(s))
                        / max(1, len([s for s in ses if math.isfinite(s)])))
        verdict = ("unresolved" if math.isnan(lo)
                   else "favorable" if lo > 0
                   else "unfavorable" if hi < 0 else "unresolved")
        pos_frac = sum(1 for d_ in deltas if d_ > 0) / n
        pt = PricePoint(
            p=p, q=q, audited=True, feasible_count=v["feasible"],
            focus_fingerprint=v["focus_fingerprint"],
            alloc_fingerprint=allocs[0] if allocs else "",
            roster_changed=bool(v["changed"]), reused_ce=reused,
            ce_buy=None, ce_pass=None, delta=mean, between_sd=sd,
            within_se=rms, ci95=(lo, hi), sign_positive=pos_frac,
            verdict=verdict, runtime_s=time.perf_counter() - ts, note=pt_note)
        points.append(pt)
        prev_fp = v["focus_fingerprint"]
        if progress is not None:
            progress(pt)

    return FrontierResult(
        candidate_position=position, legal_max=our_max, recipient=recipient,
        pass_price=pass_price, points=tuple(points), k=len(draws),
        holdout_sims=holdout_sims, selection_sims=completion.selection_sims,
        cache_hits=hits, state_fingerprint=state_fingerprint, refined=refine,
        runtime_s=time.perf_counter() - t0)
