"""An exchangeable ensemble of future allocations, and a two-stage interval.

The real-board pilot quoted a CE effect from **one** future auction. The audit
of that draw is damning: 84.4% of its 180 allocations had exactly tied
willingness -- twelve identical owners in an empty room do -- and a 6%
multiplicative shock, indexed by each owner's position in a tuple, decided
72.2% of the winners. At a fixed seed one team drew the same noise column every
time. That is a persistent advantage earned by list order, and it is why
structurally identical Team02 and Team03 finished 3.6 standard errors apart.

The fix is not more seeds. Fifteen arbitrary seeds of a shock that is keyed to
owner position reproduce the same bias fifteen times. What is needed is
**exchangeability**: across the ensemble every initially identical owner must
occupy every priority slot equally often, so no team can gain from its label.
:func:`balanced_schedule` builds that by rotating the rivals through the
priority order, which for eleven rivals is exhausted by eleven rotations.

Two uncertainties are kept apart and never added together:

**Within-allocation** is the paired season SE inside one future auction. More
seasons shrink it.

**Between-allocation** is the spread of the estimate across future auctions.
More seasons do nothing to it. Allocation draws are *clusters*, and pooling
every season across every draw as if independently sampled would understate the
real uncertainty by exactly the amount that matters.
"""

from __future__ import annotations

import dataclasses
import hashlib
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
from .joint import JointComparison, build_joint_worlds, evaluate_joint_arm

__all__ = [
    "AllocationDraw",
    "balanced_schedule",
    "DrawResult",
    "EnsembleResult",
    "run_ensemble",
    "symmetry_check",
    "SymmetryResult",
    "ensemble_cache_key",
    "FRONTIER_NOT_REACHED",
]

FRONTIER_NOT_REACHED = "FRONTIER_NOT_REACHED"


@dataclass(frozen=True)
class AllocationDraw:
    """One future auction: a seed, and a priority order over owners."""

    index: int
    seed: int
    permutation: Tuple[int, ...]
    rotation: int

    def key(self) -> Tuple:
        return (self.index, self.seed, self.permutation, self.rotation)

    def to_dict(self) -> Dict[str, object]:
        return dataclasses.asdict(self)


def balanced_schedule(n_owners: int, k: int, *, focus_index: int = 0,
                      base_seed: int = 20260906) -> Tuple[AllocationDraw, ...]:
    """``k`` draws in which every rival occupies every priority slot equally.

    The focus team keeps its slot: it is not exchangeable with a rival, because
    it is the team whose equity is being measured. The remaining owners are
    rotated, and ``n_rivals`` rotations exhaust the balanced set -- with the
    preference shock at zero, draw ``n_rivals`` is the last one that can differ
    from its predecessors, and the ensemble mean at that ``k`` is exact rather
    than sampled.

    The schedule is **predetermined**: draw ``i`` is rotation ``i mod
    n_rivals``, so no one can choose a flattering subset after seeing results.
    """
    if k < 1:
        raise ValueError("an ensemble needs at least one allocation draw")
    if not 0 <= focus_index < n_owners:
        raise ValueError("focus_index is out of range")
    rivals = [i for i in range(n_owners) if i != focus_index]
    n_r = len(rivals)
    draws: List[AllocationDraw] = []
    for i in range(k):
        rot = i % n_r
        rotated = rivals[rot:] + rivals[:rot]
        perm = [0] * n_owners
        perm[focus_index] = focus_index
        # Owner ``rivals[j]`` sits in the priority slot held by
        # ``rotated[j]`` -- a relabelling of slots, not of owners.
        for slot, owner in zip(rivals, rotated):
            perm[owner] = slot
        draws.append(AllocationDraw(index=i, seed=base_seed + i * 7919,
                                    permutation=tuple(perm), rotation=rot))
    return tuple(draws)


@dataclass(frozen=True)
class DrawResult:
    """One allocation draw's paired buy/pass estimate."""

    draw: AllocationDraw
    delta: float
    paired_se: float
    ce_buy: float
    ce_pass: float
    alloc_fingerprint: str
    joint_fingerprint: str
    conservation_ok: bool
    league_ce_sum: float
    runtime_s: float

    def to_dict(self) -> Dict[str, object]:
        return {"draw": self.draw.to_dict(), "delta": round(self.delta, 6),
                "paired_se": round(self.paired_se, 6),
                "ce_buy": round(self.ce_buy, 5),
                "ce_pass": round(self.ce_pass, 5),
                "alloc_fingerprint": self.alloc_fingerprint,
                "joint_fingerprint": self.joint_fingerprint,
                "conservation_ok": self.conservation_ok,
                "league_ce_sum": round(self.league_ce_sum, 6),
                "runtime_s": round(self.runtime_s, 2)}


# Two-sided 95% t quantiles, df = 1..30, then the normal limit.
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
        7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
        13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101,
        19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064,
        25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042}


def _t95(df: int) -> float:
    if df < 1:
        return float("nan")
    return _T95.get(df, 1.960)


@dataclass
class EnsembleResult:
    """The ensemble, its two uncertainties, and an interval for the mean."""

    draws: Tuple[DrawResult, ...]
    candidate_id: int
    price: int
    recipient: Optional[str]
    cache_key: str
    holdout_sims: int
    selection_sims: int
    tie_break: str
    shock_scenario: str
    runtime_s: float = 0.0

    @property
    def deltas(self) -> Tuple[float, ...]:
        return tuple(d.delta for d in self.draws)

    @property
    def k(self) -> int:
        return len(self.draws)

    @property
    def mean(self) -> float:
        return statistics.fmean(self.deltas)

    @property
    def median(self) -> float:
        return statistics.median(self.deltas)

    @property
    def between_sd(self) -> float:
        """Spread ACROSS future auctions. Seasons cannot shrink it."""
        return statistics.stdev(self.deltas) if self.k > 1 else 0.0

    @property
    def rms_within_se(self) -> float:
        """Season noise inside a draw. Reported beside, never added to, the above."""
        finite = [d.paired_se for d in self.draws
                  if math.isfinite(d.paired_se)]
        if not finite:
            return float("nan")
        return math.sqrt(sum(s * s for s in finite) / len(finite))

    @property
    def se_of_mean(self) -> float:
        """Cluster-level SE: the SD across draws over sqrt(k).

        Draws are clusters of seasons, not extra seasons. Pooling every season
        across every draw would divide by sqrt(k * n_seasons) and understate
        this by more than an order of magnitude.
        """
        return self.between_sd / math.sqrt(self.k) if self.k > 1 else float("nan")

    @property
    def ci95(self) -> Tuple[float, float]:
        """t-interval for the ensemble MEAN across allocation draws.

        Small-sample and cluster-level: with k draws it has k-1 degrees of
        freedom, and at k below about 6 the interval is wide because it deserves
        to be. It is an interval for the *average future auction*, not for any
        one of them.
        """
        if self.k < 2:
            return (float("nan"), float("nan"))
        half = _t95(self.k - 1) * self.se_of_mean
        return (self.mean - half, self.mean + half)

    @property
    def sign_frequency(self) -> Dict[str, float]:
        pos = sum(1 for d in self.deltas if d > 0)
        neg = sum(1 for d in self.deltas if d < 0)
        return {"positive": pos / self.k, "negative": neg / self.k,
                "zero": (self.k - pos - neg) / self.k}

    @property
    def sign_stable(self) -> bool:
        f = self.sign_frequency
        return f["positive"] == 1.0 or f["negative"] == 1.0

    def quantiles(self) -> Dict[str, float]:
        d = sorted(self.deltas)
        def q(p: float) -> float:
            if len(d) == 1:
                return d[0]
            i = p * (len(d) - 1)
            lo, hi = int(math.floor(i)), int(math.ceil(i))
            return d[lo] + (d[hi] - d[lo]) * (i - lo)
        return {"p05": q(0.05), "p25": q(0.25), "p50": q(0.50),
                "p75": q(0.75), "p95": q(0.95)}

    @property
    def verdict(self) -> str:
        lo, hi = self.ci95
        if math.isnan(lo):
            return "unresolved (k<2: no between-allocation interval exists)"
        if lo > 0:
            return "favorable"
        if hi < 0:
            return "unfavorable"
        return "unresolved"

    @property
    def distinct_allocations(self) -> int:
        return len({d.alloc_fingerprint for d in self.draws})

    def running(self, ks: Sequence[int]) -> List[Dict[str, object]]:
        """Prefix results at each k. The schedule is predetermined, so this is
        a convergence curve and not a search for a flattering subset."""
        out = []
        for k in ks:
            if k > self.k:
                continue
            sub = replace(self, draws=self.draws[:k])
            lo, hi = sub.ci95
            out.append({
                "k": k, "mean": round(sub.mean, 6),
                "between_sd": round(sub.between_sd, 6),
                "se_of_mean": (None if math.isnan(sub.se_of_mean)
                               else round(sub.se_of_mean, 6)),
                "ci95": (None if math.isnan(lo)
                         else [round(lo, 6), round(hi, 6)]),
                "sign_positive": round(sub.sign_frequency["positive"], 3),
                "verdict": sub.verdict,
                "distinct_allocations": sub.distinct_allocations,
                "runtime_s": round(sum(d.runtime_s for d in sub.draws), 1)})
        return out

    def to_dict(self, *, include_draws: bool = True) -> Dict[str, object]:
        lo, hi = self.ci95
        out: Dict[str, object] = {
            "candidate_id": self.candidate_id, "price": self.price,
            "recipient": self.recipient, "cache_key": self.cache_key,
            "k": self.k, "distinct_allocations": self.distinct_allocations,
            "tie_break": self.tie_break, "shock_scenario": self.shock_scenario,
            "holdout_sims": self.holdout_sims,
            "selection_sims": self.selection_sims,
            "mean": round(self.mean, 6), "median": round(self.median, 6),
            "min": round(min(self.deltas), 6),
            "max": round(max(self.deltas), 6),
            "between_allocation_sd": round(self.between_sd, 6),
            "rms_within_allocation_se": round(self.rms_within_se, 6),
            "se_of_ensemble_mean": (None if math.isnan(self.se_of_mean)
                                    else round(self.se_of_mean, 6)),
            "ci95_of_mean": (None if math.isnan(lo)
                             else [round(lo, 6), round(hi, 6)]),
            "quantiles": {k: round(v, 6) for k, v in self.quantiles().items()},
            "sign_frequency": self.sign_frequency,
            "sign_stable": self.sign_stable,
            "verdict": self.verdict,
            "runtime_s": round(self.runtime_s, 1),
            "interval_note": (
                "t-interval for the ENSEMBLE MEAN over allocation draws, "
                "df=k-1. Draws are clusters of seasons; between-allocation SD "
                "and within-allocation SE are reported separately and are "
                "never added."),
        }
        if include_draws:
            out["draws"] = [d.to_dict() for d in self.draws]
        return out


def ensemble_cache_key(*, state: AuctionState, market: Optional[MarketState],
                       cast: ComparisonCast, costs: CostBook,
                       candidate_id: int, price: int,
                       recipient: Optional[str],
                       draws: Sequence[AllocationDraw],
                       board: BoardSettings,
                       completion: CompletionSettings,
                       holdout_sims: int,
                       performance_scenario: str,
                       symmetry_swap: Optional[Tuple[str, str]] = None) -> str:
    """Everything that could change an ensemble answer, including its schedule.

    The ordered seed/permutation schedule is part of the identity: a one-draw
    result must never satisfy a fifteen-draw request, and two ensembles that
    visited the same allocations in a different order are not the same object.
    """
    parts = [
        f"auction={state.fingerprint()}",
        f"pool={state.pool_fingerprint()}",
        f"cast={cast.fingerprint()}",
        f"costs={costs.fingerprint()}",
        f"market={'none' if market is None else market.fingerprint()}",
        f"candidate={candidate_id}", f"price={price}",
        f"recipient={recipient}",
        f"k={len(draws)}",
        "schedule=" + "|".join(str(d.key()) for d in draws),
        f"tie_break={board.tie_break}", f"tie_tol={board.tie_tolerance}",
        f"shock={board.preference_shock}:{board.shock_scenario}",
        f"board={board.cache_key()}",
        f"completion={completion.cache_key()}",
        f"holdout={holdout_sims}",
        f"performance={performance_scenario}",
        f"symmetry_swap={symmetry_swap}",
    ]
    return "ens-" + hashlib.sha256(
        "\n".join(parts).encode("utf-8")).hexdigest()[:24]


def run_ensemble(
    state: AuctionState,
    cast: ComparisonCast,
    costs: CostBook,
    candidate_id: int,
    price: int,
    recipient: str,
    recipient_price: int,
    *,
    draws: Sequence[AllocationDraw],
    board: BoardSettings,
    completion: CompletionSettings,
    market: Optional[MarketState] = None,
    key_by_id: Optional[Dict[int, str]] = None,
    proxy: Optional[ProxyEvaluator] = None,
    holdout_sims: int = 4000,
    holdout_seed: int = 917_324_011,
    max_worlds: int = 2,
    performance_scenario: str = "unstated",
    runtime_budget_s: float = 3600.0,
    progress=None,
) -> EnsembleResult:
    """One paired buy/pass estimate per allocation draw, same draw both arms."""
    t0 = time.perf_counter()
    focus = state.focus_owner_id
    if proxy is None:
        proxy = ProxyEvaluator(state.pool, state.settings,
                               completion.proxy_reps, completion.proxy_seed)
    key = ensemble_cache_key(
        state=state, market=market, cast=cast, costs=costs,
        candidate_id=candidate_id, price=price, recipient=recipient,
        draws=draws, board=board, completion=completion,
        holdout_sims=holdout_sims, performance_scenario=performance_scenario)

    results: List[DrawResult] = []
    for d in draws:
        if time.perf_counter() - t0 > runtime_budget_s:
            break
        ts = time.perf_counter()
        bs = replace(board, seed=d.seed, owner_priority=d.permutation)
        common = dict(board_settings=bs, completion=completion, market=market,
                      key_by_id=key_by_id, proxy=proxy, default_cost=1,
                      max_worlds=max_worlds,
                      branch_acquired=frozenset({candidate_id}))
        ev = dict(focus_team_index=cast.focus_team_index, proxy=proxy,
                  selection_sims=completion.selection_sims,
                  selection_seed=completion.selection_seed,
                  holdout_sims=holdout_sims, holdout_seed=holdout_seed)
        # The SAME allocation draw serves both arms: buy and pass differ only
        # in who owns the candidate, never in which future auction they face.
        buy = evaluate_joint_arm(build_joint_worlds(
            state.apply_purchase(candidate_id, focus, price), cast, costs,
            **common), **ev)
        pas = evaluate_joint_arm(build_joint_worlds(
            state.award_to_rival(candidate_id, recipient, recipient_price),
            cast, costs, **common), **ev)
        cmp_ = JointComparison(buy=buy, pass_arm=pas, price=price,
                               recipient=recipient,
                               recipient_price=recipient_price)
        results.append(DrawResult(
            draw=d, delta=cmp_.delta_ce, paired_se=cmp_.delta_se,
            ce_buy=buy.ce, ce_pass=pas.ce,
            alloc_fingerprint=buy.world.rival_board.fingerprint(),
            joint_fingerprint=buy.world.fingerprint(),
            conservation_ok=(buy.world.conservation.ok
                             and pas.world.conservation.ok),
            league_ce_sum=buy.league_ce_sum,
            runtime_s=time.perf_counter() - ts))
        if progress is not None:
            progress(len(results), len(draws), results[-1])
    return EnsembleResult(
        draws=tuple(results), candidate_id=candidate_id, price=price,
        recipient=recipient, cache_key=key, holdout_sims=holdout_sims,
        selection_sims=completion.selection_sims, tie_break=board.tie_break,
        shock_scenario=board.shock_scenario,
        runtime_s=time.perf_counter() - t0)


@dataclass
class SymmetryResult:
    """Paired label-swap differences between two identical opponents."""

    owner_a: str
    owner_b: str
    per_draw: Tuple[float, ...]
    ce_a: Tuple[float, ...]
    ce_b: Tuple[float, ...]
    single_seed_gap: float
    cache_key: str

    @property
    def k(self) -> int:
        return len(self.per_draw)

    @property
    def mean(self) -> float:
        return statistics.fmean(self.per_draw) if self.k else float("nan")

    @property
    def paired_se(self) -> float:
        if self.k < 2:
            return float("nan")
        return statistics.stdev(self.per_draw) / math.sqrt(self.k)

    @property
    def ci95(self) -> Tuple[float, float]:
        if self.k < 2:
            return (float("nan"), float("nan"))
        h = _t95(self.k - 1) * self.paired_se
        return (self.mean - h, self.mean + h)

    @property
    def max_abs(self) -> float:
        return max((abs(x) for x in self.per_draw), default=float("nan"))

    @property
    def contains_zero(self) -> bool:
        lo, hi = self.ci95
        return not math.isnan(lo) and lo <= 0.0 <= hi

    @property
    def persistent_label_effect(self) -> bool:
        """Does one label beat the other more than chance across the ensemble?"""
        return not self.contains_zero

    def to_dict(self) -> Dict[str, object]:
        lo, hi = self.ci95
        return {"owner_a": self.owner_a, "owner_b": self.owner_b, "k": self.k,
                "single_seed_gap": round(self.single_seed_gap, 6),
                "per_draw": [round(x, 6) for x in self.per_draw],
                "ensemble_mean_difference": round(self.mean, 6),
                "paired_se": (None if math.isnan(self.paired_se)
                              else round(self.paired_se, 6)),
                "ci95": (None if math.isnan(lo)
                         else [round(lo, 6), round(hi, 6)]),
                "max_abs_paired_difference": round(self.max_abs, 6),
                "contains_zero": self.contains_zero,
                "persistent_label_effect": self.persistent_label_effect,
                "cache_key": self.cache_key,
                "note": ("before any sale two rivals are structurally "
                         "identical; the expected difference is exactly zero "
                         "and the preference shock is forced off here")}


def symmetry_check(
    state: AuctionState,
    cast: ComparisonCast,
    costs: CostBook,
    candidate_id: int,
    price: int,
    owner_a: str,
    owner_b: str,
    *,
    draws: Sequence[AllocationDraw],
    board: BoardSettings,
    completion: CompletionSettings,
    market: Optional[MarketState] = None,
    key_by_id: Optional[Dict[int, str]] = None,
    proxy: Optional[ProxyEvaluator] = None,
    holdout_sims: int = 4000,
    holdout_seed: int = 917_324_011,
    max_worlds: int = 2,
    runtime_budget_s: float = 3600.0,
) -> SymmetryResult:
    """Give the same player to two identical rivals, on matched worlds.

    The preference shock is forced to zero: a *pure* symmetry test asks whether
    the mechanism treats identical owners identically, and leaving a behavioural
    shock switched on would answer a different question.
    """
    t0 = time.perf_counter()
    if proxy is None:
        proxy = ProxyEvaluator(state.pool, state.settings,
                               completion.proxy_reps, completion.proxy_seed)
    pure = replace(board, preference_shock=0.0, shock_scenario="none_pure_symmetry")
    key = ensemble_cache_key(
        state=state, market=market, cast=cast, costs=costs,
        candidate_id=candidate_id, price=price, recipient=None, draws=draws,
        board=pure, completion=completion, holdout_sims=holdout_sims,
        performance_scenario="symmetry", symmetry_swap=(owner_a, owner_b))
    diffs, a_ce, b_ce = [], [], []
    for d in draws:
        if time.perf_counter() - t0 > runtime_budget_s:
            break
        bs = replace(pure, seed=d.seed, owner_priority=d.permutation)
        common = dict(board_settings=bs, completion=completion, market=market,
                      key_by_id=key_by_id, proxy=proxy, default_cost=1,
                      max_worlds=max_worlds,
                      branch_acquired=frozenset({candidate_id}))
        ev = dict(focus_team_index=cast.focus_team_index, proxy=proxy,
                  selection_sims=completion.selection_sims,
                  selection_seed=completion.selection_seed,
                  holdout_sims=holdout_sims, holdout_seed=holdout_seed)
        arm_a = evaluate_joint_arm(build_joint_worlds(
            state.award_to_rival(candidate_id, owner_a, price), cast, costs,
            **common), **ev)
        arm_b = evaluate_joint_arm(build_joint_worlds(
            state.award_to_rival(candidate_id, owner_b, price), cast, costs,
            **common), **ev)
        a_ce.append(arm_a.ce)
        b_ce.append(arm_b.ce)
        diffs.append(arm_a.ce - arm_b.ce)
    return SymmetryResult(
        owner_a=owner_a, owner_b=owner_b, per_draw=tuple(diffs),
        ce_a=tuple(a_ce), ce_b=tuple(b_ce),
        single_seed_gap=diffs[0] if diffs else float("nan"), cache_key=key)
