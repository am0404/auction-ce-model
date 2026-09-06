"""Split a tactical CE effect into possession, payment and denial.

The converged diagnostic found positive weekly lineup improvement for all four
audited real candidates, and the K=11 tactical CE was negative for two of them.
That is not a contradiction and it was wrong to leave it unexplained: improving
our lineup is not the same as being worth buying at the market price, above all
when passing may make an opponent overpay.

Five branches, of which two are **analytical only**:

``W``   withdrawn -- nobody has him, nobody pays, nobody spends a slot.
``UF``  we receive him for $0. One slot, no money. **Not a real auction.**
``UP``  we receive him for $p. The real buy branch.
``RF``  the named rival receives him for $0. **Not a real auction.**
``RP``  the named rival receives him for $q. The real pass branch.

``UF``, ``RF`` and ``W`` exist to hold one thing fixed at a time. They are
counterfactual instruments, never outcomes, and they must never be offered to
the recipient layer as things that might happen: a branch where a rival is
handed a player for nothing has no probability, and giving it one would corrupt
every weighted summary downstream. :class:`BranchSpec` marks them ``analytical``
and :func:`assert_not_a_recipient` refuses to let one be treated otherwise.

The decomposition telescopes along a **declared path**:

    CE(UP) - CE(RP)
      = [CE(UP) - CE(UF)]   our payment effect
      + [CE(UF) - CE(W)]    our possession effect
      + [CE(W)  - CE(RF)]   rival possession / denial effect
      + [CE(RF) - CE(RP)]   rival payment effect

It sums exactly because it is a telescoping sum, not because the world is
additive. Roster completion is nonlinear, so a different ordering attributes
different amounts to each term. This is a *declared path*, not a unique causal
decomposition, and the alternate ordering is reported so the reader can see how
much the labels move.
"""

from __future__ import annotations

import dataclasses
import math
import statistics
import time
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence, Tuple

from ..auction.completion import ComparisonCast, CompletionSettings
from ..auction.costs import CostBook
from ..auction.proxy import ProxyEvaluator
from ..auction.state import AuctionState
from ..market.live import MarketState
from .board import BoardSettings
from .ensemble import AllocationDraw, _t95, balanced_schedule
from .joint import build_joint_worlds, evaluate_joint_arm

__all__ = [
    "BRANCHES",
    "BranchSpec",
    "AnalyticalBranchMisuse",
    "assert_not_a_recipient",
    "build_branch_state",
    "DrawDecomposition",
    "Decomposition",
    "decompose",
    "COMPONENT_ORDER",
    "ALTERNATE_ORDER",
]


class AnalyticalBranchMisuse(RuntimeError):
    """An analytical branch was treated as a possible auction outcome."""


@dataclass(frozen=True)
class BranchSpec:
    """One counterfactual world, and whether it could actually happen."""

    key: str
    holder: str
    """``none`` | ``focus`` | ``rival``"""
    pays: str
    """``none`` | ``zero`` | ``price``"""
    analytical: bool
    description: str

    def to_dict(self) -> Dict[str, object]:
        return dataclasses.asdict(self)


BRANCHES: Dict[str, BranchSpec] = {b.key: b for b in (
    BranchSpec("W", "none", "none", True,
               "withdrawn: nobody holds him, nobody pays, no slot consumed"),
    BranchSpec("UF", "focus", "zero", True,
               "we hold him for $0; one slot consumed, no money spent"),
    BranchSpec("UP", "focus", "price", False,
               "we hold him at our tested price -- the real buy branch"),
    BranchSpec("RF", "rival", "zero", True,
               "the named rival holds him for $0 -- analytical only"),
    BranchSpec("RP", "rival", "price", False,
               "the named rival holds him at his price -- the real pass branch"),
)}

COMPONENT_ORDER: Tuple[Tuple[str, str, str], ...] = (
    ("our_payment", "UP", "UF"),
    ("our_possession", "UF", "W"),
    ("rival_denial", "W", "RF"),
    ("rival_payment", "RF", "RP"),
)

#: A different telescoping path over the same five branches. Reported to show
#: how much the *labels* move when the order does; the total is identical.
ALTERNATE_ORDER: Tuple[Tuple[str, str, str], ...] = (
    ("rival_payment_first", "UP", "UP"),
)


def assert_not_a_recipient(branch_key: str) -> None:
    """Refuse to let ``W``/``UF``/``RF`` be used as a pass recipient."""
    spec = BRANCHES.get(branch_key)
    if spec is not None and spec.analytical:
        raise AnalyticalBranchMisuse(
            f"branch {branch_key!r} is an ANALYTICAL decomposition instrument "
            f"({spec.description}). It is not a possible auction outcome and "
            f"may not be given a recipient probability or weighted into a "
            f"summary.")


def build_branch_state(state: AuctionState, candidate_id: int, branch: str, *,
                       focus_price: int, rival: Optional[str],
                       rival_price: int) -> Tuple[AuctionState, Dict[str, object]]:
    """The room under one branch, plus the bookkeeping the reconciler needs."""
    spec = BRANCHES[branch]
    focus = state.focus_owner_id
    if spec.holder == "none":
        return state.withdraw(candidate_id), {
            "declared_withdrawn": frozenset({candidate_id}),
            "branch_acquired": frozenset()}
    owner = focus if spec.holder == "focus" else rival
    if owner is None:
        raise ValueError(f"branch {branch} needs a named rival")
    price = 0 if spec.pays == "zero" else (
        focus_price if spec.holder == "focus" else rival_price)
    # A $0 acquisition is not a legal bid, which is exactly why the analytical
    # branches are labelled analytical. It consumes a roster slot and spends
    # nothing, which is the isolation the decomposition needs.
    new = state.apply_purchase(candidate_id, owner, price,
                               validate=(price > 0))
    return new, {"branch_acquired": frozenset({candidate_id}),
                 "declared_withdrawn": frozenset()}


@dataclass(frozen=True)
class DrawDecomposition:
    """One allocation draw: five CEs and the components they telescope into."""

    draw: AllocationDraw
    ce: Dict[str, float]
    components: Dict[str, float]
    total: float
    residual: float
    conservation_ok: bool
    league_ce_sums: Dict[str, float]
    fingerprints: Dict[str, str]

    def to_dict(self) -> Dict[str, object]:
        return {"draw": self.draw.to_dict(),
                "ce": {k: round(v, 6) for k, v in self.ce.items()},
                "components": {k: round(v, 6)
                               for k, v in self.components.items()},
                "total": round(self.total, 6),
                "residual": round(self.residual, 12),
                "conservation_ok": self.conservation_ok,
                "league_ce_sums": {k: round(v, 6)
                                   for k, v in self.league_ce_sums.items()},
                "fingerprints": self.fingerprints}


def _stats(values: Sequence[float]) -> Dict[str, object]:
    n = len(values)
    if n == 0:
        return {}
    mean = statistics.fmean(values)
    sd = statistics.stdev(values) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 1 else float("nan")
    half = _t95(n - 1) * se if n > 1 else float("nan")
    pos = sum(1 for v in values if v > 0)
    return {
        "k": n, "mean": round(mean, 6), "median": round(statistics.median(values), 6),
        "min": round(min(values), 6), "max": round(max(values), 6),
        "between_allocation_sd": round(sd, 6),
        "se_of_mean": None if math.isnan(se) else round(se, 6),
        "ci95": None if math.isnan(half) else [round(mean - half, 6),
                                               round(mean + half, 6)],
        "sign_positive": round(pos / n, 3),
        "sign_stable": pos == n or pos == 0,
    }


@dataclass
class Decomposition:
    """The whole lattice for one candidate against one named rival."""

    candidate_id: int
    position: str
    focus_price: int
    rival: str
    rival_price: int
    draws: Tuple[DrawDecomposition, ...]
    lineup_improvement: float
    role: str
    within_se: Dict[str, float]
    runtime_s: float

    @property
    def k(self) -> int:
        return len(self.draws)

    def component_values(self, name: str) -> Tuple[float, ...]:
        return tuple(d.components[name] for d in self.draws)

    def totals(self) -> Tuple[float, ...]:
        return tuple(d.total for d in self.draws)

    @property
    def max_residual(self) -> float:
        return max((abs(d.residual) for d in self.draws), default=0.0)

    @property
    def ensemble_residual(self) -> float:
        """Telescoping at the ensemble mean, not only per draw."""
        s = sum(statistics.fmean(self.component_values(n))
                for n, _, _ in COMPONENT_ORDER)
        return s - statistics.fmean(self.totals())

    def classify(self) -> Tuple[str, str]:
        """Which of the stated explanations the numbers actually support."""
        c = {n: statistics.fmean(self.component_values(n))
             for n, _, _ in COMPONENT_ORDER}
        tot = statistics.fmean(self.totals())
        stats = {n: _stats(self.component_values(n))
                 for n, _, _ in COMPONENT_ORDER}
        total_stats = _stats(self.totals())
        if total_stats.get("ci95") and total_stats["ci95"][0] <= 0 <= \
                total_stats["ci95"][1]:
            return ("unresolved",
                    "the total's ensemble interval contains zero, so no cause "
                    "can be assigned to a sign that is not established")
        if tot < 0 and c["our_possession"] > 0 and \
                c["our_payment"] < 0 and abs(c["our_payment"]) >= c["our_possession"]:
            return ("helps us but costs too much",
                    f"own possession {c['our_possession']:+.5f} is positive and "
                    f"own payment {c['our_payment']:+.5f} more than offsets it")
        if tot < 0 and c["rival_payment"] > 0 and \
                c["rival_payment"] >= abs(tot) * 0.5:
            return ("passing induces a beneficial rival overpay",
                    f"rival payment {c['rival_payment']:+.5f} is a material "
                    f"share of the {tot:+.5f} total")
        if abs(c["rival_denial"]) >= max(abs(tot) * 0.5, 1e-9):
            return ("denial value matters",
                    f"rival possession/denial {c['rival_denial']:+.5f} is "
                    f"material against a {tot:+.5f} total")
        if c["our_possession"] <= 0 and self.lineup_improvement > 0.5:
            return ("proxy is misleading",
                    f"weekly lineup improvement {self.lineup_improvement:+.2f} "
                    f"but own possession effect {c['our_possession']:+.5f}; the "
                    f"proxy likes a roster championship equity does not")
        return ("mixed",
                "no single component dominates the total at this sample size")

    def to_dict(self, *, include_draws: bool = True,
                include_identity: bool = False) -> Dict[str, object]:
        comps = {n: _stats(self.component_values(n))
                 for n, _, _ in COMPONENT_ORDER}
        label, why = self.classify()
        out: Dict[str, object] = {
            "position": self.position,
            "focus_price": self.focus_price, "rival": self.rival,
            "rival_price": self.rival_price, "k": self.k,
            "lineup_improvement": round(self.lineup_improvement, 4),
            "role": self.role,
            "ce_by_branch": {b: _stats([d.ce[b] for d in self.draws])
                             for b in BRANCHES},
            "components": comps,
            "total": _stats(self.totals()),
            "max_per_draw_residual": self.max_residual,
            "ensemble_residual": round(self.ensemble_residual, 12),
            "rms_within_allocation_se": {
                k: round(v, 6) for k, v in self.within_se.items()},
            "classification": label, "classification_reason": why,
            "runtime_s": round(self.runtime_s, 1),
            "path_note": (
                "declared telescoping path UP->UF->W->RF->RP. It sums exactly "
                "because it telescopes, not because completion is additive; a "
                "different ordering attributes different amounts to each term."),
            "analytical_note": (
                "W, UF and RF are analytical instruments, not possible auction "
                "outcomes. They carry no recipient probability."),
        }
        # The candidate id is player-level identity and belongs only in the
        # local, gitignored report. Position, price and components are
        # aggregates and are safe to commit.
        if include_identity:
            out["candidate_id"] = self.candidate_id
        if include_draws:
            out["draws"] = [d.to_dict() for d in self.draws]
        return out


def decompose(
    state: AuctionState, cast: ComparisonCast, costs: CostBook,
    candidate_id: int, *, focus_price: int, rival: str, rival_price: int,
    position: str = "?", lineup_improvement: float = 0.0, role: str = "?",
    draws: Optional[Sequence[AllocationDraw]] = None,
    board: Optional[BoardSettings] = None,
    completion: Optional[CompletionSettings] = None,
    market: Optional[MarketState] = None,
    key_by_id: Optional[Dict[int, str]] = None,
    proxy: Optional[ProxyEvaluator] = None,
    holdout_sims: int = 4000, holdout_seed: int = 917_324_011,
    max_worlds: int = 2, runtime_budget_s: float = 3600.0,
    progress=None,
) -> Decomposition:
    """Five branches per allocation draw, telescoped into four components.

    Every branch sees the *same* allocation rotation and the same season seed,
    so a difference between two branches is the thing that differs between them
    and not a different future auction.
    """
    t0 = time.perf_counter()
    if board is None:
        board = BoardSettings(pool_depth=400, max_allocations=200)
    if completion is None:
        completion = CompletionSettings(
            beam_width=32, candidate_pool=40, proxy_candidates=32, finalists=3,
            max_candidates=160, proxy_reps=16, selection_sims=800,
            evaluation_sims=holdout_sims, rival_selection="proxy")
    if draws is None:
        draws = balanced_schedule(len(state.owners), 11)
    if proxy is None:
        proxy = ProxyEvaluator(state.pool, state.settings,
                               completion.proxy_reps, completion.proxy_seed)
    if rival == state.focus_owner_id:
        raise ValueError("the named rival must not be the focus owner")

    per_draw: List[DrawDecomposition] = []
    within: Dict[str, List[float]] = {b: [] for b in BRANCHES}
    for d in draws:
        if time.perf_counter() - t0 > runtime_budget_s:
            break
        bs = replace(board, seed=d.seed, owner_priority=d.permutation)
        ce: Dict[str, float] = {}
        fps: Dict[str, str] = {}
        sums: Dict[str, float] = {}
        ok = True
        for key in BRANCHES:
            bstate, book = build_branch_state(
                state, candidate_id, key, focus_price=focus_price,
                rival=rival, rival_price=rival_price)
            worlds = build_joint_worlds(
                bstate, cast, costs, board_settings=bs, completion=completion,
                market=market, key_by_id=key_by_id, proxy=proxy,
                default_cost=1, max_worlds=max_worlds, **book)
            arm = evaluate_joint_arm(
                worlds, focus_team_index=cast.focus_team_index, proxy=proxy,
                selection_sims=completion.selection_sims,
                selection_seed=completion.selection_seed,
                holdout_sims=holdout_sims, holdout_seed=holdout_seed)
            ce[key] = arm.ce
            fps[key] = arm.world.fingerprint()
            sums[key] = arm.league_ce_sum
            ok = ok and arm.world.conservation.ok
            within[key].append(
                math.sqrt(max(arm.ce * (1.0 - arm.ce), 1e-12) / holdout_sims))
        comps = {name: ce[a] - ce[b] for name, a, b in COMPONENT_ORDER}
        total = ce["UP"] - ce["RP"]
        per_draw.append(DrawDecomposition(
            draw=d, ce=ce, components=comps, total=total,
            residual=sum(comps.values()) - total, conservation_ok=ok,
            league_ce_sums=sums, fingerprints=fps))
        if progress is not None:
            progress(len(per_draw), len(draws), per_draw[-1])

    rms = {k: (math.sqrt(sum(x * x for x in v) / len(v)) if v else float("nan"))
           for k, v in within.items()}
    return Decomposition(
        candidate_id=candidate_id, position=position, focus_price=focus_price,
        rival=rival, rival_price=rival_price, draws=tuple(per_draw),
        lineup_improvement=lineup_improvement, role=role, within_se=rms,
        runtime_s=time.perf_counter() - t0)
