"""Frontier and decomposition evaluated from ONE shared context, per draw.

Both consumers here take an :class:`~.evalcontext.EvalContext` and take their
completion offers from it. Neither may run a beam of its own: the ladder's buy
arm, the ladder's pass arm and all five decomposition branches now draw from
the same accumulated opportunity set, filtered only by the budget that branch
actually has.

Agreement is checked at the **draw** level, not after averaging eleven
rotations. Averaging can hide two paths that disagree per draw and happen to
land near each other; the per-draw check cannot.
"""

from __future__ import annotations

import dataclasses
import math
import statistics
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .decompose import BRANCHES, COMPONENT_ORDER, build_branch_state
from .evalcontext import (BRANCH_HOLDS_CANDIDATE, ContextMismatch, EvalContext,
                          IndependentSearchRefused)
from .joint import JointComparison, build_joint_worlds, evaluate_joint_arm

__all__ = [
    "AGREEMENT_TOLERANCE",
    "BranchEvaluation",
    "DrawEvaluation",
    "evaluate_branch",
    "evaluate_draw",
    "AgreementReport",
    "check_agreement",
]

#: Same context, same completions, same seeds -> the two paths run byte-identical
#: simulations, so the tolerance is floating-point noise, not modelling slack.
AGREEMENT_TOLERANCE = 1e-12


@dataclass(frozen=True)
class BranchEvaluation:
    """One branch of one draw, evaluated from the shared context."""

    branch: str
    ce: float
    focus_roster: Tuple[int, ...]
    focus_fingerprint: str
    joint_fingerprint: str
    alloc_fingerprint: str
    rival_rosters: Tuple[Tuple[str, Tuple[int, ...]], ...]
    dollars_spent: int
    pool_remaining: int
    league_ce_sum: float
    conservation_ok: bool
    n_offered: int
    selection_ce: Optional[float] = None
    """Equity of the CHOSEN world on the *selection* sample.

    Exposed because the 12->24 candidate-set stability test asks whether
    enlarging the offer changed what CE *picked*, and that question is about
    the selection sample: the holdout only ever sees the winner. Upward-biased
    for the winner by construction, which is exactly why :attr:`ce` and not
    this is the number quoted anywhere else.
    """

    focus_proxy: Optional[float] = None
    """Proxy strength of the chosen roster, so a selection change can be read
    against the certified proxy bound it was emitted under."""

    n_worlds_compared: int = 0
    """How many completions CE actually chose between. Bounded by
    ``EvalContext.max_worlds``, NOT by the size of the offer set: an offer set
    larger than this cannot influence the answer, which is the first thing to
    check when a set expansion appears to change nothing."""

    holdout_indicator: object = None
    """Per-season championship indicator on the holdout sample.

    Kept so the buy/pass delta can be differenced season by season WITHIN one
    allocation. That paired difference is the only honest within-allocation
    standard error: the two branches share the holdout seed, so forming an SE
    from two independent means would ignore the pairing and overstate it.
    Never merged with the between-allocation SD -- they answer different
    questions, and pooling them would present season noise as if it were
    uncertainty about which auction future occurs.
    """

    def to_dict(self) -> Dict[str, object]:
        return {"branch": self.branch, "ce": round(self.ce, 8),
                "focus_fingerprint": self.focus_fingerprint,
                "joint_fingerprint": self.joint_fingerprint,
                "alloc_fingerprint": self.alloc_fingerprint,
                "dollars_spent": self.dollars_spent,
                "pool_remaining": self.pool_remaining,
                "league_ce_sum": round(self.league_ce_sum, 8),
                "conservation_ok": self.conservation_ok,
                "completions_offered": self.n_offered,
                "worlds_compared": self.n_worlds_compared,
                "selection_ce": None if self.selection_ce is None
                else round(self.selection_ce, 8),
                "focus_proxy": None if self.focus_proxy is None
                else round(self.focus_proxy, 6)}


def _roster_fp(roster: Sequence[int]) -> str:
    import hashlib
    return hashlib.sha256(",".join(str(x) for x in sorted(roster))
                          .encode()).hexdigest()[:16]


def evaluate_branch(ctx: EvalContext, branch: str, *, proxy,
                    allow_independent_search: bool = False
                    ) -> BranchEvaluation:
    """Evaluate one of W/UF/UP/RF/RP strictly from ``ctx``'s opportunity set."""
    if branch not in BRANCHES:
        raise ContextMismatch(f"unknown branch {branch!r}")
    if allow_independent_search:
        raise IndependentSearchRefused(
            "a shared EvalContext was supplied, so this branch may not run its "
            "own completion search. That divergence is exactly what made the "
            "ladder and the decomposition disagree at p=$80.")
    offers = ctx.completions_for(branch)
    bstate, book = build_branch_state(
        ctx.state, ctx.candidate_id, branch, focus_price=ctx.p,
        rival=ctx.recipient, rival_price=ctx.q)
    bs = dataclasses.replace(ctx.board, seed=ctx.draw.seed,
                             owner_priority=ctx.draw.permutation)
    worlds = build_joint_worlds(
        bstate, ctx.cast, ctx.costs, board_settings=bs,
        completion=ctx.completion, market=ctx.market,
        key_by_id=ctx.key_by_id, proxy=proxy, default_cost=ctx.default_cost,
        max_worlds=ctx.max_worlds, completions=offers, **book)
    arm = evaluate_joint_arm(
        worlds, focus_team_index=ctx.cast.focus_team_index, proxy=proxy,
        selection_sims=ctx.selection_sims, selection_seed=ctx.selection_seed,
        holdout_sims=ctx.holdout_sims, holdout_seed=ctx.holdout_seed)
    w = arm.world
    focus = w.state.focus_owner_id
    rivals = tuple(sorted((o.owner_id, tuple(sorted(o.player_ids)))
                          for o in w.state.owners if o.owner_id != focus))
    return BranchEvaluation(
        branch=branch, ce=arm.ce, focus_roster=tuple(sorted(w.focus_roster)),
        focus_fingerprint=_roster_fp(w.focus_roster),
        joint_fingerprint=w.fingerprint(),
        alloc_fingerprint=w.rival_board.fingerprint(), rival_rosters=rivals,
        dollars_spent=sum(o.spent for o in w.state.owners),
        pool_remaining=len(w.state.available_ids),
        league_ce_sum=arm.league_ce_sum,
        conservation_ok=w.conservation.ok, n_offered=len(offers),
        selection_ce=float(arm.selection_ce),
        focus_proxy=float(arm.focus_proxy),
        n_worlds_compared=int(arm.n_worlds_compared),
        holdout_indicator=arm.holdout_indicator)


@dataclass(frozen=True)
class DrawEvaluation:
    """All five branches of one allocation draw, plus both derived answers."""

    ctx_fingerprint: str
    draw_index: int
    branches: Dict[str, BranchEvaluation]

    @property
    def frontier_delta(self) -> float:
        """The ladder's answer: CE(UP) - CE(RP)."""
        return self.branches["UP"].ce - self.branches["RP"].ce

    @property
    def components(self) -> Dict[str, float]:
        b = self.branches
        return {name: b[a].ce - b[c].ce for name, a, c in COMPONENT_ORDER}

    @property
    def decomposition_total(self) -> float:
        """The decomposition's answer: the four components summed."""
        return sum(self.components.values())

    @property
    def residual(self) -> float:
        return self.decomposition_total - self.frontier_delta

    @property
    def agrees(self) -> bool:
        return abs(self.residual) <= AGREEMENT_TOLERANCE

    def to_dict(self) -> Dict[str, object]:
        return {
            "ctx_fingerprint": self.ctx_fingerprint,
            "draw_index": self.draw_index,
            "frontier_delta": round(self.frontier_delta, 10),
            "decomposition_total": round(self.decomposition_total, 10),
            "residual": self.residual,
            "agrees": self.agrees,
            "components": {k: round(v, 8) for k, v in self.components.items()},
            "branches": {k: v.to_dict() for k, v in self.branches.items()},
        }


def evaluate_draw(ctx: EvalContext, *, proxy) -> DrawEvaluation:
    """Evaluate every branch of one draw from one context.

    The ladder's delta and the decomposition's total are then two readings of
    the *same five numbers*, so they agree by construction rather than by
    coincidence -- and the residual proves it every time.
    """
    branches = {b: evaluate_branch(ctx, b, proxy=proxy) for b in BRANCHES}
    return DrawEvaluation(ctx_fingerprint=ctx.fingerprint(),
                          draw_index=ctx.draw.index, branches=branches)


@dataclass
class AgreementReport:
    """Per-draw agreement across an allocation ensemble."""

    p: int
    q: int
    pass_rule: str
    recipient: str
    draws: Tuple[DrawEvaluation, ...]

    @property
    def k(self) -> int:
        return len(self.draws)

    @property
    def max_residual(self) -> float:
        return max((abs(d.residual) for d in self.draws), default=0.0)

    @property
    def all_agree(self) -> bool:
        return all(d.agrees for d in self.draws)

    @property
    def mean_delta(self) -> float:
        return statistics.fmean(d.frontier_delta for d in self.draws)

    @property
    def between_sd(self) -> float:
        vals = [d.frontier_delta for d in self.draws]
        return statistics.stdev(vals) if len(vals) > 1 else 0.0

    def component_means(self) -> Dict[str, float]:
        return {name: statistics.fmean(d.components[name] for d in self.draws)
                for name, _, _ in COMPONENT_ORDER}

    @property
    def ci95(self) -> Tuple[float, float]:
        from .ensemble import _t95
        if self.k < 2:
            return (float("nan"), float("nan"))
        se = self.between_sd / math.sqrt(self.k)
        h = _t95(self.k - 1) * se
        return (self.mean_delta - h, self.mean_delta + h)

    @property
    def verdict(self) -> str:
        lo, hi = self.ci95
        if math.isnan(lo):
            return "unresolved"
        if lo > 0:
            return "favorable"
        if hi < 0:
            return "unfavorable"
        return "unresolved"

    def to_dict(self, *, include_draws: bool = True) -> Dict[str, object]:
        lo, hi = self.ci95
        out: Dict[str, object] = {
            "p": self.p, "q": self.q, "pass_rule": self.pass_rule,
            "recipient": self.recipient, "k": self.k,
            "frontier_mean_delta": round(self.mean_delta, 8),
            "decomposition_total_mean": round(
                statistics.fmean(d.decomposition_total for d in self.draws), 8),
            "components": {k: round(v, 8)
                           for k, v in self.component_means().items()},
            "between_allocation_sd": round(self.between_sd, 8),
            "ci95": None if math.isnan(lo) else [round(lo, 8), round(hi, 8)],
            "verdict": self.verdict,
            "max_per_draw_residual": self.max_residual,
            "agreement_tolerance": AGREEMENT_TOLERANCE,
            "all_draws_agree": self.all_agree,
            "note": ("frontier delta and decomposition total are two readings "
                     "of the same five branch CEs; agreement is checked per "
                     "draw, never only after averaging"),
        }
        if include_draws:
            out["draws"] = [d.to_dict() for d in self.draws]
        return out


def check_agreement(contexts: Sequence[EvalContext], *, proxy
                    ) -> AgreementReport:
    """Evaluate one price across an ensemble; refuse mismatched contexts."""
    if not contexts:
        raise ContextMismatch("no contexts supplied")
    head = contexts[0]
    for other in contexts[1:]:
        for name in ("candidate_id", "p", "q", "recipient", "scenario"):
            if getattr(head, name) != getattr(other, name):
                raise ContextMismatch(
                    f"ensemble contexts disagree on {name}: "
                    f"{getattr(head, name)!r} != {getattr(other, name)!r}")
        if head.opportunity_fingerprint() != other.opportunity_fingerprint():
            raise ContextMismatch(
                "ensemble contexts offer different opportunity sets; every "
                "draw of one price must see the same offer")
    return AgreementReport(
        p=head.p, q=head.q, pass_rule=head.pass_price.mode.value,
        recipient=head.recipient,
        draws=tuple(evaluate_draw(c, proxy=proxy) for c in contexts))
