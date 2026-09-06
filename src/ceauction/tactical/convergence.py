"""Search-effort convergence for the quota-free marginal diagnostic.

The previous branch called the diagnostic converged because it stopped
violating *price* monotonicity. That was necessary and nowhere near sufficient.
Its own search-budget table showed the $18 improvement moving 2.29 weekly
points and reversing sign between ``160/90`` and ``320/140``:

    beam/pool    improve@$18    improve@$1
       160/90         +1.21         +1.16
      320/140         -1.08         +0.86

A legal, price-monotone search is not converged if doubling its effort changes
a player's weekly improvement by multiple points or flips his role. Those two
numbers decide whether a real player receives four thousand seasons of
championship-equity simulation.

Two different monotonicities are at stake and only one was being checked:

**Price monotonicity** -- acquiring the same player for less cannot be worse.
Enforced already; still checked here.

**Search-effort monotonicity** -- spending more compute cannot produce a worse
answer. A wider beam is a *different* heuristic, not a strictly better one, so
two independent beams routinely find complementary solutions and the wider one
can miss what the narrower one found. Comparing independent runs and shrugging
at the disagreement is not an option: this module accumulates every completion
any level discovered into one union, re-scores the union consistently, and
takes the best legal member. Monotonicity then holds **by construction** rather
than by hope, and the remaining question is the honest one -- does the union
stop improving as effort grows.

Convergence requires the objective to settle *and* the classification to settle.
A candidate whose improvement is stable to 0.25 points but whose role still
flips between levels is not converged, because role is what the sampling
decision actually reads.
"""

from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

from ..auction.completion import (Completion, CompletionSettings,
                                  complete_roster)
from ..auction.costs import CostBook
from ..auction.proxy import ProxyEvaluator
from ..auction.state import AuctionState

__all__ = [
    "EffortLevel",
    "DEFAULT_LADDER",
    "CompletionUnion",
    "LevelResult",
    "ConvergenceReport",
    "DIAGNOSTIC_NOT_CONVERGED",
    "CONVERGED",
    "TOLERANCES",
    "gather",
    "converged_diagnose",
]

DIAGNOSTIC_NOT_CONVERGED = "DIAGNOSTIC_NOT_CONVERGED"
CONVERGED = "CONVERGED"

#: Reported under both. Neither is a claim about economic materiality; they are
#: numerical stability thresholds and the report says so.
TOLERANCES: Tuple[float, ...] = (0.25, 0.50)


@dataclass(frozen=True)
class EffortLevel:
    """One rung of the search-effort ladder."""

    beam_width: int
    candidate_pool: int

    @property
    def label(self) -> str:
        return f"{self.beam_width}/{self.candidate_pool}"

    def settings(self, base: CompletionSettings) -> CompletionSettings:
        return dataclasses.replace(
            base, beam_width=self.beam_width,
            candidate_pool=self.candidate_pool,
            proxy_candidates=max(base.proxy_candidates, self.beam_width // 2),
            max_candidates=max(base.max_candidates, self.beam_width * 4))

    def to_dict(self) -> Dict[str, object]:
        return {"beam_width": self.beam_width,
                "candidate_pool": self.candidate_pool, "label": self.label}


DEFAULT_LADDER: Tuple[EffortLevel, ...] = (
    EffortLevel(64, 60), EffortLevel(160, 90),
    EffortLevel(240, 120), EffortLevel(320, 140),
)


class CompletionUnion:
    """Every completion any effort level found, scored on one common ruler.

    Keyed by the roster itself, so the same fifteen players discovered by two
    different beams collapse to one entry. Members are re-scored with the
    caller's single :class:`ProxyEvaluator`, which is what makes objectives
    from different levels comparable at all -- a completion's own ``proxy``
    field was computed under that level's settings and is not.
    """

    def __init__(self, proxy: ProxyEvaluator, budget: int,
                 required: FrozenSet[int] = frozenset(),
                 forbidden: FrozenSet[int] = frozenset()):
        self.proxy = proxy
        self.budget = int(budget)
        self.required = frozenset(required)
        self.forbidden = frozenset(forbidden)
        self._members: Dict[FrozenSet[int], Tuple[Completion, float]] = {}
        self.rejected_unaffordable = 0
        self.rejected_illegal = 0

    def add_many(self, completions: Sequence[Completion]) -> int:
        """Returns how many genuinely new rosters this batch contributed."""
        added = 0
        for c in completions:
            key = frozenset(c.roster)
            if key in self._members:
                continue
            if not self.required <= key:
                self.rejected_illegal += 1
                continue
            if key & self.forbidden:
                self.rejected_illegal += 1
                continue
            if c.added_cost > self.budget:
                self.rejected_unaffordable += 1
                continue
            self._members[key] = (c, float(self.proxy.strength(c.roster)))
            added += 1
        return added

    def __len__(self) -> int:
        return len(self._members)

    @property
    def best(self) -> Optional[Tuple[Completion, float]]:
        if not self._members:
            return None
        return max(self._members.values(),
                   key=lambda t: (t[1], -t[0].added_cost,
                                  -min(t[0].roster, default=0)))

    def best_objective(self) -> float:
        b = self.best
        return b[1] if b is not None else float("-inf")

    def keys(self) -> FrozenSet[FrozenSet[int]]:
        return frozenset(self._members)

    def members(self) -> List[Completion]:
        """Every accumulated completion, for folding into a cheaper price.

        A completion's ``added_cost`` counts only the players bought *besides*
        the candidate, and the candidate is already owned in both the $p and $1
        states, so the same cost applies to both. That is what makes a
        cross-price transfer sound rather than convenient.
        """
        return [c for c, _ in self._members.values()]


@dataclass(frozen=True)
class LevelResult:
    """What one rung of the ladder produced, over the accumulated union."""

    level: EffortLevel
    generated_with: int
    generated_without: int
    new_with: int
    new_without: int
    union_with: int
    union_without: int
    best_with: float
    best_without: float
    improvement_at_price: float
    improvement_at_min: float
    role: str
    policy: str
    composition: Dict[str, int]
    roster_fingerprint: str
    added_cost: int
    budget_left: int
    exactness: str
    runtime_s: float

    def to_dict(self) -> Dict[str, object]:
        out = dataclasses.asdict(self)
        out["level"] = self.level.to_dict()
        for k in ("best_with", "best_without", "improvement_at_price",
                  "improvement_at_min"):
            out[k] = round(out[k], 4)
        out["runtime_s"] = round(self.runtime_s, 2)
        return out


@dataclass
class ConvergenceReport:
    """The ladder, its violations, and whether the answer actually settled."""

    levels: Tuple[LevelResult, ...]
    tolerances: Tuple[float, ...] = TOLERANCES
    price_violations: Tuple[Tuple[str, float], ...] = ()
    search_violations: Tuple[Tuple[str, float], ...] = ()
    """``(level_label, regression)`` where the union objective fell. Empty by
    construction unless the union machinery is broken, which is the point of
    keeping the check."""

    @property
    def final(self) -> LevelResult:
        return self.levels[-1]

    def objective_deltas(self) -> Tuple[float, ...]:
        return tuple(b.best_with - a.best_with
                     for a, b in zip(self.levels, self.levels[1:]))

    def improvement_deltas(self) -> Tuple[float, ...]:
        return tuple(b.improvement_at_price - a.improvement_at_price
                     for a, b in zip(self.levels, self.levels[1:]))

    @property
    def role_stable(self) -> bool:
        """Did the last two levels agree on the role?"""
        return len(self.levels) >= 2 and \
            self.levels[-1].role == self.levels[-2].role

    @property
    def policy_stable(self) -> bool:
        return len(self.levels) >= 2 and \
            self.levels[-1].policy == self.levels[-2].policy

    @property
    def roster_stable(self) -> bool:
        return len(self.levels) >= 2 and \
            self.levels[-1].roster_fingerprint == \
            self.levels[-2].roster_fingerprint

    def status(self, tolerance: float) -> str:
        """Converged only if the objective settled AND the decision settled.

        Role and policy are included deliberately. An improvement stable to a
        quarter point whose role still flips is not converged for any purpose
        this diagnostic serves, because role is what the sampling decision
        reads.
        """
        if len(self.levels) < 2:
            return DIAGNOSTIC_NOT_CONVERGED
        last_obj = abs(self.objective_deltas()[-1])
        last_imp = abs(self.improvement_deltas()[-1])
        if last_obj > tolerance or last_imp > tolerance:
            return DIAGNOSTIC_NOT_CONVERGED
        if not (self.role_stable and self.policy_stable):
            return DIAGNOSTIC_NOT_CONVERGED
        if self.price_violations or self.search_violations:
            return DIAGNOSTIC_NOT_CONVERGED
        return CONVERGED

    def effort_required(self, tolerance: float) -> Optional[str]:
        """The first level after which nothing moved by more than tolerance."""
        if self.status(tolerance) != CONVERGED:
            return None
        for i in range(1, len(self.levels)):
            objs = [abs(d) for d in self.objective_deltas()[i - 1:]]
            imps = [abs(d) for d in self.improvement_deltas()[i - 1:]]
            roles = {l.role for l in self.levels[i - 1:]}
            pols = {l.policy for l in self.levels[i - 1:]}
            if (all(o <= tolerance for o in objs)
                    and all(v <= tolerance for v in imps)
                    and len(roles) == 1 and len(pols) == 1):
                return self.levels[i - 1].level.label
        return self.final.level.label

    def to_dict(self) -> Dict[str, object]:
        return {
            "levels": [l.to_dict() for l in self.levels],
            "objective_deltas": [round(d, 4) for d in self.objective_deltas()],
            "improvement_deltas": [round(d, 4)
                                   for d in self.improvement_deltas()],
            "role_stable": self.role_stable,
            "policy_stable": self.policy_stable,
            "roster_stable": self.roster_stable,
            "price_violations": [list(v) for v in self.price_violations],
            "search_violations": [list(v) for v in self.search_violations],
            "status": {str(t): self.status(t) for t in self.tolerances},
            "effort_required": {str(t): self.effort_required(t)
                                for t in self.tolerances},
            "tolerance_note": ("tolerances are NUMERICAL STABILITY thresholds "
                               "in weekly starting points, not claims about "
                               "what size of difference is economically "
                               "material"),
        }


def gather(state: AuctionState, cast, costs: CostBook, *,
           proxy: ProxyEvaluator, base: CompletionSettings,
           level: EffortLevel, reserved: FrozenSet[int],
           union: CompletionUnion, note: str) -> Tuple[int, int, str]:
    """Run one level and fold everything it found into ``union``.

    Every finalist is kept, not just the winner: a completion the wider beam
    ranked second may be the one a narrower beam ranked first, and discarding
    it is exactly how search-effort monotonicity gets lost.
    """
    focus = state.focus_owner_id
    res = complete_roster(state, cast, costs, settings=level.settings(base),
                          owner_id=focus, evaluate_ce=False, default_cost=1,
                          proxy=proxy, reserved_ids=reserved, notes=note)
    found = list(res.finalists) or ([res.best] if res.best else [])
    new = union.add_many(found)
    return len(found), new, res.result_kind


def _fingerprint(roster: Sequence[int]) -> str:
    import hashlib
    return hashlib.sha256(
        ",".join(str(p) for p in sorted(roster)).encode()).hexdigest()[:16]


def converged_diagnose(board, cand, *, proxy: ProxyEvaluator,
                       price: Optional[int] = None,
                       ladder: Sequence[EffortLevel] = DEFAULT_LADDER,
                       base: Optional[CompletionSettings] = None,
                       tolerances: Sequence[float] = TOLERANCES,
                       ) -> Tuple["object", ConvergenceReport]:
    """Walk the effort ladder over accumulated unions; report if it settled.

    Returns ``(CandidateDiagnostics, ConvergenceReport)``. The diagnostics come
    from the *final* union, which contains everything every level found, so its
    objective is the best any amount of effort discovered rather than whatever
    the last beam happened to return.
    """
    from .realpilot import (BENCH_IMPROVEMENT, MARGINAL_IMPROVEMENT,
                            MARGINAL_SHARE, NO_CONTINGENCY_MODEL,
                            STARTER_SHARE, BranchRoster, CandidateDiagnostics,
                            _composition)

    st = board.state
    focus = board.focus_owner_id
    cid = cand.player_id
    p = int(price if price is not None else (cand.price_base or 1))
    costs = board.costs["base"]
    cs = base or CompletionSettings(
        beam_width=64, candidate_pool=60, proxy_candidates=48, finalists=6,
        max_candidates=400, proxy_reps=16)
    cs = dataclasses.replace(cs, finalists=max(cs.finalists, 6))

    owner = st.owner(focus)
    shortfall = st.purchase_shortfall(cid, focus, p)
    if shortfall is not None:
        raise ValueError(f"candidate {cid} cannot be bought at ${p}: {shortfall}")
    with_state = st.apply_purchase(cid, focus, p)
    min_state = (st.apply_purchase(cid, focus, 1)
                 if st.purchase_shortfall(cid, focus, 1) is None else None)

    u_without = CompletionUnion(proxy, owner.budget_remaining,
                                forbidden=frozenset({cid}))
    u_with = CompletionUnion(proxy, with_state.owner(focus).budget_remaining,
                             required=frozenset({cid}))
    u_min = (CompletionUnion(proxy, min_state.owner(focus).budget_remaining,
                             required=frozenset({cid}))
             if min_state is not None else None)

    levels: List[LevelResult] = []
    price_bad: List[Tuple[str, float]] = []
    search_bad: List[Tuple[str, float]] = []
    prev_with = float("-inf")

    for lv in ladder:
        t0 = time.perf_counter()
        gen_wo, new_wo, kind_wo = gather(
            st, board.cast, costs, proxy=proxy, base=cs, level=lv,
            reserved=frozenset({cid}), union=u_without,
            note=f"union WITHOUT candidate at {lv.label}")
        gen_w, new_w, kind_w = gather(
            with_state, board.cast, costs, proxy=proxy, base=cs, level=lv,
            reserved=frozenset(), union=u_with,
            note=f"union WITH candidate at {lv.label}")
        if u_min is not None:
            gather(min_state, board.cast, costs, proxy=proxy, base=cs,
                   level=lv, reserved=frozenset(), union=u_min,
                   note=f"union WITH candidate at $1, {lv.label}")
            # CROSS-PRICE NESTING, enforced across the unions and not merely
            # within each price's own search. Every construction affordable
            # when the candidate costs $p is affordable when he costs $1: the
            # budget is strictly larger and the other fourteen cost the same.
            # Without this fold the two prices are two independent heuristics
            # again, and the $1 branch can miss what the $p branch found --
            # which is exactly the 5.94-point violation this produced.
            u_min.add_many(u_with.members())

        bw, bwo = u_with.best_objective(), u_without.best_objective()
        if bw < prev_with - 1e-9:
            search_bad.append((lv.label, prev_with - bw))
        prev_with = bw

        gain = bw - bwo
        at_min = (u_min.best_objective() - bwo) if u_min is not None else gain
        if p > 1 and gain - at_min > 1e-6:
            price_bad.append((lv.label, gain - at_min))

        best = u_with.best
        roster = tuple(best[0].roster) if best else ()
        shares = proxy.lineup_shares(roster) if roster else {}
        share = shares.get(cid, 0.0)

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
        policy = ("4000-season audit"
                  if role in ("clear starter", "marginal starter")
                  else "proxy only")
        levels.append(LevelResult(
            level=lv, generated_with=gen_w, generated_without=gen_wo,
            new_with=new_w, new_without=new_wo, union_with=len(u_with),
            union_without=len(u_without), best_with=bw, best_without=bwo,
            improvement_at_price=gain, improvement_at_min=at_min, role=role,
            policy=policy, composition=_composition(st, roster),
            roster_fingerprint=_fingerprint(roster),
            added_cost=best[0].added_cost if best else 0,
            budget_left=(with_state.owner(focus).budget_remaining
                         - (best[0].added_cost if best else 0)),
            exactness=kind_w, runtime_s=time.perf_counter() - t0))

    report = ConvergenceReport(levels=tuple(levels),
                               tolerances=tuple(tolerances),
                               price_violations=tuple(price_bad),
                               search_violations=tuple(search_bad))
    final = levels[-1]
    best_w, best_wo = u_with.best, u_without.best
    with_roster = tuple(best_w[0].roster)
    without_roster = tuple(best_wo[0].roster) if best_wo else ()
    shares_w = proxy.lineup_shares(with_roster)
    shares_wo = proxy.lineup_shares(without_roster) if without_roster else {}
    gone = [pid for pid in without_roster if pid not in set(with_roster)]
    displaced = (max(gone, key=lambda x: shares_wo.get(x, 0.0))
                 if gone else None)

    # A candidate whose classification never settled must NOT be quietly filed
    # as proxy-only: that would let an unconverged search silently withhold
    # championship-equity simulation from a player who may deserve it.
    status = report.status(tolerances[0])
    policy = final.policy
    reason = (f"best-legal-eight improves {final.improvement_at_price:+.2f}/wk "
              f"at ${p}; candidate starts {shares_w.get(cid, 0.0):.0%} of weeks")
    if status == DIAGNOSTIC_NOT_CONVERGED:
        policy = "unresolved -- audit conservatively"
        reason = (f"{DIAGNOSTIC_NOT_CONVERGED} at tolerance {tolerances[0]}: "
                  f"objective delta {report.objective_deltas()[-1]:+.2f}, "
                  f"improvement delta {report.improvement_deltas()[-1]:+.2f}, "
                  f"role stable {report.role_stable}. The sampling decision is "
                  f"UNRESOLVED, not proxy-only: an unconverged search may not "
                  f"withhold CE simulation by default")

    diag = CandidateDiagnostics(
        player_id=cid, position=cand.position, tier=cand.tier, price=p,
        without=BranchRoster(
            player_ids=without_roster,
            composition=_composition(st, without_roster),
            best_eight=final.best_without,
            added_cost=best_wo[0].added_cost if best_wo else 0,
            budget_left=owner.budget_remaining
            - (best_wo[0].added_cost if best_wo else 0),
            exactness=f"union of {len(ladder)} effort levels",
            start_shares=shares_wo),
        with_candidate=BranchRoster(
            player_ids=with_roster, composition=final.composition,
            best_eight=final.best_with, added_cost=final.added_cost,
            budget_left=final.budget_left,
            exactness=f"union of {len(ladder)} effort levels",
            start_shares=shares_w),
        lineup_improvement=final.improvement_at_price,
        improvement_at_min=final.improvement_at_min,
        start_share=shares_w.get(cid, 0.0), displaced_player_id=displaced,
        displaced_start_share=shares_wo.get(displaced, 0.0) if displaced else 0.0,
        bench_delta=(sum(1 for v in shares_w.values() if v < 0.5)
                     - sum(1 for v in shares_wo.values() if v < 0.5)),
        role=final.role, policy=policy, reason=reason,
        contingency_note=NO_CONTINGENCY_MODEL,
        price_monotonicity_violation=max(
            (v for _, v in price_bad), default=0.0))
    return diag, report
