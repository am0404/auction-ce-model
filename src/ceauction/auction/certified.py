r"""Certified completion: the exact optimum of the proxy objective, with a bound.

`QB_UNION_CONVERGENCE.md` measured the beam's objective moving 3.8 weekly points
with beam width, non-monotonically. `TWO_SWAP_EXCHANGE.md` improved individual
rosters but could not say how far from the best any of them was. Both are the
same missing thing: **no upper bound**. A search without one cannot distinguish
"this is the answer" from "this is where I stopped".

This module supplies the bound, and with it the optimum.

The formulation
---------------

Write ``R`` for a roster (a set of player ids) and ``f`` for the objective the
rest of the auction layer already uses, :meth:`ProxyEvaluator.strength`:

.. math::

    f(R) = \\frac{1}{|\\mathcal{S}|}\\sum_{(r,w)\\in\\mathcal{S}}
           \\max\\Big\\{\\textstyle\\sum_{p\\in I} \\pi_{r,w,p}
           \;:\; I \\subseteq R \\cap A_{r,w},\; I \\in \\mathcal{I}\\Big\\}

over the scenario set :math:`\\mathcal{S}` = (availability replicate ``r``,
scoring week ``w``), where

* :math:`\\pi_{r,w,p}` is the pregame projection -- base mean plus the
  contingency uplift ``_contingency_bonus`` -- and
* :math:`A_{r,w}` is the set of players available that week, and
* :math:`\\mathcal{I}` is the family of legal starting lineups.

**Both** :math:`\\pi` **and** :math:`A` **are roster-independent.** The
availability draw is coordinate-addressed by player id, and the contingency
uplift reads the whole pool's availability rather than the roster's, so
``bonus[r, p, w]`` is the same number whichever roster ``p`` sits on. This is
the condition the brief asks to be checked before claiming the problem is an
optimisation over ``R`` alone, and it holds: no scenario weight, projection or
availability depends on the decision variables. (A player's depth-chart
superior need not be rostered for the uplift to fire -- that is a modelling
choice made upstream, and it is roster-independent, which is what matters here.)

:math:`\\mathcal{I}` is a matroid. ``lineup_vec.select_lineups_mask`` accepts a
starter set iff its position counts :math:`(n_Q, n_R, n_T)` -- quarterbacks,
running backs, and receivers-and-tight-ends together -- satisfy

.. math::

    n_Q \\le 2,\; n_R \\le 4,\; n_T \\le 5,\;
    n_Q + n_R \\le 5,\; n_Q + n_T \\le 6,\; n_R + n_T \\le 7,\;
    n_Q + n_R + n_T \\le 8 .

Those seven right-hand sides are a set function ``cap`` on ``{Q, R, T}`` with
``cap(∅) = 0``; it is monotone, and it is submodular (the two tight checks are
``cap(QR) + cap(RT) = 12 = cap(QRT) + cap(R)`` and ``cap(QT) + cap(RT) = 13 =
cap(QRT) + cap(T)``). A monotone submodular ``cap`` makes the constraint system
a **polymatroid**, so the inner maximisation is a max-weight independent set in
a matroid: the greedy that ``select_lineups_mask`` runs is exactly optimal, and
its LP relaxation is integral.

The consequence used here is the classical one: for a matroid, the max-weight
independent subset of ``S``, viewed as a function of ``S``, is monotone and
**submodular**. A nonnegative average of submodular functions is submodular, so

    ``f`` is monotone submodular on the player pool.

``localrepair`` already states this; nothing here re-derives it differently.

The feasible set
----------------

Binary ``x_p`` for each selectable board player, with owned players and any
forced candidate pinned to ``1``:

* :math:`\\sum_{p \\in B} x_p = K` -- exactly the open slots, so the roster
  reaches exactly ``roster_size``;
* :math:`\\sum_{p \\in B} c_p x_p \\le \\text{budget}` -- acquisition costs
  against the remaining budget;
* the ``$1``-per-open-slot reserve, which at completion is *implied*: after
  ``j`` of ``K`` purchases the beam requires
  ``spend_j + (K-j)\\cdot\\text{min\\_bid} \\le \\text{budget}``, and since every
  remaining cost is at least ``min_bid`` that is implied by the total budget
  row. (:func:`reserve_is_implied` checks the premise rather than assuming it.)
* the saturating Hall constraints from :mod:`.feasibility`, which are counting
  constraints because the slot eligibility sets are laminar:
  ``#QB ≥ 1``, ``#RB ≥ 2``, ``#WR+#TE ≥ 3``, ``#RB+#WR+#TE ≥ 6``, ``total ≥ 8``.

No quarterback maximum, no dedicated tight-end slot, no bench quota: bench
players are simply the roster members no scenario starts, and they earn their
place only through conditional substitution, exactly as before.

How it is solved
----------------

Submodularity is what makes this tractable. For monotone submodular ``f`` the
Nemhauser--Wolsey inequality

.. math::

    f(T) \\le f(S) + \\sum_{p \\in T\\setminus S}\\rho_p(S)
                  - \\sum_{p \\in S\\setminus T}\\rho_p(S\\setminus p)

holds for **every** pair of sets, so each evaluated roster ``S`` yields one
linear inequality valid over the whole feasible region. The master problem
maximises a scalar ``η`` subject to the roster constraints and every cut
generated so far; because each cut is valid, the master's optimum is a genuine
**upper bound**, and any roster it proposes gives a lower bound when scored.
Iterating closes a real, reportable optimality gap. The bound is not a
heuristic score and not an admissible-looking guess: it is a bound.

The solver is HiGHS via ``highspy``, an **optional** dependency. Nothing else in
the package imports this module at import time, and the draft-day dashboard runs
without it; :func:`solver_available` reports the fact rather than crashing.
"""

from __future__ import annotations

import hashlib
import itertools
import math
import time
from dataclasses import dataclass, field
from typing import (Dict, FrozenSet, Iterable, List, Mapping, Optional,
                    Sequence, Set, Tuple)

import numpy as np

from ..league import Position
from .feasibility import PositionCounts, min_additions_for_lineup
from .proxy import ProxyEvaluator

__all__ = [
    "CERTIFIED_ABS_TOLERANCE",
    "OBJECTIVE_TOLERANCE",
    "CUT_SLACK",
    "SolverUnavailable",
    "solver_available",
    "solver_version",
    "CompletionProblem",
    "CertifiedResult",
    "NearOptimalSet",
    "solve_completion_certified",
    "solve_completion_exact",
    "enumerate_near_optimal",
    "exhaustive_optimum",
    "exhaustive_ranked",
    "problem_from_state",
    "ProxyObjective",
    "reserve_is_implied",
    "roster_fingerprint",
    "LINEUP_CAP",
    "cap_is_polymatroid",
]

#: The brief's bound: a certified absolute objective gap no larger than this,
#: in weekly proxy points, is what "certified" means in this module.
CERTIFIED_ABS_TOLERANCE = 0.10

#: Declared numerical tolerance for "the same objective value".
#:
#: ``strength`` and ``strength_many`` reduce over differently-shaped arrays, so
#: they can disagree in the last few ulps -- measured at ~3e-14 on a nine-man
#: roster. That is float associativity, not a modelling difference, and the
#: whole certification is quoted to 0.10, twelve orders of magnitude above it.
OBJECTIVE_TOLERANCE = 1e-9

#: Slack added to every generated cut's right-hand side.
#:
#: The Nemhauser--Wolsey inequality is exact in exact arithmetic. Evaluated in
#: floating point, a marginal can come back a few ulps too small and make a cut
#: very slightly too tight -- which would remove the true optimum and produce an
#: upper bound that is not one. A slack far above the noise and far below the
#: reported tolerance removes that failure mode outright.
CUT_SLACK = 1e-9

#: The seven counting constraints ``select_lineups_mask`` enforces, as a set
#: function on the position groups ``Q`` (quarterback), ``R`` (running back)
#: and ``T`` (receiver or tight end).  Kept here so the polymatroid claim in
#: the module docstring is checkable by a test rather than only asserted.
LINEUP_CAP: Dict[FrozenSet[str], int] = {
    frozenset(): 0,
    frozenset("Q"): 2,
    frozenset("R"): 4,
    frozenset("T"): 5,
    frozenset("QR"): 5,
    frozenset("QT"): 6,
    frozenset("RT"): 7,
    frozenset("QRT"): 8,
}


def cap_is_polymatroid() -> Tuple[bool, List[str]]:
    """Is :data:`LINEUP_CAP` a monotone submodular rank function?

    Returns the verdict and every violation found, so a failure names itself.
    A ``True`` here is the licence for two separate claims: that the inner
    lineup LP is integral, and that ``f`` is submodular in the roster.
    """
    problems: List[str] = []
    ground = "QRT"
    subsets = [frozenset(c) for k in range(4)
               for c in itertools.combinations(ground, k)]
    if LINEUP_CAP[frozenset()] != 0:
        problems.append("cap(empty) != 0")
    for a in subsets:
        for b in subsets:
            if a <= b and LINEUP_CAP[a] > LINEUP_CAP[b]:
                problems.append(f"not monotone: cap({set(a)}) > cap({set(b)})")
            lhs = LINEUP_CAP[a] + LINEUP_CAP[b]
            rhs = LINEUP_CAP[a | b] + LINEUP_CAP[a & b]
            if lhs < rhs:
                problems.append(
                    f"not submodular at {sorted(a)}/{sorted(b)}: "
                    f"{lhs} < {rhs}")
    return (not problems), problems


class SolverUnavailable(RuntimeError):
    """Raised when ``highspy`` is not installed.

    Deliberately its own type: a caller that must degrade to the heuristic
    search should catch this and say ``CERTIFIED_SOLVER_NO_GO``, not treat a
    missing optional dependency as a modelling failure.
    """


def solver_available() -> bool:
    """Is the optional MILP solver importable?  Never raises."""
    try:
        import highspy  # noqa: F401
    except Exception:
        return False
    return True


def solver_version() -> Optional[str]:
    try:
        import highspy
    except Exception:
        return None
    return f"highspy {getattr(highspy, '__version__', 'unknown')}"


def roster_fingerprint(player_ids: Iterable[int]) -> str:
    """Order-independent 16-hex identity for a roster."""
    body = ",".join(str(int(p)) for p in sorted(set(int(p) for p in player_ids)))
    return hashlib.sha256(body.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# The problem
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CompletionProblem:
    """One focus-team completion, stated as data rather than as an AuctionState.

    Keeping it free of :class:`AuctionState` is what lets the same solver run on
    the fabricated fixtures Phase 3 needs and on the real board, so the thing
    certified against exhaustive enumeration is the thing that runs for real.
    """

    owned: Tuple[int, ...]
    """Already on the roster. Fixed to 1; costs already paid."""

    forced: Tuple[int, ...]
    """Candidate ownership forced by the branch being priced. Fixed to 1, and
    the price paid is already out of :attr:`budget`."""

    board: Tuple[int, ...]
    """Selectable players. Must exclude everything in owned/forced."""

    cost: Mapping[int, int]
    """Acquisition cost for each board player."""

    position: Mapping[int, int]
    """Position code for every player mentioned anywhere."""

    budget: int
    """Dollars available for the completion, after any forced purchase."""

    roster_size: int = 15
    min_bid: int = 1

    def __post_init__(self) -> None:
        fixed = set(self.owned) | set(self.forced)
        if len(fixed) != len(self.owned) + len(self.forced):
            raise ValueError("owned and forced overlap")
        if fixed & set(self.board):
            raise ValueError("board overlaps owned/forced")
        if len(set(self.board)) != len(self.board):
            raise ValueError("duplicate board player")
        if self.slots_to_fill < 0:
            raise ValueError("roster already over size")
        for pid in self.board:
            if pid not in self.cost:
                raise ValueError(f"no cost for board player {pid}")
        for pid in tuple(self.board) + tuple(fixed):
            if pid not in self.position:
                raise ValueError(f"no position for player {pid}")

    @property
    def fixed(self) -> Tuple[int, ...]:
        return tuple(self.owned) + tuple(self.forced)

    @property
    def slots_to_fill(self) -> int:
        return self.roster_size - len(self.owned) - len(self.forced)

    @property
    def fixed_counts(self) -> PositionCounts:
        return PositionCounts.from_positions(
            Position(int(self.position[p])) for p in self.fixed)

    def counts_of(self, roster: Sequence[int]) -> PositionCounts:
        return PositionCounts.from_positions(
            Position(int(self.position[p])) for p in roster)

    def spend_of(self, roster: Sequence[int]) -> int:
        return sum(int(self.cost[p]) for p in roster if p in self.cost
                   and p not in set(self.fixed))

    def is_legal(self, roster: Sequence[int]) -> Tuple[bool, str]:
        """Every constraint, checked independently of how the roster was found.

        Used by the fixtures and by the real runs to verify solver output
        rather than to trust it.
        """
        ids = list(roster)
        if len(set(ids)) != len(ids):
            return False, "duplicate player"
        s = set(ids)
        if not set(self.fixed) <= s:
            return False, "owned or forced player missing"
        extra = s - set(self.fixed)
        if not extra <= set(self.board):
            return False, "player from outside the board"
        if len(ids) != self.roster_size:
            return False, f"roster size {len(ids)} != {self.roster_size}"
        spend = sum(int(self.cost[p]) for p in extra)
        if spend > self.budget:
            return False, f"spend {spend} over budget {self.budget}"
        need, _ = min_additions_for_lineup(self.counts_of(ids))
        if need > 0:
            return False, "cannot field a legal lineup"
        return True, ""

    def fingerprint(self) -> str:
        body = "|".join([
            ",".join(map(str, sorted(self.owned))),
            ",".join(map(str, sorted(self.forced))),
            ",".join(f"{p}:{self.cost[p]}" for p in sorted(self.board)),
            f"budget={self.budget}",
            f"size={self.roster_size}",
            f"minbid={self.min_bid}",
        ])
        return hashlib.sha256(body.encode()).hexdigest()[:16]


def reserve_is_implied(problem: CompletionProblem) -> bool:
    """Does the total-budget row already enforce the per-slot reserve?

    True when every board cost is at least ``min_bid``.  The beam enforces
    ``spend_j + (K-j)*min_bid <= budget`` at each step; if every unbought
    player costs at least ``min_bid`` then that quantity is bounded by the
    final total spend, so the single budget row implies all of them.  When it
    is False the caller must not use the compact model, and
    :func:`solve_completion_certified` refuses rather than quietly relaxing a
    constraint.
    """
    return all(int(problem.cost[p]) >= problem.min_bid for p in problem.board)


# ---------------------------------------------------------------------------
# The objective oracle
# ---------------------------------------------------------------------------


class ProxyObjective:
    """``f`` as the solver needs it: batched, cached, and provably the same ``f``.

    Every value is produced by :meth:`ProxyEvaluator.strength` or its batched
    sibling ``strength_many``, never by a reimplementation, so "the objective
    exactly reproduces the current ProxyEvaluator" is true by construction
    rather than by test. ``tests/test_certified_solver.py`` asserts the batched
    and scalar paths agree anyway.
    """

    def __init__(self, proxy: ProxyEvaluator):
        self.proxy = proxy
        self._cache: Dict[FrozenSet[int], float] = {}
        self.n_evals = 0

    def value(self, roster: Iterable[int]) -> float:
        key = frozenset(int(p) for p in roster)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        self.n_evals += 1
        v = float(self.proxy.strength(sorted(key)))
        self._cache[key] = v
        return v

    def values(self, rosters: Sequence[Sequence[int]]) -> List[float]:
        """Batched, grouped by size because ``strength_many`` requires one size."""
        keys = [frozenset(int(p) for p in r) for r in rosters]
        out: List[Optional[float]] = [self._cache.get(k) for k in keys]
        todo = [i for i, v in enumerate(out) if v is None]
        by_size: Dict[int, List[int]] = {}
        for i in todo:
            by_size.setdefault(len(keys[i]), []).append(i)
        for size, idxs in by_size.items():
            # Deduplicate within the batch too: a marginal sweep repeats sets.
            uniq: Dict[FrozenSet[int], List[int]] = {}
            for i in idxs:
                uniq.setdefault(keys[i], []).append(i)
            batch = [sorted(k) for k in uniq]
            self.n_evals += len(batch)
            vals = self.proxy.strength_many(batch)
            for k, v in zip(uniq, vals):
                self._cache[k] = float(v)
                for i in uniq[k]:
                    out[i] = float(v)
        return [float(v) for v in out]  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CertifiedResult:
    """An incumbent, a proven bound, and the distance between them."""

    roster: Tuple[int, ...]
    objective: float
    upper_bound: float
    iterations: int
    runtime_s: float
    n_objective_evals: int
    status: str
    """``"certified"``, ``"gap"``, ``"time_limit"``, ``"infeasible"``."""
    problem_fingerprint: str = ""

    @property
    def abs_gap(self) -> float:
        return max(0.0, float(self.upper_bound) - float(self.objective))

    @property
    def rel_gap(self) -> float:
        denom = abs(self.objective)
        return self.abs_gap / denom if denom > 1e-12 else float("inf")

    @property
    def is_certified(self) -> bool:
        return (self.status == "certified"
                and self.abs_gap <= CERTIFIED_ABS_TOLERANCE + 1e-9)

    @property
    def fingerprint(self) -> str:
        return roster_fingerprint(self.roster)

    def to_dict(self) -> Dict[str, object]:
        return {
            "objective": round(self.objective, 6),
            "upper_bound": round(self.upper_bound, 6),
            "abs_gap": round(self.abs_gap, 6),
            "rel_gap": round(self.rel_gap, 9),
            "iterations": self.iterations,
            "runtime_s": round(self.runtime_s, 3),
            "objective_evals": self.n_objective_evals,
            "status": self.status,
            "roster_fingerprint": self.fingerprint,
            "certified": self.is_certified,
        }


@dataclass(frozen=True)
class NearOptimalSet:
    """A bounded, diverse set of completions, each with its own gap."""

    entries: Tuple[Tuple[Tuple[int, ...], float, float], ...]
    """``(roster, objective, absolute gap to the certified optimum)``."""

    optimum: float
    upper_bound: float
    runtime_s: float
    exhausted: bool
    """True when the band was proven empty of further completions."""

    band: float = 0.50

    @property
    def rosters(self) -> Tuple[Tuple[int, ...], ...]:
        return tuple(e[0] for e in self.entries)

    @property
    def n_within_band(self) -> int:
        return sum(1 for e in self.entries if e[2] <= self.band + 1e-9)

    def to_dict(self) -> Dict[str, object]:
        return {
            "n": len(self.entries),
            "n_within_band": self.n_within_band,
            "band": self.band,
            "optimum": round(self.optimum, 6),
            "exhausted": self.exhausted,
            "runtime_s": round(self.runtime_s, 3),
            "entries": [
                {"fingerprint": roster_fingerprint(r),
                 "objective": round(v, 6),
                 "gap_to_optimum": round(g, 6)}
                for r, v, g in self.entries],
        }


# ---------------------------------------------------------------------------
# The master problem
# ---------------------------------------------------------------------------


class _Master:
    """The MILP over roster choice, carrying Nemhauser--Wolsey cuts.

    Column 0 is ``η``, the epigraph variable being maximised; columns ``1..m``
    are the board binaries.  HiGHS minimises, so the stored objective is ``-η``
    and every bound is negated on the way out.
    """

    def __init__(self, problem: CompletionProblem, upper: float,
                 lower: float = 0.0):
        try:
            import highspy
        except Exception as exc:  # pragma: no cover - exercised by absence
            raise SolverUnavailable(
                "highspy is not installed; install the optional 'solver' extra "
                "(pip install 'ceauction[solver]') or report "
                "CERTIFIED_SOLVER_NO_GO") from exc
        self._hs = highspy
        self.problem = problem
        self.board = list(problem.board)
        self.m = len(self.board)
        self.col_of = {pid: i + 1 for i, pid in enumerate(self.board)}
        self.inf = highspy.kHighsInf

        h = highspy.Highs()
        h.setOptionValue("output_flag", False)
        h.setOptionValue("mip_rel_gap", 0.0)
        h.setOptionValue("mip_abs_gap", 0.0)
        self.h = h

        n = self.m + 1
        lo = np.concatenate(([-self.inf], np.zeros(self.m)))
        hi = np.concatenate(([float(upper)], np.ones(self.m)))
        h.addVars(n, lo, hi)
        h.changeColsIntegrality(
            self.m, np.arange(1, n, dtype=np.int32),
            np.array([highspy.HighsVarType.kInteger] * self.m))
        cost = np.zeros(n)
        cost[0] = -1.0                      # maximise eta
        h.changeColsCost(n, np.arange(n, dtype=np.int32), cost)

        idx = np.arange(1, n, dtype=np.int32)
        ones = np.ones(self.m)

        # exactly K purchases
        K = float(problem.slots_to_fill)
        h.addRow(K, K, self.m, idx, ones)

        # budget
        costs = np.array([float(problem.cost[p]) for p in self.board])
        h.addRow(-self.inf, float(problem.budget), self.m, idx, costs)

        # the saturating Hall constraints, net of what is already owned
        fc = problem.fixed_counts
        pos = np.array([int(problem.position[p]) for p in self.board])
        for need, have, mask in (
            (1, fc.qb, pos == int(Position.QB)),
            (2, fc.rb, pos == int(Position.RB)),
            (3, fc.wt, pos >= int(Position.WR)),
            (6, fc.flex_eligible, pos >= int(Position.RB)),
        ):
            short = need - have
            if short > 0:
                sel = np.flatnonzero(mask).astype(np.int32) + 1
                if sel.size == 0:
                    self.trivially_infeasible = True
                h.addRow(float(short), self.inf, sel.size, sel,
                         np.ones(sel.size))
        # "all eight starting slots" is implied: roster_size > n_starters.
        self.n_cuts = 0

    def add_nw_cut(self, selected: Sequence[int], f_S: float,
                   rho_add: Mapping[int, float],
                   rho_drop: Mapping[int, float]) -> None:
        r"""``eta <= f(S) + sum rho_add[p] x_p - sum rho_drop[p] (1 - x_p)``.

        ``rho_drop`` must be the ground-set marginals from
        :func:`_ground_drop_marginals`, not marginals against ``S``.
        """
        chosen = set(int(p) for p in selected)
        idx = [0]
        val = [1.0]
        rhs = float(f_S)
        for pid in self.board:
            if pid in chosen:
                r = float(rho_drop.get(pid, 0.0))
                if r:
                    idx.append(self.col_of[pid])
                    val.append(-r)
                    rhs -= r
            else:
                r = float(rho_add.get(pid, 0.0))
                if r:
                    idx.append(self.col_of[pid])
                    val.append(-r)
        self.h.addRow(-self.inf, rhs + CUT_SLACK, len(idx),
                      np.array(idx, dtype=np.int32), np.array(val, dtype=float))
        self.n_cuts += 1

    def forbid(self, roster_free: Sequence[int]) -> None:
        """No-good cut: never propose this exact free selection again."""
        sel = np.array([self.col_of[p] for p in roster_free], dtype=np.int32)
        if sel.size == 0:
            return
        self.h.addRow(-self.inf, float(len(sel) - 1), sel.size, sel,
                      np.ones(sel.size))

    def add_spend_band(self, lo: Optional[int], hi: Optional[int]) -> None:
        idx = np.arange(1, self.m + 1, dtype=np.int32)
        costs = np.array([float(self.problem.cost[p]) for p in self.board])
        self.h.addRow(-self.inf if lo is None else float(lo),
                      self.inf if hi is None else float(hi),
                      self.m, idx, costs)

    def solve(self, time_limit: Optional[float] = None
              ) -> Tuple[str, Optional[List[int]], float]:
        """Return ``(status, selected free ids, valid upper bound)``."""
        if time_limit is not None:
            self.h.setOptionValue("time_limit", max(0.01, float(time_limit)))
        self.h.run()
        status = self.h.modelStatusToString(self.h.getModelStatus())
        if status in ("Infeasible", "Primal infeasible"):
            return "infeasible", None, float("-inf")
        info = self.h.getInfo()
        sol = np.asarray(self.h.getSolution().col_value)
        if sol.size == 0:
            return "no_solution", None, float("inf")
        # HiGHS minimises -eta; the dual bound negated is a valid upper bound
        # on eta, and eta bounds the objective because every cut is valid.
        bound = -float(info.mip_dual_bound)
        picked = [pid for pid in self.board
                  if sol[self.col_of[pid]] > 0.5]
        return ("optimal" if status == "Optimal" else "limit"), picked, bound


# ---------------------------------------------------------------------------
# The solve
# ---------------------------------------------------------------------------


def _ground_drop_marginals(objective: ProxyObjective,
                           problem: CompletionProblem) -> Dict[int, float]:
    r"""``rho_p(N \ p)`` for every board player, where ``N`` is the ground set.

    This is the coefficient the drop half of a Nemhauser--Wolsey cut must use,
    and the reason is worth stating because getting it wrong produces a bound
    that looks like one and is not.

    The exact inequality is

    .. math::

        f(T) \le f(S) + \sum_{j \in T\setminus S}\rho_j(S)
              - \sum_{j \in S\setminus T}\rho_j(S \cup T \setminus j),

    whose last term depends on ``T`` and so cannot appear in a linear cut. It
    is made linear by replacing it with the *smallest* marginal any set can
    give, which by submodularity is the one taken against the whole ground set.
    Substituting ``rho_j(S \ j)`` instead -- the marginal against the incumbent,
    which is the more natural-looking quantity -- subtracts something **too
    large**, tightens the cut past validity and can remove the true optimum.
    A fixture caught exactly that; ``test_certified_equals_exhaustive[60]``
    reported an "upper bound" of 93.91 against a true optimum of 94.78.

    Computed once per problem and reused by every cut, so the correct form is
    also the cheaper one.
    """
    ground = list(problem.fixed) + list(problem.board)
    f_ground = objective.value(ground)
    board = list(problem.board)
    if not board:
        return {}
    vals = objective.values([[q for q in ground if q != p] for p in board])
    return {p: max(0.0, f_ground - v) for p, v in zip(board, vals)}


def _add_marginals(objective: ProxyObjective, problem: CompletionProblem,
                   picked: Sequence[int], f_S: float) -> Dict[int, float]:
    r"""``rho_p(S)`` for every board player not in ``S``.

    Valid as an upper bound on the marginal against any superset, which is what
    the ``T \ S`` half of the cut needs.
    """
    base = list(problem.fixed) + list(picked)
    chosen = set(picked)
    adds = [p for p in problem.board if p not in chosen]
    if not adds:
        return {}
    vals = objective.values([base + [p] for p in adds])
    return {p: max(0.0, v - f_S) for p, v in zip(adds, vals)}


def _one_swap_improve(objective: ProxyObjective, problem: CompletionProblem,
                      picked: Sequence[int], value: float,
                      max_rounds: int = 8) -> Tuple[List[int], float]:
    """Cheap incumbent polish, used only to raise the lower bound faster.

    It changes nothing about correctness -- the bound comes from the master --
    but a better incumbent early means fewer cuts to close the same gap.
    """
    cur = list(picked)
    cur_val = value
    chosen = set(cur)
    outside = [p for p in problem.board if p not in chosen]
    base_spend = sum(int(problem.cost[p]) for p in cur)
    for _ in range(max_rounds):
        cands: List[Tuple[int, int]] = []
        for out in cur:
            for inn in outside:
                spend = base_spend - int(problem.cost[out]) + int(problem.cost[inn])
                if spend > problem.budget:
                    continue
                cands.append((out, inn))
        if not cands:
            break
        rosters = []
        for out, inn in cands:
            r = [q for q in cur if q != out] + [inn]
            rosters.append(list(problem.fixed) + r)
        vals = objective.values(rosters)
        best_i = int(np.argmax(vals))
        if vals[best_i] <= cur_val + 1e-9:
            break
        out, inn = cands[best_i]
        trial = [q for q in cur if q != out] + [inn]
        ok, _ = problem.is_legal(list(problem.fixed) + trial)
        if not ok:
            # Reject illegal swaps individually rather than abandoning the
            # round: the best legal one may still be an improvement.
            order = np.argsort(-np.asarray(vals))
            moved = False
            for i in order:
                if vals[i] <= cur_val + 1e-9:
                    break
                o, n = cands[int(i)]
                t = [q for q in cur if q != o] + [n]
                if problem.is_legal(list(problem.fixed) + t)[0]:
                    trial, out, inn = t, o, n
                    best_i = int(i)
                    moved = True
                    break
            if not moved:
                break
        cur = trial
        cur_val = float(vals[best_i])
        chosen = set(cur)
        outside = [p for p in problem.board if p not in chosen]
        base_spend = sum(int(problem.cost[p]) for p in cur)
    return cur, cur_val


def solve_completion_certified(
    problem: CompletionProblem,
    proxy: ProxyEvaluator,
    *,
    tolerance: float = CERTIFIED_ABS_TOLERANCE,
    max_iterations: int = 400,
    time_limit_s: Optional[float] = None,
    seed_rosters: Sequence[Sequence[int]] = (),
    polish: bool = True,
    objective: Optional[ProxyObjective] = None,
    _master: Optional["_Master"] = None,
) -> CertifiedResult:
    """Maximise ``f`` over the legal completions, with a proven upper bound.

    ``seed_rosters`` are complete rosters (fixed players included) from any
    other source -- a beam, the accumulated union, a repaired roster. They cost
    one cut each and can only help: a cut at a good set both raises the
    incumbent and tightens the bound near the optimum.
    """
    t0 = time.perf_counter()
    if not reserve_is_implied(problem):
        raise ValueError(
            "a board cost below min_bid means the compact budget row does not "
            "imply the per-slot reserve; refusing rather than relaxing it")

    obj = objective if objective is not None else ProxyObjective(proxy)
    fixed = list(problem.fixed)

    if problem.slots_to_fill == 0:
        v = obj.value(fixed)
        return CertifiedResult(tuple(sorted(fixed)), v, v, 0,
                               time.perf_counter() - t0, obj.n_evals,
                               "certified", problem.fingerprint())

    ground = fixed + list(problem.board)
    upper0 = obj.value(ground)          # monotone: f(T) <= f(ground) for all T
    rho_drop = _ground_drop_marginals(obj, problem)

    master = _master if _master is not None else _Master(problem, upper0)

    best_roster: Optional[List[int]] = None
    best_val = float("-inf")
    bound = upper0
    status = "gap"
    iters = 0

    def consider(free_sel: Sequence[int]) -> Optional[float]:
        """Score a proposal, update the incumbent, and cut at it."""
        nonlocal best_roster, best_val
        roster = fixed + list(free_sel)
        legal, why = problem.is_legal(roster)
        f_S = obj.value(roster)
        if legal and f_S > best_val:
            best_val, best_roster = f_S, list(free_sel)
        add = _add_marginals(obj, problem, list(free_sel), f_S)
        master.add_nw_cut(list(free_sel), f_S, add, rho_drop)
        return f_S

    for seed in seed_rosters:
        s = set(int(p) for p in seed)
        free_sel = [p for p in problem.board if p in s]
        if len(free_sel) != problem.slots_to_fill:
            continue
        if set(fixed) - s:
            continue
        consider(free_sel)
        if polish and best_roster is not None:
            improved, iv = _one_swap_improve(obj, problem, best_roster, best_val)
            if iv > best_val and problem.is_legal(fixed + improved)[0]:
                consider(improved)

    while iters < max_iterations:
        iters += 1
        remaining = None
        if time_limit_s is not None:
            remaining = time_limit_s - (time.perf_counter() - t0)
            if remaining <= 0:
                status = "time_limit"
                break
        mstatus, picked, mbound = master.solve(time_limit=remaining)
        if mstatus == "infeasible":
            return CertifiedResult((), float("-inf"), float("-inf"), iters,
                                   time.perf_counter() - t0, obj.n_evals,
                                   "infeasible", problem.fingerprint())
        if picked is None:
            status = "time_limit"
            break
        bound = min(bound, mbound)
        if best_roster is not None and bound - best_val <= tolerance + 1e-9:
            status = "certified"
            break
        consider(picked)
        if polish and best_roster is not None:
            improved, iv = _one_swap_improve(obj, problem, best_roster, best_val)
            if iv > best_val + 1e-9 and problem.is_legal(fixed + improved)[0]:
                consider(improved)
        if best_roster is not None and bound - best_val <= tolerance + 1e-9:
            status = "certified"
            break

    if best_roster is None:
        return CertifiedResult((), float("-inf"), bound, iters,
                               time.perf_counter() - t0, obj.n_evals,
                               "infeasible", problem.fingerprint())
    return CertifiedResult(
        tuple(sorted(fixed + best_roster)), best_val, max(bound, best_val),
        iters, time.perf_counter() - t0, obj.n_evals, status,
        problem.fingerprint())


# ---------------------------------------------------------------------------
# The exhaustive oracle, for certification only
# ---------------------------------------------------------------------------


def exhaustive_optimum(problem: CompletionProblem, proxy: ProxyEvaluator,
                       *, max_combinations: int = 2_000_000,
                       objective: Optional[ProxyObjective] = None
                       ) -> Tuple[Optional[Tuple[int, ...]], float, int]:
    """Best legal completion by brute force. Returns ``(roster, value, n)``.

    Refuses rather than hanging.  This is the oracle Phase 3 certifies the
    solver against; it must never be used on a real board.
    """
    K = problem.slots_to_fill
    n = len(problem.board)
    if K > n:
        return None, float("-inf"), 0
    total = math.comb(n, K)
    if total > max_combinations:
        raise ValueError(
            f"exhaustive enumeration would visit {total:,} combinations, "
            f"above the {max_combinations:,} limit")
    obj = objective if objective is not None else ProxyObjective(proxy)
    fixed = list(problem.fixed)
    legal: List[List[int]] = []
    for combo in itertools.combinations(problem.board, K):
        roster = fixed + list(combo)
        if problem.is_legal(roster)[0]:
            legal.append(list(combo))
    if not legal:
        return None, float("-inf"), 0
    vals = obj.values([fixed + c for c in legal])
    best = int(np.argmax(vals))
    return tuple(sorted(fixed + legal[best])), float(vals[best]), len(legal)


def exhaustive_ranked(problem: CompletionProblem, proxy: ProxyEvaluator,
                      *, max_combinations: int = 2_000_000,
                      objective: Optional[ProxyObjective] = None
                      ) -> List[Tuple[Tuple[int, ...], float]]:
    """Every legal completion with its value, best first. Fixtures only."""
    K = problem.slots_to_fill
    n = len(problem.board)
    if K > n or math.comb(n, K) > max_combinations:
        raise ValueError("refusing to enumerate")
    obj = objective if objective is not None else ProxyObjective(proxy)
    fixed = list(problem.fixed)
    legal = [list(c) for c in itertools.combinations(problem.board, K)
             if problem.is_legal(fixed + list(c))[0]]
    if not legal:
        return []
    vals = obj.values([fixed + c for c in legal])
    pairs = [(tuple(sorted(fixed + c)), float(v)) for c, v in zip(legal, vals)]
    pairs.sort(key=lambda t: (-t[1], t[0]))
    return pairs


# ---------------------------------------------------------------------------
# The near-optimal set
# ---------------------------------------------------------------------------


def enumerate_near_optimal(
    problem: CompletionProblem,
    proxy: ProxyEvaluator,
    *,
    target: int = 12,
    band: float = 0.50,
    tolerance: float = CERTIFIED_ABS_TOLERANCE,
    time_limit_s: Optional[float] = None,
    per_solve_s: Optional[float] = None,
    spend_bands: Sequence[Tuple[Optional[int], Optional[int]]] = (),
    objective: Optional[ProxyObjective] = None,
) -> NearOptimalSet:
    """The near-optimal set, built on :func:`solve_completion_exact`.

    Each round is a fresh MILP carrying one no-good cut per roster already
    emitted, so every entry is the certified best completion other than those,
    and the walk stops the moment the solver's own bound proves the band empty.
    Rebuilding the model per round costs a second; it buys a real certificate
    per entry instead of a shared one.

    Diversity is reported, never manufactured. Every entry carries its exact
    objective and its exact distance from the optimum, so a caller can see at a
    glance whether the set spans a band worth handing to championship equity or
    has collapsed onto near-clones of one roster.
    """
    t0 = time.perf_counter()
    obj = objective if objective is not None else ProxyObjective(proxy)
    entries: List[Tuple[Tuple[int, ...], float, float]] = []
    seen: Set[FrozenSet[int]] = set()
    board_set = set(problem.board)
    forbid: List[List[int]] = []
    optimum: Optional[float] = None
    upper = float("inf")
    exhausted = False

    def left() -> Optional[float]:
        if time_limit_s is None:
            return per_solve_s
        rem = time_limit_s - (time.perf_counter() - t0)
        return rem if per_solve_s is None else min(rem, per_solve_s)

    while len(entries) < target:
        lft = left()
        if lft is not None and lft <= 0:
            break
        res = solve_completion_exact(
            problem, proxy, tolerance=tolerance, time_limit_s=lft,
            objective=obj, forbid=forbid)
        if res.status == "infeasible" or not res.roster:
            exhausted = True
            break
        key = frozenset(res.roster)
        if key in seen:
            break
        seen.add(key)
        if optimum is None:
            optimum = res.objective
        upper = res.upper_bound
        entries.append((tuple(res.roster), res.objective,
                        max(0.0, optimum - res.objective)))
        forbid.append([p for p in res.roster if p in board_set])
        if res.status != "certified":
            break
        if res.upper_bound < optimum - band - 1e-9:
            exhausted = True
            break

    for lo, hi in spend_bands:
        lft = left()
        if lft is not None and lft <= 0:
            break
        res = solve_completion_exact(
            problem, proxy, tolerance=tolerance, time_limit_s=lft,
            objective=obj, forbid=forbid, spend_band=(lo, hi))
        if res.status == "infeasible" or not res.roster:
            continue
        key = frozenset(res.roster)
        if key in seen:
            continue
        seen.add(key)
        entries.append((tuple(res.roster), res.objective,
                        max(0.0, (optimum or res.objective) - res.objective)))
        forbid.append([p for p in res.roster if p in board_set])

    entries.sort(key=lambda e: (-e[1], e[0]))
    return NearOptimalSet(tuple(entries), optimum if optimum is not None
                          else float("-inf"), upper,
                          time.perf_counter() - t0, exhausted, band)


# ---------------------------------------------------------------------------
# The adapter to a real auction state
# ---------------------------------------------------------------------------


def problem_from_state(
    state,
    costs,
    *,
    owner_id: Optional[str] = None,
    candidate_pool: int = 40,
    default_cost: Optional[int] = 1,
    reserved_ids: Optional[FrozenSet[int]] = None,
) -> CompletionProblem:
    """Build the certified problem from an :class:`AuctionState`.

    The board is assembled **exactly** the way
    :func:`ceauction.auction.completion.complete_roster` assembles it -- the
    top ``candidate_pool`` by projection, plus the cheapest ``max(slots, 1)``
    players of every position as fillers -- because a certified optimum over a
    different pool would not be comparable with the beam's result over this
    one. The pool depth is a declared bound on the answer, not a detail: a
    player below the cut cannot be chosen however cheap he is, and the caller
    is expected to report the depth alongside the objective.
    """
    owner_id = owner_id or state.focus_owner_id
    owner = state.owner(owner_id)
    reserved = frozenset() if reserved_ids is None else frozenset(reserved_ids)
    board_all = [s for s in state.available_specs
                 if s.player_id not in reserved]
    board_all.sort(key=lambda s: (-s.base_mean, s.player_id))
    slots = owner.open_slots
    price_all = costs.costs_for([s.player_id for s in board_all], default_cost)

    board = list(board_all[:candidate_pool])
    chosen = {s.player_id for s in board}
    for pos in (Position.QB, Position.RB, Position.WR, Position.TE):
        cheap = sorted((s for s in board_all
                        if Position(int(s.position)) is pos
                        and s.player_id not in chosen),
                       key=lambda s: (price_all[s.player_id], -s.base_mean,
                                      s.player_id))
        for s in cheap[: max(slots, 1)]:
            board.append(s)
            chosen.add(s.player_id)
    board.sort(key=lambda s: (-s.base_mean, s.player_id))

    position = {s.player_id: int(s.position) for s in state.pool}
    return CompletionProblem(
        owned=tuple(owner.player_ids),
        forced=(),
        board=tuple(s.player_id for s in board),
        cost={s.player_id: int(price_all[s.player_id]) for s in board},
        position=position,
        budget=int(owner.budget_remaining),
        roster_size=int(state.settings.roster_size),
        min_bid=int(owner.min_bid),
    )


# ---------------------------------------------------------------------------
# The extended formulation
# ---------------------------------------------------------------------------


def _scenario_arrays(proxy: ProxyEvaluator, players: Sequence[int]):
    """``(projection, available)`` as ``(S, P)`` over scoring scenarios.

    ``S`` flattens (availability replicate, scoring week) -- the same scenario
    set :meth:`ProxyEvaluator.strength` averages over, sliced to the regular
    season for the same reason it is.
    """
    idx = np.array([proxy.index[p] for p in players], dtype=np.int64)
    w = proxy._score_weeks
    proj = proxy._projection[:, idx, :w]        # (R, P, W)
    avail = proxy._available[:, idx, :w]
    P = len(players)
    proj = np.moveaxis(proj, 1, -1).reshape(-1, P)
    avail = np.moveaxis(avail, 1, -1).reshape(-1, P)
    return proj, avail


def solve_completion_exact(
    problem: CompletionProblem,
    proxy: ProxyEvaluator,
    *,
    tolerance: float = CERTIFIED_ABS_TOLERANCE,
    time_limit_s: Optional[float] = None,
    objective: Optional[ProxyObjective] = None,
    forbid: Sequence[Sequence[int]] = (),
    spend_band: Optional[Tuple[Optional[int], Optional[int]]] = None,
    warm_start: Optional[Sequence[int]] = None,
) -> CertifiedResult:
    r"""The optimum as a single MILP, with no cutting plane at all.

    The cutting-plane solver above is correct but slow to converge, because a
    Nemhauser--Wolsey cut whose drop term must be valid for every ``T`` is a
    weak inequality: on a nine-man fixture with 27 legal completions it took
    around 400 cuts to close the gap. The reason it needed any cuts is that it
    treats ``f`` as a black box. It is not one.

    ``f`` is an average of max-weight independent sets in a matroid, and the
    matroid's constraint system is the polymatroid ``cap`` of
    :data:`LINEUP_CAP`. So the inner maximisation can be written **inside** the
    model, with a starter variable ``z[s, p]`` per scenario and player:

    .. math::

        \max\; \frac{1}{|S|}\sum_{s,p} \pi_{s,p} z_{s,p}
        \quad\text{s.t.}\quad
        0 \le z_{s,p} \le x_p a_{s,p},\;
        \sum_{p \in A} z_{s,p} \le \mathrm{cap}(A)\;\;\forall A .

    Because ``cap`` is a monotone submodular rank function, that polytope is
    the matroid polytope and is **integral**: for any fixed ``x`` the inner LP
    attains exactly the greedy value ``select_lineups_mask`` computes, with no
    integrality imposed on ``z``. The only binaries are the roster choices, and
    HiGHS reports its own dual bound, so the optimality gap is the solver's
    rather than something this module has to construct.

    The result is the same optimum the cutting plane finds, reached in one
    solve. ``tests/test_certified_solver.py`` checks both against exhaustive
    enumeration on the same fixtures, and checks they agree with each other.

    ``forbid`` supplies no-good cuts (used by the near-optimal walk),
    ``spend_band`` an explicit spending interval, and ``warm_start`` an initial
    integer solution.
    """
    t0 = time.perf_counter()
    if not reserve_is_implied(problem):
        raise ValueError(
            "a board cost below min_bid means the compact budget row does not "
            "imply the per-slot reserve; refusing rather than relaxing it")
    try:
        import highspy
    except Exception as exc:
        raise SolverUnavailable("highspy is not installed") from exc

    obj = objective if objective is not None else ProxyObjective(proxy)
    fixed = list(problem.fixed)
    board = list(problem.board)
    if problem.slots_to_fill == 0:
        v = obj.value(fixed)
        return CertifiedResult(tuple(sorted(fixed)), v, v, 0,
                               time.perf_counter() - t0, obj.n_evals,
                               "certified", problem.fingerprint())

    players = fixed + board
    m = len(board)
    P = len(players)
    proj, avail = _scenario_arrays(proxy, players)
    S = proj.shape[0]
    scale = 1.0 / float(S)

    inf = highspy.kHighsInf
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("mip_abs_gap", max(0.0, float(tolerance) * 0.5))
    h.setOptionValue("mip_rel_gap", 0.0)
    if time_limit_s is not None:
        h.setOptionValue("time_limit", max(1.0, float(time_limit_s)))

    # --- columns: x first, then z laid out scenario-major --------------------
    n_z = S * P
    n_cols = m + n_z
    lo = np.zeros(n_cols)
    hi = np.empty(n_cols)
    hi[:m] = 1.0
    # A player who is out this week can never start; fixing the bound is both
    # the correct model and the presolve's biggest single saving.
    hi[m:] = avail.reshape(-1).astype(np.float64)
    cost = np.zeros(n_cols)
    cost[m:] = -(proj.reshape(-1) * scale)      # HiGHS minimises
    h.addVars(n_cols, lo, hi)
    h.changeColsCost(n_cols, np.arange(n_cols, dtype=np.int32), cost)
    h.changeColsIntegrality(
        m, np.arange(m, dtype=np.int32),
        np.array([highspy.HighsVarType.kInteger] * m))

    xcols = np.arange(m, dtype=np.int32)

    # --- roster rows ---------------------------------------------------------
    K = float(problem.slots_to_fill)
    h.addRow(K, K, m, xcols, np.ones(m))
    costs_v = np.array([float(problem.cost[p]) for p in board])
    h.addRow(-inf, float(problem.budget), m, xcols, costs_v)
    if spend_band is not None:
        blo, bhi = spend_band
        h.addRow(-inf if blo is None else float(blo),
                 inf if bhi is None else float(bhi), m, xcols, costs_v)

    fc = problem.fixed_counts
    bpos = np.array([int(problem.position[p]) for p in board])
    for need, have, mask in ((1, fc.qb, bpos == int(Position.QB)),
                             (2, fc.rb, bpos == int(Position.RB)),
                             (3, fc.wt, bpos >= int(Position.WR)),
                             (6, fc.flex_eligible, bpos >= int(Position.RB))):
        short = need - have
        if short > 0:
            sel = np.flatnonzero(mask).astype(np.int32)
            h.addRow(float(short), inf, sel.size, sel, np.ones(sel.size))

    for sel_ids in forbid:
        sel = np.array([board.index(p) for p in sel_ids if p in set(board)],
                       dtype=np.int32)
        if sel.size:
            h.addRow(-inf, float(sel.size - 1), sel.size, sel,
                     np.ones(sel.size))

    # --- the lineup polytope, one block per scenario -------------------------
    ppos = np.array([int(problem.position[p]) for p in players])
    groups = {"Q": ppos == int(Position.QB),
              "R": ppos == int(Position.RB),
              "T": ppos >= int(Position.WR)}
    subsets = [s for s in LINEUP_CAP if s]
    r_lo: List[float] = []
    r_hi: List[float] = []
    r_start: List[int] = []
    r_idx: List[np.ndarray] = []
    nnz = 0
    for s in range(S):
        base = m + s * P
        for sub in subsets:
            mask = np.zeros(P, dtype=bool)
            for g in sub:
                mask |= groups[g]
            # Players who cannot start this week are already fixed to zero.
            mask &= avail[s]
            cols = np.flatnonzero(mask).astype(np.int64) + base
            if cols.size == 0:
                continue
            r_lo.append(-inf)
            r_hi.append(float(LINEUP_CAP[sub]))
            r_start.append(nnz)
            r_idx.append(cols)
            nnz += cols.size
    # --- linking rows: z[s, p] <= x_p for board players ----------------------
    board_off = len(fixed)
    for s in range(S):
        base = m + s * P
        rows = np.flatnonzero(avail[s, board_off:])
        for j in rows:
            r_lo.append(-inf)
            r_hi.append(0.0)
            r_start.append(nnz)
            r_idx.append(np.array([base + board_off + j, j], dtype=np.int64))
            nnz += 2

    n_rows = len(r_lo)
    if n_rows:
        idx_all = np.concatenate(r_idx).astype(np.int32)
        # A counting row is all ones; a linking row is (+1 on z, -1 on x).
        vals: List[np.ndarray] = []
        for cols in r_idx:
            if cols.size == 2 and cols[1] < m:
                vals.append(np.array([1.0, -1.0]))
            else:
                vals.append(np.ones(cols.size))
        val_all = np.concatenate(vals)
        h.addRows(n_rows, np.array(r_lo), np.array(r_hi), nnz,
                  np.array(r_start, dtype=np.int32), idx_all, val_all)

    if warm_start:
        ws = set(int(p) for p in warm_start)
        sol = np.zeros(n_cols)
        sol[:m] = [1.0 if p in ws else 0.0 for p in board]
        try:
            h.setSolution(highspy.HighsSolution())
        except Exception:
            pass

    h.run()
    status = h.modelStatusToString(h.getModelStatus())
    info = h.getInfo()
    colv = np.asarray(h.getSolution().col_value)
    runtime = time.perf_counter() - t0
    if status in ("Infeasible", "Primal infeasible") or colv.size == 0:
        return CertifiedResult((), float("-inf"), float("-inf"), 1, runtime,
                               obj.n_evals, "infeasible",
                               problem.fingerprint())
    picked = [p for j, p in enumerate(board) if colv[j] > 0.5]
    roster = sorted(fixed + picked)
    # The reported objective is ProxyEvaluator's, never the solver's: they
    # agree to floating point, and quoting the one the rest of the system uses
    # keeps every number in the report on the same scale.
    value = obj.value(roster)
    bound = -float(info.mip_dual_bound)
    legal, why = problem.is_legal(roster)
    if not legal:
        return CertifiedResult(tuple(roster), value, bound, 1, runtime,
                               obj.n_evals, f"illegal: {why}",
                               problem.fingerprint())
    certified = (status == "Optimal" or bound - value <= tolerance + 1e-9)
    return CertifiedResult(
        tuple(roster), value, max(bound, value), 1, runtime, obj.n_evals,
        "certified" if certified else "time_limit", problem.fingerprint())
