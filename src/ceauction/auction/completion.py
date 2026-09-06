"""Filling a roster to fifteen, and asking championship equity which way was best.

The best alternative to buying a player is a *whole roster*, not a list of
independently valued slots. This league's six non-QB starting slots interact:
three modest receivers who are healthy in different weeks can be worth more
than two good ones and a hole, and a fourth running back who can never start is
worth almost nothing however good he is. A greedy "take the highest projection
you can afford" walk cannot see any of that, because it commits before it knows
what it is committing to.

So the search is two-stage, and the stages answer different questions.

**Stage 1 generates candidates.** A beam search over the available board, in
descending projection order, with take/skip at each player. It is ranked by an
admissible bound -- points so far plus the best possible remainder ignoring
cost and position -- which never prunes a branch that could still lead to the
best roster *by that bound*. It carries many partial constructions rather than
one, which is what keeps aggregate combinations alive long enough to be
compared. Complete rosters are then re-ranked by
:mod:`ceauction.auction.proxy`, which does understand slot eligibility, byes,
injuries and contingency.

**Stage 2 asks championship equity.** The top finalists are built into real
twelve-team leagues against a fixed comparison cast and simulated. Common
random numbers make the finalists paired, so the difference between two of
them is measured on matched seasons rather than differenced from two noisy
means.

**The result is heuristic and says so.** A beam is not an optimum, and this
module never uses the word for one. When the pool is small enough, exhaustive
enumeration is available and is used as a testing oracle against the beam.
When two finalists' paired interval contains zero they are returned as an
unresolved set rather than ordered on noise.
"""

from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass, field, replace
from typing import (Dict, FrozenSet, List, Mapping, Optional, Sequence, Set,
                    Tuple)

import numpy as np

from ..ce import paired_se
from ..league import Position
from ..players import PlayerSpec
from ..roster import Roster, RosterSet
from ..simulate import simulate_seasons
from .costs import CostBook
from .feasibility import PositionCounts, can_complete, min_additions_for_lineup
from .proxy import ProxyEvaluator
from .state import AuctionState

__all__ = [
    "format_completion",
    "CompletionSettings",
    "Completion",
    "SearchDiagnostics",
    "CompletionResult",
    "ComparisonCast",
    "complete_roster",
    "enumerate_completions_exactly",
]


@dataclass(frozen=True)
class CompletionSettings:
    """Every knob the search has, and the budget it is allowed to spend."""

    beam_width: int = 300
    """Partial constructions carried at each step. Wider is closer to exact and
    slower; it is the main heuristic/exactness dial."""

    candidate_pool: int = 90
    """How deep down the board the search looks, by projection order.

    A bound, and a real one: a player below this cut cannot be chosen however
    cheap he is. Reported in the diagnostics so it cannot be forgotten."""

    max_candidates: int = 4_000
    """Complete rosters the beam retains at all."""

    spend_buckets: int = 8
    """Spending levels the beam keeps a slice of each.

    Diversification, not a preference. The bound saturates once eight decent
    starters exist, so without this the beam carries only the cheapest way to
    reach them and never compares a deeper, more expensive roster."""

    proxy_candidates: int = 600
    """Of those, how many are scored by the availability proxy.

    The beam's own ordering is by best-legal-eight under static projections,
    which is the cheap version of the proxy's question, so taking its top slice
    loses much less than scoring everything would cost."""

    finalists: int = 6
    """Distinct completions carried into CE evaluation."""

    proxy_reps: int = 96
    proxy_seed: int = 20260904

    selection_sims: int = 4_000
    """Seasons used to CHOOSE among finalists by championship equity."""

    selection_seed: int = 20260904

    evaluation_sims: int = 4_000
    """Seasons used to REPORT the winner's equity, on an independent sample.

    Choosing the maximum of several noisy estimates and then quoting that same
    estimate is upward-biased: the winner won partly because its sample was
    lucky. A separate seed makes the reported number an unbiased estimate of
    the chosen roster rather than a re-use of the sample that chose it."""

    evaluation_seed: int = 917_324_011
    """Deliberately unrelated to ``selection_seed``. The counter-based RNG makes
    two seeds independent streams, so this is a genuine holdout rather than a
    different slice of the same one."""

    ce_chunk: int = 64

    rival_selection: str = "ce"
    """``"ce"`` or ``"proxy"`` -- how a rival's continuation is chosen.

    ``"ce"`` optimises that rival's roster for HIS OWN championship equity,
    which is what an opponent would actually do. ``"proxy"`` stops at expected
    points and roughly halves the cost of a pass branch. Whichever is used is
    named in the result: a proxy-selected continuation may not be described as
    CE-optimised."""
    max_runtime_s: Optional[float] = None
    exact: bool = False
    """Enumerate exhaustively instead of beam-searching. Only viable for tiny
    pools; :attr:`exact_max_combinations` refuses rather than hanging."""
    exact_max_combinations: int = 400_000

    def __post_init__(self) -> None:
        for name in ("beam_width", "candidate_pool", "max_candidates",
                     "proxy_candidates", "spend_buckets", "finalists",
                     "proxy_reps", "selection_sims", "evaluation_sims"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.rival_selection not in ("ce", "proxy"):
            raise ValueError(
                f"unknown rival_selection {self.rival_selection!r}")
        if self.selection_seed == self.evaluation_seed:
            raise ValueError(
                "selection_seed and evaluation_seed must differ; reusing the "
                "sample that chose a winner to report its advantage is exactly "
                "the selection bias the holdout exists to remove")

    def cache_key(self) -> Tuple:
        """Every field, derived from the dataclass rather than listed by hand.

        The audit found the hand-written tuple silently omitting
        ``max_runtime_s`` and ``exact_max_combinations``, both of which can
        change what the search returns. Enumerating ``dataclasses.fields``
        means a field added later is covered the day it is added, which a
        hand-maintained list demonstrably was not.
        """
        import dataclasses
        return tuple((f.name, getattr(self, f.name))
                     for f in dataclasses.fields(self))

    @property
    def ce_sims(self) -> int:
        """Backwards-compatible alias for the selection sample size."""
        return self.selection_sims

    @property
    def ce_seed(self) -> int:
        return self.selection_seed

    def to_dict(self) -> Dict[str, object]:
        import dataclasses
        return {f.name: getattr(self, f.name)
                for f in dataclasses.fields(self)}


@dataclass(frozen=True)
class ComparisonCast:
    """The eleven other teams, held fixed while the focus roster is searched.

    Fixing them is a simplification with a name: it answers "which completion
    is best against *this* league" and not "which completion is best against a
    league that is also still drafting". The rivals are complete fifteen-man
    rosters, which is what the season simulator requires.
    """

    focus_team_index: int
    rosters: Tuple[Tuple[int, ...], ...]
    """Twelve entries; the focus one is a placeholder and is overwritten."""
    team_names: Tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.rosters) != len(self.team_names):
            raise ValueError("rosters and team_names must be the same length")
        if not 0 <= self.focus_team_index < len(self.rosters):
            raise ValueError("focus_team_index is out of range")
        others = [pid for i, t in enumerate(self.rosters)
                  if i != self.focus_team_index for pid in t]
        if len(set(others)) != len(others):
            raise ValueError("a rival player appears on two teams")

    @property
    def rival_ids(self) -> FrozenSet[int]:
        return frozenset(pid for i, t in enumerate(self.rosters)
                         if i != self.focus_team_index for pid in t)

    def reserved_for(self, team_index: int,
                     include_focus: bool = False) -> FrozenSet[int]:
        """Players some *other* team already holds, from this team's view.

        By default the focus slot is excluded whichever team is asking: it is a
        placeholder the search is about to overwrite, so treating its contents
        as taken would hide players from everyone.

        ``include_focus`` is for the one case where that is wrong -- choosing a
        rival's continuation while the focus slot holds a concrete provisional
        roster rather than a placeholder. Without it the rival's board and the
        focus roster overlap and the resulting league holds a player twice.
        """
        skip = {team_index}
        if not include_focus:
            skip.add(self.focus_team_index)
        return frozenset(pid for i, t in enumerate(self.rosters)
                         if i not in skip for pid in t)

    def fingerprint(self) -> str:
        """A digest of who is on which team.

        Roster *order* is not part of it: two casts differing only in the order
        players appear on a team are the same league.
        """
        import hashlib
        parts = [f"focus={self.focus_team_index}"]
        for name, team in zip(self.team_names, self.rosters):
            parts.append(f"{name}=" + ",".join(str(p) for p in sorted(team)))
        return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:16]

    def with_team(self, team_index: int,
                  player_ids: Sequence[int]) -> "ComparisonCast":
        return replace(self, rosters=tuple(
            tuple(player_ids) if i == team_index else t
            for i, t in enumerate(self.rosters)))

    def with_focus(self, player_ids: Sequence[int]) -> Tuple[Tuple[int, ...], ...]:
        return tuple(tuple(player_ids) if i == self.focus_team_index else t
                     for i, t in enumerate(self.rosters))


@dataclass(frozen=True)
class Completion:
    """One way of filling the roster, and how good it looks at each stage."""

    added: Tuple[int, ...]
    """Players bought to complete the roster, in board order."""
    roster: Tuple[int, ...]
    """All fifteen: the pre-owned players followed by ``added``."""
    added_cost: int
    proxy: float
    """Expected weekly starting projection. **Not value.** See
    :mod:`ceauction.auction.proxy`."""
    selection_ce: Optional[float] = None
    """Equity on the sample used to CHOOSE among finalists.

    Upward-biased for whichever finalist won, because the maximum of several
    noisy estimates partly measures which sample was kind. Reported for
    transparency; :attr:`ce` is the number to quote."""
    selection_ce_se: Optional[float] = None
    ce: Optional[float] = None
    """Equity on the independent holdout sample. Unbiased for this roster."""
    ce_se: Optional[float] = None
    counts: Optional[PositionCounts] = None

    @property
    def key(self) -> FrozenSet[int]:
        return frozenset(self.roster)

    def to_dict(self) -> Dict[str, object]:
        return {"added": list(self.added), "added_cost": self.added_cost,
                "roster_size": len(self.roster),
                "proxy": round(self.proxy, 4),
                "selection_ce": None if self.selection_ce is None
                else round(self.selection_ce, 6),
                "selection_ce_se": None if self.selection_ce_se is None
                else round(self.selection_ce_se, 6),
                "ce": None if self.ce is None else round(self.ce, 6),
                "ce_se": None if self.ce_se is None else round(self.ce_se, 6),
                "position_counts": self.counts.to_dict() if self.counts else None}


@dataclass
class SearchDiagnostics:
    """What the search did, so a reader can tell how much to trust it."""

    method: str = "beam"
    states_expanded: int = 0
    states_pruned_by_beam: int = 0
    states_rejected_infeasible: int = 0
    states_rejected_unaffordable: int = 0
    candidates_found: int = 0
    candidates_kept: int = 0
    retained_per_depth: Tuple[int, ...] = ()
    finalists_evaluated: int = 0
    enumeration_exact: bool = False
    """Was candidate generation exhaustive over the candidate pool?"""
    proxy_truncated: bool = False
    """Were candidates dropped before the proxy or before CE?"""
    ce_selection_exact: bool = False
    """Was EVERY feasible completion evaluated by championship equity?"""
    selection_sims: int = 0
    evaluation_sims: int = 0
    stop_reason: str = "search completed"
    proxy_seconds: float = 0.0
    ce_seconds: float = 0.0
    total_seconds: float = 0.0
    candidate_pool_size: int = 0
    board_size: int = 0
    cheap_fillers_added: int = 0
    """Players added to the pool purely so an affordable completion exists."""
    candidate_spend_levels: int = 0
    """Distinct total spends among the retained candidates.

    One means the beam collapsed onto a single spending level and the finalists
    are near-clones of each other, which makes a CE comparison between them
    almost meaningless."""
    finalist_spend_levels: int = 0

    @property
    def is_exact(self) -> bool:
        """Exact CE optimisation over every feasible completion.

        The audit found ``method == "exact"`` being reported as an exact
        result even when exhaustive enumeration was followed by truncation to
        ``proxy_candidates`` and CE evaluation of a handful of finalists. That
        is exact *enumeration* followed by a heuristic *choice*, which is not
        an exact answer to the question asked. All three stages must hold.
        """
        return (self.enumeration_exact and not self.proxy_truncated
                and self.ce_selection_exact)

    @property
    def ce_coverage(self) -> float:
        """Fraction of distinct feasible completions actually simulated."""
        if not self.candidates_kept:
            return 0.0
        return self.finalists_evaluated / float(self.candidates_kept)

    @property
    def exactness(self) -> Dict[str, object]:
        """Each stage's status, separately, because they differ."""
        return {
            "candidate_enumeration": "exhaustive" if self.enumeration_exact
            else "bounded beam",
            "proxy_ranking": "truncated" if self.proxy_truncated
            else "all candidates ranked",
            "ce_evaluated": f"{self.finalists_evaluated} of "
                            f"{self.candidates_kept} distinct completions",
            "ce_coverage": round(self.ce_coverage, 4),
            "ce_selection": "exhaustive" if self.ce_selection_exact
            else "finalists only",
            "overall": "exact" if self.is_exact else "heuristic",
        }

    def to_dict(self) -> Dict[str, object]:
        return {
            "method": self.method,
            "exact": self.is_exact,
            "result_is": "exact" if self.is_exact else "heuristic",
            "exactness": self.exactness,
            "selection_sims": self.selection_sims,
            "evaluation_sims": self.evaluation_sims,
            "states_expanded": self.states_expanded,
            "states_pruned_by_beam": self.states_pruned_by_beam,
            "states_rejected_infeasible": self.states_rejected_infeasible,
            "states_rejected_unaffordable": self.states_rejected_unaffordable,
            "candidates_found": self.candidates_found,
            "candidates_kept": self.candidates_kept,
            "retained_per_depth": list(self.retained_per_depth),
            "finalists_evaluated": self.finalists_evaluated,
            "stop_reason": self.stop_reason,
            "board_size": self.board_size,
            "candidate_pool_size": self.candidate_pool_size,
            "cheap_fillers_added": self.cheap_fillers_added,
            "candidate_spend_levels": self.candidate_spend_levels,
            "finalist_spend_levels": self.finalist_spend_levels,
            "proxy_seconds": round(self.proxy_seconds, 3),
            "ce_seconds": round(self.ce_seconds, 3),
            "total_seconds": round(self.total_seconds, 3),
        }


@dataclass
class CompletionResult:
    """The chosen completion, its rivals, and everything needed to doubt it."""

    best: Optional[Completion]
    finalists: Tuple[Completion, ...]
    diagnostics: SearchDiagnostics
    settings: CompletionSettings
    cost_level: str
    focus_owner_id: str
    ce_team_index: Optional[int] = None
    """Whose equity the finalists were chosen by. ``None`` when CE was skipped."""
    unresolved: Tuple[Completion, ...] = ()
    """Finalists whose CE is not distinguishable from the best one's.

    Non-empty means the search has a *set* of co-best completions and no
    evidence for ordering them. Picking one anyway would be inventing a
    preference out of Monte Carlo noise.
    """
    notes: str = ""

    @property
    def ce(self) -> Optional[float]:
        """Holdout equity of the chosen completion. ``None`` when CE was skipped."""
        return self.best.ce if self.best else None

    @property
    def selection_basis(self) -> str:
        """``"championship equity"`` or ``"expected-points proxy"``.

        The audit found a buy/pass comparison selecting completions by the
        proxy and then simulating only the winner, while its documentation said
        the comparison was between CE-best rosters. Every result now states
        which of the two actually chose it.
        """
        if self.best is None:
            return "none"
        return ("championship equity" if self.best.ce is not None
                else "expected-points proxy")

    @property
    def is_resolved(self) -> bool:
        """Is the winner distinguishable from every other finalist?"""
        return self.best is not None and not self.unresolved

    @property
    def result_kind(self) -> str:
        if self.best is None:
            return "no legal completion"
        if not self.diagnostics.is_exact:
            return "heuristic" if self.is_resolved else "heuristic, unresolved"
        return "exact" if self.is_resolved else "exact search, unresolved CE"

    def to_dict(self) -> Dict[str, object]:
        return {
            "focus_owner_id": self.focus_owner_id,
            "ce_team_index": self.ce_team_index,
            "selection_basis": self.selection_basis,
            "result_kind": self.result_kind,
            "resolved": self.is_resolved,
            "cost_level": self.cost_level,
            "best": self.best.to_dict() if self.best else None,
            "finalists": [c.to_dict() for c in self.finalists],
            "unresolved_with_best": [c.to_dict() for c in self.unresolved],
            "diagnostics": self.diagnostics.to_dict(),
            "settings": self.settings.to_dict(),
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Stage 1: candidate generation
# ---------------------------------------------------------------------------


def _best_eight(entries: Sequence[Tuple[float, int]]) -> float:
    """Value of the best legal eight from ``(projection, position_code)`` pairs.

    The same greedy the real optimiser uses, and correct for the same reason:
    the slot eligibility sets form a laminar family, so taking players in
    descending projection order and keeping any that still fit Hall's condition
    is exact. Availability is ignored here -- this is a static construction
    score, not a season.
    """
    q = r = t = 0
    total = 0.0
    for proj, pos in sorted(entries, key=lambda e: -e[0]):
        if pos == int(Position.QB):
            nq, nr, nt = q + 1, r, t
        elif pos == int(Position.RB):
            nq, nr, nt = q, r + 1, t
        else:
            nq, nr, nt = q, r, t + 1
        if (nq <= 2 and nr <= 4 and nt <= 5 and nq + nr <= 5 and nq + nt <= 6
                and nr + nt <= 7 and nq + nr + nt <= 8):
            q, r, t = nq, nr, nt
            total += proj
            if q + r + t == 8:
                break
    return total


@dataclass(frozen=True)
class _Partial:
    """A partial completion during the beam. Immutable so states never alias."""

    taken: Tuple[int, ...]
    spend: int
    counts: PositionCounts
    entries: Tuple[Tuple[float, int], ...]
    """``(projection, position)`` for owned plus taken players."""

    @property
    def value(self) -> float:
        """Best legal eight from what is held so far. Not a roster score."""
        return _best_eight(self.entries)


def _feasible_prefix(counts: PositionCounts, slots_left: int,
                     remaining_by_pos: Mapping[Position, int]) -> bool:
    return can_complete(counts, slots_left, remaining_by_pos)


def _prune_diverse(states: List[_Partial], width: int, budget: int,
                   n_buckets: int, score) -> List[_Partial]:
    """Keep the best ``width`` states, but spread across spending levels.

    Ranking on the bound alone collapses the beam onto one construction. The
    best legal eight saturates quickly -- once a roster can field eight decent
    starters, the next player barely moves it -- so every surviving state ends
    up being the *cheapest* way to reach that eight, and the search never
    carries an expensive-and-deep roster far enough to compare it. A smoke run
    made that concrete: every finalist spent $25 of an available $75 and they
    were near-identical.

    Bench depth is real and this scoring cannot see it, so rather than invent a
    weight for it, the beam keeps a slice of each spending level and lets the
    availability proxy -- which *can* see it -- decide between them later.
    Deterministic: buckets are fixed intervals and ties break on the state's
    own player ids.
    """
    if n_buckets <= 1 or budget <= 0:
        states.sort(key=lambda p: (-score(p), p.spend, p.taken))
        return states[:width]
    buckets: Dict[int, List[_Partial]] = {}
    for p in states:
        b = min(n_buckets - 1, p.spend * n_buckets // (budget + 1))
        buckets.setdefault(b, []).append(p)
    per = max(1, width // max(len(buckets), 1))
    kept: List[_Partial] = []
    leftovers: List[_Partial] = []
    for b in sorted(buckets):
        group = sorted(buckets[b], key=lambda p: (-score(p), p.spend, p.taken))
        kept.extend(group[:per])
        leftovers.extend(group[per:])
    if len(kept) < width and leftovers:
        leftovers.sort(key=lambda p: (-score(p), p.spend, p.taken))
        kept.extend(leftovers[: width - len(kept)])
    kept.sort(key=lambda p: (-score(p), p.spend, p.taken))
    return kept[:width]


def _beam_search(
    board: Sequence[PlayerSpec],
    costs: Mapping[int, int],
    start_counts: PositionCounts,
    start_entries: Sequence[Tuple[float, int]],
    slots_to_fill: int,
    budget: int,
    min_bid: int,
    settings: CompletionSettings,
    diag: SearchDiagnostics,
    deadline: Optional[float],
) -> List[_Partial]:
    """Take/skip beam over the board in descending projection order.

    **Ranking is slot-aware, and it has to be.** An earlier version ranked by
    the sum of every taken player's projection, which is not a roster score at
    all: it rated a sixth quarterback exactly as highly as his projection,
    although he can never start. The beam duly bought six quarterbacks and
    every finalist was the same degenerate roster. States are now ranked by the
    best legal eight they can field, so a player who cannot start contributes
    nothing until he can.

    The bound added to that is admissible. For each state it grants the
    remaining ``k`` slots the best ``k`` players *of every position* still on
    the board -- a superset of any real completion's additions -- and takes the
    best eight of the union. Best-eight is monotone under adding players, so no
    real completion can beat its own bound and pruning a branch whose bound is
    below the beam cannot discard the eventual winner *by this bound*. It can
    still discard the eventual winner by championship equity, which is why the
    result is labelled heuristic throughout.
    """
    n = len(board)
    entries = [(float(s.base_mean), int(s.position)) for s in board]

    # Per-position suffix lists, so the bound's phantom additions are the best
    # remaining players of each position rather than the best overall.
    by_pos_suffix: Dict[int, List[List[Tuple[float, int]]]] = {}
    for pos in (int(Position.QB), int(Position.RB), int(Position.WR),
                int(Position.TE)):
        idxs = [i for i in range(n) if entries[i][1] == pos]
        suffix: List[List[Tuple[float, int]]] = [[] for _ in range(n + 1)]
        acc: List[Tuple[float, int]] = []
        for i in range(n - 1, -1, -1):
            if entries[i][1] == pos:
                acc = [entries[i]] + acc
            suffix[i] = acc
        by_pos_suffix[pos] = suffix

    def bound(p: _Partial, i: int) -> float:
        k = slots_to_fill - len(p.taken)
        if k <= 0:
            return p.value
        phantom: List[Tuple[float, int]] = []
        for pos, suffix in by_pos_suffix.items():
            phantom.extend(suffix[i][:k])
        return _best_eight(list(p.entries) + phantom)

    # Cheapest possible completion from board[i:], for an affordability prune.
    sorted_costs_suffix: List[List[int]] = []
    tail: List[int] = []
    for i in range(n - 1, -1, -1):
        tail.append(costs[board[i].player_id])
        sorted_costs_suffix.append(sorted(tail))
    sorted_costs_suffix.reverse()

    def cheapest(i: int, k: int) -> int:
        if k <= 0:
            return 0
        if i >= n:
            return 1 << 30
        tailcosts = sorted_costs_suffix[i]
        if k > len(tailcosts):
            return 1 << 30
        return sum(tailcosts[:k])

    base = tuple(start_entries)
    beam: List[_Partial] = [_Partial((), 0, start_counts, base)]
    complete: List[_Partial] = []
    retained: List[int] = []

    for i, spec in enumerate(board):
        if deadline is not None and time.perf_counter() > deadline:
            diag.stop_reason = f"runtime budget reached at board index {i}"
            break
        pid = spec.player_id
        price = costs[pid]
        pos = Position(int(spec.position))
        nxt: List[_Partial] = []
        for p in beam:
            diag.states_expanded += 1
            k_after_skip = slots_to_fill - len(p.taken)
            # --- skip -----------------------------------------------------
            if cheapest(i + 1, k_after_skip) + p.spend <= budget:
                nxt.append(p)
            else:
                diag.states_rejected_unaffordable += 1
            # --- take -----------------------------------------------------
            if len(p.taken) >= slots_to_fill:
                continue
            if p.spend + price > budget:
                diag.states_rejected_unaffordable += 1
                continue
            k_after_take = slots_to_fill - len(p.taken) - 1
            if p.spend + price + k_after_take * min_bid > budget:
                diag.states_rejected_unaffordable += 1
                continue
            new_counts = p.counts.plus(pos)
            need, _ = min_additions_for_lineup(new_counts)
            if need > k_after_take:
                diag.states_rejected_infeasible += 1
                continue
            if p.spend + price + cheapest(i + 1, k_after_take) > budget:
                diag.states_rejected_unaffordable += 1
                continue
            child = _Partial(p.taken + (pid,), p.spend + price, new_counts,
                             p.entries + (entries[i],))
            if len(child.taken) == slots_to_fill:
                complete.append(child)
            else:
                nxt.append(child)

        if len(nxt) > settings.beam_width:
            before = len(nxt)
            nxt = _prune_diverse(nxt, settings.beam_width, budget,
                                 settings.spend_buckets,
                                 lambda p: bound(p, i + 1))
            diag.states_pruned_by_beam += before - len(nxt)
        beam = nxt
        retained.append(len(beam))
        # Keep the candidate list bounded WITHOUT stopping the walk: cutting
        # the search short at a board index would make everyone cheaper than
        # that index unreachable, which is a different and much worse bound
        # than keeping the strongest candidates found so far.
        if len(complete) > settings.max_candidates * 2:
            complete = _prune_diverse(complete, settings.max_candidates, budget,
                                      settings.spend_buckets,
                                      lambda p: p.value)

    if slots_to_fill == 0:
        complete = [_Partial((), 0, start_counts, base)]

    if len(complete) > settings.max_candidates:
        complete = _prune_diverse(complete, settings.max_candidates, budget,
                                  settings.spend_buckets, lambda p: p.value)

    diag.retained_per_depth = tuple(retained)
    diag.candidates_found = len(complete)
    return complete


def enumerate_completions_exactly(
    board: Sequence[PlayerSpec],
    costs: Mapping[int, int],
    start_counts: PositionCounts,
    slots_to_fill: int,
    budget: int,
    max_combinations: int = 400_000,
) -> List[Tuple[Tuple[int, ...], int]]:
    """Every legal completion, by brute force. A testing oracle, not a solver.

    Refuses rather than hanging when the combination count would be absurd, so
    a caller cannot accidentally ask for an exact answer to a real board.
    """
    n = len(board)
    if slots_to_fill == 0:
        return [((), 0)]
    if slots_to_fill > n:
        return []
    total = math.comb(n, slots_to_fill)
    if total > max_combinations:
        raise ValueError(
            f"exact enumeration would visit {total:,} combinations, above the "
            f"{max_combinations:,} limit; use the beam search and label the "
            f"result heuristic")
    out: List[Tuple[Tuple[int, ...], int]] = []
    for combo in itertools.combinations(range(n), slots_to_fill):
        spend = sum(costs[board[i].player_id] for i in combo)
        if spend > budget:
            continue
        counts = start_counts
        for i in combo:
            counts = counts.plus(Position(int(board[i].position)))
        if min_additions_for_lineup(counts)[0] > 0:
            continue
        out.append((tuple(board[i].player_id for i in combo), spend))
    return out


# ---------------------------------------------------------------------------
# The search
# ---------------------------------------------------------------------------


def complete_roster(
    state: AuctionState,
    cast: ComparisonCast,
    costs: CostBook,
    *,
    settings: CompletionSettings = CompletionSettings(),
    owner_id: Optional[str] = None,
    evaluate_ce: bool = True,
    default_cost: Optional[int] = None,
    proxy: Optional[ProxyEvaluator] = None,
    reserved_ids: Optional[FrozenSet[int]] = None,
    ce_team_index: Optional[int] = None,
    notes: str = "",
) -> CompletionResult:
    """Fill one owner's roster to fifteen and rank the ways of doing it.

    ``default_cost`` is the only route to a fallback price and passing it is an
    explicit statement about unpriced players; without it a gap raises.

    ``reserved_ids`` are players the board may not offer because some other
    team is already assumed to hold them. It defaults to the cast's rivals,
    which is right when completing the focus owner; completing a *rival*
    (as the pass branch does) has to pass the correct set for that team
    instead, or the two teams would be offered the same player.

    ``ce_team_index`` says whose championship equity is being maximised. It
    defaults to the focus slot; a rival continuation must pass that rival's own
    slot, or the search would be choosing his roster to help *us*.
    """
    t0 = time.perf_counter()
    deadline = (t0 + settings.max_runtime_s) if settings.max_runtime_s else None
    owner_id = owner_id or state.focus_owner_id
    owner = state.owner(owner_id)
    diag = SearchDiagnostics(method="exact" if settings.exact else "beam")

    # --- what is buyable ---------------------------------------------------
    reserved = cast.rival_ids if reserved_ids is None else frozenset(reserved_ids)
    board_all = [s for s in state.available_specs if s.player_id not in reserved]
    board_all.sort(key=lambda s: (-s.base_mean, s.player_id))
    diag.board_size = len(board_all)

    spec_of = state.spec_by_id
    slots = owner.open_slots
    budget = owner.budget_remaining
    price_all = costs.costs_for([s.player_id for s in board_all], default_cost)

    # The top of the board by projection, PLUS cheap fillers at every position.
    #
    # Taking only the top N is what a projection-ranked cut naturally does, and
    # it is wrong: the most expensive players on the board cannot fill twelve
    # slots on any budget, so a pool of only good players can contain no legal
    # completion at all. A smoke run hit exactly that and reported "no legal
    # completion" for a roster that plainly had several. The cheap tail is not
    # an optimisation, it is what makes the search complete.
    board = list(board_all[: settings.candidate_pool])
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
    diag.candidate_pool_size = len(board)
    diag.cheap_fillers_added = len(board) - min(settings.candidate_pool,
                                                len(board_all))

    price_map = {s.player_id: price_all[s.player_id] for s in board}

    if slots > 0 and len(board) < slots:
        diag.stop_reason = (f"only {len(board)} players on the board but "
                            f"{slots} slots to fill")
        diag.total_seconds = time.perf_counter() - t0
        return CompletionResult(None, (), diag, settings, costs.level, owner_id,
                                notes=notes)

    # --- stage 1 -----------------------------------------------------------
    if settings.exact:
        combos = enumerate_completions_exactly(
            board, price_map, owner.counts, slots, budget,
            settings.exact_max_combinations)
        diag.candidates_found = len(combos)
        raw = [(ids, spend) for ids, spend in combos]
    else:
        start_entries = [(float(spec_of[p].base_mean), int(spec_of[p].position))
                         for p in owner.player_ids]
        partials = _beam_search(board, price_map, owner.counts, start_entries,
                                slots, budget, owner.min_bid, settings, diag,
                                deadline)
        partials = _prune_diverse(partials, settings.proxy_candidates, budget,
                                  settings.spend_buckets, lambda p: p.value)
        raw = [(p.taken, p.spend) for p in partials]

    # Distinct rosters only: two orders of the same players are one candidate.
    seen: Set[FrozenSet[int]] = set()
    unique: List[Tuple[Tuple[int, ...], int]] = []
    owned = tuple(owner.player_ids)
    for ids, spend in raw:
        k = frozenset(owned + ids)
        if k in seen:
            continue
        seen.add(k)
        unique.append((ids, spend))
    # Only the strongest constructions are worth the proxy's time; the beam has
    # already ordered them by best-legal-eight, which is the cheap version of
    # the same question.
    if len(unique) > settings.proxy_candidates:
        unique = unique[: settings.proxy_candidates]
    diag.candidates_kept = len(unique)

    if not unique:
        diag.stop_reason = diag.stop_reason if diag.stop_reason != "search completed" \
            else "no legal completion fits the budget and the lineup rules"
        diag.total_seconds = time.perf_counter() - t0
        return CompletionResult(None, (), diag, settings, costs.level, owner_id,
                                notes=notes)

    # --- proxy re-rank -----------------------------------------------------
    tp = time.perf_counter()
    if proxy is None:
        proxy = ProxyEvaluator(state.pool, state.settings, settings.proxy_reps,
                               settings.proxy_seed)
    rosters = [owned + ids for ids, _ in unique]
    scores = proxy.strength_many(rosters)
    diag.proxy_seconds = time.perf_counter() - tp

    order = sorted(range(len(unique)),
                   key=lambda i: (-scores[i], unique[i][1], unique[i][0]))
    spec_by_id = state.spec_by_id
    candidates = [
        Completion(
            added=unique[i][0], roster=rosters[i], added_cost=unique[i][1],
            proxy=float(scores[i]),
            counts=PositionCounts.from_positions(
                spec_by_id[p].position for p in rosters[i]))
        for i in order]

    diag.candidate_spend_levels = len({c.added_cost for c in candidates})
    finalists = _select_finalists(candidates, settings.finalists,
                                  settings.spend_buckets)
    diag.finalist_spend_levels = len({c.added_cost for c in finalists})
    diag.enumeration_exact = settings.exact
    diag.proxy_truncated = (len(candidates) > len(finalists)
                            or diag.candidates_kept < diag.candidates_found)
    if not evaluate_ce:
        diag.total_seconds = time.perf_counter() - t0
        return CompletionResult(finalists[0], tuple(finalists), diag, settings,
                                costs.level, owner_id, ce_team_index=None,
                                notes=(notes + " PROXY-SELECTED: no championship "
                                       "equity was evaluated, so the chosen "
                                       "roster is the best by expected points "
                                       "and may not be the best by equity."
                                       ).strip())

    # --- stage 3: choose by championship equity, on the selection sample ----
    tc = time.perf_counter()
    ce_index = cast.focus_team_index if ce_team_index is None else ce_team_index
    sel_ind, selected = [], []
    for cand in finalists:
        rs = _build_roster_set(state, cast, cand.roster, ce_team_index=ce_index,
                               owner_id=owner_id)
        out = simulate_seasons(rs, settings.selection_sims,
                               settings.selection_seed, settings.ce_chunk)
        ind = out.champion_indicator(ce_index)
        sel_ind.append(ind)
        selected.append(replace(
            cand, selection_ce=float(ind.mean()),
            selection_ce_se=float(np.std(ind, ddof=1) / math.sqrt(len(ind)))))

    best_i = max(range(len(selected)),
                 key=lambda i: (selected[i].selection_ce, selected[i].proxy))

    # Co-best: paired against the winner on the SAME selection seasons, so a
    # shared player contributes an exact zero and the interval is as tight as
    # the design allows.
    unresolved_idx = []
    for i in range(len(selected)):
        if i == best_i:
            continue
        d = sel_ind[best_i] - sel_ind[i]
        se = paired_se(d)
        if math.isnan(se) or abs(d.mean()) < 1.96 * se:
            unresolved_idx.append(i)

    # --- stage 4: report the winner on an INDEPENDENT holdout sample --------
    #
    # The maximum of several noisy estimates is biased upward: the winner won
    # partly because its sample was kind to it. Quoting that same sample would
    # carry the bias into every price and every reservation frontier built on
    # it. A second, unrelated seed gives an unbiased estimate of the roster
    # that was chosen.
    evaluated = list(selected)
    for i in ({best_i} | set(unresolved_idx)):
        rs = _build_roster_set(state, cast, selected[i].roster,
                               ce_team_index=ce_index, owner_id=owner_id)
        out = simulate_seasons(rs, settings.evaluation_sims,
                               settings.evaluation_seed, settings.ce_chunk)
        ind = out.champion_indicator(ce_index)
        evaluated[i] = replace(
            selected[i], ce=float(ind.mean()),
            ce_se=float(np.std(ind, ddof=1) / math.sqrt(len(ind))))

    diag.ce_seconds = time.perf_counter() - tc
    diag.finalists_evaluated = len(selected)
    diag.selection_sims = settings.selection_sims
    diag.evaluation_sims = settings.evaluation_sims
    diag.ce_selection_exact = (len(selected) == len(unique)
                               and not diag.proxy_truncated)

    best = evaluated[best_i]
    unresolved = tuple(evaluated[i] for i in unresolved_idx)
    ordered = sorted(evaluated,
                     key=lambda c: (-(c.selection_ce or 0.0), -c.proxy))
    diag.total_seconds = time.perf_counter() - t0
    return CompletionResult(best, tuple(ordered), diag, settings, costs.level,
                            owner_id, unresolved=unresolved, notes=notes,
                            ce_team_index=ce_index)


def _select_finalists(candidates: Sequence[Completion], k: int,
                      n_buckets: int) -> List[Completion]:
    """The best candidate, then the best from other spending levels.

    Taking the top *k* by proxy alone returns near-clones: the strongest
    rosters differ by one interchangeable bench player and spend the same
    money, so simulating five of them answers nothing. A runner-up is only
    useful if it is a genuinely different construction, which in this search
    means a different level of commitment.

    The leader is always included, so diversification can never cost the best
    candidate its place.
    """
    if not candidates or k <= 0:
        return []
    chosen = [candidates[0]]
    seen_buckets = {candidates[0].added_cost}
    spends = [c.added_cost for c in candidates]
    lo, hi = min(spends), max(spends)
    span = max(hi - lo, 1)

    def bucket(c: Completion) -> int:
        return min(n_buckets - 1, (c.added_cost - lo) * n_buckets // span)

    used = {bucket(candidates[0])}
    for c in candidates[1:]:
        if len(chosen) >= k:
            break
        b = bucket(c)
        if b not in used:
            chosen.append(c)
            used.add(b)
    for c in candidates[1:]:
        if len(chosen) >= k:
            break
        if all(c.key != x.key for x in chosen):
            chosen.append(c)
    return chosen


def _build_roster_set(state: AuctionState, cast: ComparisonCast,
                      focus_roster: Sequence[int],
                      ce_team_index: Optional[int] = None,
                      owner_id: Optional[str] = None) -> RosterSet:
    """A real twelve-team league with this completion in one team's slot.

    ``ce_team_index`` selects which slot the completion goes into, so a rival
    continuation can be optimised for that rival's own equity rather than for
    the focus team's.
    """
    slot = cast.focus_team_index if ce_team_index is None else ce_team_index
    assignment = cast.with_team(slot, focus_roster).rosters
    spec_by_id = state.spec_by_id
    used = [spec_by_id[pid] for team in assignment for pid in team]
    rosters = tuple(Roster(cast.team_names[i], tuple(team))
                    for i, team in enumerate(assignment))
    return RosterSet(tuple(used), rosters, state.settings)


def format_completion(result: CompletionResult, width: int = 88,
                      cost_disclaimer: str = "") -> str:
    """Sanitized rendering: ids, counts and diagnostics; no names, no prices."""
    bar = "=" * width
    d = result.diagnostics
    out = [bar, f"ROSTER COMPLETION -- {result.focus_owner_id}", bar,
           f"result        {result.result_kind.upper()}",
           f"method        {d.method} ({'exact' if d.is_exact else 'heuristic'})",
           f"cost source   {result.cost_level}"]
    if cost_disclaimer:
        out.append(f"              {cost_disclaimer}")
    out.append("")
    if result.best is None:
        out += ["No legal completion was found.",
                f"reason: {d.stop_reason}", bar]
        return "\n".join(out)

    head = (f"  {'rank':<5}{'added':>6}{'cost':>7}{'proxy':>10}"
            f"{'CE':>10}{'+/-':>9}  {'QB/RB/WR/TE':<14}")
    out += ["FINALISTS", head, "  " + "-" * (len(head) - 2)]
    for i, c in enumerate(result.finalists):
        cnt = c.counts
        ce = "n/a" if c.ce is None else f"{c.ce:.4f}"
        se = "" if c.ce_se is None else f"{1.96 * c.ce_se:.4f}"
        mark = "*" if result.best and c.key == result.best.key else " "
        unres = " (co-best)" if any(u.key == c.key for u in result.unresolved) else ""
        out.append(f" {mark}{i + 1:<4}{len(c.added):>6}{c.added_cost:>7}"
                   f"{c.proxy:>10.3f}{ce:>10}{se:>9}  "
                   f"{cnt.qb}/{cnt.rb}/{cnt.wr}/{cnt.te:<10}{unres}")
    out += ["  " + "-" * (len(head) - 2), ""]
    out += ["  proxy = expected weekly starting projection under this roster's",
            "  own byes and injuries. It is EXPECTED POINTS, not championship",
            "  equity, and it exists only to choose which rosters to simulate.", ""]

    if result.unresolved:
        out += [f"UNRESOLVED: {len(result.unresolved)} finalist(s) are not "
                f"distinguishable from the leader at",
                f"{result.settings.ce_sims:,} seasons. They are returned as "
                f"co-best rather than ordered on noise.", ""]

    out += ["SEARCH", f"  board                {d.board_size} available, "
            f"{d.candidate_pool_size} considered",
            f"  states expanded      {d.states_expanded:,}",
            f"  pruned by beam       {d.states_pruned_by_beam:,}",
            f"  rejected infeasible  {d.states_rejected_infeasible:,}",
            f"  rejected unaffordable {d.states_rejected_unaffordable:,}",
            f"  candidates found     {d.candidates_found:,} "
            f"({d.candidates_kept:,} distinct)",
            f"  finalists simulated  {d.finalists_evaluated}",
            f"  stop reason          {d.stop_reason}",
            f"  runtime              {d.total_seconds:.2f}s "
            f"(proxy {d.proxy_seconds:.2f}s, CE {d.ce_seconds:.2f}s)", ""]
    if not d.is_exact:
        out += ["  This is a BOUNDED SEARCH. It is not a proof of optimality and",
                "  the best completion found may not be the best that exists.", ""]
    out.append(bar)
    return "\n".join(out)
