"""Estimator power by candidate tier: what is worth simulating, and what is not.

The controlled context experiment found no distinguishable effect for one weak
marginal candidate. That was the right answer to the question asked and the
wrong basis for a blanket verdict: it showed the estimator refusing false
precision on a player whose true effect is tiny, not the estimator failing.

**Weak-player irrelevance and estimator failure look identical in one cell and
are opposite findings.** A bench player who genuinely moves championship equity
by 0.0005 *should* return unresolved at any affordable sample; spending 30,000
seasons to prove he is worth nothing is the waste, not the null. What must be
established separately is whether a player who genuinely matters -- a strong
starter, an elite piece -- produces an effect the estimator can resolve at a
practical cost. This module measures that.

Two things are kept strictly apart, because collapsing them is how a sampling
plan starts lying:

**Season-simulation uncertainty** is the paired standard error inside one
shared-board allocation. It shrinks as ``1/sqrt(n)`` and more seasons fix it.

**Allocation uncertainty** is the spread of the point estimate *across*
plausible shared-board continuations. More seasons do not shrink it at all. If
it dominates, a tighter confidence interval is a more precise answer to a
question nobody asked, and the two are reported separately and never pooled
into one ordinary paired SE.

Sampling is pilot-then-confirmatory. The pilot estimates variance; the required
confirmatory size is computed from it; an independent confirmatory sample is
drawn only if that size is under the cap. Reusing the pilot -- inspecting a
sample, deciding to continue, and then quoting a fixed-sample interval over the
whole thing -- is exactly the optional-stopping error that makes an interval
narrower than its own coverage.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence, Tuple

from ..auction.completion import CompletionSettings
from ..auction.proxy import ProxyEvaluator
from ..league import Position
from ..players import PlayerSpec
from .board import BoardSettings
from .context import build_context_regime
from .demo import TacticalDemo

__all__ = [
    "CandidateTier",
    "CANDIDATE_TIERS",
    "TierCalibration",
    "build_tiered_world",
    "calibrate_tier",
    "PowerPlan",
    "required_sample",
    "plan_confirmatory",
    "AllocationSpread",
    "allocation_spread",
    "TIER_PRICE_LADDERS",
    "HALF_WIDTH_TARGETS",
]


@dataclass(frozen=True)
class CandidateTier:
    """One stated candidate strength. The scale is the ONLY thing that varies."""

    name: str
    description: str
    scale: float
    """Multiplier on the candidate's ``base_mean``. Nothing else about him --
    position, bye, injury hazard, availability, correlation -- is touched, so a
    tier cannot smuggle in a durability or usage assumption."""

    def to_dict(self) -> Dict[str, object]:
        return dataclasses.asdict(self)


#: Calibrated against measured lineup consequences; see ``docs/TACTICAL_POWER.md``.
CANDIDATE_TIERS: Dict[str, CandidateTier] = {t.name: t for t in (
    CandidateTier("bench", "rarely reaches the expected starting lineup", 0.55),
    CandidateTier("marginal_starter", "sits on the lineup boundary", 1.15),
    CandidateTier("strong_starter", "clearly displaces an existing starter", 1.85),
    CandidateTier("elite", "large weekly starting-lineup improvement", 2.90),
)}

#: Experimental ladders, tier-appropriate. **Not market predictions**: nothing
#: here claims this room would pay these prices, only that these are the prices
#: worth testing for a player of this strength.
TIER_PRICE_LADDERS: Dict[str, Tuple[int, ...]] = {
    "bench": (1, 3, 5, 10),
    "marginal_starter": (1, 10, 20, 30),
    "strong_starter": (10, 20, 35, 50),
    "elite": (20, 40, 60, 80),
}

#: Reporting choices for the power table. **Not** claims about what size of CE
#: change is economically material -- that judgement is the reader's.
HALF_WIDTH_TARGETS: Tuple[float, ...] = (0.010, 0.005, 0.0025)


def build_tiered_world(context: str, tier: str, *,
                       with_sales: bool = True) -> Tuple[TacticalDemo, int]:
    """A controlled regime whose best available candidate is scaled to ``tier``.

    Returns the world and the candidate id. The candidate keeps his id,
    position, bye week, injury hazard and week-to-week variance; only his
    projected level moves, which is the single factor under test.
    """
    if tier not in CANDIDATE_TIERS:
        raise ValueError(
            f"unknown candidate tier {tier!r}; "
            f"known: {', '.join(sorted(CANDIDATE_TIERS))}")
    d = build_context_regime(context, with_sales=with_sales)
    cid = d.candidate().player_id
    scale = CANDIDATE_TIERS[tier].scale
    pool = tuple(
        replace(s, base_mean=round(s.base_mean * scale, 6),
                data_source=f"{s.data_source}|tier:{tier}")
        if s.player_id == cid else s
        for s in d.state.pool)
    state = replace(d.state, pool=pool)
    state.validate()
    return d.with_state(state), cid


@dataclass(frozen=True)
class TierCalibration:
    """What the candidate actually does to a starting lineup."""

    tier: str
    context: str
    candidate_id: int
    position: str
    candidate_weekly: float
    """The candidate's own expected weekly starting contribution."""
    replacement_weekly: float
    """The same roster with a minimum-priced filler in his place."""
    lineup_improvement: float
    """Weekly starting-lineup points gained. The number that decides the tier."""
    displaced_player_id: Optional[int]
    displaced_weekly: float
    starts: bool
    base_mean: float

    def to_dict(self) -> Dict[str, object]:
        return {k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in dataclasses.asdict(self).items()}


def calibrate_tier(d: TacticalDemo, cid: int, tier: str, context: str, *,
                   proxy: Optional[ProxyEvaluator] = None,
                   n_reps: int = 96, seed: int = 7) -> TierCalibration:
    """Measure the candidate's weekly lineup effect in a fixed roster context.

    Deliberately not a CE measurement: it holds fourteen roster slots fixed and
    swaps only the fifteenth, so the number is the candidate's own marginal
    contribution to the expected starting lineup rather than a search result.
    The filler is the cheapest legal board player, which is what "replacement"
    means in an auction with a $1 minimum.

    The comparison uses the availability proxy, whose lineups are chosen from
    **pregame** information only -- the same information barrier the season
    simulator enforces. A player is startable here only if he could have been
    identified as startable before lineup lock.
    """
    st = d.state
    focus = st.focus_owner_id
    owned = list(st.owner(focus).player_ids)
    if proxy is None:
        proxy = ProxyEvaluator(st.pool, st.settings, n_reps, seed)
    spec_of = st.spec_by_id
    cand = spec_of[cid]

    # Fill the other fourteen slots with the BEST players still on the board,
    # not the cheapest. A candidate only rides the bench if the lineup he is
    # competing with is real; filling around him with replacement-level players
    # would make every tier a starter and destroy the distinction being tested.
    strong = sorted((s for s in st.available_specs if s.player_id != cid),
                    key=lambda s: (-s.base_mean, s.player_id))
    cheap = sorted((s for s in st.available_specs if s.player_id != cid),
                   key=lambda s: (s.base_mean, s.player_id))
    need = st.owner(focus).roster_capacity - len(owned) - 1
    fillers = [s.player_id for s in strong[:need]]
    # The fifteenth slot's alternative is a genuine $1 replacement, which is
    # what "replacement level" means in an auction with a minimum bid.
    replacement = next(s.player_id for s in cheap
                       if s.player_id not in fillers)
    with_cand = owned + fillers + [cid]
    with_repl = owned + fillers + [replacement]
    a = proxy.strength(with_cand)
    b = proxy.strength(with_repl)

    # Who the candidate displaces: the roster member whose own removal costs
    # least once the candidate is present.
    displaced, displaced_weekly = None, 0.0
    if a - b > 1e-9:
        best_gap = None
        for pid in fillers + owned:
            trimmed = [p for p in with_cand if p != pid]
            gap = a - proxy.strength(trimmed + [replacement])
            if best_gap is None or gap > best_gap:
                best_gap, displaced = gap, pid
        if displaced is not None:
            displaced_weekly = float(spec_of[displaced].base_mean)
    return TierCalibration(
        tier=tier, context=context, candidate_id=cid,
        position=Position(int(cand.position)).name,
        candidate_weekly=a, replacement_weekly=b, lineup_improvement=a - b,
        displaced_player_id=displaced, displaced_weekly=displaced_weekly,
        starts=(a - b) > 0.25, base_mean=float(cand.base_mean))


# ---------------------------------------------------------------------------
# Pilot / confirmatory power
# ---------------------------------------------------------------------------


def required_sample(pilot_se: float, pilot_n: int,
                    target_half_width: float) -> int:
    """Seasons needed for a 95% half-width of ``target_half_width``.

    A paired standard error scales as ``sigma / sqrt(n)``, so the pilot pins
    ``sigma`` and the rest is arithmetic:

        n_required = n_pilot * (1.96 * se_pilot / target) ** 2

    Returns a whole number of seasons, never below the pilot size, because a
    confirmatory run smaller than the pilot that measured it is not a
    confirmation.
    """
    if pilot_n < 2:
        raise ValueError("a pilot needs at least two seasons")
    if target_half_width <= 0:
        raise ValueError("the target half-width must be positive")
    if not math.isfinite(pilot_se) or pilot_se <= 0.0:
        return pilot_n
    ratio = (1.96 * pilot_se) / target_half_width
    return max(pilot_n, int(math.ceil(pilot_n * ratio * ratio)))


@dataclass(frozen=True)
class PowerPlan:
    """A pilot, what it implies, and whether the confirmation is affordable."""

    pilot_n: int
    pilot_seed: int
    pilot_delta: float
    pilot_se: float
    target_half_width: float
    required_n: int
    cap_n: int
    confirmatory_seed: int
    status: str
    """``READY`` or ``UNDERPOWERED_AT_CAP``."""
    requirements: Dict[str, int] = dataclasses.field(default_factory=dict)
    """Required seasons for every reporting target, whether affordable or not."""

    @property
    def affordable(self) -> bool:
        return self.required_n <= self.cap_n

    def to_dict(self) -> Dict[str, object]:
        return {
            "pilot_n": self.pilot_n, "pilot_seed": self.pilot_seed,
            "pilot_delta": round(self.pilot_delta, 6),
            "pilot_se": round(self.pilot_se, 6),
            "target_half_width": self.target_half_width,
            "required_n": self.required_n, "cap_n": self.cap_n,
            "confirmatory_seed": self.confirmatory_seed,
            "affordable": self.affordable, "status": self.status,
            "required_by_target": dict(self.requirements),
            "note": ("pilot and confirmatory samples use different seeds; the "
                     "confirmatory interval reuses no pilot observation"),
        }


def plan_confirmatory(pilot_delta: float, pilot_se: float, pilot_n: int, *,
                      target_half_width: float = 0.005,
                      cap_n: int = 40_000,
                      pilot_seed: int = 0,
                      confirmatory_seed: int = 0,
                      targets: Sequence[float] = HALF_WIDTH_TARGETS
                      ) -> PowerPlan:
    """Decide whether a confirmatory run is worth drawing, before drawing it."""
    if pilot_seed == confirmatory_seed:
        raise ValueError(
            "the pilot and confirmatory samples must use different seeds; "
            "reusing the pilot's seasons in the interval that judges them is "
            "the optional-stopping error this design exists to avoid")
    need = required_sample(pilot_se, pilot_n, target_half_width)
    return PowerPlan(
        pilot_n=pilot_n, pilot_seed=pilot_seed, pilot_delta=pilot_delta,
        pilot_se=pilot_se, target_half_width=target_half_width,
        required_n=need, cap_n=cap_n, confirmatory_seed=confirmatory_seed,
        status="READY" if need <= cap_n else "UNDERPOWERED_AT_CAP",
        requirements={str(t): required_sample(pilot_se, pilot_n, t)
                      for t in targets})


# ---------------------------------------------------------------------------
# Allocation uncertainty
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AllocationSpread:
    """Between-allocation variation, kept apart from within-allocation SE."""

    seeds: Tuple[int, ...]
    deltas: Tuple[float, ...]
    paired_ses: Tuple[float, ...]
    fingerprints: Tuple[str, ...]

    @property
    def mean_delta(self) -> float:
        return sum(self.deltas) / len(self.deltas)

    @property
    def between_sd(self) -> float:
        """Spread of the point estimate ACROSS allocations. Seasons cannot shrink it."""
        n = len(self.deltas)
        if n < 2:
            return 0.0
        m = self.mean_delta
        return math.sqrt(sum((d - m) ** 2 for d in self.deltas) / (n - 1))

    @property
    def mean_within_se(self) -> float:
        finite = [s for s in self.paired_ses if math.isfinite(s)]
        return sum(finite) / len(finite) if finite else float("nan")

    @property
    def distinct_allocations(self) -> int:
        return len(set(self.fingerprints))

    @property
    def dominated_by_allocation(self) -> bool:
        """Is allocation noise larger than SEASON noise?

        When it is, more seasons buy nothing: the estimate would move more by
        redrawing the shared board than by simulating another 10,000 years.
        This says where to spend effort. It says **nothing** about whether the
        effect itself is real -- an effect of 0.50 with an allocation spread of
        0.03 trips this flag and is still perfectly solid.
        """
        w = self.mean_within_se
        return math.isfinite(w) and self.between_sd > w

    @property
    def spread_to_effect(self) -> float:
        """Between-allocation SD as a fraction of the effect. Small is stable."""
        m = abs(self.mean_delta)
        return self.between_sd / m if m > 1e-12 else float("inf")

    @property
    def dominates_effect(self) -> bool:
        """Is the effect swamped by which continuation we happened to draw?

        This is the question that decides whether a number may be quoted, and
        it is NOT :attr:`dominated_by_allocation`. Confusing the two would
        reject an effect twenty times larger than its own allocation spread
        merely because seasons are cheaper than boards.
        """
        return self.spread_to_effect > 0.5

    @property
    def sign_stable(self) -> bool:
        return all(d > 0 for d in self.deltas) or all(d < 0 for d in self.deltas)

    def to_dict(self) -> Dict[str, object]:
        return {
            "seeds": list(self.seeds),
            "deltas": [round(d, 6) for d in self.deltas],
            "paired_ses": [round(s, 6) for s in self.paired_ses],
            "fingerprints": list(self.fingerprints),
            "distinct_allocations": self.distinct_allocations,
            "mean_delta": round(self.mean_delta, 6),
            "between_allocation_sd": round(self.between_sd, 6),
            "mean_within_allocation_se": round(self.mean_within_se, 6),
            "dominated_by_allocation": self.dominated_by_allocation,
            "spread_to_effect": (None if not math.isfinite(self.spread_to_effect)
                                 else round(self.spread_to_effect, 4)),
            "dominates_effect": self.dominates_effect,
            "sign_stable": self.sign_stable,
            "note": ("between-allocation spread does NOT shrink with more "
                     "seasons and is never pooled into the paired SE. "
                     "'dominated_by_allocation' compares it to SEASON noise "
                     "(where to spend effort); 'dominates_effect' compares it "
                     "to the EFFECT (whether a number may be quoted)."),
        }


def allocation_spread(seeds: Sequence[int], deltas: Sequence[float],
                      paired_ses: Sequence[float],
                      fingerprints: Sequence[str]) -> AllocationSpread:
    if not (len(seeds) == len(deltas) == len(paired_ses) == len(fingerprints)):
        raise ValueError("allocation_spread needs one entry per seed")
    return AllocationSpread(tuple(seeds), tuple(float(d) for d in deltas),
                            tuple(float(s) for s in paired_ses),
                            tuple(fingerprints))
