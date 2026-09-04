"""The marginal championship-equity curve for one roster slot.

The question every auction pricing scheme is a transformation of is::

    how much championship equity does one more projected point buy,
    at this roster slot, given the other fourteen players I own?

This module answers it by sweeping a single rostered slot's ``base_mean`` from
replacement level to elite, holding *everything* else fixed, and reporting
``CE(level)`` with honest uncertainty at every step.

Pricing itself is **not** built here, by instruction: no dollar values, no
opening or live max bids, no inflation model, no roster-completion solver.
This produces the curve; what a pricing layer does with it is a separate
question.

Why the sweep is not just N independent runs
--------------------------------------------
Two things make the numbers usable at the resolution pricing needs.

**Common random numbers.**  Every level is the *same player* with one field
changed.  He keeps his ``player_id`` and therefore his ``crn_key``, so his
injuries, byes, weekly conditions, observable signals, spikes and idiosyncratic
draws are bit-identical across levels; every other player in the league is
untouched; the schedule permutation is keyed by season index.  What legitimately
*does* move with his level is his own realized scoring, his team's weekly
totals, and therefore the league median and every team's record — that is the
effect being measured, not a leak.

**Paired differences, including for the slopes.**  Each level retains its
per-season champion indicator, so a difference between any two levels is a
matched per-season difference over the identical worlds.  In seasons where the
change did not alter the champion the difference is exactly zero and the
estimate tightens.  Crucially the *adjacent* slope between level ``i-1`` and
level ``i`` is computed as its own paired difference, **not** by subtracting
two separately-estimated baseline deltas.

That second point is worth more than it looks.  Two adjacent baseline deltas
share the baseline arm and are strongly positively correlated, and the two
levels themselves differ in very few seasons, so the true adjacent SE is much
smaller than either baseline SE.  Combining them as if independent
(``sqrt(se_i^2 + se_{i-1}^2)``) is not a small conservatism: measured on the
19-level example it inflates the slope's standard error by roughly 2.5x, which
would turn a resolved slope into an unresolved one.
``test_the_paired_slope_se_is_not_the_unpaired_combination`` pins the direction.

The resolution report
---------------------
:class:`ResolutionReport` sizes the question that gates every downstream
decision: *how much simulation would it take to tell a 0.002 CE difference from
zero, and how long would that run?*  It extrapolates the measured paired
standard error as ``se ∝ 1/sqrt(n)`` and prices the answer in seconds using
throughput measured during the sweep itself, not a hardcoded constant.

It is a **pilot estimate scoped to the comparison that produced it**, not a
capability claim about the engine.  Paired variance tracks the discordance rate
-- how often the focus team's outcome actually flips -- which varies between
comparisons, and helping and hurting seasons cancel in the mean while both
adding to the variance.  A decision that must actually be resolved needs its
own pilot or an adaptive stopping rule; see :class:`ResolutionReport`.
"""

from __future__ import annotations

import csv
import io
import math
import time
from dataclasses import dataclass, replace
from typing import ClassVar, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .ce import paired_se, wilson_interval
from .league import DEFAULT_LEAGUE, LeagueSettings, Position
from .players import PlayerSpec, with_overrides
from .roster import RosterSet
from .simulate import DEFAULT_CHUNK, simulate_seasons

__all__ = [
    "CurvePoint",
    "ResolutionTarget",
    "ResolutionReport",
    "MarginalCurve",
    "sweep_marginal_curve",
    "level_grid",
    "weakest_flex_slot",
    "isotonic_fit",
    "DEFAULT_RESOLUTION_TARGETS",
    "LIVE_AUCTION_BUDGET_SECONDS",
]

#: Delta-CE magnitudes the resolution report prices out, smallest last.
DEFAULT_RESOLUTION_TARGETS: Tuple[float, ...] = (0.005, 0.002, 0.001)

#: A live auction gives you roughly this long to answer "what is this player
#: worth to me?" before the clock forces a bid.  A paired comparison that
#: cannot finish inside it is not usable in the room, whatever its accuracy.
LIVE_AUCTION_BUDGET_SECONDS: float = 30.0

#: Positions the FLEX slot accepts.  The default sweep target is the focus
#: team's weakest player among these, because that is the roster spot whose
#: marginal value a $1-$3 auction decision actually turns on.
FLEX_ELIGIBLE: Tuple[Position, ...] = (Position.RB, Position.WR, Position.TE)


def weakest_flex_slot(rosters: RosterSet, team: int) -> PlayerSpec:
    """The team's lowest-``base_mean`` FLEX-eligible player.

    Ties break on ``player_id`` so the choice is deterministic.
    """
    candidates = [
        rosters.spec(pid)
        for pid in rosters.rosters[team].player_ids
        if rosters.spec(pid).position in FLEX_ELIGIBLE
    ]
    if not candidates:
        raise ValueError(f"team {team} has no FLEX-eligible player")
    return min(candidates, key=lambda s: (s.base_mean, s.player_id))


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CurvePoint:
    """One level of the sweep, with every metric the curve reports.

    ``delta_*`` fields compare this level against the replacement-level
    baseline over identical seasons.  ``slope_*`` fields compare it against the
    *previous* level in the sweep, also over identical seasons, and are
    expressed per projected point so they are directly comparable across
    unequal step sizes.
    """

    level: float
    is_baseline: bool

    championship_equity: float
    ce_se: float
    ce_ci95: Tuple[float, float]

    delta_ce: float
    delta_ce_se: float
    delta_ce_z: float

    points_per_week: float
    delta_points_per_week: float

    playoff_probability: float
    delta_playoff: float
    bye_probability: float
    delta_bye: float

    seasons_ce_differs: int
    """Seasons in which *this team's* championship outcome flipped.

    Deliberately not the same quantity as
    ``PairedComparison.seasons_differing``, which counts seasons where the
    league champion changed at all -- a strictly larger number, since the title
    can move between two rivals without touching the focus team.  This one is
    the count that drives the paired standard error, because it is exactly the
    number of non-zero entries in the difference being averaged."""

    # --- adjacent, genuinely paired against the previous level --------------
    slope_from_level: Optional[float] = None
    slope_step: Optional[float] = None
    slope_dce_per_point: Optional[float] = None
    slope_dce_per_point_se: Optional[float] = None
    slope_dce_per_point_z: Optional[float] = None
    slope_seasons_ce_differs: Optional[int] = None

    #: Optional isotonic display value for ``championship_equity``, produced by
    #: *imposing* monotonicity rather than testing for it.  Never overwrites the
    #: raw estimate; see :func:`isotonic_fit`.
    ce_isotonic: Optional[float] = None

    @property
    def significant_95(self) -> bool:
        return abs(self.delta_ce_z) >= 1.96


@dataclass(frozen=True)
class ResolutionTarget:
    """What it would cost to resolve a delta-CE of ``delta_ce`` from zero."""

    delta_ce: float
    required_sims: int
    required_seconds: float
    resolved_at_current_sims: bool
    live_auction_feasible: bool

    def verdict(self, budget: float = LIVE_AUCTION_BUDGET_SECONDS) -> str:
        """A statement about *this pilot*, not a general capability claim.

        The wording is deliberately scoped: what was measured is that one
        comparison of this shape, on this hardware, at this discordance rate,
        would take this long.  Whether some other decision resolves in time is
        a question about that decision, not one this row answers.
        """
        if self.live_auction_feasible:
            return (f"within budget in this pilot ({self.required_seconds:.0f}s "
                    f"< {budget:.0f}s)")
        if self.required_seconds < 600:
            return (f"over budget in this pilot; offline "
                    f"({self.required_seconds / 60:.1f} min per comparison)")
        if self.required_seconds < 36_000:
            return (f"far over budget in this pilot "
                    f"({self.required_seconds / 3600:.1f} h per comparison)")
        return (f"far over budget in this pilot "
                f"({self.required_seconds / 86_400:.1f} days per comparison)")


@dataclass(frozen=True)
class ResolutionReport:
    """A **pilot estimate** of what resolving a small delta-CE would cost.

    Read every number here as scoped to the comparison that produced it: this
    synthetic curve, this roster, this hardware, this pool size, and the
    discordance rate actually observed between these adjacent levels.  It is
    not a general statement about what the engine can resolve.

    **What the extrapolation assumes.**  That the paired standard error falls
    as ``1/sqrt(n)``, which is exact for a mean of i.i.d. per-season
    differences, and that the *per-season variance* of some future comparison
    resembles the median observed here.  The first assumption is safe.  The
    second is not, and it does not err in a predictable direction:

    * the paired difference is zero in every season the change did not decide,
      so its variance tracks the **discordance rate** -- how often the focus
      team's championship outcome actually flips -- which varies from
      comparison to comparison and is not a simple function of the effect size;
    * the difference takes values in ``{-1, 0, +1}``, so seasons where the
      change helps and seasons where it hurts **cancel in the mean while both
      adding to the variance**.  A comparison with a small mean and a high flip
      rate is noisier than one with the same mean and a low flip rate.

    Both push the required sample size in whichever direction that particular
    comparison happens to sit.  So this is a scenario-specific pilot, not a
    bound, and certainly not a conservative one.

    **What to do with it.**  Treat it as sizing information for a comparison of
    this shape.  Any decision that must actually be resolved needs either its
    own short pilot run to estimate that comparison's discordance rate, or an
    adaptive stopping rule that simulates until the paired interval excludes
    zero (or until a wall-clock budget is spent, whichever comes first).
    Neither is built here.
    """

    n_sims: int
    seconds_per_season: float
    seconds_per_paired_comparison: float
    observed_adjacent_se: float
    observed_adjacent_step: float
    observed_adjacent_differing_rate: float
    observed_baseline_se: float
    smallest_resolved_adjacent_dce: float
    targets: Tuple[ResolutionTarget, ...]
    live_auction_budget_seconds: float = LIVE_AUCTION_BUDGET_SECONDS

    def lines(self) -> List[str]:
        out = [
            f"  simulations per arm            {self.n_sims:,}",
            f"  measured throughput            {1.0 / self.seconds_per_season:,.0f} seasons/s "
            f"({self.seconds_per_season * 1e3:.3f} ms/season)",
            f"  cost of one paired comparison  {self.seconds_per_paired_comparison:.1f}s "
            f"at {self.n_sims:,} seasons",
            f"  paired SE, adjacent step       {self.observed_adjacent_se:.5f} "
            f"(median step {self.observed_adjacent_step:.2f} pts/week, champion "
            f"differs in {self.observed_adjacent_differing_rate:.1%} of seasons)",
            f"  paired SE, vs the baseline     {self.observed_baseline_se:.5f}",
            f"  smallest adjacent dCE this run resolves at |z|=2: "
            f"{self.smallest_resolved_adjacent_dce:.5f}",
            "",
            "  PILOT ESTIMATE, SCOPED TO THIS COMPARISON. The counts below extrapolate",
            "  the median adjacent SE observed above as se ~ 1/sqrt(n). That is exact",
            "  for the sample size, but it assumes a future comparison has similar",
            "  per-season variance -- and that variance tracks the DISCORDANCE RATE (how",
            "  often the focus team's outcome actually flips), which differs between",
            "  comparisons and is not a simple function of effect size. The difference",
            "  also takes values in {-1, 0, +1}, so helping and hurting seasons cancel",
            "  in the mean while both add to the variance. These counts are therefore",
            "  neither a bound nor conservative: they size a comparison of THIS shape.",
            "  A decision that must actually be resolved needs its own pilot run, or an",
            "  adaptive rule that simulates until the paired interval excludes zero.",
            "",
            f"  {'target dCE':>10}  {'sims needed':>13}  {'seconds/comparison':>19}   verdict",
            f"  {'-' * 10}  {'-' * 13}  {'-' * 19}   {'-' * 40}",
        ]
        for t in self.targets:
            out.append(
                f"  {t.delta_ce:>10.4f}  {t.required_sims:>13,}  "
                f"{t.required_seconds:>19,.1f}   {t.verdict(self.live_auction_budget_seconds)}"
            )
        return out


@dataclass(frozen=True)
class MarginalCurve:
    """A complete sweep: the curve, its provenance, and its resolution."""

    team_index: int
    team_name: str
    player_id: int
    player_name: str
    position: Position
    baseline_level: float
    n_sims: int
    seed: int
    chunk: int
    points: Tuple[CurvePoint, ...]
    resolution: ResolutionReport
    notes: str = ""

    # --- machine-readable ---------------------------------------------------

    CSV_COLUMNS: ClassVar[Tuple[str, ...]] = (
        "level",
        "is_baseline",
        "championship_equity",
        "ce_se",
        "ce_ci95_lo",
        "ce_ci95_hi",
        "delta_ce",
        "delta_ce_se",
        "delta_ce_z",
        "points_per_week",
        "delta_points_per_week",
        "playoff_probability",
        "delta_playoff",
        "bye_probability",
        "delta_bye",
        "seasons_ce_differs",
        "slope_from_level",
        "slope_step",
        "slope_dce_per_point",
        "slope_dce_per_point_se",
        "slope_dce_per_point_z",
        "slope_seasons_ce_differs",
        "ce_isotonic",
    )

    def rows(self) -> List[Dict[str, object]]:
        out: List[Dict[str, object]] = []
        for p in self.points:
            out.append({
                "level": p.level,
                "is_baseline": int(p.is_baseline),
                "championship_equity": p.championship_equity,
                "ce_se": p.ce_se,
                "ce_ci95_lo": p.ce_ci95[0],
                "ce_ci95_hi": p.ce_ci95[1],
                "delta_ce": p.delta_ce,
                "delta_ce_se": p.delta_ce_se,
                "delta_ce_z": p.delta_ce_z,
                "points_per_week": p.points_per_week,
                "delta_points_per_week": p.delta_points_per_week,
                "playoff_probability": p.playoff_probability,
                "delta_playoff": p.delta_playoff,
                "bye_probability": p.bye_probability,
                "delta_bye": p.delta_bye,
                "seasons_ce_differs": p.seasons_ce_differs,
                "slope_from_level": p.slope_from_level,
                "slope_step": p.slope_step,
                "slope_dce_per_point": p.slope_dce_per_point,
                "slope_dce_per_point_se": p.slope_dce_per_point_se,
                "slope_dce_per_point_z": p.slope_dce_per_point_z,
                "slope_seasons_ce_differs": p.slope_seasons_ce_differs,
                "ce_isotonic": p.ce_isotonic,
            })
        return out

    def to_csv(self, path: Optional[str] = None) -> str:
        """Write (or return) the curve as CSV.

        Empty cells rather than ``None``/``nan`` text for the fields that do
        not exist at the first level, so the file loads cleanly anywhere.
        """
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=list(self.CSV_COLUMNS),
                                restval="", extrasaction="raise",
                                lineterminator="\n")
        writer.writeheader()
        for row in self.rows():
            writer.writerow({k: ("" if v is None else v) for k, v in row.items()})
        text = buf.getvalue()
        if path is not None:
            with open(path, "w", newline="", encoding="utf-8") as fh:
                fh.write(text)
        return text

    # --- human-readable -----------------------------------------------------

    def format(self, width: int = 100) -> str:
        bar = "=" * width
        show_iso = any(p.ce_isotonic is not None for p in self.points)
        out = [
            bar,
            f"MARGINAL CHAMPIONSHIP-EQUITY CURVE -- {self.player_name} "
            f"({self.position.label}) on {self.team_name}",
            bar,
            f"{self.n_sims:,} seasons per level, seed {self.seed}, chunk {self.chunk}, "
            f"{len(self.points)} levels",
            f"baseline (replacement) level {self.baseline_level:.2f} pts/week; "
            f"deltas and slopes are paired over identical seasons",
        ]
        if self.notes:
            out.append(f"note: {self.notes}")
        out.append("")

        head = (f"{'level':>6} {'CE':>8} {'+/-95%':>8} {'dCE vs base':>12} {'+/-95%':>8} "
                f"{'z':>7} {'dCE/pt':>9} {'+/-95%':>8} {'z':>6} "
                f"{'pts/wk':>8} {'playoff':>8} {'bye':>7}")
        if show_iso:
            head += f" {'CE(iso)':>8}"
        out.append(head)
        out.append("-" * len(head))
        for p in self.points:
            row = (
                f"{p.level:>6.2f} {p.championship_equity:>8.4f} "
                f"{1.96 * p.ce_se:>8.4f} {p.delta_ce:>+12.5f} "
                f"{1.96 * p.delta_ce_se:>8.5f} {p.delta_ce_z:>+7.2f} "
            )
            if p.slope_dce_per_point is None:
                row += f"{'--':>9} {'--':>8} {'--':>6} "
            else:
                row += (f"{p.slope_dce_per_point:>+9.5f} "
                        f"{1.96 * p.slope_dce_per_point_se:>8.5f} "
                        f"{p.slope_dce_per_point_z:>+6.2f} ")
            row += (f"{p.points_per_week:>8.2f} {p.playoff_probability:>8.4f} "
                    f"{p.bye_probability:>7.4f}")
            if show_iso:
                row += f" {p.ce_isotonic:>8.4f}" if p.ce_isotonic is not None else f" {'--':>8}"
            out.append(row)
        out.append("-" * len(head))
        out.append("")
        out.append("dCE/pt is the ADJACENT paired slope: level i vs level i-1 over the same")
        out.append("seasons, divided by the step. It is not a difference of two baseline")
        out.append("deltas, which would treat two highly correlated estimates as independent.")
        if show_iso:
            out.append("CE(iso) IMPOSES monotonicity; it does not test for it. Changing a")
            out.append("player's level changes which players get started, so a local decline")
            out.append("in raw CE may be a real pathwise effect rather than noise. The raw CE,")
            out.append("its interval and every delta to its left are unchanged by it.")
        out.append("")
        out.append(bar)
        out.append("MONTE CARLO RESOLUTION -- can this engine price with these numbers?")
        out.append(bar)
        out.extend(self.resolution.lines())
        return "\n".join(out)


# ---------------------------------------------------------------------------
# Isotonic display fit (optional; never replaces a raw estimate)
# ---------------------------------------------------------------------------


def isotonic_fit(values: Sequence[float], weights: Optional[Sequence[float]] = None) -> List[float]:
    """Pool-adjacent-violators fit, non-decreasing, weighted.

    **This imposes an assumption; it does not reveal one.**  Monotonicity of CE
    in a player's level is *plausible* but not guaranteed, and it is worth
    being precise about why.  The simulated manager sets lineups from noisy
    pregame projections, so raising a player's level changes which players he
    starts, in which weeks.  Those changed start decisions propagate: a
    different starter means a different team score, a different league median
    for all twelve teams, different records, different seeding and a different
    bracket.  Nothing forces that pathwise chain to be favourable at every
    level, so a local decline in the raw curve is **not** automatically Monte
    Carlo noise -- it may be a real feature of this roster and this decision
    rule.

    Use this as a *display aid* when a monotone reading is what you want, and
    read it as "the curve under an imposed monotonicity assumption".  It is
    never written over the raw estimates: :class:`CurvePoint` keeps
    ``championship_equity``, its interval, every delta and every slope exactly
    as measured, and the report prints both side by side.  If the fit differs
    visibly from the raw values, that is information about the assumption, not
    a correction to the data.

    Weighted by ``1 / variance`` when weights are supplied, so a noisy level
    is pulled toward its neighbours more readily than a precise one.
    """
    y = [float(v) for v in values]
    if not y:
        return []
    w = [1.0] * len(y) if weights is None else [max(float(x), 1e-300) for x in weights]
    if len(w) != len(y):
        raise ValueError("weights must match values in length")

    vals: List[float] = []
    wts: List[float] = []
    runs: List[int] = []
    for value, weight in zip(y, w):
        vals.append(value)
        wts.append(weight)
        runs.append(1)
        # Merge backwards while the sequence decreases.
        while len(vals) > 1 and vals[-2] > vals[-1]:
            total = wts[-2] + wts[-1]
            merged = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / total
            vals[-2:] = [merged]
            wts[-2:] = [total]
            runs[-2:] = [runs[-2] + runs[-1]]

    out: List[float] = []
    for value, run in zip(vals, runs):
        out.extend([value] * run)
    return out


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _LevelRun:
    """Per-season focus-team outcomes retained for one level."""

    level: float
    champion: np.ndarray   # (S,) float 0/1
    playoffs: np.ndarray   # (S,) float 0/1
    bye: np.ndarray        # (S,) float 0/1
    points: np.ndarray     # (S,) points per week
    seconds: float


def _candidate(spec: PlayerSpec, level: float) -> PlayerSpec:
    """The same player at a different level, and identical in every other way.

    ``crn_key`` is pinned to the original stream key explicitly rather than
    left to default, so this stays correct even if the caller passes a spec
    that already carried a custom key.

    **Published weekly projections move with the level.**  When a spec carries
    a ``weekly_projection_override`` -- real published projections, one per
    simulated week -- that array *replaces* the modelled projection entirely
    (``worlds._build_pregame``).  Changing ``base_mean`` alone would therefore
    move the player's realized scoring while leaving the manager's pregame view
    frozen at the original level, so the sweep would be measuring "a player who
    quietly got better and whose projections never noticed" rather than "a
    better player".  Every level of the curve would share one pregame view, the
    lineup decisions would be identical everywhere, and the measured slope
    would collapse toward the value of unforecastable production.

    So the override is shifted by the same delta as ``base_mean``::

        delta = level - spec.base_mean
        override[w] -> override[w] + delta      for every week w

    which preserves the *shape* of the published projection -- its bye weeks,
    its matchup swings, its in-season drift -- while moving its overall level.
    That is the intended meaning of "vary this player's projected level".

    A spec with no override is untouched by this, and its projection continues
    to come from the model as before.
    """
    changes = dict(base_mean=float(level), crn_key=spec.stream_key)
    if spec.weekly_projection_override is not None:
        delta = float(level) - float(spec.base_mean)
        changes["weekly_projection_override"] = tuple(
            float(v) + delta for v in spec.weekly_projection_override
        )
    return with_overrides(spec, **changes)


def level_grid(
    min_level: float, max_level: float, step: float, tol: float = 1e-9
) -> List[float]:
    """Levels from ``min_level`` to ``max_level`` in ``step`` increments.

    The endpoint is always included and is **never exceeded**.  When the range
    is not an exact multiple of the step the final increment is shortened
    rather than overshooting: ``level_grid(4, 10, 4)`` is ``[4, 8, 10]``, not
    ``[4, 8, 12]``.

    The last point is snapped to ``max_level`` exactly rather than left as an
    accumulated ``min + k * step``, so a caller asking for 22.0 gets 22.0 and
    not 21.999999999999996.
    """
    if step <= 0.0:
        raise ValueError("step must be positive")
    if max_level < min_level - tol:
        raise ValueError("max_level must be >= min_level")
    if max_level <= min_level + tol:
        return [float(min_level)]

    n_full = int(math.floor((max_level - min_level) / step + tol))
    levels = [float(min_level) + k * float(step) for k in range(n_full + 1)]
    if levels[-1] < max_level - tol:
        levels.append(float(max_level))
    else:
        levels[-1] = float(max_level)
    return levels


def _resolution(
    n_sims: int,
    seconds_per_season: float,
    adjacent_ses: Sequence[float],
    adjacent_steps: Sequence[float],
    adjacent_differing: Sequence[int],
    baseline_ses: Sequence[float],
    targets: Sequence[float],
    budget: float,
) -> ResolutionReport:
    adj = float(np.median(adjacent_ses)) if len(adjacent_ses) else float("nan")
    step = float(np.median(adjacent_steps)) if len(adjacent_steps) else float("nan")
    rate = (float(np.median(adjacent_differing)) / n_sims
            if len(adjacent_differing) else float("nan"))
    base = float(np.median(baseline_ses)) if len(baseline_ses) else float("nan")
    # Two arms are simulated per paired comparison.
    per_comparison = 2.0 * n_sims * seconds_per_season

    rows: List[ResolutionTarget] = []
    for target in targets:
        if not np.isfinite(adj) or adj <= 0.0 or target <= 0.0:
            continue
        # se(n) = adj * sqrt(n_sims / n); require |z| = target / se(n) >= 2.
        needed = int(math.ceil(n_sims * (2.0 * adj / target) ** 2))
        seconds = 2.0 * needed * seconds_per_season
        rows.append(
            ResolutionTarget(
                delta_ce=target,
                required_sims=needed,
                required_seconds=seconds,
                resolved_at_current_sims=needed <= n_sims,
                live_auction_feasible=seconds <= budget,
            )
        )
    return ResolutionReport(
        n_sims=n_sims,
        seconds_per_season=seconds_per_season,
        seconds_per_paired_comparison=per_comparison,
        observed_adjacent_se=adj,
        observed_adjacent_step=step,
        observed_adjacent_differing_rate=rate,
        observed_baseline_se=base,
        smallest_resolved_adjacent_dce=2.0 * adj,
        targets=tuple(rows),
        live_auction_budget_seconds=budget,
    )


def sweep_marginal_curve(
    rosters: RosterSet,
    team: int,
    player_id: int,
    baseline_level: float,
    levels: Iterable[float],
    n_sims: int,
    seed: int,
    chunk: int = DEFAULT_CHUNK,
    settings: Optional[LeagueSettings] = None,
    resolution_targets: Sequence[float] = DEFAULT_RESOLUTION_TARGETS,
    live_auction_budget_seconds: float = LIVE_AUCTION_BUDGET_SECONDS,
    isotonic: bool = False,
    notes: str = "",
) -> MarginalCurve:
    """Sweep one roster slot's ``base_mean`` and report ``CE(level)``.

    Parameters
    ----------
    rosters:
        The league.  Every team other than ``team`` is untouched at every
        level, and ``team``'s other fourteen players are untouched too.
    team:
        Focus team index.  All reported metrics are this team's.
    player_id:
        The rostered player whose slot is varied.  He must be on ``team``.
        His position, variance, injury profile, correlation loadings, weekly
        state and signal precision are held fixed.  What moves is his
        *projected level*: ``base_mean``, and — if he carries real published
        projections — every ``weekly_projection_override`` entry shifted by the
        same delta, so the published shape is preserved while its level moves.
        See :func:`_candidate`.
    baseline_level:
        Replacement-level anchor.  Every ``delta_ce`` is measured against it.
    levels:
        Candidate ``base_mean`` values.  **Order does not matter**: the sweep
        sorts them ascending and removes duplicates, so a caller cannot change
        the answer by shuffling the request.  ``baseline_level`` is added to
        the set if absent and is simulated exactly once either way.
    n_sims, seed, chunk:
        As for :func:`ceauction.simulate.simulate_seasons`.  Results are
        identical for any ``chunk``.
    isotonic:
        Add a display column that *imposes* monotonicity on CE.  It is an
        assumption, not a measurement -- changing a player's level changes
        which players get started, so a local decline in the raw curve is not
        automatically noise.  Raw estimates and intervals are retained
        unchanged and remain primary; see :func:`isotonic_fit`.

    Notes
    -----
    Every level is the same ``player_id`` at a different projected level, so his
    ``crn_key`` — and therefore his injury, bye, spike, signal, weekly-state
    and idiosyncratic draws — is identical across levels, as is every other
    player's and the schedule permutation.  What legitimately *does* move with
    his level is his own realized scoring, his own pregame projection, his
    team's weekly totals, and hence the league median and every team's record.
    That is the effect being measured, not a leak.
    """
    settings = settings or rosters.settings
    if player_id not in rosters.rosters[team].player_ids:
        raise KeyError(
            f"player {player_id} is not on {rosters.rosters[team].team_name}"
        )
    if n_sims < 2:
        raise ValueError("n_sims must be at least 2 for a standard error")

    spec = rosters.spec(player_id)
    weeks = settings.regular_season_weeks

    requested = [float(v) for v in levels]
    # Sorted and deduplicated: the curve is a function of the level *set*, so
    # request order cannot change any reported number.
    grid = sorted({round(v, 10) for v in requested} | {round(float(baseline_level), 10)})

    runs: List[_LevelRun] = []
    for level in grid:
        variant = rosters.with_pool_player(_candidate(spec, level))
        t0 = time.perf_counter()
        out = simulate_seasons(variant, n_sims, seed, chunk, settings=settings)
        dt = time.perf_counter() - t0
        runs.append(
            _LevelRun(
                level=level,
                champion=out.champion_indicator(team),
                playoffs=out.made_playoffs[:, team].astype(np.float64),
                bye=out.has_bye[:, team].astype(np.float64),
                points=out.points[:, team].astype(np.float64) / weeks,
                seconds=dt,
            )
        )

    base_i = next(i for i, r in enumerate(runs) if r.level == round(float(baseline_level), 10))
    base = runs[base_i]

    points: List[CurvePoint] = []
    adjacent_ses: List[float] = []
    adjacent_steps: List[float] = []
    adjacent_differing: List[int] = []
    baseline_ses: List[float] = []

    for i, run in enumerate(runs):
        d_ce = run.champion - base.champion
        d_se = paired_se(d_ce)
        successes = int(run.champion.sum())

        slope_from = slope_step = slope = slope_se = slope_z = None
        slope_diff = None
        if i > 0:
            prev = runs[i - 1]
            step = run.level - prev.level
            # Genuinely paired: matched per-season differences between the two
            # adjacent levels, then scaled to a per-point slope.  Dividing the
            # difference *before* taking the SE keeps the two consistent.
            per_point = (run.champion - prev.champion) / step
            slope_from = prev.level
            slope_step = step
            slope = float(per_point.mean())
            slope_se = paired_se(per_point)
            slope_z = 0.0 if (slope_se == 0.0 or math.isnan(slope_se)) else slope / slope_se
            slope_diff = int(np.count_nonzero(run.champion != prev.champion))
            # Record the SE of the raw adjacent delta (not per point), which is
            # what the resolution report is about.
            adjacent_ses.append(paired_se(run.champion - prev.champion))
            adjacent_steps.append(step)
            adjacent_differing.append(slope_diff)

        if i != base_i:
            baseline_ses.append(d_se)

        points.append(
            CurvePoint(
                level=run.level,
                is_baseline=(i == base_i),
                championship_equity=float(run.champion.mean()),
                ce_se=paired_se(run.champion),
                ce_ci95=wilson_interval(successes, n_sims),
                delta_ce=float(d_ce.mean()),
                delta_ce_se=d_se,
                delta_ce_z=(0.0 if (d_se == 0.0 or math.isnan(d_se))
                            else float(d_ce.mean()) / d_se),
                points_per_week=float(run.points.mean()),
                delta_points_per_week=float((run.points - base.points).mean()),
                playoff_probability=float(run.playoffs.mean()),
                delta_playoff=float((run.playoffs - base.playoffs).mean()),
                bye_probability=float(run.bye.mean()),
                delta_bye=float((run.bye - base.bye).mean()),
                seasons_ce_differs=int(np.count_nonzero(run.champion != base.champion)),
                slope_from_level=slope_from,
                slope_step=slope_step,
                slope_dce_per_point=slope,
                slope_dce_per_point_se=slope_se,
                slope_dce_per_point_z=slope_z,
                slope_seasons_ce_differs=slope_diff,
            )
        )

    if isotonic:
        weights = [1.0 / max(p.ce_se ** 2, 1e-300) for p in points]
        fitted = isotonic_fit([p.championship_equity for p in points], weights)
        points = [replace(p, ce_isotonic=f) for p, f in zip(points, fitted)]

    total_seconds = sum(r.seconds for r in runs)
    seconds_per_season = total_seconds / (len(runs) * n_sims)
    resolution = _resolution(
        n_sims, seconds_per_season, adjacent_ses, adjacent_steps,
        adjacent_differing, baseline_ses, resolution_targets,
        live_auction_budget_seconds,
    )

    return MarginalCurve(
        team_index=team,
        team_name=rosters.rosters[team].team_name,
        player_id=player_id,
        player_name=spec.name,
        position=Position(int(spec.position)),
        baseline_level=round(float(baseline_level), 10),
        n_sims=n_sims,
        seed=seed,
        chunk=chunk,
        points=tuple(points),
        resolution=resolution,
        notes=notes,
    )
