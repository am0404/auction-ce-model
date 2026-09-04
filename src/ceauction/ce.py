"""Championship-equity estimation and paired scenario comparison.

Championship equity is simply ``P(this team wins week 17)`` under the league's
winner-take-all format.

The interesting part is the *comparison*.  Two rosters that differ by one
player differ in championship equity by well under a percentage point, so
comparing two independent Monte Carlo runs is hopeless: at 10,000 seasons the
standard error of a single CE estimate near 0.09 is about 0.0029, which swamps
the effect being measured.

The fix is **common random numbers**.  Both scenarios are simulated over the
same seasons: same injuries, same team environments, same schedule permutation,
same weekly noise for every player who appears in both.  The estimator is then
the mean of the *paired* difference

    d_i = 1{A wins season i} - 1{B wins season i}

whose standard error is ``sd(d) / sqrt(n)``.  Because ``d_i`` is zero in every
season where the swap changed nothing, this is typically 5-15x tighter than
differencing two independent runs -- ``paired_efficiency`` in the report says
by exactly how much for the comparison at hand.

There are **no separate playoff coin flips** anywhere.  Once the score array
exists, standings and bracket are a deterministic function of it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from .league import LeagueSettings
from .roster import RosterSet
from .simulate import DEFAULT_CHUNK, SeasonOutcomes, simulate_seasons

__all__ = [
    "TeamReport",
    "PairedComparison",
    "team_report",
    "championship_equity",
    "compare_scenarios",
    "wilson_interval",
]


def _se_mean(x: np.ndarray) -> float:
    n = x.shape[0]
    if n < 2:
        return float("nan")
    return float(np.std(x, ddof=1) / math.sqrt(n))


def wilson_interval(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval -- well behaved for the small CE probabilities here."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    d = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, center - half), min(1.0, center + half))


@dataclass(frozen=True)
class TeamReport:
    """Every headline metric for one team in one scenario."""

    team_index: int
    team_name: str
    n_sims: int
    championship_equity: float
    ce_se: float
    ce_ci95: Tuple[float, float]
    playoff_probability: float
    playoff_se: float
    bye_probability: float
    bye_se: float
    final_probability: float
    mean_regular_season_points: float
    mean_points_per_week: float
    points_se: float
    above_median_rate: float
    above_median_se: float
    head_to_head_win_rate: float
    head_to_head_se: float
    mean_wins: float
    mean_seed: float
    mean_slots_filled: float

    def lines(self) -> list:
        return [
            f"  championship equity   {self.championship_equity:8.4f}  "
            f"+/- {1.96 * self.ce_se:.4f}   (95% CI {self.ce_ci95[0]:.4f}-{self.ce_ci95[1]:.4f})",
            f"  playoff probability   {self.playoff_probability:8.4f}  +/- {1.96 * self.playoff_se:.4f}",
            f"  bye probability       {self.bye_probability:8.4f}  +/- {1.96 * self.bye_se:.4f}",
            f"  reached final         {self.final_probability:8.4f}",
            f"  reg. season points    {self.mean_regular_season_points:8.2f}  "
            f"({self.mean_points_per_week:.2f}/week, +/- {1.96 * self.points_se:.2f})",
            f"  above-median rate     {self.above_median_rate:8.4f}  +/- {1.96 * self.above_median_se:.4f}",
            f"  head-to-head win rate {self.head_to_head_win_rate:8.4f}  +/- {1.96 * self.head_to_head_se:.4f}",
            f"  mean record (of {int(self.n_weeks_total)})   {self.mean_wins:8.2f}",
            f"  mean seed             {self.mean_seed:8.2f}",
            f"  mean slots filled     {self.mean_slots_filled:8.3f} / 8",
        ]

    @property
    def n_weeks_total(self) -> float:
        return 28.0


def team_report(
    outcomes: SeasonOutcomes, team: int, settings: LeagueSettings
) -> TeamReport:
    """Summarise one team's outcomes, with Monte Carlo uncertainty on each metric."""
    n = outcomes.n_sims
    weeks = settings.regular_season_weeks
    champ = outcomes.champion_indicator(team)
    playoffs = outcomes.made_playoffs[:, team].astype(np.float64)
    bye = outcomes.has_bye[:, team].astype(np.float64)
    final = outcomes.made_final[:, team].astype(np.float64)
    pts = outcomes.points[:, team].astype(np.float64)
    med = outcomes.median_wins[:, team].astype(np.float64) / weeks
    h2h = outcomes.h2h_wins[:, team].astype(np.float64) / weeks

    successes = int(champ.sum())
    return TeamReport(
        team_index=team,
        team_name=outcomes.team_names[team],
        n_sims=n,
        championship_equity=float(champ.mean()),
        ce_se=_se_mean(champ),
        ce_ci95=wilson_interval(successes, n),
        playoff_probability=float(playoffs.mean()),
        playoff_se=_se_mean(playoffs),
        bye_probability=float(bye.mean()),
        bye_se=_se_mean(bye),
        final_probability=float(final.mean()),
        mean_regular_season_points=float(pts.mean()),
        mean_points_per_week=float(pts.mean() / weeks),
        points_se=_se_mean(pts / weeks),
        above_median_rate=float(med.mean()),
        above_median_se=_se_mean(med),
        head_to_head_win_rate=float(h2h.mean()),
        head_to_head_se=_se_mean(h2h),
        mean_wins=float(outcomes.wins[:, team].mean()),
        mean_seed=float(outcomes.seed[:, team].mean() + 1),
        mean_slots_filled=float(outcomes.starters_filled[:, team].mean()),
    )


def championship_equity(
    rosters: RosterSet, n_sims: int, seed: int, chunk: int = DEFAULT_CHUNK
) -> np.ndarray:
    """``(n_teams,)`` championship probability for every team."""
    return simulate_seasons(rosters, n_sims, seed, chunk).championship_equity()


@dataclass(frozen=True)
class PairedComparison:
    """A/B comparison of two rosters over identical simulated worlds."""

    label: str
    scenario_a: str
    scenario_b: str
    team_index: int
    n_sims: int
    seed: int
    report_a: TeamReport
    report_b: TeamReport
    delta_ce: float
    delta_ce_se: float
    delta_playoff: float
    delta_playoff_se: float
    delta_bye: float
    delta_bye_se: float
    delta_points_per_week: float
    delta_points_se: float
    delta_above_median: float
    delta_above_median_se: float
    delta_h2h: float
    delta_h2h_se: float
    seasons_differing: int
    paired_efficiency: float
    notes: str = ""

    @property
    def delta_ce_z(self) -> float:
        if self.delta_ce_se == 0.0 or math.isnan(self.delta_ce_se):
            return 0.0
        return self.delta_ce / self.delta_ce_se

    @property
    def significant_95(self) -> bool:
        return abs(self.delta_ce_z) >= 1.96

    def format(self, width: int = 78) -> str:
        bar = "=" * width
        out = [bar, f"{self.label}", bar,
               f"seasons {self.n_sims:,}   seed {self.seed}   "
               f"focus team #{self.team_index} ({self.report_a.team_name})"]
        if self.notes:
            out.append(f"note: {self.notes}")
        out.append("")
        out.append(f"[A] {self.scenario_a}")
        out.extend(self.report_a.lines())
        out.append("")
        out.append(f"[B] {self.scenario_b}")
        out.extend(self.report_b.lines())
        out.append("")
        out.append("PAIRED DIFFERENCES (A - B), common random numbers")
        out.append(
            f"  delta CE              {self.delta_ce:+9.5f}  +/- {1.96 * self.delta_ce_se:.5f}"
            f"   z = {self.delta_ce_z:+6.2f}   {'SIGNIFICANT' if self.significant_95 else 'not significant'}"
        )
        out.append(f"  delta playoff prob    {self.delta_playoff:+9.5f}  +/- {1.96 * self.delta_playoff_se:.5f}")
        out.append(f"  delta bye prob        {self.delta_bye:+9.5f}  +/- {1.96 * self.delta_bye_se:.5f}")
        out.append(f"  delta points/week     {self.delta_points_per_week:+9.5f}  +/- {1.96 * self.delta_points_se:.5f}")
        out.append(f"  delta above-median    {self.delta_above_median:+9.5f}  +/- {1.96 * self.delta_above_median_se:.5f}")
        out.append(f"  delta head-to-head    {self.delta_h2h:+9.5f}  +/- {1.96 * self.delta_h2h_se:.5f}")
        out.append(
            f"  seasons with a different champion: {self.seasons_differing:,} "
            f"({self.seasons_differing / max(self.n_sims, 1):.2%})"
        )
        out.append(
            f"  CRN variance reduction vs. unpaired: {self.paired_efficiency:.1f}x "
            f"(equivalent to {self.paired_efficiency:.1f}x the seasons)"
        )
        return "\n".join(out)


def compare_scenarios(
    rosters_a: RosterSet,
    rosters_b: RosterSet,
    team_index: int,
    n_sims: int,
    seed: int,
    label: str = "comparison",
    scenario_a: str = "A",
    scenario_b: str = "B",
    chunk: int = DEFAULT_CHUNK,
    notes: str = "",
) -> PairedComparison:
    """Compare two leagues over the *same* simulated worlds.

    The two ``RosterSet`` objects must describe the same league differing only
    in the change under test.  Pairing is achieved by the RNG design rather than
    by any special handling here: every player's draws are keyed by his
    ``crn_key``, the schedule permutation is keyed by season index, and group
    shocks are keyed by group name, so anything unchanged between the two
    scenarios draws identical numbers.
    """
    settings = rosters_a.settings
    out_a = simulate_seasons(rosters_a, n_sims, seed, chunk)
    out_b = simulate_seasons(rosters_b, n_sims, seed, chunk)

    rep_a = team_report(out_a, team_index, settings)
    rep_b = team_report(out_b, team_index, settings)
    weeks = settings.regular_season_weeks

    d_ce = out_a.champion_indicator(team_index) - out_b.champion_indicator(team_index)
    d_po = (out_a.made_playoffs[:, team_index].astype(np.float64)
            - out_b.made_playoffs[:, team_index].astype(np.float64))
    d_by = (out_a.has_bye[:, team_index].astype(np.float64)
            - out_b.has_bye[:, team_index].astype(np.float64))
    d_pt = (out_a.points[:, team_index].astype(np.float64)
            - out_b.points[:, team_index].astype(np.float64)) / weeks
    d_md = (out_a.median_wins[:, team_index].astype(np.float64)
            - out_b.median_wins[:, team_index].astype(np.float64)) / weeks
    d_hh = (out_a.h2h_wins[:, team_index].astype(np.float64)
            - out_b.h2h_wins[:, team_index].astype(np.float64)) / weeks

    paired_se = _se_mean(d_ce)
    unpaired_var = (rep_a.ce_se ** 2) + (rep_b.ce_se ** 2)
    efficiency = (unpaired_var / (paired_se ** 2)) if paired_se > 0 else float("inf")

    return PairedComparison(
        label=label,
        scenario_a=scenario_a,
        scenario_b=scenario_b,
        team_index=team_index,
        n_sims=n_sims,
        seed=seed,
        report_a=rep_a,
        report_b=rep_b,
        delta_ce=float(d_ce.mean()),
        delta_ce_se=paired_se,
        delta_playoff=float(d_po.mean()),
        delta_playoff_se=_se_mean(d_po),
        delta_bye=float(d_by.mean()),
        delta_bye_se=_se_mean(d_by),
        delta_points_per_week=float(d_pt.mean()),
        delta_points_se=_se_mean(d_pt),
        delta_above_median=float(d_md.mean()),
        delta_above_median_se=_se_mean(d_md),
        delta_h2h=float(d_hh.mean()),
        delta_h2h_se=_se_mean(d_hh),
        seasons_differing=int(np.count_nonzero(out_a.champion != out_b.champion)),
        paired_efficiency=float(efficiency),
        notes=notes,
    )
