"""The CE laboratory: controlled, paired championship-equity experiments.

Each experiment builds two leagues that differ in exactly one respect, runs
both over the *same* simulated seasons, and reports the paired difference in
championship equity for one focus team.

**These are infrastructure tests, not fantasy-football conclusions.**  Every
number they produce is a statement about the synthetic process in
``synthetic.py``, whose parameters are invented.  What they establish is that
the engine responds to each structural change in a measurable, correctly
signed, statistically resolvable way -- which is the property the eventual
pricing layer depends on.

The experiments are deliberately *not* aggregated into advice.  Read them as:
"a change of this kind, of this size, moves CE by this much, and the machinery
can detect it at this sample size."
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .ce import PairedComparison, compare_scenarios
from .league import DEFAULT_LEAGUE, LeagueSettings, Position
from .players import Contingency, PlayerSpec, ShockLoading, with_overrides
from .roster import Roster, RosterSet
from .synthetic import SyntheticConfig, make_synthetic_league

__all__ = [
    "ExperimentSpec",
    "ExperimentOutput",
    "EXPERIMENTS",
    "run_experiment",
    "run_all",
    "lab_player",
    "roster_by_strength",
]

LAB_ID_BASE = 900_000
FOCUS_TEAM = 0


# ---------------------------------------------------------------------------
# Roster surgery helpers
# ---------------------------------------------------------------------------


def roster_by_strength(rosters: RosterSet, team: int) -> List[PlayerSpec]:
    """A team's players sorted by ``base_mean``, strongest first."""
    specs = [rosters.spec(pid) for pid in rosters.rosters[team].player_ids]
    return sorted(specs, key=lambda s: (-s.base_mean, s.player_id))


def lab_player(
    player_id: int,
    name: str,
    position: Position,
    base_mean: float,
    week_sd: float = 6.5,
    crn_key: Optional[int] = None,
    **kwargs,
) -> PlayerSpec:
    """Build a controlled laboratory player.

    Defaults are deliberately inert (no injuries, no role changes, no spikes,
    no correlation) so that an experiment turns on exactly one dial and nothing
    else moves.
    """
    params = dict(
        season_sd=0.0,
        weekly_injury_hazard=0.0,
        spike_rate=0.0,
        spike_scale=0.0,
        role_change_prob=0.0,
        weekly_state_sd=0.0,
        proj_noise_sd=0.0,
        shock_loadings=(),
        contingency=None,
        bye_week=0,
        nfl_team="LAB",
        data_source="SYNTHETIC",
        notes="CE laboratory control player",
    )
    params.update(kwargs)
    return PlayerSpec(
        player_id=player_id,
        name=name,
        position=position,
        base_mean=base_mean,
        week_sd=week_sd,
        crn_key=crn_key if crn_key is not None else player_id,
        **params,
    )


def swap_in(rosters: RosterSet, team: int, out_player_id: int, new: PlayerSpec) -> RosterSet:
    """Put ``new`` on ``team`` in place of ``out_player_id``.

    The outgoing player stays in the pool but unrostered, which costs a little
    compute and changes nothing else -- importantly it keeps his random streams
    identical in both scenarios.
    """
    rs = rosters.with_pool_player(new)
    roster = rs.rosters[team].replaced(out_player_id, new.player_id)
    return rs.with_roster(team, roster)


def tweak(rosters: RosterSet, player_id: int, **overrides) -> RosterSet:
    """Change parameters of an existing pool player, preserving his CRN key.

    Preserving ``crn_key`` is what makes the comparison paired: the tweaked
    player draws the same uniforms as the original, so the *only* thing that
    moves is the parameter under test.
    """
    spec = rosters.spec(player_id)
    overrides.setdefault("crn_key", spec.stream_key)
    return rosters.with_pool_player(with_overrides(spec, **overrides))


def _matched_variance(total_sd: float, beta: float) -> float:
    """Idiosyncratic SD that leaves total weekly variance at ``total_sd**2``.

    Used by the correlation experiments so that A and B have *identical*
    marginal distributions and differ only in their dependence structure.
    """
    resid = total_sd ** 2 - beta ** 2
    if resid <= 0:
        raise ValueError("beta is too large for the requested total SD")
    return math.sqrt(resid)


# ---------------------------------------------------------------------------
# Experiment framework
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExperimentOutput:
    """Everything one experiment produced."""

    key: str
    title: str
    question: str
    comparisons: Tuple[PairedComparison, ...]
    interpretation: str = ""

    def format(self, width: int = 78) -> str:
        out = ["", "#" * width, f"# EXPERIMENT {self.key}: {self.title}",
               f"# Q: {self.question}", "#" * width]
        for c in self.comparisons:
            out.append(c.format(width))
            out.append("")
        if self.interpretation:
            out.append(f"READING: {self.interpretation}")
        return "\n".join(out)

    def summary_rows(self) -> List[Tuple[str, float, float, float, float, float]]:
        """One row per comparison for the summary table.

        Carries the scoring delta alongside the CE delta: when a CE difference
        is not significant, the scoring delta says whether the mechanism failed
        to fire at all or merely produced an effect below the resolution of the
        chosen sample size.  Those are very different diagnoses.
        """
        return [
            (c.label, c.delta_ce, c.delta_ce_se, c.delta_ce_z,
             c.delta_points_per_week, c.delta_points_se)
            for c in self.comparisons
        ]


Runner = Callable[[RosterSet, int, int], ExperimentOutput]


@dataclass(frozen=True)
class ExperimentSpec:
    key: str
    title: str
    question: str
    runner: Runner


def _run(a: RosterSet, b: RosterSet, label: str, n: int, seed: int,
         name_a: str, name_b: str, notes: str = "") -> PairedComparison:
    return compare_scenarios(
        a, b, FOCUS_TEAM, n, seed, label=label,
        scenario_a=name_a, scenario_b=name_b, notes=notes,
    )


# ---------------------------------------------------------------------------
# 1. One extra projected point, at different lineup positions
# ---------------------------------------------------------------------------


def exp_marginal_point(base: RosterSet, n: int, seed: int) -> ExperimentOutput:
    ranked = roster_by_strength(base, FOCUS_TEAM)
    targets: List[Tuple[str, PlayerSpec]] = []
    for pos in (Position.QB, Position.RB, Position.WR):
        at_pos = [s for s in ranked if s.position is pos]
        targets.append((f"best {pos.label} (roster's #1 {pos.label})", at_pos[0]))
    targets.append(("2nd RB (fills RB2)", [s for s in ranked if s.position is Position.RB][1]))
    targets.append(("marginal starter (8th-strongest player)", ranked[7]))
    targets.append(("first bench player (9th-strongest)", ranked[8]))
    targets.append(("last bench player (15th-strongest)", ranked[14]))

    comps = []
    for label, spec in targets:
        a = tweak(base, spec.player_id, base_mean=spec.base_mean + 1.0)
        comps.append(
            _run(a, base, f"+1.00 pt/week to {label}: {spec.name}", n, seed,
                 f"{spec.name} base_mean {spec.base_mean + 1.0:.2f}",
                 f"{spec.name} base_mean {spec.base_mean:.2f}",
                 notes="only base_mean changes; the player keeps his CRN key, "
                       "so every other draw in the league is identical")
        )
    return ExperimentOutput(
        key="marginal-point",
        title="One additional projected point at different lineup positions",
        question="Is a point of projection worth the same everywhere on the roster?",
        comparisons=tuple(comps),
        interpretation=(
            "A point added to a player who starts nearly every week converts almost "
            "fully into team points; a point added deep on the bench converts only "
            "through the weeks that player is actually startable. The engine should "
            "show delta-CE decreasing monotonically as the target moves down the roster."
        ),
    )


# ---------------------------------------------------------------------------
# 2. Second QB vs an equally priced flex alternative
# ---------------------------------------------------------------------------


def exp_second_qb(base: RosterSet, n: int, seed: int) -> ExperimentOutput:
    """Run the same QB-vs-WR swap on two rosters: one already deep at QB, one not.

    Running it only once would produce a single number that looks like a
    positional constant.  It is not one, and showing both sides is the point.
    """
    ranked = roster_by_strength(base, FOCUS_TEAM)
    victim = ranked[9]
    level = 14.0
    shared = dict(week_sd=6.5, season_sd=2.0, weekly_injury_hazard=0.03,
                  injury_mean_weeks=2.2, proj_noise_sd=1.0)

    # A roster that is genuinely short at QB: its backup QBs are cut to
    # replacement level, so the SUPERFLEX has to be filled by a flex player.
    thin = base
    backups = [s for s in ranked if s.position is Position.QB][1:]
    for b in backups:
        thin = tweak(thin, b.player_id, base_mean=3.0)

    comps = []
    for tag, root in (("roster already has two startable QBs", base),
                      ("roster is thin at QB (backups cut to 3.0)", thin)):
        qb = lab_player(LAB_ID_BASE + 1, "LAB-QB2", Position.QB, level,
                        crn_key=LAB_ID_BASE + 1, **shared)
        wr = lab_player(LAB_ID_BASE + 2, "LAB-FLEX", Position.WR, level,
                        crn_key=LAB_ID_BASE + 1, **shared)
        comps.append(
            _run(swap_in(root, FOCUS_TEAM, victim.player_id, qb),
                 swap_in(root, FOCUS_TEAM, victim.player_id, wr),
                 f"QB vs WR at {level:.1f} pts/week -- {tag}", n, seed,
                 f"QB at {level:.1f} pts/week",
                 f"WR at {level:.1f} pts/week",
                 notes="identical mean, variance, injury risk and CRN key; the only "
                       "difference is the position label and therefore slot eligibility")
        )
    return ExperimentOutput(
        key="second-qb",
        title="A second QB vs an equally priced flex alternative",
        question=("With a superflex that does not require a QB, is a startable second "
                  "QB worth more than a WR with the same mean and variance?"),
        comparisons=tuple(comps),
        interpretation=(
            "The two players are distributionally identical, so any difference is "
            "purely positional: a QB fills QB or SUPERFLEX; a WR fills WT, FLEX or "
            "SUPERFLEX. The sign therefore flips with the roster. On a roster that "
            "already starts two QBs, a third is nearly worthless while the WR upgrades "
            "a real flex slot. On a roster thin at QB, the QB claims a SUPERFLEX that "
            "was being filled by a weak flex player. This is exactly why a positional "
            "modifier is the wrong shape of answer and a per-roster CE measurement is "
            "the right one."
        ),
    )


# ---------------------------------------------------------------------------
# 3. Stable vs volatile at equal mean
# ---------------------------------------------------------------------------


def exp_volatility(base: RosterSet, n: int, seed: int) -> ExperimentOutput:
    ranked = roster_by_strength(base, FOCUS_TEAM)
    comps = []
    stable_sd, volatile_sd = 4.0, 11.0
    for label, idx in (("starter-level (4th-strongest)", 3), ("flex-level (8th-strongest)", 7)):
        victim = ranked[idx]
        level = victim.base_mean
        # Realized scores are not floored, so the weekly draw is symmetric about
        # the level and both arms expect exactly `level` points per week at any
        # `week_sd`.  The arms therefore need no correction of any kind: they
        # share a base_mean and differ only in dispersion.
        stable = lab_player(LAB_ID_BASE + 10 + idx, f"LAB-STABLE-{idx}", victim.position,
                            level, week_sd=stable_sd, crn_key=LAB_ID_BASE + 10 + idx)
        volatile = lab_player(LAB_ID_BASE + 20 + idx, f"LAB-VOLATILE-{idx}", victim.position,
                              level, week_sd=volatile_sd,
                              crn_key=LAB_ID_BASE + 10 + idx)
        comps.append(
            _run(swap_in(base, FOCUS_TEAM, victim.player_id, volatile),
                 swap_in(base, FOCUS_TEAM, victim.player_id, stable),
                 f"volatile (sd {volatile_sd:.1f}) vs stable (sd {stable_sd:.1f}) "
                 f"at {label}", n, seed,
                 f"weekly sd {volatile_sd:.1f}, base {level:.2f}",
                 f"weekly sd {stable_sd:.1f}, base {level:.2f}",
                 notes=f"identical base_mean ({level:.2f}) and identical CRN key; "
                       f"expected points per week match exactly, so the arms "
                       f"differ in the shape of the weekly distribution and in "
                       f"nothing else")
        )
    return ExperimentOutput(
        key="volatility",
        title="Stable vs volatile players with equal mean projections",
        question="At a fixed mean, does weekly variance help or hurt championship equity?",
        comparisons=tuple(comps),
        interpretation=(
            "Variance is not one-signed. It hurts the median-result half of the "
            "standings (you want to clear the median every week) and helps the "
            "single-elimination bracket (you need one big week against a better team). "
            "The net sign is an empirical property of this league's format, and this "
            "experiment is how you measure it rather than assume it."
        ),
    )


# ---------------------------------------------------------------------------
# 4. Predictable upside vs unforecastable spike weeks
# ---------------------------------------------------------------------------


def exp_spikes(base: RosterSet, n: int, seed: int) -> ExperimentOutput:
    ranked = roster_by_strength(base, FOCUS_TEAM)
    victim = ranked[8]
    floor_level = victim.base_mean
    bonus = 3.0
    spike_rate = 0.15

    predictable = lab_player(
        LAB_ID_BASE + 31, "LAB-PREDICT", victim.position, floor_level + bonus,
        week_sd=7.0, crn_key=LAB_ID_BASE + 31,
    )
    spiky = lab_player(
        LAB_ID_BASE + 32, "LAB-SPIKE", victim.position, floor_level,
        week_sd=7.0, crn_key=LAB_ID_BASE + 31,
        spike_rate=spike_rate, spike_scale=bonus / spike_rate,
        spike_mean_removed=False,
    )
    return ExperimentOutput(
        key="spikes",
        title="Predictable upside vs unforecastable spike weeks",
        question=("Two players average the same points. One's extra production is in "
                  "the projection every week; the other's arrives as 15%-frequency "
                  "spikes nobody can forecast. How much is forecastability worth?"),
        comparisons=(
            _run(swap_in(base, FOCUS_TEAM, victim.player_id, predictable),
                 swap_in(base, FOCUS_TEAM, victim.player_id, spiky),
                 f"predictable +{bonus:.1f} vs unforecastable +{bonus:.1f} in spikes",
                 n, seed,
                 f"{floor_level + bonus:.1f} pts/week, fully projected",
                 f"{floor_level:.1f} pts/week projected, +{bonus:.1f} in "
                 f"{spike_rate:.0%} spikes of {bonus / spike_rate:.1f}",
                 notes="identical expected points per week; only the spikes are "
                       "invisible to the lineup decision"),
        ),
        interpretation=(
            "This is the cleanest statement of the information rule. The spiky player "
            "produces the same total points, but a share of them land in weeks he was "
            "benched, where they are worth exactly zero. The gap is the price of "
            "unforecastability -- and it is the full price, because the spikes reach "
            "no future projection either. Beliefs update from the observable-signal "
            "channel, which reads the persistent level and never the realized score, "
            "so no amount of spiking teaches the manager anything. An earlier "
            "revision filtered on realized residuals and did partially recover the "
            "loss, which was the bug: it let unforecastable scoring become forecast."
        ),
    )


# ---------------------------------------------------------------------------
# 5. One strong player vs several aggregate flex options
# ---------------------------------------------------------------------------


def exp_concentration(base: RosterSet, n: int, seed: int) -> ExperimentOutput:
    ranked = roster_by_strength(base, FOCUS_TEAM)
    victims = [s for s in ranked if s.position in (Position.WR, Position.TE)][-3:]
    total = 30.0
    concentrated = (18.0, 6.0, 6.0)
    spread = (10.0, 10.0, 10.0)

    def build(levels):
        rs = base
        for k, (v, lvl) in enumerate(zip(victims, levels)):
            rs = swap_in(
                rs, FOCUS_TEAM, v.player_id,
                lab_player(LAB_ID_BASE + 40 + k, f"LAB-AGG{k}", Position.WR, lvl,
                           week_sd=7.0, crn_key=LAB_ID_BASE + 40 + k,
                           weekly_injury_hazard=0.03, injury_mean_weeks=2.2),
            )
        return rs

    return ExperimentOutput(
        key="concentration",
        title="One strong player vs multiple aggregate flex options",
        question=(f"Three roster spots, {total:.0f} projected points to distribute. "
                  "Is 18/6/6 better or worse than 10/10/10?"),
        comparisons=(
            _run(build(concentrated), build(spread),
                 "18/6/6 vs 10/10/10 across three WR spots", n, seed,
                 "one strong WR (18.0) plus two replacement (6.0)",
                 "three even WRs (10.0 each)",
                 notes="equal total expected points (30.0/week), equal weekly sd, "
                       "equal injury risk. Scores are not floored, so the two arms "
                       "expect exactly the same total production and the entire "
                       "scoring delta is the lineup effect"),
        ),
        interpretation=(
            "Only some of these three players start in any given week, so the "
            "distribution matters even at constant total. Concentration wins when the "
            "roster can reliably start the stud; spreading wins when byes, injuries and "
            "slot limits mean the third-best of the group gets used often. This is the "
            "portfolio question the 15-for-8 roster poses, measured rather than assumed."
        ),
    )


# ---------------------------------------------------------------------------
# 5b. Building a lineup spot in the aggregate
# ---------------------------------------------------------------------------


def offset_patterns(n_weeks: int, n_players: int, amplitude: float) -> List[Tuple[float, ...]]:
    """``n_players`` mean-zero weekly patterns with disjoint good weeks.

    Player *k* is hot in the weeks where ``w % n_players == k``, and exactly one
    player is hot in any given week, so the group's per-week total is constant.
    Each pattern is centred on its own hot-week frequency rather than on
    ``1/n_players``, which makes every player's season mean exactly zero even
    when ``n_weeks`` is not divisible by ``n_players`` -- otherwise the arm with
    the patterns would quietly carry a level difference too.
    """
    weeks = np.arange(n_weeks)
    out: List[Tuple[float, ...]] = []
    for k in range(n_players):
        hot = (weeks % n_players == k).astype(np.float64)
        out.append(tuple(amplitude * (hot - hot.mean())))
    return out


def exp_aggregate_lineup_spot(base: RosterSet, n: int, seed: int) -> ExperimentOutput:
    """One stable starter vs several cheaper candidates who rotate.

    Three WR/TE roster spots, 30.0 expected points per week to distribute, in
    three arms that are matched on total *and* on per-player season strength:

    ``stable``       18.0 / 6.0 / 6.0, flat every week.
    ``forecastable`` 10.0 each, with offset weekly patterns so that in every
                     week exactly one is hot and the other two are cold -- and
                     the manager can see which, before lock.
    ``hidden``       the identical numbers, applied through
                     ``hidden_weekly_pattern`` so they reach the realized score
                     but never the projection.

    **What the rotating arms actually look like.**  17 weeks do not divide by
    three, so the hot week is not shared evenly and the option set is *not*
    literally ``{18, 6, 6}`` every week.  Each pattern is centred on its own
    hot-week frequency (:func:`offset_patterns`), which is what keeps every
    candidate's season mean at exactly 10.0; the price is that two candidates
    are hot in 6 weeks of 17 and the third in 5, so their hot and cold levels
    differ slightly.  With ``level = 10.0`` and ``amplitude = 12.0`` the weekly
    option set is::

        {17.76, 6.47, 5.76}   in 12 weeks   (one of the 6/17 candidates is hot)
        {18.47, 5.76, 5.76}   in  5 weeks   (the 5/17 candidate is hot)

    Two things *are* exact, and they are the two the matched design needs:
    the three levels sum to **exactly 30.0 in every week**, and each candidate's
    season mean is **exactly 10.0**.  So the arms are matched on per-week total
    and on per-player strength, and the small spread in hot/cold levels is a
    consequence of mean-zero centring rather than a defect in the matching.

    The ``forecastable`` and ``hidden`` arms have **byte-identical realized
    production**.  They differ in exactly one thing: whether the good weeks
    were identifiable before kickoff.  That is the control the whole experiment
    turns on, and it is why this cannot be mistaken for retrospectively picking
    the players who happened to score.
    """
    ranked = roster_by_strength(base, FOCUS_TEAM)
    victims = [s for s in ranked if s.position in (Position.WR, Position.TE)][-3:]
    n_weeks = base.settings.total_weeks
    level, amplitude = 10.0, 12.0
    concentrated = (18.0, 6.0, 6.0)
    patterns = offset_patterns(n_weeks, len(victims), amplitude)

    def build(levels, forecastable_patterns=None, hidden_patterns=None):
        rs = base
        for k, (v, lvl) in enumerate(zip(victims, levels)):
            extra = {}
            if forecastable_patterns is not None:
                extra["weekly_state_pattern"] = forecastable_patterns[k]
            if hidden_patterns is not None:
                extra["hidden_weekly_pattern"] = hidden_patterns[k]
            rs = swap_in(
                rs, FOCUS_TEAM, v.player_id,
                lab_player(LAB_ID_BASE + 90 + k, f"LAB-AGG-SPOT{k}", Position.WR, lvl,
                           week_sd=7.0, crn_key=LAB_ID_BASE + 90 + k, **extra),
            )
        return rs

    stable = build(concentrated)
    forecastable = build((level,) * 3, forecastable_patterns=patterns)
    hidden = build((level,) * 3, hidden_patterns=patterns)

    # Report the levels the patterns actually produce rather than the nominal
    # 1/n ones: hot-week frequencies differ when n_weeks is not divisible by n.
    arr = np.asarray(patterns, dtype=np.float64) + level
    hot_lo, hot_hi = float(arr.max(axis=1).min()), float(arr.max(axis=1).max())
    cold_lo, cold_hi = float(arr.min(axis=1).min()), float(arr.min(axis=1).max())
    matched = ("all three arms hold exactly 30.0 expected points per week "
               "across the three spots, and every candidate's season mean is "
               f"exactly {level:.1f}; the patterns are mean-zero by "
               f"construction. {n_weeks} weeks do not divide by "
               f"{len(victims)}, so hot and cold levels differ slightly "
               "between candidates -- the per-week total and the per-player "
               "means are exact regardless")

    return ExperimentOutput(
        key="aggregate-lineup-spot",
        title="One stable starter vs several cheaper candidates who rotate",
        question=("Three roster spots and 30.0 points per week. Is a single 18.0 "
                  "starter worth more than three 10.0 candidates whose good and "
                  "bad weeks are offset -- and does the answer depend on whether "
                  "the good weeks are visible before kickoff?"),
        comparisons=(
            _run(forecastable, stable,
                 "forecastable rotation vs one stable starter", n, seed,
                 f"three candidates at {level:.1f}, offset weekly conditions "
                 f"({hot_lo:.2f}-{hot_hi:.2f} hot / {cold_lo:.2f}-{cold_hi:.2f} "
                 f"cold), visible pregame",
                 f"one starter at {concentrated[0]:.1f} plus two at "
                 f"{concentrated[1]:.1f}, flat",
                 notes=matched + ". Every week the rotating arm presents one "
                       "hot candidate and two cold ones summing to the same "
                       "30.0 the stable arm holds, so if the model prices "
                       "knowable weekly conditions correctly this should be "
                       "roughly a wash"),
            _run(hidden, stable,
                 "unforecastable rotation vs one stable starter (CONTROL)", n, seed,
                 f"three candidates at {level:.1f}, same weekly swings, "
                 f"invisible pregame",
                 f"one starter at {concentrated[0]:.1f} plus two at "
                 f"{concentrated[1]:.1f}, flat",
                 notes=matched + ". Identical realized production to the "
                       "arm above; the manager simply cannot see which "
                       "candidate is hot, so he projects 10.0 for all three "
                       "and starts whichever roster order happens to favour"),
            _run(forecastable, hidden,
                 "the same rotation, forecastable vs not", n, seed,
                 "offset weekly conditions visible before lock",
                 "the identical weekly swings, hidden until after kickoff",
                 notes="the two arms' realized scores are byte-identical, so "
                       "delta-CE here is purely the value of being able to "
                       "identify the good weeks in advance -- nothing is "
                       "selected retrospectively"),
        ),
        interpretation=(
            "This is the aggregate-roster question the 15-for-8 format poses, and "
            "it only has an answer once the model has a forecastable weekly state. "
            "Cheap candidates can substitute for an expensive starter exactly to "
            "the extent their good weeks are identifiable before lineup lock. The "
            "first comparison measures how much of the concentrated player's value "
            "the rotation recovers; the third isolates the price of forecastability "
            "with realized production held byte-identical. The second is the "
            "control that stops the first from being read as 'variance is free': "
            "the same swings, unforecastable, cannot be harvested at all."
        ),
    )


# ---------------------------------------------------------------------------
# 6. Injury probability
# ---------------------------------------------------------------------------


def exp_injury(base: RosterSet, n: int, seed: int) -> ExperimentOutput:
    ranked = roster_by_strength(base, FOCUS_TEAM)
    # Use a genuinely valuable, every-week starter: availability risk on a
    # marginal player is worth too little to resolve at any sane sample size.
    victim = ranked[1]
    level = max(victim.base_mean, 17.0)
    comps = []
    for hazard_hi in (0.08, 0.16):
        healthy = lab_player(LAB_ID_BASE + 50, "LAB-HEALTHY", victim.position, level,
                             week_sd=7.0, crn_key=LAB_ID_BASE + 50,
                             weekly_injury_hazard=0.02, injury_mean_weeks=2.5)
        fragile = lab_player(LAB_ID_BASE + 51, "LAB-FRAGILE", victim.position, level,
                             week_sd=7.0, crn_key=LAB_ID_BASE + 50,
                             weekly_injury_hazard=hazard_hi, injury_mean_weeks=2.5)
        comps.append(
            _run(swap_in(base, FOCUS_TEAM, victim.player_id, fragile),
                 swap_in(base, FOCUS_TEAM, victim.player_id, healthy),
                 f"weekly injury hazard {hazard_hi:.0%} vs 2%", n, seed,
                 f"{level:.1f} pts/week, hazard {hazard_hi:.2f}",
                 f"{level:.1f} pts/week, hazard 0.02",
                 notes="same mean when active, same variance, same CRN key")
        )
    return ExperimentOutput(
        key="injury",
        title="Different injury probabilities at equal per-week production",
        question="What does availability risk cost, holding per-week production fixed?",
        comparisons=tuple(comps),
        interpretation=(
            "Cost should scale with expected missed weeks, but not linearly: the bench "
            "absorbs part of an absence, and with no IR slot the injured player also "
            "occupies a roster spot the whole time. Whether the cost is bigger or "
            "smaller than the naive 'mean x missed weeks' is exactly what the "
            "portfolio structure decides."
        ),
    )


# ---------------------------------------------------------------------------
# 7. Independent vs correlated bench upside
# ---------------------------------------------------------------------------


def exp_bench_correlation(base: RosterSet, n: int, seed: int) -> ExperimentOutput:
    ranked = roster_by_strength(base, FOCUS_TEAM)
    victims = [s for s in ranked if s.position in (Position.WR, Position.TE)][-2:]
    # High enough that these two are genuinely in the weekly decision, and a
    # large enough loading that the dependence structure has room to matter.
    level, total_sd, beta = 11.5, 9.0, 7.0
    idio = _matched_variance(total_sd, beta)

    def build(group_ids):
        rs = base
        for k, (v, gid) in enumerate(zip(victims, group_ids)):
            rs = swap_in(
                rs, FOCUS_TEAM, v.player_id,
                lab_player(LAB_ID_BASE + 60 + k, f"LAB-BENCH{k}", Position.WR, level,
                           week_sd=idio, crn_key=LAB_ID_BASE + 60 + k,
                           shock_loadings=(ShockLoading(gid, beta),)),
            )
        return rs

    correlated = build(("lab:bench-shared", "lab:bench-shared"))
    independent = build(("lab:bench-priv-0", "lab:bench-priv-1"))
    return ExperimentOutput(
        key="bench-correlation",
        title="Independent vs correlated bench upside",
        question=("Two bench players with identical marginal distributions: does it "
                  "matter whether their good weeks arrive together?"),
        comparisons=(
            _run(correlated, independent,
                 "correlated bench pair (rho ~ 0.60) vs independent", n, seed,
                 "both load 7.0 on one shared shock",
                 "each loads 7.0 on its own private shock",
                 notes=f"idiosyncratic sd set to {idio:.3f} in both arms so the total "
                       f"weekly sd is exactly {total_sd:.1f} either way"),
        ),
        interpretation=(
            "Marginals are identical by construction, so any delta is pure dependence "
            "structure. Independent bench upside should be worth more: the lineup takes "
            "the max over available options, and independent options give the max more "
            "chances to be high. Correlated bench players are closer to being one player."
        ),
    )


# ---------------------------------------------------------------------------
# 8. QB / pass-catcher correlation (stacking)
# ---------------------------------------------------------------------------


def exp_stack(base: RosterSet, n: int, seed: int) -> ExperimentOutput:
    ranked = roster_by_strength(base, FOCUS_TEAM)
    qb_victim = [s for s in ranked if s.position is Position.QB][0]
    wr_victim = [s for s in ranked if s.position is Position.WR][0]
    qb_level, wr_level = qb_victim.base_mean, wr_victim.base_mean
    total_sd, beta = 9.0, 7.0
    idio = _matched_variance(total_sd, beta)

    def build(qb_group, wr_group):
        rs = swap_in(base, FOCUS_TEAM, qb_victim.player_id,
                     lab_player(LAB_ID_BASE + 70, "LAB-STACK-QB", Position.QB, qb_level,
                                week_sd=idio, crn_key=LAB_ID_BASE + 70,
                                shock_loadings=(ShockLoading(qb_group, beta),)))
        return swap_in(rs, FOCUS_TEAM, wr_victim.player_id,
                       lab_player(LAB_ID_BASE + 71, "LAB-STACK-WR", Position.WR, wr_level,
                                  week_sd=idio, crn_key=LAB_ID_BASE + 71,
                                  shock_loadings=(ShockLoading(wr_group, beta),)))

    stacked = build("lab:stack", "lab:stack")
    split = build("lab:stack-qb", "lab:stack-wr")
    return ExperimentOutput(
        key="stack",
        title="QB / pass-catcher positive correlation",
        question=("Does starting a QB and his own receiver -- same marginals, shared "
                  "weekly shock -- change championship equity?"),
        comparisons=(
            _run(stacked, split, "stacked QB+WR vs the same players uncorrelated", n, seed,
                 "QB and WR share one weekly shock (rho ~ 0.60)",
                 "QB and WR on independent shocks",
                 notes=f"marginals matched exactly: idiosyncratic sd {idio:.3f}, "
                       f"total weekly sd {total_sd:.1f} in both arms"),
        ),
        interpretation=(
            "Stacking adds variance to the *team* score without changing either "
            "player's own distribution. In a format with a weekly median result, extra "
            "team variance is costly in the regular season; in single elimination it is "
            "valuable. This experiment measures the net, and is the hook a real "
            "correlation matrix will plug into unchanged."
        ),
    )


# ---------------------------------------------------------------------------
# 9. Starting RB / backup RB contingency
# ---------------------------------------------------------------------------


def exp_handcuff(base: RosterSet, n: int, seed: int) -> ExperimentOutput:
    ranked = roster_by_strength(base, FOCUS_TEAM)
    own_rb = [s for s in ranked if s.position is Position.RB][0]
    victim = [s for s in ranked if s.position is Position.RB][-1]
    other_rb = roster_by_strength(base, 1)
    foreign_rb = [s for s in other_rb if s.position is Position.RB][0]

    common = dict(week_sd=6.5, weekly_injury_hazard=0.02, injury_mean_weeks=2.2)
    bonus = 14.0
    # Both starters are made equally fragile *in both arms*, so the uplift fires
    # about equally often either way and only its timing differs.  Without this
    # the effect is real but far too small to resolve.
    root = tweak(base, own_rb.player_id, weekly_injury_hazard=0.14, injury_mean_weeks=2.6)
    root = tweak(root, foreign_rb.player_id, weekly_injury_hazard=0.14, injury_mean_weeks=2.6)

    handcuff = lab_player(LAB_ID_BASE + 80, "LAB-HANDCUFF", Position.RB, 4.0,
                          crn_key=LAB_ID_BASE + 80,
                          contingency=Contingency(own_rb.player_id, bonus), **common)
    decoy = lab_player(LAB_ID_BASE + 81, "LAB-DECOY-RB", Position.RB, 4.0,
                       crn_key=LAB_ID_BASE + 80,
                       contingency=Contingency(foreign_rb.player_id, bonus), **common)
    return ExperimentOutput(
        key="handcuff",
        title="Starting RB / backup RB contingency",
        question=("A backup who erupts when *your* starter is out, vs an identical "
                  "backup who erupts when *someone else's* starter is out."),
        comparisons=(
            _run(swap_in(root, FOCUS_TEAM, victim.player_id, handcuff),
                 swap_in(root, FOCUS_TEAM, victim.player_id, decoy),
                 f"handcuff to own {own_rb.name} vs same uplift keyed to a rival's RB",
                 n, seed,
                 f"+{bonus:.1f} pts in weeks {own_rb.name} is unavailable",
                 f"+{bonus:.1f} pts in weeks {foreign_rb.name} is unavailable",
                 notes="both arms have the same unconditional distribution -- the "
                       "uplift fires about as often either way. Only the *timing* "
                       "differs: one fires exactly when the roster needs it."),
        ),
        interpretation=(
            "This isolates contingency value from raw expected points. The two players "
            "score the same on average; the handcuff's points simply arrive in the "
            "weeks a hole opened in the lineup. If the engine prices timing correctly, "
            "the handcuff wins -- and note the uplift is pregame-observable, so the "
            "lineup optimiser actually starts him in those weeks."
        ),
    )


# ---------------------------------------------------------------------------
# 10. CONTROL: the same strong player on two interchangeable rival rosters
# ---------------------------------------------------------------------------


def exp_opponent_placement(base: RosterSet, n: int, seed: int) -> ExperimentOutput:
    strong = roster_by_strength(base, 1)[0]
    counterpart = roster_by_strength(base, 2)
    match = next(s for s in counterpart if s.position is strong.position)

    a = base
    # Both rosters must change in one step: swapping them one at a time would
    # transiently put the same player on two rosters, which RosterSet rejects.
    swapped = list(base.rosters)
    swapped[1] = base.rosters[1].replaced(strong.player_id, match.player_id)
    swapped[2] = base.rosters[2].replaced(match.player_id, strong.player_id)
    b = RosterSet(base.pool, tuple(swapped), base.settings)

    return ExperimentOutput(
        key="opponent-placement",
        title="CONTROL: the same strong player on two interchangeable rivals",
        question=("Your roster is untouched and the two recipient rosters are "
                  "statistically interchangeable. Does it matter to *you* which "
                  "of them owns a stud? It must not."),
        comparisons=(
            _run(a, b, f"{strong.name} on team 1 vs on team 2", n, seed,
                 f"{strong.name} on {base.rosters[1].team_name}, "
                 f"{match.name} on {base.rosters[2].team_name}",
                 f"{strong.name} on {base.rosters[2].team_name}, "
                 f"{match.name} on {base.rosters[1].team_name}",
                 notes="the focus team's 15 players are byte-identical in both arms"),
        ),
        interpretation=(
            "This is a CONTROL and nothing else. Rival rosters 1 and 2 are near "
            "duplicates of each other, and the player is swapped for his own "
            "counterpart at the same position, so the two arms differ only in a "
            "label. The schedule is re-permuted every season, so both rivals are "
            "equally likely to be any of your opponents, and the league median "
            "depends on the multiset of rosters rather than their names. delta-CE "
            "should be statistically indistinguishable from zero. A significant "
            "result here would indicate a bug -- most likely a schedule that is not "
            "exchangeable across teams. Read it alongside `rival-fit`, which asks "
            "the same question of rivals that are *not* interchangeable and gets a "
            "very different answer; the pair is the point."
        ),
    )


# ---------------------------------------------------------------------------
# 11. The same strong player on two rivals he fits very differently
# ---------------------------------------------------------------------------


def exp_rival_fit(base: RosterSet, n: int, seed: int) -> ExperimentOutput:
    """Non-exchangeable rivals: does specific ownership matter then?

    The control above is constructed so the answer must be no. This one is
    constructed so it can be yes, by making the two recipient rosters genuinely
    different in how well they can *use* the player:

    * ``CONTENDER`` already starts two strong QBs, so its SUPERFLEX is a real
      QB slot and a stud quarterback converts almost fully;
    * ``ALSO-RAN`` has replacement-level backups and is weak overall, so the
      same player lifts it only from bad to mediocre.

    Both QB-depth modifications are applied in **both** arms -- they are the
    fixed, non-exchangeable structure, not the treatment. The treatment is
    which rival receives the stud and which receives his same-CRN counterpart,
    so the league holds the identical multiset of player parameters either way.
    The focus team's fifteen players are byte-identical throughout, and its
    realized scoring is identical to the last decimal.
    """
    also_ran, contender = 1, 2
    shared = dict(week_sd=6.5, season_sd=2.0, weekly_injury_hazard=0.03,
                  injury_mean_weeks=2.2, proj_noise_sd=1.0, weekly_state_sd=2.5)
    stud_level, plain_level = 24.0, 8.0
    depth = {also_ran: 1.0, contender: 21.0}

    root = base
    qbs = {}
    for team in (also_ran, contender):
        team_qbs = [s for s in roster_by_strength(root, team)
                    if s.position is Position.QB]
        qbs[team] = team_qbs
        for b in team_qbs[1:]:
            root = tweak(root, b.player_id, base_mean=depth[team])

    def build(stud_to: int) -> RosterSet:
        rs = root
        for team in (also_ran, contender):
            is_stud = team == stud_to
            tag = "STUD" if is_stud else "PLAIN"
            # A shared crn_key means the roster that holds the stud in one arm
            # and the plain QB in the other draws the same uniforms either way.
            qb = lab_player(LAB_ID_BASE + 100 + team, f"LAB-RIVALQB-{tag}-T{team}",
                            Position.QB,
                            stud_level if is_stud else plain_level,
                            crn_key=LAB_ID_BASE + 100 + team, **shared)
            rs = swap_in(rs, team, qbs[team][0].player_id, qb)
        return rs

    weak_name = root.rosters[also_ran].team_name
    strong_name = root.rosters[contender].team_name
    return ExperimentOutput(
        key="rival-fit",
        title="The same strong player on two rivals who can use him very differently",
        question=("Your roster is untouched, but the two rivals are not "
                  "interchangeable: one can start two quarterbacks and one cannot. "
                  "Does it matter to *you* which of them ends up with the stud QB?"),
        comparisons=(
            _run(build(also_ran), build(contender),
                 "stud QB to the weak rival vs to the contender", n, seed,
                 f"{weak_name} (QB backups {depth[also_ran]:.1f}) gets the "
                 f"{stud_level:.1f} QB; {strong_name} gets the {plain_level:.1f} QB",
                 f"{strong_name} (QB backups {depth[contender]:.1f}) gets the "
                 f"{stud_level:.1f} QB; {weak_name} gets the {plain_level:.1f} QB",
                 notes="the focus team's fifteen players are byte-identical in "
                       "both arms and its realized scoring is identical to the "
                       "last decimal; the league holds the same multiset of "
                       "player parameters either way. Both rivals' QB depth is "
                       "fixed in advance and identical across arms, so the only "
                       "thing that moves is which of them receives the stud"),
        ),
        interpretation=(
            "The CONTROL immediately above says rival ownership is irrelevant when "
            "the recipients are interchangeable. That is a statement about "
            "exchangeability, and reading it as 'who buys a player never matters to "
            "you' would be the wrong lesson -- this experiment is the counterexample. "
            "Here the recipients differ in fit, and the result is not a small "
            "correction: sending the stud to the contender is materially worse for "
            "the focus team than sending him to the weak rival. Note the direction "
            "carefully, because it is not 'the field got stronger'. The field is "
            "stronger in *total* points when the weak rival receives him; what makes "
            "the other arm worse is that the contender turns into a single dominant "
            "team, and championship equity is convex in roster strength, so one "
            "very strong rival costs you more than two moderate ones. That is a "
            "measurement about the shape of CE, not a bidding rule: nothing here "
            "decides what to do about it, and auction behaviour remains out of scope."
        ),
    )


EXPERIMENTS: Dict[str, ExperimentSpec] = {
    e.key: e
    for e in (
        ExperimentSpec("marginal-point", "One additional projected point at different lineup positions",
                       "Is a point of projection worth the same everywhere?", exp_marginal_point),
        ExperimentSpec("second-qb", "Second QB vs equally priced flex alternative",
                       "Does the superflex make a second QB special?", exp_second_qb),
        ExperimentSpec("volatility", "Stable vs volatile at equal mean",
                       "Does weekly variance help or hurt CE?", exp_volatility),
        ExperimentSpec("spikes", "Predictable upside vs unforecastable spikes",
                       "What is forecastability worth?", exp_spikes),
        ExperimentSpec("concentration", "One strong player vs aggregate flex options",
                       "Concentrate or spread a fixed projection budget?", exp_concentration),
        ExperimentSpec("aggregate-lineup-spot",
                       "One stable starter vs several cheaper rotating candidates",
                       "Can cheap candidates replace a starter if their good weeks "
                       "are knowable in advance?", exp_aggregate_lineup_spot),
        ExperimentSpec("injury", "Different injury probabilities",
                       "What does availability risk cost?", exp_injury),
        ExperimentSpec("bench-correlation", "Independent vs correlated bench upside",
                       "Does bench dependence structure matter?", exp_bench_correlation),
        ExperimentSpec("stack", "QB / pass-catcher positive correlation",
                       "Is stacking worth anything in this format?", exp_stack),
        ExperimentSpec("handcuff", "Starting RB / backup RB contingency",
                       "Is contingency timing worth more than raw points?", exp_handcuff),
        ExperimentSpec("opponent-placement",
                       "CONTROL: same strong player on two interchangeable rivals",
                       "Does relabelling which rival owns a stud affect you? "
                       "(it must not)",
                       exp_opponent_placement),
        ExperimentSpec("rival-fit", "Same strong player on two rivals he fits differently",
                       "Can specific rival ownership matter when the recipient "
                       "rosters are meaningfully different?", exp_rival_fit),
    )
}


def run_experiment(
    key: str, n_sims: int, seed: int, base: Optional[RosterSet] = None
) -> ExperimentOutput:
    """Run one named experiment against a baseline league."""
    if key not in EXPERIMENTS:
        raise KeyError(f"unknown experiment {key!r}; choose from {sorted(EXPERIMENTS)}")
    base = base if base is not None else make_synthetic_league()
    return EXPERIMENTS[key].runner(base, n_sims, seed)


def run_all(
    n_sims: int, seed: int, base: Optional[RosterSet] = None,
    keys: Optional[Sequence[str]] = None,
) -> List[ExperimentOutput]:
    base = base if base is not None else make_synthetic_league()
    return [run_experiment(k, n_sims, seed, base) for k in (keys or EXPERIMENTS)]
