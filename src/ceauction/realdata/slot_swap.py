"""The slot-swap experiment: how wide is the model's own answer range?

One roster slot, one replacement, run in all 54 model scenarios. The point is
not the swap -- it is to find out whether the assumption band on a single
marginal roster decision is narrow enough that ranking players is meaningful,
or wide enough that any value has to be reported as a range.

**This experiment does not decide whether an opportunity-cost solver is worth
building.** A best alternative is required under every scenario; a wide band
changes how a valuation is *presented*, not whether opportunity cost exists.
What the band controls is presentation:

* narrow  -- the ordering supports relatively tight values;
* wide    -- values must be reported as scenario-dependent ranges, with the
             dominant assumption named.

**And one comparison proves nothing global.** A stable sign for this pair says
nothing about whether every other pair on the board keeps its order.

Selection is deterministic and fixed *before* any outcome is seen, so there is
no room to pick a swap that produced a convenient answer:

1. the focus team is the integration roster whose baseline CE is nearest the
   median of the twelve (ties to the lower team index);
2. two outgoing players are identified, because one slot cannot speak for a
   roster:
   * the **weakest** FLEX-eligible player by ``base_mean`` -- the existing
     :func:`ceauction.curve.weakest_flex_slot`, the 15th roster spot, which is
     where a $1 auction decision lands;
   * the **marginal starter**: the 6th-ranked FLEX-eligible player, who
     occupies the last of the six non-QB starting slots (RB1, RB2, WT1-3,
     FLEX). That is where real auction dollars go, and it is the slot whose
     value a price actually turns on;
3. the incoming player is the highest-``base_mean`` FLEX-eligible player in the
   mapped pool that nobody rosters, i.e. the best alternative the board
   actually offers;
4. the *adjacent* alternative, used to give the band a scale, is the
   second-best such player.

Steps 2-4 break ties on ``player_id``, which is derived from the canonical
player key, so the whole selection is a pure function of the contract.

The cast -- which twelve teams exist and who is on them -- is chosen once under
:data:`BASELINE_SCENARIO` and then held fixed across all 54 cells. It has to
be: a cast that moved with the scenario would make the arms incomparable. That
choice makes the baseline cell the reference for *cast construction only*, and
gives it no evidential standing over any other cell.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from ..curve import FLEX_ELIGIBLE, weakest_flex_slot
from ..league import DEFAULT_LEAGUE, LeagueSettings, Position
from ..players import PlayerSpec
from ..roster import RosterSet
from ..simulate import simulate_seasons
from .mapping import map_contract_to_playerspecs
from .scenarios import (
    BASELINE_SCENARIO,
    SCENARIO_GRID,
    ArmSpec,
    ModelScenario,
    Pairing,
    ScenarioBand,
    format_band,
    run_scenario_arms,
)
from .sensitivity import MIN_COMMITTED_SIMS
from .smoke import build_test_rosters, roster_assignment

__all__ = [
    "SwapSelection",
    "select_slot_swap",
    "SlotSwapExperiment",
    "run_slot_swap",
    "format_slot_swap",
]


@dataclass(frozen=True)
class SwapSelection:
    """Who was swapped for whom, and by what rule."""

    focus_team: int
    focus_team_name: str
    baseline_ce: Tuple[float, ...]
    outgoing_id: int
    """Weakest FLEX-eligible player: the 15th roster spot."""
    marginal_starter_id: Optional[int]
    """6th-ranked FLEX-eligible player: the last non-QB starting slot."""
    incoming_id: int
    adjacent_id: Optional[int]
    rule: str
    assignment: Tuple[Tuple[int, ...], ...]
    rostered_keys: Tuple[str, ...]
    extra_keys: Tuple[str, ...]

    @property
    def only_keys(self) -> Tuple[str, ...]:
        return self.rostered_keys + self.extra_keys

    def swapped_assignment(self, incoming: int,
                           outgoing: Optional[int] = None
                           ) -> Tuple[Tuple[int, ...], ...]:
        """The assignment with one focus-team player replaced.

        Position in the tuple is preserved so the two arms stay aligned
        slot-for-slot, which keeps every *other* player's roster index --
        and therefore nothing about his random draws, which are keyed by
        player id -- identical between arms.
        """
        outgoing = self.outgoing_id if outgoing is None else outgoing
        return tuple(
            tuple(incoming if pid == outgoing else pid for pid in team)
            if t == self.focus_team else team
            for t, team in enumerate(self.assignment))

    def to_dict(self, names: Optional[Dict[int, str]] = None) -> Dict[str, object]:
        names = names or {}
        return {
            "focus_team": self.focus_team,
            "focus_team_name": self.focus_team_name,
            "baseline_ce_focus": round(self.baseline_ce[self.focus_team], 6),
            "baseline_ce_median": round(
                sorted(self.baseline_ce)[len(self.baseline_ce) // 2], 6),
            "outgoing_player_id": self.outgoing_id,
            "marginal_starter_player_id": self.marginal_starter_id,
            "incoming_player_id": self.incoming_id,
            "adjacent_player_id": self.adjacent_id,
            "outgoing_name": names.get(self.outgoing_id),
            "marginal_starter_name": names.get(self.marginal_starter_id),
            "incoming_name": names.get(self.incoming_id),
            "adjacent_name": names.get(self.adjacent_id),
            "rule": self.rule,
        }


#: Number of non-QB starting slots: RB1, RB2, WT1, WT2, WT3, FLEX. The
#: SUPERFLEX is excluded because a QB may take it, so the sixth-ranked
#: FLEX-eligible player is the last one guaranteed a starting slot.
N_NON_QB_STARTERS = 6

SELECTION_RULE = (
    "focus team = integration roster whose baseline CE is nearest the median "
    "of the twelve (ties to lower index); weakest slot = that team's lowest "
    "base_mean FLEX-eligible player; marginal starter = its 6th-ranked "
    "FLEX-eligible player, the last of the six non-QB starting slots; "
    "incoming = highest base_mean FLEX-eligible unrostered player; adjacent = "
    "the next such player; all ties by player_id. Cast chosen once under the "
    "baseline scenario and held fixed across all cells."
)


def select_slot_swap(
    payload: Dict,
    *,
    positional_miss: Optional[Dict[str, float]] = None,
    positional_cv: Optional[Dict[str, float]] = None,
    limit: int = 300,
    settings: LeagueSettings = DEFAULT_LEAGUE,
    selection_sims: int = 4_000,
    seed: int = 20260904,
    config_overrides: Optional[Dict] = None,
    outgoing_id: Optional[int] = None,
    incoming_id: Optional[int] = None,
    adjacent_id: Optional[int] = None,
    focus_team: Optional[int] = None,
) -> Tuple[SwapSelection, List[PlayerSpec]]:
    """Apply the documented rule, or honour explicit ids.

    The explicit-id path exists because "adjacent player" cannot be defined
    without an arbitrary cutoff on a real board. When ids are supplied the rule
    string records that they were, so no output can imply the choice was
    derived when it was given.

    Returns the selection plus the baseline-scenario specs, which the caller
    needs for names and for a fast smoke run.
    """
    config_overrides = config_overrides or {}
    cfg = BASELINE_SCENARIO.to_mapping_config(**config_overrides)
    mapped = map_contract_to_playerspecs(
        payload, cfg, positional_miss=positional_miss or {},
        positional_cv=positional_cv or {}, limit=limit)
    specs = mapped.specs
    key_of = {m.spec.player_id: m.canonical_key for m in mapped.players}

    rosters = build_test_rosters(specs, settings=settings)
    assignment = roster_assignment(rosters)
    rostered = {pid for team in assignment for pid in team}

    out = simulate_seasons(rosters, selection_sims, seed)
    ce = tuple(float(x) for x in out.championship_equity())

    if focus_team is None:
        # "Nearest the median" over an even count: compare against the mean of
        # the two central values, which is the ordinary definition and does not
        # privilege either side.
        ordered = sorted(ce)
        n = len(ordered)
        median = (ordered[n // 2 - 1] + ordered[n // 2]) / 2.0
        focus_team = min(range(len(ce)), key=lambda t: (abs(ce[t] - median), t))

    flex_on_team = sorted(
        (rosters.spec(pid) for pid in assignment[focus_team]
         if rosters.spec(pid).position in FLEX_ELIGIBLE),
        key=lambda s: (-s.base_mean, s.player_id))
    marginal_starter_id = (
        flex_on_team[N_NON_QB_STARTERS - 1].player_id
        if len(flex_on_team) >= N_NON_QB_STARTERS else None)

    if outgoing_id is None:
        outgoing_id = weakest_flex_slot(rosters, focus_team).player_id
    elif outgoing_id not in assignment[focus_team]:
        raise ValueError(
            f"outgoing player {outgoing_id} is not on team {focus_team}")

    free_flex = sorted(
        (s for s in specs
         if s.player_id not in rostered and s.position in FLEX_ELIGIBLE),
        key=lambda s: (-s.base_mean, s.player_id))
    if incoming_id is None:
        if not free_flex:
            raise ValueError(
                "no unrostered FLEX-eligible player is available; raise --limit "
                "so the mapped pool extends past the 180 rostered players")
        incoming_id = free_flex[0].player_id
        if adjacent_id is None and len(free_flex) > 1:
            adjacent_id = free_flex[1].player_id
    else:
        if incoming_id in rostered:
            raise ValueError(f"incoming player {incoming_id} is already rostered")

    if adjacent_id is not None and adjacent_id in rostered:
        raise ValueError(f"adjacent player {adjacent_id} is already rostered")
    for pid in (incoming_id, adjacent_id):
        if pid is not None and pid not in key_of:
            raise KeyError(f"player {pid} is not in the mapped pool")

    supplied = []
    if outgoing_id is not None and outgoing_id != weakest_flex_slot(
            rosters, focus_team).player_id:
        supplied.append("outgoing")
    if free_flex and incoming_id != free_flex[0].player_id:
        supplied.append("incoming")
    rule = SELECTION_RULE
    if supplied:
        rule += (f" OVERRIDDEN: {', '.join(supplied)} supplied explicitly by the "
                 f"caller rather than derived.")

    extra = tuple(key_of[p] for p in (incoming_id, adjacent_id) if p is not None)
    selection = SwapSelection(
        focus_team=focus_team, focus_team_name=rosters.team_names[focus_team],
        baseline_ce=ce, outgoing_id=outgoing_id,
        marginal_starter_id=marginal_starter_id, incoming_id=incoming_id,
        adjacent_id=adjacent_id, rule=rule, assignment=assignment,
        rostered_keys=tuple(key_of[p] for p in sorted(rostered)),
        extra_keys=extra)
    return selection, specs


@dataclass
class SlotSwapExperiment:
    """Three bands: two slots, plus the adjacent-player scale.

    ``bench_band``   replacing the 15th roster spot -- a $1 decision.
    ``starter_band`` replacing the last non-QB starting slot -- where auction
                     dollars actually go.
    ``adjacent_band`` taking the second-best available player instead of the
                     best, at the bench slot. This is the CE distance between
                     two neighbouring players and is the scale against which a
                     band width has to be read.

    Two slots are run because one is not evidence about a roster, let alone
    about a board. Two is still not a proof of global rank stability; it is two
    data points that happen to sit at opposite ends of the roster.
    """

    selection: SwapSelection
    bench_band: ScenarioBand
    starter_band: Optional[ScenarioBand]
    adjacent_band: Optional[ScenarioBand]
    player_names: Dict[int, str]
    total_runtime_s: float

    @property
    def bands(self) -> Dict[str, ScenarioBand]:
        out = {"bench_slot": self.bench_band}
        if self.starter_band is not None:
            out["marginal_starter_slot"] = self.starter_band
        if self.adjacent_band is not None:
            out["adjacent_alternative"] = self.adjacent_band
        return out

    @property
    def adjacent_gap(self) -> Optional[float]:
        """Largest absolute CE move from taking the next player instead.

        The most favourable reading of the gap, so the ratio below is not
        flattered by picking the smallest cell.
        """
        if self.adjacent_band is None:
            return None
        return max(abs(self.adjacent_band.min_delta),
                   abs(self.adjacent_band.max_delta))

    def band_ratio(self, band: ScenarioBand) -> Optional[float]:
        """Assumption band width divided by the adjacent-player CE gap.

        Below 1: the model's own range of answers is narrower than the distance
        between two neighbouring players, so an ordering survives the
        assumptions. At or above 1: it does not, and values for that slot must
        be reported as scenario-dependent ranges.

        ``None`` when the gap itself is unmeasurable, which is a real outcome
        rather than a missing number: if two adjacent players are
        indistinguishable, no ratio against them is meaningful.
        """
        gap = self.adjacent_gap
        if gap is None or gap <= 0:
            return None
        return band.band_width / gap

    def presentation_verdict(self) -> str:
        """``"narrow"``, ``"wide"`` or ``"unmeasurable"``.

        Governs how a value is *presented*. It does not govern whether a
        best-alternative solver is built: opportunity cost exists under every
        scenario, and a wide band makes the output a range rather than making
        the question go away.
        """
        ratios = [r for r in (self.band_ratio(b) for b in self.bands.values()
                              if b is not self.adjacent_band)
                  if r is not None]
        if not ratios:
            return "unmeasurable"
        return "narrow" if max(ratios) < 1.0 else "wide"

    def summary(self) -> Dict[str, object]:
        return {
            "selection": self.selection.to_dict(self.player_names),
            "bands": {k: b.summary() for k, b in self.bands.items()},
            "adjacent_gap": (round(self.adjacent_gap, 6)
                             if self.adjacent_gap is not None else None),
            "band_width_over_adjacent_gap": {
                k: (round(self.band_ratio(b), 4)
                    if self.band_ratio(b) is not None else None)
                for k, b in self.bands.items() if b is not self.adjacent_band},
            "presentation_verdict": self.presentation_verdict(),
            "total_runtime_s": round(self.total_runtime_s, 1),
            "warnings": [
                "Two slot comparisons on one roster. This does NOT establish "
                "that every player ranking on the board is robust.",
                "The verdict governs how values are PRESENTED. A "
                "best-alternative solver is required under every scenario; a "
                "wide band makes the output a range, it does not remove "
                "opportunity cost.",
            ],
        }


def run_slot_swap(
    payload: Dict,
    *,
    positional_miss: Optional[Dict[str, float]] = None,
    positional_cv: Optional[Dict[str, float]] = None,
    limit: int = 300,
    n_sims: int = MIN_COMMITTED_SIMS,
    seed: int = 20260904,
    scenarios: Sequence[ModelScenario] = SCENARIO_GRID,
    settings: LeagueSettings = DEFAULT_LEAGUE,
    selection_sims: int = 4_000,
    config_overrides: Optional[Dict] = None,
    with_adjacent: bool = True,
    with_starter: bool = True,
    progress=None,
    **selection_overrides,
) -> SlotSwapExperiment:
    """Select the swaps, then run every arm in every scenario.

    All arms share one mapping and one seed per cell, so every pairing below is
    properly paired and the baseline is simulated once rather than once per
    comparison.
    """
    t0 = time.perf_counter()
    selection, specs = select_slot_swap(
        payload, positional_miss=positional_miss, positional_cv=positional_cv,
        limit=limit, settings=settings, selection_sims=selection_sims, seed=seed,
        config_overrides=config_overrides, **selection_overrides)
    names = {s.player_id: s.name for s in specs}

    arms = [ArmSpec("baseline", selection.assignment),
            ArmSpec("bench-slot-upgraded",
                    selection.swapped_assignment(selection.incoming_id))]
    pairings = [Pairing(
        label="bench_slot", arm_a="baseline", arm_b="bench-slot-upgraded",
        notes=("Arm B replaces the focus team's WEAKEST FLEX-eligible player "
               "-- the 15th roster spot -- with the best FLEX-eligible player "
               "nobody rosters."))]

    do_adjacent = with_adjacent and selection.adjacent_id is not None
    if do_adjacent:
        arms.append(ArmSpec(
            "bench-slot-next-best",
            selection.swapped_assignment(selection.adjacent_id)))
        pairings.append(Pairing(
            label="adjacent_alternative", arm_a="bench-slot-upgraded",
            arm_b="bench-slot-next-best",
            notes=("Arm B takes the SECOND-best available replacement instead "
                   "of the best. This is the CE distance between two "
                   "neighbouring players on the board.")))

    do_starter = (with_starter and selection.marginal_starter_id is not None
                  and selection.marginal_starter_id != selection.outgoing_id)
    if do_starter:
        arms.append(ArmSpec(
            "marginal-starter-replaced",
            selection.swapped_assignment(selection.incoming_id,
                                         selection.marginal_starter_id)))
        pairings.append(Pairing(
            label="marginal_starter_slot", arm_a="baseline",
            arm_b="marginal-starter-replaced",
            notes=("Arm B replaces the focus team's MARGINAL STARTER -- the "
                   "6th-ranked FLEX-eligible player, holding the last non-QB "
                   "starting slot -- with the best available player. The sign "
                   "is expected to be negative: this is a downgrade, and its "
                   "size is the replacement value of a starting slot.")))

    bands = run_scenario_arms(
        payload, arms, pairings, only_keys=selection.only_keys,
        team_index=selection.focus_team, positional_miss=positional_miss,
        positional_cv=positional_cv, scenarios=scenarios, n_sims=n_sims,
        seed=seed, settings=settings, config_overrides=config_overrides,
        progress=progress)

    return SlotSwapExperiment(
        selection=selection,
        bench_band=bands["bench_slot"],
        starter_band=bands.get("marginal_starter_slot"),
        adjacent_band=bands.get("adjacent_alternative"),
        player_names=names, total_runtime_s=time.perf_counter() - t0)


def format_slot_swap(exp: SlotSwapExperiment, width: int = 96,
                     show_names: bool = False) -> str:
    """Sanitized rendering. Player names appear only when explicitly asked for."""
    bar = "=" * width
    sel = exp.selection

    def who(pid):
        if pid is None:
            return "none"
        tag = f"id {pid}"
        return tag + (f"  {exp.player_names.get(pid, '')}" if show_names else "")

    out = [bar, "54-CELL SLOT-SWAP EXPERIMENT", bar, "",
           "SELECTION (fixed before any outcome was seen)",
           f"  rule              {sel.rule}",
           f"  focus team        #{sel.focus_team} {sel.focus_team_name} "
           f"(baseline CE {sel.baseline_ce[sel.focus_team]:.4f})",
           f"  bench slot out    {who(sel.outgoing_id)}",
           f"  starter slot out  {who(sel.marginal_starter_id)}",
           f"  incoming          {who(sel.incoming_id)}",
           f"  adjacent          {who(sel.adjacent_id)}",
           ""]
    if not show_names:
        out += ["  Player names and projection values are withheld from committed",
                "  output; ids are stable hashes of the canonical player key.", ""]

    out.append(format_band(exp.bench_band, width))
    if exp.starter_band is not None:
        out += ["", format_band(exp.starter_band, width)]
    if exp.adjacent_band is not None:
        out += ["", format_band(exp.adjacent_band, width, show_cells=False)]

    gap = exp.adjacent_gap
    out += ["", bar, "WHAT THE BANDS MEAN FOR PRESENTATION", bar]
    if gap is None or gap <= 0:
        out += ["  The adjacent-player CE gap is not measurable at this sample",
                "  size, so no ratio against it is meaningful. That is itself a",
                "  finding: two neighbouring players on this board are",
                "  indistinguishable to the model at this roster slot."]
    else:
        out.append(f"  CE gap to the adjacent alternative     {gap:.5f}")
        for name, band in exp.bands.items():
            if band is exp.adjacent_band:
                continue
            r = exp.band_ratio(band)
            out.append(f"  {name:<36} band {band.band_width:.5f}"
                       f"   ratio {r:.2f}" if r is not None else
                       f"  {name:<36} band {band.band_width:.5f}")
    verdict = exp.presentation_verdict()
    out.append("")
    if verdict == "narrow":
        out += ["  NARROW: the model's own range of answers is smaller than the",
                "  distance between two neighbouring players, so an ordering at",
                "  these slots survives the unresolved assumptions and",
                "  relatively tight values can be quoted for them."]
    elif verdict == "wide":
        out += ["  WIDE: the model's own range of answers is at least as large as",
                "  the distance between two neighbouring players. Values must be",
                "  reported as scenario-dependent ranges naming the dominant",
                "  assumption, not as points."]
    else:
        out += ["  UNMEASURABLE at this sample size: the comparison this band",
                "  would be judged against is itself indistinguishable from zero."]
    out += ["",
            "  Either way a best-alternative solver is still required. A wide",
            "  band changes how a value is PRESENTED, not whether opportunity",
            "  cost exists -- passing on a player still means spending the money",
            "  on someone, under every scenario in the grid.",
            "",
            "  AND THESE ARE TWO COMPARISONS ON ONE ROSTER. They do not",
            "  establish that every player ranking on the board is robust.",
            bar]
    return "\n".join(out)
