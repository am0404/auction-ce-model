"""Buying a player, versus passing and spending the money elsewhere.

This is the comparison the whole package exists for. A player's worth is not a
property of the player; it is

    CE(best roster reachable after buying X for $p)
      minus
    CE(best roster reachable after passing on X and keeping the $p)

and neither of those rosters exists until something searches for it. Both
branches therefore run a full completion search, and both are simulated against
the same league.

**The pass branch must actually remove X from our alternatives.** Otherwise the
search would happily re-buy him in the branch where we declined him, the two
branches would converge, and every player would look worthless.

**Who gets him matters.** Passing to the strongest rival and passing to a team
with no roster room are different futures for us even though our own roster is
identical in both. The pass destination is an explicit input:

``unavailable``  X leaves the board. Nobody has him.
``rival``        X goes to one named opponent at a stated price. That opponent
                 pays, loses a roster slot, and re-completes his own roster
                 from what is left -- so the money he spent on X is money he no
                 longer has for anyone else.

Which opponent *would* win him is not predicted here. That is a behavioural
model, this package does not contain one, and inventing one silently would turn
a stated assumption into an apparent finding.

**Order matters and is stated.** In the rival branch the opponent buys first --
he has just won the player -- and completes his roster before the focus team
completes its own. That is the sequence the auction actually has.

**Both branches choose their roster by championship equity, and the comparison
is made on an independent sample.** An earlier version ran both completions
with ``evaluate_ce=False``, so each branch picked the roster its *expected
points* proxy liked and only then simulated that single roster. What it
computed was ``CE(proxy-selected completion)``, not ``CE(best modelled
completion)``, while its documentation claimed the latter. Selection now runs
on one sample and the final paired difference on a second, unrelated one --
because choosing the maximum of several noisy estimates and then quoting that
same estimate is biased upward, and the bias would propagate into every price
and every reservation frontier built on it.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..ce import paired_se
from ..simulate import simulate_seasons
from .completion import (ComparisonCast, CompletionResult, CompletionSettings,
                         _build_roster_set, complete_roster)
from .costs import CostBook
from .proxy import ProxyEvaluator
from .state import AuctionRuleError, AuctionState

__all__ = [
    "PassDestination",
    "BuyPassResult",
    "compare_buy_vs_pass",
    "format_buy_pass",
]


@dataclass(frozen=True)
class PassDestination:
    """Where the candidate goes if we decline him. An input, never a prediction."""

    kind: str
    rival_owner_id: Optional[str] = None
    rival_price: Optional[int] = None

    def __post_init__(self) -> None:
        if self.kind not in ("unavailable", "rival"):
            raise ValueError(f"unknown pass destination {self.kind!r}")
        if self.kind == "rival":
            if not self.rival_owner_id:
                raise ValueError("a rival destination needs an owner id")
            if self.rival_price is None:
                raise ValueError(
                    "a rival destination needs the price he pays; what he spends "
                    "is money he no longer has, which is half the point")
            if not isinstance(self.rival_price, int) or isinstance(self.rival_price, bool):
                raise ValueError("rival_price must be whole auction dollars")

    @classmethod
    def unavailable(cls) -> "PassDestination":
        return cls("unavailable")

    @classmethod
    def to_rival(cls, owner_id: str, price: int) -> "PassDestination":
        return cls("rival", owner_id, int(price))

    @property
    def label(self) -> str:
        if self.kind == "unavailable":
            return "unavailable (off the board)"
        return f"{self.rival_owner_id} at ${self.rival_price}"

    def cache_key(self) -> Tuple:
        return (self.kind, self.rival_owner_id, self.rival_price)

    def to_dict(self) -> Dict[str, object]:
        return {"kind": self.kind, "rival_owner_id": self.rival_owner_id,
                "rival_price": self.rival_price, "label": self.label}


@dataclass
class BuyPassResult:
    """Both branches, their difference, and everything needed to doubt it."""

    candidate_id: int
    price: int
    focus_owner_id: str
    pass_destination: PassDestination
    scenario_id: Optional[str]
    cost_level: str
    auction_fingerprint: str

    ce_buy: float
    ce_pass: float
    delta_ce: float
    delta_ce_se: float
    discordance: float
    n_sims: int
    """Seasons in the holdout sample the difference was measured on."""
    selection_sims: int
    """Seasons used inside each branch to choose its completion."""

    buy: CompletionResult
    pass_: CompletionResult
    runtime_s: float
    notes: str = ""

    @property
    def ci95(self) -> Tuple[float, float]:
        if math.isnan(self.delta_ce_se):
            return (float("nan"), float("nan"))
        half = 1.96 * self.delta_ce_se
        return (self.delta_ce - half, self.delta_ce + half)

    @property
    def resolved(self) -> bool:
        lo, hi = self.ci95
        if math.isnan(lo):
            return False
        return lo > 0.0 or hi < 0.0

    @property
    def verdict(self) -> str:
        """``favorable``, ``unfavorable`` or ``unresolved`` at this price."""
        lo, hi = self.ci95
        if math.isnan(lo):
            return "unresolved"
        if lo >= 0.0:
            return "favorable"
        if hi < 0.0:
            return "unfavorable"
        return "unresolved"

    @property
    def is_heuristic(self) -> bool:
        return not (self.buy.diagnostics.is_exact and self.pass_.diagnostics.is_exact)

    @property
    def divergence(self) -> Dict[str, List[int]]:
        """Which alternative purchases the two branches disagree about.

        The concrete answer to "what am I giving up": players the pass branch
        buys with the money the buy branch spent on the candidate, and players
        the buy branch still finds room for.
        """
        buy_ids = set(self.buy.best.added) if self.buy.best else set()
        pass_ids = set(self.pass_.best.added) if self.pass_.best else set()
        return {
            "only_in_buy_branch": sorted(buy_ids - pass_ids),
            "only_in_pass_branch": sorted(pass_ids - buy_ids),
            "in_both": sorted(buy_ids & pass_ids),
        }

    def to_dict(self) -> Dict[str, object]:
        lo, hi = self.ci95
        return {
            "candidate_id": self.candidate_id,
            "price": self.price,
            "focus_owner_id": self.focus_owner_id,
            "pass_destination": self.pass_destination.to_dict(),
            "scenario_id": self.scenario_id,
            "cost_level": self.cost_level,
            "auction_fingerprint": self.auction_fingerprint,
            "ce_buy": round(self.ce_buy, 6),
            "ce_pass": round(self.ce_pass, 6),
            "delta_ce": round(self.delta_ce, 6),
            "delta_ce_se": round(self.delta_ce_se, 6),
            "ci95": [round(lo, 6), round(hi, 6)],
            "discordance": round(self.discordance, 5),
            "verdict": self.verdict,
            "resolved": self.resolved,
            "evaluation_sims": self.n_sims,
            "selection_sims": self.selection_sims,
            "n_sims": self.n_sims,
            "selection_basis": {"buy": self.buy.selection_basis,
                                "pass": self.pass_.selection_basis},
            "search_is": "heuristic" if self.is_heuristic else "exact",
            "interval_is": "pointwise 95%, not simultaneous across prices "
                           "or scenarios",
            "divergence": self.divergence,
            "buy_branch": self.buy.to_dict(),
            "pass_branch": self.pass_.to_dict(),
            "runtime_s": round(self.runtime_s, 2),
            "notes": self.notes,
        }


def compare_buy_vs_pass(
    state: AuctionState,
    cast: ComparisonCast,
    costs: CostBook,
    candidate_id: int,
    price: int,
    pass_destination: PassDestination,
    *,
    settings: CompletionSettings = CompletionSettings(),
    scenario_id: Optional[str] = None,
    default_cost: Optional[int] = None,
    proxy: Optional[ProxyEvaluator] = None,
    notes: str = "",
) -> BuyPassResult:
    """Run both branches over matched seasons and difference them.

    Matched seasons is the whole design: the two branches share a seed and
    every player's draws are keyed by his own id, so a player on both rosters
    is byte-identical between them and contributes an exact zero to the paired
    difference.
    """
    t0 = time.perf_counter()
    focus_id = state.focus_owner_id
    if not state.is_available(candidate_id):
        raise AuctionRuleError(
            f"candidate {candidate_id} is not available in this auction state")
    problem = state.purchase_shortfall(candidate_id, focus_id, price)
    if problem is not None:
        raise AuctionRuleError(f"cannot evaluate buying at ${price}: {problem}")
    # The comparison cast assigns each rival a full fifteen, some of whom the
    # auction state has not actually sold yet -- they are that rival's assumed
    # continuation. Such a player is "available" in the room and simultaneously
    # spoken for in the cast, and buying him would put the same person on two
    # teams in the simulated league. Refusing is right: the counterfactual
    # "I buy him AND he is still on their roster" is not a possible auction.
    if candidate_id in cast.rival_ids:
        holder = next(name for name, team in zip(cast.team_names, cast.rosters)
                      if candidate_id in team)
        raise AuctionRuleError(
            f"player {candidate_id} is available in the room but the comparison "
            f"cast already assigns him to {holder} as part of that owner's "
            f"assumed continuation. Buying him would put one player on two "
            f"teams. Either choose a candidate the cast leaves free, or supply "
            f"a cast that does not claim him.")

    if proxy is None:
        proxy = ProxyEvaluator(state.pool, state.settings, settings.proxy_reps,
                               settings.proxy_seed)

    # --- buy branch --------------------------------------------------------
    buy_state = state.apply_purchase(candidate_id, focus_id, price)
    buy = complete_roster(buy_state, cast, costs, settings=settings,
                          owner_id=focus_id, evaluate_ce=True,
                          default_cost=default_cost, proxy=proxy,
                          reserved_ids=cast.reserved_for(cast.focus_team_index),
                          notes="buy branch")

    # --- pass branch -------------------------------------------------------
    pass_cast = cast
    rival_note = ""
    if pass_destination.kind == "unavailable":
        pass_state = state.withdraw(candidate_id)
    else:
        rival = pass_destination.rival_owner_id
        if rival == focus_id:
            raise AuctionRuleError(
                "the pass destination must be an opponent, not the focus owner")
        bad = state.purchase_shortfall(candidate_id, rival,
                                       pass_destination.rival_price)
        if bad is not None:
            raise AuctionRuleError(
                f"{rival} cannot legally take the candidate at "
                f"${pass_destination.rival_price}: {bad}")
        pass_state = state.award_to_rival(candidate_id, rival,
                                          pass_destination.rival_price)
        # The opponent has just won the player, so he completes first: the
        # money he spent is money he cannot spend on anyone else, and the
        # players he takes are players we cannot.
        #
        # His continuation is chosen for HIS equity, not ours. Optimising his
        # roster for our benefit would be a strange counterfactual; the
        # `rival_selection` setting can fall back to the proxy for speed, and
        # the result says which was used rather than claiming the stronger one.
        rival_index = _team_index_of(cast, state, rival)
        rival_ce = settings.rival_selection == "ce"
        rival_cast = cast
        rival_reserved = cast.reserved_for(rival_index)
        if rival_ce:
            # Measuring the rival's own equity needs a complete league, and our
            # slot is still a placeholder at this point -- we do not complete
            # until after he has bought. So he is compared against our
            # PROXY-best roster from this same state: a concrete, legal,
            # disjoint opponent rather than a placeholder that would let the
            # same player appear on two teams.
            provisional = complete_roster(
                pass_state, cast, costs, settings=settings, owner_id=focus_id,
                evaluate_ce=False, default_cost=default_cost, proxy=proxy,
                reserved_ids=cast.reserved_for(cast.focus_team_index),
                notes="provisional focus roster, proxy-selected")
            if provisional.best is not None:
                rival_cast = cast.with_team(cast.focus_team_index,
                                            provisional.best.roster)
                rival_reserved = rival_cast.reserved_for(rival_index,
                                                         include_focus=True)
            else:
                rival_ce = False
        rival_fill = complete_roster(
            pass_state, rival_cast, costs, settings=settings, owner_id=rival,
            evaluate_ce=rival_ce, default_cost=default_cost, proxy=proxy,
            reserved_ids=rival_reserved, ce_team_index=rival_index,
            notes="rival continuation")
        if rival_fill.best is None:
            raise AuctionRuleError(
                f"{rival} has no legal completion after taking the candidate at "
                f"${pass_destination.rival_price}; the counterfactual is not a "
                f"legal auction and is refused rather than approximated")
        pass_cast = cast.with_team(rival_index, rival_fill.best.roster)
        rival_note = (f"{rival} re-completed after paying "
                      f"${pass_destination.rival_price}; his continuation was "
                      f"chosen by {rival_fill.selection_basis}"
                      + (" for HIS OWN equity, against our proxy-best roster"
                         if rival_ce else " (expected points only)"))

    pass_ = complete_roster(pass_state, pass_cast, costs, settings=settings,
                            owner_id=focus_id, evaluate_ce=True,
                            default_cost=default_cost, proxy=proxy,
                            reserved_ids=pass_cast.reserved_for(
                                pass_cast.focus_team_index),
                            notes="pass branch")

    if buy.best is None or pass_.best is None:
        which = []
        if buy.best is None:
            which.append(f"buy branch ({buy.diagnostics.stop_reason})")
        if pass_.best is None:
            which.append(f"pass branch ({pass_.diagnostics.stop_reason})")
        raise AuctionRuleError(
            f"no legal completion in the {' and the '.join(which)}; a buy/pass "
            f"comparison needs both, and substituting an illegal roster would "
            f"answer a different question. Raising candidate_pool, or supplying "
            f"cheaper costs, is the usual fix.")

    # --- the paired comparison, on the holdout sample ----------------------
    #
    # Each branch chose its roster on `selection_sims` at `selection_seed`.
    # The difference between them is measured on `evaluation_sims` at
    # `evaluation_seed`, which neither branch saw while choosing. Both arms
    # share that seed, so a player on both rosters draws identical numbers and
    # contributes an exact zero to the paired difference.
    rs_buy = _build_roster_set(buy_state, cast, buy.best.roster)
    rs_pass = _build_roster_set(pass_state, pass_cast, pass_.best.roster)
    out_buy = simulate_seasons(rs_buy, settings.evaluation_sims,
                               settings.evaluation_seed, settings.ce_chunk)
    out_pass = simulate_seasons(rs_pass, settings.evaluation_sims,
                                settings.evaluation_seed, settings.ce_chunk)
    t = cast.focus_team_index
    ind_buy = out_buy.champion_indicator(t)
    ind_pass = out_pass.champion_indicator(t)
    d = ind_buy - ind_pass

    buy = replace(buy, best=replace(buy.best, ce=float(ind_buy.mean())))
    pass_ = replace(pass_, best=replace(pass_.best, ce=float(ind_pass.mean())))

    return BuyPassResult(
        candidate_id=candidate_id, price=price, focus_owner_id=focus_id,
        pass_destination=pass_destination, scenario_id=scenario_id,
        cost_level=costs.level, auction_fingerprint=state.fingerprint(),
        ce_buy=float(ind_buy.mean()), ce_pass=float(ind_pass.mean()),
        delta_ce=float(d.mean()), delta_ce_se=paired_se(d),
        discordance=float(np.mean(d != 0.0)),
        n_sims=settings.evaluation_sims,
        selection_sims=settings.selection_sims,
        buy=buy, pass_=pass_, runtime_s=time.perf_counter() - t0,
        notes=" ".join(x for x in (notes, rival_note) if x))


def _team_index_of(cast: ComparisonCast, state: AuctionState,
                   owner_id: str) -> int:
    """Which cast slot belongs to this owner.

    The cast's team names are the owner ids by construction in this package;
    anything else is a caller mistake worth catching loudly rather than
    silently comparing the wrong team.
    """
    for i, name in enumerate(cast.team_names):
        if name == owner_id:
            return i
    raise AuctionRuleError(
        f"owner {owner_id!r} has no slot in the comparison cast "
        f"({list(cast.team_names)})")


def format_buy_pass(result: BuyPassResult, width: int = 88,
                    cost_disclaimer: str = "") -> str:
    """Sanitized rendering: ids, dollars and CE; no names, no projections."""
    bar = "=" * width
    lo, hi = result.ci95
    out = [bar, "BUY VERSUS PASS", bar,
           f"candidate         id {result.candidate_id}",
           f"price             ${result.price}",
           f"focus owner       {result.focus_owner_id}",
           f"if we pass        {result.pass_destination.label}",
           f"scenario          {result.scenario_id or 'not stated'}",
           f"cost source       {result.cost_level}"]
    if cost_disclaimer:
        out.append(f"                  {cost_disclaimer}")
    out += [f"auction state     {result.auction_fingerprint}",
            f"selection         {result.selection_sims:,} seasons, used inside "
            f"each branch to CHOOSE its completion by equity",
            f"                  buy branch selected by "
            f"{result.buy.selection_basis}; pass branch by "
            f"{result.pass_.selection_basis}",
            f"evaluation        {result.n_sims:,} seasons on an INDEPENDENT "
            f"holdout, matched across",
            f"                  both arms by common random numbers. The "
            f"difference below is",
            f"                  measured here, not on the sample that chose "
            f"the rosters.",
            "",
            f"  CE if we buy        {result.ce_buy:.5f}",
            f"  CE if we pass       {result.ce_pass:.5f}",
            f"  paired difference   {result.delta_ce:+.5f}  "
            f"+/- {1.96 * result.delta_ce_se:.5f}",
            f"  95% interval        [{lo:+.5f}, {hi:+.5f}]",
            f"  seasons that differ {result.discordance:.2%}",
            f"  verdict at ${result.price}      {result.verdict.upper()}",
            ""]
    if result.notes:
        out += [f"note              {result.notes}", ""]
    div = result.divergence
    out += ["WHAT THE BRANCHES DISAGREE ABOUT",
            f"  bought only if we buy him   {div['only_in_buy_branch']}",
            f"  bought only if we pass      {div['only_in_pass_branch']}",
            f"  bought either way           {len(div['in_both'])} player(s)",
            "",
            "  The second line is the opportunity cost, made concrete: those are",
            "  the players the money buys instead.", ""]
    if result.is_heuristic:
        out += ["Both completions are BOUNDED SEARCHES, not optima. The difference",
                "is between the best rosters this search found, not between the",
                "best rosters that exist.", ""]
    if not result.resolved:
        out += [f"UNRESOLVED at {result.n_sims:,} seasons: the interval contains",
                "zero, so this run cannot say which branch is better.", ""]
    out += ["This is a CE RESERVATION comparison. It is not a prediction of what",
            "the room will pay, and it is not a bid.", bar]
    return "\n".join(out)
