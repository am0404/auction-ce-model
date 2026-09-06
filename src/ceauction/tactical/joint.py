"""Joint allocation: one reconciled league, with every player accounted for.

The previous continuation gave the focus team a shadow ledger so rivals had to
outbid it. That fixed the silent-seat defect and introduced a worse one.

The focus team shadow-won twelve players for a notional $108, and its CE-backed
completion then bought **six** of them. The other six were removed from every
rival's reach during the board loop, were never offered back, and cost us
nothing. We were holding six mutually exclusive options, denying them to eleven
opponents, and paying only for the ones we later chose. That inflates our roster
(we pick from a protected shortlist), inflates our denial value (rivals lose
players we never bought) and deflates every rival (they complete without them).

This module replaces it with **conditional reallocation**:

1. A *shadow pass* discovers which players we would plausibly contest. It is a
   search aid and nothing more; nothing it does is final.
2. The CE completion search picks finalists from that board.
3. For each finalist, we **actually buy** exactly the players it names, at
   cost-book prices, out of our real budget and into our real roster.
4. Every player we did not buy -- shadow-held or not -- goes back on the board.
5. The eleven rivals then complete **conditional on that**, competing for
   everything we left, including every unselected shadow hold.
6. The resulting world is validated against the conservation invariant and only
   then evaluated for championship equity.

The invariant, stated once and enforced by :func:`validate_joint_world`: for
every player in the pool exactly one of these is true -- he was owned before the
counterfactual, he was acquired in the nominated-player branch and paid for, he
is rostered and paid for by us, he is rostered and paid for by another owner, or
he is undrafted and was available on equal terms to every unfinished team. **A
shadow bid alone is not a disposition.**
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

from ..auction.completion import (ComparisonCast, Completion,
                                  CompletionSettings, complete_roster)
from ..auction.costs import CostBook
from ..auction.proxy import ProxyEvaluator
from ..auction.state import AuctionState
from ..market.live import MarketState
from .board import (BoardResult, BoardSettings, cast_from_board,
                    continue_shared_board)

__all__ = [
    "ConservationError",
    "ShadowAudit",
    "ConservationReport",
    "JointWorld",
    "audit_shadow_holds",
    "build_joint_world",
    "validate_joint_world",
    "format_shadow_audit",
    "format_conservation",
    "JointArm",
    "JointComparison",
    "build_joint_worlds",
    "evaluate_joint_arm",
]


class ConservationError(AssertionError):
    """A world that does not reconcile. Never evaluated, never reported."""


# ---------------------------------------------------------------------------
# Audit of the old shadow mechanism
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ShadowAudit:
    """What the shadow ledger did, and what it never paid for."""

    focus_owner_id: str
    shadow_bid: Tuple[int, ...]
    """Players the focus team bid on and won in the shadow pass."""
    shadow_prices: Dict[int, int]
    shadow_budget_before: int
    shadow_budget_after: int
    shadow_slots_before: int
    shadow_slots_after: int

    selected_held: Tuple[int, ...]
    """Held players the evaluated completion actually buys."""
    unselected_held: Tuple[int, ...]
    """Held players it does not. Every one of these is a candidate free block."""

    unselected_reaching_a_rival: Tuple[int, ...]
    unselected_left_undrafted: Tuple[int, ...]
    rivals_recompleted: bool

    shadow_cost_total: int
    shadow_cost_unselected: int
    actual_focus_cost: int

    @property
    def free_blocks(self) -> Tuple[int, ...]:
        """Held, unbought, and never offered back to a rival. The defect."""
        if self.rivals_recompleted:
            return ()
        return tuple(p for p in self.unselected_held
                     if p not in self.unselected_reaching_a_rival)

    @property
    def free_block_cost(self) -> int:
        return sum(self.shadow_prices.get(p, 0) for p in self.free_blocks)

    @property
    def cost_divergence(self) -> int:
        """Shadow spend minus what we actually paid. Nonzero is a smell."""
        return self.shadow_cost_total - self.actual_focus_cost

    @property
    def clean(self) -> bool:
        return not self.free_blocks

    def to_dict(self) -> Dict[str, object]:
        return {
            "focus_owner_id": self.focus_owner_id,
            "shadow_won": list(self.shadow_bid),
            "shadow_prices": {str(k): v for k, v in
                              sorted(self.shadow_prices.items())},
            "shadow_budget": {"before": self.shadow_budget_before,
                              "after": self.shadow_budget_after},
            "shadow_slots": {"before": self.shadow_slots_before,
                             "after": self.shadow_slots_after},
            "selected_held": list(self.selected_held),
            "unselected_held": list(self.unselected_held),
            "unselected_reaching_a_rival":
                list(self.unselected_reaching_a_rival),
            "unselected_left_undrafted": list(self.unselected_left_undrafted),
            "rivals_recompleted": self.rivals_recompleted,
            "shadow_cost_total": self.shadow_cost_total,
            "shadow_cost_unselected": self.shadow_cost_unselected,
            "actual_focus_cost": self.actual_focus_cost,
            "cost_divergence": self.cost_divergence,
            "free_blocks": list(self.free_blocks),
            "n_free_blocks": len(self.free_blocks),
            "free_block_cost": self.free_block_cost,
            "clean": self.clean,
        }


def audit_shadow_holds(
    state: AuctionState,
    cast: ComparisonCast,
    costs: CostBook,
    *,
    board_settings: BoardSettings = BoardSettings(),
    completion: CompletionSettings = CompletionSettings(),
    market: Optional[MarketState] = None,
    key_by_id: Optional[Dict[int, str]] = None,
    proxy: Optional[ProxyEvaluator] = None,
    protect: Sequence[int] = (),
    default_cost: int = 1,
    rivals_recompleted: bool = False,
) -> ShadowAudit:
    """Reproduce the shadow pass and report every hold's disposition.

    ``rivals_recompleted`` says whether the caller went on to re-offer the
    unselected holds. The old path did not, which is what turned a hold into a
    free block; the audit reports the fact rather than assuming it.
    """
    focus = state.focus_owner_id
    owner = state.owner(focus)
    board = continue_shared_board(state, settings=board_settings, costs=costs,
                                  market=market, key_by_id=key_by_id,
                                  protect=protect)
    shadow_prices = {a.player_id: a.price for a in board.allocations
                     if a.owner_id == focus}
    held = tuple(sorted(board.held_for_focus))

    new_cast = cast_from_board(board, cast, state)
    if proxy is None:
        proxy = ProxyEvaluator(state.pool, state.settings,
                               completion.proxy_reps, completion.proxy_seed)
    res = complete_roster(board.state, new_cast, costs, settings=completion,
                          owner_id=focus, evaluate_ce=False,
                          default_cost=default_cost, proxy=proxy,
                          reserved_ids=board.reserved_ids(exclude_owner=focus),
                          notes="shadow audit completion")
    chosen = set(res.best.roster) if res.best is not None else set()
    rival_ids = {pid for o in board.state.owners if o.owner_id != focus
                 for pid in o.player_ids}
    selected = tuple(p for p in held if p in chosen)
    unselected = tuple(p for p in held if p not in chosen)
    spend = sum(shadow_prices.values())
    return ShadowAudit(
        focus_owner_id=focus, shadow_bid=held, shadow_prices=shadow_prices,
        shadow_budget_before=owner.budget_remaining,
        shadow_budget_after=owner.budget_remaining - spend,
        shadow_slots_before=owner.open_slots,
        shadow_slots_after=owner.open_slots - len(held),
        selected_held=selected, unselected_held=unselected,
        unselected_reaching_a_rival=tuple(p for p in unselected
                                          if p in rival_ids),
        unselected_left_undrafted=tuple(p for p in unselected
                                        if p not in rival_ids),
        rivals_recompleted=rivals_recompleted,
        shadow_cost_total=spend,
        shadow_cost_unselected=sum(shadow_prices.get(p, 0) for p in unselected),
        actual_focus_cost=res.best.added_cost if res.best is not None else 0)


# ---------------------------------------------------------------------------
# The reconciled world
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConservationReport:
    """Every accounting identity a joint world must satisfy."""

    n_pool: int
    n_initially_owned: int
    n_branch_acquired: int
    n_focus_bought: int
    n_rival_bought: int
    n_undrafted: int

    duplicate_owners: Tuple[int, ...]
    over_budget: Tuple[str, ...]
    reserve_violations: Tuple[str, ...]
    illegal_rosters: Tuple[str, ...]
    wrong_size: Tuple[str, ...]
    unpaid_reservations: Tuple[int, ...]
    """Players off the board with no owner and no payment. Free blocks."""

    declared_withdrawn: Tuple[int, ...] = ()
    """Players the CALLER explicitly withdrew, as the ``unavailable`` pass
    branch does. Accepted, but not conservation-neutral and never silent: a
    withdrawn player is denied to eleven rivals at no cost to anybody, which
    weakens the whole field for free. Fine as a stated device for asking "what
    if he simply leaves our board"; **not** fine as a CE arm compared against a
    branch where somebody pays for him."""

    dollars_paid: int = 0
    dollars_remaining: int = 0
    dollars_started: int = 0

    @property
    def pool_balances(self) -> bool:
        return (self.n_initially_owned + self.n_branch_acquired
                + self.n_focus_bought + self.n_rival_bought
                + self.n_undrafted + len(self.declared_withdrawn)) == self.n_pool

    @property
    def dollars_balance(self) -> bool:
        return self.dollars_paid + self.dollars_remaining == self.dollars_started

    @property
    def ok(self) -> bool:
        return (self.pool_balances and self.dollars_balance
                and not self.duplicate_owners and not self.over_budget
                and not self.reserve_violations and not self.illegal_rosters
                and not self.wrong_size and not self.unpaid_reservations)

    def problems(self) -> List[str]:
        out: List[str] = []
        if not self.pool_balances:
            out.append(
                f"pool does not reconcile: {self.n_initially_owned} owned + "
                f"{self.n_branch_acquired} branch + {self.n_focus_bought} focus "
                f"+ {self.n_rival_bought} rival + {self.n_undrafted} undrafted "
                f"+ {len(self.declared_withdrawn)} declared-withdrawn "
                f"!= {self.n_pool}")
        if not self.dollars_balance:
            out.append(f"dollars do not reconcile: {self.dollars_paid} paid + "
                       f"{self.dollars_remaining} left != {self.dollars_started}")
        if self.duplicate_owners:
            out.append(f"players on two rosters: {list(self.duplicate_owners)}")
        if self.over_budget:
            out.append(f"owners over budget: {list(self.over_budget)}")
        if self.reserve_violations:
            out.append(f"owners without $1 per open slot: "
                       f"{list(self.reserve_violations)}")
        if self.illegal_rosters:
            out.append(f"rosters that cannot field a lineup: "
                       f"{list(self.illegal_rosters)}")
        if self.wrong_size:
            out.append(f"completed teams not holding 15: {list(self.wrong_size)}")
        if self.unpaid_reservations:
            out.append(
                f"UNPAID RESERVATIONS (free blocks): "
                f"{list(self.unpaid_reservations)} -- removed from rival reach "
                f"with no owner and no payment")
        return out

    def to_dict(self) -> Dict[str, object]:
        return {
            "ok": self.ok,
            "pool": {"total": self.n_pool,
                     "initially_owned": self.n_initially_owned,
                     "branch_acquired": self.n_branch_acquired,
                     "focus_bought": self.n_focus_bought,
                     "rival_bought": self.n_rival_bought,
                     "undrafted": self.n_undrafted,
                     "declared_withdrawn": list(self.declared_withdrawn),
                     "balances": self.pool_balances},
            "dollars": {"started": self.dollars_started,
                        "paid": self.dollars_paid,
                        "remaining": self.dollars_remaining,
                        "balances": self.dollars_balance},
            "duplicate_owners": list(self.duplicate_owners),
            "over_budget": list(self.over_budget),
            "reserve_violations": list(self.reserve_violations),
            "illegal_rosters": list(self.illegal_rosters),
            "wrong_size": list(self.wrong_size),
            "unpaid_reservations": list(self.unpaid_reservations),
            "problems": self.problems(),
        }


@dataclass(frozen=True)
class JointWorld:
    """One fully reconciled twelve-team league, and how it was built."""

    state: AuctionState
    """Final room: our purchases applied, then the rivals' conditional board."""

    cast: ComparisonCast
    focus_roster: Tuple[int, ...]
    focus_prices: Dict[int, int]
    focus_cost: int
    rival_board: BoardResult
    returned_to_board: Tuple[int, ...]
    """Shadow-held players we did not buy, put back for the rivals to contest."""
    reclaimed_by_rivals: Tuple[int, ...]
    conservation: ConservationReport
    finalist_index: int
    notes: str = ""

    def fingerprint(self) -> str:
        """Digest of the whole joint allocation, not just our half."""
        h = hashlib.sha256()
        h.update(("focus=" + ",".join(
            f"{p}:{self.focus_prices.get(p, 0)}"
            for p in sorted(self.focus_roster)) + "\n").encode())
        for o in sorted(self.state.owners, key=lambda x: x.owner_id):
            h.update((f"{o.owner_id}=" + ",".join(
                f"{f.player_id}:{f.price}"
                for f in sorted(o.filled, key=lambda f: f.player_id))
                + "\n").encode())
        return h.hexdigest()[:16]

    def to_dict(self) -> Dict[str, object]:
        return {
            "fingerprint": self.fingerprint(),
            "finalist_index": self.finalist_index,
            "focus_roster": sorted(self.focus_roster),
            "focus_prices": {str(k): v for k, v in
                             sorted(self.focus_prices.items())},
            "focus_cost": self.focus_cost,
            "returned_to_board": list(self.returned_to_board),
            "reclaimed_by_rivals": list(self.reclaimed_by_rivals),
            "rival_board_fingerprint": self.rival_board.fingerprint(),
            "rival_board_exactness": self.rival_board.exactness,
            "conservation": self.conservation.to_dict(),
            "notes": self.notes,
        }


def validate_joint_world(world: "JointWorld") -> "JointWorld":
    """Refuse a world that does not reconcile. Raises :class:`ConservationError`."""
    if not world.conservation.ok:
        raise ConservationError(
            "joint world violates the conservation invariant:\n  - "
            + "\n  - ".join(world.conservation.problems()))
    return world


def _reconcile(state: AuctionState, initial: AuctionState,
               branch_acquired: FrozenSet[int],
               focus_roster: Sequence[int],
               unpaid_off_board: Sequence[int],
               declared_withdrawn: FrozenSet[int] = frozenset()
               ) -> ConservationReport:
    """Count every player and every dollar in the finished world."""
    focus = state.focus_owner_id
    pool_ids = {s.player_id for s in state.pool}
    # ``initial`` is the room AFTER the nominated-player branch was applied, so
    # the branch acquisition is already sitting in someone's roster there.
    # Counting him as both pre-owned and branch-acquired would double him and
    # break the pool identity by exactly one.
    pre_owned = ({pid for o in initial.owners for pid in o.player_ids}
                 - set(branch_acquired))

    seen: Dict[int, str] = {}
    dupes: List[int] = []
    for o in state.owners:
        for pid in o.player_ids:
            if pid in seen:
                dupes.append(pid)
            seen[pid] = o.owner_id

    focus_new = [pid for pid in state.owner(focus).player_ids
                 if pid not in pre_owned and pid not in branch_acquired]
    rival_new = [pid for o in state.owners if o.owner_id != focus
                 for pid in o.player_ids
                 if pid not in pre_owned and pid not in branch_acquired]
    undrafted = [pid for pid in pool_ids
                 if pid not in seen and pid not in declared_withdrawn]

    over, reserve, illegal, wrong = [], [], [], []
    for o in state.owners:
        if o.budget_remaining < 0:
            over.append(o.owner_id)
        if o.budget_remaining < o.open_slots * o.min_bid:
            reserve.append(o.owner_id)
        if o.open_slots == 0:
            if o.n_players != o.roster_capacity:
                wrong.append(o.owner_id)
            if not o.fields_a_full_lineup:
                illegal.append(o.owner_id)
        elif not o.can_still_field_lineup(state.available_by_position()):
            illegal.append(o.owner_id)

    return ConservationReport(
        n_pool=len(pool_ids), n_initially_owned=len(pre_owned),
        n_branch_acquired=len(branch_acquired), n_focus_bought=len(focus_new),
        n_rival_bought=len(rival_new), n_undrafted=len(undrafted),
        duplicate_owners=tuple(sorted(set(dupes))), over_budget=tuple(over),
        reserve_violations=tuple(reserve), illegal_rosters=tuple(illegal),
        wrong_size=tuple(wrong),
        unpaid_reservations=tuple(sorted(set(unpaid_off_board)
                                         - set(declared_withdrawn))),
        declared_withdrawn=tuple(sorted(declared_withdrawn)),
        dollars_paid=sum(o.spent for o in state.owners),
        dollars_remaining=sum(o.budget_remaining for o in state.owners),
        dollars_started=sum(o.budget_start for o in state.owners))


def build_joint_world(
    state: AuctionState,
    cast: ComparisonCast,
    costs: CostBook,
    completion: Completion,
    *,
    finalist_index: int = 0,
    board_settings: BoardSettings = BoardSettings(),
    market: Optional[MarketState] = None,
    key_by_id: Optional[Dict[int, str]] = None,
    default_cost: int = 1,
    shadow_held: FrozenSet[int] = frozenset(),
    branch_acquired: FrozenSet[int] = frozenset(),
    declared_withdrawn: FrozenSet[int] = frozenset(),
) -> JointWorld:
    """Buy exactly what this completion names, then let the rivals finish.

    ``state`` is the room *before* our completion purchases and after any
    nominated-player branch has been applied. ``shadow_held`` is only used for
    reporting which holds came back; nothing is reserved on its account.
    """
    focus = state.focus_owner_id
    pre_owned = set(state.owner(focus).player_ids)
    to_buy = [pid for pid in completion.roster if pid not in pre_owned]

    prices: Dict[int, int] = {}
    cur = state
    for pid in to_buy:
        price = costs.cost_of(pid, default_cost)
        problem = cur.purchase_shortfall(pid, focus, price)
        if problem is not None:
            raise ConservationError(
                f"the selected completion cannot actually be bought: player "
                f"{pid} at ${price} -- {problem}. A completion we cannot pay "
                f"for is not a roster, and evaluating it would be reporting a "
                f"league that could not happen.")
        cur = cur.apply_purchase(pid, focus, price)
        prices[pid] = price

    # Everything we did not buy is back on the board on equal terms -- shadow
    # holds included. The rivals now complete against exactly that.
    rival_settings = replace(board_settings, focus_bids=False)
    rival_board = continue_shared_board(cur, settings=rival_settings,
                                        costs=costs, market=market,
                                        key_by_id=key_by_id)
    final = rival_board.state
    returned = tuple(sorted(p for p in shadow_held if p not in prices
                            and p not in pre_owned))
    rival_ids = {pid for o in final.owners if o.owner_id != focus
                 for pid in o.player_ids}
    reclaimed = tuple(p for p in returned if p in rival_ids)

    # A free block would be a player off the board with no owner and no
    # payment. After conditional reallocation there is no such mechanism left,
    # and the report proves it rather than asserting it.
    unpaid = [p for p in final.withdrawn
              if p not in rival_ids and p not in prices]
    report = _reconcile(final, state, frozenset(branch_acquired),
                        completion.roster, unpaid,
                        frozenset(declared_withdrawn))
    joint_cast = cast_from_board(rival_board, cast, final)
    joint_cast = joint_cast.with_team(joint_cast.focus_team_index,
                                      tuple(completion.roster))
    return JointWorld(
        state=final, cast=joint_cast, focus_roster=tuple(completion.roster),
        focus_prices=prices, focus_cost=sum(prices.values()),
        rival_board=rival_board, returned_to_board=returned,
        reclaimed_by_rivals=reclaimed, conservation=report,
        finalist_index=finalist_index,
        notes="conditional reallocation: rivals completed against our actual buys")


def format_shadow_audit(a: ShadowAudit, width: int = 92) -> str:
    bar = "=" * width
    out = [bar, "SHADOW-HOLD AUDIT", bar,
           f"focus owner              {a.focus_owner_id}",
           f"shadow budget            ${a.shadow_budget_before} -> "
           f"${a.shadow_budget_after}",
           f"shadow slots             {a.shadow_slots_before} -> "
           f"{a.shadow_slots_after}",
           f"shadow-won players       {len(a.shadow_bid)}  "
           f"(${a.shadow_cost_total} notional)",
           f"  of those, we bought    {len(a.selected_held)}",
           f"  of those, we did not   {len(a.unselected_held)}  "
           f"(${a.shadow_cost_unselected} notional)",
           f"actual focus roster cost ${a.actual_focus_cost}",
           f"cost divergence          ${a.cost_divergence}",
           f"rivals recompleted       {a.rivals_recompleted}",
           f"unselected -> a rival    {len(a.unselected_reaching_a_rival)}",
           f"unselected undrafted     {len(a.unselected_left_undrafted)}", "",
           f"FREE BLOCKS              {len(a.free_blocks)}  "
           f"(${a.free_block_cost} of denial we never paid for)"]
    if a.free_blocks:
        out += [f"  players {list(a.free_blocks)}",
                "  Each was removed from eleven rivals' reach, bought by nobody,",
                "  and charged to nobody. That is unpaid blocking and it inflates",
                "  both our roster and our denial value."]
    else:
        out.append("  none: every hold was bought, returned or reclaimed.")
    out.append(bar)
    return "\n".join(out)


def format_conservation(r: ConservationReport, width: int = 92) -> str:
    bar = "=" * width
    out = [bar, "CONSERVATION REPORT", bar,
           f"pool                {r.n_pool}",
           f"  initially owned   {r.n_initially_owned}",
           f"  branch acquired   {r.n_branch_acquired}",
           f"  focus bought      {r.n_focus_bought}",
           f"  rival bought      {r.n_rival_bought}",
           f"  undrafted         {r.n_undrafted}",
           f"  balances          {r.pool_balances}",
           f"dollars started     ${r.dollars_started}",
           f"  paid              ${r.dollars_paid}",
           f"  remaining         ${r.dollars_remaining}",
           f"  balances          {r.dollars_balance}",
           f"duplicate owners    {list(r.duplicate_owners) or 'none'}",
           f"over budget         {list(r.over_budget) or 'none'}",
           f"reserve violations  {list(r.reserve_violations) or 'none'}",
           f"illegal rosters     {list(r.illegal_rosters) or 'none'}",
           f"wrong roster size   {list(r.wrong_size) or 'none'}",
           f"unpaid reservations {list(r.unpaid_reservations) or 'none'}",
           f"declared withdrawn  {list(r.declared_withdrawn) or 'none'}"
           + ("   <- NOT conservation-neutral; denied to 11 rivals for free"
              if r.declared_withdrawn else ""), ""]
    out.append("VERDICT: reconciles" if r.ok else "VERDICT: REFUSED")
    for p in r.problems():
        out.append(f"  - {p}")
    out.append(bar)
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Championship equity over reconciled worlds
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JointArm:
    """One side of a buy/pass comparison, as a fully reconciled league."""

    world: JointWorld
    ce: float
    """Equity on the INDEPENDENT holdout sample. The number to quote."""
    selection_ce: float
    """Equity on the sample that chose this world. Upward-biased; reported for
    transparency only."""
    league_ce: Tuple[float, ...]
    focus_team_index: int
    focus_proxy: float
    field_proxy: Tuple[float, ...]
    n_worlds_compared: int
    holdout_indicator: object = None
    """Per-season championship indicator on the holdout sample. Kept so a paired
    buy/pass difference can be differenced season by season rather than being
    formed from two independent means."""

    @property
    def field_mean(self) -> float:
        return sum(self.field_proxy) / len(self.field_proxy)

    @property
    def league_ce_sum(self) -> float:
        return float(sum(self.league_ce))

    @property
    def rank(self) -> int:
        """1 = best. Ties broken by index, so it is a report, not a claim."""
        order = sorted(range(len(self.league_ce)),
                       key=lambda i: -self.league_ce[i])
        return order.index(self.focus_team_index) + 1

    def to_dict(self) -> Dict[str, object]:
        return {
            "ce": round(self.ce, 6),
            "selection_ce": round(self.selection_ce, 6),
            "league_ce": [round(x, 6) for x in self.league_ce],
            "league_ce_sum": round(self.league_ce_sum, 6),
            "rank": self.rank,
            "focus_proxy": round(self.focus_proxy, 4),
            "field_proxy_min": round(min(self.field_proxy), 4),
            "field_proxy_max": round(max(self.field_proxy), 4),
            "field_proxy_mean": round(self.field_mean, 4),
            "n_worlds_compared": self.n_worlds_compared,
            "world": self.world.to_dict(),
        }


def _league_ce(world: JointWorld, sims: int, seed: int, chunk: int) -> "object":
    from ..auction.completion import _build_roster_set
    from ..simulate import simulate_seasons
    rosters = _build_roster_set(world.state, world.cast, world.focus_roster)
    out = simulate_seasons(rosters, sims, seed, chunk)
    return out.championship_equity()


def _league_ce_and_indicator(world: JointWorld, team: int, sims: int, seed: int,
                             chunk: int):
    """League equities plus this team's per-season win indicator.

    The indicator is what makes a *paired* standard error possible: two arms run
    at the same seed see the same seasons, so differencing per season removes
    the season-to-season noise that dominates an unpaired comparison.
    """
    from ..auction.completion import _build_roster_set
    from ..simulate import simulate_seasons
    rosters = _build_roster_set(world.state, world.cast, world.focus_roster)
    out = simulate_seasons(rosters, sims, seed, chunk)
    return out.championship_equity(), out.champion_indicator(team)


def evaluate_joint_arm(
    worlds: Sequence[JointWorld],
    *,
    focus_team_index: int,
    proxy: ProxyEvaluator,
    selection_sims: int,
    selection_seed: int,
    holdout_sims: int,
    holdout_seed: int,
    chunk: int = 64,
) -> JointArm:
    """Choose among reconciled worlds on one sample, report on another.

    The discipline is the existing one and it matters here more than usual: the
    worlds differ by which completion we bought, so picking the maximum of
    several noisy equity estimates and then quoting that same estimate would
    report how lucky a sample was as if it were how good a roster is.
    """
    if not worlds:
        raise ValueError("evaluate_joint_arm needs at least one world")
    for w in worlds:
        validate_joint_world(w)
    scored = []
    for w in worlds:
        ce = _league_ce(w, selection_sims, selection_seed, chunk)
        scored.append((float(ce[focus_team_index]), w))
    best_sel, best = max(scored, key=lambda t: (t[0],
                                                -hash(t[1].fingerprint())))
    league, indicator = _league_ce_and_indicator(
        best, focus_team_index, holdout_sims, holdout_seed, chunk)
    field = tuple(float(v) for v in proxy.strength_many(
        [t for i, t in enumerate(best.cast.rosters) if i != focus_team_index]))
    return JointArm(
        world=best, ce=float(league[focus_team_index]), selection_ce=best_sel,
        league_ce=tuple(float(x) for x in league),
        focus_team_index=focus_team_index,
        focus_proxy=float(proxy.strength(best.focus_roster)),
        field_proxy=field, n_worlds_compared=len(worlds),
        holdout_indicator=indicator)


def build_joint_worlds(
    state: AuctionState,
    cast: ComparisonCast,
    costs: CostBook,
    *,
    board_settings: BoardSettings = BoardSettings(),
    completion: CompletionSettings = CompletionSettings(),
    market: Optional[MarketState] = None,
    key_by_id: Optional[Dict[int, str]] = None,
    proxy: Optional[ProxyEvaluator] = None,
    protect: Sequence[int] = (),
    default_cost: int = 1,
    branch_acquired: FrozenSet[int] = frozenset(),
    declared_withdrawn: FrozenSet[int] = frozenset(),
    max_worlds: int = 3,
    completions: Optional[Sequence[Completion]] = None,
) -> Tuple[JointWorld, ...]:
    """Shadow pass, finalists, then one reconciled world per finalist.

    A finalist we cannot actually afford at cost-book prices is dropped with its
    reason rather than evaluated; that is a real bound and the caller sees the
    count fall.

    ``completions`` supplies the choice set instead of running a fresh beam.
    That is how a price ladder gets a *nested* opportunity set: the beam is
    path-dependent on a shadow board that itself depends on our remaining money,
    so re-running it per price silently offers different prices different
    choices. See :mod:`ceauction.tactical.nested`.
    """
    focus = state.focus_owner_id
    if proxy is None:
        proxy = ProxyEvaluator(state.pool, state.settings,
                               completion.proxy_reps, completion.proxy_seed)
    if completions is not None:
        finalists = list(completions)
    else:
        shadow = continue_shared_board(state, settings=board_settings,
                                       costs=costs, market=market,
                                       key_by_id=key_by_id, protect=protect)
        shadow_cast = cast_from_board(shadow, cast, state)
        res = complete_roster(
            shadow.state, shadow_cast, costs, settings=completion,
            owner_id=focus, evaluate_ce=False, default_cost=default_cost,
            proxy=proxy,
            reserved_ids=shadow.reserved_ids(exclude_owner=focus),
            notes="shadow pass, finalists for joint allocation")
        finalists = list(res.finalists) or ([res.best] if res.best else [])
    worlds: List[JointWorld] = []
    if completions is None:
        shadow_held = shadow.held_for_focus
    else:
        shadow_held = frozenset()
    for i, c in enumerate(finalists[:max_worlds]):
        try:
            worlds.append(validate_joint_world(build_joint_world(
                state, cast, costs, c, finalist_index=i,
                board_settings=board_settings, market=market,
                key_by_id=key_by_id, default_cost=default_cost,
                shadow_held=shadow_held,
                branch_acquired=branch_acquired,
                declared_withdrawn=declared_withdrawn)))
        except ConservationError:
            continue
    if not worlds:
        raise ConservationError(
            "no finalist produced a reconciled world; nothing may be evaluated")
    return tuple(worlds)


@dataclass(frozen=True)
class JointComparison:
    """A paired buy/pass difference between two reconciled leagues."""

    buy: JointArm
    pass_arm: JointArm
    price: int
    recipient: Optional[str]
    recipient_price: Optional[int]

    @property
    def delta_ce(self) -> float:
        return self.buy.ce - self.pass_arm.ce

    @property
    def delta_se(self) -> float:
        """Paired standard error, differenced season by season."""
        import numpy as np
        a = np.asarray(self.buy.holdout_indicator, dtype=np.float64)
        b = np.asarray(self.pass_arm.holdout_indicator, dtype=np.float64)
        if a.shape != b.shape or a.size < 2:
            return float("nan")
        diff = a - b
        return float(diff.std(ddof=1) / np.sqrt(diff.size))

    @property
    def ci95(self) -> Tuple[float, float]:
        import math
        se = self.delta_se
        if math.isnan(se):
            return (float("nan"), float("nan"))
        return (self.delta_ce - 1.96 * se, self.delta_ce + 1.96 * se)

    @property
    def verdict(self) -> str:
        import math
        lo, hi = self.ci95
        if math.isnan(lo):
            return "unresolved"
        if lo > 0.0:
            return "favorable"
        if hi < 0.0:
            return "unfavorable"
        return "unresolved"

    @property
    def allocations_identical(self) -> bool:
        return self.buy.world.fingerprint() == self.pass_arm.world.fingerprint()

    def to_dict(self) -> Dict[str, object]:
        lo, hi = self.ci95
        return {"price": self.price, "recipient": self.recipient,
                "recipient_price": self.recipient_price,
                "ce_buy": round(self.buy.ce, 6),
                "ce_pass": round(self.pass_arm.ce, 6),
                "delta_ce": round(self.delta_ce, 6),
                "delta_se": round(self.delta_se, 6),
                "ci95": [round(lo, 6), round(hi, 6)],
                "verdict": self.verdict,
                "buy_allocation": self.buy.world.fingerprint(),
                "pass_allocation": self.pass_arm.world.fingerprint(),
                "allocations_identical": self.allocations_identical}
