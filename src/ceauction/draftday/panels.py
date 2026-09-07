"""The four dashboard panels: nomination, owners, QB scarcity, and the board.

Each function here assembles a plain dictionary the browser renders. No panel
computes a valuation; they read the auction engine, the market layer and the
existing proxy, and every dollar they emit carries the basis label that
produced it.

Player ids in this project are 64-bit hashes of the canonical key, which is
wider than a JavaScript number can hold exactly. Every id crossing into a
payload is therefore a **string**, and :func:`pid` is the only place that
conversion happens.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence, Tuple

from ..auction.state import AuctionState, OwnerAuctionState
from ..league import Position
from ..market.live import MarketState
from .board import BoardRow, DraftDayBoard, market_band
from .caps import (
    DISAGREEMENT_LABEL,
    EXACT,
    CapRails,
    ProvisionalCap,
    advice,
    provisional_cap,
)
from .session import DraftSession

__all__ = [
    "pid",
    "nomination_panel",
    "owner_table",
    "qb_panel",
    "RECIPIENT_WARNING",
]

#: Printed beside every recipient list, every time. The bidder model is a set
#: of stated scenarios, not behaviour fitted to this league's twelve managers.
RECIPIENT_WARNING = (
    "Recipient probabilities are STATED ASSUMPTIONS from the scenario bidder "
    "model. They are not fitted to this league and no one's behaviour has been "
    "observed. Read them as 'who could plausibly take him', never as odds.")


def pid(player_id: int) -> str:
    """Player ids are wider than a JS number. They travel as strings."""
    return str(int(player_id))


# ---------------------------------------------------------------------------
# B. Current nomination
# ---------------------------------------------------------------------------


def nomination_panel(session: DraftSession, player_id: int, *,
                     current_bid: Optional[int] = None,
                     high_bidder: Optional[str] = None,
                     nominated_by: Optional[str] = None,
                     proxy_ceiling: Optional[int] = None,
                     proxy_status: str = "absent") -> Dict[str, object]:
    """Everything needed to answer 'do I bid?' without a slow computation.

    Only cached and exact quantities are read here: the auction arithmetic, the
    market band, the room's candidate-specific legality, and the existing
    scenario recipient enumeration. Nothing on this path simulates a season.
    """
    t0 = time.perf_counter()
    board = session.board
    state = session.state
    market = session.market
    focus = state.focus_owner_id
    spec = state.spec(player_id)
    key = board.key_for(player_id) or ""
    position = Position(int(spec.position)).name

    increment = 1
    next_bid = (int(current_bid) + increment) if current_bid is not None else 1

    our = state.owner(focus)
    sold_to = state.owner_of.get(player_id)
    if sold_to is not None:
        our_legal_max = 0
    elif player_id in state.withdrawn:
        our_legal_max = 0
    else:
        our_legal_max = (our.max_bid
                         if our.max_bid > 0
                         and state.purchase_is_legal(player_id, focus, our.max_bid)
                         else 0)

    band = market_band(board, player_id, market)
    ov = session.overrides.get(key)
    rails = CapRails(
        legal_max=our_legal_max, market=band,
        proxy_ceiling=proxy_ceiling,
        proxy_status=proxy_status,
        manual_adjustment=ov.dollar_adjustment if ov else 0,
        manual_note=ov.reasoning if ov else "")
    cap: ProvisionalCap = provisional_cap(rails)

    # Who may legally bid the next dollar on THIS player. The candidate is
    # named, so this is the full purchase test and not the weaker money-only
    # question.
    capable = state.bid_capacity(next_bid, candidate_id=player_id)
    capable_at_cap = (state.bid_capacity(max(next_bid, cap.cap),
                                         candidate_id=player_id)
                      if cap.cap >= next_bid else {"able": []})

    recipients: List[Dict[str, object]] = []
    recipient_error = ""
    if sold_to is None and player_id not in state.withdrawn:
        try:
            from ..tactical.recipients import enumerate_recipients
            rset = enumerate_recipients(
                state, player_id, current_price=current_bid,
                increment=increment, current_leader=high_bidder,
                market=market, candidate_key=key or None, max_named=3)
            for br in rset.branches:
                recipients.append({
                    "kind": br.kind,
                    "owner_id": br.owner_id or "",
                    "team": (session.team_name(br.owner_id)
                             if br.owner_id else "(nobody -- he leaves the board)"),
                    "price": br.price,
                    "legal_max": br.candidate_legal_max,
                    "weight": round(float(br.scenario_weight), 3),
                    "willingness": {"low": br.willingness_low,
                                    "base": br.willingness_base,
                                    "high": br.willingness_high},
                    "legal": bool(br.legal),
                    "note": br.rationale or (br.refusal or ""),
                })
        except Exception as exc:  # a panel must never take the page down
            recipient_error = f"recipient model unavailable: {exc}"

    fit = improvement = None
    if sold_to is None:
        ours = list(our.player_ids)
        try:
            proxy = board.proxy
            before = proxy.strength(ours) if ours else 0.0
            after = proxy.strength(ours + [player_id])
            improvement = round(after - before, 3)
            fit = round(float(proxy.lineup_shares(ours + [player_id])
                              .get(int(player_id), 0.0)), 3)
        except Exception:
            fit = improvement = None

    verdict = advice(cap, next_bid) if our_legal_max > 0 else "STOP"

    return {
        "player_id": pid(player_id),
        "name": board.name_by_id.get(player_id, spec.name),
        "position": position,
        "nfl_team": spec.nfl_team,
        "bye_week": int(spec.bye_week),
        "ppg": board.ppg_by_id.get(player_id, 0.0),
        "season_points": board.points_by_id.get(player_id, 0.0),
        "sold_to": sold_to or "",
        "sold_to_team": session.team_name(sold_to) if sold_to else "",
        "current_bid": current_bid,
        "high_bidder": high_bidder or "",
        "high_bidder_team": (session.team_name(high_bidder)
                             if high_bidder else ""),
        "nominated_by": nominated_by or "",
        "next_legal_bid": next_bid,
        "our_legal_max": our_legal_max,
        "our_legal_max_basis": EXACT,
        "provisional_cap": cap.cap,
        "verdict": verdict,
        "cap": cap.to_dict(),
        "market": band.to_dict(),
        "opening_cap": session.opening_caps.get(player_id),
        "roster_fit": fit,
        "lineup_improvement": improvement,
        "capable_now": [{"owner_id": o, "team": session.team_name(o)}
                        for o in capable["able"] if o != focus],
        "capable_blocked": {o: r for o, r in capable["blocked"].items()},
        "n_capable_at_next_bid": len([o for o in capable["able"] if o != focus]),
        "n_capable_at_cap": len([o for o in capable_at_cap.get("able", [])
                                 if o != focus]),
        "recipients": recipients,
        "recipient_warning": RECIPIENT_WARNING,
        "recipient_error": recipient_error,
        "disagreement": cap.disagreement,
        "disagreement_label": DISAGREEMENT_LABEL if cap.disagreement else "",
        "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
    }


# ---------------------------------------------------------------------------
# C. The twelve-owner table
# ---------------------------------------------------------------------------


def owner_table(session: DraftSession, *,
                candidate_id: Optional[int] = None,
                next_bid: int = 1) -> List[Dict[str, object]]:
    """Money, roster and legality for all twelve, with the candidate named.

    ``general financial maximum`` is the owner's ceiling for *any* player.
    ``candidate-specific legal maximum`` is the ceiling for the player on the
    block, which can be zero for an owner whose last slot must hold something
    else. They are different numbers and the table shows both.
    """
    state = session.state
    rows: List[Dict[str, object]] = []
    for owner in state.owners:
        counts = owner.counts
        general = owner.max_bid
        specific = 0
        blocked = ""
        if candidate_id is not None:
            if general > 0 and state.purchase_is_legal(candidate_id,
                                                       owner.owner_id, general):
                specific = general
            else:
                specific = 0
                blocked = state.purchase_shortfall(
                    candidate_id, owner.owner_id, max(1, general)) or ""
        rows.append({
            "owner_id": owner.owner_id,
            "team_name": session.team_name(owner.owner_id),
            "is_us": owner.owner_id == state.focus_owner_id,
            "budget_remaining": owner.budget_remaining,
            "spent": owner.spent,
            "roster_size": owner.n_players,
            "open_slots": owner.open_slots,
            "general_max": general,
            "candidate_max": specific,
            "can_bid_candidate": bool(candidate_id is not None and specific >= next_bid),
            "blocked_reason": blocked,
            "QB": counts.qb, "RB": counts.rb, "WR": counts.wr, "TE": counts.te,
        })
    return rows


# ---------------------------------------------------------------------------
# D. QB scarcity
# ---------------------------------------------------------------------------

#: A quarterback is counted "startable" if he is inside the top
#: ``n_teams * QB_SEATS_PER_TEAM`` quarterbacks of the whole pool by projected
#: season points. Two seats per team is the superflex ceiling, not a
#: requirement: this league lets a skill player fill the superflex, so the
#: number is a *scarcity yardstick*, never a rule.
QB_SEATS_PER_TEAM = 2


def qb_panel(session: DraftSession, *,
             candidate_id: Optional[int] = None,
             next_bid: int = 1,
             provisional: Optional[int] = None) -> Dict[str, object]:
    """Quarterback scarcity in a superflex league that permits a fallback.

    No rule here requires a second quarterback and none caps the number a team
    may hold. What the panel reports is supply, who can still pay for it, and
    how good the skill-position fallback for the superflex seat currently is.
    """
    board = session.board
    state = session.state
    focus = state.focus_owner_id
    settings = board.settings

    all_qbs = [s for s in state.pool if int(s.position) == int(Position.QB)]
    ranked = sorted(all_qbs,
                    key=lambda s: -board.points_by_id.get(s.player_id, 0.0))
    startable_cut = settings.n_teams * QB_SEATS_PER_TEAM
    startable_ids = {s.player_id for s in ranked[:startable_cut]}
    available = set(state.available_ids)
    startable_left = sorted(
        (s for s in ranked[:startable_cut] if s.player_id in available),
        key=lambda s: -board.points_by_id.get(s.player_id, 0.0))

    buckets = {"0": [], "1": [], "2": [], "3+": []}
    for owner in state.owners:
        n = owner.counts.qb
        label = "3+" if n >= 3 else str(n)
        buckets[label].append({"owner_id": owner.owner_id,
                               "team": session.team_name(owner.owner_id),
                               "qb": n})

    # Who can still pay for the quarterback on the block.
    capable: List[Dict[str, object]] = []
    highest_opposing = 0
    n_at_next = n_at_cap = 0
    if candidate_id is not None:
        for owner in state.owners:
            if owner.owner_id == focus:
                continue
            gmax = owner.max_bid
            ok = gmax > 0 and state.purchase_is_legal(candidate_id,
                                                      owner.owner_id, gmax)
            if not ok:
                continue
            capable.append({"owner_id": owner.owner_id,
                            "team": session.team_name(owner.owner_id),
                            "candidate_max": gmax,
                            "open_slots": owner.open_slots,
                            "qb": owner.counts.qb})
            highest_opposing = max(highest_opposing, gmax)
            if gmax >= next_bid:
                n_at_next += 1
            if provisional is not None and gmax >= provisional:
                n_at_cap += 1
        capable.sort(key=lambda r: -r["candidate_max"])

    # The superflex fallback: the best available non-quarterback measured on the
    # same proxy scale as the best available quarterback. If the fallback is
    # close, the superflex seat is not really a quarterback seat.
    fallback = None
    try:
        proxy = board.proxy
        ours = list(state.owner(focus).player_ids)
        base = proxy.strength(ours) if ours else 0.0
        best_qb = best_skill = None
        qb_pool = [s for s in state.available_specs
                   if int(s.position) == int(Position.QB)][:12]
        skill_pool = [s for s in state.available_specs
                      if int(s.position) != int(Position.QB)][:24]
        if qb_pool:
            vals = proxy.strength_many([ours + [s.player_id] for s in qb_pool])
            i = int(max(range(len(vals)), key=lambda k: vals[k]))
            best_qb = {"player_id": pid(qb_pool[i].player_id),
                       "name": board.name_by_id.get(qb_pool[i].player_id, ""),
                       "improvement": round(float(vals[i]) - base, 3)}
        if skill_pool:
            vals = proxy.strength_many([ours + [s.player_id] for s in skill_pool])
            i = int(max(range(len(vals)), key=lambda k: vals[k]))
            best_skill = {"player_id": pid(skill_pool[i].player_id),
                          "name": board.name_by_id.get(skill_pool[i].player_id, ""),
                          "improvement": round(float(vals[i]) - base, 3)}
        if best_qb and best_skill:
            fallback = {
                "best_available_qb": best_qb,
                "best_available_skill": best_skill,
                "qb_premium": round(best_qb["improvement"]
                                    - best_skill["improvement"], 3),
                "basis": "PROXY/HEURISTIC",
                "note": ("Mean weekly starting-lineup projection only. This is "
                         "the existing proxy, not championship equity, and it "
                         "ignores scoring variance entirely."),
            }
    except Exception:
        fallback = None

    # A waiver warning, stated as the arithmetic it is.
    teams_without = len(buckets["0"])
    replacement_warning = ""
    if len(startable_left) <= teams_without:
        replacement_warning = (
            f"{len(startable_left)} startable quarterback(s) left for "
            f"{teams_without} team(s) holding none. A replacement on waivers "
            f"would likely NOT be available.")

    return {
        "startable_cut": startable_cut,
        "startable_definition": (
            f"top {startable_cut} quarterbacks in the pool by projected season "
            f"points ({settings.n_teams} teams x {QB_SEATS_PER_TEAM} superflex "
            f"seats). A yardstick for supply, not a roster rule -- this league "
            f"requires no second quarterback and caps none."),
        "startable_remaining": len(startable_left),
        "startable_list": [
            {"player_id": pid(s.player_id),
             "name": board.name_by_id.get(s.player_id, s.name),
             "nfl_team": s.nfl_team,
             "points": board.points_by_id.get(s.player_id, 0.0)}
            for s in startable_left[:16]],
        "total_qbs_remaining": sum(1 for s in state.available_specs
                                   if int(s.position) == int(Position.QB)),
        "buckets": buckets,
        "counts_by_bucket": {k: len(v) for k, v in buckets.items()},
        "capable_opponents": capable,
        "highest_opposing_candidate_max": highest_opposing,
        "n_capable_at_next_bid": n_at_next,
        "n_capable_at_provisional_cap": n_at_cap,
        "superflex_fallback": fallback,
        "replacement_warning": replacement_warning,
        "qb3_insurance": ("UNSUPPORTED/NOT AUDITED -- no simulation evidence "
                          "applies to this exact room state."),
    }
