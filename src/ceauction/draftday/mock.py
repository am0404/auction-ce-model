"""The fabricated end-to-end rehearsal.

Every player and price in this run is chosen by the script, not observed. The
point is not to predict tonight's auction; it is to drive the whole product
once -- sale, market update, budgets, candidate-specific legality, undo,
restart, persistence, export, import -- and prove the fingerprint comes back
identical at each step where it must.

It writes to a scratch state file, never to the live ``draft_state.json``.
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import List, Optional

from ..auction.state import AuctionRuleError
from ..league import Position
from . import DRAFTDAY_DIR

__all__ = ["run_mock"]


class _Report:
    def __init__(self) -> None:
        self.rows: List[tuple] = []
        self.failed = 0

    def step(self, n: int, label: str, ok: bool, detail: str = "") -> None:
        if not ok:
            self.failed += 1
        self.rows.append((n, label, ok, detail))
        print(f"  {n:>2}. [{'ok  ' if ok else 'FAIL'}] {label}"
              + (f"   {detail}" if detail else ""))


def run_mock(*, contract: Path, sleeper_csv: Path,
             use_cache: bool = True) -> int:
    from .board import load_board
    from .panels import nomination_panel, owner_table, qb_panel
    from .server import DraftDayServer
    from .session import DraftSession

    print("MOCK AUCTION REHEARSAL -- every price below is fabricated.\n")
    rep = _Report()
    tmpdir = Path(tempfile.mkdtemp(prefix="draftday-mock-"))
    state_path = tmpdir / "draft_state.json"
    overrides_path = tmpdir / "manual_overrides.json"

    board = load_board(contract=contract, sleeper_csv=sleeper_csv,
                       use_cache=use_cache)

    def new_session() -> DraftSession:
        return DraftSession(board, state_path=state_path,
                            overrides_path=overrides_path)

    # 1 -- load the opening room
    session = new_session()
    opening_fp = session.fingerprint()
    rep.step(1, "load opening room",
             len(session.state.pool) > 0 and not session.sales,
             f"{len(session.state.pool)} players, fp {opening_fp}")

    app = DraftDayServer(session, enable_proxy=False)
    app.bootstrap_opening()

    def pick(position: Position, skip: set) -> int:
        cands = [s for s in session.state.available_specs
                 if int(s.position) == int(position)
                 and s.player_id not in skip]
        cands.sort(key=lambda s: -board.points_by_id.get(s.player_id, 0.0))
        return cands[0].player_id

    used: set = set()

    # 2 -- nominate a quarterback
    qb_id = pick(Position.QB, used)
    used.add(qb_id)
    t0 = time.perf_counter()
    panel = nomination_panel(session, qb_id, current_bid=30,
                             high_bidder="Team02")
    nom_ms = (time.perf_counter() - t0) * 1000
    rep.step(2, "nominate a QB",
             panel["verdict"] in ("BID", "CAUTION", "STOP")
             and panel["next_legal_bid"] == 31,
             f"{panel['name']}: {panel['verdict']}, cap ${panel['provisional_cap']}"
             f" [{panel['cap']['basis']}], {nom_ms:.0f}ms")

    # 3 -- record the sale to Team02
    before_budget = session.state.owner("Team02").budget_remaining
    before_obs = len(session.market.observations)
    t0 = time.perf_counter()
    session.record_sale(qb_id, "Team02", 42)
    sale_ms = (time.perf_counter() - t0) * 1000
    after = session.state.owner("Team02")
    rep.step(3, "record sale to Team02 at $42",
             after.budget_remaining == before_budget - 42
             and after.n_players == 1 and after.counts.qb == 1,
             f"budget {before_budget} -> {after.budget_remaining}, "
             f"{sale_ms:.0f}ms")

    # 4 -- nominate a running back
    rb_id = pick(Position.RB, used)
    used.add(rb_id)
    panel_rb = nomination_panel(session, rb_id, current_bid=20)
    rep.step(4, "nominate an RB", panel_rb["our_legal_max"] > 0,
             f"{panel_rb['name']}: legal max ${panel_rb['our_legal_max']}, "
             f"cap ${panel_rb['provisional_cap']}")

    # 5 -- record the sale to us
    session.record_sale(rb_id, session.focus_owner_id, 38)
    us = session.state.owner(session.focus_owner_id)
    rep.step(5, "record sale to us at $38",
             us.n_players == 1 and us.budget_remaining == 162
             and us.max_bid == 149,
             f"our budget ${us.budget_remaining}, legal max ${us.max_bid}")

    # 6 -- create a positional inflation pattern: four RBs well over the band
    inflated = []
    for i in range(4):
        pid_ = pick(Position.RB, used)
        used.add(pid_)
        band = app.rows()  # keeps the cache warm the way the page does
        from .board import market_band
        mb = market_band(board, pid_, session.market)
        price = max(1, (mb.base or 5) + 12)
        owner = f"Team{(i % 9) + 3:02d}"
        try:
            session.record_sale(pid_, owner, price)
            inflated.append((board.name_by_id.get(pid_), mb.base, price))
        except AuctionRuleError as exc:
            rep.step(6, "inflation pattern", False, str(exc))
            break
    rep.step(6, "create a positional inflation pattern (4 RBs over the band)",
             len(inflated) == 4,
             "; ".join(f"{n} base ${b} -> ${p}" for n, b, p in inflated))

    # 7 -- verify the market moved, and moved the right position
    rb_level = session.market.position_levels.get("RB")
    wr_level = session.market.position_levels.get("WR")
    next_rb = pick(Position.RB, used)
    next_wr = pick(Position.WR, used)
    from .board import market_band
    rb_now = market_band(board, next_rb, session.market)
    rb_prior = market_band(board, next_rb, board.market)
    wr_now = market_band(board, next_wr, session.market)
    wr_prior = market_band(board, next_wr, board.market)
    rb_lift = (rb_now.base or 0) / max(1, rb_prior.base or 1)
    wr_lift = (wr_now.base or 0) / max(1, wr_prior.base or 1)
    rep.step(7, "market update responds to the observed position",
             rb_lift > 1.0 and rb_level is not None and rb_level.n > 0,
             f"RB observations {rb_level.n if rb_level else 0}, "
             f"next RB base ${rb_prior.base} -> ${rb_now.base} "
             f"(x{rb_lift:.2f})")
    # The receivers do move, and that is correct rather than a leak: sales
    # spanning two positions are evidence the whole ROOM is paying up, and the
    # market layer credits the room level only once more than one position has
    # been observed. What must not happen is the unobserved position moving as
    # much as the observed one -- that would be the positional signal leaking.
    rep.step(7, "unobserved WR moves only by the shared room level, not the "
                "RB signal",
             wr_lift < rb_lift and (wr_level is None or wr_level.n == 0),
             f"WR direct observations {wr_level.n if wr_level else 0}, "
             f"next WR base ${wr_prior.base} -> ${wr_now.base} "
             f"(x{wr_lift:.2f} vs RB x{rb_lift:.2f})")

    # 8 -- verify all twelve budgets
    spent_by_sale = {}
    for s in session.sales:
        spent_by_sale[s.owner_id] = spent_by_sale.get(s.owner_id, 0) + s.price
    ok = True
    for owner in session.state.owners:
        expected = 200 - spent_by_sale.get(owner.owner_id, 0)
        if owner.budget_remaining != expected or owner.spent != spent_by_sale.get(
                owner.owner_id, 0):
            ok = False
    rep.step(8, "all twelve budgets reconcile to the recorded sales", ok,
             f"total spent ${sum(spent_by_sale.values())} over "
             f"{len(session.sales)} sale(s)")

    # 9 -- candidate-specific bidding legality
    te_id = pick(Position.TE, used)
    rows = owner_table(session, candidate_id=te_id, next_bid=1)
    engine_ok = all(
        row["can_bid_candidate"] == (session.state.purchase_is_legal(
            te_id, row["owner_id"], 1))
        for row in rows)
    # A candidate-specific maximum is never above the general one.
    bounded = all(row["candidate_max"] <= row["general_max"] for row in rows)
    rep.step(9, "candidate-specific legality matches the engine",
             engine_ok and bounded,
             f"{sum(1 for r in rows if r['can_bid_candidate'])}/12 may bid $1 "
             f"on {board.name_by_id.get(te_id)}")

    # 10 -- undo the last sale, exactly
    fp_before_last = None
    pre = session.fingerprint()
    last = session.sales[-1]
    # Rebuild what the fingerprint was before that sale by replaying.
    replay = new_session()
    replay.autosave = False
    for s in session.sales[:-1]:
        replay.record_sale_silent(s.player_id, s.owner_id, s.price)
    fp_before_last = replay.fingerprint()
    undone = session.undo()
    rep.step(10, "undo restores the exact prior state",
             session.fingerprint() == fp_before_last and undone is not None
             and undone.player_id == last.player_id,
             f"{pre} -> {session.fingerprint()} (expected {fp_before_last})")

    # 11 & 12 -- restart the server and verify the persisted state
    saved_fp = session.fingerprint()
    saved_sales = len(session.sales)
    session.save()
    restarted = new_session()
    restored = restarted.restore()
    rep.step(11, "restart the server (fresh session from disk)", restored,
             f"read {restarted.state_path.name}")
    rep.step(12, "persisted state matches",
             restarted.fingerprint() == saved_fp
             and len(restarted.sales) == saved_sales,
             f"{len(restarted.sales)} sale(s), fp {restarted.fingerprint()}")

    # 13 -- export and re-import
    snap = tmpdir / "snapshot.json"
    restarted.export_snapshot(snap)
    reimported = new_session()
    reimported.import_snapshot(snap)
    rep.step(13, "export and re-import", snap.exists(),
             f"{snap.stat().st_size} bytes")

    # 14 -- identical fingerprint
    rep.step(14, "fingerprint identical after export/import",
             reimported.fingerprint() == saved_fp,
             f"{reimported.fingerprint()} == {saved_fp}")

    # Extra: the rejection rules, each one exercised.
    print("\n  rejection rules")
    sold_id = session.sales[0].player_id
    checks = [
        ("duplicate sale", session.check_sale(sold_id, "Team04", 5)),
        ("unaffordable purchase",
         session.check_sale(next_wr, "Team02", 500)),
        ("invalid price (zero)", session.check_sale(next_wr, "Team02", 0)),
        ("unknown owner", session.check_sale(next_wr, "Nobody", 5)),
        ("unknown player", session.check_sale(1, "Team02", 5)),
    ]
    for label, problem in checks:
        rep.step(15, f"rejected: {label}", problem is not None,
                 (problem or "NOT REJECTED")[:88])

    # A full roster, and a roster that can no longer complete a lineup.
    full = new_session()
    full.autosave = False
    filled = 0
    for spec in full.state.available_specs:
        if filled >= 15:
            break
        if full.check_sale(spec.player_id, "Team05", 1) is None:
            full.record_sale(spec.player_id, "Team05", 1)
            filled += 1
    spare = next(s for s in full.state.available_specs)
    rep.step(15, "rejected: purchase onto a full roster",
             full.check_sale(spare.player_id, "Team05", 1) is not None,
             f"Team05 holds {full.state.owner('Team05').n_players}")
    rep.step(15, "rejected: reserve violation (leaves no $1 for open slots)",
             full.check_sale(next_wr, "Team02",
                             full.state.owner("Team02").max_bid + 1) is not None)

    # QB scarcity reacts to QB sales.
    q0 = qb_panel(new_session(), candidate_id=None)
    q1 = qb_panel(session, candidate_id=None)
    rep.step(16, "QB scarcity updates after QB sales",
             q1["startable_remaining"] < q0["startable_remaining"]
             or q1["counts_by_bucket"]["1"] > 0,
             f"startable {q0['startable_remaining']} -> "
             f"{q1['startable_remaining']}; teams with 1 QB: "
             f"{q1['counts_by_bucket']['1']}")

    print()
    print("=" * 72)
    if rep.failed:
        print(f"MOCK REHEARSAL: {rep.failed} FAILED of {len(rep.rows)} checks")
    else:
        print(f"MOCK REHEARSAL: all {len(rep.rows)} checks passed")
    print(f"scratch state: {tmpdir}")
    print("=" * 72)
    return 1 if rep.failed else 0
