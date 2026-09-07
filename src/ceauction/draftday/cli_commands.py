"""``ce-lab draft-day`` -- the live tool, the board writer and the rehearsal.

    ce-lab draft-day serve      start the local dashboard (the draft-day command)
    ce-lab draft-day board      write opening_board.csv / opening_board.json
    ce-lab draft-day verify     check the install, the sources and the caches
    ce-lab draft-day mock       run the fabricated end-to-end rehearsal
    ce-lab draft-day nominate   one nomination panel, printed to the terminal

Everything reads real player data from ``local_data/`` and writes there too.
No command in this group emits player-level output anywhere else.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Optional

from . import DRAFTDAY_DIR

__all__ = ["add_draftday_parser"]

BANNER = (
    "ce-lab draft-day -- LOCAL live auction tool.\n"
    "Caps are MARKET/PROXY provisional unless a row says CE AUDITED.\n"
    "No unresolved or underconverged CE result produces a CE max bid.\n")


def _session(args, *, restore: bool = True):
    from .session import open_session
    return open_session(
        restore=restore, verbose=not getattr(args, "quiet", False),
        contract=Path(args.contract), sleeper_csv=Path(args.sleeper_csv),
        use_cache=not getattr(args, "no_cache", False))


def cmd_serve(args) -> int:
    from .server import serve
    print(BANNER)
    try:
        session = _session(args)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print("\nThe real sources live under local_data/ and are gitignored. "
              "Run `ce-lab draft-day verify` to see what is missing.",
              file=sys.stderr)
        return 2
    serve(host=args.host, port=args.port, open_browser=not args.no_browser,
          enable_proxy=not args.no_proxy, session=session)
    return 0


def cmd_board(args) -> int:
    from .board import load_board, opening_rows, write_opening_board
    print(BANNER)
    board = load_board(contract=Path(args.contract),
                       sleeper_csv=Path(args.sleeper_csv),
                       use_cache=not args.no_cache, verbose=True)
    t0 = time.perf_counter()
    rows = opening_rows(board)
    csv_path, json_path = write_opening_board(rows, coverage=board.coverage)
    anchored = sum(1 for r in rows if r.anchored)
    print(f"\n{len(rows)} rosterable players in {time.perf_counter() - t0:.1f}s")
    print(f"  anchored (Sleeper priced) : {anchored}")
    print(f"  unanchored                : {len(rows) - anchored}  "
          f"(UNPRICED -- not a $1 market observation)")
    by_pos = {}
    for r in rows:
        by_pos[r.position] = by_pos.get(r.position, 0) + 1
    print("  by position               : "
          + ", ".join(f"{k} {v}" for k, v in sorted(by_pos.items())))
    print(f"\nwritten (LOCAL ONLY, gitignored):\n  {csv_path}\n  {json_path}")
    print("\ntop of the board by market base:")
    head = (f"  {'player':<26}{'pos':<5}{'mkt':>6}{'band':>10}{'fit':>7}"
            f"{'+lineup':>9}{'legal':>7}{'cap':>6}  basis")
    print(head)
    print("  " + "-" * (len(head) - 2))
    for r in rows[:args.top]:
        band = (f"{r.live_low}-{r.live_high}" if r.anchored else "unanchored")
        print(f"  {r.name[:25]:<26}{r.position:<5}"
              f"{('$' + str(r.live_base)) if r.anchored else '-':>6}{band:>10}"
              f"{r.roster_fit if r.roster_fit is not None else '-':>7}"
              f"{r.lineup_improvement if r.lineup_improvement is not None else '-':>9}"
              f"{'$' + str(r.legal_max):>7}{'$' + str(r.provisional_cap):>6}"
              f"  {r.basis}")
    return 0


def cmd_nominate(args) -> int:
    from .panels import nomination_panel, qb_panel
    print(BANNER)
    session = _session(args)
    matches = [pid for pid, name in session.board.name_by_id.items()
               if args.player.lower() in name.lower()]
    if not matches:
        print(f"no player matches {args.player!r}", file=sys.stderr)
        return 2
    if len(matches) > 1:
        exact = [p for p in matches
                 if session.board.name_by_id[p].lower() == args.player.lower()]
        if len(exact) != 1:
            print(f"{args.player!r} matches {len(matches)} players:",
                  file=sys.stderr)
            for p in matches[:12]:
                print(f"  {session.board.name_by_id[p]}", file=sys.stderr)
            return 2
        matches = exact
    panel = nomination_panel(session, matches[0], current_bid=args.bid,
                             high_bidder=args.leader)
    _print_nomination(panel)
    if args.qb:
        print(json.dumps(qb_panel(session, candidate_id=matches[0],
                                  next_bid=panel["next_legal_bid"]), indent=1))
    return 0


def _print_nomination(p) -> None:
    cap, rails, mkt = p["cap"], p["cap"]["rails"], p["market"]
    bar = "=" * 78
    print(bar)
    print(f"{p['name']}  --  {p['position']} {p['nfl_team']}  bye {p['bye_week']}"
          f"  {p['ppg']} ppg")
    print(bar)
    print(f"  verdict                     {p['verdict']}")
    print(f"  next legal bid              ${p['next_legal_bid']}")
    print(f"  our exact legal maximum     ${p['our_legal_max']}   "
          f"[{p['our_legal_max_basis']}]")
    print(f"  provisional cap             ${p['provisional_cap']}   "
          f"[{cap['basis']}]")
    print(f"  cap label                   {cap['label']}")
    print(f"  bound by                    {cap['bound_by']}")
    print("  --- rails ---")
    print(f"  exact legal maximum         ${rails['legal_max']}")
    if mkt["anchored"]:
        print(f"  market low/base/high        ${mkt['low']} / ${mkt['base']} / "
              f"${mkt['high']}   [{mkt['basis']}]")
    else:
        print("  market low/base/high        UNANCHORED -- no Sleeper anchor. "
              "This is NOT a $1 observation.")
    print(f"  proxy permissive ceiling    "
          f"{('$' + str(rails['proxy_ceiling'])) if rails['proxy_status'] == 'cached' else rails['proxy_status'].upper()}"
          f"   [PROXY/HEURISTIC]")
    print(f"  cached CE bracket           "
          f"{rails['ce_bracket'] or 'none'}   [{rails['ce_status']}]")
    print(f"  manual adjustment           {rails['manual_adjustment'] or 'none'}")
    if cap["disagreement"]:
        print(f"  !! {cap['disagreement_label']}")
    for note in cap["notes"]:
        print(f"  note: {note}")
    print("  --- room ---")
    print(f"  roster fit / +lineup        {p['roster_fit']} / "
          f"{p['lineup_improvement']}   [PROXY/HEURISTIC]")
    print(f"  opponents able at ${p['next_legal_bid']}         "
          f"{p['n_capable_at_next_bid']}")
    print(f"  opponents able at the cap   {p['n_capable_at_cap']}")
    for b in p["recipients"]:
        print(f"    {b['kind']:<18}{b['team']:<22}"
              f"{('$' + str(b['price'])) if b['price'] else '-':>6}  w={b['weight']}")
    print(f"  {p['recipient_warning']}")
    print(bar)


def cmd_verify(args) -> int:
    """The command the morning handoff tells the user to run first."""
    print(BANNER)
    ok = True

    def check(label, good, detail=""):
        nonlocal ok
        ok = ok and good
        print(f"  [{'PASS' if good else 'FAIL'}] {label}"
              + (f"  {detail}" if detail else ""))

    print("installation")
    check("python", sys.version_info >= (3, 9), sys.version.split()[0])
    try:
        import numpy
        check("numpy", True, numpy.__version__)
    except ImportError:
        check("numpy", False, "not importable")
    try:
        from . import board as _b
        check("ceauction.draftday importable", True)
    except Exception as exc:
        check("ceauction.draftday importable", False, str(exc))
        return 1

    print("sources (gitignored, local only)")
    contract, sleeper = Path(args.contract), Path(args.sleeper_csv)
    check(f"contract {contract}", contract.exists())
    check(f"sleeper csv {sleeper}", sleeper.exists())
    if not (contract.exists() and sleeper.exists()):
        print("\nThe tool cannot start without both sources.")
        return 1

    print("board")
    t0 = time.perf_counter()
    from .board import load_board, opening_rows
    board = load_board(contract=contract, sleeper_csv=sleeper,
                       use_cache=not args.no_cache)
    load_s = time.perf_counter() - t0
    check("opening board loads", len(board.state.pool) > 0,
          f"{len(board.state.pool)} players in {load_s:.1f}s")
    t0 = time.perf_counter()
    rows = opening_rows(board)
    rows_s = time.perf_counter() - t0
    anchored = sum(1 for r in rows if r.anchored)
    check("board rows build", len(rows) == len(board.state.pool),
          f"{len(rows)} rows in {rows_s:.2f}s")
    check("anchored players present", anchored > 0,
          f"{anchored} anchored / {len(rows) - anchored} unanchored")
    check("twelve owners", len(board.state.owners) == 12)
    check("board rows under 2s", rows_s < 2.0, f"{rows_s:.2f}s")

    print("session")
    from .session import DraftSession
    session = DraftSession(board, autosave=False)
    check("opening fingerprint", bool(session.fingerprint()),
          session.fingerprint())
    check("nothing sold at open", session.state.rostered_ids == frozenset())

    print("output paths")
    check("draftday dir writable", _writable(DRAFTDAY_DIR))

    print()
    print("VERIFIED -- start the tool with:  "
          ".venv/bin/python -m ceauction.cli draft-day serve" if ok
          else "NOT READY -- see the FAIL lines above.")
    return 0 if ok else 1


def _writable(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write_probe"
        probe.write_text("x", encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        return False


def cmd_mock(args) -> int:
    from .mock import run_mock
    print(BANNER)
    return run_mock(contract=Path(args.contract),
                    sleeper_csv=Path(args.sleeper_csv),
                    use_cache=not args.no_cache)


def add_draftday_parser(sub) -> None:
    p = sub.add_parser(
        "draft-day",
        help="LIVE draft tool: opening board, dashboard, sale entry, undo")
    inner = p.add_subparsers(dest="draftday_command", required=True)

    def common(sp):
        sp.add_argument("--contract",
                        default="local_data/real_player_contract_v1.json",
                        help="normalized real player contract (gitignored)")
        sp.add_argument("--sleeper-csv",
                        default="local_data/sleeper_2qb_values_2026_clean.csv",
                        help="Sleeper 2QB auction values (gitignored)")
        sp.add_argument("--no-cache", action="store_true",
                        help="rebuild the board instead of using the cache")
        sp.add_argument("--quiet", action="store_true")
        return sp

    s = common(inner.add_parser("serve", help="start the local dashboard"))
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--no-browser", action="store_true",
                   help="do not open a browser window")
    s.add_argument("--no-proxy", action="store_true",
                   help="disable the background tactical proxy entirely")
    s.set_defaults(func=cmd_serve)

    s = common(inner.add_parser("board",
                                help="write opening_board.csv / .json"))
    s.add_argument("--top", type=int, default=25,
                   help="rows to print to the terminal")
    s.set_defaults(func=cmd_board)

    s = common(inner.add_parser("nominate",
                                help="one nomination panel, in the terminal"))
    s.add_argument("player", help="player name, or an unambiguous part of one")
    s.add_argument("--bid", type=int, default=None, help="current bid")
    s.add_argument("--leader", default=None, help="current high bidder's owner id")
    s.add_argument("--qb", action="store_true", help="also print the QB panel")
    s.set_defaults(func=cmd_nominate)

    s = common(inner.add_parser("verify", help="check the install and sources"))
    s.set_defaults(func=cmd_verify)

    s = common(inner.add_parser("mock",
                                help="fabricated end-to-end rehearsal"))
    s.set_defaults(func=cmd_mock)
