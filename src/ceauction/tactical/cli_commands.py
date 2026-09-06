"""``ce-lab tactical …`` -- bidders, recipients, endgame, max bid, benchmarks.

Same conventions as the rest of the CLI: validate first, exit non-zero on a
usage error, never show a traceback for something a caller could reasonably get
wrong, and say on every screen whether the inputs are fabricated.

No command here prints a single recommended number. What they print are named
thresholds with the rule that produced each one, and the label saying whether
the answer came from the proxy or from the equity engine.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import List, Optional, Sequence

from ..auction.completion import CompletionSettings
from .bidders import (BIDDER_SCENARIOS, DEFAULT_BIDDER_SCENARIO,
                      assess_room_bidders, format_bidders)
from .board import BoardSettings, continue_shared_board, format_board
from .demo import build_tactical_demo, demo_sales
from .endgame import assess_endgame, format_endgame
from .maxbid import (DEFAULT_SCENARIOS, TacticalCache, TacticalScenario,
                     TacticalSettings, evaluate_tactical, format_tactical)
from .nested import format_nested_ladder
from .precompute import precompute, write_report

__all__ = ["add_tactical_parser", "dispatch", "UsageError"]

_FABRICATED = (
    "INPUTS ARE FABRICATED. The board, the anchors and the sales are generated\n"
    "by formulas and describe no real player. Nothing below is a real price.")

_PRICE_NOTE = (
    "Four prices are kept apart: the Sleeper display anchor, the expected\n"
    "clearing price, the CE reservation price and the tactical maximum bid.\n"
    "A tactical maximum uses the other three and replaces none of them.")


class UsageError(Exception):
    """Something the caller can fix. Reported as a message, never a traceback."""


def _write_json(path: Optional[str], blob) -> None:
    if not path:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(blob, indent=2, default=str),
                          encoding="utf-8")
    print(f"wrote {path}")


def _demo(args):
    """The joined fabricated world every command runs on."""
    if not getattr(args, "demo", False):
        raise UsageError(
            "--demo is required: loading a live auction room from a file is "
            "not implemented, and building one from guesses would be a "
            "counterfactual pretending to be a room")
    d = build_tactical_demo()
    for sale in getattr(args, "sale", None) or []:
        d = _apply_cli_sale(d, sale)
    if getattr(args, "demo_sales", False):
        d = d.with_market(d.market.observe_all(demo_sales(d, 12)))
    return d


def _apply_cli_sale(d, spec: str):
    """``PLAYER_ID:OWNER:PRICE`` -- record a real sale into state and market."""
    parts = spec.split(":")
    if len(parts) != 3:
        raise UsageError(
            f"--sale must be PLAYER_ID:OWNER:PRICE, got {spec!r}")
    try:
        pid, owner, price = int(parts[0]), parts[1], int(parts[2])
    except ValueError:
        raise UsageError(f"--sale player id and price must be integers: {spec!r}")
    from ..auction.state import AuctionError
    from ..market.live import SaleObservation
    try:
        state = d.state.apply_purchase(pid, owner, price)
    except AuctionError as exc:
        raise UsageError(f"--sale {spec!r} is not a legal purchase: {exc}")
    d = d.with_state(state)
    key = d.key_for(pid)
    if key:
        p = d.prior.by_key[key]
        d = d.with_market(d.market.observe(SaleObservation(
            key, p.position, price, owner, len(d.market.observations),
            p.base_price, p.display_anchor, notes="entered from the CLI")))
    return d


def _candidate(args, d) -> int:
    cid = getattr(args, "candidate", None)
    if cid is None:
        return d.candidate().player_id
    if cid not in d.state.spec_by_id:
        raise UsageError(f"player {cid} is not in this auction's pool")
    if not d.state.is_available(cid):
        raise UsageError(f"player {cid} is no longer available")
    return int(cid)


def _scenarios(args) -> Sequence[TacticalScenario]:
    names = getattr(args, "market_scenario", None) or ["low", "base", "high"]
    perf = getattr(args, "performance_scenario", None) or "fabricated_base"
    bidder = getattr(args, "bidder_scenario", None) or DEFAULT_BIDDER_SCENARIO
    for n in names:
        if n not in ("low", "base", "high"):
            raise UsageError(f"market scenario must be low/base/high, got {n!r}")
    if bidder not in BIDDER_SCENARIOS:
        raise UsageError(
            f"unknown bidder scenario {bidder!r}; "
            f"known: {', '.join(sorted(BIDDER_SCENARIOS))}")
    base_name = "base" if "base" in names else names[0]
    out = []
    for n in names:
        out.append(TacticalScenario(
            scenario_id=n, market_scenario=n, performance_scenario=perf,
            bidder_scenario=bidder, is_base=(n == base_name)))
    return tuple(out)


def _settings(args) -> TacticalSettings:
    mode = getattr(args, "mode", "immediate")
    if mode not in ("immediate", "audited"):
        raise UsageError(f"--mode must be immediate or audited, got {mode!r}")
    s = TacticalSettings(mode=mode)
    board = replace(s.board, seed=getattr(args, "seed", None) or s.board.seed)
    completion = s.completion
    if mode == "audited":
        sims = getattr(args, "sims", None) or 2000
        sel = getattr(args, "selection_sims", None) or sims
        completion = replace(completion, selection_sims=sel,
                             evaluation_sims=sims, rival_selection="proxy")
    return replace(s, board=board, completion=completion,
                   max_prices=getattr(args, "max_prices", None) or s.max_prices,
                   refine=bool(getattr(args, "refine", False)),
                   max_recipients=getattr(args, "max_recipients", None)
                   or s.max_recipients,
                   max_worlds=getattr(args, "max_worlds", None) or s.max_worlds)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_validate(args) -> int:
    """Self-checks that need no simulation: legality, uniqueness, separation."""
    d = _demo(args)
    cid = _candidate(args, d)
    checks: List[tuple] = []

    eg = assess_endgame(d.state, cid, current_price=args.price,
                        increment=args.increment)
    checks.append(("every legal maximum is within the financial maximum",
                   all(o.candidate_legal_max <= o.financial_max_bid
                       for o in eg.owners)))
    checks.append(("$1 survives for every open slot",
                   all(o.budget_remaining >= o.reserved_for_open_slots
                       for o in eg.owners)))

    rows = assess_room_bidders(d.state, cid, market=d.market,
                               candidate_key=d.key_for(cid), costs=d.costs)
    checks.append(("no willingness exceeds a candidate-specific legal maximum",
                   all(r.high <= r.candidate_legal_max for r in rows)))
    checks.append(("no willingness is reported as a probability",
                   all("NOT a fitted probability" in r.score_kind for r in rows)))

    board = continue_shared_board(d.state, costs=d.costs, market=d.market,
                                  key_by_id=d.key_by_id)
    checks.append(("shared board assigns no player twice",
                   not board.has_duplicates))
    checks.append(("shared board completes every rival roster legally",
                   board.all_complete and board.state.is_valid))
    checks.append(("shared board is deterministic under a fixed seed",
                   continue_shared_board(
                       d.state, costs=d.costs, market=d.market,
                       key_by_id=d.key_by_id).fingerprint()
                   == board.fingerprint()))
    other = continue_shared_board(
        d.state, settings=BoardSettings(seed=987654321), costs=d.costs,
        market=d.market, key_by_id=d.key_by_id)
    checks.append(("a different seed is allowed to change the allocation",
                   other.fingerprint() != board.fingerprint()))

    print(_FABRICATED)
    print()
    print(_PRICE_NOTE)
    print()
    ok = True
    for name, passed in checks:
        ok = ok and passed
        print(f"  [{'ok' if passed else 'FAIL'}] {name}")
    print()
    print(f"  {len(checks)} checks, {'all passed' if ok else 'FAILURES ABOVE'}")
    _write_json(args.json_out, {"checks": [{"name": n, "passed": p}
                                           for n, p in checks], "ok": ok})
    return 0 if ok else 1


def cmd_bidders(args) -> int:
    d = _demo(args)
    cid = _candidate(args, d)
    scenario = args.bidder_scenario or DEFAULT_BIDDER_SCENARIO
    if scenario not in BIDDER_SCENARIOS:
        raise UsageError(
            f"unknown bidder scenario {scenario!r}; "
            f"known: {', '.join(sorted(BIDDER_SCENARIOS))}")
    rows = assess_room_bidders(d.state, cid, scenario=scenario,
                               market=d.market, candidate_key=d.key_for(cid),
                               costs=d.costs)
    print(_FABRICATED)
    print()
    print(format_bidders(rows))
    _write_json(args.json_out, [r.to_dict() for r in rows])
    return 0


def cmd_recipients(args) -> int:
    from .recipients import enumerate_recipients, format_recipients
    d = _demo(args)
    cid = _candidate(args, d)
    if args.leader is not None and args.leader not in d.state.owner_by_id:
        raise UsageError(f"no owner {args.leader!r} in the room")
    rs = enumerate_recipients(
        d.state, cid, current_price=args.price, increment=args.increment,
        current_leader=args.leader,
        scenario=args.bidder_scenario or DEFAULT_BIDDER_SCENARIO,
        market=d.market, candidate_key=d.key_for(cid), costs=d.costs,
        max_named=args.max_recipients or 3)
    print(_FABRICATED)
    print()
    print(format_recipients(rs))
    _write_json(args.json_out, rs.to_dict())
    return 0


def cmd_endgame(args) -> int:
    d = _demo(args)
    cid = _candidate(args, d)
    eg = assess_endgame(d.state, cid, current_price=args.price,
                        increment=args.increment)
    print(_FABRICATED)
    print()
    print(format_endgame(eg))
    _write_json(args.json_out, eg.to_dict())
    return 0


def cmd_board(args) -> int:
    d = _demo(args)
    bs = BoardSettings(seed=args.seed or BoardSettings().seed,
                       market_scenario=(args.market_scenario or ["base"])[0],
                       bidder_scenario=args.bidder_scenario
                       or DEFAULT_BIDDER_SCENARIO)
    board = continue_shared_board(d.state, settings=bs, costs=d.costs,
                                  market=d.market, key_by_id=d.key_by_id)
    print(_FABRICATED)
    print()
    print(format_board(board))
    _write_json(args.json_out, board.to_dict())
    return 0


def cmd_max_bid(args) -> int:
    d = _demo(args)
    cid = _candidate(args, d)
    if args.leader is not None and args.leader not in d.state.owner_by_id:
        raise UsageError(f"no owner {args.leader!r} in the room")
    settings = _settings(args)
    scenarios = _scenarios(args)
    cache = TacticalCache()
    t0 = time.perf_counter()
    from .joint import ConservationError
    try:
        r = evaluate_tactical(
            d.state, d.cast, d.costs, cid, settings=settings,
            scenarios=scenarios, market=d.market, candidate_key=d.key_for(cid),
            key_by_id=d.key_by_id, current_price=args.price,
            increment=args.increment, current_leader=args.leader, cache=cache,
            regime=getattr(args, "regime", None))
    except ConservationError as exc:
        raise UsageError(
            f"audited evaluation refused: {exc}")
    print(_FABRICATED)
    print()
    print(_PRICE_NOTE)
    print()
    print(format_tactical(r))
    if r.nested_ladder is not None:
        print()
        print(format_nested_ladder(r.nested_ladder))
    cached = cache.stats()
    print(f"\nwall clock {time.perf_counter() - t0:.2f}s   "
          f"cache entries {cached['entries']} hits {cached['hits']} "
          f"misses {cached['misses']}")
    if not r.is_audited:
        print(f"\n  {r.PROXY_BANNER}. Re-run with --mode audited for a "
              f"CE-backed threshold.")
    _write_json(args.json_out, r.to_dict())
    return 0


def cmd_precompute(args) -> int:
    d = _demo(args)
    if args.player_id:
        ids = []
        for pid in args.player_id:
            if pid not in d.state.spec_by_id:
                raise UsageError(f"player {pid} is not in this auction's pool")
            if not d.state.is_available(pid):
                raise UsageError(f"player {pid} is no longer available")
            ids.append(int(pid))
    else:
        n = args.top or 3
        if n < 1:
            raise UsageError("--top must be at least 1")
        ids = [d.candidate(i).player_id for i in range(n)]
    settings = _settings(args)
    scenarios = _scenarios(args)
    cache = TacticalCache()

    def progress(i, total, cid, reused):
        print(f"  [{i}/{total}] player {cid}"
              f"{'  (reused: cache key matched)' if reused else ''}")

    print(_FABRICATED)
    print()
    print(f"precomputing {len(ids)} candidate(s) in {settings.mode} mode, "
          f"scenarios {[s.scenario_id for s in scenarios]}")
    rep = precompute(d.state, d.cast, d.costs, ids, cache=cache,
                     settings=settings, scenarios=scenarios, market=d.market,
                     key_by_id=d.key_by_id, current_price=args.price,
                     increment=args.increment, current_leader=args.leader,
                     progress=progress, fabricated=True)
    if args.resume:
        print("\nsecond pass over the same inputs (resume must reuse):")
        rep2 = precompute(d.state, d.cast, d.costs, ids, cache=cache,
                          settings=settings, scenarios=scenarios,
                          market=d.market, key_by_id=d.key_by_id,
                          current_price=args.price, increment=args.increment,
                          current_leader=args.leader, progress=progress,
                          fabricated=True)
        print(f"  reused {rep2.n_reused}/{len(ids)} in "
              f"{rep2.total_runtime_s:.3f}s")
    print()
    head = f"  {'player':>8}{'legal max':>11}{'robust':>9}{'base':>7}{'permissive':>12}{'s':>8}"
    print(head)
    for e in rep.entries:
        print(f"  {e.candidate_id:>8}{e.legal_max:>11}"
              f"{str(e.robust):>9}{str(e.base):>7}{str(e.permissive):>12}"
              f"{e.runtime_s:>8.2f}")
    print(f"\ntotal {rep.total_runtime_s:.2f}s, computed {rep.n_computed}, "
          f"reused {rep.n_reused}")
    if args.out:
        write_report(rep, args.out)
        print(f"wrote {args.out}")
    _write_json(args.json_out, rep.to_dict())
    return 0


def cmd_benchmark(args) -> int:
    """Measure every stage a live decision would pay for. Nothing is hidden."""
    d = _demo(args)
    cid = _candidate(args, d)
    scenarios = _scenarios(args)
    rows = []

    t = time.perf_counter()
    assess_endgame(d.state, cid, current_price=args.price,
                   increment=args.increment)
    rows.append(("endgame arithmetic", time.perf_counter() - t))

    t = time.perf_counter()
    assess_room_bidders(d.state, cid, market=d.market,
                        candidate_key=d.key_for(cid), costs=d.costs)
    rows.append(("bidder willingness, whole room", time.perf_counter() - t))

    from .recipients import enumerate_recipients
    t = time.perf_counter()
    enumerate_recipients(d.state, cid, current_price=args.price,
                         increment=args.increment, current_leader=args.leader,
                         market=d.market, candidate_key=d.key_for(cid),
                         costs=d.costs)
    rows.append(("named recipient branches", time.perf_counter() - t))

    t = time.perf_counter()
    continue_shared_board(d.state, costs=d.costs, market=d.market,
                          key_by_id=d.key_by_id)
    rows.append(("shared-board continuation", time.perf_counter() - t))

    cache = TacticalCache()
    imm = replace(_settings(args), mode="immediate")
    t = time.perf_counter()
    evaluate_tactical(d.state, d.cast, d.costs, cid, settings=imm,
                      scenarios=scenarios, market=d.market,
                      candidate_key=d.key_for(cid), key_by_id=d.key_by_id,
                      current_price=args.price, increment=args.increment,
                      current_leader=args.leader, cache=cache)
    immediate_cold = time.perf_counter() - t
    rows.append(("immediate max-bid, cold", immediate_cold))

    t = time.perf_counter()
    evaluate_tactical(d.state, d.cast, d.costs, cid, settings=imm,
                      scenarios=scenarios, market=d.market,
                      candidate_key=d.key_for(cid), key_by_id=d.key_by_id,
                      current_price=args.price, increment=args.increment,
                      current_leader=args.leader, cache=cache)
    cached = time.perf_counter() - t
    rows.append(("immediate max-bid, cache hit", cached))

    audited = None
    if args.audited:
        aud = replace(_settings(args), mode="audited")
        aud = replace(aud, completion=replace(
            aud.completion, selection_sims=args.sims or 1200,
            evaluation_sims=args.sims or 1200, rival_selection="proxy"))
        one = (TacticalScenario("base", "base",
                                args.performance_scenario or "fabricated_base",
                                is_base=True),)
        t = time.perf_counter()
        evaluate_tactical(d.state, d.cast, d.costs, cid, settings=aud,
                          scenarios=one, market=d.market,
                          candidate_key=d.key_for(cid), key_by_id=d.key_by_id,
                          current_price=args.price, increment=args.increment,
                          current_leader=args.leader,
                          prices=[args.price + args.increment])
        audited = time.perf_counter() - t
        rows.append(("audited comparison, one price, one scenario", audited))

    print(_FABRICATED)
    print()
    print(f"  {'stage':<46}{'seconds':>10}")
    print("  " + "-" * 56)
    for name, secs in rows:
        print(f"  {name:<46}{secs:>10.3f}")
    print()
    target = 10.0
    met = immediate_cold <= target
    print(f"  10-second bid timer: immediate cold path "
          f"{'MEETS' if met else 'DOES NOT MEET'} the target "
          f"({immediate_cold:.2f}s of {target:.0f}s).")
    print(f"  cache hit path {cached * 1000:.1f}ms.")
    if audited is not None:
        print(f"  audited path {audited:.2f}s for ONE price and ONE scenario -- "
              f"precompute it, never run it on the clock.")
    _write_json(args.json_out, {
        "stages": [{"stage": n, "seconds": round(s, 4)} for n, s in rows],
        "immediate_cold_s": round(immediate_cold, 4),
        "cache_hit_s": round(cached, 6),
        "audited_s": None if audited is None else round(audited, 4),
        "target_s": target, "meets_10s_target": met})
    return 0 if met else 0


_COMMANDS = {
    "validate": cmd_validate,
    "bidders": cmd_bidders,
    "recipients": cmd_recipients,
    "endgame": cmd_endgame,
    "board": cmd_board,
    "max-bid": cmd_max_bid,
    "precompute": cmd_precompute,
    "benchmark": cmd_benchmark,
}


def dispatch(args) -> int:
    fn = _COMMANDS.get(args.tactical_command)
    if fn is None:
        print(f"unknown tactical command {args.tactical_command!r}",
              file=sys.stderr)
        return 2
    try:
        return fn(args)
    except (UsageError, ValueError, KeyError, IndexError) as exc:
        print(f"\n{type(exc).__name__.replace('Error', '')}: {exc}",
              file=sys.stderr)
        return 2


def add_tactical_parser(sub) -> None:
    p = sub.add_parser(
        "tactical",
        help="named recipients, endgame arithmetic and tactical max bids")
    inner = p.add_subparsers(dest="tactical_command", required=True)

    def common(sp, *, price=True, scenarios=False, mode=False):
        sp.add_argument("--demo", action="store_true",
                        help="use the fabricated joined auction+market world "
                             "(currently the only supported input)")
        sp.add_argument("--demo-sales", action="store_true",
                        help="apply the scripted fabricated sale sequence")
        sp.add_argument("--sale", action="append",
                        help="record a sale: PLAYER_ID:OWNER:PRICE; repeatable")
        sp.add_argument("--candidate", type=int, default=None,
                        help="player id (default: best still on the board)")
        sp.add_argument("--owner", default=None,
                        help="our owner id (default: the demo focus owner)")
        if price:
            sp.add_argument("--price", type=int, default=None,
                            help="the current high bid, if any")
            sp.add_argument("--increment", type=int, default=1)
            sp.add_argument("--leader", default=None,
                            help="the current high bidder's owner id")
        sp.add_argument("--bidder-scenario", default=None,
                        choices=sorted(BIDDER_SCENARIOS))
        if scenarios:
            sp.add_argument("--market-scenario", action="append",
                            choices=["low", "base", "high"],
                            help="repeatable; default low base high")
            sp.add_argument("--performance-scenario", default=None,
                            help="names the projection reading in force")
            sp.add_argument("--max-recipients", type=int, default=None)
            sp.add_argument("--max-prices", type=int, default=None)
            sp.add_argument("--refine", action="store_true",
                            help="walk every integer in the transition gap")
        if mode:
            sp.add_argument("--mode", default="immediate",
                            choices=["immediate", "audited"])
            sp.add_argument("--sims", type=int, default=None,
                            help="holdout seasons per arm in audited mode")
            sp.add_argument("--selection-sims", type=int, default=None,
                            help="selection-sample seasons (default: --sims)")
            sp.add_argument("--max-worlds", type=int, default=None,
                            help="reconciled joint worlds compared per arm")
            sp.add_argument("--regime", default=None,
                            help="roster-strength regime label; part of the "
                                 "cache key")
        sp.add_argument("--seed", type=int, default=None,
                        help="shared-board allocation seed")
        sp.add_argument("--json-out", default=None,
                        help="write the full result as JSON")

    s = inner.add_parser("validate",
                         help="self-checks needing no simulation")
    common(s)
    s = inner.add_parser("bidders", help="scenario willingness for every owner")
    common(s)
    s = inner.add_parser("recipients",
                         help="who gets the player if we stop bidding")
    common(s)
    s.add_argument("--max-recipients", type=int, default=None)
    s = inner.add_parser("endgame", help="room arithmetic; no simulation")
    common(s)
    s = inner.add_parser("board", help="one shared-board continuation")
    common(s, price=False)
    s.add_argument("--market-scenario", action="append",
                   choices=["low", "base", "high"])
    s = inner.add_parser("max-bid", help="named tactical maximum bids")
    common(s, scenarios=True, mode=True)
    s = inner.add_parser("precompute",
                         help="bounded precomputation for a named candidate set")
    common(s, scenarios=True, mode=True)
    s.add_argument("--top", type=int, default=None,
                   help="top N remaining players by fabricated projection")
    s.add_argument("--player-id", type=int, action="append",
                   help="explicit candidate id; repeatable")
    s.add_argument("--resume", action="store_true",
                   help="run a second pass to show that resume reuses only "
                        "entries whose full cache key matches")
    s.add_argument("--out", default=None,
                   help="write the precompute report JSON")
    s = inner.add_parser("benchmark", help="runtime for every live stage")
    common(s, scenarios=True, mode=True)
    s.add_argument("--audited", action="store_true",
                   help="also time one audited CE comparison")

    p.set_defaults(func=dispatch)
