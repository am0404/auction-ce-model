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


def cmd_context(args) -> int:
    """The controlled one-factor roster-context experiment."""
    from .context_experiment import InvalidExperiment, run
    sims = args.sims or 4000
    sel = args.selection_sims or max(400, sims // 4)
    print(_FABRICATED)
    print()
    print("CONTROLLED ROSTER-CONTEXT EXPERIMENT")
    print("Exactly one factor varies: the projected scoring of the players the")
    print("focus team ALREADY OWNS. Budget, spend, roster size, open slots,")
    print("positions, the remaining board, its costs, every rival, the market,")
    print("the candidate, its price, the recipient and every seed are held")
    print("identical and asserted so before any season is simulated.")
    print()
    try:
        blob = run(holdout_sims=sims, selection_sims=sel)
    except InvalidExperiment as exc:
        raise UsageError(str(exc))
    print()
    print(f"  {'regime':<16}{'scale':>7}{'p(playoff)':>12}{'p(bye)':>9}"
          f"{'CE':>9}{'rank':>6}   candidate delta (95%)")
    for name, r in blob["regimes"].items():
        pre = r["pre_acquisition"]
        row = r["rows"][0]
        hw = (row["ci95"][1] - row["ci95"][0]) / 2
        print(f"  {name:<16}{pre['strength_scale']:>7.2f}"
              f"{pre['playoff_probability']:>12.3f}"
              f"{pre['bye_probability']:>9.3f}"
              f"{pre['championship_equity']:>9.4f}{pre['ce_rank']:>6}   "
              f"{row['delta_ce']:+.5f} +/-{hw:.5f}  {row['verdict']}")
    print()
    print(f"  holdout {blob['holdout_sims']:,} seasons, selection "
          f"{blob['selection_sims']:,} seasons (independent)")
    print(f"  seeds: board {blob['board_seed']}, selection "
          f"{blob['selection_seed']}, holdout {blob['holdout_seed']}")
    print(f"  runtime {blob['runtime_s']:.0f}s")
    _write_json(args.json_out, blob)
    return 0


def cmd_signal_power(args) -> int:
    """Estimator power by candidate tier, and the sampling plan it implies."""
    from .power_experiment import run, verdict
    if args.pilot_sims < 2:
        raise UsageError("--pilot-sims must be at least 2")
    if args.sims < args.pilot_sims:
        raise UsageError(
            "--sims (confirmatory) must be at least --pilot-sims; a "
            "confirmation smaller than the pilot that triggered it is not a "
            "confirmation")
    if args.target_half_width <= 0:
        raise UsageError("--target-half-width must be positive")
    if args.cap_sims < args.sims:
        raise UsageError("--cap-sims must be at least --sims")
    seeds = tuple(args.alloc_seed or (20260906, 424242, 987654321))
    if len(set(seeds)) < 2:
        raise UsageError("give at least two distinct --alloc-seed values to "
                         "separate allocation noise from season noise")
    print(_FABRICATED)
    print()
    print("ESTIMATOR POWER BY CANDIDATE TIER")
    print("A weak candidate returning 'unresolved' is the estimator refusing")
    print("false precision, not the estimator failing. What this measures is")
    print("whether a player who genuinely matters produces an effect that")
    print("resolves at a practical sample size.")
    print()
    blob = run(pilot_sims=args.pilot_sims, confirm_sims=args.sims,
               cap_sims=args.cap_sims,
               target_half_width=args.target_half_width,
               alloc_seeds=seeds)
    v, why = verdict(blob)
    blob["verdict"], blob["verdict_reason"] = v, why
    print()
    print(f"  pilot {blob['pilot_sims']:,} (seed {blob['pilot_seed']}) -> "
          f"confirmatory {blob['confirm_sims']:,} (seed "
          f"{blob['confirm_seed']}); cap {blob['cap_sims']:,}")
    print(f"  allocation seeds {list(seeds)}")
    print(f"  runtime {blob['runtime_s']:.0f}s")
    print()
    print(f"  VERDICT: {v} -- {why}")
    _write_json(args.json_out, blob)
    return 0


def cmd_real_pilot(args) -> int:
    """The targeted real-board tactical pilot. Twelve players, not a board."""
    from pathlib import Path as _P
    from .realpilot import PilotInputs, SanitizationError
    from .real_pilot_experiment import run

    inputs = PilotInputs(
        contract=_P(args.contract), sleeper_csv=_P(args.sleeper_csv),
        out_dir=_P(args.out_dir), pool_limit=args.pool_limit,
        market_scenario=args.market_scenario,
        performance_scenario=args.performance_scenario)
    missing = inputs.missing()
    if missing:
        raise UsageError(
            "real inputs are missing: " + ", ".join(missing) + ".\n"
            "  Build the contract with `ce-lab ingest --projections ... "
            "--contract-out local_data/real_player_contract_v1.json`,\n"
            "  and place the cleaned Sleeper export at "
            "local_data/sleeper_2qb_values_2026_clean.csv.\n"
            "  Both live under local_data/, which is gitignored.")
    if "local_data" not in _P(args.out_dir).parts:
        raise UsageError(
            f"--out-dir must sit under local_data/ (got {args.out_dir!r}); "
            f"real player-level output must never reach version control")
    if args.sims < 100:
        raise UsageError("--sims must be at least 100")
    if args.per_position < 1:
        raise UsageError("--per-position must be at least 1")
    seeds = tuple(args.alloc_seed or (20260906, 424242, 987654321))

    print("REAL-BOARD TACTICAL PILOT")
    print("Real projections and real Sleeper anchors. Player-level output goes")
    print("ONLY to local_data/. This is a PILOT of a dozen players at the")
    print("empty-room state; it is not a draft board and prices nothing")
    print("mid-auction.")
    print()
    report, local = run(
        inputs, holdout_sims=args.sims, selection_sims=args.selection_sims,
        per_position=args.per_position,
        positions=tuple(args.position or ("QB", "RB", "WR", "TE")),
        alloc_seeds=seeds, base_price_only=args.base_price_only,
        ladders=not args.base_price_only,
        runtime_budget_s=args.runtime_budget)

    out = _P(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "real_board_pilot.json").write_text(
        json.dumps(local, indent=2, default=str), encoding="utf-8")
    names = [r.get("name") for r in local["candidates"] if r.get("name")]
    try:
        report.check(names)
    except SanitizationError as exc:
        raise UsageError(f"refusing to emit the sanitized report: {exc}")
    if args.report_out:
        _write_json(args.report_out, report.to_dict())
    print()
    print(f"  player-level output -> {out / 'real_board_pilot.json'} (IGNORED)")
    r = report.results
    print(f"  audited {len(r['audited'])}, resolved {r['n_resolved']}, "
          f"unresolved {r['n_unresolved']}, proxy-only {len(r['proxy_only'])}")
    print(f"  runtime {report.runtime['total_s']:.0f}s")
    return 0


def cmd_allocation_ensemble(args) -> int:
    """Exchangeable allocation ensemble + opening-symmetry estimator."""
    from pathlib import Path as _P
    from .realpilot import PilotInputs
    from .ensemble_experiment import run

    inputs = PilotInputs(
        contract=_P(args.contract), sleeper_csv=_P(args.sleeper_csv),
        out_dir=_P(args.out_dir), pool_limit=args.pool_limit,
        market_scenario=args.market_scenario,
        performance_scenario=args.performance_scenario)
    missing = inputs.missing()
    if missing:
        raise UsageError(
            "real inputs are missing: " + ", ".join(missing) + ".\n"
            "  See `ce-lab tactical real-pilot` for the setup; both files live "
            "under local_data/, which is gitignored.")
    if "local_data" not in _P(args.out_dir).parts:
        raise UsageError(
            f"--out-dir must sit under local_data/ (got {args.out_dir!r})")
    if args.draws < 2:
        raise UsageError(
            "--draws must be at least 2; a single allocation draw has no "
            "between-allocation interval and is exactly the estimate this "
            "command exists to replace")
    if args.shock < 0.0:
        raise UsageError("--shock must be non-negative")
    if args.tie_break not in ("mechanical", "shock"):
        raise UsageError("--tie-break must be mechanical or shock")
    if args.sims < 100:
        raise UsageError("--sims must be at least 100")

    print("ALLOCATION ENSEMBLE + OPENING SYMMETRY")
    print("A single future auction is not an estimate of opening value. This")
    print("runs a balanced ensemble in which every initially identical owner")
    print("occupies every priority slot equally often, and reports allocation")
    print("uncertainty separately from season uncertainty.")
    print()
    out, local = run(
        inputs, k=args.draws, holdout_sims=args.sims,
        selection_sims=args.selection_sims, shock=args.shock,
        shock_scenario=args.shock_scenario, secondary_k=args.secondary_draws,
        runtime_budget_s=args.runtime_budget)

    d = _P(args.out_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / "allocation_ensemble.json").write_text(
        json.dumps(local, indent=2, default=str), encoding="utf-8")
    if args.json_out:
        _write_json(args.json_out, out)
    print()
    print("  CONVERGENCE BY K")
    print(f"    {'k':>3}{'mean':>11}{'between SD':>12}{'SE(mean)':>11}"
          f"{'CI95':>26}{'sign+':>7}  verdict")
    for row in out["primary"]["convergence"]:
        ci = ("n/a" if row["ci95"] is None
              else f"[{row['ci95'][0]:+.5f},{row['ci95'][1]:+.5f}]")
        se = "n/a" if row["se_of_mean"] is None else f"{row['se_of_mean']:.5f}"
        print(f"    {row['k']:>3}{row['mean']:>+11.5f}"
              f"{row['between_sd']:>12.5f}{se:>11}{ci:>26}"
              f"{row['sign_positive']:>7.2f}  {row['verdict']}")
    print()
    print(f"  player-level output -> {d / 'allocation_ensemble.json'} (IGNORED)")
    print(f"  runtime {out['runtime_s']:.0f}s")
    return 0


def cmd_marginal_diagnostics(args) -> int:
    """Quota-free with/without diagnostics over a search-effort ladder."""
    from pathlib import Path as _P
    from ..auction.completion import CompletionSettings
    from ..auction.proxy import ProxyEvaluator
    from .convergence import (CONVERGED, DEFAULT_LADDER,
                              DIAGNOSTIC_NOT_CONVERGED, EffortLevel,
                              converged_diagnose)
    from .realpilot import (NO_CONTINGENCY_MODEL, PilotInputs,
                            SanitizationError, SanitizedReport,
                            load_real_board, select_candidates)

    inputs = PilotInputs(
        contract=_P(args.contract), sleeper_csv=_P(args.sleeper_csv),
        out_dir=_P(args.out_dir), pool_limit=args.pool_limit,
        market_scenario=args.market_scenario,
        performance_scenario=args.performance_scenario)
    missing = inputs.missing()
    if missing:
        raise UsageError(
            "real inputs are missing: " + ", ".join(missing) + ". See "
            "`ce-lab tactical real-pilot` for setup; both live under "
            "local_data/, which is gitignored.")
    if "local_data" not in _P(args.out_dir).parts:
        raise UsageError(
            f"--out-dir must sit under local_data/ (got {args.out_dir!r})")
    if args.per_position < 1:
        raise UsageError("--per-position must be at least 1")
    if args.tolerance <= 0:
        raise UsageError("--tolerance must be positive")
    ladder = DEFAULT_LADDER
    if args.effort:
        try:
            ladder = tuple(EffortLevel(*(int(x) for x in spec.split("/")))
                           for spec in args.effort)
        except (ValueError, TypeError):
            raise UsageError(
                "--effort takes BEAM/POOL pairs, e.g. --effort 64/60 "
                "--effort 160/90")
        if len(ladder) < 2:
            raise UsageError(
                "the effort ladder needs at least two levels; convergence is "
                "a statement about what happens when effort increases")
        widths = [l.beam_width for l in ladder]
        if widths != sorted(widths):
            raise UsageError("--effort levels must increase in beam width")
    if args.max_effort is not None:
        ladder = tuple(l for l in ladder if l.beam_width <= args.max_effort)
        if len(ladder) < 2:
            raise UsageError(
                f"--max-effort {args.max_effort} leaves fewer than two ladder "
                f"levels")

    print("QUOTA-FREE MARGINAL DIAGNOSTICS OVER A SEARCH-EFFORT LADDER")
    print("Roster composition is an OUTPUT of legal completion, never an input.")
    print("Every level's completions are accumulated into one union and")
    print("re-scored on a common ruler, so a wider beam cannot lose a better")
    print("construction a narrower one found. Search-effort monotonicity then")
    print("holds by construction; the open question is whether it SETTLES.")
    print()
    board = load_real_board(inputs)
    px = ProxyEvaluator(board.state.pool, board.state.settings, 16, 7)
    base = CompletionSettings(beam_width=64, candidate_pool=60,
                              proxy_candidates=48, finalists=6,
                              max_candidates=400, proxy_reps=16)
    positions = tuple(args.position or ("QB", "RB", "WR", "TE"))
    cands, notes = select_candidates(board, per_position=args.per_position,
                                     positions=positions)
    tols = (args.tolerance, args.tolerance * 2.0)

    print(f"  ladder: {[l.label for l in ladder]}   tolerances: {tols}")
    print()
    print(f"  {'pos':<4}{'tier':<11}{'$':>5}{'@price':>9}{'@$1':>8}{'start%':>8}"
          f"  {'role':<21}{'policy':<30}{'status':<26}effort")
    rows, local_rows = [], []
    for c in cands:
        d, rep = converged_diagnose(board, c, proxy=px, ladder=ladder,
                                    base=base, tolerances=tols)
        blob = rep.to_dict()
        status = blob["status"][str(tols[0])]
        rows.append({**d.to_dict(), "convergence": blob})
        local_rows.append({"name": board.name_by_id.get(c.player_id),
                           **d.to_dict(include_identity=True),
                           "convergence": blob})
        print(f"  {c.position:<4}{c.tier:<11}{d.price:>5}"
              f"{d.lineup_improvement:>9.2f}{d.improvement_at_min:>8.2f}"
              f"{d.start_share:>8.0%}  {d.role:<21}{d.policy:<30}"
              f"{status:<26}{blob['effort_required'][str(tols[0])] or '-'}")

    aud = sum(1 for r in rows if r["policy"] == "4000-season audit")
    unres = sum(1 for r in rows if r["policy"].startswith("unresolved"))
    prox = len(rows) - aud - unres
    search_bad = sum(len(r["convergence"]["search_violations"]) for r in rows)
    price_bad = sum(len(r["convergence"]["price_violations"]) for r in rows)
    print()
    print(f"  audited {aud}, proxy-only {prox}, UNRESOLVED {unres}")
    for t in tols:
        n = sum(1 for r in rows
                if r["convergence"]["status"][str(t)] == CONVERGED)
        print(f"  converged at tolerance {t}: {n}/{len(rows)}")
    print(f"  search-effort nesting violations: {search_bad} "
          f"(zero expected: the union makes it structural)")
    print(f"  price nesting violations:         {price_bad}")
    print(f"  contingency: {NO_CONTINGENCY_MODEL}")

    out = _P(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "marginal_convergence.json").write_text(
        json.dumps(local_rows, indent=2, default=str), encoding="utf-8")
    report = SanitizedReport(
        coverage=board.coverage,
        selection={"candidates": len(cands), "substitutions": notes,
                   "positions": list(positions),
                   "ladder": [l.to_dict() for l in ladder],
                   "tolerances": list(tols)},
        diagnostics=rows,
        results={"audited": aud, "proxy_only": prox, "unresolved": unres,
                 "search_violations": search_bad,
                 "price_violations": price_bad,
                 "converged": {str(t): sum(
                     1 for r in rows
                     if r["convergence"]["status"][str(t)] == CONVERGED)
                     for t in tols}},
        runtime={}, warnings=notes,
        assumptions=[NO_CONTINGENCY_MODEL,
                     "no positional quota enters diagnostics or sampling",
                     "tolerances are numerical stability thresholds, not "
                     "claims of economic materiality",
                     "market scenario: " + args.market_scenario,
                     "performance scenario: " + args.performance_scenario])
    names = [r.get("name") for r in local_rows if r.get("name")]
    try:
        report.check(names)
    except SanitizationError as exc:
        raise UsageError(f"refusing to emit the sanitized report: {exc}")
    if args.report_out:
        _write_json(args.report_out, report.to_dict())
    print(f"  player-level output -> {out / 'marginal_convergence.json'} "
          f"(IGNORED)")
    return 0


def cmd_decompose(args) -> int:
    """Split a tactical CE effect into possession, payment and denial."""
    from pathlib import Path as _P
    from .decompose_experiment import run
    from .realpilot import PilotInputs

    inputs = PilotInputs(
        contract=_P(args.contract), sleeper_csv=_P(args.sleeper_csv),
        out_dir=_P(args.out_dir), pool_limit=args.pool_limit,
        market_scenario=args.market_scenario,
        performance_scenario=args.performance_scenario)
    missing = inputs.missing()
    if missing:
        raise UsageError(
            "real inputs are missing: " + ", ".join(missing) + ". See "
            "`ce-lab tactical real-pilot` for setup; both live under "
            "local_data/, which is gitignored.")
    if "local_data" not in _P(args.out_dir).parts:
        raise UsageError(
            f"--out-dir must sit under local_data/ (got {args.out_dir!r})")
    if args.draws < 2:
        raise UsageError(
            "--draws must be at least 2; one allocation draw has no "
            "between-allocation interval")
    if args.sims < 100:
        raise UsageError("--sims must be at least 100")

    print("TACTICAL CE DECOMPOSITION")
    print()
    print("  ANALYTICAL BRANCH WARNING: W (withdrawn), UF (we get him free)")
    print("  and RF (the rival gets him free) are decomposition instruments,")
    print("  NOT possible auction outcomes. They isolate one change at a time.")
    print("  They carry no recipient probability and must never be weighted")
    print("  into a summary of what might happen.")
    print()
    print("  Declared telescoping path: UP -> UF -> W -> RF -> RP. It sums")
    print("  exactly because it telescopes, not because completion is")
    print("  additive; a different ordering moves the labels, not the total.")
    print()
    blob, local = run(inputs, k=args.draws, holdout_sims=args.sims,
                      selection_sims=args.selection_sims,
                      positions=tuple(args.position
                                      or ("RB", "TE", "QB", "WR")),
                      runtime_budget_s=args.runtime_budget)
    d = _P(args.out_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / "decomposition.json").write_text(
        json.dumps(local, indent=2, default=str), encoding="utf-8")
    if args.json_out:
        _write_json(args.json_out, blob)
    print()
    print(f"  {'pos':<4}{'improve':>9}{'our pay':>10}{'our poss':>10}"
          f"{'denial':>10}{'rival pay':>11}{'total':>10}{'resid':>10}  class")
    for pos, v in blob["positions"].items():
        if "components" not in v:
            print(f"  {pos:<4}  {v.get('status', 'n/a')}")
            continue
        c = v["components"]
        print(f"  {pos:<4}{v['lineup_improvement']:>9.2f}"
              f"{c['our_payment']['mean']:>+10.5f}"
              f"{c['our_possession']['mean']:>+10.5f}"
              f"{c['rival_denial']['mean']:>+10.5f}"
              f"{c['rival_payment']['mean']:>+11.5f}"
              f"{v['total']['mean']:>+10.5f}"
              f"{v['ensemble_residual']:>10.1e}  {v['classification']}")
    print()
    print(f"  player-level output -> {d / 'decomposition.json'} (IGNORED)")
    print(f"  runtime {blob['runtime_s']:.0f}s")
    return 0


def cmd_midauction_frontier(args) -> int:
    """SIMULATED mid-auction reservation frontier for one QB and one RB."""
    from pathlib import Path as _P
    from .frontier_experiment import run
    from .midauction import SIMULATED_WATERMARK, PassPriceMode, STATE_SPECS
    from .realpilot import PilotInputs

    inputs = PilotInputs(
        contract=_P(args.contract), sleeper_csv=_P(args.sleeper_csv),
        out_dir=_P(args.out_dir), pool_limit=args.pool_limit,
        market_scenario=args.market_scenario,
        performance_scenario=args.performance_scenario)
    missing = inputs.missing()
    if missing:
        raise UsageError(
            "real inputs are missing: " + ", ".join(missing) + ". See "
            "`ce-lab tactical real-pilot` for setup; both live under "
            "local_data/, which is gitignored.")
    if "local_data" not in _P(args.out_dir).parts:
        raise UsageError(
            f"--out-dir must sit under local_data/ (got {args.out_dir!r})")
    if args.state not in STATE_SPECS:
        raise UsageError(
            f"unknown state {args.state!r}; known: "
            f"{', '.join(sorted(STATE_SPECS))}")
    if args.draws < 2:
        raise UsageError("--draws must be at least 2")
    if args.sims < 100:
        raise UsageError("--sims must be at least 100")
    if args.increment < 1:
        raise UsageError("--increment must be at least 1")
    if args.audit_budget < 2:
        raise UsageError(
            "--audit-budget must be at least 2; one audited price cannot "
            "bracket a frontier")
    if args.pass_mode == "stop_now" and args.standing_price is None:
        raise UsageError(
            "--pass-mode stop_now needs --standing-price: the rule is that "
            "the rival already leads at q and we consider q+increment")
    if args.pass_mode == "fixed_market" and args.rival_price is None:
        raise UsageError(
            "--pass-mode fixed_market needs --rival-price: an explicit q is "
            "what makes it a stated counterfactual")
    if args.pass_mode == "rival_outbids" and (args.standing_price is not None
                                              or args.rival_price is not None):
        raise UsageError(
            "--pass-mode rival_outbids derives q as p+increment; supplying a "
            "standing or rival price would mix pass-price semantics")

    print("=" * 78)
    print(SIMULATED_WATERMARK)
    print("=" * 78)
    print()
    blob, local = run(inputs, frontier_state=args.state, k=args.draws,
                      holdout_sims=args.sims,
                      selection_sims=args.selection_sims,
                      increment=args.increment,
                      audit_budget=args.audit_budget,
                      runtime_budget_s=args.runtime_budget)
    d = _P(args.out_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / "midauction_frontier.json").write_text(
        json.dumps(local, indent=2, default=str), encoding="utf-8")
    if args.json_out:
        _write_json(args.json_out, blob)
    print()
    for pos, fr in blob["frontiers"].items():
        if "classification" not in fr:
            print(f"  {pos}: {fr.get('status')}")
            continue
        print(f"  {pos}: {fr['classification']}  highest favorable "
              f"{fr['highest_favorable']}  next unfavorable "
              f"{fr['next_unfavorable']}  bracket {fr['bracket']}  "
              f"legal max ${fr['legal_max']}  cache hits {fr['cache_hits']}")
    print()
    print(f"  player-level output -> {d / 'midauction_frontier.json'} "
          f"(IGNORED)")
    print(f"  runtime {blob['runtime_s']:.0f}s")
    print()
    print("  REMINDER: the auction history above is SIMULATED. No league sale "
          "has ever been recorded.")
    return 0


_COMMANDS = {
    "validate": cmd_validate,
    "midauction-frontier": cmd_midauction_frontier,
    "decompose": cmd_decompose,
    "marginal-diagnostics": cmd_marginal_diagnostics,
    "allocation-ensemble": cmd_allocation_ensemble,
    "real-pilot": cmd_real_pilot,
    "signal-power": cmd_signal_power,
    "context": cmd_context,
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
    s = inner.add_parser(
        "context",
        help="controlled one-factor roster-context experiment (slow)")
    s.add_argument("--sims", type=int, default=None,
                   help="holdout seasons per arm (default 4000)")
    s.add_argument("--selection-sims", type=int, default=None,
                   help="independent selection-sample seasons")
    s.add_argument("--json-out", default=None)

    s = inner.add_parser(
        "signal-power",
        help="estimator power by candidate tier + sampling plan (slow)")
    s.add_argument("--sims", type=int, default=4000,
                   help="confirmatory holdout seasons")
    s.add_argument("--pilot-sims", type=int, default=1000,
                   help="independent pilot seasons used only for variance")
    s.add_argument("--target-half-width", type=float, default=0.005,
                   help="target 95%% half-width in CE; a reporting choice, "
                        "not a claim of economic materiality")
    s.add_argument("--cap-sims", type=int, default=40000,
                   help="beyond this the plan reports UNDERPOWERED_AT_CAP")
    s.add_argument("--alloc-seed", type=int, action="append",
                   help="shared-board allocation seed; repeatable")
    s.add_argument("--json-out", default=None)

    s = inner.add_parser(
        "real-pilot",
        help="targeted real-board tactical pilot (needs local real inputs)")
    s.add_argument("--contract",
                   default="local_data/real_player_contract_v1.json")
    s.add_argument("--sleeper-csv",
                   default="local_data/sleeper_2qb_values_2026_clean.csv")
    s.add_argument("--out-dir", default="local_data/tactical",
                   help="player-level output; must be under local_data/")
    s.add_argument("--report-out", default=None,
                   help="sanitized aggregate JSON (safe to commit)")
    s.add_argument("--pool-limit", type=int, default=260)
    s.add_argument("--per-position", type=int, default=3)
    s.add_argument("--position", action="append",
                   choices=["QB", "RB", "WR", "TE"],
                   help="restrict to these positions; repeatable")
    s.add_argument("--market-scenario", default="base",
                   choices=["low", "base", "high"])
    s.add_argument("--performance-scenario",
                   default="median_target/full_health/week_sd/exclude")
    s.add_argument("--sims", type=int, default=4000,
                   help="holdout seasons per audited arm")
    s.add_argument("--selection-sims", type=int, default=800)
    s.add_argument("--alloc-seed", type=int, action="append")
    s.add_argument("--base-price-only", action="store_true",
                   help="skip ladders and allocation-seed sweeps")
    s.add_argument("--runtime-budget", type=float, default=1800.0,
                   help="seconds; work beyond it is skipped and reported")

    s = inner.add_parser(
        "allocation-ensemble",
        help="exchangeable allocation ensemble + opening symmetry (slow)")
    s.add_argument("--contract",
                   default="local_data/real_player_contract_v1.json")
    s.add_argument("--sleeper-csv",
                   default="local_data/sleeper_2qb_values_2026_clean.csv")
    s.add_argument("--out-dir", default="local_data/tactical")
    s.add_argument("--json-out", default=None,
                   help="sanitized position-level aggregate (safe to commit)")
    s.add_argument("--pool-limit", type=int, default=260)
    s.add_argument("--draws", type=int, default=15,
                   help="allocation draws for the primary candidate")
    s.add_argument("--secondary-draws", type=int, default=6)
    s.add_argument("--tie-break", default="mechanical",
                   choices=["mechanical", "shock"])
    s.add_argument("--shock", type=float, default=0.0,
                   help="preference-shock sigma; 0 disables it")
    s.add_argument("--shock-scenario", default="none")
    s.add_argument("--sims", type=int, default=4000,
                   help="holdout seasons per arm per draw")
    s.add_argument("--selection-sims", type=int, default=800)
    s.add_argument("--market-scenario", default="base",
                   choices=["low", "base", "high"])
    s.add_argument("--performance-scenario",
                   default="median_target/full_health/week_sd/exclude")
    s.add_argument("--runtime-budget", type=float, default=3000.0)

    s = inner.add_parser(
        "marginal-diagnostics",
        help="quota-free with/without completion diagnostics (real inputs)")
    s.add_argument("--contract",
                   default="local_data/real_player_contract_v1.json")
    s.add_argument("--sleeper-csv",
                   default="local_data/sleeper_2qb_values_2026_clean.csv")
    s.add_argument("--out-dir", default="local_data/tactical")
    s.add_argument("--report-out", default=None,
                   help="sanitized aggregate JSON (safe to commit)")
    s.add_argument("--pool-limit", type=int, default=260)
    s.add_argument("--per-position", type=int, default=3)
    s.add_argument("--position", action="append",
                   choices=["QB", "RB", "WR", "TE"])
    s.add_argument("--effort", action="append",
                   help="ladder level as BEAM/POOL; repeatable, must increase")
    s.add_argument("--max-effort", type=int, default=None,
                   help="drop ladder levels above this beam width")
    s.add_argument("--tolerance", type=float, default=0.25,
                   help="numerical stability threshold in weekly points; a "
                        "second report at 2x is always printed")
    s.add_argument("--market-scenario", default="base",
                   choices=["low", "base", "high"])
    s.add_argument("--performance-scenario",
                   default="median_target/full_health/week_sd/exclude")

    s = inner.add_parser(
        "decompose",
        help="split tactical CE into possession/payment/denial (real inputs)")
    s.add_argument("--contract",
                   default="local_data/real_player_contract_v1.json")
    s.add_argument("--sleeper-csv",
                   default="local_data/sleeper_2qb_values_2026_clean.csv")
    s.add_argument("--out-dir", default="local_data/tactical")
    s.add_argument("--json-out", default=None,
                   help="sanitized position-level aggregate (safe to commit)")
    s.add_argument("--pool-limit", type=int, default=260)
    s.add_argument("--position", action="append",
                   choices=["QB", "RB", "WR", "TE"])
    s.add_argument("--draws", type=int, default=11)
    s.add_argument("--sims", type=int, default=4000)
    s.add_argument("--selection-sims", type=int, default=800)
    s.add_argument("--market-scenario", default="base",
                   choices=["low", "base", "high"])
    s.add_argument("--performance-scenario",
                   default="median_target/full_health/week_sd/exclude")
    s.add_argument("--runtime-budget", type=float, default=3600.0)

    s = inner.add_parser(
        "midauction-frontier",
        help="SIMULATED mid-auction reservation frontier (real board inputs)")
    s.add_argument("--contract",
                   default="local_data/real_player_contract_v1.json")
    s.add_argument("--sleeper-csv",
                   default="local_data/sleeper_2qb_values_2026_clean.csv")
    s.add_argument("--out-dir", default="local_data/tactical")
    s.add_argument("--json-out", default=None,
                   help="sanitized position-level aggregate (safe to commit)")
    s.add_argument("--pool-limit", type=int, default=260)
    s.add_argument("--state", default="balanced",
                   help="simulated state scenario: balanced, qb_inflation, "
                        "skill_inflation")
    s.add_argument("--pass-mode", default="stop_now",
                   choices=["stop_now", "rival_outbids", "fixed_market"],
                   help="pass-price semantics; a frontier is conditional on it")
    s.add_argument("--standing-price", type=int, default=None,
                   help="stop_now only: the rival's current high bid")
    s.add_argument("--rival-price", type=int, default=None,
                   help="fixed_market only: the explicit rival price q")
    s.add_argument("--increment", type=int, default=1)
    s.add_argument("--draws", type=int, default=11)
    s.add_argument("--sims", type=int, default=4000)
    s.add_argument("--selection-sims", type=int, default=800)
    s.add_argument("--audit-budget", type=int, default=7,
                   help="how many ladder prices earn K=11 CE audits")
    s.add_argument("--refine", action="store_true",
                   help="walk every integer inside the bracket")
    s.add_argument("--market-scenario", default="base",
                   choices=["low", "base", "high"])
    s.add_argument("--performance-scenario",
                   default="median_target/full_health/week_sd/exclude")
    s.add_argument("--runtime-budget", type=float, default=4200.0)

    s = inner.add_parser("benchmark", help="runtime for every live stage")
    common(s, scenarios=True, mode=True)
    s.add_argument("--audited", action="store_true",
                   help="also time one audited CE comparison")

    p.set_defaults(func=dispatch)
