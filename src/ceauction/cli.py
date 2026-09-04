"""``ce-lab`` -- command-line harness for the CE engine.

    ce-lab league        simulate the synthetic league, print CE for all 12 teams
    ce-lab lineup        show one team-week's lineup with a reason for every slot
    ce-lab experiments   list the controlled CE experiments
    ce-lab run <key>     run one experiment (or --all)
    ce-lab curve         marginal CE curve for one roster slot, + resolution report
    ce-lab ingest        validate real player sources; write the contract locally
    ce-lab bench         runtime benchmarks and Monte Carlo uncertainty

Everything it touches is SYNTHETIC data (see ``synthetic.py``).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from .benchmark import benchmark, format_table, profile_stages
from .ce import team_report
from .curve import (
    DEFAULT_RESOLUTION_TARGETS,
    LIVE_AUCTION_BUDGET_SECONDS,
    level_grid,
    sweep_marginal_curve,
    weakest_flex_slot,
)
from .experiments import EXPERIMENTS, run_all, run_experiment
from .realdata.contract import GAMES_BASIS, TARGET_LEAGUE_CONFIG_ID
from .realdata.pipeline import IngestionPaths, ingest, write_outputs
from .realdata.scoring import FUMBLE_INTERPRETATIONS
from .realdata.sources import SyntheticSourceRefused
from .lineup import select_lineup
from .simulate import DEFAULT_CHUNK, pregame_week, simulate_seasons
from .synthetic import make_synthetic_league
from .worlds import build_pool_arrays, generate_world

BANNER = (
    "ce-lab -- championship-equity laboratory\n"
    "SYNTHETIC DATA ONLY. No real player distributions are used anywhere.\n"
)


def _league(args) -> "RosterSet":  # noqa: F821
    return make_synthetic_league()


def cmd_league(args) -> int:
    rosters = _league(args)
    t0 = time.perf_counter()
    out = simulate_seasons(rosters, args.sims, args.seed, args.chunk)
    dt = time.perf_counter() - t0
    ce = out.championship_equity()

    print(BANNER)
    print(f"{args.sims:,} seasons, seed {args.seed}, {dt:.2f}s "
          f"({args.sims / dt:,.0f} seasons/s)\n")
    head = (f"{'team':<9} {'CE':>7} {'+/-95%':>8} {'playoff':>8} {'bye':>7} "
            f"{'pts/wk':>8} {'>med':>7} {'h2h':>7} {'wins':>7} {'seed':>6} {'slots':>6}")
    print(head)
    print("-" * len(head))
    order = np.argsort(-ce)
    for t in order:
        r = team_report(out, int(t), rosters.settings)
        print(f"{r.team_name:<9} {r.championship_equity:>7.4f} "
              f"{1.96 * r.ce_se:>8.4f} {r.playoff_probability:>8.4f} "
              f"{r.bye_probability:>7.4f} {r.mean_points_per_week:>8.2f} "
              f"{r.above_median_rate:>7.4f} {r.head_to_head_win_rate:>7.4f} "
              f"{r.mean_wins:>7.2f} {r.mean_seed:>6.2f} {r.mean_slots_filled:>6.3f}")
    print("-" * len(head))
    print(f"{'sum':<9} {ce.sum():>7.4f}")
    return 0


def cmd_lineup(args) -> int:
    rosters = _league(args)
    pool = build_pool_arrays(rosters.pool, rosters.settings)
    world = generate_world(pool, args.seed, args.sim, 1)
    print(BANNER)
    print(f"season index {args.sim}, seed {args.seed}, team {args.team} "
          f"({rosters.rosters[args.team].team_name})\n")
    idx = rosters.id_to_index
    for week in args.weeks:
        pg = pregame_week(world, rosters, args.team, week - 1, sim=0)
        lu = select_lineup(pg)
        print("\n".join(lu.explain()))
        realized = sum(
            float(world.realized.points[0, idx[c.player_id], week - 1])
            for c in lu.choices
        )
        benched_best = max(
            (float(world.realized.points[0, idx[b.player_id], week - 1]) for b in lu.bench),
            default=0.0,
        )
        print(f"  -> realized team score {realized:.2f}   "
              f"(best benched player actually scored {benched_best:.2f}, "
              f"which counted for nothing)")
        print()
    return 0


def cmd_experiments(args) -> int:
    print(BANNER)
    print("available experiments:\n")
    for key, spec in EXPERIMENTS.items():
        print(f"  {key:<20} {spec.title}")
        print(f"  {'':<20} {spec.question}\n")
    return 0


def cmd_run(args) -> int:
    keys: Optional[Sequence[str]] = None
    if args.all:
        keys = list(EXPERIMENTS)
    elif args.key:
        keys = args.key
    else:
        print("specify one or more experiment keys, or --all", file=sys.stderr)
        return 2
    for k in keys:
        if k not in EXPERIMENTS:
            print(f"unknown experiment {k!r}. try: ce-lab experiments", file=sys.stderr)
            return 2

    print(BANNER)
    print(f"{args.sims:,} seasons per arm, seed {args.seed}, "
          f"paired via common random numbers\n")
    t0 = time.perf_counter()
    outputs = run_all(args.sims, args.seed, keys=keys)
    dt = time.perf_counter() - t0
    for o in outputs:
        print(o.format())
    print("\n" + "=" * 78)
    print("SUMMARY -- paired delta CE for the focus team (Team01)")
    print("=" * 78)
    head = (f"{'experiment':<19} {'comparison':<44} {'dCE':>9} {'+/-95%':>8} "
            f"{'z':>6} {'dPts/wk':>9} {'z':>6}")
    print(head)
    print("-" * len(head))
    for o in outputs:
        for label, d, se, z, dp, dpse in o.summary_rows():
            short = label if len(label) <= 43 else label[:40] + "..."
            zp = dp / dpse if dpse else 0.0
            print(f"{o.key:<19} {short:<44} {d:>+9.5f} {1.96 * se:>8.5f} "
                  f"{z:>+6.2f} {dp:>+9.4f} {zp:>+6.1f}")
    print("-" * len(head))
    print(f"total runtime {dt:.1f}s")
    print("\nA non-significant dCE with a significant dPts/wk means the mechanism "
          "fired\nbut the effect is smaller than this sample size can resolve -- "
          "raise --sims.")
    print()
    print("=" * 78)
    print("HOW TO READ THIS TABLE")
    print("=" * 78)
    print(
        "INFRASTRUCTURE finding: 'the engine responds to this structural change in a\n"
        "  measurable, correctly signed, statistically resolvable way.' That is what\n"
        "  every row above establishes, and it is a claim about this code.\n"
        "\n"
        "SYNTHETIC-FANTASY finding: 'a change of this kind is worth this much CE.'\n"
        "  No row above establishes that. Every effect size is a property of the\n"
        "  invented parameters in synthetic.py, and would change with real ones.\n"
        "\n"
        "Two rows are CONTROLS and are meant to read near zero:\n"
        "  opponent-placement    exchangeable rivals -- a signal here is a bug\n"
        "  aggregate-lineup-spot 'unforecastable rotation' arm -- byte-identical\n"
        "                        realized production, so any gap is pure knowability\n"
    )
    print("REMINDER: synthetic inputs. These validate the machinery, "
          "not any football claim.")
    return 0


def cmd_curve(args) -> int:
    """Sweep one roster slot from replacement level to elite.

    The default target is the focus team's weakest FLEX-eligible player,
    because that is the roster spot a $1-$3 auction decision actually turns
    on -- and therefore the spot whose marginal CE has to be resolvable if
    pricing is going to work at the bottom of the roster at all.
    """
    rosters = _league(args)
    if args.player_id is not None:
        spec = rosters.spec(args.player_id)
        chosen = "requested explicitly"
    else:
        spec = weakest_flex_slot(rosters, args.team)
        chosen = "weakest FLEX-eligible player on the focus team"

    if args.step <= 0:
        print("--step must be positive", file=sys.stderr)
        return 2
    if args.max_level < args.min_level:
        print("--max-level must be >= --min-level", file=sys.stderr)
        return 2

    # `level_grid` never overshoots --max-level: it shortens the final step
    # instead, so `--min-level 4 --max-level 10 --step 4` is 4, 8, 10.
    levels = level_grid(args.min_level, args.max_level, args.step)
    baseline = args.baseline if args.baseline is not None else args.min_level

    steps = {round(b - a, 9) for a, b in zip(levels, levels[1:])}
    if len(steps) > 1:
        final = levels[-1] - levels[-2]
        step_note = (f"steps of {args.step:g} with a final step of {final:g} "
                     f"(the range is not a whole number of steps)")
    elif steps:
        step_note = f"steps of {steps.pop():g}"
    else:
        step_note = "a single level"

    print(BANNER)
    print(f"sweeping {spec.name} ({spec.position.label}, base_mean "
          f"{spec.base_mean:.2f}) -- {chosen}")
    print(f"{len(levels)} levels from {levels[0]:g} to {levels[-1]:g} in "
          f"{step_note}, {args.sims:,} seasons each\n")

    t0 = time.perf_counter()
    curve = sweep_marginal_curve(
        rosters,
        team=args.team,
        player_id=spec.player_id,
        baseline_level=baseline,
        levels=levels,
        n_sims=args.sims,
        seed=args.seed,
        chunk=args.chunk,
        isotonic=args.isotonic,
        live_auction_budget_seconds=args.live_budget,
        notes=("only the player's projected level varies -- base_mean, and "
               "any weekly_projection_override shifted by the same delta; "
               "position, variance, injury profile, weekly state, signal "
               "precision and crn_key are held fixed"),
    )
    dt = time.perf_counter() - t0
    print(curve.format())
    print()
    print(f"total sweep runtime {dt:.1f}s "
          f"({len(levels)} levels x {args.sims:,} seasons)")

    if args.csv:
        curve.to_csv(args.csv)
        print(f"wrote {args.csv}")

    print("\nREMINDER: synthetic inputs. This measures whether the ENGINE can "
          "resolve\nmarginal CE at a useful scale. It is not a price, and no "
          "dollar value,\nbid or roster-completion decision follows from it.")
    return 0


#: The contract carries real player rows. It may only be written somewhere
#: git-ignored: this repository is public and the sources are subscriber-gated
#: vendor exports that are not redistributable.
ALLOWED_CONTRACT_DIRS = ("local_data",)


def _contract_path_is_allowed(path: Path) -> bool:
    """True only if `path` sits under an ignored local data directory."""
    parts = {p.lower() for p in Path(path).parts}
    return bool(parts & {d.lower() for d in ALLOWED_CONTRACT_DIRS})


def cmd_ingest(args) -> int:
    """Validate real player sources and build the normalized contract.

    Prints only the sanitized report. Real rows go to the contract file, which
    must live under an ignored directory.
    """
    if not Path(args.projections).exists():
        print(f"projections file not found: {args.projections}", file=sys.stderr)
        return 2
    if args.contract_out and not _contract_path_is_allowed(args.contract_out):
        print(f"refusing to write the contract to {args.contract_out}: it "
              f"contains real player rows and must be written under one of "
              f"{ALLOWED_CONTRACT_DIRS} (git-ignored). This repository is "
              f"public.", file=sys.stderr)
        return 2

    paths = IngestionPaths(
        projections=Path(args.projections),
        fantasypros=Path(args.fantasypros) if args.fantasypros else None,
        injuries=Path(args.injuries) if args.injuries else None,
        fits=Path(args.fits) if args.fits else None,
        aliases=Path(args.aliases) if args.aliases else None,
    )

    print(BANNER)
    try:
        outcome = ingest(
            paths,
            games_basis=args.games_basis,
            fumble_interpretation=args.fumbles,
            league_config_id=args.league_config_id,
        )
    except SyntheticSourceRefused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2

    print(outcome.format_report())

    written = write_outputs(
        outcome,
        Path(args.contract_out) if args.contract_out else None,
        Path(args.report_out) if args.report_out else None,
    )
    for kind, where in written.items():
        print(f"\nwrote {kind}: {where}")
    if "contract" in written:
        print("  (contains real player rows -- ignored location, never committed)")

    print("\nREMINDER: this builds and validates the input contract only. No "
          "dollar\nvalue, opening bid, live bid or auction behaviour follows "
          "from it, and no\nPlayerSpec field without a proven source has been "
          "populated.")
    return 0 if outcome.ok else 1


def cmd_bench(args) -> int:
    print(BANNER)
    counts = args.counts or [250, 1000, 4000, 16000]
    rows = benchmark(counts, seed=args.seed, chunk=args.chunk)
    print(format_table(rows))
    print("\nper-stage profile at 2,000 seasons:")
    for name, secs, pct in profile_stages(2000, seed=args.seed, chunk=args.chunk):
        print(f"  {name:<14} {secs:>7.3f}s  {pct:>5.1f}%")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ce-lab",
        description="Championship-equity laboratory (synthetic data only).",
    )
    p.add_argument("--seed", type=int, default=20260904, help="master RNG seed")
    p.add_argument("--chunk", type=int, default=DEFAULT_CHUNK,
                   help="seasons per batch (memory/speed tradeoff; results are "
                        "identical for any value)")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("league", help="simulate the synthetic league and print CE")
    s.add_argument("--sims", type=int, default=5000)
    s.set_defaults(func=cmd_league)

    s = sub.add_parser("lineup", help="explain lineup choices for one team-week")
    s.add_argument("--team", type=int, default=0)
    s.add_argument("--sim", type=int, default=0, help="which simulated season")
    s.add_argument("--weeks", type=int, nargs="+", default=[1, 8, 14])
    s.set_defaults(func=cmd_lineup)

    s = sub.add_parser("experiments", help="list available experiments")
    s.set_defaults(func=cmd_experiments)

    s = sub.add_parser("run", help="run controlled CE experiments")
    s.add_argument("key", nargs="*", help="experiment key(s)")
    s.add_argument("--all", action="store_true")
    s.add_argument("--sims", type=int, default=4000)
    s.set_defaults(func=cmd_run)

    s = sub.add_parser("curve",
                       help="marginal CE curve for one roster slot + resolution report")
    s.add_argument("--team", type=int, default=0, help="focus team index")
    s.add_argument("--player-id", type=int, default=None,
                   help="roster slot to vary (default: the focus team's weakest "
                        "FLEX-eligible player)")
    s.add_argument("--sims", type=int, default=16000, help="seasons per level")
    s.add_argument("--min-level", type=float, default=4.0)
    s.add_argument("--max-level", type=float, default=22.0)
    s.add_argument("--step", type=float, default=1.0)
    s.add_argument("--baseline", type=float, default=None,
                   help="replacement-level anchor (default: --min-level)")
    s.add_argument("--csv", default=None, help="also write the curve to this path")
    s.add_argument("--isotonic", action="store_true",
                   help="add a display column that IMPOSES monotonicity; raw "
                        "estimates are kept and remain primary")
    s.add_argument("--live-budget", type=float, default=LIVE_AUCTION_BUDGET_SECONDS,
                   help="seconds available per decision in a live auction")
    s.set_defaults(func=cmd_curve)

    s = sub.add_parser(
        "ingest", help="validate real player sources and build the contract")
    s.add_argument("--projections", required=True,
                   help="component stat projection CSV (required)")
    s.add_argument("--fantasypros", default=None,
                   help="expert consensus CSV: bye weeks, optional labels")
    s.add_argument("--injuries", default=None, help="injury profile JSON")
    s.add_argument("--fits", default=None,
                   help="fitted positional dispersion/availability JSON")
    s.add_argument("--aliases", default="data/player_aliases.json",
                   help="reviewed name equivalences and per-player overrides")
    s.add_argument("--contract-out", default=None,
                   help="where to write the contract. Contains real player "
                        "rows, so it must sit under local_data/")
    s.add_argument("--report-out", default=None,
                   help="where to write the sanitized JSON report (safe to commit)")
    s.add_argument("--games-basis", type=float, default=GAMES_BASIS,
                   help="games a season total is spread over under "
                        "interpretation A")
    s.add_argument("--fumbles", choices=list(FUMBLE_INTERPRETATIONS),
                   default="exclude",
                   help="how to read the Fumbles column. Default 'exclude': "
                        "the league scores fumbles LOST and the column's "
                        "meaning is unresolved")
    s.add_argument("--league-config-id", default=TARGET_LEAGUE_CONFIG_ID,
                   help="names the target league; the 12-team superflex league")
    s.set_defaults(func=cmd_ingest)

    s = sub.add_parser("bench", help="runtime benchmarks")
    s.add_argument("--counts", type=int, nargs="+", default=None)
    s.set_defaults(func=cmd_bench)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
