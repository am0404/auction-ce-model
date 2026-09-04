"""``ce-lab`` -- command-line harness for the CE engine.

    ce-lab league        simulate the synthetic league, print CE for all 12 teams
    ce-lab lineup        show one team-week's lineup with a reason for every slot
    ce-lab experiments   list the controlled CE experiments
    ce-lab run <key>     run one experiment (or --all)
    ce-lab bench         runtime benchmarks and Monte Carlo uncertainty

Everything it touches is SYNTHETIC data (see ``synthetic.py``).
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import List, Optional, Sequence

import numpy as np

from .benchmark import benchmark, format_table, profile_stages
from .ce import team_report
from .experiments import EXPERIMENTS, run_all, run_experiment
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

    s = sub.add_parser("bench", help="runtime benchmarks")
    s.add_argument("--counts", type=int, nargs="+", default=None)
    s.set_defaults(func=cmd_bench)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
