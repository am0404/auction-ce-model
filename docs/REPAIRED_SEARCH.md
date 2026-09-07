# Repairing the completion search: local improvement, not a wider beam

> **SIMULATED MID-AUCTION STATE** (`44209a94d76fce67`). Invented from the market
> prior for frontier testing. NOT observed, NOT recorded, NOT historical, NOT
> calibrated. **No league sale has ever been recorded. No maximum bid is quoted
> here.**

## Verdict

**`QB_UNION_UNDERCONVERGED`** — but the instability collapsed from **four
failing evaluated prices to one**. The CE frontier was **not run**, because the
gate failed and the gate is unchanged.

## 1. The exact objective, and whether a certified optimizer is tractable

`ProxyEvaluator.strength` computes

```
f(R) = mean over (rep r, scoring week w) of
       sum of proj[r,w,p] for p in Greedy(R, r, w)
```

where `Greedy` is `lineup_vec.select_lineups_mask`, an **exact matroid greedy**
over the league's eligibility graph restricted to players available that week.

Per scenario this is the max-weight basis of a matroid, so each scenario's value
is **submodular** in the roster set, and an average of submodular functions is
submodular. Classifying it against the brief's taxonomy:

| form | verdict |
|---|---|
| additive by player | **no** — a player's marginal value depends on who else is rostered |
| best-legal-lineup + additive bench terms | **no** — bench value exists only through conditional substitution when a starter is out |
| **nonlinear via availability/conditional effects** | **yes — this is the box** |
| dependent on other completed rosters | **no** — `f` reads only `R` |

**This explains the beam's behaviour rather than merely describing it.** A beam
ranks partial rosters by a prefix value, but under a submodular objective a
prefix's value is not a valid bound on the finished roster's value — diminishing
returns mean a strong prefix can finish weakly. So carrying more prefixes does
not monotonically improve the result, which is exactly what the sweep measured
(94.17 at beam 64, 97.99 at beam 96, 95.15 at beam 384).

### Certified optimization: not available

| facility | status |
|---|---|
| scipy (`optimize.milp`, HiGHS) | **absent** |
| pulp / mip / ortools / cvxpy / pyomo | **absent** |
| networkx | absent |
| numpy | installed — the project's *only* runtime dependency |

`pyproject.toml` declares `dependencies = ["numpy>=1.24"]`. No MILP,
branch-and-bound or DP facility is installed, and adding one is a large new
dependency this task forbids.

Even with a solver, an exact encoding is awkward rather than routine: the
objective's per-scenario matroid selection needs assignment variables for every
(rep × week) scenario — 16 × 14 = 224 of them — coupled to the roster
indicators. That is tractable on a *reduced* pool and not on a real board.

The one exact facility that does exist is the repo's own
`enumerate_completions_exactly`, which **correctly refuses** the real board at
1.5×10¹¹ combinations. It is used below for certification on reduced pools,
which is the honest scope.

**The real-board result is therefore NOT certified globally optimal, and is not
claimed to be.** It is a one-swap local optimum over an accumulated union.

## 2. The control is untouched

Control ladder fingerprint remains **`0fddb000228e1276`**, asserted by a test in
both `test_qb_convergence.py` and `test_local_repair.py`. Same rungs, same
stopping rule, same 0.25 tolerance. The repair settings are **separately**
fingerprinted `146040efb7b0b8c7` so before and after compare honestly.

Declared before results: swap shortlist depth **120** by projection order,
**12** completions repaired per rung at the full budget, **6** per evaluated
price, effort multiplier **linear 1× (cheapest) → 3× (dearest)**.

## 3. Generating where the budget binds

The old search generated only at support prices and then filtered by
affordability, so expensive prices inherited whatever happened to survive. Each
evaluated price now gets its own search under its own real post-purchase budget.

Affordable accumulated constructions, control → repaired (final rung):

| evaluated bid | remaining budget | control | repaired |
|---:|---:|---:|---:|
| $36 | $103 | 1244 | **2311** |
| $40 | $99 | 1216 | **2267** |
| $65 | $74 | 968 | **1892** |
| $80 | $59 | 810 | **1654** |
| $100 | $39 | 547 | **1144** |
| $128 | $11 | **101** | **290** |

The thinnest point — `$128` — nearly tripled.

## 4. Local improvement

Exhaustive legal one-player swaps: remove one rostered non-candidate, add one
available player, preserving roster size, candidate ownership, budget, the
`$1`-per-open-slot reserve, lineup feasibility and distinct ownership. Repeat
until a one-swap local optimum. The **best** improving swap is taken, not the
first found, and ties break on a stable roster fingerprint — never on owner
tuple position or iteration order.

**Repaired completions are added to the union, never substituted**, so
monotonicity holds: nothing an earlier search found is ever removed.

| rung | repaired | already optimal | max gain | union before → after |
|---|---:|---:|---:|---|
| `1x` | 48 | — | +2.8367 | 210 → 258 |
| `2x` | 36 | — | +3.0500 | 737 → 773 |
| `4x` | 33 | — | +2.1870 | 1630 → 1663 |
| `8x` | 25 | — | +1.1332 | 2793 → 2818 |

Local-improvement runtime 297.3s; evaluated-price generation 105.6s.

### A design error worth recording

The **first** treatment run repaired only at the full budget, and its `UP`
numbers came back byte-identical to the control. The cause is structural: local
search maximises subject to the budget it is given, so repairing at `$139`
produces rosters costing near `$139`, none of which survives once `$36` or more
has gone to the candidate. The repair could not reach the side that was failing.
Repairing under **each evaluated price's own remaining budget** is what fixed
it, and that is the difference between the two repair fingerprints.

## 5. Verifying the local search itself

`tests/test_local_repair.py` (27 tests) covers every item the brief lists:
monotone improvement, legality preservation, one-swap local optimality,
repairing a planted beam failure, order-independent tie-breaking (roster order
*and* pool order), determinism, candidate ownership, unaffordable and
lineup-infeasible refusals, deduplication, and union monotonicity.

**Certification against the exact optimum:** on a reduced fixture small enough
for `enumerate_completions_exactly`, local repair started from the *weakest*
legal completion and **reached the exact optimum**. That is a statement about
that fixture, not about the real board.

## 6. The unchanged gate: before and after

| clause | control | repaired |
|---|---|---|
| `UF` selection stable | pass | pass |
| `UF` objective within 0.25 | pass | pass |
| no evaluated price improved > 0.25 | **FAIL — $36, $40, $65, $100** | **FAIL — $80 only** |
| `UP` selection stable | **FAIL — all six prices** | **FAIL — $80 only** |
| affordability unchanged | pass | pass |
| generation paths agree | **FAIL — 0.5276** | **FAIL — 0.5276** |

`UP` best proxy, final rung:

| price | control | repaired | gain | 4x→8x movement (repaired) |
|---:|---:|---:|---:|---:|
| $36 | 97.6818 | **98.0019** | +0.320 | 0.000 |
| $40 | 97.5286 | **97.8710** | +0.342 | 0.000 |
| $65 | 94.6873 | **95.9615** | +1.274 | 0.000 |
| $80 | 91.6912 | **92.4770** | +0.786 | **+0.603** |
| $100 | 87.1937 | **88.3153** | +1.122 | 0.000 |
| $128 | 74.3468 | **74.4857** | +0.139 | 0.000 |

Five of six prices are now **exactly** stable between the final two rungs. Only
`$80` still moves. `UF` also improved, 98.1399 → **98.2648**, and was stable from
the first rung onward.

The path-disagreement clause is unchanged at 0.5276 because it is measured on
**raw generation before repair**; repair acts on the accumulated union rather
than per path, so that clause is structurally untouched by this treatment. It is
reported as a failure honestly, but it is not evidence that the repair did
nothing — the substantive remaining instability is `$80`.

**Verdict: `QB_UNION_UNDERCONVERGED`. The CE frontier was not run.**

## 7. Performance

| stage | time |
|---|---:|
| board load | 20.4s |
| candidate selection (cache hit) | 0.1s |
| candidate selection (cold) | ~25 min |
| evaluated-price generation | 105.6s |
| local improvement | 297.3s |
| convergence ladder (total) | 416s |
| CE frontier | **not run** |
| total per candidate | 436s |

Projected for 25 targeted players with warm selection: `25 × 436s` ≈ **3.0
hours** of search, plus ~805s per candidate of CE if a frontier is ever run
(≈ 5.6 hours more).

**Local improvement adds ~300s per candidate and is worth it.** It replaces the
alternative of repeatedly re-running ~25-minute candidate searches at different
beam widths hoping one lands well — which the sweep showed is not even a
convergent strategy, since the objective is non-monotone in beam width. Roughly
five minutes of deterministic, monotone improvement buys what an unbounded
amount of beam widening cannot.

## 8. Remaining blocker and the specific next algorithm

One price (`$80`) and the raw-generation path spread.

**Recommended next: a bounded two-swap (2-out/2-in) exchange pass**, for a
specific reason rather than as the next thing on a list. The objective is
submodular under a knapsack constraint, and one-swap local optima for that class
are known to be escapable by exchanges that a single swap cannot reach: at
`$80` the budget is tight enough ($59 remaining) that improving usually requires
*simultaneously* freeing money and spending it, which is precisely a two-for-two
move. A one-swap neighbourhood cannot express it.

Bound it before running: shortlist the top 40 by projection among affordable,
apply only to the top 6 completions per evaluated price, and declare that
fingerprint before seeing results. Estimated cost ~40² / 120² of the one-swap
neighbourhood per step — roughly 10× the current repair time, so ~50 min per
candidate, which is why it did not fit inside this task's cap.

Not recommended: MILP or branch-and-bound. Both require a new dependency, and
the per-scenario matroid encoding would be large without obviously closing a
0.6-point gap that a two-swap pass plausibly closes directly.

## 9. Artifacts

- `docs/REPAIRED_SEARCH_sanitized.json` — ladder, rungs, per-price generation,
  repair stats, timings. No player names, ids, or projections.
- `local_data/tactical/repaired_search.json` — player-level. **Gitignored.**
- `ce-lab tactical repaired-search --reuse-candidates <local_data json>`
- `src/ceauction/tactical/localrepair.py`, `repairedsearch.py`,
  `repaired_experiment.py`, `tests/test_local_repair.py` (27 tests)
