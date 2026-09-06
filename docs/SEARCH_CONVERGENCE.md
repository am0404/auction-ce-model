# Search-effort convergence, and the K=11 rerun it earned

`ce-lab tactical marginal-diagnostics` and `ce-lab tactical
allocation-ensemble`. Real board, sanitized aggregates. Player-level detail
is in `local_data/tactical/`, gitignored.

## The previous claim was wrong

The last branch called the diagnostic converged because it stopped violating
*price* monotonicity. Its own search-budget table showed otherwise:

```
beam/pool    improve@$18    improve@$1
   160/90         +1.21         +1.16
  320/140         -1.08         +0.86
```

A 2.29-point swing and a sign reversal from doubling the effort. Those two
numbers decide whether a real player receives four thousand seasons of
championship-equity simulation. Price monotonicity was necessary and
nowhere near sufficient.

## Two monotonicities, only one of which was being checked

**Price monotonicity** — acquiring the same player for less cannot be worse.

**Search-effort monotonicity** — more compute cannot produce a worse answer.
This was never enforced, and it does not hold for independent beams: a wider
beam is a *different* heuristic, not a strictly better one, so it routinely
misses constructions a narrower one found.

The fix makes it structural. Every level's completions are accumulated into
one union, re-scored with a single `ProxyEvaluator` so objectives from
different levels are comparable at all, and the best legal member is taken.
Monotonicity then holds by construction; the report still checks it, because
a guard that only fires on a bug you have already fixed is the one to keep.

### A second bug the union exposed

The $1 and $p unions were built by independent searches, so cross-price
nesting held only *within* each price's own search. The final union still
violated price monotonicity by **5.94 weekly points** — the $1 branch was
missing constructions the $p branch had found. Every $p member is now folded
into the $1 union, which is sound because a completion's `added_cost` counts
only the players bought besides the candidate, and the candidate is owned in
both states. Two candidates that had been reported `DIAGNOSTIC_NOT_CONVERGED`
converge once that fold is in place.

## Convergence requires the decision to settle, not just the number

A candidate stable to 0.25 points whose *role* still flips is not converged,
because role is what the sampling decision reads. `status()` requires the
objective delta, the improvement delta, the role and the policy all to
settle, and any recorded violation forces `DIAGNOSTIC_NOT_CONVERGED`.

A non-converged candidate is reported **UNRESOLVED**, never silently filed as
proxy-only: an unconverged search may not withhold CE simulation by default.

## Search-effort ladder, all twelve candidates

Ladder: 64/60 → 160/90 → 240/120 → 320/140. Deltas are between consecutive
rungs; the objective is the union's best, so it can only rise.

| pos | tier | objective deltas | improvement deltas | final @price | final @$1 | role | policy | status | effort |
|:--|:--|:--|:--|--:|--:|:--|:--|:--|:--|
| QB | expensive | +1.29 +1.02 +0.02 | -0.25 +0.74 +0.02 | +1.46 | +6.37 | marginal starter | 4000-season audit | CONVERGED | 240/120 |
| QB | mid | +0.20 +0.46 +0.00 | -1.54 -0.23 -0.19 | -0.05 | +4.52 | replaceable starter | proxy only | CONVERGED | 240/120 |
| QB | cheap | +0.20 +1.63 +0.00 | -1.55 +0.94 -0.19 | -1.10 | +3.34 | aggregate depth | proxy only | CONVERGED | 240/120 |
| RB | expensive | +1.76 +0.68 +0.20 | -0.02 +0.68 +0.20 | +0.48 | +8.30 | marginal starter | 4000-season audit | CONVERGED | 240/120 |
| RB | mid | +0.01 +0.71 +0.00 | -1.73 +0.03 +0.00 | +2.50 | +5.34 | marginal starter | 4000-season audit | CONVERGED | 240/120 |
| RB | cheap | +2.54 +0.01 +0.00 | +1.72 -0.67 +0.00 | -1.10 | +1.21 | replaceable starter | proxy only | CONVERGED | 240/120 |
| WR | expensive | +0.00 +0.00 +0.00 | -0.78 -0.68 +0.00 | +1.76 | +6.60 | marginal starter | 4000-season audit | CONVERGED | 240/120 |
| WR | mid | +0.80 +0.65 +0.00 | -0.02 -0.03 +0.00 | -1.19 | +2.36 | replaceable starter | proxy only | CONVERGED | 240/120 |
| WR | cheap | +1.58 +2.12 +0.00 | +1.58 +0.62 +0.00 | +1.63 | +2.47 | marginal starter | 4000-season audit | CONVERGED | 240/120 |
| TE | expensive | +3.72 +0.28 +0.00 | +1.98 -0.40 +0.00 | +0.77 | +3.37 | marginal starter | 4000-season audit | CONVERGED | 240/120 |
| TE | mid | +1.21 +0.49 +0.00 | +1.21 -1.02 +0.00 | -0.23 | +2.58 | replaceable starter | proxy only | CONVERGED | 240/120 |
| TE | cheap | +1.45 +1.76 +0.01 | +1.45 +0.25 +0.01 | +1.32 | +1.54 | marginal starter | 4000-season audit | CONVERGED | 240/120 |

**Every candidate's final rung moves by at most 0.20 points, and ten of
twelve by at most 0.02**, while earlier rungs move by as much as +3.72. The
answer settles at **240/120**, and 320/140 confirms rather than changes it.

```
converged at tolerance 0.25    12/12
converged at tolerance 0.5     12/12
search-effort nesting violations   0   (structural: the union)
price nesting violations           0
audited / proxy-only / UNRESOLVED  7 / 5 / 0
```

Tolerances are **numerical stability thresholds in weekly points**, not
claims about what size of difference is economically material.

## How the classification moved

Three diagnostics have now been run on these twelve players. The counts:

| diagnostic | audited | proxy-only | unresolved |
|:--|--:|--:|--:|
| 1/4/6/3 template | 6 | 6 | - |
| quota-free, single beam 160/90 | 8 | 4 | - |
| **quota-free, converged ladder** | **7** | **5** | **0** |

The `8 / 4` count from the previous branch is **not** reproduced, and was
not preserved. Notable movements at the top of each position:

```
top QB   template  6.86  audit    |  beam 160/90  +0.05  proxy-only  |  converged  +1.46  AUDIT
top TE   template  0.22  proxy    |  beam 160/90  +1.89  audit       |  converged  +0.77  AUDIT
```

The top QB has now been classified three different ways by three diagnostics.
Only the third is converged, and it is the only one whose number is stable to
0.02 under a doubling of effort.

## Corrected K=11 ensemble

Candidates are the ones the **converged** diagnostic nominates — the highest-
improvement audited candidate at each position, not the most expensive and
not a stale beam-32 pick. These supersede the previous branch's ensemble.

| position | tier | $ | lineup improvement | K | mean delta | median | min/max | between-SD | within-SE | CI95 | sign+ | verdict |
|:--|:--|--:|--:|--:|--:|--:|:--|--:|--:|:--|--:|:--|
| RB (primary) | mid | 25 | +2.50 | 11 | -0.03054 | -0.02750 | -0.0655 / +0.0027 | 0.01975 | 0.00851 | [-0.04381, -0.01728] | 0.09 | unfavorable |
| QB | expensive | 26 | +1.46 | 11 | +0.15752 | +0.15850 | +0.1465 / +0.1673 | 0.00600 | 0.00845 | [+0.15349, +0.16156] | 1.00 | favorable |
| WR | expensive | 32 | +1.76 | 11 | +0.08064 | +0.09325 | +0.0385 / +0.1270 | 0.03199 | 0.00895 | [+0.05915, +0.10212] | 1.00 | favorable |
| TE | cheap | 6 | +1.32 | 11 | -0.06714 | -0.06875 | -0.0945 / -0.0238 | 0.01726 | 0.00776 | [-0.07873, -0.05554] | 0.00 | unfavorable |

Opening symmetry held:

```
Team02 vs Team03  single-seed gap -0.03400
                  ensemble mean   +0.00116
                  CI95            [-0.01529, +0.01761]  contains zero: True
                  persistent label effect: False
```

## Proxy direction versus CE direction

All four candidates have **positive** proxy lineup improvement. Two have
negative CE.

| position | proxy improvement | CE mean | agree? |
|:--|--:|--:|:--|
| QB | +1.46 | +0.15752 | yes |
| WR | +1.76 | +0.08064 | yes |
| RB | +2.50 | -0.03054 | **no** |
| TE | +1.32 | -0.06714 | **no** |

Disagreement is permitted and is not evidence of a bug. A positive lineup
improvement says the roster scores more weekly points with him at that price;
a negative CE says the *whole joint league* is worse for us. Candidate causes,
in order of what the data supports:

* **Opportunity cost through the shared board.** The proxy compares our two
  completions; CE additionally lets eleven rivals re-complete against what we
  left. Buying the RB at $25 changes what the rivals get.
* **Denial running the wrong way.** In the pass branch the named recipient
  pays $30 for the RB and $6 for the TE. Letting a rival overpay is worth
  something to us that the proxy cannot see at all.
* **Playoff/bye nonlinearity**, which converts weekly points into
  championship probability at a rate that varies with where we sit.

What this run does **not** do is separate those three. The decomposition
needs a denial-isolated branch that is not built, and saying which dominates
would be guessing.

## Verdict: PARTIAL GO for real price-frontier testing

* Diagnostics converged for all four position candidates. **Yes** — 12/12 at
  both tolerances, 240/120, confirmed at 320/140.
* No search-effort objective regression. **Yes** — 0, structurally.
* Cross-price nesting holds. **Yes** — 0 violations after the fold.
* Sampling recommendations stable. **Yes** — 0 unresolved.
* K=11 corresponds to the final selected candidates. **Yes.**
* Symmetry and conservation valid. **Yes.**

Why not a full GO: the proxy that *selects* candidates and the CE that
*prices* them disagree in direction for two of four positions, and the cause
is unseparated. A frontier search driven by a selector that disagrees with the
objective it feeds will spend its effort in the wrong places.
