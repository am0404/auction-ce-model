# The full-budget completion union

> **SIMULATED MID-AUCTION STATE.** Every auction history in this document is
> invented from the market prior for frontier testing. It is NOT observed, NOT
> recorded, NOT historical, and NOT calibrated. **No league sale has ever been
> recorded.** No maximum bid is quoted here.

## The question

`RECONCILIATION.md` closed with both reservation brackets withdrawn and a
possession effect that had gone **negative** for both candidates — RB
`-0.09480`, QB `-0.03277` — where the earlier decomposition had it positive.
Owning a player you did not pay for cannot plausibly hurt you, so the number was
a symptom, not a finding.

The named suspect: the with-candidate opportunity set was generated only at the
**tested bid prices** (`$65`, `$80`, `$100` for the RB). `UF` — "us, holding the
candidate, having paid nothing" — pays no acquisition price at all, so it should
see every construction its full budget allows. Instead it could only choose
among constructions discovered under a budget already cut by a price it never
pays. This run rebuilds that set from **$1 upward** and re-reads the frontier.

## Three prices, three names

Merging these is how a search artifact becomes a quoted bid, so the code keeps
them structurally apart (`union.PriceRole`, `union.assert_live_biddable`):

| name | meaning | biddable? |
|---|---|---|
| `search_support_price` | a price used only to **discover** roster constructions | **never** |
| `evaluated_bid_price` | a purchase actually evaluated | only at `q + increment` and above |
| `standing_pass_price` | the named rival's current high bid `q` | it is the rival's, not ours |

With the RB's standing pass price of `$30`, the support prices `$1`, `$5`,
`$10`, `$20` are **not legal live bids**. They generate rosters and nothing
else. `assert_live_biddable` raises rather than let one be reported as a price
we could pay, and every rung in the sanitized JSON carries
`"role": "search_support_price"`.

## The union, rung by rung

Support prices `$1, $5, $10, $20, $40, $65, $80, $100, legal max ($128)`.
Completions are deduplicated by a fingerprint over the **complete** roster *and*
its cost, so two ways of filling fifteen slots at different prices stay
distinct. Both candidates behaved identically in structure:

| support price | generated | newly unique | cumulative | cost range (QB) | cost range (RB) |
|---:|---:|---:|---:|---|---|
| $1 | 3 | 3 | 3 | 57–122 | 89–122 |
| $5 | 3 | 3 | 6 | 57–122 | 89–122 |
| $10 | 3 | 3 | 9 | 57–122 | 89–127 |
| $20 | 3 | 2 | 11 | 57–122 | 89–127 |
| $40 | 3 | 3 | 14 | 57–122 | 53–127 |
| $65 | 3 | 3 | 17 | 49–122 | 45–127 |
| $80 | 3 | 3 | 20 | 42–122 | 42–127 |
| $100 | 3 | 3 | 23 | 21–122 | 21–127 |
| $128 | 3 | 3 | 26 | 11–122 | 11–127 |

Search-effort nesting and cross-price nesting held at **every** rung for both
candidates. Both are structural under a single accumulated union — we only ever
add, and the branches then read it through one cost threshold — so these
columns verify the implementation rather than discover a fact about the search.
That is the point: the restricted build had to *hope* for nesting, and did not
get it.

## Union adequacy: the restricted build was starving `UF`

| | QB | RB |
|---|---:|---:|
| restricted union size (`$65/$80/$100` only) | 9 | 9 |
| full union size (`$1` upward) | 26 | 26 |
| overlap | 9 | 9 |
| **added by low support prices** | **17** | **17** |
| `UF`'s chosen completion changed | yes | yes |
| `UF` proxy strength, restricted → full | 93.141 → 97.303 | 93.248 → 98.126 |
| **proxy gain** | **+4.16** | **+4.88** |

The restricted union was a strict subset containing a third of the
constructions, and the ones it was missing were better: `UF`'s roster gained
roughly five points of weekly starting projection once it was allowed to see
what its actual budget could buy.

## The effect on possession

| | restricted union (`RECONCILIATION.md`) | full union (this run) |
|---|---:|---:|
| QB possession | `-0.03277` | **`+0.06434`** |
| RB possession | `-0.09480` | **`-0.01027`** |

**The QB's sign flips.** Its negative possession effect was entirely an artifact
of the restricted union.

**The RB's does not.** It moves 89% of the way to zero and stays slightly
negative. `-0.01027` is smaller than the between-allocation SD of the total
delta at that price (`0.00747`) is large, so it should not be read as an
established negative — but it is not explained away by the union either. The
brief's hypothesis is **confirmed for the QB and only partly confirmed for the
RB**; something else is contributing to the RB, and this run does not identify
it.

Possession is now **price-invariant** for both candidates — one value across
every evaluated bid price — which is the structural property the restricted
build violated. `possession_offer_is_price_independent()` is asserted at every
price and every draw before any CE is computed.

## Reconciliation survived

Exact, at the unchanged tolerance. Max per-draw residual **1.39e-17** against
`AGREEMENT_TOLERANCE = 1e-12`, across all 12 price/position combinations and all
11 draws; `all_draws_agree` true everywhere. All five branches read one
`EvalContext`; no branch may launch a second beam
(`IndependentSearchRefused`).

## Frontier: still no quotable maximum bid

**QB** (recipient `Team07`, standing pass price `$35`, legal max `$128`):

| evaluated bid | pass rule | delta | 95% t-interval | verdict |
|---:|---|---:|---|---|
| $36 | `STOP_NOW` | `-0.08357` | `[-0.08670, -0.08044]` | unfavorable |
| $40 | `FIXED_MARKET` | `-0.08357` | `[-0.08670, -0.08044]` | unfavorable |
| $65 | `FIXED_MARKET` | `-0.08357` | `[-0.08670, -0.08044]` | unfavorable |
| $80 | `FIXED_MARKET` | `-0.08357` | `[-0.08670, -0.08044]` | unfavorable |
| $100 | `FIXED_MARKET` | `-0.10405` | `[-0.10800, -0.10009]` | unfavorable |
| $128 | `FIXED_MARKET` | `-0.10405` | `[-0.10800, -0.10009]` | unfavorable |

The ladder reads `PASS_AT_NEXT_BID`. **That conclusion is withheld.** The QB's
union is `UNION_NOT_CONVERGED`: tripling the search effort grew the union from
26 to 72 constructions, changed `UF`'s chosen completion, and moved its proxy by
`+0.690` — above the `0.25` tolerance. A reservation conclusion resting on an
opportunity set that more effort still moves is not a reservation conclusion.

**RB** (recipient `Team11`, standing pass price `$30`, legal max `$128`,
`UNION_CONVERGED`):

| evaluated bid | pass rule | delta | 95% t-interval | verdict |
|---:|---|---:|---|---|
| $31 | `STOP_NOW` | `+0.00291` | `[-0.00211, +0.00793]` | unresolved |
| $40 | `FIXED_MARKET` | `+0.00291` | `[-0.00211, +0.00793]` | unresolved |
| $65 | `FIXED_MARKET` | `-0.09332` | `[-0.09788, -0.08875]` | unfavorable |
| $80 | `FIXED_MARKET` | `-0.09332` | `[-0.09788, -0.08875]` | unfavorable |
| $100 | `FIXED_MARKET` | `-0.11330` | `[-0.11715, -0.10944]` | unfavorable |
| $128 | `FIXED_MARKET` | `-0.11834` | `[-0.12265, -0.11403]` | unfavorable |

`UNDERPOWERED`. **No price was favorable**, so there is no bracket: the ladder
never establishes that buying this RB is worth it at *any* price, and a
reservation price cannot be bracketed from below by a price that was never
favorable. At `$31` the interval straddles zero and only 36% of draws are
positive. No integer walk was run inside anything — there is no bracket to walk,
and the endpoints above are the only prices evaluated.

## Uncertainty

Reported at the `$31`/`$36` rung; the same structure holds at every price.

| | QB @ $36 | RB @ $31 |
|---|---:|---:|
| K (allocation draws) | 11 | 11 |
| mean delta | `-0.08357` | `+0.00291` |
| median delta | `-0.08450` | `-0.00075` |
| between-allocation SD | `0.00466` | `0.00747` |
| SE of the ensemble mean | `0.00141` | `0.00225` |
| RMS within-allocation season SE | `0.00517` | `0.00638` |
| 95% cluster t-interval (df = 10) | `[-0.08670, -0.08044]` | `[-0.00211, +0.00793]` |
| draws with positive delta | 0 / 11 | 4 / 11 |
| distinct allocation worlds | 11 | 11 |
| redundant draws | 0 | 0 |
| effective K | 11 | 11 |

The between-allocation SD and the within-allocation season SE are reported
separately and **never pooled**. They answer different questions: the first asks
how much the answer depends on which auction future occurs, the second how well
4,000 seasons pin one future down. The interval is a cluster t-interval on 11
allocation draws with df = 10 — **not** 44,000 seasons treated as independent
futures. For the RB the within-allocation season SE is nearly three times the SE
of the ensemble mean, which is why the season count is not the binding
constraint here; the draw count is.

## What this run establishes, and what it does not

**Establishes.** The restricted union was a real defect. It cost `UF` roughly
five proxy points, and it fully accounts for the QB's negative possession
effect. The fix is structural: one union built from `$1` upward, one
`EvalContext`, all five branches, possession price-invariant by assertion.

**Does not establish.** Any maximum bid, for either candidate. The QB's union is
not converged and its reservation reading is withheld. The RB's union is
converged but no price was favorable, so nothing is bracketed. The RB's
possession effect remains slightly negative and unexplained.

Convergence, where claimed, is over the **evaluated accumulated union** — more
search effort did not change the chosen construction. It is not a claim that no
better roster exists in the full combinatorial space.

## Artifacts

- `docs/FULL_BUDGET_UNION_sanitized.json` — positions, prices, CE, union
  diagnostics. No player names, ids, or projections.
- `local_data/tactical/full_budget_union.json` — player-level output. **Gitignored.**
- `ce-lab tactical full-budget-union`
- `src/ceauction/tactical/union.py`, `union_experiment.py`,
  `tests/test_full_budget_union.py` (26 tests)
