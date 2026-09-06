# Quota-free marginal diagnostics

`ce-lab tactical marginal-diagnostics`. Real board, sanitized aggregates.
Player-level detail is in `local_data/tactical/marginal_diagnostics.json`,
which is gitignored.

## What the 1/4/6/3 template was doing

The pilot's diagnostic filled fourteen reference slots with a hard-coded
**1 QB / 4 RB / 6 WR / 3 TE** roster. Legal, and arbitrary. Its exact reach:

| affected | how |
|:--|:--|
| lineup-improvement number | entirely — it *was* the comparison roster |
| displaced-player identification | entirely |
| starter vs bench classification | entirely |
| **proxy-only vs CE-audited sampling** | **entirely — it decided who got 4,000 seasons** |
| which positions got an ensemble run | entirely (TE was skipped as proxy-only) |
| position summaries, TE-vs-WR and QB findings | entirely |
| candidate *selection* | no — selection is by market band and position rank |
| completion search, shared-board allocation, buy/pass CE, max-bid | **no** |

So the CE mathematics was never touched. What the template controlled was
**which real players ever reached that mathematics**, which is worse: a
wrong number can be checked, a player who is never simulated cannot.

Reserving three places for tight ends made a fourth tight end look useless.
Reserving one for a quarterback flattered every second quarterback. Neither
reservation corresponds to anything in the league, which has no TE slot, no
required TE, no QB maximum, and seven unrestricted bench places.

## What replaces it

Two bounded completion searches over the real remaining board, under exact
lineup eligibility and the $1-per-slot reserve:

* **without** — the best legal fifteen the board offers, candidate excluded.
* **with** — the candidate bought at price `p`, then the best legal fourteen
  from the remaining board and budget.

`lineup_improvement = best_legal_eight(with) - best_legal_eight(without)`.
Composition is an **output**. No position is required, capped, or reserved.

The starting eight is chosen by the same `select_lineups_mask` the season
simulator uses, so a roster of fifteen quarterbacks starts exactly 2.0 of
them (QB seat plus superflex) and a mixed roster starts 7.98. That is the
eligibility graph, not a template, and a test asserts both numbers.

## A search-convergence defect this exposed, and its fix

Economics says acquiring the same player for **less** can never be worse:
the money not spent stays available. The first converged-looking run
violated that badly.

```
beam_width  candidate_pool   improvement@$18   improvement@$1   violation
        24              30            +1.30           -4.64        +5.94
        64              60            +2.19           -6.04        +8.23
       160              90            +1.21           +1.16        +0.05
       320             140            -1.08           +0.86        -1.94
```

At the width the pilot used, the search error was **larger than the effects
being measured**. The first quota-free table I produced at beam_width=32 was
therefore not trustworthy and has been discarded, not published. Defaults
are now 160/90, `price_monotonicity_violation` is computed for every
candidate, and the CLI prints how many candidates fail it.

Current run: **0/12 violations, worst +0.45/wk**.

## The twelve candidates, quota-free and converged

| pos | tier | $ | improve @price | improve @$1 | start% | role | policy | composition (with) |
|:--|:--|--:|--:|--:|--:|:--|:--|:--|
| QB | expensive | 26 | +0.05 | +5.02 | 81% | replaceable starter | proxy only | QB3/RB8/WR2/TE2 |
| QB | mid | 21 | +1.08 | +5.55 | 87% | marginal starter | 4000-season audit | QB3/RB8/WR3/TE1 |
| QB | cheap | 19 | -1.78 | +2.97 | 25% | aggregate depth | proxy only | QB4/RB5/WR2/TE4 |
| RB | expensive | 44 | -0.39 | +8.30 | 83% | replaceable starter | proxy only | QB3/RB7/WR4/TE1 |
| RB | mid | 25 | +2.46 | +5.75 | 92% | marginal starter | 4000-season audit | QB3/RB7/WR3/TE2 |
| RB | cheap | 15 | +0.31 | +2.36 | 88% | marginal starter | 4000-season audit | QB3/RB8/WR2/TE2 |
| WR | expensive | 32 | +1.41 | +7.51 | 88% | marginal starter | 4000-season audit | QB3/RB7/WR4/TE1 |
| WR | mid | 17 | -0.42 | +3.41 | 81% | replaceable starter | proxy only | QB3/RB7/WR3/TE2 |
| WR | cheap | 8 | +2.20 | +2.06 | 86% | marginal starter | 4000-season audit | QB3/RB8/WR3/TE1 |
| TE | expensive | 13 | +1.89 | +4.42 | 88% | marginal starter | 4000-season audit | QB3/RB7/WR3/TE2 |
| TE | mid | 10 | +1.98 | +2.11 | 84% | marginal starter | 4000-season audit | QB3/RB7/WR3/TE2 |
| TE | cheap | 6 | +2.24 | +1.79 | 47% | marginal starter | 4000-season audit | QB3/RB7/WR2/TE3 |

**8 audited, 4 proxy-only.**

### Naturally selected compositions

```
template (old, hard-coded):  QB1/RB4/WR6/TE3
naturally selected:          QB3/RB7/WR3/TE2   (4/12 candidates)
naturally selected:          QB3/RB8/WR2/TE2   (2/12 candidates)
naturally selected:          QB3/RB8/WR3/TE1   (2/12 candidates)
naturally selected:          QB3/RB7/WR4/TE1   (2/12 candidates)
naturally selected:          QB4/RB5/WR2/TE4   (1/12 candidates)
naturally selected:          QB3/RB7/WR2/TE3   (1/12 candidates)
```

The board wants **three or four quarterbacks and seven or eight backs**, not
one and four. Nothing like the template, and it varies by candidate, which a
fixed template cannot do by construction.

## Before vs after

| | template (1/4/6/3) | quota-free, converged |
|:--|:--|:--|
| audited / proxy-only | 6 / 6 | 8 / 4 |
| TE1 improvement | 0.22 | +1.89 |
| TE1 policy | proxy only | **4,000-season audit** |
| all three TEs | proxy only | **all three audited** |
| QB1 improvement | 6.86 | +0.05 at $26 (+5.02 at $1) |
| QB1 policy | audit | **proxy only — replaceable at his price** |
| roster composition | fixed QB1/RB4/WR6/TE3 | varies; QB3-4, RB5-8, WR2-4, TE1-4 |

Both headline position findings from the pilot **reversed**. The TE finding
("no scarcity premium, TE1 is proxy-only") was an artefact of three reserved
TE places. The QB finding ("all three QBs show the largest improvements") was
an artefact of one reserved QB place.

## Roles now come from measurement

Improvement leads, not start share. On a fifteen-man roster with eight
starting slots almost everyone the completion chose starts most weeks, so a
share threshold alone classified all twelve as starters and audited the whole
board. A new role, **replaceable starter**, names the honest case: he starts,
but the completion is as good without him because his price buys more
elsewhere. That is a price judgement, and `improvement_at_min` reports what
he would be worth for $1 so player quality and price are never conflated.

## Contingency value is NOT priced

No conditional-backfield or QB-insurance mapping exists in this repository.
Bench players' raw points are deliberately **not** converted into lineup
value, and no contingency premium is invented. QB3 insurance in particular
cannot be measured from an empty room, which contains no quarterback to
insure. This is recorded on every diagnostic row.

## K=11 ensemble confirmation

| position | K | mean delta | between-alloc SD | within-alloc SE | CI95 | sign+ | verdict |
|:--|--:|--:|--:|--:|:--|--:|:--|
| RB (cheap) | 11 | +0.00775 | 0.01862 | 0.00804 | [-0.00476, +0.02026] | 0.91 | unresolved |
| QB (mid) | 11 | -0.03814 | 0.02900 | 0.00781 | [-0.05762, -0.01865] | 0.00 | unfavorable |
| WR (cheap) | 11 | -0.02727 | 0.01987 | 0.00795 | [-0.04062, -0.01392] | 0.18 | unfavorable |
| TE (expensive) | 11 | -0.06091 | 0.02636 | 0.00785 | [-0.07862, -0.04320] | 0.00 | unfavorable |

**A TE qualified for CE auditing and was run**, which under the template was
impossible. Opening symmetry survived the change:

```
Team02 vs Team03  single-seed gap -0.01725
                  ensemble mean   +0.00071
                  CI95            [-0.00766, +0.00907]  contains zero: True
                  persistent label effect: False
```

**Caveat, stated rather than buried:** this ensemble ran with the *pre-
convergence* (beam_width=32) diagnostic deciding which candidate to audit per
position, so its candidates are not the ones the converged table above would
select. The CE numbers are valid for the candidates actually run; the
selection needs re-running at the converged width.

## Verdict: PARTIAL GO for real price-frontier testing

* No positional quota affects diagnostics or sampling. **Yes.**
* Roles come from legal completion. **Yes.**
* QB and TE follow exact eligibility — 15 QBs start 2.0, TEs compete for
  WR/TE seats with no reservation. **Yes.**
* Conservation and symmetry intact. **Yes.**
* Meaningful candidates still produce stable ensemble effects — **partly.**
  Three of four positions resolved at K=11; the primary RB did not
  (sign+ 0.91, CI spans zero).

The blocker is not quotas any more. It is that the ensemble was selected by
an unconverged diagnostic, so the confirmation does not yet cover the players
the corrected diagnostic actually nominates.
