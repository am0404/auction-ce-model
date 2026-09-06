# Real-board tactical pilot — twelve players, and a NO-GO

`ce-lab tactical real-pilot`. Real 2026 projections and real Sleeper `2qb`
anchors. 232s total. **Sanitized aggregate only** — no player
name, no proprietary projection row, no player-level dollar figure. The
player-level report is written to `local_data/tactical/`, which is gitignored.

**This is a pilot of twelve players at the empty-room state. It is not a
draft board, and it prices nothing mid-auction.**

## Verdict: NO-GO for full targeted precomputation

Not because the machinery is wrong. Conservation held on every arm, league
CE summed to exactly 1.0 everywhere, runtime is comfortable, and the QB and
TE behaviour matches the real lineup graph. The blocker is narrower and
worse: **on the real board, which of the ~10^18 plausible shared-board
continuations we happen to draw moves the answer more than the answer
itself.**

```
position   between-allocation SD   within-allocation SE   SD/effect   sign stable
QB                       0.02547                0.00626        0.97   True
RB                       0.04717                0.00645        2.14   False
WR                       0.04980                0.00632        2.11   False
```

On the fabricated board the same ratio was **0.016 to 0.157**
(`docs/TACTICAL_POWER.md`). Here it is **0.97 to 2.14** — thirteen to a
hundred and thirty times worse — and for RB and WR the sign of the effect
flips between seeds. Three seeds is a small sample, but a sign flip is not a
precision problem; it is the estimate having no stable value to be precise
about.

## Real input coverage

```
contract_players                549
mapped_playerspecs              260
pool_limit                      260
with_market_anchor              149
without_market_anchor           111
with_individual_injury          222
injury_fallback_positional      38
unresolved_placeholder_fields   260
mapping_warnings                3
mapping_skipped                 0
anchor_rows                     1024
position_counts                 {'QB': 35, 'RB': 70, 'WR': 112, 'TE': 43}
market_scenario                 base
performance_scenario            median_target/full_health/week_sd/exclude
```

Anchor match is 149 of 260 mapped players (57%). The 111 unanchored are
priced at the $1 league minimum — a **stated floor, not a valuation**.
Ambiguous identities from the market audit: 1 ambiguous, 1 position conflict,
3 duplicate anchor names; none was guessed.

Opening room validated: 12 owners x $200 = **$2,400**; 12 x 15 = **180**
slots; **$180** minimum committed; every owner's opening legal maximum is
**$186**; no duplicate player ids; 260 available players for 180 slots.

## Candidate selection

12 candidates, {'QB': 3, 'RB': 3, 'WR': 3, 'TE': 3}, {'expensive': 4, 'mid': 4, 'cheap': 4}.
Substitutions: 0.
Chosen by each position's own clearing-price order at fixed quantiles —
no name appears in the selection code, and a test asserts it.

## Diagnostics and the sampling policy

| pos | tier | weekly lineup improvement | role | policy |
|:--|:--|--:|:--|:--|
| QB | expensive | 6.86 | clear starter | 4000-season audit |
| QB | mid | 4.86 | clear starter | 4000-season audit |
| QB | cheap | 3.87 | clear starter | 4000-season audit |
| RB | expensive | 3.30 | clear starter | 4000-season audit |
| RB | mid | 0.67 | marginal starter | 4000-season audit |
| RB | cheap | 0.19 | bench/insurance | proxy only |
| WR | expensive | 1.71 | marginal starter | 4000-season audit |
| WR | mid | 0.09 | bench/insurance | proxy only |
| WR | cheap | 0.09 | bench/insurance | proxy only |
| TE | expensive | 0.22 | bench/insurance | proxy only |
| TE | mid | 0.05 | bench/insurance | proxy only |
| TE | cheap | 0.01 | bench/insurance | proxy only |

Six audited, six proxy-only. **The policy follows lineup improvement, not
price** — and the two disagree sharply on the real board.

## Base-price audit

| pos | tier | $ | delta CE | +/-95% | verdict | our rank pass -> buy |
|:--|:--|--:|--:|--:|:--|:--|
| QB | expensive | 26 | +0.02250 | 0.01033 | favorable | 8 -> 6 |
| QB | mid | 21 | -0.00425 | 0.00782 | unresolved | 9 -> 9 |
| QB | cheap | 19 | -0.06850 | 0.00984 | unfavorable | 6 -> 11 |
| RB | expensive | 44 | +0.06450 | 0.01320 | favorable | 6 -> 1 |
| RB | mid | 25 | +0.09025 | 0.01185 | favorable | 11 -> 2 |
| WR | expensive | 32 | +0.07950 | 0.01185 | favorable | 9 -> 2 |

Resolved 5, unresolved 1, proxy-only 6. Conservation passed on every arm; league
CE summed to exactly 1.0 on every arm.

**Read these numbers against the allocation table above before using any of
them.** Five of six 'resolved' verdicts come from a single allocation seed,
and the seed sweep shows that seed choice moves the estimate by more than
its own confidence interval.

## Quarterbacks and the superflex

The lineup-improvement diagnostic uses a reference roster of one QB, four
RBs, six WRs and three TEs, so the **superflex seat is open** and a QB
candidate competes against the best RB/WR/TE fallback for it. That is the
real question this league poses.

* All three QBs show the largest lineup improvements on the board (6.86,
  4.86, 3.87) — a second startable QB beats the skill-position fallback.
* Nothing requires a second QB. A one-QB roster is legal and tested; a
  five-QB roster is legal and tested. No QB premium exists in code.
* **QB3 insurance was not measured.** At the empty-room state there is no
  existing QB availability risk to insure against. Stating this rather than
  pretending otherwise: the pilot cannot price a QB3.

## Tight ends and the missing TE slot

This is the sharpest market/CE disagreement in the pilot:

```
TE expensive:  Sleeper anchor $34,  clearing band $12/$13/$14,
               weekly lineup improvement 0.22  ->  PROXY ONLY
WR expensive:  Sleeper anchor $48,  clearing band $30/$32/$34,
               weekly lineup improvement 1.71  ->  audited, +0.0795
```

With no dedicated TE slot, a tight end competes for a WR/TE seat directly
against receivers. The top TE projects at 11.04 weekly against the top WR's
14.78, so he does not take a seat. **CE awards no scarcity premium, and none
is hard-coded** (asserted by test).

Two honest caveats. The reference roster already carries three tight ends,
so a fourth reads low; a roster thin at WR/TE would value him differently.
And the anchor is not wrong — it is a *behavioural market input*. Sleeper
users pay $34 for this player. That the CE lineup effect is 0.22 means the
market and our equity model disagree, not that the market is a mistake.

## RB and WR

Improvement is measured through the whole eight-slot lineup graph, not by
raw points: RB mid at 13.40 projected weekly points contributes 0.67 to the
starting lineup, and RB cheap at 9.15 contributes 0.19. A bench player is
not handed CE value for points he will not start.

## Handcuffs

**No real conditional-backfield mapping exists in this repository.** The
contract carries standalone projections only. This pilot therefore does not
price handcuffs, and no named handcuff recommendation may be made from it.
Recorded as a blocker.

## Priority ladders — and why they produced no bracket

| pos | tested prices | nested | favorable | bracket | legal max | band | anchor |
|:--|:--|:--|:--|:--|--:|:--|--:|
| QB | [14, 24, 26, 28, 38] | True | all | none found | 186 | [24, 26, 28] | 52 |
| RB | [31, 41, 44, 47, 57] | True | all | none found | 186 | [41, 44, 47] | 58 |
| WR | [20, 30, 32, 34, 44] | True | all | none found | 186 | [30, 32, 34] | 48 |
| TE | [2, 12, 13, 14, 24] | True | all | none found | 186 | [12, 13, 14] | 34 |

**Every tested price returned an identical delta.** That is not a frontier;
it is the ladder never binding. With $186 and fifteen open slots, paying $14
or $38 for a quarterback leaves the same set of completions affordable, so
the same joint world is selected and the same seasons give the same equity.
The nested opportunity set is working exactly as designed — and it means an
empty-room ladder cannot locate a reservation price. **No robust maximum
from this table may be quoted**; the highest favorable price is simply the
highest price tested.

A reservation price needs a state where money is actually scarce. That is
mid-auction, which this pilot does not model.

## Opening-state symmetry — FAILED

```
same candidate to Team02: CE 0.04325  rank 8
same candidate to Team03: CE 0.055  rank 7
gap 0.01175   ~1 SE 0.00322   within noise: False
```

In an empty room Team02 and Team03 are structurally identical — same money,
same slots, same board, no history. They must be interchangeable. They are
3.6 standard errors apart.

Same root cause as the allocation instability: the shared-board continuation
draws per-(player, owner) tie-breaking jitter, and over 180 allocations on a
260-player board that jitter compounds into materially different opponents.
**Opponent identity is being manufactured before a single sale.**

## Flags raised, not silently corrected

| flag | status |
|:--|:--|
| CE reservation above legal maximum | none |
| CE reservation above every market scenario | not determinable — ladders never bound |
| elite projected player valued near zero | TE1 (anchor $34) reads 0.22 improvement — real, from the lineup graph |
| fringe player with enormous CE effect | none |
| TE scarcity premium despite no TE slot | none |
| strict-2QB behaviour | none |
| negative opportunity cost from paying more | none — ladders are flat, not decreasing |
| non-nested price sets | none; nested on all four ladders |
| allocation instability > CE effect | **YES, on all three tested positions** |
| pass branch pinned at zero | none |
| extreme recipient sensitivity in a symmetric room | **YES** |

## Runtime

```
real board load                20.5s
mean per audited candidate     7.53s  (buy + pass, 4,000-season holdout)
audited candidates             6
proxy-only candidates          6  (0 seasons spent, by policy)
total pilot                    231.8s
```

Runtime is not the blocker. Nothing here runs on a 10-second clock; the live
path remains cached output.

## What must be fixed before a GO

1. **Allocation instability.** Either average the estimate over many
   continuations and report a between-allocation interval alongside the
   paired SE, or reduce the continuation's dependence on tie-breaking noise.
   Quoting a single-seed number on this board is not defensible.
2. **Opening symmetry.** Structurally identical opponents must produce
   equivalent branches until a sale distinguishes them.
3. **A mid-auction state.** Empty-room ladders cannot bind, so they cannot
   produce a reservation price at all.
