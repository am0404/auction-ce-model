# Mid-auction reservation frontiers (SIMULATED state)

> SIMULATED MID-AUCTION STATE -- invented from the market prior for frontier testing. NOT observed, NOT recorded, NOT historical, NOT calibrated. No league sale has ever been recorded.

`ce-lab tactical midauction-frontier`. Real player board, **invented**
auction history. Player-level detail is in `local_data/tactical/`, gitignored.

## Why a mid-auction state was needed

At the empty room every owner holds $200 and fifteen slots, so paying $6 or
$32 left the same completions affordable: the opening ladders returned
`FRONTIER_NOT_REACHED` and the decomposition showed our payment effect near
-0.09 regardless of price. **Money has to be scarce before a reservation
price can exist.** These states make it scarce.

## Simulated states

| state | sales | room spend | focus $ left | focus slots | our legal max | top rival | able to bid | valid |
|:--|--:|--:|--:|--:|--:|--:|--:|:--|
| `balanced` | 66 | $1488 | $139 | 12 | $128 | $140 | 11 | ok |
| `qb_inflation` | 66 | $1613 | $115 | 12 | $104 | $130 | 11 | ok |
| `skill_inflation` | 66 | $1734 | $139 | 12 | $128 | $130 | 11 | ok |

All three reconcile: dollars balance to $2,400, no duplicate ownership, $1
per open slot preserved, both frontier candidates left unsold. Generation is
deterministic for a fixed seed. `qb_inflation` and `skill_inflation` are
stated positional-tilt scenarios (QB +40%, RB/WR +35%) — **not** measured
market behaviour.

## Pass-price semantics

A reservation price is meaningless until you say what happens when you stop.
The primary ladders use **STOP_NOW**: the named rival already leads at `q`,
and we sweep our price `p`. The counterfactual is always "the leader gets
him at his current standing bid", so `q` is held fixed across the ladder.

**Be precise about what that is.** Pure STOP_NOW is a single point,
`p = q + increment`. Sweeping `p` above that holds `q` fixed, which is
formally the FIXED_MARKET rule with `q` taken from the standing bid. Both are
printed on every row and the two are never blended.

## QB and RB coarse ladders

### QB — leader Team07 standing $35, market band [24, 26, 28], anchor $52, lineup improvement +1.46

| p | q | audited | feasible | delta | +/-95% | between-SD | within-SE | sign+ | verdict | reused | runtime |
|--:|--:|:--|--:|--:|--:|--:|--:|--:|:--|:--|--:|
| 1 | 35 | yes | 39 | +0.05754 | 0.00470 | 0.00699 | 0.00608 | 1.00 | favorable | no | 56s |
| 5 | 35 | yes | 39 | +0.05754 | 0.00470 | 0.00699 | 0.00608 | 1.00 | favorable | yes | 26s |
| 10 | 35 | no | 39 | — | — | — | — | — | proxy pre-pass only | — | — |
| 20 | 35 | no | 39 | — | — | — | — | — | proxy pre-pass only | — | — |
| 24 | 35 | no | 39 | — | — | — | — | — | proxy pre-pass only | — | — |
| 26 | 35 | no | 39 | — | — | — | — | — | proxy pre-pass only | — | — |
| 28 | 35 | no | 39 | — | — | — | — | — | proxy pre-pass only | — | — |
| 30 | 35 | no | 38 | — | — | — | — | — | proxy pre-pass only | — | — |
| 40 | 35 | yes | 35 | +0.01666 | 0.00616 | 0.00917 | 0.00561 | 1.00 | favorable | no | 57s |
| 50 | 35 | no | 30 | — | — | — | — | — | proxy pre-pass only | — | — |
| 65 | 35 | yes | 21 | -0.02807 | 0.00379 | 0.00564 | 0.00501 | 0.00 | unfavorable | no | 56s |
| 80 | 35 | yes | 14 | -0.07107 | 0.00375 | 0.00558 | 0.00421 | 0.00 | unfavorable | no | 57s |
| 100 | 35 | yes | 8 | -0.07396 | 0.00367 | 0.00547 | 0.00415 | 0.00 | unfavorable | no | 59s |
| 128 | 35 | yes | 3 | -0.07425 | 0.00374 | 0.00556 | 0.00415 | 0.00 | unfavorable | no | 56s |

**RESERVATION_BRACKET** — highest favorable **$40**, next unfavorable **$65**, bracket **(40, 65)**, untested gap (41, 64), legal max $128, cache hits 11, nonmonotonic none.

### RB — leader Team11 standing $30, market band [24, 25, 27], anchor $28, lineup improvement +2.50

| p | q | audited | feasible | delta | +/-95% | between-SD | within-SE | sign+ | verdict | reused | runtime |
|--:|--:|:--|--:|--:|--:|--:|--:|--:|:--|:--|--:|
| 1 | 30 | yes | 36 | +0.05298 | 0.00388 | 0.00577 | 0.00436 | 1.00 | favorable | no | 56s |
| 5 | 30 | yes | 36 | +0.05286 | 0.00392 | 0.00583 | 0.00436 | 1.00 | favorable | yes | 28s |
| 10 | 30 | no | 36 | — | — | — | — | — | proxy pre-pass only | — | — |
| 20 | 30 | no | 36 | — | — | — | — | — | proxy pre-pass only | — | — |
| 24 | 30 | no | 36 | — | — | — | — | — | proxy pre-pass only | — | — |
| 25 | 30 | no | 36 | — | — | — | — | — | proxy pre-pass only | — | — |
| 27 | 30 | no | 36 | — | — | — | — | — | proxy pre-pass only | — | — |
| 30 | 30 | no | 36 | — | — | — | — | — | proxy pre-pass only | — | — |
| 40 | 30 | yes | 31 | +0.05054 | 0.00415 | 0.00618 | 0.00431 | 1.00 | favorable | yes | 42s |
| 50 | 30 | no | 28 | — | — | — | — | — | proxy pre-pass only | — | — |
| 65 | 30 | yes | 19 | +0.01980 | 0.00325 | 0.00484 | 0.00358 | 1.00 | favorable | no | 56s |
| 80 | 30 | yes | 12 | +0.01298 | 0.00360 | 0.00536 | 0.00346 | 1.00 | favorable | no | 59s |
| 100 | 30 | yes | 7 | -0.01211 | 0.00286 | 0.00425 | 0.00270 | 0.00 | unfavorable | no | 58s |
| 128 | 30 | yes | 3 | -0.02102 | 0.00186 | 0.00276 | 0.00230 | 0.00 | unfavorable | no | 56s |

**RESERVATION_BRACKET** — highest favorable **$80**, next unfavorable **$100**, bracket **(80, 100)**, untested gap (81, 99), legal max $128, cache hits 15, nonmonotonic none.

Both ladders are monotone and sign-stable across K=11. **These are brackets,
not exact max bids** — the integer gap between the favorable and unfavorable
prices was not walked.

## Decomposition around the RB crossing

| price point | p | q | our payment | our possession | denial | rival payment | total | residual |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| low_favorable | 1 | 30 | +0.00143 | +0.05141 | -0.00189 | -0.00104 | +0.04991 | 0.0e+00 |
| highest_favorable | 80 | 30 | -0.05214 | +0.05141 | -0.00189 | -0.00104 | -0.00366 | -0.0e+00 |
| first_unfavorable | 100 | 30 | -0.06084 | +0.05141 | -0.00189 | -0.00104 | -0.01236 | 0.0e+00 |

**The crossing is caused by our payment opportunity cost, and by nothing
else.** Three of the four components are price-independent by construction —
possession (+0.05141), denial (-0.00189) and rival payment (-0.00104) depend
on `q` and on who holds the player, not on what we pay. Only our payment
moves: `+0.00143` at $1, `-0.05214` at $80, `-0.06084` at $100. Telescoping
residual is exactly 0.0 at every point.

This is the answer the opening-state experiment could not give: there, our
payment sat near -0.09 at every price because fifteen open slots meant the
money's alternative use was a whole roster. Here it scales with `p`, and a
crossing exists.

## Two findings that undercut a clean story

### 1. The pass rule changes the answer completely

Under **RIVAL_OUTBIDS** (we bid `p`, the rival answers `q = p+1`), the same
RB returns **FRONTIER_NOT_REACHED** — favorable through
$40 with no unfavorable price below the legal maximum.
Under STOP_NOW the same candidate brackets at (80, 100).

That is not a discrepancy to reconcile; it is the point. If the rival will
always outbid us by a dollar, we never overpay by stopping — the pass branch
gets worse exactly as fast as the buy branch does. **A reservation price is
conditional on the pass rule, and quoting one without naming the rule is
meaningless.**

### 2. Recipient identity moves the bracket enormously

Same candidate, same state, same pass rule, different named recipient:

```
leader Team11   (standing $30)  bracket (80, 100)
alt    Team09                       bracket (1, 80)   (RESERVATION_BRACKET)
```

Passing to one rival and passing to another are different decisions with
different reservation prices. At the empty room this effect was ~0.001;
here it moves the bracket's lower edge from $80 to $1. **Recipients must
never be averaged before their frontiers are computed.**

### 3. A completion-set discrepancy at the bracket's lower edge

At `p = $80` the ladder reports `+0.01298` (favorable) and the decomposition
reports `-0.00366` (negative). Same state, same `q`, same K=11 schedule. The
cause is which completion set the buy arm searches: the ladder hands it the
nested pre-pass feasible set, while `decompose` runs its own beam. **The RB
bracket's lower edge is therefore not trustworthy to the dollar** — the true
crossing may sit below $80. The upper edge ($100 unfavorable) is agreed by
both paths.

## Empty room vs mid-auction

| | empty room | SIMULATED balanced mid-auction |
|:--|:--|:--|
| our budget | $200 | $139 |
| our open slots | 15 | 12 |
| our legal max on the candidate | $186 | $128 |
| our payment effect | ~-0.09 at every price | -0.0? scaling with p |
| denial | +0.026 (RB) | -0.0019 (RB) |
| recipient identity effect | ~0.001 | moves the bracket $80 -> $1 |
| frontier | FRONTIER_NOT_REACHED | RESERVATION_BRACKET |

Money binds because 66 simulated sales removed $1,488 from the room and left
us $139 for twelve slots. **This is a property of an invented history, not
learned league behaviour.**

## Runtime and reuse

Frontier ladders: 829s for both candidates (26 joint-world cache hits). Decomposition + sensitivity + alternate recipient: 893s. ~56s per audited price at K=11 x 4,000 seasons; ~26s when the pass arm is served from cache.
