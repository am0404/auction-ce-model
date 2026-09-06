# Player value across roster-strength regimes (fabricated)

> **WITHDRAWN AS A CAUSAL COMPARISON — CONFOUNDED.**
> These four regimes varied the number of pre-owned players, and with it our
> money, our open slots, our positional pattern, the remaining board, which
> players rivals could reach, and every rival's completion. Four factors moved
> at once, so nothing measured here can be attributed to roster strength. The
> labels did not match the outcomes either: `bye_contender` finished 6th and
> `favorite` 7th.
>
> The controlled one-factor replacement is `docs/TACTICAL_CONTEXT.md`, and its
> result **contradicts the conclusion below**: with one factor isolated, the
> candidate's marginal equity is unresolved in every roster context.
>
> Retained as historical evidence only. Do not cite it as a finding.


`python -m ceauction.tactical.regime_experiment 4000`.
4,000-season holdout, 1,000-season independent
selection sample, 58s. Every player is `Fabricated####`.

Every arm is a reconciled joint world over a **nested** opportunity set.
Only our own roster differs between regimes: the eleven rival rosters, their
spending, the pool, the cost book, the market state and every seed are held
fixed and asserted identical by test.

## Starting positions (pass arm, before acquisition)

| regime | pre-owned | budget left | our proxy | field mean | field range | our CE | rank | league sum |
|:--|--:|--:|--:|--:|:--|--:|--:|--:|
| `underdog` | 2 | $176 | 92.5 | 106.89 | 105.3-112.4 | 0.00000 | 12 | 1.0000 |
| `bubble` | 3 | $146 | 102.4 | 106.29 | 105.1-108.3 | 0.02325 | 12 | 1.0000 |
| `bye_contender` | 8 | $96 | 106.6 | 106.07 | 105.0-106.6 | 0.08575 | 6 | 1.0000 |
| `favorite` | 11 | $90 | 106.8 | 106.01 | 105.0-106.6 | 0.08200 | 7 | 1.0000 |

Playoff and bye probabilities are not exposed per-team by the joint arm, so
they are not reported rather than being estimated from rank.

## The same player, the same price, four contexts

| regime | $ | ce_buy | ce_pass | delta | +/-95% | verdict | rank after | feasible set |
|:--|--:|--:|--:|--:|--:|:--|--:|--:|
| `underdog` | 1 | 0.02050 | 0.00000 | +0.02050 | 0.00439 | favorable | 12 | 11 |
| `underdog` | 13 | 0.02050 | 0.00000 | +0.02050 | 0.00439 | favorable | 12 | 11 |
| `underdog` | 20 | 0.02050 | 0.00000 | +0.02050 | 0.00439 | favorable | 12 | 11 |
| `underdog` | 30 | 0.02050 | 0.00000 | +0.02050 | 0.00439 | favorable | 12 | 11 |
| `bubble` | 1 | 0.04650 | 0.02325 | +0.02325 | 0.00745 | favorable | 12 | 10 |
| `bubble` | 13 | 0.04650 | 0.02325 | +0.02325 | 0.00745 | favorable | 12 | 10 |
| `bubble` | 20 | 0.04650 | 0.02325 | +0.02325 | 0.00745 | favorable | 12 | 10 |
| `bubble` | 30 | 0.04650 | 0.02325 | +0.02325 | 0.00745 | favorable | 12 | 10 |
| `bye_contender` | 1 | 0.08500 | 0.08575 | -0.00075 | 0.00864 | unresolved | 5 | 11 |
| `bye_contender` | 13 | 0.08500 | 0.08575 | -0.00075 | 0.00864 | unresolved | 5 | 11 |
| `bye_contender` | 20 | 0.08500 | 0.08575 | -0.00075 | 0.00864 | unresolved | 5 | 11 |
| `bye_contender` | 30 | 0.08500 | 0.08575 | -0.00075 | 0.00864 | unresolved | 5 | 11 |
| `favorite` | 1 | 0.09050 | 0.08200 | +0.00850 | 0.00903 | unresolved | 3 | 9 |
| `favorite` | 13 | 0.08775 | 0.08200 | +0.00575 | 0.00902 | unresolved | 4 | 9 |
| `favorite` | 20 | 0.08775 | 0.08200 | +0.00575 | 0.00902 | unresolved | 4 | 9 |
| `favorite` | 30 | 0.09025 | 0.08200 | +0.00825 | 0.00902 | unresolved | 3 | 9 |

### What CE actually says

**The candidate is worth most to the teams that need him and is not
resolvably worth anything to the teams that do not.**

* `underdog` (below the field): delta +0.02050 +/- 0.00439 at $1 -- **favorable**
* `bubble` (near the cutoff): delta +0.02325 +/- 0.00745 at $1 -- **favorable**
* `bye_contender` (competing for a bye): delta -0.00075 +/- 0.00864 at $1 -- **unresolved**
* `favorite` (already strong): delta +0.00850 +/- 0.00903 at $1 -- **unresolved**

This was not predetermined. The two weak regimes resolve favorable; the two
strong ones do not resolve at all at 4,000 seasons, and `bye_contender` is
very slightly negative. Valuing this player once, from one roster context,
and calling the answer his value would have been wrong in both directions.

### Price barely moves CE within a regime -- and that is the fix working

`underdog` and `bubble` return an *identical* delta at every tested price,
and `bye_contender` likewise. With a nested opportunity set the same best
construction is available at $1 and at $30, the rival continuation depends
only on which players we took (never on what we paid), so the same joint
world is selected and the same seasons produce the same equity. Price enters
only when it prices a construction out of reach -- visible in the shrinking
`feasible set` column and in `favorite`, whose selection flips between two
near-tied worlds.

Before this branch the same sweep moved CE by 0.033 across prices purely
through beam path dependence.

## Recipient identity (`bubble` regime)

| recipient | pays | our CE if he gets him | our rank | his CE after | his rank after | allocation |
|:--|--:|--:|--:|--:|--:|:--|
| Owner12 | $19 | 0.02325 | 12 | 0.13350 | 1 | `91bc049dd192994d` |
| Owner02 | $19 | 0.01825 | 12 | 0.12900 | 1 | `4acbe780c3bc71d2` |

Each branch was built and evaluated separately before any weighting. Losing
the player to Owner02 costs us more than losing him to Owner12
(0.01825 vs 0.02325), and the two recipients end
up with different equity themselves. Opponent identity is not averaged away.

## Nesting evidence

| regime | nested | union | feasible by price |
|:--|:--|--:|:--|
| `underdog` | True | 11 | {'1': 11, '13': 11, '20': 11, '30': 11} |
| `bubble` | True | 10 | {'1': 10, '13': 10, '20': 10, '30': 10} |
| `bye_contender` | True | 11 | {'1': 11, '13': 11, '20': 11, '30': 11} |
| `favorite` | True | 9 | {'1': 9, '13': 9, '20': 9, '30': 9} |
