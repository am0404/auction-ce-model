# Controlled roster-context experiment (fabricated)

`ce-lab tactical context --sims 4000`, or
`python -m ceauction.tactical.context_experiment 4000`. 66s.
Holdout 4,000 seasons; independent selection sample
1,000. Seeds: board 20260906, selection
20260904, holdout 917324011. Every player is
`Fabricated####`.

## The previous regime table is WITHDRAWN as a causal comparison

`docs/TACTICAL_REGIMES.md` varied four things at once -- pre-owned count,
money, open slots and the remaining board -- so nothing it measured could be
attributed to roster strength. Its labels did not even match its outcomes
(`bye_contender` finished sixth, `favorite` seventh). **It is confounded and
its conclusion does not stand.** It remains in the repository as historical
evidence only.

## What is held constant, and what varies

Varying: the `base_mean` of the PlayerSpecs the focus team **already owns**.
Those players are off the board in every regime, so scaling their projected
scoring cannot reach the auction -- it changes how good our existing roster
is, and nothing else.

Held identical and asserted before any season is simulated:

```
structural differences across all pairs of 4 regimes: 0
```

covering focus owner id, budget remaining, spend, roster size, open slots,
player ids, positions, prices, the remaining-board ids, the remaining-board
costs, every rival's roster/spend/budget/slots, the market fingerprint, the
cost-book fingerprint, league settings, withdrawn players, the cast, the
candidate, its price, the pass recipient, the leader and every seed.

Expected differences, reported separately because they ARE the factor:

```
pool fingerprints differ:            True
focus-owned projection totals:       {'bye_bubble': 115.9852, 'favorite': 126.9272, 'playoff_bubble': 108.3258, 'underdog': 96.2896}
every OTHER projection identical:    True
```

## Calibration: labels earned from measured outcomes, not proxy points

The strength scale was tuned until the simulated playoff, bye and equity
outcomes matched the label. No label was assigned from roster points.

| regime | scale | weekly pts | mean wins | p(playoff) | p(bye) | CE | CE rank | league sum |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| `underdog` | 0.88 | 1336.6 | 8.66 | 0.040 | 0.004 | 0.0022 | 12 | 1.0000 |
| `playoff_bubble` | 0.99 | 1474.5 | 13.61 | 0.456 | 0.142 | 0.0707 | 12 | 1.0000 |
| `bye_bubble` | 1.06 | 1567.4 | 17.06 | 0.833 | 0.496 | 0.2427 | 1 | 1.0000 |
| `favorite` | 1.16 | 1700.2 | 21.33 | 0.992 | 0.915 | 0.5305 | 1 | 1.0000 |

* `underdog` — p(playoff) 0.040, rank 12: clearly outside the field, and
  **not pinned at zero equity** (0.0022), so a marginal effect could show.
* `playoff_bubble` — p(playoff) 0.456: materially uncertain, as required.
* `bye_bubble` — p(bye) 0.496: materially uncertain, as required.
* `favorite` — p(bye) 0.915, CE rank 1: clearly among the strongest.

`bye_bubble` also ranks 1st on CE. That is a real property of this weak
fabricated field, not a mislabel: its *bye* probability is the coin-flip the
label claims, which is the boundary being tested.

## Candidate results (player 166, recipient Owner12 at $10)

| regime | $ | ce_buy | ce_pass | delta | +/-95% | verdict | rank after | focus completion changed | rival alloc changed | feasible |
|:--|--:|--:|--:|--:|--:|:--|--:|:--|:--|--:|
| `underdog` | 1 | 0.00175 | 0.00225 | -0.00050 | 0.00183 | unresolved | 12 | False | False | 10 |
| `underdog` | 13 | 0.00175 | 0.00225 | -0.00050 | 0.00183 | unresolved | 12 | False | False | 10 |
| `underdog` | 20 | 0.00275 | 0.00225 | +0.00050 | 0.00196 | unresolved | 12 | True | True | 10 |
| `underdog` | 30 | 0.00275 | 0.00225 | +0.00050 | 0.00196 | unresolved | 12 | False | False | 10 |
| `playoff_bubble` | 1 | 0.06175 | 0.06475 | -0.00300 | 0.00790 | unresolved | 12 | False | False | 11 |
| `playoff_bubble` | 13 | 0.06175 | 0.06475 | -0.00300 | 0.00790 | unresolved | 12 | False | False | 11 |
| `playoff_bubble` | 20 | 0.06175 | 0.06475 | -0.00300 | 0.00790 | unresolved | 12 | False | False | 11 |
| `playoff_bubble` | 30 | 0.06175 | 0.06475 | -0.00300 | 0.00790 | unresolved | 12 | False | False | 11 |
| `bye_bubble` | 1 | 0.24600 | 0.24075 | +0.00525 | 0.01288 | unresolved | 1 | False | False | 11 |
| `bye_bubble` | 13 | 0.24600 | 0.24075 | +0.00525 | 0.01288 | unresolved | 1 | False | False | 11 |
| `bye_bubble` | 20 | 0.24600 | 0.24075 | +0.00525 | 0.01288 | unresolved | 1 | False | False | 11 |
| `bye_bubble` | 30 | 0.24500 | 0.24075 | +0.00425 | 0.01267 | unresolved | 1 | True | True | 11 |
| `favorite` | 1 | 0.53125 | 0.52375 | +0.00750 | 0.01276 | unresolved | 1 | False | False | 11 |
| `favorite` | 13 | 0.53125 | 0.52375 | +0.00750 | 0.01276 | unresolved | 1 | False | False | 11 |
| `favorite` | 20 | 0.53125 | 0.52375 | +0.00750 | 0.01276 | unresolved | 1 | False | False | 11 |
| `favorite` | 30 | 0.53125 | 0.52375 | +0.00750 | 0.01276 | unresolved | 1 | False | False | 11 |

## The result, stated bluntly

**Every arm is unresolved.** At 4,000 holdout seasons the candidate's
marginal championship equity is not distinguishable from zero in any of the
four roster contexts.

```
regime            delta at $1        95% half-width   |delta| / SE
underdog          -0.00050          0.00183          0.53
playoff_bubble    -0.00300          0.00790          0.74
bye_bubble        +0.00525          0.01288          0.80
favorite          +0.00750          0.01276          1.15
```

The sign pattern is suggestive and nothing more: negative for the two weak
contexts, positive for the two strong ones. That is the **opposite** of the
confounded experiment's conclusion, which said the player was worth most to
teams that need him. With one factor isolated, that conclusion does not
survive -- and neither does its reverse, because none of it resolves.

### How much sample would settle it

To resolve the `playoff_bubble` point estimate (-0.00300) at 95%
confidence needs roughly **27,743 holdout seasons** -- about
7x this run. That is the honest cost of an answer,
and it is why no threshold in this document is presented as settled.

### Why the effect is small here

Observable, not hand-waved:

* **The candidate is weak.** With eight pre-owned players and eleven full
  rival rosters, the best player left on the board is a marginal one. His
  weekly contribution is small against roster noise.
* **Price changes nothing within a regime.** `focus completion changed` and
  `rival alloc changed` are almost entirely `False`: the nested opportunity
  set makes the same construction available at $1 and $30, so the same joint
  world is selected and the same seasons give the same equity. Price enters
  only where it prices a construction out (the `feasible` column).
* **No arm is pinned at zero.** `underdog` sits at CE 0.0022 pre-acquisition,
  so a real effect had room to appear. It did not.

## What this experiment does and does not prove

**Proves:** the machinery is now capable of a clean one-factor comparison.
Structural state is provably identical across regimes; the labels are earned
from simulated playoff/bye/CE outcomes rather than asserted; conservation and
nesting invariants hold throughout; and league CE sums to one in every arm.

**Does not prove:** that roster context changes this player's value. At this
sample size and with this candidate, the effect is indistinguishable from
Monte Carlo noise. Roster-dependent CE valuation is **not yet demonstrated to
behave credibly** -- it is demonstrated to be measurable without confounds,
which is a different and lesser claim.
