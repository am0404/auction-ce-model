# Estimator power by candidate tier (fabricated)

`ce-lab tactical signal-power --sims 4000 --pilot-sims 1000`. 222s.
Every player is `Fabricated####`.

## Why the prior null did not invalidate the model

`docs/TACTICAL_CONTEXT.md` found no distinguishable effect for one weak
marginal candidate. **Weak-player irrelevance and estimator failure look
identical in a single cell and are opposite findings.** A bench player whose
true effect is 0.0005 CE *should* return unresolved at any affordable
sample; spending 30,000 seasons to prove he is worth nothing is the waste,
not the null. The blanket NO-GO drawn from that result was too broad.

This experiment asks the question that actually decides the architecture:
does a player who genuinely matters produce an effect the estimator can
resolve at a practical cost? He does, decisively.

## Candidate-tier calibration

Tiers vary **only** the candidate's `base_mean`. Position, bye week,
week-to-week variance, injury hazard and availability are untouched, so a
tier cannot smuggle in a durability or usage assumption. The label is earned
from the measured weekly starting-lineup improvement over a $1 replacement,
with the other fourteen slots filled by the *best* players on the board --
filling around him with replacement-level players would make every tier a
starter and destroy the distinction.

| tier | scale | base_mean | weekly improvement | starts? | displaces |
|:--|--:|--:|--:|:--|:--|
| `bench` | 0.55 | 6.07 | 0.00 | False | - |
| `marginal_starter` | 1.15 | 12.70 | 1.50 | True | 0 |
| `strong_starter` | 1.85 | 20.42 | 8.47 | True | 0 |
| `elite` | 2.9 | 32.02 | 18.92 | True | 0 |

Lineups are chosen from **pregame** information only, the same information
barrier the season simulator enforces: no player is startable here who could
not have been identified as startable before lineup lock.

## Experiment A -- power (price fixed at $13)

Four candidate tiers x four controlled roster contexts, so scoring strength
and roster context are separated from price.

| tier | context | weekly improve | delta CE | +/-95% | \|d\|/SE | verdict | n for 0.005 |
|:--|:--|--:|--:|--:|--:|:--|--:|
| `bench` | underdog | 0.00 | +0.00050 | 0.00120 | 0.82 | unresolved | 1,000 |
| `bench` | playoff_bubble | 0.00 | +0.00400 | 0.00650 | 1.21 | unresolved | 6,307 |
| `bench` | bye_bubble | 0.00 | -0.00025 | 0.01006 | 0.05 | unresolved | 15,072 |
| `bench` | favorite | 0.00 | -0.00850 | 0.01028 | 1.62 | unresolved | 19,480 |
| `marginal_starter` | underdog | 1.79 | +0.00275 | 0.00213 | 2.52 | favorable | 1,000 |
| `marginal_starter` | playoff_bubble | 1.50 | +0.02675 | 0.00970 | 5.40 | favorable | 13,869 |
| `marginal_starter` | bye_bubble | 1.24 | +0.02925 | 0.01475 | 3.89 | favorable | 30,606 |
| `marginal_starter` | favorite | 1.02 | +0.02125 | 0.01421 | 2.93 | favorable | 32,521 |
| `strong_starter` | underdog | 8.76 | +0.05050 | 0.00696 | 14.22 | favorable | 8,712 |
| `strong_starter` | playoff_bubble | 8.47 | +0.23850 | 0.01443 | 32.41 | favorable | 31,096 |
| `strong_starter` | bye_bubble | 8.21 | +0.28025 | 0.01684 | 32.62 | favorable | 45,284 |
| `strong_starter` | favorite | 7.84 | +0.19975 | 0.01707 | 22.93 | favorable | 48,882 |
| `elite` | underdog | 19.21 | +0.35175 | 0.01480 | 46.58 | favorable | 33,247 |
| `elite` | playoff_bubble | 18.92 | +0.54000 | 0.01576 | 67.18 | favorable | 38,359 |
| `elite` | bye_bubble | 18.66 | +0.57225 | 0.01628 | 68.91 | favorable | 44,118 |
| `elite` | favorite | 18.29 | +0.46750 | 0.01686 | 54.34 | favorable | 45,935 |

**Resolved at 4,000 confirmatory seasons:**

```
bench              0/4 contexts resolved
marginal_starter   4/4 contexts resolved
strong_starter     4/4 contexts resolved
elite              4/4 contexts resolved
```

* `bench` — 0/4. Effects 0.000 to 0.009, |d|/SE at most 1.62. Correct: the
  measured lineup improvement is **0.00**, so there is nothing to detect.
* `marginal_starter` — 4/4 at |d|/SE 2.5-5.4. Effects 0.003-0.029.
* `strong_starter` — 4/4 at |d|/SE 14-33. Effects 0.05-0.28.
* `elite` — 4/4 at |d|/SE 47-69. Effects 0.35-0.57.

The `n for 0.005` column is large even where the effect is enormous. That is
an artefact of the target, not a power problem: pinning a 0.54 effect to
+/-0.005 is pointless precision. The achieved half-width at 4,000 seasons is
~0.016, which resolves a 0.54 effect thirty times over.

## Required seasons by target half-width

| tier | context | 0.010 | 0.005 | 0.0025 |
|:--|:--|--:|--:|--:|
| `bench` | underdog | 1,000 | 1,000 | 1,846 |
| `bench` | playoff_bubble | 1,577 | 6,307 | 25,226 |
| `bench` | bye_bubble | 3,768 | 15,072 | 60,287 |
| `bench` | favorite | 4,870 | 19,480 | 77,918 |
| `marginal_starter` | underdog | 1,000 | 1,000 | 3,061 |
| `marginal_starter` | playoff_bubble | 3,468 | 13,869 | 55,473 |
| `marginal_starter` | bye_bubble | 7,652 | 30,606 | 122,424 |
| `marginal_starter` | favorite | 8,131 | 32,521 | 130,084 |
| `strong_starter` | underdog | 2,178 | 8,712 | 34,847 |
| `strong_starter` | playoff_bubble | 7,774 | 31,096 | 124,383 |
| `strong_starter` | bye_bubble | 11,321 | 45,284 | 181,136 |
| `strong_starter` | favorite | 12,221 | 48,882 | 195,528 |
| `elite` | underdog | 8,312 | 33,247 | 132,988 |
| `elite` | playoff_bubble | 9,590 | 38,359 | 153,434 |
| `elite` | bye_bubble | 11,030 | 44,118 | 176,470 |
| `elite` | favorite | 11,484 | 45,935 | 183,740 |

These are reporting choices, not claims of economic materiality.

## Pilot versus confirmatory sampling

Pilot 1,000 seasons at seed 20260904; confirmatory
4,000 at seed 917324011; cap 40,000.
The pilot estimates variance only. The confirmatory interval is drawn at a
different seed and **reuses no pilot observation** — inspecting a sample,
deciding to continue, then quoting a fixed-sample interval over the whole
thing is the optional-stopping error that makes an interval narrower than
its own coverage. When the required size exceeds the cap the plan reports
`UNDERPOWERED_AT_CAP` rather than a precise-looking number.

## Experiment B -- price sensitivity

Ladders are **experimental, not market predictions**. Nothing here claims
this room would pay these prices.

| tier | $ | delta CE | +/-95% | verdict | feasible set |
|:--|--:|--:|--:|:--|--:|
| `bench` | 1 | +0.00000 | 0.00219 | unresolved | 8 |
| `bench` | 3 | +0.00000 | 0.00219 | unresolved | 8 |
| `bench` | 5 | -0.00075 | 0.00827 | unresolved | 8 |
| `bench` | 10 | +0.00000 | 0.00817 | unresolved | 8 |
| `marginal_starter` | 1 | +0.02800 | 0.01003 | favorable | 11 |
| `marginal_starter` | 10 | +0.02775 | 0.00982 | favorable | 11 |
| `marginal_starter` | 20 | +0.02675 | 0.01004 | favorable | 11 |
| `marginal_starter` | 30 | +0.02375 | 0.00979 | favorable | 11 |
| `strong_starter` | 10 | +0.23300 | 0.01446 | favorable | 11 |
| `strong_starter` | 20 | +0.23475 | 0.01438 | favorable | 11 |
| `strong_starter` | 35 | +0.23475 | 0.01438 | favorable | 11 |
| `strong_starter` | 50 | +0.24825 | 0.01453 | favorable | 11 |
| `elite` | 20 | +0.55350 | 0.01558 | favorable | 10 |
| `elite` | 40 | +0.55575 | 0.01565 | favorable | 10 |
| `elite` | 60 | +0.55400 | 0.01559 | favorable | 8 |
| `elite` | 80 | +0.52775 | 0.01581 | favorable | 3 |

Nesting held on every ladder:

```
bench              nested=True  feasible_by_price={'1': 8, '3': 8, '5': 8, '10': 8}
marginal_starter   nested=True  feasible_by_price={'1': 11, '10': 11, '20': 11, '30': 11}
strong_starter     nested=True  feasible_by_price={'10': 11, '20': 11, '35': 11, '50': 11}
elite              nested=True  feasible_by_price={'20': 10, '40': 10, '60': 8, '80': 3}
```

## Allocation uncertainty versus season uncertainty

Two different things, never pooled:

* **Season noise** is the paired SE inside one shared-board allocation. It
  shrinks as `1/sqrt(n)`; more seasons fix it.
* **Allocation noise** is the spread of the estimate *across* plausible
  continuations. More seasons do not shrink it at all.

| tier | deltas by seed | between-alloc SD | within-alloc SE | SD/effect | sign stable | seasons help? |
|:--|:--|--:|--:|--:|:--|:--|
| `marginal_starter` | [0.02675, 0.02925, 0.035] | 0.00423 | 0.00492 | 0.140 | True | yes |
| `strong_starter` | [0.2385, 0.24975, 0.2425] | 0.00570 | 0.00729 | 0.023 | True | yes |
| `elite` | [0.54, 0.55775, 0.547] | 0.00894 | 0.00802 | 0.016 | True | no |

`elite` trips `dominated_by_allocation`: its allocation spread (0.0089)
exceeds its season SE (0.0080), so buying more seasons is wasted effort
there. It does **not** trip `dominates_effect`: 0.0089 against a 0.547
effect is 1.6%. Those are different questions and conflating them would
reject an effect sixty times larger than its own instability. Sign is stable
across every seed for all three tiers.

## Verdict and operational sampling policy

**GO** — effects resolve across tiers at a practical sample size

Derived from the table above, not from an example policy:

| player class | evidence | policy |
|:--|:--|:--|
| weekly improvement ~0 (`bench`) | 0/4 resolved, |d|/SE <= 1.6, true effect ~0 | **proxy/market only.** Never spend CE seasons. |
| improvement 1-2 pts (`marginal`) | 4/4 resolved, |d|/SE 2.5-5.4 | **4,000-season audit** when nominated; label unresolved cells honestly. |
| improvement 8+ pts (`strong`) | 4/4 resolved, |d|/SE 14-33 | **4,000-season audit**, precomputed between nominations. |
| improvement 18+ pts (`elite`) | 4/4 resolved, |d|/SE 47-69 | **4,000 seasons is already 30x more than needed.** Spend the budget on more allocation seeds instead. |
| near a high-dollar bid frontier | ladder rows adjacent in CE | larger offline confirmation, or `UNDERPOWERED_AT_CAP` |

Runtime is ~7s per tier x context cell at 4,000 seasons (~27.5s per tier across all four contexts).

### The 10-second timer

None of this runs on the clock. A single audited cell is ~7s and a four-price
ladder is minutes; the live path remains the cached lookup measured at 22ms
in `docs/TACTICAL_LAYER.md`. What this experiment changes is *which* players
are worth precomputing at all: spending the offline budget on bench players
buys nothing, and spending it on elite players past ~1,000 seasons buys
almost nothing either.
