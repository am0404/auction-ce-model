# Allocation ensembles and opening symmetry

`ce-lab tactical allocation-ensemble`. Real board, sanitized aggregates only.
Runtime 328s. Player-level detail is in
`local_data/tactical/allocation_ensemble.json`, which is gitignored.

## What the old jitter actually was

Audited on the real board before changing anything, over the 180 allocations
of one future auction:

```
jitter sigma                                    0.06 (relative, multiplicative
                                                on WILLINGNESS, not a tie-break)
exact ties in raw willingness                   152 / 180  (84.4%)
winner changed by jitter                        130 / 180  (72.2%)
  ...of which overturned a genuine non-tied gap    8 / 180   (4.4%)
mean willingness gap overturned                 $0.07
clearing price changed                           92 / 180  (51.1%)
winner identical to a no-jitter run              50 / 180  (27.8%)
```

Two things follow. First, **most of what the jitter did was legitimate work**
— 84% of allocations were genuine ties, because twelve identical owners in an
empty room have identical willingness, and something must break them. Second,
**it was the wrong instrument**: a 6% shock on willingness that overturned
real gaps 4.4% of the time, and — the fatal part — it was indexed by each
owner's *position in a tuple*, so at a fixed seed one team drew the same noise
column at every allocation. That is a persistent advantage earned by list
order, and it is why Team02 and Team03 finished 3.6 SE apart.

## The two mechanisms, now separate

**Mechanical tie-break.** Highest willingness wins outright. Only bids within
`tie_tolerance` ($0.50) of the best count as tied, and among those the lowest
*priority slot* wins. It cannot promote a bid that is genuinely behind. No
randomness is involved at all.

**Preference shock.** An explicit, scenario-labelled sigma representing
unknown manager preferences. Zero by default, forced to zero in symmetry
tests, indexed by priority slot so it permutes with the ensemble instead of
adhering to one team, and carried in every fingerprint. Modelling that
managers differ is legitimate; hiding it inside a sort was not.

**Exchangeable schedule.** `balanced_schedule` rotates the eleven rivals
through the priority order; each occupies each slot exactly once in eleven
draws. The focus team keeps its slot — it is not exchangeable with a rival,
because it is the team being measured.

## Opening symmetry: restored

```
                          before (1 seed)     after (ensemble, K=15)
Team02 vs Team03 gap      +0.01175            mean +0.00187
                          3.6 SE              CI95 [-0.00726, +0.01099]
single-seed gap now       +0.00325
max |paired difference|   0.03275
contains zero             True
persistent label effect   False
```

The interval contains zero and no persistent owner-label effect remains.
Individual draws still differ — future auctions genuinely diverge — but the
*estimator* now treats identical owners identically.

## Primary real candidate (RB, the one that flipped sign)

Previously three arbitrary seeds gave `[+0.0645, -0.02875, +0.03025]` — a sign
flip. Over the balanced ensemble:

```
unique allocation draws        11  (rotations 0-10)
deltas                         [-0.0905, -0.10125, -0.09225, -0.04625, -0.06475, -0.07375, -0.0335, -0.0345, -0.0445, -0.05575, -0.069]
mean                           -0.06418
median                         -0.06475
min / max                      -0.10125 / -0.03350
between-allocation SD          0.02361
RMS within-allocation SE       0.00873
ratio between/within           2.70
SE of the ensemble mean        0.00712   (cluster-level: SD / sqrt(11))
95% t-interval for the mean    [-0.08004, -0.04832]   (df=10)
sign frequency positive        0.00   (11 of 11 negative)
```

**The sign instability was the shock, not the economics.** Every one of the
eleven exchangeable draws is negative. Buying this back at $44 while the
named recipient would pay $55 for him is unfavorable, consistently.

The two uncertainties are reported side by side and never added. Between-
allocation SD (0.0236) is 2.7x the within-allocation season SE (0.0087):
**more seasons would not help; more allocation draws would.** Pooling all
11 x 4,000 seasons as one flat sample would have produced an interval roughly
sixty times too narrow, because draws are clusters.

## Convergence by K

| K | mean | between SD | SE(mean) | CI95 | sign+ | verdict |
|--:|--:|--:|--:|:--|--:|:--|
| 1 | -0.09050 | 0.00000 | n/a | n/a | 0.00 | unresolved (k<2: no between-allocation interval exists) |
| 3 | -0.09467 | 0.00577 | 0.00333 | [-0.10900, -0.08034] | 0.00 | unfavorable |
| 6 | -0.07812 | 0.02050 | 0.00837 | [-0.09964, -0.05661] | 0.00 | unfavorable |
| 12 | -0.06638 | 0.02375 | 0.00686 | [-0.08147, -0.05128] | 0.00 | unfavorable |
| 15 | -0.06908 | 0.02448 | 0.00632 | [-0.08264, -0.05553] | 0.00 | unfavorable |

Read this carefully. **K=1 has no interval at all** — one future auction
cannot report allocation uncertainty about itself, which is precisely the
mistake the pilot made. The mean moves from -0.0905 at K=1 to -0.0642 at
K=11, a 29% shift, and the between-allocation SD only reaches its true size
once several rotations have been seen.

**K=11 is the right number here, and K=15 was wasted work.** With the
preference shock off the allocation is a function of the rotation alone, so
eleven rotations exhaust the balanced set and draws 12-15 reproduced draws
1-4 exactly. The run reported 15 distinct board fingerprints because those
hash the seed; there were 11 distinct joint worlds. `effective_k` and
`redundant_draws` now report this, and a K=15 interval built on 11 real draws
would have been narrower than its coverage.

**Required K: 11** (one full rotation set) with the shock off. With a
preference shock switched on, more draws would carry new information and K
should rise.

## Position summaries

| position | K | mean delta | between SD | CI95 | verdict |
|:--|--:|--:|--:|:--|:--|
| RB (primary) | 11 eff. | -0.06418 | 0.02361 | [-0.08004, -0.04832] | unfavorable |
| QB | 6 | +0.15596 | 0.00576 | [+0.14991, +0.16201] | favorable |
| WR | 6 | +0.07700 | 0.03213 | [+0.04327, +0.11073] | favorable |
| TE | - | - | - | - | PROXY ONLY -- retained; no CE seasons spent |

QB is the most stable effect on the board (between-SD 0.0058 against a mean
of +0.156 — 3.7%) and strongly favorable, consistent with the superflex seat
being open. WR is favorable but noticeably less stable. TE remains proxy-only
on its 0.22 lineup improvement; no CE seasons were spent on it.

## Price ladders

Status: **FRONTIER_NOT_REACHED**. No ladder is run here.
The pilot's ladders never changed the allocation across their tested range,
so no maximum bid may be quoted from them. That is a fact about an empty room
where money is not scarce, not about price insensitivity, and it is not the
same problem as allocation instability.

## Conservation

Every draw: conservation passed, no unpaid blocking, no duplicate ownership,
league CE summed to exactly 1.0. 11 distinct joint worlds, all validated.

## Verdict: GO for targeted real-player precomputation

Against the stated gate:

* Opening symmetry restored — interval contains zero, no persistent label
  effect. **Yes.**
* No systematic owner-ID effect — rotation removes it by construction, and
  the balanced-win-count test asserts it. **Yes.**
* Ensemble mean materially more stable than single-seed — single seeds gave
  a sign flip; eleven exchangeable draws give 11/11 one sign. **Yes.**
* Meaningful effects classifiable at a practical K — K=11 at ~7.3s per draw
  is ~80s per candidate per arm-pair. **Yes.**
* Joint-world invariants pass on every draw. **Yes.**

The earlier NO-GO was correct at the time and is now discharged. What
remains is not instability but honest, irreducible allocation uncertainty,
and it must be reported as an interval over futures rather than buried inside
a single CE confidence interval.
