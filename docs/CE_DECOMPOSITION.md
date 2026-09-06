# Tactical CE, decomposed

`ce-lab tactical decompose`. Real board, sanitized position-level
aggregates. K=11, 4,000-season
holdouts. Player-level detail is in `local_data/tactical/decomposition.json`,
gitignored.

## The question

The converged diagnostic found positive weekly lineup improvement for all
four audited candidates. Two had negative tactical CE. **Improving our lineup
is not the same as being worth buying at the market price**, above all when
passing may make an opponent overpay. This measures which.

## Five branches, three of them analytical

| branch | holder | pays | real? |
|:--|:--|:--|:--|
| `W` | none | none | **analytical only** |
| `UF` | focus | zero | **analytical only** |
| `UP` | focus | price | real |
| `RF` | rival | zero | **analytical only** |
| `RP` | rival | price | real |

`W`, `UF` and `RF` are counterfactual instruments. A branch in which a rival
is handed a player for nothing has no probability, and giving it one would
corrupt every weighted summary downstream. `assert_not_a_recipient` refuses.

## Declared telescoping path

```
CE(UP) - CE(RP)
  = [CE(UP) - CE(UF)]   our payment effect
  + [CE(UF) - CE(W) ]   our possession effect
  + [CE(W)  - CE(RF)]   rival possession / denial effect
  + [CE(RF) - CE(RP)]   rival payment effect
```

It sums exactly because it telescopes, **not** because roster completion is
additive. Completion is nonlinear, so a different ordering attributes
different amounts to each term. This is a declared path, not a unique causal
decomposition.

## Components by position

| pos | improve | $p | rival $q | our payment | our possession | denial | rival payment | total | residual | class |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|:--|
| RB | +2.50 | 25 | 30 | -0.09089 | +0.05139 | +0.02648 | -0.01752 | -0.03054 | -0.0e+00 | helps us but costs too much |
| TE | +1.32 | 6 | 7 | -0.07398 | +0.00809 | -0.00141 | +0.00016 | -0.06714 | 0.0e+00 | helps us but costs too much |
| QB | +1.46 | 26 | 37 | -0.09148 | +0.18641 | +0.04227 | +0.02032 | +0.15752 | -0.0e+00 | mixed |
| WR | +1.76 | 32 | 41 | -0.08786 | +0.19204 | +0.01475 | -0.03830 | +0.08064 | 0.0e+00 | mixed |

### Uncertainty (between-allocation SD, and the ensemble interval)

| pos | component | mean | between-SD | CI95 | sign+ |
|:--|:--|--:|--:|:--|--:|
| RB | our_payment | -0.09089 | 0.00846 | [-0.09657, -0.08520] | 0.00 |
| RB | our_possession | +0.05139 | 0.00778 | [+0.04616, +0.05661] | 1.00 |
| RB | rival_denial | +0.02648 | 0.01229 | [+0.01822, +0.03474] | 1.00 |
| RB | rival_payment | -0.01752 | 0.02168 | [-0.03209, -0.00296] | 0.27 |
| RB | **total** | -0.03054 | 0.01975 | [-0.04381, -0.01728] | 0.09 |
| TE | our_payment | -0.07398 | 0.00646 | [-0.07832, -0.06964] | 0.00 |
| TE | our_possession | +0.00809 | 0.00793 | [+0.00277, +0.01342] | 0.91 |
| TE | rival_denial | -0.00141 | 0.01373 | [-0.01063, +0.00781] | 0.27 |
| TE | rival_payment | +0.00016 | 0.01145 | [-0.00753, +0.00785] | 0.46 |
| TE | **total** | -0.06714 | 0.01726 | [-0.07873, -0.05554] | 0.00 |
| QB | our_payment | -0.09148 | 0.00617 | [-0.09562, -0.08733] | 0.00 |
| QB | our_possession | +0.18641 | 0.00473 | [+0.18323, +0.18959] | 1.00 |
| QB | rival_denial | +0.04227 | 0.01636 | [+0.03129, +0.05326] | 1.00 |
| QB | rival_payment | +0.02032 | 0.01985 | [+0.00698, +0.03365] | 0.73 |
| QB | **total** | +0.15752 | 0.00600 | [+0.15349, +0.16156] | 1.00 |
| WR | our_payment | -0.08786 | 0.00632 | [-0.09211, -0.08362] | 0.00 |
| WR | our_possession | +0.19204 | 0.00569 | [+0.18823, +0.19587] | 1.00 |
| WR | rival_denial | +0.01475 | 0.01330 | [+0.00581, +0.02369] | 0.91 |
| WR | rival_payment | -0.03830 | 0.03277 | [-0.06031, -0.01628] | 0.09 |
| WR | **total** | +0.08064 | 0.03199 | [+0.05915, +0.10212] | 1.00 |

Within-allocation season SE is reported per branch in the JSON and is never
pooled with between-allocation SD.

## Named recipients

| pos | rival | $q | denial | rival payment | total | class |
|:--|:--|--:|--:|--:|--:|:--|
| RB | Team12 | 30 | +0.02648 | -0.01752 | -0.03054 | helps us but costs too much |
| TE | Team12 | 7 | -0.00141 | +0.00016 | -0.06714 | helps us but costs too much |
| QB | Team12 | 37 | +0.04227 | +0.02032 | +0.15752 | mixed |
| WR | Team12 | 41 | +0.01475 | -0.03830 | +0.08064 | mixed |
| RB | Team02 | 30 | +0.02534 | -0.01764 | -0.03179 | helps us but costs too much |
| TE | Team02 | 7 | -0.00291 | +0.00332 | -0.06548 | helps us but costs too much |

Recipients are decomposed separately and never averaged first.

Runtime 1384s.

## What this actually shows

**The proxy selector is not wrong.** Own possession is positive for all four
candidates, so the proxy correctly identifies players who improve the roster we
can build. Championship equity then rejects two of them **on price**, which is
a different judgement and the correct one to make separately.

```
            improve   own possession   own payment    total
QB           +1.46         +0.18641      -0.09148   +0.15752
WR           +1.76         +0.19204      -0.08786   +0.08064
RB           +2.50         +0.05139      -0.09089   -0.03054
TE           +1.32         +0.00809      -0.07398   -0.06714
```

For RB and TE the classification is **"helps us but costs too much"**: own
possession is positive and own payment more than offsets it. That is the
answer, and it is not selector failure.

**Own payment is ~-0.09 for everyone.** It barely varies with price ($6 to
$32), which is worth stating plainly: at the empty-room state, spending
*anything* costs roughly the same in championship equity because the money's
alternative use is a whole roster, not a marginal upgrade. Price discrimination
between candidates therefore comes almost entirely from the possession side.

**Weekly points convert to equity at wildly different rates.** The TE gains
+1.32 weekly points and only +0.008 CE; the WR gains +1.76 and +0.192 — twenty
times more equity per point. The proxy is measuring the right thing and
measuring it in the wrong units for ranking. That is the strongest argument in
this document for a separate tactical-priority signal, and it is about
conversion rate, not about denial.

**Denial and rival payment are small here, as they should be.** Denial peaks at
+0.042 (QB) and rival payment at +0.020. In an *empty room* every rival is
structurally identical, so which one receives the player barely matters. This
is a property of the state, not a finding about denial in general, and it is
exactly why recipient identity must be re-measured mid-auction.

## Named recipients

Recipient identity moves the total by roughly 0.001-0.002 at this state:

```
RB   Team12 @$30  denial +0.02648  rival pay -0.01752  total -0.03054
RB   Team02 @$30  denial +0.02534  rival pay -0.01764  total -0.03179
TE   Team12 @$6   denial -0.00141  rival pay +0.00016  total -0.06714
TE   Team02 @$7   denial -0.00291  rival pay +0.00332  total -0.06548
```

Small, real, and never averaged before decomposition. The TE's denial is
*negative* against one recipient and more negative against the other — letting
that rival have him is mildly good for us.

## Telescoping

Residual is **exactly 0.0** per draw, at the ensemble mean, and in serialized
output, for every position and every recipient. It sums because it telescopes.
A different ordering would move the labels; the total is invariant.

## Selector recommendation: KEEP THE PROXY, ADD A PRIORITY SIGNAL LATER

* **Keep the proxy selector unchanged.** It identifies roster improvement
  correctly; CE rejects on price correctly. Nothing here supports changing
  completion ranking.
* **Do not add a denial coefficient.** Denial is measurable but small at this
  state, and fitting a coefficient from four players would be inventing a
  number.
* **A separate tactical-priority signal is justified but not yet buildable.**
  The evidence is the conversion-rate spread (0.008 vs 0.192 CE per comparable
  weekly gain), not denial. Building it needs more than four candidates.

## Verdict: GO for real price-frontier testing

* Components telescope — residual exactly 0. **Yes.**
* Joint-world validation passes on every branch and draw; league CE sums to
  1.0 in all five branches. **Yes.**
* The RB/TE disagreement has a measured explanation. **Yes** — own payment
  exceeds own possession, in both cases.
* The selector's role is separated from total tactical value. **Yes.**
* Named recipient differences preserved, never averaged first. **Yes.**
* Allocation uncertainty reported separately from season uncertainty. **Yes.**

## Limitations

* Empty-room state only. Denial and recipient identity are structurally small
  here and will not be at a mid-auction state.
* Four candidates, one price each. Own payment's near-constancy across $6-$32
  is a finding at this state, not a law.
* The telescoping path is declared, not unique. An alternate ordering was not
  computed; the total is invariant but the attribution is not.
* Contingency, handcuff and QB-insurance value remain unpriced.
