# Frontier / decomposition reconciliation

> SIMULATED balanced mid-auction state (`44209a94d76fce67`). No league sale has ever been recorded.

## The original cause

At `p=$80, q=$30` the RB ladder reported `+0.01298` (favorable) and the
five-branch decomposition reported `-0.00366`. Tracing both paths found
**three** different completion searches running for one world:

```
frontier buy arm    completions=<nested pre-pass set>
frontier pass arm   (none) -> independent beam
decompose, all 5    (none) -> independent beam
```

The ladder's own buy and pass arms already disagreed with each other, so at
least two of the three numbers described a different auction. This was never
a modelling disagreement to adjudicate; it was three answers to three
different questions.

## The structural correction

`EvalContext` carries everything needed to evaluate one candidate at one
price -- state, cast, board, candidate, `p`, `q`, pass rule, recipient, cost
book, market, scenario, completion settings, allocation rotation, both
sample seeds, and the accumulated opportunity set -- and hands the same offer
to every branch:

| branch | holds candidate | offer source |
|:--|:--|:--|
| `UF`, `UP` | yes | `with_candidate` union, filtered by that branch's budget |
| `W`, `RF`, `RP` | no | `without_candidate` set (our budget is untouched in all three) |

Filtering rather than re-searching is what preserves cross-price nesting.
`evaluate_branch` raises `IndependentSearchRefused` if a consumer holding a
context searches anyway; an unaffordable branch raises `ContextMismatch`
rather than quietly starting a new beam; `assert_matches` refuses two
contexts that are not the same world and names the differing field.

The ladder's delta and the decomposition's total are now **two readings of
the same five branch CEs**, so they agree by construction and the residual
proves it on every draw.

### A second bug the correction exposed

The first reconciled run fed the context the price-`p` slice of the nested
ladder. `UF` pays nothing, so it inherited `UP`'s affordability cut and the
possession effect moved with a price it does not depend on:

```
           possession (wrong)   possession (fixed)
  p=$65         -0.09480             -0.09480
  p=$80         -0.11473             -0.09480
  p=$100        -0.12257             -0.09480
```

The context now takes the ladder **union**, and
`possession_offer_is_price_independent()` is a structural guard with a test
that plants a too-expensive construction and checks the guard fires.

## Per-draw agreement

Agreement is checked at the allocation-draw level, never after averaging
eleven rotations -- averaging can hide two paths that disagree per draw and
land near each other.

| candidate | p | q | frontier delta | decomposition total | max per-draw residual | all draws agree |
|:--|--:|--:|--:|--:|--:|:--|
| RB | 65 | 30 | -0.08161 | -0.08161 | 1.4e-17 | True |
| RB | 80 | 30 | -0.10118 | -0.10118 | 1.4e-17 | True |
| RB | 100 | 30 | -0.10939 | -0.10939 | 1.4e-17 | True |
| QB | 40 | 35 | -0.01839 | -0.01839 | 0.0e+00 | True |
| QB | 65 | 35 | -0.08182 | -0.08182 | 1.4e-17 | True |

Tolerance `1e-12`. Observed maximum `1.4e-17` — same context, same offers,
same seeds means the two paths run byte-identical simulations, so the
residual is floating-point noise, not modelling slack.

Per-draw world identity is recorded for every branch: focus roster
fingerprint, joint-world fingerprint, allocation fingerprint, rival rosters,
dollars spent, remaining pool, league CE sum, conservation flag, and how many
completions were offered.

## Corrected RB results (recipient Team11, q=$30, FIXED_MARKET)

| p | q | our payment | possession | denial | rival payment | total | 95% CI | verdict |
|--:|--:|--:|--:|--:|--:|--:|:--|:--|
| 65 | 30 | +0.00000 | -0.09480 | +0.01341 | -0.00023 | -0.08161 | [-0.08596, -0.07727] | unfavorable |
| 80 | 30 | -0.01957 | -0.09480 | +0.01341 | -0.00023 | -0.10118 | [-0.10614, -0.09623] | unfavorable |
| 100 | 30 | -0.02777 | -0.09480 | +0.01341 | -0.00023 | -0.10939 | [-0.11460, -0.10417] | unfavorable |

## QB consistency check (recipient Team07, q=$35, FIXED_MARKET)

| p | q | our payment | possession | denial | rival payment | total | 95% CI | verdict |
|--:|--:|--:|--:|--:|--:|--:|:--|:--|
| 40 | 35 | +0.00000 | -0.03277 | +0.01723 | -0.00284 | -0.01839 | [-0.02241, -0.01436] | unfavorable |
| 65 | 35 | -0.06343 | -0.03277 | +0.01723 | -0.00284 | -0.08182 | [-0.08542, -0.07821] | unfavorable |

## The bracket is WITHDRAWN

**The previous RB bracket (80, 100) does not survive reconciliation.** Under
the shared context the RB is unfavorable at $65, $80 and $100 — the lowest
price tested here is already past the crossing. The old lower edge of $80
was an artifact of the ladder's buy arm searching a different completion set
from its own pass arm.

The same happens to the QB: unfavorable at both $40 and $65, where the old
ladder called $40 favorable.

**No bracket is quoted.** The crossing for both candidates lies below the
prices tested here, and finding it needs a re-run whose ladder starts low.
Calling any of these an exact maximum bid would repeat exactly the mistake
this branch exists to remove.

## A caveat that limits the absolute values

The possession effect is **negative** for both candidates under the shared
context (RB `-0.09480`, QB `-0.03277`), where the pre-reconciliation
decomposition had it positive. That is not obviously wrong — being forced to
roster a player can cost you if the offer without him is better — but the
`with_candidate` union here was generated over the tested prices only
(`$65..$100` for the RB), so it never contains constructions that are
affordable only at low prices. **The union's width is a caller-side choice,
not a defect in `EvalContext`, and it bounds how negative possession can
look.** A union built from `$1` upward is the next run.

## Pass semantics

Every row prints its rule and its `q`. The sweeps above are **FIXED_MARKET**
with `q` taken from the standing bid: `p` moves while `q` does not. That is
named honestly rather than called STOP_NOW, which is the single point
`p = q + increment`. `RIVAL_OUTBIDS` (`q = p + 1`) remains a third rule and
is not mixed with either. This branch does not choose which rule is
behaviorally correct; it only makes evaluation internally consistent under a
declared one.

## Runtime: 732s for 5 reconciled prices at K=11 x 4,000 holdout seasons (~130s per price, 5 branches each).

## Verdict: PARTIAL GO

The structural objective is met: the two paths cannot diverge, agreement is
exact per draw, telescoping holds, conservation and nesting hold, and the
hostile tests refuse mismatched offers, seeds, recipients, rules and prices.

It is not a full GO because the reconciled numbers moved far enough to
withdraw both brackets, and the possession sign depends on a union width
this run did not vary. Nothing here is quotable as a maximum bid yet.
