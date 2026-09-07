# ROUGH CE — NO-GO

The draft-day rough CE approximation was built, benchmarked, and **failed its
calibration gate**. It is not wired into the dashboard and must not be used
tonight. `draft-day-sleeper-sync` remains the shipped branch.

## What was built

`src/ceauction/draftday/roughce.py`. Real CE simulation throughout: real
`PlayerSpec`s, real scoring and lineup eligibility, real median-matchup regular
season, real byes and bracket. No regression from lineup points to CE, no
invented dollar-to-CE coefficient, and the certified solver is not in the path.
Both arms are completed by the existing bounded completion search over the
actual `AuctionState`, and every world is the existing reconciled joint world,
so duplicate ownership, exact budgets, the $1-per-open-slot reserve and roster
legality are enforced by the same code the certified path uses.

The speed came from one observation, not from cutting an invariant:
**generating the player-week draws was 82% of the cost of an evaluation, and
those draws do not depend on the rosters at all.** They are also keyed by
player identity, so a player's draws are byte-identical whether the generator
is handed 180 players or all 549 (measured). So they are generated once and
reused by every arm and every future, which is also what makes the buy/pass
comparison exactly paired.

Precompute: **0.8s** for 1200 seasons × 549 players × 17 weeks (101 MB, under
`local_data/roughce/`, gitignored). Live query: **3.1–5.4s** for k=3 futures,
inside the ≤10s band and near the ≤5s target. Conservation held in every world
evaluated (`league CE sum = 1.0000` throughout).

So the latency gate passed and the conservation gate passed. The calibration
gate did not.

## The calibration gate

Certified reference: the conditional QB case at `certified-completion-solver`
@ `7096f19` — simulated `balanced` mid-auction, candidate `3655628085077906594`,
recipient `Team07`, standing pass price `$35`, legal max `$128`. The refinement
records the frontier at **$67–$72** (pointwise 68/69, simultaneous 67/72),
which implies favorable at $36/$40/$65 and unfavorable at $80/$100/$128.

The rebuilt state reproduces the recorded economics **exactly** — Team01
`budget 139 / spent 61 / 12 open / 3 players`, league total spent `1488` — so
the fixture is the right auction. Its fingerprint is `e9680f507350972a` against
the recorded `44209a94d76fce67`; the sales and budgets being identical means
the fingerprint's inputs changed since 2026-09-06, not the state.

| price | certified | rough (beam 16) | rough (beam 96) |
|---:|---|---|---|
| $36 | favorable | **-0.00667 unfavorable ✗** | +0.02111 favorable ✓ |
| $40 | favorable | -0.01472 unfavorable ✗ | — |
| $65 | favorable | **-0.03444 unfavorable ✗** | **-0.04056 unfavorable ✗** |
| $80 | unfavorable | -0.05111 unfavorable ✓ | -0.05833 unfavorable ✓ |
| $100 | unfavorable | -0.06750 unfavorable ✓ | -0.07361 unfavorable ✓ |
| $128 | unfavorable | -0.06944 unfavorable ✓ | — |

Acceptance required the sign at $36, $65, $80 and $100, and a crossing inside
or overlapping ($65, $80).

- At the rough beam: **3 of 6 signs wrong**, and *no crossing exists at all* —
  every price is unfavorable, so there is no bracket to compare.
- At a 6× wider beam: $36 recovers, but **$65 still does not**, and the
  crossing lands somewhere in ($36, $65) — which does **not** overlap the
  certified ($65, $80).

Both fail. **NO-GO.**

## Why, and why it is not a tuning problem

Widening the beam from 16 to 96 moved the $36 estimate from `-0.00667` to
`+0.02111`. That is a **sign flip caused by search width alone**, at a price
$30 below the certified frontier.

This is the already-recorded `QB_UNION_UNDERCONVERGED` finding reappearing:
tripling the search effort on this candidate previously grew its union from 26
to 72 constructions and moved its proxy by `+0.690`, far above the `0.25`
tolerance, which is why the certified frontier conclusion for this QB was
withheld once already. The one calibration case available is precisely the case
the project has documented as search-sensitive.

A rough tool whose buy/pass *sign* depends on how wide the beam was cannot give
a direction near the frontier — and near the frontier is the only place anyone
would consult it. Widening the beam until this one case matches would be
fitting the tool to its single calibration point, which is what the sign
disagreement is evidence against, not a recipe for fixing it.

## Status

- `roughce.py` is committed for the record. **Nothing is wired to the
  dashboard**; there is no RUN ROUGH CE button and no server change.
- Tonight uses `draft-day-sleeper-sync` @ `dbb653d`: Sleeper sale tracking plus
  manual entry and undo, with the market guardrail as the only price display.
- One calibration case is not proof of universal accuracy in either direction.
  This is evidence that the approximation does not reproduce the one certified
  result available, not a measurement of how wrong it is in general.
