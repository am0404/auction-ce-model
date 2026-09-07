# The certified completion solver

> **SIMULATED MID-AUCTION STATE.** Every auction history in the real-price
> section is invented from the market prior for frontier testing. It is NOT
> observed, NOT recorded, NOT historical and NOT calibrated. **No league sale
> has ever been recorded. No maximum bid is quoted in this document.**

## Verdict

**`CERTIFIED_AND_PRACTICAL`.**

The focus-team completion problem is solved to a proven optimality gap in
**seconds per price** on the real board, and the championship-equity frontier
has been run on the certified sets. `QB_UNION_UNDERCONVERGED` is resolved: the
objective is no longer merely stable under more effort, it is **bounded**.

The result is a **`CERTIFIED CONDITIONAL RESERVATION PRICE`**, and every word of
that label is load-bearing. §7 states exactly what it applies to and, more
importantly, what it does not. It is **not** an opening max bid, not a player
value, and not applicable to the live room.

| | |
|---|---|
| solver vs exhaustive enumeration | 100/100 fixtures exact |
| worst real-board optimality gap | **0.0496** weekly points (bound 0.10) |
| certified gain over the best heuristic | up to **+3.55** weekly points |
| 12 → 24 candidate-set expansion | **`CE_CANDIDATE_SET_STABLE`** |
| frontier vs decomposition, every draw | agree to **≤1.4e-17** |
| coarse frontier | `RESERVATION_BRACKET`, favorable ≤$65, unfavorable ≥$80 |
| integer refinement $66–$79 | 14/14 prices, no invariant failure |
| **conditional reservation price, pointwise** | **$68** |
| **conditional reservation price, simultaneous** | **$67** (with $68–$71 unresolved) |

## 1. What was actually blocking

`QB_UNION_CONVERGENCE.md` measured the completion objective moving **3.8 weekly
points** with beam width, *non-monotonically*: beam 64 scored below beam 32, beam
384 below beam 96. `REPAIRED_SEARCH.md` added one-swap repair, `TWO_SWAP_EXCHANGE.md`
added a two-for-two exchange, and each found further improvement — which the
convergence gate correctly read as *failure*, since a search that still improves
when asked harder has not converged.

Every one of those results shares a single missing ingredient. None of them had
an **upper bound**. Without one, "the search stopped" and "the search finished"
are indistinguishable, and no amount of additional effort can tell them apart.
That is why widening the beam could not work and why the brief was right to
forbid more of it.

This module supplies the bound.

## 2. The formulation

Write `R` for a roster and `f` for the objective the auction layer already uses,
`ProxyEvaluator.strength`:

> `f(R)` = mean, over scenarios `s = (availability replicate r, scoring week w)`,
> of the maximum-weight legal starting lineup drawable from `R` ∩ `A_s`,
> scored with the pregame projection `π[s, p]`.

### 2.1 The scenario weights do not depend on the roster

The brief asks this to be established before anything is called an optimisation
over `R`, and it holds:

* availability is drawn **coordinate-addressed by player id**, so a player's
  byes and injuries are identical on every roster he could join;
* the contingency uplift `_contingency_bonus(pool, available)` reads the **whole
  pool's** availability, not the roster's, so `π[s, p]` is the same number
  whichever roster `p` sits on. (A player's depth-chart superior need not be
  rostered for the uplift to fire. That is a modelling choice made upstream; what
  matters here is that it is roster-independent.)

So `π` and `A` are constants of the problem, and the only decision variables are
the roster indicators. There is no conditional effect left to isolate.

### 2.2 The lineup rule is a polymatroid

`lineup_vec.select_lineups_mask` accepts a starter set exactly when its position
counts `(n_Q, n_R, n_T)` — quarterbacks, running backs, and receivers-and-tight-ends
together, because the league has **no dedicated TE slot** — satisfy

```
n_Q ≤ 2   n_R ≤ 4   n_T ≤ 5
n_Q+n_R ≤ 5   n_Q+n_T ≤ 6   n_R+n_T ≤ 7
n_Q+n_R+n_T ≤ 8
```

Those seven right-hand sides form a set function `cap` on `{Q,R,T}` with
`cap(∅)=0`. It is monotone, and it is submodular — the two tight checks being
`cap(QR)+cap(RT) = 12 = cap(QRT)+cap(R)` and `cap(QT)+cap(RT) = 13 = cap(QRT)+cap(T)`.
A monotone submodular `cap` is a **polymatroid rank function**, which gives two
things at once:

1. the family of legal lineups is a **matroid**, so the greedy that
   `select_lineups_mask` runs is exactly optimal (this was already assumed; it is
   now checked by a test);
2. the lineup polytope is **integral**, so the inner maximisation can be written
   as a linear program with no integer variables.

`cap_is_polymatroid()` verifies monotonicity and submodularity over all 64 subset
pairs, and a second test drives `select_lineups_mask` with every count triple to
confirm the seven constants are the rule the shipped code actually applies.

### 2.3 The feasible set

Binary `x_p` per selectable board player; owned players and any forced candidate
pinned to 1.

| constraint | form |
|---|---|
| exact roster size | `Σ x_p = K` (open slots) |
| budget | `Σ c_p x_p ≤ B` |
| `$1`-per-open-slot reserve | **implied**, see below |
| QB slot | `#QB ≥ 1` |
| RB1, RB2 | `#RB ≥ 2` |
| WT1–WT3 | `#WR + #TE ≥ 3` |
| six non-QB starting places | `#RB + #WR + #TE ≥ 6` |
| all eight slots | implied (`roster_size` > `n_starters`) |

These are the saturating Hall constraints from `feasibility.py`, and they are
*counting* constraints only because the slot eligibility sets are laminar. There
is no quarterback maximum and no tight-end quota, because the league has neither.
Bench players are simply the roster members no scenario starts; they earn their
place through conditional substitution and nothing else.

**The reserve is implied, not dropped.** The beam enforces
`spend_j + (K−j)·min_bid ≤ B` after each of `K` purchases. Every remaining player
costs at least `min_bid` (`CostBook.minimum_cost = 1` is a league rule), so that
quantity is bounded by the final total spend and the single budget row implies all
`K` of them. `reserve_is_implied()` checks the premise rather than assuming it, and
`solve_completion_exact` **refuses to run** when it fails rather than quietly
relaxing a constraint.

### 2.4 Two solvers, on purpose

**The extended formulation** (`solve_completion_exact`, the one used for every
result below) writes the inner maximisation into the model with a starter variable
`z[s,p]` per scenario and player:

```
max  (1/|S|) Σ_{s,p} π[s,p] · z[s,p]
s.t. 0 ≤ z[s,p] ≤ x_p · a[s,p]
     Σ_{p ∈ A} z[s,p] ≤ cap(A)      for every A ⊆ {Q,R,T}, every scenario
     + the roster constraints of §2.3
```

Because the lineup polytope is integral (§2.2), the inner LP attains exactly the
greedy value for any fixed `x` **without** `z` being constrained to integers. The
only binaries are the roster choices, and the optimality gap is HiGHS's own dual
bound rather than something this module constructs.

**The cutting plane** (`solve_completion_certified`) is kept as an independent
check. `f` is monotone submodular — an average of weighted matroid rank functions —
so each evaluated roster yields a valid Nemhauser–Wolsey inequality and the master's
optimum is an upper bound. It is much slower (≈400 cuts on a nine-man fixture
against 0.03s for the MILP), and it is retained because two methods sharing only
the objective oracle are far better evidence than one.

> **A correctness note, recorded because the test caught it and not the author.**
> The first version of the cut used `ρ_j(S \ j)` for its drop term — the marginal
> against the incumbent, which is the natural-looking quantity. That is **invalid**:
> submodularity requires the *smallest* marginal, taken against the ground set, and
> using the larger one tightens the cut past validity. Fixture 60 duly reported an
> "upper bound" of 93.91 against a true optimum of 94.78. An upper bound that is
> occasionally not one is worth nothing, because a caller cannot tell which
> occasion they are on.

### 2.5 Declared numerical tolerance

`strength` and `strength_many` reduce over differently-shaped arrays and disagree
at about **3e-14** on a nine-man roster — float associativity, nothing else.
`OBJECTIVE_TOLERANCE = 1e-9` names it. Every certified claim below is quoted to
**0.10 weekly points**, twelve orders of magnitude above the noise.

## 3. Certification evidence

### 3.1 Against exhaustive enumeration

100 deterministic randomized fixtures, each small enough to enumerate every
legal completion and take the maximum. Every one is checked on the objective,
the roster size, the budget, distinctness, forced ownership, and that the weekly
lineup the objective implies is itself legal under availability and the seven
counting caps.

| check | result |
|---|---|
| solver optimum == brute-force optimum | 100/100, to `OBJECTIVE_TOLERANCE` |
| dual bound ≥ true optimum, every fixture | 100/100 |
| extended MILP == cutting plane == enumeration | 8/8 cross-checked |
| `cap` monotone and submodular | all 64 subset pairs |
| seven constants == `select_lineups_mask` | all count triples |

Fixture coverage is asserted rather than hoped for: ≥20 with a forced candidate,
≥20 with a knapsack-tight budget, at least one with multiple equal optima
(which needs genuinely interchangeable players — equal projections are not
enough, since availability is drawn per player id), and both a bench quarterback
and a superflex skill-position fallback among the winners.

### 3.2 What the heuristics were doing

Measured on the same fixtures, against the certified optimum:

* **greedy** by projection missed the optimum;
* the **shipped beam** (`_beam_search`, driven with the fixture's own board)
  missed it;
* **one-swap** local search was trapped below it.

An honest negative belongs here too. An *exhaustive* two-swap — every pair out
against every pair in, no shortlist, no round cap — reached the optimum on every
fixture, and on a further 120-fixture sweep at K=8 with knapsack-tight budgets
and a cheapest-first start. At four to eight free slots a full two-swap
neighbourhood is close to exhaustive search, so that is what one should expect,
and planting a trap at that size would mean rigging the pool until one appeared.

The two-swap that **ships** is not exhaustive: `localrepair` bounds it to a
40-player shortlist and the top 6 completions, on a real board of 77. That one
is trapped, and §4 measures by how much.

## 4. The real board: heuristic versus certified

Real QB candidate, SIMULATED balanced mid-auction state, pool depth 40,
`proxy_reps=16`. "beam / one-swap / two-swap" are the shipped searches run at
the control settings on this state.

| price | budget | beam | one-swap | two-swap | **certified** | upper bound | gap | gain | solve |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| UF (no purchase) | $139 | 95.4539 | 98.0303 | 98.0463 | **98.0464** | 98.0960 | 0.0496 | +0.0001 | 4.4s |
| $36 | $103 | 95.0811 | 95.7059 | 96.0978 | **97.9053** | 97.9487 | 0.0434 | **+1.8075** | 2.3s |
| $40 | $99 | 95.2746 | 95.2746 | 95.5696 | **97.7630** | 97.8109 | 0.0479 | **+2.1935** | 5.7s |
| $65 | $74 | 93.1409 | 93.1409 | 93.1409 | **95.6331** | 95.6807 | 0.0476 | **+2.4922** | 3.0s |
| $80 | $59 | 88.9284 | 88.9284 | 88.9284 | **92.4770** | 92.4770 | **0.0000** | **+3.5485** | 2.2s |
| $100 | $39 | 83.9447 | 83.9447 | 83.9447 | **87.2245** | 87.2285 | 0.0040 | **+3.2798** | 2.7s |
| $128 | $11 | 73.8052 | 73.8052 | 73.8052 | **74.4857** | 74.4857 | **0.0000** | +0.6805 | 0.2s |

Every gap is inside the 0.10 bound; the worst is 0.0496 and two prices are
certified at a **zero** gap. The certified roster is at least as good as every
heuristic roster at every price, which is the brief's implementation-defect
check and it passes.

**The single most useful line is `$80`.** `TWO_SWAP_EXCHANGE.md` recorded
`92.4770` there from a much larger union than the control settings reproduce,
and the certified optimum is `92.4770` — *the same number, at a zero gap*. The
two-swap answer at `$80` was optimal all along. What was missing was never a
better roster; it was the proof that no better roster existed. That is exactly
the thing a search without an upper bound cannot supply, and it is why
`QB_UNION_UNDERCONVERGED` was the correct verdict at the time rather than an
over-cautious one.

Note also how the heuristics degrade as the budget tightens: at `$65` and above,
beam, one-swap and two-swap all return the *same* roster, and all three are
2.5–3.5 points short. A tight budget is where local search has fewest legal
moves, which is precisely where the bound matters most and where a beam is least
able to tell you it is stuck.

## 5. The 12 → 24 candidate-set expansion

One proxy-optimal roster is not enough to hand to CE, because CE ranks outcome
*distributions*: two rosters a hundredth of a weekly point apart can differ
materially in variance. So the solver emits a set, and the set has to be shown
large enough.

The rule was **predeclared and fingerprinted (`7f183724afc05780`) before the
run**, with five clauses. The materiality threshold is derived rather than
picked: the coarse crossing spans `0.075` CE across `$15`, so `0.01` CE is about
two dollars of reservation price.

**One structural fact governs the design.** `build_joint_worlds` only builds
`max_worlds` completions into reconciled worlds, and `evaluate_joint_arm` then
chooses among *those*. At the `EvalContext` default of 2, a 12→24 expansion is
provably unable to change anything, and the test would report a stability it had
not earned. Both arms therefore run at `max_worlds=8`.

| | 12 | 24 |
|---|---:|---:|
| with-candidate completions | 84 | 168 |
| without-candidate completions | 12 | 24 |
| nested (12 ⊂ 24) | — | **yes**, asserted |
| `$65` mean delta | +0.02214 | +0.02214 |
| `$80` mean delta | −0.05259 | −0.05259 |
| selection changed | — | **0 of 11 draws**, both prices |
| worlds actually compared | 8 | 8 |
| paired per-draw shift | — | **0.000000** |
| max residual | 1.4e-17 | 1.4e-17 |

All five clauses pass: **`CE_CANDIDATE_SET_STABLE`**.

**The mechanism, stated so the result is not read as stronger than it is.** The
84 added completions never entered the top eight by proxy at any budget, so CE
was never offered them. This establishes that the expansion changes nothing at
`max_worlds=8`. It does **not** establish that CE would be indifferent to an
arbitrarily deeper set, nor that proxy rank and CE rank agree in general — only
that within the offer CE actually saw, enlarging the pool behind it moved
nothing.

## 6. The frontier

### 6.1 How the certified sets reach CE

Through an adapter and nothing else. The CE layer does not consume answers, it
consumes `Completion` objects held in an `EvalContext`, and `build_eval_context`
already accepts both opportunity sets — supplying them is what makes it search
zero times. The five-branch lattice, the joint world builder, the conservation
checks, the frontier and the decomposition are **untouched** and unaware the
completions came from a MILP.

Two sets, at the budgets `evalcontext` documents. `with_candidate` is generated
at our whole remaining budget — `UF` pays nothing for a candidate it already
holds — and unioned with a certified set at every evaluated price's own
post-purchase budget, so each price's optimum is present rather than inherited.
`without_candidate` reserves the candidate off the board at the full budget.

Every adapted completion is re-validated from scratch against the auction state:
ids, availability, roster size, cost against the cost book, budget, candidate
ownership, lineup feasibility, and the objective recomputed through
`ProxyEvaluator`. Any mismatch **raises**. A quietly smaller opportunity set is
the failure mode hardest to see afterwards, and dropping a bad completion to
keep going would produce exactly that.

### 6.2 The coarse ladder

Recipient **Team07**, `q=$35`, K=11 exchangeable allocation rotations, 800
selection seasons, 4,000 holdout seasons, shared `EvalContext` per draw.

| p | pass rule | mean delta | pointwise 95% CI | verdict | residual |
|---:|---|---:|---|---|---:|
| $36 | `STOP_NOW` | +0.07575 | [+0.06841, +0.08309] | favorable | 0.0e+00 |
| $40 | `FIXED_MARKET` | +0.06580 | [+0.06131, +0.07028] | favorable | 0.0e+00 |
| $65 | `FIXED_MARKET` | +0.02111 | [+0.01561, +0.02662] | favorable | 0.0e+00 |
| $80 | `FIXED_MARKET` | −0.05386 | [−0.05865, −0.04907] | unfavorable | 1.4e-17 |
| $100 | `FIXED_MARKET` | −0.12239 | [−0.12610, −0.11868] | unfavorable | 1.4e-17 |
| $128 | `FIXED_MARKET` | −0.14839 | [−0.15376, −0.14301] | unfavorable | 0.0e+00 |

`RESERVATION_BRACKET`, crossing between `$65` and `$80`. Frontier delta and
five-branch decomposition total agree to ≤1.4e-17 on **every** draw at **every**
price — they are two readings of the same five branch equities, so anything
larger would mean the branches had stopped sharing a world.

The intervals above are **pointwise**. Six of them are six 95% statements, so
the chance at least one is wrong is well above 5%; §6.3 pays for that properly.

## 7. The label, and its boundaries

Any number this document produces is a

> ### `CERTIFIED CONDITIONAL RESERVATION PRICE`

and it is conditional on **all** of the following simultaneously:

* the recorded **SIMULATED** balanced mid-auction state — invented from the
  market prior, never observed;
* this one QB candidate;
* recipient **Team07**;
* **`q = $35`**, held fixed;
* **`FIXED_MARKET`** pass semantics above `$36` (at `$36` the rule is
  `STOP_NOW`);
* the certified completion-set configuration: pool depth 40, 12 per budget,
  0.50 band, `max_worlds=8`;
* the recorded model scenarios, K=11 rotations, 800/4,000 seasons.

It is **not**, and may not be presented as:

* an opening max bid;
* a universal player value;
* applicable to the live draft room;
* valid under `RIVAL_OUTBIDS` — the pass branch assumes `q` stays put, and a
  rival who keeps bidding is a different question with a different answer;
* automatically valid after any sale changes the room. **Any** completed
  purchase changes budgets, the board and the allocation distribution, and
  invalidates it.

**It is not integrated into the dashboard as an applicable live result.** The
draft-day UI continues to show `NO APPLICABLE CE RESULT` for the actual opening
room, and the market guardrail remains the operative draft-day artifact. That is
correct, not a shortfall: this result describes a mid-auction state that has not
occurred.

## 8. Runtime and cost

| stage | cost |
|---|---|
| real board load | 21s |
| certified solve, one price | **0.2 – 5.7s** |
| certified solve, all seven coarse branches | 21s total |
| near-optimal set of 12, one price | 8 – 83s |
| doubling to 24, one price | 20 – 194s |
| certified offer sets (7 budgets, target 12) | 289s |
| certified offer sets (7 budgets, target 24) | 687s |
| CE frontier, one price, K=11, `max_worlds=2` | 103s |
| CE frontier, one price, K=11, `max_worlds=8` | 217s |
| **coarse frontier, 6 prices, `max_worlds=2`** | **638s** |
| 12→24 stability, 2 prices × 2 sets | 1,865s total |
| peak RSS | 804 MB |

**The solver is not the cost.** Completion generation — the thing that blocked
this project for two days — is now 21 seconds for the entire coarse ladder. CE
simulation dominates by two orders of magnitude, and the dominant knob is
`max_worlds`, since each additional world costs a full selection-sample league
evaluation (`n_worlds × selection_sims + holdout_sims` per branch per draw).

For precomputation planning: one candidate's coarse frontier is ~11 minutes at
`max_worlds=2` and ~25 minutes at `max_worlds=8`, plus ~5 minutes of offer
generation. Twenty-five candidates at `max_worlds=2` is therefore roughly
**7 hours** — an overnight job, not a live one. Nothing here can be recomputed
inside a bidding clock, and this document does not claim it can.

### 6.3 Integer refinement, $66 – $79

The coarse bracket was `($65, $80)`. Every integer inside it was evaluated —
**exhaustively, not by binary search**, because a binary search assumes the
delta is monotone in price and that is the property under test. Each price uses
its own certified completion set at its own post-purchase budget (237
with-candidate completions across the generation ladder), `max_worlds=8`, K=11,
800/4,000 seasons.

| p | mean delta | pointwise 95% CI | pointwise | simultaneous 95% CI | simultaneous | certified opt | gap |
|---:|---:|---|---|---|---|---:|---:|
| $66 | +0.01641 | [+0.01292, +0.01990] | favorable | [+0.01048, +0.02234] | **favorable** | 95.5696 | 0.0000 |
| $67 | +0.00905 | [+0.00528, +0.01281] | favorable | [+0.00265, +0.01544] | **favorable** | 95.2862 | 0.0000 |
| $68 | +0.00661 | [+0.00097, +0.01226] | **favorable** | [−0.00298, +0.01621] | unresolved | 95.1786 | 0.0000 |
| $69 | −0.00555 | [−0.00894, −0.00215] | **unfavorable** | [−0.01132, +0.00023] | unresolved | 94.6865 | 0.0000 |
| $70 | −0.00307 | [−0.00861, +0.00247] | *unresolved* | [−0.01248, +0.00635] | unresolved | 94.6034 | 0.0133 |
| $71 | −0.00514 | [−0.00998, −0.00030] | unfavorable | [−0.01336, +0.00308] | unresolved | 94.4851 | 0.0013 |
| $72 | −0.02095 | [−0.02606, −0.01585] | unfavorable | [−0.02962, −0.01229] | **unfavorable** | 94.1188 | 0.0423 |
| $73 | −0.02639 | [−0.03088, −0.02189] | unfavorable | [−0.03402, −0.01875] | unfavorable | 93.7547 | 0.0179 |
| $74 | −0.02525 | [−0.02973, −0.02077] | unfavorable | [−0.03286, −0.01764] | unfavorable | 93.6963 | 0.0041 |
| $75 | −0.02561 | [−0.03004, −0.02118] | unfavorable | [−0.03314, −0.01809] | unfavorable | 93.6963 | 0.0000 |
| $76 | −0.03461 | [−0.04001, −0.02922] | unfavorable | [−0.04378, −0.02545] | unfavorable | 93.1536 | 0.0306 |
| $77 | −0.03891 | [−0.04153, −0.03629] | unfavorable | [−0.04336, −0.03446] | unfavorable | 93.0091 | 0.0090 |
| $78 | −0.03959 | [−0.04258, −0.03660] | unfavorable | [−0.04467, −0.03451] | unfavorable | 93.0091 | 0.0000 |
| $79 | −0.04836 | [−0.05168, −0.04504] | unfavorable | [−0.05400, −0.04272] | unfavorable | 92.6551 | 0.0000 |

Every solver gap is ≤ **0.0423**, inside the 0.10 bound, at every one of the
fourteen prices. No invariant failed anywhere (§6.5).

### 6.4 Two answers, and they are not the same answer

The pointwise column is 14 separate 95% statements; the simultaneous column is a
Bonferroni band with `alpha/14 = 0.00357` per statement, which on `k−1 = 10`
degrees of freedom widens every interval by a factor of `3.7852 / 2.2281 = 1.70`.

| | pointwise | simultaneous |
|---|---|---|
| highest favorable price | **$68** | **$67** |
| first unfavorable price | **$69** | **$72** |
| unresolved gap | ($68, $69) — none | **($67, $72) — four prices wide** |

**The two readings disagree, and the disagreement is the point.** Read
pointwise, the crossing is pinned between `$68` and `$69` and there is no
unresolved region at all. Read simultaneously — which is the only honest way to
read a table that was queried fourteen times — `$68` through `$71` are *not
resolved*, and all that can be defended is: favorable at or below **$67**,
unfavorable at or above **$72**.

Quoting `$68` as "the" reservation price would be reporting the narrower of two
numbers because it is narrower. The conditional maximum this run supports is
**`$67`**, with `$68`–`$71` explicitly unresolved.

### 6.5 Nonmonotonicity, and why the grid was exhaustive

The delta is **not** monotone in price. Two reversals were detected and are
recorded rather than smoothed:

* `$69 → $70`: −0.00555 rises to −0.00307
* `$73 → $74`: −0.02639 rises to −0.02525

Both are small and both sit inside the region the simultaneous band already
calls unresolved, so neither changes a conclusion. But they vindicate the
method: a binary search on `$66`–`$79` would have assumed monotonicity, and at
`$70` it would have been assuming something demonstrably false. The reversals
are a real property of the surface — the certified completion set changes
discretely as the budget crosses a player's price, so the best available roster
does not degrade smoothly with `p`.

**Invariants, all fourteen prices × 11 draws × 5 branches:**

| invariant | result |
|---|---|
| solver proxy gap ≤ 0.10 | max 0.0423 |
| frontier delta == decomposition total | ≤ 1.4e-17 |
| league CE sums to 1 | within 1e-9 |
| conservation (no duplicate ownership, no unpaid blocking) | ok |
| legal budgets, reserves, rosters | ok |
| same pass branch under fixed `q` | `FIXED_MARKET` throughout |
| possession offer price-independent | ok |
| nested certified opportunity sets | asserted |
| **total invariant failures** | **none** |

## 9. What this still does not settle

* **The proxy is not value.** Everything certified here is certified against
  `ProxyEvaluator.strength` — expected weekly starting-lineup points. The
  solver proves it found the best roster *by that measure*. CE then ranks a set
  of near-optimal rosters by championship equity, but the set handed to CE is
  chosen by the proxy, so a roster the proxy ranks 100th and CE would love is
  never seen. §5 shows the set is stable at `max_worlds=8`; it does not show
  proxy rank and CE rank agree.
* **The candidate pool depth is 40.** A player below that cut cannot be chosen
  however cheap. This is a declared bound on the answer, not a detail.
* **The comparison cast is fixed.** Eleven rival rosters are held constant; this
  answers "which completion is best against *this* league", not against a league
  that is also still drafting.
* **`q` is fixed at $35.** Under `RIVAL_OUTBIDS` the pass branch is a different
  problem and none of these numbers apply.
* **The state is SIMULATED.** It was invented from the market prior. No league
  sale has ever been recorded, and nothing here has been calibrated against one.
* **`$68`–`$71` are genuinely unresolved** under the simultaneous band. Closing
  that gap needs more allocation draws (K), not more seasons: the interval is a
  cluster t-interval on `k = 11` draws, and the between-allocation SD is what
  dominates it.
* **Nothing here runs inside a bidding clock.** §8 gives the numbers; one
  candidate is minutes, twenty-five is an overnight job.
