# The certified completion solver

> **SIMULATED MID-AUCTION STATE.** Every auction history in the real-price
> section is invented from the market prior for frontier testing. It is NOT
> observed, NOT recorded, NOT historical and NOT calibrated. **No league sale
> has ever been recorded. No maximum bid is quoted in this document.**

## Verdict

**`CERTIFIED_AND_PRACTICAL` — for the completion solver, and for nothing else.**

The focus-team completion problem is now solved to a proven optimality gap in
**seconds per price**, on the real board, at every evaluated price. The 3.8-point
beam instability that produced `QB_UNION_UNDERCONVERGED` is gone, and gone in the
strong sense: the objective is not merely stable, it is **bounded**.

**No championship-equity frontier was run, and no real-player CE maximum bid
exists.** The CE stage was not reached inside the session's hard stop. The
dashboard must continue to show `NO APPLICABLE CE RESULT`, and the market
guardrail remains the operative draft-day artifact. §9 states exactly what
remains.

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
