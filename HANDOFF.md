# HANDOFF.md — Night 1

Branch `ce-foundation`. Everything below is reproducible from the commands in §3.

> **All player data is synthetic and labelled as such.** `src/ceauction/synthetic.py`
> invents every number it produces. No real player distribution is used, estimated or
> implied anywhere in this repository.

---

## 1. What was built

A complete championship-equity engine: **drafted roster in, probability of winning the
league out**, for the specified 12-team / $200 / half-PPR / superflex Sleeper auction
league with a weekly league-median result and a fixed six-team bracket.

Auction pricing was not built, by instruction.

The five things worth knowing:

**1. The information barrier is enforced three ways, not asserted once.** The scalar
lineup API consumes `PregameEntry`, which has no field capable of holding a realized
score. The vectorised optimiser's signature accepts projections, availability and
positions — there is no parameter through which a realized score could arrive. Week
*w*'s belief is a cumulative sum over weeks strictly before *w*, so *w*'s own outcome
is arithmetically absent. And the tests permute the realized array and assert the
starter masks are bit-identical, then hand a benched player 1,000 points and assert
nothing moves.

**2. Lineup selection is exact, and provably so.** The slot eligibility sets
(`{QB}`, `{RB}`, `{WR,TE}`, `{RB,WR,TE}`, `{QB,RB,WR,TE}`) form a *laminar* family, so
the startable player sets are a transversal matroid whose independence test collapses
to seven counting constraints. Greedy over a matroid is optimal, so sorting by
projection and taking each player who keeps the counts feasible is exact — no LP, no
Hungarian algorithm, no heuristic. It is cross-checked against brute-force enumeration
of all C(15,8) subsets with an independently written bipartite matcher.

**3. Common random numbers come from the RNG design, not from bookkeeping.** All
randomness is a counter-based hash of `(seed, kind, season, entity, week)`. There is no
sequential state, so a draw's value depends only on its coordinates. Changing one player
on one roster provably perturbs no other player's draws — a test asserts the other 11
teams' weekly scores are byte-identical across scenarios. Measured variance reduction on
the experiments below is 3–20x. Two *alternative* players competing for one roster slot
can share a `crn_key` so they even share their uniform draws and differ only in
parameters.

**4. The 15-man roster is treated as a portfolio throughout.** Nothing labels eight
players as starters. All 15 are re-evaluated every week against that week's byes,
injuries, revealed role changes and contingency status. Measured over 400 simulated
seasons: **a team starts 14.3 of its 15 players at least once** in the regular season
(5th percentile 13, minimum 11). There is no such thing here as "the eight starters".

**5. One real modelling bug was found and fixed by an experiment.** See §7.

---

## 2. Project tree

```
SPEC.md                     the specification the code implements
HANDOFF.md                  this file
OPEN_QUESTIONS.md           decisions needing real data or your judgement
README.md                   install and run
pyproject.toml              packaging; `ce-lab` entry point
docs/
  example_ce_lab_output.txt full CE-laboratory run, 16,000 seasons per arm
  example_league_output.txt CE table for all 12 teams, 20,000 seasons
src/ceauction/              ~3,900 lines
  rng.py                    counter-based RNG; reproducibility + CRN
  stats.py                  dependency-free normal CDF, floored mean
  scoring.py                half-PPR rules; the seam for stat-level real data
  league.py                 settings, positions, the eight slots
  players.py                PlayerSpec -- the reversible real-data interface
  roster.py                 Roster, RosterSet, validation, vectorised views
  pregame.py                pregame-observable types (no realized field exists)
  lineup.py                 exact optimiser + per-slot explanations
  lineup_vec.py             the same algorithm, vectorised
  worlds.py                 latent state / availability / realized / pregame
  synthetic.py              SYNTHETIC pool generator (clearly labelled)
  schedule.py               round robin, permuted per season
  standings.py              dual results, records, total-points tiebreak
  playoffs.py               fixed 6-team bracket; zero randomness
  simulate.py               the pipeline, batched
  ce.py                     CE estimation + paired comparison
  experiments.py            the CE laboratory
  benchmark.py              timing + per-stage profile
  cli.py                    `ce-lab`
tests/                      ~1,600 lines, 116 tests
```

---

## 3. Install and run

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"

.venv/bin/python -m pytest              # 116 tests, ~40s
.venv/bin/ce-lab league --sims 20000    # CE for all 12 teams
.venv/bin/ce-lab lineup --weeks 1 8 14  # why each starter was chosen
.venv/bin/ce-lab experiments            # list the experiments
.venv/bin/ce-lab run --all --sims 16000 # the full laboratory (~7 min)
.venv/bin/ce-lab run spikes --sims 4000 # one experiment (~10s)
.venv/bin/ce-lab bench                  # runtime + Monte Carlo uncertainty
```

Python 3.9+. NumPy is the only runtime dependency; pytest is the only dev dependency.

---

## 4. Test results

**116 passed in 35s, 0 failed, 0 skipped, 0 warnings** (`filterwarnings = ["error"]`).
Every test is deterministic — fixed seeds, no tolerance tuned to a lucky draw, no
`flaky` markers.

| File | Tests | What it pins down |
|---|---:|---|
| `test_experiments.py` | 21 | every experiment builds a legal league and runs paired; the rival-placement control reads zero; floor-correction helpers |
| `test_worlds.py` | 19 | byes, injury hazard/duration, persistence of latent state, mean-neutral vs mean-adding spikes, shared/private/negative correlation, contingency, role-change reveal lag, filter behaviour, chunk independence, `crn_key` sharing |
| `test_simulate.py` | 17 | seed reproducibility, chunk invariance, prefix stability, exactly one champion, roster-as-portfolio, bench depth value, roster validation |
| `test_lineup.py` | 13 | Hall bounds, all eight slots filled, 2×RB and 3×WR/TE enforced, **non-QB superflex**, unavailable players, legally unfillable slots, greedy == brute force, vectorised == scalar, **projection monotonicity**, deterministic ties |
| `test_ce.py` | 12 | **12 identical teams have equal CE** (chi-square, 11 df), no seeding bias by team index, other teams' scores unchanged across paired arms, null comparison is exactly zero, pairing beats independent sampling, no extra playoff randomness |
| `test_standings.py` | 9 | median win/loss/**exact tie**, 2-0 / 1-1 / 0-2, two results per week, **total-points tiebreak** (both directions), index as final tiebreak, schedule covers all 66 pairs |
| `test_playoffs.py` | 8 | exactly six qualifiers, **top-two byes**, fixed 3v6 / 4v5 pairing, **no reseeding** (1 always faces the 4/5 winner), bye teams' week-15 scores are irrelevant, **higher seed advances a tie** in every round and by seed rather than team index |
| `test_information_barrier.py` | 6 | `PregameEntry` cannot hold a realized score; permuting realized scores leaves starter masks identical; **a benched player given +1,000 points changes nothing**; the filtration is a strictly shifted cumsum; **an observable role change moves future lineups but not past ones** |
| `test_rng.py` | 6 | moments, stream independence, value depends only on coordinates |
| `test_scoring.py` | 5 | every half-PPR rule |

Every item on the required test list is covered; the mapping is the bolded text above.

---

## 5. Example CE-laboratory output

Full run in `docs/example_ce_lab_output.txt` (16,000 seasons per arm, seed 20260904,
412s). Focus team is `Team01`. `dPts/wk` is the paired difference in the focus team's
realized weekly scoring — it is the diagnostic that separates "the mechanism did not
fire" from "the effect is real but below this sample size's resolution".

```
experiment          comparison                                         dCE   +/-95%      z   dPts/wk      z
-----------------------------------------------------------------------------------------------------------
marginal-point      +1.00 pt/week to best QB                      +0.01294  0.00241 +10.50   +0.8912 +1286.9
marginal-point      +1.00 pt/week to best RB                      +0.01144  0.00231  +9.71   +0.7948 +446.5
marginal-point      +1.00 pt/week to best WR                      +0.01287  0.00239 +10.58   +0.8511 +850.1
marginal-point      +1.00 pt/week to 2nd RB (fills RB2)           +0.01087  0.00249  +8.55   +0.7267 +141.7
marginal-point      +1.00 pt/week to marginal starter (8th)       +0.00775  0.00265  +5.73   +0.6760  +91.0
marginal-point      +1.00 pt/week to first bench player (9th)     +0.00844  0.00279  +5.93   +0.6737  +84.3
marginal-point      +1.00 pt/week to last bench player (15th)     +0.00213  0.00213  +1.96   +0.1269  +18.5
second-qb           QB vs WR at 14.0 -- roster already QB-deep    -0.08231  0.00573 -28.15   -5.0944 -216.2
second-qb           QB vs WR at 14.0 -- roster thin at QB         +0.00125  0.00230  +1.07   +0.2266  +28.1
volatility          volatile sd 11 vs stable sd 4, starter        +0.00000  0.00307  +0.00   +0.0026   +0.3
volatility          volatile sd 11 vs stable sd 4, flex           -0.00137  0.00283  -0.95   -0.0024   -0.3
spikes              predictable +3.0 vs unforecastable +3.0       +0.02081  0.00460  +8.86   +0.5225  +23.6
concentration       18/6/6 vs 10/10/10 across three WR spots      +0.08737  0.00667 +25.69   +4.0841 +142.1
injury              weekly injury hazard 8% vs 2%                 -0.01450  0.00272 -10.45   -1.2223  -82.4
injury              weekly injury hazard 16% vs 2%                -0.03525  0.00376 -18.35   -2.6411 -134.1
bench-correlation   correlated bench pair (rho ~ .60) vs indep    -0.01331  0.00684  -3.81   +0.0173   +0.5
stack               stacked QB+WR vs same players uncorrelated    -0.00294  0.00539  -1.07   +0.0083   +0.2
handcuff            handcuff to own RB vs uplift on a rival's RB  +0.00975  0.00459  +4.16   +0.2565   +9.5
opponent-placement  same stud on team 1 vs team 2 (CONTROL)       +0.00119  0.00396  +0.59   +0.0000   +0.0
```

**These are statements about the synthetic process, not about football.** What they
establish is that the engine responds to each structural change in a measurable,
correctly signed, statistically resolvable way. Reading them as infrastructure results:

* **Marginal point declines monotonically down the roster** (0.0129 at the top to
  0.0021 at the 15th man), and `dPts/wk` shows why: a full-time starter converts 0.89
  of the point, the last bench player converts 0.13. The conversion rate *is* the
  mechanism, and it is a property of the roster, not of the position.
* **The second-QB answer flips sign with roster context.** On a roster already starting
  two QBs a third is nearly worthless while the same-sized WR upgrades a real flex slot
  (−0.082); on a QB-thin roster the two are indistinguishable (+0.001, n.s.). This is
  the clearest argument in the whole run against a positional modifier and for
  per-roster CE.
* **At matched expected points, volatility is CE-neutral here** (0.000 and −0.001, both
  n.s., with `dPts/wk` at z ≈ 0.3). The median-result format punishes variance and the
  bracket rewards it, and in this league they roughly cancel. Note this is only a clean
  measurement *because* of the fix in §7.
* **Unforecastable production is worth much less than forecastable production**
  (+0.0208, z = 8.9) at identical expected points. This is the information rule showing
  up as a price.
* **Correlated bench upside is worth less than independent** (−0.0133, z = −3.8) with
  `dPts/wk` at zero — marginals are matched exactly, so this is pure dependence
  structure. The lineup takes a max over available options, and independent options
  give the max more chances to be high.
* **Contingency timing has real value beyond expected points** (+0.0098, z = 4.2). Both
  arms receive the same uplift about equally often; only the handcuff's arrives in the
  weeks a hole opened.
* **The control reads zero.** Moving a stud between two rival rosters leaves the focus
  team's `dPts/wk` at exactly 0.0000 and its CE statistically flat.

The one comparison that does not resolve is **stacking** (−0.0029, z = −1.1). Its
`dPts/wk` is zero by construction, so it is a pure variance effect, and it needs
roughly 10x the seasons to separate from zero.

---

## 6. Runtime benchmarks

MacBook, Python 3.9.6, NumPy 2.0.2, single-threaded.

```
  seasons   chunk   seconds   seasons/s  ms/season    CE(T1)        SE     +/-95%
---------------------------------------------------------------------------------
    1,000      64      0.66       1,518      0.659    0.0880   0.00896    0.01756
    4,000      64      2.65       1,507      0.664    0.0900   0.00452    0.00887
   16,000      64     10.47       1,528      0.655    0.0892   0.00225    0.00442
   64,000      64     42.23       1,516      0.660    0.0904   0.00113    0.00222
```

Throughput is flat from 500 to 64,000 seasons — the pipeline is O(n) with no growing
allocation. A *paired* comparison costs two runs plus nothing else, so a 16,000-season
A/B is about 21s.

**Per-stage profile at 2,000 seasons:**

| Stage | Time | Share |
|---|---:|---:|
| world generation | 0.994s | 75.7% |
| lineup + scoring | 0.302s | 23.0% |
| standings | 0.011s | 0.8% |
| schedule | 0.005s | 0.4% |
| playoffs | 0.002s | 0.1% |

**Bottleneck, precisely.** World generation dominates, and inside it the cost is
memory bandwidth over `(seasons × 180 players × 17 weeks)` float64 arrays, plus the
transcendentals in the normal and exponential draws. Standings and playoffs are free.

**What was optimised (after the correctness tests passed, not before):**

* `normal()` now takes both Box-Muller uniforms from the two halves of a *single*
  64-bit hash. The old version used a trailing sub-coordinate, which forced a second
  mixing round at full array size. **1,205 → 1,490 seasons/s.**
* `hash_coords()` does one mixing round per coordinate instead of two.
* Draws are skipped entirely when the corresponding parameters are all zero, which
  makes the inert laboratory and test leagues much cheaper.
* `DEFAULT_CHUNK` 256 → 64. The pipeline is bandwidth-bound and a 64-season batch keeps
  the working set in cache: 1,680 vs 1,490 at 256 and 1,270 at 2,048. Results are
  chunk-invariant (tested), so this is purely a performance knob.
* The §7 correctness fix then cost 9% (1,680 → 1,520). That was the right trade.

**What was deliberately *not* done.** Float32 transcendentals were measured at 2.7x
faster, which would have put throughput near 2,100 seasons/s. They also cap the normal
at about 5.8σ. In a model whose entire purpose is measuring tail-driven championship
outcomes, silently truncating the tail to buy speed is the wrong trade, so the math
stays float64.

**The honest headroom statement.** If CE is ever needed inside a live auction, the
remaining wins are (a) trimming the pool to the ~60 players a decision actually
touches, (b) simulating only the weeks that discriminate, and (c) multiprocessing over
seasons, which is embarrassingly parallel here because seasons share no state. None of
these were needed tonight.

---

## 7. Important modelling choices

**The projection is expected *points*, not the latent mean.** This was a real bug,
found by an experiment rather than by a test. Realized scores are floored at zero, so a
player's expected output is `E[max(0, X)]`, which exceeds his latent mean by an amount
growing with his weekly SD: **+1.3 points/week for a replacement-tier WR at 4.0 with a
weekly SD of 7.2**, and under 0.05 for an 18-point WR at the same SD. The engine had
been projecting the latent mean, which systematically under-projected exactly the
low-mean, high-variance players a 15-for-8 portfolio holds for optionality — biasing
every bench and flex decision against them. The symptom that exposed it: the volatility
experiment showed the volatile arm scoring 0.8 pts/week *more* at equal `base_mean`,
which is a level difference wearing a volatility costume. Fixed in `stats.py` /
`worlds.py`; the volatility experiment is now exactly scoring-neutral.

**The information filtration is a Gaussian conjugate filter on realized residuals**, not
a parametric "learning schedule". Week *w*'s projection uses the mean residual of weeks
strictly before *w*, shrunk by `week_var / season_var`. This gives the right behaviour
for free: a genuine persistent level gets learned over the season, while heavy-tailed
spike weeks are shrunk away and do **not** become future projection.

**Role changes are the one channel through which a surprise creates future value.** A
role change takes effect in week `wc` and becomes observable in `wc + reveal_lag`
(default 1). Realized scores move immediately; projections and therefore lineups move
one week later. Tested directly.

**All correlation goes through one mechanism.** `ShockLoading(group_id, beta)` puts a
player in a named weekly shock group. Team environments, QB/pass-catcher stacks,
negative correlation (opposite-signed betas) and arbitrary user-defined structure are
all the same object, so a real factor model drops in without touching the engine.

**The schedule is permuted per season.** A fixed round robin with weeks 12–14 recycling
rounds 1–3 would make some teams' repeated opponents structurally different. Permuting
team → schedule slot each season removes that, and under CRN the same permutation is
used in both arms so schedule luck cancels. The 12-identical-teams chi-square test is
what catches any leak here.

**Correlation experiments hold marginals fixed.** When an experiment adds a shock
loading, the idiosyncratic SD is reduced so total weekly variance is unchanged to
machine precision. Otherwise "correlation" would just be "more variance".

---

## 8. Simplifications

Full list in `SPEC.md` §10. The ones that actually matter:

1. **No waivers, no FAAB, no trades.** The drafted 15 are the 15 in week 17. This is the
   largest structural simplification, and its direction is known: it **overstates** the
   cost of injuries and **understates** the value of roster spots held as lottery
   tickets. Both biases are largest at the bottom of the roster — exactly where $1–$3
   auction decisions live.
2. **Points are modelled directly**, not built from stat lines. `scoring.py` holds the
   real half-PPR rules for when stat-level data arrives.
3. **All 12 managers play the projection-optimal lineup every week** and never err. This
   understates the value of a roster that is easy to set and overstates the value of a
   deep bench needing correct weekly decisions.
4. **The injury model is a two-parameter hazard/duration process**, independent across
   players, memoryless, with no re-injury correlation, no age and no "questionable"
   state.
5. **Byes are drawn per synthetic NFL team in weeks 5–14.** The real bye schedule is
   known in advance and should replace this.
6. **Synthetic parameters are invented** and calibrated only to be plausible in shape.
7. **Opponent rosters are exogenous** — changing yours does not change theirs.
8. **No week-17 resting-starters effect.**

---

## 9. Open questions

Full catalogue with change instructions in `OPEN_QUESTIONS.md`. The three that block
the most downstream work:

* **B3 — is winner-take-all right?** CE is `P(win the league)`. If your league pays 2nd
  and 3rd, the objective is a payout-weighted mix, which changes how much variance is
  worth and therefore changes every number the model produces. `made_final`,
  `made_playoffs` and `has_bye` are already tracked per season, so this is a small
  change in `ce.py` — but it needs your payout structure.
* **B2 — should the model include in-season roster change?** Draft-day value and value
  net of the waiver wire are different numbers, and the gap is widest exactly at the
  bottom of the roster.
* **B5 — what CE resolution does pricing need?** A $1 decision may be worth less CE than
  16,000 seasons can resolve. This sets the simulation budget per candidate roster and
  therefore whether a live-auction tool is feasible at all.

Data-blocked items: real per-week means and the shape of replacement level (A1),
whether weekly variance scales with the mean (A2, almost certainly yes and currently
modelled as constant per position), real injury hazards (A3), the real correlation
matrix (A4), and how accurate published weekly projections actually are (A5 — this one
calibrates the *entire* value of forecastability).

---

## 10. Known weaknesses

1. **The `stack` experiment does not resolve** at 16,000 seasons (z = −1.1). Pure
   variance effects with zero scoring delta need roughly 10x more seasons. Not a bug,
   but it means the model currently cannot say whether stacking is worth anything in
   this format.
2. **The residual filter is biased for players near the zero floor.** It forms its
   posterior against the latent mean while realized points are floored, so for players
   whose mean sits within ~1.5 weekly SDs of zero the residuals are systematically
   positive. Affects the deep bench, which is where it matters least for CE and most for
   $1 pricing.
3. **Synthetic team strengths span only 95–102 projected points** across the 12 rosters.
   Real auction leagues are probably more dispersed, and CE is convex in roster strength
   (a 7-point spread in projections produces a 2.6x CE ratio here), so effect sizes
   measured on this baseline may not transfer.
4. **Single-threaded.** 64,000 seasons takes 42s. Fine for offline work, likely not fine
   for a live auction with a 30-second clock.
5. **`week_sd` is constant within a position** regardless of rank. A 4-point WR almost
   certainly does not have the same weekly SD as a 17-point WR, and this directly
   affects the floor correction in §7 and therefore every bench valuation.
6. **No opponent correlation.** Your player and your weekly opponent's player in the
   same NFL game are independent here. This affects head-to-head variance but not the
   median result, so it is a second-order error in this format.
7. **The laboratory measures one focus team on one baseline league.** Every effect size
   quoted in §5 is conditional on `Team01`'s specific roster shape. The `second-qb`
   sign flip is the proof that this conditioning matters.

---

## 11. Exact recommended next step

**Build the marginal-CE curve for a single roster slot, and use it to find out whether
CE differences are resolvable at the resolution auction pricing needs.**

Concretely: pick the focus team's flex slot. Sweep a candidate player's `base_mean` from
replacement level to elite in ~1-point steps, holding everything else fixed and sharing
a `crn_key` across the sweep. At each step run a paired comparison against the
replacement-level baseline. That produces `CE(level)` — the curve every pricing scheme
is a transformation of.

Do this **before** anything else, for three reasons:

1. **It is the smallest thing that answers question B5.** The curve's slope near
   replacement level tells you the CE value of one projected point at the bottom of the
   roster, which is the smallest quantity pricing must resolve. If that slope is smaller
   than the standard error you can afford, you learn tonight that live pricing needs a
   different approach — variance reduction beyond CRN, or a surrogate model — rather
   than discovering it after building the pricing layer.
2. **It needs no new data.** Everything required is in the repo. It is one new
   experiment in `experiments.py` (a sweep rather than a pair) plus a plot.
3. **It is the natural input to pricing.** Marginal CE per dollar is
   `dCE/dlevel × dlevel/ddollar`. Tonight's engine produces the first factor exactly.
   The second is the auction problem and is explicitly out of scope.

Expected effort: two to three hours, most of it deciding how to hold the *rest* of the
roster fixed while the slot varies — which is itself the first real modelling question
of the pricing layer, since a player's CE contribution depends on what else you own.
The `second-qb` sign flip in §5 is the warning that this is not a detail.

**Do not start with** real player data ingestion. The interface for it
(`PlayerSpec`, and `weekly_projection_override` for published projections) is already
built and tested, so ingestion is mechanical work that can happen any time. The curve
above tells you whether the engine is *usable* for pricing, which is the question that
should gate everything else.
