# HANDOFF.md — audit correction pass

Branch `ce-foundation-audit-fixes`, off `ce-foundation`. Everything below is
reproducible from the commands in §3.

> **All player data is synthetic and labelled as such.** `src/ceauction/synthetic.py`
> invents every number it produces. No real player distribution is used, estimated or
> implied anywhere in this repository.

---

## 1. What this pass changed, and why it mattered

The Night 1 engine was structurally sound — the league, lineup solver, standings,
bracket, deterministic RNG and paired-comparison architecture all survive unchanged.
Its 146 tests passed. They also could not have caught what was wrong, because every
one of the four defects below was a *modelling* error that the code implemented
faithfully.

**1. A zero floor on weekly scores that the league does not have.** `_draw_realized`
applied `max(raw, 0)`. Interceptions and lost fumbles are both −2 in this league's
rules and nothing floors an individual player's total, so this was inventing a rule.
It was not a harmless clip: it made `base_mean` a latent parameter rather than
expected fantasy points — silently redefining the one field real projection data will
populate — and it forced a compensating `E[max(0, N(µ, σ))]` transform into the
projection path, which two experiments then needed further corrections to undo.

**2. Beliefs updated from realized fantasy points.** The manager's posterior was a
Gaussian conjugate filter on realized residuals, which cannot distinguish "this player
is better than we thought" from "this player got lucky". Measured before the fix:
injecting 100 points into one prior week raised the next week's projection by **+30.8**
(mean 10.0, week_sd 6.0, season_sd 4.0). Unforecastable scoring became forecast.

**3. Role changes were counted twice.** An unrevealed role increase looked to that
filter like evidence of a higher persistent level; when the role was formally revealed
the explicit delta was added again on top. Measured before the fix: a 10-point player
with a certain, revealed +20 role change projected **32.4** on average where 30 was
correct, and ~38 at the season_sd/week_sd ratio the audit used.

**4. No forecastable weekly variation existed.** Every pregame level was effectively
static — it moved only when a role was revealed or a handcuff's starter went out. Two
candidates for one lineup spot could therefore never trade places on knowable weekly
conditions, which means the model could not represent *building a roster spot in the
aggregate* at all. That is the central question a 15-for-8 roster poses.

Two further gaps were closed: the stat-line scoring seam was missing four of the
league's rules, and the "does a rival's roster matter to you?" experiment was a
control being read as a finding.

### The shape of the fix

The model now separates four quantities that were partially conflated, and adds each
**exactly once** on each side:

| Component | Realized score | Projection |
|---|---|---|
| Persistent player level | `base_mean + season_shift` | `base_mean` + posterior from **observable signals** |
| Observable role change | `true_role_delta`, from the change week | `observed_role_delta`, from the reveal week |
| Forecastable weekly state | `weekly_state[p, w]` | the same array, unchanged |
| Unforecastable realized noise | group shock + idiosyncratic + spikes | absent |

The load-bearing change is that `_build_pregame` **no longer takes the realized array
as an argument at all**. Beliefs update from `SignalBatch`, a distinct observable
process:

```
level_signal[p, w] = season_shift[p] + signal_noise_sd[p] * xi[p, w]
observed[p, w]     = available[p, w]
```

drawn from its own RNG stream, so it shares no draw with any realized score. It stands
for what a manager actually watches — snap share, route participation, target or carry
share, depth-chart reporting, the drift of a published projection — expressed on the
fantasy-points scale, so calibrating it against real data means estimating one number
and changing nothing else.

That single structural change fixes defects 2 and 3 together, and fixes them by
construction rather than by arithmetic anyone has to trust: a spike cannot reach a
future projection because there is no channel through which it could arrive, and an
unrevealed role change cannot inflate the level posterior because the signal does not
observe role changes.

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
src/ceauction/
  rng.py                    counter-based RNG; reproducibility + CRN
  scoring.py                the COMPLETE half-PPR rule set; the stat-line seam
  league.py                 settings, positions, the eight slots
  players.py                PlayerSpec -- the reversible real-data interface
  roster.py                 Roster, RosterSet, validation, vectorised views
  pregame.py                pregame-observable types (no realized field exists)
  lineup.py                 exact optimiser + per-slot explanations
  lineup_vec.py             the same algorithm, vectorised
  worlds.py                 latent / availability / SIGNALS / realized / pregame
  synthetic.py              SYNTHETIC pool generator (clearly labelled)
  schedule.py               round robin, permuted per season
  standings.py              dual results, records, total-points tiebreak
  playoffs.py               fixed 6-team bracket; zero randomness
  simulate.py               the pipeline, batched
  ce.py                     CE estimation + paired comparison
  experiments.py            the CE laboratory (12 experiments, 2 of them controls)
  benchmark.py              timing + per-stage profile
  cli.py                    `ce-lab`
tests/                      192 tests
```

`stats.py` was **deleted**. It held `floored_mean` and `match_floored_mean`, which
existed only to compensate for the zero floor; with the floor gone they had no
remaining caller, and a test now asserts they cannot come back.

---

## 3. Install and run

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip     # required: pip < 21.3 cannot do editable installs
.venv/bin/pip install -e ".[dev]"

.venv/bin/python -m pytest              # 192 tests, ~2 min
.venv/bin/ce-lab league --sims 20000    # CE for all 12 teams
.venv/bin/ce-lab lineup --weeks 1 8 14  # why each starter was chosen
.venv/bin/ce-lab experiments            # list the experiments
.venv/bin/ce-lab run --all --sims 16000 # the full laboratory
.venv/bin/ce-lab run spikes --sims 4000 # one experiment
.venv/bin/ce-lab bench                  # runtime + Monte Carlo uncertainty
```

Python 3.9+. NumPy is the only runtime dependency; pytest is the only dev dependency.

---

## 4. Test results

**192 passed, 0 failed, 0 skipped, 0 warnings** (`filterwarnings = ["error"]`).
Every test is deterministic — fixed seeds, no tolerance tuned to a lucky draw, no
`flaky` markers. 146 before this pass, 46 added.

| File | Tests | What it pins down |
|---|---:|---|
| `test_experiments.py` | 35 | every experiment builds a legal league and runs paired; the rival-placement **control** reads zero and `rival-fit` does not; the aggregate-spot arms and their byte-identical control; the floor helpers cannot return; every documented `ce-lab` command exits 0 |
| `test_worlds.py` | 21 | byes, injury hazard/duration, latent persistence, **negative realized scores**, **realized mean == base_mean**, unavailable weeks still zero, spike mean-neutrality, correlation, contingency, role reveal lag, belief convergence, chunk independence across **all seven layers**, `crn_key` sharing |
| `test_players.py` | 18 | parameter validation and boundaries, immutability, `crn_key` semantics, **every synthetic spec is labelled `SYNTHETIC`**, projection-override validation, shock-loading accumulation, snake-draft balance |
| `test_lineup.py` | 17 | Hall bounds, all eight slots filled, 2×RB and 3×WR/TE enforced, **non-QB superflex**, unavailable players, legally unfillable slots, greedy == brute force, vectorised == scalar, **projection monotonicity**, deterministic ties, per-slot explanations, a bench player stepping in for a bye |
| `test_simulate.py` | 17 | seed reproducibility, chunk invariance (six chunk sizes), prefix stability, exactly one champion, roster-as-portfolio, bench depth value, roster validation |
| `test_observable_signals.py` | 15 | **NEW.** a spike changes only its own week; **+100 injected into a past week moves nothing**; spikes / hidden team shocks / mean-adding hidden production never reach a projection; whole-league lineup masks unchanged; the persistent level *is* learned; `signal_noise_sd` is the learning dial; signals exist only for played weeks; stream independence; **role-change deltas exact to 1e-12 at lag 0, 1 and 4**; four-way decomposition |
| `test_scoring.py` | 14 | every half-PPR rule including **two-point conversions and special-teams TDs**, every `StatLine` field has a coefficient, **a weekly score can be negative**, custom rule sets |
| `test_ce.py` | 13 | **12 identical teams have equal CE** (chi-square, 11 df), no seeding bias by team index, **every shared draw is shared across paired arms** (named individually), null comparison is exactly zero, pairing beats independent sampling, no extra playoff randomness |
| `test_weekly_state.py` | 12 | **NEW.** a pattern appears in the projection as supplied and in the score's conditional mean; distinct from `proj_noise_sd` and from `week_sd`; hidden patterns reach the score only; correlated / independent / offset structure; **a lineup spot actually rotates**; the synthetic league rotates more than a static one |
| `test_standings.py` | 9 | median win/loss/**exact tie**, 2-0 / 1-1 / 0-2, two results per week, **total-points tiebreak** (both directions), index as final tiebreak, schedule covers all 66 pairs |
| `test_playoffs.py` | 8 | exactly six qualifiers, **top-two byes**, fixed 3v6 / 4v5 pairing, **no reseeding**, bye teams' week-15 scores are irrelevant, **higher seed advances a tie** in every round |
| `test_information_barrier.py` | 7 | `PregameEntry` cannot hold a realized score; permuting realized scores leaves starter masks identical; **a benched player given +1,000 points changes nothing**; the filtration is a strictly shifted cumsum over **signals**; **`_build_pregame` has no realized parameter**; an observable role change moves future lineups but not past ones |
| `test_rng.py` | 6 | moments, stream independence, value depends only on coordinates |

### Determinism, chunk invariance and CRN — verified directly

```
repeat run identical                                     yes
chunk sizes 1 / 3 / 7 / 64 / 500 / 4096 identical        yes
900-season run's first 300 == a 300-season run           yes
world layers chunk-invariant (realized, projection,
  signals, weekly_state, posterior, role, availability)  yes
paired arms share: availability, byes, group shocks,
  observable signals, weekly state, spikes, role weeks   yes
changing one player moves that player only;
  the other 11 teams' scores are byte-identical          yes
comparing a league with itself gives delta_CE == 0.0     yes
```

---

## 5. Example CE-laboratory output

Full run in `docs/example_ce_lab_output.txt` (16,000 seasons per arm, seed 20260904,
588s). Focus team is `Team01`. `dPts/wk` is the paired difference in the focus team's
realized weekly scoring — the diagnostic that separates "the mechanism did not fire"
from "the effect is real but below this sample size's resolution".

```
experiment            comparison                                   dCE   +/-95%      z   dPts/wk
------------------------------------------------------------------------------------------------
marginal-point        +1.00 pt/wk to best QB                  +0.01344  0.00243 +10.86   +0.8909
marginal-point        +1.00 pt/wk to best RB                  +0.01213  0.00238  +9.98   +0.8169
marginal-point        +1.00 pt/wk to best WR                  +0.01325  0.00248 +10.48   +0.8723
marginal-point        +1.00 pt/wk to 2nd RB (fills RB2)       +0.01181  0.00255  +9.08   +0.6811
marginal-point        +1.00 pt/wk to marginal starter (8th)   +0.00625  0.00257  +4.76   +0.5740
marginal-point        +1.00 pt/wk to first bench (9th)        +0.00838  0.00266  +6.18   +0.6042
marginal-point        +1.00 pt/wk to last bench (15th)        +0.00462  0.00224  +4.05   +0.2536
second-qb             QB vs WR at 14.0 -- roster QB-deep      -0.08669  0.00567 -29.96   -5.4553
second-qb             QB vs WR at 14.0 -- roster thin at QB   -0.00256  0.00284  -1.77   -0.1064
volatility            sd 11 vs sd 4, starter level            +0.00250  0.00337  +1.45   +0.0189
volatility            sd 11 vs sd 4, flex level               -0.00187  0.00351  -1.05   -0.0102
spikes                predictable +3.0 vs unforecastable +3.0 +0.01506  0.00466  +6.34   +0.6535
concentration         18/6/6 vs 10/10/10 across three spots   +0.09094  0.00648 +27.51   +4.3552
aggregate-lineup-spot forecastable rotation vs stable starter -0.00387  0.00572  -1.33   +0.0279
aggregate-lineup-spot unforecastable rotation vs stable (CTL) -0.09369  0.00644 -28.50   -4.7859
aggregate-lineup-spot the same rotation, forecastable vs not  +0.08981  0.00653 +26.95   +4.8138
injury                weekly injury hazard 8% vs 2%           -0.01831  0.00274 -13.11   -1.3329
injury                weekly injury hazard 16% vs 2%          -0.03794  0.00373 -19.95   -2.8857
bench-correlation     correlated bench pair vs independent    -0.00625  0.00672  -1.82   +0.0083
stack                 stacked QB+WR vs the same uncorrelated  -0.00075  0.00548  -0.27   +0.0122
handcuff              handcuff to own RB vs a rival's RB      -0.00156  0.00462  -0.66   +0.3456
opponent-placement    same stud on rival 1 vs rival 2 (CTL)   +0.00219  0.00407  +1.05   +0.0000
rival-fit             stud QB to weak rival vs to contender   +0.01044  0.00421  +4.86   +0.0000
```

### Infrastructure findings — claims about this code

These are what the run establishes, and they are the only kind of claim it can make.

* **Every mechanism fires with the right sign and is resolvable at 16,000 seasons**,
  except the three noted below. Marginal projection, positional eligibility,
  concentration, availability risk, forecastability and rival fit all separate cleanly.
* **The two controls behave as controls.** `opponent-placement` reads +0.0022 (z = 1.05)
  with the focus team's scoring at exactly 0.0000 — the schedule is exchangeable and no
  team identity leaks into the standings. The `aggregate-lineup-spot` unforecastable arm
  has byte-identical realized production to its forecastable twin, so the +0.0898 gap
  between them is purely the value of pregame knowability and cannot be an artefact of
  retrospective selection.
* **Forecastability is now measured twice, two different ways, and agrees.** `spikes`
  (+0.0151) prices unforecastable production against forecastable at equal expected
  points; `aggregate-lineup-spot` (+0.0898) prices the same thing at much larger
  amplitude with realized production held byte-identical. Both say the same thing about
  the engine: points the manager cannot see coming are worth substantially less.
* **The lineup is genuinely a weekly decision.** A team now starts 14.8 of its 15
  players at least once and changes 2.7 starters a week (was 14.0 and 2.0). The
  marginal point at the 15th roster spot is now resolvable (+0.0046, z = 4.05) where it
  previously sat right at the threshold — a deep bench player earns his value in the
  weeks conditions favour him, which the model previously could not represent.
* **Rival ownership matters when rivals differ in fit** (+0.0104, z = 4.86, focus
  scoring identical to the last decimal) and does not when they are interchangeable
  (+0.0022, z = 1.05). The pair is the finding; either alone would mislead.
* **Three comparisons do not resolve at this sample size:** `stack` (z = −0.27),
  `bench-correlation` (z = −1.82) and `handcuff` (z = −0.66). The first two are pure
  variance effects with `dPts/wk` at zero by construction and need roughly 10x the
  seasons. `handcuff` is different and worth flagging: its mechanism clearly fires
  (`dPts/wk` +0.346, z = +12.0) but the CE effect no longer separates. See §9.

### Synthetic-fantasy findings — there are none

Every effect size above is a property of the invented parameters in `synthetic.py`.
None of it is an estimate of anything about real football, and no auction decision
should be derived from any number in this section. The separation matters most exactly
where the numbers look most quotable: "concentration is worth 0.09 CE" is a statement
about an exponential decay curve someone made up.

---

## 6. Runtime benchmarks

MacBook, Python 3.9.6, NumPy 2.0.2, single-threaded.

```
  seasons   chunk   seconds   seasons/s  ms/season    CE(T1)        SE     +/-95%
---------------------------------------------------------------------------------
      250      64      0.19       1,312      0.762    0.0960   0.01863    0.03652
    1,000      64      0.76       1,324      0.755    0.1010   0.00953    0.01868
    4,000      64      3.12       1,282      0.780    0.1008   0.00476    0.00933
   16,000      64     12.09       1,324      0.755    0.0981   0.00235    0.00461
```

Throughput is flat across the range — the pipeline is O(n) with no growing allocation.
A *paired* comparison costs two runs and nothing else, so a 16,000-season A/B is ~24s
and the full 23-comparison laboratory is 588s.

**Per-stage profile at 2,000 seasons:**

| Stage | Time | Share |
|---|---:|---:|
| world generation | 1.191s | 78.6% |
| lineup + scoring | 0.307s | 20.3% |
| standings | 0.010s | 0.7% |
| schedule | 0.004s | 0.3% |
| playoffs | 0.002s | 0.1% |

**Cost of this pass: 1,520 → 1,324 seasons/s, about 13%.** It buys two new full-size
`(seasons × 180 × 17)` draws — the observable signal and the stochastic weekly state —
and world generation was already 76% of runtime, so the arithmetic is unsurprising.
Removing the zero floor and the `floored_mean` transform gave a little back. This was
the right trade: the alternative was a model that could not answer the aggregate-roster
question and that turned lucky touchdowns into projections.

**What was deliberately *not* done.** Float32 transcendentals were measured at 2.7x
faster but cap the normal at about 5.8σ. In a model whose purpose is measuring
tail-driven championship outcomes, truncating the tail to buy speed is the wrong trade,
so the math stays float64.

**Headroom, honestly.** If CE is ever needed inside a live auction the remaining wins
are (a) trimming the pool to the ~60 players a decision touches, (b) simulating only the
weeks that discriminate, and (c) multiprocessing over seasons, which is embarrassingly
parallel here because seasons share no state. None were needed for this pass.

---

## 7. Important modelling choices

**`base_mean` is expected fantasy points.** Not a latent parameter, not a median of a
truncated distribution — the thing a projection source publishes. That is only true
because the zero floor is gone, and it is what makes real-data ingestion a substitution
rather than a reinterpretation.

**Weekly scores can be negative, and negative projections are allowed.** The lineup
optimiser handles a negative projection correctly: it benches such a player, but still
starts him rather than leave a slot unfilled, which is the right decision.

**Two information barriers, enforced differently.** No same-week clairvoyance is
arithmetic — a cumulative sum shifted one week. No learning from unforecastable noise
is structural — `_build_pregame` has no realized parameter. The second is the one that
was broken, and structure was the only fix worth making: an arithmetic fix would have
had to be re-verified every time a new component was added to a realized score.

**One channel for learning, one for weekly conditions, and they do not overlap.**
`SignalBatch` observes the persistent latent level only. `weekly_state` carries what is
knowable about *this* week. Role changes go through `observed_role_delta`. Each
component reaches the projection through exactly one route, which is what makes the
role-change identity exact rather than approximately right.

**The forecastable share is carved out of `week_sd`, not added to it.** Adding it would
have made every synthetic player quietly more volatile and made the numbers below
incomparable to anything measured before. `_split_weekly_sd` preserves the marginal
weekly distribution and reclassifies part of it as knowable.

**Two experiments exist to be controls, and each belongs to a pair.**
`aggregate-lineup-spot` carries an arm whose realized production is byte-identical to
the treatment, differing only in pregame observability — without it the result reads as
"variance is free". `opponent-placement` and `rival-fit` ask the same question of
interchangeable and non-interchangeable rivals; without the second, the first reads as
a general claim about auctions, which it is not.

**All correlation still goes through one mechanism.** `ShockLoading(group_id, beta)`.
Team environments, QB/pass-catcher stacks, negative correlation and arbitrary
user-defined structure are the same object, so a real factor model drops in without
touching the engine.

**The schedule is still permuted per season**, and the 12-identical-teams chi-square
test is what would catch any leak there.

---

## 8. Simplifications

Full list in `SPEC.md` §11. The ones that actually matter:

1. **No waivers, no FAAB, no trades.** The drafted 15 are the 15 in week 17. This is
   the largest structural simplification, and its bias does **not** point one way —
   an earlier version of this document claimed it did:
   * it **overvalues** static drafted depth and handcuffs as injury protection,
     because in reality a comparable replacement is usually available on waivers, so
     much of what the model prices as insurance covers a risk you could have covered
     later for free;
   * it **undervalues** churnable lottery-ticket roster spots, because a failed bet
     here occupies a spot for seventeen weeks instead of being cut in week 4;
   * it **overstates** the damage from injuries, because the real fallback is the
     waiver wire rather than whoever you happen to own.

   A handcuff and a lottery ticket are biased in *opposite* directions by the same
   simplification, which is why "it understates bench depth" was the wrong summary.
2. **Points are modelled directly**, not built from stat lines. `scoring.py` now holds
   the complete rule set, including the two-point conversions and individual
   special-teams touchdowns it was missing.
3. **The observable-information channel is one Gaussian signal per played week.** Real
   managers read snaps, targets, routes, depth charts and Vegas lines, each with its
   own precision and lag. The *shape* is the modelling claim; `signal_noise_sd` is
   uncalibrated and defaults to a conservative placeholder.
4. **All 12 managers play the projection-optimal lineup every week** and never err.
5. **The injury model is a two-parameter hazard/duration process**, independent across
   players, memoryless, with no re-injury correlation, age or "questionable" state.
6. **Byes are drawn per synthetic NFL team in weeks 5–14.**
7. **Synthetic parameters are invented** and calibrated only to be plausible in shape.
8. **Opponent rosters are exogenous** — and `rival-fit` shows this one has teeth.
9. **No week-17 resting-starters effect.**

---

## 9. Which previous conclusions were invalidated or materially changed

Night 1's `HANDOFF.md` §5 drew seven readings from the laboratory. Here is what
survived, what moved, and what was never a valid reading in the first place.

### Invalidated — the reading was wrong, not just the number

**"A rival's roster composition does not affect you."** Night 1 reported the
`opponent-placement` control at delta-CE ≈ 0 and read it as a general statement. It is
not one. That experiment swaps a player between two *near-duplicate* rival rosters for
his own counterpart at the same position, so it can only ever measure exchangeability —
a property of the schedule and standings code. The new `rival-fit` experiment asks the
same question of rivals who differ in how well they can use the player and gets a
clearly non-zero answer. Both are now in the suite, the control is labelled as one, and
each interpretation points at the other.

**"The filter partially recovers the loss from unforecastable spikes."** Night 1's
`spikes` reading said the residual filter slowly learns a spiky player's elevated level.
It did — and that was the bug, not a feature. Unforecastable production was becoming
forecast. Beliefs now update from the observable-signal channel, which never sees a
spike, so the loss is not recovered at all and the measured gap is the full price of
unforecastability.

**"One real modelling bug was found and fixed by an experiment"** — Night 1 §7, the
`floored_mean` projection change. That "fix" was a correct response to a defect that
should not have existed: the zero floor itself. Both are now gone. The claim that
projecting through the floor was a correctness improvement is removed from `SPEC.md`
and from this document.

### Materially changed — the reading holds, the number does not

**Everything measured on the synthetic league.** Two changes move every number in the
table: removing the zero floor changes what `base_mean` means and therefore every
realized mean, and carving `weekly_state_sd` out of `week_sd` changes what fraction of
weekly variation is knowable. Any effect size quoted from the Night 1 run should be
treated as superseded, not adjusted.

**`concentration` specifically.** Its Night 1 note conceded that "about 0.8" of the
scoring gap was the zero floor rather than the lineup effect, and that figure was
accurate: under the floor the 18/6/6 arm expected 31.53 pts/week against 10/10/10's
30.72, because each 6.0 player gained 0.76 from the floor while the 18.0 player gained
0.01. What changed is that the caveat is no longer needed — the arms now expect exactly
30.0 either way, so the entire delta is the lineup effect and the experiment measures
what its title says.

**`volatility` specifically.** Night 1 equalised the arms by inverting `floored_mean`
numerically. They are now equal because equal `base_mean` means equal expected points,
full stop. The conclusion ("at matched expected points, volatility is roughly CE-neutral
in this format") is unchanged in kind, but it now rests on an identity rather than on a
bisection search.

**Two readings lost their statistical support entirely.** Both had `dPts/wk`
significant and `dCE` significant on Night 1; both now have `dPts/wk` significant and
`dCE` not. Neither is a broken mechanism — they are effects that shrank relative to
noise once the rest of the roster gained forecastable weekly variation:

* **`handcuff`: "contingency timing has real value beyond expected points" (+0.0098,
  z = 4.16) is now +0.0016 in the *opposite* direction at z = −0.66,** while the
  mechanism still fires hard (`dPts/wk` +0.346, z = +12.0). The plausible reading is
  that when every other rostered player's projection moves week to week, a hole in the
  lineup has better alternatives than it used to, so the *timing* of the handcuff's
  points is worth less. That is a hypothesis, not a measurement: what the run supports
  is only that the CE effect is no longer resolvable at 16,000 seasons.
* **`bench-correlation`: "correlated bench upside is worth less than independent"
  (−0.0133, z = −3.81) is now −0.0063 at z = −1.82.** Same sign, half the size, below
  resolution. It joins `stack` as a pure-variance comparison needing far more seasons.

**One reading flipped sign without becoming significant.** The `second-qb` QB-thin arm
went from +0.0013 (z = +1.07) to −0.0026 (z = −1.77). Night 1 read it as "on a QB-thin
roster the two are indistinguishable", which is still the right reading — but the
scoring delta is now clearly negative (−0.106, z = −8.6), so the honest statement is
that even on a QB-thin roster the WR is marginally the better asset here, not that the
two are equivalent.

### Unchanged in substance

The marginal point declining down the roster, the `second-qb` sign flip with roster
context (−0.087 on a QB-deep roster), the cost of availability risk scaling
super-linearly with hazard, and `stack` not resolving at this sample size — all still
hold, with different numbers. The `second-qb` sign flip remains the single strongest
argument in the run against a positional modifier and for per-roster CE.

One reading got *stronger*: the marginal point at the **15th roster spot** went from
+0.0021 (z = 1.96, right at the threshold) to +0.0046 (z = 4.05). That is the
forecastable-weekly-state channel doing exactly what it was added for — the last man on
the roster now has weeks in which he is knowably the right start.

### A note on what "invalidated" means here

None of these were football claims, and none of them should be read as ones now. The
laboratory establishes that the engine responds to structural changes in a measurable,
correctly signed, resolvable way. The three items above are cases where the *engine's
response* was wrong or where the *reading of a control* was wrong. Effect sizes remain
statements about invented parameters throughout.
---

## 10. Known weaknesses

1. **`signal_noise_sd` is the least-grounded parameter in the model.** It sets how fast
   a genuinely improved player becomes startable, which is most of what makes a
   mid-season breakout worth anything, and its default ("one week of usage ≈ one
   observed score") is a placeholder chosen to be conservative rather than an estimate.
   Every result involving learning is conditional on it. `OPEN_QUESTIONS.md` A5 says
   what would settle it.
2. **The forecastable share of weekly variance is also invented.** How much of a
   player's week-to-week movement is knowable before kickoff decides how much of the
   roster is a live weekly decision, and therefore how much a deep bench is worth. The
   current ~40% share is a guess with real leverage on the `aggregate-lineup-spot`
   result.
3. **The observable signal is one Gaussian channel about the persistent level only.**
   Real information also arrives about *this week* specifically (a beat writer's
   Friday report), and about role changes before they take effect (a trade). The
   first has no channel; the second would need a negative `role_reveal_lag`.
4. **`week_sd` is constant within a position** regardless of rank. A 4-point WR almost
   certainly does not have the same weekly SD as a 17-point WR, and this now affects
   the forecastable/unforecastable split as well as the level.
5. **Synthetic team strengths span a narrow band** across the 12 rosters. Real auction
   leagues are probably more dispersed, and CE is convex in roster strength — which
   `rival-fit` now demonstrates directly — so effect sizes measured on this baseline
   may not transfer.
6. **Single-threaded.** Fine for offline work, likely not fine for a live auction with
   a 30-second clock.
7. **No opponent correlation.** Your player and your weekly opponent's player in the
   same NFL game are independent here. This affects head-to-head variance but not the
   median result, so it is second-order in this format.
8. **The laboratory measures one focus team on one baseline league.** Every effect size
   is conditional on `Team01`'s specific roster shape. The `second-qb` sign flip is the
   proof that this conditioning matters.

---

## 11. Exact recommended next step

Unchanged from Night 1, and now on a foundation that can carry it:

**Build the marginal-CE curve for a single roster slot, and use it to find out whether
CE differences are resolvable at the resolution auction pricing needs.**

Pick the focus team's flex slot. Sweep a candidate's `base_mean` from replacement level
to elite in ~1-point steps, holding everything else fixed and sharing a `crn_key`
across the sweep. At each step run a paired comparison against the replacement-level
baseline. That produces `CE(level)` — the curve every pricing scheme is a
transformation of.

Three reasons it comes first:

1. **It is the smallest thing that answers `OPEN_QUESTIONS.md` B4.** The curve's slope
   near replacement level is the CE value of one projected point at the bottom of the
   roster, which is the smallest quantity pricing must resolve. If that slope is
   smaller than the standard error you can afford, you learn it before building the
   pricing layer rather than after.
2. **It needs no new data.** One new experiment (a sweep rather than a pair) plus a
   plot.
3. **It is the natural input to pricing.** Marginal CE per dollar is
   `dCE/dlevel × dlevel/ddollar`; this engine produces the first factor exactly, and
   the second is the auction problem, which is out of scope.

Expect most of the effort to go into deciding how to hold the *rest* of the roster
fixed while the slot varies — itself the first real modelling question of the pricing
layer, since a player's CE contribution depends on what else you own. The `second-qb`
sign flip is the warning that this is not a detail.

**Do not start with** real player data ingestion. The interface for it (`PlayerSpec`,
`weekly_projection_override`) is built and tested, so ingestion is mechanical work that
can happen any time. Two parameters it would newly need calibrating —
`signal_noise_sd` and `weekly_state_sd` — are called out in `OPEN_QUESTIONS.md` A2 and
A5, and both matter more than the means do for anything at the bottom of the roster.
