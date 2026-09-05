# HANDOFF.md — audit correction pass, then the marginal CE curve

Branch `ce-marginal-curve-audit-fixes`, off `ce-marginal-curve`, off
`ce-foundation-audit-fixes`, off `ce-foundation`. Everything below is
reproducible from the commands in §3.

Sections 1-10 are the audit correction pass (Phase 1). Section 11 is the
marginal championship-equity curve (Phase 2), with its audit corrections folded
in. Section 12 is the real-player ingestion layer and section 13 the next phase.

Two claims that appeared in earlier drafts of this document have been withdrawn
and are marked as such where they appeared: that the resolution extrapolation is
conservative (§11 — it can err either way), and that the curve can be divided by
dollars to produce auction value (§12 — it cannot).

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
  example_ce_lab_output.txt  full CE-laboratory run, 16,000 seasons per arm
  example_league_output.txt  CE table for all 12 teams, 20,000 seasons
  example_curve_output.txt   marginal CE curve, 19 levels x 16,000 seasons
  example_marginal_curve.csv the same curve, machine-readable
  example_sensitivity_output.txt  paired real-data sensitivity, 16,000 seasons/cell
  CALIBRATION_AUDIT.md       what the mapping calibrated, and how well
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
  realdata/                 real-player ingestion and PlayerSpec calibration:
                            sources, identity, scoring, contract, validation,
                            mapping, coverage, smoke, sensitivity, reports
  curve.py                  marginal CE curve + Monte Carlo resolution report
  experiments.py            the CE laboratory (12 experiments, 2 of them controls)
  benchmark.py              timing + per-stage profile
  cli.py                    `ce-lab`
tests/                      484 tests
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

.venv/bin/python -m pytest              # 484 tests, ~1m50s
.venv/bin/ce-lab league --sims 20000    # CE for all 12 teams
.venv/bin/ce-lab lineup --weeks 1 8 14  # why each starter was chosen
.venv/bin/ce-lab experiments            # list the experiments
.venv/bin/ce-lab run --all --sims 16000 # the full laboratory
.venv/bin/ce-lab run spikes --sims 4000 # one experiment
.venv/bin/ce-lab curve --sims 16000     # the marginal CE curve (~4 min)
.venv/bin/ce-lab bench                  # runtime + Monte Carlo uncertainty
```

Python 3.9+. NumPy is the only runtime dependency; pytest is the only dev dependency.

---

## 4. Test results

**484 passed, 0 failed, 0 skipped, 0 warnings** in 108s (`filterwarnings = ["error"]`).
Every test is deterministic — fixed seeds, no tolerance tuned to a lucky draw, no
`flaky` markers. 146 at the start of Phase 1, 46 added there, 39 in Phase 2, and 24 in the Phase 2 audit-correction pass.

| File | Tests | What it pins down |
|---|---:|---|
| `test_real_player_input_schema.py` | 24 | **NEW (inventory phase).** the real-player contract's shape: nulls mean *not projected*; a missing scoring category can only be recorded as absent; the expert grades are pinned as non-distributional; raw source fields are required and non-empty; sources are identified by content hash; and a fabricated two-player fixture validates, one of them deliberately sparse |
| `test_realdata_ingestion.py` | 87 | the ingestion layer on fabricated data: scoring arithmetic, median metadata, both availability readings, injury field separation, fumble exclusion, missing categories absent, identity matching and its four failure modes, refusal of the synthetic pool, schema failures, vendor-total and grade-to-variance prevention, determinism |
| `test_playerspec_mapping.py` | 50 | the mapping on fabricated data: no assumption outside the config; the level solve is exact and proportional; both targets supported and agreeing under symmetry; injury solved against both full-season targets with the week/game distinction and infeasibility reported; the variance split preserves total dispersion; unsupported fields are named placeholders; coverage bands; aliases carry reasons; deterministic |
| `test_calibration_audit_fixes.py` | 57 | **NEW.** the six defects the calibration audit found: ids survive reordering, pool limits, re-ranking and every scenario axis, and a collision raises; the twelve rosters are one fixed cast and a missing player refuses; both availability readings differ in the right direction and each reproduces its own target; **median and mean separate once absences are in the model**; the three horizons stay distinct and neither injury target is rescaled; signal quality is explicit and "no learning" is exactly zero posterior, not a small one; paired delta / SE / interval / discordance arithmetic, including that identical arms give exactly zero and that pairing is real on simulated worlds; and the committed docs carry no superseded claim |
| `test_curve.py` | 62 | **NEW (Phase 2).** exact-zero identical arms; order independence; CRN preserved across every level; agreement with a direct paired comparison; genuinely paired adjacent slopes; monotone shape within uncertainty; chunk determinism; the resolution report's `1/sqrt(n)` arithmetic; CSV schema; the isotonic column changing nothing; CLI |
| `test_experiments.py` | 35 | every experiment builds a legal league and runs paired; the rival-placement **control** reads zero and `rival-fit` does not; the aggregate-spot arms and their byte-identical control; the floor helpers cannot return; every documented `ce-lab` command exits 0 |
| `test_worlds.py` | 21 | byes, injury hazard/duration, latent persistence, **negative realized scores**, **realized mean == base_mean**, unavailable weeks still zero, spike mean-neutrality, correlation, contingency, role reveal lag, belief convergence, chunk independence across **all seven layers**, `crn_key` sharing |
| `test_players.py` | 18 | parameter validation and boundaries, immutability, `crn_key` semantics, **every synthetic spec is labelled `SYNTHETIC`**, projection-override validation, shock-loading accumulation, snake-draft balance |
| `test_lineup.py` | 17 | Hall bounds, all eight slots filled, 2×RB and 3×WR/TE enforced, **non-QB superflex**, unavailable players, legally unfillable slots, greedy == brute force, vectorised == scalar, **projection monotonicity**, deterministic ties, per-slot explanations, a bench player stepping in for a bye |
| `test_simulate.py` | 17 | seed reproducibility, chunk invariance (six chunk sizes), prefix stability, exactly one champion, roster-as-portfolio, bench depth value, roster validation |
| `test_observable_signals.py` | 15 | **NEW.** a spike changes only its own week; **+100 injected into a past week moves nothing**; spikes / hidden team shocks / mean-adding hidden production never reach a projection; whole-league lineup masks unchanged; the persistent level *is* learned; `signal_noise_sd` is the learning dial; signals exist only for played weeks; stream independence; **role-change deltas exact to 1e-12 at lag 0, 1 and 4**; four-way decomposition |
| `test_scoring.py` | 14 | every half-PPR rule including **two-point conversions and special-teams TDs**, every `StatLine` field has a coefficient, **a weekly score can be negative**, custom rule sets |
| `test_ce.py` | 13 | **12 identical teams have equal CE** (chi-square, 11 df), no seeding bias by team index, **every shared draw is shared across paired arms** (named individually), null comparison is exactly zero, pairing beats independent sampling, no extra playoff randomness |
| `test_weekly_state.py` | 13 | **NEW.** a pattern appears in the projection as supplied and in the score's conditional mean; distinct from `proj_noise_sd` and from `week_sd`; hidden patterns reach the score only; correlated / independent / offset structure; **a lineup spot actually rotates**; the synthetic league rotates more than a static one |
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
   proof that this conditioning matters, and the marginal curve (§11) is measured on
   one slot of that one roster.
9. **There is no calibrated sizing rule for how much simulation a decision needs.**
   §11's resolution report is a *pilot* for one comparison shape: on that pilot,
   0.005 came in at 8.6s per paired comparison and 0.002 and 0.001 at 54s and 215s,
   against a ~30s decision budget. It does not generalise. Paired variance tracks the
   discordance rate rather than the effect size, and helping and hurting seasons
   cancel in the mean while both adding to variance, so the extrapolation is neither
   a bound nor reliably conservative. Sizing any real decision needs its own pilot or
   an adaptive stopping rule, and neither is built.

---

---

## 11. Phase 2 — the marginal CE curve

`ce-lab curve` sweeps one roster slot's `base_mean` from replacement level to elite
and reports `CE(level)` with honest uncertainty at every step. It is the object any
auction pricing scheme would be a transformation of. **Pricing is not built**: no
dollar values, no opening or live max bids, no inflation model, no roster-completion
solver, no real-player ingestion.

### What makes the numbers usable

**Common random numbers across the whole sweep.** Every level is the *same player*
with one field changed, so he keeps his `player_id` and therefore his `crn_key`. His
injuries, byes, weekly conditions, observable signals, spikes and idiosyncratic draws
are bit-identical at every level; every other player in the league is untouched; the
schedule permutation is keyed by season index. A test asserts all of that at the array
level. What legitimately *does* move is his own realized scoring, his team's weekly
totals, and hence the league median and every team's record — that is the effect being
measured.

**Paired differences, including for the slopes.** Each level retains its per-season
champion indicator, so any two levels differ by a matched per-season difference. The
adjacent slope between level *i−1* and level *i* is its own paired comparison — **not**
a difference of two separately estimated baseline deltas. That distinction is not
cosmetic: the two baseline deltas share the baseline arm and are strongly positively
correlated, so combining them as if independent overstates the slope's standard error.
`test_the_paired_slope_se_is_not_the_unpaired_combination` asserts the paired estimate
is strictly smaller while the point estimates agree.

**Order independence.** The sweep sorts and deduplicates the requested levels, so a
caller cannot change any reported number by shuffling the request. Three different
orderings are asserted to produce byte-identical rows and CSV.

**Published weekly projections move with the level.** When a spec carries a
`weekly_projection_override` — real published projections, one per week — that array
*replaces* the modelled projection entirely. Changing `base_mean` alone would move the
player's realized scoring while leaving the manager's pregame view frozen at the
original level, so every level of the curve would share one pregame view, the lineup
decisions would be identical everywhere, and the measured slope would collapse toward
the value of *unforecastable* production. The candidate therefore shifts every
override entry by the same delta as `base_mean`, preserving the published shape — its
bye weeks, matchup swings and in-season drift — while moving its overall level. Specs
without an override are untouched. This matters for real data and for nothing in the
synthetic pool, which carries no overrides.

### Output

A terminal table and a CSV with a 23-column schema (`MarginalCurve.CSV_COLUMNS`),
written with an explicit `\n` terminator so it round-trips byte-for-byte. No plotting
dependency was added; NumPy remains the only runtime requirement.

`--isotonic` adds an optional display column that **imposes** monotonicity rather than
revealing it. Monotonicity of CE in a player's level is plausible but not guaranteed:
the simulated manager sets lineups from noisy pregame projections, so raising a
player's level changes which players he starts in which weeks, and that propagates
into team scores, the league median for all twelve teams, records, seeding and the
bracket. A local decline in the raw curve is therefore **not** automatically Monte
Carlo noise — it may be a real pathwise feature of this roster and this decision rule.

Read the column as "the curve under an imposed monotonicity assumption". The fit is
pool-adjacent-violators weighted by `1/variance`, in about fifteen lines. It is
strictly additive: the raw CE, its Wilson interval, every delta and every slope are
asserted unchanged when it is switched on, and remain primary.

### The one thing to read off it

**The marginal value of a projected point is not constant down the curve**, and the
variation is large: +0.0033 CE per point at the bottom against +0.018–0.022 from level
13 upward, a factor of about five. The small end is at replacement level — exactly
where $1–$3 auction decisions live, and exactly where the Monte Carlo noise is largest
relative to the effect being measured.

That has a direct consequence for anything downstream: a single "CE per projected
point" constant would be wrong everywhere, and would be *most* wrong at the bottom of
the roster where the largest number of decisions are made.

What it does **not** license is dividing the curve by dollars to get a price. This
sweeps one dimension — a mean — on one slot of one roster. A player's value also
depends on his whole outcome distribution rather than its mean, on availability and
injury, on position and therefore slot eligibility, on correlation with what you
already own, on the rest of the roster and which alternatives remain, and on which
rival gets him instead (which `rival-fit` measured directly and found to be non-zero).
The curve is an input to pricing and a resolution instrument. It is not a price. See
§12. Numbers and the resolution report below.

### Tests

`tests/test_curve.py`, 62 tests, grouped by the guarantee each defends:

| Group | What it pins |
|---|---:|
| identical arms | the baseline level's delta is **exactly** 0.0, not approximately; a duplicate of the baseline is simulated once and still reads zero; deduplication cannot produce a zero-width step |
| order independence | ascending, shuffled and descending requests give byte-identical rows and CSV; the curve is always reported ascending |
| projection overrides | an override shifts by exactly `candidate_level - original_base_mean`, preserving its week-to-week shape; pregame projections shift by that amount and realized means follow; the shift is measured from the original spec so repeated candidates never compound; specs without an override are untouched; an override sweep still produces a rising curve |
| level grid | divisible, non-divisible, decimal and single-level ranges; the maximum is never exceeded and the endpoint is exact rather than an accumulated sum; `--min-level 4 --max-level 10 --step 4` is 4, 8, 10 |
| common random numbers | every other player's realized scores, and the swept player's availability, signals, weekly state, group shocks, spikes and role weeks, are byte-identical across levels — while his own scoring moves; the candidate keeps his `crn_key` and all fourteen non-level parameters |
| agreement | a directly constructed `compare_scenarios` reproduces the sweep's delta, SE and reports exactly; an independent `simulate_seasons` reproduces the raw CE |
| paired slopes | a slope recomputed from matched per-season indicators matches to the bit; the paired SE is strictly smaller than the naive independent combination while the point estimates agree; slopes are per-point, so a 4-point step and two 2-point steps are on one scale |
| shape | points per week rise monotonically; CE rises strongly end to end and no adjacent *decrease* is significant at z = −2 |
| determinism | chunk sizes 1 / 7 / 64 / 512 give identical rows; repeating a sweep is identical |
| resolution | required-n obeys `se ∝ 1/sqrt(n)` exactly; smaller targets cost strictly more; budgets are reported in words; **every verdict is scoped "in this pilot"** and no wording claims general live feasibility or that the extrapolation is conservative |
| output | CSV schema, `\n` terminators, round-trip through the file, `CSV_COLUMNS` is a `ClassVar` and not a constructor field |
| isotonic | the fit is monotone and mean-preserving; weights pull noisy points further; every raw estimate, interval, delta and slope is unchanged when it is enabled; **it is described as imposing an assumption**, and the old "CE cannot fall as a matter of theory" justification is asserted absent |
| CLI | runs and writes CSV, accepts an explicit player and `--isotonic`, rejects a bad range, and does not claim to produce a price |

### The documented 16,000-season run

`docs/example_curve_output.txt` and `docs/example_marginal_curve.csv`, seed 20260904,
19 levels from 4.0 to 22.0 in one-point steps, 231s total.

The target is `SYN-WR070`, the focus team's weakest FLEX-eligible player at 4.46
projected points — chosen because that is the roster spot a $1–$3 auction decision
actually turns on.

```
 level       CE   +/-95%  dCE vs base   +/-95%       z    dCE/pt   +/-95%      z   pts/wk  playoff     bye
-----------------------------------------------------------------------------------------------------------
  4.00   0.0974   0.0046     +0.00000  0.00000   +0.00        --       --     --    95.41   0.5456  0.1934
  5.00   0.1007   0.0047     +0.00331  0.00207   +3.13  +0.00331  0.00207  +3.13    95.63   0.5547  0.1977
  6.00   0.1047   0.0047     +0.00731  0.00285   +5.02  +0.00400  0.00228  +3.44    95.93   0.5683  0.2063
  7.00   0.1089   0.0048     +0.01156  0.00335   +6.77  +0.00425  0.00246  +3.38    96.34   0.5817  0.2174
  8.00   0.1152   0.0049     +0.01781  0.00380   +9.18  +0.00625  0.00254  +4.81    96.83   0.6032  0.2306
  9.00   0.1231   0.0051     +0.02569  0.00419  +12.02  +0.00788  0.00261  +5.92    97.41   0.6228  0.2478
 10.00   0.1347   0.0053     +0.03731  0.00456  +16.04  +0.01162  0.00278  +8.19    98.07   0.6499  0.2688
 11.00   0.1461   0.0055     +0.04875  0.00492  +19.41  +0.01144  0.00279  +8.05    98.80   0.6781  0.2924
 12.00   0.1586   0.0057     +0.06119  0.00520  +23.08  +0.01244  0.00286  +8.51    99.58   0.7074  0.3203
 13.00   0.1747   0.0059     +0.07731  0.00546  +27.73  +0.01613  0.00299 +10.59   100.39   0.7360  0.3489
 14.00   0.1886   0.0061     +0.09125  0.00571  +31.35  +0.01394  0.00292  +9.34   101.24   0.7643  0.3827
 15.00   0.2061   0.0063     +0.10875  0.00598  +35.64  +0.01750  0.00287 +11.97   102.10   0.7882  0.4149
 16.00   0.2236   0.0065     +0.12619  0.00619  +39.95  +0.01744  0.00292 +11.70   102.97   0.8130  0.4480
 17.00   0.2416   0.0066     +0.14419  0.00642  +43.99  +0.01800  0.00297 +11.87   103.84   0.8351  0.4793
 18.00   0.2629   0.0068     +0.16556  0.00665  +48.77  +0.02138  0.00300 +13.95   104.71   0.8564  0.5159
 19.00   0.2810   0.0070     +0.18363  0.00683  +52.69  +0.01806  0.00296 +11.94   105.58   0.8741  0.5476
 20.00   0.2983   0.0071     +0.20087  0.00699  +56.34  +0.01725  0.00299 +11.31   106.45   0.8922  0.5801
 21.00   0.3164   0.0072     +0.21900  0.00714  +60.16  +0.01812  0.00299 +11.89   107.32   0.9079  0.6138
 22.00   0.3380   0.0073     +0.24063  0.00728  +64.80  +0.02162  0.00300 +14.14   108.20   0.9209  0.6453
```

**Infrastructure findings** — claims about this code, which is all the run can support:

* **The curve is monotone at every one of the eighteen steps**, without any smoothing.
  The isotonic column is therefore identical to the raw CE at every level, which is the
  cleanest possible evidence that the display fit is not doing any work here. At 16,000
  seasons every one of the eighteen one-point steps is resolved: the weakest is the
  first, at z = 3.13, and the median is z = 10.0.
* **The marginal point is worth about five times more at the top of the range than at
  the bottom**: +0.00331 CE/point at 4→5, against a mean of +0.0179 (range
  +0.0139–+0.0216) from level 13 upward — a ratio of 5.4x on the means.
  The curve is convex through the middle and flattens above ~15. Mechanically this is
  the startability threshold: a 5-point WR almost never enters the lineup, so his extra
  point converts through very few weeks, while a 15-point WR starts nearly always and
  converts through all of them.
* **Every metric moves together and in the right direction.** Points per week rises
  smoothly from 95.41 to 108.20 — 12.8 points of team scoring for 18 points of one
  player's projection, which is the ~71% conversion rate a not-quite-every-week starter
  should have. Playoff probability goes 0.546 → 0.921 and the top-two bye 0.193 → 0.645.
* **The paired design is doing real work.** The adjacent paired SE is 0.00148, against
  0.00285 for a baseline delta and 0.00234 for a raw CE estimate at the same level.
  Combining two adjacent baseline deltas as if independent gives about 2.6x the paired
  slope's standard error — enough to turn several resolved slopes into unresolved ones,
  which is why the adjacent comparison is computed as its own pairing.

**Synthetic-fantasy findings: none.** Every number above is a property of the invented
parameters in `synthetic.py` and of `Team01`'s specific roster shape. "A projected point
is worth 0.018 CE at level 15" is a statement about an exponential decay curve someone
made up, not about football.

### The resolution pilot

```
  simulations per arm            16,000
  measured throughput            1,295 seasons/s (0.772 ms/season)
  cost of one paired comparison  24.7s at 16,000 seasons
  paired SE, adjacent step       0.00148 (median step 1.00 pts/week, champion differs
                                          in 3.5% of seasons)
  paired SE, vs the baseline     0.00285
  smallest adjacent dCE this run resolves at |z|=2: 0.00295

  target dCE    sims needed   seconds/comparison   verdict
      0.0050          5,577                  8.6   within budget in this pilot (9s < 30s)
      0.0020         34,851                 53.8   over budget in this pilot; offline (0.9 min)
      0.0010        139,401                215.2   over budget in this pilot; offline (3.6 min)
```

**This is a pilot estimate, and its scope is narrow.** What was measured is that
under *this* synthetic curve, on *this* hardware, for *this* comparison structure
(one focus team, one slot, adjacent one-point steps), and at the discordance rate
actually observed here — the focus team's championship outcome flipped in 3.5% of
seasons — a delta-CE of 0.005 needed about 5,577 seasons and 8.6s per paired
comparison, while 0.002 and 0.001 needed 54s and 215s.

**It does not establish that a 0.005 CE difference is resolvable live in general.**
The extrapolation is exact in *n* — the paired SE really does fall as `1/sqrt(n)` —
but it assumes some future comparison has a per-season variance resembling the median
observed here, and that assumption has no guaranteed direction:

* Paired variance tracks the **discordance rate**, not the effect size. The difference
  is zero in every season the change did not decide, so its variance is roughly the
  rate at which the outcome flips — and that rate varies between comparisons for
  reasons other than how large the effect is.
* The difference takes values in `{−1, 0, +1}`. Seasons where the change helps and
  seasons where it hurts **cancel in the mean while both adding to the variance**, so
  a comparison with a small mean and a high flip rate is noisier than one with the
  same mean and a low flip rate.

An earlier draft of this document called the extrapolation conservative. That was
wrong: it can err in either direction, and nothing here bounds it.

A third scoping point, separate from the statistics: the numbers price **one**
comparison. A live auction decision is a comparison per candidate the money could go
to instead.

**What follows practically.** Any decision that must actually be resolved needs either
its own short pilot run to estimate that comparison's discordance rate, or an adaptive
stopping rule that simulates until the paired interval excludes zero or a wall-clock
budget is spent — whichever comes first. Neither is built. If a smaller target turns
out to be needed, the response is structural — trim the pool to the players a decision
touches, simulate fewer weeks, parallelise over seasons — rather than more seasons on
this code path.

---

## 12. The real-player ingestion layer

`ceauction.realdata` turns vendor files into the versioned contract in
`schemas/real_player_input_v1.schema.json`. It stops there: no dollar values, no
opening or live bids, no auction-room behaviour, and no `PlayerSpec` field
populated that the sources cannot support.

`ce-lab ingest --projections <csv> [--fantasypros <csv>] [--injuries <json>]
[--fits <json>] [--contract-out local_data/...] [--report-out ...]`

**Every source path is a parameter.** No absolute path appears anywhere in the
package. The sources are subscriber-gated exports that are not redistributable,
and this repository is public.

### The design rule

**An unresolved question travels with the data.** Where the inventory found a
meaning that could not be proven, the ingestion layer records the ambiguity in
its output instead of resolving it silently. That is why the contract has an
`active_rate` object with two readings and a `preferred` field pinned to null,
a `season_points` object that reports what it left out, and a required
`uncalibrated_parameters` list.

### The settled decisions, and where each is enforced

| Decision | Enforcement |
|---|---|
| Target league is the 12-team superflex league | `build_contract` refuses any other `n_teams`; the validator rejects a `league_config_id` that looks like the old 10-team non-superflex configuration |
| Central tendency is `median`, "user asserted; vendor documentation not located" | Required by the schema; the validator demands provenance whenever the claim is not `unknown` |
| Vendor fantasy total never used | The loader does not read that column; `scoring_source` is pinned to `recomputed_from_components` |
| Expert grades never become a distribution | `may_derive_dispersion` pinned false, covering variance, sd, ceiling, floor and spike alike |
| `players_provisional.csv` never imported | Refused by **column signature**, not filename — it can be renamed and still look like a real board |
| Injury fields stay separate | `injury_prob` (season risk) and `proj_games_missed` (games) preserved apart; no weekly process derived, and `weekly_injury_hazard` is declared uncalibrated |
| Both availability readings, neither preferred | A = `points/17`, B = `points/(17 − games missed)`; `preferred` pinned null and the validator rejects setting it |
| Fumbles excluded by default | Excluded from the primary total, omitted contribution reported per player; `lost`/`total` selectable, never defaulted |
| Missing categories absent, not zero | `treated_as: "absent"` is the only legal value, now required by the schema |

### Identity matching

Names are normalised for accents, punctuation and generational suffixes and
matched on an exact key. It is deliberately **not** fuzzy: edit-distance
matching would join two different players and there would be no way to notice.
Ambiguous, unmatched, duplicate and position-conflicting names are reported
rather than resolved by an arbitrary pick, and an ambiguous player carries a
null team and null bye rather than a coin-flipped one.

### The run against the real sources

Full sanitized result in `docs/INGESTION_AUDIT.md`. Headlines:

* **549 players normalized** (QB 76, RB 135, WR 213, TE 125) from 626 rows.
* **Identity join 524/549 (95.5%)**, 1 ambiguous, 1 duplicate on the right.
* **Injury join 300/549 (54.6%)** — a little over half the pool carries an
  availability profile and the rest carries none.
* **0 validation errors, 5 warnings**, three blocking questions still open.
* Sources identified by SHA-256, not filename.

### One thing the validator got wrong, twice

The check that points do not equal a preserved vendor total was written as an
error and produced false positives on the real data. It first fired on seven
players, all with zero receptions — where half-PPR and full-PPR agree exactly
and coincidence proves nothing. Narrowed to players whose receptions should
separate the two systems, it still fired on one, whose components genuinely
produce that figure under this league's scoring.

It is now an aggregated warning. Value equality is not proof of provenance; the
binding guarantee is structural. The per-player form was also dropped because it
carried a vendor value into a report that gets published.

### Tests

`tests/test_realdata_ingestion.py`, 87 tests, all on fabricated data: scoring
arithmetic, median metadata, both availability interpretations, injury field
separation, fumble exclusion and its alternatives, missing categories staying
absent, identity matching, ambiguous/unmatched/duplicate/conflicting identities,
refusal of the synthetic pool, schema failures for every newly required member,
prevention of vendor-total use, prevention of grade-to-variance mapping, and
deterministic output.

---

## 13. PlayerSpec calibration and the real-data CE smoke test

`ceauction.realdata.mapping` turns the contract into `PlayerSpec` objects.
Nothing it produces is a settled estimate: every quantity is either calibrated
numerically against a stated target with its error reported, or a labelled
**sensitivity scenario**. `PlayerSpecMappingConfig` holds every assumption so
none can hide in a constant.

`ce-lab calibrate --contract … --fits … --sensitivity` runs the whole thing.

### Source semantics, corrected against official documentation

Reading `winwithodds.com/about` and `/season_long_full_stats` settled three
things and reversed one claim this repository was carrying:

* **17 games, for the right reason.** The projection is a full NFL
  regular-season total from preseason season-long props. An earlier revision
  justified 17 as "14 regular-season weeks plus a 3-week bracket" — the right
  number by coincidence.
* **The fantasy horizon holds 16 of those games.** One bye falls inside weeks
  1–17, and the 17th game is in week 18, outside the horizon entirely.
* **The total is a hybrid, not a median.** Continuous categories take the
  over/under line (market medians); discrete categories are devigged into
  probability-weighted expectations (means). Recorded as
  `hybrid_market_location`.
* **The source is not full-health.** It says projections "do not fully capture
  a player's current health", and injury designations are applied manually, so
  a known injury may already have depressed one.

### Identity: derived from the player, not from the row

`player_id` and `crn_key` are a pure function of the canonical player key
(FNV-1a, 62 bits), with collisions raised rather than absorbed. They are RNG
coordinates — every draw a player receives is keyed by them — so an id that
moved when the pool was reordered would hand a player someone else's season,
injuries and common random numbers. **An earlier pass derived them from the
row index of a points-sorted list**, which made every "paired" comparison a
comparison between two different people. A player now keeps the same identity
and the same streams across every fumble interpretation, projection
interpretation, ranking, pool limit and scenario.

### Three horizons, kept apart

| Span | Weeks | Scheduled games | What it is |
|---|---:|---:|---|
| NFL season | 18 | 17 | what the vendor's season figures describe |
| Fantasy window | 17 | 16 | what this engine simulates |
| Difference | 1 | 1 | the bye, and NFL week 18 |

Injury parameters are fitted on the first and simulated on the second, and the
absence implied by the second is reported rather than substituted for the
target of the first.

### What is calibrated, and how well

**Level.** Solved against an explicit target rather than assigned. The season
total is exactly homogeneous in the per-game level, so the solve is a division
by a simulated unit-level statistic — residual **0.0** for all 300 players,
with the Monte Carlo error of that statistic reported per player (median
0.0029 points/game under `full_health`, 0.0133 under `availability_adjusted`,
which cannot cache the statistic per dispersion shape).

**Two independent questions about the projection, both swept.**

* *Which statistic* — `median_target` or `mean_target`. Under `full_health`
  every modelled component is symmetric, so a season total is symmetric and the
  two targets agree to **0.017%**. That is a property of this configuration,
  not a general fact.
* *Which health state* — `full_health` or `availability_adjusted`.
  `full_health` solves the level with no injury process and applies absences
  afterwards, so unconditional season output lands **below** the source total:
  median shortfall 10.9 points, max 44.3. `availability_adjusted` puts the
  fitted process inside the solve, so the simulated full season reproduces the
  source total after absences and the active-game level is higher for anyone
  projected to miss time.

**Once availability is in the model, median and mean stop agreeing.** Absences
truncate the lower tail only, so the total is no longer symmetric: the gap
between the two targets goes from 0.017% under `full_health` to **1.8% at the
median player and 12.5% at the worst** under `availability_adjusted`. Any claim
that the two readings are interchangeable is scoped to the full-health case and
must be re-checked whenever an asymmetric component is populated.

**Injury.** `weekly_injury_hazard` and `injury_mean_weeks` are solved jointly
against both supplied targets by inverting the engine's own availability
process, **over the 18-week / 17-game NFL season both targets describe**. An
earlier pass scaled projected games missed by 16/17 onto the fantasy window
while leaving injury probability on the full-season basis, asking one fit to
reproduce two targets defined on different spans; that is corrected and neither
target is rescaled now.

| Quantity | Span | Result |
|---|---|---|
| Injury probability, target vs achieved | 18 weeks / 17 games | median abs error **0.0033**, mean 0.0043, max 0.0748 |
| Games missed, target vs achieved | 18 weeks / 17 games | median abs error **0.0100**, mean 0.0129, max 0.1775 |
| Games missed, achieved | 18 weeks / 17 games | median **1.906**, mean 1.895, max 4.084 |
| Games missed, expected in fantasy Weeks 1–17 | 17 weeks / 16 games | median **1.769**, mean 1.761, max 3.707 |

The last row is a *consequence* of the fitted per-week parameters, not a target
and not a rescaled target. A week that is both a bye and an absence costs no
scheduled game.

**2 of 240** could not be jointly reproduced — a high injury probability with
very few projected games missed has no solution when frequency and duration are
tied — and are reported rather than quietly fitted.

**60 of 300 players have no individual profile** and fall back to the fitted
positional rate, labelled **all-cause availability** — it counts benching, rest
and trades. Silent perfect health is not an option the config offers.

### Signal quality is stated, never inherited

`PlayerSpec.signal_noise_sd` defaults to `None`, which the engine reads as
`week_sd`. That default is a convenience for synthetic pools, and leaving it
in place for real specs makes *how fast managers learn* a silent function of
*how noisy scoring is* — which contaminates any sweep over `season_sd`. Real
specs now always set it explicitly from `signal_quality`, one of `none`,
`week_sd` or `2x_week_sd`. `none` is encoded as an infinite signal SD and
handled exactly in `worlds.py`, giving a posterior of precisely zero; it is
deliberately distinct from `season_sd = 0`, which says there is nothing to
learn rather than that nobody learns it.

### Coverage, by the band that matters

```
  top_180   team 100.0%  bye 100.0%  injury 91.7%   unresolved 0
  top_240   team 100.0%  bye  99.6%  injury 88.3%   unresolved 1
  top_300   team 100.0%  bye  98.3%  injury 80.0%   unresolved 5
  full pool team  95.6%  bye  87.6%  injury 54.5%   unresolved 68
```

The alarming 54.5% is almost entirely the undrafted tail. Both top-240 gaps are
closed: one by a reviewed alias, one documented as genuinely unresolvable (a
free agent has no team and therefore no bye).

### The smoke test passes on real data

All 18 checks, including that lineups are **maximal** rather than always eight
— real byes cluster, and 97.4% of team-weeks fill all eight with a minimum of
five. An earlier version of that check asserted eight always and failed; the
check was wrong, not the engine.

### Sensitivity: paired, and only what resolved

**A model sensitivity diagnostic, not a player-value analysis and not a price
band.** The twelve rosters are a deterministic snake, built once from the
baseline mapping and reused verbatim so that arms differ only in the assumption
under test.

Every delta is paired season by season over 16,000 common seasons; every
standard error comes from that same difference; an axis counts as demonstrated
only where some team's own 95% interval excludes zero. Each row below is that
axis's largest-moving team.

| Axis | max &#124;ΔCE&#124; | 95% CI | Resolved | Basis |
|---|---:|---|:---:|---|
| `availability_interpretation` | **0.04869** | [+0.04390, +0.05347] | yes | vendor says health treatment is partial |
| `forecastable_share` | **0.02444** | [+0.01810, +0.03077] | yes | **none — pure scenario** |
| `season_sd_x_signal` | 0.02244 | [−0.02916, −0.01572] | yes | **none — pure scenario** |
| `season_sd` | 0.02169 | [−0.02843, −0.01494] | yes | **none — pure scenario** |
| `injury_model` | 0.01356 | [−0.02006, −0.00706] | yes | individual fit vs all-cause fallback |
| `signal_quality` | 0.01006 | [−0.01632, −0.00380] | yes | **none — pure scenario** |
| `fumbles` | 0.00544 | [+0.00161, +0.00926] | yes | column meaning unresolved |
| `target` | 0.00013 | [−0.00005, +0.00030] | **no** | hybrid; both readings carried |

**The new leader is the axis that did not previously exist.** How the source
treats health moves CE roughly twice as far as anything else, and it is the one
axis where the vendor has told us the answer is genuinely in between.

**`season_sd` cannot be quoted without stating the learning speed.** Three of
four scenario-vs-scenario contrasts resolve: at `ssd = 0.20`, switching learning
off entirely moves the largest team by −0.01063 ± 0.00253, about half the size
of the `season_sd` effect itself. The two parameters are not separable.

**Three previous numbers are withdrawn**, including "median versus mean moves
CE by 0.0005" (corrected to +0.00013 [−0.00005, +0.00030] — unresolved) and
"fumbles move CE by 0.0100" (corrected to +0.00544 [+0.00161, +0.00926]).
`docs/CALIBRATION_AUDIT.md` §0 lists what changed and why; full output is in
`docs/example_sensitivity_output.txt`.

---

## 14. Exact remaining blockers before auction values

The engine now runs end to end on real data and its invariants hold. What
stands between here and an auction value is not code.

**1. Availability treatment is now the largest single unknown.** Paired effect
**+0.04869 [+0.04390, +0.05347]** — larger than any other assumption in the
model. The source is documented as neither reliably full-health nor reliably
availability-adjusted; both readings are modelled and neither is preferred.
Choosing wrongly either double-counts injuries or ignores them, and the two
readings also disagree about the level itself by up to 12.5% for players
projected to miss time.

**2. Three uncalibrated scenario parameters, and they interact.**
`forecastable_share` (0.02444), `season_sd` (0.02169) and `signal_quality`
(0.01006) all resolve, and **none has any empirical basis**. Worse, `season_sd`
and `signal_quality` cannot be estimated separately: three of four contrasts
between them resolve.

* `forecastable_share` needs archived **weekly** projections joined to weekly
  outcomes; the R² of that join is the parameter.
* `season_sd` needs preseason projections joined to realised season means
  across several seasons; the residual spread is the parameter.
* `signal_quality` needs a weekly usage series (snap share, route
  participation, target or carry share) joined to the same latent shifts; its
  noise relative to `week_sd` is the parameter.

All three need a weekly or multi-season data feed this project does not have.

**3. The best-alternative term does not exist.** Pricing is
`CE(roster with X at p) − CE(best alternative use of $p)`. The second term is a
roster-completion problem over the remaining board, and nothing here computes
it. The twelve smoke-test rosters are a deterministic snake, explicitly not an
allocation algorithm.

**4. Resolution, now measured on real specs.** Across the 139 non-degenerate
paired team-deltas in `docs/example_sensitivity_output.txt`, the paired
standard error at 16,000 seasons has median **0.00247** and max 0.00362 —
roughly 1.6x the ~0.0015 the synthetic curve pilot reported, because real
rosters are less interchangeable than synthetic ones. Baseline CE across the
twelve teams spans 0.0505 to 0.1573.

So a single paired comparison on real data resolves differences of about
**0.005** at 95% confidence, and a live auction decision is several
comparisons.

**Not blockers:** fumbles (+0.00544, resolved but small; excluding remains the
right default) and the median-versus-mean reading (+0.00013, unresolved at
16,000 seasons *under `full_health`* — it has not been measured under
`availability_adjusted`, where the two targets do separate).

### The exact next step

**Measure the assumption band on one slot swap, before building anything that
produces a price.**

The reasoning. A price is
`CE(roster with X at $p) − CE(best alternative use of $p)`. The four resolved
assumption effects are 0.049, 0.024, 0.022 and 0.010, on a CE whose whole range
across twelve teams is 0.107. Sampling noise is not the constraint — that is
0.0025 — the assumptions are, by an order of magnitude. Building a solver for
the second term first would produce a number whose precision is entirely
fictitious.

The step, concretely:

1. Pick one roster slot on one of the twelve integration rosters and one
   plausible replacement from the undrafted pool — a single swap, not a curve.
2. Run that swap as a paired comparison in **every cell of the cross-product**
   `availability_interpretation` (2) × `forecastable_share` (3) ×
   `season_sd` (3) × `signal_quality` (3) = 54 cells at 16,000 seasons.
   `run_sensitivity` already builds a fixed cast and pairs correctly; what is
   needed is a cross-product driver over it rather than the one-axis-at-a-time
   sweep, plus the swap itself.
   Cost estimate: the current 13-cell sweep runs in a few minutes, so 54 cells
   is well under an hour.
3. Report `min` and `max` ΔCE over the 54 cells as a **band**, not a point, and
   compare that band's width to the ΔCE between two adjacent players on the
   board.

The decision it produces:

* **Band narrower than the player-to-player gap** → the assumptions do not
  prevent ranking, and the best-alternative solver is worth building next.
* **Band wider** → this data cannot support point pricing, and the honest
  output is a price *range* carrying the assumption band, with
  `availability_interpretation` named as the dominant term. Either the vendor
  clarifies the health treatment or a weekly/multi-season feed is obtained to
  estimate the three uncalibrated scenario parameters.

Either way the answer is worth more than a solver built without it, and it is
the smaller piece of work.

**Explicitly not next:** dollar values, opening or live max bids, auction-room
behaviour, opponent bidding, or a roster-completion solver. None of them is
blocked on code, and all of them would inherit the band above.
