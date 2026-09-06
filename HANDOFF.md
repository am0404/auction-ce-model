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
tests/                      843 tests
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

.venv/bin/python -m pytest              # 843 tests, ~25 min
.venv/bin/python -m pytest -m "not slow"   # 816 tests, ~4 min
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

**758 tests, 0 failed, 0 warnings**. The suite takes about 25 minutes because
the corrected buy/pass is 4.4x slower and 27 tests drive it end to end. Those
27 carry a `slow` marker, so `pytest -m "not slow"` runs the other 731 in about
four minutes for a fast pass; nothing is skipped by default (`filterwarnings = ["error"]`).
Every test is deterministic — fixed seeds, no tolerance tuned to a lucky draw, no
`flaky` markers. 146 at the start of Phase 1, 46 added there, 39 in Phase 2, and 24 in the Phase 2 audit-correction pass.

| File | Tests | What it pins down |
|---|---:|---|
| `test_real_player_input_schema.py` | 24 | **NEW (inventory phase).** the real-player contract's shape: nulls mean *not projected*; a missing scoring category can only be recorded as absent; the expert grades are pinned as non-distributional; raw source fields are required and non-empty; sources are identified by content hash; and a fabricated two-player fixture validates, one of them deliberately sparse |
| `test_realdata_ingestion.py` | 87 | the ingestion layer on fabricated data: scoring arithmetic, median metadata, both availability readings, injury field separation, fumble exclusion, missing categories absent, identity matching and its four failure modes, refusal of the synthetic pool, schema failures, vendor-total and grade-to-variance prevention, determinism |
| `test_playerspec_mapping.py` | 50 | the mapping on fabricated data: no assumption outside the config; the level solve is exact and proportional; both targets supported and agreeing under symmetry; injury solved against both full-season targets with the week/game distinction and infeasibility reported; the variance split preserves total dispersion; unsupported fields are named placeholders; coverage bands; aliases carry reasons; deterministic |
| `test_auction_state.py` | 55 | **NEW.** the immutable room: budget and capacity from the league, the legal-maximum formula, the $1-per-slot reserve, immutable transitions, duplicate and over-capacity and unaffordable rejection, per-owner bid capacity, nomination and bidding, state reconciliation, and feasibility as a matching question with no quarterback maximum |
| `test_auction_completion.py` | 43 | **NEW.** the cost contract's four provenance levels and its refusal to default a missing price; the availability proxy; the beam against an exhaustive oracle; budget, uniqueness and lineup feasibility of every completion; retained constructions across spending levels; heuristic labelling; unresolved finalists kept co-best |
| `test_auction_counterfactual.py` | 34 | **NEW.** the pass branch cannot re-buy the candidate; money moves the right way; rival identity changes the simulated league; a full rival cannot receive him; price ladders stay legal; robust and permissive definitions checked against hand-built verdicts; monotonicity classified; eight cache dimensions kept apart |
| `test_scenario_band.py` | 40 | **NEW.** the 54-cell grid is the full cross-product in a fixed order; ids round-trip and cannot disagree with their config; identical arms give exactly zero in every cell; band arithmetic, sign stability and dominant axis; the slot-swap selection rule |
| `test_auction_cli.py` | 23 | **NEW.** every `ce-lab auction` command runs; six anticipated user errors exit 2 without a traceback; committed examples are sanitized; no output claims a bid |
| `test_qb_and_handcuffs.py` | 14 | **NEW.** a skill player fills the superflex; at most two QBs ever start; QB3 value moves with absence risk and flips with roster context; committee backs are valued as their share; insurance measured as variance reduction; a grep-level ban on positional premiums |
| `test_auction_invariants.py` | 10 | **NEW.** Hall constraints checked against a brute-force bipartite matcher on every count vector; fifty randomised auctions with every money invariant asserted after every purchase; cache isolation; repository hygiene |
| `test_auction_audit_fixes.py` | 52 | **NEW.** the eight foundation-audit findings: ids and cost books and settings and states all fingerprint by content; a bid is validated against the player on the block; a controlled fixture where the proxy-best roster is not the equity-best; selection versus holdout samples; sparse ladders bracket rather than name a frontier; refinement walks every integer; an upward price step is classified as a defect, never as economics; exactness reported per stage; intervals labelled pointwise; and a scenario pipeline whose specs genuinely differ |
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

> **SUPERSEDED.** This section framed the band as a *gate* on whether a
> best-alternative solver was worth building. That was wrong and has been
> corrected: opportunity cost exists under every scenario in the grid, and a
> wide band changes how a value is **presented**, not whether the question has
> an answer. Passing on a player still means spending the money on someone. The
> band governs ranges-versus-points, nothing more. The experiment was run and
> the solver was built; see §16 below for both.

* **Band narrower than the player-to-player gap** → the assumptions do not
  prevent ranking, and relatively tight values can be quoted.
* **Band wider** → the honest output is a price *range* carrying the assumption
  band, with `availability_interpretation` named as the dominant term. Either
  the vendor clarifies the health treatment or a weekly/multi-season feed is
  obtained to estimate the three uncalibrated scenario parameters.

The band was measured as **wide**. Both bullets above are about presentation.

---

## 16. The auction best-alternative foundation

Built on branch `auction-best-alternative-foundation`. Full detail in
`docs/AUCTION_LAYER.md`; this is the summary.

> **Every runnable example in this layer uses fabricated inputs.** The demo
> pool, its projections and its acquisition costs are generated by a formula.
> Nothing produced by them is a real player value or a market price.

### What was built

```
src/ceauction/realdata/
  scenarios.py       the 54-cell model-scenario cross-product; paired bands
  slot_swap.py       the committed scenario-band experiment
src/ceauction/auction/
  feasibility.py     can this roster still field a legal lineup?
  state.py           the immutable auction room
  costs.py           the acquisition-cost contract and its provenance
  proxy.py           availability-aware roster strength (expected points)
  completion.py      the bounded roster-completion search + exact oracle
  counterfactual.py  buy versus pass, with an explicit pass destination
  reservation.py     the integer-price search and the reservation range
  demo.py            the fabricated auction every command runs on
  cli_commands.py    ce-lab auction …
```

### The 54-cell experiment

`docs/example_slot_swap_54cell.txt`, 16,000 seasons per arm, 23 minutes.

| Band | Range | Width | Resolved |
|---|---|---:|---|
| marginal starter slot | `[-0.02038, -0.00331]` | 0.01706 | **54/54** |
| bench slot (15th man) | `[-0.00250, +0.00781]` | 0.01031 | 15/54 |
| adjacent alternative | `[-0.00550, +0.00331]` | 0.00881 | 15/54 |

Direction is robust where it matters — downgrading the last non-QB starting
slot costs CE in every cell. Magnitude is not: the same effect varies sixfold,
and the band is 1.9× the adjacent-player gap at the bench slot and 3.1× at the
starter slot. **Verdict: WIDE.** Values are reported as scenario-dependent
ranges naming `availability_interpretation`, which is the dominant assumption in
all three bands.

Two comparisons on one roster. This does **not** establish that every ranking on
the board is robust.

### The three prices, kept apart

**CE reservation price** is built here. **Expected clearing price** and
**tactical winning bid** are not, and no output may present itself as either. A
test walks every command's output for "recommended bid", "opening max", "market
price" and "real value", and fails on any occurrence not inside an explicit
denial.

### Measured runtimes

| Operation | Median |
|---|---:|
| state validation / legal maxima / fingerprint | ≤0.001s |
| completion search, proxy only | 0.43s |
| completion search + CE selection + holdout (4 finalists) | 4.75s |
| buy/pass, unavailable | 11.2s |
| buy/pass, named rival (rival re-completes) | 18.3s |
| reservation, 1 scenario × 6 prices | 71.0s |
| 54-cell slot swap, 16,000 seasons | 23 min |
| *projected:* full 54-scenario × 12-price reservation | ~2 hours |

**Nothing here is live-capable.** A 10-second bid timer would need a precomputed
table, not this search.

### The foundation audit

Eight findings were repaired on `auction-foundation-audit-fixes`; see
`docs/AUCTION_LAYER.md` and the commit messages. The three that changed
*numbers* rather than labels:

* **buy/pass selected completions by the expected-points proxy**, not by
  equity, so it computed `CE(proxy-selected completion)` while claiming
  `CE(best modelled completion)`. Both branches now select by equity and report
  on an independent holdout sample, which costs about 4.4× the runtime;
* **sparse price ladders were reported as reservation prices.** Testing $20 and
  then $50 now yields a *bracket* — at least $20, below $50, with $21–$49 named
  as never evaluated — and `--refine` will go and evaluate them;
* **an upward step in delta CE was described as real economics.** It cannot be:
  every roster affordable at $p+1 was affordable at $p. Such a step is now
  classified as noise, search instability, CE-selection instability, or a bug.

### Simplifying assumptions

1. The other eleven rosters are held fixed while one owner is optimised.
2. Acquisition costs are an input; none exist for this league.
3. The completion search is a bounded beam, labelled heuristic everywhere.
4. The candidate pool is a real cut — a player below it cannot be chosen.
5. The stage-1 proxy is expected points, not equity, and never called value.
6. No in-season waivers, FAAB, trades or post-draft replacement.
7. No behavioural model of who would bid.
8. In the demo, the scenario axis exercises the API rather than moving the
   players: the fabricated pool is identical under every scenario because there
   is no contract to re-map. Real specs arrive as one state per scenario, and
   `tests/test_auction_audit_fixes.py` exercises that path with scenarios that
   genuinely alter the specs.
9. Every interval is pointwise 95%. A simultaneous band over prices and
   scenarios is not implemented; a conservative Bonferroni alternative is
   offered and labelled conservative.
10. A rival's continuation is optimised for his own equity against our
    *proxy-best* roster, because our real one is not determined until after he
    has bought.

### Recommended next phase

**Simultaneous multi-owner completion, and then opponent bidding behaviour.**

The single largest simplification in this layer is that the eleven other rosters
are frozen. That makes every reservation price an answer to "best against *this*
league" rather than "against a league still drafting", and it is the assumption
most likely to move the numbers, because the players our search wants are
exactly the players the other eleven searches want.

Concretely: extend the completion search so every owner with open slots
completes against a shared board, iterating to a fixed point (or a bounded
number of rounds) rather than one owner against a frozen cast. Measure how far
reservation prices move between the frozen-cast and shared-board versions on the
fabricated auction first, since that difference is the size of the error the
current simplification carries.

Only after that does opponent *bidding behaviour* become worth modelling, and
only after that can a CE reservation price start to become a tactical bid.

**One thing the foundation audit changed about this plan: budget it for the new
runtime.** A buy/pass comparison is now 11 seconds rather than 2.5, because
both branches genuinely select by equity and report on a holdout. Multi-owner
completion multiplies that by the number of owners still drafting and by the
number of rounds to a fixed point. Before writing it, measure one iteration on
the fabricated auction and decide how many rounds are affordable -- a design
that needs ten rounds over twelve owners at eleven seconds a comparison is a
different piece of work from one that needs two.

**Explicitly still not next:** dollar values for real players, opening or live
max bids, auction inflation, or anything presented as a market price. All of
them are blocked on the acquisition-cost data this repository does not have.


---

## 17. The market prior: provisional acquisition cost

Built on branch `auction-market-prior`. Full detail in `docs/MARKET_PRIOR.md`.

> **Every committed example uses fabricated anchors.** The real Sleeper export
> is an ignored `local_data/` path and no player-level output from it is in
> version control.

### The gap this fills

Acquisition cost was the missing input: the completion search needs a price for
every player it might buy, and this league has no auction history. Sleeper's
generic 2026 `2qb` value list is the only anchor available — and it is an
anchor, not a price.

### Four quantities, and only two of them exist

| | Built here? |
|---|---|
| `sleeper_display_anchor` — what managers see | **yes**, preserved exactly |
| `expected_clearing_price` — what this room may pay | **yes**, provisional |
| `ce_reservation_range` — what we can afford | elsewhere; unchanged |
| `tactical_max_bid` — what to bid | **nowhere** |

### What was built

```
src/ceauction/market/
  anchors.py       load and validate the Sleeper CSV; identity join
  prior.py         format credibility, budget reconciliation, price bands
  live.py          partially pooled updates from observed sales
  pressure.py      per-owner ability, legality, fit and evidence
  costbook.py      populate the acquisition-cost contract
  demo.py          fabricated anchors and a scripted sale sequence
  cli_commands.py  ce-lab market ...
```

### Real-file aggregates

1,024 rows; **149 priced, rosterable, active anchors** totalling **$2,384.26**
raw and **$2,379** displayed; 6 positive players the league cannot roster
(five defenses, one inactive QB). Positional raw totals reproduce the stated
source facts exactly: QB 29/$541.12, RB 45/$802.77, WR 56/$828.61, TE
19/$211.76. **All 149 priced anchors join to the real contract.**

### Why a reconciliation is mandatory

$2,400 room, 180 slots, $180 committed at the $1 floor, **$2,220
discretionary** — against $2,379 of priced anchors *before* the other 31 slots
are filled. Paying list is arithmetically impossible, not merely unlikely.

Reconciled boards: low $1,957, base $2,107, high $2,213, each within integer
rounding of its scenario target.

### Positional outcome

TE moves furthest from its list price (0.736 of displayed), then QB (0.855), WR
(0.885), RB (0.946) — out of the lineup graph, not a table of preferences. This
league has no dedicated TE slot and its superflex accepts an RB, so both label
mismatches in a generic `2qb` list are represented as **credibility**, a weight
that evidence can move in either direction, rather than as a hand-coded
discount that could never be wrong.

### Simplifying assumptions

1. Every coefficient is a **stated scenario**: spend rate, anchor adherence,
   four credibility weights, learn rates, prior strengths, outlier cap. Nothing
   is fitted; no auction history exists.
2. Budget reconciliation is a single monotone scale per scenario. A
   replacement-level transformation was considered and rejected for needing a
   baseline nothing here measures.
3. Buyer-level learning needs two purchases before it characterises anyone.
4. Roster fit is structural — unfilled seats, not preferences.
5. Price tiers are four coarse buckets.
6. Nonpositive anchors are *unpriced*, not $1 sales; Sleeper's UI rule for them
   was never observed.
7. Room-wide effects require sales spanning more than one position.

### Recommended next step

**Record the first live auction with `observe-sale`, then compare the observed
clearing prices against the low/base/high bands.** That is the first real
evidence about the six coefficients currently chosen rather than estimated. One
auction will not settle them, but it converts them from assumptions into
assumptions with a residual.

A tactical max bid still needs a bidder model, simultaneous multi-owner
completion, and a runtime budget that fits a bid timer. None of the three should
be attempted before the band has been checked against a real room even once.

---

## Phase: the tactical bidding layer

Built on branch `auction-tactical-bidding`, from `54506f4`. Full detail in
`docs/TACTICAL_LAYER.md`; a worked fabricated auction is in
`docs/TACTICAL_WALKTHROUGH.md`. This section is the summary.

**This is a foundation, not a draft-ready bidder.** It is honest about who gets
the player if we stop, and it keeps the four prices apart. It has not been
checked against a single real auction, its bidder scenarios are stated rather
than fitted, and its audited path can go degenerate on a fully-allocated shared
board. Do not take a number from it into a live room without reading the
"still provisional" list below.

### What was added

`src/ceauction/tactical/`

| module | what it answers |
|---|---|
| `endgame.py` | pure room arithmetic: candidate-specific legal maxima, financial-control threshold, who falls out, whose leverage is illusory. No simulation. |
| `bidders.py` | five named bidder scenarios and a willingness **range** per owner per candidate, capped by legality. |
| `recipients.py` | named pass-recipient branches, evaluated one at a time; illegal named recipients refused, never substituted. |
| `board.py` | shared-board continuation: one pool, no duplicates, legal completions, $1 per open slot, seeded and deterministic. |
| `maxbid.py` | robust / base / permissive tactical maxima over a price ladder, immediate (proxy) and audited (CE) paths, and the cache key. |
| `precompute.py` | bounded precomputation, resume only on an exact key match, refuses real output outside `local_data/`. |
| `demo.py` | one fabricated world where the auction pool and the market prior describe the same players. |
| `walkthrough.py` | regenerates `docs/TACTICAL_WALKTHROUGH.md`. |
| `cli_commands.py` | `ce-lab tactical …`. |

### Measured runtime (fabricated demo, this machine)

| stage | seconds |
|---|---|
| endgame arithmetic | 0.007 |
| bidder willingness, whole room | 0.007 |
| named recipient branches | 0.010 |
| shared-board continuation | 0.135 |
| immediate max-bid, cold (3 scenarios x 6 prices x 4 recipients) | **7.56** |
| immediate max-bid, cache hit | **0.022** |
| audited comparison, one price, one scenario | **13.25** |

The 10-second target is **met but not comfortably** by the cold immediate path
at default settings, and met with three hundred milliseconds to spare only
because the ladder is short. The cache-hit path is 22ms and is the one that
should actually be used live. The audited path must be precomputed.

### Simplifying assumptions

Every one of these is stated in `docs/TACTICAL_LAYER.md` with the field that
carries it into output:

1. Clearing price = second-highest willingness + one increment, capped. An
   **approximation**, not Sleeper's mechanism.
2. Opponent continuations use a named **proxy**, never championship equity.
3. Recipient weights are stated assumptions applied *after* the branches exist.
4. Immediate-mode value is our completion's expected starting points minus the
   league mean. A proxy ordering with no interval; not equity.
5. A bounded scarcity term (+0-20%) and a per-scenario soft budget share.
6. `pool_depth` and `max_allocations` are real cuts on the continuation.
7. Sparse ladders give brackets; only `--refine` walks integers.

### Still provisional

* **No real auction has been observed.** Every bidder coefficient is chosen.
* The **audited path can be degenerate**: when the shared board allocates the
  whole pool, buy and pass can yield an identical completed league and the
  paired difference is exactly zero. Reported as `unresolved` with `DEGENERATE`
  in the basis, but it means the audited comparison currently needs the board to
  leave our completion real choices. This is the largest known weakness.
* Only the **fabricated** world is wired in. There is no loader for a live room.
* Nonmonotonicity detection is implemented and tested structurally; the
  fabricated board has not yet produced a real instance.

### Recommended next step

**Make the audited comparison non-degenerate.** Reserve a bounded slice of the
board from the shared continuation (or complete our roster *before* the last
rival allocations) so the buy and pass branches genuinely differ, then re-run
`ce-lab tactical max-bid --mode audited` and confirm the intervals are non-zero.
Until that is fixed, the CE-backed path is architecture rather than evidence and
only the proxy ordering is usable.

---

## Phase: the CE counterfactual fix

Branch `tactical-ce-counterfactual-fix`, from `1360649`. One defect, correctly
diagnosed this time, plus the honest consequence of fixing it.

### The defect

The previous handoff called the audited path "degenerate: buy and pass produced
an identical completed league". **That diagnosis was wrong.** The rosters were
different. What was actually happening:

```
our completion's proxy strength     86.9
rival proxy strengths          105.3 - 109.4   (all eleven)
our championship equity          0.0000        in BOTH branches
audited delta                    0.000000  se 0.000000
```

The shared-board continuation ran with `include_focus=False`, so eleven rivals
drafted the entire top of the board **against a focus seat that never bid**.
They never had to outbid us, so they got better players for less money, and our
CE-backed search was handed the leftovers. We entered every comparison already
last by twenty proxy points, our equity sat on the floor in both branches, and
the paired difference could not move off zero.

### The fix

`BoardSettings.focus_bids` (default `True`, replacing `include_focus`). The
focus team now competes in the continuation like everyone else, against a shadow
ledger of its own money and slots that applies the same $1-per-open-slot reserve
and the same feasibility test.

It does **not** take delivery. Players it outbids the room for are *held*: kept
out of rival rosters, left on our board, unpaid for, and never charged against
our budget twice. Which of them we actually take stays with the CE completion
search, because letting a willingness proxy settle that would replace the real
search with the cheap one.

```
our proxy strength   86.9  ->  105.7   (rivals 105.1 - 106.7)
audited delta        0.000000 se 0.000000  ->  +0.045000 se 0.010378
                                              CI (0.0247, 0.0653) favorable
```

### The consequence, stated plainly

**The tactical maxima dropped a long way, and that is the point.**

```
opening robust max   $66  ->  $1   (bracket (1, 34))
live robust max      $70  ->  $13  (bracket (13, 42))
candidate's expected clearing band                ~$19-21
```

The old numbers were implausibly high — a robust maximum of $70 out of a $99
legal max, for a mid-tier back the market prices near $20 — precisely because
the old model let us keep the value of the rest of the board no matter what we
paid. Now spending $70 on one player visibly costs us the continuation, which is
what spending $70 actually does. The new numbers sit near the market band, which
is where a sane answer belongs. They are still proxy-mode brackets on a sparse
ladder, not recommendations.

### Tests

Four new regression tests (65 total in `tests/test_tactical.py`):

* the focus team bids and holds part of the board;
* held players reach no rival roster, stay on our board, and cost us nothing;
* the shadow ledger respects budget, slots and the $1 reserve;
* **the regression itself**: with `focus_bids=False` our completion lands more
  than 10 proxy points below the field, and bidding closes most of that gap.

The slow audited test now additionally refuses a zero-delta/zero-SE result and
asserts at least one verdict carries a real standard error.

### Still provisional

* No real auction has been observed. Every bidder coefficient is still chosen.
* Only the fabricated world is wired in; there is no live-room loader.

### Checked before handing off: the pass branch does lift off the floor

```
sims=  400  ce_buy=0.04500  ce_pass=0.00000  delta=+0.04500 se=0.01038  favorable
sims= 4000  ce_buy=0.06300  ce_pass=0.00200  delta=+0.06100 se=0.00387  favorable
```

The `0.0` pass branch at 400 sims was Monte Carlo resolution (0 titles in 400
seasons), not a structural floor. At 4,000 it resolves.

**But look at the magnitudes, because they are the next problem.** League
average CE is `1/12 = 0.083`. Our completion sits at `0.002` without the
candidate and `0.063` with him — one mid-tier back, bought for $13, moves us
from bottom of the league to just under average. Championship equity really is
convex in roster strength near the playoff cutoff, so a large marginal effect is
not automatically wrong; a swing this large from one cheap player is still more
than that convexity comfortably explains.

**Exact next step: find out whether our completion is sitting on a knife-edge.**
Sweep the candidate's price (or drop him for the next-best alternative) at 4,000
sims and plot `ce_buy`; if equity collapses toward zero the moment he is
removed, the continuation is leaving our roster balanced exactly at the cutoff
and the shared board is still handing rivals too much. Until that is understood,
treat the audited delta as directionally right and its *size* as unverified.

---

## Phase: joint allocation and conservation

Branch `tactical-joint-allocation-fix`, from
`d86e2da6be1eb4f2a8c056b5eba1e179952cb583`. Full sweep in
`docs/TACTICAL_JOINT_SWEEP.md`.

### Free shadow blocking existed. It was 6 players.

The previous branch's shadow ledger was audited before being accepted, and it
failed:

```
shadow budget        $110 -> $2        shadow slots  12 -> 0
shadow-won players   12  ($108 notional)
  of those we bought  6
  of those we did not 6  ($18 notional)
actual focus cost    $101              cost divergence $7
unselected reaching a rival   0
unselected left undrafted     6
FREE BLOCKS          6  ($18 of denial we never paid for)
```

We held twelve mutually exclusive completion options, bought six, paid for six,
and denied the other six to eleven opponents at zero cost. Rivals were never
offered them back. That inflated our roster (we picked from a protected
shortlist), inflated our denial value, and deflated every rival.

Answers to the six audit questions: (1) yes, six were excluded from every final
roster; (2) yes, we blocked without paying or rostering; (3) yes, shadow spend
$108 vs actual $101; (4) rivals completed against the **larger shadow-held
set**, which is the defect; (5) yes, changing an unselected hold changed
opponent rosters and our CE; (6) six free blocks in the committed fabricated
example.

### The correction: conditional reallocation

`tactical/joint.py`. Per focus finalist: buy exactly what it names at cost-book
prices out of our real budget, return every unselected hold to the board, then
recomplete the eleven rivals against that. `validate_joint_world` refuses any
world failing the conservation invariant, and nothing unvalidated is evaluated.

A second, smaller hole surfaced and is now explicit: **the `unavailable` pass
branch is not conservation-neutral.** Withdrawing a player denies him to eleven
rivals at nobody's expense. It must now be `declared_withdrawn`; undeclared, it
is refused as an unpaid reservation. It is reported but is not a fair comparator
against a branch where somebody pays.

### Reconciliation (fabricated demo, every row of the sweep)

```
pool 310 = 135 initially owned + 1 branch + 12 focus bought
         + 33 rival bought + 129 undrafted            balances: True
dollars $2400 = $1600 paid + $800 remaining           balances: True
duplicate owners none | over budget none | reserve violations none
illegal rosters none  | wrong size none | unpaid reservations none
```

Returned to board 6, reclaimed by rivals 2, the rest genuinely undrafted and
available to everyone.

### Proxy strengths

```
focus-silent (2 branches ago)   ours  86.9   field 105.3-109.4
shadow-hold (previous branch)   ours 105.7   field 105.1-106.7  [6 free blocks]
joint reconciled (this branch)  ours 104.9   field 105.5-106.9  [0 free blocks]
```

Removing the free blocks cost us ~0.8 proxy points, which is the size of the
advantage we were taking for nothing.

### The measured effect, corrected

4,000 sims, six prices, buy vs pass to Owner04, paired SE. Full table in the
sweep doc.

```
  $1  ce_buy 0.04275  ce_pass 0.01950  delta +0.02325 +/-0.00698  favorable
  $5  ce_buy 0.05375  ce_pass 0.01950  delta +0.03425 +/-0.00756  favorable
 $10  ce_buy 0.05425  ce_pass 0.01950  delta +0.03475 +/-0.00747  favorable
 $13  ce_buy 0.04950  ce_pass 0.01950  delta +0.03000 +/-0.00721  favorable
 $20  ce_buy 0.04850  ce_pass 0.01950  delta +0.02900 +/-0.00718  favorable
 $30  ce_buy 0.02375  ce_pass 0.01950  delta +0.00425 +/-0.00590  UNRESOLVED
```

The previous branch's `0.002 -> 0.063` swing was an artifact of unpaid blocking.
The honest figure at $13 is `0.0195 -> 0.0495`, delta `+0.030`, **half the
previously reported effect**. League equity sums to exactly `1.000000` at every
price; mean `0.08333`. We rank **12th of 12 at every price**, bought or passed.

Six prices produced six distinct allocation fingerprints, so CE moved only where
the allocation moved. The sweep is **not monotone** ($1 < $5 < $10 then falling)
— allocation instability, reported not smoothed.

### Where the swing comes from, quantified

At $13 vs passing to Owner04: passing costs **more** ($102 vs $89) and buys
**less** (101.7 vs 104.9 proxy). Five players differ between the arms. Field
mean moves only `+0.27` across eleven teams, so denial value is real but small.
The amplifier is playoff-cutoff convexity — we are last in both arms, where
proxy points convert to championship probability steeply. Allocation instability
contributes and is not separable at this sample size.

### Corrected tactical thresholds

```
CE-audited robust maximum   $20      resolved
robust bracket              ($20, $30)
base maximum                $20      resolved
permissive ceiling          $30      unresolved vs Owner04; a CEILING
expected clearing band      $17/$19/$21   proxy-only, market layer
previous $1 opening max     WITHDRAWN     invalidated by conservation
previous $13 live max       WITHDRAWN     invalidated by conservation
```

$1 and $13 are **withdrawn, not revised**: they were computed over worlds where
we held six unpaid players.

### Still provisional

* No real auction observed; every bidder coefficient is still chosen.
* We rank 12th at every price. Our completion is genuinely the weakest roster in
  the fabricated league, so every delta is measured near the floor where
  convexity is largest. This is the main reason not to trust the *magnitudes*.
* The sweep is non-monotone; the finalist set moves with the shadow clearing.
* Only the fabricated world is wired in.
* `tactical/maxbid.py` still runs the **old** proxy/audited path. The joint
  machinery is not yet wired into `ce-lab tactical max-bid`.

### Exact next step

**Wire `joint.py` into `maxbid.py`** so `ce-lab tactical max-bid --mode audited`
produces reconciled worlds, then re-derive the robust/base/permissive ladder from
CE rather than the proxy. Until that is done the CLI still reports the withdrawn
proxy numbers.

---

## Phase: audited max-bid over nested opportunity sets

Branch `tactical-audited-maxbid`, from
`d522235e2f52d41ebdc5690e568a47f1f7cc1c26`. Detail in
`docs/TACTICAL_REGIMES.md` and the module docstrings of `tactical/nested.py`.

### Proven cause of `$1 < $5`: beam path dependence

Not economics, and not any of the market/seed/cache explanations:

```
best construction found at $5   cost $95
our budget at $1                $109      -> affordable
players of it taken by a rival  none      -> legal
shadow held @$1  [..., 271]
shadow held @$5  [..., 93]     symmetric difference: {93, 271}
```

The $5 construction was legal and affordable at $1 and the search **never found
it**. The shadow continuation depends on our own remaining money, so a $4
difference changed which two players we shadow-won, and a two-player difference
in the beam's input board sent a bounded beam down a worse path.

Ruled out by direct test, all now regression-tested: market fingerprint constant
across prices; rival budgets identical to pre-buy at every price; board seed
fixed; `PlayerSpec` carries no price so price cannot enter scoring; cache keys
are mode-prefixed.

### The fix: one nested opportunity set per ladder

`tactical/nested.py`. Generate completions at every price, union them, re-offer
the union to every price, keep what is legal and affordable there. Feasibility
nesting then holds by construction, and audited mode **refuses** a ladder that
is not nested.

```
      $   budget  generated  inherited  feasible  unafford
      1      109          3         14        17         0
      5      105          3         14        17         0
     10      100          3         14        17         0
     13       97          3         14        17         0
     20       90          3         12        15         2
     30       80          3          7        10         7
nesting violations: 0
```

### Three monotonicities, kept separate

* **Feasibility** — exact. `feasible($5) ⊆ feasible($1)`, tested directly.
* **Selection objective** — exact here, and for a reason worth stating: a rival
  continuation depends only on *which players we took*, never on what we paid
  (rivals bid with `focus_bids=False` against the board we leave). So the same
  construction yields the same joint allocation at every price, and the best
  available at $1 is at least the best available at $5.
* **Holdout estimate** — not forced. `favorite` still wiggles
  (0.09050/0.08775/0.08775/0.09025) inside a ±0.009 interval. Reported, not
  smoothed.

### Audited ladder, corrected

`--mode audited`, 800-season holdout, four recipients:

```
 $13  Owner07 +0.02625 ±0.0152  favorable
 $13  Owner04 +0.02875 ±0.0144  favorable
 $13  Owner02 +0.03375 ±0.0143  favorable
 $13  unavail +0.03000 ±0.0146  favorable
 $56  all four  -0.011 to -0.019  unfavorable
 $99  all four  -0.011 to -0.019  unfavorable
robust $13   base $13   permissive $13   bracket (13, 56)   legal max $99
```

### Roster-strength regimes: the headline result

Budget alone was tried first and **does not work** — $28 to $179 moved our
finished proxy by 0.2 points and left us 12th in every case. The board binds,
not the wallet. The regimes therefore vary how much roster we start with. That
failed fixture is documented in `regimes.py` because it is a real finding.

```
regime         pre-owned  our proxy  field mean  our CE   rank
underdog           2         92.5      106.89    0.00000   12
bubble             3        102.4      106.29    0.02325   12
bye_contender      8        106.6      106.07    0.08575    6
favorite          11        106.8      106.01    0.08200    7

delta at $1:
underdog       +0.02050 ±0.00439  FAVORABLE
bubble         +0.02325 ±0.00745  FAVORABLE
bye_contender  -0.00075 ±0.00864  unresolved
favorite       +0.00850 ±0.00903  unresolved
```

**The candidate is worth most to teams that need him and is not resolvably
worth anything to teams that do not.** Not predetermined: `bye_contender` came
out very slightly negative. Valuing this player from one roster context and
calling that his value would have been wrong in both directions.

Within `underdog`, `bubble` and `bye_contender` the delta is **identical at
every tested price** — the nested set makes the same construction available
throughout, so the same joint world is selected and the same seasons give the
same equity. Before this branch the same sweep moved 0.033 across prices purely
through beam path dependence.

### Recipient identity (`bubble`)

```
Owner12 pays $22 -> our CE 0.02325, his 0.13350
Owner02 pays $21 -> our CE 0.01825, his 0.12900
```

Separate branches, separate allocations, no averaging.

### Benchmarks

```
nested candidate generation (4 prices)     0.920s
audited: 1 price, 1 recipient              6.578s
audited: 4-price ladder                   11.573s
audited: cache hit                         0.022s
proxy mode: 4-price ladder                 1.427s
```

Audited is precomputation, as intended. The cached lookup is 22ms.

### Still provisional

* No real auction observed; bidder coefficients remain chosen.
* The nested set is a union of **bounded beam searches**. Nesting is exact over
  the evaluated set, not over all legal rosters, and `exactness` says so.
* `underdog`'s pass arm sits at CE exactly 0.0 — a genuine floor, so its delta
  is measured against a boundary.
* Only the fabricated world is wired in; there is no live-room loader.
* Regime fixtures vary pre-owned roster size, which also changes our budget;
  the two are not separated.

### Exact next step

**Separate roster quality from budget in the regime fixtures** — hold
`budget_remaining` constant across regimes by adjusting `spend_per_player`, so
the regime comparison isolates roster strength. Then re-run
`regime_experiment` and confirm the ordering survives.

---

## Phase: controlled roster-context isolation

Branch `tactical-context-isolation`, from
`6150391273b644d12090ecde99fdd5f5745c82ad`. Detail in
`docs/TACTICAL_CONTEXT.md`.

### The previous regime conclusion is withdrawn

`docs/TACTICAL_REGIMES.md` varied pre-owned count, money, open slots, the
remaining board, which players rivals could reach, and every rival's completion
— four factors at once. Its labels did not match its outcomes (`bye_contender`
6th, `favorite` 7th). It is marked CONFOUNDED in place and retained as history
only.

### The controlled design

`tactical/context.py`. One fabricated auction; four regimes; the **only**
difference is the `base_mean` of the eight PlayerSpecs the focus team already
owns. Those players are off the board in every regime, so scaling their scoring
cannot reach the auction.

```
structural differences across all pairs of 4 regimes: 0
```

covering focus owner, budget, spend, roster size, open slots, player ids,
positions, prices, remaining-board ids, remaining-board costs, every rival's
roster/spend/budget/slots, market fingerprint, cost-book fingerprint, league
settings, withdrawn set, cast, candidate, price, recipient, leader and seeds.
`check_structural_equality` refuses the experiment if any of them moves, and
three tests plant a budget / board / rival difference to prove it fires.

Expected differences reported separately: pool fingerprints differ (the factor);
**every projection outside the focus roster is byte-identical**, asserted
spec-by-spec.

### Calibration — labels earned from simulated outcomes

```
regime           scale  weekly  wins   p(playoff)  p(bye)      CE   rank
underdog          0.88   102.6   8.7        0.040   0.004  0.0022     12
playoff_bubble    0.99   112.2  13.6        0.456   0.142  0.0707     12
bye_bubble        1.06   118.6  17.1        0.833   0.496  0.2427      1
favorite          1.16   127.4  21.3        0.992   0.915  0.5305      1
```

Bubbles are genuine coin-flips on the boundary they name. `underdog` is not
pinned at zero, so a marginal effect had room to appear.

### The result: NULL

```
regime            delta at $1     95% half-width   |delta|/SE
underdog           -0.00050          0.00183          0.54
playoff_bubble     -0.00300          0.00790          0.74
bye_bubble         +0.00525          0.01288          0.80
favorite           +0.00750          0.01276          1.15
```

**Every arm is unresolved at 4,000 holdout seasons, at every tested price
($1/$13/$20/$30).** The context effect is *not* statistically distinguishable.

The sign pattern (negative for weak contexts, positive for strong) is the
**opposite** of the confounded experiment's conclusion. That conclusion does not
survive isolation — and neither does its reverse, because nothing resolves.

Resolving the `playoff_bubble` point estimate would need roughly **27,000
holdout seasons**, about 7x this run.

Invariants all held: league CE sums to 1.0 in every arm; nesting exact; joint
worlds validated; price changed neither the focus completion nor the rival
allocation within a regime (nested sets working as designed).

### Remaining limitations

* The candidate is weak — with 8 pre-owned and 11 full rival rosters, the best
  player left is marginal. A stronger candidate might resolve where this does not.
* One candidate, one recipient, one fabricated board. A null on one player is
  not a null on player valuation.
* The regimes are calibrated against a weak fabricated field: `bye_bubble` ranks
  1st on CE while sitting on a genuine bye coin-flip.
* No real auction observed; bidder coefficients remain chosen.

### NO-GO for real-board precomputation

The machinery is sound — structurally controlled, conservation-clean, nested,
labels earned. But it has not yet demonstrated that it can *measure* a player's
context-dependent value at an affordable sample size. Precomputing real-board
prices on a signal indistinguishable from noise would produce confident-looking
numbers with no evidence behind them.

### Exact next step

**Re-run the controlled experiment with a materially stronger candidate** —
reduce `N_PREOWNED` or `rival_keep` so a genuinely valuable player remains on
the board — and check whether any regime resolves at 4,000 seasons. If the
effect resolves there, the null above is about this candidate; if it does not,
it is about the estimator's power, and that must be fixed before any real-board
work.

---

## Phase: estimator power by candidate tier

Branch `tactical-signal-power`, from
`b85c07bf8eeb1dc6781626afa3ce265c1e4da765`. Detail in `docs/TACTICAL_POWER.md`.

### The previous NO-GO was too broad, and is corrected

The context experiment's null was a weak candidate's true-near-zero effect, not
an estimator failure. Weak-player irrelevance and estimator failure look
identical in one cell and are opposite findings. Corrected verdict: **GO** for
targeted real-board precomputation.

### Candidate-tier calibration (one factor: `base_mean`)

Position, bye, week variance, injury hazard and availability untouched. Labels
earned from measured weekly starting-lineup improvement over a $1 replacement,
against a competitive fourteen-slot lineup.

```
tier               scale  base_mean  weekly improvement  starts
bench               0.55       6.07                0.00   False
marginal_starter    1.15      12.70                1.50    True
strong_starter      1.85      20.42                8.47    True
elite               2.90      32.02               18.92    True
```

### Experiment A — power at a fixed $13, 4 tiers x 4 contexts

```
tier               resolved   delta range        |d|/SE range
bench                 0/4     +0.000 .. -0.009    0.05 - 1.62
marginal_starter      4/4     +0.003 .. +0.029    2.53 - 5.40
strong_starter        4/4     +0.051 .. +0.280   14.22 - 32.62
elite                 4/4     +0.352 .. +0.572   46.58 - 68.91
```

`bench` failing to resolve is correct: its measured lineup improvement is
**0.00**, so there is nothing to detect.

### Required seasons by target half-width

Reporting choices, not materiality claims. Example (`elite`/`playoff_bubble`):
0.010 → 9,590; 0.005 → 38,359; 0.0025 → 153,434. Those numbers look large only
because the target is absurd for a 0.54 effect — the achieved half-width at
4,000 seasons is ~0.016, resolving it thirty times over.

### Pilot / confirmatory

Pilot 1,000 seasons (seed 20260904) estimates variance only; confirmatory 4,000
(seed 917324011) reuses no pilot observation; cap 40,000 →
`UNDERPOWERED_AT_CAP`. Different seeds are enforced by a raise.

### Allocation seeds (3 seeds: 20260906 / 424242 / 987654321)

```
tier               deltas                      between-SD  within-SE  SD/effect  sign stable
marginal_starter   0.02675 0.02925 0.03500       0.00423    0.00493      0.157      True
strong_starter     0.23850 0.24975 0.24250       0.00570    0.00729      0.023      True
elite              0.54000 0.55775 0.54700       0.00894    0.00802      0.016      True
```

`elite` trips `dominated_by_allocation` (allocation noise exceeds season noise,
so more seasons are wasted there) but **not** `dominates_effect` (0.0089 against
a 0.547 effect is 1.6%). Those are different questions; conflating them would
reject an effect sixty times larger than its own instability.

### Runtime

~7s per tier x context cell at 4,000 seasons; ~27.5s per tier across four
contexts; full experiment 222s. Nothing runs on the 10-second clock — the live
path is still the 22ms cached lookup.

### VERDICT: GO for targeted real-board precomputation

Operational policy, derived from the table:

| player class | policy |
|---|---|
| weekly improvement ~0 | proxy/market only; never spend CE seasons |
| 1-2 pts | 4,000-season audit when nominated |
| 8+ pts | 4,000-season audit, precomputed between nominations |
| 18+ pts | 4,000 is 30x more than needed; spend budget on allocation seeds |
| near a high-dollar frontier | larger offline confirmation, or UNDERPOWERED_AT_CAP |

### Remaining limitations

* Fabricated board only; no real auction observed.
* Tiers are built by scaling one candidate, so tier and "which player" are not
  separated — a different candidate at the same scale might behave differently.
* Elite effects (0.35-0.57 CE) are enormous because the fabricated pool has a
  thin top; real boards will compress this.
* Only three allocation seeds.

### Exact next step

**Run the tier ladder on a second, structurally different candidate** (a
different position and bye week at the same scales) to confirm the power curve
is a property of player strength rather than of this one fabricated player.

---

## Phase: real-board tactical pilot — NO-GO

Branch `real-board-tactical-pilot`, from
`36f533077649e99cfade1a45be996063b2dbf207`. Full detail in
`docs/REAL_BOARD_PILOT.md`; sanitized aggregates in
`docs/real_pilot_sanitized.json`. Player-level output is at
`local_data/tactical/real_board_pilot.json` (gitignored, not committed).

### VERDICT: NO-GO for full targeted precomputation

Not because the machinery is wrong — conservation held on every arm, league CE
summed to exactly 1.0 everywhere, runtime is comfortable, and QB/TE behaviour
matches the real lineup graph. The blocker is that **allocation instability
dominates every audited effect on the real board**:

```
position   between-alloc SD   within-alloc SE   SD/effect   sign stable
QB                  0.02547           0.00626        0.97      True
RB                  0.04717           0.00644        2.14      FALSE
WR                  0.04980           0.00632        2.11      FALSE
```

Fabricated board: 0.016-0.157. Real board: 0.97-2.14 — 13x to 130x worse, with
sign flips for RB and WR. A sign flip is not a precision problem; the estimate
has no stable value to be precise about.

### Two more blockers, same root cause

**Opening symmetry FAILED.** Structurally identical Team02/Team03 in an empty
room gave CE 0.04325 vs 0.05500 — a 0.01175 gap against ~0.00322 SE, 3.6 SE
apart. Opponent identity is being manufactured by continuation jitter before a
single sale.

**Ladders never bind.** Every tested price in all four ladders returned an
identical delta (QB $14 and $38 both +0.03650). With $186 and fifteen open
slots the same completions stay affordable, so the nested set correctly selects
the same joint world. **No robust maximum from this pilot may be quoted** — the
highest favorable price is just the highest price tested. A reservation price
needs money to be scarce, which means mid-auction.

All three share one cause: the shared-board continuation's per-(player, owner)
tie-breaking jitter compounds over 180 allocations on a 260-player board.

### What did validate

```
contract 549 -> 260 mapped PlayerSpecs (QB 35 / RB 70 / WR 112 / TE 43)
149 anchored (57%); 111 unanchored priced at the $1 floor (stated, not valued)
222 individual injury profiles, 38 positional fallback
opening room: $2,400 total, 180 slots, $186 opening legal max, no duplicate ids
conservation OK on every arm; league CE = 1.0 exactly on every arm
runtime: 20.5s load, 7.5s per audited candidate, 232s total
```

Selection: 12 candidates, 3 per position, 3 tiers, 0 substitutions, chosen by
each position's own clearing-price order — no name in the selection code.

6 audited (5 resolved, 1 unresolved), 6 proxy-only (0 seasons spent).

### Position findings

**QB / superflex.** All three QBs show the largest lineup improvements on the
board (6.86 / 4.86 / 3.87): a second startable QB beats the skill fallback for
the open superflex seat. No strict-2QB behaviour, no QB premium, 1-QB and 5-QB
rosters both legal and tested. **QB3 insurance was NOT measured** — an empty
room has no existing QB availability risk to insure against.

**TE / no TE slot.** The sharpest market-vs-CE disagreement in the pilot:

```
TE expensive: anchor $34, band $12/$13/$14, lineup improvement 0.22 -> PROXY ONLY
WR expensive: anchor $48, band $30/$32/$34, lineup improvement 1.71 -> +0.0795
```

No scarcity premium, none hard-coded (asserted by test). Caveat: the reference
roster already carries three TEs, so a fourth reads low.

**Handcuffs: NOT SUPPORTED.** No conditional-backfield mapping exists; the
contract carries standalone projections only. No named handcuff recommendation
may be made from this pilot.

### A bug this pilot caught

The first run filled the diagnostic's fourteen reference slots with the top
fourteen by projection. In a superflex league that is fourteen quarterbacks —
not a legal roster — and it read the best QB on the board as a 0.03-point
improvement. Fixed to a legal 1 QB / 4 RB / 6 WR / 3 TE reference and
regression-tested.

### Exact next step

**Fix allocation instability before anything else.** Run one real candidate
across ~15 continuation seeds and characterise the distribution of delta_ce.
Then either report a between-allocation interval alongside the paired SE, or
reduce the continuation's dependence on tie-breaking jitter. Until a single
number is stable across continuations, no real-board price is defensible — and
the opening-symmetry failure is the cheapest test that the fix worked.

---

## Phase: allocation ensemble and opening symmetry — GO

Branch `allocation-ensemble-symmetry`, from
`c8b4cb1d2322685aad0eed86a53abfba687a65f3`. Detail in
`docs/ALLOCATION_ENSEMBLE.md`; aggregates in
`docs/allocation_ensemble_sanitized.json`.

### What the old jitter was, measured before changing it

```
sigma 0.06, multiplicative on WILLINGNESS (not a tie-break), 180 allocations:
exact ties in raw willingness            152/180  84.4%
winner changed by jitter                 130/180  72.2%
  ...overturned a genuine non-tied gap     8/180   4.4%  (mean gap $0.07)
clearing price changed                    92/180  51.1%
identical winner to a no-jitter run       50/180  27.8%
```

84% of allocations were genuine ties — twelve identical owners in an empty room
have identical willingness — so tie-breaking was real work done by the wrong
instrument. The fatal part: it was indexed by each owner's *position in a
tuple*, so at a fixed seed one team drew the same noise column every time.

### The fix

**Mechanical tie-break**: highest willingness wins outright; only bids within
$0.50 count as tied, broken by this draw's priority slot. No randomness.
**Preference shock**: explicit, scenario-labelled, zero by default, forced to
zero in symmetry tests, indexed by priority slot so it permutes.
**Balanced schedule**: eleven rivals rotate through eleven priority slots, each
exactly once; the focus team keeps its slot.

### Opening symmetry: RESTORED

```
                        before (1 seed)    after (ensemble)
Team02 vs Team03 gap    +0.01175 (3.6 SE)  mean +0.00187
                                           CI95 [-0.00726, +0.01099]
single-seed gap now                        +0.00325
persistent label effect                    False
```

### Primary RB — the candidate that flipped sign

Before: three arbitrary seeds gave `[+0.0645, -0.02875, +0.03025]`.
After, over 11 exchangeable draws:

```
deltas          [-0.0905, -0.10125, -0.09225, -0.04625, -0.06475, -0.07375,
                 -0.0335, -0.0345, -0.0445, -0.05575, -0.069]
mean            -0.06418      median -0.06475      min/max -0.10125/-0.03350
between-alloc SD 0.02361      RMS within-alloc SE 0.00873   ratio 2.70
SE of mean       0.00712      95% t-interval [-0.08004, -0.04832] (df=10)
sign positive    0.00  (11 of 11 negative)
```

**The sign instability was the shock, not the economics.** Between-allocation SD
is 2.7x the season SE: more seasons would not help, more draws would. Pooling
11 x 4,000 seasons flat would have given an interval ~60x too narrow.

### Convergence

```
 K      mean   between SD   SE(mean)   CI95                    verdict
 1   -0.09050    0.00000        n/a    none                    unresolved (no interval exists)
 3   -0.09467    0.00577    0.00333    [-0.10900, -0.08034]    unfavorable
 6   -0.07812    0.02050    0.00837    [-0.09964, -0.05661]    unfavorable
12   -0.06638    0.02375    0.00686    [-0.08147, -0.05128]    unfavorable
15   -0.06908    0.02448    0.00632    [-0.08264, -0.05553]    unfavorable
```

**Required K = 11.** With the shock off the allocation is a function of the
rotation alone, so eleven rotations exhaust the balanced set: draws 12-15
reproduced draws 1-4 exactly. The run reported 15 distinct board fingerprints
because those hash the seed; there were **11 distinct joint worlds**.
`effective_k` and `redundant_draws` now report this — a K=15 interval built on
11 real draws would have been narrower than its coverage.

### Positions

```
position   K    mean delta   between SD   CI95                     verdict
RB        11    -0.06418     0.02361      [-0.08004, -0.04832]     unfavorable
QB         6    +0.15596     0.00576      [+0.14991, +0.16200]     favorable
WR         6    +0.07700     0.03213      [+0.04327, +0.11073]     favorable
TE         -    proxy only (0.22 lineup improvement; no CE seasons spent)
```

QB is the most stable effect (between-SD 3.7% of the mean), consistent with the
open superflex seat.

### Runtime

7.3s per allocation draw (buy + pass, 4,000-season holdout); ~80s per candidate
at K=11; 328s total including symmetry and two secondary positions.

### VERDICT: GO for targeted real-player precomputation

Symmetry restored, no owner-ID effect, sign stable across the exchangeable
ensemble, K=11 practical, invariants pass on every draw. The earlier NO-GO is
discharged. What remains is honest allocation uncertainty, which must be
reported as an interval over futures — never folded into one CE interval, and
never replaced by a single arbitrary future auction.

### Remaining limitations

* One real candidate at K=11; QB/WR only at K=6.
* Preference shock is implemented and tested but has not been exercised on the
  real board with a non-zero sigma, so no behavioural scenario is calibrated.
* Price ladders remain `FRONTIER_NOT_REACHED` — an empty room cannot bind.
* Between-allocation SD is large in absolute terms (RB 0.024 against a 0.064
  mean, 37%); the estimate is stable in sign, not tight in magnitude.

### Exact next step

**Re-run the four priority ladders at a mid-auction state under the ensemble.**
Money must be scarce for a ladder to bind, so seed the room with a plausible set
of recorded sales, then walk each ladder at K=11 and see whether a frontier
appears. That is the first point at which a real maximum bid could be quoted.

---

## Phase: quota-free marginal diagnostics — PARTIAL GO

Branch `real-board-marginal-diagnostics`, from
`fcb0cd0fbb6d0e0743196a9c2edad31b5e860324`. Detail in
`docs/MARGINAL_DIAGNOSTICS.md`.

### Exact reach of the 1/4/6/3 template

It controlled the lineup-improvement number, displaced-player identification,
starter/bench classification, **the proxy-only vs CE-audited sampling decision**,
which positions got an ensemble run, and every position finding. It did **not**
touch candidate selection, completion search, shared-board allocation, buy/pass
CE, or max-bid. So the CE mathematics was never wrong — but the template decided
*which real players ever reached it*, which is worse, because a wrong number can
be checked and an unsimulated player cannot.

### Replacement

Two bounded completion searches over the real board — without the candidate, and
with him bought at price `p` — under exact eligibility and the $1-per-slot
reserve. `improvement = best_legal_eight(with) - best_legal_eight(without)`.
Composition is an output. `ProxyEvaluator.lineup_shares` reports per-player start
rates from the same mask the simulator uses: 15 QBs start exactly 2.0, a mixed
roster starts 7.98.

### A search-convergence defect this exposed

Acquiring a player for less can never be worse. The first run violated that:

```
beam/pool    improve@$18   improve@$1   violation
   24/30          +1.30        -4.64      +5.94
   64/60          +2.19        -6.04      +8.23
  160/90          +1.21        +1.16      +0.05
 320/140          -1.08        +0.86      -1.94
```

At the pilot's width the search error exceeded the effects being measured. **My
first quota-free table was untrustworthy and was discarded, not published.**
Defaults are now 160/90; `price_monotonicity_violation` is computed per
candidate and printed. Current run: **0/12 violations, worst +0.45/wk**.

### Naturally selected compositions

```
template (old):      QB1/RB4/WR6/TE3
naturally selected:  QB3/RB7/WR3/TE2, QB3/RB8/WR2/TE2, QB3/RB7/WR4/TE1,
                     QB4/RB5/WR2/TE4, QB3/RB8/WR3/TE1, QB3/RB7/WR2/TE3
```

The board wants three or four QBs and seven or eight RBs, and it varies by
candidate — which a fixed template cannot do by construction.

### Before vs after (12-player pilot)

```
                        template        quota-free, converged
audited / proxy-only    6 / 6           8 / 4
TE1 improvement         0.22            +1.89
TE1 policy              proxy only      4,000-season audit
all three TEs           proxy only      all three audited
QB1 improvement         6.86            +0.05 @ $26  (+5.02 @ $1)
QB1 policy              audit           proxy only (replaceable at his price)
```

**Both headline pilot findings reversed.** The TE result ("no scarcity premium,
TE1 proxy-only") was an artefact of three reserved TE places; the QB result
("all three QBs show the largest improvements") was an artefact of one reserved
QB place.

### Roles now measured, not assumed

Improvement leads, not start share — a share threshold alone audited all twelve,
because 8 of 15 start most weeks. New role **replaceable starter**: he starts,
but the completion is as good without him because his price buys more elsewhere.
`improvement_at_min` separates player quality from price.

**Contingency value is not priced.** No conditional-backfield or QB-insurance
mapping exists; bench raw points are never converted into lineup value. QB3
insurance cannot be measured from an empty room.

### K=11 ensemble confirmation

```
position      K   mean delta   between-SD   within-SE   CI95                     verdict
RB (primary) 11    +0.00775      0.01862     0.00804    [-0.00476, +0.02026]     unresolved
QB           11    -0.03814      0.02901     0.00810    [-0.05762, -0.01865]     unfavorable
WR           11    -0.02727      0.01987     0.00805    [-0.04062, -0.01392]     unfavorable
TE           11    -0.06091      0.02636     0.00808    [-0.07862, -0.04320]     unfavorable
```

**A TE qualified and was run**, impossible under the template. Symmetry held:
single-seed gap -0.01725, ensemble mean +0.00070, CI95 [-0.00766, +0.00907],
contains zero, no persistent label effect.

**Caveat:** this ensemble ran with the pre-convergence (beam 32) diagnostic
choosing which candidate to audit per position, so its candidates are not the
ones the converged table nominates. The CE numbers are valid for the candidates
actually run; the selection needs re-running.

### VERDICT: PARTIAL GO for real price-frontier testing

No quota affects diagnostics or sampling; roles come from legal completion; QB
and TE follow exact eligibility; conservation and symmetry intact. But only three
of four positions resolved at K=11, and the ensemble was selected by an
unconverged diagnostic.

### Remaining unsupported roles

* Contingency / handcuff value — no conditional mapping exists.
* QB3 injury insurance — unmeasurable from an empty room.
* Bench depth value beyond its effect on the best legal eight.

### Exact next step

**Re-run the K=11 ensemble with the converged (160/90) diagnostic driving
candidate selection**, so the CE confirmation covers the players the corrected
diagnostic actually nominates — in particular TE1, which the template excluded
entirely. Only then is the frontier work safe to start.
