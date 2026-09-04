# OPEN_QUESTIONS.md

Decisions that could not be settled on Night 1 because they need **real player
data** or **your judgement**. Each entry states the question, what the code does
today, why that placeholder was chosen, how to change it, and what evidence
would settle it.

Nothing here is a bug. Every one of these is a deliberate, reversible
placeholder behind a stable interface.

**Settled and removed.** The league is **winner-take-all**. Championship equity
is `P(win week 17)` and there is no payout vector to weight. This was previously
listed as an open question; it is not one.

---

## A. Needs real data

### A1. What are the real per-week means, and what shape is replacement level?

**Today.** `synthetic.PositionProfile` gives each position an exponential decay
from a top value to a floor (`QB 22.0 -> 9.0`, `RB 17.0 -> 4.0`,
`WR 16.5 -> 4.0`, `TE 13.0 -> 3.0`). Invented. Since the zero floor was removed,
`base_mean` is unambiguously **expected fantasy points**, so a real projection
set drops in without reinterpretation.

**Why it matters more than it looks.** The *curvature* of that decay, not its
level, is what determines whether concentrating projection in one player beats
spreading it (experiment `concentration`). A flat replacement level makes studs
cheap; a steep one makes them essential.

**To change.** Populate `PlayerSpec.base_mean` from real projections. Nothing
else moves.

**What settles it.** A real projection set, plus last season's actual weekly
half-PPR distributions to check the tail shape at each position.

---

### A2. How wide is real weekly variance, and does it scale with the mean?

**Today.** A single total weekly dispersion per position (`QB 6.0`, `RB 6.8`,
`WR 7.2`, `TE 5.8`), constant across ranks, of which a fixed part
(`QB 2.5`, `RB 3.0`, `WR 3.0`, `TE 2.4`) is carved out as *forecastable*
(`weekly_state_sd`, SPEC §4.7) and the remainder left as unforecastable
`week_sd`. Constancy across ranks is almost certainly wrong: a 4 pt/week WR5
does not have the same weekly SD as a 17 pt/week WR1.

**The split is a second, separate guess.** How much of a player's week-to-week
movement is knowable before kickoff decides how much of the roster is a genuine
weekly decision at all, and therefore how much a deep bench is worth. The
current 40%-ish forecastable share is invented.

**To change.** `PlayerSpec.week_sd` and `PlayerSpec.weekly_state_sd` are both
already per player. The likely real form for the total is `a + b * base_mean`;
the interface supports any function of the player.

**What settles it.** For the total: weekly half-PPR scores by player for 2-3
seasons, regressed on preseason projection. For the split: the R² of a weekly
projection against the actual weekly score, within player. That R² *is* the
forecastable share.

---

### A3. What are the real injury hazards and absence lengths?

**Today.** A per-week hazard (`RB 0.045`, `WR/TE 0.030`, `QB 0.020`) with a
mean absence of 2.0-2.6 weeks, independent across players, memoryless, with no
age, position-detail or injury-history structure.

**Open sub-questions:** does hazard rise after a first injury? Is a "questionable"
tag worth modelling as partial availability rather than a binary?

**To change.** `weekly_injury_hazard` and `injury_mean_weeks` are already per
player. A richer process (re-injury, multi-state) would need a new function in
`worlds._draw_availability`; the rest of the engine does not care.

**What settles it.** Historical games-missed data by position and age.

---

### A4. What is the real correlation structure?

**Today.** Team shocks with invented betas (`QB 1.3`, `WR 0.9`, `RB 0.7`,
`TE 0.7`) plus a passing-game stack group on each NFL team's first QB/WR/TE.
Both magnitudes are guesses.

**Explicitly not modelled yet:** opponent correlation (you vs the rival starting
the other side of the same game), backfield timeshares as negative correlation,
game-script effects.

**To change.** `PlayerSpec.shock_loadings` accepts any number of named groups
with any betas, so an arbitrary factor structure drops straight in. Negative
correlation already works (`test_negative_beta_creates_negative_correlation`).

**What settles it.** A residual correlation matrix from historical weekly scores
after removing each player's own mean.

---

### A5. How informative is real usage data, and how good are published projections?

**Today.** Two uncalibrated numbers, doing two different jobs.

`signal_noise_sd` (SPEC §4.6) is the precision of the **observable-information
channel** — the thing that stands for snap share, route participation, target or
carry share and depth-chart reporting. It defaults to `week_sd`, meaning "one
week of observed usage tells you about as much about a player's true level as
one observed score would have". That is a placeholder chosen to be conservative,
not an estimate, and it is the single least-grounded parameter in the model.

`proj_noise_sd` (1.0–1.3 points) is analyst noise on top: how wrong the published
number is relative to the conditional mean the model already knows.

**Why it matters.** Everything about the value of forecastability (experiments
`spikes`, `aggregate-lineup-spot`) is measured relative to how good the forecast
is. And `signal_noise_sd` sets how fast a genuinely improved player becomes
startable, which is most of what makes a mid-season breakout worth anything.
Together they make the simulated manager *quite* good — arguably better than a
real one, because he never overreacts and never has a hunch.

**To change.** Both are per-player fields. Separately,
`PlayerSpec.weekly_projection_override` accepts a real published weekly
projection series and bypasses the modelled projection entirely
(`test_projection_override_replaces_the_model`).

**What settles it.** For `proj_noise_sd`: a season of archived weekly projections
joined to actual scores — the RMSE of that join *is* the parameter. For
`signal_noise_sd`: regress a player's rest-of-season scoring on his usage through
week *k*; the residual spread of that relationship is the parameter.

---

### A6. How often do observable role changes happen, and how big are they?

**Today.** 10-18% of players per season, a normal-sized shift revealed one week
after it takes effect.

**Open sub-question:** is a one-week reveal lag right? Some role changes are
announced before they show up in a box score (a trade, a benching); those are
`role_reveal_lag = 0`, which the interface supports and which is tested. A truly
*negative* lag — known to be coming before it takes effect — is not supported,
and would be the natural extension if hand-tagging shows it is common.

**What settles it.** Hand-tagging a season of depth-chart changes.

---

## B. Needs your judgement

### B1. Should the simulated manager be as good as this one?

The engine assumes all 12 managers play the exact projection-optimal lineup
every week and never make an error. That is a modelling choice with a direction:
it *understates* the value of a roster that is easy to set (obvious starters)
and *overstates* the value of a deep bench that requires correct weekly
decisions.

**Options.** (a) Keep perfect play — clean, and arguably the right benchmark for
*your* team. (b) Give opponents a start/sit error rate. (c) Give opponents
perfect play but yourself a noisier projection.

**Recommendation:** keep perfect play for now; it is the conservative choice for
valuing your own depth. Revisit only if opponent modelling becomes a goal.

---

### B2. Should the model include in-season roster change?

There are no waivers, no FAAB and no trades. The drafted 15 are the 15 in week
17. This is the single largest structural simplification in the model.

Its effects do **not** point one way, and an earlier version of this document
claimed they did. The bias depends on what kind of roster spot you are asking
about:

* **Static drafted depth and handcuffs are overvalued as injury protection.**
  The model treats a drafted backup as the only thing standing between you and
  an empty slot, because in the model it is. In reality a comparable
  replacement is usually available on waivers the Tuesday after the injury, so
  much of what the model prices as "insurance" is insurance against a risk you
  could have covered later for free.
* **Churnable lottery-ticket roster spots are undervalued.** A speculative
  bench player's real value includes the option to cut him in week 4 and use
  the spot on whatever emerges next. The model has no such option: a failed
  bet occupies a roster spot for seventeen weeks. Spots held for optionality
  are therefore worth *more* in reality than here.
* **The damage from injuries is overstated.** When a starter goes down, the
  model can only fall back on players you already own, so an injury costs the
  full gap between the starter and your bench. In reality the gap closes
  partly through acquisition.

Those three are not the same claim, and they do not net out to a single
direction. A handcuff and a lottery ticket are biased in *opposite* directions
by the same simplification.

**Question for you:** does the pricing layer need to be right about *draft-day*
value only, or about value net of the waiver wire? Those are different numbers,
and the gap is largest exactly at the bottom of the roster — which is where
$1-$3 auction decisions live.

---

### B3. How should the median result interact with lineup strategy?

The dual head-to-head + median format rewards *consistency* far more than a
pure head-to-head league does — you need to clear the median 14 times. The
engine reproduces this automatically, but it raises a strategy question the
model does not currently answer: should a team that is clearly out of playoff
contention by week 10 start tail-seeking? Today every manager maximises
projected points every week regardless of standings.

---

### B4. What tolerance do you want on CE differences?

At 16,000 seasons a paired CE difference has a standard error of roughly
0.001-0.003 depending on the size of the change. A $1 auction decision may well
be worth less CE than that. Before pricing is built, you need to decide what
resolution the pricing layer requires, because that sets the simulation budget
per candidate roster — and therefore whether a live-auction tool is feasible at
all.

---

## C. Known modelling gaps not yet parameterised

| Gap | Effect on CE | Interface exists? |
|---|---|---|
| Real NFL bye schedule (byes are currently drawn per synthetic team) | small but systematic; real byes cluster | yes, `PlayerSpec.bye_week` |
| Real matchup/volume data behind `weekly_state` (§4.7) | sets how much of the roster is a live weekly decision | yes, `weekly_state_sd` / `weekly_state_pattern` |
| Rival roster composition as an input to your own CE (`rival-fit`) | measurable and non-zero when rivals differ in fit | partly — needs a rival-roster model, not just a pool |
| Playoff-week resting starters | overstates week 17 reliability of players on locked teams | no |
| Opponent-correlation (your player vs your opponent's player in the same game) | affects head-to-head variance, not the median | partly — needs signed cross-roster groups |
| Position-scaled variance (A2) | changes stud vs depth tradeoff | yes, per-player `week_sd` |
| Multi-week "questionable" states | injuries are binary today | no |
| Kickers / DST | not in this league | n/a |
