# SPEC.md — Championship-Equity (CE) Fantasy Auction Model, Night 1

Status: **Night 1 foundation.** This document specifies the simulation engine that
converts a *drafted roster* into a *championship probability*. It deliberately stops
short of auction pricing (see §11).

---

## 1. League rules

Platform target: **Sleeper redraft auction**.

| Setting | Value |
|---|---|
| Teams | 12 |
| Auction budget | $200 per team |
| Bid increments | whole dollars, minimum bid $1 |
| Roster size | 15 |
| Starting lineup | 8 |
| Bench | 7 |
| IR slots | 0 |
| Kicker / DST | none |
| Dedicated TE slot | none |
| QB roster maximum | none |
| Scoring | half PPR |

### 1.1 Scoring

| Event | Points |
|---|---|
| Passing yard | 0.04 |
| Passing TD | 4 |
| Interception thrown | -2 |
| Rushing yard | 0.1 |
| Receiving yard | 0.1 |
| Rushing TD | 6 |
| Receiving TD | 6 |
| Reception | 0.5 |
| Fumble lost | -2 |
| Passing two-point conversion | 2 |
| Rushing two-point conversion | 2 |
| Receiving two-point conversion | 2 |
| Individual special-teams TD (kick/punt return) | 6 |

Implemented in `ceauction.scoring.score_statline`. The CE engine models fantasy
points directly; `score_statline` is the documented conversion used when real data
arrives as projected *stat lines* rather than projected *points*. The seam carries
the **complete** rule set rather than only the rules the synthetic process happens
to exercise, because a missing rule becomes silently dropped points the moment real
stat lines arrive. `tests/test_scoring.py` asserts every `StatLine` field has a
matching coefficient and that none scores zero.

Two of these rules are negative. **An individual player's weekly total can therefore
be negative**, and the league has no rule flooring it at zero. See §5.5.

### 1.2 Starting lineup slots

Exactly eight slots, in canonical order:

| # | Slot | Eligible positions |
|---|---|---|
| 0 | `QB` | QB |
| 1 | `RB1` | RB |
| 2 | `RB2` | RB |
| 3 | `WT1` | WR, TE |
| 4 | `WT2` | WR, TE |
| 5 | `WT3` | WR, TE |
| 6 | `FLEX` | RB, WR, TE |
| 7 | `SUPERFLEX` | QB, RB, WR, TE |

The superflex **does not require** a second quarterback; an RB, WR or TE is legal
there. There is no dedicated TE requirement — the three `WT` slots accept any mix of
WR and TE.

### 1.3 Regular season

* Weeks 1–14.
* Each week produces **two** standings results per team:
  1. head-to-head vs. that week's scheduled opponent;
  2. vs. the **league median** score for that week.
* Median of 12 scores = mean of the 6th and 7th ranked scores.
* Outcomes per result: win = 1.0, tie = 0.5, loss = 0.0. A team's weekly record is
  therefore 2-0, 1-1, 1-0-1, 0-1-1, 0-2, or 0-0-2. A score exactly equal to the
  median is a tie.
* Schedule: every team plays every other team **at least once**. 12 teams gives an
  11-week single round robin; weeks 12–14 repeat rounds 1–3.
* Standings sort: total wins (descending), then **total points** (descending), then
  team index (ascending, deterministic final tiebreak).

### 1.4 Playoffs

Six qualifiers, seeds 1–6 by the standings sort above.

```
Week 15 (quarterfinals)   Seeds 1 and 2 have byes
    QF-A:  4 vs 5
    QF-B:  3 vs 6
Week 16 (semifinals)      No reseeding
    SF-1:  1 vs winner(QF-A)      # 1 always faces the 4/5 winner
    SF-2:  2 vs winner(QF-B)      # 2 always faces the 3/6 winner
Week 17 (championship)
    winner(SF-1) vs winner(SF-2)
```

* No reseeding between rounds.
* A tied playoff matchup is won by the **higher seed**. There is no coin flip and no
  extra randomness anywhere in the bracket: once weekly scores exist, the whole
  postseason is a deterministic function of them.
* Winner takes all. Championship equity is the probability of winning Week 17.

---

## 2. Simulation sequence

Every simulated season follows this pipeline exactly once, in this order:

```
1.  drafted rosters                 (input; 12 x 15 disjoint players)
2.  latent player / season state    worlds.draw_latent_state
3.  availability                    worlds.draw_availability   (byes, injuries)
4.  realized weekly scores          worlds.draw_realized       (weeks 1..17)
5.  pregame-observable information  worlds.build_pregame       (filtration, week w
                                                                sees only weeks < w)
6.  lineup decision                 lineup_vec.select_lineups  (pregame only)
7.  team weekly score               simulate.team_scores       (realized of starters)
8.  standings                       standings.regular_season
9.  playoffs                        playoffs.run_bracket
10. champion                        one team index per simulated season
```

Steps 2–4 create the *world*. Step 5 projects the world onto what a manager could
know. Step 6 may read **only** step 5's output.

Note on ordering: realized scores for **all** weeks are drawn before pregame
information is built. This is an implementation convenience, not an information leak:
`build_pregame` for week *w* is a function of realized scores in weeks strictly less
than *w* only, enforced by construction (a shifted cumulative sum) and asserted by
tests.

---

## 3. The information barrier

**Rule.** A lineup for week *w* may depend only on information observable before
kickoff of week *w*.

**Enforcement.** Three independent mechanisms:

1. *Type-level.* The scalar lineup API consumes `PregameEntry` objects, which have
   **no field** capable of holding a realized score. The vectorized optimizer's
   signature accepts `(projection, available, position)` arrays and never receives
   the realized array.
2. *Construction.* `build_pregame` computes week *w*'s belief from `cumsum` of
   residuals shifted one week, so week *w*'s own realized score is arithmetically
   absent.
3. *Test.* `test_information_barrier.py` randomly permutes the realized array and
   asserts the chosen starter masks are bit-identical, and asserts a benched player
   given +1000 realized points still contributes nothing.

**Consequences that the model reproduces:**

* A benched player who unexpectedly scores 30 provides **zero** team value that week.
* That performance creates future value only through the *observable* channel: it
  moves the manager's posterior on the player's persistent season level, and/or a
  separately generated **observable role change** makes him a better projection going
  forward. Pure spike weeks (§5.5) are drawn from a distribution the manager cannot
  forecast and are correctly discounted by the filter.

---

## 4. Player-state interfaces

Five concepts are kept in distinct places in the data model.

| # | Concept | Where it lives | Visible to lineup? |
|---|---|---|---|
| 1 | Persistent latent season state | `LatentState.season_shift`, `role_delta`, `role_change_week` | no (only through §4.3) |
| 2 | Health and availability | `Availability.available[p, w]`, `out_reason` (bye / injury / active) | **yes** |
| 3 | Pregame-observable role and projection | `Pregame.projection[p, w]`, `observed_role_delta[p, w]` | **yes** |
| 4 | Realized weekly performance | `Realized.points[p, w]` | **no** |
| 5 | Information available in future weeks | `Pregame.posterior_mean[p, w]`, built from realized weeks `< w` | **yes** (for weeks > the reveal) |

### 4.1 `PlayerSpec` — the reversible interface

`PlayerSpec` is the **only** input the CE engine needs about a player. It is the seam
between "where the numbers came from" and "what the engine does with them". Real data
arrives later by populating the same fields; the engine is not rewritten.

```python
PlayerSpec(
    player_id, crn_key, name, position, nfl_team, bye_week,
    base_mean,             # <- real per-week median/mean projection
    week_sd,               # <- real boom/bust width
    season_sd,             # <- how much true talent can differ from consensus
    weekly_injury_hazard,  # <- real per-week injury probability
    injury_mean_weeks,     # <- real expected absence length
    spike_rate, spike_scale,           # <- unforecastable ceiling games
    role_change_prob, role_change_mean, role_change_sd, role_reveal_lag,
    shock_loadings,        # <- correlation relationships (team, stack, custom groups)
    contingency,           # <- handcuff / next-man-up structure
    proj_noise_sd,         # <- how wrong consensus projections are, week to week
    weekly_projection_override,  # <- real published weekly projections, if supplied
    data_source,           # "SYNTHETIC" tonight; "REAL:<vendor>" later
)
```

`crn_key` exists purely for common random numbers: two *alternative* players being
compared for the same roster slot may share a `crn_key` so that they receive
identical uniform draws and differ only through their parameters. See §8.

### 4.2 Latent season state

`season_shift[p] ~ Normal(0, season_sd)` — a persistent, per-season deviation of the
player's true weekly mean from his consensus `base_mean`. Drawn once per season, never
directly observed.

### 4.3 Observable role change

With probability `role_change_prob` a player experiences a role change in a uniformly
chosen week `wc` in weeks 2..13, of size `Normal(role_change_mean, role_change_sd)`.

* The change affects **realized** scoring from week `wc` onward.
* It becomes **observable** from week `wc + role_reveal_lag` onward (default lag 1),
  modelling "the market learns the new role after it shows up".
* Once revealed it is fully and permanently in the projection.

This is the mechanism by which a surprising performance can legitimately create future
value.

### 4.4 Contingency (handcuffs)

`Contingency(on_player_id, bonus)`: in any week where `on_player_id` is unavailable,
this player's true mean and his projection both rise by `bonus`. Availability is
pregame-observable, so the contingency is observable — a backup RB is correctly
projected up in the weeks his starter is out.

### 4.5 Correlation hooks

`shock_loadings: tuple[ShockLoading(group_id, beta), ...]`. A per-week shock is drawn
once per `group_id` and added to every member's realized score scaled by its `beta`.
This single mechanism covers:

* **team environment** — all players on an NFL team share a `team:<abbr>` group;
* **QB / pass-catcher stacks** — a `stack:<id>` group with positive betas on both;
* **negative correlation** — a shared group with betas of opposite sign;
* **arbitrary custom correlation** — any user-defined group.

Group shocks are drawn from streams keyed by `group_id`, so they are stable under CRN.

---

## 5. Synthetic player process (clearly labelled)

`ceauction.synthetic` is the **only** module that invents numbers. Every spec it emits
carries `data_source="SYNTHETIC"`. It is a *demonstration* of the interface, not an
estimate of real football. It produces:

1. persistent season-level performance states (`season_sd`);
2. pregame projections (filtered posterior + analyst noise);
3. weekly scoring variance (`week_sd`, position-scaled);
4. injuries and unavailable weeks (hazard + duration) and bye weeks;
5. observable role changes (§4.3);
6. unforecastable spike weeks (§5.5);
7. shared team-level shocks (§4.5);
8. hooks for arbitrary player correlations (§4.5).

### 5.5 Realized score

```
raw[p, w]  = base_mean[p]
           + season_shift[p]
           + true_role_delta[p, w]          # role change, revealed later
           + contingency_bonus[p, w]        # observable
           + sum_g beta[p, g] * shock[g, w] # team / stack / custom
           + week_sd[p] * eps[p, w]         # idiosyncratic
           + spike[p, w] - spike_mean[p]    # heavy right tail, mean-removed
points[p, w] = max(0, raw[p, w])
```

`spike[p, w] = Bernoulli(spike_rate) * Exponential(spike_scale)`. `spike_mean` is
subtracted so that raising `spike_rate` at fixed `spike_scale` does **not** change the
unconditional mean — this is what makes the "predictable upside vs. unforecastable
spikes" experiment a clean comparison.

### 5.6 Pregame projection

The manager knows `base_mean`, the observed role state, contingency status and
availability. He does **not** know `season_shift`, the team shock, the idiosyncratic
noise or the spike. He filters:

```
resid[p, u]      = points[p, u] - base_mean[p] - observed_role_delta[p, u]
                                 - contingency_bonus[p, u]        (played weeks only)
S[p, w]          = sum over u < w of resid[p, u] * played[p, u]
n[p, w]          = sum over u < w of played[p, u]
posterior[p, w]  = tau_obs * S[p, w] / (tau_0 + n[p, w] * tau_obs)      # prior mean 0
level[p, w]      = base_mean[p] + observed_role_delta[p, w]
                 + contingency_bonus[p, w] + posterior[p, w]
projection[p, w] = E[max(0, Normal(level[p, w], week_sd[p]))]
                 + proj_noise_sd[p] * nu[p, w]
```

with `tau_0 = 1 / season_sd^2` and `tau_obs = 1 / week_sd^2` — a Gaussian conjugate
update, treating the (heavy-tailed, non-Gaussian) spikes as if they were part of the
observation noise. This is exactly right for the intended behaviour: a spike week does
move the posterior a little, but is heavily shrunk, so **spikes do not turn into
reliable future projection**, while a genuine persistent level does get learned.

### 5.7 The projection is expected *points*, not the latent mean

Realized scores are floored at zero (§5.5), so a player's expected output is
`E[max(0, X)]`, which exceeds his latent mean by an amount that grows with his weekly
SD. For a replacement-tier synthetic WR at 4.0 points with a weekly SD of 7.2 the gap
is **+1.3 points per week**; for an 18-point WR at the same SD it is under 0.05.

The projection therefore reports `floored_mean(level, week_sd)`
(`ceauction.stats.floored_mean`, using an Abramowitz & Stegun normal CDF so that
SciPy is not a dependency). Projecting the raw latent mean instead would
systematically under-project every low-mean, high-variance player and bias every
bench and flex decision against exactly the players a 15-for-8 portfolio holds for
optionality. `tests/test_worlds.py::test_projection_is_expected_points_not_the_latent_mean`
asserts the projection is an unbiased estimate of realized points at both ends of the
range.

A consequence worth stating plainly: at a fixed *expected-points* level, changing a
player's volatility changes only the shape of his distribution — which is what makes
the `volatility` laboratory experiment a clean test rather than a disguised level
comparison.

If `weekly_projection_override` is supplied (real published projections), it replaces
the whole right-hand side for that player.

---

## 6. Lineup optimisation

**Problem.** Choose, from 15 rostered players, a highest-projected legal assignment to
the eight slots of §1.2, using pregame projections only, with unavailable players
excluded.

**Solution — exact, not heuristic.** The slot eligibility sets

```
{QB},  {RB},  {WR,TE},  {RB,WR,TE},  {QB,RB,WR,TE}
```

form a **laminar family**, so the sets of simultaneously-startable players form a
transversal matroid whose independence test collapses to seven counting constraints.
Writing `q, r, t` for the number of selected QB / RB / (WR or TE):

```
q <= 2        (QB slot + SUPERFLEX)
r <= 4        (RB1, RB2, FLEX, SUPERFLEX)
t <= 5        (WT1..3, FLEX, SUPERFLEX)
q + r <= 5
q + t <= 6
r + t <= 7
q + r + t <= 8
```

Because independent sets form a matroid, the **greedy** algorithm — sort available
players by projection descending, add each player if the counts still satisfy all
seven constraints, stop after 8 — is provably optimal *and* yields a maximum-cardinality
lineup when fewer than eight players are startable. This is exact; no LP or Hungarian
solver is required.

`tests/test_lineup.py` cross-checks the greedy result against brute-force enumeration
of all C(15,8) subsets on randomised inputs.

**Slot assignment and explanation.** Selected players are assigned to concrete slots
most-restrictive-first (`QB`, `RB1`, `RB2`, `WT1..3`, `FLEX`, `SUPERFLEX`), which is
valid for a laminar family. Each `LineupChoice` records the slot, the projection, and a
human-readable `reason` (e.g. `"RB2: 2nd-best available RB (12.40 proj)"`,
`"SUPERFLEX: best remaining flex-eligible; non-QB used"`), plus `benched` entries with
their exclusion reason (`bye`, `injured`, `out-projected`, `slot-blocked`).

**Portfolio framing.** Nothing in the engine labels eight players as "the starters".
Every week, all 15 players are re-evaluated against that week's availability,
contingency status, observed roles and projections. A bench WR is a real asset when he
out-projects a starter on bye, and the engine values him accordingly.

---

## 7. Standings and playoffs

Per week *w* and team *t*: `score[s, t, w] = sum of realized points of the chosen starters`.

* `h2h[s, t, w]` = 1 / 0.5 / 0 vs `opponent[s, t, w]`.
* `median[s, w]` = mean of the 6th and 7th ranked of the 12 scores;
  `med[s, t, w]` = 1 / 0.5 / 0.
* `wins[s, t] = sum_w (h2h + med)` out of a maximum of 28.
* `points[s, t] = sum_w score`.
* Seeding: `lexsort` on (team index asc, points desc, wins desc) → seeds 1..12.
* Playoffs exactly as §1.4, evaluated on weeks 15/16/17 scores from the same pipeline.

The schedule is a circle-method round robin (11 rounds) with rounds 1–3 repeated for
weeks 12–14. The mapping *team → schedule slot* is permuted per simulated season from
a dedicated RNG stream, so no team has a structurally easier schedule; under CRN the
same permutation is used in both scenarios of a paired comparison.

---

## 8. Championship equity and paired comparison

`CE(team t) = (# seasons where t wins week 17) / (# seasons)`, winner-take-all.

Monte Carlo standard error of a single CE: `sqrt(p(1-p)/n)`.

### 8.1 Common random numbers

All randomness comes from a **counter-based** generator (`ceauction.rng`), a
splitmix64/threefry-style hash of `(seed, kind, sim_index, entity_id, week, sub)`.
There is no sequential state, so a draw's value depends only on its coordinates.
Consequences:

* Changing one player on one roster changes **no other player's** draws.
* Injuries, team environments, spikes, role changes and the schedule permutation are
  all identical across the two scenarios of a paired comparison.
* Two alternative players competing for the same roster slot can be given the same
  `crn_key`, so they even share their *uniform draws* and differ only in parameters —
  the strongest available variance reduction.

### 8.2 Paired report

For scenarios A and B over the same `n` seasons, per-season paired indicators
`d_i = 1{A champ} - 1{B champ}` give

```
delta_CE = mean(d)          se(delta_CE) = sd(d) / sqrt(n)
```

which is far tighter than differencing two independent CE estimates. Reported for the
focus team in each scenario: championship equity, playoff probability, bye (top-2)
probability, average regular-season points, above-median rate, head-to-head win rate,
and Monte Carlo standard errors — plus the paired CE difference and its SE.

**No separate playoff coin flips exist.** Given the score array, standings and bracket
are deterministic.

---

## 9. Determinism

Every result is a pure function of `(rosters, seed, n_sims)`. Chunking the simulation
into batches does not change results: sim index `i` uses stream coordinate `i`
regardless of batch boundaries. `tests/test_reproducibility.py` asserts identical
output across repeated runs and across different chunk sizes.

---

## 10. Known simplifications

1. **Points are modelled directly**, not built up from passing/rushing/receiving stat
   lines. `scoring.py` holds the real half-PPR rules for when stat-level data arrives.
2. **Synthetic player parameters are invented.** They are labelled `SYNTHETIC` and are
   calibrated only to be *plausible in shape* (QB > RB ≈ WR > TE means, QB lower
   variance). No claim is made about real football.
3. **Gaussian conjugate filtering** approximates the manager's learning. Real managers
   use richer information (snap counts, target share, vegas lines). The residual filter
   is a stand-in for all of it.
4. **Projection = expected points + noise.** No manager optimism/pessimism bias, no
   vendor disagreement, no start/sit expected-value adjustments for tail-seeking.
   The residual filter forms its posterior against the *latent* mean while realized
   points are floored at zero, so the filter is slightly biased for players whose mean
   sits within about 1.5 weekly SDs of zero.
5. **No waivers, no FAAB, no trades, no in-season roster change of any kind.** The
   drafted 15 are the 15 all season. This understates the value of bench depth and
   overstates the cost of injuries.
6. **No IR**, per league settings — an injured player occupies a roster spot.
7. **Injury model is a two-parameter hazard/duration process**, independent across
   players, with no position-specific structure and no re-injury correlation.
8. **Byes are drawn per NFL team in weeks 5–14** and are fixed for a season; the real
   NFL bye schedule is known in advance and should replace this.
9. **Opponent rosters are exogenous.** Changing your roster does not change theirs, and
   the 180-player pool is exactly consumed by the 12 rosters.
10. **The league median uses all 12 teams' realized scores**, which is correct for
    Sleeper, but note the median is itself affected by every manager's lineup skill;
    all 12 managers here are equally and perfectly rational given their information.
11. **Every manager plays the projection-optimal lineup.** No one starts a player for
    tail-chasing reasons even when trailing badly, and no one makes start/sit mistakes.
12. **Playoff weeks use the same generative process as the regular season.** No real
    Week 17 resting-starters effect.

---

## 11. Explicitly deferred work

Not built tonight, by instruction:

* real player dollar values; opening max bids; live max bids;
* auction-room behaviour and opponent-specific bidding;
* a greedy or exact auction roster-completion solver;
* any web / Streamlit interface;
* waivers, FAAB, trades;
* arbitrary floor / ceiling / scarcity / stacking / positional modifiers;
* fabricated "real player" distributions of any kind.

The intended consumer of this engine is a marginal-CE pricing layer: `CE(roster with
player X at price p) - CE(best alternative use of $p)`. Nothing tonight presumes how
that layer will work beyond requiring that CE be cheap, paired, and low-variance.

Open modelling decisions that need real data or user judgement are catalogued in
`OPEN_QUESTIONS.md`.
