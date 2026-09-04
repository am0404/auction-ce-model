# CALIBRATION_AUDIT.md

Result of mapping the normalized real-player contract into `PlayerSpec` and
running the existing CE engine on the result.

> **This repository is public.** Everything below is an aggregate: counts,
> coverage rates, calibration error summaries and CE ranges. There are no
> player rows, no projection values and no local paths. The mapped specs and
> the unresolved-identity list stay in a git-ignored `local_data/`.

Reproduce with:

```bash
ce-lab ingest    --projections … --fantasypros … --injuries … --fits … \
                 --aliases data/player_aliases.json \
                 --contract-out local_data/real_player_contract_v1.json
ce-lab calibrate --contract local_data/real_player_contract_v1.json \
                 --contract-fumbles-lost local_data/contract_fumbles_lost.json \
                 --fits … --limit 300 --sims 2000 --sensitivity
```

---

## 1. Coverage by depth band

The full-pool injury match rate of 54.5% was never the decision-relevant
number. A 12-team, 15-man league consumes 180 players and in-season churn
reaches perhaps 300; everything past that is a tail nobody drafts.

```
  band             n   team%    bye%  injury%   disp%  unresolved
  ---------------------------------------------------------------
  top_180        180   100.0   100.0     91.7   100.0           0
  top_240        240   100.0    99.6     88.3   100.0           1
  top_300        300   100.0    98.3     80.0   100.0           5
  full_pool      549    95.6    87.6     54.5   100.0          68
```

**Identity and bye coverage is essentially complete where it matters**, and
injury coverage is 91.7% in the top 180 rather than the 54.5% the pool-wide
figure suggested.

Two top-240 identities were unresolved before this pass and both are now
closed:

* one via a reviewed alias — the identity source lists him under a nickname.
  Recorded in `data/player_aliases.json` with its reason;
* one documented as **genuinely unresolvable**: he is listed as a free agent,
  and a player with no team has no bye week. Supplying one would be
  fabrication, so he is carried without a bye and named in
  `known_unresolvable`.

Fuzzy matching is deliberately not used anywhere. It would have resolved both
and would also silently join two different players with similar names, with no
way to notice afterwards.

---

## 2. Calibration error

300 specs mapped in 46 seconds.

### Projection level

```
  level_abs_error   n=300   max 0.0   median 0.0   mean 0.0
```

Exact, and that is a property of the method rather than luck. The simulated
season total is exactly homogeneous in the per-game level — every dispersion
term is expressed as a fraction of it — so the calibration is a division by a
simulated unit-level statistic rather than a search, and the residual is zero
by construction.

**What the calibration found is worth stating plainly.** With the fields this
source can populate, every weekly component is symmetric: the idiosyncratic
draw and the season shift are both normal, and the skewed component (spikes)
has no source and stays unpopulated. A sum of symmetric variables is symmetric,
so its median equals its mean and **both targets return `season_total / 17`**,
agreeing to within 0.01%. The calibration confirms that rather than assuming
it, and would return different levels the moment a skewed component were
populated — without anyone having to remember to revisit the function.

### Injury parameters

```
  injury_source_counts        individual 240,  positional_all_cause 60
  injury_prob_abs_error       n=240   median 0.00300   mean 0.00462   max 0.09525
  games_missed_abs_error      n=240   median 0.00903   mean 0.01147   max 0.13337
  infeasible_calibrations     2 of 240
```

Both supplied targets — season injury probability and projected games missed —
are matched jointly by inverting the engine's own availability process. Median
error is 0.003 on a probability and 0.009 on a game.

**Two of 240 could not be jointly reproduced** and are reported rather than
quietly fitted. The engine's process ties frequency and duration together — an
absence blocks new onsets — so a high injury probability paired with very few
projected games missed has no solution. Both cases are of that shape. The
closest available parameters are used and both errors are carried on the
player.

**60 of 300 players have no individual profile** and fall back to the fitted
positional rate. That rate is labelled **all-cause availability**, not injury:
it counts benching, rest and trades as well as injury, and the mapping warns
about it on every run. Silently treating an unprofiled player as healthy is not
an option the configuration offers.

### The week/game distinction

The vendor's projected-games-missed figure spans 17 NFL games; the fantasy
horizon holds 16. The target is scaled by 16/17 before solving — conflating the
two spans would inflate every absence by about 6%. And a week that is both a
bye and an absence costs no *scheduled game*, which the calibration accounts
for explicitly.

---

## 3. Real-data CE smoke test

Twelve deterministic, legal, disjoint rosters built by a snake over the real
draftable pool, **solely so the engine has teams to simulate**. Not an auction
allocation; the CE levels are not advice.

```
REAL-DATA CE SMOKE TEST  ok=True
  [PASS] hall_condition_holds
  [PASS] lineups_never_exceed_eight
  [PASS] only_available_players_start
  [PASS] lineups_are_maximal
  [PASS] eight_slots_filled_when_possible  (97.4% of team-weeks fill all eight; min 5 when byes cluster)
  [PASS] non_qb_superflex_allowed  (16.4% of team-weeks)
  [PASS] one_bye_per_player_with_a_bye
  [PASS] sixteen_scheduled_games_in_horizon  (17 weeks minus one bye = 16)
  [PASS] no_week_18  (array has 17 weeks)
  [PASS] unavailable_players_score_zero
  [PASS] exactly_one_champion_per_season
  [PASS] deterministic
  [PASS] chunk_invariant
  [PASS] ce_is_a_distribution  (sum 1.000000)
  [PASS] no_synthetic_players  (sources ['REAL:winwithodds+draftsharks+nflverse-fits'])
  [PASS] all_specs_marked_real
  [PASS] base_mean_is_a_per_game_level  (max 20.35)
  [PASS] no_grade_derived_distribution
```

Two of these deserve comment.

**`eight_slots_filled_when_possible` is not "always eight".** Real byes cluster,
and a team can legitimately have too few startable players in a given week —
97.4% of team-weeks fill all eight, with a minimum of five when byes bunch. An
earlier version of this check asserted eight always and failed, which was the
check being wrong rather than the engine. What must hold is that the lineup is
**maximal**: no available player could legally have been added. That is now
what is asserted.

**Sixteen scheduled games inside a seventeen-week horizon** is the correction
this phase existed partly to make. The projection spans 17 NFL games; the
fantasy window holds 16 of them because of the bye, and week 18 does not exist
in the engine at all. Both are now asserted rather than assumed.

---

## 4. Sensitivity

One axis moves at a time from a baseline. A full cross-product would be more
scenarios and less information: what a reader needs is how far each individual
unresolved choice can move the answer.

```
scenario                                         CE min   CE max   spread   pts/wk  infeas
------------------------------------------------------------------------------------------
median_target|f=0.00|ssd=0.00|inj=individual|fum=exclude   0.0430   0.1635   0.1205    90.74       2
mean_target|f=0.00|ssd=0.00|inj=individual|fum=exclude   0.0430   0.1630   0.1200    90.73       2
median_target|f=0.25|ssd=0.00|inj=individual|fum=exclude   0.0390   0.1435   0.1045    95.09       2
median_target|f=0.50|ssd=0.00|inj=individual|fum=exclude   0.0335   0.1470   0.1135    98.54       2
median_target|f=0.00|ssd=0.10|inj=individual|fum=exclude   0.0450   0.1530   0.1080    90.62       2
median_target|f=0.00|ssd=0.20|inj=individual|fum=exclude   0.0555   0.1230   0.0675    91.17       2
median_target|f=0.00|ssd=0.00|inj=positional|fum=exclude   0.0460   0.1560   0.1100    89.08       0
median_target|f=0.00|ssd=0.00|inj=individual|fum=lost   0.0450   0.1615   0.1165    89.66       2
------------------------------------------------------------------------------------------

```

### Largest CE shift for any single roster, versus the baseline

| Axis | Max shift | What it is |
|---|---:|---|
| `season_sd` | **+0.0405** | a pure scenario; **no empirical basis at all** |
| `forecastable_share` | **+0.0275** | a pure scenario; **no empirical basis at all** |
| `injury_model` | +0.0170 | individual calibration vs the all-cause positional fallback |
| `fumbles` | +0.0100 | excluded vs treated as lost |
| `target` | **+0.0005** | median vs mean |

**The finding that matters is the ordering.** The two axes that move
championship equity most are the two with **no data behind them whatsoever** —
season-level uncertainty and the forecastable share of weekly variance. Both
are labelled scenarios in this codebase precisely because nothing in the
inventory measures them, and both move CE by more than any question about the
source's semantics does.

**The median-versus-mean question moves CE by 0.0005.** That was the most
fraught semantic issue going into this phase, and it is empirically almost
irrelevant here — exactly as the symmetry argument in §2 predicts. It would
stop being irrelevant the moment a skewed component were populated.

**Fumbles, the remaining blocking-looking question, moves CE by 0.0100.** Real
but modest, and smaller than either uncalibrated scenario axis. Excluding it
remains the right default and it is correctly not a blocker.

For scale: `docs/example_curve_output.txt` puts the paired standard error of a
CE difference at roughly 0.0015 at 16,000 seasons. Every shift above except
`target` is well outside that, so these are real movements rather than noise.

---

## 5. What this does not do

No dollar value, no opening bid, no live bid, no auction inflation, no
opponent bidding, no roster-completion optimisation. The twelve rosters exist
to give the engine legal teams. Nothing here prices anything.
