# INGESTION_AUDIT.md

Sanitized result of running `ce-lab ingest` against the canonical sources
documented in `PLAYER_DATA_INVENTORY.md`.

> **This repository is public.** Everything below is an aggregate over a whole
> column: counts, coverage, match rates, ranges, missingness, warnings and
> validation results. There are no player names, no per-player values, no
> proprietary vendor numbers and no local paths. The contract itself — which
> does contain real rows — is written only to a git-ignored `local_data/`
> location, and the CLI refuses to write it anywhere else.

Reproduce with:

```bash
ce-lab ingest \
  --projections <component stat projection CSV> \
  --fantasypros <expert consensus CSV> \
  --injuries    <injury profile JSON> \
  --fits        <fitted dispersion JSON> \
  --contract-out local_data/real_player_contract_v1.json \
  --report-out   local_data/ingestion_report.json
```

Exit status is 0 when the contract validates and 1 when it does not.

---

## The run

```
REAL-PLAYER INGESTION REPORT (sanitized)
==============================================================================
schema_version    1.0.0
generated_at      2026-09-04T20:45:18+00:00
league_config_id  spec-md-12team-superflex-half-ppr

SOURCES (identified by content hash, not filename)
  projections_2026.csv                   3ea4ef21c74ccebc  performance_projection
                                         (no retrieval timestamp: cannot be dated)
  fantasypros_2026.csv                   8b7cbaf7b58e0275  player_identity
                                         (no retrieval timestamp: cannot be dated)
  draftsharks_injury_profiles.json       b280795fbf3d0ce9  injury_availability
                                         (no retrieval timestamp: cannot be dated)
  fits.json                              2e1facb4c1b1046f  outcome_distribution

NORMALIZED PLAYERS: 549
  by position: {'QB': 76, 'RB': 135, 'TE': 125, 'WR': 213}

COVERAGE
  nfl_team                 524  (95.4%)
  bye_week                 480  (87.4%)
  injury_profile           300  (54.6%)
  cohort_dispersion        549  (100.0%)
  expert_labels            256  (46.6%)

JOINS
  fantasypros: matched 524/549 (95.5%); unmatched_left 24, unmatched_right 344, ambiguous 1, dup_left 0, dup_right 1, conflicting 0
  injury: matched 300/549 (54.6%); unmatched_left 249, unmatched_right 31, ambiguous 0, dup_left 0, dup_right 0, conflicting 0

FIELD SUMMARIES (aggregates only)
  season_points          n=549   cover=1.0     min=0.6       med=41.6      max=345.93    mean=72.173
  active_rate_a          n=549   cover=1.0     min=0.035     med=2.447     max=20.349    mean=4.245
  active_rate_b          n=300   cover=0.5464  min=0.036     med=6.666     max=23.062    mean=7.307
  omitted_fumble_points  n=125   cover=0.2277  min=-8.0      med=-2.0      max=-2.0      mean=-3.28
  injury_prob            n=300   cover=0.5464  min=0.06      med=0.36      max=0.88      mean=0.36
  proj_games_missed      n=300   cover=0.5464  min=0.1       med=1.3       max=4.5       mean=1.536
  bye_week               n=480   cover=0.8743  min=5.0       med=10.0      max=14.0      mean=9.383

SCORING SUPPORT
  supported   (8): pass_yard, pass_td, interception, rush_yard, rush_td, rec_yard, rec_td, reception
  unsupported (5):
    pass_2pt             treated_as=absent
    rush_2pt             treated_as=absent
    rec_2pt              treated_as=absent
    special_teams_td     treated_as=absent
    fumble_lost          treated_as=absent

UNCALIBRATED PARAMETERS (10)
  season_sd, signal_noise_sd, weekly_state_sd, proj_noise_sd, spike_rate, spike_scale, role_change_prob, shock_loadings, contingency, weekly_injury_hazard

OPEN QUESTIONS
  Q1    blocking=True
  Q2    blocking=True
  Q3    blocking=True
  Q5    blocking=False
  Q7    blocking=False
```

---

## Reading it

**549 players normalized** from 626 projection rows: the difference is the
non-skill positions the engine does not model, filtered at load.

**The two joins behave very differently, and both numbers matter.**

* *Identity / bye week*: 524 of 549 matched (95.5%). 344 right-hand rows went
  unmatched, which is expected — that file carries 944 rows spanning players
  the projection source does not cover. One ambiguous name and one duplicate on
  the right-hand side were reported rather than resolved by an arbitrary pick;
  the affected player carries a null team and null bye instead of a coin-flipped
  one.
* *Injury*: 300 of 549 matched (**54.6%**). That is the number to be careful
  about. A little over half the pool carries an availability profile and the
  rest carries none, so any downstream use of these fields applies information
  to some players and not others. The previous model gated a join like this at
  45 of the top 60 by value for exactly that reason.

**`active_rate_b` exists for only the 300 players with a games-missed figure.**
That is correct and not a defect: interpretation B divides by
`17 − projected games missed`, which cannot be computed without one. Note the
gap between the two interpretations at the median — 2.45 against 6.67 — is
large, and it is not a modelling result. The two are computed over different
populations: A over all 549 including deep bench players, B only over the 300
with injury profiles, who skew toward the players a vendor bothers to profile.
The pair should never be compared as if it were a treatment effect.

**`omitted_fumble_points` covers 22.8% of players**, ranging from −8.0 to −2.0
with a mean of −3.28. That is the size of the open question, stated in points:
for the roughly one player in four who carries a fumble figure, between 2 and 8
points of season scoring are currently excluded because the column's meaning is
unresolved. It is excluded rather than guessed, and reported rather than hidden.

**Five scoring categories are unsupported and all five are `absent`.** Four have
no source column at all (the three two-point conversions and the individual
special-teams touchdown). The fifth is `fumble_lost`, unsupported by choice
while Q2 is open.

**Ten engine parameters are declared uncalibrated**, and `weekly_injury_hazard`
is among them deliberately: `injury_prob` is season-level risk, not a weekly
rate, and no weekly injury process has been derived from it.

## Validation

**0 errors, 5 warnings.** The warnings are all worth keeping visible:

* Three sources carry no retrieval timestamp and cannot be dated. For the
  injury capture this is structural — the extraction records no update time.
* Three blocking open questions remain (Q1 games basis, Q2 fumbles, Q3
  availability treatment). The payload carries them rather than assuming past
  them.
* One player in 549 (0.2%) has recomputed points exactly equal to a vendor
  total preserved in `raw_fields`, despite carrying receptions that should
  separate half-PPR from full. This is reported as an aggregate count rather
  than per player, for two reasons: the per-player form would carry a vendor
  value into a published report, and a count is the more useful signal — one
  coincidence is noise, a large share would mean the vendor total was being
  reused.

That last check was initially written as an **error** and it was wrong twice
over. It first fired on seven players, every one of whom had zero receptions —
where half-PPR and full-PPR agree exactly and coincidence proves nothing. After
narrowing it to players whose receptions should separate the two systems, it
still fired on one, whose components genuinely produce the same figure under
this league's scoring. Value equality is not proof of provenance. The binding
guarantee is the structural one: `scoring_source` is pinned to
`recomputed_from_components` and the vendor total column is never read.
