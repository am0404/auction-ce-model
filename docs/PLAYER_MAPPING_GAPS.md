# PLAYER_MAPPING_GAPS.md

Every `ceauction.players.PlayerSpec` field against the existing player model
("war-room"), with an honest status for each.

**Nothing here has been implemented.** This is the gap analysis the ingestion
work needs before any mapping is written, and several rows say the mapping
cannot be written yet.

| Status | Meaning |
|---|---|
| **SUPPORTED** | A source field exists and its meaning is proven end to end. Can be populated now. |
| **PARTIAL** | A source field exists, but a stated assumption or an unresolved semantic question stands between it and the target. |
| **UNSUPPORTED** | No source field exists. Not a gap to be filled with a default. |
| **UNKNOWN** | A source field exists but its meaning cannot be established from the material. Populating it would be a guess. |

---

## 1. The table

| `PlayerSpec` field | Source | Status | Notes |
|---|---|---|---|
| `player_id` | normalised name key | **SUPPORTED** | war-room uses `normalize_name(...)` with `_` for spaces. Needs a stable integer id for this engine's `crn_key`; assign at import and persist it. |
| `name` | `projections_2026.csv :: Name` | **SUPPORTED** | 626 unique. |
| `position` | `projections_2026.csv :: Pos` | **SUPPORTED** | Filter to QB/RB/WR/TE as war-room does. |
| `nfl_team` | `fantasypros_2026.csv :: TEAM` | **PARTIAL** | Only on rows that join. war-room's join leaves some projected players unmatched, and it reports the unmatched list rather than guessing. |
| `base_mean` | 9 stat columns → `score_statline` ÷ games | **PARTIAL** | The arithmetic is proven and this engine's `score_statline` is the right seam. Blocked on **Q1** (window/divisor), **Q2** (`Fumbles` vs fumbles lost), **Q3** (already availability-discounted?) and **Q4** (mean vs median). Four missing scoring categories (§3). |
| `week_sd` | `fits.json :: player.weekly_cv` | **PARTIAL** | The *only* proven dispersion source: `week_sd = weekly_cv[pos] × base_mean`. Two mismatches to resolve first — see §2. **Not** derivable from the upside/bust grades. |
| `season_sd` | — | **UNSUPPORTED** | Nothing in the inventory measures how far a player's *true season-long* level deviates from consensus. war-room has no equivalent concept. |
| `bye_week` | `fantasypros_2026.csv :: BYE WEEK` | **SUPPORTED** | Real bye weeks, 84.3% coverage, range 5–14. Absent for the rest; must be carried as absent, not as 0. |
| `weekly_injury_hazard` | `fits.json :: availability.weekly_miss` (positional) or `draftsharks :: injury_prob` (per player) | **PARTIAL** | Positional rates are proven but are **availability** hazards, not injury hazards — they include benchings, rest and trades. Per-player `injury_prob` has an unproven vendor definition (**Q5**). war-room combines both: per-player where profiled, fitted positional rate otherwise. |
| `injury_mean_weeks` | `draftsharks :: proj_games_missed` | **PARTIAL** | Has a real unit (games) but needs `injury_prob` to convert: `mean_weeks ≈ proj_games_missed / injury_prob` only holds if both refer to the same event, which **Q5** does not establish. Coverage 331/626. |
| `spike_rate`, `spike_scale` | — | **UNSUPPORTED** | Nothing measures unforecastable spike frequency or size. `weekly_cv` captures total dispersion, not its tail shape. |
| `spike_mean_removed` | — | n/a | A modelling switch, not data. |
| `role_change_prob` / `_mean` / `_sd` / `role_reveal_lag` | — | **UNSUPPORTED** | war-room's `docs/01_player_model.md` *specifies* role states in detail, but no role-transition rates were fitted and no source column carries them. |
| `signal_noise_sd` | — | **UNSUPPORTED** | Nothing measures how informative usage data is about a player's persistent level. `OPEN_QUESTIONS.md` A5 already flags this as the least-grounded parameter in the engine; the inventory does not change that. |
| `weekly_state_sd` | — | **UNSUPPORTED** | Requires splitting weekly dispersion into forecastable and unforecastable parts. `weekly_cv` is the **total** and carries no such split. See §2.3. |
| `weekly_state_pattern` | — | **UNSUPPORTED** | Would need per-week matchup/volume projections. The source is season-long only. |
| `hidden_weekly_pattern` | — | n/a | An experimental control, not a data field. |
| `shock_loadings` | — | **UNSUPPORTED** | No correlation structure of any kind. NFL team is available, so a `team:` group could be *constructed*, but its beta would be invented. |
| `contingency` | `players_provisional.csv :: depth_rank` (synthetic) | **UNSUPPORTED** | The only depth-chart field found is in the **synthetic** provisional pool. No real depth chart is present. |
| `proj_noise_sd` | — | **UNSUPPORTED** | Would need archived weekly projections joined to outcomes. Only a season-long snapshot exists. |
| `weekly_projection_override` | — | **UNSUPPORTED** | Season-long only; no weekly series. |
| `data_source` | — | **SUPPORTED** | Set to a real provenance string; the field exists for exactly this. |
| `crn_key` | — | **SUPPORTED** | Assign from the stable player id. |
| `notes` | — | **SUPPORTED** | Free text. |

**Count, by `PlayerSpec` field (28 total): 7 SUPPORTED, 5 PARTIAL,
14 UNSUPPORTED, 2 not-applicable** (`spike_mean_removed` and
`hidden_weekly_pattern` are modelling switches, not data). No field is marked
UNKNOWN outright — but the four PARTIAL rows that matter most (`base_mean`,
`week_sd`, `weekly_injury_hazard`, `injury_mean_weeks`) are each blocked on at
least one UNKNOWN from `PLAYER_DATA_INVENTORY.md` §6, which is the same thing in
practice.

---

## 2. The two mappings that are nearly there

### 2.1 `base_mean` from stat lines

This is the mapping the whole engine most needs, and it is *close*. The source
is component stat lines; `ceauction.scoring.score_statline` converts them; the
result is season points; dividing by a games count gives a per-week mean.

Four questions stand in the way, and none is cosmetic:

* **Q1, the divisor.** A wrong games count scales every `base_mean` by a
  constant and is invisible to every test in this repository.
* **Q2, `Fumbles`.** If the column is total fumbles rather than fumbles lost,
  the penalty is roughly doubled. Compounded by `fumble_lost` being the one
  scoring value war-room never had confirmed by the owner.
* **Q3, availability.** If the vendor already discounted for missed games, then
  `base_mean` is not "points per week when active" and applying
  `weekly_injury_hazard` on top double-counts.
* **Q4, mean vs median.** `base_mean` is documented as *expected* points. If the
  source is a median, right-skewed categories bias it low, by a different amount
  per position.

**Four scoring categories have no source column** — passing, rushing and
receiving two-point conversions, and individual special-teams touchdowns. They
must be recorded as unsupported rather than defaulted to zero; the schema
requires an explicit `unsupported_categories` list for this reason.

### 2.2 `week_sd` from `weekly_cv`

`week_sd = weekly_cv[pos] × base_mean` is the natural mapping and the only one
with a proven basis. Two mismatches must be resolved deliberately:

* **Conditioning.** `weekly_cv` is computed over the weeks a player *appeared*.
  This engine's `week_sd` is also the dispersion of a week he plays, so these
  agree — but the agreement should be asserted in the ingestion, not assumed.
* **What it includes.** `weekly_cv` is the **total** observed weekly dispersion.
  This engine splits weekly variation into `week_sd` (unforecastable) and
  `weekly_state_sd` (forecastable, knowable before lock), and `synthetic.py`
  carves the second out of the first. Mapping the whole CV onto `week_sd` alone
  would silently assert that **none** of a player's weekly variation is
  forecastable — the opposite of the synthetic default, and a claim the data
  does not support either way.

Also note `weekly_cv` is a **within-player CV averaged across players**, so
applying it to an individual player is applying a cohort average, not that
player's own dispersion. Every player at a position would get the same
coefficient.

### 2.3 The forecastable/unforecastable split has no source

This is worth stating separately because it blocks two fields at once. Nothing
in the inventory measures what share of weekly variation is knowable before
kickoff. Splitting it needs archived **weekly** projections joined to weekly
outcomes — the R² of that join *is* the parameter (`OPEN_QUESTIONS.md` A2/A5).
Only a season-long projection snapshot exists.

Until that data exists, `weekly_state_sd` cannot be populated and `week_sd`
cannot be given its correct unforecastable-only value.

---

## 3. What the boom/bust grades cannot do

They cannot produce `week_sd`, and no ingestion should try.

`UPSIDE ` and `BUST ` are ordinal 1–5 expert labels on 28.5% of rows. war-room's
own measurement puts `corr(upside tag, projected points) = +0.530` and
`corr(upside tag, injury_prob) = +0.247`: the raw tag is substantially a
restatement of "this player is good" and "this player gets hurt". war-room
regresses it on both within position and uses only the residual — which
correlates −0.000 with points — and uses that for **bench ordering only**,
never for value.

Turning a 1–5 ordinal into a standard deviation requires a calibration that does
not exist in this material. `fits.json` is the dispersion source; the grades are
not.

---

## 4. Every uncalibrated parameter, and what would calibrate it

| Parameter | Why it has no source | What would settle it |
|---|---|---|
| `season_sd` | No measure of true-talent deviation from consensus | Preseason projections joined to realised season means, several seasons; the residual spread is the parameter |
| `signal_noise_sd` | No usage-vs-outcome relationship measured | Regress rest-of-season scoring on usage through week *k*; the residual spread is the parameter |
| `weekly_state_sd` | No forecastable/unforecastable split | R² of archived weekly projections against weekly outcomes, within player |
| `proj_noise_sd` | No archived weekly projections | RMSE of weekly projections against outcomes |
| `spike_rate`, `spike_scale` | No tail-shape measurement | Weekly score distributions by player; fit frequency and size of the upper tail |
| `role_change_*` | No role-transition rates fitted | Hand-tagged depth-chart changes over a season, with dates |
| `shock_loadings` | No correlation structure | Residual correlation matrix of weekly scores after removing each player's own mean |
| `contingency` | No real depth chart | A real depth chart plus a backup-usage-on-starter-absence estimate |
| `weekly_projection_override` | Season-long source only | A weekly projection feed |

**Fourteen `PlayerSpec` fields, in the nine groups above, have no real-data
source in this inventory.**
That is the honest headline: importing what exists would populate identity,
`bye_week`, `base_mean` and a cohort-average `week_sd` — and would leave every
parameter governing *forecastability*, *correlation*, *role change* and *tail
shape* exactly as invented as it is today.

The contract in `schemas/real_player_input_v1.schema.json` therefore carries an
explicit `uncalibrated_parameters` list, so a run built on it can always say
which parts of its answer rest on data and which do not.

---

## 5. Before any of this is implemented

`PLAYER_DATA_INVENTORY.md` §6 lists eight questions. Three block the mapping
outright and one is prior to all of them:

* **Q6 — which league?** war-room is configured for **10 teams, no superflex**;
  this engine's `SPEC.md` specifies **12 teams with a superflex**. These are
  different leagues. Slot eligibility drives every lineup decision and therefore
  every CE number, so this is not a detail to reconcile later.
* **Q1** (projection window), **Q2** (`Fumbles`), **Q4** (mean vs median) —
  each changes `base_mean` for every player.

`Q3`, `Q5`, `Q7` and `Q8` shape the availability mapping and the confidence that
can be attached to it.
