# PLAYER_DATA_INVENTORY.md

Inventory of the existing player model ("war-room") as a candidate real-data
source for this engine. **Inventory and semantic tracing only** — no pricing, no
ingestion code, no mapping decisions acted on.

> **This repository is public.** Nothing here contains player rows, vendor
> values, or absolute paths. Every figure below is an aggregate over a whole
> column. The source directories were read but never modified, and the raw data
> is not committed (`local_data/` is git-ignored).

---

## 1. The canonical source

The existing model lives in a git repository referred to below as **`war-room`**
(local, on the Desktop; remote `am0404/Draft`).

| | |
|---|---|
| Branch | `claude/championship-simulator-spec-0fhb5b` |
| HEAD | `ff720ea` — "Refresh season-long projections from the latest source export" |
| Tracking | in sync with `origin/<same branch>`, which is also `origin/HEAD` |
| Working tree | **dirty, but not in a way that matters**: one modified tracked file (`data/adp/ffc/coverage.csv`) and five untracked files (`.DS_Store`, `ENVIRONMENT.txt`, `data/adp/ffc/coverage.json`, `requirements-audit.txt`, `scripts/audit_repro.sh`). None of the six files below is dirty. |

Structure: `warroom/` (the Python package, 26 modules), `scripts/`, `tests/`,
`docs/` (27 numbered design and investigation notes), `data/`, `config/`,
`server.py`, `public/`.

### 1.1 Which copy is canonical

Three other directories hold overlapping data. They were compared by SHA-256,
header, row count and mtime.

| Logical file | war-room | Draft2 / audit-data / handoff | Draft |
|---|---|---|---|
| `projections_2026.csv` | `3ea4ef21…` **canonical** | `1456edb0…` superseded | `e88d52c9…` oldest |
| `fantasypros_2026.csv` | `8b7cbaf7…` | identical | identical |
| `players_provisional.csv` | `9b88f1bd…` | identical | identical |
| `draftsharks_injury_profiles.json` | `b280795f…` | identical (Draft2, audit-data, handoff) | absent |
| `ffc_10team_halfppr_2026.csv` | `399cce10…` | identical | identical |
| `fits.json` | `2e1facb4…` | identical | identical |

**Only `projections_2026.csv` genuinely differs**, and it differs three ways.
All three share an identical header and 626 data rows, so the schema is stable
and only the vendor's values were refreshed.

**The filename is actively misleading, exactly as warned.** The file distributed
in `audit-data/` and `war-room-handoff/` is named `season_long_proj_table-2.csv`
— the "-2" reads like a later revision. It is byte-identical to Draft2's
`projections_2026.csv` (`1456edb0…`, 2026-09-03 01:16) and is **superseded** by
the war-room copy (`3ea4ef21…`, 2026-09-03 19:41), which is committed as
`ff720ea`. Chronology: Draft (Aug 30) → Draft2 / audit-data / handoff (Sep 3
01:16) → war-room (Sep 3 19:41).

`season_long_proj_table-2.csv` and `projections_2026.csv` are the **same logical
file under two names** — the vendor export name and the name it takes in the
board directory.

### 1.2 Canonical file list

| Logical file | SHA-256 (full) | Rows |
|---|---|---|
| `data/board/projections_2026.csv` | `3ea4ef21c74ccebc0044973fac43a28b30218ef21b96e419ac2fcd99f6271130` | 626 |
| `data/board/fantasypros_2026.csv` | `8b7cbaf7b58e02755c8966212758858ba5b4eb08f8375b1f0f2f56f45fa89b81` | 944 |
| `data/injury/draftsharks_injury_profiles.json` | `b280795fbf3d0ce9e43b5c71205a91e154d393ab9466af71ad95c539cf30197a` | 331 players |
| `data/market/ffc_10team_halfppr_2026.csv` | `399cce104bd7ed1bd2f0fbf8e3a1bfd77e5898151b245dd8e3ad297e223f0d0e` | 192 |
| `data/calibration/fits.json` | `2e1facb4c1b1046f1adb3ecf9a38576b04cdf78c2fb65894b78a404855375ce3` | 2 fits |
| `data/players_provisional.csv` | `9b88f1bd183520e15ccf3cda09523ff5d4918f20d8cbaa47b503baa08463d995` | 448 |
| `config/league.json` | `0292a46f4f92bf00998272ea086a7c28206e039b9d7bc339a39853cdc580eaa3` | — |

---

## 2. File classification

| Logical file | Class | Note |
|---|---|---|
| `projections_2026.csv` | **football performance projection** | Season-long component **stat lines**. The single source of value in war-room. |
| `draftsharks_injury_profiles.json` | **injury/availability input** | Vendor per-player injury risk. |
| `fits.json` | **outcome-distribution input** | Fitted from real historical weekly results. The only genuine dispersion source found. |
| `fantasypros_2026.csv` | **mixed** — bye week is player identity; UPSIDE/BUST is an *ordinal expert label*, not a distribution; RK / ECR VS. ADP is **market** | war-room uses only bye, upside/bust and an ADP fallback. Its rankings are explicitly **not** used as performance truth. |
| `ffc_10team_halfppr_2026.csv` | **market price/ADP input** | Draft-market data only. |
| `adp_10team_half-ppr.json`, `data/adp/ffc/**` | **market price/ADP input** | Historical ADP for market calibration. |
| `players_provisional.csv` | **derived output — SYNTHETIC** | **Not real data.** See §5. |
| `data/nflverse/*.parquet` | **outcome-distribution input** (raw) | Real historical weekly results; git-ignored, refetchable. Input to `fits.json`. |
| `data/state/draft.json`, `public/`, `server.py` | **irrelevant to this project** | Live draft UI/state. |
| `sleeper_market_diagnostics.json` | **market** | |

war-room's own docs are explicit that FantasyPros rankings and ADP are market
signals and must not enter player value: a test named
`test_adp_never_enters_value` moves two players' ADP to 200 and 1 and asserts no
value changes. That constraint is respected here.

---

## 3. Field inventory

Aggregates only. "Proven" means the meaning is established by war-room source
code or by a vendor-facing extraction script in that repo; "inferred" means it
follows from code but rests on an unstated assumption; "unknown" means it cannot
be established from the material available.

### 3.1 `projections_2026.csv` — season-long component stat lines

626 rows, 19 columns. Header stable across all copies. Sparsity is positional:
passing columns are populated only for quarterbacks.

| Column | Type | Non-null | Min | Median | Max | Meaning |
|---|---|---:|---:|---:|---:|---|
| `Rank` | int | 100% | 0 | 312.5 | 625 | Source's own ordering. Not used by war-room. |
| `Name` | text | 100% | | | | 626 unique. Join key. |
| `Pos` | text | 100% | | | | 6 distinct incl. non-skill; war-room filters to QB/RB/WR/TE. |
| `Attempts` | float | 12.3% | | | | Pass attempts (QBs). Unused in scoring. |
| `Comps` | float | 12.1% | | | | Completions. Unused in scoring. |
| `Pass Yards` | float | 12.1% | | | | **Used.** |
| `Pass TDs` | float | 12.1% | | | | **Used.** |
| `Ints` | float | 12.1% | | | | **Used.** |
| `Receptions` | float | 74.6% | 1 | 14 | 108.5 | **Used.** |
| `Rec Yards` | float | 74.6% | 5 | 139 | 1400 | **Used.** |
| `Rec TDs` | float | 47.9% | | | | **Used.** |
| `Rec FD` | float | 74.6% | 0.5 | 13.9 | 140 | Receiving first downs. Not in this league's scoring. |
| `Rush Attempts` | float | 57.5% | 1 | 6 | 287 | Unused in scoring. |
| `Rush Yards` | float | 58.5% | 1 | 22 | 1249.5 | **Used.** |
| `Rush TDs` | float | 22.4% | | | | **Used.** |
| `Rush FD` | float | 58.5% | 0.1 | 2.2 | 140.6 | Rushing first downs. Not in this league's scoring. |
| `Fumbles` | float | 20.0% | | | | **Used** as fumbles *lost* — see §6.2, this is an **inference**. |
| `Projections` | float | 100% | 0.6 | 60.55 | 353.93 | The vendor's own fantasy total. **Full PPR — must not be used.** See §4. |
| `7-Day Delta` | float | 100% | −115.4 | 0 | 97.2 | Change in the vendor's total over 7 days. Unused. |

* **Unit**: counting stats, **season-long totals**, not per game.
* **Horizon**: one season. Which weeks it spans is **UNKNOWN** — see §6.1.
* **Conditional on playing?** **UNKNOWN**. Whether the vendor has already
  discounted for expected missed games is not stated anywhere in war-room, and
  it matters: see §6.3.
* **Mean or median?** **UNKNOWN.** Nothing in the file, the code or the docs
  says which. The column is a point projection with no distributional label.

### 3.2 `draftsharks_injury_profiles.json` — injury/availability

331 players (WR 137, RB 94, TE 59, QB 41). Extracted from an authenticated HAR
capture of a subscriber page; **there is no refresh path and war-room documents
that one should not be built.**

| Field | Non-null | Min | Median | Max | Mean | Meaning |
|---|---:|---:|---:|---:|---:|---|
| `injury_prob` | 100% | 0.05 | 0.35 | 0.88 | 0.348 | Vendor's P(significant injury this season). |
| `proj_games_missed` | 100% | 0.1 | 1.2 | 4.5 | 1.474 | Vendor's expected games missed. |
| `durability` | 100% | 0.03 | 5 | 5 | 4.411 | Vendor composite, higher = more durable. |
| `positional_risk_group` | 99.7% | 1 | 2 | 5 | 2.167 | Vendor bucket. |
| `injury_count` | 100% | 0 | 5 | 24 | 5.598 | Count of historical injuries in the payload. |
| `ds_projection_pts_ppr` | 100% | 0 | 119.5 | 375.8 | 138.9 | Vendor's own **full-PPR** total. |
| `ds_value_rank_overall` | **0%** | | | | | **Entirely null.** Dead field. |

**Two documentation mismatches found, both in war-room's favour to know:**

1. `warroom/injury_profile.py` states `injury_prob` is "0.25-0.78 observed" and
   `proj_games_missed` "0.5-2.7". The current file has **0.05–0.88** and
   **0.1–4.5**. The docstring describes an earlier snapshot.
2. The same docstring calls `durability` "a 1-5 composite". Its minimum is
   **0.03**, so it is not on a 1–5 scale in this file.

`last_update_time` is **null**, so the capture cannot be dated from the file.

### 3.3 `fits.json` — fitted outcome distribution

Fitted 2026-08-31 over seasons **2018–2024** from `data/nflverse` weekly results.

| Fit | QB | RB | WR | TE | Sample |
|---|---:|---:|---:|---:|---:|
| `player.weekly_cv` | 0.4411 | 0.6225 | 0.6493 | 0.7379 | 1,361 player-seasons |
| `availability.weekly_miss` | 0.0526 | 0.0701 | 0.0693 | 0.1056 | 14,795 player-weeks |

These are the **only** proven dispersion and hazard quantities found. Their
exact definitions are in §6.4 and their lineage in `PLAYER_DATA_LINEAGE.md`.

### 3.4 `fantasypros_2026.csv` — expert labels, bye weeks, market

944 rows.

| Column | Non-null | Domain | Meaning |
|---|---:|---|---|
| `RK` | 99.8% | 1–942 | Expert consensus rank. **Market**, not performance. |
| `TIERS` | 100% | 1–16 | Tier bucket. Unused by war-room. |
| `PLAYER NAME` | 99.8% | 942 unique | Join key. |
| `TEAM` | 99.8% | 34 distinct | **Used** for NFL team. |
| `POS` | 99.8% | | Position with rank suffix. |
| `BYE WEEK` | 84.3% | 5–14, median 10 | **Used.** Real bye weeks. |
| `UPSIDE ` | **28.5%** | `-` (673), `1–5 out of 5` (269) | Ordinal expert label. |
| `BUST ` | **28.5%** | `-` (673), `1–5 out of 5` (269) | Ordinal expert label. |
| `SOS SEASON` | 84.3% | `0–5 out of 5 stars` | Strength of schedule. Unused. |
| `ECR VS. ADP` | 33.7% | signed | **Market.** ADP-fallback input only. |

Note the trailing spaces in the `UPSIDE ` and `BUST ` header names, and that
**71.3% of rows carry no upside/bust label at all**.

### 3.5 `ffc_10team_halfppr_2026.csv` — market

192 rows: `ADP` (1.5–181.8), `Std Dev` (0.6–46.1), `High`, `Low`,
`Times Drafted` (5–886). **Market price data. Not performance truth.**

### 3.6 `players_provisional.csv` — SYNTHETIC, not real

448 rows: `player_id`, `name`, `pos`, `nfl_team` (integer 0–31, a team *index*),
`depth_rank` (0–5), `ppg_baseline` (0.949–26.304), `adp` (a dense 1–448 rank).

**This file is machine-generated, not observed.** `warroom/players.py` builds it
in `generate_provisional_pool()` / `write_provisional_pool()` from
`ROLE_PPG`, which is registered as `P.uncalibrated(...)` with the comment
"Placeholder shape only; must be fit from weekly usage data."

**It must not be imported as real player data.** Its `ppg_baseline` is a
placeholder depth-chart tier value, and its `adp` is a synthetic dense rank.

---

## 4. Scoring compatibility

**The projections are stat lines, not fantasy points** — which is the good case
for this engine, because `ceauction.scoring.score_statline` is exactly that seam.

**The vendor's own `Projections` column must not be used.** war-room solved the
scoring system out of the file by least squares and recovered a reception
coefficient of **0.98** and an interception coefficient of **−1.02**: it is
**full PPR**, not half. war-room re-ran that regression on the refreshed file
and it fails identically, with phantom points of **WR +15.3, RB +8.2, QB +6.3,
TE +12.1**. Taking the column at face value would tilt the whole board toward
pass catchers.

### 4.1 This league's scoring against the available columns

| Rule | This league | Source column | Status |
|---|---|---|---|
| Passing yard | 0.04 | `Pass Yards` | **supported** |
| Passing TD | 4 | `Pass TDs` | **supported** |
| Interception | −2 | `Ints` | **supported** |
| Rushing yard | 0.1 | `Rush Yards` | **supported** |
| Rushing TD | 6 | `Rush TDs` | **supported** |
| Receiving yard | 0.1 | `Rec Yards` | **supported** |
| Receiving TD | 6 | `Rec TDs` | **supported** |
| Reception | 0.5 | `Receptions` | **supported** |
| Fumble lost | −2 | `Fumbles` | **PARTIAL** — the column is `Fumbles`, not "fumbles lost". See §6.2. |
| Passing 2-pt conversion | 2 | — | **MISSING** |
| Rushing 2-pt conversion | 2 | — | **MISSING** |
| Receiving 2-pt conversion | 2 | — | **MISSING** |
| Individual special-teams TD | 6 | — | **MISSING** |

**Four of this league's thirteen scoring rules have no source column.** war-room
has the same four gaps and its `half_ppr_points()` simply omits them, which is
correct for it because its league config carries only the nine rules it can
compute.

**These must not be silently treated as zero.** They are small but not nil: a
returner with two special-teams touchdowns is 12 points, and 2-point
conversions accrue league-wide. The honest handling is an explicit
`unsupported_categories` list on the input contract so a projection is never
mistaken for complete. The schema in `schemas/real_player_input_v1.schema.json`
requires exactly that.

---

## 5. Do median outcomes, injury odds and boom/bust grades actually exist?

The three things the next-step plan named. Answers, with the evidence.

### 5.1 "Median stat outcomes" — **FOUND, but NOT proven to be medians**

Season-long component stat projections exist and are the real thing this engine
needs. But **nothing in the file, the code or the documentation states whether
they are means, medians or modes.** war-room consumes them as a point estimate
and never needs to know.

This matters directly. `PlayerSpec.base_mean` is documented as **expected
fantasy points** — a mean. If these are medians, then for right-skewed
categories (touchdowns especially) the median is below the mean and every
`base_mean` would be biased low by an amount that varies by position. The
distinction is not cosmetic and cannot be resolved from this material.

**Status: EXISTS / meaning UNKNOWN.** Needs the vendor's definition.

### 5.2 "Injury odds" — **FOUND, meaning partly proven**

`injury_prob` and `proj_games_missed` exist for 331 players, with proven
*extraction* lineage (a named vendor field inside `sipPlayerProfile`).

What is **not** proven is the vendor's definition of either:

* `injury_prob` is P(a **significant** injury) — "significant" is the vendor's
  word, undefined here. Any injury? One causing a missed game? An IR stint?
* `proj_games_missed` — over how many games? Regular season only? Does it
  include the fantasy playoff weeks?

`proj_games_missed` is the more usable of the two because it has a unit
(games), and war-room already derives `available_share = 1 − missed/17`
clamped to [0.3, 1.0] from it.

**Status: EXISTS / definitions UNKNOWN.** 331 of 626 projected players covered
(52.9%), so coverage is itself a gating problem — war-room gates the join at
45 of the top 60 by value for exactly this reason.

### 5.3 "Boom/bust grades" — **FOUND, but they are NOT a distribution**

`UPSIDE ` and `BUST ` exist. They are **ordinal expert labels on a 1–5 scale**
("4 out of 5"), present on **269 of 944 rows (28.5%)**, with `-` for the rest.

They are emphatically **not** dispersion estimates. war-room's own analysis
records `corr(upside tag, projected pts) = +0.530` and
`corr(upside tag, injury_prob) = +0.247` — the tag is substantially a restatement
of "this player is good" and "this player is hurt a lot", so war-room regresses
it on both within position and uses only the residual, which correlates −0.000
with points. It uses that residual for **bench ordering only**, never for value.

**There is no proven mapping from these grades to a standard deviation, and one
must not be invented.** That is precisely the speculative mapping this phase was
told not to build.

**The real dispersion source is `fits.json`**, not the grades — see §6.4.

**Status: EXISTS / not a distributional quantity / no mapping to `week_sd`.**

---

## 6. Unresolved semantic questions

These need Avrohom's answer or vendor documentation. None can be settled from
the material inspected, and each is marked **UNKNOWN** in the mapping table.

**Q1 — What weeks does the projection span, and is 16 the right divisor?**
war-room divides season totals by a hard-coded `GAMES_IN_WINDOW = 16` ("weeks
1–17 with one bye inside that window"). Whether the vendor projected 17 games,
16, or a games-played estimate is unstated. A wrong divisor scales every
`base_mean` by a constant factor and would be invisible in testing.

**Q2 — Is `Fumbles` fumbles, or fumbles *lost*?** This league scores −2 for a
fumble **lost**. war-room's `half_ppr_points()` multiplies `Fumbles` by
`fumble_lost`, which is only correct if the column already means lost fumbles.
Roughly half of all NFL fumbles are recovered by the offence, so if the column
is total fumbles this over-penalises by about a factor of two. `fumble_lost` is
also flagged in war-room's own config as the one scoring value still a platform
default, never confirmed by the owner.

**Q3 — Are the projections conditional on playing, or already availability-discounted?**
If the vendor has already discounted for expected missed games, then applying
`proj_games_missed` on top double-counts availability. war-room applies its own
haircut only from a `data/board/injuries.csv` that does not exist by default,
so its current board does not resolve this either.

**Q4 — Mean or median?** §5.1. Determines whether `base_mean` is biased low.

**Q5 — What does DraftSharks mean by a "significant" injury, and over what horizon?**
§5.2.

**Q6 — Which league is this engine for?** war-room's `config/league.json` is
**10 teams**, starters `QB 1 / RB 2 / WR_TE 3 / FLEX 2` — **no superflex** —
sourced from "Sleeper league settings for league_id 1385659541554745344, read
2026-09-03". This engine's `SPEC.md` specifies **12 teams**, 15-man rosters, and
a **SUPERFLEX**. These are different league configurations. Either they are two
different leagues, or one of the two records is stale. Every CE number this
engine produces depends on which is right.

**Q7 — How stale is the injury capture?** `last_update_time` is null and there
is no refresh path by design. The data cannot be dated from the file.

**Q8 — Is the FantasyPros export's 28.5% upside/bust coverage a scrape artefact
or the vendor's own limit?** 673 of 944 rows carry `-`.

### 6.4 What *is* proven: the two fitted quantities

These are the only two quantities found with a fully proven mathematical
definition, because war-room computes them itself from real historical data
rather than receiving them from a vendor.

**`player.weekly_cv[pos]`** — for each player with ≥8 appearances in a season
and a mean above a positional contributor threshold (QB 10.0, RB 6.0, WR 6.0,
TE 4.0 half-PPR), compute `std(weekly half-PPR) / mean(weekly half-PPR)` over
**the weeks he appeared**; then take the unweighted mean of that ratio across
qualifying player-seasons, by position. Half-PPR is derived as
`fantasy_points_ppr − 0.5 × receptions`. Regular season, weeks 1–17, 2018–2024,
n = 1,361.

*Consequences to carry:* it is a **within-player** CV averaged across players
(not a pooled CV); it is **conditional on appearing**, so it excludes missed
weeks rather than scoring them zero; and it is computed with NumPy's default
population standard deviation (`ddof=0`).

**`availability.weekly_miss[pos]`** — P(a player does not appear in week *w+1*
| he scored at least the contributor threshold in week *w*), with bye weeks
excluded from both numerator and denominator. Regular season, weeks 1–16
transitions, 2018–2024, n = 14,795 player-weeks.

*Consequences to carry:* the cohort is **contributors**, deliberately —
unconditioned, quarterbacks appear to miss 42.6% of weeks because most rostered
QBs never play. And it counts **any** non-appearance, so it includes benchings,
rest and trades, not only injuries.

---

## 7. What was read, and what was not touched

Read-only inspection of the four candidate directories. **Nothing in any
war-room directory was created, modified or deleted.** The one modified file in
war-room's working tree (`data/adp/ffc/coverage.csv`) was already modified
before this inventory began and is unrelated to it.

Raw data was copied into a git-ignored `local_data/` inside this repository for
statistics only, and is not committed.
