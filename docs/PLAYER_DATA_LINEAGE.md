# PLAYER_DATA_LINEAGE.md

Source file → transformation code → derived field, for the existing player model
("war-room") evaluated as a real-data source for this engine.

**Proven** lineage means the transformation was read in war-room's source and
the mathematics is fully determined by it. **Inferred** means the code is clear
but rests on an assumption the code does not state. **Unknown** means the step
happens outside any material available here — chiefly inside a vendor.

No transformation below has been implemented in this repository. This is a
trace, not an import.

---

## 1. The value path — projections to points

```
  [vendor, unknown process]
        │
        ▼
  projections_2026.csv            season-long component stat lines
   (a.k.a. season_long_proj_table-2.csv — same logical file, two names)
        │
        │  warroom/board.py :: half_ppr_points(row, scoring)          PROVEN
        │     season_pts = 0.04·Pass Yards + 4·Pass TDs + (−2)·Ints
        │                + 0.1·Rush Yards + 6·Rush TDs
        │                + 0.1·Rec Yards  + 6·Rec TDs
        │                + 0.5·Receptions + (−2)·Fumbles
        │     coefficients read from config/league.json → scoring
        │     blanks coerced to 0.0 by _num(v, default=0.0)
        ▼
  season fantasy points (this league's scoring)
        │
        │  warroom/board.py :: load_board                             PROVEN
        │     if a weeks_out entry exists for the player:
        │        season *= max(0, 1 − weeks_out / 16)
        │     (source data/board/injuries.csv — ABSENT by default,
        │      so this branch does not fire on the current board)
        │
        │     ppg = season / GAMES_IN_WINDOW, GAMES_IN_WINDOW = 16
        ▼
  ppg (per-game half-PPR baseline)  →  PlayerPool.ppg
```

**Proven:** the arithmetic, the coefficient source, the divisor, and the
blank→0 coercion.

**Inferred, not proven:**

* that `Fumbles` means *fumbles lost* (the scoring rule is for lost fumbles; the
  column is named `Fumbles`) — inventory Q2;
* that 16 is the right divisor for whatever window the vendor projected —
  inventory Q1;
* that a blank stat cell means zero rather than "not projected". `_num` returns
  0.0 for `""`, `"-"` and `"nan"` alike. With `Rush TDs` populated on only 22.4%
  of rows and `Rec TDs` on 47.9%, this assumption is load-bearing for a large
  share of the file.

**Unknown:** everything upstream of the CSV. The vendor's estimator, whether the
figures are means or medians, whether they are conditional on playing, and what
week range they cover are all outside this material.

### 1.1 The rejected path — the vendor's own total

```
  projections_2026.csv :: Projections column
        │
        ▼
  REJECTED — it is full PPR, not half.
```

**Proven by measurement, twice.** war-room solved the scoring system out of the
file by least squares and recovered a reception coefficient of **0.98** and an
interception coefficient of **−1.02**. It re-ran the same regression after the
2026-09-03 refresh and it fails identically, with phantom points of **WR +15.3,
RB +8.2, QB +6.3, TE +12.1**.

This is a lineage fact worth carrying into any ingestion: *a vendor's fantasy
total is a scoring assumption, not a measurement*, and this one disagrees with
the league it was going to be used for.

---

## 2. The availability path

```
  draftsharks.com/injury-predictor   (subscriber page, server-rendered)
        │
        │  authenticated browser session → HAR capture
        │
        │  scripts/extract_injury_profiles.py                          PROVEN
        │     locate `var vueAppData = {…}` in the page HTML
        │     brace-counted extraction (too large/irregular for regex)
        │     for each p in data["playerData"]:
        │        prof = p["sipPlayerProfile"]
        │        injury_prob           ← prof["injury_prob"]
        │        proj_games_missed     ← prof["proj_games_missed"]
        │        durability            ← prof["durability"]
        │        positional_risk_group ← prof["positional_risk_group"]
        │        injury_count          ← len(p["sipInjuries"])
        │        ds_projection_pts_ppr ← p["projection"]["fantasyPtsPpr"]
        ▼
  draftsharks_injury_profiles.json    331 players
        │
        │  warroom/injury_profile.py                                   PROVEN
        │     available_share = min(1.0, max(0.3, 1 − proj_games_missed/17))
        ▼
  InjuryProfile.available_share
```

**Proven:** the extraction is a straight field copy — no arithmetic is applied
between the vendor's value and the JSON. The `available_share` formula and its
[0.3, 1.0] clamp are proven.

**Unknown:** what the vendor's `injury_prob` and `proj_games_missed` *mean* —
inventory Q5. The extraction proves *where the numbers came from*; it proves
nothing about *what they measure*. The capture also carries no
`last_update_time`, so it cannot be dated.

**Note the divisor disagreement inside war-room itself:** the board scales by
`weeks_out / 16` while `available_share` divides by `17`. Both are defensible
(scoring window vs. NFL season) but they are not the same denominator, and an
import should pick one deliberately.

---

## 3. The distribution path — the only fitted quantities

```
  nflverse weekly results, 2018–2024
   data/nflverse/stats_player_week_{year}.parquet
        │
        │  warroom/calibrate/fit.py :: _load(year)                     PROVEN
        │     filter season_type == "REG", position ∈ {QB,RB,WR,TE}, weeks 1–17
        │     half = fantasy_points_ppr.fillna(0) − 0.5 × receptions.fillna(0)
        │
        ├──►  fit_weekly_cv(seasons, min_games=8)                      PROVEN
        │        for each (player, season):
        │           require ≥8 appearances and mean(half) ≥ threshold[pos]
        │           cv_ps = std(half) / mean(half)        ← ddof=0, appeared weeks only
        │        value[pos] = unweighted mean of cv_ps over qualifying player-seasons
        │        thresholds: QB 10.0, RB 6.0, WR 6.0, TE 4.0
        │        n = 1,361 player-seasons
        │        → QB 0.4411, RB 0.6225, WR 0.6493, TE 0.7379
        │
        └──►  fit_availability(seasons)                                PROVEN
                 bye_weeks() infers each team's bye as the single week it fields nobody
                 for each player, for w in 1..16:
                    skip unless half[w] ≥ threshold[pos]      ← cohort: contributors
                    skip if w+1 is that team's bye            ← a bye is certain, not a hazard
                    denominator += 1
                    numerator   += 1 if half[w+1] is absent
                 value[pos] = numerator / denominator
                 n = 14,795 player-weeks
                 → QB 0.0526, RB 0.0701, WR 0.0693, TE 0.1056
        ▼
  data/calibration/fits.json
```

**Fully proven, both.** These are the only quantities in the whole inventory
whose mathematical definition is established end to end, because war-room
computes them from raw historical results rather than receiving them.

Properties that must travel with them if they are ever used:

* `weekly_cv` is a **within-player** CV averaged across players, not a pooled
  CV. Those differ.
* `weekly_cv` is **conditional on appearing** — missed weeks are excluded, not
  scored as zero. This engine's `week_sd` describes weeks a player *plays*, so
  the conditioning matches, but that agreement should be asserted rather than
  assumed.
* `weekly_miss` counts **any** non-appearance: injury, benching, rest, trade,
  inactive. It is an availability hazard, not an injury hazard.
* The contributor cohort is deliberate and consequential: unconditioned, QBs
  appear to miss 42.6% of weeks because most rostered QBs never play.

---

## 4. The label path — upside and bust

```
  fantasypros_2026.csv :: "UPSIDE ", "BUST "      (note trailing spaces)
        │
        │  warroom/board.py :: _stars(v)                               PROVEN
        │     regex ^\s*(\d)\s*out of\s*5  →  int, else None
        ▼
  pool.upside / pool.bust      ordinal 1–5, sentinel −1 for missing
        │
        │  warroom (bench_values) — orthogonalisation                  PROVEN
        │     regress the tag within position on projected points, and on
        │     injury_prob where a profile exists; keep the RESIDUAL
        │     raw corr(upside, points) = +0.530  →  residual corr = −0.000
        │     residual sd 1.03 → 0.80 (the removed 23% was double-counted)
        │     players with no tag get exactly 0 — "no tag is not an average tag,
        │     but it is no evidence"
        ▼
  bench ordering only — explicitly NOT player value
```

**Proven:** the parse, the sentinel, the orthogonalisation and the restriction
to bench ordering.

**Unknown and not to be invented:** any mapping from a 1–5 ordinal label to a
standard deviation. Nothing in war-room attempts one. The tag is an expert
label whose raw form is 53% explained by how good the player is; it is not a
measurement of dispersion. `fits.json` is the dispersion source.

---

## 5. The market path — traced only to keep it out

```
  ffc_10team_halfppr_2026.csv  (ADP, Std Dev, High, Low, Times Drafted)
  fantasypros_2026.csv :: RK, "ECR VS. ADP"
  data/adp/ffc/**  (historical ADP)
        │
        │  warroom/market.py, rebuild_market.py, adp_sd.py
        ▼
  draft-market signals: survival, falls_by, timing, cliffs
```

**Proven that this path never reaches value.** war-room's test
`test_adp_never_enters_value` moves two players' ADP to 200 and to 1 and asserts
**no** value changes, while its partner test asserts survival *does* move — so
the first cannot pass on a board that has quietly started using the market as a
projection.

For this engine's purposes the market path is out of scope entirely. It is
recorded here so that a future ingestion does not mistake `ADP`, `RK`,
`Std Dev` or `ECR VS. ADP` for player-performance truth. In particular
FantasyPros' `Std Dev` is the **dispersion of draft position across mock
drafts** — a measure of market disagreement, not of weekly scoring variance.

---

## 6. The synthetic path — what is not real at all

```
  warroom/players.py :: ROLE_PPG      P.uncalibrated(...)
     "Placeholder shape only; must be fit from weekly usage data."
        │
        │  generate_provisional_pool() → write_provisional_pool()      PROVEN
        ▼
  data/players_provisional.csv     448 rows
```

**Proven synthetic.** `ppg_baseline` is a placeholder depth-chart tier value,
`nfl_team` is an integer index rather than an abbreviation, and `adp` is a dense
1–448 rank. The file looks like a real player table and is not one.

This is the single most dangerous file in the inventory, because its column
names (`name`, `pos`, `nfl_team`, `ppg_baseline`, `adp`) read exactly like a
real board. **It must never be imported as real data.**

---

## 7. Summary of lineage status

| Derived quantity | Lineage | Meaning |
|---|---|---|
| season fantasy points from components | **proven** | proven given Q1/Q2/Q3 |
| per-game baseline (`÷ 16`) | **proven** | divisor **inferred** (Q1) |
| `Fumbles` → fumble-lost penalty | **proven** arithmetic | column meaning **inferred** (Q2) |
| vendor `Projections` column | **proven full-PPR** | proven wrong for this league |
| `injury_prob`, `proj_games_missed` | **proven** extraction | vendor definition **unknown** (Q5) |
| `available_share` | **proven** | inherits Q5 |
| `player.weekly_cv` | **proven** | **proven** |
| `availability.weekly_miss` | **proven** | **proven** |
| `upside` / `bust` ordinal | **proven** parse | **not a distribution**; no sd mapping exists |
| upside residual | **proven** | bench ordering only |
| `players_provisional.csv` | **proven synthetic** | not real data |
| ADP / RK / ECR / Std Dev | **proven** market | must not enter value |
