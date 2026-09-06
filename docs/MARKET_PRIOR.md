# MARKET_PRIOR.md

What Sleeper's number is, what it is not, and how it becomes a provisional
expected clearing price for this league.

> **Every committed example uses fabricated anchors.** The real Sleeper export
> lives in an ignored `local_data/` path and no player-level output from it is
> in version control. Nothing in this document is a real market price.

---

## 1. Four quantities, kept apart

| | What it is | Built? |
|---|---|---|
| **`sleeper_display_anchor`** | What managers see on Sleeper. A number on a screen | **Yes** — loaded and preserved exactly |
| **`expected_clearing_price`** | What *this* room may actually pay | **Yes** — this phase, provisional |
| **`ce_reservation_range`** | What our championship-equity engine says we can afford | Elsewhere: `ceauction.auction.reservation` |
| **`tactical_max_bid`** | What to actually bid, given who else can bid and how their money is committed | **Not built anywhere** |

Collapsing any two of them is the failure this package is arranged to prevent.
The raw value, the displayed anchor, the format adjustment, the budget
reconciliation and the live-room adjustment all stay separately inspectable, and
the source anchor is never overwritten by a transformation.

## 2. What the anchor is

The response of

```
https://api.sleeper.com/players/nfl/values/regular/2026/2qb
```

— Sleeper's **generic 2026 superflex (`2qb`) projected value list**. 1,024 rows,
of which 149 are positive, rosterable in this league, and active. Those 149
total **$2,384.26** raw and **$2,379** displayed.

| Position | Players | Raw | Displayed |
|---|---:|---:|---:|
| QB | 29 | $541.12 | $539 |
| RB | 45 | $802.77 | $799 |
| WR | 56 | $828.61 | $829 |
| TE | 19 | $211.76 | $212 |

Six other positive players are excluded from budget reconciliation: five
defenses this league cannot roster, and one inactive quarterback. Money the room
cannot spend is not part of the room's money.

**Display rounding is decimal `ROUND_HALF_UP`**, fixed by four observed pairs:
58.19 → 58, 33.69 → 34, 12.78 → 13, and 16.50 → **17**. Python's built-in
`round()` is banker's rounding and gives 16 for the last, so it is not used and a
test asserts the difference.

**Nonpositive values have no displayed price here.** Sleeper's UI treatment of
them was never observed, so none is invented. They are carried as *unpriced*,
which is a different claim from a predicted $1 sale and is kept apart from it.

## 3. What it is not

**Not this league's format.** The endpoint says `2qb`. This league is
*superflex*: the flexible slot accepts a QB, RB, WR or TE, so a team can start a
running back there and never draft a second quarterback. Generic
two-quarterback pricing assumes a firmer floor under quarterbacks than this
league has.

**Not priced for a lineup without a tight-end slot.** This league has **no
dedicated TE slot**. A tight end competes with receivers for the three WR/TE
seats and with everyone for the flex. A list built for a required-TE format
overstates the position's floor here, and the user's expectation that standard
Sleeper TE prices are substantially too high is exactly this mismatch.

**Not a price at all.** See the next section.

## 4. Why direct prices violate the room budget

| | |
|---|---:|
| Nominal room budget | $2,400 |
| Roster slots | 180 |
| Committed at $1 per slot | $180 |
| Discretionary | **$2,220** |
| 149 priced anchors, at list | **$2,379** |
| Slots still to fill after those 149 | 31 |

The priced anchors alone exceed the discretionary pool *before* the other 31
players are bought. Paying list is not merely unlikely in this room — it is
arithmetically impossible. A reconciliation step is therefore mandatory rather
than a refinement.

## 5. How the adjustment works

Two stages, both inspectable, neither fitted.

### Format credibility

Credibility is a **weight on the anchor**, not a discount, and it comes out of
the lineup graph:

| Position | Weight (base) | Learn rate | Why |
|---|---:|---:|---|
| RB | 0.750 | 0.25 | Two dedicated slots plus flex. Format matches |
| WR | 0.675 | 0.25 | Three WR/TE slots plus flex. Format matches |
| QB | 0.413 | 0.35 | One dedicated slot plus a superflex an RB/WR/TE may fill |
| TE | 0.338 | **0.55** | **No dedicated slot at all** |

A player is shrunk toward his position's centre by one minus the weight. Full
credibility leaves the anchor alone; zero says the room ignores the list and
treats the position alike, which is the honest limit of "this number was
computed for another lineup".

**Geometric, not arithmetic.** Shrinking a $1 player toward a $19 positional
mean triples him, which is not what partial disbelief in a list means. A
weighted geometric mean shrinks multiplicatively — cheap players stay cheap,
expensive ones come down — and is strictly increasing, so it cannot reorder a
position.

**Why a weight and not a fixed TE discount.** A hand-coded penalty could not be
wrong, and therefore could not be learned from. Credibility can fall *and rise*
with evidence, and the tight end's high learn rate means the first few sold move
it fastest. The anchor itself is preserved in full either way.

### Budget reconciliation

One monotone scale per scenario: every format-adjusted price is multiplied by
the same factor, chosen so the draftable board sums to that scenario's spendable
dollars once the $1 floor is honoured. Monotone by construction, so it cannot
reorder anyone, and one number a reader can check.

| Scenario | Spend rate | Adherence | Factor | Board | Target |
|---|---:|---:|---:|---:|---:|
| low | 0.88 | 0.55 | 1.0804 | $1,957 | $1,954 |
| base | 0.95 | 0.75 | 1.1734 | $2,107 | $2,109 |
| high | 1.00 | 0.92 | 1.2399 | $2,213 | $2,220 |

*(Real anchors. Each board lands within integer rounding of its target; rounding
149 prices to whole dollars cannot hit one exactly, so the achieved total is
reported beside it rather than asserted equal.)*

Resulting positional shift, real anchors: TE moves furthest from list (0.736 of
displayed), then QB (0.855), WR (0.885), RB (0.946). That ordering comes from
the lineup structure, not from a table of preferences.

**An alternative was considered and rejected.** A replacement-level
transformation — subtracting a positional baseline before scaling — is
defensible and spreads the cut differently. It is not used because it needs a
replacement level nothing in this repository measures, and inventing one would
bury a second unfitted assumption inside the step whose whole job is to be
inspectable.

**Nothing here is fitted.** There is no historical auction for this league or a
comparable one. Every coefficient is a stated scenario, and the label travels
with the number to every place it surfaces.

## 6. How live observations update the prior

Partial pooling across four nested levels — room, position, tier, buyer — each
with its own configurable prior strength. An observation moves a level in
proportion to how much evidence it already has, so the first tight end sold moves
the tight-end adjustment noticeably and the twentieth barely does.

**Room and position are collinear until sales span positions.** With only tight
ends sold, "this room is cheap" and "this room is cheap on tight ends" fit the
data equally well. The room level therefore takes the residual net of the
position's current estimate *and* scales its evidence by how many distinct
positions have contributed — zero credit with one. An earlier version without
this cut an unrelated quarterback by 37% on six tight-end sales.

**Outliers are capped before pooling.** A $200 sale against a $3 prior enters at
its capped value and moves the position by about 1.15x rather than 66x.

**Ratios only where they mean something.** A ratio against a $1 prior is
arithmetic noise, so log ratios are used above a threshold and additive dollar
residuals carry the cheap players.

**Uncertainty narrows but never vanishes.** A nonzero floor holds regardless of
how many sales arrive: a room that has bought forty players is better
understood, not solved.

Deterministic: identical observations in identical order give an identical
state, and the state serializes and refuses to replay onto a different prior.

## 7. What a final sale does and does not reveal

**Does:** the room cleared this player at this price; at least two bidders were
willing to reach one dollar below it; the winner was willing to reach it. That
is a *clearing price*, and it is what the updater learns from.

**Does not:** the winner's maximum. He stopped because everyone else did, so his
true ceiling is somewhere at or above what he paid and nothing here can say
where. Nor any losing bidder's maximum, beyond it being below the final price.

Every report from the updater repeats this, because an updater that quietly
treated a sale as a revealed valuation would be inventing the most valuable data
in the auction.

## 8. Room pressure

For a nominated player and price, four questions get four answers:

* **financial ability** — budget less a dollar for every other open slot;
* **roster legality for this player** — an owner one slot from the end who still
  needs a receiver cannot bid on a quarterback at any price;
* **structural roster fit** — an unfilled seat, from the slot rules. A weak
  signal, labelled weak;
* **predicted clearing pressure** — the market model's scenario range and how
  much of it is evidence.

It does **not** predict who wins. No bidder preference has ever been observed
for this league.

## 9. Commands

```bash
# aggregates from the real anchor file (nothing player-level is printed)
ce-lab market ingest-sleeper --csv local_data/sleeper_2qb_values_2026_clean.csv

# join to the contract; names go only to an ignored local path
ce-lab market audit --csv local_data/sleeper_2qb_values_2026_clean.csv \
    --contract local_data/real_player_contract_v1.json \
    --json-out local_data/market/audit.json

# the prior, fabricated or real
ce-lab market build-prior --demo
ce-lab market build-prior --csv local_data/... \
    --players-out local_data/market/player_prices.json

# fold sales in and see what moved
ce-lab market observe-sale --demo --demo-sales
ce-lab market observe-sale --demo --sale KEY:TE:3:Owner01 --state-out /tmp/s.json

# one player, and the room
ce-lab market player --demo --key fabricated_te000 --demo-sales
ce-lab market room-pressure --demo --price 20 --demo-sales

# populate the acquisition-cost contract
ce-lab market cost-book --demo --scenario base --model-scenario fh-f000-s000-w1
```

Committed fabricated outputs live in `docs/examples/market_*.txt`.

## 10. Limitations

* **Every coefficient is a stated scenario.** Spend rate, anchor adherence, the
  four credibility weights, learn rates, prior strengths and the outlier cap are
  all chosen, not estimated. No historical auction exists to fit them to.
* **The anchor is for the wrong format**, in two known ways, and the correction
  is a modelled weight rather than a measurement.
* **Buyer-level learning is thin.** An auction produces at most 15 purchases per
  owner, and the model refuses to characterise anyone on fewer than two.
* **Roster fit is structural.** It counts unfilled seats; it does not know what
  a manager wants.
* **Nothing predicts a winner**, and nothing here is a bid.
* **Position tiers are coarse** — four buckets — because a finer partition would
  spread an auction's few observations too thin to accumulate.

## 11. Next phase

**Calibrate the scenario coefficients against the first live auction, and only
then consider a tactical layer.**

Concretely: run the draft with `observe-sale` recording every sale, then compare
the recorded clearing prices against the prior's low/base/high bands. That gives
the first real evidence about spend rate, anchor adherence and per-position
credibility — the six numbers currently chosen rather than estimated. One
auction is a small sample and will not settle them, but it converts them from
assumptions into assumptions with a residual.

A tactical max bid needs three things this repository still lacks: a bidder
model, simultaneous multi-owner completion (the CE side's largest simplification
is that eleven rosters are frozen), and the runtime budget to combine them
inside a bid timer. None should be attempted before the market band above has
been checked against a real room even once.
