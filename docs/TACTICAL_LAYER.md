# The tactical layer

What converts a price band into a bid. Everything below is fabricated-data
machinery plus stated assumptions; nothing in it is fitted to anything.

Read `docs/TACTICAL_WALKTHROUGH.md` for a worked fabricated example, and
`docs/AUCTION_LAYER.md` / `docs/MARKET_PRIOR.md` for the layers underneath.

## The four prices, and why they never merge

| name | question | lives in |
|---|---|---|
| Sleeper display anchor | what Sleeper shows for a generic `2qb` lineup | `market/prior.py` |
| expected clearing price | what **this room** may pay, after sales | `market/live.py` |
| CE reservation price | what our equity can afford vs the best alternative | `auction/reservation.py` |
| **tactical maximum bid** | what to actually **bid**, given who else can bid | `tactical/maxbid.py` |

A tactical maximum uses the other three. It never overwrites or relabels one.
`TacticalResult.to_dict()["prices"]` keeps them in separate keys and the
formatter prints them on separate lines.

## Definitions

**Bidder scenario** (`tactical/bidders.py`). A *named reading of how this room
bids*: `anchor_heavy`, `market_base` (the base case), `format_aware`,
`aggressive`, `conservative`. Each is a small set of stated coefficients —
how far to lean on the Sleeper anchor, which point of the clearing band to
centre on, an aggression multiplier, how much a starting seat is worth over
bench depth, a soft share of discretionary money, and a range half-width.
Nothing here is fitted. There is no historical auction for this league.

**Willingness range**. One owner's `low/base/high` for one candidate under one
bidder scenario. Every owner starts with the *same* prior; it differs only
through his money, his open slots and his roster shape. An owner's own
observed purchases adjust him only once he clears
`ShrinkageConfig.min_buyer_observations`, and then only through the market
layer's pooled buyer level. A sale price reveals neither the winner's ceiling
nor any loser's, and the model claims neither.

`pursuit_score` is an **uncalibrated ordinal score**, never a fitted
probability, and `score_kind` says so in every serialization.

**Legal maximum** (`tactical/endgame.py`). The highest price an owner may pay
for *this* candidate: budget, minus $1 held for every other open slot, capped
by whether a legal roster completion still exists afterwards. Found by binary
search, which is exact because legality is monotone in price. Zero means he
cannot buy this player at any price — a different fact from cannot afford.

**Financial-control threshold**. One increment above the highest *rival*
candidate-specific legal maximum. It means exactly one thing: above it, no
other owner may legally bid. **It is not a recommended bid** and carries no
equity claim.

**Named recipient branch** (`tactical/recipients.py`). Who gets the player if
we stop. At minimum: the current leader (always represented when supplied, as a
legal branch or as an explicitly refused one with the shortfall text), the
strongest remaining legal bidder, other materially plausible named owners, and
an `unavailable` branch that removes him from our board without inventing an
owner. **Each branch is evaluated separately.** Weights are attached only after
every branch exists and are labelled stated scenario assumptions.

**Shared-board continuation** (`tactical/board.py`). All unfinished rosters
complete out of *one* pool: no duplicate ownership, legal lineups via the
existing feasibility matcher, $1 held for every open slot at every step,
budgets and slots updated after each allocation, deterministic under a fixed
seed and different under a different one. Reports `exact` / `bounded` /
`truncated` / `heuristic`.

**The focus team bids in it** (`BoardSettings.focus_bids`, default `True`).
This is not optional and the first version got it wrong. With our seat silent,
eleven rivals draft the top of the board against nobody: they never have to
outbid us, they get better players for less money, and the CE search is handed
the leftovers. Measured on the fabricated demo that put our proxy strength at
**86.9 against rivals' 105-109** and our championship equity at **exactly zero
in both branches**, so every audited buy/pass comparison was a difference of
two zeroes. With the focus team bidding, our strength lands at **105.7** inside
the rivals' band and the same comparison returns `+0.045 (se 0.010)`.

What the focus team does *not* do in the continuation is take delivery. Players
it outbids the room for are **held**: kept out of rival rosters, left on our
board, unpaid for, and never counted against our budget twice. Which of them we
actually take is the question the CE completion search exists to answer, and
letting a willingness proxy settle it would replace the real search with the
cheap one.

**Robust tactical maximum**. Highest tested price favorable against *every*
included recipient and *every* selected scenario.

**Base tactical maximum**. Highest tested price favorable under the explicitly
designated base scenario (`is_base=True`), against every recipient.

**Permissive tactical ceiling**. Highest tested price favorable under *at least
one* scenario/recipient. **A ceiling, not the recommended bid.**

**Opening vs live maximum**. Opening uses the initial room and no fabricated
sales. Live applies every recorded purchase, budget, roster, market update, the
current price and the current leader. They are different questions with
different cache keys; the walkthrough shows both.

**Immediate vs audited**. `immediate` is proxy-backed or served from cache,
carries **no confidence interval**, and is not a championship-equity estimate.
`audited` runs the real CE engine through `compare_buy_vs_pass` on matched
seasons with the existing selection/holdout discipline. Every verdict names
which produced it.

## Heuristics and approximations, all of them

1. **Clearing price rule.** Second-highest scenario willingness plus one
   increment, floored at the next legal bid, capped by the winner's willingness
   and his candidate-specific legal maximum. A standard ascending-auction
   approximation. **Not Sleeper's proven mechanism** (`PRICE_RULE`).
2. **Opponent continuations use a named proxy**, not championship equity
   (`OPPONENT_METHOD`). Running CE inside ~150 allocations is not affordable.
3. **Recipient weights** are stated assumptions: proportional to base
   willingness among legal named branches, with a fixed 0.15 residual on the
   unassigned branch.
4. **Immediate-mode value** is our best completion's expected starting points
   *minus the league mean*, so recipient identity can move it. It is a proxy
   ordering, not equity.
5. **Scarcity term** in willingness: a bounded +0-20% when few players of the
   candidate's position remain. Derived from the board, not from a positional
   preference.
6. **Soft budget share** per bidder scenario caps one player's willingness at a
   fraction of discretionary money. Named in `capped_by` when it binds.
7. **Board bounds**: `pool_depth` and `max_allocations` are real cuts; the
   result says when they bit.
8. **Sparse ladders** yield a *bracket* plus the untested gaps. Only
   `--refine` walks every integer in the transition gap.
9. **Nonmonotonicity** is reported and classified, never smoothed away.
10. **Uninformative audited comparisons.** A paired difference of exactly zero
    with exactly zero spread carries nothing — either both branches produced the
    same completed league, or our equity is pinned at the floor in both. Reported
    as `unresolved` with `DEGENERATE` in the basis, never as favorable. The floor
    case was what a silent focus team caused; the guard stays anyway.
11. **The focus shadow ledger.** During the continuation our budget and slots
    are tracked in local variables rather than written into the auction state,
    with the same $1-per-open-slot reserve and the same feasibility test every
    other owner gets. It stops us winning the board for free while still forcing
    rivals to outbid us.

## What to enter during the real draft

Per sale, as it happens: **player, buying owner, final price**. That is what
`ce-lab tactical ... --sale PLAYER_ID:OWNER:PRICE` takes, and it updates both
the room state (budgets, rosters, legality) and the market state (room,
position, tier and buyer levels, plus anchor credibility).

Per decision, before or during a nomination: **the candidate**, **the current
price**, **the bid increment**, and **the current high bidder**. The leader
matters most — it is the one input that names a specific opponent whose roster
and money the counterfactual actually uses.

Precompute between nominations (`ce-lab tactical precompute --top N`), then
read the immediate answer on the clock.

## CLI

```
ce-lab tactical validate      self-checks, no simulation
ce-lab tactical bidders       scenario willingness for every owner
ce-lab tactical recipients    who gets him if we stop
ce-lab tactical endgame       room arithmetic, no CE
ce-lab tactical board         one shared-board continuation
ce-lab tactical max-bid       the named tactical thresholds
ce-lab tactical precompute    bounded precomputation, resume by fingerprint
ce-lab tactical benchmark     runtime for every live stage
```

All require `--demo`: loading a live room from a file is not implemented, and
building one from guesses would be a counterfactual pretending to be a room.

## Cache keys

`tactical_cache_key` digests the auction fingerprint, the pool fingerprint, the
league-settings fingerprint, the comparison cast, the cost book, the market
fingerprint, the candidate, the leader, the current price, the increment, the
recipient set, every scenario, and every setting including the board seed and
completion settings. A test asserts the key moves when any of them changes.

## Out of scope, deliberately

No web UI, no Sleeper automation, no auto-bidding, no fitted human-behaviour
model, no owner personalities, no waivers/FAAB/trades, no equilibrium solve, no
complete real-board precomputation, no QB/TE/stack/handcuff premiums.
