# DRAFT DAY LIMITATIONS

**Read this before trusting any number the draft-day tool shows you.**

The tool is honest on every line, but the honesty is only useful if you know
what the labels mean. These are the nine things it cannot do.

---

## 1. There is no full real-player CE max-bid board

**No championship-equity maximum bid exists for any real player.**

The project's whole purpose is to produce max bids from championship equity
rather than from projected points or market price. That board was not finished.
Nothing in the draft-day tool substitutes for it, and nothing in it pretends to.

You will not see a cap labelled `CE AUDITED` tonight.

## 2. Most caps are market or proxy provisional

The provisional cap is a minimum of rails that already existed:

```
min( exact legal maximum, adjusted market high, proxy permissive ceiling if computed )
```

- The **exact legal maximum** is arithmetic on the auction rules. It is not an
  estimate, and it is the only number on screen that is not.
- The **market band** is an expected *clearing price* — what the room will
  likely pay. It is explicitly **not** our valuation, not a reservation price
  and not a max bid.
- The **proxy ceiling** is a fast ordering of branches under a deliberately
  reduced search. It has no confidence interval and the audited path could
  contradict it.

A cap built from these is labelled `MARKET-LED PROVISIONAL CAP`. It answers
"what will he go for, and what may I legally pay", not "what is he worth to my
championship odds".

**400 of the 549 players have no market anchor at all.** They show no band. An
unanchored player is *unpriced*, which is not the same as a player the market
values at $1, and the tool refuses to conflate the two.

## 3. The QB completion search remains underconverged

The quarterback union work on `qb-union-convergence` did not converge. The
allocation interval did not resolve, and the frontier was never evaluated:
`QB_UNION_UNDERCONVERGED` still stands.

The draft-day tool therefore refuses to turn it into a cap. The gate requires
**all five** of: a converged search, a resolved allocation interval, a
reconciled frontier and decomposition, pass semantics matching the decision
displayed, and a result computed against the exact room on screen. The
quarterback work fails the first three.

You will see `SEARCH UNDERCONVERGED` or `CE UNDERPOWERED` where a CE number
would otherwise have gone. That is the tool declining to answer, which is the
correct behaviour and not a bug.

## 4. Bidder behaviour has not been fitted to this league

The "most plausible recipients" panel and the willingness numbers beside it come
from a **stated scenario model**. Nobody in your league has been observed. No
behaviour has been estimated, fitted, or validated against anything.

The weights are assumptions about how branches divide, not probabilities. The
panel says so every time it is drawn. Read it as "who could plausibly take him",
never as "who probably will".

Relatedly: a final sale price tells you about the *clearing* price. It does not
reveal the winner's maximum — he stopped because everyone else did — and it
does not reveal any losing bidder's maximum beyond it being lower.

## 5. Handcuff mappings are incomplete

There is no real depth chart in this model. Backfield contingency — who inherits
work when a starter is out — is uncalibrated, and the engine's `contingency`
field is an unmodelled placeholder for real players.

Running backs therefore display `HANDCUFF VALUE NOT MODELED`. That is a refusal,
not a valuation of zero.

If you know a handcuff relationship, enter it yourself: double-click a board row
and record the linked starter, a contingency tag, an estimated takeover share
and your reasoning. It is preserved and displayed as a **manual adjustment**
labelled `MANUAL OVERRIDE`. It is *not* fed into the conditional-backfield
engine, because that engine needs a calibrated mapping this tool cannot supply
from a typed guess.

## 6. Nothing here is connected to Sleeper

**The dashboard is a manual tally.** It holds no Sleeper credentials, opens no
outbound connection, and polls nothing. Every nomination, bid and sale is typed
in by hand, and the board reflects exactly what has been entered and nothing
else — it will not notice a sale you did not record.

The header says so on every screen:
`MANUAL TRACKING -- NOT CONNECTED TO SLEEPER`.

Live synchronisation is not implemented and was deliberately not attempted in
the draft-day hotfix. A half-finished integration that silently lagged the real
draft would be worse than no integration, because the arithmetic in section
"What the tool *is* reliable for" is only exact about the room it has been told
about.

## 7. The Sleeper price and the model range are different numbers

Two prices sit side by side on the board and they must never be read as one.

- **Sleeper** is the raw displayed price from the supplied list, carried through
  byte for byte and never adjusted by this tool. It is what the *room* is
  looking at. Your league's premise is that managers anchor on it: reluctant to
  go far above, treating prices well below it as bargains, and disregarding it
  where the format is obviously wrong — most of all at tight end.
- **Model Range** is this repository's format-adjusted expected clearing range
  for *this* league's rules.

They diverge hard, and that divergence is information rather than an error. On
the real board a leading tight end sits at **$34 on Sleeper and $11-$15 here**,
because a tight-end number computed for another lineup shape is about a
different game. A leading superflex quarterback sits at **$52 on Sleeper and
$23-$29 here**. (Names are omitted: this repository is public and no real
player-level data is committed to it. The full trace is written locally to
`local_data/draftday/price_trace.json`, which is gitignored.)

Where no converged tactical result exists, the **Draft Guardrail** column is:

```
guardrail = max( raw Sleeper displayed price, adjusted model high )
```

clamped only by our exact legal maximum, with any manual adjustment applied
explicitly on top. It is labelled `USER MARKET-ANCHOR POLICY`.

**That policy is a bidding-behaviour guardrail and nothing else.** It is not
championship equity, it is not a maximum worth paying, and it must never be
described as either. A player nobody priced gets no guardrail number at all —
not even our legal maximum, which would read as permission to spend it.

## 8. A market prior cannot say BID or STOP

This was a real defect, and it is worth stating plainly because the fix changes
what you will read on the screen.

The board used to print `STOP` whenever the next bid exceeded the adjusted
market high. A $30 bid against an adjusted high of $29 read as a hard refusal —
a claim that $30 costs us championship equity. Nothing had computed that. The market band is a forecast of **what the room will pay**; it contains
no statement about what winning the player is worth to us, so it can neither
authorise a bid nor forbid one.

Where the only basis is `MARKET PRIOR` or `MARKET + LIVE SALES`, the tool now
prints a *position* instead of an instruction:

| what you see | what it means |
| --- | --- |
| `BELOW MARKET` | the next bid is under the adjusted range |
| `IN MARKET RANGE` | the next bid sits inside it |
| `ABOVE MARKET` | the next bid is over it |

The dollar figure beside it is labelled `MARKET GUARDRAIL`, never `MAX BID`, and
`CE NOT AUDITED` is stated every time.

`BID` / `CAUTION` / `STOP` are reserved for a converged tactical proxy or an
audited CE bracket. `SEARCH UNDERCONVERGED` and `CE UNDERPOWERED` are failed
computations, not evidence, and can never produce one. `OVER LEGAL MAXIMUM` and
`CANNOT BID` are auction arithmetic rather than judgement, and say so.

## 9. The simulated mid-auction room is not observed history

Any mid-auction room used in this project's earlier analysis — and every price
in `draft-day mock` — was **fabricated**. It is a constructed scenario, not a
record of anything that happened in a real auction.

Nothing in this repository has been validated against a completed real auction,
because none has been observed.

---

## What the tool *is* reliable for

To be fair to it, three things on the screen are exact rather than estimated,
and they are the things hardest to do in your head under a ten-second clock:

1. **Every owner's legal maximum**, including the dollar that must be reserved
   for each of their remaining open slots.
2. **Candidate-specific legality** — whether a given owner may buy *this*
   player, which differs from whether they have the money. An owner whose last
   slot must hold a receiver cannot legally bid on a quarterback at any price,
   and the tool knows that.
3. **The room's money and rosters**, kept exactly in step with what you record.

Those are worth having. They are also the reason recording *every* sale matters:
the arithmetic is only exact about the room it has been told about.
