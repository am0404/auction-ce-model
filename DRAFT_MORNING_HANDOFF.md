WHAT TO DO AT 11 AM

Read this page once. It takes about five minutes. Then do the 30-minute mock
draft at the bottom. The auction is at 8 PM.

---

## 1. Start the tool

Two commands, from the repository root (`~/auction-ce-model`):

```bash
cd ~/auction-ce-model
.venv/bin/python -m ceauction.cli draft-day serve
```

That is the whole startup. It prints a URL, opens your browser, and keeps
running. Leave that terminal window open all night — closing it stops the tool.

First start of the day takes about 30 seconds while it reads the real sources.
Every start after that is about 2 seconds, because the board is cached.

## 2. The URL

```
http://127.0.0.1:8765
```

It opens by itself. If it does not, paste that into your browser. Nothing is
deployed and nothing goes over the internet — the page is served by the program
running in your terminal, and it works with the wifi off.

## 3. Enter team names

The twelve owners start as `Team01` … `Team12`. **You are `Team01`**, shown as
"US (our team)".

In the **Owners** table, click the small `edit` button next to any team name,
type the real manager's name, press OK. Do all twelve before the auction
starts — you will not want to be doing it at 8 PM.

Names are saved immediately. Renaming is not an auction event, so **undo does
not undo a rename**.

## 4. Record a sale

Every time any player is sold — to you or to anyone else — do this:

1. Type the player's name in the **Current nomination** search box at the top.
2. Click their row (or pick from the dropdown that appears).
3. In **Record sale**, choose the winning owner and type the winning price.
4. Click **RECORD SALE**.

It takes about a tenth of a second. Everything updates at once: that owner's
budget and roster, every owner's legal maximum, the remaining player pool, the
market bands, and the QB scarcity panel.

**Record every sale, including ones you lose.** The tool's numbers are only as
good as the room it knows about. A missed sale means every budget and every
legal maximum on screen is wrong from then on.

If a sale is illegal the tool refuses it and tells you why in red — duplicate
player, not enough money, full roster, or a roster that could no longer field a
legal lineup. It never silently accepts a bad entry.

## 5. Undo

Click **UNDO LAST SALE** in the header. It reverses exactly one sale — the most
recent one — and restores the auction and the market to precisely the state
they were in before it. You can click it repeatedly to walk back further.

## 6. Recover after a crash

Everything is saved to disk after every accepted change, so a crash costs you
nothing.

```bash
cd ~/auction-ce-model
.venv/bin/python -m ceauction.cli draft-day serve
```

It reloads the draft automatically and prints `restored N sale(s)`. Check that
N matches how many players have actually been sold, then carry on.

The file is `local_data/draftday/draft_state.json`. If you want a spare copy
mid-auction, click **EXPORT** in the header — it writes a timestamped snapshot
and shows you the path. **IMPORT** takes a path and restores one.

If the browser tab breaks but the terminal is still running, just reload the
page. If the terminal is gone, run the command above.

## 7. Load the real board

It loads by itself when you start the tool. To write the opening board out as
files without starting the dashboard:

```bash
.venv/bin/python -m ceauction.cli draft-day board
```

That writes, and overwrites:

- `local_data/draftday/opening_board.csv`
- `local_data/draftday/opening_board.json`

549 rosterable players. Both files hold real player names and are gitignored —
they must never be committed.

## 8. What each price number means

Left to right on the board, and in the nomination panel:

| Number | What it is |
|---|---|
| **mkt** (base) | Expected clearing price for this player — what the room is likely to pay. **Not** what he is worth to us. |
| **band** (low–high) | The same estimate's range. It moves during the auction as real sales come in. |
| **fit** | Fraction of weeks he would actually be in our starting eight. A fourth running back who can never start scores near zero. |
| **+lineup** | How much our expected weekly starting-lineup projection rises if we add him. Points, not dollars. |
| **legal** | **Exact arithmetic.** The most we may legally bid right now, given our budget and the dollar we must keep for every other open slot. This one is not an estimate at all. |
| **cap** | The provisional cap — the one working number to bid against. |
| **basis** | Where the cap came from. Read this before trusting the cap. |

The cap is computed exactly like this, with no other coefficient anywhere:

```
provisional cap = min( exact legal maximum,
                       adjusted market high,
                       proxy permissive ceiling   [only if one has been computed] )
```

and then, if you typed a manual adjustment, that is added and the result is
clamped so it can never exceed the legal maximum.

The nomination panel prints all five rails side by side, so you can always see
which one bound the number.

**BID / CAUTION / STOP** is only ever a comparison against numbers already on
the screen. STOP means the next bid is above the cap or above what we may
legally pay. CAUTION means it is above the market base, or the market and the
proxy disagree.

## 9. Which numbers are NOT CE-audited

**Almost all of them.** Only a cap labelled `CE AUDITED` rests on championship
equity, and **tonight there will not be any**.

The basis labels you will actually see:

- `EXACT FINANCIAL/ROSTER ARITHMETIC` — auction rules. Trustworthy; it is not a
  forecast of anything.
- `MARKET PRIOR` — the Sleeper-derived expected clearing price, before any sale
  has been observed.
- `MARKET + LIVE SALES` — the same, moved by sales recorded tonight.
- `PROXY/HEURISTIC` — the fast tactical proxy, or the fit/+lineup columns. An
  ordering, not a valuation, with no confidence interval.
- `MANUAL OVERRIDE` — your own typed adjustment.

And two refusals you may see, which are the tool declining to answer:

- `CE UNDERPOWERED`
- `SEARCH UNDERCONVERGED`

A cap will **never** be labelled `CE AUDITED` unless the search converged, the
allocation interval resolved, the frontier and decomposition reconciled, the
pass semantics match the decision on screen, and the result was computed against
the exact room you are looking at. The quarterback work fails the first three,
so it produces no max bid. This is deliberate. See `DRAFT_DAY_LIMITATIONS.md`.

## 10. Known blockers and unsupported features

- **No CE max bids.** The full real-player championship-equity board does not
  exist. Every cap tonight is market- or proxy-led.
- **Bidder behaviour is not fitted.** The "most plausible recipients" list is a
  set of stated scenarios, not learned behaviour. Nobody in your league has been
  observed. Treat it as "who could plausibly take him", never as odds.
- **Handcuff mappings are incomplete.** Running backs show
  `HANDCUFF VALUE NOT MODELED` rather than an invented number. If you know a
  handcuff relationship, type it in as a manual override (double-click a board
  row).
- **400 of the 549 players are unanchored** — the Sleeper list never priced
  them. They show no market band at all. That is *not* the same as a $1 player,
  and the tool will not pretend it is. For those, the cap is the legal maximum
  and your own judgement.
- **The proxy ceiling is slow (about 10 seconds)** and is opt-in. Click
  **RUN PROXY CEILING** on a player you have time to think about — between
  nominations, not during one. It runs in the background and does not slow
  anything down.
- **No waiver or trade model. No in-season advice.** This is a draft tool.

## 11. The 30-minute mock draft checklist

Do this at 11 AM, before you trust the tool with a real auction. Tick each one.

First, the automated rehearsal — it drives the whole product and checks itself:

```bash
.venv/bin/python -m ceauction.cli draft-day mock
```

Expect `MOCK REHEARSAL: all 23 checks passed`. It uses a scratch file and does
not touch your real draft.

Then, by hand, in the browser (about 20 minutes):

- [ ] Start the tool; the page opens and shows 549 players.
- [ ] Rename all twelve teams to the real managers.
- [ ] Search a player you know by name; his row appears.
- [ ] Click him; the nomination panel fills in with a verdict and a basis.
- [ ] Sort the board by `cap`, then by `mkt`, then by `fit`. Filter to QB only.
- [ ] Filter to **unanchored only** — confirm those rows show no market band.
- [ ] Record a sale of a cheap player to `Team05`. Confirm his budget drops,
      his roster count rises, and his legal maximum falls.
- [ ] Record a sale **to us**. Confirm our own legal maximum falls.
- [ ] Try to sell the same player twice — confirm it is refused in red.
- [ ] Try a price of $500 — confirm it is refused with a reason.
- [ ] Click **UNDO LAST SALE** twice; confirm budgets return exactly.
- [ ] Sell three or four quarterbacks; watch the QB scarcity panel change.
- [ ] Double-click a running back, enter a $-5 adjustment and a note, save.
      Confirm his basis changes to `MANUAL OVERRIDE`.
- [ ] Click **EXPORT**; note the path it prints.
- [ ] Stop the terminal with Ctrl-C. Start it again. Confirm it prints
      `restored N sale(s)` and the board matches what you left.
- [ ] Click **RESET**, confirm the warning, and check the room is back to
      twelve full budgets. Your manual overrides should survive.
- [ ] Rename the teams again if the reset is the state you want to start from.

**Finish the mock with a RESET so you begin the real auction from an empty
room.**

## 12. Verify the installation

If anything looks wrong at any point, run this first:

```bash
.venv/bin/python -m ceauction.cli draft-day verify
```

It checks Python, numpy, the package, both real source files, the board load,
the row build, the twelve owners and the output directory, and prints `PASS` or
`FAIL` for each. It ends with either the exact start command or the list of what
is broken.

---

### The one thing to remember

Every dollar on the screen carries a label saying where it came from. The
`legal` column is exact. Everything else is an estimate, and tonight none of it
is championship equity. The tool is honest about that on every single line —
believe the label.
