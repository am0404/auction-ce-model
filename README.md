# auction-ce-model

Championship-equity (CE) simulation for a 12-team, $200, half-PPR **Sleeper
redraft auction** league with a superflex and a weekly league-median result.

The engine answers exactly one question:

> Given a drafted 15-man roster, what is the probability this team wins the league?

Auction pricing is deliberately **not** built. See `SPEC.md` §11.

> **All player data in this repository is synthetic and clearly labelled.**
> `src/ceauction/synthetic.py` invents every number it produces. No real player
> distribution is used, estimated or implied anywhere.

## Install

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip      # required: pip < 21.3 cannot do editable installs
.venv/bin/pip install -e ".[dev]"
```

Requires Python 3.9+ and NumPy. No other runtime dependency. The pip upgrade is
not optional on a stock macOS Python, whose bundled pip (21.2.4) predates PEP 660
and fails with "editable mode currently requires a setuptools-based build".

## Run

```bash
.venv/bin/ce-lab league                    # CE for all 12 teams
.venv/bin/ce-lab lineup --weeks 1 8 14     # why each starter was chosen
.venv/bin/ce-lab experiments               # list the controlled experiments
.venv/bin/ce-lab run --all --sims 12000    # the full CE laboratory
.venv/bin/ce-lab curve --sims 16000        # the marginal CE curve for one slot
.venv/bin/ce-lab bench                     # runtime + Monte Carlo uncertainty
.venv/bin/python -m pytest                 # 484 tests
```

Two of the twelve experiments are **controls** and are meant to read near zero.
`ce-lab run --all` prints a short guide explaining which findings are claims
about this code and which would be claims about football — the laboratory only
ever makes the first kind.

## Documents

| File | What it is |
|---|---|
| `SPEC.md` | The technical specification the code implements |
| `HANDOFF.md` | What was built, how to run it, results, and the next step |
| `OPEN_QUESTIONS.md` | Decisions that need real data or your judgement |
| `docs/example_ce_lab_output.txt` | A full CE-laboratory run |
| `docs/PLAYER_DATA_INVENTORY.md` | What the existing player model actually contains |
| `docs/PLAYER_DATA_LINEAGE.md` | Source file to transformation code to derived field |
| `docs/PLAYER_MAPPING_GAPS.md` | Every `PlayerSpec` field against that source |
| `docs/INGESTION_AUDIT.md` | Sanitized result of running the ingestion layer |
| `schemas/real_player_input_v1.schema.json` | The versioned real-player input contract |
| `docs/example_curve_output.txt` | A 19-level marginal CE curve + resolution report |
| `docs/example_marginal_curve.csv` | The same curve, machine-readable |

## The marginal CE curve

```bash
ce-lab curve --sims 16000 --min-level 4 --max-level 22 --step 1 --csv curve.csv
```

Sweeps one roster slot's projected points from replacement level to elite,
holding everything else fixed, and reports `CE(level)` with honest uncertainty
at every step. This is the object any auction pricing scheme would be a
transformation of — but pricing itself is **not** built: no dollar values, no
opening or live max bids, no inflation model, no roster-completion solver.

Every level is the same player at a different projected level, so he keeps his
`crn_key` and therefore his injuries, byes, weekly conditions, signals and
idiosyncratic draws across the whole sweep; every other player and the schedule
are untouched. If he carries real published weekly projections
(`weekly_projection_override`), those are shifted by the same delta as
`base_mean`, which preserves their shape while moving their level — otherwise
his realized scoring would move while the manager's pregame view stayed frozen.
Differences between levels are matched per-season differences, including the
**adjacent slopes**, which are computed as their own paired comparison rather
than as a difference of two baseline deltas.

`--isotonic` adds a display column that *imposes* monotonicity rather than
revealing it. Changing a player's level changes which players get started, so a
local decline in the raw curve is not automatically noise. Raw estimates stay
primary and unchanged.

The command also prints a **Monte Carlo resolution report**: the observed
paired standard error, how many simulations a delta-CE of 0.005 / 0.002 / 0.001
would need to clear |z| = 2, and — using throughput measured during the sweep —
how long that would take.

It is a **pilot estimate scoped to the comparison that produced it**, not a
capability claim. Paired variance tracks how often the focus team's outcome
actually flips, which differs between comparisons, and helping and hurting
seasons cancel in the mean while both adding to the variance — so the
extrapolation is neither a bound nor reliably conservative. On the 19-level
example in `docs/example_curve_output.txt`, under that synthetic curve, that
hardware and a measured 3.5% discordance rate, 0.005 came in at 8.6s per paired
comparison and 0.002 and 0.001 at 54s and 215s. A decision that must actually be
resolved needs its own pilot run or an adaptive stopping rule; neither is built.

The marginal value of a projected point also varies about fivefold along the
curve — smallest at replacement level, which is exactly where the cheapest
auction decisions are made. That is an input to pricing, not a price: what a
player is worth also depends on his whole outcome distribution rather than its
mean, on availability, position and slot eligibility, on correlation with what
you already own, on which alternatives remain, and on which rival gets him
instead. See `HANDOFF.md` §12.

## The two rules that shape everything

```
drafted roster -> latent player and season states -> observable signals and
knowable weekly conditions -> lineup decision -> realized scores -> standings
-> playoffs -> champion
```

**A lineup may use only what was knowable before kickoff.** A benched player who
scores 30 is worth exactly zero that week.

**A projection may be moved only by information that was forecastable in the
first place.** Beliefs update from a separate observable channel standing in for
usage, snaps, routes, targets and depth-chart reporting — never from realized
fantasy points. A lucky touchdown changes that week's score and nothing else. A
genuine change in a player's role or usage changes every week that follows.

The second rule is the harder one, and the code enforces it structurally: the
projection builder has no parameter through which a realized score could arrive.

## Scoring

Half PPR, and the seam in `scoring.py` carries the complete rule set including
two-point conversions and individual special-teams touchdowns. Interceptions and
lost fumbles are both −2, so a weekly score can be negative — there is no rule
flooring an individual player's week at zero, and the model does not impose one.
