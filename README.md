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
.venv/bin/pip install -e ".[dev]"
```

Requires Python 3.9+ and NumPy. No other runtime dependency.

## Run

```bash
.venv/bin/ce-lab league                    # CE for all 12 teams
.venv/bin/ce-lab lineup --weeks 1 8 14     # why each starter was chosen
.venv/bin/ce-lab experiments               # list the controlled experiments
.venv/bin/ce-lab run --all --sims 12000    # the full CE laboratory
.venv/bin/ce-lab bench                     # runtime + Monte Carlo uncertainty
.venv/bin/python -m pytest                 # the test suite
```

## Documents

| File | What it is |
|---|---|
| `SPEC.md` | The technical specification the code implements |
| `HANDOFF.md` | What was built, how to run it, results, and the next step |
| `OPEN_QUESTIONS.md` | Decisions that need real data or your judgement |
| `docs/example_ce_lab_output.txt` | A full CE-laboratory run |

## The one rule that shapes everything

```
drafted roster -> latent player and season states -> information observable
before kickoff -> lineup decision -> realized scores -> information update ->
standings -> playoffs -> champion
```

A lineup may use only what was knowable before kickoff. A benched player who
scores 30 is worth exactly zero that week. He creates future value only if his
performance reveals something persistent and observable.
