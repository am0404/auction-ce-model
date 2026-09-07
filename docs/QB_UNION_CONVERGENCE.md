# QB union convergence: `QB_UNION_UNDERCONVERGED`

> **SIMULATED MID-AUCTION STATE.** Every auction history here is invented from
> the market prior for frontier testing. It is NOT observed, NOT recorded, NOT
> historical, and NOT calibrated. **No league sale has ever been recorded.**
> **No maximum bid is quoted in this document.**

## Verdict

The QB's full-budget completion union **does not converge** under the
predeclared effort ladder. The frontier was therefore **not evaluated** — the
full CE budget is gated on convergence, and the prior `PASS_AT_NEXT_BID` reading
remains withheld rather than reinstated.

This is outcome (3) of the three the brief allowed, with strong evidence for
outcome (2) as the underlying cause: the roster search needs an algorithmic
change, not more effort.

## 1. Why the search did not converge

Before changing anything, a knob sweep on the real QB at support price `$1`
(state `44209a94d76fce67`, 12 open slots, `$139` remaining):

| knob varied | effect on the objective |
|---|---|
| `finalists` 3 → 12 → 40 | **none** — 95.2731 at all three |
| `candidate_pool` 40 → 90 | **none** |
| `proxy_candidates` 32 → 2000 | **none** |
| `proxy_seed` (4 values) | **none** |
| `max_candidates` 160 → 4000 | +0.055 |
| `spend_buckets` 4/8/16/32 | 96.47 / 97.56 / **98.14** / 97.85 |
| `beam_width` | see below |

The beam curve, at `finalists=40`:

| beam | 32 | 64 | 96 | 128 | 192 | 256 | 384 | 512 | 768 |
|---|---|---|---|---|---|---|---|---|---|
| best proxy | 95.27 | **94.17** | **97.99** | 97.45 | 97.56 | 97.56 | **95.15** | 95.89 | 97.32 |

**The objective is non-monotone in effort.** Beam 64 is worse than beam 32; beam
384 is worse than beam 96 by 2.8 points. The spread across the curve is ~3.8
weekly points, **fifteen times** the 0.25 tolerance.

**Diagnosed cause of the prior `+0.690` movement:** insufficient *and
non-monotone* beam width, compounded by the `spend_buckets` diversification it
interacts with. Specifically **not**:

- **not** player-pool depth (`candidate_pool` does nothing here),
- **not** finalist truncation of the objective — `finalists` capped how many
  completions each search *emitted* (which is why every rung of the previous run
  reported exactly 3, and why raw union size looked like progress) but never
  changed the best one found,
- **not** noisy CE selection — these searches run `evaluate_ce=False` and the
  selection is a deterministic `max` over proxy strength,
- **not** the search seed — the search is deterministic given `(beam_width,
  spend_buckets)`.

Exact enumeration was tested and **correctly refuses**: 1.5×10¹¹ combinations at
the shallowest pool depth against a 400,000 limit. Provable convergence by
exhaustion is not available at 12 open slots.

The consequence: *"rerun with triple the effort"* cannot converge this search,
because the quantity being increased does not monotonically improve the answer.
So the design here stops treating any one configuration's output as the answer.
Every path's completions accumulate into **one monotone union**, the whole union
is rescored by one evaluator, and the reported objective is the **best over
everything seen so far** — non-decreasing by construction.

## 2. The predeclared effort ladder

Fixed before any rung was run. Fingerprint `0fddb000228e1276`, pinned by a test
so a later edit cannot retrofit it.

| rung | multiple | independent generation paths `(beam, buckets)` | finalists | max_candidates |
|---|---:|---|---:|---:|
| `1x` | 1 | (32,8) (32,16) | 8 | 400 |
| `2x` | 2 | (64,8) (64,16) (96,8) | 16 | 800 |
| `4x` | 4 | (128,8) (128,16) (192,8) (192,16) | 24 | 1600 |
| `8x` | 8 | (256,8) (256,16) (384,16) (512,16) (768,32) | 32 | 3200 |

Paths vary `(beam_width, spend_buckets)` because those are the only settings the
sweep showed actually change what the search finds. `proxy_seed` is deliberately
**not** used as a diversity source: it provably does nothing here, and using it
would fake independence that does not exist.

Each path searches at all nine support prices (`$1 … $128`); everything found is
deduplicated by the complete economic/roster fingerprint and kept forever.

## 3. Per-rung results

| rung | union size | new unique | best proxy | path spread | time |
|---|---:|---:|---:|---:|---:|
| `1x` | 137 | 137 | 97.9738 | 0.6711 | 1.8s |
| `2x` | 453 | 316 | 98.1305 | 0.3658 | 6.0s |
| `4x` | 1014 | 561 | 98.1399 | 0.1811 | 15.9s |
| `8x` | 1749 | 735 | **98.1399** | **0.5276** | 47.9s |

Per-path best at the final rung — the independent-seed comparison:

| path | (256,8) | (256,16) | (384,16) | (512,16) | (768,32) |
|---|---|---|---|---|---|
| best proxy | 97.6124 | **98.1399** | 98.0164 | 97.9884 | 98.0439 |
| new unique | +150 | +140 | +172 | +118 | +155 |

Every path was still contributing 118–172 previously unseen constructions at the
maximum rung, and they disagree by 0.5276 about how good the best one is.

## 4. The convergence decision

| clause | result |
|---|---|
| `UF` selected completion stable | **pass** |
| `UF` objective within 0.25 | **pass** (98.1399 → 98.1399, exactly 0) |
| no evaluated price improved beyond 0.25 | **FAIL** |
| `UP` selection stable at evaluated prices | **FAIL** |
| affordability unchanged at tested prices | pass |
| generation paths agree within 0.25 | **FAIL** |

**`QB_UNION_UNDERCONVERGED`.**

The failure is precise and worth stating plainly: **the sell side converged and
the buy side did not.** `UF` — us holding the QB having paid nothing, drawing on
the full `$139` — settled completely between the `4x` and `8x` rungs. But `UP`,
whose budget is cut by the purchase price, was still improving:

| evaluated bid | `4x` best | `8x` best | movement |
|---:|---:|---:|---:|
| $36 | 97.400 | 97.682 | **+0.282** |
| $40 | 96.872 | 97.529 | **+0.657** |
| $65 | 93.804 | 94.687 | **+0.883** |
| $80 | 91.474 | 91.691 | +0.217 |
| $100 | 86.893 | 87.194 | **+0.301** |
| $128 | 74.307 | 74.347 | +0.040 |

The selected `UP` completion changed at **every** evaluated price. That is
exactly the quantity a reservation price depends on, so no classification may be
read off it.

The reason is structural: the union is generated broadly but then filtered by
affordability, and the tighter the budget the thinner the surviving set — 1,244
affordable constructions at `$36` but only **101** at `$128`. The search spends
its effort where money is plentiful and the answer is easy, and arrives thinnest
exactly where the frontier question is hardest.

## 5. Frontier

**Not evaluated.** `may_evaluate_ce` is false, so the K=11 × 4,000-season run was
skipped by design rather than run and then discarded. No table appears here, no
bracket, and specifically **not** the prior `PASS_AT_NEXT_BID` — a conclusion
resting on an opportunity set that more effort still improves is not a
conclusion.

Final classification: **`QB_UNION_UNDERCONVERGED`**.

## 6. Performance, and where the bottleneck actually is

| stage | time |
|---|---:|
| board load | 20.4s |
| candidate selection (cache hit) | 0.1s |
| candidate selection (cold, `converged_diagnose`) | ~25 min |
| union rung `1x` / `2x` / `4x` / `8x` | 1.8 / 6.0 / 15.9 / 47.9s |
| **total union generation (1,749 completions)** | **71.5s** |
| CE: one allocation draw, five branches | 12.2s |
| CE: K=11 × 6 prices (projected, not run) | ~805s |
| total this run (convergence only) | 93s |

**The bottleneck is not the search.** Generating 1,749 constructions — 67× the
previous run's 26 — costs 71.5s, about 8% of what one candidate's frontier would
cost in CE. Doubling or quadrupling search effort again is affordable; it simply
does not converge.

Projected for 25 targeted players, with warm candidate selection:
`25 × (71.5s + 805s)` ≈ **6.1 hours**, of which ~30 minutes is search and the
rest is CE. Cold candidate selection would add ~25 min per player and dominate
everything — it must stay cached.

### Recommended search improvement (not implemented here)

Do not raise the beam again. The specific change, in priority order:

1. **Generate at the evaluated prices, not only the support prices.** The union
   is thinnest precisely where it is filtered hardest. Budget the generation
   effort by how binding the price is, so `$128` gets more search than `$36`
   rather than 12× less surviving material.
2. **Add a local-improvement pass over the accumulated union** — swap one
   rostered player for an affordable alternative, accept on improvement, iterate
   to a local optimum. This is monotone by construction, needs no beam, and the
   71.5s budget leaves ample room. It directly attacks the non-monotonicity: a
   bad beam draw gets repaired instead of discarded.
3. Only then consider widening the ladder.

## 7. Reconciliation and invariants

No frontier CE was spent this run, so there are no new per-draw residuals to
report. The shared-context machinery is unchanged and its guarantees are
re-verified by the existing suites, which pass unchanged: `EvalContext`,
`IndependentSearchRefused`, the 1e-12 telescoping agreement, price-independent
possession, and the joint-world conservation/budget/ownership/league-CE
invariants. **No tolerance was weakened and no existing test was relaxed** —
doing so would have been the easiest way to manufacture the convergence this
document reports as absent.

A spot check during performance measurement (one draw at `p=$36` over the
1,749-completion union) returned a telescoping residual of **0.0e+00** and
`UF` offered all 1,749 constructions against `UP`'s 1,244, confirming the union
feeds every branch and the affordability filter is the only thing separating
them.

## 8. Artifacts

- `docs/QB_UNION_CONVERGENCE_sanitized.json` — ladder, rungs, checks, timings.
  No player names, ids, or projections.
- `local_data/tactical/qb_convergence.json` — player-level. **Gitignored.**
- `ce-lab tactical qb-convergence --reuse-candidates <local_data json>`
- `src/ceauction/tactical/qbconvergence.py`, `qbconvergence_experiment.py`,
  `tests/test_qb_convergence.py` (34 tests)
