# The bounded two-swap exchange

`docs/REPAIRED_SEARCH.md` §8 recommended a two-for-two exchange pass as the
specific next algorithm, and predicted where it would matter: at the evaluated
price `$80`, where only `$59` remains and improving the roster requires
*simultaneously* freeing money and spending it. A one-swap neighbourhood cannot
express that move, so a one-swap local optimum can sit above a reachable better
roster with no single improving step out of it.

The prediction was correct. **It does not resolve `QB_UNION_UNDERCONVERGED` —
it confirms it.**

## The declared bounds

Reproduced unchanged from §8, fixed before any result was seen:

| Bound | Value |
|---|---|
| incoming shortlist | top **40** affordable by projection |
| completions repaired per price | top **6** |
| improvement epsilon | `1e-9` |
| max exchange rounds | 8 |

Choosing a neighbourhood after seeing which neighbourhood helps is how a search
convinces itself it converged, so the bounds are constants in
`localrepair.py`, asserted by a test, and were not tuned afterwards.

One implementation bound was **not** in §8 and is a correctness matter rather
than a search choice. The neighbourhood is about 71,000 rosters per round — 91
removal pairs by 780 shortlist pairs — and `strength_many` materialises a
`(reps, trials, weeks, roster)` array for the whole batch: 290 million floats,
2.3 GB, several times over. Trials are therefore scored in chunks of 1,024. A
test asserts that chunking changes no result.

## Result at the blocker

State `44209a94d76fce67` — the control run's state, byte-identical fingerprint —
candidate QB, evaluated price `$80`, `$59` remaining.

| | proxy |
|---|---|
| best generated, before any repair | 90.5453 |
| after the one-swap pass (local optimum) | 91.8738 |
| after the bounded two-swap pass | **92.4770** |
| two-swap gain | **+0.6032** |

**All 6 of 6 repaired completions escaped their one-swap local optimum**, each
by a single two-for-two exchange, and all six converged on the same roster. The
mechanism §8 described is real and is the binding one at this price: a one-swap
optimum at `$80` is not a two-swap optimum.

§8 estimated the pass would cost roughly 10× the one-swap repair. Measured at
these bounds it costs about 2.4× — 3.0s against 1.25s per completion — because
chunked scoring keeps the batch in cache and the exchange terminates after one
accepted round.

## Why this confirms underconvergence rather than fixing it

The convergence gate is unchanged and requires, among other conditions, that
**no evaluated price improve by more than 0.25**. The two-swap pass improved
`$80` by **0.6032**, which is 2.4× the tolerance.

Finding an improvement is the gate's *failure* condition, not its success
condition. A search that still finds better rosters when asked harder has not
converged; it has only been under-searched. So this result strengthens the
existing verdict with a sharper reason:

> The `$80` rung was not merely unstable. It was sitting on a one-swap local
> optimum that a two-for-two exchange beats by 0.6 points, which is why the
> objective kept improving and the selected UP completion kept changing.

Two of the three recorded failures are about `$80` and are now explained. The
third is not, and no repair pass can address it:

> *"the final rung's generation paths disagree by 0.5276, beyond 0.25;
> independent paths are still finding materially different constructions"*

That is a property of **generation**, not of repair. Local search improves the
completions it is given; it cannot make two independent generation paths agree
about which region of the space to search. A repair pass could not have flipped
the gate even if `$80` had come back stable.

**`QB_UNION_UNDERCONVERGED` stands. The CE frontier is not evaluated, and no CE
max bid is quoted.** The draft-day tool's audit gate refuses this result for
exactly this reason and shows `SEARCH UNDERCONVERGED` in its place.

## What would come next

The blocker is now generation-path disagreement rather than local optimality.
The two-swap pass should be folded into the ladder so every rung's union is
two-swap-stable before the gate reads it, and the `$80` rung re-run to see
whether a two-swap-stable union brings the paths into agreement — a stronger
union may converge the paths, or may simply reveal that the disagreement is
about genuinely different constructions of equal value, which is a different
finding and would need a different remedy.

Not recommended, still: MILP or branch-and-bound. Both need a new dependency,
and the gap they would close is no longer the one that is open.

## Artifacts

- `docs/two_swap_exchange_sanitized.json` — the table above. No player names,
  ids or projections.
- `local_data/tactical/two_swap_80.json` — identical content; gitignored by
  location.
- `src/ceauction/tactical/localrepair.py` — `two_swap_shortlist`,
  `repair_completion_two_swap`, `repair_union_two_swap`.
- `tests/test_local_repair.py` — 12 added tests covering legality,
  determinism, ordering independence, chunk invariance, pair recording, the
  declared bounds and union additivity.
