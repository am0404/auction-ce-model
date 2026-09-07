"""Deterministic local repair of completions, and its refusals.

The completion objective is an average over availability scenarios of an exact
matroid greedy, so it is **submodular** in the roster set -- not additive by
player. A beam ranks partial rosters by a prefix value that, under a submodular
objective, is not a valid bound on the finished roster's value; that is why
widening the beam moved the answer from 94.17 to 97.99 and back. Local search is
the principled repair because improvement is monotone by construction.

These tests pin that monotonicity, the legality every swap must preserve, and
the determinism that keeps the result independent of iteration order.
"""

from __future__ import annotations

import pytest

from ceauction.auction.completion import (ComparisonCast, Completion,
                                          CompletionSettings, complete_roster)
from ceauction.auction.feasibility import can_fill_lineup
from ceauction.auction.proxy import ProxyEvaluator
from ceauction.auction.state import new_auction
from ceauction.league import DEFAULT_LEAGUE, Position
from ceauction.market.costbook import cost_book_from_prior
from ceauction.tactical.demo import build_tactical_demo
from ceauction.tactical.localrepair import (IMPROVEMENT_EPSILON, RepairReport,
                                            repair_completion, repair_union)
from ceauction.tactical.midauction import STATE_SPECS, build_simulated_state
from ceauction.tactical.qbconvergence import ladder_fingerprint
from ceauction.tactical.realpilot import OWNER_IDS, RealBoard
from ceauction.tactical.repairedsearch import (REPAIR_SETTINGS_DECLARED_AT,
                                               REPAIR_TOP_K,
                                               REPAIR_TOP_K_PER_PRICE,
                                               SWAP_POOL_DEPTH,
                                               price_effort_multiplier,
                                               repair_settings_fingerprint)
from ceauction.tactical.union import completion_fingerprint

_CS = CompletionSettings(beam_width=16, candidate_pool=30, proxy_candidates=120,
                         finalists=4, max_candidates=200, proxy_reps=16,
                         selection_sims=300, evaluation_sims=400,
                         selection_seed=555_000_111,
                         evaluation_seed=917_324_011, rival_selection="proxy")


@pytest.fixture(scope="module")
def board():
    d = build_tactical_demo()
    state = new_auction(list(d.state.pool), OWNER_IDS, OWNER_IDS[0],
                        settings=DEFAULT_LEAGUE)
    state.validate()
    cb = cost_book_from_prior(d.prior, "base")
    gaps = {s.player_id: 1 for s in state.pool if s.player_id not in cb}
    costs = cb.with_costs(gaps) if gaps else cb
    return RealBoard(
        state=state,
        cast=ComparisonCast(0, tuple(() for _ in OWNER_IDS), OWNER_IDS),
        prior=d.prior, market=d.market, costs={"base": costs},
        key_by_id=dict(d.key_by_id),
        name_by_id={s.player_id: s.name for s in state.pool}, coverage={})


@pytest.fixture(scope="module")
def setup(board):
    by = {}
    for s in board.state.available_specs:
        by.setdefault(Position(int(s.position)).name, []).append(s)
    for g in by.values():
        g.sort(key=lambda s: -s.base_mean)
    cid = by["QB"][0].player_id
    sim = build_simulated_state(board, STATE_SPECS["balanced"], protect=[cid])
    st = sim.state
    px = ProxyEvaluator(st.pool, st.settings, 16, 7)
    focus = st.focus_owner_id
    bought = st.apply_purchase(cid, focus, 10)
    res = complete_roster(bought, board.cast, board.costs["base"],
                          settings=_CS, owner_id=focus, evaluate_ce=False,
                          default_cost=1, proxy=px, reserved_ids=frozenset())
    fin = list(res.finalists) or ([res.best] if res.best else [])
    owned = frozenset(st.owner(focus).player_ids)
    kw = dict(state=bought, costs=board.costs["base"], proxy=px,
              candidate_id=cid, owned=owned | {cid},
              budget=st.owner(focus).budget_remaining - 10,
              pool=[s.player_id for s in bought.available_specs][:120],
              default_cost=1)
    return fin, kw, cid, px, bought, owned


# ---------------------------------------------------------------------------
# Monotone improvement
# ---------------------------------------------------------------------------


def test_local_improvement_never_lowers_the_objective(setup):
    fin, kw, *_ = setup
    for c in fin:
        r = repair_completion(c, **kw)
        assert r.repaired.proxy >= c.proxy


def test_an_accepted_swap_strictly_improves(setup):
    fin, kw, *_ = setup
    for c in fin:
        r = repair_completion(c, **kw)
        for s in r.swaps:
            assert s.gain > IMPROVEMENT_EPSILON


def test_gains_accumulate_to_the_reported_total(setup):
    fin, kw, *_ = setup
    r = repair_completion(fin[0], **kw)
    assert r.gain == pytest.approx(sum(s.gain for s in r.swaps), abs=1e-9)


def test_the_result_is_a_one_swap_local_optimum(setup):
    """Repairing an already-repaired roster must find nothing more."""
    fin, kw, *_ = setup
    once = repair_completion(fin[0], **kw)
    twice = repair_completion(once.repaired, **kw)
    assert not twice.improved
    assert twice.repaired.proxy == once.repaired.proxy


def test_a_planted_beam_failure_is_repaired(setup):
    """The weakest beam finalist must be lifted toward the strongest."""
    fin, kw, *_ = setup
    weakest = min(fin, key=lambda c: c.proxy)
    strongest = max(fin, key=lambda c: c.proxy)
    assert weakest.proxy < strongest.proxy
    r = repair_completion(weakest, **kw)
    assert r.improved
    assert r.repaired.proxy >= strongest.proxy, (
        "local repair should reach at least the best the beam found")


# ---------------------------------------------------------------------------
# Legality is preserved by every swap
# ---------------------------------------------------------------------------


def test_roster_size_is_preserved(setup):
    fin, kw, *_ = setup
    for c in fin:
        r = repair_completion(c, **kw)
        assert len(r.repaired.roster) == len(c.roster)
        assert len(set(r.repaired.roster)) == len(c.roster)


def test_candidate_ownership_cannot_be_swapped_away(setup):
    fin, kw, cid, *_ = setup
    for c in fin:
        r = repair_completion(c, **kw)
        assert cid in r.repaired.roster


def test_already_owned_players_are_never_sold(setup):
    fin, kw, cid, px, bought, owned = setup
    for c in fin:
        r = repair_completion(c, **kw)
        assert owned <= set(r.repaired.roster)


def test_the_budget_is_never_exceeded(setup):
    fin, kw, *_ = setup
    for c in fin:
        r = repair_completion(c, **kw)
        assert r.repaired.added_cost <= kw["budget"]


def test_unaffordable_swaps_are_refused(setup):
    """A zero budget admits no swap at all."""
    fin, kw, *_ = setup
    tight = dict(kw, budget=0)
    r = repair_completion(fin[0], **tight)
    assert not r.improved


def test_lineup_feasibility_is_preserved(setup):
    fin, kw, cid, px, bought, owned = setup
    from ceauction.tactical.localrepair import _counts
    for c in fin:
        r = repair_completion(c, **kw)
        assert can_fill_lineup(_counts(r.repaired.roster, bought))


def test_no_duplicate_ownership_is_introduced(setup):
    fin, kw, *_ = setup
    for c in fin:
        r = repair_completion(c, **kw)
        assert len(set(r.repaired.roster)) == len(r.repaired.roster)


def test_the_reported_cost_matches_the_repaired_roster(setup):
    fin, kw, cid, px, bought, owned = setup
    costs = kw["costs"]
    for c in fin:
        r = repair_completion(c, **kw)
        spend = sum(costs.cost_of(p, 1) for p in r.repaired.roster
                    if p not in kw["owned"])
        assert spend == r.repaired.added_cost


def test_the_reported_proxy_matches_a_fresh_evaluation(setup):
    fin, kw, cid, px, *_ = setup
    r = repair_completion(fin[0], **kw)
    assert r.repaired.proxy == pytest.approx(
        px.strength(r.repaired.roster), abs=1e-9)


# ---------------------------------------------------------------------------
# Determinism and stable tie-breaking
# ---------------------------------------------------------------------------


def test_repeated_execution_is_deterministic(setup):
    fin, kw, *_ = setup
    a = repair_completion(fin[0], **kw)
    b = repair_completion(fin[0], **kw)
    assert completion_fingerprint(a.repaired) == completion_fingerprint(b.repaired)
    assert a.repaired.proxy == b.repaired.proxy


def test_the_result_is_independent_of_roster_input_ordering(setup):
    """Tie-breaking must not depend on incidental tuple position."""
    import dataclasses
    fin, kw, *_ = setup
    c = fin[0]
    shuffled = dataclasses.replace(c, roster=tuple(reversed(c.roster)))
    a = repair_completion(c, **kw)
    b = repair_completion(shuffled, **kw)
    assert a.repaired.proxy == b.repaired.proxy
    assert completion_fingerprint(a.repaired) == completion_fingerprint(b.repaired)


def test_the_result_is_independent_of_pool_ordering(setup):
    fin, kw, *_ = setup
    a = repair_completion(fin[0], **kw)
    b = repair_completion(fin[0], **dict(kw, pool=list(reversed(kw["pool"]))))
    assert a.repaired.proxy == b.repaired.proxy
    assert completion_fingerprint(a.repaired) == completion_fingerprint(b.repaired)


# ---------------------------------------------------------------------------
# The union stays monotone
# ---------------------------------------------------------------------------


def test_repair_adds_and_never_substitutes(setup):
    fin, kw, *_ = setup
    union = {completion_fingerprint(c): c for c in fin}
    before = set(union)
    after, rep = repair_union(dict(union), limit=None, **kw)
    assert before <= set(after), "an original completion was removed"
    assert len(after) >= len(before)


def test_the_repair_report_counts_agree(setup):
    fin, kw, *_ = setup
    union = {completion_fingerprint(c): c for c in fin}
    _, rep = repair_union(dict(union), limit=None, **kw)
    assert rep.n_improved + rep.n_already_optimal == rep.n_input
    assert rep.union_size_after >= rep.union_size_before


def test_repaired_completions_deduplicate(setup):
    """Repairing twice must not grow the union the second time."""
    fin, kw, *_ = setup
    union = {completion_fingerprint(c): c for c in fin}
    once, _ = repair_union(dict(union), limit=None, **kw)
    twice, _ = repair_union(dict(once), limit=None, **kw)
    assert len(twice) == len(once)


def test_the_repair_limit_still_keeps_every_original(setup):
    fin, kw, *_ = setup
    union = {completion_fingerprint(c): c for c in fin}
    after, _ = repair_union(dict(union), limit=1, **kw)
    assert set(union) <= set(after)


# ---------------------------------------------------------------------------
# The control ladder is untouched; repair settings are separately declared
# ---------------------------------------------------------------------------


def test_the_control_ladder_fingerprint_is_unchanged():
    """The before/after comparison is worthless if the control moved."""
    assert ladder_fingerprint() == "0fddb000228e1276"


def test_repair_settings_are_separately_fingerprinted():
    assert repair_settings_fingerprint() != ladder_fingerprint()
    assert "before any rung" in REPAIR_SETTINGS_DECLARED_AT


def test_the_effort_rule_is_declared_and_monotone():
    vals = [price_effort_multiplier(i, 6) for i in range(6)]
    assert vals == sorted(vals)
    assert vals[0] == 1.0 and vals[-1] == 3.0


def test_the_declared_bounds_are_explicit():
    assert SWAP_POOL_DEPTH == 120
    assert REPAIR_TOP_K == 12
    assert REPAIR_TOP_K_PER_PRICE == 6


# ---------------------------------------------------------------------------
# Certification against the exact optimum, where exact is possible
# ---------------------------------------------------------------------------


def test_repair_reaches_the_exact_optimum_on_a_small_fixture(board):
    """The only honest certification available: exhaustive enumeration.

    No MILP, branch-and-bound or DP facility is installed -- numpy is this
    project's sole runtime dependency -- and the real board refuses exact
    enumeration at 1.5e11 combinations. So certification is possible only where
    the pool is small enough to enumerate, which is what this fixture builds.
    """
    from ceauction.auction.completion import enumerate_completions_exactly

    st = board.state
    focus = st.focus_owner_id
    px = ProxyEvaluator(st.pool, st.settings, 16, 7)
    costs = board.costs["base"]

    # A deliberately tiny board: 14 available players, 3 slots to fill.
    specs = sorted(st.available_specs, key=lambda s: -s.base_mean)
    by = {}
    for s in specs:
        by.setdefault(Position(int(s.position)).name, []).append(s)
    small = (by["QB"][:4] + by["RB"][:4] + by["WR"][:4] + by["TE"][:2])
    ids = [s.player_id for s in small]

    held = [by["QB"][0].player_id, by["RB"][0].player_id,
            by["WR"][0].player_id] + [s.player_id for s in specs[40:52]]
    held = held[:12]
    cur = st
    for pid in held:
        cur = cur.apply_purchase(pid, focus, 1, validate=False)
    owned = frozenset(cur.owner(focus).player_ids)
    slots = cur.owner(focus).open_slots
    if slots <= 0 or slots > 3:
        pytest.skip("fixture did not produce an enumerable slot count")

    budget = cur.owner(focus).budget_remaining
    pool_specs = [s for s in small if s.player_id not in owned]
    cost_map = {s.player_id: costs.cost_of(s.player_id, 1) for s in pool_specs}
    exact = enumerate_completions_exactly(
        pool_specs, cost_map, cur.owner(focus).counts, slots, budget,
        max_combinations=400_000)
    if not exact:
        pytest.skip("no legal completion in the reduced fixture")

    best_exact = max(px.strength(tuple(owned) + add) for add, _ in exact)

    # Start local repair from the WEAKEST legal completion.
    worst_add, worst_cost = min(
        exact, key=lambda t: px.strength(tuple(owned) + t[0]))
    start = Completion(added=worst_add, roster=tuple(owned) + worst_add,
                       added_cost=worst_cost,
                       proxy=px.strength(tuple(owned) + worst_add))
    r = repair_completion(
        start, state=cur, costs=costs, proxy=px, candidate_id=None,
        owned=owned, budget=budget, pool=ids, default_cost=1)

    assert r.repaired.proxy <= best_exact + 1e-9, "beat the exact optimum"
    assert r.repaired.proxy == pytest.approx(best_exact, abs=1e-9), (
        f"local repair reached {r.repaired.proxy:.6f} but the exact optimum "
        f"is {best_exact:.6f}; a one-swap local optimum missed the global one")
