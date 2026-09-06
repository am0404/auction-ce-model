"""Property-style invariants for the auction layer, plus repository hygiene.

**Every input is fabricated.** These are the checks a hostile audit ran against
the layer, kept permanently so a future change has to break them on purpose.

Deterministic throughout: the "random" play uses a fixed seed and a fixed
number of trials, so a failure here is reproducible rather than a flake.
"""

from __future__ import annotations

import itertools
import pathlib
import random
import subprocess

import numpy as np
import pytest

from ceauction.auction import (CompletionSettings, CostBook, CostProvenance,
                               PassDestination, ReservationCache, ScenarioSetup,
                               complete_roster)
from ceauction.auction.completion import _build_roster_set
from ceauction.auction.demo import build_demo_auction
from ceauction.auction.feasibility import PositionCounts, can_fill_lineup
from ceauction.league import Position
from ceauction.simulate import simulate_seasons

REPO = pathlib.Path(__file__).resolve().parents[1]

#: The eight starting slots as raw eligibility sets, written out independently
#: of the code under test so the oracle cannot inherit its bugs.
_SLOTS = (
    frozenset({Position.QB}),
    frozenset({Position.RB}), frozenset({Position.RB}),
    frozenset({Position.WR, Position.TE}),
    frozenset({Position.WR, Position.TE}),
    frozenset({Position.WR, Position.TE}),
    frozenset({Position.RB, Position.WR, Position.TE}),
    frozenset({Position.QB, Position.RB, Position.WR, Position.TE}),
)


def _max_matching(counts: PositionCounts) -> int:
    """Largest number of slots fillable, by augmenting-path bipartite matching.

    A deliberately slow, obviously-correct oracle. It knows nothing about
    laminar families or Hall's condition; it just matches players to slots.
    """
    players = ([Position.QB] * counts.qb + [Position.RB] * counts.rb
               + [Position.WR] * counts.wr + [Position.TE] * counts.te)
    assign = [-1] * len(_SLOTS)

    def try_assign(pi, seen):
        for si, elig in enumerate(_SLOTS):
            if players[pi] in elig and si not in seen:
                seen.add(si)
                if assign[si] == -1 or try_assign(assign[si], seen):
                    assign[si] = pi
                    return True
        return False

    return sum(1 for pi in range(len(players)) if try_assign(pi, set()))


def test_the_hall_constraints_agree_with_brute_force_matching():
    """The counting rules must be the matching problem, not an approximation.

    Every count vector up to a full roster is checked against an independent
    matcher. A single disagreement would mean the auction is enforcing a quota
    it invented rather than the league's actual lineup rules.
    """
    mismatches = []
    for q, r, w, t in itertools.product(range(5), range(6), range(7), range(5)):
        if q + r + w + t > 15:
            continue
        counts = PositionCounts(q, r, w, t)
        if (_max_matching(counts) >= 8) != can_fill_lineup(counts):
            mismatches.append((q, r, w, t))
    assert not mismatches, f"{len(mismatches)} disagreements, e.g. {mismatches[:5]}"


def test_random_legal_play_never_breaks_a_money_invariant():
    """Fifty randomised auctions, every invariant checked after every purchase."""
    rng = random.Random(20260904)
    for _ in range(50):
        st = build_demo_auction().state
        for _ in range(rng.randint(1, 20)):
            owners = [o for o in st.owners if o.open_slots > 0]
            avail = st.available_ids
            if not owners or not avail:
                break
            o = rng.choice(owners)
            hi = st.max_bid(o.owner_id)
            if hi < o.min_bid:
                continue
            pid = rng.choice(avail)
            price = rng.randint(o.min_bid, hi)
            if not st.purchase_is_legal(pid, o.owner_id, price):
                continue
            st = st.apply_purchase(pid, o.owner_id, price)
            for ow in st.owners:
                assert ow.budget_remaining >= 0
                assert ow.budget_remaining >= ow.open_slots * ow.min_bid
                assert ow.spent + ow.budget_remaining == ow.budget_start
                assert ow.n_players <= ow.roster_capacity
                assert ow.can_still_field_lineup()
        assert st.problems() == []


def test_the_legal_maximum_formula_holds_for_every_owner():
    st = build_demo_auction().state
    for o in st.owners:
        expected = (o.budget_remaining - (o.open_slots - 1) * o.min_bid
                    if o.open_slots > 0 else 0)
        assert o.max_bid == expected
        assert o.max_bid == o.discretionary + o.min_bid or o.open_slots == 0


def test_spending_the_maximum_leaves_exactly_one_dollar_per_open_slot():
    st = build_demo_auction().state
    o = st.focus
    pid = next(p for p in st.available_ids
               if st.purchase_is_legal(p, o.owner_id, o.max_bid))
    after = st.apply_purchase(pid, o.owner_id, o.max_bid).focus
    assert after.budget_remaining == after.open_slots * after.min_bid


def test_no_cache_key_collides_across_any_dimension():
    """Eight ways to be a different question; eight different keys."""
    d = build_demo_auction()
    cand = d.default_candidate().player_id
    other = next(p for p in d.state.available_ids if p != cand)
    S = CompletionSettings(beam_width=50, candidate_pool=30, finalists=2,
                           selection_sims=1200, evaluation_sims=1200)
    base = ScenarioSetup("s1", d.state, d.cast, d.costs)
    dest = PassDestination.unavailable()
    k0 = ReservationCache.key(base, cand, 10, dest, S)

    variants = {
        "scenario": ReservationCache.key(
            ScenarioSetup("s2", d.state, d.cast, d.costs), cand, 10, dest, S),
        "state": ReservationCache.key(
            ScenarioSetup("s1", d.state.withdraw(other), d.cast, d.costs),
            cand, 10, dest, S),
        "candidate": ReservationCache.key(base, other, 10, dest, S),
        "price": ReservationCache.key(base, cand, 11, dest, S),
        "destination": ReservationCache.key(
            base, cand, 10, PassDestination.to_rival("Owner02", 10), S),
        "settings": ReservationCache.key(
            base, cand, 10, dest,
            CompletionSettings(beam_width=51, candidate_pool=30, finalists=2,
                               selection_sims=1200, evaluation_sims=1200)),
        "sims": ReservationCache.key(
            base, cand, 10, dest,
            CompletionSettings(beam_width=50, candidate_pool=30, finalists=2,
                               selection_sims=2400, evaluation_sims=2400)),
        "costs": ReservationCache.key(
            ScenarioSetup("s1", d.state, d.cast,
                          CostBook(d.costs.entries,
                                   CostProvenance("PROVISIONAL", "elsewhere"))),
            cand, 10, dest, S),
    }
    collisions = [name for name, k in variants.items() if k == k0]
    assert not collisions, f"cache would serve a stale answer across {collisions}"
    assert len(set(variants.values()) | {k0}) == len(variants) + 1


def test_common_random_numbers_pair_two_identical_leagues():
    d = build_demo_auction()
    res = complete_roster(d.state, d.cast, d.costs, evaluate_ce=False,
                          settings=CompletionSettings(beam_width=40,
                                                      candidate_pool=25,
                                                      finalists=2))
    rs = _build_roster_set(d.state, d.cast, res.best.roster)
    a = simulate_seasons(rs, 400, 999)
    b = simulate_seasons(rs, 400, 999)
    assert np.array_equal(a.champion, b.champion)
    assert float((a.champion_indicator(0) - b.champion_indicator(0)).sum()) == 0.0


# ==========================================================================
# Repository hygiene
# ==========================================================================


def _tracked():
    out = subprocess.run(["git", "ls-files"], capture_output=True, text=True,
                         cwd=str(REPO))
    return out.stdout.split()


def test_no_local_data_file_is_tracked():
    """This repository is public and local_data holds the vendor rows."""
    leaks = [f for f in _tracked() if f.startswith("local_data/")]
    assert not leaks, f"proprietary paths staged: {leaks}"


def test_no_committed_document_names_a_real_player():
    """Committed output carries stable ids, never identities."""
    markers = ("Josh Allen", "Ja'Marr", "Bijan", "McCaffrey", "Jefferson",
               "Chris Godwin", "Devon Achane")
    bad = []
    for f in _tracked():
        if not (f.startswith("docs/") or f.endswith(".md")):
            continue
        path = REPO / f
        if not path.exists():
            continue          # staged for deletion; nothing to read
        text = path.read_text(encoding="utf-8", errors="ignore")
        for m in markers:
            if m in text:
                bad.append((f, m))
    assert not bad, f"real player names in committed files: {bad[:5]}"


def test_the_demo_is_labelled_fabricated_everywhere():
    d = build_demo_auction()
    assert all(s.data_source.startswith("FABRICATED") for s in d.pool)
    assert d.costs.level == "FABRICATED"
    assert not d.costs.provenance.may_be_reported_as_a_value
    assert "Not a market" in d.costs.provenance.notes


def test_no_source_file_claims_an_optimal_or_recommended_result():
    """Grep the auction layer for the claims it is not entitled to make."""
    banned = ("optimal roster", "optimal completion", "the optimal ",
              "recommended bid", "your max bid", "guaranteed value")
    offenders = []
    root = REPO / "src" / "ceauction"
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8").lower()
        for phrase in banned:
            idx = text.find(phrase)
            while idx >= 0:
                window = text[max(0, idx - 60):idx]
                if not any(n in window for n in ("not ", "never ", "no ", "nothing")):
                    offenders.append((path.name, phrase))
                idx = text.find(phrase, idx + len(phrase))
    assert not offenders, offenders
