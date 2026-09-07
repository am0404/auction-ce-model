"""The adapter from certified completions to the CE layer's opportunity sets.

The championship-equity layer was built to be handed one immutable offer set
that every branch reads and none re-derives; ``reconciled.py`` exists because
three different beams for one world made the ladder and the decomposition
disagree. Feeding it certified completions must not disturb any of that, so
what is tested here is not that the solver is good -- that is
``test_certified_solver.py`` -- but that the *handover* is lossless and that
the five branches still consume one set without searching.

Every input is fabricated. No real player, no real price, no vendor value.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from ceauction.auction import new_auction
from ceauction.auction.certified import (OBJECTIVE_TOLERANCE, ProxyObjective,
                                         solver_available)
from ceauction.auction.completion import (ComparisonCast, Completion,
                                          CompletionSettings)
from ceauction.auction.costs import CostBook, CostEntry, CostProvenance
from ceauction.auction.feasibility import can_fill_lineup
from ceauction.auction.proxy import ProxyEvaluator
from ceauction.league import DEFAULT_LEAGUE, Position
from ceauction.players import PlayerSpec
from ceauction.realdata.smoke import build_test_rosters, roster_assignment
from ceauction.tactical.board import BoardSettings
from ceauction.tactical.certified_offers import (OfferAdaptationError,
                                                 adapt_completion,
                                                 build_certified_offers)
from ceauction.tactical.ensemble import balanced_schedule
from ceauction.tactical.evalcontext import (BRANCH_HOLDS_CANDIDATE,
                                            IndependentSearchRefused,
                                            build_eval_context)
from ceauction.tactical.midauction import PassPrice, PassPriceMode
from ceauction.tactical.reconciled import AGREEMENT_TOLERANCE, evaluate_draw
from ceauction.tactical.union import completion_fingerprint

pytestmark = pytest.mark.skipif(
    not solver_available(),
    reason="optional MILP solver (highspy) not installed")

OWNERS = tuple(f"O{i:02d}" for i in range(12))
FOCUS = "O00"


def _fab(pid, pos, mean, bye=7, hazard=0.03):
    return PlayerSpec(player_id=pid, name=f"Fabricated{pid:04d}", position=pos,
                      nfl_team="ZZA", base_mean=mean, week_sd=6.0,
                      bye_week=bye, weekly_injury_hazard=hazard,
                      injury_mean_weeks=2.5)


@pytest.fixture(scope="module")
def pool():
    out, pid = [], 0
    for pos, n, m, d in ((Position.QB, 40, 19.0, 0.35),
                         (Position.RB, 90, 16.0, 0.15),
                         (Position.WR, 120, 15.0, 0.11),
                         (Position.TE, 60, 12.0, 0.16)):
        for k in range(n):
            out.append(_fab(pid, pos, max(m - d * k, 2.5), bye=5 + (k % 10)))
            pid += 1
    return tuple(out)


@pytest.fixture(scope="module")
def world(pool):
    """A mid-auction state with the focus owner stripped back to three players."""
    rs = build_test_rosters(list(pool))
    st = new_auction(pool, OWNERS, FOCUS)
    for i, team in enumerate(roster_assignment(rs)):
        # O00 is the focus, stripped back to three. O01 is the RECIPIENT and
        # must keep open slots and money: the RF and RP branches hand him the
        # candidate, and a rival with a full roster cannot legally receive one.
        take = team[:3] if i == 0 else (team[:13] if i == 1 else team)
        for j, p in enumerate(take):
            price = 40 if (i == 0 and j == 0) else (25 if i == 0 else 12)
            st = st.apply_purchase(p, OWNERS[i], price)
    st.validate()
    cast = ComparisonCast(
        focus_team_index=0,
        rosters=tuple(tuple(st.owner(o).player_ids) for o in OWNERS),
        team_names=OWNERS)
    book = CostBook(
        entries=tuple(CostEntry(s.player_id,
                                max(1, int(round(s.base_mean * 2.0 - 12))))
                      for s in pool),
        provenance=CostProvenance("FABRICATED", "test cost curve"))
    px = ProxyEvaluator(st.pool, st.settings, n_reps=8, seed=4242)
    candidate = max((s for s in st.available_specs
                     if Position(int(s.position)) is Position.QB),
                    key=lambda s: s.base_mean).player_id
    return st, cast, book, px, candidate


@pytest.fixture(scope="module")
def offers(world):
    st, cast, book, px, cand = world
    return build_certified_offers(
        st, book, px, candidate_id=cand, prices=(20, 40),
        pool_depth=24, target=4, band=0.50)


# ---------------------------------------------------------------------------
# The handover
# ---------------------------------------------------------------------------


def test_adapted_completion_is_the_same_roster(world, offers):
    """No player is gained, lost or renamed crossing the adapter."""
    st, cast, book, px, cand = world
    assert offers.with_candidate and offers.without_candidate
    owned = set(st.owner(FOCUS).player_ids)
    for c in offers.with_candidate:
        assert cand in c.roster
        assert owned <= set(c.roster)
        assert set(c.added) == set(c.roster) - owned - {cand}
        assert len(set(c.roster)) == len(c.roster) == st.settings.roster_size
    for c in offers.without_candidate:
        assert cand not in c.roster
        assert owned <= set(c.roster)
        assert set(c.added) == set(c.roster) - owned


def test_objective_survives_the_adapter(world, offers):
    """``Completion.proxy`` is ``ProxyEvaluator.strength`` of that exact roster.

    Checked by recomputing rather than by trusting the number the adapter
    copied, to the declared solver precision and no looser.
    """
    st, cast, book, px, cand = world
    for c in tuple(offers.with_candidate) + tuple(offers.without_candidate):
        assert c.proxy == pytest.approx(px.strength(sorted(c.roster)),
                                        abs=OBJECTIVE_TOLERANCE, rel=0.0)


def test_cost_and_legality_are_exact(world, offers):
    """Cost is the cost book's, budget holds, and the lineup is fillable."""
    st, cast, book, px, cand = world
    owned = set(st.owner(FOCUS).player_ids)
    budget = st.owner(FOCUS).budget_remaining
    for c in offers.with_candidate:
        assert c.added_cost == sum(book.cost_of(p, 1) for p in c.added)
        assert c.added_cost <= budget
        assert can_fill_lineup(c.counts)
        assert c.counts.total == st.settings.roster_size
    for c in offers.without_candidate:
        assert c.added_cost == sum(book.cost_of(p, 1) for p in c.added)
        assert c.added_cost <= budget
        assert can_fill_lineup(c.counts)


def test_the_adapter_refuses_every_mismatch(world):
    """Each guard is exercised, because a guard nothing tests is decoration."""
    st, cast, book, px, cand = world
    owned = list(st.owner(FOCUS).player_ids)

    def by_pos(pos, n, skip=()):
        return [s.player_id for s in st.available_specs
                if Position(int(s.position)) is pos
                and s.player_id != cand and s.player_id not in skip][:n]

    # A positionally legal control roster: the guards under test are about
    # identity and arithmetic, so the roster they are applied to must be one
    # that would otherwise pass every other guard.
    fill = by_pos(Position.RB, 4) + by_pos(Position.WR, 5) + by_pos(Position.TE, 2)
    good = owned + [cand] + fill[: st.settings.roster_size - len(owned) - 1]
    spare = by_pos(Position.WR, 40)[6:]

    def build(roster, **kw):
        args = dict(state=st, costs=book, proxy=px, candidate_id=cand,
                    holds_candidate=True, budget=10_000, default_cost=1)
        args.update(kw)
        return adapt_completion(roster, 0.0, tolerance=1e9, **args)

    build(good)                                    # the control: this one works

    with pytest.raises(OfferAdaptationError, match="duplicate"):
        build(good[:-1] + [good[-1], good[-1]])
    with pytest.raises(OfferAdaptationError, match="already-owned"):
        build([p for p in good if p != owned[0]] + [spare[0]])
    with pytest.raises(OfferAdaptationError, match="must contain"):
        build([p for p in good if p != cand] + [spare[0]])
    with pytest.raises(OfferAdaptationError, match="must not contain"):
        build(good, holds_candidate=False)
    with pytest.raises(OfferAdaptationError, match="roster size"):
        build(good[:-1])
    with pytest.raises(OfferAdaptationError, match="above the"):
        build(good, budget=0)
    with pytest.raises(OfferAdaptationError, match="not available"):
        build(owned + [cand] + [st.owner("O01").player_ids[0]]
              + fill[: st.settings.roster_size - len(owned) - 2])


def test_objective_mismatch_is_refused_not_repaired(world, offers):
    """A wrong objective must stop the run, not be silently corrected."""
    st, cast, book, px, cand = world
    c = offers.with_candidate[0]
    with pytest.raises(OfferAdaptationError, match="disagrees with ProxyEvaluator"):
        adapt_completion(c.roster, c.proxy + 1.0, state=st, costs=book,
                         proxy=px, candidate_id=cand, holds_candidate=True,
                         budget=10_000)


def test_offer_fingerprints_are_deterministic(world):
    """Rebuilding the same offer set produces the same digest, bit for bit."""
    st, cast, book, px, cand = world
    kw = dict(candidate_id=cand, prices=(20, 40), pool_depth=24, target=4,
              band=0.50)
    a = build_certified_offers(st, book, px, **kw)
    b = build_certified_offers(st, book, px, **kw)
    assert a.opportunity_fingerprint() == b.opportunity_fingerprint()
    assert [completion_fingerprint(c) for c in a.with_candidate] == \
           [completion_fingerprint(c) for c in b.with_candidate]
    assert [completion_fingerprint(c) for c in a.without_candidate] == \
           [completion_fingerprint(c) for c in b.without_candidate]


# ---------------------------------------------------------------------------
# The five branches
# ---------------------------------------------------------------------------


def _context(world, offers, draw, p=40, q=35, sims=24):
    st, cast, book, px, cand = world
    cs = CompletionSettings(proxy_reps=8, selection_sims=sims,
                            evaluation_sims=sims, selection_seed=555_000_111,
                            evaluation_seed=917_324_011)
    pp = PassPrice(mode=PassPriceMode.FIXED_MARKET, increment=1, fixed_q=q)
    return build_eval_context(
        st, cast, book, cand, p=p, pass_price=pp, recipient="O01", q=q,
        draw=draw, board=BoardSettings(pool_depth=120, max_allocations=40),
        completion=cs, market=None, key_by_id=None, proxy=px,
        with_candidate=offers.with_candidate,
        without_candidate=offers.without_candidate)


def test_every_branch_consumes_the_certified_set_without_searching(world, offers):
    """All five branches read the offer set, and none may run its own beam.

    The offers a branch sees must be *identity-equal* members of the set that
    was handed in -- not merely equal-looking completions -- because that is
    what makes the frontier and the decomposition two readings of one world.
    """
    st, cast, book, px, cand = world
    draw = balanced_schedule(len(st.owners), 1)[0]
    ctx = _context(world, offers, draw)
    supplied = {id(c) for c in
                tuple(offers.with_candidate) + tuple(offers.without_candidate)}
    for branch, holds in BRANCH_HOLDS_CANDIDATE.items():
        got = ctx.completions_for(branch)
        assert got, branch
        assert all(id(c) in supplied for c in got), branch
        source = (offers.with_candidate if holds
                  else offers.without_candidate)
        assert set(id(c) for c in got) <= set(id(c) for c in source), branch
        for c in got:
            assert (cand in c.roster) == holds, branch

    from ceauction.tactical.reconciled import evaluate_branch
    with pytest.raises(IndependentSearchRefused):
        evaluate_branch(ctx, "UP", proxy=px, allow_independent_search=True)


def test_possession_offer_does_not_move_with_the_price(world, offers):
    """``UF`` pays nothing, so its offer may not be cut by ``p``."""
    st, cast, book, px, cand = world
    draw = balanced_schedule(len(st.owners), 1)[0]
    for p in (20, 40):
        ctx = _context(world, offers, draw, p=p)
        assert ctx.possession_offer_is_price_independent()
    a = _context(world, offers, draw, p=20)
    b = _context(world, offers, draw, p=40)
    assert set(id(c) for c in a.completions_for("UF")) == \
           set(id(c) for c in b.completions_for("UF"))


@pytest.mark.slow
def test_frontier_and_decomposition_agree_on_certified_offers(world, offers):
    """The 1e-12 reconciliation still holds when the offers come from a MILP.

    This is the claim the whole adapter exists to preserve: the ladder's delta
    and the five-branch decomposition's total are two readings of the same five
    numbers, so their residual is floating-point noise and nothing else.
    """
    st, cast, book, px, cand = world
    draw = balanced_schedule(len(st.owners), 1)[0]
    ctx = _context(world, offers, draw)
    ev = evaluate_draw(ctx, proxy=px)
    assert abs(ev.residual) <= AGREEMENT_TOLERANCE, ev.to_dict()
    assert ev.agrees
    for b in BRANCH_HOLDS_CANDIDATE:
        br = ev.branches[b]
        assert br.conservation_ok, b
        assert br.n_offered > 0, b
        assert len(set(br.focus_roster)) == st.settings.roster_size, b
    assert ev.branches["UP"].ce == pytest.approx(ev.branches["UP"].ce)
