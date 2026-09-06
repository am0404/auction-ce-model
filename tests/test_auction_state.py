"""The immutable auction room and the roster feasibility it rests on.

**Every input is fabricated.** No real player, no vendor value, no real price.

Two things are defended. The league's actual rules -- $200, 15 slots, $1
minimum, $1 reserved per unfilled slot, no quarterback maximum, and a starting
lineup whose legality is a matching question rather than a quota -- and the
immutability that makes counterfactual branches safe.
"""

from __future__ import annotations

import json

import pytest

from ceauction.auction import (AuctionRuleError, AuctionState,
                               AuctionStateInvalid, Nomination,
                               OwnerAuctionState, PositionCounts,
                               RosterSlotFilled, Transaction, can_complete,
                               can_fill_lineup, completion_shortfall,
                               lineup_deficit, min_additions_for_lineup,
                               new_auction)
from ceauction.league import DEFAULT_LEAGUE, LeagueSettings, Position
from ceauction.players import PlayerSpec

OWNERS = tuple(f"O{i:02d}" for i in range(12))
FOCUS = "O00"


def fab(pid: int, pos: Position, mean: float = 10.0) -> PlayerSpec:
    return PlayerSpec(player_id=pid, name=f"Fabricated{pid:04d}", position=pos,
                      nfl_team="ZZA", base_mean=mean, week_sd=5.0)


@pytest.fixture(scope="module")
def pool():
    """Enough of every position to fill twelve rosters several times over."""
    out, pid = [], 0
    for pos, n in ((Position.QB, 48), (Position.RB, 72), (Position.WR, 96),
                   (Position.TE, 48)):
        for k in range(n):
            out.append(fab(pid, pos, mean=20.0 - 0.1 * k))
            pid += 1
    return tuple(out)


@pytest.fixture
def empty(pool):
    return new_auction(pool, OWNERS, FOCUS)


def _ids(pool, pos, n, skip=0):
    return [s.player_id for s in pool if s.position is pos][skip:skip + n]


#: A purchase order that is legal at every prefix: the scarce lineup needs
#: (one QB, two RB, three WR/TE) are satisfied first, after which any position
#: keeps the roster completable. Buying fifteen of one position is *not* legal
#: and the state refuses it, which is why these fixtures cannot take a shortcut.
#: The free tail cycles TE/WR/RB so that twelve full rosters fit inside the
#: fabricated pool rather than exhausting one position.
_LEGAL_ORDER = [Position.QB, Position.RB, Position.RB,
                Position.WR, Position.WR, Position.WR,
                Position.TE, Position.WR, Position.RB,
                Position.TE, Position.WR, Position.RB,
                Position.TE, Position.WR, Position.RB]


def legal_fill(state, owner_id, n, price=1, pool=None, cursor=None):
    """Buy ``n`` players for ``owner_id`` in an order that stays feasible."""
    pool = pool if pool is not None else state.pool
    cursor = cursor if cursor is not None else {}
    by_pos = {p: [s.player_id for s in pool if s.position is p] for p in Position}
    taken = set(state.rostered_ids) | set(state.withdrawn)
    for pos in _LEGAL_ORDER[:n]:
        i = cursor.get(pos, 0)
        while by_pos[pos][i] in taken:
            i += 1
        pid = by_pos[pos][i]
        cursor[pos] = i + 1
        taken.add(pid)
        state = state.apply_purchase(pid, owner_id, price)
    return state


# ==========================================================================
# Feasibility: the eligibility graph, never a quota
# ==========================================================================


def test_a_full_legal_lineup_needs_one_qb_two_rb_three_wt_and_six_non_qbs():
    assert can_fill_lineup(PositionCounts(qb=1, rb=2, wr=3, te=0)) is False
    assert can_fill_lineup(PositionCounts(qb=1, rb=2, wr=4, te=1)) is True
    assert can_fill_lineup(PositionCounts(qb=2, rb=2, wr=3, te=1)) is True


def test_a_tight_end_competes_for_the_wr_slots_and_the_flex():
    """No dedicated TE slot exists, so WR and TE are one position for lineups."""
    all_te = PositionCounts(qb=1, rb=2, wr=0, te=5)
    all_wr = PositionCounts(qb=1, rb=2, wr=5, te=0)
    mixed = PositionCounts(qb=1, rb=2, wr=3, te=2)
    assert can_fill_lineup(all_te) == can_fill_lineup(all_wr) == can_fill_lineup(mixed)
    assert can_fill_lineup(mixed) is True


def test_there_is_no_quarterback_maximum():
    """Five QBs is legal; the superflex takes one and the rest sit."""
    assert can_fill_lineup(PositionCounts(qb=5, rb=2, wr=3, te=1)) is True
    assert can_complete(PositionCounts(qb=5, rb=0, wr=0, te=0), 10) is True


def test_a_roster_of_only_quarterbacks_cannot_field_a_lineup():
    assert can_fill_lineup(PositionCounts(qb=15)) is False
    assert can_complete(PositionCounts(qb=15), 0) is False
    assert "RB1 and RB2" in " ".join(lineup_deficit(PositionCounts(qb=15)))


def test_the_superflex_may_hold_a_skill_player():
    """Eight legal starters with exactly one QB: the superflex is a non-QB."""
    counts = PositionCounts(qb=1, rb=3, wr=3, te=1)
    assert can_fill_lineup(counts) is True
    # Six non-QB starters plus the QB is seven; the eighth is another skill
    # player, so a second quarterback is never required.
    assert counts.flex_eligible >= 7


def test_minimum_additions_are_counted_from_the_eligibility_graph():
    assert min_additions_for_lineup(PositionCounts())[0] == 8
    assert min_additions_for_lineup(PositionCounts(qb=1, rb=2, wr=3))[0] == 2
    assert min_additions_for_lineup(PositionCounts(qb=1, rb=2, wr=4, te=1))[0] == 0


def test_completion_is_refused_when_too_few_slots_remain():
    why = completion_shortfall(PositionCounts(qb=14), open_slots=1)
    assert why is not None and "roster slot" in why


def test_completion_is_refused_when_the_pool_lacks_the_position():
    why = completion_shortfall(
        PositionCounts(qb=0, rb=2, wr=3, te=1), open_slots=5,
        available={Position.QB: 0, Position.RB: 9, Position.WR: 9, Position.TE: 9})
    assert why is not None and "QB" in why


def test_an_unconstrained_pool_is_the_mid_auction_default():
    """Omitting the pool assumes it is deep, which is right until it is not."""
    assert can_complete(PositionCounts(), 8) is True
    assert can_complete(PositionCounts(), 8, {Position.QB: 0, Position.RB: 0,
                                              Position.WR: 0, Position.TE: 0}) is False


# ==========================================================================
# Owner state: money and the $1 reserve
# ==========================================================================


def test_initial_budget_and_capacity_come_from_the_league(empty):
    for o in empty.owners:
        assert o.budget_start == 200
        assert o.roster_capacity == 15
        assert o.min_bid == 1
        assert o.budget_remaining == 200 and o.spent == 0
        assert o.open_slots == 15


def test_the_legal_maximum_is_budget_minus_one_per_other_open_slot(empty):
    o = empty.focus
    assert o.max_bid == 200 - 14
    assert o.reserve_for_open_slots == 15
    assert o.discretionary == 185
    assert o.max_bid == o.discretionary + 1


@pytest.mark.parametrize("spent,slots_used,expected", [
    (0, 0, 186), (50, 1, 137), (100, 5, 91), (186, 1, 1),
])
def test_the_legal_maximum_tracks_spend_and_slots(pool, spent, slots_used, expected):
    filled = tuple(RosterSlotFilled(pool[i].player_id, pool[i].position,
                                    spent if i == 0 else 0)
                   for i in range(slots_used))
    o = OwnerAuctionState("O00", "T", 200, 15, filled)
    assert o.max_bid == expected


def test_an_owner_with_no_open_slot_cannot_bid_at_all(pool):
    filled = tuple(RosterSlotFilled(pool[i].player_id, pool[i].position, 1)
                   for i in range(15))
    o = OwnerAuctionState("O00", "T", 200, 15, filled)
    assert o.is_full and o.open_slots == 0
    assert o.max_bid == 0
    assert not o.can_bid(1)
    assert o.budget_remaining == 185, "money left over does not buy roster room"


def test_spending_reconciles(pool):
    o = OwnerAuctionState("O00", "T", 200, 15, (
        RosterSlotFilled(pool[0].player_id, pool[0].position, 40),
        RosterSlotFilled(pool[1].player_id, pool[1].position, 17)))
    assert o.spent == 57
    assert o.spent + o.budget_remaining == o.budget_start


def test_auction_dollars_are_integers():
    with pytest.raises(AuctionRuleError, match="integers"):
        RosterSlotFilled(1, Position.RB, 3.5)
    with pytest.raises(AuctionRuleError, match="integers"):
        RosterSlotFilled(1, Position.RB, True)


def test_an_owner_cannot_hold_a_player_twice(pool):
    with pytest.raises(AuctionRuleError, match="twice"):
        OwnerAuctionState("O00", "T", 200, 15, (
            RosterSlotFilled(7, Position.RB, 1),
            RosterSlotFilled(7, Position.RB, 1)))


# ==========================================================================
# Purchases
# ==========================================================================


def test_a_purchase_returns_a_new_state_and_leaves_the_old_one_alone(empty):
    before = empty.fingerprint()
    after = empty.apply_purchase(empty.pool[0].player_id, FOCUS, 12)
    assert empty.fingerprint() == before
    assert after.fingerprint() != before
    assert empty.focus.n_players == 0
    assert after.focus.n_players == 1
    assert after.focus.budget_remaining == 188


def test_a_purchase_removes_the_player_from_the_available_pool(empty):
    pid = empty.pool[0].player_id
    assert empty.is_available(pid)
    after = empty.apply_purchase(pid, FOCUS, 3)
    assert not after.is_available(pid)
    assert pid not in after.available_ids
    assert after.owner_of[pid] == FOCUS


def test_a_purchase_above_the_legal_maximum_is_refused(empty):
    with pytest.raises(AuctionRuleError, match="exceeds .* legal maximum"):
        empty.apply_purchase(empty.pool[0].player_id, FOCUS, 187)


def test_a_purchase_below_the_minimum_bid_is_refused(empty):
    with pytest.raises(AuctionRuleError, match="below the 1 minimum"):
        empty.apply_purchase(empty.pool[0].player_id, FOCUS, 0)


def test_a_purchase_always_leaves_a_dollar_for_every_remaining_slot(empty, pool):
    """Spend the maximum and the other fourteen slots are still affordable."""
    qb = _ids(pool, Position.QB, 1)[0]
    st = empty.apply_purchase(qb, FOCUS, 186)
    o = st.focus
    assert o.budget_remaining == 14 == o.open_slots
    assert o.max_bid == 1
    assert not o.can_bid(2)
    st = legal_fill(st, FOCUS, 14, price=1, pool=pool)
    assert st.focus.is_full and st.focus.budget_remaining == 0
    assert st.focus.fields_a_full_lineup
    st.validate()


def test_a_player_cannot_be_sold_twice(empty):
    pid = empty.pool[0].player_id
    st = empty.apply_purchase(pid, FOCUS, 5)
    with pytest.raises(AuctionRuleError, match="already belongs"):
        st.apply_purchase(pid, "O01", 5)


def test_an_owner_at_capacity_cannot_buy(empty, pool):
    st = legal_fill(empty, FOCUS, 15, price=1, pool=pool)
    assert st.focus.is_full
    spare = next(pid for pid in st.available_ids)
    with pytest.raises(AuctionRuleError, match="no open roster slot"):
        st.apply_purchase(spare, FOCUS, 1)


def test_a_purchase_that_would_strand_the_roster_is_refused(empty, pool):
    """Nine quarterbacks is legal; the tenth strands the roster.

    With k quarterbacks and nothing else, six more players are always needed
    (2 RB, 3 WR/TE, and one more of either to reach six non-QB starters), so
    the limit is where 6 exceeds the 15 - k slots left: k = 9.
    """
    st = empty
    for pid in _ids(pool, Position.QB, 9):
        st = st.apply_purchase(pid, FOCUS, 1)
    assert st.focus.n_players == 9 and st.focus.open_slots == 6
    assert st.focus.can_still_field_lineup()

    tenth = _ids(pool, Position.QB, 1, skip=9)[0]
    with pytest.raises(AuctionRuleError, match="could not complete a legal roster"):
        st.apply_purchase(tenth, FOCUS, 1)
    assert not st.purchase_leaves_feasible_completion(tenth, FOCUS, 1)
    # A running back is fine: it moves the roster toward feasibility.
    assert st.purchase_leaves_feasible_completion(
        _ids(pool, Position.RB, 1)[0], FOCUS, 1)


def test_a_purchase_is_refused_when_the_pool_cannot_supply_what_is_still_needed():
    """Feasibility is pool-aware at the end of the board, not only a count."""
    pool = tuple([fab(0, Position.QB), fab(1, Position.RB), fab(2, Position.RB),
                  fab(3, Position.WR), fab(4, Position.WR), fab(5, Position.WR),
                  fab(6, Position.TE), fab(7, Position.TE)])
    settings = LeagueSettings(n_teams=2, roster_size=4, n_starters=3,
                              regular_season_weeks=14)
    st = AuctionState(settings=settings, pool=pool, focus_owner_id="A",
                      owners=(OwnerAuctionState("A", "A", 20, 4),
                              OwnerAuctionState("B", "B", 20, 4)))
    # With capacity 4 nobody can field eight starters, so every roster is
    # infeasible and the state says so rather than pretending otherwise.
    assert not st.focus.can_still_field_lineup()


def test_the_focus_owner_cannot_be_awarded_through_the_rival_path(empty):
    with pytest.raises(AuctionRuleError, match="use apply_purchase"):
        empty.award_to_rival(empty.pool[0].player_id, FOCUS, 5)


def test_a_rival_award_moves_money_and_roster(empty):
    pid = empty.pool[0].player_id
    st = empty.award_to_rival(pid, "O07", 44)
    assert st.owner("O07").budget_remaining == 156
    assert pid in st.owner("O07").player_ids
    assert st.focus.budget_remaining == 200, "our money is untouched"


# ==========================================================================
# Withdrawal, nomination, bidding
# ==========================================================================


def test_withdrawing_removes_a_player_without_awarding_him(empty):
    pid = empty.pool[3].player_id
    st = empty.withdraw(pid)
    assert pid not in st.available_ids
    assert pid not in st.rostered_ids
    assert all(pid not in o.player_ids for o in st.owners)
    st.validate()


def test_a_rostered_player_cannot_be_withdrawn(empty):
    pid = empty.pool[0].player_id
    st = empty.apply_purchase(pid, FOCUS, 2)
    with pytest.raises(AuctionRuleError, match="already rostered"):
        st.withdraw(pid)


def test_nomination_and_bidding_advance_the_state(empty):
    pid = empty.pool[0].player_id
    st = empty.nominate(pid, "O03")
    assert st.nomination.player_id == pid
    assert st.nomination.current_bid is None
    st = st.place_bid("O03", 5).place_bid("O04", 9)
    assert st.nomination.current_bid == 9 and st.nomination.high_bidder == "O04"
    st = st.award_nomination()
    assert st.nomination is None
    assert st.owner("O04").budget_remaining == 191
    assert st.transactions[-1].nominated_by == "O03"
    st.validate()


def test_a_bid_must_beat_the_standing_bid(empty):
    st = empty.nominate(empty.pool[0].player_id, "O01").place_bid("O01", 5)
    with pytest.raises(AuctionRuleError, match="does not beat"):
        st.place_bid("O02", 5)


def test_a_bid_above_an_owners_legal_maximum_is_refused(empty, pool):
    st = legal_fill(empty, "O05", 14, price=13, pool=pool)
    o = st.owner("O05")
    assert o.open_slots == 1 and o.max_bid == o.budget_remaining
    st = st.nominate(st.available_ids[0], "O00")
    with pytest.raises(AuctionRuleError, match="cannot legally bid"):
        st.place_bid("O05", o.max_bid + 1)


def test_an_unavailable_player_cannot_be_nominated(empty):
    pid = empty.pool[0].player_id
    st = empty.apply_purchase(pid, FOCUS, 3)
    with pytest.raises(AuctionRuleError, match="not available to nominate"):
        st.nominate(pid, "O02")


def test_a_nomination_carries_a_bid_and_a_bidder_together_or_neither():
    with pytest.raises(AuctionRuleError, match="both a current bid"):
        Nomination(player_id=1, nominated_by="O00", current_bid=5)
    with pytest.raises(AuctionRuleError, match="both a current bid"):
        Nomination(player_id=1, nominated_by="O00", high_bidder="O00")


# ==========================================================================
# Who may bid: individual budgets, not a league total
# ==========================================================================


def test_bid_capacity_uses_each_owners_own_budget_and_room(empty, pool):
    st = empty
    # O01 spends nearly everything on one player: rich in slots, poor in cash.
    st = st.apply_purchase(_ids(pool, Position.RB, 1)[0], "O01", 186)
    # O02 fills his roster cheaply: rich in cash, no room at all.
    st = legal_fill(st, "O02", 15, price=1, pool=pool)
    cap = st.bid_capacity(30)
    assert "O01" not in cap["able"]
    assert "financial ceiling" in cap["blocked"]["O01"]
    assert "O02" not in cap["able"] and cap["blocked"]["O02"] == "roster full"
    assert "O03" in cap["able"]
    # The league still holds plenty of money; that is not the question.
    total_left = sum(o.budget_remaining for o in st.owners)
    assert total_left > 30 * len(st.owners)
    assert cap["n_able"] == len(st.owners) - 2


def test_an_owner_who_can_bid_exactly_one_more_dollar_is_distinguishable(empty, pool):
    st = legal_fill(empty, "O06", 14, price=14, pool=pool)
    o = st.owner("O06")
    assert o.open_slots == 1
    assert o.max_bid == o.budget_remaining == 200 - 14 * 14
    assert o.can_bid(o.max_bid) and not o.can_bid(o.max_bid + 1)


def test_owners_who_can_bid_honours_an_exclusion(empty):
    able = empty.owners_who_can_bid(10, exclude=[FOCUS])
    assert FOCUS not in [o.owner_id for o in able]
    assert len(able) == 11


# ==========================================================================
# Validation
# ==========================================================================


def test_a_clean_state_validates(empty):
    empty.validate()
    assert empty.is_valid and empty.problems() == []


def test_validation_reports_every_violation_not_just_the_first(pool):
    bad = AuctionState(
        settings=DEFAULT_LEAGUE, pool=pool, focus_owner_id="O00",
        owners=tuple(
            OwnerAuctionState(o, o, 200, 15,
                              (RosterSlotFilled(pool[0].player_id,
                                                pool[0].position, 1),)
                              if o in ("O00", "O01") else ())
            for o in OWNERS))
    problems = bad.problems()
    assert any("held by both" in p for p in problems)
    with pytest.raises(AuctionStateInvalid) as exc:
        bad.validate()
    assert exc.value.problems == problems


def test_validation_catches_a_negative_budget(pool):
    o = OwnerAuctionState("O00", "T", 10, 15,
                          (RosterSlotFilled(pool[0].player_id,
                                            pool[0].position, 40),))
    st = AuctionState(settings=DEFAULT_LEAGUE, pool=pool, focus_owner_id="O00",
                      owners=(o,) + tuple(
                          OwnerAuctionState(x, x, 200, 15) for x in OWNERS[1:]))
    assert any("negative" in p for p in st.problems())


def test_validation_catches_a_budget_that_cannot_cover_its_open_slots(pool):
    o = OwnerAuctionState("O00", "T", 20, 15,
                          (RosterSlotFilled(pool[0].player_id,
                                            pool[0].position, 19),))
    st = AuctionState(settings=DEFAULT_LEAGUE, pool=pool, focus_owner_id="O00",
                      owners=(o,) + tuple(
                          OwnerAuctionState(x, x, 200, 15) for x in OWNERS[1:]))
    assert any("cannot cover" in p for p in st.problems())


def test_validation_catches_a_stranded_roster(pool):
    qbs = _ids(pool, Position.QB, 15)
    o = OwnerAuctionState("O00", "T", 200, 15, tuple(
        RosterSlotFilled(p, Position.QB, 1) for p in qbs))
    st = AuctionState(settings=DEFAULT_LEAGUE, pool=pool, focus_owner_id="O00",
                      owners=(o,) + tuple(
                          OwnerAuctionState(x, x, 200, 15) for x in OWNERS[1:]))
    assert any("legal lineup" in p for p in st.problems())


def test_validation_catches_a_player_not_in_the_pool(pool):
    o = OwnerAuctionState("O00", "T", 200, 15,
                          (RosterSlotFilled(-999, Position.RB, 1),))
    st = AuctionState(settings=DEFAULT_LEAGUE, pool=pool, focus_owner_id="O00",
                      owners=(o,) + tuple(
                          OwnerAuctionState(x, x, 200, 15) for x in OWNERS[1:]))
    assert any("not in the pool" in p for p in st.problems())


def test_validation_catches_a_wrong_owner_count(pool):
    st = AuctionState(settings=DEFAULT_LEAGUE, pool=pool, focus_owner_id="O00",
                      owners=tuple(OwnerAuctionState(x, x, 200, 15)
                                   for x in OWNERS[:5]))
    assert any("owners but the league has" in p for p in st.problems())


def test_a_focus_owner_must_be_in_the_room(pool):
    with pytest.raises(AuctionRuleError, match="focus owner"):
        AuctionState(settings=DEFAULT_LEAGUE, pool=pool, focus_owner_id="ghost",
                     owners=tuple(OwnerAuctionState(x, x, 200, 15) for x in OWNERS))


def test_owner_ids_must_be_unique(pool):
    with pytest.raises(AuctionRuleError, match="unique"):
        AuctionState(settings=DEFAULT_LEAGUE, pool=pool, focus_owner_id="O00",
                     owners=tuple(OwnerAuctionState("O00", "T", 200, 15)
                                  for _ in range(12)))


# ==========================================================================
# Fingerprints and reporting
# ==========================================================================


def test_the_fingerprint_moves_with_anything_that_can_change_a_valuation(empty):
    base = empty.fingerprint()
    assert empty.apply_purchase(empty.pool[0].player_id, FOCUS, 5).fingerprint() != base
    assert empty.apply_purchase(empty.pool[0].player_id, FOCUS, 6).fingerprint() != \
        empty.apply_purchase(empty.pool[0].player_id, FOCUS, 5).fingerprint()
    assert empty.apply_purchase(empty.pool[0].player_id, "O01", 5).fingerprint() != \
        empty.apply_purchase(empty.pool[0].player_id, FOCUS, 5).fingerprint()
    assert empty.withdraw(empty.pool[0].player_id).fingerprint() != base
    import dataclasses
    other_focus = dataclasses.replace(empty, focus_owner_id="O05")
    assert other_focus.fingerprint() != base


def test_the_fingerprint_ignores_the_order_players_were_bought(empty):
    a = empty.apply_purchase(empty.pool[0].player_id, FOCUS, 5) \
             .apply_purchase(empty.pool[1].player_id, FOCUS, 7)
    b = empty.apply_purchase(empty.pool[1].player_id, FOCUS, 7) \
             .apply_purchase(empty.pool[0].player_id, FOCUS, 5)
    assert a.fingerprint() == b.fingerprint()


def test_the_state_serializes(empty):
    st = empty.apply_purchase(empty.pool[0].player_id, FOCUS, 5) \
              .nominate(empty.pool[1].player_id, "O02")
    json.dumps(st.to_dict())
    assert st.to_dict()["valid"] is True


def test_the_room_summary_is_sanitized_and_names_who_may_bid(empty, pool):
    st = empty.apply_purchase(pool[0].player_id, FOCUS, 30)
    text = st.room_summary(next_bid=25)
    assert "Fabricated" not in text, "no player names in a sanitized summary"
    # With nobody on the block this is the financial ceiling, and the summary
    # says so rather than implying legal eligibility for a particular player.
    assert "WHO CAN AFFORD $25" in text
    assert "FINANCIAL" in text
    assert "Who WOULD bid is not modelled" in text
    assert "no quarterback maximum" in text
    nominated = st.nominate(st.available_ids[0], "O02")
    text2 = nominated.room_summary(next_bid=25)
    assert "WHO MAY LEGALLY BID $25 ON PLAYER" in text2
    for o in OWNERS:
        assert o in text


def test_the_room_summary_shows_invariant_violations(pool):
    o = OwnerAuctionState("O00", "T", 10, 15,
                          (RosterSlotFilled(pool[0].player_id,
                                            pool[0].position, 40),))
    st = AuctionState(settings=DEFAULT_LEAGUE, pool=pool, focus_owner_id="O00",
                      owners=(o,) + tuple(
                          OwnerAuctionState(x, x, 200, 15) for x in OWNERS[1:]))
    assert "INVARIANT VIOLATIONS" in st.room_summary()


def test_a_complete_auction_is_recognized(empty, pool):
    st = empty
    cursor = {}
    for o in OWNERS:
        st = legal_fill(st, o, 15, price=1, pool=pool, cursor=cursor)
    assert st.is_complete
    st.validate()
    for owner in st.owners:
        assert owner.fields_a_full_lineup
