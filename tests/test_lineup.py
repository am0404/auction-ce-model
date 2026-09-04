"""Exact legal lineup selection."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from ceauction.league import SLOT_ELIGIBILITY, SLOT_ORDER, Position, Slot
from ceauction.lineup import Lineup, is_feasible, select_lineup
from ceauction.lineup_vec import select_lineups_mask
from ceauction.pregame import Availability, PregameEntry, PregameWeek

from helpers import entry, week_of

QB, RB, WR, TE = Position.QB, Position.RB, Position.WR, Position.TE
SLOTS = list(SLOT_ORDER)


# --------------------------------------------------------------------------
# An independent reference implementation, used to prove the greedy is exact.
# --------------------------------------------------------------------------


def _assignable(positions):
    """Augmenting-path bipartite matching; shares no code with the solver."""
    slot_of = [-1] * len(SLOTS)

    def try_assign(p, seen):
        for si, s in enumerate(SLOTS):
            if positions[p] in SLOT_ELIGIBILITY[s] and not seen[si]:
                seen[si] = True
                if slot_of[si] == -1 or try_assign(slot_of[si], seen):
                    slot_of[si] = p
                    return True
        return False

    return all(try_assign(p, [False] * len(SLOTS)) for p in range(len(positions)))


def _brute_force_best(entries):
    idxs = [i for i, e in enumerate(entries) if e.startable]
    for size in range(min(8, len(idxs)), -1, -1):
        best = None
        for combo in itertools.combinations(idxs, size):
            if _assignable([entries[i].position for i in combo]):
                val = sum(entries[i].projection for i in combo)
                if best is None or val > best[0]:
                    best = (val, combo)
        if best is not None:
            return best
    return (0.0, ())


def _random_entries(rs):
    positions = [Position(int(x)) for x in rs.integers(0, 4, 15)]
    projs = np.round(rs.uniform(0.0, 25.0, 15), 2)
    avail = rs.random(15) > 0.25
    return tuple(
        entry(i, positions[i], float(projs[i]),
              Availability.ACTIVE if avail[i] else Availability.INJURED)
        for i in range(15)
    )


# --------------------------------------------------------------------------


def test_hall_condition_bounds():
    assert is_feasible(1, 4, 3)          # QB + RB1/RB2/FLEX/SFLEX + WT1-3
    assert is_feasible(2, 3, 3)          # two QBs, eight slots exactly filled
    assert is_feasible(0, 0, 0)          # the empty lineup is trivially legal
    assert not is_feasible(3, 0, 0)      # only two QB-capable slots
    assert not is_feasible(0, 5, 0)      # only four RB-capable slots
    assert not is_feasible(0, 0, 6)      # only five WR/TE-capable slots
    assert not is_feasible(2, 4, 0)      # q + r <= 5: QB/RB1/RB2/FLEX/SFLEX
    assert not is_feasible(2, 0, 5)      # q + t <= 6
    assert not is_feasible(0, 4, 4)      # r + t <= 7
    assert not is_feasible(1, 4, 4)      # nine players, eight slots


def test_full_legal_lineup_is_filled():
    entries = [entry(0, QB, 20)] + [entry(i + 1, RB, 15 - i) for i in range(4)] \
        + [entry(i + 5, WR, 12 - i) for i in range(6)] \
        + [entry(i + 11, TE, 8 - i) for i in range(2)] \
        + [entry(13, QB, 14), entry(14, QB, 5)]
    lu = select_lineup(week_of(entries))
    assert lu.filled_slots == 8
    assert {c.slot for c in lu.choices} == set(SLOTS)
    for c in lu.choices:
        assert c.position in SLOT_ELIGIBILITY[c.slot]


def test_two_rb_requirement_is_enforced():
    entries = [entry(0, QB, 25)] + [entry(i + 1, WR, 20 - i) for i in range(12)] \
        + [entry(13, RB, 1.0), entry(14, RB, 0.5)]
    lu = select_lineup(week_of(entries))
    started = {c.player_id: c for c in lu.choices}
    # Both weak RBs start, because RB1/RB2 accept nothing else.
    assert 13 in started and 14 in started
    assert started[13].slot in (Slot.RB1, Slot.RB2)
    assert started[14].slot in (Slot.RB1, Slot.RB2)


def test_three_wr_te_requirement_is_enforced():
    entries = [entry(0, QB, 25), entry(1, QB, 24)] \
        + [entry(i + 2, RB, 20 - i) for i in range(8)] \
        + [entry(10, WR, 1.0), entry(11, TE, 0.9), entry(12, WR, 0.8)] \
        + [entry(13, RB, 0.2), entry(14, RB, 0.1)]
    lu = select_lineup(week_of(entries))
    wt_slots = [c for c in lu.choices if c.slot in (Slot.WT1, Slot.WT2, Slot.WT3)]
    assert len(wt_slots) == 3
    assert all(c.position in (WR, TE) for c in wt_slots)


def test_non_qb_may_fill_superflex():
    """The superflex does not require a second QB."""
    entries = [entry(0, QB, 20)] + [entry(i + 1, RB, 18 - i) for i in range(4)] \
        + [entry(i + 5, WR, 17 - i) for i in range(6)] \
        + [entry(11, TE, 10), entry(12, TE, 9)] \
        + [entry(13, QB, 0.1), entry(14, QB, 0.05)]
    lu = select_lineup(week_of(entries))
    sflex = next(c for c in lu.choices if c.slot is Slot.SUPERFLEX)
    assert sflex.position is not QB
    assert "non-QB in superflex" in sflex.reason
    assert 13 not in lu.started_ids and 14 not in lu.started_ids


def test_second_qb_used_in_superflex_when_best():
    entries = [entry(0, QB, 25), entry(1, QB, 24)] \
        + [entry(i + 2, RB, 6 - i * 0.1) for i in range(4)] \
        + [entry(i + 6, WR, 5 - i * 0.1) for i in range(7)] \
        + [entry(13, TE, 3), entry(14, TE, 2)]
    lu = select_lineup(week_of(entries))
    sflex = next(c for c in lu.choices if c.slot is Slot.SUPERFLEX)
    assert sflex.position is QB
    assert sflex.player_id == 1


def test_unavailable_players_never_start():
    entries = [entry(0, QB, 99, Availability.BYE)] + [entry(1, QB, 10)] \
        + [entry(i + 2, RB, 30, Availability.INJURED) for i in range(2)] \
        + [entry(i + 4, RB, 5) for i in range(2)] \
        + [entry(i + 6, WR, 8) for i in range(7)] \
        + [entry(13, TE, 4), entry(14, TE, 3)]
    lu = select_lineup(week_of(entries))
    assert 0 not in lu.started_ids
    assert 2 not in lu.started_ids and 3 not in lu.started_ids
    reasons = {b.player_id: b.reason for b in lu.bench}
    assert reasons[0] == "on bye"
    assert reasons[2] == "injured / unavailable"


def test_slots_go_unfilled_when_legally_impossible():
    """One RB on the roster means RB2 cannot be filled -- that is legal."""
    entries = [entry(0, QB, 20), entry(1, RB, 10)] \
        + [entry(i + 2, WR, 9 - i * 0.1) for i in range(11)] \
        + [entry(13, TE, 3), entry(14, TE, 2)]
    lu = select_lineup(week_of(entries))
    assert lu.filled_slots == 7
    assert Slot.RB2 not in {c.slot for c in lu.choices}
    # Six WR/TE cannot all play: WT1-3 + FLEX + SUPERFLEX is five.
    assert sum(1 for c in lu.choices if c.position in (WR, TE)) == 5


def test_greedy_matches_brute_force_on_random_inputs():
    rs = np.random.default_rng(0)
    for _ in range(60):
        entries = _random_entries(rs)
        lu = select_lineup(week_of(entries))
        best_val, best_set = _brute_force_best(entries)
        assert lu.projected_points == pytest.approx(best_val)
        assert lu.filled_slots == len(best_set)


def test_vectorised_matches_scalar():
    rs = np.random.default_rng(1)
    for _ in range(60):
        entries = _random_entries(rs)
        lu = select_lineup(week_of(entries))
        proj = np.array([[e.projection for e in entries]])
        avail = np.array([[e.startable for e in entries]])
        pos = np.array([[int(e.position) for e in entries]])
        mask = select_lineups_mask(proj, avail, pos)
        assert set(np.flatnonzero(mask[0]).tolist()) == set(lu.started_ids)


def test_raising_a_projection_never_lowers_the_chosen_lineup_total():
    """Monotonicity: better information cannot produce a worse decision."""
    rs = np.random.default_rng(2)
    for _ in range(150):
        entries = list(_random_entries(rs))
        before = select_lineup(week_of(entries)).projected_points
        i = int(rs.integers(0, 15))
        bump = float(rs.uniform(0.01, 12.0))
        e = entries[i]
        entries[i] = PregameEntry(
            e.player_id, e.name, e.position, e.projection + bump, e.availability
        )
        after = select_lineup(week_of(entries)).projected_points
        assert after >= before - 1e-9


def test_every_selected_player_has_a_reason_and_bench_players_do_too():
    rs = np.random.default_rng(3)
    entries = _random_entries(rs)
    lu = select_lineup(week_of(entries))
    assert all(c.reason for c in lu.choices)
    assert all(b.reason for b in lu.bench)
    assert len(lu.choices) + len(lu.bench) == 15
    assert set(lu.started_ids).isdisjoint({b.player_id for b in lu.bench})


def test_ties_are_broken_deterministically():
    entries = [entry(i, WR, 10.0) for i in range(15)]
    a = select_lineup(week_of(entries)).started_ids
    b = select_lineup(week_of(entries)).started_ids
    assert a == b
