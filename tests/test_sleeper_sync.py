"""Hostile tests for the read-only Sleeper sale sync.

Every response here is mocked. Nothing in this file opens a socket, and the
one test that asserts on request counts exists precisely to prove that a
disabled sync makes no request at all.

The tests are written from the failure side: the interesting question is never
"does a clean pick apply?" but "what does this refuse, and does refusing leave
the room untouched?"
"""

from __future__ import annotations

import json

import pytest

from ceauction.draftday.session import DraftSession
from ceauction.draftday.sleepersync import (
    NAME_ALIASES,
    STATUS_CONNECTED,
    STATUS_DISCONNECTED,
    STATUS_NEEDS_ATTENTION,
    STATUS_OFF,
    STATUS_PRE_DRAFT,
    OwnerMap,
    PickRefusal,
    PlayerIndex,
    SleeperClient,
    SleeperError,
    SleeperSync,
    board_key_for_name,
    parse_pick,
)
from ceauction.tactical.realpilot import OWNER_IDS

from test_draftday import _make_board  # the same synthetic board


N = len(OWNER_IDS)


# ---------------------------------------------------------------------------
# A fake Sleeper, shaped exactly like the real feed observed on draft day
# ---------------------------------------------------------------------------


def _draft_doc(status="pre_draft", teams=N, budget=200):
    slot_to_roster = {str(s): s for s in range(1, teams + 1)}
    draft_order = {f"user{r}": r for r in range(1, teams + 1)}
    return {
        "draft_id": "1386497531533357056",
        "league_id": "1386497530312814592",
        "status": status, "type": "auction", "season": "2026",
        "metadata": {"name": "FF 2026"},
        "settings": {"teams": teams, "budget": budget, "rounds": 15},
        "slot_to_roster_id": slot_to_roster,
        "draft_order": draft_order,
    }


def _rosters(teams=N):
    # roster r is owned by userN where slot_to_roster maps slot r -> roster r
    return [{"roster_id": r, "owner_id": f"user{r}"} for r in range(1, teams + 1)]


def _users(teams=N):
    return [{"user_id": f"user{r}", "display_name": f"owner{r}",
             "metadata": {"team_name": f"Team Name {r}"}}
            for r in range(1, teams + 1)]


def _pick(pick_no, sleeper_pid, roster_id, amount):
    return {"pick_no": pick_no, "player_id": str(sleeper_pid),
            "roster_id": roster_id, "round": 1, "draft_slot": roster_id,
            "metadata": {"amount": str(amount)}}


class FakeSleeper:
    """Serves canned responses and counts every request it is asked for."""

    def __init__(self, board, draft=None):
        self.draft_doc = draft or _draft_doc()
        self.rosters = _rosters()
        self.users = _users()
        self.picks = []
        self.fail_with = None
        self.calls = []
        # One Sleeper player per board player, ids in a distinct namespace so a
        # test can never accidentally pass by using a board id as a Sleeper id.
        self.players = {}
        self.sleeper_id_by_board = {}
        for i, (bid, name) in enumerate(sorted(board.name_by_id.items())):
            sid = str(90000 + i)
            self.players[sid] = {"player_id": sid, "full_name": name,
                                 "position": "WR", "active": True}
            self.sleeper_id_by_board[bid] = sid

    def fetch(self, url):
        self.calls.append(url)
        if self.fail_with is not None:
            raise self.fail_with
        if url.endswith("/picks"):
            return list(self.picks)
        if "/draft/" in url:
            return self.draft_doc
        if url.endswith("/rosters"):
            return list(self.rosters)
        if url.endswith("/users"):
            return list(self.users)
        raise AssertionError(f"unexpected url {url}")


@pytest.fixture(scope="module")
def board():
    return _make_board()


@pytest.fixture
def session(board, tmp_path):
    return DraftSession(board, state_path=tmp_path / "draft_state.json",
                        overrides_path=tmp_path / "manual_overrides.json")


@pytest.fixture
def fake(board):
    return FakeSleeper(board)


@pytest.fixture
def sync(session, fake, tmp_path):
    client = SleeperClient("1386497531533357056", fetch=fake.fetch)
    s = SleeperSync(session, client, owner_ids=OWNER_IDS,
                    state_path=tmp_path / "sleeper_sync.json")
    s.connect(sleeper_players=fake.players)
    return s


def _sid(fake, board, position_name="WR", index=0):
    """A Sleeper id for some board player, chosen deterministically."""
    bid = sorted(board.name_by_id)[index]
    return bid, fake.sleeper_id_by_board[bid]


# ---------------------------------------------------------------------------
# Off means off
# ---------------------------------------------------------------------------


def test_sync_off_makes_zero_network_requests(session, fake, tmp_path):
    client = SleeperClient("d", fetch=fake.fetch)
    s = SleeperSync(session, client, owner_ids=OWNER_IDS,
                    state_path=tmp_path / "s.json")
    before = len(fake.calls)
    for _ in range(5):
        out = s.poll_once()
    assert out["status"] == STATUS_OFF
    assert out["enabled"] is False
    assert len(fake.calls) == before, "a disabled sync must not call Sleeper"
    assert client.n_requests == 0


def test_sync_defaults_to_off(sync):
    assert sync.enabled is False
    assert sync.status == STATUS_OFF


# ---------------------------------------------------------------------------
# The empty pre-draft room
# ---------------------------------------------------------------------------


def test_empty_pre_draft_response_changes_nothing(sync, session):
    fingerprint = session.fingerprint()
    sync.enable()
    out = sync.poll_once()
    assert out["status"] == STATUS_PRE_DRAFT
    assert out["n_sleeper_completed"] == 0
    assert out["n_local_sales"] == 0
    assert session.fingerprint() == fingerprint
    assert out["reconcile"]["ok"] is True


# ---------------------------------------------------------------------------
# The happy path, and its idempotence
# ---------------------------------------------------------------------------


def test_one_valid_auction_pick_applies(sync, session, fake, board):
    bid, sid = _sid(fake, board, index=0)
    fake.picks = [_pick(1, sid, 3, 17)]
    fake.draft_doc["status"] = "drafting"
    sync.enable()
    out = sync.poll_once()

    assert out["status"] == STATUS_CONNECTED
    assert len(session.sales) == 1
    sale = session.sales[0]
    assert sale.player_id == bid
    assert sale.owner_id == OWNER_IDS[2]      # roster 3 -> Team03
    assert sale.price == 17
    assert out["reconcile"]["ok"] is True


def test_duplicate_poll_is_idempotent(sync, session, fake, board):
    bid, sid = _sid(fake, board, index=0)
    fake.picks = [_pick(1, sid, 3, 17)]
    sync.enable()
    sync.poll_once()
    fingerprint = session.fingerprint()
    for _ in range(4):
        sync.poll_once()
    assert len(session.sales) == 1
    assert session.fingerprint() == fingerprint


def test_multiple_picks_arriving_together(sync, session, fake, board):
    ids = sorted(board.name_by_id)[:3]
    fake.picks = [_pick(i + 1, fake.sleeper_id_by_board[b], i + 1, 5 + i)
                  for i, b in enumerate(ids)]
    sync.enable()
    sync.poll_once()
    assert len(session.sales) == 3
    assert [s.player_id for s in session.sales] == ids
    assert [s.owner_id for s in session.sales] == list(OWNER_IDS[:3])
    assert [s.price for s in session.sales] == [5, 6, 7]


def test_picks_out_of_order_are_applied_in_draft_order(sync, session, fake, board):
    ids = sorted(board.name_by_id)[:3]
    picks = [_pick(i + 1, fake.sleeper_id_by_board[b], i + 1, 5 + i)
             for i, b in enumerate(ids)]
    fake.picks = [picks[2], picks[0], picks[1]]   # shuffled, as on a reconnect
    sync.enable()
    sync.poll_once()
    assert [s.player_id for s in session.sales] == ids, "applied in pick_no order"
    assert [s.price for s in session.sales] == [5, 6, 7]


def test_missed_picks_after_interruption_are_caught_up_in_order(
        sync, session, fake, board):
    ids = sorted(board.name_by_id)[:4]
    sync.enable()
    fake.picks = [_pick(1, fake.sleeper_id_by_board[ids[0]], 1, 9)]
    sync.poll_once()
    assert len(session.sales) == 1

    # The connection drops for a while and three more picks complete.
    fake.fail_with = SleeperError("connection reset")
    sync.poll_once()
    assert len(session.sales) == 1

    fake.fail_with = None
    fake.picks += [_pick(i + 2, fake.sleeper_id_by_board[b], i + 2, 10 + i)
                   for i, b in enumerate(ids[1:])]
    sync.poll_once()
    assert [s.player_id for s in session.sales] == ids
    assert sync.status == STATUS_CONNECTED


# ---------------------------------------------------------------------------
# Restart
# ---------------------------------------------------------------------------


def test_restart_and_repoll_does_not_duplicate(board, tmp_path, fake):
    state_path = tmp_path / "draft_state.json"
    ledger = tmp_path / "sleeper_sync.json"
    bid = sorted(board.name_by_id)[0]
    fake.picks = [_pick(1, fake.sleeper_id_by_board[bid], 4, 22)]

    s1 = DraftSession(board, state_path=state_path,
                      overrides_path=tmp_path / "o.json")
    sync1 = SleeperSync(s1, SleeperClient("d", fetch=fake.fetch),
                        owner_ids=OWNER_IDS, state_path=ledger)
    sync1.connect(sleeper_players=fake.players)
    sync1.enable()
    sync1.poll_once()
    assert len(s1.sales) == 1
    fingerprint = s1.fingerprint()

    # Restart: a fresh session replays the saved room, a fresh sync re-reads
    # the ledger, and the same feed must produce no second sale.
    s2 = DraftSession(board, state_path=state_path,
                      overrides_path=tmp_path / "o.json")
    assert s2.restore() is True
    sync2 = SleeperSync(s2, SleeperClient("d", fetch=fake.fetch),
                        owner_ids=OWNER_IDS, state_path=ledger)
    sync2.connect(sleeper_players=fake.players)
    sync2.enable()
    sync2.poll_once()

    assert len(s2.sales) == 1
    assert s2.fingerprint() == fingerprint
    assert sync2.reconcile_report.ok is True


def test_ledger_from_a_different_draft_is_not_trusted(session, fake, tmp_path):
    ledger = tmp_path / "sleeper_sync.json"
    ledger.write_text(json.dumps(
        {"version": 1, "draft_id": "SOME-OTHER-DRAFT",
         "applied": {"1:90000": {"pick_no": 1}}}), encoding="utf-8")
    s = SleeperSync(session, SleeperClient("d", fetch=fake.fetch),
                    owner_ids=OWNER_IDS, state_path=ledger)
    assert s.applied == {}, "another draft's ledger must not authorise this one"


# ---------------------------------------------------------------------------
# The network is not a source of truth about local state
# ---------------------------------------------------------------------------


def test_network_timeout_changes_no_state(sync, session, fake):
    sync.enable()
    fingerprint = session.fingerprint()
    fake.fail_with = SleeperError("timed out")
    out = sync.poll_once()
    assert out["status"] == STATUS_DISCONNECTED
    assert "timed out" in out["last_error"]
    assert session.fingerprint() == fingerprint
    assert len(session.sales) == 0
    assert sync.applied == {}


def test_disconnect_then_automatic_retry_recovers(sync, session, fake, board):
    sync.enable()
    fake.fail_with = SleeperError("no route to host")
    assert sync.poll_once()["status"] == STATUS_DISCONNECTED
    fake.fail_with = None
    bid = sorted(board.name_by_id)[0]
    fake.picks = [_pick(1, fake.sleeper_id_by_board[bid], 2, 8)]
    out = sync.poll_once()
    assert out["status"] == STATUS_CONNECTED
    assert len(session.sales) == 1


# ---------------------------------------------------------------------------
# Refusals: never guess a price, a player or an owner
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("amount", [
    None, "", "  ", "abc", "12.5", 12.5, "$14", "-3", "0", 0, -3, True, {}, [],
])
def test_malformed_or_missing_amount_is_refused(sync, session, fake, board,
                                                amount):
    bid, sid = _sid(fake, board, index=0)
    pick = _pick(1, sid, 3, 0)
    pick["metadata"] = {} if amount is None else {"amount": amount}
    fake.picks = [pick]
    sync.enable()
    out = sync.poll_once()

    assert out["status"] == STATUS_NEEDS_ATTENTION
    assert len(session.sales) == 0, "no sale may be posted at a guessed price"
    assert any("amount" in a for a in out["attention"])


def test_metadata_absent_entirely_is_refused(sync, session, fake, board):
    bid, sid = _sid(fake, board, index=0)
    pick = _pick(1, sid, 3, 5)
    del pick["metadata"]
    fake.picks = [pick]
    sync.enable()
    out = sync.poll_once()
    assert out["status"] == STATUS_NEEDS_ATTENTION
    assert len(session.sales) == 0


def test_unknown_player_is_refused(sync, session, fake):
    fake.picks = [_pick(1, "does-not-exist", 3, 12)]
    sync.enable()
    out = sync.poll_once()
    assert out["status"] == STATUS_NEEDS_ATTENTION
    assert len(session.sales) == 0
    assert any("player index" in a for a in out["attention"])


def test_player_not_on_this_board_is_refused(sync, session, fake):
    # A real Sleeper player who is simply not in our 549-man pool.
    fake.players["99999"] = {"player_id": "99999", "full_name": "Deep Bench Guy",
                             "position": "WR", "active": True}
    sync.index = PlayerIndex.from_board(sync.session.board, fake.players)
    fake.picks = [_pick(1, "99999", 3, 2)]
    sync.enable()
    out = sync.poll_once()
    assert out["status"] == STATUS_NEEDS_ATTENTION
    assert len(session.sales) == 0
    assert any("not on this board" in a for a in out["attention"])


def test_unknown_roster_is_refused(sync, session, fake, board):
    bid, sid = _sid(fake, board, index=0)
    fake.picks = [_pick(1, sid, 99, 10)]
    sync.enable()
    out = sync.poll_once()
    assert out["status"] == STATUS_NEEDS_ATTENTION
    assert len(session.sales) == 0
    assert any("owner map" in a for a in out["attention"])


def test_pick_with_no_roster_is_refused(sync, session, fake, board):
    bid, sid = _sid(fake, board, index=0)
    pick = _pick(1, sid, 3, 10)
    del pick["roster_id"]
    fake.picks = [pick]
    sync.enable()
    out = sync.poll_once()
    assert out["status"] == STATUS_NEEDS_ATTENTION
    assert len(session.sales) == 0
    assert any("no roster_id" in a for a in out["attention"])


def test_nomination_in_flight_is_skipped_not_refused(sync, session, fake):
    fake.picks = [{"pick_no": 1, "player_id": None, "roster_id": 3,
                   "metadata": {}}]
    sync.enable()
    out = sync.poll_once()
    assert out["n_sleeper_completed"] == 0
    assert out["status"] == STATUS_PRE_DRAFT
    assert len(session.sales) == 0
    assert out["attention"] == []


# ---------------------------------------------------------------------------
# Ambiguous or inconsistent owner mapping refuses to exist
# ---------------------------------------------------------------------------


def test_owner_map_refuses_wrong_team_count(fake):
    doc = _draft_doc(teams=N)
    doc["settings"]["teams"] = 10
    with pytest.raises(SleeperError, match="teams"):
        OwnerMap.build(doc, _rosters(), _users(), OWNER_IDS)


def test_owner_map_refuses_non_bijective_slots(fake):
    doc = _draft_doc()
    doc["slot_to_roster_id"]["1"] = 2         # roster 2 now in two slots
    with pytest.raises(SleeperError, match="bijection"):
        OwnerMap.build(doc, _rosters(), _users(), OWNER_IDS)


def test_owner_map_refuses_when_draft_order_disagrees(fake):
    doc = _draft_doc()
    doc["draft_order"]["user1"] = 2
    doc["draft_order"]["user2"] = 1
    with pytest.raises(SleeperError, match="disagree"):
        OwnerMap.build(doc, _rosters(), _users(), OWNER_IDS)


def test_owner_map_refuses_gappy_roster_ids():
    rosters = _rosters()
    rosters[0]["roster_id"] = 99
    with pytest.raises(SleeperError, match="roster ids"):
        OwnerMap.build(_draft_doc(), rosters, _users(), OWNER_IDS)


def test_owner_map_is_positional_and_imports_real_names():
    m = OwnerMap.build(_draft_doc(), _rosters(), _users(), OWNER_IDS)
    assert m.owner_for(1) == "Team01"
    assert m.owner_for(12) == "Team12"
    assert m.team_names["Team01"] == "Team Name 1"
    assert m.display_names["Team07"] == "owner7"


def test_owner_map_leaves_a_blank_team_name_blank():
    users = _users()
    users[0]["metadata"] = {"team_name": "   "}
    users[0]["display_name"] = "  "
    m = OwnerMap.build(_draft_doc(), _rosters(), users, OWNER_IDS)
    assert "Team01" not in m.team_names, "a blank name is not invented"


def test_non_auction_draft_is_refused(session, fake, tmp_path):
    fake.draft_doc["type"] = "snake"
    s = SleeperSync(session, SleeperClient("d", fetch=fake.fetch),
                    owner_ids=OWNER_IDS, state_path=tmp_path / "s.json")
    with pytest.raises(SleeperError, match="auction"):
        s.connect(sleeper_players=fake.players)


# ---------------------------------------------------------------------------
# Conflict with a manual sale
# ---------------------------------------------------------------------------


def test_manual_conflict_is_surfaced_and_never_overwritten(
        sync, session, fake, board):
    bid, sid = _sid(fake, board, index=0)
    session.record_sale(bid, OWNER_IDS[5], 40)      # the user typed it first
    fake.picks = [_pick(1, sid, 3, 17)]             # Sleeper disagrees
    sync.enable()
    out = sync.poll_once()

    assert out["status"] == STATUS_NEEDS_ATTENTION
    assert len(session.sales) == 1
    assert session.sales[0].owner_id == OWNER_IDS[5]
    assert session.sales[0].price == 40, "the manual sale stands untouched"
    assert any("already records" in a for a in out["attention"])


def test_manual_sale_that_agrees_is_adopted_not_duplicated(
        sync, session, fake, board):
    bid, sid = _sid(fake, board, index=0)
    session.record_sale(bid, OWNER_IDS[2], 17)
    fake.picks = [_pick(1, sid, 3, 17)]
    sync.enable()
    out = sync.poll_once()

    assert out["status"] == STATUS_CONNECTED
    assert len(session.sales) == 1
    assert out["n_sleeper_applied"] == 1
    assert out["reconcile"]["ok"] is True


def test_manual_controls_stay_usable_while_attention_is_raised(
        sync, session, fake, board):
    bid, sid = _sid(fake, board, index=0)
    session.record_sale(bid, OWNER_IDS[5], 40)
    fake.picks = [_pick(1, sid, 3, 17)]
    sync.enable()
    sync.poll_once()
    assert sync.status == STATUS_NEEDS_ATTENTION

    # The fallback must still work while sync is complaining.
    other = sorted(board.name_by_id)[1]
    session.record_sale(other, OWNER_IDS[1], 6)
    assert len(session.sales) == 2
    assert session.undo() is not None
    assert len(session.sales) == 1


def test_attention_stops_further_automatic_application(
        sync, session, fake, board):
    ids = sorted(board.name_by_id)[:2]
    session.record_sale(ids[0], OWNER_IDS[5], 40)
    fake.picks = [_pick(1, fake.sleeper_id_by_board[ids[0]], 3, 17)]
    sync.enable()
    sync.poll_once()
    assert sync.status == STATUS_NEEDS_ATTENTION

    # A perfectly good second pick must NOT be applied while unresolved.
    fake.picks.append(_pick(2, fake.sleeper_id_by_board[ids[1]], 4, 9))
    sync.poll_once()
    assert len(session.sales) == 1, "application is stopped, not merely noisy"
    assert sync.status == STATUS_NEEDS_ATTENTION


# ---------------------------------------------------------------------------
# The batch snapshot restores exactly
# ---------------------------------------------------------------------------


def test_snapshot_restores_exactly_when_a_batch_fails(
        sync, session, fake, board):
    ids = sorted(board.name_by_id)[:3]
    # A pre-existing manual sale that the failing batch must not disturb.
    pre = sorted(board.name_by_id)[10]
    session.record_sale(pre, OWNER_IDS[0], 12)
    # Two good picks then one that conflicts: the whole batch must roll back.
    session.record_sale(ids[2], OWNER_IDS[7], 33)
    fingerprint = session.fingerprint()
    n_before = len(session.sales)

    fake.picks = [
        _pick(1, fake.sleeper_id_by_board[ids[0]], 1, 4),
        _pick(2, fake.sleeper_id_by_board[ids[1]], 2, 5),
        _pick(3, fake.sleeper_id_by_board[ids[2]], 3, 99),   # conflicts
    ]
    sync.enable()
    out = sync.poll_once()

    assert out["status"] == STATUS_NEEDS_ATTENTION
    assert len(session.sales) == n_before, "the batch is all-or-nothing"
    assert session.fingerprint() == fingerprint, "restored exactly"
    assert sync.applied == {}, "no pick may be marked applied after a rollback"


def test_rollback_leaves_no_partial_ledger(sync, session, fake, board):
    ids = sorted(board.name_by_id)[:2]
    session.record_sale(ids[1], OWNER_IDS[7], 33)
    fake.picks = [
        _pick(1, fake.sleeper_id_by_board[ids[0]], 1, 4),
        _pick(2, fake.sleeper_id_by_board[ids[1]], 3, 99),   # conflicts
    ]
    sync.enable()
    sync.poll_once()
    # Re-reading the ledger from disk must not resurrect the rolled-back pick.
    s2 = SleeperSync(session, SleeperClient("d", fetch=fake.fetch),
                     owner_ids=OWNER_IDS, state_path=sync.state_path)
    assert s2.applied == {}


# ---------------------------------------------------------------------------
# Reconciliation
# ---------------------------------------------------------------------------


def test_reconciles_budget_roster_availability_and_ownership(
        sync, session, fake, board):
    ids = sorted(board.name_by_id)[:6]
    fake.picks = [_pick(i + 1, fake.sleeper_id_by_board[b], (i % N) + 1, 3 + i)
                  for i, b in enumerate(ids)]
    sync.enable()
    out = sync.poll_once()

    assert out["reconcile"]["ok"] is True
    assert out["reconcile"]["problems"] == []
    assert out["n_sleeper_completed"] == 6
    assert out["n_local_sales"] == 6

    state = session.state
    for owner in state.owners:
        assert owner.spent + owner.budget_remaining == owner.budget_start
        assert owner.budget_remaining >= 0
        assert owner.n_players <= owner.roster_capacity
        # A legal maximum that respects the $1-per-open-slot reserve.
        if owner.open_slots > 0:
            assert owner.max_bid == owner.budget_remaining - (owner.open_slots - 1)


def test_no_duplicate_ownership_anywhere(sync, session, fake, board):
    ids = sorted(board.name_by_id)[:8]
    fake.picks = [_pick(i + 1, fake.sleeper_id_by_board[b], (i % N) + 1, 2)
                  for i, b in enumerate(ids)]
    sync.enable()
    sync.poll_once()
    seen = set()
    for owner in session.state.owners:
        for pid in owner.player_ids:
            assert pid not in seen, f"player {pid} owned twice"
            seen.add(pid)
    assert len(seen) == 8


def test_price_over_budget_is_refused_by_the_rules_not_applied(
        sync, session, fake, board):
    bid, sid = _sid(fake, board, index=0)
    fake.picks = [_pick(1, sid, 3, 5000)]
    sync.enable()
    out = sync.poll_once()
    assert out["status"] == STATUS_NEEDS_ATTENTION
    assert len(session.sales) == 0
    assert any("auction rules" in a for a in out["attention"])


def test_reconcile_reports_counts_every_poll(sync, session, fake, board):
    sync.enable()
    out = sync.poll_once()
    assert out["reconcile"]["n_sleeper_completed"] == 0
    bid, sid = _sid(fake, board, index=0)
    fake.picks = [_pick(1, sid, 3, 7)]
    out = sync.poll_once()
    assert out["reconcile"]["n_sleeper_completed"] == 1
    assert out["reconcile"]["n_sleeper_applied"] == 1
    assert out["reconcile"]["n_local_sales"] == 1


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


def test_board_key_transform_is_exact_not_fuzzy():
    assert board_key_for_name("A.J. Brown Jr.") == "aj_brown"
    assert board_key_for_name("Josh Allen") == "josh_allen"
    assert board_key_for_name("") == ""
    assert board_key_for_name(None) == ""


def test_declared_aliases_are_applied():
    # The one hand-verified alias on the real board.
    assert "marquise_brown" in NAME_ALIASES
    assert board_key_for_name("Marquise Brown") == "hollywood_brown"


def test_player_index_joins_only_in_the_pick_direction(board):
    # Two Sleeper skill players share a name; only the id in the pick decides
    # which one was sold, so a shared name is never ambiguous here.
    name = board.name_by_id[sorted(board.name_by_id)[0]]
    players = {"1": {"player_id": "1", "full_name": name, "position": "WR"},
               "2": {"player_id": "2", "full_name": name, "position": "TE"}}
    index = PlayerIndex.from_board(board, players)
    assert index.board_id_of("1") == index.board_id_of("2")
    assert index.board_id_of("nope") is None


def test_a_defender_sharing_a_name_never_resolves_to_the_skill_player(board):
    """Sleeper carries a linebacker named Justin Jefferson. He is not ours."""
    name = board.name_by_id[sorted(board.name_by_id)[0]]
    players = {
        "1": {"player_id": "1", "full_name": name, "position": "WR",
              "fantasy_positions": ["WR"]},
        "2": {"player_id": "2", "full_name": name, "position": "LB",
              "fantasy_positions": ["LB"]},
        "3": {"player_id": "3", "full_name": name, "position": "G"},
    }
    index = PlayerIndex.from_board(board, players)
    assert index.board_id_of("1") is not None
    assert index.board_id_of("2") is None, "a linebacker is not a receiver"
    assert index.board_id_of("3") is None


def test_a_two_way_player_still_joins_through_fantasy_positions(board):
    """Travis Hunter is filed DB with WR alongside; he must still resolve."""
    name = board.name_by_id[sorted(board.name_by_id)[0]]
    players = {"1": {"player_id": "1", "full_name": name, "position": "DB",
                     "fantasy_positions": ["DB", "WR"]}}
    index = PlayerIndex.from_board(board, players)
    assert index.board_id_of("1") is not None


def test_parse_pick_rejects_a_non_object():
    m = OwnerMap.build(_draft_doc(), _rosters(), _users(), OWNER_IDS)
    out = parse_pick(["not", "a", "dict"], PlayerIndex({}, {}), m)
    assert isinstance(out, PickRefusal)


# ---------------------------------------------------------------------------
# The persisted ledger, on its own
# ---------------------------------------------------------------------------


def test_ledger_is_written_with_the_pick_identifier(sync, session, fake, board):
    bid, sid = _sid(fake, board, index=0)
    fake.picks = [_pick(7, sid, 3, 17)]
    sync.enable()
    sync.poll_once()

    blob = json.loads(sync.state_path.read_text(encoding="utf-8"))
    assert blob["draft_id"] == sync.client.draft_id
    key = f"7:{sid}"
    assert key in blob["applied"], "the Sleeper pick identifier must persist"
    assert blob["applied"][key]["board_player_id"] == bid
    assert blob["applied"][key]["amount"] == 17
    assert blob["applied"][key]["owner_id"] == OWNER_IDS[2]


def test_ledger_alone_suppresses_reapplication(sync, session, fake, board):
    """The ledger must gate application on its own.

    Ownership adoption is a second line of defence, so this test removes it
    from the picture: the sale is undone locally while the ledger still records
    the pick. A poller that ignored its ledger would silently re-buy the player;
    the correct behaviour is to leave the room alone and report the divergence.
    """
    bid, sid = _sid(fake, board, index=0)
    fake.picks = [_pick(1, sid, 3, 17)]
    sync.enable()
    sync.poll_once()
    assert len(session.sales) == 1

    session.undo()                      # the operator removes it by hand
    assert len(session.sales) == 0

    out = sync.poll_once()
    assert len(session.sales) == 0, "an applied pick is never re-applied"
    assert out["status"] == STATUS_NEEDS_ATTENTION
    assert any("not sold on this board" in a for a in out["attention"])


def test_forget_clears_the_ledger_for_a_room_reset(sync, session, fake, board):
    bid, sid = _sid(fake, board, index=0)
    fake.picks = [_pick(1, sid, 3, 17)]
    sync.enable()
    sync.poll_once()
    assert sync.applied

    session.reset()
    sync.forget()
    assert sync.applied == {}
    assert json.loads(sync.state_path.read_text(encoding="utf-8"))["applied"] == {}

    # After the reset the same feed legitimately re-applies.
    sync.clear_attention()
    sync.poll_once()
    assert len(session.sales) == 1


# ---------------------------------------------------------------------------
# Owner-name import: the names must reach the visible room, not just the map
# ---------------------------------------------------------------------------


def test_connect_imports_real_names_into_the_visible_room(sync, session):
    # connect() ran in the fixture; the room must already show Sleeper's names.
    for r in range(1, N + 1):
        owner_id = OWNER_IDS[r - 1]
        assert session.team_names[owner_id] == f"Team Name {r}"


def test_import_preserves_stable_owner_ids_and_the_focus_seat(sync, session):
    assert sorted(session.state.owner_by_id) == sorted(OWNER_IDS)
    assert session.focus_owner_id == OWNER_IDS[0]
    assert sync.owners.owner_for(1) == OWNER_IDS[0], "roster 1 is still our seat"


def test_import_is_idempotent_and_does_not_relog_on_reconnect(sync, session,
                                                              fake):
    renames = [e for e in session.log if e.get("kind") == "rename"]
    again = sync.import_owner_names()
    assert again == [], "a second import must rename nothing"
    assert [e for e in session.log if e.get("kind") == "rename"] == renames


def test_display_name_is_used_when_no_team_name_published(session, fake,
                                                          tmp_path):
    fake.users[2] = {"user_id": "user3", "display_name": "owner3",
                     "metadata": {}}
    client = SleeperClient("d", fetch=fake.fetch)
    s = SleeperSync(session, client, owner_ids=OWNER_IDS,
                    state_path=tmp_path / "s.json")
    s.connect(sleeper_players=fake.players)
    assert session.team_names[OWNER_IDS[2]] == "owner3"


def test_a_blank_name_is_never_invented(session, fake, tmp_path):
    fake.users[3] = {"user_id": "user4", "display_name": "  ",
                     "metadata": {"team_name": ""}}
    client = SleeperClient("d", fetch=fake.fetch)
    s = SleeperSync(session, client, owner_ids=OWNER_IDS,
                    state_path=tmp_path / "s.json")
    before = session.team_names.get(OWNER_IDS[3])
    s.connect(sleeper_players=fake.players)
    assert session.team_names.get(OWNER_IDS[3]) == before


def test_imported_names_survive_a_reset(sync, session):
    session.reset()
    for r in range(1, N + 1):
        assert session.team_names[OWNER_IDS[r - 1]] == f"Team Name {r}"


def test_status_exposes_both_stable_ids_and_visible_names(sync, session):
    out = sync.snapshot_status()
    assert out["owner_map"]["1"] == OWNER_IDS[0], "stable id, not the name"
    assert out["owner_names"]["1"] == "Team Name 1"
