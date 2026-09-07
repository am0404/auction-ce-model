"""Acceptance tests for the draft-day tool.

These run on a **synthetic** board built in-process, so the suite passes on a
clean checkout where ``local_data/`` does not exist. The real board is
exercised separately by ``ce-lab draft-day verify`` and ``draft-day mock``,
which are local-only because their inputs are.

What is asserted here is the product's contract, not its internals: the
recommendation basis is always one of the closed set, an unanchored player
never acquires a market band, no unresolved CE result reaches a cap, undo is
exact, a restart replays to the same fingerprint, and a stale cache cannot
satisfy a changed state.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ceauction.auction.state import AuctionRuleError, new_auction
from ceauction.draftday import caps as C
from ceauction.draftday.board import DraftDayBoard, opening_rows, write_opening_board
from ceauction.draftday.panels import nomination_panel, owner_table, pid, qb_panel
from ceauction.draftday.session import (
    CONTINGENCY_TAGS,
    DraftSession,
    ManualOverride,
)
from ceauction.league import DEFAULT_LEAGUE, Position
from ceauction.market.anchors import AnchorBook, SleeperAnchor
from ceauction.market.live import MarketState
from ceauction.market.prior import build_market_prior
from ceauction.players import PlayerSpec
from ceauction.realdata.identity import canonical_player_key
from ceauction.tactical.realpilot import OWNER_IDS

from decimal import Decimal


# ---------------------------------------------------------------------------
# A synthetic board: enough players to fill twelve rosters, half of them
# anchored so the anchored/unanchored distinction is actually under test.
# ---------------------------------------------------------------------------

_SHAPE = ((Position.QB, 40), (Position.RB, 70), (Position.WR, 90),
          (Position.TE, 40))


def _make_board() -> DraftDayBoard:
    specs, keys, names, points, ppg = [], {}, {}, {}, {}
    anchors = []
    pid_counter = 1000
    for position, count in _SHAPE:
        for rank in range(count):
            pid_counter += 1
            player_id = pid_counter
            name = f"{position.name} Player {rank:03d}"
            # The market layer keys players by a canonical form of the NAME, so
            # the board's key must be derived the same way or nothing joins.
            key = canonical_player_key(name=name)
            level = 18.0 - rank * 0.12
            specs.append(PlayerSpec(
                player_id=player_id, name=name, position=position,
                nfl_team=f"T{rank % 32:02d}", base_mean=max(1.0, level),
                week_sd=max(1.0, level * 0.4),
                bye_week=5 + (rank % 10)))
            keys[player_id] = key
            names[player_id] = name
            points[player_id] = round(max(1.0, level) * 17.0, 2)
            ppg[player_id] = round(max(1.0, level), 2)
            # Only the top half of each position gets an anchor, so the board
            # carries genuinely unanchored players.
            if rank < count // 2:
                anchors.append(SleeperAnchor(
                    sleeper_player_id=str(player_id), player_name=name,
                    position=position.name, nfl_team=f"T{rank % 32:02d}",
                    active=True,
                    raw_value=Decimal(str(round(max(0.5, 60.0 - rank * 1.1), 2)))))

    book = AnchorBook(anchors=tuple(anchors), source_sha256="0" * 64,
                      notes="synthetic test anchors")
    prior = build_market_prior(book, settings=DEFAULT_LEAGUE,
                               player_ids={k: p for p, k in keys.items()},
                               notes="synthetic")
    state = new_auction(specs, OWNER_IDS, OWNER_IDS[0], settings=DEFAULT_LEAGUE)
    raw, disp = {}, {}
    for player_id, key in keys.items():
        pr = prior.by_key.get(key)
        raw[player_id] = float(pr.raw_value) if pr and pr.raw_value is not None else None
        disp[player_id] = pr.display_anchor if pr else None
    return DraftDayBoard(
        state=state, prior=prior, market=MarketState(prior=prior),
        key_by_id=keys, name_by_id=names, coverage={"synthetic": True},
        ppg_by_id=ppg, points_by_id=points,
        raw_anchor_by_id=raw, display_anchor_by_id=disp)


@pytest.fixture(scope="module")
def board() -> DraftDayBoard:
    return _make_board()


@pytest.fixture
def session(board, tmp_path) -> DraftSession:
    return DraftSession(board, state_path=tmp_path / "draft_state.json",
                        overrides_path=tmp_path / "manual_overrides.json")


def _available(session, position=None, skip=()):
    for spec in session.state.available_specs:
        if position is not None and int(spec.position) != int(position):
            continue
        if spec.player_id in skip:
            continue
        yield spec.player_id


def _first(session, position=None, skip=()):
    return next(_available(session, position, skip))


# ---------------------------------------------------------------------------
# The board loads, and says what it knows
# ---------------------------------------------------------------------------


def test_real_board_shape_loads(session):
    rows = opening_rows(session.board)
    assert len(rows) == sum(n for _, n in _SHAPE)
    assert all(r.position in ("QB", "RB", "WR", "TE") for r in rows)
    # Enough players to fill every roster in the league.
    assert len(rows) >= DEFAULT_LEAGUE.n_teams * DEFAULT_LEAGUE.roster_size


def test_every_displayed_basis_is_one_of_the_closed_set(session):
    rows = opening_rows(session.board)
    assert {r.basis for r in rows} <= set(C.BASES)


def test_unanchored_players_are_labelled_not_priced_at_one_dollar(session):
    rows = opening_rows(session.board)
    unanchored = [r for r in rows if not r.anchored]
    assert unanchored, "the fixture must contain unanchored players"
    for row in unanchored:
        assert row.live_low is None and row.live_base is None
        assert row.live_high is None
        assert "UNANCHORED" in row.notes
        # An unanchored player is capped by the legal maximum alone.
        assert row.basis == C.EXACT


def test_anchored_players_carry_a_band(session):
    rows = [r for r in opening_rows(session.board) if r.anchored]
    assert rows
    for row in rows[:40]:
        assert row.live_low <= row.live_base <= row.live_high
        assert row.basis in (C.MARKET_PRIOR, C.MARKET_LIVE)


def test_no_forced_te_scarcity_premium(session):
    """A tight end must not be given a premium for being a tight end."""
    rows = {r.player_id: r for r in opening_rows(session.board)}
    tes = [r for r in rows.values() if r.position == "TE" and r.anchored]
    wrs = [r for r in rows.values() if r.position == "WR" and r.anchored]
    assert tes and wrs
    # The cap is a minimum of stated rails; nothing in it reads the position.
    for row in tes:
        expected = min(row.legal_max, row.live_high)
        assert row.provisional_cap == expected


def test_board_csv_and_json_are_written(session, tmp_path):
    rows = opening_rows(session.board)
    csv_path, json_path = write_opening_board(rows, tmp_path)
    assert csv_path.exists() and json_path.exists()
    payload = json.loads(json_path.read_text())
    assert payload["counts"]["players"] == len(rows)
    assert payload["counts"]["anchored"] + payload["counts"]["unanchored"] \
        == len(rows)
    assert "not championship-equity max bids" in payload["label"]


# ---------------------------------------------------------------------------
# The cap policy
# ---------------------------------------------------------------------------


def test_cap_is_the_minimum_of_the_stated_rails():
    band = C.MarketBand(low=20, base=25, high=30, anchored=True)
    cap = C.provisional_cap(C.CapRails(legal_max=100, market=band))
    assert (cap.cap, cap.bound_by, cap.basis) == (30, "market_high",
                                                  C.MARKET_PRIOR)
    assert cap.label == C.MARKET_LED_LABEL

    cap = C.provisional_cap(C.CapRails(legal_max=12, market=band))
    assert (cap.cap, cap.bound_by) == (12, "legal_max")

    cap = C.provisional_cap(C.CapRails(legal_max=100, market=band,
                                       proxy_ceiling=18,
                                       proxy_status="cached"))
    assert (cap.cap, cap.bound_by, cap.basis) == (18, "proxy_ceiling", C.PROXY)


def test_the_basis_names_the_rail_that_actually_bound_the_number():
    """A cap the rules produced must not be credited to an estimate."""
    band = C.MarketBand(low=20, base=25, high=30, anchored=True)
    # Market binds: the recommendation really is market-led.
    assert C.provisional_cap(
        C.CapRails(legal_max=100, market=band)).basis == C.MARKET_PRIOR
    # Our own money binds: this is arithmetic, not a market opinion.
    tight = C.provisional_cap(C.CapRails(legal_max=9, market=band))
    assert tight.bound_by == "legal_max"
    assert tight.basis == C.EXACT
    assert tight.label == "LEGAL MAXIMUM BINDS"
    # A player we cannot buy at all.
    gone = C.provisional_cap(C.CapRails(legal_max=0, market=band))
    assert gone.cap == 0 and gone.basis == C.EXACT


def test_cap_with_no_proxy_is_market_led_and_never_called_ce():
    band = C.MarketBand(low=5, base=8, high=11, anchored=True)
    cap = C.provisional_cap(C.CapRails(legal_max=50, market=band))
    assert cap.label == C.MARKET_LED_LABEL
    assert cap.basis != C.CE_AUDITED
    assert not cap.is_ce


def test_unresolved_ce_never_produces_a_ce_cap():
    """Every gate condition, refused one at a time."""
    band = C.MarketBand(low=5, base=8, high=11, anchored=True)
    good = {"converged": True, "interval_resolved": True, "reconciled": True,
            "pass_semantics": "buy-vs-pass", "displayed_decision": "buy-vs-pass",
            "state_fingerprint": "abc", "bracket": [10, 14]}
    ok, why = C.ce_is_usable(good, "abc")
    assert ok and why == "usable"

    for field, expected in (
            ("converged", C.SEARCH_UNDERCONVERGED),
            ("interval_resolved", C.CE_UNDERPOWERED),
            ("reconciled", C.CE_UNDERPOWERED)):
        bad = dict(good, **{field: False})
        ok, why = C.ce_is_usable(bad, "abc")
        assert not ok and why == expected

    ok, why = C.ce_is_usable(dict(good, displayed_decision="other"), "abc")
    assert not ok and why == C.CE_UNDERPOWERED

    ok, why = C.ce_is_usable(good, "a-different-state")
    assert not ok and why.startswith("stale")

    # And a refused result cannot become a CE cap.
    for record in (None, dict(good, converged=False)):
        ok, why = C.ce_is_usable(record, "abc")
        cap = C.provisional_cap(C.CapRails(legal_max=50, market=band,
                                           ce_status=why))
        assert cap.basis != C.CE_AUDITED
        assert cap.cap == 11


def test_qb_union_underconverged_record_is_refused():
    """The actual state of the quarterback work must not produce a CE cap."""
    record = {"converged": False, "interval_resolved": False,
              "reconciled": False, "bracket": None,
              "state_fingerprint": "whatever"}
    ok, why = C.ce_is_usable(record, "whatever")
    assert not ok
    assert why == C.SEARCH_UNDERCONVERGED


def test_manual_override_takes_the_basis_and_is_clamped_to_legal():
    band = C.MarketBand(low=5, base=8, high=11, anchored=True)
    cap = C.provisional_cap(C.CapRails(legal_max=50, market=band,
                                       manual_adjustment=6))
    assert cap.cap == 17 and cap.basis == C.MANUAL

    cap = C.provisional_cap(C.CapRails(legal_max=13, market=band,
                                       manual_adjustment=100))
    assert cap.cap == 13, "an override may never exceed the legal maximum"


def test_model_disagreement_is_flagged():
    band = C.MarketBand(low=5, base=10, high=12, anchored=True)
    near = C.provisional_cap(C.CapRails(legal_max=99, market=band,
                                        proxy_ceiling=12,
                                        proxy_status="cached"))
    assert not near.disagreement
    far = C.provisional_cap(C.CapRails(legal_max=99, market=band,
                                       proxy_ceiling=40,
                                       proxy_status="cached"))
    assert far.disagreement
    assert C.DISAGREEMENT_LABEL in far.to_dict()["disagreement_label"]


def test_cached_proxy_with_no_favourable_price_says_so():
    band = C.MarketBand(low=5, base=8, high=11, anchored=True)
    cap = C.provisional_cap(C.CapRails(legal_max=50, market=band,
                                       proxy_ceiling=None,
                                       proxy_status="cached"))
    assert any("no price" in n for n in cap.notes)
    assert cap.cap == 11


# ---------------------------------------------------------------------------
# Sales, rejections, undo, persistence
# ---------------------------------------------------------------------------


def test_sale_updates_every_dependent_field(session):
    player_id = _first(session, Position.QB)
    before = session.state.owner("Team02")
    band_before = session.board.prior.by_key[session.board.key_for(player_id)]
    assert band_before.draftable

    session.record_sale(player_id, "Team02", 30)
    after = session.state.owner("Team02")

    assert after.budget_remaining == before.budget_remaining - 30
    assert after.spent == 30
    assert after.n_players == before.n_players + 1
    assert after.open_slots == before.open_slots - 1
    assert after.counts.qb == before.counts.qb + 1
    assert after.max_bid < before.max_bid
    assert session.state.owner_of[player_id] == "Team02"
    assert player_id not in session.state.available_ids
    assert len(session.market.observations) == 1
    assert session.log[-1]["kind"] == "sale"


def test_unanchored_sale_does_not_move_the_market(session):
    rows = {r.player_id: r for r in opening_rows(session.board)}
    unanchored = next(p for p, r in rows.items() if not r.anchored)
    session.record_sale(unanchored, "Team03", 4)
    assert len(session.market.observations) == 0, (
        "an unanchored sale has no prior to be a residual against")
    assert "UNANCHORED" in session.log[-1]["market_note"]


def test_duplicate_sale_is_rejected(session):
    player_id = _first(session, Position.RB)
    session.record_sale(player_id, "Team02", 10)
    assert "already belongs" in session.check_sale(player_id, "Team03", 10)
    with pytest.raises(AuctionRuleError):
        session.record_sale(player_id, "Team03", 10)


def test_overspend_is_rejected(session):
    player_id = _first(session, Position.RB)
    problem = session.check_sale(player_id, "Team02", 500)
    assert problem is not None and "exceeds" in problem


def test_reserve_violation_is_rejected(session):
    """A bid that leaves no dollar for each remaining slot is illegal."""
    owner = session.state.owner("Team02")
    legal = owner.max_bid
    player_id = _first(session, Position.RB)
    assert session.check_sale(player_id, "Team02", legal) is None
    assert session.check_sale(player_id, "Team02", legal + 1) is not None
    # The legal maximum is exactly budget - (open slots - 1) at the minimum.
    assert legal == owner.budget_remaining - (owner.open_slots - 1)


def test_full_roster_purchase_is_rejected(session):
    taken = set()
    for _ in range(DEFAULT_LEAGUE.roster_size):
        player_id = next(p for p in _available(session, skip=taken)
                         if session.check_sale(p, "Team05", 1) is None)
        session.record_sale(player_id, "Team05", 1)
        taken.add(player_id)
    assert session.state.owner("Team05").is_full
    spare = _first(session, skip=taken)
    assert "no open roster slot" in session.check_sale(spare, "Team05", 1)


def test_candidate_specific_illegal_purchase_is_rejected(session):
    """An owner whose last slot must hold a receiver cannot buy a QB.

    Built from the rules rather than asserted: buy quarterbacks and running
    backs until the engine refuses another, then check that what it refuses is
    the quarterback and what it allows is the receiver. The stopping point is
    the league's own arithmetic -- three WR/TE lineup seats out of fifteen
    roster slots means at most twelve non-receivers -- and the test reads it
    off the engine rather than restating it.
    """
    owner = "Team06"
    taken = set()
    pool = (list(_available(session, Position.QB))
            + list(_available(session, Position.RB)))
    for player_id in pool:
        if session.check_sale(player_id, owner, 1) is None:
            session.record_sale(player_id, owner, 1)
            taken.add(player_id)
    held = session.state.owner(owner)
    assert held.counts.wt == 0
    assert held.open_slots > 0, "the owner must still have room to buy someone"

    qb = _first(session, Position.QB, taken)
    wr = _first(session, Position.WR, taken)
    problem = session.check_sale(qb, owner, 1)
    assert problem is not None and "complete a legal roster" in problem
    assert session.check_sale(wr, owner, 1) is None

    # And the owner table must report the same thing the engine does.
    rows = {r["owner_id"]: r for r in owner_table(session, candidate_id=qb,
                                                  next_bid=1)}
    assert rows[owner]["can_bid_candidate"] is False
    assert rows[owner]["candidate_max"] == 0
    assert rows[owner]["general_max"] > 0, (
        "the general financial maximum and the candidate-specific one are "
        "different numbers and both must be shown")


def test_invalid_price_and_unknown_ids_are_rejected(session):
    player_id = _first(session, Position.WR)
    assert session.check_sale(player_id, "Team02", 0) is not None
    assert session.check_sale(player_id, "Team02", -5) is not None
    assert session.check_sale(player_id, "Nobody", 5) is not None
    assert session.check_sale(-1, "Team02", 5) is not None
    assert session.check_sale(player_id, "Team02", 1.5) is not None


def test_undo_is_exact(session):
    fingerprints = [session.fingerprint()]
    ids = [_first(session, Position.QB)]
    session.record_sale(ids[0], "Team02", 25)
    fingerprints.append(session.fingerprint())
    ids.append(_first(session, Position.RB))
    session.record_sale(ids[1], "Team03", 18)
    fingerprints.append(session.fingerprint())
    assert len(set(fingerprints)) == 3

    session.undo()
    assert session.fingerprint() == fingerprints[1]
    session.undo()
    assert session.fingerprint() == fingerprints[0]
    assert not session.can_undo
    assert session.undo() is None
    # Both engines, not just the auction.
    assert len(session.market.observations) == 0
    assert session.state.rostered_ids == frozenset()


def test_restart_persistence(board, tmp_path):
    paths = dict(state_path=tmp_path / "s.json",
                 overrides_path=tmp_path / "o.json")
    first = DraftSession(board, **paths)
    first.record_sale(_first(first, Position.QB), "Team02", 33)
    first.record_sale(_first(first, Position.RB), "Team04", 21)
    first.rename_team("Team04", "The Rivals")
    saved = first.fingerprint()

    second = DraftSession(board, **paths)
    assert second.restore() is True
    assert second.fingerprint() == saved
    assert len(second.sales) == 2
    assert second.team_name("Team04") == "The Rivals"
    assert second.state.owner("Team02").budget_remaining == 167


def test_export_import_round_trips(board, tmp_path):
    session = DraftSession(board, state_path=tmp_path / "s.json",
                           overrides_path=tmp_path / "o.json")
    session.record_sale(_first(session, Position.WR), "Team07", 14)
    saved = session.fingerprint()
    snapshot = session.export_snapshot(tmp_path / "snap.json")

    other = DraftSession(board, state_path=tmp_path / "s2.json",
                         overrides_path=tmp_path / "o2.json")
    other.import_snapshot(snapshot)
    assert other.fingerprint() == saved


def test_import_refuses_a_different_player_pool(board, tmp_path):
    session = DraftSession(board, state_path=tmp_path / "s.json",
                           overrides_path=tmp_path / "o.json")
    payload = session.to_payload()
    payload["pool_fingerprint"] = "a-different-board"
    with pytest.raises(AuctionRuleError):
        session.load_payload(payload)


def test_reset_clears_sales_but_keeps_overrides(session):
    player_id = _first(session, Position.QB)
    session.set_override(player_id, dollar_adjustment=-4, reasoning="hunch")
    session.record_sale(_first(session, Position.RB), "Team02", 9)
    assert session.sales
    session.reset()
    assert not session.sales
    assert session.state.rostered_ids == frozenset()
    assert session.overrides, "a reset must not discard the user's own research"


# ---------------------------------------------------------------------------
# Overrides
# ---------------------------------------------------------------------------


def test_manual_override_persists(board, tmp_path):
    paths = dict(state_path=tmp_path / "s.json",
                 overrides_path=tmp_path / "o.json")
    first = DraftSession(board, **paths)
    player_id = _first(first, Position.RB)
    first.set_override(player_id, dollar_adjustment=-6,
                       contingency_tag="committee",
                       linked_starter="RB Player 001",
                       takeover_share=0.4, reasoning="timeshare")
    assert paths["overrides_path"].exists()

    second = DraftSession(board, **paths)
    key = board.key_for(player_id)
    assert key in second.overrides
    assert second.overrides[key].dollar_adjustment == -6
    assert second.overrides[key].contingency_tag == "committee"
    assert second.overrides[key].takeover_share == 0.4


def test_override_changes_the_cap_and_the_basis(session):
    player_id = next(p for p in _available(session, Position.RB)
                     if session.board.prior.by_key[
                         session.board.key_for(p)].draftable)
    before = {r.player_id: r for r in opening_rows(
        session.board, state=session.state, market=session.market)}[player_id]
    session.set_override(player_id, dollar_adjustment=-5)
    after = {r.player_id: r for r in opening_rows(
        session.board, state=session.state, market=session.market,
        overrides=session.override_map())}[player_id]
    assert after.provisional_cap == before.provisional_cap - 5
    assert after.basis == C.MANUAL


def test_override_rejects_an_unknown_contingency_tag():
    with pytest.raises(ValueError):
        ManualOverride(player_key="x", contingency_tag="made up")
    for tag in CONTINGENCY_TAGS:
        ManualOverride(player_key="x", contingency_tag=tag)


def test_override_rejects_an_out_of_range_takeover_share():
    with pytest.raises(ValueError):
        ManualOverride(player_key="x", takeover_share=1.4)


def test_unmodelled_handcuffs_say_so_rather_than_inventing_one(session):
    rows = opening_rows(session.board)
    backs = [r for r in rows if r.position == "RB" and not r.manual_adjustment]
    assert backs
    assert any("HANDCUFF VALUE NOT MODELED" in r.notes for r in backs)


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------


def test_nomination_panel_reports_exact_arithmetic_and_a_basis(session):
    player_id = _first(session, Position.QB)
    panel = nomination_panel(session, player_id, current_bid=12,
                             high_bidder="Team03")
    assert panel["next_legal_bid"] == 13
    assert panel["our_legal_max"] == session.state.owner("Team01").max_bid
    assert panel["our_legal_max_basis"] == C.EXACT
    assert panel["cap"]["basis"] in C.BASES
    assert panel["verdict"] in ("BID", "CAUTION", "STOP", C.BELOW_MARKET,
                               C.IN_MARKET_RANGE, C.ABOVE_MARKET,
                               C.OVER_LEGAL_MAX, C.CANNOT_BID, "UNPRICED")
    assert panel["provisional_cap"] <= panel["our_legal_max"]
    assert isinstance(panel["player_id"], str), "ids must survive JavaScript"
    assert panel["recipient_warning"]
    assert "not fitted" in panel["recipient_warning"]


def test_a_converged_proxy_may_recommend_bidding_and_stopping():
    band = C.MarketBand(low=5, base=10, high=11, anchored=True)
    rails = C.CapRails(legal_max=40, market=band, proxy_ceiling=11,
                       proxy_status="cached")
    cap = C.provisional_cap(rails)
    assert C.recommend(cap, 6).decision == "BID"
    assert C.recommend(cap, 11).decision == "CAUTION"
    assert C.recommend(cap, 12).decision == "STOP"
    assert C.recommend(cap, 999).decision == C.OVER_LEGAL_MAX


def test_opponent_specific_money_is_visible(session):
    session.record_sale(_first(session, Position.QB), "Team02", 60)
    player_id = _first(session, Position.RB)
    rows = owner_table(session, candidate_id=player_id, next_bid=1)
    assert len(rows) == 12
    by_id = {r["owner_id"]: r for r in rows}
    assert by_id["Team02"]["spent"] == 60
    assert by_id["Team02"]["budget_remaining"] == 140
    assert by_id["Team02"]["general_max"] < by_id["Team03"]["general_max"]
    for row in rows:
        assert row["candidate_max"] <= row["general_max"]


def test_qb_scarcity_updates_after_qb_sales(session):
    before = qb_panel(session)
    assert before["counts_by_bucket"]["0"] == 12
    sold = []
    for player_id in list(_available(session, Position.QB))[:3]:
        session.record_sale(player_id, f"Team{len(sold) + 2:02d}", 5)
        sold.append(player_id)
    after = qb_panel(session)
    assert after["startable_remaining"] < before["startable_remaining"]
    assert after["counts_by_bucket"]["1"] == 3
    assert after["counts_by_bucket"]["0"] == 9
    assert after["total_qbs_remaining"] == before["total_qbs_remaining"] - 3


def test_qb_panel_imposes_no_roster_rule(session):
    """No second-QB requirement and no maximum: five is legal and reported."""
    for i, player_id in enumerate(list(_available(session, Position.QB))[:5]):
        session.record_sale(player_id, "Team02", 1)
    panel = qb_panel(session)
    assert panel["counts_by_bucket"]["3+"] == 1
    assert session.state.owner("Team02").counts.qb == 5
    assert "requires no second quarterback and caps none" in \
        panel["startable_definition"]
    assert "NOT AUDITED" in panel["qb3_insurance"]


def test_qb_panel_warns_when_replacement_is_unavailable(session):
    """Sell enough quarterbacks that supply cannot cover the teams without one."""
    owners = [o for o in OWNER_IDS if o != "Team01"]
    startable = qb_panel(session)["startable_cut"]
    for i, player_id in enumerate(list(_available(session, Position.QB))[:startable]):
        session.record_sale(player_id, owners[i % len(owners)], 1)
    panel = qb_panel(session)
    assert panel["startable_remaining"] == 0
    assert "would likely NOT be available" in panel["replacement_warning"]


# ---------------------------------------------------------------------------
# Caches
# ---------------------------------------------------------------------------


def test_stale_cache_cannot_satisfy_a_changed_state(session):
    from ceauction.draftday.proxycache import ProxyCeilingCache
    cache = ProxyCeilingCache(session, enabled=True)
    player_id = _first(session, Position.RB)
    key = (session.fingerprint(), player_id)
    # Plant a completed result for the CURRENT state.
    cache._done[key] = {"permissive_ceiling": 42, "error": ""}
    assert cache.get(player_id) == (42, "cached")

    session.record_sale(_first(session, Position.QB), "Team02", 11)
    assert session.fingerprint() != key[0]
    assert cache.get(player_id) == (None, "absent"), (
        "a result computed under another room must never be served")

    session.undo()
    assert cache.get(player_id) == (42, "cached"), (
        "returning to the same state may reuse it -- the room did not change")


def test_override_changes_the_fingerprint(session):
    before = session.fingerprint()
    session.set_override(_first(session, Position.RB), dollar_adjustment=-3)
    assert session.fingerprint() != before, (
        "an override changes a displayed cap, so a cache keyed without it "
        "could serve a number the user has already corrected")


def test_board_cache_key_tracks_file_contents(tmp_path):
    from ceauction.draftday.board import _digest
    a, b = tmp_path / "a.json", tmp_path / "b.csv"
    a.write_text("one")
    b.write_text("two")
    first = _digest(a, b)
    assert _digest(a, b) == first
    a.write_text("changed")
    assert _digest(a, b) != first


# ---------------------------------------------------------------------------
# The server, over its own API
# ---------------------------------------------------------------------------


@pytest.fixture
def app(session):
    from ceauction.draftday.server import DraftDayServer
    server = DraftDayServer(session, enable_proxy=False)
    server.bootstrap_opening()
    return server


def test_api_state_and_board(app):
    code, payload = app.handle("GET", "/api/state", {}, {})
    assert code == 200
    assert len(payload["owners"]) == 12
    assert payload["counts"]["players"] == sum(n for _, n in _SHAPE)
    assert payload["counts"]["anchored"] > 0
    assert payload["counts"]["unanchored"] > 0

    code, payload = app.handle("GET", "/api/board", {"mode": ["live"]}, {})
    assert code == 200 and payload["rows"]
    assert all(isinstance(r["player_id"], str) for r in payload["rows"])


def test_api_rejects_bad_input_without_a_traceback(app):
    for body in ({}, {"player_id": "nope"}, {"player_id": "1", "price": "abc"}):
        code, payload = app.handle("POST", "/api/sale", {}, body)
        assert code == 400 and payload["error"]
    code, payload = app.handle("POST", "/api/reset", {}, {})
    assert code == 400
    code, payload = app.handle("GET", "/api/nomination", {}, {})
    assert code == 400
    code, payload = app.handle("GET", "/api/nowhere", {}, {})
    assert code == 404


def test_api_sale_and_undo(app, session):
    player_id = _first(session, Position.QB)
    opening = session.fingerprint()
    code, payload = app.handle("POST", "/api/sale", {},
                               {"player_id": pid(player_id),
                                "owner_id": "Team02", "price": 20})
    assert code == 200 and payload["sale"]["price"] == 20
    code, payload = app.handle("POST", "/api/undo", {}, {})
    assert code == 200
    assert session.fingerprint() == opening
    code, payload = app.handle("POST", "/api/undo", {}, {})
    assert code == 400


def test_opening_cap_does_not_silently_mutate(app, session):
    """The frozen opening cap stays put while the live cap moves."""
    rows = {r["player_id"]: r for r in app.opening()}
    target = next(r for r in rows.values()
                  if r["anchored"] and not r["sold"] and r["position"] == "RB")
    opening_cap = target["opening_cap"]

    # Move the market with several running backs sold well over their bands.
    sold = 0
    for player_id in list(_available(session, Position.RB)):
        if str(player_id) == target["player_id"]:
            continue
        row = rows.get(str(player_id))
        if not row or not row["anchored"]:
            continue
        session.record_sale(player_id, f"Team{sold + 2:02d}",
                            int(row["live_base"]) + 10)
        sold += 1
        if sold >= 4:
            break

    live = {r["player_id"]: r for r in app.rows()}[target["player_id"]]
    assert live["opening_cap"] == opening_cap, "the opening cap must not move"
    assert app.opening()[0]["opening_cap"] == rows[
        app.opening()[0]["player_id"]]["opening_cap"]
    code, panel = app.handle("GET", "/api/nomination",
                             {"player_id": [target["player_id"]]}, {})
    assert panel["opening_cap"] == opening_cap
    assert panel["provisional_cap"] != opening_cap or \
        panel["cap"]["rails"]["market"]["basis"] == C.MARKET_LIVE


def test_market_only_moves_on_applicable_sales(app, session):
    """Selling receivers must move receivers more than the untouched backs."""
    rows = {r["player_id"]: r for r in app.rows()}
    wr_ids = [p for p in _available(session, Position.WR)
              if rows[str(p)]["anchored"]]
    rb_ids = [p for p in _available(session, Position.RB)
              if rows[str(p)]["anchored"]]
    probe_wr, probe_rb = wr_ids[-1], rb_ids[-1]
    before_wr = rows[str(probe_wr)]["live_base"]
    before_rb = rows[str(probe_rb)]["live_base"]

    for i, player_id in enumerate(wr_ids[:5]):
        session.record_sale(player_id, f"Team{i + 2:02d}",
                            int(rows[str(player_id)]["live_base"]) + 12)

    after = {r["player_id"]: r for r in app.rows()}
    wr_lift = after[str(probe_wr)]["live_base"] / max(1, before_wr)
    rb_lift = after[str(probe_rb)]["live_base"] / max(1, before_rb)
    assert wr_lift > 1.0, "the observed position must respond"
    assert rb_lift < wr_lift, (
        "an unobserved position must not move as far as the observed one")


def test_api_override_round_trip(app, session):
    player_id = _first(session, Position.RB)
    code, payload = app.handle("POST", "/api/override", {},
                               {"player_id": pid(player_id),
                                "dollar_adjustment": -8,
                                "contingency_tag": "full handcuff",
                                "reasoning": "backup to a fragile starter"})
    assert code == 200
    assert payload["override"]["dollar_adjustment"] == -8
    code, payload = app.handle("POST", "/api/override", {},
                               {"player_id": pid(player_id),
                                "contingency_tag": "invented"})
    assert code == 400

    rows = {r["player_id"]: r for r in app.rows()}
    assert rows[pid(player_id)]["basis"] == C.MANUAL


def test_page_is_self_contained(app):
    from ceauction.draftday.server import PAGE
    assert "<title>Draft Day</title>" in PAGE
    # No external resource of any kind: this must work with no network.
    for token in ("http://", "https://", "cdn.", "<script src"):
        assert token not in PAGE, f"the page must not reference {token}"
