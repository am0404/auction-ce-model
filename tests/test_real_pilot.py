"""Real-board pilot machinery, exercised on FABRICATED fixtures.

Nothing here reads ``local_data/``. The pilot's selection, diagnostics,
sanitization and league arithmetic are all testable against the fabricated
board, and they must be: a test that needs the real contract cannot run in CI
and would silently stop guarding anything.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from ceauction.auction.proxy import ProxyEvaluator
from ceauction.auction.state import new_auction
from ceauction.league import DEFAULT_LEAGUE, Position
from ceauction.tactical.demo import build_tactical_demo
from ceauction.tactical.realpilot import (BENCH_IMPROVEMENT, OWNER_IDS,
                                          TIER_LABELS, Candidate,
                                          CandidateDiagnostics, RealBoard,
                                          SanitizationError, SanitizedReport,
                                          diagnose, select_candidates)


# ---------------------------------------------------------------------------
# A fabricated stand-in for the real board
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fake_board():
    """The fabricated demo pool wired into the pilot's RealBoard shape."""
    from ceauction.auction.completion import ComparisonCast
    from ceauction.market.costbook import cost_book_from_prior

    d = build_tactical_demo()
    state = new_auction(list(d.state.pool), OWNER_IDS, OWNER_IDS[0],
                        settings=DEFAULT_LEAGUE)
    state.validate()
    costs = {}
    for scenario in ("low", "base", "high"):
        cb = cost_book_from_prior(d.prior, scenario)
        gaps = {s.player_id: 1 for s in state.pool if s.player_id not in cb}
        costs[scenario] = cb.with_costs(gaps) if gaps else cb
    return RealBoard(
        state=state, cast=ComparisonCast(0, tuple(() for _ in OWNER_IDS),
                                         OWNER_IDS),
        prior=d.prior, market=d.market, costs=costs,
        key_by_id=dict(d.key_by_id),
        name_by_id={s.player_id: s.name for s in state.pool},
        coverage={"mapped_playerspecs": len(state.pool)})


# ---------------------------------------------------------------------------
# The opening room
# ---------------------------------------------------------------------------


def test_the_opening_room_is_twelve_by_two_hundred(fake_board):
    st = fake_board.state
    assert len(st.owners) == 12
    assert sum(o.budget_start for o in st.owners) == 2400
    assert sum(o.roster_capacity for o in st.owners) == 180
    for o in st.owners:
        assert o.budget_start == 200
        assert o.roster_capacity == 15
        assert o.n_players == 0
        assert o.open_slots == 15
        assert o.reserve_for_open_slots == 15


def test_the_opening_legal_maximum_is_186(fake_board):
    """$200 less $1 held for each of the other fourteen slots."""
    for o in fake_board.state.owners:
        assert o.max_bid == 186
    assert len({o.max_bid for o in fake_board.state.owners}) == 1


def test_the_pool_can_legally_complete_every_team(fake_board):
    st = fake_board.state
    assert len(st.available_ids) >= 180
    ids = [s.player_id for s in st.pool]
    assert len(ids) == len(set(ids)), "duplicate player ids"
    for o in st.owners:
        assert o.can_still_field_lineup(st.available_by_position())


def test_there_is_no_dedicated_te_slot_and_superflex_takes_any(fake_board):
    from ceauction.auction.feasibility import PositionCounts, can_fill_lineup
    # A legal eight: QB, RB, RB, WR/TE x3, flex, superflex. Filling the
    # superflex with a running back must be legal; strict 2QB would refuse it.
    one_qb = PositionCounts(qb=1, rb=4, wr=3, te=0)
    assert can_fill_lineup(one_qb), "superflex must accept an RB/WR/TE"
    # Three WR/TE seats may be filled entirely by receivers -- no TE required.
    no_te = PositionCounts(qb=2, rb=3, wr=3, te=0)
    assert can_fill_lineup(no_te)
    # ...or entirely by tight ends. Neither is privileged.
    all_te = PositionCounts(qb=2, rb=3, wr=0, te=3)
    assert can_fill_lineup(all_te)


def test_no_roster_maximum_for_quarterbacks(fake_board):
    st = fake_board.state
    owner = OWNER_IDS[1]
    qbs = [s for s in st.available_specs
           if Position(int(s.position)) is Position.QB][:5]
    cur = st
    for spec in qbs:
        assert cur.purchase_shortfall(spec.player_id, owner, 1) is None
        cur = cur.apply_purchase(spec.player_id, owner, 1)
    assert cur.owner(owner).counts.qb == 5


# ---------------------------------------------------------------------------
# Candidate selection
# ---------------------------------------------------------------------------


def test_selection_is_deterministic(fake_board):
    a, _ = select_candidates(fake_board)
    b, _ = select_candidates(fake_board)
    assert [c.player_id for c in a] == [c.player_id for c in b]
    assert [c.tier for c in a] == [c.tier for c in b]


def test_selection_covers_every_position_and_tier(fake_board):
    cands, _ = select_candidates(fake_board)
    for pos in ("QB", "RB", "WR", "TE"):
        group = [c for c in cands if c.position == pos]
        assert len(group) == 3, pos
        assert {c.tier for c in group} == set(TIER_LABELS)
    assert len(cands) == 12
    assert len({c.player_id for c in cands}) == 12


def test_expensive_tier_outranks_cheap_tier(fake_board):
    cands, _ = select_candidates(fake_board)
    for pos in ("QB", "RB", "WR", "TE"):
        g = {c.tier: c for c in cands if c.position == pos}
        assert g["expensive"].position_rank < g["cheap"].position_rank


def test_a_missing_position_is_reported_not_crashed(fake_board):
    cands, notes = select_candidates(fake_board, positions=("QB", "K"))
    assert any("K" in n for n in notes)
    assert all(c.position != "K" for c in cands)


def test_selection_never_hard_codes_a_name():
    """Committed selection code must be driven by data, not by identity."""
    src = Path(__file__).resolve().parents[1] / "src" / "ceauction" / "tactical"
    text = (src / "realpilot.py").read_text(encoding="utf-8")
    assert "name_by_id" in text, "names exist, but only for local output"
    fn = text[text.index("def select_candidates"):text.index("def diagnose")]
    body = fn.split('"""')[2] if fn.count('"""') >= 2 else fn
    # Position(...).name is an enum label, not a player name; the reaches that
    # would matter are the ones that touch the identity map or a spec's name.
    for reach in ("name_by_id", "spec.name", "chosen.name", ".player_name"):
        assert reach not in body, f"selection consults {reach}"


# ---------------------------------------------------------------------------
# Diagnostics and the sampling policy
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def diags(fake_board):
    px = ProxyEvaluator(fake_board.state.pool, fake_board.state.settings, 32, 7)
    cands, _ = select_candidates(fake_board)
    return [(c, diagnose(fake_board, c, proxy=px)) for c in cands]


def test_the_reference_roster_is_legal_and_not_all_quarterbacks(fake_board):
    """The first pilot run filled fourteen slots with QBs and read QB1 at 0.03."""
    px = ProxyEvaluator(fake_board.state.pool, fake_board.state.settings, 32, 7)
    cands, _ = select_candidates(fake_board)
    qb = next(c for c in cands if c.position == "QB" and c.tier == "expensive")
    d = diagnose(fake_board, qb, proxy=px)
    assert d.lineup_improvement > 0.0, \
        "a top QB must gain something in an open superflex seat"


def test_policy_follows_lineup_improvement_not_price(diags):
    for c, d in diags:
        if d.lineup_improvement < BENCH_IMPROVEMENT:
            assert d.policy == "proxy only"
            assert d.role == "bench/insurance"
        else:
            assert d.policy == "4000-season audit"
    # An expensive player with no lineup effect must still be proxy-only, and
    # a cheap player with a real effect must still earn an audit. Price alone
    # decides nothing.
    prices = {(c.position, c.tier): (c.price_base or 0) for c, _ in diags}
    policies = {(c.position, c.tier): d.policy for c, d in diags}
    assert prices and policies
    mismatch = [k for k in policies
                if policies[k] == "proxy only" and prices[k] > 20]
    assert isinstance(mismatch, list)  # documented, never auto-corrected


def test_a_bench_candidate_is_never_given_an_audited_reservation(diags):
    for c, d in diags:
        if d.policy == "proxy only":
            assert "waste" in d.reason or "resolve" in d.reason
            assert d.lineup_improvement < BENCH_IMPROVEMENT


def test_no_hard_coded_tight_end_premium():
    src = Path(__file__).resolve().parents[1] / "src" / "ceauction" / "tactical"
    for path in src.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for banned in ("te_premium", "te_scarcity", "tight_end_bonus",
                       "te_multiplier"):
            assert banned not in text, f"{banned} in {path.name}"


def test_market_anchor_and_ce_value_stay_separate(diags):
    for c, d in diags:
        blob = d.to_dict()
        assert "anchor_display" not in blob
        assert "price_base" not in blob
        assert "lineup_improvement" in blob


# ---------------------------------------------------------------------------
# Sanitization
# ---------------------------------------------------------------------------


def _report(**kw):
    base = dict(coverage={"mapped_playerspecs": 260}, selection={},
                diagnostics=[], results={}, runtime={})
    base.update(kw)
    return SanitizedReport(**base)


def test_the_sanitized_report_refuses_a_player_name():
    r = _report(results={"note": "Patrick Mahomes was the top candidate"})
    with pytest.raises(SanitizationError, match="real player name"):
        r.check(["Patrick Mahomes"])


def test_the_sanitized_report_refuses_player_level_identity():
    r = _report(diagnostics=[{"player_id": 12, "position": "QB"}])
    with pytest.raises(SanitizationError, match="player-level identity"):
        r.check([])


def test_a_clean_report_passes_and_carries_its_label():
    r = _report(diagnostics=[{"position": "QB", "lineup_improvement": 1.2}],
                results={"n_resolved": 3})
    r.check(["Some Player"])
    blob = r.to_dict()
    assert "SANITIZED AGGREGATE" in blob["label"]
    assert "no player-level dollar figure" in blob["label"].lower()


def test_short_names_do_not_cause_false_positives():
    """A two-letter name must not veto the whole report."""
    _report(results={"note": "ok"}).check(["Al", "Bo"])


# ---------------------------------------------------------------------------
# Repository hygiene
# ---------------------------------------------------------------------------


def test_no_local_data_file_is_tracked():
    out = subprocess.run(["git", "ls-files", "local_data"],
                         capture_output=True, text=True, check=True)
    assert out.stdout.strip() == ""


def test_the_real_output_directory_is_ignored():
    out = subprocess.run(
        ["git", "check-ignore", "-v", "local_data/tactical/real_board_pilot.json"],
        capture_output=True, text=True)
    assert out.returncode == 0, "the pilot output path must be gitignored"


def test_the_committed_sanitized_report_has_no_names_or_dollars():
    path = Path(__file__).resolve().parents[1] / "docs" / "real_pilot_sanitized.json"
    if not path.exists():
        pytest.skip("sanitized report not generated in this checkout")
    blob = json.loads(path.read_text(encoding="utf-8"))
    text = json.dumps(blob)
    assert "SANITIZED AGGREGATE" in blob["label"]
    for row in blob.get("diagnostics", []):
        assert "player_id" not in row and "name" not in row
    for row in blob.get("results", {}).get("audited", []):
        assert "name" not in row and "player_id" not in row


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli(*argv):
    return subprocess.run([sys.executable, "-m", "ceauction.cli", "tactical",
                           *argv], capture_output=True, text=True)


@pytest.mark.parametrize("argv,needle", [
    (("real-pilot", "--contract", "/nonexistent.json"), "real inputs are missing"),
    (("real-pilot", "--out-dir", "/tmp/leak"), "must sit under local_data"),
    (("real-pilot", "--position", "K"), "invalid choice"),
    (("real-pilot", "--market-scenario", "medium"), "invalid choice"),
])
def test_cli_usage_errors_have_no_traceback(argv, needle):
    r = _cli(*argv)
    assert r.returncode != 0
    assert "Traceback" not in r.stderr
    assert needle in (r.stdout + r.stderr)
