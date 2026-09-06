"""The ``ce-lab auction`` commands.

**Every command here runs on the fabricated demo auction.** No real player, no
real price.

What is defended: that each command runs, that anticipated user errors exit
non-zero without a traceback, that fabricated inputs are labelled as such, and
that no output presents itself as an opening maximum, a recommended bid, or a
clearing-price prediction.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

#: Small enough to keep the suite fast; the point is the plumbing, not power.
TINY = ["--beam-width", "40", "--candidate-pool", "25", "--finalists", "2",
        "--sims", "300"]


def _sentences(text: str):
    """Split on sentence-ish boundaries, keeping newlines out of the way.

    A banned phrase is only a problem when the sentence containing it is an
    assertion rather than a denial, and disclaimers wrap across lines, so the
    unit of judgement has to be the sentence and not the line.
    """
    flat = " ".join(text.split())
    out, buf = [], []
    for token in flat.split(". "):
        out.append(token)
    return out


def run(*args, expect=0):
    proc = subprocess.run(
        [sys.executable, "-m", "ceauction.cli", "auction", *args],
        capture_output=True, text=True, cwd=str(REPO))
    assert proc.returncode == expect, (
        f"exit {proc.returncode} (wanted {expect})\n"
        f"stdout:\n{proc.stdout[-2000:]}\nstderr:\n{proc.stderr[-2000:]}")
    return proc


def test_validate_reports_a_clean_demo_state():
    p = run("validate")
    assert "VALID: every invariant holds." in p.stdout
    assert "FABRICATED" in p.stdout


def test_validate_can_write_json(tmp_path):
    out = tmp_path / "state.json"
    run("validate", "--json-out", str(out))
    blob = json.loads(out.read_text(encoding="utf-8"))
    assert blob["valid"] is True
    assert blob["n_owners"] == 12
    assert len(blob["owners"]) == 12


def test_room_prints_every_owner_and_who_may_bid():
    p = run("room", "--next-bid", "25")
    for i in range(1, 13):
        assert f"Owner{i:02d}" in p.stdout
    # With nobody on the block this is the financial ceiling, and the summary
    # says so rather than implying legal eligibility for a particular player.
    assert "WHO CAN AFFORD $25" in p.stdout
    assert "FINANCIAL" in p.stdout
    assert "Who WOULD bid is not modelled" in p.stdout
    assert "no quarterback maximum" in p.stdout


def test_room_output_carries_no_player_names():
    p = run("room")
    assert "Fabricated" not in p.stdout


def test_complete_solves_and_labels_itself_heuristic():
    p = run("complete", *TINY)
    assert "ROSTER COMPLETION" in p.stdout
    assert "BOUNDED SEARCH" in p.stdout
    assert "not a proof of optimality" in p.stdout
    assert "FABRICATED" in p.stdout


def test_complete_without_ce_says_it_is_expected_points():
    p = run("complete", "--no-ce", *TINY)
    assert "EXPECTED POINTS, not championship" in p.stdout


def test_buy_pass_runs_both_destinations():
    gone = run("buy-pass", "--price", "15", *TINY)
    assert "unavailable (off the board)" in gone.stdout
    rival = run("buy-pass", "--price", "15", "--pass-to", "Owner02",
                "--pass-price", "16", *TINY)
    assert "Owner02 at $16" in rival.stdout
    assert "re-completed" in rival.stdout


def test_buy_pass_requires_a_price_for_a_named_rival():
    p = run("buy-pass", "--price", "15", "--pass-to", "Owner02", *TINY, expect=2)
    assert "--pass-price" in p.stderr
    assert "Traceback" not in p.stderr


def test_a_price_above_the_legal_maximum_exits_two_without_a_traceback():
    p = run("buy-pass", "--price", "5000", *TINY, expect=2)
    assert "Traceback" not in p.stderr
    assert "legal maximum" in p.stderr or "cannot evaluate" in p.stderr


def test_an_unknown_candidate_exits_two_without_a_traceback():
    p = run("buy-pass", "--price", "10", "--candidate", "-999", *TINY, expect=2)
    assert "Traceback" not in p.stderr
    assert "not in this auction's pool" in p.stderr


def test_an_unknown_pass_destination_exits_two_without_a_traceback():
    p = run("buy-pass", "--price", "10", "--pass-to", "Nobody",
            "--pass-price", "10", *TINY, expect=2)
    assert "Traceback" not in p.stderr
    assert "no owner" in p.stderr or "no slot in the comparison cast" in p.stderr


def test_loading_a_real_state_is_refused_rather_than_guessed():
    p = run("validate", "--state", "whatever.json", expect=2)
    assert "not implemented" in p.stderr
    assert "counterfactual pretending to be a room" in p.stderr
    assert "Traceback" not in p.stderr


def test_reservation_estimates_runtime_before_committing():
    p = run("reservation", "--prices", "1", "20", "--estimate-only", *TINY)
    assert "estimated runtime" in p.stdout
    assert "comparison(s)" in p.stdout


def test_reservation_flags_a_reduced_scenario_grid():
    p = run("reservation", "--prices", "1", "20", "--estimate-only", *TINY)
    assert "REDUCED grid" in p.stdout
    assert "not the full cross-product" in p.stdout


def test_reservation_rejects_a_bad_scenario_id():
    p = run("reservation", "--prices", "1", "--scenarios", "not-a-scenario",
            "--estimate-only", *TINY, expect=2)
    assert "Traceback" not in p.stderr
    assert "not a scenario id" in p.stderr


def test_reservation_runs_and_reports_a_range(tmp_path):
    out = tmp_path / "res.json"
    p = run("reservation", "--prices", "1", "40", "--json-out", str(out), *TINY)
    assert "CE RESERVATION-PRICE RANGE" in p.stdout
    assert "robust" in p.stdout and "permissive" in p.stdout
    blob = json.loads(out.read_text(encoding="utf-8"))
    assert blob["label"].startswith("CE reservation-price range")
    assert blob["full_grid"] is False
    assert blob["cost_level"] == "FABRICATED"
    assert blob["interval_is"].startswith("pointwise")
    assert blob["ladder_is_exhaustive"] is False


def test_reservation_warns_that_a_sparse_ladder_only_brackets():
    p = run("reservation", "--prices", "1", "40", "--estimate-only", *TINY)
    assert "SPARSE ladder" in p.stdout
    assert "bracket the frontier" in p.stdout


def test_reservation_refinement_is_available_from_the_cli():
    p = run("reservation", "--prices", "1", "6", "--refine",
            "--max-refinement-prices", "6", *TINY)
    assert "CE RESERVATION-PRICE RANGE" in p.stdout


@pytest.mark.parametrize("cmd", [
    ["room"],
    ["complete", "--no-ce"] + TINY,
    ["buy-pass", "--price", "15"] + TINY,
    ["reservation", "--prices", "1", "20"] + TINY,
])
def test_no_command_claims_to_produce_a_bid(cmd):
    """The three prices must not be conflated anywhere a user can see.

    Each phrase is allowed to appear inside an explicit denial -- the whole
    point of the disclaimers is to name the thing being disclaimed -- so an
    occurrence counts only when nothing negates it just before.
    """
    p = run(*cmd)
    lowered = p.stdout.lower()
    for banned in ("max bid you should", "recommended bid", "opening max",
                   "your max bid", "market price", "real value"):
        for sentence in _sentences(lowered):
            if banned not in sentence:
                continue
            assert any(neg in sentence for neg in
                       ("not ", "never ", "nothing ", "no ", "refuse")), (
                f"{cmd[0]} printed {banned!r} in a sentence that does not deny "
                f"it: {sentence.strip()!r}")


def test_slot_swap_refuses_a_missing_contract():
    p = run("slot-swap", "--contract", "no/such/file.json", expect=2)
    assert "contract not found" in p.stderr
    assert "Traceback" not in p.stderr


def test_slot_swap_refuses_a_reduced_grid_without_the_flag(tmp_path):
    contract = tmp_path / "c.json"
    contract.write_text(json.dumps({"schema_version": "1.0.0", "players": []}),
                        encoding="utf-8")
    p = run("slot-swap", "--contract", str(contract),
            "--scenarios", "fh-f000-s000-w1", expect=2)
    assert "all 54 scenarios" in p.stderr
    assert "--allow-reduced" in p.stderr
    assert "Traceback" not in p.stderr


def test_the_committed_examples_exist_and_are_sanitized():
    """Committed output must carry no player names and no real-value claim."""
    for name in ("auction_room.txt", "auction_completion.txt",
                 "auction_buy_pass_unavailable.txt",
                 "auction_buy_pass_rival.txt",
                 "auction_reservation_sparse.txt",
                 "auction_reservation_refined.txt",
                 "auction_benchmark.txt"):
        path = REPO / "docs" / "examples" / name
        assert path.exists(), f"missing committed example {name}"
        text = path.read_text(encoding="utf-8")
        # The generated player names are "Fabricated0001" and so on. Matching
        # the digits rather than the bare word lets prose say "fabricated"
        # without tripping the check that identities never leak.
        assert not re.search(r"Fabricated\d", text), (
            f"{name} leaks generated player names")
        assert "fabricated" in text.lower(), f"{name} does not label its inputs"
        assert "recommended bid" not in text.lower()
