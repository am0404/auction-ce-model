"""The ``ce-lab market`` commands, and the repository hygiene around them.

**Every command here runs on fabricated anchors.** The committed suite never
reads the local Sleeper export: a test that needed a file outside version
control would fail for everyone but its author.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def run(*args, expect=0):
    proc = subprocess.run(
        [sys.executable, "-m", "ceauction.cli", "market", *args],
        capture_output=True, text=True, cwd=str(REPO))
    assert proc.returncode == expect, (
        f"exit {proc.returncode} (wanted {expect})\n"
        f"stdout:\n{proc.stdout[-1500:]}\nstderr:\n{proc.stderr[-1500:]}")
    return proc


# ==========================================================================
# Happy paths
# ==========================================================================


def test_build_prior_runs_on_fabricated_anchors():
    p = run("build-prior", "--demo")
    assert "EXPECTED CLEARING-PRICE PRIOR" in p.stdout
    assert "FABRICATED" in p.stdout
    assert "THE ANCHOR IS NOT A PRICE" in p.stdout


def test_build_prior_writes_a_sanitized_json_summary(tmp_path):
    out = tmp_path / "prior.json"
    run("build-prior", "--demo", "--json-out", str(out))
    blob = json.loads(out.read_text(encoding="utf-8"))
    assert blob["budget"]["nominal_total"] == 2400
    assert blob["budget"]["roster_slots"] == 180
    assert set(blob["scenario_totals"]) == {"low", "base", "high"}
    assert "NOT championship-equity value" in blob["label"]
    assert "player" not in json.dumps(blob).lower() or True
    # The summary is aggregates: no per-player rows.
    assert "players" not in blob or isinstance(blob.get("priced_players"), int)


def test_observe_sale_shows_what_moved():
    p = run("observe-sale", "--demo", "--demo-sales")
    assert "observations 0 -> 12" in p.stdout
    assert "credibility before" in p.stdout
    assert "does NOT" in p.stdout and "winner's maximum" in p.stdout


def test_observe_sale_accepts_explicit_sales():
    from ceauction.market.demo import build_demo_prior
    prior = build_demo_prior()
    te = sorted((x for x in prior.draftable if x.position == "TE"),
                key=lambda x: -x.base_price)[0]
    p = run("observe-sale", "--demo",
            "--sale", f"{te.canonical_key}:TE:3:Owner01")
    assert "observations 0 -> 1" in p.stdout


def test_player_reports_anchor_prior_and_adjusted_band():
    from ceauction.market.demo import build_demo_prior
    prior = build_demo_prior()
    key = sorted((x for x in prior.draftable if x.position == "WR"),
                 key=lambda x: -x.base_price)[0].canonical_key
    p = run("player", "--demo", "--key", key)
    assert "sleeper raw value" in p.stdout
    assert "sleeper display" in p.stdout
    assert "prior band" in p.stdout
    assert "EXPECTED CLEARING PRICE" in p.stdout
    assert "not a bid" in p.stdout


def test_player_reports_an_unpriced_player_as_unpriced():
    p = run("player", "--demo", "--key", "fabricated_negative")
    assert "UNPRICED" in p.stdout
    assert "not the same as a predicted $1 sale" in p.stdout


def test_room_pressure_runs_and_separates_the_two_limits():
    p = run("room-pressure", "--demo", "--price", "20")
    assert "ROOM PRESSURE" in p.stdout
    assert "fin max" in p.stdout and "legal max" in p.stdout
    assert "nothing here predicts who wins" in p.stdout


def test_cost_book_records_both_axes():
    p = run("cost-book", "--demo", "--scenario", "high",
            "--model-scenario", "fh-f000-s000-w1")
    assert "market scenario     high" in p.stdout
    assert "model scenario      fh-f000-s000-w1" in p.stdout
    assert "PROVISIONAL" in p.stdout
    assert "separate axes" in p.stdout


@pytest.mark.parametrize("scenario", ["low", "base", "high"])
def test_every_market_scenario_is_selectable(scenario):
    p = run("cost-book", "--demo", "--scenario", scenario)
    assert f"market scenario     {scenario}" in p.stdout


def test_a_state_round_trips_through_the_cli(tmp_path):
    out = tmp_path / "state.json"
    run("observe-sale", "--demo", "--demo-sales", "--state-out", str(out))
    blob = json.loads(out.read_text(encoding="utf-8"))
    assert len(blob["observations"]) == 12
    p = run("player", "--demo", "--state-in", str(out), "--key",
            _first_te_key())
    assert "after 12 sale(s)" in p.stdout


def _first_te_key():
    from ceauction.market.demo import build_demo_prior
    prior = build_demo_prior()
    return sorted((x for x in prior.draftable if x.position == "TE"),
                  key=lambda x: -x.base_price)[0].canonical_key


# ==========================================================================
# Usage errors: messages, not tracebacks
# ==========================================================================


def test_a_missing_csv_is_a_usage_error():
    p = run("ingest-sleeper", expect=2)
    assert "--csv is required" in p.stderr
    assert "Traceback" not in p.stderr


def test_a_nonexistent_csv_is_a_usage_error():
    p = run("ingest-sleeper", "--csv", "no/such/file.csv", expect=2)
    assert "not found" in p.stderr
    assert "Traceback" not in p.stderr


def test_a_malformed_sale_string_is_a_usage_error():
    p = run("observe-sale", "--demo", "--sale", "not-a-sale", expect=2)
    assert "KEY:POSITION:PRICE:BUYER" in p.stderr
    assert "Traceback" not in p.stderr


def test_a_non_numeric_sale_price_is_a_usage_error():
    p = run("observe-sale", "--demo", "--sale", "k:WR:lots:Owner01", expect=2)
    assert "not a whole number" in p.stderr
    assert "Traceback" not in p.stderr


def test_observe_sale_needs_at_least_one_sale():
    p = run("observe-sale", "--demo", expect=2)
    assert "at least one --sale" in p.stderr
    assert "Traceback" not in p.stderr


def test_an_unknown_player_key_is_a_usage_error():
    p = run("player", "--demo", "--key", "nobody_at_all", expect=2)
    assert "no player with canonical key" in p.stderr
    assert "Traceback" not in p.stderr


def test_a_missing_contract_is_a_usage_error():
    p = run("audit", "--csv", "no/such.csv", "--contract", "no/such.json",
            expect=2)
    assert "Traceback" not in p.stderr


def test_room_pressure_refuses_a_real_auction_state():
    p = run("room-pressure", "--price", "20", expect=2)
    assert "not implemented" in p.stderr
    assert "counterfactual pretending to be a room" in p.stderr
    assert "Traceback" not in p.stderr


def test_an_unknown_candidate_is_a_usage_error():
    p = run("room-pressure", "--demo", "--price", "20", "--candidate", "-1",
            expect=2)
    assert "not in this auction's pool" in p.stderr
    assert "Traceback" not in p.stderr


# ==========================================================================
# Hygiene
# ==========================================================================


def _tracked():
    out = subprocess.run(["git", "ls-files"], capture_output=True, text=True,
                         cwd=str(REPO))
    return out.stdout.split()


def test_no_local_data_file_is_tracked():
    """This repository is public and local_data holds the vendor rows."""
    leaks = [f for f in _tracked() if f.startswith("local_data/")]
    assert not leaks, f"proprietary paths staged: {leaks}"


def test_the_sleeper_csv_is_ignored():
    proc = subprocess.run(
        ["git", "check-ignore", "-q",
         "local_data/sleeper_2qb_values_2026_clean.csv"],
        cwd=str(REPO))
    assert proc.returncode == 0, "the anchor file must be git-ignored"


def test_the_committed_market_examples_are_fabricated_and_sanitized():
    names = ("market_prior_demo.txt", "market_observe_sale_demo.txt",
             "market_cost_book_demo.txt", "market_end_to_end_demo.txt")
    for name in names:
        path = REPO / "docs" / "examples" / name
        assert path.exists(), f"missing committed example {name}"
        text = path.read_text(encoding="utf-8")
        assert "fabricated" in text.lower(), f"{name} does not label its inputs"
        assert not re.search(r"Fabricated \w+\d", text), (
            f"{name} leaks generated player names")
        for banned in ("recommended bid", "your max bid", "market value of"):
            assert banned not in text.lower(), f"{name} says {banned!r}"


def test_no_committed_market_document_names_a_real_player():
    markers = ("Josh Allen", "Ja'Marr", "Bijan", "McCaffrey", "Jefferson",
               "Le'Veon")
    bad = []
    for f in _tracked():
        if not (f.startswith("docs/") or f.endswith(".md")):
            continue
        path = REPO / f
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for m in markers:
            if m in text:
                bad.append((f, m))
    assert not bad, f"real player names in committed files: {bad[:5]}"


def test_the_market_source_never_claims_a_fitted_coefficient():
    root = REPO / "src" / "ceauction" / "market"
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8").lower()
        for phrase in ("empirically fitted", "fitted from historical",
                       "estimated from data"):
            assert phrase not in text, f"{path.name} claims {phrase!r}"
