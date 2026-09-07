"""Targeted tests for the confidence-gated rough CE price board.

The board's job is not to be right about every player -- it is a stated
approximation. Its job is to never present a number it has not measured. So
these tests are written from that side: what does a LOW row expose, what does
the gate refuse, and can a noisy consensus reach the price field by any route.
"""

from __future__ import annotations

import statistics

import pytest

from ceauction.draftday.ceboard import (
    MEDIUM_MAX_SPREAD,
    MEDIUM_MIN_SEED_AGREEMENT,
    NOISY_MESSAGE,
    RANK_SEEDS,
    BoardEntry,
    PlayerCE,
    _pct,
    assign_dollars,
    live_value,
)


def _ce(pid=1, name="P", pos="WR"):
    return PlayerCE(player_id=pid, name=name, position=pos, anchor=10,
                    delta_by_context={"balanced": 0.01, "qb_light": 0.01,
                                      "qb_heavy": 0.01},
                    displaced_by_context={})


def _entry(prices, *, market_base=20, pid=1, pos="WR"):
    """Build an entry from fifteen (seed, context) prices, five seeds x three."""
    obs = []
    for i, seed in enumerate(RANK_SEEDS):
        for j, ctx in enumerate(("balanced", "qb_light", "qb_heavy")):
            obs.append({"seed": seed, "context": ctx,
                        "price": int(prices[i * 3 + j]), "delta": 0.01})
    return BoardEntry(player_id=pid, name="P", position=pos, anchor=10,
                      market_base=market_base, ce=_ce(pid, pos=pos),
                      observations=tuple(obs))


# ---------------------------------------------------------------------------
# Sampling shape
# ---------------------------------------------------------------------------


def test_five_distinct_predeclared_seeds():
    assert len(RANK_SEEDS) == 5
    assert len(set(RANK_SEEDS)) == 5, "seeds must be independent, not repeats"


def test_fifteen_observations_per_player():
    e = _entry([25] * 15)
    assert len(e.observations) == 15
    assert len({o["seed"] for o in e.observations}) == 5
    assert len({o["context"] for o in e.observations}) == 3


def test_percentiles_and_median_are_the_documented_ones():
    prices = list(range(1, 16))  # 1..15
    e = _entry(prices)
    assert e.center == statistics.median(prices) == 8
    assert e.low == _pct(prices, 0.20) == 3
    assert e.high == _pct(prices, 0.80) == 12
    assert e.spread == 9


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


def test_tight_and_agreeing_is_medium():
    # every seed median above a market base of 20, band well inside $12
    e = _entry([30, 31, 32] * 5, market_base=20)
    assert e.seed_agreement == 5
    assert e.spread <= MEDIUM_MAX_SPREAD
    assert e.confidence == "MEDIUM"


def test_wide_band_is_low_even_when_every_seed_agrees():
    e = _entry([25, 40, 55] * 5, market_base=20)
    assert e.seed_agreement == 5, "all five seeds still point the same way"
    assert e.spread > MEDIUM_MAX_SPREAD
    assert e.confidence == "LOW", "agreement alone must not buy a price"


def test_split_seeds_are_low_even_when_tight():
    # three seeds above the base, two below; each seed internally tight
    prices = [25, 25, 25, 25, 25, 25, 25, 25, 25, 15, 15, 15, 15, 15, 15]
    e = _entry(prices, market_base=20)
    assert e.seed_agreement == 3 < MEDIUM_MIN_SEED_AGREEMENT
    assert e.confidence == "LOW", "a direction the seeds dispute is not a price"


def test_conservation_failure_forces_low():
    e = _entry([30, 31, 32] * 5, market_base=20)
    bad = BoardEntry(player_id=e.player_id, name=e.name, position=e.position,
                     anchor=e.anchor, market_base=e.market_base, ce=e.ce,
                     observations=e.observations, conservation_ok=False)
    assert bad.confidence == "LOW"


def test_no_high_confidence_exists():
    for prices in ([30] * 15, [1] * 15, list(range(1, 16))):
        assert _entry(prices).confidence in ("MEDIUM", "LOW")


# ---------------------------------------------------------------------------
# What a LOW row is allowed to say
# ---------------------------------------------------------------------------


def test_low_row_exposes_no_actionable_max():
    e = _entry([2, 30, 55] * 5, market_base=35)
    assert e.confidence == "LOW"
    d = e.to_dict()
    assert d["rough_ce_max"] is None
    assert d["range_low"] is None and d["range_high"] is None
    assert d["message"] == NOISY_MESSAGE


def test_low_row_keeps_the_consensus_in_diagnostics():
    e = _entry([2, 30, 55] * 5, market_base=35)
    d = e.to_dict()
    assert d["diagnostics"]["consensus_median"] == e.center
    assert d["diagnostics"]["n_observations"] == 15
    assert len(d["diagnostics"]["seed_medians"]) == 5


def test_low_row_leans_only_when_seeds_agree():
    agreeing = _entry([40, 50, 60] * 5, market_base=20)
    assert agreeing.confidence == "LOW" and agreeing.lean == "CE LEANS HIGHER"
    split = _entry([25, 25, 25, 25, 25, 25, 25, 25, 25, 5, 5, 5, 5, 5, 5],
                   market_base=20)
    assert split.seed_agreement == 3
    assert split.lean == "", "no direction may be claimed when seeds disagree"


def test_medium_row_shows_its_price():
    e = _entry([30, 31, 32] * 5, market_base=20)
    d = e.to_dict()
    assert d["rough_ce_max"] == e.center
    assert d["message"] == ""


# ---------------------------------------------------------------------------
# The working number
# ---------------------------------------------------------------------------


class _Level:
    def __init__(self, m=1.0):
        self.multiplier = m


class _Market:
    def __init__(self, room=1.0, pos=1.0):
        self.room_effect = _Level(room)
        self.position_levels = {"WR": _Level(pos)}


def test_low_row_working_number_is_the_market_guardrail():
    e = _entry([2, 30, 55] * 5, market_base=35)
    lv = live_value(e, market=_Market(), position="WR", legal_max=100,
                    guardrail=37)
    assert e.confidence == "LOW"
    assert lv.working_number == 37
    assert "MARKET GUARDRAIL" in lv.working_basis
    assert lv.to_dict()["live_rough_ce_shown"] is None


def test_medium_row_working_number_is_the_ce_range():
    e = _entry([30, 31, 32] * 5, market_base=20)
    lv = live_value(e, market=_Market(), position="WR", legal_max=100,
                    guardrail=22)
    assert lv.working_number == lv.live != 22
    assert "CE RANGE" in lv.working_basis
    assert lv.to_dict()["live_rough_ce_shown"] == lv.live


def test_live_multipliers_move_a_medium_row():
    e = _entry([30, 31, 32] * 5, market_base=20)
    flat = live_value(e, market=_Market(1.0, 1.0), position="WR",
                      legal_max=200, guardrail=22)
    hot = live_value(e, market=_Market(1.0, 1.5), position="WR",
                     legal_max=200, guardrail=22)
    assert hot.live > flat.live
    assert hot.opening == flat.opening, "the opening consensus must not move"


def test_exact_legal_cap_binds_both_confidences():
    med = _entry([30, 31, 32] * 5, market_base=20)
    lv = live_value(med, market=_Market(1.0, 4.0), position="WR",
                    legal_max=9, guardrail=50)
    assert lv.live == 9 and lv.working_number == 9 and lv.clamped
    low = _entry([2, 30, 55] * 5, market_base=35)
    lv2 = live_value(low, market=_Market(), position="WR", legal_max=9,
                     guardrail=50)
    assert lv2.working_number == 9, "the guardrail is capped by the legal max too"


def test_the_two_bases_are_never_blended():
    e = _entry([2, 30, 55] * 5, market_base=35)
    lv = live_value(e, market=_Market(), position="WR", legal_max=100,
                    guardrail=37)
    # the working number is exactly one of the two inputs, never a mixture
    assert lv.working_number == 37


# ---------------------------------------------------------------------------
# Determinism and the dollar distribution
# ---------------------------------------------------------------------------


def test_rank_mapping_preserves_the_dollar_multiset():
    scores = {i: PlayerCE(player_id=i, name=f"p{i}", position="WR", anchor=1,
                          delta_by_context={"balanced": i / 100.0},
                          displaced_by_context={}) for i in range(1, 8)}
    base = {i: v for i, v in zip(range(1, 8), [50, 40, 30, 20, 10, 5, 1])}
    out = assign_dollars(scores, base, "balanced")
    assert sorted(out.values()) == sorted(base.values())
    assert sum(out.values()) == sum(base.values())


def test_ce_order_not_sleeper_order_assigns_the_dollars():
    # player 1 is cheapest on the market and best on CE: he must get the top price
    scores = {1: PlayerCE(1, "best", "WR", 1, {"balanced": 0.9}, {}),
              2: PlayerCE(2, "mid", "WR", 99, {"balanced": 0.5}, {}),
              3: PlayerCE(3, "worst", "WR", 50, {"balanced": 0.1}, {})}
    base = {1: 1, 2: 99, 3: 50}
    out = assign_dollars(scores, base, "balanced")
    assert out[1] == 99 and out[3] == 1


def test_ties_break_on_player_id_deterministically():
    scores = {i: PlayerCE(i, f"p{i}", "WR", 1, {"balanced": 0.5}, {})
              for i in (3, 1, 2)}
    base = {1: 30, 2: 20, 3: 10}
    first = assign_dollars(scores, base, "balanced")
    assert first == assign_dollars(scores, base, "balanced")
    assert first[1] == 30, "lowest player_id takes the top price on an exact tie"


def test_fingerprint_covers_every_seed():
    import ceauction.draftday.ceboard as cb
    from ceauction.draftday.ceboard import board_fingerprint

    class _S:
        def fingerprint(self):
            return "state"
        settings = "settings"

    base = {1: 10}
    a = board_fingerprint(_S(), "digest", base_prices=base, seasons=1000, seed=1)
    original = cb.RANK_SEEDS
    try:
        cb.RANK_SEEDS = original[:-1] + (original[-1] + 1,)
        b = board_fingerprint(_S(), "digest", base_prices=base, seasons=1000,
                              seed=1)
    finally:
        cb.RANK_SEEDS = original
    assert a != b, "changing any ranking seed must invalidate the board"


def test_no_prohibited_certainty_language():
    from ceauction.draftday.ceboard import (FORBIDDEN_WORDS, NOISY_MESSAGE,
                                            ROUGH_DISCLOSURE, ROUGH_LABEL)
    for text in (ROUGH_LABEL, ROUGH_DISCLOSURE, NOISY_MESSAGE):
        upper = text.upper()
        for bad in FORBIDDEN_WORDS:
            assert bad not in upper, f"{bad!r} must never appear in {text!r}"
