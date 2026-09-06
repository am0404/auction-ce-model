"""The Sleeper anchor, the format-adjusted prior, and the live sale updater.

**Every input is fabricated.** No real player, no real price, and the committed
tests never read the local Sleeper export -- a test suite that needed a file
outside version control would fail for everyone but its author.

What these defend is the separation the whole package exists for: a generic
Sleeper `2qb` anchor, an expected clearing price for *this* league, our own
championship-equity reservation, and a tactical bid are four different
quantities, and none of them may quietly become another.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

import pytest

from ceauction.auction.costs import MissingCost
from ceauction.auction.demo import build_demo_auction
from ceauction.league import DEFAULT_LEAGUE, Position
from ceauction.market import (MARKET_SCENARIOS, AnchorBook, AnchorError,
                              MarketState, RoomBudget, SaleObservation,
                              ShrinkageConfig, SleeperAnchor,
                              assess_room_pressure, build_market_prior,
                              cost_book_from_market_state, cost_book_from_prior,
                              display_price, format_prior_summary,
                              load_sleeper_csv, match_anchors_to_contract,
                              tier_of)
from ceauction.market.demo import DEMO_SALES, build_demo_anchors, build_demo_prior

REPO = Path(__file__).resolve().parents[1]


def anchor(pid, name, pos, value, *, active=True, team="ZZA"):
    return SleeperAnchor(str(pid), name, pos, team, active, Decimal(str(value)))


@pytest.fixture(scope="module")
def demo_prior():
    return build_demo_prior()


# ==========================================================================
# The source layer
# ==========================================================================


def test_positive_display_rounding_is_decimal_half_up():
    """All four observed pairs, including the one banker's rounding gets wrong."""
    assert display_price(Decimal("58.19")) == 58
    assert display_price(Decimal("33.69")) == 34
    assert display_price(Decimal("12.78")) == 13
    assert display_price(Decimal("16.50")) == 17
    assert round(16.50) == 16, "python's built-in is banker's rounding"


@pytest.mark.parametrize("value", ["0", "-0.01", "-69.85"])
def test_a_nonpositive_value_has_no_invented_display_price(value):
    """Sleeper's UI treatment of these was never observed, so none is invented."""
    a = anchor(1, "Fabricated One", "RB", value)
    assert display_price(Decimal(value)) is None
    assert a.display_anchor is None
    assert a.status == "nonpositive"
    assert not a.counts_toward_room_budget


def test_the_raw_value_is_preserved_exactly():
    a = anchor(1, "Fabricated One", "WR", "58.19")
    assert a.raw_value == Decimal("58.19")
    assert a.to_dict()["raw_value"] == "58.19"


def test_source_metadata_records_the_generic_2qb_endpoint():
    """Not this league's format, and the label must never imply otherwise."""
    book = AnchorBook((anchor(1, "Fabricated One", "WR", "5"),), source_sha256="x")
    prov = book.provenance()
    assert prov["format"] == "2qb"
    assert prov["season"] == 2026
    assert "api.sleeper.com" in prov["endpoint"]
    assert "GENERIC" in prov["label"]
    assert "NOT this league's market" in prov["label"]


def test_ineligible_positions_are_excluded_from_budget_reconciliation():
    """Money the league cannot spend is not part of the room's money."""
    book = AnchorBook((
        anchor(1, "Fabricated Wr", "WR", "10"),
        anchor(2, "Fabricated Kicker", "K", "9"),
        anchor(3, "Fabricated Defense", "DEF", "8"),
        anchor(4, "Fabricated Retired", "RB", "7", active=False),
    ), source_sha256="x")
    assert [a.position for a in book.priced] == ["WR"]
    assert book.priced_raw_total == Decimal("10")
    assert len(book.excluded_from_budget) == 3
    statuses = {a.status for a in book.excluded_from_budget}
    assert statuses == {"ineligible_position", "inactive"}


def test_a_malformed_number_is_refused_rather_than_coerced(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text('sleeper_player_id,player_name,position,nfl_team,active,'
                 'sleeper_2qb_raw_value\n1,Fabricated One,WR,ZZA,true,not-a-number\n',
                 encoding="utf-8")
    with pytest.raises(AnchorError, match="not a number"):
        load_sleeper_csv(p)


def test_a_missing_column_is_refused(tmp_path):
    p = tmp_path / "short.csv"
    p.write_text("sleeper_player_id,player_name\n1,Fabricated One\n", encoding="utf-8")
    with pytest.raises(AnchorError, match="missing required column"):
        load_sleeper_csv(p)


def test_a_conflicting_duplicate_player_id_is_refused():
    with pytest.raises(AnchorError, match="appears twice with different values"):
        AnchorBook((anchor(7, "Fabricated One", "WR", "5"),
                    anchor(7, "Fabricated One", "WR", "6")), source_sha256="x")


def test_an_identical_duplicate_row_is_tolerated():
    book = AnchorBook((anchor(7, "Fabricated One", "WR", "5"),
                       anchor(7, "Fabricated One", "WR", "5")), source_sha256="x")
    assert len(book.by_sleeper_id) == 1


def test_a_round_trip_through_the_csv_loader(tmp_path):
    p = tmp_path / "a.csv"
    p.write_text(
        'sleeper_player_id,player_name,position,nfl_team,active,sleeper_2qb_raw_value\n'
        '1,Fabricated One,WR,ZZA,true,16.50\n'
        '2,Fabricated Two,TE,ZZB,false,4.00\n', encoding="utf-8")
    book = load_sleeper_csv(p)
    assert len(book) == 2
    assert book.by_sleeper_id["1"].display_anchor == 17
    assert book.by_sleeper_id["2"].status == "inactive"
    assert book.source_sha256


# ==========================================================================
# Identity join
# ==========================================================================


def _contract(*rows):
    return [{"player_key": k, "name": n, "position": p} for k, n, p in rows]


def test_anchors_match_the_contract_by_canonical_key():
    book = AnchorBook((anchor(1, "A.J. Fabricated Jr.", "WR", "10"),
                       anchor(2, "Fabricated Two", "RB", "8")), source_sha256="x")
    matched, report = match_anchors_to_contract(
        book, _contract(("aj_fabricated", "AJ Fabricated", "WR"),
                        ("fabricated_two", "Fabricated Two", "RB")))
    assert len(matched) == 2
    assert report.matched == 2 and report.matched_priced == 2
    assert report.match_rate == 1.0


def test_an_unmatched_contract_player_is_reported_not_guessed():
    book = AnchorBook((anchor(1, "Fabricated One", "WR", "10"),), source_sha256="x")
    matched, report = match_anchors_to_contract(
        book, _contract(("fabricated_one", "Fabricated One", "WR"),
                        ("fabricated_ghost", "Fabricated Ghost", "RB")))
    assert len(matched) == 1
    assert report.unmatched_contract == ["fabricated_ghost"]


def test_an_ambiguous_anchor_name_is_left_unmatched():
    """Two Sleeper rows on one key: an arbitrary pick would be a guess."""
    book = AnchorBook((anchor(1, "Fabricated One", "WR", "10"),
                       anchor(2, "Fabricated One", "RB", "9")), source_sha256="x")
    matched, report = match_anchors_to_contract(
        book, _contract(("fabricated_one", "Fabricated One", "WR")))
    assert matched == {}
    assert report.ambiguous and report.ambiguous[0][0] == "fabricated_one"
    assert report.duplicate_anchor_names


def test_a_position_conflict_is_reported_rather_than_joined():
    book = AnchorBook((anchor(1, "Fabricated One", "TE", "10"),), source_sha256="x")
    matched, report = match_anchors_to_contract(
        book, _contract(("fabricated_one", "Fabricated One", "WR")))
    assert matched == {}
    assert report.position_conflicts == [("fabricated_one", "WR", "TE")]


def test_unusable_anchors_are_counted_by_reason():
    book = AnchorBook((anchor(1, "Fabricated One", "WR", "0"),
                       anchor(2, "Fabricated Two", "RB", "5", active=False),
                       anchor(3, "Fabricated Kick", "K", "5")), source_sha256="x")
    matched, report = match_anchors_to_contract(
        book, _contract(("fabricated_one", "Fabricated One", "WR"),
                        ("fabricated_two", "Fabricated Two", "RB"),
                        ("fabricated_kick", "Fabricated Kick", "K")))
    assert report.nonpositive == 1
    assert report.inactive == 1
    assert report.ineligible_position == 1
    assert report.matched_priced == 0


# ==========================================================================
# The room's money
# ==========================================================================


def test_the_room_budget_is_this_league_exactly():
    b = RoomBudget(DEFAULT_LEAGUE)
    assert b.nominal_total == 2400
    assert b.roster_slots == 180
    assert b.minimum_committed == 180
    assert b.discretionary == 2220


def test_a_scenario_cannot_spend_more_than_the_room_has():
    b = RoomBudget()
    assert b.spendable(1.0) == 2220
    with pytest.raises(ValueError):
        b.spendable(1.5)
    with pytest.raises(ValueError):
        b.spendable(0.0)


def test_no_scenario_board_overspends_its_room_budget(demo_prior):
    """The whole reason a reconciliation step is mandatory rather than optional."""
    for name in ("low", "base", "high"):
        r = demo_prior.reconciliation[name]
        assert r["within_integer_rounding"], (
            f"{name} board ${r['achieved_board_total']} exceeds "
            f"${r['spendable_discretionary']} by more than integer rounding")
        assert r["achieved_board_total"] <= r["spendable_discretionary"] + \
            len(demo_prior.draftable) / 2


def test_the_priced_anchors_alone_exceed_the_discretionary_pool(demo_prior):
    """Which is why paying list is impossible, not merely unlikely."""
    b = RoomBudget()
    raw_total = float(sum(p.raw_value for p in demo_prior.draftable))
    assert raw_total > b.discretionary


def test_no_transformed_price_falls_below_the_legal_minimum(demo_prior):
    for p in demo_prior.draftable:
        for value in (p.low_price, p.base_price, p.high_price):
            assert value >= DEFAULT_LEAGUE.min_bid


def test_an_unpriced_player_is_not_a_predicted_dollar_sale(demo_prior):
    """Two different claims, and the model keeps them apart."""
    unpriced = [p for p in demo_prior.players if not p.draftable]
    assert unpriced
    for p in unpriced:
        assert p.base_price == 0
        assert "not the same as a predicted $1 sale" in p.notes


# ==========================================================================
# The format adjustment
# ==========================================================================


def test_the_transformation_is_monotone_within_a_position(demo_prior):
    """A reconciliation may not reorder players at a position."""
    for pos in ("QB", "RB", "WR", "TE"):
        group = sorted((p for p in demo_prior.draftable if p.position == pos),
                       key=lambda p: p.raw_value)
        prices = [p.base_price for p in group]
        assert prices == sorted(prices), f"{pos} was reordered"


def test_the_prior_is_deterministic():
    a, b = build_demo_prior(), build_demo_prior()
    assert a.fingerprint() == b.fingerprint()
    assert [p.base_price for p in a.players] == [p.base_price for p in b.players]


def test_there_is_no_dedicated_te_requirement_or_te_premium(demo_prior):
    """No quota, no bonus. The tight end's treatment comes from the slot rules."""
    assert demo_prior.credibility["TE"].weight < demo_prior.credibility["RB"].weight
    assert "NO dedicated TE slot" in demo_prior.credibility["TE"].reason
    root = REPO / "src" / "ceauction" / "market"
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8").lower()
        for banned in ("te_premium", "te_penalty", "te_discount",
                       "positional_premium", "scarcity_multiplier"):
            assert banned not in text, f"{path.name} contains {banned!r}"


def test_the_te_anchor_is_preserved_even_though_credibility_is_low(demo_prior):
    """Low credibility is not deletion, and the source stays inspectable."""
    tes = [p for p in demo_prior.draftable if p.position == "TE"]
    assert tes
    for p in tes:
        assert p.raw_value is not None and p.raw_value > 0
        assert p.display_anchor is not None
        assert p.credibility.weight > 0.0, "the anchor is discounted, not zeroed"
    assert demo_prior.credibility["TE"].learn_rate > \
        demo_prior.credibility["RB"].learn_rate, (
            "the least credible anchor must be the fastest to learn about")


def test_generic_2qb_is_not_treated_as_strict_two_required_quarterbacks(demo_prior):
    """A non-QB may take the superflex, so 2qb demand overstates the floor here."""
    qb = demo_prior.credibility["QB"]
    assert qb.weight < demo_prior.credibility["RB"].weight
    assert "superflex" in qb.reason
    assert qb.weight > 0.0, "the dedicated QB slot is real; this is not a deletion"


def test_credibility_is_derived_from_the_lineup_and_not_hand_set(demo_prior):
    """Every position's reason cites the slot structure it comes from."""
    for pos, cred in demo_prior.credibility.items():
        assert cred.reason
        assert any(w in cred.reason for w in ("slot", "flex", "seats"))


def test_the_summary_never_calls_a_scenario_fitted(demo_prior):
    s = demo_prior.summary()
    for sc in s["scenarios"].values():
        assert "not fitted" in sc["basis"]
    text = format_prior_summary(demo_prior)
    assert "STATED SCENARIO" in text
    assert "NOT championship equity" in text or "not championship equity" in text


# ==========================================================================
# The live updater
# ==========================================================================


def _sale(key, pos, price, buyer, seq, base, anchor_display=None):
    return SaleObservation(key, pos, price, buyer, seq, base, anchor_display)


def _probe(prior, pos, rank=8):
    group = sorted((p for p in prior.draftable if p.position == pos),
                   key=lambda p: -p.base_price)
    return group[rank]


def test_a_position_run_moves_that_position_and_not_an_unrelated_one(demo_prior):
    """Room and position effects are collinear until sales span positions."""
    state = MarketState(prior=demo_prior)
    te_probe = _probe(demo_prior, "TE").canonical_key
    qb_probe = _probe(demo_prior, "QB").canonical_key
    before_te = state.adjusted(te_probe).base
    before_qb = state.adjusted(qb_probe).base

    tes = sorted((p for p in demo_prior.draftable if p.position == "TE"),
                 key=lambda p: -p.base_price)[:6]
    sales = [_sale(p.canonical_key, "TE", max(1, int(p.base_price * 0.45)),
                   f"Owner{i % 12 + 1:02d}", i, p.base_price, p.display_anchor)
             for i, p in enumerate(tes)]
    after = state.observe_all(sales)

    assert after.adjusted(te_probe).base < before_te
    assert after.adjusted(qb_probe).base == before_qb, (
        "tight-end evidence must not move an unobserved position")
    assert after.room_effect.n == 0, "one position cannot identify a room effect"


def test_a_room_effect_appears_once_sales_span_positions(demo_prior):
    state = MarketState(prior=demo_prior)
    tes = sorted((p for p in demo_prior.draftable if p.position == "TE"),
                 key=lambda p: -p.base_price)[:4]
    wrs = sorted((p for p in demo_prior.draftable if p.position == "WR"),
                 key=lambda p: -p.base_price)[:4]
    sales = [_sale(p.canonical_key, p.position, max(1, int(p.base_price * 0.5)),
                   "Owner01", i, p.base_price, p.display_anchor)
             for i, p in enumerate(tes + wrs)]
    after = state.observe_all(sales)
    assert after.room_effect.n > 0
    assert after.room_effect.multiplier < 1.0
    assert after.to_dict()["distinct_positions_observed"] == 2


def test_one_outlier_cannot_dominate(demo_prior):
    """A $200 sale on a $3 prior is a 66x miss; the cap keeps it from deciding."""
    state = MarketState(prior=demo_prior)
    cheap = min((p for p in demo_prior.draftable if p.position == "TE"),
                key=lambda p: p.base_price)
    after = state.observe(_sale(cheap.canonical_key, "TE", 200, "Owner01", 0, 3, 3))
    mult = after.position_levels["TE"].multiplier
    assert mult < math.exp(state.config.log_ratio_cap) + 1e-9
    assert mult < 1.4, f"an absurd sale moved the position by {mult:.2f}x"


def test_one_sale_cannot_create_certainty(demo_prior):
    state = MarketState(prior=demo_prior)
    probe = _probe(demo_prior, "WR")
    after = state.observe(_sale(probe.canonical_key, "WR", probe.base_price,
                                "Owner01", 0, probe.base_price,
                                probe.display_anchor))
    adj = after.adjusted(probe.canonical_key)
    assert adj.uncertainty >= state.config.uncertainty_floor
    assert adj.high > adj.low, "a band cannot collapse to a point on one sale"


def test_uncertainty_narrows_with_evidence_but_never_past_the_floor(demo_prior):
    state = MarketState(prior=demo_prior)
    probe = _probe(demo_prior, "WR")
    first = state.adjusted(probe.canonical_key).uncertainty
    wrs = sorted((p for p in demo_prior.draftable if p.position == "WR"),
                 key=lambda p: -p.base_price)[:20]
    after = state.observe_all(
        [_sale(p.canonical_key, "WR", p.base_price, f"Owner{i % 12 + 1:02d}", i,
               p.base_price, p.display_anchor) for i, p in enumerate(wrs)])
    later = after.adjusted(probe.canonical_key).uncertainty
    assert later <= first
    assert later >= state.config.uncertainty_floor


def test_te_credibility_falls_after_te_discounts_and_the_anchor_survives(demo_prior):
    state = MarketState(prior=demo_prior)
    before = state.anchor_credibility("TE")
    tes = sorted((p for p in demo_prior.draftable if p.position == "TE"),
                 key=lambda p: -p.base_price)[:5]
    after = state.observe_all(
        [_sale(p.canonical_key, "TE", max(1, int(p.base_price * 0.45)),
               f"Owner{i % 12 + 1:02d}", i, p.base_price, p.display_anchor)
         for i, p in enumerate(tes)])
    assert after.anchor_credibility("TE") < before
    assert after.anchor_credibility("QB") == state.anchor_credibility("QB")
    # The anchor itself is untouched.
    for p in tes:
        still = after.prior.by_key[p.canonical_key]
        assert still.raw_value == p.raw_value
        assert still.display_anchor == p.display_anchor


def test_credibility_can_rise_again_when_the_room_pays_the_anchor(demo_prior):
    """A discount is evidence, not a verdict; paying list is evidence too."""
    state = MarketState(prior=demo_prior)
    tes = sorted((p for p in demo_prior.draftable if p.position == "TE"),
                 key=lambda p: -p.base_price)[:5]
    discounted = state.observe_all(
        [_sale(p.canonical_key, "TE", max(1, int(p.base_price * 0.4)),
               "Owner01", i, p.base_price, p.display_anchor)
         for i, p in enumerate(tes)])
    at_anchor = state.observe_all(
        [_sale(p.canonical_key, "TE", p.display_anchor, "Owner01", i,
               p.base_price, p.display_anchor) for i, p in enumerate(tes)])
    assert discounted.anchor_credibility("TE") < at_anchor.anchor_credibility("TE")


def test_a_buyer_is_not_characterised_by_a_single_purchase(demo_prior):
    state = MarketState(prior=demo_prior)
    probe = _probe(demo_prior, "RB")
    one = state.observe(_sale(probe.canonical_key, "RB", probe.base_price * 2,
                              "Owner04", 0, probe.base_price))
    assert one.adjusted(probe.canonical_key, buyer="Owner04").buyer_multiplier is None
    rbs = sorted((p for p in demo_prior.draftable if p.position == "RB"),
                 key=lambda p: -p.base_price)[:3]
    three = state.observe_all(
        [_sale(p.canonical_key, "RB", int(p.base_price * 1.5), "Owner04", i,
               p.base_price) for i, p in enumerate(rbs)])
    adj = three.adjusted(probe.canonical_key, buyer="Owner04")
    assert adj.buyer_multiplier is not None and adj.buyer_multiplier > 1.0


def test_additive_residuals_carry_players_a_ratio_cannot(demo_prior):
    """A ratio against a $1 prior is arithmetic noise; the dollar miss is not."""
    cheap = SaleObservation("k", "WR", 4, "Owner01", 0, prior_base=1)
    assert cheap.log_ratio is None
    assert cheap.additive_residual == 3.0
    rich = SaleObservation("k", "WR", 40, "Owner01", 0, prior_base=20)
    assert rich.log_ratio == pytest.approx(math.log(2.0))


def test_a_missing_anchor_is_handled_explicitly(demo_prior):
    state = MarketState(prior=demo_prior)
    assert state.adjusted("no-such-player") is None
    unpriced = next(p for p in demo_prior.players if not p.draftable)
    assert state.adjusted(unpriced.canonical_key) is None
    no_prior = SaleObservation("k", "WR", 12, "Owner01", 0, prior_base=None)
    assert no_prior.additive_residual is None and no_prior.log_ratio is None
    state.observe(no_prior)  # must not raise


def test_the_updater_is_deterministic(demo_prior):
    sales = DEMO_SALES(demo_prior)
    a = MarketState(prior=demo_prior).observe_all(sales)
    b = MarketState(prior=demo_prior).observe_all(sales)
    assert a.fingerprint() == b.fingerprint()
    assert a.serialize() == b.serialize()


def test_the_state_round_trips_through_serialization(demo_prior):
    original = MarketState(prior=demo_prior).observe_all(DEMO_SALES(demo_prior))
    restored = MarketState.restore(demo_prior, original.serialize())
    assert restored.fingerprint() == original.fingerprint()


def test_replaying_observations_onto_a_different_prior_is_refused(demo_prior):
    original = MarketState(prior=demo_prior).observe_all(DEMO_SALES(demo_prior))
    other = build_market_prior(
        AnchorBook((anchor(1, "Fabricated Solo", "WR", "20"),), source_sha256="y"))
    with pytest.raises(ValueError, match="were recorded against prior"):
        MarketState.restore(other, original.serialize())


def test_a_sale_price_must_be_a_legal_whole_dollar():
    with pytest.raises(ValueError, match="whole auction dollars"):
        SaleObservation("k", "WR", 12.5, "Owner01", 0)
    with pytest.raises(ValueError, match="at least the \\$1 minimum"):
        SaleObservation("k", "WR", 0, "Owner01", 0)


def test_the_state_says_what_a_sale_does_not_reveal(demo_prior):
    state = MarketState(prior=demo_prior).observe_all(DEMO_SALES(demo_prior))
    text = state.to_dict()["what_a_sale_reveals"]
    assert "does NOT reveal the winner's maximum" in text
    assert "losing bidder's maximum" in text


def test_shrinkage_configuration_is_reported_and_participates_in_identity(demo_prior):
    a = MarketState(prior=demo_prior, config=ShrinkageConfig())
    b = MarketState(prior=demo_prior, config=ShrinkageConfig(position=2.0))
    assert a.fingerprint() != b.fingerprint()
    assert a.to_dict()["config"]["position"] == 5.0


def test_tiers_partition_the_price_range():
    assert tier_of(50) == "elite"
    assert tier_of(20) == "mid"
    assert tier_of(6) == "depth"
    assert tier_of(1) == "dollar"


# ==========================================================================
# Room pressure
# ==========================================================================


def test_financial_ability_and_candidate_legality_are_separate(demo_prior):
    """An owner who can afford a player may still not be allowed to hold him.

    Nine quarterbacks, three backs and two receivers is fourteen legal players
    whose fifteenth *must* be a WR or TE: three WR/TE are required and only two
    are held. He has $186 and a slot, so money is not the constraint.
    """
    from ceauction.auction import new_auction
    from ceauction.players import PlayerSpec

    owners = tuple(f"Owner{i + 1:02d}" for i in range(12))
    pool, pid = [], 0
    for pos, n in ((Position.QB, 60), (Position.RB, 60), (Position.WR, 60),
                   (Position.TE, 40)):
        for k in range(n):
            pool.append(PlayerSpec(
                player_id=pid, name=f"Fabricated{pid:04d}", position=pos,
                nfl_team="ZZA", base_mean=12.0 - 0.05 * k, week_sd=5.0,
                bye_week=5 + (k % 10), data_source="FABRICATED:market-test"))
            pid += 1
    state = new_auction(pool, owners, "Owner01")
    owner = "Owner03"
    by_pos = {p: [s.player_id for s in pool if s.position is p] for p in Position}
    cur = {p: 0 for p in Position}

    def take(pos, n):
        nonlocal state
        for _ in range(n):
            state = state.apply_purchase(by_pos[pos][cur[pos]], owner, 1)
            cur[pos] += 1

    take(Position.QB, 9)
    take(Position.RB, 3)
    take(Position.WR, 2)
    o = state.owner(owner)
    assert o.n_players == 14 and o.open_slots == 1 and o.can_still_field_lineup()

    qb = by_pos[Position.QB][cur[Position.QB]]
    wr = by_pos[Position.WR][cur[Position.WR]]

    on_qb = assess_room_pressure(state, qb, 1)
    stranded = next(x for x in on_qb.owners if x.owner_id == owner)
    assert stranded.financial_max_bid > 0, "he can afford it"
    assert stranded.candidate_legal_max_bid == 0, "but not legally hold him"
    assert not stranded.can_bid_at_price
    assert "legal roster" in stranded.blocked_reason
    assert stranded not in on_qb.competitors

    # The same owner, same money, a player who fills the gap: now legal.
    on_wr = assess_room_pressure(state, wr, 1)
    ok = next(x for x in on_wr.owners if x.owner_id == owner)
    assert ok.can_bid_at_price
    assert ok.candidate_legal_max_bid == ok.financial_max_bid


def test_a_purchase_changes_opponent_capacity():
    auction = build_demo_auction()
    st0 = auction.state
    cand = auction.default_candidate().player_id
    other = next(p for p in st0.available_ids if p != cand)
    st1 = st0.apply_purchase(other, "Owner02", 55)
    p0 = assess_room_pressure(st0, cand, 40)
    p1 = assess_room_pressure(st1, cand, 40)
    o0 = next(o for o in p0.owners if o.owner_id == "Owner02")
    o1 = next(o for o in p1.owners if o.owner_id == "Owner02")
    assert o1.budget_remaining == o0.budget_remaining - 55
    assert o1.financial_max_bid < o0.financial_max_bid
    assert o1.open_slots == o0.open_slots - 1


def test_room_pressure_never_claims_to_pick_a_winner():
    auction = build_demo_auction()
    from ceauction.market import format_room_pressure
    p = assess_room_pressure(auction.state, auction.default_candidate().player_id, 20)
    text = format_room_pressure(p)
    assert "nothing here predicts who wins" in text
    assert "STRUCTURAL" in text
    assert "anyone's maximum" in text
    assert "does NOT predict the winner" in p.to_dict()["label"]


def test_roster_fit_uses_the_real_slot_rules():
    auction = build_demo_auction()
    p = assess_room_pressure(auction.state, auction.default_candidate().player_id, 20)
    fits = {o.roster_fit for o in p.owners}
    assert fits, "every owner gets a structural read"
    assert all(0.0 <= o.fit_score <= 1.0 for o in p.owners)


# ==========================================================================
# CostBook integration
# ==========================================================================


def test_a_cost_book_is_provisional_and_never_real(demo_prior):
    book = cost_book_from_prior(demo_prior, "base")
    assert book.level == "PROVISIONAL"
    assert not book.provenance.may_be_reported_as_a_value
    assert "GENERIC Sleeper" in book.provenance.notes
    assert "separate axes" in book.provenance.notes


def test_the_market_axis_and_the_model_axis_are_both_recorded(demo_prior):
    book = cost_book_from_prior(demo_prior, "high",
                                model_scenario="aa-f050-s020-n0")
    assert "market=high" in book.provenance.scenario_id
    assert "model=aa-f050-s020-n0" in book.provenance.scenario_id


def test_each_market_scenario_gives_a_different_cost_book(demo_prior):
    books = {s: cost_book_from_prior(demo_prior, s) for s in ("low", "base", "high")}
    prints = {s: b.fingerprint() for s, b in books.items()}
    assert len(set(prints.values())) == 3
    assert books["low"].total_assumed_cost < books["high"].total_assumed_cost


def test_a_sale_changes_the_cost_book_fingerprint(demo_prior):
    before = cost_book_from_prior(demo_prior, "base")
    state = MarketState(prior=demo_prior).observe_all(DEMO_SALES(demo_prior))
    after = cost_book_from_market_state(state, "base")
    assert before.fingerprint() != after.fingerprint()


def test_prior_strength_changes_the_cost_book_fingerprint(demo_prior):
    sales = DEMO_SALES(demo_prior)
    a = cost_book_from_market_state(
        MarketState(prior=demo_prior, config=ShrinkageConfig()).observe_all(sales),
        "base")
    b = cost_book_from_market_state(
        MarketState(prior=demo_prior,
                    config=ShrinkageConfig(position=2.0)).observe_all(sales),
        "base")
    assert a.fingerprint() != b.fingerprint()


def test_an_unpriced_player_is_omitted_rather_than_priced_at_a_dollar(demo_prior):
    book = cost_book_from_prior(demo_prior, "base")
    unpriced = next(p for p in demo_prior.players if not p.draftable)
    assert unpriced.player_id not in book.by_id
    with pytest.raises(MissingCost):
        book.cost_of(unpriced.player_id)


def test_a_bad_scenario_name_is_refused(demo_prior):
    with pytest.raises(ValueError, match="unknown market scenario"):
        cost_book_from_prior(demo_prior, "medium")


def test_the_existing_fabricated_auction_demo_still_works():
    """Code paths with no Sleeper anchor must be untouched."""
    auction = build_demo_auction()
    auction.state.validate()
    assert auction.costs.level == "FABRICATED"
    assert len(auction.costs) > 0
