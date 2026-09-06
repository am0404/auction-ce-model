"""Estimator power by candidate tier, and the sampling plan. Fabricated only.

The prior null was one weak candidate's true-zero effect, not an estimator
failure. These tests keep those two apart: tiers are calibrated by measured
lineup consequences, and the sampling plan is pilot-then-confirmatory with
different seeds so a confirmatory interval never reuses the sample that
triggered it.
"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from ceauction.auction.proxy import ProxyEvaluator
from ceauction.tactical.power_experiment import verdict
from ceauction.tactical.signal import (CANDIDATE_TIERS, HALF_WIDTH_TARGETS,
                                       TIER_PRICE_LADDERS, allocation_spread,
                                       build_tiered_world, calibrate_tier,
                                       plan_confirmatory, required_sample)

_TIERS = ("bench", "marginal_starter", "strong_starter", "elite")
_CTX = "playoff_bubble"


@pytest.fixture(scope="module")
def worlds():
    return {t: build_tiered_world(_CTX, t) for t in _TIERS}


@pytest.fixture(scope="module")
def calib(worlds):
    out = {}
    for t, (d, cid) in worlds.items():
        px = ProxyEvaluator(d.state.pool, d.state.settings, 96, 7)
        out[t] = calibrate_tier(d, cid, t, _CTX, proxy=px)
    return out


# ---------------------------------------------------------------------------
# Tiers vary one parameter and nothing else
# ---------------------------------------------------------------------------


def test_tiers_differ_only_in_candidate_base_mean(worlds):
    ref_d, cid = worlds["bench"]
    ref = {s.player_id: s for s in ref_d.state.pool}
    means = {}
    for t, (d, other) in worlds.items():
        assert other == cid, "the candidate id must not move between tiers"
        for s in d.state.pool:
            if s.player_id == cid:
                means[t] = s.base_mean
                continue
            assert s == ref[s.player_id], (t, s.player_id)
    assert len(set(means.values())) == 4
    # The candidate keeps every non-scoring attribute.
    specs = {t: d.state.spec(cid) for t, (d, _) in worlds.items()}
    for attr in ("position", "bye_week", "week_sd", "weekly_injury_hazard",
                 "injury_mean_weeks", "nfl_team"):
        assert len({getattr(s, attr) for s in specs.values()}) == 1, attr


def test_auction_structure_is_identical_across_tiers(worlds):
    keys = set()
    for d, cid in worlds.values():
        st = d.state
        focus = st.focus_owner_id
        o = st.owner(focus)
        keys.add((
            o.budget_remaining, o.spent, o.n_players, o.open_slots,
            tuple(sorted(st.available_ids)),
            tuple(sorted((r.owner_id, r.player_ids, r.spent, r.budget_remaining)
                         for r in st.owners if r.owner_id != focus)),
            d.market.fingerprint(), d.costs.fingerprint()))
    assert len(keys) == 1


def test_an_unknown_tier_is_refused():
    with pytest.raises(ValueError, match="unknown candidate tier"):
        build_tiered_world(_CTX, "superstar")


# ---------------------------------------------------------------------------
# Labels earned from measured lineup roles
# ---------------------------------------------------------------------------


def test_bench_candidate_stays_benched(calib):
    c = calib["bench"]
    assert not c.starts
    assert c.lineup_improvement < 0.25
    assert c.displaced_player_id is None


def test_strong_starter_displaces_a_starter(calib):
    c = calib["strong_starter"]
    assert c.starts
    assert c.displaced_player_id is not None
    assert c.lineup_improvement > 3.0


def test_elite_creates_the_largest_lineup_improvement(calib):
    gains = {t: calib[t].lineup_improvement for t in _TIERS}
    assert gains["elite"] == max(gains.values())
    ordered = [gains[t] for t in _TIERS]
    assert ordered == sorted(ordered), gains


def test_marginal_starter_sits_on_the_boundary(calib):
    c = calib["marginal_starter"]
    assert 0.0 < c.lineup_improvement < calib["strong_starter"].lineup_improvement


def test_no_hard_coded_ce_tier_premium():
    import pathlib
    src = pathlib.Path(__file__).resolve().parents[1] / "src" / "ceauction"
    # Narrowly a CE-side check. `market/live.py` legitimately carries a
    # `tier_multiplier` for expected CLEARING PRICES, which is a market
    # quantity and not a championship-equity adjustment; banning the token
    # outright would flag the market layer for doing its job.
    banned = ("tier_premium", "elite_bonus", "starter_premium",
              "elite_ce_bonus", "ce_premium", "ce_tier_bonus")
    for path in src.rglob("*.py"):
        if path.parts[-2] == "market":
            continue
        text = path.read_text(encoding="utf-8")
        for word in banned:
            assert word not in text, f"{word} in {path.name}"
    # A tier may only carry a scoring scale, never a CE adjustment.
    import dataclasses
    fields = {f.name for f in dataclasses.fields(CANDIDATE_TIERS["elite"])}
    assert fields == {"name", "description", "scale"}


# ---------------------------------------------------------------------------
# Sample-size arithmetic
# ---------------------------------------------------------------------------


def test_required_sample_scales_with_the_square_of_the_ratio():
    # se 0.01 at n=1000 -> half-width 0.0196. Halving the target quadruples n.
    n1 = required_sample(0.01, 1000, 0.0196)
    n2 = required_sample(0.01, 1000, 0.0098)
    assert n1 == pytest.approx(1000, abs=2)
    assert n2 == pytest.approx(4 * n1, rel=0.02)


def test_required_sample_never_shrinks_below_the_pilot():
    assert required_sample(0.0001, 2000, 0.5) == 2000
    assert required_sample(0.0, 2000, 0.001) == 2000
    assert required_sample(float("nan"), 2000, 0.001) == 2000


def test_required_sample_rejects_nonsense():
    with pytest.raises(ValueError):
        required_sample(0.01, 1, 0.005)
    with pytest.raises(ValueError):
        required_sample(0.01, 1000, 0.0)


def test_pilot_and_confirmatory_must_use_different_seeds():
    with pytest.raises(ValueError, match="different seeds"):
        plan_confirmatory(0.01, 0.004, 1000, pilot_seed=5, confirmatory_seed=5)


def test_cap_produces_underpowered_at_cap():
    tight = plan_confirmatory(0.0005, 0.004, 1000, target_half_width=0.0005,
                              cap_n=10_000, pilot_seed=1, confirmatory_seed=2)
    assert tight.status == "UNDERPOWERED_AT_CAP"
    assert not tight.affordable
    assert tight.required_n > tight.cap_n

    loose = plan_confirmatory(0.05, 0.004, 1000, target_half_width=0.02,
                              cap_n=10_000, pilot_seed=1, confirmatory_seed=2)
    assert loose.status == "READY" and loose.affordable


def test_the_plan_reports_every_half_width_target():
    plan = plan_confirmatory(0.01, 0.004, 1000, pilot_seed=1,
                             confirmatory_seed=2)
    assert set(plan.requirements) == {str(t) for t in HALF_WIDTH_TARGETS}
    needs = [plan.requirements[str(t)] for t in HALF_WIDTH_TARGETS]
    assert needs == sorted(needs), "a tighter target must need more seasons"
    blob = plan.to_dict()
    assert blob["pilot_seed"] != blob["confirmatory_seed"]
    assert "reuses no pilot observation" in blob["note"]


# ---------------------------------------------------------------------------
# Allocation uncertainty is reported separately
# ---------------------------------------------------------------------------


def test_allocation_spread_separates_the_two_uncertainties():
    sp = allocation_spread([1, 2, 3], [0.50, 0.55, 0.51],
                           [0.02, 0.02, 0.02], ["a", "b", "c"])
    assert sp.distinct_allocations == 3
    assert sp.between_sd > 0.0
    assert sp.mean_within_se == pytest.approx(0.02)
    blob = sp.to_dict()
    assert "dominated_by_allocation" in blob and "dominates_effect" in blob
    assert "never pooled" in blob["note"]


def test_a_large_effect_is_not_called_unstable_merely_for_cheap_seasons():
    """The distinction the verdict rule depends on."""
    sp = allocation_spread([1, 2, 3], [0.50, 0.55, 0.51],
                           [0.008, 0.008, 0.008], ["a", "b", "c"])
    assert sp.dominated_by_allocation, "allocation noise does exceed season noise"
    assert not sp.dominates_effect, "but it is tiny against a 0.5 effect"
    assert sp.sign_stable


def test_a_sign_flipping_effect_is_unstable():
    sp = allocation_spread([1, 2, 3], [0.01, -0.02, 0.005],
                           [0.004, 0.004, 0.004], ["a", "b", "c"])
    assert not sp.sign_stable
    assert sp.dominates_effect


def test_allocation_spread_requires_one_entry_per_seed():
    with pytest.raises(ValueError, match="one entry per seed"):
        allocation_spread([1, 2], [0.1], [0.01], ["a"])


# ---------------------------------------------------------------------------
# Verdict follows the evidence
# ---------------------------------------------------------------------------


def _blob(power, spreads):
    return {"power": power, "allocation_spread": spreads}


_STABLE = {"dominates_effect": False, "sign_stable": True}
_UNSTABLE = {"dominates_effect": True, "sign_stable": False}


def test_verdict_is_no_go_when_nothing_large_resolves():
    v, _ = verdict(_blob(
        [{"tier": t, "verdict": "unresolved"} for t in _TIERS],
        {"strong_starter": _STABLE, "elite": _STABLE}))
    assert v == "NO-GO"


def test_verdict_is_no_go_when_allocation_swamps_a_large_effect():
    v, why = verdict(_blob(
        [{"tier": "elite", "verdict": "favorable"},
         {"tier": "strong_starter", "verdict": "favorable"}],
        {"strong_starter": _STABLE, "elite": _UNSTABLE}))
    assert v == "NO-GO" and "instability" in why


def test_verdict_is_partial_go_when_only_large_effects_resolve():
    v, why = verdict(_blob(
        [{"tier": "elite", "verdict": "favorable"},
         {"tier": "strong_starter", "verdict": "favorable"},
         {"tier": "bench", "verdict": "unresolved"},
         {"tier": "marginal_starter", "verdict": "unresolved"}],
        {"strong_starter": _STABLE, "elite": _STABLE}))
    assert v == "PARTIAL GO" and "proxy-price" in why


def test_verdict_is_go_when_everything_resolves():
    v, _ = verdict(_blob(
        [{"tier": t, "verdict": "favorable"} for t in _TIERS],
        {"strong_starter": _STABLE, "elite": _STABLE}))
    assert v == "GO"


def test_a_weak_candidate_returning_unresolved_does_not_force_no_go():
    """The correction this branch exists for."""
    v, _ = verdict(_blob(
        [{"tier": "bench", "verdict": "unresolved"},
         {"tier": "marginal_starter", "verdict": "unresolved"},
         {"tier": "strong_starter", "verdict": "favorable"},
         {"tier": "elite", "verdict": "favorable"}],
        {"strong_starter": _STABLE, "elite": _STABLE}))
    assert v == "PARTIAL GO"


def test_tier_price_ladders_are_tier_appropriate():
    assert set(TIER_PRICE_LADDERS) == set(_TIERS)
    tops = [max(TIER_PRICE_LADDERS[t]) for t in _TIERS]
    assert tops == sorted(tops)
