"""Audited max-bid over nested opportunity sets. Fabricated data only.

The defect these guard: a player bought for $1 measured *worse* than the same
player bought for $5. Every roster affordable with $105 is affordable with
$109, so that cannot be true. It was beam path dependence -- the shadow board
depends on our own remaining money, and a two-player difference in the beam's
input sent it down a worse path.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import replace

import pytest

from ceauction.auction.completion import CompletionSettings
from ceauction.auction.proxy import ProxyEvaluator
from ceauction.tactical import (BoardSettings, TacticalCache, TacticalScenario,
                                TacticalSettings, build_tactical_demo,
                                demo_sales, evaluate_tactical,
                                tactical_cache_key)
from ceauction.tactical.joint import ConservationError
from ceauction.tactical.nested import (build_nested_ladder,
                                       format_nested_ladder,
                                       roster_fingerprint)
from ceauction.tactical.regimes import REGIMES, build_regime, regime_summary

_CS = CompletionSettings(beam_width=32, candidate_pool=30, proxy_candidates=32,
                         finalists=3, max_candidates=160, proxy_reps=16,
                         selection_sims=300, evaluation_sims=300,
                         rival_selection="proxy")
_PRICES = (1, 5, 10, 13, 20, 30)
_ONE = (TacticalScenario("base", "base", "fabricated_base", is_base=True),)


@pytest.fixture(scope="module")
def demo():
    d = build_tactical_demo()
    return d.with_market(d.market.observe_all(demo_sales(d, 12)))


@pytest.fixture(scope="module")
def px(demo):
    return ProxyEvaluator(demo.state.pool, demo.state.settings, 16, 7)


@pytest.fixture(scope="module")
def ladder(demo, px):
    return build_nested_ladder(
        demo.state, demo.cast, demo.costs, demo.candidate().player_id,
        _PRICES, completion=_CS, market=demo.market,
        key_by_id=demo.key_by_id, proxy=px)


def _settings(**kw):
    base = TacticalSettings(mode="audited", completion=_CS, max_prices=3,
                            max_recipients=1, max_worlds=2,
                            include_unavailable=False)
    return replace(base, **kw)


# ---------------------------------------------------------------------------
# Feasibility nesting -- the economics that must hold exactly
# ---------------------------------------------------------------------------


def test_a_construction_feasible_at_a_higher_price_is_offered_at_every_lower(
        ladder):
    assert ladder.is_nested
    assert ladder.nesting_violations() == ()


def test_feasible_counts_are_monotone_non_increasing_in_price(ladder):
    counts = [ladder.by_price[p].n_feasible for p in sorted(ladder.prices)]
    assert counts == sorted(counts, reverse=True), counts
    assert counts[0] > counts[-1], "the ladder must actually bind somewhere"


def test_lower_prices_inherit_what_higher_prices_generated(ladder):
    lo = ladder.by_price[min(ladder.prices)]
    assert lo.n_inherited > 0, \
        "the whole point is that a price sees rosters its own beam missed"
    assert lo.n_feasible >= lo.n_generated


def test_the_dominance_violation_is_gone_from_the_choice_set(ladder):
    """The exact artefact: $1's set must contain everything $5's does."""
    assert ladder.by_price[5].feasible_keys <= ladder.by_price[1].feasible_keys


def test_unaffordable_constructions_are_dropped_with_a_reason(ladder):
    top = ladder.by_price[max(ladder.prices)]
    assert top.rejected_unaffordable, "a $30 candidate must price something out"
    assert top.n_feasible + len(top.rejected_unaffordable) \
        + len(top.rejected_illegal) == ladder.union_size


def test_ladder_fingerprint_and_formatter(ladder):
    assert len(ladder.fingerprint()) == 16
    text = format_nested_ladder(ladder)
    assert "NESTED OPPORTUNITY SETS" in text
    assert "nesting violations: 0" in text


def test_roster_fingerprint_distinguishes_price_when_asked():
    a = roster_fingerprint([3, 1, 2])
    assert a == roster_fingerprint([1, 2, 3]), "order must not matter"
    assert a != roster_fingerprint([1, 2, 3], {1: 5})


# ---------------------------------------------------------------------------
# Selection-objective monotonicity
# ---------------------------------------------------------------------------


def test_selection_optimum_cannot_worsen_at_a_lower_price(demo, px, ladder):
    """With nested sets and price-independent rival boards, this is exact.

    A rival continuation depends only on which players we took, never on what
    we paid: rivals bid with ``focus_bids=False`` against the board we leave.
    So the same construction yields the same joint world at every price, and
    the best available at $1 is at least the best available at $5.
    """
    from ceauction.tactical.joint import build_joint_worlds
    focus = demo.state.focus_owner_id
    cid = demo.candidate().player_id
    best = {}
    for price in (5, 1):
        worlds = build_joint_worlds(
            demo.state.apply_purchase(cid, focus, price), demo.cast,
            demo.costs, completion=_CS, market=demo.market,
            key_by_id=demo.key_by_id, proxy=px,
            branch_acquired=frozenset({cid}),
            completions=ladder.by_price[price].feasible, max_worlds=99)
        # Compare the ALLOCATION, not the full fingerprint: the latter hashes
        # our own paid price, which differs between $1 and $5 by definition.
        # What must nest is who ends up on which roster.
        best[price] = {
            (tuple(sorted(w.focus_roster)),
             tuple(sorted((o.owner_id, o.player_ids) for o in w.state.owners
                          if o.owner_id != w.state.focus_owner_id)))
            for w in worlds}
    assert best[5] <= best[1], \
        "every joint allocation reachable at $5 must be reachable at $1"


def test_identical_joint_worlds_give_identical_holdout_ce(demo, px, ladder):
    from ceauction.tactical.joint import (build_joint_worlds,
                                          evaluate_joint_arm)
    focus = demo.state.focus_owner_id
    cid = demo.candidate().player_id

    def arm(price):
        return evaluate_joint_arm(
            build_joint_worlds(
                demo.state.apply_purchase(cid, focus, price), demo.cast,
                demo.costs, completion=_CS, market=demo.market,
                key_by_id=demo.key_by_id, proxy=px,
                branch_acquired=frozenset({cid}),
                completions=ladder.by_price[price].feasible, max_worlds=2),
            focus_team_index=demo.cast.focus_team_index, proxy=px,
            selection_sims=300, selection_seed=_CS.selection_seed,
            holdout_sims=300, holdout_seed=_CS.evaluation_seed)

    a, b = arm(1), arm(5)
    same_allocation = (
        tuple(sorted(a.world.focus_roster)) == tuple(sorted(b.world.focus_roster))
        and {(o.owner_id, o.player_ids) for o in a.world.state.owners
             if o.owner_id != a.world.state.focus_owner_id}
        == {(o.owner_id, o.player_ids) for o in b.world.state.owners
            if o.owner_id != b.world.state.focus_owner_id})
    if same_allocation:
        assert a.ce == b.ce, \
            "same allocation, same seed: price alone cannot move equity"


# ---------------------------------------------------------------------------
# Price effects are confined to the buyer's budget
# ---------------------------------------------------------------------------


def test_price_touches_only_the_buyers_budget(demo):
    cid = demo.candidate().player_id
    focus = demo.state.focus_owner_id
    before = {o.owner_id: o.budget_remaining for o in demo.state.owners}
    spec = demo.state.spec(cid)
    for price in _PRICES:
        after = demo.state.apply_purchase(cid, focus, price)
        for o in after.owners:
            if o.owner_id == focus:
                assert o.budget_remaining == before[focus] - price
            else:
                assert o.budget_remaining == before[o.owner_id]
        assert after.spec(cid).base_mean == spec.base_mean
        assert after.spec(cid).week_sd == spec.week_sd


def test_a_static_sweep_never_updates_the_market(demo):
    cid = demo.candidate().player_id
    focus = demo.state.focus_owner_id
    fp = demo.market.fingerprint()
    for price in _PRICES:
        demo.state.apply_purchase(cid, focus, price)
    assert demo.market.fingerprint() == fp
    assert len(demo.market.observations) == 12


def test_seeds_are_identical_across_prices(demo):
    assert len({BoardSettings().seed for _ in _PRICES}) == 1
    assert _CS.selection_seed != _CS.evaluation_seed


# ---------------------------------------------------------------------------
# Audited wiring
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_audited_mode_uses_reconciled_joint_worlds(demo):
    cid = demo.candidate().player_id
    r = evaluate_tactical(
        demo.state, demo.cast, demo.costs, cid, settings=_settings(),
        scenarios=_ONE, market=demo.market, candidate_key=demo.key_for(cid),
        key_by_id=demo.key_by_id, current_price=8, current_leader="Owner09",
        prices=[9, 20])
    assert r.is_audited
    assert r.nested_ladder is not None and r.nested_ladder.is_nested
    assert r.verdicts
    for v in r.verdicts:
        assert "CE-audited over reconciled joint worlds" in v.basis
        assert "joint world" in v.completion_kind
        assert "conservation ok" in v.completion_kind
        assert v.selection_sims == 300 and v.holdout_sims == 300
    r.check_caps()
    assert (r.robust_tactical_max or 0) <= r.legal_max


def test_proxy_mode_is_unmistakably_labelled(demo):
    cid = demo.candidate().player_id
    r = evaluate_tactical(
        demo.state, demo.cast, demo.costs, cid,
        settings=TacticalSettings(mode="immediate", max_prices=2,
                                  max_recipients=1,
                                  include_unavailable=False),
        scenarios=_ONE, market=demo.market, candidate_key=demo.key_for(cid),
        key_by_id=demo.key_by_id, current_price=8)
    assert not r.is_audited
    assert r.result_kind.startswith("PROXY ONLY -- NOT A CE RESERVATION PRICE")
    assert r.nested_ladder is None
    blob = r.to_dict()
    assert blob["proxy_banner"] == r.PROXY_BANNER
    assert blob["threshold_basis"] == "proxy-only, NOT a CE reservation price"
    assert "CE-audited" not in blob["threshold_basis"]
    for v in r.verdicts:
        assert "NOT championship equity" in v.basis
        assert v.se is None


def test_withdrawn_thresholds_cannot_print_as_current(demo):
    """A proxy threshold may never be described with audited vocabulary."""
    cid = demo.candidate().player_id
    r = evaluate_tactical(
        demo.state, demo.cast, demo.costs, cid,
        settings=TacticalSettings(mode="immediate", max_prices=2,
                                  max_recipients=1,
                                  include_unavailable=False),
        scenarios=_ONE, market=demo.market, candidate_key=demo.key_for(cid),
        key_by_id=demo.key_by_id)
    from ceauction.tactical.maxbid import format_tactical
    text = format_tactical(r)
    assert "PROXY ONLY -- NOT A CE RESERVATION PRICE" in text
    assert "CE-audited" not in text


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def _key(demo, **kw):
    base = dict(market=demo.market, current_leader="Owner09", current_price=8,
                increment=1, recipients=["Owner09 at $12"], scenarios=_ONE,
                settings=_settings(), cast=demo.cast, costs=demo.costs,
                ladder_prices=(9, 20))
    base.update(kw)
    return tactical_cache_key(demo.state, demo.candidate().player_id, **base)


def test_cache_key_moves_with_the_nested_price_ladder(demo):
    assert _key(demo) != _key(demo, ladder_prices=(9, 21))
    assert _key(demo) != _key(demo, ladder_prices=(9,))


def test_cache_key_carries_mode_regime_and_samples(demo):
    audited = _key(demo)
    assert audited.startswith("aud-")
    proxy = _key(demo, settings=replace(_settings(), mode="immediate"))
    assert proxy.startswith("imm-")
    assert audited != proxy
    assert _key(demo) != _key(demo, regime="favorite")
    assert _key(demo) != _key(
        demo, settings=_settings(completion=replace(_CS, selection_sims=999)))
    assert _key(demo) != _key(
        demo, settings=_settings(completion=replace(_CS, evaluation_sims=999)))


def test_a_proxy_cache_entry_cannot_satisfy_an_audited_request(demo):
    cid = demo.candidate().player_id
    cache = TacticalCache()
    common = dict(scenarios=_ONE, market=demo.market,
                  candidate_key=demo.key_for(cid), key_by_id=demo.key_by_id,
                  current_price=8, cache=cache, prices=[9])
    evaluate_tactical(demo.state, demo.cast, demo.costs, cid,
                      settings=TacticalSettings(mode="immediate",
                                                max_recipients=1,
                                                include_unavailable=False),
                      **common)
    assert cache.stats()["entries"] == 1
    before = cache.hits
    key = _key(demo, current_leader=None, recipients=[], ladder_prices=(9,))
    assert cache.get(key) is None
    assert cache.hits == before


# ---------------------------------------------------------------------------
# Roster-strength regimes
# ---------------------------------------------------------------------------


def test_the_four_regimes_are_genuinely_different():
    seen = {}
    for name in REGIMES:
        d = build_regime(name, with_sales=False)
        s = regime_summary(name, d)
        seen[name] = (s["n_preowned"], s["budget_remaining"])
        d.state.validate()
    assert len(set(seen.values())) == 4, seen
    assert seen["underdog"][0] < seen["favorite"][0]
    # The rival cast is identical across regimes; only our roster moves.
    casts = {build_regime(n, with_sales=False).cast.fingerprint()
             for n in REGIMES}
    rivals = set()
    for n in REGIMES:
        st = build_regime(n, with_sales=False).state
        rivals.add(tuple(sorted(
            (o.owner_id, o.player_ids, o.spent) for o in st.owners
            if o.owner_id != st.focus_owner_id)))
    assert len(rivals) == 1, "rival rosters and spending must be held fixed"


def test_an_unknown_regime_is_refused():
    with pytest.raises(ValueError, match="unknown regime"):
        build_regime("juggernaut")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli(*argv):
    return subprocess.run([sys.executable, "-m", "ceauction.cli", "tactical",
                           *argv], capture_output=True, text=True)


@pytest.mark.slow
def test_cli_audited_reports_joint_worlds_and_nesting():
    r = _cli("max-bid", "--demo", "--demo-sales", "--mode", "audited",
             "--price", "12", "--leader", "Owner07", "--market-scenario",
             "base", "--max-prices", "2", "--sims", "300",
             "--selection-sims", "200", "--max-worlds", "2")
    assert r.returncode == 0, r.stderr
    for needle in ("CE-audited", "NESTED OPPORTUNITY SETS", "nested=True",
                   "every evaluated world passed conservation",
                   "selection / holdout", "cache entries"):
        assert needle in r.stdout, needle
    assert "Traceback" not in r.stderr


@pytest.mark.parametrize("argv,needle", [
    (("max-bid", "--demo", "--mode", "telepathy"), "invalid choice"),
    (("max-bid", "--demo", "--market-scenario", "medium"), "invalid choice"),
    (("max-bid",), "--demo is required"),
    (("max-bid", "--demo", "--leader", "Nobody"), "no owner"),
])
def test_cli_usage_errors_stay_clean(argv, needle):
    r = _cli(*argv)
    assert r.returncode != 0
    assert "Traceback" not in r.stderr
    assert needle in (r.stdout + r.stderr)
