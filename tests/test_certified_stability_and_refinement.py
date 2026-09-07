"""The predeclared stability rule, and the two intervals the refinement reports.

Neither of these is a simulation test. What they defend is the *reasoning*: that
the stability rule cannot be edited after seeing a result without the change
being visible, and that a sequence of pointwise intervals is never quietly
presented as a simultaneous band.
"""

from __future__ import annotations

import math

import pytest

from ceauction.tactical.certified_refinement import (RefinementReading,
                                                     bonferroni_t_half_width,
                                                     read_refinement,
                                                     t_quantile)
from ceauction.tactical.certified_stability import (CE_CANDIDATE_SET_UNDERCONVERGED,
                                                    MATERIAL_CE_SHIFT,
                                                    PROXY_BAND,
                                                    STABILITY_MAX_WORLDS,
                                                    STABILITY_PRICES,
                                                    STABILITY_SET_SIZES,
                                                    judge_stability,
                                                    stability_rule_fingerprint)


# ---------------------------------------------------------------------------
# The predeclared rule
# ---------------------------------------------------------------------------


def test_the_stability_rule_is_pinned():
    """Fingerprint recorded before the first 12->24 run.

    If this test fails, a threshold moved after results were seen. That may be
    entirely justified -- but it must be a visible, deliberate edit rather than
    a quiet one, which is the whole purpose of pinning it.
    """
    assert stability_rule_fingerprint() == "7f183724afc05780"
    assert STABILITY_PRICES == (65, 80)
    assert STABILITY_SET_SIZES == (12, 24)
    assert MATERIAL_CE_SHIFT == 0.01
    assert PROXY_BAND == 0.50


def test_max_worlds_is_above_the_evalcontext_default():
    """The test must be able to see a change at all.

    ``build_joint_worlds`` only builds ``max_worlds`` completions into worlds,
    so at the ``EvalContext`` default of 2 an expansion from 12 to 24 offers
    cannot alter the selection and the comparison would be vacuous.
    """
    from ceauction.tactical.evalcontext import EvalContext
    import dataclasses
    default = [f.default for f in dataclasses.fields(EvalContext)
               if f.name == "max_worlds"][0]
    assert STABILITY_MAX_WORLDS > default


def _row(**kw):
    base = dict(price=65, verdict_12="favorable", verdict_24="favorable",
                paired_mean_shift=0.0, selected_within_proxy_band=True,
                invariants_ok=True, invariant_detail="", max_residual=0.0)
    base.update(kw)
    return base


def test_a_verdict_change_fails_the_rule():
    v = judge_stability([_row(verdict_24="unfavorable")], (65, 80))
    assert not v.ok
    assert v.label == CE_CANDIDATE_SET_UNDERCONVERGED
    assert not v.clauses[0][1]


def test_a_material_ce_shift_fails_the_rule():
    v = judge_stability([_row(paired_mean_shift=MATERIAL_CE_SHIFT * 2)], (65, 80))
    assert not v.ok
    assert any("material" in c and not p for c, p, _ in v.clauses)


def test_a_shift_inside_the_threshold_passes():
    v = judge_stability([_row(price=65, paired_mean_shift=MATERIAL_CE_SHIFT / 2),
                         _row(price=80, verdict_12="unfavorable",
                              verdict_24="unfavorable")], (65, 80))
    assert v.ok, v.to_dict()
    assert v.label == "CE_CANDIDATE_SET_STABLE"


def test_a_selection_outside_the_proxy_band_fails():
    v = judge_stability([_row(selected_within_proxy_band=False)], (65, 80))
    assert not v.ok


def test_broken_reconciliation_fails():
    v = judge_stability([_row(invariants_ok=False,
                              invariant_detail="residual 1e-6")], (65, 80))
    assert not v.ok


# ---------------------------------------------------------------------------
# The intervals
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("p,df,want", [
    (0.975, 10, 2.2281), (0.995, 10, 3.1693), (0.975, 30, 2.0423),
    (0.975, 1, 12.7062), (0.95, 10, 1.8125),
])
def test_t_quantile_matches_published_tables(p, df, want):
    """No scipy, so the quantile is integrated here and must be checked."""
    assert t_quantile(p, df) == pytest.approx(want, abs=5e-4)


def test_t_quantile_is_symmetric():
    assert t_quantile(0.025, 10) == pytest.approx(-t_quantile(0.975, 10),
                                                  abs=1e-9)


def test_bonferroni_band_is_wider_than_pointwise():
    """The whole point: 14 statements cost width, and the code must pay it."""
    se = 0.01
    k, m = 11, 14
    point = t_quantile(0.975, k - 1) * se
    simul = bonferroni_t_half_width(se, k, m)
    assert simul > point
    assert simul / point == pytest.approx(3.7852 / 2.2281, rel=1e-3)


def test_simultaneous_never_calls_more_favorable_than_pointwise():
    """A wider band can only ever resolve fewer prices, never more."""
    points = [{"p": p, "mean_delta": 0.05 - 0.006 * i,
               "se_of_ensemble_mean": 0.008}
              for i, p in enumerate(range(66, 80))]
    r = read_refinement(points, k=11)
    for p in r.prices:
        if r.simultaneous_verdict[p] == "favorable":
            assert r.pointwise_verdict[p] == "favorable"
        if r.simultaneous_verdict[p] == "unfavorable":
            assert r.pointwise_verdict[p] == "unfavorable"
    hi_pw = r.pointwise_highest_favorable
    hi_sm = r.simultaneous_highest_favorable
    if hi_pw is not None and hi_sm is not None:
        assert hi_sm <= hi_pw


def test_nonmonotonicity_is_reported_not_smoothed():
    """A delta that rises with price is a finding about the surface."""
    means = [0.05, 0.04, 0.03, 0.06, -0.01]
    points = [{"p": 66 + i, "mean_delta": m, "se_of_ensemble_mean": 0.001}
              for i, m in enumerate(means)]
    r = read_refinement(points, k=11)
    assert any("rises" in s for s in r.nonmonotonicities), r.nonmonotonicities


def test_reading_reports_both_answers_and_says_they_differ():
    points = [{"p": p, "mean_delta": 0.02 - 0.004 * i,
               "se_of_ensemble_mean": 0.006}
              for i, p in enumerate(range(66, 80))]
    d = read_refinement(points, k=11).to_dict()
    assert d["m_statements"] == 14
    assert d["bonferroni_alpha_per_statement"] == pytest.approx(0.05 / 14)
    assert "NOT a simultaneous" in d["note"]
    assert set(d["prices"][0]) >= {"pointwise_ci", "simultaneous_ci",
                                   "pointwise_verdict", "simultaneous_verdict"}
