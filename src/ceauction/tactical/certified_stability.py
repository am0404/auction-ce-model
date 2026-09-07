"""Does doubling the certified candidate set change what championship equity picks?

The proxy-side answer is already known and is not the question: the 12- and
24-completion sets are nested by construction, so the proxy optimum cannot move
and the certified bound cannot widen. Reporting that as "stable" would be
circular -- it is true of any nested expansion, however much CE disagreed with
the proxy about which member to take.

The question this module asks is the one that matters. CE ranks *outcome
distributions*, and two rosters a hundredth of a weekly point apart on expected
points can differ materially in variance, so the set exists precisely because
the proxy's ordering is not CE's ordering. If enlarging the set changes which
completion CE selects, and that change moves a verdict, the coarse frontier was
read off an under-populated offer.

**One structural fact governs the whole design.** ``build_joint_worlds`` only
ever builds ``max_worlds`` completions into reconciled worlds, and
``evaluate_joint_arm`` then chooses among *those* on the selection sample. An
offer set larger than ``max_worlds`` therefore cannot influence the answer at
all. Run at the ``EvalContext`` default of 2, a 12->24 comparison would return
"identical" for a reason that has nothing to do with convergence. So the test
below raises ``max_worlds`` to :data:`STABILITY_MAX_WORLDS` in **both** arms,
which is what gives the expansion room to matter, and every result records
``n_worlds_compared`` so the reader can confirm it did.

Everything in section 1 was fixed before any result was looked at, and is
fingerprinted so a later edit cannot retrofit it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = [
    "STABILITY_RULE_DECLARED_AT", "STABILITY_PRICES", "STABILITY_SET_SIZES",
    "STABILITY_MAX_WORLDS", "MATERIAL_CE_SHIFT", "PROXY_BAND",
    "RECONCILIATION_TOLERANCE", "LEAGUE_CE_SUM_TOLERANCE",
    "CE_CANDIDATE_SET_UNDERCONVERGED", "stability_rule_fingerprint",
    "StabilityVerdict", "judge_stability",
]

STABILITY_RULE_DECLARED_AT = "2026-09-07, before any 12->24 CE result was run"

#: The two prices the brief names: one either side of the coarse crossing.
STABILITY_PRICES: Tuple[int, ...] = (65, 80)

#: Nested set sizes, per generation budget. The 24 contains the 12 because the
#: near-optimal walk emits in certified order, which is asserted, not assumed.
STABILITY_SET_SIZES: Tuple[int, int] = (12, 24)

#: How many completions CE is allowed to choose between, in BOTH arms.
#:
#: Not the ``EvalContext`` default of 2. At 2 the expansion is provably inert
#: and the test would be vacuous. At 8 the offer's top slice can genuinely be
#: rearranged by the added members, and the cost is ~1.9x the default rather
#: than the ~15x that comparing whole sets would need.
STABILITY_MAX_WORLDS = 8

#: A CE shift at or below this is not material.
#:
#: The coarse frontier's crossing runs from ``+0.02111`` at ``$65`` to
#: ``-0.05386`` at ``$80``: a span of ``0.075`` championship equity across
#: ``$15``, so roughly ``0.005`` per dollar. ``0.01`` is therefore about two
#: dollars of reservation price. A shift larger than that could move the
#: bracket and must fail the test; a shift smaller than that cannot.
MATERIAL_CE_SHIFT = 0.01

#: Selected completions must still sit inside the band they were emitted under.
PROXY_BAND = 0.50

RECONCILIATION_TOLERANCE = 1e-12
LEAGUE_CE_SUM_TOLERANCE = 1e-9

CE_CANDIDATE_SET_UNDERCONVERGED = "CE_CANDIDATE_SET_UNDERCONVERGED"


def stability_rule_fingerprint() -> str:
    """Digest of the predeclared rule, pinned by a test."""
    body = json.dumps({
        "declared_at": STABILITY_RULE_DECLARED_AT,
        "prices": list(STABILITY_PRICES),
        "sizes": list(STABILITY_SET_SIZES),
        "max_worlds": STABILITY_MAX_WORLDS,
        "material_ce_shift": MATERIAL_CE_SHIFT,
        "proxy_band": PROXY_BAND,
        "reconciliation_tolerance": RECONCILIATION_TOLERANCE,
        "league_ce_sum_tolerance": LEAGUE_CE_SUM_TOLERANCE,
    }, sort_keys=True)
    return hashlib.sha256(body.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class StabilityVerdict:
    """The five predeclared clauses, each pass/fail with its evidence."""

    clauses: Tuple[Tuple[str, bool, str], ...]
    rows: Tuple[Dict[str, object], ...]

    @property
    def ok(self) -> bool:
        return all(passed for _, passed, _ in self.clauses)

    @property
    def label(self) -> str:
        return "CE_CANDIDATE_SET_STABLE" if self.ok \
            else CE_CANDIDATE_SET_UNDERCONVERGED

    def to_dict(self) -> Dict[str, object]:
        return {
            "rule_fingerprint": stability_rule_fingerprint(),
            "declared_at": STABILITY_RULE_DECLARED_AT,
            "max_worlds": STABILITY_MAX_WORLDS,
            "material_ce_shift": MATERIAL_CE_SHIFT,
            "label": self.label, "ok": self.ok,
            "clauses": [{"clause": c, "pass": p, "evidence": e}
                        for c, p, e in self.clauses],
            "rows": list(self.rows),
        }


def judge_stability(rows: Sequence[Dict[str, object]],
                    coarse_bracket: Tuple[int, int]) -> StabilityVerdict:
    """Apply the predeclared rule. No clause is invented here.

    ``rows`` is one entry per tested price, each carrying both arms' verdict,
    mean delta, the paired per-draw shift, the selected rosters and the
    invariant flags.
    """
    clauses: List[Tuple[str, bool, str]] = []

    bad = [r for r in rows if r["verdict_12"] != r["verdict_24"]]
    clauses.append((
        "no verdict changes", not bad,
        "; ".join(f"p=${r['price']}: {r['verdict_12']} -> {r['verdict_24']}"
                  for r in bad) or
        "; ".join(f"p=${r['price']}: {r['verdict_12']}" for r in rows)))

    lo, hi = coarse_bracket
    still = all((r["verdict_24"] == "favorable") == (int(r["price"]) <= lo)
                for r in rows if int(r["price"]) in (lo, hi))
    clauses.append((
        f"crossing still bracketed by (${lo}, ${hi})", still,
        "; ".join(f"p=${r['price']}: {r['verdict_24']}" for r in rows)))

    worst = max((abs(float(r["paired_mean_shift"])) for r in rows), default=0.0)
    clauses.append((
        f"no material CE-selection improvement (|shift| <= {MATERIAL_CE_SHIFT})",
        worst <= MATERIAL_CE_SHIFT + 1e-12,
        f"worst paired mean shift {worst:.6f}"))

    outside = [r for r in rows if not r["selected_within_proxy_band"]]
    clauses.append((
        f"every selected completion within {PROXY_BAND} of its certified optimum",
        not outside,
        "; ".join(f"p=${r['price']}" for r in outside) or "all inside"))

    bad_inv = [r for r in rows if not r["invariants_ok"]]
    clauses.append((
        "reconciliation and conservation exact", not bad_inv,
        "; ".join(f"p=${r['price']}: {r['invariant_detail']}"
                  for r in bad_inv) or
        f"max residual {max((float(r['max_residual']) for r in rows), default=0.0):.1e}"))

    return StabilityVerdict(tuple(clauses), tuple(rows))
