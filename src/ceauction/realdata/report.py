"""The sanitized ingestion report.

This is the artifact that may be committed. It carries **counts, coverage,
match rates, ranges, missingness, warnings and validation results** and nothing
else. No player name, no per-player value, no proprietary vendor number, no
local path.

The distinction matters because this repository is public and the sources are
subscriber-gated vendor exports that are not redistributable. Aggregates over a
whole column are safe; the rows they aggregate are not.
"""

from __future__ import annotations

import statistics as st
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

from .identity import MatchReport
from .validate import ValidationResult

__all__ = ["numeric_summary", "build_report", "format_report"]

#: Fields whose aggregate distribution is safe to publish. Anything not listed
#: is omitted rather than guessed about.
_SUMMARISED = (
    ("season_points", lambda p: (p.get("season_points") or {}).get("points")),
    ("active_rate_a", lambda p: (p.get("active_rate") or {})
     .get("interpretation_a_full_health")),
    ("active_rate_b", lambda p: (p.get("active_rate") or {})
     .get("interpretation_b_availability_adjusted")),
    ("omitted_fumble_points", lambda p: (p.get("season_points") or {})
     .get("omitted_fumble_points")),
    ("injury_prob", lambda p: (p.get("availability") or {}).get("injury_prob")),
    ("proj_games_missed", lambda p: (p.get("availability") or {})
     .get("proj_games_missed")),
    ("bye_week", lambda p: p.get("bye_week")),
)


def numeric_summary(values: Iterable[Optional[float]], total: int
                    ) -> Dict[str, object]:
    """Count, coverage and range for one column. Never individual values."""
    nums = sorted(float(v) for v in values
                  if v is not None and not isinstance(v, bool))
    if not nums:
        return {"n": 0, "coverage": 0.0, "missing_pct": 100.0}

    def pct(p: float) -> float:
        return nums[min(len(nums) - 1, int(p * len(nums)))]

    return {
        "n": len(nums),
        "coverage": round(len(nums) / total, 4) if total else 0.0,
        "missing_pct": round(100.0 * (1 - len(nums) / total), 1) if total else 0.0,
        "min": round(nums[0], 3),
        "p10": round(pct(0.10), 3),
        "median": round(st.median(nums), 3),
        "p90": round(pct(0.90), 3),
        "max": round(nums[-1], 3),
        "mean": round(st.mean(nums), 3),
    }


def build_report(payload: Dict, reports: Dict[str, MatchReport],
                 validation: ValidationResult,
                 warnings: Sequence[str] = ()) -> Dict[str, object]:
    """Assemble the sanitized report as plain data."""
    players = payload.get("players", [])
    total = len(players)

    by_position: Dict[str, int] = {}
    for p in players:
        pos = str(p.get("position"))
        by_position[pos] = by_position.get(pos, 0) + 1

    fields = {name: numeric_summary((getter(p) for p in players), total)
              for name, getter in _SUMMARISED}

    labelled = sum(1 for p in players
                   if (p.get("expert_labels") or {}).get("upside") is not None)
    with_team = sum(1 for p in players if p.get("nfl_team"))
    with_bye = sum(1 for p in players if p.get("bye_week"))
    with_injury = sum(1 for p in players
                      if (p.get("availability") or {}).get("injury_prob") is not None)
    with_cv = sum(1 for p in players
                  if (p.get("cohort_dispersion") or {}).get("weekly_cv") is not None)

    prov = payload.get("provenance", {})
    return {
        "schema_version": payload.get("schema_version"),
        "generated_at": prov.get("generated_at"),
        "league_config_id": prov.get("league_config_id"),
        "sources": [
            {"logical_name": s.get("logical_name"), "sha256": s.get("sha256"),
             "role": s.get("role"), "retrieved_at": s.get("retrieved_at")}
            for s in prov.get("sources", [])
        ],
        "players_normalized": total,
        "players_by_position": dict(sorted(by_position.items())),
        "coverage": {
            "nfl_team": {"n": with_team,
                         "pct": round(100.0 * with_team / total, 1) if total else 0.0},
            "bye_week": {"n": with_bye,
                         "pct": round(100.0 * with_bye / total, 1) if total else 0.0},
            "injury_profile": {"n": with_injury,
                               "pct": round(100.0 * with_injury / total, 1) if total else 0.0},
            "cohort_dispersion": {"n": with_cv,
                                  "pct": round(100.0 * with_cv / total, 1) if total else 0.0},
            "expert_labels": {"n": labelled,
                              "pct": round(100.0 * labelled / total, 1) if total else 0.0},
        },
        "joins": {name: rep.summary() for name, rep in sorted(reports.items())},
        "field_summaries": fields,
        "scoring_support": {
            "supported": payload.get("scoring_support", {}).get(
                "supported_categories", []),
            "unsupported": [
                {"category": u.get("category"), "treated_as": u.get("treated_as")}
                for u in payload.get("scoring_support", {}).get(
                    "unsupported_categories", [])
            ],
        },
        "uncalibrated_parameters": [
            u.get("parameter") for u in payload.get("uncalibrated_parameters", [])
        ],
        "open_questions": [
            {"id": q.get("id"), "blocking": q.get("blocking")}
            for q in payload.get("open_questions", [])
        ],
        "validation": {
            "ok": validation.ok,
            "errors": list(validation.errors),
            "warnings": list(validation.warnings),
        },
        "build_warnings": list(warnings),
    }


def _fmt_summary(name: str, s: Dict[str, object]) -> str:
    if not s.get("n"):
        return f"  {name:<22} absent"
    return (f"  {name:<22} n={s['n']:<5} cover={s['coverage']:<7} "
            f"min={s['min']:<9} med={s['median']:<9} max={s['max']:<9} "
            f"mean={s['mean']}")


def format_report(report: Dict[str, object], width: int = 78) -> str:
    """Human-readable rendering of the sanitized report."""
    bar = "=" * width
    out = [bar, "REAL-PLAYER INGESTION REPORT (sanitized)", bar,
           f"schema_version    {report['schema_version']}",
           f"generated_at      {report['generated_at']}",
           f"league_config_id  {report['league_config_id']}",
           "",
           "SOURCES (identified by content hash, not filename)"]
    for s in report["sources"]:
        out.append(f"  {s['logical_name']:<38} {str(s['sha256'])[:16]}  "
                   f"{s['role']}")
        if s["retrieved_at"] is None:
            out.append(f"  {'':<38} (no retrieval timestamp: cannot be dated)")

    out += ["", f"NORMALIZED PLAYERS: {report['players_normalized']}",
            f"  by position: {report['players_by_position']}", "",
            "COVERAGE"]
    for k, v in report["coverage"].items():
        out.append(f"  {k:<22} {v['n']:>5}  ({v['pct']}%)")

    out += ["", "JOINS"]
    for name, j in report["joins"].items():
        out.append(f"  {name}: matched {j['matched']}/{j['left_rows']} "
                   f"({j['match_rate']:.1%}); unmatched_left {j['unmatched_left']}, "
                   f"unmatched_right {j['unmatched_right']}, "
                   f"ambiguous {j['ambiguous']}, "
                   f"dup_left {j['duplicate_left']}, dup_right {j['duplicate_right']}, "
                   f"conflicting {j['conflicting']}")

    out += ["", "FIELD SUMMARIES (aggregates only)"]
    for name, s in report["field_summaries"].items():
        out.append(_fmt_summary(name, s))

    sup = report["scoring_support"]
    out += ["", "SCORING SUPPORT",
            f"  supported   ({len(sup['supported'])}): {', '.join(sup['supported'])}",
            f"  unsupported ({len(sup['unsupported'])}):"]
    for u in sup["unsupported"]:
        out.append(f"    {u['category']:<20} treated_as={u['treated_as']}")

    out += ["", f"UNCALIBRATED PARAMETERS ({len(report['uncalibrated_parameters'])})",
            "  " + ", ".join(report["uncalibrated_parameters"]),
            "", "OPEN QUESTIONS"]
    for q in report["open_questions"]:
        out.append(f"  {q['id']:<5} blocking={q['blocking']}")

    v = report["validation"]
    out += ["", "VALIDATION", f"  ok: {v['ok']}",
            f"  errors:   {len(v['errors'])}"]
    for e in v["errors"]:
        out.append(f"    ERROR   {e}")
    out.append(f"  warnings: {len(v['warnings'])}")
    for w in v["warnings"]:
        out.append(f"    WARN    {w}")
    if report["build_warnings"]:
        out.append(f"  build warnings: {len(report['build_warnings'])}")
        for w in report["build_warnings"]:
            out.append(f"    WARN    {w}")
    out.append(bar)
    return "\n".join(out)
