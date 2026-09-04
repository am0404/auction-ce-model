"""Production validation of a real-player payload against the contract.

This is the validator the ingestion pipeline runs, not a test helper. It does
two jobs that a JSON Schema library alone would not:

1. **Structural** validation against
   ``schemas/real_player_input_v1.schema.json``. No JSON Schema package is a
   dependency of this project and one is not added for this; the subset of
   draft 2020-12 the contract actually uses is implemented here and is itself
   covered by tests.
2. **Semantic** validation of the rules the schema cannot express -- the ones
   that encode settled modelling decisions. A payload can be structurally
   perfect and still be wrong in a way that matters: a vendor fantasy total
   used as points, an expert grade turned into a standard deviation, a
   10-team non-superflex league configuration, an availability interpretation
   quietly preferred.

Both produce a :class:`ValidationResult` rather than raising, so a caller can
report every problem at once instead of one per run. :meth:`raise_for_status`
is there when failing fast is what you want.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

__all__ = [
    "ValidationError",
    "ValidationResult",
    "validate_contract",
    "validate_structure",
    "validate_semantics",
    "DEFAULT_SCHEMA_PATH",
]

DEFAULT_SCHEMA_PATH = (
    Path(__file__).resolve().parents[3] / "schemas" / "real_player_input_v1.schema.json"
)

_TYPES = {
    "object": dict, "array": list, "string": str, "boolean": bool,
    "number": (int, float), "integer": int, "null": type(None),
}


class ValidationError(RuntimeError):
    """Raised by :meth:`ValidationResult.raise_for_status`."""


@dataclass
class ValidationResult:
    """Everything wrong with a payload, in one pass.

    ``errors`` are contract violations. ``warnings`` are facts a human should
    see -- low coverage, a source that cannot be dated -- that do not make the
    payload invalid.
    """

    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def extend(self, other: "ValidationResult") -> "ValidationResult":
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        return self

    def raise_for_status(self) -> None:
        if self.errors:
            raise ValidationError(
                f"{len(self.errors)} contract violation(s):\n  "
                + "\n  ".join(self.errors))

    def summary(self) -> Dict[str, object]:
        return {"ok": self.ok, "errors": len(self.errors),
                "warnings": len(self.warnings)}


# ---------------------------------------------------------------------------
# Structural
# ---------------------------------------------------------------------------


def _resolve(node: Any, root: Dict) -> Any:
    hops = 0
    while isinstance(node, dict) and "$ref" in node:
        ref = node["$ref"]
        if not ref.startswith("#/"):
            raise ValueError(f"only local $refs are supported, got {ref!r}")
        target: Any = root
        for part in ref[2:].split("/"):
            target = target[part]
        node = target
        hops += 1
        if hops > 20:
            raise ValueError("circular $ref")
    return node


def _walk(instance: Any, schema: Any, root: Dict, path: str) -> List[str]:
    schema = _resolve(schema, root)
    errs: List[str] = []

    if "const" in schema and instance != schema["const"]:
        errs.append(f"{path}: expected {schema['const']!r}, got {instance!r}")
    if "enum" in schema and instance not in schema["enum"]:
        errs.append(f"{path}: {instance!r} not one of {schema['enum']}")

    if "type" in schema:
        want = schema["type"]
        want = [want] if isinstance(want, str) else want
        # bool is a subclass of int in Python; JSON Schema keeps them apart.
        ok = any(
            isinstance(instance, _TYPES[t])
            and not (t in ("number", "integer") and isinstance(instance, bool))
            for t in want
        )
        if not ok:
            errs.append(
                f"{path}: expected type {want}, got {type(instance).__name__}")
            return errs

    if isinstance(instance, str):
        if len(instance) < schema.get("minLength", 0):
            errs.append(f"{path}: shorter than minLength "
                        f"{schema['minLength']}")
        pattern = schema.get("pattern")
        if pattern and not re.search(pattern, instance):
            errs.append(f"{path}: does not match {pattern!r}")

    if isinstance(instance, (int, float)) and not isinstance(instance, bool):
        if "minimum" in schema and instance < schema["minimum"]:
            errs.append(f"{path}: below minimum {schema['minimum']}")
        if "maximum" in schema and instance > schema["maximum"]:
            errs.append(f"{path}: above maximum {schema['maximum']}")
        if "exclusiveMinimum" in schema and instance <= schema["exclusiveMinimum"]:
            errs.append(f"{path}: at or below exclusiveMinimum "
                        f"{schema['exclusiveMinimum']}")

    if isinstance(instance, list):
        if len(instance) < schema.get("minItems", 0):
            errs.append(f"{path}: fewer than minItems {schema['minItems']}")
        if "items" in schema:
            for i, item in enumerate(instance):
                errs += _walk(item, schema["items"], root, f"{path}[{i}]")

    if isinstance(instance, dict):
        for key in schema.get("required", []):
            if key not in instance:
                errs.append(f"{path}: missing required property {key!r}")
        if len(instance) < schema.get("minProperties", 0):
            errs.append(f"{path}: fewer than minProperties "
                        f"{schema['minProperties']}")
        props = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            for key in instance:
                if key not in props:
                    errs.append(f"{path}: unexpected property {key!r}")
        for key, sub in props.items():
            if key in instance:
                errs += _walk(instance[key], sub, root, f"{path}.{key}")
    return errs


def load_schema(path=None) -> Dict:
    return json.loads(Path(path or DEFAULT_SCHEMA_PATH).read_text(encoding="utf-8"))


def validate_structure(payload: Dict, schema: Optional[Dict] = None
                       ) -> ValidationResult:
    """Check a payload against the JSON contract."""
    schema = schema if schema is not None else load_schema()
    return ValidationResult(errors=_walk(payload, schema, schema, "$"))


# ---------------------------------------------------------------------------
# Semantic -- the settled modelling decisions
# ---------------------------------------------------------------------------

#: League configuration this engine targets. The previous model's 10-team
#: non-superflex settings must never reach the CE engine.
TARGET_LEAGUE_TEAMS = 12
TARGET_LEAGUE_REQUIRES_SUPERFLEX = True

_BANNED_LEAGUE_HINTS = ("10team", "10-team", "10 team", "non-superflex",
                        "no-superflex", "nonsuperflex")

#: Raw source keys that carry a vendor's own fantasy total. Preserving them in
#: ``raw_fields`` is correct and required; letting one become ``points`` is not.
VENDOR_TOTAL_KEYS = ("projections", "ds_projection_pts_ppr", "fantasyptsppr",
                     "fantasy_points_ppr", "projected_points")


def validate_semantics(payload: Dict) -> ValidationResult:
    """Check the rules the schema cannot express.

    Every check here corresponds to a settled decision recorded in
    ``docs/PLAYER_DATA_INVENTORY.md`` or the assignment that produced it.
    """
    res = ValidationResult()
    prov = payload.get("provenance", {}) or {}
    vendor_total_coincidences: List[int] = []

    # --- 1. target league ---------------------------------------------------
    league_id = str(prov.get("league_config_id", "") or "")
    if not league_id:
        res.errors.append(
            "provenance.league_config_id is empty: the payload must name the "
            "league it was built for. The target is the 12-team superflex "
            "league in SPEC.md.")
    lowered = league_id.lower().replace("_", "-")
    for hint in _BANNED_LEAGUE_HINTS:
        if hint in lowered:
            res.errors.append(
                f"provenance.league_config_id {league_id!r} looks like the "
                f"previous model's 10-team non-superflex configuration "
                f"({hint!r}). That configuration must not enter the CE engine; "
                f"the target is the 12-team superflex league.")
            break

    # --- 2. no vendor fantasy total may become points -----------------------
    for i, player in enumerate(payload.get("players", [])):
        where = f"players[{i}]"
        sp = player.get("season_points") or {}
        if sp and sp.get("scoring_source") != "recomputed_from_components":
            res.errors.append(
                f"{where}.season_points.scoring_source is "
                f"{sp.get('scoring_source')!r}; points must be recomputed from "
                f"component statistics. A vendor total solves to full PPR.")
        raw = player.get("raw_fields") or {}
        vendor_totals = {k: v for k, v in raw.items()
                         if k.strip().lower().replace(" ", "_") in VENDOR_TOTAL_KEYS}
        points = sp.get("points")
        # Equality with a vendor total is only evidence of misuse when the two
        # scoring systems would actually disagree. The vendor total is full PPR
        # and this league is half, so they differ by 0.5 per reception -- and
        # for a player with no receptions they legitimately coincide. Checking
        # equality alone flags every zero-reception player, which is a false
        # positive, not a finding.
        receptions = (player.get("stat_line") or {}).get("receptions")
        try:
            separation = 0.5 * float(receptions) if receptions is not None else 0.0
        except (TypeError, ValueError):
            separation = 0.0
        if separation > 1e-6:
            for key, value in vendor_totals.items():
                try:
                    as_float = float(str(value).replace(",", ""))
                except (TypeError, ValueError):
                    continue
                if points is not None and abs(as_float - float(points)) < 1e-9:
                    # Counted, not reported per player. Two reasons.
                    #
                    # It is a WARNING rather than an error because value
                    # equality is not proof of provenance: the vendor total is
                    # full-PPR only *on average* -- it was recovered by least
                    # squares across the file -- so an individual row can
                    # coincide with a correctly recomputed half-PPR figure by
                    # chance. The binding guarantee is the structural
                    # `scoring_source` check above.
                    #
                    # And it is AGGREGATED because the per-player form would
                    # carry a vendor projection value and a stat into a report
                    # that gets committed to a public repository. A count is
                    # also the more useful signal: one coincidence is noise,
                    # many at once would mean the total really is being reused.
                    vendor_total_coincidences.append(i)

        # --- 3. expert grades may not become a distribution -----------------
        labels = player.get("expert_labels") or {}
        if labels.get("may_derive_dispersion") is not False and labels:
            res.errors.append(
                f"{where}.expert_labels.may_derive_dispersion must be false. "
                f"UPSIDE/BUST are ordinal expert labels; no mapping from them "
                f"to variance, standard deviation, ceiling, floor or spike "
                f"probability exists.")

        # --- 4. neither availability interpretation may be preferred --------
        rate = player.get("active_rate") or {}
        if rate and rate.get("preferred", "missing") is not None:
            res.errors.append(
                f"{where}.active_rate.preferred must be null. The source's "
                f"availability treatment is unresolved and selecting an "
                f"interpretation is a modelling decision that has not been "
                f"made.")

        # --- 5. central tendency must be stated, with provenance ------------
        horizon = player.get("stat_line_horizon") or {}
        central = horizon.get("central_tendency")
        if central is None:
            res.errors.append(f"{where}.stat_line_horizon.central_tendency is "
                              f"required.")
        elif central != "unknown" and not horizon.get(
                "central_tendency_provenance"):
            res.errors.append(
                f"{where}.stat_line_horizon.central_tendency is {central!r} but "
                f"carries no provenance. A claim about central tendency must "
                f"say where it came from.")

        # --- 6. raw fields must survive -------------------------------------
        if not raw:
            res.errors.append(
                f"{where}.raw_fields is empty. Every mapping decision is a "
                f"judgement that will be revisited, and discarding the source "
                f"values turns a re-derivation into a re-import.")

    # --- 7. missing scoring categories stay absent --------------------------
    support = payload.get("scoring_support", {}) or {}
    supported = set(support.get("supported_categories", []))
    for entry in support.get("unsupported_categories", []):
        if entry.get("treated_as") != "absent":
            res.errors.append(
                f"scoring_support: category {entry.get('category')!r} is "
                f"treated_as {entry.get('treated_as')!r}; an unmodelled "
                f"category must be 'absent', never observed zero.")
        if entry.get("category") in supported:
            res.errors.append(
                f"scoring_support: {entry.get('category')!r} appears as both "
                f"supported and unsupported.")

    # --- 8. the payload must declare what it cannot calibrate ---------------
    if not payload.get("uncalibrated_parameters"):
        res.errors.append(
            "uncalibrated_parameters is required and must not be empty: this "
            "source cannot calibrate season_sd, signal_noise_sd, "
            "weekly_state_sd, proj_noise_sd, spike or role-change parameters, "
            "correlation loadings or contingency, and a run built on it has to "
            "be able to say so.")
    if "open_questions" not in payload:
        res.errors.append("open_questions is required.")

    # --- warnings -----------------------------------------------------------
    if vendor_total_coincidences:
        n = len(vendor_total_coincidences)
        total = len(payload.get("players", [])) or 1
        res.warnings.append(
            f"{n} of {total} players ({n / total:.1%}) have recomputed points "
            f"exactly equal to a vendor total preserved in raw_fields, despite "
            f"carrying receptions that should separate half-PPR from full. A "
            f"handful is coincidence; a large share would mean the vendor total "
            f"is being reused. Player indices are omitted here because this "
            f"report is published.")

    for src in prov.get("sources", []):
        if src.get("retrieved_at") is None:
            res.warnings.append(
                f"source {src.get('logical_name')!r} carries no retrieval "
                f"timestamp and cannot be dated.")
    blocking = [q for q in payload.get("open_questions", []) if q.get("blocking")]
    if blocking:
        res.warnings.append(
            f"{len(blocking)} blocking open question(s) remain: "
            f"{', '.join(str(q.get('id')) for q in blocking)}.")
    return res


def validate_contract(payload: Dict, schema: Optional[Dict] = None
                      ) -> ValidationResult:
    """Structural then semantic validation, all findings in one result."""
    return validate_structure(payload, schema).extend(validate_semantics(payload))
