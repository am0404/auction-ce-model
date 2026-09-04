"""The real-player input contract, and the fabricated fixture that exercises it.

The contract (``schemas/real_player_input_v1.schema.json``) is the seam real
vendor data must pass through before it reaches ``PlayerSpec``. Nothing imports
real data yet; what these tests defend is the contract's *shape*, and in
particular the handful of properties that exist to stop a known class of
mistake:

* a null must mean "not projected", never zero;
* a category with no source column must be recorded as absent, never defaulted;
* an unresolved semantic question must be carried, not assumed away;
* the raw source fields must survive, so a mapping judgement can be revisited;
* the expert upside/bust labels must not be usable as a dispersion estimate.

No JSON Schema library is a dependency of this project and one is not added for
this. ``_validate`` below implements the subset of draft 2020-12 the contract
actually uses, which is small and is itself covered by the negative tests.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO_ROOT / "schemas" / "real_player_input_v1.schema.json"
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "real_player_input_v1_example.json"

_TYPES = {
    "object": dict, "array": list, "string": str, "boolean": bool,
    "number": (int, float), "integer": int, "null": type(None),
}


def _resolve(node, root):
    seen = 0
    while isinstance(node, dict) and "$ref" in node:
        ref = node["$ref"]
        if not ref.startswith("#/"):
            raise ValueError(f"only local refs are supported, got {ref!r}")
        target = root
        for part in ref[2:].split("/"):
            target = target[part]
        node = target
        seen += 1
        if seen > 20:
            raise ValueError("circular $ref")
    return node


def _validate(instance, schema, root=None, path="$"):
    """Return a list of human-readable violations. Empty means valid."""
    root = root if root is not None else schema
    schema = _resolve(schema, root)
    errs = []

    if "const" in schema and instance != schema["const"]:
        errs.append(f"{path}: expected const {schema['const']!r}, got {instance!r}")
    if "enum" in schema and instance not in schema["enum"]:
        errs.append(f"{path}: {instance!r} not in {schema['enum']}")

    if "type" in schema:
        want = schema["type"]
        want = [want] if isinstance(want, str) else want
        # bool is a subclass of int; JSON Schema treats them as distinct.
        ok = any(
            isinstance(instance, _TYPES[t])
            and not (t in ("number", "integer") and isinstance(instance, bool))
            for t in want
        )
        if not ok:
            errs.append(f"{path}: expected type {want}, got {type(instance).__name__}")
            return errs

    if isinstance(instance, str):
        if "minLength" in schema and len(instance) < schema["minLength"]:
            errs.append(f"{path}: shorter than minLength")
        pat = schema.get("pattern")
        if pat and not re.search(pat, instance):
            errs.append(f"{path}: does not match {pat!r}")

    if isinstance(instance, (int, float)) and not isinstance(instance, bool):
        for key, cmp, label in (("minimum", lambda a, b: a < b, "below minimum"),
                                ("maximum", lambda a, b: a > b, "above maximum"),
                                ("exclusiveMinimum", lambda a, b: a <= b,
                                 "at or below exclusiveMinimum")):
            if key in schema and cmp(instance, schema[key]):
                errs.append(f"{path}: {label}")

    if isinstance(instance, list):
        if "minItems" in schema and len(instance) < schema["minItems"]:
            errs.append(f"{path}: fewer than minItems")
        if "items" in schema:
            for i, item in enumerate(instance):
                errs += _validate(item, schema["items"], root, f"{path}[{i}]")

    if isinstance(instance, dict):
        for key in schema.get("required", []):
            if key not in instance:
                errs.append(f"{path}: missing required property {key!r}")
        if "minProperties" in schema and len(instance) < schema["minProperties"]:
            errs.append(f"{path}: fewer than minProperties")
        props = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            for key in instance:
                if key not in props:
                    errs.append(f"{path}: unexpected property {key!r}")
        for key, sub in props.items():
            if key in instance:
                errs += _validate(instance[key], sub, root, f"{path}.{key}")
    return errs


@pytest.fixture(scope="module")
def schema():
    return json.loads(SCHEMA_PATH.read_text())


@pytest.fixture(scope="module")
def fixture():
    return json.loads(FIXTURE_PATH.read_text())


# --------------------------------------------------------------------------
# The validator itself, so the tests below mean something.
# --------------------------------------------------------------------------


def test_the_validator_catches_the_violations_it_is_used_to_assert():
    s = {"type": "object", "required": ["a"], "additionalProperties": False,
         "properties": {"a": {"type": "string", "minLength": 1},
                        "b": {"enum": [1, 2]},
                        "c": {"const": False},
                        "d": {"type": "number", "minimum": 0, "maximum": 1}}}
    assert _validate({"a": "x"}, s) == []
    assert any("missing required" in e for e in _validate({}, s))
    assert any("unexpected property" in e for e in _validate({"a": "x", "z": 1}, s))
    assert any("not in" in e for e in _validate({"a": "x", "b": 9}, s))
    assert any("const" in e for e in _validate({"a": "x", "c": True}, s))
    assert any("above maximum" in e for e in _validate({"a": "x", "d": 5}, s))
    assert any("expected type" in e for e in _validate({"a": 1}, s))
    # A bool must not satisfy "number".
    assert any("expected type" in e for e in _validate({"a": "x", "d": True}, s))


def test_local_refs_resolve():
    root = {"$defs": {"leaf": {"type": "string"}},
            "type": "object", "properties": {"x": {"$ref": "#/$defs/leaf"}}}
    assert _validate({"x": "ok"}, root, root) == []
    assert _validate({"x": 3}, root, root) != []


# --------------------------------------------------------------------------
# The schema.
# --------------------------------------------------------------------------


def test_the_schema_is_well_formed(schema):
    assert schema["$schema"].startswith("https://json-schema.org/draft/2020-12")
    assert schema["type"] == "object"
    assert schema["properties"]["schema_version"]["const"] == "1.0.0"
    for key in ("schema_version", "provenance", "scoring_support", "players"):
        assert key in schema["required"], f"{key} must be required"


def test_the_schema_carries_no_field_whose_meaning_is_unproven(schema):
    """Only proven quantities may be first-class fields.

    Everything unresolved is either absent, or present as an explicit
    "we do not know this" marker (a definition string, a null, or an
    'unknown' enum member) rather than as a usable number.
    """
    player = schema["$defs"]["player"]["properties"]

    # Not present at all: no source, so no field.
    for absent in ("season_sd", "signal_noise_sd", "weekly_state_sd",
                   "proj_noise_sd", "spike_rate", "spike_scale",
                   "role_change_prob", "shock_loadings", "contingency"):
        assert absent not in player, f"{absent} has no source and must not be a field"

    # Present, but each paired with a marker that it may be undefined.
    avail = player["availability"]["properties"]
    assert "injury_prob_definition" in avail
    assert "proj_games_missed_definition" in avail
    assert "null" in avail["injury_prob_definition"]["type"]

    horizon = player["stat_line_horizon"]["properties"]
    assert "unknown" in horizon["central_tendency"]["enum"]
    assert "null" in horizon["games_assumed"]["type"]
    assert "null" in horizon["conditional_on_playing"]["type"]


def test_expert_labels_are_pinned_as_non_distributional(schema):
    """The grades must not become a standard deviation by accident."""
    labels = schema["$defs"]["player"]["properties"]["expert_labels"]["properties"]
    assert labels["may_derive_dispersion"]["const"] is False
    assert labels["scale"]["const"] == "ordinal_1_to_5"
    for tag in ("upside", "bust"):
        assert labels[tag]["minimum"] == 1 and labels[tag]["maximum"] == 5


def test_a_missing_scoring_category_can_only_be_recorded_as_absent(schema):
    unsup = (schema["properties"]["scoring_support"]["properties"]
             ["unsupported_categories"]["items"])
    assert unsup["properties"]["treated_as"]["const"] == "absent"
    # `treated_as` is REQUIRED, not optional: an entry that omits it leaves the
    # reader to assume, and the assumption they would make is "zero".
    assert set(unsup["required"]) == {"category", "reason", "treated_as"}


# --------------------------------------------------------------------------
# Hardened requirements: omitting any of these must fail validation.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("member", [
    "schema_version", "provenance", "scoring_support",
    "uncalibrated_parameters", "open_questions", "players",
])
def test_every_required_root_member_is_required(schema, fixture, member):
    assert member in schema["required"]
    d = json.loads(json.dumps(fixture))
    d.pop(member)
    errs = _validate(d, schema)
    assert any("missing required" in e and member in e for e in errs), (
        f"omitting {member} must fail validation")


def test_a_payload_that_declares_nothing_uncalibrated_is_still_structurally_ok_but_semantically_not(
        schema, fixture):
    """The schema requires the key; the production validator requires content.

    An empty list satisfies the shape and still hides the thing that matters,
    so the semantic check in `ceauction.realdata.validate` rejects it.
    """
    from ceauction.realdata.validate import validate_semantics

    d = json.loads(json.dumps(fixture))
    d["uncalibrated_parameters"] = []
    assert _validate(d, schema) == [], "an empty list is structurally valid"
    res = validate_semantics(d)
    assert not res.ok
    assert any("uncalibrated" in e for e in res.errors)


def test_league_config_id_is_required(schema, fixture):
    assert "league_config_id" in schema["properties"]["provenance"]["required"]
    d = json.loads(json.dumps(fixture))
    d["provenance"].pop("league_config_id")
    assert any("league_config_id" in e for e in _validate(d, schema))


def test_central_tendency_is_required(schema, fixture):
    horizon = schema["$defs"]["player"]["properties"]["stat_line_horizon"]
    assert "central_tendency" in horizon["required"]
    d = json.loads(json.dumps(fixture))
    d["players"][0]["stat_line_horizon"].pop("central_tendency")
    assert any("central_tendency" in e for e in _validate(d, schema))


def test_treated_as_is_required_on_every_unsupported_category(schema, fixture):
    d = json.loads(json.dumps(fixture))
    d["scoring_support"]["unsupported_categories"][0].pop("treated_as")
    assert any("treated_as" in e for e in _validate(d, schema))


def test_additional_properties_are_refused_throughout(schema):
    """Strictness at every level a payload can grow one."""
    assert schema["additionalProperties"] is False
    assert schema["$defs"]["player"]["additionalProperties"] is False
    assert schema["$defs"]["stat_line"]["additionalProperties"] is False
    for name in ("provenance", "scoring_support"):
        assert schema["properties"][name]["additionalProperties"] is False
    for name in ("availability", "expert_labels", "stat_line_horizon",
                 "season_points", "active_rate", "cohort_dispersion"):
        assert schema["$defs"]["player"]["properties"][name][
            "additionalProperties"] is False


def test_every_stat_line_field_permits_null(schema):
    """Null means NOT PROJECTED. The source leaves most TD cells blank."""
    for name, sub in schema["$defs"]["stat_line"]["properties"].items():
        assert "null" in sub["type"], f"{name} must permit null"


def test_raw_fields_are_required_and_non_empty(schema):
    player = schema["$defs"]["player"]
    assert "raw_fields" in player["required"]
    assert player["properties"]["raw_fields"]["minProperties"] == 1


def test_provenance_identifies_sources_by_hash_not_filename(schema):
    src = schema["properties"]["provenance"]["properties"]["sources"]["items"]
    assert set(src["required"]) == {"logical_name", "sha256", "role"}
    assert src["properties"]["sha256"]["pattern"] == "^[0-9a-f]{64}$"
    # A source that cannot be dated must still be representable.
    assert "null" in src["properties"]["retrieved_at"]["type"]
    assert "market" in src["properties"]["role"]["enum"]


# --------------------------------------------------------------------------
# The fabricated fixture.
# --------------------------------------------------------------------------


def test_the_fixture_validates_against_the_schema(fixture, schema):
    errs = _validate(fixture, schema)
    assert errs == [], "fixture does not satisfy the contract:\n  " + "\n  ".join(errs)


def test_the_fixture_is_obviously_fabricated(fixture):
    """It must never be mistaken for, or become, real data."""
    blob = json.dumps(fixture)
    assert "FIXTURE" in blob
    assert fixture["provenance"]["league_config_id"] == "FIXTURE-LEAGUE-NOT-REAL"
    for s in fixture["provenance"]["sources"]:
        assert s["vendor"] == "FIXTURE"
        # Placeholder hashes, not real content hashes.
        assert len(set(s["sha256"])) == 1
    assert len(fixture["players"]) <= 5, "a fixture, not a dataset"
    for p in fixture["players"]:
        assert p["name"].startswith("Fabricated")


def test_the_fixture_exercises_the_sparse_path(fixture):
    """One player must be mostly null, or the null semantics are untested."""
    sparse = next(p for p in fixture["players"] if p["player_key"] == "fabricated_beta")
    assert sparse["nfl_team"] is None
    assert sparse["bye_week"] is None
    assert sparse["stat_line"]["rec_tds"] is None, "blank TD cell must stay null, not 0"
    assert sparse["availability"]["injury_prob"] is None
    assert sparse["expert_labels"]["upside"] is None
    # ...and the raw source cell that produced the null is preserved verbatim.
    assert sparse["raw_fields"]["Rec TDs"] == ""
    assert sparse["raw_fields"]["UPSIDE "] == "-"


def test_the_fixture_declares_the_known_scoring_gaps(fixture):
    missing = {u["category"] for u in
               fixture["scoring_support"]["unsupported_categories"]}
    assert {"pass_2pt", "rush_2pt", "rec_2pt", "special_teams_td"} <= missing
    for u in fixture["scoring_support"]["unsupported_categories"]:
        assert u["treated_as"] == "absent"
    # None of the gaps may also be claimed as supported.
    assert not (missing & set(fixture["scoring_support"]["supported_categories"]))


def test_the_fixture_leaves_the_blocking_questions_open(fixture):
    """The honest state of the real source today."""
    for p in fixture["players"]:
        assert p["stat_line_horizon"]["central_tendency"] == "unknown"
        assert p["stat_line_horizon"]["games_assumed"] is None
        assert p["stat_line"]["fumbles_are_lost_fumbles"] is None
    ids = {q["id"] for q in fixture["open_questions"] if q["blocking"]}
    assert {"Q1", "Q2", "Q4", "Q6"} <= ids


def test_the_fixture_lists_the_uncalibrated_parameters(fixture):
    params = {u["parameter"] for u in fixture["uncalibrated_parameters"]}
    assert {"season_sd", "signal_noise_sd", "weekly_state_sd", "proj_noise_sd",
            "spike_rate", "spike_scale", "role_change_prob", "shock_loadings",
            "contingency"} <= params
    for u in fixture["uncalibrated_parameters"]:
        assert u["reason"]


def test_the_fixture_names_only_positions_this_engine_models(fixture, schema):
    allowed = set(schema["$defs"]["player"]["properties"]["position"]["enum"])
    from ceauction.league import Position
    assert allowed == {p.name for p in Position}
    for p in fixture["players"]:
        assert p["position"] in allowed


# --------------------------------------------------------------------------
# Negative cases: the contract must reject these.
# --------------------------------------------------------------------------


def _mutated(fixture, mutate):
    d = json.loads(json.dumps(fixture))
    mutate(d)
    return d


def test_a_missing_scoring_category_defaulted_to_zero_is_rejected(fixture, schema):
    def mutate(d):
        d["scoring_support"]["unsupported_categories"][0]["treated_as"] = "zero"
    assert _validate(_mutated(fixture, mutate), schema)


def test_claiming_the_expert_labels_are_a_dispersion_estimate_is_rejected(fixture, schema):
    def mutate(d):
        d["players"][0]["expert_labels"]["may_derive_dispersion"] = True
    assert _validate(_mutated(fixture, mutate), schema)


def test_dropping_the_raw_fields_is_rejected(fixture, schema):
    for mutate in (lambda d: d["players"][0].pop("raw_fields"),
                   lambda d: d["players"][0].__setitem__("raw_fields", {})):
        assert _validate(_mutated(fixture, mutate), schema)


def test_an_unversioned_or_wrongly_versioned_payload_is_rejected(fixture, schema):
    for mutate in (lambda d: d.pop("schema_version"),
                   lambda d: d.__setitem__("schema_version", "2.0.0")):
        assert _validate(_mutated(fixture, mutate), schema)


def test_a_payload_without_provenance_is_rejected(fixture, schema):
    assert _validate(_mutated(fixture, lambda d: d.pop("provenance")), schema)


def test_a_source_without_a_content_hash_is_rejected(fixture, schema):
    for mutate in (lambda d: d["provenance"]["sources"][0].pop("sha256"),
                   lambda d: d["provenance"]["sources"][0]
                              .__setitem__("sha256", "not-a-hash")):
        assert _validate(_mutated(fixture, mutate), schema)


def test_an_unrecognised_position_is_rejected(fixture, schema):
    def mutate(d):
        d["players"][0]["position"] = "K"
    assert _validate(_mutated(fixture, mutate), schema)


def test_an_unexpected_top_level_key_is_rejected(fixture, schema):
    """Strictness is the point: a stray key is usually a renamed one."""
    def mutate(d):
        d["dollar_values"] = {"Fabricated Alpha": 42}
    assert _validate(_mutated(fixture, mutate), schema)
