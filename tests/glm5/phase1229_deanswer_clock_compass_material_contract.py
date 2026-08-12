#!/usr/bin/env python3
"""Phase 1229: zero-model de-answer-load clock/compass material contract.

The phase creates a natural-language object-relation-value task in which the
record payload uses clock-face positions while the output uses compass words.
It freezes active, matched-null, and record-order panels, exact role spans,
and a donor registry.  It never loads a tokenizer or model.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests/glm5"
PHASE = 1229
SCRIPT = Path(__file__).resolve()
AUDIT_SCRIPT = TEST_ROOT / "phase1229_deanswer_clock_compass_material_contract_audit.py"

SOURCE_ROOT = TEST_ROOT / "result/phase1228_known_truth_automorphism_quotient_camera_revision1"
SOURCE_FINAL = SOURCE_ROOT / "analysis/final.json"
SOURCE_AUDIT = SOURCE_ROOT / "audit/independent_result_audit.json"
EXPECTED_SOURCE_FINAL = "3c884d130da36ddc6d6a3208e080de8bb83f4ec3493161b803d82058e7752b81"
EXPECTED_SOURCE_AUDIT = "0332256a24c33624b9ec0d646264885a1f38ad53e573b3831070948c1a90abdd"

OUT_ROOT = TEST_ROOT / "result/phase1229_deanswer_clock_compass_material_contract"
CONTRACT_PATH = OUT_ROOT / "protocol/material_contract.json"
PREAUDIT_PATH = OUT_ROOT / "audit/independent_preaudit.json"
MATERIAL_PATH = OUT_ROOT / "material/clock_compass_binding.jsonl"
DONOR_PATH = OUT_ROOT / "material/donor_registry.jsonl"
SUMMARY_PATH = OUT_ROOT / "analysis/readiness_summary.json"
MATERIAL_AUDIT_PATH = OUT_ROOT / "audit/independent_material_audit.json"
FINAL_PATH = OUT_ROOT / "analysis/final.json"
FINAL_AUDIT_PATH = OUT_ROOT / "audit/independent_final_audit.json"

SPLITS = ("discovery", "confirmation", "natural_use")
PANELS = ("active", "matched_null", "surface_order")
BINDING_STATES = (0, 1, 2, 3)
MAPPING_VARIANTS = (0, 1)
ORDER_VARIANTS = (0, 1, 2, 3)
TARGET_INDICES = (0, 1, 2, 3)
TEMPLATE_INDICES = (0, 1)

CLOCK_VALUES = ("twelve o'clock", "three o'clock", "six o'clock", "nine o'clock")
COMPASS_VALUES = ("north", "east", "south", "west")
CLOCK_TO_COMPASS = dict(zip(CLOCK_VALUES, COMPASS_VALUES))
CANDIDATES = COMPASS_VALUES
TOKEN_PATTERN = re.compile(r"[A-Za-z]+|[0-9]+|[^\w\s]", re.UNICODE)

WORLD_SPECS: dict[str, tuple[tuple[str, ...], ...]] = {
    "discovery": (
        ("Omar", "Lina", "Pavel", "Rita", "Soren"),
        ("Nolan", "Clara", "Felix", "Maya", "Bruno"),
        ("Victor", "Elena", "Gavin", "Iris", "Kellan"),
        ("Simon", "Julia", "Derek", "Nina", "Roland"),
    ),
    "confirmation": (
        ("Quentin", "Laura", "Peter", "Rosa", "Stefan"),
        ("Miles", "Chloe", "Grant", "Hazel", "Wesley"),
        ("Walter", "Eva", "Hugo", "Ingrid", "Jonas"),
        ("Samuel", "Jasmine", "Edgar", "Naomi", "Lucian"),
    ),
    "natural_use": (
        ("Orson", "Leona", "Pierce", "Rhea", "Tobias"),
        ("Neil", "Celia", "Fraser", "Mabel", "Warren"),
        ("Vernon", "Elsa", "Gideon", "Ida", "Jasper"),
        ("Seth", "Jade", "Desmond", "Nora", "Lionel"),
    ),
}

TEMPLATES: dict[str, tuple[dict[str, str], ...]] = {
    "discovery": (
        {
            "template_id": "discovery_stands",
            "record": "{entity} stands at {clock} on a clock centered on {anchor}.",
            "record_relation": "stands at",
            "query": (
                "Using one lowercase cardinal compass word, where is {target} relative to {anchor}? Answer:"
            ),
            "query_relation": "cardinal compass word",
        },
        {
            "template_id": "discovery_occupies",
            "record": "On the clock around {anchor}, {entity} occupies {clock}.",
            "record_relation": "occupies",
            "query": (
                "Give the lowercase compass direction of {target} from {anchor}, using one word only. Answer:"
            ),
            "query_relation": "compass direction",
        },
    ),
    "confirmation": (
        {
            "template_id": "confirmation_viewpoint",
            "record": "From {anchor}'s clock-face viewpoint, {entity} appears at {clock}.",
            "record_relation": "appears at",
            "query": "Name the lowercase compass direction from {anchor} to {target}. Answer:",
            "query_relation": "compass direction",
        },
        {
            "template_id": "confirmation_located",
            "record": "{entity} is located at {clock} with {anchor} at the clock center.",
            "record_relation": "is located at",
            "query": (
                "In one lowercase cardinal word, where does {target} lie relative to {anchor}? Answer:"
            ),
            "query_relation": "cardinal word",
        },
    ),
    "natural_use": (
        {
            "template_id": "natural_imagine",
            "record": "Imagine a clock centered on {anchor}; {entity} is by {clock}.",
            "record_relation": "is by",
            "query": (
                "If {anchor} is the center, in which cardinal direction is {target}? Answer:"
            ),
            "query_relation": "cardinal direction",
        },
        {
            "template_id": "natural_middle",
            "record": (
                "With {anchor} in the middle of a clock, {entity} can be found at {clock}."
            ),
            "record_relation": "can be found at",
            "query": "What lowercase compass direction takes {anchor} to {target}? Answer:",
            "query_relation": "compass direction",
        },
    ),
}

BASE_PERMUTATIONS = (
    (0, 1, 2, 3),
    (0, 2, 3, 1),
)
DISTRACTOR_PERMUTATIONS = (
    (0, 1, 2),
    (1, 2, 0),
    (2, 0, 1),
    (1, 0, 2),
)

EXPECTED_ROWS_PER_SPLIT = 3072
EXPECTED_ROW_COUNT = 9216
EXPECTED_ACTIVE_COUNT = 3072
EXPECTED_DONOR_COUNT = EXPECTED_ACTIVE_COUNT
HEURISTIC_CHANCE = 0.25


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")


def lexical_tokens(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


def token_multiset_digest(text: str) -> str:
    return digest(sorted(lexical_tokens(text)))


def rotate(values: tuple[str, ...], offset: int) -> tuple[str, ...]:
    shift = offset % len(values)
    return values[shift:] + values[:shift]


def verify_upstream() -> None:
    final = read_json(SOURCE_FINAL)
    audit = read_json(SOURCE_AUDIT)
    if final.get("final_digest") != EXPECTED_SOURCE_FINAL or not final.get("result", {}).get("camera_gate"):
        raise RuntimeError("Phase1228 final drift")
    if audit.get("audit_digest") != EXPECTED_SOURCE_AUDIT or not audit.get("all_checks_passed"):
        raise RuntimeError("Phase1228 audit drift")


def source_hashes() -> dict[str, str]:
    return {
        "generator": file_sha256(SCRIPT),
        "independent_audit": file_sha256(AUDIT_SCRIPT),
    }


def preregister() -> None:
    verify_upstream()
    if OUT_ROOT.exists():
        raise RuntimeError(f"formal output already exists: {OUT_ROOT}")
    contract: dict[str, Any] = {
        "phase": PHASE,
        "schema_version": "phase1229.deanswer.clock_compass.contract.v1",
        "created_at_utc": utc_now(),
        "objective": (
            "Construct and independently audit a natural-language clock-position to compass-direction binding "
            "family whose record payload never contains any answer candidate and whose active four-state bundles "
            "have identical token multisets with all four different gold answers."
        ),
        "execution_scope": {
            "zero_model": True,
            "tokenizer_loaded": False,
            "model_loaded": False,
            "cuda_required": False,
            "hidden_states": False,
        },
        "family": "clock_face_object_binding_to_compass_output",
        "splits": list(SPLITS),
        "panels": list(PANELS),
        "states": list(BINDING_STATES),
        "clock_values": list(CLOCK_VALUES),
        "candidates": list(CANDIDATES),
        "clock_to_compass": CLOCK_TO_COMPASS,
        "expected": {
            "row_count": EXPECTED_ROW_COUNT,
            "rows_per_split": EXPECTED_ROWS_PER_SPLIT,
            "active_count": EXPECTED_ACTIVE_COUNT,
            "donor_count": EXPECTED_DONOR_COUNT,
            "worlds_per_split": 4,
            "templates_per_split": 2,
            "states_per_bundle": 4,
        },
        "active_contract": {
            "same_prompt_token_multiset_across_states": True,
            "same_prompt_character_length_across_states": True,
            "gold_set_per_bundle": list(CANDIDATES),
            "record_answer_token_overlap": 0,
            "prompt_answer_token_overlap": 0,
            "symbolic_answer_unique": True,
        },
        "control_contract": {
            "matched_null": "target binding and answer fixed; only three distractor clock values are permuted",
            "surface_order": "all bindings and answer fixed; only four record positions are rotated",
            "same_answer_wrong_binding_donor": "same split/template/gold but different world and target",
            "wrong_answer_same_bundle_donor": "same carrier bundle with a different binding state and answer",
        },
        "heuristic_contract": {
            "scope": "active panel only",
            "features": [
                "constant", "token_multiset", "prompt_length", "target_entity",
                "target_record_position", "first_clock_value", "last_clock_value",
                "template", "world", "order_variant", "binding_state", "mapping_variant",
            ],
            "empirical_bayes_accuracy_max": HEURISTIC_CHANCE,
        },
        "span_registry": [
            "record_full", "record_object", "record_anchor", "record_relation",
            "record_value", "query_full", "query_subject", "query_anchor",
            "query_relation", "answer_boundary",
        ],
        "mathematical_corrections": {
            "threshold_permutations": (
                "A set of permutations inside a numerical tolerance need not be a subgroup; future natural cameras "
                "must enumerate compatible true subgroups and abstain when subgroup identity is not unique."
            ),
            "quotient_types": (
                "Physical-state response quotient H/~_{B,eta} and role-orbit quotient R/G_rho are distinct types."
            ),
            "basis_refinement": (
                "If B is contained in B', then ~_{B'} is contained in ~_B: basis extension monotonically refines "
                "or preserves equivalence classes."
            ),
            "group_action": "Use the standard left action (pi dot rho)(A)=rho(pi^{-1} A).",
            "gauge_scope": (
                "Phase1228 u/v are conjugate parameterizations, not independently trained algorithms or topologies."
            ),
        },
        "split_isolation": {
            "entity_vocabulary_disjoint": True,
            "template_ids_disjoint": True,
            "query_wording_disjoint": True,
            "clock_and_compass_ontology_shared_by_design": True,
        },
        "prohibited": [
            "load any tokenizer or model",
            "run Qwen3, GLM4, or DS7B",
            "change clock-to-compass mapping after materialization",
            "add an arbitrary A/B or alpha/beta interface",
            "place north/east/south/west in any prompt or record",
            "drop a failing world, template, panel, state, or direction",
            "call a material pass a language-mechanism result",
        ],
        "gates": {
            "all_structural_checks": True,
            "all_active_heuristics_at_chance": True,
            "all_donor_links_valid": True,
            "independent_audit_required": True,
        },
        "authorization": {
            "pass_authorizes": "Phase1230 zero-model tokenizer/interface behavior protocol materialization only",
            "does_not_authorize": "model execution, hidden scan, depth search, or causal patching",
        },
        "source": {
            "phase1228_final_digest": EXPECTED_SOURCE_FINAL,
            "phase1228_audit_digest": EXPECTED_SOURCE_AUDIT,
            "phase1228_final_sha256": file_sha256(SOURCE_FINAL),
            "phase1228_audit_sha256": file_sha256(SOURCE_AUDIT),
        },
        "source_hashes": source_hashes(),
    }
    contract["contract_digest"] = digest(contract)
    write_json(CONTRACT_PATH, contract)
    print(canonical_json({"status": "preregistered", "contract_digest": contract["contract_digest"]}))


def verify_frozen() -> dict[str, Any]:
    verify_upstream()
    contract = read_json(CONTRACT_PATH)
    if contract["contract_digest"] != digest({key: value for key, value in contract.items() if key != "contract_digest"}):
        raise RuntimeError("contract digest drift")
    if contract["source_hashes"] != source_hashes():
        raise RuntimeError("source changed after preregistration")
    preaudit = read_json(PREAUDIT_PATH)
    if not preaudit.get("all_checks_passed"):
        raise RuntimeError("independent preaudit did not pass")
    return contract


def base_assignment(mapping_variant: int) -> tuple[str, ...]:
    return tuple(CLOCK_VALUES[index] for index in BASE_PERMUTATIONS[mapping_variant])


def assignment_for(
    target_index: int,
    mapping_variant: int,
    panel: str,
    state: int,
) -> tuple[str, ...]:
    base = list(base_assignment(mapping_variant))
    if panel == "active":
        return tuple(base[(index + state) % 4] for index in range(4))
    if panel == "matched_null":
        distractors = [index for index in TARGET_INDICES if index != target_index]
        source = [base[index] for index in distractors]
        permutation = DISTRACTOR_PERMUTATIONS[state]
        result = list(base)
        for output_index, entity_index in enumerate(distractors):
            result[entity_index] = source[permutation[output_index]]
        return tuple(result)
    if panel == "surface_order":
        return tuple(base)
    raise ValueError(panel)


def add_span(registry: dict[str, list[list[int]]], role: str, start: int, end: int) -> None:
    if start < 0 or end <= start:
        raise RuntimeError(f"invalid span for {role}: {start}, {end}")
    registry.setdefault(role, []).append([int(start), int(end)])


def locate_once(text: str, needle: str, offset: int = 0) -> tuple[int, int]:
    local = text.find(needle)
    if local < 0:
        raise RuntimeError(f"substring not found: {needle!r}")
    return offset + local, offset + local + len(needle)


def render_prompt(
    split: str,
    template_index: int,
    anchor: str,
    entities: tuple[str, ...],
    clocks: tuple[str, ...],
    record_order: tuple[int, ...],
    target_index: int,
) -> tuple[str, str, str, dict[str, list[list[int]]], list[str]]:
    template = TEMPLATES[split][template_index]
    spans: dict[str, list[list[int]]] = {}
    parts: list[str] = []
    records: list[str] = []
    cursor = 0
    for entity_index in record_order:
        record = template["record"].format(
            entity=entities[entity_index], clock=clocks[entity_index], anchor=anchor
        )
        if parts:
            cursor += 1
        record_start = cursor
        record_end = record_start + len(record)
        add_span(spans, "record_full", record_start, record_end)
        for role, needle in (
            ("record_object", entities[entity_index]),
            ("record_anchor", anchor),
            ("record_relation", template["record_relation"]),
            ("record_value", clocks[entity_index]),
        ):
            start, end = locate_once(record, needle, record_start)
            add_span(spans, role, start, end)
        records.append(record)
        parts.append(record)
        cursor = record_end
    query = template["query"].format(target=entities[target_index], anchor=anchor)
    cursor += 1
    query_start = cursor
    query_end = query_start + len(query)
    add_span(spans, "query_full", query_start, query_end)
    for role, needle in (
        ("query_subject", entities[target_index]),
        ("query_anchor", anchor),
        ("query_relation", template["query_relation"]),
        ("answer_boundary", "Answer:"),
    ):
        start, end = locate_once(query, needle, query_start)
        add_span(spans, role, start, end)
    parts.append(query)
    prompt = " ".join(parts)
    return prompt, " ".join(records), query, spans, records


def build_row(
    split: str,
    world_index: int,
    target_index: int,
    template_index: int,
    order_variant: int,
    mapping_variant: int,
    state: int,
    panel: str,
) -> dict[str, Any]:
    names = WORLD_SPECS[split][world_index]
    anchor = names[0]
    entities = tuple(names[1:])
    clocks = assignment_for(target_index, mapping_variant, panel, state)
    base_order = rotate(tuple(TARGET_INDICES), order_variant)
    record_order = rotate(base_order, state) if panel == "surface_order" else base_order
    prompt, records_text, query, spans, records = render_prompt(
        split, template_index, anchor, entities, clocks, record_order, target_index
    )
    gold_clock = clocks[target_index]
    gold = CLOCK_TO_COMPASS[gold_clock]
    template_id = TEMPLATES[split][template_index]["template_id"]
    bundle_key = {
        "split": split,
        "world_index": world_index,
        "target_index": target_index,
        "template_id": template_id,
        "order_variant": order_variant,
        "mapping_variant": mapping_variant,
        "panel": panel,
    }
    identity = {**bundle_key, "state": state}
    row: dict[str, Any] = {
        "schema_version": "phase1229.clock_compass.row.v1",
        "phase": PHASE,
        "item_id": f"p1229-{digest(identity)[:24]}",
        "bundle_id": f"b1229-{digest(bundle_key)[:24]}",
        "split": split,
        "world_id": f"{split}-world-{world_index}",
        "world_index": world_index,
        "panel": panel,
        "binding_state": state,
        "mapping_variant": mapping_variant,
        "order_variant": order_variant,
        "template_id": template_id,
        "template_index": template_index,
        "anchor": anchor,
        "entities": list(entities),
        "target_entity": entities[target_index],
        "target_index": target_index,
        "record_order_indices": list(record_order),
        "record_order_entities": [entities[index] for index in record_order],
        "target_record_position": record_order.index(target_index),
        "assignments": {entities[index]: clocks[index] for index in TARGET_INDICES},
        "gold_clock_value": gold_clock,
        "gold_candidate": gold,
        "candidates": list(CANDIDATES),
        "prompt": prompt,
        "records_text": records_text,
        "rendered_records": records,
        "query": query,
        "answer_prefix": "Answer:",
        "prompt_char_length": len(prompt),
        "prompt_lexical_token_count": len(lexical_tokens(prompt)),
        "prompt_token_multiset_digest": token_multiset_digest(prompt),
        "record_token_multiset_digest": token_multiset_digest(records_text),
        "spans": spans,
    }
    row["row_digest"] = digest(row)
    return row


def generate_rows() -> list[dict[str, Any]]:
    rows = [
        build_row(split, world, target, template, order, mapping, state, panel)
        for split in SPLITS
        for world in range(len(WORLD_SPECS[split]))
        for target in TARGET_INDICES
        for template in TEMPLATE_INDICES
        for order in ORDER_VARIANTS
        for mapping in MAPPING_VARIANTS
        for panel in PANELS
        for state in BINDING_STATES
    ]
    if len(rows) != EXPECTED_ROW_COUNT:
        raise RuntimeError(f"row count mismatch: {len(rows)}")
    return rows


def donor_row(base: dict[str, Any], rows_by_id: dict[str, dict[str, Any]], registry: dict[tuple[Any, ...], dict[tuple[str, int], str]], same_answer_index: dict[tuple[str, str, str], list[str]]) -> dict[str, Any]:
    key = (
        base["split"], base["world_index"], base["target_index"], base["template_id"],
        base["order_variant"], base["mapping_variant"],
    )
    state = int(base["binding_state"])
    counterfactual = [registry[key][("active", other)] for other in BINDING_STATES if other != state]
    wrong = counterfactual[0]
    same_answer_candidates = same_answer_index[(base["split"], base["template_id"], base["gold_candidate"])]
    same_answer = next(
        item_id
        for item_id in same_answer_candidates
        if rows_by_id[item_id]["world_index"] != base["world_index"]
        and rows_by_id[item_id]["target_entity"] != base["target_entity"]
    )
    value: dict[str, Any] = {
        "schema_version": "phase1229.donor-registry.v1",
        "phase": PHASE,
        "recipient_id": base["item_id"],
        "split": base["split"],
        "counterfactual_active_ids": counterfactual,
        "wrong_answer_same_bundle_id": wrong,
        "same_answer_wrong_binding_id": same_answer,
        "matched_null_id": registry[key][("matched_null", state)],
        "surface_order_id": registry[key][("surface_order", state)],
    }
    value["row_digest"] = digest(value)
    return value


def build_donor_registry(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows_by_id = {row["item_id"]: row for row in rows}
    registry: dict[tuple[Any, ...], dict[tuple[str, int], str]] = defaultdict(dict)
    same_answer_index: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for row in rows:
        key = (
            row["split"], row["world_index"], row["target_index"], row["template_id"],
            row["order_variant"], row["mapping_variant"],
        )
        registry[key][(row["panel"], int(row["binding_state"]))] = row["item_id"]
        if row["panel"] == "active":
            same_answer_index[(row["split"], row["template_id"], row["gold_candidate"])].append(row["item_id"])
    for values in same_answer_index.values():
        values.sort()
    active = [row for row in rows if row["panel"] == "active"]
    donors = [donor_row(row, rows_by_id, registry, same_answer_index) for row in active]
    donors.sort(key=lambda row: row["recipient_id"])
    if len(donors) != EXPECTED_DONOR_COUNT:
        raise RuntimeError("donor count mismatch")
    return donors


def materialize() -> None:
    contract = verify_frozen()
    if MATERIAL_PATH.exists() or DONOR_PATH.exists() or SUMMARY_PATH.exists():
        raise RuntimeError("material outputs already exist")
    rows = generate_rows()
    donors = build_donor_registry(rows)
    write_jsonl(MATERIAL_PATH, rows)
    write_jsonl(DONOR_PATH, donors)
    summary: dict[str, Any] = {
        "phase": PHASE,
        "created_at_utc": utc_now(),
        "status": "materialized_pending_independent_audit",
        "contract_digest": contract["contract_digest"],
        "row_count": len(rows),
        "donor_count": len(donors),
        "split_counts": dict(Counter(row["split"] for row in rows)),
        "panel_counts": dict(Counter(row["panel"] for row in rows)),
        "material_digest": digest(rows),
        "donor_digest": digest(donors),
        "material_sha256": file_sha256(MATERIAL_PATH),
        "donor_sha256": file_sha256(DONOR_PATH),
        "model_loaded": False,
        "tokenizer_loaded": False,
    }
    summary["summary_digest"] = digest(summary)
    write_json(SUMMARY_PATH, summary)
    print(canonical_json({"status": summary["status"], "rows": len(rows), "donors": len(donors)}))


def finalize() -> None:
    contract = verify_frozen()
    if FINAL_PATH.exists():
        raise RuntimeError("final already exists")
    summary = read_json(SUMMARY_PATH)
    audit = read_json(MATERIAL_AUDIT_PATH)
    if not audit.get("all_checks_passed"):
        raise RuntimeError("material audit did not pass")
    gate = bool(audit["all_checks_passed"])
    final: dict[str, Any] = {
        "phase": PHASE,
        "created_at_utc": utc_now(),
        "status": "deanswer_material_contract_passed" if gate else "deanswer_material_contract_failed",
        "contract_digest": contract["contract_digest"],
        "summary_digest": summary["summary_digest"],
        "material_audit_digest": audit["audit_digest"],
        "result": {
            "material_gate": gate,
            "row_count": summary["row_count"],
            "donor_count": summary["donor_count"],
            "heuristics": audit["metrics"]["active_heuristic_bayes_accuracy"],
            "record_answer_overlap_count": audit["metrics"]["record_answer_overlap_count"],
            "prompt_answer_overlap_count": audit["metrics"]["prompt_answer_overlap_count"],
        },
        "k_ledger": {
            "new_item": None,
            "reason": "Zero-model material readiness is a protocol asset, not an empirical mechanism discovery.",
        },
        "mathematical_corrections": contract["mathematical_corrections"],
        "claim_boundary": [
            "No tokenizer, model, hidden state, behavior, or causal mechanism was tested.",
            "Natural wording is synthetic and does not by itself establish ecological validity.",
            "Removing answer-token overlap does not remove all semantic or parametric shortcuts.",
            "A pass authorizes only an independently frozen behavior-interface protocol.",
        ],
        "authorization": {
            "automatic_execution": True,
            "auto_continue": 1 if gate else 0,
            "next_experiment": "Phase1230 zero-model Qwen3 tokenizer/interface behavior protocol materialization",
            "model_execution_authorized": False,
        },
    }
    final["final_digest"] = digest(final)
    write_json(FINAL_PATH, final)
    print(canonical_json({"status": final["status"], "final_digest": final["final_digest"], "auto_continue": final["authorization"]["auto_continue"]}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=("preregister", "materialize", "finalize"))
    args = parser.parse_args()
    if args.stage == "preregister":
        preregister()
    elif args.stage == "materialize":
        materialize()
    else:
        finalize()


if __name__ == "__main__":
    main()
