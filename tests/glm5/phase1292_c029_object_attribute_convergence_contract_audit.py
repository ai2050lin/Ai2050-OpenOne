#!/usr/bin/env python3
"""Independent replay audit for Phase 1292/C029.

The auditor deliberately does not import the contract compiler.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests/glm5"
MAIN = TEST_ROOT / "phase1292_c029_object_attribute_convergence_contract.py"
AUDITOR = Path(__file__).resolve()
OUT = TEST_ROOT / "result/phase1292_c029_object_attribute_convergence_contract"
PROTOCOL = OUT / "protocol/preregistration.json"
MATERIAL = OUT / "material/frozen_object_attribute_cases.jsonl"
NATURALNESS = OUT / "material/pre_model_semantic_naturalness_review.json"
MACHINE = OUT / "audit/tokenizer_semantic_program_audit.json"
FINAL = OUT / "analysis/final.json"
AUDIT = OUT / "audit/independent_final_audit.json"

PARTITIONS = ("discovery", "confirmation", "holdout")
ATTRIBUTES = ("color", "material", "location", "size", "shape", "status")
PANELS = ("active", "matched_null", "surface_only", "semantic_neighbor")
SURFACES = ("catalog_prose", "inventory_ledger")
ORDERS = (0, 1, 2)
STATES = (0, 1)
EXPECTED_CASES = 3 * 8 * 6 * 4 * 2 * 3 * 2
TOKEN_PATTERN = re.compile(r"[A-Za-z]+|[0-9]+|[^\w\s]", re.UNICODE)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def lexical_bag(text: str) -> str:
    tokens = [token.lower() for token in TOKEN_PATTERN.findall(text)]
    return digest(sorted(tokens))


def add(checks: list[dict[str, Any]], name: str, passed: bool, detail: Any) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def audit() -> None:
    protocol = read_json(PROTOCOL)
    rows = read_jsonl(MATERIAL)
    review = read_json(NATURALNESS)
    machine = read_json(MACHINE)
    checks: list[dict[str, Any]] = []

    timeless = {key: value for key, value in protocol.items() if key not in {"created_at_utc", "protocol_digest"}}
    add(checks, "protocol_digest_recomputes", digest(timeless) == protocol["protocol_digest"], protocol["protocol_digest"])
    add(checks, "source_hashes_match", protocol["source_hashes"] == {"main": file_sha256(MAIN), "auditor": file_sha256(AUDITOR)}, protocol["source_hashes"])
    add(checks, "phase_campaign_exact", (protocol["phase"], protocol["campaign"]) == (1292, "C029"), [protocol["phase"], protocol["campaign"]])
    add(checks, "selected_object_exact", protocol["selected_object"] == "object_attribute_inverse_lookup", protocol["selected_object"])
    registry_scores = {item["object"]: item["score"] for item in protocol["historical_object_registry"]}
    expected_selected = sorted(registry_scores, key=lambda key: (-registry_scores[key], key))[0]
    add(checks, "selection_rule_replays", expected_selected == protocol["selected_object"], registry_scores)
    add(checks, "material_hash_matches", file_sha256(MATERIAL) == protocol["material"]["material_sha256"], file_sha256(MATERIAL))
    add(checks, "naturalness_hash_matches", file_sha256(NATURALNESS) == protocol["material"]["naturalness_sha256"], file_sha256(NATURALNESS))
    add(checks, "row_count_exact", len(rows) == EXPECTED_CASES, len(rows))
    add(checks, "unique_case_ids", len({row["case_id"] for row in rows}) == len(rows), len({row["case_id"] for row in rows}))

    dimensions = {
        "partitions": sorted({row["partition"] for row in rows}),
        "attributes": sorted({row["attribute"] for row in rows}),
        "panels": sorted({row["panel"] for row in rows}),
        "surfaces": sorted({row["surface"] for row in rows}),
        "orders": sorted({row["candidate_order"] for row in rows}),
        "states": sorted({row["binding_state"] for row in rows}),
    }
    add(checks, "dimensions_exact", dimensions == {
        "partitions": sorted(PARTITIONS), "attributes": sorted(ATTRIBUTES), "panels": sorted(PANELS),
        "surfaces": sorted(SURFACES), "orders": list(ORDERS), "states": list(STATES),
    }, dimensions)
    partition_counts = Counter(row["partition"] for row in rows)
    add(checks, "partition_counts_balanced", set(partition_counts.values()) == {EXPECTED_CASES // 3}, partition_counts)

    vocab = {}
    for partition in PARTITIONS:
        subset = [row for row in rows if row["partition"] == partition]
        vocab[partition] = {
            "entities": {entity for row in subset for entity in row["entities"]},
            "values": {value for row in subset for fields in row["assignments"].values() for value in fields.values()},
        }
    disjoint = all(
        not vocab[left][kind] & vocab[right][kind]
        for kind in ("entities", "values")
        for i, left in enumerate(PARTITIONS)
        for right in PARTITIONS[i + 1:]
    )
    add(checks, "partition_vocabularies_disjoint", disjoint, {p: {k: len(v) for k, v in d.items()} for p, d in vocab.items()})

    gold_ok = True
    prompt_ok = True
    spans_ok = True
    type_ok = True
    for row in rows:
        matches = [entity for entity in row["entities"] if row["assignments"][entity][row["attribute"]] == row["target_value"]]
        gold_ok &= matches == [row["gold_candidate"]]
        prompt_ok &= lexical_bag(row["candidate_prompt"]) == row["prompt_token_multiset_digest"]
        prompt_ok &= row["candidate_prompt"].endswith("Answer:") and row["generation_prompt"].endswith("Answer:")
        spans = row["typed_spans"]
        spans_ok &= len(spans["records"]) == 3 and len(spans["query"]) == 1 and len(spans["answer_boundary"]) == 1
        type_ok &= row["gold_candidate"] in row["entities"] and sorted(row["candidates"]) == sorted(row["entities"])
    add(checks, "gold_recomputes_from_world", gold_ok, "all rows")
    add(checks, "prompt_digest_and_boundary", prompt_ok, "all rows")
    add(checks, "typed_spans_present", spans_ok, "all rows")
    add(checks, "entity_output_type_exact", type_ok, "all rows")

    paired = defaultdict(list)
    for row in rows:
        paired[row["group_id"]].append(row)
    pair_ok = len(paired) == EXPECTED_CASES // 2 and all(len(value) == 2 for value in paired.values())
    active_ok = null_ok = surface_ok = neighbor_ok = True
    active_collision_count = 0
    for pair in paired.values():
        pair.sort(key=lambda row: row["binding_state"])
        left, right = pair
        panel = left["panel"]
        if panel == "active":
            condition = left["gold_candidate"] != right["gold_candidate"] and left["prompt_token_multiset_digest"] == right["prompt_token_multiset_digest"]
            active_ok &= condition
            active_collision_count += int(condition)
        elif panel == "matched_null":
            null_ok &= left["gold_candidate"] == right["gold_candidate"] and left["prompt_token_multiset_digest"] == right["prompt_token_multiset_digest"]
        elif panel == "surface_only":
            surface_ok &= left["gold_candidate"] == right["gold_candidate"] and left["record_order"] != right["record_order"]
        elif panel == "semantic_neighbor":
            neighbor_ok &= left["gold_candidate"] == right["gold_candidate"]
    add(checks, "state_pairs_complete", pair_ok, len(paired))
    add(checks, "active_same_bag_different_gold", active_ok, active_collision_count)
    add(checks, "matched_null_same_bag_same_gold", null_ok, "all pairs")
    add(checks, "surface_only_order_control", surface_ok, "all pairs")
    add(checks, "semantic_neighbor_control", neighbor_ok, "all pairs")

    malformed = [
        row["case_id"] for row in rows
        if "  " in row["candidate_prompt"]
        or row["candidate_prompt"].count("?") != 1
        or any(fragment in row["candidate_prompt"].lower() for fragment in review["forbidden_phrase_inventory"])
    ]
    add(checks, "deterministic_naturalness_lint", not malformed, malformed[:10])
    add(checks, "semantic_review_is_bounded", review["semantic_unique"] is True and review["independent_human_panel"] is False and bool(review["limitation"]), review["limitation"])
    add(checks, "prototype_coverage", review["prototype_count"] == len(SURFACES) * len(ATTRIBUTES), review["prototype_count"])
    add(checks, "machine_audit_passed", machine["all_machine_checks_passed"] is True, machine)
    add(checks, "tokenizer_contract", machine["token_audit"]["all_candidates_single_token"] is True and machine["token_audit"]["candidate_token_lengths"] == [1], machine["token_audit"])
    add(checks, "shortcut_ceiling", machine["program_audit"]["shortcut_ceiling"] <= protocol["thresholds"]["shortcut_program_accuracy_max"], machine["program_audit"])
    add(checks, "thresholds_frozen", protocol["thresholds"] == {
        "finite_fraction_min": 1.0, "overall_candidate_accuracy_min": 0.95,
        "partition_candidate_accuracy_min": 0.94, "panel_candidate_accuracy_min": 0.93,
        "surface_candidate_accuracy_min": 0.93, "base_side_accuracy_min": 0.93,
        "active_pair_success_min": 0.90, "matched_null_pair_success_min": 0.90,
        "surface_only_pair_success_min": 0.90, "semantic_neighbor_pair_success_min": 0.90,
        "candidate_order_triple_success_min": 0.90, "cross_surface_pair_success_min": 0.90,
        "generation_coverage_min": 0.95, "generation_accuracy_min": 0.90,
        "generation_pair_success_min": 0.85, "shortcut_program_accuracy_max": 0.70,
    }, protocol["thresholds"])
    add(checks, "single_model_fp16_only", protocol["model"]["behavior"] == ["qwen3-4b-fp16-cuda-no-quantization"] and protocol["model"]["other_models_authorized"] is False, protocol["model"])
    branches = protocol["failure_and_stop_branches"]
    add(checks, "all_failure_branches_close", all("close_c029" in value for key, value in branches.items() if "fails" in key), branches)
    add(checks, "behavior_before_hidden", any("Behavior and list-free generation" in rule for rule in protocol["freeze_rules"]), protocol["freeze_rules"])
    add(checks, "no_weights_loaded", protocol["model_weights_loaded"] is False, protocol["model_weights_loaded"])

    if FINAL.exists():
        final = read_json(FINAL)
        add(checks, "final_matches_protocol", final["protocol_digest"] == protocol["protocol_digest"] and final["material_sha256"] == file_sha256(MATERIAL), final)

    result = {
        "phase": 1292,
        "campaign": "C029",
        "schema_version": "phase1292.c029.independent_audit.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "auditor_imports_main": False,
        "checks": checks,
        "passed_count": sum(check["passed"] for check in checks),
        "total_count": len(checks),
        "all_checks_passed": all(check["passed"] for check in checks),
        "authorization": "phase1293_qwen3_behavior_only" if all(check["passed"] for check in checks) else "none",
        "protocol_digest": protocol["protocol_digest"],
    }
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(canonical_json({"phase": 1292, "passed": result["passed_count"], "total": result["total_count"], "authorization": result["authorization"]}))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    audit()
