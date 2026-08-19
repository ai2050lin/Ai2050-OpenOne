#!/usr/bin/env python3
"""Independent replay audit for Phase 1294/C030.

This auditor does not import either the C030 compiler or its C029 scaffold.
It reconstructs every controlled prompt from the serialized world state.
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
MAIN = TEST_ROOT / "phase1294_c030_grounded_lookup_contract.py"
AUDITOR = Path(__file__).resolve()
SCAFFOLD = TEST_ROOT / "phase1292_c029_object_attribute_convergence_contract.py"
ERRATUM = TEST_ROOT / "result/phase1293_c029_naturalness_erratum/analysis/final.json"
C029_MACHINE = TEST_ROOT / "result/phase1292_c029_object_attribute_convergence_contract/audit/tokenizer_semantic_program_audit.json"
OUT = TEST_ROOT / "result/phase1294_c030_grounded_lookup_contract"
PROTOCOL = OUT / "protocol/preregistration.json"
MATERIAL = OUT / "material/frozen_grounded_lookup_cases.jsonl"
NATURALNESS = OUT / "material/pre_model_grammar_type_review.json"
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
ATTRIBUTE_LABELS = {
    "color": "color", "material": "material", "location": "storage area",
    "size": "size", "shape": "shape", "status": "status",
}
EXPECTED_THRESHOLDS = {
    "finite_fraction_min": 1.0,
    "overall_candidate_accuracy_min": 0.95,
    "partition_candidate_accuracy_min": 0.94,
    "panel_candidate_accuracy_min": 0.93,
    "surface_candidate_accuracy_min": 0.93,
    "base_side_accuracy_min": 0.93,
    "active_pair_success_min": 0.90,
    "matched_null_pair_success_min": 0.90,
    "surface_only_pair_success_min": 0.90,
    "semantic_neighbor_pair_success_min": 0.90,
    "candidate_order_triple_success_min": 0.90,
    "cross_surface_pair_success_min": 0.90,
    "generation_coverage_min": 0.95,
    "generation_accuracy_min": 0.90,
    "generation_pair_success_min": 0.85,
    "shortcut_program_accuracy_max": 0.70,
}


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def lexical_bag(text: str) -> str:
    return digest(sorted(token.lower() for token in TOKEN_PATTERN.findall(text)))


def article(word: str) -> str:
    return "an" if word[:1].lower() in "aeiou" else "a"


def record(entity: str, fields: dict[str, str], surface: str) -> str:
    if surface == "catalog_prose":
        return (
            f"The sample named {entity} has {article(fields['color'])} {fields['color']} color, "
            f"is made of {fields['material']}, is stored in the {fields['location']} area, "
            f"is {fields['size']} in size, has {article(fields['shape'])} {fields['shape']} shape, "
            f"and is marked {fields['status']}."
        )
    return (
        f"{entity} - color: {fields['color']}; material: {fields['material']}; "
        f"storage area: {fields['location']}; size: {fields['size']}; "
        f"shape: {fields['shape']}; status: {fields['status']}."
    )


def query(attribute: str, value: str, surface: str) -> str:
    if surface == "inventory_ledger":
        return f"Which listed sample has {ATTRIBUTE_LABELS[attribute]}: {value}?"
    if attribute == "color":
        return f"According to the catalog, which sample has {article(value)} {value} color?"
    if attribute == "material":
        return f"According to the catalog, which sample is made of {value}?"
    if attribute == "location":
        return f"According to the catalog, which sample is stored in the {value} area?"
    if attribute == "size":
        return f"According to the catalog, which sample is {value} in size?"
    if attribute == "shape":
        return f"According to the catalog, which sample has {article(value)} {value} shape?"
    return f"According to the catalog, which sample is marked {value}?"


def render(row: dict[str, Any], generation: bool) -> str:
    records = [record(entity, row["assignments"][entity], row["surface"]) for entity in row["record_order"]]
    prefix = "Catalog entries:" if row["surface"] == "catalog_prose" else "Inventory ledger:"
    question = query(row["attribute"], row["target_value"], row["surface"])
    if generation:
        instruction = "Reply with only the sample name. Answer:"
    else:
        instruction = f"Choose exactly one name from {', '.join(row['candidates'])}. Answer:"
    return " ".join([prefix, *records, question, instruction])


def add(checks: list[dict[str, Any]], name: str, passed: bool, detail: Any) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def audit() -> None:
    protocol = load(PROTOCOL)
    rows = read_jsonl(MATERIAL)
    review = load(NATURALNESS)
    machine = load(MACHINE)
    erratum = load(ERRATUM)
    c029_machine = load(C029_MACHINE)
    checks: list[dict[str, Any]] = []

    timeless = {key: value for key, value in protocol.items() if key not in {"created_at_utc", "protocol_digest"}}
    add(checks, "protocol_digest_recomputes", digest(timeless) == protocol["protocol_digest"], protocol["protocol_digest"])
    add(checks, "source_hashes_match", protocol["source_hashes"] == {
        "main": sha(MAIN), "auditor": sha(AUDITOR), "scaffold": sha(SCAFFOLD)
    }, protocol["source_hashes"])
    add(checks, "phase_campaign_exact", (protocol["phase"], protocol["campaign"]) == (1294, "C030"), [protocol["phase"], protocol["campaign"]])
    add(checks, "c029_closed_before_c030", erratum["authorization"] == "close_c029_before_behavior" and protocol["upstream"]["c029_erratum_sha256"] == sha(ERRATUM), erratum["authorization"])
    add(checks, "construct_and_type_exact", protocol["construct"] == {
        "world_state": "finite explicit map Entity x Attribute -> Value",
        "query": "unique inverse lookup (Attribute, Value) -> Entity",
        "type_signature": "(WorldState, Attribute, Value) -> Entity",
        "operation_requested_from_model": False,
        "gold_source": "explicit mapping, independently recomputed",
    }, protocol["construct"])
    add(checks, "material_hash_matches", sha(MATERIAL) == protocol["material"]["material_sha256"], sha(MATERIAL))
    add(checks, "naturalness_hash_matches", sha(NATURALNESS) == protocol["material"]["naturalness_sha256"], sha(NATURALNESS))
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
        "partitions": sorted(PARTITIONS), "attributes": sorted(ATTRIBUTES),
        "panels": sorted(PANELS), "surfaces": sorted(SURFACES),
        "orders": list(ORDERS), "states": list(STATES),
    }, dimensions)
    counts = Counter(row["partition"] for row in rows)
    add(checks, "partition_counts_balanced", set(counts.values()) == {EXPECTED_CASES // 3}, dict(counts))

    selected_names = set(machine["token_audit"]["selected_names"])
    prior_names = set(c029_machine["token_audit"]["selected_names"])
    row_names = {entity for row in rows for entity in row["entities"]}
    add(checks, "entity_inventory_exact", row_names == selected_names and len(row_names) == 72, len(row_names))
    add(checks, "c029_entities_disjoint", not row_names & prior_names and protocol["material"]["c029_entity_overlap"] == [], sorted(row_names & prior_names))
    add(checks, "c029_values_disjoint", machine["c029_value_overlap"] == [] and protocol["material"]["c029_value_overlap"] == [], machine["c029_value_overlap"])
    partition_vocab = {}
    for partition in PARTITIONS:
        subset = [row for row in rows if row["partition"] == partition]
        partition_vocab[partition] = {
            "entities": {entity for row in subset for entity in row["entities"]},
            "values": {value for row in subset for fields in row["assignments"].values() for value in fields.values()},
        }
    partition_disjoint = all(
        not partition_vocab[left][kind] & partition_vocab[right][kind]
        for kind in ("entities", "values")
        for index, left in enumerate(PARTITIONS)
        for right in PARTITIONS[index + 1:]
    )
    add(checks, "partition_vocabularies_disjoint", partition_disjoint, {
        p: {kind: len(values) for kind, values in inventory.items()} for p, inventory in partition_vocab.items()
    })

    gold_ok = exact_render_ok = digest_ok = spans_ok = type_ok = True
    render_failures: list[str] = []
    grammar_failures: list[str] = []
    for row in rows:
        matches = [entity for entity in row["entities"] if row["assignments"][entity][row["attribute"]] == row["target_value"]]
        gold_ok &= matches == [row["gold_candidate"]]
        expected_candidate = render(row, generation=False)
        expected_generation = render(row, generation=True)
        if row["candidate_prompt"] != expected_candidate or row["generation_prompt"] != expected_generation:
            exact_render_ok = False
            render_failures.append(row["case_id"])
        digest_ok &= lexical_bag(row["candidate_prompt"]) == row["prompt_token_multiset_digest"]
        spans = row["typed_spans"]
        spans_ok &= len(spans["records"]) == 3 and len(spans["query"]) == 1 and len(spans["answer_boundary"]) == 1
        type_ok &= row["gold_candidate"] in row["entities"] and sorted(row["candidates"]) == sorted(row["entities"])
        text = row["candidate_prompt"]
        if "  " in text or text.count("?") != 1 or not text.endswith("Answer:"):
            grammar_failures.append(row["case_id"])
        for match in re.finditer(r"\b(a|an) ([A-Za-z-]+) (color|shape)\b", text):
            if match.group(1) != article(match.group(2)):
                grammar_failures.append(row["case_id"])
        if any(fragment in text.lower() for fragment in (
            "does not apply the", "unassigned alternative the", "stored in the rooftop,", "a azure", "a emerald"
        )):
            grammar_failures.append(row["case_id"])
    add(checks, "gold_recomputes_from_world", gold_ok, "all rows")
    add(checks, "independent_exact_prompt_replay", exact_render_ok, render_failures[:10])
    add(checks, "prompt_bag_digest_recomputes", digest_ok, "all rows")
    add(checks, "typed_spans_present", spans_ok, "all rows")
    add(checks, "entity_output_type_exact", type_ok, "all rows")
    add(checks, "independent_grammar_lint", not grammar_failures, grammar_failures[:10])

    pairs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        pairs[row["group_id"]].append(row)
    add(checks, "state_pairs_complete", len(pairs) == EXPECTED_CASES // 2 and all(len(pair) == 2 for pair in pairs.values()), len(pairs))
    pair_results = {panel: True for panel in PANELS}
    for pair in pairs.values():
        left, right = sorted(pair, key=lambda row: row["binding_state"])
        same_bag = left["prompt_token_multiset_digest"] == right["prompt_token_multiset_digest"]
        same_gold = left["gold_candidate"] == right["gold_candidate"]
        if left["panel"] == "active":
            pair_results["active"] &= same_bag and not same_gold
        elif left["panel"] == "matched_null":
            pair_results["matched_null"] &= same_bag and same_gold
        elif left["panel"] == "surface_only":
            pair_results["surface_only"] &= same_gold and left["record_order"] != right["record_order"]
        else:
            pair_results["semantic_neighbor"] &= same_gold
    for panel, passed in pair_results.items():
        add(checks, f"{panel}_pair_semantics", passed, panel)

    add(checks, "naturalness_review_pre_model", review["reviewed_before_any_c030_weight_load"] is True and review["all_checks_passed"] is True and not review["issues"], review["reviewer_type"])
    add(checks, "prototype_coverage", review["prototype_count"] == len(SURFACES) * len(ATTRIBUTES), review["prototype_count"])
    add(checks, "naturalness_claim_bounded", review["independent_human_panel"] is False and bool(review["limitation"]), review["limitation"])
    add(checks, "machine_audit_passed", machine["all_machine_checks_passed"] is True, machine["all_machine_checks_passed"])
    add(checks, "tokenizer_contract", machine["token_audit"]["all_candidates_single_token"] is True and machine["token_audit"]["candidate_token_lengths"] == [1], machine["token_audit"])
    add(checks, "shortcut_ceiling", machine["program_audit"]["shortcut_ceiling"] <= EXPECTED_THRESHOLDS["shortcut_program_accuracy_max"], machine["program_audit"]["shortcut_ceiling"])
    add(checks, "thresholds_exact", protocol["thresholds"] == EXPECTED_THRESHOLDS, protocol["thresholds"])
    add(checks, "qwen3_fp16_only", protocol["model"]["behavior"] == ["qwen3-4b-fp16-cuda-no-quantization"] and protocol["model"]["other_models_authorized"] is False, protocol["model"])
    branches = protocol["failure_and_stop_branches"]
    add(checks, "every_failure_closes", all("close_c030" in value for key, value in branches.items() if "fails" in key), branches)
    add(checks, "behavior_before_hidden", any("before hidden state" in rule for rule in protocol["freeze_rules"]), protocol["freeze_rules"])
    add(checks, "no_weights_loaded", protocol["model_weights_loaded"] is False, protocol["model_weights_loaded"])

    if FINAL.exists():
        final = load(FINAL)
        add(checks, "final_matches_frozen_artifacts", final["protocol_digest"] == protocol["protocol_digest"] and final["material_sha256"] == sha(MATERIAL), final)

    passed = all(check["passed"] for check in checks)
    result = {
        "phase": 1294,
        "campaign": "C030",
        "schema_version": "phase1294.c030.independent_audit.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "auditor_imports_main_or_scaffold": False,
        "checks": checks,
        "passed_count": sum(check["passed"] for check in checks),
        "total_count": len(checks),
        "all_checks_passed": passed,
        "authorization": "phase1295_qwen3_behavior_only" if passed else "none",
        "protocol_digest": protocol["protocol_digest"],
    }
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(canonical({"phase": 1294, "passed": result["passed_count"], "total": result["total_count"], "authorization": result["authorization"]}))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    audit()
