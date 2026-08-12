#!/usr/bin/env python3
"""Independent pre/result/final audits for Phase1235.

The auditor does not import the Phase1235 implementation.  It reconstructs
the finite nuisance-program ceiling, orthogonal material contracts, all raw
readout decisions, typed behavior gates, and final claim permissions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import string
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests/glm5"
MAIN_SCRIPT = TEST_ROOT / "phase1235_qwen3_typed_generation_compiler_boundary.py"
AUDIT_SCRIPT = Path(__file__).resolve()
P1220_SCRIPT = TEST_ROOT / "phase1220_object_relation_value_master_task.py"
P1221_SCRIPT = TEST_ROOT / "phase1221_typed_operation_behavior_and_error_fingerprints.py"

OUT_ROOT = TEST_ROOT / "result/phase1235_qwen3_typed_generation_compiler_boundary"
CONTRACT_PATH = OUT_ROOT / "protocol/preregistration.json"
MATERIAL_PATH = OUT_ROOT / "material/orthogonal_readout_worlds.jsonl"
MANIFEST_PATH = OUT_ROOT / "protocol/qwen3_manifest.jsonl"
PROGRAM_PATH = OUT_ROOT / "protocol/depth2_nuisance_program_audit.json"
PLAN_PATH = OUT_ROOT / "protocol/frozen_batch_plan.json"
PREAUDIT_PATH = OUT_ROOT / "audit/independent_preaudit.json"
RAW_PATH = OUT_ROOT / "behavior/qwen3/raw_behavior.jsonl"
SUMMARY_PATH = OUT_ROOT / "behavior/qwen3/run_summary.json"
RESULT_AUDIT_PATH = OUT_ROOT / "audit/independent_result_audit.json"
FINAL_PATH = OUT_ROOT / "analysis/final.json"
FINAL_AUDIT_PATH = OUT_ROOT / "audit/independent_final_audit.json"

UPSTREAM_FINAL_PATH = TEST_ROOT / "result/phase1234_qwen3_k199_registry_sealed_confirmation/analysis/final.json"
UPSTREAM_AUDIT_PATH = TEST_ROOT / "result/phase1234_qwen3_k199_registry_sealed_confirmation/audit/independent_final_audit.json"
EXPECTED_UPSTREAM_FINAL = "4d7d28a7c969d145a95adda1fc5cacfc5595fff762c15d2dec914b048ab7d63b"
EXPECTED_UPSTREAM_AUDIT = "90ac15451fbf7239912982419c3c386ce2f49126febaddf9146b28a61ba93708"

AXES = ("object_surface", "value_surface", "template_surface")
PARTITIONS = ("discovery", "confirmation")
LEVELS = (0, 1)
BINDINGS = (0, 1)
EXPECTED_ROWS = 1536
EXPECTED_WORLDS = 96
ROWS_PER_WORLD = 16
PROGRAM_DEPTH = 2
PROGRAM_CEILING = 0.80
GENERATION_BUDGET = 24
TIE_TOLERANCE = 1e-7
THRESHOLDS = {
    "finite_rate": 1.0,
    "choice_candidate_worst_surface": 0.95,
    "bare_candidate_worst_surface": 0.95,
    "trie_worst_surface": 0.95,
    "candidate_query_quartet": 0.85,
    "candidate_binding_pair": 0.90,
    "candidate_surface_pair": 0.90,
    "bare_exact_worst_surface": 0.90,
    "bare_content_worst_surface": 0.95,
    "cued_exact_worst_surface": 0.90,
    "cued_content_worst_surface": 0.95,
    "short_binding_pair": 0.85,
    "short_surface_pair": 0.85,
    "sentence_content_worst_surface": 0.90,
    "sentence_binding_pair": 0.85,
    "sentence_surface_pair": 0.85,
    "natural_content_worst_surface": 0.90,
    "natural_binding_pair": 0.85,
    "natural_surface_pair": 0.85,
    "program_ceiling": PROGRAM_CEILING,
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
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
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def strip_digest(value: dict[str, Any], key: str) -> dict[str, Any]:
    return {name: item for name, item in value.items() if name != key}


def grouped(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    values: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get(key) is not None:
            values[str(row[key])].append(row)
    return values


def lexical_multiset(text: str) -> list[str]:
    return sorted(re.findall(r"[a-z]+(?:'[a-z]+)?|\d+", text.lower()))


def source_hashes() -> dict[str, str]:
    return {
        "main": file_sha256(MAIN_SCRIPT),
        "audit": file_sha256(AUDIT_SCRIPT),
        "phase1220_scorer": file_sha256(P1220_SCRIPT),
        "phase1221_scorer": file_sha256(P1221_SCRIPT),
    }


def base_program_predictions(row: dict[str, Any]) -> dict[str, str]:
    candidates = row["candidates"]
    displayed = row["displayed_assignments"]
    objects = row["objects"]
    query = int(row["query_index"])
    programs = {f"candidate_position_{index}": row["candidate_order"][index] for index in range(5)}
    programs.update({f"fixed_object_{index}": displayed[objects[index]] for index in range(4)})
    programs.update(
        {
            "first_record": displayed[objects[row["record_order_indices"][0]]],
            "last_record": displayed[objects[row["record_order_indices"][-1]]],
            "next_object": displayed[objects[(query + 1) % 4]],
            "opposite_object": displayed[objects[(query + 2) % 4]],
            "previous_object": displayed[objects[(query + 3) % 4]],
            "unused_value": row["unused_value"],
            "lexical_first": sorted(candidates)[0],
            "lexical_last": sorted(candidates)[-1],
        }
    )
    return programs


def nuisance_features(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "axis": row["axis"],
        "partition": row["partition"],
        "surface_level": row["surface_level"],
        "binding_state": row["binding_state"],
        "record_first_index": row["record_order_indices"][0],
        "unused_candidate_position": row["candidate_order"].index(row["unused_value"]),
    }


def depth_tree(rows: list[dict[str, Any]], depth: int) -> tuple[float, dict[str, Any]]:
    names = sorted(base_program_predictions(rows[0]))
    predictions = [base_program_predictions(row) for row in rows]
    features = [nuisance_features(row) for row in rows]
    conditions = sorted({(name, canonical_json(value)) for feature in features for name, value in feature.items()})
    decoded = {(name, encoded): json.loads(encoded) for name, encoded in conditions}
    cache: dict[tuple[tuple[int, ...], int], tuple[int, dict[str, Any]]] = {}

    def recurse(indices: tuple[int, ...], remaining: int) -> tuple[int, dict[str, Any]]:
        key = (indices, remaining)
        if key in cache:
            return cache[key]
        scores = {name: sum(predictions[index][name] == rows[index]["gold"] for index in indices) for name in names}
        maximum = max(scores.values())
        leaf = min(name for name in names if scores[name] == maximum)
        best_score = scores[leaf]
        best_tree: dict[str, Any] = {"leaf_program": leaf, "correct": best_score, "n": len(indices)}
        if remaining:
            for name, encoded in conditions:
                value = decoded[(name, encoded)]
                left = tuple(index for index in indices if features[index][name] == value)
                right = tuple(index for index in indices if features[index][name] != value)
                if not left or not right:
                    continue
                left_score, left_tree = recurse(left, remaining - 1)
                right_score, right_tree = recurse(right, remaining - 1)
                score = left_score + right_score
                tree = {
                    "condition": {"feature": name, "equals": value},
                    "true": left_tree,
                    "false": right_tree,
                    "correct": score,
                    "n": len(indices),
                }
                if score > best_score or (score == best_score and canonical_json(tree) < canonical_json(best_tree)):
                    best_score, best_tree = score, tree
        cache[key] = (best_score, best_tree)
        return cache[key]

    correct, witness = recurse(tuple(range(len(rows))), depth)
    return correct / len(rows), witness


def independent_program(material: list[dict[str, Any]], manifest: list[dict[str, Any]]) -> dict[str, Any]:
    split_results: dict[str, Any] = {}
    for axis in AXES:
        for partition in PARTITIONS:
            selected = [row for row in material if row["axis"] == axis and row["partition"] == partition]
            base = {
                name: sum(base_program_predictions(row)[name] == row["gold"] for row in selected) / len(selected)
                for name in sorted(base_program_predictions(selected[0]))
            }
            ceiling, tree = depth_tree(selected, PROGRAM_DEPTH)
            split_results[f"{axis}|{partition}"] = {
                "base_program_accuracies": base,
                "maximum_base_program_accuracy": max(base.values()),
                "depth2_conditional_program_accuracy": ceiling,
                "depth2_witness_tree": tree,
                "construct_gate": ceiling <= PROGRAM_CEILING,
            }
    manifest_by_id = {row["item_id"]: row for row in manifest}
    query = grouped(material, "query_group_id")
    surface = grouped(material, "surface_pair_id")
    binding = grouped(material, "binding_pair_id")
    collisions = {
        "query_quartets_complete": all(len(cell) == 4 and len({row["gold_slot"] for row in cell}) == 4 for cell in query.values()),
        "surface_pairs_semantically_invariant": all(len(cell) == 2 and len({row["gold_slot"] for row in cell}) == 1 for cell in surface.values()),
        "binding_pairs_discriminating": all(
            len(cell) == 2
            and len({row["gold_slot"] for row in cell}) == 2
            and len({digest(lexical_multiset(row["prompts"]["bare"])) for row in cell}) == 1
            and len({digest(sorted(manifest_by_id[row["item_id"]]["bare_input_ids"])) for row in cell}) == 1
            for cell in binding.values()
        ),
        "unused_never_gold": all(row["unused_value"] != row["gold"] for row in material),
        "five_candidates_four_assigned": all(len(row["candidates"]) == 5 and len(set(row["displayed_slots"].values())) == 4 for row in material),
    }
    target = sum(row["displayed_assignments"][row["query_object"]] == row["gold"] for row in material) / len(material)
    return {
        "split_results": split_results,
        "collision_group_counts": {"query_quartets": len(query), "surface_pairs": len(surface), "binding_pairs": len(binding)},
        "collision_checks": collisions,
        "target_equivalent_witness_accuracy": target,
        "program_construct_gate": all(cell["construct_gate"] for cell in split_results.values()) and all(collisions.values()) and target == 1.0,
    }


def audit_value(kind: str, checks: dict[str, bool], extra: dict[str, Any] | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {
        "phase": 1235,
        "schema_version": f"phase1235.independent_{kind}_audit.v1",
        "created_at_utc": utc_now(),
        "checks": checks,
        "passed_check_count": sum(bool(item) for item in checks.values()),
        "total_check_count": len(checks),
        "failed_checks": sorted(name for name, item in checks.items() if not item),
        "all_checks_passed": all(checks.values()),
    }
    if extra:
        value.update(extra)
    value["audit_digest"] = digest(value)
    return value


def preaudit() -> None:
    if PREAUDIT_PATH.exists():
        raise RuntimeError("Phase1235 preaudit exists")
    contract = read_json(CONTRACT_PATH)
    material = read_jsonl(MATERIAL_PATH)
    manifest = read_jsonl(MANIFEST_PATH)
    program = read_json(PROGRAM_PATH)
    plan = read_json(PLAN_PATH)
    upstream = read_json(UPSTREAM_FINAL_PATH)
    upstream_audit = read_json(UPSTREAM_AUDIT_PATH)
    independent = independent_program(material, manifest)
    material_by_id = {row["item_id"]: row for row in material}
    manifest_by_id = {row["item_id"]: row for row in manifest}
    worlds = grouped(material, "world_id")

    orthogonal = True
    for cell in grouped(material, "surface_pair_id").values():
        left, right = sorted(cell, key=lambda row: row["surface_level"])
        if left["axis"] == "object_surface":
            orthogonal = orthogonal and left["objects"] != right["objects"] and left["candidates"] == right["candidates"] and left["template_style"] == right["template_style"]
        elif left["axis"] == "value_surface":
            orthogonal = orthogonal and left["objects"] == right["objects"] and left["candidates"] != right["candidates"] and left["template_style"] == right["template_style"]
        else:
            orthogonal = orthogonal and left["objects"] == right["objects"] and left["candidates"] == right["candidates"] and left["template_style"] != right["template_style"]
        orthogonal = orthogonal and left["gold_slot"] == right["gold_slot"]

    axis_vocab = {
        axis: {
            "objects": {value for row in material if row["axis"] == axis for value in row["objects"]},
            "labels": {value for row in material if row["axis"] == axis for value in row["candidates"]},
        }
        for axis in AXES
    }
    disjoint = all(
        not axis_vocab[left][kind] & axis_vocab[right][kind]
        for left_index, left in enumerate(AXES)
        for right in AXES[left_index + 1 :]
        for kind in ("objects", "labels")
    )
    position_balanced = True
    for axis in AXES:
        for partition in PARTITIONS:
            for level in LEVELS:
                for binding in BINDINGS:
                    counts = Counter(row["gold_position"] for row in material if row["axis"] == axis and row["partition"] == partition and row["surface_level"] == level and row["binding_state"] == binding)
                    position_balanced = position_balanced and set(counts) == set(range(5)) and max(counts.values()) - min(counts.values()) <= 1
    suffix_equal = all(
        len({len(tokens) for tokens in row[f"{readout}_candidate_token_ids"].values()}) == 1
        for row in manifest for readout in ("choice", "bare")
    )
    plan_counts = all(
        plan["views"][readout]["length_counts"]
        == {str(key): value for key, value in sorted(Counter(row[f"{readout}_input_token_count"] for row in manifest).items())}
        for readout in ("choice", "bare", "cued", "sentence", "natural")
    )
    checks = {
        "upstream_final_pinned": upstream.get("final_digest") == EXPECTED_UPSTREAM_FINAL,
        "upstream_audit_pinned": upstream_audit.get("audit_digest") == EXPECTED_UPSTREAM_AUDIT and upstream_audit.get("all_checks_passed") is True,
        "upstream_failure_retained": upstream["authorization"]["future_response_phase"] is False and contract["upstream"]["phase1234_failure_not_reclassified"] is True,
        "contract_self_digest": contract["contract_digest"] == digest(strip_digest(contract, "contract_digest")),
        "source_hashes": contract["source_hashes"] == source_hashes(),
        "material_digest": contract["material"]["material_digest"] == digest(material),
        "manifest_digest": contract["interface"]["tokenizer_summary"]["manifest_digest"] == digest(manifest),
        "program_self_digest": program["program_audit_digest"] == digest(strip_digest(program, "program_audit_digest")),
        "plan_self_digest": plan["plan_digest"] == digest(strip_digest(plan, "plan_digest")),
        "row_cardinality": len(material) == len(manifest) == EXPECTED_ROWS,
        "world_cardinality": len(worlds) == EXPECTED_WORLDS and all(len(cell) == ROWS_PER_WORLD for cell in worlds.values()),
        "axis_partition_worlds": all(len({row["world_id"] for row in material if row["axis"] == axis and row["partition"] == partition}) == 16 for axis in AXES for partition in PARTITIONS),
        "world_cells_complete": all({(row["surface_level"], row["binding_state"], row["query_index"]) for row in cell} == {(level, binding, query) for level in LEVELS for binding in BINDINGS for query in range(4)} for cell in worlds.values()),
        "unique_ids": len(material_by_id) == len(manifest_by_id) == EXPECTED_ROWS,
        "material_row_digests": all(row["row_digest"] == digest(strip_digest(row, "row_digest")) for row in material),
        "manifest_row_digests": all(row["manifest_row_digest"] == digest(strip_digest(row, "manifest_row_digest")) for row in manifest),
        "material_manifest_links": all(row["material_row_digest"] == material_by_id[row["item_id"]]["row_digest"] for row in manifest),
        "execution_order": [row["execution_index"] for row in manifest] == list(range(EXPECTED_ROWS)) and plan["execution_item_ids"] == [row["item_id"] for row in manifest],
        "orthogonal_surface_pairs": orthogonal,
        "axis_lexicons_disjoint": disjoint,
        "new_lexicon": contract["material"]["lexicon_audit"]["upstream_exact_overlap_count"] == 0,
        "gold_positions_balanced": position_balanced,
        "candidate_suffix_lengths": suffix_equal,
        "bare_gold_suffix_links": all(row["bare_gold_token_ids"] == row["bare_candidate_token_ids"][row["gold"]] for row in manifest),
        "tokenizer_gate": contract["interface"]["tokenizer_summary"]["tokenizer_gate"] is True,
        "max_input": contract["interface"]["tokenizer_summary"]["maximum_input_length"] <= 320,
        "plan_counts": plan_counts,
        "same_generation_budget": contract["interface"]["same_budget_across_free_readouts"] is True and contract["interface"]["greedy_generation_budget"] == GENERATION_BUDGET,
        "query_group_count": independent["collision_group_counts"]["query_quartets"] == 384,
        "pair_group_counts": independent["collision_group_counts"]["surface_pairs"] == 768 and independent["collision_group_counts"]["binding_pairs"] == 768,
        "collision_checks": all(independent["collision_checks"].values()),
        "program_splits_recomputed": all(independent["split_results"][key] == program["split_results"][key] for key in independent["split_results"]),
        "program_gate": independent["program_construct_gate"] is program["program_construct_gate"] is True,
        "target_witness": independent["target_equivalent_witness_accuracy"] == program["target_equivalent_witness_accuracy"] == 1.0,
        "thresholds": contract["thresholds"] == THRESHOLDS,
        "behavior_only": contract["execution"]["hidden_states"] is False and contract["execution"]["attentions"] is False and contract["execution"]["interventions"] is False and contract["execution"]["cross_model"] is False,
        "no_model_weights_materialization": contract["interface"]["tokenizer_summary"]["model_weights_loaded"] is False,
        "no_outputs_before_run": not RAW_PATH.exists() and not SUMMARY_PATH.exists(),
    }
    value = audit_value(
        "preaudit",
        checks,
        {
            "contract_digest": contract["contract_digest"],
            "material_digest": digest(material),
            "manifest_digest": digest(manifest),
            "program_audit_digest": program["program_audit_digest"],
            "independent_program_audit": independent,
        },
    )
    write_json(PREAUDIT_PATH, value)
    print(canonical_json({"status": "phase1235_preaudit", "passed": value["all_checks_passed"], "checks": f'{value["passed_check_count"]}/{value["total_check_count"]}', "failed": value["failed_checks"], "audit_digest": value["audit_digest"]}))
    if not value["all_checks_passed"]:
        raise RuntimeError(f"preaudit failed: {value['failed_checks']}")


def normalize_text(text: str) -> str:
    value = text.strip() if text.strip() else ""
    value = value.strip().strip(string.whitespace + string.punctuation)
    return re.sub(r"\s+", " ", value.lower())


def phrase_present(normalized: str, phrase: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(normalize_text(phrase))}(?!\w)", normalized) is not None


def parse_output(generation: dict[str, Any], candidates: list[str], gold: str, query_object: str, expected: str) -> dict[str, Any]:
    normalized = normalize_text(generation["generated_text"])
    mentions = [candidate for candidate in candidates if phrase_present(normalized, candidate)]
    prediction = mentions[0] if len(mentions) == 1 else None
    exact = normalized == normalize_text(expected)
    content = prediction == gold
    words = normalize_text(gold).split()
    if exact:
        category = "exact"
    elif content:
        category = "gold_with_extra"
    elif len(mentions) > 1:
        category = "multiple_candidates"
    elif prediction is not None:
        category = "wrong_complete_candidate"
    elif words and words[0] in normalized.split():
        category = "gold_prefix_fragment"
    elif len(words) > 1 and words[-1] in normalized.split():
        category = "gold_suffix_fragment"
    elif phrase_present(normalized, query_object):
        category = "query_object_restatement"
    elif "marker" in normalized.split():
        category = "relation_restatement"
    elif generation["reached_budget"]:
        category = "budget_without_candidate"
    else:
        category = "other_unparsed"
    return {"normalized_text": normalized, "mentioned_candidates": mentions, "prediction": prediction, "exact": exact, "content_correct": content, "error_category": category}


def argmax_set(scores: dict[str, float]) -> list[str]:
    maximum = max(scores.values())
    return sorted(name for name, value in scores.items() if maximum - value <= TIE_TOLERANCE)


def rate(rows: list[dict[str, Any]], field: str) -> float:
    return sum(bool(row[field]) for row in rows) / len(rows) if rows else float("nan")


READOUT_FIELDS = {
    "choice": ("choice_correct", "choice_prediction_slot"),
    "bare_candidate": ("bare_candidate_correct", "bare_candidate_prediction_slot"),
    "trie": ("trie_correct", "trie_prediction_slot"),
    "bare": ("bare_content_correct", "bare_prediction_slot"),
    "cued": ("cued_content_correct", "cued_prediction_slot"),
    "sentence": ("sentence_content_correct", "sentence_prediction_slot"),
    "natural": ("natural_content_correct", "natural_prediction_slot"),
}


def group_success(rows: list[dict[str, Any]], key: str, readout: str, size: int, mode: str) -> float:
    correct_field, slot_field = READOUT_FIELDS[readout]
    outcomes: list[bool] = []
    for cell in grouped(rows, key).values():
        slots = [row[slot_field] for row in cell]
        success = len(cell) == size and all(row[correct_field] for row in cell)
        if mode in ("distinct", "different_slot"):
            success = success and len(set(slots)) == size
        elif mode == "same_slot":
            success = success and len(set(slots)) == 1
        else:
            raise ValueError(mode)
        outcomes.append(success)
    return sum(outcomes) / len(outcomes) if outcomes else float("nan")


def worst_surface(rows: list[dict[str, Any]], field: str) -> tuple[float, dict[str, float]]:
    cells = {
        f"level{level}|binding{binding}": rate([row for row in rows if row["surface_level"] == level and row["binding_state"] == binding], field)
        for level in LEVELS for binding in BINDINGS
    }
    return min(cells.values()), cells


def independent_ledgers(raw: list[dict[str, Any]], program: dict[str, Any]) -> dict[str, Any]:
    cells: dict[str, Any] = {}
    candidate_passes: list[bool] = []
    short_passes: list[bool] = []
    sentence_passes: list[bool] = []
    natural_passes: list[bool] = []
    metric_fields = (
        ("choice", "choice_correct"), ("bare_candidate", "bare_candidate_correct"),
        ("trie", "trie_correct"), ("bare_exact", "bare_exact"),
        ("bare_content", "bare_content_correct"), ("cued_exact", "cued_exact"),
        ("cued_content", "cued_content_correct"), ("sentence_exact", "sentence_exact"),
        ("sentence_content", "sentence_content_correct"), ("natural_exact", "natural_exact"),
        ("natural_content", "natural_content_correct"),
    )
    for axis in AXES:
        for partition in PARTITIONS:
            key = f"{axis}|{partition}"
            selected = [row for row in raw if row["axis"] == axis and row["partition"] == partition]
            worst: dict[str, float] = {}
            by_surface: dict[str, Any] = {}
            for name, field in metric_fields:
                worst[name], by_surface[name] = worst_surface(selected, field)
            metrics = {
                "case_count": len(selected),
                "finite_rate": sum(row["choice_finite"] and row["bare_candidate_finite"] for row in selected) / len(selected),
                "overall": {name: rate(selected, field) for name, field in metric_fields},
                "worst_surface": worst,
                "by_surface": by_surface,
                "query_quartet": {name: group_success(selected, "query_group_id", name, 4, "distinct") for name in READOUT_FIELDS},
                "binding_pair": {name: group_success(selected, "binding_pair_id", name, 2, "different_slot") for name in READOUT_FIELDS},
                "surface_pair": {name: group_success(selected, "surface_pair_id", name, 2, "same_slot") for name in READOUT_FIELDS},
                "teacher_all_top1": {readout: sum(row[f"{readout}_teacher"]["all_gold_tokens_top1"] for row in selected) / len(selected) for readout in ("bare", "cued", "sentence")},
                "teacher_first_not_top1": {readout: dict(Counter(str(row[f"{readout}_teacher"]["first_not_top1_index"]) for row in selected)) for readout in ("bare", "cued", "sentence")},
                "error_categories": {readout: dict(Counter(row[f"{readout}_parse"]["error_category"] for row in selected)) for readout in ("bare", "cued", "sentence", "natural")},
                "depth2_program_ceiling": program["split_results"][key]["depth2_conditional_program_accuracy"],
            }
            candidate_gates = {
                "finite": metrics["finite_rate"] >= THRESHOLDS["finite_rate"],
                "choice_surface": worst["choice"] >= THRESHOLDS["choice_candidate_worst_surface"],
                "bare_candidate_surface": worst["bare_candidate"] >= THRESHOLDS["bare_candidate_worst_surface"],
                "trie_surface": worst["trie"] >= THRESHOLDS["trie_worst_surface"],
                "query_quartet": min(metrics["query_quartet"][name] for name in ("choice", "bare_candidate", "trie")) >= THRESHOLDS["candidate_query_quartet"],
                "binding_pair": min(metrics["binding_pair"][name] for name in ("choice", "bare_candidate", "trie")) >= THRESHOLDS["candidate_binding_pair"],
                "surface_pair": min(metrics["surface_pair"][name] for name in ("choice", "bare_candidate", "trie")) >= THRESHOLDS["candidate_surface_pair"],
                "program": metrics["depth2_program_ceiling"] <= THRESHOLDS["program_ceiling"],
            }
            short_gates = {
                "bare_exact": worst["bare_exact"] >= THRESHOLDS["bare_exact_worst_surface"],
                "bare_content": worst["bare_content"] >= THRESHOLDS["bare_content_worst_surface"],
                "cued_exact": worst["cued_exact"] >= THRESHOLDS["cued_exact_worst_surface"],
                "cued_content": worst["cued_content"] >= THRESHOLDS["cued_content_worst_surface"],
                "binding_pair": min(metrics["binding_pair"][name] for name in ("bare", "cued")) >= THRESHOLDS["short_binding_pair"],
                "surface_pair": min(metrics["surface_pair"][name] for name in ("bare", "cued")) >= THRESHOLDS["short_surface_pair"],
            }
            sentence_gates = {
                "content_surface": worst["sentence_content"] >= THRESHOLDS["sentence_content_worst_surface"],
                "binding_pair": metrics["binding_pair"]["sentence"] >= THRESHOLDS["sentence_binding_pair"],
                "surface_pair": metrics["surface_pair"]["sentence"] >= THRESHOLDS["sentence_surface_pair"],
            }
            natural_gates = {
                "content_surface": worst["natural_content"] >= THRESHOLDS["natural_content_worst_surface"],
                "binding_pair": metrics["binding_pair"]["natural"] >= THRESHOLDS["natural_binding_pair"],
                "surface_pair": metrics["surface_pair"]["natural"] >= THRESHOLDS["natural_surface_pair"],
            }
            typed = {"candidate_selection": all(candidate_gates.values()), "short_string": all(short_gates.values()), "sentence": all(sentence_gates.values()), "natural": all(natural_gates.values())}
            cells[key] = {"metrics": metrics, "gates": {"candidate_selection": candidate_gates, "short_string": short_gates, "sentence": sentence_gates, "natural": natural_gates}, "typed_pass": typed}
            candidate_passes.append(typed["candidate_selection"])
            short_passes.append(typed["short_string"])
            sentence_passes.append(typed["sentence"])
            natural_passes.append(typed["natural"])
    typed_global = {"program_construct": bool(program["program_construct_gate"]), "candidate_selection": all(candidate_passes), "short_string": all(short_passes), "sentence": all(sentence_passes), "natural": all(natural_passes)}
    cross = all(typed_global.values())
    return {
        "axis_partition_cells": cells,
        "typed_global_gates": typed_global,
        "cross_readout_gate": cross,
        "future_response_eligibility": cross,
        "overall": {name: rate(raw, field) for name, field in metric_fields},
        "nonfinite_count": sum(not (row["choice_finite"] and row["bare_candidate_finite"]) for row in raw),
        "choice_tie_count": sum(len(row["choice_argmax_set"]) != 1 for row in raw),
        "bare_candidate_tie_count": sum(len(row["bare_argmax_set"]) != 1 for row in raw),
    }


def result_audit() -> None:
    if RESULT_AUDIT_PATH.exists():
        raise RuntimeError("Phase1235 result audit exists")
    contract = read_json(CONTRACT_PATH)
    material = read_jsonl(MATERIAL_PATH)
    manifest = read_jsonl(MANIFEST_PATH)
    program = read_json(PROGRAM_PATH)
    plan = read_json(PLAN_PATH)
    pre = read_json(PREAUDIT_PATH)
    raw = read_jsonl(RAW_PATH)
    summary = read_json(SUMMARY_PATH)
    material_by_id = {row["item_id"]: row for row in material}
    manifest_by_id = {row["item_id"]: row for row in manifest}
    row_checks: list[bool] = []
    teacher_checks: list[bool] = []
    for row in raw:
        item = material_by_id[row["item_id"]]
        man = manifest_by_id[row["item_id"]]
        candidates = item["candidates"]
        choice_map = {candidate: float(row["choice_candidate_scores"][candidate]["sum_log_probability"]) for candidate in candidates}
        bare_map = {candidate: float(row["bare_candidate_scores"][candidate]["sum_log_probability"]) for candidate in candidates}
        choice_set = argmax_set(choice_map)
        bare_set = argmax_set(bare_map)
        choice_prediction = choice_set[0] if len(choice_set) == 1 else None
        bare_prediction = bare_set[0] if len(bare_set) == 1 else None
        trie_tokens = row["trie_generation"]["generated_token_ids"]
        trie_matches = [candidate for candidate, tokens in man["bare_candidate_token_ids"].items() if tokens == trie_tokens]
        trie_prediction = trie_matches[0] if len(trie_matches) == 1 else None
        parses = {
            "bare": parse_output(row["bare_generation"], candidates, item["gold"], item["query_object"], item["gold"]),
            "cued": parse_output(row["cued_generation"], candidates, item["gold"], item["query_object"], item["gold"]),
            "sentence": parse_output(row["sentence_generation"], candidates, item["gold"], item["query_object"], item["expected_sentence"]),
            "natural": parse_output(row["natural_generation"], candidates, item["gold"], item["query_object"], item["gold"]),
        }
        slot = lambda prediction: candidates.index(prediction) if prediction in candidates else None
        row_checks.append(
            row["choice_argmax_set"] == choice_set
            and row["bare_argmax_set"] == bare_set
            and row["choice_prediction"] == choice_prediction
            and row["bare_candidate_prediction"] == bare_prediction
            and row["choice_correct"] is (choice_prediction == item["gold"])
            and row["bare_candidate_correct"] is (bare_prediction == item["gold"])
            and row["choice_prediction_slot"] == slot(choice_prediction)
            and row["bare_candidate_prediction_slot"] == slot(bare_prediction)
            and row["choice_finite"] is all(bool(row["choice_candidate_scores"][candidate]["all_vocab_logits_finite"]) for candidate in candidates)
            and row["bare_candidate_finite"] is all(bool(row["bare_candidate_scores"][candidate]["all_vocab_logits_finite"]) for candidate in candidates)
            and row["trie_generation"]["prediction"] == trie_prediction
            and row["trie_correct"] is (trie_prediction == item["gold"])
            and row["trie_prediction_slot"] == slot(trie_prediction)
            and all(row[f"{readout}_parse"] == parses[readout] for readout in parses)
            and all(row[f"{readout}_prediction_slot"] == slot(parses[readout]["prediction"]) for readout in parses)
            and all(row[f"{readout}_content_correct"] is parses[readout]["content_correct"] for readout in parses)
            and row["bare_exact"] is parses["bare"]["exact"]
            and row["cued_exact"] is parses["cued"]["exact"]
            and row["sentence_exact"] is parses["sentence"]["exact"]
            and row["natural_exact"] is parses["natural"]["exact"]
        )
        for readout in ("bare", "cued", "sentence"):
            diag = row[f"{readout}_teacher"]
            ranks = diag["gold_token_ranks"]
            margins = diag["gold_token_margins"]
            first = next((index for index, rank in enumerate(ranks) if rank != 1), None)
            expected_tokens = man[f"{readout}_gold_token_ids"]
            teacher_checks.append(
                diag["gold_token_ids"] == expected_tokens
                and len(ranks) == len(margins) == len(diag["gold_token_log_probabilities"]) == len(expected_tokens)
                and diag["all_gold_tokens_top1"] is all(rank == 1 for rank in ranks)
                and diag["first_not_top1_index"] == first
                and math.isclose(diag["minimum_gold_margin"], min(margins), rel_tol=0.0, abs_tol=1e-12)
                and math.isclose(diag["mean_gold_rank"], sum(ranks) / len(ranks), rel_tol=0.0, abs_tol=1e-12)
                and all(rank >= 1 for rank in ranks)
            )
    ledgers = independent_ledgers(raw, program)
    checks = {
        "preaudit_pass": pre.get("all_checks_passed") is True,
        "source_hashes_unchanged": contract["source_hashes"] == source_hashes(),
        "raw_cardinality": len(raw) == EXPECTED_ROWS,
        "raw_identity": {row["item_id"] for row in raw} == set(material_by_id) == set(manifest_by_id),
        "raw_order": [row["execution_index"] for row in raw] == list(range(EXPECTED_ROWS)),
        "raw_self_digests": all(row["behavior_row_digest"] == digest(strip_digest(row, "behavior_row_digest")) for row in raw),
        "manifest_links": all(row["manifest_row_digest"] == manifest_by_id[row["item_id"]]["manifest_row_digest"] for row in raw),
        "contract_links": all(row["contract_digest"] == contract["contract_digest"] for row in raw),
        "all_readout_decisions_recomputed": all(row_checks),
        "teacher_diagnostics_consistent": all(teacher_checks),
        "summary_self_digest": summary["summary_digest"] == digest(strip_digest(summary, "summary_digest")),
        "summary_raw_digest": summary["raw_digest"] == digest(raw),
        "summary_case_count": summary["case_count"] == EXPECTED_ROWS,
        "summary_contract": summary["contract_digest"] == contract["contract_digest"],
        "summary_plan": summary["batch_plan_digest"] == plan["plan_digest"],
        "cuda_fp16": summary["placement"]["placement"] == "full_cuda" and set(summary["precision_audit"]["parameter_dtypes"]) == {"float16"} and summary["precision_audit"]["has_quantized_modules"] is False,
        "no_hidden_or_intervention": summary["hidden_states_saved"] is False and summary["attentions_saved"] is False and summary["interventions_performed"] is False,
        "all_typed_cells_present": set(ledgers["axis_partition_cells"]) == {f"{axis}|{partition}" for axis in AXES for partition in PARTITIONS},
        "program_construct_retained": ledgers["typed_global_gates"]["program_construct"] is True,
    }
    value = audit_value("result", checks, {"contract_digest": contract["contract_digest"], "raw_digest": digest(raw), "run_summary_digest": summary["summary_digest"], "recomputed_ledgers": ledgers})
    write_json(RESULT_AUDIT_PATH, value)
    print(canonical_json({"status": "phase1235_result_audit", "passed": value["all_checks_passed"], "checks": f'{value["passed_check_count"]}/{value["total_check_count"]}', "failed": value["failed_checks"], "audit_digest": value["audit_digest"], "future_response_eligibility": ledgers["future_response_eligibility"]}))
    if not value["all_checks_passed"]:
        raise RuntimeError(f"result audit failed: {value['failed_checks']}")


def final_audit() -> None:
    if FINAL_AUDIT_PATH.exists():
        raise RuntimeError("Phase1235 final audit exists")
    pre = read_json(PREAUDIT_PATH)
    result = read_json(RESULT_AUDIT_PATH)
    final = read_json(FINAL_PATH)
    ledgers = result["recomputed_ledgers"]
    passed = bool(ledgers["future_response_eligibility"])
    checks = {
        "preaudit_pass": pre.get("all_checks_passed") is True,
        "result_audit_pass": result.get("all_checks_passed") is True,
        "final_self_digest": final["final_digest"] == digest(strip_digest(final, "final_digest")),
        "ledgers_recomputed": final["ledgers"] == ledgers,
        "result_audit_link": final["result_audit_digest"] == result["audit_digest"],
        "status_typed": final["status"] == ("cross_readout_object_qualified" if passed else "typed_generation_boundary_localized"),
        "k210": final["k_item"]["identifier"] == "K210" and final["k_item"]["evidence_grade"] == ("E3-BEHAVIOR-CROSS-READOUT" if passed else "E3-TYPED-BOUNDARY"),
        "candidate_permission": final["authorization"]["candidate_selection_claim"] is ledgers["typed_global_gates"]["candidate_selection"],
        "short_permission": final["authorization"]["short_string_claim"] is ledgers["typed_global_gates"]["short_string"],
        "sentence_permission": final["authorization"]["sentence_claim"] is ledgers["typed_global_gates"]["sentence"],
        "natural_permission": final["authorization"]["natural_sentence_claim"] is ledgers["typed_global_gates"]["natural"],
        "future_permission": final["authorization"]["future_response_phase"] is passed,
        "auto_continue": final["authorization"]["auto_continue"] is passed,
        "next_experiment": (final["authorization"]["next_experiment"] is not None) is passed,
        "no_hidden_in_phase": final["authorization"]["hidden_scan_in_this_phase"] is False,
        "no_cross_model": final["authorization"]["cross_model_run"] is False,
        "no_module_claim": final["authorization"]["separate_neural_module_claim"] is False,
        "no_new_math": final["new_mathematics_required"] is False,
    }
    value = audit_value("final", checks, {"final_digest": final["final_digest"], "result_audit_digest": result["audit_digest"], "future_response_eligibility": passed, "authorized_next": final["authorization"]["next_experiment"]})
    write_json(FINAL_AUDIT_PATH, value)
    print(canonical_json({"status": "phase1235_final_audit", "passed": value["all_checks_passed"], "checks": f'{value["passed_check_count"]}/{value["total_check_count"]}', "failed": value["failed_checks"], "audit_digest": value["audit_digest"], "auto_continue": passed}))
    if not value["all_checks_passed"]:
        raise RuntimeError(f"final audit failed: {value['failed_checks']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=("preaudit", "result", "final"))
    stage = parser.parse_args().stage
    {"preaudit": preaudit, "result": result_audit, "final": final_audit}[stage]()


if __name__ == "__main__":
    main()
