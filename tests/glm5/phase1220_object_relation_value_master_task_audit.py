#!/usr/bin/env python3
"""Independent pre- and post-execution audit for Phase 1220."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests/glm5"
sys.path.insert(0, str(TEST_ROOT))

from model_utils import MODEL_CONFIGS


PHASE = 1220
MAIN_SCRIPT = TEST_ROOT / "phase1220_object_relation_value_master_task.py"
AUDIT_SCRIPT = Path(__file__).resolve()
OUT_ROOT = TEST_ROOT / "result/phase1220_object_relation_value_master_task"
PROTOCOL_PATH = OUT_ROOT / "protocol/preregistration.json"
MATERIAL_PATH = OUT_ROOT / "material/master_task.jsonl"
MANIFEST_PATH = OUT_ROOT / "protocol/qwen3_manifest.jsonl"
TOKEN_AUDIT_PATH = OUT_ROOT / "audit/tokenizer_audit.json"
PREAUDIT_PATH = OUT_ROOT / "audit/independent_preaudit.json"
RAW_PATH = OUT_ROOT / "behavior/qwen3/raw_behavior.jsonl"
RUN_SUMMARY_PATH = OUT_ROOT / "behavior/qwen3/run_summary.json"
FINAL_PATH = OUT_ROOT / "analysis/final.json"
RESULT_AUDIT_PATH = OUT_ROOT / "audit/independent_result_audit.json"

SPLITS = ("discovery", "confirmation", "composition", "sealed")
TRACKS = ("natural", "symbolic")
PANELS = ("canonical", "record_order", "paraphrase", "same_answer_carrier")
EXPECTED_WORLDS = 512
EXPECTED_ROWS = 12288
EXPECTED_PER_SPLIT = 3072
EXPECTED_GENERATION_PER_SPLIT = 768


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
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def add(checks: dict[str, bool], name: str, value: Any) -> None:
    checks[name] = bool(value)


def validate_embedded(value: dict[str, Any], key: str) -> bool:
    expected = value.get(key)
    candidate = {name: item for name, item in value.items() if name != key}
    return isinstance(expected, str) and digest(candidate) == expected


def recompute_gold(row: dict[str, Any]) -> str:
    assignments = row["rendered_assignments"]
    entities = row["entities"]
    links = row["links"]
    derivation = row["derivation"]
    operation = row["operation"]
    relation = row["target_relation"]

    if operation in {"direct", "query_object", "query_relation", "binding_swap", "object_relation_compose", "binding_query_compose"}:
        if operation in {"direct", "query_relation", "binding_swap"}:
            entity = entities[0]
        elif operation in {"query_object", "object_relation_compose"}:
            entity = entities[1]
        else:
            entity = entities[2]
        return assignments[entity][relation]
    if operation == "inverse_lookup":
        target = derivation[1]
        matches = [entity for entity in entities if assignments[entity][relation] == target]
        if len(matches) != 1:
            raise RuntimeError("inverse audit is not unique")
        return matches[0]
    if operation == "link_then_value":
        return assignments[links[entities[0]]][relation]
    if operation == "link_relation_compose":
        return assignments[links[entities[1]]][relation]
    if operation == "double_link_relation":
        return assignments[links[links[entities[0]]]][relation]
    if operation == "inverse_link_compose":
        source_relation = row["relations"][0]
        target = derivation[1]
        found = [entity for entity in entities if assignments[entity][source_relation] == target]
        if len(found) != 1:
            raise RuntimeError("inverse-link audit is not unique")
        return assignments[links[found[0]]][relation]
    if operation == "link_binding_compose":
        return assignments[links[entities[0]]][relation]
    raise KeyError(operation)


def render_native(tokenizer: Any, system_prompt: str, prompt: str) -> str:
    return str(
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    )


def preaudit() -> dict[str, Any]:
    protocol = read_json(PROTOCOL_PATH)
    rows = read_jsonl(MATERIAL_PATH)
    manifest = read_jsonl(MANIFEST_PATH)
    token_audit = read_json(TOKEN_AUDIT_PATH)
    checks: dict[str, bool] = {}

    add(checks, "protocol_embedded_digest", validate_embedded(protocol, "protocol_digest"))
    add(checks, "main_hash_frozen", protocol["source_hashes"]["main"] == file_sha256(MAIN_SCRIPT))
    add(checks, "audit_hash_frozen", protocol["source_hashes"]["audit"] == file_sha256(AUDIT_SCRIPT))
    add(checks, "phase", protocol.get("phase") == PHASE)
    add(checks, "row_count", len(rows) == EXPECTED_ROWS)
    add(checks, "manifest_count", len(manifest) == EXPECTED_ROWS)
    add(checks, "world_count", len({row["world_id"] for row in rows}) == EXPECTED_WORLDS)
    add(checks, "unique_item_ids", len({row["item_id"] for row in rows}) == EXPECTED_ROWS)
    add(checks, "material_digest", protocol["material"]["material_digest"] == digest(rows))
    add(checks, "manifest_digest", protocol["material"]["manifest_digest"] == digest(manifest))
    add(checks, "token_audit_embedded", validate_embedded(token_audit, "tokenizer_audit_digest"))
    add(checks, "token_audit_match", protocol["material"]["tokenizer_audit_digest"] == token_audit["tokenizer_audit_digest"])
    add(checks, "multi_token_present", token_audit["multi_token_candidate_fraction"] > 0.25)
    add(checks, "no_empty_candidates", token_audit["empty_candidate_count"] == 0)

    material_by_id = {row["item_id"]: row for row in rows}
    manifest_by_id = {row["item_id"]: row for row in manifest}
    add(checks, "manifest_items_exact", set(material_by_id) == set(manifest_by_id))

    for split in SPLITS:
        selected = [row for row in rows if row["split"] == split]
        add(checks, f"{split}_count", len(selected) == EXPECTED_PER_SPLIT)
        add(checks, f"{split}_worlds", len({row["world_id"] for row in selected}) == 128)
        add(checks, f"{split}_generation", sum(row["generation_required"] for row in selected) == EXPECTED_GENERATION_PER_SPLIT)
        for track in TRACKS:
            add(checks, f"{split}_{track}_balance", sum(row["track"] == track for row in selected) == EXPECTED_PER_SPLIT // 2)
        for panel in PANELS:
            add(checks, f"{split}_{panel}_balance", sum(row["panel"] == panel for row in selected) == EXPECTED_PER_SPLIT // 4)

    world_splits: dict[str, set[str]] = defaultdict(set)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    truth_ok = True
    row_digest_ok = True
    carrier_changed = True
    for row in rows:
        world_splits[row["world_id"]].add(row["split"])
        groups[row["group_id"]].append(row)
        row_digest_ok = row_digest_ok and validate_embedded(row, "row_digest")
        truth_ok = truth_ok and recompute_gold(row) == row["gold"] and row["gold"] in row["candidates"]
        if row["panel"] == "same_answer_carrier":
            carrier_changed = carrier_changed and row["rendered_assignments"] != row["base_assignments"] and row["carrier_note"] is not None
    add(checks, "split_disjoint_worlds", all(len(values) == 1 for values in world_splits.values()))
    add(checks, "row_embedded_digests", row_digest_ok)
    add(checks, "truth_recomputed", truth_ok)
    add(checks, "carrier_is_real_control", carrier_changed)
    add(
        checks,
        "surface_groups_complete",
        len(groups) == EXPECTED_WORLDS * 6
        and all(
            len(values) == len(PANELS)
            and {row["panel"] for row in values} == set(PANELS)
            and len({row["gold"] for row in values}) == 1
            for values in groups.values()
        ),
    )

    # Independent exact tokenizer replay.
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_CONFIGS["qwen3"]["path"],
        trust_remote_code=True,
        local_files_only=True,
        use_fast=False,
    )
    tokenizer_ok = True
    manifest_digest_ok = True
    for index, manifest_row in enumerate(manifest):
        material = material_by_id[manifest_row["item_id"]]
        rendered = render_native(tokenizer, protocol["interface"]["system_prompt"], material["prompt"])
        base = [int(value) for value in tokenizer.encode(rendered, add_special_tokens=False)]
        tokenizer_ok = tokenizer_ok and base == manifest_row["input_ids"]
        for candidate in material["candidates"]:
            extended = [int(value) for value in tokenizer.encode(rendered + " " + candidate, add_special_tokens=False)]
            tokenizer_ok = tokenizer_ok and extended[: len(base)] == base
            tokenizer_ok = tokenizer_ok and extended[len(base) :] == manifest_row["candidate_token_ids"][candidate]
        manifest_digest_ok = manifest_digest_ok and validate_embedded(manifest_row, "manifest_row_digest")
        if not tokenizer_ok:
            break
        if (index + 1) % 3072 == 0:
            print(f"[phase1220/preaudit-tokenizer] {index + 1}/{len(manifest)}", flush=True)
    add(checks, "tokenizer_exact_replay", tokenizer_ok)
    add(checks, "manifest_row_digests", manifest_digest_ok)

    # A positive sentinel must be detected by the independent truth function.
    sentinel = dict(rows[0])
    sentinel["gold"] = next(value for value in sentinel["candidates"] if value != sentinel["gold"])
    add(checks, "truth_leak_positive_sentinel", recompute_gold(sentinel) != sentinel["gold"])
    add(checks, "behavior_output_absent_before_preaudit", not RAW_PATH.exists() and not FINAL_PATH.exists())
    add(checks, "hidden_access_forbidden", not protocol["interface"]["hidden_states"] and not protocol["interface"]["attentions"] and not protocol["interface"]["hooks"])

    result = {
        "phase": PHASE,
        "mode": "preaudit",
        "created_at": utc_now(),
        "check_count": len(checks),
        "passed_count": sum(checks.values()),
        "all_checks_passed": all(checks.values()),
        "checks": checks,
        "protocol_digest": protocol["protocol_digest"],
    }
    result["audit_digest"] = digest(result)
    return result


def rate(rows: list[dict[str, Any]], field: str) -> float:
    return sum(bool(row[field]) for row in rows) / len(rows)


def grouped_rates(rows: list[dict[str, Any]], fields: tuple[str, ...], metric: str) -> dict[str, float]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(str(row[field]) for field in fields)].append(row)
    return {"|".join(key): rate(values, metric) for key, values in sorted(groups.items())}


def independent_summary(raw: list[dict[str, Any]], protocol: dict[str, Any]) -> dict[str, Any]:
    ledgers = protocol["behavior_ledgers"]
    result: dict[str, Any] = {"splits": {}, "all_split_gates": {}}
    for split in SPLITS:
        selected = [row for row in raw if row["split"] == split]
        generated = [row for row in selected if row["generation_required"]]
        candidate_cells = grouped_rates(selected, ("track", "operation"), "candidate_correct")
        generation_cells = grouped_rates(generated, ("track", "operation"), "generation_correct")
        surface: dict[str, list[dict[str, Any]]] = defaultdict(list)
        semantic: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in selected:
            surface[row["group_id"]].append(row)
            if row["panel"] == "canonical":
                semantic[row["world_id"]].append(row)
        surface_rate = sum(
            len(values) == 4
            and all(row["candidate_correct"] for row in values)
            and len({row["candidate_prediction"] for row in values}) == 1
            for values in surface.values()
        ) / len(surface)
        semantic_rate = sum(
            len(values) == 6 and all(row["candidate_correct"] for row in values)
            for values in semantic.values()
        ) / len(semantic)
        metrics = {
            "case_count": len(selected),
            "generation_case_count": len(generated),
            "finite_rate": rate(selected, "all_candidate_scores_finite"),
            "candidate_accuracy": rate(selected, "candidate_correct"),
            "candidate_worst_track_operation_cell": min(candidate_cells.values()),
            "candidate_cells": candidate_cells,
            "surface_group_rate": surface_rate,
            "semantic_set_rate": semantic_rate,
            "generation_semantic_accuracy": rate(generated, "generation_correct"),
            "generation_normalized_exact_rate": rate(generated, "generation_normalized_exact"),
            "generation_worst_track_operation_cell": min(generation_cells.values()),
            "generation_cells": generation_cells,
            "candidate_by_panel": grouped_rates(selected, ("panel",), "candidate_correct"),
            "candidate_by_track": grouped_rates(selected, ("track",), "candidate_correct"),
            "generation_by_track": grouped_rates(generated, ("track",), "generation_correct"),
            "mean_gold_margin": sum(float(row["gold_margin"]) for row in selected) / len(selected),
        }
        gates = {
            "finite": metrics["finite_rate"] >= ledgers["L1_numerical"]["finite_rate_min"],
            "candidate_overall": metrics["candidate_accuracy"] >= ledgers["L2_candidate"]["overall_each_split_min"],
            "candidate_worst_cell": metrics["candidate_worst_track_operation_cell"] >= ledgers["L2_candidate"]["worst_track_operation_cell_each_split_min"],
            "surface": metrics["surface_group_rate"] >= ledgers["L3_surface"]["all_four_panels_correct_group_rate_each_split_min"],
            "semantic_set": metrics["semantic_set_rate"] >= ledgers["L4_semantic_set"]["all_six_canonical_operations_correct_world_rate_each_split_min"],
            "generation_overall": metrics["generation_semantic_accuracy"] >= ledgers["L5_generation"]["semantic_each_split_min"],
            "generation_worst_cell": metrics["generation_worst_track_operation_cell"] >= ledgers["L5_generation"]["worst_track_operation_cell_each_split_min"],
            "generation_exact": metrics["generation_normalized_exact_rate"] >= ledgers["L5_generation"]["normalized_exact_each_split_min"],
        }
        if split in {"composition", "sealed"}:
            gates["heldout_candidate"] = metrics["candidate_accuracy"] >= ledgers["L6_heldout"]["composition_and_sealed_candidate_min"]
            gates["heldout_generation"] = metrics["generation_semantic_accuracy"] >= ledgers["L6_heldout"]["composition_and_sealed_generation_min"]
        passed = all(gates.values())
        result["splits"][split] = {"metrics": metrics, "gates": gates, "passed": passed}
        result["all_split_gates"][f"split_{split}"] = passed
    result["passed"] = all(result["all_split_gates"].values())
    return result


def result_audit() -> dict[str, Any]:
    protocol = read_json(PROTOCOL_PATH)
    rows = read_jsonl(MATERIAL_PATH)
    manifest = read_jsonl(MANIFEST_PATH)
    raw = read_jsonl(RAW_PATH)
    run_summary = read_json(RUN_SUMMARY_PATH)
    final = read_json(FINAL_PATH)
    checks: dict[str, bool] = {}
    add(checks, "protocol_still_valid", validate_embedded(protocol, "protocol_digest"))
    add(checks, "main_hash_unchanged", protocol["source_hashes"]["main"] == file_sha256(MAIN_SCRIPT))
    add(checks, "audit_hash_unchanged", protocol["source_hashes"]["audit"] == file_sha256(AUDIT_SCRIPT))
    add(checks, "raw_count", len(raw) == EXPECTED_ROWS)
    add(checks, "unique_raw_items", len({row["item_id"] for row in raw}) == EXPECTED_ROWS)
    add(checks, "run_summary_embedded", validate_embedded(run_summary, "summary_digest"))
    add(checks, "raw_digest", run_summary["raw_digest"] == digest(raw))
    add(checks, "final_embedded", validate_embedded(final, "final_digest"))
    add(checks, "no_quantization", not run_summary["precision_audit"]["has_quantized_modules"])
    add(checks, "fp16_parameters", run_summary["precision_audit"]["has_fp16_parameters"])
    add(checks, "no_pending", not any(OUT_ROOT.rglob("*.pending")))

    material_by_id = {row["item_id"]: row for row in rows}
    manifest_by_id = {row["item_id"]: row for row in manifest}
    row_ok = True
    digest_ok = True
    score_ok = True
    generation_scope_ok = True
    for row in raw:
        material = material_by_id[row["item_id"]]
        manifest_row = manifest_by_id[row["item_id"]]
        digest_ok = digest_ok and validate_embedded(row, "behavior_row_digest")
        ordered = sorted(
            row["candidate_scores"],
            key=lambda name: row["candidate_scores"][name]["mean_log_probability"],
            reverse=True,
        )
        tie = abs(
            row["candidate_scores"][ordered[0]]["mean_log_probability"]
            - row["candidate_scores"][ordered[1]]["mean_log_probability"]
        ) <= 1e-7
        prediction = None if tie else ordered[0]
        margin = row["candidate_scores"][material["gold"]]["mean_log_probability"] - max(
            value["mean_log_probability"]
            for name, value in row["candidate_scores"].items()
            if name != material["gold"]
        )
        score_ok = score_ok and prediction == row["candidate_prediction"]
        score_ok = score_ok and bool(prediction == material["gold"]) == row["candidate_correct"]
        score_ok = score_ok and abs(margin - row["gold_margin"]) < 1e-9
        score_ok = score_ok and set(row["candidate_scores"]) == set(manifest_row["candidates"])
        generation_scope_ok = generation_scope_ok and (
            (material["generation_required"] and row["generation_correct"] is not None)
            or (not material["generation_required"] and row["generation_correct"] is None)
        )
        row_ok = row_ok and row["row_digest"] == material["row_digest"]
    add(checks, "raw_embedded_digests", digest_ok)
    add(checks, "raw_material_binding", row_ok)
    add(checks, "candidate_scores_recomputed", score_ok)
    add(checks, "generation_scope", generation_scope_ok)

    independent = independent_summary(raw, protocol)
    add(checks, "summary_exact_recompute", independent == final["behavior"])
    add(checks, "status_recompute", final["status"] == ("qwen3_master_task_behavior_qualified" if independent["passed"] else "qwen3_master_task_behavior_gate_failed"))
    add(checks, "authorization_recompute", bool(final["authorized_next"]["automatic_execution"]) == bool(independent["passed"]))
    add(checks, "claim_boundary", final["evidence_scope"] == protocol["claim_boundary"])
    add(checks, "no_hidden_claim", not final["evidence_scope"]["hidden_state"] and not final["evidence_scope"]["causal"])

    result = {
        "phase": PHASE,
        "mode": "result",
        "created_at": utc_now(),
        "check_count": len(checks),
        "passed_count": sum(checks.values()),
        "all_checks_passed": all(checks.values()),
        "checks": checks,
        "recomputed_behavior_digest": digest(independent),
        "final_digest": final["final_digest"],
    }
    result["audit_digest"] = digest(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=("preaudit", "result"))
    args = parser.parse_args()
    result = preaudit() if args.mode == "preaudit" else result_audit()
    path = PREAUDIT_PATH if args.mode == "preaudit" else RESULT_AUDIT_PATH
    if path.exists():
        raise RuntimeError(f"audit output already exists: {path}")
    write_json(path, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
