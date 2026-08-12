#!/usr/bin/env python3
"""Independent three-stage audit for Phase1234.

This module deliberately does not import the Phase1234 implementation.  It
recomputes registry selection, the registered nuisance-program ceiling,
material collisions, raw score decisions, behavior ledgers, and final claim
permissions from frozen artifacts.
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
MAIN_SCRIPT = TEST_ROOT / "phase1234_qwen3_k199_registry_sealed_confirmation.py"
AUDIT_SCRIPT = Path(__file__).resolve()
P1220_SCRIPT = TEST_ROOT / "phase1220_object_relation_value_master_task.py"
P1221_SCRIPT = TEST_ROOT / "phase1221_typed_operation_behavior_and_error_fingerprints.py"

OUT_ROOT = TEST_ROOT / "result/phase1234_qwen3_k199_registry_sealed_confirmation"
CONTRACT_PATH = OUT_ROOT / "protocol/preregistration.json"
MATERIAL_PATH = OUT_ROOT / "material/sealed_query_object_worlds.jsonl"
MANIFEST_PATH = OUT_ROOT / "protocol/qwen3_manifest.jsonl"
PROGRAM_PATH = OUT_ROOT / "protocol/depth2_program_grammar_audit.json"
PLAN_PATH = OUT_ROOT / "protocol/frozen_batch_plan.json"
PREAUDIT_PATH = OUT_ROOT / "audit/independent_preaudit.json"
RAW_PATH = OUT_ROOT / "behavior/qwen3/raw_behavior.jsonl"
SUMMARY_PATH = OUT_ROOT / "behavior/qwen3/run_summary.json"
RESULT_AUDIT_PATH = OUT_ROOT / "audit/independent_result_audit.json"
FINAL_PATH = OUT_ROOT / "analysis/final.json"
FINAL_AUDIT_PATH = OUT_ROOT / "audit/independent_final_audit.json"

K199_FINAL_PATH = TEST_ROOT / "result/phase1222_atomic_operation_independent_confirmation/analysis/final.json"
K199_FP16_PATH = TEST_ROOT / "result/phase1222_atomic_operation_independent_confirmation/audit/fp16_schema_resolution.json"
P1233_FINAL_PATH = TEST_ROOT / "result/phase1233_qwen3_program_identifiable_medal_binding/analysis/final.json"
P1233_AUDIT_PATH = TEST_ROOT / "result/phase1233_qwen3_program_identifiable_medal_binding/audit/independent_final_audit.json"

EXPECTED_K199_FINAL = "a6be67cce38afa78aef432c8d01b1c8007cd40039dc4cc66c190a360753a65e2"
EXPECTED_K199_FP16 = "0599a6d47b67add164bbbc3937fefa4e57f46919b1d98767c24f06ba498ce4a0"
EXPECTED_P1233_FINAL = "692faa033e949733dbd1900f89d26fdc140f386e8891309a3746b502437321c2"
EXPECTED_P1233_AUDIT = "2e61a16d092dade8a65c1aa9cd414f643f96ad2030b8073d0648794f4735354b"

SPLITS = ("sealed_alpha", "sealed_beta", "sealed_gamma")
PANELS = ("canonical", "record_order", "paraphrase", "binding_rotation")
EXPECTED_ROWS = 2304
ROWS_PER_SPLIT = 768
WORLDS_PER_SPLIT = 48
ROWS_PER_WORLD = 16
PROGRAM_DEPTH = 2
PROGRAM_CEILING = 0.80
TIE_TOLERANCE = 1e-7
THRESHOLDS = {
    "Q0_finite_rate": 1.0,
    "Q1_candidate_accuracy": 0.95,
    "Q1_context_adjusted_accuracy": 0.90,
    "Q1_open_generation_accuracy": 0.90,
    "Q2_query_quartet_success": 0.80,
    "Q3_binding_rotation_pair_success": 0.85,
    "Q4_order_pair_success": 0.90,
    "Q4_paraphrase_pair_success": 0.90,
    "Q5_worst_panel_candidate": 0.90,
    "Q5_reliable_world_rate": 0.80,
    "Q6_sum_mean_argmax_agreement": 1.0,
    "Q7_program_ceiling": PROGRAM_CEILING,
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


def independent_registry_selection(k199: dict[str, Any]) -> dict[str, Any]:
    scopes = sorted(k199["behavior"]["authorized_target_operation_tracks"])
    expected = {
        "direct|natural",
        "inverse_lookup|symbolic",
        "query_object|natural",
        "query_relation|natural",
        "query_relation|symbolic",
    }
    if set(scopes) != expected:
        raise RuntimeError("K199 scope registry changed")
    registry: dict[str, Any] = {}
    for scope in scopes:
        split_cells = k199["behavior"]["operation_track_results"][scope]["splits"]
        minima = {
            metric: min(float(cell["metrics"][metric]) for cell in split_cells.values())
            for metric in (
                "candidate_accuracy",
                "open_generation_accuracy",
                "context_adjusted_accuracy",
                "worst_panel_candidate",
                "all_panel_world_rate",
            )
        }
        registry[scope] = {"minimum_metrics": minima, "selection_score": min(minima.values())}
    ranking = sorted(registry, key=lambda scope: (-registry[scope]["selection_score"], scope))
    value = {
        "selection_source": "K199 Phase1222 frozen behavior registry; historical discovery evidence only",
        "selection_rule": "maximize the minimum across candidate, open-generation, context-adjusted, worst-panel, and all-panel-world metrics; lexical scope tie-break",
        "registry": registry,
        "ranking": ranking,
        "selected_scope": ranking[0],
        "selection_is_new_evidence": False,
    }
    value["selection_digest"] = digest(value)
    return value


def base_program_predictions(row: dict[str, Any]) -> dict[str, str]:
    candidates = row["candidates"]
    displayed = row["display_assignments"]
    objects = row["objects"]
    query_index = int(row["query_index"])
    programs = {
        f"candidate_position_{index}": row["candidate_order"][index]
        for index in range(len(candidates))
    }
    programs.update({f"fixed_object_{index}": displayed[objects[index]] for index in range(len(objects))})
    programs.update(
        {
            "first_record": displayed[objects[row["record_order_indices"][0]]],
            "last_record": displayed[objects[row["record_order_indices"][-1]]],
            "next_object": displayed[objects[(query_index + 1) % len(objects)]],
            "opposite_object": displayed[objects[(query_index + 2) % len(objects)]],
            "previous_object": displayed[objects[(query_index + 3) % len(objects)]],
            "unused_value": row["unused_value"],
            "lexical_first": sorted(candidates)[0],
            "lexical_last": sorted(candidates)[-1],
        }
    )
    return programs


def nuisance_features(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "panel": row["panel"],
        "record_first_index": row["record_order_indices"][0],
        "unused_candidate_position": row["candidate_order"].index(row["unused_value"]),
    }


def independent_depth_tree(rows: list[dict[str, Any]], depth: int) -> tuple[float, dict[str, Any]]:
    programs = sorted(base_program_predictions(rows[0]))
    predictions = [base_program_predictions(row) for row in rows]
    features = [nuisance_features(row) for row in rows]
    conditions = sorted(
        {(name, canonical_json(value)) for feature in features for name, value in feature.items()}
    )
    decoded = {(name, encoded): json.loads(encoded) for name, encoded in conditions}
    cache: dict[tuple[tuple[int, ...], int], tuple[int, dict[str, Any]]] = {}

    def recurse(indices: tuple[int, ...], remaining: int) -> tuple[int, dict[str, Any]]:
        cache_key = (indices, remaining)
        if cache_key in cache:
            return cache[cache_key]
        scores = {
            program: sum(predictions[index][program] == rows[index]["gold"] for index in indices)
            for program in programs
        }
        maximum = max(scores.values())
        leaf = min(program for program in programs if scores[program] == maximum)
        best_score = scores[leaf]
        best_tree: dict[str, Any] = {"leaf_program": leaf, "correct": best_score, "n": len(indices)}
        if remaining:
            for name, encoded in conditions:
                expected = decoded[(name, encoded)]
                left = tuple(index for index in indices if features[index][name] == expected)
                right = tuple(index for index in indices if features[index][name] != expected)
                if not left or not right:
                    continue
                left_score, left_tree = recurse(left, remaining - 1)
                right_score, right_tree = recurse(right, remaining - 1)
                score = left_score + right_score
                tree = {
                    "condition": {"feature": name, "equals": expected},
                    "true": left_tree,
                    "false": right_tree,
                    "correct": score,
                    "n": len(indices),
                }
                if score > best_score or (score == best_score and canonical_json(tree) < canonical_json(best_tree)):
                    best_score, best_tree = score, tree
        cache[cache_key] = (best_score, best_tree)
        return cache[cache_key]

    correct, tree = recurse(tuple(range(len(rows))), depth)
    return correct / len(rows), tree


def independent_program_audit(material: list[dict[str, Any]], manifest: list[dict[str, Any]]) -> dict[str, Any]:
    manifest_by_id = {row["item_id"]: row for row in manifest}
    split_results: dict[str, Any] = {}
    for split in SPLITS:
        selected = [row for row in material if row["split"] == split]
        base = {
            name: sum(base_program_predictions(row)[name] == row["gold"] for row in selected) / len(selected)
            for name in sorted(base_program_predictions(selected[0]))
        }
        ceiling, witness = independent_depth_tree(selected, PROGRAM_DEPTH)
        split_results[split] = {
            "base_program_accuracies": base,
            "maximum_base_program_accuracy": max(base.values()),
            "depth2_conditional_program_accuracy": ceiling,
            "depth2_witness_tree": witness,
            "construct_gate": ceiling <= PROGRAM_CEILING,
        }
    query = grouped(material, "query_group_id")
    order = grouped(material, "order_pair_id")
    paraphrase = grouped(material, "paraphrase_pair_id")
    binding = grouped(material, "binding_pair_id")
    collisions = {
        "query_quartets_complete": all(
            len(cell) == 4 and len({row["gold"] for row in cell}) == 4 for cell in query.values()
        ),
        "order_pairs_invariant": all(
            len(cell) == 2 and len({row["gold"] for row in cell}) == 1 for cell in order.values()
        ),
        "paraphrase_pairs_invariant": all(
            len(cell) == 2 and len({row["gold"] for row in cell}) == 1 for cell in paraphrase.values()
        ),
        "binding_pairs_discriminating": all(
            len(cell) == 2
            and len({row["gold"] for row in cell}) == 2
            and len({digest(lexical_multiset(row["candidate_prompt"])) for row in cell}) == 1
            and len({digest(sorted(manifest_by_id[row["item_id"]]["world_input_ids"])) for row in cell}) == 1
            for cell in binding.values()
        ),
        "unused_value_never_gold": all(row["unused_value"] != row["gold"] for row in material),
        "five_candidates_four_assigned": all(
            len(row["candidates"]) == 5 and len(set(row["base_assignments"].values())) == 4
            for row in material
        ),
    }
    target_accuracy = sum(
        row["display_assignments"][row["query_object"]] == row["gold"] for row in material
    ) / len(material)
    return {
        "split_results": split_results,
        "collision_group_counts": {
            "query_quartets": len(query),
            "order_pairs": len(order),
            "paraphrase_pairs": len(paraphrase),
            "binding_pairs": len(binding),
        },
        "collision_checks": collisions,
        "target_equivalent_accuracy": target_accuracy,
        "program_construct_gate": (
            all(cell["construct_gate"] for cell in split_results.values())
            and all(collisions.values())
            and target_accuracy == 1.0
        ),
    }


def audit_value(kind: str, checks: dict[str, bool], extra: dict[str, Any] | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {
        "phase": 1234,
        "schema_version": f"phase1234.independent_{kind}_audit.v1",
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
        raise RuntimeError("Phase1234 preaudit already exists")
    contract = read_json(CONTRACT_PATH)
    material = read_jsonl(MATERIAL_PATH)
    manifest = read_jsonl(MANIFEST_PATH)
    program = read_json(PROGRAM_PATH)
    plan = read_json(PLAN_PATH)
    k199 = read_json(K199_FINAL_PATH)
    k199_fp16 = read_json(K199_FP16_PATH)
    p1233 = read_json(P1233_FINAL_PATH)
    p1233_audit = read_json(P1233_AUDIT_PATH)
    selection = independent_registry_selection(k199)
    independent_program = independent_program_audit(material, manifest)
    material_by_id = {row["item_id"]: row for row in material}

    split_lexicons = {
        split: {
            "objects": {item for row in material if row["split"] == split for item in row["objects"]},
            "values": {item for row in material if row["split"] == split for item in row["candidates"]},
        }
        for split in SPLITS
    }
    disjoint = all(
        not split_lexicons[left][kind] & split_lexicons[right][kind]
        for left_index, left in enumerate(SPLITS)
        for right in SPLITS[left_index + 1 :]
        for kind in ("objects", "values")
    )
    position_balanced = True
    for split in SPLITS:
        for panel in PANELS:
            counts = Counter(
                row["gold_position"] for row in material if row["split"] == split and row["panel"] == panel
            )
            position_balanced = position_balanced and set(counts) == set(range(5)) and max(counts.values()) - min(counts.values()) <= 1
    world_cells = grouped(material, "world_id")
    manifest_by_id = {row["item_id"]: row for row in manifest}
    exact_program_match = all(
        independent_program["split_results"][split] == program["split_results"][split]
        for split in SPLITS
    )
    suffix_equal = all(
        len({len(tokens) for tokens in row[f"{prefix}_candidate_token_ids"].values()}) == 1
        for row in manifest
        for prefix in ("world", "null", "open")
    )
    role_spans = all(
        set(row["world_role_token_spans"])
        == {"record_full", "record_object", "record_relation", "record_value", "query_full", "query_object", "query_relation", "answer_boundary"}
        for row in manifest
    )
    plan_counts = all(
        plan["views"][prefix]["length_counts"]
        == {str(key): value for key, value in sorted(Counter(row[f"{prefix}_input_token_count"] for row in manifest).items())}
        for prefix in ("world", "null", "open")
    )
    checks = {
        "phase1233_final_pinned": p1233.get("final_digest") == EXPECTED_P1233_FINAL,
        "phase1233_audit_pinned": p1233_audit.get("audit_digest") == EXPECTED_P1233_AUDIT and p1233_audit.get("all_checks_passed") is True,
        "k199_final_pinned": k199.get("final_digest") == EXPECTED_K199_FINAL,
        "k199_fp16_pinned": k199_fp16.get("audit_digest") == EXPECTED_K199_FP16 and k199_fp16.get("all_checks_passed") is True,
        "contract_self_digest": contract["contract_digest"] == digest(strip_digest(contract, "contract_digest")),
        "source_hashes_frozen": contract["source_hashes"] == source_hashes(),
        "material_digest": contract["material"]["material_digest"] == digest(material),
        "manifest_digest": contract["interface"]["tokenizer_summary"]["manifest_digest"] == digest(manifest),
        "program_self_digest": program["program_audit_digest"] == digest(strip_digest(program, "program_audit_digest")),
        "plan_self_digest": plan["plan_digest"] == digest(strip_digest(plan, "plan_digest")),
        "historical_selection_recomputed": selection == contract["historical_registry_selection"] and selection["selected_scope"] == "query_object|natural",
        "row_cardinality": len(material) == len(manifest) == EXPECTED_ROWS,
        "split_cardinality": all(sum(row["split"] == split for row in material) == ROWS_PER_SPLIT for split in SPLITS),
        "world_cardinality": len(world_cells) == 3 * WORLDS_PER_SPLIT and all(len(cell) == ROWS_PER_WORLD for cell in world_cells.values()),
        "world_panel_query_completeness": all(
            {(row["panel"], row["query_index"]) for row in cell} == {(panel, query) for panel in PANELS for query in range(4)}
            for cell in world_cells.values()
        ),
        "unique_item_ids": len(material_by_id) == EXPECTED_ROWS and len(manifest_by_id) == EXPECTED_ROWS,
        "material_row_self_digests": all(row["row_digest"] == digest(strip_digest(row, "row_digest")) for row in material),
        "manifest_row_self_digests": all(row["manifest_row_digest"] == digest(strip_digest(row, "manifest_row_digest")) for row in manifest),
        "material_manifest_links": all(
            row["material_row_digest"] == material_by_id[row["item_id"]]["row_digest"] for row in manifest
        ),
        "execution_order_frozen": [row["execution_index"] for row in manifest] == list(range(EXPECTED_ROWS)) and plan["execution_item_ids"] == [row["item_id"] for row in manifest],
        "three_split_lexicons_disjoint": disjoint,
        "gold_positions_balanced": position_balanced,
        "candidate_suffix_lengths_equal": suffix_equal,
        "role_spans_complete": role_spans,
        "tokenizer_gate": contract["interface"]["tokenizer_summary"]["tokenizer_gate"] is True,
        "maximum_input_length": contract["interface"]["tokenizer_summary"]["maximum_input_length"] <= 320,
        "batch_plan_counts": plan_counts,
        "query_group_count": independent_program["collision_group_counts"]["query_quartets"] == 576,
        "pair_group_counts": all(independent_program["collision_group_counts"][name] == 576 for name in ("order_pairs", "paraphrase_pairs", "binding_pairs")),
        "collision_checks": all(independent_program["collision_checks"].values()),
        "depth2_program_results_recomputed": exact_program_match,
        "program_ceilings_below_threshold": all(
            independent_program["split_results"][split]["depth2_conditional_program_accuracy"] <= PROGRAM_CEILING
            for split in SPLITS
        ),
        "target_equivalent_witness": independent_program["target_equivalent_accuracy"] == program["target_equivalent_witness"]["accuracy"] == 1.0,
        "program_construct_gate": independent_program["program_construct_gate"] is program["program_construct_gate"] is True,
        "thresholds_frozen": contract["thresholds"] == THRESHOLDS,
        "behavior_only_contract": contract["execution"] == {
            "batch_plan_digest": plan["plan_digest"],
            "hidden_states": False,
            "attentions": False,
            "interventions": False,
            "cross_model": False,
        },
        "no_model_weights_in_materialization": contract["interface"]["tokenizer_summary"]["model_weights_loaded"] is False,
        "new_lexicon_audit": contract["material"]["lexicon_audit"]["old_exact_term_overlap_count"] == 0,
    }
    value = audit_value(
        "preaudit",
        checks,
        {
            "contract_digest": contract["contract_digest"],
            "material_digest": digest(material),
            "manifest_digest": digest(manifest),
            "program_audit_digest": program["program_audit_digest"],
            "independent_registry_selection": selection,
            "independent_program_audit": independent_program,
        },
    )
    write_json(PREAUDIT_PATH, value)
    print(canonical_json({"status": "phase1234_preaudit", "passed": value["all_checks_passed"], "checks": f'{value["passed_check_count"]}/{value["total_check_count"]}', "failed": value["failed_checks"], "audit_digest": value["audit_digest"]}))
    if not value["all_checks_passed"]:
        raise RuntimeError(f"Phase1234 preaudit failed: {value['failed_checks']}")


def normalize_generated(text: str) -> str:
    value = text.strip().splitlines()[0] if text.strip() else ""
    value = value.strip().strip(string.whitespace + string.punctuation)
    return re.sub(r"\s+", " ", value.lower())


def parse_generated(text: str, candidates: list[str]) -> tuple[str | None, bool]:
    normalized = normalize_generated(text)
    matches: list[str] = []
    for candidate in candidates:
        candidate_norm = normalize_generated(candidate)
        if normalized == candidate_norm:
            return candidate, True
        if normalized.startswith(candidate_norm):
            suffix = normalized[len(candidate_norm) :]
            if not suffix or suffix[0] in " .,:;!?)]}\"'":
                matches.append(candidate)
    return (matches[0], False) if len(matches) == 1 else (None, False)


def argmax_set(scores: dict[str, float]) -> list[str]:
    maximum = max(scores.values())
    return sorted(name for name, value in scores.items() if maximum - value <= TIE_TOLERANCE)


def rate(rows: list[dict[str, Any]], field: str) -> float:
    return sum(bool(row[field]) for row in rows) / len(rows) if rows else float("nan")


def group_success_rate(rows: list[dict[str, Any]], key: str, size: int, mode: str) -> float:
    outcomes: list[bool] = []
    for cell in grouped(rows, key).values():
        predictions = [row["sum_prediction"] for row in cell]
        success = len(cell) == size and all(row["candidate_correct"] for row in cell)
        if mode == "distinct":
            success = success and len(set(predictions)) == size
        elif mode == "invariant":
            success = success and len(set(predictions)) == 1
        elif mode == "different":
            success = success and len(set(predictions)) == 2
        else:
            raise ValueError(mode)
        outcomes.append(success)
    return sum(outcomes) / len(outcomes) if outcomes else float("nan")


def independent_ledgers(raw: list[dict[str, Any]], program: dict[str, Any]) -> dict[str, Any]:
    ledgers: dict[str, Any] = {}
    passes: list[bool] = []
    for split in SPLITS:
        selected = [row for row in raw if row["split"] == split]
        by_panel = {
            panel: rate([row for row in selected if row["panel"] == panel], "candidate_correct")
            for panel in PANELS
        }
        worlds = grouped(selected, "world_id")
        reliable = sum(
            sum(row["candidate_correct"] for row in cell) / len(cell) >= 0.875 for cell in worlds.values()
        ) / len(worlds)
        metrics = {
            "finite_rate": sum(row["all_candidate_scores_finite"] and row["all_null_scores_finite"] for row in selected) / len(selected),
            "candidate_accuracy": rate(selected, "candidate_correct"),
            "context_adjusted_accuracy": rate(selected, "context_correct"),
            "open_generation_accuracy": rate(selected, "open_generation_correct"),
            "open_generation_exact_rate": rate(selected, "open_generation_exact"),
            "sum_mean_argmax_set_agreement": rate(selected, "sum_mean_argmax_set_agreement"),
            "candidate_by_panel": by_panel,
            "worst_panel_candidate": min(by_panel.values()),
            "query_quartet_success": group_success_rate(selected, "query_group_id", 4, "distinct"),
            "binding_rotation_pair_success": group_success_rate(selected, "binding_pair_id", 2, "different"),
            "order_pair_success": group_success_rate(selected, "order_pair_id", 2, "invariant"),
            "paraphrase_pair_success": group_success_rate(selected, "paraphrase_pair_id", 2, "invariant"),
            "reliable_world_rate": reliable,
            "depth2_program_ceiling": program["split_results"][split]["depth2_conditional_program_accuracy"],
        }
        gates = {
            "Q0_finite": metrics["finite_rate"] >= THRESHOLDS["Q0_finite_rate"],
            "Q1_candidate": metrics["candidate_accuracy"] >= THRESHOLDS["Q1_candidate_accuracy"],
            "Q1_context": metrics["context_adjusted_accuracy"] >= THRESHOLDS["Q1_context_adjusted_accuracy"],
            "Q1_open_generation": metrics["open_generation_accuracy"] >= THRESHOLDS["Q1_open_generation_accuracy"],
            "Q2_query_quartet": metrics["query_quartet_success"] >= THRESHOLDS["Q2_query_quartet_success"],
            "Q3_binding_rotation": metrics["binding_rotation_pair_success"] >= THRESHOLDS["Q3_binding_rotation_pair_success"],
            "Q4_order": metrics["order_pair_success"] >= THRESHOLDS["Q4_order_pair_success"],
            "Q4_paraphrase": metrics["paraphrase_pair_success"] >= THRESHOLDS["Q4_paraphrase_pair_success"],
            "Q5_worst_panel": metrics["worst_panel_candidate"] >= THRESHOLDS["Q5_worst_panel_candidate"],
            "Q5_reliable_world": metrics["reliable_world_rate"] >= THRESHOLDS["Q5_reliable_world_rate"],
            "Q6_sum_mean": metrics["sum_mean_argmax_set_agreement"] >= THRESHOLDS["Q6_sum_mean_argmax_agreement"],
            "Q7_program_ceiling": metrics["depth2_program_ceiling"] <= THRESHOLDS["Q7_program_ceiling"],
        }
        passed = all(gates.values())
        ledgers[split] = {"metrics": metrics, "gates": gates, "passed": passed}
        passes.append(passed)
    behavior = all(passes)
    return {
        "split_ledgers": ledgers,
        "program_construct_gate": bool(program["program_construct_gate"]),
        "behavior_gate": behavior,
        "future_response_eligibility": bool(program["program_construct_gate"] and behavior),
        "overall_candidate_accuracy": rate(raw, "candidate_correct"),
        "overall_context_adjusted_accuracy": rate(raw, "context_correct"),
        "overall_open_generation_accuracy": rate(raw, "open_generation_correct"),
        "tie_count": sum(len(row["sum_argmax_set"]) != 1 for row in raw),
        "nonfinite_count": sum(not (row["all_candidate_scores_finite"] and row["all_null_scores_finite"]) for row in raw),
    }


def result_audit() -> None:
    if RESULT_AUDIT_PATH.exists():
        raise RuntimeError("Phase1234 result audit already exists")
    contract = read_json(CONTRACT_PATH)
    material = read_jsonl(MATERIAL_PATH)
    manifest = read_jsonl(MANIFEST_PATH)
    program = read_json(PROGRAM_PATH)
    plan = read_json(PLAN_PATH)
    preaudit_value = read_json(PREAUDIT_PATH)
    raw = read_jsonl(RAW_PATH)
    summary = read_json(SUMMARY_PATH)
    material_by_id = {row["item_id"]: row for row in material}
    manifest_by_id = {row["item_id"]: row for row in manifest}

    score_checks: list[bool] = []
    generation_checks: list[bool] = []
    for row in raw:
        material_row = material_by_id[row["item_id"]]
        candidates = material_row["candidates"]
        scores = row["candidate_scores"]
        null_scores = row["null_candidate_scores"]
        sum_map = {candidate: float(scores[candidate]["sum_log_probability"]) for candidate in candidates}
        mean_map = {candidate: float(scores[candidate]["mean_log_probability"]) for candidate in candidates}
        null_map = {candidate: float(null_scores[candidate]["sum_log_probability"]) for candidate in candidates}
        context_map = {candidate: sum_map[candidate] - null_map[candidate] for candidate in candidates}
        sum_set = argmax_set(sum_map)
        mean_set = argmax_set(mean_map)
        context_set = argmax_set(context_map)
        ordered = sorted(sum_map.values(), reverse=True)
        score_checks.append(
            set(scores) == set(null_scores) == set(candidates)
            and row["context_adjusted_scores"] == context_map
            and row["sum_argmax_set"] == sum_set
            and row["mean_argmax_set"] == mean_set
            and row["context_argmax_set"] == context_set
            and row["sum_prediction"] == (sum_set[0] if len(sum_set) == 1 else None)
            and row["context_prediction"] == (context_set[0] if len(context_set) == 1 else None)
            and row["candidate_correct"] is (row["sum_prediction"] == material_row["gold"])
            and row["context_correct"] is (row["context_prediction"] == material_row["gold"])
            and row["sum_mean_argmax_set_agreement"] is (sum_set == mean_set)
            and math.isclose(row["sum_margin"], ordered[0] - ordered[1], rel_tol=0.0, abs_tol=1e-12)
            and row["all_candidate_scores_finite"] is all(bool(scores[candidate]["all_vocab_logits_finite"]) for candidate in candidates)
            and row["all_null_scores_finite"] is all(bool(null_scores[candidate]["all_vocab_logits_finite"]) for candidate in candidates)
        )
        generation_prediction, generation_exact = parse_generated(row["open_generated_text"], candidates)
        generation_checks.append(
            row["open_generation_prediction"] == generation_prediction
            and row["open_generation_exact"] is generation_exact
            and row["open_generation_correct"] is (generation_prediction == material_row["gold"])
        )
    ledgers = independent_ledgers(raw, program)
    checks = {
        "preaudit_passed": preaudit_value.get("all_checks_passed") is True,
        "source_hashes_unchanged": contract["source_hashes"] == source_hashes(),
        "raw_cardinality": len(raw) == EXPECTED_ROWS,
        "raw_identity_coverage": {row["item_id"] for row in raw} == set(material_by_id) == set(manifest_by_id),
        "raw_execution_order": [row["execution_index"] for row in raw] == list(range(EXPECTED_ROWS)),
        "raw_row_self_digests": all(row["behavior_row_digest"] == digest(strip_digest(row, "behavior_row_digest")) for row in raw),
        "raw_manifest_links": all(row["manifest_row_digest"] == manifest_by_id[row["item_id"]]["manifest_row_digest"] for row in raw),
        "raw_contract_links": all(row["contract_digest"] == contract["contract_digest"] for row in raw),
        "score_decisions_recomputed": all(score_checks),
        "generation_decisions_recomputed": all(generation_checks),
        "summary_self_digest": summary["summary_digest"] == digest(strip_digest(summary, "summary_digest")),
        "summary_raw_digest": summary["raw_digest"] == digest(raw),
        "summary_case_count": summary["case_count"] == EXPECTED_ROWS,
        "summary_contract_link": summary["contract_digest"] == contract["contract_digest"],
        "summary_plan_link": summary["batch_plan_digest"] == plan["plan_digest"],
        "cuda_fp16_unquantized": summary["placement"]["placement"] == "full_cuda" and set(summary["precision_audit"]["parameter_dtypes"]) == {"float16"} and summary["precision_audit"]["has_quantized_modules"] is False,
        "no_hidden_attentions_interventions": summary["hidden_states_saved"] is False and summary["attentions_saved"] is False and summary["interventions_performed"] is False,
        "ledgers_complete": set(ledgers["split_ledgers"]) == set(SPLITS),
        "program_gate_retained": ledgers["program_construct_gate"] is True,
    }
    value = audit_value(
        "result",
        checks,
        {
            "contract_digest": contract["contract_digest"],
            "raw_digest": digest(raw),
            "run_summary_digest": summary["summary_digest"],
            "recomputed_ledgers": ledgers,
        },
    )
    write_json(RESULT_AUDIT_PATH, value)
    print(canonical_json({"status": "phase1234_result_audit", "passed": value["all_checks_passed"], "checks": f'{value["passed_check_count"]}/{value["total_check_count"]}', "failed": value["failed_checks"], "audit_digest": value["audit_digest"], "future_response_eligibility": ledgers["future_response_eligibility"]}))
    if not value["all_checks_passed"]:
        raise RuntimeError(f"Phase1234 result audit failed: {value['failed_checks']}")


def final_audit() -> None:
    if FINAL_AUDIT_PATH.exists():
        raise RuntimeError("Phase1234 final audit already exists")
    preaudit_value = read_json(PREAUDIT_PATH)
    result_value = read_json(RESULT_AUDIT_PATH)
    final = read_json(FINAL_PATH)
    passed = bool(result_value["recomputed_ledgers"]["future_response_eligibility"])
    checks = {
        "preaudit_passed": preaudit_value.get("all_checks_passed") is True,
        "result_audit_passed": result_value.get("all_checks_passed") is True,
        "final_self_digest": final["final_digest"] == digest(strip_digest(final, "final_digest")),
        "ledger_recomputation": final["ledgers"] == result_value["recomputed_ledgers"],
        "result_audit_link": final["result_audit_digest"] == result_value["audit_digest"],
        "status_typed": final["status"] == ("sealed_atomic_object_confirmed" if passed else "sealed_atomic_object_confirmation_failed"),
        "k209_identifier": final["k_item"]["identifier"] == "K209",
        "k209_grade": final["k_item"]["evidence_grade"] == ("E3-BEHAVIOR-CONSTRUCT" if passed else "E3-NEGATIVE-BOUNDARY"),
        "behavior_authorization": final["authorization"]["selected_behavior_object"] is passed,
        "future_response_authorization": final["authorization"]["future_response_phase"] is passed,
        "auto_continue_typed": final["authorization"]["auto_continue"] is passed,
        "next_experiment_typed": (final["authorization"]["next_experiment"] is not None) is passed,
        "no_hidden_in_phase": final["authorization"]["hidden_scan_in_this_phase"] is False,
        "no_cross_model_rescue": final["authorization"]["cross_model_run"] is False,
        "no_unique_algorithm_claim": final["authorization"]["unique_neural_algorithm_claim"] is False,
        "no_new_mathematics_claim": final["new_mathematics_required"] is False,
    }
    value = audit_value(
        "final",
        checks,
        {
            "final_digest": final["final_digest"],
            "result_audit_digest": result_value["audit_digest"],
            "future_response_eligibility": passed,
            "authorized_next": final["authorization"]["next_experiment"],
        },
    )
    write_json(FINAL_AUDIT_PATH, value)
    print(canonical_json({"status": "phase1234_final_audit", "passed": value["all_checks_passed"], "checks": f'{value["passed_check_count"]}/{value["total_check_count"]}', "failed": value["failed_checks"], "audit_digest": value["audit_digest"], "auto_continue": passed}))
    if not value["all_checks_passed"]:
        raise RuntimeError(f"Phase1234 final audit failed: {value['failed_checks']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=("preaudit", "result", "final"))
    stage = parser.parse_args().stage
    {"preaudit": preaudit, "result": result_audit, "final": final_audit}[stage]()


if __name__ == "__main__":
    main()
