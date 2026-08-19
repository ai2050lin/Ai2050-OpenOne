#!/usr/bin/env python3
"""Phase1309: freeze C034 typed-response camera, material, gates, and stop branches."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
T = ROOT / "tests/glm5"
sys.path.insert(0, str(T))
from model_utils import MODEL_CONFIGS  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

PHASE = 1309
CAMPAIGN = "C034"
SCRIPT = Path(__file__).resolve()
AUDITOR = T / "phase1309_c034_typed_response_camera_contract_audit.py"
C033 = T / "result/phase1304_c033_role_typed_causal_graph_contract"
C033_FINAL = T / "result/phase1308_c033_cross_surface_block_rescue/analysis/final.json"
C033_AUDIT = T / "result/phase1308_c033_cross_surface_block_rescue/audit/independent_final_audit.json"
SOURCE_MATERIAL = C033 / "material/frozen_role_typed_lookup_cases.jsonl"
OUT = T / "result/phase1309_c034_typed_response_camera_contract"
PROTOCOL = OUT / "protocol/preregistration.json"
MATERIAL = OUT / "material/frozen_typed_response_pairs.jsonl"
NATURALNESS = OUT / "material/semantic_naturalness_review.json"
CALIBRATION = OUT / "analysis/known_truth_camera_calibration.json"
MACHINE = OUT / "audit/tokenizer_semantic_program_audit.json"
AUDIT = OUT / "audit/independent_final_audit.json"
FINAL = OUT / "analysis/final.json"

SYSTEM = "Use only the supplied catalog. Reply exactly as requested and do not explain."
PARTITIONS = ("discovery", "confirmation", "holdout")
ATTRS = ("color", "material", "location", "size", "shape", "status")
SURFACES = ("catalog_prose", "inventory_ledger")
PANELS = ("active", "matched_null")
QUERY_DEPTHS = (8, 14, 20, 26, 32)
LATE_ROLE = "answer_boundary"
LATE_DEPTH = 26

BEHAVIOR_TH = {
    "finite_fraction_min": 1.0,
    "candidate_accuracy_min": 0.98,
    "partition_accuracy_min": 0.97,
    "attribute_accuracy_min": 0.96,
    "surface_accuracy_min": 0.97,
    "active_pair_success_min": 0.95,
    "attribute_family_success_min": 0.90,
    "generation_coverage_min": 0.98,
    "generation_label_accuracy_min": 0.97,
    "generation_pair_success_min": 0.93,
}
TRAJECTORY_TH = {
    "finite_fraction_min": 1.0,
    "behavior_replay_accuracy_min": 0.98,
    "same_attribute_cross_surface_cosine_median_min": 0.15,
    "type_gap_median_min": 0.05,
    "type_gap_positive_fraction_min": 0.65,
    "active_to_null_norm_ratio_min": 1.10,
    "upstream_over_late_type_gap_min": 0.05,
}
CAUSAL_TH = {
    "finite_fraction_min": 1.0,
    "baseline_accuracy_min": 0.98,
    "blocked_target_identity_accuracy_max": 0.60,
    "correct_rescue_accuracy_min": 0.65,
    "recovery_fraction_median_min": 0.50,
    "correct_over_null_margin_ratio_min": 1.25,
    "pairwise_correct_win_fraction_min": 0.70,
    "natural_retention_min": 0.98,
}


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_rows(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for value in values:
            f.write(canonical(value) + "\n")


def render(tokenizer: Any, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def tokenized_state(tokenizer: Any, row: dict[str, Any]) -> dict[str, Any]:
    text = render(tokenizer, row["candidate_prompt"])
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = [int(x) for x in encoded["input_ids"]]
    offsets = [(int(a), int(b)) for a, b in encoded["offset_mapping"]]
    base = text.find(row["candidate_prompt"])
    if base < 0:
        raise RuntimeError("prompt not found in rendered chat")
    positions = {}
    for role, span in (
        ("query_end", row["typed_spans"]["query"][0]),
        ("answer_boundary", row["typed_spans"]["answer_boundary"][0]),
    ):
        left, right = base + span[0], base + span[1]
        hits = [i for i, (a, b) in enumerate(offsets) if b > left and a < right and b > a]
        if not hits:
            raise RuntimeError(f"missing role {role}")
        positions[role] = hits[-1]
    candidate_ids = []
    for name in row["candidates"]:
        full = tokenizer.encode(text + " " + name, add_special_tokens=False)
        if full[:len(ids)] != ids or len(full) != len(ids) + 1:
            raise RuntimeError("candidate token drift")
        candidate_ids.append(int(full[-1]))
    return {
        "case_id": row["case_id"],
        "ids": ids,
        "positions": positions,
        "candidate_ids": candidate_ids,
        "gold_position": int(row["gold_position"]),
    }


def build_material(tokenizer: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source = [r for r in rows(SOURCE_MATERIAL) if r["candidate_order"] == 0 and r["panel"] in PANELS]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in source:
        grouped[row["group_id"]].append(row)
    result = []
    role_lengths = defaultdict(list)
    for group_id, pair in sorted(grouped.items()):
        pair = sorted(pair, key=lambda x: x["binding_state"])
        if len(pair) != 2:
            raise RuntimeError("incomplete binding pair")
        states = [tokenized_state(tokenizer, row) for row in pair]
        first = pair[0]
        identity_positions = [int(pair[0]["gold_position"]), int(pair[1]["gold_position"])]
        for state in states:
            role_lengths["query_end"].append(state["positions"]["query_end"])
            role_lengths["answer_boundary"].append(state["positions"]["answer_boundary"])
        result.append({
            "pair_key": f"{first['partition']}|p{first['profile_index']:02d}|{first['attribute']}|{first['surface']}|{first['panel']}",
            "group_id": group_id,
            "partition": first["partition"],
            "profile_index": first["profile_index"],
            "attribute": first["attribute"],
            "surface": first["surface"],
            "panel": first["panel"],
            "candidates": first["candidates"],
            "identity_positions": identity_positions,
            "states": states,
        })
    audit = {
        "pair_count": len(result),
        "state_count": 2 * len(result),
        "partition_counts": {p: sum(x["partition"] == p for x in result) for p in PARTITIONS},
        "panel_counts": {p: sum(x["panel"] == p for x in result) for p in PANELS},
        "all_candidate_ids_unique_within_case": all(len(set(s["candidate_ids"])) == 3 for x in result for s in x["states"]),
        "all_active_pairs_change_gold": all(x["identity_positions"][0] != x["identity_positions"][1] for x in result if x["panel"] == "active"),
        "all_null_pairs_keep_gold": all(x["states"][0]["gold_position"] == x["states"][1]["gold_position"] for x in result if x["panel"] == "matched_null"),
        "all_roles_ordered": all(s["positions"]["query_end"] < s["positions"]["answer_boundary"] < len(s["ids"]) for x in result for s in x["states"]),
        "role_position_ranges": {k: [min(v), max(v)] for k, v in role_lengths.items()},
    }
    return result, audit


def known_truth_calibration() -> dict[str, Any]:
    examples = []
    for target_attribute in range(len(ATTRS)):
        signatures = {
            "generic": [1] * len(ATTRS),
            "typed": [int(a == target_attribute) for a in range(len(ATTRS))],
            "mixed": [int(a in {target_attribute, (target_attribute + 1) % len(ATTRS)}) for a in range(len(ATTRS))],
            "null": [0] * len(ATTRS),
        }
        for label, signature in signatures.items():
            single = signature[target_attribute]
            if sum(signature) == len(ATTRS):
                predicted = "generic"
            elif sum(signature) == 1 and signature[target_attribute] == 1:
                predicted = "typed"
            elif sum(signature) == 2 and signature[target_attribute] == 1:
                predicted = "mixed"
            else:
                predicted = "null"
            examples.append({"target_attribute": ATTRS[target_attribute], "label": label,
                             "single_readout": single, "signature": signature, "prediction": predicted})
    accuracy = sum(x["label"] == x["prediction"] for x in examples) / len(examples)
    collision = sum(x["single_readout"] == 1 for x in examples if x["label"] in {"generic", "typed"}) / (2 * len(ATTRS))
    return {
        "schema_version": "phase1309.known_truth.typed_camera.v1",
        "classes": ["generic", "typed", "mixed", "null"],
        "attribute_count": len(ATTRS),
        "example_count": len(examples),
        "single_target_readout_generic_typed_collision_fraction": collision,
        "multi_readout_classification_accuracy": accuracy,
        "null_false_positive_fraction": sum(x["prediction"] != "null" for x in examples if x["label"] == "null") / len(ATTRS),
        "constant_first_class_baseline_accuracy": 0.25,
        "examples": examples,
        "all_gates_passed": collision == 1.0 and accuracy == 1.0,
    }


def build(force: bool) -> None:
    if load(C033_FINAL).get("authorization") != "close_c033_at_rescue_boundary" or not load(C033_AUDIT).get("all_checks_passed"):
        raise RuntimeError("C033 terminal branch unavailable")
    if OUT.exists() and not force:
        raise RuntimeError(f"{OUT} already exists")
    if OUT.exists():
        shutil.rmtree(OUT)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIGS["qwen3"]["path"], trust_remote_code=True,
                                              local_files_only=True, use_fast=True)
    material, material_audit = build_material(tokenizer)
    write_rows(MATERIAL, material)
    source_rows = rows(SOURCE_MATERIAL)
    semantic_unique = all(
        sum(fields[row["attribute"]] == row["target_value"] for fields in row["assignments"].values()) == 1
        for row in source_rows
    )
    source_naturalness = load(C033 / "material/pre_model_semantic_naturalness_review.json")
    naturalness = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "reviewed_before_model_weight_load": True,
        "source_material_reused_intentionally": True,
        "source_material_sha256": sha(SOURCE_MATERIAL),
        "source_naturalness_sha256": sha(C033 / "material/pre_model_semantic_naturalness_review.json"),
        "source_naturalness_passed": source_naturalness["all_checks_passed"],
        "semantic_uniqueness_recomputed": semantic_unique,
        "pair_prompts_unchanged_from_c033": True,
        "limitation": "C034 reuses C033 controlled-English worlds to isolate measurement changes; it is not an independent-material confirmation and has no independent human panel.",
        "all_checks_passed": bool(source_naturalness["all_checks_passed"] and semantic_unique),
    }
    save(NATURALNESS, naturalness)
    save(MACHINE, {"phase": PHASE, "campaign": CAMPAIGN, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                   **material_audit, "material_sha256": sha(MATERIAL),
                   "all_machine_checks_passed": all([
                       material_audit["pair_count"] == 576,
                       material_audit["state_count"] == 1152,
                       material_audit["all_candidate_ids_unique_within_case"],
                       material_audit["all_active_pairs_change_gold"],
                       material_audit["all_null_pairs_keep_gold"],
                       material_audit["all_roles_ordered"],
                   ])})
    calibration = known_truth_calibration()
    save(CALIBRATION, calibration)
    timeless = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "schema_version": "phase1309.c034.typed_response_contract.v1",
        "purpose": "distinguish generic identity displacement from attribute-typed computation using a multi-readout response family, then test upstream type separation and late convergence in Qwen3",
        "adjudication": {
            "accepted": [
                "C033 executed the full behavior-descriptive-causal-rescue chain but did not close its terminal gate",
                "C033 established strong local answer-identity manipulation and descriptive cross-surface rescue",
                "correct and wrong-attribute identity-matched deltas were indistinguishable under the registered answer-identity readout",
            ],
            "corrected_overclaims": [
                "C033 did not prove that the late residual lacks attribute information; it failed to identify such information under the registered intervention/readout family",
                "C033 did not prove a universal generic-identity operator; shared output displacement remains an alternative",
                "cross-surface identity rescue was descriptively successful but the registered rescue gate failed",
                "T=0 denotes failure of one frozen type-selectivity test, not absence of type information in Qwen3",
                "the full evidential chain was executed, not closed",
            ],
        },
        "known_truth_camera": {
            "classes": ["generic", "typed", "mixed", "null"],
            "single_readout_collision_required": 1.0,
            "multi_readout_accuracy_required": 1.0,
            "calibration_sha256": sha(CALIBRATION),
        },
        "material": {
            "source": "C033 frozen controlled-English material; prompts unchanged",
            "independent_material": False,
            "pair_count": 576,
            "state_count": 1152,
            "partitions": list(PARTITIONS),
            "attributes": list(ATTRS),
            "surfaces": list(SURFACES),
            "panels": list(PANELS),
            "material_sha256": sha(MATERIAL),
            "naturalness_sha256": sha(NATURALNESS),
        },
        "model": {"model_id": "qwen3-4b-fp16-cuda-no-quantization", "compiler": "right_padding",
                  "formal_runs_per_model_phase": 1, "other_models_authorized": False},
        "behavior": {"thresholds": BEHAVIOR_TH},
        "trajectory": {
            "roles": ["query_end", "answer_boundary"],
            "query_depths": list(QUERY_DEPTHS),
            "late_comparator": {"role": LATE_ROLE, "depth": LATE_DEPTH},
            "candidate_selection": "on discovery only, choose the eligible query_end depth with maximal median type gap; deterministic shallow-depth tie break",
            "same_attribute": "cosine between identity-aligned active deltas across the two surfaces",
            "wrong_attribute": "cosine between anchor-surface active delta and opposite-surface next-attribute delta aligned to the same identity transition",
            "type_gap": "same-attribute cosine minus wrong-attribute cosine",
            "thresholds": TRAJECTORY_TH,
            "confirmation_rule": "selected depth must pass all thresholds separately in confirmation and holdout and exceed each partition's late type gap",
        },
        "causal": {
            "block": "at query_end selected depth, replace target active-state1 with same-target active-state0 residual",
            "rescue": "at selected depth+1 add opposite-surface correct-attribute active identity1-minus-identity0 delta",
            "controls": ["block_only", "matched_null_delta", "identity-matched_wrong_attribute_delta", "self_retention"],
            "thresholds": CAUSAL_TH,
            "claim_scope": "upstream attribute-selective local rescue within one Qwen3 model and entity world; not minimal, cross-world, or cross-model",
        },
        "branches": {
            "phase1309_fail": "close_c034_before_model",
            "phase1309_pass": "phase1310_qwen3_behavior_only",
            "phase1310_fail": "close_c034_without_hidden",
            "phase1310_pass": "phase1311_typed_trajectory_only",
            "phase1311_fail": "close_c034_without_causal",
            "phase1311_pass": "phase1312_upstream_selective_rescue_only",
            "phase1312_any_verdict": "close_c034",
        },
        "hard_stops": [
            "No model weights before Phase1309 independent audit passes",
            "No hidden states before Phase1310 behavior gate passes",
            "No causal intervention before Phase1311 trajectory gate passes",
            "No post-unblinding role, depth, metric, donor, threshold, or partition change",
            "No head, MLP, neuron, or subspace search in C034",
            "C034 closes after Phase1312 or at the first failed gate",
        ],
        "dependencies": {"c033_final": sha(C033_FINAL), "c033_audit": sha(C033_AUDIT),
                         "source_material": sha(SOURCE_MATERIAL), "material": sha(MATERIAL),
                         "calibration": sha(CALIBRATION)},
        "source_hashes": {"main": sha(SCRIPT), "auditor": sha(AUDITOR)},
        "model_weights_loaded": False,
    }
    protocol = {**timeless, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "protocol_digest": digest(timeless)}
    save(PROTOCOL, protocol)
    passed = bool(load(MACHINE)["all_machine_checks_passed"] and naturalness["all_checks_passed"] and calibration["all_gates_passed"])
    authorization = "phase1310_qwen3_behavior_only" if passed else "close_c034_before_model"
    save(FINAL, {"phase": PHASE, "campaign": CAMPAIGN,
                 "verdict": "contract_and_camera_qualified" if passed else "contract_or_camera_failed",
                 "all_gates_passed": passed, "authorization": authorization,
                 "protocol_digest": protocol["protocol_digest"]})
    print(canonical({"pairs": len(material), "camera": calibration["multi_readout_classification_accuracy"],
                     "authorization": authorization, "digest": protocol["protocol_digest"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build",))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    build(args.force)
