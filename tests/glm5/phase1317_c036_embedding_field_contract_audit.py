#!/usr/bin/env python3
"""Independent audit for Phase1317 C036 preregistration and frozen material."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
T = ROOT / "tests/glm5"
OUT = T / "result/phase1317_c036_embedding_field_contract"
PROTOCOL = OUT / "protocol/preregistration.json"
SOURCE = OUT / "material/frozen_forward_lookup_cases.jsonl"
PAIRS = OUT / "material/frozen_forward_lookup_pairs.jsonl"
NATURALNESS = OUT / "material/pre_model_semantic_naturalness_review.json"
MACHINE = OUT / "audit/tokenizer_semantic_program_audit.json"
CALIBRATION = OUT / "analysis/known_truth_field_decomposition_calibration.json"
FINAL = OUT / "analysis/final.json"
AUDIT = OUT / "audit/independent_final_audit.json"
MAIN = T / "phase1317_c036_embedding_field_contract.py"
SELF = Path(__file__).resolve()


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main() -> None:
    protocol, source, pairs = load(PROTOCOL), rows(SOURCE), rows(PAIRS)
    naturalness, machine, calibration, final = load(NATURALNESS), load(MACHINE), load(CALIBRATION), load(FINAL)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in source:
        grouped.setdefault(row["group_id"], []).append(row)
    semantic_unique = all(
        row["assignments"][row["query_entity"]][row["attribute"]] == row["gold_value"]
        and row["candidates"].count(row["gold_value"]) == 1 for row in source
    )
    panel_changes: dict[str, list[bool]] = {}
    for pair in pairs:
        panel_changes.setdefault(pair["panel"], []).append(pair["states"][0]["gold_value"] != pair["states"][1]["gold_value"])
    checks = {
        "protocol_material_hashes": protocol["material"]["source_sha256"] == sha(SOURCE)
        and protocol["material"]["pairs_sha256"] == sha(PAIRS)
        and protocol["material"]["naturalness_sha256"] == sha(NATURALNESS)
        and protocol["material"]["machine_sha256"] == sha(MACHINE),
        "source_hashes": protocol["source_hashes"] == {"main": sha(MAIN), "auditor": sha(SELF)},
        "parent_terminal_audited": load(T / "result/phase1316_c035_typed_multireadout_rescue/analysis/final.json").get("authorization")
        == "close_c035_with_multisite_dependence_without_type_selectivity"
        and load(T / "result/phase1316_c035_typed_multireadout_rescue/audit/independent_final_audit.json").get("all_checks_passed"),
        "counts": len(source) == 1152 and len(pairs) == 576 and len(grouped) == 576,
        "partition_balance": Counter(row["partition"] for row in source) == Counter({"discovery": 384, "confirmation": 384, "holdout": 384}),
        "attribute_balance": len(set(Counter(row["attribute"] for row in source).values())) == 1,
        "surface_balance": len(set(Counter(row["surface"] for row in source).values())) == 1,
        "panel_balance": len(set(Counter(row["panel"] for row in source).values())) == 1,
        "pairs_exact": all(len(value) == 2 and {row["binding_state"] for row in value} == {0, 1} for value in grouped.values()),
        "semantic_unique": semantic_unique,
        "active_changes_only": all(panel_changes["active"]) and all(not changed for panel in panel_changes if panel != "active" for changed in panel_changes[panel]),
        "machine_pass": machine["all_machine_checks_passed"],
        "no_prior_names": not machine["token_audit"]["prior_name_overlap"],
        "no_prior_values": not machine["token_audit"]["prior_value_overlap"],
        "single_tokens": machine["token_audit"]["all_names_single_token"]
        and machine["token_audit"]["all_values_single_token"]
        and machine["token_audit"]["all_attributes_single_token"],
        "same_shape_roles": machine["token_audit"]["same_shape_and_site_alignment_within_pairs"],
        "shortcuts_chance": max(machine["shortcut_accuracy"].values()) <= 0.5,
        "naturalness_pass": naturalness["all_checks_passed"],
        "naturalness_scope_honest": naturalness["independent_human_panel"] is False and "no independent human" in naturalness["limitation"],
        "known_truth_pass": calibration["all_gates_passed"] and all(v == 1.0 for v in calibration["class_accuracy"].values()),
        "response_twin_abstains": calibration["response_twin_generator_identification"] == "abstain",
        "behavior_before_hidden": protocol["behavior"]["hidden_states_read"] is False,
        "qwen_only": protocol["model"] == "qwen3-4b-fp16-cuda-no-quantization" and set(protocol["models_excluded"]) == {"glm4", "deepseek7b"},
        "frozen_field_object": "all-layer/all-position" in protocol["research_object"] and len(protocol["field"]["registered_roles"]) == 6,
        "decomposition_frozen": protocol["field"]["decomposition"].startswith("G=mean_attribute"),
        "nulls_present": {"wrong_attribute", "matched_null", "equal_norm_fixed_random", "response_twin_generator"}.issubset(protocol["zero_models"]),
        "hard_stop_present": any("C036 closes" in item for item in protocol["hard_stops"]),
        "final_consistent": final["authorization"] == "phase1318_qwen3_behavior_only" and final["all_gates_passed"],
        "no_model_loaded": protocol["model_weights_loaded"] is False,
    }
    all_passed = all(checks.values())
    authorization = "run_phase1318_preregistration" if all_passed else "close_c036_before_model"
    save(AUDIT, {"phase": 1317, "campaign": "C036", "checks": checks, "passed": sum(checks.values()),
                 "total": len(checks), "all_checks_passed": all_passed, "authorization": authorization,
                 "claim_boundary": "Audits frozen material and contract integrity; does not inspect model weights or hidden states."})
    print(json.dumps({"passed": sum(checks.values()), "total": len(checks), "authorization": authorization}, ensure_ascii=False))


if __name__ == "__main__":
    main()
