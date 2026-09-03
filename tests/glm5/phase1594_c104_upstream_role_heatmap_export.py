#!/usr/bin/env python3
"""Phase1594 / C104-C105: export the upstream barcode, raw role trajectory and corrected causal map."""
from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
C104 = TESTS / "result/phase1589_c104_upstream_candidate_validation"
C105 = TESTS / "result/phase1593_c105_candidate_order_intervention_correction"
PUBLIC = ROOT / "frontend/public/vis_data/research_kernel/c104_upstream_role_barcode_heatmap.json"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core

PHASE = 1594
CAMPAIGN = "C104-C105"
FAMILIES = ("attribute_binding", "agent_patient", "negation_scope", "whole_part_exception")
EFFECTS = ("truth", "code", "truth_x_code")


def decode_bf16(bits: np.ndarray) -> np.ndarray:
    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def partition_vector(coeff: np.ndarray, units: list[dict[str, Any]], family: str, partition: str,
                     effect: int, state: int, role: int) -> np.ndarray:
    selected = [row["row_index"] for row in units if row["family"] == family and row["partition"] == partition]
    return np.asarray(coeff[selected, effect, state, role], dtype=np.float64).mean(axis=0).astype(np.float32)


def main() -> None:
    c105 = core.load(C105 / "analysis/final.json")
    audit = core.load(C105 / "audit/independent_final_audit.json")
    if c105["authorization"] != "export_corrected_c104_heatmap_and_close_c102_c104_c105_stage" or not audit["all_checks_passed"]:
        raise RuntimeError("C104-C105 heatmap export not authorized")
    contract = core.load(C104 / "protocol/preregistration.json")
    predictions = {row["family"]: row for row in contract["predictions"]}
    source = np.load(ROOT / contract["barcode_path"], mmap_mode="r")
    coeff = np.load(C104 / "raw/qwen3_breadth_three_effect_coefficients.float32.npy", mmap_mode="r")
    units = core.rows(C104 / "raw/qwen3_breadth_three_effect_index.jsonl")
    effect_rows = []
    for family_index, family in enumerate(FAMILIES):
        prediction = predictions[family]
        effect_rows.append({
            "dataset": "c103_frozen_source", "partition": "revealed_source", "family": family,
            "effect": "truth_residual", "role": prediction["role"], "state": prediction["state"],
            "values": np.asarray(source[family_index], dtype=np.float32).tolist(),
        })
        for partition in ("response_discovery", "confirmation", "lockbox"):
            for effect_index, effect in enumerate(EFFECTS):
                effect_rows.append({
                    "dataset": "c104_fresh", "partition": partition, "family": family, "effect": effect,
                    "role": prediction["role"], "state": prediction["state"],
                    "values": partition_vector(coeff, units, family, partition, effect_index,
                                                int(prediction["state"]), int(prediction["role_index"])).tolist(),
                })
    raw_field = np.load(C104 / "raw/qwen3_all_token_state_coordinate_field.uint16.npy", mmap_mode="r")
    raw_index = core.rows(C104 / "raw/qwen3_all_token_state_coordinate_index.jsonl")
    representatives = {family: next(row for row in raw_index if row["family"] == family and row["partition"] == "lockbox") for family in FAMILIES}
    raw_rows = []
    for family, row in representatives.items():
        role = predictions[family]["role"]
        for state in range(37):
            for subtoken, local_position in enumerate(row["role_positions"][role]):
                raw_rows.append({
                    "scope": "frozen_role_all_states", "case_id": row["case_id"], "family": family,
                    "partition": row["partition"], "token_position": local_position, "subtoken": subtoken,
                    "token_id": row["token_ids"][local_position], "token_text": row["token_texts"][local_position],
                    "role": role, "state": state, "state_kind": "embedding" if state == 0 else "hidden_state",
                    "values": decode_bf16(raw_field[state, row["token_start"] + local_position]).tolist(),
                })
    representative = representatives["attribute_binding"]
    selected_state = int(predictions["attribute_binding"]["state"])
    for state in (0, selected_state):
        for local_position, (token_id, token_text) in enumerate(zip(representative["token_ids"], representative["token_texts"], strict=True)):
            raw_rows.append({
                "scope": "all_tokens_representative", "case_id": representative["case_id"], "family": "attribute_binding",
                "partition": representative["partition"], "token_position": local_position, "subtoken": 0,
                "token_id": token_id, "token_text": token_text,
                "role": next((role for role, positions in representative["role_positions"].items() if local_position in positions), "unregistered"),
                "state": state, "state_kind": "embedding" if state == 0 else "hidden_state",
                "values": decode_bf16(raw_field[state, representative["token_start"] + local_position]).tolist(),
            })
    intervention_rows = core.rows(C105 / "analysis/c104_corrected_intervention_summary.jsonl")
    rollup_rows = core.rows(C105 / "analysis/c104_corrected_family_rollup.jsonl")
    lockbox_truth = [row for row in effect_rows if row["dataset"] == "c104_fresh" and row["partition"] == "lockbox" and row["effect"] == "truth"]
    mean_abs = np.mean(np.stack([np.abs(np.asarray(row["values"], dtype=np.float64)) for row in lockbox_truth]), axis=0)
    default_coordinates = np.argsort(-mean_abs, kind="stable")[:64].astype(int).tolist()
    effect_abs = np.concatenate([np.abs(np.asarray(row["values"], dtype=np.float32)) for row in effect_rows])
    behavior = core.load(C104 / "analysis/qwen_full_field_capture_summary.json")["behavior"]
    payload = {
        "schema": "c104_upstream_role_barcode_heatmap.v1",
        "result_type": "upstream_role_barcode_heatmap",
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "model": "Qwen3-4B",
        "title": "C104 Frozen Upstream Role-State Barcode and Causal Map",
        "coordinate_semantics": "H[state, token, activation_coordinate]; all 2560 residual-stream activation coordinates are preserved. They are not model weights, MLP neurons or attention heads.",
        "dimensions": list(range(2560)),
        "default_coordinates": default_coordinates,
        "scale": {"effect_symmetric_abs_q99": float(np.quantile(effect_abs, 0.99))},
        "effect_rows": effect_rows,
        "raw_rows": raw_rows,
        "intervention_rows": intervention_rows,
        "rollup_rows": rollup_rows,
        "headline": {
            "fresh_barcode_passed": 4, "fresh_barcode_total": 4,
            "fully_controlled_intervention_passed": len(c105["c104"]["fully_controlled_families"]),
            "controlled_intervention_total": 4,
            "partially_controlled_cells": c105["c104"]["partially_controlled"],
            "behavior_accuracy": behavior["global_accuracy"],
            "standard_accuracy": behavior["by_code"]["standard"],
            "reversed_accuracy": behavior["by_code"]["reversed"],
        },
        "source": {
            "raw_sha256": core.load(C104 / "analysis/qwen_full_field_capture_summary.json")["raw_sha256"],
            "validation_sha256": core.sha(C104 / "analysis/frozen_candidate_validation_final.json"),
            "correction_sha256": core.sha(C105 / "analysis/final.json"),
            "independent_audits": ["10/10 capture", "6/6 validation final", "11/11 candidate-order correction"],
        },
        "claim_boundary": "4/4 frozen upstream barcodes replicated; corrected whole-role interventions close all four partition-by-code cells only for attribute binding and agent-patient. This is a conditional task-response mechanism, not a universal semantic code, sparse neuron set, weight mechanism or cross-model law.",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    canonical = C104 / "visualization/c104_upstream_role_barcode_heatmap.json"
    core.save(canonical, payload)
    PUBLIC.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(canonical, PUBLIC)
    checks = {
        "dimensions": payload["dimensions"] == list(range(2560)),
        "default": len(default_coordinates) == len(set(default_coordinates)) == 64,
        "effects": len(effect_rows) == 40 and all(len(row["values"]) == 2560 for row in effect_rows),
        "raw": len(raw_rows) > 400 and all(len(row["values"]) == 2560 for row in raw_rows),
        "embedding_hidden": {row["state_kind"] for row in raw_rows} == {"embedding", "hidden_state"},
        "all_states": {row["state"] for row in raw_rows if row["scope"] == "frozen_role_all_states"} == set(range(37)),
        "all_tokens": any(row["scope"] == "all_tokens_representative" and row["role"] == "unregistered" for row in raw_rows),
        "intervention": len(intervention_rows) == 16 and len(rollup_rows) == 4,
        "identity": core.sha(canonical) == core.sha(PUBLIC),
        "scope": "not model weights" in payload["coordinate_semantics"],
    }
    if not all(checks.values()):
        raise RuntimeError({"checks": checks, "raw_rows": len(raw_rows)})
    report = {
        "phase": PHASE, "campaign": CAMPAIGN, "status": "c104_upstream_role_heatmap_exported",
        "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()),
        "asset": str(canonical.relative_to(ROOT)).replace("\\", "/"), "public": str(PUBLIC.relative_to(ROOT)).replace("\\", "/"),
        "bytes": canonical.stat().st_size, "sha256": core.sha(canonical),
        "authorization": "integrate_build_and_close_c104_c105_major_stage",
    }
    core.save(C104 / "analysis/upstream_heatmap_export.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
