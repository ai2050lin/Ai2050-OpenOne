#!/usr/bin/env python3
"""Phase1586 / C102: export full-coordinate barcode and token/state heatmap data."""
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
OUT = TESTS / "result/phase1581_c102_typed_relation_coordinate_campaign"
PUBLIC = ROOT / "frontend/public/vis_data/research_kernel/c102_coordinate_barcode_heatmap.json"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core
import phase1577_c101_dual_arm_analysis as c101_analysis

GRAPH_FAMILIES = ("taxonomy", "containment", "comparison", "precedence")
BREADTH_FAMILIES = ("attribute_binding", "agent_patient", "negation_scope", "whole_part_exception")
EFFECTS = ("primary", "code", "primary_x_code")


def decode_bf16(bits: np.ndarray) -> np.ndarray:
    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def family_selector(family: str) -> dict[str, Any]:
    frozen = core.load(OUT / "protocol/frozen_coordinate_barcode_predictions.json")
    return next(row for row in frozen["selectors"] if row["family"] == family)


def partition_vector(coeff: np.ndarray, units: list[dict[str, Any]], family: str, partition: str, effect: int, state: int, role: int) -> np.ndarray:
    selected = [row["row_index"] for row in units if row["family"] == family and row["partition"] == partition]
    return np.asarray(coeff[selected, effect, state, role], dtype=np.float64).mean(axis=0).astype(np.float32)


def main() -> None:
    intervention = core.load(OUT / "analysis/coordinate_coalition_intervention_final.json")
    audit = core.load(OUT / "audit/independent_coordinate_intervention_audit.json")
    if intervention["authorization"] != "export_c102_coordinate_and_token_heatmap" or not audit["all_checks_passed"]:
        raise RuntimeError("C102 heatmap export not authorized")
    selection = core.load(OUT / "protocol/response_discovery_selection.json")["selection"]
    frozen = core.load(OUT / "protocol/frozen_coordinate_barcode_predictions.json")
    graph_source = np.load(ROOT / frozen["barcodes"]["graph_path"], mmap_mode="r")
    breadth_source = np.load(ROOT / frozen["barcodes"]["breadth_path"], mmap_mode="r")
    graph_coeff = np.load(OUT / "raw/qwen3_graph_three_effect_coefficients.float32.npy", mmap_mode="r")
    breadth_coeff = np.load(OUT / "raw/qwen3_breadth_three_effect_coefficients.float32.npy", mmap_mode="r")
    graph_units = core.rows(OUT / "raw/qwen3_graph_three_effect_index.jsonl")
    breadth_units = core.rows(OUT / "raw/qwen3_breadth_three_effect_index.jsonl")
    effect_rows = []
    for family in (*GRAPH_FAMILIES, *BREADTH_FAMILIES):
        arm = "graph" if family in GRAPH_FAMILIES else "breadth"
        families = GRAPH_FAMILIES if arm == "graph" else BREADTH_FAMILIES
        source = graph_source if arm == "graph" else breadth_source
        coeff = graph_coeff if arm == "graph" else breadth_coeff
        units = graph_units if arm == "graph" else breadth_units
        selector = family_selector(family)["selector"]
        family_index = families.index(family)
        for effect_index, effect in enumerate(EFFECTS):
            effect_rows.append({"dataset": "c101_discovery", "partition": "revealed_source", "arm": arm, "family": family, "effect": effect, "role": selector["role"], "state": selector["state"], "values": np.asarray(source[family_index, effect_index], dtype=np.float32).tolist()})
            for partition in ("confirmation", "lockbox"):
                vector = partition_vector(coeff, units, family, partition, effect_index, selector["state"], selector["role_index"])
                effect_rows.append({"dataset": "c102_fresh", "partition": partition, "arm": arm, "family": family, "effect": effect, "role": selector["role"], "state": selector["state"], "values": vector.tolist()})

    raw_field = np.load(OUT / "raw/qwen3_all_token_state_coordinate_field.uint16.npy", mmap_mode="r")
    raw_index = core.rows(OUT / "raw/qwen3_all_token_state_coordinate_index.jsonl")
    representatives = {}
    for arm, family in (("graph", "taxonomy"), ("breadth", "attribute_binding")):
        representatives[arm] = next(row for row in raw_index if row["arm"] == arm and row["family"] == family and row["partition"] == "lockbox")
    raw_rows = []
    for arm, row in representatives.items():
        selected_state = int(selection[row["family"]]["state"])
        for state in (0, selected_state):
            for local_position, (token_id, token_text) in enumerate(zip(row["token_ids"], row["token_texts"], strict=True)):
                raw_rows.append({"scope": "all_tokens_representative", "case_id": row["case_id"], "arm": arm, "family": row["family"], "partition": row["partition"], "token_position": local_position, "token_id": token_id, "token_text": token_text, "role": next((role for role, positions in row["role_positions"].items() if local_position in positions), "unregistered"), "state": state, "state_kind": "embedding" if state == 0 else "hidden_state", "values": decode_bf16(raw_field[state, row["token_start"] + local_position]).tolist()})
    for family in (*GRAPH_FAMILIES, *BREADTH_FAMILIES):
        row = next(item for item in raw_index if item["family"] == family and item["partition"] == "lockbox")
        boundary = int(row["role_positions"]["boundary"][0])
        for state in range(37):
            raw_rows.append({"scope": "all_states_boundary", "case_id": row["case_id"], "arm": row["arm"], "family": family, "partition": row["partition"], "token_position": boundary, "token_id": row["token_ids"][boundary], "token_text": row["token_texts"][boundary], "role": "boundary", "state": state, "state_kind": "embedding" if state == 0 else "hidden_state", "values": decode_bf16(raw_field[state, row["token_start"] + boundary]).tolist()})

    formation_rows = core.rows(OUT / "analysis/formation_trajectory_validation.jsonl")
    intervention_rows = core.rows(OUT / "analysis/coordinate_coalition_intervention_summary.jsonl")
    lockbox_primary = [row for row in effect_rows if row["dataset"] == "c102_fresh" and row["partition"] == "lockbox" and row["effect"] == "primary"]
    mean_abs = np.mean(np.stack([np.abs(np.asarray(row["values"], dtype=np.float64)) for row in lockbox_primary]), axis=0)
    default_coordinates = np.argsort(-mean_abs, kind="stable")[:64].astype(int).tolist()
    effect_abs = np.concatenate([np.abs(np.asarray(row["values"], dtype=np.float32)) for row in effect_rows])
    payload = {
        "schema": "c102_coordinate_barcode_heatmap.v1",
        "result_type": "coordinate_barcode_heatmap",
        "phase": 1586,
        "campaign": "C102",
        "model": "Qwen3-4B",
        "title": "C102 Full-Coordinate Task-Response Barcode Field",
        "coordinate_semantics": "H[state, token, activation_coordinate]; the 2560 values are residual-stream activation coordinates, not weight parameters, MLP neurons or attention heads",
        "dimensions": list(range(2560)),
        "default_coordinates": default_coordinates,
        "scale": {"effect_symmetric_abs_q99": float(np.quantile(effect_abs, 0.99))},
        "effect_rows": effect_rows,
        "raw_rows": raw_rows,
        "formation_rows": formation_rows,
        "intervention_rows": intervention_rows,
        "headline": {"barcode_three_stage_passed": 8, "barcode_total": 8, "controlled_intervention_passed": intervention["important_count"], "controlled_intervention_total": 8, "behavior_accuracy": core.load(OUT / "analysis/qwen_full_field_capture_summary.json")["behavior"]["global_accuracy"]},
        "source": {"raw_sha256": core.load(OUT / "analysis/qwen_full_field_capture_summary.json")["raw_sha256"], "analysis_sha256": core.sha(OUT / "analysis/staged_barcode_final.json"), "intervention_sha256": intervention["results_sha256"], "independent_audits": ["10/10 capture", "6/6 staged final", "10/10 intervention"]},
        "claim_boundary": "8/8 frozen barcodes repeated, but 0/8 controlled interventions closed across confirmation and lockbox; this is a distributed late task/output response regularity, not a semantic code or sparse-neuron mechanism",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    canonical = OUT / "visualization/c102_coordinate_barcode_heatmap.json"
    core.save(canonical, payload)
    PUBLIC.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(canonical, PUBLIC)
    checks = {
        "dimensions": payload["dimensions"] == list(range(2560)),
        "default": len(default_coordinates) == len(set(default_coordinates)) == 64,
        "effects": len(effect_rows) == 72 and all(len(row["values"]) == 2560 for row in effect_rows),
        "raw": len(raw_rows) > 500 and all(len(row["values"]) == 2560 for row in raw_rows),
        "embedding_hidden": {row["state_kind"] for row in raw_rows} == {"embedding", "hidden_state"},
        "all_token": any(row["scope"] == "all_tokens_representative" and row["role"] == "unregistered" for row in raw_rows),
        "all_state": {row["state"] for row in raw_rows if row["scope"] == "all_states_boundary"} == set(range(37)),
        "identity": core.sha(canonical) == core.sha(PUBLIC),
        "scope": "not weight parameters" in payload["coordinate_semantics"],
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    report = {"phase": 1586, "campaign": "C102", "status": "coordinate_barcode_heatmap_exported", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "asset": str(canonical.relative_to(ROOT)).replace("\\", "/"), "public": str(PUBLIC.relative_to(ROOT)).replace("\\", "/"), "bytes": canonical.stat().st_size, "sha256": core.sha(canonical), "authorization": "integrate_and_build_c102_heatmap_client"}
    core.save(OUT / "analysis/heatmap_export.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
