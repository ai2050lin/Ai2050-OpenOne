#!/usr/bin/env python3
"""Phase1579: export all 2560 C101 activation coordinates for client heatmaps."""
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
OUT = TESTS / "result/phase1575_c101_dual_arm"
CLIENT = ROOT / "frontend/public/vis_data/research_kernel/c101_activation_coordinate_heatmap.json"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core
import phase1571_c098_observation_first_graph_campaign as graph_base
import phase1577_c101_dual_arm_analysis as analysis

STATES = (0, 16, 24, 31, 32, 36)
COEFF_STATES = (24, 31, 32)


def mean_coeff(coeff: np.ndarray, units: list[dict[str, Any]], effect: int, state: int, role: int, **filters: str) -> np.ndarray:
    selected = [row["row_index"] for row in units if all(row[key] == value for key, value in filters.items())]
    if not selected:
        raise RuntimeError(filters)
    return np.asarray(coeff[selected, effect, state, role], dtype=np.float64).mean(axis=0)


def compiled_map() -> dict[str, dict[str, Any]]:
    rows = [*core.rows(OUT / "compiled/qwen3_confirmation.jsonl"), *core.rows(OUT / "compiled/qwen3_breadth.jsonl")]
    return {row["case_id"]: row for row in rows}


def main() -> None:
    if (OUT / "visualization/c101_activation_coordinate_heatmap.json").exists():
        raise RuntimeError("C101 heatmap already exists")
    final = core.load(OUT / "analysis/final.json")
    audit = core.load(OUT / "audit/independent_final_audit.json")
    if final["result"]["authorization"] != "export_c101_parameter_level_heatmap" or not audit["all_checks_passed"]:
        raise RuntimeError("heatmap export unauthorized")
    conf_coeff = np.load(OUT / "raw/qwen3_confirmation_walsh_coefficients_v2.float32.npy", mmap_mode="r")
    breadth_coeff = np.load(OUT / "raw/qwen3_breadth_walsh_coefficients_v2.float32.npy", mmap_mode="r")
    conf_units = core.rows(OUT / "raw/qwen3_confirmation_walsh_index_v2.jsonl")
    breadth_units = core.rows(OUT / "raw/qwen3_breadth_walsh_index_v2.jsonl")
    raw = np.load(OUT / "raw/qwen3_registered_role_field.float16.npy", mmap_mode="r")
    raw_index = core.rows(OUT / "raw/qwen3_registered_role_index.jsonl")
    by_case = {row["case_id"]: row for row in raw_index}
    compiled = compiled_map()
    tok = graph_base.tokenizer()
    walsh_rows = []
    xy = analysis.GRAPH_EFFECTS.index("xy")
    boundary_conf = analysis.CONF_ROLES.index("boundary")
    for partition in graph_base.PARTITIONS:
        for world in graph_base.WORLDS:
            for family in graph_base.FAMILIES:
                for state in COEFF_STATES:
                    vector = mean_coeff(conf_coeff, conf_units, xy, state, boundary_conf, partition=partition, world=world, family=family)
                    walsh_rows.append({
                        "arm": "confirmation", "partition": partition, "world": world, "family": family,
                        "effect": "xy", "role": "boundary", "state": state,
                        "state_kind": "hidden_state", "values": vector.astype(np.float32).tolist(),
                    })
    truth = analysis.BREADTH_EFFECTS.index("truth")
    boundary_breadth = analysis.BREADTH_ROLES.index("boundary")
    for partition in graph_base.PARTITIONS:
        for family in sorted({row["family"] for row in breadth_units}):
            for state in COEFF_STATES:
                vector = mean_coeff(breadth_coeff, breadth_units, truth, state, boundary_breadth, partition=partition, family=family)
                walsh_rows.append({
                    "arm": "breadth", "partition": partition, "world": "controlled_natural", "family": family,
                    "effect": "truth", "role": "boundary", "state": state,
                    "state_kind": "hidden_state", "values": vector.astype(np.float32).tolist(),
                })
    example_ids = [row["case_id"] for row in core.rows(OUT / "material/frozen_test_examples.jsonl")]
    raw_rows = []
    for case_id in example_ids:
        meta = by_case[case_id]
        source = compiled[case_id]
        roles = ("target_pre", "target_post", "query_target", "boundary") if meta["arm"] == "confirmation" else ("focus_pre", "focus_record", "query_focus", "boundary")
        for role in roles:
            left, right = meta["role_offsets"][role]
            prompt_positions = source["role_positions"][role]
            for subtoken, (raw_position, prompt_position) in enumerate(zip(range(left, right), prompt_positions, strict=True)):
                token_id = int(source["prompt_ids"][prompt_position])
                token_text = tok.decode([token_id], skip_special_tokens=False)
                for state in STATES:
                    raw_rows.append({
                        "case_id": case_id,
                        "arm": meta["arm"],
                        "family": meta["family"],
                        "world": meta["world"],
                        "partition": meta["partition"],
                        "role": role,
                        "subtoken": subtoken,
                        "token_id": token_id,
                        "token_text": token_text,
                        "state": state,
                        "state_kind": "embedding" if state == 0 else "hidden_state",
                        "values": np.asarray(raw[state, raw_position], dtype=np.float32).tolist(),
                    })
    key_vectors = np.stack([np.abs(np.asarray(row["values"], dtype=np.float64)) for row in walsh_rows if row["state"] == 24], axis=0)
    score = np.mean(key_vectors, axis=0)
    default_coordinates = [int(v) for v in np.argsort(score)[-64:][::-1]]
    all_abs = np.concatenate([np.abs(np.asarray(row["values"], dtype=np.float32)) for row in walsh_rows])
    scale = float(np.quantile(all_abs, 0.99))
    payload = {
        "schema": "c101_activation_coordinate_heatmap.v1",
        "result_type": "activation_coordinate_heatmap",
        "phase": 1579,
        "campaign": "C101",
        "model": "Qwen3-4B",
        "title": "C101 Embedding and Hidden-State Activation Coordinates",
        "coordinate_semantics": "H[layer, registered token, coordinate]; these are activation coordinates, not weight parameters or identified neurons",
        "dimensions": list(range(2560)),
        "default_coordinates": default_coordinates,
        "scale": {"symmetric_abs_q99": scale},
        "walsh_rows": walsh_rows,
        "raw_rows": raw_rows,
        "source": {
            "field_sha256": core.load(OUT / "analysis/qwen_capture_summary.json")["raw_sha256"],
            "confirmation_coeff_sha256": final["result"]["coefficients"]["confirmation"]["sha256"],
            "breadth_coeff_sha256": final["result"]["coefficients"]["breadth"]["sha256"],
            "analysis_audit": "6/6",
        },
        "claim_boundary": final["result"]["claim_boundary"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    canonical = OUT / "visualization/c101_activation_coordinate_heatmap.json"
    core.save(canonical, payload)
    CLIENT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(canonical, CLIENT)
    checks = {
        "dimensions": len(payload["dimensions"]) == 2560,
        "default": len(payload["default_coordinates"]) == 64 and len(set(payload["default_coordinates"])) == 64,
        "walsh": len(walsh_rows) == 144 and all(len(row["values"]) == 2560 for row in walsh_rows),
        "raw": len(raw_rows) > 100 and all(len(row["values"]) == 2560 for row in raw_rows),
        "embedding": any(row["state_kind"] == "embedding" for row in raw_rows),
        "hidden": any(row["state_kind"] == "hidden_state" for row in raw_rows),
        "identity": core.sha(canonical) == core.sha(CLIENT),
        "scope": "not weight parameters" in payload["coordinate_semantics"],
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    report = {"phase": 1579, "campaign": "C101", "status": "activation_coordinate_heatmap_exported", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "asset": str(canonical.relative_to(ROOT)), "client": str(CLIENT.relative_to(ROOT)), "bytes": canonical.stat().st_size, "sha256": core.sha(canonical), "authorization": "integrate_c101_activation_coordinate_heatmap_client"}
    core.save(OUT / "analysis/visualization_export.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
