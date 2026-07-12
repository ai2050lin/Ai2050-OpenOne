#!/usr/bin/env python3
"""Audit adjacent files and directly test local layer-to-layer continuity."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from phase358_multiresolution_component_conservation import relative_error  # noqa: E402


PHASE371 = ROOT / "tests/gpt5/result/phase371_exact_vector_coactivity"
BASE = PHASE371 / "phase371c_internal_discovery"
ADJ = PHASE371 / "phase371c_adjacent_extension"
PROTOCOL = PHASE371 / "phase371c_adjacent_extension_protocol.json"
OUT = PHASE371 / "phase371c_adjacent_extension_audit.json"
MODELS = ("qwen3", "glm4", "deepseek7b")
CONTINUITY_GATE = 0.01


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    protocol = read_json(PROTOCOL)
    errors = []
    model_rows = []
    total_adjacent_bytes = 0
    total_base_bytes = 0
    for model in MODELS:
        base_manifest = read_json(BASE / "models" / model / "manifest.json")
        manifest = read_json(ADJ / "models" / model / "manifest.json")
        model_errors = []
        if manifest["case_count"] != 88 or manifest["file_count"] != 792:
            model_errors.append("denominator")
        if not manifest["all_numeric_gates_pass"]:
            model_errors.append("numeric_gate")
        if not manifest["all_generation_tokens_match_base"]:
            model_errors.append("generation_token_mismatch")
        for file_row in manifest["files"]:
            path = ADJ / file_row["relative_path"]
            if not path.is_file():
                model_errors.append(f"missing:{file_row['relative_path']}")
                continue
            if path.stat().st_size != int(file_row["byte_count"]):
                model_errors.append(f"size:{file_row['relative_path']}")
            if sha256_file(path) != file_row["sha256"]:
                model_errors.append(f"sha256:{file_row['relative_path']}")
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if payload["claim_boundary"]["semantic_labels_available"] is not False:
                model_errors.append(f"label:{file_row['relative_path']}")
            if not payload["quality"]["all_gates_pass"]:
                model_errors.append(f"gate:{file_row['relative_path']}")
            if "head_writes_all_receivers" in payload["attention"]:
                model_errors.append(f"materialized:{file_row['relative_path']}")
        case_ids = [row["blind_case_id"] for row in manifest["case_rows"]]
        base_anchors = list(base_manifest["anchor_layers"])
        adjacent_layers = list(manifest["selected_layers"])
        pair_definitions = [
            (base_anchors[0], adjacent_layers[0], "base_to_adjacent"),
            (base_anchors[1], adjacent_layers[1], "base_to_adjacent"),
            (adjacent_layers[2], base_anchors[2], "adjacent_to_base"),
        ]
        continuity_rows = []
        for case_id in case_ids:
            for generation_time in range(3):
                for source_layer, receiver_layer, direction in pair_definitions:
                    if direction == "base_to_adjacent":
                        source_root, receiver_root = BASE, ADJ
                    else:
                        source_root, receiver_root = ADJ, BASE
                    source = torch.load(
                        source_root / "private/models" / model / case_id / f"time_{generation_time}" / f"layer_{source_layer:03d}.pt",
                        map_location="cpu", weights_only=True,
                    )
                    receiver = torch.load(
                        receiver_root / "private/models" / model / case_id / f"time_{generation_time}" / f"layer_{receiver_layer:03d}.pt",
                        map_location="cpu", weights_only=True,
                    )
                    _, continuity_error = relative_error(
                        source["component_vectors"]["layer_output_all_positions"].float(),
                        receiver["component_vectors"]["layer_input_all_positions"].float(),
                    )
                    continuity_rows.append({
                        "case_id": case_id,
                        "generation_time": generation_time,
                        "source_layer": source_layer,
                        "receiver_layer": receiver_layer,
                        "relative_error": continuity_error,
                        "pass": continuity_error <= CONTINUITY_GATE,
                    })
        max_continuity = max(row["relative_error"] for row in continuity_rows)
        if not all(row["pass"] for row in continuity_rows):
            model_errors.append("continuity_gate")
        errors.extend(f"{model}:{error}" for error in model_errors)
        total_adjacent_bytes += int(manifest["total_byte_count"])
        total_base_bytes += int(base_manifest["total_byte_count"])
        model_rows.append({
            "model": model,
            "base_layers": base_anchors,
            "adjacent_layers": adjacent_layers,
            "case_count": manifest["case_count"],
            "file_count": manifest["file_count"],
            "byte_count": manifest["total_byte_count"],
            "continuity_row_count": len(continuity_rows),
            "max_layer_continuity_relative_error": max_continuity,
            "all_continuity_gates_pass": all(row["pass"] for row in continuity_rows),
            "contract_error_count": len(model_errors),
        })
    combined = total_base_bytes + total_adjacent_bytes
    valid = (
        protocol["authorization"]["run_adjacent_collection"]
        and not errors
        and combined <= int(protocol["storage"]["budget_bytes"])
    )
    summary = {
        "schema_version": "47.14.0",
        "phase_id": "Phase371C-Adj",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "objective": "verify_adjacent_extension_and_direct_layer_output_to_next_input_continuity",
        "valid": valid,
        "denominator": {
            "model_count": 3,
            "case_count": 264,
            "adjacent_file_count": 2376,
            "continuity_row_count": sum(row["continuity_row_count"] for row in model_rows),
        },
        "storage": {
            "base_bytes": total_base_bytes,
            "adjacent_bytes": total_adjacent_bytes,
            "combined_bytes": combined,
            "budget_bytes": int(protocol["storage"]["budget_bytes"]),
            "free_disk_bytes_after_extension": shutil.disk_usage(ROOT).free,
        },
        "models": model_rows,
        "errors": errors,
        "results": {
            "all_generation_tokens_match": all(row["contract_error_count"] == 0 for row in model_rows),
            "all_local_layer_continuity_gates_pass": all(row["all_continuity_gates_pass"] for row in model_rows),
            "local_same_graph_next_layer_state_available": valid,
            "global_all_layer_path_available": False,
            "language_mechanism_claimed": False,
        },
        "authorization": {
            "extract_lazy_exact_path_objects": valid,
            "claim_global_layer_continuity": False,
            "open_calibration": False,
            "open_physical": False,
        },
        "next_decision": "build_lazy_exact_event_graph_and_freeze_label_free_candidate_index",
    }
    OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
