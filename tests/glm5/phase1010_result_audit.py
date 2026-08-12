#!/usr/bin/env python3
"""Audit Phase1010 protocol, behavior, scan, analysis, and causal outputs."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from phase1010_output_type_protocol import (
    ANALYSIS_OPERATIONS,
    FAMILIES,
    MODELS,
    OUT_ROOT,
    OUTPUT_TYPES,
    PHASE,
    read_json,
    read_jsonl,
    write_json,
)


EXPECTED_CASES_PER_MODEL = 4608
EXPECTED_UNITS_PER_MODEL = 576
EXPECTED_UNITS_PER_PANEL = 48
OP_INDEX = {name: index for index, name in enumerate(ANALYSIS_OPERATIONS)}
FROZEN_GLM4_HEADS = [18, 28, 26]


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def assert_finite(name: str, values: np.ndarray) -> None:
    if not np.all(np.isfinite(values)):
        count = int(np.sum(~np.isfinite(values)))
        raise RuntimeError(f"{name}: {count} non-finite values")


def protocol_audit(model_name: str) -> dict[str, Any]:
    root = OUT_ROOT / "protocol" / model_name
    cases = read_jsonl(root / "cases.jsonl")
    units = read_jsonl(root / "units.jsonl")
    if len(cases) != EXPECTED_CASES_PER_MODEL:
        raise RuntimeError(f"{model_name}: protocol case count drift")
    if len(units) != EXPECTED_UNITS_PER_MODEL:
        raise RuntimeError(f"{model_name}: protocol unit count drift")
    record_ids = [row["record_id"] for row in cases]
    unit_ids = [row["unit_id"] for row in units]
    if len(set(record_ids)) != len(record_ids):
        raise RuntimeError(f"{model_name}: duplicate record id")
    if len(set(unit_ids)) != len(unit_ids):
        raise RuntimeError(f"{model_name}: duplicate unit id")
    token_sets: dict[str, set[int]] = {}
    for output_type in OUTPUT_TYPES:
        values = {
            int(token_id)
            for case in cases
            if case["output_type"] == output_type
            for token_id in case["candidate_label_ids"].values()
        }
        token_sets[output_type] = values
    intersections = {}
    for left_index, left in enumerate(OUTPUT_TYPES):
        for right in OUTPUT_TYPES[left_index + 1 :]:
            overlap = sorted(token_sets[left] & token_sets[right])
            intersections[f"{left}:{right}"] = overlap
            if overlap:
                raise RuntimeError(
                    f"{model_name}: output token sets overlap "
                    f"{left}/{right}: {overlap}"
                )
    return {
        "case_count": len(cases),
        "unit_count": len(units),
        "output_token_set_sizes": {
            key: len(value) for key, value in token_sets.items()
        },
        "output_token_set_intersections": intersections,
    }


def behavior_audit(model_name: str) -> dict[str, Any]:
    root = OUT_ROOT / "behavior" / model_name
    rows = read_jsonl(root / "rows.jsonl")
    pairs = read_jsonl(root / "pair_qualification.jsonl")
    summary = read_json(root / "summary.json")
    if len(rows) != EXPECTED_CASES_PER_MODEL:
        raise RuntimeError(f"{model_name}: behavior coverage drift")
    if len({row["record_id"] for row in rows}) != len(rows):
        raise RuntimeError(f"{model_name}: duplicate behavior row")
    expected_pairs = EXPECTED_UNITS_PER_MODEL * 8
    if len(pairs) != expected_pairs:
        raise RuntimeError(
            f"{model_name}: pair coverage {len(pairs)} != {expected_pairs}"
        )
    if not 0 <= summary["overall_semantic_panel_rate"] <= 1:
        raise RuntimeError(f"{model_name}: invalid behavior rate")
    return {
        "case_count": len(rows),
        "pair_count": len(pairs),
        "semantic_panel_rate": summary[
            "overall_semantic_panel_rate"
        ],
        "strict_teacher_rate": summary[
            "overall_strict_teacher_rate"
        ],
        "rollout_rate": summary["overall_rollout_case_rate"],
    }


def scan_audit(model_name: str) -> dict[str, Any]:
    model_root = OUT_ROOT / "scan" / model_name
    summary = read_json(model_root / "summary.json")
    if summary["unit_count"] != EXPECTED_UNITS_PER_MODEL:
        raise RuntimeError(f"{model_name}: scan unit count drift")
    scalar_count = 0
    centroid_count = 0
    identity_maximum = 0.0
    panel_count = 0
    for output_type in OUTPUT_TYPES:
        for family in FAMILIES:
            panel_count += 1
            panel_root = model_root / output_type / family
            events = read_jsonl(panel_root / "events.jsonl")
            units = read_jsonl(panel_root / "units.jsonl")
            panel_summary = read_json(panel_root / "summary.json")
            if len(units) != EXPECTED_UNITS_PER_PANEL:
                raise RuntimeError(
                    f"{model_name}/{output_type}/{family}: "
                    "panel unit count drift"
                )
            with np.load(panel_root / "response_scalars.npz") as payload:
                raw = payload["raw_magnitude"]
                normalized = payload["normalized_magnitude"]
                semantic = payload["semantic_qualified"]
                strict = payload["strict_qualified"]
                rollout = payload["rollout_qualified"]
            expected_shape = (
                len(units),
                len(ANALYSIS_OPERATIONS),
                len(events),
            )
            if raw.shape != expected_shape:
                raise RuntimeError(
                    f"{model_name}/{output_type}/{family}: "
                    f"raw shape {raw.shape} != {expected_shape}"
                )
            if normalized.shape != expected_shape:
                raise RuntimeError("normalized shape drift")
            if semantic.shape != expected_shape[:2]:
                raise RuntimeError("qualification shape drift")
            if strict.shape != semantic.shape or rollout.shape != semantic.shape:
                raise RuntimeError("qualification mask shape mismatch")
            assert_finite("raw_magnitude", raw)
            assert_finite("normalized_magnitude", normalized)
            identity = normalized[:, OP_INDEX["I"], :]
            panel_identity_max = float(np.max(np.abs(identity)))
            identity_maximum = max(identity_maximum, panel_identity_max)
            if panel_identity_max > 1e-12:
                raise RuntimeError(
                    f"{model_name}/{output_type}/{family}: "
                    f"identity floor {panel_identity_max}"
                )
            with np.load(
                panel_root / "direction_consistency.npz"
            ) as payload:
                consistency = payload["direction_consistency"]
                counts = payload["direction_count"]
            if consistency.shape != (
                len(ANALYSIS_OPERATIONS),
                2,
                len(events),
            ):
                raise RuntimeError("direction consistency shape drift")
            if counts.shape != consistency.shape:
                raise RuntimeError("direction count shape drift")
            finite_consistency = consistency[np.isfinite(consistency)]
            if finite_consistency.size and (
                np.min(finite_consistency) < -1.00001
                or np.max(finite_consistency) > 1.00001
            ):
                raise RuntimeError("direction consistency out of range")
            metadata = read_jsonl(
                panel_root / "peak_direction_metadata.jsonl"
            )
            with np.load(
                panel_root / "peak_direction_centroids.npz"
            ) as payload:
                centroids = payload["centroids"]
            if len(metadata) != centroids.shape[0]:
                raise RuntimeError("centroid metadata coverage drift")
            if centroids.shape[1] != summary["model_info"]["d_model"]:
                raise RuntimeError("centroid d_model drift")
            if centroids.size:
                assert_finite("centroids", centroids)
                norms = np.linalg.norm(
                    centroids.astype(np.float32),
                    axis=1,
                )
                if float(np.max(np.abs(norms - 1.0))) > 5e-3:
                    raise RuntimeError("centroid normalization drift")
            if panel_summary["raw_hidden_tensors_persisted"] != 0:
                raise RuntimeError("raw hidden persistence policy failed")
            scalar_count += panel_summary["scalar_measurement_count"]
            centroid_count += len(metadata)
    if scalar_count != summary["scalar_measurement_count"]:
        raise RuntimeError(f"{model_name}: scalar count summary drift")
    if centroid_count != summary["peak_direction_centroid_count"]:
        raise RuntimeError(f"{model_name}: centroid count summary drift")
    return {
        "panel_count": panel_count,
        "unit_count": summary["unit_count"],
        "scalar_measurement_count": scalar_count,
        "peak_direction_centroid_count": centroid_count,
        "identity_maximum": identity_maximum,
        "raw_hidden_tensors_persisted": 0,
    }


def causal_audit(model_name: str) -> dict[str, Any]:
    root = OUT_ROOT / "causal_screen" / model_name
    summary = read_json(root / "summary.json")
    cells = read_jsonl(root / "cell_summaries.jsonl")
    units = read_jsonl(root / "units.jsonl")
    if len(cells) != 24:
        raise RuntimeError(f"{model_name}: causal cell count drift")
    if summary["selection_used_phase1010_data"]:
        raise RuntimeError(f"{model_name}: causal selection leakage")
    if not summary["no_op_audit_pass"]:
        raise RuntimeError(f"{model_name}: no-op audit failed")
    if len(units) != summary["unit_operation_count"]:
        raise RuntimeError(f"{model_name}: causal unit coverage drift")
    return {
        "cell_count": len(cells),
        "measured_cell_count": summary["measured_cell_count"],
        "underpowered_cell_count": summary[
            "underpowered_cell_count"
        ],
        "positive_cell_count": summary["positive_cell_count"],
        "output_general_operations_by_frozen_gate": summary[
            "output_general_operations_by_frozen_gate"
        ],
        "person_specific_operations_by_frozen_gate": summary[
            "person_specific_operations_by_frozen_gate"
        ],
        "no_op_audit_pass": True,
    }


def rollout_surface_audit() -> dict[str, Any]:
    root = OUT_ROOT / "behavior"
    combined = read_json(root / "rollout_surface_summary.json")
    summaries = {
        row["model"]: row for row in combined["models"]
    }
    if set(summaries) != set(MODELS):
        raise RuntimeError("rollout surface model coverage drift")
    results = {}
    for model_name in MODELS:
        summary = read_json(
            root / model_name / "rollout_surface_summary.json"
        )
        rows = read_jsonl(
            root / model_name / "rollout_surface_rows.jsonl"
        )
        if summary != summaries[model_name]:
            raise RuntimeError(
                f"{model_name}: rollout surface combined summary drift"
            )
        if len(rows) != EXPECTED_CASES_PER_MODEL:
            raise RuntimeError(
                f"{model_name}: rollout surface case coverage drift"
            )
        for field in (
            "label_case_insensitive_rate",
            "flexible_full_protocol_rate",
            "frozen_strict_exact_rate",
        ):
            value = float(summary[field])
            if not 0.0 <= value <= 1.0:
                raise RuntimeError(
                    f"{model_name}: invalid rollout field {field}"
                )
        results[model_name] = {
            "case_count": len(rows),
            "flexible_full_protocol_rate": summary[
                "flexible_full_protocol_rate"
            ],
            "frozen_strict_exact_rate": summary[
                "frozen_strict_exact_rate"
            ],
        }
    return {
        "strict_exact_is_not_semantic_accuracy": combined[
            "strict_exact_is_not_semantic_accuracy"
        ],
        "models": results,
    }


def source_role_audit() -> dict[str, Any]:
    root = OUT_ROOT / "source_role_mapping"
    discovery = read_json(root / "discovery" / "summary.json")
    confirmation = read_json(
        root / "bf16_confirmation" / "summary.json"
    )
    if discovery["selected_heads"] != FROZEN_GLM4_HEADS:
        raise RuntimeError("source-role discovery head drift")
    if confirmation["selected_heads"] != FROZEN_GLM4_HEADS:
        raise RuntimeError("source-role confirmation head drift")
    if discovery["maximum_noop_logit_error"] != 0.0:
        raise RuntimeError("source-role discovery no-op failed")
    if confirmation["maximum_noop_logit_error"] != 0.0:
        raise RuntimeError("source-role confirmation no-op failed")
    if not confirmation["response_map_instruction_repeats"]:
        raise RuntimeError("source-role frozen confirmation failed")
    discovery_rows = read_jsonl(
        root / "discovery" / "units.jsonl"
    )
    confirmation_rows = read_jsonl(
        root / "bf16_confirmation" / "units.jsonl"
    )
    if len(discovery_rows) != discovery["unit_role_row_count"]:
        raise RuntimeError("source-role discovery row coverage drift")
    if not confirmation_rows:
        raise RuntimeError("source-role confirmation rows missing")
    return {
        "discovery_row_count": len(discovery_rows),
        "confirmation_row_count": len(confirmation_rows),
        "selected_atomic_roles": discovery[
            "selected_atomic_roles"
        ],
        "response_map_instruction_repeats": True,
        "maximum_attention_reconstruction_error": max(
            discovery["maximum_attention_reconstruction_error"],
            confirmation["maximum_attention_reconstruction_error"],
        ),
        "maximum_noop_logit_error": 0.0,
    }


def relay_depth_audit() -> dict[str, Any]:
    root = OUT_ROOT / "relay_depth_mapping"
    discovery = read_json(root / "discovery" / "summary.json")
    confirmation = read_json(
        root / "bf16_confirmation" / "summary.json"
    )
    if discovery["selected_heads"] != FROZEN_GLM4_HEADS:
        raise RuntimeError("relay-depth discovery head drift")
    if confirmation["selected_heads"] != FROZEN_GLM4_HEADS:
        raise RuntimeError("relay-depth confirmation head drift")
    if discovery["maximum_noop_logit_error"] != 0.0:
        raise RuntimeError("relay-depth discovery no-op failed")
    if confirmation["maximum_noop_logit_error"] != 0.0:
        raise RuntimeError("relay-depth confirmation no-op failed")
    if not confirmation["l0_negative_control_pass"]:
        raise RuntimeError("relay-depth L0 negative control failed")
    if not set(confirmation["confirmed_depths"]).issubset(
        set(discovery["confirmation_depths"])
    ):
        raise RuntimeError("relay-depth confirmation selection drift")
    cells = confirmation["cell_summaries"]
    if len(cells) != 6 or any(int(row["n"]) != 8 for row in cells):
        raise RuntimeError("relay-depth BF16 cell coverage drift")
    return {
        "discovery_depths": discovery["depths"],
        "confirmation_depths": discovery["confirmation_depths"],
        "confirmed_depths": confirmation["confirmed_depths"],
        "l0_negative_control_pass": True,
        "maximum_noop_logit_error": 0.0,
    }


def relay_subregion_audit() -> dict[str, Any]:
    root = OUT_ROOT / "relay_subregion_mapping" / "bf16"
    summary = read_json(root / "summary.json")
    cells = read_jsonl(root / "cell_summaries.jsonl")
    if len(cells) != 8:
        raise RuntimeError("relay-subregion cell coverage drift")
    if summary["maximum_noop_logit_error"] != 0.0:
        raise RuntimeError("relay-subregion no-op failed")
    expected_regions = set(
        summary["subregions_frozen_from_prompt_construction"]
    )
    if {row["subregion"] for row in cells} != expected_regions:
        raise RuntimeError("relay-subregion region coverage drift")
    if any(int(row["n"]) != 8 for row in cells):
        raise RuntimeError("relay-subregion unit coverage drift")
    return {
        "depth": summary["depth"],
        "regions": sorted(expected_regions),
        "maximum_noop_logit_error": 0.0,
    }


def qkv_audit() -> dict[str, Any]:
    root = OUT_ROOT / "map_qkv_decomposition" / "bf16"
    summary = read_json(root / "summary.json")
    cells = read_jsonl(root / "cell_summaries.jsonl")
    if summary["selected_heads"] != FROZEN_GLM4_HEADS:
        raise RuntimeError("QKV decomposition head drift")
    if not summary["identity_is_measurement_not_language_formula"]:
        raise RuntimeError("QKV identity claim-limit flag missing")
    if summary["maximum_qkv_identity_error"] > 1e-4:
        raise RuntimeError("QKV decomposition identity error")
    expected = {
        (family, component)
        for family in ("negation", "semantic_role")
        for component in (
            "qk_routing",
            "value_content",
            "interaction",
            "all",
        )
    }
    observed = {
        (row["family"], row["component"]) for row in cells
    }
    if observed != expected or any(int(row["n"]) != 8 for row in cells):
        raise RuntimeError("QKV decomposition coverage drift")
    return {
        "source_region": summary["source_region"],
        "component_identity": summary["component_identity"],
        "maximum_qkv_identity_error": summary[
            "maximum_qkv_identity_error"
        ],
    }


def map_entry_qk_audit() -> dict[str, Any]:
    root = OUT_ROOT / "map_entry_qk" / "bf16"
    summary = read_json(root / "summary.json")
    cells = read_jsonl(root / "cell_summaries.jsonl")
    if summary["selected_heads"] != FROZEN_GLM4_HEADS:
        raise RuntimeError("map-entry QK head drift")
    expected = {
        (family, group)
        for family in ("negation", "semantic_role")
        for group in summary["semantic_entry_groups"]
    }
    observed = {
        (row["family"], row["entry_group"]) for row in cells
    }
    if observed != expected or any(int(row["n"]) != 8 for row in cells):
        raise RuntimeError("map-entry QK coverage drift")
    finite_fields = (
        "median_component_to_all_map_qk_norm_ratio",
        "median_sufficiency_fraction",
        "median_restore_fraction",
        "median_shuffled_sufficiency_fraction",
    )
    for row in cells:
        values = np.asarray(
            [row[field] for field in finite_fields],
            dtype=np.float64,
        )
        assert_finite("map-entry QK cell", values)
    return {
        "component": summary["component"],
        "semantic_entry_groups": summary[
            "semantic_entry_groups"
        ],
        "cell_count": len(cells),
    }


def inventory() -> list[dict[str, Any]]:
    audit_root = OUT_ROOT / "audit"
    rows = []
    for path in sorted(OUT_ROOT.rglob("*")):
        if not path.is_file():
            continue
        if audit_root in path.parents:
            continue
        rows.append({
            "path": path.relative_to(OUT_ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    return rows


def main() -> None:
    protocol = read_json(OUT_ROOT / "protocol" / "protocol.json")
    protocol_results = {
        model: protocol_audit(model) for model in MODELS
    }
    behavior_results = {
        model: behavior_audit(model) for model in MODELS
    }
    scan_results = {model: scan_audit(model) for model in MODELS}
    analysis = read_json(OUT_ROOT / "analysis" / "summary.json")
    causal_results = {
        model: causal_audit(model)
        for model in ("qwen3", "glm4")
    }
    precision = read_json(
        OUT_ROOT
        / "precision_audit"
        / "glm4_bf16"
        / "summary.json"
    )
    rollout_surface = rollout_surface_audit()
    source_role = source_role_audit()
    relay_depth = relay_depth_audit()
    relay_subregion = relay_subregion_audit()
    qkv = qkv_audit()
    map_entry_qk = map_entry_qk_audit()
    files = inventory()
    result = {
        "schema_version": "phase1010_result_audit.v2",
        "phase": PHASE,
        "protocol_digest": protocol["preregistration_digest"],
        "protocol": protocol_results,
        "behavior": behavior_results,
        "scan": scan_results,
        "analysis": analysis,
        "causal": causal_results,
        "rollout_surface": rollout_surface,
        "glm4_bf16_precision": {
            "entered_cell_count": precision["entered_cell_count"],
            "no_op_audit_pass": precision["no_op_audit_pass"],
            "positive_nonperson_cell_ids": precision[
                "positive_nonperson_cell_ids"
            ],
            "upstream_source_mapping_authorized": precision[
                "upstream_source_mapping_authorized"
            ],
        },
        "glm4_source_role_mapping": source_role,
        "glm4_relay_depth_mapping": relay_depth,
        "glm4_relay_subregion_mapping": relay_subregion,
        "glm4_map_qkv_decomposition": qkv,
        "glm4_map_entry_qk": map_entry_qk,
        "file_count": len(files),
        "total_bytes": int(sum(row["bytes"] for row in files)),
        "all_required_checks_pass": True,
    }
    audit_root = OUT_ROOT / "audit"
    audit_root.mkdir(parents=True, exist_ok=True)
    with (audit_root / "inventory.jsonl").open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        for row in files:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
    write_json(audit_root / "summary.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
