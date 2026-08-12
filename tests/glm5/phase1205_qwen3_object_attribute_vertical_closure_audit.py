#!/usr/bin/env python3
"""Independent preexecution and result audit for Phase1205.

The result audit recomputes the frozen generation-boundary residual gate from
the compressed arrays.  It deliberately does not import the Phase1205 driver,
so the scientific verdict is not obtained by calling the implementation under
test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests/glm5"
PHASE = 1205
OUT_ROOT = ROOT / "tests/glm5/result/phase1205_qwen3_object_attribute_vertical_closure"
PROTOCOL_PATH = OUT_ROOT / "protocol/preregistration.json"
PAIR_MANIFEST_PATH = OUT_ROOT / "protocol/pair_manifest.jsonl"
PREAUDIT_PATH = OUT_ROOT / "audit/preexecution_audit.json"
ARRAY_PATH = OUT_ROOT / "runs/hidden_response_arrays.npz"
RUN_SUMMARY_PATH = OUT_ROOT / "runs/run_summary.json"
VERDICT_PATH = OUT_ROOT / "analysis/hidden_specificity_verdict.json"
TRAJECTORY_PATH = OUT_ROOT / "analysis/role_component_trajectories.json"
RESULT_AUDIT_PATH = OUT_ROOT / "audit/independent_result_audit.json"
FINAL_PATH = OUT_ROOT / "analysis/final.json"

MAIN_SCRIPT = TEST_ROOT / "phase1205_qwen3_object_attribute_vertical_closure.py"
AUDIT_SCRIPT = Path(__file__).resolve()
RUNNER_SCRIPT = TEST_ROOT / "phase1205_run_sequential.py"
SOURCE1202 = ROOT / "tests/glm5/result/phase1202_object_attribute_mother_contract"
SOURCE1203 = ROOT / "tests/glm5/result/phase1203_object_attribute_behavior_protocol"
SOURCE1204 = ROOT / "tests/glm5/result/phase1204_object_attribute_behavior_execution"
SOURCE_ROWS = SOURCE1202 / "material/object_attribute_binding.jsonl"
SOURCE_MANIFEST = SOURCE1203 / "protocol/model_manifests/qwen3.jsonl"
SOURCE_BEHAVIOR = SOURCE1204 / "behavior/qwen3/raw_scores.jsonl"
SOURCE_FINAL = SOURCE1204 / "analysis/final.json"
SOURCE_AUDIT = SOURCE1204 / "audit/independent_result_audit.json"

EXPECTED_PHASE1204_FINAL_DIGEST = "5f35f53486123e4aa04806fec0a2ccf3633486a127ae72e9b6afb2c1a72c81dd"
EXPECTED_PHASE1204_AUDIT_DIGEST = "a0c87e4426ce3d56cd7af6405d776cf4c9113bb836b07fc8e3696a0ea165bbf8"
EXPECTED_QWEN_MANIFEST_DIGEST = "892b6a5b8904090d849f4b4cd85e8307b7f7a555d727d0d273db17741430b590"

PANELS = ("active", "matched_null", "surface_only", "semantic_neighbor")
SPLITS = ("discovery", "confirmation", "unseen_composition")
ROLES = (
    "record_entity0",
    "record_value0",
    "record_entity1",
    "record_value1",
    "record_anchor_value",
    "query_attribute",
    "query_value",
    "answer_prefix",
    "generation_boundary",
)
PREQUERY_ROLES = ROLES[:5]
LAYER_COUNT = 36
EVENT_COUNT = 109
PROJECTION_DIM = 64
EPSILON = 1e-8
THRESHOLDS = {
    "finite_fraction": 1.0,
    "minimum_active_relative_distance": 0.001,
    "active_to_max_control_median_ratio": 1.25,
    "active_over_all_controls_fraction": 0.75,
    "minimum_contiguous_discovery_depths": 2,
    "prequery_active_null_max_abs_difference": 1e-4,
}


def canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def validate_embedded_digest(value: dict[str, Any], key: str) -> None:
    candidate = {name: item for name, item in value.items() if name != key}
    if digest(candidate) != value.get(key):
        raise RuntimeError(f"embedded digest mismatch: {key}")


def add(checks: list[dict[str, Any]], name: str, passed: bool, detail: Any = None) -> None:
    checks.append({"name": name, "pass": bool(passed), "detail": detail})


def current_source_hashes() -> dict[str, str]:
    return {
        "main": sha256_file(MAIN_SCRIPT),
        "audit": sha256_file(AUDIT_SCRIPT),
        "runner": sha256_file(RUNNER_SCRIPT),
    }


def expected_event_registry() -> list[dict[str, Any]]:
    events = [{"event_id": "residual_d00", "component": "residual", "depth": 0}]
    for depth in range(1, LAYER_COUNT + 1):
        events.append({"event_id": f"residual_d{depth:02d}", "component": "residual", "depth": depth})
    for component in ("attention_output", "mlp_output"):
        for depth in range(1, LAYER_COUNT + 1):
            events.append({
                "event_id": f"{component}_d{depth:02d}",
                "component": component,
                "depth": depth,
            })
    return events


def protocol_and_pairs() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    protocol = read_json(PROTOCOL_PATH)
    validate_embedded_digest(protocol, "protocol_digest")
    pairs = read_jsonl(PAIR_MANIFEST_PATH)
    return protocol, pairs


def audit_pair_manifest(
    protocol: dict[str, Any], pairs: list[dict[str, Any]], checks: list[dict[str, Any]]
) -> dict[str, Any]:
    source_rows = read_jsonl(SOURCE_ROWS)
    source_manifest = read_jsonl(SOURCE_MANIFEST)
    behavior = read_jsonl(SOURCE_BEHAVIOR)
    source_index = {str(row["item_id"]): row for row in source_rows}
    manifest_index = {str(row["item_id"]): row for row in source_manifest}
    behavior_index = {str(row["item_id"]): row for row in behavior}

    add(checks, "source_case_count_4608", len(source_rows) == 4608, len(source_rows))
    add(checks, "source_item_sets_match", set(source_index) == set(manifest_index) == set(behavior_index))
    add(checks, "qwen_source_behavior_all_correct", all(bool(row["correct"]) for row in behavior))
    add(checks, "qwen_manifest_digest", digest(source_manifest) == EXPECTED_QWEN_MANIFEST_DIGEST)
    add(checks, "pair_manifest_digest", digest(pairs) == protocol["material"]["pair_manifest_digest"])
    add(
        checks,
        "pair_manifest_file_sha256",
        sha256_file(PAIR_MANIFEST_PATH) == protocol["material"]["pair_manifest_file_sha256"],
    )
    add(checks, "pair_indices_contiguous", [row["pair_index"] for row in pairs] == list(range(len(pairs))))

    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    row_errors: list[str] = []
    for pair in pairs:
        group_id = str(pair["group_id"])
        panel = str(pair["panel"])
        if panel in grouped[group_id]:
            row_errors.append(f"duplicate:{group_id}:{panel}")
        grouped[group_id][panel] = pair
        for state in (0, 1):
            item_id = str(pair[f"state{state}_item_id"])
            if item_id not in source_index or item_id not in manifest_index or item_id not in behavior_index:
                row_errors.append(f"missing_item:{item_id}")
                continue
            manifest_row = manifest_index[item_id]
            source_row = source_index[item_id]
            if int(manifest_row["input_length"]) != int(pair["input_length"]):
                row_errors.append(f"length:{item_id}")
            if str(manifest_row["input_ids_digest"]) != str(pair[f"state{state}_input_ids_digest"]):
                row_errors.append(f"ids_digest:{item_id}")
            if int(source_row["binding_state"]) != state or str(source_row["panel"]) != panel:
                row_errors.append(f"state_panel:{item_id}")
            positions = pair[f"state{state}_positions"]
            if set(positions) != set(ROLES):
                row_errors.append(f"roles:{item_id}")
            if any(not isinstance(value, int) or value < 0 or value >= int(pair["input_length"]) for value in positions.values()):
                row_errors.append(f"role_range:{item_id}")
    add(checks, "pair_rows_well_formed", not row_errors, row_errors[:20])
    add(
        checks,
        "eligible_groups_have_four_panels",
        all(set(panel_map) == set(PANELS) and len(panel_map) == 4 for panel_map in grouped.values()),
    )
    add(checks, "pair_count_four_per_group", len(pairs) == 4 * len(grouped))

    material_audit = protocol["material"]["material_audit"]
    excluded = list(material_audit["excluded_groups"])
    add(checks, "source_group_count_576", material_audit["source_group_count"] == 576)
    add(
        checks,
        "eligible_plus_excluded_576",
        len(grouped) + len(excluded) == 576,
        {"eligible": len(grouped), "excluded": len(excluded)},
    )
    add(checks, "eligible_group_count", len(grouped) == material_audit["eligible_group_count"])
    add(checks, "eligible_pair_count", len(pairs) == material_audit["eligible_pair_count"])
    add(
        checks,
        "split_group_counts",
        {
            split: len({str(row["group_id"]) for row in pairs if row["split"] == split})
            for split in SPLITS
        } == material_audit["split_group_counts"],
    )
    return {
        "source_case_count": len(source_rows),
        "eligible_group_count": len(grouped),
        "excluded_group_count": len(excluded),
        "eligible_pair_count": len(pairs),
    }


def preexecution_audit(write: bool) -> dict[str, Any]:
    if write and PREAUDIT_PATH.exists():
        raise RuntimeError("Phase1205 preexecution audit already exists")
    protocol, pairs = protocol_and_pairs()
    checks: list[dict[str, Any]] = []
    add(checks, "phase", protocol.get("phase") == PHASE)
    add(checks, "schema", protocol.get("schema_version") == "phase1205.qwen3_object_attribute_vertical_closure.v1")
    add(checks, "source_hashes", protocol.get("source_hashes") == current_source_hashes())
    add(checks, "event_registry", protocol.get("event_registry") == expected_event_registry())
    add(checks, "event_count_109", len(protocol.get("event_registry", [])) == EVENT_COUNT)
    add(checks, "roles_exact", protocol["material"]["roles"] == list(ROLES))
    add(checks, "prequery_roles_exact", protocol["material"]["prequery_roles"] == list(PREQUERY_ROLES))
    add(checks, "panels_exact", protocol["material"]["panels"] == list(PANELS))
    add(checks, "splits_exact", protocol["material"]["splits"] == list(SPLITS))
    add(checks, "thresholds_frozen", protocol["primary_gate"]["thresholds"] == THRESHOLDS)
    add(checks, "primary_scope_residual_generation", protocol["primary_gate"]["selection_scope"] == "residual component at generation_boundary only")
    add(checks, "primary_discovery_only", protocol["primary_gate"]["selection_split"] == "discovery")
    add(checks, "candidate_depths", protocol["primary_gate"]["candidate_depths"] == list(range(37)))
    add(checks, "three_controls", protocol["primary_gate"]["controls"] == list(PANELS[1:]))
    add(checks, "descriptive_cannot_select", "descriptive only" in protocol["primary_gate"]["no_hotspot_rule"])
    add(checks, "model_qwen3_only", protocol["scope"]["model"] == "qwen3" and protocol["scope"]["model_specific_only"] is True)
    add(checks, "no_cross_model_claim", protocol["scope"]["cross_model_claim"] is False)
    add(checks, "no_causal_claim", protocol["scope"]["causal_claim_in_this_phase"] is False)
    add(checks, "phase1204_not_reopened", protocol["scope"]["phase1204_registry_reopened"] is False)
    add(checks, "fp16_full_cuda", protocol["model"]["precision"] == "FP16" and protocol["model"]["quantization"] == "none" and protocol["model"]["placement"] == "full_cuda")

    final1204 = read_json(SOURCE_FINAL)
    audit1204 = read_json(SOURCE_AUDIT)
    validate_embedded_digest(final1204, "final_digest")
    validate_embedded_digest(audit1204, "audit_digest")
    add(checks, "phase1204_final_digest", final1204["final_digest"] == EXPECTED_PHASE1204_FINAL_DIGEST)
    add(checks, "phase1204_audit_digest", audit1204["audit_digest"] == EXPECTED_PHASE1204_AUDIT_DIGEST)
    add(checks, "phase1204_audit_pass", audit1204["gate_pass"] is True)
    add(checks, "phase1204_qwen_only", final1204["passing_models"] == ["qwen3"])
    add(checks, "phase1204_cross_failed", final1204["cross_model_behavior_pass"] is False)
    add(checks, "phase1204_cross_hidden_denied", final1204["authorized_next"]["cross_model_hidden_claim"] is False)
    add(checks, "upstream_file_hashes", protocol["upstream"]["files"] == {
        "phase1202_rows": sha256_file(SOURCE_ROWS),
        "phase1203_qwen_manifest": sha256_file(SOURCE_MANIFEST),
        "phase1204_qwen_behavior": sha256_file(SOURCE_BEHAVIOR),
        "phase1204_final": sha256_file(SOURCE_FINAL),
        "phase1204_audit": sha256_file(SOURCE_AUDIT),
    })
    material_counts = audit_pair_manifest(protocol, pairs, checks)
    add(checks, "zero_hidden_output_array", not ARRAY_PATH.exists())
    add(checks, "zero_hidden_run_summary", not RUN_SUMMARY_PATH.exists())
    add(checks, "zero_hidden_verdict", not VERDICT_PATH.exists())
    add(checks, "zero_hidden_final", not FINAL_PATH.exists())

    output: dict[str, Any] = {
        "phase": PHASE,
        "audit_stage": "preexecution",
        "protocol_digest": protocol["protocol_digest"],
        "checks": checks,
        "passed_checks": sum(item["pass"] for item in checks),
        "total_checks": len(checks),
        "gate_pass": all(item["pass"] for item in checks),
        "hidden_outputs_observed": 0,
        "material_counts": material_counts,
        "authorization": {
            "qwen3_hidden_run": all(item["pass"] for item in checks),
            "cross_model_hidden_run": False,
            "causal_intervention": False,
        },
    }
    output["audit_digest"] = digest(output)
    if write:
        if not output["gate_pass"]:
            raise RuntimeError("Phase1205 preexecution audit failed")
        write_json(PREAUDIT_PATH, output)
    return output


def median(values: Iterable[float]) -> float:
    items = [float(value) for value in values]
    return float(np.median(np.asarray(items, dtype=np.float64))) if items else 0.0


def event_metrics(
    relative: np.ndarray,
    pairs: list[dict[str, Any]],
    event_index: int,
    role_index: int,
    split: str,
) -> dict[str, Any]:
    by_group: dict[str, dict[str, float]] = defaultdict(dict)
    for pair in pairs:
        if pair["split"] == split:
            by_group[str(pair["group_id"])][str(pair["panel"])] = float(
                relative[int(pair["pair_index"]), event_index, role_index]
            )
    ratios: list[float] = []
    advantages: list[float] = []
    active_values: list[float] = []
    controls = {panel: [] for panel in PANELS if panel != "active"}
    for group_id, values in by_group.items():
        if set(values) != set(PANELS):
            raise RuntimeError(f"incomplete audit quartet: {group_id}")
        active = values["active"]
        maximum_control = max(values[panel] for panel in controls)
        active_values.append(active)
        for panel in controls:
            controls[panel].append(values[panel])
        ratios.append(active / (maximum_control + EPSILON))
        advantages.append(active - maximum_control)
    flattened = active_values + ratios + advantages + [v for values in controls.values() for v in values]
    result = {
        "split": split,
        "group_count": len(by_group),
        "finite_fraction": float(np.isfinite(np.asarray(flattened)).mean()),
        "active_median_relative_distance": median(active_values),
        "control_median_relative_distance": {panel: median(values) for panel, values in controls.items()},
        "active_to_max_control_median_ratio": median(ratios),
        "active_over_all_controls_fraction": sum(value > 0 for value in advantages) / max(len(advantages), 1),
        "median_active_minus_max_control": median(advantages),
    }
    result["pass"] = bool(
        result["finite_fraction"] >= THRESHOLDS["finite_fraction"]
        and result["active_median_relative_distance"] >= THRESHOLDS["minimum_active_relative_distance"]
        and result["active_to_max_control_median_ratio"] >= THRESHOLDS["active_to_max_control_median_ratio"]
        and result["active_over_all_controls_fraction"] >= THRESHOLDS["active_over_all_controls_fraction"]
    )
    return result


def contiguous_runs(depths: list[int]) -> list[list[int]]:
    runs: list[list[int]] = []
    for depth in sorted(depths):
        if not runs or depth != runs[-1][-1] + 1:
            runs.append([depth])
        else:
            runs[-1].append(depth)
    return runs


def precision_is_strict_fp16(precision: dict[str, Any]) -> bool:
    return bool(
        precision.get("has_fp16_parameters")
        and not precision.get("has_bf16_parameters")
        and not precision.get("has_quantized_modules")
        and set(precision.get("parameter_dtypes", [])) == {"float16"}
    )


def result_audit(write: bool) -> dict[str, Any]:
    if write and RESULT_AUDIT_PATH.exists():
        raise RuntimeError("Phase1205 independent result audit already exists")
    protocol, pairs = protocol_and_pairs()
    preaudit = read_json(PREAUDIT_PATH)
    validate_embedded_digest(preaudit, "audit_digest")
    summary = read_json(RUN_SUMMARY_PATH)
    verdict = read_json(VERDICT_PATH)
    trajectories = read_json(TRAJECTORY_PATH)
    validate_embedded_digest(summary, "summary_digest")
    validate_embedded_digest(verdict, "verdict_digest")
    validate_embedded_digest(trajectories, "trajectory_digest")
    checks: list[dict[str, Any]] = []
    add(checks, "preexecution_gate", preaudit["gate_pass"] is True)
    add(checks, "protocol_digest", summary["protocol_digest"] == protocol["protocol_digest"] == verdict["protocol_digest"])
    add(checks, "source_hashes_stable", protocol["source_hashes"] == current_source_hashes())
    add(checks, "array_file_sha256", summary["array_file_sha256"] == sha256_file(ARRAY_PATH))
    add(checks, "pair_count", summary["pair_count"] == len(pairs))
    add(checks, "event_count", summary["event_count"] == EVENT_COUNT)
    add(checks, "role_count", summary["role_count"] == len(ROLES))
    add(checks, "all_arrays_finite_summary", summary["all_arrays_finite"] is True)
    add(checks, "repeat_behavior_finite", summary["repeat_behavior_finite_rate"] == 1.0)
    add(checks, "repeat_behavior_accuracy", summary["repeat_behavior_accuracy"] == 1.0)
    add(checks, "strict_fp16_no_quantization", precision_is_strict_fp16(summary["precision_audit"]), summary["precision_audit"])
    add(checks, "full_cuda", summary["placement"].get("all_parameters_on_cuda") is True, summary["placement"])
    add(checks, "claim_boundary_no_causal", summary["claim_boundary"]["causal_evidence"] is False)

    with np.load(ARRAY_PATH, allow_pickle=False) as arrays:
        relative = arrays["relative_distance"]
        absolute = arrays["absolute_rms_distance"]
        projections = arrays["signed_generation_residual_projection"]
        behavior_correct = arrays["behavior_correct"]
        behavior_finite = arrays["behavior_finite"]
        margins = arrays["gold_margins"]
        expected_shapes = {
            "relative_distance": [len(pairs), EVENT_COUNT, len(ROLES)],
            "absolute_rms_distance": [len(pairs), EVENT_COUNT, len(ROLES)],
            "signed_generation_residual_projection": [len(pairs), LAYER_COUNT + 1, PROJECTION_DIM],
            "behavior_correct": [len(pairs), 2],
            "behavior_finite": [len(pairs), 2],
            "gold_margins": [len(pairs), 2],
        }
        actual_shapes = {
            "relative_distance": list(relative.shape),
            "absolute_rms_distance": list(absolute.shape),
            "signed_generation_residual_projection": list(projections.shape),
            "behavior_correct": list(behavior_correct.shape),
            "behavior_finite": list(behavior_finite.shape),
            "gold_margins": list(margins.shape),
        }
        add(checks, "array_shapes", actual_shapes == expected_shapes == summary["array_shapes"], actual_shapes)
        add(checks, "arrays_independently_finite", all(np.isfinite(value).all() for value in (relative, absolute, projections, margins)))
        add(checks, "behavior_arrays_all_one", bool(behavior_correct.all() and behavior_finite.all()))
        add(checks, "relative_nonnegative", bool((relative >= 0).all()))
        add(checks, "absolute_nonnegative", bool((absolute >= 0).all()))

        generation_role = ROLES.index("generation_boundary")
        discovery = {
            depth: event_metrics(relative, pairs, depth, generation_role, "discovery")
            for depth in range(LAYER_COUNT + 1)
        }
        passing_depths = [depth for depth, metrics in discovery.items() if metrics["pass"]]
        runs = [
            run for run in contiguous_runs(passing_depths)
            if len(run) >= THRESHOLDS["minimum_contiguous_discovery_depths"]
        ]
        selected_depth = runs[0][0] if runs else None
        selected_metrics = None
        if selected_depth is not None:
            selected_metrics = {
                split: event_metrics(relative, pairs, selected_depth, generation_role, split)
                for split in SPLITS
            }

        by_group_panel = {
            (str(pair["group_id"]), str(pair["panel"])): int(pair["pair_index"])
            for pair in pairs
        }
        prequery_differences: list[float] = []
        for group_id in sorted({str(pair["group_id"]) for pair in pairs}):
            active_index = by_group_panel[(group_id, "active")]
            null_index = by_group_panel[(group_id, "matched_null")]
            for event_index in range(EVENT_COUNT):
                for role in PREQUERY_ROLES:
                    role_index = ROLES.index(role)
                    prequery_differences.append(abs(float(
                        relative[active_index, event_index, role_index]
                        - relative[null_index, event_index, role_index]
                    )))
        prequery_max_abs = max(prequery_differences) if prequery_differences else math.inf
        instrument_pass = prequery_max_abs <= THRESHOLDS["prequery_active_null_max_abs_difference"]
        discovery_band_pass = bool(runs)
        confirmation_pass = bool(selected_metrics is not None and selected_metrics["confirmation"]["pass"])
        unseen_pass = bool(selected_metrics is not None and selected_metrics["unseen_composition"]["pass"])
        hidden_gate = bool(instrument_pass and discovery_band_pass and confirmation_pass and unseen_pass)

    stored_gate = verdict["primary_gate"]
    add(checks, "discovery_metrics", stored_gate["discovery_metrics_by_depth"] == {str(depth): value for depth, value in discovery.items()})
    add(checks, "passing_depths", stored_gate["discovery_passing_depths"] == passing_depths)
    add(checks, "qualifying_runs", stored_gate["discovery_qualifying_runs"] == runs)
    add(checks, "selected_depth", stored_gate["selected_depth"] == selected_depth)
    add(checks, "selected_metrics", stored_gate["selected_metrics"] == selected_metrics)
    add(checks, "instrument_max_abs", math.isclose(stored_gate["instrument_prequery_max_abs_difference"], prequery_max_abs, rel_tol=0.0, abs_tol=1e-12))
    add(checks, "instrument_pass", stored_gate["instrument_pass"] is instrument_pass)
    add(checks, "discovery_band_pass", stored_gate["discovery_band_pass"] is discovery_band_pass)
    add(checks, "confirmation_pass", stored_gate["confirmation_pass"] is confirmation_pass)
    add(checks, "unseen_pass", stored_gate["unseen_composition_pass"] is unseen_pass)
    add(checks, "hidden_gate", stored_gate["hidden_specificity_gate"] is hidden_gate)
    expected_status = "qwen3_hidden_specificity_qualified" if hidden_gate else "qwen3_hidden_specificity_not_qualified"
    add(checks, "status", verdict["status"] == expected_status)
    add(checks, "qwen_only_claim", verdict["claim_boundary"]["qwen3_model_specific"] is True)
    add(checks, "no_cross_model_evidence", verdict["claim_boundary"]["cross_model_evidence"] is False)
    add(checks, "no_causal_evidence", verdict["claim_boundary"]["causal_evidence"] is False)
    add(checks, "no_mechanism_closure", verdict["claim_boundary"]["mechanism_closure"] is False)
    add(checks, "trajectory_link", verdict["trajectory_digest"] == trajectories["trajectory_digest"])
    add(checks, "trajectory_primary_scope", trajectories["evidence_scope"] == "descriptive_only_except_primary_generation_residual_gate")

    output: dict[str, Any] = {
        "phase": PHASE,
        "audit_stage": "result",
        "protocol_digest": protocol["protocol_digest"],
        "run_summary_digest": summary["summary_digest"],
        "verdict_digest": verdict["verdict_digest"],
        "checks": checks,
        "passed_checks": sum(item["pass"] for item in checks),
        "total_checks": len(checks),
        "gate_pass": all(item["pass"] for item in checks),
        "independent_recomputation": {
            "passing_depths": passing_depths,
            "qualifying_runs": runs,
            "selected_depth": selected_depth,
            "selected_metrics": selected_metrics,
            "prequery_max_abs_difference": prequery_max_abs,
            "hidden_specificity_gate": hidden_gate,
        },
        "claim_boundary": {
            "qwen3_specific_hidden_specificity_only": True,
            "causal": False,
            "natural_use": False,
            "cross_model": False,
            "mechanism_closure": False,
        },
    }
    output["audit_digest"] = digest(output)
    if write:
        if not output["gate_pass"]:
            failed = [item["name"] for item in checks if not item["pass"]]
            raise RuntimeError(f"Phase1205 result audit failed: {failed}")
        write_json(RESULT_AUDIT_PATH, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("preexecution", "result"))
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    output = preexecution_audit(args.write) if args.stage == "preexecution" else result_audit(args.write)
    print(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
