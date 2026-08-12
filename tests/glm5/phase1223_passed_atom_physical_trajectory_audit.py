#!/usr/bin/env python3
"""Independent preexecution and result audits for Phase 1223."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests/glm5"
sys.path.insert(0, str(TEST_ROOT))

import phase1223_passed_atom_physical_trajectory as p


def add(checks: list[dict[str, Any]], name: str, passed: bool, detail: Any) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def report(stage: str, checks: list[dict[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {
        "phase": p.PHASE,
        "audit_stage": stage,
        "check_count": len(checks),
        "passed_count": sum(row["passed"] for row in checks),
        "all_checks_passed": all(row["passed"] for row in checks),
        "checks": checks,
    }
    value["audit_digest"] = p.digest(value)
    return value


def preaudit() -> dict[str, Any]:
    protocol = p.read_json(p.PROTOCOL_PATH)
    pairs = p.read_jsonl(p.PAIR_PATH)
    states = p.read_jsonl(p.STATE_PATH)
    checks: list[dict[str, Any]] = []
    source_final = p.read_json(p.SOURCE_FINAL)
    source_audit = p.read_json(p.SOURCE_RESULT_AUDIT)
    source_precision = p.read_json(p.SOURCE_PRECISION_AUDIT)

    add(
        checks,
        "upstream_identity_and_automatic_authorization",
        source_final.get("final_digest") == p.EXPECTED_SOURCE_FINAL_DIGEST
        and source_final.get("authorized_next", {}).get("automatic_execution") is True,
        source_final.get("authorized_next"),
    )
    failed_source = [row["name"] for row in source_audit["checks"] if not row["passed"]]
    add(
        checks,
        "upstream_scientific_audit_boundary",
        failed_source == ["fp16_nonquantized_execution"]
        and source_precision.get("all_checks_passed") is True
        and source_precision.get("audit_digest") == p.EXPECTED_SOURCE_PRECISION_AUDIT_DIGEST,
        {"frozen_failures": failed_source, "precision": source_precision.get("audit_digest")},
    )
    claimed = protocol.get("protocol_digest")
    add(
        checks,
        "protocol_embedded_digest",
        claimed == p.digest({key: value for key, value in protocol.items() if key != "protocol_digest"}),
        claimed,
    )
    add(
        checks,
        "source_hashes_frozen",
        protocol["source_hashes"]["main"] == p.file_sha256(p.SCRIPT)
        and protocol["source_hashes"]["audit"] == p.file_sha256(p.AUDIT_SCRIPT)
        and protocol["source_hashes"]["phase1222_final"] == p.file_sha256(p.SOURCE_FINAL)
        and protocol["source_hashes"]["phase1222_raw"] == p.file_sha256(p.SOURCE_RAW),
        protocol["source_hashes"],
    )
    expected_scopes = source_final["behavior"]["authorized_target_operation_tracks"]
    add(
        checks,
        "exact_atomic_scopes_only",
        protocol["upstream"]["authorized_scopes"] == expected_scopes
        and not (set(protocol["upstream"]["authorized_scopes"]) - set(expected_scopes)),
        expected_scopes,
    )
    expected_pairs = len(expected_scopes) * len(p.SPLITS) * p.PAIRS_PER_SCOPE_SPLIT
    add(
        checks,
        "formal_counts",
        len(pairs) == expected_pairs and len(states) == expected_pairs * len(p.PANELS),
        {"pairs": len(pairs), "states": len(states), "expected_pairs": expected_pairs},
    )
    add(
        checks,
        "formal_digests",
        protocol["material"]["pair_digest"] == p.digest(pairs)
        and protocol["material"]["state_digest"] == p.digest(states),
        protocol["material"],
    )
    add(
        checks,
        "row_digests_and_indices",
        all(row["pair_digest"] == p.digest({k: v for k, v in row.items() if k != "pair_digest"}) for row in pairs)
        and all(row["state_digest"] == p.digest({k: v for k, v in row.items() if k != "state_digest"}) for row in states)
        and sorted(row["pair_index"] for row in pairs) == list(range(len(pairs)))
        and sorted(row["state_index"] for row in states) == list(range(len(states))),
        True,
    )
    cell_counts = Counter((row["scope"], row["split"]) for row in pairs)
    add(
        checks,
        "scope_split_pair_balance",
        len(cell_counts) == len(expected_scopes) * len(p.SPLITS)
        and set(cell_counts.values()) == {p.PAIRS_PER_SCOPE_SPLIT},
        {"cell_count": len(cell_counts), "counts": sorted(set(cell_counts.values()))},
    )
    state_by_id = {row["state_id"]: row for row in states}
    add(
        checks,
        "four_panel_states_per_pair",
        len(state_by_id) == len(states)
        and all(set(pair["panel_states"]) == set(p.PANELS) for pair in pairs)
        and all(state_id in state_by_id for pair in pairs for state_id in pair["panel_states"].values()),
        len(state_by_id),
    )
    add(
        checks,
        "counterfactual_gold_changes",
        all(pair["recipient_gold"] != pair["donor_gold"] for pair in pairs),
        True,
    )
    role_bounds = all(
        set(state["role_positions"]) == set(p.ROLES)
        and all(0 <= int(position) < int(state["input_token_count"]) for position in state["role_positions"].values())
        and len(set(state["position_audit"]["candidate_token_lengths"].values())) == 1
        for state in states
    )
    add(checks, "role_and_candidate_token_contracts", role_bounds, role_bounds)

    rebuilt_pairs, rebuilt_states, rebuilt_audit = p.build_material()
    add(
        checks,
        "independent_material_and_role_replay",
        p.digest(rebuilt_pairs) == p.digest(pairs)
        and p.digest(rebuilt_states) == p.digest(states),
        {"pair_digest": rebuilt_audit["pair_digest"], "state_digest": rebuilt_audit["state_digest"]},
    )
    source_raw = {row["item_id"]: row for row in p.read_jsonl(p.SOURCE_RAW)}
    add(
        checks,
        "selected_states_are_behavior_correct",
        all(
            source_raw[state["source_item_id"]]["candidate_correct"]
            and source_raw[state["source_item_id"]]["context_correct"]
            and source_raw[state["source_item_id"]]["open_generation_correct"]
            for state in states
        ),
        True,
    )
    events = protocol["camera"]["events"]
    event_counts = Counter(row["component"] for row in events)
    add(
        checks,
        "complete_six_component_event_registry",
        events == p.event_registry()
        and len(events) == 1 + p.LAYER_COUNT * 6
        and event_counts == Counter(
            {
                "residual": p.LAYER_COUNT + 1,
                "attention_output": p.LAYER_COUNT,
                "mlp_output": p.LAYER_COUNT,
                "q_output": p.LAYER_COUNT,
                "k_output": p.LAYER_COUNT,
                "v_output": p.LAYER_COUNT,
            }
        ),
        dict(event_counts),
    )
    add(
        checks,
        "causal_scan_is_residual_boundary_only",
        protocol["causal_handoff"]["patch_component"] == "whole residual stream"
        and protocol["causal_handoff"]["patch_role"] == "generation_boundary"
        and protocol["causal_handoff"]["scan_depths"] == list(range(p.LAYER_COUNT + 1)),
        protocol["causal_handoff"],
    )
    add(
        checks,
        "discovery_holdout_separation",
        protocol["causal_handoff"]["discovery_split"] == "discovery"
        and protocol["causal_handoff"]["holdout_splits"] == list(p.HOLDOUT_SPLITS)
        and "discovery" not in protocol["causal_handoff"]["holdout_splits"],
        protocol["causal_handoff"]["holdout_splits"],
    )
    add(
        checks,
        "thresholds_and_contiguous_rule_frozen",
        protocol["causal_handoff"]["discovery_thresholds"] == p.DISCOVERY_THRESHOLDS
        and protocol["causal_handoff"]["holdout_thresholds"] == p.HOLDOUT_THRESHOLDS
        and p.DISCOVERY_THRESHOLDS["contiguous_depths_min"] == 2,
        {"discovery": p.DISCOVERY_THRESHOLDS, "holdout": p.HOLDOUT_THRESHOLDS},
    )
    add(
        checks,
        "descriptive_camera_cannot_select_targets",
        protocol["camera"]["trajectory_is_descriptive"] is True
        and protocol["camera"]["trajectory_cannot_select_components_or_neurons"] is True,
        protocol["camera"]["contrasts"],
    )
    add(
        checks,
        "claim_and_stop_boundaries",
        protocol["claim_boundary"]["head_or_neuron"] is False
        and protocol["claim_boundary"]["cross_model"] is False
        and protocol["authorization"]["no_head_or_neuron_search"] is True,
        protocol["claim_boundary"],
    )
    value = report("preexecution", checks)
    p.write_json(p.PREAUDIT_PATH, value)
    return value


def recompute_patch_row(row: dict[str, Any]) -> bool:
    recipient_margin = p.margin(row["recipient_scores"], row["donor_gold"], row["recipient_gold"])
    donor_margin = p.margin(row["donor_scores"], row["donor_gold"], row["recipient_gold"])
    patched_margin = p.margin(row["patched_scores"], row["donor_gold"], row["recipient_gold"])
    target = donor_margin - recipient_margin
    shift = patched_margin - recipient_margin
    completion = shift / target if abs(target) > p.EPSILON else 0.0
    prediction = max(row["patched_scores"], key=lambda key: (row["patched_scores"][key], key))
    return (
        math.isclose(row["recipient_margin"], recipient_margin, rel_tol=0.0, abs_tol=1e-12)
        and math.isclose(row["donor_margin"], donor_margin, rel_tol=0.0, abs_tol=1e-12)
        and math.isclose(row["patched_margin"], patched_margin, rel_tol=0.0, abs_tol=1e-12)
        and math.isclose(row["target_shift"], target, rel_tol=0.0, abs_tol=1e-12)
        and math.isclose(row["patch_shift"], shift, rel_tol=0.0, abs_tol=1e-12)
        and math.isclose(row["completion"], completion, rel_tol=0.0, abs_tol=1e-12)
        and row["patched_prediction"] == prediction
    )


def result_audit() -> dict[str, Any]:
    protocol = p.read_json(p.PROTOCOL_PATH)
    pairs = p.read_jsonl(p.PAIR_PATH)
    states = p.read_jsonl(p.STATE_PATH)
    pre = p.read_json(p.PREAUDIT_PATH)
    summary = p.read_json(p.RUN_SUMMARY_PATH)
    selection = p.read_json(p.SELECTION_PATH)
    patches = p.read_jsonl(p.PATCH_PATH)
    trajectory = p.read_json(p.TRAJECTORY_PATH)
    final = p.read_json(p.FINAL_PATH)
    arrays = np.load(p.ARRAY_PATH, allow_pickle=False)
    checks: list[dict[str, Any]] = []

    add(checks, "preaudit_passed", pre.get("all_checks_passed") is True, pre.get("audit_digest"))
    expected_shapes = {
        "projections": (len(states), len(p.event_registry()), len(p.ROLES), p.PROJECTION_DIM),
        "rms": (len(states), len(p.event_registry()), len(p.ROLES)),
        "residual_boundary": (len(states), p.LAYER_COUNT + 1, p.HIDDEN_SIZE),
    }
    add(
        checks,
        "array_shapes_and_summary",
        all(tuple(arrays[key].shape) == shape for key, shape in expected_shapes.items())
        and summary["array_shapes"] == {key: list(value) for key, value in expected_shapes.items()},
        {key: list(arrays[key].shape) for key in arrays.files},
    )
    add(
        checks,
        "arrays_finite",
        all(np.isfinite(arrays[key]).all() for key in arrays.files),
        {key: str(arrays[key].dtype) for key in arrays.files},
    )
    add(
        checks,
        "run_file_digests",
        summary["array_file_sha256"] == p.file_sha256(p.ARRAY_PATH)
        and summary["patch_digest"] == p.digest(patches)
        and summary["summary_digest"] == p.digest({key: value for key, value in summary.items() if key != "summary_digest"}),
        summary["summary_digest"],
    )
    add(
        checks,
        "selection_digest",
        selection["selection_digest"] == p.digest({key: value for key, value in selection.items() if key != "selection_digest"})
        and summary["selection_digest"] == selection["selection_digest"],
        selection["selection_digest"],
    )
    discovery_records = [row for row in patches if row["split"] == "discovery"]
    recomputed_selection = p.select_discovery(
        discovery_records,
        list(protocol["upstream"]["authorized_scopes"]),
        protocol["protocol_digest"],
    )
    # created_at is execution metadata and is excluded from deterministic comparison.
    left = {key: value for key, value in selection.items() if key not in {"created_at", "selection_digest"}}
    right = {key: value for key, value in recomputed_selection.items() if key not in {"created_at", "selection_digest"}}
    add(checks, "discovery_selection_recomputation", left == right, selection["scope_selections"])
    discovered_scopes = [
        scope
        for scope, value in selection["scope_selections"].items()
        if value["discovery_authorized"]
    ]
    expected_patch_count = (
        len(protocol["upstream"]["authorized_scopes"])
        * p.PAIRS_PER_SCOPE_SPLIT
        * (p.LAYER_COUNT + 1)
        * 2
        + len(discovered_scopes)
        * len(p.HOLDOUT_SPLITS)
        * p.PAIRS_PER_SCOPE_SPLIT
        * 3
    )
    add(
        checks,
        "patch_record_count_and_conditions",
        len(patches) == expected_patch_count == summary["patch_record_count"]
        and all(
            (row["split"] == "discovery" and row["condition"] in {"correct", "wrong"} and row["candidate_count"] == 2)
            or (row["split"] in p.HOLDOUT_SPLITS and row["condition"] in {"correct", "wrong", "zero"} and row["candidate_count"] == 4)
            for row in patches
        ),
        {"actual": len(patches), "expected": expected_patch_count},
    )
    add(
        checks,
        "patch_rows_recompute_and_digest",
        all(recompute_patch_row(row) for row in patches)
        and all(row["patch_digest"] == p.digest({key: value for key, value in row.items() if key != "patch_digest"}) for row in patches),
        True,
    )
    add(
        checks,
        "patch_finite_and_single_call",
        all(row["finite"] and row["patch_calls"] == 1 for row in patches),
        Counter(row["patch_calls"] for row in patches),
    )
    add(
        checks,
        "holdouts_only_for_discovery_selected_scopes",
        {row["scope"] for row in patches if row["split"] in p.HOLDOUT_SPLITS} == set(discovered_scopes)
        and all(
            row["depth"] == selection["scope_selections"][row["scope"]]["selected_depth"]
            for row in patches
            if row["split"] in p.HOLDOUT_SPLITS
        ),
        discovered_scopes,
    )
    zero_rows = [row for row in patches if row["condition"] == "zero"]
    zero_drift = max(
        (
            abs(row["patched_scores"][candidate] - row["recipient_scores"][candidate])
            for row in zero_rows
            for candidate in row["patched_scores"]
        ),
        default=0.0,
    )
    add(
        checks,
        "zero_patch_identity",
        zero_drift <= p.HOLDOUT_THRESHOLDS["zero_patch_max_abs_score_drift_max"],
        zero_drift,
    )
    recomputed_trajectory = p.trajectory_summary(protocol, pairs, states, arrays)
    add(
        checks,
        "trajectory_recomputation",
        trajectory == recomputed_trajectory,
        trajectory["trajectory_digest"],
    )
    recomputed_scope_results: dict[str, Any] = {}
    physical_scopes: list[str] = []
    for scope in protocol["upstream"]["authorized_scopes"]:
        discovery = selection["scope_selections"][scope]
        holdouts: dict[str, Any] = {}
        for split in p.HOLDOUT_SPLITS:
            values = [row for row in patches if row["scope"] == scope and row["split"] == split]
            holdouts[split] = p.holdout_metrics(values) if values else None
        physical = bool(discovery["discovery_authorized"]) and all(
            holdouts[split] is not None and holdouts[split]["passed"] for split in p.HOLDOUT_SPLITS
        )
        if physical:
            physical_scopes.append(scope)
        recomputed_scope_results[scope] = {
            "selected_depth": discovery["selected_depth"],
            "discovery_authorized": discovery["discovery_authorized"],
            "holdouts": holdouts,
            "physical_scope_closed": physical,
        }
    add(
        checks,
        "physical_gate_recomputation",
        final["scope_results"] == recomputed_scope_results
        and final["physical_scopes"] == physical_scopes,
        physical_scopes,
    )
    precision = summary["precision_audit"]
    add(
        checks,
        "pure_fp16_execution",
        set(precision.get("parameter_dtypes", {})) == {"float16"}
        and precision.get("has_quantized_modules") is False
        and precision.get("has_bf16_parameters") is False,
        precision,
    )
    add(
        checks,
        "final_digest_and_no_automatic_escalation",
        final["final_digest"] == p.digest({key: value for key, value in final.items() if key != "final_digest"})
        and final["authorized_next"]["automatic_execution"] is False
        and final["authorized_next"]["head_or_neuron_search"] is False,
        final["authorized_next"],
    )
    add(
        checks,
        "claim_boundary_preserved",
        final["claim_boundary"] == protocol["claim_boundary"],
        final["claim_boundary"],
    )
    value = report("result", checks)
    p.write_json(p.RESULT_AUDIT_PATH, value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=("pre", "result"))
    args = parser.parse_args()
    value = preaudit() if args.stage == "pre" else result_audit()
    print(json.dumps(value, ensure_ascii=False, indent=2))
    if not value["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
