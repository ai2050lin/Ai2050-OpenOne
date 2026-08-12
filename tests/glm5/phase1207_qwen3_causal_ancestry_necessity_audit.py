#!/usr/bin/env python3
"""Independent zero-output and result audit for Phase1207."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

import phase1207_qwen3_causal_ancestry_necessity as run


EXPECTED_ONSET_IDS = tuple(item["id"] for item in run.ONSET_CONDITIONS)
EXPECTED_NECESSITY_IDS = tuple(item["id"] for item in run.NECESSITY_CONDITIONS)
EXPECTED_RESCUE_IDS = tuple(item["id"] for item in run.RESCUE_CONDITIONS)


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def validate(value: dict[str, Any], key: str) -> None:
    if digest({name: item for name, item in value.items() if name != key}) != value.get(key):
        raise RuntimeError(f"embedded digest mismatch: {key}")


def add(checks: list[dict[str, Any]], name: str, passed: bool, detail: Any = None) -> None:
    checks.append({"name": name, "pass": bool(passed), "detail": detail})


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def source_hashes() -> dict[str, str]:
    return {
        "main": run.sha256_file(Path(run.__file__).resolve()),
        "audit": run.sha256_file(Path(__file__).resolve()),
        "runner": run.sha256_file(run.RUNNER_SCRIPT),
        "phase1206_main": run.sha256_file(Path(run.phase1206.__file__).resolve()),
    }


def same(left: Any, right: Any) -> bool:
    return digest(left) == digest(right)


def median(values: Iterable[float]) -> float:
    data = np.asarray([float(value) for value in values], dtype=np.float64)
    return float(np.median(data)) if data.size else 0.0


def strict_runtime(summary: dict[str, Any]) -> bool:
    precision = summary["precision_audit"]
    placement = summary["placement"]
    return bool(
        precision["has_fp16_parameters"]
        and not precision["has_bf16_parameters"]
        and not precision["has_quantized_modules"]
        and set(precision["parameter_dtypes"]) == {"float16"}
        and placement["placement"] == "full_cuda"
        and placement["devices"] == ["cuda:0"]
        and placement["quantization"] == "none"
    )


def preexecution(write_output: bool) -> dict[str, Any]:
    if write_output and run.PREAUDIT_PATH.exists():
        raise RuntimeError("Phase1207 preexecution audit already exists")
    protocol = run.read_json(run.PROTOCOL_PATH)
    validate(protocol, "protocol_digest")
    final1206 = run.read_json(run.UPSTREAM_FINAL_PATH)
    audit1206 = run.read_json(run.UPSTREAM_AUDIT_PATH)
    validate(final1206, "final_digest")
    validate(audit1206, "audit_digest")
    pairs, _, _, active = run.load_material()
    checks: list[dict[str, Any]] = []
    add(checks, "phase", protocol.get("phase") == 1207)
    add(checks, "schema", protocol.get("schema_version") == "phase1207.qwen3_causal_ancestry_necessity.v1")
    add(checks, "source_hashes", protocol.get("source_hashes") == source_hashes())
    add(checks, "upstream_final_digest", final1206["final_digest"] == run.EXPECTED_1206_FINAL)
    add(checks, "upstream_audit_digest", audit1206["audit_digest"] == run.EXPECTED_1206_AUDIT)
    add(checks, "upstream_audit_pass", audit1206["gate_pass"] is True)
    add(checks, "upstream_causal_gate", final1206["causal_transfer_gate"] is True)
    add(checks, "upstream_authorization", final1206["authorized_next"]["phase1207_qwen3_necessity_rescue_preregistration"] is True)
    add(checks, "qwen_only", protocol["scope"]["qwen3_only"] is True and protocol["scope"]["model"] == "qwen3")
    add(checks, "controlled_only", protocol["scope"]["controlled_object_attribute_only"] is True)
    add(checks, "no_component_claim", protocol["scope"]["head_or_neuron_claim"] is False)
    add(checks, "no_natural_claim", protocol["scope"]["natural_use_claim"] is False)
    add(checks, "no_cross_model_claim", protocol["scope"]["cross_model_claim"] is False)
    add(checks, "no_brain_claim", protocol["scope"]["brain_claim"] is False)
    add(checks, "no_closure_claim", protocol["scope"]["mechanism_closure_claim"] is False)
    add(checks, "strict_fp16_contract", protocol["model"]["precision"] == "FP16" and protocol["model"]["quantization"] == "none" and protocol["model"]["placement"] == "full_cuda")
    add(checks, "depths", tuple(protocol["causal_onset"]["depths"]) == run.DEPTHS)
    add(checks, "capture_depths", run.CAPTURE_DEPTHS == (20, 21, 22, 23, 24, 25, 26, 30, 36))
    add(checks, "onset_conditions", tuple(item["id"] for item in protocol["causal_onset"]["conditions"]) == EXPECTED_ONSET_IDS)
    add(checks, "onset_controls", tuple(protocol["causal_onset"]["primary_controls"]) == run.ONSET_PRIMARY_CONTROLS)
    add(checks, "onset_thresholds", protocol["causal_onset"]["thresholds"] == run.ONSET_THRESHOLDS)
    add(checks, "adjacent_selection", protocol["causal_onset"]["thresholds"]["minimum_adjacent_discovery_depths"] == 2)
    add(checks, "heldout_no_reselection", "no reselection" in protocol["causal_onset"]["confirmation_rule"])
    add(checks, "necessity_conditions", tuple(item["id"] for item in protocol["necessity"]["conditions"]) == EXPECTED_NECESSITY_IDS)
    add(checks, "necessity_controls", tuple(protocol["necessity"]["primary_controls"]) == run.NECESSITY_PRIMARY_CONTROLS)
    add(checks, "necessity_thresholds", protocol["necessity"]["thresholds"] == run.NECESSITY_THRESHOLDS)
    add(checks, "midpoint_preserving_operation", "h_s' = h_s + (0.5-s)(d_A-d_S)" in protocol["necessity"]["operation"])
    add(checks, "rescue_conditions", tuple(item["id"] for item in protocol["rescue"]["conditions"]) == EXPECTED_RESCUE_IDS)
    add(checks, "rescue_controls", tuple(protocol["rescue"]["primary_controls"]) == run.RESCUE_PRIMARY_CONTROLS)
    add(checks, "rescue_thresholds", protocol["rescue"]["thresholds"] == run.RESCUE_THRESHOLDS)
    add(checks, "conditional_rescue", protocol["rescue"]["rescue_depth"] == 25 and "only if" in protocol["rescue"]["authorization"])
    add(checks, "no_automatic_phase1208", protocol["authorization"]["automatic_phase1208"] is False)
    add(checks, "pair_hash", protocol["upstream"]["pair_file_sha256"] == run.sha256_file(run.PAIR_PATH))
    add(checks, "pair_digest", protocol["upstream"]["pair_digest"] == digest(pairs))
    add(checks, "manifest_hash", protocol["upstream"]["manifest_file_sha256"] == run.sha256_file(run.MANIFEST_PATH))
    add(checks, "upstream_vector_hash", protocol["upstream"]["phase1206_vector_sha256"] == run.sha256_file(run.UPSTREAM_VECTOR_PATH))
    add(checks, "pair_counts", len(pairs) == 2016 and len(active) == 504)
    add(checks, "onset_count", protocol["counts"]["onset_records"] == 504 * 2 * 6 * len(EXPECTED_ONSET_IDS))
    add(checks, "necessity_count", protocol["counts"]["necessity_records_if_authorized"] == 504 * 2 * len(EXPECTED_NECESSITY_IDS))
    add(checks, "rescue_count", protocol["counts"]["rescue_records_if_authorized"] == 504 * 2 * len(EXPECTED_RESCUE_IDS))
    output_paths = (
        run.CAPTURE_PATH, run.CAPTURE_SUMMARY_PATH, run.ONSET_RAW_PATH, run.ONSET_SUMMARY_PATH,
        run.ONSET_VERDICT_PATH, run.NECESSITY_RAW_PATH, run.NECESSITY_SUMMARY_PATH,
        run.NECESSITY_VERDICT_PATH, run.RESCUE_RAW_PATH, run.RESCUE_SUMMARY_PATH,
        run.RESCUE_VERDICT_PATH, run.RESULT_AUDIT_PATH, run.FINAL_PATH,
    )
    add(checks, "zero_model_outputs", not any(path.exists() for path in output_paths))
    output: dict[str, Any] = {
        "phase": 1207,
        "audit_stage": "preexecution",
        "protocol_digest": protocol["protocol_digest"],
        "checks": checks,
        "passed_checks": sum(item["pass"] for item in checks),
        "total_checks": len(checks),
        "gate_pass": all(item["pass"] for item in checks),
        "model_outputs_observed": 0,
        "authorization": {
            "capture_and_onset": all(item["pass"] for item in checks),
            "necessity": False,
            "rescue": False,
            "component_search": False,
        },
    }
    output["audit_digest"] = digest(output)
    if write_output:
        if not output["gate_pass"]:
            raise RuntimeError([item["name"] for item in checks if not item["pass"]])
        write(run.PREAUDIT_PATH, output)
    return output


def transfer_enrich(row: dict[str, Any]) -> dict[str, Any]:
    labels = list(row["candidate_labels"])
    ri = labels.index(str(row["recipient_gold"]))
    di = labels.index(str(row["donor_gold"]))
    recipient = np.asarray(row["recipient_scores"], dtype=np.float64)
    donor = np.asarray(row["donor_scores"], dtype=np.float64)
    patched = np.asarray(row["patched_scores"], dtype=np.float64)
    base = float(recipient[di] - recipient[ri])
    donor_margin = float(donor[di] - donor[ri])
    patched_margin = float(patched[di] - patched[ri])
    shift = patched_margin - base
    full = donor_margin - base
    return {
        **row,
        "donor_margin_shift": shift,
        "transfer_fraction": shift / (full + run.EPSILON),
        "positive_shift": shift > 0,
        "donor_choice": row["patched_prediction"] == row["donor_gold"],
        "recipient_correct": row["recipient_prediction"] == row["recipient_gold"],
        "donor_correct": row["donor_prediction"] == row["donor_gold"],
    }


def onset_metrics(rows: list[dict[str, Any]], split: str, depth: int) -> dict[str, Any]:
    members = [row for row in rows if row["split"] == split and int(row["depth"]) == depth]
    lookup = {(row["group_id"], int(row["recipient_state"]), row["condition"]): row for row in members}
    target = [row for row in members if row["condition"] == "active_full"]
    advantages = []
    for row in target:
        controls = [lookup[(row["group_id"], int(row["recipient_state"]), name)]["donor_margin_shift"] for name in run.ONSET_PRIMARY_CONTROLS]
        advantages.append(float(row["donor_margin_shift"]) - max(float(value) for value in controls))
    directions = {}
    for state in (0, 1):
        subset = [row for row in target if int(row["recipient_state"]) == state]
        directions[f"state{state}_to_state{1-state}"] = sum(bool(row["donor_choice"]) for row in subset) / max(len(subset), 1)
    result = {
        "split": split,
        "depth": depth,
        "target_count": len(target),
        "finite_fraction": sum(bool(row["recipient_finite"] and row["donor_finite"] and row["patched_finite"]) for row in target) / max(len(target), 1),
        "baseline_accuracy": sum(bool(row["recipient_correct"]) for row in target) / max(len(target), 1),
        "donor_accuracy": sum(bool(row["donor_correct"]) for row in target) / max(len(target), 1),
        "positive_shift_fraction": sum(bool(row["positive_shift"]) for row in target) / max(len(target), 1),
        "donor_choice_fraction": sum(bool(row["donor_choice"]) for row in target) / max(len(target), 1),
        "median_shift": median(row["donor_margin_shift"] for row in target),
        "median_transfer_fraction": median(row["transfer_fraction"] for row in target),
        "beats_all_controls_fraction": sum(value > 0 for value in advantages) / max(len(advantages), 1),
        "median_advantage": median(advantages),
        "direction_donor_choice": directions,
    }
    t = run.ONSET_THRESHOLDS
    result["pass"] = bool(
        result["finite_fraction"] >= t["finite_fraction"]
        and result["baseline_accuracy"] >= t["baseline_accuracy"]
        and result["donor_accuracy"] >= t["donor_accuracy"]
        and result["positive_shift_fraction"] >= t["positive_shift_fraction"]
        and result["donor_choice_fraction"] >= t["donor_choice_fraction"]
        and result["median_transfer_fraction"] >= t["median_transfer_fraction"]
        and result["beats_all_controls_fraction"] >= t["beats_all_controls_fraction"]
        and result["median_advantage"] >= t["median_advantage"]
        and min(directions.values()) >= t["minimum_each_direction_donor_choice"]
    )
    return result


def contiguous_runs(depths: list[int]) -> list[list[int]]:
    result: list[list[int]] = []
    for depth in sorted(depths):
        if not result or depth != result[-1][-1] + 1:
            result.append([depth])
        else:
            result[-1].append(depth)
    return result


def necessity_enrich(row: dict[str, Any], full_shift: float) -> dict[str, Any]:
    labels = list(row["candidate_labels"])
    ri = labels.index(str(row["recipient_gold"]))
    di = labels.index(str(row["donor_gold"]))
    recipient = np.asarray(row["recipient_scores"], dtype=np.float64)
    patched = np.asarray(row["patched_scores"], dtype=np.float64)
    base_margin = float(recipient[ri] - recipient[di])
    patched_margin = float(patched[ri] - patched[di])
    damage = base_margin - patched_margin
    return {
        **row,
        "recipient_margin": base_margin,
        "patched_recipient_margin": patched_margin,
        "margin_damage": damage,
        "damage_fraction": damage / (abs(float(full_shift)) + run.EPSILON),
        "positive_damage": damage > 0,
        "behavior_damage": row["patched_prediction"] != row["recipient_gold"],
        "recipient_correct": row["recipient_prediction"] == row["recipient_gold"],
    }


def necessity_metrics(rows: list[dict[str, Any]], split: str) -> dict[str, Any]:
    members = [row for row in rows if row["split"] == split]
    lookup = {(row["group_id"], int(row["recipient_state"]), row["condition"]): row for row in members}
    target = [row for row in members if row["condition"] == "active_vs_surface_remove"]
    advantages = []
    for row in target:
        controls = [lookup[(row["group_id"], int(row["recipient_state"]), name)]["damage_fraction"] for name in run.NECESSITY_PRIMARY_CONTROLS]
        advantages.append(float(row["damage_fraction"]) - max(float(value) for value in controls))
    directions = {}
    for state in (0, 1):
        subset = [row for row in target if int(row["recipient_state"]) == state]
        directions[f"state{state}"] = sum(bool(row["behavior_damage"]) for row in subset) / max(len(subset), 1)
    result = {
        "split": split,
        "target_count": len(target),
        "finite_fraction": sum(bool(row["recipient_finite"] and row["patched_finite"]) for row in target) / max(len(target), 1),
        "baseline_accuracy": sum(bool(row["recipient_correct"]) for row in target) / max(len(target), 1),
        "positive_damage_fraction": sum(bool(row["positive_damage"]) for row in target) / max(len(target), 1),
        "behavior_damage_fraction": sum(bool(row["behavior_damage"]) for row in target) / max(len(target), 1),
        "median_margin_damage": median(row["margin_damage"] for row in target),
        "median_damage_fraction": median(row["damage_fraction"] for row in target),
        "beats_all_controls_fraction": sum(value > 0 for value in advantages) / max(len(advantages), 1),
        "median_normalized_advantage": median(advantages),
        "direction_behavior_damage": directions,
    }
    t = run.NECESSITY_THRESHOLDS
    result["pass"] = bool(
        result["finite_fraction"] >= t["finite_fraction"]
        and result["baseline_accuracy"] >= t["baseline_accuracy"]
        and result["positive_damage_fraction"] >= t["positive_damage_fraction"]
        and result["behavior_damage_fraction"] >= t["behavior_damage_fraction"]
        and result["median_damage_fraction"] >= t["median_damage_fraction"]
        and result["beats_all_controls_fraction"] >= t["beats_all_controls_fraction"]
        and result["median_normalized_advantage"] >= t["median_normalized_advantage"]
        and min(directions.values()) >= t["minimum_each_direction_behavior_damage"]
    )
    return result


def rescue_enrich(row: dict[str, Any], damage: dict[str, Any]) -> dict[str, Any]:
    labels = list(row["candidate_labels"])
    ri = labels.index(str(row["recipient_gold"]))
    di = labels.index(str(row["donor_gold"]))
    base = np.asarray(row["recipient_scores"], dtype=np.float64)
    damaged = np.asarray(damage["patched_scores"], dtype=np.float64)
    rescued = np.asarray(row["patched_scores"], dtype=np.float64)
    base_margin = float(base[ri] - base[di])
    damaged_margin = float(damaged[ri] - damaged[di])
    rescued_margin = float(rescued[ri] - rescued[di])
    lost = base_margin - damaged_margin
    recovery = (rescued_margin - damaged_margin) / (lost + run.EPSILON)
    damage_error = float(damage["response_error_to_clean"])
    response_recovery = 1.0 - float(row["response_error_to_clean"]) / (damage_error + run.EPSILON)
    return {
        **row,
        "damage_prediction": damage["patched_prediction"],
        "damage_behavior": damage["patched_prediction"] != row["recipient_gold"],
        "behavior_restored": row["patched_prediction"] == row["recipient_gold"],
        "base_margin": base_margin,
        "damaged_margin": damaged_margin,
        "rescued_margin": rescued_margin,
        "margin_recovery": recovery,
        "positive_margin_recovery": recovery > 0,
        "damage_response_error": damage_error,
        "response_recovery": response_recovery,
    }


def rescue_metrics(rows: list[dict[str, Any]], split: str) -> dict[str, Any]:
    members = [row for row in rows if row["split"] == split]
    lookup = {(row["group_id"], int(row["recipient_state"]), row["condition"]): row for row in members}
    target = [row for row in members if row["condition"] == "specific_addback"]
    damaged = [row for row in target if row["damage_behavior"]]
    margin_advantages = []
    response_advantages = []
    for row in target:
        controls = [lookup[(row["group_id"], int(row["recipient_state"]), name)] for name in run.RESCUE_PRIMARY_CONTROLS]
        margin_advantages.append(float(row["margin_recovery"]) - max(float(item["margin_recovery"]) for item in controls))
        response_advantages.append(float(row["response_recovery"]) - max(float(item["response_recovery"]) for item in controls))
    directions = {}
    for state in (0, 1):
        subset = [row for row in damaged if int(row["recipient_state"]) == state]
        directions[f"state{state}"] = {
            "count": len(subset),
            "restore_fraction": sum(bool(row["behavior_restored"]) for row in subset) / max(len(subset), 1),
        }
    clamps = [row for row in members if row["condition"] == "clean_state_clamp"]
    result = {
        "split": split,
        "target_count": len(target),
        "damaged_count": len(damaged),
        "finite_fraction": sum(bool(row["patched_finite"]) for row in target) / max(len(target), 1),
        "behavior_restore_fraction": sum(bool(row["behavior_restored"]) for row in damaged) / max(len(damaged), 1),
        "median_margin_recovery": median(row["margin_recovery"] for row in target),
        "positive_margin_recovery_fraction": sum(bool(row["positive_margin_recovery"]) for row in target) / max(len(target), 1),
        "median_response_recovery": median(row["response_recovery"] for row in target),
        "margin_beats_all_controls_fraction": sum(value > 0 for value in margin_advantages) / max(len(margin_advantages), 1),
        "response_beats_all_controls_fraction": sum(value > 0 for value in response_advantages) / max(len(response_advantages), 1),
        "median_margin_advantage": median(margin_advantages),
        "direction_restore": directions,
        "clean_clamp_restore_fraction": sum(row["patched_prediction"] == row["recipient_gold"] for row in clamps) / max(len(clamps), 1),
        "clean_clamp_median_response_recovery": median(row["response_recovery"] for row in clamps),
    }
    t = run.RESCUE_THRESHOLDS
    result["pass"] = bool(
        result["finite_fraction"] >= t["finite_fraction"]
        and result["damaged_count"] >= t["minimum_damaged_records_per_split"]
        and all(value["count"] >= t["minimum_damaged_records_per_direction"] for value in directions.values())
        and result["behavior_restore_fraction"] >= t["behavior_restore_fraction"]
        and result["median_margin_recovery"] >= t["median_margin_recovery"]
        and result["positive_margin_recovery_fraction"] >= t["positive_margin_recovery_fraction"]
        and result["median_response_recovery"] >= t["median_response_recovery"]
        and result["margin_beats_all_controls_fraction"] >= t["margin_beats_all_controls_fraction"]
        and result["response_beats_all_controls_fraction"] >= t["response_beats_all_controls_fraction"]
        and result["median_margin_advantage"] >= t["median_margin_advantage"]
        and min(value["restore_fraction"] for value in directions.values()) >= t["minimum_each_direction_restore"]
        and result["clean_clamp_restore_fraction"] >= t["clean_clamp_restore_fraction"]
        and result["clean_clamp_median_response_recovery"] >= t["clean_clamp_median_response_recovery"]
    )
    return result


def result(write_output: bool) -> dict[str, Any]:
    if write_output and run.RESULT_AUDIT_PATH.exists():
        raise RuntimeError("Phase1207 result audit already exists")
    protocol = run.verify_protocol()
    preaudit = run.read_json(run.PREAUDIT_PATH)
    validate(preaudit, "audit_digest")
    checks: list[dict[str, Any]] = []
    add(checks, "preexecution_pass", preaudit["gate_pass"] is True)
    add(checks, "protocol_source_hashes", protocol["source_hashes"] == source_hashes())
    add(checks, "final_absent_before_audit", not run.FINAL_PATH.exists())

    capture_summary = run.read_json(run.CAPTURE_SUMMARY_PATH)
    validate(capture_summary, "summary_digest")
    add(checks, "capture_protocol", capture_summary["protocol_digest"] == protocol["protocol_digest"])
    add(checks, "capture_hash", capture_summary["capture_file_sha256"] == run.sha256_file(run.CAPTURE_PATH))
    add(checks, "capture_runtime", strict_runtime(capture_summary))
    with np.load(run.CAPTURE_PATH, allow_pickle=False) as arrays:
        residuals = arrays["residuals"]
        scores = arrays["baseline_scores"]
        finite = arrays["baseline_finite"]
        depths = tuple(int(value) for value in arrays["capture_depths"].tolist())
        add(checks, "capture_shapes", list(residuals.shape) == [2016, 2, 9, 2560] and list(scores.shape) == [2016, 2, 3] and list(finite.shape) == [2016, 2])
        add(checks, "capture_depth_order", depths == run.CAPTURE_DEPTHS)
        add(checks, "capture_finite", bool(np.isfinite(residuals).all() and np.isfinite(scores).all() and finite.all()))
        add(checks, "capture_summary_values", capture_summary["residual_shape"] == list(residuals.shape) and capture_summary["scores_shape"] == list(scores.shape) and math.isclose(capture_summary["finite_fraction"], float(finite.mean()), abs_tol=0.0))
        upstream = np.load(run.UPSTREAM_VECTOR_PATH, allow_pickle=False)
        replay24 = float(np.max(np.abs(residuals[:, :, depths.index(24)].astype(np.float32) - upstream["d24_generation_boundary"].astype(np.float32))))
        replay25 = float(np.max(np.abs(residuals[:, :, depths.index(25)].astype(np.float32) - upstream["d25_generation_boundary"].astype(np.float32))))
        replay_scores = float(np.max(np.abs(scores.astype(np.float32) - upstream["baseline_scores"].astype(np.float32))))
        add(checks, "upstream_replay_values", capture_summary["upstream_replay_max_abs"] == {"depth24": replay24, "depth25": replay25, "scores": replay_scores})
        add(checks, "upstream_replay_exact", replay24 == 0.0 and replay25 == 0.0 and replay_scores == 0.0)
        upstream.close()

    onset_summary = run.read_json(run.ONSET_SUMMARY_PATH)
    onset_verdict = run.read_json(run.ONSET_VERDICT_PATH)
    validate(onset_summary, "summary_digest")
    validate(onset_verdict, "verdict_digest")
    onset_raw = run.read_jsonl_gz(run.ONSET_RAW_PATH)
    add(checks, "onset_protocol", onset_summary["protocol_digest"] == onset_verdict["protocol_digest"] == protocol["protocol_digest"])
    add(checks, "onset_hash", onset_summary["raw_file_sha256"] == run.sha256_file(run.ONSET_RAW_PATH))
    add(checks, "onset_digest", onset_summary["raw_digest"] == digest(onset_raw))
    add(checks, "onset_count", len(onset_raw) == onset_summary["record_count"] == protocol["counts"]["onset_records"])
    add(checks, "onset_unique", len({row["record_id"] for row in onset_raw}) == len(onset_raw))
    add(checks, "onset_finite", all(row["recipient_finite"] and row["donor_finite"] and row["patched_finite"] and np.isfinite(row["recipient_scores"]).all() and np.isfinite(row["donor_scores"]).all() and np.isfinite(row["patched_scores"]).all() for row in onset_raw))
    add(checks, "onset_baseline", all(row["recipient_prediction"] == row["recipient_gold"] and row["donor_prediction"] == row["donor_gold"] for row in onset_raw))
    add(checks, "onset_runtime", strict_runtime(onset_summary))
    onset_rows = [transfer_enrich(row) for row in onset_raw]
    onset_metrics_recomputed = {split: {str(depth): onset_metrics(onset_rows, split, depth) for depth in run.DEPTHS} for split in run.SPLITS}
    passing = [depth for depth in run.DEPTHS if onset_metrics_recomputed["discovery"][str(depth)]["pass"]]
    runs = contiguous_runs(passing)
    qualified = [item for item in runs if len(item) >= run.ONSET_THRESHOLDS["minimum_adjacent_discovery_depths"]]
    selected = qualified[0][0] if qualified else None
    heldout = bool(selected is not None and all(onset_metrics_recomputed[split][str(selected)]["pass"] for split in ("confirmation", "unseen_composition")))
    zero_rows = [row for row in onset_raw if row["condition"] == "zero"]
    zero_drift = max(abs(float(a) - float(b)) for row in zero_rows for a, b in zip(row["patched_scores"], row["recipient_scores"]))
    identity = zero_drift <= run.ONSET_THRESHOLDS["zero_max_abs_logit_drift"]
    onset_gate = bool(selected is not None and heldout and identity)
    add(checks, "onset_metrics", same(onset_verdict["metrics"], onset_metrics_recomputed))
    add(checks, "onset_selection", onset_verdict["discovery_passing_depths"] == passing and onset_verdict["discovery_runs"] == runs and onset_verdict["qualifying_runs"] == qualified and onset_verdict["selected_depth"] == selected)
    add(checks, "onset_heldout", onset_verdict["heldout_pass_at_selected"] is heldout)
    add(checks, "onset_identity", math.isclose(onset_verdict["zero_max_abs_logit_drift"], zero_drift, rel_tol=0.0, abs_tol=1e-12) and onset_verdict["identity_pass"] is identity)
    add(checks, "onset_gate", onset_verdict["onset_gate"] is onset_gate)
    add(checks, "onset_authorization", onset_verdict["authorization"]["necessity_run"] is onset_gate and onset_verdict["authorization"]["component_search"] is False)

    necessity_gate = False
    rescue_gate = False
    necessity_recomputed: dict[str, Any] | None = None
    rescue_recomputed: dict[str, Any] | None = None
    if not onset_gate:
        add(checks, "necessity_absent_after_onset_stop", not run.NECESSITY_RAW_PATH.exists() and not run.NECESSITY_SUMMARY_PATH.exists() and not run.NECESSITY_VERDICT_PATH.exists())
        add(checks, "rescue_absent_after_onset_stop", not run.RESCUE_RAW_PATH.exists() and not run.RESCUE_SUMMARY_PATH.exists() and not run.RESCUE_VERDICT_PATH.exists())
    else:
        necessity_summary = run.read_json(run.NECESSITY_SUMMARY_PATH)
        necessity_verdict = run.read_json(run.NECESSITY_VERDICT_PATH)
        validate(necessity_summary, "summary_digest")
        validate(necessity_verdict, "verdict_digest")
        necessity_raw = run.read_jsonl_gz(run.NECESSITY_RAW_PATH)
        add(checks, "necessity_links", necessity_summary["protocol_digest"] == necessity_verdict["protocol_digest"] == protocol["protocol_digest"] and necessity_summary["onset_verdict_digest"] == necessity_verdict["onset_verdict_digest"] == onset_verdict["verdict_digest"])
        add(checks, "necessity_hash", necessity_summary["raw_file_sha256"] == run.sha256_file(run.NECESSITY_RAW_PATH))
        add(checks, "necessity_digest", necessity_summary["raw_digest"] == digest(necessity_raw))
        add(checks, "necessity_count", len(necessity_raw) == necessity_summary["record_count"] == protocol["counts"]["necessity_records_if_authorized"])
        add(checks, "necessity_unique", len({row["record_id"] for row in necessity_raw}) == len(necessity_raw))
        add(checks, "necessity_finite", all(row["recipient_finite"] and row["patched_finite"] and np.isfinite(row["recipient_scores"]).all() and np.isfinite(row["patched_scores"]).all() for row in necessity_raw))
        add(checks, "necessity_baseline", all(row["recipient_prediction"] == row["recipient_gold"] for row in necessity_raw))
        add(checks, "necessity_runtime", strict_runtime(necessity_summary))
        add(checks, "necessity_frozen_depth", necessity_summary["selected_depth"] == necessity_verdict["selected_depth"] == selected)
        full = {(row["group_id"], int(row["recipient_state"])): float(row["donor_margin_shift"]) for row in onset_rows if int(row["depth"]) == selected and row["condition"] == "active_full"}
        necessity_rows = [necessity_enrich(row, full[(row["group_id"], int(row["recipient_state"]))]) for row in necessity_raw]
        necessity_metrics_recomputed = {split: necessity_metrics(necessity_rows, split) for split in run.SPLITS}
        necessity_zero = [row for row in necessity_raw if row["condition"] == "zero"]
        necessity_zero_drift = max(abs(float(a) - float(b)) for row in necessity_zero for a, b in zip(row["patched_scores"], row["recipient_scores"]))
        necessity_identity = necessity_zero_drift <= run.NECESSITY_THRESHOLDS["zero_max_abs_logit_drift"]
        necessity_gate = bool(necessity_identity and all(necessity_metrics_recomputed[split]["pass"] for split in run.SPLITS))
        rescue_authorized = bool(necessity_gate and selected < run.RESCUE_DEPTH)
        add(checks, "necessity_metrics", same(necessity_verdict["metrics"], necessity_metrics_recomputed))
        add(checks, "necessity_identity", math.isclose(necessity_verdict["zero_max_abs_logit_drift"], necessity_zero_drift, rel_tol=0.0, abs_tol=1e-12) and necessity_verdict["identity_pass"] is necessity_identity)
        add(checks, "necessity_gate", necessity_verdict["necessity_gate"] is necessity_gate)
        add(checks, "necessity_authorization", necessity_verdict["authorization"]["rescue_run"] is rescue_authorized and necessity_verdict["authorization"]["component_search"] is False)
        necessity_recomputed = {"metrics": necessity_metrics_recomputed, "zero_drift": necessity_zero_drift, "gate": necessity_gate}
        if not rescue_authorized:
            add(checks, "rescue_absent_when_denied", not run.RESCUE_RAW_PATH.exists() and not run.RESCUE_SUMMARY_PATH.exists() and not run.RESCUE_VERDICT_PATH.exists())
        else:
            rescue_summary = run.read_json(run.RESCUE_SUMMARY_PATH)
            rescue_verdict = run.read_json(run.RESCUE_VERDICT_PATH)
            validate(rescue_summary, "summary_digest")
            validate(rescue_verdict, "verdict_digest")
            rescue_raw = run.read_jsonl_gz(run.RESCUE_RAW_PATH)
            add(checks, "rescue_links", rescue_summary["protocol_digest"] == rescue_verdict["protocol_digest"] == protocol["protocol_digest"] and rescue_summary["necessity_verdict_digest"] == rescue_verdict["necessity_verdict_digest"] == necessity_verdict["verdict_digest"])
            add(checks, "rescue_hash", rescue_summary["raw_file_sha256"] == run.sha256_file(run.RESCUE_RAW_PATH))
            add(checks, "rescue_digest", rescue_summary["raw_digest"] == digest(rescue_raw))
            add(checks, "rescue_count", len(rescue_raw) == rescue_summary["record_count"] == protocol["counts"]["rescue_records_if_authorized"])
            add(checks, "rescue_unique", len({row["record_id"] for row in rescue_raw}) == len(rescue_raw))
            add(checks, "rescue_finite", all(row["patched_finite"] and np.isfinite(row["patched_scores"]).all() and math.isfinite(float(row["response_error_to_clean"])) for row in rescue_raw))
            add(checks, "rescue_runtime", strict_runtime(rescue_summary))
            add(checks, "rescue_frozen_depths", rescue_summary["damage_depth"] == selected and rescue_summary["rescue_depth"] == run.RESCUE_DEPTH)
            damage = {(row["group_id"], int(row["recipient_state"])): row for row in rescue_raw if row["condition"] == "damage_only"}
            necessity_target = {(row["group_id"], int(row["recipient_state"])): row for row in necessity_raw if row["condition"] == "active_vs_surface_remove"}
            damage_match = max(abs(float(a) - float(b)) for key, row in damage.items() for a, b in zip(row["patched_scores"], necessity_target[key]["patched_scores"]))
            add(checks, "damage_replay_matches_necessity", damage_match <= 1e-6, damage_match)
            rescue_rows = [rescue_enrich(row, damage[(row["group_id"], int(row["recipient_state"]))]) for row in rescue_raw]
            rescue_metrics_recomputed = {split: rescue_metrics(rescue_rows, split) for split in run.SPLITS}
            rescue_gate = all(rescue_metrics_recomputed[split]["pass"] for split in run.SPLITS)
            add(checks, "rescue_metrics", same(rescue_verdict["metrics"], rescue_metrics_recomputed))
            add(checks, "rescue_gate", rescue_verdict["rescue_gate"] is rescue_gate)
            add(checks, "rescue_no_component_search", rescue_verdict["authorization"]["component_search"] is False)
            rescue_recomputed = {"metrics": rescue_metrics_recomputed, "gate": rescue_gate, "damage_replay_max_abs": damage_match}

    output: dict[str, Any] = {
        "phase": 1207,
        "audit_stage": "result",
        "protocol_digest": protocol["protocol_digest"],
        "checks": checks,
        "passed_checks": sum(item["pass"] for item in checks),
        "total_checks": len(checks),
        "gate_pass": all(item["pass"] for item in checks),
        "independent_recomputation": {
            "onset": {
                "metrics": onset_metrics_recomputed,
                "passing_depths": passing,
                "runs": runs,
                "qualifying_runs": qualified,
                "selected_depth": selected,
                "heldout": heldout,
                "zero_drift": zero_drift,
                "gate": onset_gate,
            },
            "necessity": necessity_recomputed,
            "rescue": rescue_recomputed,
        },
        "claim_boundary": {
            "qwen3_controlled_only": True,
            "full_state_onset_not_mechanism_onset": True,
            "necessity_is_active_surface_contrast_only": True,
            "minimal_implementation": False,
            "natural_use": False,
            "cross_model": False,
            "brain": False,
            "mechanism_closure": False,
        },
    }
    output["audit_digest"] = digest(output)
    if write_output:
        if not output["gate_pass"]:
            raise RuntimeError([item["name"] for item in checks if not item["pass"]])
        write(run.RESULT_AUDIT_PATH, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preexecution", "result"))
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    output = preexecution(args.write) if args.command == "preexecution" else result(args.write)
    print(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
