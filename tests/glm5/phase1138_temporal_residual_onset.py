#!/usr/bin/env python3
"""Same-family residual-onset test for temporal binding.

The phase is behavior-conditioned and prospective with respect to all hidden
states. It tests whole residual-state sufficiency at the answer boundary. A
positive result is an event-level candidate, not a component or mechanism map.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests/glm5"
sys.path.insert(0, str(TEST_ROOT))

from model_utils import get_layers  # noqa: E402
from phase1023_fp16_utils import load_fp16, quantization_audit, release_fp16  # noqa: E402
import phase1135_temporal_binding_intervention as source  # noqa: E402
import phase1137_qwen14b_temporal_binding_endpoint as endpoint  # noqa: E402


PHASE = 1138
MODELS = ("qwen3_4b", "qwen3_14b")
OUT_ROOT = ROOT / "tests/glm5/result/phase1138_temporal_residual_onset"
SOURCE1135 = ROOT / "tests/glm5/result/phase1135_temporal_binding_intervention"
SOURCE1137 = ROOT / "tests/glm5/result/phase1137_qwen14b_temporal_binding_endpoint"
SOURCE_ITEMS = source.SOURCE

EXPECTED_PARAMETER_COUNTS = {
    "qwen3_4b": 4_022_468_096,
    "qwen3_14b": 14_768_307_200,
}
BATCH_SIZES = {"qwen3_4b": 8, "qwen3_14b": 16}
REQUESTED_FRACTIONS = tuple(float(value) for value in source.DEPTH_FRACTIONS)
MAXIMUM_MECHANISTIC_FRACTION = 0.80
MINIMUM_CONTIGUOUS_SHARED_DEPTHS = 2
BEHAVIOR_VALID_FRACTION_MIN = 0.95

CAUSAL_THRESHOLDS = {
    **source.CAUSAL_THRESHOLDS,
    "behavior_valid_fraction": BEHAVIOR_VALID_FRACTION_MIN,
    "maximum_mechanistic_fraction": MAXIMUM_MECHANISTIC_FRACTION,
    "minimum_contiguous_shared_depths": MINIMUM_CONTIGUOUS_SHARED_DEPTHS,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical(row) + "\n")


def median(values: Iterable[float | None]) -> float | None:
    finite = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    return statistics.median(finite) if finite else None


def behavior_good_items(
    decisions: list[dict[str, Any]],
    candidate_ids: list[str],
) -> set[str]:
    by_item: dict[str, dict[str, bool]] = {}
    for row in decisions:
        item_id = str(row["item_id"])
        state = str(row["state"])
        if item_id in candidate_ids and state in source.GATED_STATES:
            by_item.setdefault(item_id, {})[state] = bool(row["correct"])
    return {
        item_id
        for item_id in candidate_ids
        if item_id in by_item
        and all(by_item[item_id].get(state, False) for state in source.GATED_STATES)
    }


def protocol_command() -> None:
    if (OUT_ROOT / "causal").exists():
        raise RuntimeError("refusing to rewrite Phase1138 protocol after hidden output exists")

    prereg1135 = read_json(SOURCE1135 / "protocol/preregistration.json")
    audit1135 = read_json(SOURCE1135 / "audit/independent_result_audit.json")
    authorization1135 = read_json(SOURCE1135 / "analysis/behavior_authorization.json")
    final1137 = read_json(SOURCE1137 / "analysis/final_summary.json")
    audit1137 = read_json(SOURCE1137 / "audit/independent_result_audit.json")
    prereg1137 = read_json(SOURCE1137 / "protocol/preregistration.json")
    q4_decision_path = SOURCE1135 / "analysis/behavior_decisions.qwen3.jsonl"
    q14_decision_path = SOURCE1137 / "analysis/behavior_decisions.qwen3_14b.jsonl"
    q4_decisions = read_jsonl(q4_decision_path)
    q14_decisions = read_jsonl(q14_decision_path)

    cohorts: dict[str, list[str]] = {}
    cohort_audit: dict[str, Any] = {}
    for split in ("discovery", "confirmation"):
        candidate_ids = list(prereg1135["causal_items"][split])
        q4_good = behavior_good_items(q4_decisions, candidate_ids)
        q14_good = behavior_good_items(q14_decisions, candidate_ids)
        shared = sorted(q4_good & q14_good)
        cohorts[split] = shared
        cohort_audit[split] = {
            "source_count": len(candidate_ids),
            "qwen3_4b_all_four_count": len(q4_good),
            "qwen3_14b_all_four_count": len(q14_good),
            "shared_all_four_count": len(shared),
            "shared_item_ids": shared,
        }

    checks = {
        "phase1135_audit_passed": bool(audit1135["all_checks_passed"]),
        "phase1135_hidden_was_denied": authorization1135["hidden_scan_authorized"] is False,
        "phase1137_audit_passed": bool(audit1137["all_checks_passed"]),
        "phase1137_auto_continue_authorized": final1137["auto_continue"] is True,
        "phase1137_same_family_behavior_replication": final1137["same_family_behavior_replication"] is True,
        "phase1137_no_hidden_output": final1137["hidden_scanned"] is False,
        "discovery_shared_count_13": len(cohorts["discovery"]) == 13,
        "confirmation_shared_count_13": len(cohorts["confirmation"]) == 13,
        "cohorts_disjoint": set(cohorts["discovery"]).isdisjoint(cohorts["confirmation"]),
        "frozen_depth_grid": REQUESTED_FRACTIONS == tuple(source.DEPTH_FRACTIONS),
        "no_hidden_output_before_freeze": not (OUT_ROOT / "causal").exists(),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Phase1138 protocol checks failed: {checks}")

    phase1119_prereg = read_json(
        ROOT / "tests/glm5/result/phase1119_qwen3_4b_14b_scale/protocol/preregistration.json"
    )
    model_manifests = {
        name: phase1119_prereg["model_manifest_digests"][name]
        for name in MODELS
    }
    prereg_core = {
        "schema_version": "phase1138_temporal_residual_onset_preregistration.v1",
        "phase": PHASE,
        "created_at_utc": utc_now(),
        "objective": (
            "Test whether exact answer-boundary residual replacement yields a content-specific temporal-answer "
            "transition in Qwen3-4B and Qwen3-14B at a shared, non-late normalized depth band."
        ),
        "source": {
            "phase1135_protocol_digest": prereg1135["protocol_digest"],
            "phase1135_authorization_digest": authorization1135["authorization_digest"],
            "phase1135_audit_file_sha256": sha256_file(
                SOURCE1135 / "audit/independent_result_audit.json"
            ),
            "phase1137_protocol_digest": prereg1137["protocol_digest"],
            "phase1137_final_digest": final1137["final_digest"],
            "phase1137_audit_digest": audit1137["audit_digest"],
            "qwen4_decisions_sha256": sha256_file(q4_decision_path),
            "qwen14_decisions_sha256": sha256_file(q14_decision_path),
            "evidence_scope": "external_machine_consensus_not_human_gold",
        },
        "models": {
            "qwen3_4b": {
                "expected_parameter_count": EXPECTED_PARAMETER_COUNTS["qwen3_4b"],
                "manifest_digest": model_manifests["qwen3_4b"],
                "precision": "fp16",
                "quantization": "none",
                "placement": "full_cuda",
                "batch_size": BATCH_SIZES["qwen3_4b"],
            },
            "qwen3_14b": {
                "expected_parameter_count": EXPECTED_PARAMETER_COUNTS["qwen3_14b"],
                "manifest_digest": model_manifests["qwen3_14b"],
                "precision": "fp16",
                "quantization": "none",
                "placement": "frozen Phase1118 CUDA-plus-disk map",
                "batch_size": BATCH_SIZES["qwen3_14b"],
                "device_map": prereg1137["model"]["device_map"],
            },
        },
        "behavior_conditioning": {
            "rule": "retain only Phase1135-preselected items with all four gated states correct in both endpoints",
            "cohorts": cohorts,
            "cohort_audit": cohort_audit,
            "discovery_confirmation_disjoint": True,
            "conditioning_read_before_hidden": True,
            "hidden_read_before_protocol": False,
        },
        "intervention": {
            "component": "whole residual stream after layer at answer boundary",
            "operation": "exact donor-state replacement",
            "requested_depth_fractions": list(REQUESTED_FRACTIONS),
            "maximum_mechanistic_fraction": MAXIMUM_MECHANISTIC_FRACTION,
            "minimum_contiguous_shared_depths": MINIMUM_CONTIGUOUS_SHARED_DEPTHS,
            "discovery_selection": (
                "find shared passing requested fractions no later than 0.80; require at least two adjacent grid "
                "fractions; freeze the earliest fraction of the earliest qualifying run"
            ),
            "confirmation": "one frozen requested fraction per model on the independent cohort",
            "main_donors": [
                "original_pre <- original_post",
                "original_post <- original_pre",
                "swapped_pre <- swapped_post",
                "swapped_post <- swapped_pre",
            ],
            "negative_controls": [
                "same-answer nearby-date donor",
                "cross-item shuffled donor",
                "self replacement",
                "swapped-binding panel",
            ],
        },
        "thresholds": CAUSAL_THRESHOLDS,
        "predictions": {
            "P1": "all source identity, behavior conditioning, model, precision, cohort, and no-hidden-before-freeze checks pass",
            "P2": "both discovery scans have finite and behavior-valid fractions at least 0.95",
            "P3": "at least two adjacent shared requested fractions no later than 0.80 pass every causal and control threshold",
            "P4": "the earliest qualifying shared onset independently passes in both endpoints",
            "P5": "self replacement is numerically inert and main recovery exceeds same-answer and shuffled controls",
            "P6": "a pass identifies only a same-family whole-residual event candidate, not a component, circuit, neuron, or cross-architecture invariant",
        },
        "hard_stops": [
            "do not amend or reopen Phase1135",
            "do not select a lone depth spike",
            "do not select a requested fraction later than 0.80 as a mechanistic onset",
            "do not select different requested fractions independently after seeing confirmation",
            "do not change cohorts, thresholds, controls, precision, batch sizes, or donor definitions after hidden output",
            "do not inspect attention, MLP, heads, SAE, or neurons in Phase1138",
            "if no shared contiguous discovery band passes, deny confirmation",
            "if either endpoint fails independent confirmation, stop component expansion",
            "same-family confirmation cannot be called cross-architecture conservation",
            "machine consensus cannot be upgraded to human gold",
        ],
        "auto_continue_rule": (
            "only a two-endpoint independent confirmation authorizes a separately frozen minimal component "
            "mediation phase; otherwise stop"
        ),
    }
    prereg = dict(prereg_core)
    prereg["protocol_digest"] = digest(prereg_core)
    audit_core = {
        "schema_version": "phase1138_temporal_residual_protocol_audit.v1",
        "phase": PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "checks": checks,
        "check_count": len(checks),
        "passed_count": sum(bool(value) for value in checks.values()),
        "all_checks_passed": all(checks.values()),
    }
    audit = dict(audit_core)
    audit["audit_digest"] = digest(audit_core)
    write_json(OUT_ROOT / "protocol/preregistration.json", prereg)
    write_json(OUT_ROOT / "protocol/audit.json", audit)
    write_json(OUT_ROOT / "protocol/behavior_conditioned_cohorts.json", cohort_audit)
    print(json.dumps({
        "phase": PHASE,
        "command": "protocol",
        "checks": f"{audit['passed_count']}/{audit['check_count']}",
        "cohort_counts": {split: len(ids) for split, ids in cohorts.items()},
        "protocol_digest": prereg["protocol_digest"],
    }, ensure_ascii=False), flush=True)


def load_model(model_name: str, prereg: dict[str, Any]) -> tuple[Any, Any, torch.device, str]:
    if model_name == "qwen3_4b":
        return load_fp16("qwen3")
    endpoint_prereg = read_json(SOURCE1137 / "protocol/preregistration.json")
    model, tokenizer, _ = endpoint.load_model(endpoint_prereg)
    return model, tokenizer, torch.device("cuda:0"), "cuda_disk_offload"


def causal_record(
    model_name: str,
    split: str,
    requested_fraction: float,
    entry: dict[str, Any],
    patched: dict[str, float],
    clean: dict[str, dict[str, float]],
) -> dict[str, Any]:
    row = source.causal_record(model_name, split, entry, patched, clean)
    row["schema_version"] = "phase1138_temporal_residual_replacement_record.v1"
    row["phase"] = PHASE
    row["requested_fraction"] = requested_fraction
    row["same_family_only"] = True
    return row


def depth_rows_for_model(layer_count: int) -> list[dict[str, Any]]:
    rows = source.sampled_depths(layer_count)
    if tuple(float(row["requested_fraction"]) for row in rows) != REQUESTED_FRACTIONS:
        raise RuntimeError("sampled depth grid drift")
    return rows


def causal_command(model_name: str, split: str) -> None:
    prereg = read_json(OUT_ROOT / "protocol/preregistration.json")
    audit = read_json(OUT_ROOT / "protocol/audit.json")
    if not audit["all_checks_passed"]:
        raise RuntimeError("Phase1138 protocol audit failed")
    if model_name not in MODELS or split not in ("discovery", "confirmation"):
        raise RuntimeError("invalid Phase1138 endpoint")

    selection = None
    if split == "confirmation":
        selection = read_json(OUT_ROOT / "analysis/discovery_selection.json")
        if not selection["confirmation_authorized"]:
            raise RuntimeError("Phase1138 discovery did not authorize confirmation")

    items = read_jsonl(SOURCE_ITEMS)
    selected_ids = list(prereg["behavior_conditioning"]["cohorts"][split])
    selected_items, cases = source.causal_cases(items, split, selected_ids)
    model = None
    capture = None
    started = time.time()
    records: list[dict[str, Any]] = []
    try:
        model, tokenizer, device, placement = load_model(model_name, prereg)
        precision = quantization_audit(model)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        expected = prereg["models"][model_name]
        if parameter_count != expected["expected_parameter_count"]:
            raise RuntimeError(f"{model_name} parameter count mismatch")
        if precision["has_quantized_modules"] or precision["has_bf16_parameters"] or not precision["has_fp16_parameters"]:
            raise RuntimeError(f"{model_name} FP16/no-quantization gate failed")

        layers = get_layers(model)
        depth_rows = depth_rows_for_model(len(layers))
        if split == "confirmation":
            requested = float(selection["selected_requested_fraction"])
            depth_rows = [
                row
                for row in depth_rows
                if math.isclose(float(row["requested_fraction"]), requested, abs_tol=1e-12)
            ]
            if len(depth_rows) != 1:
                raise RuntimeError("frozen confirmation fraction did not map to exactly one depth")

        depths = [int(row["depth"]) for row in depth_rows]
        fraction_by_depth = {
            int(row["depth"]): float(row["requested_fraction"])
            for row in depth_rows
        }
        capture = source.ResidualCapture(layers, depths)
        capture.register()
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        batch_size = int(expected["batch_size"])
        prompt_rows = [source.prompt_token_row(tokenizer, case) for case in cases.values()]
        vectors = source.capture_vectors(
            model,
            capture,
            prompt_rows,
            int(pad_id),
            device,
            batch_size,
        )
        # The capture hook is read-only and must not remain live during clean
        # scoring, whose final batch can have a different cardinality.
        capture.close()
        capture = None
        clean = source.clean_scores_for_cases(
            model,
            tokenizer,
            cases,
            int(pad_id),
            device,
            batch_size,
        )
        for depth in depths:
            entries = source.build_patch_entries(selected_items, depth, vectors)
            entries_per_batch = max(1, batch_size // 2)
            for start in range(0, len(entries), entries_per_batch):
                batch_entries = entries[start : start + entries_per_batch]
                patched_rows = source.score_patch_batch(
                    model,
                    layers[depth - 1],
                    batch_entries,
                    cases,
                    tokenizer,
                    int(pad_id),
                    device,
                )
                for patched_row in patched_rows:
                    records.append(causal_record(
                        model_name,
                        split,
                        fraction_by_depth[depth],
                        patched_row["entry"],
                        patched_row["patched_scores"],
                        clean,
                    ))
            print(json.dumps({
                "phase": PHASE,
                "model": model_name,
                "split": split,
                "depth": depth,
                "requested_fraction": fraction_by_depth[depth],
                "records": len(records),
            }), flush=True)

        core = {
            "schema_version": "phase1138_temporal_residual_scan_summary.v1",
            "phase": PHASE,
            "model": model_name,
            "split": split,
            "protocol_digest": prereg["protocol_digest"],
            "precision": precision,
            "parameter_count": parameter_count,
            "placement": placement,
            "layer_count": len(layers),
            "sampled_depths": depth_rows,
            "item_count": len(selected_items),
            "record_count": len(records),
            "finite_fraction": sum(bool(row["finite"]) for row in records) / max(len(records), 1),
            "behavior_valid_fraction": sum(bool(row["behavior_valid"]) for row in records) / max(len(records), 1),
            "elapsed_seconds": time.time() - started,
            "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "record_digest": digest(records),
            "component": "whole residual stream after layer at answer boundary",
            "intervention": "exact state replacement",
            "evidence_scope": "same_family_machine_consensus_event_test",
        }
        summary = dict(core)
        summary["summary_digest"] = digest(core)
        output_root = OUT_ROOT / "causal" / split / model_name
        write_jsonl(output_root / "patch_records.jsonl", records)
        write_json(output_root / "summary.json", summary)
        print(json.dumps({
            "phase": PHASE,
            "command": "causal",
            "model": model_name,
            "split": split,
            "records": len(records),
            "finite_fraction": summary["finite_fraction"],
            "behavior_valid_fraction": summary["behavior_valid_fraction"],
            "elapsed_seconds": summary["elapsed_seconds"],
            "summary_digest": summary["summary_digest"],
        }, ensure_ascii=False), flush=True)
    finally:
        if capture is not None:
            capture.close()
        if model is not None:
            if model_name == "qwen3_4b":
                release_fp16(model)
            else:
                del model
                gc.collect()
                torch.cuda.empty_cache()


def depth_metrics(rows: list[dict[str, Any]], requested_fraction: float) -> dict[str, Any]:
    selected = [
        row
        for row in rows
        if math.isclose(float(row["requested_fraction"]), requested_fraction, abs_tol=1e-12)
    ]
    valid = [
        row
        for row in selected
        if row["finite"] and row["behavior_valid"] and row["recovery"] is not None
    ]
    kinds = {
        kind: [row for row in valid if row["patch_kind"] == kind]
        for kind in (
            "main",
            "same_answer_temporal_control",
            "shuffled_donor_control",
            "self_patch_audit",
        )
    }
    main = kinds["main"]
    original = [row for row in main if row["panel"] == "original"]
    swapped = [row for row in main if row["panel"] == "swapped"]
    same_answer_abs = median(
        abs(float(row["recovery"])) for row in kinds["same_answer_temporal_control"]
    )
    shuffled_abs = median(
        abs(float(row["recovery"])) for row in kinds["shuffled_donor_control"]
    )
    main_median = median(row["recovery"] for row in main)
    controls = [value for value in (same_answer_abs, shuffled_abs) if value is not None]
    specificity = main_median - max(controls) if main_median is not None and controls else None
    self_changes = [
        abs(float(row["margin_change"]))
        for row in kinds["self_patch_audit"]
        if row["margin_change"] is not None
    ]
    result = {
        "requested_fraction": requested_fraction,
        "depth": int(selected[0]["depth"]) if selected else None,
        "relative_depth": (
            int(selected[0]["depth"]) / int(selected[0].get("layer_count", 1))
            if selected and selected[0].get("layer_count") else None
        ),
        "record_count": len(selected),
        "valid_count": len(valid),
        "finite_fraction": sum(bool(row["finite"]) for row in selected) / max(len(selected), 1),
        "behavior_valid_fraction": sum(bool(row["behavior_valid"]) for row in selected) / max(len(selected), 1),
        "main_count": len(main),
        "main_median_recovery": main_median,
        "main_positive_fraction": sum(float(row["recovery"]) > 0.0 for row in main) / max(len(main), 1),
        "main_flip_fraction": sum(bool(row["flip"]) for row in main) / max(len(main), 1),
        "original_median_recovery": median(row["recovery"] for row in original),
        "swapped_median_recovery": median(row["recovery"] for row in swapped),
        "same_answer_control_median_abs_recovery": same_answer_abs,
        "shuffled_control_median_abs_recovery": shuffled_abs,
        "specificity_advantage": specificity,
        "self_patch_max_abs_margin_change": max(self_changes) if self_changes else None,
    }
    result["passed"] = bool(
        result["finite_fraction"] >= CAUSAL_THRESHOLDS["finite_fraction"]
        and result["behavior_valid_fraction"] >= CAUSAL_THRESHOLDS["behavior_valid_fraction"]
        and result["main_median_recovery"] is not None
        and result["main_median_recovery"] >= CAUSAL_THRESHOLDS["main_median_recovery"]
        and result["main_positive_fraction"] >= CAUSAL_THRESHOLDS["main_positive_fraction"]
        and result["original_median_recovery"] is not None
        and result["original_median_recovery"] >= CAUSAL_THRESHOLDS["panel_median_recovery"]
        and result["swapped_median_recovery"] is not None
        and result["swapped_median_recovery"] >= CAUSAL_THRESHOLDS["panel_median_recovery"]
        and result["specificity_advantage"] is not None
        and result["specificity_advantage"] >= CAUSAL_THRESHOLDS["specificity_advantage"]
        and result["self_patch_max_abs_margin_change"] is not None
        and result["self_patch_max_abs_margin_change"] <= CAUSAL_THRESHOLDS["self_patch_max_abs_margin_change"]
    )
    return result


def contiguous_runs(fractions: list[float]) -> list[list[float]]:
    ordered = sorted(set(fractions))
    runs: list[list[float]] = []
    for value in ordered:
        if not runs or not math.isclose(value - runs[-1][-1], 0.10, abs_tol=1e-9):
            runs.append([value])
        else:
            runs[-1].append(value)
    return runs


def finalize_discovery_command() -> None:
    prereg = read_json(OUT_ROOT / "protocol/preregistration.json")
    model_metrics: dict[str, Any] = {}
    for model_name in MODELS:
        rows = read_jsonl(OUT_ROOT / f"causal/discovery/{model_name}/patch_records.jsonl")
        metrics = [depth_metrics(rows, value) for value in REQUESTED_FRACTIONS]
        model_metrics[model_name] = {
            "depth_metrics": metrics,
            "passing_requested_fractions": [
                row["requested_fraction"] for row in metrics if row["passed"]
            ],
        }

    shared = sorted(
        set(model_metrics["qwen3_4b"]["passing_requested_fractions"])
        & set(model_metrics["qwen3_14b"]["passing_requested_fractions"])
    )
    mechanistic_shared = [
        value for value in shared if value <= MAXIMUM_MECHANISTIC_FRACTION + 1e-12
    ]
    runs = contiguous_runs(mechanistic_shared)
    qualifying = [run for run in runs if len(run) >= MINIMUM_CONTIGUOUS_SHARED_DEPTHS]
    selected = qualifying[0][0] if qualifying else None
    selected_depths = {}
    if selected is not None:
        for model_name in MODELS:
            row = next(
                metric
                for metric in model_metrics[model_name]["depth_metrics"]
                if math.isclose(metric["requested_fraction"], selected, abs_tol=1e-12)
            )
            selected_depths[model_name] = row["depth"]

    core = {
        "schema_version": "phase1138_temporal_residual_discovery_selection.v1",
        "phase": PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "models": model_metrics,
        "shared_passing_requested_fractions": shared,
        "mechanistic_shared_requested_fractions": mechanistic_shared,
        "contiguous_runs": runs,
        "qualifying_runs": qualifying,
        "selected_requested_fraction": selected,
        "selected_depths": selected_depths,
        "confirmation_authorized": selected is not None,
        "selection_rule": (
            "earliest requested fraction in earliest shared passing contiguous run of length at least two, "
            "restricted to fractions no later than 0.80"
        ),
    }
    result = dict(core)
    result["selection_digest"] = digest(core)
    write_json(OUT_ROOT / "analysis/discovery_selection.json", result)
    print(json.dumps({
        "phase": PHASE,
        "command": "finalize-discovery",
        "shared_passing_requested_fractions": shared,
        "qualifying_runs": qualifying,
        "selected_requested_fraction": selected,
        "selected_depths": selected_depths,
        "confirmation_authorized": result["confirmation_authorized"],
        "selection_digest": result["selection_digest"],
    }, ensure_ascii=False), flush=True)


def finalize_confirmation_command() -> None:
    prereg = read_json(OUT_ROOT / "protocol/preregistration.json")
    selection = read_json(OUT_ROOT / "analysis/discovery_selection.json")
    if not selection["confirmation_authorized"]:
        core = {
            "schema_version": "phase1138_temporal_residual_confirmation.v1",
            "phase": PHASE,
            "protocol_digest": prereg["protocol_digest"],
            "confirmation_run": False,
            "models": {},
            "same_family_residual_event_confirmed": False,
            "component_mediation_protocol_authorized": False,
            "auto_continue": False,
            "reason": "discovery lacked a shared non-late contiguous passing band",
            "claim_boundary": "confirmation untested, not negative",
        }
    else:
        selected = float(selection["selected_requested_fraction"])
        model_results = {}
        for model_name in MODELS:
            rows = read_jsonl(
                OUT_ROOT / f"causal/confirmation/{model_name}/patch_records.jsonl"
            )
            metrics = depth_metrics(rows, selected)
            model_results[model_name] = {
                "confirmed": bool(metrics["passed"]),
                "depth": metrics["depth"],
                "requested_fraction": selected,
                "metrics": metrics,
            }
        confirmed = all(row["confirmed"] for row in model_results.values())
        core = {
            "schema_version": "phase1138_temporal_residual_confirmation.v1",
            "phase": PHASE,
            "protocol_digest": prereg["protocol_digest"],
            "confirmation_run": True,
            "selected_requested_fraction": selected,
            "models": model_results,
            "same_family_residual_event_confirmed": confirmed,
            "cross_architecture_conservation": False,
            "component_mediation_protocol_authorized": confirmed,
            "auto_continue": confirmed,
            "next_action": (
                "freeze one minimal attention-versus-MLP mediation phase at the confirmed event"
                if confirmed
                else "stop temporal residual expansion; independent two-endpoint confirmation failed"
            ),
            "claim_boundary": (
                "A pass is same-family whole-residual causal sufficiency at one answer-boundary event. It does "
                "not identify a component, circuit, neuron, semantic code, necessity, or cross-architecture invariant."
            ),
            "evidence_scope": "same_family_machine_consensus_event_candidate",
            "human_annotation_eligible": False,
        }
    result = dict(core)
    result["confirmation_digest"] = digest(core)
    write_json(OUT_ROOT / "analysis/causal_confirmation.json", result)
    print(json.dumps({
        "phase": PHASE,
        "command": "finalize-confirmation",
        "confirmation_run": result["confirmation_run"],
        "same_family_residual_event_confirmed": result["same_family_residual_event_confirmed"],
        "auto_continue": result["auto_continue"],
        "confirmation_digest": result["confirmation_digest"],
    }, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("protocol")
    causal_parser = subparsers.add_parser("causal")
    causal_parser.add_argument("model", choices=MODELS)
    causal_parser.add_argument("split", choices=("discovery", "confirmation"))
    subparsers.add_parser("finalize-discovery")
    subparsers.add_parser("finalize-confirmation")
    args = parser.parse_args()
    if args.command == "protocol":
        protocol_command()
    elif args.command == "causal":
        causal_command(args.model, args.split)
    elif args.command == "finalize-discovery":
        finalize_discovery_command()
    else:
        finalize_confirmation_command()


if __name__ == "__main__":
    main()
