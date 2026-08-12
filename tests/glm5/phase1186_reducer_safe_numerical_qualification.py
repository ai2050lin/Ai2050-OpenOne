#!/usr/bin/env python3
"""Phase1186: one-shot reducer-safe numerical qualification repair.

The only semantic correction from Phase1185 is that universal Boolean
predicates are reduced with integer counts rather than an FP32 mean compared
exactly with one. Phase1185 remains frozen and unchanged.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))

import phase1185_orthogonal_numerical_behavior_qualification as p1185  # noqa: E402


PHASE = 1186
SCRIPT = Path(__file__).resolve()
AUDIT_SCRIPT = ROOT / "tests/glm5/phase1186_reducer_safe_numerical_qualification_audit.py"
OUT_ROOT = ROOT / "tests/glm5/result/phase1186_reducer_safe_numerical_qualification"
PROTOCOL_PATH = OUT_ROOT / "protocol/preregistration.json"
FINAL_PATH = OUT_ROOT / "analysis/final.json"

SCALES = p1185.ENGINEERED_SCALES
STRUCTURES = p1185.ENGINEERED_STRUCTURES
REPLICATES = 8
TRANSFORMS = p1185.ENGINEERED_GAUGE_TRANSFORMS
THRESHOLDS = dict(p1185.THRESHOLDS)
REDUCER_SENTINEL_LENGTHS = (1, 3, 7, 61, 227, 3721, 8191)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
    os.replace(temporary, path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def model_seed(scale_index: int, structure_index: int, replicate: int) -> int:
    return 11860000 + scale_index * 100_003 + structure_index * 10_007 + replicate * 1_009


def transform_seed(case_index: int, transform: int) -> int:
    return 11865000 + case_index * 101 + transform


def positive_seed(case_index: int) -> int:
    return 11869000 + case_index * 103


def validate_phase1185() -> tuple[dict[str, Any], dict[str, Any]]:
    final = read_json(p1185.FINAL_PATH)
    audit_path = p1185.OUT_ROOT / "audit/independent_audit.json"
    audit = read_json(audit_path)
    final_copy = dict(final)
    stored_final = final_copy.pop("final_digest")
    audit_copy = dict(audit)
    stored_audit = audit_copy.pop("audit_digest")
    if digest(final_copy) != stored_final:
        raise RuntimeError("Phase1185 final digest mismatch")
    if digest(audit_copy) != stored_audit or not audit["audit_pass"]:
        raise RuntimeError("Phase1185 audit invalid")
    return final, audit


def preregister() -> None:
    if PROTOCOL_PATH.exists():
        raise RuntimeError("Phase1186 already preregistered")
    if not AUDIT_SCRIPT.exists():
        raise RuntimeError("audit script must exist before registration")
    previous, previous_audit = validate_phase1185()
    protocol = {
        "phase": PHASE,
        "registered_at_utc": utc_now(),
        "scientific_object": "One-shot repair of the universal-Boolean reducer used by numerical qualification.",
        "single_changed_variable": {
            "phase1185": "FP32 mean(Boolean) >= 1.0",
            "phase1186": "integer agreement_count == integer eligible_count",
        },
        "frozen_factors": {
            "architecture": "RoleSquareNetwork",
            "modulus": p1185.MODULUS,
            "width": p1185.WIDTH,
            "scales": list(SCALES),
            "structures": list(STRUCTURES),
            "gauge_group": "signed hidden-channel permutation with exact output compensation",
            "transforms_per_system": TRANSFORMS,
            "error_thresholds": THRESHOLDS,
            "positive_control": "broken output compensation",
        },
        "freshness": {
            "new_model_seeds": True,
            "new_transform_seeds": True,
            "systems_per_scale_structure": REPLICATES,
            "known_truth_system_count": len(SCALES) * len(STRUCTURES) * REPLICATES,
            "gauge_row_count": len(SCALES) * len(STRUCTURES) * REPLICATES * TRANSFORMS,
        },
        "raw_count_contract": {
            "decision": "agree_count == eligible_count; empty eligible set abstains and passes equality only",
            "margin": "margin_sign_agree_count == margin_eligible_count; empty set abstains",
            "coverage": "not required for engineered scale strata, exactly as in Phase1185",
            "ratios": "reported descriptively from Python integer counts and never used for exact universal truth",
        },
        "claim_exclusions": [
            "Phase1185 remains a frozen formal failure.",
            "A Phase1186 pass qualifies only the corrected reducer on the declared known-truth domain.",
            "A pass does not confirm K165 or any natural mechanism.",
            "No result transfers directly to Transformers or language models.",
        ],
        "decision": {
            "reducer_sentinel": "all integer universal predicates pass",
            "gauge": "all 256 fresh gauge rows pass the unchanged algebraic and numerical contract",
            "positive": "at least 95 percent of 64 broken-compensation sentinels trigger",
            "pass_action": "close qualification development and authorize exactly one fresh three-evidence mechanism confirmation",
            "failure_action": "close reducer repair and do not retune this implementation family",
        },
        "phase1185_evidence": {
            "final_digest": previous["final_digest"],
            "audit_digest": previous_audit["audit_digest"],
            "natural_gauge_pass_fraction": previous["numerical_axis"]["natural_gauge_pass_fraction"],
            "support_coverage": previous["numerical_axis"]["safety_coverage"],
            "behavior_axis_pass": previous["behavior_axis"]["behavior_axis_pass"],
            "support_behavior_intersection_pass": previous["science_intersection"]["intersection_pass"],
        },
        "scripts": {
            "runner": file_sha256(SCRIPT),
            "audit": file_sha256(AUDIT_SCRIPT),
            "phase1185_runner": file_sha256(p1185.SCRIPT),
            "phase1185_final": file_sha256(p1185.FINAL_PATH),
            "phase1185_audit": file_sha256(p1185.OUT_ROOT / "audit/independent_audit.json"),
            "phase1183_source": file_sha256(Path(p1185.p1183.__file__)),
        },
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }
    protocol["protocol_digest"] = digest(protocol)
    write_json(PROTOCOL_PATH, protocol)
    print(canonical_json({"registered": str(PROTOCOL_PATH), "digest": protocol["protocol_digest"]}))


def validate_protocol() -> dict[str, Any]:
    protocol = read_json(PROTOCOL_PATH)
    copy = dict(protocol)
    stored = copy.pop("protocol_digest")
    if digest(copy) != stored:
        raise RuntimeError("protocol digest mismatch")
    paths = {
        "runner": SCRIPT,
        "audit": AUDIT_SCRIPT,
        "phase1185_runner": p1185.SCRIPT,
        "phase1185_final": p1185.FINAL_PATH,
        "phase1185_audit": p1185.OUT_ROOT / "audit/independent_audit.json",
        "phase1183_source": Path(p1185.p1183.__file__),
    }
    for name, path in paths.items():
        if file_sha256(path) != protocol["scripts"][name]:
            raise RuntimeError(f"frozen source changed: {name}")
    validate_phase1185()
    return protocol


def exact_boolean_reducer(equal: torch.Tensor, eligible: torch.Tensor) -> dict[str, Any]:
    if equal.dtype != torch.bool or eligible.dtype != torch.bool:
        raise TypeError("exact reducer requires Boolean tensors")
    eligible_count = int(torch.count_nonzero(eligible).item())
    agree_count = int(torch.count_nonzero(equal & eligible).item())
    return {
        "eligible_count": eligible_count,
        "agree_count": agree_count,
        "all_equal": agree_count == eligible_count,
        "abstained": eligible_count == 0,
        "descriptive_ratio": float(agree_count / eligible_count) if eligible_count else None,
    }


def exact_decision_metrics(
    reference: torch.Tensor,
    changed: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, Any]:
    threshold = THRESHOLDS
    max_abs = max(float(reference.abs().max().item()), 1e-30)
    uncertainty = 2.0 * (
        threshold["fp32_absolute_floor"] + threshold["fp32_relative_multiplier"] * max_abs
    )
    top = torch.topk(reference, k=2, dim=1).values
    decision_eligible = (top[:, 0] - top[:, 1]) > uncertainty
    decision_equal = reference.argmax(1) == changed.argmax(1)
    decision = exact_boolean_reducer(decision_equal, decision_eligible)
    margin_reference = p1185.p1181.correct_margin(reference, targets.to(reference.device))
    margin_changed = p1185.p1181.correct_margin(changed, targets.to(reference.device))
    margin_eligible = margin_reference.abs() > uncertainty
    margin_equal = (margin_reference >= 0) == (margin_changed >= 0)
    margin = exact_boolean_reducer(margin_equal, margin_eligible)
    return {"uncertainty_bound": uncertainty, "decision": decision, "margin_sign": margin}


def gauge_case(
    model,
    x: torch.Tensor,
    y: torch.Tensor,
    seed: int,
    device: torch.device,
    design: np.ndarray,
    design_pinv: np.ndarray,
) -> dict[str, Any]:
    transformed = p1185.p1183.gauge_model(model, seed, device)
    original32, _ = p1185.p1181.fp32_state(model, x, device)
    changed32, _ = p1185.p1181.fp32_state(transformed, x, device)
    _, original64 = p1185.p1183.cpu_hidden_and_logits(model, x)
    _, changed64 = p1185.p1183.cpu_hidden_and_logits(transformed, x)
    feature = np.asarray(p1185.p1183.algebraic_internal_features(model, x), dtype=np.float64)
    changed_feature = np.asarray(p1185.p1183.algebraic_internal_features(transformed, x), dtype=np.float64)
    fp32 = p1185.forward_metrics(
        original32.detach().cpu().double().numpy(), changed32.detach().cpu().double().numpy(), "fp32"
    )
    fp64 = p1185.forward_metrics(original64, changed64, "fp64")
    decision = exact_decision_metrics(original32, changed32, y)
    delta = changed32.detach().cpu().double().numpy() - original32.detach().cpu().double().numpy()
    delta_w_t = design_pinv @ delta
    reconstructed = design @ delta_w_t
    output_norm = float(np.linalg.norm(model.output.weight.detach().cpu().double().numpy()))
    feature_error = float(np.max(np.abs(feature - changed_feature)))
    passed = bool(
        feature_error <= THRESHOLDS["algebraic_feature_error_max"]
        and fp64["absolute_max"] <= fp64["mixed_absolute_bound"]
        and fp64["scaled_max"] <= THRESHOLDS["fp64_scaled_error_max"]
        and fp64["rms_relative"] <= THRESHOLDS["fp64_scaled_error_max"]
        and fp32["absolute_max"] <= fp32["mixed_absolute_bound"]
        and fp32["scaled_max"] <= THRESHOLDS["fp32_scaled_error_max"]
        and fp32["rms_relative"] <= THRESHOLDS["fp32_scaled_error_max"]
        and decision["decision"]["all_equal"]
        and decision["margin_sign"]["all_equal"]
    )
    result = {
        "seed": seed,
        "feature_error": feature_error,
        "fp64": fp64,
        "fp32": fp32,
        "exact_decision": decision,
        "restricted_output_backward_witness": {
            "relative_output_weight_norm": float(np.linalg.norm(delta_w_t.T) / max(output_norm, 1e-30)),
            "relative_reconstruction_residual": p1185.rms(delta - reconstructed) / max(p1185.rms(delta), 1e-30),
            "claim_status": "descriptive_not_full_network_backward_error",
        },
        "gauge_pass": passed,
    }
    del transformed
    return result


def reducer_sentinels(device: torch.device) -> list[dict[str, Any]]:
    rows = []
    for length in REDUCER_SENTINEL_LENGTHS:
        values = torch.ones(length, dtype=torch.bool, device=device)
        exact = exact_boolean_reducer(values, values)
        old_fp32_mean = float(values.float().mean().item())
        rows.append(
            {
                "length": length,
                "eligible_count": exact["eligible_count"],
                "agree_count": exact["agree_count"],
                "exact_all_equal": exact["all_equal"],
                "old_fp32_mean": old_fp32_mean,
                "old_exact_one": old_fp32_mean >= 1.0,
            }
        )
    return rows


def run() -> None:
    protocol = validate_protocol()
    if FINAL_PATH.exists():
        raise RuntimeError("Phase1186 already finalized")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    x = torch.tensor(
        [(a, b) for a in range(p1185.MODULUS) for b in range(p1185.MODULUS)], dtype=torch.long
    )
    y = torch.tensor(p1185.task_table("axis_affine_b").reshape(-1), dtype=torch.long)
    sentinels = reducer_sentinels(device)
    gauge_rows: list[dict[str, Any]] = []
    positive_rows: list[dict[str, Any]] = []
    case_index = 0
    for scale_index, scale in enumerate(SCALES):
        for structure_index, structure in enumerate(STRUCTURES):
            for replicate in range(REPLICATES):
                seed = model_seed(scale_index, structure_index, replicate)
                model = p1185.engineered_model(scale, structure, seed, device)
                _, hidden32 = p1185.p1181.fp32_state(model, x, device)
                design = hidden32.detach().cpu().double().numpy() ** 2
                design_pinv = np.linalg.pinv(design, rcond=1e-10)
                for transform in range(TRANSFORMS):
                    gauge_rows.append(
                        {
                            "case": case_index,
                            "scale": scale,
                            "structure": structure,
                            "replicate": replicate,
                            "model_seed": seed,
                            "transform": transform,
                            **gauge_case(
                                model,
                                x,
                                y,
                                transform_seed(case_index, transform),
                                device,
                                design,
                                design_pinv,
                            ),
                        }
                    )
                positive_rows.append(
                    {
                        "case": case_index,
                        "scale": scale,
                        "structure": structure,
                        "replicate": replicate,
                        "model_seed": seed,
                        **p1185.positive_control(
                            model,
                            x,
                            positive_seed(case_index),
                            device,
                            require_decision_difference=False,
                        ),
                    }
                )
                print(canonical_json({"case": case_index, "scale": scale, "structure": structure, "replicate": replicate}), flush=True)
                del model
                case_index += 1
                gc.collect()
                torch.cuda.empty_cache()

    sentinel_path = OUT_ROOT / "analysis/reducer_sentinels.jsonl"
    gauge_path = OUT_ROOT / "analysis/gauge_rows.jsonl"
    positive_path = OUT_ROOT / "analysis/positive_control_rows.jsonl"
    write_jsonl(sentinel_path, sentinels)
    write_jsonl(gauge_path, gauge_rows)
    write_jsonl(positive_path, positive_rows)
    gauge_fraction = sum(row["gauge_pass"] for row in gauge_rows) / len(gauge_rows)
    positive_fraction = sum(row["positive_control_pass"] for row in positive_rows) / len(positive_rows)
    sentinel_pass = all(row["exact_all_equal"] for row in sentinels)
    reducer_qualification_pass = bool(
        sentinel_pass
        and gauge_fraction == 1.0
        and positive_fraction >= THRESHOLDS["positive_control_pass_fraction_min"]
    )
    previous, previous_audit = validate_phase1185()
    preaudit_authorized = bool(
        reducer_qualification_pass
        and previous_audit["audit_pass"]
        and previous["numerical_axis"]["natural_gauge_pass_fraction"] == 1.0
        and previous["numerical_axis"]["safety_coverage"] == 1.0
        and previous["behavior_axis"]["behavior_axis_pass"]
        and previous["science_intersection"]["intersection_pass"]
    )
    all_exact = [row["exact_decision"] for row in gauge_rows]
    final = {
        "phase": PHASE,
        "created_at_utc": utc_now(),
        "protocol_digest": protocol["protocol_digest"],
        "scientific_object": "reducer_safe_numerical_qualification_repair_only",
        "phase1185_formal_verdict_unchanged": True,
        "system_count": case_index,
        "gauge_row_count": len(gauge_rows),
        "positive_row_count": len(positive_rows),
        "reducer_sentinel_pass": sentinel_pass,
        "old_reducer_failure_lengths": [row["length"] for row in sentinels if not row["old_exact_one"]],
        "gauge_pass_fraction": gauge_fraction,
        "positive_control_pass_fraction": positive_fraction,
        "maximum_feature_error": max(row["feature_error"] for row in gauge_rows),
        "maximum_fp64_scaled_error": max(row["fp64"]["scaled_max"] for row in gauge_rows),
        "maximum_fp32_scaled_error": max(row["fp32"]["scaled_max"] for row in gauge_rows),
        "minimum_nonzero_decision_eligible_count": min(
            row["decision"]["eligible_count"]
            for row in all_exact
            if row["decision"]["eligible_count"] > 0
        ),
        "minimum_nonzero_margin_eligible_count": min(
            row["margin_sign"]["eligible_count"]
            for row in all_exact
            if row["margin_sign"]["eligible_count"] > 0
        ),
        "all_decision_integer_equalities": all(row["decision"]["all_equal"] for row in all_exact),
        "all_margin_integer_equalities": all(row["margin_sign"]["all_equal"] for row in all_exact),
        "reducer_qualification_pass": reducer_qualification_pass,
        "mechanism_camera_status": "not_tested_by_design",
        "phase1187_authorized_before_audit": preaudit_authorized,
        "auto_continue": {
            "authorized_before_audit": preaudit_authorized,
            "next": "one_fresh_three_evidence_mechanism_confirmation" if preaudit_authorized else None,
        },
        "artifacts": {
            "sentinels_sha256": file_sha256(sentinel_path),
            "gauge_rows_sha256": file_sha256(gauge_path),
            "positive_rows_sha256": file_sha256(positive_path),
        },
    }
    final["final_digest"] = digest(final)
    write_json(FINAL_PATH, final)
    print(canonical_json(final))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preregister", "run"))
    args = parser.parse_args()
    if args.command == "preregister":
        preregister()
    else:
        run()


if __name__ == "__main__":
    main()
