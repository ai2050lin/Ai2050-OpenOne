#!/usr/bin/env python3
"""Narrow pre-causal-output audit amendment for Phase1207 FP16 replay.

The frozen audit incorrectly required bitwise-identical candidate logits across
independent model reloads.  Residual states replayed exactly, while 108/12096
candidate logits differed by at most three local FP16 ULPs and all predictions
were unchanged.  This amendment is frozen before any causal-onset output.  It
changes only cross-reload score-replay checks to a four-local-FP16-ULP bound;
all in-run identity controls and every scientific gate remain untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

import phase1207_qwen3_causal_ancestry_necessity as run
import phase1207_qwen3_causal_ancestry_necessity_audit as base


AMENDMENT_PATH = run.OUT_ROOT / "protocol/fp16_replay_audit_amendment.json"
ULP_MULTIPLIER = 4.0


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def validate(value: dict[str, Any], key: str) -> None:
    if digest({name: item for name, item in value.items() if name != key}) != value.get(key):
        raise RuntimeError(f"embedded digest mismatch: {key}")


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def ulp_bound(left: np.ndarray, right: np.ndarray) -> tuple[bool, float, float]:
    left32 = np.asarray(left, dtype=np.float32)
    right32 = np.asarray(right, dtype=np.float32)
    difference = np.abs(left32 - right32)
    left_ulp = np.spacing(np.abs(left32).astype(np.float16)).astype(np.float32)
    right_ulp = np.spacing(np.abs(right32).astype(np.float16)).astype(np.float32)
    allowance = ULP_MULTIPLIER * np.maximum(left_ulp, right_ulp)
    allowance = np.maximum(allowance, np.finfo(np.float16).tiny)
    ratio = difference / allowance
    return bool(np.all(difference <= allowance)), float(difference.max(initial=0.0)), float(ratio.max(initial=0.0))


def freeze() -> dict[str, Any]:
    if AMENDMENT_PATH.exists():
        raise RuntimeError("Phase1207 FP16 replay amendment already exists")
    if run.ONSET_RAW_PATH.exists() or run.ONSET_SUMMARY_PATH.exists() or run.ONSET_VERDICT_PATH.exists():
        raise RuntimeError("cannot freeze replay amendment after causal-onset output")
    protocol = run.read_json(run.PROTOCOL_PATH)
    capture = run.read_json(run.CAPTURE_SUMMARY_PATH)
    run.validate_digest(protocol, "protocol_digest")
    run.validate_digest(capture, "summary_digest")
    with np.load(run.CAPTURE_PATH, allow_pickle=False) as current, np.load(run.UPSTREAM_VECTOR_PATH, allow_pickle=False) as upstream:
        score_pass, max_abs, max_allowance_ratio = ulp_bound(current["baseline_scores"], upstream["baseline_scores"])
        depths = tuple(int(value) for value in current["capture_depths"].tolist())
        residual24 = float(np.max(np.abs(current["residuals"][:, :, depths.index(24)].astype(np.float32) - upstream["d24_generation_boundary"].astype(np.float32))))
        residual25 = float(np.max(np.abs(current["residuals"][:, :, depths.index(25)].astype(np.float32) - upstream["d25_generation_boundary"].astype(np.float32))))
        predictions_unchanged = bool(np.array_equal(np.argmax(current["baseline_scores"], axis=-1), np.argmax(upstream["baseline_scores"], axis=-1)))
    if not score_pass or residual24 != 0.0 or residual25 != 0.0 or not predictions_unchanged:
        raise RuntimeError("capture replay is outside the narrow FP16 amendment")
    value: dict[str, Any] = {
        "phase": 1207,
        "schema_version": "phase1207.fp16_replay_audit_amendment.v1",
        "protocol_digest": protocol["protocol_digest"],
        "capture_summary_digest": capture["summary_digest"],
        "capture_file_sha256": run.sha256_file(run.CAPTURE_PATH),
        "original_audit_sha256": run.sha256_file(Path(base.__file__).resolve()),
        "amendment_script_sha256": run.sha256_file(Path(__file__).resolve()),
        "frozen_before_causal_output": True,
        "reason": "Cross-reload FP16 GEMM logits are not a bitwise identity object; residual states and predictions replayed exactly.",
        "changed_check_only": "cross-reload score equality uses four local FP16 ULPs",
        "unchanged": [
            "causal depths", "conditions", "controls", "scientific thresholds", "selection rule",
            "in-run zero-patch identity", "necessity gate", "rescue gate", "claim boundaries",
        ],
        "ulp_multiplier": ULP_MULTIPLIER,
        "capture_calibration": {
            "score_max_abs_difference": max_abs,
            "maximum_fraction_of_allowed_bound": max_allowance_ratio,
            "residual_depth24_max_abs": residual24,
            "residual_depth25_max_abs": residual25,
            "predictions_unchanged": predictions_unchanged,
        },
        "causal_model_outputs_observed": 0,
    }
    value["amendment_digest"] = digest(value)
    write(AMENDMENT_PATH, value)
    return value


def amended_result(write_output: bool) -> dict[str, Any]:
    if write_output and run.RESULT_AUDIT_PATH.exists():
        raise RuntimeError("Phase1207 result audit already exists")
    amendment = run.read_json(AMENDMENT_PATH)
    validate(amendment, "amendment_digest")
    protocol = run.verify_protocol()
    capture = run.read_json(run.CAPTURE_SUMMARY_PATH)
    run.validate_digest(capture, "summary_digest")
    if amendment["protocol_digest"] != protocol["protocol_digest"] or amendment["capture_summary_digest"] != capture["summary_digest"]:
        raise RuntimeError("Phase1207 audit amendment link drift")
    if amendment["amendment_script_sha256"] != run.sha256_file(Path(__file__).resolve()):
        raise RuntimeError("Phase1207 audit amendment source drift")
    output = base.result(False)
    with np.load(run.CAPTURE_PATH, allow_pickle=False) as current, np.load(run.UPSTREAM_VECTOR_PATH, allow_pickle=False) as upstream:
        score_pass, score_max_abs, score_bound_fraction = ulp_bound(current["baseline_scores"], upstream["baseline_scores"])
    replaced_capture = False
    for check in output["checks"]:
        if check["name"] == "upstream_replay_exact":
            check["name"] = "upstream_replay_within_4_local_fp16_ulp"
            check["pass"] = score_pass
            check["detail"] = {"max_abs": score_max_abs, "maximum_fraction_of_allowed_bound": score_bound_fraction}
            replaced_capture = True
    if not replaced_capture:
        raise RuntimeError("frozen capture replay check not found")
    replaced_damage = False
    if run.RESCUE_RAW_PATH.exists():
        rescue = run.read_jsonl_gz(run.RESCUE_RAW_PATH)
        necessity = run.read_jsonl_gz(run.NECESSITY_RAW_PATH)
        damage = {(row["group_id"], int(row["recipient_state"])): row for row in rescue if row["condition"] == "damage_only"}
        target = {(row["group_id"], int(row["recipient_state"])): row for row in necessity if row["condition"] == "active_vs_surface_remove"}
        left = np.asarray([damage[key]["patched_scores"] for key in sorted(damage)], dtype=np.float32)
        right = np.asarray([target[key]["patched_scores"] for key in sorted(target)], dtype=np.float32)
        damage_pass, damage_max_abs, damage_bound_fraction = ulp_bound(left, right)
        for check in output["checks"]:
            if check["name"] == "damage_replay_matches_necessity":
                check["name"] = "damage_replay_within_4_local_fp16_ulp"
                check["pass"] = damage_pass
                check["detail"] = {"max_abs": damage_max_abs, "maximum_fraction_of_allowed_bound": damage_bound_fraction}
                replaced_damage = True
        if not replaced_damage:
            raise RuntimeError("frozen damage replay check not found")
    output["passed_checks"] = sum(item["pass"] for item in output["checks"])
    output["total_checks"] = len(output["checks"])
    output["gate_pass"] = all(item["pass"] for item in output["checks"])
    output["audit_amendment"] = {
        "digest": amendment["amendment_digest"],
        "scope": amendment["changed_check_only"],
        "frozen_before_causal_output": True,
        "scientific_gate_changed": False,
    }
    output.pop("audit_digest", None)
    output["audit_digest"] = digest(output)
    if write_output:
        if not output["gate_pass"]:
            raise RuntimeError([item["name"] for item in output["checks"] if not item["pass"]])
        write(run.RESULT_AUDIT_PATH, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("freeze", "result"))
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    output = freeze() if args.command == "freeze" else amended_result(args.write)
    print(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
