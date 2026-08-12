#!/usr/bin/env python3
"""Audit Phase1094 artifacts, digests, array shapes, and sequential execution."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1094_semantic_topology_protocol as protocol


def valid_embedded_digest(row: dict[str, Any], key: str) -> bool:
    expected = str(row[key])
    body = dict(row)
    body.pop(key, None)
    return protocol.digest(body) == expected


def line_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def main() -> None:
    root = protocol.OUT_ROOT
    prereg = protocol.read_json(root / "protocol" / "preregistration.json")
    static = protocol.read_json(root / "protocol" / "audit.json")
    behavior = protocol.read_json(root / "analysis" / "behavior_authorization.json")
    final = protocol.read_json(root / "analysis" / "final_summary.json")
    checks: dict[str, bool] = {
        "preregistration_digest": valid_embedded_digest(prereg, "protocol_digest"),
        "static_audit_digest": valid_embedded_digest(static, "audit_digest"),
        "behavior_digest": valid_embedded_digest(behavior, "summary_digest"),
        "final_summary_digest": valid_embedded_digest(final, "summary_digest"),
        "protocol_digest_chain": (
            static["protocol_digest"] == prereg["protocol_digest"]
            and behavior["protocol_digest"] == prereg["protocol_digest"]
            and final["protocol_digest"] == prereg["protocol_digest"]
        ),
        "static_passed": bool(static["all_checks_passed"]),
        "hidden_scan_was_authorized": bool(behavior["hidden_scan_authorized"]),
        "causal_not_authorized": not bool(final["causal_authorized"]),
        "all_prediction_values_boolean": all(
            isinstance(row["passed"], bool) for row in final["predictions"].values()
        ),
    }

    model_rows = {}
    pilot_times = []
    atlas_times = []
    for model_name in protocol.MODELS:
        case_path = root / "protocol" / f"cases.{model_name}.jsonl"
        pilot_path = root / "pilot" / f"{model_name}.json"
        atlas_root = root / "atlas" / model_name
        summary_path = atlas_root / "summary.json"
        npz_path = atlas_root / "signed_fields.npz"
        pilot = protocol.read_json(pilot_path)
        summary = protocol.read_json(summary_path)
        with np.load(npz_path) as archive:
            shapes = {key: list(archive[key].shape) for key in archive.files}
            arrays_finite = all(np.isfinite(archive[key]).all() for key in archive.files)
            direction_shape = archive["direction_sum"].shape
            count_shape = archive["direction_count"].shape
        expected_direction = (
            len(protocol.FAMILIES), len(protocol.SPLITS), int(summary["event_count"]),
            len(protocol.CAPTURE_ROLES), len(protocol.SIGNED_FIELDS),
            len(protocol.TEMPLATE_IDS), len(protocol.OUTPUT_SET_IDS),
            protocol.SIGNED_PROJECTION_REPLICATES, protocol.SIGNED_PROJECTION_DIM,
        )
        expected_count = expected_direction[:-1]
        model_checks = {
            "case_line_count": line_count(case_path) == int(prereg["case_count_per_model"]),
            "pilot_digest": valid_embedded_digest(pilot, "result_digest"),
            "summary_digest": valid_embedded_digest(summary, "summary_digest"),
            "pilot_protocol_digest": pilot["protocol_digest"] == prereg["protocol_digest"],
            "atlas_protocol_digest": summary["protocol_digest"] == prereg["protocol_digest"],
            "case_digest_chain": (
                pilot["case_digest"] == prereg["model_case_digests"][model_name]
                and summary["case_digest"] == prereg["model_case_digests"][model_name]
            ),
            "direction_shape": tuple(direction_shape) == expected_direction,
            "count_shape": tuple(count_shape) == expected_count,
            "stored_arrays_finite": bool(arrays_finite),
            "fp16_no_quantization": (
                summary["precision"]["has_fp16_parameters"]
                and not summary["precision"]["has_bf16_parameters"]
                and not summary["precision"]["has_quantized_modules"]
            ),
        }
        model_rows[model_name] = {
            "checks": model_checks,
            "all_checks_passed": all(model_checks.values()),
            "array_shapes": shapes,
            "pilot_mtime": pilot_path.stat().st_mtime,
            "atlas_mtime": summary_path.stat().st_mtime,
        }
        pilot_times.append(pilot_path.stat().st_mtime)
        atlas_times.append(summary_path.stat().st_mtime)
    checks["all_model_artifacts_passed"] = all(
        row["all_checks_passed"] for row in model_rows.values()
    )
    checks["pilot_model_order_sequential"] = pilot_times == sorted(pilot_times)
    checks["atlas_model_order_sequential"] = atlas_times == sorted(atlas_times)
    checks["all_checks_boolean"] = all(isinstance(value, bool) for value in checks.values())
    result = {
        "schema_version": "phase1094_result_audit.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "checks": checks,
        "models": model_rows,
        "all_checks_passed": all(checks.values()),
    }
    result["result_audit_digest"] = protocol.digest(result)
    protocol.write_json(root / "analysis" / "result_audit.json", result)
    print({
        "phase": protocol.PHASE,
        "all_checks_passed": result["all_checks_passed"],
        "failed_checks": [key for key, value in checks.items() if not value],
        "result_audit_digest": result["result_audit_digest"],
    })


if __name__ == "__main__":
    main()
