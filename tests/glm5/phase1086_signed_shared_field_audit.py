#!/usr/bin/env python3
"""Audit Phase1086 protocol, model arrays, and frozen analysis outputs."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1086_signed_shared_field_protocol as protocol


MODEL_NAMES = tuple(protocol.MODELS)
ANALYSIS_DIGEST_KEYS = {
    "shared_field_audit.json": "shared_field_digest",
    "surface_control_audit.json": "surface_control_digest",
    "attribute_residual_audit.json": "attribute_residual_digest",
    "cross_model_geometry.json": "cross_model_geometry_digest",
    "signed_decomposition.json": "signed_decomposition_digest",
    "physical_map.json": "physical_map_digest",
    "prediction_audit.json": "prediction_audit_digest",
    "automatic_next.json": "automatic_next_digest",
    "final_summary.json": "summary_digest",
}


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def digest_valid(payload: dict[str, Any], key: str) -> bool:
    expected = payload.get(key)
    unsigned = {name: value for name, value in payload.items() if name != key}
    return expected == protocol.digest(unsigned)


def main() -> None:
    protocol_root = protocol.OUT_ROOT / "protocol"
    analysis_root = protocol.OUT_ROOT / "analysis"
    prereg = protocol.read_json(protocol_root / "preregistration.json")
    protocol_audit = protocol.read_json(protocol_root / "audit.json")
    behavior = protocol.read_json(analysis_root / "behavior_authorization.json")
    final = protocol.read_json(analysis_root / "final_summary.json")

    checks: dict[str, bool] = {
        "protocol_digest": digest_valid(prereg, "protocol_digest"),
        "protocol_audit_digest": digest_valid(protocol_audit, "audit_digest"),
        "protocol_static_checks": bool(protocol_audit["all_checks_passed"]),
        "behavior_authorization_digest": digest_valid(
            behavior, "authorization_digest"
        ),
        "behavior_scan_authorized": bool(behavior["hidden_scan_authorized"]),
        "final_summary_digest": digest_valid(final, "summary_digest"),
        "final_protocol_digest": final["protocol_digest"] == prereg["protocol_digest"],
        "case_count": final["case_count_per_model"] == 12288,
        "unit_count": final["unit_count_per_model"] == 384,
    }
    model_rows = {}
    file_hashes = {}
    for model_name in MODEL_NAMES:
        atlas_root = protocol.OUT_ROOT / "atlas" / model_name
        summary_path = atlas_root / "summary.json"
        projection_path = atlas_root / "projection_audit.json"
        npz_path = atlas_root / "signed_fields.npz"
        summary = protocol.read_json(summary_path)
        projection = protocol.read_json(projection_path)
        with np.load(npz_path) as archive:
            keys = set(archive.files)
            required = {
                "direction_sum", "direction_count", "relative_sum",
                "relative_count", "surface_relative_sum",
                "surface_relative_count", "output_relative_sum",
                "output_relative_count",
            }
            key_check = keys == required
            direction_shape = archive["direction_sum"].shape
            expected_shape = (
                len(protocol.FAMILIES), len(protocol.SPLITS),
                int(summary["event_count"]), len(protocol.CAPTURE_ROLES),
                len(protocol.SIGNED_FIELDS), 2, 2,
                protocol.SIGNED_PROJECTION_REPLICATES,
                protocol.SIGNED_PROJECTION_DIM,
            )
            shape_check = direction_shape == expected_shape
            arrays_finite = all(
                bool(np.all(np.isfinite(archive[key])))
                for key in (
                    "direction_sum", "relative_sum", "surface_relative_sum",
                    "output_relative_sum",
                )
            )
            counts_nonnegative = all(
                bool(np.all(archive[key] >= 0))
                for key in (
                    "direction_count", "relative_count",
                    "surface_relative_count", "output_relative_count",
                )
            )
        projection_digest = digest_valid(
            projection, "projection_audit_digest"
        )
        row = {
            "summary_digest_valid": digest_valid(summary, "summary_digest"),
            "projection_digest_valid": projection_digest,
            "protocol_digest_matches": (
                summary["protocol_digest"] == prereg["protocol_digest"]
            ),
            "npz_keys_exact": key_check,
            "npz_shape_valid": shape_check,
            "npz_arrays_finite": arrays_finite,
            "npz_counts_nonnegative": counts_nonnegative,
            "npz_sha256_matches_final": (
                file_sha256(npz_path)
                == final["models"][model_name]["npz_sha256"]
            ),
            "pre_query_exact": summary["pre_query_global_max_abs"] == 0.0,
            "identity_exact": summary["identity_maximum"] == 0.0,
        }
        row["all_checks_passed"] = all(row.values())
        model_rows[model_name] = row
        file_hashes[str(summary_path.relative_to(ROOT))] = file_sha256(summary_path)
        file_hashes[str(projection_path.relative_to(ROOT))] = file_sha256(projection_path)
        file_hashes[str(npz_path.relative_to(ROOT))] = file_sha256(npz_path)
        checks[f"model_{model_name}"] = row["all_checks_passed"]

    analysis_rows = {}
    for filename, digest_key in ANALYSIS_DIGEST_KEYS.items():
        path = analysis_root / filename
        payload = protocol.read_json(path)
        row = {
            "exists": path.exists(),
            "digest_valid": digest_valid(payload, digest_key),
            "phase_valid": payload.get("phase") == protocol.PHASE,
        }
        if filename != "automatic_next.json":
            row["protocol_digest_matches"] = (
                payload.get("protocol_digest") == prereg["protocol_digest"]
            )
        row["all_checks_passed"] = all(row.values())
        analysis_rows[filename] = row
        checks[f"analysis_{filename}"] = row["all_checks_passed"]
        file_hashes[str(path.relative_to(ROOT))] = file_sha256(path)

    automatic = protocol.read_json(analysis_root / "automatic_next.json")
    failed = set(final["failed_predictions"])
    checks["failed_gate_restriction"] = (
        bool(failed.intersection({"P3", "P4", "P5", "P6", "P7", "P8", "P10"}))
        and not automatic["full_atlas_authorized"]
        and not automatic["local_causal_authorized"]
    )
    checks["prediction_lists_partition"] = (
        set(final["passed_predictions"]).isdisjoint(failed)
        and set(final["passed_predictions"]).union(failed)
        == {f"P{index}" for index in range(1, 11)}
    )

    result = {
        "schema_version": "phase1086_result_integrity_audit.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "checks": checks,
        "models": model_rows,
        "analysis_outputs": analysis_rows,
        "file_sha256": file_hashes,
        "all_checks_passed": all(checks.values()),
    }
    result["result_integrity_digest"] = protocol.digest(result)
    protocol.write_json(analysis_root / "result_integrity_audit.json", result)
    print({
        "phase": protocol.PHASE,
        "all_checks_passed": result["all_checks_passed"],
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "result_integrity_digest": result["result_integrity_digest"],
    })
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
