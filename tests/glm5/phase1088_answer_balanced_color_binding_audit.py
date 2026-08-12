#!/usr/bin/env python3
"""Audit Phase1088 answer-balanced binding result integrity."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1088_answer_balanced_color_binding_protocol as protocol


ANALYSIS = {
    "shared_binding_field_audit.json": "shared_binding_field_digest",
    "surface_control_audit.json": "surface_control_digest",
    "color_pair_residual_audit.json": "color_pair_residual_digest",
    "heldout_color_pair_audit.json": "heldout_color_pair_digest",
    "cross_model_geometry.json": "cross_model_geometry_digest",
    "signed_decomposition.json": "signed_decomposition_digest",
    "physical_map.json": "physical_map_digest",
    "projection_audit.json": "projection_gate_digest",
    "numeric_audit.json": "numeric_audit_digest",
    "prediction_audit.json": "prediction_audit_digest",
    "automatic_next.json": "automatic_next_digest",
    "final_summary.json": "summary_digest",
    "pair_fingerprint_posthoc_map.json": "pair_fingerprint_map_digest",
}


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def valid_digest(payload: dict[str, Any], key: str) -> bool:
    expected = payload.get(key)
    unsigned = {name: value for name, value in payload.items() if name != key}
    return expected == protocol.digest(unsigned)


def main() -> None:
    protocol_root = protocol.OUT_ROOT / "protocol"
    analysis_root = protocol.OUT_ROOT / "analysis"
    prereg = protocol.read_json(protocol_root / "preregistration.json")
    static = protocol.read_json(protocol_root / "audit.json")
    authorization = protocol.read_json(
        analysis_root / "behavior_authorization.json"
    )
    final = protocol.read_json(analysis_root / "final_summary.json")
    checks = {
        "protocol_digest": valid_digest(prereg, "protocol_digest"),
        "protocol_audit_digest": valid_digest(static, "audit_digest"),
        "protocol_static_checks": bool(static["all_checks_passed"]),
        "authorization_digest": valid_digest(
            authorization, "authorization_digest"
        ),
        "hidden_scan_authorized": bool(authorization["hidden_scan_authorized"]),
        "final_summary_digest": valid_digest(final, "summary_digest"),
        "final_protocol_digest": final["protocol_digest"] == prereg["protocol_digest"],
        "prediction_partition": (
            set(final["passed_predictions"]).isdisjoint(
                final["failed_predictions"]
            )
            and set(final["passed_predictions"]).union(
                final["failed_predictions"]
            ) == {f"P{index}" for index in range(1, 11)}
        ),
    }
    model_rows = {}
    for model_name in protocol.MODELS:
        root = protocol.OUT_ROOT / "atlas" / model_name
        summary = protocol.read_json(root / "summary.json")
        projection = protocol.read_json(root / "projection_audit.json")
        path = root / "signed_fields.npz"
        with np.load(path) as archive:
            expected = (
                len(protocol.FAMILIES), len(protocol.SPLITS),
                int(summary["event_count"]), len(protocol.CAPTURE_ROLES),
                len(protocol.SIGNED_FIELDS), 2, 2,
                protocol.SIGNED_PROJECTION_REPLICATES,
                protocol.SIGNED_PROJECTION_DIM,
            )
            content = protocol.SIGNED_FIELDS.index("content")
            pre = [
                protocol.CAPTURE_ROLES.index(role)
                for role in protocol.PRE_QUERY_ROLES
            ]
            row = {
                "npz_keys": set(archive.files) == {
                    "direction_sum", "direction_count", "relative_sum",
                    "relative_count", "surface_relative_sum",
                    "surface_relative_count", "output_relative_sum",
                    "output_relative_count",
                },
                "shape": archive["direction_sum"].shape == expected,
                "arrays_finite": all(bool(np.all(np.isfinite(archive[key]))) for key in (
                    "direction_sum", "relative_sum", "surface_relative_sum",
                    "output_relative_sum",
                )),
                "prequery_exact": (
                    float(np.max(np.abs(
                        archive["direction_sum"][:, :, :, pre, content]
                    ))) == 0.0
                    and float(np.max(np.abs(
                        archive["relative_sum"][:, :, :, pre, content]
                    ))) == 0.0
                ),
            }
        row.update({
            "summary_digest": valid_digest(summary, "summary_digest"),
            "projection_digest": valid_digest(
                projection, "projection_audit_digest"
            ),
            "protocol_matches": summary["protocol_digest"] == prereg["protocol_digest"],
            "npz_hash": sha256(path) == final["models"][model_name]["npz_sha256"],
            "identity_exact": summary["identity_maximum"] == 0.0,
        })
        row["all_checks_passed"] = all(row.values())
        model_rows[model_name] = row
        checks[f"model_{model_name}"] = row["all_checks_passed"]

    analysis_rows = {}
    for filename, key in ANALYSIS.items():
        payload = protocol.read_json(analysis_root / filename)
        row = {
            "digest": valid_digest(payload, key),
            "phase": payload.get("phase") == protocol.PHASE,
            "protocol": payload.get("protocol_digest") == prereg["protocol_digest"],
        }
        row["all_checks_passed"] = all(row.values())
        analysis_rows[filename] = row
        checks[f"analysis_{filename}"] = row["all_checks_passed"]
    automatic = protocol.read_json(analysis_root / "automatic_next.json")
    checks["failed_gate_restriction"] = (
        not final["failed_predictions"]
        or (
            not automatic["full_atlas_authorized"]
            and not automatic["local_causal_authorized"]
        )
    )
    result = {
        "schema_version": "phase1088_result_integrity_audit.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "checks": checks,
        "models": model_rows,
        "analysis_outputs": analysis_rows,
        "all_checks_passed": all(checks.values()),
    }
    result["result_integrity_digest"] = protocol.digest(result)
    protocol.write_json(
        analysis_root / "result_integrity_audit.json", result
    )
    print({
        "phase": protocol.PHASE,
        "all_checks_passed": result["all_checks_passed"],
        "failed_checks": [name for name, value in checks.items() if not value],
        "result_integrity_digest": result["result_integrity_digest"],
    })
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
