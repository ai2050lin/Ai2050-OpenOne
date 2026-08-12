#!/usr/bin/env python3
"""Audit Phase1022 protocol, behavior, matching, scans, and analysis."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1022_ability_family_protocol as protocol


def file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            hasher.update(block)
    return hasher.hexdigest()


def npz_audit(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        count_names = [name for name in data.files if name.endswith("_count")]
        direction_names = [
            name for name in data.files if name.endswith("_mean_direction")
        ]
        magnitude_names = [
            name
            for name in data.files
            if name.endswith("_mean_normalized_magnitude")
        ]
        shapes = {
            name: list(data[name].shape)
            for name in count_names + direction_names + magnitude_names
        }
        shape_pairs = []
        for prefix in ("whole", "head", "key", "value"):
            count_name = f"{prefix}_count"
            direction_name = f"{prefix}_mean_direction"
            magnitude_name = f"{prefix}_mean_normalized_magnitude"
            shape_pairs.append(bool(
                count_name in data
                and direction_name in data
                and magnitude_name in data
                and data[direction_name].shape[:-1]
                == data[count_name].shape
                == data[magnitude_name].shape
            ))
        finite_magnitudes = {
            name: int(np.isfinite(data[name]).sum())
            for name in magnitude_names
        }
        positive_counts = {
            name: int((data[name] > 0).sum())
            for name in count_names
        }
    return {
        "path": str(path.relative_to(protocol.OUT_ROOT)),
        "shapes": shapes,
        "shape_pairs_valid": all(shape_pairs),
        "finite_magnitudes": finite_magnitudes,
        "positive_counts": positive_counts,
        "all_checks_passed": bool(
            all(shape_pairs)
            and all(value > 0 for value in finite_magnitudes.values())
            and all(value > 0 for value in positive_counts.values())
        ),
    }


def main() -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    protocol_audit = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "audit.json"
    )
    pairing = protocol.read_json(
        protocol.OUT_ROOT / "pairing" / "summary.json"
    )
    pairing_audit = protocol.read_json(
        protocol.OUT_ROOT / "pairing" / "audit.json"
    )
    analysis = protocol.read_json(
        protocol.OUT_ROOT / "analysis" / "summary.json"
    )

    behavior_checks = {}
    behavior_case_keys = {}
    for model in protocol.MODELS:
        summary = protocol.read_json(
            protocol.OUT_ROOT / "behavior" / model / "summary.json"
        )
        rows = protocol.read_jsonl(
            protocol.OUT_ROOT / "behavior" / model / "formal.jsonl"
        )
        behavior_case_keys[model] = {row["case_key"] for row in rows}
        behavior_checks[model] = {
            "protocol_digest_match": (
                summary["protocol_digest"] == prereg["protocol_digest"]
            ),
            "precision_bf16": summary["precision"] == "bf16",
            "no_quantization": summary["quantization"] == "none",
            "case_count_match": (
                int(summary["case_count"]) == len(rows) == 2472
            ),
            "unique_case_keys": (
                len(rows) == len({row["case_key"] for row in rows})
            ),
            "generated_ids_present": all(
                isinstance(row["generated_token_ids"], list) for row in rows
            ),
        }
        behavior_checks[model]["all_checks_passed"] = all(
            behavior_checks[model].values()
        )

    scan_checks = {}
    npz_checks = []
    for model in protocol.MODELS:
        summary = protocol.read_json(
            protocol.OUT_ROOT / "internal_scan" / model / "summary.json"
        )
        events = protocol.read_json(
            protocol.OUT_ROOT / "internal_scan" / model / "events.json"
        )
        model_npz = sorted(
            (protocol.OUT_ROOT / "internal_scan" / model).rglob("*.npz")
        )
        model_npz_checks = [npz_audit(path) for path in model_npz]
        npz_checks.extend(model_npz_checks)
        scan_checks[model] = {
            "protocol_digest_match": (
                summary["protocol_digest"] == prereg["protocol_digest"]
            ),
            "pairing_digest_match": (
                summary["pairing_digest"] == pairing["pairing_digest"]
            ),
            "precision_bf16": summary["precision"] == "bf16",
            "no_quantization": summary["quantization"] == "none",
            "kv_width_positive": int(events["kv_width"]) > 0,
            "all_component_groups_present": all(
                check["shape_pairs_valid"] for check in model_npz_checks
            ),
            "npz_count": len(model_npz),
        }
        scan_checks[model]["all_checks_passed"] = bool(
            all(
                value
                for key, value in scan_checks[model].items()
                if key not in ("npz_count",)
            )
            and len(model_npz) >= 20
        )

    common_keys_equal = (
        behavior_case_keys["qwen3"]
        == behavior_case_keys["glm4"]
        == behavior_case_keys["deepseek7b"]
    )
    all_files = sorted(
        path
        for path in protocol.OUT_ROOT.rglob("*")
        if path.is_file()
    )
    manifest = [{
        "path": str(path.relative_to(protocol.OUT_ROOT)).replace("\\", "/"),
        "bytes": path.stat().st_size,
        "sha256": file_digest(path),
    } for path in all_files]
    checks = {
        "protocol_audit_passed": protocol_audit["all_checks_passed"],
        "pairing_audit_passed": pairing_audit["all_checks_passed"],
        "translation_scan_authorized": pairing[
            "translation_internal_authorized"
        ],
        "behavior_checks": behavior_checks,
        "scan_checks": scan_checks,
        "common_behavior_case_keys_equal": common_keys_equal,
        "analysis_protocol_digest_match": (
            analysis["protocol_digest"] == prereg["protocol_digest"]
        ),
        "analysis_pairing_digest_match": (
            analysis["pairing_digest"] == pairing["pairing_digest"]
        ),
        "npz_all_passed": all(
            row["all_checks_passed"] for row in npz_checks
        ),
        "artifact_file_count": len(all_files),
        "artifact_bytes": sum(row["bytes"] for row in manifest),
        "artifact_suffix_counts": dict(Counter(
            Path(row["path"]).suffix for row in manifest
        )),
    }
    checks["all_checks_passed"] = bool(
        checks["protocol_audit_passed"]
        and checks["pairing_audit_passed"]
        and checks["translation_scan_authorized"]
        and all(
            row["all_checks_passed"] for row in behavior_checks.values()
        )
        and all(
            row["all_checks_passed"] for row in scan_checks.values()
        )
        and checks["common_behavior_case_keys_equal"]
        and checks["analysis_protocol_digest_match"]
        and checks["analysis_pairing_digest_match"]
        and checks["npz_all_passed"]
    )
    audit_root = protocol.OUT_ROOT / "audit"
    protocol.write_json(audit_root / "audit.json", checks)
    protocol.write_jsonl(audit_root / "artifact_manifest.jsonl", manifest)
    protocol.write_jsonl(audit_root / "npz_audit.jsonl", npz_checks)
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    if not checks["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
