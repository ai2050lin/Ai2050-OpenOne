#!/usr/bin/env python3
"""Strict completeness and claim-boundary audit for Phase1023."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1023_ecological_niche_protocol as protocol


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    protocol_summary = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "summary.json"
    )
    pairing = protocol.read_json(
        protocol.OUT_ROOT / "pairing" / "summary.json"
    )
    final = protocol.read_json(
        protocol.OUT_ROOT / "final" / "summary.json"
    )
    model_checks: dict[str, Any] = {}
    for model in protocol.MODELS:
        behavior = protocol.read_json(
            protocol.OUT_ROOT / "behavior" / model / "summary.json"
        )
        ecology_root = protocol.OUT_ROOT / "ecology" / model
        ecology = protocol.read_json(ecology_root / "summary.json")
        residual = np.load(
            ecology_root / "residual_states.fp16.npy",
            mmap_mode="r",
        )
        heads = np.load(
            ecology_root / "attention_heads.fp16.npy",
            mmap_mode="r",
        )
        mlp = np.load(
            ecology_root / "mlp_intermediate.fp16.npy",
            mmap_mode="r",
        )
        patterns = np.load(
            ecology_root / "pattern_residual_states.fp16.npy",
            mmap_mode="r",
        )
        selected = ecology["selected_layers"]
        model_checks[model] = {
            "behavior_count": behavior["case_count"] == 1600,
            "atlas_count": ecology["case_count"] == 960,
            "protocol_digest_match": (
                behavior["protocol_digest"]
                == ecology["protocol_digest"]
                == prereg["protocol_digest"]
            ),
            "fp16_behavior": (
                behavior["precision"] == "fp16"
                and behavior["quantization"] == "none"
                and behavior["runtime_precision_audit"][
                    "has_fp16_parameters"
                ]
                and not behavior["runtime_precision_audit"][
                    "has_quantized_modules"
                ]
            ),
            "fp16_ecology": (
                ecology["precision"] == "fp16"
                and ecology["quantization"] == "none"
                and ecology["runtime_audit"]["has_fp16_parameters"]
                and not ecology["runtime_audit"]["has_quantized_modules"]
            ),
            "residual_dtype": str(residual.dtype) == "float16",
            "head_dtype": str(heads.dtype) == "float16",
            "mlp_dtype": str(mlp.dtype) == "float16",
            "pattern_dtype": str(patterns.dtype) == "float16",
            "raw_case_axes": (
                residual.shape[0] == 960
                and heads.shape[0] == 960
                and mlp.shape[0] == 960
            ),
            "pattern_case_axis": patterns.shape[0] == 160,
            "roles_complete": (
                residual.shape[1]
                == heads.shape[1]
                == mlp.shape[1]
                == len(protocol.ATLAS_ROLES)
            ),
            "three_layers_per_role": all(
                len(selected[role]) == 3
                for role in protocol.ATLAS_ROLES
            ),
            "component_counts": (
                ecology["attention_head_metric_count"] > 0
                and ecology["mlp_candidate_count"] == 384
            ),
        }
        model_checks[model]["all_checks_passed"] = all(
            model_checks[model].values()
        )

    claim_checks = {
        "does_not_claim_causal_mechanism": (
            final["conclusion"]["causal_mechanism_established"] is False
        ),
        "does_not_claim_optimality": (
            final["conclusion"]["near_optimality_established"] is False
        ),
        "does_not_claim_brain_homology": (
            final["conclusion"]["brain_homology_established"] is False
        ),
        "does_not_claim_storage_cells": (
            final["conclusion"][
                "single_word_storage_cell_established"
            ] is False
        ),
        "automatic_action_matches_gate": (
            final["conclusion"]["automatic_causal_followup_authorized"]
            == final["ability_fork"]["causal_followup_authorized"]
        ),
    }
    checks = {
        "protocol_audited": (
            protocol_summary["common_audit_passed"]
            and protocol_summary["model_audits_passed"]
        ),
        "protocol_digest_match": (
            protocol_summary["protocol_digest"]
            == prereg["protocol_digest"]
            == final["protocol_digest"]
        ),
        "pairing_audited": (
            protocol.OUT_ROOT / "pairing" / "audit.json"
        ).exists(),
        "all_models_complete": all(
            row["all_checks_passed"] for row in model_checks.values()
        ),
        "claim_boundaries_pass": all(claim_checks.values()),
    }
    checks["all_checks_passed"] = all(checks.values())
    result = {
        "schema_version": "phase1023_result_audit.v1",
        "phase": protocol.PHASE,
        "checks": checks,
        "model_checks": model_checks,
        "claim_checks": claim_checks,
        "pairing_authorization": pairing[
            "ability_scan_authorized_by_model"
        ],
    }
    excluded = {
        protocol.OUT_ROOT / "final" / "audit.json",
        protocol.OUT_ROOT / "final" / "artifact_manifest.json",
    }
    artifact_files = sorted(
        path
        for path in protocol.OUT_ROOT.rglob("*")
        if path.is_file() and path not in excluded
    )
    manifest_rows = [
        {
            "path": str(path.relative_to(ROOT)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in artifact_files
    ]
    protocol.write_json(
        protocol.OUT_ROOT / "final" / "artifact_manifest.json",
        {
            "schema_version": "phase1023_artifact_manifest.v1",
            "phase": protocol.PHASE,
            "file_count": len(manifest_rows),
            "total_bytes": sum(row["bytes"] for row in manifest_rows),
            "files": manifest_rows,
        },
    )
    protocol.write_json(
        protocol.OUT_ROOT / "final" / "audit.json",
        result,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not checks["all_checks_passed"]:
        raise RuntimeError("Phase1023 audit failed")


if __name__ == "__main__":
    main()
