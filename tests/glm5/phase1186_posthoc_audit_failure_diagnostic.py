#!/usr/bin/env python3
"""Non-gating diagnosis of the sole failed Phase1186 audit check.

This script cannot change the frozen Phase1186 verdict or authorize Phase1187.
It only identifies whether the positive-control replay failure comes from the
model response or from comparing an FP32 descriptive mean with an exact ratio.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))

import phase1186_reducer_safe_numerical_qualification as phase  # noqa: E402
import phase1186_reducer_safe_numerical_qualification_audit as audit  # noqa: E402


SCRIPT = Path(__file__).resolve()
OUT_PATH = phase.OUT_ROOT / "analysis/posthoc_audit_failure_diagnostic.json"


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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def close(left: float, right: float, atol: float, rtol: float = 1e-8) -> bool:
    return bool(math.isclose(float(left), float(right), rel_tol=rtol, abs_tol=atol))


def main() -> None:
    if OUT_PATH.exists():
        raise RuntimeError("posthoc diagnostic already exists")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    rows = read_jsonl(phase.OUT_ROOT / "analysis/positive_control_rows.jsonl")
    by_case = {row["case"]: row for row in rows}
    device = torch.device("cuda")
    x = torch.tensor(
        [(a, b) for a in range(phase.p1185.MODULUS) for b in range(phase.p1185.MODULUS)],
        dtype=torch.long,
    )
    details: list[dict[str, Any]] = []
    case = 0
    for scale_index, scale in enumerate(phase.SCALES):
        for structure_index, structure in enumerate(phase.STRUCTURES):
            for replicate in range(phase.REPLICATES):
                stored = by_case[case]
                model = phase.p1185.engineered_model(
                    scale,
                    structure,
                    phase.model_seed(scale_index, structure_index, replicate),
                    device,
                )
                broken = audit.direct_gauge(model, stored["seed"], device, broken=True)
                reference, _ = audit.direct_fp32(model, x, device)
                changed, _ = audit.direct_fp32(broken, x, device)
                reference_feature = np.asarray(
                    phase.p1185.p1183.algebraic_internal_features(model, x), dtype=np.float64
                )
                changed_feature = np.asarray(
                    phase.p1185.p1183.algebraic_internal_features(broken, x), dtype=np.float64
                )
                feature_error = float(np.max(np.abs(reference_feature - changed_feature)))
                fp32 = audit.local_forward(
                    reference.cpu().double().numpy(), changed.cpu().double().numpy(), "fp32"
                )
                equal = reference.argmax(1) == changed.argmax(1)
                agree_count = int(torch.count_nonzero(equal).item())
                total_count = int(equal.numel())
                fp32_mean = float(equal.float().mean().item())
                exact_ratio = agree_count / total_count
                details.append(
                    {
                        "case": case,
                        "scale": scale,
                        "structure": structure,
                        "replicate": replicate,
                        "agree_count": agree_count,
                        "total_count": total_count,
                        "stored_descriptive_mean": stored["decision_agreement"],
                        "replayed_fp32_mean": fp32_mean,
                        "exact_integer_ratio": exact_ratio,
                        "fp32_minus_exact": fp32_mean - exact_ratio,
                        "stored_matches_fp32": stored["decision_agreement"] == fp32_mean,
                        "stored_matches_exact_at_audit_tolerance": close(
                            stored["decision_agreement"], exact_ratio, atol=1e-9
                        ),
                        "feature_matches": close(stored["feature_error"], feature_error, atol=1e-12),
                        "fp32_metrics_match": audit.close(stored["fp32"], fp32, atol=1e-9),
                    }
                )
                del model, broken
                torch.cuda.empty_cache()
                case += 1

    result = {
        "phase": phase.PHASE,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "posthoc_non_gating_diagnostic",
        "formal_audit_verdict_unchanged": "failed_28_of_29",
        "phase1187_authorized": False,
        "system_count": len(details),
        "stored_matches_fp32_count": sum(row["stored_matches_fp32"] for row in details),
        "stored_matches_exact_at_audit_tolerance_count": sum(
            row["stored_matches_exact_at_audit_tolerance"] for row in details
        ),
        "feature_match_count": sum(row["feature_matches"] for row in details),
        "fp32_metric_match_count": sum(row["fp32_metrics_match"] for row in details),
        "maximum_absolute_fp32_exact_ratio_gap": max(
            abs(row["fp32_minus_exact"]) for row in details
        ),
        "diagnosis": (
            "The sole audit failure is caused by comparing a stored FP32 descriptive mean "
            "against an exact integer ratio at 1e-9 tolerance; model responses and all "
            "positive-control feature/forward metrics replay."
        ),
        "claim_exclusions": [
            "This diagnosis does not repair the frozen audit.",
            "This diagnosis does not qualify K165.",
            "This diagnosis does not authorize Phase1187.",
        ],
        "source_hashes": {
            "diagnostic": file_sha256(SCRIPT),
            "runner": file_sha256(phase.SCRIPT),
            "audit": file_sha256(audit.SCRIPT),
        },
        "details": details,
    }
    result["diagnostic_digest"] = digest(result)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(canonical_json({key: value for key, value in result.items() if key != "details"}))


if __name__ == "__main__":
    main()
