"""Independent result audit for Phase 1197 rescue-failure tomography."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))

import phase1194_natural_minibatch_tangent_and_minimal_rescue as p1194  # noqa: E402
import phase1195_continuous_sparse_coalition_rescue as p1195  # noqa: E402
import phase1197_rescue_failure_tomography as p1197  # noqa: E402


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def close(left: Any, right: Any, tolerance: float = 1e-9) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(
            close(left[key], right[key], tolerance) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            close(a, b, tolerance) for a, b in zip(left, right)
        )
    if isinstance(left, (int, float, bool)) and isinstance(right, (int, float, bool)):
        if isinstance(left, bool) or isinstance(right, bool):
            return bool(left) == bool(right)
        return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)
    return left == right


def add(checks: list[dict[str, Any]], name: str, passed: bool, detail: Any = None) -> None:
    checks.append({"name": name, "pass": bool(passed), "detail": detail})


def finite_tree(value: Any) -> bool:
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def main() -> None:
    checks: list[dict[str, Any]] = []
    protocol = p1197.read_json(p1197.PROTOCOL_PATH)
    seal = p1197.read_json(p1197.TRAINING_SEAL)
    rows = p1197.read_jsonl(p1197.RAW_ROWS)
    summary = p1197.read_json(p1197.SUMMARY_PATH)
    claims = p1197.read_json(p1197.CLAIMS_PATH)

    protocol_candidate = {
        key: value for key, value in protocol.items() if key != "protocol_digest"
    }
    add(checks, "protocol_digest", digest(protocol_candidate) == protocol["protocol_digest"])
    add(checks, "protocol_phase", protocol["phase"] == p1197.PHASE)
    add(checks, "source_hashes", protocol["source_hashes"] == p1197.source_hashes())
    add(
        checks,
        "upstream_phase1195_hash",
        p1197.file_sha256(p1195.FINAL_PATH)
        == protocol["upstream"]["phase1195_final_sha256"],
    )
    add(checks, "seal_protocol", seal["protocol_digest"] == protocol["protocol_digest"])
    seal_candidate = {key: value for key, value in seal.items() if key != "seal_digest"}
    add(checks, "seal_digest", digest(seal_candidate) == seal["seal_digest"])
    add(
        checks,
        "raw_rows_hash",
        p1197.file_sha256(p1197.RAW_ROWS) == seal["analysis_rows_sha256"],
    )
    row_manifest = {
        path.name: p1197.file_sha256(path)
        for path in sorted(p1197.FORMAL_ROW_ROOT.glob("*.json"))
    }
    replay_manifest = {
        path.name: p1197.file_sha256(path)
        for path in sorted(p1197.REPLAY_ROOT.glob("*.pt"))
    }
    add(checks, "row_manifest", row_manifest == seal["row_manifest"])
    add(checks, "replay_manifest", replay_manifest == seal["replay_manifest"])
    add(checks, "row_count", len(rows) == seal["row_count"] == 96)
    add(checks, "trajectory_unique", len({row["trajectory_id"] for row in rows}) == 96)
    add(
        checks,
        "split_counts",
        {split: sum(row["split"] == split for row in rows) for split in ("discovery", "confirmation")}
        == {"discovery": 48, "confirmation": 48},
    )
    add(
        checks,
        "architecture_counts",
        {name: sum(row["architecture"] == name for row in rows) for name in p1197.ARCHITECTURES}
        == {"compact": 48, "deep": 48},
    )
    add(
        checks,
        "family_counts",
        {name: sum(row["family"] == name for row in rows) for name in ("affine", "bitmix", "random")}
        == {"affine": 32, "bitmix": 32, "random": 32},
    )
    task_counts = {task["name"]: 0 for task in p1197.FORMAL_TASKS}
    for row in rows:
        task_counts[row["task_name"]] += 1
    add(checks, "task_counts", all(value == 8 for value in task_counts.values()))
    seeds = [task["task_seed"] for task in p1197.FORMAL_TASKS]
    upstream_seeds = {
        int(task["task_seed"])
        for task in (*p1194.DEVELOPMENT_TASKS, *p1194.FORMAL_TASKS, *p1195.FORMAL_TASKS)
    }
    add(checks, "task_seed_unique", len(seeds) == len(set(seeds)))
    add(checks, "task_seed_disjoint", not (set(seeds) & upstream_seeds))
    add(checks, "all_finite", all(finite_tree(row) for row in rows))
    add(checks, "eligible_fraction", sum(row["eligible"] for row in rows) >= 92)
    add(
        checks,
        "partition_contract",
        all(row["partition_complete"] and row["partition_overlap"] == 0 for row in rows),
    )
    add(
        checks,
        "full_difference_positive_control",
        all(row["full_eval_recovery"] >= 0.99 for row in rows),
    )
    add(
        checks,
        "derived_gap_formulas",
        all(
            math.isclose(
                row["box_gap"], row["box_cal_error"] - row["span_cal_error"], abs_tol=1e-12
            )
            and math.isclose(
                row["sparsity_gap"], row["l1_cal_error"] - row["box_cal_error"], abs_tol=1e-12
            )
            for row in rows
        ),
    )
    recomputed_discovery = p1197.summarize_rows(rows, "discovery")
    recomputed_confirmation = p1197.summarize_rows(rows, "confirmation")
    add(checks, "discovery_recompute", close(recomputed_discovery, summary["discovery"]))
    add(checks, "confirmation_recompute", close(recomputed_confirmation, summary["confirmation"]))
    same = recomputed_discovery["primary_diagnosis"] == recomputed_confirmation["primary_diagnosis"]
    expected_diagnosis = (
        recomputed_discovery["primary_diagnosis"] if same else "split_diagnosis_disagreement"
    )
    expected_confirmed = bool(
        same
        and expected_diagnosis == "omitted_high_leverage_basis"
        and recomputed_discovery["omitted_high_leverage_basis_gate_pass"]
        and recomputed_confirmation["omitted_high_leverage_basis_gate_pass"]
    )
    add(
        checks,
        "decision_recompute",
        summary["primary_diagnosis"] == expected_diagnosis
        and summary["omitted_high_leverage_basis_confirmed"] == expected_confirmed,
    )
    expected_type = "E3-KT" if expected_confirmed else "E3-KT-scope-boundary"
    add(
        checks,
        "claim_type",
        claims["rescue_failure_tomography"]["type"] == expected_type
        and claims["rescue_failure_tomography"]["accepted"] is True,
    )
    add(checks, "cuda_available", torch.cuda.is_available())
    capsule_paths = sorted(p1197.REPLAY_ROOT.glob("*.pt"))
    add(checks, "cuda_replay_count", len(capsule_paths) == 4)
    replay_errors = []
    if torch.cuda.is_available() and len(capsule_paths) == 4:
        by_id = {row["trajectory_id"]: row for row in rows}
        replay_keys = (
            "solver_objective_gap",
            "span_cal_error",
            "box_cal_error",
            "l1_cal_error",
            "core_eval_recovery",
            "core_update_norm_fraction",
            "full_eval_recovery",
            "best_embedding_only_recovery",
            "design_cross_panel_cosine",
            "l1_nonlinear_eval_error",
        )
        for path in capsule_paths:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            replayed = p1197.diagnose_payload(payload, torch.device("cuda"))
            stored = by_id[payload["trajectory_id"]]
            for key in replay_keys:
                replay_errors.append(abs(float(replayed[key]) - float(stored[key])))
        add(
            checks,
            "cuda_diagnostic_replay",
            max(replay_errors, default=0.0) <= 1e-6,
            {"max_error": max(replay_errors, default=0.0)},
        )
    else:
        add(checks, "cuda_diagnostic_replay", False, "CUDA or capsules unavailable")
    add(
        checks,
        "forbidden_scope_absent",
        not any(
            phrase in claims["rescue_failure_tomography"]["claim"].lower()
            for phrase in ("natural-language mechanism confirmed", "full controllability confirmed")
        ),
    )

    passed = sum(check["pass"] for check in checks)
    audit = {
        "phase": p1197.PHASE,
        "kind": "independent_result_audit",
        "gate_pass": passed == len(checks),
        "checks_passed": passed,
        "checks_total": len(checks),
        "checks": checks,
        "protocol_digest": protocol["protocol_digest"],
        "seal_digest": seal["seal_digest"],
    }
    audit["audit_digest"] = digest(audit)
    p1197.write_json(p1197.AUDIT_PATH, audit)
    print(canonical_json({"gate_pass": audit["gate_pass"], "checks": f"{passed}/{len(checks)}", "audit_digest": audit["audit_digest"]}))
    if not audit["gate_pass"]:
        failed = [check["name"] for check in checks if not check["pass"]]
        raise RuntimeError(f"independent audit failed: {failed}")


if __name__ == "__main__":
    main()
