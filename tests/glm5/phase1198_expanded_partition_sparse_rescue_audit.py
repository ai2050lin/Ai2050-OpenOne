"""Independent development/formal audit for Phase 1198."""

from __future__ import annotations

import argparse
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
import phase1198_expanded_partition_sparse_rescue as p  # noqa: E402


def close(left: float, right: float, tolerance: float = 1e-9) -> bool:
    return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)


def add(checks: list[dict[str, Any]], name: str, passed: bool, detail: Any = None) -> None:
    checks.append({"name": name, "pass": bool(passed), "detail": detail})


def common_row_checks(
    checks: list[dict[str, Any]], rows: list[dict[str, Any]], expected_count: int
) -> None:
    add(checks, "row_count", len(rows) == expected_count, len(rows))
    add(checks, "trajectory_unique", len({row["trajectory_id"] for row in rows}) == expected_count)
    formulas = True
    alpha_ok = True
    partition_ok = True
    null_names = ("wrong_component", "wrong_time", "wrong_task", "negative", "random")
    for row in rows:
        variants = row["rescue_variants"]
        control_error = variants["control"]["response_error"]
        correct = (control_error - variants["correct"]["response_error"]) / max(control_error, 1e-12)
        coarse = (control_error - variants["coarse"]["response_error"]) / max(control_error, 1e-12)
        null = max((control_error - variants[name]["response_error"]) / max(control_error, 1e-12) for name in null_names)
        formulas &= close(row["rescue_control_error"], control_error)
        formulas &= close(row["rescue_correct_recovery"], correct)
        formulas &= close(row["coarse_recovery"], coarse)
        formulas &= close(row["expanded_recovery_gain"], correct - coarse)
        formulas &= close(row["rescue_null_recovery"], null)
        formulas &= close(row["rescue_advantage"], correct - null)
        alpha = np.asarray(row["alpha"], dtype=np.float64)
        expected_groups = 2 * p.ARCHITECTURES[row["architecture"]].layers + 4
        alpha_ok &= len(alpha) == expected_groups
        alpha_ok &= bool(np.all(np.isfinite(alpha)))
        alpha_ok &= bool(np.all(alpha >= -1e-12) and np.all(alpha <= 1.0 + 1e-12))
        alpha_ok &= int(np.sum(alpha > p.SUPPORT_EPSILON)) == int(row["support_count"])
        partition_ok &= row["partition"]["complete"]
        partition_ok &= row["partition"]["uncovered"] == 0
        partition_ok &= row["partition"]["overlap"] == 0
        partition_ok &= close(row["partition"]["parameter_fraction_sum"], 1.0, 1e-6)
    add(checks, "row_formulas", formulas)
    add(checks, "alpha_domain_and_support", alpha_ok)
    add(checks, "expanded_partition_exact", partition_ok)
    add(checks, "all_finite", all(np.isfinite(row["rescue_advantage"]) for row in rows))


def audit_development(write: bool) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    rows = p.read_jsonl(p.DEVELOPMENT_ROWS)
    summary = p.read_json(p.DEVELOPMENT_SUMMARY)
    common_row_checks(checks, rows, len(p.DEVELOPMENT_TASKS) * len(p.ARCHITECTURES) * p.DEVELOPMENT_REPLICATES)
    add(checks, "development_split", all(row["split"] == "development" for row in rows))
    recomputed = p.summarize(rows, "development")
    add(checks, "development_recompute", p.digest(recomputed) == p.digest(summary["development"]))
    add(checks, "decision_recompute", summary["development_gate_pass"] == recomputed["rescue_gate_pass"])
    formal_seeds = {int(task["task_seed"]) for task in p.FORMAL_TASKS}
    development_seeds = {int(task["task_seed"]) for task in p.DEVELOPMENT_TASKS}
    upstream_seeds = {
        int(task["task_seed"])
        for task in (*p1194.DEVELOPMENT_TASKS, *p1194.FORMAL_TASKS, *p1195.FORMAL_TASKS, *p1197.FORMAL_TASKS)
    }
    add(checks, "development_seed_unique", len(development_seeds) == len(p.DEVELOPMENT_TASKS))
    add(checks, "development_seed_not_formal", development_seeds.isdisjoint(formal_seeds))
    add(checks, "development_seed_not_upstream", development_seeds.isdisjoint(upstream_seeds))
    gate_pass = all(check["pass"] for check in checks)
    output = {
        "phase": p.PHASE,
        "kind": "independent_development_audit",
        "gate_pass": gate_pass,
        "checks_passed": sum(check["pass"] for check in checks),
        "checks_total": len(checks),
        "checks": checks,
    }
    output["audit_digest"] = p.digest(output)
    if write:
        p.write_json(p.DEVELOPMENT_AUDIT, output)
    return output


def audit_formal(write: bool) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    protocol = p.read_json(p.PROTOCOL_PATH)
    seal = p.read_json(p.TRAINING_SEAL)
    rows = p.read_jsonl(p.RAW_ROWS)
    summary = p.read_json(p.SUMMARY_PATH)
    claims = p.read_json(p.CLAIMS_PATH)
    candidate = {key: value for key, value in protocol.items() if key != "protocol_digest"}
    add(checks, "protocol_digest", p.digest(candidate) == protocol["protocol_digest"])
    add(checks, "protocol_phase", protocol.get("phase") == p.PHASE)
    add(checks, "source_hashes", protocol.get("source_hashes") == p.source_hashes())
    add(checks, "upstream_phase1197", protocol["upstream"]["phase1197_final_sha256"] == p.file_sha256(p1197.FINAL_PATH))
    add(checks, "development_rows", protocol["development"]["rows_sha256"] == p.file_sha256(p.DEVELOPMENT_ROWS))
    add(checks, "development_summary", protocol["development"]["summary_sha256"] == p.file_sha256(p.DEVELOPMENT_SUMMARY))
    add(checks, "development_audit", protocol["development"]["audit_sha256"] == p.file_sha256(p.DEVELOPMENT_AUDIT))
    add(checks, "seal_protocol", seal["protocol_digest"] == protocol["protocol_digest"])
    seal_candidate = {key: value for key, value in seal.items() if key != "seal_digest"}
    add(checks, "seal_digest", p.digest(seal_candidate) == seal["seal_digest"])
    add(checks, "raw_rows_hash", p.file_sha256(p.RAW_ROWS) == seal["analysis_rows_sha256"])
    add(checks, "row_manifest", {path.name: p.file_sha256(path) for path in sorted(p.FORMAL_ROW_ROOT.glob("*.json"))} == seal["row_manifest"])
    add(checks, "replay_manifest", {path.name: p.file_sha256(path) for path in sorted(p.REPLAY_ROOT.glob("*.pt"))} == seal["replay_manifest"])
    common_row_checks(checks, rows, 96)
    add(checks, "split_counts", {split: sum(row["split"] == split for row in rows) for split in ("discovery", "confirmation")} == {"discovery": 48, "confirmation": 48})
    add(checks, "architecture_counts", {name: sum(row["architecture"] == name for row in rows) for name in p.ARCHITECTURES} == {"compact": 48, "deep": 48})
    add(checks, "family_counts", {name: sum(row["family"] == name for row in rows) for name in ("affine", "bitmix", "random")} == {"affine": 32, "bitmix": 32, "random": 32})
    formal_seeds = {int(task["task_seed"]) for task in p.FORMAL_TASKS}
    development_seeds = {int(task["task_seed"]) for task in p.DEVELOPMENT_TASKS}
    upstream_seeds = {int(task["task_seed"]) for task in (*p1194.DEVELOPMENT_TASKS, *p1194.FORMAL_TASKS, *p1195.FORMAL_TASKS, *p1197.FORMAL_TASKS)}
    discovery_seeds = {int(task["task_seed"]) for task in p.FORMAL_TASKS if task["split"] == "discovery"}
    confirmation_seeds = formal_seeds - discovery_seeds
    add(checks, "formal_seed_unique", len(formal_seeds) == len(p.FORMAL_TASKS))
    add(checks, "formal_seed_not_development", formal_seeds.isdisjoint(development_seeds))
    add(checks, "formal_seed_not_upstream", formal_seeds.isdisjoint(upstream_seeds))
    add(checks, "split_seed_disjoint", discovery_seeds.isdisjoint(confirmation_seeds))
    discovery = p.summarize(rows, "discovery")
    confirmation = p.summarize(rows, "confirmation")
    add(checks, "discovery_recompute", p.digest(discovery) == p.digest(summary["discovery"]))
    add(checks, "confirmation_recompute", p.digest(confirmation) == p.digest(summary["confirmation"]))
    expected_positive = discovery["rescue_gate_pass"] and confirmation["rescue_gate_pass"]
    add(checks, "decision_recompute", summary["rescue_decision"] == ("positive" if expected_positive else "not_confirmed"))
    claim = claims["expanded_partition_sparse_rescue"]
    add(checks, "claim_type", claim["type"] == ("E3-KT" if expected_positive else "E3-KT-scope-boundary"))

    replay_results = []
    if not torch.cuda.is_available():
        add(checks, "cuda_replay_available", False, "CUDA unavailable")
    else:
        row_lookup = {row["trajectory_id"]: row for row in rows}
        replay_ok = True
        for path in sorted(p.REPLAY_ROOT.glob("*.pt")):
            result = p.replay_capsule(path, torch.device("cuda"))
            replay_results.append(result)
            row = row_lookup[result["trajectory_id"]]
            replay_ok &= result["alpha_max_error"] <= 1e-8
            replay_ok &= result["patch_relative_error"] <= 1e-7
            for name, metrics in result["measured"].items():
                expected = row["rescue_variants"][name]
                replay_ok &= close(metrics["response_error"], expected["response_error"], 1e-6)
                replay_ok &= close(metrics["output_error"], expected["output_error"], 1e-6)
                replay_ok &= close(metrics["accuracy"], expected["accuracy"], 1e-7)
        add(checks, "cuda_replay_count", len(replay_results) == 4)
        add(checks, "cuda_solver_and_variant_replay", replay_ok, replay_results)
    gate_pass = all(check["pass"] for check in checks)
    output = {
        "phase": p.PHASE,
        "kind": "independent_formal_audit",
        "gate_pass": gate_pass,
        "checks_passed": sum(check["pass"] for check in checks),
        "checks_total": len(checks),
        "checks": checks,
        "replay_results": replay_results,
    }
    output["audit_digest"] = p.digest(output)
    if write:
        p.write_json(p.AUDIT_PATH, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development", action="store_true")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    output = audit_development(args.write) if args.development else audit_formal(args.write)
    print(json.dumps(output, ensure_ascii=False, indent=2))
    if not output["gate_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
