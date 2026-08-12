"""Independent development/formal audit for Phase 1199."""

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
import phase1198_expanded_partition_sparse_rescue as p1198  # noqa: E402
import phase1199_expanded_rescue_role_decomposition as p  # noqa: E402


def close(left: float, right: float, tolerance: float = 1e-9) -> bool:
    return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)


def add(checks: list[dict[str, Any]], name: str, passed: bool, detail: Any = None) -> None:
    checks.append({"name": name, "pass": bool(passed), "detail": detail})


def common_checks(checks: list[dict[str, Any]], rows: list[dict[str, Any]], expected: int) -> None:
    add(checks, "row_count", len(rows) == expected, len(rows))
    add(checks, "trajectory_unique", len({row["trajectory_id"] for row in rows}) == expected)
    formulas = True
    candidates = True
    alpha_ok = True
    for row in rows:
        values = row["role_variants"]
        control_error = values["control"]["response_error"]
        for metrics in values.values():
            formulas &= close(metrics["response_recovery"], (control_error - metrics["response_error"]) / max(control_error, 1e-12))
        full = values["full"]["response_recovery"]
        formulas &= close(row["full_recovery"], full)
        formulas &= close(row["embedding_necessity"], full - values["full_without_embeddings"]["response_recovery"])
        formulas &= close(row["token_necessity"], full - values["full_without_token"]["response_recovery"])
        formulas &= close(row["position_necessity"], full - values["full_without_position"]["response_recovery"])
        null = max(values[name]["response_recovery"] for name in ("embedding_negative", "embedding_random", "embedding_wrong_task"))
        formulas &= close(row["embedding_selectivity_null_recovery"], null)
        formulas &= close(row["embedding_selectivity_advantage"], row["embedding_pair_recovery"] - null)
        qualifying = [
            name for name in p.CANDIDATE_ORDER
            if values[name]["response_recovery"] >= p.THRESHOLDS["candidate_recovery_min"]
            and full - values[name]["response_recovery"] <= p.THRESHOLDS["candidate_full_gap_max"]
        ]
        expected_name = min(qualifying, key=lambda name: (row["role_parameter_fractions"][name], p.CANDIDATE_ORDER.index(name))) if qualifying else None
        candidates &= row["minimal_candidate"] == expected_name
        candidates &= row["minimal_candidate_success"] == (expected_name is not None)
        alpha = np.asarray(row["expanded_alpha"], dtype=np.float64)
        alpha_ok &= len(alpha) == 2 * p.ARCHITECTURES[row["architecture"]].layers + 4
        alpha_ok &= bool(np.all(np.isfinite(alpha)) and np.all(alpha >= -1e-12) and np.all(alpha <= 1.0 + 1e-12))
    add(checks, "row_formulas", formulas)
    add(checks, "minimal_candidate_recompute", candidates)
    add(checks, "alpha_domain", alpha_ok)
    add(checks, "all_finite", all(np.isfinite(row["embedding_selectivity_advantage"]) for row in rows))


def upstream_seeds() -> set[int]:
    tasks = (
        *p1194.DEVELOPMENT_TASKS,
        *p1194.FORMAL_TASKS,
        *p1195.FORMAL_TASKS,
        *p1197.FORMAL_TASKS,
        *p1198.DEVELOPMENT_TASKS,
        *p1198.FORMAL_TASKS,
    )
    return {int(task["task_seed"]) for task in tasks}


def audit_development(write: bool) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    rows = p.read_jsonl(p.DEVELOPMENT_ROWS)
    summary = p.read_json(p.DEVELOPMENT_SUMMARY)
    common_checks(checks, rows, len(p.DEVELOPMENT_TASKS) * len(p.ARCHITECTURES) * p.DEVELOPMENT_REPLICATES)
    recomputed = p.summarize(rows, "development")
    add(checks, "development_recompute", p.digest(recomputed) == p.digest(summary["development"]))
    add(checks, "decision_recompute", summary["development_gate_pass"] == recomputed["role_gate_pass"])
    development = {int(task["task_seed"]) for task in p.DEVELOPMENT_TASKS}
    formal = {int(task["task_seed"]) for task in p.FORMAL_TASKS}
    add(checks, "development_seed_unique", len(development) == len(p.DEVELOPMENT_TASKS))
    add(checks, "development_seed_not_formal", development.isdisjoint(formal))
    add(checks, "development_seed_not_upstream", development.isdisjoint(upstream_seeds()))
    gate = all(check["pass"] for check in checks)
    output = {"phase": p.PHASE, "kind": "independent_development_audit", "gate_pass": gate, "checks_passed": sum(c["pass"] for c in checks), "checks_total": len(checks), "checks": checks}
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
    add(checks, "source_hashes", protocol["source_hashes"] == p.source_hashes())
    add(checks, "upstream_phase1198", protocol["upstream"]["phase1198_final_sha256"] == p.file_sha256(p1198.FINAL_PATH))
    add(checks, "development_assets", protocol["development"] == {"rows_sha256": p.file_sha256(p.DEVELOPMENT_ROWS), "summary_sha256": p.file_sha256(p.DEVELOPMENT_SUMMARY), "audit_sha256": p.file_sha256(p.DEVELOPMENT_AUDIT)})
    add(checks, "seal_protocol", seal["protocol_digest"] == protocol["protocol_digest"])
    seal_candidate = {key: value for key, value in seal.items() if key != "seal_digest"}
    add(checks, "seal_digest", p.digest(seal_candidate) == seal["seal_digest"])
    add(checks, "raw_rows_hash", p.file_sha256(p.RAW_ROWS) == seal["analysis_rows_sha256"])
    add(checks, "row_manifest", {path.name: p.file_sha256(path) for path in sorted(p.FORMAL_ROW_ROOT.glob("*.json"))} == seal["row_manifest"])
    add(checks, "replay_manifest", {path.name: p.file_sha256(path) for path in sorted(p.REPLAY_ROOT.glob("*.pt"))} == seal["replay_manifest"])
    common_checks(checks, rows, 96)
    add(checks, "split_counts", {s: sum(row["split"] == s for row in rows) for s in ("discovery", "confirmation")} == {"discovery": 48, "confirmation": 48})
    add(checks, "architecture_counts", {a: sum(row["architecture"] == a for row in rows) for a in p.ARCHITECTURES} == {"compact": 48, "deep": 48})
    add(checks, "family_counts", {f: sum(row["family"] == f for row in rows) for f in ("affine", "bitmix", "random")} == {"affine": 32, "bitmix": 32, "random": 32})
    formal = {int(task["task_seed"]) for task in p.FORMAL_TASKS}
    development = {int(task["task_seed"]) for task in p.DEVELOPMENT_TASKS}
    discovery_seeds = {int(task["task_seed"]) for task in p.FORMAL_TASKS if task["split"] == "discovery"}
    add(checks, "formal_seed_unique", len(formal) == len(p.FORMAL_TASKS))
    add(checks, "formal_seed_not_development", formal.isdisjoint(development))
    add(checks, "formal_seed_not_upstream", formal.isdisjoint(upstream_seeds()))
    add(checks, "split_seed_disjoint", discovery_seeds.isdisjoint(formal - discovery_seeds))
    discovery = p.summarize(rows, "discovery")
    confirmation = p.summarize(rows, "confirmation")
    add(checks, "discovery_recompute", p.digest(discovery) == p.digest(summary["discovery"]))
    add(checks, "confirmation_recompute", p.digest(confirmation) == p.digest(summary["confirmation"]))
    positive = discovery["role_gate_pass"] and confirmation["role_gate_pass"]
    add(checks, "decision_recompute", summary["role_decision"] == ("positive" if positive else "not_confirmed"))
    add(checks, "claim_type", claims["expanded_rescue_role_decomposition"]["type"] == ("E3-KT" if positive else "E3-KT-scope-boundary"))
    replay_results = []
    if not torch.cuda.is_available():
        add(checks, "cuda_replay_available", False)
    else:
        lookup = {row["trajectory_id"]: row for row in rows}
        replay_ok = True
        for path in sorted(p.REPLAY_ROOT.glob("*.pt")):
            result = p.replay_capsule(path, torch.device("cuda"))
            replay_results.append(result)
            expected = lookup[result["trajectory_id"]]["role_variants"]
            for name, metrics in result["measured"].items():
                replay_ok &= close(metrics["response_error"], expected[name]["response_error"], 1e-6)
                replay_ok &= close(metrics["response_recovery"], expected[name]["response_recovery"], 1e-6)
        add(checks, "cuda_replay_count", len(replay_results) == 4)
        add(checks, "cuda_role_replay", replay_ok, replay_results)
    gate = all(check["pass"] for check in checks)
    output = {"phase": p.PHASE, "kind": "independent_formal_audit", "gate_pass": gate, "checks_passed": sum(c["pass"] for c in checks), "checks_total": len(checks), "checks": checks, "replay_results": replay_results}
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
