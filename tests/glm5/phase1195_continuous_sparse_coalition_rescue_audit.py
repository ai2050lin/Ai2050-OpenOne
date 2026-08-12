"""Independent result audit for Phase 1195."""

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
import phase1195_continuous_sparse_coalition_rescue as p  # noqa: E402


def close(left: float, right: float, tolerance: float = 1e-9) -> bool:
    return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)


def add(checks: list[dict[str, Any]], name: str, passed: bool, detail: Any = None) -> None:
    checks.append({"name": name, "pass": bool(passed), "detail": detail})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    checks: list[dict[str, Any]] = []

    protocol = p.read_json(p.PROTOCOL_PATH)
    seal = p.read_json(p.TRAINING_SEAL)
    rows = p.read_jsonl(p.RAW_ROWS)
    summary = p.read_json(p.SUMMARY_PATH)
    claims = p.read_json(p.CLAIMS_PATH)

    protocol_candidate = {
        key: value for key, value in protocol.items() if key != "protocol_digest"
    }
    add(checks, "protocol_digest", p.digest(protocol_candidate) == protocol["protocol_digest"])
    add(checks, "protocol_phase", protocol.get("phase") == p.PHASE)
    add(checks, "source_hashes", protocol.get("source_hashes") == p.source_hashes())
    add(
        checks,
        "upstream_phase1194_hash",
        protocol["upstream"]["phase1194_final_sha256"] == p.file_sha256(p1194.FINAL_PATH),
    )
    add(checks, "seal_protocol", seal["protocol_digest"] == protocol["protocol_digest"])
    seal_candidate = {key: value for key, value in seal.items() if key != "seal_digest"}
    add(checks, "seal_digest", p.digest(seal_candidate) == seal["seal_digest"])
    add(checks, "raw_rows_hash", p.file_sha256(p.RAW_ROWS) == seal["analysis_rows_sha256"])

    row_manifest = {
        path.name: p.file_sha256(path) for path in sorted(p.FORMAL_ROW_ROOT.glob("*.json"))
    }
    replay_manifest = {
        path.name: p.file_sha256(path) for path in sorted(p.REPLAY_ROOT.glob("*.pt"))
    }
    add(checks, "row_manifest", row_manifest == seal["row_manifest"])
    add(checks, "replay_manifest", replay_manifest == seal["replay_manifest"])
    add(checks, "row_count", len(rows) == 96 == seal["row_count"])
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
        {name: sum(row["architecture"] == name for row in rows) for name in p.ARCHITECTURES}
        == {"compact": 48, "deep": 48},
    )
    add(
        checks,
        "family_counts",
        {name: sum(row["family"] == name for row in rows) for name in ("affine", "bitmix", "random")}
        == {"affine": 32, "bitmix": 32, "random": 32},
    )

    formal_seeds = {int(task["task_seed"]) for task in p.FORMAL_TASKS}
    development_seeds = {int(task["task_seed"]) for task in p.DEVELOPMENT_TASKS}
    upstream_seeds = {int(task["task_seed"]) for task in p1194.FORMAL_TASKS}
    discovery_seeds = {
        int(task["task_seed"]) for task in p.FORMAL_TASKS if task["split"] == "discovery"
    }
    confirmation_seeds = formal_seeds - discovery_seeds
    add(checks, "task_seed_unique", len(formal_seeds) == len(p.FORMAL_TASKS))
    add(checks, "task_seed_not_development", formal_seeds.isdisjoint(development_seeds))
    add(checks, "task_seed_not_phase1194", formal_seeds.isdisjoint(upstream_seeds))
    add(checks, "split_seed_disjoint", discovery_seeds.isdisjoint(confirmation_seeds))

    formula_ok = True
    alpha_ok = True
    null_names = ("wrong_component", "wrong_time", "wrong_task", "negative", "random")
    for row in rows:
        variants = row["rescue_variants"]
        control_error = variants["control"]["response_error"]
        expected_correct = (
            control_error - variants["correct"]["response_error"]
        ) / max(control_error, 1e-12)
        expected_null = max(
            (control_error - variants[name]["response_error"]) / max(control_error, 1e-12)
            for name in null_names
        )
        formula_ok &= close(row["rescue_control_error"], control_error)
        formula_ok &= close(row["rescue_correct_recovery"], expected_correct)
        formula_ok &= close(row["rescue_null_recovery"], expected_null)
        formula_ok &= close(row["rescue_advantage"], expected_correct - expected_null)
        alpha = np.asarray(row["alpha"], dtype=np.float64)
        expected_groups = 2 * p.ARCHITECTURES[row["architecture"]].layers
        alpha_ok &= len(alpha) == expected_groups
        alpha_ok &= bool(np.all(np.isfinite(alpha)))
        alpha_ok &= bool(np.all(alpha >= -1e-12) and np.all(alpha <= 1.0 + 1e-12))
        alpha_ok &= int(np.sum(alpha > p.SUPPORT_EPSILON)) == int(row["support_count"])
    add(checks, "row_formulas", formula_ok)
    add(checks, "alpha_domain_and_support", alpha_ok)
    add(checks, "all_finite", all(np.isfinite(row["rescue_advantage"]) for row in rows))
    add(
        checks,
        "eligible_controls_valid",
        all(
            (not row["rescue_eligible"])
            or (
                row["rescue_control_error"] >= p.CONTROL_THRESHOLDS["control_error_min"]
                and row["support_parameter_fraction"]
                <= p.CONTROL_THRESHOLDS["support_parameter_fraction_max"]
                and row["patch_update_fraction"]
                <= p.CONTROL_THRESHOLDS["patch_update_fraction_max"]
                and p.control_match_pass(row["control_match"])
            )
            for row in rows
        ),
    )

    discovery = p.summarize(rows, "discovery")
    confirmation = p.summarize(rows, "confirmation")
    add(checks, "discovery_recompute", p.digest(discovery) == p.digest(summary["discovery"]))
    add(checks, "confirmation_recompute", p.digest(confirmation) == p.digest(summary["confirmation"]))
    expected_positive = discovery["rescue_gate_pass"] and confirmation["rescue_gate_pass"]
    add(
        checks,
        "decision_recompute",
        summary["rescue_decision"] == ("positive" if expected_positive else "not_confirmed"),
    )
    claim = claims["continuous_sparse_coalition_rescue"]
    add(
        checks,
        "claim_type",
        claim["type"] == ("E3-KT" if expected_positive else "E3-KT-scope-boundary"),
    )

    replay_results = []
    if not torch.cuda.is_available():
        add(checks, "cuda_replay_available", False, "CUDA unavailable")
    else:
        device = torch.device("cuda")
        row_lookup = {row["trajectory_id"]: row for row in rows}
        replay_ok = True
        for path in sorted(p.REPLAY_ROOT.glob("*.pt")):
            result = p.replay_capsule(path, device)
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
        "kind": "independent_result_audit",
        "gate_pass": gate_pass,
        "checks_passed": sum(check["pass"] for check in checks),
        "checks_total": len(checks),
        "checks": checks,
        "replay_results": replay_results,
    }
    output["audit_digest"] = p.digest(output)
    if args.write:
        p.write_json(p.AUDIT_PATH, output)
    print(json.dumps(output, ensure_ascii=False, indent=2))
    if not gate_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
