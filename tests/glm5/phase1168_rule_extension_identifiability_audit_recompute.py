#!/usr/bin/env python3
"""Independent exact recomputation for Phase1168."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve()
PRIMARY_SCRIPT = ROOT / "tests/glm5/phase1168_rule_extension_identifiability_audit.py"
OUT_ROOT = ROOT / "tests/glm5/result/phase1168_rule_extension_identifiability_audit"
sys.path.insert(0, str(ROOT / "tests/glm5"))

import phase1168_rule_extension_identifiability_audit as phase  # noqa: E402


p1163 = phase.p1163


def audit_command() -> None:
    protocol = phase.verify_protocol()
    result = p1163.read_json(OUT_ROOT / "analysis/result.json")
    final = p1163.read_json(OUT_ROOT / "analysis/final.json")
    recomputed_rows = [phase.analyze_task(task) for task in phase.task_panel()]
    recomputed_cells = []
    for row in recomputed_rows:
        constraints = row["constraint_results"]
        recomputed_cells.append(
            {
                "task": row["name"],
                "passed": bool(
                    constraints["data_only"]["holdout_extension_count"] > 1
                    and constraints["train_internal_equivariance"][
                        "holdout_extension_count"
                    ]
                    > 1
                    and constraints["partition_internal_equivariance"][
                        "holdout_extension_count"
                    ]
                    > 1
                    and constraints["global_equivariance"][
                        "holdout_extension_count"
                    ]
                    == 1
                    and constraints["separable_additive"][
                        "holdout_extension_count"
                    ]
                    == 1
                ),
            }
        )
    recomputed_authorization = all(row["passed"] for row in recomputed_cells)
    checks = {
        "protocol_digest": p1163.digest(
            {key: value for key, value in protocol.items() if key != "protocol_digest"}
        )
        == protocol["protocol_digest"],
        "primary_hash": p1163.sha256_file(PRIMARY_SCRIPT)
        == protocol["source_hashes"]["primary_script"],
        "audit_hash": p1163.sha256_file(SCRIPT)
        == protocol["source_hashes"]["audit_script"],
        "phase1167_hash": p1163.sha256_file(phase.P1167_SCRIPT)
        == protocol["source_hashes"]["phase1167_script"],
        "protocol_checks": all(protocol["checks"].values()),
        "task_recompute": recomputed_rows == result["tasks"],
        "cell_recompute": recomputed_cells == result["primary_cells"],
        "analysis_digest": p1163.digest(
            {key: value for key, value in result.items() if key != "analysis_digest"}
        )
        == result["analysis_digest"],
        "final_digest": p1163.digest(
            {key: value for key, value in final.items() if key != "final_digest"}
        )
        == final["final_digest"],
        "authorization_recompute": recomputed_authorization
        == result["trajectory_protocol_authorized"]
        == final["trajectory_protocol_authorized"]
        == final["auto_continue"],
        "hidden_scan_denied": not result["hidden_state_scan_authorized"]
        and not final["hidden_state_scan_authorized"],
        "mechanism_not_claimed": not final["natural_mechanism_recovered"],
        "phase1167_unchanged": p1163.read_json(phase.P1167_FINAL)["final_digest"]
        == protocol["source_digests"]["phase1167_final"],
    }
    report = {
        "phase": phase.PHASE,
        "created_at_utc": p1163.now(),
        "checks": checks,
        "check_count": len(checks),
        "passed_count": sum(checks.values()),
        "all_checks_passed": all(checks.values()),
        "recomputed_authorization": recomputed_authorization,
    }
    report["audit_digest"] = p1163.digest(report)
    p1163.write_json(OUT_ROOT / "audit/report.json", report)
    print(p1163.canonical(report))
    if not report["all_checks_passed"]:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("audit",))
    parser.parse_args()
    audit_command()


if __name__ == "__main__":
    main()
