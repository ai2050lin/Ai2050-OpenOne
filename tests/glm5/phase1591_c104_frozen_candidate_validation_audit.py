#!/usr/bin/env python3
"""Independent staged audits for Phase1591 / C104."""
from __future__ import annotations

import argparse
import json
import py_compile
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1589_c104_upstream_candidate_validation"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def save(name: str, checks: dict[str, bool]) -> None:
    result = {"phase": 1591, "campaign": "C104", "stage": name, "checks": checks,
              "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    if not result["all_checks_passed"]:
        raise RuntimeError(result)
    core.save(OUT / f"audit/independent_{name}_audit.json", result)
    print(json.dumps(result, indent=2))


def coefficient() -> None:
    producer = TESTS / "phase1591_c104_frozen_candidate_validation.py"
    py_compile.compile(str(producer), doraise=True)
    adapter = core.load(OUT / "protocol/frozen_candidate_validation_adapter.json")
    report = core.load(OUT / "analysis/role_effect_coefficient_summary.json")
    path = OUT / "raw/qwen3_breadth_three_effect_coefficients.float32.npy"
    coeff = np.load(path, mmap_mode="r")
    checks = {
        "producer": core.sha(producer) == adapter["producer_sha256"],
        "source": report["finite"] and report["authorization"] == "reveal_response_discovery_descriptively",
        "shape": coeff.shape == (36, 3, 37, 7, 2560),
        "hash": core.sha(path) == report["sha256"],
        "index": len(core.rows(OUT / "raw/qwen3_breadth_three_effect_index.jsonl")) == 36,
        "unrevealed": not (OUT / "analysis/response_discovery_frozen_candidate_results.jsonl").exists(),
    }
    save("coefficient", checks)


def partition(name: str, next_name: str) -> None:
    path = OUT / f"analysis/{name}_frozen_candidate_results.jsonl"
    summary_name = "response_discovery_reveal_summary.json" if name == "response_discovery" else f"{name}_reveal_summary.json"
    summary = core.load(OUT / f"analysis/{summary_name}")
    rows = core.rows(path)
    checks = {
        "rows": len(rows) == 4 and {row["partition"] for row in rows} == {name},
        "frozen": [(row["family"], row["role"], row["state"]) for row in rows] == [
            ("attribute_binding", "query_anchor", 19), ("agent_patient", "query_anchor", 19),
            ("negation_scope", "focus_record", 3), ("whole_part_exception", "focus_post", 23)],
        "coordinates": all(row["coordinates"] == 2560 for row in rows),
        "hash": core.sha(path) == summary["rows_sha256"],
        "authorization": summary["authorization"] == next_name,
        "next_unrevealed": not (OUT / ("analysis/confirmation_frozen_candidate_results.jsonl" if name == "response_discovery" else "analysis/lockbox_frozen_candidate_results.jsonl")).exists(),
    }
    save(name, checks)


def final() -> None:
    final_result = core.load(OUT / "analysis/frozen_candidate_validation_final.json")
    rows = core.rows(OUT / "analysis/fresh_validation_family_summary.jsonl")
    checks = {
        "rows": len(rows) == 4,
        "hash": core.sha(OUT / "analysis/fresh_validation_family_summary.jsonl") == final_result["family_summary_sha256"],
        "counts": final_result["formal_replication_passed"] == len(final_result["formal_replication_families"]),
        "authorized": sorted(final_result["formal_replication_families"]) == sorted(row["family"] for row in rows if row["formal_replication_pass"]),
        "scope": "not yet causal" in final_result["interpretation"],
        "authorization": final_result["authorization"] in ("run_phase1592_c104_upstream_role_intervention", "close_c104_without_intervention"),
    }
    save("frozen_candidate_validation_final", checks)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("coefficient", "response_discovery", "confirmation", "lockbox", "final"))
    args = parser.parse_args()
    if args.stage == "coefficient": coefficient()
    elif args.stage == "response_discovery": partition("response_discovery", "reveal_confirmation_without_modification")
    elif args.stage == "confirmation": partition("confirmation", "reveal_lockbox_once_without_modification")
    elif args.stage == "lockbox":
        summary = core.load(OUT / "analysis/lockbox_reveal_summary.json")
        rows = core.rows(OUT / "analysis/lockbox_frozen_candidate_results.jsonl")
        save("lockbox", {"rows": len(rows) == 4 and {r["partition"] for r in rows} == {"lockbox"},
                         "hash": core.sha(OUT / "analysis/lockbox_frozen_candidate_results.jsonl") == summary["rows_sha256"],
                         "authorization": summary["authorization"] == "finalize_fresh_validation"})
    else: final()


if __name__ == "__main__":
    main()
