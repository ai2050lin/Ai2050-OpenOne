#!/usr/bin/env python3
"""Independent staged audits for Phase1584 / C102."""
from __future__ import annotations

import argparse
import json
import py_compile
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1581_c102_typed_relation_coordinate_campaign"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def save(name: str, checks: dict[str, bool]) -> None:
    result = {"phase": 1584, "campaign": "C102", "stage": name, "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    if not result["all_checks_passed"]:
        raise RuntimeError(result)
    core.save(OUT / f"audit/independent_{name}_audit.json", result)
    print(json.dumps(result, indent=2))


def coefficient_audit() -> None:
    producer = TESTS / "phase1584_c102_staged_barcode_analysis.py"
    py_compile.compile(str(producer), doraise=True)
    adapter = core.load(OUT / "protocol/staged_analysis_adapter.json")
    report = core.load(OUT / "analysis/role_effect_coefficient_summary.json")
    graph_path = OUT / "raw/qwen3_graph_three_effect_coefficients.float32.npy"
    breadth_path = OUT / "raw/qwen3_breadth_three_effect_coefficients.float32.npy"
    graph = np.load(graph_path, mmap_mode="r")
    breadth = np.load(breadth_path, mmap_mode="r")
    checks = {
        "producer": core.sha(producer) == adapter["producer_sha256"],
        "source": report["finite"] and report["authorization"] == "reveal_response_discovery_only",
        "graph": graph.shape == (36, 3, 37, 6, 2560) and core.sha(graph_path) == report["graph"]["sha256"],
        "breadth": breadth.shape == (36, 3, 37, 7, 2560) and core.sha(breadth_path) == report["breadth"]["sha256"],
        "index": len(core.rows(OUT / "raw/qwen3_graph_three_effect_index.jsonl")) == len(core.rows(OUT / "raw/qwen3_breadth_three_effect_index.jsonl")) == 36,
        "unrevealed": not (OUT / "protocol/response_discovery_selection.json").exists(),
    }
    save("coefficient", checks)


def response_discovery_audit() -> None:
    selection = core.load(OUT / "protocol/response_discovery_selection.json")
    rows = core.rows(OUT / "analysis/response_discovery_nested_k.jsonl")
    nested = core.load(OUT / "protocol/frozen_coordinate_barcode_predictions.json")["validation"]["nested_k"]
    checks = {
        "rows": len(rows) == 8 * len(nested) and {row["partition"] for row in rows} == {"response_discovery"},
        "families": len(selection["selection"]) == 8,
        "k": all(row["k"] in nested for row in selection["selection"].values()),
        "source": core.sha(OUT / "analysis/response_discovery_nested_k.jsonl") == selection["source_rows_sha256"],
        "authorization": selection["authorization"] == "reveal_confirmation_with_frozen_selection",
        "unrevealed": not (OUT / "analysis/confirmation_barcode_results.jsonl").exists(),
    }
    save("response_discovery", checks)


def confirmation_audit() -> None:
    selection = core.load(OUT / "protocol/response_discovery_selection.json")
    summary = core.load(OUT / "analysis/confirmation_reveal_summary.json")
    rows = core.rows(OUT / "analysis/confirmation_barcode_results.jsonl")
    checks = {
        "rows": len(rows) == 8 and {row["partition"] for row in rows} == {"confirmation"},
        "selection": core.sha(OUT / "protocol/response_discovery_selection.json") == summary["selection_sha256"],
        "k": all(row["k"] == selection["selection"][row["family"]]["k"] for row in rows),
        "hash": core.sha(OUT / "analysis/confirmation_barcode_results.jsonl") == summary["rows_sha256"],
        "authorization": summary["authorization"] == "reveal_lockbox_once_with_unchanged_selection",
        "unrevealed": not (OUT / "analysis/lockbox_barcode_results.jsonl").exists(),
    }
    save("confirmation", checks)


def final_audit() -> None:
    final = core.load(OUT / "analysis/staged_barcode_final.json")
    lockbox = core.rows(OUT / "analysis/lockbox_barcode_results.jsonl")
    formation = core.rows(OUT / "analysis/formation_trajectory_validation.jsonl")
    checks = {
        "rows": len(lockbox) == 8 and {row["partition"] for row in lockbox} == {"lockbox"},
        "formation": len(formation) == 24,
        "hashes": core.sha(OUT / "analysis/lockbox_barcode_results.jsonl") == final["lockbox_sha256"] and core.sha(OUT / "analysis/formation_trajectory_validation.jsonl") == final["formation_sha256"],
        "authorized": sorted(final["authorized_intervention_families"]) == sorted(row["family"] for row in lockbox if row["three_stage_pass"]),
        "counts": final["three_stage_passed"] == len(final["authorized_intervention_families"]) and final["total_families"] == 8,
        "scope": "chance behavior" in final["interpretation"],
    }
    save("staged_barcode_final", checks)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("coefficient", "response_discovery", "confirmation", "final"))
    args = parser.parse_args()
    {"coefficient": coefficient_audit, "response_discovery": response_discovery_audit, "confirmation": confirmation_audit, "final": final_audit}[args.stage]()


if __name__ == "__main__":
    main()
