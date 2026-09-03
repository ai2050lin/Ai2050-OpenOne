#!/usr/bin/env python3
"""Independent audit for Phase1547."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1547_c093_discovery_behavior_breadth_screen"
CONTRACT = TESTS / "result/phase1546_c093_symmetric_code_interface_breadth_contract"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def recall(rows: list[dict], truth: bool) -> float:
    subset = [row for row in rows if row["semantic_truth"] is truth]
    return sum(row["semantic_correct"] for row in subset) / len(subset)


def ba(rows: list[dict]) -> float:
    return 0.5 * (recall(rows, True) + recall(rows, False))


def main() -> None:
    report = core.load(OUT / "analysis/discovery_behavior_summary.json")
    rows = core.rows(OUT / "raw/discovery_behavior_logits.jsonl")
    protocol = core.load(CONTRACT / "protocol/preregistration.json")
    metric_identity = True
    recomputed_passes = []
    gate = protocol["discovery_gate_per_interface"]
    for interface in protocol["interfaces"]:
        codebook_passes = []
        for codebook in protocol["codebooks"]:
            subset = [row for row in rows if row["interface"] == interface and row["codebook"] == codebook]
            stored = report["interface_results"][interface]["codebooks"][codebook]
            values = {"ba": ba(subset), "true": recall(subset, True), "false": recall(subset, False), "surface": {surface: ba([row for row in subset if row["surface"] == surface]) for surface in protocol["surfaces"]}}
            metric_identity &= values["ba"] == stored["semantic_balanced_accuracy"] and values["true"] == stored["semantic_true_recall"] and values["false"] == stored["semantic_false_recall"] and values["surface"] == stored["surface"]
            codebook_passes.append(values["ba"] >= gate["each_codebook_semantic_balanced_accuracy"] and all(value >= gate["each_codebook_each_surface_semantic_balanced_accuracy"] for value in values["surface"].values()) and values["true"] >= gate["each_codebook_true_recall"] and values["false"] >= gate["each_codebook_false_recall"])
        if all(codebook_passes):
            recomputed_passes.append(interface)
    checks = {
        "coverage": len(rows) == 320,
        "hash": core.sha(OUT / "raw/discovery_behavior_logits.jsonl") == report["files"]["logits"]["sha256"],
        "metrics": metric_identity,
        "passing_interfaces": recomputed_passes == report["preview_passing_interfaces"],
        "repeat": report["repeat_logit_max_abs"] <= 1e-6,
        "hidden_disabled": report["checks"]["hidden_disabled"],
        "authorization": core.load(OUT / "analysis/final.json")["authorization"] == "run_phase1548_c093_discovery_interface_adjudication",
    }
    result = {"phase": 1547, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "checks": checks, "recomputed_passing_interfaces": recomputed_passes}
    core.save(OUT / "audit/independent_final_audit.json", result)
    if not result["all_checks_passed"]:
        raise RuntimeError(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
