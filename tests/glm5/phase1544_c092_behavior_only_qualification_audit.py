#!/usr/bin/env python3
"""Independent audit for Phase1544."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1544_c092_behavior_only_qualification"
CONTRACT = TESTS / "result/phase1543_c092_truth_output_code_factorial_contract"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def recall(rows: list[dict], truth: bool) -> float:
    subset = [row for row in rows if row["semantic_truth"] is truth]
    return sum(row["semantic_correct"] for row in subset) / len(subset)


def ba(rows: list[dict]) -> float:
    return 0.5 * (recall(rows, True) + recall(rows, False))


def main() -> None:
    report = core.load(OUT / "analysis/behavior_summary.json")
    rows = core.rows(OUT / "raw/behavior_logits.jsonl")
    protocol = core.load(CONTRACT / "protocol/preregistration.json")
    recomputed = {}
    for codebook in protocol["codebooks"]:
        subset = [row for row in rows if row["codebook"] == codebook and row["partition"] == "response_discovery"]
        recomputed[codebook] = {
            "ba": ba(subset),
            "true": recall(subset, True),
            "false": recall(subset, False),
            "surface": {surface: ba([row for row in subset if row["surface"] == surface]) for surface in protocol["surfaces"]},
        }
    metric_identity = all(
        abs(recomputed[codebook]["ba"] - report["codebooks"][codebook]["discovery"]["semantic_balanced_accuracy"]) < 1e-12
        and abs(recomputed[codebook]["true"] - report["codebooks"][codebook]["discovery"]["semantic_true_recall"]) < 1e-12
        and abs(recomputed[codebook]["false"] - report["codebooks"][codebook]["discovery"]["semantic_false_recall"]) < 1e-12
        and recomputed[codebook]["surface"] == report["codebooks"][codebook]["discovery"]["surface_semantic_balanced_accuracy"]
        for codebook in protocol["codebooks"]
    )
    checks = {
        "coverage": len(rows) == 240 and len({row["case_id"] for row in rows}) == 240,
        "hash": core.sha(OUT / "raw/behavior_logits.jsonl") == report["files"]["behavior_logits"]["sha256"],
        "metric_identity": metric_identity,
        "emission_semantic_identity": all(row["emitted_correct"] == row["semantic_correct"] for row in rows),
        "repeat": report["repeat_logit_max_abs"] <= protocol["numeric_gate"]["repeat_logits_max_abs"],
        "hidden_disabled": report["checks"]["hidden_disabled"],
        "authorization": core.load(OUT / "analysis/final.json")["authorization"] == "run_phase1545_c092_behavior_gate_adjudication",
    }
    result = {"phase": 1544, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "checks": checks, "recomputed": recomputed}
    core.save(OUT / "audit/independent_final_audit.json", result)
    if not result["all_checks_passed"]:
        raise RuntimeError(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
