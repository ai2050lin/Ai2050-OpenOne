#!/usr/bin/env python3
"""Independent audit for Phase1550."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1550_c094_discovery_behavior_qualification"
CONTRACT = TESTS / "result/phase1549_c094_demonstrated_codebook_contract"
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
    identity = True
    for codebook in protocol["codebooks"]:
        subset = [row for row in rows if row["codebook"] == codebook]
        stored = report["codebooks"][codebook]
        identity &= ba(subset) == stored["semantic_balanced_accuracy"] and recall(subset, True) == stored["semantic_true_recall"] and recall(subset, False) == stored["semantic_false_recall"] and {surface: ba([row for row in subset if row["surface"] == surface]) for surface in protocol["surfaces"]} == stored["surface"]
    checks = {"coverage": len(rows) == 80, "hash": core.sha(OUT / "raw/discovery_behavior_logits.jsonl") == report["files"]["logits"]["sha256"], "metrics": identity, "aggregate": report["preview_both_pass"] == all(value["qualified"] for value in report["codebooks"].values()), "repeat": report["repeat_logit_max_abs"] <= 1e-6, "hidden_disabled": report["checks"]["hidden_disabled"], "authorization": core.load(OUT / "analysis/final.json")["authorization"] == "run_phase1551_c094_discovery_behavior_adjudication"}
    result = {"phase": 1550, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "checks": checks}
    core.save(OUT / "audit/independent_final_audit.json", result)
    if not result["all_checks_passed"]:
        raise RuntimeError(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
