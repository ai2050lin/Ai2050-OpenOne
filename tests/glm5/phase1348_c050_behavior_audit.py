#!/usr/bin/env python3
"""Independent audit for Phase1348 C050 behavior."""
from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core

CONTRACT = TESTS / "result/phase1347_c050_formation_clock_contract"
OUT = TESTS / "result/phase1348_c050_behavior"
MODELS = ("qwen3", "glm4", "deepseek7b")


def close(left, right, tolerance=1e-10):
    return abs(float(left) - float(right)) <= tolerance


def main():
    protocol = core.load(CONTRACT / "protocol/preregistration.json")
    manifest = core.load(OUT / "protocol/execution_manifest.json")
    final = core.load(OUT / "analysis/final.json")
    checks = {
        "contract": manifest["contract_sha256"] == protocol["contract_sha256"],
        "authorization": final["authorization"] in ("run_phase1349_c050_formation_field", "close_c050_behavior"),
    }
    qualified = []
    for model_name in MODELS:
        rows = core.rows(OUT / f"raw/{model_name}_behavior.jsonl")
        summary = core.load(OUT / f"analysis/{model_name}_summary.json")
        panels = Counter(row["panel"] for row in rows)
        core_rows = [row for row in rows if row["panel"] == "core_membership"]
        label_rows = [row for row in rows if row["panel"] == "label_only"]
        equality_rows = [row for row in rows if row["panel"] == "generic_equality"]
        groups = defaultdict(list)
        for row in core_rows:
            groups[row["quartet_key"]].append(row)
        interactions = []
        for group in groups.values():
            cells = {row["cell"]: row for row in group}
            interactions.append(
                cells["aa"]["semantic_margin"]
                - cells["ab"]["semantic_margin"]
                - cells["ba"]["semantic_margin"]
                + cells["bb"]["semantic_margin"]
            )
        checks[f"{model_name}_counts"] = panels == {
            "core_membership": 1536,
            "label_only": 768,
            "generic_equality": 768,
        }
        checks[f"{model_name}_core_accuracy"] = close(
            summary["core_metrics"]["accuracy"], sum(row["correct"] for row in core_rows) / len(core_rows)
        )
        checks[f"{model_name}_interaction"] = close(
            summary["core_metrics"]["median_interaction"], median(interactions)
        )
        checks[f"{model_name}_label"] = close(
            summary["null_metrics"]["label_only_accuracy"], sum(row["correct"] for row in label_rows) / len(label_rows)
        )
        checks[f"{model_name}_equality"] = close(
            summary["null_metrics"]["generic_equality_accuracy"],
            sum(row["correct"] for row in equality_rows) / len(equality_rows),
        )
        checks[f"{model_name}_finite"] = all(
            math.isfinite(value) for row in rows for value in row["scores"]
        )
        checks[f"{model_name}_executor"] = summary["executor"]["qualified"]
        if summary["qualified"]:
            qualified.append(model_name)
    checks["qualified"] = qualified == final["qualified_models"]
    checks["cross_model"] = final["cross_model_behavior_repetition"] == (len(qualified) >= 2)
    result = {
        "phase": 1348,
        "campaign": "C050",
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "all_checks_passed": all(checks.values()),
    }
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
