#!/usr/bin/env python3
"""Independent audit for Phase1535."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1535_c091_external_human_material_and_analysis_adjudication"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    report = core.load(OUT / "analysis/material_and_analysis_adjudication.json")
    manifest = core.load(OUT / "protocol/source_manifest.json")
    a = pd.read_csv(OUT / "source/word_pair_A.csv", encoding="gb18030")
    b = pd.read_csv(OUT / "source/word_pair_B.csv", encoding="gb18030")
    checks = {
        "source_hashes": all(core.sha(OUT / "source" / name) == item["sha256"] for name, item in manifest.items()),
        "source_sizes": all((OUT / "source" / name).stat().st_size == item["bytes"] for name, item in manifest.items()),
        "material_total": len(a) + len(b) == 1000,
        "similar_count": int((a.relation == "相似关系").sum()) == 100,
        "class_count": int((b.relation == "类别关系").sum()) == 100,
        "whole_part_count": int((b.relation == "整体-部分关系").sum()) == 100,
        "adjudication_retained": len(report["uploaded_analysis_adjudication"]["retained"]) >= 5,
        "adjudication_corrected": len(report["uploaded_analysis_adjudication"]["corrected"]) >= 7,
        "post_training_scope": report["checks"]["post_training_release"],
        "no_model": report["checks"]["model_not_loaded"] and report["checks"]["hidden_not_accessed"],
        "authorization": core.load(OUT / "analysis/final.json")["authorization"] == "run_phase1536_c091_human_validated_chinese_relation_contract",
    }
    result = {
        "phase": 1535,
        "passed": sum(checks.values()),
        "total": len(checks),
        "all_checks_passed": all(checks.values()),
        "checks": checks,
    }
    core.save(OUT / "audit/independent_final_audit.json", result)
    if not result["all_checks_passed"]:
        raise RuntimeError(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
