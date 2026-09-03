#!/usr/bin/env python3
"""Independent factorial-field audit for C162."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/glm5/result/phase1696_c162_linguistic_program_field"


def load(path):
    return json.loads((OUT / path).read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(line) for line in (OUT / path).read_text(encoding="utf-8").splitlines()]


def main():
    cases = rows("compiled/qwen3.jsonl")
    behavior = rows("raw/qwen3_behavior_index.jsonl")
    terms = rows("analysis/term_index.jsonl")
    fields = np.load(OUT / "analysis/unit_term_fields.float16.npy", mmap_mode="r")
    report = load("analysis/program_field.json")
    accuracy = float(np.mean([row["correct"] for row in behavior]))
    checks = {
        "contract": load("audit/internal_contract_audit.json")["all_checks_passed"],
        "run": load("audit/internal_run_audit.json")["all_checks_passed"],
        "analysis": load("audit/internal_analysis_audit.json")["all_checks_passed"],
        "cases": len(cases) == len(behavior) == 2048,
        "field_shape": list(fields.shape) == [8, 21, 11, 6, 2560],
        "terms": sum(row["order"] == 1 for row in terms) == 6 and sum(row["order"] == 2 for row in terms) == 15,
        "behavior": abs(accuracy - report["behavior"]["global"]) < 1e-12,
        "first_terms": set(report["passing_first_order"]) == {"coreference_form", "attitude", "action", "negation_scope"},
        "gates": report["transfer_passed"] and all(report["gates"].values()),
        "scope": "not a complete syntax" in report["claim_boundary"],
    }
    audit = {"phase": 1696, "campaign": "C162", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "scientific_transfer_passed": report["transfer_passed"], "authorization": "memo_then_C163"}
    (OUT / "audit/independent_final_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    if not audit["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
