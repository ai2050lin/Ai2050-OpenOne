#!/usr/bin/env python3
"""Independent integrity and arithmetic audit for C159."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/glm5/result/phase1693_c159_natural_isomorphic_dual_graph_atlas"


def load(path):
    return json.loads((OUT / path).read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(line) for line in (OUT / path).read_text(encoding="utf-8").splitlines()]


def main():
    protocol = load("protocol/preregistration.json")
    atlas = load("analysis/atlas.json")
    behavior = rows("raw/qwen3_behavior_index.jsonl")
    pair_index = rows("analysis/late_half_difference_index.jsonl")
    raw = np.load(OUT / "raw/qwen3_six_role_all_checkpoint.bf16.npy", mmap_mode="r")
    late = np.load(OUT / "analysis/late_half_difference.float16.npy", mmap_mode="r")
    accuracies = {panel: float(np.mean([row["correct"] for row in behavior if row["panel"] == panel])) for panel in ("natural_lexical", "isomorphic_nonce")}
    checks = {
        "contract": load("audit/internal_contract_audit.json")["all_checks_passed"],
        "run": load("audit/internal_run_audit.json")["all_checks_passed"],
        "analysis": load("audit/internal_analysis_audit.json")["all_checks_passed"],
        "cases": len(behavior) == protocol["cases"] == 1536,
        "role_shape": list(raw.shape) == [1536, 6, 38, 2560],
        "late_shape": list(late.shape) == [768, 11, 6, 2560],
        "pairs": len(pair_index) == 768,
        "accuracy_natural": abs(accuracies["natural_lexical"] - atlas["behavior"]["panel"]["natural_lexical"]) < 1e-12,
        "accuracy_nonce": abs(accuracies["isomorphic_nonce"] - atlas["behavior"]["panel"]["isomorphic_nonce"]) < 1e-12,
        "typed_failure": atlas["natural_nonce_response"]["passed"] is False,
        "no_human_overclaim": "human blind rating missing" in protocol["naturalness"],
        "scope": "not natural world-knowledge" in protocol["claim_boundary"],
    }
    report = {"phase": 1693, "campaign": "C159", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "scientific_behavior_qualified": atlas["behavior"]["qualified"], "scientific_cross_panel_response_passed": atlas["natural_nonce_response"]["passed"], "authorization": "memo_then_C160"}
    (OUT / "audit/independent_final_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
