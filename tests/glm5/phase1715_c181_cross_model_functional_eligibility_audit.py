#!/usr/bin/env python3
"""Independent audit for C181."""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1715_c181_cross_model_functional_eligibility"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main():
    protocol = core.load(OUT / "protocol/preregistration.json")
    summary = core.load(OUT / "analysis/summary.json")
    final = core.load(OUT / "analysis/final.json")
    checks = {
        "closed": final["status"] == "closed" and final["all_checks_passed"],
        "sequential": protocol["sequential_loading"],
        "models": set(summary["models"]) == {"qwen3", "glm4", "deepseek7b"},
        "sequence_score": "sum log probability" in protocol["score"],
        "typed": isinstance(summary["cross_model_hidden_eligible"], bool),
        "hash": core.sha(Path(__file__).with_name("phase1715_c181_cross_model_functional_eligibility.py")) == protocol["producer_sha256"],
    }
    result = {"phase": 1715, "campaign": "C181", "checks": checks, "all_checks_passed": all(checks.values()), "authorization": final["next_authorization"]}
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
