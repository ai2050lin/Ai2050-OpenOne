#!/usr/bin/env python3
"""Independent audit for C190 phrase x wrapper factorial."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; TESTS = ROOT / "tests/glm5"; OUT = TESTS / "result/phase1724_c190_relation_phrase_wrapper_factorial"; sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main():
    protocol = core.load(OUT / "protocol/preregistration.json"); final = core.load(OUT / "analysis/final.json"); report = core.load(OUT / "analysis/factorial_response_atlas.json"); behavior = core.rows(OUT / "raw/behavior_index.jsonl"); response = np.load(OUT / "raw/off_diagonal_relation_response.float16.npy", mmap_mode="r"); producer = Path(__file__).with_name("phase1724_c190_relation_phrase_wrapper_factorial.py")
    checks = {
        "closed": final["status"] == "closed" and final["all_checks_passed"],
        "behavior_336": len(behavior) == 336,
        "factor_balance": sum(row["phrase_variant"] != row["wrapper_variant"] for row in behavior) == 168,
        "response_shape": response.shape[1:] == (64, 6, 2560),
        "missing_aware": report["observed_cells"] + len(report["registered_missing"]) == report["possible_cells"],
        "pair_support": min(len(report[name]) for name in ("wrapper_pairs", "phrase_pairs", "vocabulary_pairs")) >= 20,
        "finite": bool(np.isfinite(list(report["medians"].values()) + [report["wrapper_minus_phrase_similarity"]]).all()),
        "hash": core.sha(producer) == protocol["producer_sha256"],
    }
    result = {"phase": 1724, "campaign": "C190", "checks": checks, "all_checks_passed": all(checks.values()), "authorization": final["next_authorization"]}
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
