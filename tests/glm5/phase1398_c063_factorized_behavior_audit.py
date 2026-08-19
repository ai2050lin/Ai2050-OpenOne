#!/usr/bin/env python3
"""Independent audit for Phase1398."""
from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core

CONTRACT = TESTS / "result/phase1397_c063_identity_polarity_campaign_contract"
OUT = TESTS / "result/phase1398_c063_factorized_behavior"


def main() -> None:
    protocol = core.load(CONTRACT / "protocol/preregistration.json")
    summary = core.load(OUT / "analysis/qwen3_behavior_summary.json")
    final = core.load(OUT / "analysis/final.json")
    active = core.rows(OUT / "raw/active_behavior.jsonl")
    status = core.rows(OUT / "raw/status_behavior.jsonl")
    selected = core.rows(OUT / "material/eligible_factor_sets.jsonl")
    qualified = summary["qualified_families"]
    checks = {
        "counts": len(active) == 1728 and len(status) == 576,
        "four_answer_coverage": Counter(r["gold_position"] for r in active) == {0: 432, 1: 432, 2: 432, 3: 432},
        "finite": all(math.isfinite(score) for r in active + status for score in r["scores"]),
        "family_decisions_recomputed": all(result["qualified"] == all(result["checks"].values()) for result in summary["family_results"].values()),
        "breadth_recomputed": summary["breadth_checks"]["family_count"] == (len(qualified) >= protocol["material"]["minimum_qualified_families"]),
        "selected_only_qualified": all(r["family"] in qualified for r in selected),
        "selected_balance": (len(selected) == len(qualified) * protocol["material"]["selected_per_family"] and
                             Counter(r["partition"] for r in selected) ==
                             Counter({p: len(qualified) * protocol["material"]["selected_per_family_partition"]
                                      for p in protocol["material"]["partitions"] if qualified})),
        "numeric": summary["global"]["numeric_same_shape_max_abs_diff"] <= protocol["behavior"]["same_shape_repeat_max_abs_diff"],
        "authorization_faithful": final["authorization"] == ("run_phase1399_c063_state_swap_camera" if summary["behavior_qualified"] else "close_c063_at_behavior_gate"),
    }
    result = {"phase": 1398, "campaign": "C063", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    core.save(OUT / "audit/independent_final_audit.json", result)
    if not result["all_checks_passed"]:
        raise RuntimeError({k: v for k, v in checks.items() if not v})
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
