#!/usr/bin/env python3
"""Independent closure audit for the C130-C132 composition behavior stage."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PUBLIC = ROOT / "frontend/public/vis_data/research_kernel/c109_role_state_field_atlas.json"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    paths = {
        "c130": TESTS / "result/phase1664_c130_composed_precedence_typed_transition",
        "c131": TESTS / "result/phase1665_c131_composed_precedence_repaired_transition",
        "c132": TESTS / "result/phase1666_c132_fixed_frame_composed_precedence",
    }
    memo = MEMO.read_text(encoding="utf-8")
    c129_closure = core.load(TESTS / "result/phase1663_c129_direct_precedence_typed_transition/analysis/closure.json")
    checks = {
        "c130_contract": core.load(paths["c130"] / "audit/independent_contract_audit.json")["all_checks_passed"],
        "c130_closed": core.load(paths["c130"] / "audit/independent_behavior_failure_audit.json")["all_checks_passed"],
        "c131_contract": core.load(paths["c131"] / "audit/independent_contract_audit.json")["all_checks_passed"],
        "c131_closed": core.load(paths["c131"] / "audit/independent_behavior_failure_audit.json")["all_checks_passed"],
        "c132_contract": core.load(paths["c132"] / "audit/independent_contract_audit.json")["all_checks_passed"],
        "c132_closed": core.load(paths["c132"] / "audit/independent_behavior_failure_audit.json")["all_checks_passed"],
        "no_hiddenstate": all(not (path / "raw/qwen3_uniform_role_checkpoints.bf16.npy").exists() for path in paths.values()),
        "no_confirmation": all(not (path / "analysis/confirmation.json").exists() for path in paths.values()),
        "heatmap_unchanged": core.sha(PUBLIC) == c129_closure["heatmap"]["sha256"],
        "memo_phase1664_once": memo.count("## Phase 1664:") == 1,
        "memo_phase1665_once": memo.count("## Phase 1665:") == 1,
        "memo_phase1666_once": memo.count("## Phase 1666:") == 1,
    }
    report = {"phase": 1666, "campaign": "C130-C132", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "scientific_result": "three behavior routes failed under frozen gates; no HiddenState comparison was authorized", "authorization": "composition_branch_closed_start_new_observation_contract" if all(checks.values()) else "stop"}
    core.save(paths["c132"] / "audit/major_stage_audit.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
