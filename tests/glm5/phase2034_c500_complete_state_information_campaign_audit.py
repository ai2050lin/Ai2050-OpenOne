#!/usr/bin/env python3
"""Independent audit for C485-C500 / Phase2019-2034."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase2019_c485_c500_complete_state_information_campaign as p


checks: list[dict] = []


def check(name: str, condition: bool, detail=None) -> None:
    checks.append({"name": name, "passed": bool(condition), "detail": detail})


def finite(value) -> bool:
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, list):
        return all(finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


finals = {name: p.final(name) for name in p.PHASES}
for offset, name in enumerate(p.PHASES):
    expected = 2019 + offset
    check(f"{name}.phase", finals[name]["phase"] == expected, finals[name]["phase"])
    check(f"{name}.closed", finals[name]["status"] == "closed")
    if name != "C487":
        check(f"{name}.audit_pass", finals[name]["all_checks_passed"])

check("C487.expected_premodel_failure", not finals["C487"]["all_checks_passed"])
check("C487.failure_is_role_occurrence", finals["C487"]["headline"]["role_occurrence"]["secondary"] is False)
check("C487.model_not_run", not (p.OUTS["C487"] / "raw/behavior.jsonl").exists())
repair = p.load(p.OUTS["C488"] / "audit/premodel_role_repair.json")
check("C488.repair_count", repair["repair_count"] == 240, repair["repair_count"])
check("C488.repair_metadata_only", repair["prompts_answers_partitions_unchanged"] is True)
check("C488.role_reaudit", all(repair["role_reaudit"].values()))
check("C488.raw_complete", sum(1 for _ in (p.OUTS["C488"] / "raw/behavior.jsonl").open(encoding="utf-8")) == 5280)
check("C488.accuracy", abs(finals["C488"]["headline"]["accuracy"] - 0.9638257575757576) < 1e-12)
check("C488.eleven_families", len(finals["C488"]["headline"]["eligible_families"]) == 11)
check("C488.new_nested_behavior", finals["C488"]["headline"]["family_accuracy"]["nested_composition"] == 1.0)
check("C488.new_temporal_behavior", finals["C488"]["headline"]["family_accuracy"]["temporal_composition"] == 1.0)
check("C488.graph_behavior_boundary", abs(finals["C488"]["headline"]["family_accuracy"]["typed_graph_path"] - 0.7875) < 1e-12)

check("C489.role_shape", finals["C489"]["headline"]["role_shape"] == [5280, 38, 6, 2560])
check("C489.full_shape", finals["C489"]["headline"]["full_shape"] == [2112, 10, 120, 2560])
check("C489.bf16", finals["C489"]["headline"]["quantization"]["has_bf16_parameters"])
check("C489.no_quant", not finals["C489"]["headline"]["quantization"]["has_quantized_modules"])

check("C490.programs", finals["C490"]["headline"]["programs"] == 330)
check("C490.shape", finals["C490"]["headline"]["shape"] == [330, 16, 10, 6, 2560])
check("C490.complete_correct", finals["C490"]["headline"]["behavior_strata"]["complete_correct"] == 273)
check("C490.mixed_retained", finals["C490"]["headline"]["behavior_strata"]["mixed"] == 57)
check("C490.reconstruction", finals["C490"]["headline"]["max_float32_reconstruction_error"] < 2e-6)
check("C491.all_families", finals["C491"]["headline"]["families"] == 11)
check("C491.zero_order_dominant", finals["C491"]["headline"]["zero_to_nonzero_rms_ratio"] > 10.0)

c492 = finals["C492"]["headline"]
check("C492.candidate", c492["complete_spectrum_candidate"])
for panel in ("within", "family", "report"):
    check(f"C492.{panel}.gain", c492["spectrum_gains_over_identity"][panel] >= 0.01, c492["spectrum_gains_over_identity"][panel])
    check(f"C492.{panel}.spectrum_beats_identity", c492["metrics"][panel]["spectrum"]["nrmse"] < c492["metrics"][panel]["identity"]["nrmse"])
check("C492.family_same_beats_spectrum", c492["metrics"]["family"]["same"]["nrmse"] < c492["metrics"]["family"]["spectrum"]["nrmse"])

c493 = finals["C493"]["headline"]
check("C493.not_candidate", not c493["all_role_candidate"])
for panel in ("within", "family", "report"):
    check(f"C493.{panel}.negative_increment", c493["gains_over_best_local"][panel] < 0.0, c493["gains_over_best_local"][panel])

c494 = finals["C494"]["headline"]
check("C494.not_candidate", not c494["all_token_candidate"])
check("C494.report_roll_better", c494["gains_over_token_roll"]["report"] < 0.0)

c495 = finals["C495"]["headline"]
check("C495.strong_baseline_used", "spectrum" in c495["metrics"]["family"])
check("C495.not_candidate", not c495["cross_coordinate_candidate"])
check("C495.family_positive_signal", c495["gains"]["family"]["over_spectrum"] > 0.0)
check("C495.report_below_gate", c495["gains"]["report"]["over_spectrum"] < 0.01)
check("C495.beats_shuffle_family", c495["gains"]["family"]["over_shuffle"] > 0.005)
check("C495.beats_roll_family", c495["gains"]["family"]["over_roll"] > 0.005)

c496 = finals["C496"]["headline"]
check("C496.program_not_candidate", not c496["program_guard_candidate"])
check("C496.token_not_candidate", not c496["best_baseline_token_candidate"])
check("C496.program_gain_small", c496["program_gains"]["family"]["over_spectrum"] < 0.005)
check("C496.token_report_negative", c496["token_gains"]["report"]["over_spectrum"] < 0.0)

for name, expected_family in (("C497", "nested_composition"), ("C498", "typed_graph_path"), ("C499", "temporal_composition")):
    headline = finals[name]["headline"]
    check(f"{name}.family", headline["family"] == expected_family)
    check(f"{name}.lockbox_programs", headline["lockbox_programs"] == 6)
    check(f"{name}.not_candidate", not headline["panel_candidate"])
    check(f"{name}.identity_beats_all_roles", headline["high_order_metrics"]["identity"]["nrmse"] < headline["high_order_metrics"]["all_roles"]["nrmse"])

c500 = finals["C500"]["headline"]
check("C500.only_complete_spectrum_gate", c500["gates"]["complete_spectrum"] and sum(bool(value) for value in c500["gates"].values()) == 1)
check("C500.no_predictive_chain", not c500["predictive_candidate"])
check("C500.causal_not_authorized", c500["causal"]["authorized"] is False and c500["causal"]["ran"] is False)
check("C500.new_math_closed", c500["new_math_gate"] is False)
check("C500.cleanup_bytes", c500["cleanup_bytes"] == 21050456032, c500["cleanup_bytes"])
cleanup = p.load(p.OUTS["C500"] / "audit/cleanup.json")
check("C500.cleanup_count", len(cleanup) == 14, len(cleanup))
check("C500.cleanup_hashes", all(len(row["sha256"]) == 64 for row in cleanup))
check("C500.cleanup_absent", all(not (ROOT / row["path"]).exists() for row in cleanup))

visual = p.load(p.VISUAL)
check("visual.schema", visual["schema"] == "c500.complete-state-information-ladder.v1")
check("visual.rows", len(visual["rows"]) == 1084, len(visual["rows"]))
check("visual.all_coordinates", all(len(row["values"]) == 2560 for row in visual["rows"]))
check("visual.finite", finite(visual))
registry = p.load(p.REGISTRY)
entries = [row for row in registry["datasets"] if row["id"] == "c500_complete_state_information_ladder"]
check("registry.unique", len(entries) == 1)
check("registry.path", bool(entries) and entries[0]["source_path"] == "/vis_data/research_kernel/c500_complete_state_information_ladder.json")
check("registry.coordinates", bool(entries) and entries[0]["coordinate_count"] == 2560)

passed = sum(row["passed"] for row in checks)
adjudication = {
    "strict_gates": c500["gates"],
    "qualified_result": "complete_zero_plus_fifteen_effect_spectrum_improves_local_checkpoint_prediction",
    "not_qualified": ["unconditional all-role mixing", "registered all-token linear increment", "broad cross-coordinate closure", "construction guard", "nested/graph/temporal high-order transfer", "causal writer", "cross-model closure", "new mathematics"],
    "central_boundary": "The positive C492 result is a prospective local transition law in a complete factorial accounting basis. It is not a semantic algebra, a minimal state, a unique circuit, or high-order composition closure.",
}
report = {"status": "passed" if passed == len(checks) else "failed", "passed": passed, "total": len(checks), "checks": checks, "adjudication": adjudication}
p.save(p.OUTS["C500"] / "audit/independent_audit.json", report)
visual["strict_adjudication"] = adjudication
visual["claim_boundary"] = adjudication["central_boundary"]
p.save(p.VISUAL, visual)
print(json.dumps({"status": report["status"], "passed": passed, "total": len(checks), "failed": [row["name"] for row in checks if not row["passed"]]}, ensure_ascii=False))
if passed != len(checks):
    raise SystemExit(1)
