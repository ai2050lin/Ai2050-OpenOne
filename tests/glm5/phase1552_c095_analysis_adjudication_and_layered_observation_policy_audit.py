#!/usr/bin/env python3
"""Independent audit for Phase1552."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1552_c095_analysis_adjudication_and_layered_observation_policy"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    policy = core.load(OUT / "protocol/layered_observation_policy_v2.json")
    pre = core.load(OUT / "audit/preimplementation_audit.json")
    checks = {
        "preimplementation": pre["all_checks_passed"],
        "inherits_1482": policy["inherits"]["phase"] == 1482,
        "five_missingness_codes": set(policy["missingness"]) == {"M_BEHAVIOR", "M_OUTPUT", "M_CAUSAL", "M_EXTERNAL", "M_BLIND"},
        "all_strata_retained": "All behavior strata remain" in policy["route_rules"][1],
        "no_hard_erasure": "No scalar hard gate" in policy["route_rules"][2],
        "retrospective_scope": "retrospective repetition" in policy["evidence_layers"]["O3_cross_partition_repetition"],
        "raw_coordinates_only": "raw 2560 coordinates" in policy["evidence_layers"]["O4_coordinate_structure"],
        "analysis_corrections": len(policy["analysis_adjudication"]["corrections"]) == 9,
        "asset_hashes": all(core.sha(ROOT / value["path"]) == value["sha256"] for value in policy["source_assets"].values()),
        "claim_boundary": {"pure semantic vector", "causal circuit", "identified neuron group", "new mathematics"}.issubset(policy["forbidden_claims"]),
        "authorization": core.load(OUT / "analysis/final.json")["authorization"] == "run_phase1553_c095_existing_field_batch_mining_contract",
    }
    result = {"phase": 1552, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "checks": checks}
    core.save(OUT / "audit/independent_final_audit.json", result)
    if not result["all_checks_passed"]:
        raise RuntimeError(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
