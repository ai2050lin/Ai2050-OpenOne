#!/usr/bin/env python3
"""Independent audit for C221."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import phase1739_c205_response_ecology_common as common

core = common.core
OUT = common.RESULT / "phase1755_c221_independent_response_state_prediction"


def main() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    final = core.load(OUT / "analysis/final.json")
    report = final["headline"]
    fields = np.load(OUT / "raw/full_fields.float16.npy", mmap_mode="r")
    checks = {
        "final": final["all_checks_passed"],
        "third_material": protocol["status"].startswith("third_material"),
        "fixed_subset": protocol["selected_subset"]["name"] == "q24_q25_relation_boundary",
        "two_claims_separated": "classification" in report and "exact_signed_field_prediction" in report,
        "typed_causal_gate": report["causal_eligible"] == (report["behavior"]["passed"] and report["classification_passed"] and report["exact_prediction_passed"]),
        "full_field": list(fields.shape) == [160, 4, 96, 2560],
        "producer_hash": core.sha(Path(__file__).with_name("phase1755_c221_independent_response_state_prediction.py")) == protocol["producer_sha256"],
    }
    audit = {"phase": 1755, "campaign": "C221", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "authorization": final["next_authorization"]}
    core.save(OUT / "audit/independent_final_audit.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
