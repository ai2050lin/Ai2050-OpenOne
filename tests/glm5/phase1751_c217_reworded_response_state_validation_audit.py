#!/usr/bin/env python3
"""Independent audit for C217."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import phase1739_c205_response_ecology_common as common

core = common.core
OUT = common.RESULT / "phase1751_c217_reworded_response_state_validation"


def main() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json"); final = core.load(OUT / "analysis/final.json"); report = final["headline"]
    fields = np.load(OUT / "raw/full_fields.float16.npy", mmap_mode="r")
    checks = {"final": final["all_checks_passed"], "frozen_source": protocol["source_templates"].startswith("C216 discovery"), "twenty_tests": report["frozen_template_classification"]["support"] == 20, "five_arms": len(report["composition"]) == 5, "full_field": list(fields.shape) == [80, 4, 96, 2560], "no_refit": "template refitting" in protocol["forbidden"], "producer_hash": core.sha(Path(__file__).with_name("phase1751_c217_reworded_response_state_validation.py")) == protocol["producer_sha256"]}
    audit = {"phase": 1751, "campaign": "C217", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "authorization": final["next_authorization"]}
    core.save(OUT / "audit/independent_final_audit.json", audit); print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
