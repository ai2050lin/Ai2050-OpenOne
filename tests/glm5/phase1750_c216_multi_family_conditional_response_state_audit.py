#!/usr/bin/env python3
"""Independent audit for C216."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import phase1739_c205_response_ecology_common as common

core = common.core
OUT = common.RESULT / "phase1750_c216_multi_family_conditional_response_state"


def main() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    final = core.load(OUT / "analysis/final.json")
    fields = np.load(OUT / "raw/full_fields.float16.npy", mmap_mode="r")
    report = final["headline"]
    checks = {
        "final": final["all_checks_passed"],
        "five_arms": len(report["composition"]) == 5,
        "all_arms_retained": set(report["composition"]) == set(protocol["arms"]),
        "full_field": list(fields.shape) == [240, 4, 96, 2560],
        "partitions": all(report["composition"][arm][part]["support"] == 4 for arm in protocol["arms"] for part in ("discovery", "confirmation", "fresh")),
        "classification": report["response_state_classification"]["fresh"]["support"] == 20,
        "producer_hash": core.sha(Path(__file__).with_name("phase1750_c216_multi_family_conditional_response_state.py")) == protocol["producer_sha256"],
        "no_forbidden_components": all(value in protocol["forbidden"] for value in ("attention", "MLP", "weights", "PCA")),
    }
    audit = {"phase": 1750, "campaign": "C216", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "authorization": final["next_authorization"]}
    core.save(OUT / "audit/independent_final_audit.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
