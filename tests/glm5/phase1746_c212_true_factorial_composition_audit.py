#!/usr/bin/env python3
"""Independent audit for C212."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import phase1739_c205_response_ecology_common as common

core = common.core
OUT = common.C212


def main() -> None:
    final = core.load(OUT / "analysis/final.json")
    protocol = core.load(OUT / "protocol/preregistration.json")
    material = core.rows(OUT / "material/cases.jsonl")
    checks = {"final": final["all_checks_passed"], "material": len(material) == 192, "balance": sum(row["gold_position"] == 0 for row in material) == 96, "two_arms": {row["arm"] for row in material} == {"surface_factorial", "path_factorial"}, "hidden_shape": np.load(OUT / "raw/role_states.float16.npy", mmap_mode="r").shape == (96, 4, 6, common.DIM), "combined_not_fit": "without fitting" in protocol["factorial_prediction"], "producer_hash": core.sha(Path(__file__).with_name("phase1746_c212_true_factorial_composition.py")) == protocol["producer_sha256"]}
    audit = {"phase": 1746, "campaign": "C212", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "authorization": final["next_authorization"]}
    core.save(OUT / "audit/independent_final_audit.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
