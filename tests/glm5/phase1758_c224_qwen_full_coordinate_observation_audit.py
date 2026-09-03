#!/usr/bin/env python3
"""Independent audit for C224."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import phase1757_c223_surface_transport_common as common

core = common.core
OUT = common.OUTS["C224"]


def main() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    final = core.load(OUT / "analysis/final.json")
    fields = np.load(OUT / "raw/full_fields.float16.npy", mmap_mode="r")
    roles = np.load(OUT / "raw/role_states.float16.npy", mmap_mode="r")
    behavior = core.rows(OUT / "raw/behavior_index.jsonl")
    checks = {
        "final": final["all_checks_passed"],
        "behavior": len(behavior) == 2304,
        "fields": fields.shape == (1152, 4, 128, 2560),
        "roles": roles.shape == (1152, 4, 6, 2560),
        "all_families": set(row["family"] for row in behavior) == set(common.FAMILIES),
        "all_surfaces": set(row["surface"] for row in behavior) == set(common.SURFACES),
        "gpu_released_metadata": core.load(OUT / "raw/run_metadata.json")["quantization"]["has_bf16_parameters"],
        "producer_hash": core.sha(Path(__file__).with_name("phase1758_c224_qwen_full_coordinate_observation.py")) == protocol["producer_sha256"],
    }
    audit = {"phase": 1758, "campaign": "C224", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "authorization": final["next_authorization"]}
    core.save(OUT / "audit/independent_final_audit.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
