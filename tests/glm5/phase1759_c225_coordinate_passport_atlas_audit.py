#!/usr/bin/env python3
"""Independent audit for C225."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import phase1757_c223_surface_transport_common as common

core = common.core
OUT = common.OUTS["C225"]


def main() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    final = core.load(OUT / "analysis/final.json")
    cubes = np.load(OUT / "analysis/response_cubes.float16.npy", mmap_mode="r")
    passport = np.load(OUT / "analysis/passport_mean.float16.npy", mmap_mode="r")
    rows = core.rows(OUT / "analysis/raw_cross_surface_rows.jsonl")
    checks = {
        "final": final["all_checks_passed"], "cubes": cubes.shape == (8, 4, 9, 3, 4, 6, 2560),
        "passport": passport.shape == (8, 4, 3, 4, 6, 2560), "cross_rows": len(rows) == 72,
        "lockbox_sealed": max(row["unit"] for row in rows) == 5 and protocol["forbidden_units"] == [6, 7, 8],
        "all_coordinates": cubes.shape[-1] == 2560,
        "producer_hash": core.sha(Path(__file__).with_name("phase1759_c225_coordinate_passport_atlas.py")) == protocol["producer_sha256"],
    }
    audit = {"phase": 1759, "campaign": "C225", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "authorization": final["next_authorization"]}
    core.save(OUT / "audit/independent_final_audit.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
