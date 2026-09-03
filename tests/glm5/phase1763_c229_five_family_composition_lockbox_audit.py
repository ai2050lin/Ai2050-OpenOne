#!/usr/bin/env python3
"""Independent audit for C229."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import phase1757_c223_surface_transport_common as common

core = common.core
OUT = common.OUTS["C229"]


def main() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    final = core.load(OUT / "analysis/final.json")
    rows = core.rows(OUT / "analysis/lockbox_rows.jsonl")
    atlas = np.load(OUT / "analysis/prediction_truth_interaction.float16.npy", mmap_mode="r")
    checks = {"final": final["all_checks_passed"], "rows": len(rows) == 300, "lockbox": {row["unit"] for row in rows} == {6, 7, 8}, "five_family_account": len(final["headline"]["by_family"]) == 5, "atlas": atlas.shape == (60, 3, 3, 4, 6, 2560), "producer_hash": core.sha(Path(__file__).with_name("phase1763_c229_five_family_composition_lockbox.py")) == protocol["producer_sha256"]}
    audit = {"phase": 1763, "campaign": "C229", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "authorization": final["next_authorization"]}
    core.save(OUT / "audit/independent_final_audit.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
