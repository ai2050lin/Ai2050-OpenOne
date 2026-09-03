#!/usr/bin/env python3
"""Independent audit for C227."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import phase1757_c223_surface_transport_common as common

core = common.core
OUT = common.OUTS["C227"]


def main() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    final = core.load(OUT / "analysis/final.json")
    rows = core.rows(OUT / "analysis/lockbox_rows.jsonl")
    fields = np.load(OUT / "analysis/selected_truth_fields.float16.npy", mmap_mode="r")
    checks = {
        "final": final["all_checks_passed"], "rows": len(rows) == 360,
        "lockbox": {row["unit"] for row in rows} == {6, 7, 8},
        "methods": set(row["method"] for row in rows) == set(protocol["methods"]),
        "fields": fields.shape == (45, 2, 3, 4, 6, 2560),
        "confirmation_not_rewritten": final["headline"]["confirmation_gate_passed"] is False,
        "producer_hash": core.sha(Path(__file__).with_name("phase1761_c227_surface_transport_lockbox.py")) == protocol["producer_sha256"],
    }
    audit = {"phase": 1761, "campaign": "C227", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "authorization": final["next_authorization"]}
    core.save(OUT / "audit/independent_final_audit.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
