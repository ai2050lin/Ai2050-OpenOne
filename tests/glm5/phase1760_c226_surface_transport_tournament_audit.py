#!/usr/bin/env python3
"""Independent audit for C226."""
from __future__ import annotations

import json
from pathlib import Path

import phase1757_c223_surface_transport_common as common

core = common.core
OUT = common.OUTS["C226"]


def main() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    final = core.load(OUT / "analysis/final.json")
    freeze = core.load(OUT / "protocol/model_freeze.json")
    rows = core.rows(OUT / "analysis/confirmation_rows.jsonl")
    checks = {
        "final": final["all_checks_passed"], "rows": len(rows) == 225,
        "calibration_only_fit": protocol["fit"]["families"] == list(common.CALIBRATION_FAMILIES),
        "target_only_selection": protocol["selection"]["families"] == list(common.TARGET_FAMILIES),
        "lockbox_sealed": max(row["unit"] for row in rows) == 5,
        "three_models_frozen": len(freeze["selected_by_target_surface"]) == 3,
        "parameter_hash": core.sha(OUT / "protocol/fitted_parameters.npz") == freeze["parameter_sha256"],
        "producer_hash": core.sha(Path(__file__).with_name("phase1760_c226_surface_transport_tournament.py")) == protocol["producer_sha256"],
    }
    audit = {"phase": 1760, "campaign": "C226", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "authorization": final["next_authorization"]}
    core.save(OUT / "audit/independent_final_audit.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
