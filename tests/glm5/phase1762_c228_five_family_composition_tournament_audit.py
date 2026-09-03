#!/usr/bin/env python3
"""Independent audit for C228."""
from __future__ import annotations

import json
from pathlib import Path

import phase1757_c223_surface_transport_common as common

core = common.core
OUT = common.OUTS["C228"]


def main() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    final = core.load(OUT / "analysis/final.json")
    freeze = core.load(OUT / "protocol/composition_model_freeze.json")
    rows = core.rows(OUT / "analysis/confirmation_rows.jsonl")
    checks = {"final": final["all_checks_passed"], "rows": len(rows) == 300, "lockbox_sealed": max(row["unit"] for row in rows) == 5, "five_families": len(freeze["selected_by_family"]) == 5, "template_hash": core.sha(OUT / "protocol/interaction_templates.npz") == freeze["template_sha256"], "producer_hash": core.sha(Path(__file__).with_name("phase1762_c228_five_family_composition_tournament.py")) == protocol["producer_sha256"]}
    audit = {"phase": 1762, "campaign": "C228", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "authorization": final["next_authorization"]}
    core.save(OUT / "audit/independent_final_audit.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()

