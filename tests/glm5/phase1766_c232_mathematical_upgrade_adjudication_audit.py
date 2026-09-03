#!/usr/bin/env python3
"""Independent audit for C232."""
from __future__ import annotations

import json
from pathlib import Path

import phase1757_c223_surface_transport_common as common

core = common.core
OUT = common.OUTS["C232"]


def main() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    final = core.load(OUT / "analysis/final.json")
    headline = final["headline"]
    checks = {"final": final["all_checks_passed"], "zero_of_four": headline["requirements_passed"] == 0 and headline["requirements_total"] == 4, "no_new_math": not headline["new_foundational_mathematics_authorized"], "stable_name": headline["theory"]["name"] == "Conditional Output Field Closure Theory", "typed_object": "surface-indexed" in headline["theory"]["current_object"], "producer_hash": core.sha(Path(__file__).with_name("phase1766_c232_mathematical_upgrade_adjudication.py")) == protocol["producer_sha256"]}
    audit = {"phase": 1766, "campaign": "C232", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "authorization": final["next_authorization"]}
    core.save(OUT / "audit/independent_final_audit.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
