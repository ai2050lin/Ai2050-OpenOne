#!/usr/bin/env python3
"""Independent audit for C211."""
from __future__ import annotations

import json
from pathlib import Path

import phase1739_c205_response_ecology_common as common

core = common.core
OUT = common.C211


def main() -> None:
    final = core.load(OUT / "analysis/final.json")
    protocol = core.load(OUT / "protocol/preregistration.json")
    checks = {"final": final["all_checks_passed"], "five_routes": len(final["headline"]["route_rows"]) == 5, "posthoc_boundary": "cannot retroactively" in protocol["policy"], "candidate_only": "observational candidate" in final["headline"]["interpretation"], "producer_hash": core.sha(Path(__file__).with_name("phase1745_c211_flagship_route_ledger.py")) == protocol["producer_sha256"]}
    audit = {"phase": 1745, "campaign": "C211", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "authorization": final["next_authorization"]}
    core.save(OUT / "audit/independent_final_audit.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
