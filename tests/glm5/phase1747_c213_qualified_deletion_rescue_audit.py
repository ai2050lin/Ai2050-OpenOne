#!/usr/bin/env python3
"""Independent audit for C213."""
from __future__ import annotations

import json
from pathlib import Path

import phase1739_c205_response_ecology_common as common

core = common.core
OUT = common.C213


def main() -> None:
    final = core.load(OUT / "analysis/final.json")
    protocol = core.load(OUT / "protocol/preregistration.json")
    checks = {"final": final["all_checks_passed"], "typed_not_tested": protocol["status"] == "typed_not_tested", "no_causal_run": not protocol["causal_tests_run"], "path_not_retrofit": "post hoc" in protocol["reason"], "producer_hash": core.sha(Path(__file__).with_name("phase1747_c213_qualified_deletion_rescue.py")) == protocol["producer_sha256"]}
    audit = {"phase": 1747, "campaign": "C213", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "authorization": final["next_authorization"]}
    core.save(OUT / "audit/independent_final_audit.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
