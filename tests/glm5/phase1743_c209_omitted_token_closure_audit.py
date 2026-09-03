#!/usr/bin/env python3
"""Independent audit for C209."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import phase1739_c205_response_ecology_common as common

core = common.core
OUT = common.C209


def main() -> None:
    final = core.load(OUT / "analysis/final.json")
    protocol = core.load(OUT / "protocol/preregistration.json")
    report = final["headline"]
    checks = {"final": final["all_checks_passed"], "models_frozen": tuple(protocol["models"]) == ("boundary_only", "six_roles", "six_roles_plus_nonrole", "six_roles_plus_quartiles"), "fresh_once": "fresh unit 6" in protocol["selection"], "odd_even": set(report["components"]) == {"odd", "even"}, "operators": len(list((OUT / "analysis/operators").glob("*.npy"))) == 8, "finite": all(np.isfinite(np.load(path)).all() for path in (OUT / "analysis/operators").glob("*.npy")), "producer_hash": core.sha(Path(__file__).with_name("phase1743_c209_omitted_token_closure.py")) == protocol["producer_sha256"]}
    audit = {"phase": 1743, "campaign": "C209", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "authorization": final["next_authorization"]}
    core.save(OUT / "audit/independent_final_audit.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
