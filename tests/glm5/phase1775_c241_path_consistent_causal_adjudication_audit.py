#!/usr/bin/env python3
"""Independent audit for C241."""
from __future__ import annotations

import json
from pathlib import Path

import phase1768_c234_event_campaign_common as common

core = common.core
OUT = common.OUTS["C241"]


def main() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    final = core.load(OUT / "analysis/final.json")
    report = final["headline"]
    checks = {"final": final["all_checks_passed"], "typed": report["status"] == "typed_not_tested", "inputs_failed": not report["C238_unseen_event_passed"] and not report["C240_composition_passed"], "no_patch": not report["patch_executed"], "producer_hash": core.sha(Path(__file__).with_name("phase1775_c241_path_consistent_causal_adjudication.py")) == protocol["producer_sha256"]}
    audit = {"phase": 1775, "campaign": "C241", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "authorization": final["next_authorization"]}
    core.save(OUT / "audit/independent_final_audit.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
