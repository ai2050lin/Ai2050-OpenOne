#!/usr/bin/env python3
"""Independent audit for C237."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import phase1768_c234_event_campaign_common as common

core = common.core
OUT = common.OUTS["C237"]


def main() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    final = core.load(OUT / "analysis/final.json")
    effects = np.load(OUT / "raw/role_effects.float16.npy", mmap_mode="r")
    events = np.load(OUT / "raw/role_events.int8.npy", mmap_mode="r")
    rules = np.load(OUT / "analysis/rule_codes.int8.npy", mmap_mode="r")
    amendment = core.load(OUT / "audit/provenance_amendment.json")
    checks = {
        "final": final["all_checks_passed"],
        "role_effects": effects.shape == (160, 3, 37, 6, 2560),
        "role_events": events.shape == effects.shape,
        "rules": rules.shape == (5, 3, 37, 6, 2560),
        "fit_partition": protocol["fit_partition"] == "discovery only",
        "readable": len(core.rows(OUT / "analysis/readable_rules.jsonl")) == 90,
        "precedence": len(core.rows(OUT / "analysis/precedence_rules.jsonl")) == 450,
        "transparent_amendment": amendment["scientific_fields_changed"] is False and amendment["thresholds_changed"] is False,
        "producer_hash": core.sha(Path(__file__).with_name("phase1771_c237_conditional_event_rule_discovery.py")) == protocol["producer_sha256"],
    }
    audit = {"phase": 1771, "campaign": "C237", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "authorization": final["next_authorization"]}
    core.save(OUT / "audit/independent_final_audit.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
