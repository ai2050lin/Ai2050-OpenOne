#!/usr/bin/env python3
"""Independent audit for C245."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import phase1768_c234_event_campaign_common as common

core = common.core
OUT = common.RESULT / "phase1779_c245_confirmed_event_core"


def main() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    final = core.load(OUT / "analysis/final.json")
    confirmed = np.load(OUT / "analysis/confirmed_rule_codes.int8.npy", mmap_mode="r")
    events = np.load(OUT / "raw/c244_role_events.int8.npy", mmap_mode="r")
    checks = {
        "internal": final["all_checks_passed"],
        "confirmed_shape": confirmed.shape == (5, 3, 37, 6, 2560),
        "event_shape": events.shape == (60, 3, 37, 6, 2560),
        "counts": int(np.count_nonzero(confirmed)) == final["headline"]["confirmed_events"],
        "subset": final["headline"]["confirmed_events"] <= final["headline"]["old_rule_events"],
        "all_coordinates": protocol["axes"][-1] == "all_2560_physical_coordinates",
        "producer_hash": core.sha(Path(__file__).with_name("phase1779_c245_confirmed_event_core.py")) == protocol["producer_sha256"],
    }
    audit = {"phase": 1779, "campaign": "C245", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "authorization": final["next_authorization"]}
    core.save(OUT / "audit/independent_final_audit.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
