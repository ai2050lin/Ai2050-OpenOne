#!/usr/bin/env python3
"""Independent audit for C236."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import phase1768_c234_event_campaign_common as common

core = common.core
OUT = common.OUTS["C236"]


def main() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    final = core.load(OUT / "analysis/final.json")
    effects = np.load(OUT / "raw/effects.float16.npy", mmap_mode="r")
    events = np.load(OUT / "raw/events.int8.npy", mmap_mode="r")
    first = np.load(OUT / "raw/first_formation.int8.npy", mmap_mode="r")
    persistence = np.load(OUT / "raw/persistence.uint8.npy", mmap_mode="r")
    checks = {
        "final": final["all_checks_passed"],
        "effects": effects.shape == (160, 3, 37, 128, 2560),
        "events": events.shape == effects.shape,
        "first": first.shape == (160, 3, 128, 2560),
        "persistence": persistence.shape == first.shape,
        "groups": len(core.rows(OUT / "protocol/effect_groups.jsonl")) == 160,
        "thresholds": len(core.load(OUT / "protocol/frozen_event_thresholds.json")["thresholds"]) == 37,
        "producer_hash": core.sha(Path(__file__).with_name("phase1770_c236_full_coordinate_interval_events.py")) == protocol["producer_sha256"],
    }
    audit = {"phase": 1770, "campaign": "C236", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "authorization": final["next_authorization"]}
    core.save(OUT / "audit/independent_final_audit.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
