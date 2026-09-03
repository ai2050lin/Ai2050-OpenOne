#!/usr/bin/env python3
"""Independent audit for C244."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import phase1768_c234_event_campaign_common as common

core = common.core
OUT = common.RESULT / "phase1778_c244_independent_event_replication"


def main() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    final = core.load(OUT / "analysis/final.json")
    rows = core.rows(OUT / "compiled/qwen3.jsonl")
    fields = np.load(OUT / "raw/full_fields.float16.npy", mmap_mode="r")
    checks = {
        "internal": final["all_checks_passed"],
        "rows": len(rows) == 240,
        "new_surface": not ({row["surface"] for row in rows} & set(common.SURFACES)),
        "full_field": fields.shape == (240, 37, 128, 2560),
        "behavior_gate_typed": isinstance(final["headline"]["behavior"]["behavior_eligible"], bool),
        "event_gate_typed": "event_rules_tested" in final["headline"]["event_replication"],
        "five_controls": protocol["controls"] == ["best_wrong_family", "discovery_generic", "relation_only", "nearest_length_discovery_group", "zero"],
        "five_diagnostics": len(final["headline"]["cross_model_scaffold_diagnostic"]["transforms"]) == 5,
        "producer_hash": core.sha(Path(__file__).with_name("phase1778_c244_independent_event_replication.py")) == protocol["producer_sha256"],
    }
    audit = {"phase": 1778, "campaign": "C244", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "authorization": final["next_authorization"]}
    core.save(OUT / "audit/independent_final_audit.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
