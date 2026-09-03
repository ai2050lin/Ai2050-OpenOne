#!/usr/bin/env python3
"""Independent audit for C183 synthesis and public parameter-level heatmap."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1717_c183_response_ecology_synthesis_heatmap"
PUBLIC = ROOT / "frontend/public/vis_data/research_kernel/c183_natural_response_ecology_heatmap.json"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core


def main() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    final = core.load(OUT / "analysis/final.json")
    asset = core.load(OUT / "analysis/public_asset.json")
    payload = core.load(PUBLIC)
    state_rows = [row for row in payload["rows"] if row["kind"] == "anchor_state"]
    response_rows = [row for row in payload["rows"] if row["kind"] == "local_response"]
    producer = Path(__file__).with_name("phase1717_c183_response_ecology_synthesis_heatmap.py")
    checks = {
        "closed": final["status"] == "closed" and final["all_checks_passed"],
        "schema": payload["schema"] == "c183_natural_response_ecology_heatmap.v1",
        "all_2560": len(payload["dimensions"]) == 2560 and all(len(row["values"]) == 2560 for row in payload["rows"]),
        "embedding_hidden": any(row["checkpoint"] == 0 for row in state_rows) and any(row["checkpoint"] == 37 for row in state_rows),
        "fresh_response": any(row["partition"] == "fresh" for row in response_rows),
        "asset_hash": core.sha(PUBLIC) == asset["sha256"],
        "producer_hash": core.sha(producer) == protocol["producer_sha256"],
        "no_attention_mlp": all(term not in json.dumps(payload).lower() for term in ("attention weight", "mlp activation")),
    }
    result = {
        "phase": 1717,
        "campaign": "C183",
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "authorization": final["next_authorization"],
    }
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
