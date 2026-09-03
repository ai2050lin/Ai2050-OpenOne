#!/usr/bin/env python3
"""Normalize the frozen C272 atlas for the generic all-coordinate client route."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ASSET = ROOT / "frontend/public/vis_data/research_kernel/c272_state_conditioned_operator_atlas.json"
OUT = ROOT / "tests/glm5/result/phase1806_c272_state_conditioned_operator_closure/visualization"


def main() -> None:
    payload = json.loads(ASSET.read_text(encoding="utf-8"))
    before_schema = {
        "dimensions": payload.get("dimensions"),
        "default_coordinates": payload.get("default_coordinates"),
    }
    dimension_count = int(payload["dimensions"])
    default_count = int(payload["default_coordinates"])
    payload["dimensions"] = list(range(dimension_count))
    payload["default_coordinates"] = list(range(default_count))
    payload["total_rows"] = len(payload["rows"])
    ASSET.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    checks = {
        "schema": payload["schema"] == "c272_state_conditioned_operator_atlas.v1",
        "all_2560_coordinate_ids": payload["dimensions"] == list(range(2560)),
        "default_coordinate_ids": payload["default_coordinates"] == list(range(64)),
        "all_rows_full_coordinate": all(len(row["values"]) == 2560 for row in payload["rows"]),
        "row_count": payload["total_rows"] == 570,
    }
    audit = {
        "phase": 1806,
        "campaign": "C272",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "kind": "client_schema_normalization",
        "before_schema": before_schema,
        "after_schema": {
            "dimensions": "integer coordinate id array [0,2559]",
            "default_coordinates": "integer coordinate id array [0,63]",
            "total_rows": payload["total_rows"],
        },
        "claim_or_measurement_changed": False,
        "sha256": hashlib.sha256(ASSET.read_bytes()).hexdigest(),
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "client_integration_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
