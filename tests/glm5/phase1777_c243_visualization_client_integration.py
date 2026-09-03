#!/usr/bin/env python3
"""Create a browser-sized C243 view without dropping any physical coordinate columns."""
from __future__ import annotations

import json
from pathlib import Path

import phase1768_c234_event_campaign_common as common

core = common.core
OUT = common.OUTS["C243"]
FULL = common.ROOT / "frontend/public/vis_data/research_kernel/c243_conditional_event_atlas.json"
COMPACT = common.ROOT / "frontend/public/vis_data/research_kernel/c243_conditional_event_atlas_compact.json"


def main() -> None:
    with FULL.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = [row for row in payload["rows"] if row["source"] == "C243_core"]
    compact = {key: value for key, value in payload.items() if key != "rows"}
    compact.update({
        "schema": "c243_conditional_event_atlas_compact.v1",
        "total_rows": len(payload["rows"]),
        "archive_path": str(FULL.relative_to(common.ROOT)).replace("\\", "/"),
        "view_contract": "Five checkpoints x relation/boundary roles; every row retains all 2560 physical coordinates.",
        "rows": rows,
    })
    with COMPACT.open("w", encoding="utf-8") as handle:
        json.dump(compact, handle, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
    checks = {
        "rows": len(rows) == 150,
        "all_coordinates": all(len(row["values"]) == 2560 for row in rows),
        "checkpoints": {row["checkpoint"] for row in rows} == {0, 8, 16, 24, 36},
        "roles": {row["role"] for row in rows} == {"relation", "boundary"},
        "full_archive_unchanged": core.sha(FULL) == core.load(OUT / "analysis/heatmap_manifest.json")["sha256"],
    }
    result = {
        "phase": 1777,
        "campaign": "C243",
        "status": "visualization_client_asset_ready",
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "compact_asset": str(COMPACT.relative_to(common.ROOT)).replace("\\", "/"),
        "compact_bytes": COMPACT.stat().st_size,
        "compact_sha256": core.sha(COMPACT),
    }
    core.save(OUT / "analysis/visualization_client_integration.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
