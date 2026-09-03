#!/usr/bin/env python3
"""Independent source and asset audit for the C243 heatmap integration."""
from __future__ import annotations

import json

import phase1768_c234_event_campaign_common as common

core = common.core
OUT = common.OUTS["C243"]
COMPACT = common.ROOT / "frontend/public/vis_data/research_kernel/c243_conditional_event_atlas_compact.json"


def main() -> None:
    report = core.load(OUT / "analysis/visualization_client_integration.json")
    with COMPACT.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    route = (common.ROOT / "frontend/src/researchKernel/heatmapResearchRoute.js").read_text(encoding="utf-8")
    hook = (common.ROOT / "frontend/src/researchKernel/useResearchKernel.js").read_text(encoding="utf-8")
    component = (common.ROOT / "frontend/src/components/app/ResearchHeatmapRoute.jsx").read_text(encoding="utf-8")
    app = (common.ROOT / "frontend/src/App.jsx").read_text(encoding="utf-8")
    checks = {
        "producer": report["all_checks_passed"],
        "asset_hash": core.sha(COMPACT) == report["compact_sha256"],
        "compact_schema": payload["schema"] == "c243_conditional_event_atlas_compact.v1",
        "archive_declared": payload["total_rows"] == 3330 and payload["archive_path"].endswith("c243_conditional_event_atlas.json"),
        "physical_columns": payload["dimensions"] == list(range(2560)),
        "route": "c243_conditional_event_atlas_compact.json" in route,
        "hook": "setC243ConditionalEventAtlas" in hook,
        "component": "C243_CONDITIONAL_EVENT_ATLAS_ROUTE.title" in component and "stable discovery events" in component,
        "app": "c243ConditionalEventAtlas={realResearchTrace.c243ConditionalEventAtlas}" in app,
    }
    audit = {
        "phase": 1777,
        "campaign": "C243",
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "all_checks_passed": all(checks.values()),
    }
    core.save(OUT / "audit/visualization_client_integration_audit.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
