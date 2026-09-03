#!/usr/bin/env python3
"""Audit C245 event-core heatmap and client integration."""
from __future__ import annotations

import json

import phase1768_c234_event_campaign_common as common

core = common.core
OUT = common.RESULT / "phase1779_c245_confirmed_event_core"
ASSET = common.ROOT / "frontend/public/vis_data/research_kernel/c245_confirmed_event_core.json"


def main() -> None:
    manifest = core.load(OUT / "analysis/heatmap_manifest.json")
    with ASSET.open("r", encoding="utf-8") as handle: payload = json.load(handle)
    sources = "\n".join((common.ROOT / path).read_text(encoding="utf-8") for path in ("frontend/src/researchKernel/heatmapResearchRoute.js", "frontend/src/researchKernel/useResearchKernel.js", "frontend/src/components/app/ResearchHeatmapRoute.jsx", "frontend/src/App.jsx"))
    checks = {"producer": manifest["all_checks_passed"], "hash": core.sha(ASSET) == manifest["sha256"], "schema": payload["schema"] == "c245_confirmed_event_core.v1", "rows": len(payload["rows"]) == 150, "dimensions": payload["dimensions"] == list(range(2560)), "client": "C245_CONFIRMED_EVENT_CORE_ROUTE" in sources and "c245ConfirmedEventCore" in sources}
    audit = {"phase": 1779, "campaign": "C245", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    core.save(OUT / "audit/heatmap_client_audit.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__": main()
