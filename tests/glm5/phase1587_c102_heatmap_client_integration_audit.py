#!/usr/bin/env python3
"""Phase1587: verify the built C102 coordinate-barcode heatmap client."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1581_c102_typed_relation_coordinate_campaign"
ASSET = OUT / "visualization/c102_coordinate_barcode_heatmap.json"
PUBLIC = ROOT / "frontend/public/vis_data/research_kernel/c102_coordinate_barcode_heatmap.json"
BUILT = ROOT / "frontend/dist/vis_data/research_kernel/c102_coordinate_barcode_heatmap.json"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    export_audit = core.load(OUT / "audit/independent_heatmap_export_audit.json")
    payload = core.load(ASSET)
    route = (ROOT / "frontend/src/researchKernel/heatmapResearchRoute.js").read_text(encoding="utf-8")
    hook = (ROOT / "frontend/src/researchKernel/useResearchKernel.js").read_text(encoding="utf-8")
    card = (ROOT / "frontend/src/components/app/ResearchHeatmapRoute.jsx").read_text(encoding="utf-8")
    app = (ROOT / "frontend/src/App.jsx").read_text(encoding="utf-8")
    hashes = {"canonical": core.sha(ASSET), "public": core.sha(PUBLIC), "built": core.sha(BUILT)}
    checks = {
        "export": export_audit["all_checks_passed"],
        "schema": payload["schema"] == "c102_coordinate_barcode_heatmap.v1",
        "full_coordinates": payload["dimensions"] == list(range(2560)),
        "asset_identity": len(set(hashes.values())) == 1,
        "route": "C102_COORDINATE_BARCODE_HEATMAP_ROUTE" in route and "coordinate_barcode_heatmap" in route,
        "hook": "c102CoordinateBarcodeHeatmap" in hook,
        "card": "buildC102CoordinateBarcodeHeatmapData" in card and "activation coordinate" in card,
        "app": "c102CoordinateBarcodeHeatmap={realResearchTrace.c102CoordinateBarcodeHeatmap}" in app,
        "embedding_hidden": {row["state_kind"] for row in payload["raw_rows"]} == {"embedding", "hidden_state"},
        "claim_scope": "not weight parameters" in payload["coordinate_semantics"] and payload["headline"]["controlled_intervention_passed"] == 0,
    }
    result = {"phase": 1587, "campaign": "C102", "status": "built_heatmap_client_verified", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "hashes": hashes, "built_asset_bytes": BUILT.stat().st_size, "authorization": "append_c102_complete_phase_memo"}
    if not result["all_checks_passed"]:
        raise RuntimeError(result)
    core.save(OUT / "analysis/client_integration.json", result)
    core.save(OUT / "audit/independent_client_integration_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
