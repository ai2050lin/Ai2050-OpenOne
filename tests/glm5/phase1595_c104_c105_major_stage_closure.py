#!/usr/bin/env python3
"""Phase1595: integrate the C104 client, correct the C102 asset, and close C104-C105."""
from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
C102 = TESTS / "result/phase1581_c102_typed_relation_coordinate_campaign"
C104 = TESTS / "result/phase1589_c104_upstream_candidate_validation"
C105 = TESTS / "result/phase1593_c105_candidate_order_intervention_correction"
FRONTEND = ROOT / "frontend"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    c105 = core.load(C105 / "analysis/final.json")
    c105_audit = core.load(C105 / "audit/independent_final_audit.json")
    heatmap = core.load(C104 / "analysis/upstream_heatmap_export.json")
    heatmap_audit = core.load(C104 / "audit/independent_upstream_heatmap_export_audit.json")
    if not c105_audit["all_checks_passed"] or not heatmap_audit["all_checks_passed"]:
        raise RuntimeError("closure parents failed")

    c102_asset = C102 / "visualization/c102_coordinate_barcode_heatmap.json"
    c102_public = FRONTEND / "public/vis_data/research_kernel/c102_coordinate_barcode_heatmap.json"
    c102_payload = core.load(c102_asset)
    c102_payload["intervention_rows"] = core.rows(C105 / "analysis/c102_corrected_intervention_summary.jsonl")
    c102_payload["headline"]["controlled_intervention_passed"] = len(c105["c102"]["fully_controlled_families"])
    c102_payload["claim_boundary"] = "8/8 frozen barcodes replicated; after C105 corrected the [Yes, No] candidate-order readout, 5/8 families passed controlled interventions across confirmation and lockbox. This remains a conditional task-response mechanism, not a universal semantic code or sparse-neuron law."
    c102_payload["candidate_order_correction"] = {
        "phase": 1593, "campaign": "C105", "candidate_order": ["yes", "no"],
        "fully_controlled_families": c105["c102"]["fully_controlled_families"],
        "source_sha256": core.sha(C105 / "analysis/final.json"),
    }
    c102_payload["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    core.save(c102_asset, c102_payload)
    shutil.copyfile(c102_asset, c102_public)
    c102_update = {
        "status": "c102_heatmap_intervention_readout_corrected_by_c105",
        "asset_sha256": core.sha(c102_asset), "public_sha256": core.sha(c102_public),
        "controlled_intervention_passed": 5, "controlled_intervention_total": 8,
        "correction_source_sha256": core.sha(C105 / "analysis/final.json"),
    }
    core.save(C105 / "analysis/c102_heatmap_correction.json", c102_update)

    source_files = {
        "route": FRONTEND / "src/researchKernel/heatmapResearchRoute.js",
        "hook": FRONTEND / "src/researchKernel/useResearchKernel.js",
        "component": FRONTEND / "src/components/app/ResearchHeatmapRoute.jsx",
        "app": FRONTEND / "src/App.jsx",
    }
    source_text = {key: path.read_text(encoding="utf-8") for key, path in source_files.items()}
    dist_asset = FRONTEND / "dist/vis_data/research_kernel/c104_upstream_role_barcode_heatmap.json"
    public_asset = ROOT / heatmap["public"]
    checks = {
        "parents": c105_audit["all_checks_passed"] and heatmap_audit["all_checks_passed"],
        "route": "upstream_role_barcode_heatmap" in source_text["route"] and "C104_UPSTREAM_ROLE_BARCODE_HEATMAP_ROUTE" in source_text["route"],
        "hook": "c104UpstreamRoleBarcodeHeatmap" in source_text["hook"],
        "component": "buildC104UpstreamRoleBarcodeHeatmapData" in source_text["component"] and "C104 Upstream Role-State" in source_text["component"],
        "app": "c104UpstreamRoleBarcodeHeatmap" in source_text["app"],
        "asset": public_asset.exists() and core.sha(public_asset) == heatmap["sha256"],
        "dist": dist_asset.exists() and core.sha(dist_asset) == heatmap["sha256"],
        "c102_corrected": core.sha(c102_asset) == core.sha(c102_public) and c102_payload["headline"]["controlled_intervention_passed"] == 5,
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    final = {
        "phase": 1595,
        "campaign": "C104-C105",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "fresh_upstream_barcode_validation_causal_correction_heatmap_and_major_stage_complete",
        "checks": checks,
        "fresh_barcode": {"passed": 4, "total": 4},
        "c104_corrected_causal": {"fully_controlled_families": c105["c104"]["fully_controlled_families"], "partially_controlled": c105["c104"]["partially_controlled"]},
        "c102_corrected_causal": {"fully_controlled_families": c105["c102"]["fully_controlled_families"], "passed": 5, "total": 8},
        "behavior": core.load(C104 / "analysis/qwen_full_field_capture_summary.json")["behavior"],
        "heatmap": {"public": heatmap["public"], "bytes": heatmap["bytes"], "sha256": heatmap["sha256"], "result_type": "upstream_role_barcode_heatmap"},
        "new_puzzles": {
            "K276": "four frozen upstream full-coordinate truth-response barcodes prospectively replicated on lexically fresh confirmation and lockbox materials",
            "K277": "after exact candidate-order correction, whole-role-state transport is causally sufficient for attribute binding and agent-patient across both partitions and both code strata, but not uniformly for negation or whole-part exception",
            "K278": "predictive response equivalence and causal substitutability overlap only for a subset; cosine replication alone is insufficient for mechanism identity",
        },
        "claim_boundary": "single Qwen3 controlled tasks; whole role-state activation transport, not sparse coordinates, model weights, natural language universality or cross-model law",
        "next_authorization": "observation-first C106: discover minimal stable coordinate coalitions inside the two replicated causal families, with nested ablations and fresh held-out units; no attention or MLP analysis",
    }
    core.save(C104 / "analysis/final.json", final)
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
