#!/usr/bin/env python3
"""Publish the Phase364-A algorithm audit without raw tensors or private cases."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase364_projection_sufficiency_audit/offline_projection_audit"
TARGETS = (
    ROOT / "tests/gpt5/result/pattern_family_atlas/v2",
    ROOT / "frontend/public/vis_data/pattern_family_atlas/v2",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main() -> None:
    summary = read_json(OUT / "phase364_projection_audit_summary.json")
    protocol = read_json(OUT / "phase365_instrumentation_protocol.json")
    updated_at = now()
    for target in TARGETS:
        write_json(target / "phase364_projection_audit_summary.json", summary)
        write_json(target / "phase365_instrumentation_protocol.json", protocol)
        manifest_path = target / "manifest.json"
        manifest = read_json(manifest_path)
        manifest["updated_at"] = updated_at
        manifest["phase364"] = {
            "status": "p0_lossy_skeleton_p2_p3_instrumentation_incomplete",
            "discovery_case_count": summary["denominator"]["discovery_case_count"],
            "anchor_count": summary["denominator"]["anchor_count"],
            "p0_structurally_noninjective": summary["results"]["p0_structurally_noninjective"],
            "strict_formula_survivor_count": summary["results"]["strict_formula_survivor_count"],
            "mlp_single_neuron_self_contained_model_count": summary["results"]["mlp_single_neuron_self_contained_model_count"],
            "dynamic_flow_bundle_schema_model_count": summary["results"]["dynamic_flow_bundle_schema_model_count"],
            "new_model_execution_authorized": protocol["new_model_execution_authorized"],
            "physical_confirmation_opened": False,
            "raw_tensors_frontend_exported": False,
            "files": ["phase364_projection_audit_summary.json", "phase365_instrumentation_protocol.json"],
        }
        write_json(manifest_path, manifest)
        progress_path = target / "progress.json"
        progress = read_json(progress_path)
        progress["last_phase"] = "Phase364-A"
        progress["updated_at"] = updated_at
        progress["phase364_decision"] = summary["decision"]
        progress["single_global_progress_percentage_valid"] = False
        write_json(progress_path, progress)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
