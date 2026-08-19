"""Audit the client SNN dashboard against frozen SpikeICSPB evidence."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = ROOT / "tests/codex_temp"


def load_json(name: str) -> dict:
    return json.loads((RESULT_ROOT / name).read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "tests/glm5/result/phase1237_snn_research_dashboard/phase1237_summary.json",
    )
    args = parser.parse_args()

    homeostasis = load_json(
        "spike_icspb_3d_homeostatic_control_law_block_20260315.json"
    )
    inventory = load_json(
        "spike_icspb_3d_feature_inventory_measurement_block_20260315.json"
    )
    successor = load_json(
        "spike_icspb_3d_successor_quality_audit_block_20260315.json"
    )
    retention = load_json(
        "spike_icspb_3d_retention_instant_learning_benchmark_block_20260315.json"
    )
    scaling = load_json("spike_icspb_3d_scaling_readiness_block_20260315.json")

    app_source = (ROOT / "frontend/src/App.jsx").read_text(encoding="utf-8")
    state_source = (
        ROOT / "frontend/src/researchKernel/snnResearchState.js"
    ).read_text(encoding="utf-8")
    dashboard_source = (
        ROOT / "frontend/src/components/app/SNNResearchDashboard.jsx"
    ).read_text(encoding="utf-8")
    dashboard_css = (
        ROOT / "frontend/src/components/app/SNNResearchDashboard.css"
    ).read_text(encoding="utf-8")
    runtime_source = (
        ROOT / "frontend/src/researchKernel/snnRuntime.js"
    ).read_text(encoding="utf-8")
    server_source = (ROOT / "server/server.py").read_text(encoding="utf-8")

    checks = {
        "dashboard_mounted_in_snn_workspace": (
            "<SNNResearchDashboard runtimeState={snnState}" in app_source
        ),
        "latest_architecture_named": "SpikeICSPB-3D-MR-PhaseA" in state_source,
        "architecture_pipeline_visible": all(
            name in state_source
            for name in (
                "EventPatchSelector",
                "BurstSectionBinder",
                "PhaseGatedSuccessorCore",
                "PopulationReadout",
            )
        ),
        "scheme_and_effect_sections_visible": (
            "当前方案" in dashboard_source and "模型效果" in dashboard_source
        ),
        "scientific_boundary_visible": (
            "尚未证明强语言能力" in state_source
            and "不等于上方 SpikeICSPB 模型" in dashboard_source
        ),
        "homeostasis_values_match_artifact": (
            homeostasis["active_homeostasis"]["pre_loss"] >
            homeostasis["active_homeostasis"]["post_loss"]
            and "5.6906 → 4.9753" in state_source
            and "优势 0.2263" in state_source
        ),
        "inventory_values_match_artifact": (
            inventory["headline_metrics"]["measured_inventory_size"] == 36
            and "36 / 分离比 0.9252" in state_source
        ),
        "successor_failure_visible": (
            successor["headline_metrics"]["successor_quality_score"] < 0.2
            and successor["headline_metrics"]["next_token_margin"] < 0
            and "0.1952" in state_source
            and "−0.0032" in state_source
        ),
        "retention_non_advantage_visible": (
            retention["strict_verdict"]["controls_already_outperform_fixed"] is False
            and "+0.0004 / −0.0001" in state_source
        ),
        "scale_is_labeled_estimate_not_training": (
            scaling["headline_metrics"]["phase_a_parameter_count_m"] > 70
            and state_source.count("仅结构估算") == 3
            and "规模化路线（尚未实训）" in dashboard_source
        ),
        "live_lif_runtime_retained": (
            "<StructureAnalysisControls" in app_source
            and "实时 3D LIF 演示" in dashboard_source
        ),
        "snn_step_returns_spikes": (
            '"spikes": active_spikes' in server_source
            and "active_spikes.setdefault" in server_source
        ),
        "legacy_backend_history_is_supported": (
            "collectSnnSpikes(res.data)" in app_source
            and "payload?.history" in runtime_source
        ),
        "dashboard_styles_present": ".snn-research__effects" in dashboard_css,
    }
    passed = all(checks.values())
    result = {
        "schema_version": "1.0.0",
        "phase_id": "Phase1237",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "passed": passed,
        "checks": checks,
        "research_track": "3D SpikeICSPB Multi-Region Phase-A",
        "evidence_frozen_at": "2026-03-15",
        "current_project_phase": 1236,
        "artifact_model_reruns": 0,
        "live_lif_smoke_steps": 5,
        "gpu_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
