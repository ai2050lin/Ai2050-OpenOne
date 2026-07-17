#!/usr/bin/env python3
"""Basic contract checks for the Phase439/441 protocol and manifest freeze."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tests" / "gpt5" / "phase439_natural_stable_entry_protocol.py"
MANIFEST_SCRIPT = ROOT / "tests" / "gpt5" / "phase441_task_split_manifest.py"
OUT_PATH = ROOT / "tests" / "gpt5" / "result" / "phase439_natural_stable_entry" / "phase439_protocol_freeze.json"
MANIFEST_PATH = ROOT / "tests" / "gpt5" / "result" / "phase439_natural_stable_entry" / "phase441_task_split_manifest.json"


def test_phase439_protocol_freeze_contract() -> None:
    subprocess.run([sys.executable, str(SCRIPT)], check=True, cwd=ROOT)
    data = json.loads(OUT_PATH.read_text(encoding="utf-8"))

    assert data["schema_version"] == "phase439_natural_stable_entry_protocol.v2"
    assert data["theory_name"] == "语言是动态模式网络"
    assert data["method_frame"] == "条件物理状态图谱"
    assert data["status"] == "protocol_v2_frozen_no_cuda_run"
    assert set(data["task_library"]) == {"knowledge_network", "single_step_reasoning", "syntax_system"}
    assert all(len(tasks) == 5 for tasks in data["task_library"].values())
    assert len(data["splits"]) == 6
    assert "physical_prediction_holdout" in data["splits"]
    assert data["sample_freeze_plan"]["base_semantic_groups_per_task"] == 384
    assert data["behavior_gates"]["semantic_lcb_95_min"] == 0.85
    assert data["behavior_gates"]["other_ucb_95_max"] == 0.05
    assert data["behavior_gates"]["surface_orbit_max_gap"] == 0.05
    assert data["behavior_gates"]["orbit_group_consistency_lcb_95_min"] == 0.80
    assert data["selection_rule"]["order"][0] == "pass_hard_behavior_gates"
    assert data["physical_scope"]["no_causal_intervention"] is True
    assert data["physical_scope"]["no_head_channel_neuron_scan"] is True
    assert data["orbit_metrics"]["requires_permutation_null"] is True
    assert data["equivariance_metric"]["requires_pre_registered_node_mapping"] is True
    assert "G_prediction" in data["sealed_authorization_gate"]


def test_phase441_manifest_freeze_contract() -> None:
    subprocess.run([sys.executable, str(MANIFEST_SCRIPT)], check=True, cwd=ROOT)
    data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    assert data["schema_version"] == "phase441_task_split_manifest.v1"
    assert data["status"] == "sample_split_manifest_frozen_no_cuda_run"
    assert data["protocol_schema_version"] == "phase439_natural_stable_entry_protocol.v2"
    assert data["groups_per_split"] == 64
    assert data["groups_per_task"] == 384
    assert data["total_sample_families"] == 3 * 5 * 6 * 64
    assert set(data["split_names"]) == {
        "interface_calibration",
        "task_discovery",
        "surface_orbit_holdout",
        "physical_window_freeze",
        "physical_prediction_holdout",
        "sealed_physical_holdout",
    }
    assert "manifest_sha256" in data

    leakage_keys = [entry["leakage_group_key"] for entry in data["entries"]]
    assert len(leakage_keys) == len(set(leakage_keys))
    assert all(entry["semantic_preservation_proof"] == "required_before_cuda" for entry in data["entries"])
    assert all(entry["node_mapping_status"] == "pre_registered_required" for entry in data["entries"])
