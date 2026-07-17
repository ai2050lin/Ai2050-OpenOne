#!/usr/bin/env python3
"""Contract checks for Phase442 static sample and feasibility artifacts."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tests" / "gpt5" / "phase442_static_sample_contract.py"
TOKEN_SCRIPT = ROOT / "tests" / "gpt5" / "phase442_tokenization_contract.py"
OUT_DIR = ROOT / "tests" / "gpt5" / "result" / "phase442_static_sample_contract"
PROTOCOL_PATH = OUT_DIR / "phase442_protocol_v3_freeze.json"
SAMPLES_PATH = OUT_DIR / "phase442_samples.jsonl"
CERTS_PATH = OUT_DIR / "phase442_semantic_certificates.jsonl"
AUDIT_PATH = OUT_DIR / "phase442_static_audit_report.json"
MANIFEST_PATH = OUT_DIR / "phase442_artifact_manifest.json"
TOKEN_PATH = OUT_DIR / "phase442_tokenization_report.json"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_phase442_static_sample_contract() -> None:
    subprocess.run([sys.executable, str(SCRIPT)], check=True, cwd=ROOT)
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    samples = load_jsonl(SAMPLES_PATH)
    certs = load_jsonl(CERTS_PATH)

    assert protocol["schema_version"] == "phase442_static_sample_contract.v3"
    assert protocol["status"] == "static_samples_and_feasibility_frozen_no_cuda_run"
    assert protocol["confidence_interval"]["kind"] == "wilson"
    assert protocol["confidence_interval"]["sidedness"] == "two_sided"
    assert protocol["groups_per_split"] >= 73
    assert protocol["groups_per_split"] == 80
    assert protocol["groups_per_task"] == 480
    assert protocol["behavior_gates"]["other_max_failures_per_split"] == 0

    assert audit["status"] == "static_contract_pass_no_cuda_run"
    assert audit["feasibility"]["semantic_gate"]["feasible"] is True
    assert audit["feasibility"]["other_output_gate"]["feasible"] is True
    assert audit["feasibility"]["other_output_gate"]["ucb_at_zero"] <= 0.05
    assert audit["feasibility"]["orbit_group_gate"]["feasible"] is True
    assert audit["split_disjoint"]["entity_role_string_disjoint"] is True
    assert audit["semantic_certificates"]["semantic_hash_preserved"] is True
    assert audit["baseline"]["majority_baseline_below_semantic_gate"] is True
    assert audit["budget"]["requires_staged_authorization"] is True

    assert len(samples) == 3 * 5 * 6 * 80
    assert len(certs) == len(samples)
    assert all(sample["input_text"] for sample in samples)
    assert all(sample["canonical_answer"] for sample in samples)
    assert all(len(sample["surface_variants"]) == 6 for sample in samples)
    assert all(variant["semantic_hash"] == sample["semantic_hash"] for sample in samples for variant in sample["surface_variants"])

    assert manifest["status"] == "frozen_no_cuda_run"
    assert "joint_sha256" in manifest
    assert str(SAMPLES_PATH.relative_to(ROOT)) in manifest["artifacts"]


def test_phase442_tokenization_contract() -> None:
    subprocess.run([sys.executable, str(SCRIPT)], check=True, cwd=ROOT)
    subprocess.run([sys.executable, str(TOKEN_SCRIPT)], check=True, cwd=ROOT)
    data = json.loads(TOKEN_PATH.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    assert data["schema_version"] == "phase442_tokenization_report.v1"
    assert data["status"] == "pass"
    assert data["cuda_used"] is False
    assert data["model_weights_loaded"] is False
    assert {report["model"] for report in data["reports"]} == {"qwen3", "glm4", "deepseek7b"}
    assert all(report["samples_checked"] == 7200 for report in data["reports"])
    assert all(report["empty_alias_failure_count"] == 0 for report in data["reports"])
    assert all(report["prompt_over_limit_count"] == 0 for report in data["reports"])
    assert str(TOKEN_PATH.relative_to(ROOT)) in manifest["artifacts"]
