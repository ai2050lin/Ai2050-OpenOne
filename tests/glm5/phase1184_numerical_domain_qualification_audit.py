#!/usr/bin/env python3
"""Independent audit for Phase1184 numerical-domain qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))

import phase1184_numerical_domain_qualification as phase  # noqa: E402


SCRIPT = Path(__file__).resolve()
AUDIT_PATH = phase.OUT_ROOT / "audit/independent_audit.json"


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def close(a: float, b: float, atol: float = 1e-12, rtol: float = 1e-9) -> bool:
    return bool(abs(a - b) <= atol + rtol * max(abs(a), abs(b)))


def rms(array: np.ndarray) -> float:
    values = np.asarray(array, dtype=np.float64)
    return float(math.sqrt(max(float(np.mean(values * values)), 0.0)))


def local_forward(reference: np.ndarray, changed: np.ndarray, unit: float, precision: str) -> dict[str, float]:
    delta = np.asarray(changed, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    scale_max = max(float(np.max(np.abs(reference))), 1e-30)
    scale_rms = max(rms(reference), 1e-30)
    multiplier = 128.0 if precision == "fp64" else 32.0
    absolute_floor = (256.0 if precision == "fp64" else 128.0) * unit
    absolute_max = float(np.max(np.abs(delta)))
    return {
        "absolute_max": absolute_max,
        "reference_max_abs": scale_max,
        "scaled_max": absolute_max / scale_max,
        "rms_relative": rms(delta) / scale_rms,
        "mixed_absolute_bound": absolute_floor + multiplier * unit * scale_max,
    }


@torch.inference_mode()
def direct_fp32(model, x: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    ids = x.to(device)
    left = F.embedding(ids[:, 0], model.left_embedding.weight.float())
    right = F.embedding(ids[:, 1], model.right_embedding.weight.float())
    hidden = F.linear(left + right, model.hidden.weight.float())
    logits = F.linear(hidden.square(), model.output.weight.float())
    return logits, hidden


def direct_gauge(model, seed: int, device: torch.device, broken: bool = False):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    permutation = torch.randperm(model.config.width, generator=generator)
    signs = torch.where(
        torch.rand(model.config.width, generator=generator) < 0.5,
        torch.tensor(-1.0),
        torch.tensor(1.0),
    )
    transformed = phase.p1171.RoleSquareNetwork(model.config).to(device)
    with torch.no_grad():
        transformed.left_embedding.weight.copy_(model.left_embedding.weight)
        transformed.right_embedding.weight.copy_(model.right_embedding.weight)
        transformed.hidden.weight.copy_(
            signs[:, None].to(device) * model.hidden.weight[permutation.to(device)]
        )
        if broken:
            transformed.output.weight.copy_(model.output.weight)
        else:
            transformed.output.weight.copy_(model.output.weight[:, permutation.to(device)])
    transformed.eval()
    return transformed


def append_check(checks: list[dict[str, Any]], name: str, passed: bool, detail: Any = None) -> None:
    checks.append({"name": name, "pass": bool(passed), "detail": detail})


def audit() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for independent replay")
    protocol = read_json(phase.PROTOCOL_PATH)
    domain = read_json(phase.DOMAIN_PATH)
    final = read_json(phase.FINAL_PATH)
    discovery_rows_path = phase.OUT_ROOT / "runs/discovery/systems.jsonl"
    confirmation_rows_path = phase.OUT_ROOT / "runs/confirmation/systems.jsonl"
    gauge_rows_path = phase.OUT_ROOT / "analysis/gauge_rows.jsonl"
    positive_rows_path = phase.OUT_ROOT / "analysis/positive_control_rows.jsonl"
    discovery_rows = read_jsonl(discovery_rows_path)
    confirmation_rows = read_jsonl(confirmation_rows_path)
    gauge_rows = read_jsonl(gauge_rows_path)
    positive_rows = read_jsonl(positive_rows_path)
    checks: list[dict[str, Any]] = []

    protocol_copy = dict(protocol)
    protocol_digest = protocol_copy.pop("protocol_digest")
    append_check(checks, "protocol_digest", digest(protocol_copy) == protocol_digest)
    append_check(checks, "protocol_phase", protocol["phase"] == phase.PHASE)
    append_check(checks, "runner_hash", file_sha256(phase.SCRIPT) == protocol["scripts"]["runner"])
    append_check(checks, "audit_hash", file_sha256(SCRIPT) == protocol["scripts"]["audit"])
    append_check(checks, "phase1171_hash", file_sha256(Path(phase.p1171.__file__)) == protocol["scripts"]["phase1171_source"])
    append_check(checks, "phase1181_hash", file_sha256(Path(phase.p1181.__file__)) == protocol["scripts"]["phase1181_source"])
    append_check(checks, "phase1183_hash", file_sha256(Path(phase.p1183.__file__)) == protocol["scripts"]["phase1183_source"])
    phase1183_stop = phase.p1183.OUT_ROOT / "analysis/final_stop.json"
    append_check(checks, "phase1183_stop_hash", file_sha256(phase1183_stop) == protocol["scripts"]["phase1183_stop"])
    append_check(checks, "task_count", len(protocol["tasks"]) == 8)
    append_check(checks, "task_table_uniqueness", len({row["signature"]["table_digest"] for row in protocol["tasks"]}) == 8)
    append_check(checks, "split_balance", sum(row["split"] == "discovery" for row in protocol["tasks"]) == 4 and sum(row["split"] == "confirmation" for row in protocol["tasks"]) == 4)

    discovery_seal = read_json(phase.OUT_ROOT / "runs/discovery/training_seal.json")
    ds_copy = dict(discovery_seal)
    ds_digest = ds_copy.pop("seal_digest")
    append_check(checks, "discovery_training_seal_digest", digest(ds_copy) == ds_digest)
    append_check(checks, "discovery_training_count", discovery_seal["system_count"] == 32)
    append_check(checks, "discovery_holdout_unread_at_seal", discovery_seal["all_holdout_labels_unread"] is True)
    append_check(checks, "discovery_rows_hash", file_sha256(discovery_rows_path) == domain["rows_sha256"])
    append_check(checks, "discovery_rows_count", len(discovery_rows) == 32)
    qualified_discovery = [row for row in discovery_rows if row["qualified"]]
    append_check(checks, "discovery_qualified_count", len(qualified_discovery) == domain["qualified_system_count"])
    recomputed_passing_discovery = sum(
        sum(row["qualified"] for row in discovery_rows if row["task_name"] == task.name)
        >= phase.THRESHOLDS["qualified_system_count_per_task_min"]
        for task in phase.split_tasks("discovery")
    )
    append_check(checks, "discovery_passing_task_count", recomputed_passing_discovery == domain["passing_task_count"])
    append_check(checks, "discovery_behavior_pass", domain["behavior_pass"] is True)
    domain_copy = dict(domain)
    domain_digest = domain_copy.pop("domain_seal_digest")
    append_check(checks, "domain_seal_digest", digest(domain_copy) == domain_digest)
    bounds_exact = True
    for metric in phase.PROFILE_METRICS:
        values = [row["profile"][metric] for row in qualified_discovery]
        expected = {
            "natural_min": min(values),
            "natural_max": max(values),
            "safety_min": min(values) / phase.SAFETY_MULTIPLIER,
            "safety_max": max(values) * phase.SAFETY_MULTIPLIER,
        }
        bounds_exact = bounds_exact and all(close(domain["bounds"][metric][key], value) for key, value in expected.items())
    append_check(checks, "domain_bounds_exact_recompute", bounds_exact)
    append_check(checks, "domain_seal_pass", domain["domain_seal_pass"] is True)

    confirmation_seal = read_json(phase.OUT_ROOT / "runs/confirmation/training_seal.json")
    cs_copy = dict(confirmation_seal)
    cs_digest = cs_copy.pop("seal_digest")
    append_check(checks, "confirmation_training_seal_digest", digest(cs_copy) == cs_digest)
    append_check(checks, "confirmation_training_count", confirmation_seal["system_count"] == 32)
    append_check(checks, "confirmation_holdout_unread_at_seal", confirmation_seal["all_holdout_labels_unread"] is True)
    append_check(checks, "confirmation_rows_count", len(confirmation_rows) == 32)
    qualified_confirmation = [row for row in confirmation_rows if row["qualified"]]
    append_check(checks, "confirmation_qualified_count", len(qualified_confirmation) == final["confirmation"]["qualified_system_count"])
    recomputed_safety = []
    for row in qualified_confirmation:
        inside = all(
            domain["bounds"][metric]["safety_min"] <= row["profile"][metric] <= domain["bounds"][metric]["safety_max"]
            for metric in phase.PROFILE_METRICS
        )
        recomputed_safety.append(inside)
    safety_fraction = sum(recomputed_safety) / max(len(recomputed_safety), 1)
    append_check(checks, "confirmation_safety_fraction", close(safety_fraction, final["confirmation"]["safety_domain_fraction"]))
    append_check(checks, "confirmation_domain_pass", final["confirmation"]["domain_coverage_pass"] is True)
    append_check(checks, "gauge_rows_hash", file_sha256(gauge_rows_path) == final["gauge"]["rows_sha256"])
    append_check(checks, "gauge_rows_count", len(gauge_rows) == len(qualified_confirmation) * phase.GAUGE_TRANSFORMS)
    gauge_fraction = sum(row["gauge_pass"] for row in gauge_rows) / max(len(gauge_rows), 1)
    append_check(checks, "gauge_fraction", close(gauge_fraction, final["gauge"]["valid_fraction"]))
    append_check(checks, "gauge_pass", final["gauge"]["gauge_pass"] is True)
    append_check(checks, "positive_rows_hash", file_sha256(positive_rows_path) == final["positive_control"]["rows_sha256"])
    append_check(checks, "positive_rows_count", len(positive_rows) == len(qualified_confirmation))
    positive_fraction = sum(row["positive_control_pass"] for row in positive_rows) / max(len(positive_rows), 1)
    append_check(checks, "positive_fraction", close(positive_fraction, final["positive_control"]["valid_fraction"]))
    append_check(checks, "positive_control_pass", final["positive_control"]["positive_control_pass"] is True)

    device = torch.device("cuda")
    stored_first = {(row["checkpoint"], row["transform"]): row for row in gauge_rows}
    stored_positive = {row["checkpoint"]: row for row in positive_rows}
    gauge_replay_ok = True
    positive_replay_ok = True
    maximum_gauge_metric_delta = 0.0
    for row in qualified_confirmation:
        path = phase.OUT_ROOT / "runs/confirmation/checkpoints" / row["checkpoint"]
        payload = torch.load(path, map_location="cpu", weights_only=False)
        model = phase.load_model(payload, device)
        data = phase.make_data(payload["task_name"], payload["seed"] + 17)
        seed = 11848000 + payload["task_index"] * 10_000 + payload["replicate"] * 100
        transformed = direct_gauge(model, seed, device)
        original, _ = direct_fp32(model, data["all_x"], device)
        changed, _ = direct_fp32(transformed, data["all_x"], device)
        metrics = local_forward(
            original.detach().cpu().double().numpy(),
            changed.detach().cpu().double().numpy(),
            phase.FP32_U,
            "fp32",
        )
        stored = stored_first[(row["checkpoint"], 0)]
        for key in ("absolute_max", "scaled_max", "rms_relative"):
            delta = abs(metrics[key] - stored["fp32"][key])
            maximum_gauge_metric_delta = max(maximum_gauge_metric_delta, delta)
            gauge_replay_ok = gauge_replay_ok and close(metrics[key], stored["fp32"][key], atol=1e-12, rtol=1e-8)
        gauge_replay_ok = gauge_replay_ok and close(
            float((original.argmax(1) == changed.argmax(1)).float().mean().item()),
            stored["decision_agreement"],
        )
        broken_seed = 11848500 + payload["task_index"] * 100 + payload["replicate"]
        broken = direct_gauge(model, broken_seed, device, broken=True)
        broken_logits, _ = direct_fp32(broken, data["all_x"], device)
        broken_metrics = local_forward(
            original.detach().cpu().double().numpy(),
            broken_logits.detach().cpu().double().numpy(),
            phase.FP32_U,
            "fp32",
        )
        stored_broken = stored_positive[row["checkpoint"]]
        positive_replay_ok = positive_replay_ok and close(
            broken_metrics["scaled_max"], stored_broken["fp32"]["scaled_max"], atol=1e-12, rtol=1e-8
        )
        positive_replay_ok = positive_replay_ok and close(
            float((original.argmax(1) == broken_logits.argmax(1)).float().mean().item()),
            stored_broken["decision_agreement"],
        )
        del model, transformed, broken
        torch.cuda.empty_cache()
    append_check(checks, "independent_first_gauge_replay", gauge_replay_ok, maximum_gauge_metric_delta)
    append_check(checks, "independent_positive_replay", positive_replay_ok)
    append_check(checks, "stress_non_gating", final["stress_tier"]["status"] == "descriptive_non_gating")
    final_copy = dict(final)
    final_digest = final_copy.pop("final_digest")
    append_check(checks, "final_digest", digest(final_copy) == final_digest)
    expected_primary = bool(
        final["confirmation"]["behavior_pass"]
        and final["confirmation"]["domain_coverage_pass"]
        and final["gauge"]["gauge_pass"]
        and final["positive_control"]["positive_control_pass"]
    )
    append_check(checks, "primary_decision_recompute", final["primary_pass"] == expected_primary)
    append_check(checks, "mechanism_not_tested", final["mechanism_camera_status"] == "not_tested_by_design")
    append_check(checks, "phase1183_unchanged", final["phase1183_status"] == "unchanged_frozen_failure")
    append_check(checks, "auto_continue_consistent", final["auto_continue"]["authorized"] == final["primary_pass"])

    passed = all(item["pass"] for item in checks)
    result = {
        "phase": phase.PHASE,
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_digest": protocol_digest,
        "final_digest": final_digest,
        "check_count": len(checks),
        "passed_check_count": sum(int(item["pass"]) for item in checks),
        "audit_pass": passed,
        "checks": checks,
    }
    result["audit_digest"] = digest(result)
    write_json(AUDIT_PATH, result)
    print(canonical_json(result))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("audit",))
    args = parser.parse_args()
    if args.command == "audit":
        audit()


if __name__ == "__main__":
    main()
