#!/usr/bin/env python3
"""Independent recomputation audit for Phase1185."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))

import phase1185_orthogonal_numerical_behavior_qualification as phase  # noqa: E402


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


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def close(left: Any, right: Any, atol: float = 1e-10, rtol: float = 1e-8) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(close(left[key], right[key], atol, rtol) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(close(a, b, atol, rtol) for a, b in zip(left, right))
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if isinstance(left, bool) or isinstance(right, bool):
            return left == right
        return bool(math.isclose(float(left), float(right), rel_tol=rtol, abs_tol=atol))
    return left == right


def append_check(checks: list[dict[str, Any]], name: str, passed: bool, detail: Any = None) -> None:
    checks.append({"name": name, "pass": bool(passed), "detail": detail})


def rms(array: np.ndarray) -> float:
    values = np.asarray(array, dtype=np.float64)
    return float(math.sqrt(max(float(np.mean(values * values)), 0.0)))


def local_forward(reference: np.ndarray, changed: np.ndarray, precision: str) -> dict[str, float]:
    unit = phase.FP64_U if precision == "fp64" else phase.FP32_U
    multiplier = 128.0 if precision == "fp64" else 32.0
    floor = (256.0 if precision == "fp64" else 128.0) * unit
    delta = np.asarray(changed, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    scale_max = max(float(np.max(np.abs(reference))), 1e-30)
    scale_rms = max(rms(reference), 1e-30)
    absolute = float(np.max(np.abs(delta)))
    return {
        "absolute_max": absolute,
        "reference_max_abs": scale_max,
        "scaled_max": absolute / scale_max,
        "rms_relative": rms(delta) / scale_rms,
        "mixed_absolute_bound": floor + multiplier * unit * scale_max,
    }


@torch.inference_mode()
def direct_fp32(model, x: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    ids = x.to(device)
    left = F.embedding(ids[:, 0], model.left_embedding.weight.float())
    right = F.embedding(ids[:, 1], model.right_embedding.weight.float())
    hidden = F.linear(left + right, model.hidden.weight.float())
    logits = F.linear(hidden.square(), model.output.weight.float())
    return logits, hidden


@torch.inference_mode()
def direct_fp64(model, x: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    left_e = model.left_embedding.weight.detach().cpu().double().numpy()
    right_e = model.right_embedding.weight.detach().cpu().double().numpy()
    hidden_w = model.hidden.weight.detach().cpu().double().numpy()
    output_w = model.output.weight.detach().cpu().double().numpy()
    ids = x.cpu().numpy()
    hidden = (left_e[ids[:, 0]] + right_e[ids[:, 1]]) @ hidden_w.T
    logits = (hidden * hidden) @ output_w.T
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


def local_decision(reference: torch.Tensor, changed: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    threshold = phase.THRESHOLDS
    max_abs = max(float(reference.abs().max().item()), 1e-30)
    uncertainty = 2.0 * (
        threshold["fp32_absolute_floor"] + threshold["fp32_relative_multiplier"] * max_abs
    )
    gaps = torch.topk(reference, k=2, dim=1).values
    eligible = (gaps[:, 0] - gaps[:, 1]) > uncertainty
    equal = reference.argmax(1) == changed.argmax(1)
    margin_reference = phase.p1181.correct_margin(reference, targets.to(reference.device))
    margin_changed = phase.p1181.correct_margin(changed, targets.to(reference.device))
    margin_eligible = margin_reference.abs() > uncertainty
    margin_equal = (margin_reference >= 0) == (margin_changed >= 0)
    return {
        "uncertainty_bound": uncertainty,
        "decision_eligible_fraction": float(eligible.float().mean().item()),
        "decision_agreement": float(equal[eligible].float().mean().item()) if bool(eligible.any()) else 1.0,
        "margin_sign_eligible_fraction": float(margin_eligible.float().mean().item()),
        "margin_sign_agreement": (
            float(margin_equal[margin_eligible].float().mean().item()) if bool(margin_eligible.any()) else 1.0
        ),
    }


def verify_training_seal(split: str, protocol_digest: str) -> tuple[dict[str, Any], bool]:
    base = phase.OUT_ROOT / "runs" / split
    seal = read_json(base / "training_seal.json")
    copy = dict(seal)
    stored = copy.pop("seal_digest")
    valid = digest(copy) == stored and seal["protocol_digest"] == protocol_digest
    valid = valid and file_sha256(base / "training_metrics.jsonl") == seal["training_metrics_sha256"]
    valid = valid and all(
        file_sha256(base / "checkpoints" / name) == expected
        for name, expected in seal["checkpoint_hashes"].items()
    )
    return seal, bool(valid)


def replay_gauge(model, x: torch.Tensor, y: torch.Tensor, stored: dict[str, Any], device: torch.device) -> bool:
    transformed = direct_gauge(model, stored["seed"], device)
    reference32, _ = direct_fp32(model, x, device)
    changed32, _ = direct_fp32(transformed, x, device)
    reference64, _ = direct_fp64(model, x)
    changed64, _ = direct_fp64(transformed, x)
    reference_feature = np.asarray(phase.p1183.algebraic_internal_features(model, x), dtype=np.float64)
    changed_feature = np.asarray(phase.p1183.algebraic_internal_features(transformed, x), dtype=np.float64)
    expected_feature = float(np.max(np.abs(reference_feature - changed_feature)))
    expected32 = local_forward(reference32.cpu().double().numpy(), changed32.cpu().double().numpy(), "fp32")
    expected64 = local_forward(reference64, changed64, "fp64")
    expected_decision = local_decision(reference32, changed32, y)
    valid = (
        close(stored["feature_error"], expected_feature, atol=1e-12)
        and close(stored["fp32"], expected32, atol=1e-9)
        and close(stored["fp64"], expected64, atol=1e-13)
        and close(stored["decision"], expected_decision, atol=1e-9)
    )
    del transformed
    return bool(valid)


def replay_positive(model, x: torch.Tensor, stored: dict[str, Any], device: torch.device) -> bool:
    broken = direct_gauge(model, stored["seed"], device, broken=True)
    reference, _ = direct_fp32(model, x, device)
    changed, _ = direct_fp32(broken, x, device)
    reference_feature = np.asarray(phase.p1183.algebraic_internal_features(model, x), dtype=np.float64)
    changed_feature = np.asarray(phase.p1183.algebraic_internal_features(broken, x), dtype=np.float64)
    feature_error = float(np.max(np.abs(reference_feature - changed_feature)))
    fp32 = local_forward(reference.cpu().double().numpy(), changed.cpu().double().numpy(), "fp32")
    agreement = float((reference.argmax(1) == changed.argmax(1)).float().mean().item())
    valid = (
        close(stored["feature_error"], feature_error, atol=1e-12)
        and close(stored["fp32"], fp32, atol=1e-9)
        and close(stored["decision_agreement"], agreement, atol=1e-9)
    )
    del broken
    return bool(valid)


def audit() -> None:
    if AUDIT_PATH.exists():
        raise RuntimeError("audit already exists")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for independent replay")

    protocol = read_json(phase.PROTOCOL_PATH)
    support = read_json(phase.DOMAIN_PATH)
    final = read_json(phase.FINAL_PATH)
    discovery_path = phase.OUT_ROOT / "runs/discovery/systems.jsonl"
    confirmation_path = phase.OUT_ROOT / "runs/confirmation/systems.jsonl"
    discovery = read_jsonl(discovery_path)
    confirmation = read_jsonl(confirmation_path)
    natural_gauge_path = phase.OUT_ROOT / "analysis/natural_gauge_rows.jsonl"
    natural_positive_path = phase.OUT_ROOT / "analysis/natural_positive_rows.jsonl"
    engineered_gauge_path = phase.OUT_ROOT / "analysis/engineered_gauge_rows.jsonl"
    engineered_positive_path = phase.OUT_ROOT / "analysis/engineered_positive_rows.jsonl"
    natural_gauge = read_jsonl(natural_gauge_path)
    natural_positive = read_jsonl(natural_positive_path)
    engineered_gauge = read_jsonl(engineered_gauge_path)
    engineered_positive = read_jsonl(engineered_positive_path)
    checks: list[dict[str, Any]] = []

    protocol_copy = dict(protocol)
    protocol_digest = protocol_copy.pop("protocol_digest")
    append_check(checks, "protocol_digest", digest(protocol_copy) == protocol_digest)
    append_check(checks, "runner_hash", file_sha256(phase.SCRIPT) == protocol["scripts"]["runner"])
    append_check(checks, "audit_hash", file_sha256(SCRIPT) == protocol["scripts"]["audit"])
    for key, module in (
        ("phase1171_source", phase.p1171),
        ("phase1181_source", phase.p1181),
        ("phase1183_source", phase.p1183),
        ("phase1184_source", phase.p1184),
    ):
        append_check(checks, f"{key}_hash", file_sha256(Path(module.__file__)) == protocol["scripts"][key])
    append_check(
        checks,
        "phase1184_stop_hash",
        file_sha256(phase.p1184.OUT_ROOT / "analysis/final_stop.json") == protocol["scripts"]["phase1184_stop"],
    )

    signatures = {task.name: phase.task_signature(task.name) for task in phase.TASK_SPECS}
    append_check(checks, "task_tables_unique", len({row["table_digest"] for row in signatures.values()}) == 8)
    append_check(checks, "task_signatures_frozen", all(row["signature"] == signatures[row["name"]] for row in protocol["tasks"]))
    discovery_seeds = {
        phase.model_seed(index, replicate)
        for index, task in enumerate(phase.TASK_SPECS)
        if task.split == "discovery"
        for replicate in range(phase.REPLICATES)
    }
    confirmation_seeds = {
        phase.model_seed(index, replicate)
        for index, task in enumerate(phase.TASK_SPECS)
        if task.split == "confirmation"
        for replicate in range(phase.REPLICATES)
    }
    append_check(checks, "split_seeds_disjoint", discovery_seeds.isdisjoint(confirmation_seeds))

    discovery_seal, discovery_seal_ok = verify_training_seal("discovery", protocol_digest)
    confirmation_seal, confirmation_seal_ok = verify_training_seal("confirmation", protocol_digest)
    append_check(checks, "discovery_training_seal", discovery_seal_ok)
    append_check(checks, "confirmation_training_seal", confirmation_seal_ok)
    append_check(checks, "discovery_system_count", len(discovery) == discovery_seal["system_count"] == 32)
    append_check(checks, "confirmation_system_count", len(confirmation) == confirmation_seal["system_count"] == 32)
    append_check(checks, "holdout_unread_at_both_seals", discovery_seal["all_holdout_labels_unread"] and confirmation_seal["all_holdout_labels_unread"])

    support_copy = dict(support)
    support_digest = support_copy.pop("support_digest")
    append_check(checks, "support_digest", digest(support_copy) == support_digest)
    append_check(checks, "support_rows_hash", file_sha256(discovery_path) == support["rows_sha256"])
    finite_discovery = [row for row in discovery if row["numerical_finite"]]
    cloud = np.stack([phase.profile_vector(row["profile"]) for row in finite_discovery])
    stored_cloud = np.asarray(support["log_profile_cloud"], dtype=np.float64)
    append_check(checks, "support_uses_all_finite_without_behavior_filter", len(finite_discovery) == len(discovery) == len(stored_cloud))
    append_check(checks, "support_cloud_exact", bool(np.allclose(cloud, stored_cloud, rtol=0.0, atol=1e-12)))
    nearest = [
        min(phase.l_inf_distance(cloud[index], cloud[j]) for j in range(len(cloud)) if j != index)
        for index in range(len(cloud))
    ]
    radius = phase.SAFETY_RADIUS_MULTIPLIER * max(nearest)
    append_check(checks, "support_nearest_recompute", close(nearest, support["leave_one_out_nearest_distances"], atol=1e-12))
    append_check(checks, "support_radius_recompute", close(radius, support["safety_radius"], atol=1e-12))
    expected_support_pass = bool(len(finite_discovery) >= 32 and math.isfinite(radius) and radius > 0.0)
    append_check(checks, "support_decision", support["numerical_support_pass"] == expected_support_pass)
    append_check(
        checks,
        "discovery_behavior_separate_recompute",
        close(support["behavior_ledger_descriptive_only"], phase.behavior_summary(discovery, "discovery"), atol=1e-9),
    )

    append_check(checks, "confirmation_rows_hash", file_sha256(confirmation_path) == final["systems_sha256"])
    finite_confirmation = [row for row in confirmation if row["numerical_finite"]]
    distance_ok = all(
        close(row["support_distance"], phase.support_distance(row["profile"], stored_cloud), atol=1e-12)
        and row["inside_safety_envelope"] == (row["support_distance"] <= support["safety_radius"])
        for row in finite_confirmation
    )
    append_check(checks, "confirmation_support_distances", distance_ok)
    coverage = sum(row["inside_safety_envelope"] for row in finite_confirmation) / max(len(finite_confirmation), 1)
    coverage_pass = bool(len(finite_confirmation) >= 32 and coverage >= phase.THRESHOLDS["confirmation_safety_coverage_min"])
    append_check(checks, "confirmation_coverage", close(coverage, final["numerical_axis"]["safety_coverage"]) and coverage_pass == final["numerical_axis"]["coverage_pass"])

    artifact_paths = {
        "natural_gauge": natural_gauge_path,
        "natural_positive": natural_positive_path,
        "engineered_gauge": engineered_gauge_path,
        "engineered_positive": engineered_positive_path,
    }
    append_check(
        checks,
        "analysis_artifact_hashes",
        all(file_sha256(path) == final["numerical_axis"]["artifact_hashes"][key] for key, path in artifact_paths.items()),
    )
    append_check(checks, "natural_gauge_row_count", len(natural_gauge) == 32 * phase.NATURAL_GAUGE_TRANSFORMS)
    append_check(checks, "engineered_gauge_row_count", len(engineered_gauge) == len(phase.ENGINEERED_SCALES) * len(phase.ENGINEERED_STRUCTURES) * phase.ENGINEERED_GAUGE_TRANSFORMS)
    append_check(checks, "natural_positive_row_count", len(natural_positive) == 32)
    append_check(checks, "engineered_positive_row_count", len(engineered_positive) == len(phase.ENGINEERED_SCALES) * len(phase.ENGINEERED_STRUCTURES))

    natural_fraction = sum(row["gauge_pass"] for row in natural_gauge) / max(len(natural_gauge), 1)
    engineered_fraction = sum(row["gauge_pass"] for row in engineered_gauge) / max(len(engineered_gauge), 1)
    natural_positive_fraction = sum(row["positive_control_pass"] for row in natural_positive) / max(len(natural_positive), 1)
    engineered_positive_fraction = sum(row["positive_control_pass"] for row in engineered_positive) / max(len(engineered_positive), 1)
    append_check(checks, "gauge_aggregates", close(natural_fraction, final["numerical_axis"]["natural_gauge_pass_fraction"]) and close(engineered_fraction, final["numerical_axis"]["engineered_gauge_pass_fraction"]))
    append_check(checks, "positive_aggregates", close(natural_positive_fraction, final["numerical_axis"]["natural_positive_fraction"]) and close(engineered_positive_fraction, final["numerical_axis"]["engineered_positive_fraction"]))
    numerical_axis_pass = bool(
        support["numerical_support_pass"]
        and coverage_pass
        and natural_fraction >= phase.THRESHOLDS["natural_gauge_pass_fraction_min"]
        and engineered_fraction >= phase.THRESHOLDS["engineered_gauge_pass_fraction_min"]
        and natural_positive_fraction >= phase.THRESHOLDS["positive_control_pass_fraction_min"]
        and engineered_positive_fraction >= phase.THRESHOLDS["positive_control_pass_fraction_min"]
    )
    append_check(checks, "numerical_axis_decision", numerical_axis_pass == final["numerical_axis"]["numerical_axis_pass"])

    behavior = phase.behavior_summary(confirmation, "confirmation")
    append_check(checks, "behavior_axis_recompute", close(behavior, final["behavior_axis"], atol=1e-9))
    intersection = [row for row in confirmation if row["behavior"]["qualified"] and row["inside_safety_envelope"]]
    per_task = {
        row["task_name"]: sum(item["task_name"] == row["task_name"] for item in intersection)
        for row in intersection
    }
    intersection_tasks = sum(count >= phase.THRESHOLDS["behavior_qualified_per_task_min"] for count in per_task.values())
    intersection_pass = bool(
        len(intersection) >= phase.THRESHOLDS["science_intersection_system_count_min"]
        and intersection_tasks >= phase.THRESHOLDS["science_intersection_task_count_min"]
    )
    append_check(
        checks,
        "science_intersection_recompute",
        final["science_intersection"] == {
            "system_count": len(intersection),
            "passing_task_count": intersection_tasks,
            "intersection_pass": intersection_pass,
        },
    )

    stress = final["stress_tier"]
    stress_rows = stress["rows"]
    stress_recompute = bool(
        stress["status"] == "descriptive_non_gating"
        and stress["row_count"] == len(stress_rows) == 16
        and close(stress["maximum_absolute_error"], max(row["fp32"]["absolute_max"] for row in stress_rows))
        and close(stress["maximum_scaled_error"], max(row["fp32"]["scaled_max"] for row in stress_rows))
        and close(stress["minimum_decision_agreement"], min(row["decision_agreement"] for row in stress_rows))
        and stress["all_finite"] == all(row["all_finite"] for row in stress_rows)
    )
    append_check(checks, "stress_descriptive_recompute", stress_recompute)

    device = torch.device("cuda")
    natural_gauge_first = {row["checkpoint"]: row for row in natural_gauge if row["transform"] == 0}
    natural_positive_by_checkpoint = {row["checkpoint"]: row for row in natural_positive}
    natural_replay_ok = True
    positive_replay_ok = True
    for row in confirmation:
        payload = torch.load(phase.OUT_ROOT / "runs/confirmation/checkpoints" / row["checkpoint"], map_location="cpu", weights_only=False)
        model = phase.load_model(payload, device)
        data = phase.make_data(payload["task_name"], payload["seed"] + 17)
        natural_replay_ok = natural_replay_ok and replay_gauge(
            model, data["all_x"], data["all_y"], natural_gauge_first[row["checkpoint"]], device
        )
        positive_replay_ok = positive_replay_ok and replay_positive(
            model, data["all_x"], natural_positive_by_checkpoint[row["checkpoint"]], device
        )
        del model
        torch.cuda.empty_cache()
    append_check(checks, "natural_first_transform_independent_replay", natural_replay_ok)
    append_check(checks, "natural_positive_independent_replay", positive_replay_ok)

    engineered_gauge_first = {row["case"]: row for row in engineered_gauge if row["transform"] == 0}
    engineered_positive_by_case = {row["case"]: row for row in engineered_positive}
    engineered_replay_ok = True
    engineered_positive_ok = True
    case = 0
    x = torch.tensor([(a, b) for a in range(phase.MODULUS) for b in range(phase.MODULUS)], dtype=torch.long)
    y = torch.tensor(phase.task_table("axis_affine_b").reshape(-1), dtype=torch.long)
    for scale in phase.ENGINEERED_SCALES:
        for structure in phase.ENGINEERED_STRUCTURES:
            model = phase.engineered_model(scale, structure, 11858000 + case, device)
            engineered_replay_ok = engineered_replay_ok and replay_gauge(
                model, x, y, engineered_gauge_first[case], device
            )
            engineered_positive_ok = engineered_positive_ok and replay_positive(
                model, x, engineered_positive_by_case[case], device
            )
            del model
            torch.cuda.empty_cache()
            case += 1
    append_check(checks, "engineered_first_transform_independent_replay", engineered_replay_ok)
    append_check(checks, "engineered_positive_independent_replay", engineered_positive_ok)

    final_copy = dict(final)
    final_digest = final_copy.pop("final_digest")
    append_check(checks, "final_digest", digest(final_copy) == final_digest)
    expected_authorized_before_audit = bool(numerical_axis_pass and behavior["behavior_axis_pass"] and intersection_pass)
    append_check(checks, "preaudit_authorization", final["mechanism_confirmation_authorized"] == expected_authorized_before_audit)
    append_check(checks, "mechanism_camera_not_tested", final["mechanism_camera_status"] == "not_tested_by_design")

    audit_pass = all(check["pass"] for check in checks)
    authorized = bool(audit_pass and expected_authorized_before_audit)
    result = {
        "phase": phase.PHASE,
        "created_at_utc": phase.utc_now(),
        "protocol_digest": protocol_digest,
        "support_digest": support_digest,
        "final_digest": final_digest,
        "check_count": len(checks),
        "pass_count": sum(check["pass"] for check in checks),
        "checks": checks,
        "audit_pass": audit_pass,
        "mechanism_confirmation_authorized_after_audit": authorized,
        "auto_continue": {
            "authorized": authorized,
            "next": "one_fresh_three_evidence_mechanism_confirmation" if authorized else None,
        },
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
