#!/usr/bin/env python3
"""Independent recomputation audit for Phase1186.

The audit deliberately does not call the runner's Boolean reducer.  It
reconstructs all universal predicates from integer counts and independently
replays one gauge transform plus the broken-compensation control for every
fresh known-truth system.
"""

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

import phase1186_reducer_safe_numerical_qualification as phase  # noqa: E402


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
        return left.keys() == right.keys() and all(
            close(left[key], right[key], atol, rtol) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            close(a, b, atol, rtol) for a, b in zip(left, right)
        )
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
    unit = phase.p1185.FP64_U if precision == "fp64" else phase.p1185.FP32_U
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
    transformed = phase.p1185.p1171.RoleSquareNetwork(model.config).to(device)
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


def integer_reducer(equal: torch.Tensor, eligible: torch.Tensor) -> dict[str, Any]:
    eligible_count = int(torch.count_nonzero(eligible).item())
    agree_count = int(torch.count_nonzero(equal & eligible).item())
    return {
        "eligible_count": eligible_count,
        "agree_count": agree_count,
        "all_equal": agree_count == eligible_count,
        "abstained": eligible_count == 0,
        "descriptive_ratio": float(agree_count / eligible_count) if eligible_count else None,
    }


def local_exact_decision(
    reference: torch.Tensor,
    changed: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, Any]:
    threshold = phase.THRESHOLDS
    max_abs = max(float(reference.abs().max().item()), 1e-30)
    uncertainty = 2.0 * (
        threshold["fp32_absolute_floor"] + threshold["fp32_relative_multiplier"] * max_abs
    )
    top = torch.topk(reference, k=2, dim=1).values
    decision_eligible = (top[:, 0] - top[:, 1]) > uncertainty
    decision_equal = reference.argmax(1) == changed.argmax(1)
    margin_reference = phase.p1185.p1181.correct_margin(reference, targets.to(reference.device))
    margin_changed = phase.p1185.p1181.correct_margin(changed, targets.to(reference.device))
    margin_eligible = margin_reference.abs() > uncertainty
    margin_equal = (margin_reference >= 0) == (margin_changed >= 0)
    return {
        "uncertainty_bound": uncertainty,
        "decision": integer_reducer(decision_equal, decision_eligible),
        "margin_sign": integer_reducer(margin_equal, margin_eligible),
    }


def replay_gauge(
    model,
    x: torch.Tensor,
    y: torch.Tensor,
    stored: dict[str, Any],
    device: torch.device,
) -> bool:
    transformed = direct_gauge(model, stored["seed"], device)
    reference32, _ = direct_fp32(model, x, device)
    changed32, _ = direct_fp32(transformed, x, device)
    reference64, _ = direct_fp64(model, x)
    changed64, _ = direct_fp64(transformed, x)
    reference_feature = np.asarray(
        phase.p1185.p1183.algebraic_internal_features(model, x), dtype=np.float64
    )
    changed_feature = np.asarray(
        phase.p1185.p1183.algebraic_internal_features(transformed, x), dtype=np.float64
    )
    expected_feature = float(np.max(np.abs(reference_feature - changed_feature)))
    expected32 = local_forward(reference32.cpu().double().numpy(), changed32.cpu().double().numpy(), "fp32")
    expected64 = local_forward(reference64, changed64, "fp64")
    expected_decision = local_exact_decision(reference32, changed32, y)
    valid = (
        close(stored["feature_error"], expected_feature, atol=1e-12)
        and close(stored["fp32"], expected32, atol=1e-9)
        and close(stored["fp64"], expected64, atol=1e-13)
        and close(stored["exact_decision"], expected_decision, atol=1e-9)
    )
    del transformed
    return bool(valid)


def replay_positive(model, x: torch.Tensor, stored: dict[str, Any], device: torch.device) -> bool:
    broken = direct_gauge(model, stored["seed"], device, broken=True)
    reference, _ = direct_fp32(model, x, device)
    changed, _ = direct_fp32(broken, x, device)
    reference_feature = np.asarray(
        phase.p1185.p1183.algebraic_internal_features(model, x), dtype=np.float64
    )
    changed_feature = np.asarray(
        phase.p1185.p1183.algebraic_internal_features(broken, x), dtype=np.float64
    )
    feature_error = float(np.max(np.abs(reference_feature - changed_feature)))
    fp32 = local_forward(reference.cpu().double().numpy(), changed.cpu().double().numpy(), "fp32")
    agreement_count = int(torch.count_nonzero(reference.argmax(1) == changed.argmax(1)).item())
    total_count = int(reference.shape[0])
    agreement = agreement_count / total_count
    valid = (
        close(stored["feature_error"], feature_error, atol=1e-12)
        and close(stored["fp32"], fp32, atol=1e-9)
        and close(stored["decision_agreement"], agreement, atol=1e-9)
    )
    del broken
    return bool(valid)


def audit() -> None:
    if AUDIT_PATH.exists():
        raise RuntimeError("Phase1186 audit already exists")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for independent replay")

    protocol = read_json(phase.PROTOCOL_PATH)
    final = read_json(phase.FINAL_PATH)
    sentinel_path = phase.OUT_ROOT / "analysis/reducer_sentinels.jsonl"
    gauge_path = phase.OUT_ROOT / "analysis/gauge_rows.jsonl"
    positive_path = phase.OUT_ROOT / "analysis/positive_control_rows.jsonl"
    sentinels = read_jsonl(sentinel_path)
    gauge_rows = read_jsonl(gauge_path)
    positive_rows = read_jsonl(positive_path)
    checks: list[dict[str, Any]] = []

    protocol_copy = dict(protocol)
    protocol_digest = protocol_copy.pop("protocol_digest")
    append_check(checks, "protocol_digest", digest(protocol_copy) == protocol_digest)
    append_check(checks, "runner_hash", file_sha256(phase.SCRIPT) == protocol["scripts"]["runner"])
    append_check(checks, "audit_hash", file_sha256(SCRIPT) == protocol["scripts"]["audit"])
    append_check(
        checks,
        "phase1185_runner_hash",
        file_sha256(phase.p1185.SCRIPT) == protocol["scripts"]["phase1185_runner"],
    )
    append_check(
        checks,
        "phase1185_final_hash",
        file_sha256(phase.p1185.FINAL_PATH) == protocol["scripts"]["phase1185_final"],
    )
    previous_audit_path = phase.p1185.OUT_ROOT / "audit/independent_audit.json"
    append_check(
        checks,
        "phase1185_audit_hash",
        file_sha256(previous_audit_path) == protocol["scripts"]["phase1185_audit"],
    )
    append_check(
        checks,
        "phase1183_source_hash",
        file_sha256(Path(phase.p1185.p1183.__file__)) == protocol["scripts"]["phase1183_source"],
    )
    previous, previous_audit = phase.validate_phase1185()
    append_check(checks, "phase1185_independent_audit_pass", previous_audit["audit_pass"])
    append_check(checks, "phase1185_digest_link", previous["final_digest"] == protocol["phase1185_evidence"]["final_digest"])

    expected_systems = len(phase.SCALES) * len(phase.STRUCTURES) * phase.REPLICATES
    append_check(checks, "sentinel_count", len(sentinels) == len(phase.REDUCER_SENTINEL_LENGTHS))
    append_check(checks, "gauge_row_count", len(gauge_rows) == expected_systems * phase.TRANSFORMS)
    append_check(checks, "positive_row_count", len(positive_rows) == expected_systems)
    append_check(
        checks,
        "artifact_hashes",
        final["artifacts"]
        == {
            "sentinels_sha256": file_sha256(sentinel_path),
            "gauge_rows_sha256": file_sha256(gauge_path),
            "positive_rows_sha256": file_sha256(positive_path),
        },
    )

    sentinel_ok = True
    for stored, length in zip(sentinels, phase.REDUCER_SENTINEL_LENGTHS):
        values = torch.ones(length, dtype=torch.bool, device="cuda")
        eligible_count = int(torch.count_nonzero(values).item())
        agree_count = int(torch.count_nonzero(values & values).item())
        sentinel_ok = sentinel_ok and stored["length"] == length
        sentinel_ok = sentinel_ok and stored["eligible_count"] == eligible_count == length
        sentinel_ok = sentinel_ok and stored["agree_count"] == agree_count == length
        sentinel_ok = sentinel_ok and stored["exact_all_equal"] == (agree_count == eligible_count)
    append_check(checks, "integer_sentinels_independent_recompute", sentinel_ok)

    cases = {(row["scale"], row["structure"], row["replicate"]) for row in gauge_rows}
    append_check(checks, "fresh_case_factorial_complete", len(cases) == expected_systems)
    seeds = {row["model_seed"] for row in gauge_rows}
    append_check(checks, "fresh_model_seeds_unique", len(seeds) == expected_systems)
    transform_seeds = {row["seed"] for row in gauge_rows}
    append_check(checks, "fresh_transform_seeds_unique", len(transform_seeds) == len(gauge_rows))
    append_check(
        checks,
        "each_case_has_four_transforms",
        all(sum(row["case"] == case for row in gauge_rows) == phase.TRANSFORMS for case in range(expected_systems)),
    )

    exact_counts_valid = all(
        row["exact_decision"][name]["agree_count"]
        == row["exact_decision"][name]["eligible_count"]
        and row["exact_decision"][name]["all_equal"]
        and row["exact_decision"][name]["abstained"]
        == (row["exact_decision"][name]["eligible_count"] == 0)
        for row in gauge_rows
        for name in ("decision", "margin_sign")
    )
    append_check(checks, "stored_integer_universals_consistent", exact_counts_valid)

    gauge_fraction = sum(row["gauge_pass"] for row in gauge_rows) / len(gauge_rows)
    positive_fraction = sum(row["positive_control_pass"] for row in positive_rows) / len(positive_rows)
    sentinel_pass = all(row["exact_all_equal"] for row in sentinels)
    reducer_pass = bool(
        sentinel_pass
        and gauge_fraction == 1.0
        and positive_fraction >= phase.THRESHOLDS["positive_control_pass_fraction_min"]
    )
    append_check(checks, "gauge_fraction_recompute", close(gauge_fraction, final["gauge_pass_fraction"]))
    append_check(checks, "positive_fraction_recompute", close(positive_fraction, final["positive_control_pass_fraction"]))
    append_check(checks, "reducer_decision_recompute", reducer_pass == final["reducer_qualification_pass"])
    append_check(checks, "all_decision_count_equalities", final["all_decision_integer_equalities"])
    append_check(checks, "all_margin_count_equalities", final["all_margin_integer_equalities"])

    device = torch.device("cuda")
    x = torch.tensor(
        [(a, b) for a in range(phase.p1185.MODULUS) for b in range(phase.p1185.MODULUS)],
        dtype=torch.long,
    )
    y = torch.tensor(phase.p1185.task_table("axis_affine_b").reshape(-1), dtype=torch.long)
    first_by_case = {row["case"]: row for row in gauge_rows if row["transform"] == 0}
    positive_by_case = {row["case"]: row for row in positive_rows}
    gauge_replay_ok = True
    positive_replay_ok = True
    case = 0
    for scale_index, scale in enumerate(phase.SCALES):
        for structure_index, structure in enumerate(phase.STRUCTURES):
            for replicate in range(phase.REPLICATES):
                seed = phase.model_seed(scale_index, structure_index, replicate)
                model = phase.p1185.engineered_model(scale, structure, seed, device)
                gauge_replay_ok = gauge_replay_ok and replay_gauge(
                    model, x, y, first_by_case[case], device
                )
                positive_replay_ok = positive_replay_ok and replay_positive(
                    model, x, positive_by_case[case], device
                )
                del model
                torch.cuda.empty_cache()
                case += 1
    append_check(checks, "all_systems_first_transform_independent_replay", gauge_replay_ok)
    append_check(checks, "all_systems_positive_control_independent_replay", positive_replay_ok)

    final_copy = dict(final)
    final_digest = final_copy.pop("final_digest")
    append_check(checks, "final_digest", digest(final_copy) == final_digest)
    prior_ledger = bool(
        previous_audit["audit_pass"]
        and previous["numerical_axis"]["natural_gauge_pass_fraction"] == 1.0
        and previous["numerical_axis"]["safety_coverage"] == 1.0
        and previous["behavior_axis"]["behavior_axis_pass"]
        and previous["science_intersection"]["intersection_pass"]
    )
    expected_preaudit = bool(reducer_pass and prior_ledger)
    append_check(
        checks,
        "preaudit_authorization_recompute",
        final["phase1187_authorized_before_audit"] == expected_preaudit,
    )
    append_check(checks, "mechanism_camera_not_tested", final["mechanism_camera_status"] == "not_tested_by_design")

    audit_pass = all(check["pass"] for check in checks)
    authorized = bool(audit_pass and expected_preaudit)
    result = {
        "phase": phase.PHASE,
        "created_at_utc": phase.utc_now(),
        "protocol_digest": protocol_digest,
        "final_digest": final_digest,
        "check_count": len(checks),
        "pass_count": sum(check["pass"] for check in checks),
        "checks": checks,
        "audit_pass": audit_pass,
        "phase1187_authorized_after_audit": authorized,
        "claim_scope": "reducer_safe_numerical_qualification_only",
        "k165_status": "untested",
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
