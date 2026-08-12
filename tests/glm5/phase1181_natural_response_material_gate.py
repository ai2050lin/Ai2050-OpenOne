#!/usr/bin/env python3
"""Phase1181: quotient causal-response material gate in freely trained networks.

The discovery panel reuses sealed Phase1171 endpoints solely to calibrate a
material-existence criterion.  Confirmation uses the four response-unseen
Phase1172 confirmation task classes.  This phase does not fit a mechanism
camera, select a hotspot, or claim a semantic mechanism.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))

import phase1171_fixed_dimension_formation_trajectory_tomography as p1171  # noqa: E402
import phase1172_cross_quotient_event_time_prediction as p1172  # noqa: E402


PHASE = 1181
SCRIPT = Path(__file__).resolve()
AUDIT_SCRIPT = ROOT / "tests/glm5/phase1181_natural_response_material_gate_audit.py"
OUT_ROOT = ROOT / "tests/glm5/result/phase1181_natural_response_material_gate"
PROTOCOL_PATH = OUT_ROOT / "protocol/preregistration.json"
DISCOVERY_ROWS = OUT_ROOT / "runs/discovery/systems.jsonl"
DISCOVERY_SUMMARY = OUT_ROOT / "runs/discovery/summary.json"
CONFIRMATION_ROWS = OUT_ROOT / "runs/confirmation/systems.jsonl"
CONFIRMATION_SUMMARY = OUT_ROOT / "runs/confirmation/summary.json"
FINAL_PATH = OUT_ROOT / "analysis/final.json"

P1171_ROOT = ROOT / "tests/glm5/result/phase1171_fixed_dimension_formation_trajectory_tomography"
P1172_ROOT = ROOT / "tests/glm5/result/phase1172_cross_quotient_event_time_prediction"
P1171_CHECKPOINTS = P1171_ROOT / "runs/training/checkpoints"
P1172_CHECKPOINTS = P1172_ROOT / "runs/training/checkpoints"

THRESHOLDS = {
    "train_accuracy_min": 0.95,
    "holdout_accuracy_min": 0.90,
    "response_centered_norm_min": 1e-6,
    "replay_max_error_max": 1e-7,
    "gauge_fp32_logit_max_error_max": 1e-4,
    "gauge_ordered_response_max_error_max": 1e-5,
    "qualified_system_count_per_task_min": 6,
    "within_task_median_unit_shape_distance_min": 0.08,
    "behavior_matched_pair_count_per_task_min": 4,
    "behavior_matched_median_response_distance_min": 0.08,
    "absolute_behavior_response_distance_correlation_max": 0.75,
    "response_scale_coefficient_of_variation_min": 0.05,
    "discovery_passing_task_count_min": 7,
    "confirmation_passing_task_count_min": 3,
}

BEHAVIOR_FEATURES = (
    "train_accuracy",
    "holdout_accuracy",
    "train_loss",
    "holdout_loss",
    "train_mean_margin",
    "holdout_mean_margin",
    "parameter_norm",
)


@dataclass(frozen=True)
class DataPanel:
    x: torch.Tensor
    y: torch.Tensor
    train_mask: torch.Tensor
    holdout_mask: torch.Tensor


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


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
    os.replace(temporary, path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def endpoint_paths(split: str) -> list[Path]:
    if split == "discovery":
        paths = sorted(P1171_CHECKPOINTS.glob("*step10000.pt"))
        if len(paths) != 64:
            raise RuntimeError(f"expected 64 Phase1171 endpoints, found {len(paths)}")
        return paths
    if split == "confirmation":
        allowed = {task.name for task in p1172.TASK_SPECS if task.split == "confirmation"}
        paths = sorted(
            path
            for path in P1172_CHECKPOINTS.glob("*step12000.pt")
            if any(path.name.startswith(task_name + "_") for task_name in allowed)
        )
        if len(paths) != 32:
            raise RuntimeError(f"expected 32 Phase1172 confirmation endpoints, found {len(paths)}")
        return paths
    raise ValueError(split)


def checkpoint_manifest(paths: list[Path]) -> dict[str, str]:
    return {str(path.relative_to(ROOT)).replace("\\", "/"): file_sha256(path) for path in paths}


def source_artifacts() -> list[Path]:
    return [
        P1171_ROOT / "analysis/final.json",
        P1171_ROOT / "audit/independent_audit.json",
        P1171_ROOT / "audit/endpoint_representation_recompute.json",
        P1172_ROOT / "analysis/final.json",
        P1172_ROOT / "audit/independent_audit.json",
    ]


def preregister() -> None:
    if PROTOCOL_PATH.exists():
        raise RuntimeError("Phase1181 is already preregistered")
    if not AUDIT_SCRIPT.exists():
        raise RuntimeError(f"missing audit script: {AUDIT_SCRIPT}")
    for path in source_artifacts():
        if not path.exists():
            raise RuntimeError(f"missing source artifact: {path}")
    discovery = endpoint_paths("discovery")
    confirmation = endpoint_paths("confirmation")
    protocol = {
        "phase": PHASE,
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_object": (
            "Existence of non-degenerate causal response material in freely trained, "
            "ungated RoleSquareNetwork endpoints after quotienting exact signed hidden-channel permutations."
        ),
        "claim_exclusions": [
            "No mechanism identity is inferred.",
            "No semantic or language mechanism is inferred.",
            "No camera, hotspot, donor rescue, or prefix predictor is fitted.",
            "Passing only authorizes a separately preregistered camera phase.",
        ],
        "development_disclosure": (
            "Phase1171 endpoint responses were inspected in a development probe and are discovery-only. "
            "No Phase1172 confirmation endpoint response was inspected before this registration."
        ),
        "splits": {
            "discovery": {
                "source_phase": 1171,
                "task_count": 8,
                "system_count": 64,
                "endpoint_step": 10000,
            },
            "confirmation": {
                "source_phase": 1172,
                "task_names": [task.name for task in p1172.TASK_SPECS if task.split == "confirmation"],
                "task_count": 4,
                "system_count": 32,
                "endpoint_step": 12000,
            },
        },
        "response_operator": {
            "state": "FP32 pre-square hidden channel at the freely trained endpoint",
            "intervention": "set exactly one hidden channel to zero",
            "effect": "mean drop in correct-class margin on the sealed holdout panel",
            "quotient": "sort the 128 single-channel effects, then center and unit-normalize",
            "gauge": "exact signed hidden-channel permutations with compensating output-column permutation",
        },
        "behavior_matching": {
            "features": list(BEHAVIOR_FEATURES),
            "rule": (
                "Within each task, z-score behavior features, pair every system with its nearest "
                "behavior neighbor without reading responses, deduplicate unordered pairs, then "
                "measure quotient-response distances."
            ),
        },
        "thresholds": THRESHOLDS,
        "decision": {
            "discovery_authorization": "all numerical gates and at least 7/8 task gates",
            "primary_confirmation": "all numerical gates and at least 3/4 task gates",
            "failure_action": "stop; do not fit a response camera or tune task, regularizer, or intervention",
            "pass_action": "authorize a new, separately preregistered camera-and-rescue phase",
        },
        "scripts": {
            "runner": file_sha256(SCRIPT),
            "audit": file_sha256(AUDIT_SCRIPT),
            "phase1171_source": file_sha256(Path(p1171.__file__)),
            "phase1172_source": file_sha256(Path(p1172.__file__)),
        },
        "source_artifacts": {
            str(path.relative_to(ROOT)).replace("\\", "/"): file_sha256(path)
            for path in source_artifacts()
        },
        "checkpoint_manifests": {
            "discovery": checkpoint_manifest(discovery),
            "confirmation": checkpoint_manifest(confirmation),
        },
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }
    protocol["protocol_digest"] = digest(protocol)
    write_json(PROTOCOL_PATH, protocol)
    print(canonical_json({"registered": str(PROTOCOL_PATH), "digest": protocol["protocol_digest"]}))


def validate_protocol() -> dict[str, Any]:
    if not PROTOCOL_PATH.exists():
        raise RuntimeError("Phase1181 is not preregistered")
    protocol = read_json(PROTOCOL_PATH)
    stored_digest = protocol.pop("protocol_digest")
    if digest(protocol) != stored_digest:
        raise RuntimeError("protocol digest mismatch")
    protocol["protocol_digest"] = stored_digest
    expected_scripts = {
        "runner": SCRIPT,
        "audit": AUDIT_SCRIPT,
        "phase1171_source": Path(p1171.__file__),
        "phase1172_source": Path(p1172.__file__),
    }
    for name, path in expected_scripts.items():
        if file_sha256(path) != protocol["scripts"][name]:
            raise RuntimeError(f"frozen script changed: {name}")
    for relative, expected in protocol["source_artifacts"].items():
        if file_sha256(ROOT / relative) != expected:
            raise RuntimeError(f"source artifact changed: {relative}")
    for split in ("discovery", "confirmation"):
        current = checkpoint_manifest(endpoint_paths(split))
        if current != protocol["checkpoint_manifests"][split]:
            raise RuntimeError(f"checkpoint manifest changed: {split}")
    return protocol


def correct_margin(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    correct = logits.gather(1, targets[:, None]).squeeze(1)
    masked = logits.clone()
    masked.scatter_(1, targets[:, None], -torch.inf)
    return correct - masked.max(dim=1).values


def load_panel(payload: dict[str, Any], split: str) -> DataPanel:
    if split == "discovery":
        data = p1171.make_data(tuple(payload["operation"]), int(payload["seed"]))
    else:
        data = p1172.make_data(str(payload["task_name"]), int(payload["seed"]))
    x = torch.cat((data["train_x"], data["holdout_x"]), dim=0)
    y = torch.cat((data["train_y"], data["holdout_y"]), dim=0)
    train_mask = torch.zeros(len(x), dtype=torch.bool)
    train_mask[: len(data["train_x"])] = True
    return DataPanel(x=x, y=y, train_mask=train_mask, holdout_mask=~train_mask)


def load_model(payload: dict[str, Any], device: torch.device) -> p1171.RoleSquareNetwork:
    model = p1171.RoleSquareNetwork(p1171.RoleSquareConfig(**payload["config"])).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


@torch.inference_mode()
def fp32_state(
    model: p1171.RoleSquareNetwork,
    x: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = x.to(device)
    left = F.embedding(x[:, 0], model.left_embedding.weight.float())
    right = F.embedding(x[:, 1], model.right_embedding.weight.float())
    hidden = F.linear(left + right, model.hidden.weight.float())
    logits = F.linear(hidden.square(), model.output.weight.float())
    return logits, hidden


@torch.inference_mode()
def behavior_metrics(
    model: p1171.RoleSquareNetwork,
    panel: DataPanel,
    device: torch.device,
) -> dict[str, float]:
    x = panel.x.to(device)
    targets = panel.y.to(device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(x).float()
    predictions = logits.argmax(dim=1)
    margins = correct_margin(logits, targets)
    result: dict[str, float] = {}
    for name, mask in (("train", panel.train_mask), ("holdout", panel.holdout_mask)):
        selected = mask.to(device)
        result[f"{name}_accuracy"] = float((predictions[selected] == targets[selected]).float().mean().item())
        result[f"{name}_loss"] = float(F.cross_entropy(logits[selected], targets[selected]).item())
        result[f"{name}_mean_margin"] = float(margins[selected].mean().item())
    result["parameter_norm"] = math.sqrt(
        sum(float(parameter.detach().float().square().sum().item()) for parameter in model.parameters())
    )
    result["all_logits_finite"] = float(torch.isfinite(logits).all().item())
    return result


@torch.inference_mode()
def response_spectrum(
    model: p1171.RoleSquareNetwork,
    panel: DataPanel,
    device: torch.device,
) -> dict[str, Any]:
    logits, hidden = fp32_state(model, panel.x, device)
    targets = panel.y.to(device)
    selected = panel.holdout_mask.to(device)
    baseline_margin = correct_margin(logits, targets)
    squared = hidden.square()
    output_weight = model.output.weight.detach().float()
    responses: list[float] = []
    for start in range(0, hidden.shape[1], 16):
        stop = min(start + 16, hidden.shape[1])
        channels = torch.arange(start, stop, device=device)
        contribution = (
            squared[:, channels].transpose(0, 1)[:, :, None]
            * output_weight[:, channels].transpose(0, 1)[:, None, :]
        )
        changed_logits = logits[None] - contribution
        flat_targets = targets.repeat(stop - start)
        changed_margin = correct_margin(
            changed_logits.reshape(-1, changed_logits.shape[-1]), flat_targets
        ).reshape(stop - start, -1)
        effect = (baseline_margin[None, selected] - changed_margin[:, selected]).mean(dim=1)
        responses.extend(float(value) for value in effect.cpu().tolist())
    ordered = np.sort(np.asarray(responses, dtype=np.float64))
    centered = ordered - ordered.mean()
    centered_norm = float(np.linalg.norm(centered))
    unit_shape = centered / max(centered_norm, 1e-12)
    return {
        "ordered": ordered.tolist(),
        "unit_shape": unit_shape.tolist(),
        "mean": float(ordered.mean()),
        "standard_deviation": float(ordered.std()),
        "centered_norm": centered_norm,
        "mean_absolute_response": float(np.abs(ordered).mean()),
        "maximum_absolute_response": float(np.abs(ordered).max()),
    }


def gauge_model(
    model: p1171.RoleSquareNetwork,
    seed: int,
    device: torch.device,
) -> p1171.RoleSquareNetwork:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    width = model.config.width
    permutation = torch.randperm(width, generator=generator)
    signs = torch.where(
        torch.rand(width, generator=generator) < 0.5,
        torch.tensor(-1.0),
        torch.tensor(1.0),
    )
    transformed = p1171.RoleSquareNetwork(model.config).to(device)
    with torch.no_grad():
        transformed.left_embedding.weight.copy_(model.left_embedding.weight)
        transformed.right_embedding.weight.copy_(model.right_embedding.weight)
        transformed.hidden.weight.copy_(
            signs[:, None].to(device) * model.hidden.weight[permutation.to(device)]
        )
        transformed.output.weight.copy_(model.output.weight[:, permutation.to(device)])
    transformed.eval()
    return transformed


def gauge_check(
    model: p1171.RoleSquareNetwork,
    panel: DataPanel,
    reference_response: dict[str, Any],
    seed: int,
    device: torch.device,
) -> dict[str, float]:
    transformed = gauge_model(model, seed, device)
    original_logits, _ = fp32_state(model, panel.x, device)
    transformed_logits, _ = fp32_state(transformed, panel.x, device)
    transformed_response = response_spectrum(transformed, panel, device)
    result = {
        "fp32_logit_maximum_error": float((original_logits - transformed_logits).abs().max().item()),
        "ordered_response_maximum_error": float(
            np.max(
                np.abs(
                    np.asarray(reference_response["ordered"])
                    - np.asarray(transformed_response["ordered"])
                )
            )
        ),
    }
    del transformed
    return result


def run_split(split: str) -> None:
    protocol = validate_protocol()
    output_rows = DISCOVERY_ROWS if split == "discovery" else CONFIRMATION_ROWS
    output_summary = DISCOVERY_SUMMARY if split == "discovery" else CONFIRMATION_SUMMARY
    if output_rows.exists() or output_summary.exists():
        raise RuntimeError(f"{split} output already exists")
    if split == "confirmation":
        if not DISCOVERY_SUMMARY.exists() or not read_json(DISCOVERY_SUMMARY)["split_pass"]:
            raise RuntimeError("confirmation denied because discovery material gate did not pass")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    rows: list[dict[str, Any]] = []
    gauged_tasks: set[str] = set()
    for index, checkpoint_path in enumerate(endpoint_paths(split)):
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        task_name = str(payload["task_name"])
        model = load_model(payload, device)
        panel = load_panel(payload, split)
        behavior = behavior_metrics(model, panel, device)
        response = response_spectrum(model, panel, device)
        replay = response_spectrum(model, panel, device)
        replay_error = float(
            np.max(np.abs(np.asarray(response["ordered"]) - np.asarray(replay["ordered"])))
        )
        gauge = None
        if task_name not in gauged_tasks:
            gauge = gauge_check(model, panel, response, 11810000 + index, device)
            gauged_tasks.add(task_name)
        row = {
            "split": split,
            "source_phase": int(payload["phase"]),
            "checkpoint": checkpoint_path.name,
            "checkpoint_sha256": file_sha256(checkpoint_path),
            "task_name": task_name,
            "task_index": int(payload["task_index"]),
            "replicate": int(payload["replicate"]),
            "seed": int(payload["seed"]),
            "behavior": behavior,
            "response": response,
            "replay_maximum_error": replay_error,
            "gauge": gauge,
        }
        rows.append(row)
        del model
        torch.cuda.empty_cache()
        print(canonical_json({"split": split, "completed": index + 1, "total": len(endpoint_paths(split))}), flush=True)
    summary = summarize(rows, split, protocol["thresholds"])
    write_jsonl(output_rows, rows)
    write_json(output_summary, summary)
    print(canonical_json(summary))


def behavior_matched(items: list[dict[str, Any]]) -> dict[str, Any]:
    if len(items) < 2:
        return {"matched_pair_count": 0, "matched_response_distances": []}
    behavior = np.asarray(
        [[item["behavior"][name] for name in BEHAVIOR_FEATURES] for item in items],
        dtype=np.float64,
    )
    behavior = (behavior - behavior.mean(axis=0)) / np.maximum(behavior.std(axis=0), 1e-12)
    response = np.asarray([item["response"]["unit_shape"] for item in items], dtype=np.float64)
    nearest_pairs: set[tuple[int, int]] = set()
    all_behavior: list[float] = []
    all_response: list[float] = []
    for left in range(len(items)):
        candidates: list[tuple[float, int]] = []
        for right in range(len(items)):
            if left == right:
                continue
            behavior_distance = float(np.linalg.norm(behavior[left] - behavior[right]))
            candidates.append((behavior_distance, right))
            if right > left:
                all_behavior.append(behavior_distance)
                all_response.append(float(np.linalg.norm(response[left] - response[right])))
        nearest_pairs.add(tuple(sorted((left, min(candidates)[1]))))
    matched_response = [
        float(np.linalg.norm(response[left] - response[right]))
        for left, right in sorted(nearest_pairs)
    ]
    correlation = 0.0
    if np.std(all_behavior) > 1e-12 and np.std(all_response) > 1e-12:
        correlation = float(np.corrcoef(all_behavior, all_response)[0, 1])
    return {
        "matched_pair_count": len(matched_response),
        "matched_response_distances": matched_response,
        "matched_response_distance_median": float(np.median(matched_response)),
        "matched_response_distance_minimum": min(matched_response),
        "pairwise_behavior_response_distance_correlation": correlation,
    }


def summarize(rows: list[dict[str, Any]], split: str, thresholds: dict[str, Any]) -> dict[str, Any]:
    qualified = [
        row
        for row in rows
        if row["behavior"]["all_logits_finite"] == 1.0
        and row["behavior"]["train_accuracy"] >= thresholds["train_accuracy_min"]
        and row["behavior"]["holdout_accuracy"] >= thresholds["holdout_accuracy_min"]
        and row["response"]["centered_norm"] >= thresholds["response_centered_norm_min"]
    ]
    task_summaries: dict[str, Any] = {}
    for task_name in sorted({row["task_name"] for row in rows}):
        items = [row for row in qualified if row["task_name"] == task_name]
        shape = np.asarray([item["response"]["unit_shape"] for item in items], dtype=np.float64)
        distances = [
            float(np.linalg.norm(shape[left] - shape[right]))
            for left in range(len(shape))
            for right in range(left + 1, len(shape))
        ]
        matched = behavior_matched(items)
        median_distance = float(np.median(distances)) if distances else None
        correlation = matched.get("pairwise_behavior_response_distance_correlation")
        task_pass = bool(
            len(items) >= thresholds["qualified_system_count_per_task_min"]
            and median_distance is not None
            and median_distance >= thresholds["within_task_median_unit_shape_distance_min"]
            and matched["matched_pair_count"] >= thresholds["behavior_matched_pair_count_per_task_min"]
            and matched.get("matched_response_distance_median", -math.inf)
            >= thresholds["behavior_matched_median_response_distance_min"]
            and correlation is not None
            and abs(correlation) <= thresholds["absolute_behavior_response_distance_correlation_max"]
        )
        task_summaries[task_name] = {
            "system_count": sum(row["task_name"] == task_name for row in rows),
            "qualified_system_count": len(items),
            "pair_count": len(distances),
            "median_unit_shape_distance": median_distance,
            "minimum_unit_shape_distance": min(distances) if distances else None,
            "maximum_unit_shape_distance": max(distances) if distances else None,
            "behavior_matched": matched,
            "task_pass": task_pass,
        }
    scales = [row["response"]["centered_norm"] for row in qualified]
    gauges = [row["gauge"] for row in rows if row["gauge"] is not None]
    maximum_replay = max(row["replay_maximum_error"] for row in rows)
    maximum_gauge_logit = max(item["fp32_logit_maximum_error"] for item in gauges)
    maximum_gauge_response = max(item["ordered_response_maximum_error"] for item in gauges)
    scale_cv = float(np.std(scales) / max(np.mean(scales), 1e-12))
    numerical_pass = bool(
        maximum_replay <= thresholds["replay_max_error_max"]
        and maximum_gauge_logit <= thresholds["gauge_fp32_logit_max_error_max"]
        and maximum_gauge_response <= thresholds["gauge_ordered_response_max_error_max"]
        and scale_cv >= thresholds["response_scale_coefficient_of_variation_min"]
    )
    passing_task_count = sum(item["task_pass"] for item in task_summaries.values())
    task_required = (
        thresholds["discovery_passing_task_count_min"]
        if split == "discovery"
        else thresholds["confirmation_passing_task_count_min"]
    )
    summary = {
        "phase": PHASE,
        "split": split,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "system_count": len(rows),
        "qualified_system_count": len(qualified),
        "minimum_train_accuracy": min(row["behavior"]["train_accuracy"] for row in rows),
        "minimum_holdout_accuracy": min(row["behavior"]["holdout_accuracy"] for row in rows),
        "maximum_replay_error": maximum_replay,
        "maximum_gauge_fp32_logit_error": maximum_gauge_logit,
        "maximum_gauge_ordered_response_error": maximum_gauge_response,
        "response_scale_coefficient_of_variation": scale_cv,
        "numerical_pass": numerical_pass,
        "passing_task_count": passing_task_count,
        "required_passing_task_count": task_required,
        "task_summaries": task_summaries,
        "split_pass": bool(numerical_pass and passing_task_count >= task_required),
    }
    summary["rows_digest"] = digest(rows)
    summary["summary_digest"] = digest(summary)
    return summary


def analyze() -> None:
    protocol = validate_protocol()
    if not DISCOVERY_SUMMARY.exists():
        raise RuntimeError("missing discovery summary")
    discovery = read_json(DISCOVERY_SUMMARY)
    confirmation = read_json(CONFIRMATION_SUMMARY) if CONFIRMATION_SUMMARY.exists() else None
    primary_pass = bool(
        discovery["split_pass"] and confirmation is not None and confirmation["split_pass"]
    )
    final = {
        "phase": PHASE,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_digest": protocol["protocol_digest"],
        "discovery": discovery,
        "confirmation": confirmation,
        "primary_pass": primary_pass,
        "scientific_status": (
            "confirmed_response_material_only" if primary_pass else "response_material_gate_failed_or_untested"
        ),
        "claim_scope": (
            "A pass establishes response-spectrum diversity after an exact channel gauge quotient in this "
            "freely trained micro-network family. It does not identify, predict, or causally close a mechanism."
        ),
        "auto_continue": {
            "authorized": primary_pass,
            "next_phase": "separately preregistered camera and sealed-donor rescue" if primary_pass else None,
        },
    }
    final["final_digest"] = digest(final)
    write_json(FINAL_PATH, final)
    print(canonical_json(final))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("preregister", "run-discovery", "run-confirmation", "analyze"),
    )
    args = parser.parse_args()
    commands: dict[str, Callable[[], None]] = {
        "preregister": preregister,
        "run-discovery": lambda: run_split("discovery"),
        "run-confirmation": lambda: run_split("confirmation"),
        "analyze": analyze,
    }
    commands[args.command]()


if __name__ == "__main__":
    main()
