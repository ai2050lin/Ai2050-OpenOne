#!/usr/bin/env python3
"""Phase1178 controlled dual-path implementation-library development package.

Commands are intentionally separated so the preregistration exists before any
formal result is generated.  The package closes the engineering prerequisites
from Phase1177; it does not claim a natural-network mechanism discovery.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

import phase1178_implementation_library as lib


PHASE = 1178
ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "tests/glm5/result/phase1178_dual_path_implementation_library"
PROTOCOL_PATH = OUT_ROOT / "protocol/preregistration.json"
SCRIPT_PATH = Path(__file__).resolve()
LIBRARY_PATH = ROOT / "tests/glm5/phase1178_implementation_library.py"
AUDIT_PATH = ROOT / "tests/glm5/phase1178_dual_path_implementation_library_audit.py"
MEMO_PATH = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
BLOCKS_PER_TASK = 16


@dataclass(frozen=True)
class SplitConfig:
    name: str
    seed: int
    tasks: tuple[lib.TaskSpec, ...]


SPLITS = {
    "discovery": SplitConfig(
        "discovery",
        1_178_100_000,
        (
            lib.TaskSpec("d_linear_2", 11, "a + (1 + 2b) mod 11", (1, 2)),
            lib.TaskSpec("d_quadratic", 11, "a + (3 + b^2) mod 11", (3, 0, 1)),
            lib.TaskSpec("d_cubic", 11, "a + (2b + b^3) mod 11", (0, 2, 0, 1)),
            lib.TaskSpec("d_mixed", 11, "a + (4 + b + 2b^2) mod 11", (4, 1, 2)),
        ),
    ),
    "confirmation": SplitConfig(
        "confirmation",
        1_178_900_000,
        (
            lib.TaskSpec("c_linear_3", 13, "a + (2 + 3b) mod 13", (2, 3)),
            lib.TaskSpec("c_quadratic", 13, "a + (5 + 2b^2) mod 13", (5, 0, 2)),
            lib.TaskSpec("c_power5", 13, "a + (b + b^5) mod 13", (0, 1, 0, 0, 0, 1)),
            lib.TaskSpec("c_mixed", 13, "a + (6 + 2b + 3b^3) mod 13", (6, 2, 0, 3)),
        ),
    ),
}

THRESHOLDS = {
    "natural_accuracy_min": 0.999999,
    "natural_margin_gap_max": 1.0e-9,
    "neutral_response_gap_max": 1.0e-9,
    "diagnostic_response_gap_min": 7.9,
    "correct_rescue_recovery_min": 0.999999,
    "correct_rescue_spectrum_error_max": 1.0e-9,
    "wrong_family_spectrum_error_min": 7.9,
    "wrong_task_accuracy_max": 0.40,
    "shuffled_accuracy_max": 0.40,
    "random_accuracy_max": 0.40,
    "non_target_recovery_max": 1.0e-9,
    "paired_public_state_gap_max": 0.0,
    "gauge_gram_gap_max": 1.0e-9,
    "null_decode_accuracy_max": 0.55,
    "leaky_sentinel_accuracy_min": 0.99,
}


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(lib.canonical(row) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def task_payload(task: lib.TaskSpec) -> dict[str, Any]:
    value = asdict(task)
    value["coefficients"] = list(task.coefficients)
    value["table_digest"] = lib.digest(task.table().tolist())
    value["offset_digest"] = lib.digest(task.offsets().tolist())
    return value


def protocol_payload() -> dict[str, Any]:
    payload = {
        "phase": PHASE,
        "schema_version": "phase1178.protocol.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "known-truth development package; no natural-network mechanism claim",
        "scripts": {
            "main_sha256": sha256_file(SCRIPT_PATH),
            "library_sha256": sha256_file(LIBRARY_PATH),
            "audit_sha256": sha256_file(AUDIT_PATH),
        },
        "splits": {
            name: {
                "seed": config.seed,
                "tasks": [task_payload(task) for task in config.tasks],
                "blocks_per_task": BLOCKS_PER_TASK,
                "expected_system_count": len(config.tasks) * BLOCKS_PER_TASK * 4,
            }
            for name, config in SPLITS.items()
        },
        "implementations": list(lib.IMPLEMENTATIONS),
        "neutral_interventions": list(lib.NEUTRAL_INTERVENTIONS),
        "diagnostic_interventions": list(lib.DIAGNOSTIC_INTERVENTIONS),
        "rescue_controls": [
            "correct_family_same_task",
            "wrong_family_same_task",
            "same_family_wrong_task",
            "shuffled_order",
            "scale_matched_random",
            "non_target_slot",
        ],
        "anti_leakage_controls": [
            "architecture_only",
            "slot_only",
            "channel_only",
            "norm_only",
            "loss_only",
            "confidence_only",
            "progress_only",
            "gate_summary_only",
            "public_state",
            "balanced_random_label",
            "leaky_family_sentinel",
        ],
        "public_schema_excludes": [
            "implementation_family",
            "active_slot",
            "mechanisms_by_slot",
            "channel_permutation",
            "channel_signs",
            "rescue_donor_family",
        ],
        "thresholds": THRESHOLDS,
        "stopping_rule": "one-shot; no threshold, task, intervention, or observation search after formal generation",
    }
    payload["protocol_digest"] = lib.digest(payload)
    return payload


def preregister(force: bool) -> dict[str, Any]:
    if OUT_ROOT.exists() and force:
        shutil.rmtree(OUT_ROOT)
    if PROTOCOL_PATH.exists() and not force:
        raise RuntimeError(f"protocol already exists: {PROTOCOL_PATH}")
    payload = protocol_payload()
    write_json(PROTOCOL_PATH, payload)
    return payload


def device_from_arg(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(name)


def family_label(family: str) -> int:
    return -1 if family == lib.IMPLEMENTATIONS[0] else 1


def opaque_system_id(split: str, task: str, block: int, family: str, slot: int) -> str:
    return lib.digest({
        "salt": "phase1178-opaque-system-id",
        "split": split,
        "task": task,
        "block": block,
        "family": family,
        "slot": slot,
    })[:20]


def donor_logits(
    task: lib.TaskSpec,
    family: str,
    intervention: str | None,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    payload = lib.ImplementationFamilyGenerator().generate(task)
    model = lib.SymmetricDualPathHypernetwork(payload, device)
    a, b = lib.all_pairs(task.modulus, device)
    mechanisms = lib.mechanisms_for(family, 0)
    return model(a, b, mechanisms, 0, intervention=intervention), task.table().reshape(-1)


def rescue_metrics(
    model: lib.SymmetricDualPathHypernetwork,
    task: lib.TaskSpec,
    wrong_task: lib.TaskSpec,
    family: str,
    active_slot: int,
    mechanisms: tuple[str, str],
    a: torch.Tensor,
    b: torch.Tensor,
    target: torch.Tensor,
    seed: int,
) -> dict[str, Any]:
    baseline_logits = model(a, b, mechanisms, active_slot)
    baseline_margin = lib.correct_margin(baseline_logits, target)
    destroyed_logits = model(a, b, mechanisms, active_slot, destroy_slot=active_slot)
    destroyed_margin = lib.correct_margin(destroyed_logits, target)
    denominator = max(baseline_margin - destroyed_margin, 1.0e-12)
    receiver_spectrum = lib.response_spectrum(
        model, a, b, target, mechanisms, active_slot, lib.DIAGNOSTIC_INTERVENTIONS,
    )

    def inject(payload: torch.Tensor, slot: int = active_slot) -> tuple[float, float]:
        logits = model(
            a, b, mechanisms, active_slot,
            destroy_slot=active_slot,
            rescue_slot=slot,
            rescue_logits=payload,
        )
        margin = lib.correct_margin(logits, target)
        return lib.accuracy(logits, target), (margin - destroyed_margin) / denominator

    correct_payload, _ = donor_logits(task, family, None, a.device)
    wrong_family = lib.IMPLEMENTATIONS[1] if family == lib.IMPLEMENTATIONS[0] else lib.IMPLEMENTATIONS[0]
    wrong_family_payload, _ = donor_logits(task, wrong_family, None, a.device)
    wrong_task_payload, _ = donor_logits(wrong_task, family, None, a.device)
    shuffled_payload = torch.flip(correct_payload, dims=(0,))
    random_payload = lib.normalized_random_logits(correct_payload, seed + 71)

    correct_accuracy, correct_recovery = inject(correct_payload)
    wrong_family_accuracy, wrong_family_recovery = inject(wrong_family_payload)
    wrong_task_accuracy, wrong_task_recovery = inject(wrong_task_payload)
    shuffled_accuracy, shuffled_recovery = inject(shuffled_payload)
    random_accuracy, random_recovery = inject(random_payload)
    non_target_accuracy, non_target_recovery = inject(correct_payload, 1 - active_slot)

    correct_spectrum = {}
    wrong_family_spectrum = {}
    for intervention in lib.DIAGNOSTIC_INTERVENTIONS:
        correct_diag, _ = donor_logits(task, family, intervention, a.device)
        wrong_diag, _ = donor_logits(task, wrong_family, intervention, a.device)
        correct_spectrum[intervention] = lib.correct_margin(correct_diag, target) - baseline_margin
        wrong_family_spectrum[intervention] = lib.correct_margin(wrong_diag, target) - baseline_margin

    return {
        "baseline_accuracy": lib.accuracy(baseline_logits, target),
        "baseline_margin": baseline_margin,
        "destroyed_accuracy": lib.accuracy(destroyed_logits, target),
        "destroyed_margin": destroyed_margin,
        "correct": {
            "accuracy": correct_accuracy,
            "recovery_fraction": correct_recovery,
            "spectrum_max_error": lib.diagnostic_distance(receiver_spectrum, correct_spectrum),
        },
        "wrong_family": {
            "accuracy": wrong_family_accuracy,
            "recovery_fraction": wrong_family_recovery,
            "spectrum_max_error": lib.diagnostic_distance(receiver_spectrum, wrong_family_spectrum),
        },
        "wrong_task": {"accuracy": wrong_task_accuracy, "recovery_fraction": wrong_task_recovery},
        "shuffled": {"accuracy": shuffled_accuracy, "recovery_fraction": shuffled_recovery},
        "scale_matched_random": {"accuracy": random_accuracy, "recovery_fraction": random_recovery},
        "non_target_slot": {"accuracy": non_target_accuracy, "recovery_fraction": non_target_recovery},
        "receiver_diagnostic_spectrum": receiver_spectrum,
        "correct_donor_spectrum": correct_spectrum,
        "wrong_family_donor_spectrum": wrong_family_spectrum,
    }


def build_split(split: str, device: torch.device) -> dict[str, Any]:
    if not PROTOCOL_PATH.exists():
        raise RuntimeError("preregister before generating formal data")
    config = SPLITS[split]
    run_root = OUT_ROOT / f"runs/{split}"
    public_rows: list[dict[str, Any]] = []
    truth_rows: list[dict[str, Any]] = []
    rescue_rows: list[dict[str, Any]] = []
    state_arrays: dict[str, np.ndarray] = {}
    generator = lib.ImplementationFamilyGenerator()

    for task_index, task in enumerate(config.tasks):
        payload = generator.generate(task)
        wrong_task = config.tasks[(task_index + 1) % len(config.tasks)]
        for block in range(BLOCKS_PER_TASK):
            channel_seed = config.seed + task_index * 100_003 + block * 1_009
            permutation, signs = lib.make_channel_gauge(channel_seed)
            channel_digest = lib.digest({"permutation": permutation.tolist(), "signs": signs.tolist()})
            reference_state = None
            reference_gram = None
            for family in lib.IMPLEMENTATIONS:
                for active_slot in (0, 1):
                    system_id = opaque_system_id(split, task.name, block, family, active_slot)
                    model = lib.SymmetricDualPathHypernetwork(payload, device)
                    mechanisms = lib.mechanisms_for(family, active_slot)
                    a, b = lib.all_pairs(task.modulus, device)
                    target = torch.as_tensor(task.table().reshape(-1), device=device, dtype=torch.long)
                    logits = model(a, b, mechanisms, active_slot)
                    state = lib.observation_state(a, b, logits, task.modulus, permutation, signs)
                    state_arrays[system_id] = state
                    if reference_state is None:
                        reference_state = state
                        reference_gram = lib.gauge_invariant_gram(state)
                    state_gap = float(np.max(np.abs(state - reference_state)))
                    identity_permutation = np.arange(lib.OBSERVATION_WIDTH)
                    identity_signs = np.ones(lib.OBSERVATION_WIDTH)
                    ungauged = lib.observation_state(
                        a, b, logits, task.modulus, identity_permutation, identity_signs,
                    )
                    gram_gap = float(np.max(np.abs(lib.gauge_invariant_gram(ungauged) - reference_gram)))
                    natural_accuracy = lib.accuracy(logits, target)
                    natural_margin = lib.correct_margin(logits, target)
                    neutral = lib.response_spectrum(
                        model, a, b, target, mechanisms, active_slot, lib.NEUTRAL_INTERVENTIONS,
                    )
                    diagnostic = lib.response_spectrum(
                        model, a, b, target, mechanisms, active_slot, lib.DIAGNOSTIC_INTERVENTIONS,
                    )
                    rescue = rescue_metrics(
                        model, task, wrong_task, family, active_slot, mechanisms,
                        a, b, target, channel_seed + family_label(family) * 13,
                    )
                    rescue_rows.append({"system_id": system_id, **rescue})
                    public_rows.append({
                        "system_id": system_id,
                        "split": split,
                        "task_name": task.name,
                        "task_public_digest": lib.digest({"task": task.name, "table": task.table().tolist()}),
                        "block": block,
                        "observation_key": system_id,
                        "observation_digest": hashlib.sha256(state.tobytes()).hexdigest(),
                        "architecture_digest": model.architecture_digest,
                        "parameter_budget": model.parameter_budget,
                        "parameter_l2": model.parameter_l2,
                        "natural_accuracy": natural_accuracy,
                        "natural_loss": float(torch.nn.functional.cross_entropy(logits, target).item()),
                        "natural_confidence": float(torch.softmax(logits, dim=1).gather(1, target[:, None]).mean().item()),
                        "natural_margin": natural_margin,
                        "progress": 1.0,
                        "gate_l1": 1.0,
                        "gate_l2": 1.0,
                        "active_gate_count": 1,
                    })
                    truth_rows.append({
                        "system_id": system_id,
                        "split": split,
                        "task_name": task.name,
                        "block": block,
                        "implementation_family": family,
                        "family_label": family_label(family),
                        "active_slot": active_slot,
                        "mechanisms_by_slot": list(mechanisms),
                        "channel_seed": channel_seed,
                        "channel_key_digest": channel_digest,
                        "payload_digest": payload.payload_digest,
                        "public_state_pair_gap": state_gap,
                        "gauge_gram_gap": gram_gap,
                        "neutral_response_spectrum": neutral,
                        "diagnostic_response_spectrum": diagnostic,
                    })

    write_jsonl(run_root / "public_manifest.jsonl", public_rows)
    write_jsonl(run_root / "sealed_truth.jsonl", truth_rows)
    write_jsonl(run_root / "rescue_responses.jsonl", rescue_rows)
    np.savez_compressed(run_root / "public_states.npz", **state_arrays)
    summary = summarize_split(split, public_rows, truth_rows, rescue_rows, state_arrays)
    write_json(run_root / "summary.json", summary)
    return summary


def one_hot(values: list[Any]) -> np.ndarray:
    unique = {value: index for index, value in enumerate(sorted(set(values), key=str))}
    matrix = np.zeros((len(values), len(unique)), dtype=np.float64)
    for row, value in enumerate(values):
        matrix[row, unique[value]] = 1.0
    return matrix


def balanced_random_labels(truth_rows: list[dict[str, Any]]) -> np.ndarray:
    # Each task/block quartet remains exactly 2/2, independently of family.
    return np.asarray([1 if (row["active_slot"] ^ (row["block"] & 1)) else -1 for row in truth_rows])


def decode_control(
    features: np.ndarray,
    labels: np.ndarray,
    truth_rows: list[dict[str, Any]],
) -> float:
    train = np.asarray([row["block"] % 2 == 0 for row in truth_rows])
    test = ~train
    return lib.ridge_binary_accuracy(features[train], labels[train], features[test], labels[test])


def summarize_split(
    split: str,
    public_rows: list[dict[str, Any]],
    truth_rows: list[dict[str, Any]],
    rescue_rows: list[dict[str, Any]],
    states: dict[str, np.ndarray],
) -> dict[str, Any]:
    joined = [(public, truth) for public, truth in zip(public_rows, truth_rows)]
    labels = np.asarray([truth["family_label"] for _, truth in joined], dtype=np.int64)
    slot = one_hot([truth["active_slot"] for _, truth in joined])
    channel = one_hot([truth["channel_key_digest"] for _, truth in joined])
    architecture = one_hot([public["architecture_digest"] for public, _ in joined])
    public_state = np.stack([states[public["observation_key"]].reshape(-1) for public, _ in joined])
    scalar_names = (
        "parameter_l2", "natural_loss", "natural_confidence", "progress",
        "gate_l1", "gate_l2", "active_gate_count",
    )
    controls = {
        "architecture_only": decode_control(architecture, labels, truth_rows),
        "slot_only": decode_control(slot, labels, truth_rows),
        "channel_only": decode_control(channel, labels, truth_rows),
        "public_state": decode_control(public_state, labels, truth_rows),
    }
    for name in scalar_names:
        controls[f"{name}_only"] = decode_control(
            np.asarray([[public[name]] for public, _ in joined]), labels, truth_rows,
        )
    random_labels = balanced_random_labels(truth_rows)
    controls["balanced_random_label"] = decode_control(public_state, random_labels, truth_rows)
    controls["leaky_family_sentinel"] = decode_control(labels[:, None], labels, truth_rows)

    family_spectra = {}
    neutral_spectra = {}
    for family in lib.IMPLEMENTATIONS:
        rows = [row for row in truth_rows if row["implementation_family"] == family]
        family_spectra[family] = {
            name: float(np.median([row["diagnostic_response_spectrum"][name] for row in rows]))
            for name in lib.DIAGNOSTIC_INTERVENTIONS
        }
        neutral_spectra[family] = {
            name: float(np.median([row["neutral_response_spectrum"][name] for row in rows]))
            for name in lib.NEUTRAL_INTERVENTIONS
        }
    neutral_gap = max(
        abs(neutral_spectra[lib.IMPLEMENTATIONS[0]][name] - neutral_spectra[lib.IMPLEMENTATIONS[1]][name])
        for name in lib.NEUTRAL_INTERVENTIONS
    )
    diagnostic_gap = lib.diagnostic_distance(
        family_spectra[lib.IMPLEMENTATIONS[0]], family_spectra[lib.IMPLEMENTATIONS[1]],
    )
    rescue_by_id = {row["system_id"]: row for row in rescue_rows}

    metrics = {
        "system_count": len(public_rows),
        "task_count": len(set(row["task_name"] for row in public_rows)),
        "natural_accuracy_min": min(row["natural_accuracy"] for row in public_rows),
        "natural_margin_range": max(row["natural_margin"] for row in public_rows) - min(row["natural_margin"] for row in public_rows),
        "neutral_response_family_gap": neutral_gap,
        "diagnostic_response_family_gap": diagnostic_gap,
        "paired_public_state_gap_max": max(row["public_state_pair_gap"] for row in truth_rows),
        "gauge_gram_gap_max": max(row["gauge_gram_gap"] for row in truth_rows),
        "correct_rescue_recovery_min": min(row["correct"]["recovery_fraction"] for row in rescue_rows),
        "correct_rescue_spectrum_error_max": max(row["correct"]["spectrum_max_error"] for row in rescue_rows),
        "wrong_family_natural_accuracy_min": min(row["wrong_family"]["accuracy"] for row in rescue_rows),
        "wrong_family_spectrum_error_min": min(row["wrong_family"]["spectrum_max_error"] for row in rescue_rows),
        "wrong_task_accuracy_max": max(row["wrong_task"]["accuracy"] for row in rescue_rows),
        "shuffled_accuracy_max": max(row["shuffled"]["accuracy"] for row in rescue_rows),
        "random_accuracy_max": max(row["scale_matched_random"]["accuracy"] for row in rescue_rows),
        "non_target_recovery_max": max(row["non_target_slot"]["recovery_fraction"] for row in rescue_rows),
    }
    checks = {
        "natural_behavior_exact": metrics["natural_accuracy_min"] >= THRESHOLDS["natural_accuracy_min"],
        "natural_margin_matched": metrics["natural_margin_range"] <= THRESHOLDS["natural_margin_gap_max"],
        "neutral_response_matched": neutral_gap <= THRESHOLDS["neutral_response_gap_max"],
        "diagnostic_response_separated": diagnostic_gap >= THRESHOLDS["diagnostic_response_gap_min"],
        "correct_rescue_recovers": metrics["correct_rescue_recovery_min"] >= THRESHOLDS["correct_rescue_recovery_min"],
        "correct_rescue_restores_spectrum": metrics["correct_rescue_spectrum_error_max"] <= THRESHOLDS["correct_rescue_spectrum_error_max"],
        "wrong_family_rejected_by_spectrum": metrics["wrong_family_spectrum_error_min"] >= THRESHOLDS["wrong_family_spectrum_error_min"],
        "wrong_task_rejected": metrics["wrong_task_accuracy_max"] <= THRESHOLDS["wrong_task_accuracy_max"],
        "shuffled_rejected": metrics["shuffled_accuracy_max"] <= THRESHOLDS["shuffled_accuracy_max"],
        "random_rejected": metrics["random_accuracy_max"] <= THRESHOLDS["random_accuracy_max"],
        "non_target_rejected": abs(metrics["non_target_recovery_max"]) <= THRESHOLDS["non_target_recovery_max"],
        "public_state_exactly_matched": metrics["paired_public_state_gap_max"] <= THRESHOLDS["paired_public_state_gap_max"],
        "gauge_invariant": metrics["gauge_gram_gap_max"] <= THRESHOLDS["gauge_gram_gap_max"],
        "all_null_decoders_at_chance": all(
            accuracy <= THRESHOLDS["null_decode_accuracy_max"]
            for name, accuracy in controls.items() if name != "leaky_family_sentinel"
        ),
        "leaky_sentinel_detected": controls["leaky_family_sentinel"] >= THRESHOLDS["leaky_sentinel_accuracy_min"],
        "public_schema_clean": all(
            excluded not in public_rows[0]
            for excluded in read_json(PROTOCOL_PATH)["public_schema_excludes"]
        ),
        "family_slot_exact_balance": all(
            sum(row["implementation_family"] == family and row["active_slot"] == slot for row in truth_rows)
            == len(truth_rows) // 4
            for family in lib.IMPLEMENTATIONS for slot in (0, 1)
        ),
        "architecture_constant": len({row["architecture_digest"] for row in public_rows}) == 1,
        "parameter_budget_constant": len({row["parameter_budget"] for row in public_rows}) == 1,
    }
    result = {
        "phase": PHASE,
        "split": split,
        "observation_dtype": str(next(iter(states.values())).dtype),
        "metrics": metrics,
        "diagnostic_family_spectra": family_spectra,
        "neutral_family_spectra": neutral_spectra,
        "anti_leakage_decoder_accuracy": controls,
        "checks": checks,
        "passed": all(checks.values()),
    }
    result["summary_digest"] = lib.digest(result)
    return result


def analyze() -> dict[str, Any]:
    summaries = {name: read_json(OUT_ROOT / f"runs/{name}/summary.json") for name in SPLITS}
    payload = {
        "phase": PHASE,
        "schema_version": "phase1178.final.v1",
        "development_package_complete": all(summary["passed"] for summary in summaries.values()),
        "split_passes": {name: summary["passed"] for name, summary in summaries.items()},
        "evidence_scope": (
            "Known-truth instrument calibration only. It establishes a runnable symmetric dual-path object, "
            "implementation-family generator, joint natural-plus-spectrum rescue protocol, and anti-leakage "
            "controls; it does not establish early formation prediction or a natural-network mechanism."
        ),
        "component_status": {
            "symmetric_dual_path_hypernetwork": True,
            "implementation_family_generator": True,
            "rescue_protocol": True,
            "complete_anti_leakage_negative_controls": True,
            "endpoint_camera": False,
            "prefix_formation_camera": False,
            "natural_network_external_validity": False,
        },
        "auto_continue": False,
        "next_gate": (
            "Freeze an endpoint response-spectrum camera and an independent prefix prediction ledger before "
            "training any free network; do not infer a language mechanism from this development calibration."
        ),
        "summaries": summaries,
    }
    payload["final_digest"] = lib.digest(payload)
    write_json(OUT_ROOT / "analysis/final.json", payload)
    return payload


def probe(device: torch.device) -> dict[str, Any]:
    task = SPLITS["discovery"].tasks[0]
    payload = lib.ImplementationFamilyGenerator().generate(task)
    model = lib.SymmetricDualPathHypernetwork(payload, device)
    a, b = lib.all_pairs(task.modulus, device)
    target = torch.as_tensor(task.table().reshape(-1), device=device, dtype=torch.long)
    rows = []
    for family in lib.IMPLEMENTATIONS:
        mechanisms = lib.mechanisms_for(family, 0)
        rows.append({
            "family": family,
            "accuracy": lib.accuracy(model(a, b, mechanisms, 0), target),
            "diagnostic": lib.response_spectrum(
                model, a, b, target, mechanisms, 0, lib.DIAGNOSTIC_INTERVENTIONS,
            ),
        })
    value = {
        "device": str(device),
        "rows": rows,
        "passed": all(
            row["accuracy"] >= THRESHOLDS["natural_accuracy_min"]
            for row in rows
        ),
    }
    write_json(ROOT / "tests/glm5_temp/phase1178_dual_path_probe.json", value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("probe", "preregister", "run", "analyze", "all"))
    parser.add_argument("--split", choices=tuple(SPLITS), default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    device = device_from_arg(args.device)
    if args.command == "probe":
        result = probe(device)
    elif args.command == "preregister":
        result = preregister(args.force)
    elif args.command == "run":
        if args.split is None:
            raise SystemExit("--split is required for run")
        result = build_split(args.split, device)
    elif args.command == "analyze":
        result = analyze()
    else:
        probe(device)
        preregister(args.force)
        for split in SPLITS:
            build_split(split, device)
        result = analyze()
    print(lib.canonical(result))
    if isinstance(result, dict) and result.get("passed") is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
