#!/usr/bin/env python3
"""Dense pre-rule trajectory acquisition on new free micro-Transformers.

Phase 1218 repairs the sampling-geometry failure exposed by Phase 1217.  It
adds dense early observations without changing the six-clock outcome labels:
clock onsets are still computed only on the original 100-step anchor grid.
The extra observations are reserved for a separately preregistered precursor
prediction phase.  This phase acquires data and checks identifiability only;
it does not select or evaluate a predictor.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import itertools
import json
import os
import random
import sys
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests/glm5"
sys.path.insert(0, str(TEST_ROOT))

import phase1217_factorial_free_formation_clock_transfer as core  # noqa: E402
from phase1146_learned_composition_benchmark import ModelConfig, TinyCausalTransformer  # noqa: E402


PHASE = 1218
SCRIPT = Path(__file__).resolve()
AUDIT_SCRIPT = SCRIPT.with_name("phase1218_dense_prerule_trajectory_acquisition_audit.py")
OUT_ROOT = TEST_ROOT / "result/phase1218_dense_prerule_trajectory_acquisition"
PROTOCOL_PATH = OUT_ROOT / "protocol/preregistration.json"
PREAUDIT_PATH = OUT_ROOT / "audit/independent_preaudit.json"
FINAL_PATH = OUT_ROOT / "analysis/final.json"
SOURCE1217 = TEST_ROOT / "result/phase1217_factorial_free_formation_clock_transfer"

EXPECTED_1217_FINAL = "33c4c376795c2f84286de649b7762c01ca9d4983039982de866ea85232852497"
EXPECTED_1217_AUDIT = "c45e655220afa863ba3501a42234e992cae57f40de492af2254f38e79ad656ba"

CLOCKS = core.CLOCKS
REPLICATES = 2
LANDMARK_STEP = 50
ANCHOR_INTERVAL = 100
OBSERVATION_STEPS = tuple(
    sorted(
        set(range(0, 101, 5))
        | set(range(125, 801, 25))
        | set(range(900, 2401, 100))
    )
)
ANCHOR_STEPS = tuple(range(0, 2401, ANCHOR_INTERVAL))
OBSERVATION_STEP_SET = set(OBSERVATION_STEPS)

# The abstract rules remain split-disjoint within this phase.  New lexicons,
# architectures, initialization seeds, and minibatch orders make every trained
# network independent of Phase 1217 while retaining outcome comparability.
TASKS = {
    "discovery": (
        {"name": "dense_identity", "source_roles": {"row": "row", "column": "column", "context": "context"}},
        {"name": "dense_cycle_forward", "source_roles": {"row": "column", "column": "context", "context": "row"}},
    ),
    "confirmation": (
        {"name": "dense_cycle_reverse", "source_roles": {"row": "context", "column": "row", "context": "column"}},
        {"name": "dense_swap_row_column", "source_roles": {"row": "column", "column": "row", "context": "context"}},
    ),
}

LEXICONS = {
    "discovery": (
        {"name": "dense_lexicon_d0", "seed": 1_218_101},
        {"name": "dense_lexicon_d1", "seed": 1_218_103},
    ),
    "confirmation": (
        {"name": "dense_lexicon_c0", "seed": 1_218_211},
        {"name": "dense_lexicon_c1", "seed": 1_218_223},
    ),
}

ARCHITECTURES = {
    "discovery": {
        "d4_w88": ModelConfig(4, 88, 4, 176, core.SEQUENCE_LENGTH, core.VOCAB_SIZE),
        "d6_w120": ModelConfig(6, 120, 4, 240, core.SEQUENCE_LENGTH, core.VOCAB_SIZE),
    },
    "confirmation": {
        "d5_w104": ModelConfig(5, 104, 4, 208, core.SEQUENCE_LENGTH, core.VOCAB_SIZE),
        "d7_w136": ModelConfig(7, 136, 4, 272, core.SEQUENCE_LENGTH, core.VOCAB_SIZE),
    },
}

TRAINING = dict(core.TRAINING)
TRAINING.update(
    {
        "maximum_steps": 2400,
        "evaluation_interval": ANCHOR_INTERVAL,
        "observation_steps": OBSERVATION_STEPS,
        "landmark_step": LANDMARK_STEP,
        "no_early_stopping": True,
    }
)

DATA_GATES = {
    "dense_prefix_eligible_per_split_min": 8,
    "dense_prefix_eligible_per_binary_level_min": 3,
    "dense_pre_rule_observations_min": 8,
    "landmark_pre_rule_systems_min": 12,
    "observed_onset_after_landmark_min": 8,
    "binary_class_per_split_min": 4,
    "threshold_status_stability_min": 0.75,
    "finite_fraction_min": 1.0,
    "zero_drift_max": 1.0e-6,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(pending, path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_digest(value: dict[str, Any], field: str) -> None:
    clean = dict(value)
    stored = clean.pop(field)
    if digest(clean) != stored:
        raise RuntimeError(f"digest mismatch for {field}")


def install_core_overrides() -> None:
    core.PHASE = PHASE
    core.SCRIPT = SCRIPT
    core.AUDIT_SCRIPT = AUDIT_SCRIPT
    core.OUT_ROOT = OUT_ROOT
    core.PROTOCOL_PATH = PROTOCOL_PATH
    core.PREAUDIT_PATH = PREAUDIT_PATH
    core.FINAL_PATH = FINAL_PATH
    core.TASKS = TASKS
    core.LEXICONS = LEXICONS
    core.ARCHITECTURES = ARCHITECTURES
    core.TRAINING = TRAINING


install_core_overrides()


def model_seed(split: str, task_index: int, lexicon_index: int, architecture_index: int, replicate: int) -> int:
    base = 1_218_300_000 if split == "discovery" else 1_218_700_000
    return base + task_index * 1_000_003 + lexicon_index * 100_003 + architecture_index * 10_007 + replicate * 1_009


def run_id(split: str, condition: dict[str, Any], architecture: str, replicate: int) -> str:
    return f"{split}__{condition['name']}__{condition['lexicon_name']}__{architecture}__s{replicate}"


def parameter_norm(model: torch.nn.Module) -> float:
    squared = sum(float(torch.sum(value.detach().float().square()).item()) for value in model.parameters())
    return float(np.sqrt(squared))


def source_gate() -> dict[str, bool]:
    final = read_json(SOURCE1217 / "analysis/final.json")
    audit = read_json(SOURCE1217 / "audit/independent_result_audit.json")
    validate_digest(final, "final_digest")
    validate_digest(audit, "audit_digest")
    return {
        "phase1217_final_frozen": final["final_digest"] == EXPECTED_1217_FINAL,
        "phase1217_audit_frozen": audit["audit_digest"] == EXPECTED_1217_AUDIT,
        "phase1217_audit_passed": audit["all_checks_passed"] is True,
        "phase1217_prefix_gate_failed": final["authorized_next"]["automatic_execution"] is False,
        "phase1217_dense_restart_required": (
            final["summaries"]["discovery"]["pre_behavior_prefix_eligible_count"] == 6
            and final["summaries"]["confirmation"]["pre_behavior_prefix_eligible_count"] == 4
        ),
    }


def script_hashes() -> dict[str, str]:
    return {
        "phase1218_main": sha256_file(SCRIPT),
        "phase1218_audit": sha256_file(AUDIT_SCRIPT),
        "phase1217_measurement_source": sha256_file(Path(core.__file__)),
        "phase1213_material_source": sha256_file(Path(core.p1213.__file__)),
        "tiny_transformer_source": sha256_file(Path(sys.modules[TinyCausalTransformer.__module__].__file__)),
    }


def protocol_payload() -> dict[str, Any]:
    return {
        "phase": PHASE,
        "created_at": utc_now(),
        "title": "Dense early pre-rule trajectory acquisition on new free micro-Transformers",
        "source_phase": 1217,
        "source_gate": source_gate(),
        "script_hashes": script_hashes(),
        "formal_run_count": 32,
        "runs_per_split": 16,
        "tasks": TASKS,
        "lexicons": LEXICONS,
        "architectures": {
            split: {name: asdict(config) for name, config in values.items()}
            for split, values in ARCHITECTURES.items()
        },
        "replicates": REPLICATES,
        "training": TRAINING,
        "observation_contract": {
            "dense_grid": OBSERVATION_STEPS,
            "anchor_grid": ANCHOR_STEPS,
            "clock_outcomes_use_anchor_grid_only": True,
            "dense_points_used_only_as_future_predictor_inputs": True,
            "fixed_landmark_step": LANDMARK_STEP,
            "landmark_rule": "future prediction may use only observations with step <= 50 and only systems with R false through step 50",
            "right_censoring": "R not observed by step 2400 is right censored, not permanent absence",
        },
        "threshold_profiles": core.THRESHOLD_PROFILES,
        "fixed_controls": core.FIXED_CONTROLS,
        "data_gates": DATA_GATES,
        "frozen_targets_for_separate_phase1219": {
            "formation_by_horizon": (800, 2400),
            "onset_interval": "primary R onset among systems forming strictly after the step-50 landmark",
            "prediction_unit": "system, never checkpoint",
            "baseline_families": (
                "accuracy",
                "loss",
                "confidence",
                "gradient_norm",
                "parameter_norm",
                "updates",
                "tokens",
                "parameter_token_proxy",
            ),
            "mechanistic_candidate_families": (
                "conditional_routing",
                "shared_differential_RDC",
                "single_joint_redundancy",
                "functional_quotient",
                "dynamic_response_trajectory",
                "local_formation_sensitivity",
            ),
        },
        "claims_allowed": (
            "dense early trajectory acquisition succeeded or failed under the frozen gate",
            "formation labels remain comparable to Phase 1217 because only anchor checkpoints define clocks",
            "which separately frozen prediction targets are identifiable from the acquired class balance",
        ),
        "forbidden": (
            "fitting or selecting a precursor predictor in Phase 1218",
            "changing the anchor clock labels after dense observations are seen",
            "treating checkpoints as independent prediction samples",
            "calling U1 or UJ a semantic clock",
            "claiming a universal formation order",
            "claiming pretrained-language or human-brain external validity",
            "posthoc threshold, landmark, horizon, task, architecture, or seed replacement",
        ),
    }


def preregister() -> dict[str, Any]:
    if PROTOCOL_PATH.exists():
        existing = read_json(PROTOCOL_PATH)
        validate_digest(existing, "protocol_digest")
        if existing["script_hashes"] != script_hashes():
            raise RuntimeError("frozen Phase 1218 protocol script hashes differ from current scripts")
        return existing
    payload = protocol_payload()
    if not all(payload["source_gate"].values()):
        raise RuntimeError(f"Phase 1217 source gate failed: {payload['source_gate']}")
    payload["protocol_digest"] = digest(payload)
    write_json(PROTOCOL_PATH, payload)
    return payload


def verify_protocol() -> dict[str, Any]:
    protocol = read_json(PROTOCOL_PATH)
    validate_digest(protocol, "protocol_digest")
    if protocol["script_hashes"] != script_hashes():
        raise RuntimeError("script hash drift after preregistration")
    if not PREAUDIT_PATH.exists():
        raise RuntimeError("independent preaudit missing")
    preaudit = read_json(PREAUDIT_PATH)
    validate_digest(preaudit, "audit_digest")
    if preaudit["all_checks_passed"] is not True or preaudit["protocol_digest"] != protocol["protocol_digest"]:
        raise RuntimeError("independent preaudit did not authorize execution")
    return protocol


def summarize_dense_trajectory(trajectory: list[dict[str, Any]], count: int) -> dict[str, Any]:
    anchors = [row for row in trajectory if int(row["step"]) in set(ANCHOR_STEPS)]
    if [int(row["step"]) for row in anchors] != list(ANCHOR_STEPS):
        raise RuntimeError("anchor trajectory is incomplete")
    result = core.summarize_trajectory(anchors, count)
    r_clock = result["primary_clocks"]["R"]
    dense_prefix_count = (
        sum(int(row["step"]) < int(r_clock["step"]) for row in trajectory)
        if r_clock["status"] == "observed"
        else sum(int(row["step"]) < TRAINING["maximum_steps"] for row in trajectory)
    )
    landmark_rows = [row for row in trajectory if int(row["step"]) <= LANDMARK_STEP]
    result["anchor_pre_behavior_prefix_count"] = int(result["pre_behavior_prefix_count"])
    result["dense_pre_behavior_prefix_count"] = int(dense_prefix_count)
    result["landmark_pre_rule"] = bool(all(not row["gates"]["primary"]["R"] for row in landmark_rows))
    result["landmark_observation_count"] = len(landmark_rows)
    return result


def execute_run(
    split: str,
    task_index: int,
    lexicon_index: int,
    architecture_index: int,
    replicate: int,
    device: torch.device,
) -> dict[str, Any]:
    protocol = verify_protocol()
    condition = core.make_condition(split, task_index, lexicon_index)
    architecture, config = list(ARCHITECTURES[split].items())[architecture_index]
    identifier = run_id(split, condition, architecture, replicate)
    run_root = OUT_ROOT / "runs" / split / identifier
    metrics_path = run_root / "metrics.json"
    if metrics_path.exists():
        existing = read_json(metrics_path)
        validate_digest(existing, "metrics_digest")
        if existing["protocol_digest"] != protocol["protocol_digest"]:
            raise RuntimeError(f"stale metrics for {identifier}")
        return existing

    seed = model_seed(split, task_index, lexicon_index, architecture_index, replicate)
    core.set_seed(seed)
    model = TinyCausalTransformer(config).to(device)
    count = core.parameter_count(model)
    train_combinations, _ = core.split_combinations(condition)
    train_inputs, train_targets, _ = core.build_examples(condition, train_combinations, range(len(core.TEMPLATES)))
    candidates = core.candidate_ids(condition, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(TRAINING["learning_rate"]),
        weight_decay=float(TRAINING["weight_decay"]),
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 37)
    initial_controls = core.initial_camera_controls(model, condition)
    trajectory: list[dict[str, Any]] = []
    checkpoint_manifest: list[dict[str, Any]] = []

    def record(step: int, loss: float | None, gradient_norm: float | None) -> None:
        checkpoint_path = run_root / "checkpoints" / f"step_{step:04d}.pt"
        core.write_checkpoint(
            checkpoint_path,
            core.checkpoint_payload(model, config, identifier, step, protocol["protocol_digest"]),
        )
        scan = core.scan_checkpoint(model, condition, initial_controls, step)
        scan["loss"] = loss
        scan["gradient_norm"] = gradient_norm
        scan["parameter_norm"] = parameter_norm(model)
        scan["updates"] = int(step)
        scan["parameter_token_proxy"] = int(scan["tokens_seen"] * count)
        trajectory.append(scan)
        checkpoint_manifest.append(
            {
                "step": int(step),
                "path": str(checkpoint_path.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha256_file(checkpoint_path),
            }
        )
        gates = scan["gates"]["primary"]
        print(
            f"[{utc_now()}] {identifier} step={step} "
            + " ".join(f"{clock}={int(gates[clock])}" for clock in CLOCKS)
            + f" holdout={scan['holdout_behavior']['accuracy']:.4f} pmin={scan['holdout_behavior']['minimum_probability']:.4f}",
            flush=True,
        )

    record(0, None, None)
    last_loss: float | None = None
    last_gradient: float | None = None
    for step in range(1, int(TRAINING["maximum_steps"]) + 1):
        model.train()
        indices = torch.randint(0, len(train_inputs), (int(TRAINING["batch_size"]),), generator=generator)
        batch_inputs = train_inputs[indices].to(device, non_blocking=True)
        batch_targets = train_targets[indices].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(batch_inputs)[:, -1].index_select(-1, candidates)
            loss = F.cross_entropy(logits.float(), batch_targets)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"nonfinite loss in {identifier} at step {step}")
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), float(TRAINING["gradient_clip_norm"]))
        if not bool(torch.isfinite(torch.as_tensor(gradient))):
            raise RuntimeError(f"nonfinite gradient in {identifier} at step {step}")
        optimizer.step()
        last_loss = float(loss.item())
        last_gradient = float(gradient)
        if step in OBSERVATION_STEP_SET:
            record(step, last_loss, last_gradient)

    metrics = {
        "phase": PHASE,
        "created_at": utc_now(),
        "protocol_digest": protocol["protocol_digest"],
        "run_id": identifier,
        "split": split,
        "task_index": task_index,
        "task_name": condition["name"],
        "source_roles": condition["source_roles"],
        "lexicon_index": lexicon_index,
        "lexicon_name": condition["lexicon_name"],
        "lexicon_seed": condition["lexicon_seed"],
        "architecture_index": architecture_index,
        "architecture": architecture,
        "config": asdict(config),
        "replicate": replicate,
        "seed": seed,
        "parameter_count": count,
        "initial_camera_controls": initial_controls,
        "trajectory": trajectory,
        "formation": summarize_dense_trajectory(trajectory, count),
        "checkpoint_manifest": checkpoint_manifest,
    }
    metrics["metrics_digest"] = digest(metrics)
    write_json(metrics_path, metrics)
    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return metrics


def load_rows(split: str) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((OUT_ROOT / "runs" / split).glob("*/metrics.json")):
        row = read_json(path)
        validate_digest(row, "metrics_digest")
        rows.append(row)
    return rows


def clock_status(row: dict[str, Any], clock: str) -> str:
    return str(row["formation"]["primary_clocks"][clock]["status"])


def clock_step(row: dict[str, Any], clock: str) -> int | None:
    value = row["formation"]["primary_clocks"][clock]
    return int(value["step"]) if value["status"] == "observed" else None


def matched_factor_effect(rows: list[dict[str, Any]], factor: str, clock: str) -> dict[str, Any]:
    factors = ("task_index", "lexicon_index", "architecture_index", "replicate")
    others = tuple(value for value in factors if value != factor)
    groups: dict[tuple[int, ...], dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        groups[tuple(int(row[value]) for value in others)][int(row[factor])] = row
    pairs = [group for group in groups.values() if set(group) == {0, 1}]
    differences: list[int] = []
    discordant = 0
    for pair in pairs:
        left, right = clock_step(pair[0], clock), clock_step(pair[1], clock)
        if (left is None) != (right is None):
            discordant += 1
        if left is not None and right is not None:
            differences.append(int(right - left))
    return {
        "matched_pair_count": len(pairs),
        "status_discordant_count": discordant,
        "status_discordant_fraction": float(discordant / len(pairs)) if pairs else 0.0,
        "both_observed_count": len(differences),
        "signed_step_differences_level1_minus_level0": differences,
        "median_signed_step_difference": float(np.median(differences)) if differences else None,
        "median_absolute_step_difference": float(np.median(np.abs(differences))) if differences else None,
    }


def group_summary(split: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    per_clock: dict[str, Any] = {}
    for clock in CLOCKS:
        observed_steps = [clock_step(row, clock) for row in rows if clock_status(row, clock) == "observed"]
        per_clock[clock] = {
            "observed": len(observed_steps),
            "right_censored": len(rows) - len(observed_steps),
            "steps": observed_steps,
            "median_step": float(np.median(observed_steps)) if observed_steps else None,
            "threshold_status_stability_mean": float(
                np.mean([row["formation"]["threshold_status_stability"][clock] for row in rows])
            ),
        }

    eligible = [
        row
        for row in rows
        if clock_status(row, "R") == "observed"
        and row["formation"]["dense_pre_behavior_prefix_count"] >= DATA_GATES["dense_pre_rule_observations_min"]
        and row["formation"]["landmark_pre_rule"]
    ]
    landmark = [row for row in rows if row["formation"]["landmark_pre_rule"]]
    onset_after_landmark = [
        row for row in landmark if clock_status(row, "R") == "observed" and int(clock_step(row, "R")) > LANDMARK_STEP
    ]
    per_level: dict[str, Any] = {}
    level_gate = True
    for factor in ("task_index", "lexicon_index", "architecture_index", "replicate"):
        per_level[factor] = {}
        for level in (0, 1):
            subset = [row for row in rows if int(row[factor]) == level]
            eligible_count = sum(row in eligible for row in subset)
            per_level[factor][str(level)] = {
                "run_count": len(subset),
                "R_observed": sum(clock_status(row, "R") == "observed" for row in subset),
                "dense_prefix_eligible": eligible_count,
            }
            level_gate = level_gate and eligible_count >= DATA_GATES["dense_prefix_eligible_per_binary_level_min"]

    target_counts: dict[str, Any] = {}
    for horizon in (800, 2400):
        positive = sum(clock_status(row, "R") == "observed" and int(clock_step(row, "R")) <= horizon for row in landmark)
        negative = len(landmark) - positive
        target_counts[f"formed_by_{horizon}"] = {
            "eligible": len(landmark),
            "positive": positive,
            "negative": negative,
            "balanced": bool(
                positive >= DATA_GATES["binary_class_per_split_min"]
                and negative >= DATA_GATES["binary_class_per_split_min"]
            ),
        }
    target_counts["primary_onset"] = {
        "eligible_observed": len(onset_after_landmark),
        "identifiable": len(onset_after_landmark) >= DATA_GATES["observed_onset_after_landmark_min"],
    }

    finite = all(
        all(
            point["train_behavior"]["finite_fraction"] >= DATA_GATES["finite_fraction_min"]
            and point["holdout_behavior"]["finite_fraction"] >= DATA_GATES["finite_fraction_min"]
            for point in row["trajectory"]
        )
        for row in rows
    )
    zero_drift = all(
        all(point["necessity"]["zero_drift_max"] <= DATA_GATES["zero_drift_max"] for point in row["trajectory"])
        for row in rows
    )
    threshold_stable = per_clock["R"]["threshold_status_stability_mean"] >= DATA_GATES["threshold_status_stability_min"]
    gates = {
        "run_count": len(rows) == 16,
        "dense_prefix_breadth": len(eligible) >= DATA_GATES["dense_prefix_eligible_per_split_min"],
        "factor_level_breadth": level_gate,
        "landmark_pre_rule_breadth": len(landmark) >= DATA_GATES["landmark_pre_rule_systems_min"],
        "onset_target_breadth": len(onset_after_landmark) >= DATA_GATES["observed_onset_after_landmark_min"],
        "threshold_status_stability": threshold_stable,
        "all_finite": finite,
        "zero_drift": zero_drift,
    }
    return {
        "split": split,
        "run_count": len(rows),
        "observation_count_per_run": len(OBSERVATION_STEPS),
        "per_clock": per_clock,
        "per_level": per_level,
        "factor_effects": {
            factor: {clock: matched_factor_effect(rows, factor, clock) for clock in CLOCKS}
            for factor in ("task_index", "lexicon_index", "architecture_index", "replicate")
        },
        "primary_signature_counts": dict(
            sorted(
                {
                    signature: sum(row["formation"]["primary_signature"] == signature for row in rows)
                    for signature in {row["formation"]["primary_signature"] for row in rows}
                }.items()
            )
        ),
        "dense_prefix_eligible_count": len(eligible),
        "landmark_pre_rule_count": len(landmark),
        "observed_onset_after_landmark_count": len(onset_after_landmark),
        "prediction_target_counts": target_counts,
        "gates": gates,
        "dense_acquisition_gate": all(gates.values()),
    }


def finalize() -> dict[str, Any]:
    protocol = verify_protocol()
    rows = {split: load_rows(split) for split in ("discovery", "confirmation")}
    if any(len(values) != 16 for values in rows.values()):
        raise RuntimeError(f"incomplete runs: { {split: len(values) for split, values in rows.items()} }")
    summaries = {split: group_summary(split, values) for split, values in rows.items()}
    acquisition_passed = all(summary["dense_acquisition_gate"] for summary in summaries.values())
    common_targets = []
    for target in ("formed_by_800", "formed_by_2400"):
        if all(summaries[split]["prediction_target_counts"][target]["balanced"] for split in summaries):
            common_targets.append(target)
    if all(summaries[split]["prediction_target_counts"]["primary_onset"]["identifiable"] for split in summaries):
        common_targets.append("primary_onset")
    predictor_authorized = bool(acquisition_passed and common_targets)

    manifest = []
    for split, values in rows.items():
        for row in values:
            path = OUT_ROOT / "runs" / split / row["run_id"] / "metrics.json"
            manifest.append(
                {
                    "split": split,
                    "run_id": row["run_id"],
                    "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                    "sha256": sha256_file(path),
                    "metrics_digest": row["metrics_digest"],
                }
            )
    result = {
        "phase": PHASE,
        "created_at": utc_now(),
        "status": "dense_prerule_acquisition_passed" if acquisition_passed else "dense_prerule_acquisition_not_confirmed",
        "protocol_digest": protocol["protocol_digest"],
        "summaries": summaries,
        "run_manifest": manifest,
        "claims": {
            "dense_early_data_geometry_confirmed": acquisition_passed,
            "clock_outcomes_recomputed_on_anchor_grid_only": True,
            "precursor_predictor_fitted": False,
            "universal_clock_order": "not_claimed",
            "semantic_mechanism": "not_tested",
            "pretrained_language_external_validity": "not_tested",
        },
        "authorized_next": {
            "experiment": "PHASE1219_FROZEN_LANDMARK_PRECURSOR_PREDICTION" if predictor_authorized else None,
            "automatic_execution": predictor_authorized,
            "common_identifiable_targets": common_targets,
            "scope": "separate preregistration; system-level prediction using only step <= 50 inputs",
            "reason": (
                "both splits passed dense acquisition and share at least one identifiable target"
                if predictor_authorized
                else "dense acquisition or common target-identifiability gate failed"
            ),
            "pretrained_model_run": False,
        },
        "k_item": {
            "identifier": "K195",
            "evidence_grade": "E3-METHOD" if acquisition_passed else "E3-NEGATIVE-BOUNDARY",
            "statement": (
                "A preregistered dense early grid produced adequate fixed-landmark pre-rule trajectories in both independent free-Transformer splits without changing anchor-defined clock outcomes."
                if acquisition_passed
                else "The preregistered dense early grid did not produce adequate fixed-landmark pre-rule trajectory breadth in both free-Transformer splits."
            ),
            "scope": "32 new finite micro-Transformer systems; data geometry, not a mechanism claim",
        },
        "new_mathematics_required": False,
    }
    result["final_digest"] = digest(result)
    write_json(FINAL_PATH, result)
    return result


def smoke() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    condition = core.make_condition("discovery", 0, 0)
    config = ARCHITECTURES["discovery"]["d4_w88"]
    core.set_seed(1_218_999)
    model = TinyCausalTransformer(config).cuda()
    controls = core.initial_camera_controls(model, condition)
    scan = core.scan_checkpoint(model, condition, controls, 0)
    result = {
        "cuda_device": torch.cuda.get_device_name(0),
        "observation_count": len(OBSERVATION_STEPS),
        "anchor_count": len(ANCHOR_STEPS),
        "landmark_observation_count": sum(step <= LANDMARK_STEP for step in OBSERVATION_STEPS),
        "grid_contains_all_anchors": set(ANCHOR_STEPS).issubset(OBSERVATION_STEP_SET),
        "layer_count": len(scan["layers"]),
        "all_gates_present": all(clock in scan["gates"]["primary"] for clock in CLOCKS),
        "zero_drift": scan["necessity"]["zero_drift_max"],
    }
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run_split(split: str) -> list[dict[str, Any]]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda:0")
    rows = []
    for task_index, lexicon_index, architecture_index, replicate in itertools.product(range(2), repeat=4):
        rows.append(execute_run(split, task_index, lexicon_index, architecture_index, replicate, device))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("smoke", "preregister", "run", "finalize"), required=True)
    parser.add_argument("--split", choices=("discovery", "confirmation"))
    args = parser.parse_args()
    if args.stage == "smoke":
        print(json.dumps(smoke(), indent=2))
    elif args.stage == "preregister":
        print(json.dumps(preregister(), indent=2))
    elif args.stage == "run":
        if args.split is None:
            raise SystemExit("--split is required for run")
        print(json.dumps({"split": args.split, "run_count": len(run_split(args.split))}, indent=2))
    else:
        print(json.dumps(finalize(), indent=2))


if __name__ == "__main__":
    main()
