#!/usr/bin/env python3
"""Cross-task confirmation of low-order intervention-response predictability.

The intervention camera is frozen to Phase1165's non-overwriting matched
residual-difference addition.  Discovery networks select one estimator from
each predeclared order library; selections are sealed before independent
confirmation-network outcomes are generated.  A separate compositional
holdout task is behavior-gated and is never scanned when it cannot generalize.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve()
AUDIT_SCRIPT = ROOT / "tests/glm5/phase1166_cross_task_predictive_order_confirmation_audit.py"
P1165_SCRIPT = ROOT / "tests/glm5/phase1165_intervention_semantics_coverage_falsification.py"
P1165_AUDIT = ROOT / "tests/glm5/phase1165_intervention_semantics_coverage_falsification_audit.py"
OUT_ROOT = ROOT / "tests/glm5/result/phase1166_cross_task_predictive_order_confirmation"
sys.path.insert(0, str(ROOT / "tests/glm5"))

import phase1165_intervention_semantics_coverage_falsification as p1165  # noqa: E402


p1164 = p1165.p1164
p1163 = p1165.p1163
p1161 = p1165.p1161
source = p1165.source
PHASE = 1166
FULL_TASKS = ("identity_decode", "modular_sum", "balanced_interaction")
COMPOSITION_TASK = "modular_composition_holdout"
ALL_TASKS = FULL_TASKS + (COMPOSITION_TASK,)
FACTORS = source.FACTORS
ROLES = source.ROLES
ARCHITECTURES = source.ARCHITECTURES
REPLICATES = 4
DISCOVERY_REPLICATES = (0, 1)
CONFIRMATION_REPLICATES = (2, 3)
ORDER_LIBRARY = {0: ("zero",), 1: ("main", "max_single"), 2: ("pairwise", "max_pair")}
ALGORITHMS = tuple(name for order in (0, 1, 2) for name in ORDER_LIBRARY[order])
TRAINING = {
    "max_steps": 3000,
    "minimum_steps": 400,
    "evaluation_interval": 100,
    "required_consecutive_train_passes": 3,
    "batch_size": 128,
    "learning_rate": 0.0005,
    "weight_decay": 0.001,
    "gradient_clip_norm": 1.0,
}
THRESHOLDS = {
    "behavior_train_accuracy_min": 1.0,
    "behavior_eval_accuracy_min": 1.0,
    "composition_eval_accuracy_min": 0.90,
    "finite_fraction_min": 1.0,
    "task_total_qualified_min": 6,
    "task_split_qualified_min": 3,
    "task_architecture_qualified_min": 3,
    "denominator_min": 1e-5,
    "null_abs_max": 1e-8,
    "transport_unit_q95_min": 0.05,
    "transport_unit_fraction_min": 0.75,
    "transport_median_q95_min": 0.10,
    "risk_median_unit_mae_max": 0.03,
    "risk_median_relative_mae_max": 0.15,
    "risk_unit_mae_max": 0.05,
    "risk_unit_relative_mae_max": 0.25,
    "risk_unit_fraction_min": 0.75,
    "risk_schedule_abs_error_q95_max": 0.10,
}
SCHEDULE_SEED = 1166007
SCHEDULES_PER_CARDINALITY = 128
SURFACE_CHUNK_SIZE = 32


def output_classes(task: str) -> int:
    return 32 if task == "identity_decode" else 8


def target_index(task: str, row: int, col: int, context: int) -> int:
    row, col, context = int(row), int(col), int(context)
    if task == "identity_decode":
        return 16 * context + 4 * row + col
    if task in ("modular_sum", COMPOSITION_TASK):
        return (row + col) % 4 + 4 * context
    if task == "balanced_interaction":
        low = (row + col + 2 * (row % 2) * (col % 2) + context * (row + 1)) % 4
        high = (context + (row % 2) * (col % 2)) % 2
        return low + 4 * high
    raise KeyError(task)


def composition_holdout(row: int, col: int, context: int) -> bool:
    del context
    return int(row) in (0, 2) and int(col) in (0, 2)


def model_seed(task: str, architecture: str, replicate: int) -> int:
    return (
        1166100
        + ALL_TASKS.index(task) * 10000
        + list(ARCHITECTURES).index(architecture) * 1009
        + int(replicate) * 107
    )


def model_id(task: str, seed: int) -> str:
    return p1163.digest({"phase": PHASE, "task": task, "seed": seed})[:16]


def make_lexicon(seed: int) -> dict[str, Any]:
    return source.make_lexicon(seed)


def encode(
    row: int, col: int, context: int, template_index: int, lexicon: dict[str, Any]
) -> tuple[list[int], dict[str, int]]:
    return source.encode(row, col, context, template_index, lexicon)


def task_examples(
    task: str, lexicon: dict[str, Any]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    train_inputs, train_targets, eval_inputs, eval_targets = [], [], [], []
    for template_index in range(len(source.TEMPLATES)):
        for context in range(source.CONTEXTS):
            for row in range(source.ROWS):
                for col in range(source.COLS):
                    tokens, _ = encode(row, col, context, template_index, lexicon)
                    target = target_index(task, row, col, context)
                    held_out = task == COMPOSITION_TASK and composition_holdout(row, col, context)
                    if held_out:
                        eval_inputs.append(tokens)
                        eval_targets.append(target)
                    else:
                        train_inputs.append(tokens)
                        train_targets.append(target)
                    if task != COMPOSITION_TASK:
                        eval_inputs.append(tokens)
                        eval_targets.append(target)
    return (
        torch.tensor(train_inputs, dtype=torch.long),
        torch.tensor(train_targets, dtype=torch.long),
        torch.tensor(eval_inputs, dtype=torch.long),
        torch.tensor(eval_targets, dtype=torch.long),
    )


def answer_ids(task: str, lexicon: dict[str, Any], device: torch.device) -> torch.Tensor:
    return torch.tensor(
        lexicon["answer"][: output_classes(task)], dtype=torch.long, device=device
    )


def evaluate(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    task: str,
    lexicon: dict[str, Any],
) -> dict[str, Any]:
    device = next(model.parameters()).device
    model.eval()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        raw = model(inputs.to(device))
    logits = raw[:, -1].float().index_select(-1, answer_ids(task, lexicon, device))
    probabilities = torch.softmax(logits, dim=-1)
    target_probability = probabilities.gather(1, targets.to(device)[:, None]).squeeze(1)
    return {
        "case_count": int(len(targets)),
        "accuracy": float((logits.argmax(-1).cpu() == targets).float().mean().item()),
        "minimum_probability": float(target_probability.min().item()),
        "mean_probability": float(target_probability.mean().item()),
        "finite_fraction": float(torch.isfinite(logits).float().mean().item()),
    }


def train_model(
    task: str,
    config: Any,
    seed: int,
    lexicon: dict[str, Any],
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    source.set_seed(seed)
    model = source.TinyCausalTransformer(config).to(device)
    train_inputs, train_targets, eval_inputs, eval_targets = task_examples(task, lexicon)
    candidates = answer_ids(task, lexicon, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=TRAINING["learning_rate"],
        weight_decay=TRAINING["weight_decay"],
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 31)
    consecutive = 0
    logs = []
    final_step = 0
    for step in range(1, TRAINING["max_steps"] + 1):
        model.train()
        indices = torch.randint(
            0, len(train_inputs), (TRAINING["batch_size"],), generator=generator
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(train_inputs[indices].to(device))[:, -1].index_select(-1, candidates)
            loss = F.cross_entropy(logits.float(), train_targets[indices].to(device))
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"nonfinite loss: {task}/{seed}/{step}")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), TRAINING["gradient_clip_norm"]
        )
        if not bool(torch.isfinite(torch.as_tensor(gradient_norm))):
            raise RuntimeError(f"nonfinite gradient: {task}/{seed}/{step}")
        optimizer.step()
        final_step = step
        if step % TRAINING["evaluation_interval"] == 0:
            train_metrics = evaluate(model, train_inputs, train_targets, task, lexicon)
            eval_metrics = evaluate(model, eval_inputs, eval_targets, task, lexicon)
            train_pass = (
                train_metrics["accuracy"] >= THRESHOLDS["behavior_train_accuracy_min"]
                and train_metrics["finite_fraction"] == 1.0
            )
            consecutive = consecutive + 1 if train_pass else 0
            logs.append(
                {
                    "step": step,
                    "loss": float(loss.item()),
                    "gradient_norm": float(gradient_norm),
                    "train_accuracy": train_metrics["accuracy"],
                    "eval_accuracy": eval_metrics["accuracy"],
                    "train_minimum_probability": train_metrics["minimum_probability"],
                    "eval_minimum_probability": eval_metrics["minimum_probability"],
                }
            )
            if (
                step >= TRAINING["minimum_steps"]
                and consecutive >= TRAINING["required_consecutive_train_passes"]
            ):
                break
    train_metrics = evaluate(model, train_inputs, train_targets, task, lexicon)
    eval_metrics = evaluate(model, eval_inputs, eval_targets, task, lexicon)
    qualified = bool(
        train_metrics["accuracy"] >= THRESHOLDS["behavior_train_accuracy_min"]
        and eval_metrics["accuracy"] >= THRESHOLDS["behavior_eval_accuracy_min"]
        and train_metrics["finite_fraction"] == 1.0
        and eval_metrics["finite_fraction"] == 1.0
        and consecutive >= TRAINING["required_consecutive_train_passes"]
    )
    composition_generalized = bool(
        task == COMPOSITION_TASK
        and train_metrics["accuracy"] >= THRESHOLDS["behavior_train_accuracy_min"]
        and eval_metrics["accuracy"] >= THRESHOLDS["composition_eval_accuracy_min"]
        and train_metrics["finite_fraction"] == 1.0
        and eval_metrics["finite_fraction"] == 1.0
    )
    return model, {
        "task": task,
        "steps": final_step,
        "consecutive_train_passes": consecutive,
        "qualified": qualified,
        "composition_generalized": composition_generalized,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "train": train_metrics,
        "eval": eval_metrics,
        "logs": logs,
    }


def split_name(replicate: int) -> str:
    return "discovery" if replicate in DISCOVERY_REPLICATES else "confirmation"


def task_authorization(rows: list[dict[str, Any]], task: str) -> dict[str, Any]:
    key = "composition_generalized" if task == COMPOSITION_TASK else "qualified"
    selected = [row for row in rows if row["task"] == task]
    passed = [row for row in selected if row[key]]
    split_counts = {
        split: sum(row[key] and row["split"] == split for row in selected)
        for split in ("discovery", "confirmation")
    }
    architecture_counts = {
        architecture: sum(row[key] and row["architecture"] == architecture for row in selected)
        for architecture in ARCHITECTURES
    }
    checks = {
        "total": len(passed) >= THRESHOLDS["task_total_qualified_min"],
        "splits": all(
            value >= THRESHOLDS["task_split_qualified_min"] for value in split_counts.values()
        ),
        "architectures": all(
            value >= THRESHOLDS["task_architecture_qualified_min"]
            for value in architecture_counts.values()
        ),
    }
    return {
        "authorized": all(checks.values()),
        "passed_count": len(passed),
        "total_count": len(selected),
        "split_counts": split_counts,
        "architecture_counts": architecture_counts,
        "checks": checks,
    }


def changed_values(row: int, col: int, context: int, factor: str) -> tuple[int, int, int]:
    values = {"row": int(row), "col": int(col), "context": int(context)}
    modulus = {"row": source.ROWS, "col": source.COLS, "context": source.CONTEXTS}[factor]
    values[factor] = (values[factor] + 1) % modulus
    return values["row"], values["col"], values["context"]


def scan_batch(
    task: str, lexicon: dict[str, Any], factor: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    receivers, donors, receiver_targets, donor_targets, positions = [], [], [], [], []
    for template_index in (3, 4, 5):
        for context in range(source.CONTEXTS):
            for row in range(source.ROWS):
                for col in range(source.COLS):
                    donor_values = changed_values(row, col, context, factor)
                    receiver_tokens, receiver_positions = encode(
                        row, col, context, template_index, lexicon
                    )
                    donor_tokens, _ = encode(*donor_values, template_index, lexicon)
                    receivers.append(receiver_tokens)
                    donors.append(donor_tokens)
                    receiver_targets.append(target_index(task, row, col, context))
                    donor_targets.append(target_index(task, *donor_values))
                    positions.append([receiver_positions[role] for role in ROLES])
    if any(a == b for a, b in zip(receiver_targets, donor_targets, strict=True)):
        raise RuntimeError(f"target-preserving donor in {task}/{factor}")
    return (
        torch.tensor(receivers, dtype=torch.long),
        torch.tensor(donors, dtype=torch.long),
        torch.tensor(receiver_targets, dtype=torch.long),
        torch.tensor(donor_targets, dtype=torch.long),
        torch.tensor(positions, dtype=torch.long),
    )


def candidate_logits(
    raw: torch.Tensor, task: str, lexicon: dict[str, Any], device: torch.device
) -> torch.Tensor:
    return raw[:, -1].float().index_select(-1, answer_ids(task, lexicon, device))


def target_margin(
    logits: torch.Tensor, donor_targets: torch.Tensor, receiver_targets: torch.Tensor
) -> torch.Tensor:
    return logits.gather(1, donor_targets[:, None]).squeeze(1) - logits.gather(
        1, receiver_targets[:, None]
    ).squeeze(1)


def delta_surface(
    model: torch.nn.Module,
    config: Any,
    lexicon: dict[str, Any],
    task: str,
    factor: str,
    subsets: list[tuple[int, ...]],
) -> tuple[np.ndarray, dict[str, Any]]:
    device = next(model.parameters()).device
    receiver_cpu, donor_cpu, receiver_target_cpu, donor_target_cpu, positions_cpu = scan_batch(
        task, lexicon, factor
    )
    receiver, donor = receiver_cpu.to(device), donor_cpu.to(device)
    receiver_targets, donor_targets = receiver_target_cpu.to(device), donor_target_cpu.to(device)
    positions = positions_cpu.to(device)
    role_positions = {role: positions[:, ROLES.index(role)] for role in ROLES}
    actual_by_depth = {
        depth: source.actual_depth_index(config, depth) for depth in p1161.INTERIOR_DEPTHS
    }
    model.eval()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        receiver_raw, receiver_states, _, _ = p1165.clean_trace(model, receiver)
        donor_raw, donor_states, _, _ = p1165.clean_trace(model, donor)
    receiver_logits = candidate_logits(receiver_raw, task, lexicon, device)
    donor_logits = candidate_logits(donor_raw, task, lexicon, device)
    base_margin = target_margin(receiver_logits, donor_targets, receiver_targets)
    donor_margin = target_margin(donor_logits, donor_targets, receiver_targets)
    denominator = donor_margin - base_margin
    denominator_min = float(denominator.min().item())
    if denominator_min <= THRESHOLDS["denominator_min"]:
        raise RuntimeError(f"nonpositive denominator: {task}/{factor}/{denominator_min}")
    batch_size = len(receiver)
    rows = []
    for start in range(0, len(subsets), SURFACE_CHUNK_SIZE):
        chunk = subsets[start : start + SURFACE_CHUNK_SIZE]
        schedule_count = len(chunk)
        hidden = model.embed(receiver).repeat(schedule_count, 1, 1)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for layer_index, block in enumerate(model.blocks, start=1):
                hidden = block(hidden)
                hidden = p1165.patch_selected_sites(
                    hidden,
                    donor_states[layer_index],
                    receiver_states[layer_index],
                    chunk,
                    layer_index,
                    actual_by_depth,
                    role_positions,
                    additive=True,
                )
            patched_raw = model.lm_head(model.final_norm(hidden))
        patched_logits = candidate_logits(patched_raw, task, lexicon, device)
        patched_margin = target_margin(
            patched_logits,
            donor_targets.repeat(schedule_count),
            receiver_targets.repeat(schedule_count),
        )
        effect = (patched_margin - base_margin.repeat(schedule_count)) / denominator.repeat(
            schedule_count
        )
        rows.extend(effect.float().reshape(schedule_count, batch_size).cpu().numpy())
    matrix = np.stack(rows, axis=0).astype(np.float32)
    return matrix, {
        "case_count": batch_size,
        "denominator_min": denominator_min,
        "denominator_median": float(torch.median(denominator).item()),
        "finite_fraction": float(np.isfinite(matrix).mean()),
    }


def schedule_splits() -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
    rng = np.random.default_rng(SCHEDULE_SEED)
    discovery, confirmation = [], []
    for cardinality in (3, 4, 5):
        population = list(itertools.combinations(range(len(p1165.sites())), cardinality))
        chosen = rng.choice(
            len(population), size=2 * SCHEDULES_PER_CARDINALITY, replace=False
        ).tolist()
        discovery.extend(population[int(index)] for index in sorted(chosen[:SCHEDULES_PER_CARDINALITY]))
        confirmation.extend(
            population[int(index)] for index in sorted(chosen[SCHEDULES_PER_CARDINALITY:])
        )
    return discovery, confirmation


def calibration_subsets() -> list[tuple[int, ...]]:
    return p1165.calibration_subsets()


def checkpoint_payload(
    model: torch.nn.Module, config: Any, lexicon: dict[str, Any]
) -> dict[str, Any]:
    return {
        "config": asdict(config),
        "lexicon": lexicon,
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
    }


def load_checkpoint(
    path: Path, device: torch.device
) -> tuple[torch.nn.Module, Any, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = source.ModelConfig(**payload["config"])
    model = source.TinyCausalTransformer(config).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, config, payload["lexicon"]


def structural_task_checks() -> dict[str, bool]:
    checks = {}
    for task in FULL_TASKS:
        outputs = [
            target_index(task, row, col, context)
            for context in range(2)
            for row in range(4)
            for col in range(4)
        ]
        checks[f"{task}_range"] = min(outputs) == 0 and max(outputs) < output_classes(task)
        checks[f"{task}_all_classes"] = set(outputs) == set(range(output_classes(task)))
        checks[f"{task}_factor_changes_target"] = all(
            target_index(task, row, col, context)
            != target_index(task, *changed_values(row, col, context, factor))
            for factor in FACTORS
            for context in range(2)
            for row in range(4)
            for col in range(4)
        )
    held = [
        (row, col, context)
        for context in range(2)
        for row in range(4)
        for col in range(4)
        if composition_holdout(row, col, context)
    ]
    checks["composition_holdout_count"] = len(held) == 8
    checks["composition_train_count"] = 32 - len(held) == 24
    train_cells = [
        (row, col, context)
        for context in range(2)
        for row in range(4)
        for col in range(4)
        if not composition_holdout(row, col, context)
    ]
    checks["composition_all_values_seen_in_train"] = (
        {row for row, _, _ in train_cells} == set(range(4))
        and {col for _, col, _ in train_cells} == set(range(4))
        and {context for _, _, context in train_cells} == set(range(2))
    )
    checks["composition_outputs_seen_in_train"] = all(
        any(
            not composition_holdout(r, c, k)
            and target_index(COMPOSITION_TASK, r, c, k) == output
            for k in range(2)
            for r in range(4)
            for c in range(4)
        )
        for output in range(output_classes(COMPOSITION_TASK))
    )
    return checks


def prior_artifacts() -> dict[str, Any]:
    return {
        "final": p1163.read_json(p1165.OUT_ROOT / "analysis/final.json"),
        "audit": p1163.read_json(p1165.OUT_ROOT / "audit/independent_audit.json"),
    }


def protocol_command() -> None:
    if OUT_ROOT.exists():
        raise RuntimeError("refusing to overwrite Phase1166 artifacts")
    prior = prior_artifacts()
    discovery, confirmation = schedule_splits()
    task_checks = structural_task_checks()
    checks = {
        "phase1165_audit_passed": prior["audit"]["all_checks_passed"],
        "phase1165_semantic_branch_closed": prior["final"]["branch_status"]
        == "closed_after_one_shot_semantic_comparison",
        "camera_frozen_to_delta_add": True,
        "full_task_count": len(FULL_TASKS) == 3,
        "composition_separate_behavior_axis": True,
        "order_library_frozen": ORDER_LIBRARY
        == {0: ("zero",), 1: ("main", "max_single"), 2: ("pairwise", "max_pair")},
        "calibration_count": len(calibration_subsets()) == 121,
        "discovery_schedule_count": len(discovery) == 3 * SCHEDULES_PER_CARDINALITY,
        "confirmation_schedule_count": len(confirmation) == 3 * SCHEDULES_PER_CARDINALITY,
        "schedule_splits_disjoint": not bool(set(discovery).intersection(confirmation)),
        "task_structures": all(task_checks.values()),
        "selection_before_confirmation": True,
        "primary_script_exists": SCRIPT.exists(),
        "audit_script_exists": AUDIT_SCRIPT.exists(),
    }
    if not all(checks.values()):
        raise RuntimeError(f"protocol checks failed: {checks}; task={task_checks}")
    protocol = {
        "phase": PHASE,
        "created_at_utc": p1163.now(),
        "title": "cross-task predictive-order confirmation with separate composition behavior gate",
        "source_digests": {
            "phase1165_final": prior["final"]["final_digest"],
            "phase1165_audit": prior["audit"]["audit_digest"],
        },
        "source_hashes": {
            "primary_script": p1163.sha256_file(SCRIPT),
            "audit_script": p1163.sha256_file(AUDIT_SCRIPT),
            "phase1165_script": p1163.sha256_file(P1165_SCRIPT),
            "phase1165_audit": p1163.sha256_file(P1165_AUDIT),
        },
        "full_tasks": list(FULL_TASKS),
        "composition_task": COMPOSITION_TASK,
        "task_formulas": {
            "identity_decode": "16*context + 4*row + col",
            "modular_sum": "((row + col) mod 4) + 4*context",
            "balanced_interaction": "low=(row+col+2*(row mod2)*(col mod2)+context*(row+1)) mod4; high=(context+(row mod2)*(col mod2)) mod2; y=low+4*high",
            COMPOSITION_TASK: "modular_sum with the frozen row in {0,2}, col in {0,2} subcube absent from training",
        },
        "task_structure_checks": task_checks,
        "architectures": {name: asdict(config) for name, config in ARCHITECTURES.items()},
        "replicates": REPLICATES,
        "discovery_replicates": list(DISCOVERY_REPLICATES),
        "confirmation_replicates": list(CONFIRMATION_REPLICATES),
        "training": TRAINING,
        "behavior_gate_note": "accuracy and finiteness authorize response measurement; confidence is reported but is not a hard stop",
        "intervention_semantic": "matched residual-delta addition after each selected block-token site",
        "calibration_subsets": [list(row) for row in calibration_subsets()],
        "discovery_schedules": [list(row) for row in discovery],
        "confirmation_schedules": [list(row) for row in confirmation],
        "order_library": {str(key): list(value) for key, value in ORDER_LIBRARY.items()},
        "selection_rule": "within each task and order, minimize discovery median relative MAE, then median absolute MAE, then algorithm name",
        "operational_order_definition": "smallest order in the frozen finite estimator library passing absolute and relative confirmation risk gates; this is an upper bound conditional on epsilon, schedule distribution, site registry, camera, task, model panel, and estimator library",
        "thresholds": THRESHOLDS,
        "primary_endpoint": "all three authorized full-supervision tasks have independently confirmed operational predictive order at most two",
        "composition_endpoint": "held-out input combinations generalize behaviorally in a separately authorized panel; no hidden scan is allowed on failure",
        "hard_stops": [
            "The intervention camera cannot change in this phase.",
            "Estimator selection uses discovery networks only and is sealed before confirmation outcomes.",
            "Confidence is descriptive unless denominator or finiteness fails.",
            "Composition failure remains a behavior boundary and cannot be repaired by shrinking or changing the holdout.",
            "Success establishes only an operational upper bound in a frozen estimator library, not a natural intrinsic causal order.",
            "No new estimator, threshold, schedule resampling, or component search is allowed after protocol freeze.",
        ],
        "checks": checks,
    }
    protocol["protocol_digest"] = p1163.digest(protocol)
    p1163.write_json(OUT_ROOT / "protocol/preregistration.json", protocol)
    p1163.write_json(
        OUT_ROOT / "protocol/audit.json",
        {
            "checks": checks,
            "task_structure_checks": task_checks,
            "all_checks_passed": all(checks.values()) and all(task_checks.values()),
            "protocol_digest": protocol["protocol_digest"],
        },
    )
    print(p1163.canonical({"protocol_digest": protocol["protocol_digest"], "checks": checks}))


def verify_protocol() -> dict[str, Any]:
    protocol = p1163.read_json(OUT_ROOT / "protocol/preregistration.json")
    body = dict(protocol)
    stored = body.pop("protocol_digest")
    if p1163.digest(body) != stored:
        raise RuntimeError("protocol digest mismatch")
    for key, path in (
        ("primary_script", SCRIPT),
        ("audit_script", AUDIT_SCRIPT),
        ("phase1165_script", P1165_SCRIPT),
        ("phase1165_audit", P1165_AUDIT),
    ):
        if p1163.sha256_file(path) != protocol["source_hashes"][key]:
            raise RuntimeError(f"frozen source changed: {key}")
    discovery, confirmation = schedule_splits()
    if protocol["discovery_schedules"] != [list(row) for row in discovery]:
        raise RuntimeError("discovery schedule drift")
    if protocol["confirmation_schedules"] != [list(row) for row in confirmation]:
        raise RuntimeError("confirmation schedule drift")
    return protocol


def run_models_command() -> None:
    protocol = verify_protocol()
    root = OUT_ROOT / "runs/models"
    if root.exists():
        raise RuntimeError("refusing to overwrite model run")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    rows = []
    for task in ALL_TASKS:
        for architecture, config in ARCHITECTURES.items():
            for replicate in range(REPLICATES):
                seed = model_seed(task, architecture, replicate)
                identifier = model_id(task, seed)
                lexicon = make_lexicon(seed + 18017)
                model, metrics = train_model(task, config, seed, lexicon, device)
                checkpoint = root / task / "checkpoints" / f"{identifier}.pt"
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                torch.save(checkpoint_payload(model, config, lexicon), checkpoint)
                rows.append(
                    {
                        "model_id": identifier,
                        "task": task,
                        "architecture": architecture,
                        "replicate": replicate,
                        "split": split_name(replicate),
                        "seed": seed,
                        "lexicon_digest": p1163.digest(lexicon),
                        "checkpoint_sha256": p1163.sha256_file(checkpoint),
                        **metrics,
                    }
                )
                del model
                torch.cuda.empty_cache()
    authorizations = {task: task_authorization(rows, task) for task in ALL_TASKS}
    checks = {
        "model_count": len(rows) == len(ALL_TASKS) * len(ARCHITECTURES) * REPLICATES,
        "finite": all(
            row["train"]["finite_fraction"] == 1.0 and row["eval"]["finite_fraction"] == 1.0
            for row in rows
        ),
        "all_full_tasks_authorized": all(authorizations[task]["authorized"] for task in FULL_TASKS),
        "confidence_not_used_as_gate": True,
    }
    root.mkdir(parents=True, exist_ok=True)
    p1163.write_jsonl(root / "model_metrics.jsonl", rows)
    summary = {
        "phase": PHASE,
        "created_at_utc": p1163.now(),
        "protocol_digest": protocol["protocol_digest"],
        "authorizations": authorizations,
        "minimum_train_probability": min(row["train"]["minimum_probability"] for row in rows),
        "minimum_eval_probability": min(row["eval"]["minimum_probability"] for row in rows),
        "checks": checks,
        "full_task_behavior_gate_passed": all(checks.values()),
        "composition_generalization_authorized": authorizations[COMPOSITION_TASK]["authorized"],
        "metrics_sha256": p1163.sha256_file(root / "model_metrics.jsonl"),
    }
    summary["summary_digest"] = p1163.digest(summary)
    p1163.write_json(root / "summary.json", summary)
    print(
        p1163.canonical(
            {
                "checks": checks,
                "authorizations": authorizations,
                "summary_digest": summary["summary_digest"],
            }
        )
    )


def qualified_rows(task: str, split: str | None = None) -> list[dict[str, Any]]:
    rows = p1163.read_jsonl(OUT_ROOT / "runs/models/model_metrics.jsonl")
    result = [row for row in rows if row["task"] == task and row["qualified"]]
    if split is not None:
        result = [row for row in result if row["split"] == split]
    return result


def collect_task_responses(
    task: str, rows: list[dict[str, Any]], subsets: list[tuple[int, ...]], device: torch.device
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    model_arrays, diagnostics = [], []
    for row in rows:
        checkpoint = OUT_ROOT / "runs/models" / task / "checkpoints" / f"{row['model_id']}.pt"
        model, config, lexicon = load_checkpoint(checkpoint, device)
        factor_arrays, factor_diagnostics = [], {}
        for factor in FACTORS:
            matrix, detail = delta_surface(model, config, lexicon, task, factor, subsets)
            factor_arrays.append(np.median(matrix, axis=1).astype(np.float32))
            factor_diagnostics[factor] = detail
        model_arrays.append(np.stack(factor_arrays, axis=0))
        diagnostics.append({"model_id": row["model_id"], "factor": factor_diagnostics})
        del model
        torch.cuda.empty_cache()
    return np.stack(model_arrays, axis=0), diagnostics


def run_calibration_command() -> None:
    protocol = verify_protocol()
    model_summary = p1163.read_json(OUT_ROOT / "runs/models/summary.json")
    if not model_summary["full_task_behavior_gate_passed"]:
        raise RuntimeError("full-task behavior gate failed")
    root = OUT_ROOT / "runs/calibration"
    if root.exists():
        raise RuntimeError("refusing to overwrite calibration")
    root.mkdir(parents=True)
    device = torch.device("cuda")
    task_summaries = {}
    for task in FULL_TASKS:
        rows = qualified_rows(task)
        response, diagnostics = collect_task_responses(
            task, rows, calibration_subsets(), device
        )
        np.savez_compressed(
            root / f"{task}.npz",
            response=response,
            model_ids=np.asarray([row["model_id"] for row in rows]),
            splits=np.asarray([row["split"] for row in rows]),
        )
        p1163.write_jsonl(root / f"{task}_diagnostics.jsonl", diagnostics)
        denominator_min = min(
            item["factor"][factor]["denominator_min"]
            for item in diagnostics
            for factor in FACTORS
        )
        checks = {
            "model_count": len(rows) >= THRESHOLDS["task_total_qualified_min"],
            "shape": response.shape == (len(rows), len(FACTORS), len(calibration_subsets())),
            "finite": bool(np.isfinite(response).all()),
            "null": float(np.max(np.abs(response[:, :, 0]))) <= THRESHOLDS["null_abs_max"],
            "positive_denominator": denominator_min > THRESHOLDS["denominator_min"],
        }
        task_summaries[task] = {
            "model_count": len(rows),
            "denominator_min": denominator_min,
            "checks": checks,
            "passed": all(checks.values()),
            "pack_sha256": p1163.sha256_file(root / f"{task}.npz"),
        }
    summary = {
        "phase": PHASE,
        "created_at_utc": p1163.now(),
        "protocol_digest": protocol["protocol_digest"],
        "tasks": task_summaries,
        "calibration_gate_passed": all(row["passed"] for row in task_summaries.values()),
    }
    summary["summary_digest"] = p1163.digest(summary)
    p1163.write_json(root / "summary.json", summary)
    print(p1163.canonical(summary))


def load_calibration(task: str) -> tuple[np.ndarray, list[str], list[str]]:
    with np.load(OUT_ROOT / "runs/calibration" / f"{task}.npz") as pack:
        return (
            np.asarray(pack["response"], dtype=np.float64),
            [str(value) for value in pack["model_ids"].tolist()],
            [str(value) for value in pack["splits"].tolist()],
        )


def algorithm_predictions(
    calibration: np.ndarray, targets: list[tuple[int, ...]]
) -> dict[str, np.ndarray]:
    shape = calibration.shape[:-1] + (len(targets),)
    predictions = {name: np.zeros(shape, dtype=np.float64) for name in ALGORITHMS}
    predictions["zero"].fill(0.0)
    for model_index in range(calibration.shape[0]):
        for factor_index in range(calibration.shape[1]):
            values = calibration[model_index, factor_index]
            for algorithm in ("main", "pairwise"):
                coefficient = p1161.fit_coefficients(
                    algorithm, calibration_subsets(), values
                )
                predictions[algorithm][model_index, factor_index] = p1161.predict_values(
                    algorithm, coefficient, targets
                )
    predictions["max_single"] = p1165.max_lower_prediction(
        calibration, targets, 1
    )
    predictions["max_pair"] = p1165.max_lower_prediction(calibration, targets, 2)
    return predictions


def risk_metrics(prediction: np.ndarray, observed: np.ndarray) -> dict[str, Any]:
    response_q95 = np.quantile(np.abs(observed), 0.95, axis=2)
    error = np.abs(prediction - observed)
    unit_mae = np.mean(error, axis=2)
    relative_mae = unit_mae / np.maximum(
        response_q95, THRESHOLDS["transport_unit_q95_min"]
    )
    transport_units = response_q95 >= THRESHOLDS["transport_unit_q95_min"]
    risk_units = (unit_mae <= THRESHOLDS["risk_unit_mae_max"]) & (
        relative_mae <= THRESHOLDS["risk_unit_relative_mae_max"]
    )
    total_units = int(unit_mae.size)
    transport_identified = bool(
        float(np.median(response_q95)) >= THRESHOLDS["transport_median_q95_min"]
        and int(np.sum(transport_units))
        >= int(np.ceil(total_units * THRESHOLDS["transport_unit_fraction_min"]))
    )
    risk_passed = bool(
        float(np.median(unit_mae)) <= THRESHOLDS["risk_median_unit_mae_max"]
        and float(np.median(relative_mae))
        <= THRESHOLDS["risk_median_relative_mae_max"]
        and int(np.sum(risk_units))
        >= int(np.ceil(total_units * THRESHOLDS["risk_unit_fraction_min"]))
        and float(np.quantile(error, 0.95))
        <= THRESHOLDS["risk_schedule_abs_error_q95_max"]
    )
    return {
        "transport_identified": transport_identified,
        "risk_passed": risk_passed,
        "unit_count": total_units,
        "response_q95_median": float(np.median(response_q95)),
        "transport_unit_count": int(np.sum(transport_units)),
        "median_unit_mae": float(np.median(unit_mae)),
        "median_relative_mae": float(np.median(relative_mae)),
        "risk_unit_count": int(np.sum(risk_units)),
        "schedule_abs_error_q95": float(np.quantile(error, 0.95)),
        "schedule_abs_error_max": float(np.max(error)),
    }


def run_discovery_and_seal_command() -> None:
    protocol = verify_protocol()
    calibration_summary = p1163.read_json(OUT_ROOT / "runs/calibration/summary.json")
    if not calibration_summary["calibration_gate_passed"]:
        raise RuntimeError("calibration gate failed")
    discovery_root = OUT_ROOT / "runs/discovery"
    prediction_root = OUT_ROOT / "predictions"
    if discovery_root.exists() or prediction_root.exists():
        raise RuntimeError("refusing to overwrite discovery or predictions")
    confirmation_outcome_root = OUT_ROOT / "runs/confirmation"
    if confirmation_outcome_root.exists():
        raise RuntimeError("confirmation outcomes already exist")
    discovery_schedules, confirmation_schedules = schedule_splits()
    device = torch.device("cuda")
    selections, discovery_results = {}, {}
    prediction_root.mkdir(parents=True)
    discovery_root.mkdir(parents=True)
    for task in FULL_TASKS:
        rows = qualified_rows(task, "discovery")
        observed, diagnostics = collect_task_responses(task, rows, discovery_schedules, device)
        np.savez_compressed(
            discovery_root / f"{task}.npz",
            response=observed,
            model_ids=np.asarray([row["model_id"] for row in rows]),
        )
        p1163.write_jsonl(discovery_root / f"{task}_diagnostics.jsonl", diagnostics)
        calibration, model_ids, splits = load_calibration(task)
        indices = [index for index, split in enumerate(splits) if split == "discovery"]
        if [model_ids[index] for index in indices] != [row["model_id"] for row in rows]:
            raise RuntimeError("discovery calibration order mismatch")
        predictions = algorithm_predictions(calibration[indices], discovery_schedules)
        metrics = {
            algorithm: risk_metrics(predictions[algorithm], observed)
            for algorithm in ALGORITHMS
        }
        selected = {}
        for order in (1, 2):
            candidates = ORDER_LIBRARY[order]
            selected[str(order)] = min(
                candidates,
                key=lambda name: (
                    metrics[name]["median_relative_mae"],
                    metrics[name]["median_unit_mae"],
                    name,
                ),
            )
        selections[task] = selected
        discovery_results[task] = metrics

        confirmation_rows = qualified_rows(task, "confirmation")
        confirmation_indices = [
            index for index, split in enumerate(splits) if split == "confirmation"
        ]
        if [model_ids[index] for index in confirmation_indices] != [
            row["model_id"] for row in confirmation_rows
        ]:
            raise RuntimeError("confirmation calibration order mismatch")
        confirmation_predictions = algorithm_predictions(
            calibration[confirmation_indices], confirmation_schedules
        )
        np.savez_compressed(
            prediction_root / f"{task}.npz",
            **{key: value.astype(np.float32) for key, value in confirmation_predictions.items()},
            model_ids=np.asarray([row["model_id"] for row in confirmation_rows]),
        )
    metadata = {
        "phase": PHASE,
        "created_at_utc": p1163.now(),
        "protocol_digest": protocol["protocol_digest"],
        "selections": selections,
        "discovery_results": discovery_results,
        "confirmation_outcomes_absent_at_sealing": True,
        "prediction_hashes": {
            task: p1163.sha256_file(prediction_root / f"{task}.npz") for task in FULL_TASKS
        },
        "discovery_hashes": {
            task: p1163.sha256_file(discovery_root / f"{task}.npz") for task in FULL_TASKS
        },
    }
    metadata["selection_digest"] = p1163.digest(metadata)
    p1163.write_json(prediction_root / "selection.json", metadata)
    print(
        p1163.canonical(
            {
                "selections": selections,
                "selection_digest": metadata["selection_digest"],
                "confirmation_outcomes_absent": True,
            }
        )
    )


def run_confirmation_command() -> None:
    protocol = verify_protocol()
    selection = p1163.read_json(OUT_ROOT / "predictions/selection.json")
    if not selection["confirmation_outcomes_absent_at_sealing"]:
        raise RuntimeError("invalid selection seal")
    root = OUT_ROOT / "runs/confirmation"
    if root.exists():
        raise RuntimeError("refusing to overwrite confirmation")
    _, schedules = schedule_splits()
    device = torch.device("cuda")
    root.mkdir(parents=True)
    task_summaries = {}
    for task in FULL_TASKS:
        rows = qualified_rows(task, "confirmation")
        observed, diagnostics = collect_task_responses(task, rows, schedules, device)
        np.savez_compressed(
            root / f"{task}.npz",
            response=observed,
            model_ids=np.asarray([row["model_id"] for row in rows]),
        )
        p1163.write_jsonl(root / f"{task}_diagnostics.jsonl", diagnostics)
        checks = {
            "model_count": len(rows) >= THRESHOLDS["task_split_qualified_min"],
            "shape": observed.shape == (len(rows), len(FACTORS), len(schedules)),
            "finite": bool(np.isfinite(observed).all()),
            "prediction_integrity": p1163.sha256_file(OUT_ROOT / "predictions" / f"{task}.npz")
            == selection["prediction_hashes"][task],
        }
        task_summaries[task] = {
            "checks": checks,
            "passed": all(checks.values()),
            "pack_sha256": p1163.sha256_file(root / f"{task}.npz"),
        }
    summary = {
        "phase": PHASE,
        "created_at_utc": p1163.now(),
        "protocol_digest": protocol["protocol_digest"],
        "selection_digest": selection["selection_digest"],
        "tasks": task_summaries,
        "confirmation_gate_passed": all(row["passed"] for row in task_summaries.values()),
        "selection_precedes_confirmation": selection["created_at_utc"] < p1163.now(),
    }
    summary["summary_digest"] = p1163.digest(summary)
    p1163.write_json(root / "summary.json", summary)
    print(p1163.canonical(summary))


def calculate_score(
    selection: dict[str, Any],
    model_summary: dict[str, Any],
) -> dict[str, Any]:
    task_results = {}
    for task in FULL_TASKS:
        with np.load(OUT_ROOT / "runs/confirmation" / f"{task}.npz") as pack:
            observed = np.asarray(pack["response"], dtype=np.float64)
            observed_ids = [str(value) for value in pack["model_ids"].tolist()]
        with np.load(OUT_ROOT / "predictions" / f"{task}.npz") as pack:
            predictions = {
                algorithm: np.asarray(pack[algorithm], dtype=np.float64)
                for algorithm in ALGORITHMS
            }
            prediction_ids = [str(value) for value in pack["model_ids"].tolist()]
        if observed_ids != prediction_ids:
            raise RuntimeError(f"confirmation model order mismatch: {task}")
        selected1 = selection["selections"][task]["1"]
        selected2 = selection["selections"][task]["2"]
        metrics = {
            "zero": risk_metrics(predictions["zero"], observed),
            selected1: risk_metrics(predictions[selected1], observed),
            selected2: risk_metrics(predictions[selected2], observed),
        }
        transport = metrics[selected2]["transport_identified"]
        if metrics["zero"]["risk_passed"]:
            bound: int | str = 0
        elif metrics[selected1]["risk_passed"]:
            bound = 1
        elif metrics[selected2]["risk_passed"]:
            bound = 2
        else:
            bound = ">2_or_outside_library"
        task_results[task] = {
            "selected_order1": selected1,
            "selected_order2": selected2,
            "metrics": metrics,
            "transport_identified": transport,
            "operational_predictive_order_upper_bound": bound,
            "confirmed_at_most_two": bool(transport and isinstance(bound, int) and bound <= 2),
        }
    cross_function = all(row["confirmed_at_most_two"] for row in task_results.values())
    composition = model_summary["authorizations"][COMPOSITION_TASK]
    composition_confirmed = bool(model_summary["composition_generalization_authorized"])
    if cross_function and composition_confirmed:
        decision = "cross_function_low_order_and_composition_behavior_confirmed"
    elif cross_function:
        decision = "cross_function_low_order_confirmed_composition_not_authorized"
    else:
        decision = "cross_function_low_order_not_confirmed"
    return {
        "decision": decision,
        "cross_function_low_order_confirmed": cross_function,
        "composition_generalization_confirmed": composition_confirmed,
        "task_results": task_results,
        "composition_behavior": composition,
        "claim_scope": "independent confirmation of an operational predictive-order upper bound in a frozen finite estimator library across three fully supervised micro-tasks; composition is a separate behavior gate",
        "non_implications": [
            "The operational bound is not an intrinsic natural causal order.",
            "Full-supervision cross-task repetition is not compositional generalization.",
            "A failed composition behavior gate does not test hidden intervention responses on held-out combinations.",
            "Micro-task results do not establish a pretrained language-model mechanism.",
        ],
    }


def score_command() -> None:
    protocol = verify_protocol()
    confirmation_summary = p1163.read_json(OUT_ROOT / "runs/confirmation/summary.json")
    if not confirmation_summary["confirmation_gate_passed"]:
        raise RuntimeError("confirmation gate failed")
    selection = p1163.read_json(OUT_ROOT / "predictions/selection.json")
    model_summary = p1163.read_json(OUT_ROOT / "runs/models/summary.json")
    results = calculate_score(selection, model_summary)
    integrity = {
        "selection_precedes_confirmation": selection["created_at_utc"]
        < confirmation_summary["created_at_utc"],
        "selection_link": confirmation_summary["selection_digest"]
        == selection["selection_digest"],
        "confirmation_hashes": all(
            p1163.sha256_file(OUT_ROOT / "runs/confirmation" / f"{task}.npz")
            == confirmation_summary["tasks"][task]["pack_sha256"]
            for task in FULL_TASKS
        ),
    }
    if not all(integrity.values()):
        raise RuntimeError(f"integrity failure: {integrity}")
    score = {
        "phase": PHASE,
        "created_at_utc": p1163.now(),
        "protocol_digest": protocol["protocol_digest"],
        "selection_digest": selection["selection_digest"],
        "confirmation_summary_digest": confirmation_summary["summary_digest"],
        "results": results,
        "integrity_checks": integrity,
        "branch_status": "closed_after_cross_task_confirmation",
    }
    score["score_digest"] = p1163.digest(score)
    p1163.write_json(OUT_ROOT / "analysis/score.json", score)
    compact = {
        task: {
            "order1": row["selected_order1"],
            "order2": row["selected_order2"],
            "bound": row["operational_predictive_order_upper_bound"],
            "order1_mae": row["metrics"][row["selected_order1"]]["median_unit_mae"],
            "order2_mae": row["metrics"][row["selected_order2"]]["median_unit_mae"],
        }
        for task, row in results["task_results"].items()
    }
    print(
        p1163.canonical(
            {
                "decision": results["decision"],
                "tasks": compact,
                "composition": results["composition_behavior"],
                "score_digest": score["score_digest"],
            }
        )
    )


def finalize_command() -> None:
    protocol = verify_protocol()
    score = p1163.read_json(OUT_ROOT / "analysis/score.json")
    results = score["results"]
    auto_continue = bool(
        results["cross_function_low_order_confirmed"]
        and results["composition_generalization_confirmed"]
    )
    final = {
        "phase": PHASE,
        "created_at_utc": p1163.now(),
        "title": protocol["title"],
        "protocol_digest": protocol["protocol_digest"],
        "score_digest": score["score_digest"],
        "decision": results["decision"],
        "cross_function_low_order_confirmed": results["cross_function_low_order_confirmed"],
        "composition_generalization_confirmed": results["composition_generalization_confirmed"],
        "natural_mechanism_recovered": False,
        "branch_status": "closed_after_cross_task_confirmation",
        "auto_continue": auto_continue,
        "auto_continue_reason": (
            "Both cross-function low-order response prediction and compositional behavior passed; an independently preregistered composition-response phase is authorized."
            if auto_continue
            else "The cross-task phase is complete, but automatic hidden-state continuation requires both the full-supervision response result and the separately frozen compositional behavior gate."
        ),
        "non_implications": results["non_implications"],
    }
    final["final_digest"] = p1163.digest(final)
    p1163.write_json(OUT_ROOT / "analysis/final.json", final)
    print(p1163.canonical(final))


def smoke_command() -> None:
    discovery, confirmation = schedule_splits()
    print(
        p1163.canonical(
            {
                "task_checks": structural_task_checks(),
                "calibration_count": len(calibration_subsets()),
                "discovery_count": len(discovery),
                "confirmation_count": len(confirmation),
                "schedule_disjoint": not bool(set(discovery).intersection(confirmation)),
            }
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=(
            "protocol",
            "run-models",
            "run-calibration",
            "run-discovery-and-seal",
            "run-confirmation",
            "score",
            "finalize",
            "smoke",
        ),
    )
    command = parser.parse_args().command
    {
        "protocol": protocol_command,
        "run-models": run_models_command,
        "run-calibration": run_calibration_command,
        "run-discovery-and-seal": run_discovery_and_seal_command,
        "run-confirmation": run_confirmation_command,
        "score": score_command,
        "finalize": finalize_command,
        "smoke": smoke_command,
    }[command]()


if __name__ == "__main__":
    main()
