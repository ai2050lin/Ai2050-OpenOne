#!/usr/bin/env python3
"""Preregistered formation-axis test for compositional generalization.

Phase1166 showed that low-order intervention-response prediction can hold while
an unseen input subcube does not generalize.  This phase asks a logically prior
question: can a frozen, training-domain-only intervention create a robust pair
of generalizers and failures without exposing held-out examples to gradients?

The held-out panel is not evaluated until every arm, seed, and checkpoint has
been trained and sealed.  Hidden-state comparison is authorized only if the
predeclared combined arm generalizes in both splits and architectures while the
baseline remains a matched failure.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve()
AUDIT_SCRIPT = ROOT / "tests/glm5/phase1167_compositional_formation_axis_audit.py"
P1166_SCRIPT = ROOT / "tests/glm5/phase1166_cross_task_predictive_order_confirmation.py"
P1166_FINAL = (
    ROOT
    / "tests/glm5/result/phase1166_cross_task_predictive_order_confirmation/analysis/final.json"
)
P1166_AUDIT = (
    ROOT
    / "tests/glm5/result/phase1166_cross_task_predictive_order_confirmation/audit/independent_audit.json"
)
OUT_ROOT = ROOT / "tests/glm5/result/phase1167_compositional_formation_axis"
sys.path.insert(0, str(ROOT / "tests/glm5"))

import phase1166_cross_task_predictive_order_confirmation as p1166  # noqa: E402


p1163 = p1166.p1163
PHASE = 1167
TASK = p1166.COMPOSITION_TASK
ARCHITECTURES = p1166.ARCHITECTURES
ARMS = {
    "baseline": {"factor_aux": False, "equivariance": False},
    "factor_aux": {"factor_aux": True, "equivariance": False},
    "equivariance": {"factor_aux": False, "equivariance": True},
    "factor_aux_equivariance": {"factor_aux": True, "equivariance": True},
}
PRIMARY_GENERALIZER_ARM = "factor_aux_equivariance"
REPLICATES = 4
DISCOVERY_REPLICATES = (0, 1)
CONFIRMATION_REPLICATES = (2, 3)
TRAINING = {
    "fixed_steps": 1500,
    "evaluation_steps": (500, 1000, 1500),
    "batch_size": 128,
    "learning_rate": 0.0005,
    "weight_decay": 0.001,
    "gradient_clip_norm": 1.0,
    "factor_aux_weight": 0.3,
    "equivariance_weight": 0.3,
}
THRESHOLDS = {
    "train_accuracy_min": 1.0,
    "finite_fraction_min": 1.0,
    "generalizer_holdout_accuracy_min": 0.90,
    "failure_holdout_accuracy_max": 0.10,
    "required_models_per_architecture_split": 2,
}


def split_name(replicate: int) -> str:
    if replicate in DISCOVERY_REPLICATES:
        return "discovery"
    if replicate in CONFIRMATION_REPLICATES:
        return "confirmation"
    raise ValueError(replicate)


def model_seed(arm: str, architecture: str, replicate: int) -> int:
    return (
        21670001
        + list(ARMS).index(arm) * 100_003
        + list(ARCHITECTURES).index(architecture) * 10_009
        + int(replicate) * 1_009
    )


def model_id(arm: str, architecture: str, replicate: int, seed: int) -> str:
    return p1163.digest(
        {
            "phase": PHASE,
            "arm": arm,
            "architecture": architecture,
            "replicate": replicate,
            "seed": seed,
        }
    )[:16]


def build_examples(lexicon: dict[str, Any]) -> dict[str, torch.Tensor]:
    train_x, train_y, holdout_x, holdout_y = p1166.task_examples(TASK, lexicon)
    rows, cols, contexts = [], [], []
    lookup: dict[tuple[int, int, int, int], int] = {}
    index = 0
    for template in range(len(p1166.source.TEMPLATES)):
        for context in range(p1166.source.CONTEXTS):
            for row in range(p1166.source.ROWS):
                for col in range(p1166.source.COLS):
                    if p1166.composition_holdout(row, col, context):
                        continue
                    lookup[(template, row, col, context)] = index
                    rows.append(row)
                    cols.append(col)
                    contexts.append(context)
                    index += 1

    pair_a, pair_b, permutations = [], [], []
    shift_permutation = [
        ((value % 4) + 1) % 4 + 4 * (value // 4) for value in range(8)
    ]
    context_permutation = [value ^ 4 for value in range(8)]
    for (template, row, col, context), source_index in lookup.items():
        transforms = (
            ((row + 1) % 4, col, context, shift_permutation),
            (row, (col + 1) % 4, context, shift_permutation),
            (row, col, 1 - context, context_permutation),
        )
        for new_row, new_col, new_context, permutation in transforms:
            target_index = lookup.get((template, new_row, new_col, new_context))
            if target_index is not None:
                pair_a.append(source_index)
                pair_b.append(target_index)
                permutations.append(permutation)

    result = {
        "train_x": train_x,
        "train_y": train_y,
        "holdout_x": holdout_x,
        "holdout_y": holdout_y,
        "rows": torch.tensor(rows, dtype=torch.long),
        "cols": torch.tensor(cols, dtype=torch.long),
        "contexts": torch.tensor(contexts, dtype=torch.long),
        "pair_a": torch.tensor(pair_a, dtype=torch.long),
        "pair_b": torch.tensor(pair_b, dtype=torch.long),
        "permutations": torch.tensor(permutations, dtype=torch.long),
    }
    if len(result["rows"]) != len(train_x):
        raise RuntimeError("factor-label order does not match training examples")
    return result


def evaluate(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    lexicon: dict[str, Any],
) -> dict[str, Any]:
    return p1166.evaluate(model, inputs, targets, TASK, lexicon)


def checkpoint_payload(
    model: torch.nn.Module,
    config: Any,
    lexicon: dict[str, Any],
    arm: str,
    seed: int,
) -> dict[str, Any]:
    return {
        "phase": PHASE,
        "task": TASK,
        "arm": arm,
        "seed": seed,
        "config": asdict(config),
        "lexicon": lexicon,
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
    }


def load_checkpoint(path: Path, device: torch.device) -> tuple[torch.nn.Module, Any, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = p1166.source.ModelConfig(**payload["config"])
    model = p1166.source.TinyCausalTransformer(config).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, config, payload["lexicon"]


def train_one(
    arm: str,
    architecture: str,
    replicate: int,
    device: torch.device,
) -> tuple[torch.nn.Module, Any, dict[str, Any], dict[str, Any]]:
    settings = ARMS[arm]
    config = ARCHITECTURES[architecture]
    seed = model_seed(arm, architecture, replicate)
    p1166.source.set_seed(seed)
    lexicon = p1166.make_lexicon(seed + 18_017)
    data = build_examples(lexicon)
    model = p1166.source.TinyCausalTransformer(config).to(device)
    row_head = nn.Linear(config.width, p1166.source.ROWS).to(device)
    col_head = nn.Linear(config.width, p1166.source.COLS).to(device)
    context_head = nn.Linear(config.width, p1166.source.CONTEXTS).to(device)
    parameters = list(model.parameters())
    if settings["factor_aux"]:
        parameters += (
            list(row_head.parameters())
            + list(col_head.parameters())
            + list(context_head.parameters())
        )
    optimizer = torch.optim.AdamW(
        parameters,
        lr=TRAINING["learning_rate"],
        weight_decay=TRAINING["weight_decay"],
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 31)
    candidates = p1166.answer_ids(TASK, lexicon, device)
    trace = []
    final_gradient_norm = 0.0
    for step in range(1, TRAINING["fixed_steps"] + 1):
        model.train()
        indices = torch.randint(
            0,
            len(data["train_x"]),
            (TRAINING["batch_size"],),
            generator=generator,
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            raw, states = model(data["train_x"][indices].to(device), return_states=True)
            logits = raw[:, -1].index_select(-1, candidates)
            task_loss = F.cross_entropy(logits.float(), data["train_y"][indices].to(device))
            loss = task_loss
            aux_loss = torch.zeros((), device=device)
            if settings["factor_aux"]:
                hidden = model.final_norm(states[-1])[:, -1]
                aux_loss = (
                    F.cross_entropy(row_head(hidden).float(), data["rows"][indices].to(device))
                    + F.cross_entropy(col_head(hidden).float(), data["cols"][indices].to(device))
                    + F.cross_entropy(
                        context_head(hidden).float(), data["contexts"][indices].to(device)
                    )
                ) / 3.0
                loss = loss + TRAINING["factor_aux_weight"] * aux_loss
            equivariance_loss = torch.zeros((), device=device)
            if settings["equivariance"]:
                pair_indices = torch.randint(
                    0,
                    len(data["pair_a"]),
                    (TRAINING["batch_size"],),
                    generator=generator,
                )
                pair_a = data["pair_a"][pair_indices]
                pair_b = data["pair_b"][pair_indices]
                permutation = data["permutations"][pair_indices].to(device)
                logits_a = (
                    model(data["train_x"][pair_a].to(device))[:, -1]
                    .index_select(-1, candidates)
                    .float()
                )
                logits_b = (
                    model(data["train_x"][pair_b].to(device))[:, -1]
                    .index_select(-1, candidates)
                    .float()
                )
                logits_a = logits_a - logits_a.mean(dim=1, keepdim=True)
                logits_b = logits_b - logits_b.mean(dim=1, keepdim=True)
                mapped_b = logits_b.gather(1, permutation)
                normalized_a = logits_a / (logits_a.std(dim=1, keepdim=True) + 1e-5)
                normalized_b = mapped_b / (mapped_b.std(dim=1, keepdim=True) + 1e-5)
                equivariance_loss = F.mse_loss(normalized_b, normalized_a)
                loss = loss + TRAINING["equivariance_weight"] * equivariance_loss
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"nonfinite loss: {arm}/{architecture}/{replicate}/{step}")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            parameters, TRAINING["gradient_clip_norm"]
        )
        if not bool(torch.isfinite(torch.as_tensor(gradient_norm))):
            raise RuntimeError(f"nonfinite gradient: {arm}/{architecture}/{replicate}/{step}")
        final_gradient_norm = float(gradient_norm)
        optimizer.step()
        if step in TRAINING["evaluation_steps"]:
            train_metrics = evaluate(
                model, data["train_x"], data["train_y"], lexicon
            )
            trace.append(
                {
                    "step": step,
                    "loss": float(loss.item()),
                    "task_loss": float(task_loss.item()),
                    "aux_loss": float(aux_loss.item()),
                    "equivariance_loss": float(equivariance_loss.item()),
                    "train": train_metrics,
                }
            )
    train_metrics = evaluate(model, data["train_x"], data["train_y"], lexicon)
    metrics = {
        "model_id": model_id(arm, architecture, replicate, seed),
        "arm": arm,
        "architecture": architecture,
        "replicate": replicate,
        "split": split_name(replicate),
        "seed": seed,
        "training_case_count": len(data["train_x"]),
        "sealed_holdout_case_count": len(data["holdout_x"]),
        "equivariance_pair_count": len(data["pair_a"]),
        "train_input_digest": p1163.digest(data["train_x"].tolist()),
        "holdout_input_digest": p1163.digest(data["holdout_x"].tolist()),
        "lexicon_digest": p1163.digest(lexicon),
        "fixed_steps_completed": TRAINING["fixed_steps"],
        "final_gradient_norm": final_gradient_norm,
        "train": train_metrics,
        "trace": trace,
        "holdout_evaluated_during_training": False,
        "holdout_used_by_gradient": False,
    }
    del row_head, col_head, context_head
    return model, config, lexicon, metrics


def protocol_command() -> None:
    if OUT_ROOT.exists():
        raise RuntimeError("refusing to overwrite an existing Phase1167 result directory")
    prior_final = p1163.read_json(P1166_FINAL)
    prior_audit = p1163.read_json(P1166_AUDIT)
    probe_lexicon = p1166.make_lexicon(21679999)
    probe = build_examples(probe_lexicon)
    disjoint = not {
        tuple(row) for row in probe["train_x"].tolist()
    }.intersection({tuple(row) for row in probe["holdout_x"].tolist()})
    checks = {
        "phase1166_cross_function_confirmed": prior_final[
            "cross_function_low_order_confirmed"
        ],
        "phase1166_composition_failed": not prior_final[
            "composition_generalization_confirmed"
        ],
        "phase1166_audit_passed": prior_audit["all_checks_passed"],
        "cuda_available": torch.cuda.is_available(),
        "arm_count": len(ARMS) == 4,
        "primary_arm_frozen": PRIMARY_GENERALIZER_ARM in ARMS,
        "split_disjoint": set(DISCOVERY_REPLICATES).isdisjoint(CONFIRMATION_REPLICATES),
        "split_complete": set(DISCOVERY_REPLICATES + CONFIRMATION_REPLICATES)
        == set(range(REPLICATES)),
        "input_partitions_disjoint": disjoint,
        "training_cases_positive": len(probe["train_x"]) > 0,
        "holdout_cases_positive": len(probe["holdout_x"]) > 0,
        "equivariance_training_only": int(probe["pair_a"].max()) < len(probe["train_x"])
        and int(probe["pair_b"].max()) < len(probe["train_x"]),
    }
    if not all(checks.values()):
        raise RuntimeError(f"protocol checks failed: {checks}")
    protocol = {
        "phase": PHASE,
        "created_at_utc": p1163.now(),
        "title": "sealed compositional-generalization formation axis with training-domain-only interventions",
        "source_digests": {
            "phase1166_final": prior_final["final_digest"],
            "phase1166_audit": prior_audit["audit_digest"],
        },
        "source_hashes": {
            "primary_script": p1163.sha256_file(SCRIPT),
            "audit_script": p1163.sha256_file(AUDIT_SCRIPT),
            "phase1166_script": p1163.sha256_file(P1166_SCRIPT),
        },
        "task": TASK,
        "task_formula": "((row + col) mod 4) + 4*context",
        "holdout_rule": "row in {0,2} and col in {0,2}; both contexts and every template held out",
        "training_case_count": len(probe["train_x"]),
        "holdout_case_count": len(probe["holdout_x"]),
        "equivariance_pair_count": len(probe["pair_a"]),
        "architectures": {name: asdict(config) for name, config in ARCHITECTURES.items()},
        "arms": ARMS,
        "primary_generalizer_arm": PRIMARY_GENERALIZER_ARM,
        "replicates": REPLICATES,
        "discovery_replicates": list(DISCOVERY_REPLICATES),
        "confirmation_replicates": list(CONFIRMATION_REPLICATES),
        "training": TRAINING,
        "thresholds": THRESHOLDS,
        "primary_endpoint": "baseline is a stable matched failure and factor_aux_equivariance is a stable generalizer in both splits and both architectures",
        "authorization_rule": "hidden-state mechanism signatures require the primary endpoint; single-arm or single-split success is insufficient",
        "scope": "formation-axis calibration only; it cannot repair or relabel the Phase1166 0/8 compositional result",
        "hard_stops": [
            "No held-out example, label, or transformed held-out pair enters any gradient.",
            "All arms, architectures, replicates, steps, weights, and thresholds are frozen before checkpoint training.",
            "All checkpoints and training-only metrics are sealed before the first holdout evaluation.",
            "The combined arm is the sole primary candidate; no post-hoc arm selection is allowed.",
            "Failure to form the paired behavior object denies hidden-state classification, component search, and causal localization.",
            "An explicitly factorized architecture is not substituted after failure because architecture identity would trivially reveal the class.",
            "Pilot outputs are engineering diagnostics and are not counted as confirmation evidence.",
        ],
        "checks": checks,
    }
    protocol["protocol_digest"] = p1163.digest(protocol)
    p1163.write_json(OUT_ROOT / "protocol/preregistration.json", protocol)
    p1163.write_json(
        OUT_ROOT / "protocol/audit.json",
        {
            "checks": checks,
            "all_checks_passed": all(checks.values()),
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
        ("phase1166_script", P1166_SCRIPT),
    ):
        if p1163.sha256_file(path) != protocol["source_hashes"][key]:
            raise RuntimeError(f"frozen source changed: {key}")
    return protocol


def train_and_seal_command() -> None:
    protocol = verify_protocol()
    root = OUT_ROOT / "runs/training"
    if root.exists():
        raise RuntimeError("refusing to overwrite sealed training run")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    rows = []
    for arm in ARMS:
        for architecture in ARCHITECTURES:
            for replicate in range(REPLICATES):
                model, config, lexicon, metrics = train_one(
                    arm, architecture, replicate, device
                )
                checkpoint = root / "checkpoints" / f"{metrics['model_id']}.pt"
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    checkpoint_payload(
                        model, config, lexicon, arm, metrics["seed"]
                    ),
                    checkpoint,
                )
                metrics["checkpoint_sha256"] = p1163.sha256_file(checkpoint)
                rows.append(metrics)
                print(
                    p1163.canonical(
                        {
                            "trained": metrics["model_id"],
                            "arm": arm,
                            "architecture": architecture,
                            "replicate": replicate,
                            "train_accuracy": metrics["train"]["accuracy"],
                        }
                    ),
                    flush=True,
                )
                del model
                torch.cuda.empty_cache()
    root.mkdir(parents=True, exist_ok=True)
    p1163.write_jsonl(root / "training_metrics.jsonl", rows)
    checks = {
        "model_count": len(rows) == len(ARMS) * len(ARCHITECTURES) * REPLICATES,
        "all_fixed_budget": all(
            row["fixed_steps_completed"] == TRAINING["fixed_steps"] for row in rows
        ),
        "all_train_qualified": all(
            row["train"]["accuracy"] >= THRESHOLDS["train_accuracy_min"]
            and row["train"]["finite_fraction"] >= THRESHOLDS["finite_fraction_min"]
            for row in rows
        ),
        "no_holdout_evaluated": all(
            not row["holdout_evaluated_during_training"] for row in rows
        ),
        "no_holdout_gradient": all(not row["holdout_used_by_gradient"] for row in rows),
        "checkpoint_hashes_present": all(bool(row["checkpoint_sha256"]) for row in rows),
    }
    if not all(checks.values()):
        raise RuntimeError(f"training seal failed: {checks}")
    summary = {
        "phase": PHASE,
        "created_at_utc": p1163.now(),
        "protocol_digest": protocol["protocol_digest"],
        "checks": checks,
        "training_sealed": True,
        "holdout_outcomes_absent_at_sealing": not (OUT_ROOT / "runs/holdout").exists(),
        "training_metrics_sha256": p1163.sha256_file(root / "training_metrics.jsonl"),
        "checkpoint_hashes": {
            row["model_id"]: row["checkpoint_sha256"] for row in rows
        },
    }
    summary["seal_digest"] = p1163.digest(summary)
    p1163.write_json(root / "seal.json", summary)
    print(p1163.canonical(summary))


def evaluate_holdout_command() -> None:
    protocol = verify_protocol()
    training_root = OUT_ROOT / "runs/training"
    seal = p1163.read_json(training_root / "seal.json")
    if not seal["training_sealed"] or not seal["holdout_outcomes_absent_at_sealing"]:
        raise RuntimeError("invalid training seal")
    if p1163.sha256_file(training_root / "training_metrics.jsonl") != seal[
        "training_metrics_sha256"
    ]:
        raise RuntimeError("training metrics changed after sealing")
    root = OUT_ROOT / "runs/holdout"
    if root.exists():
        raise RuntimeError("refusing to overwrite holdout outcomes")
    root.mkdir(parents=True)
    device = torch.device("cuda")
    rows = []
    for train_row in p1163.read_jsonl(training_root / "training_metrics.jsonl"):
        checkpoint = training_root / "checkpoints" / f"{train_row['model_id']}.pt"
        if p1163.sha256_file(checkpoint) != train_row["checkpoint_sha256"]:
            raise RuntimeError(f"checkpoint changed: {train_row['model_id']}")
        model, _, lexicon = load_checkpoint(checkpoint, device)
        data = build_examples(lexicon)
        holdout = evaluate(model, data["holdout_x"], data["holdout_y"], lexicon)
        row = {
            "model_id": train_row["model_id"],
            "arm": train_row["arm"],
            "architecture": train_row["architecture"],
            "replicate": train_row["replicate"],
            "split": train_row["split"],
            "seed": train_row["seed"],
            "train": train_row["train"],
            "holdout": holdout,
            "generalizer": bool(
                holdout["accuracy"]
                >= THRESHOLDS["generalizer_holdout_accuracy_min"]
                and holdout["finite_fraction"] >= THRESHOLDS["finite_fraction_min"]
            ),
            "matched_failure": bool(
                holdout["accuracy"] <= THRESHOLDS["failure_holdout_accuracy_max"]
                and holdout["finite_fraction"] >= THRESHOLDS["finite_fraction_min"]
            ),
        }
        rows.append(row)
        print(
            p1163.canonical(
                {
                    "evaluated": row["model_id"],
                    "arm": row["arm"],
                    "holdout_accuracy": holdout["accuracy"],
                    "holdout_mean_probability": holdout["mean_probability"],
                }
            ),
            flush=True,
        )
        del model
        torch.cuda.empty_cache()
    p1163.write_jsonl(root / "holdout_metrics.jsonl", rows)
    summary = {
        "phase": PHASE,
        "created_at_utc": p1163.now(),
        "protocol_digest": protocol["protocol_digest"],
        "seal_digest": seal["seal_digest"],
        "model_count": len(rows),
        "finite": all(row["holdout"]["finite_fraction"] == 1.0 for row in rows),
        "holdout_metrics_sha256": p1163.sha256_file(root / "holdout_metrics.jsonl"),
    }
    summary["summary_digest"] = p1163.digest(summary)
    p1163.write_json(root / "summary.json", summary)
    print(p1163.canonical(summary))


def cell_summary(rows: list[dict[str, Any]], arm: str, split: str, architecture: str) -> dict[str, Any]:
    selected = [
        row
        for row in rows
        if row["arm"] == arm
        and row["split"] == split
        and row["architecture"] == architecture
    ]
    return {
        "count": len(selected),
        "train_accuracy_min": min(row["train"]["accuracy"] for row in selected),
        "holdout_accuracy_min": min(row["holdout"]["accuracy"] for row in selected),
        "holdout_accuracy_max": max(row["holdout"]["accuracy"] for row in selected),
        "holdout_accuracy_mean": sum(row["holdout"]["accuracy"] for row in selected)
        / len(selected),
        "holdout_probability_mean": sum(
            row["holdout"]["mean_probability"] for row in selected
        )
        / len(selected),
        "all_generalizers": all(row["generalizer"] for row in selected),
        "all_matched_failures": all(row["matched_failure"] for row in selected),
    }


def score_command() -> None:
    protocol = verify_protocol()
    root = OUT_ROOT / "analysis"
    if root.exists():
        raise RuntimeError("refusing to overwrite score")
    holdout_root = OUT_ROOT / "runs/holdout"
    holdout_summary = p1163.read_json(holdout_root / "summary.json")
    rows = p1163.read_jsonl(holdout_root / "holdout_metrics.jsonl")
    cells = {
        arm: {
            split: {
                architecture: cell_summary(rows, arm, split, architecture)
                for architecture in ARCHITECTURES
            }
            for split in ("discovery", "confirmation")
        }
        for arm in ARMS
    }
    cell_count_ok = all(
        cells[arm][split][architecture]["count"]
        == THRESHOLDS["required_models_per_architecture_split"]
        for arm in ARMS
        for split in ("discovery", "confirmation")
        for architecture in ARCHITECTURES
    )
    baseline_failure = all(
        cells["baseline"][split][architecture]["all_matched_failures"]
        for split in ("discovery", "confirmation")
        for architecture in ARCHITECTURES
    )
    primary_generalizer = all(
        cells[PRIMARY_GENERALIZER_ARM][split][architecture]["all_generalizers"]
        for split in ("discovery", "confirmation")
        for architecture in ARCHITECTURES
    )
    training_matched = all(
        cells[arm][split][architecture]["train_accuracy_min"]
        >= THRESHOLDS["train_accuracy_min"]
        for arm in ARMS
        for split in ("discovery", "confirmation")
        for architecture in ARCHITECTURES
    )
    behavior_contrast = bool(
        cell_count_ok and training_matched and baseline_failure and primary_generalizer
    )
    arm_generalization_counts = {
        arm: sum(row["generalizer"] for row in rows if row["arm"] == arm)
        for arm in ARMS
    }
    arm_failure_counts = {
        arm: sum(row["matched_failure"] for row in rows if row["arm"] == arm)
        for arm in ARMS
    }
    results = {
        "cell_count_ok": cell_count_ok,
        "training_matched": training_matched,
        "baseline_stable_failure": baseline_failure,
        "primary_arm_stable_generalizer": primary_generalizer,
        "behavior_contrast_authorized": behavior_contrast,
        "hidden_state_scan_authorized": behavior_contrast,
        "arm_generalization_counts": arm_generalization_counts,
        "arm_failure_counts": arm_failure_counts,
        "decision": (
            "paired formation object established; an independently frozen mechanism-signature phase is authorized"
            if behavior_contrast
            else "paired formation object not established; hidden-state comparison is denied and the finite formation panel stops"
        ),
        "non_implications": [
            "Training-set fit does not imply held-out composition generalization.",
            "Factor readability pressure does not imply factor use.",
            "Training-domain logit equivariance does not imply unseen-subcube behavior.",
            "Failure of this finite arm panel does not prove that no training intervention can create compositional generalization.",
            "Success, if observed, would authorize comparison but would not itself identify a mechanism.",
        ],
    }
    root.mkdir(parents=True)
    score = {
        "phase": PHASE,
        "created_at_utc": p1163.now(),
        "protocol_digest": protocol["protocol_digest"],
        "holdout_summary_digest": holdout_summary["summary_digest"],
        "cells": cells,
        "results": results,
    }
    score["score_digest"] = p1163.digest(score)
    p1163.write_json(root / "score.json", score)
    print(p1163.canonical({"results": results, "score_digest": score["score_digest"]}))


def finalize_command() -> None:
    protocol = verify_protocol()
    score = p1163.read_json(OUT_ROOT / "analysis/score.json")
    authorized = bool(score["results"]["hidden_state_scan_authorized"])
    final = {
        "phase": PHASE,
        "created_at_utc": p1163.now(),
        "title": protocol["title"],
        "protocol_digest": protocol["protocol_digest"],
        "score_digest": score["score_digest"],
        "decision": score["results"]["decision"],
        "behavior_contrast_authorized": authorized,
        "hidden_state_scan_authorized": authorized,
        "natural_mechanism_recovered": False,
        "branch_status": (
            "open_only_for_independent_mechanism_signature_preregistration"
            if authorized
            else "closed_after_finite_formation_panel"
        ),
        "auto_continue": authorized,
        "auto_continue_reason": (
            "The preregistered paired behavior object exists in both splits and architectures."
            if authorized
            else "No preregistered generalizer/failure pair exists, so internal signatures would classify an arm label rather than a validated behavioral mechanism."
        ),
        "non_implications": score["results"]["non_implications"],
    }
    final["final_digest"] = p1163.digest(final)
    p1163.write_json(OUT_ROOT / "analysis/final.json", final)
    print(p1163.canonical(final))


def smoke_command() -> None:
    lexicon = p1166.make_lexicon(21679999)
    data = build_examples(lexicon)
    print(
        p1163.canonical(
            {
                "train_shape": list(data["train_x"].shape),
                "holdout_shape": list(data["holdout_x"].shape),
                "equivariance_pairs": len(data["pair_a"]),
                "arms": list(ARMS),
                "model_count": len(ARMS) * len(ARCHITECTURES) * REPLICATES,
            }
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=(
            "protocol",
            "train-and-seal",
            "evaluate-holdout",
            "score",
            "finalize",
            "smoke",
        ),
    )
    command = parser.parse_args().command
    {
        "protocol": protocol_command,
        "train-and-seal": train_and_seal_command,
        "evaluate-holdout": evaluate_holdout_command,
        "score": score_command,
        "finalize": finalize_command,
        "smoke": smoke_command,
    }[command]()


if __name__ == "__main__":
    main()
