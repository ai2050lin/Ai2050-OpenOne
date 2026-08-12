#!/usr/bin/env python3
"""Fresh-task breadth confirmation for natural rule selection.

Phase1169 observed six memorizer-to-generalizer trajectories among eight runs,
but the task split was asymmetric (2/4 for p=29 and 4/4 for p=37).  This phase
does not search for a predictor and does not tune the learner.  It freezes a
random breadth panel of four previously unused prime moduli, runs eight seeds
per task with the exact Phase1169 learner and training schedule, seals all
training-only records, and only then reveals held-out performance.

The endpoint is deliberately mixed: a future prospective predictor needs both
transition and non-transition examples.  Passing therefore requires at least
three tasks with two examples of each outcome and at least eight examples of
each outcome globally.  A pass authorizes a separate, fresh-task prediction
phase; it is not a mechanism claim.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))

import phase1169_natural_training_trajectory_bifurcation as base  # noqa: E402


SCRIPT = Path(__file__).resolve()
AUDIT_SCRIPT = ROOT / "tests/glm5/phase1170_natural_rule_selection_breadth_confirmation_audit.py"
OUT_ROOT = ROOT / "tests/glm5/result/phase1170_natural_rule_selection_breadth_confirmation"
P1169_FINAL = ROOT / "tests/glm5/result/phase1169_natural_training_trajectory_bifurcation/analysis/final.json"
P1169_AUDIT = ROOT / "tests/glm5/result/phase1169_natural_training_trajectory_bifurcation/audit/independent_audit.json"

PHASE = 1170
PRIME_POOL = (19, 23, 41, 43, 47, 53, 59, 61)
TASK_SELECTION_SEED = 11700001
TASK_COUNT = 4
REPLICATES = 8
CHECKPOINT_STEPS = base.CHECKPOINT_STEPS
TRAIN_FRACTION = base.TRAIN_FRACTION
MODEL_WIDTH = base.MODEL_WIDTH
TRAINING = dict(base.TRAINING)
TRAJECTORY_THRESHOLDS = {
    "train_accuracy_min": base.THRESHOLDS["train_accuracy_min"],
    "memorizer_holdout_accuracy_max": base.THRESHOLDS["memorizer_holdout_accuracy_max"],
    "generalizer_holdout_accuracy_min": base.THRESHOLDS["generalizer_holdout_accuracy_min"],
    "stable_generalizer_checkpoint_count_min": base.THRESHOLDS["stable_generalizer_checkpoint_count_min"],
    "finite_fraction_min": base.THRESHOLDS["finite_fraction_min"],
}
BREADTH_THRESHOLDS = {
    "informative_task_count_min": 3,
    "successes_per_informative_task_min": 2,
    "failures_per_informative_task_min": 2,
    "global_success_count_min": 8,
    "global_failure_count_min": 8,
}


def task_permutation() -> tuple[int, ...]:
    return tuple(random.Random(TASK_SELECTION_SEED).sample(PRIME_POOL, len(PRIME_POOL)))


SELECTED_MODULI = task_permutation()[:TASK_COUNT]
RESERVED_MODULI = task_permutation()[TASK_COUNT:]
TASKS = {f"breadth_{index}_p{modulus}": modulus for index, modulus in enumerate(SELECTED_MODULI)}


def model_seed(task_index: int, replicate: int) -> int:
    return 11700000 + int(task_index) * 100_003 + int(replicate) * 1_009


def checkpoint_payload(
    model: base.SymmetricSquareNetwork,
    task_name: str,
    task_index: int,
    replicate: int,
    seed: int,
    step: int,
) -> dict[str, Any]:
    return {
        "phase": PHASE,
        "task_name": task_name,
        "task_index": task_index,
        "replicate": replicate,
        "seed": seed,
        "step": step,
        "config": asdict(model.config),
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
    }


def load_checkpoint(path: Path, device: torch.device) -> base.SymmetricSquareNetwork:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = base.SymmetricSquareNetwork(base.SquareConfig(**payload["config"])).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def trajectory_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: row["step"])
    memorizers = [
        row
        for row in ordered
        if row["train"]["accuracy"] >= TRAJECTORY_THRESHOLDS["train_accuracy_min"]
        and row["holdout"]["accuracy"] <= TRAJECTORY_THRESHOLDS["memorizer_holdout_accuracy_max"]
    ]
    generalizers = [
        row
        for row in ordered
        if row["train"]["accuracy"] >= TRAJECTORY_THRESHOLDS["train_accuracy_min"]
        and row["holdout"]["accuracy"] >= TRAJECTORY_THRESHOLDS["generalizer_holdout_accuracy_min"]
    ]
    valid_pairs = [(memorizer, generalizer) for memorizer in memorizers for generalizer in generalizers if memorizer["step"] < generalizer["step"]]
    selected = min(valid_pairs, key=lambda pair: (pair[1]["step"], -pair[0]["step"])) if valid_pairs else None
    stable_pair = any(ordered[index] in generalizers and ordered[index + 1] in generalizers for index in range(len(ordered) - 1))
    transition = selected is not None and stable_pair
    return {
        "trajectory_id": ordered[0]["trajectory_id"],
        "task_name": ordered[0]["task_name"],
        "task_index": ordered[0]["task_index"],
        "modulus": ordered[0]["modulus"],
        "replicate": ordered[0]["replicate"],
        "seed": ordered[0]["seed"],
        "memorizer_checkpoint_count": len(memorizers),
        "generalizer_checkpoint_count": len(generalizers),
        "stable_generalizer_pair_present": stable_pair,
        "transition_present": transition,
        "memorizer_step": selected[0]["step"] if selected else None,
        "memorizer_holdout_accuracy": selected[0]["holdout"]["accuracy"] if selected else None,
        "generalizer_step": selected[1]["step"] if selected else None,
        "generalizer_holdout_accuracy": selected[1]["holdout"]["accuracy"] if selected else None,
        "maximum_holdout_accuracy": max(row["holdout"]["accuracy"] for row in ordered),
        "final_holdout_accuracy": ordered[-1]["holdout"]["accuracy"],
    }


def task_summary(trajectories: list[dict[str, Any]], task_name: str, modulus: int) -> dict[str, Any]:
    selected = [row for row in trajectories if row["task_name"] == task_name]
    successes = sum(bool(row["transition_present"]) for row in selected)
    failures = len(selected) - successes
    informative = (
        successes >= BREADTH_THRESHOLDS["successes_per_informative_task_min"]
        and failures >= BREADTH_THRESHOLDS["failures_per_informative_task_min"]
    )
    return {
        "task_name": task_name,
        "modulus": modulus,
        "trajectory_count": len(selected),
        "transition_count": successes,
        "non_transition_count": failures,
        "informative_mixed_panel": informative,
        "median_maximum_holdout_accuracy": float(torch.tensor([row["maximum_holdout_accuracy"] for row in selected]).median().item()),
    }


def breadth_decision(trajectories: list[dict[str, Any]]) -> dict[str, Any]:
    task_summaries = [task_summary(trajectories, task_name, modulus) for task_name, modulus in TASKS.items()]
    successes = sum(bool(row["transition_present"]) for row in trajectories)
    failures = len(trajectories) - successes
    informative_tasks = sum(bool(row["informative_mixed_panel"]) for row in task_summaries)
    pass_conditions = {
        "informative_task_count": informative_tasks >= BREADTH_THRESHOLDS["informative_task_count_min"],
        "global_success_count": successes >= BREADTH_THRESHOLDS["global_success_count_min"],
        "global_failure_count": failures >= BREADTH_THRESHOLDS["global_failure_count_min"],
    }
    return {
        "task_summaries": task_summaries,
        "informative_task_count": informative_tasks,
        "global_transition_count": successes,
        "global_non_transition_count": failures,
        "pass_conditions": pass_conditions,
        "primary_endpoint_pass": all(pass_conditions.values()),
    }


def protocol_command() -> None:
    if OUT_ROOT.exists():
        raise RuntimeError("refusing to overwrite existing Phase1170 output")
    prior_final = base.read_json(P1169_FINAL)
    prior_audit = base.read_json(P1169_AUDIT)
    if prior_final["decision"]["primary_endpoint_pass"]:
        raise RuntimeError("Phase1169 was expected to have a failed robust split endpoint")
    if not prior_audit["overall_pass"]:
        raise RuntimeError("Phase1169 independent audit did not pass")
    allocation = []
    for task_index, (task_name, modulus) in enumerate(TASKS.items()):
        for replicate in range(REPLICATES):
            seed = model_seed(task_index, replicate)
            data = base.make_data(modulus, seed + 17)
            allocation.append({
                "task_name": task_name,
                "task_index": task_index,
                "modulus": modulus,
                "replicate": replicate,
                "seed": seed,
                "train_pair_digest": base.digest(data["train_pairs"].tolist()),
                "sealed_holdout_pair_digest": base.digest(data["holdout_pairs"].tolist()),
                "train_case_count": len(data["train_x"]),
                "holdout_case_count": len(data["holdout_x"]),
            })
    protocol = {
        "phase": PHASE,
        "created_at_utc": base.utc_now(),
        "question": "Does the unchanged Phase1169 learner produce a broad, mixed panel of natural memorizer-to-generalizer transitions on randomly frozen fresh tasks?",
        "authorization": "The user's Phase1168-1169 request explicitly authorizes systematic automatic continuation after the failed Phase1169 robust split endpoint.",
        "prerequisite": {
            "phase1169_final_sha256": base.sha256_file(P1169_FINAL),
            "phase1169_audit_sha256": base.sha256_file(P1169_AUDIT),
            "phase1169_primary_endpoint_pass": prior_final["decision"]["primary_endpoint_pass"],
            "phase1169_audit_pass": prior_audit["overall_pass"],
        },
        "source_hashes": {
            "primary_script": base.sha256_file(SCRIPT),
            "audit_script": base.sha256_file(AUDIT_SCRIPT),
        },
        "task_selection": {
            "eligible_prime_pool": PRIME_POOL,
            "selection_seed": TASK_SELECTION_SEED,
            "random_permutation": task_permutation(),
            "selected_moduli": SELECTED_MODULI,
            "reserved_fresh_moduli": RESERVED_MODULI,
            "selection_rule": "Take the first four positions of one deterministic full random permutation; reserve the rest without model evaluation.",
        },
        "tasks": TASKS,
        "replicates_per_task": REPLICATES,
        "trajectory_count": len(TASKS) * REPLICATES,
        "allocation": allocation,
        "checkpoint_steps": CHECKPOINT_STEPS,
        "train_fraction": TRAIN_FRACTION,
        "model": {
            "class": "SymmetricSquareNetwork",
            "width": MODEL_WIDTH,
            "unchanged_from_phase1169": True,
        },
        "training": TRAINING,
        "trajectory_thresholds": TRAJECTORY_THRESHOLDS,
        "breadth_thresholds": BREADTH_THRESHOLDS,
        "sealed_rules": [
            "No held-out logits, losses, labels, or accuracies are computed during training.",
            "Every checkpoint and training-only structural record is sealed before any held-out outcome exists.",
            "The Phase1169 exploratory K144 features cannot select tasks, seeds, checkpoints, or thresholds in Phase1170.",
            "No task may be replaced because it looks too easy, too hard, or numerically inconvenient.",
            "A pass only authorizes a separately preregistered prospective predictor on the reserved fresh tasks.",
            "A fail closes this square-network natural-selection line under the frozen regime; no modulus search is authorized.",
            "Hidden-state selection, component intervention, and mechanism claims are forbidden in Phase1170.",
        ],
    }
    protocol["protocol_digest"] = base.digest(protocol)
    base.write_json(OUT_ROOT / "protocol/preregistration.json", protocol)
    print(json.dumps({"protocol_digest": protocol["protocol_digest"], "selected_moduli": SELECTED_MODULI, "reserved_moduli": RESERVED_MODULI}))


def train_and_seal_command() -> None:
    protocol = base.read_json(OUT_ROOT / "protocol/preregistration.json")
    if (OUT_ROOT / "runs/training/seal.json").exists():
        raise RuntimeError("training is already sealed")
    if (OUT_ROOT / "runs/holdout").exists():
        raise RuntimeError("holdout outcomes exist before training seal")
    device = torch.device("cuda")
    rows: list[dict[str, Any]] = []
    checkpoints: dict[str, str] = {}
    for task_index, (task_name, modulus) in enumerate(TASKS.items()):
        for replicate in range(REPLICATES):
            seed = model_seed(task_index, replicate)
            base.set_seed(seed)
            data = base.make_data(modulus, seed + 17)
            model = base.SymmetricSquareNetwork(base.SquareConfig(modulus=modulus)).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=TRAINING["learning_rate"], weight_decay=TRAINING["weight_decay"])
            train_x_device = data["train_x"].to(device)
            train_y_device = data["train_y"].to(device)
            for step in range(1, max(CHECKPOINT_STEPS) + 1):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(train_x_device).float()
                    loss = F.cross_entropy(logits, train_y_device)
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError(f"nonfinite loss: {task_name}/{replicate}/{step}")
                loss.backward()
                optimizer.step()
                if step not in CHECKPOINT_STEPS:
                    continue
                train_metrics = base.evaluate(model, data["train_x"], data["train_y"], device)
                structure = base.training_only_structure(model, data, device)
                trajectory_id = f"{task_name}_r{replicate}_s{seed}"
                checkpoint_id = f"{trajectory_id}_step{step:05d}"
                checkpoint_path = OUT_ROOT / "runs/training/checkpoints" / f"{checkpoint_id}.pt"
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(checkpoint_payload(model, task_name, task_index, replicate, seed, step), checkpoint_path)
                checkpoint_hash = base.sha256_file(checkpoint_path)
                checkpoints[checkpoint_id] = checkpoint_hash
                rows.append({
                    "trajectory_id": trajectory_id,
                    "checkpoint_id": checkpoint_id,
                    "task_name": task_name,
                    "task_index": task_index,
                    "modulus": modulus,
                    "replicate": replicate,
                    "seed": seed,
                    "step": step,
                    "loss": float(loss.item()),
                    "train": train_metrics,
                    "training_only_structure": structure,
                    "train_pair_digest": base.digest(data["train_pairs"].tolist()),
                    "sealed_holdout_pair_digest": base.digest(data["holdout_pairs"].tolist()),
                    "checkpoint_sha256": checkpoint_hash,
                    "holdout_evaluated_during_training": False,
                    "holdout_used_by_gradient": False,
                })
            print(json.dumps({"trained": trajectory_id, "checkpoints": len(CHECKPOINT_STEPS)}), flush=True)
            del model, optimizer, train_x_device, train_y_device
            gc.collect()
            torch.cuda.empty_cache()
    metrics_path = OUT_ROOT / "runs/training/training_metrics.jsonl"
    base.write_jsonl(metrics_path, rows)
    seal = {
        "phase": PHASE,
        "sealed_at_utc": base.utc_now(),
        "protocol_digest": protocol["protocol_digest"],
        "trajectory_count": len(TASKS) * REPLICATES,
        "checkpoint_count": len(rows),
        "training_metrics_sha256": base.sha256_file(metrics_path),
        "checkpoint_hashes": checkpoints,
        "holdout_outcomes_absent_at_sealing": not (OUT_ROOT / "runs/holdout").exists(),
        "no_holdout_evaluated": all(not row["holdout_evaluated_during_training"] for row in rows),
        "no_holdout_gradient": all(not row["holdout_used_by_gradient"] for row in rows),
        "training_sealed": True,
    }
    seal["seal_digest"] = base.digest(seal)
    base.write_json(OUT_ROOT / "runs/training/seal.json", seal)
    print(json.dumps({"seal_digest": seal["seal_digest"], "trajectories": seal["trajectory_count"], "checkpoints": seal["checkpoint_count"]}))


def evaluate_holdout_command() -> None:
    seal = base.read_json(OUT_ROOT / "runs/training/seal.json")
    if not seal["training_sealed"] or not seal["holdout_outcomes_absent_at_sealing"]:
        raise RuntimeError("invalid training seal")
    root = OUT_ROOT / "runs/holdout"
    if root.exists():
        raise RuntimeError("refusing to overwrite held-out outcomes")
    training_rows = base.read_jsonl(OUT_ROOT / "runs/training/training_metrics.jsonl")
    device = torch.device("cuda")
    rows: list[dict[str, Any]] = []
    for train_row in training_rows:
        checkpoint_id = train_row["checkpoint_id"]
        checkpoint_path = OUT_ROOT / "runs/training/checkpoints" / f"{checkpoint_id}.pt"
        if base.sha256_file(checkpoint_path) != train_row["checkpoint_sha256"]:
            raise RuntimeError(f"checkpoint hash mismatch: {checkpoint_id}")
        data = base.make_data(train_row["modulus"], train_row["seed"] + 17)
        model = load_checkpoint(checkpoint_path, device)
        metrics = base.evaluate(model, data["holdout_x"], data["holdout_y"], device)
        rows.append({
            "trajectory_id": train_row["trajectory_id"],
            "checkpoint_id": checkpoint_id,
            "task_name": train_row["task_name"],
            "task_index": train_row["task_index"],
            "modulus": train_row["modulus"],
            "replicate": train_row["replicate"],
            "seed": train_row["seed"],
            "step": train_row["step"],
            "train": train_row["train"],
            "training_only_structure": train_row["training_only_structure"],
            "holdout": metrics,
        })
        del model
    output_path = root / "holdout_metrics.jsonl"
    base.write_jsonl(output_path, rows)
    summary = {
        "phase": PHASE,
        "evaluated_at_utc": base.utc_now(),
        "seal_digest": seal["seal_digest"],
        "row_count": len(rows),
        "finite": all(row["holdout"]["finite_fraction"] >= TRAJECTORY_THRESHOLDS["finite_fraction_min"] for row in rows),
        "holdout_metrics_sha256": base.sha256_file(output_path),
    }
    summary["summary_digest"] = base.digest(summary)
    base.write_json(root / "summary.json", summary)
    print(json.dumps({"summary_digest": summary["summary_digest"], "rows": len(rows), "finite": summary["finite"]}))


def score_command() -> None:
    holdout_summary = base.read_json(OUT_ROOT / "runs/holdout/summary.json")
    rows = base.read_jsonl(OUT_ROOT / "runs/holdout/holdout_metrics.jsonl")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["trajectory_id"], []).append(row)
    trajectories = [trajectory_summary(group) for group in grouped.values()]
    decision = breadth_decision(trajectories)
    score = {
        "phase": PHASE,
        "scored_at_utc": base.utc_now(),
        "holdout_summary_digest": holdout_summary["summary_digest"],
        "trajectory_count": len(trajectories),
        "trajectories": sorted(trajectories, key=lambda row: (row["task_index"], row["replicate"])),
        **decision,
        "interpretation": {
            "if_pass": "The unchanged regime produced a sufficiently broad mixed outcome panel to support a separately preregistered prospective predictor test on reserved fresh tasks.",
            "if_fail": "The frozen regime did not produce a broad mixed panel; no predictor fitting, modulus search, or hidden scan is authorized.",
            "scope": "This endpoint concerns behavioral formation breadth in a controlled square network, not language, mechanism identity, or intelligent theory closure.",
        },
    }
    score["score_digest"] = base.digest(score)
    base.write_json(OUT_ROOT / "analysis/score.json", score)
    print(json.dumps({
        "primary_endpoint_pass": score["primary_endpoint_pass"],
        "informative_task_count": score["informative_task_count"],
        "global_transition_count": score["global_transition_count"],
        "global_non_transition_count": score["global_non_transition_count"],
        "score_digest": score["score_digest"],
    }))


def finalize_command() -> None:
    protocol = base.read_json(OUT_ROOT / "protocol/preregistration.json")
    seal = base.read_json(OUT_ROOT / "runs/training/seal.json")
    score = base.read_json(OUT_ROOT / "analysis/score.json")
    passed = bool(score["primary_endpoint_pass"])
    final = {
        "phase": PHASE,
        "finalized_at_utc": base.utc_now(),
        "protocol_digest": protocol["protocol_digest"],
        "seal_digest": seal["seal_digest"],
        "score_digest": score["score_digest"],
        "decision": {
            "primary_endpoint_pass": passed,
            "broad_mixed_formation_panel_exists": passed,
            "prospective_predictor_phase_authorized": passed,
            "hidden_scan_authorized": False,
            "mechanism_claim_authorized": False,
            "modulus_search_authorized": False,
            "auto_continue": passed,
            "authorized_next": "Phase1171: freeze a simple training-only predictor on Phase1170, then test it once on the untouched reserved moduli" if passed else None,
        },
        "claims": [
            "Task allocation was frozen by deterministic random permutation before training.",
            "The architecture, optimizer, precision, schedule, split fraction, and transition thresholds match Phase1169.",
            "K144 exploratory signatures were recorded but did not select or alter any Phase1170 condition.",
            "A positive breadth endpoint would authorize prediction, not establish a causal mechanism.",
        ],
    }
    final["final_digest"] = base.digest(final)
    base.write_json(OUT_ROOT / "analysis/final.json", final)
    print(json.dumps({"final_digest": final["final_digest"], "auto_continue": final["decision"]["auto_continue"]}))


def smoke_command() -> None:
    if task_permutation() != (41, 19, 59, 61, 23, 47, 43, 53):
        raise RuntimeError("unexpected task permutation")
    if set(SELECTED_MODULI).intersection(RESERVED_MODULI):
        raise RuntimeError("selected and reserved tasks overlap")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    for task_index, (task_name, modulus) in enumerate(TASKS.items()):
        seed = model_seed(task_index, 0)
        data = base.make_data(modulus, seed + 17)
        overlap = set(map(tuple, data["train_pairs"].tolist())).intersection(map(tuple, data["holdout_pairs"].tolist()))
        if overlap:
            raise RuntimeError("train/holdout overlap")
        print(json.dumps({"task_name": task_name, "modulus": modulus, "train": len(data["train_x"]), "holdout": len(data["holdout_x"])}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("protocol", "train-and-seal", "evaluate-holdout", "score", "finalize", "smoke"))
    args = parser.parse_args()
    commands = {
        "protocol": protocol_command,
        "train-and-seal": train_and_seal_command,
        "evaluate-holdout": evaluate_holdout_command,
        "score": score_command,
        "finalize": finalize_command,
        "smoke": smoke_command,
    }
    commands[args.command]()


if __name__ == "__main__":
    main()
