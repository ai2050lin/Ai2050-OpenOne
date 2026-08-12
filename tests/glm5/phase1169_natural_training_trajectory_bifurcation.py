#!/usr/bin/env python3
"""Sealed training-trajectory test for emergent modular generalization.

Phase1168 showed that the local constraints used in Phase1167 did not identify
the held-out extension.  This phase therefore stops forcing a preferred rule.
It trains one unchanged network and optimizer through time, seals every
checkpoint before any held-out evaluation, and asks whether a memorizing state
naturally becomes a generalizing state.  Training-domain structural summaries
are saved for a later, fresh-task prediction experiment; they are exploratory
here and are not interpreted as mechanisms.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve()
AUDIT_SCRIPT = ROOT / "tests/glm5/phase1169_natural_training_trajectory_bifurcation_audit.py"
OUT_ROOT = ROOT / "tests/glm5/result/phase1169_natural_training_trajectory_bifurcation"
P1168_FINAL = ROOT / "tests/glm5/result/phase1168_rule_extension_identifiability_audit/analysis/final.json"
P1168_AUDIT = ROOT / "tests/glm5/result/phase1168_rule_extension_identifiability_audit/audit/report.json"

PHASE = 1169
TASKS = {"discovery": 29, "confirmation": 37}
REPLICATES = 4
CHECKPOINT_STEPS = (100, 250, 500, 1000, 1500, 2000, 2500, 3000, 4000, 6000, 10000)
TRAIN_FRACTION = 0.50
MODEL_WIDTH = 128
TRAINING = {
    "learning_rate": 0.001,
    "weight_decay": 1.0,
    "precision": "bfloat16",
    "batching": "full_batch",
    "maximum_step": max(CHECKPOINT_STEPS),
}
THRESHOLDS = {
    "train_accuracy_min": 0.99,
    "memorizer_holdout_accuracy_max": 0.60,
    "generalizer_holdout_accuracy_min": 0.90,
    "stable_generalizer_checkpoint_count_min": 2,
    "successful_trajectories_per_split_min": 3,
    "finite_fraction_min": 1.0,
}


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical(row) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def model_seed(split: str, replicate: int) -> int:
    return 11690000 + list(TASKS).index(split) * 100_003 + int(replicate) * 1_009


@dataclass(frozen=True)
class SquareConfig:
    modulus: int
    width: int = MODEL_WIDTH


class SymmetricSquareNetwork(nn.Module):
    """Small unconstrained learner with a Fourier-capable nonlinear path."""

    def __init__(self, config: SquareConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.modulus, config.width)
        self.hidden = nn.Linear(config.width, config.width, bias=False)
        self.output = nn.Linear(config.width, config.modulus, bias=False)
        for module in (self.embedding, self.hidden, self.output):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, return_hidden: bool = False) -> Any:
        operands = input_ids[:, (1, 3)]
        summed = self.embedding(operands).sum(dim=1)
        pre_square = self.hidden(summed)
        hidden = pre_square.square()
        logits = self.output(hidden)
        if return_hidden:
            return logits, {"summed": summed, "pre_square": pre_square, "hidden": hidden}
        return logits


def make_data(modulus: int, seed: int) -> dict[str, torch.Tensor]:
    pairs = [(a, b) for a in range(modulus) for b in range(modulus)]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(pairs))
    cutoff = int(round(len(pairs) * TRAIN_FRACTION))
    train_indices = set(int(value) for value in order[:cutoff])
    bos, plus, equals = modulus, modulus + 1, modulus + 2
    rows, labels, pair_rows, mask = [], [], [], []
    for index, (a, b) in enumerate(pairs):
        rows.append([bos, a, plus, b, equals])
        labels.append((a + b) % modulus)
        pair_rows.append([a, b])
        mask.append(index in train_indices)
    x = torch.tensor(rows, dtype=torch.long)
    y = torch.tensor(labels, dtype=torch.long)
    pair_tensor = torch.tensor(pair_rows, dtype=torch.long)
    train_mask = torch.tensor(mask, dtype=torch.bool)
    return {
        "train_x": x[train_mask],
        "train_y": y[train_mask],
        "train_pairs": pair_tensor[train_mask],
        "holdout_x": x[~train_mask],
        "holdout_y": y[~train_mask],
        "holdout_pairs": pair_tensor[~train_mask],
    }


@torch.inference_mode()
def evaluate(
    model: SymmetricSquareNetwork,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(inputs.to(device)).float()
    probabilities = torch.softmax(logits, dim=-1)
    predicted = logits.argmax(dim=-1).cpu()
    target_probability = probabilities.gather(1, targets.to(device)[:, None]).squeeze(1)
    finite = torch.isfinite(logits)
    return {
        "case_count": len(targets),
        "accuracy": float((predicted == targets).float().mean().item()),
        "mean_target_probability": float(target_probability.mean().item()),
        "minimum_target_probability": float(target_probability.min().item()),
        "finite_fraction": float(finite.float().mean().item()),
    }


def circulant_gram_score(matrix: torch.Tensor) -> float:
    values = matrix.detach().float().cpu()
    values = values - values.mean(dim=0, keepdim=True)
    gram = values @ values.T
    modulus = gram.shape[0]
    fitted = torch.empty_like(gram)
    for delta in range(modulus):
        cells = torch.stack([gram[index, (index + delta) % modulus] for index in range(modulus)])
        mean = cells.mean()
        for index in range(modulus):
            fitted[index, (index + delta) % modulus] = mean
    denominator = float(torch.sum((gram - gram.mean()).square()).item())
    if denominator <= 1e-20:
        return 0.0
    residual = float(torch.sum((gram - fitted).square()).item())
    return float(max(-1.0, min(1.0, 1.0 - residual / denominator)))


def fourier_top_share(matrix: torch.Tensor, top_k: int = 4) -> float:
    values = matrix.detach().float().cpu()
    values = values - values.mean(dim=0, keepdim=True)
    power = torch.fft.rfft(values, dim=0).abs().square().sum(dim=1)
    power = power[1:]
    total = float(power.sum().item())
    if total <= 1e-20:
        return 0.0
    return float(torch.topk(power, k=min(top_k, len(power))).values.sum().item() / total)


@torch.inference_mode()
def local_equivariance_score(
    model: SymmetricSquareNetwork,
    train_x: torch.Tensor,
    train_pairs: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(train_x.to(device)).float().cpu()
    centered = logits - logits.mean(dim=1, keepdim=True)
    lookup = {tuple(pair): index for index, pair in enumerate(train_pairs.tolist())}
    cosines: list[float] = []
    path_cosines: list[float] = []
    modulus = model.config.modulus
    for index, (a, b) in enumerate(train_pairs.tolist()):
        shifted_a = lookup.get(((a + 1) % modulus, b))
        shifted_b = lookup.get((a, (b + 1) % modulus))
        expected = torch.roll(centered[index], shifts=1)
        if shifted_a is not None:
            cosines.append(float(F.cosine_similarity(centered[shifted_a], expected, dim=0).item()))
        if shifted_b is not None:
            cosines.append(float(F.cosine_similarity(centered[shifted_b], expected, dim=0).item()))
        if shifted_a is not None and shifted_b is not None:
            path_cosines.append(
                float(F.cosine_similarity(centered[shifted_a], centered[shifted_b], dim=0).item())
            )
    return {
        "local_equivariance_cosine": float(np.mean(cosines)),
        "local_equivariance_edge_count": len(cosines),
        "path_independence_cosine": float(np.mean(path_cosines)),
        "path_independence_cell_count": len(path_cosines),
    }


def training_only_structure(
    model: SymmetricSquareNetwork,
    data: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, Any]:
    transformed_embedding = F.linear(model.embedding.weight.detach().float(), model.hidden.weight.detach().float())
    output_basis = model.output.weight.detach().float()
    total_norm = math.sqrt(sum(float(parameter.detach().float().square().sum().item()) for parameter in model.parameters()))
    result = {
        "embedding_circulant_gram": circulant_gram_score(transformed_embedding),
        "output_circulant_gram": circulant_gram_score(output_basis),
        "embedding_fourier_top4_share": fourier_top_share(transformed_embedding),
        "output_fourier_top4_share": fourier_top_share(output_basis),
        "parameter_l2_norm": total_norm,
    }
    result.update(local_equivariance_score(model, data["train_x"], data["train_pairs"], device))
    return result


def checkpoint_payload(model: SymmetricSquareNetwork, split: str, replicate: int, seed: int, step: int) -> dict[str, Any]:
    return {
        "phase": PHASE,
        "split": split,
        "replicate": replicate,
        "seed": seed,
        "step": step,
        "config": asdict(model.config),
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
    }


def load_checkpoint(path: Path, device: torch.device) -> SymmetricSquareNetwork:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = SymmetricSquareNetwork(SquareConfig(**payload["config"])).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def protocol_command() -> None:
    if OUT_ROOT.exists():
        raise RuntimeError("refusing to overwrite existing Phase1169 output")
    prior_final = read_json(P1168_FINAL)
    prior_audit = read_json(P1168_AUDIT)
    if not prior_final["primary_endpoint_passed"] or not prior_audit["all_checks_passed"]:
        raise RuntimeError("Phase1168 prerequisite did not pass")
    task_a = make_data(TASKS["discovery"], model_seed("discovery", 0) + 17)
    task_b = make_data(TASKS["confirmation"], model_seed("confirmation", 0) + 17)
    protocol = {
        "phase": PHASE,
        "created_at_utc": utc_now(),
        "question": "Can an unchanged learner naturally move from memorization to held-out modular generalization through training time?",
        "prerequisite": {
            "phase1168_final_sha256": sha256_file(P1168_FINAL),
            "phase1168_audit_sha256": sha256_file(P1168_AUDIT),
        },
        "source_hashes": {
            "primary_script": sha256_file(SCRIPT),
            "audit_script": sha256_file(AUDIT_SCRIPT),
        },
        "tasks": TASKS,
        "replicates": REPLICATES,
        "checkpoint_steps": CHECKPOINT_STEPS,
        "train_fraction": TRAIN_FRACTION,
        "model": {"class": "SymmetricSquareNetwork", "width": MODEL_WIDTH, "note": "same architecture family; modulus only changes vocabulary/output cardinality"},
        "training": TRAINING,
        "thresholds": THRESHOLDS,
        "pilot_exclusion": "modulus 31 and all pilot seeds/configuration traces are engineering-only and excluded from evidence",
        "sealed_rules": [
            "No held-out logits, labels, losses, or accuracies are computed during training.",
            "All checkpoints and training-only summaries are sealed before the held-out directory is created.",
            "Training-only structural summaries are exploratory and cannot establish mechanism identity in this phase.",
            "A successful trajectory requires an earlier memorizer and at least two later generalizer checkpoints.",
            "Hidden-state or component intervention is forbidden in Phase1169.",
        ],
        "smoke_shapes": {
            "discovery_train": list(task_a["train_x"].shape),
            "discovery_holdout": list(task_a["holdout_x"].shape),
            "confirmation_train": list(task_b["train_x"].shape),
            "confirmation_holdout": list(task_b["holdout_x"].shape),
        },
    }
    protocol["protocol_digest"] = digest(protocol)
    write_json(OUT_ROOT / "protocol/preregistration.json", protocol)
    print(json.dumps({"protocol_digest": protocol["protocol_digest"], "output": str(OUT_ROOT)}))


def train_and_seal_command() -> None:
    protocol = read_json(OUT_ROOT / "protocol/preregistration.json")
    if (OUT_ROOT / "runs/training/seal.json").exists():
        raise RuntimeError("training is already sealed")
    if (OUT_ROOT / "runs/holdout").exists():
        raise RuntimeError("holdout outcomes exist before training seal")
    device = torch.device("cuda")
    rows: list[dict[str, Any]] = []
    checkpoints: dict[str, str] = {}
    for split, modulus in TASKS.items():
        for replicate in range(REPLICATES):
            seed = model_seed(split, replicate)
            set_seed(seed)
            data = make_data(modulus, seed + 17)
            model = SymmetricSquareNetwork(SquareConfig(modulus=modulus)).to(device)
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
                    raise RuntimeError(f"nonfinite loss: {split}/{replicate}/{step}")
                loss.backward()
                optimizer.step()
                if step not in CHECKPOINT_STEPS:
                    continue
                train_metrics = evaluate(model, data["train_x"], data["train_y"], device)
                structure = training_only_structure(model, data, device)
                trajectory_id = f"{split}_m{modulus}_r{replicate}_s{seed}"
                checkpoint_id = f"{trajectory_id}_step{step:05d}"
                checkpoint_path = OUT_ROOT / "runs/training/checkpoints" / f"{checkpoint_id}.pt"
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(checkpoint_payload(model, split, replicate, seed, step), checkpoint_path)
                checkpoint_hash = sha256_file(checkpoint_path)
                checkpoints[checkpoint_id] = checkpoint_hash
                rows.append({
                    "trajectory_id": trajectory_id,
                    "checkpoint_id": checkpoint_id,
                    "split": split,
                    "modulus": modulus,
                    "replicate": replicate,
                    "seed": seed,
                    "step": step,
                    "loss": float(loss.item()),
                    "train": train_metrics,
                    "training_only_structure": structure,
                    "train_pair_digest": digest(data["train_pairs"].tolist()),
                    "sealed_holdout_pair_digest": digest(data["holdout_pairs"].tolist()),
                    "checkpoint_sha256": checkpoint_hash,
                    "holdout_evaluated_during_training": False,
                    "holdout_used_by_gradient": False,
                })
            del model, optimizer, train_x_device, train_y_device
            gc.collect()
            torch.cuda.empty_cache()
    metrics_path = OUT_ROOT / "runs/training/training_metrics.jsonl"
    write_jsonl(metrics_path, rows)
    seal = {
        "phase": PHASE,
        "sealed_at_utc": utc_now(),
        "protocol_digest": protocol["protocol_digest"],
        "trajectory_count": len(TASKS) * REPLICATES,
        "checkpoint_count": len(rows),
        "training_metrics_sha256": sha256_file(metrics_path),
        "checkpoint_hashes": checkpoints,
        "holdout_outcomes_absent_at_sealing": not (OUT_ROOT / "runs/holdout").exists(),
        "no_holdout_evaluated": all(not row["holdout_evaluated_during_training"] for row in rows),
        "no_holdout_gradient": all(not row["holdout_used_by_gradient"] for row in rows),
        "training_sealed": True,
    }
    seal["seal_digest"] = digest(seal)
    write_json(OUT_ROOT / "runs/training/seal.json", seal)
    print(json.dumps({"seal_digest": seal["seal_digest"], "checkpoints": len(rows)}))


def evaluate_holdout_command() -> None:
    seal = read_json(OUT_ROOT / "runs/training/seal.json")
    if not seal["training_sealed"] or not seal["holdout_outcomes_absent_at_sealing"]:
        raise RuntimeError("invalid training seal")
    root = OUT_ROOT / "runs/holdout"
    if root.exists():
        raise RuntimeError("refusing to overwrite held-out outcomes")
    training_rows = read_jsonl(OUT_ROOT / "runs/training/training_metrics.jsonl")
    device = torch.device("cuda")
    rows = []
    for train_row in training_rows:
        checkpoint_id = train_row["checkpoint_id"]
        checkpoint_path = OUT_ROOT / "runs/training/checkpoints" / f"{checkpoint_id}.pt"
        if sha256_file(checkpoint_path) != train_row["checkpoint_sha256"]:
            raise RuntimeError(f"checkpoint hash mismatch: {checkpoint_id}")
        data = make_data(train_row["modulus"], train_row["seed"] + 17)
        model = load_checkpoint(checkpoint_path, device)
        metrics = evaluate(model, data["holdout_x"], data["holdout_y"], device)
        rows.append({
            "trajectory_id": train_row["trajectory_id"],
            "checkpoint_id": checkpoint_id,
            "split": train_row["split"],
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
    write_jsonl(output_path, rows)
    summary = {
        "phase": PHASE,
        "evaluated_at_utc": utc_now(),
        "seal_digest": seal["seal_digest"],
        "row_count": len(rows),
        "finite": all(row["holdout"]["finite_fraction"] >= THRESHOLDS["finite_fraction_min"] for row in rows),
        "holdout_metrics_sha256": sha256_file(output_path),
    }
    summary["summary_digest"] = digest(summary)
    write_json(root / "summary.json", summary)
    print(json.dumps({"summary_digest": summary["summary_digest"], "rows": len(rows)}))


def trajectory_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: row["step"])
    memorizers = [
        row for row in ordered
        if row["train"]["accuracy"] >= THRESHOLDS["train_accuracy_min"]
        and row["holdout"]["accuracy"] <= THRESHOLDS["memorizer_holdout_accuracy_max"]
    ]
    generalizers = [
        row for row in ordered
        if row["train"]["accuracy"] >= THRESHOLDS["train_accuracy_min"]
        and row["holdout"]["accuracy"] >= THRESHOLDS["generalizer_holdout_accuracy_min"]
    ]
    valid_pairs = [(m, g) for m in memorizers for g in generalizers if m["step"] < g["step"]]
    selected = min(valid_pairs, key=lambda pair: (pair[1]["step"], -pair[0]["step"])) if valid_pairs else None
    stable_count = 0
    for index in range(len(ordered) - 1):
        if ordered[index] in generalizers and ordered[index + 1] in generalizers:
            stable_count = max(stable_count, 2)
    success = selected is not None and stable_count >= THRESHOLDS["stable_generalizer_checkpoint_count_min"]
    return {
        "trajectory_id": ordered[0]["trajectory_id"],
        "split": ordered[0]["split"],
        "modulus": ordered[0]["modulus"],
        "replicate": ordered[0]["replicate"],
        "seed": ordered[0]["seed"],
        "memorizer_checkpoint_count": len(memorizers),
        "generalizer_checkpoint_count": len(generalizers),
        "stable_generalizer_pair_present": stable_count >= 2,
        "transition_present": success,
        "memorizer_step": selected[0]["step"] if selected else None,
        "memorizer_holdout_accuracy": selected[0]["holdout"]["accuracy"] if selected else None,
        "generalizer_step": selected[1]["step"] if selected else None,
        "generalizer_holdout_accuracy": selected[1]["holdout"]["accuracy"] if selected else None,
        "maximum_holdout_accuracy": max(row["holdout"]["accuracy"] for row in ordered),
        "final_holdout_accuracy": ordered[-1]["holdout"]["accuracy"],
    }


def score_command() -> None:
    holdout_summary = read_json(OUT_ROOT / "runs/holdout/summary.json")
    rows = read_jsonl(OUT_ROOT / "runs/holdout/holdout_metrics.jsonl")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["trajectory_id"], []).append(row)
    trajectories = [trajectory_summary(group) for group in grouped.values()]
    split_counts = {
        split: sum(row["transition_present"] for row in trajectories if row["split"] == split)
        for split in TASKS
    }
    primary_pass = all(
        count >= THRESHOLDS["successful_trajectories_per_split_min"] for count in split_counts.values()
    )
    score = {
        "phase": PHASE,
        "scored_at_utc": utc_now(),
        "holdout_summary_digest": holdout_summary["summary_digest"],
        "trajectory_count": len(trajectories),
        "trajectories": sorted(trajectories, key=lambda row: (row["split"], row["replicate"])),
        "split_transition_counts": split_counts,
        "primary_endpoint_pass": primary_pass,
        "interpretation": {
            "if_pass": "The same learner can naturally cross from memorization to generalization through training time on two fresh modular tasks.",
            "if_fail": "This frozen regime did not provide a robust natural behavioral bifurcation; no process predictor or hidden scan is authorized.",
            "scope": "A pass establishes a controlled formation object, not a language mechanism and not a unique internal code.",
        },
    }
    score["score_digest"] = digest(score)
    write_json(OUT_ROOT / "analysis/score.json", score)
    print(json.dumps({"primary_endpoint_pass": primary_pass, "split_transition_counts": split_counts, "score_digest": score["score_digest"]}))


def finalize_command() -> None:
    protocol = read_json(OUT_ROOT / "protocol/preregistration.json")
    seal = read_json(OUT_ROOT / "runs/training/seal.json")
    score = read_json(OUT_ROOT / "analysis/score.json")
    final = {
        "phase": PHASE,
        "finalized_at_utc": utc_now(),
        "protocol_digest": protocol["protocol_digest"],
        "seal_digest": seal["seal_digest"],
        "score_digest": score["score_digest"],
        "decision": {
            "primary_endpoint_pass": score["primary_endpoint_pass"],
            "natural_trajectory_object_exists": score["primary_endpoint_pass"],
            "hidden_scan_authorized": False,
            "mechanism_claim_authorized": False,
            "auto_continue": score["primary_endpoint_pass"],
            "authorized_next": "fresh-task preregistered prediction from training-only process signatures" if score["primary_endpoint_pass"] else None,
        },
        "claims": [
            "Any successful transition is temporal and behavioral: the architecture, data, loss, and optimizer remain unchanged within a trajectory.",
            "Phase1169 structural summaries are exploratory correlations and cannot be promoted to mechanisms.",
            "The square network is a controlled Fourier-capable learner; external validity to deep language models is untested.",
        ],
    }
    final["final_digest"] = digest(final)
    write_json(OUT_ROOT / "analysis/final.json", final)
    print(json.dumps({"final_digest": final["final_digest"], "auto_continue": final["decision"]["auto_continue"]}))


def smoke_command() -> None:
    for split, modulus in TASKS.items():
        data = make_data(modulus, model_seed(split, 0) + 17)
        overlap = set(map(tuple, data["train_pairs"].tolist())).intersection(map(tuple, data["holdout_pairs"].tolist()))
        if overlap:
            raise RuntimeError("train/holdout overlap")
        print(json.dumps({"split": split, "modulus": modulus, "train": len(data["train_x"]), "holdout": len(data["holdout_x"])}))


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
