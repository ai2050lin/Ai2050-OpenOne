from __future__ import annotations

import argparse
import gc
import json
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import phase1148_mandatory_mediation_calibration as p1148


PHASE = 1149
ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "tests" / "glm5" / "result" / "phase1149_role_factorized_mediation"
TEMP_ROOT = ROOT / "tests" / "glm5_temp" / "phase1149_smoke"
PREREG_PATH = OUT_ROOT / "protocol" / "preregistration.json"
CONDITIONS = {
    "answer_boundary": {"row_position": -1, "column_position": -1},
    "role_factorized": {"row_position": -3, "column_position": -2},
}
CONDITION_ORDER = ["answer_boundary", "role_factorized"]


class RoleMediationModel(nn.Module):
    def __init__(self, config: p1148.p1147.p1146.ModelConfig, readout_mode: str) -> None:
        super().__init__()
        if readout_mode not in CONDITIONS:
            raise ValueError(f"Unknown readout mode: {readout_mode}")
        base = p1148.p1147.p1146
        self.config = config
        self.readout_mode = readout_mode
        self.backbone = base.TinyCausalTransformer(config)
        self.row_head = nn.Linear(config.width, base.RECORD_COUNT)
        self.column_head = nn.Linear(config.width, base.FIELD_COUNT)
        nn.init.normal_(self.row_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.row_head.bias)
        nn.init.normal_(self.column_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.column_head.bias)

    def forward_with_positions(
        self,
        input_ids: torch.Tensor,
        row_position: int,
        column_position: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, states = self.backbone(input_ids, return_states=True)
        normalized = self.backbone.final_norm(states[-1])
        return (
            logits,
            self.row_head(normalized[:, row_position, :]),
            self.column_head(normalized[:, column_position, :]),
        )

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        positions = CONDITIONS[self.readout_mode]
        return self.forward_with_positions(
            input_ids,
            int(positions["row_position"]),
            int(positions["column_position"]),
        )


def protocol_body() -> dict[str, Any]:
    parent = p1148.read_json(p1148.PREREG_PATH)
    p1148.verify_preregistration(parent)
    return {
        "phase": PHASE,
        "title": "Role-aligned row-column mediation versus answer-boundary fusion",
        "claim_scope": (
            "Tests whether assigning row and column addresses to their causally aligned query roles "
            "changes formation under identical parameters, initialization, data, batches, losses, and budget. "
            "The mediator and readout roles are planted; success cannot establish a natural Transformer mechanism."
        ),
        "parent_phase1148_protocol_digest": parent["protocol_digest"],
        "source_hashes": {
            "primary_script": p1148.file_sha256(Path(__file__).resolve()),
            "phase1148_dependency": p1148.file_sha256(Path(p1148.__file__).resolve()),
            "phase1147_dependency": p1148.file_sha256(Path(p1148.p1147.__file__).resolve()),
            "phase1146_dependency": p1148.file_sha256(
                Path(p1148.p1147.p1146.__file__).resolve()
            ),
        },
        "conditions": deepcopy(CONDITIONS),
        "condition_order": list(CONDITION_ORDER),
        "data": deepcopy(parent["data"]),
        "replicates": deepcopy(parent["replicates"]),
        "training_control": {
            "same_parameterization": True,
            "same_initial_state_per_replicate": True,
            "same_training_material_per_replicate": True,
            "same_batch_schedule_per_replicate": True,
            "same_answer_row_column_losses": True,
            "only_difference": "row_and_column_readout_token_positions",
        },
        "thresholds": {
            **deepcopy(parent["thresholds"]),
            "minimum_paired_accuracy_gain": 0.50,
            "position_ablation_drop": 0.50,
            "single_role_ablation_drop": 0.30,
        },
        "selection": {
            "discovery_requires_role_factorized_qualified_in_all_four_replicates": True,
            "discovery_requires_gain_on_holdout_and_quartet_in_every_replicate": True,
            "confirmation_uses_same_frozen_condition_and_four_new_replicates": True,
            "causal_validation_only_after_confirmation": True,
        },
        "causal_modes": {
            "normal": {"row_position": -3, "column_position": -2},
            "both_answer": {"row_position": -1, "column_position": -1},
            "row_answer": {"row_position": -1, "column_position": -2},
            "column_answer": {"row_position": -3, "column_position": -1},
            "swapped": {"row_position": -2, "column_position": -3},
            "oracle_both": "ground_truth_row_and_column",
        },
        "gate_policy": {
            "hard_stop": [
                "nonfinite training",
                "label-grid or oracle failure",
                "hash, recomputation, or pairing failure",
            ],
            "claim_stop": [
                "role-factorized formation failure",
                "paired gain failure",
                "independent confirmation failure",
                "position intervention failure",
            ],
            "branch_after_failure": (
                "Stop the role-aligned readout claim, retain trajectory data, and branch to "
                "content-address comparison or distributed-coalition calibration."
            ),
        },
        "forbidden": [
            "No development-probe cases or seeds enter formal evidence",
            "No post-freeze change to readout positions, losses, budget, or thresholds",
            "No confirmation unless every discovery replicate and paired-gain gate passes",
            "No natural-language or free-Transformer mechanism claim from the planted mediator",
            "No hidden-state hotspot selection in this phase",
        ],
    }


def verify_preregistration(prereg: dict[str, Any]) -> None:
    body = dict(prereg)
    digest = body.pop("protocol_digest")
    if p1148.canonical_digest(body) != digest:
        raise RuntimeError("Phase1149 protocol digest mismatch")
    hashes = prereg["source_hashes"]
    if p1148.file_sha256(Path(__file__).resolve()) != hashes["primary_script"]:
        raise RuntimeError("Phase1149 primary script changed after preregistration")
    if p1148.file_sha256(Path(p1148.__file__).resolve()) != hashes["phase1148_dependency"]:
        raise RuntimeError("Phase1148 dependency changed after Phase1149 preregistration")
    parent = p1148.read_json(p1148.PREREG_PATH)
    if parent["protocol_digest"] != prereg["parent_phase1148_protocol_digest"]:
        raise RuntimeError("Parent Phase1148 protocol changed")


def create_protocol() -> dict[str, Any]:
    body = protocol_body()
    prereg = dict(body)
    prereg["protocol_digest"] = p1148.canonical_digest(body)
    if PREREG_PATH.exists():
        if p1148.read_json(PREREG_PATH) != prereg:
            raise RuntimeError("Existing Phase1149 protocol differs from current script")
    else:
        p1148.write_json(PREREG_PATH, prereg)

    spec = prereg["replicates"]["discovery_small_r1"]
    dataset = p1148.make_dataset(
        2048,
        prereg["data"]["pairs"]["train"],
        int(spec["data_seeds"]["training"]),
        spec["lexicon"],
    )
    oracle = dataset["grid_values"][
        np.arange(len(dataset["targets"])),
        dataset["row_targets"],
        dataset["column_targets"],
    ]
    base = p1148.p1147.p1146
    decoded_suffix = [
        spec["lexicon"].index(int(token)) for token in dataset["inputs"][0, -4:]
    ]
    checks = {
        "two_conditions": set(prereg["conditions"]) == set(CONDITIONS),
        "eight_replicates": len(prereg["replicates"]) == 8,
        "oracle_exact": bool(np.all(oracle == dataset["targets"])),
        "query_suffix_roles": (
            decoded_suffix[0] == base.QUERY
            and base.ENTITY_START <= decoded_suffix[1] < base.FIELD_START
            and base.FIELD_START <= decoded_suffix[2] < base.VALUE_START
            and decoded_suffix[3] == base.ANSWER
        ),
        "role_positions": prereg["conditions"]["role_factorized"]
        == {"row_position": -3, "column_position": -2},
        "source_hash": p1148.file_sha256(Path(__file__).resolve())
        == prereg["source_hashes"]["primary_script"],
    }
    audit = {
        "phase": PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }
    audit["audit_digest"] = p1148.canonical_digest(audit)
    p1148.write_json(OUT_ROOT / "protocol" / "audit.json", audit)
    if not audit["all_checks_passed"]:
        raise RuntimeError(f"Phase1149 protocol audit failed: {checks}")
    return prereg


def split_replicates(prereg: dict[str, Any], split: str) -> list[str]:
    return [
        name for name, spec in prereg["replicates"].items() if spec["split"] == split
    ]


def save_checkpoint(
    model: RoleMediationModel,
    replicate: str,
    condition: str,
    split: str,
    step: int,
    prereg: dict[str, Any],
    output_root: Path = OUT_ROOT,
) -> dict[str, Any]:
    path = (
        output_root
        / "runs"
        / split
        / replicate
        / condition
        / "checkpoints"
        / f"step_{step:06d}.pt"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "phase": PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "replicate": replicate,
        "condition": condition,
        "step": step,
        "config": asdict(model.config),
        "readout_mode": model.readout_mode,
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
    }
    torch.save(payload, path)
    return {
        "step": step,
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "sha256": p1148.file_sha256(path),
        "state_digest": p1148.state_digest(model.state_dict()),
    }


def load_model(summary: dict[str, Any]) -> RoleMediationModel:
    checkpoint = torch.load(
        ROOT / summary["model_path"], map_location="cpu", weights_only=True
    )
    model = RoleMediationModel(
        p1148.p1147.p1146.ModelConfig(**checkpoint["config"]),
        str(checkpoint["readout_mode"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model.cuda().eval()


def train_condition(
    replicate: str,
    condition: str,
    prereg: dict[str, Any],
    output_root: Path = OUT_ROOT,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("Phase1149 requires CUDA")
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition: {condition}")
    spec = prereg["replicates"][replicate]
    if spec["split"] == "confirmation":
        selection_path = OUT_ROOT / "analysis" / "discovery_selection.json"
        if not selection_path.exists() or not p1148.read_json(selection_path)[
            "confirmation_authorized"
        ]:
            raise RuntimeError("Confirmation is not authorized")

    p1148.set_seed(int(spec["training_seed"]))
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    model = RoleMediationModel(
        p1148.p1147.p1146.ModelConfig(**spec["architecture"]), condition
    ).to(device)
    initial_digest = p1148.state_digest(model.state_dict())
    training_data, schedule = p1148.make_training_material(spec, prereg)
    training_digest = p1148.dataset_digest(training_data)
    schedule_digest = p1148.array_digest(schedule)
    cpu = {
        key: torch.from_numpy(training_data[key])
        for key in ("inputs", "targets", "row_targets", "column_targets", "grid_values")
    }
    training = spec["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        betas=(0.9, 0.95),
    )
    trajectory_steps = sorted(
        set(int(step) for step in training["trajectory_steps"])
        | {int(training["max_steps"])}
    )
    logs: list[dict[str, Any]] = []
    trajectory: list[dict[str, Any]] = []
    torch.cuda.reset_peak_memory_stats()
    checkpoint = save_checkpoint(
        model, replicate, condition, spec["split"], 0, prereg, output_root
    )
    initial_evaluation, _, digests = p1148.evaluate_bundle(
        model, "soft_EF", spec, prereg, "trajectory", False
    )
    trajectory.append(
        {"step": 0, "evaluation": initial_evaluation, "dataset_digests": digests, "checkpoint": checkpoint}
    )
    model.train()
    values = p1148.candidate_ids(spec["lexicon"], device)
    nonfinite_steps = 0
    for step in range(int(training["max_steps"])):
        lr = p1148.p1147.p1146.learning_rate(step, training)
        for group in optimizer.param_groups:
            group["lr"] = lr
        indices = torch.from_numpy(schedule[step])
        ids = cpu["inputs"][indices].to(device, non_blocking=True)
        targets = cpu["targets"][indices].to(device, non_blocking=True)
        rows = cpu["row_targets"][indices].to(device, non_blocking=True)
        columns = cpu["column_targets"][indices].to(device, non_blocking=True)
        grids = cpu["grid_values"][indices].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, row_logits, column_logits = model(ids)
        answer_loss, predictions, _ = p1148.answer_loss_and_prediction(
            "soft_EF", logits, row_logits, column_logits, grids, targets, values
        )
        row_loss = F.cross_entropy(row_logits.float(), rows)
        column_loss = F.cross_entropy(column_logits.float(), columns)
        loss = answer_loss + row_loss + column_loss
        if not torch.isfinite(loss):
            nonfinite_steps += 1
            raise RuntimeError(f"Nonfinite loss at {replicate}/{condition}/{step + 1}")
        loss.backward()
        if not all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        ):
            nonfinite_steps += 1
            raise RuntimeError(f"Nonfinite gradients at {replicate}/{condition}/{step + 1}")
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        current = step + 1
        if current % int(training["log_interval"]) == 0 or current in trajectory_steps:
            log = {
                "step": current,
                "loss": float(loss.detach().cpu()),
                "answer_loss": float(answer_loss.detach().cpu()),
                "row_loss": float(row_loss.detach().cpu()),
                "column_loss": float(column_loss.detach().cpu()),
                "batch_answer_accuracy": float((predictions == targets).float().mean().cpu()),
                "batch_row_accuracy": float((row_logits.argmax(-1) == rows).float().mean().cpu()),
                "batch_column_accuracy": float(
                    (column_logits.argmax(-1) == columns).float().mean().cpu()
                ),
                "learning_rate": float(lr),
                "gradient_norm": float(torch.as_tensor(gradient_norm).detach().cpu()),
            }
            logs.append(log)
            print(json.dumps({"replicate": replicate, "condition": condition, **log}), flush=True)
        if current in trajectory_steps:
            checkpoint = save_checkpoint(
                model, replicate, condition, spec["split"], current, prereg, output_root
            )
            evaluation, _, digests = p1148.evaluate_bundle(
                model, "soft_EF", spec, prereg, "trajectory", False
            )
            trajectory.append(
                {
                    "step": current,
                    "evaluation": evaluation,
                    "dataset_digests": digests,
                    "checkpoint": checkpoint,
                }
            )
            model.train()

    model.eval()
    evaluation, rows, evaluation_digests = p1148.evaluate_bundle(
        model, "soft_EF", spec, prereg, "formal", True
    )
    answer_checks = p1148.answer_gate(evaluation, prereg["thresholds"])
    address_checks = p1148.address_gate(evaluation, "soft_EF", prereg["thresholds"])
    qualified = all(answer_checks.values()) and all(address_checks.values())
    run_dir = output_root / "runs" / spec["split"] / replicate / condition
    predictions_path = run_dir / "predictions.jsonl"
    p1148.write_jsonl(predictions_path, rows)
    model_path = run_dir / "model.pt"
    payload = {
        "phase": PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "replicate": replicate,
        "condition": condition,
        "config": asdict(model.config),
        "readout_mode": model.readout_mode,
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
    }
    torch.save(payload, model_path)
    summary = {
        "phase": PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "replicate": replicate,
        "split": spec["split"],
        "scale": spec["scale"],
        "condition": condition,
        "readout_positions": prereg["conditions"][condition],
        "architecture": spec["architecture"],
        "parameter_count": p1148.p1147.p1146.model_parameter_count(model),
        "training_steps": int(training["max_steps"]),
        "initial_state_digest": initial_digest,
        "training_dataset_digest": training_digest,
        "batch_schedule_digest": schedule_digest,
        "nonfinite_steps": nonfinite_steps,
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
        "logs": logs,
        "trajectory": trajectory,
        "evaluation": evaluation,
        "evaluation_dataset_digests": evaluation_digests,
        "answer_gate_checks": answer_checks,
        "address_gate_checks": address_checks,
        "qualified": qualified,
        "model_path": str(model_path.relative_to(ROOT)).replace("\\", "/"),
        "model_sha256": p1148.file_sha256(model_path),
        "predictions_path": str(predictions_path.relative_to(ROOT)).replace("\\", "/"),
        "predictions_sha256": p1148.file_sha256(predictions_path),
    }
    summary["summary_digest"] = p1148.canonical_digest(summary)
    p1148.write_json(run_dir / "summary.json", summary)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return summary


def load_summary(replicate: str, condition: str, prereg: dict[str, Any]) -> dict[str, Any]:
    split = prereg["replicates"][replicate]["split"]
    return p1148.read_json(
        OUT_ROOT / "runs" / split / replicate / condition / "summary.json"
    )


def run_split(split: str, prereg: dict[str, Any]) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for replicate in split_replicates(prereg, split):
        results[replicate] = {}
        for condition in CONDITION_ORDER:
            results[replicate][condition] = train_condition(
                replicate, condition, prereg
            )
    return results


def analyze_split(split: str, prereg: dict[str, Any]) -> dict[str, Any]:
    replicates = split_replicates(prereg, split)
    summaries = {
        replicate: {
            condition: load_summary(replicate, condition, prereg)
            for condition in CONDITION_ORDER
        }
        for replicate in replicates
    }
    all_qualified = {
        condition: all(summaries[replicate][condition]["qualified"] for replicate in replicates)
        for condition in CONDITION_ORDER
    }
    effects: dict[str, Any] = {}
    gain_passes: list[bool] = []
    for replicate in replicates:
        effects[replicate] = {}
        for evaluation_split in ("seen", "holdout", "quartet"):
            baseline = float(
                summaries[replicate]["answer_boundary"]["evaluation"][evaluation_split]["accuracy"]
            )
            factorized = float(
                summaries[replicate]["role_factorized"]["evaluation"][evaluation_split]["accuracy"]
            )
            gain = factorized - baseline
            pass_gate = gain >= prereg["thresholds"]["minimum_paired_accuracy_gain"]
            effects[replicate][evaluation_split] = {
                "answer_boundary_accuracy": baseline,
                "role_factorized_accuracy": factorized,
                "paired_gain": gain,
                "gain_gate": pass_gate,
            }
            if evaluation_split in ("holdout", "quartet"):
                gain_passes.append(pass_gate)
    gain_scope_pass = all(gain_passes)
    discovery = None
    if split == "confirmation":
        discovery = p1148.read_json(OUT_ROOT / "analysis" / "discovery_selection.json")
    selected = (
        "role_factorized"
        if split == "discovery" and all_qualified["role_factorized"] and gain_scope_pass
        else discovery["selected_condition"]
        if split == "confirmation" and discovery
        else None
    )
    selected_qualified = bool(
        selected
        and all_qualified[str(selected)]
        and gain_scope_pass
    )
    result = {
        "phase": PHASE,
        "split": split,
        "protocol_digest": prereg["protocol_digest"],
        "replicates": replicates,
        "condition_all_qualified": all_qualified,
        "effects": effects,
        "gain_scope_pass": gain_scope_pass,
        "selected_condition": selected,
        "confirmation_authorized": selected_qualified if split == "discovery" else None,
        "causal_validation_authorized": selected_qualified if split == "confirmation" else None,
        "claim_status": (
            "candidate_requires_confirmation"
            if split == "discovery" and selected_qualified
            else "independently_confirmed_role_formation"
            if split == "confirmation" and selected_qualified
            else "role_factorized_formation_not_confirmed"
        ),
        "next_action": (
            "confirmation"
            if split == "discovery" and selected_qualified
            else "causal_position_validation"
            if split == "confirmation" and selected_qualified
            else "branch_without_role_formation_claim"
        ),
        "summary_digests": {
            replicate: {
                condition: summaries[replicate][condition]["summary_digest"]
                for condition in CONDITION_ORDER
            }
            for replicate in replicates
        },
    }
    result["selection_digest"] = p1148.canonical_digest(result)
    p1148.write_json(OUT_ROOT / "analysis" / f"{split}_selection.json", result)
    return result


def intervention_accuracy(
    model: RoleMediationModel,
    dataset: dict[str, np.ndarray],
    mode: str,
    batch_size: int,
) -> float:
    device = next(model.parameters()).device
    predictions: list[int] = []
    mode_spec = {
        "normal": (-3, -2),
        "both_answer": (-1, -1),
        "row_answer": (-1, -2),
        "column_answer": (-3, -1),
        "swapped": (-2, -3),
    }
    with torch.inference_mode():
        for start in range(0, len(dataset["inputs"]), batch_size):
            stop = start + batch_size
            ids = torch.from_numpy(dataset["inputs"][start:stop]).to(device)
            grids = torch.from_numpy(dataset["grid_values"][start:stop]).to(device)
            rows = torch.from_numpy(dataset["row_targets"][start:stop]).to(device)
            columns = torch.from_numpy(dataset["column_targets"][start:stop]).to(device)
            if mode == "oracle_both":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    _, row_logits, column_logits = model(ids)
                distribution = p1148.mediated_distribution(
                    row_logits,
                    column_logits,
                    grids,
                    row_mode="oracle",
                    column_mode="oracle",
                    row_targets=rows,
                    column_targets=columns,
                )
            else:
                row_position, column_position = mode_spec[mode]
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    _, row_logits, column_logits = model.forward_with_positions(
                        ids, row_position, column_position
                    )
                distribution = p1148.mediated_distribution(row_logits, column_logits, grids)
            predictions.extend(distribution.argmax(-1).cpu().tolist())
    return float(np.mean(np.asarray(predictions) == dataset["targets"]))


def run_causal_validation(prereg: dict[str, Any]) -> dict[str, Any]:
    confirmation = p1148.read_json(OUT_ROOT / "analysis" / "confirmation_selection.json")
    if not confirmation["causal_validation_authorized"]:
        raise RuntimeError("Causal position validation is not authorized")
    condition = str(confirmation["selected_condition"])
    per_replicate: dict[str, Any] = {}
    all_passes: list[bool] = []
    for replicate in split_replicates(prereg, "confirmation"):
        spec = prereg["replicates"][replicate]
        model = load_model(load_summary(replicate, condition, prereg))
        datasets, _ = p1148.build_evaluation_sets(spec, prereg, "formal")
        split_metrics: dict[str, Any] = {}
        replicate_checks: list[bool] = []
        for split_name, dataset in datasets:
            if split_name == "seen":
                continue
            metrics = {
                mode: intervention_accuracy(
                    model,
                    dataset,
                    mode,
                    int(spec["training"]["evaluation_batch_size"]),
                )
                for mode in (
                    "normal",
                    "both_answer",
                    "row_answer",
                    "column_answer",
                    "swapped",
                    "oracle_both",
                )
            }
            metrics["both_answer_drop"] = metrics["normal"] - metrics["both_answer"]
            metrics["row_answer_drop"] = metrics["normal"] - metrics["row_answer"]
            metrics["column_answer_drop"] = metrics["normal"] - metrics["column_answer"]
            metrics["swapped_drop"] = metrics["normal"] - metrics["swapped"]
            gate = {
                "base_behavior": metrics["normal"] >= prereg["thresholds"]["holdout_accuracy"],
                "both_answer_necessity": metrics["both_answer_drop"]
                >= prereg["thresholds"]["position_ablation_drop"],
                "row_role_necessity": metrics["row_answer_drop"]
                >= prereg["thresholds"]["single_role_ablation_drop"],
                "column_role_necessity": metrics["column_answer_drop"]
                >= prereg["thresholds"]["single_role_ablation_drop"],
                "swap_specificity": metrics["swapped_drop"]
                >= prereg["thresholds"]["position_ablation_drop"],
                "oracle_rescue": metrics["oracle_both"]
                >= prereg["thresholds"]["oracle_rescue_accuracy"],
            }
            replicate_checks.append(all(gate.values()))
            split_metrics[split_name] = {
                "metrics": metrics,
                "gate": gate,
                "dataset_digest": p1148.dataset_digest(dataset),
            }
        per_replicate[replicate] = {
            "splits": split_metrics,
            "passed": all(replicate_checks),
        }
        all_passes.append(all(replicate_checks))
        del model
        torch.cuda.empty_cache()
    result = {
        "phase": PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "selected_condition": condition,
        "per_replicate": per_replicate,
        "all_replicates_passed": all(all_passes),
        "claim_scope": "causal_use_of_planted_role_specific_readout_positions_only",
    }
    result["causal_digest"] = p1148.canonical_digest(result)
    p1148.write_json(OUT_ROOT / "analysis" / "causal_position_validation.json", result)
    return result


def finalize(prereg: dict[str, Any]) -> dict[str, Any]:
    discovery = p1148.read_json(OUT_ROOT / "analysis" / "discovery_selection.json")
    confirmation_path = OUT_ROOT / "analysis" / "confirmation_selection.json"
    causal_path = OUT_ROOT / "analysis" / "causal_position_validation.json"
    confirmation = p1148.read_json(confirmation_path) if confirmation_path.exists() else None
    causal = p1148.read_json(causal_path) if causal_path.exists() else None
    calibrated = bool(causal and causal["all_replicates_passed"])
    result = {
        "phase": PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "discovery_digest": discovery["selection_digest"],
        "confirmation_digest": confirmation["selection_digest"] if confirmation else None,
        "causal_digest": causal["causal_digest"] if causal else None,
        "outcome": (
            "role_factorized_mediator_calibrated"
            if calibrated
            else "role_factorized_candidate_not_causally_closed"
            if confirmation and confirmation["causal_validation_authorized"]
            else "role_factorized_formation_not_confirmed"
        ),
        "evidence_vector": {
            "integrity": True,
            "behavior": bool(confirmation and confirmation["causal_validation_authorized"]),
            "role_specificity": bool(discovery["gain_scope_pass"]),
            "replication": bool(confirmation and confirmation["causal_validation_authorized"]),
            "causal_use": calibrated,
            "natural_mechanism": False,
        },
        "claim_boundary": (
            "A positive result calibrates a planted role-aligned mediator and shows that readout geometry "
            "can determine formation. It does not show that pretrained language models use this mediator."
        ),
        "auto_continue": False,
        "auto_continue_reason": (
            "A free-network functional-equivalence test requires a new frozen protocol and cannot be inferred "
            "from the planted architecture."
        ),
    }
    result["final_digest"] = p1148.canonical_digest(result)
    p1148.write_json(OUT_ROOT / "analysis" / "final.json", result)
    return result


def smoke() -> dict[str, Any]:
    parent = p1148.read_json(p1148.PREREG_PATH)
    spec = parent["replicates"]["discovery_small_r1"]
    dataset = p1148.make_dataset(64, parent["data"]["pairs"]["train"], 149990, spec["lexicon"])
    device = torch.device("cuda")
    states = {}
    losses = {}
    parameters = {}
    finite = {}
    for condition in CONDITION_ORDER:
        p1148.set_seed(149991)
        model = RoleMediationModel(
            p1148.p1147.p1146.ModelConfig(**spec["architecture"]), condition
        ).to(device)
        states[condition] = p1148.state_digest(model.state_dict())
        parameters[condition] = p1148.p1147.p1146.model_parameter_count(model)
        ids = torch.from_numpy(dataset["inputs"]).to(device)
        targets = torch.from_numpy(dataset["targets"]).to(device)
        rows = torch.from_numpy(dataset["row_targets"]).to(device)
        columns = torch.from_numpy(dataset["column_targets"]).to(device)
        grids = torch.from_numpy(dataset["grid_values"]).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, row_logits, column_logits = model(ids)
        answer_loss, _, _ = p1148.answer_loss_and_prediction(
            "soft_EF",
            logits,
            row_logits,
            column_logits,
            grids,
            targets,
            p1148.candidate_ids(spec["lexicon"], device),
        )
        loss = answer_loss + F.cross_entropy(row_logits.float(), rows) + F.cross_entropy(
            column_logits.float(), columns
        )
        loss.backward()
        losses[condition] = float(loss.detach().cpu())
        finite[condition] = bool(torch.isfinite(loss)) and all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        )
        del model
        torch.cuda.empty_cache()
    result = {
        "phase": PHASE,
        "paired_initial_states": len(set(states.values())) == 1,
        "equal_parameters": len(set(parameters.values())) == 1,
        "finite_gradients": finite,
        "losses": losses,
        "state_digests": states,
        "parameter_counts": parameters,
    }
    result["all_checks_passed"] = (
        result["paired_initial_states"]
        and result["equal_parameters"]
        and all(finite.values())
    )
    result["smoke_digest"] = p1148.canonical_digest(result)
    p1148.write_json(TEMP_ROOT / "smoke.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("smoke", "create-protocol", "run-split", "analyze-split", "run-causal", "finalize"),
        required=True,
    )
    parser.add_argument("--split", choices=("discovery", "confirmation"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "smoke":
        result = smoke()
    elif args.mode == "create-protocol":
        result = create_protocol()
    else:
        prereg = p1148.read_json(PREREG_PATH)
        verify_preregistration(prereg)
        if args.mode == "run-split":
            if not args.split:
                raise ValueError("--split required")
            result = run_split(args.split, prereg)
        elif args.mode == "analyze-split":
            if not args.split:
                raise ValueError("--split required")
            result = analyze_split(args.split, prereg)
        elif args.mode == "run-causal":
            result = run_causal_validation(prereg)
        else:
            result = finalize(prereg)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
