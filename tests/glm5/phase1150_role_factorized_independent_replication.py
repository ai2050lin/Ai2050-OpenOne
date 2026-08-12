from __future__ import annotations

import argparse
import gc
import hashlib
import json
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

import phase1148_mandatory_mediation_calibration as p1148
import phase1149_role_factorized_mediation as p1149


PHASE = 1150
ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "tests" / "glm5" / "result" / "phase1150_role_factorized_independent_replication"
TEMP_ROOT = ROOT / "tests" / "glm5_temp" / "phase1150_smoke"
PREREG_PATH = OUT_ROOT / "protocol" / "preregistration.json"
CONDITIONS = deepcopy(p1149.CONDITIONS)
CONDITION_ORDER = list(p1149.CONDITION_ORDER)


def fresh_pair_partition() -> dict[str, list[list[int]]]:
    """Give every entity two train fields and one field in each sealed split."""
    base = p1148.p1147.p1146
    result = {"train": [], "discovery": [], "confirmation": []}
    for entity in range(base.ENTITY_COUNT):
        shift = (3 * entity + 1) % base.FIELD_COUNT
        ordered = [(shift + offset) % base.FIELD_COUNT for offset in range(base.FIELD_COUNT)]
        result["train"].extend([[entity, ordered[0]], [entity, ordered[1]]])
        result["discovery"].append([entity, ordered[2]])
        result["confirmation"].append([entity, ordered[3]])
    return result


def replicate_specifications() -> dict[str, dict[str, Any]]:
    base = p1148.p1147.p1146
    common_training = {
        "learning_rate": 0.003,
        "minimum_learning_rate": 0.0003,
        "warmup_steps": 100,
        "weight_decay": 0.01,
        "gradient_clip_norm": 1.0,
        "row_loss_weight": 1.0,
        "column_loss_weight": 1.0,
        "log_interval": 500,
        "evaluation_batch_size": 512,
    }
    definitions = [
        ("discovery_small_r1", "discovery", "small", 115001),
        ("discovery_small_r2", "discovery", "small", 115002),
        ("discovery_medium_r1", "discovery", "medium", 115003),
        ("discovery_medium_r2", "discovery", "medium", 115004),
        ("confirmation_small_r1", "confirmation", "small", 115005),
        ("confirmation_small_r2", "confirmation", "small", 115006),
        ("confirmation_medium_r1", "confirmation", "medium", 115007),
        ("confirmation_medium_r2", "confirmation", "medium", 115008),
    ]
    result: dict[str, dict[str, Any]] = {}
    for offset, (name, split, scale, training_seed) in enumerate(definitions):
        small = scale == "small"
        maximum_steps = 2500 if small else 3000
        result[name] = {
            "split": split,
            "scale": scale,
            "training_seed": training_seed,
            "lexicon_seed": 115011 + offset,
            "sampler_seed": 115021 + offset,
            "architecture": asdict(
                base.ModelConfig(
                    layers=4 if small else 6,
                    width=64 if small else 96,
                    heads=4,
                    mlp_width=256 if small else 384,
                )
            ),
            "training": {
                **common_training,
                "batch_size": 512 if small else 384,
                "max_steps": maximum_steps,
                "trajectory_steps": [0, 250, 1000, maximum_steps],
            },
            "data_seeds": {
                "training": 115031 + offset,
                "seen_evaluation": 115041 + offset,
                "holdout_evaluation": 115051 + offset,
                "quartet": 115061 + offset,
                "trajectory_seen": 115071 + offset,
                "trajectory_holdout": 115081 + offset,
                "trajectory_quartet": 115091 + offset,
            },
        }
        result[name]["lexicon"] = base.make_lexicon(int(result[name]["lexicon_seed"]))
    return result


def protocol_body() -> dict[str, Any]:
    parent = p1148.read_json(p1149.PREREG_PATH)
    p1149.verify_preregistration(parent)
    return {
        "phase": PHASE,
        "title": "Independent absolute-formation replication of role-factorized mediation",
        "claim_scope": (
            "Separates the absolute formation claim from Phase1149's failed fixed paired-gain claim. "
            "It tests a planted role-factorized mediator on new initialization, data seeds, token "
            "permutations, and a new entity-field partition. It cannot establish a natural language mechanism."
        ),
        "parent_phase1149_protocol_digest": parent["protocol_digest"],
        "source_hashes": {
            "primary_script": p1148.file_sha256(Path(__file__).resolve()),
            "phase1149_dependency": p1148.file_sha256(Path(p1149.__file__).resolve()),
            "phase1148_dependency": p1148.file_sha256(Path(p1148.__file__).resolve()),
            "phase1147_dependency": p1148.file_sha256(Path(p1148.p1147.__file__).resolve()),
            "phase1146_dependency": p1148.file_sha256(
                Path(p1148.p1147.p1146.__file__).resolve()
            ),
        },
        "conditions": deepcopy(CONDITIONS),
        "condition_order": list(CONDITION_ORDER),
        "semantics": deepcopy(p1148.read_json(p1148.PREREG_PATH)["semantics"]),
        "data": {
            "pairs": fresh_pair_partition(),
            "training_count": 65536,
            "evaluation_count": 4096,
            "trajectory_evaluation_count": 1024,
            "field_order_control": "one random field order shared across records within each example",
            "surface_control": "new full non-special-token permutation per replicate, paired across conditions",
            "independence_scope": (
                "No initialization, data seed, sampler seed, token permutation, or realized dataset digest "
                "is reused from Phase1148-1149. The semantic task family remains fixed."
            ),
        },
        "replicates": replicate_specifications(),
        "training_control": {
            "same_parameterization": True,
            "same_initial_state_per_replicate": True,
            "same_training_material_per_replicate": True,
            "same_batch_schedule_per_replicate": True,
            "same_answer_row_column_losses": True,
            "only_difference": "row_and_column_readout_token_positions",
        },
        "thresholds": {
            "seen_accuracy": 0.995,
            "holdout_accuracy": 0.95,
            "quartet_accuracy": 0.95,
            "minimum_field_accuracy": 0.90,
            "minimum_entity_accuracy": 0.80,
            "address_accuracy": 0.95,
            "oracle_accuracy": 1.0,
        },
        "primary_gate": {
            "role_factorized_qualified_in_all_four_replicates": True,
            "answer_boundary_unqualified_in_all_four_replicates": True,
            "discovery_and_confirmation_must_pass_independently": True,
            "paired_gain_is_descriptive_not_authorizing": True,
        },
        "contingent_phase1151": {
            "authorized_only_after_confirmation": True,
            "splits": ["holdout", "quartet"],
            "minimum_normal_accuracy": 0.95,
            "minimum_address_transport_accuracy": 0.95,
            "minimum_counterfactual_answer_accuracy": 0.95,
            "minimum_orthogonal_role_preservation": 0.95,
            "minimum_cross_role_specificity_advantage": 0.20,
            "claims_separated": [
                "counterfactual_role_state_sufficiency",
                "cross_role_position_specificity",
            ],
        },
        "gate_policy": {
            "hard_stop": [
                "nonfinite training",
                "label-grid or oracle failure",
                "hash, recomputation, freshness, or pairing failure",
            ],
            "claim_stop": [
                "absolute role-factorized formation failure",
                "paired answer-boundary control qualification",
                "independent confirmation failure",
            ],
            "branch_after_failure": (
                "Seal Phase1150 data and stop this exact absolute-formation branch. Do not tune readout "
                "positions, thresholds, or training budget on the same material."
            ),
        },
        "forbidden": [
            "No Phase1148-1149 seed, token permutation, realized dataset, or model reuse",
            "No post-freeze change to positions, losses, budget, thresholds, or pair partition",
            "No confirmation unless the complete discovery primary gate passes",
            "No use of paired-gain magnitude as an authorizing gate or post-hoc rescue",
            "No Phase1151 intervention unless the complete confirmation primary gate passes",
            "No free-Transformer, pretrained-LLM, natural-language, or brain mechanism claim",
        ],
    }


def verify_preregistration(prereg: dict[str, Any]) -> None:
    body = dict(prereg)
    digest = body.pop("protocol_digest")
    if p1148.canonical_digest(body) != digest:
        raise RuntimeError("Phase1150 protocol digest mismatch")
    hashes = prereg["source_hashes"]
    current = {
        "primary_script": p1148.file_sha256(Path(__file__).resolve()),
        "phase1149_dependency": p1148.file_sha256(Path(p1149.__file__).resolve()),
        "phase1148_dependency": p1148.file_sha256(Path(p1148.__file__).resolve()),
        "phase1147_dependency": p1148.file_sha256(Path(p1148.p1147.__file__).resolve()),
        "phase1146_dependency": p1148.file_sha256(Path(p1148.p1147.p1146.__file__).resolve()),
    }
    if current != hashes:
        raise RuntimeError("Phase1150 source changed after preregistration")
    parent = p1148.read_json(p1149.PREREG_PATH)
    if parent["protocol_digest"] != prereg["parent_phase1149_protocol_digest"]:
        raise RuntimeError("Phase1149 parent protocol changed")


def split_replicates(prereg: dict[str, Any], split: str) -> list[str]:
    return [name for name, spec in prereg["replicates"].items() if spec["split"] == split]


def realized_material_digests(spec: dict[str, Any], prereg: dict[str, Any]) -> dict[str, str]:
    training, schedule = p1148.make_training_material(spec, prereg)
    evaluation_sets, _ = p1148.build_evaluation_sets(spec, prereg, "formal")
    return {
        "training": p1148.dataset_digest(training),
        "schedule": p1148.array_digest(schedule),
        **{name: p1148.dataset_digest(dataset) for name, dataset in evaluation_sets},
    }


def create_protocol() -> dict[str, Any]:
    body = protocol_body()
    prereg = dict(body)
    prereg["protocol_digest"] = p1148.canonical_digest(body)
    if PREREG_PATH.exists():
        if p1148.read_json(PREREG_PATH) != prereg:
            raise RuntimeError("Existing Phase1150 protocol differs from current script")
    else:
        p1148.write_json(PREREG_PATH, prereg)

    old = p1148.read_json(p1148.PREREG_PATH)
    old_training_seeds = {int(spec["training_seed"]) for spec in old["replicates"].values()}
    old_lexicons = {tuple(spec["lexicon"]) for spec in old["replicates"].values()}
    old_data_seeds = {
        int(seed)
        for spec in old["replicates"].values()
        for seed in spec["data_seeds"].values()
    }
    material = {
        name: realized_material_digests(spec, prereg)
        for name, spec in prereg["replicates"].items()
    }
    all_material_digests = [digest for item in material.values() for digest in item.values()]
    audit_spec = prereg["replicates"]["discovery_small_r1"]
    audit_dataset = p1148.make_dataset(
        4096,
        prereg["data"]["pairs"]["train"],
        int(audit_spec["data_seeds"]["training"]),
        audit_spec["lexicon"],
    )
    oracle = audit_dataset["grid_values"][
        np.arange(len(audit_dataset["targets"])),
        audit_dataset["row_targets"],
        audit_dataset["column_targets"],
    ]
    pair_sets = {
        split: {tuple(pair) for pair in prereg["data"]["pairs"][split]}
        for split in ("train", "discovery", "confirmation")
    }
    checks = {
        "eight_replicates": len(prereg["replicates"]) == 8,
        "four_replicates_per_split": all(
            len(split_replicates(prereg, split)) == 4 for split in ("discovery", "confirmation")
        ),
        "pair_partition_complete": len(set.union(*pair_sets.values())) == 96,
        "pair_partition_disjoint": not (
            pair_sets["train"] & pair_sets["discovery"]
            or pair_sets["train"] & pair_sets["confirmation"]
            or pair_sets["discovery"] & pair_sets["confirmation"]
        ),
        "new_pair_partition": prereg["data"]["pairs"] != old["data"]["pairs"],
        "training_seeds_fresh": not old_training_seeds
        & {int(spec["training_seed"]) for spec in prereg["replicates"].values()},
        "data_seeds_fresh": not old_data_seeds
        & {
            int(seed)
            for spec in prereg["replicates"].values()
            for seed in spec["data_seeds"].values()
        },
        "lexicons_fresh": not old_lexicons
        & {tuple(spec["lexicon"]) for spec in prereg["replicates"].values()},
        "realized_material_unique": len(all_material_digests) == len(set(all_material_digests)),
        "oracle_exact": bool(np.all(oracle == audit_dataset["targets"])),
        "rows_complete": set(audit_dataset["row_targets"].tolist()) == {0, 1, 2},
        "columns_complete": set(audit_dataset["column_targets"].tolist()) == {0, 1, 2, 3},
        "conditions_frozen": prereg["conditions"] == CONDITIONS,
    }
    audit = {
        "phase": PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "material_digests": material,
    }
    audit["audit_digest"] = p1148.canonical_digest(audit)
    p1148.write_json(OUT_ROOT / "protocol" / "audit.json", audit)
    if not audit["all_checks_passed"]:
        raise RuntimeError(f"Phase1150 protocol audit failed: {checks}")
    return prereg


def save_model(
    model: p1149.RoleMediationModel,
    replicate: str,
    condition: str,
    split: str,
    prereg: dict[str, Any],
) -> tuple[Path, str]:
    path = OUT_ROOT / "runs" / split / replicate / condition / "model.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "phase": PHASE,
            "protocol_digest": prereg["protocol_digest"],
            "replicate": replicate,
            "condition": condition,
            "config": asdict(model.config),
            "readout_mode": model.readout_mode,
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        },
        path,
    )
    return path, p1148.file_sha256(path)


def load_model(summary: dict[str, Any]) -> p1149.RoleMediationModel:
    payload = torch.load(ROOT / summary["model_path"], map_location="cpu", weights_only=True)
    model = p1149.RoleMediationModel(
        p1148.p1147.p1146.ModelConfig(**payload["config"]),
        str(payload["readout_mode"]),
    )
    model.load_state_dict(payload["state_dict"])
    return model.cuda().eval()


def train_condition(replicate: str, condition: str, prereg: dict[str, Any]) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("Phase1150 requires CUDA")
    spec = prereg["replicates"][replicate]
    split = str(spec["split"])
    if split == "confirmation":
        discovery_path = OUT_ROOT / "analysis" / "discovery_selection.json"
        if not discovery_path.exists() or not p1148.read_json(discovery_path)[
            "confirmation_authorized"
        ]:
            raise RuntimeError("Phase1150 confirmation is not authorized")

    p1148.set_seed(int(spec["training_seed"]))
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    model = p1149.RoleMediationModel(
        p1148.p1147.p1146.ModelConfig(**spec["architecture"]), condition
    ).to(device)
    initial_digest = p1148.state_digest(model.state_dict())
    training_data, schedule = p1148.make_training_material(spec, prereg)
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
    values = p1148.candidate_ids(spec["lexicon"], device)
    trajectory_steps = set(int(step) for step in training["trajectory_steps"])
    trajectory: list[dict[str, Any]] = []
    logs: list[dict[str, Any]] = []
    torch.cuda.reset_peak_memory_stats()

    initial_evaluation, _, initial_digests = p1148.evaluate_bundle(
        model, "soft_EF", spec, prereg, "trajectory", False
    )
    trajectory.append({"step": 0, "evaluation": initial_evaluation, "dataset_digests": initial_digests})
    model.train()
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
            raise RuntimeError(f"Nonfinite loss at {replicate}/{condition}/{step + 1}")
        loss.backward()
        if not all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        ):
            raise RuntimeError(f"Nonfinite gradient at {replicate}/{condition}/{step + 1}")
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
            model.eval()
            evaluation, _, digests = p1148.evaluate_bundle(
                model, "soft_EF", spec, prereg, "trajectory", False
            )
            trajectory.append({"step": current, "evaluation": evaluation, "dataset_digests": digests})
            model.train()

    model.eval()
    evaluation, prediction_rows, evaluation_digests = p1148.evaluate_bundle(
        model, "soft_EF", spec, prereg, "formal", True
    )
    answer_checks = p1148.answer_gate(evaluation, prereg["thresholds"])
    address_checks = p1148.address_gate(evaluation, "soft_EF", prereg["thresholds"])
    qualified = all(answer_checks.values()) and all(address_checks.values())
    run_dir = OUT_ROOT / "runs" / split / replicate / condition
    prediction_path = run_dir / "predictions.jsonl"
    p1148.write_jsonl(prediction_path, prediction_rows)
    model_path, model_hash = save_model(model, replicate, condition, split, prereg)
    summary = {
        "phase": PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "replicate": replicate,
        "split": split,
        "scale": spec["scale"],
        "condition": condition,
        "readout_positions": prereg["conditions"][condition],
        "architecture": spec["architecture"],
        "parameter_count": p1148.p1147.p1146.model_parameter_count(model),
        "training_steps": int(training["max_steps"]),
        "initial_state_digest": initial_digest,
        "training_dataset_digest": p1148.dataset_digest(training_data),
        "batch_schedule_digest": p1148.array_digest(schedule),
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
        "logs": logs,
        "trajectory": trajectory,
        "evaluation": evaluation,
        "evaluation_dataset_digests": evaluation_digests,
        "answer_gate_checks": answer_checks,
        "address_gate_checks": address_checks,
        "qualified": qualified,
        "model_path": str(model_path.relative_to(ROOT)).replace("\\", "/"),
        "model_sha256": model_hash,
        "predictions_path": str(prediction_path.relative_to(ROOT)).replace("\\", "/"),
        "predictions_sha256": p1148.file_sha256(prediction_path),
    }
    summary["summary_digest"] = p1148.canonical_digest(summary)
    p1148.write_json(run_dir / "summary.json", summary)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return summary


def summary_path(replicate: str, condition: str, prereg: dict[str, Any]) -> Path:
    split = prereg["replicates"][replicate]["split"]
    return OUT_ROOT / "runs" / split / replicate / condition / "summary.json"


def load_summary(replicate: str, condition: str, prereg: dict[str, Any]) -> dict[str, Any]:
    return p1148.read_json(summary_path(replicate, condition, prereg))


def run_split(split: str, prereg: dict[str, Any]) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for replicate in split_replicates(prereg, split):
        results[replicate] = {}
        for condition in CONDITION_ORDER:
            results[replicate][condition] = train_condition(replicate, condition, prereg)
    return results


def build_split_analysis(split: str, prereg: dict[str, Any]) -> dict[str, Any]:
    replicates = split_replicates(prereg, split)
    summaries = {
        replicate: {
            condition: load_summary(replicate, condition, prereg)
            for condition in CONDITION_ORDER
        }
        for replicate in replicates
    }
    qualified = {
        condition: {
            replicate: bool(summaries[replicate][condition]["qualified"])
            for replicate in replicates
        }
        for condition in CONDITION_ORDER
    }
    role_all = all(qualified["role_factorized"].values())
    boundary_none = not any(qualified["answer_boundary"].values())
    primary_pass = role_all and boundary_none
    effects: dict[str, Any] = {}
    for replicate in replicates:
        effects[replicate] = {}
        for evaluation_split in ("seen", "holdout", "quartet"):
            baseline = float(
                summaries[replicate]["answer_boundary"]["evaluation"][evaluation_split]["accuracy"]
            )
            candidate = float(
                summaries[replicate]["role_factorized"]["evaluation"][evaluation_split]["accuracy"]
            )
            effects[replicate][evaluation_split] = {
                "answer_boundary_accuracy": baseline,
                "role_factorized_accuracy": candidate,
                "paired_gain_descriptive": candidate - baseline,
            }
    discovery = None
    if split == "confirmation":
        discovery = p1148.read_json(OUT_ROOT / "analysis" / "discovery_selection.json")
    result = {
        "phase": PHASE,
        "split": split,
        "protocol_digest": prereg["protocol_digest"],
        "replicates": replicates,
        "qualified": qualified,
        "role_factorized_all_qualified": role_all,
        "answer_boundary_all_unqualified": boundary_none,
        "primary_gate_passed": primary_pass,
        "paired_effects_descriptive": effects,
        "confirmation_authorized": primary_pass if split == "discovery" else None,
        "phase1151_authorized": (
            bool(primary_pass and discovery and discovery["confirmation_authorized"])
            if split == "confirmation"
            else None
        ),
        "claim_status": (
            "absolute_formation_candidate_requires_confirmation"
            if split == "discovery" and primary_pass
            else "absolute_formation_independently_confirmed"
            if split == "confirmation" and primary_pass
            else "absolute_formation_claim_stopped"
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
    return result


def analyze_split(split: str, prereg: dict[str, Any]) -> dict[str, Any]:
    result = build_split_analysis(split, prereg)
    p1148.write_json(OUT_ROOT / "analysis" / f"{split}_selection.json", result)
    return result


def finalize(prereg: dict[str, Any]) -> dict[str, Any]:
    discovery = p1148.read_json(OUT_ROOT / "analysis" / "discovery_selection.json")
    confirmation_path = OUT_ROOT / "analysis" / "confirmation_selection.json"
    confirmation = p1148.read_json(confirmation_path) if confirmation_path.exists() else None
    confirmed = bool(confirmation and confirmation["phase1151_authorized"])
    result = {
        "phase": PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "discovery_digest": discovery["selection_digest"],
        "confirmation_digest": confirmation["selection_digest"] if confirmation else None,
        "outcome": (
            "role_factorized_absolute_formation_independently_confirmed"
            if confirmed
            else "role_factorized_absolute_formation_not_confirmed"
        ),
        "evidence_vector": {
            "integrity": True,
            "absolute_behavior": confirmed,
            "paired_control": confirmed,
            "independent_replication": confirmed,
            "causal_role_state_use": False,
            "free_network_equivalence": False,
            "natural_mechanism": False,
        },
        "historical_phase1149_claim_unchanged": True,
        "paired_gain_status": "descriptive_only_in_phase1150",
        "claim_boundary": (
            "A positive result confirms only that a planted role-factorized mediator is a robust formation "
            "bias in this synthetic lookup family. It does not identify the mechanism of a free or pretrained model."
        ),
        "auto_continue": confirmed,
        "auto_continue_action": "phase1151_counterfactual_role_state_transplant" if confirmed else None,
    }
    result["final_digest"] = p1148.canonical_digest(result)
    p1148.write_json(OUT_ROOT / "analysis" / "final.json", result)
    return result


def smoke() -> dict[str, Any]:
    prereg = create_protocol() if not PREREG_PATH.exists() else p1148.read_json(PREREG_PATH)
    verify_preregistration(prereg)
    spec = deepcopy(prereg["replicates"]["discovery_small_r1"])
    mini = deepcopy(prereg)
    mini["data"]["training_count"] = 1024
    dataset, _ = p1148.make_training_material(spec, mini)
    device = torch.device("cuda")
    state_digests = {}
    parameter_counts = {}
    finite = {}
    for condition in CONDITION_ORDER:
        p1148.set_seed(int(spec["training_seed"]))
        model = p1149.RoleMediationModel(
            p1148.p1147.p1146.ModelConfig(**spec["architecture"]), condition
        ).to(device)
        state_digests[condition] = p1148.state_digest(model.state_dict())
        parameter_counts[condition] = p1148.p1147.p1146.model_parameter_count(model)
        ids = torch.from_numpy(dataset["inputs"][:64]).to(device)
        targets = torch.from_numpy(dataset["targets"][:64]).to(device)
        rows = torch.from_numpy(dataset["row_targets"][:64]).to(device)
        columns = torch.from_numpy(dataset["column_targets"][:64]).to(device)
        grids = torch.from_numpy(dataset["grid_values"][:64]).to(device)
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
        finite[condition] = bool(torch.isfinite(loss)) and all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        )
        del model
        torch.cuda.empty_cache()
    result = {
        "phase": PHASE,
        "paired_initial_state": len(set(state_digests.values())) == 1,
        "equal_parameter_count": len(set(parameter_counts.values())) == 1,
        "finite_gradients": finite,
        "state_digests": state_digests,
        "parameter_counts": parameter_counts,
    }
    result["all_checks_passed"] = (
        result["paired_initial_state"]
        and result["equal_parameter_count"]
        and all(finite.values())
    )
    result["smoke_digest"] = p1148.canonical_digest(result)
    p1148.write_json(TEMP_ROOT / "smoke.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        required=True,
        choices=("create-protocol", "smoke", "run-split", "analyze-split", "finalize"),
    )
    parser.add_argument("--split", choices=("discovery", "confirmation"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "create-protocol":
        result = create_protocol()
    elif args.mode == "smoke":
        result = smoke()
    else:
        prereg = p1148.read_json(PREREG_PATH)
        verify_preregistration(prereg)
        if args.mode in ("run-split", "analyze-split") and args.split is None:
            raise ValueError("--split is required")
        if args.mode == "run-split":
            result = run_split(str(args.split), prereg)
        elif args.mode == "analyze-split":
            result = analyze_split(str(args.split), prereg)
        elif args.mode == "finalize":
            result = finalize(prereg)
        else:
            raise ValueError(args.mode)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
