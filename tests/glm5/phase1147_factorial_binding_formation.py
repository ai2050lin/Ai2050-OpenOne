from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import phase1146_learned_composition_benchmark as p1146


PHASE = 1147
ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "tests" / "glm5" / "result" / "phase1147_factorial_binding_formation"
TEMP_ROOT = ROOT / "tests" / "glm5_temp" / "phase1147_smoke"
PREREG_PATH = OUT_ROOT / "protocol" / "preregistration.json"

ARMS = {
    "00": {"entity_loss": False, "field_loss": False},
    "E0": {"entity_loss": True, "field_loss": False},
    "0F": {"entity_loss": False, "field_loss": True},
    "EF": {"entity_loss": True, "field_loss": True},
}
ARM_PRIORITY = ["00", "E0", "0F", "EF"]


def canonical_digest(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_digest(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(json.dumps(list(contiguous.shape)).encode("ascii"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def state_digest(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_factorized_dataset(
    count: int,
    pairs: list[list[int]],
    seed: int,
    lexicon: list[int],
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    inputs = np.empty((count, p1146.SEQUENCE_LENGTH), dtype=np.int64)
    targets = np.empty(count, dtype=np.int64)
    entities_meta = np.empty(count, dtype=np.int64)
    fields_meta = np.empty(count, dtype=np.int64)
    row_targets = np.empty(count, dtype=np.int64)
    column_targets = np.empty(count, dtype=np.int64)
    pair_order = [tuple(pair) for pair in pairs]
    for index in range(count):
        query_entity, query_field = pair_order[index % len(pair_order)]
        target_value = (index // len(pair_order)) % p1146.VALUE_COUNT
        distractors = rng.choice(
            [entity for entity in range(p1146.ENTITY_COUNT) if entity != query_entity],
            size=p1146.RECORD_COUNT - 1,
            replace=False,
        ).tolist()
        entities = [query_entity, *[int(value) for value in distractors]]
        assignments = {
            entity: rng.integers(
                0, p1146.VALUE_COUNT, size=p1146.FIELD_COUNT, dtype=np.int64
            ).tolist()
            for entity in entities
        }
        assignments[query_entity][query_field] = int(target_value)
        record_order = list(entities)
        rng.shuffle(record_order)
        shared_field_order = list(range(p1146.FIELD_COUNT))
        rng.shuffle(shared_field_order)
        field_orders = {entity: list(shared_field_order) for entity in entities}
        sequence, _ = p1146.build_sequence(
            entities,
            assignments,
            query_entity,
            query_field,
            record_order,
            field_orders,
            lexicon,
        )
        inputs[index] = sequence
        targets[index] = target_value
        entities_meta[index] = query_entity
        fields_meta[index] = query_field
        row_targets[index] = record_order.index(query_entity)
        column_targets[index] = shared_field_order.index(query_field)
    permutation = rng.permutation(count)
    return {
        "inputs": inputs[permutation],
        "targets": targets[permutation],
        "entities": entities_meta[permutation],
        "fields": fields_meta[permutation],
        "row_targets": row_targets[permutation],
        "column_targets": column_targets[permutation],
    }


def make_factorized_quartets(
    pairs: list[list[int]],
    seed: int,
    lexicon: list[int],
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    rows: list[np.ndarray] = []
    labels: list[int] = []
    entities_meta: list[int] = []
    fields_meta: list[int] = []
    row_targets: list[int] = []
    column_targets: list[int] = []
    metadata: list[dict[str, Any]] = []
    state_names = ["active_minus", "active_plus", "null_minus", "null_plus"]
    for item_index, pair in enumerate(pairs):
        query_entity, query_field = int(pair[0]), int(pair[1])
        destination = 1 + (item_index % 6)
        base = 0
        anchor = 7
        distractors = rng.choice(
            [entity for entity in range(p1146.ENTITY_COUNT) if entity != query_entity],
            size=2,
            replace=False,
        ).tolist()
        entity1, entity2 = int(distractors[0]), int(distractors[1])
        entities = [query_entity, entity1, entity2]
        common_assignments = {
            entity: rng.integers(
                0, p1146.VALUE_COUNT, size=p1146.FIELD_COUNT, dtype=np.int64
            ).tolist()
            for entity in entities
        }
        record_order = list(entities)
        rng.shuffle(record_order)
        shared_field_order = list(range(p1146.FIELD_COUNT))
        rng.shuffle(shared_field_order)
        field_orders = {entity: list(shared_field_order) for entity in entities}
        state_values = [
            (base, destination, anchor, base),
            (destination, base, anchor, destination),
            (anchor, base, destination, anchor),
            (anchor, destination, base, anchor),
        ]
        for state_index, (value0, value1, value2, target) in enumerate(state_values):
            assignments = {entity: list(values) for entity, values in common_assignments.items()}
            assignments[query_entity][query_field] = value0
            assignments[entity1][query_field] = value1
            assignments[entity2][query_field] = value2
            sequence, positions = p1146.build_sequence(
                entities,
                assignments,
                query_entity,
                query_field,
                record_order,
                field_orders,
                lexicon,
            )
            rows.append(sequence)
            labels.append(target)
            entities_meta.append(query_entity)
            fields_meta.append(query_field)
            row_targets.append(record_order.index(query_entity))
            column_targets.append(shared_field_order.index(query_field))
            metadata.append(
                {
                    "item_index": item_index,
                    "item_id": f"e{query_entity:02d}.f{query_field}.d{destination}",
                    "state_index": state_index,
                    "state": state_names[state_index],
                    "query_entity": query_entity,
                    "query_field": query_field,
                    "target_value": target,
                    "answer_position": p1146.SEQUENCE_LENGTH - 1,
                    "queried_source_position": positions[(query_entity, query_field)],
                }
            )
    dataset = {
        "inputs": np.stack(rows),
        "targets": np.asarray(labels, dtype=np.int64),
        "entities": np.asarray(entities_meta, dtype=np.int64),
        "fields": np.asarray(fields_meta, dtype=np.int64),
        "row_targets": np.asarray(row_targets, dtype=np.int64),
        "column_targets": np.asarray(column_targets, dtype=np.int64),
    }
    return dataset, metadata


class FactorialBindingModel(nn.Module):
    def __init__(self, config: p1146.ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.backbone = p1146.TinyCausalTransformer(config)
        self.entity_head = nn.Linear(config.width, p1146.RECORD_COUNT)
        self.field_head = nn.Linear(config.width, p1146.FIELD_COUNT)
        nn.init.normal_(self.entity_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.entity_head.bias)
        nn.init.normal_(self.field_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.field_head.bias)

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, states = self.backbone(input_ids, return_states=True)
        answer_state = self.backbone.final_norm(states[-1])[:, -1, :]
        return logits, self.entity_head(answer_state), self.field_head(answer_state)


def candidate_ids(lexicon: list[int], device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [lexicon[p1146.VALUE_START + value] for value in range(p1146.VALUE_COUNT)],
        device=device,
    )


def dataset_digest(dataset: dict[str, np.ndarray]) -> str:
    return array_digest(
        dataset["inputs"],
        dataset["targets"],
        dataset["entities"],
        dataset["fields"],
        dataset["row_targets"],
        dataset["column_targets"],
    )


def grouped_majority_accuracy(labels: np.ndarray, groups: np.ndarray) -> float:
    correct = 0
    for group in np.unique(groups, axis=0):
        if groups.ndim == 1:
            mask = groups == group
        else:
            mask = np.all(groups == group, axis=1)
        counts = np.bincount(labels[mask], minlength=p1146.VALUE_COUNT)
        correct += int(np.max(counts))
    return float(correct / len(labels))


def evaluate_model(
    model: FactorialBindingModel,
    dataset: dict[str, np.ndarray],
    lexicon: list[int],
    batch_size: int,
    split_name: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    device = next(model.parameters()).device
    values = candidate_ids(lexicon, device)
    predictions: list[int] = []
    row_predictions: list[int] = []
    column_predictions: list[int] = []
    confidences: list[float] = []
    with torch.inference_mode():
        for start in range(0, len(dataset["inputs"]), batch_size):
            batch_ids = torch.from_numpy(dataset["inputs"][start : start + batch_size]).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, entity_logits, field_logits = model(batch_ids)
            answer_logits = logits[:, -1, :].float().index_select(-1, values)
            probability = torch.softmax(answer_logits, dim=-1)
            predictions.extend(torch.argmax(answer_logits, dim=-1).cpu().tolist())
            row_predictions.extend(torch.argmax(entity_logits.float(), dim=-1).cpu().tolist())
            column_predictions.extend(torch.argmax(field_logits.float(), dim=-1).cpu().tolist())
            confidences.extend(torch.max(probability, dim=-1).values.cpu().tolist())
    predicted = np.asarray(predictions, dtype=np.int64)
    predicted_rows = np.asarray(row_predictions, dtype=np.int64)
    predicted_columns = np.asarray(column_predictions, dtype=np.int64)
    correct = predicted == dataset["targets"]
    row_correct = predicted_rows == dataset["row_targets"]
    column_correct = predicted_columns == dataset["column_targets"]
    field_metrics: dict[str, float] = {}
    for field in range(p1146.FIELD_COUNT):
        mask = dataset["fields"] == field
        field_metrics[str(field)] = float(np.mean(correct[mask]))
    entity_metrics: dict[str, float] = {}
    for entity in sorted(set(dataset["entities"].tolist())):
        mask = dataset["entities"] == entity
        entity_metrics[str(entity)] = float(np.mean(correct[mask]))
    metrics = {
        "split": split_name,
        "case_count": int(len(correct)),
        "accuracy": float(np.mean(correct)),
        "minimum_field_accuracy": float(min(field_metrics.values())),
        "minimum_entity_accuracy": float(min(entity_metrics.values())),
        "entity_address_accuracy": float(np.mean(row_correct)),
        "field_address_accuracy": float(np.mean(column_correct)),
        "joint_address_accuracy": float(np.mean(row_correct & column_correct)),
        "mean_confidence": float(np.mean(confidences)),
        "per_field_accuracy": field_metrics,
        "per_entity_accuracy": entity_metrics,
        "dataset_digest": dataset_digest(dataset),
    }
    rows = [
        {
            "split": split_name,
            "index": index,
            "query_entity": int(dataset["entities"][index]),
            "query_field": int(dataset["fields"][index]),
            "target_value": int(dataset["targets"][index]),
            "predicted_value": int(predicted[index]),
            "target_row": int(dataset["row_targets"][index]),
            "predicted_row": int(predicted_rows[index]),
            "target_column": int(dataset["column_targets"][index]),
            "predicted_column": int(predicted_columns[index]),
            "correct": bool(correct[index]),
            "confidence": float(confidences[index]),
        }
        for index in range(len(correct))
    ]
    return metrics, rows


def evaluate_all(
    model: FactorialBindingModel,
    replicate_spec: dict[str, Any],
    prereg: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, str]]:
    pairs = prereg["data"]["pairs"]
    lexicon = replicate_spec["lexicon"]
    count = int(prereg["data"]["evaluation_count"])
    batch_size = int(replicate_spec["training"]["evaluation_batch_size"])
    seen = make_factorized_dataset(
        count, pairs["train"], int(replicate_spec["data_seeds"]["seen_evaluation"]), lexicon
    )
    holdout = make_factorized_dataset(
        count,
        pairs[replicate_spec["split"]],
        int(replicate_spec["data_seeds"]["holdout_evaluation"]),
        lexicon,
    )
    quartet, quartet_metadata = make_factorized_quartets(
        pairs[replicate_spec["split"]],
        int(replicate_spec["data_seeds"]["quartet"]),
        lexicon,
    )
    evaluation: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    digests: dict[str, str] = {}
    for name, dataset in (("seen", seen), ("holdout", holdout), ("quartet", quartet)):
        metrics, split_rows = evaluate_model(model, dataset, lexicon, batch_size, name)
        evaluation[name] = metrics
        rows.extend(split_rows)
        digests[name] = metrics["dataset_digest"]
    quartet_rows = [row for row in rows if row["split"] == "quartet"]
    for row, metadata in zip(quartet_rows, quartet_metadata):
        row.update(metadata)
    return evaluation, rows, digests


def answer_gate(evaluation: dict[str, Any], thresholds: dict[str, float]) -> dict[str, bool]:
    return {
        "seen_accuracy": evaluation["seen"]["accuracy"] >= thresholds["seen_accuracy"],
        "holdout_accuracy": evaluation["holdout"]["accuracy"] >= thresholds["holdout_accuracy"],
        "quartet_accuracy": evaluation["quartet"]["accuracy"] >= thresholds["quartet_accuracy"],
        "holdout_field_floor": evaluation["holdout"]["minimum_field_accuracy"]
        >= thresholds["minimum_field_accuracy"],
        "holdout_entity_floor": evaluation["holdout"]["minimum_entity_accuracy"]
        >= thresholds["minimum_entity_accuracy"],
        "quartet_field_floor": evaluation["quartet"]["minimum_field_accuracy"]
        >= thresholds["minimum_field_accuracy"],
    }


def auxiliary_gate(
    evaluation: dict[str, Any], arm: str, thresholds: dict[str, float]
) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    if ARMS[arm]["entity_loss"]:
        checks["entity_address_holdout"] = (
            evaluation["holdout"]["entity_address_accuracy"]
            >= thresholds["auxiliary_address_accuracy"]
        )
        checks["entity_address_quartet"] = (
            evaluation["quartet"]["entity_address_accuracy"]
            >= thresholds["auxiliary_address_accuracy"]
        )
    if ARMS[arm]["field_loss"]:
        checks["field_address_holdout"] = (
            evaluation["holdout"]["field_address_accuracy"]
            >= thresholds["auxiliary_address_accuracy"]
        )
        checks["field_address_quartet"] = (
            evaluation["quartet"]["field_address_accuracy"]
            >= thresholds["auxiliary_address_accuracy"]
        )
    return checks


def learning_rate(step: int, training: dict[str, Any]) -> float:
    return p1146.learning_rate(step, training)


def verify_preregistration(prereg: dict[str, Any]) -> None:
    body = dict(prereg)
    expected = body.pop("protocol_digest")
    if canonical_digest(body) != expected:
        raise RuntimeError("Phase1147 protocol digest mismatch")
    if file_sha256(Path(__file__).resolve()) != prereg["source_hashes"]["primary_script"]:
        raise RuntimeError("Phase1147 primary script changed after preregistration")


def make_training_material(
    replicate_spec: dict[str, Any], prereg: dict[str, Any]
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    training = replicate_spec["training"]
    dataset = make_factorized_dataset(
        int(prereg["data"]["training_count"]),
        prereg["data"]["pairs"]["train"],
        int(replicate_spec["data_seeds"]["training"]),
        replicate_spec["lexicon"],
    )
    schedule_rng = np.random.default_rng(int(replicate_spec["sampler_seed"]))
    schedule = schedule_rng.integers(
        0,
        len(dataset["inputs"]),
        size=(int(training["max_steps"]), int(training["batch_size"])),
        dtype=np.int64,
    )
    return dataset, schedule


def train_arm(
    replicate: str,
    arm: str,
    prereg: dict[str, Any],
    output_root: Path = OUT_ROOT,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("Phase1147 requires CUDA")
    if arm not in ARMS:
        raise ValueError(f"Unknown arm: {arm}")
    replicate_spec = prereg["replicates"][replicate]
    if replicate_spec["split"] == "confirmation":
        selection_path = OUT_ROOT / "analysis" / "discovery_selection.json"
        if not selection_path.exists() or not read_json(selection_path)["confirmation_authorized"]:
            raise RuntimeError("Confirmation training is not authorized")
    set_seed(int(replicate_spec["training_seed"]))
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    config = p1146.ModelConfig(**replicate_spec["architecture"])
    model = FactorialBindingModel(config).to(device)
    initial_digest = state_digest(model.state_dict())
    training_data, schedule = make_training_material(replicate_spec, prereg)
    training_digest = dataset_digest(training_data)
    schedule_digest = array_digest(schedule)
    train_inputs = torch.from_numpy(training_data["inputs"])
    train_targets = torch.from_numpy(training_data["targets"])
    train_rows = torch.from_numpy(training_data["row_targets"])
    train_columns = torch.from_numpy(training_data["column_targets"])
    value_ids = candidate_ids(replicate_spec["lexicon"], device)
    training = replicate_spec["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        betas=(0.9, 0.95),
    )
    logs: list[dict[str, Any]] = []
    nonfinite_steps = 0
    torch.cuda.reset_peak_memory_stats()
    model.train()
    for step in range(int(training["max_steps"])):
        lr = learning_rate(step, training)
        for group in optimizer.param_groups:
            group["lr"] = lr
        indices = torch.from_numpy(schedule[step])
        batch_ids = train_inputs[indices].to(device, non_blocking=True)
        batch_targets = train_targets[indices].to(device, non_blocking=True)
        batch_rows = train_rows[indices].to(device, non_blocking=True)
        batch_columns = train_columns[indices].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, entity_logits, field_logits = model(batch_ids)
            answer_logits = logits[:, -1, :].index_select(-1, value_ids)
            answer_loss = F.cross_entropy(answer_logits.float(), batch_targets)
            entity_loss = F.cross_entropy(entity_logits.float(), batch_rows)
            field_loss = F.cross_entropy(field_logits.float(), batch_columns)
            loss = answer_loss
            if ARMS[arm]["entity_loss"]:
                loss = loss + float(training["entity_loss_weight"]) * entity_loss
            if ARMS[arm]["field_loss"]:
                loss = loss + float(training["field_loss_weight"]) * field_loss
        if not torch.isfinite(loss):
            nonfinite_steps += 1
            raise RuntimeError(f"Nonfinite loss for {replicate}/{arm} at step {step + 1}")
        loss.backward()
        if not all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        ):
            nonfinite_steps += 1
            raise RuntimeError(f"Nonfinite gradient for {replicate}/{arm} at step {step + 1}")
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.grad is not None],
            float(training["gradient_clip_norm"]),
        )
        optimizer.step()
        current_step = step + 1
        if current_step % int(training["log_interval"]) == 0 or current_step == int(
            training["max_steps"]
        ):
            with torch.no_grad():
                answer_accuracy = float(
                    (torch.argmax(answer_logits, dim=-1) == batch_targets).float().mean().cpu()
                )
                entity_accuracy = float(
                    (torch.argmax(entity_logits, dim=-1) == batch_rows).float().mean().cpu()
                )
                field_accuracy = float(
                    (torch.argmax(field_logits, dim=-1) == batch_columns).float().mean().cpu()
                )
            log = {
                "step": current_step,
                "loss": float(loss.detach().cpu()),
                "answer_loss": float(answer_loss.detach().cpu()),
                "entity_loss": float(entity_loss.detach().cpu()),
                "field_loss": float(field_loss.detach().cpu()),
                "batch_answer_accuracy": answer_accuracy,
                "batch_entity_address_accuracy": entity_accuracy,
                "batch_field_address_accuracy": field_accuracy,
                "learning_rate": lr,
                "gradient_norm": float(torch.as_tensor(gradient_norm).detach().cpu()),
            }
            logs.append(log)
            print(
                json.dumps({"replicate": replicate, "arm": arm, **log}, sort_keys=True),
                flush=True,
            )
    evaluation, prediction_rows, evaluation_digests = evaluate_all(model, replicate_spec, prereg)
    answer_checks = answer_gate(evaluation, prereg["thresholds"])
    auxiliary_checks = auxiliary_gate(evaluation, arm, prereg["thresholds"])
    qualified = all(answer_checks.values()) and all(auxiliary_checks.values())
    run_dir = output_root / "runs" / replicate_spec["split"] / replicate / arm
    model_path = run_dir / "model.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "phase": PHASE,
        "replicate": replicate,
        "arm": arm,
        "protocol_digest": prereg["protocol_digest"],
        "config": asdict(config),
        "lexicon": replicate_spec["lexicon"],
        "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
    }
    torch.save(checkpoint, model_path)
    predictions_path = run_dir / "predictions.jsonl"
    write_jsonl(predictions_path, prediction_rows)
    summary = {
        "phase": PHASE,
        "replicate": replicate,
        "arm": arm,
        "arm_definition": ARMS[arm],
        "split": replicate_spec["split"],
        "scale": replicate_spec["scale"],
        "protocol_digest": prereg["protocol_digest"],
        "architecture": asdict(config),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "training_seed": replicate_spec["training_seed"],
        "lexicon_seed": replicate_spec["lexicon_seed"],
        "initial_state_digest": initial_digest,
        "training_dataset_digest": training_digest,
        "batch_schedule_digest": schedule_digest,
        "training_steps": int(training["max_steps"]),
        "nonfinite_steps": nonfinite_steps,
        "training_logs": logs,
        "evaluation": evaluation,
        "evaluation_dataset_digests": evaluation_digests,
        "answer_gate_checks": answer_checks,
        "auxiliary_gate_checks": auxiliary_checks,
        "qualified": qualified,
        "model_path": str(model_path.relative_to(ROOT)).replace("\\", "/"),
        "model_sha256": file_sha256(model_path),
        "predictions_path": str(predictions_path.relative_to(ROOT)).replace("\\", "/"),
        "predictions_sha256": file_sha256(predictions_path),
        "peak_allocated_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "evidence_scope": "paired_factorial_training_intervention_behavior_only",
    }
    summary["summary_digest"] = canonical_digest(summary)
    write_json(run_dir / "summary.json", summary)
    del optimizer, model, train_inputs, train_targets, train_rows, train_columns
    gc.collect()
    torch.cuda.empty_cache()
    return summary


def replicate_specifications() -> dict[str, dict[str, Any]]:
    common_training = {
        "learning_rate": 0.003,
        "minimum_learning_rate": 0.0003,
        "warmup_steps": 100,
        "weight_decay": 0.01,
        "gradient_clip_norm": 1.0,
        "entity_loss_weight": 1.0,
        "field_loss_weight": 1.0,
        "log_interval": 500,
        "evaluation_batch_size": 512,
    }
    definitions = [
        ("discovery_small_r1", "discovery", "small", 114701),
        ("discovery_small_r2", "discovery", "small", 114702),
        ("discovery_medium_r1", "discovery", "medium", 114703),
        ("discovery_medium_r2", "discovery", "medium", 114704),
        ("confirmation_small_r1", "confirmation", "small", 114705),
        ("confirmation_small_r2", "confirmation", "small", 114706),
        ("confirmation_medium_r1", "confirmation", "medium", 114707),
        ("confirmation_medium_r2", "confirmation", "medium", 114708),
    ]
    result: dict[str, dict[str, Any]] = {}
    for offset, (name, split, scale, training_seed) in enumerate(definitions):
        small = scale == "small"
        result[name] = {
            "split": split,
            "scale": scale,
            "training_seed": training_seed,
            "lexicon_seed": 114711 + offset,
            "sampler_seed": 114721 + offset,
            "architecture": asdict(
                p1146.ModelConfig(
                    layers=4 if small else 6,
                    width=64 if small else 96,
                    heads=4,
                    mlp_width=256 if small else 384,
                )
            ),
            "training": {
                **common_training,
                "batch_size": 512 if small else 384,
                "max_steps": 2500 if small else 3000,
            },
            "data_seeds": {
                "training": 114731 + offset,
                "seen_evaluation": 114741 + offset,
                "holdout_evaluation": 114751 + offset,
                "quartet": 114761 + offset,
            },
        }
        result[name]["lexicon"] = p1146.make_lexicon(int(result[name]["lexicon_seed"]))
    return result


def protocol_body() -> dict[str, Any]:
    return {
        "phase": PHASE,
        "title": "Paired 2x2 causal formation experiment for entity-field binding",
        "claim_scope": (
            "Training-level causal effect of entity-address and field-address auxiliary losses on exact "
            "joint lookup. The experiment does not identify a hidden mechanism or generalize to natural language."
        ),
        "source_hashes": {"primary_script": file_sha256(Path(__file__).resolve())},
        "semantics": {
            "entity_count": p1146.ENTITY_COUNT,
            "field_count": p1146.FIELD_COUNT,
            "value_count": p1146.VALUE_COUNT,
            "record_count": p1146.RECORD_COUNT,
            "sequence_length": p1146.SEQUENCE_LENGTH,
            "task": "retrieve the value at the intersection of a shuffled record row and shared shuffled field column",
            "entity_auxiliary_target": "record row index of the queried entity",
            "field_auxiliary_target": "shared within-record column index of the queried field",
            "nonleakage": (
                "Neither address target identifies the cell value alone; both heads are parallel training-only "
                "readouts and do not feed logits or each other."
            ),
        },
        "arms": ARMS,
        "arm_priority_for_mechanism_candidate": ARM_PRIORITY,
        "data": {
            "pairs": p1146.semantic_pairs(),
            "training_count": 65536,
            "evaluation_count": 4096,
            "field_order_control": "one random field order shared by all three records within each example",
            "surface_control": "one full non-special-token permutation per paired replicate; independent across replicates",
            "training_holdout": "even-parity entity-field pairs never occur as training queries",
        },
        "replicates": replicate_specifications(),
        "thresholds": {
            "seen_accuracy": 0.995,
            "holdout_accuracy": 0.95,
            "quartet_accuracy": 0.95,
            "minimum_field_accuracy": 0.90,
            "minimum_entity_accuracy": 0.80,
            "auxiliary_address_accuracy": 0.95,
            "dual_over_single_accuracy": 0.15,
            "factorial_interaction": 0.10,
        },
        "selection": {
            "replicates_per_scale_per_split": 2,
            "confirmation_requires_one_arm_qualified_in_every_discovery_replicate": True,
            "confirmation_runs_all_four_arms": True,
            "candidate_arm_priority": ARM_PRIORITY,
            "mechanism_requires_same_selected_arm_qualified_in_every_confirmation_replicate": True,
            "dual_synergy_is_secondary_and_requires_all_replicates": True,
        },
        "forbidden": [
            "No arm-specific seeds, examples, batch schedules, token budgets, steps, or architectures",
            "No early stopping or arm-specific training extension",
            "No threshold or auxiliary-loss-weight changes after protocol creation",
            "No confirmation unless one arm passes every discovery replicate",
            "No hidden-state or component analysis unless the same selected arm passes independent confirmation",
            "No interpretation of auxiliary-head accuracy as exact binding",
            "No direct comparison of Phase1146 and Phase1147 answer-only accuracy as a pure curriculum effect because field-order structure changed",
        ],
    }


def create_protocol() -> dict[str, Any]:
    body = protocol_body()
    prereg = dict(body)
    prereg["protocol_digest"] = canonical_digest(body)
    if PREREG_PATH.exists():
        existing = read_json(PREREG_PATH)
        if existing != prereg:
            raise RuntimeError("Existing Phase1147 protocol differs from current script")
    else:
        write_json(PREREG_PATH, prereg)
    replicates = prereg["replicates"]
    audit_spec = replicates["discovery_small_r1"]
    audit_dataset = make_factorized_dataset(
        int(prereg["data"]["training_count"]),
        prereg["data"]["pairs"]["train"],
        int(audit_spec["data_seeds"]["training"]),
        audit_spec["lexicon"],
    )
    row_only_accuracy = grouped_majority_accuracy(
        audit_dataset["targets"], audit_dataset["row_targets"]
    )
    column_only_accuracy = grouped_majority_accuracy(
        audit_dataset["targets"], audit_dataset["column_targets"]
    )
    address_only_accuracy = grouped_majority_accuracy(
        audit_dataset["targets"],
        np.stack([audit_dataset["row_targets"], audit_dataset["column_targets"]], axis=1),
    )
    checks = {
        "four_arms": set(prereg["arms"]) == set(ARMS),
        "eight_paired_replicates": len(replicates) == 8,
        "two_replicates_per_scale_per_split": all(
            sum(
                1
                for spec in replicates.values()
                if spec["split"] == split and spec["scale"] == scale
            )
            == 2
            for split in ("discovery", "confirmation")
            for scale in ("small", "medium")
        ),
        "independent_replicate_lexicons": len(
            {tuple(spec["lexicon"]) for spec in replicates.values()}
        )
        == len(replicates),
        "pair_partition_complete": len(
            {
                tuple(pair)
                for pairs in prereg["data"]["pairs"].values()
                for pair in pairs
            }
        )
        == p1146.ENTITY_COUNT * p1146.FIELD_COUNT,
        "source_hash_matches": file_sha256(Path(__file__).resolve())
        == prereg["source_hashes"]["primary_script"],
        "all_row_addresses_present": set(audit_dataset["row_targets"].tolist())
        == set(range(p1146.RECORD_COUNT)),
        "all_column_addresses_present": set(audit_dataset["column_targets"].tolist())
        == set(range(p1146.FIELD_COUNT)),
        "row_target_does_not_leak_value": row_only_accuracy <= 0.14,
        "column_target_does_not_leak_value": column_only_accuracy <= 0.14,
        "joint_address_labels_do_not_leak_value_without_records": address_only_accuracy <= 0.15,
    }
    audit = {
        "phase": PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "checks": checks,
        "nonleakage_majority_accuracies": {
            "row_only": row_only_accuracy,
            "column_only": column_only_accuracy,
            "row_and_column_without_records": address_only_accuracy,
            "uniform_value_chance": 1.0 / p1146.VALUE_COUNT,
        },
        "all_checks_passed": all(checks.values()),
    }
    audit["audit_digest"] = canonical_digest(audit)
    write_json(OUT_ROOT / "protocol" / "audit.json", audit)
    if not audit["all_checks_passed"]:
        raise RuntimeError("Phase1147 protocol audit failed")
    return prereg


def split_replicates(prereg: dict[str, Any], split: str) -> list[str]:
    return [
        name for name, spec in prereg["replicates"].items() if spec["split"] == split
    ]


def load_summary(replicate: str, arm: str, prereg: dict[str, Any]) -> dict[str, Any]:
    split = prereg["replicates"][replicate]["split"]
    return read_json(OUT_ROOT / "runs" / split / replicate / arm / "summary.json")


def analyze_split(split: str, prereg: dict[str, Any]) -> dict[str, Any]:
    replicates = split_replicates(prereg, split)
    summaries = {
        replicate: {arm: load_summary(replicate, arm, prereg) for arm in ARMS}
        for replicate in replicates
    }
    arm_all_qualified = {
        arm: all(summaries[replicate][arm]["qualified"] for replicate in replicates)
        for arm in ARMS
    }
    eligible_arms = [arm for arm in ARM_PRIORITY if arm_all_qualified[arm]]
    factorial: dict[str, Any] = {}
    threshold = prereg["thresholds"]
    for replicate in replicates:
        factorial[replicate] = {}
        for evaluation_split in ("seen", "holdout", "quartet"):
            accuracy = {
                arm: summaries[replicate][arm]["evaluation"][evaluation_split]["accuracy"]
                for arm in ARMS
            }
            interaction = accuracy["EF"] - accuracy["E0"] - accuracy["0F"] + accuracy["00"]
            over_single = accuracy["EF"] - max(accuracy["E0"], accuracy["0F"])
            factorial[replicate][evaluation_split] = {
                "accuracy": accuracy,
                "factorial_interaction": interaction,
                "dual_over_best_single": over_single,
                "interaction_pass": interaction >= threshold["factorial_interaction"],
                "dual_over_single_pass": over_single >= threshold["dual_over_single_accuracy"],
            }
    dual_synergy_pass = all(
        factorial[replicate][evaluation_split]["interaction_pass"]
        and factorial[replicate][evaluation_split]["dual_over_single_pass"]
        for replicate in replicates
        for evaluation_split in ("holdout", "quartet")
    ) and arm_all_qualified["EF"]
    discovery_selection = None
    if split == "confirmation":
        discovery_selection = read_json(OUT_ROOT / "analysis" / "discovery_selection.json")
        selected_arm = discovery_selection["selected_arm"]
    else:
        selected_arm = eligible_arms[0] if eligible_arms else None
    selected_arm_qualified = bool(selected_arm) and arm_all_qualified[str(selected_arm)]
    result = {
        "phase": PHASE,
        "split": split,
        "protocol_digest": prereg["protocol_digest"],
        "replicates": replicates,
        "summary_digests": {
            replicate: {
                arm: summaries[replicate][arm]["summary_digest"] for arm in ARMS
            }
            for replicate in replicates
        },
        "arm_all_qualified": arm_all_qualified,
        "eligible_arms": eligible_arms,
        "selected_arm": selected_arm,
        "factorial": factorial,
        "dual_synergy_pass": dual_synergy_pass,
        "confirmation_authorized": bool(eligible_arms) if split == "discovery" else None,
        "mechanism_phase_authorized": bool(selected_arm_qualified)
        if split == "confirmation"
        else None,
        "evidence_scope": "training_intervention_behavioral_selection_only",
    }
    result["selection_digest"] = canonical_digest(result)
    write_json(OUT_ROOT / "analysis" / f"{split}_selection.json", result)
    return result


def finalize(prereg: dict[str, Any]) -> dict[str, Any]:
    discovery = read_json(OUT_ROOT / "analysis" / "discovery_selection.json")
    confirmation_path = OUT_ROOT / "analysis" / "confirmation_selection.json"
    confirmation = read_json(confirmation_path) if confirmation_path.exists() else None
    mechanism_authorized = bool(
        confirmation is not None and confirmation["mechanism_phase_authorized"]
    )
    outcome = (
        "confirmed_behavioral_formation_object"
        if mechanism_authorized
        else "discovery_behavior_gate_failed"
        if not discovery["confirmation_authorized"]
        else "confirmation_behavior_gate_failed"
    )
    result = {
        "phase": PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "outcome": outcome,
        "discovery_selection_digest": discovery["selection_digest"],
        "confirmation_selection_digest": confirmation["selection_digest"]
        if confirmation
        else None,
        "selected_arm": discovery["selected_arm"],
        "discovery_dual_synergy_pass": discovery["dual_synergy_pass"],
        "confirmation_dual_synergy_pass": confirmation["dual_synergy_pass"]
        if confirmation
        else None,
        "confirmation_authorized": discovery["confirmation_authorized"],
        "mechanism_phase_authorized": mechanism_authorized,
        "auto_continue": mechanism_authorized,
        "claim_boundary": (
            "A qualified arm establishes a learned behavioral lookup object under this synthetic protocol. "
            "It does not establish hidden factor modules, natural-language composition, or human-brain homology."
        ),
    }
    result["final_digest"] = canonical_digest(result)
    write_json(OUT_ROOT / "analysis" / "final.json", result)
    return result


def run_split(split: str, prereg: dict[str, Any]) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for replicate in split_replicates(prereg, split):
        summaries[replicate] = {}
        for arm in ARM_PRIORITY:
            summaries[replicate][arm] = train_arm(replicate, arm, prereg)
    return summaries


def smoke() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("Phase1147 smoke requires CUDA")
    spec = replicate_specifications()["discovery_small_r1"]
    mini_prereg = {
        "data": {
            "pairs": p1146.semantic_pairs(),
            "training_count": 512,
            "evaluation_count": 128,
        },
        "thresholds": {
            "seen_accuracy": 0.0,
            "holdout_accuracy": 0.0,
            "quartet_accuracy": 0.0,
            "minimum_field_accuracy": 0.0,
            "minimum_entity_accuracy": 0.0,
            "auxiliary_address_accuracy": 0.0,
        },
    }
    spec = json.loads(json.dumps(spec))
    spec["training"]["max_steps"] = 3
    spec["training"]["batch_size"] = 32
    spec["training"]["evaluation_batch_size"] = 64
    mini_prereg["replicates"] = {"smoke": spec}
    initial: dict[str, str] = {}
    material: dict[str, str] = {}
    schedules: dict[str, str] = {}
    losses: dict[str, float] = {}
    gradients_finite: dict[str, bool] = {}
    for arm in ARM_PRIORITY:
        set_seed(int(spec["training_seed"]))
        model = FactorialBindingModel(p1146.ModelConfig(**spec["architecture"])).cuda()
        initial[arm] = state_digest(model.state_dict())
        dataset, schedule = make_training_material(spec, mini_prereg)
        material[arm] = dataset_digest(dataset)
        schedules[arm] = array_digest(schedule)
        batch = torch.from_numpy(dataset["inputs"][schedule[0]]).cuda()
        targets = torch.from_numpy(dataset["targets"][schedule[0]]).cuda()
        rows = torch.from_numpy(dataset["row_targets"][schedule[0]]).cuda()
        columns = torch.from_numpy(dataset["column_targets"][schedule[0]]).cuda()
        values = candidate_ids(spec["lexicon"], torch.device("cuda"))
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, entity_logits, field_logits = model(batch)
            loss = F.cross_entropy(logits[:, -1, :].index_select(-1, values).float(), targets)
            if ARMS[arm]["entity_loss"]:
                loss = loss + F.cross_entropy(entity_logits.float(), rows)
            if ARMS[arm]["field_loss"]:
                loss = loss + F.cross_entropy(field_logits.float(), columns)
        losses[arm] = float(loss.detach().cpu())
        loss.backward()
        gradients_finite[arm] = all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        )
        del model
        torch.cuda.empty_cache()
    result = {
        "phase": PHASE,
        "all_initial_states_paired": len(set(initial.values())) == 1,
        "all_training_material_paired": len(set(material.values())) == 1,
        "all_batch_schedules_paired": len(set(schedules.values())) == 1,
        "initial_state_digests": initial,
        "training_material_digests": material,
        "batch_schedule_digests": schedules,
        "first_batch_losses": losses,
        "first_backward_gradients_finite": gradients_finite,
    }
    result["all_checks_passed"] = all(
        result[key]
        for key in (
            "all_initial_states_paired",
            "all_training_material_paired",
            "all_batch_schedules_paired",
        )
    ) and all(np.isfinite(list(losses.values()))) and all(gradients_finite.values())
    result["smoke_digest"] = canonical_digest(result)
    write_json(TEMP_ROOT / "smoke.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=[
            "smoke",
            "create-protocol",
            "train-arm",
            "run-split",
            "analyze-split",
            "finalize",
        ],
        required=True,
    )
    parser.add_argument("--replicate")
    parser.add_argument("--arm", choices=list(ARMS))
    parser.add_argument("--split", choices=["discovery", "confirmation"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "smoke":
        result = smoke()
    elif args.mode == "create-protocol":
        result = create_protocol()
    else:
        prereg = read_json(PREREG_PATH)
        verify_preregistration(prereg)
        if args.mode == "train-arm":
            if not args.replicate or not args.arm:
                raise ValueError("--replicate and --arm are required")
            result = train_arm(args.replicate, args.arm, prereg)
        elif args.mode == "run-split":
            if not args.split:
                raise ValueError("--split is required")
            result = run_split(args.split, prereg)
        elif args.mode == "analyze-split":
            if not args.split:
                raise ValueError("--split is required")
            result = analyze_split(args.split, prereg)
        else:
            result = finalize(prereg)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
