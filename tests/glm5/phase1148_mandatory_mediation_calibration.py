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

import phase1147_factorial_binding_formation as p1147


PHASE = 1148
ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "tests" / "glm5" / "result" / "phase1148_mandatory_mediation_calibration"
TEMP_ROOT = ROOT / "tests" / "glm5_temp" / "phase1148_smoke"
PREREG_PATH = OUT_ROOT / "protocol" / "preregistration.json"

CONDITIONS = {
    "free_00": {"mandatory_mediation": False, "address_auxiliary": False},
    "free_EF": {"mandatory_mediation": False, "address_auxiliary": True},
    "soft_00": {"mandatory_mediation": True, "address_auxiliary": False},
    "soft_EF": {"mandatory_mediation": True, "address_auxiliary": True},
}
CONDITION_ORDER = ["free_00", "free_EF", "soft_00", "soft_EF"]
SOFT_PRIORITY = ["soft_00", "soft_EF"]


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


def make_dataset(
    count: int,
    pairs: list[list[int]],
    seed: int,
    lexicon: list[int],
) -> dict[str, np.ndarray]:
    base = p1147.p1146
    rng = np.random.default_rng(seed)
    inputs = np.empty((count, base.SEQUENCE_LENGTH), dtype=np.int64)
    targets = np.empty(count, dtype=np.int64)
    entities_meta = np.empty(count, dtype=np.int64)
    fields_meta = np.empty(count, dtype=np.int64)
    row_targets = np.empty(count, dtype=np.int64)
    column_targets = np.empty(count, dtype=np.int64)
    grid_values = np.empty(
        (count, base.RECORD_COUNT, base.FIELD_COUNT), dtype=np.int64
    )
    pair_order = [tuple(pair) for pair in pairs]
    for index in range(count):
        query_entity, query_field = pair_order[index % len(pair_order)]
        target_value = (index // len(pair_order)) % base.VALUE_COUNT
        distractors = rng.choice(
            [entity for entity in range(base.ENTITY_COUNT) if entity != query_entity],
            size=base.RECORD_COUNT - 1,
            replace=False,
        ).tolist()
        entities = [query_entity, *[int(value) for value in distractors]]
        assignments = {
            entity: rng.integers(
                0, base.VALUE_COUNT, size=base.FIELD_COUNT, dtype=np.int64
            ).tolist()
            for entity in entities
        }
        assignments[query_entity][query_field] = int(target_value)
        record_order = list(entities)
        rng.shuffle(record_order)
        shared_field_order = list(range(base.FIELD_COUNT))
        rng.shuffle(shared_field_order)
        field_orders = {entity: list(shared_field_order) for entity in entities}
        sequence, _ = base.build_sequence(
            entities,
            assignments,
            query_entity,
            query_field,
            record_order,
            field_orders,
            lexicon,
        )
        row_target = record_order.index(query_entity)
        column_target = shared_field_order.index(query_field)
        grid = np.asarray(
            [
                [assignments[entity][field] for field in shared_field_order]
                for entity in record_order
            ],
            dtype=np.int64,
        )
        if int(grid[row_target, column_target]) != target_value:
            raise RuntimeError("Grid construction does not match target")
        inputs[index] = sequence
        targets[index] = target_value
        entities_meta[index] = query_entity
        fields_meta[index] = query_field
        row_targets[index] = row_target
        column_targets[index] = column_target
        grid_values[index] = grid
    permutation = rng.permutation(count)
    return {
        "inputs": inputs[permutation],
        "targets": targets[permutation],
        "entities": entities_meta[permutation],
        "fields": fields_meta[permutation],
        "row_targets": row_targets[permutation],
        "column_targets": column_targets[permutation],
        "grid_values": grid_values[permutation],
    }


def make_quartets(
    pairs: list[list[int]],
    seed: int,
    lexicon: list[int],
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    base = p1147.p1146
    rng = np.random.default_rng(seed)
    rows: list[np.ndarray] = []
    labels: list[int] = []
    entities_meta: list[int] = []
    fields_meta: list[int] = []
    row_targets: list[int] = []
    column_targets: list[int] = []
    grids: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    state_names = ["active_minus", "active_plus", "null_minus", "null_plus"]
    for item_index, pair in enumerate(pairs):
        query_entity, query_field = int(pair[0]), int(pair[1])
        destination = 1 + (item_index % 6)
        base_value = 0
        anchor = 7
        distractors = rng.choice(
            [entity for entity in range(base.ENTITY_COUNT) if entity != query_entity],
            size=2,
            replace=False,
        ).tolist()
        entity1, entity2 = int(distractors[0]), int(distractors[1])
        entities = [query_entity, entity1, entity2]
        common_assignments = {
            entity: rng.integers(
                0, base.VALUE_COUNT, size=base.FIELD_COUNT, dtype=np.int64
            ).tolist()
            for entity in entities
        }
        record_order = list(entities)
        rng.shuffle(record_order)
        shared_field_order = list(range(base.FIELD_COUNT))
        rng.shuffle(shared_field_order)
        field_orders = {entity: list(shared_field_order) for entity in entities}
        state_values = [
            (base_value, destination, anchor, base_value),
            (destination, base_value, anchor, destination),
            (anchor, base_value, destination, anchor),
            (anchor, destination, base_value, anchor),
        ]
        for state_index, (value0, value1, value2, target) in enumerate(state_values):
            assignments = {entity: list(values) for entity, values in common_assignments.items()}
            assignments[query_entity][query_field] = value0
            assignments[entity1][query_field] = value1
            assignments[entity2][query_field] = value2
            sequence, positions = base.build_sequence(
                entities,
                assignments,
                query_entity,
                query_field,
                record_order,
                field_orders,
                lexicon,
            )
            row_target = record_order.index(query_entity)
            column_target = shared_field_order.index(query_field)
            grid = np.asarray(
                [
                    [assignments[entity][field] for field in shared_field_order]
                    for entity in record_order
                ],
                dtype=np.int64,
            )
            if int(grid[row_target, column_target]) != target:
                raise RuntimeError("Quartet grid construction does not match target")
            rows.append(sequence)
            labels.append(target)
            entities_meta.append(query_entity)
            fields_meta.append(query_field)
            row_targets.append(row_target)
            column_targets.append(column_target)
            grids.append(grid)
            metadata.append(
                {
                    "item_index": item_index,
                    "item_id": f"e{query_entity:02d}.f{query_field}.d{destination}",
                    "state_index": state_index,
                    "state": state_names[state_index],
                    "query_entity": query_entity,
                    "query_field": query_field,
                    "target_value": target,
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
        "grid_values": np.stack(grids),
    }
    return dataset, metadata


def dataset_digest(dataset: dict[str, np.ndarray]) -> str:
    return array_digest(
        dataset["inputs"],
        dataset["targets"],
        dataset["entities"],
        dataset["fields"],
        dataset["row_targets"],
        dataset["column_targets"],
        dataset["grid_values"],
    )


class MediationBindingModel(nn.Module):
    def __init__(self, config: p1147.p1146.ModelConfig) -> None:
        super().__init__()
        base = p1147.p1146
        self.config = config
        self.backbone = base.TinyCausalTransformer(config)
        self.row_head = nn.Linear(config.width, base.RECORD_COUNT)
        self.column_head = nn.Linear(config.width, base.FIELD_COUNT)
        nn.init.normal_(self.row_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.row_head.bias)
        nn.init.normal_(self.column_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.column_head.bias)

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, states = self.backbone(input_ids, return_states=True)
        answer_state = self.backbone.final_norm(states[-1])[:, -1, :]
        return logits, self.row_head(answer_state), self.column_head(answer_state)


def candidate_ids(lexicon: list[int], device: torch.device) -> torch.Tensor:
    base = p1147.p1146
    return torch.tensor(
        [lexicon[base.VALUE_START + value] for value in range(base.VALUE_COUNT)],
        device=device,
    )


def mediated_distribution(
    row_logits: torch.Tensor,
    column_logits: torch.Tensor,
    grid_values: torch.Tensor,
    row_mode: str = "predicted",
    column_mode: str = "predicted",
    row_targets: torch.Tensor | None = None,
    column_targets: torch.Tensor | None = None,
) -> torch.Tensor:
    base = p1147.p1146
    if row_mode == "predicted":
        row_probability = torch.softmax(row_logits.float(), dim=-1)
    elif row_mode == "uniform":
        row_probability = torch.full_like(row_logits.float(), 1.0 / base.RECORD_COUNT)
    elif row_mode == "oracle":
        if row_targets is None:
            raise ValueError("row_targets required for oracle row mode")
        row_probability = F.one_hot(row_targets, num_classes=base.RECORD_COUNT).float()
    else:
        raise ValueError(f"Unknown row mode: {row_mode}")
    if column_mode == "predicted":
        column_probability = torch.softmax(column_logits.float(), dim=-1)
    elif column_mode == "uniform":
        column_probability = torch.full_like(
            column_logits.float(), 1.0 / base.FIELD_COUNT
        )
    elif column_mode == "oracle":
        if column_targets is None:
            raise ValueError("column_targets required for oracle column mode")
        column_probability = F.one_hot(
            column_targets, num_classes=base.FIELD_COUNT
        ).float()
    else:
        raise ValueError(f"Unknown column mode: {column_mode}")
    cell_weights = row_probability[:, :, None] * column_probability[:, None, :]
    value_indicators = F.one_hot(grid_values, num_classes=base.VALUE_COUNT).float()
    distribution = torch.einsum("brc,brcv->bv", cell_weights, value_indicators)
    return distribution / distribution.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def answer_loss_and_prediction(
    condition: str,
    logits: torch.Tensor,
    row_logits: torch.Tensor,
    column_logits: torch.Tensor,
    grid_values: torch.Tensor,
    targets: torch.Tensor,
    value_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if CONDITIONS[condition]["mandatory_mediation"]:
        distribution = mediated_distribution(row_logits, column_logits, grid_values)
        loss = F.nll_loss(torch.log(distribution.clamp_min(1e-9)), targets)
        prediction = torch.argmax(distribution, dim=-1)
        confidence = torch.max(distribution, dim=-1).values
    else:
        answer_logits = logits[:, -1, :].float().index_select(-1, value_ids)
        loss = F.cross_entropy(answer_logits, targets)
        prediction = torch.argmax(answer_logits, dim=-1)
        confidence = torch.max(torch.softmax(answer_logits, dim=-1), dim=-1).values
    return loss, prediction, confidence


def evaluate_model(
    model: MediationBindingModel,
    condition: str,
    dataset: dict[str, np.ndarray],
    lexicon: list[int],
    batch_size: int,
    split_name: str,
    return_rows: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    base = p1147.p1146
    model.eval()
    device = next(model.parameters()).device
    values = candidate_ids(lexicon, device)
    predictions: list[int] = []
    row_predictions: list[int] = []
    column_predictions: list[int] = []
    confidences: list[float] = []
    with torch.inference_mode():
        for start in range(0, len(dataset["inputs"]), batch_size):
            stop = start + batch_size
            batch_ids = torch.from_numpy(dataset["inputs"][start:stop]).to(device)
            batch_grids = torch.from_numpy(dataset["grid_values"][start:stop]).to(device)
            batch_targets = torch.from_numpy(dataset["targets"][start:stop]).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, row_logits, column_logits = model(batch_ids)
            _, prediction, confidence = answer_loss_and_prediction(
                condition,
                logits,
                row_logits,
                column_logits,
                batch_grids,
                batch_targets,
                values,
            )
            predictions.extend(prediction.cpu().tolist())
            row_predictions.extend(torch.argmax(row_logits.float(), dim=-1).cpu().tolist())
            column_predictions.extend(
                torch.argmax(column_logits.float(), dim=-1).cpu().tolist()
            )
            confidences.extend(confidence.cpu().tolist())
    predicted = np.asarray(predictions, dtype=np.int64)
    predicted_rows = np.asarray(row_predictions, dtype=np.int64)
    predicted_columns = np.asarray(column_predictions, dtype=np.int64)
    correct = predicted == dataset["targets"]
    row_correct = predicted_rows == dataset["row_targets"]
    column_correct = predicted_columns == dataset["column_targets"]
    field_metrics: dict[str, float] = {}
    for field in range(base.FIELD_COUNT):
        mask = dataset["fields"] == field
        field_metrics[str(field)] = float(np.mean(correct[mask]))
    entity_metrics: dict[str, float] = {}
    for entity in sorted(set(dataset["entities"].tolist())):
        mask = dataset["entities"] == entity
        entity_metrics[str(entity)] = float(np.mean(correct[mask]))
    oracle_prediction = dataset["grid_values"][
        np.arange(len(correct)), dataset["row_targets"], dataset["column_targets"]
    ]
    metrics = {
        "split": split_name,
        "case_count": int(len(correct)),
        "accuracy": float(np.mean(correct)),
        "minimum_field_accuracy": float(min(field_metrics.values())),
        "minimum_entity_accuracy": float(min(entity_metrics.values())),
        "row_address_accuracy": float(np.mean(row_correct)),
        "column_address_accuracy": float(np.mean(column_correct)),
        "joint_address_accuracy": float(np.mean(row_correct & column_correct)),
        "mean_confidence": float(np.mean(confidences)),
        "oracle_accuracy": float(np.mean(oracle_prediction == dataset["targets"])),
        "per_field_accuracy": field_metrics,
        "per_entity_accuracy": entity_metrics,
        "dataset_digest": dataset_digest(dataset),
    }
    rows: list[dict[str, Any]] = []
    if return_rows:
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


def build_evaluation_sets(
    replicate_spec: dict[str, Any],
    prereg: dict[str, Any],
    panel: str,
) -> tuple[list[tuple[str, dict[str, np.ndarray]]], list[dict[str, Any]]]:
    pairs = prereg["data"]["pairs"]
    lexicon = replicate_spec["lexicon"]
    if panel == "formal":
        count = int(prereg["data"]["evaluation_count"])
        keys = ("seen_evaluation", "holdout_evaluation", "quartet")
    elif panel == "trajectory":
        count = int(prereg["data"]["trajectory_evaluation_count"])
        keys = ("trajectory_seen", "trajectory_holdout", "trajectory_quartet")
    else:
        raise ValueError(f"Unknown panel: {panel}")
    seen = make_dataset(
        count,
        pairs["train"],
        int(replicate_spec["data_seeds"][keys[0]]),
        lexicon,
    )
    holdout = make_dataset(
        count,
        pairs[replicate_spec["split"]],
        int(replicate_spec["data_seeds"][keys[1]]),
        lexicon,
    )
    quartet, metadata = make_quartets(
        pairs[replicate_spec["split"]],
        int(replicate_spec["data_seeds"][keys[2]]),
        lexicon,
    )
    return [("seen", seen), ("holdout", holdout), ("quartet", quartet)], metadata


def evaluate_bundle(
    model: MediationBindingModel,
    condition: str,
    replicate_spec: dict[str, Any],
    prereg: dict[str, Any],
    panel: str,
    return_rows: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, str]]:
    sets, quartet_metadata = build_evaluation_sets(replicate_spec, prereg, panel)
    evaluation: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    digests: dict[str, str] = {}
    for name, dataset in sets:
        metrics, split_rows = evaluate_model(
            model,
            condition,
            dataset,
            replicate_spec["lexicon"],
            int(replicate_spec["training"]["evaluation_batch_size"]),
            name,
            return_rows,
        )
        evaluation[name] = metrics
        rows.extend(split_rows)
        digests[name] = metrics["dataset_digest"]
    if return_rows:
        quartet_rows = [row for row in rows if row["split"] == "quartet"]
        for row, metadata in zip(quartet_rows, quartet_metadata):
            row.update(metadata)
    return evaluation, rows, digests


def answer_gate(evaluation: dict[str, Any], thresholds: dict[str, float]) -> dict[str, bool]:
    return {
        "seen_accuracy": evaluation["seen"]["accuracy"] >= thresholds["seen_accuracy"],
        "holdout_accuracy": evaluation["holdout"]["accuracy"]
        >= thresholds["holdout_accuracy"],
        "quartet_accuracy": evaluation["quartet"]["accuracy"]
        >= thresholds["quartet_accuracy"],
        "holdout_field_floor": evaluation["holdout"]["minimum_field_accuracy"]
        >= thresholds["minimum_field_accuracy"],
        "holdout_entity_floor": evaluation["holdout"]["minimum_entity_accuracy"]
        >= thresholds["minimum_entity_accuracy"],
        "quartet_field_floor": evaluation["quartet"]["minimum_field_accuracy"]
        >= thresholds["minimum_field_accuracy"],
        "oracle_integrity": all(
            evaluation[split]["oracle_accuracy"] >= thresholds["oracle_accuracy"]
            for split in ("seen", "holdout", "quartet")
        ),
    }


def address_gate(
    evaluation: dict[str, Any], condition: str, thresholds: dict[str, float]
) -> dict[str, bool]:
    if not CONDITIONS[condition]["mandatory_mediation"]:
        return {}
    return {
        "holdout_row_address": evaluation["holdout"]["row_address_accuracy"]
        >= thresholds["address_accuracy"],
        "holdout_column_address": evaluation["holdout"]["column_address_accuracy"]
        >= thresholds["address_accuracy"],
        "quartet_row_address": evaluation["quartet"]["row_address_accuracy"]
        >= thresholds["address_accuracy"],
        "quartet_column_address": evaluation["quartet"]["column_address_accuracy"]
        >= thresholds["address_accuracy"],
    }


def verify_preregistration(prereg: dict[str, Any]) -> None:
    body = dict(prereg)
    expected = body.pop("protocol_digest")
    if canonical_digest(body) != expected:
        raise RuntimeError("Phase1148 protocol digest mismatch")
    source_hashes = prereg["source_hashes"]
    if file_sha256(Path(__file__).resolve()) != source_hashes["primary_script"]:
        raise RuntimeError("Phase1148 primary script changed after preregistration")
    if file_sha256(Path(p1147.__file__).resolve()) != source_hashes["phase1147_dependency"]:
        raise RuntimeError("Phase1147 dependency changed after Phase1148 preregistration")
    if (
        file_sha256(Path(p1147.p1146.__file__).resolve())
        != source_hashes["phase1146_dependency"]
    ):
        raise RuntimeError("Phase1146 dependency changed after Phase1148 preregistration")


def make_training_material(
    replicate_spec: dict[str, Any], prereg: dict[str, Any]
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    training = replicate_spec["training"]
    dataset = make_dataset(
        int(prereg["data"]["training_count"]),
        prereg["data"]["pairs"]["train"],
        int(replicate_spec["data_seeds"]["training"]),
        replicate_spec["lexicon"],
    )
    rng = np.random.default_rng(int(replicate_spec["sampler_seed"]))
    schedule = rng.integers(
        0,
        len(dataset["inputs"]),
        size=(int(training["max_steps"]), int(training["batch_size"])),
        dtype=np.int64,
    )
    return dataset, schedule


def save_checkpoint(
    model: MediationBindingModel,
    replicate: str,
    condition: str,
    split: str,
    step: int,
    prereg: dict[str, Any],
    output_root: Path,
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
        "replicate": replicate,
        "condition": condition,
        "step": step,
        "protocol_digest": prereg["protocol_digest"],
        "config": asdict(model.config),
        "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
    }
    torch.save(payload, path)
    return {
        "step": step,
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "sha256": file_sha256(path),
        "state_digest": state_digest(model.state_dict()),
    }


def train_condition(
    replicate: str,
    condition: str,
    prereg: dict[str, Any],
    output_root: Path = OUT_ROOT,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("Phase1148 requires CUDA")
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition: {condition}")
    spec = prereg["replicates"][replicate]
    if spec["split"] == "confirmation":
        selection_path = OUT_ROOT / "analysis" / "discovery_selection.json"
        if not selection_path.exists() or not read_json(selection_path)["confirmation_authorized"]:
            raise RuntimeError("Confirmation training is not authorized")
    set_seed(int(spec["training_seed"]))
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    config = p1147.p1146.ModelConfig(**spec["architecture"])
    model = MediationBindingModel(config).to(device)
    initial_digest = state_digest(model.state_dict())
    training_data, schedule = make_training_material(spec, prereg)
    training_digest = dataset_digest(training_data)
    schedule_digest = array_digest(schedule)
    train_inputs = torch.from_numpy(training_data["inputs"])
    train_targets = torch.from_numpy(training_data["targets"])
    train_rows = torch.from_numpy(training_data["row_targets"])
    train_columns = torch.from_numpy(training_data["column_targets"])
    train_grids = torch.from_numpy(training_data["grid_values"])
    values = candidate_ids(spec["lexicon"], device)
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
    nonfinite_steps = 0
    torch.cuda.reset_peak_memory_stats()

    checkpoint = save_checkpoint(
        model, replicate, condition, spec["split"], 0, prereg, output_root
    )
    initial_evaluation, _, initial_digests = evaluate_bundle(
        model, condition, spec, prereg, "trajectory", False
    )
    trajectory.append(
        {
            "step": 0,
            "evaluation": initial_evaluation,
            "dataset_digests": initial_digests,
            "checkpoint": checkpoint,
        }
    )
    model.train()
    for step in range(int(training["max_steps"])):
        lr = p1147.p1146.learning_rate(step, training)
        for group in optimizer.param_groups:
            group["lr"] = lr
        indices = torch.from_numpy(schedule[step])
        batch_ids = train_inputs[indices].to(device, non_blocking=True)
        batch_targets = train_targets[indices].to(device, non_blocking=True)
        batch_rows = train_rows[indices].to(device, non_blocking=True)
        batch_columns = train_columns[indices].to(device, non_blocking=True)
        batch_grids = train_grids[indices].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, row_logits, column_logits = model(batch_ids)
        answer_loss, prediction, _ = answer_loss_and_prediction(
            condition,
            logits,
            row_logits,
            column_logits,
            batch_grids,
            batch_targets,
            values,
        )
        row_loss = F.cross_entropy(row_logits.float(), batch_rows)
        column_loss = F.cross_entropy(column_logits.float(), batch_columns)
        loss = answer_loss
        if CONDITIONS[condition]["address_auxiliary"]:
            loss = loss + float(training["row_loss_weight"]) * row_loss
            loss = loss + float(training["column_loss_weight"]) * column_loss
        if not torch.isfinite(loss):
            nonfinite_steps += 1
            raise RuntimeError(f"Nonfinite loss for {replicate}/{condition} at step {step + 1}")
        loss.backward()
        if not all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        ):
            nonfinite_steps += 1
            raise RuntimeError(
                f"Nonfinite gradient for {replicate}/{condition} at step {step + 1}"
            )
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.grad is not None],
            float(training["gradient_clip_norm"]),
        )
        optimizer.step()
        current_step = step + 1
        if current_step % int(training["log_interval"]) == 0 or current_step in trajectory_steps:
            log = {
                "step": current_step,
                "loss": float(loss.detach().cpu()),
                "answer_loss": float(answer_loss.detach().cpu()),
                "row_loss": float(row_loss.detach().cpu()),
                "column_loss": float(column_loss.detach().cpu()),
                "batch_answer_accuracy": float(
                    (prediction == batch_targets).float().mean().detach().cpu()
                ),
                "batch_row_accuracy": float(
                    (torch.argmax(row_logits, dim=-1) == batch_rows)
                    .float()
                    .mean()
                    .detach()
                    .cpu()
                ),
                "batch_column_accuracy": float(
                    (torch.argmax(column_logits, dim=-1) == batch_columns)
                    .float()
                    .mean()
                    .detach()
                    .cpu()
                ),
                "learning_rate": lr,
                "gradient_norm": float(torch.as_tensor(gradient_norm).detach().cpu()),
            }
            logs.append(log)
            print(
                json.dumps({"replicate": replicate, "condition": condition, **log}, sort_keys=True),
                flush=True,
            )
        if current_step in trajectory_steps:
            checkpoint = save_checkpoint(
                model,
                replicate,
                condition,
                spec["split"],
                current_step,
                prereg,
                output_root,
            )
            trajectory_evaluation, _, trajectory_digests = evaluate_bundle(
                model, condition, spec, prereg, "trajectory", False
            )
            trajectory.append(
                {
                    "step": current_step,
                    "evaluation": trajectory_evaluation,
                    "dataset_digests": trajectory_digests,
                    "checkpoint": checkpoint,
                }
            )
            model.train()
    evaluation, prediction_rows, evaluation_digests = evaluate_bundle(
        model, condition, spec, prereg, "formal", True
    )
    answer_checks = answer_gate(evaluation, prereg["thresholds"])
    address_checks = address_gate(evaluation, condition, prereg["thresholds"])
    qualified = all(answer_checks.values()) and all(address_checks.values())
    run_dir = output_root / "runs" / spec["split"] / replicate / condition
    model_path = run_dir / "model.pt"
    checkpoint_payload = {
        "phase": PHASE,
        "replicate": replicate,
        "condition": condition,
        "protocol_digest": prereg["protocol_digest"],
        "config": asdict(config),
        "lexicon": spec["lexicon"],
        "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
    }
    torch.save(checkpoint_payload, model_path)
    predictions_path = run_dir / "predictions.jsonl"
    write_jsonl(predictions_path, prediction_rows)
    summary = {
        "phase": PHASE,
        "replicate": replicate,
        "condition": condition,
        "condition_definition": CONDITIONS[condition],
        "split": spec["split"],
        "scale": spec["scale"],
        "protocol_digest": prereg["protocol_digest"],
        "architecture": asdict(config),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "training_seed": spec["training_seed"],
        "lexicon_seed": spec["lexicon_seed"],
        "initial_state_digest": initial_digest,
        "training_dataset_digest": training_digest,
        "batch_schedule_digest": schedule_digest,
        "training_steps": int(training["max_steps"]),
        "nonfinite_steps": nonfinite_steps,
        "training_logs": logs,
        "trajectory": trajectory,
        "evaluation": evaluation,
        "evaluation_dataset_digests": evaluation_digests,
        "answer_gate_checks": answer_checks,
        "address_gate_checks": address_checks,
        "qualified": qualified,
        "model_path": str(model_path.relative_to(ROOT)).replace("\\", "/"),
        "model_sha256": file_sha256(model_path),
        "predictions_path": str(predictions_path.relative_to(ROOT)).replace("\\", "/"),
        "predictions_sha256": file_sha256(predictions_path),
        "peak_allocated_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "evidence_scope": "mandatory_mediation_calibration_behavior_and_formation_trajectory",
    }
    summary["summary_digest"] = canonical_digest(summary)
    write_json(run_dir / "summary.json", summary)
    del optimizer, model, train_inputs, train_targets, train_rows, train_columns, train_grids
    gc.collect()
    torch.cuda.empty_cache()
    return summary


def replicate_specifications() -> dict[str, dict[str, Any]]:
    base = p1147.p1146
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
        ("discovery_small_r1", "discovery", "small", 114801),
        ("discovery_small_r2", "discovery", "small", 114802),
        ("discovery_medium_r1", "discovery", "medium", 114803),
        ("discovery_medium_r2", "discovery", "medium", 114804),
        ("confirmation_small_r1", "confirmation", "small", 114805),
        ("confirmation_small_r2", "confirmation", "small", 114806),
        ("confirmation_medium_r1", "confirmation", "medium", 114807),
        ("confirmation_medium_r2", "confirmation", "medium", 114808),
    ]
    result: dict[str, dict[str, Any]] = {}
    for offset, (name, split, scale, training_seed) in enumerate(definitions):
        small = scale == "small"
        max_steps = 2500 if small else 3000
        result[name] = {
            "split": split,
            "scale": scale,
            "training_seed": training_seed,
            "lexicon_seed": 114811 + offset,
            "sampler_seed": 114821 + offset,
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
                "max_steps": max_steps,
                "trajectory_steps": [0, 100, 250, 500, 1000, 2000, max_steps],
            },
            "data_seeds": {
                "training": 114831 + offset,
                "seen_evaluation": 114841 + offset,
                "holdout_evaluation": 114851 + offset,
                "quartet": 114861 + offset,
                "trajectory_seen": 114871 + offset,
                "trajectory_holdout": 114881 + offset,
                "trajectory_quartet": 114891 + offset,
            },
        }
        result[name]["lexicon"] = base.make_lexicon(int(result[name]["lexicon_seed"]))
    return result


def protocol_body() -> dict[str, Any]:
    base = p1147.p1146
    return {
        "phase": PHASE,
        "title": "Mandatory row-column mediation calibration with claim-action gate separation",
        "claim_scope": (
            "Calibrates whether a forced differentiable address mediator can make the synthetic lookup task "
            "learnable. It plants an information-use path and therefore cannot establish the mechanism of a "
            "free Transformer or natural language model."
        ),
        "source_hashes": {
            "primary_script": file_sha256(Path(__file__).resolve()),
            "phase1147_dependency": file_sha256(Path(p1147.__file__).resolve()),
            "phase1146_dependency": file_sha256(Path(base.__file__).resolve()),
        },
        "conditions": CONDITIONS,
        "condition_order": CONDITION_ORDER,
        "soft_candidate_priority": SOFT_PRIORITY,
        "semantics": {
            "entity_count": base.ENTITY_COUNT,
            "field_count": base.FIELD_COUNT,
            "value_count": base.VALUE_COUNT,
            "record_count": base.RECORD_COUNT,
            "sequence_length": base.SEQUENCE_LENGTH,
            "oracle": "ground-truth row and column select the target grid value exactly",
            "soft_mediator": "outer product of predicted row and column probabilities weights one-hot cell values",
            "no_bypass": "soft conditions derive answer distribution only from mediator and grid values",
        },
        "data": {
            "pairs": base.semantic_pairs(),
            "training_count": 65536,
            "evaluation_count": 4096,
            "trajectory_evaluation_count": 1024,
            "field_order_control": "one random field order shared across records within each example",
            "surface_control": "one full token permutation per replicate, paired across conditions",
        },
        "replicates": replicate_specifications(),
        "thresholds": {
            "seen_accuracy": 0.995,
            "holdout_accuracy": 0.95,
            "quartet_accuracy": 0.95,
            "minimum_field_accuracy": 0.90,
            "minimum_entity_accuracy": 0.80,
            "address_accuracy": 0.95,
            "oracle_accuracy": 1.0,
            "soft_over_matched_free": 0.50,
            "uniform_address_drop": 0.30,
            "oracle_rescue_accuracy": 0.999,
        },
        "gate_policy": {
            "hard_stop": [
                "nonfinite training",
                "label-grid mismatch",
                "oracle accuracy below one",
                "hash or recomputation failure",
                "data or schedule pairing failure",
            ],
            "claim_stop": [
                "soft behavior failure stops learned-mediator success claim",
                "cross-condition disagreement stops generic mechanism claim",
                "confirmation failure stops independent formation claim",
            ],
            "branch_rules": {
                "oracle_fail": "retire_as_engineering_invalid",
                "oracle_pass_soft_fail": "branch_to_address_acquisition_without_success_mechanism_claim",
                "soft_pass": "independent_confirmation_then_mediation_ablation",
                "free_pass_soft_fail": "branch_to_alternative_free_mechanism",
            },
            "evidence_vector": [
                "integrity",
                "behavior",
                "address",
                "specificity",
                "prediction",
                "replication",
                "causal_use",
                "formation",
            ],
        },
        "selection": {
            "replicates_per_scale_per_split": 2,
            "confirmation_requires_one_soft_condition_qualified_in_all_discovery_replicates": True,
            "candidate_priority": SOFT_PRIORITY,
            "causal_validation_requires_same_condition_in_confirmation": True,
        },
        "forbidden": [
            "No condition-specific initial weights, data, batch schedules, steps, or token budgets",
            "No early stopping or post-freeze threshold and loss changes",
            "No confirmation unless a soft condition passes all four discovery replicates",
            "No naming the planted mediator as a discovered free-Transformer mechanism",
            "No topology, attractor, redundancy, or manifold claim without a calibrated fatal prediction",
            "No hidden-state success-mechanism scan after behavior failure; frozen trajectory behavior remains allowed",
        ],
    }


def create_protocol() -> dict[str, Any]:
    base = p1147.p1146
    body = protocol_body()
    prereg = dict(body)
    prereg["protocol_digest"] = canonical_digest(body)
    if PREREG_PATH.exists():
        existing = read_json(PREREG_PATH)
        if existing != prereg:
            raise RuntimeError("Existing Phase1148 protocol differs from current script")
    else:
        write_json(PREREG_PATH, prereg)
    replicates = prereg["replicates"]
    audit_spec = replicates["discovery_small_r1"]
    audit_dataset = make_dataset(
        int(prereg["data"]["training_count"]),
        prereg["data"]["pairs"]["train"],
        int(audit_spec["data_seeds"]["training"]),
        audit_spec["lexicon"],
    )
    oracle_prediction = audit_dataset["grid_values"][
        np.arange(len(audit_dataset["targets"])),
        audit_dataset["row_targets"],
        audit_dataset["column_targets"],
    ]
    checks = {
        "four_conditions": set(prereg["conditions"]) == set(CONDITIONS),
        "eight_replicates": len(replicates) == 8,
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
        "oracle_exact": bool(np.all(oracle_prediction == audit_dataset["targets"])),
        "all_rows_present": set(audit_dataset["row_targets"].tolist())
        == set(range(base.RECORD_COUNT)),
        "all_columns_present": set(audit_dataset["column_targets"].tolist())
        == set(range(base.FIELD_COUNT)),
        "source_hash_matches": file_sha256(Path(__file__).resolve())
        == prereg["source_hashes"]["primary_script"],
        "dependency_hashes_match": file_sha256(Path(p1147.__file__).resolve())
        == prereg["source_hashes"]["phase1147_dependency"]
        and file_sha256(Path(base.__file__).resolve())
        == prereg["source_hashes"]["phase1146_dependency"],
    }
    audit = {
        "phase": PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }
    audit["audit_digest"] = canonical_digest(audit)
    write_json(OUT_ROOT / "protocol" / "audit.json", audit)
    if not audit["all_checks_passed"]:
        raise RuntimeError("Phase1148 protocol audit failed")
    return prereg


def split_replicates(prereg: dict[str, Any], split: str) -> list[str]:
    return [name for name, spec in prereg["replicates"].items() if spec["split"] == split]


def load_summary(replicate: str, condition: str, prereg: dict[str, Any]) -> dict[str, Any]:
    split = prereg["replicates"][replicate]["split"]
    return read_json(
        OUT_ROOT / "runs" / split / replicate / condition / "summary.json"
    )


def analyze_split(split: str, prereg: dict[str, Any]) -> dict[str, Any]:
    replicates = split_replicates(prereg, split)
    summaries = {
        replicate: {
            condition: load_summary(replicate, condition, prereg)
            for condition in CONDITIONS
        }
        for replicate in replicates
    }
    all_qualified = {
        condition: all(
            summaries[replicate][condition]["qualified"] for replicate in replicates
        )
        for condition in CONDITIONS
    }
    eligible_soft = [condition for condition in SOFT_PRIORITY if all_qualified[condition]]
    effects: dict[str, Any] = {}
    gain_passes: list[bool] = []
    for replicate in replicates:
        effects[replicate] = {}
        for evaluation_split in ("seen", "holdout", "quartet"):
            accuracy = {
                condition: summaries[replicate][condition]["evaluation"][evaluation_split][
                    "accuracy"
                ]
                for condition in CONDITIONS
            }
            force_without_aux = accuracy["soft_00"] - accuracy["free_00"]
            force_with_aux = accuracy["soft_EF"] - accuracy["free_EF"]
            auxiliary_free = accuracy["free_EF"] - accuracy["free_00"]
            auxiliary_soft = accuracy["soft_EF"] - accuracy["soft_00"]
            interaction = auxiliary_soft - auxiliary_free
            effects[replicate][evaluation_split] = {
                "accuracy": accuracy,
                "force_without_aux": force_without_aux,
                "force_with_aux": force_with_aux,
                "auxiliary_free": auxiliary_free,
                "auxiliary_soft": auxiliary_soft,
                "factorial_interaction": interaction,
                "soft_00_gain_pass": force_without_aux
                >= prereg["thresholds"]["soft_over_matched_free"],
                "soft_EF_gain_pass": force_with_aux
                >= prereg["thresholds"]["soft_over_matched_free"],
            }
            if evaluation_split in ("holdout", "quartet"):
                gain_passes.append(
                    effects[replicate][evaluation_split]["soft_00_gain_pass"]
                    or effects[replicate][evaluation_split]["soft_EF_gain_pass"]
                )
    discovery = None
    if split == "confirmation":
        discovery = read_json(OUT_ROOT / "analysis" / "discovery_selection.json")
        selected = discovery["selected_condition"]
    else:
        selected = eligible_soft[0] if eligible_soft else None
    selected_qualified = bool(selected) and all_qualified[str(selected)]
    result = {
        "phase": PHASE,
        "split": split,
        "protocol_digest": prereg["protocol_digest"],
        "replicates": replicates,
        "condition_all_qualified": all_qualified,
        "eligible_soft_conditions": eligible_soft,
        "selected_condition": selected,
        "effects": effects,
        "soft_gain_scope_pass": all(gain_passes),
        "confirmation_authorized": bool(eligible_soft) if split == "discovery" else None,
        "causal_validation_authorized": bool(selected_qualified)
        if split == "confirmation"
        else None,
        "summary_digests": {
            replicate: {
                condition: summaries[replicate][condition]["summary_digest"]
                for condition in CONDITIONS
            }
            for replicate in replicates
        },
        "claim_status": (
            "candidate_requires_confirmation"
            if split == "discovery" and eligible_soft
            else "independently_confirmed_behavior"
            if split == "confirmation" and selected_qualified
            else "learned_mediator_not_confirmed"
        ),
        "next_action": (
            "confirmation"
            if split == "discovery" and eligible_soft
            else "causal_mediation_validation"
            if split == "confirmation" and selected_qualified
            else "branch_without_success_mechanism_claim"
        ),
    }
    result["selection_digest"] = canonical_digest(result)
    write_json(OUT_ROOT / "analysis" / f"{split}_selection.json", result)
    return result


def load_model(summary: dict[str, Any]) -> MediationBindingModel:
    checkpoint = torch.load(ROOT / summary["model_path"], map_location="cpu", weights_only=True)
    config = p1147.p1146.ModelConfig(**checkpoint["config"])
    model = MediationBindingModel(config)
    model.load_state_dict(checkpoint["state_dict"])
    return model.cuda().eval()


def intervention_accuracy(
    model: MediationBindingModel,
    dataset: dict[str, np.ndarray],
    mode: str,
    batch_size: int,
) -> float:
    device = next(model.parameters()).device
    predictions: list[int] = []
    with torch.inference_mode():
        for start in range(0, len(dataset["inputs"]), batch_size):
            stop = start + batch_size
            ids = torch.from_numpy(dataset["inputs"][start:stop]).to(device)
            grids = torch.from_numpy(dataset["grid_values"][start:stop]).to(device)
            rows = torch.from_numpy(dataset["row_targets"][start:stop]).to(device)
            columns = torch.from_numpy(dataset["column_targets"][start:stop]).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, row_logits, column_logits = model(ids)
            row_mode, column_mode = {
                "predicted": ("predicted", "predicted"),
                "uniform_row": ("uniform", "predicted"),
                "uniform_column": ("predicted", "uniform"),
                "oracle_row": ("oracle", "predicted"),
                "oracle_column": ("predicted", "oracle"),
                "oracle_both": ("oracle", "oracle"),
            }[mode]
            distribution = mediated_distribution(
                row_logits,
                column_logits,
                grids,
                row_mode=row_mode,
                column_mode=column_mode,
                row_targets=rows,
                column_targets=columns,
            )
            predictions.extend(torch.argmax(distribution, dim=-1).cpu().tolist())
    return float(np.mean(np.asarray(predictions) == dataset["targets"]))


def run_causal_validation(prereg: dict[str, Any]) -> dict[str, Any]:
    confirmation = read_json(OUT_ROOT / "analysis" / "confirmation_selection.json")
    if not confirmation["causal_validation_authorized"]:
        raise RuntimeError("Causal validation is not authorized")
    condition = str(confirmation["selected_condition"])
    per_replicate: dict[str, Any] = {}
    checks: list[bool] = []
    for replicate in split_replicates(prereg, "confirmation"):
        spec = prereg["replicates"][replicate]
        summary = load_summary(replicate, condition, prereg)
        model = load_model(summary)
        dataset = make_dataset(
            int(prereg["data"]["evaluation_count"]),
            prereg["data"]["pairs"]["confirmation"],
            int(spec["data_seeds"]["holdout_evaluation"]),
            spec["lexicon"],
        )
        metrics = {
            mode: intervention_accuracy(
                model,
                dataset,
                mode,
                int(spec["training"]["evaluation_batch_size"]),
            )
            for mode in (
                "predicted",
                "uniform_row",
                "uniform_column",
                "oracle_row",
                "oracle_column",
                "oracle_both",
            )
        }
        metrics["uniform_row_drop"] = metrics["predicted"] - metrics["uniform_row"]
        metrics["uniform_column_drop"] = metrics["predicted"] - metrics["uniform_column"]
        gate = {
            "base_behavior": metrics["predicted"] >= prereg["thresholds"]["holdout_accuracy"],
            "row_necessity": metrics["uniform_row_drop"]
            >= prereg["thresholds"]["uniform_address_drop"],
            "column_necessity": metrics["uniform_column_drop"]
            >= prereg["thresholds"]["uniform_address_drop"],
            "oracle_rescue": metrics["oracle_both"]
            >= prereg["thresholds"]["oracle_rescue_accuracy"],
        }
        checks.append(all(gate.values()))
        per_replicate[replicate] = {
            "metrics": metrics,
            "gate": gate,
            "dataset_digest": dataset_digest(dataset),
        }
        del model
        torch.cuda.empty_cache()
    result = {
        "phase": PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "selected_condition": condition,
        "per_replicate": per_replicate,
        "all_replicates_passed": all(checks),
        "evidence_scope": "causal_validation_of_planted_mandatory_mediator_only",
    }
    result["causal_digest"] = canonical_digest(result)
    write_json(OUT_ROOT / "analysis" / "causal_mediation_validation.json", result)
    return result


def finalize(prereg: dict[str, Any]) -> dict[str, Any]:
    discovery = read_json(OUT_ROOT / "analysis" / "discovery_selection.json")
    confirmation_path = OUT_ROOT / "analysis" / "confirmation_selection.json"
    confirmation = read_json(confirmation_path) if confirmation_path.exists() else None
    causal_path = OUT_ROOT / "analysis" / "causal_mediation_validation.json"
    causal = read_json(causal_path) if causal_path.exists() else None
    if causal and causal["all_replicates_passed"]:
        outcome = "forced_mediator_calibrated"
    elif confirmation and confirmation["causal_validation_authorized"]:
        outcome = "confirmed_behavior_without_causal_mediation_gate"
    elif discovery["confirmation_authorized"]:
        outcome = "confirmation_not_confirmed"
    else:
        outcome = "learned_mediator_discovery_failed"
    result = {
        "phase": PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "outcome": outcome,
        "discovery_selection_digest": discovery["selection_digest"],
        "confirmation_selection_digest": confirmation["selection_digest"]
        if confirmation
        else None,
        "causal_digest": causal["causal_digest"] if causal else None,
        "selected_condition": discovery["selected_condition"],
        "confirmation_authorized": discovery["confirmation_authorized"],
        "causal_validation_authorized": confirmation["causal_validation_authorized"]
        if confirmation
        else False,
        "forced_mediator_calibrated": bool(causal and causal["all_replicates_passed"]),
        "claim_status": (
            "calibrated_planted_mechanism"
            if causal and causal["all_replicates_passed"]
            else "not_confirmed"
        ),
        "next_action": (
            "new_protocol_for_free_network_functional_equivalence"
            if causal and causal["all_replicates_passed"]
            else discovery["next_action"]
        ),
        "auto_continue": False,
        "claim_boundary": (
            "Even a positive result calibrates an imposed mediator. It does not discover the mechanism of "
            "the free Transformer, Qwen3, GLM4, DS7B, natural language, or the brain."
        ),
    }
    result["final_digest"] = canonical_digest(result)
    write_json(OUT_ROOT / "analysis" / "final.json", result)
    return result


def run_split(split: str, prereg: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for replicate in split_replicates(prereg, split):
        result[replicate] = {}
        for condition in CONDITION_ORDER:
            result[replicate][condition] = train_condition(replicate, condition, prereg)
    return result


def smoke() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("Phase1148 smoke requires CUDA")
    spec = replicate_specifications()["discovery_small_r1"]
    spec = json.loads(json.dumps(spec))
    spec["training"]["max_steps"] = 3
    spec["training"]["batch_size"] = 32
    spec["training"]["trajectory_steps"] = [0, 3]
    mini = {
        "data": {
            "pairs": p1147.p1146.semantic_pairs(),
            "training_count": 512,
            "evaluation_count": 128,
            "trajectory_evaluation_count": 64,
        },
        "replicates": {"smoke": spec},
    }
    initial: dict[str, str] = {}
    materials: dict[str, str] = {}
    schedules: dict[str, str] = {}
    losses: dict[str, float] = {}
    gradients: dict[str, bool] = {}
    oracle_checks: dict[str, bool] = {}
    for condition in CONDITION_ORDER:
        set_seed(int(spec["training_seed"]))
        model = MediationBindingModel(
            p1147.p1146.ModelConfig(**spec["architecture"])
        ).cuda()
        initial[condition] = state_digest(model.state_dict())
        dataset, schedule = make_training_material(spec, mini)
        materials[condition] = dataset_digest(dataset)
        schedules[condition] = array_digest(schedule)
        indices = schedule[0]
        ids = torch.from_numpy(dataset["inputs"][indices]).cuda()
        targets = torch.from_numpy(dataset["targets"][indices]).cuda()
        rows = torch.from_numpy(dataset["row_targets"][indices]).cuda()
        columns = torch.from_numpy(dataset["column_targets"][indices]).cuda()
        grids = torch.from_numpy(dataset["grid_values"][indices]).cuda()
        values = candidate_ids(spec["lexicon"], torch.device("cuda"))
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, row_logits, column_logits = model(ids)
        answer_loss, _, _ = answer_loss_and_prediction(
            condition,
            logits,
            row_logits,
            column_logits,
            grids,
            targets,
            values,
        )
        loss = answer_loss
        if CONDITIONS[condition]["address_auxiliary"]:
            loss = loss + F.cross_entropy(row_logits.float(), rows)
            loss = loss + F.cross_entropy(column_logits.float(), columns)
        loss.backward()
        losses[condition] = float(loss.detach().cpu())
        gradients[condition] = all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        )
        oracle_distribution = mediated_distribution(
            row_logits,
            column_logits,
            grids,
            row_mode="oracle",
            column_mode="oracle",
            row_targets=rows,
            column_targets=columns,
        )
        oracle_checks[condition] = bool(
            torch.all(torch.argmax(oracle_distribution, dim=-1) == targets)
        )
        del model
        torch.cuda.empty_cache()
    result = {
        "phase": PHASE,
        "paired_initial_states": len(set(initial.values())) == 1,
        "paired_training_material": len(set(materials.values())) == 1,
        "paired_batch_schedules": len(set(schedules.values())) == 1,
        "finite_gradients": gradients,
        "oracle_exact": oracle_checks,
        "losses": losses,
        "initial_state_digests": initial,
        "training_material_digests": materials,
        "batch_schedule_digests": schedules,
    }
    result["all_checks_passed"] = (
        result["paired_initial_states"]
        and result["paired_training_material"]
        and result["paired_batch_schedules"]
        and all(gradients.values())
        and all(oracle_checks.values())
        and all(np.isfinite(list(losses.values())))
    )
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
            "train-condition",
            "run-split",
            "analyze-split",
            "run-causal",
            "finalize",
        ],
        required=True,
    )
    parser.add_argument("--replicate")
    parser.add_argument("--condition", choices=list(CONDITIONS))
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
        if args.mode == "train-condition":
            if not args.replicate or not args.condition:
                raise ValueError("--replicate and --condition required")
            result = train_condition(args.replicate, args.condition, prereg)
        elif args.mode == "run-split":
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
