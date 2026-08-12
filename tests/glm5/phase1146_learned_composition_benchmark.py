from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


PHASE = 1146
ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "tests" / "glm5" / "result" / "phase1146_learned_composition_benchmark"
TEMP_ROOT = ROOT / "tests" / "glm5_temp" / "phase1146_smoke"
PREREG_PATH = OUT_ROOT / "protocol" / "preregistration.json"

PAD = 0
BOS = 1
SEP = 2
QUERY = 3
ANSWER = 4
SPECIAL_COUNT = 5
ENTITY_COUNT = 24
FIELD_COUNT = 4
VALUE_COUNT = 8
ENTITY_START = SPECIAL_COUNT
FIELD_START = ENTITY_START + ENTITY_COUNT
VALUE_START = FIELD_START + FIELD_COUNT
VOCAB_SIZE = VALUE_START + VALUE_COUNT
RECORD_COUNT = 3
SEQUENCE_LENGTH = 1 + RECORD_COUNT * (1 + 2 * FIELD_COUNT + 1) + 4


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


def semantic_pairs() -> dict[str, list[list[int]]]:
    train: list[list[int]] = []
    holdout: list[tuple[int, int]] = []
    for entity in range(ENTITY_COUNT):
        for field in range(FIELD_COUNT):
            if (entity + field) % 2:
                train.append([entity, field])
            else:
                holdout.append((entity, field))
    holdout.sort(key=lambda pair: hashlib.sha256(f"{pair[0]}:{pair[1]}".encode("ascii")).hexdigest())
    midpoint = len(holdout) // 2
    return {
        "train": train,
        "discovery": [list(pair) for pair in holdout[:midpoint]],
        "confirmation": [list(pair) for pair in holdout[midpoint:]],
    }


def make_lexicon(seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    mapping = list(range(VOCAB_SIZE))
    physical = np.arange(SPECIAL_COUNT, VOCAB_SIZE, dtype=np.int64)
    rng.shuffle(physical)
    for semantic_id, physical_id in zip(range(SPECIAL_COUNT, VOCAB_SIZE), physical.tolist()):
        mapping[semantic_id] = int(physical_id)
    return mapping


def encode_sequence(semantic: list[int], lexicon: list[int]) -> np.ndarray:
    return np.asarray([lexicon[token] for token in semantic], dtype=np.int64)


def build_sequence(
    entities: list[int],
    assignments: dict[int, list[int]],
    query_entity: int,
    query_field: int,
    record_order: list[int],
    field_orders: dict[int, list[int]],
    lexicon: list[int],
) -> tuple[np.ndarray, dict[tuple[int, int], int]]:
    if sorted(entities) != sorted(record_order) or len(entities) != RECORD_COUNT:
        raise ValueError("Malformed record order")
    semantic = [BOS]
    positions: dict[tuple[int, int], int] = {}
    for entity in record_order:
        semantic.append(ENTITY_START + entity)
        for field in field_orders[entity]:
            semantic.append(FIELD_START + field)
            semantic.append(VALUE_START + int(assignments[entity][field]))
            positions[(entity, field)] = len(semantic) - 1
        semantic.append(SEP)
    semantic.extend([QUERY, ENTITY_START + query_entity, FIELD_START + query_field, ANSWER])
    if len(semantic) != SEQUENCE_LENGTH:
        raise RuntimeError(f"Unexpected sequence length: {len(semantic)}")
    return encode_sequence(semantic, lexicon), positions


def make_dataset(
    count: int,
    pairs: list[list[int]],
    seed: int,
    lexicon: list[int],
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    inputs = np.empty((count, SEQUENCE_LENGTH), dtype=np.int64)
    targets = np.empty(count, dtype=np.int64)
    entities_meta = np.empty(count, dtype=np.int64)
    fields_meta = np.empty(count, dtype=np.int64)
    pair_order = [tuple(pair) for pair in pairs]
    for index in range(count):
        query_entity, query_field = pair_order[index % len(pair_order)]
        target_value = (index // len(pair_order)) % VALUE_COUNT
        distractors = rng.choice(
            [entity for entity in range(ENTITY_COUNT) if entity != query_entity],
            size=RECORD_COUNT - 1,
            replace=False,
        ).tolist()
        entities = [query_entity, *[int(value) for value in distractors]]
        assignments = {
            entity: rng.integers(0, VALUE_COUNT, size=FIELD_COUNT, dtype=np.int64).tolist()
            for entity in entities
        }
        assignments[query_entity][query_field] = int(target_value)
        record_order = list(entities)
        rng.shuffle(record_order)
        field_orders: dict[int, list[int]] = {}
        for entity in entities:
            order = list(range(FIELD_COUNT))
            rng.shuffle(order)
            field_orders[entity] = order
        sequence, _ = build_sequence(
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
    permutation = rng.permutation(count)
    return {
        "inputs": inputs[permutation],
        "targets": targets[permutation],
        "entities": entities_meta[permutation],
        "fields": fields_meta[permutation],
    }


def make_quartets(
    pairs: list[list[int]],
    seed: int,
    lexicon: list[int],
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    rows: list[np.ndarray] = []
    labels: list[int] = []
    entities_meta: list[int] = []
    fields_meta: list[int] = []
    metadata: list[dict[str, Any]] = []
    state_names = ["active_minus", "active_plus", "null_minus", "null_plus"]
    for item_index, pair in enumerate(pairs):
        query_entity, query_field = int(pair[0]), int(pair[1])
        destination = 1 + (item_index % 6)
        base = 0
        anchor = 7
        distractors = rng.choice(
            [entity for entity in range(ENTITY_COUNT) if entity != query_entity],
            size=2,
            replace=False,
        ).tolist()
        entity1, entity2 = int(distractors[0]), int(distractors[1])
        entities = [query_entity, entity1, entity2]
        common_assignments = {
            entity: rng.integers(0, VALUE_COUNT, size=FIELD_COUNT, dtype=np.int64).tolist()
            for entity in entities
        }
        record_order = list(entities)
        rng.shuffle(record_order)
        field_orders: dict[int, list[int]] = {}
        for entity in entities:
            order = list(range(FIELD_COUNT))
            rng.shuffle(order)
            field_orders[entity] = order
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
            sequence, positions = build_sequence(
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
            metadata.append(
                {
                    "item_index": item_index,
                    "item_id": f"e{query_entity:02d}.f{query_field}.d{destination}",
                    "state_index": state_index,
                    "state": state_names[state_index],
                    "query_entity": query_entity,
                    "query_field": query_field,
                    "base_value": base,
                    "destination_value": destination,
                    "anchor_value": anchor,
                    "target_value": target,
                    "answer_position": SEQUENCE_LENGTH - 1,
                    "queried_source_position": positions[(query_entity, query_field)],
                    "distractor1_source_position": positions[(entity1, query_field)],
                    "distractor2_source_position": positions[(entity2, query_field)],
                }
            )
    dataset = {
        "inputs": np.stack(rows),
        "targets": np.asarray(labels, dtype=np.int64),
        "entities": np.asarray(entities_meta, dtype=np.int64),
        "fields": np.asarray(fields_meta, dtype=np.int64),
    }
    return dataset, metadata


@dataclass(frozen=True)
class ModelConfig:
    layers: int
    width: int
    heads: int
    mlp_width: int
    max_length: int = SEQUENCE_LENGTH
    vocab_size: int = VOCAB_SIZE


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.width % config.heads:
            raise ValueError("Width must be divisible by heads")
        self.heads = config.heads
        self.head_dim = config.width // config.heads
        self.qkv = nn.Linear(config.width, 3 * config.width, bias=False)
        self.out = nn.Linear(config.width, config.width, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        batch, length, width = hidden.shape
        qkv = self.qkv(hidden).view(batch, length, 3, self.heads, self.head_dim)
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        attended = F.scaled_dot_product_attention(query, key, value, is_causal=True)
        attended = attended.transpose(1, 2).contiguous().view(batch, length, width)
        return self.out(attended)


class DecoderBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(config.width)
        self.attn = CausalSelfAttention(config)
        self.mlp_norm = nn.LayerNorm(config.width)
        self.mlp = nn.Sequential(
            nn.Linear(config.width, config.mlp_width),
            nn.GELU(),
            nn.Linear(config.mlp_width, config.width),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = hidden + self.attn(self.attn_norm(hidden))
        hidden = hidden + self.mlp(self.mlp_norm(hidden))
        return hidden


class TinyCausalTransformer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.width)
        self.position_embedding = nn.Embedding(config.max_length, config.width)
        self.blocks = nn.ModuleList([DecoderBlock(config) for _ in range(config.layers)])
        self.final_norm = nn.LayerNorm(config.width)
        self.lm_head = nn.Linear(config.width, config.vocab_size, bias=False)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        return self.token_embedding(input_ids) + self.position_embedding(positions)[None, :, :]

    def forward(
        self,
        input_ids: torch.Tensor,
        return_states: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor]] | torch.Tensor:
        hidden = self.embed(input_ids)
        states = [hidden]
        for block in self.blocks:
            hidden = block(hidden)
            states.append(hidden)
        logits = self.lm_head(self.final_norm(hidden))
        if return_states:
            return logits, states
        return logits

    def forward_from(self, hidden: torch.Tensor, layer_index: int) -> torch.Tensor:
        for block in self.blocks[layer_index:]:
            hidden = block(hidden)
        return self.lm_head(self.final_norm(hidden))


def model_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def candidate_ids(lexicon: list[int], device: torch.device) -> torch.Tensor:
    return torch.tensor([lexicon[VALUE_START + value] for value in range(VALUE_COUNT)], device=device)


def evaluate_model(
    model: TinyCausalTransformer,
    dataset: dict[str, np.ndarray],
    lexicon: list[int],
    batch_size: int,
    split_name: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    device = next(model.parameters()).device
    values = candidate_ids(lexicon, device)
    predictions: list[int] = []
    confidences: list[float] = []
    inputs = dataset["inputs"]
    targets = dataset["targets"]
    with torch.inference_mode():
        for start in range(0, len(inputs), batch_size):
            batch_ids = torch.from_numpy(inputs[start : start + batch_size]).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(batch_ids)
            answer_logits = logits[:, -1, :].float().index_select(-1, values)
            probability = torch.softmax(answer_logits, dim=-1)
            prediction = torch.argmax(answer_logits, dim=-1)
            predictions.extend(prediction.cpu().tolist())
            confidences.extend(torch.max(probability, dim=-1).values.cpu().tolist())
    predicted = np.asarray(predictions, dtype=np.int64)
    correct = predicted == targets
    field_metrics: dict[str, float] = {}
    for field in range(FIELD_COUNT):
        mask = dataset["fields"] == field
        field_metrics[str(field)] = float(np.mean(correct[mask]))
    entity_metrics: dict[str, float] = {}
    for entity in sorted(set(dataset["entities"].tolist())):
        mask = dataset["entities"] == entity
        entity_metrics[str(entity)] = float(np.mean(correct[mask]))
    metrics = {
        "split": split_name,
        "case_count": int(len(targets)),
        "accuracy": float(np.mean(correct)),
        "minimum_field_accuracy": float(min(field_metrics.values())),
        "minimum_entity_accuracy": float(min(entity_metrics.values())),
        "mean_confidence": float(np.mean(confidences)),
        "per_field_accuracy": field_metrics,
        "per_entity_accuracy": entity_metrics,
        "dataset_digest": array_digest(
            dataset["inputs"], dataset["targets"], dataset["entities"], dataset["fields"]
        ),
    }
    rows = [
        {
            "split": split_name,
            "index": index,
            "query_entity": int(dataset["entities"][index]),
            "query_field": int(dataset["fields"][index]),
            "target_value": int(targets[index]),
            "predicted_value": int(predicted[index]),
            "correct": bool(correct[index]),
            "confidence": float(confidences[index]),
        }
        for index in range(len(targets))
    ]
    return metrics, rows


def gate_metrics(evaluation: dict[str, Any], thresholds: dict[str, float]) -> dict[str, bool]:
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


def evaluate_all(
    model: TinyCausalTransformer,
    model_spec: dict[str, Any],
    prereg: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, str]]:
    lexicon = model_spec["lexicon"]
    pair_sets = prereg["data"]["pairs"]
    eval_count = int(prereg["data"]["evaluation_count"])
    batch_size = int(model_spec["training"]["evaluation_batch_size"])
    seen = make_dataset(
        eval_count,
        pair_sets["train"],
        int(model_spec["data_seeds"]["seen_evaluation"]),
        lexicon,
    )
    holdout = make_dataset(
        eval_count,
        pair_sets[model_spec["split"]],
        int(model_spec["data_seeds"]["holdout_evaluation"]),
        lexicon,
    )
    quartet, quartet_metadata = make_quartets(
        pair_sets[model_spec["split"]],
        int(model_spec["data_seeds"]["quartet"]),
        lexicon,
    )
    evaluation: dict[str, Any] = {}
    all_rows: list[dict[str, Any]] = []
    digests: dict[str, str] = {}
    for split_name, dataset in [("seen", seen), ("holdout", holdout), ("quartet", quartet)]:
        metrics, rows = evaluate_model(model, dataset, lexicon, batch_size, split_name)
        evaluation[split_name] = metrics
        all_rows.extend(rows)
        digests[split_name] = metrics["dataset_digest"]
    for row, metadata in zip((row for row in all_rows if row["split"] == "quartet"), quartet_metadata):
        row.update(metadata)
    return evaluation, all_rows, digests


def learning_rate(step: int, spec: dict[str, Any]) -> float:
    warmup = int(spec["warmup_steps"])
    maximum = int(spec["max_steps"])
    peak = float(spec["learning_rate"])
    floor = float(spec["minimum_learning_rate"])
    if step < warmup:
        return peak * float(step + 1) / float(warmup)
    progress = min(1.0, float(step - warmup) / float(max(1, maximum - warmup)))
    return floor + 0.5 * (peak - floor) * (1.0 + math.cos(math.pi * progress))


def verify_preregistration(prereg: dict[str, Any]) -> None:
    body = dict(prereg)
    expected = body.pop("protocol_digest")
    if canonical_digest(body) != expected:
        raise RuntimeError("Phase1146 protocol digest mismatch")
    if file_sha256(Path(__file__).resolve()) != prereg["source_hashes"]["primary_script"]:
        raise RuntimeError("Phase1146 primary script changed after preregistration")


def train_model(model_name: str, prereg: dict[str, Any], output_root: Path = OUT_ROOT) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("Phase1146 requires CUDA")
    model_spec = prereg["models"][model_name]
    if model_spec["split"] == "confirmation":
        selection_path = OUT_ROOT / "analysis" / "discovery_selection.json"
        if not selection_path.exists() or not read_json(selection_path)["confirmation_authorized"]:
            raise RuntimeError("Confirmation training is not authorized")
    set_seed(int(model_spec["training_seed"]))
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    config = ModelConfig(**model_spec["architecture"])
    model = TinyCausalTransformer(config).to(device)
    train_data = make_dataset(
        int(prereg["data"]["training_count"]),
        prereg["data"]["pairs"]["train"],
        int(model_spec["data_seeds"]["training"]),
        model_spec["lexicon"],
    )
    train_inputs = torch.from_numpy(train_data["inputs"])
    train_targets = torch.from_numpy(train_data["targets"])
    value_ids = candidate_ids(model_spec["lexicon"], device)
    training = model_spec["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        betas=(0.9, 0.95),
    )
    sampler = torch.Generator(device="cpu")
    sampler.manual_seed(int(model_spec["training_seed"]) + 17)
    logs: list[dict[str, Any]] = []
    consecutive_passes = 0
    nonfinite_steps = 0
    torch.cuda.reset_peak_memory_stats()
    model.train()
    last_evaluation: dict[str, Any] | None = None
    last_rows: list[dict[str, Any]] = []
    last_dataset_digests: dict[str, str] = {}
    for step in range(int(training["max_steps"])):
        lr = learning_rate(step, training)
        for group in optimizer.param_groups:
            group["lr"] = lr
        indices = torch.randint(
            0,
            len(train_inputs),
            (int(training["batch_size"]),),
            generator=sampler,
        )
        batch_ids = train_inputs[indices].to(device, non_blocking=True)
        batch_targets = train_targets[indices].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(batch_ids)
            answer_logits = logits[:, -1, :].index_select(-1, value_ids)
            loss = F.cross_entropy(answer_logits.float(), batch_targets)
        if not torch.isfinite(loss):
            nonfinite_steps += 1
            raise RuntimeError(f"Nonfinite loss for {model_name} at step {step + 1}")
        loss.backward()
        gradients_finite = all(
            parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        )
        if not gradients_finite:
            nonfinite_steps += 1
            raise RuntimeError(f"Nonfinite gradient for {model_name} at step {step + 1}")
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip_norm"]))
        optimizer.step()

        current_step = step + 1
        should_evaluate = current_step % int(training["evaluation_interval"]) == 0
        should_evaluate = should_evaluate or current_step == int(training["max_steps"])
        if should_evaluate:
            last_evaluation, last_rows, last_dataset_digests = evaluate_all(model, model_spec, prereg)
            checks = gate_metrics(last_evaluation, prereg["thresholds"])
            qualified = all(checks.values())
            consecutive_passes = consecutive_passes + 1 if qualified else 0
            log = {
                "step": current_step,
                "loss": float(loss.detach().cpu()),
                "learning_rate": lr,
                "gradient_norm": float(torch.as_tensor(gradient_norm).detach().cpu()),
                "seen_accuracy": last_evaluation["seen"]["accuracy"],
                "holdout_accuracy": last_evaluation["holdout"]["accuracy"],
                "quartet_accuracy": last_evaluation["quartet"]["accuracy"],
                "gate_checks": checks,
                "qualified": qualified,
                "consecutive_passes": consecutive_passes,
            }
            logs.append(log)
            print(json.dumps({"model": model_name, **log}, sort_keys=True), flush=True)
            if (
                current_step >= int(training["minimum_steps"])
                and consecutive_passes >= int(training["required_consecutive_passes"])
            ):
                break
            model.train()
    if last_evaluation is None:
        raise RuntimeError("Phase1146 training produced no evaluation")
    checks = gate_metrics(last_evaluation, prereg["thresholds"])
    qualified = all(checks.values()) and consecutive_passes >= int(training["required_consecutive_passes"])
    run_dir = output_root / "runs" / model_spec["split"] / model_name
    model_path = run_dir / "model.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "phase": PHASE,
        "model_name": model_name,
        "protocol_digest": prereg["protocol_digest"],
        "config": asdict(config),
        "lexicon": model_spec["lexicon"],
        "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
    }
    torch.save(checkpoint, model_path)
    predictions_path = run_dir / "predictions.jsonl"
    write_jsonl(predictions_path, last_rows)
    summary = {
        "phase": PHASE,
        "model_name": model_name,
        "split": model_spec["split"],
        "scale": model_spec["scale"],
        "protocol_digest": prereg["protocol_digest"],
        "architecture": asdict(config),
        "parameter_count": model_parameter_count(model),
        "training_seed": model_spec["training_seed"],
        "lexicon_seed": model_spec["lexicon_seed"],
        "lexicon": model_spec["lexicon"],
        "training_steps": logs[-1]["step"],
        "nonfinite_steps": nonfinite_steps,
        "training_logs": logs,
        "evaluation": last_evaluation,
        "dataset_digests": last_dataset_digests,
        "gate_checks": checks,
        "qualified": qualified,
        "model_path": str(model_path.relative_to(ROOT)).replace("\\", "/"),
        "model_sha256": file_sha256(model_path),
        "predictions_path": str(predictions_path.relative_to(ROOT)).replace("\\", "/"),
        "predictions_sha256": file_sha256(predictions_path),
        "peak_allocated_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "evidence_scope": "learned_behavioral_composition_only",
    }
    summary["summary_digest"] = canonical_digest(summary)
    write_json(run_dir / "summary.json", summary)
    del optimizer, model, train_inputs, train_targets
    gc.collect()
    torch.cuda.empty_cache()
    return summary


def protocol_body() -> dict[str, Any]:
    pairs = semantic_pairs()
    common_training = {
        "learning_rate": 0.003,
        "minimum_learning_rate": 0.0003,
        "warmup_steps": 100,
        "weight_decay": 0.01,
        "gradient_clip_norm": 1.0,
        "evaluation_interval": 250,
        "minimum_steps": 1000,
        "required_consecutive_passes": 2,
        "evaluation_batch_size": 512,
    }
    specifications = {
        "discovery_small": {
            "split": "discovery",
            "scale": "small",
            "training_seed": 114601,
            "lexicon_seed": 114611,
            "architecture": asdict(ModelConfig(layers=4, width=64, heads=4, mlp_width=256)),
            "training": {**common_training, "batch_size": 512, "max_steps": 2500},
            "data_seeds": {
                "training": 114621,
                "seen_evaluation": 114631,
                "holdout_evaluation": 114641,
                "quartet": 114651,
            },
        },
        "discovery_medium": {
            "split": "discovery",
            "scale": "medium",
            "training_seed": 114602,
            "lexicon_seed": 114612,
            "architecture": asdict(ModelConfig(layers=6, width=96, heads=4, mlp_width=384)),
            "training": {**common_training, "batch_size": 384, "max_steps": 3000},
            "data_seeds": {
                "training": 114622,
                "seen_evaluation": 114632,
                "holdout_evaluation": 114642,
                "quartet": 114652,
            },
        },
        "confirmation_small": {
            "split": "confirmation",
            "scale": "small",
            "training_seed": 114603,
            "lexicon_seed": 114613,
            "architecture": asdict(ModelConfig(layers=4, width=64, heads=4, mlp_width=256)),
            "training": {**common_training, "batch_size": 512, "max_steps": 2500},
            "data_seeds": {
                "training": 114623,
                "seen_evaluation": 114633,
                "holdout_evaluation": 114643,
                "quartet": 114653,
            },
        },
        "confirmation_medium": {
            "split": "confirmation",
            "scale": "medium",
            "training_seed": 114604,
            "lexicon_seed": 114614,
            "architecture": asdict(ModelConfig(layers=6, width=96, heads=4, mlp_width=384)),
            "training": {**common_training, "batch_size": 384, "max_steps": 3000},
            "data_seeds": {
                "training": 114624,
                "seen_evaluation": 114634,
                "holdout_evaluation": 114644,
                "quartet": 114654,
            },
        },
    }
    for specification in specifications.values():
        specification["lexicon"] = make_lexicon(int(specification["lexicon_seed"]))
    return {
        "phase": PHASE,
        "title": "Learned compositional query benchmark with functional ground truth",
        "claim_scope": (
            "Behavioral compositional generalization in randomly initialized tiny causal transformers. "
            "Ground truth is functional and program-level; no hidden coordinate or mechanism is planted."
        ),
        "source_hashes": {"primary_script": file_sha256(Path(__file__).resolve())},
        "semantics": {
            "entity_count": ENTITY_COUNT,
            "field_count": FIELD_COUNT,
            "value_count": VALUE_COUNT,
            "record_count": RECORD_COUNT,
            "sequence_length": SEQUENCE_LENGTH,
            "vocab_size": VOCAB_SIZE,
            "task": "retrieve the value bound to a queried entity-field pair from three shuffled records",
            "quartet": (
                "active swaps base/destination across queried and distractor records; null swaps them only "
                "between distractors while the queried record keeps an anchor value"
            ),
        },
        "data": {
            "pairs": pairs,
            "training_count": 65536,
            "evaluation_count": 4096,
            "surface_control": "independent full non-special-token permutation per model",
            "training_holdout": "all entity-field pairs with even parity are absent as queries during training",
        },
        "models": specifications,
        "thresholds": {
            "seen_accuracy": 0.995,
            "holdout_accuracy": 0.95,
            "quartet_accuracy": 0.95,
            "minimum_field_accuracy": 0.90,
            "minimum_entity_accuracy": 0.80,
        },
        "selection": {
            "discovery_models": ["discovery_small", "discovery_medium"],
            "confirmation_models": ["confirmation_small", "confirmation_medium"],
            "confirmation_requires_both_discovery_models": True,
            "mechanism_phase_requires_all_four_models": True,
        },
        "forbidden": [
            "No hidden-state scan before both discovery and confirmation scales pass behavior gates",
            "No threshold changes after protocol creation",
            "No replacing a failed formal seed",
            "No interpreting program-level ground truth as a planted hidden mechanism",
            "No physical-coordinate comparison across independently permuted lexicons",
        ],
    }


def create_protocol() -> dict[str, Any]:
    body = protocol_body()
    prereg = dict(body)
    prereg["protocol_digest"] = canonical_digest(body)
    if PREREG_PATH.exists():
        existing = read_json(PREREG_PATH)
        if existing != prereg:
            raise RuntimeError("Existing Phase1146 protocol differs from current script")
    else:
        write_json(PREREG_PATH, prereg)
    audit = {
        "phase": PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "checks": {
            "train_pair_count": len(prereg["data"]["pairs"]["train"]) == 48,
            "discovery_pair_count": len(prereg["data"]["pairs"]["discovery"]) == 24,
            "confirmation_pair_count": len(prereg["data"]["pairs"]["confirmation"]) == 24,
            "pair_disjointness": len(
                {
                    tuple(pair)
                    for split in prereg["data"]["pairs"].values()
                    for pair in split
                }
            )
            == ENTITY_COUNT * FIELD_COUNT,
            "four_independent_lexicons": len(
                {tuple(spec["lexicon"]) for spec in prereg["models"].values()}
            )
            == 4,
            "source_hash_matches": file_sha256(Path(__file__).resolve())
            == prereg["source_hashes"]["primary_script"],
        },
    }
    audit["all_checks_passed"] = all(audit["checks"].values())
    audit["audit_digest"] = canonical_digest(audit)
    write_json(OUT_ROOT / "protocol" / "audit.json", audit)
    if not audit["all_checks_passed"]:
        raise RuntimeError("Phase1146 protocol audit failed")
    return prereg


def select_split(split: str, prereg: dict[str, Any]) -> dict[str, Any]:
    names = prereg["selection"][f"{split}_models"]
    summaries = {
        name: read_json(OUT_ROOT / "runs" / split / name / "summary.json") for name in names
    }
    all_qualified = all(summary["qualified"] for summary in summaries.values())
    result = {
        "phase": PHASE,
        "split": split,
        "protocol_digest": prereg["protocol_digest"],
        "model_qualified": {name: summary["qualified"] for name, summary in summaries.items()},
        "summary_digests": {name: summary["summary_digest"] for name, summary in summaries.items()},
        "all_qualified": all_qualified,
        "confirmation_authorized": bool(all_qualified) if split == "discovery" else None,
        "mechanism_phase_authorized": bool(all_qualified) if split == "confirmation" else None,
    }
    result["selection_digest"] = canonical_digest(result)
    write_json(OUT_ROOT / "analysis" / f"{split}_selection.json", result)
    return result


def finalize(prereg: dict[str, Any]) -> dict[str, Any]:
    discovery = read_json(OUT_ROOT / "analysis" / "discovery_selection.json")
    confirmation_path = OUT_ROOT / "analysis" / "confirmation_selection.json"
    confirmation = read_json(confirmation_path) if confirmation_path.exists() else None
    mechanism_authorized = bool(
        discovery["all_qualified"] and confirmation is not None and confirmation["all_qualified"]
    )
    final = {
        "phase": PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "discovery_passed": discovery["all_qualified"],
        "confirmation_ran": confirmation is not None,
        "confirmation_passed": bool(confirmation and confirmation["all_qualified"]),
        "mechanism_phase_authorized": mechanism_authorized,
        "component_search_authorized": False,
        "natural_llm_claim_authorized": False,
        "outcome": "learned_behavior_benchmark_confirmed" if mechanism_authorized else "behavior_gate_failed",
        "claim_scope": prereg["claim_scope"],
        "auto_continue": mechanism_authorized,
    }
    final["final_digest"] = canonical_digest(final)
    write_json(OUT_ROOT / "analysis" / "final.json", final)
    return final


def smoke() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("Phase1146 smoke requires CUDA")
    if TEMP_ROOT.exists():
        for path in sorted(TEMP_ROOT.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    set_seed(114600)
    config = ModelConfig(layers=2, width=32, heads=4, mlp_width=96)
    model = TinyCausalTransformer(config).cuda()
    lexicon = make_lexicon(114610)
    pairs = semantic_pairs()
    dataset = make_dataset(2048, pairs["train"], 114620, lexicon)
    ids = torch.from_numpy(dataset["inputs"][:64]).cuda()
    labels = torch.from_numpy(dataset["targets"][:64]).cuda()
    values = candidate_ids(lexicon, torch.device("cuda"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    losses: list[float] = []
    for _ in range(20):
        optimizer.zero_grad(set_to_none=True)
        logits, states = model(ids, return_states=True)
        loss = F.cross_entropy(logits[:, -1, :].index_select(-1, values).float(), labels)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    quartet, metadata = make_quartets(pairs["discovery"], 114650, lexicon)
    result = {
        "phase": PHASE,
        "scope": "engineering_smoke_only",
        "formal_seed_overlap": False,
        "finite": bool(all(np.isfinite(losses))),
        "loss_decreased": losses[-1] < losses[0],
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "state_count": len(states),
        "expected_state_count": config.layers + 1,
        "quartet_case_count": len(quartet["targets"]),
        "quartet_metadata_count": len(metadata),
        "sequence_length": int(quartet["inputs"].shape[1]),
    }
    result["all_checks_passed"] = bool(
        result["finite"]
        and result["loss_decreased"]
        and result["state_count"] == result["expected_state_count"]
        and result["quartet_case_count"] == 96
        and result["quartet_metadata_count"] == 96
        and result["sequence_length"] == SEQUENCE_LENGTH
    )
    result["smoke_digest"] = canonical_digest(result)
    write_json(TEMP_ROOT / "smoke.json", result)
    del optimizer, model
    gc.collect()
    torch.cuda.empty_cache()
    if not result["all_checks_passed"]:
        raise RuntimeError("Phase1146 smoke failed")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=["smoke", "protocol", "train", "select", "finalize"],
    )
    parser.add_argument("--model", choices=[
        "discovery_small",
        "discovery_medium",
        "confirmation_small",
        "confirmation_medium",
    ])
    parser.add_argument("--split", choices=["discovery", "confirmation"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.action == "smoke":
        result = smoke()
    elif args.action == "protocol":
        result = create_protocol()
    else:
        prereg = read_json(PREREG_PATH)
        verify_preregistration(prereg)
        if args.action == "train":
            if args.model is None:
                raise SystemExit("--model is required for train")
            result = train_model(args.model, prereg)
        elif args.action == "select":
            if args.split is None:
                raise SystemExit("--split is required for select")
            result = select_split(args.split, prereg)
        elif args.action == "finalize":
            result = finalize(prereg)
        else:
            raise AssertionError(args.action)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
