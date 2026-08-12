from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer


PHASE = 1125
ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "tests" / "glm5" / "result" / "phase1125_pythia_controlled_bridge_calibration"
PREREG_PATH = OUT_ROOT / "protocol" / "preregistration.json"


def canonical_digest(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def verify_preregistration(prereg: dict[str, Any]) -> None:
    body = dict(prereg)
    expected = body.pop("protocol_digest")
    if canonical_digest(body) != expected:
        raise RuntimeError("Phase1125 protocol digest mismatch")
    paths = {
        "phase1121_cases": ROOT
        / "tests"
        / "glm5"
        / "result"
        / "phase1121_wordnet_adjective_double_orthogonal"
        / "protocol"
        / "cases.pythia.jsonl",
        "phase1121_final": ROOT
        / "tests"
        / "glm5"
        / "result"
        / "phase1121_wordnet_adjective_double_orthogonal"
        / "analysis"
        / "final_summary.json",
        "model_weights": ROOT / prereg["model"]["path"] / "model.safetensors",
        "derived_cases": OUT_ROOT / "protocol" / "cases.pythia.jsonl",
        "projection": ROOT / prereg["evaluation"]["projection_path"],
    }
    for name, path in paths.items():
        if file_sha256(path) != prereg["source_hashes"][name]:
            raise RuntimeError(f"Phase1125 source hash mismatch: {name}")


class ResidualAdapter(nn.Module):
    def __init__(self, hidden_size: int, rank: int, seed: int) -> None:
        super().__init__()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        self.down = nn.Linear(hidden_size, rank, bias=False, dtype=torch.float32)
        self.up = nn.Linear(rank, hidden_size, bias=False, dtype=torch.float32)
        with torch.no_grad():
            self.down.weight.copy_(torch.randn(self.down.weight.shape, generator=generator) * 0.02)
            self.up.weight.zero_()

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.up(torch.nn.functional.gelu(self.down(hidden.float())))


def group_interactions(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row["interaction_id"], []).append(row)
    for interaction_id, quartet in groups.items():
        quartet.sort(key=lambda row: (int(row["context_sense"]), int(row["definition_sense"])))
        if [(int(row["context_sense"]), int(row["definition_sense"])) for row in quartet] != [
            (0, 0), (0, 1), (1, 0), (1, 1)
        ]:
            raise RuntimeError(f"Malformed quartet: {interaction_id}")
    return groups


def make_batch(rows: list[dict[str, Any]], pad_token_id: int) -> dict[str, torch.Tensor]:
    lengths = [len(row["input_ids"]) for row in rows]
    maximum = max(lengths)
    input_ids = torch.full((len(rows), maximum), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(rows), maximum), dtype=torch.long)
    for index, row in enumerate(rows):
        values = torch.tensor(row["input_ids"], dtype=torch.long)
        input_ids[index, : len(values)] = values
        attention_mask[index, : len(values)] = 1
    return {
        "input_ids": input_ids.cuda(non_blocking=True),
        "attention_mask": attention_mask.cuda(non_blocking=True),
        "query_positions": torch.tensor([length - 1 for length in lengths], device="cuda", dtype=torch.long),
        "context_positions": torch.tensor(
            [int(row["role_indices"]["context_end"]) for row in rows], device="cuda", dtype=torch.long
        ),
        "definition_positions": torch.tensor(
            [int(row["role_indices"]["definition_end"]) for row in rows], device="cuda", dtype=torch.long
        ),
        "targets": torch.tensor([1 if row["truth"] else 0 for row in rows], device="cuda", dtype=torch.long),
    }


def gather_roles(hidden: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    batch_indices = torch.arange(hidden.shape[0], device=hidden.device)
    return hidden[batch_indices, positions, :]


def fields(context_states: torch.Tensor, definition_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    c = 0.5 * ((context_states[0] + context_states[1]) - (context_states[2] + context_states[3]))
    d = 0.5 * ((definition_states[0] + definition_states[2]) - (definition_states[1] + definition_states[3]))
    return c, d


def cosine(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cosine_similarity(left.float().unsqueeze(0), right.float().unsqueeze(0), dim=-1)[0]


def centered_cosine(left: np.ndarray, right: np.ndarray) -> float | None:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left = left - np.mean(left)
    right = right - np.mean(right)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-12 or not np.isfinite(denominator):
        return None
    return float(np.dot(left, right) / denominator)


def gram_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    cells: dict[str, Any] = {}
    for template in sorted({int(record["template"]) for record in records}):
        for surface in sorted({record["surface"] for record in records}):
            selected = [
                record for record in records if int(record["template"]) == template and record["surface"] == surface
            ]
            concepts = sorted({record["concept_id"] for record in selected})
            lookup = {record["concept_id"]: record for record in selected}
            if len(selected) != len(concepts):
                raise RuntimeError("Duplicate concept within evaluation cell")
            c = np.stack([lookup[concept]["c_projected"] for concept in concepts]).astype(np.float64)
            d = np.stack([lookup[concept]["d_projected"] for concept in concepts]).astype(np.float64)
            c /= np.maximum(np.linalg.norm(c, axis=1, keepdims=True), 1e-12)
            d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
            gram_c = c @ c.T
            gram_d = d @ d.T
            triangle = np.triu_indices(len(concepts), k=1)
            same = centered_cosine(gram_c[triangle], gram_d[triangle])
            concept_index = {concept: index for index, concept in enumerate(concepts)}
            permutation = np.asarray(
                [concept_index[lookup[concept]["control_concept_id"]] for concept in concepts], dtype=np.int64
            )
            null_gram = gram_d[np.ix_(permutation, permutation)]
            null = centered_cosine(gram_c[triangle], null_gram[triangle])
            cells[f"template{template}.{surface}"] = {
                "concept_count": len(concepts),
                "same_gram_cosine": same,
                "fixed_deranged_cosine": null,
                "fixed_derangement_advantage": same - null if same is not None and null is not None else None,
            }
    same_values = [cell["same_gram_cosine"] for cell in cells.values() if cell["same_gram_cosine"] is not None]
    advantage_values = [
        cell["fixed_derangement_advantage"]
        for cell in cells.values()
        if cell["fixed_derangement_advantage"] is not None
    ]
    return {
        "cells": cells,
        "median_same_gram_cosine": float(np.median(same_values)),
        "minimum_same_gram_cosine": float(np.min(same_values)),
        "median_fixed_derangement_advantage": float(np.median(advantage_values)),
    }


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    full = np.asarray([record["cd_cosine_full"] for record in records], dtype=np.float64)
    projected = np.asarray([record["cd_cosine_projected"] for record in records], dtype=np.float64)
    margins = np.asarray([record["behavior_interaction"] for record in records], dtype=np.float64)
    correct = sum(record["correct_count"] for record in records)
    cases = sum(record["case_count"] for record in records)
    return {
        "interaction_count": len(records),
        "case_count": cases,
        "candidate_accuracy": float(correct / cases),
        "median_behavior_interaction": float(np.median(margins)),
        "behavior_interaction_positive_rate": float(np.mean(margins > 0.0)),
        "median_cd_cosine_full": float(np.median(full)),
        "median_cd_cosine_projected": float(np.median(projected)),
        "projected_cd_positive_rate": float(np.mean(projected > 0.0)),
        "median_full_projection_gap": float(abs(np.median(full) - np.median(projected))),
        "gram": gram_metrics(records),
    }


def evaluate(
    model: nn.Module,
    groups: dict[str, list[dict[str, Any]]],
    projection: torch.Tensor,
    pad_token_id: int,
    true_id: int,
    false_id: int,
    final_capture: dict[str, torch.Tensor],
) -> dict[str, Any]:
    model.eval()
    records_by_partition: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "calibration": [],
        "transfer": [],
    }
    with torch.inference_mode():
        for interaction_id in sorted(groups):
            rows = groups[interaction_id]
            batch = make_batch(rows, pad_token_id)
            output = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
            )
            hidden = final_capture["hidden"]
            context_states = gather_roles(hidden, batch["context_positions"])
            definition_states = gather_roles(hidden, batch["definition_positions"])
            c_full, d_full = fields(context_states, definition_states)
            c_projected = c_full.float() @ projection
            d_projected = d_full.float() @ projection

            indices = torch.arange(len(rows), device="cuda")
            query_logits = output.logits[indices, batch["query_positions"], :].float()
            candidates = query_logits[:, [false_id, true_id]]
            predictions = torch.argmax(candidates, dim=-1)
            z = candidates[:, 1] - candidates[:, 0]
            interaction = 0.5 * ((z[0] + z[3]) - (z[1] + z[2]))
            record = {
                "interaction_id": interaction_id,
                "partition": rows[0]["phase1125_partition"],
                "concept_id": rows[0]["concept_id"],
                "control_concept_id": rows[0]["deranged_control_concept_id"],
                "split": rows[0]["split"],
                "template": int(rows[0]["template"]),
                "surface": rows[0]["surface"],
                "case_count": 4,
                "correct_count": int((predictions == batch["targets"]).sum().cpu()),
                "behavior_interaction": float(interaction.cpu()),
                "cd_cosine_full": float(cosine(c_full, d_full).cpu()),
                "cd_cosine_projected": float(cosine(c_projected, d_projected).cpu()),
                "c_projected": c_projected.cpu().numpy(),
                "d_projected": d_projected.cpu().numpy(),
            }
            records_by_partition[record["partition"]].append(record)
            del output, hidden, context_states, definition_states, c_full, d_full, c_projected, d_projected

    summary = {partition: summarize_records(records) for partition, records in records_by_partition.items()}
    return summary


def train_arm(
    arm_name: str,
    arm_spec: dict[str, Any],
    prereg: dict[str, Any],
    model: nn.Module,
    groups: dict[str, list[dict[str, Any]]],
    projection: torch.Tensor,
    pad_token_id: int,
    true_id: int,
    false_id: int,
    final_capture: dict[str, torch.Tensor],
) -> dict[str, Any]:
    seed = int(arm_spec["seed"])
    torch.manual_seed(seed)
    np_generator = np.random.default_rng(seed)
    adapter = ResidualAdapter(
        hidden_size=int(prereg["model"]["hidden_size"]),
        rank=int(prereg["adapter"]["rank"]),
        seed=seed,
    ).cuda().train()
    latest: dict[str, torch.Tensor] = {}

    def adapter_hook(_module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: object) -> object:
        if isinstance(output, tuple):
            hidden = output[0]
            delta = adapter(hidden)
            latest["delta"] = delta
            return (hidden + delta.to(hidden.dtype), *output[1:])
        delta = adapter(output)
        latest["delta"] = delta
        return output + delta.to(output.dtype)

    layer = model.gpt_neox.layers[int(prereg["adapter"]["layer_index"])]
    adapter_handle = layer.register_forward_hook(adapter_hook)
    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=float(prereg["training"]["learning_rate"]),
        weight_decay=float(prereg["training"]["weight_decay"]),
    )
    train_ids = sorted(
        interaction_id
        for interaction_id, rows in groups.items()
        if rows[0]["phase1125_partition"] == "train"
    )
    epoch_logs: list[dict[str, Any]] = []
    nonfinite_steps = 0
    model.eval()
    adapter.train()
    for epoch in range(int(prereg["training"]["epochs"])):
        ordered = list(train_ids)
        np_generator.shuffle(ordered)
        totals = {"loss": 0.0, "behavior": 0.0, "bridge": 0.0, "delta": 0.0, "accuracy": 0.0}
        for interaction_id in ordered:
            rows = groups[interaction_id]
            batch = make_batch(rows, pad_token_id)
            optimizer.zero_grad(set_to_none=True)
            output = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
            )
            hidden = final_capture["hidden"]
            context_states = gather_roles(hidden, batch["context_positions"])
            definition_states = gather_roles(hidden, batch["definition_positions"])
            c_full, d_full = fields(context_states, definition_states)
            bridge_loss = 1.0 - cosine(c_full, d_full)

            indices = torch.arange(len(rows), device="cuda")
            query_logits = output.logits[indices, batch["query_positions"], :].float()
            candidates = query_logits[:, [false_id, true_id]]
            behavior_loss = torch.nn.functional.cross_entropy(candidates, batch["targets"])
            delta_loss = latest["delta"].float().pow(2).mean()
            loss = (
                behavior_loss
                + float(arm_spec["bridge_loss_weight"]) * bridge_loss
                + float(prereg["training"]["delta_l2_weight"]) * delta_loss
            )
            if not torch.isfinite(loss):
                nonfinite_steps += 1
                raise RuntimeError(f"Nonfinite loss in {arm_name}: {interaction_id}")
            loss.backward()
            gradients_finite = all(
                parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in adapter.parameters()
            )
            if not gradients_finite:
                nonfinite_steps += 1
                raise RuntimeError(f"Nonfinite gradient in {arm_name}: {interaction_id}")
            torch.nn.utils.clip_grad_norm_(
                adapter.parameters(), float(prereg["training"]["gradient_clip_norm"])
            )
            optimizer.step()
            totals["loss"] += float(loss.detach().cpu())
            totals["behavior"] += float(behavior_loss.detach().cpu())
            totals["bridge"] += float(bridge_loss.detach().cpu())
            totals["delta"] += float(delta_loss.detach().cpu())
            totals["accuracy"] += float((torch.argmax(candidates, dim=-1) == batch["targets"]).float().mean().cpu())
            del output, hidden, context_states, definition_states, c_full, d_full, loss

        denominator = len(ordered)
        epoch_log = {
            "epoch": epoch + 1,
            "mean_loss": totals["loss"] / denominator,
            "mean_behavior_loss": totals["behavior"] / denominator,
            "mean_bridge_loss": totals["bridge"] / denominator,
            "mean_delta_loss": totals["delta"] / denominator,
            "mean_case_accuracy": totals["accuracy"] / denominator,
        }
        epoch_logs.append(epoch_log)
        print(json.dumps({"arm": arm_name, **epoch_log}, sort_keys=True), flush=True)

    adapter_path = OUT_ROOT / "training" / arm_name / "adapter.pt"
    adapter_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({key: value.detach().cpu() for key, value in adapter.state_dict().items()}, adapter_path)
    evaluation = evaluate(model, groups, projection, pad_token_id, true_id, false_id, final_capture)
    base_gradients_absent = all(parameter.grad is None for parameter in model.parameters())
    result = {
        "schema_version": "phase1125_pythia_controlled_bridge_arm.v1",
        "phase": PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "arm": arm_name,
        "arm_spec": arm_spec,
        "nonfinite_training_steps": nonfinite_steps,
        "base_gradients_absent": base_gradients_absent,
        "epoch_logs": epoch_logs,
        "adapter_path": str(adapter_path.relative_to(ROOT)).replace("\\", "/"),
        "adapter_sha256": file_sha256(adapter_path),
        "evaluation": evaluation,
    }
    result["result_digest"] = canonical_digest(result)
    write_json(OUT_ROOT / "training" / arm_name / "summary.json", result)
    adapter_handle.remove()
    del optimizer, adapter
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    prereg = read_json(PREREG_PATH)
    verify_preregistration(prereg)
    rows = read_jsonl(OUT_ROOT / "protocol" / "cases.pythia.jsonl")
    groups = group_interactions(rows)
    projection_np = np.load(ROOT / prereg["evaluation"]["projection_path"], allow_pickle=False)
    projection = torch.from_numpy(projection_np).cuda().float()

    model_path = ROOT / prereg["model"]["path"]
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
    pad_token_id = int(prereg["model"]["padding_token_id"])
    if tokenizer.convert_tokens_to_ids(prereg["model"]["padding_token"]) != pad_token_id:
        raise RuntimeError("Frozen Pythia padding token identity mismatch")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16,
        local_files_only=True,
    ).cuda().eval()
    model.config.use_cache = False
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    true_ids = {int(row["candidate_first_token_ids"]["true"][0]) for row in rows}
    false_ids = {int(row["candidate_first_token_ids"]["false"][0]) for row in rows}
    if len(true_ids) != 1 or len(false_ids) != 1:
        raise RuntimeError("Candidate token IDs are not frozen singletons")
    true_id, false_id = next(iter(true_ids)), next(iter(false_ids))

    final_capture: dict[str, torch.Tensor] = {}

    def final_hook(_module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        final_capture["hidden"] = output

    final_handle = model.gpt_neox.final_layer_norm.register_forward_hook(final_hook)
    torch.cuda.reset_peak_memory_stats()
    base_evaluation = evaluate(model, groups, projection, int(pad_token_id), true_id, false_id, final_capture)
    write_json(OUT_ROOT / "evaluation" / "base.json", {
        "schema_version": "phase1125_pythia_controlled_bridge_base.v1",
        "phase": PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "evaluation": base_evaluation,
    })
    print(json.dumps({"condition": "base", "evaluation": base_evaluation}, sort_keys=True), flush=True)

    arm_results: dict[str, Any] = {}
    for arm_name, arm_spec in prereg["training"]["arms"].items():
        arm_results[arm_name] = train_arm(
            arm_name,
            arm_spec,
            prereg,
            model,
            groups,
            projection,
            int(pad_token_id),
            true_id,
            false_id,
            final_capture,
        )
    final_handle.remove()

    run_summary = {
        "schema_version": "phase1125_pythia_controlled_bridge_run.v1",
        "phase": PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "base_evaluation": base_evaluation,
        "arms": {name: result["result_digest"] for name, result in arm_results.items()},
        "peak_allocated_memory_gb": float(torch.cuda.max_memory_allocated() / (1024**3)),
        "all_base_parameters_frozen": all(not parameter.requires_grad for parameter in model.parameters()),
    }
    run_summary["run_digest"] = canonical_digest(run_summary)
    write_json(OUT_ROOT / "training" / "run_summary.json", run_summary)
    print(json.dumps({
        "phase": PHASE,
        "run_digest": run_summary["run_digest"],
        "peak_allocated_memory_gb": run_summary["peak_allocated_memory_gb"],
        "arms": run_summary["arms"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
