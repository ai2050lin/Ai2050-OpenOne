#!/usr/bin/env python3
"""Collect full-layer scalar event traces for Phase569 natural phenotypes."""

from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))
os.environ.setdefault("PROBE_TORCH_DTYPE", "bfloat16")

from hf_probe_env import get_layers, load_probe_model, release_loaded  # noqa: E402
from phase548_shared_attention_compute_protocol import render_chat  # noqa: E402
from phase569_role_position_utils import ROLE_GROUPS, role_positions  # noqa: E402


PHASE = "Phase569"
MODELS = ("qwen3", "glm4", "deepseek7b")
COMPONENTS = ("layer_input", "attention_output", "mlp_output", "layer_output")
RESIDUAL_COMPONENTS = {"layer_input", "layer_output"}
TRACE_CASE_CAP = 96
OUT_DIR = ROOT / "tests/gpt5/result/phase569_relation_competition"
CASES_PATH = OUT_DIR / "phase569_open_cases.jsonl.gz"
REGISTRY_PATH = OUT_DIR / "phase569_path_phenotype_registry.json"
BEHAVIOR_SUMMARY_PATH = OUT_DIR / "phase569_behavior_summary.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def finite(value: float) -> float:
    return float(value) if math.isfinite(value) else 0.0


def tensor_from_output(output: Any) -> torch.Tensor:
    value = output[0] if isinstance(output, tuple) else output
    if not torch.is_tensor(value):
        raise TypeError(f"Unexpected hook output: {type(value).__name__}")
    return value


def rows_path(model: str) -> Path:
    return OUT_DIR / f"phase569_{model}_coarse_trace_rows.jsonl"


def summary_path(model: str) -> Path:
    return OUT_DIR / f"phase569_{model}_coarse_trace_summary.json"


def contract_path(model: str) -> Path:
    return OUT_DIR / f"phase569_{model}_coarse_trace_contract.json"


def selected_registry(model: str) -> tuple[list[dict[str, Any]], set[str]]:
    registry = read_json(REGISTRY_PATH)
    if model not in registry["authorized_models"] or registry["sealed_split_read"]:
        raise RuntimeError(f"Phase569 coarse trace is not authorized for {model}")
    model_registry = next(item for item in registry["models"] if item["model"] == model)
    if not model_registry["authorized_for_coarse_internal_trace"]:
        raise RuntimeError(f"Phase569 model registry does not authorize {model}")
    entries = []
    selected = set()
    for entry in model_registry["entries"]:
        if not entry["qualified"] or entry["case_count"] < TRACE_CASE_CAP:
            raise RuntimeError(f"Phase569 trace entry is under-qualified: {model}/{entry}")
        case_ids = entry["semantic_case_ids"][:TRACE_CASE_CAP]
        entries.append({
            "phenotype": entry["phenotype"],
            "split": entry["split"],
            "semantic_case_ids": case_ids,
        })
        selected.update(case_ids)
    if len(entries) != 4 or len(selected) != TRACE_CASE_CAP * 4:
        raise RuntimeError(f"Phase569 trace registry overlap/drift for {model}")
    return entries, selected


def load_cases(model: str) -> list[dict[str, Any]]:
    entries, selected = selected_registry(model)
    labels = {
        case_id: (entry["phenotype"], entry["split"])
        for entry in entries
        for case_id in entry["semantic_case_ids"]
    }
    cases = []
    for row in iter_jsonl(CASES_PATH):
        case_id = row["semantic_case_id"]
        if case_id not in selected:
            continue
        phenotype, split = labels[case_id]
        if row["sealed"] or row["split"] != split:
            raise RuntimeError("Phase569 trace bank identity/seal drift")
        cases.append({**row, "trace_phenotype": phenotype})
    if len(cases) != TRACE_CASE_CAP * 4:
        raise RuntimeError(f"Phase569 trace case count drift for {model}: {len(cases)}")
    return sorted(
        cases,
        key=lambda row: (row["trace_phenotype"], row["split"], row["semantic_case_id"]),
    )


def final_norm_module(model: Any) -> Any:
    core = getattr(model, "model", None)
    norm = getattr(core, "norm", None)
    if norm is None:
        raise TypeError(f"Cannot locate final norm for {type(model).__name__}")
    return norm


def prepare_contract(model: str, cases: list[dict[str, Any]], restart: bool) -> None:
    output = rows_path(model)
    summary = summary_path(model)
    contract = contract_path(model)
    payload = {
        "schema_version": "phase569_coarse_trace_contract.v1",
        "phase_id": PHASE,
        "created_at": now(),
        "model": model,
        "registry_sha256": sha256_file(REGISTRY_PATH),
        "behavior_summary_sha256": sha256_file(BEHAVIOR_SUMMARY_PATH),
        "open_cases_sha256": sha256_file(CASES_PATH),
        "case_count": len(cases),
        "semantic_case_ids": [row["semantic_case_id"] for row in cases],
        "components": list(COMPONENTS),
        "role_groups": list(ROLE_GROUPS),
        "trace_case_cap_per_phenotype_split": TRACE_CASE_CAP,
        "full_vectors_persisted": False,
        "causal_intervention_executed": False,
        "sealed_split_read": False,
    }
    if restart:
        output.unlink(missing_ok=True)
        summary.unlink(missing_ok=True)
        contract.unlink(missing_ok=True)
    if contract.exists():
        existing = read_json(contract)
        for key in (
            "model", "registry_sha256", "behavior_summary_sha256", "open_cases_sha256",
            "case_count", "semantic_case_ids", "components", "role_groups",
            "trace_case_cap_per_phenotype_split", "full_vectors_persisted",
            "causal_intervention_executed", "sealed_split_read",
        ):
            if existing[key] != payload[key]:
                raise RuntimeError(f"Phase569 trace contract drift: {model}/{key}")
    else:
        write_json(contract, payload)


def run(model_key: str, batch_size: int, restart: bool) -> Path:
    if model_key not in MODELS:
        raise ValueError(model_key)
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    cases = load_cases(model_key)
    prepare_contract(model_key, cases, restart)
    if rows_path(model_key).exists() and summary_path(model_key).exists() and not restart:
        return summary_path(model_key)

    loaded = None
    handles: list[Any] = []
    current_indices: torch.Tensor | None = None
    current_direction: torch.Tensor | None = None
    current_groups: list[tuple[str, str]] = []
    selected_cache: dict[tuple[int, str], torch.Tensor] = {}
    sums: defaultdict[tuple[str, str, int, str, str, str], float] = defaultdict(float)
    counts: defaultdict[tuple[str, str, int, str, str], int] = defaultdict(int)
    positive_direct: defaultdict[tuple[str, str, int, str, str], int] = defaultdict(int)
    positive_decoded: defaultdict[tuple[str, str, int, str, str], int] = defaultdict(int)
    decoded_counts: defaultdict[tuple[str, str, int, str, str], int] = defaultdict(int)
    max_ledger_error = 0.0
    ledger_error_sum = 0.0
    ledger_error_count = 0
    started = time.monotonic()
    try:
        loaded = load_probe_model(model_key)
        loaded.tokenizer.padding_side = "left"
        layers = get_layers(loaded.model)
        run_dtype = str(next(loaded.model.parameters()).dtype)
        if run_dtype != "torch.bfloat16":
            raise RuntimeError(f"Phase569 trace requires BF16, got {run_dtype}")
        norm_module = final_norm_module(loaded.model)
        output_embeddings = loaded.model.get_output_embeddings()
        if output_embeddings is None or not hasattr(output_embeddings, "weight"):
            raise TypeError("Phase569 cannot locate output embedding weights")

        def selected_vectors(value: torch.Tensor) -> torch.Tensor:
            if current_indices is None:
                raise RuntimeError("Phase569 trace indices are unset")
            indices = current_indices.to(value.device)
            batch_indices = torch.arange(value.shape[0], device=value.device)[:, None]
            return value[batch_indices, indices, :]

        def record(layer_index: int, component: str, value: torch.Tensor) -> None:
            nonlocal max_ledger_error, ledger_error_sum, ledger_error_count
            if current_direction is None:
                raise RuntimeError("Phase569 target-other direction is unset")
            selected_native = selected_vectors(value)
            selected = selected_native.float()
            direction = current_direction.to(selected.device).float()[:, None, :]
            vector_norm = selected.norm(dim=-1)
            direction_norm = direction.norm(dim=-1).clamp_min(1e-12)
            direct = (selected * direction).sum(dim=-1)
            cosine = direct / (vector_norm.clamp_min(1e-12) * direction_norm)
            decoded = None
            if component in RESIDUAL_COMPONENTS:
                normalized = norm_module(
                    selected_native.reshape(-1, selected_native.shape[-1])
                ).reshape_as(selected_native).float()
                decoded = (normalized * direction).sum(dim=-1)
            metric_parts = [vector_norm, direct, cosine]
            if decoded is not None:
                metric_parts.append(decoded)
            metric_tensor = torch.stack(metric_parts, dim=-1).detach().cpu()
            for row_index, (phenotype, split) in enumerate(current_groups):
                for role_index, role in enumerate(ROLE_GROUPS):
                    key = (phenotype, split, layer_index, component, role)
                    values = metric_tensor[row_index, role_index]
                    counts[key] += 1
                    sums[(*key, "vector_norm")] += finite(float(values[0].item()))
                    sums[(*key, "direct_target_minus_other")] += finite(float(values[1].item()))
                    sums[(*key, "normalized_target_other_projection")] += finite(
                        float(values[2].item())
                    )
                    positive_direct[key] += int(float(values[1].item()) > 0.0)
                    if decoded is not None:
                        sums[(*key, "decoded_target_minus_other_margin")] += finite(
                            float(values[3].item())
                        )
                        positive_decoded[key] += int(float(values[3].item()) > 0.0)
                        decoded_counts[key] += 1
            selected_cache[(layer_index, component)] = selected.detach()
            if component == "layer_output":
                required = [
                    selected_cache[(layer_index, name)]
                    for name in ("layer_input", "attention_output", "mlp_output")
                ]
                residual = selected - required[0] - required[1] - required[2]
                relative = residual.norm(dim=-1) / selected.norm(dim=-1).clamp_min(1e-12)
                max_ledger_error = max(max_ledger_error, float(relative.max().item()))
                ledger_error_sum += float(relative.sum().item())
                ledger_error_count += int(relative.numel())
                for name in COMPONENTS:
                    selected_cache.pop((layer_index, name), None)

        def make_pre_hook(layer_index: int):
            def hook(_module: Any, inputs: tuple[Any, ...]) -> None:
                record(layer_index, "layer_input", inputs[0])
            return hook

        def make_hook(layer_index: int, component: str):
            def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
                record(layer_index, component, tensor_from_output(output))
            return hook

        for layer_index, layer in enumerate(layers):
            handles.append(layer.register_forward_pre_hook(make_pre_hook(layer_index)))
            handles.append(
                layer.self_attn.register_forward_hook(
                    make_hook(layer_index, "attention_output")
                )
            )
            handles.append(
                layer.mlp.register_forward_hook(make_hook(layer_index, "mlp_output"))
            )
            handles.append(
                layer.register_forward_hook(make_hook(layer_index, "layer_output"))
            )

        for batch_start in range(0, len(cases), batch_size):
            batch_rows = cases[batch_start:batch_start + batch_size]
            prompts = [render_chat(loaded.tokenizer, model_key, row["raw_prompt"]) for row in batch_rows]
            individual = [
                role_positions(loaded.tokenizer, prompt, row)
                for prompt, row in zip(prompts, batch_rows)
            ]
            encoded = loaded.tokenizer(prompts, return_tensors="pt", padding=True, truncation=False)
            sequence_length = int(encoded["input_ids"].shape[1])
            index_rows = []
            for row_index, (ids, groups) in enumerate(individual):
                mask_ids = encoded["input_ids"][row_index][
                    encoded["attention_mask"][row_index].bool()
                ].tolist()
                if [int(value) for value in mask_ids] != ids:
                    raise RuntimeError("Phase569 individual/batch tokenization drift")
                offset = sequence_length - len(ids)
                index_rows.append([
                    offset + groups[role][-1] for role in ROLE_GROUPS
                ])
            current_indices = torch.tensor(index_rows, dtype=torch.long)
            target_ids = torch.tensor([
                row["candidate_token_ids_by_model"][model_key][row["target"]][0]
                for row in batch_rows
            ], dtype=torch.long, device=output_embeddings.weight.device)
            other_ids = torch.tensor([
                row["candidate_token_ids_by_model"][model_key][row["other_relation_target"]][0]
                for row in batch_rows
            ], dtype=torch.long, device=output_embeddings.weight.device)
            current_direction = (
                output_embeddings.weight[target_ids] - output_embeddings.weight[other_ids]
            ).detach()
            current_groups = [
                (row["trace_phenotype"], row["split"]) for row in batch_rows
            ]
            encoded = {key: value.to(loaded.input_device) for key, value in encoded.items()}
            with torch.inference_mode():
                result = loaded.model(**encoded, use_cache=False)
            if selected_cache:
                raise RuntimeError(f"Phase569 trace component cache did not close: {selected_cache.keys()}")
            del result, encoded, prompts, individual, current_direction
            current_direction = None
            current_indices = None
            current_groups = []
            done = min(batch_start + batch_size, len(cases))
            if batch_start == 0 or done == len(cases) or (batch_start // batch_size) % 4 == 3:
                print(
                    f"[{time.strftime('%H:%M:%S')}] {model_key} Phase569 coarse trace "
                    f"{done}/{len(cases)}",
                    flush=True,
                )

        output_rows = []
        for layer_index in range(len(layers)):
            for component in COMPONENTS:
                for role in ROLE_GROUPS:
                    for phenotype in ("stable_correct", "stable_relation_confusion"):
                        for split in ("path_discovery", "path_confirmation"):
                            key = (phenotype, split, layer_index, component, role)
                            count = counts[key]
                            if count != TRACE_CASE_CAP:
                                raise RuntimeError(f"Phase569 trace aggregate count drift: {key}/{count}")
                            decoded_count = decoded_counts[key]
                            output_rows.append({
                                "schema_version": "phase569_coarse_trace_event.v1",
                                "phase_id": PHASE,
                                "created_at": now(),
                                "model": model_key,
                                "torch_dtype": run_dtype,
                                "phenotype": phenotype,
                                "split": split,
                                "layer": layer_index,
                                "layer_count": len(layers),
                                "relative_depth": layer_index / max(1, len(layers) - 1),
                                "component": component,
                                "semantic_role": role,
                                "case_count": count,
                                "mean_vector_norm": finite(
                                    sums[(*key, "vector_norm")] / count
                                ),
                                "mean_direct_target_minus_other": finite(
                                    sums[(*key, "direct_target_minus_other")] / count
                                ),
                                "direct_target_minus_other_positive_rate": (
                                    positive_direct[key] / count
                                ),
                                "mean_normalized_target_other_projection": finite(
                                    sums[(*key, "normalized_target_other_projection")] / count
                                ),
                                "decoded_margin_available": decoded_count == count,
                                "mean_decoded_target_minus_other_margin": (
                                    finite(
                                        sums[(*key, "decoded_target_minus_other_margin")]
                                        / decoded_count
                                    ) if decoded_count else None
                                ),
                                "decoded_target_minus_other_positive_rate": (
                                    positive_decoded[key] / decoded_count
                                    if decoded_count else None
                                ),
                                "full_vector_persisted": False,
                                "observer_only": True,
                                "causal": False,
                                "sealed": False,
                            })
        write_jsonl(rows_path(model_key), output_rows)
        summary = {
            "schema_version": "phase569_coarse_trace_summary.v1",
            "phase_id": PHASE,
            "created_at": now(),
            "status": "complete",
            "model": model_key,
            "torch_dtype": run_dtype,
            "case_count": len(cases),
            "case_count_per_phenotype_split": TRACE_CASE_CAP,
            "layer_count": len(layers),
            "components": list(COMPONENTS),
            "semantic_roles": list(ROLE_GROUPS),
            "event_row_count": len(output_rows),
            "max_component_ledger_relative_error": finite(max_ledger_error),
            "mean_component_ledger_relative_error": finite(
                ledger_error_sum / max(1, ledger_error_count)
            ),
            "runtime_seconds": time.monotonic() - started,
            "rows_path": str(rows_path(model_key).relative_to(ROOT)),
            "rows_sha256": sha256_file(rows_path(model_key)),
            "full_vectors_persisted": False,
            "causal_intervention_executed": False,
            "head_channel_parameter_neuron_scan_executed": False,
            "sealed_split_read": False,
        }
        write_json(summary_path(model_key), summary)
        print(summary_path(model_key), flush=True)
        return summary_path(model_key)
    finally:
        for handle in handles:
            handle.remove()
        selected_cache.clear()
        release_loaded(loaded)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()
    run(args.model, args.batch_size, args.restart)


if __name__ == "__main__":
    main()
