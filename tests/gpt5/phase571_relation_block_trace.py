#!/usr/bin/env python3
"""Collect compact per-case Phase571 attention/MLP signed-write trajectories."""

from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import json
import os
import sys
import time
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
import phase571_relation_block_protocol as protocol  # noqa: E402


MODELS = protocol.MODELS
OUT_DIR = protocol.OUT_DIR
TRACE_POOLS = ("block_discovery", "block_confirmation")
COMPONENTS = ("attention_output", "mlp_output")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
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
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_from_output(output: Any) -> torch.Tensor:
    value = output[0] if isinstance(output, tuple) else output
    if not torch.is_tensor(value):
        raise TypeError(f"Unexpected hook output: {type(value).__name__}")
    return value


def rows_path(model: str) -> Path:
    return OUT_DIR / f"phase571_{model}_signed_write_rows.jsonl.gz"


def summary_path(model: str) -> Path:
    return OUT_DIR / f"phase571_{model}_signed_write_summary.json"


def contract_path(model: str) -> Path:
    return OUT_DIR / f"phase571_{model}_signed_write_contract.json"


def behavior_summary_path(model: str) -> Path:
    return OUT_DIR / f"phase571_{model}_matched_behavior_summary.json"


def load_cases(model: str) -> list[dict[str, Any]]:
    summary = read_json(behavior_summary_path(model))
    if not summary["qualified_for_signed_write_trace"]:
        raise RuntimeError(f"Phase571 behavior did not authorize trace for {model}")
    labels = {
        case_id: phenotype
        for pool in TRACE_POOLS
        for phenotype in protocol.PHENOTYPES
        for case_id in summary["selected_case_ids_by_pool_phenotype"][pool][phenotype]
    }
    selected = set(labels)
    expected = (
        len(TRACE_POOLS) * len(protocol.PHENOTYPES)
        * protocol.TRACE_SELECTION_PER_PHENOTYPE
    )
    if len(selected) != expected:
        raise RuntimeError(f"Phase571 selected trace denominator drift: {model}/{len(selected)}")
    cases = [
        {**row, "trace_phenotype": labels[row["case_id"]]}
        for row in iter_jsonl(protocol.OPEN_CASES_PATH)
        if row["model"] == model and row["case_id"] in selected
    ]
    if len(cases) != expected or any(row["sealed"] for row in cases):
        raise RuntimeError(f"Phase571 trace case bank drift: {model}/{len(cases)}")
    return sorted(cases, key=lambda row: (row["pool"], row["intended_phenotype"], row["case_id"]))


def prepare(model: str, cases: list[dict[str, Any]], restart: bool) -> None:
    payload = {
        "schema_version": "phase571_signed_write_contract.v1",
        "phase_id": protocol.PHASE,
        "created_at": now(),
        "model": model,
        "open_cases_sha256": sha256_file(protocol.OPEN_CASES_PATH),
        "behavior_summary_sha256": sha256_file(behavior_summary_path(model)),
        "case_ids": [row["case_id"] for row in cases],
        "case_count": len(cases),
        "components": list(COMPONENTS),
        "semantic_roles": list(ROLE_GROUPS),
        "full_vectors_persisted": False,
        "causal_intervention_executed": False,
        "sealed_split_read": False,
    }
    if restart:
        for path in (rows_path(model), summary_path(model), contract_path(model)):
            path.unlink(missing_ok=True)
    if contract_path(model).exists():
        existing = read_json(contract_path(model))
        for key in (
            "model", "open_cases_sha256", "behavior_summary_sha256", "case_ids",
            "case_count", "components", "semantic_roles", "full_vectors_persisted",
            "causal_intervention_executed", "sealed_split_read",
        ):
            if existing[key] != payload[key]:
                raise RuntimeError(f"Phase571 trace contract drift: {model}/{key}")
    else:
        write_json(contract_path(model), payload)


def run(model: str, batch_size: int, restart: bool) -> Path:
    cases = load_cases(model)
    prepare(model, cases, restart)
    if rows_path(model).exists() and summary_path(model).exists() and not restart:
        return summary_path(model)
    loaded = None
    handles: list[Any] = []
    current_positions: torch.Tensor | None = None
    current_direction: torch.Tensor | None = None
    batch_trace: dict[str, list[torch.Tensor | None]] = {}
    output_rows: list[dict[str, Any]] = []
    started = time.monotonic()
    try:
        loaded = load_probe_model(model)
        loaded.tokenizer.padding_side = "left"
        layers = get_layers(loaded.model)
        layer_count = len(layers)
        run_dtype = str(next(loaded.model.parameters()).dtype)
        if run_dtype != "torch.bfloat16":
            raise RuntimeError(f"Phase571 trace requires BF16, got {run_dtype}")
        output_embeddings = loaded.model.get_output_embeddings()
        if output_embeddings is None or not hasattr(output_embeddings, "weight"):
            raise TypeError("Phase571 cannot locate output embeddings")

        def record(layer_index: int, component: str, output: Any) -> None:
            if current_positions is None or current_direction is None:
                raise RuntimeError("Phase571 trace coordinates are unset")
            value = tensor_from_output(output)
            positions = current_positions.to(value.device)
            batch_indices = torch.arange(value.shape[0], device=value.device)[:, None]
            selected = value[batch_indices, positions, :].float()
            direction = current_direction.to(value.device).float()[:, None, :]
            projection = (selected * direction).sum(dim=-1)
            batch_trace[component][layer_index] = projection.detach().cpu()

        def make_hook(layer_index: int, component: str):
            def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
                record(layer_index, component, output)
            return hook

        for layer_index, layer in enumerate(layers):
            handles.append(
                layer.self_attn.register_forward_hook(
                    make_hook(layer_index, "attention_output")
                )
            )
            handles.append(
                layer.mlp.register_forward_hook(make_hook(layer_index, "mlp_output"))
            )

        for start in range(0, len(cases), batch_size):
            batch_rows = cases[start:start + batch_size]
            prompts = [render_chat(loaded.tokenizer, model, row["raw_prompt"]) for row in batch_rows]
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
                    raise RuntimeError("Phase571 individual/batch tokenization drift")
                offset = sequence_length - len(ids)
                index_rows.append([offset + groups[role][-1] for role in ROLE_GROUPS])
            current_positions = torch.tensor(index_rows, dtype=torch.long)
            target_ids = torch.tensor([
                row["candidate_token_ids"][row["target"]][0] for row in batch_rows
            ], dtype=torch.long, device=output_embeddings.weight.device)
            other_ids = torch.tensor([
                row["candidate_token_ids"][row["other_relation_target"]][0]
                for row in batch_rows
            ], dtype=torch.long, device=output_embeddings.weight.device)
            direction = (
                output_embeddings.weight[target_ids] - output_embeddings.weight[other_ids]
            ).detach().float()
            current_direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            batch_trace = {
                component: [None for _ in range(layer_count)] for component in COMPONENTS
            }
            encoded = {key: value.to(loaded.input_device) for key, value in encoded.items()}
            with torch.inference_mode():
                result = loaded.model(**encoded, use_cache=False)
            if any(value is None for component in COMPONENTS for value in batch_trace[component]):
                raise RuntimeError("Phase571 trace hook ledger is incomplete")
            answer_positions = current_positions[:, ROLE_GROUPS.index("answer_boundary")].to(
                result.logits.device
            )
            batch_indices = torch.arange(len(batch_rows), device=result.logits.device)
            final_logits = result.logits[batch_indices, answer_positions, :].float()
            margins = final_logits[batch_indices, target_ids.to(final_logits.device)] - final_logits[
                batch_indices, other_ids.to(final_logits.device)
            ]
            attention = torch.stack(
                [value for value in batch_trace["attention_output"] if value is not None], dim=-1
            )
            mlp = torch.stack(
                [value for value in batch_trace["mlp_output"] if value is not None], dim=-1
            )
            for row_index, row in enumerate(batch_rows):
                for role_index, role in enumerate(ROLE_GROUPS):
                    a_values = [float(value) for value in attention[row_index, role_index].tolist()]
                    m_values = [float(value) for value in mlp[row_index, role_index].tolist()]
                    output_rows.append({
                        "schema_version": "phase571_signed_write_trace.v1",
                        "phase_id": protocol.PHASE,
                        "created_at": now(),
                        "model": model,
                        "case_id": row["case_id"],
                        "pool": row["pool"],
                        "phenotype": row["trace_phenotype"],
                        "source_factorial_cell": row["source_factorial_cell"],
                        "target": row["target"],
                        "other_relation_target": row["other_relation_target"],
                        "semantic_role": role,
                        "layer_count": layer_count,
                        "attention_signed_unit_projection": a_values,
                        "mlp_signed_unit_projection": m_values,
                        "total_signed_write": float(sum(a_values) + sum(m_values)),
                        "natural_final_target_other_logit_margin": float(margins[row_index].item()),
                        "full_vector_persisted": False,
                        "observer_only": True,
                        "causal": False,
                        "sealed": False,
                    })
            del result, encoded, final_logits, margins, attention, mlp
            current_positions = None
            current_direction = None
            batch_trace = {}
            done = min(start + batch_size, len(cases))
            if start == 0 or done == len(cases) or (start // batch_size) % 8 == 7:
                print(
                    f"[{time.strftime('%H:%M:%S')}] {model} Phase571 signed trace "
                    f"{done}/{len(cases)}",
                    flush=True,
                )

        write_jsonl(rows_path(model), output_rows)
        summary = {
            "schema_version": "phase571_signed_write_summary.v1",
            "phase_id": protocol.PHASE,
            "created_at": now(),
            "status": "complete",
            "model": model,
            "torch_dtype": run_dtype,
            "case_count": len(cases),
            "case_count_per_pool_phenotype": protocol.TRACE_SELECTION_PER_PHENOTYPE,
            "layer_count": layer_count,
            "component_count": len(COMPONENTS),
            "semantic_role_count": len(ROLE_GROUPS),
            "trace_row_count": len(output_rows),
            "runtime_seconds": time.monotonic() - started,
            "rows_sha256": sha256_file(rows_path(model)),
            "full_vectors_persisted": False,
            "causal_intervention_executed": False,
            "head_channel_parameter_neuron_scan_executed": False,
            "sealed_split_read": False,
        }
        write_json(summary_path(model), summary)
        print(summary_path(model), flush=True)
        return summary_path(model)
    finally:
        for handle in handles:
            handle.remove()
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
