#!/usr/bin/env python3
"""Collect natural component trajectories without selecting causal units."""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from hf_probe_env import load_probe_model, release_loaded  # noqa: E402
from phase334_natural_contrast_survey import role_positions  # noqa: E402
from phase338_block_causal_screen import (  # noqa: E402
    component_tensor, continuation_ids, get_layers, layers_for_bin, prompt_ids,
)
from phase347_three_core_natural_trace_case_bank import (  # noqa: E402
    OUT, PHASE, ROUND_DEFAULT, SCHEMA_VERSION,
)


MODELS = ("qwen3", "glm4", "deepseek7b")
ROLES = ("source", "query", "answer_start")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def depth_lookup(layer_count: int) -> dict[int, str]:
    return {
        layer: depth
        for depth in ("early", "middle", "late")
        for layer in layers_for_bin(layer_count, depth)
    }


def install_capture_hooks(loaded: Any, state: dict[str, Any]) -> list[Any]:
    handles = []
    layers = get_layers(loaded.model)
    state["pre_inputs"] = {}

    for layer_index, layer in enumerate(layers):
        def pre(_module: Any, inputs: tuple[Any, ...], idx: int = layer_index) -> None:
            if inputs and torch.is_tensor(inputs[0]):
                state["pre_inputs"][idx] = inputs[0].detach()

        def capture(output: Any, idx: int, component: str) -> None:
            tensor = component_tensor(output)
            positions = state.get("positions")
            if positions is None or tensor.ndim != 3 or tensor.shape[0] != 1:
                return
            if int(positions.max().item()) >= tensor.shape[1]:
                return
            state["captures"].append((idx, component, tensor[0, positions].detach()))

        def attention(_module: Any, _inputs: tuple[Any, ...], output: Any, idx: int = layer_index) -> None:
            capture(output, idx, "attention_output")

        def mlp(_module: Any, _inputs: tuple[Any, ...], output: Any, idx: int = layer_index) -> None:
            capture(output, idx, "mlp_output")

        def residual(_module: Any, _inputs: tuple[Any, ...], output: Any, idx: int = layer_index) -> None:
            tensor = component_tensor(output)
            before = state["pre_inputs"].get(idx)
            positions = state.get("positions")
            if before is None or positions is None or tensor.ndim != 3:
                return
            increment = tensor[0, positions] - before[0, positions].to(tensor.device)
            state["captures"].append((idx, "residual_increment", increment.detach()))

        handles.append(layer.register_forward_pre_hook(pre))
        handles.append(layer.self_attn.register_forward_hook(attention))
        handles.append(layer.mlp.register_forward_hook(mlp))
        handles.append(layer.register_forward_hook(residual))
    return handles


@torch.inference_mode()
def run_model(model: str, round_name: str = ROUND_DEFAULT) -> dict[str, Any]:
    root = OUT / round_name
    cases = [row for row in read_jsonl(root / "phase347_registered_cases.jsonl") if row["model"] == model]
    loaded = None
    handles: list[Any] = []
    case_rows = []
    aggregate: dict[tuple[Any, ...], dict[str, float]] = defaultdict(
        lambda: {
            "count": 0.0, "finite_count": 0.0, "norm_sum": 0.0,
            "projection_sum": 0.0, "abs_projection_sum": 0.0,
            "abs_cosine_sum": 0.0,
            "positive_projection_count": 0.0,
        }
    )
    try:
        loaded = load_probe_model(model)
        layers = get_layers(loaded.model)
        depth_by_layer = depth_lookup(len(layers))
        output_weight = loaded.model.get_output_embeddings().weight.detach()
        state: dict[str, Any] = {"captures": [], "positions": None}
        handles = install_capture_hooks(loaded, state)
        for index, case in enumerate(cases):
            ids = prompt_ids(loaded, case)
            role_map = role_positions(loaded, case, ids)
            positions = [role_map[role][0] for role in ROLES]
            state["positions"] = torch.tensor(positions, dtype=torch.long, device=loaded.input_device)
            state["captures"] = []
            state["pre_inputs"] = {}
            input_ids = torch.tensor([ids], dtype=torch.long, device=loaded.input_device)
            attention_mask = torch.ones_like(input_ids)
            target_id = continuation_ids(loaded, case, case["target"])[0]
            target_direction = output_weight[target_id].detach().float()
            target_direction = target_direction / target_direction.norm().clamp_min(1e-8)
            output = loaded.model(
                input_ids=input_ids, attention_mask=attention_mask,
                use_cache=False, return_dict=True,
            )
            logits = output.logits[0, -1].detach().float()
            target_logit = float(logits[target_id].item())
            target_rank = int((logits > logits[target_id]).sum().item()) + 1
            captures = state["captures"]
            vectors = torch.stack([capture[2] for capture in captures]).float()
            norms = vectors.norm(dim=-1)
            projections = torch.einsum("crh,h->cr", vectors, target_direction.to(vectors.device))
            finite = torch.isfinite(norms) & torch.isfinite(projections)
            norms_cpu = norms.cpu()
            projections_cpu = projections.cpu()
            finite_cpu = finite.cpu()
            for capture_index, (layer_index, component, _vectors) in enumerate(captures):
                for role_index, role in enumerate(ROLES):
                    key = (
                        case["mechanism_id"], case["task_class"], case["template_id"],
                        case["split"], component, layer_index, depth_by_layer[layer_index], role,
                    )
                    bucket = aggregate[key]
                    bucket["count"] += 1
                    if bool(finite_cpu[capture_index, role_index].item()):
                        norm = float(norms_cpu[capture_index, role_index].item())
                        projection = float(projections_cpu[capture_index, role_index].item())
                        bucket["finite_count"] += 1
                        bucket["norm_sum"] += norm
                        bucket["projection_sum"] += projection
                        bucket["abs_projection_sum"] += abs(projection)
                        bucket["abs_cosine_sum"] += abs(projection) / max(norm, 1e-8)
                        bucket["positive_projection_count"] += projection > 0
            case_rows.append({
                "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
                "model": model, "case_id": case["case_id"],
                "mechanism_id": case["mechanism_id"], "task_class": case["task_class"],
                "split": case["split"], "template_id": case["template_id"],
                "target_first_token_id": int(target_id),
                "target_first_logit": round(target_logit, 7) if math.isfinite(target_logit) else None,
                "target_first_rank": target_rank,
                "capture_count": len(captures) * len(ROLES),
                "finite_capture_count": int(finite.sum().item()),
                "natural_trace_only": True, "single_unit_causal": False,
            })
            del output, logits, vectors, norms, projections, input_ids, attention_mask
            if (index + 1) % 40 == 0 or index + 1 == len(cases):
                print(f"[{model}] {index + 1}/{len(cases)}", flush=True)
        trace_rows = []
        for key, bucket in aggregate.items():
            task_id, task_class, template, split, component, layer_index, depth_bin, role = key
            finite_count = bucket["finite_count"]
            trace_rows.append({
                "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
                "model": model, "mechanism_id": task_id, "task_class": task_class,
                "template_id": template, "split": split, "component": component,
                "layer_index": layer_index, "depth_bin": depth_bin, "position_role": role,
                "case_count": int(bucket["count"]), "finite_count": int(finite_count),
                "finite_rate": round(finite_count / bucket["count"], 7),
                "mean_component_l2_norm": round(bucket["norm_sum"] / finite_count, 7) if finite_count else None,
                "mean_target_first_token_projection": round(bucket["projection_sum"] / finite_count, 7) if finite_count else None,
                "mean_abs_target_first_token_projection": round(bucket["abs_projection_sum"] / finite_count, 7) if finite_count else None,
                "mean_abs_target_first_token_cosine": round(bucket["abs_cosine_sum"] / finite_count, 7) if finite_count else None,
                "positive_projection_rate": round(bucket["positive_projection_count"] / finite_count, 7) if finite_count else None,
                "natural_trace_only": True, "single_unit_causal": False,
            })
        trace_rows.sort(key=lambda row: (
            row["mechanism_id"], row["template_id"], row["split"],
            row["layer_index"], row["component"], row["position_role"],
        ))
        model_root = root / "models" / model
        write_jsonl(model_root / "phase347_case_rows.jsonl", case_rows)
        write_jsonl(model_root / "phase347_trace_rows.jsonl", trace_rows)
        complete = {
            "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
            "model": model, "registered_case_count": len(cases),
            "case_row_count": len(case_rows), "trace_row_count": len(trace_rows),
            "nonfinite_case_count": sum(row["finite_capture_count"] != row["capture_count"] for row in case_rows),
            "valid": len(cases) == 240 and len(case_rows) == 240 and bool(trace_rows),
        }
        write_json(model_root / "complete.json", complete)
        return complete
    finally:
        for handle in handles:
            handle.remove()
        release_loaded(loaded)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--round", default=ROUND_DEFAULT)
    args = parser.parse_args()
    print(json.dumps(run_model(args.model, args.round), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
