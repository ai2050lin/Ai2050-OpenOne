#!/usr/bin/env python3
"""Collect observer-free, prefill-only physical measurements for Phase416.

Only the instrument domain qualified by phase416_qualification_analysis.py is
used.  Every prompt layer and every prompt position is reduced without an
unembedding observer.  Exact role vectors are retained for four sealed anchor
cases per model; the remaining denominator uses compact physical summaries.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
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

from hf_probe_env import get_layers, load_probe_model, release_loaded, vram_gb  # noqa: E402
from phase334_natural_contrast_survey import role_positions  # noqa: E402
from phase416_dual_track_case_bank import (  # noqa: E402
    MODELS,
    OUT,
    PHASE_ID,
    SCHEMA_VERSION,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)


CORE_COMPONENTS = (
    "layer_input",
    "attention_output",
    "mlp_output",
    "residual_increment",
    "layer_output",
)
SUBCOMPONENTS = (
    "q_projection",
    "k_projection",
    "v_projection",
    "mlp_gate",
    "mlp_up",
    "mlp_product",
)
POSITION_ROLES = ("source", "query", "answer_start")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def component_tensor(value: Any) -> torch.Tensor:
    if torch.is_tensor(value):
        return value
    if isinstance(value, (tuple, list)) and value and torch.is_tensor(value[0]):
        return value[0]
    raise TypeError(f"Unsupported component output: {type(value).__name__}")


def depth_bin(layer: int, layer_count: int) -> str:
    relative = layer / max(1, layer_count - 1)
    if relative < 1 / 3:
        return "early"
    if relative < 2 / 3:
        return "middle"
    return "late"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def module_if_present(parent: Any, name: str) -> Any | None:
    value = getattr(parent, name, None)
    return value if value is not None and hasattr(value, "register_forward_hook") else None


def encode_case(loaded: Any, case: dict[str, Any]) -> tuple[dict[str, torch.Tensor], list[int]]:
    encoded = loaded.tokenizer(
        case["prompt"],
        return_tensors="pt",
        truncation=True,
        max_length=256,
        add_special_tokens=bool(case["tokenization_add_special_tokens"]),
    )
    prompt_ids = [int(value) for value in encoded["input_ids"][0].tolist()]
    return {key: value.to(loaded.input_device) for key, value in encoded.items()}, prompt_ids


def anchor_case_ids(cases: list[dict[str, Any]]) -> set[str]:
    anchors: set[str] = set()
    for family in ("knowledge_network", "reasoning", "grammar", "protocol_control"):
        match = next(row for row in cases if row["family_id"] == family)
        anchors.add(match["case_id"])
    return anchors


def behavior_qualification(model: str) -> dict[str, bool]:
    complete = read_json(OUT / "models" / model / "phase416_collector_complete.json")
    return {
        row["family_id"]: bool(row["formal_behavior_qualified"])
        for row in complete["behavior_cells"]
    }


class PhysicalCollector:
    def __init__(
        self,
        loaded: Any,
        cases: list[dict[str, Any]],
        qualified_families: dict[str, bool],
    ) -> None:
        self.loaded = loaded
        self.layers = get_layers(loaded.model)
        self.cases = cases
        self.qualified_families = qualified_families
        self.anchor_ids = anchor_case_ids(cases)
        self.rows: list[dict[str, Any]] = []
        self.case_rows: list[dict[str, Any]] = []
        self.anchor_vectors: dict[str, torch.Tensor] = {}
        self.state: dict[str, Any] = {
            "case": None,
            "positions": None,
            "layer_inputs": {},
            "current_case_row_count": 0,
        }
        self.handles: list[Any] = []

    def base_row(self, layer: int, component: str) -> dict[str, Any]:
        case = self.state["case"]
        return {
            "schema_version": SCHEMA_VERSION,
            "phase_id": PHASE_ID,
            "created_at": now(),
            "model": self.loaded.key,
            "case_id": case["case_id"],
            "semantic_case_id": case["semantic_case_id"],
            "family_id": case["family_id"],
            "mechanism_id": case["mechanism_id"],
            "item_index": int(case["item_index"]),
            "split": case["split"],
            "template_id": case["template_id"],
            "formal_behavior_qualified": self.qualified_families.get(case["family_id"], False),
            "layer": layer,
            "relative_depth": layer / max(1, len(self.layers) - 1),
            "depth_bin": depth_bin(layer, len(self.layers)),
            "component": component,
            "observer_id": None,
            "physical": True,
            "natural_prefill": True,
            "generation_time": False,
            "predictive": False,
            "causal": False,
            "single_neuron_causal": False,
        }

    def record(self, layer: int, component: str, value: Any, all_positions: bool = False) -> None:
        tensor = component_tensor(value)
        if tensor.ndim != 3 or tensor.shape[0] != 1:
            return
        positions: dict[str, tuple[int, bool]] = self.state["positions"]
        valid_roles = [role for role in POSITION_ROLES if positions[role][0] < tensor.shape[1]]
        if not valid_roles:
            return
        indices = torch.tensor(
            [positions[role][0] for role in valid_roles], dtype=torch.long, device=tensor.device
        )
        vectors = tensor[0, indices].detach().float()
        norms = torch.linalg.vector_norm(vectors, dim=-1)
        rms = torch.sqrt(torch.mean(vectors.square(), dim=-1))
        means = torch.mean(vectors, dim=-1)
        maxima, max_indices = torch.max(vectors.abs(), dim=-1)
        metrics = torch.stack([norms, rms, means, maxima], dim=-1).cpu()
        max_indices_cpu = max_indices.cpu()
        case = self.state["case"]
        for role_index, role in enumerate(valid_roles):
            position, exact = positions[role]
            row = {
                **self.base_row(layer, component),
                "position_role": role,
                "position_index": int(position),
                "position_exact": bool(exact),
                "vector_width": int(vectors.shape[-1]),
                "l2_norm": float(metrics[role_index, 0].item()),
                "rms": float(metrics[role_index, 1].item()),
                "signed_mean": float(metrics[role_index, 2].item()),
                "max_abs": float(metrics[role_index, 3].item()),
                "max_abs_unit": int(max_indices_cpu[role_index].item()),
                "all_position_mean_l2": None,
                "all_position_max_l2": None,
                "all_position_max_index": None,
            }
            self.rows.append(row)
            if case["case_id"] in self.anchor_ids and component in CORE_COMPONENTS:
                key = f"{case['case_id']}|L{layer}|{component}|{role}"
                self.anchor_vectors[key] = vectors[role_index].to(dtype=torch.float16).cpu().contiguous()
        if all_positions:
            matrix = tensor[0].detach().float()
            position_norms = torch.linalg.vector_norm(matrix, dim=-1)
            max_norm, max_position = torch.max(position_norms, dim=0)
            self.rows.append(
                {
                    **self.base_row(layer, component),
                    "position_role": "all_positions",
                    "position_index": None,
                    "position_exact": True,
                    "vector_width": int(matrix.shape[-1]),
                    "l2_norm": None,
                    "rms": None,
                    "signed_mean": None,
                    "max_abs": None,
                    "max_abs_unit": None,
                    "all_position_mean_l2": float(position_norms.mean().item()),
                    "all_position_max_l2": float(max_norm.item()),
                    "all_position_max_index": int(max_position.item()),
                }
            )
        self.state["current_case_row_count"] += len(valid_roles) + int(all_positions)

    def install(self) -> None:
        for layer_index, layer in enumerate(self.layers):
            def layer_pre(_module: Any, inputs: tuple[Any, ...], index: int = layer_index) -> None:
                if inputs and torch.is_tensor(inputs[0]):
                    self.state["layer_inputs"][index] = inputs[0]
                    self.record(index, "layer_input", inputs[0], all_positions=True)

            def attention_post(_module: Any, _inputs: tuple[Any, ...], output: Any, index: int = layer_index) -> None:
                self.record(index, "attention_output", output, all_positions=True)

            def mlp_post(_module: Any, _inputs: tuple[Any, ...], output: Any, index: int = layer_index) -> None:
                self.record(index, "mlp_output", output, all_positions=True)

            def layer_post(_module: Any, _inputs: tuple[Any, ...], output: Any, index: int = layer_index) -> None:
                tensor = component_tensor(output)
                before = self.state["layer_inputs"].pop(index)
                self.record(index, "residual_increment", tensor - before, all_positions=True)
                self.record(index, "layer_output", tensor, all_positions=True)

            self.handles.extend(
                [
                    layer.register_forward_pre_hook(layer_pre),
                    layer.self_attn.register_forward_hook(attention_post),
                    layer.mlp.register_forward_hook(mlp_post),
                    layer.register_forward_hook(layer_post),
                ]
            )
            attention = layer.self_attn
            for module_name, component in (
                ("q_proj", "q_projection"),
                ("k_proj", "k_projection"),
                ("v_proj", "v_projection"),
            ):
                module = module_if_present(attention, module_name)
                if module is None:
                    continue

                def projection_post(
                    _module: Any,
                    _inputs: tuple[Any, ...],
                    output: Any,
                    index: int = layer_index,
                    component_name: str = component,
                ) -> None:
                    self.record(index, component_name, output)

                self.handles.append(module.register_forward_hook(projection_post))

            mlp = layer.mlp
            for module_name, component in (("gate_proj", "mlp_gate"), ("up_proj", "mlp_up")):
                module = module_if_present(mlp, module_name)
                if module is None:
                    continue

                def mlp_projection_post(
                    _module: Any,
                    _inputs: tuple[Any, ...],
                    output: Any,
                    index: int = layer_index,
                    component_name: str = component,
                ) -> None:
                    self.record(index, component_name, output)

                self.handles.append(module.register_forward_hook(mlp_projection_post))
            down_projection = module_if_present(mlp, "down_proj")
            if down_projection is not None:
                def product_pre(_module: Any, inputs: tuple[Any, ...], index: int = layer_index) -> None:
                    if inputs and torch.is_tensor(inputs[0]):
                        self.record(index, "mlp_product", inputs[0])

                self.handles.append(down_projection.register_forward_pre_hook(product_pre))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    @torch.inference_mode()
    def run(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, torch.Tensor]]:
        self.install()
        try:
            for case_index, case in enumerate(self.cases, start=1):
                encoded, prompt_ids = encode_case(self.loaded, case)
                self.state["case"] = case
                self.state["positions"] = role_positions(self.loaded, case, prompt_ids)
                self.state["current_case_row_count"] = 0
                output = self.loaded.model(**encoded, use_cache=False, return_dict=True)
                finite_logits = bool(torch.isfinite(output.logits).all().item())
                self.case_rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "phase_id": PHASE_ID,
                        "created_at": now(),
                        "model": self.loaded.key,
                        "case_id": case["case_id"],
                        "semantic_case_id": case["semantic_case_id"],
                        "family_id": case["family_id"],
                        "mechanism_id": case["mechanism_id"],
                        "prompt_token_count": len(prompt_ids),
                        "physical_row_count": self.state["current_case_row_count"],
                        "finite_terminal_logits": finite_logits,
                        "prefill_instrument_qualified": True,
                        "formal_behavior_qualified": self.qualified_families.get(case["family_id"], False),
                        "observer_id": None,
                        "causal": False,
                    }
                )
                del output, encoded
                if case_index % 5 == 0 or case_index == len(self.cases):
                    print(
                        f"[Phase416:{self.loaded.key}:physical] {case_index}/{len(self.cases)} "
                        f"rows={len(self.rows)}",
                        flush=True,
                    )
        finally:
            self.close()
        return self.rows, self.case_rows, self.anchor_vectors


def run_model(model_key: str) -> dict[str, Any]:
    qualification = read_json(OUT / "phase416_instrument_domain_qualification.json")
    model_gate = next(row for row in qualification["models"] if row["model"] == model_key)
    if not model_gate["prefill_physical_collection_authorized"]:
        raise RuntimeError(f"Prefill physical collection is not authorized for {model_key}")
    cases = [row for row in read_jsonl(OUT / "phase416_registered_cases.jsonl") if row["model"] == model_key]
    if len(cases) != 55:
        raise RuntimeError(f"Expected 55 cases for {model_key}, found {len(cases)}")
    loaded = None
    try:
        print(f"[Phase416] loading {model_key} for prefill physical atlas", flush=True)
        loaded = load_probe_model(model_key)
        collector = PhysicalCollector(loaded, cases, behavior_qualification(model_key))
        rows, case_rows, anchor_vectors = collector.run()
        model_root = OUT / "models" / model_key
        rows_path = model_root / "phase416_prefill_physical_rows.jsonl"
        anchors_path = model_root / "phase416_lossless_role_anchors.pt"
        write_jsonl(rows_path, rows)
        torch.save(
            {
                "schema_version": "phase416_lossless_role_anchors.v1",
                "phase_id": PHASE_ID,
                "model": model_key,
                "vectors": anchor_vectors,
            },
            anchors_path,
        )
        write_jsonl(model_root / "phase416_prefill_case_rows.jsonl", case_rows)
        component_counts: dict[str, int] = defaultdict(int)
        role_counts: dict[str, int] = defaultdict(int)
        for row in rows:
            component_counts[row["component"]] += 1
            role_counts[row["position_role"]] += 1
        summary = {
            "schema_version": SCHEMA_VERSION,
            "phase_id": "Phase416-PrefillPhysicalTrace",
            "created_at": now(),
            "model": model_key,
            "case_count": len(case_rows),
            "physical_row_count": len(rows),
            "layer_count": len(get_layers(loaded.model)),
            "component_count": len(component_counts),
            "component_row_count": dict(sorted(component_counts.items())),
            "position_role_row_count": dict(sorted(role_counts.items())),
            "all_prompt_positions_reduced": True,
            "lossless_anchor_case_count": len({key.split("|", 1)[0] for key in anchor_vectors}),
            "lossless_anchor_vector_count": len(anchor_vectors),
            "lossless_anchor_path": str(anchors_path.relative_to(ROOT)),
            "lossless_anchor_sha256": sha256_file(anchors_path),
            "physical_rows_sha256": sha256_file(rows_path),
            "finite_case_count": sum(row["finite_terminal_logits"] for row in case_rows),
            "observer_id": None,
            "generation_time_collected": False,
            "causal": False,
            "single_neuron_causal": False,
            "vram_gb": vram_gb(),
            "valid": bool(
                len(case_rows) == 55
                and len(rows) > 0
                and all(row["finite_terminal_logits"] for row in case_rows)
                and len({key.split("|", 1)[0] for key in anchor_vectors}) == 4
            ),
            "claim_boundary": "observer_free_prefill_physical_measurements_not_lossless_full_tensor_atlas_or_mechanism",
        }
        write_json(model_root / "phase416_prefill_physical_complete.json", summary)
        return summary
    finally:
        release_loaded(loaded)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    args = parser.parse_args()
    summary = run_model(args.model)
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    if not summary["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
