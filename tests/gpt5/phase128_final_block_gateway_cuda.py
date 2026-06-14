#!/usr/bin/env python3
"""
Phase 128: final block and norm/readout gateway test.

Split the peak final block into finer observable sites, then test whether the
pre-answer category causal field survives into final norm/readout space.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hf_probe_env import get_layers, load_probe_model, release_loaded, vram_gb  # noqa: E402
from phase105_global_category_atlas_cuda import CATEGORY_OBJECTS, collect_readout_rows  # noqa: E402
from phase106_multitemplate_residual_cuda import TEMPLATES  # noqa: E402
from phase107_causal_boundary_removal_cuda import BOUNDARY_LAYER, score_logits, summarize_delta  # noqa: E402
from phase114_answer_site_causal_subspace_cuda import build_category_contrast_matrix  # noqa: E402
from phase116_subspace_basis_component_audit_cuda import build_prompts, svd_basis  # noqa: E402
from phase120_post_object_token_localization_cuda import item_positions, select_local_varimax_axis  # noqa: E402
from phase122_pre_answer_to_answer_projection_closure_cuda import projection_values  # noqa: E402
from phase123_attention_mlp_writer_localization_cuda import get_mlp_module  # noqa: E402
from phase126_residual_gap_decomposition_cuda import (  # noqa: E402
    ANSWER_SITE,
    PRE_SITE,
    capture_answer_centers,
    positions_for_site,
    replace_output_tensor,
    tensor_from_output,
)
from phase112_attention_transport_head_mapping_cuda import get_attention_module  # noqa: E402


OUT_ROOT = Path("results/gpt5_phase128_final_block_gateway")
TEST_CATEGORIES = ["number", "container", "plant"]
DEFAULT_COMPONENTS = [
    "block_input",
    "attention_output",
    "post_attention_norm_input",
    "mlp_input",
    "mlp_output",
    "block_output",
    "final_norm_input",
    "final_norm_output",
]


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _get_attr_path(obj: Any, path: str) -> Any | None:
    cur = obj
    for part in path.split("."):
        if not hasattr(cur, part):
            return None
        cur = getattr(cur, part)
    return cur


def get_post_attention_norm(layer: Any) -> Any | None:
    for name in (
        "post_attention_layernorm",
        "post_attention_norm",
        "post_attn_layernorm",
        "post_attn_norm",
        "ln2",
        "ffn_norm",
    ):
        if hasattr(layer, name):
            return getattr(layer, name)
    return None


def get_final_norm(model: Any) -> Any | None:
    candidates = (
        "model.norm",
        "model.final_layernorm",
        "transformer.norm",
        "transformer.final_layernorm",
        "transformer.encoder.final_layernorm",
        "transformer.encoder.final_layer_norm",
        "transformer.ln_f",
        "gpt_neox.final_layer_norm",
        "norm",
        "final_layernorm",
    )
    for path in candidates:
        module = _get_attr_path(model, path)
        if module is not None:
            return module
    return None


def get_gateway_module(model: Any, layers: list[Any], layer_id: int, component: str) -> Any | None:
    layer = layers[layer_id - 1]
    if component in {"block_input", "block_output"}:
        return layer
    if component == "attention_output":
        return get_attention_module(layer)
    if component == "post_attention_norm_input":
        return get_post_attention_norm(layer)
    if component == "mlp_input" or component == "mlp_output":
        return get_mlp_module(layer)
    if component in {"final_norm_input", "final_norm_output"}:
        return get_final_norm(model)
    raise ValueError(component)


def is_pre_component(component: str) -> bool:
    return component in {
        "block_input",
        "post_attention_norm_input",
        "mlp_input",
        "final_norm_input",
    }


def make_capture_hook(component: str, store: dict[str, torch.Tensor]):
    if is_pre_component(component):
        def pre_hook(_module: Any, inputs: tuple[Any, ...]):
            store["value"] = inputs[0].detach()
        return pre_hook, True

    def hook(_module: Any, _inputs: Any, output: Any):
        store["value"] = tensor_from_output(output).detach()
    return hook, False


def make_subspace_patch_hook(component: str, basis: torch.Tensor, batch_positions: list[list[int]], scale: float):
    basis = basis / (basis.norm(dim=1, keepdim=True) + 1e-8)

    def patch_tensor(x: torch.Tensor) -> torch.Tensor:
        out = x.clone()
        b = basis.to(out.device).float()
        for bi, positions in enumerate(batch_positions):
            if not positions:
                continue
            pos = torch.tensor(positions, device=out.device, dtype=torch.long)
            vecs = out[bi, pos, :].float()
            proj = (vecs @ b.T) @ b
            out[bi, pos, :] = out[bi, pos, :] - scale * proj.to(out.dtype)
        return out

    if is_pre_component(component):
        def pre_hook(_module: Any, inputs: tuple[Any, ...]):
            return (patch_tensor(inputs[0]),) + inputs[1:]
        return pre_hook, True

    def hook(_module: Any, _inputs: Any, output: Any):
        return replace_output_tensor(output, patch_tensor(tensor_from_output(output)))
    return hook, False


def audit_prompt_positions(tokenizer: Any, prompts: list[dict[str, Any]], max_length: int) -> dict[str, Any]:
    answer_in_pre = 0
    empty_pre = 0
    lengths = []
    max_pre_positions = []
    for item in prompts:
        batch = tokenizer([item["prompt"]], return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        answer_pos = int(batch["attention_mask"].sum(dim=1).item() - 1)
        positions = positions_for_site(tokenizer, item, answer_pos, PRE_SITE)
        if answer_pos in positions:
            answer_in_pre += 1
        if not positions:
            empty_pre += 1
        lengths.append(len(positions))
        max_pre_positions.append(None if not positions else max(positions))
    return {
        "n_prompts": len(prompts),
        "answer_in_pre_count": int(answer_in_pre),
        "empty_pre_count": int(empty_pre),
        "mean_pre_len": float(np.mean(lengths)) if lengths else 0.0,
        "min_pre_len": int(min(lengths)) if lengths else 0,
        "max_pre_len": int(max(lengths)) if lengths else 0,
        "max_pre_position_max": None if all(x is None for x in max_pre_positions) else int(max(x for x in max_pre_positions if x is not None)),
    }


def capture_gateway_centers(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    categories: list[str],
    layer_id: int,
    component: str,
    train_objects: int,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    d_model = int(model.get_input_embeddings().weight.shape[1])
    centers = np.zeros((len(TEMPLATES), len(categories), d_model), dtype=np.float64)
    counts = np.zeros((len(TEMPLATES), len(categories)), dtype=np.int64)
    items: list[dict[str, Any]] = []
    for ti, tpl in enumerate(TEMPLATES):
        for ci, cat in enumerate(categories):
            for obj in CATEGORY_OBJECTS[cat][:train_objects]:
                items.append({"ti": ti, "ci": ci, "cat": cat, "obj": obj, "prompt": tpl["text"].format(obj=obj)})

    module = get_gateway_module(model, layers, layer_id, component)
    if module is None:
        raise RuntimeError(f"component unavailable: {component}")
    with torch.no_grad():
        for start in range(0, len(items), batch_size):
            batch_items = items[start:start + batch_size]
            texts = [x["prompt"] for x in batch_items]
            batch = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            batch = {k: v.to(device) for k, v in batch.items()}
            store: dict[str, torch.Tensor] = {}
            hook_fn, is_pre = make_capture_hook(component, store)
            handle = module.register_forward_pre_hook(hook_fn) if is_pre else module.register_forward_hook(hook_fn)
            _ = model(**batch, use_cache=False)
            handle.remove()
            value = store["value"]
            answer_pos = (batch["attention_mask"].sum(dim=1) - 1).detach().cpu().tolist()
            for bi, item in enumerate(batch_items):
                positions = positions_for_site(tokenizer, item, answer_pos[bi], PRE_SITE)
                pos = torch.tensor(positions, device=value.device, dtype=torch.long)
                vec = value[bi, pos, :].float().mean(dim=0).detach().cpu().numpy()
                centers[item["ti"], item["ci"]] += vec
                counts[item["ti"], item["ci"]] += 1
            del batch
            torch.cuda.empty_cache()
    return (centers / counts[:, :, None]).astype(np.float32)


def run_gateway_condition(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    prompts: list[dict[str, Any]],
    cat_local_ids: dict[str, list[int]],
    categories: list[str],
    batch_size: int,
    max_length: int,
    monitor_layer: int,
    monitor_basis: np.ndarray,
    patch_layer: int | None = None,
    component: str | None = None,
    basis_np: np.ndarray | None = None,
    scale: float = 1.5,
) -> dict[str, np.ndarray]:
    scores = []
    answer_proj = []
    for start in range(0, len(prompts), batch_size):
        items = prompts[start:start + batch_size]
        texts = [x["prompt"] for x in items]
        batch = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        batch = {k: v.to(device) for k, v in batch.items()}
        answer_pos = (batch["attention_mask"].sum(dim=1) - 1).detach().cpu().tolist()
        handles = []
        if patch_layer is not None and component is not None and basis_np is not None:
            positions = [positions_for_site(tokenizer, item, answer_pos[bi], PRE_SITE) for bi, item in enumerate(items)]
            basis = torch.tensor(basis_np, device=device, dtype=torch.float32)
            hook_fn, is_pre = make_subspace_patch_hook(component, basis, positions, scale)
            module = get_gateway_module(model, layers, patch_layer, component)
            if module is None:
                raise RuntimeError(f"component unavailable: {component}")
            handles.append(module.register_forward_pre_hook(hook_fn) if is_pre else module.register_forward_hook(hook_fn))
        with torch.no_grad():
            out = model(**batch, output_hidden_states=True, use_cache=False)
        for handle in handles:
            handle.remove()
        pos_gpu = torch.tensor(answer_pos, device=out.logits.device, dtype=torch.long)
        logits = out.logits[torch.arange(out.logits.shape[0], device=out.logits.device), pos_gpu]
        scores.append(score_logits(logits, cat_local_ids, categories))
        hs = out.hidden_states[monitor_layer]
        ans = hs[torch.arange(hs.shape[0], device=hs.device), pos_gpu.to(hs.device), :].float()
        answer_proj.append(projection_values(ans, monitor_basis))
        del out, batch
        torch.cuda.empty_cache()
    return {"scores": np.concatenate(scores, axis=0), "answer_proj": np.concatenate(answer_proj, axis=0)}


def summarize_gateway(patched: dict[str, np.ndarray], baseline: dict[str, np.ndarray], target_idx: int, categories: list[str]) -> dict[str, Any]:
    out = summarize_delta(patched["scores"] - baseline["scores"], target_idx, categories)
    out["answer_proj_delta"] = float((patched["answer_proj"] - baseline["answer_proj"]).mean())
    return out


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    loaded = load_probe_model(args.model)
    try:
        model = loaded.model
        tokenizer = loaded.tokenizer
        device = loaded.input_device
        layers = get_layers(model)
        categories = list(CATEGORY_OBJECTS.keys())
        test_categories = [x.strip() for x in args.categories.split(",") if x.strip()] or TEST_CATEGORIES
        peak_layer = args.peak_layer if args.peak_layer is not None else BOUNDARY_LAYER[args.model]
        requested_components = [x.strip() for x in args.components.split(",") if x.strip()]
        available_components = []
        unavailable_components = []
        for component in requested_components:
            module = get_gateway_module(model, layers, peak_layer, component)
            if module is None:
                unavailable_components.append(component)
            else:
                available_components.append(component)
        cat_local_ids, _rows, token_labels = collect_readout_rows(model, tokenizer, categories)
        alloc, reserved = vram_gb()
        log(
            f"{args.model}: peak=L{peak_layer}, components={available_components}, "
            f"unavailable={unavailable_components}, train/test={args.train_objects}/{args.test_objects}, "
            f"vram={alloc:.2f}/{reserved:.2f}GB"
        )

        result: dict[str, Any] = {
            "phase": 128,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "peak_layer": peak_layer,
            "monitor_layer": peak_layer,
            "requested_components": requested_components,
            "available_components": available_components,
            "unavailable_components": unavailable_components,
            "train_objects_per_category": args.train_objects,
            "test_objects_per_category": args.test_objects,
            "templates": [t["name"] for t in TEMPLATES],
            "test_categories": test_categories,
            "rank": args.rank,
            "scale": args.scale,
            "readout_token_labels": token_labels,
            "category_results": {},
        }

        log("Building answer monitor centers")
        answer_centers = capture_answer_centers(
            model, tokenizer, device, categories, peak_layer,
            args.train_objects, args.batch_size, args.max_length,
        )
        center_cache: dict[str, np.ndarray] = {}
        for component in available_components:
            log(f"Capturing centers L{peak_layer} {component}")
            center_cache[component] = capture_gateway_centers(
                model, tokenizer, device, layers, categories, peak_layer, component,
                args.train_objects, args.batch_size, args.max_length,
            )

        for ci, cat in enumerate(test_categories, 1):
            log(f"Testing {args.model} {ci}/{len(test_categories)} {cat}")
            target_idx = categories.index(cat)
            prompts = build_prompts(cat, args.train_objects, args.test_objects)
            answer_basis, answer_sv = svd_basis(build_category_contrast_matrix(answer_centers, categories, cat), args.rank)
            baseline_for_selection = run_gateway_condition(
                model, tokenizer, device, layers, prompts, cat_local_ids, categories,
                args.batch_size, args.max_length, peak_layer, answer_basis,
            )
            answer_choice = select_local_varimax_axis(
                model, tokenizer, device, layers, prompts, baseline_for_selection["scores"],
                peak_layer, ANSWER_SITE, cat_local_ids, categories, target_idx,
                args.batch_size, args.max_length, args.scale, answer_basis,
            )
            monitor_basis = answer_choice["axis"] if args.monitor_axis == "varimax" else answer_basis
            baseline = run_gateway_condition(
                model, tokenizer, device, layers, prompts, cat_local_ids, categories,
                args.batch_size, args.max_length, peak_layer, monitor_basis,
            )
            cat_out = {
                "n_prompts": len(prompts),
                "position_audit": audit_prompt_positions(tokenizer, prompts, args.max_length),
                "baseline_target_mean": float(baseline["scores"][:, target_idx].mean()),
                "baseline_answer_proj_mean": float(baseline["answer_proj"].mean()),
                "answer_singular_values": [float(x) for x in answer_sv],
                "answer_varimax_selection": {
                    "basis_index": int(answer_choice["basis_index"]),
                    "selection_target_delta": float(answer_choice["selection_target_delta"]),
                },
                "conditions": [],
            }
            for component in available_components:
                basis, sv = svd_basis(build_category_contrast_matrix(center_cache[component], categories, cat), args.rank)
                patched = run_gateway_condition(
                    model, tokenizer, device, layers, prompts, cat_local_ids, categories,
                    args.batch_size, args.max_length, peak_layer, monitor_basis,
                    patch_layer=peak_layer,
                    component=component,
                    basis_np=basis,
                    scale=args.scale,
                )
                cat_out["conditions"].append({
                    "patch_layer": int(peak_layer),
                    "component": component,
                    "singular_values": [float(x) for x in sv],
                    **summarize_gateway(patched, baseline, target_idx, categories),
                })
            result["category_results"][cat] = cat_out
        return result
    finally:
        release_loaded(loaded)


def _fmt(row: dict[str, Any] | None) -> str:
    if row is None:
        return "NA"
    return f"T{row['target_delta']:+.2f} R{row['max_other_delta']:+.2f} A{row['answer_proj_delta']:+.2f}"


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [f"# Phase 128 Final Block Gateway: {result['model']}", ""]
    lines.append(f"Generated: {result['timestamp']}")
    lines.append(f"Peak layer: L{result['peak_layer']}; components: {', '.join(result['available_components'])}")
    if result["unavailable_components"]:
        lines.append(f"Unavailable components: {', '.join(result['unavailable_components'])}")
    lines.append("")
    cols = result["available_components"]
    lines.append("| category | position audit | " + " | ".join(cols) + " |")
    lines.append("|---|---|" + "|".join(["---"] * len(cols)) + "|")
    for cat, item in result["category_results"].items():
        conds = {x["component"]: x for x in item["conditions"]}
        audit = item["position_audit"]
        audit_text = f"answer_in_pre={audit['answer_in_pre_count']}, mean_pre_len={audit['mean_pre_len']:.1f}"
        lines.append(
            f"| {cat} | {audit_text} | "
            + " | ".join(_fmt(conds.get(component)) for component in cols)
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--train-objects", type=int, default=8)
    parser.add_argument("--test-objects", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--max-length", type=int, default=80)
    parser.add_argument("--peak-layer", type=int, default=None)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--scale", type=float, default=1.5)
    parser.add_argument("--components", default=",".join(DEFAULT_COMPONENTS))
    parser.add_argument("--monitor-axis", choices=["varimax", "subspace"], default="varimax")
    parser.add_argument("--categories", default="")
    parser.add_argument("--output-dir", default=str(OUT_ROOT))
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = run_model(args)
    json_path = out_dir / f"phase128_{args.model}_final_block_gateway.json"
    md_path = out_dir / f"phase128_{args.model}_final_block_gateway.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(result, md_path)
    log(f"Wrote {json_path}")
    log(f"Wrote {md_path}")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
