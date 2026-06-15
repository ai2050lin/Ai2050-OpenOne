#!/usr/bin/env python3
"""
Phase 138: mechanism variable and transfer-map closure.

Learn a low-rank transfer map from last-layer pre-answer residual field
coordinates to answer-site readout coordinates, then test remove / restore /
swap causal closure under long templates.
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
from phase107_causal_boundary_removal_cuda import BOUNDARY_LAYER, score_logits, summarize_delta  # noqa: E402
from phase114_answer_site_causal_subspace_cuda import build_category_contrast_matrix  # noqa: E402
from phase116_subspace_basis_component_audit_cuda import svd_basis  # noqa: E402
from phase122_pre_answer_to_answer_projection_closure_cuda import projection_values  # noqa: E402
from phase126_residual_gap_decomposition_cuda import tensor_from_output  # noqa: E402
from phase135_long_template_source_field_cuda import (  # noqa: E402
    LONG_TEMPLATES,
    batch_context,
    build_long_prompt,
    build_long_prompts,
    position_audit_long,
    source_audit,
)


OUT_ROOT = Path("results/gpt5_phase138_mechanism_transfer_closure")
TEST_CATEGORIES = ["number", "container", "plant", "time", "clothing", "furniture"]


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def normalize_basis(basis: np.ndarray) -> np.ndarray:
    return basis / (np.linalg.norm(basis, axis=1, keepdims=True) + 1e-8)


def project_np(vecs: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return vecs.astype(np.float32) @ normalize_basis(basis).astype(np.float32).T


def ridge_map(x: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    xtx = x.T @ x
    reg = ridge * np.eye(xtx.shape[0], dtype=np.float32)
    return np.linalg.solve(xtx + reg, x.T @ y).astype(np.float32)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean(axis=0, keepdims=True)) ** 2))
    return 1.0 - ss_res / (ss_tot + 1e-8)


def cosine_mean(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    num = np.sum(y_true * y_pred, axis=1)
    den = (np.linalg.norm(y_true, axis=1) * np.linalg.norm(y_pred, axis=1)) + 1e-8
    return float(np.mean(num / den))


def build_items(categories: list[str], train_n: int, test_n: int, split: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    start = 0 if split == "train" else train_n
    count = train_n if split == "train" else test_n
    for ti, tpl in enumerate(LONG_TEMPLATES):
        for cat in categories:
            for obj in CATEGORY_OBJECTS[cat][start:start + count]:
                items.append({
                    "ti": ti,
                    "template": tpl,
                    "template_name": tpl["name"],
                    "cat": cat,
                    "obj": obj,
                    "split": split,
                    "prompt": build_long_prompt(tpl, obj),
                })
    return items


def capture_records(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    items: list[dict[str, Any]],
    cat_local_ids: dict[str, list[int]],
    categories: list[str],
    batch_size: int,
    max_length: int,
    last_layer: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    module = layers[last_layer - 1]
    with torch.no_grad():
        for start in range(0, len(items), batch_size):
            batch_items = items[start:start + batch_size]
            batch = tokenizer([x["prompt"] for x in batch_items], return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            batch = {k: v.to(device) for k, v in batch.items()}
            ctx = batch_context(tokenizer, batch, batch_items)
            stores: dict[str, torch.Tensor] = {}

            def pre_hook(_module: Any, inputs: tuple[Any, ...]):
                stores["pre"] = inputs[0].detach()

            def post_hook(_module: Any, _inputs: Any, output: Any):
                stores["answer"] = tensor_from_output(output).detach()

            h1 = module.register_forward_pre_hook(pre_hook)
            h2 = module.register_forward_hook(post_hook)
            out = model(**batch, use_cache=False)
            h1.remove()
            h2.remove()
            pos_gpu = torch.tensor(ctx["last_pos"], device=out.logits.device, dtype=torch.long)
            logits = out.logits[torch.arange(out.logits.shape[0], device=out.logits.device), pos_gpu]
            scores = score_logits(logits, cat_local_ids, categories)
            pre_value = stores["pre"]
            ans_value = stores["answer"]
            for bi, item in enumerate(batch_items):
                src = ctx["source_groups"]["all_pre_answer"][bi]
                pos = torch.tensor(src, device=pre_value.device, dtype=torch.long)
                pre_vec = pre_value[bi, pos, :].float().mean(dim=0).detach().cpu().numpy()
                ans_vec = ans_value[bi, ctx["last_pos"][bi], :].float().detach().cpu().numpy()
                records.append({
                    "ti": item["ti"],
                    "template_name": item.get("template_name", item.get("template", {}).get("name", f"T{item['ti']}")),
                    "cat": item["cat"],
                    "obj": item["obj"],
                    "split": item.get("split", "test"),
                    "prompt": item["prompt"],
                    "pre_vec": pre_vec.astype(np.float32),
                    "answer_vec": ans_vec.astype(np.float32),
                    "scores": scores[bi].astype(np.float32),
                })
            del out, batch
            torch.cuda.empty_cache()
    return records


def centers_from_records(records: list[dict[str, Any]], categories: list[str], key: str) -> np.ndarray:
    d_model = int(records[0][key].shape[0])
    centers = np.zeros((len(LONG_TEMPLATES), len(categories), d_model), dtype=np.float64)
    counts = np.zeros((len(LONG_TEMPLATES), len(categories)), dtype=np.int64)
    cat_index = {cat: i for i, cat in enumerate(categories)}
    for rec in records:
        ti = int(rec["ti"])
        ci = cat_index[rec["cat"]]
        centers[ti, ci] += rec[key]
        counts[ti, ci] += 1
    return (centers / np.maximum(counts[:, :, None], 1)).astype(np.float32)


def make_restore_hooks(
    tokenizer: Any,
    items: list[dict[str, Any]],
    batch: dict[str, torch.Tensor],
    pre_basis: np.ndarray,
    ans_basis: np.ndarray,
    transfer: np.ndarray,
    restore_scale: float,
    remove_scale: float,
    fixed_answer_coeff: np.ndarray | None = None,
):
    ctx = batch_context(tokenizer, batch, items)
    pre_b = torch.tensor(normalize_basis(pre_basis), dtype=torch.float32)
    ans_b = torch.tensor(normalize_basis(ans_basis), dtype=torch.float32)
    w = torch.tensor(transfer, dtype=torch.float32)
    fixed = None if fixed_answer_coeff is None else torch.tensor(fixed_answer_coeff, dtype=torch.float32)
    restore_store: dict[str, torch.Tensor] = {}

    def pre_hook(_module: Any, inputs: tuple[Any, ...]):
        x = inputs[0]
        out = x.clone()
        pb = pre_b.to(out.device)
        coeff_rows = []
        for bi, positions in enumerate(ctx["source_groups"]["all_pre_answer"]):
            if not positions:
                coeff_rows.append(torch.zeros((pre_b.shape[0],), device=out.device))
                continue
            pos = torch.tensor(positions, device=out.device, dtype=torch.long)
            vecs = out[bi, pos, :].float()
            proj = (vecs @ pb.T) @ pb
            out[bi, pos, :] = out[bi, pos, :] - remove_scale * proj.to(out.dtype)
            coeff_rows.append(vecs.mean(dim=0) @ pb.T)
        coeff = torch.stack(coeff_rows, dim=0)
        pred = fixed.to(out.device).unsqueeze(0).repeat(out.shape[0], 1) if fixed is not None else coeff @ w.to(out.device)
        restore_store["answer_add"] = pred @ ans_b.to(out.device)
        return (out,) + inputs[1:]

    def post_hook(_module: Any, _inputs: Any, output: Any):
        x = tensor_from_output(output)
        out = x.clone()
        add = restore_store["answer_add"].to(out.device, dtype=out.dtype)
        bidx = torch.arange(out.shape[0], device=out.device)
        pos = torch.tensor(ctx["last_pos"], device=out.device, dtype=torch.long)
        out[bidx, pos, :] = out[bidx, pos, :] + restore_scale * add
        if isinstance(output, tuple):
            return (out,) + output[1:]
        return out

    return pre_hook, post_hook, ctx


def run_transfer_condition(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    prompts: list[dict[str, Any]],
    cat_local_ids: dict[str, list[int]],
    categories: list[str],
    batch_size: int,
    max_length: int,
    last_layer: int,
    mode: str,
    pre_basis: np.ndarray,
    ans_basis: np.ndarray,
    transfer: np.ndarray,
    remove_scale: float,
    restore_scale: float,
    fixed_answer_coeff: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    scores = []
    answer_proj = []
    module = layers[last_layer - 1]
    for start in range(0, len(prompts), batch_size):
        items = prompts[start:start + batch_size]
        batch = tokenizer([x["prompt"] for x in items], return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        batch = {k: v.to(device) for k, v in batch.items()}
        ctx = batch_context(tokenizer, batch, items)
        handles = []
        if mode == "remove":
            basis = torch.tensor(normalize_basis(pre_basis), device=device, dtype=torch.float32)

            def remove_hook(_module: Any, inputs: tuple[Any, ...]):
                x = inputs[0]
                out = x.clone()
                b = basis.to(out.device)
                for bi, positions in enumerate(ctx["source_groups"]["all_pre_answer"]):
                    pos = torch.tensor(positions, device=out.device, dtype=torch.long)
                    vecs = out[bi, pos, :].float()
                    proj = (vecs @ b.T) @ b
                    out[bi, pos, :] = out[bi, pos, :] - remove_scale * proj.to(out.dtype)
                return (out,) + inputs[1:]

            handles.append(module.register_forward_pre_hook(remove_hook))
        elif mode in {"restore", "swap"}:
            pre_hook, post_hook, ctx = make_restore_hooks(
                tokenizer, items, batch, pre_basis, ans_basis, transfer,
                restore_scale, remove_scale, fixed_answer_coeff,
            )
            handles.append(module.register_forward_pre_hook(pre_hook))
            handles.append(module.register_forward_hook(post_hook))
        elif mode != "clean":
            raise ValueError(mode)

        with torch.no_grad():
            out = model(**batch, output_hidden_states=True, use_cache=False)
        for h in handles:
            h.remove()
        pos_gpu = torch.tensor(ctx["last_pos"], device=out.logits.device, dtype=torch.long)
        logits = out.logits[torch.arange(out.logits.shape[0], device=out.logits.device), pos_gpu]
        scores.append(score_logits(logits, cat_local_ids, categories))
        hs = out.hidden_states[last_layer]
        ans = hs[torch.arange(hs.shape[0], device=hs.device), pos_gpu.to(hs.device), :].float()
        answer_proj.append(projection_values(ans, ans_basis))
        del out, batch
        torch.cuda.empty_cache()
    return {"scores": np.concatenate(scores, axis=0), "answer_proj": np.concatenate(answer_proj, axis=0)}


def summarize_pair(patched: dict[str, np.ndarray], baseline: dict[str, np.ndarray], target_idx: int, categories: list[str]) -> dict[str, Any]:
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
        last_layer = len(layers)
        peak_layer = args.peak_layer if args.peak_layer is not None else BOUNDARY_LAYER[args.model]
        categories = list(CATEGORY_OBJECTS.keys())
        test_categories = [x.strip() for x in args.categories.split(",") if x.strip()] or TEST_CATEGORIES
        cat_local_ids, _rows, token_labels = collect_readout_rows(model, tokenizer, categories)
        alloc, reserved = vram_gb()
        log(
            f"{args.model}: peak=L{peak_layer}, true_last=L{last_layer}, "
            f"train/test={args.train_objects}/{args.test_objects}, rank={args.rank}, vram={alloc:.2f}/{reserved:.2f}GB"
        )

        train_items = build_items(categories, args.train_objects, args.test_objects, "train")
        log(f"Capturing train mechanism records: {len(train_items)}")
        train_records = capture_records(
            model, tokenizer, device, layers, train_items, cat_local_ids, categories,
            args.batch_size, args.max_length, last_layer,
        )
        pre_centers = centers_from_records(train_records, categories, "pre_vec")
        ans_centers = centers_from_records(train_records, categories, "answer_vec")

        result: dict[str, Any] = {
            "phase": 138,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "peak_layer": peak_layer,
            "true_last_layer": last_layer,
            "train_objects_per_category": args.train_objects,
            "test_objects_per_category": args.test_objects,
            "test_categories": test_categories,
            "templates": [x["name"] for x in LONG_TEMPLATES],
            "rank": args.rank,
            "ridge": args.ridge,
            "remove_scale": args.remove_scale,
            "restore_scale": args.restore_scale,
            "readout_token_labels": token_labels,
            "category_results": {},
        }

        transfer_cache: dict[str, dict[str, Any]] = {}
        for cat in test_categories:
            target_idx = categories.index(cat)
            pre_basis, pre_sv = svd_basis(build_category_contrast_matrix(pre_centers, categories, cat), args.rank)
            ans_basis, ans_sv = svd_basis(build_category_contrast_matrix(ans_centers, categories, cat), args.rank)
            cat_train = [r for r in train_records if r["cat"] == cat]
            x_train = project_np(np.stack([r["pre_vec"] for r in cat_train]), pre_basis)
            y_train = project_np(np.stack([r["answer_vec"] for r in cat_train]), ans_basis)
            transfer = ridge_map(x_train, y_train, args.ridge)
            transfer_cache[cat] = {
                "pre_basis": pre_basis,
                "ans_basis": ans_basis,
                "transfer": transfer,
                "train_pre_coeff_mean": x_train.mean(axis=0).astype(np.float32),
                "train_answer_coeff_mean": y_train.mean(axis=0).astype(np.float32),
                "pre_sv": pre_sv,
                "ans_sv": ans_sv,
            }

        for ci, cat in enumerate(test_categories, 1):
            log(f"Testing {args.model} {ci}/{len(test_categories)} {cat}")
            target_idx = categories.index(cat)
            prompts = build_long_prompts(cat, args.train_objects, args.test_objects)
            test_records = capture_records(
                model, tokenizer, device, layers, prompts, cat_local_ids, categories,
                args.batch_size, args.max_length, last_layer,
            )
            cache = transfer_cache[cat]
            pre_basis = cache["pre_basis"]
            ans_basis = cache["ans_basis"]
            transfer = cache["transfer"]
            x_test = project_np(np.stack([r["pre_vec"] for r in test_records]), pre_basis)
            y_test = project_np(np.stack([r["answer_vec"] for r in test_records]), ans_basis)
            pred_test = x_test @ transfer

            baseline = {
                "scores": np.stack([r["scores"] for r in test_records]),
                "answer_proj": projection_values(
                    torch.tensor(np.stack([r["answer_vec"] for r in test_records]), device=device),
                    ans_basis,
                ),
            }
            remove = run_transfer_condition(
                model, tokenizer, device, layers, prompts, cat_local_ids, categories,
                args.batch_size, args.max_length, last_layer, "remove",
                pre_basis, ans_basis, transfer, args.remove_scale, args.restore_scale,
            )
            restore = run_transfer_condition(
                model, tokenizer, device, layers, prompts, cat_local_ids, categories,
                args.batch_size, args.max_length, last_layer, "restore",
                pre_basis, ans_basis, transfer, args.remove_scale, args.restore_scale,
            )
            swap_cat = test_categories[(ci % len(test_categories))]
            swap_cache = transfer_cache[swap_cat]
            swap = run_transfer_condition(
                model, tokenizer, device, layers, prompts, cat_local_ids, categories,
                args.batch_size, args.max_length, last_layer, "swap",
                pre_basis,
                swap_cache["ans_basis"],
                transfer,
                args.remove_scale,
                args.restore_scale,
                fixed_answer_coeff=swap_cache["train_answer_coeff_mean"],
            )
            swap_delta = swap["scores"] - baseline["scores"]
            cat_out = {
                "n_prompts": len(prompts),
                "position_audit": position_audit_long(tokenizer, prompts, args.max_length),
                "source_audit": source_audit(tokenizer, prompts, args.max_length),
                "transfer_r2": r2_score(y_test, pred_test),
                "transfer_cosine": cosine_mean(y_test, pred_test),
                "pre_coeff_norm_mean": float(np.linalg.norm(x_test, axis=1).mean()),
                "answer_coeff_norm_mean": float(np.linalg.norm(y_test, axis=1).mean()),
                "pred_answer_coeff_norm_mean": float(np.linalg.norm(pred_test, axis=1).mean()),
                "pre_singular_values": [float(x) for x in cache["pre_sv"]],
                "answer_singular_values": [float(x) for x in cache["ans_sv"]],
                "conditions": {
                    "remove": summarize_pair(remove, baseline, target_idx, categories),
                    "restore": summarize_pair(restore, baseline, target_idx, categories),
                    "swap": {
                        **summarize_pair(swap, baseline, target_idx, categories),
                        "swap_category": swap_cat,
                        "swap_category_delta": float(swap_delta[:, categories.index(swap_cat)].mean()),
                    },
                },
            }
            rem = cat_out["conditions"]["remove"]["target_delta"]
            res = cat_out["conditions"]["restore"]["target_delta"]
            cat_out["restore_recovery_ratio"] = float((res - rem) / (abs(rem) + 1e-8))
            result["category_results"][cat] = cat_out
        return result
    finally:
        release_loaded(loaded)


def _fmt(row: dict[str, Any] | None) -> str:
    if row is None:
        return "NA"
    extra = ""
    if "swap_category" in row:
        extra = f" swap={row['swap_category']} SΔ{row['swap_category_delta']:+.2f}"
    return f"T{row['target_delta']:+.2f} R{row['max_other_delta']:+.2f} A{row['answer_proj_delta']:+.2f}{extra}"


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [f"# Phase 138 Mechanism Transfer Closure: {result['model']}", ""]
    lines.append(f"Generated: {result['timestamp']}")
    lines.append(
        f"Peak layer: L{result['peak_layer']}; true last layer: L{result['true_last_layer']}; "
        f"rank: {result['rank']}; ridge: {result['ridge']}"
    )
    lines.append("")
    lines.append("| category | audit | transfer | remove | restore | recovery | swap |")
    lines.append("|---|---|---|---|---|---|---|")
    for cat, item in result["category_results"].items():
        audit = item["position_audit"]
        audit_text = f"old_mismatch={audit['old_answer_pos_mismatch_count']}, mean_pre={audit['mean_pre_len']:.1f}"
        transfer = f"R2={item['transfer_r2']:+.2f}, cos={item['transfer_cosine']:+.2f}"
        cond = item["conditions"]
        lines.append(
            f"| {cat} | {audit_text} | {transfer} | {_fmt(cond['remove'])} | "
            f"{_fmt(cond['restore'])} | {item['restore_recovery_ratio']:+.2f} | {_fmt(cond['swap'])} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--train-objects", type=int, default=8)
    parser.add_argument("--test-objects", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--peak-layer", type=int, default=None)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--remove-scale", type=float, default=1.0)
    parser.add_argument("--restore-scale", type=float, default=1.0)
    parser.add_argument("--categories", default="")
    parser.add_argument("--output-dir", default=str(OUT_ROOT))
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = run_model(args)
    json_path = out_dir / f"phase138_{args.model}_mechanism_transfer_closure.json"
    md_path = out_dir / f"phase138_{args.model}_mechanism_transfer_closure.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(result, md_path)
    log(f"Wrote {json_path}")
    log(f"Wrote {md_path}")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
