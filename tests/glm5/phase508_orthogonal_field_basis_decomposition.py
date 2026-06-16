#!/usr/bin/env python3
"""
Phase 508: Orthogonal field causal basis decomposition.

Goal:
  Split the large readout-orthogonal semantic field Phi_perp into small
  causal directions and test whether component removal behaves like support,
  suppressor/interface, format/task, or weak structure.

The script keeps Phase507's broad data scope:
  7 categories x 30 objects/category x 3 templates

It uses train objects/templates to build Phi_perp bases, then tests held-out
objects/templates with batched CUDA interventions.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_utils import get_W_U, get_layers, get_model_info, release_model  # noqa: E402
from phase507_orthogonal_field import (  # noqa: E402
    ALL_CLASS,
    CATEGORIES,
    get_norm_g,
    get_token_ids,
    load_bf16_auto,
    rms_norm,
)


OUT_ROOT = Path("results/glm5_phase508_orthogonal_field_basis")
RICH_TEMPLATES = [
    ("plain", "The {obj} {relation}"),
    ("taxonomy", "In taxonomy, {obj} {relation}"),
    ("classify", "Classify {obj}: it {relation}"),
]
NEUTRAL_TEMPLATES = [
    ("plain", "The {obj} is a thing"),
    ("taxonomy", "In taxonomy, {obj} is an item"),
    ("classify", "Describe {obj}: it is a thing"),
]


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_examples(cat: str, train_n: int, test_n: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cfg = CATEGORIES[cat]
    train_objs = cfg["objects"][:train_n]
    test_objs = cfg["objects"][train_n: train_n + test_n]
    train, test = [], []
    for split, objs, out in [("train", train_objs, train), ("test", test_objs, test)]:
        for oi, obj in enumerate(objs):
            for ti, ((r_name, r_tpl), (n_name, n_tpl)) in enumerate(zip(RICH_TEMPLATES, NEUTRAL_TEMPLATES)):
                out.append({
                    "split": split,
                    "cat": cat,
                    "obj": obj,
                    "object_index": oi,
                    "template_id": ti,
                    "template_name": r_name,
                    "rich": r_tpl.format(obj=obj, relation=cfg["relation"]),
                    "neutral": n_tpl.format(obj=obj),
                })
    return train, test


def batched_hidden(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    prompts: list[str],
    hidden_layers: list[int],
    batch_size: int,
    max_length: int,
) -> dict[int, np.ndarray]:
    rows = {int(l): [] for l in hidden_layers}
    for start in range(0, len(prompts), batch_size):
        texts = prompts[start:start + batch_size]
        batch = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
        pos = batch["attention_mask"].sum(dim=1) - 1
        with torch.no_grad():
            out = model(**batch, output_hidden_states=True, return_dict=True, use_cache=False)
        for layer_id in hidden_layers:
            hs = out.hidden_states[layer_id]
            take = hs[torch.arange(hs.shape[0], device=hs.device), pos.to(hs.device)]
            rows[layer_id].append(take.float().cpu().numpy().astype(np.float32))
        del out, batch
        torch.cuda.empty_cache()
    return {l: np.concatenate(v, axis=0) for l, v in rows.items()}


def batched_logits_with_delta(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    prompts: list[str],
    layer_id: int,
    deltas: np.ndarray | None,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    outs = []
    module_index = max(0, min(layer_id - 1, len(layers) - 1))
    for start in range(0, len(prompts), batch_size):
        texts = prompts[start:start + batch_size]
        batch = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
        pos = batch["attention_mask"].sum(dim=1) - 1
        handle = None
        if deltas is not None:
            delta_np = deltas[start:start + len(texts)]
            delta_t = torch.tensor(delta_np, device=device, dtype=torch.bfloat16)
            pos_t = pos.to(device)

            def hook(_module, _inp, output):
                if isinstance(output, tuple):
                    hs = output[0].clone()
                    hs[torch.arange(hs.shape[0], device=hs.device), pos_t.to(hs.device)] += delta_t.to(hs.device, hs.dtype)
                    return (hs,) + output[1:]
                hs = output.clone()
                hs[torch.arange(hs.shape[0], device=hs.device), pos_t.to(hs.device)] += delta_t.to(hs.device, hs.dtype)
                return hs

            handle = layers[module_index].register_forward_hook(hook)
        with torch.no_grad():
            out = model(**batch, return_dict=True, use_cache=False)
        if handle is not None:
            handle.remove()
        logits = out.logits[torch.arange(out.logits.shape[0], device=out.logits.device), pos.to(out.logits.device)]
        outs.append(logits.float().cpu().numpy().astype(np.float32))
        del out, batch
        torch.cuda.empty_cache()
    return np.concatenate(outs, axis=0)


def orthonormal_rows(mat: np.ndarray, max_rank: int) -> np.ndarray:
    x = mat.astype(np.float32)
    x = x[np.linalg.norm(x, axis=1) > 1e-7]
    if x.shape[0] == 0:
        return np.zeros((0, mat.shape[1]), dtype=np.float32)
    q, _ = np.linalg.qr(x.T)
    k = min(max_rank, q.shape[1])
    out = q[:, :k].T.astype(np.float32)
    out /= np.linalg.norm(out, axis=1, keepdims=True) + 1e-8
    return out


def svd_basis(mat: np.ndarray, rank: int) -> tuple[np.ndarray, list[float], list[float]]:
    x = mat.astype(np.float32)
    x = x - x.mean(axis=0, keepdims=True)
    _u, s, vt = np.linalg.svd(x, full_matrices=False)
    k = min(rank, vt.shape[0])
    b = vt[:k].astype(np.float32)
    b /= np.linalg.norm(b, axis=1, keepdims=True) + 1e-8
    total = float(np.sum(s ** 2) + 1e-10)
    var = [float(v) for v in (s[:k] ** 2 / total)]
    return b, [float(v) for v in s[:k]], var


def random_basis(dim: int, rank: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mat = rng.standard_normal((rank, dim)).astype(np.float32)
    q, _ = np.linalg.qr(mat.T)
    out = q[:, :rank].T.astype(np.float32)
    out /= np.linalg.norm(out, axis=1, keepdims=True) + 1e-8
    return out


def project_remove_deltas(phi_perp: np.ndarray, basis: np.ndarray, scale: float) -> np.ndarray:
    coeff = phi_perp @ basis.T
    recon = coeff @ basis
    return (-scale * recon).astype(np.float32)


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-8) * (np.linalg.norm(b) + 1e-8)))


def max_abs_basis_cos(vec: np.ndarray, basis: np.ndarray) -> float:
    if basis.size == 0:
        return 0.0
    v = vec / (np.linalg.norm(vec) + 1e-8)
    return float(np.max(np.abs(basis @ v)))


def score_logits(logits: np.ndarray, target_ids: list[int], competitor_ids: list[int]) -> dict[str, Any]:
    valid_t = [i for i in target_ids if 0 <= i < logits.shape[1]]
    valid_c = [i for i in competitor_ids if 0 <= i < logits.shape[1]]
    target = logits[:, valid_t].mean(axis=1) if valid_t else np.zeros(logits.shape[0], dtype=np.float32)
    comp = logits[:, valid_c].mean(axis=1) if valid_c else np.zeros(logits.shape[0], dtype=np.float32)
    shifted = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(shifted)
    probs /= probs.sum(axis=1, keepdims=True)
    target_prob = probs[:, valid_t].sum(axis=1) if valid_t else np.zeros(logits.shape[0], dtype=np.float32)
    argmax = np.argmax(logits, axis=1)
    return {
        "D": target - comp,
        "T": target,
        "C": comp,
        "target_prob": target_prob,
        "target_argmax": np.array([1.0 if int(x) in valid_t else 0.0 for x in argmax], dtype=np.float32),
    }


def summarize_scores(scores: dict[str, np.ndarray]) -> dict[str, float]:
    return {k: round(float(np.mean(v)), 6) for k, v in scores.items()}


def delta_summary(patched: dict[str, np.ndarray], baseline: dict[str, np.ndarray]) -> dict[str, float]:
    return {
        "delta_D": round(float(np.mean(patched["D"] - baseline["D"])), 6),
        "delta_T": round(float(np.mean(patched["T"] - baseline["T"])), 6),
        "delta_C": round(float(np.mean(patched["C"] - baseline["C"])), 6),
        "delta_target_prob": round(float(np.mean(patched["target_prob"] - baseline["target_prob"])), 8),
        "delta_argmax_rate": round(float(np.mean(patched["target_argmax"] - baseline["target_argmax"])), 6),
    }


def label_effect(row: dict[str, float], threshold: float = 0.25) -> str:
    d = row["delta_D"]
    dt = row["delta_T"]
    dc = row["delta_C"]
    if d <= -threshold:
        return "support"
    if d >= threshold and dc <= -threshold:
        return "competitor_suppressor"
    if d >= threshold and dt >= threshold:
        return "target_release"
    if d >= threshold:
        return "suppressor_or_interface"
    return "weak"


def build_cat_meta(tokenizer: Any, W_U: np.ndarray, g: np.ndarray, d: int) -> dict[str, Any]:
    meta = {}
    for cat in CATEGORIES:
        target_ids = get_token_ids(tokenizer, [cat])
        other_cats = [c for c in ALL_CLASS if c != cat]
        competitor_ids = get_token_ids(tokenizer, other_cats)
        w_t = np.mean([W_U[i] for i in target_ids if i < len(W_U)], axis=0) if target_ids else np.zeros(d)
        w_c = np.mean([W_U[i] for i in competitor_ids if i < len(W_U)], axis=0) if competitor_ids else np.zeros(d)
        qc = ((w_t - w_c) * g).astype(np.float32)
        meta[cat] = {
            "target_ids": target_ids,
            "competitor_ids": competitor_ids,
            "qc": qc,
            "q_hat": qc / (np.linalg.norm(qc) + 1e-8),
            "q_norm": float(np.linalg.norm(qc)),
        }
    return meta


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    model, tokenizer, device = load_bf16_auto(args.model)
    try:
        info = get_model_info(model, args.model)
        layers = get_layers(model)
        L = info.n_layers
        d = info.d_model
        hidden_layers = [max(1, min(L, int(x))) for x in [L // 2, 3 * L // 4, L - 3]]
        hidden_layers = sorted(set(hidden_layers))
        W_U = get_W_U(model, args.model).astype(np.float32)
        g = get_norm_g(model, args.model)
        if g is None:
            raise RuntimeError("cannot read final norm gain")
        g = g.astype(np.float32)
        cat_meta = build_cat_meta(tokenizer, W_U, g, d)
        log(f"{args.model}: L={L}, d={d}, hidden_layers={hidden_layers}, rank={args.rank}, scale={args.scale}")

        result = {
            "phase": 508,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "L": L,
            "d_model": d,
            "categories": list(CATEGORIES.keys()),
            "train_objects": args.train_objects,
            "test_objects": args.test_objects,
            "templates": [x[0] for x in RICH_TEMPLATES],
            "hidden_layers": hidden_layers,
            "rank": args.rank,
            "scale": args.scale,
            "category_results": {},
        }

        for ci, cat in enumerate(CATEGORIES, 1):
            log(f"{args.model}: category {ci}/{len(CATEGORIES)} {cat}")
            train_ex, test_ex = build_examples(cat, args.train_objects, args.test_objects)
            rich_train = [x["rich"] for x in train_ex]
            neutral_train = [x["neutral"] for x in train_ex]
            rich_test = [x["rich"] for x in test_ex]
            neutral_test = [x["neutral"] for x in test_ex]

            train_r = batched_hidden(model, tokenizer, device, rich_train, hidden_layers, args.batch_size, args.max_length)
            train_n = batched_hidden(model, tokenizer, device, neutral_train, hidden_layers, args.batch_size, args.max_length)
            test_r = batched_hidden(model, tokenizer, device, rich_test, hidden_layers, args.batch_size, args.max_length)
            test_n = batched_hidden(model, tokenizer, device, neutral_test, hidden_layers, args.batch_size, args.max_length)

            cat_out = {
                "n_train_prompts": len(train_ex),
                "n_test_prompts": len(test_ex),
                "layers": {},
            }
            q_hat = cat_meta[cat]["q_hat"]
            target_ids = cat_meta[cat]["target_ids"]
            competitor_ids = cat_meta[cat]["competitor_ids"]

            for layer_id in hidden_layers:
                phi_train = train_r[layer_id] - train_n[layer_id]
                phi_test = test_r[layer_id] - test_n[layer_id]
                para_train = (phi_train @ q_hat)[:, None] * q_hat[None, :]
                para_test = (phi_test @ q_hat)[:, None] * q_hat[None, :]
                perp_train = (phi_train - para_train).astype(np.float32)
                perp_test = (phi_test - para_test).astype(np.float32)

                basis, singular_values, var_ratio = svd_basis(perp_train, args.rank)
                rbasis = random_basis(d, basis.shape[0], 50800 + ci * 97 + layer_id)

                # Format basis: template means in Phi_perp space.
                tpl_rows = []
                for tid in range(len(RICH_TEMPLATES)):
                    mask = np.array([x["template_id"] == tid for x in train_ex])
                    tpl_rows.append(perp_train[mask].mean(axis=0))
                tpl_basis = orthonormal_rows(np.stack(tpl_rows) - np.mean(tpl_rows, axis=0), max_rank=3)

                mean_perp = perp_train.mean(axis=0)
                baseline_logits = batched_logits_with_delta(
                    model, tokenizer, device, layers, rich_test, layer_id, None,
                    args.batch_size, args.max_length
                )
                baseline = score_logits(baseline_logits, target_ids, competitor_ids)

                layer_out = {
                    "baseline": summarize_scores(baseline),
                    "perp_norm_mean": round(float(np.linalg.norm(perp_train, axis=1).mean()), 6),
                    "para_norm_mean": round(float(np.linalg.norm(para_train, axis=1).mean()), 6),
                    "perp_para_ratio": round(float(
                        np.linalg.norm(perp_train, axis=1).mean() /
                        (np.linalg.norm(para_train, axis=1).mean() + 1e-8)
                    ), 6),
                    "singular_values": singular_values,
                    "variance_ratio": var_ratio,
                    "components": [],
                    "random_components": [],
                    "sets": [],
                }

                for bi, vec in enumerate(basis):
                    deltas = project_remove_deltas(perp_test, vec[None, :], args.scale)
                    logits = batched_logits_with_delta(
                        model, tokenizer, device, layers, rich_test, layer_id, deltas,
                        args.batch_size, args.max_length
                    )
                    patched = score_logits(logits, target_ids, competitor_ids)
                    row = {
                        "basis_index": bi,
                        "singular_value": singular_values[bi],
                        "var_ratio": var_ratio[bi],
                        "readout_cos": round(cos(vec, q_hat), 8),
                        "mean_perp_cos": round(cos(vec, mean_perp), 6),
                        "format_abs_cos": round(max_abs_basis_cos(vec, tpl_basis), 6),
                        **delta_summary(patched, baseline),
                    }
                    row["label"] = label_effect(row)
                    layer_out["components"].append(row)

                for bi, vec in enumerate(rbasis):
                    deltas = project_remove_deltas(perp_test, vec[None, :], args.scale)
                    logits = batched_logits_with_delta(
                        model, tokenizer, device, layers, rich_test, layer_id, deltas,
                        args.batch_size, args.max_length
                    )
                    patched = score_logits(logits, target_ids, competitor_ids)
                    row = {
                        "basis_index": bi,
                        "readout_cos": round(cos(vec, q_hat), 8),
                        "mean_perp_cos": round(cos(vec, mean_perp), 6),
                        "format_abs_cos": round(max_abs_basis_cos(vec, tpl_basis), 6),
                        **delta_summary(patched, baseline),
                    }
                    row["label"] = label_effect(row)
                    layer_out["random_components"].append(row)

                sorted_support = sorted(layer_out["components"], key=lambda r: r["delta_D"])
                sorted_suppress = sorted(layer_out["components"], key=lambda r: r["delta_D"], reverse=True)
                set_specs = [
                    ("support_top1", sorted_support[:1]),
                    ("support_top2", sorted_support[:2]),
                    ("support_top4", sorted_support[:4]),
                    ("suppressor_top1", sorted_suppress[:1]),
                    ("suppressor_top2", sorted_suppress[:2]),
                    ("format_aligned_top2", sorted(layer_out["components"], key=lambda r: r["format_abs_cos"], reverse=True)[:2]),
                ]
                for name, rows in set_specs:
                    ids = [int(r["basis_index"]) for r in rows]
                    if not ids:
                        continue
                    deltas = project_remove_deltas(perp_test, basis[ids], args.scale)
                    logits = batched_logits_with_delta(
                        model, tokenizer, device, layers, rich_test, layer_id, deltas,
                        args.batch_size, args.max_length
                    )
                    patched = score_logits(logits, target_ids, competitor_ids)
                    layer_out["sets"].append({
                        "set_name": name,
                        "basis_indices": ids,
                        "set_size": len(ids),
                        **delta_summary(patched, baseline),
                    })

                comp_best = min(layer_out["components"], key=lambda r: r["delta_D"])
                comp_pos = max(layer_out["components"], key=lambda r: r["delta_D"])
                rand_best = min(layer_out["random_components"], key=lambda r: r["delta_D"])
                log(
                    f"  {cat} L{layer_id}: ratio={layer_out['perp_para_ratio']:.1f} "
                    f"best={comp_best['delta_D']:+.3f}/{comp_best['label']} "
                    f"pos={comp_pos['delta_D']:+.3f}/{comp_pos['label']} "
                    f"rand_best={rand_best['delta_D']:+.3f}"
                )
                cat_out["layers"][str(layer_id)] = layer_out

            result["category_results"][cat] = cat_out
            del train_r, train_n, test_r, test_n
            gc.collect()
            torch.cuda.empty_cache()

        return result
    finally:
        release_model(model)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--train-objects", type=int, default=20)
    parser.add_argument("--test-objects", type=int, default=10)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--output-dir", default=str(OUT_ROOT))
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = run_model(args)
    path = out_dir / f"phase508_{args.model}_orthogonal_field_basis.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {path}")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
