#!/usr/bin/env python3
"""
Phase 509: rotation-stable orthogonal field factor audit.

Phase508 showed Phi_perp contains causal directions. This phase tests the
main weakness: whether support/release/format effects are stable properties of
the subspace or artifacts of one SVD basis.

For each model we focus on the categories that Phase508 made most diagnostic:
  qwen3: fruit, action, emotion
  glm4: emotion, color, fruit
  deepseek7b: action, fruit, color

The audit compares:
  - SVD basis components
  - random rotations inside the same Phi_perp subspace
  - causal candidate axes inside the same subspace
  - random outside-subspace controls
  - small surface-token probes
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
)
from phase508_orthogonal_field_basis_decomposition import (  # noqa: E402
    NEUTRAL_TEMPLATES,
    RICH_TEMPLATES,
    batched_hidden,
    batched_logits_with_delta,
    build_cat_meta,
    build_examples,
    cos,
    delta_summary,
    label_effect,
    max_abs_basis_cos,
    orthonormal_rows,
    project_remove_deltas,
    random_basis,
    score_logits,
    summarize_scores,
    svd_basis,
)


OUT_ROOT = Path("results/glm5_phase509_rotation_stable_orthogonal_field")
FOCUS_CATEGORIES = {
    "qwen3": ["fruit", "action", "emotion"],
    "glm4": ["emotion", "color", "fruit"],
    "deepseek7b": ["action", "fruit", "color"],
}
SURFACE_TOKEN_GROUPS = {
    "punctuation": [".", ",", ":", ";"],
    "generic": [" thing", " item", " type", " kind"],
    "category": [],  # filled per category
}


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def random_rotation(rank: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mat = rng.standard_normal((rank, rank)).astype(np.float32)
    q, _ = np.linalg.qr(mat)
    return q.astype(np.float32)


def outside_random_basis(dim: int, rank: int, subspace: np.ndarray, seed: int) -> np.ndarray:
    raw = random_basis(dim, rank + subspace.shape[0], seed)
    rows = []
    for vec in raw:
        v = vec.copy()
        if subspace.size:
            v = v - (v @ subspace.T) @ subspace
        n = np.linalg.norm(v)
        if n > 1e-7:
            rows.append((v / n).astype(np.float32))
        if len(rows) >= rank:
            break
    if len(rows) < rank:
        return random_basis(dim, rank, seed + 999)
    return np.stack(rows).astype(np.float32)


def make_candidate_axes(basis: np.ndarray, seed: int, n_random: int) -> list[dict[str, Any]]:
    axes: list[dict[str, Any]] = []
    for i, v in enumerate(basis):
        axes.append({"name": f"svd{i}", "vec": v.astype(np.float32)})
        axes.append({"name": f"neg_svd{i}", "vec": (-v).astype(np.float32)})
    rng = np.random.default_rng(seed)
    for i in range(n_random):
        coeff = rng.standard_normal(basis.shape[0]).astype(np.float32)
        coeff /= np.linalg.norm(coeff) + 1e-8
        v = coeff @ basis
        v /= np.linalg.norm(v) + 1e-8
        axes.append({"name": f"combo{i}", "vec": v.astype(np.float32), "coeff": coeff.tolist()})
    return axes


def token_ids(tokenizer: Any, words: list[str]) -> list[int]:
    ids: list[int] = []
    for w in words:
        toks = tokenizer.encode(w, add_special_tokens=False)
        if toks:
            ids.append(int(toks[0]))
    return sorted(set(ids))


def surface_scores(logits: np.ndarray, tokenizer: Any, cat: str) -> dict[str, np.ndarray]:
    groups = {
        "punctuation": token_ids(tokenizer, SURFACE_TOKEN_GROUPS["punctuation"]),
        "generic": token_ids(tokenizer, SURFACE_TOKEN_GROUPS["generic"]),
        "category": token_ids(tokenizer, [cat, " " + cat]),
    }
    out: dict[str, np.ndarray] = {}
    for name, ids in groups.items():
        ids = [i for i in ids if 0 <= i < logits.shape[1]]
        out[name] = logits[:, ids].max(axis=1) if ids else np.zeros(logits.shape[0], dtype=np.float32)
    return out


def summarize_surface_delta(patched_logits: np.ndarray, baseline_logits: np.ndarray, tokenizer: Any, cat: str) -> dict[str, float]:
    p = surface_scores(patched_logits, tokenizer, cat)
    b = surface_scores(baseline_logits, tokenizer, cat)
    return {f"surface_delta_{k}": round(float(np.mean(p[k] - b[k])), 6) for k in p}


def evaluate_basis_rows(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    prompts: list[str],
    layer_id: int,
    phi_perp: np.ndarray,
    rows: np.ndarray,
    baseline: dict[str, np.ndarray],
    baseline_logits: np.ndarray,
    target_ids: list[int],
    competitor_ids: list[int],
    batch_size: int,
    max_length: int,
    scale: float,
    cat: str,
    prefix: str,
) -> list[dict[str, Any]]:
    out_rows = []
    for i, vec in enumerate(rows):
        deltas = project_remove_deltas(phi_perp, vec[None, :], scale)
        logits = batched_logits_with_delta(
            model, tokenizer, device, layers, prompts, layer_id, deltas, batch_size, max_length
        )
        patched = score_logits(logits, target_ids, competitor_ids)
        row = {
            "name": f"{prefix}{i}",
            "basis_index": i,
            **delta_summary(patched, baseline),
            **summarize_surface_delta(logits, baseline_logits, tokenizer, cat),
        }
        row["label"] = label_effect(row)
        out_rows.append(row)
    return out_rows


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    model, tokenizer, device = load_bf16_auto(args.model)
    try:
        info = get_model_info(model, args.model)
        layers = get_layers(model)
        L, d = info.n_layers, info.d_model
        hidden_layers = sorted(set([max(1, min(L, int(x))) for x in [L // 2, 3 * L // 4, L - 3]]))
        W_U = get_W_U(model, args.model).astype(np.float32)
        g = get_norm_g(model, args.model)
        if g is None:
            raise RuntimeError("cannot read final norm gain")
        cat_meta = build_cat_meta(tokenizer, W_U, g.astype(np.float32), d)
        categories = args.categories.split(",") if args.categories else FOCUS_CATEGORIES[args.model]
        log(f"{args.model}: L={L}, d={d}, categories={categories}, layers={hidden_layers}")

        result = {
            "phase": 509,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "L": L,
            "d_model": d,
            "categories": categories,
            "train_objects": args.train_objects,
            "test_objects": args.test_objects,
            "templates": [x[0] for x in RICH_TEMPLATES],
            "layers": hidden_layers,
            "rank": args.rank,
            "candidate_random_axes": args.candidate_random_axes,
            "scale": args.scale,
            "category_results": {},
        }

        for ci, cat in enumerate(categories, 1):
            log(f"{args.model}: category {ci}/{len(categories)} {cat}")
            train_ex, test_ex = build_examples(cat, args.train_objects, args.test_objects)
            rich_train = [x["rich"] for x in train_ex]
            neutral_train = [x["neutral"] for x in train_ex]
            rich_test = [x["rich"] for x in test_ex]
            neutral_test = [x["neutral"] for x in test_ex]
            q_hat = cat_meta[cat]["q_hat"]
            target_ids = cat_meta[cat]["target_ids"]
            competitor_ids = cat_meta[cat]["competitor_ids"]

            train_r = batched_hidden(model, tokenizer, device, rich_train, hidden_layers, args.batch_size, args.max_length)
            train_n = batched_hidden(model, tokenizer, device, neutral_train, hidden_layers, args.batch_size, args.max_length)
            test_r = batched_hidden(model, tokenizer, device, rich_test, hidden_layers, args.batch_size, args.max_length)
            test_n = batched_hidden(model, tokenizer, device, neutral_test, hidden_layers, args.batch_size, args.max_length)

            cat_out = {"n_train_prompts": len(train_ex), "n_test_prompts": len(test_ex), "layers": {}}
            for layer_id in hidden_layers:
                phi_train = train_r[layer_id] - train_n[layer_id]
                phi_test = test_r[layer_id] - test_n[layer_id]
                para_train = (phi_train @ q_hat)[:, None] * q_hat[None, :]
                para_test = (phi_test @ q_hat)[:, None] * q_hat[None, :]
                perp_train = (phi_train - para_train).astype(np.float32)
                perp_test = (phi_test - para_test).astype(np.float32)

                basis, singular_values, var_ratio = svd_basis(perp_train, args.rank)
                rot = random_rotation(basis.shape[0], 50900 + ci * 101 + layer_id)
                rotated = (rot @ basis).astype(np.float32)
                outside = outside_random_basis(d, basis.shape[0], basis, 50950 + ci * 103 + layer_id)

                tpl_rows = []
                for tid in range(len(RICH_TEMPLATES)):
                    mask = np.array([x["template_id"] == tid for x in train_ex])
                    tpl_rows.append(perp_train[mask].mean(axis=0))
                tpl_basis = orthonormal_rows(np.stack(tpl_rows) - np.mean(tpl_rows, axis=0), max_rank=3)

                baseline_logits = batched_logits_with_delta(
                    model, tokenizer, device, layers, rich_test, layer_id, None, args.batch_size, args.max_length
                )
                baseline = score_logits(baseline_logits, target_ids, competitor_ids)
                base_surface = surface_scores(baseline_logits, tokenizer, cat)

                svd_rows = evaluate_basis_rows(
                    model, tokenizer, device, layers, rich_test, layer_id, perp_test, basis,
                    baseline, baseline_logits, target_ids, competitor_ids,
                    args.batch_size, args.max_length, args.scale, cat, "svd"
                )
                for i, row in enumerate(svd_rows):
                    row["singular_value"] = singular_values[i]
                    row["var_ratio"] = var_ratio[i]
                    row["readout_cos"] = round(cos(basis[i], q_hat), 8)
                    row["format_abs_cos"] = round(max_abs_basis_cos(basis[i], tpl_basis), 6)

                rotated_rows = evaluate_basis_rows(
                    model, tokenizer, device, layers, rich_test, layer_id, perp_test, rotated,
                    baseline, baseline_logits, target_ids, competitor_ids,
                    args.batch_size, args.max_length, args.scale, cat, "rot"
                )
                for i, row in enumerate(rotated_rows):
                    row["readout_cos"] = round(cos(rotated[i], q_hat), 8)
                    row["format_abs_cos"] = round(max_abs_basis_cos(rotated[i], tpl_basis), 6)

                outside_rows = evaluate_basis_rows(
                    model, tokenizer, device, layers, rich_test, layer_id, perp_test, outside,
                    baseline, baseline_logits, target_ids, competitor_ids,
                    args.batch_size, args.max_length, args.scale, cat, "outside"
                )

                # Causal candidate search inside the same subspace. This is a
                # deliberately small candidate pool: it tests whether the
                # subspace contains recoverable support/release axes without
                # exploding GPU work.
                candidate_axes = make_candidate_axes(
                    basis, 50980 + ci * 107 + layer_id, args.candidate_random_axes
                )
                candidate_rows = []
                for cand in candidate_axes:
                    vec = cand["vec"]
                    deltas = project_remove_deltas(perp_test, vec[None, :], args.scale)
                    logits = batched_logits_with_delta(
                        model, tokenizer, device, layers, rich_test, layer_id, deltas,
                        args.batch_size, args.max_length
                    )
                    patched = score_logits(logits, target_ids, competitor_ids)
                    row = {
                        "name": cand["name"],
                        "readout_cos": round(cos(vec, q_hat), 8),
                        "format_abs_cos": round(max_abs_basis_cos(vec, tpl_basis), 6),
                        **delta_summary(patched, baseline),
                        **summarize_surface_delta(logits, baseline_logits, tokenizer, cat),
                    }
                    row["label"] = label_effect(row)
                    candidate_rows.append(row)

                best_svd = min(svd_rows, key=lambda r: r["delta_D"])
                best_rot = min(rotated_rows, key=lambda r: r["delta_D"])
                best_out = min(outside_rows, key=lambda r: r["delta_D"])
                best_causal = min(candidate_rows, key=lambda r: r["delta_D"])
                pos_causal = max(candidate_rows, key=lambda r: r["delta_D"])
                layer_out = {
                    "baseline": summarize_scores(baseline),
                    "baseline_surface": {k: round(float(np.mean(v)), 6) for k, v in base_surface.items()},
                    "perp_para_ratio": round(float(
                        np.linalg.norm(perp_train, axis=1).mean() /
                        (np.linalg.norm(para_train, axis=1).mean() + 1e-8)
                    ), 6),
                    "singular_values": singular_values,
                    "variance_ratio": var_ratio,
                    "svd_components": svd_rows,
                    "rotated_components": rotated_rows,
                    "outside_random_components": outside_rows,
                    "causal_candidates": candidate_rows,
                    "best_summary": {
                        "svd_best_delta_D": best_svd["delta_D"],
                        "rotated_best_delta_D": best_rot["delta_D"],
                        "outside_best_delta_D": best_out["delta_D"],
                        "causal_best_delta_D": best_causal["delta_D"],
                        "causal_positive_delta_D": pos_causal["delta_D"],
                        "svd_best_label": best_svd["label"],
                        "rotated_best_label": best_rot["label"],
                        "causal_best_name": best_causal["name"],
                        "causal_positive_name": pos_causal["name"],
                    },
                }
                cat_out["layers"][str(layer_id)] = layer_out
                log(
                    f"  {cat} L{layer_id}: "
                    f"svd={best_svd['delta_D']:+.3f} rot={best_rot['delta_D']:+.3f} "
                    f"causal={best_causal['delta_D']:+.3f} pos={pos_causal['delta_D']:+.3f} "
                    f"outside={best_out['delta_D']:+.3f}"
                )

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
    parser.add_argument("--candidate-random-axes", type=int, default=4)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--categories", default="")
    parser.add_argument("--output-dir", default=str(OUT_ROOT))
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = run_model(args)
    path = out_dir / f"phase509_{args.model}_rotation_stable_orthogonal_field.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {path}")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
