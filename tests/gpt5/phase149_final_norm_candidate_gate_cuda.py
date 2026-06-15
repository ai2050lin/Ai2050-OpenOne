#!/usr/bin/env python3
"""
Phase 149: final-norm and candidate-set token gate audit.

Reuse Phase147 train-selected restore paths. For heldout prompts, compare:
  - full-vocab token closure
  - candidate-set category closure
  - final-norm input/output logit lens
  - support restore plus final-norm LM-head steering
  - simple candidate competitor suppression at final-norm output
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hf_probe_env import get_layers, load_probe_model, release_loaded, vram_gb  # noqa: E402
from phase105_global_category_atlas_cuda import CATEGORY_OBJECTS, CATEGORY_READOUT_WORDS, collect_readout_rows, find_token_id  # noqa: E402
from phase107_causal_boundary_removal_cuda import score_logits, summarize_delta  # noqa: E402
from phase114_answer_site_causal_subspace_cuda import build_category_contrast_matrix  # noqa: E402
from phase116_subspace_basis_component_audit_cuda import svd_basis  # noqa: E402
from phase122_pre_answer_to_answer_projection_closure_cuda import projection_values  # noqa: E402
from phase126_residual_gap_decomposition_cuda import tensor_from_output  # noqa: E402
from phase128_final_block_gateway_cuda import get_final_norm  # noqa: E402
from phase135_long_template_source_field_cuda import batch_context  # noqa: E402
from phase138_mechanism_transfer_closure_cuda import normalize_basis, project_np, ridge_map  # noqa: E402
from phase139_restore_swap_calibration_cuda import parse_float_list, parse_str_list  # noqa: E402
from phase145_mechanism_stability_generation_cuda import split_indices  # noqa: E402
from phase146_template_router_token_gap_cuda import capture_records, centers_from_records, get_site_module, target_token_ids  # noqa: E402
from phase147_train_router_format_token_cuda import build_items, clean_baseline, score_row  # noqa: E402
from phase148_router_feature_lmhead_alignment_cuda import lm_head_direction  # noqa: E402


OUT_ROOT = Path("results/gpt5_phase149_final_norm_candidate_gate")
PHASE147_ROOT = Path("results/gpt5_phase147_train_router_format_token")


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def candidate_ids(tokenizer: Any, categories: list[str]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for cat in categories:
        ids: list[int] = []
        for word in CATEGORY_READOUT_WORDS[cat]:
            tid = find_token_id(tokenizer, word)
            if tid is not None:
                ids.append(int(tid))
        out[cat] = sorted(set(ids))
    return out


def output_head(model: Any, hidden: torch.Tensor) -> torch.Tensor:
    emb = model.get_output_embeddings()
    if emb is not None:
        return emb(hidden.to(dtype=emb.weight.dtype))
    if hasattr(model, "lm_head"):
        return model.lm_head(hidden.to(dtype=model.lm_head.weight.dtype))
    raise TypeError("Cannot locate output embedding / lm_head")


def token_metrics(
    logits: torch.Tensor,
    clean_logits: torch.Tensor,
    tokenizer: Any,
    target_ids_: list[int],
    candidate_by_cat: dict[str, list[int]],
    target_cat: str,
) -> dict[str, Any]:
    if not target_ids_:
        return {}
    target_tensor = torch.tensor(target_ids_, device=logits.device, dtype=torch.long)
    target_max = logits[:, target_tensor].float().max(dim=1).values
    clean_target_max = clean_logits[:, target_tensor].float().max(dim=1).values
    full_rank = (logits.float() > target_max[:, None]).sum(dim=1).float() + 1
    argmax = logits.argmax(dim=-1).detach().cpu().numpy().astype(np.int64)
    target_set = set(target_ids_)
    candidate_scores = {}
    for cat, ids in candidate_by_cat.items():
        if not ids:
            continue
        tid = torch.tensor(ids, device=logits.device, dtype=torch.long)
        candidate_scores[cat] = logits[:, tid].float().max(dim=1).values
    target_score = candidate_scores[target_cat]
    other_scores = [v for k, v in candidate_scores.items() if k != target_cat]
    if other_scores:
        other_stack = torch.stack(other_scores, dim=1)
        cand_rank = (other_stack > target_score[:, None]).sum(dim=1).float() + 1
        cand_arg = (cand_rank == 1).float()
        cand_margin = target_score - other_stack.max(dim=1).values
    else:
        cand_rank = torch.ones_like(target_score)
        cand_arg = torch.ones_like(target_score)
        cand_margin = torch.zeros_like(target_score)
    topk = torch.topk(logits.float(), k=min(5, logits.shape[-1]), dim=-1).indices.detach().cpu().numpy()
    decoded = []
    for row in topk[: min(3, topk.shape[0])]:
        decoded.append([{"id": int(t), "text": tokenizer.decode([int(t)])} for t in row.tolist()])
    return {
        "target_token_delta": float((target_max - clean_target_max).mean().detach().cpu()),
        "full_vocab_rank_mean": float(full_rank.mean().detach().cpu()),
        "full_vocab_rank_median": float(full_rank.median().detach().cpu()),
        "full_vocab_argmax_rate": float(np.mean([int(x) in target_set for x in argmax.tolist()])),
        "candidate_rank_mean": float(cand_rank.mean().detach().cpu()),
        "candidate_argmax_rate": float(cand_arg.mean().detach().cpu()),
        "candidate_margin_mean": float(cand_margin.mean().detach().cpu()),
        "top_tokens_examples": decoded,
    }


def merge_metric_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    keys = [
        "target_token_delta",
        "full_vocab_rank_mean",
        "full_vocab_rank_median",
        "full_vocab_argmax_rate",
        "candidate_rank_mean",
        "candidate_argmax_rate",
        "candidate_margin_mean",
    ]
    merged = {k: float(np.mean([r[k] for r in rows if k in r])) for k in keys if k in rows[0]}
    merged["top_tokens_examples"] = rows[0].get("top_tokens_examples", [])
    return merged


def run_gate_condition(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    prompts: list[dict[str, Any]],
    cat_local_ids: dict[str, list[int]],
    all_categories: list[str],
    candidate_by_cat: dict[str, list[int]],
    target_cat: str,
    batch_size: int,
    max_length: int,
    layer_id: int,
    site: str,
    pre_basis: np.ndarray,
    ans_basis: np.ndarray,
    transfer: np.ndarray,
    support_scale: float,
    target_ids_: list[int],
    final_mode: str = "none",
    lm_dir: np.ndarray | None = None,
    lm_scale: float = 0.0,
    suppress_scale: float = 0.0,
    return_logits: bool = False,
) -> dict[str, Any]:
    scores = []
    answer_proj = []
    token_rows = []
    logits_rows = []
    lens_rows: dict[str, list[dict[str, Any]]] = {"final_norm_input": [], "final_norm_output": []}
    layer = layers[layer_id - 1]
    site_module, site_kind = get_site_module(layers, layer_id, site)
    final_norm = get_final_norm(model)
    if final_norm is None:
        raise TypeError("Cannot locate final norm")
    pre_b = torch.tensor(normalize_basis(pre_basis), dtype=torch.float32)
    ans_b = torch.tensor(normalize_basis(ans_basis), dtype=torch.float32)
    w = torch.tensor(transfer, dtype=torch.float32)
    lm_b = None if lm_dir is None else torch.tensor(lm_dir.reshape(1, -1), dtype=torch.float32)

    for start in range(0, len(prompts), batch_size):
        items = prompts[start:start + batch_size]
        batch = tokenizer([x["prompt"] for x in items], return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        batch = {k: v.to(device) for k, v in batch.items()}
        ctx = batch_context(tokenizer, batch, items)
        with torch.no_grad():
            clean_out = model(**batch, use_cache=False)
        pos_gpu = torch.tensor(ctx["last_pos"], device=clean_out.logits.device, dtype=torch.long)
        clean_logits = clean_out.logits[torch.arange(clean_out.logits.shape[0], device=clean_out.logits.device), pos_gpu]
        store: dict[str, torch.Tensor] = {}
        handles = []

        def layer_pre_hook(_module: Any, inputs: tuple[Any, ...]):
            x = inputs[0]
            out = x.clone()
            pb = pre_b.to(out.device)
            coeff_rows = []
            for bi, positions in enumerate(ctx["source_groups"]["all_pre_answer"]):
                pos = torch.tensor(positions, device=out.device, dtype=torch.long)
                vecs = out[bi, pos, :].float()
                proj = (vecs @ pb.T) @ pb
                out[bi, pos, :] = out[bi, pos, :] - proj.to(out.dtype)
                coeff_rows.append(vecs.mean(dim=0) @ pb.T)
            coeff = torch.stack(coeff_rows, dim=0)
            store["support_add"] = (coeff @ w.to(out.device)) @ ans_b.to(out.device)
            return (out,) + inputs[1:]

        handles.append(layer.register_forward_pre_hook(layer_pre_hook))

        def apply_support(x: torch.Tensor) -> torch.Tensor:
            out = x.clone()
            bidx = torch.arange(out.shape[0], device=out.device)
            apos = torch.tensor(ctx["last_pos"], device=out.device, dtype=torch.long)
            out[bidx, apos, :] = out[bidx, apos, :] + support_scale * store["support_add"].to(out.device, dtype=out.dtype)
            return out

        if support_scale != 0.0:
            if site_kind == "pre":
                def site_pre_hook(_module: Any, inputs: tuple[Any, ...]):
                    return (apply_support(inputs[0]),) + inputs[1:]
                handles.append(site_module.register_forward_pre_hook(site_pre_hook))
            else:
                def site_post_hook(_module: Any, _inputs: Any, output: Any):
                    out = apply_support(tensor_from_output(output))
                    if isinstance(output, tuple):
                        return (out,) + output[1:]
                    return out
                handles.append(site_module.register_forward_hook(site_post_hook))

        def final_pre_hook(_module: Any, inputs: tuple[Any, ...]):
            x = inputs[0]
            bidx = torch.arange(x.shape[0], device=x.device)
            apos = torch.tensor(ctx["last_pos"], device=x.device, dtype=torch.long)
            before = x[bidx, apos, :].float()
            lens_logits = output_head(model, before)
            lens_rows["final_norm_input"].append(token_metrics(lens_logits, clean_logits.to(lens_logits.device), tokenizer, target_ids_, candidate_by_cat, target_cat))
            if final_mode == "final_norm_input_lm" and lm_b is not None and lm_scale != 0.0:
                out = x.clone()
                out[bidx, apos, :] = out[bidx, apos, :] + lm_scale * lm_b.to(out.device, dtype=out.dtype)
                return (out,) + inputs[1:]
            return None

        def final_post_hook(_module: Any, _inputs: Any, output: Any):
            x = tensor_from_output(output)
            bidx = torch.arange(x.shape[0], device=x.device)
            apos = torch.tensor(ctx["last_pos"], device=x.device, dtype=torch.long)
            out = x
            if final_mode in {"final_norm_output_lm", "final_norm_output_suppress"}:
                out = x.clone()
                if final_mode == "final_norm_output_lm" and lm_b is not None and lm_scale != 0.0:
                    out[bidx, apos, :] = out[bidx, apos, :] + lm_scale * lm_b.to(out.device, dtype=out.dtype)
                if final_mode == "final_norm_output_suppress" and lm_b is not None and suppress_scale != 0.0:
                    hidden = out[bidx, apos, :].float()
                    logits_now = output_head(model, hidden)
                    other_ids = []
                    target_set = set(target_ids_)
                    for cat, ids in candidate_by_cat.items():
                        if cat == target_cat:
                            continue
                        other_ids.extend([xid for xid in ids if xid not in target_set])
                    if other_ids:
                        oid = torch.tensor(sorted(set(other_ids)), device=logits_now.device, dtype=torch.long)
                        best_other = oid[logits_now[:, oid].argmax(dim=1)]
                        weight = model.get_output_embeddings().weight.detach().to(out.device).float()
                        comp = weight[best_other]
                        comp = comp / (comp.norm(dim=1, keepdim=True) + 1e-8)
                        out[bidx, apos, :] = out[bidx, apos, :] - suppress_scale * comp.to(out.dtype)
            lens_logits = output_head(model, out[bidx, apos, :].float())
            lens_rows["final_norm_output"].append(token_metrics(lens_logits, clean_logits.to(lens_logits.device), tokenizer, target_ids_, candidate_by_cat, target_cat))
            if isinstance(output, tuple):
                return (out,) + output[1:]
            return out

        handles.append(final_norm.register_forward_pre_hook(final_pre_hook))
        handles.append(final_norm.register_forward_hook(final_post_hook))

        with torch.no_grad():
            out = model(**batch, output_hidden_states=True, use_cache=False)
        for h in handles:
            h.remove()
        logits = out.logits[torch.arange(out.logits.shape[0], device=out.logits.device), pos_gpu]
        if return_logits:
            logits_rows.append(logits.detach().float().cpu())
        scores.append(score_logits(logits, cat_local_ids, all_categories))
        hs = out.hidden_states[layer_id]
        ans = hs[torch.arange(hs.shape[0], device=hs.device), pos_gpu.to(hs.device), :].float()
        answer_proj.append(projection_values(ans, ans_basis))
        token_rows.append(token_metrics(logits, clean_logits, tokenizer, target_ids_, candidate_by_cat, target_cat))
        del out, clean_out, batch
        torch.cuda.empty_cache()

    result = {
        "scores": np.concatenate(scores, axis=0),
        "answer_proj": np.concatenate(answer_proj, axis=0),
        "token": merge_metric_rows(token_rows),
        "logit_lens": {k: merge_metric_rows(v) for k, v in lens_rows.items()},
    }
    if return_logits:
        result["logits"] = torch.cat(logits_rows, dim=0)
    return result


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    phase147_path = PHASE147_ROOT / f"phase147_{args.model}_train_router_format_token.json"
    if not phase147_path.exists():
        raise SystemExit(f"Missing Phase147 result: {phase147_path}")
    phase147 = json.loads(phase147_path.read_text(encoding="utf-8"))
    loaded = load_probe_model(args.model)
    try:
        model = loaded.model
        tokenizer = loaded.tokenizer
        device = loaded.input_device
        layers = get_layers(model)
        all_categories = list(CATEGORY_OBJECTS.keys())
        test_categories = parse_str_list(args.categories)
        allowed_formats = set(parse_str_list(args.formats))
        allowed_families = set(parse_str_list(args.template_families))
        allowed_splits = set(parse_str_list(args.splits))
        lm_scales = parse_float_list(args.lm_scales)
        suppress_scales = parse_float_list(args.suppress_scales)
        cat_local_ids, _rows, token_labels = collect_readout_rows(model, tokenizer, all_categories)
        cand = candidate_ids(tokenizer, test_categories)
        alloc, reserved = vram_gb()
        log(f"{args.model}: phase149 cases from Phase147, vram={alloc:.2f}/{reserved:.2f}GB")
        result: dict[str, Any] = {
            "phase": 149,
            "model": args.model,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "categories": test_categories,
            "formats": sorted(allowed_formats),
            "families": sorted(allowed_families),
            "splits": sorted(allowed_splits),
            "lm_scales": lm_scales,
            "suppress_scales": suppress_scales,
            "readout_token_labels": token_labels,
            "candidate_ids": cand,
            "results": {},
        }
        train_tpl = [0, 1]
        heldout_tpl = [2]
        options = phase147["categories"]
        train_cache: dict[tuple[str, str, str, int], dict[str, Any]] = {}
        for key, prev in phase147["results"].items():
            split, family, fmt, cat = key.split(":")
            if cat not in test_categories or fmt not in allowed_formats or family not in allowed_families or split not in allowed_splits:
                continue
            best = prev["train_best"]
            layer_id = int(best["layer_id"])
            site = best["site"]
            support_scale = float(best["scale"])
            train_idx, test_idx = split_indices(split, int(args.train_objects), int(args.test_objects))
            train_all = []
            train_cat = build_items(cat, family, train_tpl, train_idx, fmt, options)
            held_cat = build_items(cat, family, heldout_tpl, test_idx, fmt, options)
            cache_key = (split, family, fmt, layer_id)
            if cache_key not in train_cache:
                train_all = []
                for c in all_categories:
                    train_all.extend(build_items(c, family, train_tpl, train_idx, fmt, options))
                train_records_all = capture_records(model, tokenizer, device, layers, train_all, cat_local_ids, all_categories, args.batch_size, args.max_length, layer_id)
                train_cache[cache_key] = {
                    "records": train_records_all,
                    "pre_centers": centers_from_records(train_records_all, all_categories, "pre_vec", len(train_tpl)),
                    "ans_centers": centers_from_records(train_records_all, all_categories, "answer_vec", len(train_tpl)),
                }
            cached = train_cache[cache_key]
            train_records_all = cached["records"]
            train_records_cat = [r for r in train_records_all if r["cat"] == cat]
            pre_basis, _ = svd_basis(build_category_contrast_matrix(cached["pre_centers"], all_categories, cat), args.rank)
            ans_basis, _ = svd_basis(build_category_contrast_matrix(cached["ans_centers"], all_categories, cat), args.rank)
            x_train = project_np(np.stack([r["pre_vec"] for r in train_records_cat]), pre_basis)
            y_train = project_np(np.stack([r["answer_vec"] for r in train_records_cat]), ans_basis)
            transfer = ridge_map(x_train, y_train, args.ridge)
            held_records = capture_records(model, tokenizer, device, layers, held_cat, cat_local_ids, all_categories, args.batch_size, args.max_length, layer_id)
            clean = clean_baseline(held_records, ans_basis, device)
            target_idx = all_categories.index(cat)
            tids = target_token_ids(tokenizer, cat)
            lm_dir = lm_head_direction(model, tids)
            remove = run_gate_condition(
                model, tokenizer, device, layers, held_cat, cat_local_ids, all_categories, cand, cat,
                args.batch_size, args.max_length, layer_id, site, pre_basis, ans_basis, transfer,
                0.0, tids,
            )
            remove_summary = summarize_delta(remove["scores"] - clean["scores"], target_idx, all_categories)
            rows = []
            base_variants: list[tuple[str, float, float]] = [("support_only", 0.0, 0.0)]
            for s in lm_scales:
                base_variants.append(("final_norm_input_lm", s, 0.0))
                base_variants.append(("final_norm_output_lm", s, 0.0))
            for s in suppress_scales:
                base_variants.append(("final_norm_output_suppress", 0.0, s))
            for mode, lm_scale, suppress_scale in base_variants:
                patched = run_gate_condition(
                    model, tokenizer, device, layers, held_cat, cat_local_ids, all_categories, cand, cat,
                    args.batch_size, args.max_length, layer_id, site, pre_basis, ans_basis, transfer,
                    support_scale, tids, final_mode=("none" if mode == "support_only" else mode),
                    lm_dir=lm_dir, lm_scale=lm_scale, suppress_scale=suppress_scale,
                )
                row = score_row(patched, clean, remove_summary, target_idx, all_categories, args.release_threshold)
                row.update({
                    "variant": mode,
                    "lm_scale": lm_scale,
                    "suppress_scale": suppress_scale,
                    "logit_lens": patched["logit_lens"],
                })
                rows.append(row)
            best_candidate = max(rows, key=lambda r: (r["token"].get("candidate_argmax_rate", 0.0), r["token"].get("candidate_margin_mean", -1e9), -r["max_other_delta"]))
            best_full = min(rows, key=lambda r: (r["token"].get("full_vocab_rank_mean", 1e9), -r["token"].get("full_vocab_argmax_rate", 0.0)))
            result["results"][key] = {
                "path": {"layer_id": layer_id, "site": site, "scale": support_scale},
                "phase147_held_clean": prev["heldout"]["is_constrained_clean"],
                "remove_target_delta": remove_summary["target_delta"],
                "rows": rows,
                "best_candidate": best_candidate,
                "best_full_vocab": best_full,
            }
        return result
    finally:
        release_loaded(loaded)


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [f"# Phase 149 Final-Norm Candidate Gate: {result['model']}", ""]
    lines.append(f"Generated: {result['timestamp']}")
    lines.append("")
    lines.append("| case | path | held clean | best cand variant | cand arg | cand rank | full rank | full arg | lens in/out cand |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for key, item in sorted(result["results"].items()):
        b = item["best_candidate"]
        tok = b.get("token", {})
        lens = b.get("logit_lens", {})
        lin = lens.get("final_norm_input", {}).get("candidate_argmax_rate", 0.0)
        lout = lens.get("final_norm_output", {}).get("candidate_argmax_rate", 0.0)
        p = item["path"]
        lines.append(
            f"| {key} | L{p['layer_id']} {p['site']} s{p['scale']} | {item['phase147_held_clean']} | "
            f"{b['variant']} lm={b['lm_scale']} sup={b['suppress_scale']} | "
            f"{tok.get('candidate_argmax_rate', 0):.2f} | {tok.get('candidate_rank_mean', 0):.2f} | "
            f"{tok.get('full_vocab_rank_mean', 0):.1f} | {tok.get('full_vocab_argmax_rate', 0):.2f} | "
            f"{lin:.2f}/{lout:.2f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--train-objects", type=int, default=8)
    parser.add_argument("--test-objects", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=160)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--release-threshold", type=float, default=0.25)
    parser.add_argument("--categories", default="plant,time,container,number")
    parser.add_argument("--template-families", default="long,short,neutral")
    parser.add_argument("--splits", default="front_back,back_front")
    parser.add_argument("--formats", default="label_colon,multiple_choice,answer_one_word")
    parser.add_argument("--lm-scales", default="0.0,0.5,1.0,2.0,4.0")
    parser.add_argument("--suppress-scales", default="0.5,1.0")
    parser.add_argument("--output-dir", default=str(OUT_ROOT))
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = run_model(args)
    json_path = out_dir / f"phase149_{args.model}_final_norm_candidate_gate.json"
    md_path = out_dir / f"phase149_{args.model}_final_norm_candidate_gate.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(result, md_path)
    log(f"Wrote {json_path}")
    log(f"Wrote {md_path}")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
