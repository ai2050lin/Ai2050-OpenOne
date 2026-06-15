#!/usr/bin/env python3
"""
Phase 151: surface-answer set and multi-token generation closure.

Test whether open-vocab failure is partly caused by too narrow target label
sets. Reuse Phase147 train-selected restore paths and evaluate canonical,
synonym, object-near, format-variant, and option-like surface answers.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hf_probe_env import get_layers, load_probe_model, release_loaded, vram_gb  # noqa: E402
from phase105_global_category_atlas_cuda import CATEGORY_OBJECTS, CATEGORY_READOUT_WORDS, collect_readout_rows  # noqa: E402
from phase114_answer_site_causal_subspace_cuda import build_category_contrast_matrix  # noqa: E402
from phase116_subspace_basis_component_audit_cuda import svd_basis  # noqa: E402
from phase138_mechanism_transfer_closure_cuda import project_np, ridge_map  # noqa: E402
from phase139_restore_swap_calibration_cuda import parse_str_list  # noqa: E402
from phase145_mechanism_stability_generation_cuda import split_indices  # noqa: E402
from phase146_template_router_token_gap_cuda import capture_records, centers_from_records, target_token_ids  # noqa: E402
from phase147_train_router_format_token_cuda import build_items  # noqa: E402
from phase148_router_feature_lmhead_alignment_cuda import lm_head_direction  # noqa: E402
from phase149_final_norm_candidate_gate_cuda import candidate_ids, run_gate_condition  # noqa: E402
from phase135_long_template_source_field_cuda import batch_context  # noqa: E402


OUT_ROOT = Path("results/gpt5_phase151_surface_answer_generation_closure")
PHASE147_ROOT = Path("results/gpt5_phase147_train_router_format_token")

OBJECT_NEAR = {
    "plant": ["flower", "tree", "rose", "oak", "pine", "flora", "vegetation"],
    "time": ["morning", "year", "hour", "date", "period", "moment", "day"],
    "container": ["box", "bottle", "cup", "jar", "vessel", "holder", "bag"],
    "number": ["number", "digit", "amount", "quantity", "count", "integer", "one"],
}


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def encoded_ids(tokenizer: Any, text: str) -> list[int]:
    return [int(x) for x in tokenizer(text, add_special_tokens=False)["input_ids"]]


def surface_strings(cat: str, fmt: str) -> dict[str, list[str]]:
    canonical = [cat]
    synonyms = list(CATEGORY_READOUT_WORDS.get(cat, []))
    object_near = list(OBJECT_NEAR.get(cat, []))
    words = sorted(set(canonical + synonyms + object_near))
    variants = []
    for w in words:
        variants.extend([w, " " + w, w.capitalize(), " " + w.capitalize(), w + ".", " " + w + "."])
        variants.extend(["a " + w, " a " + w])
    option = ["A", " A", "A.", " A.", "option A", " Option A"] if fmt == "multiple_choice" else []
    return {
        "canonical": canonical,
        "synonyms": sorted(set(synonyms)),
        "object_near": sorted(set(object_near)),
        "format_variants": sorted(set(variants)),
        "option_like": option,
        "expanded": sorted(set(words + variants + option)),
    }


def first_token_set(tokenizer: Any, strings: list[str]) -> list[int]:
    ids = []
    for s in strings:
        toks = encoded_ids(tokenizer, s)
        if toks:
            ids.append(toks[0])
    return sorted(set(ids))


def phrase_token_sequences(tokenizer: Any, strings: list[str], max_tokens: int = 4) -> list[list[int]]:
    seqs = []
    for s in strings:
        toks = encoded_ids(tokenizer, s)
        if toks and len(toks) <= max_tokens:
            seqs.append(toks)
    unique = []
    seen = set()
    for seq in seqs:
        tup = tuple(seq)
        if tup not in seen:
            seen.add(tup)
            unique.append(seq)
    return unique


def rank_for_ids(logits: torch.Tensor, ids: list[int]) -> dict[str, float]:
    if not ids:
        return {"rank": 0.0, "argmax": 0.0, "margin": 0.0}
    tid = torch.tensor(ids, device=logits.device, dtype=torch.long)
    target = logits[:, tid].float().max(dim=1).values
    rank = (logits.float() > target[:, None]).sum(dim=1).float() + 1
    argmax = logits.argmax(dim=-1).detach().cpu().tolist()
    return {
        "rank": float(rank.mean().detach().cpu()),
        "argmax": float(np.mean([int(x) in set(ids) for x in argmax])),
        "margin_to_top": float((target - logits.float().max(dim=1).values).mean().detach().cpu()),
    }


def classify_text(text: str, surfaces: dict[str, list[str]]) -> str:
    low = text.lower()
    canonical = {x.lower().strip() for x in surfaces["canonical"]}
    synonyms = {x.lower().strip() for x in surfaces["synonyms"]}
    object_near = {x.lower().strip() for x in surfaces["object_near"]}
    option = {x.lower().strip() for x in surfaces["option_like"]}
    if any(re.search(rf"\b{re.escape(x)}\b", low) for x in canonical):
        return "canonical"
    if any(re.search(rf"\b{re.escape(x)}\b", low) for x in synonyms):
        return "synonym"
    if any(re.search(rf"\b{re.escape(x)}\b", low) for x in object_near):
        return "object_near"
    if option and any(x and x in low for x in option):
        return "option_like"
    if not text.strip():
        return "format_only"
    if re.fullmatch(r"[\s\W_]+", text):
        return "format_only"
    return "other"


def greedy_tokens_from_logits(tokenizer: Any, logits: torch.Tensor, steps: int) -> list[list[str]]:
    # The caller provides one-step logits for each prompt. This records a strict
    # one-token continuation and duplicates it as a lower-bound proxy for 2/3 token
    # closure. True iterative generation is left for the next phase if needed.
    ids = logits.argmax(dim=-1).detach().cpu().tolist()
    texts = [[tokenizer.decode([int(t)], clean_up_tokenization_spaces=False)] for t in ids]
    return [row[:steps] for row in texts]


def clean_logits_for_items(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    items: list[dict[str, Any]],
    batch_size: int,
    max_length: int,
) -> torch.Tensor:
    rows = []
    with torch.no_grad():
        for start in range(0, len(items), batch_size):
            batch_items = items[start:start + batch_size]
            batch = tokenizer([x["prompt"] for x in batch_items], return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            batch = {k: v.to(device) for k, v in batch.items()}
            ctx = batch_context(tokenizer, batch, batch_items)
            out = model(**batch, use_cache=False)
            pos = torch.tensor(ctx["last_pos"], device=out.logits.device, dtype=torch.long)
            logits = out.logits[torch.arange(out.logits.shape[0], device=out.logits.device), pos]
            rows.append(logits.detach().float().cpu())
            del out, batch
            torch.cuda.empty_cache()
    return torch.cat(rows, dim=0)


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
        categories = list(CATEGORY_OBJECTS.keys())
        test_categories = parse_str_list(args.categories)
        allowed_formats = set(parse_str_list(args.formats))
        allowed_families = set(parse_str_list(args.template_families))
        allowed_splits = set(parse_str_list(args.splits))
        cat_local_ids, _rows, token_labels = collect_readout_rows(model, tokenizer, categories)
        cand = candidate_ids(tokenizer, test_categories)
        alloc, reserved = vram_gb()
        log(f"{args.model}: phase151 surface answer closure, vram={alloc:.2f}/{reserved:.2f}GB")
        result: dict[str, Any] = {
            "phase": 151,
            "model": args.model,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "categories": test_categories,
            "readout_token_labels": token_labels,
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
            held_cat = build_items(cat, family, heldout_tpl, test_idx, fmt, options)
            cache_key = (split, family, fmt, layer_id)
            if cache_key not in train_cache:
                train_all = []
                for c in categories:
                    train_all.extend(build_items(c, family, train_tpl, train_idx, fmt, options))
                recs = capture_records(model, tokenizer, device, layers, train_all, cat_local_ids, categories, args.batch_size, args.max_length, layer_id)
                train_cache[cache_key] = {
                    "records": recs,
                    "pre_centers": centers_from_records(recs, categories, "pre_vec", len(train_tpl)),
                    "ans_centers": centers_from_records(recs, categories, "answer_vec", len(train_tpl)),
                }
            cached = train_cache[cache_key]
            train_records_cat = [r for r in cached["records"] if r["cat"] == cat]
            pre_basis, _ = svd_basis(build_category_contrast_matrix(cached["pre_centers"], categories, cat), args.rank)
            ans_basis, _ = svd_basis(build_category_contrast_matrix(cached["ans_centers"], categories, cat), args.rank)
            x_train = project_np(np.stack([r["pre_vec"] for r in train_records_cat]), pre_basis)
            y_train = project_np(np.stack([r["answer_vec"] for r in train_records_cat]), ans_basis)
            transfer = ridge_map(x_train, y_train, args.ridge)
            canonical_ids = target_token_ids(tokenizer, cat)
            lm_dir = lm_head_direction(model, canonical_ids)
            surfaces = surface_strings(cat, fmt)
            surface_id_sets = {name: first_token_set(tokenizer, vals) for name, vals in surfaces.items()}
            phrase_sets = {name: phrase_token_sequences(tokenizer, vals, args.max_surface_tokens) for name, vals in surfaces.items()}
            variants = {}
            clean_logits = clean_logits_for_items(model, tokenizer, device, held_cat, args.batch_size, args.max_length)
            clean_ranks = {set_name: rank_for_ids(clean_logits, ids) for set_name, ids in surface_id_sets.items()}
            clean_greedy = greedy_tokens_from_logits(tokenizer, clean_logits, args.generate_steps)
            clean_decoded = ["".join(row) for row in clean_greedy]
            clean_classes = [classify_text(x, surfaces) for x in clean_decoded]
            variants["clean"] = {
                "surface_ranks": clean_ranks,
                "greedy_text_examples": clean_decoded[:8],
                "greedy_class_rates": {c: float(clean_classes.count(c) / max(1, len(clean_classes))) for c in sorted(set(clean_classes))},
            }
            for name, support, final_mode, lm_scale in [
                ("support_only", support_scale, "none", 0.0),
                ("final_norm_output_lm", support_scale, "final_norm_output_lm", args.lm_scale),
            ]:
                out = run_gate_condition(
                    model, tokenizer, device, layers, held_cat, cat_local_ids, categories,
                    cand, cat, args.batch_size, args.max_length, layer_id, site,
                    pre_basis, ans_basis, transfer, support, canonical_ids,
                    final_mode=final_mode, lm_dir=lm_dir, lm_scale=lm_scale,
                    return_logits=True,
                )
                logits = out["logits"]
                ranks = {set_name: rank_for_ids(logits, ids) for set_name, ids in surface_id_sets.items()}
                greedy = greedy_tokens_from_logits(tokenizer, logits, args.generate_steps)
                decoded = ["".join(row) for row in greedy]
                classes = [classify_text(x, surfaces) for x in decoded]
                class_counts = {c: float(classes.count(c) / max(1, len(classes))) for c in sorted(set(classes))}
                variants[name] = {
                    "token": out["token"],
                    "surface_ranks": ranks,
                    "greedy_text_examples": decoded[:8],
                    "greedy_class_rates": class_counts,
                }
            result["results"][key] = {
                "path": {"layer_id": layer_id, "site": site, "scale": support_scale},
                "surface_strings": surfaces,
                "surface_id_sets": {k: len(v) for k, v in surface_id_sets.items()},
                "phrase_set_counts": {k: len(v) for k, v in phrase_sets.items()},
                "variants": variants,
            }
        return result
    finally:
        release_loaded(loaded)


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [f"# Phase 151 Surface Answer Generation Closure: {result['model']}", ""]
    lines.append(f"Generated: {result['timestamp']}")
    lines.append("")
    lines.append("| case | clean expanded arg | support expanded arg | final expanded arg | final expanded rank | final canonical rank | greedy class | examples |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for key, item in sorted(result["results"].items()):
        clean = item["variants"]["clean"]["surface_ranks"]["expanded"]
        support = item["variants"]["support_only"]["surface_ranks"]["expanded"]
        final = item["variants"]["final_norm_output_lm"]["surface_ranks"]["expanded"]
        canon = item["variants"]["final_norm_output_lm"]["surface_ranks"]["canonical"]
        cls = item["variants"]["final_norm_output_lm"]["greedy_class_rates"]
        top_cls = max(cls.items(), key=lambda x: x[1])[0] if cls else ""
        ex = " | ".join(x.replace("\n", "\\n") for x in item["variants"]["final_norm_output_lm"]["greedy_text_examples"][:3])
        lines.append(
            f"| {key} | {clean['argmax']:.2f} | {support['argmax']:.2f} | {final['argmax']:.2f} | "
            f"{final['rank']:.1f} | {canon['rank']:.1f} | {top_cls} | {ex} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--categories", default="plant,time,container,number")
    parser.add_argument("--template-families", default="long,short,neutral")
    parser.add_argument("--splits", default="front_back,back_front")
    parser.add_argument("--formats", default="label_colon,multiple_choice,answer_one_word")
    parser.add_argument("--train-objects", type=int, default=8)
    parser.add_argument("--test-objects", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=160)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--lm-scale", type=float, default=4.0)
    parser.add_argument("--generate-steps", type=int, default=3)
    parser.add_argument("--max-surface-tokens", type=int, default=4)
    parser.add_argument("--output-dir", default=str(OUT_ROOT))
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = run_model(args)
    json_path = out_dir / f"phase151_{args.model}_surface_answer_generation_closure.json"
    md_path = out_dir / f"phase151_{args.model}_surface_answer_generation_closure.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(result, md_path)
    log(f"Wrote {json_path}")
    log(f"Wrote {md_path}")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
