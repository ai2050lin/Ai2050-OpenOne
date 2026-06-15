#!/usr/bin/env python3
"""
Phase 150: open-vocab competitor and format gate decomposition.

Reuse Phase147 train-selected paths and decompose why full-vocab argmax fails
after support restore / final-norm output LM steering.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import string
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hf_probe_env import get_layers, load_probe_model, release_loaded, vram_gb  # noqa: E402
from phase105_global_category_atlas_cuda import CATEGORY_OBJECTS, CATEGORY_READOUT_WORDS, collect_readout_rows, find_token_id  # noqa: E402
from phase114_answer_site_causal_subspace_cuda import build_category_contrast_matrix  # noqa: E402
from phase116_subspace_basis_component_audit_cuda import svd_basis  # noqa: E402
from phase138_mechanism_transfer_closure_cuda import project_np, ridge_map  # noqa: E402
from phase139_restore_swap_calibration_cuda import parse_str_list  # noqa: E402
from phase145_mechanism_stability_generation_cuda import split_indices  # noqa: E402
from phase146_template_router_token_gap_cuda import capture_records, centers_from_records, target_token_ids  # noqa: E402
from phase147_train_router_format_token_cuda import build_items  # noqa: E402
from phase148_router_feature_lmhead_alignment_cuda import lm_head_direction  # noqa: E402
from phase149_final_norm_candidate_gate_cuda import candidate_ids, run_gate_condition  # noqa: E402


OUT_ROOT = Path("results/gpt5_phase150_open_vocab_competitor_gate")
PHASE147_ROOT = Path("results/gpt5_phase147_train_router_format_token")

GENERIC_WORDS = {
    "type", "kind", "thing", "object", "category", "word", "term", "class",
    "concept", "item", "entity", "example", "answer", "option",
}
ARTICLE_PREP = {
    "a", "an", "the", "of", "in", "on", "to", "for", "with", "as", "by", "at", "from",
}
PUNCT = set(string.punctuation)


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def first_ids(tokenizer: Any, words: list[str]) -> list[int]:
    ids = []
    for word in words:
        tid = find_token_id(tokenizer, word)
        if tid is not None:
            ids.append(int(tid))
    return sorted(set(ids))


def all_category_token_ids(tokenizer: Any, cats: list[str]) -> list[int]:
    ids: list[int] = []
    for cat in cats:
        ids.extend(first_ids(tokenizer, CATEGORY_READOUT_WORDS[cat]))
    return sorted(set(ids))


def build_vocab_sets(tokenizer: Any, all_categories: list[str], test_categories: list[str]) -> dict[str, list[int]]:
    vocab = tokenizer.get_vocab()
    alpha_ids = []
    non_format_ids = []
    for token, tid in vocab.items():
        text = decode_clean(tokenizer, int(tid))
        stripped = text.strip()
        if re.fullmatch(r"[A-Za-z][A-Za-z_-]*", stripped):
            alpha_ids.append(int(tid))
        if stripped and not all(ch in PUNCT for ch in stripped) and re.search(r"[A-Za-z]", stripped):
            non_format_ids.append(int(tid))
    return {
        "full": list(range(len(vocab))),
        "alphabetic": sorted(set(alpha_ids)),
        "non_format": sorted(set(non_format_ids)),
        "semantic_all_categories": all_category_token_ids(tokenizer, all_categories),
        "candidate4": all_category_token_ids(tokenizer, test_categories),
    }


def decode_clean(tokenizer: Any, tid: int) -> str:
    return tokenizer.decode([int(tid)], clean_up_tokenization_spaces=False)


def token_class(text: str, target_ids_: set[int], tid: int, object_words: set[str], category_words: set[str], synonym_words: set[str]) -> str:
    raw = text
    stripped = raw.strip()
    lower = stripped.lower()
    if tid in target_ids_:
        return "target_category"
    if not stripped:
        return "whitespace"
    if all(ch in PUNCT for ch in stripped):
        return "punctuation"
    if lower in ARTICLE_PREP:
        return "article_prep"
    if re.fullmatch(r"[A-Da-d]|\(?[1-4]\)?", stripped):
        return "option_label"
    if lower in synonym_words:
        return "target_synonym"
    if lower in category_words:
        return "other_category"
    if lower in object_words:
        return "object_token"
    if lower in GENERIC_WORDS:
        return "generic_continuation"
    if re.fullmatch(r"[A-Za-z][A-Za-z_-]*", stripped):
        return "alphabetic_other"
    if any(ord(ch) > 127 for ch in stripped):
        return "non_ascii_or_fragment"
    return "format_or_fragment"


def rank_in_subset(logits: torch.Tensor, target_ids_: list[int], subset_ids: list[int]) -> dict[str, float]:
    if not target_ids_ or not subset_ids:
        return {"rank": 0.0, "argmax": 0.0, "margin": 0.0}
    subset = torch.tensor(sorted(set(subset_ids)), device=logits.device, dtype=torch.long)
    target = torch.tensor(sorted(set(target_ids_) & set(subset_ids)), device=logits.device, dtype=torch.long)
    if target.numel() == 0:
        return {"rank": 0.0, "argmax": 0.0, "margin": 0.0}
    vals = logits[:, subset].float()
    target_vals = logits[:, target].float()
    target_max = target_vals.max(dim=1).values
    rank = (vals > target_max[:, None]).sum(dim=1).float() + 1
    arg = (rank == 1).float()
    non_target_mask = torch.ones(subset.numel(), dtype=torch.bool, device=logits.device)
    target_set = set(int(x) for x in target.detach().cpu().tolist())
    for i, sid in enumerate(subset.detach().cpu().tolist()):
        if int(sid) in target_set:
            non_target_mask[i] = False
    if non_target_mask.any():
        margin = target_max - vals[:, non_target_mask].max(dim=1).values
    else:
        margin = torch.zeros_like(target_max)
    return {
        "rank": float(rank.mean().detach().cpu()),
        "argmax": float(arg.mean().detach().cpu()),
        "margin": float(margin.mean().detach().cpu()),
    }


def logit_audit(
    logits: torch.Tensor,
    tokenizer: Any,
    target_ids_: list[int],
    target_cat: str,
    test_categories: list[str],
    all_categories: list[str],
    prompt_objects: list[str],
    top_k: int,
    vocab_sets: dict[str, list[int]],
) -> dict[str, Any]:
    vocab_size = int(logits.shape[-1])
    top = torch.topk(logits.float(), k=min(top_k, vocab_size), dim=-1).indices.detach().cpu().numpy()
    target_set = set(target_ids_)
    object_words = {x.lower() for x in prompt_objects}
    category_words = {
        w.lower()
        for cat in all_categories
        for w in CATEGORY_READOUT_WORDS[cat]
    }
    synonym_words = {w.lower() for w in CATEGORY_READOUT_WORDS[target_cat]}
    class_counts: Counter[str] = Counter()
    token_counts: Counter[int] = Counter()
    first_class_counts: Counter[str] = Counter()
    for row in top:
        if len(row):
            txt0 = decode_clean(tokenizer, int(row[0]))
            first_class_counts[token_class(txt0, target_set, int(row[0]), object_words, category_words, synonym_words)] += 1
        for tid in row:
            tid = int(tid)
            token_counts[tid] += 1
            txt = decode_clean(tokenizer, tid)
            class_counts[token_class(txt, target_set, tid, object_words, category_words, synonym_words)] += 1
    target_syn_subset = first_ids(tokenizer, CATEGORY_READOUT_WORDS[target_cat])
    subset_metrics = {
        "full": rank_in_subset(logits, target_ids_, vocab_sets["full"]),
        "non_format": rank_in_subset(logits, target_ids_, vocab_sets["non_format"]),
        "alphabetic": rank_in_subset(logits, target_ids_, vocab_sets["alphabetic"]),
        "candidate4": rank_in_subset(logits, target_ids_, vocab_sets["candidate4"]),
        "semantic_all_categories": rank_in_subset(logits, target_ids_, vocab_sets["semantic_all_categories"]),
        "target_synonyms": rank_in_subset(logits, target_ids_, target_syn_subset),
    }
    total_top = max(1, top.shape[0] * top.shape[1])
    first_total = max(1, top.shape[0])
    common = [
        {
            "token_id": int(tid),
            "token": decode_clean(tokenizer, int(tid)),
            "count": int(cnt),
            "rate": float(cnt / max(1, top.shape[0])),
        }
        for tid, cnt in token_counts.most_common(10)
    ]
    return {
        "top_class_rates": {k: float(v / total_top) for k, v in sorted(class_counts.items())},
        "argmax_class_rates": {k: float(v / first_total) for k, v in sorted(first_class_counts.items())},
        "subset_metrics": subset_metrics,
        "top_tokens": common,
        "format_competitor_ids": [
            int(tid) for tid, _cnt in token_counts.most_common(50)
            if token_class(decode_clean(tokenizer, int(tid)), target_set, int(tid), object_words, category_words, synonym_words)
            in {"whitespace", "punctuation", "format_or_fragment", "non_ascii_or_fragment", "article_prep", "generic_continuation"}
        ][:8],
    }


def run_logits_condition(
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
    lm_dir: np.ndarray | None,
    mode: str,
    lm_scale: float,
) -> tuple[torch.Tensor, list[str]]:
    logits_rows = []
    objects: list[str] = []
    for start in range(0, len(prompts), batch_size):
        batch_items = prompts[start:start + batch_size]
        out = run_gate_condition(
            model, tokenizer, device, layers, batch_items, cat_local_ids, all_categories,
            candidate_by_cat, target_cat, len(batch_items), max_length, layer_id, site,
            pre_basis, ans_basis, transfer, support_scale, target_ids_,
            final_mode=mode, lm_dir=lm_dir, lm_scale=lm_scale,
        )
        # Re-run a small local forward to get final logits is expensive in phase149's API.
        # Instead, use top-token proxy from token audit is insufficient, so keep this
        # path disabled by calling the lower-level helper below in future phases.
        raise RuntimeError("run_logits_condition placeholder should not be called")
    return torch.cat(logits_rows, dim=0), objects


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
        cat_local_ids, _rows, _token_labels = collect_readout_rows(model, tokenizer, categories)
        cand = candidate_ids(tokenizer, test_categories)
        vocab_sets = build_vocab_sets(tokenizer, categories, test_categories)
        alloc, reserved = vram_gb()
        log(f"{args.model}: phase150 real-logit ecology, vram={alloc:.2f}/{reserved:.2f}GB")
        result: dict[str, Any] = {
            "phase": 150,
            "model": args.model,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source": str(phase147_path),
            "categories": test_categories,
            "top_k": args.top_k,
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
                train_records_all = capture_records(model, tokenizer, device, layers, train_all, cat_local_ids, categories, args.batch_size, args.max_length, layer_id)
                train_cache[cache_key] = {
                    "records": train_records_all,
                    "pre_centers": centers_from_records(train_records_all, categories, "pre_vec", len(train_tpl)),
                    "ans_centers": centers_from_records(train_records_all, categories, "answer_vec", len(train_tpl)),
                }
            cached = train_cache[cache_key]
            train_records_cat = [r for r in cached["records"] if r["cat"] == cat]
            pre_basis, _ = svd_basis(build_category_contrast_matrix(cached["pre_centers"], categories, cat), args.rank)
            ans_basis, _ = svd_basis(build_category_contrast_matrix(cached["ans_centers"], categories, cat), args.rank)
            x_train = project_np(np.stack([r["pre_vec"] for r in train_records_cat]), pre_basis)
            y_train = project_np(np.stack([r["answer_vec"] for r in train_records_cat]), ans_basis)
            transfer = ridge_map(x_train, y_train, args.ridge)
            tids = target_token_ids(tokenizer, cat)
            lm_dir = lm_head_direction(model, tids)
            prompt_objects = [x["obj"] for x in held_cat]
            variants = {}
            variant_specs = [
                ("support_only", "none", 0.0, 0.0),
                ("final_norm_output_lm", "final_norm_output_lm", args.lm_scale, 0.0),
                ("final_norm_output_suppress", "final_norm_output_suppress", 0.0, args.suppress_scale),
            ]
            for name, mode, lm_scale, suppress_scale in variant_specs:
                out = run_gate_condition(
                    model, tokenizer, device, layers, held_cat, cat_local_ids, categories,
                    cand, cat, args.batch_size, args.max_length, layer_id, site,
                    pre_basis, ans_basis, transfer, support_scale, tids,
                    final_mode=mode, lm_dir=lm_dir, lm_scale=lm_scale,
                    suppress_scale=suppress_scale, return_logits=True,
                )
                audit = logit_audit(
                    out["logits"], tokenizer, tids, cat, test_categories, categories,
                    prompt_objects, args.top_k, vocab_sets,
                )
                variants[name] = {
                    "token": out["token"],
                    "logit_audit": audit,
                }
            result["results"][key] = {
                "path": {"layer_id": layer_id, "site": site, "scale": support_scale},
                "variants": variants,
            }
        return result
    finally:
        release_loaded(loaded)


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [f"# Phase 150 Open-Vocab Competitor Gate: {result['model']}", ""]
    lines.append(f"Generated: {result['timestamp']}")
    lines.append("")
    lines.append("| case | support arg class | output_lm arg class | output cand4 arg | output semantic arg | output nonfmt rank | output full rank | output full arg | top tokens |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for key, item in sorted(result["results"].items()):
        support = item["variants"].get("support_only", {})
        output = item["variants"].get("final_norm_output_lm", {})
        def top_class(v: dict[str, Any]) -> str:
            rates = v.get("logit_audit", {}).get("argmax_class_rates", {})
            if not rates:
                return ""
            return max(rates.items(), key=lambda x: x[1])[0]
        audit = output.get("logit_audit", {})
        subsets = audit.get("subset_metrics", {})
        top_text = " ".join(x["token"].replace("\n", "\\n") for x in audit.get("top_tokens", [])[:5])
        lines.append(
            f"| {key} | {top_class(support)} | {top_class(output)} | "
            f"{subsets.get('candidate4', {}).get('argmax', 0):.2f} | "
            f"{subsets.get('semantic_all_categories', {}).get('argmax', 0):.2f} | "
            f"{subsets.get('non_format', {}).get('rank', 0):.1f} | "
            f"{subsets.get('full', {}).get('rank', 0):.1f} | "
            f"{subsets.get('full', {}).get('argmax', 0):.2f} | {top_text} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--categories", default="plant,time,container,number")
    parser.add_argument("--template-families", default="long,short,neutral")
    parser.add_argument("--splits", default="front_back,back_front")
    parser.add_argument("--formats", default="label_colon,multiple_choice,answer_one_word")
    parser.add_argument("--phase149-dir", default="results/gpt5_phase149_final_norm_candidate_gate")
    parser.add_argument("--lm-scale", type=float, default=4.0)
    parser.add_argument("--suppress-scale", type=float, default=1.0)
    parser.add_argument("--train-objects", type=int, default=8)
    parser.add_argument("--test-objects", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=160)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--output-dir", default=str(OUT_ROOT))
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = run_model(args)
    json_path = out_dir / f"phase150_{args.model}_open_vocab_competitor_gate.json"
    md_path = out_dir / f"phase150_{args.model}_open_vocab_competitor_gate.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(result, md_path)
    log(f"Wrote {json_path}")
    log(f"Wrote {md_path}")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
