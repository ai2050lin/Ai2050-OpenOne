#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import string
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))

from model_utils import MODEL_CONFIGS, release_model  # noqa: E402
from phase722_functional_head_atlas_causal_ablation import logit_diag, write_json, write_jsonl  # noqa: E402
from phase735_source_restricted_writer_validation import MODELS, select_evenly  # noqa: E402
from phase755_cross_domain_route_invariance_atlas import get_first_token_id  # noqa: E402
from phase762_semantic_numeric_fiber_atlas import VALUE_POOLS  # noqa: E402
from phase771_matched_causal_intervention_reliability_test import (  # noqa: E402
    case_map_for,
    margin,
    pair_info_map,
    phase770_path,
    phase767_path,
    select_matched_case_ids,
    semantic_label,
)
from phase773_instruction_source_disentanglement import fmt, load_json, load_jsonl  # noqa: E402
from phase775_semantic_latent_route_output_closure import focus_filter, pool_diag, prompt_for_variant, value_pool  # noqa: E402


OUT_ROOT = Path("results/glm5_phase776_readout_bridge_competition_audit")
RESULT_ROOT = Path("tests/result/phase776_readout_bridge_competition_audit")

PROMPT_VARIANTS = ["without_candidate_list", "constrained_free_prompt", "with_candidate_list"]
FUNCTION_OR_EXPLANATION = {
    "the",
    "a",
    "an",
    "it",
    "its",
    "this",
    "that",
    "they",
    "there",
    "is",
    "are",
    "was",
    "were",
    "be",
    "to",
    "of",
    "for",
    "in",
    "on",
    "with",
    "as",
    "and",
    "or",
    "because",
    "usually",
    "typically",
    "generally",
    "answer",
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def safe_mean(values: list[Any]) -> float | None:
    vals = []
    for value in values:
        try:
            val = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(val):
            vals.append(val)
    return sum(vals) / len(vals) if vals else None


def load_model_bf16_prefer_flash(model_name: str, attn_impls: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = MODEL_CONFIGS[model_name]
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"],
        trust_remote_code=True,
        local_files_only=True,
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    errors = []
    for impl in [x.strip() for x in attn_impls.split(",") if x.strip()]:
        try:
            log(f"[load] {model_name}: bf16 device_map=auto attn={impl} quantization=off")
            model = AutoModelForCausalLM.from_pretrained(
                cfg["path"],
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                local_files_only=True,
                attn_implementation=impl,
            )
            model.eval()
            device = next(model.parameters()).device
            gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
            log(f"[load] {model_name}: loaded {impl}, device={device}, gpu={gpu_mem:.2f}GB")
            return model, tokenizer, device, impl
        except Exception as exc:
            errors.append(f"{impl}: {exc}")
            log(f"[load] {model_name}: {impl} failed: {exc}")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    raise RuntimeError("all attention implementations failed: " + " | ".join(errors))


def run_next_logits(model, tokenizer, device, prompt: str) -> torch.Tensor:
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    with torch.inference_mode():
        out = model(
            input_ids=torch.tensor([ids], device=device),
            return_dict=True,
            use_cache=False,
        )
    return out.logits[0, -1].detach().float().cpu()


def normalize_token_text(text: str) -> str:
    return text.replace("Ġ", " ").replace("▁", " ").strip()


def all_value_token_ids(tokenizer) -> set[int]:
    ids = set()
    for values in VALUE_POOLS.values():
        for value in values:
            ids.add(get_first_token_id(tokenizer, str(value)))
    return ids


def all_value_strings() -> set[str]:
    values = set()
    for pool in VALUE_POOLS.values():
        for value in pool:
            values.add(str(value).strip().lower())
    return values


def classify_token(
    tokenizer,
    token_id: int,
    token_text: str,
    target_id: int,
    contrast_id: int,
    pool_ids: set[int],
    all_value_ids: set[int],
    target_value: str,
    contrast_value: str,
    pool_value_strings: set[str],
    all_value_texts: set[str],
) -> str:
    norm = normalize_token_text(token_text)
    lower = norm.lower()
    target_lower = str(target_value).strip().lower()
    contrast_lower = str(contrast_value).strip().lower()
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    if token_id == target_id:
        return "target_value"
    if lower == target_lower:
        return "case_variant_target_value"
    if token_id == contrast_id:
        return "contrast_value"
    if lower == contrast_lower:
        return "case_variant_contrast_value"
    if token_id in pool_ids:
        return "relation_pool_wrong_value"
    if lower in pool_value_strings:
        return "case_variant_pool_value"
    if token_id in all_value_ids:
        return "other_relation_value"
    if lower in all_value_texts:
        return "case_variant_other_value"
    if token_id in special_ids:
        return "special_token"
    if not norm:
        return "whitespace_or_empty"
    if any(ch in token_text for ch in ["\n", "\r", "\t"]):
        return "whitespace_or_empty"
    if all(ch in string.punctuation for ch in norm):
        return "punctuation"
    if lower in FUNCTION_OR_EXPLANATION:
        return "format_or_explanation_word"
    if any(ch.isdigit() for ch in norm):
        return "number_or_symbol"
    if lower in {"yes", "no", "true", "false"}:
        return "boolean_value"
    if norm[:1].isupper():
        return "lexical_capitalized"
    if any(ch.isalpha() for ch in norm):
        return "lexical_word"
    return "other_token"


def topk_competitors(
    tokenizer,
    logits: torch.Tensor,
    target_id: int,
    contrast_id: int,
    pool: list[dict[str, Any]],
    target_value: str,
    contrast_value: str,
    top_k: int,
) -> list[dict[str, Any]]:
    k = min(top_k, logits.numel())
    vals, ids = torch.topk(logits, k)
    target_logit = float(logits[target_id].item())
    pool_ids = {int(x["token_id"]) for x in pool}
    pool_value_strings = {str(x["value"]).strip().lower() for x in pool}
    all_value_ids = all_value_token_ids(tokenizer)
    all_value_texts = all_value_strings()
    rows = []
    for rank, (val, tid_tensor) in enumerate(zip(vals.tolist(), ids.tolist()), 1):
        tid = int(tid_tensor)
        text = tokenizer.decode([tid], skip_special_tokens=False)
        cls = classify_token(
            tokenizer,
            tid,
            text,
            target_id,
            contrast_id,
            pool_ids,
            all_value_ids,
            target_value,
            contrast_value,
            pool_value_strings,
            all_value_texts,
        )
        rows.append(
            {
                "top_rank": rank,
                "token_id": tid,
                "token_text": text,
                "token_text_norm": normalize_token_text(text),
                "competitor_class": cls,
                "is_target": tid == target_id,
                "is_pool_value": tid in pool_ids,
                "logit": float(val),
                "gap_above_target": float(val - target_logit),
            }
        )
    return rows


def observation_for(
    tokenizer,
    logits: torch.Tensor,
    case: dict[str, Any],
    prompt_variant: str,
    case_label: dict[str, Any],
    pair_info: dict[str, Any],
    top_k: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    target_id = get_first_token_id(tokenizer, case["answer"])
    contrast_id = get_first_token_id(tokenizer, case["contrast_answer"])
    pool = value_pool(tokenizer, case, target_id, contrast_id)
    target_diag = logit_diag(logits, target_id)
    contrast_diag = logit_diag(logits, contrast_id)
    pdiag = pool_diag(logits, pool, target_id, contrast_id)
    top_rows = topk_competitors(tokenizer, logits, target_id, contrast_id, pool, case["answer"], case["contrast_answer"], top_k)
    top1 = top_rows[0] if top_rows else {}
    phase767 = case_label["phase767"]
    base = {
        "row_kind": "readout_bridge_observation",
        "prompt_variant": prompt_variant,
        "include_candidate_list": prompt_variant == "with_candidate_list",
        "case_id": case_label["case_id"],
        "pair_index": case_label["pair_index"],
        "matched_arm": case_label["matched_arm"],
        "stratum": case_label["stratum"],
        "object": case["object"],
        "domain": case["domain"],
        "relation": case["relation"],
        "context_format": case["context_format"],
        "target_answer": case["answer"],
        "contrast_answer": case["contrast_answer"],
        "phase767_exact_top1": bool(phase767.get("exact_target_top1")),
        "phase767_semantic_top1": bool(phase767.get("target_top1")),
        "semantic_label": semantic_label(phase767),
        **pair_info,
        "base_target_rank": target_diag["target_rank"],
        "base_target_top1": bool(target_diag["target_top1"]),
        "base_contrast_rank": contrast_diag["target_rank"],
        "base_margin_target_vs_contrast": margin(logits, target_id, contrast_id),
        "pool_size": pdiag["pool_size"],
        "pool_target_rank": pdiag["pool_target_rank"],
        "pool_target_top1": bool(pdiag["pool_target_top1"]),
        "pool_best_value": pdiag["pool_best_value"],
        "pool_margin_target_vs_best_other": pdiag["pool_margin_target_vs_best_other"],
        "latent_pool_hit": (not bool(target_diag["target_top1"])) and bool(pdiag["pool_target_top1"]),
        "top1_token_id": top1.get("token_id"),
        "top1_token_text": top1.get("token_text"),
        "top1_token_text_norm": top1.get("token_text_norm"),
        "top1_competitor_class": top1.get("competitor_class"),
        "top1_gap_above_target": top1.get("gap_above_target"),
    }
    comp_rows = []
    for row in top_rows:
        comp_rows.append(
            {
                "row_kind": "readout_bridge_topk_competitor",
                **{k: v for k, v in base.items() if k != "row_kind"},
                **row,
            }
        )
    return base, comp_rows


def audit_case(model, tokenizer, device, args: argparse.Namespace, case: dict[str, Any], case_label: dict[str, Any], pair_info: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for prompt_variant in [v.strip() for v in args.prompt_variants.split(",") if v.strip()]:
        prompt = prompt_for_variant(case, prompt_variant)
        logits = run_next_logits(model, tokenizer, device, prompt)
        obs, comps = observation_for(tokenizer, logits, case, prompt_variant, case_label, pair_info, args.top_k)
        rows.append(obs)
        rows.extend(comps)
    return rows


def group_observations(rows: list[dict[str, Any]], key_fields: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(k) for k in key_fields)].append(row)
    out = []
    for key, vals in sorted(groups.items(), key=lambda kv: str(kv[0])):
        payload = {field: value for field, value in zip(key_fields, key)}
        payload.update(
            {
                "n": len(vals),
                "case_n": len({v["case_id"] for v in vals}),
                "base_top1_rate": sum(1 for v in vals if v.get("base_target_top1")) / len(vals) if vals else None,
                "pool_top1_rate": sum(1 for v in vals if v.get("pool_target_top1")) / len(vals) if vals else None,
                "latent_pool_hit_rate": sum(1 for v in vals if v.get("latent_pool_hit")) / len(vals) if vals else None,
                "mean_base_target_rank": safe_mean([v.get("base_target_rank") for v in vals]),
                "mean_pool_target_rank": safe_mean([v.get("pool_target_rank") for v in vals]),
                "mean_top1_gap_above_target": safe_mean([v.get("top1_gap_above_target") for v in vals]),
            }
        )
        out.append(payload)
    out.sort(key=lambda r: (r.get("prompt_variant") or "", r.get("latent_pool_hit_rate") or 0.0), reverse=True)
    return out


def group_competitors(rows: list[dict[str, Any]], key_fields: list[str], top1_only: bool = False, latent_only: bool = False) -> list[dict[str, Any]]:
    vals = rows
    if top1_only:
        vals = [r for r in vals if r.get("top_rank") == 1 and not r.get("is_target")]
    if latent_only:
        vals = [r for r in vals if r.get("latent_pool_hit")]
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in vals:
        groups[tuple(row.get(k) for k in key_fields)].append(row)
    out = []
    for key, items in sorted(groups.items(), key=lambda kv: str(kv[0])):
        payload = {field: value for field, value in zip(key_fields, key)}
        payload.update(
            {
                "n": len(items),
                "case_n": len({r["case_id"] for r in items}),
                "mean_gap_above_target": safe_mean([r.get("gap_above_target") for r in items]),
                "mean_base_target_rank": safe_mean([r.get("base_target_rank") for r in items]),
                "mean_pool_target_rank": safe_mean([r.get("pool_target_rank") for r in items]),
            }
        )
        out.append(payload)
    out.sort(key=lambda r: (r.get("n") or 0, r.get("mean_gap_above_target") or 0.0), reverse=True)
    return out


def summarize_rows(rows: list[dict[str, Any]], args: argparse.Namespace, model_name: str, attn_impl: str) -> dict[str, Any]:
    observations = [r for r in rows if r.get("row_kind") == "readout_bridge_observation"]
    competitors = [r for r in rows if r.get("row_kind") == "readout_bridge_topk_competitor"]
    return {
        "phase": 776,
        "title": "Readout-Bridge Competition Audit",
        "model": model_name,
        "round": args.round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "dtype": "bfloat16",
        "quantization": "off",
        "focus": args.focus,
        "top_k": args.top_k,
        "prompt_variants": [v.strip() for v in args.prompt_variants.split(",") if v.strip()],
        "n_rows": len(rows),
        "n_observation_rows": len(observations),
        "n_competitor_rows": len(competitors),
        "n_cases": len({r["case_id"] for r in observations}),
        "n_pairs": len({r["pair_index"] for r in observations}),
        "by_prompt_observation": group_observations(observations, ["prompt_variant"]),
        "top1_competitor_by_prompt": group_competitors(competitors, ["prompt_variant", "competitor_class"], top1_only=True),
        "latent_top1_competitor_by_prompt": group_competitors(
            competitors, ["prompt_variant", "competitor_class"], top1_only=True, latent_only=True
        ),
        "topk_competitor_by_prompt": group_competitors(competitors, ["prompt_variant", "competitor_class"], top1_only=False),
        "strict_interpretation": "This phase classifies open-vocabulary competitors that beat the target; it does not repair them.",
    }


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = OUT_ROOT / args.round_name
    result_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    rows767 = load_jsonl(phase767_path(args.model, args.phase767_round))
    phase770 = load_json(phase770_path(args.phase770_round))
    selected = select_matched_case_ids(rows767, args)
    cmap = case_map_for(args)
    pinfo = pair_info_map(phase770, args.model)
    selected = focus_filter(selected, pinfo, args)
    log(f"{args.model}/{args.round_name}: focus={args.focus} cases={len(selected)} variants={args.prompt_variants} top_k={args.top_k}")
    model, tokenizer, device, attn_impl = load_model_bf16_prefer_flash(args.model, args.attn_implementations)
    try:
        rows: list[dict[str, Any]] = []
        for idx, item in enumerate(selected, 1):
            case = cmap[item["case_id"]]
            rows.extend(audit_case(model, tokenizer, device, args, case, item, pinfo.get(case["case_id"], {})))
            if idx % args.log_every == 0 or idx == len(selected):
                log(f"{args.model}: readout bridge audit {idx}/{len(selected)} cases; rows={len(rows)}")
    finally:
        release_model(model)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    summary = summarize_rows(rows, args, args.model, attn_impl)
    for root in (out_dir, result_dir):
        write_jsonl(root / f"phase776_{args.model}_rows.jsonl", rows)
        write_json(root / f"phase776_{args.model}_summary.json", summary)
    print(
        json.dumps(
            {
                "model": args.model,
                "round": args.round_name,
                "attn": attn_impl,
                "n_cases": summary["n_cases"],
                "by_prompt_observation": summary["by_prompt_observation"],
                "latent_top1_competitor_by_prompt": summary["latent_top1_competitor_by_prompt"][:8],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return summary


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# Phase 776 Readout-Bridge Competition Audit ({payload['round']})",
        "",
        f"- Status: `{payload['status']}`",
        "- Test: classify open-vocabulary top-k competitors that beat the semantic target.",
        "- Models are run sequentially; bf16, quantization off; attention implementation prefers flash/sdpa and falls back to eager.",
        "",
        "## Prompt Observation Summary",
        "",
        "| model | variant | rows | cases | base top1 | pool top1 | latent hit | base rank | pool rank | top1 gap |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        data = payload["by_model"].get(model)
        if not data:
            continue
        for row in data["by_prompt_observation"]:
            lines.append(
                f"| {model} | `{row['prompt_variant']}` | {row['n']} | {row['case_n']} | "
                f"{fmt(row['base_top1_rate'])} | {fmt(row['pool_top1_rate'])} | {fmt(row['latent_pool_hit_rate'])} | "
                f"{fmt(row['mean_base_target_rank'])} | {fmt(row['mean_pool_target_rank'])} | {fmt(row['mean_top1_gap_above_target'])} |"
            )
    lines += [
        "",
        "## Latent-Hit Top1 Competitor Classes",
        "",
        "| model | variant | class | rows | cases | mean gap above target |",
        "|---|---|---|---:|---:|---:|",
    ]
    for model in MODELS:
        data = payload["by_model"].get(model)
        if not data:
            continue
        for row in data["latent_top1_competitor_by_prompt"]:
            lines.append(
                f"| {model} | `{row['prompt_variant']}` | `{row['competitor_class']}` | {row['n']} | {row['case_n']} | "
                f"{fmt(row['mean_gap_above_target'])} |"
            )
    lines += [
        "",
        "## All Top1 Competitor Classes",
        "",
        "| model | variant | class | rows | cases | mean gap above target |",
        "|---|---|---|---:|---:|---:|",
    ]
    for model in MODELS:
        data = payload["by_model"].get(model)
        if not data:
            continue
        for row in data["top1_competitor_by_prompt"]:
            lines.append(
                f"| {model} | `{row['prompt_variant']}` | `{row['competitor_class']}` | {row['n']} | {row['case_n']} | "
                f"{fmt(row['mean_gap_above_target'])} |"
            )
    lines += [
        "",
        "## Strict Interpretation",
        "",
        "- This audit names the open-vocabulary competitor classes that beat the target.",
        "- It does not prove the competitor class is causally suppressing the target.",
        "- It separates readout competition from semantic value-pool selection.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_cross_summary(round_name: str) -> dict[str, Any]:
    by_model = {}
    for model in MODELS:
        path = OUT_ROOT / round_name / f"phase776_{model}_summary.json"
        if path.exists():
            by_model[model] = load_json(path)
    payload = {
        "phase": 776,
        "title": "Readout-Bridge Competition Audit",
        "round": round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "complete" if len(by_model) == len(MODELS) else "partial",
        "models": list(by_model),
        "by_model": by_model,
    }
    for root in (OUT_ROOT, RESULT_ROOT):
        out_dir = root / round_name
        write_json(out_dir / "phase776_cross_model_summary.json", payload)
        write_markdown(out_dir / "phase776_cross_model_summary.md", payload)
    print(json.dumps({"round": round_name, "status": payload["status"], "models": payload["models"]}, ensure_ascii=False, indent=2), flush=True)
    return payload


def dry_run(args: argparse.Namespace) -> None:
    payload = {"round": args.round_name, "focus": args.focus, "models": {}}
    for model in MODELS:
        args.model = model
        rows767 = load_jsonl(phase767_path(model, args.phase767_round))
        phase770 = load_json(phase770_path(args.phase770_round))
        selected = select_matched_case_ids(rows767, args)
        pinfo = pair_info_map(phase770, model)
        selected = focus_filter(selected, pinfo, args)
        payload["models"][model] = {
            "selected_cases": len(selected),
            "pairs": len({x["pair_index"] for x in selected}),
            "arms": dict(Counter(x["matched_arm"] for x in selected)),
            "fiber": dict(Counter(pinfo.get(x["case_id"], {}).get("fiber_bucket", "missing") for x in selected)),
            "prompt_variants": [v.strip() for v in args.prompt_variants.split(",") if v.strip()],
            "top_k": args.top_k,
        }
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="smoke")
    parser.add_argument("--phase767-round", default="main")
    parser.add_argument("--phase770-round", default="confirm_x_main")
    parser.add_argument("--relations", default="category,edible,grows_on_tree")
    parser.add_argument("--focus", choices=["all_matched", "clean", "fiber_high", "clean_fiber_high"], default="clean_fiber_high")
    parser.add_argument("--prompt-variants", default="without_candidate_list,constrained_free_prompt,with_candidate_list")
    parser.add_argument("--max-per-stratum", type=int, default=1)
    parser.add_argument("--max-pairs", type=int, default=8)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--attn-implementations", default="flash_attention_2,sdpa,eager")
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        dry_run(args)
        return
    if args.summarize_only:
        write_cross_summary(args.round_name)
        return
    if not args.model:
        raise SystemExit("--model is required unless --dry-run or --summarize-only")
    run_model(args)
    if args.hard_exit_after_model:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
