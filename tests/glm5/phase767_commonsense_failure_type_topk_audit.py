#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
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
from phase765_commonsense_context_identity_closure_test import (  # noqa: E402
    CONTEXT_FORMATS,
    VALUE_POOLS,
    build_cases,
    prompt_for_case,
)
from phase755_cross_domain_route_invariance_atlas import get_first_token_id  # noqa: E402


MODELS = ["qwen3", "glm4", "deepseek7b"]
OUT_ROOT = Path("results/glm5_phase767_commonsense_failure_type_topk_audit")
RESULT_ROOT = Path("tests/result/phase767_commonsense_failure_type_topk_audit")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


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


def fmt(value: Any) -> str:
    if value is None:
        return "null"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def token_text(tokenizer, token_id: int) -> str:
    text = tokenizer.decode([int(token_id)], skip_special_tokens=False)
    return text.replace("\n", "\\n").replace("\r", "\\r")


def value_aliases(value: str) -> list[str]:
    variants = [value, value.strip(), value.lower(), value.capitalize(), value.upper()]
    out = []
    for item in variants:
        if item and item not in out:
            out.append(item)
    return out


def value_token_ids(tokenizer, value: str) -> list[int]:
    ids = []
    for alias in value_aliases(value):
        try:
            tid = int(get_first_token_id(tokenizer, alias))
        except Exception:
            continue
        if tid not in ids:
            ids.append(tid)
    return ids


def protocol_like(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    if stripped in {":", "-", "–", "—", ".", ",", ";", "(", ")", "[", "]", "{", "}", "\"", "'"}:
        return True
    if stripped.lower() in {"the", "a", "an", "it", "this", "that", "answer", "because", "according", "based"}:
        return True
    return False


def load_model_bf16(model_name: str, prefer_flash: bool):
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

    attempts = ["flash_attention_2", "sdpa", "eager"] if prefer_flash else ["eager"]
    last_error: Exception | None = None
    for attn_impl in attempts:
        try:
            log(f"[load] {model_name}: bf16 device_map=auto attn={attn_impl} quantization=off")
            model = AutoModelForCausalLM.from_pretrained(
                cfg["path"],
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                local_files_only=True,
                low_cpu_mem_usage=True,
                attn_implementation=attn_impl,
            )
            model.eval()
            gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
            if hasattr(model, "hf_device_map"):
                dmap = model.hf_device_map
                gpu_count = sum(1 for v in dmap.values() if "cuda" in str(v))
                cpu_count = sum(1 for v in dmap.values() if "cpu" in str(v))
                log(
                    f"[load] {model_name}: loaded {attn_impl}, "
                    f"gpu_components={gpu_count}, cpu_components={cpu_count}, gpu={gpu_mem:.2f}GB"
                )
            else:
                log(f"[load] {model_name}: loaded {attn_impl}, device={next(model.parameters()).device}, gpu={gpu_mem:.2f}GB")
            return model, tokenizer, next(model.parameters()).device, attn_impl
        except Exception as exc:  # pragma: no cover - depends on local flash support.
            last_error = exc
            log(f"[load] {model_name}: attn={attn_impl} failed: {type(exc).__name__}: {exc}")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    raise RuntimeError(f"failed to load {model_name}") from last_error


def rank_for(logits: torch.Tensor, token_id: int) -> int:
    score = logits[int(token_id)]
    return int((logits > score).sum().item()) + 1


def allowed_value_ranks(tokenizer, logits: torch.Tensor, case: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for value in VALUE_POOLS[case["relation_key"]]:
        tids = value_token_ids(tokenizer, value)
        if not tids:
            continue
        best_tid = max(tids, key=lambda tid: float(logits[int(tid)].item()))
        rows.append(
            {
                "value": value,
                "aliases": value_aliases(value),
                "token_ids": [int(tid) for tid in tids],
                "best_token_id": int(best_tid),
                "best_token_text": token_text(tokenizer, int(best_tid)),
                "rank": rank_for(logits, best_tid),
                "logit": float(logits[int(best_tid)].item()),
            }
        )
    rows.sort(key=lambda r: (-r["logit"], r["rank"], r["value"]))
    for idx, row in enumerate(rows, 1):
        row["allowed_rank"] = idx
    return rows


def classify_failure(
    *,
    target_top1: bool,
    target_rank: int,
    contrast_rank: int,
    target_allowed_rank: int | None,
    target_in_topk: bool,
    top1_text: str,
    top_k: int,
) -> str:
    if target_top1:
        return "success_top1"
    if protocol_like(top1_text):
        return "format_protocol_miss"
    if target_rank == 2:
        return "readout_threshold_miss"
    if contrast_rank <= 2 and contrast_rank < target_rank:
        return "known_contrast_competition"
    if target_allowed_rank is not None and target_allowed_rank > 1 and target_rank <= max(10, top_k):
        return "allowed_value_candidate_competition"
    if target_in_topk:
        return "candidate_competition_other"
    if protocol_like(top1_text):
        return "format_protocol_miss"
    return "knowledge_or_state_formation_miss"


def audit_case(model, tokenizer, device, case: dict[str, Any], top_k: int) -> dict[str, Any]:
    prompt = prompt_for_case(case)
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    input_ids = torch.tensor([ids], device=device)
    with torch.inference_mode():
        out = model(input_ids=input_ids, return_dict=True, use_cache=False)
    logits = out.logits[0, -1].detach().float().cpu()

    target_id = get_first_token_id(tokenizer, case["answer"])
    contrast_id = get_first_token_id(tokenizer, case["contrast_answer"])
    target_alias_ids = value_token_ids(tokenizer, case["answer"])
    contrast_alias_ids = value_token_ids(tokenizer, case["contrast_answer"])
    exact_target_rank = rank_for(logits, target_id)
    exact_contrast_rank = rank_for(logits, contrast_id)
    semantic_target_id = max(target_alias_ids, key=lambda tid: float(logits[int(tid)].item()))
    semantic_contrast_id = max(contrast_alias_ids, key=lambda tid: float(logits[int(tid)].item()))
    target_rank = rank_for(logits, semantic_target_id)
    contrast_rank = rank_for(logits, semantic_contrast_id)
    target_top1 = target_rank == 1
    topn = min(int(top_k), int(logits.numel()))
    top_vals, top_ids = torch.topk(logits, k=topn)
    top_tokens = []
    for idx, (tid, val) in enumerate(zip(top_ids.tolist(), top_vals.tolist()), 1):
        top_tokens.append(
            {
                "rank": idx,
                "token_id": int(tid),
                "text": token_text(tokenizer, int(tid)),
                "logit": float(val),
            }
        )
    allowed_ranks = allowed_value_ranks(tokenizer, logits, case)
    target_allowed = next((r for r in allowed_ranks if r["value"] == case["answer"]), None)
    masked = logits.clone()
    for tid in target_alias_ids:
        masked[int(tid)] = -torch.inf
    best_other_logit = float(torch.max(masked).item())
    target_logit = float(logits[int(semantic_target_id)].item())
    exact_target_logit = float(logits[int(target_id)].item())
    top1 = top_tokens[0]
    failure_type = classify_failure(
        target_top1=target_top1,
        target_rank=target_rank,
        contrast_rank=contrast_rank,
        target_allowed_rank=target_allowed.get("allowed_rank") if target_allowed else None,
        target_in_topk=target_rank <= topn,
        top1_text=top1["text"],
        top_k=topn,
    )
    return {
        "row_kind": "failure_type_topk_observation",
        "case_id": case["case_id"],
        "context_format": case["context_format"],
        "object": case["object"],
        "domain": case["domain"],
        "relation": case["relation"],
        "target_answer": case["answer"],
        "contrast_answer": case["contrast_answer"],
        "target_token_id": int(target_id),
        "contrast_token_id": int(contrast_id),
        "target_alias_token_ids": [int(tid) for tid in target_alias_ids],
        "contrast_alias_token_ids": [int(tid) for tid in contrast_alias_ids],
        "semantic_target_token_id": int(semantic_target_id),
        "semantic_contrast_token_id": int(semantic_contrast_id),
        "exact_target_rank": exact_target_rank,
        "exact_contrast_rank": exact_contrast_rank,
        "target_rank": target_rank,
        "contrast_rank": contrast_rank,
        "exact_target_top1": exact_target_rank == 1,
        "target_top1": target_top1,
        "target_in_topk": target_rank <= topn,
        "top_k": topn,
        "target_logit": target_logit,
        "exact_target_logit": exact_target_logit,
        "top1_logit": float(top1["logit"]),
        "target_margin_vs_best_other": target_logit - best_other_logit,
        "top1_token": top1,
        "target_allowed_rank": target_allowed.get("allowed_rank") if target_allowed else None,
        "allowed_value_ranks": allowed_ranks,
        "top_tokens": top_tokens[: min(20, topn)],
        "failure_type": failure_type,
        "prompt_chars": len(prompt),
        "prompt_tokens": len(ids),
    }


def summarize_rows(model_name: str, round_name: str, rows: list[dict[str, Any]], attn_impl: str) -> dict[str, Any]:
    def group_counts(fields: list[str]) -> list[dict[str, Any]]:
        groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[tuple(row.get(f) for f in fields)].append(row)
        out = []
        for key, items in sorted(groups.items()):
            payload = {field: value for field, value in zip(fields, key)}
            payload.update(
                {
                    "n": len(items),
                    "target_top1_rate": safe_mean([1.0 if r["target_top1"] else 0.0 for r in items]),
                    "exact_target_top1_rate": safe_mean([1.0 if r["exact_target_top1"] else 0.0 for r in items]),
                    "mean_target_rank": safe_mean([r["target_rank"] for r in items]),
                    "mean_exact_target_rank": safe_mean([r["exact_target_rank"] for r in items]),
                    "mean_target_margin_vs_best_other": safe_mean([r["target_margin_vs_best_other"] for r in items]),
                    "target_in_topk_rate": safe_mean([1.0 if r["target_in_topk"] else 0.0 for r in items]),
                    "mean_target_allowed_rank": safe_mean([r["target_allowed_rank"] for r in items]),
                }
            )
            out.append(payload)
        return out

    type_counts = Counter(r["failure_type"] for r in rows)
    rank_bands = Counter()
    for row in rows:
        rank = int(row["target_rank"])
        if rank == 1:
            band = "rank_1"
        elif rank == 2:
            band = "rank_2"
        elif rank <= 5:
            band = "rank_3_5"
        elif rank <= 10:
            band = "rank_6_10"
        elif rank <= 20:
            band = "rank_11_20"
        else:
            band = "rank_gt_20"
        rank_bands[band] += 1
    clean = [r for r in rows if r["target_top1"]]
    rank_le2 = [r for r in rows if int(r["target_rank"]) <= 2]
    return {
        "phase": 767,
        "title": "Commonsense Failure-Type Top-k Audit",
        "model": model_name,
        "round": round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "dtype": "bfloat16",
        "quantization": "off",
        "n_cases": len(rows),
        "top_k": rows[0]["top_k"] if rows else None,
        "failure_type_counts": dict(sorted(type_counts.items())),
        "target_rank_bands": dict(sorted(rank_bands.items())),
        "clean_subset_n": len(clean),
        "rank_le2_subset_n": len(rank_le2),
        "overall": {
            "target_top1_rate": safe_mean([1.0 if r["target_top1"] else 0.0 for r in rows]),
            "exact_target_top1_rate": safe_mean([1.0 if r["exact_target_top1"] else 0.0 for r in rows]),
            "target_in_topk_rate": safe_mean([1.0 if r["target_in_topk"] else 0.0 for r in rows]),
            "mean_target_rank": safe_mean([r["target_rank"] for r in rows]),
            "mean_exact_target_rank": safe_mean([r["exact_target_rank"] for r in rows]),
            "mean_target_allowed_rank": safe_mean([r["target_allowed_rank"] for r in rows]),
            "mean_target_margin_vs_best_other": safe_mean([r["target_margin_vs_best_other"] for r in rows]),
        },
        "by_failure_type": group_counts(["failure_type"]),
        "by_relation_failure_type": group_counts(["relation", "failure_type"]),
        "by_context_failure_type": group_counts(["context_format", "failure_type"]),
        "by_domain_failure_type": group_counts(["domain", "failure_type"]),
    }


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    relation_filter = set(args.relations.split(",")) if args.relations else {"category", "edible", "grows_on_tree"}
    cases = build_cases(args.max_cases, relation_filter)
    for case in cases:
        case["include_candidate_list"] = bool(args.include_candidate_list)
    log(
        f"{args.model}/{args.round_name}: cases={len(cases)} top_k={args.top_k} "
        f"relations={sorted(relation_filter)} include_candidate_list={args.include_candidate_list}"
    )
    model, tokenizer, device, attn_impl = load_model_bf16(args.model, prefer_flash=not args.no_flash)
    try:
        rows = []
        for idx, case in enumerate(cases, 1):
            rows.append(audit_case(model, tokenizer, device, case, args.top_k))
            if idx % args.log_every == 0 or idx == len(cases):
                log(f"{args.model}: top-k audit {idx}/{len(cases)} cases")
    finally:
        release_model(model)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = summarize_rows(args.model, args.round_name, rows, attn_impl)
    for root in (OUT_ROOT, RESULT_ROOT):
        out_dir = root / args.round_name
        write_jsonl(out_dir / f"phase767_{args.model}_rows.jsonl", rows)
        write_json(out_dir / f"phase767_{args.model}_summary.json", summary)
    print(json.dumps({"model": args.model, "round": args.round_name, "summary": summary["overall"]}, ensure_ascii=False, indent=2))
    return summary


def write_cross_summary(round_name: str) -> dict[str, Any]:
    summaries = []
    for model in MODELS:
        path = OUT_ROOT / round_name / f"phase767_{model}_summary.json"
        if path.exists():
            summaries.append(json.loads(path.read_text(encoding="utf-8")))
    payload = {
        "phase": 767,
        "title": "Commonsense Failure-Type Top-k Audit",
        "round": round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "complete" if len(summaries) == len(MODELS) else "partial",
        "models": [s["model"] for s in summaries],
        "by_model": {s["model"]: s for s in summaries},
        "strict_interpretation": (
            "Top-k audit classifies Phase 765 commonsense failures by observed token ranks. "
            "It is still observational; it does not prove the causal source of each failure."
        ),
    }
    lines = [
        f"# Phase 767 Commonsense Failure-Type Top-k Audit ({round_name})",
        "",
        f"- Status: `{payload['status']}`",
        "- Test: Phase 765 commonsense prompts, logits-only top-k audit.",
        "- Quantization: `off`; dtype: `bfloat16`.",
        "",
        "## Overall Reliability",
        "",
        "| model | cases | semantic top1 | exact top1 | in top-k | semantic rank | exact rank | allowed rank | margin | clean n | rank<=2 n |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        overall = summary["overall"]
        lines.append(
            f"| {summary['model']} | {summary['n_cases']} | {fmt(overall['target_top1_rate'])} | "
            f"{fmt(overall['exact_target_top1_rate'])} | {fmt(overall['target_in_topk_rate'])} | "
            f"{fmt(overall['mean_target_rank'])} | {fmt(overall['mean_exact_target_rank'])} | "
            f"{fmt(overall['mean_target_allowed_rank'])} | {fmt(overall['mean_target_margin_vs_best_other'])} | "
            f"{summary['clean_subset_n']} | {summary['rank_le2_subset_n']} |"
        )
    lines += [
        "",
        "## Failure-Type Counts",
        "",
        "| model | failure type | n | semantic top1 | exact top1 | in top-k | semantic rank | exact rank | allowed rank | margin |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        for row in summary["by_failure_type"]:
            lines.append(
                f"| {summary['model']} | `{row['failure_type']}` | {row['n']} | {fmt(row['target_top1_rate'])} | "
                f"{fmt(row['exact_target_top1_rate'])} | {fmt(row['target_in_topk_rate'])} | "
                f"{fmt(row['mean_target_rank'])} | {fmt(row['mean_exact_target_rank'])} | "
                f"{fmt(row['mean_target_allowed_rank'])} | {fmt(row['mean_target_margin_vs_best_other'])} |"
            )
    lines += [
        "",
        "## Relation By Failure Type",
        "",
        "| model | relation | failure type | n | top1 | in top-k | rank | allowed rank |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        for row in summary["by_relation_failure_type"]:
            lines.append(
                f"| {summary['model']} | `{row['relation']}` | `{row['failure_type']}` | {row['n']} | "
                f"{fmt(row['target_top1_rate'])} | {fmt(row['target_in_topk_rate'])} | "
                f"{fmt(row['mean_target_rank'])} | {fmt(row['mean_target_allowed_rank'])} |"
            )
    lines += [
        "",
        "## Strict Interpretation",
        "",
        "- `semantic top1` merges simple lexical aliases such as `yes/Yes/YES`; `exact top1` is the stricter first-token match.",
        "- `success_top1` is the semantic clean subset proxy for prediction-sufficient state.",
        "- `readout_threshold_miss` means the target was rank 2: close to closure, but not closed.",
        "- `allowed_value_candidate_competition` means the allowed value set favored another candidate.",
        "- `knowledge_or_state_formation_miss` means the target did not appear in top-k and cannot be used as a reliable mechanism sample.",
        "- `format_protocol_miss` can be identified only when the top token is visibly format-like; broader protocol failures need generation traces.",
    ]
    markdown = "\n".join(lines) + "\n"
    for root in (OUT_ROOT, RESULT_ROOT):
        out_dir = root / round_name
        write_json(out_dir / "phase767_cross_model_summary.json", payload)
        (out_dir / "phase767_cross_model_summary.md").write_text(markdown, encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="smoke")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--relations", default="category,edible,grows_on_tree")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--include-candidate-list", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-every", type=int, default=18)
    parser.add_argument("--no-flash", action="store_true")
    parser.add_argument("--write-cross-summary", action="store_true")
    parser.add_argument("--hard-exit-after-model", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        cases = build_cases(args.max_cases, set(args.relations.split(",")) if args.relations else None)
        print(json.dumps({"n_cases": len(cases), "sample_cases": cases[:6]}, ensure_ascii=False, indent=2))
        return
    if args.write_cross_summary:
        write_cross_summary(args.round_name)
        return
    if not args.model:
        raise SystemExit("--model is required unless --write-cross-summary or --dry-run")
    run_model(args)
    if args.hard_exit_after_model:
        raise SystemExit(0)


if __name__ == "__main__":
    main()
