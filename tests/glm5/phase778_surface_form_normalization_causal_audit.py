#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import os
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

from phase722_functional_head_atlas_causal_ablation import logit_diag, write_json, write_jsonl  # noqa: E402
from phase735_source_restricted_writer_validation import MODELS, release_model, select_evenly  # noqa: E402
from phase755_cross_domain_route_invariance_atlas import get_first_token_id  # noqa: E402
from phase765_commonsense_context_identity_closure_test import prompt_for_case, question_for, relation_label  # noqa: E402
from phase771_matched_causal_intervention_reliability_test import case_map_for, margin  # noqa: E402
from phase773_instruction_source_disentanglement import fmt, load_jsonl  # noqa: E402
from phase775_semantic_latent_route_output_closure import pool_diag, prompt_for_variant, value_pool  # noqa: E402
from phase776_readout_bridge_competition_audit import (  # noqa: E402
    load_model_bf16_prefer_flash,
    normalize_token_text,
    run_next_logits,
    topk_competitors,
)


OUT_ROOT = Path("results/glm5_phase778_surface_form_normalization_causal_audit")
RESULT_ROOT = Path("tests/result/phase778_surface_form_normalization_causal_audit")
PHASE776_ROOT = Path("tests/result/phase776_readout_bridge_competition_audit")

PROMPT_VARIANTS = [
    "without_candidate_list",
    "constrained_free_prompt",
    "lowercase_short_value",
    "lowercase_no_punctuation",
    "token_identity_contract",
    "with_candidate_list",
]
SEMANTIC_EQUIV_CLASSES = {"target_value", "case_variant_target_value"}


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


def norm_text(value: Any) -> str:
    text = "" if value is None else str(value)
    return normalize_token_text(text).strip().lower()


def semantic_equiv(row: dict[str, Any]) -> bool:
    if row.get("top1_competitor_class") in SEMANTIC_EQUIV_CLASSES:
        return True
    return norm_text(row.get("top1_token_text_norm")) == norm_text(row.get("target_answer"))


def surface_prompt_for_variant(case: dict[str, Any], prompt_variant: str) -> str:
    if prompt_variant in {"without_candidate_list", "constrained_free_prompt", "with_candidate_list"}:
        return prompt_for_variant(case, prompt_variant)

    item = dict(case)
    item["include_candidate_list"] = False
    obj = item["object"]
    relation = item["relation"]
    if item["context_format"] == "commonsense_question":
        task_line = f"Question: {question_for(obj, relation)}"
    else:
        task_line = f"Task: For {obj}, give {relation_label(relation)}."

    if prompt_variant == "lowercase_short_value":
        return (
            "Answer using common everyday knowledge.\n"
            "Write exactly one short lowercase value.\n"
            "Do not use capital letters or an explanation.\n"
            f"{task_line}\n"
            "Answer:"
        )
    if prompt_variant == "lowercase_no_punctuation":
        return (
            "Answer using common everyday knowledge.\n"
            "Output exactly one lowercase value.\n"
            "No capital letters, no punctuation, no sentence.\n"
            f"{task_line}\n"
            "Answer:"
        )
    if prompt_variant == "token_identity_contract":
        return (
            "Answer using common everyday knowledge.\n"
            "Return only the canonical lowercase answer token after the colon.\n"
            "Do not add spaces, punctuation, capitals, or extra words.\n"
            f"{task_line}\n"
            "Answer:"
        )
    raise ValueError(prompt_variant)


def phase776_rows_path(model: str, round_name: str) -> Path:
    path = PHASE776_ROOT / round_name / f"phase776_{model}_rows.jsonl"
    if path.exists():
        return path
    return Path("results/glm5_phase776_readout_bridge_competition_audit") / round_name / f"phase776_{model}_rows.jsonl"


def load_phase776_observations(model: str, round_name: str) -> list[dict[str, Any]]:
    rows = []
    for row in load_jsonl(phase776_rows_path(model, round_name)):
        if row.get("row_kind") == "readout_bridge_observation":
            rows.append(row)
    return rows


def select_surface_cases(model: str, args: argparse.Namespace) -> list[dict[str, Any]]:
    source_prompts = {x.strip() for x in args.source_prompt_variants.split(",") if x.strip()}
    candidates = []
    for row in load_phase776_observations(model, args.source_phase776_round):
        if row.get("prompt_variant") not in source_prompts:
            continue
        if row.get("top1_competitor_class") != "case_variant_target_value":
            continue
        if row.get("base_target_top1"):
            continue
        candidates.append(row)

    seen = set()
    unique = []
    candidates.sort(
        key=lambda r: (
            bool(r.get("latent_pool_hit")),
            float(r.get("top1_gap_above_target") or 0.0),
            str(r.get("case_id")),
        ),
        reverse=True,
    )
    for row in candidates:
        cid = row["case_id"]
        if cid in seen:
            continue
        seen.add(cid)
        unique.append(row)
    if args.max_cases and len(unique) > args.max_cases:
        unique = [unique[i] for i in select_evenly(len(unique), args.max_cases)]
    return unique


def observe_prompt(
    tokenizer,
    logits: torch.Tensor,
    case: dict[str, Any],
    prompt_variant: str,
    source_row: dict[str, Any],
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
    base = {
        "row_kind": "surface_form_observation",
        "case_id": case["case_id"],
        "source_prompt_variant": source_row.get("prompt_variant"),
        "source_top1_competitor_class": source_row.get("top1_competitor_class"),
        "source_top1_token_text": source_row.get("top1_token_text"),
        "source_top1_token_text_norm": source_row.get("top1_token_text_norm"),
        "source_base_target_rank": source_row.get("base_target_rank"),
        "source_top1_gap_above_target": source_row.get("top1_gap_above_target"),
        "source_latent_pool_hit": bool(source_row.get("latent_pool_hit")),
        "prompt_variant": prompt_variant,
        "object": case["object"],
        "domain": case["domain"],
        "relation": case["relation"],
        "context_format": case["context_format"],
        "target_answer": case["answer"],
        "contrast_answer": case["contrast_answer"],
        "target_token_id": target_id,
        "contrast_token_id": contrast_id,
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
    base["semantic_equiv_open"] = semantic_equiv(base)
    base["surface_gain_vs_strict"] = bool(base["semantic_equiv_open"]) and not bool(base["base_target_top1"])
    base["hard_readout_after_equiv"] = bool(base["pool_target_top1"]) and not bool(base["semantic_equiv_open"])
    base["strict_repaired_from_source"] = bool(base["base_target_top1"])
    comp_rows = []
    for row in top_rows:
        comp_rows.append(
            {
                "row_kind": "surface_form_topk_competitor",
                **{k: v for k, v in base.items() if k != "row_kind"},
                **row,
            }
        )
    return base, comp_rows


def audit_case(model, tokenizer, device, args: argparse.Namespace, case: dict[str, Any], source_row: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for prompt_variant in [x.strip() for x in args.prompt_variants.split(",") if x.strip()]:
        prompt = surface_prompt_for_variant(case, prompt_variant)
        logits = run_next_logits(model, tokenizer, device, prompt)
        obs, comps = observe_prompt(tokenizer, logits, case, prompt_variant, source_row, args.top_k)
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
                "strict_open_rate": sum(1 for v in vals if v.get("base_target_top1")) / len(vals) if vals else None,
                "semantic_equiv_open_rate": sum(1 for v in vals if v.get("semantic_equiv_open")) / len(vals) if vals else None,
                "surface_gain_rate": sum(1 for v in vals if v.get("surface_gain_vs_strict")) / len(vals) if vals else None,
                "pool_top1_rate": sum(1 for v in vals if v.get("pool_target_top1")) / len(vals) if vals else None,
                "case_variant_top1_rate": sum(1 for v in vals if v.get("top1_competitor_class") == "case_variant_target_value") / len(vals)
                if vals
                else None,
                "hard_readout_after_equiv_rate": sum(1 for v in vals if v.get("hard_readout_after_equiv")) / len(vals) if vals else None,
                "mean_base_target_rank": safe_mean([v.get("base_target_rank") for v in vals]),
                "mean_top1_gap_above_target": safe_mean([v.get("top1_gap_above_target") for v in vals]),
            }
        )
        out.append(payload)
    out.sort(key=lambda r: (r.get("prompt_variant") or "", r.get("semantic_equiv_open_rate") or 0.0), reverse=True)
    return out


def group_top1(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    obs = [r for r in rows if r.get("row_kind") == "surface_form_observation"]
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in obs:
        groups[(row.get("prompt_variant"), row.get("top1_competitor_class"))].append(row)
    out = []
    for (variant, cls), vals in groups.items():
        out.append(
            {
                "prompt_variant": variant,
                "competitor_class": cls,
                "n": len(vals),
                "case_n": len({v["case_id"] for v in vals}),
                "strict_open_rate": sum(1 for v in vals if v.get("base_target_top1")) / len(vals) if vals else None,
                "semantic_equiv_open_rate": sum(1 for v in vals if v.get("semantic_equiv_open")) / len(vals) if vals else None,
                "mean_gap_above_target": safe_mean([v.get("top1_gap_above_target") for v in vals]),
            }
        )
    out.sort(key=lambda r: (r["n"], r.get("mean_gap_above_target") or 0.0), reverse=True)
    return out


def summarize_rows(rows: list[dict[str, Any]], args: argparse.Namespace, model_name: str, attn_impl: str) -> dict[str, Any]:
    observations = [r for r in rows if r.get("row_kind") == "surface_form_observation"]
    competitors = [r for r in rows if r.get("row_kind") == "surface_form_topk_competitor"]
    return {
        "phase": 778,
        "title": "Surface-Form Normalization Causal Audit",
        "model": model_name,
        "round": args.round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "dtype": "bfloat16",
        "quantization": "off",
        "source_phase776_round": args.source_phase776_round,
        "prompt_variants": [v.strip() for v in args.prompt_variants.split(",") if v.strip()],
        "n_rows": len(rows),
        "n_observation_rows": len(observations),
        "n_competitor_rows": len(competitors),
        "n_cases": len({r["case_id"] for r in observations}),
        "selected_source": "phase776 strict-fail case_variant_target_value observations",
        "by_prompt_observation": group_observations(observations, ["prompt_variant"]),
        "by_domain_prompt": group_observations(observations, ["domain", "prompt_variant"]),
        "top1_competitor_by_prompt": group_top1(observations),
        "strict_interpretation": (
            "Prompt-level surface-form interventions can indicate whether surface instructions repair strict token identity; "
            "they do not identify the internal head/MLP component."
        ),
    }


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = OUT_ROOT / args.round_name
    result_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    selected = select_surface_cases(args.model, args)
    log(f"{args.model}/{args.round_name}: selected surface cases={len(selected)} variants={args.prompt_variants}")
    cmap = case_map_for(args)
    model, tokenizer, device, attn_impl = load_model_bf16_prefer_flash(args.model, args.attn_implementations)
    try:
        rows: list[dict[str, Any]] = []
        for idx, source_row in enumerate(selected, 1):
            case = cmap[source_row["case_id"]]
            rows.extend(audit_case(model, tokenizer, device, args, case, source_row))
            if idx % args.log_every == 0 or idx == len(selected):
                log(f"{args.model}: surface-form audit {idx}/{len(selected)} cases; rows={len(rows)}")
    finally:
        release_model(model)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    summary = summarize_rows(rows, args, args.model, attn_impl)
    for root in (out_dir, result_dir):
        write_jsonl(root / f"phase778_{args.model}_rows.jsonl", rows)
        write_json(root / f"phase778_{args.model}_summary.json", summary)
    print(
        json.dumps(
            {
                "model": args.model,
                "round": args.round_name,
                "attn": attn_impl,
                "n_cases": summary["n_cases"],
                "by_prompt_observation": summary["by_prompt_observation"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return summary


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# Phase 778 Surface-Form Normalization Causal Audit ({payload['round']})",
        "",
        f"- Status: `{payload['status']}`",
        "- Test: prompt-level surface-form interventions on Phase 776 case-variant strict failures.",
        "- Models are run sequentially; bf16, quantization off; attention implementation prefers flash/sdpa and falls back to eager.",
        "",
        "## Prompt Observation Summary",
        "",
        "| model | variant | rows | cases | strict open | semantic-equiv open | surface gain | pool top1 | case-variant top1 | hard readout after equiv | base rank | top1 gap |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        data = payload["by_model"].get(model)
        if not data:
            continue
        for row in data["by_prompt_observation"]:
            lines.append(
                f"| {model} | `{row['prompt_variant']}` | {row['n']} | {row['case_n']} | "
                f"{fmt(row['strict_open_rate'])} | {fmt(row['semantic_equiv_open_rate'])} | "
                f"{fmt(row['surface_gain_rate'])} | {fmt(row['pool_top1_rate'])} | "
                f"{fmt(row['case_variant_top1_rate'])} | {fmt(row['hard_readout_after_equiv_rate'])} | "
                f"{fmt(row['mean_base_target_rank'])} | {fmt(row['mean_top1_gap_above_target'])} |"
            )
    lines += [
        "",
        "## Top1 Competitor Classes",
        "",
        "| model | variant | class | rows | cases | strict open | semantic-equiv open | mean gap above target |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        data = payload["by_model"].get(model)
        if not data:
            continue
        for row in data["top1_competitor_by_prompt"]:
            lines.append(
                f"| {model} | `{row['prompt_variant']}` | `{row['competitor_class']}` | {row['n']} | {row['case_n']} | "
                f"{fmt(row['strict_open_rate'])} | {fmt(row['semantic_equiv_open_rate'])} | {fmt(row['mean_gap_above_target'])} |"
            )
    lines += [
        "",
        "## Strict Interpretation",
        "",
        "- This is a prompt-level causal audit, not a component-level localization.",
        "- If lowercase/token-identity instructions repair strict open closure, the missing layer is at least partly surface-form normalization.",
        "- If candidate_list still dominates, candidate_list supplies more than a generic lowercase instruction.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_cross_summary(round_name: str) -> dict[str, Any]:
    by_model = {}
    for model in MODELS:
        path = OUT_ROOT / round_name / f"phase778_{model}_summary.json"
        if path.exists():
            by_model[model] = json.loads(path.read_text(encoding="utf-8"))
    payload = {
        "phase": 778,
        "title": "Surface-Form Normalization Causal Audit",
        "round": round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "complete" if len(by_model) == len(MODELS) else "partial",
        "models": list(by_model),
        "by_model": by_model,
    }
    for root in (OUT_ROOT, RESULT_ROOT):
        out_dir = root / round_name
        write_json(out_dir / "phase778_cross_model_summary.json", payload)
        write_markdown(out_dir / "phase778_cross_model_summary.md", payload)
    print(json.dumps({"round": round_name, "status": payload["status"], "models": payload["models"]}, ensure_ascii=False, indent=2), flush=True)
    return payload


def dry_run(args: argparse.Namespace) -> None:
    payload = {"round": args.round_name, "source_phase776_round": args.source_phase776_round, "models": {}}
    for model in MODELS:
        args.model = model
        selected = select_surface_cases(model, args)
        payload["models"][model] = {
            "selected_cases": len(selected),
            "domains": dict(Counter(r.get("domain") for r in selected)),
            "source_prompt_variants": dict(Counter(r.get("prompt_variant") for r in selected)),
            "prompt_variants": [v.strip() for v in args.prompt_variants.split(",") if v.strip()],
        }
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="smoke")
    parser.add_argument("--source-phase776-round", default="confirm")
    parser.add_argument("--source-prompt-variants", default="without_candidate_list,constrained_free_prompt,with_candidate_list")
    parser.add_argument("--prompt-variants", default="without_candidate_list,constrained_free_prompt,lowercase_short_value,lowercase_no_punctuation,token_identity_contract,with_candidate_list")
    parser.add_argument("--relations", default="category,edible,grows_on_tree")
    parser.add_argument("--max-cases", type=int, default=8)
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
