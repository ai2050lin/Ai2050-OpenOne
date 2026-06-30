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

from model_utils import get_layers, release_model  # noqa: E402
from phase722_functional_head_atlas_causal_ablation import logit_diag, write_json, write_jsonl  # noqa: E402
from phase735_source_restricted_writer_validation import MODELS, load_model_bf16_eager, select_evenly  # noqa: E402
from phase739_readout_threshold_closure_boundary import get_unembed  # noqa: E402
from phase755_cross_domain_route_invariance_atlas import get_first_token_id  # noqa: E402
from phase765_commonsense_context_identity_closure_test import capture_state, route_ids_for_case  # noqa: E402
from phase771_matched_causal_intervention_reliability_test import (  # noqa: E402
    case_map_for,
    margin,
    pair_info_map,
    phase770_path,
    phase767_path,
    select_matched_case_ids,
    semantic_label,
)
from phase773_instruction_source_disentanglement import (  # noqa: E402
    DEFAULT_SCAN_LAYERS,
    DEFAULT_SOURCE_GROUPS,
    add_controls,
    build_disentangled_source_groups,
    causal_test_candidate,
    direct_candidates_for_case,
    fmt,
    group_summary,
    load_json,
    load_jsonl,
    scan_layers_for,
    select_candidates,
    source_groups_for,
)


OUT_ROOT = Path("results/glm5_phase774_candidate_list_ablation_transfer")
RESULT_ROOT = Path("tests/result/phase774_candidate_list_ablation_transfer")
PROMPT_VARIANTS = ["with_candidate_list", "without_candidate_list"]


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


def focus_filter(selected: list[dict[str, Any]], pair_info: dict[str, dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.focus == "all_matched":
        out = selected
    elif args.focus == "clean":
        out = [x for x in selected if x["matched_arm"] == "clean"]
    elif args.focus == "fiber_high":
        out = [x for x in selected if pair_info.get(x["case_id"], {}).get("fiber_bucket") == "fiber_high"]
    elif args.focus == "clean_fiber_high":
        out = [
            x
            for x in selected
            if x["matched_arm"] == "clean" and pair_info.get(x["case_id"], {}).get("fiber_bucket") == "fiber_high"
        ]
    else:
        raise ValueError(args.focus)
    if args.max_cases and len(out) > args.max_cases:
        out = [out[i] for i in select_evenly(len(out), args.max_cases)]
    return out


def variant_cases(case: dict[str, Any], variants: list[str]) -> list[tuple[str, dict[str, Any]]]:
    out = []
    for variant in variants:
        item = dict(case)
        item["include_candidate_list"] = variant == "with_candidate_list"
        out.append((variant, item))
    return out


def base_payload(state: dict[str, Any], target_id: int, contrast_id: int) -> dict[str, Any]:
    target_diag = logit_diag(state["logits"], target_id)
    contrast_diag = logit_diag(state["logits"], contrast_id)
    return {
        "base_target_rank": target_diag["target_rank"],
        "base_target_top1": bool(target_diag["target_top1"]),
        "base_contrast_rank": contrast_diag["target_rank"],
        "base_margin_target_vs_contrast": margin(state["logits"], target_id, contrast_id),
    }


def audit_variant(
    model,
    tokenizer,
    device,
    args: argparse.Namespace,
    case: dict[str, Any],
    case_label: dict[str, Any],
    pair_info: dict[str, Any],
    prompt_variant: str,
    scan_layers: list[int],
    source_groups: list[str],
    unembed: torch.Tensor,
) -> list[dict[str, Any]]:
    state = capture_state(model, tokenizer, device, case, scan_layers)
    state["source_groups"] = build_disentangled_source_groups(tokenizer, state["prompt"], case, state["ids"])
    target_id = get_first_token_id(tokenizer, case["answer"])
    contrast_id = get_first_token_id(tokenizer, case["contrast_answer"])
    route_ids = route_ids_for_case(tokenizer, case, target_id)
    base = base_payload(state, target_id, contrast_id)
    direct_rows = direct_candidates_for_case(model, state, source_groups, scan_layers, target_id, route_ids, args, unembed)
    selected = select_candidates(direct_rows, args)
    selected = add_controls(model, selected, direct_rows, args)
    phase767 = case_label["phase767"]
    out = []
    for rank, row in enumerate(selected, 1):
        causal = causal_test_candidate(model, device, state, row, target_id, contrast_id)
        out.append(
            {
                "row_kind": "candidate_list_ablation_transfer",
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
                **base,
                "selection_rank": rank,
                "site": row["site"],
                "layer": row["layer"],
                "head": row["head"],
                "component_key": f"{row['site']}:H{row['head']}:{row['source_group']}",
                "source_group": row["source_group"],
                "source_family": row["source_family"],
                "source_positions_n": row["source_positions_n"],
                "candidate_kind": row["candidate_kind"],
                "selection": row["selection"],
                "control_of": row.get("control_of"),
                "scan_score": row["scan_score"],
                "attention_mass_to_source": row["attention_mass_to_source"],
                "source_direct_score": row["source_direct_score"],
                **causal,
            }
        )
    if not selected:
        out.append(
            {
                "row_kind": "candidate_list_ablation_observation",
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
                **pair_info,
                **base,
            }
        )
    return out


def audit_case(
    model,
    tokenizer,
    device,
    args: argparse.Namespace,
    case: dict[str, Any],
    case_label: dict[str, Any],
    pair_info: dict[str, Any],
    scan_layers: list[int],
    source_groups: list[str],
    unembed: torch.Tensor,
) -> list[dict[str, Any]]:
    variants = [v.strip() for v in args.prompt_variants.split(",") if v.strip()]
    rows: list[dict[str, Any]] = []
    for prompt_variant, variant_case in variant_cases(case, variants):
        rows.extend(audit_variant(model, tokenizer, device, args, variant_case, case_label, pair_info, prompt_variant, scan_layers, source_groups, unembed))
    return rows


def group_with_base(rows: list[dict[str, Any]], key_fields: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(k) for k in key_fields)].append(row)
    out = []
    for key, vals in sorted(groups.items(), key=lambda kv: str(kv[0])):
        direct = [v.get("source_direct_score") or {} for v in vals]
        payload = {field: value for field, value in zip(key_fields, key)}
        payload.update(
            {
                "n": len(vals),
                "case_n": len({v["case_id"] for v in vals}),
                "base_top1_rate": sum(1 for v in vals if v.get("base_target_top1")) / len(vals) if vals else None,
                "mean_base_target_rank": safe_mean([v.get("base_target_rank") for v in vals]),
                "mean_base_margin": safe_mean([v.get("base_margin_target_vs_contrast") for v in vals]),
                "mean_scan_score": safe_mean([v.get("scan_score") for v in vals]),
                "mean_target_logit_drop": safe_mean([v.get("target_logit_drop") for v in vals]),
                "mean_margin_drop_target_vs_contrast": safe_mean([v.get("margin_drop_target_vs_contrast") for v in vals]),
                "mean_attention_mass": safe_mean([v.get("attention_mass_to_source") for v in vals]),
                "mean_direct_target_boost": safe_mean([d.get("direct_target_boost") for d in direct]),
                "mean_direct_route_suppression": safe_mean([d.get("direct_total_route_suppression") for d in direct]),
                "top1_loss_rate": sum(1 for v in vals if v.get("top1_loss")) / len(vals) if vals else None,
            }
        )
        out.append(payload)
    out.sort(
        key=lambda r: (
            r.get("prompt_variant") or "",
            r.get("top1_loss_rate") or 0.0,
            r.get("mean_target_logit_drop") or 0.0,
            r.get("mean_margin_drop_target_vs_contrast") or 0.0,
        ),
        reverse=True,
    )
    return out


def summarize_rows(rows: list[dict[str, Any]], args: argparse.Namespace, model_name: str, attn_impl: str, scan_layers: list[int]) -> dict[str, Any]:
    effect_rows = [r for r in rows if r.get("candidate_kind")]
    return {
        "phase": 774,
        "title": "Candidate-List Ablation and Free-Semantic Transfer",
        "model": model_name,
        "round": args.round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "dtype": "bfloat16",
        "quantization": "off",
        "focus": args.focus,
        "prompt_variants": [v.strip() for v in args.prompt_variants.split(",") if v.strip()],
        "scan_layers": scan_layers,
        "n_rows": len(rows),
        "n_effect_rows": len(effect_rows),
        "n_cases": len({r["case_id"] for r in rows}),
        "n_pairs": len({r["pair_index"] for r in rows}),
        "source_groups": source_groups_for(args),
        "by_prompt_variant": group_with_base(effect_rows, ["prompt_variant", "candidate_kind"]),
        "by_prompt_source_family": group_with_base(effect_rows, ["prompt_variant", "source_family", "candidate_kind"]),
        "by_prompt_source_group": group_with_base(effect_rows, ["prompt_variant", "source_group", "candidate_kind"]),
        "top_components": group_with_base(effect_rows, ["prompt_variant", "component_key", "candidate_kind"])[:50],
        "strict_interpretation": "This phase compares allowed-values prompts to no-candidate-list prompts. It is still head/source-level.",
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
    source_groups = source_groups_for(args)
    log(f"{args.model}/{args.round_name}: focus={args.focus} cases={len(selected)} variants={args.prompt_variants} sources={source_groups}")
    model, tokenizer, device, attn_impl = load_model_bf16_eager(args.model)
    try:
        scan_layers = scan_layers_for(args, len(get_layers(model)))
        unembed = get_unembed(model)
        rows: list[dict[str, Any]] = []
        for idx, item in enumerate(selected, 1):
            case = cmap[item["case_id"]]
            rows.extend(audit_case(model, tokenizer, device, args, case, item, pinfo.get(case["case_id"], {}), scan_layers, source_groups, unembed))
            if idx % args.log_every == 0 or idx == len(selected):
                log(f"{args.model}: candidate-list ablation {idx}/{len(selected)} cases; rows={len(rows)}")
    finally:
        release_model(model)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    summary = summarize_rows(rows, args, args.model, attn_impl, scan_layers)
    for root in (out_dir, result_dir):
        write_jsonl(root / f"phase774_{args.model}_rows.jsonl", rows)
        write_json(root / f"phase774_{args.model}_summary.json", summary)
    print(json.dumps({"model": args.model, "round": args.round_name, "n_cases": summary["n_cases"], "by_prompt_variant": summary["by_prompt_variant"][:6]}, ensure_ascii=False, indent=2), flush=True)
    return summary


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# Phase 774 Candidate-List Ablation ({payload['round']})",
        "",
        f"- Status: `{payload['status']}`",
        "- Test: compare prompts with and without allowed-values candidate list.",
        "- Models are run sequentially; bf16, quantization off. Attention extraction requires eager attention.",
        "",
        "## Prompt Variant Summary",
        "",
        "| model | variant | kind | rows | cases | base top1 | base rank | base margin | target drop | margin drop | top1 loss |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        data = payload["by_model"].get(model)
        if not data:
            continue
        for row in data["by_prompt_variant"]:
            lines.append(
                f"| {model} | `{row['prompt_variant']}` | `{row['candidate_kind']}` | {row['n']} | {row['case_n']} | "
                f"{fmt(row['base_top1_rate'])} | {fmt(row['mean_base_target_rank'])} | {fmt(row['mean_base_margin'])} | "
                f"{fmt(row['mean_target_logit_drop'])} | {fmt(row['mean_margin_drop_target_vs_contrast'])} | {fmt(row['top1_loss_rate'])} |"
            )
    lines += [
        "",
        "## Prompt Source Family Summary",
        "",
        "| model | variant | family | kind | rows | cases | target drop | margin drop | direct boost | route suppression |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        data = payload["by_model"].get(model)
        if not data:
            continue
        for row in data["by_prompt_source_family"]:
            lines.append(
                f"| {model} | `{row['prompt_variant']}` | `{row['source_family']}` | `{row['candidate_kind']}` | {row['n']} | {row['case_n']} | "
                f"{fmt(row['mean_target_logit_drop'])} | {fmt(row['mean_margin_drop_target_vs_contrast'])} | "
                f"{fmt(row['mean_direct_target_boost'])} | {fmt(row['mean_direct_route_suppression'])} |"
            )
    lines += [
        "",
        "## Strict Interpretation",
        "",
        "- If removing the candidate list collapses base output and object/relation effects do not rise, the previous atlas mostly describes candidate-conditioned closure.",
        "- If object/relation sources strengthen without the candidate list, the route can move toward free semantic closure.",
        "- Causal removal remains head/source-level, not neuron/channel-level.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_cross_summary(round_name: str) -> dict[str, Any]:
    by_model = {}
    for model in MODELS:
        path = OUT_ROOT / round_name / f"phase774_{model}_summary.json"
        if path.exists():
            by_model[model] = load_json(path)
    payload = {
        "phase": 774,
        "title": "Candidate-List Ablation and Free-Semantic Transfer",
        "round": round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "complete" if len(by_model) == len(MODELS) else "partial",
        "models": list(by_model),
        "by_model": by_model,
    }
    for root in (OUT_ROOT, RESULT_ROOT):
        out_dir = root / round_name
        write_json(out_dir / "phase774_cross_model_summary.json", payload)
        write_markdown(out_dir / "phase774_cross_model_summary.md", payload)
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
            "scan_layers": DEFAULT_SCAN_LAYERS[model][: args.max_scan_layers] if args.max_scan_layers else DEFAULT_SCAN_LAYERS[model],
            "source_groups": source_groups_for(args),
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
    parser.add_argument("--prompt-variants", default="with_candidate_list,without_candidate_list")
    parser.add_argument("--max-per-stratum", type=int, default=1)
    parser.add_argument("--max-pairs", type=int, default=8)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--scan-layers", default="")
    parser.add_argument("--max-scan-layers", type=int, default=4)
    parser.add_argument("--source-groups", default="")
    parser.add_argument("--max-source-groups", type=int, default=8)
    parser.add_argument("--top-global-components-per-case", type=int, default=0)
    parser.add_argument("--top-components-per-source", type=int, default=1)
    parser.add_argument("--include-controls", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--control-offset", type=int, default=5)
    parser.add_argument("--target-weight", type=float, default=1.0)
    parser.add_argument("--route-weight", type=float, default=0.5)
    parser.add_argument("--margin-weight", type=float, default=0.5)
    parser.add_argument("--attention-weight", type=float, default=0.05)
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
