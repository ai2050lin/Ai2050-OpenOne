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
from phase112_attention_transport_head_mapping_cuda import get_attention_module, get_num_heads  # noqa: E402
from phase132_source_value_contribution_cuda import compute_source_contribution, get_num_kv_heads, get_v_proj  # noqa: E402
from phase722_functional_head_atlas_causal_ablation import logit_diag, write_json, write_jsonl  # noqa: E402
from phase735_source_restricted_writer_validation import MODELS, load_model_bf16_eager, select_evenly  # noqa: E402
from phase739_readout_threshold_closure_boundary import get_unembed  # noqa: E402
from phase751_natural_attention_head_mechanism_backtrace import install_source_contribution_removal, project_source_contribution  # noqa: E402
from phase752_natural_writer_stability_path_chain import attention_mass_for_group  # noqa: E402
from phase755_cross_domain_route_invariance_atlas import get_first_token_id  # noqa: E402
from phase756_cross_domain_writer_control_downstream_carrier import run_logits  # noqa: E402
from phase762_semantic_numeric_fiber_atlas import VALUE_POOLS  # noqa: E402
from phase765_commonsense_context_identity_closure_test import prompt_for_case, question_for, relation_label, route_ids_for_case  # noqa: E402
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
    direct_candidates_for_case,
    fmt,
    load_json,
    load_jsonl,
    scan_layers_for,
    select_candidates,
    source_groups_for,
)


OUT_ROOT = Path("results/glm5_phase775_semantic_latent_route_output_closure")
RESULT_ROOT = Path("tests/result/phase775_semantic_latent_route_output_closure")
PROMPT_VARIANTS = ["without_candidate_list", "constrained_free_prompt", "with_candidate_list"]


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


def prompt_for_variant(case: dict[str, Any], prompt_variant: str) -> str:
    item = dict(case)
    if prompt_variant == "with_candidate_list":
        item["include_candidate_list"] = True
        return prompt_for_case(item)
    if prompt_variant == "without_candidate_list":
        item["include_candidate_list"] = False
        return prompt_for_case(item)
    if prompt_variant == "constrained_free_prompt":
        item["include_candidate_list"] = False
        obj = item["object"]
        relation = item["relation"]
        if item["context_format"] == "commonsense_question":
            return (
                "Answer using common everyday knowledge.\n"
                "Use exactly one short value.\n"
                "Do not write a sentence or explanation.\n"
                f"Question: {question_for(obj, relation)}\n"
                "Answer:"
            )
        if item["context_format"] == "commonsense_statement":
            return (
                "Use common everyday knowledge.\n"
                "Answer with exactly one short value.\n"
                "Do not write a sentence or explanation.\n"
                f"Task: For {obj}, give {relation_label(relation)}.\n"
                "Answer:"
            )
    raise ValueError(prompt_variant)


def capture_state_for_prompt(model, tokenizer, device, case: dict[str, Any], prompt_variant: str, layers: list[int]) -> dict[str, Any]:
    prompt = prompt_for_variant(case, prompt_variant)
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    handles = []
    value_store: dict[int, torch.Tensor] = {}
    for li in sorted(set(layers)):
        attn = get_attention_module(get_layers(model)[li])
        v_proj = get_v_proj(attn)

        def v_hook(_module, _inputs, output, li=li):
            value_store[li] = output.detach().float().cpu()

        handles.append(v_proj.register_forward_hook(v_hook))
    try:
        with torch.inference_mode():
            out = model(
                input_ids=torch.tensor([ids], device=device),
                return_dict=True,
                use_cache=False,
                output_attentions=True,
                output_hidden_states=True,
            )
        attentions = {li: out.attentions[li].detach().float().cpu().numpy() for li in sorted(set(layers))}
        logits = out.logits[0, -1].detach().float().cpu()
        final_hidden = out.hidden_states[-1][0, -1].detach().float().cpu()
    finally:
        for h in handles:
            h.remove()
    return {
        "ids": ids,
        "prompt": prompt,
        "answer_pos": len(ids) - 1,
        "logits": logits,
        "final_hidden": final_hidden,
        "attentions": attentions,
        "values": value_store,
        "source_groups": build_disentangled_source_groups(tokenizer, prompt, case, ids),
    }


def value_pool(tokenizer, case: dict[str, Any], target_id: int, contrast_id: int) -> list[dict[str, Any]]:
    values = list(VALUE_POOLS.get(case["relation_key"], []))
    if case["answer"] not in values:
        values.append(case["answer"])
    if case["contrast_answer"] not in values:
        values.append(case["contrast_answer"])
    out = []
    seen = set()
    for value in values:
        tid = get_first_token_id(tokenizer, str(value))
        if tid in seen:
            continue
        seen.add(tid)
        out.append({"value": str(value), "token_id": tid})
    if target_id not in seen:
        out.append({"value": case["answer"], "token_id": target_id})
    if contrast_id not in seen:
        out.append({"value": case["contrast_answer"], "token_id": contrast_id})
    return out


def pool_diag(logits: torch.Tensor, pool: list[dict[str, Any]], target_id: int, contrast_id: int) -> dict[str, Any]:
    scored = []
    for item in pool:
        tid = int(item["token_id"])
        scored.append((float(logits[tid].item()), tid, item["value"]))
    scored.sort(reverse=True, key=lambda x: x[0])
    rank_by_id = {tid: idx + 1 for idx, (_score, tid, _value) in enumerate(scored)}
    target_score = float(logits[target_id].item())
    best_other = max((score for score, tid, _value in scored if tid != target_id), default=float("-inf"))
    best_score, best_id, best_value = scored[0] if scored else (None, None, None)
    contrast_score = float(logits[contrast_id].item())
    return {
        "pool_size": len(scored),
        "pool_target_rank": rank_by_id.get(target_id),
        "pool_target_top1": rank_by_id.get(target_id) == 1,
        "pool_best_token_id": best_id,
        "pool_best_value": best_value,
        "pool_best_is_target": best_id == target_id,
        "pool_margin_target_vs_best_other": target_score - best_other if math.isfinite(best_other) else None,
        "pool_margin_target_vs_contrast": target_score - contrast_score,
    }


def top_token_payload(tokenizer, logits: torch.Tensor) -> dict[str, Any]:
    top_id = int(torch.argmax(logits).item())
    return {
        "global_top_token_id": top_id,
        "global_top_token_text": tokenizer.decode([top_id], skip_special_tokens=False),
        "global_top_logit": float(logits[top_id].item()),
    }


def base_payload(tokenizer, state: dict[str, Any], pool: list[dict[str, Any]], target_id: int, contrast_id: int) -> dict[str, Any]:
    target_diag = logit_diag(state["logits"], target_id)
    contrast_diag = logit_diag(state["logits"], contrast_id)
    pdiag = pool_diag(state["logits"], pool, target_id, contrast_id)
    return {
        "base_target_rank": target_diag["target_rank"],
        "base_target_top1": bool(target_diag["target_top1"]),
        "base_contrast_rank": contrast_diag["target_rank"],
        "base_margin_target_vs_contrast": margin(state["logits"], target_id, contrast_id),
        "latent_pool_hit": (not bool(target_diag["target_top1"])) and bool(pdiag["pool_target_top1"]),
        **pdiag,
        **top_token_payload(tokenizer, state["logits"]),
    }


def causal_test_latent(
    model,
    device,
    state: dict[str, Any],
    row: dict[str, Any],
    pool: list[dict[str, Any]],
    target_id: int,
    contrast_id: int,
) -> dict[str, Any]:
    layer = int(row["layer"])
    head = int(row["head"])
    source_group = row["source_group"]
    src_positions = [int(p) for p in state["source_groups"].get(source_group, [])]
    answer_pos = state["answer_pos"]
    attn = get_attention_module(get_layers(model)[layer])
    n_heads = get_num_heads(model, attn)
    num_kv_heads = get_num_kv_heads(model, attn, n_heads)
    contribution = compute_source_contribution(
        state["attentions"][layer],
        state["values"][layer],
        [answer_pos],
        [src_positions],
        n_heads,
        num_kv_heads,
    )
    install = install_source_contribution_removal(model, row["site"], [head], contribution)
    after_logits = run_logits(model, device, state["ids"], install)
    base_logits = state["logits"]
    base_target_diag = logit_diag(base_logits, target_id)
    after_target_diag = logit_diag(after_logits, target_id)
    base_margin = margin(base_logits, target_id, contrast_id)
    after_margin = margin(after_logits, target_id, contrast_id)
    base_pool = pool_diag(base_logits, pool, target_id, contrast_id)
    after_pool = pool_diag(after_logits, pool, target_id, contrast_id)
    base_pool_rank = base_pool.get("pool_target_rank")
    after_pool_rank = after_pool.get("pool_target_rank")
    return {
        "after_target_rank": after_target_diag["target_rank"],
        "after_target_top1": after_target_diag["target_top1"],
        "target_rank_delta_after_minus_base": (
            int(after_target_diag["target_rank"]) - int(base_target_diag["target_rank"])
            if after_target_diag["target_rank"] is not None and base_target_diag["target_rank"] is not None
            else None
        ),
        "top1_loss": bool(base_target_diag["target_top1"]) and not bool(after_target_diag["target_top1"]),
        "target_logit_drop": float(base_logits[target_id].item() - after_logits[target_id].item()),
        "contrast_logit_gain": float(after_logits[contrast_id].item() - base_logits[contrast_id].item()),
        "margin_drop_target_vs_contrast": float(base_margin - after_margin),
        "after_pool_target_rank": after_pool_rank,
        "after_pool_target_top1": bool(after_pool.get("pool_target_top1")),
        "pool_rank_delta_after_minus_base": (
            int(after_pool_rank) - int(base_pool_rank) if after_pool_rank is not None and base_pool_rank is not None else None
        ),
        "pool_top1_loss": bool(base_pool.get("pool_target_top1")) and not bool(after_pool.get("pool_target_top1")),
        "pool_margin_drop_target_vs_best_other": (
            float(base_pool["pool_margin_target_vs_best_other"] - after_pool["pool_margin_target_vs_best_other"])
            if base_pool.get("pool_margin_target_vs_best_other") is not None
            and after_pool.get("pool_margin_target_vs_best_other") is not None
            else None
        ),
        "pool_margin_drop_target_vs_contrast": float(
            base_pool["pool_margin_target_vs_contrast"] - after_pool["pool_margin_target_vs_contrast"]
        ),
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
    state = capture_state_for_prompt(model, tokenizer, device, case, prompt_variant, scan_layers)
    target_id = get_first_token_id(tokenizer, case["answer"])
    contrast_id = get_first_token_id(tokenizer, case["contrast_answer"])
    route_ids = route_ids_for_case(tokenizer, case, target_id)
    pool = value_pool(tokenizer, case, target_id, contrast_id)
    base = base_payload(tokenizer, state, pool, target_id, contrast_id)
    phase767 = case_label["phase767"]
    rows: list[dict[str, Any]] = [
        {
            "row_kind": "semantic_latent_observation",
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
        }
    ]
    direct_rows = direct_candidates_for_case(model, state, source_groups, scan_layers, target_id, route_ids, args, unembed)
    selected = select_candidates(direct_rows, args)
    selected = add_controls(model, selected, direct_rows, args)
    for rank, row in enumerate(selected, 1):
        causal = causal_test_latent(model, device, state, row, pool, target_id, contrast_id)
        rows.append(
            {
                "row_kind": "semantic_latent_effect",
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
    return rows


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
    rows: list[dict[str, Any]] = []
    for prompt_variant in [v.strip() for v in args.prompt_variants.split(",") if v.strip()]:
        rows.extend(audit_variant(model, tokenizer, device, args, case, case_label, pair_info, prompt_variant, scan_layers, source_groups, unembed))
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
                "latent_pool_hit_rate": sum(1 for v in vals if v.get("latent_pool_hit")) / len(vals) if vals else None,
                "pool_target_top1_rate": sum(1 for v in vals if v.get("pool_target_top1")) / len(vals) if vals else None,
                "mean_base_target_rank": safe_mean([v.get("base_target_rank") for v in vals]),
                "mean_pool_target_rank": safe_mean([v.get("pool_target_rank") for v in vals]),
                "mean_base_margin": safe_mean([v.get("base_margin_target_vs_contrast") for v in vals]),
                "mean_pool_margin_target_vs_best_other": safe_mean([v.get("pool_margin_target_vs_best_other") for v in vals]),
            }
        )
        out.append(payload)
    out.sort(key=lambda r: (r.get("prompt_variant") or "", r.get("pool_target_top1_rate") or 0.0), reverse=True)
    return out


def group_effects(rows: list[dict[str, Any]], key_fields: list[str]) -> list[dict[str, Any]]:
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
                "mean_target_logit_drop": safe_mean([v.get("target_logit_drop") for v in vals]),
                "mean_margin_drop_target_vs_contrast": safe_mean([v.get("margin_drop_target_vs_contrast") for v in vals]),
                "mean_target_rank_delta": safe_mean([v.get("target_rank_delta_after_minus_base") for v in vals]),
                "mean_pool_rank_delta": safe_mean([v.get("pool_rank_delta_after_minus_base") for v in vals]),
                "mean_pool_margin_drop": safe_mean([v.get("pool_margin_drop_target_vs_best_other") for v in vals]),
                "mean_attention_mass": safe_mean([v.get("attention_mass_to_source") for v in vals]),
                "mean_direct_target_boost": safe_mean([d.get("direct_target_boost") for d in direct]),
                "mean_direct_route_suppression": safe_mean([d.get("direct_total_route_suppression") for d in direct]),
                "top1_loss_rate": sum(1 for v in vals if v.get("top1_loss")) / len(vals) if vals else None,
                "pool_top1_loss_rate": sum(1 for v in vals if v.get("pool_top1_loss")) / len(vals) if vals else None,
            }
        )
        out.append(payload)
    out.sort(
        key=lambda r: (
            r.get("prompt_variant") or "",
            r.get("pool_top1_loss_rate") or 0.0,
            r.get("mean_pool_margin_drop") or 0.0,
            r.get("mean_target_logit_drop") or 0.0,
        ),
        reverse=True,
    )
    return out


def summarize_rows(rows: list[dict[str, Any]], args: argparse.Namespace, model_name: str, attn_impl: str, scan_layers: list[int]) -> dict[str, Any]:
    observations = [r for r in rows if r.get("row_kind") == "semantic_latent_observation"]
    effects = [r for r in rows if r.get("row_kind") == "semantic_latent_effect"]
    return {
        "phase": 775,
        "title": "Semantic Latent Route vs Output Closure Audit",
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
        "n_observation_rows": len(observations),
        "n_effect_rows": len(effects),
        "n_cases": len({r["case_id"] for r in observations}),
        "n_pairs": len({r["pair_index"] for r in observations}),
        "source_groups": source_groups_for(args),
        "by_prompt_observation": group_observations(observations, ["prompt_variant"]),
        "by_prompt_effect": group_effects(effects, ["prompt_variant", "candidate_kind"]),
        "by_prompt_source_family": group_effects(effects, ["prompt_variant", "source_family", "candidate_kind"]),
        "by_prompt_source_group": group_effects(effects, ["prompt_variant", "source_group", "candidate_kind"]),
        "top_components": group_effects(effects, ["prompt_variant", "component_key", "candidate_kind"])[:50],
        "strict_interpretation": "This phase checks value-pool latent selection separately from open-vocabulary output closure.",
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
                log(f"{args.model}: semantic latent audit {idx}/{len(selected)} cases; rows={len(rows)}")
    finally:
        release_model(model)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    summary = summarize_rows(rows, args, args.model, attn_impl, scan_layers)
    for root in (out_dir, result_dir):
        write_jsonl(root / f"phase775_{args.model}_rows.jsonl", rows)
        write_json(root / f"phase775_{args.model}_summary.json", summary)
    print(
        json.dumps(
            {
                "model": args.model,
                "round": args.round_name,
                "n_cases": summary["n_cases"],
                "by_prompt_observation": summary["by_prompt_observation"],
                "by_prompt_effect": summary["by_prompt_effect"][:6],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return summary


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# Phase 775 Semantic Latent Route vs Output Closure ({payload['round']})",
        "",
        f"- Status: `{payload['status']}`",
        "- Test: separate value-pool latent selection from open-vocabulary output closure.",
        "- Models are run sequentially; bf16, quantization off. Attention extraction requires eager attention.",
        "",
        "## Prompt Observation Summary",
        "",
        "| model | variant | rows | cases | base top1 | latent pool hit | pool top1 | base rank | pool rank | base margin | pool margin |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        data = payload["by_model"].get(model)
        if not data:
            continue
        for row in data["by_prompt_observation"]:
            lines.append(
                f"| {model} | `{row['prompt_variant']}` | {row['n']} | {row['case_n']} | "
                f"{fmt(row['base_top1_rate'])} | {fmt(row['latent_pool_hit_rate'])} | {fmt(row['pool_target_top1_rate'])} | "
                f"{fmt(row['mean_base_target_rank'])} | {fmt(row['mean_pool_target_rank'])} | "
                f"{fmt(row['mean_base_margin'])} | {fmt(row['mean_pool_margin_target_vs_best_other'])} |"
            )
    lines += [
        "",
        "## Component Effect Summary",
        "",
        "| model | variant | kind | rows | cases | target drop | margin drop | target rank delta | pool rank delta | pool margin drop | pool top1 loss |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        data = payload["by_model"].get(model)
        if not data:
            continue
        for row in data["by_prompt_effect"]:
            lines.append(
                f"| {model} | `{row['prompt_variant']}` | `{row['candidate_kind']}` | {row['n']} | {row['case_n']} | "
                f"{fmt(row['mean_target_logit_drop'])} | {fmt(row['mean_margin_drop_target_vs_contrast'])} | "
                f"{fmt(row['mean_target_rank_delta'])} | {fmt(row['mean_pool_rank_delta'])} | "
                f"{fmt(row['mean_pool_margin_drop'])} | {fmt(row['pool_top1_loss_rate'])} |"
            )
    lines += [
        "",
        "## Prompt Source Family Summary",
        "",
        "| model | variant | family | kind | rows | cases | target drop | pool margin drop | direct boost | route suppression |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        data = payload["by_model"].get(model)
        if not data:
            continue
        for row in data["by_prompt_source_family"]:
            lines.append(
                f"| {model} | `{row['prompt_variant']}` | `{row['source_family']}` | `{row['candidate_kind']}` | {row['n']} | {row['case_n']} | "
                f"{fmt(row['mean_target_logit_drop'])} | {fmt(row['mean_pool_margin_drop'])} | "
                f"{fmt(row['mean_direct_target_boost'])} | {fmt(row['mean_direct_route_suppression'])} |"
            )
    lines += [
        "",
        "## Strict Interpretation",
        "",
        "- `base top1` measures open-vocabulary output closure.",
        "- `pool top1` measures whether the target wins inside the relation value pool without using that pool as prompt evidence.",
        "- A high `pool top1` with low `base top1` suggests latent semantic selection without readout closure.",
        "- Component removal remains head/source-level and does not prove neuron-level coding.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_cross_summary(round_name: str) -> dict[str, Any]:
    by_model = {}
    for model in MODELS:
        path = OUT_ROOT / round_name / f"phase775_{model}_summary.json"
        if path.exists():
            by_model[model] = load_json(path)
    payload = {
        "phase": 775,
        "title": "Semantic Latent Route vs Output Closure Audit",
        "round": round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "complete" if len(by_model) == len(MODELS) else "partial",
        "models": list(by_model),
        "by_model": by_model,
    }
    for root in (OUT_ROOT, RESULT_ROOT):
        out_dir = root / round_name
        write_json(out_dir / "phase775_cross_model_summary.json", payload)
        write_markdown(out_dir / "phase775_cross_model_summary.md", payload)
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
    parser.add_argument("--prompt-variants", default="without_candidate_list,constrained_free_prompt,with_candidate_list")
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
