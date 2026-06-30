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
from phase735_source_restricted_writer_validation import MODELS, select_evenly  # noqa: E402
from phase755_cross_domain_route_invariance_atlas import get_first_token_id  # noqa: E402
from phase771_matched_causal_intervention_reliability_test import case_map_for, margin  # noqa: E402
from phase773_instruction_source_disentanglement import fmt, load_jsonl  # noqa: E402
from phase775_semantic_latent_route_output_closure import pool_diag, value_pool  # noqa: E402
from phase776_readout_bridge_competition_audit import load_model_bf16_prefer_flash, topk_competitors  # noqa: E402
from phase778_surface_form_normalization_causal_audit import (  # noqa: E402
    semantic_equiv,
    select_surface_cases,
    surface_prompt_for_variant,
)


OUT_ROOT = Path("results/glm5_phase780_surface_form_component_localization")
RESULT_ROOT = Path("tests/result/phase780_surface_form_component_localization")

PROMPT_VARIANTS = ["without_candidate_list", "lowercase_short_value", "with_candidate_list"]
COMPARE_BASELINE = "without_candidate_list"


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


def safe_rate(values: list[Any]) -> float | None:
    vals = [bool(v) for v in values]
    return sum(1 for v in vals if v) / len(vals) if vals else None


def lm_head_weight(model) -> torch.Tensor:
    if hasattr(model, "lm_head"):
        return model.lm_head.weight.detach().float().cpu()
    if hasattr(model, "embed_out"):
        return model.embed_out.weight.detach().float().cpu()
    raise ValueError("cannot find lm head weight")


def tensor_from_output(output: Any) -> torch.Tensor | None:
    if isinstance(output, tuple):
        if not output:
            return None
        output = output[0]
    if torch.is_tensor(output):
        return output
    return None


def capture_components_for_prompt(model, tokenizer, device, prompt: str) -> dict[str, Any]:
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    answer_pos = len(ids) - 1
    captured: dict[tuple[str, int], torch.Tensor] = {}
    handles = []
    layers = get_layers(model)
    for li, layer in enumerate(layers):
        if hasattr(layer, "self_attn"):
            def attn_hook(_module, _inputs, output, li=li):
                tensor = tensor_from_output(output)
                if tensor is not None:
                    captured[("attn", li)] = tensor[0, answer_pos].detach().float().cpu()

            handles.append(layer.self_attn.register_forward_hook(attn_hook))
        if hasattr(layer, "mlp"):
            def mlp_hook(_module, _inputs, output, li=li):
                tensor = tensor_from_output(output)
                if tensor is not None:
                    captured[("mlp", li)] = tensor[0, answer_pos].detach().float().cpu()

            handles.append(layer.mlp.register_forward_hook(mlp_hook))
    try:
        with torch.inference_mode():
            out = model(
                input_ids=torch.tensor([ids], device=device),
                return_dict=True,
                use_cache=False,
            )
        logits = out.logits[0, -1].detach().float().cpu()
    finally:
        for handle in handles:
            handle.remove()
    return {"ids": ids, "answer_pos": answer_pos, "logits": logits, "components": captured}


def observation_for(
    tokenizer,
    logits: torch.Tensor,
    case: dict[str, Any],
    prompt_variant: str,
    source_row: dict[str, Any],
    case_variant_token_id: int,
    top_k: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    target_id = get_first_token_id(tokenizer, case["answer"])
    contrast_id = get_first_token_id(tokenizer, case["contrast_answer"])
    pool = value_pool(tokenizer, case, target_id, contrast_id)
    target_diag = logit_diag(logits, target_id)
    contrast_diag = logit_diag(logits, contrast_id)
    casevar_diag = logit_diag(logits, case_variant_token_id)
    pdiag = pool_diag(logits, pool, target_id, contrast_id)
    top_rows = topk_competitors(tokenizer, logits, target_id, contrast_id, pool, case["answer"], case["contrast_answer"], top_k)
    top1 = top_rows[0] if top_rows else {}
    base = {
        "row_kind": "component_localization_observation",
        "case_id": case["case_id"],
        "prompt_variant": prompt_variant,
        "object": case["object"],
        "domain": case["domain"],
        "relation": case["relation"],
        "context_format": case["context_format"],
        "target_answer": case["answer"],
        "contrast_answer": case["contrast_answer"],
        "target_token_id": target_id,
        "contrast_token_id": contrast_id,
        "case_variant_token_id": int(case_variant_token_id),
        "case_variant_token_text": tokenizer.decode([int(case_variant_token_id)], skip_special_tokens=False),
        "source_prompt_variant": source_row.get("prompt_variant"),
        "source_top1_token_text": source_row.get("top1_token_text"),
        "source_top1_token_id": source_row.get("top1_token_id"),
        "source_top1_gap_above_target": source_row.get("top1_gap_above_target"),
        "base_target_rank": target_diag["target_rank"],
        "base_target_top1": bool(target_diag["target_top1"]),
        "base_contrast_rank": contrast_diag["target_rank"],
        "base_case_variant_rank": casevar_diag["target_rank"],
        "base_margin_target_vs_contrast": margin(logits, target_id, contrast_id),
        "base_margin_target_vs_case_variant": margin(logits, target_id, case_variant_token_id),
        "pool_size": pdiag["pool_size"],
        "pool_target_rank": pdiag["pool_target_rank"],
        "pool_target_top1": bool(pdiag["pool_target_top1"]),
        "pool_best_value": pdiag["pool_best_value"],
        "pool_margin_target_vs_best_other": pdiag["pool_margin_target_vs_best_other"],
        "top1_token_id": top1.get("token_id"),
        "top1_token_text": top1.get("token_text"),
        "top1_token_text_norm": top1.get("token_text_norm"),
        "top1_competitor_class": top1.get("competitor_class"),
        "top1_gap_above_target": top1.get("gap_above_target"),
    }
    base["semantic_equiv_open"] = semantic_equiv(base)
    base["hard_readout_after_equiv"] = bool(base["pool_target_top1"]) and not bool(base["semantic_equiv_open"])
    comp_rows = []
    for row in top_rows:
        comp_rows.append(
            {
                "row_kind": "component_localization_topk_competitor",
                **{k: v for k, v in base.items() if k != "row_kind"},
                **row,
            }
        )
    return base, comp_rows


def component_rows_for(
    model_name: str,
    prompt_variant: str,
    case: dict[str, Any],
    obs: dict[str, Any],
    components: dict[tuple[str, int], torch.Tensor],
    unembed: torch.Tensor,
) -> list[dict[str, Any]]:
    target_id = int(obs["target_token_id"])
    case_variant_id = int(obs["case_variant_token_id"])
    contrast_id = int(obs["contrast_token_id"])
    w_target = unembed[target_id]
    w_case_variant = unembed[case_variant_id]
    w_contrast = unembed[contrast_id]
    direction_case = w_target - w_case_variant
    direction_contrast = w_target - w_contrast
    rows = []
    for (kind, layer), vec in sorted(components.items(), key=lambda x: (x[0][0], x[0][1])):
        rows.append(
            {
                "row_kind": "component_localization_component",
                "model": model_name,
                "case_id": case["case_id"],
                "prompt_variant": prompt_variant,
                "domain": case["domain"],
                "relation": case["relation"],
                "context_format": case["context_format"],
                "component_kind": kind,
                "layer": int(layer),
                "target_token_id": target_id,
                "case_variant_token_id": case_variant_id,
                "contrast_token_id": contrast_id,
                "base_target_top1": bool(obs["base_target_top1"]),
                "semantic_equiv_open": bool(obs["semantic_equiv_open"]),
                "pool_target_top1": bool(obs["pool_target_top1"]),
                "actual_margin_target_vs_case_variant": obs["base_margin_target_vs_case_variant"],
                "actual_margin_target_vs_contrast": obs["base_margin_target_vs_contrast"],
                "component_norm": float(vec.norm().item()),
                "direct_target_vs_case_variant": float(torch.dot(vec, direction_case).item()),
                "direct_target_vs_contrast": float(torch.dot(vec, direction_contrast).item()),
                "direct_target_logit": float(torch.dot(vec, w_target).item()),
                "direct_case_variant_logit": float(torch.dot(vec, w_case_variant).item()),
            }
        )
    return rows


def delta_rows_for(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    comp = [r for r in rows if r.get("row_kind") == "component_localization_component"]
    obs = [r for r in rows if r.get("row_kind") == "component_localization_observation"]
    comp_map = {
        (r["case_id"], r["prompt_variant"], r["component_kind"], int(r["layer"])): r
        for r in comp
    }
    obs_map = {(r["case_id"], r["prompt_variant"]): r for r in obs}
    out = []
    variants = sorted({r["prompt_variant"] for r in obs if r["prompt_variant"] != COMPARE_BASELINE})
    for (case_id, base_variant), base_obs in obs_map.items():
        if base_variant != COMPARE_BASELINE:
            continue
        for variant in variants:
            other_obs = obs_map.get((case_id, variant))
            if not other_obs:
                continue
            for kind in ("attn", "mlp"):
                layers = sorted(
                    {
                        key[3]
                        for key in comp_map
                        if key[0] == case_id and key[2] == kind and key[1] in {COMPARE_BASELINE, variant}
                    }
                )
                for layer in layers:
                    base_comp = comp_map.get((case_id, COMPARE_BASELINE, kind, layer))
                    other_comp = comp_map.get((case_id, variant, kind, layer))
                    if not base_comp or not other_comp:
                        continue
                    delta_contrib = float(other_comp["direct_target_vs_case_variant"]) - float(base_comp["direct_target_vs_case_variant"])
                    delta_margin = float(other_obs["base_margin_target_vs_case_variant"]) - float(base_obs["base_margin_target_vs_case_variant"])
                    out.append(
                        {
                            "row_kind": "component_localization_delta",
                            "case_id": case_id,
                            "compare_variant": variant,
                            "baseline_variant": COMPARE_BASELINE,
                            "component_kind": kind,
                            "layer": int(layer),
                            "domain": base_obs.get("domain"),
                            "relation": base_obs.get("relation"),
                            "baseline_strict": bool(base_obs.get("base_target_top1")),
                            "variant_strict": bool(other_obs.get("base_target_top1")),
                            "strict_repaired": (not bool(base_obs.get("base_target_top1"))) and bool(other_obs.get("base_target_top1")),
                            "baseline_pool_top1": bool(base_obs.get("pool_target_top1")),
                            "variant_pool_top1": bool(other_obs.get("pool_target_top1")),
                            "baseline_semantic_equiv": bool(base_obs.get("semantic_equiv_open")),
                            "variant_semantic_equiv": bool(other_obs.get("semantic_equiv_open")),
                            "delta_actual_margin_target_vs_case_variant": delta_margin,
                            "delta_direct_target_vs_case_variant": delta_contrib,
                            "delta_direct_target_vs_contrast": float(other_comp["direct_target_vs_contrast"]) - float(base_comp["direct_target_vs_contrast"]),
                            "sign_aligned_with_margin": (delta_contrib > 0 and delta_margin > 0) or (delta_contrib < 0 and delta_margin < 0),
                        }
                    )
    return out


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
                "strict_open_rate": safe_rate([v.get("base_target_top1") for v in vals]),
                "semantic_equiv_open_rate": safe_rate([v.get("semantic_equiv_open") for v in vals]),
                "pool_top1_rate": safe_rate([v.get("pool_target_top1") for v in vals]),
                "mean_margin_target_vs_case_variant": safe_mean([v.get("base_margin_target_vs_case_variant") for v in vals]),
                "mean_target_rank": safe_mean([v.get("base_target_rank") for v in vals]),
                "top1_classes": dict(Counter(str(v.get("top1_competitor_class")) for v in vals)),
            }
        )
        out.append(payload)
    out.sort(key=lambda r: (r.get("prompt_variant") or "", r.get("strict_open_rate") or 0.0), reverse=True)
    return out


def group_deltas(rows: list[dict[str, Any]], key_fields: list[str]) -> list[dict[str, Any]]:
    vals = [r for r in rows if r.get("row_kind") == "component_localization_delta"]
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in vals:
        groups[tuple(row.get(k) for k in key_fields)].append(row)
    out = []
    for key, items in sorted(groups.items(), key=lambda kv: str(kv[0])):
        payload = {field: value for field, value in zip(key_fields, key)}
        mean_delta = safe_mean([r.get("delta_direct_target_vs_case_variant") for r in items])
        positive_rate = safe_rate([(r.get("delta_direct_target_vs_case_variant") or 0.0) > 0 for r in items])
        align_rate = safe_rate([r.get("sign_aligned_with_margin") for r in items])
        payload.update(
            {
                "n": len(items),
                "case_n": len({r["case_id"] for r in items}),
                "strict_repair_rate": safe_rate([r.get("strict_repaired") for r in items]),
                "mean_delta_actual_margin": safe_mean([r.get("delta_actual_margin_target_vs_case_variant") for r in items]),
                "mean_delta_direct_target_vs_case_variant": mean_delta,
                "mean_delta_direct_target_vs_contrast": safe_mean([r.get("delta_direct_target_vs_contrast") for r in items]),
                "positive_delta_rate": positive_rate,
                "sign_align_rate": align_rate,
                "candidate_score": (mean_delta or 0.0) * (positive_rate or 0.0) * (align_rate or 0.0),
            }
        )
        out.append(payload)
    out.sort(key=lambda r: r.get("candidate_score") or 0.0, reverse=True)
    return out


def summarize_rows(rows: list[dict[str, Any]], args: argparse.Namespace, model_name: str, attn_impl: str) -> dict[str, Any]:
    observations = [r for r in rows if r.get("row_kind") == "component_localization_observation"]
    components = [r for r in rows if r.get("row_kind") == "component_localization_component"]
    deltas = [r for r in rows if r.get("row_kind") == "component_localization_delta"]
    return {
        "phase": 780,
        "title": "Surface-Form Component Candidate Localization",
        "model": model_name,
        "round": args.round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "dtype": "bfloat16",
        "quantization": "off",
        "source_phase778_round": args.source_phase778_round,
        "prompt_variants": [x.strip() for x in args.prompt_variants.split(",") if x.strip()],
        "n_rows": len(rows),
        "n_observation_rows": len(observations),
        "n_component_rows": len(components),
        "n_delta_rows": len(deltas),
        "n_cases": len({r["case_id"] for r in observations}),
        "method_note": (
            "Direct-logit attribution over captured attention and MLP outputs. "
            "This localizes candidate components; it is not yet activation patch or ablation causality."
        ),
        "by_prompt_observation": group_observations(observations, ["prompt_variant"]),
        "top_delta_by_compare_kind_layer": group_deltas(deltas, ["compare_variant", "component_kind", "layer"])[:40],
        "top_delta_by_compare_kind": group_deltas(deltas, ["compare_variant", "component_kind"]),
        "strict_interpretation": (
            "Candidate localization only. A component is a stronger candidate when its direct target-vs-case-variant "
            "contribution increases with the surface prompt and aligns with actual margin repair."
        ),
    }


def audit_case(model, tokenizer, device, unembed: torch.Tensor, args: argparse.Namespace, case: dict[str, Any], source_row: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    case_variant_token_id = int(source_row.get("top1_token_id") or source_row.get("source_top1_token_id"))
    for prompt_variant in [x.strip() for x in args.prompt_variants.split(",") if x.strip()]:
        prompt = surface_prompt_for_variant(case, prompt_variant)
        state = capture_components_for_prompt(model, tokenizer, device, prompt)
        obs, top_rows = observation_for(tokenizer, state["logits"], case, prompt_variant, source_row, case_variant_token_id, args.top_k)
        rows.append(obs)
        rows.extend(top_rows)
        rows.extend(component_rows_for(args.model, prompt_variant, case, obs, state["components"], unembed))
    return rows


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = OUT_ROOT / args.round_name
    result_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    selected = select_surface_cases(args.model, args)
    log(f"{args.model}/{args.round_name}: selected cases={len(selected)} variants={args.prompt_variants}")
    cmap = case_map_for(args)
    model, tokenizer, device, attn_impl = load_model_bf16_prefer_flash(args.model, args.attn_implementations)
    try:
        unembed = lm_head_weight(model)
        rows: list[dict[str, Any]] = []
        for idx, source_row in enumerate(selected, 1):
            case = cmap[source_row["case_id"]]
            rows.extend(audit_case(model, tokenizer, device, unembed, args, case, source_row))
            if idx % args.log_every == 0 or idx == len(selected):
                log(f"{args.model}: component localization {idx}/{len(selected)} cases; rows={len(rows)}")
    finally:
        release_model(model)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    rows.extend(delta_rows_for(rows))
    summary = summarize_rows(rows, args, args.model, attn_impl)
    for root in (out_dir, result_dir):
        write_jsonl(root / f"phase780_{args.model}_rows.jsonl", rows)
        write_json(root / f"phase780_{args.model}_summary.json", summary)
    print(
        json.dumps(
            {
                "model": args.model,
                "round": args.round_name,
                "attn": attn_impl,
                "n_cases": summary["n_cases"],
                "by_prompt_observation": summary["by_prompt_observation"],
                "top_delta_by_compare_kind_layer": summary["top_delta_by_compare_kind_layer"][:8],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return summary


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# Phase 780 Surface-Form Component Candidate Localization ({payload['round']})",
        "",
        f"- Status: `{payload['status']}`",
        "- Test: direct-logit attribution over attention/MLP outputs for surface-form repair prompts.",
        "- Models are run sequentially; bf16, quantization off; attention implementation prefers flash/sdpa and falls back to eager.",
        "- Strict interpretation: candidate localization, not final causal proof.",
        "",
        "## Prompt Outcome Summary",
        "",
        "| model | variant | rows | cases | strict open | semantic-equiv open | pool top1 | margin target-case | target rank | top1 classes |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for model in MODELS:
        data = payload["by_model"].get(model)
        if not data:
            continue
        for row in data["by_prompt_observation"]:
            lines.append(
                f"| {model} | `{row['prompt_variant']}` | {row['n']} | {row['case_n']} | "
                f"{fmt(row['strict_open_rate'])} | {fmt(row['semantic_equiv_open_rate'])} | {fmt(row['pool_top1_rate'])} | "
                f"{fmt(row['mean_margin_target_vs_case_variant'])} | {fmt(row['mean_target_rank'])} | `{json.dumps(row['top1_classes'], ensure_ascii=False)}` |"
            )
    lines += [
        "",
        "## Top Component Delta Candidates",
        "",
        "| model | compare | kind | layer | rows | cases | repair rate | actual margin delta | direct target-case delta | positive rate | align rate | score |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        data = payload["by_model"].get(model)
        if not data:
            continue
        for row in data["top_delta_by_compare_kind_layer"][:16]:
            lines.append(
                f"| {model} | `{row['compare_variant']}` | `{row['component_kind']}` | {row['layer']} | {row['n']} | {row['case_n']} | "
                f"{fmt(row['strict_repair_rate'])} | {fmt(row['mean_delta_actual_margin'])} | "
                f"{fmt(row['mean_delta_direct_target_vs_case_variant'])} | {fmt(row['positive_delta_rate'])} | "
                f"{fmt(row['sign_align_rate'])} | {fmt(row['candidate_score'])} |"
            )
    lines += [
        "",
        "## By Component Kind",
        "",
        "| model | compare | kind | rows | cases | repair rate | direct target-case delta | positive rate | align rate | score |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        data = payload["by_model"].get(model)
        if not data:
            continue
        for row in data["top_delta_by_compare_kind"]:
            lines.append(
                f"| {model} | `{row['compare_variant']}` | `{row['component_kind']}` | {row['n']} | {row['case_n']} | "
                f"{fmt(row['strict_repair_rate'])} | {fmt(row['mean_delta_direct_target_vs_case_variant'])} | "
                f"{fmt(row['positive_delta_rate'])} | {fmt(row['sign_align_rate'])} | {fmt(row['candidate_score'])} |"
            )
    lines += [
        "",
        "## Strict Interpretation",
        "",
        "- This phase uses direct-logit attribution on captured component outputs.",
        "- Positive candidate layers are not yet proven necessary or sufficient.",
        "- The next step must patch or ablate top candidate components while measuring C_pool, C_surface, and C_token.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_cross_summary(round_name: str) -> dict[str, Any]:
    by_model = {}
    for model in MODELS:
        path = OUT_ROOT / round_name / f"phase780_{model}_summary.json"
        if path.exists():
            by_model[model] = json.loads(path.read_text(encoding="utf-8"))
    payload = {
        "phase": 780,
        "title": "Surface-Form Component Candidate Localization",
        "round": round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "complete" if len(by_model) == len(MODELS) else "partial",
        "models": list(by_model),
        "by_model": by_model,
    }
    for root in (OUT_ROOT, RESULT_ROOT):
        out_dir = root / round_name
        write_json(out_dir / "phase780_cross_model_summary.json", payload)
        write_markdown(out_dir / "phase780_cross_model_summary.md", payload)
    print(json.dumps({"round": round_name, "status": payload["status"], "models": payload["models"]}, ensure_ascii=False, indent=2), flush=True)
    return payload


def dry_run(args: argparse.Namespace) -> None:
    payload = {"round": args.round_name, "source_phase778_round": args.source_phase778_round, "models": {}}
    for model in MODELS:
        args.model = model
        selected = select_surface_cases(model, args)
        payload["models"][model] = {
            "selected_cases": len(selected),
            "domains": dict(Counter(r.get("domain") for r in selected)),
            "source_prompt_variants": dict(Counter(r.get("prompt_variant") for r in selected)),
            "prompt_variants": [x.strip() for x in args.prompt_variants.split(",") if x.strip()],
        }
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="smoke")
    parser.add_argument("--source-phase776-round", default="confirm")
    parser.add_argument("--source-phase778-round", default="confirm")
    parser.add_argument("--source-prompt-variants", default="without_candidate_list,constrained_free_prompt,with_candidate_list")
    parser.add_argument("--prompt-variants", default="without_candidate_list,lowercase_short_value,with_candidate_list")
    parser.add_argument("--relations", default="category,edible,grows_on_tree")
    parser.add_argument("--max-cases", type=int, default=6)
    parser.add_argument("--top-k", type=int, default=10)
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
