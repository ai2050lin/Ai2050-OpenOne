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
from phase722_functional_head_atlas_causal_ablation import write_json, write_jsonl  # noqa: E402
from phase735_source_restricted_writer_validation import MODELS, select_evenly  # noqa: E402
from phase771_matched_causal_intervention_reliability_test import case_map_for  # noqa: E402
from phase773_instruction_source_disentanglement import fmt  # noqa: E402
from phase776_readout_bridge_competition_audit import load_model_bf16_prefer_flash  # noqa: E402
from phase778_surface_form_normalization_causal_audit import select_surface_cases, surface_prompt_for_variant  # noqa: E402
from phase780_surface_form_component_localization import (  # noqa: E402
    COMPARE_BASELINE,
    capture_components_for_prompt,
    observation_for,
    tensor_from_output,
)


OUT_ROOT = Path("results/glm5_phase781_surface_form_candidate_causal_patch")
RESULT_ROOT = Path("tests/result/phase781_surface_form_candidate_causal_patch")
PHASE780_ROOT = Path("tests/result/phase780_surface_form_component_localization")


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


def load_phase780_summary(model: str, round_name: str) -> dict[str, Any]:
    path = PHASE780_ROOT / round_name / f"phase780_{model}_summary.json"
    if not path.exists():
        path = Path("results/glm5_phase780_surface_form_component_localization") / round_name / f"phase780_{model}_summary.json"
    return json.loads(path.read_text(encoding="utf-8"))


def select_candidates(model: str, args: argparse.Namespace) -> list[dict[str, Any]]:
    summary = load_phase780_summary(model, args.source_phase780_round)
    rows = list(summary.get("top_delta_by_compare_kind_layer") or [])
    rows = [
        r
        for r in rows
        if float(r.get("candidate_score") or 0.0) >= args.min_candidate_score
        and int(r.get("case_n") or 0) >= args.min_candidate_case_n
        and r.get("compare_variant") != COMPARE_BASELINE
    ]
    if args.candidate_compare_variants:
        allowed = {x.strip() for x in args.candidate_compare_variants.split(",") if x.strip()}
        rows = [r for r in rows if r.get("compare_variant") in allowed]
    if args.max_candidates and len(rows) > args.max_candidates:
        rows = rows[: args.max_candidates]
    out = []
    seen = set()
    for idx, row in enumerate(rows, 1):
        key = (row["compare_variant"], row["component_kind"], int(row["layer"]))
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "candidate_id": f"{row['compare_variant']}:{row['component_kind']}:L{int(row['layer'])}",
                "compare_variant": row["compare_variant"],
                "component_kind": row["component_kind"],
                "layer": int(row["layer"]),
                "phase780_candidate_score": row.get("candidate_score"),
                "phase780_direct_delta": row.get("mean_delta_direct_target_vs_case_variant"),
                "phase780_actual_margin_delta": row.get("mean_delta_actual_margin"),
                "phase780_strict_repair_rate": row.get("strict_repair_rate"),
                "phase780_rank": idx,
            }
        )
    return out


def replacement_output(output: Any, answer_pos: int, vector: torch.Tensor) -> Any:
    tensor = tensor_from_output(output)
    if tensor is None:
        return output
    patched = tensor.clone()
    patched[0, answer_pos] = vector.to(device=tensor.device, dtype=tensor.dtype)
    if isinstance(output, tuple):
        return (patched, *output[1:])
    return patched


def run_with_component_replacement(
    model,
    tokenizer,
    device,
    prompt: str,
    component_kind: str,
    layer_idx: int,
    replacement_vec: torch.Tensor,
) -> dict[str, Any]:
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    answer_pos = len(ids) - 1
    layers = get_layers(model)
    layer = layers[layer_idx]
    module = getattr(layer, "self_attn" if component_kind == "attn" else "mlp")

    def hook(_module, _inputs, output):
        return replacement_output(output, answer_pos, replacement_vec)

    handle = module.register_forward_hook(hook)
    try:
        with torch.inference_mode():
            out = model(
                input_ids=torch.tensor([ids], device=device),
                return_dict=True,
                use_cache=False,
            )
        logits = out.logits[0, -1].detach().float().cpu()
    finally:
        handle.remove()
    return {"ids": ids, "answer_pos": answer_pos, "logits": logits}


def make_row(
    model_name: str,
    candidate: dict[str, Any],
    case: dict[str, Any],
    prompt_variant: str,
    intervention_kind: str,
    obs: dict[str, Any],
    reference_obs: dict[str, Any] | None,
) -> dict[str, Any]:
    row = {
        "row_kind": "phase781_causal_intervention_observation",
        "model": model_name,
        "case_id": case["case_id"],
        "domain": case["domain"],
        "relation": case["relation"],
        "context_format": case["context_format"],
        "object": case["object"],
        "candidate_id": candidate["candidate_id"],
        "compare_variant": candidate["compare_variant"],
        "component_kind": candidate["component_kind"],
        "layer": candidate["layer"],
        "phase780_candidate_score": candidate.get("phase780_candidate_score"),
        "phase780_direct_delta": candidate.get("phase780_direct_delta"),
        "phase780_actual_margin_delta": candidate.get("phase780_actual_margin_delta"),
        "phase780_strict_repair_rate": candidate.get("phase780_strict_repair_rate"),
        "prompt_variant": prompt_variant,
        "intervention_kind": intervention_kind,
        "target_answer": obs.get("target_answer"),
        "contrast_answer": obs.get("contrast_answer"),
        "target_token_id": obs.get("target_token_id"),
        "case_variant_token_id": obs.get("case_variant_token_id"),
        "case_variant_token_text": obs.get("case_variant_token_text"),
        "target_top1": bool(obs.get("base_target_top1")),
        "semantic_equiv_open": bool(obs.get("semantic_equiv_open")),
        "pool_target_top1": bool(obs.get("pool_target_top1")),
        "target_rank": obs.get("base_target_rank"),
        "case_variant_rank": obs.get("base_case_variant_rank"),
        "margin_target_vs_case_variant": obs.get("base_margin_target_vs_case_variant"),
        "margin_target_vs_contrast": obs.get("base_margin_target_vs_contrast"),
        "top1_competitor_class": obs.get("top1_competitor_class"),
        "top1_token_text": obs.get("top1_token_text"),
        "top1_gap_above_target": obs.get("top1_gap_above_target"),
    }
    if reference_obs:
        row["delta_margin_vs_reference"] = (
            float(obs["base_margin_target_vs_case_variant"])
            - float(reference_obs["base_margin_target_vs_case_variant"])
        )
        row["strict_gain_vs_reference"] = bool(obs.get("base_target_top1")) and not bool(reference_obs.get("base_target_top1"))
        row["strict_loss_vs_reference"] = bool(reference_obs.get("base_target_top1")) and not bool(obs.get("base_target_top1"))
        row["semantic_loss_vs_reference"] = bool(reference_obs.get("semantic_equiv_open")) and not bool(obs.get("semantic_equiv_open"))
        row["pool_loss_vs_reference"] = bool(reference_obs.get("pool_target_top1")) and not bool(obs.get("pool_target_top1"))
    return row


def observe_logits(tokenizer, logits: torch.Tensor, case: dict[str, Any], prompt_variant: str, source_row: dict[str, Any], case_variant_token_id: int, top_k: int) -> dict[str, Any]:
    obs, _top = observation_for(tokenizer, logits, case, prompt_variant, source_row, case_variant_token_id, top_k)
    return obs


def audit_case_candidate(
    model,
    tokenizer,
    device,
    args: argparse.Namespace,
    case: dict[str, Any],
    source_row: dict[str, Any],
    candidate: dict[str, Any],
) -> list[dict[str, Any]]:
    case_variant_token_id = int(source_row.get("top1_token_id") or source_row.get("source_top1_token_id"))
    baseline_prompt = surface_prompt_for_variant(case, COMPARE_BASELINE)
    donor_variant = candidate["compare_variant"]
    donor_prompt = surface_prompt_for_variant(case, donor_variant)
    kind = candidate["component_kind"]
    layer = int(candidate["layer"])

    base_state = capture_components_for_prompt(model, tokenizer, device, baseline_prompt)
    donor_state = capture_components_for_prompt(model, tokenizer, device, donor_prompt)
    base_obs = observe_logits(tokenizer, base_state["logits"], case, COMPARE_BASELINE, source_row, case_variant_token_id, args.top_k)
    donor_obs = observe_logits(tokenizer, donor_state["logits"], case, donor_variant, source_row, case_variant_token_id, args.top_k)
    base_vec = base_state["components"].get((kind, layer))
    donor_vec = donor_state["components"].get((kind, layer))
    if base_vec is None or donor_vec is None:
        return []

    rows = [
        make_row(args.model, candidate, case, COMPARE_BASELINE, "normal_baseline", base_obs, None),
        make_row(args.model, candidate, case, donor_variant, "normal_donor", donor_obs, None),
    ]

    patch_state = run_with_component_replacement(model, tokenizer, device, baseline_prompt, kind, layer, donor_vec)
    patch_obs = observe_logits(tokenizer, patch_state["logits"], case, COMPARE_BASELINE, source_row, case_variant_token_id, args.top_k)
    rows.append(make_row(args.model, candidate, case, COMPARE_BASELINE, "patch_baseline_from_donor", patch_obs, base_obs))

    replace_state = run_with_component_replacement(model, tokenizer, device, donor_prompt, kind, layer, base_vec)
    replace_obs = observe_logits(tokenizer, replace_state["logits"], case, donor_variant, source_row, case_variant_token_id, args.top_k)
    rows.append(make_row(args.model, candidate, case, donor_variant, "replace_donor_with_baseline", replace_obs, donor_obs))

    zero_state = run_with_component_replacement(model, tokenizer, device, donor_prompt, kind, layer, torch.zeros_like(donor_vec))
    zero_obs = observe_logits(tokenizer, zero_state["logits"], case, donor_variant, source_row, case_variant_token_id, args.top_k)
    rows.append(make_row(args.model, candidate, case, donor_variant, "zero_donor_component", zero_obs, donor_obs))
    return rows


def group_rows(rows: list[dict[str, Any]], key_fields: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(k) for k in key_fields)].append(row)
    out = []
    for key, items in sorted(groups.items(), key=lambda kv: str(kv[0])):
        payload = {field: value for field, value in zip(key_fields, key)}
        payload.update(
            {
                "n": len(items),
                "case_n": len({r["case_id"] for r in items}),
                "strict_open_rate": safe_rate([r.get("target_top1") for r in items]),
                "semantic_equiv_open_rate": safe_rate([r.get("semantic_equiv_open") for r in items]),
                "pool_top1_rate": safe_rate([r.get("pool_target_top1") for r in items]),
                "mean_margin_target_vs_case_variant": safe_mean([r.get("margin_target_vs_case_variant") for r in items]),
                "mean_target_rank": safe_mean([r.get("target_rank") for r in items]),
                "mean_delta_margin_vs_reference": safe_mean([r.get("delta_margin_vs_reference") for r in items]),
                "strict_gain_rate_vs_reference": safe_rate([r.get("strict_gain_vs_reference") for r in items if "strict_gain_vs_reference" in r]),
                "strict_loss_rate_vs_reference": safe_rate([r.get("strict_loss_vs_reference") for r in items if "strict_loss_vs_reference" in r]),
                "semantic_loss_rate_vs_reference": safe_rate([r.get("semantic_loss_vs_reference") for r in items if "semantic_loss_vs_reference" in r]),
                "pool_loss_rate_vs_reference": safe_rate([r.get("pool_loss_vs_reference") for r in items if "pool_loss_vs_reference" in r]),
                "top1_classes": dict(Counter(str(r.get("top1_competitor_class")) for r in items)),
            }
        )
        payload["sufficiency_score"] = (
            (payload["mean_delta_margin_vs_reference"] or 0.0)
            * (payload["strict_gain_rate_vs_reference"] or 0.0)
            if payload.get("intervention_kind") == "patch_baseline_from_donor"
            else 0.0
        )
        payload["necessity_score"] = (
            (-(payload["mean_delta_margin_vs_reference"] or 0.0))
            * ((payload["strict_loss_rate_vs_reference"] or 0.0) + 0.5 * (payload["semantic_loss_rate_vs_reference"] or 0.0))
            if payload.get("intervention_kind") in {"replace_donor_with_baseline", "zero_donor_component"}
            else 0.0
        )
        out.append(payload)
    out.sort(
        key=lambda r: (
            r.get("model") or "",
            r.get("candidate_id") or "",
            r.get("intervention_kind") or "",
        )
    )
    return out


def summarize_rows(rows: list[dict[str, Any]], args: argparse.Namespace, attn_impl: str, candidates: list[dict[str, Any]]) -> dict[str, Any]:
    by_candidate_intervention = group_rows(rows, ["model", "candidate_id", "compare_variant", "component_kind", "layer", "intervention_kind"])
    top_sufficiency = sorted(
        [r for r in by_candidate_intervention if r.get("intervention_kind") == "patch_baseline_from_donor"],
        key=lambda r: (r.get("sufficiency_score") or 0.0, r.get("mean_delta_margin_vs_reference") or 0.0),
        reverse=True,
    )
    top_necessity = sorted(
        [r for r in by_candidate_intervention if r.get("intervention_kind") in {"replace_donor_with_baseline", "zero_donor_component"}],
        key=lambda r: (r.get("necessity_score") or 0.0, -(r.get("mean_delta_margin_vs_reference") or 0.0)),
        reverse=True,
    )
    return {
        "phase": 781,
        "title": "Surface-Form Candidate Causal Patch and Ablation",
        "model": args.model,
        "round": args.round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "dtype": "bfloat16",
        "quantization": "off",
        "source_phase780_round": args.source_phase780_round,
        "n_rows": len(rows),
        "n_cases": len({r["case_id"] for r in rows}),
        "n_candidates": len(candidates),
        "candidates": candidates,
        "method_note": (
            "Patch baseline answer-position component output with donor repair-prompt component output, "
            "then replace or zero donor component to test necessity. Candidate block granularity only."
        ),
        "by_candidate_intervention": by_candidate_intervention,
        "top_sufficiency_candidates": top_sufficiency,
        "top_necessity_candidates": top_necessity,
        "strict_interpretation": (
            "This phase tests causal direction at attention/MLP block granularity. "
            "A weak patch result does not disprove distributed causality; it may mean the candidate is one part of a multi-component route."
        ),
    }


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = OUT_ROOT / args.round_name
    result_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    selected = select_surface_cases(args.model, args)
    candidates = select_candidates(args.model, args)
    log(f"{args.model}/{args.round_name}: selected cases={len(selected)} candidates={len(candidates)}")
    cmap = case_map_for(args)
    model, tokenizer, device, attn_impl = load_model_bf16_prefer_flash(args.model, args.attn_implementations)
    try:
        rows: list[dict[str, Any]] = []
        for ci, source_row in enumerate(selected, 1):
            case = cmap[source_row["case_id"]]
            for candidate in candidates:
                rows.extend(audit_case_candidate(model, tokenizer, device, args, case, source_row, candidate))
            if ci % args.log_every == 0 or ci == len(selected):
                log(f"{args.model}: causal patch {ci}/{len(selected)} cases; rows={len(rows)}")
    finally:
        release_model(model)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = summarize_rows(rows, args, attn_impl, candidates)
    for root in (out_dir, result_dir):
        write_jsonl(root / f"phase781_{args.model}_rows.jsonl", rows)
        write_json(root / f"phase781_{args.model}_summary.json", summary)
    print(
        json.dumps(
            {
                "model": args.model,
                "round": args.round_name,
                "attn": attn_impl,
                "n_cases": summary["n_cases"],
                "n_candidates": summary["n_candidates"],
                "top_sufficiency_candidates": summary["top_sufficiency_candidates"][:6],
                "top_necessity_candidates": summary["top_necessity_candidates"][:6],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return summary


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# Phase 781 Surface-Form Candidate Causal Patch and Ablation ({payload['round']})",
        "",
        f"- Status: `{payload['status']}`",
        "- Test: patch baseline component from donor repair prompt; replace/zero donor component for necessity.",
        "- Models are run sequentially; bf16, quantization off; attention implementation prefers flash/sdpa/eager.",
        "- Strict interpretation: block-level causal evidence, not head/neuron-level proof.",
        "",
        "## Candidates",
        "",
        "| model | candidate | compare | kind | layer | phase780 score |",
        "|---|---|---|---|---:|---:|",
    ]
    for model in MODELS:
        data = payload["by_model"].get(model)
        if not data:
            continue
        for cand in data.get("candidates") or []:
            lines.append(
                f"| {model} | `{cand['candidate_id']}` | `{cand['compare_variant']}` | `{cand['component_kind']}` | "
                f"{cand['layer']} | {fmt(cand.get('phase780_candidate_score'))} |"
            )

    lines += [
        "",
        "## Intervention Summary",
        "",
        "| model | candidate | intervention | cases | strict | semantic-equiv | pool top1 | margin | delta margin | strict gain | strict loss | semantic loss | pool loss | top1 classes |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for model in MODELS:
        data = payload["by_model"].get(model)
        if not data:
            continue
        for row in data.get("by_candidate_intervention") or []:
            lines.append(
                f"| {model} | `{row['candidate_id']}` | `{row['intervention_kind']}` | {row['case_n']} | "
                f"{fmt(row['strict_open_rate'])} | {fmt(row['semantic_equiv_open_rate'])} | {fmt(row['pool_top1_rate'])} | "
                f"{fmt(row['mean_margin_target_vs_case_variant'])} | {fmt(row['mean_delta_margin_vs_reference'])} | "
                f"{fmt(row['strict_gain_rate_vs_reference'])} | {fmt(row['strict_loss_rate_vs_reference'])} | "
                f"{fmt(row['semantic_loss_rate_vs_reference'])} | {fmt(row['pool_loss_rate_vs_reference'])} | "
                f"`{json.dumps(row['top1_classes'], ensure_ascii=False)}` |"
            )

    lines += [
        "",
        "## Top Sufficiency Candidates",
        "",
        "| model | candidate | intervention | cases | delta margin | strict gain | score |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        data = payload["by_model"].get(model)
        if not data:
            continue
        for row in (data.get("top_sufficiency_candidates") or [])[:8]:
            lines.append(
                f"| {model} | `{row['candidate_id']}` | `{row['intervention_kind']}` | {row['case_n']} | "
                f"{fmt(row['mean_delta_margin_vs_reference'])} | {fmt(row['strict_gain_rate_vs_reference'])} | {fmt(row['sufficiency_score'])} |"
            )

    lines += [
        "",
        "## Top Necessity Candidates",
        "",
        "| model | candidate | intervention | cases | delta margin | strict loss | semantic loss | score |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        data = payload["by_model"].get(model)
        if not data:
            continue
        for row in (data.get("top_necessity_candidates") or [])[:8]:
            lines.append(
                f"| {model} | `{row['candidate_id']}` | `{row['intervention_kind']}` | {row['case_n']} | "
                f"{fmt(row['mean_delta_margin_vs_reference'])} | {fmt(row['strict_loss_rate_vs_reference'])} | "
                f"{fmt(row['semantic_loss_rate_vs_reference'])} | {fmt(row['necessity_score'])} |"
            )

    lines += [
        "",
        "## Strict Interpretation",
        "",
        "- `patch_baseline_from_donor` tests sufficiency at block granularity.",
        "- `replace_donor_with_baseline` and `zero_donor_component` test necessity at block granularity.",
        "- Weak single-block patch is still compatible with a distributed multi-component route.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_cross_summary(round_name: str) -> dict[str, Any]:
    by_model = {}
    for model in MODELS:
        path = OUT_ROOT / round_name / f"phase781_{model}_summary.json"
        if path.exists():
            by_model[model] = json.loads(path.read_text(encoding="utf-8"))
    payload = {
        "phase": 781,
        "title": "Surface-Form Candidate Causal Patch and Ablation",
        "round": round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "complete" if len(by_model) == len(MODELS) else "partial",
        "models": list(by_model),
        "by_model": by_model,
    }
    for root in (OUT_ROOT, RESULT_ROOT):
        out_dir = root / round_name
        write_json(out_dir / "phase781_cross_model_summary.json", payload)
        write_markdown(out_dir / "phase781_cross_model_summary.md", payload)
    print(json.dumps({"round": round_name, "status": payload["status"], "models": payload["models"]}, ensure_ascii=False, indent=2), flush=True)
    return payload


def dry_run(args: argparse.Namespace) -> None:
    payload = {"round": args.round_name, "source_phase780_round": args.source_phase780_round, "models": {}}
    for model in MODELS:
        args.model = model
        selected = select_surface_cases(model, args)
        candidates = select_candidates(model, args)
        payload["models"][model] = {
            "selected_cases": len(selected),
            "domains": dict(Counter(r.get("domain") for r in selected)),
            "source_prompt_variants": dict(Counter(r.get("prompt_variant") for r in selected)),
            "candidates": candidates,
        }
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="smoke")
    parser.add_argument("--source-phase776-round", default="confirm")
    parser.add_argument("--source-phase780-round", default="confirm")
    parser.add_argument("--source-prompt-variants", default="without_candidate_list,constrained_free_prompt,with_candidate_list")
    parser.add_argument("--relations", default="category,edible,grows_on_tree")
    parser.add_argument("--max-cases", type=int, default=6)
    parser.add_argument("--max-candidates", type=int, default=4)
    parser.add_argument("--min-candidate-score", type=float, default=0.0)
    parser.add_argument("--min-candidate-case-n", type=int, default=1)
    parser.add_argument("--candidate-compare-variants", default="")
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
