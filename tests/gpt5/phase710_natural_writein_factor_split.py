#!/usr/bin/env python3
"""
Phase 710: Natural Write-In Factor Split Audit.

Phase 709 showed that answer-start source-channel patches can sometimes close
to natural target-value generation, but the same patch still mixes route,
format, context identity, and possible value residue.

This phase keeps the same source-channel delta and changes only the injection
site:
  - pre_o_input: add selected channel delta before o_proj (Phase709 style);
  - post_o_output: add the projected attention output delta immediately after
    o_proj, before the layer residual/MLP path consumes it;
  - post_layer_output: add the same projected delta after the whole layer, so
    same-layer MLP cannot transform it.

If pre_o_input/post_o_output works but post_layer_output fails, same-layer
residual/MLP processing is implicated in natural write-in. If all three are
similar, the effect is mostly downstream readout-route gain.
"""
from __future__ import annotations

import argparse
import gc
import json
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
from phase112_attention_transport_head_mapping_cuda import get_attention_module  # noqa: E402
from phase132_source_value_contribution_cuda import get_num_kv_heads  # noqa: E402
from phase584_gate_repair import load_model_flash  # noqa: E402
from phase683_prose_route_bias_source_decomposition import expected_first_ids, expected_for, prompt_for, route_id_sets, select_base_cases  # noqa: E402
from phase685_natural_value_readout_writer_localization import SHORT_VARIANT, TERSE_VARIANT, select_paired_cases, value_minus_prose_direction  # noqa: E402
from phase687_l26_l27_value_support_state_decomposition import paired_case_metadata  # noqa: E402
from phase693_boundary_attention_head_candidate_audit import head_meta  # noqa: E402
from phase697_answer_last_route_transfer_decomposition import transfer_layers  # noqa: E402
from phase707_full_value_phrase_likelihood_audit import (  # noqa: E402
    COMBO_GROUPS,
    build_channel_scores,
    capture_state,
    choose_same_value_donor,
    choose_unrelated_donor,
    condition_channel_sets,
    load_top_head_sets,
    make_delta_patch_sets,
    score_channels_for_case,
    target_prose_phrase,
)
from phase709_natural_generation_writein_closure import generated_category  # noqa: E402


OUT_ROOT = Path("results/glm5_phase710_natural_writein_factor_split")
PATCH_SITES = ["pre_o_input", "post_o_output", "post_layer_output"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def select_conditions(channel_scores: list[dict[str, Any]], seed: int) -> dict[str, list[dict[str, Any]]]:
    raw = condition_channel_sets(channel_scores, [512], seed)
    return {
        "source_top_channel_512": raw.get("source_top_channel_512", []),
        "source_random_channel_512": raw.get("source_random_channel_512", []),
    }


def project_delta(o_proj, delta_full: torch.Tensor) -> torch.Tensor:
    weight = o_proj.weight.detach().float().cpu()
    return torch.mv(weight, delta_full.detach().float().cpu())


def install_site_hooks(
    model,
    patches: list[dict[str, Any]],
    layer_meta: dict[int, tuple[Any, int, int, int]],
    patch_site: str,
):
    handles = []
    layers = get_layers(model)
    for patch in patches:
        li = int(patch["layer"])
        delta_full = patch["delta_full"]
        o_proj, _n_heads, _head_dim, _n_kv_heads = layer_meta[li]
        if patch_site == "pre_o_input":
            def pre_hook(_module, inputs, delta_full=delta_full):
                x = inputs[0]
                y = x.clone()
                delta = delta_full.to(device=y.device, dtype=y.dtype)
                y[0, -1, :] = y[0, -1, :] + delta
                return (y,) + tuple(inputs[1:])

            handles.append(o_proj.register_forward_pre_hook(pre_hook))
        elif patch_site == "post_o_output":
            delta_out = project_delta(o_proj, delta_full)

            def out_hook(_module, _inputs, output, delta_out=delta_out):
                y = output.clone()
                delta = delta_out.to(device=y.device, dtype=y.dtype)
                y[0, -1, :] = y[0, -1, :] + delta
                return y

            handles.append(o_proj.register_forward_hook(out_hook))
        elif patch_site == "post_layer_output":
            delta_out = project_delta(o_proj, delta_full)

            def layer_hook(_module, _inputs, output, delta_out=delta_out):
                delta = delta_out.to(device=output[0].device if isinstance(output, tuple) else output.device)
                if isinstance(output, tuple):
                    first = output[0].clone()
                    first[0, -1, :] = first[0, -1, :] + delta.to(dtype=first.dtype)
                    return (first,) + tuple(output[1:])
                y = output.clone()
                y[0, -1, :] = y[0, -1, :] + delta.to(dtype=y.dtype)
                return y

            handles.append(layers[li].register_forward_hook(layer_hook))
        else:
            raise ValueError(f"unknown patch site: {patch_site}")
    return handles


def greedy_generate_site(
    model,
    tokenizer,
    device,
    prompt: str,
    patches: list[dict[str, Any]],
    layer_meta: dict[int, tuple[Any, int, int, int]],
    patch_site: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    generated: list[int] = []
    for step in range(max_new_tokens):
        handles = install_site_hooks(model, patches, layer_meta, patch_site) if step == 0 and patches else []
        try:
            with torch.inference_mode():
                out = model(input_ids=torch.tensor([input_ids + generated], device=device), return_dict=True, use_cache=False)
            next_id = int(torch.argmax(out.logits[0, -1]).item())
            generated.append(next_id)
            if tokenizer.eos_token_id is not None and next_id == int(tokenizer.eos_token_id):
                break
        finally:
            for handle in handles:
                handle.remove()
    return {
        "generated_ids": generated,
        "generated_text": tokenizer.decode(generated, skip_special_tokens=True),
        "first_token_id": generated[0] if generated else None,
    }


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    cats = Counter(row["generation_category"] for row in rows)

    def rate(cat: str) -> float:
        return cats.get(cat, 0) / n if n else 0.0

    return {
        "n": n,
        "category_counts": dict(cats.most_common()),
        "target_value_rate": rate("target_value"),
        "donor_value_rate": rate("donor_value"),
        "prose_target_rate": rate("prose_target"),
        "prose_donor_rate": rate("prose_donor"),
        "other_rate": rate("other"),
        "target_or_target_prose_rate": rate("target_value") + rate("prose_target"),
        "donor_or_donor_prose_rate": rate("donor_value") + rate("prose_donor"),
    }


def summarize_model(model_name: str, rows: list[dict[str, Any]], paired_ids: list[str], donor_pairs: list[dict[str, str]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["patch_site"], row["donor_type"], row["phase_kind"], row["condition"])].append(row)
    by_condition = {
        f"{site}|{donor_type}|{kind}|{cond}": summarize_group(vals)
        for (site, donor_type, kind, cond), vals in grouped.items()
    }
    restore = sorted(
        [{"condition": key, **vals} for key, vals in by_condition.items() if "|restore|" in key],
        key=lambda r: (r["target_value_rate"], r["target_or_target_prose_rate"], -r["donor_or_donor_prose_rate"]),
        reverse=True,
    )
    return {
        "model": model_name,
        "n_paired_cases": len(paired_ids),
        "n_donor_pairs": len(donor_pairs),
        "n_rows": len(rows),
        "best_restore_conditions": restore,
        "by_condition": by_condition,
    }


def run_model(args) -> dict[str, Any]:
    paired_ids = select_paired_cases(args.model, args.limit)
    case_map = {case["case_id"]: case for case in select_base_cases()}
    meta = paired_case_metadata(case_map, paired_ids)
    top_head_rows, head_sets = load_top_head_sets(args.model, args.top_heads)
    model, tokenizer, device = load_model_flash(args.model)
    rows: list[dict[str, Any]] = []
    case_score_rows: dict[str, list[dict[str, Any]]] = {}
    case_cache: dict[str, dict[str, Any]] = {}
    donor_pairs: list[dict[str, str]] = []
    try:
        dtype = next(model.parameters()).dtype
        scan_layers = transfer_layers(args.model, len(get_layers(model)))
        layer_meta: dict[int, tuple[Any, int, int, int]] = {}
        for li in scan_layers:
            o_proj, n_heads, head_dim = head_meta(model, li)
            attn = get_attention_module(get_layers(model)[li])
            layer_meta[li] = (o_proj, n_heads, head_dim, get_num_kv_heads(model, attn, n_heads))
        head_sets = {li: heads for li, heads in head_sets.items() if li in layer_meta}

        for idx, case_id in enumerate(paired_ids, 1):
            case = case_map[case_id]
            expected_text = expected_for(case, SHORT_VARIANT)
            expected_ids = expected_first_ids(tokenizer, expected_text)
            routes = route_id_sets(tokenizer, case, expected_text)
            direction = value_minus_prose_direction(model, routes, expected_ids, device, dtype).detach().cpu()
            short_prompt = prompt_for(case, SHORT_VARIANT)
            terse_prompt = prompt_for(case, TERSE_VARIANT)
            short_state = capture_state(model, tokenizer, device, short_prompt, case, scan_layers, direction, routes, expected_ids, layer_meta)
            terse_state = capture_state(model, tokenizer, device, terse_prompt, case, scan_layers, direction, routes, expected_ids, layer_meta)
            case_cache[case_id] = {
                "short_prompt": short_prompt,
                "terse_prompt": terse_prompt,
                "short_state": short_state,
                "terse_state": terse_state,
            }
            case_score_rows[case_id] = score_channels_for_case(layer_meta, head_sets, short_state, terse_state, direction)
            if idx % args.log_every == 0 or idx == len(paired_ids):
                log(f"{args.model}: captured/source-scored {idx}/{len(paired_ids)} paired cases")

        all_score_rows = []
        for case_id in paired_ids:
            all_score_rows.extend(case_score_rows[case_id])
        conditions = select_conditions(build_channel_scores(all_score_rows, layer_meta), args.seed)

        for case_id in paired_ids:
            same = choose_same_value_donor(case_id, paired_ids, meta)
            if same is not None:
                donor_pairs.append({"target": case_id, "donor": same, "donor_type": "same_value"})
            unrelated = choose_unrelated_donor(case_id, paired_ids, meta, args.seed)
            if unrelated is not None:
                donor_pairs.append({"target": case_id, "donor": unrelated, "donor_type": "unrelated"})

        for idx, pair in enumerate(donor_pairs, 1):
            case_id = pair["target"]
            donor_id = pair["donor"]
            cur = case_cache[case_id]
            donor = case_cache[donor_id]
            target_prose = target_prose_phrase(case_map[case_id])
            for cond_name, selected in conditions.items():
                for phase_kind, prompt, src_state, dst_state in [
                    ("restore", cur["short_prompt"], donor["terse_state"], cur["short_state"]),
                    ("degradation", cur["terse_prompt"], donor["short_state"], cur["terse_state"]),
                ]:
                    patches = make_delta_patch_sets(src_state, dst_state, selected)
                    for patch_site in PATCH_SITES:
                        gen = greedy_generate_site(model, tokenizer, device, prompt, patches, layer_meta, patch_site, args.max_new_tokens)
                        rows.append({
                            "case_id": case_id,
                            "donor_case_id": donor_id,
                            "donor_type": pair["donor_type"],
                            "phase_kind": phase_kind,
                            "condition": cond_name,
                            "patch_site": patch_site,
                            "value": meta[case_id]["value"],
                            "donor_value": meta[donor_id]["value"],
                            "generated_text": gen["generated_text"],
                            "generated_ids": gen["generated_ids"],
                            "generation_category": generated_category(gen["generated_text"], meta[case_id]["value"], meta[donor_id]["value"], target_prose),
                        })
            if idx % args.log_every == 0 or idx == len(donor_pairs):
                log(f"{args.model}: patch-site generated {idx}/{len(donor_pairs)} donor pairs")
    finally:
        release_model(model)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = summarize_model(args.model, rows, paired_ids, donor_pairs)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / f"phase710_{args.model}_factor_split_rows.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    payload = {
        "phase": 710,
        "title": "Natural Write-In Factor Split Audit",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": args.model,
        "patch_sites": PATCH_SITES,
        "top_heads": args.top_heads,
        "max_new_tokens": args.max_new_tokens,
        "source_combo": COMBO_GROUPS,
        "summary": summary,
        "phase698_top_heads": top_head_rows,
    }
    (OUT_ROOT / f"phase710_{args.model}_factor_split_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return payload


def write_cross_summary() -> dict[str, Any]:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    models = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(OUT_ROOT.glob("phase710_*_factor_split_summary.json"))]
    payload = {
        "phase": 710,
        "title": "Natural Write-In Factor Split Audit Cross-Model Summary",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": models,
    }
    (OUT_ROOT / "phase710_cross_model_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# Phase 710 Natural Write-In Factor Split Audit",
        "",
        f"- generated: `{payload['timestamp']}`",
        "",
        "| model | condition | n | target_value | donor_value | prose_target | other |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for item in models:
        for row in item["summary"]["best_restore_conditions"][:18]:
            lines.append(
                f"| {item['model']} | {row['condition']} | {row['n']} | {row['target_value_rate']:.3f} | "
                f"{row['donor_value_rate']:.3f} | {row['prose_target_rate']:.3f} | {row['other_rate']:.3f} |"
            )
    (OUT_ROOT / "phase710_cross_model_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--hard-exit-after-model", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top-heads", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=710)
    parser.add_argument("--log-every", type=int, default=12)
    args = parser.parse_args()
    if args.summarize_only:
        write_cross_summary()
        return
    if not args.model:
        raise SystemExit("--model is required unless --summarize-only is used")
    run_model(args)
    if args.hard_exit_after_model:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
