#!/usr/bin/env python3
"""
Phase 693: Boundary Attention Head Candidate Audit.

Phase 692 showed that multi-layer attn+mlp window patches are effective in the
L13-L18 DS7B boundary region, but it did not identify attention heads. This
phase performs a constrained head-level audit:

1. Capture o_proj inputs at the last prompt token for short_only and
   terse_no_explain.
2. Score each boundary head by the direct o_proj(delta_head) projection onto
   the case-specific value-minus-prose readout direction. This is only a
   candidate screen.
3. Run causal head-slice transplants for per-layer top1/top2/top4/all_heads
   and window top8/top16/random controls.

It does not yet prove source-token routing. Source-token path is the next
stage if stable head candidates appear.
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
from phase584_gate_repair import load_model_flash  # noqa: E402
from phase683_prose_route_bias_source_decomposition import (  # noqa: E402
    expected_first_ids,
    expected_for,
    prompt_for,
    route_id_sets,
    select_base_cases,
)
from phase685_natural_value_readout_writer_localization import (  # noqa: E402
    SHORT_VARIANT,
    TERSE_VARIANT,
    projection,
    select_paired_cases,
    value_minus_prose_direction,
)
from phase687_l26_l27_value_support_state_decomposition import (  # noqa: E402
    classify,
    get_module,
    model_layers,
    paired_case_metadata,
)
from phase691_boundary_component_residual_carry_decomposition import boundary_layers  # noqa: E402
from phase112_attention_transport_head_mapping_cuda import (  # noqa: E402
    get_attention_module,
    get_num_heads,
    get_o_proj,
)


OUT_ROOT = Path("results/glm5_phase693_boundary_attention_head_candidate_audit")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def head_meta(model, layer_idx: int) -> tuple[Any, int, int]:
    layer = get_layers(model)[layer_idx]
    attn = get_attention_module(layer)
    o_proj = get_o_proj(attn)
    n_heads = get_num_heads(model, attn)
    in_features = int(o_proj.in_features)
    if in_features % n_heads != 0:
        raise RuntimeError(f"L{layer_idx}: o_proj input {in_features} not divisible by heads {n_heads}")
    return o_proj, n_heads, in_features // n_heads


def capture_oproj_and_target(
    model,
    tokenizer,
    device,
    prompt: str,
    scan_layers: list[int],
    target_layer: int,
) -> tuple[torch.Tensor, dict[int, torch.Tensor], torch.Tensor]:
    captured_heads: dict[int, torch.Tensor] = {}
    captured_target: dict[str, torch.Tensor] = {}
    handles = []
    for li in scan_layers:
        o_proj, _n, _d = head_meta(model, li)

        def pre_hook(_module, inputs, li=li):
            captured_heads[li] = inputs[0][0, -1].detach().cpu()

        handles.append(o_proj.register_forward_pre_hook(pre_hook))

    target_module = get_module(model, target_layer, "layer_input")

    def target_pre_hook(_module, inputs):
        captured_target["target"] = inputs[0][0, -1].detach().cpu()

    handles.append(target_module.register_forward_pre_hook(target_pre_hook))
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    try:
        with torch.inference_mode():
            out = model(input_ids=torch.tensor([ids], device=device), return_dict=True, use_cache=False)
        logits = out.logits[0, -1].detach()
    finally:
        for h in handles:
            h.remove()
    return logits, captured_heads, captured_target["target"]


def install_head_patch_hooks(model, patch_sets: list[dict[str, Any]]):
    by_layer: dict[int, dict[str, Any]] = {}
    for patch in patch_sets:
        li = patch["layer"]
        rec = by_layer.setdefault(li, {"heads": set(), "new_full": patch["new_full"]})
        rec["heads"].update(int(h) for h in patch["heads"])
    handles = []
    for li, rec in by_layer.items():
        o_proj, n_heads, head_dim = head_meta(model, li)
        heads = sorted(h for h in rec["heads"] if 0 <= h < n_heads)
        new_full = rec["new_full"]

        def pre_hook(_module, inputs, heads=heads, new_full=new_full, n_heads=n_heads, head_dim=head_dim):
            x = inputs[0]
            y = x.clone()
            src = new_full.to(device=y.device, dtype=y.dtype)
            yv = y.view(y.shape[0], y.shape[1], n_heads, head_dim)
            srcv = src.view(n_heads, head_dim)
            for h in heads:
                yv[0, -1, h, :] = srcv[h]
            return (y,) + tuple(inputs[1:])

        handles.append(o_proj.register_forward_pre_hook(pre_hook))
    return handles


def run_head_patched_with_target_capture(
    model,
    tokenizer,
    device,
    prompt: str,
    patch_sets: list[dict[str, Any]],
    target_layer: int,
    direction: torch.Tensor,
    routes,
    expected_ids,
) -> dict[str, Any]:
    captured: dict[str, torch.Tensor] = {}
    handles = install_head_patch_hooks(model, patch_sets)
    target_module = get_module(model, target_layer, "layer_input")

    def target_pre_hook(_module, inputs):
        captured["target"] = inputs[0][0, -1].detach()

    handles.append(target_module.register_forward_pre_hook(target_pre_hook))
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    try:
        with torch.inference_mode():
            out = model(input_ids=torch.tensor([ids], device=device), return_dict=True, use_cache=False)
        diag = classify(out.logits[0, -1].detach(), routes, expected_ids)
    finally:
        for h in handles:
            h.remove()
    diag["target_proj"] = projection(captured["target"], direction)
    return diag


def oproj_head_contribution(o_proj, delta_full: torch.Tensor, head_id: int, n_heads: int, head_dim: int) -> torch.Tensor:
    start = head_id * head_dim
    end = start + head_dim
    w = o_proj.weight.detach().float().cpu()[:, start:end]
    return torch.mv(w, delta_full[start:end].float().cpu())


def make_row(
    meta,
    case_id,
    phase_kind,
    condition,
    target_layer,
    short_diag,
    terse_diag,
    patched,
    short_target_proj,
    terse_target_proj,
    head_sets,
):
    if phase_kind == "restore":
        final_success_change = (not short_diag["expected_top1"]) and patched["expected_top1"]
        rank_effect = short_diag["expected_rank"] - patched["expected_rank"]
        pmv_effect = short_diag["prose_minus_value"] - patched["prose_minus_value"]
        target_effect = patched["target_proj"] - short_target_proj
    else:
        final_success_change = terse_diag["expected_top1"] and not patched["expected_top1"]
        rank_effect = patched["expected_rank"] - terse_diag["expected_rank"]
        pmv_effect = patched["prose_minus_value"] - terse_diag["prose_minus_value"]
        target_effect = terse_target_proj - patched["target_proj"]
    target_delta = terse_target_proj - short_target_proj
    return {
        "case_id": case_id,
        "family": meta[case_id]["family"],
        "relation": meta[case_id]["relation"],
        "value": meta[case_id]["value"],
        "phase_kind": phase_kind,
        "condition": condition,
        "head_sets": head_sets,
        "target_site": f"L{target_layer}_layer_input",
        "short_rank": short_diag["expected_rank"],
        "terse_rank": terse_diag["expected_rank"],
        "patched_rank": patched["expected_rank"],
        "short_top1": short_diag["expected_top1"],
        "terse_top1": terse_diag["expected_top1"],
        "patched_top1": patched["expected_top1"],
        "final_success_change": final_success_change,
        "rank_effect": rank_effect,
        "short_pmv": short_diag["prose_minus_value"],
        "terse_pmv": terse_diag["prose_minus_value"],
        "patched_pmv": patched["prose_minus_value"],
        "pmv_effect": pmv_effect,
        "short_target_proj": short_target_proj,
        "terse_target_proj": terse_target_proj,
        "target_delta_terse_minus_short": target_delta,
        "patched_target_proj": patched["target_proj"],
        "target_effect": target_effect,
        "target_delta_fraction": target_effect / target_delta if abs(target_delta) > 1e-8 else None,
        "patched_best_other_route": patched["best_other_route"],
    }


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    valid_frac = [r["target_delta_fraction"] for r in rows if r["target_delta_fraction"] is not None]
    return {
        "n": n,
        "success_change_rate": sum(1 for r in rows if r["final_success_change"]) / n,
        "patched_top1_rate": sum(1 for r in rows if r["patched_top1"]) / n,
        "mean_short_rank": sum(r["short_rank"] for r in rows) / n,
        "mean_terse_rank": sum(r["terse_rank"] for r in rows) / n,
        "mean_patched_rank": sum(r["patched_rank"] for r in rows) / n,
        "mean_rank_effect": sum(r["rank_effect"] for r in rows) / n,
        "mean_pmv_effect": sum(r["pmv_effect"] for r in rows) / n,
        "mean_target_effect": sum(r["target_effect"] for r in rows) / n,
        "mean_target_delta": sum(r["target_delta_terse_minus_short"] for r in rows) / n,
        "mean_target_delta_fraction": sum(valid_frac) / max(1, len(valid_frac)),
        "patched_best_other_route": dict(Counter(r["patched_best_other_route"] for r in rows).most_common()),
    }


def summarize_model(model_name: str, paired_ids: list[str], rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        grouped[(r["phase_kind"], r["condition"])].append(r)
    by_condition = {f"{k}|{c}": summarize_group(v) for (k, c), v in grouped.items()}
    best_restore = sorted(
        ((k, v) for k, v in by_condition.items() if k.startswith("restore|")),
        key=lambda kv: (kv[1]["success_change_rate"], kv[1]["mean_target_effect"], kv[1]["mean_rank_effect"]),
        reverse=True,
    )
    best_degradation = sorted(
        ((k, v) for k, v in by_condition.items() if k.startswith("degradation|")),
        key=lambda kv: (kv[1]["success_change_rate"], kv[1]["mean_target_effect"], kv[1]["mean_rank_effect"]),
        reverse=True,
    )
    return {
        "model": model_name,
        "n_paired_cases": len(paired_ids),
        "by_condition": by_condition,
        "best_restore_conditions": [{"condition": k, **v} for k, v in best_restore],
        "best_degradation_conditions": [{"condition": k, **v} for k, v in best_degradation],
        "top_candidate_heads": sorted(candidate_rows, key=lambda r: r["mean_direct_effect"], reverse=True)[:32],
    }


def random_heads(n_heads: int, k: int, seed: int, avoid: set[int] | None = None) -> list[int]:
    avoid = avoid or set()
    choices = [h for h in range(n_heads) if h not in avoid]
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    perm = torch.randperm(len(choices), generator=gen).tolist()
    return [choices[i] for i in perm[: min(k, len(choices))]]


def run_model(args) -> dict[str, Any]:
    paired_ids = select_paired_cases(args.model, args.limit)
    case_map = {c["case_id"]: c for c in select_base_cases()}
    meta = paired_case_metadata(case_map, paired_ids)
    model, tokenizer, device = load_model_flash(args.model)
    rows: list[dict[str, Any]] = []
    candidate_acc: dict[tuple[int, int], list[dict[str, float]]] = defaultdict(list)
    case_cache: dict[str, dict[str, Any]] = {}
    try:
        dtype = next(model.parameters()).dtype
        target_layer = model_layers(args.model, len(get_layers(model)))[0]
        scan_layers = boundary_layers(args.model, target_layer)
        layer_meta = {li: head_meta(model, li) for li in scan_layers}
        heads_by_layer = {li: list(range(layer_meta[li][1])) for li in scan_layers}

        for idx, case_id in enumerate(paired_ids, 1):
            case = case_map[case_id]
            expected_text = expected_for(case, SHORT_VARIANT)
            expected_ids = expected_first_ids(tokenizer, expected_text)
            routes = route_id_sets(tokenizer, case, expected_text)
            direction = value_minus_prose_direction(model, routes, expected_ids, device, dtype)
            direction_cpu = direction.detach().cpu()
            short_prompt = prompt_for(case, SHORT_VARIANT)
            terse_prompt = prompt_for(case, TERSE_VARIANT)

            short_logits, short_heads, short_target = capture_oproj_and_target(model, tokenizer, device, short_prompt, scan_layers, target_layer)
            terse_logits, terse_heads, terse_target = capture_oproj_and_target(model, tokenizer, device, terse_prompt, scan_layers, target_layer)
            short_diag = classify(short_logits, routes, expected_ids)
            terse_diag = classify(terse_logits, routes, expected_ids)
            short_target_proj = projection(short_target, direction_cpu)
            terse_target_proj = projection(terse_target, direction_cpu)
            case_cache[case_id] = {
                "short_prompt": short_prompt,
                "terse_prompt": terse_prompt,
                "routes": routes,
                "expected_ids": expected_ids,
                "direction": direction_cpu,
                "short_heads": short_heads,
                "terse_heads": terse_heads,
                "short_diag": short_diag,
                "terse_diag": terse_diag,
                "short_target_proj": short_target_proj,
                "terse_target_proj": terse_target_proj,
            }
            for li in scan_layers:
                o_proj, n_heads, head_dim = layer_meta[li]
                delta = terse_heads[li] - short_heads[li]
                for h in range(n_heads):
                    contrib = oproj_head_contribution(o_proj, delta, h, n_heads, head_dim)
                    direct = projection(contrib, direction_cpu)
                    candidate_acc[(li, h)].append({
                        "direct_effect": direct,
                        "delta_norm": float(delta[h * head_dim:(h + 1) * head_dim].float().norm().item()),
                    })
            if idx % args.log_every == 0 or idx == len(paired_ids):
                log(f"{args.model}: cached/scored {idx}/{len(paired_ids)} paired cases")

        candidate_rows = []
        for (li, h), vals in candidate_acc.items():
            candidate_rows.append({
                "layer": li,
                "head": h,
                "head_key": f"L{li}H{h}",
                "mean_direct_effect": sum(v["direct_effect"] for v in vals) / len(vals),
                "mean_delta_norm": sum(v["delta_norm"] for v in vals) / len(vals),
            })
        by_layer_ranked = {
            li: sorted([r for r in candidate_rows if r["layer"] == li], key=lambda r: r["mean_direct_effect"], reverse=True)
            for li in scan_layers
        }
        global_ranked = sorted(candidate_rows, key=lambda r: r["mean_direct_effect"], reverse=True)
        mid = len(scan_layers) // 2
        window_layers = {
            "early": scan_layers[:mid],
            "late": scan_layers[mid:],
            "all": scan_layers,
        }

        def condition_head_sets() -> dict[str, dict[int, list[int]]]:
            conds: dict[str, dict[int, list[int]]] = {}
            for li in scan_layers:
                n_heads = layer_meta[li][1]
                ranked = by_layer_ranked[li]
                for k in [1, 2, 4]:
                    conds[f"L{li}_top{k}"] = {li: [int(r["head"]) for r in ranked[: min(k, len(ranked))]]}
                conds[f"L{li}_all_heads"] = {li: list(range(n_heads))}
                top4 = set(conds[f"L{li}_top4"][li])
                conds[f"L{li}_random4"] = {li: random_heads(n_heads, min(4, n_heads), 69300 + li, top4)}
            for name, layers in window_layers.items():
                subset = [r for r in global_ranked if r["layer"] in layers]
                for k in [8, 16]:
                    selected: dict[int, list[int]] = defaultdict(list)
                    for r in subset[: min(k, len(subset))]:
                        selected[int(r["layer"])].append(int(r["head"]))
                    conds[f"{name}_top{k}"] = dict(selected)
                avoid = {(int(r["layer"]), int(r["head"])) for r in subset[: min(16, len(subset))]}
                selected = defaultdict(list)
                for li in layers:
                    n_heads = layer_meta[li][1]
                    rand = random_heads(n_heads, min(4, n_heads), 69400 + li, {h for l, h in avoid if l == li})
                    selected[li].extend(rand)
                conds[f"{name}_random"] = dict(selected)
            return conds

        conds = condition_head_sets()

        for idx, case_id in enumerate(paired_ids, 1):
            cur = case_cache[case_id]
            for cond_name, head_sets in conds.items():
                for phase_kind, prompt, donor_key in [
                    ("restore", cur["short_prompt"], "terse_heads"),
                    ("degradation", cur["terse_prompt"], "short_heads"),
                ]:
                    patches = [
                        {"layer": li, "heads": heads, "new_full": cur[donor_key][li]}
                        for li, heads in head_sets.items()
                        if heads
                    ]
                    patched = run_head_patched_with_target_capture(
                        model,
                        tokenizer,
                        device,
                        prompt,
                        patches,
                        target_layer,
                        cur["direction"].to(device),
                        cur["routes"],
                        cur["expected_ids"],
                    )
                    rows.append(make_row(
                        meta,
                        case_id,
                        phase_kind,
                        cond_name,
                        target_layer,
                        cur["short_diag"],
                        cur["terse_diag"],
                        patched,
                        cur["short_target_proj"],
                        cur["terse_target_proj"],
                        {str(li): heads for li, heads in head_sets.items()},
                    ))
            if idx % args.log_every == 0 or idx == len(paired_ids):
                log(f"{args.model}: causal patched {idx}/{len(paired_ids)} paired cases")
    finally:
        release_model(model)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = summarize_model(args.model, paired_ids, rows, candidate_rows)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / f"phase693_{args.model}_head_candidate_rows.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + "\n",
        encoding="utf-8",
    )
    (OUT_ROOT / f"phase693_{args.model}_candidate_scores.json").write_text(
        json.dumps(sorted(candidate_rows, key=lambda r: r["mean_direct_effect"], reverse=True), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    payload = {
        "phase": 693,
        "title": "Boundary Attention Head Candidate Audit",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": args.model,
        "target_layer": target_layer,
        "scan_layers": scan_layers,
        "n_paired_cases": len(paired_ids),
        "n_rows": len(rows),
        "summary": summary,
    }
    (OUT_ROOT / f"phase693_{args.model}_head_candidate_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return payload


def write_cross_summary() -> dict[str, Any]:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    models = []
    for path in sorted(OUT_ROOT.glob("phase693_*_head_candidate_summary.json")):
        models.append(json.loads(path.read_text(encoding="utf-8")))
    payload = {
        "phase": 693,
        "title": "Boundary Attention Head Candidate Audit Cross-Model Summary",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": models,
    }
    (OUT_ROOT / "phase693_cross_model_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    lines = [
        "# Phase 693 Boundary Attention Head Candidate Audit",
        "",
        f"- generated: `{payload['timestamp']}`",
        "",
        "| model | pairs | target | scan_layers | best_restore | repair | target_gain | rank_effect | best_degrade | drop | target_loss | rank_effect |",
        "|---|---:|---|---|---|---:|---:|---:|---|---:|---:|---:|",
    ]
    for item in models:
        br = item["summary"]["best_restore_conditions"][0] if item["summary"]["best_restore_conditions"] else {}
        bd = item["summary"]["best_degradation_conditions"][0] if item["summary"]["best_degradation_conditions"] else {}
        lines.append(
            f"| {item['model']} | {item['n_paired_cases']} | L{item['target_layer']}_layer_input | {item['scan_layers']} | "
            f"{br.get('condition', '')} | {br.get('success_change_rate', 0.0):.3f} | {br.get('mean_target_effect', 0.0):.3f} | {br.get('mean_rank_effect', 0.0):.2f} | "
            f"{bd.get('condition', '')} | {bd.get('success_change_rate', 0.0):.3f} | {bd.get('mean_target_effect', 0.0):.3f} | {bd.get('mean_rank_effect', 0.0):.2f} |"
        )
    for section, key in [("Top Candidate Heads", "top_candidate_heads"), ("Best Restore", "best_restore_conditions"), ("Best Degradation", "best_degradation_conditions")]:
        lines.extend(["", f"## {section}", ""])
        for item in models:
            lines.append(f"### {item['model']}")
            lines.append("")
            if key == "top_candidate_heads":
                lines.append("| head | mean_direct_effect | mean_delta_norm |")
                lines.append("|---|---:|---:|")
                for row in item["summary"][key][:24]:
                    lines.append(f"| {row['head_key']} | {row['mean_direct_effect']:.3f} | {row['mean_delta_norm']:.3f} |")
            else:
                lines.append("| condition | change | patched_top1 | patched_rank | rank_effect | target_effect | target_fraction | pmv_effect | best_other |")
                lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|")
                for row in item["summary"][key][:24]:
                    lines.append(
                        f"| {row['condition']} | {row['success_change_rate']:.3f} | {row['patched_top1_rate']:.3f} | "
                        f"{row['mean_patched_rank']:.2f} | {row['mean_rank_effect']:.2f} | "
                        f"{row['mean_target_effect']:.3f} | {row['mean_target_delta_fraction']:.3f} | "
                        f"{row['mean_pmv_effect']:.3f} | {row['patched_best_other_route']} |"
                    )
            lines.append("")
    (OUT_ROOT / "phase693_cross_model_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--hard-exit-after-model", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
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
