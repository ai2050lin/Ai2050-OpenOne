#!/usr/bin/env python3
"""
Phase 702: Source-Restricted Channel Ensemble Audit.

Phase 701 showed that Phase698 top heads contain channel-level specificity, but
small channel sets do not fully restore DS7B. Phase 702 tests whether the
effective channel ensemble can be explained by the Phase700 source combo:
target_value + answer_line + self_last.

The patch is source-restricted and channel-restricted:
  short + selected_channels(combo_contrib_terse - combo_contrib_short)
  terse + selected_channels(combo_contrib_short - combo_contrib_terse)
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import random
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
from phase132_source_value_contribution_cuda import compute_source_contribution, get_num_kv_heads, get_v_proj  # noqa: E402
from phase584_gate_repair import load_model_flash  # noqa: E402
from phase599_final_layer_washout_decomposition import get_final_norm  # noqa: E402
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
from phase687_l26_l27_value_support_state_decomposition import classify, paired_case_metadata  # noqa: E402
from phase693_boundary_attention_head_candidate_audit import head_meta  # noqa: E402
from phase694_boundary_head_source_token_attention_audit import token_groups  # noqa: E402
from phase697_answer_last_route_transfer_decomposition import transfer_layers  # noqa: E402


OUT_ROOT = Path("results/glm5_phase702_source_restricted_channel_ensemble_audit")
PHASE698_ROOT = Path("results/glm5_phase698_answer_last_attention_head_source_audit")
COMBO_GROUPS = ["target_value", "answer_line", "self_last"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_top_head_sets(model_name: str, top_k: int) -> tuple[list[dict[str, Any]], dict[int, list[int]]]:
    path = PHASE698_ROOT / f"phase698_{model_name}_candidate_scores.json"
    if not path.exists():
        raise FileNotFoundError(f"missing Phase698 candidate scores: {path}")
    rows = json.loads(path.read_text(encoding="utf-8"))[:top_k]
    head_sets: dict[int, list[int]] = defaultdict(list)
    for row in rows:
        head_sets[int(row["layer"])].append(int(row["head"]))
    return rows, dict(head_sets)


def source_positions(tokenizer, prompt: str, case: dict[str, Any], ids: list[int]) -> dict[str, list[int]]:
    groups = token_groups(tokenizer, prompt, case, ids)
    return {
        "target_value": [p for p in groups.get("target_value", []) if 0 <= p < len(ids)],
        "answer_line": [p for p in groups.get("answer_line", []) if 0 <= p < len(ids)],
        "self_last": [p for p in groups.get("self_last", []) if 0 <= p < len(ids)],
    }


def capture_state(
    model,
    tokenizer,
    device,
    prompt: str,
    case: dict[str, Any],
    scan_layers: list[int],
    direction: torch.Tensor,
    routes,
    expected_ids,
    layer_meta: dict[int, tuple[Any, int, int, int]],
) -> dict[str, Any]:
    captured_values: dict[int, torch.Tensor] = {}
    final_box: dict[str, torch.Tensor] = {}
    handles = []
    for li in scan_layers:
        attn = get_attention_module(get_layers(model)[li])
        v_proj = get_v_proj(attn)

        def v_out(_module, _inputs, output, li=li):
            captured_values[li] = output.detach().cpu()

        handles.append(v_proj.register_forward_hook(v_out))

    final_norm = get_final_norm(model)
    if final_norm is not None:
        def final_pre(_module, inputs):
            final_box["final"] = inputs[0][0, -1].detach().cpu()

        handles.append(final_norm.register_forward_pre_hook(final_pre))

    ids = tokenizer.encode(prompt, add_special_tokens=False)
    try:
        with torch.inference_mode():
            out = model(
                input_ids=torch.tensor([ids], device=device),
                return_dict=True,
                use_cache=False,
                output_attentions=True,
            )
        diag = classify(out.logits[0, -1].detach(), routes, expected_ids)
        attentions = {li: out.attentions[li].detach().float().cpu().numpy() for li in scan_layers}
    finally:
        for handle in handles:
            handle.remove()

    groups = source_positions(tokenizer, prompt, case, ids)
    answer_pos = len(ids) - 1
    contributions: dict[int, dict[str, torch.Tensor]] = {}
    for li in scan_layers:
        _o_proj, n_heads, _head_dim, n_kv_heads = layer_meta[li]
        contributions[li] = {}
        for group_name, positions in groups.items():
            contributions[li][group_name] = compute_source_contribution(
                attentions[li],
                captured_values[li],
                [answer_pos],
                [positions],
                n_heads,
                n_kv_heads,
            )[0].detach().cpu()
    final = final_box.get("final")
    diag["final_proj"] = projection(final, direction.detach().cpu()) if final is not None else None
    return {
        "diag": diag,
        "contributions": contributions,
        "group_lengths": {k: len(v) for k, v in groups.items()},
    }


def combo_contribution(state: dict[str, Any], li: int) -> torch.Tensor:
    total = None
    for group_name in COMBO_GROUPS:
        contrib = state["contributions"][li][group_name]
        total = contrib if total is None else total + contrib
    return total.reshape(-1)


def score_channels_for_case(
    layer_meta: dict[int, tuple[Any, int, int, int]],
    head_sets: dict[int, list[int]],
    short_state: dict[str, Any],
    terse_state: dict[str, Any],
    direction: torch.Tensor,
) -> list[dict[str, Any]]:
    rows = []
    for li, heads in head_sets.items():
        o_proj, _n_heads, head_dim, _n_kv_heads = layer_meta[li]
        delta = (combo_contribution(terse_state, li) - combo_contribution(short_state, li)).float()
        out_dir_proj = torch.mv(o_proj.weight.detach().float().cpu().t(), direction.float())
        for head in heads:
            start = head * head_dim
            for ch in range(head_dim):
                idx = start + ch
                direct = float(delta[idx].item() * out_dir_proj[idx].item())
                rows.append({
                    "layer": li,
                    "head": head,
                    "channel": ch,
                    "index": idx,
                    "direct_effect": direct,
                    "abs_effect": abs(direct),
                    "combo_delta_value": float(delta[idx].item()),
                    "output_dir_proj": float(out_dir_proj[idx].item()),
                })
    return rows


def make_delta_patch_sets(
    src_state: dict[str, Any],
    dst_state: dict[str, Any],
    channel_rows: list[dict[str, Any]],
    scale: float = 1.0,
) -> list[dict[str, Any]]:
    by_layer: dict[int, list[int]] = defaultdict(list)
    for row in channel_rows:
        by_layer[int(row["layer"])].append(int(row["index"]))
    patches = []
    for li, indices in by_layer.items():
        delta = scale * (combo_contribution(src_state, li) - combo_contribution(dst_state, li))
        masked = torch.zeros_like(delta)
        for idx in sorted(set(indices)):
            if 0 <= idx < masked.numel():
                masked[idx] = delta[idx]
        patches.append({"layer": li, "delta_full": masked})
    return patches


def install_delta_hooks(model, patches: list[dict[str, Any]], layer_meta: dict[int, tuple[Any, int, int, int]]):
    handles = []
    for patch in patches:
        li = int(patch["layer"])
        delta_full = patch["delta_full"]
        o_proj, _n_heads, _head_dim, _n_kv_heads = layer_meta[li]

        def pre_hook(_module, inputs, delta_full=delta_full):
            x = inputs[0]
            y = x.clone()
            delta = delta_full.to(device=y.device, dtype=y.dtype)
            y[0, -1, :] = y[0, -1, :] + delta
            return (y,) + tuple(inputs[1:])

        handles.append(o_proj.register_forward_pre_hook(pre_hook))
    return handles


def run_delta_patched(
    model,
    tokenizer,
    device,
    prompt: str,
    patches: list[dict[str, Any]],
    direction: torch.Tensor,
    routes,
    expected_ids,
    layer_meta: dict[int, tuple[Any, int, int, int]],
) -> dict[str, Any]:
    handles = install_delta_hooks(model, patches, layer_meta)
    final_box: dict[str, torch.Tensor] = {}
    final_norm = get_final_norm(model)
    if final_norm is not None:
        def final_pre(_module, inputs):
            final_box["final"] = inputs[0][0, -1].detach().cpu()

        handles.append(final_norm.register_forward_pre_hook(final_pre))

    ids = tokenizer.encode(prompt, add_special_tokens=False)
    try:
        with torch.inference_mode():
            out = model(input_ids=torch.tensor([ids], device=device), return_dict=True, use_cache=False)
        diag = classify(out.logits[0, -1].detach(), routes, expected_ids)
    finally:
        for handle in handles:
            handle.remove()
    final = final_box.get("final")
    diag["final_proj"] = projection(final, direction.detach().cpu()) if final is not None else None
    return diag


def make_row(
    meta,
    case_id: str,
    phase_kind: str,
    condition: str,
    n_channels: int,
    short_diag: dict[str, Any],
    terse_diag: dict[str, Any],
    patched: dict[str, Any],
) -> dict[str, Any]:
    if phase_kind == "restore":
        success_change = (not short_diag["expected_top1"]) and patched["expected_top1"]
        rank_effect = short_diag["expected_rank"] - patched["expected_rank"]
        pmv_effect = short_diag["prose_minus_value"] - patched["prose_minus_value"]
        final_effect = None if short_diag["final_proj"] is None or patched["final_proj"] is None else patched["final_proj"] - short_diag["final_proj"]
    else:
        success_change = terse_diag["expected_top1"] and not patched["expected_top1"]
        rank_effect = patched["expected_rank"] - terse_diag["expected_rank"]
        pmv_effect = patched["prose_minus_value"] - terse_diag["prose_minus_value"]
        final_effect = None if terse_diag["final_proj"] is None or patched["final_proj"] is None else terse_diag["final_proj"] - patched["final_proj"]
    return {
        "case_id": case_id,
        "family": meta[case_id]["family"],
        "relation": meta[case_id]["relation"],
        "value": meta[case_id]["value"],
        "phase_kind": phase_kind,
        "condition": condition,
        "n_channels": n_channels,
        "patched_rank": patched["expected_rank"],
        "patched_top1": patched["expected_top1"],
        "success_change": success_change,
        "rank_effect": rank_effect,
        "pmv_effect": pmv_effect,
        "final_proj_effect": final_effect,
        "patched_best_other_route": patched["best_other_route"],
    }


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    finals = [r["final_proj_effect"] for r in rows if r["final_proj_effect"] is not None]
    return {
        "n": n,
        "success_change_rate": sum(1 for r in rows if r["success_change"]) / n,
        "patched_top1_rate": sum(1 for r in rows if r["patched_top1"]) / n,
        "mean_patched_rank": sum(r["patched_rank"] for r in rows) / n,
        "mean_rank_effect": sum(r["rank_effect"] for r in rows) / n,
        "mean_pmv_effect": sum(r["pmv_effect"] for r in rows) / n,
        "mean_final_proj_effect": sum(finals) / max(1, len(finals)),
        "mean_n_channels": sum(r["n_channels"] for r in rows) / n,
        "patched_best_other_route": dict(Counter(r["patched_best_other_route"] for r in rows).most_common()),
    }


def condition_channel_sets(channel_scores: list[dict[str, Any]], counts: list[int], seed: int) -> dict[str, list[dict[str, Any]]]:
    positive = [row for row in channel_scores if row["mean_direct_effect"] > 0]
    universe = channel_scores[:]
    rng = random.Random(seed)
    conditions = {"all_positive_source_channels": positive}
    for count in counts:
        conditions[f"source_top_channel_{count}"] = positive[: min(count, len(positive))]
        random_pool = universe[:]
        rng.shuffle(random_pool)
        conditions[f"source_random_channel_{count}"] = random_pool[: min(count, len(random_pool))]
    return conditions


def summarize_model(model_name: str, rows: list[dict[str, Any]], channel_scores: list[dict[str, Any]], paired_ids: list[str]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["phase_kind"], row["condition"])].append(row)
    by_condition = {f"{kind}|{cond}": summarize_group(vals) for (kind, cond), vals in grouped.items()}
    restore = sorted(
        [{"condition": key, **vals} for key, vals in by_condition.items() if key.startswith("restore|")],
        key=lambda r: (r["success_change_rate"], r["mean_final_proj_effect"], r["mean_rank_effect"]),
        reverse=True,
    )
    degradation = sorted(
        [{"condition": key, **vals} for key, vals in by_condition.items() if key.startswith("degradation|")],
        key=lambda r: (r["success_change_rate"], r["mean_final_proj_effect"], r["mean_rank_effect"]),
        reverse=True,
    )
    return {
        "model": model_name,
        "n_paired_cases": len(paired_ids),
        "n_rows": len(rows),
        "best_restore_conditions": restore,
        "best_degradation_conditions": degradation,
        "top_channel_scores": channel_scores[:64],
    }


def run_model(args) -> dict[str, Any]:
    paired_ids = select_paired_cases(args.model, args.limit)
    case_map = {case["case_id"]: case for case in select_base_cases()}
    meta = paired_case_metadata(case_map, paired_ids)
    top_head_rows, head_sets = load_top_head_sets(args.model, args.top_heads)
    model, tokenizer, device = load_model_flash(args.model)
    rows: list[dict[str, Any]] = []
    channel_acc: dict[tuple[int, int, int], list[dict[str, float]]] = defaultdict(list)
    case_cache: dict[str, dict[str, Any]] = {}
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
            short_state = capture_state(
                model, tokenizer, device, short_prompt, case, scan_layers, direction, routes, expected_ids, layer_meta
            )
            terse_state = capture_state(
                model, tokenizer, device, terse_prompt, case, scan_layers, direction, routes, expected_ids, layer_meta
            )
            case_cache[case_id] = {
                "short_prompt": short_prompt,
                "terse_prompt": terse_prompt,
                "routes": routes,
                "expected_ids": expected_ids,
                "direction": direction,
                "short_state": short_state,
                "terse_state": terse_state,
            }
            for row in score_channels_for_case(layer_meta, head_sets, short_state, terse_state, direction):
                key = (int(row["layer"]), int(row["head"]), int(row["channel"]))
                channel_acc[key].append({
                    "direct_effect": row["direct_effect"],
                    "abs_effect": row["abs_effect"],
                    "combo_delta_value": row["combo_delta_value"],
                    "output_dir_proj": row["output_dir_proj"],
                })
            if idx % args.log_every == 0 or idx == len(paired_ids):
                log(f"{args.model}: captured/source-scored {idx}/{len(paired_ids)} paired cases")

        channel_scores = []
        for (li, head, ch), vals in channel_acc.items():
            channel_scores.append({
                "layer": li,
                "head": head,
                "channel": ch,
                "index": head * layer_meta[li][2] + ch,
                "unit_key": f"L{li}H{head}C{ch}",
                "mean_direct_effect": sum(v["direct_effect"] for v in vals) / len(vals),
                "mean_abs_effect": sum(v["abs_effect"] for v in vals) / len(vals),
                "mean_combo_delta_value": sum(v["combo_delta_value"] for v in vals) / len(vals),
                "mean_output_dir_proj": sum(v["output_dir_proj"] for v in vals) / len(vals),
            })
        channel_scores.sort(key=lambda r: r["mean_direct_effect"], reverse=True)
        counts = [int(x) for x in args.channel_counts.split(",") if x.strip()]
        conditions = condition_channel_sets(channel_scores, counts, args.seed)

        for idx, case_id in enumerate(paired_ids, 1):
            cur = case_cache[case_id]
            short_state = cur["short_state"]
            terse_state = cur["terse_state"]
            for cond_name, selected in conditions.items():
                for phase_kind, prompt, src_state, dst_state in [
                    ("restore", cur["short_prompt"], terse_state, short_state),
                    ("degradation", cur["terse_prompt"], short_state, terse_state),
                ]:
                    patches = make_delta_patch_sets(src_state, dst_state, selected)
                    patched = run_delta_patched(
                        model,
                        tokenizer,
                        device,
                        prompt,
                        patches,
                        cur["direction"],
                        cur["routes"],
                        cur["expected_ids"],
                        layer_meta,
                    )
                    rows.append(make_row(
                        meta,
                        case_id,
                        phase_kind,
                        cond_name,
                        len(selected),
                        short_state["diag"],
                        terse_state["diag"],
                        patched,
                    ))
            if idx % args.log_every == 0 or idx == len(paired_ids):
                log(f"{args.model}: source-channel patched {idx}/{len(paired_ids)} paired cases")
    finally:
        release_model(model)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = summarize_model(args.model, rows, channel_scores, paired_ids)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / f"phase702_{args.model}_source_channel_rows.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    payload = {
        "phase": 702,
        "title": "Source-Restricted Channel Ensemble Audit",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": args.model,
        "transfer_layers": scan_layers,
        "top_heads": args.top_heads,
        "source_combo": COMBO_GROUPS,
        "channel_counts": counts,
        "phase698_top_heads": top_head_rows,
        "n_paired_cases": len(paired_ids),
        "summary": summary,
    }
    (OUT_ROOT / f"phase702_{args.model}_source_channel_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return payload


def write_cross_summary() -> dict[str, Any]:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    models = []
    for path in sorted(OUT_ROOT.glob("phase702_*_source_channel_summary.json")):
        models.append(json.loads(path.read_text(encoding="utf-8")))
    payload = {
        "phase": 702,
        "title": "Source-Restricted Channel Ensemble Audit Cross-Model Summary",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": models,
    }
    (OUT_ROOT / "phase702_cross_model_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    lines = [
        "# Phase 702 Source-Restricted Channel Ensemble Audit",
        "",
        f"- generated: `{payload['timestamp']}`",
        "",
        "| model | pairs | layers | top_heads | best_restore | change | rank_effect | final_effect | best_degrade | drop | rank_effect | final_effect |",
        "|---|---:|---|---:|---|---:|---:|---:|---|---:|---:|---:|",
    ]
    for item in models:
        s = item["summary"]
        br = s["best_restore_conditions"][0] if s["best_restore_conditions"] else {}
        bd = s["best_degradation_conditions"][0] if s["best_degradation_conditions"] else {}
        lines.append(
            f"| {item['model']} | {item['n_paired_cases']} | {item['transfer_layers']} | {item['top_heads']} | "
            f"{br.get('condition','')} | {br.get('success_change_rate',0.0):.3f} | {br.get('mean_rank_effect',0.0):.2f} | {br.get('mean_final_proj_effect',0.0):.3f} | "
            f"{bd.get('condition','')} | {bd.get('success_change_rate',0.0):.3f} | {bd.get('mean_rank_effect',0.0):.2f} | {bd.get('mean_final_proj_effect',0.0):.3f} |"
        )
    for section, key in [("Best Restore", "best_restore_conditions"), ("Best Degradation", "best_degradation_conditions")]:
        lines.extend(["", f"## {section}", ""])
        for item in models:
            lines.append(f"### {item['model']}")
            lines.append("")
            lines.append("| condition | change | patched_top1 | patched_rank | rank_effect | final_effect | n_channels | best_other |")
            lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
            for row in item["summary"][key][:24]:
                lines.append(
                    f"| {row['condition']} | {row['success_change_rate']:.3f} | {row['patched_top1_rate']:.3f} | "
                    f"{row['mean_patched_rank']:.2f} | {row['mean_rank_effect']:.2f} | {row['mean_final_proj_effect']:.3f} | "
                    f"{row['mean_n_channels']:.1f} | {row['patched_best_other_route']} |"
                )
            lines.append("")
    (OUT_ROOT / "phase702_cross_model_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--hard-exit-after-model", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top-heads", type=int, default=32)
    parser.add_argument("--channel-counts", default="32,64,128,256,512")
    parser.add_argument("--seed", type=int, default=702)
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
