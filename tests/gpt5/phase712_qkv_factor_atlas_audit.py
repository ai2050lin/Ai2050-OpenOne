#!/usr/bin/env python3
"""
Phase 712: QK-V Factor Atlas Audit.

Phase 711 initialized a mechanism atlas, but most rows still inherit
condition-level evidence. This phase adds a lower-level factor diagnostic for
the selected source-channel effect:

  total source contribution delta
    = QK/addressing-weight term
    + V/content term
    + QK*V interaction term

This is not a full causal Q/K replacement. It is a measured decomposition of
the same source contribution used by Phase 707-710, designed to decide whether
the next expensive causal test should focus on attention pattern, value
content, or their coupling.
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

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))

from model_utils import get_layers, release_model  # noqa: E402
from phase112_attention_transport_head_mapping_cuda import get_attention_module  # noqa: E402
from phase132_source_value_contribution_cuda import get_num_kv_heads, get_v_proj  # noqa: E402
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
    select_paired_cases,
    value_minus_prose_direction,
)
from phase693_boundary_attention_head_candidate_audit import head_meta  # noqa: E402
from phase697_answer_last_route_transfer_decomposition import transfer_layers  # noqa: E402
from phase707_full_value_phrase_likelihood_audit import (  # noqa: E402
    COMBO_GROUPS,
    build_channel_scores,
    load_top_head_sets,
    source_positions,
)


OUT_ROOT = Path("results/glm5_phase712_qkv_factor_atlas_audit")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def capture_factor_state(
    model,
    tokenizer,
    device,
    prompt: str,
    case: dict[str, Any],
    scan_layers: list[int],
    layer_meta: dict[int, tuple[Any, int, int, int]],
) -> dict[str, Any]:
    value_outputs: dict[int, torch.Tensor] = {}
    handles = []
    layers = get_layers(model)
    for li in scan_layers:
        attn = get_attention_module(layers[li])
        v_proj = get_v_proj(attn)

        def v_hook(_module, _inputs, output, li=li):
            value_outputs[li] = output.detach().float().cpu()

        handles.append(v_proj.register_forward_hook(v_hook))

    ids = tokenizer.encode(prompt, add_special_tokens=False)
    try:
        with torch.inference_mode():
            out = model(
                input_ids=torch.tensor([ids], device=device),
                return_dict=True,
                use_cache=False,
                output_attentions=True,
            )
        attentions = {li: out.attentions[li].detach().float().cpu().numpy() for li in scan_layers}
    finally:
        for handle in handles:
            handle.remove()
    return {
        "ids": ids,
        "answer_pos": len(ids) - 1,
        "groups": source_positions(tokenizer, prompt, case, ids),
        "attentions": attentions,
        "value_outputs": value_outputs,
    }


def values_by_head(value_output: torch.Tensor, num_heads: int, num_kv_heads: int) -> torch.Tensor:
    values = value_output.detach().float().cpu()
    batch, seq_len, value_width = values.shape
    head_dim = value_width // num_kv_heads
    values = values.view(batch, seq_len, num_kv_heads, head_dim)
    if num_kv_heads != num_heads:
        repeat = num_heads // num_kv_heads
        values = values.repeat_interleave(repeat, dim=2)
    return values[0]


def attn_for_answer(attn_weights: np.ndarray, answer_pos: int) -> torch.Tensor:
    arr = np.asarray(attn_weights[0, :, answer_pos, :], dtype=np.float32)
    return torch.tensor(arr, dtype=torch.float32)


def combo_factor_terms(
    short_state: dict[str, Any],
    terse_state: dict[str, Any],
    layer: int,
    num_heads: int,
    num_kv_heads: int,
) -> dict[str, torch.Tensor]:
    short_attn = attn_for_answer(short_state["attentions"][layer], short_state["answer_pos"])
    terse_attn = attn_for_answer(terse_state["attentions"][layer], terse_state["answer_pos"])
    short_values = values_by_head(short_state["value_outputs"][layer], num_heads, num_kv_heads)
    terse_values = values_by_head(terse_state["value_outputs"][layer], num_heads, num_kv_heads)

    head_dim = short_values.shape[-1]
    total = torch.zeros((num_heads, head_dim), dtype=torch.float32)
    qk = torch.zeros_like(total)
    v = torch.zeros_like(total)
    interaction = torch.zeros_like(total)
    for group_name in COMBO_GROUPS:
        src = short_state["groups"].get(group_name, [])
        src = [p for p in src if 0 <= p < short_values.shape[0] and 0 <= p < terse_values.shape[0]]
        if not src:
            continue
        sa = short_attn[:, src]
        ta = terse_attn[:, src]
        sv = short_values[src, :, :].permute(1, 0, 2)
        tv = terse_values[src, :, :].permute(1, 0, 2)

        # terse - short, matching the positive delta screen in Phase 707.
        total += (ta.unsqueeze(-1) * tv).sum(dim=1) - (sa.unsqueeze(-1) * sv).sum(dim=1)
        qk += ((ta - sa).unsqueeze(-1) * sv).sum(dim=1)
        v += (sa.unsqueeze(-1) * (tv - sv)).sum(dim=1)
        interaction += ((ta - sa).unsqueeze(-1) * (tv - sv)).sum(dim=1)
    return {
        "total": total.reshape(-1),
        "qk": qk.reshape(-1),
        "v": v.reshape(-1),
        "interaction": interaction.reshape(-1),
    }


def score_factor_channels(
    layer_meta: dict[int, tuple[Any, int, int, int]],
    head_sets: dict[int, list[int]],
    short_state: dict[str, Any],
    terse_state: dict[str, Any],
    direction: torch.Tensor,
) -> list[dict[str, Any]]:
    rows = []
    direction = direction.detach().float().cpu()
    for li, heads in head_sets.items():
        o_proj, _n_heads, head_dim, n_kv_heads = layer_meta[li]
        n_heads = int(o_proj.in_features) // head_dim
        terms = combo_factor_terms(short_state, terse_state, li, n_heads, n_kv_heads)
        out_dir_proj = torch.mv(o_proj.weight.detach().float().cpu().t(), direction)
        for head in heads:
            start = head * head_dim
            for ch in range(head_dim):
                idx = start + ch
                total_direct = float(terms["total"][idx].item() * out_dir_proj[idx].item())
                qk_direct = float(terms["qk"][idx].item() * out_dir_proj[idx].item())
                v_direct = float(terms["v"][idx].item() * out_dir_proj[idx].item())
                interaction_direct = float(terms["interaction"][idx].item() * out_dir_proj[idx].item())
                residual = total_direct - qk_direct - v_direct - interaction_direct
                rows.append({
                    "layer": li,
                    "head": head,
                    "channel": ch,
                    "index": idx,
                    "direct_effect": total_direct,
                    "abs_effect": abs(total_direct),
                    "combo_delta_value": float(terms["total"][idx].item()),
                    "output_dir_proj": float(out_dir_proj[idx].item()),
                    "total_direct": total_direct,
                    "qk_direct": qk_direct,
                    "v_direct": v_direct,
                    "interaction_direct": interaction_direct,
                    "residual_direct": residual,
                    "abs_qk_direct": abs(qk_direct),
                    "abs_v_direct": abs(v_direct),
                    "abs_interaction_direct": abs(interaction_direct),
                })
    return rows


def mean(vals: list[float]) -> float | None:
    return None if not vals else sum(vals) / len(vals)


def classify_factor(qk: float, v: float, interaction: float) -> str:
    mags = {"qk_addressing": abs(qk), "v_content": abs(v), "qk_v_interaction": abs(interaction)}
    best, best_val = max(mags.items(), key=lambda x: x[1])
    second = sorted(mags.values(), reverse=True)[1]
    if best_val < 1e-9:
        return "weak_or_zero"
    if best_val >= 1.35 * max(second, 1e-12):
        return best
    return "mixed_coupled"


def summarize_selected(name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    def val(row: dict[str, Any], key: str) -> float:
        if key in row:
            return float(row[key])
        if key == "total_direct" and "mean_direct_effect" in row:
            return float(row.get("mean_direct_effect", 0.0) or 0.0)
        mean_key = f"mean_{key}"
        return float(row.get(mean_key, 0.0) or 0.0)

    qk_vals = [val(r, "qk_direct") for r in rows]
    v_vals = [val(r, "v_direct") for r in rows]
    inter_vals = [val(r, "interaction_direct") for r in rows]
    total_vals = [val(r, "total_direct") for r in rows]
    abs_den = sum(abs(x) for x in qk_vals) + sum(abs(x) for x in v_vals) + sum(abs(x) for x in inter_vals)
    labels = [classify_factor(q, v, i) for q, v, i in zip(qk_vals, v_vals, inter_vals)]
    return {
        "condition": name,
        "n": n,
        "mean_total_direct": mean(total_vals),
        "mean_qk_direct": mean(qk_vals),
        "mean_v_direct": mean(v_vals),
        "mean_interaction_direct": mean(inter_vals),
        "sum_total_direct": sum(total_vals),
        "sum_qk_direct": sum(qk_vals),
        "sum_v_direct": sum(v_vals),
        "sum_interaction_direct": sum(inter_vals),
        "abs_qk_share": 0.0 if abs_den == 0 else sum(abs(x) for x in qk_vals) / abs_den,
        "abs_v_share": 0.0 if abs_den == 0 else sum(abs(x) for x in v_vals) / abs_den,
        "abs_interaction_share": 0.0 if abs_den == 0 else sum(abs(x) for x in inter_vals) / abs_den,
        "factor_label_counts": dict(Counter(labels).most_common()),
        "dominant_factor": classify_factor(sum(qk_vals), sum(v_vals), sum(inter_vals)),
    }


def compact_conditions(channel_scores: list[dict[str, Any]], count: int, seed: int) -> dict[str, list[dict[str, Any]]]:
    positive = [r for r in channel_scores if float(r["mean_direct_effect"]) > 0]
    rng = np.random.default_rng(seed)
    random_rows = channel_scores[:]
    if random_rows:
        order = rng.permutation(len(random_rows)).tolist()
        random_rows = [random_rows[i] for i in order]
    return {
        "source_top_channel": positive[: min(count, len(positive))],
        "source_random_channel": random_rows[: min(count, len(random_rows))],
        "all_positive_source_channels": positive,
    }


def summarize_model(
    model_name: str,
    paired_ids: list[str],
    channel_scores: list[dict[str, Any]],
    conditions: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    selected = {name: summarize_selected(name, rows) for name, rows in conditions.items()}
    by_layer_head: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in channel_scores:
        by_layer_head[(int(row["layer"]), int(row["head"]))].append(row)
    head_rows = []
    for (li, head), vals in by_layer_head.items():
        top_vals = sorted(vals, key=lambda r: float(r["mean_direct_effect"]), reverse=True)[: min(8, len(vals))]
        s = summarize_selected(f"L{li}H{head}_top8", top_vals)
        head_rows.append({
            "layer": li,
            "head": head,
            "n_channels": len(vals),
            "top8": s,
        })
    head_rows.sort(key=lambda r: float(r["top8"]["sum_total_direct"] or 0.0), reverse=True)
    return {
        "model": model_name,
        "n_paired_cases": len(paired_ids),
        "n_channel_scores": len(channel_scores),
        "conditions": selected,
        "top_heads_by_factor": head_rows[:32],
    }


def run_model(args) -> dict[str, Any]:
    paired_ids = select_paired_cases(args.model, args.limit)
    case_map = {case["case_id"]: case for case in select_base_cases()}
    top_head_rows, head_sets = load_top_head_sets(args.model, args.top_heads)
    model, tokenizer, device = load_model_flash(args.model)
    all_score_rows: list[dict[str, Any]] = []
    try:
        dtype = next(model.parameters()).dtype
        layers = get_layers(model)
        scan_layers = transfer_layers(args.model, len(layers))
        layer_meta: dict[int, tuple[Any, int, int, int]] = {}
        for li in scan_layers:
            o_proj, n_heads, head_dim = head_meta(model, li)
            attn = get_attention_module(layers[li])
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
            short_state = capture_factor_state(model, tokenizer, device, short_prompt, case, scan_layers, layer_meta)
            terse_state = capture_factor_state(model, tokenizer, device, terse_prompt, case, scan_layers, layer_meta)
            case_rows = score_factor_channels(layer_meta, head_sets, short_state, terse_state, direction)
            for row in case_rows:
                row["case_id"] = case_id
            all_score_rows.extend(case_rows)
            if idx % args.log_every == 0 or idx == len(paired_ids):
                log(f"{args.model}: qkv-factor scored {idx}/{len(paired_ids)} paired cases")
    finally:
        release_model(model)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    channel_scores = build_channel_scores(all_score_rows, layer_meta)
    for row in channel_scores:
        vals = [
            r for r in all_score_rows
            if int(r["layer"]) == int(row["layer"])
            and int(r["head"]) == int(row["head"])
            and int(r["channel"]) == int(row["channel"])
        ]
        row["mean_qk_direct"] = mean([float(v["qk_direct"]) for v in vals])
        row["mean_v_direct"] = mean([float(v["v_direct"]) for v in vals])
        row["mean_interaction_direct"] = mean([float(v["interaction_direct"]) for v in vals])
        row["factor_label"] = classify_factor(
            float(row["mean_qk_direct"] or 0.0),
            float(row["mean_v_direct"] or 0.0),
            float(row["mean_interaction_direct"] or 0.0),
        )
    conditions = compact_conditions(channel_scores, args.channel_count, args.seed)
    summary = summarize_model(args.model, paired_ids, channel_scores, conditions)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / f"phase712_{args.model}_qkv_factor_case_rows.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in all_score_rows) + "\n",
        encoding="utf-8",
    )
    (OUT_ROOT / f"phase712_{args.model}_qkv_factor_channel_scores.json").write_text(
        json.dumps(channel_scores, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    payload = {
        "phase": 712,
        "title": "QK-V Factor Atlas Audit",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": args.model,
        "top_heads": args.top_heads,
        "channel_count": args.channel_count,
        "source_combo": COMBO_GROUPS,
        "transfer_layers": scan_layers,
        "phase698_top_heads": top_head_rows,
        "summary": summary,
        "notes": [
            "This is a measured contribution decomposition, not a full causal Q/K replacement.",
            "QK/addressing term uses short-state V content while swapping terse-vs-short attention weights.",
            "V/content term uses short-state attention weights while swapping terse-vs-short V content.",
        ],
    }
    (OUT_ROOT / f"phase712_{args.model}_qkv_factor_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return payload


def write_cross_summary() -> dict[str, Any]:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    models = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(OUT_ROOT.glob("phase712_*_qkv_factor_summary.json"))]
    payload = {
        "phase": 712,
        "title": "QK-V Factor Atlas Audit Cross-Model Summary",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": models,
    }
    (OUT_ROOT / "phase712_cross_model_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    lines = [
        "# Phase 712 QK-V Factor Atlas Audit",
        "",
        f"- generated: `{payload['timestamp']}`",
        "",
        "| model | cases | condition | n | dominant | abs_qk | abs_v | abs_interaction | sum_total | sum_qk | sum_v | sum_interaction |",
        "|---|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in models:
        s = item["summary"]
        for cond in ["source_top_channel", "all_positive_source_channels", "source_random_channel"]:
            row = s["conditions"].get(cond, {})
            lines.append(
                f"| {item['model']} | {s['n_paired_cases']} | {cond} | {row.get('n', 0)} | {row.get('dominant_factor', '')} | "
                f"{row.get('abs_qk_share', 0.0):.3f} | {row.get('abs_v_share', 0.0):.3f} | {row.get('abs_interaction_share', 0.0):.3f} | "
                f"{row.get('sum_total_direct', 0.0):.6f} | {row.get('sum_qk_direct', 0.0):.6f} | "
                f"{row.get('sum_v_direct', 0.0):.6f} | {row.get('sum_interaction_direct', 0.0):.6f} |"
            )
    for item in models:
        lines.extend(["", f"## {item['model']} top heads", ""])
        lines.append("| layer | head | dominant | abs_qk | abs_v | abs_interaction | sum_total |")
        lines.append("|---:|---:|---|---:|---:|---:|---:|")
        for row in item["summary"]["top_heads_by_factor"][:12]:
            top = row["top8"]
            lines.append(
                f"| {row['layer']} | {row['head']} | {top['dominant_factor']} | "
                f"{top['abs_qk_share']:.3f} | {top['abs_v_share']:.3f} | {top['abs_interaction_share']:.3f} | "
                f"{top['sum_total_direct']:.6f} |"
            )
    (OUT_ROOT / "phase712_cross_model_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--hard-exit-after-model", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top-heads", type=int, default=32)
    parser.add_argument("--channel-count", type=int, default=512)
    parser.add_argument("--seed", type=int, default=712)
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
