#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))

from model_utils import get_layers, release_model  # noqa: E402
from phase693_boundary_attention_head_candidate_audit import head_meta  # noqa: E402
from phase722_functional_head_atlas_causal_ablation import install_zero_head_ablation, logit_diag, target_token_ids, write_json, write_jsonl  # noqa: E402
from phase723_apple_fruit_attribute_micro_atlas import prompt_for  # noqa: E402
from phase733_prompt_type_skeleton_source_localization import (  # noqa: E402
    MODELS,
    get_tensor,
    load_model_bf16_flash,
    select_prompt_pairs,
    site_kind_layer,
    site_module,
)


OUT_ROOT = Path("results/glm5_phase734_prompt_type_skeleton_writer_decomposition")
PHASE733_ROOT = Path("results/glm5_phase733_prompt_type_skeleton_source_localization")

FALLBACK_WINDOWS = {
    "qwen3": {"target_site": "hidden_36", "layers": [26, 28, 34, 35]},
    "glm4": {"target_site": "hidden_40", "layers": [23, 31, 35, 38, 39]},
    "deepseek7b": {"target_site": "hidden_28", "layers": [21, 22, 25, 27]},
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def norm(vec: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(vec.float()).item())


def dot(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.dot(a.float().flatten(), b.float().flatten()).item())


def safe_mean(vals: list[float | int | None]) -> float | None:
    xs = [float(v) for v in vals if v is not None]
    return sum(xs) / len(xs) if xs else None


def load_phase733_window(model_name: str, round_name: str) -> dict[str, Any]:
    path = PHASE733_ROOT / round_name / f"phase733_{model_name}_summary.json"
    if not path.exists():
        return FALLBACK_WINDOWS[model_name]
    data = json.loads(path.read_text(encoding="utf-8"))
    layers: set[int] = set()
    target_site = FALLBACK_WINDOWS[model_name]["target_site"]
    best_delta = -1e9
    for rec in data.get("formation_summary", {}).values():
        for key in ["earliest_35pct_layer", "max_layer"]:
            if rec.get(key) is not None:
                layers.add(int(rec[key]))
    for key, rec in data.get("transfer_summary", {}).items():
        if "commonsense<-explicit" not in key:
            continue
        delta = rec.get("mean_patched_delta_vs_recipient")
        if delta is not None and float(delta) > best_delta:
            best_delta = float(delta)
            target_site = key.split("|", 1)[1]
    if not layers:
        layers.update(FALLBACK_WINDOWS[model_name]["layers"])
    return {"target_site": target_site, "layers": sorted(layers)}


def select_evenly(n_items: int, max_items: int | None) -> list[int]:
    if max_items is None or max_items >= n_items:
        return list(range(n_items))
    if max_items <= 1:
        return [0]
    idxs = []
    for i in range(max_items):
        idx = round(i * (n_items - 1) / (max_items - 1))
        if idx not in idxs:
            idxs.append(idx)
    return idxs


def install_mlp_group_ablation(model, layer_idx: int, start: int, end: int):
    layer = get_layers(model)[layer_idx]
    module = layer.mlp

    def hook(_module, _inputs, output):
        if isinstance(output, tuple):
            y = output[0].clone()
            y[0, -1, start:end] = 0
            return (y,) + output[1:]
        y = output.clone()
        y[0, -1, start:end] = 0
        return y

    return [module.register_forward_hook(hook)]


def first_token_diag(logits: torch.Tensor, tokenizer, answer: str) -> dict[str, Any]:
    tid = target_token_ids(tokenizer, answer)[0]
    return logit_diag(logits, int(tid))


def forward_site_and_logits(
    model,
    device,
    ids: list[int],
    target_site: str,
    install_ablation: Callable[[], list[Any]] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    captured: dict[str, torch.Tensor] = {}
    handles = install_ablation() if install_ablation else []
    module = site_module(model, target_site)

    def hook(_module, _inputs, output):
        captured["vec"] = get_tensor(output)[0, -1].detach().float().cpu()

    handles.append(module.register_forward_hook(hook))
    try:
        with torch.inference_mode():
            out = model(input_ids=torch.tensor([ids], device=device), return_dict=True, use_cache=False)
        return captured["vec"], out.logits[0, -1].detach().float().cpu()
    finally:
        for h in handles:
            h.remove()


def scan_attention_writers(
    model,
    tokenizer,
    device,
    model_name: str,
    target_site: str,
    layers: list[int],
    pairs: list[dict[str, Any]],
    max_heads_per_layer: int | None,
    log_every: int,
) -> list[dict[str, Any]]:
    rows = []
    head_specs: list[dict[str, int]] = []
    for layer_idx in layers:
        _o_proj, n_heads, _head_dim = head_meta(model, layer_idx)
        for head_idx in select_evenly(n_heads, max_heads_per_layer):
            head_specs.append({"layer": int(layer_idx), "head": int(head_idx)})
    log(f"{model_name}: attention head specs={len(head_specs)} over layers={layers}")

    for pair_idx, pair in enumerate(pairs, 1):
        common = pair["commonsense"]
        explicit = pair["explicit_profile"]
        c_ids = tokenizer.encode(prompt_for(common), add_special_tokens=False)
        e_ids = tokenizer.encode(prompt_for(explicit), add_special_tokens=False)
        answer = common["answer"]
        e_base_vec, e_base_logits = forward_site_and_logits(model, device, e_ids, target_site)
        c_base_vec, c_base_logits = forward_site_and_logits(model, device, c_ids, target_site)
        skeleton = e_base_vec - c_base_vec
        skeleton_norm = norm(skeleton)
        if skeleton_norm <= 1e-9:
            continue
        d_hat = skeleton / skeleton_norm
        e_base_diag = first_token_diag(e_base_logits, tokenizer, answer)
        c_base_diag = first_token_diag(c_base_logits, tokenizer, answer)
        for spec in head_specs:
            layer_idx = spec["layer"]
            head_idx = spec["head"]

            def install(spec=spec):
                return install_zero_head_ablation(model, [spec])

            e_ab_vec, e_ab_logits = forward_site_and_logits(model, device, e_ids, target_site, install)
            c_ab_vec, c_ab_logits = forward_site_and_logits(model, device, c_ids, target_site, install)
            e_ab_diag = first_token_diag(e_ab_logits, tokenizer, answer)
            c_ab_diag = first_token_diag(c_ab_logits, tokenizer, answer)
            e_shift = e_ab_vec - e_base_vec
            c_shift = c_ab_vec - c_base_vec
            e_proj_delta = dot(e_shift, d_hat)
            c_proj_delta = dot(c_shift, d_hat)
            rows.append(
                {
                    "model": model_name,
                    "component_type": "attention_head",
                    "component_id": f"L{layer_idx}H{head_idx}",
                    "layer": layer_idx,
                    "head": head_idx,
                    "target_site": target_site,
                    "pair_id": pair["pair_id"],
                    "object": common["object"],
                    "relation": common["relation"],
                    "answer": answer,
                    "baseline_skeleton_norm": skeleton_norm,
                    "explicit_projection_delta": e_proj_delta,
                    "explicit_skeleton_loss": -e_proj_delta,
                    "commonsense_projection_delta": c_proj_delta,
                    "explicit_target_delta_norm": norm(e_shift),
                    "commonsense_target_delta_norm": norm(c_shift),
                    "explicit_logprob_delta": e_ab_diag["target_logprob"] - e_base_diag["target_logprob"],
                    "commonsense_logprob_delta": c_ab_diag["target_logprob"] - c_base_diag["target_logprob"],
                    "explicit_rank_delta": e_ab_diag["target_rank"] - e_base_diag["target_rank"],
                    "commonsense_rank_delta": c_ab_diag["target_rank"] - c_base_diag["target_rank"],
                }
            )
        if pair_idx % log_every == 0 or pair_idx == len(pairs):
            log(f"{model_name}: attention writer scan {pair_idx}/{len(pairs)} pairs; rows={len(rows)}")
    return rows


def scan_mlp_writers(
    model,
    tokenizer,
    device,
    model_name: str,
    target_site: str,
    layers: list[int],
    pairs: list[dict[str, Any]],
    groups_per_layer: int,
    log_every: int,
) -> list[dict[str, Any]]:
    rows = []
    d_model = int(model.get_input_embeddings().weight.shape[1])
    group_specs = []
    for layer_idx in layers:
        step = max(1, (d_model + groups_per_layer - 1) // groups_per_layer)
        for start in range(0, d_model, step):
            group_specs.append({"layer": int(layer_idx), "start": int(start), "end": int(min(d_model, start + step))})
    log(f"{model_name}: mlp group specs={len(group_specs)} over layers={layers}, d_model={d_model}")

    for pair_idx, pair in enumerate(pairs, 1):
        common = pair["commonsense"]
        explicit = pair["explicit_profile"]
        c_ids = tokenizer.encode(prompt_for(common), add_special_tokens=False)
        e_ids = tokenizer.encode(prompt_for(explicit), add_special_tokens=False)
        answer = common["answer"]
        e_base_vec, e_base_logits = forward_site_and_logits(model, device, e_ids, target_site)
        c_base_vec, c_base_logits = forward_site_and_logits(model, device, c_ids, target_site)
        skeleton = e_base_vec - c_base_vec
        skeleton_norm = norm(skeleton)
        if skeleton_norm <= 1e-9:
            continue
        d_hat = skeleton / skeleton_norm
        e_base_diag = first_token_diag(e_base_logits, tokenizer, answer)
        c_base_diag = first_token_diag(c_base_logits, tokenizer, answer)
        for spec in group_specs:
            layer_idx = spec["layer"]
            start = spec["start"]
            end = spec["end"]

            def install(layer_idx=layer_idx, start=start, end=end):
                return install_mlp_group_ablation(model, layer_idx, start, end)

            e_ab_vec, e_ab_logits = forward_site_and_logits(model, device, e_ids, target_site, install)
            c_ab_vec, c_ab_logits = forward_site_and_logits(model, device, c_ids, target_site, install)
            e_ab_diag = first_token_diag(e_ab_logits, tokenizer, answer)
            c_ab_diag = first_token_diag(c_ab_logits, tokenizer, answer)
            e_shift = e_ab_vec - e_base_vec
            c_shift = c_ab_vec - c_base_vec
            e_proj_delta = dot(e_shift, d_hat)
            c_proj_delta = dot(c_shift, d_hat)
            rows.append(
                {
                    "model": model_name,
                    "component_type": "mlp_output_group",
                    "component_id": f"L{layer_idx}:mlp[{start}:{end}]",
                    "layer": layer_idx,
                    "start": start,
                    "end": end,
                    "target_site": target_site,
                    "pair_id": pair["pair_id"],
                    "object": common["object"],
                    "relation": common["relation"],
                    "answer": answer,
                    "baseline_skeleton_norm": skeleton_norm,
                    "explicit_projection_delta": e_proj_delta,
                    "explicit_skeleton_loss": -e_proj_delta,
                    "commonsense_projection_delta": c_proj_delta,
                    "explicit_target_delta_norm": norm(e_shift),
                    "commonsense_target_delta_norm": norm(c_shift),
                    "explicit_logprob_delta": e_ab_diag["target_logprob"] - e_base_diag["target_logprob"],
                    "commonsense_logprob_delta": c_ab_diag["target_logprob"] - c_base_diag["target_logprob"],
                    "explicit_rank_delta": e_ab_diag["target_rank"] - e_base_diag["target_rank"],
                    "commonsense_rank_delta": c_ab_diag["target_rank"] - c_base_diag["target_rank"],
                }
            )
        if pair_idx % log_every == 0 or pair_idx == len(pairs):
            log(f"{model_name}: mlp writer scan {pair_idx}/{len(pairs)} pairs; rows={len(rows)}")
    return rows


def summarize_component_rows(rows: list[dict[str, Any]], component_type: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["component_id"]].append(row)
    out = []
    for cid, vals in groups.items():
        explicit_loss = safe_mean([v["explicit_skeleton_loss"] for v in vals]) or 0.0
        explicit_logprob = safe_mean([v["explicit_logprob_delta"] for v in vals]) or 0.0
        commonsense_logprob = safe_mean([v["commonsense_logprob_delta"] for v in vals]) or 0.0
        target_norm = safe_mean([v["explicit_target_delta_norm"] for v in vals]) or 0.0
        if explicit_loss > 0 and explicit_logprob < 0:
            role = "writer_candidate"
        elif target_norm > 0 and abs(explicit_loss) < target_norm * 0.05:
            role = "carrier_or_rewriter_candidate"
        elif explicit_logprob < 0:
            role = "readout_support_candidate"
        else:
            role = "weak_or_mixed"
        rec = {
            "component_type": component_type,
            "component_id": cid,
            "layer": vals[0]["layer"],
            "n": len(vals),
            "mean_explicit_skeleton_loss": explicit_loss,
            "mean_commonsense_projection_delta": safe_mean([v["commonsense_projection_delta"] for v in vals]),
            "mean_explicit_logprob_delta": explicit_logprob,
            "mean_commonsense_logprob_delta": commonsense_logprob,
            "mean_explicit_rank_delta": safe_mean([v["explicit_rank_delta"] for v in vals]),
            "mean_commonsense_rank_delta": safe_mean([v["commonsense_rank_delta"] for v in vals]),
            "mean_explicit_target_delta_norm": target_norm,
            "mean_baseline_skeleton_norm": safe_mean([v["baseline_skeleton_norm"] for v in vals]),
            "role_guess": role,
        }
        if "head" in vals[0]:
            rec["head"] = vals[0]["head"]
        if "start" in vals[0]:
            rec["start"] = vals[0]["start"]
            rec["end"] = vals[0]["end"]
        out.append(rec)
    return sorted(out, key=lambda r: (r["mean_explicit_skeleton_loss"], -abs(r["mean_explicit_logprob_delta"])), reverse=True)


def build_summary(model_name: str, round_name: str, attn_impl: str, window: dict[str, Any], attention_rows: list[dict[str, Any]], mlp_rows: list[dict[str, Any]]) -> dict[str, Any]:
    attention_summary = summarize_component_rows(attention_rows, "attention_head")
    mlp_summary = summarize_component_rows(mlp_rows, "mlp_output_group")
    return {
        "phase": 734,
        "title": "Prompt-Type Skeleton Writer Decomposition",
        "model": model_name,
        "round": round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "quantization": "off",
        "dtype": "bfloat16",
        "target_site": window["target_site"],
        "scan_layers": window["layers"],
        "n_attention_rows": len(attention_rows),
        "n_mlp_rows": len(mlp_rows),
        "top_attention_writer_candidates": attention_summary[:16],
        "top_mlp_writer_candidates": mlp_summary[:16],
        "attention_role_counts": dict((r, sum(1 for x in attention_summary if x["role_guess"] == r)) for r in sorted({x["role_guess"] for x in attention_summary})),
        "mlp_role_counts": dict((r, sum(1 for x in mlp_summary if x["role_guess"] == r)) for r in sorted({x["role_guess"] for x in mlp_summary})),
        "strict_interpretation": "writer decomposition v0; ablation-caused downstream skeleton loss indicates writer/contributor candidates, not pure neuron-level proof",
    }


def run_model(args) -> dict[str, Any]:
    out_dir = OUT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = select_prompt_pairs(args.max_pairs)
    window = load_phase733_window(args.model, args.phase733_round)
    if args.max_layers and len(window["layers"]) > args.max_layers:
        idxs = select_evenly(len(window["layers"]), args.max_layers)
        window["layers"] = [window["layers"][i] for i in idxs]
    log(f"{args.model}/{args.round_name}: pairs={len(pairs)} target_site={window['target_site']} layers={window['layers']}")
    model, tokenizer, device, attn_impl = load_model_bf16_flash(args.model)
    try:
        attention_rows = scan_attention_writers(
            model,
            tokenizer,
            device,
            args.model,
            window["target_site"],
            window["layers"],
            pairs,
            args.max_heads_per_layer,
            args.log_every,
        )
        mlp_rows = scan_mlp_writers(
            model,
            tokenizer,
            device,
            args.model,
            window["target_site"],
            window["layers"],
            pairs,
            args.mlp_groups_per_layer,
            args.log_every,
        )
    finally:
        release_model(model)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    summary = build_summary(args.model, args.round_name, attn_impl, window, attention_rows, mlp_rows)
    write_jsonl(out_dir / f"phase734_{args.model}_attention_writer_rows.jsonl", attention_rows)
    write_jsonl(out_dir / f"phase734_{args.model}_mlp_writer_rows.jsonl", mlp_rows)
    write_json(out_dir / f"phase734_{args.model}_summary.json", summary)
    print(json.dumps({"model": args.model, "round": args.round_name, "target_site": window["target_site"], "layers": window["layers"], "top_attention": summary["top_attention_writer_candidates"][:3], "top_mlp": summary["top_mlp_writer_candidates"][:3]}, ensure_ascii=False, indent=2), flush=True)
    return summary


def build_atlas_graph(payload: dict[str, Any], round_name: str) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add_node(node: dict[str, Any]) -> None:
        if node["id"] in seen:
            return
        seen.add(node["id"])
        nodes.append(node)

    models = payload.get("models", [])
    for model_index, model in enumerate(models):
        lane_z = (model_index - (len(models) - 1) / 2) * 8
        summary = payload["by_model"][model]
        model_node = f"{model}:model"
        phase_node = f"{model}:phase:734:{round_name}"
        target_node = f"{model}:target:{summary['target_site']}"
        add_node({"id": model_node, "type": "model", "label": model, "model": model, "position": [-18, 0, lane_z], "role": "tested_model"})
        add_node({"id": phase_node, "type": "phase", "label": f"Phase 734 {round_name}", "model": model, "position": [-12, 2, lane_z], "role": "writer_decomposition"})
        add_node({"id": target_node, "type": "layer", "label": summary["target_site"], "model": model, "role": "downstream_prompt_type_carrier", "evidence_level": "phase733_target_site"})
        edges.append({"source": model_node, "target": phase_node, "relation": "contains", "phase": 734})
        edges.append({"source": phase_node, "target": target_node, "relation": "measures_downstream_site", "phase": 734})
        for comp_key in ["top_attention_writer_candidates", "top_mlp_writer_candidates"]:
            for rec in summary.get(comp_key, [])[:8]:
                node_id = f"{model}:writer:{round_name}:{rec['component_id']}"
                add_node(
                    {
                        "id": node_id,
                        "type": "head" if rec["component_type"] == "attention_head" else "channel_group",
                        "label": rec["component_id"],
                        "model": model,
                        "layer": rec["layer"],
                        "role": rec["role_guess"],
                        "evidence_level": "downstream_skeleton_loss_under_ablation",
                        "score": rec["mean_explicit_skeleton_loss"],
                        "logprob_delta": rec["mean_explicit_logprob_delta"],
                    }
                )
                relation = "writer_candidate_for" if rec["role_guess"] == "writer_candidate" else "perturbs"
                edges.append({"source": node_id, "target": target_node, "relation": relation, "weight": rec["mean_explicit_skeleton_loss"], "phase": 734})
    return {
        "schema_version": "atlas_graph_v1",
        "title": f"Phase 734 Prompt-Type Skeleton Writer Decomposition ({round_name})",
        "model_info": {"model": "cross_model", "models": models, "phase": 734, "round": round_name, "timestamp": payload.get("timestamp"), "evidence_type": payload.get("evidence_type")},
        "layout": {"x": "writer -> downstream carrier", "y": "layer index", "z": "model lane"},
        "graph": {"nodes": nodes, "edges": edges},
        "metrics": {"node_count": len(nodes), "edge_count": len(edges), "source_phase": 734},
        "source_files": [str(OUT_ROOT / round_name / "phase734_cross_model_summary.json")],
    }


def write_cross_summary(round_name: str) -> dict[str, Any]:
    out_dir = OUT_ROOT / round_name
    summaries = []
    for model in MODELS:
        path = out_dir / f"phase734_{model}_summary.json"
        if path.exists():
            summaries.append(json.loads(path.read_text(encoding="utf-8")))
    payload = {
        "phase": 734,
        "title": "Prompt-Type Skeleton Writer Decomposition",
        "round": round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": [s["model"] for s in summaries],
        "status": "complete" if len(summaries) == len(MODELS) else "partial",
        "evidence_type": "attention-head and MLP-output-group ablation measured by downstream prompt-type skeleton loss",
        "by_model": {s["model"]: s for s in summaries},
    }
    write_json(out_dir / "phase734_cross_model_summary.json", payload)
    graph = build_atlas_graph(payload, round_name)
    write_json(out_dir / "phase734_atlas_graph.json", graph)
    lines = [
        f"# Phase 734 Prompt-Type Skeleton Writer Decomposition ({round_name})",
        "",
        f"- Status: `{payload['status']}`",
        f"- Models: `{payload['models']}`",
        "- Evidence type: component ablation -> downstream prompt-type skeleton loss.",
        "",
        "| model | target site | top attention | attn loss | attn logprob | top MLP group | MLP loss | MLP logprob |",
        "|---|---|---|---:|---:|---|---:|---:|",
    ]
    for model, summary in payload["by_model"].items():
        attn = (summary.get("top_attention_writer_candidates") or [{}])[0]
        mlp = (summary.get("top_mlp_writer_candidates") or [{}])[0]
        lines.append(
            f"| {model} | {summary.get('target_site')} | {attn.get('component_id')} | "
            f"{(attn.get('mean_explicit_skeleton_loss') or 0):.3f} | {(attn.get('mean_explicit_logprob_delta') or 0):.3f} | "
            f"{mlp.get('component_id')} | {(mlp.get('mean_explicit_skeleton_loss') or 0):.3f} | {(mlp.get('mean_explicit_logprob_delta') or 0):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Strict Interpretation",
            "",
            "- Positive skeleton loss means ablation moved the explicit path away from the explicit-vs-commonsense downstream direction.",
            "- A component is only a writer candidate when skeleton loss is positive and target likelihood is hurt.",
            "- This is component-level v0, not neuron-level proof.",
            "",
            f"Atlas graph: nodes={graph['metrics']['node_count']} edges={graph['metrics']['edge_count']}",
            "",
        ]
    )
    (out_dir / "phase734_cross_model_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"round": round_name, "status": payload["status"], "models": payload["models"], "atlas": graph["metrics"]}, ensure_ascii=False, indent=2), flush=True)
    return payload


def dry_run(args) -> None:
    pairs = select_prompt_pairs(args.max_pairs)
    payload = {"round": args.round_name, "pairs": len(pairs), "models": {}}
    for model in MODELS:
        window = load_phase733_window(model, args.phase733_round)
        if args.max_layers and len(window["layers"]) > args.max_layers:
            idxs = select_evenly(len(window["layers"]), args.max_layers)
            window["layers"] = [window["layers"][i] for i in idxs]
        payload["models"][model] = window
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="main")
    parser.add_argument("--phase733-round", default="confirm")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-pairs", type=int, default=12)
    parser.add_argument("--max-layers", type=int, default=None)
    parser.add_argument("--max-heads-per-layer", type=int, default=12)
    parser.add_argument("--mlp-groups-per-layer", type=int, default=12)
    parser.add_argument("--log-every", type=int, default=4)
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        dry_run(args)
        return
    if args.summarize_only:
        write_cross_summary(args.round_name)
        return
    if not args.model:
        raise SystemExit("--model is required unless --summarize-only or --dry-run is used")
    run_model(args)
    if args.hard_exit_after_model:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
