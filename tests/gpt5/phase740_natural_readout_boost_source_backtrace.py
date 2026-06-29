#!/usr/bin/env python3
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
from typing import Any, Callable

import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))

from model_utils import get_layers, release_model  # noqa: E402
from phase599_final_layer_washout_decomposition import extract_tensor, get_final_norm  # noqa: E402
from phase722_functional_head_atlas_causal_ablation import logit_diag, target_token_ids, write_json, write_jsonl  # noqa: E402
from phase723_apple_fruit_attribute_micro_atlas import prompt_for  # noqa: E402
from phase735_source_restricted_writer_validation import MODELS, load_model_bf16_eager, norm, safe_mean  # noqa: E402
from phase736_source_replacement_generation_closure import select_conflict_pairs  # noqa: E402
from phase737_writer_rewriter_joint_replacement import intervention_label  # noqa: E402
from phase739_readout_threshold_closure_boundary import (  # noqa: E402
    OUT_ROOT as PHASE739_ROOT,
    choose_donor_recipient,
    get_unembed,
    normalized_direction,
    prepare_joint_install,
    top_token_info,
)
from phase697_answer_last_route_transfer_decomposition import transfer_layers  # noqa: E402


OUT_ROOT = Path("results/glm5_phase740_natural_readout_boost_source_backtrace")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def first_token_id(tokenizer, text: str) -> int:
    ids = target_token_ids(tokenizer, text)
    return int(ids[0])


def get_attn(layer):
    for name in ["self_attn", "attention", "attn"]:
        if hasattr(layer, name):
            return getattr(layer, name)
    return None


def get_mlp(layer):
    for name in ["mlp", "feed_forward", "ffn"]:
        if hasattr(layer, name):
            return getattr(layer, name)
    return None


def projection(vec: torch.Tensor | None, direction: torch.Tensor) -> float | None:
    if vec is None:
        return None
    return float(torch.dot(vec.detach().float().cpu().flatten(), direction.detach().float().cpu().flatten()).item())


def capture_components(
    model,
    tokenizer,
    device,
    ids: list[int],
    scan_layers: list[int],
    install_hooks: Callable[[], list[Any]] | None = None,
) -> dict[str, Any]:
    handles = install_hooks() if install_hooks else []
    captured: dict[str, Any] = {"components": {}, "final_norm_input": None, "final_norm_output": None}
    layers = get_layers(model)
    for li in scan_layers:
        layer = layers[li]
        attn = get_attn(layer)
        mlp = get_mlp(layer)
        if attn is not None:
            def attn_hook(_module, _inputs, output, li=li):
                y = extract_tensor(output)
                captured["components"][f"L{li}:attn_out"] = y[0, -1].detach().float().cpu()

            handles.append(attn.register_forward_hook(attn_hook))
        if mlp is not None:
            def mlp_hook(_module, _inputs, output, li=li):
                y = extract_tensor(output)
                captured["components"][f"L{li}:mlp_out"] = y[0, -1].detach().float().cpu()

            handles.append(mlp.register_forward_hook(mlp_hook))

    final_norm = get_final_norm(model)
    if final_norm is None:
        raise RuntimeError("final norm not found")

    def final_pre_hook(_module, inputs):
        captured["final_norm_input"] = inputs[0][0, -1].detach().float().cpu()

    def final_out_hook(_module, _inputs, output):
        y = extract_tensor(output)
        captured["final_norm_output"] = y[0, -1].detach().float().cpu()

    handles.append(final_norm.register_forward_pre_hook(final_pre_hook))
    handles.append(final_norm.register_forward_hook(final_out_hook))
    try:
        with torch.inference_mode():
            out = model(input_ids=torch.tensor([ids], device=device), return_dict=True, use_cache=False)
        captured["logits"] = out.logits[0, -1].detach().float().cpu()
        return captured
    finally:
        for h in handles:
            h.remove()


def load_phase739_row_index(model_name: str, round_name: str) -> dict[tuple[str, str, str], dict[str, Any]]:
    path = PHASE739_ROOT / round_name / f"phase739_{model_name}_threshold_rows.jsonl"
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {(r["pair_id"], r["intervention_label"], r["direction"]): r for r in rows}


def load_phase739_audits(model_name: str, phase739_round: str, top_audits: int) -> dict[str, Any]:
    from phase737_writer_rewriter_joint_replacement import build_interventions, load_phase735_mlp_specs, load_phase735_source_specs

    source_payload = load_phase735_source_specs(model_name, "confirm", 3, None)
    mlp_specs = load_phase735_mlp_specs(model_name, "confirm", 2)
    all_interventions = {intervention_label(x): x for x in build_interventions(source_payload["paths"], mlp_specs, "compact")}
    summary_path = PHASE739_ROOT / phase739_round / f"phase739_{model_name}_summary.json"
    audits: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        source_payload["target_site"] = summary.get("target_site") or source_payload["target_site"]
        for row in summary.get("top_threshold_audits", []):
            label = row.get("intervention_label")
            direction = row.get("direction")
            if not label or not direction or label not in all_interventions:
                continue
            key = (label, direction)
            if key in seen:
                continue
            audits.append({"intervention": all_interventions[label], "direction": direction, "phase739_summary_row": row})
            seen.add(key)
            if len(audits) >= top_audits:
                break
    if len(audits) < top_audits:
        for intervention in all_interventions.values():
            for direction in ["conflict<-explicit", "explicit<-conflict"]:
                key = (intervention_label(intervention), direction)
                if key in seen:
                    continue
                audits.append({"intervention": intervention, "direction": direction, "phase739_summary_row": None})
                seen.add(key)
                if len(audits) >= top_audits:
                    break
            if len(audits) >= top_audits:
                break
    return {"target_site": source_payload["target_site"], "audits": audits[:top_audits]}


def alpha_needed(logits: torch.Tensor, unembed: torch.Tensor, donor_id: int, top_id: int, direction: torch.Tensor | None) -> float | None:
    if top_id == donor_id:
        return 0.0
    if direction is None:
        return None
    gap = float((logits[top_id] - logits[donor_id]).item())
    if gap <= 0:
        return 0.0
    denom = float(torch.dot(unembed[donor_id] - unembed[top_id], direction).item())
    if denom <= 1e-8:
        return None
    return gap / denom


def audit_pair(
    model,
    tokenizer,
    device,
    model_name: str,
    target_site: str,
    scan_layers: list[int],
    pair: dict[str, Any],
    audit: dict[str, Any],
    phase739_index: dict[tuple[str, str, str], dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    intervention = audit["intervention"]
    direction_name = audit["direction"]
    donor, recipient = choose_donor_recipient(pair, direction_name)
    donor_ids = tokenizer.encode(prompt_for(donor), add_special_tokens=False)
    recipient_ids = tokenizer.encode(prompt_for(recipient), add_special_tokens=False)
    meta, install_joint = prepare_joint_install(model, tokenizer, device, target_site, recipient, donor, recipient_ids, donor_ids, intervention)

    recipient_state = capture_components(model, tokenizer, device, recipient_ids, scan_layers, None)
    donor_state = capture_components(model, tokenizer, device, donor_ids, scan_layers, None)
    patched_state = capture_components(model, tokenizer, device, recipient_ids, scan_layers, install_joint)

    unembed = get_unembed(model)
    donor_id = first_token_id(tokenizer, donor["answer"])
    recipient_id = first_token_id(tokenizer, recipient["answer"])
    patched_logits = patched_state["logits"]
    patched_top = top_token_info(patched_logits, tokenizer)
    top_id = int(patched_top["token_id"])
    d = normalized_direction(unembed, donor_id, top_id)
    if d is None:
        d = torch.zeros_like(unembed[donor_id])
    alpha_star = alpha_needed(patched_logits, unembed, donor_id, top_id, d)
    phase739_row = phase739_index.get((pair["pair_id"], intervention_label(intervention), direction_name), {})
    first_alpha = phase739_row.get("first_alpha_donor_vocab_top")
    threshold = first_alpha if first_alpha is not None else alpha_star

    recipient_final = recipient_state["final_norm_output"]
    donor_final = donor_state["final_norm_output"]
    patched_final = patched_state["final_norm_output"]
    recipient_final_in = recipient_state["final_norm_input"]
    donor_final_in = donor_state["final_norm_input"]
    patched_final_in = patched_state["final_norm_input"]

    donor_final_delta_proj = projection(donor_final - recipient_final, d)
    patched_final_delta_proj = projection(patched_final - recipient_final, d)
    donor_final_input_delta_proj = projection(donor_final_in - recipient_final_in, d)
    patched_final_input_delta_proj = projection(patched_final_in - recipient_final_in, d)

    donor_diag = logit_diag(patched_logits, donor_id)
    recipient_diag = logit_diag(patched_logits, recipient_id)
    component_rows: list[dict[str, Any]] = []
    keys = sorted(set(recipient_state["components"]) | set(donor_state["components"]) | set(patched_state["components"]))
    for key in keys:
        rec = recipient_state["components"].get(key)
        don = donor_state["components"].get(key)
        pat = patched_state["components"].get(key)
        if rec is None or don is None or pat is None:
            continue
        layer = int(key.split(":", 1)[0][1:])
        component = key.split(":", 1)[1]
        donor_delta = projection(don - rec, d)
        patched_delta = projection(pat - rec, d)
        component_rows.append(
            {
                "model": model_name,
                "pair_id": pair["pair_id"],
                "direction": direction_name,
                "intervention_label": intervention_label(intervention),
                "intervention_mode": intervention["mode"],
                "source_component_id": (intervention.get("source_spec") or {}).get("component_id"),
                "source_group": (intervention.get("source_spec") or {}).get("source_group"),
                "mlp_components": [m["component_id"] for m in intervention.get("mlp_specs") or []],
                "component_id": key,
                "layer": layer,
                "component": component,
                "donor_delta_proj": donor_delta,
                "patched_delta_proj": patched_delta,
                "donor_fraction_of_threshold": (donor_delta / threshold) if threshold else None,
                "patched_fraction_of_threshold": (patched_delta / threshold) if threshold else None,
            }
        )

    top_patched_components = sorted(component_rows, key=lambda r: r["patched_delta_proj"], reverse=True)[:8]
    top_donor_components = sorted(component_rows, key=lambda r: r["donor_delta_proj"], reverse=True)[:8]
    case_row = {
        "model": model_name,
        "target_site": target_site,
        "scan_layers": scan_layers,
        "pair_id": pair["pair_id"],
        "direction": direction_name,
        "intervention_mode": intervention["mode"],
        "intervention_label": intervention_label(intervention),
        "source_component_id": (intervention.get("source_spec") or {}).get("component_id"),
        "source_group": (intervention.get("source_spec") or {}).get("source_group"),
        "mlp_components": [m["component_id"] for m in intervention.get("mlp_specs") or []],
        "object": donor["object"],
        "relation": donor["relation"],
        "donor_answer": donor["answer"],
        "recipient_answer": recipient["answer"],
        "donor_token_id": donor_id,
        "recipient_token_id": recipient_id,
        "patched_vocab_top_token_id": top_id,
        "patched_vocab_top_token_text": patched_top["token_text"],
        "patched_margin_donor_vs_vocab_top": float((patched_logits[donor_id] - patched_logits[top_id]).item()),
        "patched_margin_donor_vs_recipient": float((donor_diag["target_logit"] - recipient_diag["target_logit"])),
        "alpha_star_vocab_top": alpha_star,
        "phase739_first_alpha_donor_vocab_top": first_alpha,
        "threshold_used": threshold,
        "donor_final_delta_proj": donor_final_delta_proj,
        "patched_final_delta_proj": patched_final_delta_proj,
        "donor_final_fraction_of_threshold": (donor_final_delta_proj / threshold) if threshold else None,
        "patched_final_fraction_of_threshold": (patched_final_delta_proj / threshold) if threshold else None,
        "donor_final_input_delta_proj": donor_final_input_delta_proj,
        "patched_final_input_delta_proj": patched_final_input_delta_proj,
        "top_patched_components": top_patched_components,
        "top_donor_components": top_donor_components,
        **meta,
    }
    return case_row, component_rows


def summarize_cases(case_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in case_rows:
        groups[row["intervention_label"] + " " + row["direction"]].append(row)
    out = []
    for key, vals in groups.items():
        n = len(vals)
        out.append(
            {
                "intervention_key": key,
                "intervention_label": vals[0]["intervention_label"],
                "direction": vals[0]["direction"],
                "intervention_mode": vals[0]["intervention_mode"],
                "source_component_id": vals[0]["source_component_id"],
                "source_group": vals[0]["source_group"],
                "mlp_components": vals[0]["mlp_components"],
                "n": n,
                "patched_top_counts": dict(Counter(v["patched_vocab_top_token_text"] for v in vals)),
                "mean_threshold_used": safe_mean([v["threshold_used"] for v in vals]),
                "mean_alpha_star_vocab_top": safe_mean([v["alpha_star_vocab_top"] for v in vals]),
                "mean_patched_margin_donor_vs_vocab_top": safe_mean([v["patched_margin_donor_vs_vocab_top"] for v in vals]),
                "mean_donor_final_delta_proj": safe_mean([v["donor_final_delta_proj"] for v in vals]),
                "mean_patched_final_delta_proj": safe_mean([v["patched_final_delta_proj"] for v in vals]),
                "mean_donor_final_fraction_of_threshold": safe_mean([v["donor_final_fraction_of_threshold"] for v in vals]),
                "mean_patched_final_fraction_of_threshold": safe_mean([v["patched_final_fraction_of_threshold"] for v in vals]),
                "mean_donor_final_input_delta_proj": safe_mean([v["donor_final_input_delta_proj"] for v in vals]),
                "mean_patched_final_input_delta_proj": safe_mean([v["patched_final_input_delta_proj"] for v in vals]),
                "mean_source_contribution_delta_norm": safe_mean([v["source_contribution_delta_norm"] for v in vals]),
                "mean_mlp_delta_norm_total": safe_mean([v["mlp_delta_norm_total"] for v in vals]),
            }
        )
    return sorted(out, key=lambda r: r["mean_patched_final_fraction_of_threshold"] or -999, reverse=True)


def summarize_components(component_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in component_rows:
        groups[row["component_id"]].append(row)
    out = []
    for cid, vals in groups.items():
        n = len(vals)
        mean_patched = safe_mean([v["patched_delta_proj"] for v in vals])
        mean_donor = safe_mean([v["donor_delta_proj"] for v in vals])
        out.append(
            {
                "component_id": cid,
                "layer": vals[0]["layer"],
                "component": vals[0]["component"],
                "n": n,
                "mean_patched_delta_proj": mean_patched,
                "mean_donor_delta_proj": mean_donor,
                "mean_patched_fraction_of_threshold": safe_mean([v["patched_fraction_of_threshold"] for v in vals]),
                "mean_donor_fraction_of_threshold": safe_mean([v["donor_fraction_of_threshold"] for v in vals]),
                "patched_positive_rate": sum(1 for v in vals if (v["patched_delta_proj"] or 0) > 0) / n,
                "donor_positive_rate": sum(1 for v in vals if (v["donor_delta_proj"] or 0) > 0) / n,
                "role_guess": (
                    "patched_threshold_alignment_candidate"
                    if (mean_patched or 0) > 0
                    else "donor_context_only_candidate"
                    if (mean_donor or 0) > 0
                    else "opposes_or_irrelevant"
                ),
            }
        )
    return sorted(out, key=lambda r: r["mean_patched_delta_proj"] or -999, reverse=True)


def run_model(args) -> dict[str, Any]:
    out_dir = OUT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    audit_payload = load_phase739_audits(args.model, args.phase739_round, args.top_audits)
    pairs = select_conflict_pairs(args.max_pairs, args.include_extended_relations)
    phase739_index = load_phase739_row_index(args.model, args.phase739_round)
    log(f"{args.model}/{args.round_name}: pairs={len(pairs)} target={audit_payload['target_site']} audits={len(audit_payload['audits'])}")
    model, tokenizer, device, _attn_impl = load_model_bf16_eager(args.model)
    try:
        layers = get_layers(model)
        scan_layers = transfer_layers(args.model, len(layers))
        if args.scan_last_n and args.scan_last_n > 0:
            scan_layers = scan_layers[-args.scan_last_n:]
        case_rows: list[dict[str, Any]] = []
        component_rows: list[dict[str, Any]] = []
        for pair_idx, pair in enumerate(pairs, 1):
            for audit in audit_payload["audits"]:
                case_row, comp_rows = audit_pair(model, tokenizer, device, args.model, audit_payload["target_site"], scan_layers, pair, audit, phase739_index)
                case_rows.append(case_row)
                component_rows.extend(comp_rows)
            if pair_idx % args.log_every == 0 or pair_idx == len(pairs):
                log(f"{args.model}: natural source backtrace {pair_idx}/{len(pairs)} pairs; cases={len(case_rows)} components={len(component_rows)}")
    finally:
        release_model(model)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = {
        "phase": 740,
        "title": "Natural Readout Boost Source Backtrace",
        "model": args.model,
        "round": args.round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": "eager",
        "attention_note": "eager attention is required because Phase739-selected joint states can include source contribution replacement",
        "quantization": "off",
        "dtype": "bfloat16",
        "phase739_round": args.phase739_round,
        "phase738_round": args.phase738_round,
        "target_site": audit_payload["target_site"],
        "scan_layers": scan_layers,
        "top_audits": args.top_audits,
        "max_pairs": args.max_pairs,
        "n_case_rows": len(case_rows),
        "n_component_rows": len(component_rows),
        "audited_interventions": [{"label": intervention_label(a["intervention"]), "direction": a["direction"]} for a in audit_payload["audits"]],
        "case_summary": summarize_cases(case_rows),
        "top_component_sources": summarize_components(component_rows)[:64],
        "strict_interpretation": "Final-state projections are directly comparable to Phase739 alpha thresholds; raw attn/mlp component projections are candidate backtrace signals, not causal proof.",
    }
    write_jsonl(out_dir / f"phase740_{args.model}_case_rows.jsonl", case_rows)
    write_jsonl(out_dir / f"phase740_{args.model}_component_rows.jsonl", component_rows)
    write_json(out_dir / f"phase740_{args.model}_summary.json", summary)
    print(json.dumps({"model": args.model, "round": args.round_name, "target_site": summary["target_site"], "case_summary": summary["case_summary"][:4], "top_component_sources": summary["top_component_sources"][:8]}, ensure_ascii=False, indent=2), flush=True)
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

    for model_index, model in enumerate(payload.get("models", [])):
        lane_z = (model_index - (len(payload.get("models", [])) - 1) / 2) * 10
        summary = payload["by_model"][model]
        threshold_node = f"{model}:phase739_threshold"
        final_node = f"{model}:final_readout_delta"
        add_node({"id": f"{model}:model", "type": "model", "label": model, "model": model, "position": [-34, 0, lane_z], "role": "tested_model"})
        add_node({"id": threshold_node, "type": "threshold", "label": "Phase739 threshold", "model": model, "position": [-18, 0, lane_z], "role": "readout_threshold"})
        add_node({"id": final_node, "type": "final_projection", "label": "natural final projection", "model": model, "position": [0, 0, lane_z], "role": "threshold_fraction"})
        edges.append({"source": f"{model}:model", "target": threshold_node, "relation": "has_threshold", "phase": 740})
        edges.append({"source": final_node, "target": threshold_node, "relation": "compared_to", "phase": 740})
        for rec in summary.get("case_summary", [])[:4]:
            audit_node = f"{model}:audit:{rec['intervention_key']}"
            add_node({"id": audit_node, "type": "audit_summary", "label": rec["intervention_mode"], "model": model, "role": "natural_projection", "mean_patched_final_fraction": rec.get("mean_patched_final_fraction_of_threshold")})
            edges.append({"source": audit_node, "target": final_node, "relation": "contributes_fraction", "weight": rec.get("mean_patched_final_fraction_of_threshold"), "phase": 740})
        for comp in summary.get("top_component_sources", [])[:8]:
            comp_node = f"{model}:component:{comp['component_id']}"
            add_node({"id": comp_node, "type": "component", "label": comp["component_id"], "model": model, "role": comp.get("role_guess"), "mean_patched_delta_proj": comp.get("mean_patched_delta_proj")})
            edges.append({"source": comp_node, "target": final_node, "relation": "candidate_source_projection", "weight": comp.get("mean_patched_delta_proj"), "phase": 740})
    return {
        "schema_version": "atlas_graph_v1",
        "title": f"Phase 740 Natural Readout Boost Source Backtrace ({round_name})",
        "model_info": {"model": "cross_model", "models": payload.get("models", []), "phase": 740, "round": round_name, "timestamp": payload.get("timestamp"), "evidence_type": payload.get("evidence_type")},
        "layout": {"x": "model -> threshold -> final projection -> candidate source components", "y": "projection strength", "z": "model lane"},
        "graph": {"nodes": nodes, "edges": edges},
        "metrics": {"node_count": len(nodes), "edge_count": len(edges), "source_phase": 740},
        "source_files": [str(OUT_ROOT / round_name / "phase740_cross_model_summary.json")],
    }


def write_cross_summary(round_name: str) -> dict[str, Any]:
    out_dir = OUT_ROOT / round_name
    summaries = []
    for model in MODELS:
        path = out_dir / f"phase740_{model}_summary.json"
        if path.exists():
            summaries.append(json.loads(path.read_text(encoding="utf-8")))
    payload = {
        "phase": 740,
        "title": "Natural Readout Boost Source Backtrace",
        "round": round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": [s["model"] for s in summaries],
        "status": "complete" if len(summaries) == len(MODELS) else "partial",
        "evidence_type": "final-state threshold-fraction measurement plus late attn/mlp raw projection backtrace",
        "by_model": {s["model"]: s for s in summaries},
    }
    write_json(out_dir / "phase740_cross_model_summary.json", payload)
    graph = build_atlas_graph(payload, round_name)
    write_json(out_dir / "phase740_atlas_graph.json", graph)
    lines = [
        f"# Phase 740 Natural Readout Boost Source Backtrace ({round_name})",
        "",
        f"- Status: `{payload['status']}`",
        f"- Models: `{payload['models']}`",
        "- Evidence type: final-state threshold fraction and late-component raw projections.",
        "",
        "| model | target site | top audit | threshold | patched final fraction | donor final fraction | top component | component patched fraction |",
        "|---|---|---|---:|---:|---:|---|---:|",
    ]
    for model, summary in payload["by_model"].items():
        rec = (summary.get("case_summary") or [{}])[0]
        comp = (summary.get("top_component_sources") or [{}])[0]
        lines.append(
            f"| {model} | {summary.get('target_site')} | {rec.get('intervention_key')} | "
            f"{(rec.get('mean_threshold_used') or 0):.3f} | "
            f"{(rec.get('mean_patched_final_fraction_of_threshold') or 0):.3f} | "
            f"{(rec.get('mean_donor_final_fraction_of_threshold') or 0):.3f} | "
            f"{comp.get('component_id')} | {(comp.get('mean_patched_fraction_of_threshold') or 0):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Strict Interpretation",
            "",
            "- Final-state projection fractions are comparable to Phase739 alpha thresholds.",
            "- Component projections are pre-final-norm raw signals and should be treated as backtrace candidates, not causal closure proof.",
            "",
            f"Atlas graph: nodes={graph['metrics']['node_count']} edges={graph['metrics']['edge_count']}",
            "",
        ]
    )
    (out_dir / "phase740_cross_model_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"round": round_name, "status": payload["status"], "models": payload["models"], "atlas": graph["metrics"]}, ensure_ascii=False, indent=2), flush=True)
    return payload


def dry_run(args) -> None:
    payload = {"round": args.round_name, "pairs": len(select_conflict_pairs(args.max_pairs, args.include_extended_relations)), "models": {}}
    for model in MODELS:
        audits = load_phase739_audits(model, args.phase739_round, args.top_audits)
        payload["models"][model] = {
            "target_site": audits["target_site"],
            "scan_layers": transfer_layers(model, 128)[-args.scan_last_n:] if args.scan_last_n else transfer_layers(model, 128),
            "audits": [{"label": intervention_label(a["intervention"]), "direction": a["direction"]} for a in audits["audits"]],
        }
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="main")
    parser.add_argument("--phase739-round", default="confirm")
    parser.add_argument("--phase738-round", default="confirm")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-pairs", type=int, default=8)
    parser.add_argument("--top-audits", type=int, default=1)
    parser.add_argument("--scan-last-n", type=int, default=0)
    parser.add_argument("--include-extended-relations", action="store_true")
    parser.add_argument("--log-every", type=int, default=2)
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
