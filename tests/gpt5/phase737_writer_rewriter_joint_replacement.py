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
from typing import Any

import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))

from model_utils import get_layers, release_model  # noqa: E402
from phase722_functional_head_atlas_causal_ablation import write_json, write_jsonl  # noqa: E402
from phase723_apple_fruit_attribute_micro_atlas import prompt_for  # noqa: E402
from phase727_category_fruit_cluster_intervention import hit_answer, norm_text  # noqa: E402
from phase735_source_restricted_writer_validation import (  # noqa: E402
    MODELS,
    dot,
    forward_site_logits,
    load_model_bf16_eager,
    norm,
    parse_component_id,
    safe_mean,
)
from phase736_source_replacement_generation_closure import (  # noqa: E402
    OUT_ROOT as PHASE736_ROOT,
    answer_diag,
    greedy_generate_plain,
    load_phase735_source_specs,
    select_conflict_pairs,
    source_contribution_for_case,
    install_source_contribution_replacement,
)


OUT_ROOT = Path("results/glm5_phase737_writer_rewriter_joint_replacement")
PHASE735_ROOT = Path("results/glm5_phase735_source_restricted_writer_validation")

FALLBACK_MLP_SPECS = {
    "qwen3": [
        {"component_id": "L34:mlp[85:128]", "layer": 34, "start": 85, "end": 128},
        {"component_id": "L28:mlp[299:341]", "layer": 28, "start": 299, "end": 341},
    ],
    "glm4": [
        {"component_id": "L38:mlp[2597:2665]", "layer": 38, "start": 2597, "end": 2665},
        {"component_id": "L38:mlp[3007:3075]", "layer": 38, "start": 3007, "end": 3075},
    ],
    "deepseek7b": [
        {"component_id": "L27:mlp[2872:2932]", "layer": 27, "start": 2872, "end": 2932},
        {"component_id": "L22:mlp[957:1017]", "layer": 22, "start": 957, "end": 1017},
    ],
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_phase735_mlp_specs(model_name: str, round_name: str, top_mlp: int) -> list[dict[str, Any]]:
    fallback = FALLBACK_MLP_SPECS[model_name]
    path = PHASE735_ROOT / round_name / f"phase735_{model_name}_summary.json"
    if not path.exists():
        return fallback[:top_mlp]
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("top_mlp_fine_candidates", [])
    preferred = [r for r in rows if r.get("role_guess") == "fine_mlp_writer_candidate"]
    ordered = preferred + [r for r in rows if r not in preferred]
    specs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in ordered:
        cid = row.get("component_id")
        parsed = parse_component_id(cid or "")
        if not cid or not parsed or cid in seen:
            continue
        specs.append(
            {
                "component_id": cid,
                "layer": int(parsed["layer"]),
                "start": int(parsed["start"]),
                "end": int(parsed["end"]),
                "phase735_mean_explicit_skeleton_loss": row.get("mean_explicit_skeleton_loss"),
                "phase735_mean_explicit_logprob_delta": row.get("mean_explicit_logprob_delta"),
                "phase735_role_guess": row.get("role_guess"),
            }
        )
        seen.add(cid)
        if len(specs) >= top_mlp:
            break
    for rec in fallback:
        if len(specs) >= top_mlp:
            break
        if rec["component_id"] not in seen:
            specs.append(rec)
            seen.add(rec["component_id"])
    return specs[:top_mlp]


def mlp_group_outputs_for_case(model, device, ids: list[int], mlp_specs: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    if not mlp_specs:
        return {}
    by_layer: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for spec in mlp_specs:
        by_layer[int(spec["layer"])].append(spec)
    captures: dict[str, torch.Tensor] = {}
    handles = []
    for layer_idx, specs in by_layer.items():
        module = get_layers(model)[layer_idx].mlp

        def hook(_module, _inputs, output, specs=specs):
            y = output[0] if isinstance(output, tuple) else output
            y_cpu = y[0, -1].detach().float().cpu()
            for spec in specs:
                captures[spec["component_id"]] = y_cpu[int(spec["start"]): int(spec["end"])].clone()

        handles.append(module.register_forward_hook(hook))
    try:
        with torch.inference_mode():
            model(input_ids=torch.tensor([ids], device=device), return_dict=True, use_cache=False)
    finally:
        for h in handles:
            h.remove()
    missing = [s["component_id"] for s in mlp_specs if s["component_id"] not in captures]
    if missing:
        raise RuntimeError(f"missing mlp captures: {missing}")
    return captures


def install_mlp_group_replacements(model, replacements: list[dict[str, Any]]):
    if not replacements:
        return []
    by_layer: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for rep in replacements:
        by_layer[int(rep["layer"])].append(rep)
    handles = []
    for layer_idx, reps in by_layer.items():
        module = get_layers(model)[layer_idx].mlp

        def hook(_module, _inputs, output, reps=reps):
            if isinstance(output, tuple):
                y = output[0].clone()
                rest = output[1:]
            else:
                y = output.clone()
                rest = None
            for rep in reps:
                start = int(rep["start"])
                end = int(rep["end"])
                donor = rep["donor"].to(device=y.device, dtype=y.dtype)
                y[0, -1, start:end] = donor
            if rest is not None:
                return (y,) + rest
            return y

        handles.append(module.register_forward_hook(hook))
    return handles


def patched_joint_site_logits(
    model,
    tokenizer,
    device,
    target_site: str,
    recipient_case: dict[str, Any],
    donor_case: dict[str, Any],
    recipient_ids: list[int],
    donor_ids: list[int],
    source_spec: dict[str, Any] | None,
    mlp_specs: list[dict[str, Any]],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    handles_meta: dict[str, Any] = {
        "donor_attention_mass": None,
        "recipient_attention_mass": None,
        "donor_source_token_count": None,
        "recipient_source_token_count": None,
        "source_contribution_delta_norm": None,
        "mlp_delta_norm_total": 0.0,
        "mlp_components": [s["component_id"] for s in mlp_specs],
    }
    recipient_contrib = donor_contrib = None
    if source_spec is not None:
        layer_idx = int(source_spec["layer"])
        head_idx = int(source_spec["head"])
        source_group = source_spec["source_group"]
        _d_vec, _d_logits, donor_contrib, donor_mass, donor_count = source_contribution_for_case(
            model, tokenizer, device, target_site, donor_case, donor_ids, layer_idx, head_idx, source_group
        )
        _r_vec, _r_logits, recipient_contrib, recipient_mass, recipient_count = source_contribution_for_case(
            model, tokenizer, device, target_site, recipient_case, recipient_ids, layer_idx, head_idx, source_group
        )
        handles_meta.update(
            {
                "donor_attention_mass": donor_mass,
                "recipient_attention_mass": recipient_mass,
                "donor_source_token_count": donor_count,
                "recipient_source_token_count": recipient_count,
                "source_contribution_delta_norm": norm(donor_contrib - recipient_contrib),
            }
        )
    donor_mlp = mlp_group_outputs_for_case(model, device, donor_ids, mlp_specs)
    recipient_mlp = mlp_group_outputs_for_case(model, device, recipient_ids, mlp_specs)
    mlp_replacements = []
    for spec in mlp_specs:
        cid = spec["component_id"]
        delta = donor_mlp[cid] - recipient_mlp[cid]
        handles_meta["mlp_delta_norm_total"] += norm(delta)
        mlp_replacements.append({**spec, "donor": donor_mlp[cid], "recipient": recipient_mlp[cid], "delta_norm": norm(delta)})

    def install():
        handles = []
        if source_spec is not None and donor_contrib is not None and recipient_contrib is not None:
            handles.extend(
                install_source_contribution_replacement(
                    model,
                    int(source_spec["layer"]),
                    int(source_spec["head"]),
                    recipient_contrib,
                    donor_contrib,
                )
            )
        handles.extend(install_mlp_group_replacements(model, mlp_replacements))
        return handles

    patched_vec, patched_logits = forward_site_logits(model, device, recipient_ids, target_site, install)
    return patched_vec, patched_logits, handles_meta


def build_interventions(source_specs: list[dict[str, Any]], mlp_specs: list[dict[str, Any]], mode_set: str) -> list[dict[str, Any]]:
    interventions: list[dict[str, Any]] = []
    for source in source_specs:
        interventions.append({"mode": "source_only", "source_spec": source, "mlp_specs": []})
    for mlp in mlp_specs:
        interventions.append({"mode": "mlp_only", "source_spec": None, "mlp_specs": [mlp]})
    if len(mlp_specs) > 1:
        interventions.append({"mode": "mlp_all", "source_spec": None, "mlp_specs": mlp_specs})
    for source in source_specs:
        if mlp_specs:
            interventions.append({"mode": "source_plus_top_mlp", "source_spec": source, "mlp_specs": [mlp_specs[0]]})
        if mode_set == "full" and len(mlp_specs) > 1:
            for mlp in mlp_specs[1:]:
                interventions.append({"mode": "source_plus_one_mlp", "source_spec": source, "mlp_specs": [mlp]})
        if len(mlp_specs) > 1:
            interventions.append({"mode": "source_plus_all_mlp", "source_spec": source, "mlp_specs": mlp_specs})
    return interventions


def intervention_label(intervention: dict[str, Any]) -> str:
    source = intervention.get("source_spec")
    mlps = intervention.get("mlp_specs") or []
    src = f"{source['component_id']}<-{source['source_group']}" if source else "no_source"
    mlp = "+".join(m["component_id"] for m in mlps) if mlps else "no_mlp"
    return f"{intervention['mode']}|{src}|{mlp}"


def greedy_generate_joint(
    model,
    tokenizer,
    device,
    recipient_case: dict[str, Any],
    donor_case: dict[str, Any],
    target_site: str,
    intervention: dict[str, Any],
    max_new_tokens: int,
) -> dict[str, Any]:
    recipient_ids = tokenizer.encode(prompt_for(recipient_case), add_special_tokens=False)
    donor_ids = tokenizer.encode(prompt_for(donor_case), add_special_tokens=False)
    new_ids: list[int] = []
    for _ in range(max_new_tokens):
        _vec, logits, _meta = patched_joint_site_logits(
            model,
            tokenizer,
            device,
            target_site,
            recipient_case,
            donor_case,
            recipient_ids,
            donor_ids,
            intervention.get("source_spec"),
            intervention.get("mlp_specs") or [],
        )
        tok = int(torch.argmax(logits).item())
        new_ids.append(tok)
        recipient_ids.append(tok)
        donor_ids.append(tok)
        text = tokenizer.decode(new_ids, skip_special_tokens=True)
        if "\n" in text or "." in text or ";" in text:
            break
    return {"text": tokenizer.decode(new_ids, skip_special_tokens=True).strip(), "token_ids": new_ids}


def test_intervention_for_pair(
    model,
    tokenizer,
    device,
    model_name: str,
    target_site: str,
    pair: dict[str, Any],
    intervention: dict[str, Any],
    direction: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    if direction == "conflict<-explicit":
        donor = pair["explicit_profile"]
        recipient = pair["conflict_profile"]
    elif direction == "explicit<-conflict":
        donor = pair["conflict_profile"]
        recipient = pair["explicit_profile"]
    else:
        raise ValueError(direction)
    donor_ids = tokenizer.encode(prompt_for(donor), add_special_tokens=False)
    recipient_prompt = prompt_for(recipient)
    recipient_ids = tokenizer.encode(recipient_prompt, add_special_tokens=False)
    donor_vec, _donor_logits = forward_site_logits(model, device, donor_ids, target_site)
    recipient_vec, recipient_logits = forward_site_logits(model, device, recipient_ids, target_site)
    patched_vec, patched_logits, meta = patched_joint_site_logits(
        model,
        tokenizer,
        device,
        target_site,
        recipient,
        donor,
        recipient_ids,
        donor_ids,
        intervention.get("source_spec"),
        intervention.get("mlp_specs") or [],
    )
    donor_answer = donor["answer"]
    recipient_answer = recipient["answer"]
    base_donor_diag = answer_diag(recipient_logits, tokenizer, donor_answer)
    base_recipient_diag = answer_diag(recipient_logits, tokenizer, recipient_answer)
    patched_donor_diag = answer_diag(patched_logits, tokenizer, donor_answer)
    patched_recipient_diag = answer_diag(patched_logits, tokenizer, recipient_answer)
    d = donor_vec - recipient_vec
    d_norm = norm(d)
    d_hat = d / max(d_norm, 1e-9)
    shift = patched_vec - recipient_vec
    restore_projection = dot(shift, d_hat)
    base_margin = base_donor_diag["target_logit"] - base_recipient_diag["target_logit"]
    patched_margin = patched_donor_diag["target_logit"] - patched_recipient_diag["target_logit"]
    baseline_gen = greedy_generate_plain(model, tokenizer, device, recipient_prompt, max_new_tokens)
    patched_gen = greedy_generate_joint(model, tokenizer, device, recipient, donor, target_site, intervention, max_new_tokens)
    source = intervention.get("source_spec") or {}
    mlps = intervention.get("mlp_specs") or []
    return {
        "model": model_name,
        "target_site": target_site,
        "pair_id": pair["pair_id"],
        "direction": direction,
        "intervention_mode": intervention["mode"],
        "intervention_label": intervention_label(intervention),
        "source_component_id": source.get("component_id"),
        "source_group": source.get("source_group"),
        "source_layer": source.get("layer"),
        "source_head": source.get("head"),
        "mlp_components": [m["component_id"] for m in mlps],
        "mlp_layers": [m["layer"] for m in mlps],
        "object": donor["object"],
        "relation": donor["relation"],
        "donor_prompt_type": donor["prompt_type"],
        "recipient_prompt_type": recipient["prompt_type"],
        "donor_answer": donor_answer,
        "recipient_answer": recipient_answer,
        "target_state_distance": d_norm,
        "patched_target_delta_norm": norm(shift),
        "restore_projection": restore_projection,
        "restore_fraction": restore_projection / max(d_norm, 1e-9),
        "base_donor_logprob": base_donor_diag["target_logprob"],
        "patched_donor_logprob": patched_donor_diag["target_logprob"],
        "donor_logprob_delta": patched_donor_diag["target_logprob"] - base_donor_diag["target_logprob"],
        "base_recipient_logprob": base_recipient_diag["target_logprob"],
        "patched_recipient_logprob": patched_recipient_diag["target_logprob"],
        "recipient_logprob_delta": patched_recipient_diag["target_logprob"] - base_recipient_diag["target_logprob"],
        "base_donor_rank": base_donor_diag["target_rank"],
        "patched_donor_rank": patched_donor_diag["target_rank"],
        "donor_rank_delta": patched_donor_diag["target_rank"] - base_donor_diag["target_rank"],
        "base_recipient_rank": base_recipient_diag["target_rank"],
        "patched_recipient_rank": patched_recipient_diag["target_rank"],
        "recipient_rank_delta": patched_recipient_diag["target_rank"] - base_recipient_diag["target_rank"],
        "base_donor_vs_recipient_logit_margin": base_margin,
        "patched_donor_vs_recipient_logit_margin": patched_margin,
        "donor_vs_recipient_margin_delta": patched_margin - base_margin,
        "baseline_generated_text": baseline_gen["text"],
        "patched_generated_text": patched_gen["text"],
        "baseline_donor_hit": hit_answer(baseline_gen["text"], donor_answer),
        "patched_donor_hit": hit_answer(patched_gen["text"], donor_answer),
        "baseline_recipient_hit": hit_answer(baseline_gen["text"], recipient_answer),
        "patched_recipient_hit": hit_answer(patched_gen["text"], recipient_answer),
        "changed_vs_baseline": norm_text(baseline_gen["text"]) != norm_text(patched_gen["text"]),
        **meta,
    }


def run_joint_tests(
    model,
    tokenizer,
    device,
    model_name: str,
    target_site: str,
    interventions: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
    max_new_tokens: int,
    log_every: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair_idx, pair in enumerate(pairs, 1):
        for intervention in interventions:
            for direction in ["conflict<-explicit", "explicit<-conflict"]:
                rows.append(test_intervention_for_pair(model, tokenizer, device, model_name, target_site, pair, intervention, direction, max_new_tokens))
        if pair_idx % log_every == 0 or pair_idx == len(pairs):
            log(f"{model_name}: joint closure {pair_idx}/{len(pairs)} pairs; rows={len(rows)}")
    return rows


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["intervention_label"], row["direction"])].append(row)
    out = []
    for (label, direction), vals in groups.items():
        n = len(vals)
        donor_hit_gain = sum(1 for v in vals if (not v["baseline_donor_hit"]) and v["patched_donor_hit"]) / n
        recip_hit_loss = sum(1 for v in vals if v["baseline_recipient_hit"] and (not v["patched_recipient_hit"])) / n
        restore = safe_mean([v["restore_projection"] for v in vals]) or 0.0
        donor_gain = safe_mean([v["donor_logprob_delta"] for v in vals]) or 0.0
        margin_gain = safe_mean([v["donor_vs_recipient_margin_delta"] for v in vals]) or 0.0
        if donor_hit_gain > 0:
            role = "generation_closure_candidate"
        elif restore > 0 and donor_gain > 0 and margin_gain > 0:
            role = "joint_readout_transfer_candidate"
        elif restore > 0 and donor_gain > 0:
            role = "joint_state_likelihood_transfer"
        elif restore > 0:
            role = "state_transfer_only"
        elif donor_gain > 0 or margin_gain > 0:
            role = "readout_transfer_only"
        else:
            role = "weak_or_negative"
        first = vals[0]
        out.append(
            {
                "intervention_label": label,
                "intervention_mode": first["intervention_mode"],
                "direction": direction,
                "source_component_id": first["source_component_id"],
                "source_group": first["source_group"],
                "mlp_components": first["mlp_components"],
                "n": n,
                "mean_restore_projection": restore,
                "mean_restore_fraction": safe_mean([v["restore_fraction"] for v in vals]),
                "mean_donor_logprob_delta": donor_gain,
                "mean_recipient_logprob_delta": safe_mean([v["recipient_logprob_delta"] for v in vals]),
                "mean_donor_rank_delta": safe_mean([v["donor_rank_delta"] for v in vals]),
                "mean_donor_vs_recipient_margin_delta": margin_gain,
                "mean_patched_donor_vs_recipient_margin": safe_mean([v["patched_donor_vs_recipient_logit_margin"] for v in vals]),
                "donor_hit_gain_rate": donor_hit_gain,
                "recipient_hit_loss_rate": recip_hit_loss,
                "patched_donor_hit_rate": sum(1 for v in vals if v["patched_donor_hit"]) / n,
                "baseline_donor_hit_rate": sum(1 for v in vals if v["baseline_donor_hit"]) / n,
                "patched_recipient_hit_rate": sum(1 for v in vals if v["patched_recipient_hit"]) / n,
                "baseline_recipient_hit_rate": sum(1 for v in vals if v["baseline_recipient_hit"]) / n,
                "changed_rate": sum(1 for v in vals if v["changed_vs_baseline"]) / n,
                "mean_source_contribution_delta_norm": safe_mean([v["source_contribution_delta_norm"] for v in vals]),
                "mean_mlp_delta_norm_total": safe_mean([v["mlp_delta_norm_total"] for v in vals]),
                "role_guess": role,
            }
        )
    return sorted(
        out,
        key=lambda r: (
            r["donor_hit_gain_rate"],
            r["changed_rate"],
            r["mean_donor_vs_recipient_margin_delta"] or 0,
            r["mean_restore_projection"],
            r["mean_donor_logprob_delta"],
        ),
        reverse=True,
    )


def build_summary(
    model_name: str,
    round_name: str,
    source_payload: dict[str, Any],
    mlp_specs: list[dict[str, Any]],
    interventions: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    intervention_summary = summarize_rows(rows)
    return {
        "phase": 737,
        "title": "Writer-Rewriter Joint Replacement and Generation Closure",
        "model": model_name,
        "round": round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": "eager",
        "attention_note": "eager attention is required for source contribution replacement; flash attention cannot expose the needed weights here",
        "quantization": "off",
        "dtype": "bfloat16",
        "phase735_round": args.phase735_round,
        "target_site": source_payload["target_site"],
        "source_specs": source_payload["paths"],
        "mlp_specs": mlp_specs,
        "intervention_count": len(interventions),
        "max_pairs": args.max_pairs,
        "max_new_tokens": args.max_new_tokens,
        "n_rows": len(rows),
        "top_joint_interventions": intervention_summary[:32],
        "role_counts": dict((r, sum(1 for x in intervention_summary if x["role_guess"] == r)) for r in sorted({x["role_guess"] for x in intervention_summary})),
        "strict_interpretation": "joint replacement tests whether source writer plus MLP rewriter pieces improve hidden/readout/generation closure; MLP groups are output-channel groups, not single-neuron proof",
    }


def run_model(args) -> dict[str, Any]:
    out_dir = OUT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    preferred = set(args.preferred_sources.split(",")) if args.preferred_sources else None
    source_payload = load_phase735_source_specs(args.model, args.phase735_round, args.top_paths, preferred)
    mlp_specs = load_phase735_mlp_specs(args.model, args.phase735_round, args.top_mlp)
    interventions = build_interventions(source_payload["paths"], mlp_specs, args.mode_set)
    pairs = select_conflict_pairs(args.max_pairs, args.include_extended_relations)
    log(
        f"{args.model}/{args.round_name}: pairs={len(pairs)} target={source_payload['target_site']} "
        f"sources={len(source_payload['paths'])} mlp={len(mlp_specs)} interventions={len(interventions)}"
    )
    model, tokenizer, device, _attn_impl = load_model_bf16_eager(args.model)
    try:
        rows = run_joint_tests(
            model,
            tokenizer,
            device,
            args.model,
            source_payload["target_site"],
            interventions,
            pairs,
            args.max_new_tokens,
            args.log_every,
        )
    finally:
        release_model(model)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    summary = build_summary(args.model, args.round_name, source_payload, mlp_specs, interventions, rows, args)
    write_jsonl(out_dir / f"phase737_{args.model}_joint_rows.jsonl", rows)
    write_json(out_dir / f"phase737_{args.model}_summary.json", summary)
    print(
        json.dumps(
            {
                "model": args.model,
                "round": args.round_name,
                "target_site": source_payload["target_site"],
                "top_joint_interventions": summary["top_joint_interventions"][:5],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
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
        lane_z = (model_index - (len(models) - 1) / 2) * 10
        summary = payload["by_model"][model]
        phase_node = f"{model}:phase:737:{round_name}"
        target_node = f"{model}:target:{summary['target_site']}"
        add_node({"id": f"{model}:model", "type": "model", "label": model, "model": model, "position": [-34, 0, lane_z], "role": "tested_model"})
        add_node({"id": phase_node, "type": "phase", "label": f"Phase 737 {round_name}", "model": model, "position": [-28, 2, lane_z], "role": "writer_rewriter_joint_replacement"})
        add_node({"id": target_node, "type": "layer", "label": summary["target_site"], "model": model, "position": [16, 0, lane_z], "role": "target_carrier_readout"})
        edges.append({"source": f"{model}:model", "target": phase_node, "relation": "contains", "phase": 737})
        edges.append({"source": phase_node, "target": target_node, "relation": "measures_joint_closure_at", "phase": 737})
        for rec in summary.get("top_joint_interventions", [])[:16]:
            int_node = f"{model}:joint:{round_name}:{rec['intervention_label']}:{rec['direction']}"
            add_node(
                {
                    "id": int_node,
                    "type": "intervention",
                    "label": rec["intervention_mode"],
                    "model": model,
                    "role": rec["role_guess"],
                    "score": rec["mean_restore_projection"],
                    "logprob_delta": rec["mean_donor_logprob_delta"],
                    "margin_delta": rec["mean_donor_vs_recipient_margin_delta"],
                    "generation_hit_gain": rec["donor_hit_gain_rate"],
                }
            )
            if rec.get("source_component_id"):
                src_node = f"{model}:source_writer:{rec['source_component_id']}:{rec['source_group']}"
                add_node({"id": src_node, "type": "head", "label": f"{rec['source_component_id']}<-{rec['source_group']}", "model": model, "role": "source_writer"})
                edges.append({"source": src_node, "target": int_node, "relation": "source_writer_participates", "weight": rec["mean_restore_projection"], "phase": 737})
            for cid in rec.get("mlp_components") or []:
                mlp_node = f"{model}:mlp_rewriter:{cid}"
                add_node({"id": mlp_node, "type": "mlp_group", "label": cid, "model": model, "role": "mlp_rewriter_candidate"})
                edges.append({"source": mlp_node, "target": int_node, "relation": "mlp_rewriter_participates", "weight": rec["mean_mlp_delta_norm_total"], "phase": 737})
            edges.append({"source": int_node, "target": target_node, "relation": "moves_readout_or_generation", "weight": rec["mean_donor_vs_recipient_margin_delta"], "phase": 737})
    return {
        "schema_version": "atlas_graph_v1",
        "title": f"Phase 737 Writer-Rewriter Joint Replacement ({round_name})",
        "model_info": {"model": "cross_model", "models": models, "phase": 737, "round": round_name, "timestamp": payload.get("timestamp"), "evidence_type": payload.get("evidence_type")},
        "layout": {"x": "source writer + MLP rewriter -> target/readout", "y": "layer index", "z": "model lane"},
        "graph": {"nodes": nodes, "edges": edges},
        "metrics": {"node_count": len(nodes), "edge_count": len(edges), "source_phase": 737},
        "source_files": [str(OUT_ROOT / round_name / "phase737_cross_model_summary.json")],
    }


def write_cross_summary(round_name: str) -> dict[str, Any]:
    out_dir = OUT_ROOT / round_name
    summaries = []
    for model in MODELS:
        path = out_dir / f"phase737_{model}_summary.json"
        if path.exists():
            summaries.append(json.loads(path.read_text(encoding="utf-8")))
    payload = {
        "phase": 737,
        "title": "Writer-Rewriter Joint Replacement and Generation Closure",
        "round": round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": [s["model"] for s in summaries],
        "status": "complete" if len(summaries) == len(MODELS) else "partial",
        "evidence_type": "source writer plus MLP rewriter donor-recipient replacement with readout margin and greedy generation checks",
        "by_model": {s["model"]: s for s in summaries},
    }
    write_json(out_dir / "phase737_cross_model_summary.json", payload)
    graph = build_atlas_graph(payload, round_name)
    write_json(out_dir / "phase737_atlas_graph.json", graph)
    lines = [
        f"# Phase 737 Writer-Rewriter Joint Replacement ({round_name})",
        "",
        f"- Status: `{payload['status']}`",
        f"- Models: `{payload['models']}`",
        "- Evidence type: joint replacement of source writer and MLP rewriter candidates.",
        "",
        "| model | target site | top intervention | restore | donor logprob | margin delta | donor hit gain | changed | role |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for model, summary in payload["by_model"].items():
        rec = (summary.get("top_joint_interventions") or [{}])[0]
        lines.append(
            f"| {model} | {summary.get('target_site')} | {rec.get('intervention_label')} {rec.get('direction')} | "
            f"{(rec.get('mean_restore_projection') or 0):.3f} | {(rec.get('mean_donor_logprob_delta') or 0):.3f} | "
            f"{(rec.get('mean_donor_vs_recipient_margin_delta') or 0):.3f} | {(rec.get('donor_hit_gain_rate') or 0):.3f} | "
            f"{(rec.get('changed_rate') or 0):.3f} | {rec.get('role_guess')} |"
        )
    lines.extend(
        [
            "",
            "## Strict Interpretation",
            "",
            "- A positive restore projection means the target hidden state moved toward the donor state.",
            "- A positive margin delta means donor answer readout improved against the recipient answer, not just in isolation.",
            "- Generation hit gain remains the strictest closure criterion.",
            "",
            f"Atlas graph: nodes={graph['metrics']['node_count']} edges={graph['metrics']['edge_count']}",
            "",
        ]
    )
    (out_dir / "phase737_cross_model_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"round": round_name, "status": payload["status"], "models": payload["models"], "atlas": graph["metrics"]}, ensure_ascii=False, indent=2), flush=True)
    return payload


def dry_run(args) -> None:
    preferred = set(args.preferred_sources.split(",")) if args.preferred_sources else None
    payload: dict[str, Any] = {"round": args.round_name, "pairs": len(select_conflict_pairs(args.max_pairs, args.include_extended_relations)), "models": {}}
    for model in MODELS:
        sources = load_phase735_source_specs(model, args.phase735_round, args.top_paths, preferred)
        mlps = load_phase735_mlp_specs(model, args.phase735_round, args.top_mlp)
        payload["models"][model] = {
            "target_site": sources["target_site"],
            "sources": sources["paths"],
            "mlp_specs": mlps,
            "interventions": [intervention_label(i) for i in build_interventions(sources["paths"], mlps, args.mode_set)],
        }
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="main")
    parser.add_argument("--phase735-round", default="confirm")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-pairs", type=int, default=4)
    parser.add_argument("--top-paths", type=int, default=3)
    parser.add_argument("--top-mlp", type=int, default=2)
    parser.add_argument("--preferred-sources", default="")
    parser.add_argument("--include-extended-relations", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=3)
    parser.add_argument("--mode-set", choices=["compact", "full"], default="compact")
    parser.add_argument("--log-every", type=int, default=1)
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
