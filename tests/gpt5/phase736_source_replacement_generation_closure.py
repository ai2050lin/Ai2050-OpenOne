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

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))

from model_utils import get_layers, release_model  # noqa: E402
from phase112_attention_transport_head_mapping_cuda import get_attention_module  # noqa: E402
from phase132_source_value_contribution_cuda import compute_source_contribution, get_num_kv_heads  # noqa: E402
from phase693_boundary_attention_head_candidate_audit import head_meta  # noqa: E402
from phase722_functional_head_atlas_causal_ablation import logit_diag, target_token_ids, write_json, write_jsonl  # noqa: E402
from phase723_apple_fruit_attribute_micro_atlas import build_cases, prompt_for  # noqa: E402
from phase727_category_fruit_cluster_intervention import hit_answer, norm_text  # noqa: E402
from phase735_source_restricted_writer_validation import (  # noqa: E402
    MODELS,
    build_source_groups,
    dot,
    first_token_diag,
    forward_base_with_attention,
    forward_site_logits,
    load_model_bf16_eager,
    norm,
    safe_mean,
    select_evenly,
)


OUT_ROOT = Path("results/glm5_phase736_source_replacement_generation_closure")
PHASE735_ROOT = Path("results/glm5_phase735_source_restricted_writer_validation")
DEFAULT_RELATIONS = {"category", "color", "taste"}

FALLBACK_SOURCE_SPECS = {
    "qwen3": {
        "target_site": "hidden_36",
        "paths": [
            {"component_id": "L35H0", "layer": 35, "head": 0, "source_group": "self_last"},
            {"component_id": "L28H28", "layer": 28, "head": 28, "source_group": "instruction"},
        ],
    },
    "glm4": {
        "target_site": "hidden_40",
        "paths": [
            {"component_id": "L39H21", "layer": 39, "head": 21, "source_group": "self_last"},
            {"component_id": "L23H17", "layer": 23, "head": 17, "source_group": "instruction"},
        ],
    },
    "deepseek7b": {
        "target_site": "hidden_28",
        "paths": [
            {"component_id": "L22H24", "layer": 22, "head": 24, "source_group": "records_all"},
            {"component_id": "L22H24", "layer": 22, "head": 24, "source_group": "target_record_line"},
            {"component_id": "L22H24", "layer": 22, "head": 24, "source_group": "target_value_tokens"},
        ],
    },
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_phase735_source_specs(model_name: str, round_name: str, top_paths: int, preferred_sources: set[str] | None = None) -> dict[str, Any]:
    fallback = FALLBACK_SOURCE_SPECS[model_name]
    path = PHASE735_ROOT / round_name / f"phase735_{model_name}_summary.json"
    if not path.exists():
        return {"target_site": fallback["target_site"], "paths": fallback["paths"][:top_paths]}
    data = json.loads(path.read_text(encoding="utf-8"))
    chosen: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    rows = data.get("top_attention_source_paths", [])
    if preferred_sources:
        rows = [r for r in rows if r.get("source_group") in preferred_sources] + [r for r in rows if r.get("source_group") not in preferred_sources]
    for row in rows:
        if row.get("role_guess") not in {"source_restricted_writer_path", "source_state_contributor"}:
            continue
        cid = row.get("component_id")
        source = row.get("source_group")
        if not cid or not source:
            continue
        key = (cid, source)
        if key in seen:
            continue
        rec = {
            "component_id": cid,
            "layer": int(row["layer"]),
            "head": int(row["head"]),
            "source_group": source,
            "phase735_mean_explicit_skeleton_loss": row.get("mean_explicit_skeleton_loss"),
            "phase735_mean_explicit_logprob_delta": row.get("mean_explicit_logprob_delta"),
            "phase735_mean_attention_mass": row.get("mean_attention_mass"),
            "phase735_role_guess": row.get("role_guess"),
        }
        chosen.append(rec)
        seen.add(key)
        if len(chosen) >= top_paths:
            break
    if len(chosen) < top_paths:
        for row in fallback["paths"]:
            key = (row["component_id"], row["source_group"])
            if key not in seen:
                chosen.append(row)
                seen.add(key)
            if len(chosen) >= top_paths:
                break
    return {"target_site": data.get("target_site") or fallback["target_site"], "paths": chosen[:top_paths]}


def select_conflict_pairs(max_pairs: int | None, include_extended_relations: bool = False) -> list[dict[str, Any]]:
    allowed = None if include_extended_relations else DEFAULT_RELATIONS
    cases = build_cases(None)
    explicit = {
        (c["object"], c["relation"]): c
        for c in cases
        if c["prompt_type"] == "explicit_profile" and (allowed is None or c["relation"] in allowed)
    }
    pairs = []
    for conflict in cases:
        if conflict["prompt_type"] != "conflict_profile":
            continue
        if allowed is not None and conflict["relation"] not in allowed:
            continue
        key = (conflict["object"], conflict["relation"])
        donor = explicit.get(key)
        if not donor or donor["answer"] == conflict["answer"]:
            continue
        pairs.append(
            {
                "pair_id": f"{conflict['object']}:{conflict['relation']}",
                "explicit_profile": donor,
                "conflict_profile": conflict,
            }
        )
    if not max_pairs or max_pairs >= len(pairs):
        return pairs
    return [pairs[i] for i in select_evenly(len(pairs), max_pairs)]


def install_source_contribution_replacement(model, layer_idx: int, head_idx: int, recipient_contribution: torch.Tensor, donor_contribution: torch.Tensor):
    o_proj, n_heads, head_dim = head_meta(model, layer_idx)
    recipient = recipient_contribution.detach().float().cpu()
    donor = donor_contribution.detach().float().cpu()

    def pre_hook(_module, inputs):
        x = inputs[0]
        y = x.clone()
        yv = y.view(y.shape[0], y.shape[1], n_heads, head_dim)
        yv[0, -1, head_idx, :] = yv[0, -1, head_idx, :] - recipient.to(y.device, y.dtype) + donor.to(y.device, y.dtype)
        return (y,) + tuple(inputs[1:])

    return [o_proj.register_forward_pre_hook(pre_hook)]


def source_contribution_for_case(
    model,
    tokenizer,
    device,
    target_site: str,
    case: dict[str, Any],
    ids: list[int],
    layer_idx: int,
    head_idx: int,
    source_group: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, int]:
    prompt = prompt_for(case)
    vec, logits, attn_store, value_store = forward_base_with_attention(model, device, ids, target_site, [layer_idx])
    source_groups = build_source_groups(tokenizer, prompt, case, ids)
    source_positions = source_groups.get(source_group, [])
    if not source_positions:
        raise RuntimeError(f"empty source group {source_group} for {case['case_id']}")
    attn = get_attention_module(get_layers(model)[layer_idx])
    _o_proj, n_heads, _head_dim = head_meta(model, layer_idx)
    n_kv_heads = get_num_kv_heads(model, attn, n_heads)
    contrib = compute_source_contribution(
        attn_store[layer_idx],
        value_store[layer_idx],
        [len(ids) - 1],
        [source_positions],
        n_heads,
        n_kv_heads,
    )[0, head_idx]
    mass = float(np.asarray(attn_store[layer_idx][0, head_idx, len(ids) - 1, source_positions], dtype=np.float32).sum())
    return vec, logits, contrib, mass, len(source_positions)


def patched_site_logits(
    model,
    tokenizer,
    device,
    target_site: str,
    recipient_case: dict[str, Any],
    donor_case: dict[str, Any],
    recipient_ids: list[int],
    donor_ids: list[int],
    spec: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    layer_idx = int(spec["layer"])
    head_idx = int(spec["head"])
    source_group = spec["source_group"]
    _d_vec, _d_logits, donor_contrib, donor_mass, donor_count = source_contribution_for_case(
        model, tokenizer, device, target_site, donor_case, donor_ids, layer_idx, head_idx, source_group
    )
    _r_vec, _r_logits, recipient_contrib, recipient_mass, recipient_count = source_contribution_for_case(
        model, tokenizer, device, target_site, recipient_case, recipient_ids, layer_idx, head_idx, source_group
    )

    def install():
        return install_source_contribution_replacement(model, layer_idx, head_idx, recipient_contrib, donor_contrib)

    patched_vec, patched_logits = forward_site_logits(model, device, recipient_ids, target_site, install)
    return patched_vec, patched_logits, {
        "donor_attention_mass": donor_mass,
        "recipient_attention_mass": recipient_mass,
        "donor_source_token_count": donor_count,
        "recipient_source_token_count": recipient_count,
        "donor_contribution_norm": norm(donor_contrib),
        "recipient_contribution_norm": norm(recipient_contrib),
        "contribution_delta_norm": norm(donor_contrib - recipient_contrib),
    }


def greedy_generate_plain(model, tokenizer, device, prompt: str, max_new_tokens: int) -> dict[str, Any]:
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    new_ids: list[int] = []
    for _ in range(max_new_tokens):
        with torch.inference_mode():
            out = model(input_ids=torch.tensor([ids], device=device), return_dict=True, use_cache=False)
        tok = int(torch.argmax(out.logits[0, -1]).item())
        new_ids.append(tok)
        ids.append(tok)
        text = tokenizer.decode(new_ids, skip_special_tokens=True)
        if "\n" in text or "." in text or ";" in text:
            break
    return {"text": tokenizer.decode(new_ids, skip_special_tokens=True).strip(), "token_ids": new_ids}


def greedy_generate_replacement(
    model,
    tokenizer,
    device,
    recipient_case: dict[str, Any],
    donor_case: dict[str, Any],
    spec: dict[str, Any],
    target_site: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    recipient_ids = tokenizer.encode(prompt_for(recipient_case), add_special_tokens=False)
    donor_ids = tokenizer.encode(prompt_for(donor_case), add_special_tokens=False)
    new_ids: list[int] = []
    for _ in range(max_new_tokens):
        _vec, logits, _meta = patched_site_logits(
            model,
            tokenizer,
            device,
            target_site,
            recipient_case,
            donor_case,
            recipient_ids,
            donor_ids,
            spec,
        )
        tok = int(torch.argmax(logits).item())
        new_ids.append(tok)
        recipient_ids.append(tok)
        donor_ids.append(tok)
        text = tokenizer.decode(new_ids, skip_special_tokens=True)
        if "\n" in text or "." in text or ";" in text:
            break
    return {"text": tokenizer.decode(new_ids, skip_special_tokens=True).strip(), "token_ids": new_ids}


def answer_diag(logits: torch.Tensor, tokenizer, answer: str) -> dict[str, Any]:
    tid = target_token_ids(tokenizer, answer)[0]
    return logit_diag(logits, int(tid))


def direction_tests_for_pair(
    model,
    tokenizer,
    device,
    model_name: str,
    target_site: str,
    pair: dict[str, Any],
    spec: dict[str, Any],
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
    donor_prompt = prompt_for(donor)
    recipient_prompt = prompt_for(recipient)
    donor_ids = tokenizer.encode(donor_prompt, add_special_tokens=False)
    recipient_ids = tokenizer.encode(recipient_prompt, add_special_tokens=False)
    donor_vec, donor_logits = forward_site_logits(model, device, donor_ids, target_site)
    recipient_vec, recipient_logits = forward_site_logits(model, device, recipient_ids, target_site)
    donor_answer = donor["answer"]
    recipient_answer = recipient["answer"]
    base_donor_diag = answer_diag(recipient_logits, tokenizer, donor_answer)
    base_recipient_diag = answer_diag(recipient_logits, tokenizer, recipient_answer)
    patched_vec, patched_logits, contrib_meta = patched_site_logits(
        model,
        tokenizer,
        device,
        target_site,
        recipient,
        donor,
        recipient_ids,
        donor_ids,
        spec,
    )
    patched_donor_diag = answer_diag(patched_logits, tokenizer, donor_answer)
    patched_recipient_diag = answer_diag(patched_logits, tokenizer, recipient_answer)
    d = donor_vec - recipient_vec
    d_norm = norm(d)
    d_hat = d / max(d_norm, 1e-9)
    shift = patched_vec - recipient_vec
    restore_projection = dot(shift, d_hat)
    baseline_gen = greedy_generate_plain(model, tokenizer, device, recipient_prompt, max_new_tokens)
    patched_gen = greedy_generate_replacement(model, tokenizer, device, recipient, donor, spec, target_site, max_new_tokens)
    return {
        "model": model_name,
        "component_id": spec["component_id"],
        "layer": int(spec["layer"]),
        "head": int(spec["head"]),
        "source_group": spec["source_group"],
        "target_site": target_site,
        "pair_id": pair["pair_id"],
        "direction": direction,
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
        "baseline_generated_text": baseline_gen["text"],
        "patched_generated_text": patched_gen["text"],
        "baseline_donor_hit": hit_answer(baseline_gen["text"], donor_answer),
        "patched_donor_hit": hit_answer(patched_gen["text"], donor_answer),
        "baseline_recipient_hit": hit_answer(baseline_gen["text"], recipient_answer),
        "patched_recipient_hit": hit_answer(patched_gen["text"], recipient_answer),
        "changed_vs_baseline": norm_text(baseline_gen["text"]) != norm_text(patched_gen["text"]),
        **contrib_meta,
    }


def run_replacement_tests(
    model,
    tokenizer,
    device,
    model_name: str,
    target_site: str,
    source_specs: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
    max_new_tokens: int,
    log_every: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair_idx, pair in enumerate(pairs, 1):
        for spec in source_specs:
            for direction in ["conflict<-explicit", "explicit<-conflict"]:
                rows.append(direction_tests_for_pair(model, tokenizer, device, model_name, target_site, pair, spec, direction, max_new_tokens))
        if pair_idx % log_every == 0 or pair_idx == len(pairs):
            log(f"{model_name}: replacement closure {pair_idx}/{len(pairs)} pairs; rows={len(rows)}")
    return rows


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["component_id"], row["source_group"], row["direction"])].append(row)
    out = []
    for (cid, source, direction), vals in groups.items():
        n = len(vals)
        donor_gain = safe_mean([v["donor_logprob_delta"] for v in vals]) or 0.0
        recip_delta = safe_mean([v["recipient_logprob_delta"] for v in vals]) or 0.0
        restore = safe_mean([v["restore_projection"] for v in vals]) or 0.0
        donor_hit_gain = sum(1 for v in vals if (not v["baseline_donor_hit"]) and v["patched_donor_hit"]) / n
        recip_hit_loss = sum(1 for v in vals if v["baseline_recipient_hit"] and (not v["patched_recipient_hit"])) / n
        if restore > 0 and donor_gain > 0:
            role = "content_transfer_candidate"
        elif restore > 0:
            role = "state_transfer_only"
        elif donor_gain > 0:
            role = "likelihood_transfer_only"
        else:
            role = "weak_or_negative"
        out.append(
            {
                "component_id": cid,
                "layer": vals[0]["layer"],
                "head": vals[0]["head"],
                "source_group": source,
                "direction": direction,
                "n": n,
                "mean_restore_projection": restore,
                "mean_restore_fraction": safe_mean([v["restore_fraction"] for v in vals]),
                "mean_donor_logprob_delta": donor_gain,
                "mean_recipient_logprob_delta": recip_delta,
                "mean_donor_rank_delta": safe_mean([v["donor_rank_delta"] for v in vals]),
                "mean_recipient_rank_delta": safe_mean([v["recipient_rank_delta"] for v in vals]),
                "donor_hit_gain_rate": donor_hit_gain,
                "recipient_hit_loss_rate": recip_hit_loss,
                "patched_donor_hit_rate": sum(1 for v in vals if v["patched_donor_hit"]) / n,
                "baseline_donor_hit_rate": sum(1 for v in vals if v["baseline_donor_hit"]) / n,
                "patched_recipient_hit_rate": sum(1 for v in vals if v["patched_recipient_hit"]) / n,
                "baseline_recipient_hit_rate": sum(1 for v in vals if v["baseline_recipient_hit"]) / n,
                "changed_rate": sum(1 for v in vals if v["changed_vs_baseline"]) / n,
                "mean_donor_attention_mass": safe_mean([v["donor_attention_mass"] for v in vals]),
                "mean_recipient_attention_mass": safe_mean([v["recipient_attention_mass"] for v in vals]),
                "mean_contribution_delta_norm": safe_mean([v["contribution_delta_norm"] for v in vals]),
                "role_guess": role,
            }
        )
    return sorted(out, key=lambda r: (r["mean_restore_projection"], r["mean_donor_logprob_delta"]), reverse=True)


def build_summary(model_name: str, round_name: str, attn_impl: str, source_payload: dict[str, Any], rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    path_summary = summarize_rows(rows)
    return {
        "phase": 736,
        "title": "Source-Restricted Replacement and Generation Closure",
        "model": model_name,
        "round": round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "attention_note": "eager attention is required for source-restricted contribution replacement",
        "quantization": "off",
        "dtype": "bfloat16",
        "phase735_round": args.phase735_round,
        "target_site": source_payload["target_site"],
        "source_specs": source_payload["paths"],
        "max_pairs": args.max_pairs,
        "max_new_tokens": args.max_new_tokens,
        "n_rows": len(rows),
        "top_replacement_paths": path_summary[:24],
        "role_counts": dict((r, sum(1 for x in path_summary if x["role_guess"] == r)) for r in sorted({x["role_guess"] for x in path_summary})),
        "strict_interpretation": "replacement tests content transfer at source-contribution level; generation hit remains a strict and often sparse closure metric",
    }


def run_model(args) -> dict[str, Any]:
    out_dir = OUT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    preferred = set(args.preferred_sources.split(",")) if args.preferred_sources else None
    source_payload = load_phase735_source_specs(args.model, args.phase735_round, args.top_paths, preferred)
    pairs = select_conflict_pairs(args.max_pairs, args.include_extended_relations)
    log(f"{args.model}/{args.round_name}: pairs={len(pairs)} target_site={source_payload['target_site']} source_paths={len(source_payload['paths'])}")
    model, tokenizer, device, attn_impl = load_model_bf16_eager(args.model)
    try:
        rows = run_replacement_tests(
            model,
            tokenizer,
            device,
            args.model,
            source_payload["target_site"],
            source_payload["paths"],
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
    summary = build_summary(args.model, args.round_name, attn_impl, source_payload, rows, args)
    write_jsonl(out_dir / f"phase736_{args.model}_replacement_rows.jsonl", rows)
    write_json(out_dir / f"phase736_{args.model}_summary.json", summary)
    print(json.dumps({"model": args.model, "round": args.round_name, "target_site": source_payload["target_site"], "top_replacement_paths": summary["top_replacement_paths"][:5]}, ensure_ascii=False, indent=2), flush=True)
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
        phase_node = f"{model}:phase:736:{round_name}"
        target_node = f"{model}:target:{summary['target_site']}"
        add_node({"id": model_node, "type": "model", "label": model, "model": model, "position": [-28, 0, lane_z], "role": "tested_model"})
        add_node({"id": phase_node, "type": "phase", "label": f"Phase 736 {round_name}", "model": model, "position": [-22, 2, lane_z], "role": "source_replacement_generation_closure"})
        add_node({"id": target_node, "type": "layer", "label": summary["target_site"], "model": model, "role": "downstream_prompt_type_carrier"})
        edges.append({"source": model_node, "target": phase_node, "relation": "contains", "phase": 736})
        edges.append({"source": phase_node, "target": target_node, "relation": "measures_replacement_at", "phase": 736})
        for rec in summary.get("top_replacement_paths", [])[:12]:
            source_node = f"{model}:replacement_source:{round_name}:{rec['component_id']}:{rec['source_group']}:{rec['direction']}"
            head_node = f"{model}:writer:{round_name}:{rec['component_id']}"
            add_node({"id": head_node, "type": "head", "label": rec["component_id"], "model": model, "layer": rec["layer"], "head": rec["head"], "role": "source_replacement_head"})
            add_node({"id": source_node, "type": "token_group", "label": f"{rec['source_group']} {rec['direction']}", "model": model, "role": rec["role_guess"], "score": rec["mean_restore_projection"], "logprob_delta": rec["mean_donor_logprob_delta"]})
            edges.append({"source": source_node, "target": head_node, "relation": "donor_source_replaces_recipient_source", "weight": rec["mean_restore_projection"], "phase": 736})
            edges.append({"source": head_node, "target": target_node, "relation": "moves_recipient_toward_donor", "weight": rec["mean_restore_projection"], "phase": 736})
    return {
        "schema_version": "atlas_graph_v1",
        "title": f"Phase 736 Source-Restricted Replacement and Generation Closure ({round_name})",
        "model_info": {"model": "cross_model", "models": models, "phase": 736, "round": round_name, "timestamp": payload.get("timestamp"), "evidence_type": payload.get("evidence_type")},
        "layout": {"x": "donor source -> writer -> recipient carrier/readout", "y": "layer index", "z": "model lane"},
        "graph": {"nodes": nodes, "edges": edges},
        "metrics": {"node_count": len(nodes), "edge_count": len(edges), "source_phase": 736},
        "source_files": [str(OUT_ROOT / round_name / "phase736_cross_model_summary.json")],
    }


def write_cross_summary(round_name: str) -> dict[str, Any]:
    out_dir = OUT_ROOT / round_name
    summaries = []
    for model in MODELS:
        path = out_dir / f"phase736_{model}_summary.json"
        if path.exists():
            summaries.append(json.loads(path.read_text(encoding="utf-8")))
    payload = {
        "phase": 736,
        "title": "Source-Restricted Replacement and Generation Closure",
        "round": round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": [s["model"] for s in summaries],
        "status": "complete" if len(summaries) == len(MODELS) else "partial",
        "evidence_type": "donor-recipient source contribution replacement plus greedy generation closure",
        "by_model": {s["model"]: s for s in summaries},
    }
    write_json(out_dir / "phase736_cross_model_summary.json", payload)
    graph = build_atlas_graph(payload, round_name)
    write_json(out_dir / "phase736_atlas_graph.json", graph)
    lines = [
        f"# Phase 736 Source-Restricted Replacement and Generation Closure ({round_name})",
        "",
        f"- Status: `{payload['status']}`",
        f"- Models: `{payload['models']}`",
        "- Evidence type: source contribution replacement from donor prompt to recipient prompt, with likelihood and greedy generation checks.",
        "",
        "| model | target site | top replacement path | restore | donor logprob | donor hit gain | changed | role |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    for model, summary in payload["by_model"].items():
        rec = (summary.get("top_replacement_paths") or [{}])[0]
        label = f"{rec.get('component_id')}<-{rec.get('source_group')} {rec.get('direction')}"
        lines.append(
            f"| {model} | {summary.get('target_site')} | {label} | "
            f"{(rec.get('mean_restore_projection') or 0):.3f} | {(rec.get('mean_donor_logprob_delta') or 0):.3f} | "
            f"{(rec.get('donor_hit_gain_rate') or 0):.3f} | {(rec.get('changed_rate') or 0):.3f} | {rec.get('role_guess')} |"
        )
    lines.extend(
        [
            "",
            "## Strict Interpretation",
            "",
            "- Positive restore projection means the recipient hidden state moved toward the donor hidden state after source contribution replacement.",
            "- Positive donor logprob delta means the donor answer became more supported in the recipient context.",
            "- Generation hit gain is the strictest metric and can remain sparse even when hidden/readout effects are present.",
            "",
            f"Atlas graph: nodes={graph['metrics']['node_count']} edges={graph['metrics']['edge_count']}",
            "",
        ]
    )
    (out_dir / "phase736_cross_model_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"round": round_name, "status": payload["status"], "models": payload["models"], "atlas": graph["metrics"]}, ensure_ascii=False, indent=2), flush=True)
    return payload


def dry_run(args) -> None:
    preferred = set(args.preferred_sources.split(",")) if args.preferred_sources else None
    payload = {"round": args.round_name, "pairs": len(select_conflict_pairs(args.max_pairs, args.include_extended_relations)), "models": {}}
    for model in MODELS:
        payload["models"][model] = load_phase735_source_specs(model, args.phase735_round, args.top_paths, preferred)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="main")
    parser.add_argument("--phase735-round", default="confirm")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-pairs", type=int, default=8)
    parser.add_argument("--top-paths", type=int, default=4)
    parser.add_argument("--preferred-sources", default="")
    parser.add_argument("--include-extended-relations", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=3)
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
