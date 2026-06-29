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

from model_utils import release_model  # noqa: E402
from phase722_functional_head_atlas_causal_ablation import logit_diag, target_token_ids, write_json, write_jsonl  # noqa: E402
from phase723_apple_fruit_attribute_micro_atlas import prompt_for  # noqa: E402
from phase735_source_restricted_writer_validation import MODELS, forward_site_logits, load_model_bf16_eager, safe_mean  # noqa: E402
from phase736_source_replacement_generation_closure import select_conflict_pairs  # noqa: E402
from phase737_writer_rewriter_joint_replacement import (  # noqa: E402
    OUT_ROOT as PHASE737_ROOT,
    build_interventions,
    intervention_label,
    load_phase735_mlp_specs,
    load_phase735_source_specs,
    patched_joint_site_logits,
)


OUT_ROOT = Path("results/glm5_phase738_readout_margin_continuation_audit")

FORMAT_CANDIDATES = [
    ("format_the", "The"),
    ("format_answer", "Answer"),
    ("format_value", "value"),
    ("format_it", "It"),
    ("format_of", "of"),
]

CONTINUATION_CANDIDATES = [
    ("cont_stop_period", "."),
    ("cont_stop_newline", "\n"),
    ("cont_is", "is"),
    ("cont_of", "of"),
    ("cont_colon", ":"),
    ("cont_comma", ","),
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def decode_token(tokenizer, tid: int) -> str:
    try:
        return tokenizer.decode([int(tid)], skip_special_tokens=False)
    except Exception:
        return f"<id:{tid}>"


def first_token_id(tokenizer, text: str) -> int:
    ids = target_token_ids(tokenizer, text)
    return int(ids[0])


def token_diag(logits: torch.Tensor, tokenizer, label: str, text: str) -> dict[str, Any]:
    tid = first_token_id(tokenizer, text)
    diag = logit_diag(logits, tid)
    return {
        "label": label,
        "text": text,
        "token_id": tid,
        "token_text": decode_token(tokenizer, tid),
        **diag,
    }


def unique_candidate_specs(tokenizer, specs: list[tuple[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[int] = set()
    for label, text in specs:
        try:
            tid = first_token_id(tokenizer, text)
        except Exception:
            continue
        if tid in seen:
            continue
        seen.add(tid)
        out.append({"label": label, "text": text, "token_id": tid, "token_text": decode_token(tokenizer, tid)})
    return out


def candidate_specs_for_case(tokenizer, donor: dict[str, Any], recipient: dict[str, Any]) -> list[dict[str, Any]]:
    relation = str(donor["relation"]).replace("_", " ")
    specs = [
        ("donor_answer", donor["answer"]),
        ("recipient_answer", recipient["answer"]),
        ("object_echo", donor["object"]),
        ("relation_echo", relation),
    ] + FORMAT_CANDIDATES
    return unique_candidate_specs(tokenizer, specs)


def continuation_specs_for_case(tokenizer, donor: dict[str, Any], recipient: dict[str, Any]) -> tuple[list[dict[str, Any]], list[int], list[int]]:
    donor_ids = target_token_ids(tokenizer, donor["answer"])
    recipient_ids = target_token_ids(tokenizer, recipient["answer"])
    specs: list[tuple[str, str]] = []
    if len(donor_ids) > 1:
        specs.append(("donor_answer_token1", tokenizer.decode([donor_ids[1]], skip_special_tokens=False)))
    if len(recipient_ids) > 1:
        specs.append(("recipient_answer_token1", tokenizer.decode([recipient_ids[1]], skip_special_tokens=False)))
    specs.extend(CONTINUATION_CANDIDATES)
    specs.append(("relation_echo", str(donor["relation"]).replace("_", " ")))
    specs.append(("object_echo", donor["object"]))
    return unique_candidate_specs(tokenizer, specs), donor_ids, recipient_ids


def rank_candidates(logits: torch.Tensor, tokenizer, specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for spec in specs:
        diag = logit_diag(logits, int(spec["token_id"]))
        rows.append({**spec, **diag})
    return sorted(rows, key=lambda r: r["target_logit"], reverse=True)


def load_phase737_audits(model_name: str, round_name: str, top_audits: int) -> dict[str, Any]:
    source_payload = load_phase735_source_specs(model_name, "confirm", 3, None)
    mlp_specs = load_phase735_mlp_specs(model_name, "confirm", 2)
    all_interventions = {intervention_label(x): x for x in build_interventions(source_payload["paths"], mlp_specs, "compact")}
    summary_path = PHASE737_ROOT / round_name / f"phase737_{model_name}_summary.json"
    audits: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        source_payload["target_site"] = summary.get("target_site") or source_payload["target_site"]
        for row in summary.get("top_joint_interventions", []):
            label = row.get("intervention_label")
            direction = row.get("direction")
            if not label or not direction or label not in all_interventions:
                continue
            key = (label, direction)
            if key in seen:
                continue
            audits.append({"intervention": all_interventions[label], "direction": direction, "phase737_row": row})
            seen.add(key)
            if len(audits) >= top_audits:
                break
    if len(audits) < top_audits:
        for intervention in build_interventions(source_payload["paths"], mlp_specs, "compact"):
            for direction in ["conflict<-explicit", "explicit<-conflict"]:
                key = (intervention_label(intervention), direction)
                if key in seen:
                    continue
                audits.append({"intervention": intervention, "direction": direction, "phase737_row": None})
                seen.add(key)
                if len(audits) >= top_audits:
                    break
            if len(audits) >= top_audits:
                break
    return {"target_site": source_payload["target_site"], "audits": audits[:top_audits]}


def choose_donor_recipient(pair: dict[str, Any], direction: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if direction == "conflict<-explicit":
        return pair["explicit_profile"], pair["conflict_profile"]
    if direction == "explicit<-conflict":
        return pair["conflict_profile"], pair["explicit_profile"]
    raise ValueError(direction)


def top_vocab(logits: torch.Tensor, tokenizer, k: int = 8) -> list[dict[str, Any]]:
    vals = logits.detach().float()
    top = torch.topk(vals, k=min(k, vals.numel()))
    return [
        {
            "rank": i + 1,
            "token_id": int(tid.item()),
            "token_text": decode_token(tokenizer, int(tid.item())),
            "logit": float(score.item()),
        }
        for i, (score, tid) in enumerate(zip(top.values, top.indices))
    ]


def audit_pair(
    model,
    tokenizer,
    device,
    model_name: str,
    target_site: str,
    pair: dict[str, Any],
    audit: dict[str, Any],
) -> dict[str, Any]:
    intervention = audit["intervention"]
    direction = audit["direction"]
    donor, recipient = choose_donor_recipient(pair, direction)
    donor_prompt = prompt_for(donor)
    recipient_prompt = prompt_for(recipient)
    donor_ids = tokenizer.encode(donor_prompt, add_special_tokens=False)
    recipient_ids = tokenizer.encode(recipient_prompt, add_special_tokens=False)
    _base_vec, base_logits = forward_site_logits(model, device, recipient_ids, target_site)
    _patched_vec, patched_logits, meta = patched_joint_site_logits(
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
    token0_specs = candidate_specs_for_case(tokenizer, donor, recipient)
    base_candidates = rank_candidates(base_logits, tokenizer, token0_specs)
    patched_candidates = rank_candidates(patched_logits, tokenizer, token0_specs)
    by_label_base = {r["label"]: r for r in base_candidates}
    by_label_patch = {r["label"]: r for r in patched_candidates}
    donor0_base = by_label_base["donor_answer"]
    donor0_patch = by_label_patch["donor_answer"]
    recipient0_base = by_label_base["recipient_answer"]
    recipient0_patch = by_label_patch["recipient_answer"]
    base_margin = donor0_base["target_logit"] - recipient0_base["target_logit"]
    patched_margin = donor0_patch["target_logit"] - recipient0_patch["target_logit"]
    patched_best_candidate = patched_candidates[0]
    base_best_candidate = base_candidates[0]

    cont_specs, donor_answer_ids, recipient_answer_ids = continuation_specs_for_case(tokenizer, donor, recipient)
    donor0_id = int(donor_answer_ids[0])
    forced_recipient_ids = recipient_ids + [donor0_id]
    forced_donor_ids = donor_ids + [donor0_id]
    _forced_base_vec, forced_base_logits = forward_site_logits(model, device, forced_recipient_ids, target_site)
    _forced_patch_vec, forced_patched_logits, _forced_meta = patched_joint_site_logits(
        model,
        tokenizer,
        device,
        target_site,
        recipient,
        donor,
        forced_recipient_ids,
        forced_donor_ids,
        intervention.get("source_spec"),
        intervention.get("mlp_specs") or [],
    )
    _forced_donor_vec, forced_donor_context_logits = forward_site_logits(model, device, forced_donor_ids, target_site)
    forced_base_candidates = rank_candidates(forced_base_logits, tokenizer, cont_specs)
    forced_patched_candidates = rank_candidates(forced_patched_logits, tokenizer, cont_specs)
    forced_donor_candidates = rank_candidates(forced_donor_context_logits, tokenizer, cont_specs)
    by_cont_base = {r["label"]: r for r in forced_base_candidates}
    by_cont_patch = {r["label"]: r for r in forced_patched_candidates}
    donor_token1_available = "donor_answer_token1" in by_cont_patch
    donor_token1_delta = None
    if donor_token1_available:
        donor_token1_delta = by_cont_patch["donor_answer_token1"]["target_logprob"] - by_cont_base["donor_answer_token1"]["target_logprob"]

    return {
        "model": model_name,
        "target_site": target_site,
        "pair_id": pair["pair_id"],
        "direction": direction,
        "intervention_mode": intervention["mode"],
        "intervention_label": intervention_label(intervention),
        "source_component_id": (intervention.get("source_spec") or {}).get("component_id"),
        "source_group": (intervention.get("source_spec") or {}).get("source_group"),
        "mlp_components": [m["component_id"] for m in intervention.get("mlp_specs") or []],
        "object": donor["object"],
        "relation": donor["relation"],
        "donor_answer": donor["answer"],
        "recipient_answer": recipient["answer"],
        "token0_base_donor_rank": donor0_base["target_rank"],
        "token0_patched_donor_rank": donor0_patch["target_rank"],
        "token0_donor_rank_delta": donor0_patch["target_rank"] - donor0_base["target_rank"],
        "token0_donor_logprob_delta": donor0_patch["target_logprob"] - donor0_base["target_logprob"],
        "token0_base_margin_donor_vs_recipient": base_margin,
        "token0_patched_margin_donor_vs_recipient": patched_margin,
        "token0_margin_delta_donor_vs_recipient": patched_margin - base_margin,
        "token0_patched_margin_donor_vs_best_candidate": donor0_patch["target_logit"] - patched_best_candidate["target_logit"],
        "token0_base_best_candidate_label": base_best_candidate["label"],
        "token0_patched_best_candidate_label": patched_best_candidate["label"],
        "token0_base_best_candidate_text": base_best_candidate["token_text"],
        "token0_patched_best_candidate_text": patched_best_candidate["token_text"],
        "token0_base_vocab_top": top_vocab(base_logits, tokenizer, 8),
        "token0_patched_vocab_top": top_vocab(patched_logits, tokenizer, 8),
        "token0_candidate_base": base_candidates,
        "token0_candidate_patched": patched_candidates,
        "token1_forced_token_id": donor0_id,
        "token1_forced_token_text": decode_token(tokenizer, donor0_id),
        "token1_donor_second_available": donor_token1_available,
        "token1_donor_second_logprob_delta": donor_token1_delta,
        "token1_base_best_candidate_label": forced_base_candidates[0]["label"] if forced_base_candidates else None,
        "token1_patched_best_candidate_label": forced_patched_candidates[0]["label"] if forced_patched_candidates else None,
        "token1_donor_context_best_candidate_label": forced_donor_candidates[0]["label"] if forced_donor_candidates else None,
        "token1_base_vocab_top": top_vocab(forced_base_logits, tokenizer, 8),
        "token1_patched_vocab_top": top_vocab(forced_patched_logits, tokenizer, 8),
        "token1_donor_context_vocab_top": top_vocab(forced_donor_context_logits, tokenizer, 8),
        "token1_candidate_base": forced_base_candidates,
        "token1_candidate_patched": forced_patched_candidates,
        "token1_candidate_donor_context": forced_donor_candidates,
        **meta,
    }


def run_audits(model, tokenizer, device, model_name: str, target_site: str, pairs: list[dict[str, Any]], audits: list[dict[str, Any]], log_every: int) -> list[dict[str, Any]]:
    rows = []
    for pair_idx, pair in enumerate(pairs, 1):
        for audit in audits:
            rows.append(audit_pair(model, tokenizer, device, model_name, target_site, pair, audit))
        if pair_idx % log_every == 0 or pair_idx == len(pairs):
            log(f"{model_name}: readout audit {pair_idx}/{len(pairs)} pairs; rows={len(rows)}")
    return rows


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["intervention_label"] + " " + row["direction"]].append(row)
    out = []
    for label, vals in groups.items():
        n = len(vals)
        patched_best_counts: dict[str, int] = defaultdict(int)
        token1_best_counts: dict[str, int] = defaultdict(int)
        for v in vals:
            patched_best_counts[str(v["token0_patched_best_candidate_label"])] += 1
            token1_best_counts[str(v["token1_patched_best_candidate_label"])] += 1
        donor_top_rate = sum(1 for v in vals if v["token0_patched_best_candidate_label"] == "donor_answer") / n
        recipient_top_rate = sum(1 for v in vals if v["token0_patched_best_candidate_label"] == "recipient_answer") / n
        format_top_rate = sum(1 for v in vals if str(v["token0_patched_best_candidate_label"]).startswith("format_")) / n
        echo_top_rate = sum(1 for v in vals if str(v["token0_patched_best_candidate_label"]).endswith("_echo")) / n
        out.append(
            {
                "intervention_key": label,
                "intervention_label": vals[0]["intervention_label"],
                "direction": vals[0]["direction"],
                "intervention_mode": vals[0]["intervention_mode"],
                "source_component_id": vals[0]["source_component_id"],
                "source_group": vals[0]["source_group"],
                "mlp_components": vals[0]["mlp_components"],
                "n": n,
                "mean_token0_donor_logprob_delta": safe_mean([v["token0_donor_logprob_delta"] for v in vals]),
                "mean_token0_margin_delta_donor_vs_recipient": safe_mean([v["token0_margin_delta_donor_vs_recipient"] for v in vals]),
                "mean_token0_patched_margin_donor_vs_recipient": safe_mean([v["token0_patched_margin_donor_vs_recipient"] for v in vals]),
                "mean_token0_patched_margin_donor_vs_best_candidate": safe_mean([v["token0_patched_margin_donor_vs_best_candidate"] for v in vals]),
                "mean_token0_donor_rank_delta": safe_mean([v["token0_donor_rank_delta"] for v in vals]),
                "mean_token0_patched_donor_rank": safe_mean([v["token0_patched_donor_rank"] for v in vals]),
                "token0_donor_top_candidate_rate": donor_top_rate,
                "token0_recipient_top_candidate_rate": recipient_top_rate,
                "token0_format_top_candidate_rate": format_top_rate,
                "token0_echo_top_candidate_rate": echo_top_rate,
                "token0_patched_best_counts": dict(sorted(patched_best_counts.items())),
                "token1_patched_best_counts": dict(sorted(token1_best_counts.items())),
                "token1_format_or_echo_top_rate": sum(
                    1
                    for v in vals
                    if str(v["token1_patched_best_candidate_label"]).startswith("cont_")
                    or str(v["token1_patched_best_candidate_label"]).endswith("_echo")
                )
                / n,
                "mean_source_contribution_delta_norm": safe_mean([v["source_contribution_delta_norm"] for v in vals]),
                "mean_mlp_delta_norm_total": safe_mean([v["mlp_delta_norm_total"] for v in vals]),
            }
        )
    return sorted(
        out,
        key=lambda r: (
            r["token0_donor_top_candidate_rate"],
            r["mean_token0_margin_delta_donor_vs_recipient"] or 0,
            -(r["mean_token0_patched_donor_rank"] or 999999),
        ),
        reverse=True,
    )


def build_summary(model_name: str, round_name: str, target_site: str, audits: list[dict[str, Any]], rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    summary = summarize_rows(rows)
    return {
        "phase": 738,
        "title": "Readout Margin and Token Continuation Gate Audit",
        "model": model_name,
        "round": round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": "eager",
        "attention_note": "eager attention is required because audited Phase737 interventions can include source contribution replacement",
        "quantization": "off",
        "dtype": "bfloat16",
        "phase737_round": args.phase737_round,
        "target_site": target_site,
        "top_audits": args.top_audits,
        "max_pairs": args.max_pairs,
        "n_rows": len(rows),
        "audited_interventions": [
            {"label": intervention_label(a["intervention"]), "direction": a["direction"], "phase737_role": (a.get("phase737_row") or {}).get("role_guess")}
            for a in audits
        ],
        "top_readout_audits": summary[:32],
        "strict_interpretation": "This phase audits why positive margin deltas still fail generation: token0 candidate competition and forced-token continuation are reported separately.",
    }


def run_model(args) -> dict[str, Any]:
    out_dir = OUT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    audit_payload = load_phase737_audits(args.model, args.phase737_round, args.top_audits)
    pairs = select_conflict_pairs(args.max_pairs, args.include_extended_relations)
    log(f"{args.model}/{args.round_name}: pairs={len(pairs)} target={audit_payload['target_site']} audits={len(audit_payload['audits'])}")
    model, tokenizer, device, _attn_impl = load_model_bf16_eager(args.model)
    try:
        rows = run_audits(model, tokenizer, device, args.model, audit_payload["target_site"], pairs, audit_payload["audits"], args.log_every)
    finally:
        release_model(model)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    summary = build_summary(args.model, args.round_name, audit_payload["target_site"], audit_payload["audits"], rows, args)
    write_jsonl(out_dir / f"phase738_{args.model}_readout_rows.jsonl", rows)
    write_json(out_dir / f"phase738_{args.model}_summary.json", summary)
    print(json.dumps({"model": args.model, "round": args.round_name, "target_site": audit_payload["target_site"], "top_readout_audits": summary["top_readout_audits"][:5]}, ensure_ascii=False, indent=2), flush=True)
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
        phase_node = f"{model}:phase:738:{round_name}"
        readout_node = f"{model}:readout_competition:{summary['target_site']}"
        add_node({"id": f"{model}:model", "type": "model", "label": model, "model": model, "position": [-34, 0, lane_z], "role": "tested_model"})
        add_node({"id": phase_node, "type": "phase", "label": f"Phase 738 {round_name}", "model": model, "position": [-26, 2, lane_z], "role": "readout_margin_continuation_audit"})
        add_node({"id": readout_node, "type": "readout", "label": summary["target_site"], "model": model, "position": [12, 0, lane_z], "role": "readout_competition"})
        edges.append({"source": f"{model}:model", "target": phase_node, "relation": "contains", "phase": 738})
        edges.append({"source": phase_node, "target": readout_node, "relation": "audits", "phase": 738})
        for rec in summary.get("top_readout_audits", [])[:12]:
            audit_node = f"{model}:readout_audit:{round_name}:{rec['intervention_key']}"
            add_node(
                {
                    "id": audit_node,
                    "type": "readout_audit",
                    "label": rec["intervention_mode"],
                    "model": model,
                    "role": "token0_competition",
                    "margin_delta": rec["mean_token0_margin_delta_donor_vs_recipient"],
                    "patched_margin": rec["mean_token0_patched_margin_donor_vs_recipient"],
                    "donor_top_rate": rec["token0_donor_top_candidate_rate"],
                }
            )
            edges.append({"source": audit_node, "target": readout_node, "relation": "explains_competition_failure", "weight": rec["mean_token0_patched_margin_donor_vs_best_candidate"], "phase": 738})
            for label, count in rec.get("token0_patched_best_counts", {}).items():
                comp_node = f"{model}:competitor:{label}"
                add_node({"id": comp_node, "type": "token_competitor", "label": label, "model": model, "role": "readout_competitor"})
                edges.append({"source": comp_node, "target": audit_node, "relation": "wins_candidate_competition", "weight": count, "phase": 738})
    return {
        "schema_version": "atlas_graph_v1",
        "title": f"Phase 738 Readout Margin and Continuation Audit ({round_name})",
        "model_info": {"model": "cross_model", "models": payload.get("models", []), "phase": 738, "round": round_name, "timestamp": payload.get("timestamp"), "evidence_type": payload.get("evidence_type")},
        "layout": {"x": "patched intervention -> readout competitors -> continuation gate", "y": "competition class", "z": "model lane"},
        "graph": {"nodes": nodes, "edges": edges},
        "metrics": {"node_count": len(nodes), "edge_count": len(edges), "source_phase": 738},
        "source_files": [str(OUT_ROOT / round_name / "phase738_cross_model_summary.json")],
    }


def write_cross_summary(round_name: str) -> dict[str, Any]:
    out_dir = OUT_ROOT / round_name
    summaries = []
    for model in MODELS:
        path = out_dir / f"phase738_{model}_summary.json"
        if path.exists():
            summaries.append(json.loads(path.read_text(encoding="utf-8")))
    payload = {
        "phase": 738,
        "title": "Readout Margin and Token Continuation Gate Audit",
        "round": round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": [s["model"] for s in summaries],
        "status": "complete" if len(summaries) == len(MODELS) else "partial",
        "evidence_type": "token0 readout competition plus forced donor-token continuation audit",
        "by_model": {s["model"]: s for s in summaries},
    }
    write_json(out_dir / "phase738_cross_model_summary.json", payload)
    graph = build_atlas_graph(payload, round_name)
    write_json(out_dir / "phase738_atlas_graph.json", graph)
    lines = [
        f"# Phase 738 Readout Margin and Token Continuation Gate Audit ({round_name})",
        "",
        f"- Status: `{payload['status']}`",
        f"- Models: `{payload['models']}`",
        "- Evidence type: token0 candidate competition and forced donor-token continuation.",
        "",
        "| model | target site | top audit | margin delta | patched margin | donor top rate | top patched competitor counts | token1 counts |",
        "|---|---|---|---:|---:|---:|---|---|",
    ]
    for model, summary in payload["by_model"].items():
        rec = (summary.get("top_readout_audits") or [{}])[0]
        lines.append(
            f"| {model} | {summary.get('target_site')} | {rec.get('intervention_key')} | "
            f"{(rec.get('mean_token0_margin_delta_donor_vs_recipient') or 0):.3f} | "
            f"{(rec.get('mean_token0_patched_margin_donor_vs_recipient') or 0):.3f} | "
            f"{(rec.get('token0_donor_top_candidate_rate') or 0):.3f} | "
            f"{rec.get('token0_patched_best_counts')} | {rec.get('token1_patched_best_counts')} |"
        )
    lines.extend(
        [
            "",
            "## Strict Interpretation",
            "",
            "- Positive margin delta only means donor answer improved against recipient answer.",
            "- Negative patched margin means donor answer still loses readout competition.",
            "- Token1 counts show what continuation route is preferred after forcing donor token0.",
            "",
            f"Atlas graph: nodes={graph['metrics']['node_count']} edges={graph['metrics']['edge_count']}",
            "",
        ]
    )
    (out_dir / "phase738_cross_model_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"round": round_name, "status": payload["status"], "models": payload["models"], "atlas": graph["metrics"]}, ensure_ascii=False, indent=2), flush=True)
    return payload


def dry_run(args) -> None:
    payload = {"round": args.round_name, "pairs": len(select_conflict_pairs(args.max_pairs, args.include_extended_relations)), "models": {}}
    for model in MODELS:
        audits = load_phase737_audits(model, args.phase737_round, args.top_audits)
        payload["models"][model] = {
            "target_site": audits["target_site"],
            "audits": [{"label": intervention_label(a["intervention"]), "direction": a["direction"]} for a in audits["audits"]],
        }
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="main")
    parser.add_argument("--phase737-round", default="confirm")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-pairs", type=int, default=8)
    parser.add_argument("--top-audits", type=int, default=5)
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
