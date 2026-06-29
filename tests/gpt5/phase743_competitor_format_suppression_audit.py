#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import re
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

from model_utils import release_model  # noqa: E402
from phase599_final_layer_washout_decomposition import extract_tensor, get_final_norm  # noqa: E402
from phase722_functional_head_atlas_causal_ablation import logit_diag, target_token_ids, write_json, write_jsonl  # noqa: E402
from phase723_apple_fruit_attribute_micro_atlas import prompt_for  # noqa: E402
from phase727_category_fruit_cluster_intervention import norm_text  # noqa: E402
from phase735_source_restricted_writer_validation import MODELS, load_model_bf16_eager, safe_mean  # noqa: E402
from phase736_source_replacement_generation_closure import select_conflict_pairs  # noqa: E402
from phase737_writer_rewriter_joint_replacement import intervention_label  # noqa: E402
from phase738_readout_margin_continuation_audit import decode_token, top_vocab  # noqa: E402
from phase739_readout_threshold_closure_boundary import (  # noqa: E402
    choose_donor_recipient,
    get_unembed,
    prepare_joint_install,
)
from phase740_natural_readout_boost_source_backtrace import first_token_id, load_phase739_audits  # noqa: E402
from phase741_threshold_candidate_causal_validation import capture_state, combine_installers  # noqa: E402
from phase742_combined_threshold_component_closure import install_combo_add, load_phase741_ranked_candidates  # noqa: E402


OUT_ROOT = Path("results/glm5_phase743_competitor_format_suppression_audit")

FORMAT_WORDS = {
    "",
    "answer",
    "the",
    "a",
    "an",
    "is",
    "are",
    "value",
    "category",
    "color",
    "taste",
    "shape",
    "edible",
    "yes",
    "no",
    "true",
    "false",
    "it",
    "this",
    "object",
    "relation",
    "record",
    "records",
    "apple",
    "fruit",
}
PROSE_WORDS = {"because", "therefore", "so", "in", "on", "of", "for", "with", "and", "but", "as"}
PUNCT_OR_STOP = {".", ",", ":", ";", "-", "=", "(", ")", "[", "]", "{", "}", "\"", "'", "\n", "\\n"}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def clean_token_text(text: str) -> str:
    return text.replace("Ġ", " ").replace("▁", " ").replace("\n", "\\n").strip().lower()


def parse_record_values(case: dict[str, Any]) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in str(case.get("records") or "").splitlines():
        if "=" not in line or "." not in line:
            continue
        left, right = line.split("=", 1)
        relation = left.split(".", 1)[1].strip()
        values[relation] = right.strip()
    return values


def token_id_for_text(tokenizer, text: str) -> int | None:
    try:
        ids = target_token_ids(tokenizer, text)
        if not ids:
            return None
        return int(ids[0])
    except Exception:
        return None


def taxonomy_context(tokenizer, donor: dict[str, Any], recipient: dict[str, Any]) -> dict[str, Any]:
    donor_values = parse_record_values(donor)
    recipient_values = parse_record_values(recipient)
    all_values = sorted(set(donor_values.values()) | set(recipient_values.values()) | {donor["answer"], recipient["answer"]})
    value_token_ids: dict[int, str] = {}
    for value in all_values:
        tid = token_id_for_text(tokenizer, value)
        if tid is not None:
            value_token_ids[tid] = value
    object_ids: dict[int, str] = {}
    for text in {donor.get("object", ""), recipient.get("object", ""), donor.get("relation", ""), recipient.get("relation", "")}:
        if not text:
            continue
        tid = token_id_for_text(tokenizer, text)
        if tid is not None:
            object_ids[tid] = text
    donor_id = first_token_id(tokenizer, donor["answer"])
    recipient_id = first_token_id(tokenizer, recipient["answer"])
    return {
        "donor_values": donor_values,
        "recipient_values": recipient_values,
        "all_values": all_values,
        "value_token_ids": value_token_ids,
        "object_ids": object_ids,
        "donor_id": donor_id,
        "recipient_id": recipient_id,
    }


def classify_token(tokenizer, token_id: int, text: str, ctx: dict[str, Any]) -> str:
    tid = int(token_id)
    cleaned = clean_token_text(text)
    if tid == int(ctx["donor_id"]):
        return "donor_answer"
    if tid == int(ctx["recipient_id"]):
        return "recipient_answer"
    if tid in ctx["value_token_ids"]:
        return "other_semantic_value"
    if tid in ctx["object_ids"]:
        return "echo_object_or_relation"
    if cleaned in FORMAT_WORDS:
        return "format_or_schema"
    if cleaned in PROSE_WORDS:
        return "prose_prefix"
    if cleaned in PUNCT_OR_STOP or re.fullmatch(r"[\\n\s\W]+", text or ""):
        return "punctuation_or_stop"
    if norm_text(cleaned) in {norm_text(v) for v in ctx["all_values"]}:
        return "other_semantic_value"
    return "other_vocab"


def margin(logits: torch.Tensor, left_id: int, right_id: int) -> float:
    return float((logits[int(left_id)] - logits[int(right_id)]).item())


def competitor_direction(unembed: torch.Tensor, competitor_id: int, donor_id: int) -> torch.Tensor | None:
    diff = unembed[int(competitor_id)] - unembed[int(donor_id)]
    n = float(torch.linalg.vector_norm(diff).item())
    if n <= 1e-8:
        return None
    return diff / n


def suppression_needed(logits: torch.Tensor, unembed: torch.Tensor, competitor_id: int, donor_id: int, direction: torch.Tensor | None) -> float | None:
    if int(competitor_id) == int(donor_id):
        return 0.0
    if direction is None:
        return None
    gap = float((logits[int(competitor_id)] - logits[int(donor_id)]).item())
    if gap <= 0:
        return 0.0
    denom = float(torch.dot(unembed[int(competitor_id)] - unembed[int(donor_id)], direction).item())
    if denom <= 1e-8:
        return None
    return gap / denom


def forward_with_final_delta(
    model,
    device,
    ids: list[int],
    install_hooks: Callable[[], list[Any]] | None,
    final_delta: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    handles = install_hooks() if install_hooks else []
    final_vec: torch.Tensor | None = None
    final_norm = get_final_norm(model)
    if final_norm is None:
        raise RuntimeError("final norm not found")

    def final_hook(_module, _inputs, output):
        nonlocal final_vec
        y = extract_tensor(output)
        final_vec = y[0, -1].detach().float().cpu()
        if final_delta is None:
            return output
        y_new = y.clone()
        y_new[0, -1, :] = y_new[0, -1, :] + final_delta.to(device=y_new.device, dtype=y_new.dtype)
        if isinstance(output, tuple):
            return (y_new,) + output[1:]
        return y_new

    handles.append(final_norm.register_forward_hook(final_hook))
    try:
        with torch.inference_mode():
            out = model(input_ids=torch.tensor([ids], device=device), return_dict=True, use_cache=False)
        return out.logits[0, -1].detach().float().cpu(), final_vec
    finally:
        for h in handles:
            h.remove()


def build_combo_installer(
    model,
    install_joint: Callable[[], list[Any]],
    combo: list[dict[str, Any]],
    deltas: dict[str, torch.Tensor],
) -> Callable[[], list[Any]]:
    return combine_installers(install_joint, install_combo_add(model, combo, deltas))


def top_vocab_with_classes(logits: torch.Tensor, tokenizer, ctx: dict[str, Any], k: int) -> list[dict[str, Any]]:
    rows = []
    for item in top_vocab(logits, tokenizer, k):
        rows.append({**item, "class": classify_token(tokenizer, int(item["token_id"]), item["token_text"], ctx)})
    return rows


def audit_pair(
    model,
    tokenizer,
    device,
    model_name: str,
    target_site: str,
    pair: dict[str, Any],
    audit: dict[str, Any],
    candidates: list[dict[str, Any]],
    top_k_vocab: int,
    suppress_scales: list[float],
) -> list[dict[str, Any]]:
    intervention = audit["intervention"]
    direction_name = audit["direction"]
    donor, recipient = choose_donor_recipient(pair, direction_name)
    donor_ids = tokenizer.encode(prompt_for(donor), add_special_tokens=False)
    recipient_ids = tokenizer.encode(prompt_for(recipient), add_special_tokens=False)
    ctx = taxonomy_context(tokenizer, donor, recipient)
    donor_id = int(ctx["donor_id"])
    recipient_id = int(ctx["recipient_id"])
    component_sites = [c["component_id"] for c in candidates]

    _meta, install_joint = prepare_joint_install(model, tokenizer, device, target_site, recipient, donor, recipient_ids, donor_ids, intervention)
    recipient_state = capture_state(model, device, recipient_ids, component_sites)
    donor_state = capture_state(model, device, donor_ids, component_sites)
    deltas = {site: donor_state["components"][site] - recipient_state["components"][site] for site in component_sites}
    combo = candidates
    install_combo = build_combo_installer(model, install_joint, combo, deltas)
    logits, final_vec = forward_with_final_delta(model, device, recipient_ids, install_combo, None)
    vocab_rows = top_vocab_with_classes(logits, tokenizer, ctx, top_k_vocab)
    top = vocab_rows[0]
    top_id = int(top["token_id"])
    unembed = get_unembed(model)
    direction = competitor_direction(unembed, top_id, donor_id)
    alpha = suppression_needed(logits, unembed, top_id, donor_id, direction)
    donor_diag = logit_diag(logits, donor_id)
    recip_diag = logit_diag(logits, recipient_id)
    rows: list[dict[str, Any]] = [
        {
            "model": model_name,
            "target_site": target_site,
            "pair_id": pair["pair_id"],
            "direction": direction_name,
            "object": donor["object"],
            "relation": donor["relation"],
            "donor_answer": donor["answer"],
            "recipient_answer": recipient["answer"],
            "donor_token_id": donor_id,
            "recipient_token_id": recipient_id,
            "intervention_label": intervention_label(intervention),
            "intervention_mode": intervention["mode"],
            "combo_components": [c["component_id"] for c in combo],
            "condition": "joint_add_topK",
            "suppression_scale": 0.0,
            "suppressed_token_id": top_id,
            "suppressed_token_text": top["token_text"],
            "suppressed_token_class": top["class"],
            "suppression_alpha_needed": alpha,
            "top_token_id": top_id,
            "top_token_text": top["token_text"],
            "top_token_class": top["class"],
            "donor_target_rank": donor_diag["target_rank"],
            "donor_top1": donor_diag["target_top1"],
            "recipient_target_rank": recip_diag["target_rank"],
            "margin_donor_vs_top": margin(logits, donor_id, top_id),
            "margin_donor_vs_recipient": margin(logits, donor_id, recipient_id),
            "final_norm_output_norm": float(torch.linalg.vector_norm(final_vec).item()) if final_vec is not None else None,
            "top_vocab": vocab_rows,
        }
    ]
    if alpha is None or direction is None or alpha <= 0:
        return rows

    for scale in suppress_scales:
        if scale <= 0:
            continue
        delta = -float(scale) * float(alpha) * direction
        logits2, final_vec2 = forward_with_final_delta(model, device, recipient_ids, install_combo, delta)
        vocab2 = top_vocab_with_classes(logits2, tokenizer, ctx, top_k_vocab)
        top2 = vocab2[0]
        donor_diag2 = logit_diag(logits2, donor_id)
        recip_diag2 = logit_diag(logits2, recipient_id)
        rows.append(
            {
                "model": model_name,
                "target_site": target_site,
                "pair_id": pair["pair_id"],
                "direction": direction_name,
                "object": donor["object"],
                "relation": donor["relation"],
                "donor_answer": donor["answer"],
                "recipient_answer": recipient["answer"],
                "donor_token_id": donor_id,
                "recipient_token_id": recipient_id,
                "intervention_label": intervention_label(intervention),
                "intervention_mode": intervention["mode"],
                "combo_components": [c["component_id"] for c in combo],
                "condition": "suppress_current_top",
                "suppression_scale": float(scale),
                "suppressed_token_id": top_id,
                "suppressed_token_text": top["token_text"],
                "suppressed_token_class": top["class"],
                "suppression_alpha_needed": alpha,
                "top_token_id": int(top2["token_id"]),
                "top_token_text": top2["token_text"],
                "top_token_class": top2["class"],
                "donor_target_rank": donor_diag2["target_rank"],
                "donor_top1": donor_diag2["target_top1"],
                "recipient_target_rank": recip_diag2["target_rank"],
                "margin_donor_vs_top": margin(logits2, donor_id, int(top2["token_id"])),
                "margin_donor_vs_recipient": margin(logits2, donor_id, recipient_id),
                "final_norm_output_norm": float(torch.linalg.vector_norm(final_vec2).item()) if final_vec2 is not None else None,
                "top_vocab": vocab2,
            }
        )
    return rows


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["condition"], float(row["suppression_scale"]))].append(row)
    out = []
    for (condition, scale), vals in groups.items():
        n = len(vals)
        out.append(
            {
                "condition": condition,
                "suppression_scale": scale,
                "n": n,
                "donor_top1_rate": sum(1 for v in vals if v["donor_top1"]) / n,
                "mean_donor_rank": safe_mean([v["donor_target_rank"] for v in vals]),
                "mean_margin_donor_vs_top": safe_mean([v["margin_donor_vs_top"] for v in vals]),
                "mean_margin_donor_vs_recipient": safe_mean([v["margin_donor_vs_recipient"] for v in vals]),
                "mean_suppression_alpha_needed": safe_mean([v["suppression_alpha_needed"] for v in vals]),
                "top_token_class_counts": dict(Counter(v["top_token_class"] for v in vals)),
                "top_token_counts": dict(Counter(v["top_token_text"] for v in vals)),
                "suppressed_token_class_counts": dict(Counter(v["suppressed_token_class"] for v in vals)),
            }
        )
    return sorted(out, key=lambda r: (r["condition"], r["suppression_scale"]))


def summarize_by_suppressed_class(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    base = [r for r in rows if r["condition"] == "joint_add_topK"]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in base:
        groups[row["suppressed_token_class"]].append(row)
    out = []
    for cls, vals in groups.items():
        out.append(
            {
                "suppressed_token_class": cls,
                "n": len(vals),
                "mean_margin_donor_vs_top": safe_mean([v["margin_donor_vs_top"] for v in vals]),
                "mean_suppression_alpha_needed": safe_mean([v["suppression_alpha_needed"] for v in vals]),
                "top_token_counts": dict(Counter(v["top_token_text"] for v in vals)),
            }
        )
    return sorted(out, key=lambda r: r["n"], reverse=True)


def run_model(args) -> dict[str, Any]:
    out_dir = OUT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    audit_payload = load_phase739_audits(args.model, args.phase739_round, args.top_audits)
    candidates = load_phase741_ranked_candidates(args.model, args.phase741_round, args.top_candidates)
    pairs = select_conflict_pairs(args.max_pairs, args.include_extended_relations)
    log(
        f"{args.model}/{args.round_name}: pairs={len(pairs)} target={audit_payload['target_site']} "
        f"audits={len(audit_payload['audits'])} combo_candidates={len(candidates)}"
    )
    model, tokenizer, device, attn_impl = load_model_bf16_eager(args.model)
    try:
        rows: list[dict[str, Any]] = []
        for pair_idx, pair in enumerate(pairs, 1):
            for audit in audit_payload["audits"]:
                rows.extend(
                    audit_pair(
                        model,
                        tokenizer,
                        device,
                        args.model,
                        audit_payload["target_site"],
                        pair,
                        audit,
                        candidates,
                        args.top_k_vocab,
                        args.suppress_scales,
                    )
                )
            if pair_idx % args.log_every == 0 or pair_idx == len(pairs):
                log(f"{args.model}: competitor suppression audit {pair_idx}/{len(pairs)} pairs; rows={len(rows)}")
    finally:
        release_model(model)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = {
        "phase": 743,
        "title": "Competitor and Format Suppression Audit",
        "model": args.model,
        "round": args.round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "quantization": "off",
        "dtype": "bfloat16",
        "phase739_round": args.phase739_round,
        "phase741_round": args.phase741_round,
        "target_site": audit_payload["target_site"],
        "max_pairs": args.max_pairs,
        "top_audits": args.top_audits,
        "top_candidates": args.top_candidates,
        "top_k_vocab": args.top_k_vocab,
        "suppress_scales": args.suppress_scales,
        "audited_interventions": [{"label": intervention_label(a["intervention"]), "direction": a["direction"]} for a in audit_payload["audits"]],
        "ranked_candidate_components": candidates,
        "n_rows": len(rows),
        "condition_summary": summarize_rows(rows),
        "suppressed_class_summary": summarize_by_suppressed_class(rows),
        "strict_interpretation": "Final-norm suppression of the current top competitor is a readout-geometry intervention. It tests whether failure is local token competition, not whether the model naturally performs that suppression.",
    }
    write_jsonl(out_dir / f"phase743_{args.model}_rows.jsonl", rows)
    write_json(out_dir / f"phase743_{args.model}_summary.json", summary)
    print(json.dumps({"model": args.model, "round": args.round_name, "condition_summary": summary["condition_summary"], "suppressed_class_summary": summary["suppressed_class_summary"]}, ensure_ascii=False, indent=2), flush=True)
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
        base_node = f"{model}:combo_state"
        closure_node = f"{model}:token0_closure"
        add_node({"id": f"{model}:model", "type": "model", "label": model, "model": model, "position": [-24, 0, lane_z], "role": "tested_model"})
        add_node({"id": base_node, "type": "state", "label": "joint+topK threshold components", "model": model, "position": [-4, 0, lane_z], "role": "near_readout_state"})
        add_node({"id": closure_node, "type": "readout", "label": "token0 closure", "model": model, "position": [24, 0, lane_z], "role": "closure_target"})
        edges.append({"source": f"{model}:model", "target": base_node, "relation": "produces_combo_state", "phase": 743})
        edges.append({"source": base_node, "target": closure_node, "relation": "competes_for_readout", "phase": 743})
        for rec in summary.get("suppressed_class_summary", []):
            cls_node = f"{model}:competitor:{rec['suppressed_token_class']}"
            add_node(
                {
                    "id": cls_node,
                    "type": "competitor_class",
                    "label": rec["suppressed_token_class"],
                    "model": model,
                    "position": [8, rec.get("mean_suppression_alpha_needed") or 0, lane_z],
                    "role": "readout_competitor",
                    "n": rec["n"],
                    "mean_suppression_alpha_needed": rec.get("mean_suppression_alpha_needed"),
                }
            )
            edges.append({"source": base_node, "target": cls_node, "relation": "has_competitor_class", "weight": rec["n"], "phase": 743})
            edges.append({"source": cls_node, "target": closure_node, "relation": "blocks_closure", "weight": rec.get("mean_suppression_alpha_needed"), "phase": 743})
    return {
        "schema_version": "atlas_graph_v1",
        "title": f"Phase 743 Competitor and Format Suppression Audit ({round_name})",
        "model_info": {"model": "cross_model", "models": payload.get("models", []), "phase": 743, "round": round_name, "timestamp": payload.get("timestamp"), "evidence_type": payload.get("evidence_type")},
        "layout": {"x": "model -> combo state -> competitor class -> closure", "y": "suppression alpha", "z": "model lane"},
        "graph": {"nodes": nodes, "edges": edges},
        "metrics": {"node_count": len(nodes), "edge_count": len(edges), "source_phase": 743},
        "source_files": [str(OUT_ROOT / round_name / "phase743_cross_model_summary.json")],
    }


def write_cross_summary(round_name: str) -> dict[str, Any]:
    out_dir = OUT_ROOT / round_name
    summaries = []
    for model in MODELS:
        path = out_dir / f"phase743_{model}_summary.json"
        if path.exists():
            summaries.append(json.loads(path.read_text(encoding="utf-8")))
    payload = {
        "phase": 743,
        "title": "Competitor and Format Suppression Audit",
        "round": round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": [s["model"] for s in summaries],
        "status": "complete" if len(summaries) == len(MODELS) else "partial",
        "evidence_type": "top-k vocabulary competitor taxonomy plus final-norm current-top suppression against joint+threshold-component state",
        "by_model": {s["model"]: s for s in summaries},
    }
    write_json(out_dir / "phase743_cross_model_summary.json", payload)
    graph = build_atlas_graph(payload, round_name)
    write_json(out_dir / "phase743_atlas_graph.json", graph)
    lines = [
        f"# Phase 743 Competitor and Format Suppression Audit ({round_name})",
        "",
        f"- Status: `{payload['status']}`",
        f"- Models: `{payload['models']}`",
        "- Evidence type: top-k vocabulary competitor classes and final-norm suppression of current top competitor.",
        "",
        "| model | condition | scale | donor top1 | mean donor rank | margin donor vs top | top classes | suppressed classes |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    for model, summary in payload["by_model"].items():
        for row in summary.get("condition_summary", []):
            lines.append(
                f"| {model} | {row['condition']} | {row['suppression_scale']:.2f} | "
                f"{(row.get('donor_top1_rate') or 0):.3f} | "
                f"{(row.get('mean_donor_rank') or 0):.2f} | "
                f"{(row.get('mean_margin_donor_vs_top') or 0):.3f} | "
                f"`{json.dumps(row.get('top_token_class_counts') or {}, ensure_ascii=False)}` | "
                f"`{json.dumps(row.get('suppressed_token_class_counts') or {}, ensure_ascii=False)}` |"
            )
    lines.extend(
        [
            "",
            "## Suppressed Class Summary",
            "",
            "| model | class | n | mean alpha needed | mean margin donor vs top | top tokens |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    for model, summary in payload["by_model"].items():
        for row in summary.get("suppressed_class_summary", []):
            lines.append(
                f"| {model} | {row['suppressed_token_class']} | {row['n']} | "
                f"{(row.get('mean_suppression_alpha_needed') or 0):.3f} | "
                f"{(row.get('mean_margin_donor_vs_top') or 0):.3f} | "
                f"`{json.dumps(row.get('top_token_counts') or {}, ensure_ascii=False)}` |"
            )
    lines.extend(
        [
            "",
            "## Strict Interpretation",
            "",
            "- If suppressing only the current top competitor does not make donor top1, the failure is multi-competitor or global readout geometry, not a single blocking token.",
            "- If it does make donor top1, Phase 742 near-closure was mainly blocked by a local competitor class.",
            "- This is still a final readout intervention; it does not prove the natural circuit that performs suppression.",
            "",
            f"Atlas graph: nodes={graph['metrics']['node_count']} edges={graph['metrics']['edge_count']}",
            "",
        ]
    )
    (out_dir / "phase743_cross_model_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"round": round_name, "status": payload["status"], "models": payload["models"], "atlas": graph["metrics"]}, ensure_ascii=False, indent=2), flush=True)
    return payload


def dry_run(args) -> None:
    payload = {"round": args.round_name, "pairs": len(select_conflict_pairs(args.max_pairs, args.include_extended_relations)), "models": {}}
    for model in MODELS:
        audits = load_phase739_audits(model, args.phase739_round, args.top_audits)
        candidates = load_phase741_ranked_candidates(model, args.phase741_round, args.top_candidates)
        payload["models"][model] = {
            "target_site": audits["target_site"],
            "audits": [{"label": intervention_label(a["intervention"]), "direction": a["direction"]} for a in audits["audits"]],
            "ranked_candidates": candidates,
        }
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="main")
    parser.add_argument("--phase739-round", default="confirm")
    parser.add_argument("--phase741-round", default="confirm")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-pairs", type=int, default=6)
    parser.add_argument("--top-audits", type=int, default=2)
    parser.add_argument("--top-candidates", type=int, default=3)
    parser.add_argument("--top-k-vocab", type=int, default=12)
    parser.add_argument("--suppress-scales", type=float, nargs="+", default=[1.0, 1.25])
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
