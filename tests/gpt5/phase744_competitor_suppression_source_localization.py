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
from phase722_functional_head_atlas_causal_ablation import logit_diag, write_json, write_jsonl  # noqa: E402
from phase723_apple_fruit_attribute_micro_atlas import prompt_for  # noqa: E402
from phase735_source_restricted_writer_validation import MODELS, load_model_bf16_eager, safe_mean  # noqa: E402
from phase736_source_replacement_generation_closure import select_conflict_pairs  # noqa: E402
from phase737_writer_rewriter_joint_replacement import intervention_label  # noqa: E402
from phase738_readout_margin_continuation_audit import top_vocab  # noqa: E402
from phase739_readout_threshold_closure_boundary import choose_donor_recipient, prepare_joint_install  # noqa: E402
from phase740_natural_readout_boost_source_backtrace import first_token_id, load_phase739_audits  # noqa: E402
from phase741_threshold_candidate_causal_validation import (  # noqa: E402
    capture_state,
    combine_installers,
    install_component_edit,
    module_for_component,
    parse_component_site,
)
from phase742_combined_threshold_component_closure import install_combo_add, load_phase741_ranked_candidates  # noqa: E402
from phase743_competitor_format_suppression_audit import classify_token, taxonomy_context, top_vocab_with_classes  # noqa: E402


OUT_ROOT = Path("results/glm5_phase744_competitor_suppression_source_localization")

SCAN_LAYERS = {
    "qwen3": [28, 30, 31, 32, 33, 34, 35],
    "glm4": [34, 35, 36, 37, 38, 39],
    "deepseek7b": [22, 23, 24, 25, 26, 27],
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def margin(logits: torch.Tensor, left_id: int, right_id: int) -> float:
    return float((logits[int(left_id)] - logits[int(right_id)]).item())


def build_scan_candidates(model, model_name: str, combo: list[dict[str, Any]], include_combo: bool) -> list[dict[str, Any]]:
    combo_ids = {c["component_id"] for c in combo}
    n_layers = len(get_layers(model))
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for layer in SCAN_LAYERS[model_name]:
        if layer < 0 or layer >= n_layers:
            continue
        for component in ["attn_out", "mlp_out"]:
            cid = f"L{layer}:{component}"
            if cid in seen:
                continue
            if (not include_combo) and cid in combo_ids:
                continue
            try:
                module_for_component(model, cid)
                parsed_layer, parsed_component = parse_component_site(cid)
            except Exception:
                continue
            out.append(
                {
                    "component_id": cid,
                    "layer": parsed_layer,
                    "component": parsed_component,
                    "already_in_threshold_combo": cid in combo_ids,
                }
            )
            seen.add(cid)
    return out


def build_combo_installer(
    model,
    install_joint: Callable[[], list[Any]],
    combo: list[dict[str, Any]],
    deltas: dict[str, torch.Tensor],
) -> Callable[[], list[Any]]:
    return combine_installers(install_joint, install_combo_add(model, combo, deltas))


def classify_role(delta_margin: float, delta_donor_logit: float, delta_comp_logit: float, new_top1: bool) -> str:
    if new_top1 and delta_comp_logit < -0.10 and delta_donor_logit > 0.10:
        return "boost_and_suppress_closure_candidate"
    if new_top1 and delta_comp_logit < -0.10:
        return "suppression_closure_candidate"
    if new_top1 and delta_donor_logit > 0.10:
        return "boost_closure_candidate"
    if delta_margin > 0.50 and delta_comp_logit < -0.10 and delta_donor_logit > 0.10:
        return "boost_and_suppress_margin_candidate"
    if delta_margin > 0.50 and delta_comp_logit < -0.10:
        return "suppression_margin_candidate"
    if delta_margin > 0.50 and delta_donor_logit > 0.10:
        return "boost_margin_candidate"
    if delta_margin > 0.10 and delta_comp_logit < -0.05:
        return "weak_suppression_candidate"
    if delta_margin > 0.10 and delta_donor_logit > 0.05:
        return "weak_boost_candidate"
    if delta_margin < -0.10:
        return "harmful_or_competitor_support"
    return "small_or_no_effect"


def audit_pair(
    model,
    tokenizer,
    device,
    model_name: str,
    target_site: str,
    pair: dict[str, Any],
    audit: dict[str, Any],
    combo: list[dict[str, Any]],
    scan_candidates: list[dict[str, Any]],
    top_k_vocab: int,
) -> list[dict[str, Any]]:
    intervention = audit["intervention"]
    direction_name = audit["direction"]
    donor, recipient = choose_donor_recipient(pair, direction_name)
    donor_ids = tokenizer.encode(prompt_for(donor), add_special_tokens=False)
    recipient_ids = tokenizer.encode(prompt_for(recipient), add_special_tokens=False)
    ctx = taxonomy_context(tokenizer, donor, recipient)
    donor_id = int(ctx["donor_id"])
    recipient_id = int(ctx["recipient_id"])
    combo_ids = [c["component_id"] for c in combo]
    scan_ids = [c["component_id"] for c in scan_candidates]
    all_sites = sorted(set(combo_ids + scan_ids))

    _meta, install_joint = prepare_joint_install(model, tokenizer, device, target_site, recipient, donor, recipient_ids, donor_ids, intervention)
    recipient_state = capture_state(model, device, recipient_ids, all_sites)
    donor_state = capture_state(model, device, donor_ids, all_sites)
    deltas = {site: donor_state["components"][site] - recipient_state["components"][site] for site in all_sites}
    install_combo = build_combo_installer(model, install_joint, combo, deltas)
    base_state = capture_state(model, device, recipient_ids, [], install_combo)
    base_logits = base_state["logits"]
    base_vocab = top_vocab_with_classes(base_logits, tokenizer, ctx, top_k_vocab)
    top = base_vocab[0]
    top_id = int(top["token_id"])
    if top_id == donor_id:
        return []

    base_donor_logit = float(base_logits[donor_id].item())
    base_comp_logit = float(base_logits[top_id].item())
    base_recipient_logit = float(base_logits[recipient_id].item())
    base_margin_comp = base_donor_logit - base_comp_logit
    base_margin_recipient = base_donor_logit - base_recipient_logit

    rows: list[dict[str, Any]] = []
    for cand in scan_candidates:
        site = cand["component_id"]
        candidate_delta = deltas[site]

        def install_candidate(site=site, candidate_delta=candidate_delta) -> list[Any]:
            return install_component_edit(model, site, add_delta=candidate_delta)

        installer = combine_installers(install_combo, install_candidate)
        state = capture_state(model, device, recipient_ids, [], installer)
        logits = state["logits"]
        vocab = top_vocab_with_classes(logits, tokenizer, ctx, top_k_vocab)
        new_top = vocab[0]
        donor_diag = logit_diag(logits, donor_id)
        recipient_diag = logit_diag(logits, recipient_id)
        new_donor_logit = float(logits[donor_id].item())
        new_comp_logit = float(logits[top_id].item())
        new_recipient_logit = float(logits[recipient_id].item())
        new_margin_comp = new_donor_logit - new_comp_logit
        delta_margin = new_margin_comp - base_margin_comp
        delta_donor = new_donor_logit - base_donor_logit
        delta_comp = new_comp_logit - base_comp_logit
        delta_recipient = new_recipient_logit - base_recipient_logit
        new_top_class = classify_token(tokenizer, int(new_top["token_id"]), new_top["token_text"], ctx)
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
                "combo_components": combo_ids,
                "candidate_component_id": site,
                "candidate_layer": cand["layer"],
                "candidate_component": cand["component"],
                "candidate_already_in_threshold_combo": cand["already_in_threshold_combo"],
                "base_competitor_token_id": top_id,
                "base_competitor_token_text": top["token_text"],
                "base_competitor_class": top["class"],
                "base_donor_rank": logit_diag(base_logits, donor_id)["target_rank"],
                "base_margin_donor_vs_competitor": base_margin_comp,
                "base_margin_donor_vs_recipient": base_margin_recipient,
                "delta_margin_donor_vs_competitor": delta_margin,
                "delta_donor_logit": delta_donor,
                "delta_competitor_logit": delta_comp,
                "delta_recipient_logit": delta_recipient,
                "new_margin_donor_vs_competitor": new_margin_comp,
                "new_margin_donor_vs_top": margin(logits, donor_id, int(new_top["token_id"])),
                "new_margin_donor_vs_recipient": new_donor_logit - new_recipient_logit,
                "new_donor_rank": donor_diag["target_rank"],
                "new_donor_top1": donor_diag["target_top1"],
                "new_recipient_rank": recipient_diag["target_rank"],
                "new_top_token_id": int(new_top["token_id"]),
                "new_top_token_text": new_top["token_text"],
                "new_top_token_class": new_top_class,
                "top_changed": int(new_top["token_id"]) != top_id,
                "candidate_delta_norm": float(torch.linalg.vector_norm(candidate_delta.float()).item()),
                "role_guess": classify_role(delta_margin, delta_donor, delta_comp, bool(donor_diag["target_top1"])),
                "base_top_vocab": base_vocab,
                "new_top_vocab": vocab,
            }
        )
    return rows


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["candidate_component_id"]].append(row)
    out = []
    for cid, vals in groups.items():
        n = len(vals)
        out.append(
            {
                "candidate_component_id": cid,
                "candidate_layer": vals[0]["candidate_layer"],
                "candidate_component": vals[0]["candidate_component"],
                "candidate_already_in_threshold_combo": vals[0]["candidate_already_in_threshold_combo"],
                "n": n,
                "mean_base_margin_donor_vs_competitor": safe_mean([v["base_margin_donor_vs_competitor"] for v in vals]),
                "mean_delta_margin_donor_vs_competitor": safe_mean([v["delta_margin_donor_vs_competitor"] for v in vals]),
                "mean_delta_donor_logit": safe_mean([v["delta_donor_logit"] for v in vals]),
                "mean_delta_competitor_logit": safe_mean([v["delta_competitor_logit"] for v in vals]),
                "mean_delta_recipient_logit": safe_mean([v["delta_recipient_logit"] for v in vals]),
                "new_donor_top1_rate": sum(1 for v in vals if v["new_donor_top1"]) / n,
                "mean_new_donor_rank": safe_mean([v["new_donor_rank"] for v in vals]),
                "top_changed_rate": sum(1 for v in vals if v["top_changed"]) / n,
                "base_competitor_class_counts": dict(Counter(v["base_competitor_class"] for v in vals)),
                "new_top_class_counts": dict(Counter(v["new_top_token_class"] for v in vals)),
                "role_guess_counts": dict(Counter(v["role_guess"] for v in vals)),
                "dominant_role_guess": Counter(v["role_guess"] for v in vals).most_common(1)[0][0],
            }
        )
    return sorted(
        out,
        key=lambda r: (
            r["new_donor_top1_rate"] or 0,
            r["mean_delta_margin_donor_vs_competitor"] or -999,
            -(r["mean_delta_competitor_logit"] or 999),
        ),
        reverse=True,
    )


def summarize_by_competitor_class(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["base_competitor_class"], row["candidate_component_id"])].append(row)
    out = []
    for (cls, cid), vals in groups.items():
        n = len(vals)
        out.append(
            {
                "base_competitor_class": cls,
                "candidate_component_id": cid,
                "n": n,
                "mean_delta_margin_donor_vs_competitor": safe_mean([v["delta_margin_donor_vs_competitor"] for v in vals]),
                "mean_delta_donor_logit": safe_mean([v["delta_donor_logit"] for v in vals]),
                "mean_delta_competitor_logit": safe_mean([v["delta_competitor_logit"] for v in vals]),
                "new_donor_top1_rate": sum(1 for v in vals if v["new_donor_top1"]) / n,
                "role_guess_counts": dict(Counter(v["role_guess"] for v in vals)),
            }
        )
    return sorted(
        out,
        key=lambda r: (
            r["base_competitor_class"],
            r["new_donor_top1_rate"] or 0,
            r["mean_delta_margin_donor_vs_competitor"] or -999,
        ),
        reverse=True,
    )


def run_model(args) -> dict[str, Any]:
    out_dir = OUT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    audit_payload = load_phase739_audits(args.model, args.phase739_round, args.top_audits)
    combo = load_phase741_ranked_candidates(args.model, args.phase741_round, args.top_candidates)
    pairs = select_conflict_pairs(args.max_pairs, args.include_extended_relations)
    log(
        f"{args.model}/{args.round_name}: pairs={len(pairs)} target={audit_payload['target_site']} "
        f"audits={len(audit_payload['audits'])} combo_candidates={len(combo)}"
    )
    model, tokenizer, device, attn_impl = load_model_bf16_eager(args.model)
    try:
        scan_candidates = build_scan_candidates(model, args.model, combo, args.include_combo_candidates)
        if args.max_scan_candidates and args.max_scan_candidates > 0:
            scan_candidates = scan_candidates[: args.max_scan_candidates]
        log(f"{args.model}: scan_candidates={len(scan_candidates)} {[c['component_id'] for c in scan_candidates]}")
        rows: list[dict[str, Any]] = []
        skipped_top1 = 0
        for pair_idx, pair in enumerate(pairs, 1):
            for audit in audit_payload["audits"]:
                before = len(rows)
                out = audit_pair(
                    model,
                    tokenizer,
                    device,
                    args.model,
                    audit_payload["target_site"],
                    pair,
                    audit,
                    combo,
                    scan_candidates,
                    args.top_k_vocab,
                )
                if not out:
                    skipped_top1 += 1
                rows.extend(out)
                if len(rows) - before == 0:
                    continue
            if pair_idx % args.log_every == 0 or pair_idx == len(pairs):
                log(f"{args.model}: suppression source scan {pair_idx}/{len(pairs)} pairs; rows={len(rows)} skipped_top1={skipped_top1}")
    finally:
        release_model(model)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary_rows = summarize_rows(rows)
    class_rows = summarize_by_competitor_class(rows)
    summary = {
        "phase": 744,
        "title": "Competitor Suppression Source Localization",
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
        "include_combo_candidates": args.include_combo_candidates,
        "audited_interventions": [{"label": intervention_label(a["intervention"]), "direction": a["direction"]} for a in audit_payload["audits"]],
        "threshold_combo_components": combo,
        "scan_candidates": scan_candidates,
        "skipped_cases_already_donor_top1": skipped_top1,
        "n_rows": len(rows),
        "candidate_summary": summary_rows,
        "competitor_class_summary": class_rows,
        "top_candidate_summary": summary_rows[:12],
        "strict_interpretation": "Candidate donor-recipient delta add localizes possible natural margin/suppression sources at whole-component granularity. A negative competitor-logit delta is stronger evidence for suppression than a margin gain caused only by donor boost.",
    }
    write_jsonl(out_dir / f"phase744_{args.model}_rows.jsonl", rows)
    write_json(out_dir / f"phase744_{args.model}_summary.json", summary)
    print(
        json.dumps(
            {
                "model": args.model,
                "round": args.round_name,
                "skipped_top1": skipped_top1,
                "top_candidate_summary": summary["top_candidate_summary"][:8],
                "top_class_summary": summary["competitor_class_summary"][:10],
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

    for model_index, model in enumerate(payload.get("models", [])):
        lane_z = (model_index - (len(payload.get("models", [])) - 1) / 2) * 10
        summary = payload["by_model"][model]
        model_node = f"{model}:model"
        comp_field = f"{model}:competitor_field"
        add_node({"id": model_node, "type": "model", "label": model, "model": model, "position": [-24, 0, lane_z], "role": "tested_model"})
        add_node({"id": comp_field, "type": "competition", "label": "competitor field", "model": model, "position": [0, 0, lane_z], "role": "readout_competition"})
        edges.append({"source": model_node, "target": comp_field, "relation": "has_competitor_field", "phase": 744})
        for row in summary.get("top_candidate_summary", [])[:8]:
            cid = row["candidate_component_id"]
            node = f"{model}:suppressor:{cid}"
            add_node(
                {
                    "id": node,
                    "type": "component",
                    "label": cid,
                    "model": model,
                    "position": [18, row.get("mean_delta_margin_donor_vs_competitor") or 0, lane_z],
                    "role": row.get("dominant_role_guess"),
                    "mean_delta_margin": row.get("mean_delta_margin_donor_vs_competitor"),
                    "mean_delta_competitor_logit": row.get("mean_delta_competitor_logit"),
                    "new_donor_top1_rate": row.get("new_donor_top1_rate"),
                }
            )
            edges.append({"source": node, "target": comp_field, "relation": "modulates_competition", "weight": row.get("mean_delta_margin_donor_vs_competitor"), "phase": 744})
    return {
        "schema_version": "atlas_graph_v1",
        "title": f"Phase 744 Competitor Suppression Source Localization ({round_name})",
        "model_info": {"model": "cross_model", "models": payload.get("models", []), "phase": 744, "round": round_name, "timestamp": payload.get("timestamp"), "evidence_type": payload.get("evidence_type")},
        "layout": {"x": "model -> competitor field -> source component", "y": "margin delta", "z": "model lane"},
        "graph": {"nodes": nodes, "edges": edges},
        "metrics": {"node_count": len(nodes), "edge_count": len(edges), "source_phase": 744},
        "source_files": [str(OUT_ROOT / round_name / "phase744_cross_model_summary.json")],
    }


def write_cross_summary(round_name: str) -> dict[str, Any]:
    out_dir = OUT_ROOT / round_name
    summaries = []
    for model in MODELS:
        path = out_dir / f"phase744_{model}_summary.json"
        if path.exists():
            summaries.append(json.loads(path.read_text(encoding="utf-8")))
    payload = {
        "phase": 744,
        "title": "Competitor Suppression Source Localization",
        "round": round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": [s["model"] for s in summaries],
        "status": "complete" if len(summaries) == len(MODELS) else "partial",
        "evidence_type": "whole-component donor-recipient delta add measured by donor-vs-current-competitor margin and competitor logit change",
        "by_model": {s["model"]: s for s in summaries},
    }
    write_json(out_dir / "phase744_cross_model_summary.json", payload)
    graph = build_atlas_graph(payload, round_name)
    write_json(out_dir / "phase744_atlas_graph.json", graph)
    lines = [
        f"# Phase 744 Competitor Suppression Source Localization ({round_name})",
        "",
        f"- Status: `{payload['status']}`",
        f"- Models: `{payload['models']}`",
        "- Evidence type: donor-recipient component delta add against the Phase743 current top competitor.",
        "",
        "| model | component | in topK | n | margin delta | donor logit delta | competitor logit delta | donor top1 | role counts |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for model, summary in payload["by_model"].items():
        for row in summary.get("top_candidate_summary", [])[:8]:
            lines.append(
                f"| {model} | {row['candidate_component_id']} | {int(bool(row.get('candidate_already_in_threshold_combo')))} | "
                f"{row['n']} | "
                f"{(row.get('mean_delta_margin_donor_vs_competitor') or 0):.3f} | "
                f"{(row.get('mean_delta_donor_logit') or 0):.3f} | "
                f"{(row.get('mean_delta_competitor_logit') or 0):.3f} | "
                f"{(row.get('new_donor_top1_rate') or 0):.3f} | "
                f"`{json.dumps(row.get('role_guess_counts') or {}, ensure_ascii=False)}` |"
            )
    lines.extend(
        [
            "",
            "## By Competitor Class",
            "",
            "| model | class | component | n | margin delta | donor delta | competitor delta | donor top1 | roles |",
            "|---|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for model, summary in payload["by_model"].items():
        for row in summary.get("competitor_class_summary", [])[:16]:
            lines.append(
                f"| {model} | {row['base_competitor_class']} | {row['candidate_component_id']} | {row['n']} | "
                f"{(row.get('mean_delta_margin_donor_vs_competitor') or 0):.3f} | "
                f"{(row.get('mean_delta_donor_logit') or 0):.3f} | "
                f"{(row.get('mean_delta_competitor_logit') or 0):.3f} | "
                f"{(row.get('new_donor_top1_rate') or 0):.3f} | "
                f"`{json.dumps(row.get('role_guess_counts') or {}, ensure_ascii=False)}` |"
            )
    lines.extend(
        [
            "",
            "## Strict Interpretation",
            "",
            "- A positive margin delta means the component can improve donor-vs-current-competitor competition when transplanted.",
            "- A negative competitor-logit delta is direct evidence of suppression; a positive donor-logit delta is boost-dominant rather than pure suppression.",
            "- This phase is still whole-component level and does not yet identify head/channel/neuron mechanisms.",
            "",
            f"Atlas graph: nodes={graph['metrics']['node_count']} edges={graph['metrics']['edge_count']}",
            "",
        ]
    )
    (out_dir / "phase744_cross_model_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"round": round_name, "status": payload["status"], "models": payload["models"], "atlas": graph["metrics"]}, ensure_ascii=False, indent=2), flush=True)
    return payload


def dry_run(args) -> None:
    payload = {"round": args.round_name, "pairs": len(select_conflict_pairs(args.max_pairs, args.include_extended_relations)), "models": {}}
    for model in MODELS:
        audits = load_phase739_audits(model, args.phase739_round, args.top_audits)
        combo = load_phase741_ranked_candidates(model, args.phase741_round, args.top_candidates)
        payload["models"][model] = {
            "target_site": audits["target_site"],
            "audits": [{"label": intervention_label(a["intervention"]), "direction": a["direction"]} for a in audits["audits"]],
            "threshold_combo_components": combo,
            "scan_layers": SCAN_LAYERS[model],
            "include_combo_candidates": args.include_combo_candidates,
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
    parser.add_argument("--max-scan-candidates", type=int, default=0)
    parser.add_argument("--include-combo-candidates", action="store_true")
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
