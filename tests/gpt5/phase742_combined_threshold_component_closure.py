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

from model_utils import release_model  # noqa: E402
from phase722_functional_head_atlas_causal_ablation import logit_diag, write_json, write_jsonl  # noqa: E402
from phase723_apple_fruit_attribute_micro_atlas import prompt_for  # noqa: E402
from phase735_source_restricted_writer_validation import MODELS, load_model_bf16_eager, safe_mean  # noqa: E402
from phase736_source_replacement_generation_closure import select_conflict_pairs  # noqa: E402
from phase737_writer_rewriter_joint_replacement import intervention_label  # noqa: E402
from phase739_readout_threshold_closure_boundary import (  # noqa: E402
    choose_donor_recipient,
    get_unembed,
    normalized_direction,
    prepare_joint_install,
    top_token_info,
)
from phase740_natural_readout_boost_source_backtrace import (  # noqa: E402
    alpha_needed,
    first_token_id,
    load_phase739_audits,
    load_phase739_row_index,
    projection,
)
from phase741_threshold_candidate_causal_validation import (  # noqa: E402
    OUT_ROOT as PHASE741_ROOT,
    capture_state,
    combine_installers,
    install_component_edit,
    parse_component_site,
)


OUT_ROOT = Path("results/glm5_phase742_combined_threshold_component_closure")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_phase741_ranked_candidates(model_name: str, round_name: str, max_candidates: int) -> list[dict[str, Any]]:
    path = PHASE741_ROOT / round_name / f"phase741_{model_name}_summary.json"
    if not path.exists():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    by_cid: dict[str, dict[str, Any]] = {}
    meta = {c["component_id"]: c for c in data.get("candidate_components", [])}
    for row in data.get("condition_summary", []):
        if row.get("condition") != "joint_add_donor_delta":
            continue
        cid = row["candidate_component_id"]
        layer, component = parse_component_site(cid)
        by_cid[cid] = {
            "component_id": cid,
            "layer": layer,
            "component": component,
            "phase741_joint_add_effect": row.get("mean_effect_vs_joint_fraction"),
            "phase741_joint_add_fraction": row.get("mean_fraction_of_threshold"),
            "phase741_target_top1_rate": row.get("target_top1_rate"),
            "phase740_mean_patched_fraction": (meta.get(cid) or {}).get("phase740_mean_patched_fraction"),
            "phase740_mean_donor_fraction": (meta.get(cid) or {}).get("phase740_mean_donor_fraction"),
        }
    return sorted(by_cid.values(), key=lambda r: r.get("phase741_joint_add_effect") or -999, reverse=True)[:max_candidates]


def install_combo_add(model, combo: list[dict[str, Any]], deltas: dict[str, torch.Tensor]) -> Callable[[], list[Any]]:
    def install() -> list[Any]:
        handles: list[Any] = []
        for c in combo:
            handles.extend(install_component_edit(model, c["component_id"], add_delta=deltas[c["component_id"]]))
        return handles

    return install


def install_combo_replace(model, combo: list[dict[str, Any]], replacements: dict[str, torch.Tensor]) -> Callable[[], list[Any]]:
    def install() -> list[Any]:
        handles: list[Any] = []
        for c in combo:
            handles.extend(install_component_edit(model, c["component_id"], replace_vec=replacements[c["component_id"]]))
        return handles

    return install


def margin_vs_top(logits: torch.Tensor, target_id: int, top_id: int) -> float:
    return float((logits[int(target_id)] - logits[int(top_id)]).item())


def make_row(
    model_name: str,
    target_site: str,
    pair: dict[str, Any],
    direction_name: str,
    intervention: dict[str, Any],
    donor: dict[str, Any],
    recipient: dict[str, Any],
    donor_id: int,
    recipient_id: int,
    threshold: float | None,
    d: torch.Tensor,
    recipient_final: torch.Tensor,
    joint_final: torch.Tensor,
    donor_final: torch.Tensor,
    state: dict[str, Any],
    tokenizer,
    condition: str,
    combo: list[dict[str, Any]],
) -> dict[str, Any]:
    logits = state["logits"]
    top = top_token_info(logits, tokenizer)
    target_diag = logit_diag(logits, donor_id)
    recipient_diag = logit_diag(logits, recipient_id)
    final = state["final_norm_output"]
    final_delta = projection(final - recipient_final, d)
    effect_vs_joint = projection(final - joint_final, d)
    effect_vs_donor = projection(final - donor_final, d)
    return {
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
        "source_component_id": (intervention.get("source_spec") or {}).get("component_id"),
        "source_group": (intervention.get("source_spec") or {}).get("source_group"),
        "mlp_components": [m["component_id"] for m in intervention.get("mlp_specs") or []],
        "condition": condition,
        "combo_k": len(combo),
        "combo_components": [c["component_id"] for c in combo],
        "combo_phase741_joint_add_effect_sum": sum(float(c.get("phase741_joint_add_effect") or 0.0) for c in combo),
        "threshold_used": threshold,
        "final_delta_proj": final_delta,
        "fraction_of_threshold": (final_delta / threshold) if threshold else None,
        "effect_vs_joint_proj": effect_vs_joint,
        "effect_vs_joint_fraction": (effect_vs_joint / threshold) if threshold else None,
        "effect_vs_donor_proj": effect_vs_donor,
        "effect_vs_donor_fraction": (effect_vs_donor / threshold) if threshold else None,
        "target_logit": target_diag["target_logit"],
        "target_logprob": target_diag["target_logprob"],
        "target_rank": target_diag["target_rank"],
        "target_top1": target_diag["target_top1"],
        "recipient_answer_logit": recipient_diag["target_logit"],
        "top_token_id": int(top["token_id"]),
        "top_token_text": top["token_text"],
        "margin_donor_vs_top": margin_vs_top(logits, donor_id, int(top["token_id"])),
    }


def audit_pair(
    model,
    tokenizer,
    device,
    model_name: str,
    target_site: str,
    pair: dict[str, Any],
    audit: dict[str, Any],
    candidates: list[dict[str, Any]],
    phase739_index: dict[tuple[str, str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    intervention = audit["intervention"]
    direction_name = audit["direction"]
    donor, recipient = choose_donor_recipient(pair, direction_name)
    donor_ids = tokenizer.encode(prompt_for(donor), add_special_tokens=False)
    recipient_ids = tokenizer.encode(prompt_for(recipient), add_special_tokens=False)
    candidate_sites = [c["component_id"] for c in candidates]
    _meta, install_joint = prepare_joint_install(model, tokenizer, device, target_site, recipient, donor, recipient_ids, donor_ids, intervention)

    recipient_state = capture_state(model, device, recipient_ids, candidate_sites)
    donor_state = capture_state(model, device, donor_ids, candidate_sites)
    joint_state = capture_state(model, device, recipient_ids, candidate_sites, install_joint)

    unembed = get_unembed(model)
    donor_id = first_token_id(tokenizer, donor["answer"])
    recipient_id = first_token_id(tokenizer, recipient["answer"])
    joint_logits = joint_state["logits"]
    joint_top = top_token_info(joint_logits, tokenizer)
    top_id = int(joint_top["token_id"])
    d = normalized_direction(unembed, donor_id, top_id)
    if d is None:
        d = torch.zeros_like(unembed[donor_id])
    alpha_star = alpha_needed(joint_logits, unembed, donor_id, top_id, d)
    phase739_row = phase739_index.get((pair["pair_id"], intervention_label(intervention), direction_name), {})
    threshold = phase739_row.get("first_alpha_donor_vocab_top")
    if threshold is None:
        threshold = alpha_star

    recipient_final = recipient_state["final_norm_output"]
    joint_final = joint_state["final_norm_output"]
    donor_final = donor_state["final_norm_output"]
    deltas = {site: donor_state["components"][site] - recipient_state["components"][site] for site in candidate_sites}
    replacements = {site: recipient_state["components"][site] for site in candidate_sites}

    rows: list[dict[str, Any]] = []
    base_state = capture_state(model, device, recipient_ids, [], install_joint)
    rows.append(
        make_row(
            model_name,
            target_site,
            pair,
            direction_name,
            intervention,
            donor,
            recipient,
            donor_id,
            recipient_id,
            threshold,
            d,
            recipient_final,
            joint_final,
            donor_final,
            base_state,
            tokenizer,
            "joint_base",
            [],
        )
    )

    for k in range(1, len(candidates) + 1):
        combo = candidates[:k]
        joint_add = combine_installers(install_joint, install_combo_add(model, combo, deltas))
        recipient_add = install_combo_add(model, combo, deltas)
        donor_erase = install_combo_replace(model, combo, replacements)
        for condition, ids, installer in [
            (f"joint_add_top{k}", recipient_ids, joint_add),
            (f"recipient_add_top{k}", recipient_ids, recipient_add),
            (f"donor_erase_top{k}", donor_ids, donor_erase),
        ]:
            state = capture_state(model, device, ids, [], installer)
            rows.append(
                make_row(
                    model_name,
                    target_site,
                    pair,
                    direction_name,
                    intervention,
                    donor,
                    recipient,
                    donor_id,
                    recipient_id,
                    threshold,
                    d,
                    recipient_final,
                    joint_final,
                    donor_final,
                    state,
                    tokenizer,
                    condition,
                    combo,
                )
            )
    return rows


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["condition"], row["combo_k"])].append(row)
    out = []
    for (condition, combo_k), vals in groups.items():
        n = len(vals)
        out.append(
            {
                "condition": condition,
                "combo_k": combo_k,
                "combo_components": vals[0].get("combo_components") or [],
                "n": n,
                "mean_fraction_of_threshold": safe_mean([v["fraction_of_threshold"] for v in vals]),
                "mean_effect_vs_joint_fraction": safe_mean([v["effect_vs_joint_fraction"] for v in vals]),
                "mean_effect_vs_donor_fraction": safe_mean([v["effect_vs_donor_fraction"] for v in vals]),
                "mean_target_rank": safe_mean([v["target_rank"] for v in vals]),
                "target_top1_rate": sum(1 for v in vals if v["target_top1"]) / n,
                "mean_margin_donor_vs_top": safe_mean([v["margin_donor_vs_top"] for v in vals]),
                "top_token_counts": dict(Counter(v["top_token_text"] for v in vals)),
                "expected_add_effect_sum": vals[0].get("combo_phase741_joint_add_effect_sum"),
            }
        )
    return sorted(out, key=lambda r: (r["combo_k"], r["condition"]))


def run_model(args) -> dict[str, Any]:
    out_dir = OUT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    audit_payload = load_phase739_audits(args.model, args.phase739_round, args.top_audits)
    candidates = load_phase741_ranked_candidates(args.model, args.phase741_round, args.top_candidates)
    pairs = select_conflict_pairs(args.max_pairs, args.include_extended_relations)
    phase739_index = load_phase739_row_index(args.model, args.phase739_round)
    log(
        f"{args.model}/{args.round_name}: pairs={len(pairs)} target={audit_payload['target_site']} "
        f"audits={len(audit_payload['audits'])} combo_candidates={len(candidates)}"
    )
    model, tokenizer, device, _attn_impl = load_model_bf16_eager(args.model)
    try:
        rows: list[dict[str, Any]] = []
        for pair_idx, pair in enumerate(pairs, 1):
            for audit in audit_payload["audits"]:
                rows.extend(audit_pair(model, tokenizer, device, args.model, audit_payload["target_site"], pair, audit, candidates, phase739_index))
            if pair_idx % args.log_every == 0 or pair_idx == len(pairs):
                log(f"{args.model}: combined closure {pair_idx}/{len(pairs)} pairs; rows={len(rows)}")
    finally:
        release_model(model)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = {
        "phase": 742,
        "title": "Combined Threshold Component Closure",
        "model": args.model,
        "round": args.round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": "eager",
        "quantization": "off",
        "dtype": "bfloat16",
        "phase739_round": args.phase739_round,
        "phase741_round": args.phase741_round,
        "target_site": audit_payload["target_site"],
        "max_pairs": args.max_pairs,
        "top_audits": args.top_audits,
        "top_candidates": args.top_candidates,
        "audited_interventions": [{"label": intervention_label(a["intervention"]), "direction": a["direction"]} for a in audit_payload["audits"]],
        "ranked_candidate_components": candidates,
        "n_rows": len(rows),
        "condition_summary": summarize_rows(rows),
        "strict_interpretation": "Cumulative component edits test whether validated threshold-source components are sufficient for closure; whole-component cumulative edits can still be off-manifold.",
    }
    write_jsonl(out_dir / f"phase742_{args.model}_rows.jsonl", rows)
    write_json(out_dir / f"phase742_{args.model}_summary.json", summary)
    print(json.dumps({"model": args.model, "round": args.round_name, "target_site": summary["target_site"], "ranked_candidates": candidates, "condition_summary": summary["condition_summary"]}, ensure_ascii=False, indent=2), flush=True)
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
        target_node = f"{model}:threshold_closure"
        add_node({"id": f"{model}:model", "type": "model", "label": model, "model": model, "position": [-26, 0, lane_z], "role": "tested_model"})
        add_node({"id": target_node, "type": "readout", "label": "combined threshold closure", "model": model, "position": [24, 0, lane_z], "role": "closure_target"})
        edges.append({"source": f"{model}:model", "target": target_node, "relation": "tests_combined_closure", "phase": 742})
        for row in summary.get("condition_summary", []):
            if not row["condition"].startswith("joint_add_top"):
                continue
            combo_node = f"{model}:combo:{row['condition']}"
            add_node(
                {
                    "id": combo_node,
                    "type": "component_combo",
                    "label": row["condition"],
                    "model": model,
                    "role": "cumulative_threshold_source",
                    "combo_components": row.get("combo_components"),
                    "position": [0, row.get("mean_fraction_of_threshold") or 0, lane_z],
                }
            )
            edges.append({"source": combo_node, "target": target_node, "relation": "joint_add_effect", "weight": row.get("mean_effect_vs_joint_fraction"), "top1_rate": row.get("target_top1_rate"), "phase": 742})
    return {
        "schema_version": "atlas_graph_v1",
        "title": f"Phase 742 Combined Threshold Component Closure ({round_name})",
        "model_info": {"model": "cross_model", "models": payload.get("models", []), "phase": 742, "round": round_name, "timestamp": payload.get("timestamp"), "evidence_type": payload.get("evidence_type")},
        "layout": {"x": "model -> component combo -> threshold closure", "y": "threshold fraction", "z": "model lane"},
        "graph": {"nodes": nodes, "edges": edges},
        "metrics": {"node_count": len(nodes), "edge_count": len(edges), "source_phase": 742},
        "source_files": [str(OUT_ROOT / round_name / "phase742_cross_model_summary.json")],
    }


def write_cross_summary(round_name: str) -> dict[str, Any]:
    out_dir = OUT_ROOT / round_name
    summaries = []
    for model in MODELS:
        path = out_dir / f"phase742_{model}_summary.json"
        if path.exists():
            summaries.append(json.loads(path.read_text(encoding="utf-8")))
    payload = {
        "phase": 742,
        "title": "Combined Threshold Component Closure",
        "round": round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": [s["model"] for s in summaries],
        "status": "complete" if len(summaries) == len(MODELS) else "partial",
        "evidence_type": "cumulative donor-recipient component delta add tested against final readout threshold closure",
        "by_model": {s["model"]: s for s in summaries},
    }
    write_json(out_dir / "phase742_cross_model_summary.json", payload)
    graph = build_atlas_graph(payload, round_name)
    write_json(out_dir / "phase742_atlas_graph.json", graph)
    lines = [
        f"# Phase 742 Combined Threshold Component Closure ({round_name})",
        "",
        f"- Status: `{payload['status']}`",
        f"- Models: `{payload['models']}`",
        "- Evidence type: cumulative component donor-delta add measured by threshold fraction and target top1 rate.",
        "",
        "| model | condition | components | fraction | joint add effect | target top1 rate | margin donor vs top |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for model, summary in payload["by_model"].items():
        for row in summary.get("condition_summary", []):
            if not (row["condition"] == "joint_base" or row["condition"].startswith("joint_add_top")):
                continue
            lines.append(
                f"| {model} | {row['condition']} | {','.join(row.get('combo_components') or [])} | "
                f"{(row.get('mean_fraction_of_threshold') or 0):.3f} | "
                f"{(row.get('mean_effect_vs_joint_fraction') or 0):.3f} | "
                f"{(row.get('target_top1_rate') or 0):.3f} | "
                f"{(row.get('mean_margin_donor_vs_top') or 0):.3f} |"
            )
    lines.extend(
        [
            "",
            "## Strict Interpretation",
            "",
            "- If joint_add_topK reaches fraction near or above 1, the validated components are close to sufficient for readout closure.",
            "- If it remains below 1, the missing mechanism is probably competitor/format suppression or final readout geometry, not merely these visible components.",
            "- Whole-component cumulative edits remain coarse and can be off-manifold.",
            "",
            f"Atlas graph: nodes={graph['metrics']['node_count']} edges={graph['metrics']['edge_count']}",
            "",
        ]
    )
    (out_dir / "phase742_cross_model_summary.md").write_text("\n".join(lines), encoding="utf-8")
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
