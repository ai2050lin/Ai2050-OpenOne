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
    OUT_ROOT as PHASE740_ROOT,
    alpha_needed,
    first_token_id,
    load_phase739_audits,
    load_phase739_row_index,
    projection,
)


OUT_ROOT = Path("results/glm5_phase741_threshold_candidate_causal_validation")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


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


def parse_component_site(site: str) -> tuple[int, str]:
    prefix, component = site.split(":", 1)
    if not prefix.startswith("L"):
        raise ValueError(f"bad component site: {site}")
    if component not in {"attn_out", "mlp_out"}:
        raise ValueError(f"unsupported component site: {site}")
    return int(prefix[1:]), component


def module_for_component(model, site: str):
    layer_idx, component = parse_component_site(site)
    layer = get_layers(model)[layer_idx]
    if component == "attn_out":
        mod = get_attn(layer)
    else:
        mod = get_mlp(layer)
    if mod is None:
        raise ValueError(f"module not found for {site}")
    return mod


def replace_or_add_output(output: Any, vec: torch.Tensor | None = None, delta: torch.Tensor | None = None) -> Any:
    y = extract_tensor(output)
    y_new = y.clone()
    if vec is not None:
        y_new[0, -1, :] = vec.to(device=y_new.device, dtype=y_new.dtype)
    if delta is not None:
        y_new[0, -1, :] = y_new[0, -1, :] + delta.to(device=y_new.device, dtype=y_new.dtype)
    if isinstance(output, tuple):
        return (y_new,) + output[1:]
    return y_new


def install_component_edit(model, site: str, *, replace_vec: torch.Tensor | None = None, add_delta: torch.Tensor | None = None):
    module = module_for_component(model, site)

    def hook(_module, _inputs, output):
        return replace_or_add_output(output, vec=replace_vec, delta=add_delta)

    return [module.register_forward_hook(hook)]


def combine_installers(*installers: Callable[[], list[Any]] | None) -> Callable[[], list[Any]]:
    def install() -> list[Any]:
        handles: list[Any] = []
        for installer in installers:
            if installer is not None:
                handles.extend(installer())
        return handles

    return install


def capture_state(
    model,
    device,
    ids: list[int],
    candidate_sites: list[str] | None = None,
    install_hooks: Callable[[], list[Any]] | None = None,
) -> dict[str, Any]:
    handles = install_hooks() if install_hooks else []
    captured: dict[str, Any] = {"components": {}, "final_norm_input": None, "final_norm_output": None}
    for site in candidate_sites or []:
        module = module_for_component(model, site)

        def hook(_module, _inputs, output, site=site):
            y = extract_tensor(output)
            captured["components"][site] = y[0, -1].detach().float().cpu()

        handles.append(module.register_forward_hook(hook))

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


def load_phase740_candidates(model_name: str, round_name: str, top_candidates: int) -> list[dict[str, Any]]:
    path = PHASE740_ROOT / round_name / f"phase740_{model_name}_summary.json"
    if not path.exists():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    out = []
    seen: set[str] = set()
    for row in data.get("top_component_sources", []):
        cid = row.get("component_id")
        if not cid or cid in seen:
            continue
        try:
            layer, component = parse_component_site(cid)
        except ValueError:
            continue
        out.append(
            {
                "component_id": cid,
                "layer": layer,
                "component": component,
                "phase740_mean_patched_fraction": row.get("mean_patched_fraction_of_threshold"),
                "phase740_mean_donor_fraction": row.get("mean_donor_fraction_of_threshold"),
                "phase740_role_guess": row.get("role_guess"),
            }
        )
        seen.add(cid)
        if len(out) >= top_candidates:
            break
    return out


def margin_vs_top(logits: torch.Tensor, target_id: int, top_id: int) -> float:
    return float((logits[int(target_id)] - logits[int(top_id)]).item())


def condition_row(
    model_name: str,
    target_site: str,
    pair: dict[str, Any],
    direction_name: str,
    intervention: dict[str, Any],
    candidate: dict[str, Any],
    condition: str,
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
        "candidate_component_id": candidate["component_id"],
        "candidate_layer": candidate["layer"],
        "candidate_component": candidate["component"],
        "phase740_mean_patched_fraction": candidate.get("phase740_mean_patched_fraction"),
        "phase740_mean_donor_fraction": candidate.get("phase740_mean_donor_fraction"),
        "condition": condition,
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

    rows: list[dict[str, Any]] = []
    recipient_final = recipient_state["final_norm_output"]
    joint_final = joint_state["final_norm_output"]
    donor_final = donor_state["final_norm_output"]

    for candidate in candidates:
        site = candidate["component_id"]
        rec_vec = recipient_state["components"][site]
        don_vec = donor_state["components"][site]
        delta = don_vec - rec_vec

        condition_installers: dict[str, Callable[[], list[Any]]] = {
            "recipient_add_donor_delta": lambda site=site, delta=delta: install_component_edit(model, site, add_delta=delta),
            "joint_base": install_joint,
            "joint_add_donor_delta": combine_installers(install_joint, lambda site=site, delta=delta: install_component_edit(model, site, add_delta=delta)),
            "joint_replace_with_donor_component": combine_installers(install_joint, lambda site=site, don_vec=don_vec: install_component_edit(model, site, replace_vec=don_vec)),
            "joint_erase_to_recipient_component": combine_installers(install_joint, lambda site=site, rec_vec=rec_vec: install_component_edit(model, site, replace_vec=rec_vec)),
            "donor_erase_to_recipient_component": lambda site=site, rec_vec=rec_vec: install_component_edit(model, site, replace_vec=rec_vec),
        }

        for condition, installer in condition_installers.items():
            ids = donor_ids if condition == "donor_erase_to_recipient_component" else recipient_ids
            state = capture_state(model, device, ids, [], installer)
            row = condition_row(
                model_name,
                target_site,
                pair,
                direction_name,
                intervention,
                candidate,
                condition,
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
            )
            row.update(
                {
                    "joint_base_fraction": projection(joint_final - recipient_final, d) / threshold if threshold else None,
                    "donor_base_fraction": projection(donor_final - recipient_final, d) / threshold if threshold else None,
                    "candidate_donor_minus_recipient_proj": projection(delta, d),
                    "alpha_star_vocab_top": alpha_star,
                    "phase739_first_alpha_donor_vocab_top": phase739_row.get("first_alpha_donor_vocab_top"),
                    "joint_vocab_top_token_text": joint_top["token_text"],
                    "joint_margin_donor_vs_vocab_top": margin_vs_top(joint_logits, donor_id, top_id),
                }
            )
            rows.append(row)
    return rows


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["candidate_component_id"], row["condition"])].append(row)
    out = []
    for (cid, condition), vals in groups.items():
        n = len(vals)
        out.append(
            {
                "candidate_component_id": cid,
                "condition": condition,
                "n": n,
                "candidate_component": vals[0]["candidate_component"],
                "candidate_layer": vals[0]["candidate_layer"],
                "mean_fraction_of_threshold": safe_mean([v["fraction_of_threshold"] for v in vals]),
                "mean_effect_vs_joint_fraction": safe_mean([v["effect_vs_joint_fraction"] for v in vals]),
                "mean_effect_vs_donor_fraction": safe_mean([v["effect_vs_donor_fraction"] for v in vals]),
                "mean_target_rank": safe_mean([v["target_rank"] for v in vals]),
                "target_top1_rate": sum(1 for v in vals if v["target_top1"]) / n,
                "mean_margin_donor_vs_top": safe_mean([v["margin_donor_vs_top"] for v in vals]),
                "top_token_counts": dict(Counter(v["top_token_text"] for v in vals)),
                "phase740_mean_patched_fraction": vals[0].get("phase740_mean_patched_fraction"),
                "phase740_mean_donor_fraction": vals[0].get("phase740_mean_donor_fraction"),
                "role_guess": role_guess(condition, vals),
            }
        )
    return sorted(out, key=lambda r: (r["candidate_component_id"], r["condition"]))


def role_guess(condition: str, vals: list[dict[str, Any]]) -> str:
    effect_joint = safe_mean([v["effect_vs_joint_fraction"] for v in vals])
    effect_donor = safe_mean([v["effect_vs_donor_fraction"] for v in vals])
    frac = safe_mean([v["fraction_of_threshold"] for v in vals])
    if condition in {"joint_add_donor_delta", "joint_replace_with_donor_component"}:
        if effect_joint is not None and effect_joint > 0.05:
            return "causal_boost_candidate"
        if effect_joint is not None and effect_joint > 0.005:
            return "weak_boost_candidate"
        return "no_material_boost"
    if condition == "joint_erase_to_recipient_component":
        if effect_joint is not None and effect_joint < -0.05:
            return "joint_path_necessary_candidate"
        if effect_joint is not None and effect_joint < -0.005:
            return "weak_joint_path_necessity"
        return "not_necessary_in_joint_path"
    if condition == "donor_erase_to_recipient_component":
        if effect_donor is not None and effect_donor < -0.05:
            return "donor_path_necessary_candidate"
        if effect_donor is not None and effect_donor < -0.005:
            return "weak_donor_path_necessity"
        return "not_necessary_in_donor_path"
    if frac is not None and frac > 0.05:
        return "standalone_boost_candidate"
    return "baseline_or_small_effect"


def run_model(args) -> dict[str, Any]:
    out_dir = OUT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    audit_payload = load_phase739_audits(args.model, args.phase739_round, args.top_audits)
    candidates = load_phase740_candidates(args.model, args.phase740_round, args.top_candidates)
    pairs = select_conflict_pairs(args.max_pairs, args.include_extended_relations)
    phase739_index = load_phase739_row_index(args.model, args.phase739_round)
    log(
        f"{args.model}/{args.round_name}: pairs={len(pairs)} target={audit_payload['target_site']} "
        f"audits={len(audit_payload['audits'])} candidates={len(candidates)}"
    )
    model, tokenizer, device, _attn_impl = load_model_bf16_eager(args.model)
    try:
        rows: list[dict[str, Any]] = []
        for pair_idx, pair in enumerate(pairs, 1):
            for audit in audit_payload["audits"]:
                rows.extend(audit_pair(model, tokenizer, device, args.model, audit_payload["target_site"], pair, audit, candidates, phase739_index))
            if pair_idx % args.log_every == 0 or pair_idx == len(pairs):
                log(f"{args.model}: causal validation {pair_idx}/{len(pairs)} pairs; rows={len(rows)}")
    finally:
        release_model(model)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = {
        "phase": 741,
        "title": "Threshold Candidate Causal Validation",
        "model": args.model,
        "round": args.round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": "eager",
        "quantization": "off",
        "dtype": "bfloat16",
        "phase739_round": args.phase739_round,
        "phase740_round": args.phase740_round,
        "target_site": audit_payload["target_site"],
        "max_pairs": args.max_pairs,
        "top_audits": args.top_audits,
        "top_candidates": args.top_candidates,
        "audited_interventions": [{"label": intervention_label(a["intervention"]), "direction": a["direction"]} for a in audit_payload["audits"]],
        "candidate_components": candidates,
        "n_rows": len(rows),
        "condition_summary": summarize_rows(rows),
        "strict_interpretation": "Component transplant/erasure is causal at component-output granularity, but whole-component edits can be off-manifold and do not identify individual neurons.",
    }
    write_jsonl(out_dir / f"phase741_{args.model}_rows.jsonl", rows)
    write_json(out_dir / f"phase741_{args.model}_summary.json", summary)
    print(json.dumps({"model": args.model, "round": args.round_name, "target_site": summary["target_site"], "condition_summary": summary["condition_summary"][:18]}, ensure_ascii=False, indent=2), flush=True)
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
        target_node = f"{model}:final_readout"
        add_node({"id": model_node, "type": "model", "label": model, "model": model, "position": [-24, 0, lane_z], "role": "tested_model"})
        add_node({"id": target_node, "type": "readout", "label": "final readout threshold", "model": model, "position": [22, 0, lane_z], "role": "threshold_output"})
        edges.append({"source": model_node, "target": target_node, "relation": "audits", "phase": 741})
        for comp in summary.get("candidate_components", []):
            comp_node = f"{model}:component:{comp['component_id']}"
            add_node(
                {
                    "id": comp_node,
                    "type": "component",
                    "label": comp["component_id"],
                    "model": model,
                    "role": "causal_candidate",
                    "phase740_patched_fraction": comp.get("phase740_mean_patched_fraction"),
                    "position": [0, 0, lane_z],
                }
            )
            for row in summary.get("condition_summary", []):
                if row["candidate_component_id"] != comp["component_id"]:
                    continue
                if row["condition"] not in {"joint_add_donor_delta", "joint_erase_to_recipient_component", "donor_erase_to_recipient_component"}:
                    continue
                edges.append(
                    {
                        "source": comp_node,
                        "target": target_node,
                        "relation": row["condition"],
                        "weight": row.get("mean_effect_vs_joint_fraction")
                        if row["condition"] != "donor_erase_to_recipient_component"
                        else row.get("mean_effect_vs_donor_fraction"),
                        "role_guess": row.get("role_guess"),
                        "phase": 741,
                    }
                )
    return {
        "schema_version": "atlas_graph_v1",
        "title": f"Phase 741 Threshold Candidate Causal Validation ({round_name})",
        "model_info": {"model": "cross_model", "models": payload.get("models", []), "phase": 741, "round": round_name, "timestamp": payload.get("timestamp"), "evidence_type": payload.get("evidence_type")},
        "layout": {"x": "model -> candidate component -> final readout threshold", "y": "causal effect", "z": "model lane"},
        "graph": {"nodes": nodes, "edges": edges},
        "metrics": {"node_count": len(nodes), "edge_count": len(edges), "source_phase": 741},
        "source_files": [str(OUT_ROOT / round_name / "phase741_cross_model_summary.json")],
    }


def write_cross_summary(round_name: str) -> dict[str, Any]:
    out_dir = OUT_ROOT / round_name
    summaries = []
    for model in MODELS:
        path = out_dir / f"phase741_{model}_summary.json"
        if path.exists():
            summaries.append(json.loads(path.read_text(encoding="utf-8")))
    payload = {
        "phase": 741,
        "title": "Threshold Candidate Causal Validation",
        "round": round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": [s["model"] for s in summaries],
        "status": "complete" if len(summaries) == len(MODELS) else "partial",
        "evidence_type": "component-output transplant and erasure causal validation against final readout threshold fractions",
        "by_model": {s["model"]: s for s in summaries},
    }
    write_json(out_dir / "phase741_cross_model_summary.json", payload)
    graph = build_atlas_graph(payload, round_name)
    write_json(out_dir / "phase741_atlas_graph.json", graph)
    lines = [
        f"# Phase 741 Threshold Candidate Causal Validation ({round_name})",
        "",
        f"- Status: `{payload['status']}`",
        f"- Models: `{payload['models']}`",
        "- Evidence type: component-output transplant and erasure measured by final readout threshold fraction.",
        "",
        "| model | component | joint add effect | joint erase effect | donor erase effect | role |",
        "|---|---|---:|---:|---:|---|",
    ]
    for model, summary in payload["by_model"].items():
        by_component: dict[str, dict[str, Any]] = defaultdict(dict)
        for row in summary.get("condition_summary", []):
            by_component[row["candidate_component_id"]][row["condition"]] = row
        for cid, conds in by_component.items():
            add = conds.get("joint_add_donor_delta", {})
            erase = conds.get("joint_erase_to_recipient_component", {})
            donor_erase = conds.get("donor_erase_to_recipient_component", {})
            role = add.get("role_guess") or erase.get("role_guess") or donor_erase.get("role_guess")
            lines.append(
                f"| {model} | {cid} | "
                f"{(add.get('mean_effect_vs_joint_fraction') or 0):.3f} | "
                f"{(erase.get('mean_effect_vs_joint_fraction') or 0):.3f} | "
                f"{(donor_erase.get('mean_effect_vs_donor_fraction') or 0):.3f} | "
                f"{role} |"
            )
    lines.extend(
        [
            "",
            "## Strict Interpretation",
            "",
            "- Positive joint add effect means the donor-recipient component delta can push the final readout direction.",
            "- Negative joint erase or donor erase effect means the component is necessary at this coarse output granularity.",
            "- Whole-component edits are stronger than neuron-level proof and can be off-manifold.",
            "",
            f"Atlas graph: nodes={graph['metrics']['node_count']} edges={graph['metrics']['edge_count']}",
            "",
        ]
    )
    (out_dir / "phase741_cross_model_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"round": round_name, "status": payload["status"], "models": payload["models"], "atlas": graph["metrics"]}, ensure_ascii=False, indent=2), flush=True)
    return payload


def dry_run(args) -> None:
    payload = {"round": args.round_name, "pairs": len(select_conflict_pairs(args.max_pairs, args.include_extended_relations)), "models": {}}
    for model in MODELS:
        audits = load_phase739_audits(model, args.phase739_round, args.top_audits)
        candidates = load_phase740_candidates(model, args.phase740_round, args.top_candidates)
        payload["models"][model] = {
            "target_site": audits["target_site"],
            "audits": [{"label": intervention_label(a["intervention"]), "direction": a["direction"]} for a in audits["audits"]],
            "candidates": candidates,
        }
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="main")
    parser.add_argument("--phase739-round", default="confirm")
    parser.add_argument("--phase740-round", default="confirm")
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
