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
from typing import Any

import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))

from model_utils import get_layers, load_model, release_model  # noqa: E402
from phase722_functional_head_atlas_causal_ablation import (  # noqa: E402
    install_zero_head_ablation,
    logit_diag,
    target_token_ids,
    write_json,
    write_jsonl,
)
from phase723_apple_fruit_attribute_micro_atlas import build_cases, prompt_for  # noqa: E402
from phase727_category_fruit_cluster_intervention import hit_answer, norm_text  # noqa: E402
from phase729_full_head_vs_cluster_residual_propagation import install_multi_ablation  # noqa: E402
from phase730_downstream_node_cancellation import module_for_site, replace_tensor  # noqa: E402


OUT_ROOT = Path("results/glm5_phase732_full_path_atlas_causal_edge_validation")
PHASE731_ROOT = Path("results/glm5_phase731_full_path_functional_atlas_v0")
PHASE729_ROOT = Path("results/glm5_phase729_full_head_vs_cluster_residual_propagation")
MODELS = ["qwen3", "glm4", "deepseek7b"]

PROMPT_SITES = {
    "qwen3": ["hidden_35", "hidden_33"],
    "glm4": ["hidden_39", "hidden_40"],
    "deepseek7b": ["L27_mlp_out", "hidden_27"],
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def norm(vec: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(vec.float()).item())


def cosine(a: torch.Tensor, b: torch.Tensor) -> float | None:
    aa = a.float().flatten()
    bb = b.float().flatten()
    denom = torch.linalg.vector_norm(aa) * torch.linalg.vector_norm(bb)
    if float(denom.item()) <= 1e-9:
        return None
    return float(torch.dot(aa, bb).div(denom).item())


def select_prompt_pairs(max_pairs: int | None = None) -> list[dict[str, Any]]:
    cases = build_cases(None)
    explicit = {
        (c["object"], c["relation"], c["answer"]): c
        for c in cases
        if c["prompt_type"] == "explicit_profile" and c["relation"] in {"category", "color", "taste"}
    }
    commonsense = [
        c for c in cases
        if c["prompt_type"] == "commonsense" and c["relation"] in {"category", "color", "taste"}
    ]
    pairs = []
    for c in commonsense:
        key = (c["object"], c["relation"], c["answer"])
        if key in explicit:
            pairs.append({"pair_id": f"{c['object']}:{c['relation']}", "explicit": explicit[key], "commonsense": c})
    return pairs[:max_pairs] if max_pairs else pairs


def select_head_cases(max_cases: int | None = None) -> list[dict[str, Any]]:
    rows = [c for c in build_cases(None) if c["relation"] in {"category", "color", "taste"}]
    return rows[:max_cases] if max_cases else rows


def get_logits(model, device, ids: list[int], handles_extra: list[Any] | None = None) -> torch.Tensor:
    handles = handles_extra or []
    try:
        with torch.inference_mode():
            out = model(input_ids=torch.tensor([ids], device=device), return_dict=True, use_cache=False)
        return out.logits[0, -1].detach().float().cpu()
    finally:
        for h in handles:
            h.remove()


def capture_site_vec(model, device, ids: list[int], site: str) -> torch.Tensor:
    captured: dict[str, torch.Tensor] = {}
    module = module_for_site(model, site)

    def hook(_module, _inputs, output):
        captured["vec"] = output[0][0, -1].detach().float().cpu() if isinstance(output, tuple) else output[0, -1].detach().float().cpu()

    handle = module.register_forward_hook(hook)
    try:
        with torch.inference_mode():
            model(input_ids=torch.tensor([ids], device=device), return_dict=True, use_cache=False)
    finally:
        handle.remove()
    return captured["vec"]


def logits_with_site_replacement(
    model,
    device,
    recipient_ids: list[int],
    donor_vec: torch.Tensor,
    site: str,
    head_ablation: list[dict[str, int]] | None = None,
) -> torch.Tensor:
    handles = []
    if head_ablation:
        handles.extend(install_zero_head_ablation(model, head_ablation))
    module = module_for_site(model, site)

    def hook(_module, _inputs, output, vec=donor_vec):
        return replace_tensor(output, vec)

    handles.append(module.register_forward_hook(hook))
    return get_logits(model, device, recipient_ids, handles)


def phrase_diag_plain(model, tokenizer, device, prompt: str, answer: str, head_ablation: list[dict[str, int]] | None = None) -> dict[str, Any]:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    ans_ids = target_token_ids(tokenizer, answer)
    cur = list(prompt_ids)
    diags = []
    for tid in ans_ids:
        handles = install_zero_head_ablation(model, head_ablation or []) if head_ablation else []
        logits = get_logits(model, device, cur, handles)
        diags.append(logit_diag(logits, int(tid)))
        cur.append(int(tid))
    return {
        "mean_logprob": sum(d["target_logprob"] for d in diags) / len(diags),
        "sum_logprob": sum(d["target_logprob"] for d in diags),
        "first_rank": diags[0]["target_rank"],
        "first_margin": diags[0]["margin_vs_best_other"],
        "first_top1": diags[0]["target_top1"],
    }


def phrase_diag_site_replace(
    model,
    tokenizer,
    device,
    recipient_prompt: str,
    donor_prompt: str,
    answer: str,
    site: str,
    head_ablation: list[dict[str, int]] | None = None,
) -> dict[str, Any]:
    recipient_ids = tokenizer.encode(recipient_prompt, add_special_tokens=False)
    donor_ids = tokenizer.encode(donor_prompt, add_special_tokens=False)
    ans_ids = target_token_ids(tokenizer, answer)
    rcur = list(recipient_ids)
    dcur = list(donor_ids)
    diags = []
    for tid in ans_ids:
        donor_vec = capture_site_vec(model, device, dcur, site)
        logits = logits_with_site_replacement(model, device, rcur, donor_vec, site, head_ablation=head_ablation)
        diags.append(logit_diag(logits, int(tid)))
        rcur.append(int(tid))
        dcur.append(int(tid))
    return {
        "mean_logprob": sum(d["target_logprob"] for d in diags) / len(diags),
        "sum_logprob": sum(d["target_logprob"] for d in diags),
        "first_rank": diags[0]["target_rank"],
        "first_margin": diags[0]["margin_vs_best_other"],
        "first_top1": diags[0]["target_top1"],
    }


def greedy_generate_plain(model, tokenizer, device, prompt: str, max_new_tokens: int, head_ablation: list[dict[str, int]] | None = None) -> dict[str, Any]:
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    new_ids = []
    for _ in range(max_new_tokens):
        handles = install_zero_head_ablation(model, head_ablation or []) if head_ablation else []
        logits = get_logits(model, device, ids, handles)
        tok = int(torch.argmax(logits).item())
        new_ids.append(tok)
        ids.append(tok)
        text = tokenizer.decode(new_ids, skip_special_tokens=True)
        if "\n" in text or "." in text or ";" in text:
            break
    return {"text": tokenizer.decode(new_ids, skip_special_tokens=True).strip(), "token_ids": new_ids}


def greedy_generate_site_replace(
    model,
    tokenizer,
    device,
    recipient_prompt: str,
    donor_prompt: str,
    site: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    rids = tokenizer.encode(recipient_prompt, add_special_tokens=False)
    dids = tokenizer.encode(donor_prompt, add_special_tokens=False)
    new_ids = []
    for _ in range(max_new_tokens):
        donor_vec = capture_site_vec(model, device, dids, site)
        logits = logits_with_site_replacement(model, device, rids, donor_vec, site)
        tok = int(torch.argmax(logits).item())
        new_ids.append(tok)
        rids.append(tok)
        dids.append(tok)
        text = tokenizer.decode(new_ids, skip_special_tokens=True)
        if "\n" in text or "." in text or ";" in text:
            break
    return {"text": tokenizer.decode(new_ids, skip_special_tokens=True).strip(), "token_ids": new_ids}


def load_candidate_heads(model_name: str, top_k: int) -> list[dict[str, Any]]:
    p = PHASE731_ROOT / f"phase731_{model_name}_summary.json"
    if p.exists():
        data = json.loads(p.read_text(encoding="utf-8"))
        rows = data.get("candidate_head_attention_summary", [])[:top_k]
        return [{"layer": int(r["layer"]), "head": int(r["head"]), "head_key": r["head_key"]} for r in rows]
    fallback = {
        "qwen3": [(24, 29), (28, 0), (29, 11), (26, 26)],
        "glm4": [(29, 18), (29, 28), (24, 19), (28, 6)],
        "deepseek7b": [(20, 17), (23, 0), (24, 21), (27, 23)],
    }
    return [{"layer": l, "head": h, "head_key": f"L{l}H{h}"} for l, h in fallback[model_name][:top_k]]


def prompt_transfer_rows(model, tokenizer, device, model_name: str, pairs: list[dict[str, Any]], sites: list[str], max_new_tokens: int) -> list[dict[str, Any]]:
    rows = []
    for pair in pairs:
        common = pair["commonsense"]
        explicit = pair["explicit"]
        common_prompt = prompt_for(common)
        explicit_prompt = prompt_for(explicit)
        answer = common["answer"]
        common_base = phrase_diag_plain(model, tokenizer, device, common_prompt, answer)
        explicit_base = phrase_diag_plain(model, tokenizer, device, explicit_prompt, answer)
        common_gen = greedy_generate_plain(model, tokenizer, device, common_prompt, max_new_tokens)
        explicit_gen = greedy_generate_plain(model, tokenizer, device, explicit_prompt, max_new_tokens)
        for site in sites:
            c_with_e = phrase_diag_site_replace(model, tokenizer, device, common_prompt, explicit_prompt, answer, site)
            e_with_c = phrase_diag_site_replace(model, tokenizer, device, explicit_prompt, common_prompt, answer, site)
            c_gen_e = greedy_generate_site_replace(model, tokenizer, device, common_prompt, explicit_prompt, site, max_new_tokens)
            e_gen_c = greedy_generate_site_replace(model, tokenizer, device, explicit_prompt, common_prompt, site, max_new_tokens)
            rows.append(
                {
                    "model": model_name,
                    "edge_family": "prompt_type_skeleton_to_site",
                    "pair_id": pair["pair_id"],
                    "object": common["object"],
                    "relation": common["relation"],
                    "answer": answer,
                    "site": site,
                    "direction": "commonsense<-explicit",
                    "baseline_logprob": common_base["mean_logprob"],
                    "donor_logprob": explicit_base["mean_logprob"],
                    "patched_logprob": c_with_e["mean_logprob"],
                    "patched_delta_vs_recipient": c_with_e["mean_logprob"] - common_base["mean_logprob"],
                    "recipient_first_rank": common_base["first_rank"],
                    "patched_first_rank": c_with_e["first_rank"],
                    "recipient_generated_text": common_gen["text"],
                    "donor_generated_text": explicit_gen["text"],
                    "patched_generated_text": c_gen_e["text"],
                    "recipient_hit": hit_answer(common_gen["text"], answer),
                    "patched_hit": hit_answer(c_gen_e["text"], answer),
                    "changed_vs_recipient": norm_text(c_gen_e["text"]) != norm_text(common_gen["text"]),
                }
            )
            rows.append(
                {
                    "model": model_name,
                    "edge_family": "prompt_type_skeleton_to_site",
                    "pair_id": pair["pair_id"],
                    "object": common["object"],
                    "relation": common["relation"],
                    "answer": answer,
                    "site": site,
                    "direction": "explicit<-commonsense",
                    "baseline_logprob": explicit_base["mean_logprob"],
                    "donor_logprob": common_base["mean_logprob"],
                    "patched_logprob": e_with_c["mean_logprob"],
                    "patched_delta_vs_recipient": e_with_c["mean_logprob"] - explicit_base["mean_logprob"],
                    "recipient_first_rank": explicit_base["first_rank"],
                    "patched_first_rank": e_with_c["first_rank"],
                    "recipient_generated_text": explicit_gen["text"],
                    "donor_generated_text": common_gen["text"],
                    "patched_generated_text": e_gen_c["text"],
                    "recipient_hit": hit_answer(explicit_gen["text"], answer),
                    "patched_hit": hit_answer(e_gen_c["text"], answer),
                    "changed_vs_recipient": norm_text(e_gen_c["text"]) != norm_text(explicit_gen["text"]),
                }
            )
    return rows


def head_ablation_rows(model, tokenizer, device, model_name: str, cases: list[dict[str, Any]], heads: list[dict[str, Any]], max_new_tokens: int) -> list[dict[str, Any]]:
    rows = []
    for case in cases:
        prompt = prompt_for(case)
        answer = case["answer"]
        base = phrase_diag_plain(model, tokenizer, device, prompt, answer)
        base_gen = greedy_generate_plain(model, tokenizer, device, prompt, max_new_tokens)
        for head in heads:
            spec = [{"layer": head["layer"], "head": head["head"]}]
            patched = phrase_diag_plain(model, tokenizer, device, prompt, answer, head_ablation=spec)
            patched_gen = greedy_generate_plain(model, tokenizer, device, prompt, max_new_tokens, head_ablation=spec)
            rows.append(
                {
                    "model": model_name,
                    "edge_family": "candidate_head_to_readout",
                    "case_id": case["case_id"],
                    "prompt_type": case["prompt_type"],
                    "object": case["object"],
                    "object_group": case["object_group"],
                    "relation": case["relation"],
                    "answer": answer,
                    "head_key": head["head_key"],
                    "layer": head["layer"],
                    "head": head["head"],
                    "baseline_logprob": base["mean_logprob"],
                    "patched_logprob": patched["mean_logprob"],
                    "logprob_delta": patched["mean_logprob"] - base["mean_logprob"],
                    "rank_delta": patched["first_rank"] - base["first_rank"],
                    "baseline_generated_text": base_gen["text"],
                    "patched_generated_text": patched_gen["text"],
                    "baseline_hit": hit_answer(base_gen["text"], answer),
                    "patched_hit": hit_answer(patched_gen["text"], answer),
                    "changed_vs_baseline": norm_text(patched_gen["text"]) != norm_text(base_gen["text"]),
                }
            )
    return rows


def capture_two_sites(model, device, ids: list[int], site_a: str, site_b: str, head_ablation: list[dict[str, int]] | None = None) -> dict[str, torch.Tensor]:
    captured: dict[str, torch.Tensor] = {}
    handles = install_zero_head_ablation(model, head_ablation or []) if head_ablation else []
    for site in [site_a, site_b]:
        module = module_for_site(model, site)

        def hook(_module, _inputs, output, key=site):
            x = output[0] if isinstance(output, tuple) else output
            captured[key] = x[0, -1].detach().float().cpu()

        handles.append(module.register_forward_hook(hook))
    try:
        with torch.inference_mode():
            out = model(input_ids=torch.tensor([ids], device=device), return_dict=True, use_cache=False)
        captured["final_logits"] = out.logits[0, -1].detach().float().cpu()
    finally:
        for h in handles:
            h.remove()
    return captured


def edge_mediation_rows(model, tokenizer, device, model_name: str, cases: list[dict[str, Any]], heads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    if model_name != "deepseek7b":
        return rows
    head = next((h for h in heads if h["head_key"] == "L20H17"), {"layer": 20, "head": 17, "head_key": "L20H17"})
    source_site = "L20_attn_out"
    target_site = "L24_mlp_out"
    for case in cases:
        if case["relation"] != "category":
            continue
        prompt = prompt_for(case)
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        base = capture_two_sites(model, device, ids, source_site, target_site)
        ablated = capture_two_sites(model, device, ids, source_site, target_site, head_ablation=[{"layer": head["layer"], "head": head["head"]}])
        target_id = int(target_token_ids(tokenizer, case["answer"])[0])
        bdiag = logit_diag(base["final_logits"], target_id)
        adiag = logit_diag(ablated["final_logits"], target_id)
        rows.append(
            {
                "model": model_name,
                "edge_family": "full_head_to_mlp_readout",
                "case_id": case["case_id"],
                "prompt_type": case["prompt_type"],
                "object": case["object"],
                "object_group": case["object_group"],
                "relation": case["relation"],
                "answer": case["answer"],
                "head_key": head["head_key"],
                "source_site": source_site,
                "target_site": target_site,
                "source_delta_norm": norm(ablated[source_site] - base[source_site]),
                "target_delta_norm": norm(ablated[target_site] - base[target_site]),
                "source_target_delta_cos": cosine(ablated[source_site] - base[source_site], ablated[target_site] - base[target_site]),
                "target_logprob_delta": adiag["target_logprob"] - bdiag["target_logprob"],
                "target_rank_delta": adiag["target_rank"] - bdiag["target_rank"],
            }
        )
    return rows


def mean(vals: list[float | None]) -> float | None:
    xs = [float(v) for v in vals if v is not None]
    return sum(xs) / len(xs) if xs else None


def summarize_rows(model_name: str, prompt_rows: list[dict[str, Any]], head_rows: list[dict[str, Any]], mediation_rows: list[dict[str, Any]]) -> dict[str, Any]:
    prompt_summary = {}
    for (direction, site), vals in defaultdict(list, {}).items():
        pass
    grouped_prompt: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in prompt_rows:
        grouped_prompt[(row["direction"], row["site"])].append(row)
    for (direction, site), vals in grouped_prompt.items():
        prompt_summary[f"{direction}|{site}"] = {
            "n": len(vals),
            "mean_patched_delta_vs_recipient": mean([v["patched_delta_vs_recipient"] for v in vals]),
            "changed_rate": sum(1 for v in vals if v["changed_vs_recipient"]) / len(vals),
            "recipient_hit_rate": sum(1 for v in vals if v["recipient_hit"]) / len(vals),
            "patched_hit_rate": sum(1 for v in vals if v["patched_hit"]) / len(vals),
            "hit_gain": sum(1 for v in vals if (not v["recipient_hit"]) and v["patched_hit"]) / len(vals),
            "hit_loss": sum(1 for v in vals if v["recipient_hit"] and (not v["patched_hit"])) / len(vals),
        }

    grouped_head: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in head_rows:
        grouped_head[(row["head_key"], row["relation"], row["prompt_type"])].append(row)
    head_summary = []
    for (head_key, relation, prompt_type), vals in grouped_head.items():
        head_summary.append(
            {
                "head_key": head_key,
                "relation": relation,
                "prompt_type": prompt_type,
                "n": len(vals),
                "mean_logprob_delta": mean([v["logprob_delta"] for v in vals]),
                "mean_rank_delta": mean([v["rank_delta"] for v in vals]),
                "changed_rate": sum(1 for v in vals if v["changed_vs_baseline"]) / len(vals),
                "hit_drop_rate": sum(1 for v in vals if v["baseline_hit"] and not v["patched_hit"]) / len(vals),
            }
        )

    mediation_summary = {}
    if mediation_rows:
        mediation_summary = {
            "n": len(mediation_rows),
            "mean_source_delta_norm": mean([r["source_delta_norm"] for r in mediation_rows]),
            "mean_target_delta_norm": mean([r["target_delta_norm"] for r in mediation_rows]),
            "mean_source_target_delta_cos": mean([r["source_target_delta_cos"] for r in mediation_rows]),
            "mean_target_logprob_delta": mean([r["target_logprob_delta"] for r in mediation_rows]),
            "mean_target_rank_delta": mean([r["target_rank_delta"] for r in mediation_rows]),
        }

    return {
        "phase": 732,
        "title": "Full-Path Atlas Causal Edge Validation",
        "model": model_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_prompt_transfer_rows": len(prompt_rows),
        "n_head_ablation_rows": len(head_rows),
        "n_edge_mediation_rows": len(mediation_rows),
        "prompt_transfer_summary": prompt_summary,
        "head_ablation_summary": sorted(head_summary, key=lambda r: (r["mean_logprob_delta"] or 0)),
        "edge_mediation_summary": mediation_summary,
        "strict_interpretation": "causal edge validation v0; site replacement can be distribution-shifting and head ablation remains coarse",
    }


def run_model(args) -> dict[str, Any]:
    prompt_pairs = select_prompt_pairs(args.max_prompt_pairs)
    head_cases = select_head_cases(args.max_head_cases)
    model, tokenizer, device = load_model(args.model)
    try:
        sites = PROMPT_SITES[args.model]
        heads = load_candidate_heads(args.model, args.top_heads)
        log(f"{args.model}: prompt_pairs={len(prompt_pairs)}, head_cases={len(head_cases)}, sites={sites}, heads={[h['head_key'] for h in heads]}")
        prompt_rows = prompt_transfer_rows(model, tokenizer, device, args.model, prompt_pairs, sites, args.max_new_tokens)
        log(f"{args.model}: prompt_transfer_rows={len(prompt_rows)}")
        head_rows = head_ablation_rows(model, tokenizer, device, args.model, head_cases, heads, args.max_new_tokens)
        log(f"{args.model}: head_ablation_rows={len(head_rows)}")
        mediation_rows = edge_mediation_rows(model, tokenizer, device, args.model, head_cases, heads)
        log(f"{args.model}: edge_mediation_rows={len(mediation_rows)}")
    finally:
        release_model(model)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = summarize_rows(args.model, prompt_rows, head_rows, mediation_rows)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT_ROOT / f"phase732_{args.model}_prompt_transfer_rows.jsonl", prompt_rows)
    write_jsonl(OUT_ROOT / f"phase732_{args.model}_head_ablation_rows.jsonl", head_rows)
    write_jsonl(OUT_ROOT / f"phase732_{args.model}_edge_mediation_rows.jsonl", mediation_rows)
    write_json(OUT_ROOT / f"phase732_{args.model}_summary.json", summary)
    print(json.dumps({"model": args.model, "prompt_transfer_summary": summary["prompt_transfer_summary"], "top_head_edges": summary["head_ablation_summary"][:4], "edge_mediation_summary": summary["edge_mediation_summary"]}, ensure_ascii=False, indent=2), flush=True)
    return summary


def build_atlas_graph(payload: dict[str, Any]) -> dict[str, Any]:
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
        phase_node = f"{model}:phase:732"
        add_node({"id": model_node, "type": "model", "label": model, "model": model, "position": [-16, 0, lane_z], "role": "tested_model"})
        add_node({"id": phase_node, "type": "phase", "label": "Phase 732", "model": model, "position": [-12, 2, lane_z], "role": "causal_edge_validation"})
        edges.append({"source": model_node, "target": phase_node, "relation": "contains", "weight": 0, "phase": 732})
        for key, rec in summary.get("prompt_transfer_summary", {}).items():
            direction, site = key.split("|", 1)
            edge_node = f"{model}:edge:prompt:{direction}:{site}"
            add_node(
                {
                    "id": edge_node,
                    "type": "intervention",
                    "label": key,
                    "model": model,
                    "role": "prompt_type_site_transfer",
                    "evidence_level": "causal_replacement",
                    "site": site,
                    "direction": direction,
                    "score": rec.get("mean_patched_delta_vs_recipient"),
                    "changed_rate": rec.get("changed_rate"),
                    "hit_gain": rec.get("hit_gain"),
                    "hit_loss": rec.get("hit_loss"),
                }
            )
            relation = "supports_likelihood" if (rec.get("mean_patched_delta_vs_recipient") or 0) > 0 else "negative_effect"
            if rec.get("changed_rate", 0) > 0:
                relation = "changes_generation"
            edges.append({"source": phase_node, "target": edge_node, "relation": relation, "weight": abs(rec.get("mean_patched_delta_vs_recipient") or 0), "phase": 732})
        for rec in summary.get("head_ablation_summary", [])[:12]:
            head_node = f"{model}:head_edge:{rec['head_key']}:{rec['relation']}:{rec['prompt_type']}"
            add_node(
                {
                    "id": head_node,
                    "type": "head",
                    "label": f"{rec['head_key']} {rec['relation']} {rec['prompt_type']}",
                    "model": model,
                    "role": "candidate_head_causal_edge",
                    "evidence_level": "head_ablation",
                    "mean_logprob_delta": rec.get("mean_logprob_delta"),
                    "changed_rate": rec.get("changed_rate"),
                    "hit_drop_rate": rec.get("hit_drop_rate"),
                    "score": rec.get("mean_logprob_delta"),
                }
            )
            relation = "supports_likelihood" if (rec.get("mean_logprob_delta") or 0) < 0 else "weak_or_null"
            if rec.get("changed_rate", 0) > 0:
                relation = "changes_generation"
            edges.append({"source": phase_node, "target": head_node, "relation": relation, "weight": abs(rec.get("mean_logprob_delta") or 0), "phase": 732})
        med = summary.get("edge_mediation_summary") or {}
        if med:
            med_node = f"{model}:edge:L20H17_to_L24_mlp"
            add_node(
                {
                    "id": med_node,
                    "type": "cluster",
                    "label": "L20H17 -> L24_mlp_out",
                    "model": model,
                    "role": "full_head_to_mlp_edge",
                    "evidence_level": "mediation_delta",
                    "mean_source_delta_norm": med.get("mean_source_delta_norm"),
                    "mean_target_delta_norm": med.get("mean_target_delta_norm"),
                    "mean_target_logprob_delta": med.get("mean_target_logprob_delta"),
                    "score": med.get("mean_target_delta_norm"),
                }
            )
            edges.append({"source": phase_node, "target": med_node, "relation": "partial_mediator", "weight": med.get("mean_target_delta_norm") or 0, "phase": 732})
    return {
        "schema_version": "atlas_graph_v1",
        "title": "Phase 732 Full-Path Atlas Causal Edge Validation",
        "model_info": {"model": "cross_model", "models": models, "phase": 732, "timestamp": payload.get("timestamp"), "evidence_type": payload.get("evidence_type")},
        "layout": {"x": "edge family", "y": "layer/site", "z": "model lane"},
        "graph": {"nodes": nodes, "edges": edges},
        "metrics": {"node_count": len(nodes), "edge_count": len(edges), "source_phase": 732},
        "source_files": [str(OUT_ROOT / "phase732_cross_model_summary.json")],
    }


def write_cross_summary() -> dict[str, Any]:
    summaries = []
    for model in MODELS:
        path = OUT_ROOT / f"phase732_{model}_summary.json"
        if path.exists():
            summaries.append(json.loads(path.read_text(encoding="utf-8")))
    payload = {
        "phase": 732,
        "title": "Full-Path Atlas Causal Edge Validation",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": [s["model"] for s in summaries],
        "status": "complete" if len(summaries) == len(MODELS) else "partial",
        "evidence_type": "prompt-type site replacement + candidate-head ablation + DS7B full-head-to-MLP delta",
        "small_model_caution": "edge validation is still v0 and may be small-model/template-specific",
        "by_model": {s["model"]: s for s in summaries},
    }
    write_json(OUT_ROOT / "phase732_cross_model_summary.json", payload)
    graph = build_atlas_graph(payload)
    write_json(OUT_ROOT / "phase732_atlas_graph.json", graph)
    lines = [
        "# Phase 732 Full-Path Atlas Causal Edge Validation",
        "",
        f"- Status: `{payload['status']}`",
        f"- Models: `{payload['models']}`",
        "- Evidence type: prompt-type site replacement + candidate-head ablation + DS7B full-head-to-MLP delta.",
        "",
        "| model | best prompt transfer | delta | changed | strongest head edge | head delta | mediation target delta |",
        "|---|---|---:|---:|---|---:|---:|",
    ]
    for model, summary in payload["by_model"].items():
        prompt_items = list(summary.get("prompt_transfer_summary", {}).items())
        best_prompt = max(prompt_items, key=lambda kv: abs(kv[1].get("mean_patched_delta_vs_recipient") or 0)) if prompt_items else ("-", {})
        head_items = summary.get("head_ablation_summary", [])
        best_head = min(head_items, key=lambda r: r.get("mean_logprob_delta") or 0) if head_items else {}
        med = summary.get("edge_mediation_summary") or {}
        lines.append(
            f"| {model} | {best_prompt[0]} | {best_prompt[1].get('mean_patched_delta_vs_recipient', 0):.3f} | "
            f"{best_prompt[1].get('changed_rate', 0):.3f} | "
            f"{best_head.get('head_key', '-')}/{best_head.get('relation', '-')}/{best_head.get('prompt_type', '-')} | "
            f"{best_head.get('mean_logprob_delta', 0):.3f} | "
            f"{med.get('mean_target_delta_norm', 0):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Strict Interpretation",
            "",
            "- Positive prompt transfer delta means donor site replacement improved recipient target likelihood.",
            "- Negative prompt transfer delta means replacement hurt target likelihood or caused distribution shift.",
            "- Head ablation is coarse and tests necessity, not semantic purity.",
            "- DS7B full-head-to-MLP delta tests propagation of perturbation, not full closure.",
            "",
            f"Atlas graph: nodes={graph['metrics']['node_count']} edges={graph['metrics']['edge_count']}",
            "",
        ]
    )
    (OUT_ROOT / "phase732_cross_model_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "models": payload["models"], "atlas": graph["metrics"]}, ensure_ascii=False, indent=2), flush=True)
    return payload


def dry_run(max_prompt_pairs: int | None, max_head_cases: int | None) -> None:
    pairs = select_prompt_pairs(max_prompt_pairs)
    cases = select_head_cases(max_head_cases)
    print(json.dumps({"prompt_pairs": len(pairs), "head_cases": len(cases), "pairs_sample": pairs[:3], "head_case_counts": dict(Counter(c["prompt_type"] for c in cases))}, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-prompt-pairs", type=int, default=None)
    parser.add_argument("--max-head-cases", type=int, default=None)
    parser.add_argument("--top-heads", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        dry_run(args.max_prompt_pairs, args.max_head_cases)
        return
    if args.summarize_only:
        write_cross_summary()
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
