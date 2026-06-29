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

from model_utils import MODEL_CONFIGS, get_layers, release_model  # noqa: E402
from phase722_functional_head_atlas_causal_ablation import logit_diag, target_token_ids, write_json, write_jsonl  # noqa: E402
from phase723_apple_fruit_attribute_micro_atlas import build_cases, prompt_for  # noqa: E402
from phase727_category_fruit_cluster_intervention import hit_answer, norm_text  # noqa: E402


OUT_ROOT = Path("results/glm5_phase733_prompt_type_skeleton_source_localization")
MODELS = ["qwen3", "glm4", "deepseek7b"]
PROMPT_TYPES = ["explicit_profile", "conflict_profile", "commonsense"]
PAIR_NAMES = [
    ("commonsense", "explicit_profile"),
    ("conflict_profile", "explicit_profile"),
    ("commonsense", "conflict_profile"),
]
SITE_KINDS = ["layer_input", "attn_out", "mlp_out", "layer_out"]

LATE_REFERENCE = {
    "qwen3": ["hidden_35", "hidden_33"],
    "glm4": ["hidden_39", "hidden_40"],
    "deepseek7b": ["L27_mlp_out", "hidden_27"],
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def norm(vec: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(vec.float()).item())


def get_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, tuple):
        return output[0]
    return output


def replace_tensor(output: Any, vec: torch.Tensor) -> Any:
    if isinstance(output, tuple):
        first = output[0].clone()
        first[0, -1] = vec.to(device=first.device, dtype=first.dtype)
        return (first,) + output[1:]
    y = output.clone()
    y[0, -1] = vec.to(device=y.device, dtype=y.dtype)
    return y


def input_device(model) -> torch.device:
    return next(model.parameters()).device


def load_model_bf16_flash(model_name: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = MODEL_CONFIGS[model_name]
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"],
        trust_remote_code=True,
        local_files_only=True,
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    last_error: Exception | None = None
    for attn_impl in ["flash_attention_2", "sdpa", "eager"]:
        try:
            log(f"[load] {model_name}: bf16 device_map=auto attn={attn_impl} quantization=off")
            model = AutoModelForCausalLM.from_pretrained(
                cfg["path"],
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                local_files_only=True,
                low_cpu_mem_usage=True,
                attn_implementation=attn_impl,
            )
            model.eval()
            gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
            if hasattr(model, "hf_device_map"):
                dmap = model.hf_device_map
                gpu_count = sum(1 for v in dmap.values() if "cuda" in str(v))
                cpu_count = sum(1 for v in dmap.values() if "cpu" in str(v))
                log(f"[load] {model_name}: loaded attn={attn_impl}, gpu_components={gpu_count}, cpu_components={cpu_count}, gpu={gpu_mem:.2f}GB")
            else:
                log(f"[load] {model_name}: loaded attn={attn_impl}, device={input_device(model)}, gpu={gpu_mem:.2f}GB")
            return model, tokenizer, input_device(model), attn_impl
        except Exception as exc:  # flash may be unavailable for some local builds.
            last_error = exc
            log(f"[load] {model_name}: attn={attn_impl} failed: {type(exc).__name__}: {exc}")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    raise RuntimeError(f"failed to load {model_name} with bf16 non-quantized fallback path") from last_error


def select_scan_cases(max_scan_cases: int | None) -> list[dict[str, Any]]:
    rows = [c for c in build_cases(None) if c["relation"] in {"category", "color", "taste"}]
    if not max_scan_cases:
        return rows
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_type[row["prompt_type"]].append(row)
    selected: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    while len(selected) < max_scan_cases:
        progressed = False
        for prompt_type in PROMPT_TYPES:
            bucket = by_type.get(prompt_type, [])
            if bucket:
                row = bucket.pop(0)
                selected.append(row)
                used_ids.add(row["case_id"])
                progressed = True
                if len(selected) >= max_scan_cases:
                    break
        if not progressed:
            break
    if len(selected) < max_scan_cases:
        for row in rows:
            if row["case_id"] in used_ids:
                continue
            selected.append(row)
            if len(selected) >= max_scan_cases:
                break
    return selected


def select_prompt_pairs(max_pairs: int | None) -> list[dict[str, Any]]:
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
            pairs.append({"pair_id": f"{c['object']}:{c['relation']}", "explicit_profile": explicit[key], "commonsense": c})
    return pairs[:max_pairs] if max_pairs else pairs


def capture_all_sites(model, device, ids: list[int]) -> dict[str, torch.Tensor]:
    layers = get_layers(model)
    captured: dict[str, torch.Tensor] = {}
    handles = []

    def make_pre(li: int):
        def hook(_module, inputs):
            captured[f"L{li}_layer_input"] = inputs[0][0, -1].detach().float().cpu()
        return hook

    def make_component(li: int, name: str):
        def hook(_module, _inputs, output):
            captured[f"L{li}_{name}"] = get_tensor(output)[0, -1].detach().float().cpu()
        return hook

    def make_layer(li: int):
        def hook(_module, _inputs, output):
            captured[f"hidden_{li + 1}"] = get_tensor(output)[0, -1].detach().float().cpu()
        return hook

    try:
        for li, layer in enumerate(layers):
            handles.append(layer.register_forward_pre_hook(make_pre(li)))
            handles.append(layer.register_forward_hook(make_layer(li)))
            if hasattr(layer, "self_attn"):
                handles.append(layer.self_attn.register_forward_hook(make_component(li, "attn_out")))
            if hasattr(layer, "mlp"):
                handles.append(layer.mlp.register_forward_hook(make_component(li, "mlp_out")))
        with torch.inference_mode():
            model(input_ids=torch.tensor([ids], device=device), return_dict=True, use_cache=False)
        return captured
    finally:
        for h in handles:
            h.remove()


def site_kind_layer(site: str) -> tuple[str, int]:
    if site.startswith("hidden_"):
        return "layer_out", int(site.split("_", 1)[1]) - 1
    if site.startswith("L") and site.endswith("_layer_input"):
        return "layer_input", int(site[1:].split("_", 1)[0])
    if site.startswith("L") and site.endswith("_attn_out"):
        return "attn_out", int(site[1:].split("_", 1)[0])
    if site.startswith("L") and site.endswith("_mlp_out"):
        return "mlp_out", int(site[1:].split("_", 1)[0])
    raise ValueError(site)


def site_module(model, site: str):
    kind, layer_idx = site_kind_layer(site)
    layer = get_layers(model)[layer_idx]
    if kind in {"layer_input", "layer_out"}:
        return layer
    if kind == "attn_out":
        return layer.self_attn
    if kind == "mlp_out":
        return layer.mlp
    raise ValueError(kind)


def capture_site_vec(model, device, ids: list[int], site: str) -> torch.Tensor:
    captured: dict[str, torch.Tensor] = {}
    kind, _layer_idx = site_kind_layer(site)
    module = site_module(model, site)
    if kind == "layer_input":
        def pre_hook(_module, inputs):
            captured["vec"] = inputs[0][0, -1].detach().float().cpu()
        handle = module.register_forward_pre_hook(pre_hook)
    else:
        def hook(_module, _inputs, output):
            captured["vec"] = get_tensor(output)[0, -1].detach().float().cpu()
        handle = module.register_forward_hook(hook)
    try:
        with torch.inference_mode():
            model(input_ids=torch.tensor([ids], device=device), return_dict=True, use_cache=False)
    finally:
        handle.remove()
    return captured["vec"]


def logits_with_site_replace(model, device, ids: list[int], site: str, donor_vec: torch.Tensor) -> torch.Tensor:
    kind, _layer_idx = site_kind_layer(site)
    module = site_module(model, site)
    if kind == "layer_input":
        def pre_hook(_module, inputs, vec=donor_vec):
            x = inputs[0].clone()
            x[0, -1] = vec.to(device=x.device, dtype=x.dtype)
            return (x,) + tuple(inputs[1:])
        handle = module.register_forward_pre_hook(pre_hook)
    else:
        def hook(_module, _inputs, output, vec=donor_vec):
            return replace_tensor(output, vec)
        handle = module.register_forward_hook(hook)
    try:
        with torch.inference_mode():
            out = model(input_ids=torch.tensor([ids], device=device), return_dict=True, use_cache=False)
        return out.logits[0, -1].detach().float().cpu()
    finally:
        handle.remove()


def phrase_diag_site_replace(model, tokenizer, device, recipient_prompt: str, donor_prompt: str, answer: str, site: str) -> dict[str, Any]:
    recipient_ids = tokenizer.encode(recipient_prompt, add_special_tokens=False)
    donor_ids = tokenizer.encode(donor_prompt, add_special_tokens=False)
    ans_ids = target_token_ids(tokenizer, answer)
    rcur = list(recipient_ids)
    dcur = list(donor_ids)
    diags = []
    for tid in ans_ids:
        donor_vec = capture_site_vec(model, device, dcur, site)
        logits = logits_with_site_replace(model, device, rcur, site, donor_vec)
        diags.append(logit_diag(logits, int(tid)))
        rcur.append(int(tid))
        dcur.append(int(tid))
    return {
        "mean_logprob": sum(d["target_logprob"] for d in diags) / len(diags),
        "first_rank": diags[0]["target_rank"],
        "first_margin": diags[0]["margin_vs_best_other"],
        "first_top1": diags[0]["target_top1"],
    }


def phrase_diag_plain(model, tokenizer, device, prompt: str, answer: str) -> dict[str, Any]:
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    ans_ids = target_token_ids(tokenizer, answer)
    cur = list(ids)
    diags = []
    for tid in ans_ids:
        with torch.inference_mode():
            out = model(input_ids=torch.tensor([cur], device=device), return_dict=True, use_cache=False)
        diags.append(logit_diag(out.logits[0, -1].detach().float().cpu(), int(tid)))
        cur.append(int(tid))
    return {
        "mean_logprob": sum(d["target_logprob"] for d in diags) / len(diags),
        "first_rank": diags[0]["target_rank"],
        "first_margin": diags[0]["margin_vs_best_other"],
        "first_top1": diags[0]["target_top1"],
    }


def greedy_generate_site_replace(model, tokenizer, device, recipient_prompt: str, donor_prompt: str, site: str, max_new_tokens: int) -> dict[str, Any]:
    rids = tokenizer.encode(recipient_prompt, add_special_tokens=False)
    dids = tokenizer.encode(donor_prompt, add_special_tokens=False)
    new_ids = []
    for _ in range(max_new_tokens):
        donor_vec = capture_site_vec(model, device, dids, site)
        logits = logits_with_site_replace(model, device, rids, site, donor_vec)
        tok = int(torch.argmax(logits).item())
        new_ids.append(tok)
        rids.append(tok)
        dids.append(tok)
        text = tokenizer.decode(new_ids, skip_special_tokens=True)
        if "\n" in text or "." in text or ";" in text:
            break
    return {"text": tokenizer.decode(new_ids, skip_special_tokens=True).strip(), "token_ids": new_ids}


def greedy_generate_plain(model, tokenizer, device, prompt: str, max_new_tokens: int) -> dict[str, Any]:
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    new_ids = []
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


def scan_formation(model, tokenizer, device, cases: list[dict[str, Any]], log_every: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sums: dict[tuple[str, str], torch.Tensor] = {}
    counts: Counter[tuple[str, str]] = Counter()
    site_meta: dict[str, tuple[str, int]] = {}
    case_rows = []
    for idx, case in enumerate(cases, 1):
        prompt = prompt_for(case)
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        cap = capture_all_sites(model, device, ids)
        for site, vec in cap.items():
            kind, layer = site_kind_layer(site)
            site_meta[site] = (kind, layer)
            key = (site, case["prompt_type"])
            sums[key] = sums.get(key, torch.zeros_like(vec)) + vec
            counts[key] += 1
        case_rows.append(
            {
                "case_id": case["case_id"],
                "prompt_type": case["prompt_type"],
                "object": case["object"],
                "object_group": case["object_group"],
                "relation": case["relation"],
                "answer": case["answer"],
                "n_sites": len(cap),
            }
        )
        if idx % log_every == 0 or idx == len(cases):
            log(f"scan {idx}/{len(cases)} cases; current_sites={len(site_meta)}")
    means = {key: val / max(counts[key], 1) for key, val in sums.items()}
    rows = []
    for site, (kind, layer) in sorted(site_meta.items(), key=lambda kv: (kv[1][1], kv[1][0], kv[0])):
        for a, b in PAIR_NAMES:
            if (site, a) not in means or (site, b) not in means:
                continue
            delta = means[(site, a)] - means[(site, b)]
            mean_norm = (norm(means[(site, a)]) + norm(means[(site, b)])) / 2
            rows.append(
                {
                    "site": site,
                    "site_kind": kind,
                    "layer": layer,
                    "pair": f"{a}_vs_{b}",
                    "effect_norm": norm(delta),
                    "normalized_effect": norm(delta) / max(mean_norm, 1e-9),
                    "n_a": counts[(site, a)],
                    "n_b": counts[(site, b)],
                }
            )
    return rows, case_rows


def choose_candidate_sites(model_name: str, scan_rows: list[dict[str, Any]], max_sites: int) -> list[str]:
    ce = [r for r in scan_rows if r["pair"] == "commonsense_vs_explicit_profile"]
    chosen: list[str] = []
    for kind in SITE_KINDS:
        vals = [r for r in ce if r["site_kind"] == kind]
        if not vals:
            continue
        max_effect = max(v["effect_norm"] for v in vals)
        threshold = max_effect * 0.35
        early = next((v for v in sorted(vals, key=lambda x: x["layer"]) if v["effect_norm"] >= threshold), None)
        top = max(vals, key=lambda x: x["effect_norm"])
        for row in [early, top]:
            if row and row["site"] not in chosen:
                chosen.append(row["site"])
    for site in LATE_REFERENCE.get(model_name, []):
        if site not in chosen:
            chosen.append(site)
    return chosen[:max_sites]


def transfer_validation(
    model,
    tokenizer,
    device,
    pairs: list[dict[str, Any]],
    sites: list[str],
    max_new_tokens: int,
    log_every: int,
) -> list[dict[str, Any]]:
    rows = []
    for idx, pair in enumerate(pairs, 1):
        common = pair["commonsense"]
        explicit = pair["explicit_profile"]
        c_prompt = prompt_for(common)
        e_prompt = prompt_for(explicit)
        answer = common["answer"]
        c_base = phrase_diag_plain(model, tokenizer, device, c_prompt, answer)
        e_base = phrase_diag_plain(model, tokenizer, device, e_prompt, answer)
        c_gen = greedy_generate_plain(model, tokenizer, device, c_prompt, max_new_tokens)
        e_gen = greedy_generate_plain(model, tokenizer, device, e_prompt, max_new_tokens)
        for site in sites:
            for direction, recipient_prompt, donor_prompt, recipient_base, donor_base, recipient_gen, donor_gen in [
                ("commonsense<-explicit", c_prompt, e_prompt, c_base, e_base, c_gen, e_gen),
                ("explicit<-commonsense", e_prompt, c_prompt, e_base, c_base, e_gen, c_gen),
            ]:
                patched = phrase_diag_site_replace(model, tokenizer, device, recipient_prompt, donor_prompt, answer, site)
                patched_gen = greedy_generate_site_replace(model, tokenizer, device, recipient_prompt, donor_prompt, site, max_new_tokens)
                kind, layer = site_kind_layer(site)
                rows.append(
                    {
                        "pair_id": pair["pair_id"],
                        "object": common["object"],
                        "relation": common["relation"],
                        "answer": answer,
                        "site": site,
                        "site_kind": kind,
                        "layer": layer,
                        "direction": direction,
                        "recipient_logprob": recipient_base["mean_logprob"],
                        "donor_logprob": donor_base["mean_logprob"],
                        "patched_logprob": patched["mean_logprob"],
                        "patched_delta_vs_recipient": patched["mean_logprob"] - recipient_base["mean_logprob"],
                        "recipient_rank": recipient_base["first_rank"],
                        "patched_rank": patched["first_rank"],
                        "recipient_generated_text": recipient_gen["text"],
                        "donor_generated_text": donor_gen["text"],
                        "patched_generated_text": patched_gen["text"],
                        "recipient_hit": hit_answer(recipient_gen["text"], answer),
                        "patched_hit": hit_answer(patched_gen["text"], answer),
                        "changed_vs_recipient": norm_text(patched_gen["text"]) != norm_text(recipient_gen["text"]),
                    }
                )
        if idx % log_every == 0 or idx == len(pairs):
            log(f"transfer {idx}/{len(pairs)} pairs; rows={len(rows)}")
    return rows


def mean(vals: list[float | None]) -> float | None:
    xs = [float(v) for v in vals if v is not None]
    return sum(xs) / len(xs) if xs else None


def summarize(model_name: str, round_name: str, attn_impl: str, scan_rows: list[dict[str, Any]], transfer_rows: list[dict[str, Any]], candidate_sites: list[str]) -> dict[str, Any]:
    ce = [r for r in scan_rows if r["pair"] == "commonsense_vs_explicit_profile"]
    formation = {}
    for kind in SITE_KINDS:
        vals = [r for r in ce if r["site_kind"] == kind]
        if not vals:
            continue
        top = max(vals, key=lambda r: r["effect_norm"])
        max_effect = top["effect_norm"]
        early = next((r for r in sorted(vals, key=lambda x: x["layer"]) if r["effect_norm"] >= max_effect * 0.35), None)
        formation[kind] = {
            "max_site": top["site"],
            "max_layer": top["layer"],
            "max_effect_norm": top["effect_norm"],
            "earliest_35pct_site": early["site"] if early else None,
            "earliest_35pct_layer": early["layer"] if early else None,
            "earliest_35pct_effect_norm": early["effect_norm"] if early else None,
        }
    by_transfer: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in transfer_rows:
        by_transfer[(row["direction"], row["site"])].append(row)
    transfer_summary = {}
    for (direction, site), vals in by_transfer.items():
        transfer_summary[f"{direction}|{site}"] = {
            "site_kind": vals[0]["site_kind"],
            "layer": vals[0]["layer"],
            "n": len(vals),
            "mean_patched_delta_vs_recipient": mean([v["patched_delta_vs_recipient"] for v in vals]),
            "changed_rate": sum(1 for v in vals if v["changed_vs_recipient"]) / len(vals),
            "recipient_hit_rate": sum(1 for v in vals if v["recipient_hit"]) / len(vals),
            "patched_hit_rate": sum(1 for v in vals if v["patched_hit"]) / len(vals),
            "hit_gain": sum(1 for v in vals if (not v["recipient_hit"]) and v["patched_hit"]) / len(vals),
            "hit_loss": sum(1 for v in vals if v["recipient_hit"] and (not v["patched_hit"])) / len(vals),
        }
    return {
        "phase": 733,
        "title": "Prompt-Type Skeleton Source Localization",
        "model": model_name,
        "round": round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "quantization": "off",
        "dtype": "bfloat16",
        "n_scan_rows": len(scan_rows),
        "n_transfer_rows": len(transfer_rows),
        "candidate_sites": candidate_sites,
        "formation_summary": formation,
        "transfer_summary": transfer_summary,
        "strict_interpretation": "source localization v0; early effect formation is descriptive and site replacement remains causal but distribution-shifting",
    }


def run_model(args) -> dict[str, Any]:
    out_dir = OUT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    scan_cases = select_scan_cases(args.max_scan_cases)
    pairs = select_prompt_pairs(args.max_pairs)
    model, tokenizer, device, attn_impl = load_model_bf16_flash(args.model)
    try:
        log(f"{args.model}/{args.round_name}: scan_cases={len(scan_cases)}, pairs={len(pairs)}")
        scan_rows, scan_case_rows = scan_formation(model, tokenizer, device, scan_cases, args.log_every)
        candidate_sites = choose_candidate_sites(args.model, scan_rows, args.max_candidate_sites)
        log(f"{args.model}/{args.round_name}: candidate_sites={candidate_sites}")
        transfer_rows = transfer_validation(model, tokenizer, device, pairs, candidate_sites, args.max_new_tokens, args.log_every)
    finally:
        release_model(model)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    summary = summarize(args.model, args.round_name, attn_impl, scan_rows, transfer_rows, candidate_sites)
    write_jsonl(out_dir / f"phase733_{args.model}_scan_rows.jsonl", scan_rows)
    write_jsonl(out_dir / f"phase733_{args.model}_scan_case_rows.jsonl", scan_case_rows)
    write_jsonl(out_dir / f"phase733_{args.model}_transfer_rows.jsonl", transfer_rows)
    write_json(out_dir / f"phase733_{args.model}_summary.json", summary)
    print(json.dumps({"model": args.model, "round": args.round_name, "attn": attn_impl, "formation": summary["formation_summary"], "top_transfer": list(summary["transfer_summary"].items())[:4]}, ensure_ascii=False, indent=2), flush=True)
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
        phase_node = f"{model}:phase:733:{round_name}"
        add_node({"id": model_node, "type": "model", "label": model, "model": model, "position": [-16, 0, lane_z], "role": "tested_model"})
        add_node({"id": phase_node, "type": "phase", "label": f"Phase 733 {round_name}", "model": model, "position": [-12, 2, lane_z], "role": "source_localization"})
        edges.append({"source": model_node, "target": phase_node, "relation": "contains", "weight": 0, "phase": 733})
        for kind, rec in summary.get("formation_summary", {}).items():
            form_node = f"{model}:formation:{round_name}:{kind}:{rec.get('earliest_35pct_site')}"
            add_node(
                {
                    "id": form_node,
                    "type": "layer" if kind in {"layer_input", "layer_out"} else "cluster",
                    "label": f"{kind} earliest {rec.get('earliest_35pct_site')}",
                    "model": model,
                    "layer": rec.get("earliest_35pct_layer"),
                    "role": "prompt_type_formation_site",
                    "evidence_level": "mean_difference_source_scan",
                    "score": rec.get("earliest_35pct_effect_norm"),
                    "max_site": rec.get("max_site"),
                    "max_effect_norm": rec.get("max_effect_norm"),
                }
            )
            edges.append({"source": phase_node, "target": form_node, "relation": "candidate_of", "weight": rec.get("earliest_35pct_effect_norm") or 0, "phase": 733})
        for key, rec in summary.get("transfer_summary", {}).items():
            if "commonsense<-explicit" not in key:
                continue
            transfer_node = f"{model}:transfer:{round_name}:{key}"
            add_node(
                {
                    "id": transfer_node,
                    "type": "intervention",
                    "label": key,
                    "model": model,
                    "layer": rec.get("layer"),
                    "role": "early_site_transfer_validation",
                    "evidence_level": "causal_replacement",
                    "score": rec.get("mean_patched_delta_vs_recipient"),
                    "changed_rate": rec.get("changed_rate"),
                    "hit_gain": rec.get("hit_gain"),
                    "hit_loss": rec.get("hit_loss"),
                }
            )
            relation = "supports_likelihood" if (rec.get("mean_patched_delta_vs_recipient") or 0) > 0 else "negative_effect"
            if rec.get("changed_rate", 0) > 0:
                relation = "changes_generation"
            edges.append({"source": phase_node, "target": transfer_node, "relation": relation, "weight": abs(rec.get("mean_patched_delta_vs_recipient") or 0), "phase": 733})
    return {
        "schema_version": "atlas_graph_v1",
        "title": f"Phase 733 Prompt-Type Skeleton Source Localization ({round_name})",
        "model_info": {"model": "cross_model", "models": models, "phase": 733, "round": round_name, "timestamp": payload.get("timestamp"), "evidence_type": payload.get("evidence_type")},
        "layout": {"x": "formation / transfer", "y": "layer index", "z": "model lane"},
        "graph": {"nodes": nodes, "edges": edges},
        "metrics": {"node_count": len(nodes), "edge_count": len(edges), "source_phase": 733},
        "source_files": [str(OUT_ROOT / round_name / "phase733_cross_model_summary.json")],
    }


def write_cross_summary(round_name: str) -> dict[str, Any]:
    out_dir = OUT_ROOT / round_name
    summaries = []
    for model in MODELS:
        path = out_dir / f"phase733_{model}_summary.json"
        if path.exists():
            summaries.append(json.loads(path.read_text(encoding="utf-8")))
    payload = {
        "phase": 733,
        "title": "Prompt-Type Skeleton Source Localization",
        "round": round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": [s["model"] for s in summaries],
        "status": "complete" if len(summaries) == len(MODELS) else "partial",
        "evidence_type": "formation scan + early/mid/late site replacement",
        "by_model": {s["model"]: s for s in summaries},
    }
    write_json(out_dir / "phase733_cross_model_summary.json", payload)
    graph = build_atlas_graph(payload, round_name)
    write_json(out_dir / "phase733_atlas_graph.json", graph)
    lines = [
        f"# Phase 733 Prompt-Type Skeleton Source Localization ({round_name})",
        "",
        f"- Status: `{payload['status']}`",
        f"- Models: `{payload['models']}`",
        "- Evidence type: prompt-type formation scan + site replacement.",
        "",
        "| model | attn | earliest layer_out | top transfer | delta | changed | hit_gain | hit_loss |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for model, summary in payload["by_model"].items():
        layer_rec = summary.get("formation_summary", {}).get("layer_out", {})
        transfer_items = [(k, v) for k, v in summary.get("transfer_summary", {}).items() if "commonsense<-explicit" in k]
        best = max(transfer_items, key=lambda kv: abs(kv[1].get("mean_patched_delta_vs_recipient") or 0)) if transfer_items else ("-", {})
        lines.append(
            f"| {model} | {summary.get('attn_implementation')} | {layer_rec.get('earliest_35pct_site')} | "
            f"{best[0]} | {best[1].get('mean_patched_delta_vs_recipient', 0):.3f} | "
            f"{best[1].get('changed_rate', 0):.3f} | {best[1].get('hit_gain', 0):.3f} | {best[1].get('hit_loss', 0):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Strict Interpretation",
            "",
            "- Earliest layer is based on 35% of the model/kind maximum commonsense-vs-explicit effect.",
            "- Replacement validates causal influence at a site, but still may introduce distribution shift.",
            "- This phase localizes source candidates; it does not prove neuron-level writers.",
            "",
            f"Atlas graph: nodes={graph['metrics']['node_count']} edges={graph['metrics']['edge_count']}",
            "",
        ]
    )
    (out_dir / "phase733_cross_model_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"round": round_name, "status": payload["status"], "models": payload["models"], "atlas": graph["metrics"]}, ensure_ascii=False, indent=2), flush=True)
    return payload


def dry_run(args) -> None:
    cases = select_scan_cases(args.max_scan_cases)
    pairs = select_prompt_pairs(args.max_pairs)
    print(json.dumps({"round": args.round_name, "scan_cases": len(cases), "pairs": len(pairs), "scan_counts": dict(Counter(c["prompt_type"] for c in cases)), "pair_sample": pairs[:3]}, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="main")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-scan-cases", type=int, default=None)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--max-candidate-sites", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=8)
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
