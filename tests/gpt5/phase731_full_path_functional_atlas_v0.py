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
from phase721_global_functional_head_atlas_expansion import (  # noqa: E402
    all_occurrences,
    positions_for_char_spans,
)
from phase722_functional_head_atlas_causal_ablation import (  # noqa: E402
    logit_diag,
    target_token_ids,
    write_json,
    write_jsonl,
)
from phase723_apple_fruit_attribute_micro_atlas import build_cases, prompt_for  # noqa: E402
from phase727_category_fruit_cluster_intervention import hit_answer, norm_text  # noqa: E402


OUT_ROOT = Path("results/glm5_phase731_full_path_functional_atlas_v0")
MODELS = ["qwen3", "glm4", "deepseek7b"]
RELATION_SET = {"category", "color", "taste"}

FALLBACK_HEADS = {
    "qwen3": [(28, 0), (24, 29), (26, 26)],
    "glm4": [(24, 19), (29, 28), (29, 18)],
    "deepseek7b": [(20, 17), (27, 23), (23, 0)],
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


def get_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, tuple):
        return output[0]
    return output


def select_cases(max_cases: int | None) -> list[dict[str, Any]]:
    selected = [
        c for c in build_cases(None)
        if c["relation"] in RELATION_SET
        and c["prompt_type"] in {"explicit_profile", "conflict_profile", "commonsense"}
    ]
    # Keep the v0 atlas balanced enough to expose shared/delta structure:
    # explicit 36 + conflict 12 + commonsense 18 = 66 cases before optional truncation.
    return selected[:max_cases] if max_cases else selected


def line_spans(prompt: str) -> dict[str, tuple[int, int]]:
    q = prompt.find("Question:")
    a = prompt.find("Answer:")
    if q < 0:
        q = 0
    if a < 0:
        a = len(prompt)
    first_use = prompt.find("Use ")
    if first_use < 0 or first_use > q:
        first_use = q
    return {
        "record_line": (0, first_use),
        "instruction_line": (first_use, q),
        "question_line": (q, a),
        "answer_line": (a, len(prompt)),
    }


def find_subseq_positions(haystack: list[int], needle: list[int]) -> list[int]:
    if not needle:
        return []
    out: set[int] = set()
    n = len(needle)
    for i in range(0, len(haystack) - n + 1):
        if haystack[i:i + n] == needle:
            out.update(range(i, i + n))
    return sorted(out)


def token_positions_for_text(tokenizer, ids: list[int], text: str) -> list[int]:
    if not text:
        return []
    out: set[int] = set()
    for variant in [text, " " + text, "\n" + text, text + "\n"]:
        toks = tokenizer.encode(variant, add_special_tokens=False)
        out.update(find_subseq_positions(ids, toks))
    return sorted(out)


def token_groups(tokenizer, prompt: str, case: dict[str, Any], ids: list[int]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    try:
        enc = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
        offsets = [(int(a), int(b)) for a, b in enc["offset_mapping"]]
        for name, span in line_spans(prompt).items():
            groups[name] = positions_for_char_spans(offsets, [span])
        groups["object_name"] = positions_for_char_spans(offsets, all_occurrences(prompt, case["object"]))
        groups["relation_name"] = positions_for_char_spans(offsets, all_occurrences(prompt, case["relation"]))
        groups["target_value"] = positions_for_char_spans(offsets, all_occurrences(prompt, case["answer"]))
    except Exception:
        for name, span in line_spans(prompt).items():
            groups[name] = token_positions_for_text(tokenizer, ids, prompt[span[0]:span[1]])
        groups["object_name"] = token_positions_for_text(tokenizer, ids, case["object"])
        groups["relation_name"] = token_positions_for_text(tokenizer, ids, case["relation"])
        groups["target_value"] = token_positions_for_text(tokenizer, ids, case["answer"])
    groups["source_value"] = groups.get("target_value", [])
    groups["self_last"] = [len(ids) - 1]
    return groups


def group_mass(head_row: torch.Tensor, groups: dict[str, list[int]]) -> dict[str, float]:
    n = head_row.numel()
    out = {}
    for name in [
        "record_line",
        "instruction_line",
        "question_line",
        "answer_line",
        "object_name",
        "relation_name",
        "target_value",
        "source_value",
        "self_last",
    ]:
        idxs = [i for i in groups.get(name, []) if 0 <= i < n]
        out[f"mass_{name}"] = float(head_row[idxs].sum().detach().float().cpu().item()) if idxs else 0.0
    return out


def load_candidate_heads(model_name: str, top_k: int) -> list[dict[str, Any]]:
    heads: list[dict[str, Any]] = []
    phase721 = Path("results/glm5_phase721_global_functional_head_atlas_expansion") / f"phase721_{model_name}_head_scores.json"
    if phase721.exists():
        data = json.loads(phase721.read_text(encoding="utf-8"))
        seen: set[tuple[int, int]] = set()
        for family in ["fruit_identity_reuse_difference", "color_value_reuse_difference"]:
            for row in data.get("by_family", {}).get(family, {}).get("top_source_focus_heads", [])[:top_k]:
                key = (int(row["layer"]), int(row["head"]))
                if key in seen:
                    continue
                seen.add(key)
                heads.append(
                    {
                        "layer": key[0],
                        "head": key[1],
                        "head_key": f"L{key[0]}H{key[1]}",
                        "source": "phase721",
                        "family": family,
                        "source_focus_score": row.get("source_focus_score"),
                    }
                )
                if len(heads) >= top_k:
                    break
            if len(heads) >= top_k:
                break
    seen = {(h["layer"], h["head"]) for h in heads}
    for layer, head in FALLBACK_HEADS[model_name]:
        if (layer, head) not in seen:
            heads.append({"layer": layer, "head": head, "head_key": f"L{layer}H{head}", "source": "fallback"})
        if len(heads) >= top_k:
            break
    return heads[:top_k]


def capture_forward(
    model,
    tokenizer,
    device,
    case: dict[str, Any],
    prompt: str,
    capture_layers: list[int],
    candidate_heads: list[dict[str, Any]],
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]], list[int]]:
    layers = get_layers(model)
    captured: dict[str, torch.Tensor] = {}
    handles = []

    def make_layer_input_hook(li: int):
        def hook(_module, inputs):
            captured[f"L{li}_layer_input"] = inputs[0][0, -1].detach().float().cpu()
        return hook

    def make_component_hook(li: int, component: str):
        def hook(_module, _inputs, output):
            captured[f"L{li}_{component}"] = get_tensor(output)[0, -1].detach().float().cpu()
        return hook

    try:
        for li in capture_layers:
            if 0 <= li < len(layers):
                handles.append(layers[li].register_forward_pre_hook(make_layer_input_hook(li)))
                if hasattr(layers[li], "self_attn"):
                    handles.append(layers[li].self_attn.register_forward_hook(make_component_hook(li, "attn_out")))
                if hasattr(layers[li], "mlp"):
                    handles.append(layers[li].mlp.register_forward_hook(make_component_hook(li, "mlp_out")))
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        with torch.inference_mode():
            out = model(
                input_ids=torch.tensor([ids], device=device),
                return_dict=True,
                use_cache=False,
                output_hidden_states=True,
                output_attentions=True,
            )
        for hs_idx, hs in enumerate(out.hidden_states):
            captured[f"hidden_{hs_idx}"] = hs[0, -1].detach().float().cpu()
        captured["final_hidden"] = out.hidden_states[-1][0, -1].detach().float().cpu()
        captured["final_logits"] = out.logits[0, -1].detach().float().cpu()

        attn_rows: list[dict[str, Any]] = []
        if out.attentions is not None:
            groups = token_groups(tokenizer, prompt, case, ids)
            answer_pos = len(ids) - 1
            for h in candidate_heads:
                layer = int(h["layer"])
                head = int(h["head"])
                if 0 <= layer < len(out.attentions) and head < out.attentions[layer].shape[1]:
                    row = out.attentions[layer][0, head, answer_pos, :].detach()
                    top_pos = int(torch.argmax(row).detach().cpu().item())
                    attn_rows.append(
                        {
                            "layer": layer,
                            "head": head,
                            "head_key": h["head_key"],
                            "head_source": h.get("source"),
                            "top_attn_pos": top_pos,
                            "top_attn_token": tokenizer.decode([ids[top_pos]]),
                            "top_attn_mass": float(row[top_pos].detach().float().cpu().item()),
                            **group_mass(row, groups),
                        }
                    )
        return captured, attn_rows, ids
    finally:
        for h in handles:
            h.remove()


def phrase_diag(model, tokenizer, device, prompt: str, answer: str) -> dict[str, Any]:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    ans_ids = target_token_ids(tokenizer, answer)
    cur = list(prompt_ids)
    diags = []
    with torch.inference_mode():
        for tid in ans_ids:
            out = model(input_ids=torch.tensor([cur], device=device), return_dict=True, use_cache=False)
            diag = logit_diag(out.logits[0, -1].detach().float().cpu(), int(tid))
            diags.append(diag)
            cur.append(int(tid))
    return {
        "answer_token_ids": [int(x) for x in ans_ids],
        "answer_token_texts": [tokenizer.decode([int(x)]) for x in ans_ids],
        "mean_logprob": sum(d["target_logprob"] for d in diags) / len(diags),
        "sum_logprob": sum(d["target_logprob"] for d in diags),
        "first_rank": diags[0]["target_rank"],
        "first_margin": diags[0]["margin_vs_best_other"],
        "first_top1": diags[0]["target_top1"],
    }


def greedy_generate(model, tokenizer, device, prompt: str, max_new_tokens: int) -> dict[str, Any]:
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    new_ids = []
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            out = model(input_ids=torch.tensor([ids], device=device), return_dict=True, use_cache=False)
            tok = int(torch.argmax(out.logits[0, -1]).item())
            new_ids.append(tok)
            ids.append(tok)
            text = tokenizer.decode(new_ids, skip_special_tokens=True)
            if "\n" in text or "." in text or ";" in text:
                break
    return {"text": tokenizer.decode(new_ids, skip_special_tokens=True).strip(), "token_ids": new_ids}


def site_rows_from_capture(case: dict[str, Any], cap: dict[str, torch.Tensor], capture_layers: list[int]) -> list[dict[str, Any]]:
    rows = []
    prev_hidden = None
    final_hidden = cap["final_hidden"]
    hidden_keys = sorted([k for k in cap if k.startswith("hidden_")], key=lambda x: int(x.split("_", 1)[1]))
    for key in hidden_keys:
        vec = cap[key]
        hs_idx = int(key.split("_", 1)[1])
        delta_prev = norm(vec - prev_hidden) if prev_hidden is not None else 0.0
        rows.append(
            {
                "case_id": case["case_id"],
                "site": key,
                "site_kind": "hidden",
                "layer": hs_idx - 1,
                "hidden_index": hs_idx,
                "vector_norm": norm(vec),
                "delta_from_prev_hidden_norm": delta_prev,
                "cos_with_final_hidden": cosine(vec, final_hidden),
                "object": case["object"],
                "object_group": case["object_group"],
                "relation": case["relation"],
                "prompt_type": case["prompt_type"],
                "answer": case["answer"],
            }
        )
        prev_hidden = vec
    for li in capture_layers:
        input_key = f"L{li}_layer_input"
        input_vec = cap.get(input_key)
        input_norm = norm(input_vec) if input_vec is not None else 0.0
        for component in ["attn_out", "mlp_out"]:
            key = f"L{li}_{component}"
            if key not in cap:
                continue
            vec = cap[key]
            rows.append(
                {
                    "case_id": case["case_id"],
                    "site": key,
                    "site_kind": component,
                    "layer": li,
                    "vector_norm": norm(vec),
                    "component_vs_layer_input": norm(vec) / max(input_norm, 1e-9),
                    "cos_with_final_hidden": cosine(vec, final_hidden),
                    "object": case["object"],
                    "object_group": case["object_group"],
                    "relation": case["relation"],
                    "prompt_type": case["prompt_type"],
                    "answer": case["answer"],
                }
            )
    return rows


def mean(vals: list[float | None]) -> float | None:
    xs = [float(v) for v in vals if v is not None]
    return sum(xs) / len(xs) if xs else None


def factor_effects_from_vectors(vector_rows: list[dict[str, Any]], vector_cache: dict[tuple[str, str], torch.Tensor]) -> list[dict[str, Any]]:
    by_site: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in vector_rows:
        by_site[row["site"]].append(row)
    out = []
    for site, rows in by_site.items():
        vecs = [vector_cache[(r["case_id"], site)] for r in rows if (r["case_id"], site) in vector_cache]
        if not vecs:
            continue
        global_mean = torch.stack(vecs).mean(dim=0)
        for factor in ["object_group", "relation", "prompt_type"]:
            levels = sorted({r[factor] for r in rows})
            for level in levels:
                level_rows = [r for r in rows if r[factor] == level and (r["case_id"], site) in vector_cache]
                if not level_rows:
                    continue
                level_mean = torch.stack([vector_cache[(r["case_id"], site)] for r in level_rows]).mean(dim=0)
                effect = level_mean - global_mean
                out.append(
                    {
                        "site": site,
                        "site_kind": level_rows[0]["site_kind"],
                        "layer": level_rows[0]["layer"],
                        "factor": factor,
                        "level": level,
                        "n": len(level_rows),
                        "effect_norm": norm(effect),
                        "global_mean_norm": norm(global_mean),
                        "level_mean_norm": norm(level_mean),
                    }
                )
    return out


def summarize_model(
    model_name: str,
    case_rows: list[dict[str, Any]],
    trajectory_rows: list[dict[str, Any]],
    attention_rows: list[dict[str, Any]],
    factor_rows: list[dict[str, Any]],
    candidate_heads: list[dict[str, Any]],
) -> dict[str, Any]:
    by_relation = {}
    for relation in sorted({r["relation"] for r in case_rows}):
        vals = [r for r in case_rows if r["relation"] == relation]
        by_relation[relation] = {
            "n": len(vals),
            "hit_rate": sum(1 for r in vals if r["hit"]) / len(vals),
            "mean_logprob": mean([r["mean_logprob"] for r in vals]),
            "mean_first_rank": mean([r["first_rank"] for r in vals]),
            "outputs": Counter(norm_text(r["generated_text"]) for r in vals).most_common(8),
        }
    by_prompt_type = {}
    for prompt_type in sorted({r["prompt_type"] for r in case_rows}):
        vals = [r for r in case_rows if r["prompt_type"] == prompt_type]
        by_prompt_type[prompt_type] = {
            "n": len(vals),
            "hit_rate": sum(1 for r in vals if r["hit"]) / len(vals),
            "mean_logprob": mean([r["mean_logprob"] for r in vals]),
        }
    top_factor_effects = sorted(factor_rows, key=lambda r: r["effect_norm"], reverse=True)[:60]
    attn_grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in attention_rows:
        attn_grouped[row["head_key"]].append(row)
    head_summaries = []
    for head_key, vals in sorted(attn_grouped.items()):
        head_summaries.append(
            {
                "head_key": head_key,
                "layer": vals[0]["layer"],
                "head": vals[0]["head"],
                "n": len(vals),
                "mean_mass_target_value": mean([v["mass_target_value"] for v in vals]),
                "mean_mass_object_name": mean([v["mass_object_name"] for v in vals]),
                "mean_mass_relation_name": mean([v["mass_relation_name"] for v in vals]),
                "mean_mass_record_line": mean([v["mass_record_line"] for v in vals]),
                "mean_mass_question_line": mean([v["mass_question_line"] for v in vals]),
                "mean_top_attn_mass": mean([v["top_attn_mass"] for v in vals]),
                "top_tokens": Counter(str(v["top_attn_token"]) for v in vals).most_common(8),
            }
        )
    return {
        "phase": 731,
        "title": "Full-Path Functional Atlas v0",
        "model": model_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_cases": len(case_rows),
        "n_trajectory_rows": len(trajectory_rows),
        "n_attention_rows": len(attention_rows),
        "candidate_heads": candidate_heads,
        "by_relation": by_relation,
        "by_prompt_type": by_prompt_type,
        "top_factor_effects": top_factor_effects,
        "candidate_head_attention_summary": sorted(
            head_summaries,
            key=lambda r: (r.get("mean_mass_target_value") or 0) + 0.5 * (r.get("mean_mass_object_name") or 0),
            reverse=True,
        ),
        "strict_interpretation": "absolute trajectory + factor mean-difference atlas; factor effects are descriptive, not causal proof",
    }


def run_model(args) -> dict[str, Any]:
    cases = select_cases(args.max_cases)
    model, tokenizer, device = load_model(args.model)
    layers = get_layers(model)
    source_guess = min(max(0, args.source_layer), len(layers) - 1)
    capture_layers = sorted(set([source_guess, min(source_guess + 4, len(layers) - 1), min(source_guess + 7, len(layers) - 1)]))
    candidate_heads = load_candidate_heads(args.model, args.top_heads)
    log(f"{args.model}: cases={len(cases)}, capture_layers={capture_layers}, candidate_heads={[h['head_key'] for h in candidate_heads]}")

    case_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    attention_rows: list[dict[str, Any]] = []
    vector_cache: dict[tuple[str, str], torch.Tensor] = {}
    try:
        for idx, case in enumerate(cases, 1):
            prompt = prompt_for(case)
            cap, attn_rows, ids = capture_forward(model, tokenizer, device, case, prompt, capture_layers, candidate_heads)
            pdiag = phrase_diag(model, tokenizer, device, prompt, case["answer"])
            gen = greedy_generate(model, tokenizer, device, prompt, args.max_new_tokens)
            hit = hit_answer(gen["text"], case["answer"])
            case_rows.append(
                {
                    "model": args.model,
                    "case_id": case["case_id"],
                    "prompt_type": case["prompt_type"],
                    "object": case["object"],
                    "object_group": case["object_group"],
                    "relation": case["relation"],
                    "answer": case["answer"],
                    "seq_len": len(ids),
                    "mean_logprob": pdiag["mean_logprob"],
                    "sum_logprob": pdiag["sum_logprob"],
                    "first_rank": pdiag["first_rank"],
                    "first_margin": pdiag["first_margin"],
                    "first_top1": pdiag["first_top1"],
                    "generated_text": gen["text"],
                    "hit": hit,
                }
            )
            srows = site_rows_from_capture(case, cap, capture_layers)
            trajectory_rows.extend(srows)
            for srow in srows:
                if srow["site_kind"] in {"hidden", "attn_out", "mlp_out"} and srow["site"] in cap:
                    vector_cache[(case["case_id"], srow["site"])] = cap[srow["site"]]
            for row in attn_rows:
                row.update(
                    {
                        "model": args.model,
                        "case_id": case["case_id"],
                        "prompt_type": case["prompt_type"],
                        "object": case["object"],
                        "object_group": case["object_group"],
                        "relation": case["relation"],
                        "answer": case["answer"],
                    }
                )
                attention_rows.append(row)
            if idx % args.log_every == 0 or idx == len(cases):
                log(f"{args.model}: {idx}/{len(cases)} cases; trajectory_rows={len(trajectory_rows)} attention_rows={len(attention_rows)}")
    finally:
        release_model(model)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    factor_rows = factor_effects_from_vectors(trajectory_rows, vector_cache)
    summary = summarize_model(args.model, case_rows, trajectory_rows, attention_rows, factor_rows, candidate_heads)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT_ROOT / f"phase731_{args.model}_case_summary.jsonl", case_rows)
    write_jsonl(OUT_ROOT / f"phase731_{args.model}_trajectory_rows.jsonl", trajectory_rows)
    write_jsonl(OUT_ROOT / f"phase731_{args.model}_attention_rows.jsonl", attention_rows)
    write_jsonl(OUT_ROOT / f"phase731_{args.model}_factor_effect_rows.jsonl", factor_rows)
    write_json(OUT_ROOT / f"phase731_{args.model}_summary.json", summary)
    print(json.dumps({"model": args.model, "n_cases": summary["n_cases"], "by_relation": summary["by_relation"]}, ensure_ascii=False, indent=2), flush=True)
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
        model_node = f"{model}:model"
        phase_node = f"{model}:phase:731"
        add_node({"id": model_node, "type": "model", "label": model, "model": model, "position": [-16, 0, lane_z], "role": "tested_model"})
        add_node({"id": phase_node, "type": "phase", "label": "Phase 731", "model": model, "position": [-12, 2, lane_z], "role": "full_path_atlas_v0"})
        edges.append({"source": model_node, "target": phase_node, "relation": "contains", "weight": 0, "phase": 731})
        summary = payload["by_model"][model]
        for relation, rec in summary.get("by_relation", {}).items():
            rel_node = f"{model}:relation:{relation}"
            add_node(
                {
                    "id": rel_node,
                    "type": "concept",
                    "label": relation,
                    "model": model,
                    "role": "relation_route",
                    "evidence_level": "absolute_generation_summary",
                    "hit_rate": rec.get("hit_rate"),
                    "mean_logprob": rec.get("mean_logprob"),
                    "position": [-6, 6 + len(nodes) % 4, lane_z],
                }
            )
            edges.append({"source": phase_node, "target": rel_node, "relation": "contains", "weight": rec.get("hit_rate") or 0, "phase": 731})
        for row in summary.get("top_factor_effects", [])[:18]:
            site = row["site"]
            site_node = f"{model}:site:{site}"
            factor_node = f"{model}:factor:{row['factor']}:{row['level']}"
            add_node(
                {
                    "id": site_node,
                    "type": "layer" if row.get("site_kind") == "hidden" else "cluster",
                    "label": site,
                    "model": model,
                    "layer": row.get("layer"),
                    "role": "absolute_trajectory_site",
                    "evidence_level": "factor_mean_difference",
                    "score": row.get("effect_norm"),
                }
            )
            add_node(
                {
                    "id": factor_node,
                    "type": "concept",
                    "label": f"{row['factor']}={row['level']}",
                    "model": model,
                    "role": "factor_branch",
                    "evidence_level": "descriptive_factor_effect",
                    "score": row.get("effect_norm"),
                }
            )
            edges.append(
                {
                    "source": factor_node,
                    "target": site_node,
                    "relation": "factor_effect",
                    "weight": row.get("effect_norm") or 0,
                    "phase": 731,
                    "evidence": "absolute_trajectory_factor_mean_difference",
                }
            )
        for head in summary.get("candidate_head_attention_summary", [])[:6]:
            head_node = f"{model}:head:{head['head_key']}"
            head_score = (head.get("mean_mass_target_value") or 0) + 0.5 * (head.get("mean_mass_object_name") or 0)
            add_node(
                {
                    "id": head_node,
                    "type": "head",
                    "label": head["head_key"],
                    "model": model,
                    "layer": head.get("layer"),
                    "head": head.get("head"),
                    "role": "candidate_route_head",
                    "evidence_level": "observational_attention",
                    "mean_mass_target_value": head.get("mean_mass_target_value"),
                    "mean_mass_object_name": head.get("mean_mass_object_name"),
                    "score": head_score,
                }
            )
            edges.append({"source": phase_node, "target": head_node, "relation": "candidate_of", "weight": head_score, "phase": 731})
    return {
        "schema_version": "atlas_graph_v1",
        "title": "Phase 731 Full-Path Functional Atlas v0",
        "model_info": {
            "model": "cross_model",
            "models": models,
            "phase": 731,
            "timestamp": payload.get("timestamp"),
            "evidence_type": payload.get("evidence_type"),
        },
        "layout": {"x": "factor / route / site", "y": "layer index", "z": "model lane"},
        "graph": {"nodes": nodes, "edges": edges},
        "metrics": {"node_count": len(nodes), "edge_count": len(edges), "source_phase": 731},
        "source_files": [str(OUT_ROOT / "phase731_cross_model_summary.json")],
    }


def write_cross_summary() -> dict[str, Any]:
    summaries = []
    for model in MODELS:
        path = OUT_ROOT / f"phase731_{model}_summary.json"
        if path.exists():
            summaries.append(json.loads(path.read_text(encoding="utf-8")))
    payload = {
        "phase": 731,
        "title": "Full-Path Functional Atlas v0",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": [s["model"] for s in summaries],
        "status": "complete" if len(summaries) == len(MODELS) else "partial",
        "evidence_type": "absolute trajectory + descriptive factor mean-difference + candidate-head attention",
        "small_model_caution": "v0 atlas is descriptive and small-model-specific; factor edges require later causal validation",
        "by_model": {s["model"]: s for s in summaries},
    }
    write_json(OUT_ROOT / "phase731_cross_model_summary.json", payload)
    graph = build_atlas_graph(payload)
    write_json(OUT_ROOT / "phase731_atlas_graph.json", graph)
    lines = [
        "# Phase 731 Full-Path Functional Atlas v0",
        "",
        f"- Status: `{payload['status']}`",
        f"- Models: `{payload['models']}`",
        "- Evidence type: absolute trajectory + factor mean-difference + candidate-head attention.",
        "",
        "| model | cases | category hit | color hit | taste hit | top factor effect | effect norm |",
        "|---|---:|---:|---:|---:|---|---:|",
    ]
    for model, summary in payload["by_model"].items():
        by_rel = summary.get("by_relation", {})
        top = summary.get("top_factor_effects", [{}])[0] if summary.get("top_factor_effects") else {}
        lines.append(
            f"| {model} | {summary.get('n_cases', 0)} | "
            f"{by_rel.get('category', {}).get('hit_rate', 0):.3f} | "
            f"{by_rel.get('color', {}).get('hit_rate', 0):.3f} | "
            f"{by_rel.get('taste', {}).get('hit_rate', 0):.3f} | "
            f"{top.get('factor', '-')}/{top.get('level', '-')}@{top.get('site', '-')} | "
            f"{top.get('effect_norm', 0):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Strict Interpretation",
            "",
            "- This is a v0 full-path descriptive atlas, not a causal closure proof.",
            "- Factor effects are mean-vector differences against the global centroid.",
            "- Candidate-head attention is observational; causal edge validation must follow.",
            "",
            f"Atlas graph: nodes={graph['metrics']['node_count']} edges={graph['metrics']['edge_count']}",
            "",
        ]
    )
    (OUT_ROOT / "phase731_cross_model_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "models": payload["models"], "atlas": graph["metrics"]}, ensure_ascii=False, indent=2), flush=True)
    return payload


def dry_run(max_cases: int | None) -> None:
    cases = select_cases(max_cases)
    print(json.dumps({"n": len(cases), "by_relation": dict(Counter(c["relation"] for c in cases)), "by_prompt_type": dict(Counter(c["prompt_type"] for c in cases)), "sample": cases[:6]}, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--source-layer", type=int, default=20)
    parser.add_argument("--top-heads", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=8)
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        dry_run(args.max_cases)
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
