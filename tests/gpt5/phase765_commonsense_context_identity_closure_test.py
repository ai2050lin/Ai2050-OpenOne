#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
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
from phase112_attention_transport_head_mapping_cuda import get_attention_module, get_num_heads  # noqa: E402
from phase132_source_value_contribution_cuda import compute_source_contribution, get_num_kv_heads, get_v_proj  # noqa: E402
from phase722_functional_head_atlas_causal_ablation import logit_diag, write_json, write_jsonl  # noqa: E402
from phase723_apple_fruit_attribute_micro_atlas import question_for  # noqa: E402
from phase735_source_restricted_writer_validation import (  # noqa: E402
    MODELS,
    bare_phrase_positions,
    char_span_positions,
    line_span_positions,
    load_model_bf16_eager,
    phrase_positions,
    safe_mean,
    select_evenly,
)
from phase739_readout_threshold_closure_boundary import get_unembed  # noqa: E402
from phase741_threshold_candidate_causal_validation import parse_component_site  # noqa: E402
from phase749_suppressor_component_decomposition import direct_delta_score  # noqa: E402
from phase751_natural_attention_head_mechanism_backtrace import (  # noqa: E402
    install_source_contribution_removal,
    project_source_contribution,
)
from phase752_natural_writer_stability_path_chain import attention_mass_for_group  # noqa: E402
from phase755_cross_domain_route_invariance_atlas import get_first_token_id  # noqa: E402
from phase756_cross_domain_writer_control_downstream_carrier import expanded_candidates, run_logits  # noqa: E402
from phase762_semantic_numeric_fiber_atlas import RELATION_KEYS, SEMANTIC_OBJECTS, VALUE_POOLS, alternate_value  # noqa: E402


OUT_ROOT = Path("results/glm5_phase765_commonsense_context_identity_closure_test")
CONTEXT_FORMATS = ["commonsense_question", "commonsense_statement"]
DEFAULT_SOURCE_GROUPS = ["instruction", "question", "object_tokens", "relation_tokens", "answer_prefix", "all_pre_answer"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def relation_key_map() -> dict[str, str]:
    return {relation: key for relation, key in RELATION_KEYS}


def relation_label(relation: str) -> str:
    if relation == "grows_on_tree":
        return "whether it grows on a tree"
    if relation == "edible":
        return "whether it is edible"
    return relation.replace("_", " ")


def allowed_values_line(case: dict[str, Any]) -> str:
    values = VALUE_POOLS[case["relation_key"]]
    return "Allowed values: " + ", ".join(values) + ".\n"


def prompt_for_case(case: dict[str, Any]) -> str:
    obj = case["object"]
    relation = case["relation"]
    allowed = allowed_values_line(case) if case.get("include_candidate_list", True) else ""
    if case["context_format"] == "commonsense_question":
        return (
            "Answer using common everyday knowledge.\n"
            "Use exactly one short value.\n"
            f"{allowed}"
            f"Question: {question_for(obj, relation)}\n"
            "Answer:"
        )
    if case["context_format"] == "commonsense_statement":
        return (
            "Use common everyday knowledge.\n"
            "Answer with exactly one short value.\n"
            f"{allowed}"
            f"Task: For {obj}, give {relation_label(relation)}.\n"
            "Answer:"
        )
    raise ValueError(case["context_format"])


def build_cases(max_cases: int | None = None, relation_filter: set[str] | None = None) -> list[dict[str, Any]]:
    rows = []
    key_by_relation = relation_key_map()
    cid = 0
    for obj in SEMANTIC_OBJECTS:
        for relation, key in RELATION_KEYS:
            if relation_filter and relation not in relation_filter:
                continue
            answer = str(obj[key])
            for context_format in CONTEXT_FORMATS:
                cid += 1
                rows.append(
                    {
                        "case_id": f"p765_{cid:04d}_{context_format}_{obj['domain']}:{obj['object']}:{relation}",
                        "context_format": context_format,
                        "object": obj["object"],
                        "domain": obj["domain"],
                        "relation": relation,
                        "relation_key": key,
                        "answer": answer,
                        "contrast_answer": alternate_value(key_by_relation[relation], answer),
                        "include_candidate_list": True,
                    }
                )
    if max_cases and max_cases < len(rows):
        return [rows[i] for i in select_evenly(len(rows), max_cases)]
    return rows


def source_groups_for(args: argparse.Namespace) -> list[str]:
    if args.source_groups:
        return [x.strip() for x in args.source_groups.split(",") if x.strip()]
    return DEFAULT_SOURCE_GROUPS[: args.max_source_groups]


def build_source_groups_custom(tokenizer, prompt: str, case: dict[str, Any], ids: list[int]) -> dict[str, list[int]]:
    answer_pos = len(ids) - 1
    obj = case["object"]
    relation = case["relation"]
    question = line_span_positions(tokenizer, prompt, lambda s: s.startswith("Question:") or s.startswith("Task:"))
    answer_prefix = line_span_positions(tokenizer, prompt, lambda s: s.startswith("Answer:"))
    instruction = line_span_positions(
        tokenizer,
        prompt,
        lambda s: (
            s.startswith("Answer using common everyday knowledge.")
            or s.startswith("Use exactly one short value.")
            or s.startswith("Use common everyday knowledge.")
            or s.startswith("Answer with exactly one short value.")
            or s.startswith("Allowed values:")
        ),
    )
    relation_phrases = [relation, relation.replace("_", " "), relation_label(relation)]
    if relation == "grows_on_tree":
        relation_phrases += ["tree", "grow", "grows"]
    if relation == "edible":
        relation_phrases += ["edible", "eat"]
    answer_hits = [p for p in bare_phrase_positions(tokenizer, ids, [case["answer"]]) if p < answer_pos]
    groups = {
        "instruction": instruction,
        "question": question,
        "object_tokens": phrase_positions(tokenizer, ids, [obj, obj.capitalize()]),
        "relation_tokens": phrase_positions(tokenizer, ids, relation_phrases),
        "target_value_tokens": answer_hits,
        "answer_prefix": [p for p in answer_prefix if p < answer_pos],
        "all_pre_answer": list(range(0, max(0, answer_pos))),
        "self_last": [answer_pos],
    }
    return {k: [p for p in sorted(set(v)) if 0 <= p < len(ids)] for k, v in groups.items()}


def capture_state(model, tokenizer, device, case: dict[str, Any], layers: list[int]) -> dict[str, Any]:
    prompt = prompt_for_case(case)
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    handles = []
    value_store: dict[int, torch.Tensor] = {}
    for li in sorted(set(layers)):
        attn = get_attention_module(get_layers(model)[li])
        v_proj = get_v_proj(attn)

        def v_hook(_module, _inputs, output, li=li):
            value_store[li] = output.detach().float().cpu()

        handles.append(v_proj.register_forward_hook(v_hook))
    try:
        with torch.inference_mode():
            out = model(
                input_ids=torch.tensor([ids], device=device),
                return_dict=True,
                use_cache=False,
                output_attentions=True,
            )
        attentions = {li: out.attentions[li].detach().float().cpu().numpy() for li in sorted(set(layers))}
        logits = out.logits[0, -1].detach().float().cpu()
    finally:
        for h in handles:
            h.remove()
    return {
        "ids": ids,
        "prompt": prompt,
        "answer_pos": len(ids) - 1,
        "logits": logits,
        "attentions": attentions,
        "values": value_store,
        "source_groups": build_source_groups_custom(tokenizer, prompt, case, ids),
    }


def route_ids_for_case(tokenizer, case: dict[str, Any], target_id: int) -> list[int]:
    pool = VALUE_POOLS[case["relation_key"]]
    ids = []
    for value in pool:
        tid = get_first_token_id(tokenizer, value)
        if tid != target_id and tid not in ids:
            ids.append(tid)
    return ids


def audit_case(
    model,
    tokenizer,
    device,
    case: dict[str, Any],
    candidates: list[dict[str, Any]],
    source_groups: list[str],
    unembed: torch.Tensor,
) -> list[dict[str, Any]]:
    candidate_layers = sorted({parse_component_site(c["site"])[0] for c in candidates})
    state = capture_state(model, tokenizer, device, case, candidate_layers)
    target_id = get_first_token_id(tokenizer, case["answer"])
    contrast_id = get_first_token_id(tokenizer, case["contrast_answer"])
    route_ids = route_ids_for_case(tokenizer, case, target_id)
    target_diag = logit_diag(state["logits"], target_id)
    contrast_diag = logit_diag(state["logits"], contrast_id)
    rows: list[dict[str, Any]] = [
        {
            "row_kind": "commonsense_task_observation",
            "case_id": case["case_id"],
            "context_format": case["context_format"],
            "object": case["object"],
            "domain": case["domain"],
            "relation": case["relation"],
            "target_answer": case["answer"],
            "contrast_answer": case["contrast_answer"],
            "target_rank": target_diag["target_rank"],
            "target_top1": target_diag["target_top1"],
            "contrast_rank": contrast_diag["target_rank"],
            "source_group_sizes": {k: len(v) for k, v in state["source_groups"].items()},
        }
    ]
    answer_pos = state["answer_pos"]
    for cand in candidates:
        site = cand["site"]
        layer, _component = parse_component_site(site)
        head = int(cand["head"])
        attn = get_attention_module(get_layers(model)[layer])
        n_heads = get_num_heads(model, attn)
        if not (0 <= head < n_heads):
            continue
        num_kv_heads = get_num_kv_heads(model, attn, n_heads)
        for source_group in source_groups:
            src_positions = [int(p) for p in state["source_groups"].get(source_group, [])]
            if not src_positions:
                continue
            contribution = compute_source_contribution(
                state["attentions"][layer],
                state["values"][layer],
                [answer_pos],
                [src_positions],
                n_heads,
                num_kv_heads,
            )
            projected = project_source_contribution(model, layer, [head], contribution)
            direct = direct_delta_score(projected, unembed, target_id, route_ids)
            removal_install = install_source_contribution_removal(model, site, [head], contribution)
            after_logits = run_logits(model, device, state["ids"], removal_install)
            rows.append(
                {
                    "row_kind": "commonsense_fiber_effect",
                    "case_id": case["case_id"],
                    "context_format": case["context_format"],
                    "object": case["object"],
                    "domain": case["domain"],
                    "relation": case["relation"],
                    "target_answer": case["answer"],
                    "contrast_answer": case["contrast_answer"],
                    "site": site,
                    "layer": layer,
                    "head": head,
                    "subunit_id": cand["subunit_id"],
                    "candidate_kind": cand["candidate_kind"],
                    "selection": cand["selection"],
                    "control_of": cand.get("control_of"),
                    "source_group": source_group,
                    "source_positions_n": len(src_positions),
                    "attention_mass_to_source": attention_mass_for_group(state["attentions"][layer], head, answer_pos, src_positions),
                    "source_direct_score": direct,
                    "target_logit_drop": float(state["logits"][target_id].item() - after_logits[target_id].item()),
                }
            )
    return rows


def add_feature(bucket: dict[str, list[float]], feature: str, value: Any) -> None:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return
    if math.isfinite(val):
        bucket[feature].append(val)


def mean_dict(values: dict[str, list[float]]) -> dict[str, float]:
    return {k: float(sum(v) / len(v)) for k, v in values.items() if v}


def build_vectors(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, float]]]:
    buckets: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in rows:
        if row.get("row_kind") != "commonsense_fiber_effect":
            continue
        obj = row["object"]
        fmt = row["context_format"]
        prefix = f"rel={row['relation']}|{row['subunit_id']}|{row['source_group']}"
        add_feature(buckets[fmt][obj], f"{prefix}|target_logit_drop", row.get("target_logit_drop"))
        add_feature(buckets[fmt][obj], f"{prefix}|attention_mass", row.get("attention_mass_to_source"))
        direct = row.get("source_direct_score") or {}
        add_feature(buckets[fmt][obj], f"{prefix}|direct_target_boost", direct.get("direct_target_boost"))
        add_feature(buckets[fmt][obj], f"{prefix}|direct_total_route_suppression", direct.get("direct_total_route_suppression"))
    return {fmt: {obj: mean_dict(feats) for obj, feats in obj_map.items()} for fmt, obj_map in buckets.items()}


def cosine(a: dict[str, float], b: dict[str, float], features: list[str]) -> float:
    dot = 0.0
    na = 0.0
    nb = 0.0
    for f in features:
        av = float(a.get(f, 0.0))
        bv = float(b.get(f, 0.0))
        dot += av * bv
        na += av * av
        nb += bv * bv
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    return dot / math.sqrt(na * nb)


def center_vectors(vectors: dict[str, dict[str, float]], features: list[str]) -> dict[str, dict[str, float]]:
    means = {f: safe_mean([v.get(f, 0.0) for v in vectors.values()]) or 0.0 for f in features}
    return {k: {f: v.get(f, 0.0) - means[f] for f in features} for k, v in vectors.items()}


def object_meta(cases: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    meta = {}
    for case in cases:
        meta[case["object"]] = {"object": case["object"], "domain": case["domain"]}
    return meta


def pair_summary(vectors: dict[str, dict[str, float]], meta: dict[str, dict[str, str]], features: list[str]) -> dict[str, Any]:
    objects = sorted(vectors)
    same = []
    diff = []
    neighbors: dict[str, tuple[str, float, bool]] = {}
    for i, a in enumerate(objects):
        for b in objects[i + 1 :]:
            sim = cosine(vectors[a], vectors[b], features)
            same_domain = meta[a]["domain"] == meta[b]["domain"]
            (same if same_domain else diff).append(sim)
            for x, y in [(a, b), (b, a)]:
                if x not in neighbors or sim > neighbors[x][1]:
                    neighbors[x] = (y, sim, same_domain)
    same_mean = safe_mean(same)
    diff_mean = safe_mean(diff)
    return {
        "same_domain_mean": same_mean,
        "different_domain_mean": diff_mean,
        "separation": (same_mean or 0.0) - (diff_mean or 0.0),
        "same_n": len(same),
        "different_n": len(diff),
        "nn_domain_accuracy": sum(1 for obj in objects if neighbors.get(obj, ("", 0.0, False))[2]) / len(objects) if objects else None,
    }


def context_stability(vectors_by_context: dict[str, dict[str, dict[str, float]]], meta: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    rows = []
    contexts = sorted(vectors_by_context)
    for i, a in enumerate(contexts):
        for b in contexts[i + 1 :]:
            common = sorted(set(vectors_by_context[a]) & set(vectors_by_context[b]))
            features = sorted(
                {f for obj in common for f in vectors_by_context[a][obj]}
                | {f for obj in common for f in vectors_by_context[b][obj]}
            )
            if not features:
                continue
            merged = {f"{a}:{obj}": vectors_by_context[a][obj] for obj in common}
            merged.update({f"{b}:{obj}": vectors_by_context[b][obj] for obj in common})
            centered = center_vectors(merged, features)
            same_object = []
            same_domain_diff_object = []
            diff_domain = []
            for obj_a in common:
                for obj_b in common:
                    sim = cosine(centered[f"{a}:{obj_a}"], centered[f"{b}:{obj_b}"], features)
                    if obj_a == obj_b:
                        same_object.append(sim)
                    elif meta[obj_a]["domain"] == meta[obj_b]["domain"]:
                        same_domain_diff_object.append(sim)
                    else:
                        diff_domain.append(sim)
            same_obj = safe_mean(same_object)
            same_domain = safe_mean(same_domain_diff_object)
            diff = safe_mean(diff_domain)
            rows.append(
                {
                    "context_a": a,
                    "context_b": b,
                    "common_objects": len(common),
                    "same_object_mean": same_obj,
                    "same_domain_diff_object_mean": same_domain,
                    "different_domain_mean": diff,
                    "object_stability_gap": (same_obj or 0.0) - (same_domain or 0.0),
                    "domain_stability_gap": (same_domain or 0.0) - (diff or 0.0),
                }
            )
    return rows


def observation_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    obs = [r for r in rows if r.get("row_kind") == "commonsense_task_observation"]
    by_context: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_relation: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in obs:
        by_context[row["context_format"]].append(row)
        by_relation[row["relation"]].append(row)

    def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "n": len(items),
            "target_top1_rate": safe_mean([1.0 if x["target_top1"] else 0.0 for x in items]),
            "mean_target_rank": safe_mean([x["target_rank"] for x in items]),
            "mean_contrast_rank": safe_mean([x["contrast_rank"] for x in items]),
        }

    return {
        "overall": summarize(obs),
        "by_context": {k: summarize(v) for k, v in sorted(by_context.items())},
        "by_relation": {k: summarize(v) for k, v in sorted(by_relation.items())},
    }


def build_summary(args: argparse.Namespace, rows: list[dict[str, Any]], cases: list[dict[str, Any]], candidates: list[dict[str, Any]], source_groups: list[str], attn_impl: str) -> dict[str, Any]:
    meta = object_meta(cases)
    vectors_by_context = build_vectors(rows)
    context_rows = {}
    for fmt, vectors in vectors_by_context.items():
        features = sorted({f for vec in vectors.values() for f in vec})
        centered = center_vectors(vectors, features)
        context_rows[fmt] = {
            "n_objects": len(vectors),
            "n_features": len(features),
            "pair_summary": pair_summary(centered, meta, features),
        }
    return {
        "phase": 765,
        "title": "Commonsense Context and Object Identity Closure Test",
        "model": args.model,
        "round": args.round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "dtype": "bfloat16",
        "quantization": "off",
        "n_rows": len(rows),
        "n_cases": sum(1 for r in rows if r.get("row_kind") == "commonsense_task_observation"),
        "n_effect_rows": sum(1 for r in rows if r.get("row_kind") == "commonsense_fiber_effect"),
        "context_formats": CONTEXT_FORMATS,
        "source_groups": source_groups,
        "candidates": candidates,
        "observation_summary": observation_summary(rows),
        "context_pair_summaries": context_rows,
        "cross_context_stability": context_stability(vectors_by_context, meta),
        "strict_interpretation": "Commonsense fibers test internal knowledge without explicit fact profiles. Low target-top1 rates or weak object stability mean no natural semantic closure.",
    }


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = OUT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    relation_filter = set(args.relations.split(",")) if args.relations else {"category", "edible", "grows_on_tree"}
    cases = build_cases(args.max_cases, relation_filter)
    source_groups = source_groups_for(args)
    for case in cases:
        case["include_candidate_list"] = bool(args.include_candidate_list)
    log(f"{args.model}/{args.round_name}: cases={len(cases)} sources={source_groups} relations={sorted(relation_filter)}")
    model, tokenizer, device, attn_impl = load_model_bf16_eager(args.model)
    try:
        candidates = expanded_candidates(model, args.model, args)
        unembed = get_unembed(model)
        rows: list[dict[str, Any]] = []
        for idx, case in enumerate(cases, 1):
            rows.extend(audit_case(model, tokenizer, device, case, candidates, source_groups, unembed))
            if idx % args.log_every == 0 or idx == len(cases):
                log(f"{args.model}: commonsense fibers {idx}/{len(cases)} cases; rows={len(rows)}")
    finally:
        release_model(model)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    summary = build_summary(args, rows, cases, candidates, source_groups, attn_impl)
    write_jsonl(out_dir / f"phase765_{args.model}_rows.jsonl", rows)
    write_json(out_dir / f"phase765_{args.model}_summary.json", summary)
    print(
        json.dumps(
            {
                "model": args.model,
                "round": args.round_name,
                "n_cases": summary["n_cases"],
                "observation_summary": summary["observation_summary"],
                "context_pair_summaries": summary["context_pair_summaries"],
                "cross_context_stability": summary["cross_context_stability"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return summary


def fmt(value: Any) -> str:
    if value is None:
        return "null"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def write_cross_summary(round_name: str) -> dict[str, Any]:
    out_dir = OUT_ROOT / round_name
    summaries = []
    for model_name in MODELS:
        path = out_dir / f"phase765_{model_name}_summary.json"
        if path.exists():
            summaries.append(json.loads(path.read_text(encoding="utf-8")))
    payload = {
        "phase": 765,
        "title": "Commonsense Context and Object Identity Closure Test",
        "round": round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "complete" if len(summaries) == len(MODELS) else "partial",
        "models": [s["model"] for s in summaries],
        "by_model": {s["model"]: s for s in summaries},
    }
    write_json(out_dir / "phase765_cross_model_summary.json", payload)
    lines = [
        f"# Phase 765 Commonsense Context and Object Identity Closure Test ({round_name})",
        "",
        f"- Status: `{payload['status']}`",
        "- Test: no explicit fact profile; use commonsense prompts only.",
        "",
        "## Base Answer Reliability",
        "",
        "| model | top1 rate | mean target rank | mean contrast rank |",
        "|---|---:|---:|---:|",
    ]
    for summary in summaries:
        obs = summary["observation_summary"]["overall"]
        lines.append(f"| {summary['model']} | {fmt(obs['target_top1_rate'])} | {fmt(obs['mean_target_rank'])} | {fmt(obs['mean_contrast_rank'])} |")
    lines += [
        "",
        "## Commonsense Domain Separation",
        "",
        "| model | context | cases | features | NN | same | diff | sep |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        for ctx, row in summary["context_pair_summaries"].items():
            ps = row["pair_summary"]
            lines.append(
                f"| {summary['model']} | `{ctx}` | {summary['n_cases']} | {row['n_features']} | "
                f"{fmt(ps['nn_domain_accuracy'])} | {fmt(ps['same_domain_mean'])} | "
                f"{fmt(ps['different_domain_mean'])} | {fmt(ps['separation'])} |"
            )
    lines += [
        "",
        "## Cross-Context Stability",
        "",
        "| model | context pair | same object | same-domain other | diff-domain | object gap | domain gap |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        for row in summary["cross_context_stability"]:
            lines.append(
                f"| {summary['model']} | `{row['context_a']}__{row['context_b']}` | "
                f"{fmt(row['same_object_mean'])} | {fmt(row['same_domain_diff_object_mean'])} | "
                f"{fmt(row['different_domain_mean'])} | {fmt(row['object_stability_gap'])} | "
                f"{fmt(row['domain_stability_gap'])} |"
            )
    lines += [
        "",
        "## Strict Interpretation",
        "",
        "- If target-top1 is low, the commonsense prompt does not form a reliable target state and causal fibers are hard to interpret.",
        "- A natural semantic closure result requires positive domain separation and positive same-object stability across commonsense prompt formats.",
    ]
    (out_dir / "phase765_cross_model_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="smoke")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--max-candidates", type=int, default=1)
    parser.add_argument("--include-controls", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--controls-per-candidate", type=int, default=1)
    parser.add_argument("--control-offset", type=int, default=3)
    parser.add_argument("--max-source-groups", type=int, default=6)
    parser.add_argument("--source-groups", default="")
    parser.add_argument("--relations", default="category,edible,grows_on_tree")
    parser.add_argument("--include-candidate-list", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-every", type=int, default=18)
    parser.add_argument("--write-cross-summary", action="store_true")
    parser.add_argument("--hard-exit-after-model", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        cases = build_cases(args.max_cases, set(args.relations.split(",")) if args.relations else None)
        print(json.dumps({"n_cases": len(cases), "sample_cases": cases[:6], "source_groups": source_groups_for(args)}, ensure_ascii=False, indent=2))
        return
    if args.write_cross_summary:
        write_cross_summary(args.round_name)
        return
    if not args.model:
        raise SystemExit("--model is required unless --write-cross-summary or --dry-run")
    run_model(args)
    if args.hard_exit_after_model:
        raise SystemExit(0)


if __name__ == "__main__":
    main()
