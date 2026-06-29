#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
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

from model_utils import get_layers, release_model  # noqa: E402
from phase112_attention_transport_head_mapping_cuda import get_attention_module, get_num_heads  # noqa: E402
from phase132_source_value_contribution_cuda import compute_source_contribution, get_num_kv_heads  # noqa: E402
from phase722_functional_head_atlas_causal_ablation import logit_diag, write_json, write_jsonl  # noqa: E402
from phase723_apple_fruit_attribute_micro_atlas import prompt_for, records_for  # noqa: E402
from phase735_source_restricted_writer_validation import MODELS, load_model_bf16_eager, safe_mean, select_evenly  # noqa: E402
from phase739_readout_threshold_closure_boundary import get_unembed  # noqa: E402
from phase741_threshold_candidate_causal_validation import parse_component_site  # noqa: E402
from phase749_suppressor_component_decomposition import direct_delta_score  # noqa: E402
from phase751_natural_attention_head_mechanism_backtrace import (  # noqa: E402
    capture_attention_value_state,
    install_source_contribution_removal,
    project_source_contribution,
)
from phase752_natural_writer_stability_path_chain import attention_mass_for_group, norm  # noqa: E402
from phase755_cross_domain_route_invariance_atlas import get_first_token_id  # noqa: E402
from phase756_cross_domain_writer_control_downstream_carrier import expanded_candidates, run_logits  # noqa: E402
from phase760_route_suppression_matrix_atlas import build_explicit_route_groups, route_ids_from_groups, route_matrix  # noqa: E402


OUT_ROOT = Path("results/glm5_phase762_semantic_numeric_fiber_atlas")

RELATION_KEYS = [
    ("category", "category"),
    ("color", "color"),
    ("taste", "taste"),
    ("shape", "shape"),
    ("edible", "edible"),
    ("grows_on_tree", "tree"),
]

SEMANTIC_OBJECTS = [
    {"object": "apple", "domain": "fruit", "category": "fruit", "color": "red", "taste": "sweet", "shape": "round", "edible": "yes", "tree": "yes"},
    {"object": "banana", "domain": "fruit", "category": "fruit", "color": "yellow", "taste": "sweet", "shape": "long", "edible": "yes", "tree": "no"},
    {"object": "pear", "domain": "fruit", "category": "fruit", "color": "green", "taste": "sweet", "shape": "round", "edible": "yes", "tree": "yes"},
    {"object": "cat", "domain": "animal", "category": "animal", "color": "black", "taste": "none", "shape": "small", "edible": "no", "tree": "no"},
    {"object": "bird", "domain": "animal", "category": "animal", "color": "blue", "taste": "none", "shape": "small", "edible": "no", "tree": "no"},
    {"object": "dog", "domain": "animal", "category": "animal", "color": "brown", "taste": "none", "shape": "medium", "edible": "no", "tree": "no"},
    {"object": "oak", "domain": "plant", "category": "plant", "color": "green", "taste": "none", "shape": "tall", "edible": "no", "tree": "yes"},
    {"object": "rose", "domain": "plant", "category": "plant", "color": "red", "taste": "none", "shape": "round", "edible": "no", "tree": "no"},
    {"object": "wheat", "domain": "plant", "category": "plant", "color": "yellow", "taste": "plain", "shape": "tall", "edible": "yes", "tree": "no"},
    {"object": "chair", "domain": "object", "category": "furniture", "color": "brown", "taste": "none", "shape": "rectangular", "edible": "no", "tree": "no"},
    {"object": "stone", "domain": "object", "category": "object", "color": "gray", "taste": "none", "shape": "irregular", "edible": "no", "tree": "no"},
    {"object": "cup", "domain": "object", "category": "object", "color": "white", "taste": "none", "shape": "round", "edible": "no", "tree": "no"},
    {"object": "hammer", "domain": "tool", "category": "tool", "color": "silver", "taste": "none", "shape": "long", "edible": "no", "tree": "no"},
    {"object": "knife", "domain": "tool", "category": "tool", "color": "silver", "taste": "none", "shape": "long", "edible": "no", "tree": "no"},
    {"object": "scissors", "domain": "tool", "category": "tool", "color": "silver", "taste": "none", "shape": "small", "edible": "no", "tree": "no"},
    {"object": "freedom", "domain": "abstract", "category": "abstract", "color": "none", "taste": "none", "shape": "none", "edible": "no", "tree": "no"},
    {"object": "time", "domain": "abstract", "category": "abstract", "color": "none", "taste": "none", "shape": "none", "edible": "no", "tree": "no"},
    {"object": "justice", "domain": "abstract", "category": "abstract", "color": "none", "taste": "none", "shape": "none", "edible": "no", "tree": "no"},
]

VALUE_POOLS = {
    "category": ["fruit", "animal", "plant", "tool", "object", "furniture", "abstract"],
    "color": ["red", "yellow", "green", "blue", "black", "brown", "gray", "white", "silver", "none"],
    "taste": ["sweet", "bitter", "plain", "none"],
    "shape": ["round", "long", "small", "medium", "tall", "rectangular", "irregular", "none"],
    "edible": ["yes", "no"],
    "tree": ["yes", "no"],
}

DEFAULT_SOURCE_GROUPS = ["target_record_line", "target_value_tokens", "object_tokens", "relation_tokens", "records_all"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def alternate_value(key: str, value: str) -> str:
    for cand in VALUE_POOLS[key]:
        if cand != value:
            return cand
    return "unknown"


def build_semantic_pairs(max_pairs: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cid = 0
    for obj in SEMANTIC_OBJECTS:
        explicit = {**obj, "object_group": obj["domain"]}
        for relation, key in RELATION_KEYS:
            cid += 1
            answer = str(explicit[key])
            conflict = {**explicit, key: alternate_value(key, answer)}
            rows.append(
                {
                    "pair_id": f"p762_{cid:04d}_{obj['domain']}:{obj['object']}:{relation}",
                    "semantic_object": obj["object"],
                    "semantic_domain": obj["domain"],
                    "semantic_relation": relation,
                    "explicit_profile": {
                        "case_id": f"p762_explicit_{cid:04d}",
                        "prompt_type": "explicit_profile",
                        "object": obj["object"],
                        "object_group": obj["domain"],
                        "domain": obj["domain"],
                        "relation": relation,
                        "answer": answer,
                        "records": records_for(explicit),
                    },
                    "conflict_profile": {
                        "case_id": f"p762_conflict_{cid:04d}",
                        "prompt_type": "conflict_profile",
                        "object": obj["object"],
                        "object_group": obj["domain"],
                        "domain": obj["domain"],
                        "relation": relation,
                        "answer": str(conflict[key]),
                        "records": records_for(conflict),
                    },
                }
            )
    if max_pairs and max_pairs < len(rows):
        return [rows[i] for i in select_evenly(len(rows), max_pairs)]
    return rows


def source_groups_for(args: argparse.Namespace) -> list[str]:
    if args.source_groups:
        return [x.strip() for x in args.source_groups.split(",") if x.strip()]
    return DEFAULT_SOURCE_GROUPS[: args.max_source_groups]


def object_meta_from_pairs(pairs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    meta = {}
    for pair in pairs:
        obj = pair["semantic_object"]
        meta[obj] = {"object": obj, "domain": pair["semantic_domain"]}
    return meta


def mean_dict_values(values: dict[str, list[float]]) -> dict[str, float]:
    return {k: float(sum(v) / len(v)) for k, v in values.items() if v}


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


def pairwise_similarity(
    vectors: dict[str, dict[str, float]],
    meta: dict[str, dict[str, Any]],
    features: list[str],
    kind: str,
) -> list[dict[str, Any]]:
    objects = sorted(vectors)
    rows = []
    for i, a in enumerate(objects):
        for b in objects[i + 1 :]:
            sim = cosine(vectors[a], vectors[b], features)
            rows.append(
                {
                    "kind": kind,
                    "object_a": a,
                    "object_b": b,
                    "domain_a": meta[a]["domain"],
                    "domain_b": meta[b]["domain"],
                    "same_domain": meta[a]["domain"] == meta[b]["domain"],
                    "similarity": sim,
                }
            )
    return rows


def nn_domain_accuracy(pair_rows: list[dict[str, Any]], objects: list[str]) -> dict[str, Any]:
    neighbors: dict[str, tuple[str, float, bool]] = {}
    for row in pair_rows:
        a, b = row["object_a"], row["object_b"]
        sim = float(row["similarity"])
        for x, y in [(a, b), (b, a)]:
            if x not in neighbors or sim > neighbors[x][1]:
                neighbors[x] = (y, sim, bool(row["same_domain"]))
    if not neighbors:
        return {"accuracy": None, "neighbors": {}}
    acc = sum(1 for obj in objects if neighbors.get(obj, ("", 0.0, False))[2]) / len(objects)
    return {
        "accuracy": acc,
        "neighbors": {
            obj: {
                "nearest": neighbors[obj][0],
                "similarity": neighbors[obj][1],
                "same_domain": neighbors[obj][2],
            }
            for obj in sorted(neighbors)
        },
    }


def same_diff_summary(pair_rows: list[dict[str, Any]]) -> dict[str, Any]:
    same = [float(r["similarity"]) for r in pair_rows if r["same_domain"]]
    diff = [float(r["similarity"]) for r in pair_rows if not r["same_domain"]]
    return {
        "same_domain_mean": safe_mean(same),
        "different_domain_mean": safe_mean(diff),
        "separation": (safe_mean(same) or 0.0) - (safe_mean(diff) or 0.0),
        "same_n": len(same),
        "different_n": len(diff),
    }


def embedding_baseline(model, tokenizer, meta: dict[str, dict[str, Any]]) -> dict[str, Any]:
    emb = model.get_input_embeddings().weight.detach().float().cpu()
    vectors = {}
    for obj in sorted(meta):
        tid = get_first_token_id(tokenizer, obj)
        vectors[obj] = emb[int(tid)].tolist()
    features = [str(i) for i in range(len(next(iter(vectors.values()))))]
    dict_vectors = {obj: {str(i): float(v) for i, v in enumerate(vec)} for obj, vec in vectors.items()}
    pairs = pairwise_similarity(dict_vectors, meta, features, "embedding_first_token")
    objects = sorted(meta)
    return {
        "pair_summary": same_diff_summary(pairs),
        "nn_domain": nn_domain_accuracy(pairs, objects),
        "pairs": pairs,
    }


def add_feature(bucket: dict[str, list[float]], feature: str, value: float) -> None:
    if math.isfinite(float(value)):
        bucket[feature].append(float(value))


def audit_pair(
    model,
    tokenizer,
    device,
    args: argparse.Namespace,
    pair: dict[str, Any],
    candidates: list[dict[str, Any]],
    source_groups: list[str],
    unembed: torch.Tensor,
) -> list[dict[str, Any]]:
    target = pair["explicit_profile"]
    contrast = pair["conflict_profile"]
    candidate_layers = sorted({parse_component_site(c["site"])[0] for c in candidates})
    state = capture_attention_value_state(model, tokenizer, device, target, candidate_layers)
    target_id = get_first_token_id(tokenizer, target["answer"])
    contrast_id = get_first_token_id(tokenizer, contrast["answer"])
    route_groups = build_explicit_route_groups(tokenizer, state["logits"], target, contrast, target_id, contrast_id, args)
    if not route_groups:
        return []
    route_ids = route_ids_from_groups(route_groups)
    target_diag = logit_diag(state["logits"], target_id)
    contrast_diag = logit_diag(state["logits"], contrast_id)
    rows: list[dict[str, Any]] = [
        {
            "row_kind": "semantic_task_observation",
            "pair_id": pair["pair_id"],
            "object": pair["semantic_object"],
            "domain": pair["semantic_domain"],
            "relation": pair["semantic_relation"],
            "target_answer": target["answer"],
            "contrast_answer": contrast["answer"],
            "target_token_id": target_id,
            "contrast_token_id": contrast_id,
            "target_rank": target_diag["target_rank"],
            "target_top1": target_diag["target_top1"],
            "contrast_rank": contrast_diag["target_rank"],
            "route_group_names": sorted(route_groups),
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
            target_drop = float(state["logits"][target_id].item() - after_logits[target_id].item())
            matrix = route_matrix(state["logits"], after_logits, route_groups, target_id)
            rows.append(
                {
                    "row_kind": "semantic_fiber_effect",
                    "pair_id": pair["pair_id"],
                    "object": pair["semantic_object"],
                    "domain": pair["semantic_domain"],
                    "relation": pair["semantic_relation"],
                    "target_answer": target["answer"],
                    "contrast_answer": contrast["answer"],
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
                    "source_projected_delta_norm": norm(projected),
                    "source_direct_score": direct,
                    "target_logit_drop": target_drop,
                    "route_matrix": matrix,
                    "total_positive_route_release": float(sum(max(0.0, float(v["route_release"])) for v in matrix.values())),
                }
            )
    return rows


def build_fiber_vectors(rows: list[dict[str, Any]], args: argparse.Namespace) -> tuple[dict[str, dict[str, float]], list[str], list[dict[str, Any]]]:
    values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    relation_summary: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row.get("row_kind") != "semantic_fiber_effect":
            continue
        obj = row["object"]
        relation = row["relation"]
        prefix = f"rel={relation}|{row['subunit_id']}|{row['source_group']}"
        add_feature(values[obj], f"{prefix}|target_logit_drop", row["target_logit_drop"])
        add_feature(values[obj], f"{prefix}|attention_mass", row["attention_mass_to_source"])
        add_feature(values[obj], f"{prefix}|direct_target_boost", row["source_direct_score"]["direct_target_boost"])
        add_feature(values[obj], f"{prefix}|direct_total_route_suppression", row["source_direct_score"]["direct_total_route_suppression"])
        add_feature(relation_summary[relation], "target_logit_drop", row["target_logit_drop"])
        add_feature(relation_summary[relation], "total_positive_route_release", row["total_positive_route_release"])
        for route_group, cell in row["route_matrix"].items():
            add_feature(values[obj], f"{prefix}|route_release:{route_group}", cell["route_release"])
            add_feature(values[obj], f"{prefix}|margin_drop:{route_group}", cell["margin_drop_target_vs_route"])
            add_feature(relation_summary[relation], f"route_release:{route_group}", cell["route_release"])
    vectors = {obj: mean_dict_values(feats) for obj, feats in values.items()}
    features = sorted({f for vec in vectors.values() for f in vec})
    rel_rows = []
    for relation, feats in relation_summary.items():
        meaned = mean_dict_values(feats)
        rel_rows.append(
            {
                "relation": relation,
                "mean_target_logit_drop": meaned.get("target_logit_drop"),
                "mean_total_positive_route_release": meaned.get("total_positive_route_release"),
                "top_route_release_features": dict(
                    sorted(
                        ((k, v) for k, v in meaned.items() if k.startswith("route_release:")),
                        key=lambda kv: abs(kv[1]),
                        reverse=True,
                    )[:8]
                ),
            }
        )
    rel_rows.sort(key=lambda r: r["relation"])
    return vectors, features, rel_rows


def build_summary(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    source_groups: list[str],
    attn_impl: str,
    embedding_payload: dict[str, Any],
    pairs: list[dict[str, Any]],
) -> dict[str, Any]:
    meta = object_meta_from_pairs(pairs)
    vectors, features, relation_rows = build_fiber_vectors(rows, args)
    objects = sorted(vectors)
    raw_pairs = pairwise_similarity(vectors, meta, features, "causal_fiber_raw")
    centered = center_vectors(vectors, features)
    centered_pairs = pairwise_similarity(centered, meta, features, "causal_fiber_centered")
    top_features = []
    for f in features:
        vals = [abs(vectors[obj].get(f, 0.0)) for obj in objects]
        top_features.append({"feature": f, "mean_abs": safe_mean(vals), "nonzero_n": sum(1 for v in vals if abs(v) > 1e-8)})
    top_features.sort(key=lambda r: (r["mean_abs"] or 0.0, r["nonzero_n"]), reverse=True)
    random_nearest_same_domain = (len({o["domain"] for o in meta.values()}) and 2 / max(1, len(meta) - 1)) or None
    raw_nn = nn_domain_accuracy(raw_pairs, objects)
    centered_nn = nn_domain_accuracy(centered_pairs, objects)
    embedding_acc = embedding_payload["nn_domain"]["accuracy"]
    centered_acc = centered_nn["accuracy"]
    if centered_acc is not None and embedding_acc is not None and centered_acc > embedding_acc and (same_diff_summary(centered_pairs)["separation"] or 0.0) > 0:
        interface_status = "causal_fiber_domain_signal_above_embedding"
    elif centered_acc is not None and centered_acc > (random_nearest_same_domain or 0):
        interface_status = "weak_causal_fiber_domain_signal"
    else:
        interface_status = "semantic_numeric_interface_not_established"
    return {
        "phase": 762,
        "title": "Semantic Numeric Fiber Atlas",
        "model": args.model,
        "round": args.round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "dtype": "bfloat16",
        "quantization": "off",
        "n_rows": len(rows),
        "n_tasks": sum(1 for r in rows if r.get("row_kind") == "semantic_task_observation"),
        "n_effect_rows": sum(1 for r in rows if r.get("row_kind") == "semantic_fiber_effect"),
        "n_objects": len(objects),
        "n_features": len(features),
        "random_nearest_same_domain_baseline": random_nearest_same_domain,
        "candidates": candidates,
        "source_groups": source_groups,
        "interface_status": interface_status,
        "causal_fiber_raw_pair_summary": same_diff_summary(raw_pairs),
        "causal_fiber_centered_pair_summary": same_diff_summary(centered_pairs),
        "causal_fiber_raw_nn_domain": raw_nn,
        "causal_fiber_centered_nn_domain": centered_nn,
        "embedding_first_token_baseline": embedding_payload,
        "object_pair_similarities": {
            "raw": raw_pairs,
            "centered": centered_pairs,
            "embedding": embedding_payload["pairs"],
        },
        "relation_effect_summary": relation_rows,
        "top_causal_fiber_features": top_features[:96],
        "strict_interpretation": "Object fibers are head/source causal fingerprints over relation tasks. This is a head-level semantic-numeric interface probe, not a neuron-level global atlas.",
    }


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = OUT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = build_semantic_pairs(args.max_pairs)
    source_groups = source_groups_for(args)
    log(f"{args.model}/{args.round_name}: semantic_tasks={len(pairs)} sources={source_groups}")
    model, tokenizer, device, attn_impl = load_model_bf16_eager(args.model)
    try:
        candidates = expanded_candidates(model, args.model, args)
        unembed = get_unembed(model)
        embedding_payload = embedding_baseline(model, tokenizer, object_meta_from_pairs(pairs))
        log(f"{args.model}: candidates={len(candidates)} embedding_nn={embedding_payload['nn_domain']['accuracy']}")
        rows: list[dict[str, Any]] = []
        for idx, pair in enumerate(pairs, 1):
            rows.extend(audit_pair(model, tokenizer, device, args, pair, candidates, source_groups, unembed))
            if idx % args.log_every == 0 or idx == len(pairs):
                log(f"{args.model}: semantic fibers {idx}/{len(pairs)} tasks; rows={len(rows)}")
    finally:
        release_model(model)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    summary = build_summary(args, rows, candidates, source_groups, attn_impl, embedding_payload, pairs)
    write_jsonl(out_dir / f"phase762_{args.model}_rows.jsonl", rows)
    write_json(out_dir / f"phase762_{args.model}_summary.json", summary)
    print(
        json.dumps(
            {
                "model": args.model,
                "round": args.round_name,
                "n_tasks": summary["n_tasks"],
                "n_features": summary["n_features"],
                "interface_status": summary["interface_status"],
                "causal_centered": summary["causal_fiber_centered_pair_summary"],
                "causal_centered_nn": summary["causal_fiber_centered_nn_domain"]["accuracy"],
                "embedding_nn": summary["embedding_first_token_baseline"]["nn_domain"]["accuracy"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return summary


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 1e-12 or vy <= 1e-12:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def pair_map(summary: dict[str, Any], kind: str) -> dict[str, float]:
    rows = summary["object_pair_similarities"][kind]
    out = {}
    for row in rows:
        key = "||".join(sorted([row["object_a"], row["object_b"]]))
        out[key] = float(row["similarity"])
    return out


def write_cross_summary(round_name: str) -> dict[str, Any]:
    out_dir = OUT_ROOT / round_name
    summaries = []
    for model_name in MODELS:
        path = out_dir / f"phase762_{model_name}_summary.json"
        if path.exists():
            summaries.append(json.loads(path.read_text(encoding="utf-8")))
    by_model = {s["model"]: s for s in summaries}
    correlations = {}
    for i, a in enumerate(summaries):
        for b in summaries[i + 1 :]:
            ma = pair_map(a, "centered")
            mb = pair_map(b, "centered")
            keys = sorted(set(ma) & set(mb))
            correlations[f"{a['model']}__{b['model']}"] = {
                "common_pairs": len(keys),
                "centered_pair_similarity_pearson": pearson([ma[k] for k in keys], [mb[k] for k in keys]),
            }
    payload = {
        "phase": 762,
        "title": "Semantic Numeric Fiber Atlas",
        "round": round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": [s["model"] for s in summaries],
        "status": "complete" if len(summaries) == len(MODELS) else "partial",
        "by_model": by_model,
        "cross_model_centered_similarity_correlations": correlations,
        "strict_interpretation": "Causal fibers are first-pass head/source functional fingerprints. Positive domain separation is not yet a global neuron atlas.",
    }
    write_json(out_dir / "phase762_cross_model_summary.json", payload)
    lines = [
        f"# Phase 762 Semantic Numeric Fiber Atlas ({round_name})",
        "",
        f"- Status: `{payload['status']}`",
        f"- Models: `{payload['models']}`",
        "- Test: convert objects into causal functional fingerprints over object-relation tasks, then test same-domain clustering and compare with first-token embedding baseline.",
        "",
        "## Object Fiber Results",
        "",
        "| model | tasks | objects | features | interface status | causal NN | embed NN | causal same | causal diff | causal sep | embed sep |",
        "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model_name, summary in by_model.items():
        c = summary["causal_fiber_centered_pair_summary"]
        e = summary["embedding_first_token_baseline"]["pair_summary"]
        lines.append(
            f"| {model_name} | {summary['n_tasks']} | {summary['n_objects']} | {summary['n_features']} | `{summary['interface_status']}` | "
            f"{(summary['causal_fiber_centered_nn_domain']['accuracy'] or 0):.3f} | "
            f"{(summary['embedding_first_token_baseline']['nn_domain']['accuracy'] or 0):.3f} | "
            f"{(c.get('same_domain_mean') or 0):.3f} | {(c.get('different_domain_mean') or 0):.3f} | {(c.get('separation') or 0):.3f} | "
            f"{(e.get('separation') or 0):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Cross Model Object-Topology Correlation",
            "",
            "| pair | common object pairs | centered similarity correlation |",
            "|---|---:|---:|",
        ]
    )
    for key, row in correlations.items():
        corr = row["centered_pair_similarity_pearson"]
        lines.append(f"| `{key}` | {row['common_pairs']} | {(corr if corr is not None else 0):.3f} |")
    lines.extend(
        [
            "",
            "## Nearest Neighbors",
            "",
            "| model | object | nearest causal | causal same domain | nearest embedding | embedding same domain |",
            "|---|---|---|---:|---|---:|",
        ]
    )
    for model_name, summary in by_model.items():
        cn = summary["causal_fiber_centered_nn_domain"]["neighbors"]
        en = summary["embedding_first_token_baseline"]["nn_domain"]["neighbors"]
        for obj in sorted(cn)[:18]:
            lines.append(
                f"| {model_name} | `{obj}` | `{cn[obj]['nearest']}` | {int(bool(cn[obj]['same_domain']))} | "
                f"`{en.get(obj, {}).get('nearest', '')}` | {int(bool(en.get(obj, {}).get('same_domain', False)))} |"
            )
    lines.extend(
        [
            "",
            "## Strict Interpretation",
            "",
            "- If causal NN domain accuracy exceeds the embedding baseline, this supports a first semantic-numeric interface signal.",
            "- If the signal appears only in one model, it is model-local and not a universal semantic fiber.",
            "- This phase stays at head/source level; it does not claim neuron-level or parameter-level localization.",
            "",
        ]
    )
    (out_dir / "phase762_cross_model_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"round": round_name, "status": payload["status"], "models": payload["models"]}, ensure_ascii=False, indent=2), flush=True)
    return payload


def dry_run(args: argparse.Namespace) -> None:
    payload = {
        "phase": 762,
        "round": args.round_name,
        "semantic_tasks": len(build_semantic_pairs(args.max_pairs)),
        "objects": len({x["object"] for x in SEMANTIC_OBJECTS}),
        "domains": sorted({x["domain"] for x in SEMANTIC_OBJECTS}),
        "relations": [r for r, _k in RELATION_KEYS],
        "source_groups": source_groups_for(args),
        "max_candidates": args.max_candidates,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="main")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-pairs", type=int, default=54)
    parser.add_argument("--max-candidates", type=int, default=2)
    parser.add_argument("--include-controls", action="store_true", default=True)
    parser.add_argument("--controls-per-candidate", type=int, default=1)
    parser.add_argument("--control-offset", type=int, default=13)
    parser.add_argument("--max-source-groups", type=int, default=5)
    parser.add_argument("--source-groups", default="")
    parser.add_argument("--top-k-vocab", type=int, default=16)
    parser.add_argument("--max-topk-tokens", type=int, default=10)
    parser.add_argument("--max-dynamic-route-classes", type=int, default=5)
    parser.add_argument("--log-every", type=int, default=9)
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
