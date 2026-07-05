#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase846_geometry_boundary_equation_fitting as p846  # noqa: E402
import phase862_negative_blocker_sign_mechanism_audit as p862  # noqa: E402
import phase901_stop_token_competitiveness_audit as p901  # noqa: E402
from model_utils import get_layers  # noqa: E402


PHASE = 937
MODELS = ["qwen3", "glm4", "deepseek7b"]
RESULT_ROOT = Path("tests/result/phase937_semantic_reuse_difference_state_atlas")


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def finite(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def mean(values: list[float | int | None]) -> float | None:
    cleaned = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return None if not cleaned else float(sum(cleaned) / len(cleaned))


def safe_div(num: float, den: float) -> float | None:
    return None if den == 0 else float(num / den)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def semantic_cases() -> list[dict[str, str]]:
    return [
        {"domain": "fruit", "object": "apple", "color": "red", "function": "eat"},
        {"domain": "fruit", "object": "cherry", "color": "red", "function": "eat"},
        {"domain": "fruit", "object": "banana", "color": "yellow", "function": "peel"},
        {"domain": "fruit", "object": "lemon", "color": "yellow", "function": "squeeze"},
        {"domain": "fruit", "object": "orange", "color": "orange", "function": "juice"},
        {"domain": "fruit", "object": "grape", "color": "purple", "function": "eat"},
        {"domain": "animal", "object": "cardinal", "color": "red", "function": "fly"},
        {"domain": "animal", "object": "canary", "color": "yellow", "function": "fly"},
        {"domain": "animal", "object": "shark", "color": "gray", "function": "swim"},
        {"domain": "animal", "object": "whale", "color": "gray", "function": "swim"},
        {"domain": "animal", "object": "horse", "color": "brown", "function": "run"},
        {"domain": "animal", "object": "dog", "color": "brown", "function": "run"},
        {"domain": "vehicle", "object": "car", "color": "red", "function": "transport"},
        {"domain": "vehicle", "object": "bus", "color": "yellow", "function": "transport"},
        {"domain": "vehicle", "object": "taxi", "color": "yellow", "function": "transport"},
        {"domain": "vehicle", "object": "bicycle", "color": "black", "function": "ride"},
        {"domain": "vehicle", "object": "train", "color": "silver", "function": "transport"},
        {"domain": "vehicle", "object": "boat", "color": "white", "function": "transport"},
        {"domain": "tool", "object": "knife", "color": "silver", "function": "cut"},
        {"domain": "tool", "object": "saw", "color": "silver", "function": "cut"},
        {"domain": "tool", "object": "key", "color": "silver", "function": "open"},
        {"domain": "tool", "object": "hammer", "color": "gray", "function": "hit"},
        {"domain": "tool", "object": "cup", "color": "white", "function": "hold"},
        {"domain": "tool", "object": "spoon", "color": "silver", "function": "eat"},
        {"domain": "material", "object": "metal", "color": "silver", "function": "conduct"},
        {"domain": "material", "object": "glass", "color": "transparent", "function": "hold"},
        {"domain": "material", "object": "wood", "color": "brown", "function": "burn"},
        {"domain": "material", "object": "stone", "color": "gray", "function": "build"},
        {"domain": "material", "object": "plastic", "color": "white", "function": "mold"},
        {"domain": "material", "object": "rubber", "color": "black", "function": "stretch"},
    ]


PROMPT_TEMPLATES: dict[str, list[str]] = {
    "category": [
        "In one word, {article} {object} is a type of",
        "The category of {article} {object} is",
    ],
    "color": [
        "The typical color of {article} {object} is",
        "A common color for {article} {object} is",
    ],
    "function": [
        "{Article} {object} can often",
        "A common action or use for {article} {object} is to",
    ],
}


def parse_csv(text: str) -> list[str]:
    return [part.strip() for part in str(text or "").split(",") if part.strip()]


def selected_cases(args: argparse.Namespace) -> list[dict[str, str]]:
    domains = set(parse_csv(args.domains)) if args.domains else set()
    per_domain: Counter[str] = Counter()
    out = []
    for case in semantic_cases():
        if domains and case["domain"] not in domains:
            continue
        if int(args.max_objects_per_domain) > 0 and per_domain[case["domain"]] >= int(args.max_objects_per_domain):
            continue
        out.append(dict(case))
        per_domain[case["domain"]] += 1
    return out


def build_samples(args: argparse.Namespace) -> list[dict[str, Any]]:
    relations = parse_csv(args.relations)
    samples = []
    for case in selected_cases(args):
        article = "an" if case["object"][0].lower() in {"a", "e", "i", "o", "u"} else "a"
        format_case = {**case, "article": article, "Article": article.capitalize()}
        for relation in relations:
            templates = PROMPT_TEMPLATES[relation][: int(args.templates_per_relation)]
            for template_idx, template in enumerate(templates):
                target = case["domain"] if relation == "category" else case[relation]
                sample_id = f"{case['domain']}:{case['object']}:{relation}:t{template_idx}"
                samples.append(
                    {
                        "phase": PHASE,
                        "sample_id": sample_id,
                        "domain": case["domain"],
                        "object": case["object"],
                        "relation": relation,
                        "target_label": target,
                        "prompt_template": f"{relation}_{template_idx}",
                        "prompt": template.format(**format_case),
                    }
                )
    return samples


def auto_hidden_indices(model) -> list[int]:
    n_layers = len(get_layers(model))
    raw = [0, n_layers // 4, n_layers // 2, (3 * n_layers) // 4, n_layers]
    return sorted(set(int(x) for x in raw if 0 <= int(x) <= n_layers))


def first_token_candidates(tokenizer, label: str) -> list[int]:
    ids = []
    for phrase in [label, " " + label]:
        try:
            encoded = tokenizer.encode(phrase, add_special_tokens=False)
        except Exception:
            encoded = []
        if encoded:
            token_id = int(encoded[0])
            if token_id not in ids:
                ids.append(token_id)
    return ids


def rank_of(logits: torch.Tensor, token_id: int | None) -> int | None:
    if token_id is None:
        return None
    token_id = int(token_id)
    if token_id < 0 or token_id >= int(logits.numel()):
        return None
    value = logits[token_id]
    return int((logits > value).sum().item() + 1)


def best_label_token(tokenizer, logits: torch.Tensor, label: str) -> dict[str, Any]:
    candidates = first_token_candidates(tokenizer, label)
    if not candidates:
        return {"target_token_id": None, "target_token": None, "target_logit": None, "target_rank": None}
    best_id = max(candidates, key=lambda token_id: float(logits[int(token_id)].item()))
    return {
        "target_token_id": int(best_id),
        "target_token": p901.decode_token(tokenizer, int(best_id)),
        "target_logit": float(logits[int(best_id)].item()),
        "target_rank": rank_of(logits, int(best_id)),
    }


def batch_extract_states(
    model,
    tokenizer,
    device: torch.device,
    samples: list[dict[str, Any]],
    hidden_indices: list[int],
    batch_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    state_rows: list[dict[str, Any]] = []
    vector_rows: list[dict[str, Any]] = []
    for start in range(0, len(samples), max(1, int(batch_size))):
        batch = samples[start : start + max(1, int(batch_size))]
        prompts = [str(sample["prompt"]) for sample in batch]
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        last_pos = attention_mask.sum(dim=1).long() - 1
        batch_index = torch.arange(input_ids.shape[0], device=device)
        with torch.inference_mode():
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        logits = out.logits[batch_index, last_pos].detach().float().cpu()
        for idx, sample in enumerate(batch):
            row_base = dict(sample)
            top_id = int(torch.argmax(logits[idx]).item())
            target = best_label_token(tokenizer, logits[idx], str(sample["target_label"]))
            row_base.update(
                {
                    "next_top_id": top_id,
                    "next_top_token": p901.decode_token(tokenizer, top_id),
                    **target,
                }
            )
            for hidden_idx in hidden_indices:
                h = out.hidden_states[int(hidden_idx)][idx, int(last_pos[idx].item())].detach().float().cpu()
                state_row = {
                    **row_base,
                    "hidden_idx": int(hidden_idx),
                    "hidden_norm": float(torch.linalg.vector_norm(h).item()),
                }
                state_rows.append(state_row)
                vector_rows.append({**state_row, "_vector": h})
        del out, logits, input_ids, attention_mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return state_rows, vector_rows


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() == 0 or b.numel() == 0:
        return 0.0
    return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=-1).item())


def centroid(rows: list[dict[str, Any]]) -> torch.Tensor | None:
    if not rows:
        return None
    return torch.stack([row["_vector"].float() for row in rows]).mean(dim=0)


def pairwise_metric(rows: list[dict[str, Any]], predicate) -> tuple[float | None, int]:
    values = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            if predicate(rows[i], rows[j]):
                values.append(cosine(rows[i]["_vector"], rows[j]["_vector"]))
    return mean(values), len(values)


def classify_by_template_holdout(rows: list[dict[str, Any]]) -> dict[str, Any]:
    hits = 0
    total = 0
    labels = sorted({str(row["target_label"]) for row in rows})
    for row in rows:
        train = [r for r in rows if r["prompt_template"] != row["prompt_template"]]
        centroids = {label: centroid([r for r in train if str(r["target_label"]) == label]) for label in labels}
        centroids = {label: vec for label, vec in centroids.items() if vec is not None}
        if str(row["target_label"]) not in centroids or len(centroids) < 2:
            continue
        scores = {label: cosine(row["_vector"], vec) for label, vec in centroids.items()}
        pred = max(scores, key=scores.get)
        hits += int(pred == str(row["target_label"]))
        total += 1
    return {
        "template_holdout_centroid_hits": hits,
        "template_holdout_centroid_total": total,
        "template_holdout_centroid_accuracy": safe_div(hits, total),
        "template_holdout_label_count": len(labels),
        "template_holdout_chance": safe_div(1.0, len(labels)),
    }


def residual_stability(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_label = defaultdict(list)
    for row in rows:
        by_label[str(row["target_label"])].append(row)
    centers = {label: centroid(items) for label, items in by_label.items()}
    residual_rows = []
    for row in rows:
        center = centers.get(str(row["target_label"]))
        if center is None:
            continue
        copied = dict(row)
        copied["_vector"] = row["_vector"] - center
        residual_rows.append(copied)
    same_object, same_object_n = pairwise_metric(
        residual_rows,
        lambda a, b: a["object"] == b["object"] and a["prompt_template"] != b["prompt_template"],
    )
    same_label_diff_object, same_label_diff_object_n = pairwise_metric(
        residual_rows,
        lambda a, b: a["target_label"] == b["target_label"] and a["object"] != b["object"],
    )
    return {
        "object_residual_same_object_template_cos": same_object,
        "object_residual_same_object_pairs": same_object_n,
        "object_residual_same_label_diff_object_cos": same_label_diff_object,
        "object_residual_same_label_diff_object_pairs": same_label_diff_object_n,
        "object_residual_stability_gap": None
        if same_object is None or same_label_diff_object is None
        else float(same_object - same_label_diff_object),
    }


def summarize_vectors(model_name: str, vector_rows: list[dict[str, Any]], hidden_indices: list[int]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    metric_rows = []
    for relation in sorted({str(row["relation"]) for row in vector_rows}):
        for hidden_idx in hidden_indices:
            rows = [row for row in vector_rows if str(row["relation"]) == relation and int(row["hidden_idx"]) == int(hidden_idx)]
            if len(rows) < 2:
                continue
            same_target, same_target_n = pairwise_metric(rows, lambda a, b: a["target_label"] == b["target_label"])
            diff_target, diff_target_n = pairwise_metric(rows, lambda a, b: a["target_label"] != b["target_label"])
            same_domain, same_domain_n = pairwise_metric(rows, lambda a, b: a["domain"] == b["domain"])
            diff_domain, diff_domain_n = pairwise_metric(rows, lambda a, b: a["domain"] != b["domain"])
            same_target_cross_domain, same_target_cross_domain_n = pairwise_metric(
                rows,
                lambda a, b: a["target_label"] == b["target_label"] and a["domain"] != b["domain"],
            )
            diff_target_cross_domain, diff_target_cross_domain_n = pairwise_metric(
                rows,
                lambda a, b: a["target_label"] != b["target_label"] and a["domain"] != b["domain"],
            )
            same_domain_diff_target, same_domain_diff_target_n = pairwise_metric(
                rows,
                lambda a, b: a["domain"] == b["domain"] and a["target_label"] != b["target_label"],
            )
            cls = classify_by_template_holdout(rows)
            resid = residual_stability(rows)
            label_domain_counts = defaultdict(set)
            for row in rows:
                label_domain_counts[str(row["target_label"])].add(str(row["domain"]))
            metric_rows.append(
                {
                    "phase": PHASE,
                    "model": model_name,
                    "relation": relation,
                    "hidden_idx": int(hidden_idx),
                    "rows": len(rows),
                    "target_label_count": len({str(row["target_label"]) for row in rows}),
                    "domain_count": len({str(row["domain"]) for row in rows}),
                    "object_count": len({str(row["object"]) for row in rows}),
                    "same_target_mean_cos": same_target,
                    "same_target_pairs": same_target_n,
                    "diff_target_mean_cos": diff_target,
                    "diff_target_pairs": diff_target_n,
                    "target_reuse_gap": None if same_target is None or diff_target is None else float(same_target - diff_target),
                    "same_domain_mean_cos": same_domain,
                    "same_domain_pairs": same_domain_n,
                    "diff_domain_mean_cos": diff_domain,
                    "diff_domain_pairs": diff_domain_n,
                    "domain_gap": None if same_domain is None or diff_domain is None else float(same_domain - diff_domain),
                    "same_target_cross_domain_mean_cos": same_target_cross_domain,
                    "same_target_cross_domain_pairs": same_target_cross_domain_n,
                    "diff_target_cross_domain_mean_cos": diff_target_cross_domain,
                    "diff_target_cross_domain_pairs": diff_target_cross_domain_n,
                    "cross_domain_target_gap": None
                    if same_target_cross_domain is None or diff_target_cross_domain is None
                    else float(same_target_cross_domain - diff_target_cross_domain),
                    "same_domain_diff_target_mean_cos": same_domain_diff_target,
                    "same_domain_diff_target_pairs": same_domain_diff_target_n,
                    "labels_with_cross_domain_support": sum(1 for values in label_domain_counts.values() if len(values) >= 2),
                    **cls,
                    **resid,
                }
            )
    best_by_relation = {}
    for relation in sorted({row["relation"] for row in metric_rows}):
        rel_rows = [row for row in metric_rows if row["relation"] == relation]
        rel_rows.sort(
            key=lambda row: (
                finite(row.get("template_holdout_centroid_accuracy"), -1.0),
                finite(row.get("target_reuse_gap"), -9.0),
                finite(row.get("cross_domain_target_gap"), -9.0),
            ),
            reverse=True,
        )
        best_by_relation[relation] = rel_rows[0] if rel_rows else {}
    positive_relations = 0
    for relation, row in best_by_relation.items():
        acc = row.get("template_holdout_centroid_accuracy")
        chance = row.get("template_holdout_chance")
        reuse_gap = finite(row.get("target_reuse_gap"), 0.0)
        cross_gap = finite(row.get("cross_domain_target_gap"), 0.0)
        if acc is not None and chance is not None and float(acc) > float(chance) + 0.15 and (reuse_gap > 0 or cross_gap > 0):
            positive_relations += 1
    if positive_relations >= 2:
        evidence = "semantic_reuse_difference_signals_observed"
    elif positive_relations == 1:
        evidence = "partial_semantic_reuse_difference_signal"
    else:
        evidence = "semantic_reuse_difference_signal_weak_or_absent"
    summary = {
        "phase": PHASE,
        "model": model_name,
        "relations": sorted(best_by_relation),
        "hidden_indices": hidden_indices,
        "metric_rows": len(metric_rows),
        "best_by_relation": best_by_relation,
        "positive_relation_count": positive_relations,
        "evidence_label": evidence,
    }
    return metric_rows, summary


def strip_vectors(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{k: v for k, v in row.items() if k != "_vector"} for row in rows]


def eval_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    samples = build_samples(args)
    dry_payload = {
        "phase": PHASE,
        "title": "Semantic Reuse-Difference State Atlas",
        "model": args.model,
        "sample_count": len(samples),
        "domains": sorted({row["domain"] for row in samples}),
        "objects": len({row["object"] for row in samples}),
        "relations": sorted({row["relation"] for row in samples}),
        "templates_per_relation": int(args.templates_per_relation),
    }
    if args.dry_run:
        payload = {**dry_payload, "status": "dry_run", "samples": samples[:20]}
        write_json(out_dir / f"phase937_{args.model}_summary.json", payload)
        write_jsonl(out_dir / f"phase937_{args.model}_state_rows.jsonl", [])
        write_jsonl(out_dir / f"phase937_{args.model}_metric_rows.jsonl", [])
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return payload

    model = None
    tokenizer = None
    attn_impl = None
    state_rows: list[dict[str, Any]] = []
    vector_rows: list[dict[str, Any]] = []
    hidden_indices: list[int] = []
    try:
        model, tokenizer, device, attn_impl = p862.p844.p828.p796.load_model_bf16_prefer_flash(
            args.model, args.attn_implementations
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        hidden_indices = auto_hidden_indices(model)
        state_rows, vector_rows = batch_extract_states(
            model,
            tokenizer,
            device,
            samples,
            hidden_indices,
            int(args.batch_size),
        )
    finally:
        if model is not None:
            p862.p844.p828.release_model(model)
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    metric_rows, summary_core = summarize_vectors(args.model, vector_rows, hidden_indices)
    target_rank_values = [row.get("target_rank") for row in state_rows if row.get("target_rank") is not None]
    payload = {
        **dry_payload,
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "hidden_indices": hidden_indices,
        "state_rows": len(state_rows),
        "metric_rows": len(metric_rows),
        "target_rank_mean": mean([float(x) for x in target_rank_values]),
        "target_rank_top1": sum(1 for x in target_rank_values if int(x) == 1),
        "target_rank_top10": sum(1 for x in target_rank_values if int(x) <= 10),
        "target_rank_count": len(target_rank_values),
        "summary": summary_core,
        "boundary": "non-intervention hidden-state atlas only; not causal closure",
    }
    write_json(out_dir / f"phase937_{args.model}_summary.json", payload)
    write_jsonl(out_dir / f"phase937_{args.model}_state_rows.jsonl", strip_vectors(vector_rows))
    write_jsonl(out_dir / f"phase937_{args.model}_metric_rows.jsonl", metric_rows)
    print(
        json.dumps(
            {
                "phase": PHASE,
                "model": args.model,
                "status": payload["status"],
                "evidence": summary_core["evidence_label"],
                "best_by_relation": summary_core["best_by_relation"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return payload


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    summaries = [read_json(out_dir / f"phase937_{model}_summary.json") for model in MODELS]
    summaries = [summary for summary in summaries if summary]
    evidence_counts = Counter(str((summary.get("summary") or {}).get("evidence_label") or summary.get("status")) for summary in summaries)
    all_metric_rows = []
    for model in MODELS:
        all_metric_rows.extend(read_jsonl(out_dir / f"phase937_{model}_metric_rows.jsonl"))
    top_rows = sorted(
        all_metric_rows,
        key=lambda row: (
            finite(row.get("template_holdout_centroid_accuracy"), -1.0),
            finite(row.get("target_reuse_gap"), -9.0),
            finite(row.get("cross_domain_target_gap"), -9.0),
        ),
        reverse=True,
    )[:120]
    payload = {
        "phase": PHASE,
        "round": round_name,
        "status": "complete" if summaries else "missing",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": [summary.get("model") for summary in summaries],
        "evidence_counts": dict(evidence_counts),
        "model_summaries": summaries,
        "top_metric_rows": top_rows,
    }
    write_json(out_dir / "phase937_cross_model_summary.json", payload)
    write_summary_md(out_dir / "phase937_cross_model_summary.md", payload)
    return payload


def write_summary_md(path: Path, payload: dict[str, Any]) -> None:
    lines = ["# Phase 937 semantic reuse-difference state atlas", ""]
    lines += ["## Evidence", ""]
    for key, value in sorted((payload.get("evidence_counts") or {}).items()):
        lines.append(f"- {key}: {value}")
    lines += ["", "## Model Best Relations", ""]
    lines.append("| model | relation | hidden | acc | chance | reuse gap | cross-domain target gap | residual gap |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for summary in payload.get("model_summaries") or []:
        model = summary.get("model")
        best = ((summary.get("summary") or {}).get("best_by_relation") or {})
        for relation, row in sorted(best.items()):
            lines.append(
                "| {model} | {relation} | {hidden_idx} | {acc} | {chance} | {reuse} | {cross} | {resid} |".format(
                    model=model,
                    relation=relation,
                    hidden_idx=row.get("hidden_idx"),
                    acc=row.get("template_holdout_centroid_accuracy"),
                    chance=row.get("template_holdout_chance"),
                    reuse=row.get("target_reuse_gap"),
                    cross=row.get("cross_domain_target_gap"),
                    resid=row.get("object_residual_stability_gap"),
                )
            )
    lines += ["", "## Top Metric Rows", ""]
    lines.append("| model | relation | hidden | rows | labels | acc | reuse gap | same target | diff target | cross target gap |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in payload.get("top_metric_rows") or []:
        lines.append(
            "| {model} | {relation} | {hidden_idx} | {rows} | {target_label_count} | {template_holdout_centroid_accuracy} | {target_reuse_gap} | {same_target_mean_cos} | {diff_target_mean_cos} | {cross_domain_target_gap} |".format(
                **row
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="semantic_reuse_difference_state_atlas")
    parser.add_argument("--domains", default="")
    parser.add_argument("--relations", default="category,color,function")
    parser.add_argument("--max-objects-per-domain", type=int, default=6)
    parser.add_argument("--templates-per-relation", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--attn-implementations", default="flash_attention_2,sdpa")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize-round", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.summarize_round:
        payload = summarize_round(args.round_name)
        print(json.dumps({"phase": PHASE, "status": payload["status"], "evidence": payload["evidence_counts"]}, ensure_ascii=False, indent=2))
        return
    if not args.model:
        raise SystemExit("--model is required unless --summarize-round is set")
    eval_model(args)


if __name__ == "__main__":
    main()
