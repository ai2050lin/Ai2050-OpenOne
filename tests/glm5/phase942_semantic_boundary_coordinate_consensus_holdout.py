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

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase938_semantic_factor_causal_transfer_audit as p938  # noqa: E402
import phase939_bilingual_specificity_tightening_audit as p939  # noqa: E402
import phase940_semantic_boundary_bridge_audit as p940  # noqa: E402
import phase941_semantic_direction_coordinate_bridge_audit as p941  # noqa: E402


PHASE = 942
MODELS = ["qwen3", "glm4", "deepseek7b"]
RESULT_ROOT = Path("tests/result/phase942_semantic_boundary_coordinate_consensus_holdout")


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


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


def finite(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def mean(values: list[float | int | None]) -> float | None:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return None if not vals else float(sum(vals) / len(vals))


def parse_csv(text: str) -> list[str]:
    return [part.strip() for part in str(text or "").split(",") if part.strip()]


def parse_ints(text: str) -> list[int]:
    out = []
    for part in parse_csv(text):
        try:
            value = int(part)
        except ValueError:
            continue
        if value > 0 and value not in out:
            out.append(value)
    return out


def norm(vec: torch.Tensor | None) -> float:
    return p941.norm(vec)


def topk_indices(direction: torch.Tensor, k: int, largest: bool = True) -> list[int]:
    vec = direction.float().cpu()
    use_k = max(1, min(int(k), int(vec.numel())))
    return [int(x) for x in torch.topk(vec.abs(), use_k, largest=largest).indices.tolist()]


def masked_direction(direction: torch.Tensor, indices: list[int]) -> torch.Tensor | None:
    vec = direction.float().cpu()
    if not indices or int(vec.numel()) <= 0:
        return None
    valid = sorted({int(idx) for idx in indices if 0 <= int(idx) < int(vec.numel())})
    if not valid:
        return None
    out = torch.zeros_like(vec)
    out[torch.tensor(valid, dtype=torch.long)] = vec[torch.tensor(valid, dtype=torch.long)]
    return out if norm(out) > 0 else None


def random_mask_direction(direction: torch.Tensor, k: int, salt: str) -> torch.Tensor | None:
    return p941.random_k_direction(direction, int(k), salt)


def spec_sort_key(spec: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(spec.get("target_label")),
        str(spec.get("train_template")),
        str(spec.get("test_template")),
        str(len(spec.get("test_samples") or [])),
    )


def select_candidate_specs(
    specs: list[dict[str, Any]],
    selected_pairs: set[tuple[str, str]],
    max_specs_per_pair: int,
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for spec in specs:
        key = (str(spec.get("relation")), str(spec.get("language_pair")))
        if key not in selected_pairs or spec.get("specific_direction") is None:
            continue
        grouped[key].append(spec)
    out: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for key, items in grouped.items():
        ordered = sorted(items, key=spec_sort_key)
        if int(max_specs_per_pair) > 0:
            ordered = ordered[: int(max_specs_per_pair)]
        out[key] = ordered
    return out


def split_specs(items: list[dict[str, Any]], train_fraction: float, min_train: int, min_holdout: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ordered = sorted(items, key=spec_sort_key)
    if len(ordered) < int(min_train) + int(min_holdout):
        return [], []
    train_count = max(int(min_train), int(round(len(ordered) * float(train_fraction))))
    train_count = min(train_count, len(ordered) - int(min_holdout))
    train = ordered[:train_count]
    holdout = ordered[train_count:]
    return train, holdout


def consensus_indices(train_specs: list[dict[str, Any]], top_k: int, consensus_k: int) -> tuple[list[int], dict[str, Any]]:
    counts: Counter[int] = Counter()
    mag: Counter[int] = Counter()
    signed: Counter[int] = Counter()
    for spec in train_specs:
        direction = spec.get("specific_direction")
        if direction is None:
            continue
        vec = direction.float().cpu()
        for idx in topk_indices(vec, int(top_k), largest=True):
            counts[idx] += 1
            mag[idx] += float(abs(vec[idx].item()))
            signed[idx] += 1 if float(vec[idx].item()) >= 0 else -1
    ranked = sorted(counts, key=lambda idx: (-counts[idx], -mag[idx], idx))
    selected = ranked[: max(1, min(int(consensus_k), len(ranked)))]
    coverage_counts = Counter(counts[idx] for idx in selected)
    meta = {
        "train_spec_count": len(train_specs),
        "vote_pool_size": len(ranked),
        "consensus_size": len(selected),
        "max_vote": max(counts.values()) if counts else 0,
        "mean_vote_selected": mean([counts[idx] for idx in selected]),
        "vote_hist_selected": {str(k): int(v) for k, v in sorted(coverage_counts.items())},
        "mean_abs_weight_selected": mean([mag[idx] for idx in selected]),
        "mean_abs_signed_vote_selected": mean([abs(signed[idx]) for idx in selected]),
    }
    return selected, meta


def overlap_metrics(indices: list[int], direction: torch.Tensor, top_k: int) -> dict[str, Any]:
    own = set(topk_indices(direction, int(top_k), largest=True))
    consensus = set(int(x) for x in indices)
    inter = own & consensus
    union = own | consensus
    return {
        "own_topk_size": len(own),
        "consensus_size": len(consensus),
        "overlap_count": len(inter),
        "overlap_recall": None if not own else float(len(inter) / len(own)),
        "overlap_precision": None if not consensus else float(len(inter) / len(consensus)),
        "overlap_jaccard": None if not union else float(len(inter) / len(union)),
    }


def make_row(
    model_name: str,
    sample: dict[str, Any],
    spec: dict[str, Any],
    condition: str,
    alpha: float | None,
    consensus_k: int | None,
    top_k: int | None,
    direction: torch.Tensor | None,
    full_direction: torch.Tensor | None,
    consensus_meta: dict[str, Any],
    overlap: dict[str, Any],
    base: dict[str, Any],
    patched: dict[str, Any],
) -> dict[str, Any]:
    row = p940.make_row(model_name, sample, spec, condition, alpha, base, patched)
    row["phase"] = PHASE
    row["row_kind"] = "phase942_semantic_boundary_coordinate_consensus_holdout_row"
    row["consensus_k"] = consensus_k
    row["top_k_for_votes"] = top_k
    row["direction_norm_used"] = norm(direction)
    row["full_specific_direction_norm"] = norm(full_direction)
    row["norm_fraction_vs_full"] = None
    if full_direction is not None and norm(full_direction) > 0:
        row["norm_fraction_vs_full"] = row["direction_norm_used"] / norm(full_direction)
    for key, value in consensus_meta.items():
        if key != "indices":
            row[f"consensus_{key}"] = value
    for key, value in overlap.items():
        row[key] = value
    return row


def condition_vectors(
    spec: dict[str, Any],
    consensus: list[int],
    top_k: int,
) -> list[tuple[str, torch.Tensor | None]]:
    direction = spec.get("specific_direction")
    if direction is None:
        return []
    full = direction.float().cpu()
    full_norm = norm(full)
    own_raw = p941.topk_direction(full, int(top_k), largest=True)
    consensus_raw = masked_direction(full, consensus)
    bottom_raw = p941.topk_direction(full, int(top_k), largest=False)
    salt = "|".join(
        [
            str(spec.get("relation")),
            str(spec.get("target_label")),
            str(spec.get("train_template")),
            str(spec.get("test_template")),
            str(top_k),
            "phase942",
        ]
    )
    return [
        ("full_specific", full),
        ("own_topk_raw", own_raw),
        ("own_topk_same_norm", p941.scale_to(own_raw, full_norm)),
        ("consensus_raw", consensus_raw),
        ("consensus_same_norm", p941.scale_to(consensus_raw, full_norm)),
        ("randomk_same_norm", random_mask_direction(full, int(top_k), salt)),
        ("bottomk_same_norm", p941.scale_to(bottom_raw, full_norm)),
    ]


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "target_logit_delta_mean": mean([row.get("target_label_logit_delta") for row in rows]),
        "relation_margin_delta_mean": mean([row.get("target_margin_vs_relation_best_other_delta") for row in rows]),
        "boundary_margin_delta_mean": mean([row.get("target_vs_all_boundary_margin_delta") for row in rows]),
        "period_margin_delta_mean": mean([row.get("target_vs_punctuation_period_margin_delta") for row in rows]),
        "eos_margin_delta_mean": mean([row.get("target_vs_eos_margin_delta") for row in rows]),
        "boundary_logit_delta_mean": mean([row.get("all_boundary_logit_delta") for row in rows]),
        "overlap_recall_mean": mean([row.get("overlap_recall") for row in rows]),
        "overlap_jaccard_mean": mean([row.get("overlap_jaccard") for row in rows]),
        "norm_fraction_mean": mean([row.get("norm_fraction_vs_full") for row in rows]),
        "rank_improved": sum(1 for row in rows if row.get("rank_improved")),
        "new_relation_winner_target": sum(1 for row in rows if row.get("new_relation_winner_target")),
    }


def summarize_by(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[tuple(row.get(key) for key in keys)].append(row)
    out = []
    for key_tuple, items in buckets.items():
        item = {key: value for key, value in zip(keys, key_tuple)}
        item.update(summarize_rows(items))
        out.append(item)
    out.sort(
        key=lambda row: (
            finite(row.get("boundary_margin_delta_mean"), -999.0),
            finite(row.get("relation_margin_delta_mean"), -999.0),
        ),
        reverse=True,
    )
    return out


def consensus_eval_rows(group_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, Any, Any, Any, Any], list[dict[str, Any]]] = defaultdict(list)
    for row in group_rows:
        buckets[
            (
                row.get("relation"),
                row.get("language_pair"),
                row.get("consensus_k"),
                row.get("top_k_for_votes"),
                row.get("alpha"),
            )
        ].append(row)
    out = []
    for (relation, language_pair, consensus_k, top_k, alpha), rows in buckets.items():
        if alpha is None:
            continue
        by_cond = {row.get("condition"): row for row in rows}
        full = by_cond.get("full_specific")
        consensus_raw = by_cond.get("consensus_raw")
        consensus_same = by_cond.get("consensus_same_norm")
        own_raw = by_cond.get("own_topk_raw")
        random_row = by_cond.get("randomk_same_norm")
        bottom_row = by_cond.get("bottomk_same_norm")
        if not full or not consensus_raw or not consensus_same:
            continue
        controls = [row for row in [random_row, bottom_row] if row is not None]
        control_best = None if not controls else max(finite(row.get("boundary_margin_delta_mean"), -999.0) for row in controls)
        full_boundary = finite(full.get("boundary_margin_delta_mean"), 0.0)
        consensus_raw_boundary = finite(consensus_raw.get("boundary_margin_delta_mean"), 0.0)
        consensus_same_boundary = finite(consensus_same.get("boundary_margin_delta_mean"), 0.0)
        out.append(
            {
                "relation": relation,
                "language_pair": language_pair,
                "consensus_k": consensus_k,
                "top_k_for_votes": top_k,
                "alpha": alpha,
                "full_relation_margin_delta_mean": full.get("relation_margin_delta_mean"),
                "full_boundary_margin_delta_mean": full.get("boundary_margin_delta_mean"),
                "own_topk_boundary_margin_delta_mean": None if not own_raw else own_raw.get("boundary_margin_delta_mean"),
                "consensus_raw_relation_margin_delta_mean": consensus_raw.get("relation_margin_delta_mean"),
                "consensus_raw_boundary_margin_delta_mean": consensus_raw.get("boundary_margin_delta_mean"),
                "consensus_same_relation_margin_delta_mean": consensus_same.get("relation_margin_delta_mean"),
                "consensus_same_boundary_margin_delta_mean": consensus_same.get("boundary_margin_delta_mean"),
                "control_best_boundary_margin_delta_mean": control_best,
                "consensus_raw_fraction_of_full": None
                if full_boundary <= 0
                else float(consensus_raw_boundary / full_boundary),
                "consensus_same_gain_vs_control": None
                if control_best is None
                else float(consensus_same_boundary - control_best),
                "joint_consensus_holdout_score": None
                if control_best is None
                else float(
                    min(
                        consensus_same_boundary,
                        finite(consensus_same.get("relation_margin_delta_mean"), -999.0),
                        consensus_same_boundary - control_best,
                    )
                ),
                "overlap_recall_mean": consensus_same.get("overlap_recall_mean"),
                "overlap_jaccard_mean": consensus_same.get("overlap_jaccard_mean"),
                "rows": consensus_same.get("rows"),
            }
        )
    out.sort(
        key=lambda row: (
            finite(row.get("joint_consensus_holdout_score"), -999.0),
            finite(row.get("consensus_same_gain_vs_control"), -999.0),
            finite(row.get("consensus_raw_fraction_of_full"), -999.0),
        ),
        reverse=True,
    )
    return out


def evidence_label(rows: list[dict[str, Any]]) -> str:
    positives = [
        row
        for row in rows
        if finite(row.get("consensus_raw_fraction_of_full"), 0.0) >= 0.10
        and finite(row.get("consensus_same_gain_vs_control"), -999.0) > 0.02
        and finite(row.get("joint_consensus_holdout_score"), -999.0) > 0.02
    ]
    rels = {str(row.get("relation")) for row in positives}
    if len(rels) >= 2:
        return "consensus_coordinate_holdout_positive"
    if positives:
        return "partial_consensus_coordinate_holdout_positive"
    return "consensus_coordinate_holdout_weak_or_controlled"


def eval_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    samples = p939.build_samples(args)
    selected_pairs, selected_phase940_rows = p941.selected_positive_pairs(args)
    hidden_by_relation = p938.phase937_best_hidden(args.model, args.phase937_round)
    if not hidden_by_relation:
        hidden_by_relation = {relation: -1 for relation in parse_csv(args.relations)}
    topks = parse_ints(args.topks)
    consensus_ks = parse_ints(args.consensus_ks)
    dry_payload = {
        "phase": PHASE,
        "title": "Semantic Boundary Coordinate Consensus Holdout",
        "model": args.model,
        "sample_count": len(samples),
        "selected_relation_language_pairs": sorted([f"{rel}:{pair}" for rel, pair in selected_pairs]),
        "selected_phase940_bridge_rows": selected_phase940_rows,
        "requested_topks": topks,
        "requested_consensus_ks": consensus_ks,
        "hidden_by_relation": hidden_by_relation,
    }
    if args.dry_run:
        payload = {**dry_payload, "status": "dry_run"}
        write_json(out_dir / f"phase942_{args.model}_summary.json", payload)
        write_jsonl(out_dir / f"phase942_{args.model}_rows.jsonl", [])
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return payload

    model = None
    tokenizer = None
    attn_impl = None
    rows: list[dict[str, Any]] = []
    consensus_records: list[dict[str, Any]] = []
    direction_specs: list[dict[str, Any]] = []
    try:
        model, tokenizer, device, attn_impl = p938.p862.p844.p828.p796.load_model_bf16_prefer_flash(
            args.model, args.attn_implementations
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        auto_indices = p940.p937.auto_hidden_indices(model)
        hidden_by_relation = {
            rel: (auto_indices[len(auto_indices) // 2] if int(idx) < 0 else int(idx))
            for rel, idx in hidden_by_relation.items()
        }
        hidden_indices = sorted(set(hidden_by_relation.values()))
        vectors = p938.forward_vectors(model, tokenizer, device, samples, hidden_indices, int(args.batch_size))
        baseline_logits = p938.forward_logits(model, tokenizer, device, samples, int(args.batch_size))
        direction_specs = p939.build_direction_specs(samples, vectors, hidden_by_relation, int(args.min_train_per_label))
        grouped_specs = select_candidate_specs(direction_specs, selected_pairs, int(args.max_specs_per_pair))
        labels_by_relation = {relation: p938.relation_labels(samples, relation) for relation in hidden_by_relation}
        token_maps = {relation: p938.label_token_map(tokenizer, labels) for relation, labels in labels_by_relation.items()}
        boundary_groups = p940.boundary_token_groups(tokenizer)
        baseline_metrics: dict[str, dict[str, Any]] = {}
        for sample in samples:
            relation = str(sample["relation"])
            baseline_metrics[p938.sample_key(sample)] = p940.boundary_metrics(
                tokenizer,
                baseline_logits[p938.sample_key(sample)],
                str(sample["target_label"]),
                token_maps[relation],
                boundary_groups,
            )

        group_count = 0
        for key in sorted(grouped_specs):
            specs = grouped_specs[key]
            train_specs, holdout_specs = split_specs(
                specs,
                float(args.train_fraction),
                int(args.min_train_specs),
                int(args.min_holdout_specs),
            )
            if not train_specs or not holdout_specs:
                continue
            relation, language_pair = key
            label_tokens = token_maps[relation]
            for top_k in topks:
                for consensus_k in consensus_ks:
                    indices, meta = consensus_indices(train_specs, int(top_k), int(consensus_k))
                    meta = {
                        **meta,
                        "indices": indices,
                        "relation": relation,
                        "language_pair": language_pair,
                        "top_k_for_votes": int(top_k),
                        "consensus_k": int(consensus_k),
                        "holdout_spec_count": len(holdout_specs),
                    }
                    consensus_records.append({k: v for k, v in meta.items() if k != "indices"})
                    for spec in holdout_specs:
                        full_direction = spec.get("specific_direction")
                        if full_direction is None:
                            continue
                        overlap = overlap_metrics(indices, full_direction, int(top_k))
                        for sample in spec["test_samples"]:
                            base = baseline_metrics[p938.sample_key(sample)]
                            rows.append(
                                make_row(
                                    args.model,
                                    sample,
                                    spec,
                                    "baseline",
                                    None,
                                    int(consensus_k),
                                    int(top_k),
                                    None,
                                    full_direction,
                                    meta,
                                    overlap,
                                    base,
                                    base,
                                )
                            )
                        for condition, direction in condition_vectors(spec, indices, int(top_k)):
                            if direction is None or norm(direction) <= 0:
                                continue
                            for alpha in [float(x) for x in parse_csv(args.alphas)]:
                                patched_logits = p938.patched_logits_batch(
                                    model,
                                    tokenizer,
                                    device,
                                    spec["test_samples"],
                                    int(spec["hidden_idx"]),
                                    direction,
                                    float(alpha),
                                    int(args.batch_size),
                                )
                                for sample in spec["test_samples"]:
                                    base = baseline_metrics[p938.sample_key(sample)]
                                    patched = p940.boundary_metrics(
                                        tokenizer,
                                        patched_logits[p938.sample_key(sample)],
                                        str(sample["target_label"]),
                                        label_tokens,
                                        boundary_groups,
                                    )
                                    rows.append(
                                        make_row(
                                            args.model,
                                            sample,
                                            spec,
                                            condition,
                                            float(alpha),
                                            int(consensus_k),
                                            int(top_k),
                                            direction,
                                            full_direction,
                                            meta,
                                            overlap,
                                            base,
                                            patched,
                                        )
                                    )
                    group_count += 1
                    log(
                        f"{args.model}/{args.round_name}: pair={relation}:{language_pair} top_k={top_k} consensus_k={consensus_k} train={len(train_specs)} holdout={len(holdout_specs)} rows={len(rows)}"
                    )
    finally:
        if model is not None:
            p938.p862.p844.p828.release_model(model)
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    by_condition = summarize_by(rows, ["condition", "consensus_k", "top_k_for_votes", "alpha"])
    by_relation_language_condition = summarize_by(rows, ["relation", "language_pair", "condition", "consensus_k", "top_k_for_votes", "alpha"])
    by_pair = summarize_by(rows, ["relation", "language_pair", "condition", "consensus_k", "top_k_for_votes", "alpha"])
    evaluations = consensus_eval_rows(by_relation_language_condition)
    evidence = evidence_label(evaluations)
    payload = {
        **dry_payload,
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "hidden_by_relation": hidden_by_relation,
        "direction_specs_all": len(direction_specs),
        "candidate_pair_count": len(grouped_specs),
        "consensus_record_count": len(consensus_records),
        "rows": len(rows),
        "overall": summarize_rows(rows),
        "consensus_records": consensus_records,
        "by_condition": by_condition,
        "by_pair_condition": by_pair,
        "consensus_eval_rows": evaluations,
        "evidence_label": evidence,
        "boundary": "spec-level consensus-coordinate holdout; not object-level holdout or MLP-channel closure",
    }
    write_json(out_dir / f"phase942_{args.model}_summary.json", payload)
    write_jsonl(out_dir / f"phase942_{args.model}_rows.jsonl", rows)
    print(
        json.dumps(
            {
                "phase": PHASE,
                "model": args.model,
                "status": "complete",
                "evidence": evidence,
                "rows": len(rows),
                "top_consensus_rows": evaluations[:12],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return payload


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    summaries = [read_json(out_dir / f"phase942_{model}_summary.json") for model in MODELS]
    summaries = [summary for summary in summaries if summary]
    evidence_counts = Counter(str(summary.get("evidence_label") or summary.get("status")) for summary in summaries)
    condition_rows = []
    consensus_rows = []
    for summary in summaries:
        model_name = summary.get("model")
        for row in summary.get("by_condition") or []:
            item = dict(row)
            item["model"] = model_name
            condition_rows.append(item)
        for row in summary.get("consensus_eval_rows") or []:
            item = dict(row)
            item["model"] = model_name
            consensus_rows.append(item)
    consensus_rows.sort(
        key=lambda row: (
            finite(row.get("joint_consensus_holdout_score"), -999.0),
            finite(row.get("consensus_same_gain_vs_control"), -999.0),
            finite(row.get("consensus_raw_fraction_of_full"), -999.0),
        ),
        reverse=True,
    )
    payload = {
        "phase": PHASE,
        "round": round_name,
        "status": "complete" if summaries else "missing",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": [summary.get("model") for summary in summaries],
        "evidence_counts": dict(evidence_counts),
        "model_summaries": summaries,
        "condition_rows": condition_rows,
        "top_consensus_rows": consensus_rows[:220],
    }
    write_json(out_dir / "phase942_cross_model_summary.json", payload)
    write_summary_md(out_dir / "phase942_cross_model_summary.md", payload)
    return payload


def write_summary_md(path: Path, payload: dict[str, Any]) -> None:
    lines = ["# Phase 942 semantic boundary coordinate consensus holdout", ""]
    lines += ["## Evidence", ""]
    for key, value in sorted((payload.get("evidence_counts") or {}).items()):
        lines.append(f"- {key}: {value}")
    lines += ["", "## Condition Rows", ""]
    lines.append("| model | condition | consensus_k | top_k | alpha | rows | relation margin | boundary margin | overlap recall | overlap jaccard |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in payload.get("condition_rows") or []:
        lines.append(
            "| {model} | {condition} | {consensus_k} | {top_k_for_votes} | {alpha} | {rows} | {relation_margin_delta_mean} | {boundary_margin_delta_mean} | {overlap_recall_mean} | {overlap_jaccard_mean} |".format(
                **row
            )
        )
    lines += ["", "## Top Consensus Holdout Rows", ""]
    lines.append("| model | relation | pair | consensus_k | top_k | alpha | full boundary | consensus raw | raw fraction | consensus same | control best | gain | joint | overlap recall |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in payload.get("top_consensus_rows") or []:
        lines.append(
            "| {model} | {relation} | {language_pair} | {consensus_k} | {top_k_for_votes} | {alpha} | {full_boundary_margin_delta_mean} | {consensus_raw_boundary_margin_delta_mean} | {consensus_raw_fraction_of_full} | {consensus_same_boundary_margin_delta_mean} | {control_best_boundary_margin_delta_mean} | {consensus_same_gain_vs_control} | {joint_consensus_holdout_score} | {overlap_recall_mean} |".format(
                **row
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="semantic_boundary_coordinate_consensus_holdout")
    parser.add_argument("--phase937-round", default="semantic_reuse_difference_state_atlas")
    parser.add_argument("--phase939-round", default="bilingual_specificity_tightening_audit")
    parser.add_argument("--phase940-round", default="semantic_boundary_bridge_audit")
    parser.add_argument("--domains", default="")
    parser.add_argument("--relations", default="category,color,function")
    parser.add_argument("--max-objects-per-domain", type=int, default=6)
    parser.add_argument("--templates-per-language", type=int, default=2)
    parser.add_argument("--min-train-per-label", type=int, default=2)
    parser.add_argument("--min-specific-margin", type=float, default=0.05)
    parser.add_argument("--min-specific-gain", type=float, default=0.05)
    parser.add_argument("--min-phase940-bridge-gain", type=float, default=0.02)
    parser.add_argument("--max-specs-per-pair", type=int, default=12)
    parser.add_argument("--train-fraction", type=float, default=0.5)
    parser.add_argument("--min-train-specs", type=int, default=4)
    parser.add_argument("--min-holdout-specs", type=int, default=3)
    parser.add_argument("--topks", default="256")
    parser.add_argument("--consensus-ks", default="256")
    parser.add_argument("--alphas", default="0.5,1.0")
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
