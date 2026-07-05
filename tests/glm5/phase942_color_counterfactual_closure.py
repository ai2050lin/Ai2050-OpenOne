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

from phase941_color_feature_neuron_atlas import (  # noqa: E402
    RESULT_ROOT as PHASE941_RESULT_ROOT,
    article_for,
    build_support_vectors,
    color_margin_for_logits,
    decode_token,
    encode_batch,
    forward_with_channel_scale,
    get_device_for_input,
    get_down_proj,
    get_layers,
    label_score,
    load_model_bf16,
    mean,
    parse_csv,
    read_jsonl,
    release_model,
    select_layers,
    write_json,
    write_jsonl,
)


PHASE = 942
MODELS = ["qwen3", "deepseek7b", "glm4"]
RESULT_ROOT = Path("tests/result/phase942_color_counterfactual_closure")
PHASE941_ROUNDS = {
    "qwen3": "color_feature_neuron_atlas_qwen3_full",
    "deepseek7b": "color_feature_neuron_atlas_deepseek7b_full",
    "glm4": "color_feature_neuron_atlas_glm4_full",
}

DEFAULT_COLORS = [
    "black",
    "blue",
    "brown",
    "gray",
    "green",
    "orange",
    "purple",
    "red",
    "silver",
    "white",
    "yellow",
]

NEUTRAL_OBJECTS = [
    {"domain": "shape", "object": "cube"},
    {"domain": "object", "object": "card"},
    {"domain": "toy", "object": "block"},
    {"domain": "container", "object": "box"},
    {"domain": "shape", "object": "disk"},
    {"domain": "object", "object": "marker"},
]

COUNTERFACTUAL_TEMPLATES = [
    "A {color} {object} is placed on the table. The color of the {object} is",
    "In the picture, the {object} is {color}. The object's color is",
    "The label says this {object} is {color}. One word for the color is",
]


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def phase941_summary_path(model_name: str) -> Path:
    round_name = PHASE941_ROUNDS[model_name]
    return PHASE941_RESULT_ROOT / round_name / f"phase941_{model_name}_summary.json"


def phase941_cross_atlas_path() -> Path:
    return PHASE941_RESULT_ROOT / "cross_model_atlas" / "phase941_color_cross_model_atlas.json"


def stable_channel_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if int(row.get("coverage_objects") or 0) >= 3
        and int(row.get("coverage_templates") or 0) >= 3
        and int(row.get("count") or 0) >= 8
    ]


def load_candidates(model_name: str, labels: list[str], intervention_top_channels: int) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    atlas = load_json(phase941_cross_atlas_path())
    summary = load_json(phase941_summary_path(model_name))

    candidates: dict[str, dict[str, Any]] = {}
    intervention_groups: dict[str, dict[str, Any]] = {}
    for label in labels:
        atlas_row = (atlas.get("colors") or {}).get(label, {}).get(model_name)
        if atlas_row:
            candidates[label] = {
                "target_label": label,
                "layer": int(atlas_row["layer"]),
                "channel": int(atlas_row["channel"]),
                "phase941_score": float(atlas_row["effective_score"]),
                "phase941_mean_contribution": float(atlas_row["mean_contribution"]),
                "phase941_selectivity_delta": float(atlas_row["selectivity_delta"]),
                "phase941_causal_grade": atlas_row.get("causal_grade"),
            }

        top_rows = list((summary.get("top_channels_by_label") or {}).get(label) or [])
        stable = stable_channel_rows(top_rows)
        group_source = stable or top_rows
        if group_source:
            top_layer = int(group_source[0]["layer"])
            channels = [
                int(row["channel"])
                for row in group_source
                if int(row["layer"]) == top_layer
            ][: max(1, int(intervention_top_channels))]
            intervention_groups[label] = {
                "target_label": label,
                "layer": top_layer,
                "channels": channels,
            }

    missing = sorted(set(labels) - set(candidates))
    if missing:
        raise ValueError(f"Missing Phase941 candidates for {model_name}: {missing}")
    return candidates, intervention_groups


def build_counterfactual_samples(args: argparse.Namespace) -> list[dict[str, Any]]:
    labels = parse_csv(args.colors) if args.colors else list(DEFAULT_COLORS)
    objects = NEUTRAL_OBJECTS[: max(1, int(args.objects))]
    templates = COUNTERFACTUAL_TEMPLATES[: max(1, int(args.templates_per_object))]
    samples: list[dict[str, Any]] = []
    for color in labels:
        for obj_idx, obj in enumerate(objects):
            fmt = {
                **obj,
                "color": color,
                "article": article_for(obj["object"]),
            }
            for template_idx, template in enumerate(templates):
                samples.append(
                    {
                        "phase": PHASE,
                        "sample_id": f"cf_color:{color}:{obj['object'].replace(' ', '_')}:t{template_idx}",
                        "relation": "counterfactual_color",
                        "domain": obj["domain"],
                        "object": obj["object"],
                        "object_index": obj_idx,
                        "target_label": color,
                        "prompt_template": f"counterfactual_color_{template_idx}",
                        "prompt": template.format(**fmt),
                    }
                )
    return samples


def contribution_for_candidate(
    captured: dict[int, torch.Tensor],
    row_idx: int,
    candidate: dict[str, Any],
    label: str,
    support_vectors: dict[tuple[int, str], torch.Tensor],
) -> dict[str, float | int]:
    layer_idx = int(candidate["layer"])
    channel = int(candidate["channel"])
    activation = float(captured[layer_idx][row_idx][channel].item())
    support = support_vectors[(layer_idx, label)]
    readout_support = float(support[channel].item())
    contribution = float(activation * readout_support)
    denom = max(abs(float(candidate.get("phase941_mean_contribution") or 0.0)), 1e-6)
    normalized = contribution / denom
    return {
        "layer": layer_idx,
        "channel": channel,
        "activation": activation,
        "readout_support": readout_support,
        "contribution": contribution,
        "normalized_contribution": normalized,
    }


def update_candidate_bucket(bucket: dict[str, Any], sample: dict[str, Any], contribution: float, normalized: float) -> None:
    bucket["count"] += 1
    bucket["sum"] += contribution
    bucket["abs_sum"] += abs(contribution)
    bucket["norm_sum"] += normalized
    bucket["norm_abs_sum"] += abs(normalized)
    bucket["positive_count"] += 1 if contribution > 0 else 0
    bucket["object_values"][str(sample["object"])].append(contribution)
    bucket["template_values"][str(sample["prompt_template"])].append(contribution)


def new_bucket() -> dict[str, Any]:
    return {
        "count": 0,
        "sum": 0.0,
        "abs_sum": 0.0,
        "norm_sum": 0.0,
        "norm_abs_sum": 0.0,
        "positive_count": 0,
        "object_values": defaultdict(list),
        "template_values": defaultdict(list),
    }


def bucket_mean(bucket: dict[str, Any], key: str = "sum") -> float:
    count = max(1, int(bucket["count"]))
    return float(bucket[key] / count)


def object_std(bucket: dict[str, Any]) -> float | None:
    object_means = [
        float(sum(values) / len(values))
        for values in bucket["object_values"].values()
        if values
    ]
    if len(object_means) < 2:
        return None
    mean_value = sum(object_means) / len(object_means)
    variance = sum((value - mean_value) ** 2 for value in object_means) / len(object_means)
    return math.sqrt(variance)


def make_candidate_row(
    model_name: str,
    label: str,
    candidate: dict[str, Any],
    own: dict[str, Any],
    other: dict[str, Any],
    target_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    own_mean = bucket_mean(own)
    other_mean = bucket_mean(other) if other["count"] else 0.0
    own_norm_mean = bucket_mean(own, "norm_sum")
    other_norm_mean = bucket_mean(other, "norm_sum") if other["count"] else 0.0
    target_feature_top1_raw = sum(1 for row in target_rows if row.get("feature_pred_label_raw") == label)
    target_feature_top1_norm = sum(1 for row in target_rows if row.get("feature_pred_label_norm") == label)
    target_feature_count = max(1, len(target_rows))
    norm_margins = [
        float(row["feature_margin_norm"])
        for row in target_rows
        if row.get("feature_margin_norm") is not None
    ]
    raw_margins = [
        float(row["feature_margin_raw"])
        for row in target_rows
        if row.get("feature_margin_raw") is not None
    ]
    delta = own_mean - other_mean
    norm_delta = own_norm_mean - other_norm_mean
    row = {
        "phase": PHASE,
        "model": model_name,
        "target_label": label,
        "layer": int(candidate["layer"]),
        "channel": int(candidate["channel"]),
        "phase941_score": float(candidate["phase941_score"]),
        "phase941_causal_grade": candidate.get("phase941_causal_grade"),
        "own_count": int(own["count"]),
        "other_count": int(other["count"]),
        "own_mean_contribution": own_mean,
        "other_mean_contribution": other_mean,
        "counterfactual_delta": delta,
        "own_mean_normalized": own_norm_mean,
        "other_mean_normalized": other_norm_mean,
        "counterfactual_delta_normalized": norm_delta,
        "own_positive_rate": float(own["positive_count"] / max(1, own["count"])),
        "object_mean_std": object_std(own),
        "feature_top1_raw": target_feature_top1_raw,
        "feature_top1_norm": target_feature_top1_norm,
        "feature_top1_norm_rate": float(target_feature_top1_norm / target_feature_count),
        "feature_margin_raw_mean": mean(raw_margins),
        "feature_margin_norm_mean": mean(norm_margins),
    }
    row["counterfactual_score"] = float(
        norm_delta
        + row["feature_top1_norm_rate"]
        + 0.1 * row["own_positive_rate"]
        - 0.02 * (row["object_mean_std"] or 0.0)
    )
    if norm_delta > 0.25 and row["feature_top1_norm_rate"] >= 0.5 and row["own_positive_rate"] >= 0.75:
        grade = "strong_counterfactual"
    elif norm_delta > 0.1 and row["own_positive_rate"] >= 0.6:
        grade = "directional_counterfactual"
    elif norm_delta > 0.0:
        grade = "weak_counterfactual"
    else:
        grade = "object_or_competition_mixed"
    row["counterfactual_grade"] = grade
    return row


def run_counterfactual_interventions(
    args: argparse.Namespace,
    model,
    tokenizer,
    device: torch.device,
    samples: list[dict[str, Any]],
    labels: list[str],
    intervention_groups: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    if not args.run_interventions:
        return []
    factors = [float(x) for x in parse_csv(args.intervention_factors)]
    rows: list[dict[str, Any]] = []
    for label in labels[: int(args.max_intervention_labels)]:
        group = intervention_groups.get(label)
        if not group or not group.get("channels"):
            continue
        target_samples = [sample for sample in samples if str(sample["target_label"]) == label]
        target_samples = target_samples[: int(args.max_intervention_samples_per_label)]
        if not target_samples:
            continue
        base_logits = forward_with_channel_scale(
            model,
            tokenizer,
            device,
            target_samples,
            int(group["layer"]),
            list(group["channels"]),
            1.0,
            int(args.batch_size),
        )
        for factor in factors:
            patched_logits = forward_with_channel_scale(
                model,
                tokenizer,
                device,
                target_samples,
                int(group["layer"]),
                list(group["channels"]),
                factor,
                int(args.batch_size),
            )
            for sample in target_samples:
                sample_id = str(sample["sample_id"])
                base = color_margin_for_logits(tokenizer, base_logits[sample_id], labels, label)
                patched = color_margin_for_logits(tokenizer, patched_logits[sample_id], labels, label)
                rows.append(
                    {
                        "phase": PHASE,
                        "model": args.model,
                        "sample_id": sample_id,
                        "target_label": label,
                        "layer": int(group["layer"]),
                        "channels": list(group["channels"]),
                        "factor": float(factor),
                        "base_margin": base.get("margin"),
                        "patched_margin": patched.get("margin"),
                        "margin_delta": None
                        if base.get("margin") is None or patched.get("margin") is None
                        else float(patched["margin"]) - float(base["margin"]),
                        "base_target_rank": base.get("target_rank"),
                        "patched_target_rank": patched.get("target_rank"),
                        "evidence_level": "counterfactual_color_channel_group_scale",
                    }
                )
    return rows


def scan_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    samples = build_counterfactual_samples(args)
    labels = sorted({str(sample["target_label"]) for sample in samples})
    candidates, intervention_groups = load_candidates(args.model, labels, int(args.intervention_top_channels))
    selected_layers = sorted(
        {
            int(candidate["layer"])
            for candidate in candidates.values()
        }
        | {
            int(group["layer"])
            for group in intervention_groups.values()
        }
    )

    dry_payload = {
        "phase": PHASE,
        "title": "Color Counterfactual Closure",
        "model": args.model,
        "sample_count": len(samples),
        "labels": labels,
        "objects": sorted({str(sample["object"]) for sample in samples}),
        "templates_per_object": int(args.templates_per_object),
        "selected_candidate_layers": selected_layers,
        "candidate_source": "phase941_color_cross_model_atlas",
        "schema": {
            "dataset_jsonl": "factorial color x neutral object prompts",
            "sample_rows_jsonl": "one row per prompt with logits and candidate-feature prediction",
            "candidate_rows_jsonl": "one row per color candidate with counterfactual selectivity",
            "intervention_rows_jsonl": "optional causal scale validation on counterfactual prompts",
        },
    }
    if args.generate_only or args.dry_run:
        payload = {**dry_payload, "status": "generate_only" if args.generate_only else "dry_run", "samples_preview": samples[:20]}
        write_json(out_dir / f"phase942_{args.model}_summary.json", payload)
        write_jsonl(out_dir / f"phase942_{args.model}_dataset.jsonl", samples)
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return payload

    model = None
    tokenizer = None
    sample_rows: list[dict[str, Any]] = []
    candidate_sample_rows: list[dict[str, Any]] = []
    candidate_own: dict[str, dict[str, Any]] = defaultdict(new_bucket)
    candidate_other: dict[str, dict[str, Any]] = defaultdict(new_bucket)
    try:
        model, tokenizer, _device = load_model_bf16(args.model)
        device = get_device_for_input(model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        support_vectors = build_support_vectors(model, tokenizer, labels, selected_layers)[0]
        layers = get_layers(model)

        for start in range(0, len(samples), max(1, int(args.batch_size))):
            batch = samples[start : start + max(1, int(args.batch_size))]
            input_ids, attention_mask, last_pos = encode_batch(tokenizer, device, batch)
            batch_idx = torch.arange(input_ids.shape[0], device=device)
            captured: dict[int, torch.Tensor] = {}
            hooks = []

            def make_capture_hook(layer_idx: int):
                def hook(_module, module_input):
                    hidden = module_input[0]
                    idx = batch_idx.to(hidden.device)
                    pos = last_pos.to(hidden.device)
                    if hidden.dim() == 3:
                        captured[layer_idx] = hidden[idx, pos].detach().float().cpu()
                    elif hidden.dim() == 2:
                        captured[layer_idx] = hidden.detach().float().cpu()
                    return None

                return hook

            for layer_idx in selected_layers:
                hooks.append(get_down_proj(layers[layer_idx]).register_forward_pre_hook(make_capture_hook(layer_idx)))

            with torch.inference_mode():
                result = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)
            logits = result.logits[batch_idx, last_pos].detach().float().cpu()
            for hook in hooks:
                hook.remove()

            for row_idx, sample in enumerate(batch):
                target_label = str(sample["target_label"])
                label_scores = {label: label_score(tokenizer, logits[row_idx], label) for label in labels}
                target_info = label_scores.get(target_label) or {}
                competitor_items = [
                    (label, score)
                    for label, score in label_scores.items()
                    if label != target_label and score.get("logit") is not None
                ]
                best_competitor_label = None
                best_competitor_logit = None
                if competitor_items:
                    best_competitor_label, best_comp = max(competitor_items, key=lambda item: float(item[1]["logit"]))
                    best_competitor_logit = float(best_comp["logit"])
                target_logit = target_info.get("logit")
                color_margin = None
                if target_logit is not None and best_competitor_logit is not None:
                    color_margin = float(target_logit) - float(best_competitor_logit)

                raw_scores: dict[str, float] = {}
                norm_scores: dict[str, float] = {}
                detail_scores: dict[str, dict[str, float | int]] = {}
                for label in labels:
                    contrib = contribution_for_candidate(captured, row_idx, candidates[label], label, support_vectors)
                    raw_scores[label] = float(contrib["contribution"])
                    norm_scores[label] = float(contrib["normalized_contribution"])
                    detail_scores[label] = contrib
                    bucket = candidate_own[label] if label == target_label else candidate_other[label]
                    update_candidate_bucket(bucket, sample, float(contrib["contribution"]), float(contrib["normalized_contribution"]))
                    candidate_sample_rows.append(
                        {
                            "phase": PHASE,
                            "model": args.model,
                            "sample_id": sample["sample_id"],
                            "target_label": target_label,
                            "candidate_label": label,
                            "object": sample["object"],
                            "prompt_template": sample["prompt_template"],
                            **contrib,
                        }
                    )

                raw_pred = max(raw_scores, key=lambda label: raw_scores[label])
                norm_pred = max(norm_scores, key=lambda label: norm_scores[label])
                raw_other = max((value for label, value in raw_scores.items() if label != target_label), default=None)
                norm_other = max((value for label, value in norm_scores.items() if label != target_label), default=None)
                target_raw = raw_scores[target_label]
                target_norm = norm_scores[target_label]
                sample_rows.append(
                    {
                        **sample,
                        "model": args.model,
                        "target_token_id": target_info.get("token_id"),
                        "target_token": target_info.get("token"),
                        "target_logit": target_logit,
                        "target_rank": target_info.get("rank"),
                        "best_competitor_label": best_competitor_label,
                        "best_competitor_logit": best_competitor_logit,
                        "color_margin": color_margin,
                        "feature_pred_label_raw": raw_pred,
                        "feature_pred_label_norm": norm_pred,
                        "feature_correct_raw": raw_pred == target_label,
                        "feature_correct_norm": norm_pred == target_label,
                        "target_feature_contribution_raw": target_raw,
                        "target_feature_contribution_norm": target_norm,
                        "feature_margin_raw": None if raw_other is None else target_raw - raw_other,
                        "feature_margin_norm": None if norm_other is None else target_norm - norm_other,
                    }
                )

            del result, logits, input_ids, attention_mask
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if int(args.log_every) > 0 and (start // max(1, int(args.batch_size))) % int(args.log_every) == 0:
                log(f"{args.model}: processed {min(start + len(batch), len(samples))}/{len(samples)} samples")

        candidate_rows = []
        by_label_sample_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in sample_rows:
            by_label_sample_rows[str(row["target_label"])].append(row)
        for label in labels:
            candidate_rows.append(
                make_candidate_row(
                    args.model,
                    label,
                    candidates[label],
                    candidate_own[label],
                    candidate_other[label],
                    by_label_sample_rows[label],
                )
            )
        candidate_rows.sort(key=lambda row: float(row["counterfactual_score"]), reverse=True)
        intervention_rows = run_counterfactual_interventions(args, model, tokenizer, device, samples, labels, intervention_groups)
    finally:
        if model is not None:
            release_model(model)
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    target_ranks = [row.get("target_rank") for row in sample_rows if row.get("target_rank") is not None]
    margins = [row.get("color_margin") for row in sample_rows if row.get("color_margin") is not None]
    feature_norm_margins = [row.get("feature_margin_norm") for row in sample_rows if row.get("feature_margin_norm") is not None]
    payload = {
        **dry_payload,
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "sample_rows": len(sample_rows),
        "candidate_sample_rows": len(candidate_sample_rows),
        "candidate_rows": len(candidate_rows),
        "intervention_rows": len(intervention_rows),
        "target_rank_mean": mean([float(x) for x in target_ranks]),
        "target_rank_top1": sum(1 for x in target_ranks if int(x) == 1),
        "target_rank_top10": sum(1 for x in target_ranks if int(x) <= 10),
        "color_margin_mean": mean([float(x) for x in margins]),
        "feature_top1_raw": sum(1 for row in sample_rows if row.get("feature_correct_raw")),
        "feature_top1_norm": sum(1 for row in sample_rows if row.get("feature_correct_norm")),
        "feature_top1_norm_rate": sum(1 for row in sample_rows if row.get("feature_correct_norm")) / max(1, len(sample_rows)),
        "feature_margin_norm_mean": mean([float(x) for x in feature_norm_margins]),
        "candidates_by_label": {row["target_label"]: row for row in candidate_rows},
        "boundary": "counterfactual closure tests explicit color switching on neutral objects; it is not yet full natural-context color semantics.",
    }
    write_json(out_dir / f"phase942_{args.model}_summary.json", payload)
    write_jsonl(out_dir / f"phase942_{args.model}_dataset.jsonl", samples)
    write_jsonl(out_dir / f"phase942_{args.model}_sample_rows.jsonl", sample_rows)
    write_jsonl(out_dir / f"phase942_{args.model}_candidate_sample_rows.jsonl", candidate_sample_rows[: int(args.keep_candidate_sample_rows)])
    write_jsonl(out_dir / f"phase942_{args.model}_candidate_rows.jsonl", candidate_rows)
    write_jsonl(out_dir / f"phase942_{args.model}_intervention_rows.jsonl", intervention_rows)
    print(
        json.dumps(
            {
                "phase": PHASE,
                "model": args.model,
                "status": "complete",
                "sample_rows": len(sample_rows),
                "feature_top1_norm": payload["feature_top1_norm"],
                "feature_top1_norm_rate": payload["feature_top1_norm_rate"],
                "candidate_grades": {
                    row["target_label"]: row["counterfactual_grade"]
                    for row in candidate_rows
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, default="qwen3")
    parser.add_argument("--round-name", default="color_counterfactual_closure")
    parser.add_argument("--colors", default="")
    parser.add_argument("--objects", type=int, default=6)
    parser.add_argument("--templates-per-object", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--intervention-top-channels", type=int, default=4)
    parser.add_argument("--run-interventions", action="store_true")
    parser.add_argument("--intervention-factors", default="0.0,2.0")
    parser.add_argument("--max-intervention-labels", type=int, default=11)
    parser.add_argument("--max-intervention-samples-per-label", type=int, default=3)
    parser.add_argument("--keep-candidate-sample-rows", type=int, default=50000)
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-every", type=int, default=10)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    scan_model(args)


if __name__ == "__main__":
    main()
