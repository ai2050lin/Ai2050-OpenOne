#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
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
import phase942_semantic_boundary_coordinate_consensus_holdout as p942  # noqa: E402
import phase943_consensus_coordinate_component_mapping_audit as p943  # noqa: E402
from model_utils import get_layers  # noqa: E402


PHASE = 944
MODELS = ["qwen3", "glm4", "deepseek7b"]
RESULT_ROOT = Path("tests/result/phase944_activation_weighted_mlp_channel_causal_audit")
PHASE943_ROOT = Path("tests/result/phase943_consensus_coordinate_component_mapping_audit")


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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


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


def stable_seed(*parts: Any) -> int:
    text = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False) % (2**31 - 1)


def clean_indices(indices: list[int], dim: int) -> list[int]:
    return sorted({int(idx) for idx in indices if 0 <= int(idx) < int(dim)})


def vector_fraction(vec: torch.Tensor | None, indices: list[int]) -> float | None:
    if vec is None:
        return None
    work = vec.float().cpu()
    total = float(torch.sum(work * work).item())
    if total <= 0:
        return None
    valid = clean_indices(indices, int(work.numel()))
    if not valid:
        return None
    idx = torch.tensor(valid, dtype=torch.long)
    part = float(torch.sum(work[idx] * work[idx]).item())
    return float(part / total)


def down_proj_for_layer(model, layer_idx: int):
    layers = get_layers(model)
    if int(layer_idx) < 0 or int(layer_idx) >= len(layers):
        return None
    mlp = getattr(layers[int(layer_idx)], "mlp", None)
    return getattr(mlp, "down_proj", None)


def select_random_channels(width: int, excluded: list[int], count: int, seed_parts: tuple[Any, ...]) -> list[int]:
    excluded_set = {int(x) for x in excluded}
    pool = [idx for idx in range(int(width)) if idx not in excluded_set]
    if not pool:
        return []
    rnd = random.Random(stable_seed(*seed_parts))
    rnd.shuffle(pool)
    return sorted(pool[: max(1, min(int(count), len(pool)))])


def load_phase943_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    path = PHASE943_ROOT / args.phase943_round / f"phase943_{args.model}_consensus_records.jsonl"
    records = read_jsonl(path)
    out = []
    for row in records:
        top_mlp = ((row.get("component_mapping") or {}).get("top_mlp_columns") or [])[: int(args.top_channels)]
        if not top_mlp:
            continue
        copied = dict(row)
        copied["candidate_channel_ids"] = [int(item["column_id"]) for item in top_mlp]
        copied["candidate_channel_static_lifts"] = [finite(item.get("coordinate_energy_lift"), 0.0) for item in top_mlp]
        copied["candidate_channel_static_energy"] = [finite(item.get("column_energy"), 0.0) for item in top_mlp]
        out.append(copied)
    return out


def unique_samples(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    out = []
    for sample in samples:
        key = p938.sample_key(sample)
        if key in seen:
            continue
        seen.add(key)
        out.append(sample)
    return out


def build_holdout_samples(args: argparse.Namespace, model, tokenizer, device: torch.device) -> tuple[dict[tuple[str, str], list[dict[str, Any]]], dict[str, Any]]:
    samples = p939.build_samples(args)
    selected_pairs, selected_phase940_rows = p941.selected_positive_pairs(args)
    hidden_by_relation = p938.phase937_best_hidden(args.model, args.phase937_round)
    if not hidden_by_relation:
        hidden_by_relation = {relation: -1 for relation in parse_csv(args.relations)}
    auto_indices = p940.p937.auto_hidden_indices(model)
    hidden_by_relation = {
        rel: (auto_indices[len(auto_indices) // 2] if int(idx) < 0 else int(idx)) for rel, idx in hidden_by_relation.items()
    }
    hidden_indices = sorted(set(hidden_by_relation.values()))
    vectors = p938.forward_vectors(model, tokenizer, device, samples, hidden_indices, int(args.batch_size))
    direction_specs = p939.build_direction_specs(samples, vectors, hidden_by_relation, int(args.min_train_per_label))
    grouped_specs = p942.select_candidate_specs(direction_specs, selected_pairs, int(args.max_specs_per_pair))
    out: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for key, specs in grouped_specs.items():
        train_specs, holdout_specs = p942.split_specs(
            specs,
            float(args.train_fraction),
            int(args.min_train_specs),
            int(args.min_holdout_specs),
        )
        if not train_specs or not holdout_specs:
            continue
        holdout = []
        for spec in holdout_specs:
            holdout.extend(spec.get("test_samples") or [])
        out[key] = unique_samples(holdout)
    meta = {
        "sample_count": len(samples),
        "selected_relation_language_pairs": sorted([f"{rel}:{pair}" for rel, pair in selected_pairs]),
        "selected_phase940_bridge_rows": selected_phase940_rows,
        "hidden_by_relation": hidden_by_relation,
        "direction_specs_all": len(direction_specs),
        "candidate_pair_count": len(grouped_specs),
    }
    return out, meta


def capture_down_inputs(
    model,
    tokenizer,
    device: torch.device,
    samples: list[dict[str, Any]],
    layer_idx: int,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    down_proj = down_proj_for_layer(model, int(layer_idx))
    if down_proj is None:
        return {}
    out: dict[str, torch.Tensor] = {}
    for start in range(0, len(samples), max(1, int(batch_size))):
        batch = samples[start : start + max(1, int(batch_size))]
        input_ids, attention_mask, last_pos = p938.encode_batch(tokenizer, device, batch)
        captured: dict[str, torch.Tensor] = {}

        def hook(_module, inputs):
            if not inputs or not torch.is_tensor(inputs[0]):
                return None
            hidden = inputs[0]
            pos = last_pos.to(device=hidden.device)
            idx = torch.arange(hidden.shape[0], device=hidden.device)
            captured["activation"] = hidden[idx, pos, :].detach().float().cpu()
            return None

        handle = down_proj.register_forward_pre_hook(hook)
        try:
            with torch.inference_mode():
                model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)
        finally:
            handle.remove()
        acts = captured.get("activation")
        if acts is not None:
            for row_idx, sample in enumerate(batch):
                out[p938.sample_key(sample)] = acts[row_idx].clone()
        del input_ids, attention_mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return out


def patched_logits_with_channel_scale(
    model,
    tokenizer,
    device: torch.device,
    samples: list[dict[str, Any]],
    layer_idx: int,
    channel_ids: list[int],
    factor: float,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    down_proj = down_proj_for_layer(model, int(layer_idx))
    if down_proj is None or not channel_ids:
        return {}
    out: dict[str, torch.Tensor] = {}
    channel_tensor = torch.tensor(sorted(set(int(x) for x in channel_ids)), dtype=torch.long)
    for start in range(0, len(samples), max(1, int(batch_size))):
        batch = samples[start : start + max(1, int(batch_size))]
        input_ids, attention_mask, last_pos = p938.encode_batch(tokenizer, device, batch)
        batch_idx = torch.arange(input_ids.shape[0], device=device)

        def hook(_module, inputs):
            if not inputs or not torch.is_tensor(inputs[0]):
                return None
            hidden = inputs[0]
            local_channels = channel_tensor.to(device=hidden.device)
            if int(local_channels.max().item()) >= int(hidden.shape[-1]):
                return None
            patched = hidden.clone()
            pos = last_pos.to(device=patched.device)
            idx = torch.arange(patched.shape[0], device=patched.device)
            patched[idx[:, None], pos[:, None], local_channels[None, :]] *= float(factor)
            return (patched, *inputs[1:])

        handle = down_proj.register_forward_pre_hook(hook)
        try:
            with torch.inference_mode():
                result = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)
            logits = result.logits[batch_idx, last_pos].detach().float().cpu()
            for row_idx, sample in enumerate(batch):
                out[p938.sample_key(sample)] = logits[row_idx]
            del result, logits
        finally:
            handle.remove()
        del input_ids, attention_mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return out


def contribution_metrics(
    activation_by_sample: dict[str, torch.Tensor],
    samples: list[dict[str, Any]],
    down_weight: torch.Tensor,
    channel_ids: list[int],
    consensus_indices: list[int],
) -> dict[str, Any]:
    valid_channels = p943.clean_indices(channel_ids, int(down_weight.shape[1]))
    if not valid_channels:
        return {"channel_count": 0}
    d_model = int(down_weight.shape[0])
    expected = len(p943.clean_indices(consensus_indices, d_model)) / max(1, d_model)
    rows = []
    for sample in samples:
        act = activation_by_sample.get(p938.sample_key(sample))
        if act is None:
            continue
        channels = torch.tensor(valid_channels, dtype=torch.long)
        selected_act = act[channels].float().cpu()
        cols = down_weight[:, channels].float().cpu()
        contribution = torch.matmul(cols, selected_act)
        fraction = vector_fraction(contribution, consensus_indices)
        lift = None if fraction is None or expected <= 0 else float(fraction / expected)
        rows.append(
            {
                "activation_mean_abs": float(torch.mean(torch.abs(selected_act)).item()),
                "activation_rms": float(torch.sqrt(torch.mean(selected_act * selected_act)).item()),
                "contribution_norm": float(torch.linalg.vector_norm(contribution).item()),
                "contribution_consensus_fraction": fraction,
                "contribution_consensus_lift": lift,
            }
        )
    return {
        "channel_count": len(valid_channels),
        "activation_mean_abs_mean": mean([row["activation_mean_abs"] for row in rows]),
        "activation_rms_mean": mean([row["activation_rms"] for row in rows]),
        "contribution_norm_mean": mean([row["contribution_norm"] for row in rows]),
        "contribution_consensus_fraction_mean": mean([row["contribution_consensus_fraction"] for row in rows]),
        "contribution_consensus_lift_mean": mean([row["contribution_consensus_lift"] for row in rows]),
        "sample_count": len(rows),
    }


def make_delta_row(
    model_name: str,
    sample: dict[str, Any],
    record: dict[str, Any],
    condition: str,
    factor: float | None,
    channel_group: str,
    channel_ids: list[int],
    base: dict[str, Any],
    patched: dict[str, Any],
) -> dict[str, Any]:
    spec = {
        "relation": sample.get("relation"),
        "target_label": sample.get("target_label"),
        "hidden_idx": record.get("hidden_idx"),
        "train_language": str(record.get("language_pair", "unknown->unknown")).split("->")[0],
        "test_language": sample.get("prompt_language"),
        "language_pair": record.get("language_pair"),
        "train_template": "phase944_consensus_train_group",
        "test_template": sample.get("prompt_template"),
    }
    row = p940.make_row(model_name, sample, spec, condition, factor, base, patched)
    row["phase"] = PHASE
    row["row_kind"] = "phase944_activation_weighted_mlp_channel_causal_row"
    row["record_key"] = record_key(record)
    row["layer_idx"] = int((record.get("component_mapping") or {}).get("layer_idx", int(record.get("hidden_idx", 1)) - 1))
    row["factor"] = factor
    row["channel_group"] = channel_group
    row["channel_count"] = len(channel_ids)
    row["channel_ids"] = [int(x) for x in channel_ids]
    return row


def record_key(record: dict[str, Any]) -> str:
    return f"{record.get('model')}|{record.get('relation')}|{record.get('language_pair')}|h{record.get('hidden_idx')}"


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "relation_margin_delta_mean": mean([row.get("target_margin_vs_relation_best_other_delta") for row in rows]),
        "boundary_margin_delta_mean": mean([row.get("target_vs_all_boundary_margin_delta") for row in rows]),
        "period_margin_delta_mean": mean([row.get("target_vs_punctuation_period_margin_delta") for row in rows]),
        "eos_margin_delta_mean": mean([row.get("target_vs_eos_margin_delta") for row in rows]),
        "target_logit_delta_mean": mean([row.get("target_label_logit_delta") for row in rows]),
        "boundary_logit_delta_mean": mean([row.get("all_boundary_logit_delta") for row in rows]),
        "rank_improved": sum(1 for row in rows if row.get("rank_improved")),
        "new_relation_winner_target": sum(1 for row in rows if row.get("new_relation_winner_target")),
    }


def summarize_by(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[tuple(row.get(key) for key in keys)].append(row)
    out = []
    for key_tuple, items in buckets.items():
        row = {key: value for key, value in zip(keys, key_tuple)}
        row.update(summarize_rows(items))
        out.append(row)
    out.sort(
        key=lambda row: (
            str(row.get("model")),
            str(row.get("relation")),
            str(row.get("language_pair")),
            str(row.get("condition")),
        )
    )
    return out


def causal_eval_records(records: list[dict[str, Any]], condition_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in condition_rows:
        by_key[(str(row.get("record_key")), str(row.get("condition")))] = row
    out = []
    for record in records:
        key = record_key(record)
        cand_ablate = by_key.get((key, "candidate_ablate"))
        cand_boost = by_key.get((key, "candidate_boost"))
        ctrl_ablate = by_key.get((key, "random_ablate"))
        ctrl_boost = by_key.get((key, "random_boost"))
        if not cand_ablate or not cand_boost or not ctrl_ablate or not ctrl_boost:
            continue
        cand_slope = finite(cand_boost.get("boundary_margin_delta_mean")) - finite(cand_ablate.get("boundary_margin_delta_mean"))
        ctrl_slope = finite(ctrl_boost.get("boundary_margin_delta_mean")) - finite(ctrl_ablate.get("boundary_margin_delta_mean"))
        cand_relation_slope = finite(cand_boost.get("relation_margin_delta_mean")) - finite(cand_ablate.get("relation_margin_delta_mean"))
        ctrl_relation_slope = finite(ctrl_boost.get("relation_margin_delta_mean")) - finite(ctrl_ablate.get("relation_margin_delta_mean"))
        act_gap = finite(record.get("candidate_contribution", {}).get("contribution_consensus_lift_mean")) - finite(
            record.get("random_contribution", {}).get("contribution_consensus_lift_mean")
        )
        out.append(
            {
                "record_key": key,
                "model": record.get("model"),
                "relation": record.get("relation"),
                "language_pair": record.get("language_pair"),
                "hidden_idx": record.get("hidden_idx"),
                "layer_idx": (record.get("component_mapping") or {}).get("layer_idx"),
                "candidate_channels": record.get("candidate_channel_ids"),
                "random_channels": record.get("random_channel_ids"),
                "candidate_contribution_lift": (record.get("candidate_contribution") or {}).get(
                    "contribution_consensus_lift_mean"
                ),
                "random_contribution_lift": (record.get("random_contribution") or {}).get(
                    "contribution_consensus_lift_mean"
                ),
                "activation_weighted_lift_gap": act_gap,
                "candidate_boundary_slope": cand_slope,
                "random_boundary_slope": ctrl_slope,
                "boundary_slope_gain_vs_random": float(cand_slope - ctrl_slope),
                "candidate_relation_slope": cand_relation_slope,
                "random_relation_slope": ctrl_relation_slope,
                "relation_slope_gain_vs_random": float(cand_relation_slope - ctrl_relation_slope),
                "candidate_ablate_boundary_delta": cand_ablate.get("boundary_margin_delta_mean"),
                "candidate_boost_boundary_delta": cand_boost.get("boundary_margin_delta_mean"),
                "random_ablate_boundary_delta": ctrl_ablate.get("boundary_margin_delta_mean"),
                "random_boost_boundary_delta": ctrl_boost.get("boundary_margin_delta_mean"),
                "rows": cand_ablate.get("rows"),
            }
        )
    out.sort(
        key=lambda row: (
            finite(row.get("boundary_slope_gain_vs_random"), -999.0),
            finite(row.get("activation_weighted_lift_gap"), -999.0),
        ),
        reverse=True,
    )
    return out


def evidence_label(eval_rows: list[dict[str, Any]]) -> str:
    positives = [
        row
        for row in eval_rows
        if finite(row.get("activation_weighted_lift_gap"), -999.0) > 0.25
        and finite(row.get("boundary_slope_gain_vs_random"), -999.0) > 0.02
    ]
    rels = {str(row.get("relation")) for row in positives}
    if len(rels) >= 2:
        return "activation_weighted_mlp_channel_causal_positive"
    if positives:
        return "partial_activation_weighted_mlp_channel_causal_positive"
    return "activation_weighted_mlp_channel_causal_weak_or_mixed"


def eval_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    phase943_records = load_phase943_records(args)
    dry_payload = {
        "phase": PHASE,
        "title": "Activation-Weighted MLP Channel Causal Audit",
        "model": args.model,
        "phase943_record_count": len(phase943_records),
        "phase943_records_selected": [
            {
                "relation": row.get("relation"),
                "language_pair": row.get("language_pair"),
                "hidden_idx": row.get("hidden_idx"),
                "candidate_channel_ids": row.get("candidate_channel_ids"),
            }
            for row in phase943_records
        ],
    }
    if args.dry_run:
        payload = {**dry_payload, "status": "dry_run"}
        write_json(out_dir / f"phase944_{args.model}_summary.json", payload)
        write_jsonl(out_dir / f"phase944_{args.model}_rows.jsonl", [])
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return payload

    rows: list[dict[str, Any]] = []
    activation_records: list[dict[str, Any]] = []
    model = None
    tokenizer = None
    attn_impl = None
    meta: dict[str, Any] = {}
    try:
        model, tokenizer, device, attn_impl = p938.p862.p844.p828.p796.load_model_bf16_prefer_flash(
            args.model, args.attn_implementations
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        holdout_by_pair, meta = build_holdout_samples(args, model, tokenizer, device)
        all_samples = unique_samples([sample for vals in holdout_by_pair.values() for sample in vals])
        labels_by_relation = {relation: p938.relation_labels(all_samples, relation) for relation in parse_csv(args.relations)}
        token_maps = {relation: p938.label_token_map(tokenizer, labels) for relation, labels in labels_by_relation.items()}
        boundary_groups = p940.boundary_token_groups(tokenizer)
        baseline_logits = p938.forward_logits(model, tokenizer, device, all_samples, int(args.batch_size))
        baseline_metrics: dict[str, dict[str, Any]] = {}
        for sample in all_samples:
            relation = str(sample["relation"])
            baseline_metrics[p938.sample_key(sample)] = p940.boundary_metrics(
                tokenizer,
                baseline_logits[p938.sample_key(sample)],
                str(sample["target_label"]),
                token_maps.get(relation) or {},
                boundary_groups,
            )

        for record in phase943_records:
            relation = str(record.get("relation"))
            language_pair = str(record.get("language_pair"))
            samples = holdout_by_pair.get((relation, language_pair)) or []
            if not samples:
                continue
            if int(args.max_samples_per_record) > 0:
                samples = samples[: int(args.max_samples_per_record)]
            layer_idx = int((record.get("component_mapping") or {}).get("layer_idx", int(record.get("hidden_idx", 1)) - 1))
            down_proj = down_proj_for_layer(model, layer_idx)
            if down_proj is None:
                continue
            down_weight = down_proj.weight.detach().float().cpu()
            candidate_channels = p943.clean_indices(record.get("candidate_channel_ids") or [], int(down_weight.shape[1]))
            random_channels = select_random_channels(
                int(down_weight.shape[1]),
                candidate_channels,
                len(candidate_channels),
                (PHASE, args.model, relation, language_pair, layer_idx),
            )
            activations = capture_down_inputs(model, tokenizer, device, samples, layer_idx, int(args.batch_size))
            candidate_contrib = contribution_metrics(
                activations,
                samples,
                down_weight,
                candidate_channels,
                record.get("consensus_indices") or [],
            )
            random_contrib = contribution_metrics(
                activations,
                samples,
                down_weight,
                random_channels,
                record.get("consensus_indices") or [],
            )
            enriched = {
                **record,
                "random_channel_ids": random_channels,
                "candidate_contribution": candidate_contrib,
                "random_contribution": random_contrib,
                "test_sample_count": len(samples),
            }
            activation_records.append(enriched)
            interventions = [
                ("candidate_ablate", "candidate", candidate_channels, 0.0),
                ("candidate_boost", "candidate", candidate_channels, float(args.boost_factor)),
                ("random_ablate", "same_layer_random", random_channels, 0.0),
                ("random_boost", "same_layer_random", random_channels, float(args.boost_factor)),
            ]
            for condition, group_kind, channel_ids, factor in interventions:
                patched_logits = patched_logits_with_channel_scale(
                    model,
                    tokenizer,
                    device,
                    samples,
                    layer_idx,
                    channel_ids,
                    float(factor),
                    int(args.batch_size),
                )
                for sample in samples:
                    key = p938.sample_key(sample)
                    if key not in patched_logits:
                        continue
                    relation_sample = str(sample["relation"])
                    base = baseline_metrics[key]
                    patched = p940.boundary_metrics(
                        tokenizer,
                        patched_logits[key],
                        str(sample["target_label"]),
                        token_maps.get(relation_sample) or {},
                        boundary_groups,
                    )
                    rows.append(
                        make_delta_row(
                            args.model,
                            sample,
                            enriched,
                            condition,
                            float(factor),
                            group_kind,
                            channel_ids,
                            base,
                            patched,
                        )
                    )
            log(
                f"{args.model}/{args.round_name}: {relation}:{language_pair} L{layer_idx} samples={len(samples)} cand={candidate_channels} rand={random_channels}"
            )
    finally:
        if model is not None:
            p938.p862.p844.p828.release_model(model)
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    condition_rows = summarize_by(rows, ["record_key", "model", "relation", "language_pair", "condition", "channel_group", "factor"])
    eval_rows = causal_eval_records(activation_records, condition_rows)
    payload = {
        **dry_payload,
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        **meta,
        "rows": len(rows),
        "activation_record_count": len(activation_records),
        "activation_records": activation_records,
        "by_condition": condition_rows,
        "causal_eval_rows": eval_rows,
        "evidence_label": evidence_label(eval_rows),
        "boundary": "MLP down-proj activation-weighted contribution and channel scaling audit; not attention-head or natural-rollout closure",
    }
    write_json(out_dir / f"phase944_{args.model}_summary.json", payload)
    write_jsonl(out_dir / f"phase944_{args.model}_rows.jsonl", rows)
    print(
        json.dumps(
            {
                "phase": PHASE,
                "model": args.model,
                "status": "complete",
                "evidence": payload["evidence_label"],
                "rows": len(rows),
                "top_eval_rows": eval_rows[:8],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return payload


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    summaries = [read_json(out_dir / f"phase944_{model}_summary.json") for model in MODELS]
    summaries = [summary for summary in summaries if summary]
    evidence_counts = Counter(str(summary.get("evidence_label") or summary.get("status")) for summary in summaries)
    eval_rows = []
    condition_rows = []
    for summary in summaries:
        for row in summary.get("causal_eval_rows") or []:
            eval_rows.append(dict(row))
        for row in summary.get("by_condition") or []:
            condition_rows.append(dict(row))
    eval_rows.sort(
        key=lambda row: (
            finite(row.get("boundary_slope_gain_vs_random"), -999.0),
            finite(row.get("activation_weighted_lift_gap"), -999.0),
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
        "top_eval_rows": eval_rows[:120],
        "condition_rows": condition_rows,
    }
    write_json(out_dir / "phase944_cross_model_summary.json", payload)
    write_summary_md(out_dir / "phase944_cross_model_summary.md", payload)
    return payload


def write_summary_md(path: Path, payload: dict[str, Any]) -> None:
    lines = ["# Phase 944 activation-weighted MLP channel causal audit", ""]
    lines += ["## Evidence", ""]
    for key, value in sorted((payload.get("evidence_counts") or {}).items()):
        lines.append(f"- {key}: {value}")
    lines += ["", "## Top Causal Eval Rows", ""]
    lines.append("| model | relation | pair | hidden | channels | act lift | random lift | act gap | cand slope | rand slope | slope gain |")
    lines.append("| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in payload.get("top_eval_rows") or []:
        lines.append(
            "| {model} | {relation} | {pair} | {hidden} | {channels} | {act} | {rand} | {gap} | {cs} | {rs} | {sg} |".format(
                model=row.get("model"),
                relation=row.get("relation"),
                pair=row.get("language_pair"),
                hidden=row.get("hidden_idx"),
                channels=",".join(str(x) for x in (row.get("candidate_channels") or [])),
                act=row.get("candidate_contribution_lift"),
                rand=row.get("random_contribution_lift"),
                gap=row.get("activation_weighted_lift_gap"),
                cs=row.get("candidate_boundary_slope"),
                rs=row.get("random_boundary_slope"),
                sg=row.get("boundary_slope_gain_vs_random"),
            )
        )
    lines += ["", "## Condition Rows", ""]
    lines.append("| model | relation | pair | condition | factor | rows | relation delta | boundary delta | target delta | boundary logit delta |")
    lines.append("| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in payload.get("condition_rows") or []:
        lines.append(
            "| {model} | {relation} | {pair} | {condition} | {factor} | {rows} | {rel} | {bd} | {td} | {bld} |".format(
                model=row.get("model"),
                relation=row.get("relation"),
                pair=row.get("language_pair"),
                condition=row.get("condition"),
                factor=row.get("factor"),
                rows=row.get("rows"),
                rel=row.get("relation_margin_delta_mean"),
                bd=row.get("boundary_margin_delta_mean"),
                td=row.get("target_logit_delta_mean"),
                bld=row.get("boundary_logit_delta_mean"),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="activation_weighted_mlp_channel_causal_audit")
    parser.add_argument("--phase937-round", default="semantic_reuse_difference_state_atlas")
    parser.add_argument("--phase939-round", default="bilingual_specificity_tightening_audit")
    parser.add_argument("--phase940-round", default="semantic_boundary_bridge_audit")
    parser.add_argument("--phase943-round", default="consensus_coordinate_component_mapping_audit")
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
    parser.add_argument("--top-channels", type=int, default=3)
    parser.add_argument("--boost-factor", type=float, default=1.5)
    parser.add_argument("--max-samples-per-record", type=int, default=0)
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
