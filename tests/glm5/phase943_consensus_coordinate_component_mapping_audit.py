#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from collections import Counter
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
from model_utils import get_layers  # noqa: E402


PHASE = 943
MODELS = ["qwen3", "glm4", "deepseek7b"]
RESULT_ROOT = Path("tests/result/phase943_consensus_coordinate_component_mapping_audit")


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


def lm_head_weight(model) -> torch.Tensor:
    if hasattr(model, "lm_head") and hasattr(model.lm_head, "weight"):
        return model.lm_head.weight.detach().float().cpu()
    output_embeddings = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
    if output_embeddings is not None and hasattr(output_embeddings, "weight"):
        return output_embeddings.weight.detach().float().cpu()
    if hasattr(model, "embed_out") and hasattr(model.embed_out, "weight"):
        return model.embed_out.weight.detach().float().cpu()
    raise ValueError("cannot find lm head weight")


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


def cosine(a: torch.Tensor | None, b: torch.Tensor | None) -> float | None:
    if a is None or b is None:
        return None
    x = a.float().cpu()
    y = b.float().cpu()
    if x.numel() != y.numel():
        return None
    nx = float(torch.linalg.vector_norm(x).item())
    ny = float(torch.linalg.vector_norm(y).item())
    if nx <= 0 or ny <= 0:
        return None
    return float(torch.dot(x, y).item() / (nx * ny))


def mean_weight_rows(weight: torch.Tensor, token_ids: list[int]) -> torch.Tensor | None:
    valid = clean_indices(token_ids, int(weight.shape[0]))
    if not valid:
        return None
    idx = torch.tensor(valid, dtype=torch.long)
    return weight[idx].float().mean(dim=0).cpu()


def safe_token_list(token_map: dict[str, list[int]], labels: list[str], skip_label: str | None = None) -> list[int]:
    out: list[int] = []
    for label in labels:
        if skip_label is not None and label == skip_label:
            continue
        out.extend(int(x) for x in token_map.get(label, []) if isinstance(x, int))
    return sorted(set(out))


def readout_rows(
    model_name: str,
    relation: str,
    language_pair: str,
    indices: list[int],
    consensus_direction: torch.Tensor | None,
    tokenizer,
    lm_weight: torch.Tensor,
    labels: list[str],
    label_tokens: dict[str, list[int]],
    boundary_tokens: list[int],
) -> list[dict[str, Any]]:
    out = []
    d_model = int(lm_weight.shape[1])
    expected = len(clean_indices(indices, d_model)) / max(1, d_model)
    boundary_vec = mean_weight_rows(lm_weight, boundary_tokens)
    for label in labels:
        label_vec = mean_weight_rows(lm_weight, label_tokens.get(label) or [])
        other_vec = mean_weight_rows(lm_weight, safe_token_list(label_tokens, labels, skip_label=label))
        if label_vec is None:
            continue
        boundary_delta = None if boundary_vec is None else label_vec - boundary_vec
        relation_delta = None if other_vec is None else label_vec - other_vec
        boundary_fraction = vector_fraction(boundary_delta, indices)
        relation_fraction = vector_fraction(relation_delta, indices)
        out.append(
            {
                "phase": PHASE,
                "row_kind": "phase943_readout_mapping_row",
                "model": model_name,
                "relation": relation,
                "language_pair": language_pair,
                "target_label": label,
                "expected_coordinate_fraction": expected,
                "label_tokens": label_tokens.get(label) or [],
                "label_token_text": [
                    p940.p901.decode_token(tokenizer, int(token_id)) for token_id in (label_tokens.get(label) or [])[:4]
                ],
                "boundary_token_count": len(boundary_tokens),
                "readout_boundary_fraction": boundary_fraction,
                "readout_boundary_lift": None
                if boundary_fraction is None or expected <= 0
                else float(boundary_fraction / expected),
                "readout_relation_fraction": relation_fraction,
                "readout_relation_lift": None
                if relation_fraction is None or expected <= 0
                else float(relation_fraction / expected),
                "consensus_direction_cos_readout_boundary": cosine(consensus_direction, boundary_delta),
                "consensus_direction_cos_readout_relation": cosine(consensus_direction, relation_delta),
            }
        )
    return out


def nested_weight(obj: Any, paths: list[tuple[str, ...]]) -> torch.Tensor | None:
    for path in paths:
        cur = obj
        ok = True
        for name in path:
            if not hasattr(cur, name):
                ok = False
                break
            cur = getattr(cur, name)
        if ok and hasattr(cur, "weight"):
            return cur.weight.detach().float().cpu()
    return None


def mlp_down_weight(layer: Any) -> torch.Tensor | None:
    return nested_weight(
        layer,
        [
            ("mlp", "down_proj"),
            ("mlp", "dense_4h_to_h"),
            ("mlp", "fc2"),
            ("mlp", "c_proj"),
            ("feed_forward", "down_proj"),
            ("feed_forward", "w2"),
        ],
    )


def attn_o_weight(layer: Any) -> torch.Tensor | None:
    return nested_weight(
        layer,
        [
            ("self_attn", "o_proj"),
            ("self_attn", "dense"),
            ("self_attention", "o_proj"),
            ("self_attention", "dense"),
            ("attention", "o_proj"),
            ("attention", "dense"),
        ],
    )


def matrix_row_fraction(weight: torch.Tensor | None, indices: list[int]) -> float | None:
    if weight is None or weight.ndim != 2:
        return None
    valid = clean_indices(indices, int(weight.shape[0]))
    if not valid:
        return None
    total = float(torch.sum(weight * weight).item())
    if total <= 0:
        return None
    idx = torch.tensor(valid, dtype=torch.long)
    part = float(torch.sum(weight[idx, :] * weight[idx, :]).item())
    return float(part / total)


def top_column_candidates(weight: torch.Tensor | None, indices: list[int], max_items: int) -> list[dict[str, Any]]:
    if weight is None or weight.ndim != 2:
        return []
    valid = clean_indices(indices, int(weight.shape[0]))
    if not valid:
        return []
    idx = torch.tensor(valid, dtype=torch.long)
    col_total = torch.sum(weight * weight, dim=0)
    col_part = torch.sum(weight[idx, :] * weight[idx, :], dim=0)
    ratio = torch.where(col_total > 0, col_part / col_total.clamp_min(1e-30), torch.zeros_like(col_total))
    expected = len(valid) / max(1, int(weight.shape[0]))
    lift = ratio / max(expected, 1e-30)
    use_k = max(1, min(int(max_items), int(weight.shape[1])))
    top = torch.topk(lift, use_k).indices.tolist()
    out = []
    for col in top:
        out.append(
            {
                "column_id": int(col),
                "coordinate_energy_fraction": float(ratio[int(col)].item()),
                "coordinate_energy_lift": float(lift[int(col)].item()),
                "column_energy": float(col_total[int(col)].item()),
            }
        )
    return out


def attention_head_candidates(
    weight: torch.Tensor | None,
    indices: list[int],
    num_heads: int | None,
    max_items: int,
) -> list[dict[str, Any]]:
    if weight is None or weight.ndim != 2 or not num_heads:
        return []
    valid = clean_indices(indices, int(weight.shape[0]))
    if not valid:
        return []
    heads = int(num_heads)
    if heads <= 0 or int(weight.shape[1]) % heads != 0:
        return []
    head_dim = int(weight.shape[1]) // heads
    idx = torch.tensor(valid, dtype=torch.long)
    expected = len(valid) / max(1, int(weight.shape[0]))
    out = []
    for head in range(heads):
        cols = slice(head * head_dim, (head + 1) * head_dim)
        block = weight[:, cols]
        total = float(torch.sum(block * block).item())
        part = float(torch.sum(block[idx, :] * block[idx, :]).item()) if total > 0 else 0.0
        frac = None if total <= 0 else float(part / total)
        out.append(
            {
                "head_id": int(head),
                "head_dim": int(head_dim),
                "coordinate_energy_fraction": frac,
                "coordinate_energy_lift": None if frac is None or expected <= 0 else float(frac / expected),
                "head_energy": total,
            }
        )
    out.sort(key=lambda row: finite(row.get("coordinate_energy_lift"), -999.0), reverse=True)
    return out[: max(1, int(max_items))]


def component_mapping(
    model,
    relation: str,
    language_pair: str,
    hidden_idx: int,
    indices: list[int],
    max_items: int,
) -> dict[str, Any]:
    layers = get_layers(model)
    layer_idx = int(hidden_idx) - 1
    if layer_idx < 0 or layer_idx >= len(layers):
        return {
            "relation": relation,
            "language_pair": language_pair,
            "hidden_idx": int(hidden_idx),
            "layer_idx": layer_idx,
            "component_available": False,
        }
    layer = layers[layer_idx]
    down = mlp_down_weight(layer)
    attn = attn_o_weight(layer)
    d_model = int(down.shape[0]) if down is not None and down.ndim == 2 else (int(attn.shape[0]) if attn is not None else 0)
    expected = len(clean_indices(indices, d_model)) / max(1, d_model) if d_model else None
    mlp_fraction = matrix_row_fraction(down, indices)
    attn_fraction = matrix_row_fraction(attn, indices)
    num_heads = getattr(getattr(model, "config", None), "num_attention_heads", None)
    if num_heads is None:
        num_heads = getattr(getattr(model, "config", None), "n_head", None)
    return {
        "relation": relation,
        "language_pair": language_pair,
        "hidden_idx": int(hidden_idx),
        "layer_idx": layer_idx,
        "component_available": bool(down is not None or attn is not None),
        "d_model": d_model,
        "expected_coordinate_fraction": expected,
        "mlp_down_shape": None if down is None else list(down.shape),
        "mlp_down_row_energy_fraction": mlp_fraction,
        "mlp_down_row_energy_lift": None
        if mlp_fraction is None or not expected or expected <= 0
        else float(mlp_fraction / expected),
        "attention_o_shape": None if attn is None else list(attn.shape),
        "attention_o_row_energy_fraction": attn_fraction,
        "attention_o_row_energy_lift": None
        if attn_fraction is None or not expected or expected <= 0
        else float(attn_fraction / expected),
        "top_mlp_columns": top_column_candidates(down, indices, int(max_items)),
        "top_attention_heads": attention_head_candidates(attn, indices, None if num_heads is None else int(num_heads), int(max_items)),
    }


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "records": len(records),
        "readout_boundary_lift_mean": mean([row.get("readout_boundary_lift_mean") for row in records]),
        "readout_relation_lift_mean": mean([row.get("readout_relation_lift_mean") for row in records]),
        "mlp_down_lift_mean": mean([row.get("component_mapping", {}).get("mlp_down_row_energy_lift") for row in records]),
        "attention_o_lift_mean": mean([row.get("component_mapping", {}).get("attention_o_row_energy_lift") for row in records]),
        "max_mlp_column_lift_mean": mean([row.get("max_mlp_column_lift") for row in records]),
        "max_attention_head_lift_mean": mean([row.get("max_attention_head_lift") for row in records]),
    }


def evidence_label(records: list[dict[str, Any]]) -> str:
    readout_positive = [
        row for row in records if finite(row.get("readout_boundary_lift_mean"), 0.0) > 1.10
    ]
    component_positive = [
        row
        for row in records
        if max(
            finite(row.get("component_mapping", {}).get("mlp_down_row_energy_lift"), 0.0),
            finite(row.get("component_mapping", {}).get("attention_o_row_energy_lift"), 0.0),
            finite(row.get("max_mlp_column_lift"), 0.0),
            finite(row.get("max_attention_head_lift"), 0.0),
        )
        > 1.25
    ]
    if readout_positive and component_positive:
        return "residual_consensus_export_with_component_candidates"
    if readout_positive:
        return "residual_consensus_export_with_readout_candidates"
    if component_positive:
        return "residual_consensus_export_with_weak_component_candidates"
    return "residual_consensus_export_no_strong_static_mapping"


def build_consensus_records(args: argparse.Namespace, model, tokenizer, device: torch.device) -> tuple[list[dict[str, Any]], dict[str, Any]]:
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
    label_sets = {relation: p938.relation_labels(samples, relation) for relation in hidden_by_relation}
    label_tokens = {relation: p938.label_token_map(tokenizer, labels) for relation, labels in label_sets.items()}
    boundary_groups = p940.boundary_token_groups(tokenizer)
    boundary_tokens = boundary_groups.get("all_boundary") or []
    lm_weight = lm_head_weight(model)
    records: list[dict[str, Any]] = []
    for key in sorted(grouped_specs):
        specs = grouped_specs[key]
        train_specs, holdout_specs = p942.split_specs(
            specs,
            float(args.train_fraction),
            int(args.min_train_specs),
            int(args.min_holdout_specs),
        )
        if not train_specs or not holdout_specs:
            continue
        relation, language_pair = key
        hidden_idx = int(hidden_by_relation[relation])
        for top_k in parse_ints(args.topks):
            for consensus_k in parse_ints(args.consensus_ks):
                indices, meta = p942.consensus_indices(train_specs, int(top_k), int(consensus_k))
                valid = clean_indices(indices, int(lm_weight.shape[1]))
                train_dirs = [spec.get("specific_direction") for spec in train_specs if spec.get("specific_direction") is not None]
                mean_dir = torch.stack([direction.float().cpu() for direction in train_dirs]).mean(dim=0) if train_dirs else None
                masked_mean_dir = p942.masked_direction(mean_dir, valid) if mean_dir is not None else None
                readouts = readout_rows(
                    args.model,
                    relation,
                    language_pair,
                    valid,
                    masked_mean_dir,
                    tokenizer,
                    lm_weight,
                    label_sets.get(relation) or [],
                    label_tokens.get(relation) or {},
                    boundary_tokens,
                )
                comp = component_mapping(model, relation, language_pair, hidden_idx, valid, int(args.max_component_items))
                top_mlp = comp.get("top_mlp_columns") or []
                top_heads = comp.get("top_attention_heads") or []
                record = {
                    "phase": PHASE,
                    "row_kind": "phase943_consensus_coordinate_artifact",
                    "model": args.model,
                    "relation": relation,
                    "language_pair": language_pair,
                    "hidden_idx": hidden_idx,
                    "top_k_for_votes": int(top_k),
                    "consensus_k": int(consensus_k),
                    "train_spec_count": len(train_specs),
                    "holdout_spec_count": len(holdout_specs),
                    "consensus_indices": valid,
                    "consensus_meta": {k: v for k, v in meta.items() if k != "indices"},
                    "readout_rows": readouts,
                    "readout_boundary_lift_mean": mean([row.get("readout_boundary_lift") for row in readouts]),
                    "readout_relation_lift_mean": mean([row.get("readout_relation_lift") for row in readouts]),
                    "readout_boundary_cos_mean": mean([row.get("consensus_direction_cos_readout_boundary") for row in readouts]),
                    "readout_relation_cos_mean": mean([row.get("consensus_direction_cos_readout_relation") for row in readouts]),
                    "component_mapping": comp,
                    "max_mlp_column_lift": None
                    if not top_mlp
                    else max(finite(row.get("coordinate_energy_lift"), 0.0) for row in top_mlp),
                    "max_attention_head_lift": None
                    if not top_heads
                    else max(finite(row.get("coordinate_energy_lift"), 0.0) for row in top_heads),
                }
                records.append(record)
                log(
                    f"{args.model}/{args.round_name}: {relation}:{language_pair} top_k={top_k} consensus_k={consensus_k} indices={len(valid)}"
                )
    meta = {
        "sample_count": len(samples),
        "selected_relation_language_pairs": sorted([f"{rel}:{pair}" for rel, pair in selected_pairs]),
        "selected_phase940_bridge_rows": selected_phase940_rows,
        "hidden_by_relation": hidden_by_relation,
        "direction_specs_all": len(direction_specs),
        "candidate_pair_count": len(grouped_specs),
    }
    return records, meta


def eval_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        selected_pairs, selected_rows = p941.selected_positive_pairs(args)
        payload = {
            "phase": PHASE,
            "model": args.model,
            "status": "dry_run",
            "selected_relation_language_pairs": sorted([f"{rel}:{pair}" for rel, pair in selected_pairs]),
            "selected_phase940_bridge_rows": selected_rows,
        }
        write_json(out_dir / f"phase943_{args.model}_summary.json", payload)
        write_jsonl(out_dir / f"phase943_{args.model}_consensus_records.jsonl", [])
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return payload

    model = None
    tokenizer = None
    attn_impl = None
    try:
        model, tokenizer, device, attn_impl = p938.p862.p844.p828.p796.load_model_bf16_prefer_flash(
            args.model, args.attn_implementations
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        records, meta = build_consensus_records(args, model, tokenizer, device)
    finally:
        if model is not None:
            p938.p862.p844.p828.release_model(model)
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    payload = {
        "phase": PHASE,
        "title": "Consensus Coordinate Artifact Export and Component Mapping Audit",
        "model": args.model,
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        **meta,
        "records": len(records),
        "summary": summarize_records(records),
        "evidence_label": evidence_label(records),
        "boundary": "static readout/component weight mapping only; not component causal closure",
    }
    write_json(out_dir / f"phase943_{args.model}_summary.json", payload)
    write_jsonl(out_dir / f"phase943_{args.model}_consensus_records.jsonl", records)
    print(
        json.dumps(
            {
                "phase": PHASE,
                "model": args.model,
                "status": "complete",
                "evidence": payload["evidence_label"],
                "records": len(records),
                "summary": payload["summary"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return payload


def load_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    summaries = [read_json(out_dir / f"phase943_{model}_summary.json") for model in MODELS]
    summaries = [summary for summary in summaries if summary]
    evidence_counts = Counter(str(summary.get("evidence_label") or summary.get("status")) for summary in summaries)
    records = []
    for model_name in MODELS:
        records.extend(load_records(out_dir / f"phase943_{model_name}_consensus_records.jsonl"))
    records.sort(
        key=lambda row: (
            finite(row.get("readout_boundary_lift_mean"), -999.0),
            max(
                finite(row.get("component_mapping", {}).get("mlp_down_row_energy_lift"), -999.0),
                finite(row.get("component_mapping", {}).get("attention_o_row_energy_lift"), -999.0),
            ),
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
        "top_records": records[:80],
    }
    write_json(out_dir / "phase943_cross_model_summary.json", payload)
    write_summary_md(out_dir / "phase943_cross_model_summary.md", payload)
    return payload


def write_summary_md(path: Path, payload: dict[str, Any]) -> None:
    lines = ["# Phase 943 consensus coordinate component mapping audit", ""]
    lines += ["## Evidence", ""]
    for key, value in sorted((payload.get("evidence_counts") or {}).items()):
        lines.append(f"- {key}: {value}")
    lines += ["", "## Model Summary", ""]
    lines.append("| model | evidence | records | readout boundary lift | readout relation lift | mlp lift | attn lift | max mlp col lift | max attn head lift |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for summary in payload.get("model_summaries") or []:
        row = summary.get("summary") or {}
        lines.append(
            "| {model} | {evidence} | {records} | {rbl} | {rrl} | {mlp} | {attn} | {mcol} | {ahead} |".format(
                model=summary.get("model"),
                evidence=summary.get("evidence_label"),
                records=summary.get("records"),
                rbl=row.get("readout_boundary_lift_mean"),
                rrl=row.get("readout_relation_lift_mean"),
                mlp=row.get("mlp_down_lift_mean"),
                attn=row.get("attention_o_lift_mean"),
                mcol=row.get("max_mlp_column_lift_mean"),
                ahead=row.get("max_attention_head_lift_mean"),
            )
        )
    lines += ["", "## Top Records", ""]
    lines.append("| model | relation | pair | hidden | readout boundary lift | readout relation lift | mlp lift | attn lift | max mlp col | max attn head |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in payload.get("top_records") or []:
        comp = row.get("component_mapping") or {}
        lines.append(
            "| {model} | {relation} | {pair} | {hidden} | {rbl} | {rrl} | {mlp} | {attn} | {mcol} | {ahead} |".format(
                model=row.get("model"),
                relation=row.get("relation"),
                pair=row.get("language_pair"),
                hidden=row.get("hidden_idx"),
                rbl=row.get("readout_boundary_lift_mean"),
                rrl=row.get("readout_relation_lift_mean"),
                mlp=comp.get("mlp_down_row_energy_lift"),
                attn=comp.get("attention_o_row_energy_lift"),
                mcol=row.get("max_mlp_column_lift"),
                ahead=row.get("max_attention_head_lift"),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="consensus_coordinate_component_mapping_audit")
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
    parser.add_argument("--max-component-items", type=int, default=12)
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
