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
import phase940_semantic_boundary_bridge_audit as p940  # noqa: E402
import phase944_activation_weighted_mlp_channel_causal_audit as p944  # noqa: E402


PHASE = 198
SOURCE_PHASE = 944
MODELS = ["qwen3", "glm4", "deepseek7b"]
RESULT_ROOT = Path("tests/result/phase198_single_channel_sign_decomposition_atlas")
PHASE944_ROOT = Path("tests/result/phase944_activation_weighted_mlp_channel_causal_audit")


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


def mean(values: list[Any]) -> float | None:
    vals = []
    for value in values:
        try:
            fval = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(fval):
            vals.append(fval)
    return None if not vals else float(sum(vals) / len(vals))


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "relation_margin_delta_mean": mean([row.get("target_margin_vs_relation_best_other_delta") for row in rows]),
        "boundary_margin_delta_mean": mean([row.get("target_vs_all_boundary_margin_delta") for row in rows]),
        "period_margin_delta_mean": mean([row.get("target_vs_punctuation_period_margin_delta") for row in rows]),
        "protocol_margin_delta_mean": mean([row.get("target_vs_protocol_boundary_margin_delta") for row in rows]),
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
    out.sort(key=lambda row: tuple(str(row.get(key)) for key in keys))
    return out


def record_key(record: dict[str, Any]) -> str:
    return f"{record.get('model')}|{record.get('relation')}|{record.get('language_pair')}|h{record.get('hidden_idx')}"


def channel_record_key(record: dict[str, Any], channel_id: int, channel_source: str) -> str:
    return f"{record_key(record)}|{channel_source}|c{int(channel_id)}"


def load_phase944_activation_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    path = PHASE944_ROOT / args.phase944_round / f"phase944_{args.model}_summary.json"
    summary = read_json(path)
    records = [dict(row) for row in summary.get("activation_records") or []]
    if args.only_positive_source:
        eval_by_record = {str(row.get("record_key")): row for row in summary.get("causal_eval_rows") or []}
        kept = []
        for record in records:
            ev = eval_by_record.get(record_key(record)) or {}
            if finite(ev.get("activation_weighted_lift_gap"), -999.0) > float(args.min_activation_gap):
                kept.append(record)
        records = kept
    if int(args.max_records) > 0:
        records = records[: int(args.max_records)]
    return records


def make_delta_row(
    model_name: str,
    sample: dict[str, Any],
    record: dict[str, Any],
    condition: str,
    factor: float,
    channel_source: str,
    channel_id: int,
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
        "train_template": "phase198_single_channel_source_phase944",
        "test_template": sample.get("prompt_template"),
    }
    row = p940.make_row(model_name, sample, spec, condition, factor, base, patched)
    row["phase"] = PHASE
    row["source_phase"] = SOURCE_PHASE
    row["row_kind"] = "phase198_single_channel_sign_decomposition_row"
    row["record_key"] = record_key(record)
    row["channel_record_key"] = channel_record_key(record, channel_id, channel_source)
    row["layer_idx"] = int((record.get("component_mapping") or {}).get("layer_idx", int(record.get("hidden_idx", 1)) - 1))
    row["factor"] = float(factor)
    row["channel_source"] = channel_source
    row["channel_id"] = int(channel_id)
    return row


def build_channel_eval(records: list[dict[str, Any]], condition_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in condition_rows:
        by_key[(str(row.get("channel_record_key")), str(row.get("condition")))] = row
    out = []
    for record in records:
        for channel_source, channel_ids in [
            ("candidate", record.get("candidate_channel_ids") or []),
            ("same_layer_random", record.get("random_channel_ids") or []),
        ]:
            for channel_id in channel_ids:
                key = channel_record_key(record, int(channel_id), channel_source)
                ablate = by_key.get((key, f"{channel_source}_single_ablate"))
                boost = by_key.get((key, f"{channel_source}_single_boost"))
                if not ablate or not boost:
                    continue
                boundary_slope = finite(boost.get("boundary_margin_delta_mean")) - finite(ablate.get("boundary_margin_delta_mean"))
                relation_slope = finite(boost.get("relation_margin_delta_mean")) - finite(ablate.get("relation_margin_delta_mean"))
                target_slope = finite(boost.get("target_logit_delta_mean")) - finite(ablate.get("target_logit_delta_mean"))
                protocol_slope = finite(boost.get("protocol_margin_delta_mean")) - finite(ablate.get("protocol_margin_delta_mean"))
                period_slope = finite(boost.get("period_margin_delta_mean")) - finite(ablate.get("period_margin_delta_mean"))
                eos_slope = finite(boost.get("eos_margin_delta_mean")) - finite(ablate.get("eos_margin_delta_mean"))
                if boundary_slope > 0.02 and relation_slope >= -0.02:
                    sign = "support_channel"
                elif boundary_slope < -0.02:
                    sign = "suppressor_or_blocker_channel"
                elif abs(boundary_slope) <= 0.02 and abs(relation_slope) <= 0.02:
                    sign = "near_zero_or_correlational"
                else:
                    sign = "mixed_side_effect_channel"
                out.append(
                    {
                        "phase": PHASE,
                        "source_phase": SOURCE_PHASE,
                        "channel_record_key": key,
                        "record_key": record_key(record),
                        "model": record.get("model"),
                        "relation": record.get("relation"),
                        "language_pair": record.get("language_pair"),
                        "hidden_idx": record.get("hidden_idx"),
                        "layer_idx": (record.get("component_mapping") or {}).get("layer_idx"),
                        "channel_source": channel_source,
                        "channel_id": int(channel_id),
                        "single_channel_sign": sign,
                        "boundary_slope": float(boundary_slope),
                        "relation_slope": float(relation_slope),
                        "target_logit_slope": float(target_slope),
                        "protocol_margin_slope": float(protocol_slope),
                        "period_margin_slope": float(period_slope),
                        "eos_margin_slope": float(eos_slope),
                        "ablate_boundary_delta": ablate.get("boundary_margin_delta_mean"),
                        "boost_boundary_delta": boost.get("boundary_margin_delta_mean"),
                        "ablate_relation_delta": ablate.get("relation_margin_delta_mean"),
                        "boost_relation_delta": boost.get("relation_margin_delta_mean"),
                        "rows": ablate.get("rows"),
                    }
                )
    out.sort(
        key=lambda row: (
            row.get("channel_source") != "candidate",
            -finite(row.get("boundary_slope"), -999.0),
            str(row.get("model")),
            str(row.get("relation")),
        )
    )
    return out


def evidence_label(channel_eval_rows: list[dict[str, Any]]) -> str:
    candidate_rows = [row for row in channel_eval_rows if row.get("channel_source") == "candidate"]
    support = [row for row in candidate_rows if row.get("single_channel_sign") == "support_channel"]
    suppress = [row for row in candidate_rows if row.get("single_channel_sign") == "suppressor_or_blocker_channel"]
    if support and suppress:
        return "single_channel_mixed_sign_decomposition_positive"
    if support:
        return "single_channel_support_positive"
    if suppress:
        return "single_channel_suppressor_only"
    return "single_channel_weak_or_correlational"


def atlas_from_eval(payload: dict[str, Any]) -> dict[str, Any]:
    nodes = []
    edges = []
    for row in payload.get("channel_eval_rows") or []:
        axis_id = f"axis:{row.get('model')}:{row.get('relation')}:{row.get('language_pair')}:h{row.get('hidden_idx')}"
        component_id = f"component:{row.get('model')}:L{row.get('layer_idx')}:mlp_channel:{row.get('channel_id')}"
        boundary_id = f"boundary:{row.get('model')}:{row.get('relation')}:{row.get('language_pair')}:full_vocab"
        nodes.extend(
            [
                {"id": axis_id, "kind": "axis_node", "domain": row.get("relation"), "model": row.get("model")},
                {
                    "id": component_id,
                    "kind": "component_node",
                    "component_type": "mlp_channel",
                    "model": row.get("model"),
                    "layer_idx": row.get("layer_idx"),
                    "channel_id": row.get("channel_id"),
                },
                {"id": boundary_id, "kind": "boundary_node", "boundary_type": "full_vocab", "model": row.get("model")},
            ]
        )
        level = "L4_component_causal_node" if abs(finite(row.get("boundary_slope"))) > 0.02 else "L3_transition_or_correlational"
        edges.append(
            {
                "id": f"edge:{row.get('channel_record_key')}",
                "source": axis_id,
                "target": component_id,
                "then": boundary_id,
                "kind": "axis_to_component_to_boundary",
                "model": row.get("model"),
                "domain": row.get("relation"),
                "evidence_level": level,
                "single_channel_sign": row.get("single_channel_sign"),
                "boundary_slope": row.get("boundary_slope"),
                "relation_slope": row.get("relation_slope"),
                "channel_source": row.get("channel_source"),
                "failure_or_risk": "group_level_mixed" if row.get("single_channel_sign") == "mixed_side_effect_channel" else None,
            }
        )
    dedup_nodes = {node["id"]: node for node in nodes}
    return {
        "schema_version": "mechanism_atlas_v1",
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "title": "Single-Channel Sign Decomposition Atlas",
        "model": payload.get("model"),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "nodes": list(dedup_nodes.values()),
        "edges": edges,
    }


def eval_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    records = load_phase944_activation_records(args)
    dry_payload = {
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "title": "Single-Channel Sign Decomposition Atlas",
        "model": args.model,
        "phase944_round": args.phase944_round,
        "record_count": len(records),
        "selected_records": [
            {
                "record_key": record_key(row),
                "relation": row.get("relation"),
                "language_pair": row.get("language_pair"),
                "hidden_idx": row.get("hidden_idx"),
                "candidate_channel_ids": row.get("candidate_channel_ids"),
                "random_channel_ids": row.get("random_channel_ids"),
            }
            for row in records
        ],
    }
    if args.dry_run:
        payload = {**dry_payload, "status": "dry_run"}
        write_json(out_dir / f"phase198_{args.model}_summary.json", payload)
        write_jsonl(out_dir / f"phase198_{args.model}_rows.jsonl", [])
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return payload

    rows: list[dict[str, Any]] = []
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
        holdout_by_pair, meta = p944.build_holdout_samples(args, model, tokenizer, device)
        all_samples = p944.unique_samples([sample for vals in holdout_by_pair.values() for sample in vals])
        if int(args.max_total_samples) > 0:
            all_samples = all_samples[: int(args.max_total_samples)]
        labels_by_relation = {relation: p938.relation_labels(all_samples, relation) for relation in p944.parse_csv(args.relations)}
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

        for record in records:
            relation = str(record.get("relation"))
            language_pair = str(record.get("language_pair"))
            samples = holdout_by_pair.get((relation, language_pair)) or []
            samples = [sample for sample in samples if p938.sample_key(sample) in baseline_metrics]
            if int(args.max_samples_per_record) > 0:
                samples = samples[: int(args.max_samples_per_record)]
            if not samples:
                continue
            layer_idx = int((record.get("component_mapping") or {}).get("layer_idx", int(record.get("hidden_idx", 1)) - 1))
            channel_specs = []
            for channel_id in record.get("candidate_channel_ids") or []:
                channel_specs.append(("candidate", int(channel_id)))
            if not args.no_random_controls:
                for channel_id in record.get("random_channel_ids") or []:
                    channel_specs.append(("same_layer_random", int(channel_id)))
            for channel_source, channel_id in channel_specs:
                for suffix, factor in [("ablate", 0.0), ("boost", float(args.boost_factor))]:
                    condition = f"{channel_source}_single_{suffix}"
                    patched_logits = p944.patched_logits_with_channel_scale(
                        model,
                        tokenizer,
                        device,
                        samples,
                        layer_idx,
                        [int(channel_id)],
                        float(factor),
                        int(args.batch_size),
                    )
                    for sample in samples:
                        key = p938.sample_key(sample)
                        if key not in patched_logits:
                            continue
                        relation_sample = str(sample["relation"])
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
                                record,
                                condition,
                                float(factor),
                                channel_source,
                                int(channel_id),
                                baseline_metrics[key],
                                patched,
                            )
                        )
                log(
                    f"{args.model}/{args.round_name}: {relation}:{language_pair} L{layer_idx} {channel_source} c{channel_id} samples={len(samples)}"
                )
    finally:
        if model is not None:
            p938.p862.p844.p828.release_model(model)
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    condition_rows = summarize_by(
        rows,
        [
            "channel_record_key",
            "record_key",
            "model",
            "relation",
            "language_pair",
            "condition",
            "channel_source",
            "channel_id",
            "factor",
        ],
    )
    channel_eval_rows = build_channel_eval(records, condition_rows)
    sign_counts = Counter(str(row.get("single_channel_sign")) for row in channel_eval_rows if row.get("channel_source") == "candidate")
    payload = {
        **dry_payload,
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        **meta,
        "rows": len(rows),
        "by_condition": condition_rows,
        "channel_eval_rows": channel_eval_rows,
        "candidate_sign_counts": dict(sign_counts),
        "evidence_label": evidence_label(channel_eval_rows),
        "boundary": "Single-channel MLP down-proj scaling audit; not natural-gate or rollout closure.",
    }
    atlas = atlas_from_eval(payload)
    write_json(out_dir / f"phase198_{args.model}_summary.json", payload)
    write_jsonl(out_dir / f"phase198_{args.model}_rows.jsonl", rows)
    write_json(out_dir / f"phase198_{args.model}_mechanism_atlas_v1.json", atlas)
    print(
        json.dumps(
            {
                "phase": PHASE,
                "model": args.model,
                "status": "complete",
                "evidence": payload["evidence_label"],
                "candidate_sign_counts": payload["candidate_sign_counts"],
                "top_channel_eval_rows": channel_eval_rows[:10],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return payload


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    summaries = [read_json(out_dir / f"phase198_{model}_summary.json") for model in MODELS]
    summaries = [summary for summary in summaries if summary]
    evidence_counts = Counter(str(summary.get("evidence_label") or summary.get("status")) for summary in summaries)
    channel_eval_rows = []
    condition_rows = []
    atlas_nodes = {}
    atlas_edges = []
    for summary in summaries:
        for row in summary.get("channel_eval_rows") or []:
            channel_eval_rows.append(dict(row))
        for row in summary.get("by_condition") or []:
            condition_rows.append(dict(row))
        atlas = read_json(out_dir / f"phase198_{summary.get('model')}_mechanism_atlas_v1.json")
        for node in atlas.get("nodes") or []:
            atlas_nodes[node["id"]] = node
        atlas_edges.extend(atlas.get("edges") or [])
    channel_eval_rows.sort(
        key=lambda row: (
            row.get("channel_source") != "candidate",
            -finite(row.get("boundary_slope"), -999.0),
            str(row.get("model")),
        )
    )
    payload = {
        "schema_version": "phase198_cross_model_summary_v1",
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "round": round_name,
        "status": "complete" if summaries else "missing",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": [summary.get("model") for summary in summaries],
        "evidence_counts": dict(evidence_counts),
        "model_summaries": summaries,
        "top_channel_eval_rows": channel_eval_rows[:160],
        "condition_rows": condition_rows,
    }
    atlas_payload = {
        "schema_version": "mechanism_atlas_v1",
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "round": round_name,
        "title": "Cross-Model Single-Channel Sign Decomposition Atlas",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "nodes": list(atlas_nodes.values()),
        "edges": atlas_edges,
    }
    write_json(out_dir / "phase198_cross_model_summary.json", payload)
    write_json(out_dir / "phase198_cross_model_mechanism_atlas_v1.json", atlas_payload)
    write_summary_md(out_dir / "phase198_cross_model_summary.md", payload)
    return payload


def write_summary_md(path: Path, payload: dict[str, Any]) -> None:
    lines = ["# Phase 198 single-channel sign decomposition atlas", ""]
    lines += ["## Evidence", ""]
    for key, value in sorted((payload.get("evidence_counts") or {}).items()):
        lines.append(f"- {key}: {value}")
    lines += ["", "## Top Channel Eval Rows", ""]
    lines.append("| model | relation | pair | hidden | channel | source | sign | boundary slope | relation slope | target slope | rows |")
    lines.append("| --- | --- | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |")
    for row in payload.get("top_channel_eval_rows") or []:
        lines.append(
            "| {model} | {relation} | {pair} | {hidden} | {channel} | {source} | {sign} | {bs} | {rs} | {ts} | {rows} |".format(
                model=row.get("model"),
                relation=row.get("relation"),
                pair=row.get("language_pair"),
                hidden=row.get("hidden_idx"),
                channel=row.get("channel_id"),
                source=row.get("channel_source"),
                sign=row.get("single_channel_sign"),
                bs=row.get("boundary_slope"),
                rs=row.get("relation_slope"),
                ts=row.get("target_logit_slope"),
                rows=row.get("rows"),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="single_channel_sign_decomposition_atlas")
    parser.add_argument("--phase944-round", default="activation_weighted_mlp_channel_causal_audit")
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
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--max-samples-per-record", type=int, default=0)
    parser.add_argument("--max-total-samples", type=int, default=0)
    parser.add_argument("--min-activation-gap", type=float, default=0.25)
    parser.add_argument("--only-positive-source", action="store_true")
    parser.add_argument("--no-random-controls", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--attn-implementations", default="flash_attention_2,sdpa")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize-round", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.summarize_round:
        payload = summarize_round(args.round_name)
        print(json.dumps({"phase": PHASE, "status": payload["status"], "evidence": payload["evidence_counts"]}, ensure_ascii=False, indent=2))
        return
    if not args.model:
        raise SystemExit("--model is required unless --summarize-round is set")
    eval_model(args)


if __name__ == "__main__":
    main()
