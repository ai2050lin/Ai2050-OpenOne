#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))

import phase938_semantic_factor_causal_transfer_audit as p938  # noqa: E402
import phase944_activation_weighted_mlp_channel_causal_audit as p944  # noqa: E402
import phase199_l4_edge_natural_gate_rollout_audit as p199  # noqa: E402


PHASE = 200
SOURCE_PHASE = 199
MODELS = ["qwen3", "glm4", "deepseek7b"]
RESULT_ROOT = Path("tests/result/phase200_protocol_gated_rollout_repair_audit")


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


def mean(values: list[Any]) -> float | None:
    vals = []
    for value in values:
        try:
            vals.append(float(value))
        except (TypeError, ValueError):
            continue
    return None if not vals else float(sum(vals) / len(vals))


def article_for(obj: str) -> str:
    return "an" if str(obj or "object")[:1].lower() in {"a", "e", "i", "o", "u"} else "a"


def protocol_prompt(sample: dict[str, Any], protocol: str) -> str:
    obj = str(sample.get("object") or "object")
    relation = str(sample.get("relation") or "")
    article = article_for(obj)
    if protocol == "plain":
        return str(sample.get("prompt") or "")
    if relation == "color":
        if protocol == "short_answer":
            return f"Answer with one English color word only. The color of {article} {obj} is"
        if protocol == "stop_explicit":
            return f"Answer with exactly one English color word and no explanation. The color of {article} {obj} is"
    if relation == "function":
        if protocol == "short_answer":
            return f"Answer with one English verb only. A common use for {article} {obj} is to"
        if protocol == "stop_explicit":
            return f"Answer with exactly one English verb and no explanation. A common use for {article} {obj} is to"
    if relation == "category":
        if protocol == "short_answer":
            return f"Answer with one short English category only. {article.capitalize()} {obj} is a type of"
        if protocol == "stop_explicit":
            return f"Answer with exactly one short English category and no explanation. {article.capitalize()} {obj} is a type of"
    return str(sample.get("prompt") or "")


def summarize_condition(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "rollout_clear_answer_class": sum(1 for row in rows if row.get("rollout_clear_answer_class")),
        "protocol_drift": sum(1 for row in rows if row.get("protocol_drift")),
        "long_rollout_stable": sum(1 for row in rows if row.get("long_rollout_stable")),
        "rollout_object_echo": sum(1 for row in rows if row.get("rollout_object_echo")),
        "labels": dict(Counter(str(row.get("rollout_label")) for row in rows)),
    }


def summarize_by(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[tuple(row.get(key) for key in keys)].append(row)
    out = []
    for key_tuple, items in buckets.items():
        row = {key: value for key, value in zip(keys, key_tuple)}
        row.update(summarize_condition(items))
        out.append(row)
    out.sort(key=lambda row: tuple(str(row.get(key)) for key in keys))
    return out


def build_protocol_eval(condition_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in condition_rows:
        grouped[(str(row.get("edge_key")), str(row.get("prompt_protocol")))].append(row)
    for (edge_key, protocol), items in grouped.items():
        by_condition = {str(row.get("condition")): row for row in items}
        base = by_condition.get("baseline") or {}
        ablate = by_condition.get("ablate") or {}
        boost = by_condition.get("boost") or {}
        out.append(
            {
                "edge_key": edge_key,
                "prompt_protocol": protocol,
                "baseline_stable": base.get("long_rollout_stable", 0),
                "ablate_stable": ablate.get("long_rollout_stable", 0),
                "boost_stable": boost.get("long_rollout_stable", 0),
                "baseline_clear": base.get("rollout_clear_answer_class", 0),
                "ablate_clear": ablate.get("rollout_clear_answer_class", 0),
                "boost_clear": boost.get("rollout_clear_answer_class", 0),
                "baseline_drift": base.get("protocol_drift", 0),
                "ablate_drift": ablate.get("protocol_drift", 0),
                "boost_drift": boost.get("protocol_drift", 0),
                "stable_gain_short_protocol": None,
                "stable_gain_boost": int(boost.get("long_rollout_stable", 0)) - int(base.get("long_rollout_stable", 0)),
                "stable_loss_ablate": int(base.get("long_rollout_stable", 0)) - int(ablate.get("long_rollout_stable", 0)),
            }
        )
    out.sort(key=lambda row: (str(row.get("edge_key")), str(row.get("prompt_protocol"))))
    return out


def eval_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    edges = p199.load_phase198_edges(args)
    for edge in edges:
        edge["edge_key"] = (
            f"{edge.get('model')}|{edge.get('relation')}|{edge.get('language_pair')}|"
            f"h{edge.get('hidden_idx')}|c{edge.get('channel_id')}|{edge.get('single_channel_sign')}"
        )
    if int(args.max_edges_per_model) > 0:
        edges = edges[: int(args.max_edges_per_model)]
    dry_payload = {
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "title": "Protocol-Gated Rollout Repair Audit",
        "model": args.model,
        "selected_edges": edges,
        "prompt_protocols": args.prompt_protocols.split(","),
    }
    if args.dry_run:
        payload = {**dry_payload, "status": "dry_run"}
        write_json(out_dir / f"phase200_{args.model}_summary.json", payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return payload

    rows: list[dict[str, Any]] = []
    model = None
    tokenizer = None
    meta: dict[str, Any] = {}
    try:
        model, tokenizer, device, attn_impl = p938.p862.p844.p828.p796.load_model_bf16_prefer_flash(
            args.model, args.attn_implementations
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        holdout_by_pair, meta = p944.build_holdout_samples(args, model, tokenizer, device)
        protocols = [p.strip() for p in args.prompt_protocols.split(",") if p.strip()]
        for edge in edges:
            samples = holdout_by_pair.get((str(edge.get("relation")), str(edge.get("language_pair")))) or []
            if int(args.max_samples_per_edge) > 0:
                samples = samples[: int(args.max_samples_per_edge)]
            layer_idx = int(edge.get("layer_idx"))
            channel_id = int(edge.get("channel_id"))
            for sample in samples:
                for protocol in protocols:
                    prompt = protocol_prompt(sample, protocol)
                    for condition, factor in [("baseline", None), ("ablate", 0.0), ("boost", float(args.boost_factor))]:
                        generated = p199.generate_with_channel_scale(
                            model,
                            tokenizer,
                            device,
                            prompt,
                            layer_idx,
                            channel_id if factor is not None else None,
                            factor,
                            int(args.max_new_tokens),
                        )
                        rollout = p199.classify_rollout(generated, {**sample, "prompt": prompt})
                        rows.append(
                            {
                                "phase": PHASE,
                                "source_phase": SOURCE_PHASE,
                                "row_kind": "phase200_protocol_gated_rollout_repair_row",
                                "edge_key": edge.get("edge_key"),
                                "model": args.model,
                                "relation": edge.get("relation"),
                                "language_pair": edge.get("language_pair"),
                                "hidden_idx": edge.get("hidden_idx"),
                                "layer_idx": layer_idx,
                                "channel_id": channel_id,
                                "single_channel_sign": edge.get("single_channel_sign"),
                                "boundary_slope": edge.get("boundary_slope"),
                                "prompt_protocol": protocol,
                                "condition": condition,
                                "factor": factor,
                                "sample_id": sample.get("sample_id"),
                                "domain": sample.get("domain"),
                                "object": sample.get("object"),
                                "target_label": sample.get("target_label"),
                                "prompt": prompt,
                                "generated": generated,
                                **rollout,
                            }
                        )
            log(f"{args.model}/{args.round_name}: {edge.get('edge_key')} samples={len(samples)} protocols={len(protocols)}")
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
        ["edge_key", "model", "relation", "language_pair", "channel_id", "single_channel_sign", "prompt_protocol", "condition"],
    )
    protocol_eval_rows = build_protocol_eval(condition_rows)
    payload = {
        **dry_payload,
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **meta,
        "rows": len(rows),
        "condition_rows": condition_rows,
        "protocol_eval_rows": protocol_eval_rows,
        "boundary": "Prompt-protocol repair plus single-channel ablate/boost; no internal stop/protocol component is patched.",
    }
    write_json(out_dir / f"phase200_{args.model}_summary.json", payload)
    write_jsonl(out_dir / f"phase200_{args.model}_rows.jsonl", rows)
    print(
        json.dumps(
            {
                "phase": PHASE,
                "model": args.model,
                "status": "complete",
                "rows": len(rows),
                "top_protocol_eval_rows": protocol_eval_rows[:12],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return payload


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    summaries = [read_json(out_dir / f"phase200_{model}_summary.json") for model in MODELS]
    summaries = [summary for summary in summaries if summary]
    condition_rows = []
    protocol_eval_rows = []
    for summary in summaries:
        condition_rows.extend(dict(row) for row in summary.get("condition_rows") or [])
        protocol_eval_rows.extend(dict(row) for row in summary.get("protocol_eval_rows") or [])
    payload = {
        "schema_version": "phase200_cross_model_summary_v1",
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "round": round_name,
        "status": "complete" if summaries else "missing",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": [summary.get("model") for summary in summaries],
        "model_summaries": summaries,
        "condition_rows": condition_rows,
        "protocol_eval_rows": protocol_eval_rows,
    }
    write_json(out_dir / "phase200_cross_model_summary.json", payload)
    write_summary_md(out_dir / "phase200_cross_model_summary.md", payload)
    return payload


def write_summary_md(path: Path, payload: dict[str, Any]) -> None:
    lines = ["# Phase 200 protocol-gated rollout repair audit", ""]
    lines.append("| model | edge | protocol | base stable | ablate stable | boost stable | base clear | drift |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for row in payload.get("protocol_eval_rows") or []:
        lines.append(
            f"| {str(row.get('edge_key')).split('|')[0]} | {row.get('edge_key')} | {row.get('prompt_protocol')} | "
            f"{row.get('baseline_stable')} | {row.get('ablate_stable')} | {row.get('boost_stable')} | "
            f"{row.get('baseline_clear')} | {row.get('baseline_drift')} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="protocol_gated_rollout_repair_audit")
    parser.add_argument("--phase198-round", default="single_channel_sign_decomposition_atlas")
    parser.add_argument("--phase944-round", default="activation_weighted_mlp_channel_causal_audit")
    parser.add_argument("--phase937-round", default="semantic_reuse_difference_state_atlas")
    parser.add_argument("--phase939-round", default="bilingual_specificity_tightening_audit")
    parser.add_argument("--phase940-round", default="semantic_boundary_bridge_audit")
    parser.add_argument("--phase943-round", default="consensus_coordinate_component_mapping_audit")
    parser.add_argument("--domains", default="")
    parser.add_argument("--relations", default="category,color,function")
    parser.add_argument("--max-objects-per-domain", type=int, default=8)
    parser.add_argument("--templates-per-language", type=int, default=2)
    parser.add_argument("--min-train-per-label", type=int, default=2)
    parser.add_argument("--min-specific-margin", type=float, default=0.05)
    parser.add_argument("--min-specific-gain", type=float, default=0.05)
    parser.add_argument("--min-phase940-bridge-gain", type=float, default=0.02)
    parser.add_argument("--max-specs-per-pair", type=int, default=12)
    parser.add_argument("--train-fraction", type=float, default=0.5)
    parser.add_argument("--min-train-specs", type=int, default=4)
    parser.add_argument("--min-holdout-specs", type=int, default=3)
    parser.add_argument("--max-edges-per-model", type=int, default=5)
    parser.add_argument("--max-samples-per-edge", type=int, default=16)
    parser.add_argument("--prompt-protocols", default="plain,short_answer,stop_explicit")
    parser.add_argument("--boost-factor", type=float, default=1.5)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--include-primary-near-zero-control", action="store_true", default=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--attn-implementations", default="flash_attention_2,sdpa")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize-round", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.summarize_round:
        payload = summarize_round(args.round_name)
        print(json.dumps({"phase": PHASE, "status": payload["status"], "models": payload["models"]}, ensure_ascii=False, indent=2))
        return
    if not args.model:
        raise SystemExit("--model is required unless --summarize-round is set")
    eval_model(args)


if __name__ == "__main__":
    main()
