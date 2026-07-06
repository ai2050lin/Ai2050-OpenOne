#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase856_identity_class_overlap_cross_domain_rollout_audit as p856  # noqa: E402
import phase938_semantic_factor_causal_transfer_audit as p938  # noqa: E402
import phase944_activation_weighted_mlp_channel_causal_audit as p944  # noqa: E402


PHASE = 199
SOURCE_PHASE = 198
MODELS = ["qwen3", "glm4", "deepseek7b"]
RESULT_ROOT = Path("tests/result/phase199_l4_edge_natural_gate_rollout_audit")
PHASE198_ROOT = Path("tests/result/phase198_single_channel_sign_decomposition_atlas")
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


def parse_csv(text: str) -> list[str]:
    return [part.strip() for part in str(text or "").split(",") if part.strip()]


def record_key(row: dict[str, Any]) -> str:
    return f"{row.get('model')}|{row.get('relation')}|{row.get('language_pair')}|h{row.get('hidden_idx')}"


def clean_generated(text: str) -> str:
    return p856.clean_text(text)


def strict_protocol_drift(text: str) -> bool:
    raw = str(text or "").strip()
    raw_words = re.findall(r"[A-Za-z\u4e00-\u9fff]+", raw)
    raw_lowered = raw.lower()
    if "\n" in raw:
        return True
    if len(raw_words) > 4:
        return True
    if any(marker in raw_lowered for marker in ["because", "typically", "usually", "answer", "what is", ":", "。", "，"]):
        return True
    cleaned = clean_generated(text)
    if not cleaned:
        return True
    words = re.findall(r"[A-Za-z\u4e00-\u9fff]+", cleaned)
    if len(words) > 4:
        return True
    lowered = cleaned.lower()
    return any(marker in lowered for marker in ["because", "typically", "usually", "answer", ":", "。", "，"])


def rollout_case(sample: dict[str, Any]) -> dict[str, Any]:
    target = str(sample.get("target_label") or "")
    return {
        "object": str(sample.get("object") or ""),
        "canonical_answer": target,
        "answer_aliases": [target],
    }


def classify_rollout(generated: str, sample: dict[str, Any]) -> dict[str, Any]:
    out = p856.classify_rollout(generated, rollout_case(sample))
    out["protocol_drift"] = bool(strict_protocol_drift(generated))
    out["long_rollout_stable"] = bool(
        out.get("rollout_clear_answer_class") and not out.get("rollout_object_echo") and not out.get("protocol_drift")
    )
    return out


def load_phase198_edges(args: argparse.Namespace) -> list[dict[str, Any]]:
    path = PHASE198_ROOT / args.phase198_round / f"phase198_{args.model}_summary.json"
    summary = read_json(path)
    rows = [dict(row) for row in summary.get("channel_eval_rows") or [] if row.get("channel_source") == "candidate"]
    selected = []
    for row in rows:
        sign = str(row.get("single_channel_sign"))
        if sign in {"support_channel", "mixed_side_effect_channel", "suppressor_or_blocker_channel"}:
            selected.append(row)
            continue
        if (
            args.include_primary_near_zero_control
            and row.get("model") == "qwen3"
            and row.get("relation") == "color"
            and row.get("language_pair") == "en->en"
            and int(row.get("channel_id")) == 2509
        ):
            selected.append(row)
    selected.sort(
        key=lambda row: (
            str(row.get("single_channel_sign")) != "support_channel",
            -abs(finite(row.get("boundary_slope"))),
            str(row.get("relation")),
        )
    )
    if int(args.max_edges_per_model) > 0:
        selected = selected[: int(args.max_edges_per_model)]
    return selected


def load_phase944_records(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    path = PHASE944_ROOT / args.phase944_round / f"phase944_{args.model}_summary.json"
    summary = read_json(path)
    return {record_key(row): dict(row) for row in summary.get("activation_records") or []}


def encode_prompt(tokenizer, device: torch.device, prompt: str) -> dict[str, torch.Tensor]:
    enc = tokenizer(str(prompt), return_tensors="pt", truncation=True, max_length=256)
    return {key: value.to(device) for key, value in enc.items()}


def generated_text(tokenizer, output_ids: torch.Tensor, prompt_len: int) -> str:
    return tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True).strip()


def generate_with_channel_scale(
    model,
    tokenizer,
    device: torch.device,
    prompt: str,
    layer_idx: int,
    channel_id: int | None,
    factor: float | None,
    max_new_tokens: int,
) -> str:
    enc = encode_prompt(tokenizer, device, prompt)
    prompt_len = int(enc["input_ids"].shape[1])
    down_proj = None if channel_id is None or factor is None else p944.down_proj_for_layer(model, int(layer_idx))
    handle = None
    if down_proj is not None:

        def hook(_module, inputs):
            if not inputs or not torch.is_tensor(inputs[0]):
                return None
            hidden = inputs[0]
            if int(channel_id) < 0 or int(channel_id) >= int(hidden.shape[-1]):
                return None
            patched = hidden.clone()
            patched[:, -1, int(channel_id)] *= float(factor)
            return (patched, *inputs[1:])

        handle = down_proj.register_forward_pre_hook(hook)
    try:
        with torch.inference_mode():
            out = model.generate(
                **enc,
                max_new_tokens=int(max_new_tokens),
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        return generated_text(tokenizer, out, prompt_len)
    finally:
        if handle is not None:
            handle.remove()
        del enc
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def capture_channel_activation(
    model,
    tokenizer,
    device: torch.device,
    sample: dict[str, Any],
    layer_idx: int,
    channel_id: int,
) -> float | None:
    down_proj = p944.down_proj_for_layer(model, int(layer_idx))
    if down_proj is None:
        return None
    enc = encode_prompt(tokenizer, device, str(sample.get("prompt") or ""))
    captured: dict[str, float] = {}

    def hook(_module, inputs):
        if not inputs or not torch.is_tensor(inputs[0]):
            return None
        hidden = inputs[0]
        if int(channel_id) < int(hidden.shape[-1]):
            captured["value"] = float(hidden[0, -1, int(channel_id)].detach().float().cpu().item())
        return None

    handle = down_proj.register_forward_pre_hook(hook)
    try:
        with torch.inference_mode():
            model(**enc, use_cache=False, return_dict=True)
    finally:
        handle.remove()
        del enc
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return captured.get("value")


def summarize_condition(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "rollout_answer_class": sum(1 for row in rows if row.get("rollout_answer_class")),
        "rollout_clear_answer_class": sum(1 for row in rows if row.get("rollout_clear_answer_class")),
        "rollout_strict_canonical": sum(1 for row in rows if row.get("rollout_strict_canonical")),
        "rollout_object_echo": sum(1 for row in rows if row.get("rollout_object_echo")),
        "protocol_drift": sum(1 for row in rows if row.get("protocol_drift")),
        "long_rollout_stable": sum(1 for row in rows if row.get("long_rollout_stable")),
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


def build_edge_eval(edges: list[dict[str, Any]], condition_rows: list[dict[str, Any]], activation_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(str(row.get("edge_key")), str(row.get("condition"))): row for row in condition_rows}
    acts_by_edge: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in activation_rows:
        acts_by_edge[str(row.get("edge_key"))].append(row)
    out = []
    for edge in edges:
        edge_key = str(edge.get("edge_key"))
        base = by_key.get((edge_key, "baseline")) or {}
        ablate = by_key.get((edge_key, "ablate")) or {}
        boost = by_key.get((edge_key, "boost")) or {}
        stable_gain_boost = int(boost.get("long_rollout_stable", 0)) - int(base.get("long_rollout_stable", 0))
        stable_loss_ablate = int(base.get("long_rollout_stable", 0)) - int(ablate.get("long_rollout_stable", 0))
        clear_gain_boost = int(boost.get("rollout_clear_answer_class", 0)) - int(base.get("rollout_clear_answer_class", 0))
        clear_loss_ablate = int(base.get("rollout_clear_answer_class", 0)) - int(ablate.get("rollout_clear_answer_class", 0))
        act_rows = acts_by_edge.get(edge_key) or []
        clear_acts = [row.get("activation_abs") for row in act_rows if row.get("baseline_long_rollout_stable")]
        fail_acts = [row.get("activation_abs") for row in act_rows if not row.get("baseline_long_rollout_stable")]
        out.append(
            {
                **edge,
                "baseline_stable": base.get("long_rollout_stable", 0),
                "ablate_stable": ablate.get("long_rollout_stable", 0),
                "boost_stable": boost.get("long_rollout_stable", 0),
                "stable_gain_boost": stable_gain_boost,
                "stable_loss_ablate": stable_loss_ablate,
                "baseline_clear": base.get("rollout_clear_answer_class", 0),
                "ablate_clear": ablate.get("rollout_clear_answer_class", 0),
                "boost_clear": boost.get("rollout_clear_answer_class", 0),
                "clear_gain_boost": clear_gain_boost,
                "clear_loss_ablate": clear_loss_ablate,
                "activation_abs_mean_stable": mean(clear_acts),
                "activation_abs_mean_unstable": mean(fail_acts),
                "activation_abs_gap_stable_minus_unstable": None
                if mean(clear_acts) is None or mean(fail_acts) is None
                else float(mean(clear_acts) - mean(fail_acts)),
            }
        )
    out.sort(key=lambda row: (-(finite(row.get("stable_loss_ablate")) + finite(row.get("stable_gain_boost"))), str(row.get("edge_key"))))
    return out


def eval_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    edges = load_phase198_edges(args)
    phase944_records = load_phase944_records(args)
    for edge in edges:
        edge["edge_key"] = (
            f"{edge.get('model')}|{edge.get('relation')}|{edge.get('language_pair')}|"
            f"h{edge.get('hidden_idx')}|c{edge.get('channel_id')}|{edge.get('single_channel_sign')}"
        )
    dry_payload = {
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "title": "L4 Edge Natural-Gate and Rollout Closure Audit",
        "model": args.model,
        "selected_edges": edges,
    }
    if args.dry_run:
        payload = {**dry_payload, "status": "dry_run"}
        write_json(out_dir / f"phase199_{args.model}_summary.json", payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return payload

    rows: list[dict[str, Any]] = []
    activation_rows: list[dict[str, Any]] = []
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
        for edge in edges:
            rec = phase944_records.get(str(edge.get("record_key")))
            if not rec:
                continue
            samples = holdout_by_pair.get((str(edge.get("relation")), str(edge.get("language_pair")))) or []
            if int(args.max_samples_per_edge) > 0:
                samples = samples[: int(args.max_samples_per_edge)]
            layer_idx = int(edge.get("layer_idx"))
            channel_id = int(edge.get("channel_id"))
            for sample in samples:
                act = capture_channel_activation(model, tokenizer, device, sample, layer_idx, channel_id)
                condition_outputs = {}
                for condition, factor in [("baseline", None), ("ablate", 0.0), ("boost", float(args.boost_factor))]:
                    generated = generate_with_channel_scale(
                        model,
                        tokenizer,
                        device,
                        str(sample.get("prompt") or ""),
                        layer_idx,
                        channel_id if factor is not None else None,
                        factor,
                        int(args.max_new_tokens),
                    )
                    rollout = classify_rollout(generated, sample)
                    row = {
                        "phase": PHASE,
                        "source_phase": SOURCE_PHASE,
                        "row_kind": "phase199_l4_edge_rollout_row",
                        "edge_key": edge.get("edge_key"),
                        "model": args.model,
                        "relation": edge.get("relation"),
                        "language_pair": edge.get("language_pair"),
                        "hidden_idx": edge.get("hidden_idx"),
                        "layer_idx": layer_idx,
                        "channel_id": channel_id,
                        "single_channel_sign": edge.get("single_channel_sign"),
                        "boundary_slope": edge.get("boundary_slope"),
                        "relation_slope": edge.get("relation_slope"),
                        "condition": condition,
                        "factor": factor,
                        "sample_id": sample.get("sample_id"),
                        "domain": sample.get("domain"),
                        "object": sample.get("object"),
                        "target_label": sample.get("target_label"),
                        "prompt_language": sample.get("prompt_language"),
                        "prompt_template": sample.get("prompt_template"),
                        "prompt": sample.get("prompt"),
                        "generated": generated,
                        "activation": act,
                        "activation_abs": None if act is None else abs(float(act)),
                        **rollout,
                    }
                    rows.append(row)
                    condition_outputs[condition] = row
                base_row = condition_outputs.get("baseline") or {}
                activation_rows.append(
                    {
                        "edge_key": edge.get("edge_key"),
                        "model": args.model,
                        "sample_id": sample.get("sample_id"),
                        "activation": act,
                        "activation_abs": None if act is None else abs(float(act)),
                        "baseline_long_rollout_stable": bool(base_row.get("long_rollout_stable")),
                        "baseline_clear_answer": bool(base_row.get("rollout_clear_answer_class")),
                        "baseline_label": base_row.get("rollout_label"),
                    }
                )
            log(f"{args.model}/{args.round_name}: {edge.get('edge_key')} samples={len(samples)}")
    finally:
        if model is not None:
            p938.p862.p844.p828.release_model(model)
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    condition_rows = summarize_by(rows, ["edge_key", "model", "relation", "language_pair", "channel_id", "single_channel_sign", "condition"])
    edge_eval_rows = build_edge_eval(edges, condition_rows, activation_rows)
    payload = {
        **dry_payload,
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        **meta,
        "rows": len(rows),
        "condition_rows": condition_rows,
        "activation_rows": activation_rows,
        "edge_eval_rows": edge_eval_rows,
        "boundary": "Natural-gate activation association and short greedy rollout audit; not a proof of full language closure.",
    }
    write_json(out_dir / f"phase199_{args.model}_summary.json", payload)
    write_jsonl(out_dir / f"phase199_{args.model}_rows.jsonl", rows)
    write_jsonl(out_dir / f"phase199_{args.model}_activation_rows.jsonl", activation_rows)
    print(
        json.dumps(
            {
                "phase": PHASE,
                "model": args.model,
                "status": "complete",
                "rows": len(rows),
                "top_edge_eval_rows": edge_eval_rows[:8],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return payload


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    summaries = [read_json(out_dir / f"phase199_{model}_summary.json") for model in MODELS]
    summaries = [summary for summary in summaries if summary]
    edge_eval_rows = []
    condition_rows = []
    for summary in summaries:
        edge_eval_rows.extend(dict(row) for row in summary.get("edge_eval_rows") or [])
        condition_rows.extend(dict(row) for row in summary.get("condition_rows") or [])
    edge_eval_rows.sort(
        key=lambda row: (-(finite(row.get("stable_loss_ablate")) + finite(row.get("stable_gain_boost"))), str(row.get("edge_key")))
    )
    payload = {
        "schema_version": "phase199_cross_model_summary_v1",
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "round": round_name,
        "status": "complete" if summaries else "missing",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": [summary.get("model") for summary in summaries],
        "model_summaries": summaries,
        "edge_eval_rows": edge_eval_rows,
        "condition_rows": condition_rows,
    }
    write_json(out_dir / "phase199_cross_model_summary.json", payload)
    write_summary_md(out_dir / "phase199_cross_model_summary.md", payload)
    return payload


def write_summary_md(path: Path, payload: dict[str, Any]) -> None:
    lines = ["# Phase 199 L4 edge natural-gate and rollout audit", ""]
    lines.append("| model | edge | sign | base stable | ablate stable | boost stable | boost gain | ablate loss | act gap stable-unstable |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in payload.get("edge_eval_rows") or []:
        lines.append(
            "| {model} | {edge} | {sign} | {base} | {ablate} | {boost} | {gain} | {loss} | {actgap} |".format(
                model=row.get("model"),
                edge=row.get("edge_key"),
                sign=row.get("single_channel_sign"),
                base=row.get("baseline_stable"),
                ablate=row.get("ablate_stable"),
                boost=row.get("boost_stable"),
                gain=row.get("stable_gain_boost"),
                loss=row.get("stable_loss_ablate"),
                actgap=row.get("activation_abs_gap_stable_minus_unstable"),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="l4_edge_natural_gate_rollout_audit")
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
    parser.add_argument("--max-edges-per-model", type=int, default=7)
    parser.add_argument("--max-samples-per-edge", type=int, default=24)
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
