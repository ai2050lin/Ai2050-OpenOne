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
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))

import phase938_semantic_factor_causal_transfer_audit as p938  # noqa: E402
import phase940_semantic_boundary_bridge_audit as p940  # noqa: E402
import phase944_activation_weighted_mlp_channel_causal_audit as p944  # noqa: E402
import phase200_protocol_gated_rollout_repair_audit as p200  # noqa: E402
from model_utils import get_layers  # noqa: E402


PHASE = 201
SOURCE_PHASE = 200
MODELS = ["qwen3", "glm4", "deepseek7b"]
RESULT_ROOT = Path("tests/result/phase201_stop_prose_component_atlas")


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


def token_candidates(tokenizer, phrases: list[str]) -> list[int]:
    ids: list[int] = []
    for phrase in phrases:
        try:
            encoded = tokenizer.encode(phrase, add_special_tokens=False)
        except Exception:
            encoded = []
        if encoded:
            token_id = int(encoded[0])
            if token_id not in ids:
                ids.append(token_id)
    return ids


def decode_ids(tokenizer, ids: list[int]) -> list[str]:
    out = []
    for token_id in ids:
        try:
            out.append(tokenizer.decode([int(token_id)]))
        except Exception:
            out.append(str(token_id))
    return out


def max_score(logits: torch.Tensor, token_ids: list[int]) -> tuple[float | None, int | None]:
    valid = [int(x) for x in token_ids if 0 <= int(x) < int(logits.numel())]
    if not valid:
        return None, None
    best_id = max(valid, key=lambda token_id: float(logits[token_id].item()))
    return float(logits[best_id].item()), int(best_id)


def token_groups(tokenizer) -> dict[str, list[int]]:
    boundary = p940.boundary_token_groups(tokenizer)
    eos = list(boundary.get("eos") or [])
    period = list(boundary.get("punctuation_period") or [])
    newline = token_candidates(tokenizer, ["\n", "\n\n"])
    stop = sorted(set(eos + period + newline))
    prose_words = ["because", "and", "usually", "which", "that", "what", "is", "it", "the", "a", "an", "this"]
    prose_phrases = []
    for word in prose_words:
        prose_phrases.extend([word, " " + word, word.capitalize(), " " + word.capitalize()])
    prose = sorted(set(token_candidates(tokenizer, prose_phrases)))
    return {
        "eos": sorted(set(eos)),
        "period": sorted(set(period)),
        "newline": sorted(set(newline)),
        "stop": stop,
        "prose": prose,
    }


def object_token_ids(tokenizer, sample: dict[str, Any]) -> list[int]:
    obj = str(sample.get("object") or "")
    if not obj:
        return []
    pieces = [obj, " " + obj, obj.capitalize(), " " + obj.capitalize()]
    return sorted(set(token_candidates(tokenizer, pieces)))


def make_post_answer_sample(sample: dict[str, Any], protocol: str, language_pair: str | None = None) -> dict[str, Any]:
    prompt = p200.protocol_prompt(sample, protocol).rstrip()
    target = str(sample.get("target_label") or "").strip()
    post_prompt = f"{prompt} {target}".strip()
    return {
        **sample,
        "sample_id": f"{sample.get('sample_id')}|post_answer|{protocol}",
        "source_sample_id": sample.get("sample_id"),
        "prompt": post_prompt,
        "prompt_protocol": protocol,
        "post_answer_target": target,
        "source_language_pair": language_pair or sample.get("language_pair"),
    }


def metric_row(tokenizer, logits: torch.Tensor, sample: dict[str, Any], groups: dict[str, list[int]]) -> dict[str, Any]:
    stop_score, stop_id = max_score(logits, groups["stop"])
    period_score, period_id = max_score(logits, groups["period"])
    eos_score, eos_id = max_score(logits, groups["eos"])
    newline_score, newline_id = max_score(logits, groups["newline"])
    prose_score, prose_id = max_score(logits, groups["prose"])
    echo_ids = object_token_ids(tokenizer, sample)
    echo_score, echo_id = max_score(logits, echo_ids)
    target_ids = p938.first_token_candidates(tokenizer, str(sample.get("target_label") or ""))
    target_score, target_id = max_score(logits, target_ids)
    return {
        "stop_logit": stop_score,
        "stop_token_id": stop_id,
        "stop_token": None if stop_id is None else tokenizer.decode([stop_id]),
        "period_logit": period_score,
        "period_token_id": period_id,
        "period_token": None if period_id is None else tokenizer.decode([period_id]),
        "eos_logit": eos_score,
        "eos_token_id": eos_id,
        "newline_logit": newline_score,
        "newline_token_id": newline_id,
        "prose_logit": prose_score,
        "prose_token_id": prose_id,
        "prose_token": None if prose_id is None else tokenizer.decode([prose_id]),
        "echo_logit": echo_score,
        "echo_token_id": echo_id,
        "echo_token": None if echo_id is None else tokenizer.decode([echo_id]),
        "target_logit_after_answer": target_score,
        "target_token_after_answer": target_id,
        "stop_margin": None if stop_score is None or prose_score is None else float(stop_score - prose_score),
        "prose_margin": None if prose_score is None or stop_score is None else float(prose_score - stop_score),
        "echo_margin": None if echo_score is None or stop_score is None else float(echo_score - stop_score),
        "period_vs_prose_margin": None if period_score is None or prose_score is None else float(period_score - prose_score),
        "eos_vs_prose_margin": None if eos_score is None or prose_score is None else float(eos_score - prose_score),
    }


def pearson_by_channel(acts: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    valid = torch.isfinite(y)
    if int(valid.sum().item()) < 3:
        return torch.zeros((acts.shape[1],), dtype=torch.float32)
    x = acts[valid].float()
    yy = y[valid].float()
    x = x - x.mean(dim=0, keepdim=True)
    yy = yy - yy.mean()
    denom = torch.sqrt((x * x).sum(dim=0) * torch.sum(yy * yy)).clamp_min(1e-8)
    return (x * yy[:, None]).sum(dim=0) / denom


def top_corr_rows(
    model_name: str,
    relation: str,
    language_pair: str,
    protocol: str,
    layer_idx: int,
    samples: list[dict[str, Any]],
    activations: dict[str, torch.Tensor],
    metrics: dict[str, dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    usable = [sample for sample in samples if p938.sample_key(sample) in activations and p938.sample_key(sample) in metrics]
    if len(usable) < 3:
        return []
    acts = torch.stack([activations[p938.sample_key(sample)] for sample in usable]).float()
    stop_y = torch.tensor([finite(metrics[p938.sample_key(sample)].get("stop_margin"), float("nan")) for sample in usable])
    prose_y = torch.tensor([finite(metrics[p938.sample_key(sample)].get("prose_margin"), float("nan")) for sample in usable])
    echo_y = torch.tensor([finite(metrics[p938.sample_key(sample)].get("echo_margin"), float("nan")) for sample in usable])
    stop_corr = pearson_by_channel(acts, stop_y)
    anti_prose_corr = -pearson_by_channel(acts, prose_y)
    echo_suppress_corr = -pearson_by_channel(acts, echo_y)
    rows = []
    specs = [
        ("stop_candidate", stop_corr, "stop_corr"),
        ("anti_prose_candidate", anti_prose_corr, "anti_prose_corr"),
        ("echo_suppress_candidate", echo_suppress_corr, "echo_suppress_corr"),
    ]
    for candidate_type, corr, corr_key in specs:
        values, indices = torch.topk(corr, k=min(int(top_k), int(corr.numel())))
        for score, channel in zip(values.tolist(), indices.tolist()):
            rows.append(
                {
                    "phase": PHASE,
                    "source_phase": SOURCE_PHASE,
                    "row_kind": "phase201_candidate_scan_row",
                    "model": model_name,
                    "relation": relation,
                    "language_pair": language_pair,
                    "prompt_protocol": protocol,
                    "layer_idx": int(layer_idx),
                    "channel_id": int(channel),
                    "candidate_type": candidate_type,
                    corr_key: float(score),
                    "candidate_score": float(score),
                    "sample_count": len(usable),
                    "stop_margin_mean": mean([metrics[p938.sample_key(sample)].get("stop_margin") for sample in usable]),
                    "prose_margin_mean": mean([metrics[p938.sample_key(sample)].get("prose_margin") for sample in usable]),
                    "echo_margin_mean": mean([metrics[p938.sample_key(sample)].get("echo_margin") for sample in usable]),
                }
            )
    return rows


def summarize_metrics(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[tuple(row.get(key) for key in keys)].append(row)
    out = []
    for key_tuple, items in buckets.items():
        row = {key: value for key, value in zip(keys, key_tuple)}
        row.update(
            {
                "rows": len(items),
                "stop_margin_mean": mean([item.get("stop_margin") for item in items]),
                "prose_margin_mean": mean([item.get("prose_margin") for item in items]),
                "echo_margin_mean": mean([item.get("echo_margin") for item in items]),
                "period_vs_prose_margin_mean": mean([item.get("period_vs_prose_margin") for item in items]),
                "eos_vs_prose_margin_mean": mean([item.get("eos_vs_prose_margin") for item in items]),
                "stop_token_counts": dict(Counter(str(item.get("stop_token")) for item in items)),
                "prose_token_counts": dict(Counter(str(item.get("prose_token")) for item in items)),
            }
        )
        out.append(row)
    out.sort(key=lambda row: tuple(str(row.get(key)) for key in keys))
    return out


def select_causal_candidates(scan_rows: list[dict[str, Any]], max_per_type: int) -> list[dict[str, Any]]:
    out = []
    seen = set()
    for candidate_type in ["stop_candidate", "anti_prose_candidate", "echo_suppress_candidate"]:
        rows = [row for row in scan_rows if row.get("candidate_type") == candidate_type]
        rows.sort(key=lambda row: finite(row.get("candidate_score"), -999.0), reverse=True)
        count = 0
        for row in rows:
            key = (
                row.get("model"),
                row.get("relation"),
                row.get("language_pair"),
                row.get("prompt_protocol"),
                row.get("layer_idx"),
                row.get("channel_id"),
                row.get("candidate_type"),
            )
            if key in seen:
                continue
            seen.add(key)
            out.append(dict(row))
            count += 1
            if count >= int(max_per_type):
                break
    return out


def delta_row(base: dict[str, Any], patched: dict[str, Any]) -> dict[str, Any]:
    keys = ["stop_margin", "prose_margin", "echo_margin", "period_vs_prose_margin", "eos_vs_prose_margin"]
    out = {}
    for key in keys:
        out[f"{key}_delta"] = None if base.get(key) is None or patched.get(key) is None else float(patched[key] - base[key])
    return out


def summarize_causal(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    keys = ["model", "candidate_type", "relation", "language_pair", "prompt_protocol", "layer_idx", "channel_id", "condition"]
    for row in rows:
        buckets[tuple(row.get(key) for key in keys)].append(row)
    out = []
    for key_tuple, items in buckets.items():
        row = {key: value for key, value in zip(keys, key_tuple)}
        row.update(
            {
                "rows": len(items),
                "stop_margin_delta_mean": mean([item.get("stop_margin_delta") for item in items]),
                "prose_margin_delta_mean": mean([item.get("prose_margin_delta") for item in items]),
                "echo_margin_delta_mean": mean([item.get("echo_margin_delta") for item in items]),
                "period_vs_prose_margin_delta_mean": mean([item.get("period_vs_prose_margin_delta") for item in items]),
                "eos_vs_prose_margin_delta_mean": mean([item.get("eos_vs_prose_margin_delta") for item in items]),
            }
        )
        if row["candidate_type"] == "stop_candidate":
            score = finite(row.get("stop_margin_delta_mean")) - max(0.0, finite(row.get("prose_margin_delta_mean")))
        elif row["candidate_type"] == "anti_prose_candidate":
            score = -finite(row.get("prose_margin_delta_mean")) + max(0.0, finite(row.get("stop_margin_delta_mean")))
        else:
            score = -finite(row.get("echo_margin_delta_mean")) + max(0.0, finite(row.get("stop_margin_delta_mean")))
        row["causal_candidate_score"] = float(score)
        out.append(row)
    out.sort(key=lambda row: finite(row.get("causal_candidate_score"), -999.0), reverse=True)
    return out


def scan_layers_for_model(model, args: argparse.Namespace) -> list[int]:
    layers = get_layers(model)
    total = len(layers)
    if args.scan_layers:
        return sorted({idx for idx in [int(x) for x in parse_csv(args.scan_layers)] if 0 <= idx < total})
    fractions = [0.45, 0.6, 0.75, 0.9]
    return sorted({min(total - 1, max(0, int(round((total - 1) * frac)))) for frac in fractions})


def eval_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    dry_payload = {
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "title": "Stop / EOS / Punctuation / Prose-Continuation Component Atlas",
        "model": args.model,
        "prompt_protocols": parse_csv(args.prompt_protocols),
    }
    if args.dry_run:
        payload = {**dry_payload, "status": "dry_run"}
        write_json(out_dir / f"phase201_{args.model}_summary.json", payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return payload

    model = None
    tokenizer = None
    metric_rows: list[dict[str, Any]] = []
    scan_rows: list[dict[str, Any]] = []
    causal_rows: list[dict[str, Any]] = []
    meta: dict[str, Any] = {}
    token_group_sizes: dict[str, int] = {}
    try:
        model, tokenizer, device, _attn_impl = p938.p862.p844.p828.p796.load_model_bf16_prefer_flash(
            args.model, args.attn_implementations
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        groups = token_groups(tokenizer)
        token_group_sizes = {name: len(ids) for name, ids in groups.items()}
        holdout_by_pair, meta = p944.build_holdout_samples(args, model, tokenizer, device)
        protocols = parse_csv(args.prompt_protocols)
        selected_pairs = sorted(holdout_by_pair.keys())[: int(args.max_pairs)]
        post_by_pair_protocol: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
        all_post_samples = []
        for relation, language_pair in selected_pairs:
            samples = holdout_by_pair.get((relation, language_pair)) or []
            if int(args.max_samples_per_pair) > 0:
                samples = samples[: int(args.max_samples_per_pair)]
            for protocol in protocols:
                post_samples = [make_post_answer_sample(sample, protocol, language_pair) for sample in samples]
                post_by_pair_protocol[(relation, language_pair, protocol)] = post_samples
                all_post_samples.extend(post_samples)
        baseline_logits = p938.forward_logits(model, tokenizer, device, all_post_samples, int(args.batch_size))
        metric_by_key = {}
        for sample in all_post_samples:
            key = p938.sample_key(sample)
            if key not in baseline_logits:
                continue
            metrics = metric_row(tokenizer, baseline_logits[key], sample, groups)
            metric_by_key[key] = metrics
            metric_rows.append(
                {
                    "phase": PHASE,
                    "source_phase": SOURCE_PHASE,
                    "row_kind": "phase201_post_answer_metric_row",
                    "model": args.model,
                    "relation": sample.get("relation"),
                    "language_pair": sample.get("source_language_pair"),
                    "prompt_protocol": sample.get("prompt_protocol"),
                    "sample_id": key,
                    "source_sample_id": sample.get("source_sample_id"),
                    "object": sample.get("object"),
                    "target_label": sample.get("target_label"),
                    "prompt": sample.get("prompt"),
                    **metrics,
                }
            )

        layers = scan_layers_for_model(model, args)
        for layer_idx in layers:
            activations = p944.capture_down_inputs(model, tokenizer, device, all_post_samples, int(layer_idx), int(args.batch_size))
            for relation, language_pair, protocol in sorted(post_by_pair_protocol):
                samples = post_by_pair_protocol[(relation, language_pair, protocol)]
                scan_rows.extend(
                    top_corr_rows(
                        args.model,
                        relation,
                        language_pair,
                        protocol,
                        int(layer_idx),
                        samples,
                        activations,
                        metric_by_key,
                        int(args.top_channels_per_layer),
                    )
                )
            log(f"{args.model}/{args.round_name}: scanned layer={layer_idx} samples={len(all_post_samples)}")
            del activations
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        causal_candidates = select_causal_candidates(scan_rows, int(args.max_causal_per_type))
        by_bucket = {(rel, pair, protocol): samples for (rel, pair, protocol), samples in post_by_pair_protocol.items()}
        for cand in causal_candidates:
            samples = by_bucket.get((cand["relation"], cand["language_pair"], cand["prompt_protocol"])) or []
            if int(args.max_causal_samples) > 0:
                samples = samples[: int(args.max_causal_samples)]
            base_metrics = {p938.sample_key(sample): metric_by_key[p938.sample_key(sample)] for sample in samples if p938.sample_key(sample) in metric_by_key}
            for condition, factor in [("ablate", 0.0), ("boost", float(args.boost_factor))]:
                patched_logits = p944.patched_logits_with_channel_scale(
                    model,
                    tokenizer,
                    device,
                    samples,
                    int(cand["layer_idx"]),
                    [int(cand["channel_id"])],
                    float(factor),
                    int(args.batch_size),
                )
                for sample in samples:
                    key = p938.sample_key(sample)
                    if key not in patched_logits or key not in base_metrics:
                        continue
                    patched = metric_row(tokenizer, patched_logits[key], sample, groups)
                    causal_rows.append(
                        {
                            "phase": PHASE,
                            "source_phase": SOURCE_PHASE,
                            "row_kind": "phase201_causal_validation_row",
                            "model": args.model,
                            "candidate_type": cand.get("candidate_type"),
                            "candidate_score": cand.get("candidate_score"),
                            "relation": cand.get("relation"),
                            "language_pair": cand.get("language_pair"),
                            "prompt_protocol": cand.get("prompt_protocol"),
                            "layer_idx": int(cand["layer_idx"]),
                            "channel_id": int(cand["channel_id"]),
                            "condition": condition,
                            "factor": factor,
                            "sample_id": key,
                            "object": sample.get("object"),
                            "target_label": sample.get("target_label"),
                            **delta_row(base_metrics[key], patched),
                        }
                    )
                log(
                    f"{args.model}/{args.round_name}: causal {cand.get('candidate_type')} L{cand.get('layer_idx')} c{cand.get('channel_id')} {condition} samples={len(samples)}"
                )
                del patched_logits
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    finally:
        if model is not None:
            p938.p862.p844.p828.release_model(model)
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    causal_summary_rows = summarize_causal(causal_rows)
    payload = {
        **dry_payload,
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **meta,
        "metric_rows": len(metric_rows),
        "scan_rows": len(scan_rows),
        "causal_rows": len(causal_rows),
        "token_group_sizes": token_group_sizes,
        "metric_summary_rows": summarize_metrics(metric_rows, ["model", "relation", "language_pair", "prompt_protocol"]),
        "top_scan_rows": sorted(scan_rows, key=lambda row: finite(row.get("candidate_score"), -999.0), reverse=True)[:50],
        "causal_summary_rows": causal_summary_rows,
        "boundary": "Post-answer logit/activation atlas; candidates are L5c/L5d probes, not L6 natural closure.",
    }
    write_json(out_dir / f"phase201_{args.model}_summary.json", payload)
    write_jsonl(out_dir / f"phase201_{args.model}_metric_rows.jsonl", metric_rows)
    write_jsonl(out_dir / f"phase201_{args.model}_scan_rows.jsonl", scan_rows)
    write_jsonl(out_dir / f"phase201_{args.model}_causal_rows.jsonl", causal_rows)
    print(
        json.dumps(
            {
                "phase": PHASE,
                "model": args.model,
                "status": "complete",
                "metric_rows": len(metric_rows),
                "scan_rows": len(scan_rows),
                "causal_rows": len(causal_rows),
                "top_causal_summary_rows": causal_summary_rows[:12],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return payload


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    summaries = [read_json(out_dir / f"phase201_{model}_summary.json") for model in MODELS]
    summaries = [summary for summary in summaries if summary]
    metric_summary_rows = []
    scan_rows = []
    causal_summary_rows = []
    for summary in summaries:
        metric_summary_rows.extend(dict(row) for row in summary.get("metric_summary_rows") or [])
        scan_rows.extend(dict(row) for row in summary.get("top_scan_rows") or [])
        causal_summary_rows.extend(dict(row) for row in summary.get("causal_summary_rows") or [])
    causal_summary_rows.sort(key=lambda row: finite(row.get("causal_candidate_score"), -999.0), reverse=True)
    payload = {
        "schema_version": "phase201_cross_model_summary_v1",
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "round": round_name,
        "status": "complete" if summaries else "missing",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": [summary.get("model") for summary in summaries],
        "model_summaries": summaries,
        "metric_summary_rows": metric_summary_rows,
        "top_scan_rows": scan_rows,
        "causal_summary_rows": causal_summary_rows,
    }
    write_json(out_dir / "phase201_cross_model_summary.json", payload)
    write_summary_md(out_dir / "phase201_cross_model_summary.md", payload)
    return payload


def write_summary_md(path: Path, payload: dict[str, Any]) -> None:
    lines = ["# Phase 201 stop/prose component atlas", ""]
    lines.append("## Post-answer metric summary")
    lines.append("| model | relation | pair | protocol | rows | stop margin | prose margin | echo margin |")
    lines.append("| --- | --- | --- | --- | ---: | ---: | ---: | ---: |")
    for row in payload.get("metric_summary_rows") or []:
        lines.append(
            f"| {row.get('model')} | {row.get('relation')} | {row.get('language_pair')} | {row.get('prompt_protocol')} | "
            f"{row.get('rows')} | {finite(row.get('stop_margin_mean')):.3f} | {finite(row.get('prose_margin_mean')):.3f} | {finite(row.get('echo_margin_mean')):.3f} |"
        )
    lines.append("")
    lines.append("## Top causal candidates")
    lines.append("| model | type | relation | protocol | L | c | condition | stop Δ | prose Δ | echo Δ | score |")
    lines.append("| --- | --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |")
    for row in (payload.get("causal_summary_rows") or [])[:40]:
        lines.append(
            f"| {row.get('model')} | {row.get('candidate_type')} | {row.get('relation')} | {row.get('prompt_protocol')} | "
            f"{row.get('layer_idx')} | {row.get('channel_id')} | {row.get('condition')} | "
            f"{finite(row.get('stop_margin_delta_mean')):.3f} | {finite(row.get('prose_margin_delta_mean')):.3f} | "
            f"{finite(row.get('echo_margin_delta_mean')):.3f} | {finite(row.get('causal_candidate_score')):.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="stop_prose_component_atlas")
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
    parser.add_argument("--max-pairs", type=int, default=4)
    parser.add_argument("--max-samples-per-pair", type=int, default=18)
    parser.add_argument("--prompt-protocols", default="plain,short_answer,stop_explicit")
    parser.add_argument("--scan-layers", default="")
    parser.add_argument("--top-channels-per-layer", type=int, default=4)
    parser.add_argument("--max-causal-per-type", type=int, default=3)
    parser.add_argument("--max-causal-samples", type=int, default=18)
    parser.add_argument("--boost-factor", type=float, default=1.5)
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
