#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))

import phase938_semantic_factor_causal_transfer_audit as p938  # noqa: E402
import phase239_stable_protocol_prompt_trigger_atlas as p239  # noqa: E402
import phase259_template_semantic_done_disentanglement as p259  # noqa: E402


PHASE = 262
SOURCE_PHASE = 261
SCHEMA_VERSION = "1.0.0"
MODELS = ["qwen3", "glm4", "deepseek7b"]
ATLAS_ROOT = Path("tests/result/pattern_family_atlas/v1")
RESULT_ROOT = Path("tests/result/phase262_continuation_regime_decomposition_atlas")
ROUND_DEFAULT = "continuation_regime_decomposition_atlas"

SPECS = {
    "qwen3": {"final_layer": 33},
    "glm4": {"final_layer": 32},
    "deepseek7b": {"final_layer": 27},
}

STOP_GROUPS = {
    "eos": ["<eos>"],
    "period": [".", "。"],
    "newline": ["\n", "\n\n"],
    "end_boundary": ["</s>", "<|endoftext|>", "<|im_end|>"],
}

CONT_GROUPS = {
    "continue_the": [" the", " The", " this", " This"],
    "continue_because": [" because", " Because", " since", " Since"],
    "continue_and": [" and", " And", " also", " Also"],
    "continue_comma": [",", "，", ";", "；"],
    "continue_is": [" is", " are", " was", " means"],
    "continue_for": [" for", " to", " of", " in order"],
    "continue_next_sentence": ["\nThe", "\n\nThe", " Moreover", " However", " Therefore"],
    "continue_format": ["\nAnswer", "\nExplanation", "\nResult", "\nNote"],
    "continue_json_structure": ['"', '":', '",', "{", "}"],
    "continue_list_item": ["\n-", "\n1", "\n2", "- "],
}

REGIMES = [
    {
        "regime_id": "plain",
        "family": "natural_language",
        "description": "original prefix",
        "apply": lambda text: text,
    },
    {
        "regime_id": "period_boundary",
        "family": "boundary",
        "description": "period boundary appended",
        "apply": lambda text: text.rstrip() + ".",
    },
    {
        "regime_id": "newline_boundary",
        "family": "boundary",
        "description": "newline boundary appended",
        "apply": lambda text: text.rstrip() + "\n",
    },
    {
        "regime_id": "comma_stub",
        "family": "natural_language",
        "description": "comma continuation stub",
        "apply": lambda text: text.rstrip() + ",",
    },
    {
        "regime_id": "because_stub",
        "family": "explanation",
        "description": "because continuation stub",
        "apply": lambda text: text.rstrip() + " because",
    },
    {
        "regime_id": "answer_anchor",
        "family": "protocol",
        "description": "answer anchor appended",
        "apply": lambda text: text.rstrip() + "\nAnswer:",
    },
    {
        "regime_id": "json_structure",
        "family": "structured_protocol",
        "description": "json field prefix appended",
        "apply": lambda text: text.rstrip() + '\n{"answer":',
    },
    {
        "regime_id": "list_item",
        "family": "structured_protocol",
        "description": "list item prefix appended",
        "apply": lambda text: text.rstrip() + "\n-",
    },
    {
        "regime_id": "next_sentence",
        "family": "natural_language",
        "description": "next sentence cue appended",
        "apply": lambda text: text.rstrip() + "\nThe",
    },
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def append_unique_jsonl(path: Path, rows: list[dict[str, Any]], id_key: str) -> None:
    old_rows = read_jsonl(path)
    by_id: dict[str, dict[str, Any]] = {}
    for row in old_rows + rows:
        key = str(row.get(id_key) or row.get("metric_id") or row.get("edge_id") or row.get("observation_id") or len(by_id))
        by_id[key] = row
    write_jsonl(path, list(by_id.values()))


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return default
        return value
    except (TypeError, ValueError):
        return default


def mean_by(rows: list[dict[str, Any]], group_key: str, value_key: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(group_key))].append(safe_float(row.get(value_key)))
    return {k: round(mean(v), 6) for k, v in grouped.items() if v}


def token_ids(tokenizer: Any, texts: list[str]) -> list[int]:
    ids = set()
    for text in texts:
        if text == "<eos>":
            if tokenizer.eos_token_id is not None:
                ids.add(int(tokenizer.eos_token_id))
            continue
        try:
            encoded = tokenizer.encode(text, add_special_tokens=False)
        except Exception:
            encoded = []
        if encoded:
            ids.add(int(encoded[-1]))
    return sorted(ids)


def group_score(logits: torch.Tensor, ids: list[int]) -> float:
    vals = [float(logits[i].item()) for i in ids if 0 <= int(i) < logits.numel()]
    return max(vals) if vals else -1e9


def capture_logits(model_obj: Any, tokenizer: Any, device: torch.device, text: str, aliases: list[str]) -> tuple[torch.Tensor, dict[str, Any], int]:
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=1536).to(device)
    last_pos = int(encoded["attention_mask"].sum(dim=1).item()) - 1
    with torch.inference_mode():
        out = model_obj(**encoded, use_cache=False, output_hidden_states=False, return_dict=True)
    logits = out.logits[0, last_pos].detach().float().cpu()
    readout = p239.readout_metrics(tokenizer, logits, aliases)
    return logits, readout, int(encoded["input_ids"].shape[1])


def score_channels(logits: torch.Tensor, stop_ids: dict[str, list[int]], cont_ids: dict[str, list[int]]) -> dict[str, Any]:
    stop_scores = {name: group_score(logits, ids) for name, ids in stop_ids.items()}
    cont_scores = {name: group_score(logits, ids) for name, ids in cont_ids.items()}
    r_stop_name, r_stop = max(stop_scores.items(), key=lambda x: x[1])
    r_cont_name, r_cont = max(cont_scores.items(), key=lambda x: x[1])
    channel_margins = {name: round(value - r_stop, 6) for name, value in cont_scores.items()}
    sorted_channels = sorted(cont_scores.items(), key=lambda x: x[1], reverse=True)
    return {
        **{f"stop_{k}_logit": round(v, 6) for k, v in stop_scores.items()},
        **{f"{k}_logit": round(v, 6) for k, v in cont_scores.items()},
        **{f"{k}_vs_stop_margin": channel_margins[k] for k in cont_scores},
        "r_stop_name": r_stop_name,
        "r_continue_name": r_cont_name,
        "top_continue_channel": r_cont_name,
        "second_continue_channel": sorted_channels[1][0] if len(sorted_channels) > 1 else "",
        "r_stop": round(r_stop, 6),
        "r_continue": round(r_cont, 6),
        "stop_continue_margin": round(r_stop - r_cont, 6),
        "top_continue_vs_stop_margin": round(r_cont - r_stop, 6),
        "competition_winner": "stop" if r_stop > r_cont else "continue",
    }


def token_coverage_rows(model: str, tokenizer: Any, stop_ids: dict[str, list[int]], cont_ids: dict[str, list[int]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scope, groups in [("stop", stop_ids), ("continuation", cont_ids)]:
        for group_id, ids in groups.items():
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": "Phase262",
                    "created_at": utc_now(),
                    "coverage_id": f"phase262:coverage:{model}:{scope}:{group_id}",
                    "model": model,
                    "scope": scope,
                    "group_id": group_id,
                    "token_count": len(ids),
                    "token_ids": ids,
                    "coverage_ok": bool(ids),
                }
            )
    return rows


def evaluate_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    behavior_rows = p259.load_rows(args.model, int(args.max_cases_per_mode))
    model_obj = None
    tokenizer = None
    channel_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    matrix_rows: list[dict[str, Any]] = []
    protocol_rows: list[dict[str, Any]] = []
    structured_rows: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    try:
        model_obj, tokenizer, device, _attn_impl = p938.p862.p844.p828.p796.load_model_bf16_prefer_flash(
            args.model, args.attn_implementations
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        stop_ids = {name: token_ids(tokenizer, texts) for name, texts in STOP_GROUPS.items()}
        cont_ids = {name: token_ids(tokenizer, texts) for name, texts in CONT_GROUPS.items()}
        coverage = token_coverage_rows(args.model, tokenizer, stop_ids, cont_ids)
        for row_idx, row in enumerate(behavior_rows, start=1):
            case_id = str(row["case_id"])
            variant_id = str(row["variant_id"])
            mode_id = str(row["mode_id"])
            aliases = list(row.get("target_aliases") or [])
            for condition, base_text in p259.condition_texts(row).items():
                for regime in REGIMES:
                    regime_id = str(regime["regime_id"])
                    regime_family = str(regime["family"])
                    text = regime["apply"](base_text)
                    logits, readout, token_len = capture_logits(model_obj, tokenizer, device, text, aliases)
                    scores = score_channels(logits, stop_ids, cont_ids)
                    base = {
                        "schema_version": SCHEMA_VERSION,
                        "phase_id": "Phase262",
                        "created_at": utc_now(),
                        "model": args.model,
                        "case_id": case_id,
                        "variant_id": variant_id,
                        "mode_id": mode_id,
                        "condition": condition,
                        "regime_id": regime_id,
                        "regime_family": regime_family,
                        "prompt_token_length": token_len,
                        **scores,
                        **readout,
                    }
                    matrix_id = f"phase262:matrix:{args.model}:{case_id}:{variant_id}:{condition}:{regime_id}"
                    matrix_row = {**base, "matrix_id": matrix_id}
                    matrix_rows.append(matrix_row)
                    source_rows.append(
                        {
                            **base,
                            "source_id": f"phase262:source:{args.model}:{case_id}:{variant_id}:{condition}:{regime_id}",
                            "source_hypothesis": continuation_source(condition, regime_id, regime_family, scores["top_continue_channel"]),
                            "candidate_for_suppression": bool(scores["competition_winner"] == "continue" and scores["top_continue_vs_stop_margin"] > 0),
                        }
                    )
                    if regime_family in {"protocol", "explanation"}:
                        protocol_rows.append({**base, "protocol_id": f"phase262:protocol:{args.model}:{case_id}:{variant_id}:{condition}:{regime_id}"})
                    if regime_family == "structured_protocol":
                        structured_rows.append({**base, "structured_id": f"phase262:structured:{args.model}:{case_id}:{variant_id}:{condition}:{regime_id}"})
                    for channel in CONT_GROUPS:
                        channel_rows.append(
                            {
                                "schema_version": SCHEMA_VERSION,
                                "phase_id": "Phase262",
                                "created_at": utc_now(),
                                "channel_id": f"phase262:channel:{args.model}:{case_id}:{variant_id}:{condition}:{regime_id}:{channel}",
                                "model": args.model,
                                "case_id": case_id,
                                "variant_id": variant_id,
                                "mode_id": mode_id,
                                "condition": condition,
                                "regime_id": regime_id,
                                "regime_family": regime_family,
                                "continuation_channel": channel,
                                "channel_logit": base[f"{channel}_logit"],
                                "channel_vs_stop_margin": base[f"{channel}_vs_stop_margin"],
                                "is_top_continue_channel": channel == scores["top_continue_channel"],
                                "competition_winner": scores["competition_winner"],
                                "r_stop": scores["r_stop"],
                                "r_continue": scores["r_continue"],
                            }
                        )
                    observations.append(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "phase_id": "Phase262",
                            "created_at": utc_now(),
                            "observation_id": matrix_id,
                            "case_id": case_id,
                            "model": args.model,
                            "family_id": "output_protocol",
                            "mode_id": mode_id,
                            "variant_id": variant_id,
                            "level": "continuation_regime_decomposition",
                            "component": f"{condition}:{regime_id}",
                            "metric_name": "top_continue_vs_stop_margin",
                            "metric_value": scores["top_continue_vs_stop_margin"],
                            "metric_unit": "logit",
                            "winner": scores["competition_winner"],
                            "top_continue_channel": scores["top_continue_channel"],
                        }
                    )
            if row_idx % 6 == 0:
                log(f"{args.model}: captured {row_idx}/{len(behavior_rows)} base cases")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        if model_obj is not None:
            p938.p862.p844.p828.release_model(model_obj)
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    metrics.extend(make_metrics(args.model, matrix_rows, channel_rows, coverage))
    edges.extend(make_edges(args.model, matrix_rows, channel_rows))
    winner_counts = Counter(str(x["competition_winner"]) for x in matrix_rows)
    channel_counts = Counter(str(x["top_continue_channel"]) for x in matrix_rows if x.get("competition_winner") == "continue")
    payload = {
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "title": "Continuation regime decomposition atlas",
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "schema_version": SCHEMA_VERSION,
        "model": args.model,
        "base_case_count": len(behavior_rows),
        "regime_count": len(REGIMES),
        "channel_rows": len(channel_rows),
        "source_map_rows": len(source_rows),
        "matrix_rows": len(matrix_rows),
        "protocol_rows": len(protocol_rows),
        "structured_rows": len(structured_rows),
        "observation_rows": len(observations),
        "metric_rows": len(metrics),
        "graph_edges": len(edges),
        "token_coverage_rows": len(coverage),
        "missing_rows": len(missing_rows),
        "competition_winner_counts": dict(winner_counts.most_common()),
        "stop_win_rate": round(winner_counts.get("stop", 0) / len(matrix_rows), 6) if matrix_rows else 0.0,
        "top_continue_channel_counts": dict(channel_counts.most_common()),
        "mean_top_continue_vs_stop_margin_by_regime": mean_by(matrix_rows, "regime_id", "top_continue_vs_stop_margin"),
        "mean_stop_continue_margin_by_regime": mean_by(matrix_rows, "regime_id", "stop_continue_margin"),
        "mean_channel_vs_stop_margin": mean_by(channel_rows, "continuation_channel", "channel_vs_stop_margin"),
        "top_suppression_candidates": top_suppression_candidates(source_rows),
    }
    write_model_outputs(out_dir, args.model, payload, channel_rows, source_rows, matrix_rows, protocol_rows, structured_rows, observations, metrics, edges, coverage, missing_rows)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def continuation_source(condition: str, regime_id: str, regime_family: str, top_channel: str) -> str:
    if regime_family == "structured_protocol":
        return "structured_protocol_continuation"
    if regime_family == "protocol":
        return "answer_protocol_continuation"
    if regime_family == "explanation" or "because" in top_channel:
        return "explanation_continuation"
    if regime_family == "boundary":
        return "boundary_aftereffect_or_stop_failure"
    if regime_id == "next_sentence" or "next_sentence" in top_channel:
        return "next_sentence_continuation"
    if "template_complete" in condition:
        return "template_induced_continuation"
    return "natural_language_continuation"


def top_suppression_candidates(rows: list[dict[str, Any]], limit: int = 10) -> list[dict[str, Any]]:
    candidates = [r for r in rows if r.get("candidate_for_suppression")]
    candidates.sort(key=lambda r: safe_float(r.get("top_continue_vs_stop_margin")), reverse=True)
    return [
        {
            "model": r["model"],
            "case_id": r["case_id"],
            "condition": r["condition"],
            "regime_id": r["regime_id"],
            "top_continue_channel": r["top_continue_channel"],
            "top_continue_vs_stop_margin": r["top_continue_vs_stop_margin"],
            "source_hypothesis": r["source_hypothesis"],
        }
        for r in candidates[:limit]
    ]


def make_metrics(model: str, matrix_rows: list[dict[str, Any]], channel_rows: list[dict[str, Any]], coverage: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    for regime_id, value in mean_by(matrix_rows, "regime_id", "top_continue_vs_stop_margin").items():
        metrics.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase262",
                "created_at": utc_now(),
                "metric_id": f"phase262:{model}:regime:{regime_id}:mean_top_continue_vs_stop",
                "scope": "continuation_regime",
                "model": model,
                "regime_id": regime_id,
                "metric_name": "mean_top_continue_vs_stop_margin",
                "metric_value": value,
                "rows": sum(1 for x in matrix_rows if x.get("regime_id") == regime_id),
            }
        )
    for channel, value in mean_by(channel_rows, "continuation_channel", "channel_vs_stop_margin").items():
        top_count = sum(1 for x in channel_rows if x.get("continuation_channel") == channel and x.get("is_top_continue_channel"))
        metrics.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase262",
                "created_at": utc_now(),
                "metric_id": f"phase262:{model}:channel:{channel}:mean_vs_stop",
                "scope": "continuation_channel",
                "model": model,
                "continuation_channel": channel,
                "metric_name": "mean_channel_vs_stop_margin",
                "metric_value": value,
                "top_channel_count": top_count,
                "rows": sum(1 for x in channel_rows if x.get("continuation_channel") == channel),
            }
        )
    bad_coverage = sum(1 for x in coverage if not x.get("coverage_ok"))
    metrics.append(
        {
            "schema_version": SCHEMA_VERSION,
            "phase_id": "Phase262",
            "created_at": utc_now(),
            "metric_id": f"phase262:{model}:tokenbank:coverage_missing_groups",
            "scope": "tokenbank_coverage",
            "model": model,
            "metric_name": "coverage_missing_groups",
            "metric_value": bad_coverage,
            "rows": len(coverage),
        }
    )
    return metrics


def make_edges(model: str, matrix_rows: list[dict[str, Any]], channel_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    for regime_id, value in mean_by(matrix_rows, "regime_id", "top_continue_vs_stop_margin").items():
        edges.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase262",
                "created_at": utc_now(),
                "edge_id": f"phase262:regime:{model}:{regime_id}:to_continuation_field",
                "source": f"node:regime:{regime_id}",
                "target": "node:ContinuationField",
                "edge_type": "continuation_regime_effect",
                "model": model,
                "evidence_type": "static_prefix_logit_channel_decomposition",
                "effect_direction": "continuation_dominant" if value > 0 else "stop_dominant",
                "effect_size": value,
                "confidence": 0.42 if value > 0 else 0.35,
                "supporting_phases": ["Phase261", "Phase262"],
                "status": "source_map_not_causal_closure",
            }
        )
    for channel, value in mean_by(channel_rows, "continuation_channel", "channel_vs_stop_margin").items():
        edges.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase262",
                "created_at": utc_now(),
                "edge_id": f"phase262:channel:{model}:{channel}:to_stop_competition",
                "source": f"node:{channel}",
                "target": "node:StopVsContinuationCompetition",
                "edge_type": "continuation_channel_competition",
                "model": model,
                "evidence_type": "tokenbank_logit_margin",
                "effect_direction": "beats_stop_on_average" if value > 0 else "below_stop_on_average",
                "effect_size": value,
                "confidence": 0.40 if value > 0 else 0.32,
                "supporting_phases": ["Phase261", "Phase262"],
                "status": "channel_decomposition_not_suppression_test",
            }
        )
    return edges


def write_model_outputs(
    out_dir: Path,
    model: str,
    payload: dict[str, Any],
    channel_rows: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
    matrix_rows: list[dict[str, Any]],
    protocol_rows: list[dict[str, Any]],
    structured_rows: list[dict[str, Any]],
    observations: list[dict[str, Any]],
    metrics: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    coverage: list[dict[str, Any]],
    missing_rows: list[dict[str, Any]],
) -> None:
    write_json(out_dir / f"phase262_{model}_summary.json", payload)
    write_jsonl(out_dir / f"phase262_{model}_continuation_channel_rows.jsonl", channel_rows)
    write_jsonl(out_dir / f"phase262_{model}_continuation_source_map_rows.jsonl", source_rows)
    write_jsonl(out_dir / f"phase262_{model}_stop_continue_matrix_rows.jsonl", matrix_rows)
    write_jsonl(out_dir / f"phase262_{model}_protocol_continuation_rows.jsonl", protocol_rows)
    write_jsonl(out_dir / f"phase262_{model}_structured_continuation_rows.jsonl", structured_rows)
    write_jsonl(out_dir / f"phase262_{model}_observations.jsonl", observations)
    write_jsonl(out_dir / f"phase262_{model}_metrics.jsonl", metrics)
    write_jsonl(out_dir / f"phase262_{model}_graph_edges.jsonl", edges)
    write_jsonl(out_dir / f"phase262_{model}_token_coverage_rows.jsonl", coverage)
    write_jsonl(out_dir / f"phase262_{model}_missing_rows.jsonl", missing_rows)


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    summaries = [read_json(out_dir / f"phase262_{model}_summary.json") for model in MODELS]
    summaries = [x for x in summaries if x]
    channels: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    matrix: list[dict[str, Any]] = []
    protocols: list[dict[str, Any]] = []
    structured: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for model in MODELS:
        channels.extend(read_jsonl(out_dir / f"phase262_{model}_continuation_channel_rows.jsonl"))
        sources.extend(read_jsonl(out_dir / f"phase262_{model}_continuation_source_map_rows.jsonl"))
        matrix.extend(read_jsonl(out_dir / f"phase262_{model}_stop_continue_matrix_rows.jsonl"))
        protocols.extend(read_jsonl(out_dir / f"phase262_{model}_protocol_continuation_rows.jsonl"))
        structured.extend(read_jsonl(out_dir / f"phase262_{model}_structured_continuation_rows.jsonl"))
        observations.extend(read_jsonl(out_dir / f"phase262_{model}_observations.jsonl"))
        metrics.extend(read_jsonl(out_dir / f"phase262_{model}_metrics.jsonl"))
        edges.extend(read_jsonl(out_dir / f"phase262_{model}_graph_edges.jsonl"))
        coverage.extend(read_jsonl(out_dir / f"phase262_{model}_token_coverage_rows.jsonl"))
        missing.extend(read_jsonl(out_dir / f"phase262_{model}_missing_rows.jsonl"))
    progress = {
        "pattern_family_atlas": 0.85,
        "trace_signature_validation": 0.46,
        "semantic_done_signature": 0.24,
        "done_state_cluster_map": 0.21,
        "template_semantic_disentanglement": 0.19,
        "sdone_rstop_bridge": 0.08,
        "stop_continuation_competition": 0.18,
        "continuation_regime_decomposition": 0.16,
        "residual_state_signature": 0.55,
        "readout_competition_trace": 0.76,
        "stepwise_rollout_trace": 0.41,
        "causal_closure": 0.17,
        "general_language_mechanism_confidence": 0.65,
    }
    winner_counts = Counter(str(x["competition_winner"]) for x in matrix)
    channel_counts = Counter(str(x["top_continue_channel"]) for x in matrix if x.get("competition_winner") == "continue")
    source_counts = Counter(str(x.get("source_hypothesis")) for x in sources if x.get("competition_winner") == "continue")
    payload = {
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "title": "Continuation regime decomposition atlas",
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "schema_version": SCHEMA_VERSION,
        "round_name": round_name,
        "model_summaries": summaries,
        "channel_rows": len(channels),
        "source_map_rows": len(sources),
        "matrix_rows": len(matrix),
        "protocol_rows": len(protocols),
        "structured_rows": len(structured),
        "observation_rows": len(observations),
        "metric_rows": len(metrics),
        "graph_edges": len(edges),
        "token_coverage_rows": len(coverage),
        "missing_rows": len(missing),
        "competition_winner_counts": dict(winner_counts.most_common()),
        "stop_win_rate": round(winner_counts.get("stop", 0) / len(matrix), 6) if matrix else 0.0,
        "top_continue_channel_counts": dict(channel_counts.most_common()),
        "source_hypothesis_counts": dict(source_counts.most_common()),
        "mean_top_continue_vs_stop_margin_by_regime": mean_by(matrix, "regime_id", "top_continue_vs_stop_margin"),
        "mean_stop_continue_margin_by_regime": mean_by(matrix, "regime_id", "stop_continue_margin"),
        "mean_channel_vs_stop_margin": mean_by(channels, "continuation_channel", "channel_vs_stop_margin"),
        "top_suppression_candidates": top_suppression_candidates(sources),
        "progress": progress,
    }
    write_json(out_dir / "phase262_cross_model_summary.json", payload)
    write_jsonl(out_dir / "phase262_continuation_channel_rows.jsonl", channels)
    write_jsonl(out_dir / "phase262_continuation_source_map_rows.jsonl", sources)
    write_jsonl(out_dir / "phase262_stop_continue_matrix_rows.jsonl", matrix)
    write_jsonl(out_dir / "phase262_protocol_continuation_rows.jsonl", protocols)
    write_jsonl(out_dir / "phase262_structured_continuation_rows.jsonl", structured)
    write_jsonl(out_dir / "phase262_observations.jsonl", observations)
    write_jsonl(out_dir / "phase262_metrics.jsonl", metrics)
    write_jsonl(out_dir / "phase262_graph_edges.jsonl", edges)
    write_jsonl(out_dir / "phase262_token_coverage_rows.jsonl", coverage)
    write_jsonl(out_dir / "phase262_missing_rows.jsonl", missing)
    write_report(out_dir, payload)
    append_unique_jsonl(ATLAS_ROOT / "observations.jsonl", observations, "observation_id")
    append_unique_jsonl(ATLAS_ROOT / "metrics.jsonl", metrics, "metric_id")
    append_unique_jsonl(ATLAS_ROOT / "graph_edges.jsonl", edges, "edge_id")
    write_json(ATLAS_ROOT / "progress.json", {**read_json(ATLAS_ROOT / "progress.json"), **progress, "last_phase": "Phase262", "updated_at": utc_now()})
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def write_report(out_dir: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Phase262 Continuation Regime Decomposition Atlas",
        "",
        f"- status: {payload['status']}",
        f"- matrix_rows: {payload['matrix_rows']}",
        f"- channel_rows: {payload['channel_rows']}",
        f"- source_map_rows: {payload['source_map_rows']}",
        f"- stop_win_rate: {payload['stop_win_rate']}",
        f"- competition_winner_counts: {json.dumps(payload['competition_winner_counts'], ensure_ascii=False)}",
        f"- top_continue_channel_counts: {json.dumps(payload['top_continue_channel_counts'], ensure_ascii=False)}",
        f"- source_hypothesis_counts: {json.dumps(payload['source_hypothesis_counts'], ensure_ascii=False)}",
        f"- mean_top_continue_vs_stop_margin_by_regime: {json.dumps(payload['mean_top_continue_vs_stop_margin_by_regime'], ensure_ascii=False)}",
        f"- mean_channel_vs_stop_margin: {json.dumps(payload['mean_channel_vs_stop_margin'], ensure_ascii=False)}",
        f"- top_suppression_candidates: {json.dumps(payload['top_suppression_candidates'][:5], ensure_ascii=False)}",
    ]
    (out_dir / "phase262_continuation_decomposition_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default=ROUND_DEFAULT)
    parser.add_argument("--max-cases-per-mode", type=int, default=8)
    parser.add_argument("--attn-implementations", default="flash_attention_2,sdpa,eager")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.summarize:
        summarize_round(args.round_name)
        return
    if args.model:
        evaluate_model(args)
        return
    for model in MODELS:
        args.model = model
        evaluate_model(args)
    summarize_round(args.round_name)


if __name__ == "__main__":
    main()
