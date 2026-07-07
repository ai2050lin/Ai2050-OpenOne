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
import phase201_stop_prose_component_atlas as p201  # noqa: E402
import phase210_minimal_pattern_transition_atlas as p210  # noqa: E402
from model_utils import get_layers  # noqa: E402


PHASE = 214
SOURCE_PHASE = 213
MODELS = ["qwen3", "glm4", "deepseek7b"]
INPUT_ROOT = Path("tests/result/phase210_minimal_pattern_transition_atlas")
RESULT_ROOT = Path("tests/result/phase214_prompt_trigger_token_path_atlas")


TRIGGER_SPECS: dict[str, list[dict[str, str]]] = {
    "answer_short": [
        {"label": "short_one_word", "phrase": "one English color word"},
        {"label": "short_only", "phrase": "only"},
    ],
    "answer_explain": [
        {"label": "explain_answer_first", "phrase": "answer first"},
        {"label": "explain_reason", "phrase": "short reason"},
        {"label": "explain_because", "phrase": "because"},
    ],
    "answer_list": [
        {"label": "list_three", "phrase": "three"},
        {"label": "list_plausible", "phrase": "plausible"},
        {"label": "list_short_answers", "phrase": "short answers"},
        {"label": "list_commas", "phrase": "commas"},
    ],
    "answer_repeat": [
        {"label": "repeat_exactly", "phrase": "exactly"},
        {"label": "repeat_same_answer_word", "phrase": "same answer word"},
        {"label": "repeat_twice", "phrase": "twice"},
        {"label": "repeat_comma", "phrase": "comma"},
    ],
    "answer_target_seeded": [
        {"label": "target_likely", "phrase": "likely"},
        {"label": "target_final_answer", "phrase": "final answer"},
        {"label": "target_only", "phrase": "only"},
    ],
}
COMMON_SPECS = [{"label": "answer_slot", "phrase": "Answer:"}]
ANCHOR_STEPS = [1, 2, 3, 6]


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def iter_jsonl(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_trajectories(input_dir: Path, model_name: str) -> list[dict[str, Any]]:
    return list(iter_jsonl(input_dir / f"phase210_{model_name}_trajectory_rows.jsonl") or [])


def balanced_rows(rows: list[dict[str, Any]], max_rows_per_pattern: int) -> list[dict[str, Any]]:
    buckets: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: {"match": [], "drift": []})
    for row in rows:
        pattern = str(row.get("pattern_id") or "")
        key = "match" if row.get("pattern_match") else "drift"
        buckets[pattern][key].append(row)
    selected: list[dict[str, Any]] = []
    half = max(1, int(max_rows_per_pattern) // 2)
    for pattern in sorted(buckets):
        matches = buckets[pattern]["match"]
        drifts = buckets[pattern]["drift"]
        picked = matches[:half] + drifts[:half]
        if len(picked) < int(max_rows_per_pattern):
            rest = [row for row in matches[half:] + drifts[half:] if row not in picked]
            picked.extend(rest[: int(max_rows_per_pattern) - len(picked)])
        selected.extend(picked)
    selected.sort(key=lambda row: (str(row.get("pattern_id")), not bool(row.get("pattern_match")), str(row.get("trajectory_id"))))
    return selected


def scan_layers_for_model(model, model_name: str, layer_text: str) -> list[int]:
    layers = get_layers(model)
    total = len(layers)
    if layer_text:
        return sorted({idx for idx in [int(x.strip()) for x in layer_text.split(",") if x.strip()] if 0 <= idx < total})
    fractions = [0.08, 0.18, 0.32, 0.5, 0.68, 0.82, 0.94]
    selected = {min(total - 1, max(0, int(round((total - 1) * frac)))) for frac in fractions}
    candidate_layers = {
        "qwen3": [31, 32, 33],
        "glm4": [28, 29, 30, 34, 35, 36],
        "deepseek7b": [23, 24, 25, 26, 27],
    }
    selected.update(idx for idx in candidate_layers.get(model_name, []) if 0 <= idx < total)
    return sorted(selected)


def encode_with_offsets(tokenizer, text: str) -> tuple[torch.Tensor, list[tuple[int, int]], list[str]]:
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False, return_offsets_mapping=True)
    input_ids = encoded["input_ids"][0]
    offsets = [(int(a), int(b)) for a, b in encoded["offset_mapping"][0].tolist()]
    tokens = [tokenizer.decode([int(tok)]) for tok in input_ids.tolist()]
    return input_ids, offsets, tokens


def find_phrase_spans(text: str, phrase: str) -> list[tuple[int, int]]:
    low = text.lower()
    target = phrase.lower()
    spans: list[tuple[int, int]] = []
    start = 0
    while True:
        pos = low.find(target, start)
        if pos < 0:
            break
        spans.append((pos, pos + len(target)))
        start = pos + max(1, len(target))
    return spans


def overlapping_token_indices(offsets: list[tuple[int, int]], start: int, end: int) -> list[int]:
    out = []
    for idx, (a, b) in enumerate(offsets):
        if b <= start or a >= end or a == b:
            continue
        out.append(idx)
    return out


def token_specs_for_prompt(pattern_id: str, prompt: str, offsets: list[tuple[int, int]], tokens: list[str]) -> list[dict[str, Any]]:
    specs = TRIGGER_SPECS.get(pattern_id, []) + COMMON_SPECS
    out: list[dict[str, Any]] = []
    seen = set()
    for spec in specs:
        for span_start, span_end in find_phrase_spans(prompt, spec["phrase"]):
            token_indices = overlapping_token_indices(offsets, span_start, span_end)
            if not token_indices:
                continue
            token_idx = int(token_indices[-1])
            key = (spec["label"], span_start, span_end, token_idx)
            if key in seen:
                continue
            seen.add(key)
            out.append(
                {
                    "trigger_label": spec["label"],
                    "trigger_phrase": spec["phrase"],
                    "char_start": span_start,
                    "char_end": span_end,
                    "token_start": int(token_indices[0]),
                    "token_end": int(token_indices[-1]),
                    "token_idx": token_idx,
                    "token_text": tokens[token_idx] if token_idx < len(tokens) else "",
                    "span_token_text": "".join(tokens[i] for i in token_indices if i < len(tokens)),
                }
            )
    return out


def forward_prompt_states(model, tokenizer, device: torch.device, text: str, layers: list[int]):
    input_ids, offsets, tokens = encode_with_offsets(tokenizer, text)
    attention_mask = torch.ones_like(input_ids).unsqueeze(0).to(device)
    input_ids_gpu = input_ids.unsqueeze(0).to(device)
    with torch.inference_mode():
        result = model(
            input_ids=input_ids_gpu,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
    states: dict[int, torch.Tensor] = {}
    for layer_idx in layers:
        hidden_idx = int(layer_idx) + 1
        if hidden_idx < len(result.hidden_states):
            states[int(layer_idx)] = result.hidden_states[hidden_idx][0].detach().float().cpu()
    del result, input_ids_gpu, attention_mask
    return input_ids, offsets, tokens, states


def forward_last_states(model, tokenizer, device: torch.device, texts: list[str], layers: list[int]) -> dict[str, dict[int, torch.Tensor]]:
    valid = [(idx, text) for idx, text in enumerate(texts) if text]
    out: dict[str, dict[int, torch.Tensor]] = {}
    if not valid:
        return out
    encoded = tokenizer([text for _, text in valid], return_tensors="pt", padding=True, add_special_tokens=False)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    last_pos = attention_mask.sum(dim=1).long() - 1
    batch_idx = torch.arange(input_ids.shape[0], device=device)
    with torch.inference_mode():
        result = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
    for layer_idx in layers:
        hidden_idx = int(layer_idx) + 1
        if hidden_idx >= len(result.hidden_states):
            continue
        values = result.hidden_states[hidden_idx][batch_idx, last_pos].detach().float().cpu()
        for local_idx, (source_idx, _text) in enumerate(valid):
            out.setdefault(str(source_idx), {})[int(layer_idx)] = values[local_idx]
    del result, input_ids, attention_mask
    return out


def vector_stats(vec: torch.Tensor) -> dict[str, float]:
    return {
        "residual_norm": float(torch.linalg.vector_norm(vec).item()),
        "residual_mean": float(vec.mean().item()),
        "residual_std": float(vec.std(unbiased=False).item()),
    }


def cosine(a: torch.Tensor, b: torch.Tensor) -> float | None:
    denom = float(torch.linalg.vector_norm(a).item() * torch.linalg.vector_norm(b).item())
    if denom <= 0:
        return None
    return float(torch.dot(a, b).item() / denom)


def anchor_texts(row: dict[str, Any]) -> list[tuple[str, str]]:
    prompt = str(row.get("prompt") or "")
    emitted = [str(tok) for tok in row.get("emitted_tokens") or []]
    anchors = [("prompt_last", prompt)]
    for step in ANCHOR_STEPS:
        if len(emitted) >= step:
            anchors.append((f"gen_after_step_{step}", prompt + "".join(emitted[:step])))
    if emitted:
        anchors.append(("gen_after_final", prompt + "".join(emitted)))
    return anchors


def eval_model(args: argparse.Namespace) -> dict[str, Any]:
    input_dir = INPUT_ROOT / args.phase210_round
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    model = None
    tokenizer = None
    token_rows: list[dict[str, Any]] = []
    path_rows: list[dict[str, Any]] = []
    selected_layers: list[int] = []
    selected: list[dict[str, Any]] = []
    try:
        model, tokenizer, device, _attn_impl = p938.p862.p844.p828.p796.load_model_bf16_prefer_flash(
            args.model, args.attn_implementations
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        _groups = p201.token_groups(tokenizer)
        rows = load_trajectories(input_dir, args.model)
        selected = balanced_rows(rows, int(args.max_rows_per_pattern))
        selected_layers = scan_layers_for_model(model, args.model, args.scan_layers)
        for row_idx, row in enumerate(selected):
            prompt = str(row.get("prompt") or "")
            try:
                _input_ids, offsets, tokens, prompt_states = forward_prompt_states(model, tokenizer, device, prompt, selected_layers)
            except Exception as exc:
                log(f"skip prompt states {row.get('trajectory_id')}: {exc}")
                continue
            specs = token_specs_for_prompt(str(row.get("pattern_id") or ""), prompt, offsets, tokens)
            if not specs:
                continue
            anchors = anchor_texts(row)
            anchor_state_map = forward_last_states(model, tokenizer, device, [text for _, text in anchors], selected_layers)
            prompt_last_idx = max((idx for idx, (a, b) in enumerate(offsets) if b > a), default=len(offsets) - 1)
            for layer_idx in selected_layers:
                if layer_idx not in prompt_states:
                    continue
                layer_states = prompt_states[layer_idx]
                prompt_last_vec = layer_states[prompt_last_idx]
                for spec in specs:
                    token_idx = int(spec["token_idx"])
                    if token_idx >= layer_states.shape[0]:
                        continue
                    trigger_vec = layer_states[token_idx]
                    base = {
                        "phase": PHASE,
                        "source_phase": SOURCE_PHASE,
                        "row_kind": "phase214_prompt_trigger_token_row",
                        "model": args.model,
                        "row_index": row_idx,
                        "trajectory_id": row.get("trajectory_id"),
                        "source_sample_id": row.get("source_sample_id"),
                        "relation": row.get("relation"),
                        "language_pair": row.get("language_pair"),
                        "pattern_id": row.get("pattern_id"),
                        "expected_output_pattern": row.get("expected_output_pattern"),
                        "pattern_match": row.get("pattern_match"),
                        "pattern_drift": row.get("pattern_drift"),
                        "failure_mode": row.get("failure_mode"),
                        "output_pattern": row.get("output_pattern"),
                        "answer_present": row.get("answer_present"),
                        "steps_generated": row.get("steps_generated"),
                        "layer_idx": int(layer_idx),
                        **spec,
                    }
                    token_rows.append(
                        {
                            **base,
                            **vector_stats(trigger_vec),
                            "cosine_to_prompt_last": cosine(trigger_vec, prompt_last_vec),
                            "l2_to_prompt_last": float(torch.linalg.vector_norm(trigger_vec - prompt_last_vec).item()),
                        }
                    )
                    for anchor_idx, (anchor_label, _anchor_text) in enumerate(anchors):
                        anchor_vec = anchor_state_map.get(str(anchor_idx), {}).get(int(layer_idx))
                        if anchor_vec is None:
                            continue
                        path_rows.append(
                            {
                                **base,
                                "row_kind": "phase214_trigger_to_anchor_path_row",
                                "anchor_label": anchor_label,
                                "cosine_trigger_to_anchor": cosine(trigger_vec, anchor_vec),
                                "l2_trigger_to_anchor": float(torch.linalg.vector_norm(trigger_vec - anchor_vec).item()),
                                "anchor_residual_norm": float(torch.linalg.vector_norm(anchor_vec).item()),
                            }
                        )
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

    token_summary_rows = summarize_token_rows(token_rows)
    path_summary_rows = summarize_path_rows(path_rows)
    payload = {
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "title": "Prompt Trigger Token Path Atlas",
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": args.model,
        "selected_layers": selected_layers,
        "selected_trajectory_rows": len(selected),
        "trigger_token_rows": len(token_rows),
        "path_rows": len(path_rows),
        "token_summary_rows": token_summary_rows,
        "path_summary_rows": path_summary_rows,
    }
    write_json(out_dir / f"phase214_{args.model}_summary.json", payload)
    write_jsonl(out_dir / f"phase214_{args.model}_trigger_token_rows.jsonl", token_rows)
    write_jsonl(out_dir / f"phase214_{args.model}_path_rows.jsonl", path_rows)
    write_jsonl(out_dir / f"phase214_{args.model}_token_summary_rows.jsonl", token_summary_rows)
    write_jsonl(out_dir / f"phase214_{args.model}_path_summary_rows.jsonl", path_summary_rows)
    print(
        json.dumps(
            {
                "phase": PHASE,
                "model": args.model,
                "status": "complete",
                "selected_trajectory_rows": len(selected),
                "trigger_token_rows": len(token_rows),
                "path_rows": len(path_rows),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return payload


def summarize_token_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = ["model", "pattern_id", "trigger_label", "layer_idx", "pattern_match"]
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[tuple(row.get(key) for key in keys)].append(row)
    out = []
    for key, items in buckets.items():
        row = {name: value for name, value in zip(keys, key)}
        row.update(
            {
                "rows": len(items),
                "failure_modes": dict(Counter(str(item.get("failure_mode")) for item in items).most_common()),
                "residual_norm_mean": mean([item.get("residual_norm") for item in items]),
                "cosine_to_prompt_last_mean": mean([item.get("cosine_to_prompt_last") for item in items]),
                "l2_to_prompt_last_mean": mean([item.get("l2_to_prompt_last") for item in items]),
            }
        )
        out.append(row)
    out.sort(key=lambda row: (str(row.get("model")), str(row.get("pattern_id")), str(row.get("trigger_label")), int(row.get("layer_idx")), str(row.get("pattern_match"))))
    return out


def summarize_path_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = ["model", "pattern_id", "trigger_label", "anchor_label", "layer_idx", "pattern_match"]
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[tuple(row.get(key) for key in keys)].append(row)
    out = []
    for key, items in buckets.items():
        row = {name: value for name, value in zip(keys, key)}
        row.update(
            {
                "rows": len(items),
                "failure_modes": dict(Counter(str(item.get("failure_mode")) for item in items).most_common()),
                "cosine_trigger_to_anchor_mean": mean([item.get("cosine_trigger_to_anchor") for item in items]),
                "l2_trigger_to_anchor_mean": mean([item.get("l2_trigger_to_anchor") for item in items]),
                "anchor_residual_norm_mean": mean([item.get("anchor_residual_norm") for item in items]),
            }
        )
        out.append(row)
    out.sort(
        key=lambda row: (
            str(row.get("model")),
            str(row.get("pattern_id")),
            str(row.get("trigger_label")),
            str(row.get("anchor_label")),
            int(row.get("layer_idx")),
            str(row.get("pattern_match")),
        )
    )
    return out


def success_drift_deltas(path_summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], dict[bool, dict[str, Any]]] = defaultdict(dict)
    keys = ["model", "pattern_id", "trigger_label", "anchor_label", "layer_idx"]
    for row in path_summary_rows:
        buckets[tuple(row.get(key) for key in keys)][bool(row.get("pattern_match"))] = row
    out = []
    for key, sides in buckets.items():
        if True not in sides or False not in sides:
            continue
        success = sides[True]
        drift = sides[False]
        row = {name: value for name, value in zip(keys, key)}
        row.update(
            {
                "success_rows": success.get("rows"),
                "drift_rows": drift.get("rows"),
                "cosine_success_mean": success.get("cosine_trigger_to_anchor_mean"),
                "cosine_drift_mean": drift.get("cosine_trigger_to_anchor_mean"),
                "cosine_success_minus_drift": finite(success.get("cosine_trigger_to_anchor_mean")) - finite(drift.get("cosine_trigger_to_anchor_mean")),
                "l2_success_mean": success.get("l2_trigger_to_anchor_mean"),
                "l2_drift_mean": drift.get("l2_trigger_to_anchor_mean"),
                "l2_success_minus_drift": finite(success.get("l2_trigger_to_anchor_mean")) - finite(drift.get("l2_trigger_to_anchor_mean")),
            }
        )
        out.append(row)
    out.sort(key=lambda row: abs(finite(row.get("cosine_success_minus_drift"))), reverse=True)
    return out


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    summaries = [read_json(out_dir / f"phase214_{model}_summary.json") for model in MODELS]
    summaries = [summary for summary in summaries if summary]
    token_summary_rows = []
    path_summary_rows = []
    for summary in summaries:
        token_summary_rows.extend(summary.get("token_summary_rows") or [])
        path_summary_rows.extend(summary.get("path_summary_rows") or [])
    delta_rows = success_drift_deltas(path_summary_rows)
    payload = {
        "schema_version": "phase214_prompt_trigger_token_path_atlas_v1",
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "status": "complete" if summaries else "missing",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "round": round_name,
        "models": [summary.get("model") for summary in summaries],
        "selected_trajectory_rows": sum(int(summary.get("selected_trajectory_rows") or 0) for summary in summaries),
        "trigger_token_rows": sum(int(summary.get("trigger_token_rows") or 0) for summary in summaries),
        "path_rows": sum(int(summary.get("path_rows") or 0) for summary in summaries),
        "token_summary_rows": token_summary_rows,
        "path_summary_rows": path_summary_rows,
        "success_drift_delta_rows": delta_rows,
        "top_success_drift_deltas": delta_rows[:40],
    }
    write_json(out_dir / "phase214_cross_model_summary.json", payload)
    write_jsonl(out_dir / "phase214_success_drift_delta_rows.jsonl", delta_rows)
    write_summary_md(out_dir / "phase214_cross_model_summary.md", payload)
    return payload


def write_summary_md(path: Path, payload: dict[str, Any]) -> None:
    lines = ["# Phase 214 prompt trigger token path atlas", ""]
    lines.append(f"Selected trajectory rows: {payload.get('selected_trajectory_rows')}")
    lines.append(f"Trigger token rows: {payload.get('trigger_token_rows')}")
    lines.append(f"Path rows: {payload.get('path_rows')}")
    lines.append(f"Success/drift delta rows: {len(payload.get('success_drift_delta_rows') or [])}")
    lines.append("")
    lines.append("| model | pattern | trigger | anchor | layer | success rows | drift rows | cosine delta | l2 delta |")
    lines.append("| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for row in payload.get("top_success_drift_deltas") or []:
        lines.append(
            f"| {row.get('model')} | {row.get('pattern_id')} | {row.get('trigger_label')} | {row.get('anchor_label')} | "
            f"{row.get('layer_idx')} | {row.get('success_rows')} | {row.get('drift_rows')} | "
            f"{finite(row.get('cosine_success_minus_drift')):.6f} | {finite(row.get('l2_success_minus_drift')):.6f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="prompt_trigger_token_path_atlas")
    parser.add_argument("--phase210-round", default="minimal_pattern_transition_atlas")
    parser.add_argument("--max-rows-per-pattern", type=int, default=30)
    parser.add_argument("--scan-layers", default="")
    parser.add_argument("--attn-implementations", default="flash_attention_2,sdpa")
    parser.add_argument("--summarize", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.summarize:
        payload = summarize_round(args.round_name)
        print(
            json.dumps(
                {
                    "phase": PHASE,
                    "status": payload.get("status"),
                    "models": payload.get("models"),
                    "selected_trajectory_rows": payload.get("selected_trajectory_rows"),
                    "trigger_token_rows": payload.get("trigger_token_rows"),
                    "path_rows": payload.get("path_rows"),
                    "delta_rows": len(payload.get("success_drift_delta_rows") or []),
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
        return
    if not args.model:
        raise SystemExit("--model is required unless --summarize is used")
    eval_model(args)


if __name__ == "__main__":
    main()
