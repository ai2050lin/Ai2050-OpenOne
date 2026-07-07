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

import phase214_prompt_trigger_token_path_atlas as p214  # noqa: E402
from phase735_source_restricted_writer_validation import load_model_bf16_eager  # noqa: E402
from model_utils import get_layers, release_model  # noqa: E402


PHASE = 215
SOURCE_PHASE = 214
MODELS = ["qwen3", "glm4", "deepseek7b"]
INPUT_ROOT = Path("tests/result/phase210_minimal_pattern_transition_atlas")
RESULT_ROOT = Path("tests/result/phase215_prompt_attention_route_atlas")
ANCHOR_STEPS = [0, 1, 3, 6]

FOCUS_PATTERNS = {
    "qwen3": ["answer_repeat", "answer_explain", "answer_list"],
    "glm4": ["answer_target_seeded", "answer_explain", "answer_repeat"],
    "deepseek7b": ["answer_explain", "answer_list"],
}


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


def load_rows(model_name: str, phase210_round: str) -> list[dict[str, Any]]:
    path = INPUT_ROOT / phase210_round / f"phase210_{model_name}_trajectory_rows.jsonl"
    return list(p214.iter_jsonl(path) or [])


def select_focus_rows(rows: list[dict[str, Any]], model_name: str, max_per_pattern_group: int) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    focus = set(FOCUS_PATTERNS[model_name])
    for row in rows:
        pattern = str(row.get("pattern_id") or "")
        if pattern not in focus:
            continue
        group = "match" if row.get("pattern_match") else "drift"
        buckets[(pattern, group)].append(row)
    selected: list[dict[str, Any]] = []
    for key in sorted(buckets):
        selected.extend(buckets[key][: int(max_per_pattern_group)])
    selected.sort(key=lambda row: (str(row.get("pattern_id")), not bool(row.get("pattern_match")), str(row.get("trajectory_id"))))
    return selected


def scan_layers_for_model(model, model_name: str, layer_text: str) -> list[int]:
    total = len(get_layers(model))
    if layer_text:
        return sorted({int(x.strip()) for x in layer_text.split(",") if x.strip() and 0 <= int(x.strip()) < total})
    preferred = {
        "qwen3": [3, 6, 11, 24, 29, 31, 32, 33],
        "glm4": [3, 7, 12, 20, 27, 28, 29, 30],
        "deepseek7b": [2, 5, 14, 22, 23, 24, 25, 26, 27],
    }
    return [idx for idx in preferred[model_name] if 0 <= idx < total]


def all_occurrences(text: str, needle: str) -> list[tuple[int, int]]:
    if not needle:
        return []
    out = []
    low = text.lower()
    target = needle.lower()
    start = 0
    while True:
        idx = low.find(target, start)
        if idx < 0:
            break
        out.append((idx, idx + len(needle)))
        start = idx + max(1, len(needle))
    return out


def positions_for_spans(offsets: list[tuple[int, int]], spans: list[tuple[int, int]]) -> list[int]:
    pos = []
    for idx, (a, b) in enumerate(offsets):
        if b <= a:
            continue
        for start, end in spans:
            if b > start and a < end:
                pos.append(idx)
                break
    return sorted(set(pos))


def prompt_source_groups(row: dict[str, Any], prompt: str, offsets: list[tuple[int, int]], tokens: list[str]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    specs = p214.token_specs_for_prompt(str(row.get("pattern_id") or ""), prompt, offsets, tokens)
    all_trigger_positions: set[int] = set()
    for spec in specs:
        positions = list(range(int(spec["token_start"]), int(spec["token_end"]) + 1))
        groups[f"trigger:{spec['trigger_label']}"] = positions
        all_trigger_positions.update(positions)
    if all_trigger_positions:
        groups["trigger:any"] = sorted(all_trigger_positions)
    for label, field in [("object", "object"), ("target_label", "target_label"), ("relation", "relation")]:
        text = str(row.get(field) or "").strip()
        positions = positions_for_spans(offsets, all_occurrences(prompt, text))
        if positions:
            groups[label] = positions
    qmark = prompt.find("?")
    answer_idx = prompt.lower().find("answer:")
    if qmark >= 0:
        groups["question_prefix"] = positions_for_spans(offsets, [(0, qmark + 1)])
    if answer_idx >= 0:
        groups["instruction_to_answer"] = positions_for_spans(offsets, [(qmark + 1 if qmark >= 0 else 0, answer_idx)])
        groups["answer_slot"] = positions_for_spans(offsets, [(answer_idx, len(prompt))])
    groups["prompt_all"] = [idx for idx, (a, b) in enumerate(offsets) if b > a]
    return groups


def anchor_text(row: dict[str, Any], step: int) -> tuple[str, str]:
    prompt = str(row.get("prompt") or "")
    emitted = [str(tok) for tok in row.get("emitted_tokens") or []]
    if step <= 0:
        return "prompt_last", prompt
    used = emitted[: min(step, len(emitted))]
    return f"gen_after_step_{step}", prompt + "".join(used)


def run_attention(model, tokenizer, device: torch.device, text: str):
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False, return_offsets_mapping=True)
    offsets = [(int(a), int(b)) for a, b in encoded["offset_mapping"][0].tolist()]
    input_ids = encoded["input_ids"].to(device)
    with torch.inference_mode():
        out = model(input_ids=input_ids, use_cache=False, output_attentions=True, return_dict=True)
    if out.attentions is None:
        raise RuntimeError("model returned no attentions; eager attention is required")
    ids = input_ids[0].detach().cpu().tolist()
    tokens = [tokenizer.decode([int(tok)]) for tok in ids]
    del encoded, input_ids
    return ids, offsets, tokens, out.attentions


def mass(row: torch.Tensor, idxs: list[int]) -> float:
    valid = [idx for idx in idxs if 0 <= idx < row.numel()]
    if not valid:
        return 0.0
    return float(row[valid].sum().detach().float().cpu().item())


def make_attention_rows(model, tokenizer, device, model_name: str, rows: list[dict[str, Any]], layers: list[int]) -> list[dict[str, Any]]:
    out_rows: list[dict[str, Any]] = []
    for row_idx, row in enumerate(rows):
        prompt = str(row.get("prompt") or "")
        prompt_ids, prompt_offsets, prompt_tokens = p214.encode_with_offsets(tokenizer, prompt)
        groups = prompt_source_groups(row, prompt, prompt_offsets, prompt_tokens)
        del prompt_ids
        if not groups.get("trigger:any") and not groups.get("answer_slot"):
            continue
        max_generated = len(row.get("emitted_tokens") or [])
        for step in ANCHOR_STEPS:
            if step > max_generated:
                continue
            anchor_label, text = anchor_text(row, step)
            try:
                ids, _offsets, tokens, attentions = run_attention(model, tokenizer, device, text)
            except Exception as exc:
                log(f"skip attention {row.get('trajectory_id')} {anchor_label}: {exc}")
                continue
            query_pos = len(ids) - 1
            for layer_idx in layers:
                if layer_idx >= len(attentions):
                    continue
                attn = attentions[layer_idx].detach()
                if attn.ndim != 4:
                    continue
                n_heads = int(attn.shape[1])
                for head_idx in range(n_heads):
                    qrow = attn[0, head_idx, query_pos, :].detach()
                    top_pos = int(torch.argmax(qrow).detach().cpu().item())
                    base = {
                        "phase": PHASE,
                        "source_phase": SOURCE_PHASE,
                        "row_kind": "phase215_attention_route_row",
                        "model": model_name,
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
                        "anchor_label": anchor_label,
                        "layer_idx": int(layer_idx),
                        "head_idx": int(head_idx),
                        "query_pos": int(query_pos),
                        "query_token": tokens[query_pos] if query_pos < len(tokens) else "",
                        "top_attn_pos": top_pos,
                        "top_attn_token": tokens[top_pos] if top_pos < len(tokens) else "",
                        "top_attn_mass": float(qrow[top_pos].detach().float().cpu().item()),
                    }
                    for group_name, positions in groups.items():
                        base[f"mass_{group_name}"] = mass(qrow, positions)
                    out_rows.append(base)
            del attentions
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return out_rows


def summarize_attention(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = ["model", "pattern_id", "anchor_label", "layer_idx", "head_idx", "pattern_match"]
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[tuple(row.get(key) for key in keys)].append(row)
    out = []
    mass_keys = sorted({key for row in rows for key in row if key.startswith("mass_")})
    for key, items in buckets.items():
        rec = {name: value for name, value in zip(keys, key)}
        rec["rows"] = len(items)
        rec["failure_modes"] = dict(Counter(str(item.get("failure_mode")) for item in items).most_common())
        rec["top_attn_tokens"] = dict(Counter(str(item.get("top_attn_token")) for item in items).most_common(8))
        for mass_key in mass_keys:
            rec[f"{mass_key}_mean"] = mean([item.get(mass_key) for item in items])
        out.append(rec)
    out.sort(key=lambda row: (str(row.get("model")), str(row.get("pattern_id")), str(row.get("anchor_label")), int(row.get("layer_idx")), int(row.get("head_idx")), str(row.get("pattern_match"))))
    return out


def route_deltas(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = ["model", "pattern_id", "anchor_label", "layer_idx", "head_idx"]
    buckets: dict[tuple[Any, ...], dict[bool, dict[str, Any]]] = defaultdict(dict)
    for row in summary_rows:
        buckets[tuple(row.get(key) for key in keys)][bool(row.get("pattern_match"))] = row
    out = []
    for key, sides in buckets.items():
        if True not in sides or False not in sides:
            continue
        success = sides[True]
        drift = sides[False]
        mass_keys = sorted({k for k in success if k.startswith("mass_") and k.endswith("_mean")})
        rec = {name: value for name, value in zip(keys, key)}
        rec["success_rows"] = success.get("rows")
        rec["drift_rows"] = drift.get("rows")
        for mass_key in mass_keys:
            short = mass_key[len("mass_") : -len("_mean")]
            s = finite(success.get(mass_key))
            d = finite(drift.get(mass_key))
            rec[f"{short}_success_mean"] = s
            rec[f"{short}_drift_mean"] = d
            rec[f"{short}_success_minus_drift"] = s - d
        rec["max_abs_delta"] = max((abs(finite(v)) for k, v in rec.items() if k.endswith("_success_minus_drift")), default=0.0)
        out.append(rec)
    out.sort(key=lambda row: finite(row.get("max_abs_delta")), reverse=True)
    return out


def eval_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    model = None
    tokenizer = None
    attention_rows: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    selected_layers: list[int] = []
    try:
        model, tokenizer, device, _attn_impl = load_model_bf16_eager(args.model)
        rows = load_rows(args.model, args.phase210_round)
        selected = select_focus_rows(rows, args.model, int(args.max_per_pattern_group))
        selected_layers = scan_layers_for_model(model, args.model, args.scan_layers)
        attention_rows = make_attention_rows(model, tokenizer, device, args.model, selected, selected_layers)
    finally:
        if model is not None:
            release_model(model)
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    summary_rows = summarize_attention(attention_rows)
    delta_rows = route_deltas(summary_rows)
    payload = {
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "title": "Prompt Attention Route Atlas",
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": args.model,
        "focus_patterns": FOCUS_PATTERNS[args.model],
        "selected_layers": selected_layers,
        "selected_trajectory_rows": len(selected),
        "attention_route_rows": len(attention_rows),
        "summary_rows": len(summary_rows),
        "route_delta_rows": len(delta_rows),
        "top_route_deltas": delta_rows[:40],
    }
    write_json(out_dir / f"phase215_{args.model}_summary.json", payload)
    write_jsonl(out_dir / f"phase215_{args.model}_attention_route_rows.jsonl", attention_rows)
    write_jsonl(out_dir / f"phase215_{args.model}_attention_summary_rows.jsonl", summary_rows)
    write_jsonl(out_dir / f"phase215_{args.model}_route_delta_rows.jsonl", delta_rows)
    print(json.dumps({"phase": PHASE, "model": args.model, "status": "complete", "selected": len(selected), "attention_rows": len(attention_rows), "delta_rows": len(delta_rows)}, ensure_ascii=False, indent=2), flush=True)
    return payload


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    summaries = [read_json(out_dir / f"phase215_{model}_summary.json") for model in MODELS]
    summaries = [summary for summary in summaries if summary]
    delta_rows: list[dict[str, Any]] = []
    for model in MODELS:
        delta_rows.extend(list(p214.iter_jsonl(out_dir / f"phase215_{model}_route_delta_rows.jsonl") or []))
    payload = {
        "schema_version": "phase215_prompt_attention_route_atlas_v1",
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "status": "complete" if summaries else "missing",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "round": round_name,
        "models": [summary.get("model") for summary in summaries],
        "selected_trajectory_rows": sum(int(summary.get("selected_trajectory_rows") or 0) for summary in summaries),
        "attention_route_rows": sum(int(summary.get("attention_route_rows") or 0) for summary in summaries),
        "summary_rows": sum(int(summary.get("summary_rows") or 0) for summary in summaries),
        "route_delta_rows": len(delta_rows),
        "top_route_deltas": sorted(delta_rows, key=lambda row: finite(row.get("max_abs_delta")), reverse=True)[:60],
    }
    write_json(out_dir / "phase215_cross_model_summary.json", payload)
    write_summary_md(out_dir / "phase215_cross_model_summary.md", payload)
    return payload


def write_summary_md(path: Path, payload: dict[str, Any]) -> None:
    lines = ["# Phase 215 prompt attention route atlas", ""]
    lines.append(f"Selected trajectory rows: {payload.get('selected_trajectory_rows')}")
    lines.append(f"Attention route rows: {payload.get('attention_route_rows')}")
    lines.append(f"Route delta rows: {payload.get('route_delta_rows')}")
    lines.append("")
    lines.append("| model | pattern | anchor | layer | head | success | drift | max delta | trigger:any delta | answer_slot delta | object delta | target delta |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in payload.get("top_route_deltas") or []:
        lines.append(
            f"| {row.get('model')} | {row.get('pattern_id')} | {row.get('anchor_label')} | {row.get('layer_idx')} | {row.get('head_idx')} | "
            f"{row.get('success_rows')} | {row.get('drift_rows')} | {finite(row.get('max_abs_delta')):.6f} | "
            f"{finite(row.get('trigger:any_success_minus_drift')):.6f} | {finite(row.get('answer_slot_success_minus_drift')):.6f} | "
            f"{finite(row.get('object_success_minus_drift')):.6f} | {finite(row.get('target_label_success_minus_drift')):.6f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="prompt_attention_route_atlas")
    parser.add_argument("--phase210-round", default="minimal_pattern_transition_atlas")
    parser.add_argument("--max-per-pattern-group", type=int, default=8)
    parser.add_argument("--scan-layers", default="")
    parser.add_argument("--summarize", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.summarize:
        payload = summarize_round(args.round_name)
        print(json.dumps({"phase": PHASE, "status": payload.get("status"), "models": payload.get("models"), "selected": payload.get("selected_trajectory_rows"), "attention_rows": payload.get("attention_route_rows"), "delta_rows": payload.get("route_delta_rows")}, ensure_ascii=False, indent=2), flush=True)
        return
    if not args.model:
        raise SystemExit("--model is required unless --summarize is used")
    eval_model(args)


if __name__ == "__main__":
    main()
