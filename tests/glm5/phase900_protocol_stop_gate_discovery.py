#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
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

import phase846_geometry_boundary_equation_fitting as p846  # noqa: E402
import phase856_identity_class_overlap_cross_domain_rollout_audit as p856  # noqa: E402
import phase862_negative_blocker_sign_mechanism_audit as p862  # noqa: E402
import phase885_stable_boundary_minimality_cross_model_audit as p885  # noqa: E402
import phase893_attention_head_complementarity_holdout_probe as p893  # noqa: E402
import phase899_domain_axis_rollout_protocol_audit as p899  # noqa: E402


PHASE = 900
MODELS = ["qwen3", "glm4", "deepseek7b"]
RESULT_ROOT = Path("tests/result/phase900_protocol_stop_gate_discovery")
PHASE899_ROOT = Path("tests/result/phase899_domain_axis_rollout_protocol_audit")
PHASE899_ROUND = "domain_axis_rollout_protocol_audit"

HISTORICAL_HEAD_SETS = {
    "qwen3": ["L31H19", "L31H26", "L31H30", "L31H12", "L31H17", "L31H19+L31H26+L31H30+L31H12+L31H17"],
    "deepseek7b": ["L26H3", "L26H7", "L26H11", "L26H14", "L26H3+L26H7+L26H11+L26H14"],
    "glm4": [],
}


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def parse_head_key(text: str) -> tuple[int, int] | None:
    match = re.fullmatch(r"L(\d+)H(\d+)", str(text).strip())
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def head_set_key(heads: list[tuple[int, int]]) -> str:
    return "+".join(f"L{layer}H{head}" for layer, head in heads) if heads else "none"


def parse_head_set(text: str) -> list[tuple[int, int]]:
    heads = []
    for part in str(text or "").split("+"):
        head = parse_head_key(part)
        if head is not None:
            heads.append(head)
    return heads


def parse_gears(subset_key: str) -> list[dict[str, Any]]:
    gears = []
    for part in str(subset_key or "").split("+"):
        if part.startswith("L") and "C" in part:
            gear = p862.parse_gear_key(part)
            if gear is not None:
                gears.append(gear)
    return gears


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z]+", str(text or "")))


def drift_taxonomy(text: str) -> dict[str, Any]:
    raw = str(text or "").lower()
    norm = p856.normalize(text)
    field = any(marker in raw for marker in ["answer:", "category:", "subcategory:", "subclass:", "item:", "class:"])
    explanation = any(
        marker in raw or marker in norm
        for marker in ["okay, so", "i need", "please", "the answer is", "category is", "the best", "concise category"]
    )
    list_like = raw.count(",") >= 2 or " or " in raw or "1." in raw or "2." in raw
    long_phrase = word_count(raw) > 5
    return {
        "field_drift": field,
        "explanation_drift": explanation,
        "list_drift": list_like,
        "long_phrase_drift": long_phrase,
        "drift_type_count": int(field) + int(explanation) + int(list_like) + int(long_phrase),
    }


def rollout_flags(text: str, case: dict[str, Any]) -> dict[str, Any]:
    rollout = p856.classify_rollout(text, case)
    taxonomy = drift_taxonomy(text)
    drift = bool(p899.protocol_drift(text) or any(taxonomy[key] for key in taxonomy if key.endswith("_drift")))
    clear = bool(rollout.get("rollout_clear_answer_class"))
    return {
        **rollout,
        **taxonomy,
        "protocol_drift": drift,
        "rollout_clear_answer_no_protocol": clear and not drift and not rollout.get("rollout_object_echo"),
        "rollout_class_no_echo_no_protocol": bool(rollout.get("rollout_answer_class"))
        and not drift
        and not rollout.get("rollout_object_echo"),
        "rollout_bad_transition": bool(rollout.get("rollout_object_echo") or rollout.get("rollout_other_or_format") or drift),
    }


def selected_phase899_rows(model_name: str, args: argparse.Namespace) -> list[dict[str, Any]]:
    path = PHASE899_ROOT / args.phase899_round / f"phase899_{model_name}_rollout_rows.jsonl"
    rows = [
        row
        for row in read_jsonl(path)
        if row.get("is_source_candidate") and row.get("rollout_answer_class") and row.get("protocol_drift")
    ]
    rows.sort(
        key=lambda row: (
            str(row.get("eval_domain")),
            str(row.get("source_subset_key")),
            str(row.get("case_id")),
            str(row.get("prompt_variant")),
            str(row.get("edit_mode")),
        )
    )
    max_rows = int(args.max_rows_per_model)
    return rows[:max_rows] if max_rows > 0 else rows


def same_layer_head_sets(model, source_gears: list[dict[str, Any]], max_heads: int) -> list[str]:
    out = []
    seen_layers = []
    for gear in source_gears:
        layer_idx = int(gear["layer_idx"])
        if layer_idx not in seen_layers:
            seen_layers.append(layer_idx)
    for layer_idx in seen_layers[:2]:
        n_heads = p893.attention_head_count(model, layer_idx)
        heads = [f"L{layer_idx}H{head}" for head in range(min(int(max_heads), int(n_heads)))]
        out.extend(heads)
        if len(heads) > 1:
            out.append("+".join(heads))
    return out


def control_specs(model, model_name: str, source_gears: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    specs = [
        {
            "control_type": "source_step0_only",
            "control_label": "baseline",
            "gear_steps": [0],
            "after_mode": None,
            "head_steps": [],
            "heads": [],
        },
        {
            "control_type": "source_repeat_step1",
            "control_label": "semantic_axis_repeated_to_step1",
            "gear_steps": [0, 1],
            "after_mode": None,
            "head_steps": [],
            "heads": [],
        },
        {
            "control_type": "source_repeat_step1_2",
            "control_label": "semantic_axis_repeated_to_step1_2",
            "gear_steps": [0, 1, 2],
            "after_mode": None,
            "head_steps": [],
            "heads": [],
        },
        {
            "control_type": "source_after_zero_step1_2",
            "control_label": "semantic_axis_zero_after_answer",
            "gear_steps": [0, 1, 2],
            "after_mode": "zero",
            "head_steps": [],
            "heads": [],
        },
        {
            "control_type": "source_after_flip_step1_2",
            "control_label": "semantic_axis_flip_after_answer",
            "gear_steps": [0, 1, 2],
            "after_mode": "flip",
            "head_steps": [],
            "heads": [],
        },
    ]
    head_keys = list(HISTORICAL_HEAD_SETS.get(model_name, []))
    head_keys.extend(same_layer_head_sets(model, source_gears, int(args.max_same_layer_heads)))
    seen = set()
    for head_key in head_keys:
        heads = parse_head_set(head_key)
        if not heads:
            continue
        key = head_set_key(heads)
        if key in seen:
            continue
        seen.add(key)
        specs.append(
            {
                "control_type": "head_zero_step1",
                "control_label": f"head_zero_step1::{key}",
                "gear_steps": [0],
                "after_mode": None,
                "head_steps": [1],
                "heads": heads,
            }
        )
        specs.append(
            {
                "control_type": "head_zero_step1_2",
                "control_label": f"head_zero_step1_2::{key}",
                "gear_steps": [0],
                "after_mode": None,
                "head_steps": [1, 2],
                "heads": heads,
            }
        )
    return specs


def install_heads(model, heads: list[tuple[int, int]]) -> list[Any]:
    handles = []
    for layer_idx, head_idx in heads:
        handles.extend(p893.install_attention_head_zero(model, int(layer_idx), int(head_idx)))
    return handles


def greedy_with_control(
    model,
    tokenizer,
    device: torch.device,
    prompt_ids: list[int],
    gears: list[dict[str, Any]],
    source_mode: str,
    spec: dict[str, Any],
    max_new_tokens: int,
    scale_up_factor: float,
) -> tuple[str, list[int]]:
    current = [int(x) for x in prompt_ids]
    new_ids: list[int] = []
    eos_id = tokenizer.eos_token_id
    gear_steps = {int(step) for step in spec.get("gear_steps") or []}
    head_steps = {int(step) for step in spec.get("head_steps") or []}
    after_mode = spec.get("after_mode")
    heads = list(spec.get("heads") or [])
    for step in range(int(max_new_tokens)):
        input_ids = torch.tensor([current], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        handles: list[Any] = []
        try:
            if step in gear_steps and gears:
                mode = str(source_mode) if step == 0 or after_mode is None else str(after_mode)
                if mode != "original":
                    handles.extend(p862.install_scaled_gear_edit(model, gears, mode, scale_up_factor))
            if step in head_steps and heads:
                handles.extend(install_heads(model, heads))
            with torch.no_grad():
                logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits[0, -1].detach().float()
        finally:
            for handle in handles:
                handle.remove()
        next_id = int(torch.argmax(logits).item())
        new_ids.append(next_id)
        current.append(next_id)
        if eos_id is not None and next_id == int(eos_id):
            break
    return tokenizer.decode(new_ids, skip_special_tokens=True), new_ids


def make_row(
    model_name: str,
    source_row: dict[str, Any],
    case: dict[str, Any],
    spec: dict[str, Any],
    text: str,
    ids: list[int],
    baseline_flags: dict[str, Any] | None,
) -> dict[str, Any]:
    flags = rollout_flags(text, case)
    drift_count = int(flags.get("drift_type_count") or 0)
    base_count = None if baseline_flags is None else int(baseline_flags.get("drift_type_count") or 0)
    return {
        "phase": PHASE,
        "row_kind": "phase900_protocol_stop_control_row",
        "model": model_name,
        "source_key": source_row.get("source_key"),
        "source_subset_key": source_row.get("source_subset_key"),
        "eval_domain": source_row.get("eval_domain"),
        "case_id": source_row.get("case_id"),
        "case_split": source_row.get("case_split"),
        "object": source_row.get("object"),
        "prompt_variant": source_row.get("prompt_variant"),
        "edit_mode": source_row.get("edit_mode"),
        "control_type": spec.get("control_type"),
        "control_label": spec.get("control_label"),
        "head_set": head_set_key(list(spec.get("heads") or [])),
        "head_steps": spec.get("head_steps"),
        "gear_steps": spec.get("gear_steps"),
        "after_mode": spec.get("after_mode"),
        "generated_text": text,
        "generated_ids": ids,
        "baseline_drift_type_count": base_count,
        "drift_type_count_delta_vs_baseline": None if base_count is None else drift_count - base_count,
        **flags,
    }


def summarize_relation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "answer_class": sum(1 for row in rows if row.get("rollout_answer_class")),
        "clean_answer_no_protocol": sum(1 for row in rows if row.get("rollout_clear_answer_no_protocol")),
        "class_no_echo_no_protocol": sum(1 for row in rows if row.get("rollout_class_no_echo_no_protocol")),
        "protocol_drift": sum(1 for row in rows if row.get("protocol_drift")),
        "object_echo": sum(1 for row in rows if row.get("rollout_object_echo")),
        "other_or_format": sum(1 for row in rows if row.get("rollout_other_or_format")),
        "field_drift": sum(1 for row in rows if row.get("field_drift")),
        "explanation_drift": sum(1 for row in rows if row.get("explanation_drift")),
        "list_drift": sum(1 for row in rows if row.get("list_drift")),
        "long_phrase_drift": sum(1 for row in rows if row.get("long_phrase_drift")),
        "drift_type_count_reduced": sum(
            1
            for row in rows
            if row.get("drift_type_count_delta_vs_baseline") is not None and int(row.get("drift_type_count_delta_vs_baseline")) < 0
        ),
        "labels": dict(sorted(Counter(str(row.get("rollout_label")) for row in rows).items())),
    }


def summarize_model(model_name: str, rows: list[dict[str, Any]], selected_rows: list[dict[str, Any]], attn_impl: str | None) -> dict[str, Any]:
    by_control: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_control[str(row.get("control_label"))].append(row)
    control_summaries = []
    for label, vals in by_control.items():
        rel = summarize_relation(vals)
        rel.update({"control_label": label, "control_type": vals[0].get("control_type"), "head_set": vals[0].get("head_set")})
        control_summaries.append(rel)
    control_summaries.sort(
        key=lambda row: (
            row.get("clean_answer_no_protocol") or 0,
            row.get("drift_type_count_reduced") or 0,
            row.get("answer_class") or 0,
        ),
        reverse=True,
    )
    baseline = summarize_relation([row for row in rows if row.get("control_type") == "source_step0_only"])
    non_base = [row for row in rows if row.get("control_type") != "source_step0_only"]
    overall = {
        "selected_answer_drift_rows": len(selected_rows),
        "control_rows": len(rows),
        "baseline": baseline,
        "non_baseline": summarize_relation(non_base),
        "best_clean_control": control_summaries[0] if control_summaries else {},
    }
    if overall["non_baseline"]["clean_answer_no_protocol"] > 0:
        evidence_label = "limited_protocol_control_found"
    elif overall["non_baseline"]["drift_type_count_reduced"] > 0:
        evidence_label = "weak_protocol_drift_reduction_without_clean_closure"
    else:
        evidence_label = "no_simple_protocol_stop_gate_found"
    return {
        "phase": PHASE,
        "title": "Protocol Stop Gate Discovery",
        "model": model_name,
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "overall": overall,
        "control_summaries": control_summaries,
        "evidence_label": evidence_label,
        "boundary": (
            "Phase900 is a limited step-1/2 control pre-screen over Phase899 answer-class-but-drift rows. "
            "It does not exhaust all channels or heads."
        ),
    }


def eval_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    selected_rows = selected_phase899_rows(args.model, args)
    if args.dry_run or not selected_rows:
        payload = {
            "phase": PHASE,
            "model": args.model,
            "status": "dry_run" if selected_rows else "no_rows",
            "selected_rows": selected_rows,
        }
        p846.write_json(out_dir / f"phase900_{args.model}_summary.json", payload)
        p846.write_jsonl(out_dir / f"phase900_{args.model}_rows.jsonl", [])
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return payload

    case_map = {str(case.get("case_id")): case for case in p885.extended_cases()}
    model = None
    tokenizer = None
    rows: list[dict[str, Any]] = []
    attn_impl = None
    try:
        model, tokenizer, device, attn_impl = p862.p844.p828.p796.load_model_bf16_prefer_flash(
            args.model, args.attn_implementations
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        prompt_cache: dict[tuple[str, str], list[int]] = {}
        for idx, source_row in enumerate(selected_rows, 1):
            case = case_map.get(str(source_row.get("case_id")))
            if not case:
                continue
            prompt_key = (str(source_row.get("case_id")), str(source_row.get("prompt_variant")))
            if prompt_key not in prompt_cache:
                prompt = p885.prompt_for_case(case, str(source_row.get("prompt_variant")))
                prompt_cache[prompt_key] = p862.p844.encode_prompt(tokenizer, prompt)
            prompt_ids = prompt_cache[prompt_key]
            gears = parse_gears(str(source_row.get("source_subset_key")))
            specs = control_specs(model, args.model, gears, args)
            baseline_flags = None
            for spec in specs:
                text, ids = greedy_with_control(
                    model,
                    tokenizer,
                    device,
                    prompt_ids,
                    gears,
                    str(source_row.get("edit_mode")),
                    spec,
                    int(args.max_new_tokens),
                    float(args.scale_up_factor),
                )
                row = make_row(args.model, source_row, case, spec, text, ids, baseline_flags)
                if spec.get("control_type") == "source_step0_only":
                    baseline_flags = row
                    row["baseline_drift_type_count"] = row.get("drift_type_count")
                    row["drift_type_count_delta_vs_baseline"] = 0
                rows.append(row)
            if idx % max(1, int(args.log_every)) == 0 or idx == len(selected_rows):
                log(f"{args.model}/{args.round_name}: row={idx}/{len(selected_rows)} control_rows={len(rows)}")
    finally:
        if model is not None:
            p862.p844.p828.release_model(model)
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    payload = summarize_model(args.model, rows, selected_rows, attn_impl)
    p846.write_json(out_dir / f"phase900_{args.model}_summary.json", payload)
    p846.write_jsonl(out_dir / f"phase900_{args.model}_rows.jsonl", rows)
    print(json.dumps({"phase": PHASE, "model": args.model, "overall": payload["overall"]}, ensure_ascii=False, indent=2), flush=True)
    return payload


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Phase 900 protocol stop gate discovery",
        "",
        "## Overall",
        "",
        f"- models: {', '.join(payload.get('models') or [])}",
    ]
    for key, value in (payload.get("overall_scalar") or {}).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Control summaries", ""])
    lines.append(
        "| model | control | type | head set | rows | answer | clean | drift | reduced | field | explanation | list | long | labels |"
    )
    lines.append("| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for row in payload.get("control_summaries") or []:
        lines.append(
            "| {model} | {control_label} | {control_type} | {head_set} | {rows} | {answer_class} | "
            "{clean_answer_no_protocol} | {protocol_drift} | {drift_type_count_reduced} | {field_drift} | "
            "{explanation_drift} | {list_drift} | {long_phrase_drift} | {labels} |".format(**row)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    summaries = []
    for model_name in MODELS:
        path = out_dir / f"phase900_{model_name}_summary.json"
        if path.exists():
            summaries.append(read_json(path))
    scalar = Counter()
    controls = []
    evidence = Counter()
    for summary in summaries:
        evidence[str(summary.get("evidence_label"))] += 1
        overall = summary.get("overall") or {}
        scalar["selected_answer_drift_rows"] += int(overall.get("selected_answer_drift_rows") or 0)
        scalar["control_rows"] += int(overall.get("control_rows") or 0)
        for row in summary.get("control_summaries") or []:
            row = dict(row)
            row["model"] = summary.get("model")
            controls.append(row)
    controls.sort(
        key=lambda row: (
            row.get("clean_answer_no_protocol") or 0,
            row.get("drift_type_count_reduced") or 0,
            row.get("answer_class") or 0,
        ),
        reverse=True,
    )
    payload = {
        "phase": PHASE,
        "round": round_name,
        "status": "complete" if len(summaries) == len(MODELS) else "partial",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": [summary.get("model") for summary in summaries],
        "overall_scalar": {key: int(value) for key, value in sorted(scalar.items())},
        "control_summaries": controls,
        "evidence_label_counts": dict(sorted(evidence.items())),
    }
    p846.write_json(out_dir / "phase900_cross_model_summary.json", payload)
    write_markdown(out_dir / "phase900_cross_model_summary.md", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="protocol_stop_gate_discovery")
    parser.add_argument("--phase899-round", default=PHASE899_ROUND)
    parser.add_argument("--max-rows-per-model", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--scale-up-factor", type=float, default=2.0)
    parser.add_argument("--max-same-layer-heads", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=8)
    parser.add_argument("--attn-implementations", default="flash_attention_2,sdpa")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize-round", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.summarize_round:
        payload = summarize_round(args.round_name)
        print(json.dumps({"phase": PHASE, "status": payload["status"], "overall": payload["overall_scalar"]}, ensure_ascii=False, indent=2))
        return
    if not args.model:
        raise SystemExit("--model is required unless --summarize-round is set")
    eval_model(args)


if __name__ == "__main__":
    main()
