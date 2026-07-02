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

import phase844_geometry_route_natural_gear_set_search as p844  # noqa: E402
import phase846_geometry_boundary_equation_fitting as p846  # noqa: E402
import phase854_full_vocab_blocker_min_cut_validation as p854  # noqa: E402
from model_utils import get_layers  # noqa: E402


PHASE = 855
MODELS = p846.MODELS
RESULT_ROOT = Path("tests/result/phase855_answer_class_alias_rollout_closure_validation")
PHASE854_ROOT = Path("tests/result/phase854_full_vocab_blocker_min_cut_validation")


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def finite(value: Any, default: float = 0.0) -> float:
    return p846.finite(value, default)


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def parse_gear_key(text: str) -> tuple[int, int] | None:
    return p854.parse_gear_key(text)


def gear_from_key(text: str) -> dict[str, Any] | None:
    return p854.gear_from_key(text)


def gear_key(gear: dict[str, Any]) -> str:
    return p854.gear_key(gear)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return p846.read_jsonl(path)


def phase854_rows_path(round_name: str, model_name: str) -> Path:
    return PHASE854_ROOT / round_name / f"phase854_{model_name}_rows.jsonl"


def phase854_edges_path(round_name: str, model_name: str) -> Path:
    return PHASE854_ROOT / round_name / f"phase854_{model_name}_edge_rows.jsonl"


def source_row_key(row: dict[str, Any]) -> str:
    return str(row.get("source_row_key") or "")


def expanded_aliases_for_object(obj: str) -> list[str]:
    object_name = str(obj or "").strip().lower()
    base = [
        "geometric shape",
        "geometric shapes",
        "geometry",
        "geometric",
        "shape",
        "shapes",
    ]
    polygon_like = {"triangle", "square", "rectangle", "polygon"}
    quadrilateral_like = {"square", "rectangle"}
    if object_name in polygon_like:
        base.extend(["polygon", "polygons"])
    if object_name in quadrilateral_like:
        base.extend(["quadrilateral", "quadrilaterals", "quadrangle"])
    if object_name == "circle":
        base.extend(["round shape", "closed curve"])
    variants: list[str] = []
    for alias in base:
        if not alias:
            continue
        variants.extend([alias, alias.capitalize(), alias.title(), f" {alias}", f" {alias.capitalize()}", f" {alias.title()}"])
    out: list[str] = []
    seen: set[str] = set()
    for item in variants:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


def strict_aliases() -> list[str]:
    return ["geometric shape", " Geometric shape", "Geometric shape", " geometric shape"]


def token_sets(tokenizer, case: dict[str, Any]) -> dict[str, Any]:
    strict_ids = p854.token_variant_ids(tokenizer, strict_aliases())
    class_ids = p854.token_variant_ids(tokenizer, expanded_aliases_for_object(str(case.get("object") or "")))
    object_ids = p854.token_variant_ids(tokenizer, p854.object_aliases(str(case.get("object") or "")))
    return {
        "strict_target_ids": strict_ids,
        "strict_target_tokens": [p854.decode_token(tokenizer, token_id) for token_id in strict_ids],
        "class_target_ids": class_ids,
        "class_target_tokens": [p854.decode_token(tokenizer, token_id) for token_id in class_ids],
        "object_ids": object_ids,
        "object_tokens": [p854.decode_token(tokenizer, token_id) for token_id in object_ids],
    }


def clean_text(text: str) -> str:
    return p844.p828.p825.clean_generated(text)


def normalize_for_match(text: str) -> str:
    text = clean_text(text).strip()
    text = re.sub(r"^[\\s\\\"'`:\\-–—,.;()\\[\\]{}]+", "", text)
    text = re.sub(r"\\s+", " ", text)
    return text.lower()


def classify_rollout(text: str, case: dict[str, Any]) -> dict[str, Any]:
    cleaned = clean_text(text)
    norm = normalize_for_match(cleaned)
    obj = str(case.get("object") or "").strip().lower()
    aliases = [normalize_for_match(alias) for alias in expanded_aliases_for_object(obj)]
    strict = [normalize_for_match(alias) for alias in strict_aliases()]
    if not norm:
        label = "format_or_empty"
    elif obj and (norm == obj or norm.startswith(obj + " ")):
        label = "object_echo"
    elif any(norm.startswith(alias) for alias in strict if alias):
        label = "strict_canonical"
    elif any(norm.startswith(alias) for alias in aliases if alias):
        label = "answer_alias"
    elif re.fullmatch(r"[\\W_]+", norm):
        label = "format_or_empty"
    else:
        label = "other"
    return {
        "generated_clean": cleaned,
        "rollout_label": label,
        "rollout_answer_class": label in {"strict_canonical", "answer_alias"},
        "rollout_strict_canonical": label == "strict_canonical",
        "rollout_object_echo": label == "object_echo",
        "rollout_other_or_format": label in {"other", "format_or_empty"},
    }


def case_from_phase854(row: dict[str, Any]) -> dict[str, Any]:
    obj = str(row.get("object") or "")
    return {
        "case_id": str(row.get("case_id") or f"p855_{obj}_case"),
        "object": obj,
        "question": f"Which category best describes a {obj}?",
        "answer": str(row.get("target_answer") or "geometric shape"),
        "contrast_answer": "living thing",
        "distractors": ["hand tool", "public transport", "musical instrument", "warm color"],
        "synthetic_case": bool(row.get("synthetic_case")),
    }


def select_sources_fixed(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    full_rows = [
        row
        for row in read_jsonl(phase854_rows_path(args.source_round, args.model))
        if row.get("condition_type") == "full_combo"
    ]
    edge_rows = read_jsonl(phase854_edges_path(args.source_round, args.model))
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in full_rows:
        group = str(row.get("source_group") or "")
        if group == "strong_target_not_exact" and row.get("answer_class_closure"):
            buckets["strong_target_class_closed"].append(row)
        elif group == "strong_target_not_exact":
            buckets["strong_target_class_open"].append(row)
        elif group == "strong_non_target_or_unknown":
            buckets["strong_other"].append(row)
        else:
            buckets["additive_control"].append(row)

    def sort_key(row: dict[str, Any]) -> tuple[float, float, str, str]:
        return (
            finite(row.get("class_best_target_rank"), 999999.0),
            finite(row.get("class_blocker_count"), 999999.0),
            str(row.get("object")),
            str(row.get("prompt_variant")),
        )

    for rows in buckets.values():
        rows.sort(key=sort_key)
    selected: list[dict[str, Any]] = []
    caps = {
        "strong_target_class_closed": int(args.max_strong_target_class_closed),
        "strong_target_class_open": int(args.max_strong_target_class_open),
        "strong_other": int(args.max_strong_other),
        "additive_control": int(args.max_controls),
    }
    seen: set[str] = set()
    for name in ["strong_target_class_closed", "strong_target_class_open", "strong_other", "additive_control"]:
        for row in buckets[name][: caps[name]]:
            key = source_row_key(row)
            if key in seen:
                continue
            selected.append(row)
            seen.add(key)
    if int(args.max_sources) > 0:
        selected = selected[: int(args.max_sources)]
    return selected, edge_rows


def build_conditions(source: dict[str, Any], edge_rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    gears = [gear_from_key(key) for key in source.get("source_gear_keys") or []]
    gears = [gear for gear in gears if gear is not None]
    mode = str(source.get("source_edit_mode") or "zero")
    specs: list[dict[str, Any]] = [
        {"condition_type": "original", "candidate_key": None, "mode": "original", "gears": []},
        {"condition_type": "full_combo", "candidate_key": None, "mode": mode, "gears": gears},
    ]
    if not args.include_min_cut_conditions:
        return specs
    key = source_row_key(source)
    candidates = [
        row
        for row in edge_rows
        if str(row.get("source_row_key")) == key and row.get("label") == "necessary_blocker_reducer"
    ]
    by_key = {gear_key(gear): gear for gear in gears}
    added = 0
    for edge in candidates:
        candidate = str(edge.get("candidate_key") or "")
        if candidate not in by_key:
            continue
        remain = [gear for gear in gears if gear_key(gear) != candidate]
        if remain:
            specs.append({"condition_type": "without_necessary", "candidate_key": candidate, "mode": mode, "gears": remain})
            added += 1
        if added >= int(args.max_min_cut_conditions_per_source):
            break
    return specs


def eval_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    sources, edge_rows = select_sources_fixed(args)
    if args.dry_run:
        payload = {
            "phase": PHASE,
            "model": args.model,
            "round": args.round_name,
            "source_round": args.source_round,
            "sources": [
                {
                    "source_group": row.get("source_group"),
                    "case_id": row.get("case_id"),
                    "object": row.get("object"),
                    "prompt_variant": row.get("prompt_variant"),
                    "answer_class_closure": row.get("answer_class_closure"),
                    "class_best_target_token": row.get("class_best_target_token"),
                    "class_blocker_count": row.get("class_blocker_count"),
                }
                for row in sources
            ],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return payload

    model = None
    tokenizer = None
    rows: list[dict[str, Any]] = []
    try:
        model, tokenizer, device, attn_impl = p844.p828.p796.load_model_bf16_prefer_flash(args.model, args.attn_implementations)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        n_layers = len(get_layers(model))
        for idx, source in enumerate(sources, 1):
            case = case_from_phase854(source)
            prompt = str(source.get("prompt") or p844.prompt_for_case(case, str(source.get("prompt_variant") or "natural_question")))
            prompt_ids = p844.encode_prompt(tokenizer, prompt)
            tsets = token_sets(tokenizer, case)
            for spec in build_conditions(source, edge_rows, args):
                valid_gears = [
                    gear
                    for gear in spec["gears"]
                    if 0 <= int(gear["layer_idx"]) < n_layers and int(gear["channel_id"]) >= 0
                ]
                logits = p844.first_logits_with_gears(model, device, prompt_ids, valid_gears, str(spec["mode"]))
                blocker = p854.blocker_metrics(tokenizer, logits, tsets, int(args.topk_blockers))
                generated, token_ids = p844.greedy_with_gears(
                    model,
                    tokenizer,
                    device,
                    prompt_ids,
                    valid_gears,
                    str(spec["mode"]),
                    int(args.max_new_tokens),
                )
                rollout = classify_rollout(generated, case)
                rows.append(
                    {
                        "row_kind": "phase855_answer_class_alias_rollout_closure_validation",
                        "phase": PHASE,
                        "model": args.model,
                        "round": args.round_name,
                        "source_round": args.source_round,
                        "source_row_key": source_row_key(source),
                        "source_group": source.get("source_group"),
                        "source_answer_class_closure": source.get("answer_class_closure"),
                        "source_class_best_target_token": source.get("class_best_target_token"),
                        "source_class_blocker_count": source.get("class_blocker_count"),
                        "source_combo_key": source.get("source_combo_key"),
                        "source_edit_mode": source.get("source_edit_mode"),
                        "case_id": case["case_id"],
                        "object": case.get("object"),
                        "prompt_variant": source.get("prompt_variant"),
                        "prompt": prompt,
                        "condition_type": spec["condition_type"],
                        "candidate_key": spec["candidate_key"],
                        "edit_mode": spec["mode"],
                        "gear_count": len(valid_gears),
                        "gear_keys": [gear_key(gear) for gear in valid_gears],
                        "token_ids": token_ids,
                        "expanded_answer_aliases": expanded_aliases_for_object(str(case.get("object") or "")),
                        **tsets,
                        **blocker,
                        **rollout,
                    }
                )
            if idx % max(1, int(args.log_every)) == 0 or idx == len(sources):
                log(f"{args.model}/{args.round_name}: rollout source rows {idx}/{len(sources)} emitted_rows={len(rows)}")
    finally:
        if model is not None:
            p844.p828.release_model(model)
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = summarize(args, attn_impl, sources, rows)
    p846.write_jsonl(out_dir / f"phase855_{args.model}_rows.jsonl", rows)
    p846.write_json(out_dir / f"phase855_{args.model}_summary.json", summary)
    print(
        json.dumps(
            {
                "model": args.model,
                "round": args.round_name,
                "sources": len(sources),
                "rows": len(rows),
                "full_combo_rollout_answer_class": summary["condition_summary"].get("full_combo", {}).get("rollout_answer_class", 0),
                "full_combo_first_token_answer_class": summary["condition_summary"].get("full_combo", {}).get("answer_class_closure", 0),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return summary


def binary_stats(actual: list[bool], predicted: list[bool]) -> dict[str, Any]:
    tp = sum(1 for a, p in zip(actual, predicted) if a and p)
    tn = sum(1 for a, p in zip(actual, predicted) if (not a) and (not p))
    fp = sum(1 for a, p in zip(actual, predicted) if (not a) and p)
    fn = sum(1 for a, p in zip(actual, predicted) if a and (not p))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    n = len(actual)
    return {
        "n": n,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": (tp + tn) / n if n else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def compact(rows: list[dict[str, Any]]) -> dict[str, Any]:
    actual = [bool(row.get("rollout_answer_class")) for row in rows]
    predicted = [bool(row.get("answer_class_closure")) for row in rows]
    return {
        "n": len(rows),
        "answer_class_closure": sum(1 for row in rows if row.get("answer_class_closure")),
        "strict_closure": sum(1 for row in rows if row.get("strict_closure")),
        "rollout_answer_class": sum(1 for row in rows if row.get("rollout_answer_class")),
        "rollout_strict_canonical": sum(1 for row in rows if row.get("rollout_strict_canonical")),
        "rollout_object_echo": sum(1 for row in rows if row.get("rollout_object_echo")),
        "rollout_other_or_format": sum(1 for row in rows if row.get("rollout_other_or_format")),
        "rollout_labels": dict(Counter(str(row.get("rollout_label")) for row in rows)),
        "mean_class_blocker_count": mean([finite(row.get("class_blocker_count")) for row in rows]),
        "mean_class_rank": mean([finite(row.get("class_best_target_rank")) for row in rows]),
        "first_token_predicts_rollout": binary_stats(actual, predicted),
    }


def summarize(args: argparse.Namespace, attn_impl: str | None, sources: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_condition[str(row.get("condition_type"))].append(row)
        by_group[str(row.get("source_group"))].append(row)
    return {
        "phase": PHASE,
        "title": "Answer-Class Alias Field and Rollout Closure Validation",
        "model": args.model,
        "round": args.round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "dtype": "bfloat16",
        "quantization": "off",
        "source_round": args.source_round,
        "n_sources": len(sources),
        "source_groups": dict(Counter(str(row.get("source_group")) for row in sources)),
        "n_rows": len(rows),
        "condition_summary": {key: compact(group) for key, group in sorted(by_condition.items())},
        "source_group_summary": {
            key: compact([row for row in group if row.get("condition_type") == "full_combo"])
            for key, group in sorted(by_group.items())
        },
        "top_rows": sorted(
            rows,
            key=lambda row: (
                int(row.get("condition_type") == "full_combo"),
                int(row.get("answer_class_closure")) - int(row.get("rollout_answer_class")),
                finite(row.get("class_blocker_count")),
            ),
            reverse=True,
        )[:80],
        "boundary": (
            "This phase tests whether expanded answer-class first-token closure predicts short greedy rollout closure. "
            "It still does not prove full language closure or cross-domain invariance."
        ),
    }


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# Phase 855 Answer-Class Alias Field and Rollout Closure Validation ({payload['round']})",
        "",
        "- Source: Phase 854 full-combo rows.",
        "- Boundary: short greedy rollout, not final language closure.",
        "",
        "## Cross-Model Summary",
        "",
        "| model | sources | full first-token class | full rollout class | full strict rollout | full object echo | full predictor F1 | labels |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for model_name in MODELS:
        data = payload.get("model_summaries", {}).get(model_name) or {}
        full = (data.get("condition_summary") or {}).get("full_combo") or {}
        pred = full.get("first_token_predicts_rollout") or {}
        lines.append(
            f"| {model_name} | {data.get('n_sources', 0)} | {full.get('answer_class_closure', 0)} | "
            f"{full.get('rollout_answer_class', 0)} | {full.get('rollout_strict_canonical', 0)} | "
            f"{full.get('rollout_object_echo', 0)} | {fmt(pred.get('f1'))} | "
            f"`{json.dumps(full.get('rollout_labels') or {}, ensure_ascii=False)}` |"
        )
    lines += [
        "",
        "## Conditions",
        "",
        "| model | condition | n | first-token class | rollout class | strict token | class blockers | class rank | predictor F1 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model_name in MODELS:
        data = payload.get("model_summaries", {}).get(model_name) or {}
        for condition in ["original", "full_combo", "without_necessary"]:
            stats = (data.get("condition_summary") or {}).get(condition) or {}
            pred = stats.get("first_token_predicts_rollout") or {}
            lines.append(
                f"| {model_name} | `{condition}` | {stats.get('n', 0)} | {stats.get('answer_class_closure', 0)} | "
                f"{stats.get('rollout_answer_class', 0)} | {stats.get('strict_closure', 0)} | "
                f"{fmt(stats.get('mean_class_blocker_count'))} | {fmt(stats.get('mean_class_rank'))} | {fmt(pred.get('f1'))} |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    payload: dict[str, Any] = {
        "phase": PHASE,
        "round": round_name,
        "status": "missing",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": [],
        "model_summaries": {},
    }
    for model_name in MODELS:
        path = out_dir / f"phase855_{model_name}_summary.json"
        if path.exists():
            payload["models"].append(model_name)
            payload["model_summaries"][model_name] = p846.read_json(path)
    payload["status"] = "complete" if len(payload["models"]) == len(MODELS) else "partial"
    p846.write_json(out_dir / "phase855_cross_model_summary.json", payload)
    write_markdown(out_dir / "phase855_cross_model_summary.md", payload)
    print(json.dumps({"status": payload["status"], "round": round_name, "models": payload["models"]}, ensure_ascii=False, indent=2))
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="smoke")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--source-round", default="confirm")
    parser.add_argument("--max-strong-target-class-closed", type=int, default=12)
    parser.add_argument("--max-strong-target-class-open", type=int, default=6)
    parser.add_argument("--max-strong-other", type=int, default=6)
    parser.add_argument("--max-controls", type=int, default=6)
    parser.add_argument("--max-sources", type=int, default=0)
    parser.add_argument("--include-min-cut-conditions", action="store_true")
    parser.add_argument("--max-min-cut-conditions-per-source", type=int, default=1)
    parser.add_argument("--topk-blockers", type=int, default=30)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--attn-implementations", default="flash_attention_2,sdpa,eager")
    parser.add_argument("--log-every", type=int, default=4)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.summarize_only:
        summarize_round(args.round_name)
        return
    if not args.model:
        raise SystemExit("--model is required unless --summarize-only is used")
    eval_model(args)


if __name__ == "__main__":
    main()
