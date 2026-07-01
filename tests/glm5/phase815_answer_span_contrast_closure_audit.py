#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))

from model_utils import MODEL_CONFIGS  # noqa: E402
from phase735_source_restricted_writer_validation import MODELS  # noqa: E402
from phase722_functional_head_atlas_causal_ablation import write_json, write_jsonl  # noqa: E402
from phase776_readout_bridge_competition_audit import normalize_token_text  # noqa: E402


PHASE = 815
SOURCE_ROOT = Path("tests/result/phase814_tokenizer_valid_answer_unit_closure")
RESULT_ROOT = Path("tests/result/phase815_answer_span_contrast_closure_audit")


def log(msg: str) -> None:
    print(f"[phase815] {msg}", flush=True)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def norm_text(value: Any) -> str:
    return normalize_token_text("" if value is None else str(value)).strip().lower()


def surface_text_variants(value: Any) -> list[str]:
    raw = "" if value is None else str(value).strip()
    if not raw:
        return []
    variants = {
        raw,
        raw.lower(),
        raw.upper(),
        raw.title(),
        f" {raw}",
        f" {raw.lower()}",
        f" {raw.upper()}",
        f" {raw.title()}",
    }
    return sorted(v for v in variants if v)


def load_tokenizer(model_name: str):
    from transformers import AutoTokenizer

    cfg = MODEL_CONFIGS[model_name]
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"],
        trust_remote_code=True,
        local_files_only=True,
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def span_variants(tokenizer, answer: Any, max_saved: int) -> list[dict[str, Any]]:
    variants = []
    for text in surface_text_variants(answer):
        try:
            token_ids = [int(x) for x in tokenizer.encode(text, add_special_tokens=False)]
        except Exception:
            continue
        if not token_ids:
            continue
        decoded = tokenizer.decode(token_ids, skip_special_tokens=False)
        variants.append(
            {
                "text": text,
                "token_ids": token_ids,
                "span_len": len(token_ids),
                "decoded": decoded,
                "normalized_decoded": norm_text(decoded),
            }
        )
    variants.sort(key=lambda item: (item["span_len"], item["text"]))
    return variants[:max_saved]


def bool_value(value: Any) -> bool:
    return bool(value)


def finite_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def is_answer_norm(text: Any, answer: Any) -> bool:
    return norm_text(text) == norm_text(answer)


def first_non_answer_rank(row: dict[str, Any], answer: Any, contrast: Any) -> dict[str, Any] | None:
    answer_norm = norm_text(answer)
    contrast_norm = norm_text(contrast)
    for item in row.get("after_rank_window") or []:
        text = item.get("token_text")
        token_norm = norm_text(text)
        if token_norm == answer_norm:
            continue
        out = dict(item)
        out["is_contrast_equiv"] = bool(token_norm and token_norm == contrast_norm)
        return out
    return None


def blocker_class_counts(row: dict[str, Any]) -> dict[str, int]:
    counts = row.get("after_class_counts") or {}
    return {str(k): finite_int(v) for k, v in counts.items()}


def dominant_blocker_class(row: dict[str, Any]) -> str | None:
    counts = blocker_class_counts(row)
    if not counts:
        return None
    ordered = sorted(counts.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    return ordered[0][0] if ordered else None


def audit_row(row: dict[str, Any], tokenizer, args: argparse.Namespace) -> dict[str, Any]:
    target_answer = row.get("target_answer")
    contrast_answer = row.get("contrast_answer")
    target_spans = span_variants(tokenizer, target_answer, int(args.max_span_variants_saved))
    contrast_spans = span_variants(tokenizer, contrast_answer, int(args.max_span_variants_saved))
    target_span_lens = [int(item["span_len"]) for item in target_spans]
    contrast_span_lens = [int(item["span_len"]) for item in contrast_spans]
    target_has_single_token_span = any(length == 1 for length in target_span_lens)
    target_has_multi_token_span = any(length > 1 for length in target_span_lens)
    target_requires_multi_token_span = bool(target_spans) and not target_has_single_token_span
    contrast_has_single_token_span = any(length == 1 for length in contrast_span_lens)
    contrast_requires_multi_token_span = bool(contrast_spans) and not contrast_has_single_token_span

    unit_text = row.get("after_answer_unit_text")
    raw_text = row.get("after_canonical_surface_token_text")
    unit_surface_valid = is_answer_norm(unit_text, target_answer)
    raw_surface_valid = is_answer_norm(raw_text, target_answer)
    answer_class = bool_value(row.get("after_answer_class_closure"))
    answer_unit = bool_value(row.get("after_answer_unit_closure"))
    contrast_cleared = bool_value(row.get("after_contrast_class_cleared_by_answer"))
    strict_token = bool_value(row.get("token_closure"))
    raw_canon = bool_value(row.get("after_canonical_surface_closure"))
    span_proxy = answer_class and answer_unit and contrast_cleared and unit_surface_valid
    contrast_above_answer = finite_int(row.get("after_contrast_class_above_answer_count"))
    first_blocker = first_non_answer_rank(row, target_answer, contrast_answer) or {}

    if strict_token:
        label = "strict_token_closed"
    elif span_proxy:
        label = "span_proxy_closed_no_strict_token"
    elif answer_class and answer_unit and not contrast_cleared:
        label = "span_unit_closed_contrast_not_cleared"
    elif answer_class and not answer_unit:
        label = "answer_closed_unit_fragmented"
    elif answer_unit and not answer_class:
        label = "unit_closed_answer_not_closed"
    elif contrast_above_answer > 0:
        label = "contrast_interference"
    else:
        label = "global_competition_unclosed"

    return {
        "row_kind": "phase815_answer_span_contrast_audit",
        "source_phase": 814,
        "model": row.get("model"),
        "round": row.get("round"),
        "case_id": row.get("case_id"),
        "domain": row.get("domain"),
        "relation": row.get("relation"),
        "object": row.get("object"),
        "target_answer": target_answer,
        "contrast_answer": contrast_answer,
        "search_stage": row.get("search_stage"),
        "combo_size": row.get("combo_size"),
        "combo_item_ids": row.get("combo_item_ids"),
        "target_span_variants": target_spans,
        "contrast_span_variants": contrast_spans,
        "target_has_single_token_span": target_has_single_token_span,
        "target_has_multi_token_span": target_has_multi_token_span,
        "target_requires_multi_token_span": target_requires_multi_token_span,
        "contrast_has_single_token_span": contrast_has_single_token_span,
        "contrast_requires_multi_token_span": contrast_requires_multi_token_span,
        "after_answer_unit_text": unit_text,
        "after_raw_canonical_token_text": raw_text,
        "answer_unit_surface_valid": unit_surface_valid,
        "raw_canonical_surface_valid": raw_surface_valid,
        "answer_class_closure": answer_class,
        "answer_unit_closure": answer_unit,
        "contrast_class_cleared": contrast_cleared,
        "contrast_class_above_answer_count": contrast_above_answer,
        "raw_canonical_surface_closure": raw_canon,
        "strict_token_closure": strict_token,
        "span_proxy_closure": span_proxy,
        "answer_class_margin_vs_top_non_answer": row.get("after_answer_class_margin_vs_top_non_answer"),
        "answer_unit_margin_vs_best_variant": row.get("after_answer_unit_margin_vs_best_variant"),
        "answer_unit_variant_count_above_unit": row.get("after_answer_unit_variant_count_above_unit"),
        "after_full_above_count": row.get("after_full_above_count"),
        "after_required_bias_to_clear_all": row.get("after_required_bias_to_clear_all"),
        "dominant_blocker_class": dominant_blocker_class(row),
        "blocker_class_counts": blocker_class_counts(row),
        "first_non_answer_token": first_blocker,
        "phase815_label": label,
    }


def load_round_rows(round_name: str, model_name: str) -> list[dict[str, Any]]:
    path = SOURCE_ROOT / round_name / f"phase814_{model_name}_rows.jsonl"
    rows = []
    for row in read_jsonl(path):
        if row.get("row_kind") == "phase814_tokenizer_valid_answer_unit_closure" and not row.get("combo_error"):
            rows.append(row)
    return rows


def summarize_rows(rows: list[dict[str, Any]], model_name: str, round_name: str) -> dict[str, Any]:
    labels = Counter(row.get("phase815_label") for row in rows)
    unit_closed_answer_not_closed = [row for row in rows if row.get("phase815_label") == "unit_closed_answer_not_closed"]
    blocker_counts = Counter(row.get("dominant_blocker_class") for row in unit_closed_answer_not_closed if row.get("dominant_blocker_class"))
    best = sorted(
        rows,
        key=lambda row: (
            not bool(row.get("span_proxy_closure")),
            not bool(row.get("answer_class_closure")),
            not bool(row.get("answer_unit_closure")),
            finite_int(row.get("after_full_above_count"), 999999),
            finite_int(row.get("combo_size"), 999999),
        ),
    )[:60]
    return {
        "phase": PHASE,
        "title": "Answer Span And Contrast Class Closure Audit",
        "model": model_name,
        "round": round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_root": str(SOURCE_ROOT),
        "n_rows": len(rows),
        "span_proxy_closure_rows": sum(1 for row in rows if row.get("span_proxy_closure")),
        "answer_class_closure_rows": sum(1 for row in rows if row.get("answer_class_closure")),
        "answer_unit_closure_rows": sum(1 for row in rows if row.get("answer_unit_closure")),
        "contrast_class_cleared_rows": sum(1 for row in rows if row.get("contrast_class_cleared")),
        "raw_canonical_surface_closure_rows": sum(1 for row in rows if row.get("raw_canonical_surface_closure")),
        "strict_token_closure_rows": sum(1 for row in rows if row.get("strict_token_closure")),
        "target_has_multi_token_span_rows": sum(1 for row in rows if row.get("target_has_multi_token_span")),
        "target_requires_multi_token_span_rows": sum(1 for row in rows if row.get("target_requires_multi_token_span")),
        "contrast_requires_multi_token_span_rows": sum(1 for row in rows if row.get("contrast_requires_multi_token_span")),
        "unit_closed_answer_not_closed_rows": len(unit_closed_answer_not_closed),
        "answer_closed_unit_fragmented_rows": sum(1 for row in rows if row.get("phase815_label") == "answer_closed_unit_fragmented"),
        "by_label": dict(labels),
        "dominant_blockers_when_unit_closed_answer_not_closed": dict(blocker_counts),
        "best_rows": best,
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# Phase 815 Answer Span And Contrast Closure Audit ({payload['round']})",
        "",
        "- Source: Phase 814 saved rows; no new model forward pass.",
        "- Boundary: answer span proxy closure requires answer-class closure, answer-unit closure, contrast-class cleared, and surface-valid answer unit.",
        "",
        "## Model Summary",
        "",
        "| model | rows | span_proxy | answer_class | answer_unit | contrast_cleared | raw_canon | strict_token | unit_closed_answer_not_closed | answer_closed_unit_fragmented | multi_token_target | labels |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for model_name in MODELS:
        data = payload.get("model_summaries", {}).get(model_name)
        if not data:
            continue
        lines.append(
            f"| {model_name} | {data.get('n_rows')} | {data.get('span_proxy_closure_rows')} | "
            f"{data.get('answer_class_closure_rows')} | {data.get('answer_unit_closure_rows')} | "
            f"{data.get('contrast_class_cleared_rows')} | {data.get('raw_canonical_surface_closure_rows')} | "
            f"{data.get('strict_token_closure_rows')} | {data.get('unit_closed_answer_not_closed_rows')} | "
            f"{data.get('answer_closed_unit_fragmented_rows')} | {data.get('target_has_multi_token_span_rows')} | "
            f"`{json.dumps(data.get('by_label') or {}, ensure_ascii=False)}` |"
        )
    lines += ["", "## Unit Closed But Answer Not Closed Blockers", ""]
    lines += ["| model | dominant blocker classes |", "|---|---|"]
    for model_name in MODELS:
        data = payload.get("model_summaries", {}).get(model_name)
        if not data:
            continue
        lines.append(
            f"| {model_name} | `{json.dumps(data.get('dominant_blockers_when_unit_closed_answer_not_closed') or {}, ensure_ascii=False)}` |"
        )
    lines += [
        "",
        "## Best Rows",
        "",
        "| model | case | unit | span_proxy | answer_class | answer_unit | contrast_clear | raw_canon | strict | first_non_answer | label |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for model_name in MODELS:
        data = payload.get("model_summaries", {}).get(model_name)
        if not data:
            continue
        for row in data.get("best_rows", [])[:24]:
            first = row.get("first_non_answer_token") or {}
            first_txt = first.get("token_text")
            first_cls = first.get("class")
            lines.append(
                f"| {model_name} | {row.get('case_id')} | `{row.get('after_answer_unit_text')}` | "
                f"{int(bool(row.get('span_proxy_closure')))} | {int(bool(row.get('answer_class_closure')))} | "
                f"{int(bool(row.get('answer_unit_closure')))} | {int(bool(row.get('contrast_class_cleared')))} | "
                f"{int(bool(row.get('raw_canonical_surface_closure')))} | {int(bool(row.get('strict_token_closure')))} | "
                f"`{first_txt}`/{first_cls} | `{row.get('phase815_label')}` |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_round(round_name: str, args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "phase": PHASE,
        "round": round_name,
        "status": "partial",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_summaries": {},
        "models": [],
    }
    for model_name in MODELS:
        source_rows = load_round_rows(round_name, model_name)
        if not source_rows:
            continue
        log(f"{round_name}/{model_name}: audit {len(source_rows)} phase814 rows")
        tokenizer = load_tokenizer(model_name)
        rows = [audit_row(row, tokenizer, args) for row in source_rows]
        del tokenizer
        summary = summarize_rows(rows, model_name, round_name)
        write_jsonl(out_dir / f"phase815_{model_name}_rows.jsonl", rows)
        write_json(out_dir / f"phase815_{model_name}_summary.json", summary)
        payload["model_summaries"][model_name] = summary
        payload["models"].append(model_name)
    payload["status"] = "complete" if len(payload["models"]) == len(MODELS) else "partial"
    write_json(out_dir / "phase815_cross_model_summary.json", payload)
    write_markdown(out_dir / "phase815_cross_model_summary.md", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", default="smoke,main,confirm")
    parser.add_argument("--max-span-variants-saved", type=int, default=24)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    results = {}
    for round_name in [x.strip() for x in args.rounds.split(",") if x.strip()]:
        results[round_name] = run_round(round_name, args)
    print(json.dumps({"phase": PHASE, "rounds": {k: v["status"] for k, v in results.items()}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
