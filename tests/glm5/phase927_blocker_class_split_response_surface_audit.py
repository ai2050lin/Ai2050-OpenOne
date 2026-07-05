#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase846_geometry_boundary_equation_fitting as p846  # noqa: E402


PHASE = 927
MODELS = ["qwen3", "glm4", "deepseek7b"]
RESULT_ROOT = Path("tests/result/phase927_blocker_class_split_response_surface_audit")
PHASE926_ROOT = Path("tests/result/phase926_generalized_route_protocol_surface_validation")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def median(values: list[float | int | None]) -> float | None:
    cleaned = [float(value) for value in values if value is not None]
    return None if not cleaned else float(statistics.median(cleaned))


def mean(values: list[float | int | None]) -> float | None:
    cleaned = [float(value) for value in values if value is not None]
    return None if not cleaned else float(sum(cleaned) / len(cleaned))


def blocker_class(token: Any) -> str:
    text = "" if token is None else str(token)
    stripped = text.strip()
    if text == "a" or stripped == "a":
        return "article_a"
    if stripped in {".", "。"}:
        return "punctuation_period"
    if stripped in {"", "<0x0A>", "\\n"}:
        return "blank_or_newline"
    if stripped.lower() in {"the", "an"}:
        return "article_other"
    return "other"


def row_seed_class(row: dict[str, Any]) -> str:
    return str(row.get("phase925_seed_blocker_class") or blocker_class(row.get("phase925_seed_blocker_token")))


def surface_seed_class(row: dict[str, Any]) -> str:
    return str(row.get("phase925_seed_blocker_class") or blocker_class(row.get("phase925_seed_blocker_token")))


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    base_rows = [
        row
        for row in rows
        if abs(float(row.get("route_alpha") or 0.0) - 1.0) <= 1e-9
        and abs(float(row.get("protocol_span_factor") or 0.0) - 1.0) <= 1e-9
    ]
    non_base_rows = [row for row in rows if row not in base_rows]
    return {
        "rows": len(rows),
        "base_rows": len(base_rows),
        "non_base_rows": len(non_base_rows),
        "unique_states": len({row.get("target_state_key") for row in rows}),
        "unique_cases": len({row.get("target_case_id") for row in rows}),
        "unique_domains": len({row.get("target_eval_domain") for row in rows}),
        "unique_l4_groups": len({row.get("phase925_group_kind") for row in rows}),
        "top1": sum(1 for row in rows if row.get("patched_eos_top1")),
        "margin_nonnegative": sum(1 for row in rows if row.get("patched_eos_margin_nonnegative")),
        "strict_clean_candidate": sum(1 for row in rows if row.get("strict_clean_candidate")),
        "base_top1": sum(1 for row in base_rows if row.get("patched_eos_top1")),
        "base_margin_nonnegative": sum(1 for row in base_rows if row.get("patched_eos_margin_nonnegative")),
        "base_strict_clean_candidate": sum(1 for row in base_rows if row.get("strict_clean_candidate")),
        "non_base_top1": sum(1 for row in non_base_rows if row.get("patched_eos_top1")),
        "non_base_margin_nonnegative": sum(1 for row in non_base_rows if row.get("patched_eos_margin_nonnegative")),
        "non_base_strict_clean_candidate": sum(1 for row in non_base_rows if row.get("strict_clean_candidate")),
        "improved_margin_vs_surface_base": sum(1 for row in non_base_rows if row.get("improved_margin_vs_surface_base")),
        "worsened_margin_vs_surface_base": sum(1 for row in non_base_rows if row.get("worsened_margin_vs_surface_base")),
        "lost_margin_closure_vs_surface_base": sum(1 for row in non_base_rows if row.get("lost_margin_closure_vs_surface_base")),
        "new_margin_closure_vs_surface_base": sum(1 for row in non_base_rows if row.get("new_margin_closure_vs_surface_base")),
        "new_top1_vs_surface_base": sum(1 for row in non_base_rows if row.get("new_top1_vs_surface_base")),
        "new_strict_vs_surface_base": sum(1 for row in non_base_rows if row.get("new_strict_vs_surface_base")),
        "median_margin_delta_vs_surface_base": median([row.get("margin_delta_vs_surface_base") for row in non_base_rows]),
        "mean_margin_delta_vs_surface_base": mean([row.get("margin_delta_vs_surface_base") for row in non_base_rows]),
        "median_patched_margin": median([row.get("patched_eos_margin_vs_blocker") for row in rows]),
        "median_seed_margin": median([row.get("phase925_seed_margin") for row in rows]),
        "median_seed_rank": median([row.get("phase925_seed_rank") for row in rows]),
        "target_state_coverage_top1": len({row.get("target_state_key") for row in rows if row.get("patched_eos_top1")}),
        "target_state_coverage_margin": len(
            {row.get("target_state_key") for row in rows if row.get("patched_eos_margin_nonnegative")}
        ),
        "target_state_coverage_strict": len({row.get("target_state_key") for row in rows if row.get("strict_clean_candidate")}),
        "patched_blocker_class_distribution": dict(Counter(blocker_class(row.get("patched_blocker_token")) for row in rows)),
        "target_boundary_blocker_class_distribution": dict(
            Counter(blocker_class(row.get("target_boundary_blocker_token")) for row in rows)
        ),
    }


def summarize_surfaces(surfaces: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "surfaces": len(surfaces),
        "unique_states": len({row.get("target_state_key") for row in surfaces}),
        "unique_cases": len({row.get("target_case_id") for row in surfaces}),
        "unique_domains": len({row.get("target_eval_domain") for row in surfaces}),
        "best_coord_is_base": sum(1 for row in surfaces if row.get("best_coord_is_base")),
        "best_alpha_lt_1": sum(1 for row in surfaces if row.get("best_alpha_lt_1")),
        "best_alpha_eq_1": sum(1 for row in surfaces if row.get("best_alpha_eq_1")),
        "best_alpha_gt_1": sum(1 for row in surfaces if row.get("best_alpha_gt_1")),
        "best_protocol_lt_1": sum(1 for row in surfaces if row.get("best_protocol_lt_1")),
        "best_protocol_eq_1": sum(1 for row in surfaces if row.get("best_protocol_eq_1")),
        "best_protocol_gt_1": sum(1 for row in surfaces if row.get("best_protocol_gt_1")),
        "with_closure_coord": sum(1 for row in surfaces if (row.get("closure_coord_count") or 0) > 0),
        "closure_coord_count_total": sum(int(row.get("closure_coord_count") or 0) for row in surfaces),
        "median_best_margin_delta_vs_surface_base": median(
            [row.get("best_margin_delta_vs_surface_base") for row in surfaces]
        ),
        "mean_best_margin_delta_vs_surface_base": mean(
            [row.get("best_margin_delta_vs_surface_base") for row in surfaces]
        ),
        "best_alpha_distribution": dict(Counter(str(row.get("best_alpha")) for row in surfaces)),
        "best_protocol_distribution": dict(Counter(str(row.get("best_protocol_factor")) for row in surfaces)),
        "best_l39_factor_distribution": dict(Counter(str(row.get("l39_factor")) for row in surfaces)),
    }


def summarize_combo(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[tuple(str(row.get(key)) for key in keys)].append(row)
    out = []
    for key_tuple, vals in buckets.items():
        payload = summarize_rows(vals)
        for key, value in zip(keys, key_tuple):
            payload[key] = value
        out.append(payload)
    out.sort(
        key=lambda item: (
            item.get("new_strict_vs_surface_base") or 0,
            item.get("new_top1_vs_surface_base") or 0,
            item.get("new_margin_closure_vs_surface_base") or 0,
            item.get("non_base_top1") or 0,
            item.get("improved_margin_vs_surface_base") or 0,
        ),
        reverse=True,
    )
    return out


def classify_evidence(by_class: dict[str, dict[str, Any]]) -> str:
    article = by_class.get("article_a", {})
    punct = by_class.get("punctuation_period", {})
    if not article and not punct:
        return "no_blocker_class_data"
    article_closes = int(article.get("new_top1_vs_surface_base") or 0) > 0 or int(article.get("top1") or 0) > 0
    punct_closes = int(punct.get("new_top1_vs_surface_base") or 0) > 0 or int(punct.get("top1") or 0) > 0
    punct_moves = int(punct.get("surfaces", 0)) > int(punct.get("best_coord_is_base", 0))
    if article_closes and not punct_closes and punct_moves:
        return "blocker_class_split_confirmed_article_closes_punctuation_moves_only"
    if article_closes and punct_closes:
        return "both_article_and_punctuation_close"
    if punct_moves:
        return "punctuation_surface_moves_without_article_closure"
    return "limited_blocker_class_split_signal"


def eval_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    src_dir = PHASE926_ROOT / args.phase926_round
    rows = read_jsonl(src_dir / f"phase926_{args.model}_rows.jsonl")
    surfaces = read_jsonl(src_dir / f"phase926_{args.model}_surfaces.jsonl")
    seeds = read_jsonl(src_dir / f"phase926_{args.model}_selected_seeds.jsonl")
    rows_by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_class[row_seed_class(row)].append(row)
    surfaces_by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for surface in surfaces:
        surfaces_by_class[surface_seed_class(surface)].append(surface)
    seed_by_class = Counter(blocker_class(seed.get("patched_blocker_token")) for seed in seeds)
    class_payloads = []
    by_class_for_evidence: dict[str, dict[str, Any]] = {}
    for cls in sorted(set(rows_by_class) | set(surfaces_by_class) | set(seed_by_class)):
        row_summary = summarize_rows(rows_by_class.get(cls, []))
        surface_summary = summarize_surfaces(surfaces_by_class.get(cls, []))
        merged = {
            "blocker_class": cls,
            "selected_seeds": int(seed_by_class.get(cls, 0)),
            **row_summary,
            **surface_summary,
        }
        class_payloads.append(merged)
        by_class_for_evidence[cls] = merged
    evidence = classify_evidence(by_class_for_evidence)
    payload = {
        "phase": PHASE,
        "title": "Blocker-Class Split Response Surface Audit",
        "model": args.model,
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "phase926_rows": len(rows),
        "phase926_surfaces": len(surfaces),
        "phase926_selected_seeds": len(seeds),
        "seed_by_class": dict(seed_by_class),
        "class_summaries": class_payloads,
        "by_class_domain": summarize_combo(rows, ["phase925_seed_blocker_class", "target_eval_domain"]),
        "by_class_group": summarize_combo(rows, ["phase925_seed_blocker_class", "phase925_group_kind"]),
        "by_class_alpha_protocol": summarize_combo(rows, ["phase925_seed_blocker_class", "route_alpha", "protocol_span_factor"]),
        "surfaces_with_closure": [
            row for row in surfaces if int(row.get("closure_coord_count") or 0) > 0
        ],
        "new_closure_rows": [
            row
            for row in rows
            if row.get("new_top1_vs_surface_base") or row.get("new_margin_closure_vs_surface_base")
        ],
        "evidence_label": evidence,
        "boundary": (
            "Phase927 is an offline audit over Phase926 forward results. It tests whether article and punctuation "
            "blockers form different response-surface families before searching for new punctuation-specific gears."
        ),
    }
    p846.write_json(out_dir / f"phase927_{args.model}_summary.json", payload)
    p846.write_jsonl(out_dir / f"phase927_{args.model}_class_summaries.jsonl", class_payloads)
    print(
        json.dumps(
            {
                "phase": PHASE,
                "model": args.model,
                "rows": len(rows),
                "surfaces": len(surfaces),
                "classes": [item["blocker_class"] for item in class_payloads],
                "evidence_label": evidence,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return payload


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    summaries = []
    evidence = Counter()
    scalar = Counter()
    all_classes = []
    new_closures = []
    closure_surfaces = []
    for model_name in MODELS:
        path = out_dir / f"phase927_{model_name}_summary.json"
        if not path.exists():
            continue
        summary = read_json(path)
        summaries.append(summary)
        evidence[str(summary.get("evidence_label"))] += 1
        for key in ["phase926_rows", "phase926_surfaces", "phase926_selected_seeds"]:
            scalar[key] += int(summary.get(key) or 0)
        for item in summary.get("class_summaries") or []:
            cls_item = dict(item)
            cls_item["model"] = model_name
            all_classes.append(cls_item)
            for metric in [
                "selected_seeds",
                "rows",
                "surfaces",
                "top1",
                "new_top1_vs_surface_base",
                "new_margin_closure_vs_surface_base",
                "new_strict_vs_surface_base",
                "with_closure_coord",
                "best_coord_is_base",
            ]:
                scalar[f"{item.get('blocker_class')}_{metric}"] += int(item.get(metric) or 0)
        for row in summary.get("new_closure_rows") or []:
            new_closures.append({"model": model_name, **row})
        for row in summary.get("surfaces_with_closure") or []:
            closure_surfaces.append({"model": model_name, **row})
    all_classes.sort(
        key=lambda item: (
            item.get("new_top1_vs_surface_base") or 0,
            item.get("top1") or 0,
            item.get("surfaces") or 0,
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
        "evidence_label_counts": dict(sorted(evidence.items())),
        "model_summaries": summaries,
        "class_summaries": all_classes,
        "new_closure_rows": new_closures,
        "surfaces_with_closure": closure_surfaces,
    }
    p846.write_json(out_dir / "phase927_cross_model_summary.json", payload)
    write_markdown(out_dir / "phase927_cross_model_summary.md", payload)
    return payload


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Phase 927 blocker-class split response surface audit",
        "",
        "## Overall",
        "",
        f"- models: {', '.join(payload.get('models') or [])}",
    ]
    for key, value in (payload.get("overall_scalar") or {}).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Evidence", ""])
    for key, value in (payload.get("evidence_label_counts") or {}).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Class Summaries", ""])
    lines.append(
        "| model | class | seeds | rows | surfaces | best base | top1 | new top1 | new margin | new strict | closure surfaces | best alpha | best protocol |"
    )
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |")
    for item in payload.get("class_summaries") or []:
        lines.append(
            "| {model} | {blocker_class} | {selected_seeds} | {rows} | {surfaces} | {best_coord_is_base} | {top1} | {new_top1_vs_surface_base} | {new_margin_closure_vs_surface_base} | {new_strict_vs_surface_base} | {with_closure_coord} | {best_alpha_distribution} | {best_protocol_distribution} |".format(
                **{
                    "model": item.get("model"),
                    "blocker_class": item.get("blocker_class"),
                    "selected_seeds": item.get("selected_seeds"),
                    "rows": item.get("rows"),
                    "surfaces": item.get("surfaces"),
                    "best_coord_is_base": item.get("best_coord_is_base"),
                    "top1": item.get("top1"),
                    "new_top1_vs_surface_base": item.get("new_top1_vs_surface_base"),
                    "new_margin_closure_vs_surface_base": item.get("new_margin_closure_vs_surface_base"),
                    "new_strict_vs_surface_base": item.get("new_strict_vs_surface_base"),
                    "with_closure_coord": item.get("with_closure_coord"),
                    "best_alpha_distribution": item.get("best_alpha_distribution"),
                    "best_protocol_distribution": item.get("best_protocol_distribution"),
                }
            )
        )
    lines.extend(["", "## New Closure Rows", ""])
    lines.append("| model | class | state | case | domain | group | l39 | alpha | protocol | base margin | margin | strict |")
    lines.append("| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |")
    for row in payload.get("new_closure_rows") or []:
        row = {
            "model": "",
            "phase925_seed_blocker_class": "",
            "target_state_key": "",
            "target_case_id": "",
            "target_eval_domain": "",
            "phase925_group_kind": "",
            "l39_factor": "",
            "route_alpha": "",
            "protocol_span_factor": "",
            "surface_base_margin": "",
            "patched_eos_margin_vs_blocker": "",
            "strict_clean_candidate": "",
            **row,
        }
        lines.append(
            "| {model} | {phase925_seed_blocker_class} | {target_state_key} | {target_case_id} | {target_eval_domain} | {phase925_group_kind} | {l39_factor} | {route_alpha} | {protocol_span_factor} | {surface_base_margin} | {patched_eos_margin_vs_blocker} | {strict_clean_candidate} |".format(
                **row
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="blocker_class_split_response_surface_audit")
    parser.add_argument("--phase926-round", default="generalized_route_protocol_surface_validation")
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
