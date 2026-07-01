#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import itertools
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))

import phase828_cross_component_consistency_fiber_composition as p828  # noqa: E402


PHASE = 829
RESULT_ROOT = Path("tests/result/phase829_non_interference_constrained_component_composition")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def finite(value: Any, default: float = 0.0) -> float:
    return p828.finite(value, default)


def group_has_signal(group: dict[str, Any]) -> bool:
    return bool(
        group.get("single_exact_target")
        or int(group.get("single_natural_target_count", 0)) > 0
        or int(group.get("single_target_total", 0)) > 0
        or int(group.get("single_improved_total", 0)) > 0
    )


def group_is_safe(group: dict[str, Any], args: argparse.Namespace) -> bool:
    return int(group.get("single_degraded", 0)) <= int(args.max_single_degraded)


def group_filter_reason(group: dict[str, Any], args: argparse.Namespace) -> str | None:
    if args.require_single_safe and not group_is_safe(group, args):
        return "single_degraded"
    if args.require_signal and not group_has_signal(group):
        return "no_single_signal"
    return None


def all_same_case_pairs(groups: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    by_case_budget: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for group in groups:
        by_case_budget[(str(group["case_id"]), int(group["budget"]))].append(group)
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for vals in by_case_budget.values():
        for a, b in itertools.combinations(vals, 2):
            if (int(a["layer_idx"]), a["component_kind"], a["component_label"]) == (
                int(b["layer_idx"]),
                b["component_kind"],
                b["component_label"],
            ):
                continue
            pairs.append((a, b))
    return pairs


def pair_has_anchor(pair: tuple[dict[str, Any], dict[str, Any]]) -> bool:
    a, b = pair
    return bool(a.get("single_exact_plus_multi") or b.get("single_exact_plus_multi"))


def build_constrained_pairs(
    groups: list[dict[str, Any]], args: argparse.Namespace
) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], dict[str, Any]]:
    excluded = Counter()
    safe_groups: list[dict[str, Any]] = []
    for group in groups:
        reason = group_filter_reason(group, args)
        if reason:
            excluded[reason] += 1
            continue
        safe_groups.append(group)

    raw_pairs = all_same_case_pairs(groups)
    safe_pairs = all_same_case_pairs(safe_groups)
    anchor_excluded = 0
    if args.require_pair_anchor:
        before = len(safe_pairs)
        safe_pairs = [pair for pair in safe_pairs if pair_has_anchor(pair)]
        anchor_excluded = before - len(safe_pairs)

    def pair_score(pair: tuple[dict[str, Any], dict[str, Any]]) -> tuple[int, int, int, int, int, int, float, str]:
        a, b = pair
        exact_plus = int(a["single_exact_plus_multi"]) + int(b["single_exact_plus_multi"])
        safety = -(int(a["single_degraded"]) + int(b["single_degraded"]))
        exact = int(a["single_exact_target"]) + int(b["single_exact_target"])
        natural = int(a["single_natural_target_count"]) + int(b["single_natural_target_count"])
        min_natural = min(int(a["single_natural_target_count"]), int(b["single_natural_target_count"]))
        target = int(a["single_target_total"]) + int(b["single_target_total"])
        delta = finite(a["single_delta_sum"]) + finite(b["single_delta_sum"])
        label = f"{p828.compact_component_label(a)} + {p828.compact_component_label(b)}"
        return (exact_plus, safety, exact, natural, min_natural, target, delta, label)

    safe_pairs.sort(key=pair_score, reverse=True)
    if int(args.max_pairs) > 0:
        safe_pairs = safe_pairs[: int(args.max_pairs)]

    diagnostics = {
        "raw_component_groups": len(groups),
        "eligible_component_groups": len(safe_groups),
        "excluded_component_groups": dict(excluded),
        "raw_same_case_pairs": len(raw_pairs),
        "safe_same_case_pairs_before_anchor": len(all_same_case_pairs(safe_groups)),
        "anchor_excluded_pairs": anchor_excluded,
        "selected_pairs": len(safe_pairs),
        "constraint": {
            "require_single_safe": bool(args.require_single_safe),
            "max_single_degraded": int(args.max_single_degraded),
            "require_signal": bool(args.require_signal),
            "require_pair_anchor": bool(args.require_pair_anchor),
        },
    }
    return safe_pairs, diagnostics


def patch_phase829_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    for row in rows:
        row["row_kind"] = "phase829_non_interference_constrained_component_composition"
        row["phase"] = PHASE
        row["validation_mode"] = "non_interference_constrained_cross_component_composition"
        row["non_interference_constraint"] = {
            "require_single_safe": bool(args.require_single_safe),
            "max_single_degraded": int(args.max_single_degraded),
            "require_signal": bool(args.require_signal),
            "require_pair_anchor": bool(args.require_pair_anchor),
        }
        row["component_a_single_degraded"] = int(row.get("component_a", {}).get("single_degraded", 0))
        row["component_b_single_degraded"] = int(row.get("component_b", {}).get("single_degraded", 0))


def pair_counts(rows: list[dict[str, Any]], min_natural_targets: int) -> dict[str, Any]:
    return p828.pair_counts(rows, min_natural_targets)


def compact(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return p828.compact(rows)


def summarize_rows(
    rows: list[dict[str, Any]],
    groups: list[dict[str, Any]],
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    diagnostics: dict[str, Any],
    args: argparse.Namespace,
    attn_impl: str | None = None,
) -> dict[str, Any]:
    by_donor = defaultdict(list)
    by_pair = defaultdict(list)
    for row in rows:
        by_donor[str(row.get("donor_variant"))].append(row)
        by_pair[str(row.get("pair_label"))].append(row)

    pair_records = []
    composition_new_exact_multi = 0
    composition_preserve_exact_multi = 0
    pair_exact_multi = 0
    for label, vals in by_pair.items():
        counts = pair_counts(vals, int(args.min_natural_targets))
        a = vals[0]["component_a"]
        b = vals[0]["component_b"]
        single_had_exact_multi = bool(a["single_exact_plus_multi"] or b["single_exact_plus_multi"])
        if counts["exact_plus_multi"]:
            pair_exact_multi += 1
            if single_had_exact_multi:
                composition_preserve_exact_multi += 1
            else:
                composition_new_exact_multi += 1
        pair_records.append(
            {
                "pair_label": label,
                "case_id": vals[0].get("case_id"),
                **counts,
                "single_had_exact_multi": single_had_exact_multi,
                "single_a_degraded": int(a["single_degraded"]),
                "single_b_degraded": int(b["single_degraded"]),
                "single_a_natural_target_count": int(a["single_natural_target_count"]),
                "single_b_natural_target_count": int(b["single_natural_target_count"]),
            }
        )

    natural_rows = [row for row in rows if row.get("donor_variant") != "exact_choices"]
    exact_rows = [row for row in rows if row.get("donor_variant") == "exact_choices"]
    natural_category_rows = [row for row in rows if row.get("donor_variant") == "natural_category"]
    return {
        "phase": PHASE,
        "title": "Non-Interference Constrained Component Composition",
        "model": args.model,
        "round": args.round_name,
        "source_round": args.source_round,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "dtype": "bfloat16",
        "quantization": "off",
        "n_rows": len(rows),
        "n_component_groups": len(groups),
        "n_pairs": len(pair_records),
        "diagnostics": diagnostics,
        "exact_target_rows": sum(1 for row in exact_rows if row.get("target_transition")),
        "natural_target_rows": sum(1 for row in natural_rows if row.get("target_transition")),
        "natural_degraded_rows": sum(1 for row in natural_rows if row.get("degraded_boundary")),
        "natural_category_target_rows": sum(1 for row in natural_category_rows if row.get("target_transition")),
        "natural_category_degraded_rows": sum(1 for row in natural_category_rows if row.get("degraded_boundary")),
        "pair_exact_plus_multi": pair_exact_multi,
        "composition_new_exact_multi": composition_new_exact_multi,
        "composition_preserve_exact_multi": composition_preserve_exact_multi,
        "donor_summary": {donor: compact(vals) for donor, vals in sorted(by_donor.items())},
        "pair_records": pair_records[:120],
        "selected_pairs": [
            [p828.compact_component_label(a), p828.compact_component_label(b)] for a, b in pairs
        ],
        "selected_component_groups": [
            {
                "label": p828.compact_component_label(group),
                "case_id": group["case_id"],
                "single_exact_plus_multi": group["single_exact_plus_multi"],
                "single_exact_target": group["single_exact_target"],
                "single_natural_target_count": group["single_natural_target_count"],
                "single_degraded": group["single_degraded"],
            }
            for group in groups
        ],
        "boundary": "This phase tests whether Phase 828 interference can be reduced by excluding unsafe single-component groups before composition.",
    }


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    standards = p828.p820.standard_rows()
    groups = p828.load_component_groups(args.model, args)
    pairs, diagnostics = build_constrained_pairs(groups, args)
    cmap = p828.p825.case_map()
    log(
        f"{args.model}/{args.round_name}: groups={len(groups)} "
        f"eligible={diagnostics['eligible_component_groups']} pairs={len(pairs)}"
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "model": args.model,
                    "diagnostics": diagnostics,
                    "groups": [p828.compact_component_label(group) for group in groups],
                    "pairs": [[p828.compact_component_label(a), p828.compact_component_label(b)] for a, b in pairs],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return {"groups": groups, "pairs": pairs, "diagnostics": diagnostics}
    if not pairs:
        summary = summarize_rows([], groups, pairs, diagnostics, args, attn_impl=None)
        summary["skipped_model_load"] = True
        summary["skip_reason"] = "no eligible non-interfering same-case cross-component pairs"
        p828.write_jsonl(out_dir / f"phase829_{args.model}_rows.jsonl", [])
        p828.write_json(out_dir / f"phase829_{args.model}_summary.json", summary)
        print(
            json.dumps(
                {
                    "model": args.model,
                    "round": args.round_name,
                    "groups": summary["n_component_groups"],
                    "pairs": summary["n_pairs"],
                    "skipped_model_load": True,
                    "skip_reason": summary["skip_reason"],
                    "diagnostics": diagnostics,
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
        return summary

    model, tokenizer, device, attn_impl = p828.p796.load_model_bf16_prefer_flash(
        args.model, args.attn_implementations
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    rows: list[dict[str, Any]] = []
    try:
        for idx, pair in enumerate(pairs, 1):
            case = cmap.get(str(pair[0]["case_id"]))
            if not case:
                continue
            new_rows = p828.eval_pair(model, tokenizer, device, standards, case, pair, args)
            patch_phase829_rows(new_rows, args)
            rows.extend(new_rows)
            if idx % int(args.log_every) == 0 or idx == len(pairs):
                log(f"{args.model}: evaluated {idx}/{len(pairs)} pairs; rows={len(rows)}")
    finally:
        p828.release_model(model)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = summarize_rows(rows, groups, pairs, diagnostics, args, attn_impl)
    p828.write_jsonl(out_dir / f"phase829_{args.model}_rows.jsonl", rows)
    p828.write_json(out_dir / f"phase829_{args.model}_summary.json", summary)
    print(
        json.dumps(
            {
                "model": args.model,
                "round": args.round_name,
                "groups": summary["n_component_groups"],
                "eligible_groups": diagnostics["eligible_component_groups"],
                "pairs": summary["n_pairs"],
                "exact_target_rows": summary["exact_target_rows"],
                "natural_target_rows": summary["natural_target_rows"],
                "natural_degraded_rows": summary["natural_degraded_rows"],
                "natural_category_degraded_rows": summary["natural_category_degraded_rows"],
                "pair_exact_plus_multi": summary["pair_exact_plus_multi"],
                "composition_new_exact_multi": summary["composition_new_exact_multi"],
                "composition_preserve_exact_multi": summary["composition_preserve_exact_multi"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return summary


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# Phase 829 Non-Interference Constrained Component Composition ({payload['round']})",
        "",
        "- Source: Phase 827 selected subspaces, with Phase 828 composition logic.",
        "- Objective: exclude unsafe single-component groups before two-component composition.",
        "",
        "## Model Summary",
        "",
        "| model | groups | eligible | pairs | exact target | natural target | natural degraded | natural_category degraded | pair exact+multi | new exact+multi | preserve exact+multi |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model_name in p828.MODELS:
        data = payload.get("model_summaries", {}).get(model_name)
        if not data:
            continue
        diag = data.get("diagnostics") or {}
        lines.append(
            f"| {model_name} | {data.get('n_component_groups')} | {diag.get('eligible_component_groups')} | "
            f"{data.get('n_pairs')} | {data.get('exact_target_rows')} | {data.get('natural_target_rows')} | "
            f"{data.get('natural_degraded_rows')} | {data.get('natural_category_degraded_rows')} | "
            f"{data.get('pair_exact_plus_multi')} | {data.get('composition_new_exact_multi')} | "
            f"{data.get('composition_preserve_exact_multi')} |"
        )

    lines += ["", "## Constraint Diagnostics", ""]
    lines += [
        "| model | raw pairs | safe pairs before anchor | anchor excluded | selected pairs | excluded groups |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for model_name in p828.MODELS:
        data = payload.get("model_summaries", {}).get(model_name) or {}
        diag = data.get("diagnostics") or {}
        lines.append(
            f"| {model_name} | {diag.get('raw_same_case_pairs')} | "
            f"{diag.get('safe_same_case_pairs_before_anchor')} | {diag.get('anchor_excluded_pairs')} | "
            f"{diag.get('selected_pairs')} | "
            f"`{json.dumps(diag.get('excluded_component_groups') or {}, ensure_ascii=False)}` |"
        )

    lines += ["", "## Donor Summary", ""]
    lines += [
        "| model | donor | n | improved | target | degraded | mean delta | classes |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for model_name in p828.MODELS:
        data = payload.get("model_summaries", {}).get(model_name) or {}
        for donor, row in sorted((data.get("donor_summary") or {}).items()):
            lines.append(
                f"| {model_name} | `{donor}` | {row.get('n')} | {row.get('improved_rows')} | "
                f"{row.get('target_transition_rows')} | {row.get('degraded_rows')} | "
                f"{finite(row.get('mean_delta_boundary_rank')):.3f} | "
                f"`{json.dumps(row.get('patched_classes') or {}, ensure_ascii=False)}` |"
            )

    lines += ["", "## Pair Records", ""]
    lines += [
        "| model | case | pair | exact | natural count | exact+multi | degraded | single had exact+multi |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for model_name in p828.MODELS:
        data = payload.get("model_summaries", {}).get(model_name) or {}
        for rec in data.get("pair_records") or []:
            lines.append(
                f"| {model_name} | `{rec.get('case_id')}` | `{rec.get('pair_label')}` | "
                f"{int(bool(rec.get('exact_target')))} | {rec.get('natural_target_count')} | "
                f"{int(bool(rec.get('exact_plus_multi')))} | {rec.get('degraded')} | "
                f"{int(bool(rec.get('single_had_exact_multi')))} |"
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
        "model_summaries": {},
        "models": [],
    }
    for model_name in p828.MODELS:
        path = out_dir / f"phase829_{model_name}_summary.json"
        if path.exists():
            payload["model_summaries"][model_name] = json.loads(path.read_text(encoding="utf-8"))
            payload["models"].append(model_name)
    payload["status"] = "complete" if len(payload["models"]) == len(p828.MODELS) else "partial"
    p828.write_json(out_dir / "phase829_cross_model_summary.json", payload)
    write_markdown(out_dir / "phase829_cross_model_summary.md", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=p828.MODELS)
    parser.add_argument("--round-name", default="smoke")
    parser.add_argument("--source-round", default="confirm")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--recipient-prompt", default="no_choices")
    parser.add_argument("--search-donor-prompts", default="exact_choices,natural_category,natural_question,object_only")
    parser.add_argument("--min-natural-targets", type=int, default=2)
    parser.add_argument("--budgets", default="16,32")
    parser.add_argument("--max-component-groups", type=int, default=16)
    parser.add_argument("--max-pairs", type=int, default=12)
    parser.add_argument("--include-weak-groups", action="store_true")
    parser.add_argument("--component-kinds", default="layer_residual,attention_output,mlp_output,attention_head,mlp_channel_group")
    parser.add_argument("--max-source-rows", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--attn-implementations", default="flash_attention_2,sdpa,eager")
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--require-single-safe", action="store_true")
    parser.add_argument("--max-single-degraded", type=int, default=0)
    parser.add_argument("--require-signal", action="store_true")
    parser.add_argument("--require-pair-anchor", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.summarize_only:
        payload = summarize_round(args.round_name)
        print(
            json.dumps(
                {"round": args.round_name, "status": payload["status"], "models": payload["models"]},
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if not args.model:
        raise SystemExit("--model is required unless --summarize-only")
    run_model(args)


if __name__ == "__main__":
    main()
