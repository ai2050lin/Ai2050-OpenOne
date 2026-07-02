#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import itertools
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

import phase844_geometry_route_natural_gear_set_search as p844  # noqa: E402
from model_utils import get_layers  # noqa: E402


PHASE = 845
RESULT_ROOT = Path("tests/result/phase845_geometry_gear_interaction_edge_atlas")
PHASE844_ROOT = Path("tests/result/phase844_geometry_route_natural_gear_set_search")
TARGET_CLASS = "target_equivalent"


def log(msg: str) -> None:
    p844.log(msg)


def parse_csv(text: str) -> list[str]:
    return p844.parse_csv(text)


def finite(value: Any, default: float = 0.0) -> float:
    return p844.finite(value, default)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_phase844_gears(model_name: str, round_name: str, top_n: int) -> list[dict[str, Any]]:
    path = PHASE844_ROOT / round_name / f"phase844_{model_name}_summary.json"
    if not path.exists():
        raise FileNotFoundError(f"missing Phase 844 summary: {path}")
    data = read_json(path)
    gears = [dict(row) for row in data.get("top_gears", [])[: int(top_n)]]
    for idx, gear in enumerate(gears, 1):
        gear["gear_rank"] = idx
        gear["gear_key"] = f"L{gear['layer_idx']}C{gear['channel_id']}"
    return gears


def combo_key(gears: list[dict[str, Any]]) -> str:
    return "+".join(str(g.get("gear_key") or f"L{g['layer_idx']}C{g['channel_id']}") for g in gears)


def combo_specs(gears: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = [{"spec_name": "original", "combo_type": "original", "mode": "original", "gears": []}]
    modes = [m for m in parse_csv(args.edit_modes) if m != "original"]
    pairs = list(itertools.combinations(gears, 2))[: int(args.max_pairs)]
    triplets: list[tuple[dict[str, Any], ...]] = []
    if args.include_triplets:
        triplets = list(itertools.combinations(gears, 3))[: int(args.max_triplets)]
    raw: list[tuple[str, tuple[dict[str, Any], ...]]] = []
    if "single" in parse_csv(args.combo_types):
        raw += [("single", (gear,)) for gear in gears]
    if "pair" in parse_csv(args.combo_types):
        raw += [("pair", pair) for pair in pairs]
    if "triplet" in parse_csv(args.combo_types):
        raw += [("triplet", triplet) for triplet in triplets]
    for combo_type, group in raw:
        group_list = [dict(g) for g in group]
        key = combo_key(group_list)
        for mode in modes:
            specs.append(
                {
                    "spec_name": f"{combo_type}_{mode}_{key}",
                    "combo_type": combo_type,
                    "mode": mode,
                    "gears": group_list,
                    "combo_key": key,
                    "gear_keys": [str(g.get("gear_key")) for g in group_list],
                }
            )
    return specs


def clean_margin(scores: dict[str, Any]) -> float | None:
    value = scores.get("target_minus_object_logit")
    if value is None:
        return None
    f = finite(value)
    return f if math.isfinite(f) else None


def eval_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = p844.selected_cases(args)
    gears = load_phase844_gears(args.model, args.phase844_round, int(args.top_gears))
    if args.dry_run:
        print(
            json.dumps(
                {
                    "phase": PHASE,
                    "model": args.model,
                    "round": args.round_name,
                    "cases": [case["case_id"] for case in cases],
                    "gears": [g["gear_key"] for g in gears],
                    "specs": len(combo_specs(gears, args)),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return {}

    model, tokenizer, device, attn_impl = p844.p828.p796.load_model_bf16_prefer_flash(
        args.model, args.attn_implementations
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    n_layers = len(get_layers(model))
    valid_gears = [g for g in gears if 0 <= int(g["layer_idx"]) < n_layers]
    if len(valid_gears) != len(gears):
        log(f"{args.model}/{args.round_name}: filtered gears {len(gears)} -> {len(valid_gears)} for n_layers={n_layers}")
    specs = combo_specs(valid_gears, args)
    rows: list[dict[str, Any]] = []
    standards = p844.p828.p820.standard_rows()
    prompt_variants = parse_csv(args.prompt_variants)
    try:
        for case_idx, case in enumerate(cases, 1):
            for variant in prompt_variants:
                prompt = p844.prompt_for_case(case, variant)
                prompt_ids = p844.encode_prompt(tokenizer, prompt)

                original_text, original_ids = p844.greedy_with_gears(
                    model, tokenizer, device, prompt_ids, [], "original", int(args.max_new_tokens)
                )
                original_boundary = p844.classify_output(case, original_text, standards)
                baseline_id = int(original_ids[0]) if original_ids else None
                original_logits = p844.first_logits_with_gears(model, device, prompt_ids, [], "original")
                original_scores = p844.token_scores(tokenizer, original_logits, case, baseline_id)
                original_margin = clean_margin(original_scores)
                single_delta_by_mode_key: dict[tuple[str, str], float] = {}

                local_rows: list[dict[str, Any]] = []
                for spec in specs:
                    if spec["combo_type"] == "original":
                        generated = original_text
                        token_ids = original_ids
                        logits = original_logits
                        boundary = original_boundary
                    else:
                        logits = p844.first_logits_with_gears(
                            model, device, prompt_ids, spec["gears"], spec["mode"]
                        )
                        generated, token_ids = p844.greedy_with_gears(
                            model,
                            tokenizer,
                            device,
                            prompt_ids,
                            spec["gears"],
                            spec["mode"],
                            int(args.max_new_tokens),
                        )
                        boundary = p844.classify_output(case, generated, standards)
                    scores = p844.token_scores(tokenizer, logits, case, baseline_id)
                    margin = clean_margin(scores)
                    delta = None if margin is None or original_margin is None else float(margin - original_margin)
                    gear_keys = [str(g.get("gear_key")) for g in spec.get("gears", [])]
                    if spec["combo_type"] == "single" and delta is not None:
                        single_delta_by_mode_key[(str(spec["mode"]), gear_keys[0])] = delta
                    row = {
                        "row_kind": "phase845_geometry_gear_interaction_edge_atlas",
                        "phase": PHASE,
                        "model": args.model,
                        "round": args.round_name,
                        "phase844_round": args.phase844_round,
                        "case_id": case["case_id"],
                        "object": case.get("object"),
                        "synthetic_case": bool(case.get("synthetic_case")),
                        "prompt_variant": variant,
                        "prompt": prompt,
                        "combo_type": spec["combo_type"],
                        "spec_name": spec["spec_name"],
                        "edit_mode": spec["mode"],
                        "gear_count": len(spec.get("gears", [])),
                        "gear_keys": gear_keys,
                        "combo_key": combo_key(spec.get("gears", [])) if spec.get("gears") else "original",
                        "generated": p844.p828.p825.clean_generated(generated),
                        "token_ids": token_ids,
                        "boundary_class": boundary.get("final_boundary_class"),
                        "boundary_rank": int(boundary.get("boundary_rank", 0)),
                        "target_transition": boundary.get("final_boundary_class") == TARGET_CLASS,
                        "original_generated": p844.p828.p825.clean_generated(original_text),
                        "original_boundary_class": original_boundary.get("final_boundary_class"),
                        "original_boundary_rank": int(original_boundary.get("boundary_rank", 0)),
                        "original_target_transition": original_boundary.get("final_boundary_class") == TARGET_CLASS,
                        "target_lost_vs_original": bool(
                            original_boundary.get("final_boundary_class") == TARGET_CLASS
                            and boundary.get("final_boundary_class") != TARGET_CLASS
                        ),
                        "target_gained_vs_original": bool(
                            original_boundary.get("final_boundary_class") != TARGET_CLASS
                            and boundary.get("final_boundary_class") == TARGET_CLASS
                        ),
                        "original_target_minus_object_logit": original_margin,
                        "margin_delta_vs_original": delta,
                        **scores,
                    }
                    local_rows.append(row)

                for row in local_rows:
                    if row["combo_type"] in {"pair", "triplet"} and row.get("margin_delta_vs_original") is not None:
                        expected = 0.0
                        missing = False
                        for key in row["gear_keys"]:
                            val = single_delta_by_mode_key.get((str(row["edit_mode"]), str(key)))
                            if val is None:
                                missing = True
                                break
                            expected += val
                        if not missing:
                            residual = float(row["margin_delta_vs_original"]) - expected
                            row["expected_additive_delta"] = expected
                            row["interaction_residual"] = residual
                            threshold = float(args.interaction_threshold)
                            if residual >= threshold:
                                row["interaction_class"] = "synergy"
                            elif residual <= -threshold:
                                row["interaction_class"] = "antagonistic"
                            else:
                                row["interaction_class"] = "additive"
                rows.extend(local_rows)
            if case_idx % int(args.log_every) == 0 or case_idx == len(cases):
                log(f"{args.model}: evaluated cases {case_idx}/{len(cases)} rows={len(rows)}")
    finally:
        p844.p828.release_model(model)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = summarize_rows(rows, valid_gears, specs, cases, args, attn_impl)
    p844.p828.write_jsonl(out_dir / f"phase845_{args.model}_rows.jsonl", rows)
    p844.p828.write_json(out_dir / f"phase845_{args.model}_summary.json", summary)
    print(
        json.dumps(
            {
                "model": args.model,
                "round": args.round_name,
                "gears": summary["n_gears"],
                "specs": summary["n_specs"],
                "rows": summary["n_rows"],
                "target_gained_vs_original_rows": summary["target_gained_vs_original_rows"],
                "target_lost_vs_original_rows": summary["target_lost_vs_original_rows"],
                "synergy_rows": summary["interaction_class_summary"].get("synergy", {}).get("n", 0),
                "antagonistic_rows": summary["interaction_class_summary"].get("antagonistic", {}).get("n", 0),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return summary


def avg(vals: list[float]) -> float | None:
    return sum(vals) / len(vals) if vals else None


def compact(vals: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(vals),
        "target_rows": sum(1 for row in vals if row.get("target_transition")),
        "target_lost_vs_original_rows": sum(1 for row in vals if row.get("target_lost_vs_original")),
        "target_gained_vs_original_rows": sum(1 for row in vals if row.get("target_gained_vs_original")),
        "object_echo_rows": sum(1 for row in vals if row.get("boundary_class") == "object_echo"),
        "unknown_other_rows": sum(1 for row in vals if row.get("boundary_class") == "unknown_other"),
        "mean_margin_delta": avg(
            [finite(row.get("margin_delta_vs_original")) for row in vals if row.get("margin_delta_vs_original") is not None]
        ),
        "mean_interaction_residual": avg(
            [finite(row.get("interaction_residual")) for row in vals if row.get("interaction_residual") is not None]
        ),
        "classes": dict(Counter(str(row.get("boundary_class")) for row in vals)),
    }


def edge_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_edge: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("combo_type") not in {"pair", "triplet"}:
            continue
        by_edge[(str(row.get("combo_type")), str(row.get("edit_mode")), str(row.get("combo_key")))].append(row)
    out: list[dict[str, Any]] = []
    for (combo_type, mode, key), vals in by_edge.items():
        residuals = [finite(row.get("interaction_residual")) for row in vals if row.get("interaction_residual") is not None]
        classes = Counter(str(row.get("interaction_class", "missing")) for row in vals)
        out.append(
            {
                "combo_type": combo_type,
                "edit_mode": mode,
                "combo_key": key,
                "n": len(vals),
                "synergy_rows": classes.get("synergy", 0),
                "antagonistic_rows": classes.get("antagonistic", 0),
                "additive_rows": classes.get("additive", 0),
                "target_gained_vs_original_rows": sum(1 for row in vals if row.get("target_gained_vs_original")),
                "target_lost_vs_original_rows": sum(1 for row in vals if row.get("target_lost_vs_original")),
                "mean_residual": avg(residuals),
                "mean_abs_residual": avg([abs(x) for x in residuals]),
            }
        )
    out.sort(
        key=lambda row: (
            int(row.get("synergy_rows", 0)) + int(row.get("antagonistic_rows", 0)),
            finite(row.get("mean_abs_residual")),
        ),
        reverse=True,
    )
    return out


def summarize_rows(
    rows: list[dict[str, Any]],
    gears: list[dict[str, Any]],
    specs: list[dict[str, Any]],
    cases: list[dict[str, Any]],
    args: argparse.Namespace,
    attn_impl: str | None,
) -> dict[str, Any]:
    by_combo: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_interaction: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_object: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_combo[str(row.get("combo_type"))].append(row)
        by_object[str(row.get("object"))].append(row)
        if row.get("combo_type") in {"pair", "triplet"}:
            by_interaction[str(row.get("interaction_class", "missing"))].append(row)
    original_rows = [row for row in rows if row.get("combo_type") == "original"]
    edges = edge_summary(rows)
    top_rows = sorted(
        [row for row in rows if row.get("combo_type") in {"pair", "triplet"}],
        key=lambda row: (
            abs(finite(row.get("interaction_residual"))),
            int(bool(row.get("target_gained_vs_original"))) + int(bool(row.get("target_lost_vs_original"))),
        ),
        reverse=True,
    )[:100]
    return {
        "phase": PHASE,
        "title": "Geometry Gear Interaction Edge Atlas",
        "model": args.model,
        "round": args.round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "dtype": "bfloat16",
        "quantization": "off",
        "phase844_round": args.phase844_round,
        "n_cases": len(cases),
        "case_ids": [case["case_id"] for case in cases],
        "prompt_variants": parse_csv(args.prompt_variants),
        "n_gears": len(gears),
        "top_gears": gears,
        "n_specs": len(specs),
        "n_rows": len(rows),
        "original_target_rows": sum(1 for row in original_rows if row.get("target_transition")),
        "target_rows": sum(1 for row in rows if row.get("target_transition")),
        "target_lost_vs_original_rows": sum(1 for row in rows if row.get("target_lost_vs_original")),
        "target_gained_vs_original_rows": sum(1 for row in rows if row.get("target_gained_vs_original")),
        "combo_type_summary": {k: compact(v) for k, v in sorted(by_combo.items())},
        "interaction_class_summary": {k: compact(v) for k, v in sorted(by_interaction.items())},
        "object_summary": {k: compact(v) for k, v in sorted(by_object.items())},
        "edge_summary": edges[: int(args.report_top_edges)],
        "top_rows": [
            {
                "object": row.get("object"),
                "prompt_variant": row.get("prompt_variant"),
                "combo_type": row.get("combo_type"),
                "edit_mode": row.get("edit_mode"),
                "combo_key": row.get("combo_key"),
                "boundary_class": row.get("boundary_class"),
                "original_boundary_class": row.get("original_boundary_class"),
                "target_gained_vs_original": bool(row.get("target_gained_vs_original")),
                "target_lost_vs_original": bool(row.get("target_lost_vs_original")),
                "margin_delta_vs_original": row.get("margin_delta_vs_original"),
                "expected_additive_delta": row.get("expected_additive_delta"),
                "interaction_residual": row.get("interaction_residual"),
                "interaction_class": row.get("interaction_class"),
                "generated": row.get("generated"),
                "top_tokens": row.get("top_tokens"),
            }
            for row in top_rows
        ],
        "boundary": (
            "This phase measures whether Phase 844 geometry gears combine additively, synergistically, "
            "or antagonistically. It is an interaction-edge atlas probe, not token closure."
        ),
    }


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{finite(value):.4f}"


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# Phase 845 Geometry Gear Interaction Edge Atlas ({payload['round']})",
        "",
        "- Search: pair/triplet interaction residuals over Phase 844 top geometry gears.",
        "- Boundary: interaction-edge atlas probe; not token closure.",
        "",
        "## Model Summary",
        "",
        "| model | gears | specs | rows | original target | target | lost | gained | synergy | antagonistic | additive |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model_name in p844.p828.MODELS:
        data = payload.get("model_summaries", {}).get(model_name) or {}
        classes = data.get("interaction_class_summary") or {}
        lines.append(
            f"| {model_name} | {data.get('n_gears', 0)} | {data.get('n_specs', 0)} | {data.get('n_rows', 0)} | "
            f"{data.get('original_target_rows', 0)} | {data.get('target_rows', 0)} | "
            f"{data.get('target_lost_vs_original_rows', 0)} | {data.get('target_gained_vs_original_rows', 0)} | "
            f"{classes.get('synergy', {}).get('n', 0)} | {classes.get('antagonistic', {}).get('n', 0)} | "
            f"{classes.get('additive', {}).get('n', 0)} |"
        )

    lines += ["", "## Top Gears", ""]
    lines += ["| model | rank | layer | channel | score | neg ratio |"]
    lines += ["|---|---:|---:|---:|---:|---:|"]
    for model_name in p844.p828.MODELS:
        data = payload.get("model_summaries", {}).get(model_name) or {}
        for idx, gear in enumerate(data.get("top_gears") or [], 1):
            lines.append(
                f"| {model_name} | {idx} | {gear.get('layer_idx')} | {gear.get('channel_id')} | "
                f"{fmt(gear.get('gear_score'))} | {fmt(gear.get('neg_ratio'))} |"
            )

    lines += ["", "## Combo Type Summary", ""]
    lines += ["| model | combo | n | target | lost | gained | mean delta | mean residual | classes |"]
    lines += ["|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for model_name in p844.p828.MODELS:
        data = payload.get("model_summaries", {}).get(model_name) or {}
        for combo, row in (data.get("combo_type_summary") or {}).items():
            lines.append(
                f"| {model_name} | `{combo}` | {row.get('n', 0)} | {row.get('target_rows', 0)} | "
                f"{row.get('target_lost_vs_original_rows', 0)} | {row.get('target_gained_vs_original_rows', 0)} | "
                f"{fmt(row.get('mean_margin_delta'))} | {fmt(row.get('mean_interaction_residual'))} | "
                f"`{json.dumps(row.get('classes') or {}, ensure_ascii=False)}` |"
            )

    lines += ["", "## Interaction Class Summary", ""]
    lines += ["| model | class | n | target | lost | gained | mean residual | classes |"]
    lines += ["|---|---|---:|---:|---:|---:|---:|---|"]
    for model_name in p844.p828.MODELS:
        data = payload.get("model_summaries", {}).get(model_name) or {}
        for cls, row in (data.get("interaction_class_summary") or {}).items():
            lines.append(
                f"| {model_name} | `{cls}` | {row.get('n', 0)} | {row.get('target_rows', 0)} | "
                f"{row.get('target_lost_vs_original_rows', 0)} | {row.get('target_gained_vs_original_rows', 0)} | "
                f"{fmt(row.get('mean_interaction_residual'))} | "
                f"`{json.dumps(row.get('classes') or {}, ensure_ascii=False)}` |"
            )

    lines += ["", "## Edge Summary", ""]
    lines += ["| model | type | mode | combo | n | synergy | antagonistic | additive | gained | lost | mean residual | mean abs residual |"]
    lines += ["|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for model_name in p844.p828.MODELS:
        data = payload.get("model_summaries", {}).get(model_name) or {}
        for row in data.get("edge_summary") or []:
            lines.append(
                f"| {model_name} | `{row.get('combo_type')}` | `{row.get('edit_mode')}` | `{row.get('combo_key')}` | "
                f"{row.get('n', 0)} | {row.get('synergy_rows', 0)} | {row.get('antagonistic_rows', 0)} | "
                f"{row.get('additive_rows', 0)} | {row.get('target_gained_vs_original_rows', 0)} | "
                f"{row.get('target_lost_vs_original_rows', 0)} | {fmt(row.get('mean_residual'))} | "
                f"{fmt(row.get('mean_abs_residual'))} |"
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
    for model_name in p844.p828.MODELS:
        path = out_dir / f"phase845_{model_name}_summary.json"
        if path.exists():
            payload["model_summaries"][model_name] = read_json(path)
            payload["models"].append(model_name)
    payload["status"] = "complete" if len(payload["models"]) == len(p844.p828.MODELS) else "partial"
    p844.p828.write_json(out_dir / "phase845_cross_model_summary.json", payload)
    write_markdown(out_dir / "phase845_cross_model_summary.md", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=p844.p828.MODELS)
    parser.add_argument("--round-name", default="smoke")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--phase844-round", default="confirm")
    parser.add_argument("--include-seed-triangle", action="store_true")
    parser.add_argument("--geometry-objects", default="triangle,square,rectangle,circle")
    parser.add_argument("--max-cases", type=int, default=4)
    parser.add_argument("--prompt-variants", default="natural_question,natural_category")
    parser.add_argument("--top-gears", type=int, default=6)
    parser.add_argument("--combo-types", default="single,pair")
    parser.add_argument("--include-triplets", action="store_true")
    parser.add_argument("--max-pairs", type=int, default=15)
    parser.add_argument("--max-triplets", type=int, default=6)
    parser.add_argument("--edit-modes", default="zero,flip")
    parser.add_argument("--interaction-threshold", type=float, default=0.5)
    parser.add_argument("--max-new-tokens", type=int, default=6)
    parser.add_argument("--attn-implementations", default="flash_attention_2,sdpa,eager")
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--report-top-edges", type=int, default=30)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.summarize_only:
        print(json.dumps(summarize_round(args.round_name), ensure_ascii=False, indent=2), flush=True)
        return
    eval_model(args)


if __name__ == "__main__":
    main()
