#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase846_geometry_boundary_equation_fitting as p846  # noqa: E402
import phase867_clean_route_holdout_prediction as p867  # noqa: E402


PHASE = 876
MODELS = p846.MODELS
RESULT_ROOT = Path("tests/result/phase876_nonclean_route_causal_validation")


def validation_cases() -> list[dict[str, Any]]:
    raw = [
        ("animal", "seal", ["animal", "living thing", "living creature", "creature", "mammal"], "semantic_pressure_probe"),
        ("animal", "bat", ["animal", "living thing", "living creature", "creature", "mammal"], "semantic_pressure_probe"),
        ("animal", "salmon", ["animal", "living thing", "living creature", "creature", "fish"], "semantic_pressure_probe"),
        ("animal", "turkey", ["animal", "living thing", "living creature", "creature", "bird"], "semantic_pressure_probe"),
        ("animal", "sheep", ["animal", "living thing", "living creature", "creature", "mammal"], "object_echo_probe"),
        ("animal", "wolf", ["animal", "living thing", "living creature", "creature", "mammal"], "object_echo_probe"),
        ("material", "wood", ["material", "substance", "matter"], "clean_counter_probe"),
        ("material", "brick", ["material", "substance", "matter"], "clean_counter_probe"),
        ("material", "copper", ["material", "substance", "matter", "metal"], "clean_counter_probe"),
        ("material", "nylon", ["material", "substance", "matter", "fabric"], "clean_counter_probe"),
        ("material", "granite", ["material", "substance", "matter"], "clean_counter_probe"),
        ("material", "wax", ["material", "substance", "matter"], "clean_counter_probe"),
        ("color", "gold", ["color", "hue", "colour"], "semantic_pressure_probe"),
        ("color", "violet", ["color", "hue", "colour"], "semantic_pressure_probe"),
        ("color", "teal", ["color", "hue", "colour"], "format_probe"),
        ("color", "maroon", ["color", "hue", "colour"], "format_probe"),
        ("color", "navy", ["color", "hue", "colour"], "object_echo_probe"),
        ("color", "ivory", ["color", "hue", "colour"], "object_echo_probe"),
    ]
    return [
        {
            "case_id": f"p876_{idx:03d}_{domain}_{obj}",
            "domain": domain,
            "object": obj,
            "answer_aliases": aliases,
            "canonical_answer": aliases[0],
            "overlap_kind": route_probe,
        }
        for idx, (domain, obj, aliases, route_probe) in enumerate(raw, 1)
    ]


def validation_prompt(case: dict[str, Any], variant: str) -> str:
    obj = case["object"]
    if variant == "nonclean_direct":
        return f"Give one lowercase broad category word only.\nItem: {obj}\nCategory:"
    if variant == "semantic_pressure":
        return f"In a broad taxonomy, the best general category for {obj} is"
    if variant == "echo_pressure":
        return f"Do not repeat the item. Item = {obj}\nThe category is"
    if variant == "format_pressure":
        return f"item,category\n{obj},"
    raise ValueError(f"unknown validation prompt variant: {variant}")


def configure_phase867_backend() -> None:
    p867.PHASE = PHASE
    p867.RESULT_ROOT = RESULT_ROOT
    p867.holdout_cases = validation_cases
    p867.prompt_for_holdout = validation_prompt


def rename_outputs(model_name: str, round_name: str) -> None:
    out_dir = RESULT_ROOT / round_name
    mapping = {
        out_dir / f"phase867_{model_name}_summary.json": out_dir / f"phase876_{model_name}_summary.json",
        out_dir / f"phase867_{model_name}_rows.jsonl": out_dir / f"phase876_{model_name}_rows.jsonl",
        out_dir / f"phase867_{model_name}_effects.jsonl": out_dir / f"phase876_{model_name}_effects.jsonl",
    }
    for src, dst in mapping.items():
        if src.exists():
            if dst.exists():
                dst.unlink()
            src.rename(dst)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def binary_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tp = sum(1 for row in rows if row.get("source_predict_clean_mixed") and row.get("holdout_clean_mixed"))
    fp = sum(1 for row in rows if row.get("source_predict_clean_mixed") and not row.get("holdout_clean_mixed"))
    fn = sum(1 for row in rows if not row.get("source_predict_clean_mixed") and row.get("holdout_clean_mixed"))
    tn = sum(1 for row in rows if not row.get("source_predict_clean_mixed") and not row.get("holdout_clean_mixed"))
    n = len(rows)
    return {
        "n": n,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "accuracy": (tp + tn) / n if n else 0.0,
        "source_clean_count": sum(1 for row in rows if row.get("source_predict_clean_mixed")),
        "holdout_clean_count": sum(1 for row in rows if row.get("holdout_clean_mixed")),
    }


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    payload: dict[str, Any] = {
        "phase": PHASE,
        "title": "Nonclean Route Causal Validation",
        "round": round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "missing",
        "models": [],
        "model_summaries": {},
    }
    effects: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    domain_counts: Counter[str] = Counter()
    source_purity_counts: Counter[str] = Counter()
    for model_name in MODELS:
        path = out_dir / f"phase876_{model_name}_summary.json"
        if not path.exists():
            continue
        summary = read_json(path)
        payload["models"].append(model_name)
        payload["model_summaries"][model_name] = summary
        status_counts[str(summary.get("status"))] += 1
        for domain in summary.get("domains") or []:
            domain_counts[f"{model_name}:{domain}"] += 1
        for row in summary.get("holdout_effects") or []:
            effects.append(row)
            source_purity_counts[str(row.get("source_purity_class"))] += 1
    payload["status"] = "complete" if len(payload["models"]) == len(MODELS) else "partial"
    payload["model_status_counts"] = dict(status_counts)
    payload["model_domain_presence"] = dict(domain_counts)
    payload["source_purity_counts"] = dict(source_purity_counts)
    payload["overall_source_to_holdout_clean_stats"] = binary_stats(effects)
    payload["validation_boundary"] = (
        "New-object/new-prompt causal validation for Phase 875 nonclean route candidates. "
        "The raw model rows still need Phase 870/872/874/875 post-processing before any route-family claim."
    )
    p846.write_json(out_dir / "phase876_cross_model_summary.json", payload)
    write_markdown(out_dir / "phase876_cross_model_summary.md", payload)
    return payload


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# Phase 876 Nonclean Route Causal Validation ({payload['round']})",
        "",
        "- Source: Phase 865 gear candidates with new Phase 876 objects and route-pressure prompts.",
        "- Boundary: causal validation input for Phase 870/872/874/875; not closure.",
        "",
        "| model | status | candidates | domains | source-clean -> validation-clean stats |",
        "|---|---|---:|---|---|",
    ]
    for model_name in MODELS:
        summary = payload.get("model_summaries", {}).get(model_name) or {}
        lines.append(
            f"| {model_name} | {summary.get('status', 'missing')} | {summary.get('candidate_count', 0)} | "
            f"`{summary.get('domains', [])}` | `{summary.get('source_to_holdout_clean_stats', {})}` |"
        )
    lines += [
        "",
        f"- Source purity counts: `{payload.get('source_purity_counts', {})}`",
        f"- Overall stats: `{payload.get('overall_source_to_holdout_clean_stats', {})}`",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="validation")
    parser.add_argument("--max-cases-per-domain", type=int, default=6)
    parser.add_argument(
        "--prompt-variants",
        default="nonclean_direct,semantic_pressure,echo_pressure,format_pressure",
    )
    parser.add_argument("--include-non-clean-controls", action="store_true")
    parser.add_argument("--object-delta-threshold", type=float, default=0.25)
    parser.add_argument("--scale-up-factor", type=float, default=2.0)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--topk-tokens", type=int, default=20)
    parser.add_argument("--topk-blockers", type=int, default=10)
    parser.add_argument("--attn-implementations", default="flash_attention_2,sdpa")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize-round", action="store_true")
    return parser


def main() -> None:
    configure_phase867_backend()
    args = build_parser().parse_args()
    if args.summarize_round:
        payload = summarize_round(args.round_name)
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return
    if not args.model:
        raise SystemExit("--model is required unless --summarize-round is set")
    p867.eval_model(args)
    rename_outputs(args.model, args.round_name)


if __name__ == "__main__":
    main()
