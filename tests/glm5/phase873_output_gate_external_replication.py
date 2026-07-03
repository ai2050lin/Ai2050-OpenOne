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


PHASE = 873
MODELS = p846.MODELS
RESULT_ROOT = Path("tests/result/phase873_output_gate_external_replication")


def replication_cases() -> list[dict[str, Any]]:
    raw = [
        ("animal", "tiger", ["animal", "living thing", "living creature", "creature", "mammal"]),
        ("animal", "dolphin", ["animal", "living thing", "living creature", "creature", "mammal"]),
        ("animal", "sparrow", ["animal", "living thing", "living creature", "creature", "bird"]),
        ("animal", "turtle", ["animal", "living thing", "living creature", "creature"]),
        ("animal", "goat", ["animal", "living thing", "living creature", "creature", "mammal"]),
        ("animal", "bear", ["animal", "living thing", "living creature", "creature", "mammal"]),
        ("material", "glass", ["material", "substance", "matter"]),
        ("material", "steel", ["material", "substance", "matter", "metal"]),
        ("material", "wool", ["material", "substance", "matter", "fabric"]),
        ("material", "clay", ["material", "substance", "matter"]),
        ("material", "plastic", ["material", "substance", "matter"]),
        ("material", "sand", ["material", "substance", "matter"]),
        ("color", "red", ["color", "hue", "colour"]),
        ("color", "green", ["color", "hue", "colour"]),
        ("color", "yellow", ["color", "hue", "colour"]),
        ("color", "pink", ["color", "hue", "colour"]),
        ("color", "silver", ["color", "hue", "colour"]),
        ("color", "beige", ["color", "hue", "colour"]),
    ]
    return [
        {
            "case_id": f"p873_{idx:03d}_{domain}_{obj}",
            "domain": domain,
            "object": obj,
            "answer_aliases": aliases,
            "canonical_answer": aliases[0],
            "overlap_kind": "output_gate_replication_member_not_alias",
        }
        for idx, (domain, obj, aliases) in enumerate(raw, 1)
    ]


def replication_prompt(case: dict[str, Any], variant: str) -> str:
    obj = case["object"]
    if variant == "replication_direct":
        return f"Give one lowercase category word only.\n{obj}:"
    if variant == "replication_sentence":
        return f"Complete with the broad category.\n{obj} is a type of"
    if variant == "replication_form":
        return f"Name: {obj}\nType:"
    raise ValueError(f"unknown replication prompt variant: {variant}")


def configure_phase867_backend() -> None:
    p867.PHASE = PHASE
    p867.RESULT_ROOT = RESULT_ROOT
    p867.holdout_cases = replication_cases
    p867.prompt_for_holdout = replication_prompt


def rename_outputs(model_name: str, round_name: str) -> None:
    out_dir = RESULT_ROOT / round_name
    mapping = {
        out_dir / f"phase867_{model_name}_summary.json": out_dir / f"phase873_{model_name}_summary.json",
        out_dir / f"phase867_{model_name}_rows.jsonl": out_dir / f"phase873_{model_name}_rows.jsonl",
        out_dir / f"phase867_{model_name}_effects.jsonl": out_dir / f"phase873_{model_name}_effects.jsonl",
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
        "title": "Output Gate External Replication",
        "round": round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "missing",
        "models": [],
        "model_summaries": {},
    }
    effects: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    for model_name in MODELS:
        path = out_dir / f"phase873_{model_name}_summary.json"
        if not path.exists():
            continue
        summary = read_json(path)
        payload["models"].append(model_name)
        payload["model_summaries"][model_name] = summary
        effects.extend(summary.get("holdout_effects") or [])
        status_counts[str(summary.get("status"))] += 1
    payload["status"] = "complete" if len(payload["models"]) == len(MODELS) else "partial"
    payload["model_status_counts"] = dict(status_counts)
    payload["overall_source_to_holdout_clean_stats"] = binary_stats(effects)
    p846.write_json(out_dir / "phase873_cross_model_summary.json", payload)
    write_markdown(out_dir / "phase873_cross_model_summary.md", payload)
    return payload


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# Phase 873 Output Gate External Replication ({payload['round']})",
        "",
        "- Source: Phase 865 candidates, new objects and prompts.",
        "- Boundary: data collection for Phase 872 output gate rule replication.",
        "",
        "| model | status | candidates | domains | source-clean -> holdout-clean stats |",
        "|---|---|---:|---|---|",
    ]
    for model_name in MODELS:
        summary = payload.get("model_summaries", {}).get(model_name) or {}
        lines.append(
            f"| {model_name} | {summary.get('status', 'missing')} | {summary.get('candidate_count', 0)} | "
            f"`{summary.get('domains', [])}` | `{summary.get('source_to_holdout_clean_stats', {})}` |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="replication")
    parser.add_argument("--max-cases-per-domain", type=int, default=6)
    parser.add_argument("--prompt-variants", default="replication_direct,replication_sentence,replication_form")
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
