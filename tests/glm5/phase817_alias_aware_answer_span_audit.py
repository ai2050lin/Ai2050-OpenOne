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

from phase722_functional_head_atlas_causal_ablation import write_json, write_jsonl  # noqa: E402
from phase735_source_restricted_writer_validation import MODELS  # noqa: E402
from phase776_readout_bridge_competition_audit import normalize_token_text  # noqa: E402


PHASE = 817
SOURCE_ROOT = Path("tests/result/phase816_multi_token_answer_span_rollout_closure")
RESULT_ROOT = Path("tests/result/phase817_alias_aware_answer_span_audit")


ALIASES: dict[str, list[str]] = {
    "p816_cat_living_thing": ["living thing", "animal", "mammal", "domestic animal", "pet", "pet category"],
    "p816_hammer_hand_tool": ["hand tool", "tool", "tools", "tool category"],
    "p816_bus_public_transport": ["public transport", "public transportation", "bus transport", "transport vehicle"],
    "p816_guitar_musical_instrument": ["musical instrument", "instrument", "music instrument"],
    "p816_rose_flowering_plant": ["flowering plant", "flower", "flowers", "plant", "flower category"],
    "p816_chair_household_furniture": ["household furniture", "furniture", "furniture category"],
    "p816_apple_edible_fruit": ["edible fruit", "fruit", "fruit category"],
    "p816_heart_body_organ": ["body organ", "organ", "body part", "human body part"],
    "p816_red_warm_color": ["warm color", "color", "color category"],
    "p816_salmon_aquatic_animal": ["aquatic animal", "fish", "freshwater fish", "salmon fish"],
    "p816_oak_tall_tree": ["tall tree", "tree", "hardwood tree", "tree species"],
    "p816_carrot_root_vegetable": ["root vegetable", "vegetable", "vegetables"],
    "p816_laptop_electronic_device": ["electronic device", "electronics", "computer", "personal computer", "personal computing device"],
    "p816_spoon_eating_utensil": ["eating utensil", "kitchen utensil", "utensil"],
    "p816_triangle_geometric_shape": ["geometric shape", "geometry shape", "shape"],
    "p816_winter_cold_season": ["cold season", "season", "cold weather"],
    "p816_gold_precious_metal": ["precious metal", "metal"],
    "p816_oxygen_chemical_element": ["chemical element", "element"],
    "p816_cactus_desert_plant": ["desert plant", "cactus plant", "cactus plants", "plant"],
    "p816_doctor_medical_worker": ["medical worker", "medical professional", "health professional", "healthcare worker", "doctor"],
}


def log(msg: str) -> None:
    print(f"[phase817] {msg}", flush=True)


def norm_text(value: Any) -> str:
    text = normalize_token_text("" if value is None else str(value)).strip().lower()
    # Keep this intentionally simple; no stemming or embedding similarity.
    for ch in ['"', "'", "`", ".", ",", ";", ":", "[", "]", "(", ")"]:
        text = text.replace(ch, " ")
    return " ".join(text.split())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def alias_match(generated: Any, aliases: list[str]) -> tuple[bool, str | None, str]:
    gen = norm_text(generated)
    if not gen:
        return False, None, "empty"
    for alias in aliases:
        alias_norm = norm_text(alias)
        if gen == alias_norm:
            return True, alias, "exact_alias"
    for alias in aliases:
        alias_norm = norm_text(alias)
        if gen.startswith(alias_norm) or alias_norm.startswith(gen):
            return True, alias, "prefix_alias"
    return False, None, "no_alias"


def audit_row(row: dict[str, Any]) -> dict[str, Any]:
    aliases = ALIASES.get(str(row.get("case_id")), [str(row.get("target_answer") or "")])
    matched, alias, match_kind = alias_match(row.get("generated_clean"), aliases)
    exact = bool(row.get("rollout_closure"))
    span_score = bool(row.get("span_score_closure"))
    exact_full = bool(row.get("full_span_rollout_closure"))
    alias_rollout = exact or matched
    alias_full = span_score and alias_rollout and bool(row.get("contrast_span_cleared")) and bool(row.get("generic_blocker_cleared"))
    if exact_full:
        label = "exact_span_and_rollout_closed"
    elif alias_full:
        label = "alias_rollout_rescues_span_score"
    elif alias_rollout and not span_score:
        label = "alias_rollout_without_span_score"
    elif span_score and not alias_rollout:
        label = "span_score_closed_alias_rollout_not_closed"
    else:
        label = "alias_unclosed"
    return {
        "row_kind": "phase817_alias_aware_answer_span_audit",
        "phase": PHASE,
        "source_phase": 816,
        "model": row.get("model"),
        "round": row.get("round"),
        "case_id": row.get("case_id"),
        "prompt_variant": row.get("prompt_variant"),
        "target_answer": row.get("target_answer"),
        "generated_clean": row.get("generated_clean"),
        "generated_norm": norm_text(row.get("generated_clean")),
        "aliases": aliases,
        "matched_alias": alias,
        "alias_match_kind": match_kind,
        "exact_rollout_closure": exact,
        "alias_rollout_closure": alias_rollout,
        "span_score_closure": span_score,
        "contrast_span_cleared": bool(row.get("contrast_span_cleared")),
        "generic_blocker_cleared": bool(row.get("generic_blocker_cleared")),
        "exact_full_span_rollout_closure": exact_full,
        "alias_full_span_rollout_closure": alias_full,
        "target_margin_vs_non_target_mean_logprob": row.get("target_margin_vs_non_target_mean_logprob"),
        "best_target": row.get("best_target"),
        "best_non_target": row.get("best_non_target"),
        "phase816_label": row.get("phase816_label"),
        "phase817_label": label,
    }


def load_rows(round_name: str, model_name: str) -> list[dict[str, Any]]:
    path = SOURCE_ROOT / round_name / f"phase816_{model_name}_rows.jsonl"
    return [row for row in read_jsonl(path) if row.get("row_kind") == "phase816_multi_token_answer_span_rollout"]


def summarize(rows: list[dict[str, Any]], model_name: str, round_name: str) -> dict[str, Any]:
    by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_prompt[str(row.get("prompt_variant"))].append(row)
    prompt_summary = {}
    for prompt, vals in by_prompt.items():
        prompt_summary[prompt] = {
            "n": len(vals),
            "exact_rollout_rows": sum(1 for row in vals if row.get("exact_rollout_closure")),
            "alias_rollout_rows": sum(1 for row in vals if row.get("alias_rollout_closure")),
            "exact_full_rows": sum(1 for row in vals if row.get("exact_full_span_rollout_closure")),
            "alias_full_rows": sum(1 for row in vals if row.get("alias_full_span_rollout_closure")),
            "by_label": dict(Counter(row.get("phase817_label") for row in vals)),
        }
    return {
        "phase": PHASE,
        "title": "Alias Aware Answer Span Audit",
        "model": model_name,
        "round": round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_rows": len(rows),
        "exact_rollout_rows": sum(1 for row in rows if row.get("exact_rollout_closure")),
        "alias_rollout_rows": sum(1 for row in rows if row.get("alias_rollout_closure")),
        "exact_full_span_rollout_rows": sum(1 for row in rows if row.get("exact_full_span_rollout_closure")),
        "alias_full_span_rollout_rows": sum(1 for row in rows if row.get("alias_full_span_rollout_closure")),
        "alias_gain_rollout": sum(1 for row in rows if row.get("alias_rollout_closure")) - sum(1 for row in rows if row.get("exact_rollout_closure")),
        "alias_gain_full": sum(1 for row in rows if row.get("alias_full_span_rollout_closure"))
        - sum(1 for row in rows if row.get("exact_full_span_rollout_closure")),
        "by_label": dict(Counter(row.get("phase817_label") for row in rows)),
        "by_prompt_variant": prompt_summary,
        "rescued_rows": [
            row
            for row in rows
            if row.get("alias_rollout_closure") and not row.get("exact_rollout_closure")
        ][:80],
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# Phase 817 Alias Aware Answer Span Audit ({payload['round']})",
        "",
        "- Source: Phase 816 saved rollout rows; no new model forward pass.",
        "- Boundary: tests whether exact phrase rollout undercounts semantically acceptable multi-token answers.",
        "",
        "## Model Summary",
        "",
        "| model | rows | exact rollout | alias rollout | exact full | alias full | rollout gain | full gain | labels |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for model_name in MODELS:
        data = payload.get("model_summaries", {}).get(model_name)
        if not data:
            continue
        lines.append(
            f"| {model_name} | {data.get('n_rows')} | {data.get('exact_rollout_rows')} | "
            f"{data.get('alias_rollout_rows')} | {data.get('exact_full_span_rollout_rows')} | "
            f"{data.get('alias_full_span_rollout_rows')} | {data.get('alias_gain_rollout')} | "
            f"{data.get('alias_gain_full')} | `{json.dumps(data.get('by_label') or {}, ensure_ascii=False)}` |"
        )
    lines += ["", "## Prompt Variant Summary", ""]
    lines += ["| model | prompt | n | exact rollout | alias rollout | exact full | alias full | labels |", "|---|---|---:|---:|---:|---:|---:|---|"]
    for model_name in MODELS:
        data = payload.get("model_summaries", {}).get(model_name)
        if not data:
            continue
        for prompt, row in sorted((data.get("by_prompt_variant") or {}).items()):
            lines.append(
                f"| {model_name} | {prompt} | {row.get('n')} | {row.get('exact_rollout_rows')} | "
                f"{row.get('alias_rollout_rows')} | {row.get('exact_full_rows')} | {row.get('alias_full_rows')} | "
                f"`{json.dumps(row.get('by_label') or {}, ensure_ascii=False)}` |"
            )
    lines += ["", "## Rescued Rows", ""]
    lines += ["| model | prompt | case | target | generated | alias | label |", "|---|---|---|---|---|---|---|"]
    for model_name in MODELS:
        data = payload.get("model_summaries", {}).get(model_name)
        if not data:
            continue
        for row in data.get("rescued_rows", [])[:30]:
            lines.append(
                f"| {model_name} | {row.get('prompt_variant')} | {row.get('case_id')} | `{row.get('target_answer')}` | "
                f"`{row.get('generated_clean')}` | `{row.get('matched_alias')}` | `{row.get('phase817_label')}` |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_round(round_name: str) -> dict[str, Any]:
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
        source = load_rows(round_name, model_name)
        if not source:
            continue
        log(f"{round_name}/{model_name}: audit {len(source)} phase816 rows")
        rows = [audit_row(row) for row in source]
        summary = summarize(rows, model_name, round_name)
        write_jsonl(out_dir / f"phase817_{model_name}_rows.jsonl", rows)
        write_json(out_dir / f"phase817_{model_name}_summary.json", summary)
        payload["model_summaries"][model_name] = summary
        payload["models"].append(model_name)
    payload["status"] = "complete" if len(payload["models"]) == len(MODELS) else "partial"
    write_json(out_dir / "phase817_cross_model_summary.json", payload)
    write_markdown(out_dir / "phase817_cross_model_summary.md", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", default="smoke,main,confirm")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    results = {}
    for round_name in [x.strip() for x in args.rounds.split(",") if x.strip()]:
        results[round_name] = run_round(round_name)
    print(json.dumps({"phase": PHASE, "rounds": {k: v["status"] for k, v in results.items()}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
