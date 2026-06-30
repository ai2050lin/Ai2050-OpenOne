#!/usr/bin/env python3
"""Phase 777: recompute open closure with semantic-equivalent surface forms.

This script does not load models. It consumes Phase 776 observation rows and
counts top1 target case variants as semantic-equivalent open closure.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PHASE776_ROOT = Path("tests/result/phase776_readout_bridge_competition_audit")
PHASE777_ROOT = Path("tests/result/phase777_semantic_equivalent_open_closure_recompute")
MODELS = ("qwen3", "glm4", "deepseek7b")
TARGET_EQUIV_CLASSES = {"target_value", "case_variant_target_value"}


def norm_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text.lower()


def load_observations(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("row_kind") == "readout_bridge_observation":
                rows.append(row)
    return rows


def rate(num: int, den: int) -> float:
    return float(num) / float(den) if den else 0.0


def semantic_equiv_open(row: dict[str, Any]) -> bool:
    if bool(row.get("base_target_top1")):
        return True
    cls = row.get("top1_competitor_class")
    if cls in TARGET_EQUIV_CLASSES:
        return True
    return norm_text(row.get("top1_token_text_norm")) == norm_text(row.get("target_answer"))


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    strict_n = sum(bool(r.get("base_target_top1")) for r in rows)
    pool_n = sum(bool(r.get("pool_target_top1")) for r in rows)
    equiv_n = sum(semantic_equiv_open(r) for r in rows)
    surface_gain_n = sum((not bool(r.get("base_target_top1"))) and semantic_equiv_open(r) for r in rows)
    latent_n = sum((not bool(r.get("base_target_top1"))) and bool(r.get("pool_target_top1")) for r in rows)
    hard_readout_n = sum((not semantic_equiv_open(r)) and bool(r.get("pool_target_top1")) for r in rows)
    pool_missing_n = sum(not bool(r.get("pool_target_top1")) for r in rows)
    strict_fail_n = n - strict_n

    return {
        "n": n,
        "strict_open_n": strict_n,
        "strict_open_rate": rate(strict_n, n),
        "semantic_equiv_open_n": equiv_n,
        "semantic_equiv_open_rate": rate(equiv_n, n),
        "surface_norm_gain_n": surface_gain_n,
        "surface_norm_gain_rate": rate(surface_gain_n, n),
        "surface_norm_gain_among_strict_fail_rate": rate(surface_gain_n, strict_fail_n),
        "pool_top1_n": pool_n,
        "pool_top1_rate": rate(pool_n, n),
        "latent_pool_hit_n": latent_n,
        "latent_pool_hit_rate": rate(latent_n, n),
        "hard_readout_failure_after_equiv_n": hard_readout_n,
        "hard_readout_failure_after_equiv_rate": rate(hard_readout_n, n),
        "pool_missing_n": pool_missing_n,
        "pool_missing_rate": rate(pool_missing_n, n),
    }


def build_summary(round_name: str, source_dir: Path) -> dict[str, Any]:
    model_summaries: dict[str, Any] = {}
    all_rows: list[dict[str, Any]] = []

    for model in MODELS:
        rows_path = source_dir / f"phase776_{model}_rows.jsonl"
        rows = load_observations(rows_path)
        all_rows.extend({**r, "model": model} for r in rows)

        by_prompt: list[dict[str, Any]] = []
        prompt_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            prompt_groups[str(row.get("prompt_variant"))].append(row)
        for prompt, prompt_rows in sorted(prompt_groups.items()):
            by_prompt.append({"prompt_variant": prompt, **summarize_group(prompt_rows)})

        latent_counter: Counter[str] = Counter()
        latent_equiv_counter: Counter[str] = Counter()
        for row in rows:
            if (not bool(row.get("base_target_top1"))) and bool(row.get("pool_target_top1")):
                prompt = str(row.get("prompt_variant"))
                latent_counter[prompt] += 1
                if semantic_equiv_open(row):
                    latent_equiv_counter[prompt] += 1

        model_summaries[model] = {
            "model": model,
            "round": round_name,
            "n_observations": len(rows),
            "overall": summarize_group(rows),
            "by_prompt": by_prompt,
            "latent_equiv_by_prompt": [
                {
                    "prompt_variant": prompt,
                    "latent_n": latent_counter[prompt],
                    "latent_semantic_equiv_open_n": latent_equiv_counter[prompt],
                    "latent_semantic_equiv_open_rate": rate(latent_equiv_counter[prompt], latent_counter[prompt]),
                }
                for prompt in sorted(latent_counter)
            ],
        }

    domain_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        domain_groups[(str(row.get("model")), str(row.get("domain")))].append(row)

    by_model_domain = [
        {"model": model, "domain": domain, **summarize_group(rows)}
        for (model, domain), rows in sorted(domain_groups.items())
    ]

    return {
        "phase": 777,
        "title": "Semantic-Equivalent Open Closure Recompute",
        "round": round_name,
        "source_phase": 776,
        "source_dir": str(source_dir),
        "target_equiv_classes": sorted(TARGET_EQUIV_CLASSES),
        "interpretation": (
            "Counts target_value and case_variant_target_value top1 tokens as semantic-equivalent open closure; "
            "this corrects strict token failures caused by surface-form mismatch."
        ),
        "by_model": model_summaries,
        "by_model_domain": by_model_domain,
    }


def fmt_float(value: float) -> str:
    return f"{value:.3f}"


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines: list[str] = []
    lines.append(f"# Phase 777 Semantic-Equivalent Open Closure Recompute ({summary['round']})")
    lines.append("")
    lines.append(f"- Source: `{summary['source_dir']}`")
    lines.append("- Model loading: not used; this is a Phase 776 result recomputation.")
    lines.append("- Semantic-equivalent classes: `target_value`, `case_variant_target_value`.")
    lines.append("")

    lines.append("## By Prompt")
    lines.append("")
    lines.append(
        "| model | variant | n | strict open | semantic-equivalent open | gain | pool top1 | latent hit | hard readout after equiv |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for model, model_summary in summary["by_model"].items():
        for row in model_summary["by_prompt"]:
            lines.append(
                "| {model} | `{variant}` | {n} | {strict} | {equiv} | {gain} | {pool} | {latent} | {hard} |".format(
                    model=model,
                    variant=row["prompt_variant"],
                    n=row["n"],
                    strict=fmt_float(row["strict_open_rate"]),
                    equiv=fmt_float(row["semantic_equiv_open_rate"]),
                    gain=fmt_float(row["surface_norm_gain_rate"]),
                    pool=fmt_float(row["pool_top1_rate"]),
                    latent=fmt_float(row["latent_pool_hit_rate"]),
                    hard=fmt_float(row["hard_readout_failure_after_equiv_rate"]),
                )
            )
    lines.append("")

    lines.append("## Latent-Hit Reclassification")
    lines.append("")
    lines.append("| model | variant | latent n | latent semantic-equivalent open | rate |")
    lines.append("|---|---|---:|---:|---:|")
    for model, model_summary in summary["by_model"].items():
        for row in model_summary["latent_equiv_by_prompt"]:
            lines.append(
                "| {model} | `{variant}` | {latent_n} | {equiv_n} | {rate} |".format(
                    model=model,
                    variant=row["prompt_variant"],
                    latent_n=row["latent_n"],
                    equiv_n=row["latent_semantic_equiv_open_n"],
                    rate=fmt_float(row["latent_semantic_equiv_open_rate"]),
                )
            )
    lines.append("")

    lines.append("## By Domain")
    lines.append("")
    lines.append("| model | domain | n | strict open | semantic-equivalent open | gain | pool top1 | hard readout after equiv |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for row in summary["by_model_domain"]:
        lines.append(
            "| {model} | `{domain}` | {n} | {strict} | {equiv} | {gain} | {pool} | {hard} |".format(
                model=row["model"],
                domain=row["domain"],
                n=row["n"],
                strict=fmt_float(row["strict_open_rate"]),
                equiv=fmt_float(row["semantic_equiv_open_rate"]),
                gain=fmt_float(row["surface_norm_gain_rate"]),
                pool=fmt_float(row["pool_top1_rate"]),
                hard=fmt_float(row["hard_readout_failure_after_equiv_rate"]),
            )
        )
    lines.append("")

    lines.append("## Strict Interpretation")
    lines.append("")
    lines.append("- This is an evaluation correction, not a new causal intervention.")
    lines.append("- A semantic-equivalent open hit means the top1 token matches the target value after surface-form normalization.")
    lines.append("- Remaining hard readout failures are cases where the value pool is correct but top1 is not target-equivalent.")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round-name", default="main")
    parser.add_argument("--source-root", type=Path, default=PHASE776_ROOT)
    parser.add_argument("--output-root", type=Path, default=PHASE777_ROOT)
    args = parser.parse_args()

    source_dir = args.source_root / args.round_name
    if not source_dir.exists():
        raise FileNotFoundError(source_dir)

    out_dir = args.output_root / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_summary(args.round_name, source_dir)

    json_path = out_dir / "phase777_semantic_equivalent_open_closure_summary.json"
    md_path = out_dir / "phase777_semantic_equivalent_open_closure_summary.md"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(summary, md_path)
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
