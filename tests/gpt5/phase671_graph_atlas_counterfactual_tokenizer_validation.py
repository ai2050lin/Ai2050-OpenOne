#!/usr/bin/env python3
"""
Phase 671: Graph Atlas Counterfactual Tokenizer Validation.

This phase loads local tokenizers only. It does not load model weights or run
generation. The goal is to validate the Phase 670 counterfactual control set
before expensive cross-model inference.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))

from model_registry import all_model_keys, get_model_spec  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402


CONTROL_PATH = Path(
    "results/glm5_phase670_graph_atlas_counterfactual_control_set/phase670_counterfactual_control_set.json"
)
OUT_ROOT = Path("results/glm5_phase671_graph_atlas_counterfactual_tokenizer_validation")


def load_tokenizer(model_key: str) -> Any:
    spec = get_model_spec(model_key)
    if not spec.local_dir.exists():
        raise FileNotFoundError(f"Missing local model dir: {spec.local_dir}")
    tok = AutoTokenizer.from_pretrained(
        str(spec.local_dir),
        trust_remote_code=spec.trust_remote_code,
        local_files_only=True,
        use_fast=False,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def encode(tokenizer: Any, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def token_text(tokenizer: Any, ids: list[int]) -> list[str]:
    return [tokenizer.decode([tid]) for tid in ids]


def validate_model(model_key: str, max_prompt_tokens: int) -> dict:
    data = json.loads(CONTROL_PATH.read_text(encoding="utf-8"))
    cases = data["cases"]
    pairs = data["pairs"]
    by_id = {c["case_id"]: c for c in cases}
    tokenizer = load_tokenizer(model_key)

    case_rows = []
    invalid_cases = []
    for c in cases:
        prompt_ids = encode(tokenizer, c["prompt"])
        out_ids = encode(tokenizer, c["expected_output"])
        row = {
            "case_id": c["case_id"],
            "family": c["family"],
            "axis": c["axis"],
            "format_name": c["format_name"],
            "prompt_tokens": len(prompt_ids),
            "expected_tokens": len(out_ids),
            "first_expected_id": out_ids[0] if out_ids else None,
            "first_expected_text": tokenizer.decode([out_ids[0]]) if out_ids else "",
            "expected_token_text": token_text(tokenizer, out_ids[:8]),
        }
        errors = []
        if not out_ids:
            errors.append("empty_expected_output_tokens")
        if len(prompt_ids) > max_prompt_tokens:
            errors.append("prompt_too_long")
        if len(out_ids) > 24:
            errors.append("expected_output_too_long")
        row["errors"] = errors
        if errors:
            invalid_cases.append(row)
        case_rows.append(row)

    pair_rows = []
    invalid_pairs = []
    same_prefix_valid = 0
    for p in pairs:
        left = by_id[p["left_case_id"]]
        right = by_id[p["right_case_id"]]
        left_ids = encode(tokenizer, left["expected_output"])
        right_ids = encode(tokenizer, right["expected_output"])
        row = {
            "pair_id": p["pair_id"],
            "family": p["family"],
            "isolated_factor": p["isolated_factor"],
            "left_case_id": p["left_case_id"],
            "right_case_id": p["right_case_id"],
            "left_expected_tokens": len(left_ids),
            "right_expected_tokens": len(right_ids),
            "same_first_token": bool(left_ids and right_ids and left_ids[0] == right_ids[0]),
            "diverges_after_first": bool(
                left_ids and right_ids and left_ids[0] == right_ids[0] and left_ids != right_ids
            ),
            "left_token_text": token_text(tokenizer, left_ids[:8]),
            "right_token_text": token_text(tokenizer, right_ids[:8]),
        }
        errors = []
        if p["family"] == "same_prefix_different_continuation":
            if not row["same_first_token"]:
                errors.append("same_prefix_pair_does_not_share_first_token")
            if not row["diverges_after_first"]:
                errors.append("same_prefix_pair_does_not_diverge_after_first")
            if not errors:
                same_prefix_valid += 1
        if p["family"] == "different_value_same_format" and left["format_name"] != right["format_name"]:
            errors.append("different_value_pair_format_mismatch")
        if p["family"] == "same_value_different_format" and left["value"] != right["value"]:
            errors.append("same_value_pair_value_mismatch")
        row["errors"] = errors
        if errors:
            invalid_pairs.append(row)
        pair_rows.append(row)

    family_counts: dict[str, int] = {}
    prompt_lengths: list[int] = []
    output_lengths: list[int] = []
    for r in case_rows:
        family_counts[r["family"]] = family_counts.get(r["family"], 0) + 1
        prompt_lengths.append(r["prompt_tokens"])
        output_lengths.append(r["expected_tokens"])

    summary = {
        "model": model_key,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_cases": len(cases),
        "n_pairs": len(pairs),
        "invalid_case_count": len(invalid_cases),
        "invalid_pair_count": len(invalid_pairs),
        "same_prefix_pair_count": sum(1 for p in pairs if p["family"] == "same_prefix_different_continuation"),
        "same_prefix_valid_pair_count": same_prefix_valid,
        "max_prompt_tokens": max(prompt_lengths) if prompt_lengths else 0,
        "max_expected_tokens": max(output_lengths) if output_lengths else 0,
        "mean_prompt_tokens": sum(prompt_lengths) / max(1, len(prompt_lengths)),
        "mean_expected_tokens": sum(output_lengths) / max(1, len(output_lengths)),
        "case_family_counts": dict(sorted(family_counts.items())),
        "status": "pass" if not invalid_cases and not invalid_pairs else "needs_filtering",
    }
    return {
        "phase": 671,
        "title": "Graph Atlas Counterfactual Tokenizer Validation",
        "summary": summary,
        "invalid_cases": invalid_cases[:200],
        "invalid_pairs": invalid_pairs[:200],
        "case_tokenization_sample": case_rows[:40],
        "pair_tokenization_sample": pair_rows[:80],
    }


def write_cross_model_summary() -> dict:
    model_summaries = []
    for model in all_model_keys():
        path = OUT_ROOT / f"phase671_{model}_tokenizer_validation_confirm.json"
        if path.exists():
            model_summaries.append(json.loads(path.read_text(encoding="utf-8"))["summary"])
    status = "pass" if model_summaries and all(s["status"] == "pass" for s in model_summaries) else "needs_filtering"
    payload = {
        "phase": 671,
        "title": "Graph Atlas Counterfactual Tokenizer Validation Cross-Model Summary",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": status,
        "models": model_summaries,
        "strict_next_step": (
            "If status is pass, run Phase 672 natural trajectory audit. "
            "If status is needs_filtering, generate a filtered Phase 670B control set."
        ),
    }
    md = [
        "# Phase 671 Graph Atlas Counterfactual Tokenizer Validation",
        "",
        f"- generated: `{payload['timestamp']}`",
        f"- status: `{payload['status']}`",
        "",
        "| model | status | cases | pairs | invalid_cases | invalid_pairs | same_prefix_valid/total | max_prompt_tokens |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for s in model_summaries:
        md.append(
            f"| {s['model']} | {s['status']} | {s['n_cases']} | {s['n_pairs']} | "
            f"{s['invalid_case_count']} | {s['invalid_pair_count']} | "
            f"{s['same_prefix_valid_pair_count']}/{s['same_prefix_pair_count']} | {s['max_prompt_tokens']} |"
        )
    md += [
        "",
        "## Interpretation",
        "",
        "- A pass means the prompt-level controls are tokenizer-safe for the current audit.",
        "- Same-prefix continuation controls are strict: they must share the first expected token and diverge later.",
        "- Writer topology and residual-boundary nodes still require later internal activation tests.",
        "",
    ]
    (OUT_ROOT / "phase671_cross_model_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT_ROOT / "phase671_cross_model_summary.md").write_text("\n".join(md), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=all_model_keys())
    parser.add_argument("--max-prompt-tokens", type=int, default=512)
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    if args.summarize_only:
        payload = write_cross_model_summary()
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return
    if not args.model:
        raise SystemExit("--model is required unless --summarize-only is set")

    payload = validate_model(args.model, args.max_prompt_tokens)
    out_path = OUT_ROOT / f"phase671_{args.model}_tokenizer_validation_confirm.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}", flush=True)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2), flush=True)
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
