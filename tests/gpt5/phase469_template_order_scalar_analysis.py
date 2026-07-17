#!/usr/bin/env python3
"""Phase469 analysis for Phase468 template-order scalar precheck."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
ROWS_PATH = ROOT / "tests" / "gpt5" / "result" / "phase468_template_order_physical_precheck" / "phase468_template_order_physical_scalar_rows.jsonl"
OUT_DIR = ROOT / "tests" / "gpt5" / "result" / "phase469_template_order_scalar_analysis"
OUT_PATH = OUT_DIR / "phase469_template_order_scalar_analysis.json"

SUCCESS_TRANSFORMS = {"factor_plain_anchor", "factor_semicolon_only"}
FAILURE_TRANSFORM = "factor_claim_first_only"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def avg(rows: list[dict[str, Any]], key: str) -> float | None:
    if not rows:
        return None
    return mean(float(row[key]) for row in rows)


def layer_contrasts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    layers = sorted({int(row["layer_index"]) for row in rows})
    out = []
    for layer in layers:
        layer_rows = [row for row in rows if int(row["layer_index"]) == layer]
        success_b = [
            row for row in layer_rows
            if row["transform"] in SUCCESS_TRANSFORMS and row["expected_label"] == "B" and row["classification"] == "semantic"
        ]
        failure_b_wrong = [
            row for row in layer_rows
            if row["transform"] == FAILURE_TRANSFORM and row["expected_label"] == "B" and row["classification"] == "wrong"
        ]
        success_all = [
            row for row in layer_rows
            if row["transform"] in SUCCESS_TRANSFORMS and row["classification"] == "semantic"
        ]
        failure_all = [
            row for row in layer_rows
            if row["transform"] == FAILURE_TRANSFORM
        ]
        item = {
            "layer_index": layer,
            "success_b_n": len(success_b),
            "failure_b_wrong_n": len(failure_b_wrong),
            "success_b_mean_l2": avg(success_b, "last_token_l2"),
            "failure_b_wrong_mean_l2": avg(failure_b_wrong, "last_token_l2"),
            "success_b_mean_abs": avg(success_b, "last_token_abs_mean"),
            "failure_b_wrong_mean_abs": avg(failure_b_wrong, "last_token_abs_mean"),
            "success_all_mean_l2": avg(success_all, "last_token_l2"),
            "failure_all_mean_l2": avg(failure_all, "last_token_l2"),
        }
        if item["success_b_mean_l2"] is not None and item["failure_b_wrong_mean_l2"] is not None:
            item["b_wrong_minus_success_l2"] = item["failure_b_wrong_mean_l2"] - item["success_b_mean_l2"]
        if item["success_b_mean_abs"] is not None and item["failure_b_wrong_mean_abs"] is not None:
            item["b_wrong_minus_success_abs"] = item["failure_b_wrong_mean_abs"] - item["success_b_mean_abs"]
        if item["success_all_mean_l2"] is not None and item["failure_all_mean_l2"] is not None:
            item["failure_minus_success_all_l2"] = item["failure_all_mean_l2"] - item["success_all_mean_l2"]
        out.append(item)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(ROWS_PATH)
    prompt_rows = [row for row in rows if int(row["layer_index"]) == 0]
    behavior = Counter((row["transform"], row["expected_label"], row["classification"]) for row in prompt_rows)
    contrasts = layer_contrasts(rows)
    ranked_l2 = sorted(
        [row for row in contrasts if "b_wrong_minus_success_l2" in row],
        key=lambda row: abs(row["b_wrong_minus_success_l2"]),
        reverse=True,
    )[:10]
    ranked_abs = sorted(
        [row for row in contrasts if "b_wrong_minus_success_abs" in row],
        key=lambda row: abs(row["b_wrong_minus_success_abs"]),
        reverse=True,
    )[:10]
    out = {
        "schema_version": "phase469_template_order_scalar_analysis.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "analysis_only_no_model_run_no_cuda",
        "input_trace": str(ROWS_PATH.relative_to(ROOT)),
        "prompt_count": len(prompt_rows),
        "trace_row_count": len(rows),
        "behavior_counts": {str(key): value for key, value in behavior.items()},
        "layer_count": len({row["layer_index"] for row in rows}),
        "top_b_wrong_minus_success_l2_layers": ranked_l2,
        "top_b_wrong_minus_success_abs_layers": ranked_abs,
        "interpretation": {
            "scalar_precheck_observed": True,
            "claim_first_failure_has_physical_scalar_contrast": bool(ranked_l2),
            "allowed_claim": "Scalar hidden-state summaries differ between frozen successful templates and claim-first B failures.",
            "forbidden_claim": "No causal, head, neuron, or semantic truth-state claim is authorized by these scalar summaries.",
            "next_step": "If continuing, collect a vector-free position-resolved diagnostic for evidence tokens, claim tokens and terminal token on the same frozen 96 prompts.",
        },
    }
    OUT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(OUT_PATH)


if __name__ == "__main__":
    main()
