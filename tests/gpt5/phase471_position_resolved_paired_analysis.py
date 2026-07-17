#!/usr/bin/env python3
"""Phase471 paired analysis for Phase470 position-resolved scalar precheck."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
ROWS_PATH = ROOT / "tests" / "gpt5" / "result" / "phase470_position_resolved_scalar_precheck" / "phase470_position_resolved_scalar_rows.jsonl"
OUT_DIR = ROOT / "tests" / "gpt5" / "result" / "phase471_position_resolved_paired_analysis"
OUT_PATH = OUT_DIR / "phase471_position_resolved_paired_analysis.json"

BASELINE_TRANSFORMS = ("factor_plain_anchor", "factor_semicolon_only")
CLAIM_FIRST = "factor_claim_first_only"
SCALARS = ("mean_token_l2", "mean_vector_l2", "mean_abs_mean", "mean_signed_mean")
LAYER_FAMILIES = {
    "early": range(0, 9),
    "mid_front": range(9, 21),
    "mid_back": range(21, 33),
    "late": range(33, 40),
    "final": range(40, 41),
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def family_for(layer: int) -> str:
    for name, layers in LAYER_FAMILIES.items():
        if layer in layers:
            return name
    raise ValueError(layer)


def paired_deltas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keyed = {(row["sample_id"], row["transform"], row["role"], int(row["layer_index"])): row for row in rows}
    out = []
    samples = sorted({row["sample_id"] for row in rows})
    roles = sorted({row["role"] for row in rows})
    layers = sorted({int(row["layer_index"]) for row in rows})
    for sample_id in samples:
        for role in roles:
            for layer in layers:
                claim = keyed.get((sample_id, CLAIM_FIRST, role, layer))
                if claim is None:
                    continue
                for baseline in BASELINE_TRANSFORMS:
                    base = keyed.get((sample_id, baseline, role, layer))
                    if base is None:
                        continue
                    item = {
                        "sample_id": sample_id,
                        "baseline_transform": baseline,
                        "role": role,
                        "layer_index": layer,
                        "layer_family": family_for(layer),
                        "expected_label": claim["expected_label"],
                        "pair_role": claim["pair_role"],
                        "claim_first_classification": claim["classification"],
                        "baseline_classification": base["classification"],
                    }
                    for scalar in SCALARS:
                        item[f"delta_{scalar}"] = float(claim[scalar]) - float(base[scalar])
                    out.append(item)
    return out


def summarize(rows: list[dict[str, Any]], fields: list[str]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[tuple(row[field] for field in fields)].append(row)
    out = []
    for key, items in sorted(buckets.items()):
        item = {field: value for field, value in zip(fields, key, strict=True)}
        item["n"] = len(items)
        for scalar in SCALARS:
            values = [row[f"delta_{scalar}"] for row in items]
            item[f"mean_delta_{scalar}"] = mean(values)
            item[f"mean_abs_delta_{scalar}"] = mean(abs(value) for value in values)
        out.append(item)
    return out


def top_single_layer(rows: list[dict[str, Any]], scalar: str) -> list[dict[str, Any]]:
    grouped = summarize(rows, ["role", "layer_index"])
    return sorted(grouped, key=lambda row: abs(row[f"mean_delta_{scalar}"]), reverse=True)[:12]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(ROWS_PATH)
    deltas = paired_deltas(rows)
    prompt_rows = [row for row in rows if row["layer_index"] == 0 and row["role"] == "terminal_token"]
    behavior = Counter((row["transform"], row["expected_label"], row["classification"]) for row in prompt_rows)
    out = {
        "schema_version": "phase471_position_resolved_paired_analysis.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "analysis_only_no_model_run_no_cuda",
        "input_trace": str(ROWS_PATH.relative_to(ROOT)),
        "paired_delta_count": len(deltas),
        "behavior_counts": {str(key): value for key, value in behavior.items()},
        "primary_no_outcome_filter_by_role_family": summarize(deltas, ["role", "layer_family"]),
        "primary_no_outcome_filter_by_role_family_baseline": summarize(deltas, ["baseline_transform", "role", "layer_family"]),
        "secondary_by_claim_first_outcome": summarize(deltas, ["claim_first_classification", "expected_label", "role", "layer_family"]),
        "top_single_layer_mean_vector_l2": top_single_layer(deltas, "mean_vector_l2"),
        "top_single_layer_mean_abs": top_single_layer(deltas, "mean_abs_mean"),
        "interpretation": {
            "strict_pairing_used": True,
            "main_analysis_conditioned_on_outcome": False,
            "allowed_claim": "Claim-first versus evidence-first scalar differences can be compared by role and layer family on identical logical samples.",
            "forbidden_claim": "No relation truth state, attention head function, neuron function or causal component is authorized.",
            "missing_from_full_phase470_plan": [
                "label_mapping_reversal",
                "component ledger for attention and MLP writes",
                "A/B candidate margin readout",
                "independent physical prediction holdout",
            ],
        },
    }
    OUT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(OUT_PATH)


if __name__ == "__main__":
    main()
