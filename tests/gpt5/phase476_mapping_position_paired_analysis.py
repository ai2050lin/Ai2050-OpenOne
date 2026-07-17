#!/usr/bin/env python3
"""Phase476 paired analysis for Phase475 mapping-position scalar precheck."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
ROWS_PATH = ROOT / "tests" / "gpt5" / "result" / "phase475_mapping_position_scalar_precheck" / "phase475_mapping_position_scalar_rows.jsonl"
OUT_DIR = ROOT / "tests" / "gpt5" / "result" / "phase476_mapping_position_paired_analysis"
OUT_PATH = OUT_DIR / "phase476_mapping_position_paired_analysis.json"

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
    keyed = {(row["source_sample_id"], row["label_mapping"], row["role"], int(row["layer_index"])): row for row in rows}
    out = []
    samples = sorted({row["source_sample_id"] for row in rows})
    roles = sorted({row["role"] for row in rows})
    layers = sorted({int(row["layer_index"]) for row in rows})
    for sample_id in samples:
        for role in roles:
            for layer in layers:
                ab = keyed.get((sample_id, "mu_ab", role, layer))
                ba = keyed.get((sample_id, "mu_ba", role, layer))
                if ab is None or ba is None:
                    continue
                item = {
                    "source_sample_id": sample_id,
                    "role": role,
                    "layer_index": layer,
                    "layer_family": family_for(layer),
                    "truth_value": ab["truth_value"],
                    "mu_ab_classification": ab["classification"],
                    "mu_ba_classification": ba["classification"],
                    "mu_ab_behavior_truth": ab["behavior_truth"],
                    "mu_ba_behavior_truth": ba["behavior_truth"],
                }
                for scalar in SCALARS:
                    item[f"delta_{scalar}"] = float(ba[scalar]) - float(ab[scalar])
                    denom = (abs(float(ba[scalar])) + abs(float(ab[scalar]))) / 2 + 1e-9
                    item[f"rel_delta_{scalar}"] = item[f"delta_{scalar}"] / denom
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
            vals = [row[f"delta_{scalar}"] for row in items]
            rels = [row[f"rel_delta_{scalar}"] for row in items]
            item[f"mean_delta_{scalar}"] = mean(vals)
            item[f"mean_abs_delta_{scalar}"] = mean(abs(value) for value in vals)
            item[f"mean_rel_delta_{scalar}"] = mean(rels)
        out.append(item)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(ROWS_PATH)
    deltas = paired_deltas(rows)
    prompt_rows = [row for row in rows if row["layer_index"] == 0 and row["role"] == "terminal_token"]
    behavior = Counter((row["label_mapping"], row["truth_value"], row["classification"], row["behavior_truth"]) for row in prompt_rows)
    out = {
        "schema_version": "phase476_mapping_position_paired_analysis.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "analysis_only_no_model_run_no_cuda",
        "input_trace": str(ROWS_PATH.relative_to(ROOT)),
        "paired_delta_count": len(deltas),
        "behavior_counts": {str(key): value for key, value in behavior.items()},
        "primary_by_role_family": summarize(deltas, ["role", "layer_family"]),
        "by_truth_role_family": summarize(deltas, ["truth_value", "role", "layer_family"]),
        "by_behavior_truth_role_family": summarize(deltas, ["mu_ab_behavior_truth", "mu_ba_behavior_truth", "role", "layer_family"]),
        "interpretation": {
            "strict_mapping_pairing_used": True,
            "main_analysis": "mu_ba minus mu_ab on evidence-first prompts for identical logical samples.",
            "allowed_claim": "Label mapping changes can be compared by role and layer family in scalar summaries.",
            "forbidden_claim": "No relation-state invariance, label-state equivariance, head, neuron, or causal claim is authorized.",
            "next_step": "If continuing, add an A/B candidate-margin readout and component ledger before any stronger physical claim.",
        },
    }
    OUT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(OUT_PATH)


if __name__ == "__main__":
    main()
