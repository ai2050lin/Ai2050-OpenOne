#!/usr/bin/env python3
"""Phase478 paired analysis for Phase477 A/B candidate-margin readout."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
ROWS_PATH = ROOT / "tests" / "gpt5" / "result" / "phase477_candidate_margin_readout" / "phase477_candidate_margin_rows.jsonl"
OUT_DIR = ROOT / "tests" / "gpt5" / "result" / "phase478_candidate_margin_paired_analysis"
OUT_PATH = OUT_DIR / "phase478_candidate_margin_paired_analysis.json"

MARGINS = ("margin_ab", "margin_true", "margin_correct")
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


def paired_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
                for margin in MARGINS:
                    item[f"mu_ab_{margin}"] = float(ab[margin])
                    item[f"mu_ba_{margin}"] = float(ba[margin])
                    item[f"delta_{margin}"] = float(ba[margin]) - float(ab[margin])
                item["raw_margin_flip_residual"] = float(ab["margin_ab"]) + float(ba["margin_ab"])
                item["true_margin_stability_delta"] = float(ba["margin_true"]) - float(ab["margin_true"])
                item["correct_margin_stability_delta"] = float(ba["margin_correct"]) - float(ab["margin_correct"])
                item["raw_margin_flip_aligned"] = abs(item["raw_margin_flip_residual"]) < abs(float(ab["margin_ab"])) + abs(float(ba["margin_ab"]))
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
        for margin in MARGINS:
            item[f"mean_mu_ab_{margin}"] = mean(row[f"mu_ab_{margin}"] for row in items)
            item[f"mean_mu_ba_{margin}"] = mean(row[f"mu_ba_{margin}"] for row in items)
            item[f"mean_delta_{margin}"] = mean(row[f"delta_{margin}"] for row in items)
        item["mean_abs_raw_margin_flip_residual"] = mean(abs(row["raw_margin_flip_residual"]) for row in items)
        item["mean_true_margin_stability_delta"] = mean(row["true_margin_stability_delta"] for row in items)
        item["mean_correct_margin_stability_delta"] = mean(row["correct_margin_stability_delta"] for row in items)
        item["raw_margin_flip_aligned_rate"] = mean(1.0 if row["raw_margin_flip_aligned"] else 0.0 for row in items)
        out.append(item)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(ROWS_PATH)
    paired = paired_rows(rows)
    prompt_rows = [row for row in rows if row["layer_index"] == 0 and row["role"] == "terminal_token"]
    behavior = Counter((row["label_mapping"], row["truth_value"], row["classification"], row["behavior_truth"]) for row in prompt_rows)
    out = {
        "schema_version": "phase478_candidate_margin_paired_analysis.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "analysis_only_no_model_run_no_cuda",
        "input_trace": str(ROWS_PATH.relative_to(ROOT)),
        "paired_margin_count": len(paired),
        "behavior_counts": {str(key): value for key, value in behavior.items()},
        "primary_by_role_family": summarize(paired, ["role", "layer_family"]),
        "by_truth_role_family": summarize(paired, ["truth_value", "role", "layer_family"]),
        "by_behavior_truth_role_family": summarize(paired, ["mu_ab_behavior_truth", "mu_ba_behavior_truth", "role", "layer_family"]),
        "interpretation": {
            "strict_mapping_pairing_used": True,
            "readout_scope": "final norm plus lm_head logit lens is an external observer readout.",
            "allowed_claim": "A/B candidate margins can be compared across mapping reversal at matched roles and layers.",
            "forbidden_claim": "No internal relation-state, component, head, neuron, or causal claim is authorized.",
            "next_step": "Use truth contrast before label instruction and component ledger before any stronger physical mechanism claim.",
        },
    }
    OUT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(OUT_PATH)


if __name__ == "__main__":
    main()
