#!/usr/bin/env python3
"""Phase479 pre-label truth contrast analysis.

Uses Phase477 logit-lens rows to compare true vs false claims before the label
instruction. This is analysis-only and does not load a model.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
ROWS_PATH = ROOT / "tests" / "gpt5" / "result" / "phase477_candidate_margin_readout" / "phase477_candidate_margin_rows.jsonl"
OUT_DIR = ROOT / "tests" / "gpt5" / "result" / "phase479_prelabel_truth_contrast_analysis"
OUT_PATH = OUT_DIR / "phase479_prelabel_truth_contrast_analysis.json"

PRELABEL_ROLES = ("evidence_span", "claim_span")
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


def truth_contrasts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keyed = {
        (
            row["source_pair_id"],
            row["label_mapping"],
            row["truth_value"],
            row["role"],
            int(row["layer_index"]),
        ): row
        for row in rows
        if row["role"] in PRELABEL_ROLES
    }
    out = []
    pair_ids = sorted({row["source_pair_id"] for row in rows})
    mappings = sorted({row["label_mapping"] for row in rows})
    layers = sorted({int(row["layer_index"]) for row in rows})
    for pair_id in pair_ids:
        for mapping in mappings:
            for role in PRELABEL_ROLES:
                for layer in layers:
                    true_row = keyed.get((pair_id, mapping, True, role, layer))
                    false_row = keyed.get((pair_id, mapping, False, role, layer))
                    if true_row is None or false_row is None:
                        continue
                    item = {
                        "source_pair_id": pair_id,
                        "label_mapping": mapping,
                        "role": role,
                        "layer_index": layer,
                        "layer_family": family_for(layer),
                        "true_behavior_truth": true_row["behavior_truth"],
                        "false_behavior_truth": false_row["behavior_truth"],
                    }
                    for margin in MARGINS:
                        item[f"true_{margin}"] = float(true_row[margin])
                        item[f"false_{margin}"] = float(false_row[margin])
                        item[f"truth_delta_{margin}"] = float(true_row[margin]) - float(false_row[margin])
                    out.append(item)
    return out


def mapping_stability(contrasts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keyed = {
        (row["source_pair_id"], row["label_mapping"], row["role"], int(row["layer_index"])): row
        for row in contrasts
    }
    out = []
    pair_ids = sorted({row["source_pair_id"] for row in contrasts})
    roles = sorted({row["role"] for row in contrasts})
    layers = sorted({int(row["layer_index"]) for row in contrasts})
    for pair_id in pair_ids:
        for role in roles:
            for layer in layers:
                ab = keyed.get((pair_id, "mu_ab", role, layer))
                ba = keyed.get((pair_id, "mu_ba", role, layer))
                if ab is None or ba is None:
                    continue
                item = {
                    "source_pair_id": pair_id,
                    "role": role,
                    "layer_index": layer,
                    "layer_family": family_for(layer),
                }
                for margin in MARGINS:
                    item[f"mu_ab_truth_delta_{margin}"] = float(ab[f"truth_delta_{margin}"])
                    item[f"mu_ba_truth_delta_{margin}"] = float(ba[f"truth_delta_{margin}"])
                    item[f"mapping_stability_delta_{margin}"] = (
                        float(ba[f"truth_delta_{margin}"]) - float(ab[f"truth_delta_{margin}"])
                    )
                out.append(item)
    return out


def summarize(rows: list[dict[str, Any]], fields: list[str], prefix: str = "") -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[tuple(row[field] for field in fields)].append(row)
    out = []
    for key, items in sorted(buckets.items()):
        item = {field: value for field, value in zip(fields, key, strict=True)}
        item["n"] = len(items)
        for margin in MARGINS:
            truth_key = f"{prefix}truth_delta_{margin}" if prefix else f"truth_delta_{margin}"
            if truth_key in items[0]:
                vals = [row[truth_key] for row in items]
                item[f"mean_{truth_key}"] = mean(vals)
                item[f"mean_abs_{truth_key}"] = mean(abs(value) for value in vals)
            stability_key = f"mapping_stability_delta_{margin}"
            if stability_key in items[0]:
                vals = [row[stability_key] for row in items]
                item[f"mean_{stability_key}"] = mean(vals)
                item[f"mean_abs_{stability_key}"] = mean(abs(value) for value in vals)
        out.append(item)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(ROWS_PATH)
    contrasts = truth_contrasts(rows)
    stability = mapping_stability(contrasts)
    out = {
        "schema_version": "phase479_prelabel_truth_contrast_analysis.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "analysis_only_no_model_run_no_cuda",
        "input_trace": str(ROWS_PATH.relative_to(ROOT)),
        "prelabel_roles": list(PRELABEL_ROLES),
        "truth_contrast_count": len(contrasts),
        "mapping_stability_count": len(stability),
        "truth_contrast_by_mapping_role_family": summarize(contrasts, ["label_mapping", "role", "layer_family"]),
        "truth_contrast_by_role_family": summarize(contrasts, ["role", "layer_family"]),
        "mapping_stability_by_role_family": summarize(stability, ["role", "layer_family"]),
        "interpretation": {
            "allowed_claim": "Pre-label true/false readout differences can be described as weak relation-candidate evidence only.",
            "forbidden_claim": "A stable internal relation state, component source, or causal edge is not authorized.",
            "main_caution": "A/B logit-lens margins before label instruction are external readouts with arbitrary label-token directions.",
            "next_step": "Run component-level residual ledger around label instruction and terminal positions before stronger mechanism claims.",
        },
    }
    OUT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(OUT_PATH)


if __name__ == "__main__":
    main()
