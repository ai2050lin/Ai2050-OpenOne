#!/usr/bin/env python3
"""Phase481 analysis for Phase480 component margin ledger."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
ROWS_PATH = ROOT / "tests" / "gpt5" / "result" / "phase480_component_margin_ledger" / "phase480_component_margin_ledger_rows.jsonl"
OUT_DIR = ROOT / "tests" / "gpt5" / "result" / "phase481_component_margin_ledger_analysis"
OUT_PATH = OUT_DIR / "phase481_component_margin_ledger_analysis.json"

MARGINS = ("margin_ab", "margin_true", "margin_correct")
DELTAS = ("delta_margin_ab", "delta_margin_true", "delta_margin_correct")
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


def with_family(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        item = dict(row)
        item["layer_family"] = family_for(int(row["layer_index"]))
        out.append(item)
    return out


def summarize(rows: list[dict[str, Any]], fields: list[str], values: tuple[str, ...]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[tuple(row[field] for field in fields)].append(row)
    out = []
    for key, items in sorted(buckets.items()):
        item = {field: value for field, value in zip(fields, key, strict=True)}
        item["n"] = len(items)
        for value in values:
            vals = [float(row[value]) for row in items]
            item[f"mean_{value}"] = mean(vals)
            item[f"mean_abs_{value}"] = mean(abs(v) for v in vals)
            item[f"positive_rate_{value}"] = mean(1.0 if v > 0 else 0.0 for v in vals)
        out.append(item)
    return out


def paired_mapping_deltas(state_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keyed = {
        (row["source_sample_id"], row["label_mapping"], row["role"], row["state"], int(row["layer_index"])): row
        for row in state_rows
    }
    out = []
    samples = sorted({row["source_sample_id"] for row in state_rows})
    roles = sorted({row["role"] for row in state_rows})
    states = sorted({row["state"] for row in state_rows})
    layers = sorted({int(row["layer_index"]) for row in state_rows})
    for sample_id in samples:
        for role in roles:
            for state in states:
                for layer in layers:
                    ab = keyed.get((sample_id, "mu_ab", role, state, layer))
                    ba = keyed.get((sample_id, "mu_ba", role, state, layer))
                    if ab is None or ba is None:
                        continue
                    item = {
                        "source_sample_id": sample_id,
                        "role": role,
                        "state": state,
                        "layer_index": layer,
                        "layer_family": family_for(layer),
                        "truth_value": ab["truth_value"],
                    }
                    for margin in MARGINS:
                        item[f"mu_ab_{margin}"] = float(ab[margin])
                        item[f"mu_ba_{margin}"] = float(ba[margin])
                        item[f"delta_{margin}"] = float(ba[margin]) - float(ab[margin])
                    item["raw_flip_residual"] = item["mu_ab_margin_ab"] + item["mu_ba_margin_ab"]
                    item["raw_flip_aligned"] = abs(item["raw_flip_residual"]) < (
                        abs(item["mu_ab_margin_ab"]) + abs(item["mu_ba_margin_ab"])
                    )
                    out.append(item)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = with_family(load_jsonl(ROWS_PATH))
    state_rows = [row for row in rows if row["row_type"] == "state_margin"]
    component_rows = [row for row in rows if row["row_type"] == "component_delta"]
    paired = paired_mapping_deltas(state_rows)
    out = {
        "schema_version": "phase481_component_margin_ledger_analysis.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "analysis_only_no_model_run_no_cuda",
        "input_trace": str(ROWS_PATH.relative_to(ROOT)),
        "state_row_count": len(state_rows),
        "component_delta_row_count": len(component_rows),
        "paired_mapping_delta_count": len(paired),
        "state_margins_by_role_state_family": summarize(state_rows, ["role", "state", "layer_family"], MARGINS),
        "component_deltas_by_role_component_family": summarize(component_rows, ["role", "component", "layer_family"], DELTAS),
        "component_deltas_by_truth_role_component_family": summarize(
            component_rows,
            ["truth_value", "role", "component", "layer_family"],
            DELTAS,
        ),
        "mapping_pairs_by_role_state_family": summarize(
            paired,
            ["role", "state", "layer_family"],
            ("delta_margin_ab", "delta_margin_true", "delta_margin_correct", "raw_flip_residual"),
        ),
        "interpretation": {
            "allowed_claim": "Coarse attention/MLP readout changes can be described by role and layer family.",
            "forbidden_claim": "No causal mediation, head/neuron attribution, or internal mechanism closure is authorized.",
            "next_step": "If stable component windows appear, validate on an independent holdout before any stronger claim.",
        },
    }
    OUT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(OUT_PATH)


if __name__ == "__main__":
    main()
