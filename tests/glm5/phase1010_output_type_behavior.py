#!/usr/bin/env python3
"""Run Phase1010 behavior qualification for all output-type panels."""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_utils import get_layers, get_model_info, load_model, release_model
from phase1006_blind_source_and_behavior import eos_token_ids
from phase1009_crossfamily_response_behavior import (
    aggregate_rates,
    behavior_batch,
    chunks,
    pair_rows,
)
from phase1010_output_type_protocol import (
    FAMILIES,
    MODELS,
    OUT_ROOT,
    OUTPUT_TYPES,
    PAIR_OPERATIONS,
    PHASE,
    PROTOCOL_REVISION,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)


BATCH_SIZE = 16


def decorate_behavior_rows(
    rows: list[dict[str, Any]],
    case_by_id: dict[str, dict[str, Any]],
) -> None:
    for row in rows:
        case = case_by_id[row["record_id"]]
        row["schema_version"] = "phase1010_behavior_row.v1"
        row["phase"] = PHASE
        row["output_type"] = case["output_type"]
        row["gold_entity"] = case["gold_entity"]
        row["gold_label"] = case["gold"]


def decorate_pair_rows(
    rows: list[dict[str, Any]],
    unit_by_id: dict[str, dict[str, Any]],
) -> None:
    for row in rows:
        unit = unit_by_id[row["unit_id"]]
        row["schema_version"] = "phase1010_pair_qualification.v1"
        row["phase"] = PHASE
        row["output_type"] = unit["output_type"]


def run_model(model_name: str) -> dict[str, Any]:
    protocol = read_json(OUT_ROOT / "protocol" / "protocol.json")
    if int(protocol["protocol_revision"]) != PROTOCOL_REVISION:
        raise RuntimeError("Phase1010 protocol revision drift")
    model_root = OUT_ROOT / "protocol" / model_name
    cases = read_jsonl(model_root / "cases.jsonl")
    units = read_jsonl(model_root / "units.jsonl")
    case_by_id = {case["record_id"]: case for case in cases}
    unit_by_id = {unit["unit_id"]: unit for unit in units}
    started = time.time()
    model = tokenizer = device = None
    rows: list[dict[str, Any]] = []
    try:
        model, tokenizer, device = load_model(model_name, use_8bit=True)
        layers = get_layers(model)
        info = get_model_info(model, model_name)
        effective_eos = eos_token_ids(model, tokenizer, model_name)
        grouped: dict[
            tuple[str, str, str, int, int],
            list[dict[str, Any]],
        ] = defaultdict(list)
        for case in cases:
            grouped[(
                case["family"],
                case["output_type"],
                case["split"],
                int(case["template"]),
                len(case["input_ids"]),
            )].append(case)
        for key, group in sorted(grouped.items()):
            family, output_type, split, template, _ = key
            panel_rows = []
            for batch in chunks(group, BATCH_SIZE):
                panel_rows.extend(behavior_batch(
                    model,
                    layers,
                    tokenizer,
                    device,
                    model_name,
                    batch,
                    effective_eos,
                ))
            decorate_behavior_rows(panel_rows, case_by_id)
            rows.extend(panel_rows)
            print(
                f"[behavior] {model_name}/{family}/{output_type}/"
                f"{split}/t{template} n={len(panel_rows)} "
                f"panel={np.mean([r['semantic_gate'] for r in panel_rows]):.3f} "
                f"strict={np.mean([r['strict_teacher_gate'] for r in panel_rows]):.3f} "
                f"rollout={np.mean([r['rollout_gate'] for r in panel_rows]):.3f}",
                flush=True,
            )
        if len(rows) != len(cases):
            raise RuntimeError(
                f"{model_name}: behavior coverage {len(rows)} != {len(cases)}"
            )
        behavior_by_id = {row["record_id"]: row for row in rows}
        if len(behavior_by_id) != len(rows):
            raise RuntimeError(f"{model_name}: duplicate behavior record")
        pairs = pair_rows(units, behavior_by_id)
        decorate_pair_rows(pairs, unit_by_id)

        panel_rates = aggregate_rates(rows, ("family", "output_type"))
        output_type_rates = aggregate_rates(rows, ("output_type",))
        family_rates = aggregate_rates(rows, ("family",))
        operation_summary = {}
        for family in FAMILIES:
            for output_type in OUTPUT_TYPES:
                key = f"{family}:{output_type}"
                operation_summary[key] = {}
                for operation in PAIR_OPERATIONS:
                    selected = [
                        row
                        for row in pairs
                        if row["family"] == family
                        and row["output_type"] == output_type
                        and row["operation"] == operation
                    ]
                    operation_summary[key][operation] = {
                        "n": len(selected),
                        "semantic_pair_qualified": int(sum(
                            row["semantic_pair_qualified"]
                            for row in selected
                        )),
                        "semantic_pair_rate": float(np.mean([
                            row["semantic_pair_qualified"]
                            for row in selected
                        ])),
                        "strict_teacher_pair_rate": float(np.mean([
                            row["strict_teacher_pair_qualified"]
                            for row in selected
                        ])),
                        "rollout_pair_rate": float(np.mean([
                            row["rollout_pair_qualified"]
                            for row in selected
                        ])),
                    }
        summary = {
            "schema_version": "phase1010_behavior_summary.v1",
            "phase": PHASE,
            "model": model_name,
            "protocol_digest": protocol["preregistration_digest"],
            "model_info": {
                "n_layers": int(info.n_layers),
                "d_model": int(info.d_model),
                "model_class": info.model_class,
                "loaded_8bit": True,
            },
            "case_count": len(rows),
            "pair_count": len(pairs),
            "effective_eos_ids": sorted(effective_eos),
            "panel_rates": panel_rates,
            "output_type_rates": output_type_rates,
            "family_rates": family_rates,
            "operation_summary": operation_summary,
            "overall_semantic_panel_rate": float(np.mean([
                row["semantic_gate"] for row in rows
            ])),
            "overall_semantic_full_vocab_rate": float(np.mean([
                row["semantic_full_vocab_hit"] for row in rows
            ])),
            "overall_strict_teacher_rate": float(np.mean([
                row["strict_teacher_gate"] for row in rows
            ])),
            "overall_rollout_case_rate": float(np.mean([
                row["rollout_gate"] for row in rows
            ])),
            "elapsed_seconds": time.time() - started,
            "policy": (
                "behavior qualification annotates each family/output cell; "
                "a weak cell is retained descriptively and cannot support "
                "a causal generalization"
            ),
        }
        output_root = OUT_ROOT / "behavior" / model_name
        write_jsonl(output_root / "rows.jsonl", rows)
        write_jsonl(output_root / "pair_qualification.jsonl", pairs)
        write_json(output_root / "summary.json", summary)
        print(json.dumps({
            "model": model_name,
            "cases": len(rows),
            "semantic_panel_rate": summary["overall_semantic_panel_rate"],
            "strict_teacher_rate": summary["overall_strict_teacher_rate"],
            "rollout_rate": summary["overall_rollout_case_rate"],
            "output_type_rates": {
                row["output_type"]: row["semantic_panel_rate"]
                for row in output_type_rates
            },
            "elapsed_seconds": summary["elapsed_seconds"],
        }, ensure_ascii=False, indent=2))
        return summary
    finally:
        if model is not None:
            release_model(model)
        model = tokenizer = device = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=MODELS)
    args = parser.parse_args()
    run_model(args.model)


if __name__ == "__main__":
    main()
