#!/usr/bin/env python3
"""Freeze independent Phase570 late answer-competition bridge cases."""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

import phase569_relation_competition_protocol as p569  # noqa: E402
from phase548_shared_attention_compute_protocol import render_chat, tokenizer_for  # noqa: E402


PHASE = "Phase570"
MODELS = p569.MODELS
WORLD_COUNT = 96
CELLS_PER_PHENOTYPE = 4
OUT_DIR = ROOT / "tests/gpt5/result/phase570_answer_bridge_causal"
CASES_PATH = OUT_DIR / "phase570_registered_cases.jsonl.gz"
PROTOCOL_PATH = OUT_DIR / "phase570_frozen_protocol.json"
AUDIT_PATH = OUT_DIR / "phase570_static_audit.json"
PHASE569_SUMMARY = ROOT / "tests/gpt5/result/phase569_relation_competition/phase569_behavior_summary.json"
PHASE569_TRACE = ROOT / "tests/gpt5/result/phase569_relation_competition/phase569_coarse_trace_analysis.json"
CELL_PATTERN = re.compile(
    r"binding(?P<binding>[0-2])_query(?P<query>[0-2])_relation(?P<relation>body|tag)_"
    r"surface(?P<surface>[0-2])_order(?P<order>[0-1])$"
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def parse_cell(cell: str) -> dict[str, Any]:
    match = CELL_PATTERN.fullmatch(cell)
    if not match:
        raise ValueError(f"Invalid Phase569 factorial cell: {cell}")
    values = match.groupdict()
    return {
        "binding": int(values["binding"]),
        "query": int(values["query"]),
        "relation": values["relation"],
        "surface": int(values["surface"]),
        "order": int(values["order"]),
    }


def select_cells(report: dict[str, Any]) -> dict[str, list[str]]:
    metrics = report["phenotype_cell_metrics"]
    selected = {}
    correct_cells = report["stable_correct_cells"]
    selected["stable_correct"] = sorted(
        correct_cells,
        key=lambda cell: (
            -min(metrics[split][cell]["accuracy"] for split in p569.OPEN_SPLITS[:2]),
            -sum(metrics[split][cell]["accuracy"] for split in p569.OPEN_SPLITS[:2]),
            cell,
        ),
    )[:CELLS_PER_PHENOTYPE]
    confusion_cells = report["stable_relation_confusion_cells"]
    selected["stable_relation_confusion"] = sorted(
        confusion_cells,
        key=lambda cell: (
            -min(
                metrics[split][cell]["relation_confusion_rate_all_rows"]
                for split in p569.OPEN_SPLITS[:2]
            ),
            -min(
                metrics[split][cell]["relation_confusion_share_of_registered_errors"]
                for split in p569.OPEN_SPLITS[:2]
            ),
            cell,
        ),
    )[:CELLS_PER_PHENOTYPE]
    if any(len(cells) != CELLS_PER_PHENOTYPE for cells in selected.values()):
        raise RuntimeError(f"Phase570 cannot select four cells for {report['model']}")
    if set(selected["stable_correct"]) & set(selected["stable_relation_confusion"]):
        raise RuntimeError("Phase570 phenotype cell sets overlap")
    return selected


def selected_layers(trace: dict[str, Any]) -> dict[str, dict[str, int]]:
    shared = next(
        row for row in trace["cross_model_shared_topology"]
        if row["component"] == "attention_output"
        and row["semantic_role"] == "answer_boundary"
        and row["depth_band_8"] == 6
        and row["model_count"] == 3
        and row["normalized_gap_sign"] == 1
    )
    layers = {}
    for model in MODELS:
        event = shared["best_event_by_model"][model]
        target = int(event["layer"])
        layer_count = next(
            report["layer_count"] for report in trace["model_reports"]
            if report["model"] == model
        )
        wrong = max(0, min(layer_count - 1, round(0.25 * (layer_count - 1))))
        if wrong == target:
            wrong = max(0, target - 1)
        layers[model] = {
            "target_layer": target,
            "wrong_layer_control": wrong,
            "layer_count": layer_count,
        }
    return layers


def build_rows(
    selected_by_model: dict[str, dict[str, list[str]]]
) -> list[dict[str, Any]]:
    tokenizers = {model: tokenizer_for(model) for model in MODELS}
    rows = []
    for model in MODELS:
        tokenizer = tokenizers[model]
        for phenotype, cells in selected_by_model[model].items():
            for cell_rank, cell in enumerate(cells):
                factors = parse_cell(cell)
                for world_index in range(WORLD_COUNT):
                    source_split = "phenotype_discovery" if world_index < 48 else "path_confirmation"
                    source_world_index = 4000 + world_index + cell_rank * 100
                    row = p569.controlled_case(
                        source_split,
                        source_world_index,
                        factors["binding"],
                        factors["query"],
                        factors["relation"],
                        factors["surface"],
                        factors["order"],
                    )
                    prompt = render_chat(tokenizer, model, row["raw_prompt"])
                    candidate_ids = {
                        value: [
                            int(token) for token in tokenizer(
                                value, add_special_tokens=False
                            )["input_ids"]
                        ]
                        for value in row["all_candidates"]
                    }
                    rows.append({
                        **row,
                        "schema_version": "phase570_answer_bridge_case.v1",
                        "phase_id": PHASE,
                        "created_at": now(),
                        "model": model,
                        "split": "causal_validation",
                        "intended_phenotype": phenotype,
                        "source_factorial_cell": cell,
                        "cell_rank": cell_rank,
                        "causal_world_index": world_index,
                        "source_generation_split": source_split,
                        "source_generation_world_index": source_world_index,
                        "case_id": (
                            f"phase570_{model}_{phenotype}_cell{cell_rank}_world{world_index:03d}"
                        ),
                        "prompt_token_count": len(
                            tokenizer(prompt, add_special_tokens=True)["input_ids"]
                        ),
                        "candidate_token_ids": candidate_ids,
                        "sealed": False,
                    })
    return rows


def freeze() -> dict[str, Any]:
    behavior = read_json(PHASE569_SUMMARY)
    trace = read_json(PHASE569_TRACE)
    selected_by_model = {
        report["model"]: select_cells(report) for report in behavior["model_reports"]
    }
    layers = selected_layers(trace)
    rows = build_rows(selected_by_model)
    expected_per_model = WORLD_COUNT * CELLS_PER_PHENOTYPE * 2
    failures = []
    if len(rows) != expected_per_model * len(MODELS):
        failures.append("registered_count")
    if len({row["case_id"] for row in rows}) != len(rows):
        failures.append("case_id_collision")
    if any(row["target"] == row["other_relation_target"] for row in rows):
        failures.append("target_other_collision")
    if any(row["sealed"] for row in rows):
        failures.append("sealed_row")
    if any(
        len(ids) != 1
        for row in rows
        for ids in row["candidate_token_ids"].values()
    ):
        failures.append("candidate_not_single_token")
    if any(
        len({tuple(ids) for ids in row["candidate_token_ids"].values()}) != 4
        for row in rows
    ):
        failures.append("candidate_token_collision")
    if failures:
        raise RuntimeError(f"Phase570 static audit failed: {failures}")
    write_jsonl(CASES_PATH, rows)
    protocol = {
        "schema_version": "phase570_frozen_protocol.v1",
        "phase_id": PHASE,
        "created_at": now(),
        "models": list(MODELS),
        "world_count": WORLD_COUNT,
        "cells_per_phenotype": CELLS_PER_PHENOTYPE,
        "registered_case_count": len(rows),
        "registered_case_count_per_model": expected_per_model,
        "selected_cells_by_model": selected_by_model,
        "selected_layers_by_model": layers,
        "conditions": [
            "baseline",
            "target_projection_remove",
            "random_matched_remove",
            "wrong_layer_projection_remove",
        ],
        "causal_screen_scope": (
            "late answer-boundary attention contribution, not upstream relation encoding"
        ),
        "phase569_behavior_summary_sha256": sha256_file(PHASE569_SUMMARY),
        "phase569_trace_analysis_sha256": sha256_file(PHASE569_TRACE),
        "sealed_split_read": False,
    }
    write_json(PROTOCOL_PATH, protocol)
    audit = {
        "schema_version": "phase570_static_audit.v1",
        "phase_id": PHASE,
        "created_at": now(),
        "valid": True,
        "failures": [],
        "registered_case_count": len(rows),
        "registered_case_count_per_model": expected_per_model,
        "target_other_collision_count": 0,
        "candidate_non_single_token_count": 0,
        "candidate_token_collision_count": 0,
        "cases_sha256": sha256_file(CASES_PATH),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "model_execution_performed": False,
        "sealed_split_read": False,
    }
    write_json(AUDIT_PATH, audit)
    print(json.dumps({
        "registered": len(rows),
        "per_model": expected_per_model,
        "selected_cells": selected_by_model,
        "layers": layers,
        "valid": True,
    }, ensure_ascii=False, indent=2))
    return protocol


if __name__ == "__main__":
    freeze()
