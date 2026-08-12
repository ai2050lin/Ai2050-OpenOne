#!/usr/bin/env python3
"""Expand Phase1001 natural confirmation to every selected direction.

The main cross-model run used 64 stratified natural-generation directions per
model. This script opens all 256 previously selected confirmation directions
without changing source depth or receiver identities. GLM4 uses its audited
newline-aligned color decision boundary. Qwen3 additionally reports the
pre-registered top-two sensitivity set as a posthoc robustness result.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))

import phase1000_scpg_discovery as scpg
import phase1001_cross_model_functional_topology as cross
from model_utils import get_layers, load_model, release_model


PHASE = 1001
MODELS = ("qwen3", "glm4", "deepseek7b")
BASE_ROOT = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1001_attention_physical_decomposition"
    / "cross_model_topology_causal_screen"
)
OUT_ROOT = BASE_ROOT / "natural_confirmation_expansion"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def protocol_for(model_name: str) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, int],
    dict[str, Any],
]:
    model_root = BASE_ROOT / model_name
    frozen = read_json(model_root / "frozen_topology.json")
    pairs = read_jsonl(model_root / "confirmation_selected_pairs.jsonl")
    if model_name == "glm4":
        aligned_root = BASE_ROOT / "glm4_decision_boundary_audit"
        cases = read_jsonl(aligned_root / "aligned_cases.jsonl")
        audit = read_json(aligned_root / "protocol_audit.json")
        candidate_ids = {
            key: int(value)
            for key, value in audit["candidate_token_ids"].items()
        }
        boundary = "newline_aligned_capitalized_color"
    else:
        cases = read_jsonl(model_root / "cases.jsonl")
        audit = read_json(model_root / "protocol_audit.json")
        candidate_ids = {
            key: int(value)
            for key, value in audit["candidate_token_ids"].items()
        }
        boundary = "original_answer_boundary"
    return cases, pairs, candidate_ids, {
        "frozen": frozen,
        "boundary": boundary,
        "protocol_audit": audit,
    }


def run(model_name: str, batch_size: int, natural_budget: int) -> dict[str, Any]:
    cases, pairs, candidate_ids, contract = protocol_for(model_name)
    frozen = contract["frozen"]
    case_by_id = {row["record_id"]: row for row in cases}
    rows = cross.directional(
        pairs, case_by_id, "confirmation_full_natural"
    )
    if len(rows) != 256:
        raise RuntimeError(f"confirmation direction drift: {len(rows)}")
    ranked = frozen["ranked_receivers"]
    official_size = int(frozen["joint_size"])
    sizes = [official_size]
    if model_name == "qwen3" and official_size == 1:
        sizes.append(2)

    model = tokenizer = None
    started = time.time()
    try:
        model, tokenizer, device = load_model(
            model_name, dtype=torch.bfloat16, use_8bit=True
        )
        tokenizer.padding_side = "left"
        layers = get_layers(model)
        scpg.MODEL = model_name
        scpg.PHASE = PHASE
        cross.NATURAL_PER_STRATUM = 8
        size_summaries = {}
        all_rows = []
        for size in sizes:
            result_rows, summary = cross.natural_joint(
                model,
                layers,
                tokenizer,
                device,
                rows,
                candidate_ids,
                int(frozen["source_depth"]),
                ranked,
                size,
                batch_size,
                natural_budget,
            )
            if summary["subset_size"] != 256:
                raise RuntimeError(
                    f"expanded subset drift: {summary['subset_size']}"
                )
            for row in result_rows:
                row["official_frozen_size"] = size == official_size
                row["posthoc_robustness_size"] = (
                    model_name == "qwen3" and size == 2
                )
                row["decision_boundary"] = contract["boundary"]
            all_rows.extend(result_rows)
            size_summaries[str(size)] = {
                **summary,
                "official_frozen_size": size == official_size,
                "posthoc_robustness_size": (
                    model_name == "qwen3" and size == 2
                ),
            }
        output_root = OUT_ROOT / model_name
        cross.write_jsonl(output_root / "rows.jsonl", all_rows)
        official = size_summaries[str(official_size)]["conditions"]
        checks = {
            "source_do_source_rate": (
                official["source_do"]["source_rate"] >= 0.70
            ),
            "joint_restore_target_rate": (
                official["source_plus_joint_restore"]["target_rate"] >= 0.50
            ),
            "full_256_directions": (
                size_summaries[str(official_size)]["subset_size"] == 256
            ),
        }
        summary = {
            "schema_version": (
                "phase1001_natural_confirmation_expansion_summary.v1"
            ),
            "phase": PHASE,
            "model": model_name,
            "status": "complete",
            "decision_boundary": contract["boundary"],
            "source_depth": frozen["source_depth"],
            "official_joint_size": official_size,
            "official_joint_event_ids": frozen["joint_event_ids"],
            "confirmation_direction_count": len(rows),
            "size_summaries": size_summaries,
            "gate_checks": checks,
            "official_expanded_natural_gate_pass": all(checks.values()),
            "quantized_8bit": True,
            "elapsed_seconds": time.time() - started,
            "cuda_device": torch.cuda.get_device_name(0),
        }
        cross.write_json(output_root / "summary.json", summary)
        print(json.dumps({
            "model": model_name,
            "decision_boundary": contract["boundary"],
            "official_joint_size": official_size,
            "size_summaries": size_summaries,
            "gate_checks": checks,
        }, ensure_ascii=False, indent=2))
        return summary
    finally:
        if model is not None:
            release_model(model)
        model = tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def aggregate() -> dict[str, Any]:
    rows = []
    for model_name in MODELS:
        summary = read_json(OUT_ROOT / model_name / "summary.json")
        official = summary["size_summaries"][
            str(summary["official_joint_size"])
        ]["conditions"]
        row = {
            "model": model_name,
            "decision_boundary": summary["decision_boundary"],
            "official_joint_size": summary["official_joint_size"],
            "n": official["source_do"]["n"],
            "source_do_source_rate": official[
                "source_do"
            ]["source_rate"],
            "restore_target_rate": official[
                "source_plus_joint_restore"
            ]["target_rate"],
            "restore_exact_short_rate": official[
                "source_plus_joint_restore"
            ]["exact_short_rate"],
            "gate_pass": summary["official_expanded_natural_gate_pass"],
        }
        if model_name == "qwen3" and "2" in summary["size_summaries"]:
            row["posthoc_k2_restore_target_rate"] = summary[
                "size_summaries"
            ]["2"]["conditions"]["source_plus_joint_restore"]["target_rate"]
        rows.append(row)
    payload = {
        "schema_version": (
            "phase1001_natural_confirmation_expansion_aggregate.v1"
        ),
        "phase": PHASE,
        "rows": rows,
        "all_models_pass": all(row["gate_pass"] for row in rows),
        "claim_boundary": (
            "Official rates use frozen joint sizes. Qwen3 k=2 is explicitly "
            "posthoc robustness evidence and not the registered primary set."
        ),
    }
    cross.write_json(OUT_ROOT / "summary.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--natural-budget", type=int, default=6)
    args = parser.parse_args()
    if args.aggregate:
        aggregate()
        return
    if args.model is None:
        parser.error("--model is required unless --aggregate is used")
    run(args.model, args.batch_size, args.natural_budget)


if __name__ == "__main__":
    main()
