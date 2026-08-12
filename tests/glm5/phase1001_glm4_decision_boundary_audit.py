#!/usr/bin/env python3
"""Audit GLM4 at its generation-aligned color decision boundary.

GLM4 naturally emits a newline before the capitalized color token. The main
cross-model test measured lowercase color candidates one step earlier. This
audit appends the observed fixed newline prefix, switches to the actual
capitalized color token IDs, and retests the already-frozen source and receiver
set. It does not select new layers, components, or a new joint size.
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))

import phase1000_scpg_discovery as scpg
import phase1001_cross_model_functional_topology as cross
from model_utils import get_layers, load_model, release_model


PHASE = 1001
MODEL = "glm4"
PREFIX = "\n"
OUT_ROOT = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1001_attention_physical_decomposition"
    / "cross_model_topology_causal_screen"
)
MODEL_ROOT = OUT_ROOT / MODEL
AUDIT_ROOT = OUT_ROOT / "glm4_decision_boundary_audit"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def aligned_cases(
    tokenizer,
    cases: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, Any]]:
    candidate_ids: dict[str, int] = {}
    prefix_ids: set[tuple[int, ...]] = set()
    aligned = []
    for source in cases:
        row = copy.deepcopy(source)
        base = tokenizer.encode(
            row["rendered_prompt"], add_special_tokens=False
        )
        with_prefix = tokenizer.encode(
            row["rendered_prompt"] + PREFIX,
            add_special_tokens=False,
        )
        if base != row["input_ids"]:
            raise RuntimeError(f"base token drift: {row['record_id']}")
        delta = tuple(int(value) for value in with_prefix[len(base):])
        if not delta:
            raise RuntimeError("empty generation prefix")
        prefix_ids.add(delta)
        for color in cross.COLORS:
            extended = tokenizer.encode(
                row["rendered_prompt"] + PREFIX + color.capitalize(),
                add_special_tokens=False,
            )
            suffix = extended[len(with_prefix):]
            if len(suffix) != 1:
                raise RuntimeError(
                    f"capital candidate drift: {row['record_id']}/{color}/{suffix}"
                )
            token_id = int(suffix[0])
            previous = candidate_ids.setdefault(color, token_id)
            if previous != token_id:
                raise RuntimeError(f"candidate ID drift: {color}")
        row["input_ids"] = [*base, *delta]
        row["input_token_count"] = len(row["input_ids"])
        row["role_positions"]["answer_boundary"] = len(row["input_ids"]) - 1
        row["candidate_token_ids"] = dict(candidate_ids)
        row["decision_boundary_alignment"] = {
            "forced_prefix": PREFIX,
            "forced_prefix_token_ids": list(delta),
            "candidate_surface": "capitalized_color",
        }
        aligned.append(row)
    if len(prefix_ids) != 1:
        raise RuntimeError(f"prefix token drift: {prefix_ids}")
    for row in aligned:
        row["candidate_token_ids"] = dict(candidate_ids)
    audit = {
        "schema_version": "phase1001_glm4_boundary_protocol_audit.v1",
        "phase": PHASE,
        "model": MODEL,
        "case_count": len(aligned),
        "forced_prefix": PREFIX,
        "forced_prefix_token_ids": list(next(iter(prefix_ids))),
        "candidate_surface": {
            color: color.capitalize() for color in cross.COLORS
        },
        "candidate_token_ids": candidate_ids,
        "all_boundaries_exact": True,
        "selection_changed": False,
    }
    return aligned, candidate_ids, audit


def source_test(
    model,
    layers,
    device,
    rows: list[dict[str, Any]],
    candidate_ids: dict[str, int],
    source_depth: int,
    batch_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    result_rows = []
    batches = list(scpg.batches_by_template(rows, batch_size))
    for batch_number, batch in enumerate(batches, 1):
        source_cases = [row["source"] for row in batch]
        target_cases = [row["target"] for row in batch]
        source_logits, source_residuals = scpg.capture_residuals(
            model, device, source_cases, (source_depth,), candidate_ids
        )
        target_logits, _ = scpg.capture_residuals(
            model, device, target_cases, (source_depth,), candidate_ids
        )
        patch = scpg.source_patch_spec(
            source_depth,
            target_cases,
            source_residuals[source_depth],
            "joint",
        )
        patched = scpg.forward_candidate(
            model,
            layers,
            device,
            target_cases,
            candidate_ids,
            source_patch=patch,
        )
        result_rows.extend(scpg.intervention_rows(
            batch,
            source_logits,
            target_logits,
            patched,
            "aligned_joint_entity",
            "phase1001_glm4_boundary_source_row.v1",
        ))
        del source_logits, source_residuals, target_logits, patched
        print(
            f"[aligned-source-{rows[0]['partition']}] "
            f"{batch_number}/{len(batches)}",
            flush=True,
        )
    return result_rows, scpg.summarize_interventions(result_rows)[
        "aligned_joint_entity"
    ]


def run(batch_size: int, natural_budget: int) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("GLM4 decision-boundary audit requires CUDA")
    original_summary = read_json(MODEL_ROOT / "summary.json")
    frozen = read_json(MODEL_ROOT / "frozen_topology.json")
    if frozen["model"] != MODEL or not frozen["frozen_before_confirmation"]:
        raise RuntimeError("frozen topology contract drift")
    original_cases = read_jsonl(MODEL_ROOT / "cases.jsonl")
    discovery_pairs = read_jsonl(
        MODEL_ROOT / "discovery_selected_pairs.jsonl"
    )
    confirmation_pairs = read_jsonl(
        MODEL_ROOT / "confirmation_selected_pairs.jsonl"
    )

    model = tokenizer = None
    started = time.time()
    try:
        model, tokenizer, device = load_model(
            MODEL, dtype=torch.bfloat16, use_8bit=True
        )
        tokenizer.padding_side = "left"
        layers = get_layers(model)
        cases, candidate_ids, protocol_audit = aligned_cases(
            tokenizer, original_cases
        )
        case_by_id = {row["record_id"]: row for row in cases}
        discovery = cross.directional(
            discovery_pairs, case_by_id, "discovery_aligned"
        )
        confirmation = cross.directional(
            confirmation_pairs, case_by_id, "confirmation_aligned"
        )
        ranked = frozen["ranked_receivers"]
        joint_size = int(frozen["joint_size"])
        source_depth = int(frozen["source_depth"])
        events = [
            {
                key: item[key]
                for key in (
                    "event_id", "block_index", "layer_number",
                    "component", "role",
                )
            }
            for item in ranked
        ]
        scpg.MODEL = MODEL
        scpg.PHASE = PHASE
        scpg.JOINT_SIZES = (joint_size,)

        behavior_rows, behavior_summary = cross.candidate_behavior(
            model, device, cases, candidate_ids, batch_size
        )
        discovery_source_rows, discovery_source = source_test(
            model,
            layers,
            device,
            discovery,
            candidate_ids,
            source_depth,
            batch_size,
        )
        confirmation_source_rows, confirmation_source = source_test(
            model,
            layers,
            device,
            confirmation,
            candidate_ids,
            source_depth,
            batch_size,
        )
        discovery_joint_rows = scpg.joint_screen(
            model,
            layers,
            device,
            discovery,
            candidate_ids,
            source_depth,
            ranked,
            batch_size,
        )
        confirmation_joint_rows = scpg.joint_screen(
            model,
            layers,
            device,
            confirmation,
            candidate_ids,
            source_depth,
            ranked,
            batch_size,
        )
        discovery_joint = scpg.summarize_joint(
            discovery_joint_rows
        )[str(joint_size)]
        confirmation_joint = scpg.summarize_joint(
            confirmation_joint_rows
        )[str(joint_size)]

        discovery_natural_rows, discovery_natural = cross.natural_joint(
            model,
            layers,
            tokenizer,
            device,
            discovery,
            candidate_ids,
            source_depth,
            ranked,
            joint_size,
            batch_size,
            natural_budget,
        )
        confirmation_natural_rows, confirmation_natural = cross.natural_joint(
            model,
            layers,
            tokenizer,
            device,
            confirmation,
            candidate_ids,
            source_depth,
            ranked,
            joint_size,
            batch_size,
            natural_budget,
        )

        cross.write_jsonl(AUDIT_ROOT / "aligned_cases.jsonl", cases)
        cross.write_json(
            AUDIT_ROOT / "protocol_audit.json", protocol_audit
        )
        cross.write_jsonl(
            AUDIT_ROOT / "behavior_rows.jsonl", behavior_rows
        )
        cross.write_json(
            AUDIT_ROOT / "behavior_summary.json", behavior_summary
        )
        cross.write_jsonl(
            AUDIT_ROOT / "discovery_source_rows.jsonl",
            discovery_source_rows,
        )
        cross.write_jsonl(
            AUDIT_ROOT / "confirmation_source_rows.jsonl",
            confirmation_source_rows,
        )
        cross.write_jsonl(
            AUDIT_ROOT / "discovery_joint_rows.jsonl",
            discovery_joint_rows,
        )
        cross.write_jsonl(
            AUDIT_ROOT / "confirmation_joint_rows.jsonl",
            confirmation_joint_rows,
        )
        cross.write_jsonl(
            AUDIT_ROOT / "discovery_natural_rows.jsonl",
            discovery_natural_rows,
        )
        cross.write_jsonl(
            AUDIT_ROOT / "confirmation_natural_rows.jsonl",
            confirmation_natural_rows,
        )

        original_natural = original_summary[
            "confirmation_natural"
        ]["conditions"]["source_plus_joint_restore"]["target_rate"]
        aligned_natural = confirmation_natural[
            "conditions"
        ]["source_plus_joint_restore"]["target_rate"]
        checks = {
            "aligned_behavior": behavior_summary["gate_pass"],
            "aligned_discovery_source": (
                discovery_source["flip_rate"] >= 0.70
                and discovery_source["median_transfer"] >= 0.50
            ),
            "aligned_confirmation_source": (
                confirmation_source["flip_rate"] >= 0.70
                and confirmation_source["median_transfer"] >= 0.50
            ),
            "aligned_discovery_joint": (
                discovery_joint["median_mediation_fraction"] >= 0.30
                and discovery_joint["mean_sufficiency_transfer"] >= 0.30
            ),
            "aligned_confirmation_joint": (
                confirmation_joint["median_mediation_fraction"] >= 0.30
                and confirmation_joint["mean_sufficiency_transfer"] >= 0.30
            ),
            "aligned_natural_source": (
                confirmation_natural["conditions"]["source_do"]["source_rate"]
                >= 0.70
            ),
            "aligned_natural_restore": aligned_natural >= 0.50,
            "natural_restore_improves_over_unaligned": (
                aligned_natural > original_natural
            ),
        }
        summary = {
            "schema_version": (
                "phase1001_glm4_decision_boundary_audit_summary.v1"
            ),
            "phase": PHASE,
            "model": MODEL,
            "status": "complete",
            "protocol_audit": protocol_audit,
            "frozen_source_depth": source_depth,
            "frozen_joint_size": joint_size,
            "frozen_joint_event_ids": frozen["joint_event_ids"],
            "selection_changed": False,
            "behavior": behavior_summary,
            "discovery_source": discovery_source,
            "confirmation_source": confirmation_source,
            "discovery_joint": discovery_joint,
            "confirmation_joint": confirmation_joint,
            "discovery_natural": discovery_natural,
            "confirmation_natural": confirmation_natural,
            "unaligned_confirmation_natural_restore_target_rate": (
                original_natural
            ),
            "aligned_confirmation_natural_restore_target_rate": (
                aligned_natural
            ),
            "gate_checks": checks,
            "decision_boundary_alignment_pass": all(checks.values()),
            "elapsed_seconds": time.time() - started,
            "cuda_device": torch.cuda.get_device_name(0),
            "interpretation": (
                "The audit changes only the autoregressive observation "
                "boundary and candidate surface. It does not discover a new "
                "mechanism."
            ),
        }
        cross.write_json(AUDIT_ROOT / "summary.json", summary)
        print(json.dumps({
            "original_natural_restore": original_natural,
            "aligned_natural_restore": aligned_natural,
            "confirmation_joint": confirmation_joint,
            "checks": checks,
            "pass": summary["decision_boundary_alignment_pass"],
        }, ensure_ascii=False, indent=2))
        return summary
    finally:
        if model is not None:
            release_model(model)
        model = tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--natural-budget", type=int, default=6)
    args = parser.parse_args()
    run(args.batch_size, args.natural_budget)


if __name__ == "__main__":
    main()
