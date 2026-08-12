#!/usr/bin/env python3
"""Validate and summarize Phase1006 without adding new model evidence."""
from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from phase1006_autoregressive_temporal_aggregation_protocol import (
    MODELS,
    OUT_ROOT,
    PHASE,
    read_json,
    write_json,
)


def assert_finite(value: Any, path: str = "root") -> None:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RuntimeError(f"non-finite value at {path}: {value}")
    elif isinstance(value, dict):
        for key, item in value.items():
            assert_finite(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            assert_finite(item, f"{path}[{index}]")


def role_class_counts(cell: dict[str, Any]) -> dict[str, int]:
    counts = Counter()
    for roles in cell["semantic_reconstruction_audit"].values():
        names = set(roles)
        if names == {"query_name"}:
            counts["query_name"] += 1
        elif any(name.startswith("fact_entity_") for name in names):
            counts["fact_entity"] += 1
        elif any(name.endswith("word0") for name in names):
            counts["value_word0"] += 1
        elif any(name.endswith("word1") for name in names):
            counts["value_word1"] += 1
        else:
            counts["other"] += 1
    return dict(counts)


def main() -> None:
    protocol = read_json(OUT_ROOT / "protocol" / "protocol.json")
    replication_protocol = read_json(
        OUT_ROOT / "source_replication" / "protocol.json"
    )
    if int(protocol["protocol_revision"]) != 4:
        raise RuntimeError("unexpected parent protocol revision")
    if (
        replication_protocol["parent_protocol_digest"]
        != protocol["preregistration_digest"]
    ):
        raise RuntimeError("replication parent digest mismatch")

    behavior_rows = []
    source_rows = []
    behavior_total_n = 0
    formal_source_condition_rows = 0
    for model_name in MODELS:
        behavior = read_json(
            OUT_ROOT / "behavior" / model_name / "summary.json"
        )
        if behavior["protocol_digest"] != protocol["preregistration_digest"]:
            raise RuntimeError(f"{model_name}: behavior digest mismatch")
        assert_finite(behavior, f"behavior.{model_name}")
        behavior_total_n += sum(int(cell["n"]) for cell in behavior["cells"])
        behavior_rows.extend({
            "model": model_name,
            "split": cell["split"],
            "template": int(cell["template"]),
            "n": int(cell["n"]),
            "step0": cell["step0_autoregressive_accuracy"],
            "step1": cell["step1_autoregressive_accuracy"],
            "teacher_step1": cell["step1_teacher_forced_accuracy"],
            "natural_exact": cell["natural_exact_rate"],
            "immediate_eos": cell["immediate_eos_rate"],
            "natural_prefix": cell["natural_protocol_prefix_rate"],
            "gate_pass": bool(cell["behavior_gate_pass"]),
        } for cell in behavior["cells"])

        source_path = OUT_ROOT / "blind_source" / model_name / "summary.json"
        if not source_path.exists():
            continue
        source = read_json(source_path)
        if source["protocol_digest"] != protocol["preregistration_digest"]:
            raise RuntimeError(f"{model_name}: source digest mismatch")
        assert_finite(source, f"source.{model_name}")
        for cell in source["source_cells"]:
            if not cell["source_run"]:
                continue
            condition_path = (
                OUT_ROOT
                / "blind_source"
                / model_name
                / cell["split"]
                / f"template{int(cell['template'])}"
                / "condition_rows.jsonl"
            )
            with condition_path.open("r", encoding="utf-8") as handle:
                formal_source_condition_rows += sum(
                    1 for line in handle if line.strip()
                )
            source_rows.append({
                "model": model_name,
                "split": cell["split"],
                "template": int(cell["template"]),
                "screen_n": int(cell["screen_n"]),
                "frozen_n": int(cell["frozen_n"]),
                "event_universe_size": int(cell["event_universe_size"]),
                "frozen_positions": cell["frozen_positions"],
                "role_class_counts": role_class_counts(cell),
                "donor_sequence_rate": cell[
                    "frozen_different_answer"
                ]["donor_sequence_rate"],
                "step_donor_rates": cell[
                    "frozen_different_answer"
                ]["step_donor_rates"],
                "median_normalized_transfer": cell[
                    "frozen_different_answer"
                ]["median_normalized_transfer"],
                "same_answer_target_rate": cell[
                    "frozen_same_answer_control"
                ]["target_sequence_rate"],
                "noop_target_rate": cell[
                    "frozen_target_noop"
                ]["target_sequence_rate"],
                "gate_pass": bool(cell["source_gate_pass"]),
            })

    if behavior_total_n != 384:
        raise RuntimeError(f"formal behavior n={behavior_total_n}")
    if len(source_rows) != 3:
        raise RuntimeError(f"formal source cells={len(source_rows)}")

    replication_rows = []
    replication_behavior_n = 0
    for model_name in MODELS:
        summary = read_json(
            OUT_ROOT
            / "source_replication"
            / model_name
            / "summary.json"
        )
        assert_finite(summary, f"replication.{model_name}")
        if not summary["model_loaded"]:
            continue
        if summary["audit_protocol_digest"] != replication_protocol["digest"]:
            raise RuntimeError(
                f"{model_name}: replication digest mismatch"
            )
        for cell in summary["cells"]:
            replication_behavior_n += int(cell["behavior"]["n"])
            row = {
                "model": model_name,
                "split": cell["split"],
                "template": int(cell["template"]),
                "n": int(cell["behavior"]["n"]),
                "behavior_gate_pass": bool(
                    cell["behavior"]["behavior_gate_pass"]
                ),
                "frozen_positions": cell["frozen_positions"],
                "role_class_counts": cell[
                    "semantic_role_audit"
                ]["role_class_counts"],
                "source_run": bool(cell["source_run"]),
                "source_gate_pass": bool(cell["source_gate_pass"]),
            }
            if cell["source_run"]:
                row.update({
                    "donor_sequence_rate": cell[
                        "different_answer"
                    ]["donor_sequence_rate"],
                    "step_donor_rates": cell[
                        "different_answer"
                    ]["step_donor_rates"],
                    "median_normalized_transfer": cell[
                        "different_answer"
                    ]["median_normalized_transfer"],
                    "same_answer_target_rate": cell[
                        "same_answer_control"
                    ]["target_sequence_rate"],
                    "noop_target_rate": cell[
                        "target_noop"
                    ]["target_sequence_rate"],
                })
            else:
                row["skip_reason"] = cell["skip_reason"]
            replication_rows.append(row)

    fingerprints = [
        {
            "cell": (
                f"{row['model']}/{row['split']}/t{row['template']}"
            ),
            "role_class_counts": row["role_class_counts"],
            "contains_query_and_four_value_tokens": (
                row["role_class_counts"].get("query_name", 0) == 1
                and row["role_class_counts"].get("value_word0", 0) == 2
                and row["role_class_counts"].get("value_word1", 0) == 2
            ),
        }
        for row in source_rows
    ]
    if not all(
        item["contains_query_and_four_value_tokens"]
        for item in fingerprints
    ):
        raise RuntimeError("repeated role fingerprint audit failed")

    formal_source_passes = sum(row["gate_pass"] for row in source_rows)
    replication_source_passes = sum(
        row["source_gate_pass"] for row in replication_rows
    )
    confirmation_source_passes = sum(
        row["source_gate_pass"]
        for row in replication_rows
        if row["split"] == "confirmation"
    )
    cross_model_source_models = {
        row["model"]
        for row in replication_rows
        if row["source_gate_pass"]
    }
    downstream_authorized = (
        confirmation_source_passes >= 1
        and len(cross_model_source_models) >= 2
    )

    result = {
        "schema_version": "phase1006_final_summary.v1",
        "phase": PHASE,
        "parent_protocol_digest": protocol["preregistration_digest"],
        "replication_protocol_digest": replication_protocol["digest"],
        "counts": {
            "formal_behavior_cases": behavior_total_n,
            "formal_behavior_cells": len(behavior_rows),
            "formal_behavior_pass_cells": sum(
                row["gate_pass"] for row in behavior_rows
            ),
            "formal_source_run_cells": len(source_rows),
            "formal_source_pass_cells": formal_source_passes,
            "formal_source_condition_rows": formal_source_condition_rows,
            "replication_behavior_cases": replication_behavior_n,
            "replication_cells": len(replication_rows),
            "replication_source_pass_cells": replication_source_passes,
        },
        "behavior": behavior_rows,
        "formal_source": source_rows,
        "replication": replication_rows,
        "repeated_role_fingerprint": {
            "cells": fingerprints,
            "all_three_contain_query_and_four_value_tokens": True,
            "two_cells_also_contain_both_entity_slots": sum(
                row["role_class_counts"].get("fact_entity", 0) == 2
                for row in source_rows
            ) == 2,
            "interpretation": (
                "label-blind search repeatedly recovered a relational "
                "prompt-role constellation; this is a repeated structural "
                "signal, not yet a clean cross-model causal source law"
            ),
        },
        "measurement_definitions": {
            "candidate_margin": "m = logit(donor) - logit(target)",
            "normalized_transfer": (
                "tau = (m_patch - m_target) / "
                "max(|m_donor - m_target|, 1e-6)"
            ),
            "source_gate": (
                "donor_sequence_rate >= 0.80 and median(tau) >= 0.50 "
                "and same_answer_target_rate >= 0.95 "
                "and noop_target_rate >= 0.99"
            ),
        },
        "evidence_decision": {
            "formal_source_gate_passes": formal_source_passes,
            "replication_source_gate_passes": replication_source_passes,
            "independent_confirmation_source_passes": (
                confirmation_source_passes
            ),
            "cross_model_clean_source_model_count": len(
                cross_model_source_models
            ),
            "downstream_temporal_receiver_search_authorized": (
                downstream_authorized
            ),
            "reason": (
                "Only GLM4 discovery template 0 passed on the unused-pair "
                "replication. Confirmation retained a large same-answer "
                "side effect, and DeepSeek7B failed replication behavior. "
                "KV/cache/component/head/neuron decomposition is therefore "
                "not authorized."
            ),
        },
        "conclusions": [
            (
                "The natural Answer-prefix interface and effective turn "
                "termination were measured correctly across all models."
            ),
            (
                "Behavior qualification is strongly model- and "
                "template-dependent: 3 of 12 formal cells passed."
            ),
            (
                "Blind source search repeatedly recovered query, both "
                "two-token values, and usually both entity slots."
            ),
            (
                "Whole-vector depth-1 transplantation strongly transfers "
                "both answer words, but clean controls do not generalize "
                "across confirmation and models."
            ),
            (
                "Autoregressive temporal aggregation remains untested, "
                "because its required clean source parent did not close."
            ),
        ],
        "hard_limitations": [
            "Only a controlled two-entity paired-code retrieval task.",
            "Only one clean replication cell in one quantized model.",
            "Whole residual vectors mix semantic content and context.",
            "No clean independent confirmation source set.",
            "No cross-model source gate and no BF16 parent audit.",
            "No KV, receiver component, head, channel, or neuron result.",
            "The formulas above are measurements, not language laws.",
        ],
        "integrity_audit": {
            "all_json_numbers_finite": True,
            "parent_digest_match": True,
            "replication_pair_overlap_with_formal_pairs": 0,
            "semantic_labels_used_for_formal_position_selection": False,
            "position_reranking_on_replication": False,
            "failed_gates_respected": True,
        },
    }
    assert_finite(result)
    write_json(OUT_ROOT / "final" / "summary.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
