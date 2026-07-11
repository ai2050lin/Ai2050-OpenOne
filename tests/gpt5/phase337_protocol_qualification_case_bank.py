#!/usr/bin/env python3
"""Freeze the Phase337 relation-binding protocol qualification denominator."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from model_registry import get_model_spec  # noqa: E402
from phase330_nine_family_case_bank import MODELS  # noqa: E402
from phase333_dynamic_case_bank import INTERFACES, interface_prompt  # noqa: E402
from phase334_natural_necessity_case_bank import KNOWLEDGE_ITEMS, render_raw, task_for  # noqa: E402


PHASE = "Phase337"
SCHEMA_VERSION = "13.0.0"
ROUND_DEFAULT = "material_relation_binding"
OUT = ROOT / "tests/gpt5/result/phase337_protocol_qualification"
MECHANISM = "material_relation_binding"
SOURCE_MECHANISM = "material"
TEMPLATE = "template_a"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def split_for(item_index: int) -> str:
    if item_index < 6:
        return "discovery"
    if item_index < 9:
        return "calibration"
    return "heldout"


def register(round_name: str = ROUND_DEFAULT) -> dict[str, Any]:
    root = OUT / round_name
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        spec = get_model_spec(model)
        tokenizer = AutoTokenizer.from_pretrained(
            str(spec.local_dir), trust_remote_code=spec.trust_remote_code,
            local_files_only=True, use_fast=False,
        )
        for item_index in range(len(KNOWLEDGE_ITEMS)):
            task = task_for("content_knowledge", SOURCE_MECHANISM, item_index)
            raw_prompt = render_raw(task, TEMPLATE)
            for interface in INTERFACES:
                prompt, add_special, answer_phase = interface_prompt(
                    tokenizer, model, raw_prompt, interface
                )
                rows.append({
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": PHASE,
                    "created_at": now(),
                    "case_id": f"phase337_{model}_{item_index:02d}_{interface}",
                    "semantic_case_id": f"phase337_{item_index:02d}_{interface}",
                    "model": model,
                    "family_id": "content_knowledge",
                    "mechanism_id": MECHANISM,
                    "source_mechanism_id": SOURCE_MECHANISM,
                    "item_index": item_index,
                    "split": split_for(item_index),
                    "template_id": TEMPLATE,
                    "interface": interface,
                    "answer_phase": answer_phase,
                    "prompt": prompt,
                    "raw_prompt": raw_prompt,
                    "tokenization_add_special_tokens": add_special,
                    "source_fragment": task["source_fragment"],
                    "query_fragment": task["query_fragment"],
                    "target": task["target"],
                    "target_aliases": task["target_aliases"],
                    "distractors": task["distractors"],
                    "target_class": "explicit_relation_binding_value",
                    "language": "en",
                    "protocol": "semantic_relation_binding",
                    "expected_structure": "natural_answer",
                    "selection_updates_allowed": False,
                    "internal_intervention_allowed": False,
                })
    if len(rows) != 108:
        raise RuntimeError(f"Expected 108 cases, got {len(rows)}")
    if len({row["case_id"] for row in rows}) != len(rows):
        raise RuntimeError("Duplicate Phase337 case id")

    contract = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE,
        "created_at": now(),
        "mechanism_id": MECHANISM,
        "claim_scope": "Explicitly stated relation binding, not parametric knowledge origin.",
        "trigger": "A context states object-material and object-attribute relations; query asks material.",
        "state_precondition": "The target material occurs in the source context and the query identifies the object.",
        "operation": ["read_relation", "bind_object_value", "route_to_answer"],
        "expected_path": ["source", "query", "answer_start"],
        "competitors": ["stated_attribute", "unknown"],
        "readout": "Registered material phrase under the selected interface.",
        "rollout": "Natural generation reaches an answer and states the registered material.",
        "physical_path_status": "not_tested_in_phase337",
        "causal_status": "not_tested_in_phase337",
    }
    protocol = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE,
        "created_at": now(),
        "purpose": "Separate answer arrival, semantic correctness, phrase validity, and baseline capability.",
        "models": list(MODELS),
        "interfaces": list(INTERFACES),
        "template": TEMPLATE,
        "objects_per_model_interface": 12,
        "registered_case_count": len(rows),
        "max_new_tokens": 128,
        "cell_capable_case_min": 9,
        "cell_answer_reached_min": 9,
        "stage_gate_common_interface_model_min": 2,
        "baseline_capability_definition": (
            "answer_reached AND answer_head_semantic_correct AND target_phrase_valid"
        ),
        "diagnostic_only": [
            "initial_target_rank", "protocol_followed", "token_budget_exhausted",
            "semantic_correct_outside_answer",
        ],
        "claim_boundaries": [
            "No internal activation or intervention is measured.",
            "No parametric-memory origin claim is allowed.",
            "No mechanism or neuron node is published.",
            "A passing cell only qualifies a denominator for later causal tests.",
        ],
    }
    write_jsonl(root / "phase337_registered_cases.jsonl", rows)
    write_json(root / "phase337_rule_contract.json", contract)
    write_json(root / "phase337_registered_protocol.json", protocol)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE,
        "created_at": now(),
        "registered_case_count": len(rows),
        "model_case_count": {
            model: sum(row["model"] == model for row in rows) for model in MODELS
        },
        "interface_case_count": {
            interface: sum(row["interface"] == interface for row in rows)
            for interface in INTERFACES
        },
        "split_case_count": {
            split: sum(row["split"] == split for row in rows)
            for split in ("discovery", "calibration", "heldout")
        },
        "valid": True,
    }
    write_json(root / "phase337_case_bank_validation.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", default=ROUND_DEFAULT)
    args = parser.parse_args()
    print(json.dumps(register(args.round), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
