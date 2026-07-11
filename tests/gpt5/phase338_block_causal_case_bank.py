#!/usr/bin/env python3
"""Freeze Phase338 staged coarse-block causal screening cases and gates."""

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
from phase330_nine_family_case_bank import MODELS, TEMPLATES  # noqa: E402
from phase333_dynamic_case_bank import interface_prompt  # noqa: E402


PHASE = "Phase338"
SCHEMA_VERSION = "14.0.0"
ROUND_DEFAULT = "material_relation_block_screen"
OUT = ROOT / "tests/gpt5/result/phase338_block_causal_screen"
INTERFACE = "answer_aligned_chat"
MECHANISM = "material_relation_binding"

ITEMS = (
    ("tripod", "titanium", "foldable"), ("mortar", "granite", "heavy"),
    ("basket", "bamboo", "woven"), ("apron", "linen", "pleated"),
    ("hinge", "nickel", "adjustable"), ("statue", "marble", "carved"),
    ("satchel", "leather", "buckled"), ("goblet", "silver", "etched"),
    ("figurine", "porcelain", "painted"), ("gasket", "silicone", "flexible"),
    ("bell", "bronze", "polished"), ("stopper", "cork", "tapered"),
    ("bracket", "zinc", "angled"), ("cabinet", "birch", "lacquered"),
    ("hammock", "canvas", "suspended"), ("vase", "clay", "fired"),
    ("brooch", "platinum", "inlaid"), ("rope", "nylon", "braided"),
    ("column", "sandstone", "fluted"), ("curtain", "velvet", "lined"),
    ("anvil", "iron", "forged"), ("lantern", "paper", "ribbed"),
    ("pendant", "quartz", "faceted"), ("canister", "tin", "sealed"),
)

COMPONENTS = ("attention_output", "mlp_output", "residual_increment")
DEPTH_BINS = ("early", "middle", "late")
POSITION_ROLES = ("source", "query", "answer_start")


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


def split_for(index: int) -> str:
    if index < 12:
        return "discovery"
    if index < 18:
        return "calibration"
    if index < 22:
        return "heldout"
    return "private_heldout"


def raw_prompt(obj: str, material: str, attribute: str, template: str) -> str:
    fact = f"The registry states that the {obj} is made from {material} and is {attribute}."
    question = f"What material is the {obj} made from?"
    rule = "Answer with the stated material word."
    if template == "template_a":
        return f"{fact}\n{question}\n{rule}\nAnswer:"
    if template == "template_b":
        return f"Registry record: {fact}\nQuery: {question}\nOutput: {rule}\nResponse:"
    if template == "template_c":
        return f"Use this record: {fact}\n{rule}\nQuestion: {question}\nResult:"
    raise KeyError(template)


def register(round_name: str = ROUND_DEFAULT) -> dict[str, Any]:
    root = OUT / round_name
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        spec = get_model_spec(model)
        tokenizer = AutoTokenizer.from_pretrained(
            str(spec.local_dir), trust_remote_code=spec.trust_remote_code,
            local_files_only=True, use_fast=False,
        )
        for item_index, (obj, material, attribute) in enumerate(ITEMS):
            for template in TEMPLATES:
                raw = raw_prompt(obj, material, attribute, template)
                prompt, add_special, answer_phase = interface_prompt(
                    tokenizer, model, raw, INTERFACE
                )
                rows.append({
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": PHASE,
                    "created_at": now(),
                    "case_id": f"phase338_{model}_{item_index:02d}_{template}",
                    "semantic_case_id": f"phase338_{item_index:02d}_{template}",
                    "model": model,
                    "family_id": "content_knowledge",
                    "mechanism_id": MECHANISM,
                    "item_index": item_index,
                    "split": split_for(item_index),
                    "template_id": template,
                    "interface": INTERFACE,
                    "answer_phase": answer_phase,
                    "prompt": prompt,
                    "raw_prompt": raw,
                    "tokenization_add_special_tokens": add_special,
                    "source_fragment": f"made from {material}",
                    "query_fragment": f"What material is the {obj} made from?",
                    "object": obj,
                    "target": material,
                    "target_aliases": [material],
                    "distractors": [attribute, "unknown"],
                    "target_class": "explicit_relation_binding_value",
                    "selection_eligible": item_index < 12,
                    "calibration_eligible": 12 <= item_index < 18,
                    "heldout_eligible": 18 <= item_index < 22,
                    "private_heldout_eligible": item_index >= 22,
                    "selection_updates_allowed": False,
                    "single_unit_intervention_allowed": False,
                })
    if len(rows) != 216:
        raise RuntimeError(f"Expected 216 cases, got {len(rows)}")
    if len({row["case_id"] for row in rows}) != len(rows):
        raise RuntimeError("Duplicate Phase338 case id")

    blocks = [
        {
            "block_id": f"{component}__{depth}__{role}",
            "component": component,
            "depth_bin": depth,
            "position_role": role,
        }
        for component in COMPONENTS for depth in DEPTH_BINS for role in POSITION_ROLES
    ]
    protocol = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE,
        "created_at": now(),
        "purpose": "Staged coarse-block causal screen after Phase337 protocol qualification.",
        "claim_scope": "Explicit material relation binding only; no parametric knowledge claim.",
        "models": list(MODELS),
        "interface": INTERFACE,
        "templates": list(TEMPLATES),
        "item_count": len(ITEMS),
        "registered_case_count": len(rows),
        "splits": {
            "discovery_items": list(range(12)),
            "calibration_items": list(range(12, 18)),
            "heldout_items": list(range(18, 22)),
            "private_heldout_items": [22, 23],
        },
        "blocks": blocks,
        "stages": {
            "discovery": "All 27 blocks; phrase margin; zero only.",
            "calibration": "Discovery top 3 per model; zero, half, permutation.",
            "heldout": (
                "One frozen block per model; baseline, zero, half, permutation, "
                "wrong depth, wrong position; phrase and rollout."
            ),
        },
        "thresholds": {
            "discovery_mean_phrase_loss_min": 0.1,
            "discovery_positive_case_rate_min": 0.6666667,
            "calibration_mean_phrase_loss_min": 0.1,
            "calibration_positive_case_rate_min": 0.6666667,
            "calibration_permutation_phrase_loss_min": 0.1,
            "heldout_phrase_loss_min": 0.2,
            "heldout_positive_case_rate_min": 0.6666667,
            "heldout_control_superiority_min": 0.05,
            "heldout_behavior_loss_rate_min": 0.25,
            "heldout_control_behavior_loss_rate_max": 0.1,
            "cross_model_pass_min": 2,
        },
        "claim_boundaries": [
            "A coarse block spans all layers in one depth third at one token role.",
            "Attribute binding is a real mechanism and is not treated as a null control.",
            "Same-block permutation tests content-structure sensitivity, not a null location control.",
            "No SAE or transcoder feature is merged into physical-neuron coordinates.",
            "No single-neuron claim is allowed.",
            "Private heldout rows never participate in candidate or threshold selection.",
        ],
    }
    write_jsonl(root / "phase338_registered_cases.jsonl", rows)
    write_json(root / "phase338_registered_protocol.json", protocol)
    write_jsonl(root / "phase338_registered_blocks.jsonl", blocks)
    validation = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE,
        "created_at": now(),
        "registered_case_count": len(rows),
        "registered_block_count": len(blocks),
        "model_case_count": {
            model: sum(row["model"] == model for row in rows) for model in MODELS
        },
        "split_case_count": {
            split: sum(row["split"] == split for row in rows)
            for split in ("discovery", "calibration", "heldout", "private_heldout")
        },
        "valid": True,
    }
    write_json(root / "phase338_case_bank_validation.json", validation)
    return validation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", default=ROUND_DEFAULT)
    args = parser.parse_args()
    print(json.dumps(register(args.round), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
