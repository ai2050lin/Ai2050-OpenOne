#!/usr/bin/env python3
"""Freeze the Phase331 expanded heldout and matched-control denominator."""

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


PHASE = "Phase331"
SCHEMA_VERSION = "9.0.0"
ROUND_DEFAULT = "refined_mechanism_audit"
SOURCE = ROOT / "tests/gpt5/result/phase330_nine_family_global_atlas/nine_family_global_atlas"
OUT = ROOT / "tests/gpt5/result/phase331_refined_mechanism_audit"
HELDOUT_ITEMS = (19, 20, 21, 22)
INTERFACES = ("raw_completion", "chat_template")

# Frozen before any Phase331 model execution. Controls are selected by same
# family and nearby task structure, not by Phase331 effect size.
PAIR_REGISTRY = (
    ("content_knowledge", "negated_attribute", "attribute"),
    ("language_action", "summarize", "rewrite"),
    ("language_action", "transform", "extract"),
    ("reasoning_constraint", "missing_condition_control", "two_hop_blocked"),
    ("syntax_structure", "singular_agreement", "plural_agreement"),
)

POSITIVE_CONDITIONS = (
    "baseline",
    "joint_set_zero",
    "attention_set_zero",
    "mlp_set_zero",
    "single_member_0_zero",
    "single_member_1_zero",
    "single_member_2_zero",
    "single_member_3_zero",
    "set_without_member_0_zero",
    "set_without_member_1_zero",
    "set_without_member_2_zero",
    "set_without_member_3_zero",
    "matched_random_joint_zero",
    "wrong_layer_joint_zero",
    "paired_control_joint_zero",
    "correct_donor_transplant",
    "wrong_donor_transplant",
    "same_target_donor_transplant",
    "matched_random_donor_transplant",
    "wrong_layer_donor_transplant",
    "correct_donor_restoration",
)

CONTROL_CONDITIONS = (
    "baseline",
    "joint_set_zero",
    "matched_random_joint_zero",
    "wrong_layer_joint_zero",
    "paired_positive_joint_zero",
)

GENERATION_CONDITIONS = (
    "baseline",
    "joint_set_zero",
    "matched_random_joint_zero",
    "wrong_layer_joint_zero",
    "paired_control_joint_zero",
    "paired_positive_joint_zero",
    "correct_donor_transplant",
    "wrong_donor_transplant",
    "same_target_donor_transplant",
    "matched_random_donor_transplant",
    "wrong_layer_donor_transplant",
    "correct_donor_restoration",
)

TRACE_CONDITIONS = POSITIVE_CONDITIONS


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def target_class(case: dict[str, Any]) -> str:
    if case["mechanism_id"] in {"summarize", "transform", "rewrite"}:
        return "transformed"
    if not bool(case["target_absent_from_prompt"]):
        return "present"
    return "absent"


def alternate_template(template_id: str) -> str:
    return {
        "template_a": "template_b",
        "template_b": "template_c",
        "template_c": "template_a",
    }[template_id]


def render_prompt(tokenizer: Any, raw_prompt: str, interface: str) -> tuple[str, bool]:
    if interface == "raw_completion":
        return raw_prompt, True
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": raw_prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt, False


def choose_random_donor(
    all_cases: list[dict[str, Any]], recipient: dict[str, Any], template_id: str
) -> dict[str, Any]:
    candidates = sorted(
        (
            row for row in all_cases
            if row["template_id"] == template_id
            and row["language"] == recipient["language"]
            and row["target_bucket"] == recipient["target_bucket"]
            and row["family_id"] != recipient["family_id"]
            and row["target"] != recipient["target"]
        ),
        key=lambda row: row["case_id"],
    )
    if not candidates:
        candidates = sorted(
            (
                row for row in all_cases
                if row["template_id"] == template_id
                and row["language"] == recipient["language"]
                and row["target_bucket"] == recipient["target_bucket"]
                and row["mechanism_id"] != recipient["mechanism_id"]
                and row["target"] != recipient["target"]
            ),
            key=lambda row: row["case_id"],
        )
    if not candidates:
        raise RuntimeError(f"No cross- or within-family matched random donor for {recipient['case_id']}")
    index = sum(ord(char) for char in recipient["case_id"]) % len(candidates)
    return candidates[index]


def register(round_name: str = ROUND_DEFAULT) -> dict[str, Any]:
    root = OUT / round_name
    phase330_cases = read_jsonl(SOURCE / "phase330_case_bank.jsonl")
    by_key = {
        (row["family_id"], row["mechanism_id"], row["item_index"], row["template_id"]): row
        for row in phase330_cases
    }
    cross_rows = read_jsonl(SOURCE / "cross_model_mechanism_summary.jsonl")
    cross = {(row["family_id"], row["mechanism_id"]): row for row in cross_rows}
    rows: list[dict[str, Any]] = []
    tokenizers: dict[str, Any] = {}
    try:
        for model in MODELS:
            spec = get_model_spec(model)
            tokenizer = AutoTokenizer.from_pretrained(
                str(spec.local_dir), trust_remote_code=spec.trust_remote_code,
                local_files_only=True, use_fast=False,
            )
            tokenizers[model] = tokenizer
            for family, positive, control in PAIR_REGISTRY:
                if not cross[(family, positive)]["cross_model_joint_readout_specific"]:
                    raise RuntimeError(f"Positive mechanism no longer passes Phase330 gate: {family}/{positive}")
                if cross[(family, control)]["cross_model_joint_readout_specific"]:
                    raise RuntimeError(f"Matched control unexpectedly passes Phase330 gate: {family}/{control}")
                for cohort, mechanism, paired in (
                    ("positive", positive, control),
                    ("matched_negative_control", control, positive),
                ):
                    for item_index in HELDOUT_ITEMS:
                        wrong_item = HELDOUT_ITEMS[(HELDOUT_ITEMS.index(item_index) + 1) % len(HELDOUT_ITEMS)]
                        for template_id in TEMPLATES:
                            recipient = by_key[(family, mechanism, item_index, template_id)]
                            donor_template = alternate_template(template_id)
                            correct_donor = by_key[(family, mechanism, item_index, donor_template)]
                            wrong_donor = by_key[(family, mechanism, wrong_item, donor_template)]
                            random_donor = choose_random_donor(phase330_cases, recipient, donor_template)
                            for interface in INTERFACES:
                                prompt, add_special_tokens = render_prompt(tokenizer, recipient["prompt"], interface)
                                correct_prompt, _ = render_prompt(tokenizer, correct_donor["prompt"], interface)
                                wrong_prompt, _ = render_prompt(tokenizer, wrong_donor["prompt"], interface)
                                random_prompt, _ = render_prompt(tokenizer, random_donor["prompt"], interface)
                                audit_case_id = (
                                    f"phase331_{model}_{cohort}_{family}_{mechanism}_{item_index:02d}_"
                                    f"{template_id}_{interface}"
                                )
                                rows.append({
                                    "schema_version": SCHEMA_VERSION,
                                    "phase_id": PHASE,
                                    "created_at": now(),
                                    "audit_case_id": audit_case_id,
                                    "model": model,
                                    "cohort": cohort,
                                    "family_id": family,
                                    "mechanism_id": mechanism,
                                    "paired_mechanism_id": paired,
                                    "item_index": item_index,
                                    "template_id": template_id,
                                    "interface": interface,
                                    "split": "expanded_heldout",
                                    "source_case_id": recipient["case_id"],
                                    "correct_donor_case_id": correct_donor["case_id"],
                                    "wrong_donor_case_id": wrong_donor["case_id"],
                                    "matched_random_donor_case_id": random_donor["case_id"],
                                    "prompt": prompt,
                                    "raw_prompt": recipient["prompt"],
                                    "correct_donor_prompt": correct_prompt,
                                    "correct_donor_source_fragments": correct_donor["source_fragments"],
                                    "correct_donor_query_fragment": correct_donor["query_fragment"],
                                    "wrong_donor_prompt": wrong_prompt,
                                    "wrong_donor_source_fragments": wrong_donor["source_fragments"],
                                    "wrong_donor_query_fragment": wrong_donor["query_fragment"],
                                    "matched_random_donor_prompt": random_prompt,
                                    "matched_random_donor_source_fragments": random_donor["source_fragments"],
                                    "matched_random_donor_query_fragment": random_donor["query_fragment"],
                                    "tokenization_add_special_tokens": add_special_tokens,
                                    "source_fragments": recipient["source_fragments"],
                                    "query_fragment": recipient["query_fragment"],
                                    "target": recipient["target"],
                                    "target_aliases": recipient["target_aliases"],
                                    "distractors": recipient["distractors"],
                                    "target_bucket": recipient["target_bucket"],
                                    "target_class": target_class(recipient),
                                    "target_absent_from_prompt": recipient["target_absent_from_prompt"],
                                    "protocol": recipient["protocol"],
                                    "expected_structure": recipient["expected_structure"],
                                    "language": recipient["language"],
                                    "condition_count": len(POSITIVE_CONDITIONS if cohort == "positive" else CONTROL_CONDITIONS),
                                    "selection_updates_allowed": False,
                                    "phase330_denominator_updates_allowed": False,
                                    "single_unit_intervention_gate_open": False,
                                })
    finally:
        tokenizers.clear()

    expected = len(MODELS) * len(PAIR_REGISTRY) * 2 * len(HELDOUT_ITEMS) * len(TEMPLATES) * len(INTERFACES)
    if len(rows) != expected:
        raise RuntimeError(f"Expected {expected} interface cases, got {len(rows)}")
    if len({row["audit_case_id"] for row in rows}) != len(rows):
        raise RuntimeError("Duplicate Phase331 audit_case_id")
    write_jsonl(root / "phase331_registered_cases.jsonl", rows)
    protocol = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE,
        "created_at": now(),
        "purpose": "Refine only the five frozen Phase330 cross-model set-readout candidates.",
        "positive_mechanisms": [f"{family}/{positive}" for family, positive, _ in PAIR_REGISTRY],
        "matched_negative_controls": [f"{family}/{control}" for family, _, control in PAIR_REGISTRY],
        "heldout_item_indices": list(HELDOUT_ITEMS),
        "templates": list(TEMPLATES),
        "interfaces": list(INTERFACES),
        "models": list(MODELS),
        "positive_interface_case_count": expected // 2,
        "control_interface_case_count": expected // 2,
        "interface_case_count": expected,
        "positive_conditions": list(POSITIVE_CONDITIONS),
        "control_conditions": list(CONTROL_CONDITIONS),
        "generation_conditions": list(GENERATION_CONDITIONS),
        "trace_conditions": list(TRACE_CONDITIONS),
        "trace_template": "template_c",
        "trace_scope": "all registered intervention conditions on template_c for every item and interface",
        "thresholds": {
            "item_direction_consistency_min": 0.75,
            "joint_margin_delta_max": -0.05,
            "member_margin_delta_max": -0.05,
            "generation_behavior_change_min": 0.10,
            "paired_control_behavior_change_max": 0.10,
            "paired_control_protocol_loss_max": 0.10,
            "nonselected_compensation_ratio_max": 1.25,
            "late_residual_recovery_fraction_max": 0.25,
        },
        "success_gate": [
            "readout_specific", "expanded_heldout", "cross_interface", "cross_model",
            "member_localized", "compensation_accounted", "full_generation_changed", "low_side_effect",
        ],
        "denominator_frozen": True,
        "selection_updates_allowed": False,
        "phase330_denominator_updates_allowed": False,
        "single_unit_intervention_gate_open": False,
        "theory_update_gate_open": False,
    }
    write_json(root / "phase331_registered_protocol.json", protocol)
    validation = {
        "phase_id": PHASE,
        "created_at": now(),
        "interface_case_count": len(rows),
        "positive_interface_case_count": sum(row["cohort"] == "positive" for row in rows),
        "control_interface_case_count": sum(row["cohort"] == "matched_negative_control" for row in rows),
        "model_counts": {model: sum(row["model"] == model for row in rows) for model in MODELS},
        "mechanism_count": len({(row["family_id"], row["mechanism_id"]) for row in rows}),
        "target_class_counts": {
            value: sum(row["target_class"] == value for row in rows)
            for value in ("present", "absent", "transformed")
        },
        "all_phase330_heldout": all(row["split"] == "expanded_heldout" for row in rows),
        "valid": len(rows) == 720,
    }
    write_json(root / "phase331_case_bank_validation.json", validation)
    return validation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", default=ROUND_DEFAULT)
    args = parser.parse_args()
    print(json.dumps(register(args.round), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
