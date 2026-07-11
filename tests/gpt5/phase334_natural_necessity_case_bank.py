#!/usr/bin/env python3
"""Freeze the Phase334 three-family natural-necessity denominator."""

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
from phase333_dynamic_case_bank import INTERFACES, interface_prompt  # noqa: E402


PHASE = "Phase334"
SCHEMA_VERSION = "12.0.0"
ROUND_DEFAULT = "natural_necessity_atlas"
OUT = ROOT / "tests/gpt5/result/phase334_natural_necessity_atlas"

MECHANISM_PAIRS = (
    ("content_knowledge", "material", "primary", "attribute"),
    ("content_knowledge", "attribute", "matched_comparator", "material"),
    ("reasoning_constraint", "missing_condition_control", "primary", "two_hop_blocked"),
    ("reasoning_constraint", "two_hop_blocked", "matched_comparator", "missing_condition_control"),
    ("syntax_structure", "past_tense", "primary", "plural_agreement"),
    ("syntax_structure", "plural_agreement", "matched_comparator", "past_tense"),
)

KNOWLEDGE_ITEMS = (
    ("astrolabe", "brass", "engraved"),
    ("kettle", "steel", "dented"),
    ("compass", "aluminum", "calibrated"),
    ("violin", "maple", "varnished"),
    ("helmet", "carbon", "reinforced"),
    ("jar", "ceramic", "glazed"),
    ("cable", "copper", "insulated"),
    ("bench", "oak", "weathered"),
    ("lens", "glass", "polished"),
    ("tire", "rubber", "treaded"),
    ("scarf", "wool", "striped"),
    ("panel", "acrylic", "translucent"),
)

REASONING_ITEMS = (
    ("Aren", "quiet", "round", "approved"),
    ("Bela", "bright", "level", "accepted"),
    ("Ciro", "calm", "solid", "recorded"),
    ("Dena", "clean", "wide", "released"),
    ("Eren", "dry", "smooth", "verified"),
    ("Fara", "firm", "pale", "stored"),
    ("Gino", "gentle", "square", "tagged"),
    ("Hela", "light", "steady", "tested"),
    ("Iven", "narrow", "warm", "admitted"),
    ("Juna", "plain", "soft", "certified"),
    ("Karo", "rapid", "thin", "listed"),
    ("Lena", "scarlet", "tall", "reserved"),
)

PAST_ITEMS = (
    ("pilot", "inspect", "inspected", "hangar"),
    ("chemist", "balance", "balanced", "sample"),
    ("artist", "polish", "polished", "frame"),
    ("surveyor", "measure", "measured", "bridge"),
    ("clerk", "arrange", "arranged", "ledger"),
    ("mechanic", "repair", "repaired", "motor"),
    ("sailor", "signal", "signaled", "harbor"),
    ("nurse", "record", "recorded", "reading"),
    ("guard", "open", "opened", "gate"),
    ("vendor", "close", "closed", "stall"),
    ("engineer", "test", "tested", "circuit"),
    ("porter", "clean", "cleaned", "platform"),
)

AGREEMENT_ITEMS = (
    ("pilots", "are", "is", "ready"),
    ("chemists", "have", "has", "notes"),
    ("artists", "work", "works", "quietly"),
    ("surveyors", "remain", "remains", "outside"),
    ("clerks", "seem", "seems", "careful"),
    ("mechanics", "move", "moves", "quickly"),
    ("sailors", "stand", "stands", "nearby"),
    ("nurses", "look", "looks", "prepared"),
    ("guards", "run", "runs", "daily"),
    ("vendors", "rest", "rests", "indoors"),
    ("engineers", "sound", "sounds", "confident"),
    ("porters", "glow", "glows", "with pride"),
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


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


def task_for(family: str, mechanism: str, item_index: int) -> dict[str, Any]:
    if family == "content_knowledge":
        name, material, attribute = KNOWLEDGE_ITEMS[item_index]
        context = f"The archive states that the {name} is made from {material} and is {attribute}."
        if mechanism == "material":
            return {
                "context": context,
                "question": f"What material is the {name} made from?",
                "instruction": "Answer with the one stated material word only.",
                "target": material,
                "target_aliases": [material],
                "distractors": [attribute, "unknown"],
                "source_fragment": f"made from {material}",
                "query_fragment": f"What material is the {name} made from?",
                "target_class": "explicit_material_relation",
            }
        return {
            "context": context,
            "question": f"Which stated attribute describes the {name}?",
            "instruction": "Answer with the one stated attribute word only.",
            "target": attribute,
            "target_aliases": [attribute],
            "distractors": [material, "unknown"],
            "source_fragment": f"is {attribute}",
            "query_fragment": f"Which stated attribute describes the {name}?",
            "target_class": "explicit_attribute_relation",
        }
    if family == "reasoning_constraint":
        name, prop_a, prop_b, conclusion = REASONING_ITEMS[item_index]
        if mechanism == "missing_condition_control":
            context = (
                f"If something is {prop_a} and {prop_b}, then it is {conclusion}. "
                f"{name} is {prop_a}. No fact states whether {name} is {prop_b}."
            )
            return {
                "context": context,
                "question": f"Is {name} definitely {conclusion}?",
                "instruction": "Answer yes, no, or unknown only.",
                "target": "unknown",
                "target_aliases": ["unknown", "cannot determine", "not enough information"],
                "distractors": ["yes", "no"],
                "source_fragment": f"No fact states whether {name} is {prop_b}",
                "query_fragment": f"Is {name} definitely {conclusion}?",
                "target_class": "missing_premise",
            }
        group = f"set{item_index + 51}"
        context = (
            f"Every {group} is {prop_a}. No {prop_a} thing is {prop_b}. "
            f"{name} is a {group}."
        )
        return {
            "context": context,
            "question": f"Is {name} {prop_b}?",
            "instruction": "Answer yes, no, or unknown only.",
            "target": "no",
            "target_aliases": ["no", "false", "contradicted"],
            "distractors": ["yes", "unknown"],
            "source_fragment": f"No {prop_a} thing is {prop_b}",
            "query_fragment": f"Is {name} {prop_b}?",
            "target_class": "two_hop_contradiction",
        }
    if mechanism == "past_tense":
        subject, base, past, obj = PAST_ITEMS[item_index]
        context = f"Yesterday, the {subject} needed to {base} the {obj}."
        return {
            "context": context,
            "question": f"Complete the sentence: Yesterday, the {subject} ___ the {obj}.",
            "instruction": "Answer with the one correct past-tense verb only.",
            "target": past,
            "target_aliases": [past],
            "distractors": [base, f"{base}s"],
            "source_fragment": f"Yesterday, the {subject} needed to {base}",
            "query_fragment": f"Yesterday, the {subject} ___ the {obj}",
            "target_class": "past_tense_inflection",
        }
    subject, target, distractor, tail = AGREEMENT_ITEMS[item_index]
    context = f"Use standard subject-verb agreement for this sentence about {subject}."
    return {
        "context": context,
        "question": f"Complete the sentence: The {subject} ___ {tail}.",
        "instruction": "Answer with the one correct verb only.",
        "target": target,
        "target_aliases": [target],
        "distractors": [distractor, "unknown"],
        "source_fragment": f"subject-verb agreement",
        "query_fragment": f"The {subject} ___ {tail}",
        "target_class": "plural_agreement_inflection",
    }


def render_raw(task: dict[str, Any], template_id: str) -> str:
    if template_id == "template_a":
        return f"{task['context']}\n{task['question']}\n{task['instruction']}\nAnswer:"
    if template_id == "template_b":
        return (
            f"Information: {task['context']}\nTask: {task['question']}\n"
            f"Output rule: {task['instruction']}\nResponse:"
        )
    if template_id == "template_c":
        return f"{task['context']}\n{task['instruction']}\nPrompt: {task['question']}\nResult:"
    raise KeyError(template_id)


def checkpoint_audit() -> list[dict[str, Any]]:
    rows = []
    for model in MODELS:
        spec = get_model_spec(model)
        checkpoints = sorted(str(path) for path in spec.local_dir.glob("checkpoint-*"))
        rows.append({
            "model": model,
            "local_model_dir": str(spec.local_dir),
            "same_run_checkpoint_count": len(checkpoints),
            "checkpoint_paths": checkpoints,
            "training_formation_track_available": len(checkpoints) >= 2,
        })
    return rows


def register(round_name: str = ROUND_DEFAULT) -> dict[str, Any]:
    root = OUT / round_name
    rows = []
    for model in MODELS:
        spec = get_model_spec(model)
        tokenizer = AutoTokenizer.from_pretrained(
            str(spec.local_dir), trust_remote_code=spec.trust_remote_code,
            local_files_only=True, use_fast=False,
        )
        for family, mechanism, cohort, paired in MECHANISM_PAIRS:
            for item_index in range(12):
                task = task_for(family, mechanism, item_index)
                for template_id in TEMPLATES:
                    raw_prompt = render_raw(task, template_id)
                    for interface in INTERFACES:
                        prompt, add_special, answer_phase = interface_prompt(
                            tokenizer, model, raw_prompt, interface
                        )
                        rows.append({
                            "schema_version": SCHEMA_VERSION,
                            "phase_id": PHASE,
                            "created_at": now(),
                            "case_id": (
                                f"phase334_{model}_{mechanism}_{item_index:02d}_"
                                f"{template_id}_{interface}"
                            ),
                            "semantic_case_id": (
                                f"phase334_{mechanism}_{item_index:02d}_{template_id}_{interface}"
                            ),
                            "item_id": f"phase334_{mechanism}_{item_index:02d}",
                            "model": model,
                            "family_id": family,
                            "mechanism_id": mechanism,
                            "cohort": cohort,
                            "paired_mechanism_id": paired,
                            "item_index": item_index,
                            "split": split_for(item_index),
                            "template_id": template_id,
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
                            "target_class": task["target_class"],
                            "language": "en",
                            "protocol": "single_word_or_short_verdict",
                            "expected_structure": "plain",
                            "selection_eligible": item_index < 6,
                            "calibration_eligible": 6 <= item_index < 9,
                            "heldout_eligible": item_index >= 9,
                            "selection_updates_allowed": False,
                            "single_unit_intervention_gate_open": False,
                        })
    if len(rows) != 1944:
        raise RuntimeError(f"Expected 1944 cases, got {len(rows)}")
    if len({row["case_id"] for row in rows}) != len(rows):
        raise RuntimeError("Duplicate Phase334 case id")
    audits = checkpoint_audit()
    write_jsonl(root / "phase334_registered_cases.jsonl", rows)
    write_json(root / "phase334_training_checkpoint_audit.json", {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE,
        "created_at": now(),
        "models": audits,
        "all_models_have_training_formation_track": all(
            row["training_formation_track_available"] for row in audits
        ),
    })
    protocol = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE,
        "created_at": now(),
        "purpose": "Map natural receiver-path necessity across knowledge, reasoning, and syntax.",
        "families": ["content_knowledge", "reasoning_constraint", "syntax_structure"],
        "mechanism_pairs": [list(row) for row in MECHANISM_PAIRS],
        "new_item_count_per_mechanism": 12,
        "discovery_items": list(range(6)),
        "calibration_items": [6, 7, 8],
        "heldout_items": [9, 10, 11],
        "templates": list(TEMPLATES),
        "interfaces": list(INTERFACES),
        "models": list(MODELS),
        "registered_case_count": 1944,
        "max_new_tokens": 24,
        "candidate_components": ["attention_output", "mlp_output", "residual_increment"],
        "position_roles": ["source", "query", "answer_start"],
        "depth_bins": ["early", "middle", "late"],
        "baseline_eligibility": {
            "behavior_success_required": True,
            "target_rank_max": 50,
            "finite_phrase_logprob_required": True,
            "model_cell_eligible_cases_min": 6,
        },
        "calibration_selection": (
            "Within each model/mechanism/interface, select one of three discovery-frozen "
            "depth candidates on calibration-only natural deletion loss."
        ),
        "heldout_conditions": [
            "baseline", "correct_selected_delete", "correct_attention_delete",
            "correct_mlp_delete", "correct_residual_delete", "correct_joint_delete",
            "wrong_time_delete", "wrong_object_increment", "matched_mechanism_increment",
            "moment_matched_permutation", "wrong_layer_delete",
        ],
        "thresholds": {
            "common_valid_case_count_min": 6,
            "phrase_logprob_loss_min": 0.1,
            "target_rank_loss_min": 1.0,
            "behavior_loss_rate_min": 0.1,
            "control_superiority_min": 0.05,
            "protocol_side_effect_max": 0.1,
            "propagation_case_rate_min": 0.6666667,
        },
        "claim_boundaries": [
            "The knowledge pair tests explicit relation binding, not parametric-memory origin.",
            "Matched comparators are real mechanisms, not expected-null negative controls.",
            "Natural contrast magnitude selects candidates without a target-output direction.",
            "No training-formation claim is allowed without same-run checkpoints.",
            "No single-unit claim is allowed in Phase334.",
        ],
    }
    write_json(root / "phase334_registered_protocol.json", protocol)
    quality = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE,
        "created_at": now(),
        "registered_case_count": len(rows),
        "model_case_count": {model: sum(row["model"] == model for row in rows) for model in MODELS},
        "family_case_count": {
            family: sum(row["family_id"] == family for row in rows)
            for family in ("content_knowledge", "reasoning_constraint", "syntax_structure")
        },
        "mechanism_count": len({row["mechanism_id"] for row in rows}),
        "semantic_case_count": len({row["semantic_case_id"] for row in rows}),
        "training_formation_track_available_count": sum(
            row["training_formation_track_available"] for row in audits
        ),
        "valid": True,
    }
    write_json(root / "phase334_case_bank_validation.json", quality)
    return quality


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", default=ROUND_DEFAULT)
    args = parser.parse_args()
    print(json.dumps(register(args.round), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
