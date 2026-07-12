#!/usr/bin/env python3
"""Freeze a fresh six-mechanism case bank for Phase386 multi-time relations."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from model_registry import get_model_spec  # noqa: E402
from phase333_dynamic_case_bank import interface_prompt  # noqa: E402


OUT = ROOT / "tests/gpt5/result/phase386_multitime_relation_atlas"
MODELS = ("qwen3", "glm4", "deepseek7b")
MECHANISMS = (
    ("content_knowledge", "relation_binding"),
    ("state_drift", "entity_recency"),
    ("syntax_structure", "number_agreement"),
    ("readout_competition", "target_vs_wrong"),
    ("reasoning_constraint", "missing_condition_control"),
    ("language_action", "field_extraction"),
)
CONDITIONS = (
    "A_operation_lex_x",
    "B_control_lex_x",
    "C_operation_lex_y",
    "D_control_lex_y",
)
CANDIDATE_GROUPS_PER_MECHANISM = 40
FROZEN_SPLIT_GROUPS = {"discovery": 8, "calibration": 4, "physical_holdout": 4}
TEMPLATES = (
    "Sealed Phase386 ledger. {context}\nRequest: {question}\nContract: {instruction}\nAnswer:",
    "Use only this isolated record. {context}\n{instruction}\nQuestion: {question}\nValue:",
    "Independent verification card: {context}\nTask: {question}\nOutput rule: {instruction}\nResult:",
    "Fresh causal-sequence note. {context}\nRequested field: {question}\n{instruction}\nReply:",
    "New evidence packet: {context}\nConstraint: {instruction}\nQuery: {question}\nAnswer:",
)
OBJECTS = (
    "astrolabe", "binder", "casket", "docket", "emblem", "folder", "gauge", "hammock",
    "insignia", "journal", "keystone", "ledger", "mallet", "nozzle", "obelisk", "pouch",
    "quiver", "rudder", "satchel", "trowel", "utensil", "vial", "winch", "xylophone",
    "yardstick", "zipper", "anvil", "brooch", "cylinder", "decanter", "eyepiece", "fan",
    "grinder", "hinge", "index", "jigsaw", "knob", "latch", "mosaic", "needle",
)
VALUES = (
    "acrylic", "burlap", "charcoal", "dolomite", "enamel", "fiberglass", "graphite", "hickory",
    "insulation", "jute", "kaolin", "limestone", "mica", "neoprene", "onyx", "pumice",
    "rattan", "suede", "travertine", "urethane", "vinyl", "wax", "yarn", "zinc",
    "alloy", "brick", "cork", "dacron", "epoxy", "flannel", "garnet", "hardwood",
    "ink", "jasper", "kraft", "laminate", "mesh", "nylon", "opal", "plywood",
)
NAMES = (
    "Alden", "Blythe", "Corin", "Daphne", "Eamon", "Fiona", "Gideon", "Hadley",
    "Isla", "Julian", "Kendra", "Lars", "Maeve", "Noel", "Orla", "Quentin",
    "Rhea", "Soren", "Thalia", "Uriah", "Valerie", "Warren", "Xanthe", "Yvette",
    "Arlo", "Beatrix", "Clive", "Daria", "Elio", "Fern", "Galen", "Holly",
    "Ivo", "Juno", "Kellan", "Lyra", "Marek", "Nadia", "Otto", "Petra",
)
NOUNS = (
    ("badge", "badges"), ("courier", "couriers"), ("diagram", "diagrams"),
    ("engine", "engines"), ("flag", "flags"), ("guide", "guides"),
    ("hammer", "hammers"), ("invoice", "invoices"), ("jewel", "jewels"),
    ("kernel", "kernels"), ("label", "labels"), ("map", "maps"),
    ("needle", "needles"), ("operator", "operators"), ("permit", "permits"),
    ("queue", "queues"), ("receipt", "receipts"), ("signal", "signals"),
    ("ticket", "tickets"), ("unit", "units"), ("valve", "valves"),
    ("wafer", "wafers"), ("axis", "axes"), ("yarn", "yarns"),
    ("archive", "archives"), ("bridge", "bridges"), ("column", "columns"),
    ("device", "devices"), ("entry", "entries"), ("frame", "frames"),
    ("groove", "grooves"), ("handle", "handles"), ("item", "items"),
    ("junction", "junctions"), ("keypad", "keypads"), ("lever", "levers"),
    ("marker", "markers"), ("node", "nodes"), ("outlet", "outlets"),
    ("panel", "panels"),
)
LABELS = (
    "alpine", "bright", "cobalt", "dune", "elm", "frost", "glacier", "hazel",
    "ivory", "juniper", "khaki", "lagoon", "mint", "night", "ocean", "peach",
    "quartz", "rust", "spruce", "topaz", "ultramarine", "verdant", "wheat", "xanthic",
    "yarrow", "zinnia", "ash", "berry", "citrine", "drift", "ecru", "fern",
    "ginger", "heather", "iris", "jet", "kelp", "lemon", "mauve", "nutmeg",
)
PREPOSITIONS = ("near", "beside", "behind", "opposite", "beyond", "alongside")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(value: str, length: int = 64) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


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
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)
                + "\n"
            )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def task(mechanism: str, group_index: int, lexical_slot: str, operation: bool) -> dict[str, Any]:
    offset = 0 if lexical_slot == "x" else 19
    index = (group_index + offset) % CANDIDATE_GROUPS_PER_MECHANISM
    code = f"p386-{mechanism[:4]}-{group_index:02d}-{lexical_slot}"
    if mechanism == "relation_binding":
        obj, target = OBJECTS[index], VALUES[index]
        wrong = VALUES[(index + 11) % len(VALUES)]
        site = f"sector-{code}"
        context = (
            f"The {obj}-{code} is registered in {site}. Every object registered in {site} is made of {target}."
            if operation
            else f"The {obj}-{code} is made of {target}. Its registration site is {site}."
        )
        question = f"What is the {obj}-{code} made of?"
        instruction = "Return only the material without explanation."
    elif mechanism == "entity_recency":
        target = NAMES[index]
        wrong = NAMES[(index + 13) % len(NAMES)]
        context = (
            f"The verified custodian for {code} is {target}. A later unrelated memo mentions {wrong}."
            if operation
            else f"An unrelated memo mentions {wrong}. The verified custodian for {code} is {target}."
        )
        question = f"Who is the verified custodian for {code}?"
        instruction = "Return only the person's name without explanation."
    elif mechanism == "number_agreement":
        singular, plural = NOUNS[index]
        plural_head = index % 2 == 0
        head = plural if plural_head else singular
        target, wrong = ("are", "is") if plural_head else ("is", "are")
        attr_singular, attr_plural = NOUNS[(index + 9) % len(NOUNS)]
        attractor = attr_singular if plural_head else attr_plural
        subject = (
            f"the {head} {PREPOSITIONS[group_index % len(PREPOSITIONS)]} the {attractor}"
            if operation
            else f"the {head}"
        )
        context = (
            f"Record {code}: '{head}' is the head noun; the noun after the preposition cannot control agreement."
            if operation
            else f"Record {code}: apply ordinary agreement to the head noun '{head}'."
        )
        question = f"Fill only the blank: {subject.capitalize()} ___ ready."
        instruction = "Return exactly one word: is or are."
    elif mechanism == "target_vs_wrong":
        target = LABELS[index]
        wrong = LABELS[(index + 17) % len(LABELS)]
        context = (
            f"Register {code} explicitly approves {target} and explicitly rejects {wrong}."
            if operation
            else f"Register {code} approves {target}; unrelated rejected labels must be ignored."
        )
        question = f"Which label does register {code} approve?"
        instruction = "Return only the approved label without explanation."
    elif mechanism == "missing_condition_control":
        prop_a = LABELS[index]
        prop_b = LABELS[(index + 7) % len(LABELS)]
        conclusion = LABELS[(index + 14) % len(LABELS)]
        subject = f"item-{code}"
        target, wrong = "unknown", "yes"
        context = (
            f"If something is {prop_a} and {prop_b}, then it is {conclusion}. {subject} is {prop_a}. No fact states whether {subject} is {prop_b}."
            if operation
            else f"The sealed record explicitly says that whether {subject} is {conclusion} cannot be determined."
        )
        question = f"Is {subject} definitely {conclusion}?"
        instruction = "Answer yes, no, or unknown only."
    elif mechanism == "field_extraction":
        target = LABELS[index]
        wrong = LABELS[(index + 15) % len(LABELS)]
        batch = LABELS[(index + 5) % len(LABELS)]
        owner = NAMES[(index + 3) % len(NAMES)]
        context = (
            f"Record {code} has owner={owner}; batch={batch}; status={target}; rejected_status={wrong}."
            if operation
            else f"Record {code} has status={target}."
        )
        question = f"Extract the status field from record {code}."
        instruction = "Return only the status value without explanation."
    else:
        raise KeyError(mechanism)
    return {
        "context": context,
        "question": question,
        "instruction": instruction,
        "target": target,
        "target_aliases": [target],
        "distractors": [wrong],
    }


def main() -> None:
    created_at = now()
    execution_rows: list[dict[str, Any]] = []
    blind_rows: list[dict[str, Any]] = []
    prompt_hashes: set[str] = set()
    for model in MODELS:
        spec = get_model_spec(model)
        tokenizer = AutoTokenizer.from_pretrained(
            str(spec.local_dir),
            trust_remote_code=spec.trust_remote_code,
            local_files_only=True,
            use_fast=False,
        )
        for family, mechanism in MECHANISMS:
            for group_index in range(CANDIDATE_GROUPS_PER_MECHANISM):
                semantic_group = f"phase386_{family}_{mechanism}_{group_index:02d}"
                parallel_group = "parallel386_" + digest(semantic_group, 20)
                model_group = "group386_" + digest(f"{model}:{semantic_group}", 20)
                items = {
                    "A": task(mechanism, group_index, "x", True),
                    "B": task(mechanism, group_index, "x", False),
                    "C": task(mechanism, group_index, "y", True),
                    "D": task(mechanism, group_index, "y", False),
                }
                for condition in CONDITIONS:
                    item = items[condition[0]]
                    raw_prompt = TEMPLATES[group_index % len(TEMPLATES)].format(**item)
                    prompt, add_special, answer_phase = interface_prompt(
                        tokenizer, model, raw_prompt, "answer_aligned_chat"
                    )
                    prompt_hash = digest(prompt)
                    if prompt_hash in prompt_hashes:
                        raise RuntimeError("Duplicate Phase386 rendered prompt")
                    prompt_hashes.add(prompt_hash)
                    case_id = "p386c_" + digest(
                        f"{model}:{semantic_group}:{condition}", 26
                    )
                    common = {
                        "schema_version": "60.0.0",
                        "phase_id": "Phase386-Protocol",
                        "created_at": created_at,
                        "blind_case_id": case_id,
                        "anonymous_model_id": "am386_" + digest(model, 12),
                        "anonymous_parallel_group_id": parallel_group,
                        "anonymous_group_id": model_group,
                        "anonymous_condition_slot": "slot386_"
                        + digest(f"{model_group}:{condition}", 12),
                        "prompt": prompt,
                        "raw_prompt": raw_prompt,
                        "source_fragment": item["context"],
                        "query_fragment": item["question"],
                        "prompt_token_count": len(
                            tokenizer(prompt, add_special_tokens=False)["input_ids"]
                        ),
                        "tokenization_add_special_tokens": add_special,
                        "interface": "answer_aligned_chat",
                        "answer_phase": answer_phase,
                    }
                    execution_rows.append(
                        {
                            **common,
                            "private_execution_model": model,
                            "family_id": family,
                            "mechanism_id": mechanism,
                            "semantic_group_id": semantic_group,
                            "contrast_condition": condition,
                            "operation_demanded": condition[0] in {"A", "C"},
                            "target": item["target"],
                            "target_aliases": item["target_aliases"],
                            "distractors": item["distractors"],
                            "language": "en",
                        }
                    )
                    blind_rows.append(
                        {
                            **common,
                            "semantic_label_used_for_collection": False,
                            "target_or_distractor_exported": False,
                        }
                    )

    expected = (
        len(MODELS)
        * len(MECHANISMS)
        * CANDIDATE_GROUPS_PER_MECHANISM
        * len(CONDITIONS)
    )
    if len(execution_rows) != expected or len(prompt_hashes) != expected:
        raise RuntimeError(
            f"Invalid Phase386 case bank: rows={len(execution_rows)} hashes={len(prompt_hashes)}"
        )
    prior_files = (
        ROOT
        / "tests/gpt5/result/phase380_independent_layout_validation/private/phase380_execution_cases.jsonl",
        ROOT
        / "tests/gpt5/result/phase380_independent_layout_validation/number_agreement_expansion/private/phase380x_execution_cases.jsonl",
        ROOT
        / "tests/gpt5/result/phase381_joint_state_formation/private/phase381_execution_cases.jsonl",
    )
    prior_prompt_hashes = {
        digest(row["prompt"])
        for path in prior_files
        for row in read_jsonl(path)
        if row.get("prompt")
    }
    prior_overlap_count = len(prompt_hashes & prior_prompt_hashes)
    if prior_overlap_count:
        raise RuntimeError(f"Phase386 prior prompt overlap: {prior_overlap_count}")

    private = OUT / "protocol/private"
    write_jsonl(private / "phase386_candidate_execution_cases.jsonl", execution_rows)
    write_jsonl(OUT / "protocol/phase386_blind_case_registry.jsonl", blind_rows)
    protocol = {
        "schema_version": "60.0.0",
        "phase_id": "Phase386-Protocol",
        "created_at": created_at,
        "objective": "build_a_fresh_multitime_exact_event_relation_graph",
        "denominator": {
            "models": list(MODELS),
            "families": [family for family, _ in MECHANISMS],
            "mechanisms": [mechanism for _, mechanism in MECHANISMS],
            "candidate_groups_per_mechanism": CANDIDATE_GROUPS_PER_MECHANISM,
            "conditions_per_group": len(CONDITIONS),
            "candidate_parallel_group_count": len(MECHANISMS)
            * CANDIDATE_GROUPS_PER_MECHANISM,
            "candidate_case_count": len(execution_rows),
            "frozen_split_groups_per_qualified_mechanism": FROZEN_SPLIT_GROUPS,
        },
        "runtime_contract": {
            "execution_batch_size": 1,
            "output_attentions": True,
            "dtype_by_model": {
                "qwen3": "float16",
                "glm4": "float16",
                "deepseek7b": "bfloat16",
            },
            "model_order": list(MODELS),
        },
        "semantic_coordinates": {
            "source_encoded": "source token position in the predecision causal forward",
            "query_integrated": "query token position in the predecision causal forward",
            "pre_decision": "last position before the first target token",
            "target_encoded": "first target token after it is appended",
            "post_decision_next_token": "next generated token after the first target token",
            "forward_pass_count_per_case": 3,
            "independent_clock_times_claimed": False,
        },
        "exact_event_contract": {
            "top_k_used": False,
            "parent_child_conservation_max_relative_error": 0.01,
            "full_pairwise_channel_gram_materialized": False,
            "vector_cancellation": "1-norm(sum(delta_child))/(sum(norm(delta_child))+epsilon)",
            "terminal_projection_cancellation_retained": True,
            "composite_relation_score_used": False,
        },
        "relation_families": [
            "physical_parent_child",
            "adjacent_layer_continuity",
            "semantic_coordinate_persistence",
            "appearance_disappearance",
            "sign_reversal",
            "descriptive_compensation",
            "multi_event_convergence",
        ],
        "separate_gates": [
            "single_sample_behavior",
            "component_conservation",
            "wrong_time",
            "wrong_depth",
            "wrong_receiver",
            "time_order_shuffle",
            "independent_calibration",
            "next_coordinate_prediction",
        ],
        "authorization": {
            "run_behavior_qualification": True,
            "run_internal_collection_before_behavior_freeze": False,
            "open_physical_holdout": False,
            "run_causal_intervention": False,
            "run_single_neuron_scan": False,
        },
        "claim_boundary": {
            "phase384_proves_raw_neuron_activity_cancels": False,
            "phase384_proves_contrastive_terminal_projection_cancels": True,
            "semantic_coordinates_are_five_independent_times": False,
            "relation_similarity_is_causality": False,
            "language_encoding_mechanism_closed": False,
        },
    }
    summary = {
        "schema_version": "60.0.0",
        "phase_id": "Phase386-CaseBank",
        "created_at": created_at,
        "candidate_case_count": len(execution_rows),
        "candidate_parallel_group_count": len(MECHANISMS)
        * CANDIDATE_GROUPS_PER_MECHANISM,
        "unique_rendered_prompt_count": len(prompt_hashes),
        "prior_prompt_overlap_count": prior_overlap_count,
        "internal_collection_started": False,
        "physical_holdout_opened": False,
    }
    write_json(OUT / "phase386_protocol.json", protocol)
    write_json(OUT / "phase386_case_bank_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
