#!/usr/bin/env python3
"""Generate the frozen fresh Phase371C four-mechanism parallel case bank."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from model_registry import get_model_spec  # noqa: E402
from phase333_dynamic_case_bank import interface_prompt  # noqa: E402


PHASE = "Phase371C"
SCHEMA = "47.7.0"
OUT = ROOT / "tests/gpt5/result/phase371_exact_vector_coactivity/phase371c_case_bank"
MODELS = ("qwen3", "glm4", "deepseek7b")
MECHANISMS = (
    ("content_knowledge", "relation_binding"),
    ("readout_competition", "target_competition"),
    ("state_drift", "entity_recency"),
    ("syntax_structure", "number_agreement"),
)
CONDITIONS = ("A_target_lex_x", "B_control_lex_x", "C_target_lex_y", "D_control_lex_y")
TEMPLATES = (
    "Archive note: {context}\nArchive query: {question}\nReply rule: {instruction}\nReply:",
    "Verified statements: {context}\nRequested field: {question}\nFormat: {instruction}\nValue:",
    "Consult this record only. {context}\nQuestion: {question}\n{instruction}\nAnswer:",
    "Record contents: {context}\nTask: {question}\nConstraint: {instruction}\nResult:",
    "Source card: {context}\nLookup request: {question}\nOutput requirement: {instruction}\nOutput:",
)
OBJECTS = (
    "anchor", "basket", "camera", "drum", "envelope", "folder", "helmet", "jacket",
    "kettle", "mirror", "notebook", "pillow", "quilt", "radio", "spoon", "ticket",
    "umbrella", "violin", "wallet", "xylophone", "yarn", "zipper", "badge", "candle",
    "drawer", "engine", "frame", "glove", "hammer", "island", "ladder", "magnet",
    "needle", "orange", "pencil", "ribbon", "saddle", "towel", "vase", "whistle",
    "album", "bucket", "curtain", "dish", "fork", "guitar", "hose", "inkwell",
)
MATERIALS = (
    "bronze", "cotton", "marble", "wool", "aluminum", "clay", "velvet", "quartz",
    "nickel", "silk", "wood", "stone", "cardboard", "iron", "porcelain", "tin",
    "canvas", "foam", "cork", "lead", "platinum", "nylon", "concrete", "wax",
)
LABELS = (
    "crimson", "cyan", "magenta", "ochre", "pearl", "saffron", "jade", "navy",
    "plum", "rose", "mint", "charcoal", "lime", "cobalt", "cream", "khaki",
    "lavender", "peach", "ruby", "turquoise", "bronze", "white", "black", "gray",
)
NAMES = (
    "Adela", "Bennett", "Celia", "Devon", "Elise", "Gideon", "Helena", "Isaac",
    "Jasmine", "Keaton", "Lucia", "Marcus", "Noelle", "Orson", "Priya", "Quentin",
    "Rhea", "Simon", "Theresa", "Uriah", "Vera", "Wesley", "Yvette", "Zane",
)
NOUNS = (
    ("actor", "actors"), ("bell", "bells"), ("chef", "chefs"), ("desk", "desks"),
    ("editor", "editors"), ("farmer", "farmers"), ("garden", "gardens"),
    ("host", "hosts"), ("island", "islands"), ("judge", "judges"),
    ("kitten", "kittens"), ("letter", "letters"), ("museum", "museums"),
    ("nurse", "nurses"), ("ocean", "oceans"), ("pilot", "pilots"),
    ("queen", "queens"), ("river", "rivers"), ("singer", "singers"),
    ("truck", "trucks"), ("village", "villages"), ("writer", "writers"),
    ("artist", "artists"), ("bridge", "bridges"), ("captain", "captains"),
    ("dancer", "dancers"), ("engineer", "engineers"), ("flag", "flags"),
    ("guest", "guests"), ("hotel", "hotels"), ("journal", "journals"),
    ("kitchen", "kitchens"), ("lake", "lakes"), ("manager", "managers"),
    ("neighbor", "neighbors"), ("office", "offices"), ("poet", "poets"),
    ("road", "roads"), ("station", "stations"), ("tower", "towers"),
    ("visitor", "visitors"), ("worker", "workers"), ("author", "authors"),
    ("bank", "banks"), ("driver", "drivers"), ("field", "fields"),
    ("guard", "guards"), ("hall", "halls"),
)
PREPOSITIONS = ("beside", "near", "behind", "in front of", "next to", "across from")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def split_for(group_index: int) -> str:
    if group_index < 12:
        return "fresh_discovery"
    if group_index < 18:
        return "sealed_calibration"
    return "sealed_physical"


def task(mechanism: str, group_index: int, lexical_slot: str, demanded: bool) -> dict[str, Any]:
    offset = 0 if lexical_slot == "x" else 17
    index = (group_index + offset) % 24
    code = f"p371c-{mechanism[:3]}-{group_index:02d}-{lexical_slot}"
    exact = "Return only the requested answer without explanation."
    if mechanism == "relation_binding":
        obj = f"{OBJECTS[(group_index + offset) % len(OBJECTS)]}-{code}"
        depot = f"depot-{code}"
        target = MATERIALS[index]
        wrong = MATERIALS[(index + 7) % len(MATERIALS)]
        if demanded:
            context = f"The {obj} is stored in {depot}. Every item stored in {depot} is made of {target}."
        else:
            context = f"The {obj} is made of {target}. Its storage location is {depot}."
        question = f"What is the {obj} made of?"
    elif mechanism == "target_competition":
        target = LABELS[index]
        wrong = LABELS[(index + 9) % len(LABELS)]
        if demanded:
            context = f"Entry {code} marks {target} as accepted and {wrong} as rejected."
        else:
            context = f"Entry {code} marks {target} as accepted. No rejected label is listed."
        question = f"Which label is accepted for entry {code}?"
    elif mechanism == "entity_recency":
        target = NAMES[index]
        wrong = NAMES[(index + 11) % len(NAMES)]
        if demanded:
            context = f"The supervisor of record {code} is {target}. A later weather note mentions {wrong}."
        else:
            context = f"A weather note mentions {wrong}. The supervisor of record {code} is {target}."
        question = f"Who is the supervisor of record {code}?"
    elif mechanism == "number_agreement":
        noun_index = (group_index + offset) % len(NOUNS)
        plural = (group_index + offset) % 2 == 0
        singular, plural_form = NOUNS[noun_index]
        head = plural_form if plural else singular
        target, wrong = ("are", "is") if plural else ("is", "are")
        attractor_singular, attractor_plural = NOUNS[(noun_index + 13) % len(NOUNS)]
        attractor = attractor_singular if plural else attractor_plural
        subject = f"the {head} {PREPOSITIONS[group_index % len(PREPOSITIONS)]} the {attractor}" if demanded else f"the {head}"
        context = f"Identifier {code} is metadata. Apply agreement using the head noun '{head}'."
        question = f"Fill the blank: {subject.capitalize()} ___ prepared."
        exact = "Answer with exactly is or are."
    else:
        raise KeyError(mechanism)
    return {
        "context": context,
        "question": question,
        "instruction": exact,
        "target": target,
        "target_aliases": [target],
        "distractors": [wrong, "unknown"],
        "language": "en",
    }


def prior_prompt_hashes() -> set[str]:
    hashes = set()
    result_root = ROOT / "tests/gpt5/result"
    for path in result_root.rglob("*.jsonl"):
        if OUT in path.parents:
            continue
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip() or '"prompt"' not in line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            for key in ("prompt", "raw_prompt"):
                value = row.get(key)
                if isinstance(value, str) and value:
                    hashes.add(digest(value))
    return hashes


def main() -> None:
    prior_hashes = prior_prompt_hashes()
    execution_rows = []
    blind_rows = []
    label_rows = []
    prompt_hashes = set()
    raw_prompt_hashes = set()
    for model in MODELS:
        spec = get_model_spec(model)
        tokenizer = AutoTokenizer.from_pretrained(
            str(spec.local_dir), trust_remote_code=spec.trust_remote_code,
            local_files_only=True, use_fast=False,
        )
        for family, mechanism in MECHANISMS:
            for group_index in range(22):
                split = split_for(group_index)
                semantic_group = f"phase371c_{family}_{mechanism}_{group_index:02d}"
                parallel_group = "parallel371c_" + digest(semantic_group)[:18]
                model_group = "group371c_" + digest(f"{model}:{semantic_group}")[:18]
                items = {}
                for lexical_slot, demanded_letter, control_letter in (("x", "A", "B"), ("y", "C", "D")):
                    items[demanded_letter] = task(mechanism, group_index, lexical_slot, True)
                    items[control_letter] = task(mechanism, group_index, lexical_slot, False)
                for condition in CONDITIONS:
                    letter = condition[0]
                    item = items[letter]
                    raw_prompt = TEMPLATES[group_index % len(TEMPLATES)].format(
                        context=item["context"], question=item["question"], instruction=item["instruction"],
                    )
                    prompt, add_special, answer_phase = interface_prompt(tokenizer, model, raw_prompt, "answer_aligned_chat")
                    raw_hash = digest(raw_prompt)
                    prompt_hash = digest(prompt)
                    if raw_hash in prior_hashes or prompt_hash in prior_hashes:
                        raise RuntimeError(f"Prior prompt overlap: {model}/{semantic_group}/{condition}")
                    if prompt_hash in prompt_hashes:
                        raise RuntimeError(f"Duplicate rendered prompt: {model}/{semantic_group}/{condition}")
                    prompt_hashes.add(prompt_hash)
                    raw_prompt_hashes.add(raw_hash)
                    blind_case_id = "p371c_" + digest(f"{model}:{semantic_group}:{condition}")[:23]
                    common = {
                        "schema_version": SCHEMA,
                        "phase_id": PHASE,
                        "created_at": now(),
                        "blind_case_id": blind_case_id,
                        "anonymous_model_id": "am371c_" + digest(model)[:11],
                        "anonymous_parallel_group_id": parallel_group,
                        "anonymous_group_id": model_group,
                        "anonymous_condition_slot": "slot371c_" + digest(f"{model_group}:{condition}")[:10],
                        "phase371c_split": split,
                        "prompt": prompt,
                        "raw_prompt": raw_prompt,
                        "source_fragment": item["context"],
                        "query_fragment": item["question"],
                        "tokenization_add_special_tokens": add_special,
                        "prompt_token_count": len(tokenizer(prompt, add_special_tokens=False)["input_ids"]),
                        "interface": "answer_aligned_chat",
                        "answer_phase": answer_phase,
                    }
                    execution_rows.append({
                        **common,
                        "private_execution_model": model,
                        "family_id": family,
                        "mechanism_id": mechanism,
                        "semantic_group_id": semantic_group,
                        "contrast_condition": condition,
                        "operation_demanded": letter in {"A", "C"},
                        "target": item["target"],
                        "target_aliases": item["target_aliases"],
                        "distractors": item["distractors"],
                        "language": item["language"],
                        "instruction": item["instruction"],
                        "semantic_labels_available_to_collector": False,
                        "target_specific_competition_available_to_collector": False,
                    })
                    blind_rows.append({
                        **common,
                        "semantic_label_used_for_selection": False,
                        "target_or_distractor_exported": False,
                    })
                    label_rows.append({
                        "blind_case_id": blind_case_id,
                        "model": model,
                        "family_id": family,
                        "mechanism_id": mechanism,
                        "semantic_group_id": semantic_group,
                        "contrast_condition": condition,
                        "phase371c_split": split,
                        "target": item["target"],
                        "target_aliases": item["target_aliases"],
                        "distractors": item["distractors"],
                    })
    counts = Counter(row["phase371c_split"] for row in execution_rows)
    expected = {"fresh_discovery": 576, "sealed_calibration": 288, "sealed_physical": 192}
    if len(execution_rows) != 1056 or dict(counts) != expected:
        raise RuntimeError(f"Invalid Phase371C denominator: total={len(execution_rows)} splits={dict(counts)}")
    by_group = Counter((row["anonymous_model_id"], row["anonymous_group_id"]) for row in execution_rows)
    if set(by_group.values()) != {4}:
        raise RuntimeError("Every Phase371C model group must have four conditions")
    summary = {
        "schema_version": SCHEMA,
        "phase_id": PHASE,
        "created_at": now(),
        "objective": "freeze_fresh_exact_vector_cycle_denominator_before_model_execution",
        "denominator": {
            "model_count": 3,
            "mechanism_count": 4,
            "parallel_group_count": 88,
            "model_group_count": 264,
            "condition_count_per_model_group": 4,
            "case_count": len(execution_rows),
            "fresh_discovery_case_count": counts["fresh_discovery"],
            "sealed_calibration_case_count": counts["sealed_calibration"],
            "sealed_physical_case_count": counts["sealed_physical"],
        },
        "quality": {
            "unique_rendered_prompt_count": len(prompt_hashes),
            "unique_raw_prompt_count": len(raw_prompt_hashes),
            "prior_prompt_overlap_count": 0,
            "every_model_group_has_four_conditions": True,
            "parallel_group_ids_shared_across_models": True,
            "semantic_labels_exported_to_blind_registry": False,
            "physical_holdout_opened": False,
        },
        "authorization": {
            "static_contract_audit": True,
            "behavior_model_execution": False,
            "internal_collection": False,
        },
        "next_decision": "audit_case_contract_and_freeze_execution_hashes",
    }
    write_json(OUT / "phase371c_case_bank_summary.json", summary)
    write_jsonl(OUT / "phase371c_blind_case_registry.jsonl", blind_rows)
    write_jsonl(OUT / "phase371c_nonphysical_blind_cases.jsonl", [
        row for row in blind_rows if row["phase371c_split"] != "sealed_physical"
    ])
    write_jsonl(OUT / "sealed/phase371c_physical_blind_cases.jsonl", [
        row for row in blind_rows if row["phase371c_split"] == "sealed_physical"
    ])
    write_jsonl(OUT / "private/phase371c_execution_cases.jsonl", execution_rows)
    write_jsonl(OUT / "private/phase371c_label_key.jsonl", label_rows)
    write_jsonl(OUT / "private/phase371c_nonphysical_execution_cases.jsonl", [
        row for row in execution_rows if row["phase371c_split"] != "sealed_physical"
    ])
    write_jsonl(OUT / "sealed/private/phase371c_physical_execution_cases.jsonl", [
        row for row in execution_rows if row["phase371c_split"] == "sealed_physical"
    ])
    write_jsonl(OUT / "private/phase371c_nonphysical_label_key.jsonl", [
        row for row in label_rows if row["phase371c_split"] != "sealed_physical"
    ])
    write_jsonl(OUT / "sealed/private/phase371c_physical_label_key.jsonl", [
        row for row in label_rows if row["phase371c_split"] == "sealed_physical"
    ])
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
