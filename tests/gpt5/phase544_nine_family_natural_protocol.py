#!/usr/bin/env python3
"""Freeze the Phase544 nine-family natural behavior qualification protocol.

The protocol deliberately separates independent semantic world pairs from
surface rewrites. Repeating one item under two surfaces does not increase the
independent sample denominator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from model_registry import get_model_spec  # noqa: E402
from phase330_nine_family_case_bank import FAMILY_MECHANISMS  # noqa: E402


PHASE = "Phase544"
SCHEMA_VERSION = "phase544_nine_family_natural_behavior.v1"
MODELS = ("qwen3", "glm4", "deepseek7b")
SPLITS = ("discovery", "independent_confirmation")
SURFACES = ("direct_natural", "reordered_natural")
CONDITIONS = ("world_a", "world_b")
PAIR_UNITS_PER_SPLIT = 73
OUT_DIR = ROOT / "tests/gpt5/result/phase544_nine_family_natural_behavior"
REGISTERED_CASES = OUT_DIR / "phase544_registered_cases.jsonl"
PROTOCOL_PATH = OUT_DIR / "phase544_frozen_protocol.json"
STATIC_AUDIT_PATH = OUT_DIR / "phase544_static_audit.json"
REPRESENTATIVE_PATH = OUT_DIR / "phase544_representative_registry.json"
Z = 1.96


REPRESENTATIVES: dict[str, tuple[str, str]] = {
    "content_knowledge": ("category", "negated_attribute"),
    "output_protocol": ("answer_only", "json"),
    "reasoning_constraint": ("direct_entailment", "two_hop_entailment"),
    "syntax_structure": ("subject_role", "singular_agreement"),
    "language_action": ("extract", "transform"),
    "cross_lingual": ("translation", "role_binding"),
    "readout_competition": ("target_vs_wrong", "multi_token_answer"),
    "state_drift": ("entity_drift", "role_drift"),
    "closure": ("semantic_completion", "multi_token_completion"),
}


CATEGORIES = ("bird", "tool", "plant", "vehicle", "mammal", "instrument", "fruit", "mineral")
ATTRIBUTES = ("amber", "violet", "silver", "crimson", "teal", "ivory", "copper", "indigo")
VERBS = ("carried", "opened", "moved", "painted", "washed", "lifted", "placed", "found")
NOUNS = ("parcel", "lantern", "tablet", "basket", "folder", "vessel", "marker", "ticket")
ADJECTIVES = ("quiet", "bright", "narrow", "gentle", "steady", "clear", "rapid", "calm")
SUFFIXES = ("harbor", "meadow", "bridge", "garden", "station", "valley", "window", "circle")
SYLLABLE_A = ("ba", "ce", "di", "fo", "ga", "hu", "ji", "ke", "lu", "mi", "no", "pa", "ri")
SYLLABLE_B = ("lan", "mer", "tin", "vor", "sen", "dak", "pel", "rin", "sol", "wen", "yas", "kor", "zen")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denominator = 1 + Z * Z / n
    center = (p + Z * Z / (2 * n)) / denominator
    radius = Z * math.sqrt((p * (1 - p) + Z * Z / (4 * n)) / n) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def minimum_n_for_perfect_lower(threshold: float) -> int:
    n = 1
    while wilson(n, n)[0] < threshold:
        n += 1
    return n


def minimum_n_for_zero_upper(threshold: float) -> int:
    n = 1
    while wilson(0, n)[1] > threshold:
        n += 1
    return n


def pseudo_word(index: int, *, confirmation: bool = False) -> str:
    shifted = index + (79 if confirmation else 0)
    a = SYLLABLE_A[shifted % len(SYLLABLE_A)]
    b = SYLLABLE_B[(shifted // len(SYLLABLE_A)) % len(SYLLABLE_B)]
    c = SYLLABLE_A[(shifted * 5 + 3) % len(SYLLABLE_A)]
    return (a + b + c).capitalize()


def english_number(value: int) -> str:
    ones = (
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
        "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
        "seventeen", "eighteen", "nineteen",
    )
    tens = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")
    if value < 20:
        return ones[value]
    if value < 100:
        return tens[value // 10] if value % 10 == 0 else f"{tens[value // 10]}-{ones[value % 10]}"
    if value < 200:
        return "one hundred" if value == 100 else f"one hundred {english_number(value - 100)}"
    raise ValueError(value)


def chinese_number(value: int) -> str:
    digits = "零一二三四五六七八九"
    if value < 10:
        return digits[value]
    if value < 20:
        return "十" if value == 10 else "十" + digits[value % 10]
    if value < 100:
        return digits[value // 10] + "十" + (digits[value % 10] if value % 10 else "")
    if value < 200:
        tail = value - 100
        if tail == 0:
            return "一百"
        return "一百" + ("零" if tail < 10 else "") + chinese_number(tail)
    raise ValueError(value)


def task(
    family: str,
    mechanism: str,
    split: str,
    pair_index: int,
    condition: str,
) -> dict[str, Any]:
    confirmation = split == "independent_confirmation"
    base = pair_index + (73 if confirmation else 0)
    flip = condition == "world_b"
    entity = pseudo_word(base * 2, confirmation=confirmation)
    other = pseudo_word(base * 2 + 1, confirmation=confirmation)
    category = CATEGORIES[(base + (3 if flip else 0)) % len(CATEGORIES)]
    alternate_category = CATEGORIES[(base + (0 if flip else 3)) % len(CATEGORIES)]
    attribute = ATTRIBUTES[(base + (5 if flip else 1)) % len(ATTRIBUTES)]
    wrong_attribute = ATTRIBUTES[(base + (1 if flip else 5)) % len(ATTRIBUTES)]
    number_a = 11 + (base % 71)
    number_b = 3 + ((base * 7) % 19) + (1 if flip else 0)
    total = number_a + number_b
    verb = VERBS[base % len(VERBS)]
    noun = NOUNS[(base * 3) % len(NOUNS)]
    strict_kind = "plain"
    operation = mechanism
    target_source = "derived"

    if family == "content_knowledge" and mechanism == "category":
        context = f"In the stated local taxonomy, the {entity} is classified as a {category}."
        question = f"What category is assigned to the {entity}?"
        target, distractors = category, [alternate_category, "unknown"]
        target_source = "explicit_source_fact"
    elif family == "content_knowledge" and mechanism == "negated_attribute":
        context = f"The {entity} is not {wrong_attribute}; its stated attribute is {attribute}."
        question = f"Which stated attribute applies to the {entity}?"
        target, distractors = attribute, [wrong_attribute, "unknown"]
        target_source = "explicit_negated_source_fact"
    elif family == "output_protocol" and mechanism in {"answer_only", "json"}:
        context = f"Calculate {number_a} plus {number_b}."
        question = "What is the result?"
        target, distractors = str(total), [str(total - 1), str(total + 1)]
        if mechanism == "json":
            strict_kind = "json_answer"
    elif family == "reasoning_constraint" and mechanism == "direct_entailment":
        prop = ADJECTIVES[base % len(ADJECTIVES)]
        if flip:
            context = f"No member of group {entity} is {prop}. {other} belongs to group {entity}."
            target, distractors = "no", ["yes", "unknown"]
        else:
            context = f"Every member of group {entity} is {prop}. {other} belongs to group {entity}."
            target, distractors = "yes", ["no", "unknown"]
        question = f"Is {other} {prop}?"
    elif family == "reasoning_constraint" and mechanism == "two_hop_entailment":
        prop1 = ADJECTIVES[base % len(ADJECTIVES)]
        prop2 = ATTRIBUTES[(base + 2) % len(ATTRIBUTES)]
        second = (
            f"No {prop1} thing is {prop2}." if flip else f"Every {prop1} thing is {prop2}."
        )
        context = f"Every member of group {entity} is {prop1}. {second} {other} belongs to group {entity}."
        question = f"Is {other} {prop2}?"
        target, distractors = ("no", ["yes", "unknown"]) if flip else ("yes", ["no", "unknown"])
    elif family == "syntax_structure" and mechanism == "subject_role":
        subject, obj = (other, entity) if flip else (entity, other)
        context = f"{subject} {verb} the {noun} beside {obj}."
        question = "Who performed the action?"
        target, distractors = subject, [obj, noun]
        target_source = "source_role"
    elif family == "syntax_structure" and mechanism == "singular_agreement":
        plural = flip
        subject = f"the {noun}s" if plural else f"the {noun}"
        context = f"The grammatical subject is '{subject}'."
        question = "Complete with the agreeing verb: The subject ___ daily."
        target, distractors = ("run", ["runs", "running"]) if plural else ("runs", ["run", "running"])
    elif family == "language_action" and mechanism == "extract":
        code = f"{('Q' if flip else 'R')}{2000 + base * 2 + int(flip)}"
        context = f"Record: owner={entity}; code={code}; status=active."
        question = "Extract the code value."
        target, distractors = code, [entity, "active"]
        target_source = "explicit_field"
    elif family == "language_action" and mechanism == "transform":
        source = pseudo_word(base * 3 + int(flip), confirmation=confirmation).lower()
        context = f"The source word is {source}."
        question = "Convert the source word to uppercase."
        target, distractors = source.upper(), [source, source.capitalize()]
        target_source = "deterministic_transform"
    elif family == "cross_lingual" and mechanism == "translation":
        value = 1 + ((base * 2 + int(flip)) % 149)
        source = english_number(value)
        context = f"The English number expression is '{source}'."
        question = "Translate that number expression into Chinese."
        target = chinese_number(value)
        distractors = [chinese_number(1 + (value % 149)), source]
        target_source = "pretrained_translation"
    elif family == "cross_lingual" and mechanism == "role_binding":
        holder, non_holder = (other, entity) if flip else (entity, other)
        object_zh = ("书", "杯子", "钥匙", "地图", "灯", "球", "票", "盒子")[base % 8]
        context = f"{holder}拿着{object_zh}，而{non_holder}站在旁边。"
        question = "Who is holding the object?"
        target, distractors = holder, [non_holder, object_zh]
        target_source = "mixed_language_source_role"
    elif family == "readout_competition" and mechanism == "target_vs_wrong":
        context = f"Compute {number_a} plus {number_b}; ignore nearby wrong guesses."
        question = "Return the correct result."
        target, distractors = str(total), [str(total - 1), str(total + 1)]
    elif family == "readout_competition" and mechanism == "multi_token_answer":
        multiplier = 2 + (base % 5)
        units = 3 + ((base + int(flip)) % 13)
        answer = multiplier * units
        context = f"A route has {units} segments of {multiplier} meters each."
        question = "Give the total distance including the unit."
        target, distractors = f"{answer} meters", [str(answer), f"{answer + multiplier} meters"]
        strict_kind = "multi_token"
    elif family == "state_drift" and mechanism == "entity_drift":
        owner, later = (other, entity) if flip else (entity, other)
        context = f"The original owner is {owner}. A later unrelated note mentions {later}."
        question = "Who is the original owner?"
        target, distractors = owner, [later, "unknown"]
        target_source = "persistent_source_entity"
    elif family == "state_drift" and mechanism == "role_drift":
        subject, obj = (other, entity) if flip else (entity, other)
        context = f"{subject} {verb} the {noun} to {obj}. A later note lists {obj} first."
        question = "Who performed the action?"
        target, distractors = subject, [obj, noun]
        target_source = "persistent_source_role"
    elif family == "closure" and mechanism in {"semantic_completion", "multi_token_completion"}:
        adjective = ADJECTIVES[(base + int(flip)) % len(ADJECTIVES)]
        suffix = SUFFIXES[(base * 3 + int(flip)) % len(SUFFIXES)]
        phrase = f"{adjective} {suffix}"
        context = f"The complete registered phrase is '{phrase}'."
        question = "Return the complete phrase and stop."
        target, distractors = phrase, [adjective, suffix]
        strict_kind = "multi_token"
        target_source = "explicit_completion_source"
    else:
        raise KeyError((family, mechanism))

    instruction = (
        'Return one JSON object with exactly the key "answer" and a string value; add no other text.'
        if strict_kind == "json_answer"
        else "Return exactly the requested natural answer and no explanation."
    )
    strict_expected = (
        json.dumps({"answer": target}, ensure_ascii=False, separators=(",", ":"))
        if strict_kind == "json_answer"
        else target
    )
    return {
        "context": context,
        "question": question,
        "instruction": instruction,
        "target": target,
        "distractors": list(dict.fromkeys(value for value in distractors if value != target)),
        "strict_expected": strict_expected,
        "strict_kind": strict_kind,
        "operation": operation,
        "target_source_type": target_source,
        "lexical_key": f"{split}:{pair_index:03d}:{entity}:{other}",
    }


def render_surface(task_row: dict[str, Any], surface: str) -> str:
    if surface == "direct_natural":
        return (
            f"Context: {task_row['context']}\n"
            f"Question: {task_row['question']}\n"
            f"Instruction: {task_row['instruction']}"
        )
    if surface == "reordered_natural":
        return (
            f"Task: {task_row['question']}\n"
            f"Use this information: {task_row['context']}\n"
            f"Output requirement: {task_row['instruction']}"
        )
    raise KeyError(surface)


def tokenizer_for(model: str) -> Any:
    spec = get_model_spec(model)
    tokenizer = AutoTokenizer.from_pretrained(
        str(spec.local_dir), trust_remote_code=spec.trust_remote_code,
        local_files_only=True, use_fast=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def render_chat(tokenizer: Any, model: str, content: str) -> str:
    kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
    if model == "qwen3":
        kwargs["enable_thinking"] = False
    rendered = tokenizer.apply_chat_template([{"role": "user", "content": content}], **kwargs)
    if model == "deepseek7b" and rendered.endswith("<think>\n"):
        rendered += "</think>\n\n"
    return rendered


def build_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    tokenizers = {model: tokenizer_for(model) for model in MODELS}
    created_at = now()
    for model in MODELS:
        tokenizer = tokenizers[model]
        for family, mechanisms in REPRESENTATIVES.items():
            for mechanism in mechanisms:
                for split in SPLITS:
                    for pair_index in range(PAIR_UNITS_PER_SPLIT):
                        vocab_set = "vocab_a" if pair_index < 37 else "vocab_b"
                        unit_id = f"phase544_{family}_{mechanism}_{split}_{pair_index:03d}"
                        for condition in CONDITIONS:
                            task_row = task(family, mechanism, split, pair_index, condition)
                            for surface in SURFACES:
                                raw_prompt = render_surface(task_row, surface)
                                prompt = render_chat(tokenizer, model, raw_prompt)
                                rows.append({
                                    "schema_version": SCHEMA_VERSION,
                                    "phase_id": PHASE,
                                    "created_at": created_at,
                                    "case_id": f"{unit_id}_{model}_{condition}_{surface}",
                                    "semantic_unit_id": unit_id,
                                    "surface_pair_id": f"{unit_id}_{model}_{surface}",
                                    "model": model,
                                    "family_id": family,
                                    "mechanism_id": mechanism,
                                    "split": split,
                                    "pair_index": pair_index,
                                    "vocab_set": vocab_set,
                                    "world_condition": condition,
                                    "surface": surface,
                                    "contrast_kind": "matched_counterfactual_world_pair",
                                    "raw_prompt": raw_prompt,
                                    "prompt": prompt,
                                    "target": task_row["target"],
                                    "target_aliases": [task_row["target"]],
                                    "distractors": task_row["distractors"],
                                    "strict_expected": task_row["strict_expected"],
                                    "strict_kind": task_row["strict_kind"],
                                    "operation": task_row["operation"],
                                    "target_source_type": task_row["target_source_type"],
                                    "lexical_key": task_row["lexical_key"],
                                    "semantic_event_is_natural_answer": True,
                                    "arbitrary_label_output": False,
                                    "internal_intervention_allowed": False,
                                    "physical_collection_allowed": False,
                                    "single_neuron_scan_allowed": False,
                                    "sealed": False,
                                })
    return rows


def old_denominator_audit() -> dict[str, Any]:
    phase330_root = ROOT / "tests/gpt5/result/phase330_nine_family_global_atlas/nine_family_global_atlas"
    validation = json.loads((phase330_root / "phase330_case_bank_validation.json").read_text(encoding="utf-8"))
    old_rows = [
        json.loads(line)
        for line in (phase330_root / "phase330_case_bank.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    mechanism_signatures: dict[tuple[str, str], list[tuple[Any, ...]]] = defaultdict(list)
    for row in old_rows:
        if row["template_id"] != "template_a":
            continue
        mechanism_signatures[(row["family_id"], row["mechanism_id"])].append((
            row["item_index"], row["context"], row["question"], row["target"], tuple(row["distractors"])
        ))
    signature_groups: dict[str, list[str]] = defaultdict(list)
    for (family, mechanism), signature in mechanism_signatures.items():
        signature_groups[stable_hash(sorted(signature))].append(f"{family}/{mechanism}")
    duplicate_groups = [sorted(group) for group in signature_groups.values() if len(group) > 1]
    phase350_protocol = json.loads((
        ROOT / "tests/gpt5/result/phase350_nine_family_minimal_contrast/"
        "nine_family_minimal_contrast_qualification/phase350_registered_protocol.json"
    ).read_text(encoding="utf-8"))
    phase424 = json.loads((
        ROOT / "tests/gpt5/result/phase424_global_physical_path_atlas/phase424_protocol.json"
    ).read_text(encoding="utf-8"))
    phase543 = json.loads((
        ROOT / "tests/gpt5/result/phase543_seal_contamination_audit/phase543_seal_contamination_audit.json"
    ).read_text(encoding="utf-8"))
    return {
        "phase330_family_count": validation["family_count"],
        "phase330_mechanism_count": validation["mechanism_count"],
        "phase330_prompt_case_count": validation["prompt_case_count"],
        "phase330_target_leak_count": validation["target_leak_count"],
        "phase330_target_leak_rate": validation["target_leak_count"] / validation["prompt_case_count"],
        "phase330_heldout_independent_items_per_mechanism": 6,
        "phase330_counterfactual_group_field_present": False,
        "phase330_identical_semantic_contract_group_count": len(duplicate_groups),
        "phase330_identical_semantic_contract_groups": duplicate_groups,
        "phase350_representative_mechanisms_per_family": 1,
        "phase350_control_is_explicit_shortcut_or_relaxed_protocol": any(
            "shortcut" in boundary for boundary in phase350_protocol["claim_boundaries"]
        ),
        "phase424_strict_double_blind_holdout": phase424["split_status"]["strict_double_blind_holdout"],
        "phase424_source_previously_exposed": True,
        "phase543_historical_phase535_sealed_read": phase543["evidence_boundary"]["historical_phase535_sealed_read"],
        "old_denominator_direct_reuse_authorized": False,
        "direct_reuse_blockers": [
            "old heldout has only 6 independent items per mechanism",
            "Phase330 has no typed counterfactual group field",
            "some mechanism labels share identical semantic contracts",
            "Phase350 controls may expose shortcuts or relax the protocol",
            "Phase424 has no strict double-blind physical holdout",
            "Phase535 historical seal is contaminated",
        ],
    }


def validate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    expected = (
        len(MODELS) * 18 * len(SPLITS) * PAIR_UNITS_PER_SPLIT
        * len(CONDITIONS) * len(SURFACES)
    )
    semantic_units = {row["semantic_unit_id"] for row in rows}
    pair_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    unit_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        pair_groups[row["surface_pair_id"]].append(row)
        unit_groups[(row["model"], row["semantic_unit_id"])].append(row)
    target_flip_errors = sum(
        len({row["target"] for row in group}) != 2 for group in pair_groups.values()
    )
    discovery_lexical = {row["lexical_key"] for row in rows if row["split"] == "discovery"}
    confirmation_lexical = {
        row["lexical_key"] for row in rows if row["split"] == "independent_confirmation"
    }
    perfect_semantic_lcb, _ = wilson(PAIR_UNITS_PER_SPLIT, PAIR_UNITS_PER_SPLIT)
    _, zero_unrecoverable_ucb = wilson(0, PAIR_UNITS_PER_SPLIT)
    perfect_pair_lcb, _ = wilson(PAIR_UNITS_PER_SPLIT * len(SURFACES), PAIR_UNITS_PER_SPLIT * len(SURFACES))
    result = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE,
        "created_at": now(),
        "registered_case_count": len(rows),
        "expected_case_count": expected,
        "model_case_counts": dict(Counter(row["model"] for row in rows)),
        "family_count": len({row["family_id"] for row in rows}),
        "representative_mechanism_count": len({(row["family_id"], row["mechanism_id"]) for row in rows}),
        "semantic_unit_count": len(semantic_units),
        "semantic_units_per_mechanism_split": sorted(set(Counter(
            (row["family_id"], row["mechanism_id"], row["split"])
            for row in rows if row["model"] == MODELS[0] and row["surface"] == SURFACES[0]
            and row["world_condition"] == CONDITIONS[0]
        ).values())),
        "rows_per_model_semantic_unit": sorted(set(len(group) for group in unit_groups.values())),
        "surface_pair_size_values": sorted(set(len(group) for group in pair_groups.values())),
        "target_flip_error_count": target_flip_errors,
        "duplicate_case_id_count": len(rows) - len({row["case_id"] for row in rows}),
        "empty_target_count": sum(not row["target"] for row in rows),
        "empty_distractor_count": sum(not row["distractors"] for row in rows),
        "arbitrary_label_output_count": sum(bool(row["arbitrary_label_output"]) for row in rows),
        "sealed_row_count": sum(bool(row["sealed"]) for row in rows),
        "discovery_confirmation_lexical_overlap_count": len(discovery_lexical & confirmation_lexical),
        "minimum_independent_n": {
            "perfect_semantic_lcb95_ge_0_90": minimum_n_for_perfect_lower(0.90),
            "perfect_pair_lcb95_ge_0_80": minimum_n_for_perfect_lower(0.80),
            "zero_unrecoverable_ucb95_le_0_05": minimum_n_for_zero_upper(0.05),
            "frozen_pair_units_per_split": PAIR_UNITS_PER_SPLIT,
            "perfect_semantic_lcb95_at_frozen_n": perfect_semantic_lcb,
            "perfect_pair_lcb95_at_frozen_n": perfect_pair_lcb,
            "zero_unrecoverable_ucb95_at_frozen_n": zero_unrecoverable_ucb,
        },
        "old_denominator_audit": old_denominator_audit(),
    }
    result["valid"] = (
        result["registered_case_count"] == expected
        and set(result["model_case_counts"].values()) == {expected // len(MODELS)}
        and result["family_count"] == 9
        and result["representative_mechanism_count"] == 18
        and result["semantic_units_per_mechanism_split"] == [73]
        and result["rows_per_model_semantic_unit"] == [4]
        and result["surface_pair_size_values"] == [2]
        and all(result[key] == 0 for key in (
            "target_flip_error_count", "duplicate_case_id_count", "empty_target_count",
            "empty_distractor_count", "arbitrary_label_output_count", "sealed_row_count",
            "discovery_confirmation_lexical_overlap_count",
        ))
        and perfect_semantic_lcb >= 0.90
        and perfect_pair_lcb >= 0.80
        and zero_unrecoverable_ucb <= 0.05
    )
    result["status"] = "static_pass_no_model_run" if result["valid"] else "static_fail"
    return result


def protocol(rows_path: Path, audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE,
        "created_at": now(),
        "title": "Nine-family natural behavior qualification and physical-entry reselection",
        "models_in_required_execution_order": list(MODELS),
        "frozen_denominator": {"families": 9, "registered_mechanisms": 72},
        "screening_denominator": {
            "families": 9,
            "representative_mechanisms": 18,
            "representatives": {family: list(mechanisms) for family, mechanisms in REPRESENTATIVES.items()},
            "independent_world_pairs_per_mechanism_split": PAIR_UNITS_PER_SPLIT,
            "surfaces_per_world_condition": len(SURFACES),
            "world_conditions_per_pair": len(CONDITIONS),
            "prompts_per_model": len(rows_path.read_text(encoding="utf-8").splitlines()) // len(MODELS),
        },
        "behavior_gates": {
            "semantic_unit_exact_lcb95_min": 0.90,
            "surface_pair_exact_lcb95_min": 0.80,
            "unrecoverable_unit_ucb95_max": 0.05,
            "surface_row_lcb95_min": 0.90,
            "vocabulary_unit_lcb95_min": 0.90,
            "all_splits_required": list(SPLITS),
            "all_surfaces_required": list(SURFACES),
            "all_vocab_sets_required": ["vocab_a", "vocab_b"],
            "selection_updates_allowed": False,
        },
        "evidence_policy": {
            "semantic_unit_is_independent_denominator": True,
            "surface_rewrites_increase_independent_n": False,
            "behavior_qualification_is_physical_mechanism": False,
            "behavior_qualification_is_cross_model_mechanism": False,
            "historical_phase424_trace_can_close_new_contract": False,
            "physical_collection_requires_behavior_gate": True,
            "single_neuron_scan_requires_compute_and_causal_gate": True,
            "confirmation_split_is_open_but_frozen_before_model_run": True,
            "sealed_split_read": False,
        },
        "registered_cases_path": str(rows_path.relative_to(ROOT)),
        "registered_cases_sha256": sha256_file(rows_path),
        "static_audit_path": str(STATIC_AUDIT_PATH.relative_to(ROOT)),
        "static_audit_sha256": stable_hash(audit),
    }


def register() -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    write_jsonl(REGISTERED_CASES, rows)
    audit = validate(rows)
    write_json(STATIC_AUDIT_PATH, audit)
    write_json(REPRESENTATIVE_PATH, {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE,
        "created_at": now(),
        "frozen_72_mechanism_catalog": {family: list(mechanisms) for family, mechanisms in FAMILY_MECHANISMS.items()},
        "representative_screen": {family: list(mechanisms) for family, mechanisms in REPRESENTATIVES.items()},
        "selection_basis": "two structurally distinct natural contracts per family, frozen without Phase544 model effects",
        "selection_is_72_mechanism_completion": False,
    })
    write_json(PROTOCOL_PATH, protocol(REGISTERED_CASES, audit))
    if not audit["valid"]:
        raise SystemExit(json.dumps(audit, ensure_ascii=False, indent=2))
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    print(json.dumps(register(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
