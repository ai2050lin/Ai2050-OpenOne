#!/usr/bin/env python3
"""Phase1246: C001-WP01 typed behavior qualification.

This stage freezes a fresh answer-payload-free object-marker task, calibrates
an abstaining known-truth response camera, and runs Qwen3-4B once in FP16 CUDA.
It never reads hidden states and never authorizes a mechanism claim.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import random
import re
import string
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests/glm5"
OS_ROOT = ROOT / "research/ai2050_research_os"
sys.path.insert(0, str(TEST_ROOT))

from model_utils import MODEL_CONFIGS
from phase1023_fp16_utils import load_fp16, quantization_audit, release_fp16


PHASE = 1246
CONTRACT_ID = "EXP-C001-WP01-001"
RUN_ID = "RUN-EXP-C001-WP01-001-001"
SCRIPT = Path(__file__).resolve()
AUDIT_SCRIPT = TEST_ROOT / "phase1246_c001_wp01_typed_behavior_qualification_audit.py"
CONTRACT_PATH = OS_ROOT / f"contracts/{CONTRACT_ID}.json"
CONTRACT_INDEX_PATH = OS_ROOT / "registry/contracts.json"
RUNS_PATH = OS_ROOT / "registry/runs.json"
ARTIFACTS_PATH = OS_ROOT / "registry/artifacts.json"
MANIFEST_PATH = OS_ROOT / f"manifests/{CONTRACT_ID}.manifest.json"
SCHEMA_PATH = OS_ROOT / "schemas/experiment_contract.schema.json"
RESEARCHCTL = OS_ROOT / "scripts/researchctl.py"

OUT_ROOT = TEST_ROOT / "result/phase1246_c001_wp01_typed_behavior_qualification"
LOCAL_CONTRACT_PATH = OUT_ROOT / "protocol/scientific_contract_snapshot.json"
MATERIAL_PATH = OUT_ROOT / "material/frozen_typed_worlds.jsonl"
TOKEN_MANIFEST_PATH = OUT_ROOT / "material/qwen3_token_manifest.jsonl"
FIXTURE_PATH = OUT_ROOT / "material/evaluator_fixtures.jsonl"
CAMERA_PATH = OUT_ROOT / "calibration/known_truth_response_camera.json"
PROGRAM_PATH = OUT_ROOT / "calibration/alternative_program_audit.json"
PLAN_PATH = OUT_ROOT / "protocol/frozen_execution_plan.json"
ENVIRONMENT_PATH = OUT_ROOT / "protocol/environment_snapshot.json"
PREAUDIT_PATH = OUT_ROOT / "audit/independent_preaudit.json"
RAW_PATH = OUT_ROOT / "behavior/qwen3/raw_behavior.jsonl"
RUN_SUMMARY_PATH = OUT_ROOT / "behavior/qwen3/run_summary.json"
ADJUDICATION_PATH = OUT_ROOT / "analysis/typed_adjudication.json"
FINAL_PATH = OUT_ROOT / "analysis/final.json"
FINAL_AUDIT_PATH = OUT_ROOT / "audit/independent_final_audit.json"

PARTITIONS = ("calibration", "discovery", "selection", "confirmation")
PARTITION_SEEDS = {
    "calibration": 2026081101,
    "discovery": 2026081102,
    "selection": 2026081103,
    "confirmation": 2026081104,
}
SEALED_PARTITIONS = {"selection", "confirmation"}
WORLDS_PER_PARTITION = 128
WORLD_COUNT = WORLDS_PER_PARTITION * len(PARTITIONS)
STATES = (0, 1)
PROTOCOLS = ("candidate", "bare_short", "prompted_short", "fixed_sentence", "natural_sentence")
GENERATION_PROTOCOLS = PROTOCOLS[1:]
COLLISION_GROUPS = (
    "target_change",
    "nontarget_noop",
    "query_switch",
    "same_bag_binding_swap",
    "order_invariance",
    "template_invariance",
)
GENERATION_BUDGET = 18
GENERATION_BATCH_SIZE = 24
CANDIDATE_BATCH_SIZE = 32
RECOMPUTE_BATCH_SIZE = 24
MAX_INPUT_TOKENS = 256
TIE_TOLERANCE = 1e-7

CODEBOOK = {
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
    "10": "ten",
    "20": "twenty",
}

SYSTEM_PROMPT = (
    "Use only the supplied registry. Resolve the queried object, then translate its "
    "numeric marker code to the code's canonical lowercase English name. Do not explain."
)

RECORD_TEMPLATES = (
    "Registry: {object} carries numeric marker {code}; lane {lane}; seal {seal}.",
    "File for {object}: code {code} is its numeric marker, with lane {lane} and seal {seal}.",
    "For {object}, the numeric marker code reads {code}; its lane is {lane}; its seal is {seal}.",
    "Entry {object} reports numeric marker {code}, lane {lane}, and seal {seal}.",
)
QUERY_TEMPLATES = (
    "Which canonical English marker name belongs to {object}?",
    "Translate {object}'s numeric marker code into its canonical English name.",
    "Read {object}'s marker and return the code's English name.",
    "What is the English name of the numeric marker assigned to {object}?",
)
LANES = ("central", "eastern", "northern", "southern")
SEALS = ("ready", "stored", "checked", "sealed")
NAME_PREFIXES = (
    ("Zav", "Kiv", "Rul", "Mep"),
    ("Tov", "Lem", "Pax", "Nir"),
    ("Bex", "Sod", "Wyn", "Cav"),
    ("Jor", "Fyn", "Dax", "Huv"),
)
NAME_SUFFIXES = ("ara", "eno", "iri", "uma", "ora", "eli", "avo", "une")

THRESHOLDS = {
    "content_partition_accuracy": 0.95,
    "content_worst_template_name_accuracy": 0.90,
    "collision_complete_rate": 0.90,
    "alternative_program_advantage": 0.15,
    "format_partition_accuracy": 0.95,
    "format_worst_template_accuracy": 0.95,
    "natural_partition_accuracy": 0.95,
    "natural_worst_template_accuracy": 0.90,
    "cache_top1_agreement": 1.0,
    "correct_stop_rate": 0.95,
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")


def scientific_contract(contract: dict[str, Any]) -> dict[str, Any]:
    value = dict(contract)
    value.pop("status", None)
    value.pop("phase", None)
    value.pop("frozen_artifacts", None)
    return value


def render_chat(tokenizer: Any, prompt: str) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
    kwargs = {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
    try:
        return str(tokenizer.apply_chat_template(messages, **kwargs))
    except (TypeError, ValueError):
        kwargs.pop("enable_thinking", None)
        return str(tokenizer.apply_chat_template(messages, **kwargs))


def continuation_suffix(tokenizer: Any, rendered: str, continuation: str) -> tuple[list[int], list[int]]:
    prefix = [int(item) for item in tokenizer.encode(rendered, add_special_tokens=False)]
    appended = [int(item) for item in tokenizer.encode(rendered + continuation, add_special_tokens=False)]
    if appended[: len(prefix)] != prefix or len(appended) <= len(prefix):
        raise RuntimeError("continuation does not preserve native assistant boundary")
    return prefix, appended[len(prefix) :]


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def candidate_mentions(text: str, candidates: list[str]) -> list[str]:
    value = normalize(text)
    found: list[str] = []
    for candidate in candidates:
        if re.search(rf"(?<!\w){re.escape(candidate)}(?!\w)", value):
            found.append(candidate)
    return sorted(set(found))


def parse_generation(text: str, candidates: list[str], protocol: str, expected: str) -> dict[str, Any]:
    value = normalize(text)
    mentions = candidate_mentions(value, candidates)
    prediction: str | None = mentions[0] if len(mentions) == 1 else None
    rejected = None
    if prediction is None:
        rejected = "none_or_multiple_candidates"
    else:
        span = re.search(rf"(?<!\w){re.escape(prediction)}(?!\w)", value)
        before = value[max(0, (span.start() if span else 0) - 36) : (span.start() if span else 0)]
        after = value[(span.end() if span else 0) : (span.end() if span else 0) + 36]
        if re.search(r"\b(?:not|wrong|incorrect|isn't|isnt)\b|rather than", before):
            prediction, rejected = None, "negated_candidate"
        elif re.search(r"\b(?:sorry|correction|instead)\b", after):
            prediction, rejected = None, "self_correction"
    exact = text.strip() == expected
    escaped = "|".join(re.escape(candidate) for candidate in candidates)
    if protocol == "bare_short":
        format_valid = re.fullmatch(rf"(?:{escaped})", text.strip()) is not None
    elif protocol == "prompted_short":
        format_valid = re.fullmatch(rf"Marker name: (?:{escaped})\.", text.strip()) is not None
    elif protocol == "fixed_sentence":
        format_valid = re.fullmatch(rf"The canonical marker name is (?:{escaped})\.", text.strip()) is not None
    elif protocol == "natural_sentence":
        format_valid = prediction is not None and len(text.strip().split()) <= 20 and text.strip().endswith((".", "!", "?"))
    else:
        raise ValueError(protocol)
    return {
        "normalized": value,
        "mentions": mentions,
        "prediction": prediction,
        "rejected_reason": rejected,
        "content_correct": prediction == normalize(expected).split()[-1].strip(string.punctuation)
        if protocol != "bare_short"
        else prediction == normalize(expected),
        "format_valid": bool(format_valid),
        "exact": bool(exact),
    }


def expected_output(protocol: str, gold: str) -> str:
    if protocol == "bare_short":
        return gold
    if protocol == "prompted_short":
        return f"Marker name: {gold}."
    if protocol == "fixed_sentence":
        return f"The canonical marker name is {gold}."
    if protocol == "natural_sentence":
        return f"The marker is {gold}."
    raise ValueError(protocol)


def protocol_instruction(protocol: str) -> str:
    if protocol == "candidate":
        return "Return the canonical lowercase marker name and nothing else."
    if protocol == "bare_short":
        return "Reply with only the lowercase marker name."
    if protocol == "prompted_short":
        return "Reply exactly as: Marker name: <lowercase-name>."
    if protocol == "fixed_sentence":
        return "Reply exactly with this pattern: The canonical marker name is <lowercase-name>."
    if protocol == "natural_sentence":
        return "Answer in a concise grammatical sentence naming the marker."
    raise ValueError(protocol)


def make_name(partition: str, world_index: int, slot: int, name_world: int) -> str:
    prefix = NAME_PREFIXES[name_world][slot]
    suffix = NAME_SUFFIXES[(world_index * 3 + slot * 5 + name_world) % len(NAME_SUFFIXES)]
    letters = string.ascii_uppercase
    tag = letters[(world_index + slot) % 26] + letters[(world_index * 7 + slot * 11) % 26]
    return f"{prefix}{suffix}-{partition[:2].upper()}{tag}"


def base_assignments(codes: list[str], rng: random.Random) -> list[str]:
    values = [codes[0], codes[0], codes[1], codes[2]]
    rng.shuffle(values)
    return values


def unique_index(assignments: list[str], code: str) -> int:
    indices = [index for index, value in enumerate(assignments) if value == code]
    if len(indices) != 1:
        raise RuntimeError("expected unique assignment")
    return indices[0]


def render_records(
    objects: list[str], assignments: list[str], order: list[int], template_index: int, lanes: list[str], seals: list[str]
) -> list[str]:
    template = RECORD_TEMPLATES[template_index]
    return [
        template.format(object=objects[index], code=assignments[index], lane=lanes[index], seal=seals[index])
        for index in order
    ]


def build_prompt(records: list[str], query_object: str, template_index: int, protocol: str) -> str:
    question = QUERY_TEMPLATES[template_index].format(object=query_object)
    return "Records:\n" + "\n".join(records) + f"\nQuestion: {question}\n{protocol_instruction(protocol)}"


def build_material() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    candidate_codes = list(CODEBOOK)
    for partition_index, partition in enumerate(PARTITIONS):
        rng = random.Random(PARTITION_SEEDS[partition])
        for world_index in range(WORLDS_PER_PARTITION):
            collision = COLLISION_GROUPS[(world_index + partition_index * 2) % len(COLLISION_GROUPS)]
            # Cycle name worlds by complete six-collision blocks so every
            # collision-induced gold slot crosses every name world.
            name_world = (world_index // len(COLLISION_GROUPS) + partition_index) % len(NAME_PREFIXES)
            objects = [make_name(partition, world_index, slot, name_world) for slot in range(4)]
            codes = rng.sample(candidate_codes, 5)
            assignments0 = base_assignments(codes, rng)
            assignments1 = list(assignments0)
            code_a, code_b, code_c = codes[:3]
            index_b = unique_index(assignments0, code_b)
            index_c = unique_index(assignments0, code_c)
            query0 = query1 = index_b
            order0 = list(range(4))
            rng.shuffle(order0)
            order1 = list(order0)
            template0 = rng.randrange(len(RECORD_TEMPLATES))
            template1 = template0
            if collision == "target_change":
                assignments1[index_b] = code_a
            elif collision == "nontarget_noop":
                assignments1[index_c] = code_a
            elif collision == "query_switch":
                query1 = index_c
            elif collision == "same_bag_binding_swap":
                assignments1[index_b], assignments1[index_c] = assignments1[index_c], assignments1[index_b]
            elif collision == "order_invariance":
                order1 = order0[1:] + order0[:1]
            elif collision == "template_invariance":
                template1 = (template0 + 1 + (world_index % 2)) % len(RECORD_TEMPLATES)
            else:
                raise ValueError(collision)
            lanes = [LANES[(world_index + slot + partition_index) % len(LANES)] for slot in range(4)]
            seals = [SEALS[(world_index * 2 + slot + partition_index) % len(SEALS)] for slot in range(4)]
            candidate_order = [CODEBOOK[code] for code in codes]
            rng.shuffle(candidate_order)
            world_id = f"{partition[:3]}-{world_index:03d}-{collision}"
            states = (
                (0, assignments0, query0, order0, template0),
                (1, assignments1, query1, order1, template1),
            )
            for state, assignments, query_index, order, template_index in states:
                records = render_records(objects, assignments, order, template_index, lanes, seals)
                gold_code = assignments[query_index]
                gold = CODEBOOK[gold_code]
                prompts = {
                    protocol: build_prompt(records, objects[query_index], template_index, protocol)
                    for protocol in PROTOCOLS
                }
                row: dict[str, Any] = {
                    "phase": PHASE,
                    "schema_version": "phase1246.material.row.v1",
                    "partition": partition,
                    "sealed": partition in SEALED_PARTITIONS,
                    "world_id": world_id,
                    "world_index": world_index,
                    "state": state,
                    "row_id": f"{world_id}-s{state}",
                    "collision_group": collision,
                    "name_world": name_world,
                    "objects": objects,
                    "assignments": dict(zip(objects, assignments)),
                    "candidate_codes": codes,
                    "candidate_order": candidate_order,
                    "query_index": query_index,
                    "query_object": objects[query_index],
                    "gold_code": gold_code,
                    "gold": gold,
                    "record_order": order,
                    "template_index": template_index,
                    "lanes": lanes,
                    "seals": seals,
                    "records": records,
                    "prompts": prompts,
                    "expected_outputs": {
                        protocol: expected_output(protocol, gold) for protocol in GENERATION_PROTOCOLS
                    },
                    "natural_references": [
                        f"The marker is {gold}.",
                        f"Its marker is {gold}.",
                        f"{gold.capitalize()} is the marker.",
                    ],
                }
                row["row_digest"] = digest(row)
                rows.append(row)
    rows.sort(key=lambda row: (PARTITIONS.index(row["partition"]), row["world_index"], row["state"]))
    if len(rows) != WORLD_COUNT * 2:
        raise RuntimeError("material row count drift")
    return rows


def make_token_manifest(rows: list[dict[str, Any]], slow: Any, fast: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    token_rows: list[dict[str, Any]] = []
    candidate_token_ids: dict[str, int] = {}
    for word in CODEBOOK.values():
        encoded = slow.encode(word, add_special_tokens=False)
        if len(encoded) != 1:
            raise RuntimeError(f"candidate is not native single token: {word} -> {encoded}")
        candidate_token_ids[word] = int(encoded[0])
    if len(set(candidate_token_ids.values())) != len(candidate_token_ids):
        raise RuntimeError("candidate token collision")
    maximum_input = 0
    mismatch = 0
    answer_overlap = 0
    boundary_failures = 0
    for execution_index, row in enumerate(rows):
        inputs: dict[str, list[int]] = {}
        for protocol in PROTOCOLS:
            rendered = render_chat(slow, row["prompts"][protocol])
            slow_ids = [int(value) for value in slow.encode(rendered, add_special_tokens=False)]
            fast_ids = [int(value) for value in fast.encode(rendered, add_special_tokens=False)]
            mismatch += int(slow_ids != fast_ids)
            inputs[protocol] = slow_ids
            maximum_input = max(maximum_input, len(slow_ids))
            forbidden_ids = {candidate_token_ids[word] for word in row["candidate_order"]}
            answer_overlap += len(forbidden_ids.intersection(slow_ids))
            for candidate in row["candidate_order"]:
                prefix, suffix = continuation_suffix(slow, rendered, candidate)
                if prefix != slow_ids or suffix != [candidate_token_ids[candidate]]:
                    boundary_failures += 1
        token_row: dict[str, Any] = {
            "phase": PHASE,
            "schema_version": "phase1246.token_manifest.row.v1",
            "execution_index": execution_index,
            "row_id": row["row_id"],
            "partition": row["partition"],
            "world_id": row["world_id"],
            "state": row["state"],
            "candidate_token_ids": {word: candidate_token_ids[word] for word in row["candidate_order"]},
            "input_ids": inputs,
            "input_lengths": {protocol: len(ids) for protocol, ids in inputs.items()},
        }
        token_row["token_row_digest"] = digest(token_row)
        token_rows.append(token_row)
    summary = {
        "candidate_token_ids": candidate_token_ids,
        "slow_tokenizer": type(slow).__name__,
        "fast_tokenizer": type(fast).__name__,
        "row_count": len(token_rows),
        "slow_fast_mismatch_count": mismatch,
        "answer_token_overlap_count": answer_overlap,
        "assistant_boundary_failure_count": boundary_failures,
        "maximum_input_tokens": maximum_input,
        "tokenizer_gate": mismatch == 0 and answer_overlap == 0 and boundary_failures == 0 and maximum_input <= MAX_INPUT_TOKENS,
    }
    return token_rows, summary


def multiset_digest(prompt: str) -> str:
    return digest(sorted(re.findall(r"[A-Za-z]+|\d+|[^\w\s]", prompt.lower())))


def alternative_predictions(row: dict[str, Any]) -> dict[str, str]:
    candidates = row["candidate_order"]
    assignment = row["assignments"]
    objects = row["objects"]
    ordered_objects = [objects[index] for index in row["record_order"]]
    values = [assignment[obj] for obj in objects]
    value_names = [CODEBOOK[value] for value in values]
    counts = Counter(value_names)
    modal = sorted(counts, key=lambda item: (-counts[item], item))[0]
    query_position = ordered_objects.index(row["query_object"])
    query_index = row["query_index"]
    result = {f"candidate_slot_{index}": candidates[index] for index in range(5)}
    result.update(
        {
            "first_record": CODEBOOK[assignment[ordered_objects[0]]],
            "last_record": CODEBOOK[assignment[ordered_objects[-1]]],
            "next_object": value_names[(query_index + 1) % 4],
            "previous_object": value_names[(query_index - 1) % 4],
            "opposite_object": value_names[(query_index + 2) % 4],
            "modal_value": modal,
            "minimum_numeric": CODEBOOK[min(row["candidate_codes"], key=int)],
            "maximum_numeric": CODEBOOK[max(row["candidate_codes"], key=int)],
            "query_position_to_candidate_slot": candidates[query_position],
            "template_to_candidate_slot": candidates[row["template_index"]],
            "name_world_to_candidate_slot": candidates[row["name_world"]],
        }
    )
    return result


def build_program_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    names = sorted(alternative_predictions(rows[0]))
    by_partition: dict[str, Any] = {}
    for partition in PARTITIONS:
        selected = [row for row in rows if row["partition"] == partition]
        scores = {
            name: sum(alternative_predictions(row)[name] == row["gold"] for row in selected) / len(selected)
            for name in names
        }
        group_conditioned = 0
        for group in COLLISION_GROUPS:
            cell = [row for row in selected if row["collision_group"] == group]
            group_conditioned += max(sum(alternative_predictions(row)[name] == row["gold"] for row in cell) for name in names)
        group_conditioned /= len(selected)
        template_conditioned = 0
        for template in range(len(RECORD_TEMPLATES)):
            cell = [row for row in selected if row["template_index"] == template]
            if cell:
                template_conditioned += max(sum(alternative_predictions(row)[name] == row["gold"] for row in cell) for name in names)
        template_conditioned /= len(selected)
        by_partition[partition] = {
            "program_accuracies": scores,
            "maximum_fixed_program_accuracy": max(scores.values()),
            "collision_group_conditioned_ceiling": group_conditioned,
            "template_conditioned_ceiling": template_conditioned,
            "strongest_registered_ceiling": max(max(scores.values()), group_conditioned, template_conditioned),
        }
    checks: dict[str, bool] = {}
    grouped_worlds: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped_worlds[row["world_id"]].append(row)
    checks["every_world_has_two_states"] = all(len(cell) == 2 for cell in grouped_worlds.values())
    checks["nonbijective_each_state"] = all(len(set(row["assignments"].values())) < len(row["assignments"]) for row in rows)
    checks["unused_candidate_each_state"] = all(
        len(set(row["candidate_codes"]) - set(row["assignments"].values())) >= 1 for row in rows
    )
    checks["same_bag_swap_exact"] = all(
        multiset_digest(cell[0]["prompts"]["candidate"]) == multiset_digest(cell[1]["prompts"]["candidate"])
        for cell in grouped_worlds.values()
        if cell[0]["collision_group"] == "same_bag_binding_swap"
    )
    expected_change = {"target_change", "query_switch", "same_bag_binding_swap"}
    checks["collision_semantics"] = all(
        ((cell[0]["gold"] != cell[1]["gold"]) == (cell[0]["collision_group"] in expected_change))
        for cell in grouped_worlds.values()
    )
    checks["all_collision_groups_each_partition"] = all(
        set(row["collision_group"] for row in rows if row["partition"] == partition) == set(COLLISION_GROUPS)
        for partition in PARTITIONS
    )
    checks["partition_names_disjoint"] = all(
        not ({obj for row in rows if row["partition"] == left for obj in row["objects"]} &
             {obj for row in rows if row["partition"] == right for obj in row["objects"]})
        for index, left in enumerate(PARTITIONS) for right in PARTITIONS[index + 1 :]
    )
    value: dict[str, Any] = {
        "phase": PHASE,
        "schema_version": "phase1246.alternative_program_audit.v1",
        "created_at_utc": utc_now(),
        "registered_programs": names,
        "conditioning_grammar": ["collision_group", "template_index"],
        "partitions": by_partition,
        "construct_checks": checks,
    }
    value["program_gate"] = all(checks.values()) and max(
        cell["strongest_registered_ceiling"] for cell in by_partition.values()
    ) <= 0.80
    value["program_audit_digest"] = digest(value)
    return value


def response_template(family: str) -> np.ndarray:
    value = np.zeros((6, 4, 4, 6), dtype=np.float64)
    if family == "boundary_store":
        value[0, 0, 1:, 0] = 1
        value[1, 0, 2:, 1] = 1
        value[2] = value[0] + value[1]
        value[3, 1, :, 2] = 1
        value[3, 2, 2:, 3] = 1
        value[4, 3, 2:, 4] = 1
    elif family == "query_recompute":
        value[0, 0, :2, 0] = (1, 0.5)
        value[1, 0, :, 1] = (0.25, 0.5, 0.75, 1)
        value[2] = value[0] + value[1]
        value[2, 0, 2:, 5] = 1
        value[3, 1, 1:, 2] = 1
        value[3, 2, 1:, 3] = 0.5
        value[4, 3, 1:, 4] = 1
    elif family == "distributed_gate":
        value[0, 0, 1:, 0] = 0.25
        value[1, 0, 1:, 1] = 0.25
        value[2, 0, 1:, 5] = (0.5, 1, 1)
        value[2, 2, 2:, 5] = 0.75
        value[3, 1, :, 2] = 0.75
        value[3, 2, :, 2] = 0.75
        value[4, 3, 2:, 4] = 1
    elif family == "alternating_unknown":
        value[0, 0, :, 0] = (1, -1, 1, -1)
        value[1, 2, :, 1] = (-1, 1, -1, 1)
        value[2, 1, :, 5] = (1, 0, -1, 0)
        value[3, 3, :, 3] = (0, 1, 0, -1)
        value[4, 0, :, 4] = (0.2, 0.4, 0.8, 1.6)
    elif family == "twin_a":
        value = response_template("boundary_store")
        value[4, 3, :, 5] = 0
    elif family == "twin_b":
        value = response_template("boundary_store")
        value[4, 3, :, 5] = (0, 0, 1, 2)
    else:
        raise ValueError(family)
    return value


def gauge_transform(value: np.ndarray, rng: random.Random) -> np.ndarray:
    permutation = list(range(value.shape[-1]))
    rng.shuffle(permutation)
    signs = np.array([rng.choice((-1.0, 1.0)) for _ in permutation], dtype=np.float64)
    return value[..., permutation] * signs


def response_signature(value: np.ndarray, interventions: tuple[int, ...] = tuple(range(6))) -> np.ndarray:
    matrix = value[list(interventions)].reshape(-1, value.shape[-1])
    gram = matrix @ matrix.T
    norm = np.linalg.norm(gram)
    return gram.reshape(-1) / norm if norm else gram.reshape(-1)


def classify_response(
    value: np.ndarray, library: dict[str, np.ndarray], interventions: tuple[int, ...], tolerance: float = 1e-10
) -> dict[str, Any]:
    observed = response_signature(value, interventions)
    distances = {
        name: float(np.linalg.norm(observed - response_signature(template, interventions)))
        for name, template in library.items()
    }
    minimum = min(distances.values())
    winners = sorted(name for name, distance in distances.items() if abs(distance - minimum) <= tolerance)
    if minimum > tolerance:
        label = "open_unknown"
    elif len(winners) != 1:
        label = "abstain_nonidentifiable"
    else:
        label = winners[0]
    return {"label": label, "minimum_distance": minimum, "winners": winners, "distances": distances}


def build_known_truth_camera() -> dict[str, Any]:
    families = ("boundary_store", "query_recompute", "distributed_gate")
    library = {family: response_template(family) for family in families}
    rng = random.Random(12460031)
    recovery: list[dict[str, Any]] = []
    unknown: list[dict[str, Any]] = []
    unseen: list[dict[str, Any]] = []
    observed_interventions = (0, 1, 3, 4, 5)
    for split in ("discovery", "confirmation"):
        for family in families:
            for index in range(32):
                tensor = gauge_transform(response_template(family), rng)
                result = classify_response(tensor, library, tuple(range(6)))
                recovery.append({"split": split, "family": family, "index": index, "prediction": result["label"]})
                partial = classify_response(tensor, library, observed_interventions)
                unseen.append({"split": split, "family": family, "index": index, "prediction": partial["label"]})
        for index in range(32):
            tensor = gauge_transform(response_template("alternating_unknown"), rng)
            result = classify_response(tensor, library, tuple(range(6)))
            unknown.append({"split": split, "index": index, "prediction": result["label"]})
    twin_library = {"twin_a": response_template("twin_a"), "twin_b": response_template("twin_b")}
    twin_restricted = []
    twin_expanded = []
    restricted = (0, 1, 2, 3, 5)
    for family in twin_library:
        for index in range(32):
            tensor = gauge_transform(response_template(family), rng)
            twin_restricted.append(classify_response(tensor, twin_library, restricted)["label"])
            twin_expanded.append((family, classify_response(tensor, twin_library, tuple(range(6)))["label"]))
    null = np.zeros_like(response_template("boundary_store"))
    null_norm = float(np.linalg.norm(null))
    wrong_donor = float(
        np.linalg.norm(
            response_signature(response_template("boundary_store"))
            - response_signature(response_template("distributed_gate"))
        )
    )
    metrics = {
        "known_family_recovery": sum(row["family"] == row["prediction"] for row in recovery) / len(recovery),
        "open_unknown_rejection": sum(row["prediction"] == "open_unknown" for row in unknown) / len(unknown),
        "restricted_twin_abstention": sum(label == "abstain_nonidentifiable" for label in twin_restricted) / len(twin_restricted),
        "expanded_basis_twin_recovery": sum(expected == predicted for expected, predicted in twin_expanded) / len(twin_expanded),
        "heldout_intervention_family_recovery": sum(row["family"] == row["prediction"] for row in unseen) / len(unseen),
        "null_response_norm": null_norm,
        "wrong_donor_signature_distance": wrong_donor,
    }
    checks = {
        "target_recovery": metrics["known_family_recovery"] == 1.0,
        "open_discovery_channel": metrics["open_unknown_rejection"] == 1.0,
        "nonidentifiable_abstention": metrics["restricted_twin_abstention"] == 1.0,
        "basis_expansion_recovers_twins": metrics["expanded_basis_twin_recovery"] == 1.0,
        "unseen_intervention_prediction": metrics["heldout_intervention_family_recovery"] == 1.0,
        "matched_null_zero": metrics["null_response_norm"] == 0.0,
        "wrong_donor_specificity": metrics["wrong_donor_signature_distance"] > 0.10,
    }
    value: dict[str, Any] = {
        "phase": PHASE,
        "schema_version": "phase1246.known_truth_response_camera.v1",
        "created_at_utc": utc_now(),
        "object": "synthetic future-response tensors with known mechanism family, gauge and identifiability status",
        "families": list(families),
        "open_unknown_family": "alternating_unknown",
        "gauge": "signed permutation of latent response channels",
        "primary_rule": "predictive signature match with explicit abstention on ties and open_unknown on residual mismatch",
        "description_length_role": "secondary report only; never overrides predictive sufficiency or abstention",
        "metrics": metrics,
        "checks": checks,
        "claim_boundary": "This calibrates an adjudication camera on scripted truth; it does not show that Qwen3 uses any listed mechanism family.",
    }
    value["camera_gate"] = all(checks.values())
    value["camera_digest"] = digest(value)
    return value


def evaluator_fixtures() -> list[dict[str, Any]]:
    candidates = ["one", "two", "three", "four", "five"]
    raw = [
        ("bare_cc", "bare_short", "one", "one", "one", True, True),
        ("bare_cf", "bare_short", "one.", "one", "one", True, False),
        ("bare_wc", "bare_short", "two", "one", "two", False, True),
        ("bare_wf", "bare_short", "Marker name: two.", "one", "two", False, False),
        ("prompt_cc", "prompted_short", "Marker name: one.", "Marker name: one.", "one", True, True),
        ("prompt_cf", "prompted_short", "one", "Marker name: one.", "one", True, False),
        ("prompt_wc", "prompted_short", "Marker name: two.", "Marker name: one.", "two", False, True),
        ("fixed_cc", "fixed_sentence", "The canonical marker name is one.", "The canonical marker name is one.", "one", True, True),
        ("fixed_cf", "fixed_sentence", "The marker is one.", "The canonical marker name is one.", "one", True, False),
        ("fixed_wc", "fixed_sentence", "The canonical marker name is two.", "The canonical marker name is one.", "two", False, True),
        ("natural_cc", "natural_sentence", "Its marker is one.", "The marker is one.", "one", True, True),
        ("natural_wrong", "natural_sentence", "Its marker is two.", "The marker is one.", "two", False, True),
        ("negation", "natural_sentence", "The marker is not one.", "The marker is one.", None, False, False),
        ("multiple", "natural_sentence", "It is either one or two.", "The marker is one.", None, False, False),
        ("correction", "natural_sentence", "It is one, sorry, two.", "The marker is one.", None, False, False),
    ]
    rows = []
    for fixture_id, protocol, text, expected, prediction, content, fmt in raw:
        rows.append(
            {
                "fixture_id": fixture_id,
                "protocol": protocol,
                "text": text,
                "candidates": candidates,
                "expected": expected,
                "expected_prediction": prediction,
                "expected_content_correct": content,
                "expected_format_valid": fmt,
            }
        )
    return rows


def environment_snapshot(tokenizer_summary: dict[str, Any]) -> dict[str, Any]:
    gpu = None
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        gpu = {"name": properties.name, "total_memory_bytes": int(properties.total_memory)}
    model_path = Path(MODEL_CONFIGS["qwen3"]["path"])
    identity_files = [model_path / name for name in ("config.json", "tokenizer_config.json", "tokenizer.json")]
    return {
        "phase": PHASE,
        "created_at_utc": utc_now(),
        "python_executable": sys.executable,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "gpu": gpu,
        "model_path": str(model_path),
        "model_identity_files": {
            path.name: file_sha256(path) for path in identity_files if path.is_file()
        },
        "precision": "float16",
        "quantization": "none",
        "tokenizer_summary": tokenizer_summary,
        "model_weights_loaded": False,
    }


def prepare() -> None:
    if OUT_ROOT.exists():
        raise RuntimeError(f"output directory already exists: {OUT_ROOT}")
    contract = read_json(CONTRACT_PATH)
    if contract.get("status") != "preregistered" or contract.get("frozen_artifacts", {}).get("readiness") != "contract_frozen":
        raise RuntimeError("C001-WP01 contract is not in preregistered contract_frozen state")
    from transformers import AutoTokenizer

    slow = AutoTokenizer.from_pretrained(
        MODEL_CONFIGS["qwen3"]["path"], trust_remote_code=True, local_files_only=True, use_fast=False
    )
    fast = AutoTokenizer.from_pretrained(
        MODEL_CONFIGS["qwen3"]["path"], trust_remote_code=True, local_files_only=True, use_fast=True
    )
    rows = build_material()
    token_rows, tokenizer_summary = make_token_manifest(rows, slow, fast)
    program = build_program_audit(rows)
    camera = build_known_truth_camera()
    fixtures = evaluator_fixtures()
    if not tokenizer_summary["tokenizer_gate"] or not program["program_gate"] or not camera["camera_gate"]:
        raise RuntimeError("zero-model construction gate failed")
    local_contract = {
        "phase": PHASE,
        "contract_id": CONTRACT_ID,
        "created_at_utc": utc_now(),
        "scientific_contract": scientific_contract(contract),
        "scientific_contract_digest": digest(scientific_contract(contract)),
        "world_count": WORLD_COUNT,
        "row_count": len(rows),
        "protocols": list(PROTOCOLS),
        "thresholds": THRESHOLDS,
        "material_digest": digest(rows),
        "token_manifest_digest": digest(token_rows),
        "program_audit_digest": program["program_audit_digest"],
        "camera_digest": camera["camera_digest"],
        "forbidden": [
            "read hidden states, attentions, heads, MLP outputs or neurons",
            "change prompts, candidates, parser, partitions, thresholds or denominators after preparation",
            "run GLM4 or DS7B",
            "retry Qwen3 after a behavioral failure",
            "promote known-truth camera calibration to Qwen3 mechanism evidence",
        ],
    }
    local_contract["snapshot_digest"] = digest(local_contract)
    plan = {
        "phase": PHASE,
        "schema_version": "phase1246.execution_plan.v1",
        "execution_order": [row["row_id"] for row in rows],
        "candidate_batch_size": CANDIDATE_BATCH_SIZE,
        "generation_batch_size": GENERATION_BATCH_SIZE,
        "recompute_batch_size": RECOMPUTE_BATCH_SIZE,
        "generation_budget": GENERATION_BUDGET,
        "cache_recompute_scope": "state0 of every selection and confirmation world under natural_sentence",
        "model_runs": 1,
        "adaptive_rounds": 0,
    }
    plan["plan_digest"] = digest(plan)
    write_json(LOCAL_CONTRACT_PATH, local_contract)
    write_jsonl(MATERIAL_PATH, rows)
    write_jsonl(TOKEN_MANIFEST_PATH, token_rows)
    write_jsonl(FIXTURE_PATH, fixtures)
    write_json(CAMERA_PATH, camera)
    write_json(PROGRAM_PATH, program)
    write_json(PLAN_PATH, plan)
    write_json(ENVIRONMENT_PATH, environment_snapshot(tokenizer_summary))
    print(canonical_json({
        "status": "phase1246_prepared",
        "worlds": WORLD_COUNT,
        "rows": len(rows),
        "tokenizer_gate": tokenizer_summary["tokenizer_gate"],
        "program_gate": program["program_gate"],
        "camera_gate": camera["camera_gate"],
        "snapshot_digest": local_contract["snapshot_digest"],
    }))


def workspace_relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def frozen_input_files() -> list[tuple[str, Path]]:
    return [
        ("contract", CONTRACT_PATH),
        ("schema", SCHEMA_PATH),
        ("main_code", SCRIPT),
        ("independent_auditor", AUDIT_SCRIPT),
        ("model_loader", TEST_ROOT / "model_utils.py"),
        ("fp16_loader", TEST_ROOT / "phase1023_fp16_utils.py"),
        ("scientific_contract_snapshot", LOCAL_CONTRACT_PATH),
        ("material", MATERIAL_PATH),
        ("token_manifest", TOKEN_MANIFEST_PATH),
        ("evaluator_fixtures", FIXTURE_PATH),
        ("known_truth_camera", CAMERA_PATH),
        ("alternative_program_audit", PROGRAM_PATH),
        ("execution_plan", PLAN_PATH),
        ("environment", ENVIRONMENT_PATH),
        ("preaudit", PREAUDIT_PATH),
    ]


def write_input_manifest(readiness: str) -> None:
    previous = read_json(MANIFEST_PATH) if MANIFEST_PATH.is_file() else {}
    files = []
    for role, path in frozen_input_files():
        if not path.is_file():
            raise RuntimeError(f"missing frozen input {role}: {path}")
        files.append(
            {
                "role": role,
                "path": workspace_relative(path),
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    value = {
        "manifest_version": "1.1.0",
        "contract_id": CONTRACT_ID,
        "frozen_at": previous.get("frozen_at", utc_now()),
        "readiness": readiness,
        "files": files,
    }
    write_json(MANIFEST_PATH, value)


def mutate_contract_status(status: str, previous_status: str, readiness: str) -> None:
    contract = read_json(CONTRACT_PATH)
    snapshot = read_json(LOCAL_CONTRACT_PATH)
    if digest(scientific_contract(contract)) != snapshot["scientific_contract_digest"]:
        raise RuntimeError("scientific contract drift")
    contract["phase"] = PHASE
    contract["status"] = status
    contract["frozen_artifacts"] = {
        "freeze_mode": "external_manifest",
        "manifest_path": f"manifests/{CONTRACT_ID}.manifest.json",
        "code_paths": [workspace_relative(SCRIPT), workspace_relative(AUDIT_SCRIPT)],
        "material_paths": [workspace_relative(MATERIAL_PATH), workspace_relative(TOKEN_MANIFEST_PATH)],
        "environment_paths": [workspace_relative(ENVIRONMENT_PATH), workspace_relative(SCHEMA_PATH)],
        # The contract index keeps run_ready=true as historical authorization;
        # terminal state is represented by status and manifest readiness.
        "readiness": "run_ready",
    }
    write_json(CONTRACT_PATH, contract)
    write_input_manifest(readiness)
    indices = read_json(CONTRACT_INDEX_PATH)
    record = next(row for row in indices if row["id"] == CONTRACT_ID)
    record.update(
        {
            "status": status,
            "previous_status": previous_status,
            "contract_sha256": file_sha256(CONTRACT_PATH),
            "run_ready": True,
        }
    )
    write_json(CONTRACT_INDEX_PATH, indices)


def upsert_run(status: str, previous_status: str, **updates: Any) -> None:
    runs = read_json(RUNS_PATH)
    existing = next((row for row in runs if row["id"] == RUN_ID), None)
    if existing is None:
        existing = {
            "id": RUN_ID,
            "contract_id": CONTRACT_ID,
            "status": status,
            "previous_status": previous_status,
            "model": "Qwen3-4B",
            "started_at": None,
            "ended_at": None,
            "finished_at": None,
            "gpu_hours": 0.0,
            "artifact_ids": [],
            "verdict": None,
            "notes": "Single preregistered FP16 CUDA run for C001-WP01; no hidden-state collection.",
        }
        runs.append(existing)
    else:
        existing["status"] = status
        existing["previous_status"] = previous_status
    existing.update(updates)
    write_json(RUNS_PATH, runs)


def run_researchctl(*args: str) -> None:
    environment = dict(os.environ)
    environment["PYTHONIOENCODING"] = "utf-8"
    normalized_args = list(args)
    if normalized_args[:2] == ["verify-manifest", CONTRACT_ID]:
        normalized_args[1] = f"manifests/{CONTRACT_ID}.manifest.json"
    process = subprocess.run(
        [sys.executable, str(RESEARCHCTL), *normalized_args],
        cwd=ROOT,
        env=environment,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    if process.returncode != 0:
        raise RuntimeError(f"researchctl {' '.join(normalized_args)} failed:\n{process.stdout}\n{process.stderr}")


def authorize() -> None:
    if not PREAUDIT_PATH.is_file() or read_json(PREAUDIT_PATH).get("all_checks_passed") is not True:
        raise RuntimeError("independent preaudit is missing or failed")
    if RUN_ID in {row["id"] for row in read_json(RUNS_PATH)}:
        raise RuntimeError("run record already exists")
    mutate_contract_status("ready", "preregistered", "run_ready")
    upsert_run("ready", "planned")
    run_researchctl("validate")
    run_researchctl("verify-manifest", CONTRACT_ID)
    print(canonical_json({"status": "phase1246_run_ready", "contract": CONTRACT_ID, "run": RUN_ID}))


def verify_frozen_inputs() -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    local_contract = read_json(LOCAL_CONTRACT_PATH)
    if local_contract["snapshot_digest"] != digest({k: v for k, v in local_contract.items() if k != "snapshot_digest"}):
        raise RuntimeError("local scientific contract snapshot drift")
    contract = read_json(CONTRACT_PATH)
    if digest(scientific_contract(contract)) != local_contract["scientific_contract_digest"]:
        raise RuntimeError("scientific contract drift")
    if read_json(PREAUDIT_PATH).get("all_checks_passed") is not True:
        raise RuntimeError("preaudit did not pass")
    run_researchctl("verify-manifest", CONTRACT_ID)
    rows = read_jsonl(MATERIAL_PATH)
    token_rows = read_jsonl(TOKEN_MANIFEST_PATH)
    plan = read_json(PLAN_PATH)
    if digest(rows) != local_contract["material_digest"]:
        raise RuntimeError("material drift")
    if digest(token_rows) != local_contract["token_manifest_digest"]:
        raise RuntimeError("token manifest drift")
    if [row["row_id"] for row in rows] != plan["execution_order"]:
        raise RuntimeError("execution order drift")
    return local_contract, rows, token_rows, plan


def homogeneous_batches(entries: list[dict[str, Any]], length_key: str, batch_size: int) -> Iterable[list[dict[str, Any]]]:
    buckets: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        buckets[int(entry[length_key])].append(entry)
    for length in sorted(buckets):
        values = buckets[length]
        for start in range(0, len(values), batch_size):
            yield values[start : start + batch_size]


def argmax_set(scores: dict[str, float]) -> list[str]:
    maximum = max(scores.values())
    return sorted(name for name, value in scores.items() if maximum - value <= TIE_TOLERANCE)


def score_candidates(model: Any, device: torch.device, token_rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    entries = [
        {
            "row_id": row["row_id"],
            "input_ids": row["input_ids"]["candidate"],
            "length": row["input_lengths"]["candidate"],
            "candidate_token_ids": row["candidate_token_ids"],
        }
        for row in token_rows
    ]
    output: dict[str, Any] = {}
    started = time.time()
    batch_count = 0
    with torch.inference_mode():
        for batch in homogeneous_batches(entries, "length", CANDIDATE_BATCH_SIZE):
            input_ids = torch.tensor([row["input_ids"] for row in batch], dtype=torch.long, device=device)
            logits = model(input_ids=input_ids, use_cache=False).logits[:, -1].float()
            finite_rows = torch.isfinite(logits).all(dim=-1).detach().cpu().tolist()
            for index, row in enumerate(batch):
                scores = {
                    candidate: float(logits[index, token_id].item())
                    for candidate, token_id in row["candidate_token_ids"].items()
                }
                winners = argmax_set(scores) if all(math.isfinite(value) for value in scores.values()) else []
                output[row["row_id"]] = {
                    "scores": scores,
                    "argmax_set": winners,
                    "prediction": winners[0] if len(winners) == 1 else None,
                    "all_vocab_logits_finite": bool(finite_rows[index]),
                    "tie": len(winners) != 1,
                }
            batch_count += 1
    return output, {"batches": batch_count, "elapsed_seconds": time.time() - started}


def generate_protocol(
    model: Any, tokenizer: Any, device: torch.device, token_rows: list[dict[str, Any]], protocol: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    entries = [
        {
            "row_id": row["row_id"],
            "input_ids": row["input_ids"][protocol],
            "length": row["input_lengths"][protocol],
        }
        for row in token_rows
    ]
    output: dict[str, Any] = {}
    started = time.time()
    batch_count = 0
    eos_id = int(tokenizer.eos_token_id)
    with torch.inference_mode():
        for batch in homogeneous_batches(entries, "length", GENERATION_BATCH_SIZE):
            prompt_length = int(batch[0]["length"])
            input_ids = torch.tensor([row["input_ids"] for row in batch], dtype=torch.long, device=device)
            generated = model.generate(
                input_ids=input_ids,
                do_sample=False,
                max_new_tokens=GENERATION_BUDGET,
                eos_token_id=eos_id,
                pad_token_id=eos_id,
                return_dict_in_generate=True,
                output_scores=True,
                use_cache=True,
            )
            sequences = generated.sequences.detach().cpu().tolist()
            score_finite = [True] * len(batch)
            for score in generated.scores:
                finite = torch.isfinite(score).all(dim=-1).detach().cpu().tolist()
                score_finite = [left and bool(right) for left, right in zip(score_finite, finite)]
            for index, row in enumerate(batch):
                suffix = [int(value) for value in sequences[index][prompt_length:]]
                stop_index = suffix.index(eos_id) if eos_id in suffix else None
                meaningful = suffix[:stop_index] if stop_index is not None else suffix
                trajectory = suffix[: stop_index + 1] if stop_index is not None else suffix
                output[row["row_id"]] = {
                    "text": tokenizer.decode(meaningful, skip_special_tokens=True),
                    "generated_token_ids": trajectory,
                    "generated_token_count": len(trajectory),
                    "stop_source": "model_eos" if stop_index is not None else "budget_exhausted",
                    "model_stopped": stop_index is not None,
                    "score_logits_finite": score_finite[index],
                }
            batch_count += 1
    return output, {"protocol": protocol, "batches": batch_count, "elapsed_seconds": time.time() - started}


def full_recompute_agreement(
    model: Any,
    device: torch.device,
    rows: list[dict[str, Any]],
    token_rows: list[dict[str, Any]],
    natural: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    material = {row["row_id"]: row for row in rows}
    token_map = {row["row_id"]: row for row in token_rows}
    selected_ids = [
        row["row_id"]
        for row in rows
        if row["partition"] in SEALED_PARTITIONS and row["state"] == 0
    ]
    events: list[dict[str, Any]] = []
    for row_id in selected_ids:
        prefix = list(token_map[row_id]["input_ids"]["natural_sentence"])
        for step, expected_token in enumerate(natural[row_id]["generated_token_ids"]):
            events.append(
                {
                    "row_id": row_id,
                    "step": step,
                    "input_ids": prefix + natural[row_id]["generated_token_ids"][:step],
                    "length": len(prefix) + step,
                    "expected": int(expected_token),
                }
            )
    results = {row_id: {"step_count": 0, "match_count": 0, "all_vocab_logits_finite": True} for row_id in selected_ids}
    started = time.time()
    batches = 0
    with torch.inference_mode():
        for batch in homogeneous_batches(events, "length", RECOMPUTE_BATCH_SIZE):
            input_ids = torch.tensor([event["input_ids"] for event in batch], dtype=torch.long, device=device)
            logits = model(input_ids=input_ids, use_cache=False).logits[:, -1].float()
            top = logits.argmax(dim=-1).detach().cpu().tolist()
            finite = torch.isfinite(logits).all(dim=-1).detach().cpu().tolist()
            for index, event in enumerate(batch):
                item = results[event["row_id"]]
                item["step_count"] += 1
                item["match_count"] += int(int(top[index]) == event["expected"])
                item["all_vocab_logits_finite"] = item["all_vocab_logits_finite"] and bool(finite[index])
            batches += 1
    for row_id, item in results.items():
        item["agreement"] = item["match_count"] / item["step_count"] if item["step_count"] else None
        item["world_id"] = material[row_id]["world_id"]
    return results, {"events": len(events), "batches": batches, "elapsed_seconds": time.time() - started}


def run_qwen3() -> None:
    if RAW_PATH.exists() or RUN_SUMMARY_PATH.exists():
        raise RuntimeError("Qwen3 output already exists; max_runs=1 forbids rerun")
    local_contract, rows, token_rows, plan = verify_frozen_inputs()
    mutate_contract_status("running", "ready", "run_ready")
    upsert_run("running", "ready", started_at=utc_now())
    run_researchctl("validate")
    started = time.time()
    model, tokenizer, device, placement = load_fp16("qwen3")
    precision = quantization_audit(model)
    if device.type != "cuda" or precision["has_quantized_modules"] or set(precision["parameter_dtypes"]) != {"float16"}:
        release_fp16(model)
        raise RuntimeError("Qwen3 numerical contract failed")
    try:
        candidate, candidate_runtime = score_candidates(model, device, token_rows)
        generations: dict[str, dict[str, Any]] = {}
        generation_runtimes: dict[str, Any] = {}
        for protocol in GENERATION_PROTOCOLS:
            generations[protocol], generation_runtimes[protocol] = generate_protocol(
                model, tokenizer, device, token_rows, protocol
            )
        recompute, recompute_runtime = full_recompute_agreement(
            model, device, rows, token_rows, generations["natural_sentence"]
        )
        raw: list[dict[str, Any]] = []
        for row in rows:
            row_id = row["row_id"]
            candidate_value = candidate[row_id]
            generation_values: dict[str, Any] = {}
            for protocol in GENERATION_PROTOCOLS:
                value = dict(generations[protocol][row_id])
                value["parse"] = parse_generation(
                    value["text"], row["candidate_order"], protocol, row["expected_outputs"][protocol]
                )
                generation_values[protocol] = value
            result: dict[str, Any] = {
                "phase": PHASE,
                "schema_version": "phase1246.qwen3.behavior.row.v1",
                "scientific_contract_digest": local_contract["scientific_contract_digest"],
                "row_id": row_id,
                "partition": row["partition"],
                "world_id": row["world_id"],
                "state": row["state"],
                "collision_group": row["collision_group"],
                "template_index": row["template_index"],
                "name_world": row["name_world"],
                "gold": row["gold"],
                "candidate": candidate_value,
                "candidate_correct": candidate_value["prediction"] == row["gold"],
                "generations": generation_values,
                "cache_full_recompute": recompute.get(row_id),
            }
            result["behavior_row_digest"] = digest(result)
            raw.append(result)
        write_jsonl(RAW_PATH, raw)
        elapsed = time.time() - started
        summary: dict[str, Any] = {
            "phase": PHASE,
            "schema_version": "phase1246.qwen3.run_summary.v1",
            "created_at_utc": utc_now(),
            "model": "Qwen3-4B",
            "row_count": len(raw),
            "world_count": WORLD_COUNT,
            "raw_digest": digest(raw),
            "scientific_contract_digest": local_contract["scientific_contract_digest"],
            "plan_digest": plan["plan_digest"],
            "precision_audit": precision,
            "placement": placement,
            "candidate_runtime": candidate_runtime,
            "generation_runtimes": generation_runtimes,
            "recompute_runtime": recompute_runtime,
            "elapsed_seconds": elapsed,
            "gpu_hours": elapsed / 3600.0,
            "gpu_budget_hours": 2.0,
            "gpu_budget_respected": elapsed / 3600.0 <= 2.0,
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "hidden_states_saved": False,
            "attentions_saved": False,
            "interventions_performed": False,
        }
        summary["summary_digest"] = digest(summary)
        write_json(RUN_SUMMARY_PATH, summary)
    finally:
        release_fp16(model)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    mutate_contract_status("auditing", "running", "run_ready")
    upsert_run(
        "auditing",
        "running",
        ended_at=utc_now(),
        finished_at=utc_now(),
        gpu_hours=read_json(RUN_SUMMARY_PATH)["gpu_hours"],
    )
    run_researchctl("validate")
    print(canonical_json({
        "status": "phase1246_qwen3_complete",
        "rows": len(read_jsonl(RAW_PATH)),
        "summary_digest": read_json(RUN_SUMMARY_PATH)["summary_digest"],
    }))


def mean_bool(values: Iterable[bool]) -> float:
    items = [bool(value) for value in values]
    return sum(items) / len(items) if items else float("nan")


def model_content_correct(row: dict[str, Any], protocol: str) -> bool:
    if protocol == "candidate":
        return bool(row["candidate_correct"])
    return bool(row["generations"][protocol]["parse"]["content_correct"])


def adjudicate() -> None:
    local_contract, material, _token_rows, _plan = verify_frozen_inputs()
    raw = read_jsonl(RAW_PATH)
    if len(raw) != len(material):
        raise RuntimeError("raw row count mismatch")
    raw_by_id = {row["row_id"]: row for row in raw}
    material_by_id = {row["row_id"]: row for row in material}
    required = ("discovery", "selection", "confirmation")
    partition_content: dict[str, float] = {}
    partition_protocol_content: dict[str, Any] = {}
    worst_template_name: dict[str, float] = {}
    collision_complete: dict[str, float] = {}
    strongest_alt: dict[str, float] = {}
    program = read_json(PROGRAM_PATH)
    for partition in required:
        rows = [row for row in raw if row["partition"] == partition]
        partition_content[partition] = mean_bool(
            model_content_correct(row, protocol) for row in rows for protocol in PROTOCOLS
        )
        partition_protocol_content[partition] = {
            protocol: mean_bool(model_content_correct(row, protocol) for row in rows) for protocol in PROTOCOLS
        }
        cells = []
        for template in range(len(RECORD_TEMPLATES)):
            for name_world in range(len(NAME_PREFIXES)):
                cell = [row for row in rows if row["template_index"] == template and row["name_world"] == name_world]
                if cell:
                    cells.append(mean_bool(model_content_correct(row, protocol) for row in cell for protocol in PROTOCOLS))
        worst_template_name[partition] = min(cells)
        strongest_alt[partition] = program["partitions"][partition]["strongest_registered_ceiling"]
    for group in COLLISION_GROUPS:
        worlds = sorted({row["world_id"] for row in raw if row["partition"] in required and row["collision_group"] == group})
        complete = []
        for world_id in worlds:
            pair = [row for row in raw if row["world_id"] == world_id]
            complete.append(all(model_content_correct(row, protocol) for row in pair for protocol in PROTOCOLS))
        collision_complete[group] = mean_bool(complete)
    content_checks = {
        "partition_accuracy": min(partition_content.values()) >= THRESHOLDS["content_partition_accuracy"],
        "worst_template_name": min(worst_template_name.values()) >= THRESHOLDS["content_worst_template_name_accuracy"],
        "collision_complete": min(collision_complete.values()) >= THRESHOLDS["collision_complete_rate"],
        "alternative_advantage": min(
            partition_content[partition] - strongest_alt[partition] for partition in required
        ) >= THRESHOLDS["alternative_program_advantage"],
        "finite_and_unique_candidate": all(
            row["candidate"]["all_vocab_logits_finite"] and not row["candidate"]["tie"] for row in raw
        ),
    }
    format_protocols = ("bare_short", "prompted_short", "fixed_sentence")
    format_partition = {}
    format_worst_template = {}
    for partition in required:
        rows = [row for row in raw if row["partition"] == partition]
        format_partition[partition] = mean_bool(
            row["generations"][protocol]["parse"]["exact"] for row in rows for protocol in format_protocols
        )
        cells = []
        for template in range(len(RECORD_TEMPLATES)):
            cell = [row for row in rows if row["template_index"] == template]
            if cell:
                cells.append(mean_bool(
                    row["generations"][protocol]["parse"]["exact"] for row in cell for protocol in format_protocols
                ))
        format_worst_template[partition] = min(cells)
    fixtures = read_jsonl(FIXTURE_PATH)
    fixture_checks = []
    for fixture in fixtures:
        parsed = parse_generation(
            fixture["text"], fixture["candidates"], fixture["protocol"], fixture["expected"]
        )
        fixture_checks.append(
            parsed["prediction"] == fixture["expected_prediction"]
            and parsed["content_correct"] == fixture["expected_content_correct"]
            and parsed["format_valid"] == fixture["expected_format_valid"]
        )
    format_checks = {
        "partition_accuracy": min(format_partition.values()) >= THRESHOLDS["format_partition_accuracy"],
        "worst_template": min(format_worst_template.values()) >= THRESHOLDS["format_worst_template_accuracy"],
        "four_cell_and_adversarial_evaluator": all(fixture_checks),
    }
    natural_partition = {}
    natural_worst_template = {}
    for partition in ("selection", "confirmation"):
        rows = [row for row in raw if row["partition"] == partition]
        natural_partition[partition] = mean_bool(
            row["generations"]["natural_sentence"]["parse"]["content_correct"] for row in rows
        )
        natural_worst_template[partition] = min(
            mean_bool(
                row["generations"]["natural_sentence"]["parse"]["content_correct"]
                for row in rows if row["template_index"] == template
            )
            for template in range(len(RECORD_TEMPLATES))
        )
    natural_checks = {
        "partition_accuracy": min(natural_partition.values()) >= THRESHOLDS["natural_partition_accuracy"],
        "worst_template": min(natural_worst_template.values()) >= THRESHOLDS["natural_worst_template_accuracy"],
        "multi_reference_evaluator": all(fixture_checks),
    }
    cache_rows = [
        row for row in raw if row["partition"] in SEALED_PARTITIONS and row["state"] == 0
    ]
    cache_agreement = sum(row["cache_full_recompute"]["match_count"] for row in cache_rows) / sum(
        row["cache_full_recompute"]["step_count"] for row in cache_rows
    )
    stop_rows = [row for row in raw if row["partition"] in SEALED_PARTITIONS]
    stop_rate = mean_bool(row["generations"]["natural_sentence"]["model_stopped"] for row in stop_rows)
    generation_finite = all(
        row["generations"][protocol]["score_logits_finite"] for row in raw for protocol in GENERATION_PROTOCOLS
    )
    cache_checks = {
        "cache_top1_agreement": cache_agreement == THRESHOLDS["cache_top1_agreement"],
        "correct_stop_rate": stop_rate >= THRESHOLDS["correct_stop_rate"],
        "no_external_truncation_counted": all(
            row["generations"]["natural_sentence"]["stop_source"] == "model_eos"
            for row in stop_rows if row["generations"]["natural_sentence"]["model_stopped"]
        ),
        "generation_logits_finite": generation_finite,
    }
    gates = {
        "G-CONTENT": all(content_checks.values()),
        "G-FORMAT": all(format_checks.values()),
        "G-NATURAL": all(natural_checks.values()),
        "G-STOP-CACHE": all(cache_checks.values()),
    }
    if all(gates.values()):
        verdict = "typed_behavior_qualified"
    elif any(gates.values()):
        verdict = "partial_typed_qualification"
    else:
        verdict = "bounded_rejected"
    adjudication: dict[str, Any] = {
        "phase": PHASE,
        "schema_version": "phase1246.typed_adjudication.v1",
        "created_at_utc": utc_now(),
        "scientific_contract_digest": local_contract["scientific_contract_digest"],
        "content": {
            "partition_accuracy": partition_content,
            "partition_protocol_accuracy": partition_protocol_content,
            "worst_template_name_accuracy": worst_template_name,
            "collision_complete_rate": collision_complete,
            "strongest_registered_alternative": strongest_alt,
            "checks": content_checks,
        },
        "format": {
            "partition_exact_accuracy": format_partition,
            "worst_template_exact_accuracy": format_worst_template,
            "fixture_count": len(fixtures),
            "checks": format_checks,
        },
        "natural": {
            "partition_content_accuracy": natural_partition,
            "worst_template_content_accuracy": natural_worst_template,
            "checks": natural_checks,
        },
        "stop_cache": {
            "trajectory_count": len(cache_rows),
            "cache_top1_agreement": cache_agreement,
            "correct_stop_rate": stop_rate,
            "checks": cache_checks,
        },
        "typed_gates": gates,
        "verdict": verdict,
        "authorization": {
            "content_low_cost_observation": gates["G-CONTENT"],
            "format_low_cost_observation": gates["G-FORMAT"],
            "natural_autoregressive_observation": gates["G-NATURAL"],
            "stop_cache_next_layer_test": gates["G-STOP-CACHE"],
            "hidden_scan_in_this_phase": False,
            "cross_model": False,
            "automatic_TB03": False,
        },
    }
    adjudication["adjudication_digest"] = digest(adjudication)
    write_json(ADJUDICATION_PATH, adjudication)
    summary = read_json(RUN_SUMMARY_PATH)
    camera = read_json(CAMERA_PATH)
    final: dict[str, Any] = {
        "phase": PHASE,
        "schema_version": "phase1246.final.v1",
        "created_at_utc": utc_now(),
        "contract_id": CONTRACT_ID,
        "run_id": RUN_ID,
        "model": "Qwen3-4B FP16 CUDA",
        "world_count": WORLD_COUNT,
        "behavior_row_count": len(raw),
        "known_truth_camera_gate": camera["camera_gate"],
        "known_truth_camera_digest": camera["camera_digest"],
        "typed_gates": gates,
        "verdict": verdict,
        "adjudication_digest": adjudication["adjudication_digest"],
        "run_summary_digest": summary["summary_digest"],
        "claim": "C001-WP01 adjudicates four behavior constructs independently on a fresh answer-payload-free Qwen3 task.",
        "non_claims": [
            "No hidden state, attention head, MLP, neuron or causal mechanism was measured.",
            "Known-truth camera success is instrument calibration, not evidence that Qwen3 uses those mechanism families.",
            "A failed construct bounds this interface and does not negate the underlying object-marker operation.",
            "A passed construct authorizes only a separately preregistered low-cost observation stage.",
        ],
        "auto_continue": False,
        "next_stage": "C001-WP02 only for independently passed constructs, under a new contract and no automatic launch.",
    }
    final["final_digest"] = digest(final)
    write_json(FINAL_PATH, final)
    print(canonical_json({"status": "phase1246_adjudicated_locally", "verdict": verdict, "gates": gates, "final_digest": final["final_digest"]}))


def artifact_record(artifact_id: str, kind: str, path: Path) -> dict[str, Any]:
    return {
        "id": artifact_id,
        "run_id": RUN_ID,
        "contract_id": CONTRACT_ID,
        "status": "verified",
        "kind": kind,
        "path": workspace_relative(path),
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
        "created_at": utc_now(),
        "preserve": True,
    }


def register_final() -> None:
    audit = read_json(FINAL_AUDIT_PATH)
    if audit.get("all_checks_passed") is not True:
        raise RuntimeError("independent final audit missing or failed")
    final = read_json(FINAL_PATH)
    adjudication = read_json(ADJUDICATION_PATH)
    artifact_specs = [
        ("ART-C001-WP01-MATERIAL", "material", MATERIAL_PATH),
        ("ART-C001-WP01-TOKEN-MANIFEST", "token_manifest", TOKEN_MANIFEST_PATH),
        ("ART-C001-WP01-CAMERA", "known_truth_calibration", CAMERA_PATH),
        ("ART-C001-WP01-RAW", "raw_result", RAW_PATH),
        ("ART-C001-WP01-SUMMARY", "run_summary", RUN_SUMMARY_PATH),
        ("ART-C001-WP01-FINAL", "adjudication", FINAL_PATH),
        ("ART-C001-WP01-AUDIT", "independent_audit", FINAL_AUDIT_PATH),
    ]
    artifacts = read_json(ARTIFACTS_PATH)
    existing_ids = {row["id"] for row in artifacts}
    for artifact_id, kind, path in artifact_specs:
        record = artifact_record(artifact_id, kind, path)
        if artifact_id in existing_ids:
            artifacts = [record if row["id"] == artifact_id else row for row in artifacts]
        else:
            artifacts.append(record)
    write_json(ARTIFACTS_PATH, artifacts)
    artifact_ids = [item[0] for item in artifact_specs]
    mutate_contract_status("adjudicated", "auditing", "adjudicated")
    upsert_run(
        "adjudicated",
        "auditing",
        artifact_ids=artifact_ids,
        verdict=final["verdict"],
        gpu_hours=read_json(RUN_SUMMARY_PATH)["gpu_hours"],
        ended_at=read_json(RUN_SUMMARY_PATH)["created_at_utc"],
        finished_at=read_json(RUN_SUMMARY_PATH)["created_at_utc"],
    )
    campaigns_path = OS_ROOT / "registry/campaigns.json"
    campaigns = read_json(campaigns_path)
    campaign = next(row for row in campaigns if row["id"] == "C001")
    wp01 = next(row for row in campaign["stages"] if row["id"] == "WP01")
    wp02 = next(row for row in campaign["stages"] if row["id"] == "WP02")
    wp01["status"] = "completed"
    if adjudication["typed_gates"]["G-CONTENT"]:
        campaign["status"] = "active"
        wp02["status"] = "active"
    else:
        campaign["status"] = "blocked"
        wp02["status"] = "blocked"
    write_json(campaigns_path, campaigns)
    tests_path = OS_ROOT / "registry/tests.json"
    tests = read_json(tests_path)
    for row in tests:
        if row["id"] == "TB01":
            row["status"] = "passed"
        if row["id"] == "TB02":
            row["status"] = "passed" if adjudication["typed_gates"]["G-CONTENT"] else "failed"
    write_json(tests_path, tests)
    corrections_path = OS_ROOT / "registry/corrections.json"
    corrections = read_json(corrections_path)
    additions = [
        {
            "id": "COR-WP01-001",
            "date": "2026-08-12",
            "status": "active",
            "target_type": "preauthorization_material",
            "target_ids": [CONTRACT_ID],
            "problem": "The first no-model material draft covered only 10 of 12 gold-slot by name-world cells.",
            "correction": "No weights were loaded; the draft was quarantined and all 512 worlds were regenerated with complete blockwise name-world rotation before the passing 35/35 preaudit.",
            "preserves_original_claim": True,
        },
        {
            "id": "COR-WP01-002",
            "date": "2026-08-12",
            "status": "active",
            "target_type": "run_authorization_wrapper",
            "target_ids": [CONTRACT_ID, RUN_ID],
            "problem": "The wrapper passed a contract ID where researchctl expected a manifest path and decoded UTF-8 output using the host code page.",
            "correction": "Before loading weights, the wrapper was fixed to pass the frozen manifest path with explicit UTF-8 decoding; validation and manifest verification then passed.",
            "preserves_original_claim": True,
        },
        {
            "id": "COR-WP01-003",
            "date": "2026-08-12",
            "status": "active",
            "target_type": "adjudicated_readiness_state",
            "target_ids": [CONTRACT_ID],
            "problem": "The schema allowed adjudicated readiness, but the validator requires contract readiness=run_ready whenever the index retains historical run_ready=true.",
            "correction": "The contract keeps run_ready as historical authorization while status and manifest readiness carry the terminal adjudicated state.",
            "preserves_original_claim": True,
        },
    ]
    existing_corrections = {row["id"] for row in corrections}
    corrections.extend(row for row in additions if row["id"] not in existing_corrections)
    write_json(corrections_path, corrections)
    run_researchctl("validate")
    run_researchctl("verify-manifest", CONTRACT_ID)
    print(canonical_json({
        "status": "phase1246_registered",
        "verdict": final["verdict"],
        "content_authorized_wp02": adjudication["typed_gates"]["G-CONTENT"],
        "evidence_ledger_pending_git_snapshot": True,
    }))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("prepare", "authorize", "run", "adjudicate", "register"), required=True)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare()
    elif args.mode == "authorize":
        authorize()
    elif args.mode == "run":
        run_qwen3()
    elif args.mode == "adjudicate":
        adjudicate()
    elif args.mode == "register":
        register_final()


if __name__ == "__main__":
    main()
