#!/usr/bin/env python3
"""Phase1236: global functional structure identification campaign.

This phase separates prompt interface (pi), decoder (rho), and evaluator
(kappa), then asks competing response-transition models to predict a sealed
partition.  Only a sealed predictive pass may authorize one Qwen3 residual
interchange experiment.  The object is a typed response law, not a hotspot.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import platform
import random
import re
import string
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
sys.path.insert(0, str(TEST_ROOT))

from model_utils import MODEL_CONFIGS, get_layers
from phase1023_fp16_utils import load_fp16, quantization_audit, release_fp16


PHASE = 1236
SCRIPT = Path(__file__).resolve()
AUDIT_SCRIPT = TEST_ROOT / "phase1236_global_functional_structure_identification_audit.py"
UPSTREAM_ROOT = TEST_ROOT / "result/phase1235_qwen3_typed_generation_compiler_boundary"
UPSTREAM_FINAL = UPSTREAM_ROOT / "analysis/final.json"
UPSTREAM_AUDIT = UPSTREAM_ROOT / "audit/independent_final_audit.json"
UPSTREAM_MATERIAL = UPSTREAM_ROOT / "material/orthogonal_readout_worlds.jsonl"
P1234_MATERIAL = TEST_ROOT / "result/phase1234_qwen3_k199_registry_sealed_confirmation/material/sealed_query_object_worlds.jsonl"
EXPECTED_UPSTREAM_FINAL = "300322d5392a0e94d80881284615970e5d04b03d6feefb4e738e9764359683d4"
EXPECTED_UPSTREAM_AUDIT = "df06cadf8bfa4ffb15e496c36bec4da82395c566605829af08cdbb1072f4d6c9"

OUT_ROOT = TEST_ROOT / "result/phase1236_global_functional_structure_identification"
CONTRACT_PATH = OUT_ROOT / "protocol/preregistration.json"
MATERIAL_PATH = OUT_ROOT / "material/frozen_response_worlds.jsonl"
PARSER_FIXTURES_PATH = OUT_ROOT / "material/evaluator_adversarial_fixtures.jsonl"
PREAUDIT_PATH = OUT_ROOT / "audit/independent_preaudit.json"
BEHAVIOR_ADJUDICATION_PATH = OUT_ROOT / "analysis/behavior_adjudication.json"
CAPTURE_ARRAY_PATH = OUT_ROOT / "hidden/qwen3/response_tensor.npz"
CAPTURE_META_PATH = OUT_ROOT / "hidden/qwen3/response_tensor_metadata.json"
STRUCTURE_PATH = OUT_ROOT / "analysis/structure_competition.json"
STRUCTURE_AUDIT_PATH = OUT_ROOT / "audit/independent_structure_audit.json"
CAUSAL_PATH = OUT_ROOT / "causal/qwen3/cross_protocol_interchange.json"
FINAL_PATH = OUT_ROOT / "analysis/final.json"
FINAL_AUDIT_PATH = OUT_ROOT / "audit/independent_final_audit.json"

MODELS = ("qwen3", "glm4", "deepseek7b")
PROTOCOLS = ("bare", "sentence", "natural")
PARTITIONS = ("discovery", "model_selection", "sealed")
TOPOLOGIES = ("same_left", "same_right")
WORLD_COUNT = 48
WORLDS_PER_PARTITION = 16
OBJECT_COUNT = 4
CANDIDATE_COUNT = 5
QUERY_COUNT = 4
BINDING_STATES = (0, 1)
EXPECTED_ROWS = WORLD_COUNT * QUERY_COUNT * len(BINDING_STATES) * len(PROTOCOLS)
EXPECTED_BASE_PAIRS = WORLD_COUNT * QUERY_COUNT
PROJECTION_DIM = 64
PROJECTION_SEED = 12360017
GENERATION_BUDGET = 32
TIE_TOLERANCE = 1e-7
EPSILON = 1e-8
SYSTEM_PROMPT = (
    "Use only the supplied registry records. Answer immediately without reasoning "
    "and follow the requested output contract."
)

FIRST_NAMES = (
    "Adrian Alina Amara Ansel Briar Celine Conrad Dahlia Dorian Elara Elias Emilia "
    "Flora Gideon Helena Imani Isadora Julian Leona Lucian Marina Nadia Noemi Orion "
    "Petra Rafael Sabine Selene Silas Talia Tobias Vera Vivian Xavier Yasmin Zora"
).split()
SURNAMES = (
    "Abbott Bellamy Calder Dorsey Ellison Farrow Gresham Hollis Ingram Jarvis Keaton "
    "Langley Mercer Nolan Osborn Prescott Quincy Ramsey Sawyer Thayer Ulrich Varden "
    "Winslow York Arden Blaine Corwin Devlin Everly Fenwick Garland Hadley Iverson "
    "Jensen Kendall Lowell Monroe Northcott Oakley Palmer Reeves Sutton Travers"
).split()
LABEL_LEFT = (
    "bright calm clear cool deep dry fair faint fresh gentle glossy grand light lucid "
    "mellow mild neat pale plain pure quiet rapid rich round sharp sleek soft solid steady "
    "still strong subtle swift warm wide vivid bold crisp dense even firm level matte muted"
).split()
LABEL_RIGHT = (
    "arch beacon bloom bridge cedar circle crest delta ember field flame gate grove harbor "
    "isle jewel key lantern meadow moon oak path peak pearl pine plume quartz reed ridge ring "
    "river seal shore spark spire star stone stream summit torch vale wave wheel wing yard"
).split()
ZONE_WORDS = ("central", "eastern", "northern", "southern")
STATUS_WORDS = ("ready", "stored", "checked", "sealed")
TEXTURE_WORDS = ("linen", "marble", "oak", "glass")
RECORD_TEMPLATES = (
    "Registry record for {object}: marker = {marker}; zone = {zone}; texture = {texture}; status = {status}.",
    "The registry lists {object} with marker {marker}, zone {zone}, texture {texture}, and status {status}.",
    "For {object}, the dossier gives {marker} as marker, {zone} as zone, texture {texture}, and status {status}.",
    "Entry for {object}: its marker is {marker}, its zone is {zone}, its texture is {texture}, and its status is {status}.",
)
QUERY_TEMPLATES = (
    "Which marker is assigned to {object}?",
    "According to the registry, what marker belongs to {object}?",
    "Read the marker recorded for {object}.",
    "What is the registered marker for {object}?",
)

THRESHOLDS = {
    "finite_rate": 0.99,
    "content_score_worst_partition_protocol": 0.85,
    "content_score_worst_protocol": 0.90,
    "contract_score_worst_partition_protocol": 0.80,
    "generation_content_worst_partition_protocol": 0.75,
    "generation_content_worst_protocol": 0.80,
    "format_valid_worst_bare_sentence": 0.75,
    "structure_improvement_over_mean": 0.10,
    "structure_shuffled_advantage": 0.10,
    "structure_median_cosine": 0.20,
    "structure_positive_cosine_fraction": 0.65,
    "causal_oracle_positive_fraction": 0.70,
    "causal_mapped_positive_fraction": 0.60,
    "causal_mapped_state1_gain": 0.15,
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


def strip_digest(value: dict[str, Any], key: str) -> dict[str, Any]:
    return {name: item for name, item in value.items() if name != key}


def model_manifest_path(model_name: str) -> Path:
    return OUT_ROOT / f"protocol/{model_name}_manifest.jsonl"


def behavior_raw_path(model_name: str) -> Path:
    return OUT_ROOT / f"behavior/{model_name}/raw_behavior.jsonl"


def behavior_summary_path(model_name: str) -> Path:
    return OUT_ROOT / f"behavior/{model_name}/run_summary.json"


def behavior_audit_path(model_name: str) -> Path:
    return OUT_ROOT / f"audit/{model_name}_behavior_audit.json"


def verify_upstream() -> tuple[dict[str, Any], dict[str, Any]]:
    final = read_json(UPSTREAM_FINAL)
    audit = read_json(UPSTREAM_AUDIT)
    if final.get("final_digest") != EXPECTED_UPSTREAM_FINAL:
        raise RuntimeError("Phase1235 final digest mismatch")
    if audit.get("audit_digest") != EXPECTED_UPSTREAM_AUDIT or audit.get("all_checks_passed") is not True:
        raise RuntimeError("Phase1235 audit mismatch")
    if final.get("authorization", {}).get("future_response_phase") is not False:
        raise RuntimeError("Phase1235 authorization boundary drift")
    return final, audit


def render_chat(tokenizer: Any, prompt: str, model_name: str) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    if model_name == "qwen3":
        kwargs["enable_thinking"] = False
    try:
        return str(tokenizer.apply_chat_template(messages, **kwargs))
    except (TypeError, ValueError):
        kwargs.pop("enable_thinking", None)
        try:
            return str(tokenizer.apply_chat_template(messages, **kwargs))
        except (TypeError, ValueError):
            return f"System: {SYSTEM_PROMPT}\nUser: {prompt}\nAssistant:"


def continuation_suffix(tokenizer: Any, rendered: str, continuation: str) -> tuple[list[int], list[int]]:
    prefix = [int(value) for value in tokenizer.encode(rendered, add_special_tokens=False)]
    appended = [int(value) for value in tokenizer.encode(rendered + continuation, add_special_tokens=False)]
    if appended[: len(prefix)] != prefix or len(appended) == len(prefix):
        raise RuntimeError("continuation does not preserve native prefix")
    return prefix, appended[len(prefix) :]


def normalized(text: str) -> str:
    value = re.sub(r"\s+", " ", text.strip().lower())
    return value.strip(string.whitespace)


def phrase_spans(text: str, phrase: str) -> list[tuple[int, int]]:
    return [match.span() for match in re.finditer(rf"(?<!\w){re.escape(normalized(phrase))}(?!\w)", normalized(text))]


def parse_output(text: str, candidates: list[str], expected_exact: str, protocol: str) -> dict[str, Any]:
    value = normalized(text)
    mentions: list[tuple[str, int, int]] = []
    for candidate in candidates:
        mentions.extend((candidate, start, end) for start, end in phrase_spans(value, candidate))
    mentions.sort(key=lambda row: row[1])
    unique = sorted({row[0] for row in mentions})
    rejected_reason = None
    prediction = None
    if len(unique) != 1:
        rejected_reason = "none_or_multiple_candidates"
    else:
        candidate, start, _end = mentions[0]
        before = value[max(0, start - 32) : start]
        after = value[_end : min(len(value), _end + 32)]
        if re.search(r"\b(?:not|incorrect|wrong)\b|isn't|rather than", before):
            rejected_reason = "negated_candidate"
        elif re.search(r"\b(?:sorry|correction|instead)\b", after):
            rejected_reason = "self_correction"
        else:
            prediction = candidate
    exact = value.strip(string.whitespace + string.punctuation) == normalized(expected_exact).strip(
        string.whitespace + string.punctuation
    )
    escaped = "|".join(re.escape(normalized(candidate)) for candidate in candidates)
    if protocol == "bare":
        format_valid = re.fullmatch(rf"(?:{escaped})[.!]?", value) is not None
    elif protocol == "sentence":
        format_valid = re.fullmatch(rf"the recorded marker is (?:{escaped})\.", value) is not None
    elif protocol == "natural":
        format_valid = bool(prediction is not None and len(value.split()) <= 20 and value.endswith((".", "!", "?")))
    else:
        raise ValueError(protocol)
    return {
        "normalized_text": value,
        "mentioned_candidates": unique,
        "prediction": prediction,
        "exact": exact,
        "format_valid": format_valid,
        "rejected_reason": rejected_reason,
    }


def parser_fixtures() -> list[dict[str, Any]]:
    candidates = ["clear river", "soft stone", "bright arch", "calm field", "deep grove"]
    fixtures = [
        ("correct_content_correct_format", "clear river", "clear river", "clear river", True, True),
        ("correct_content_wrong_format", "The answer is clear river.", "clear river", "clear river", True, False),
        ("wrong_content_correct_format", "soft stone", "clear river", "soft stone", False, True),
        ("wrong_content_wrong_format", "The answer is soft stone.", "clear river", "soft stone", False, False),
        ("negation", "The marker is not clear river.", "clear river", None, False, False),
        ("quotation_negation", 'The record does not say "clear river".', "clear river", None, False, False),
        ("self_correction", "clear river, sorry, soft stone", "clear river", None, False, False),
        ("multiple", "Either clear river or soft stone.", "clear river", None, False, False),
    ]
    rows = []
    for name, text, gold, expected_prediction, content_expected, format_expected in fixtures:
        parsed = parse_output(text, candidates, gold, "bare")
        row = {
            "fixture": name,
            "text": text,
            "gold": gold,
            "candidates": candidates,
            "expected_prediction": expected_prediction,
            "expected_content": content_expected,
            "expected_format": format_expected,
            "observed": parsed,
        }
        row["pass"] = bool(
            parsed["prediction"] == expected_prediction
            and (parsed["prediction"] == gold) == content_expected
            and parsed["format_valid"] == format_expected
        )
        row["row_digest"] = digest(row)
        rows.append(row)
    return rows


def prior_terms() -> set[str]:
    terms: set[str] = set()
    for path in (P1234_MATERIAL, UPSTREAM_MATERIAL):
        if not path.exists():
            continue
        for row in read_jsonl(path):
            for key in ("objects", "candidates", "entities", "values"):
                value = row.get(key, [])
                if isinstance(value, list):
                    terms.update(str(item) for item in value)
            for key in ("query_object", "gold", "unused_value"):
                if row.get(key):
                    terms.add(str(row[key]))
    return terms


def direct_lengths(tokenizers: dict[str, Any], text: str) -> tuple[int, ...]:
    return tuple(len(tokenizers[name].encode(" " + text, add_special_tokens=False)) for name in MODELS)


def build_object_groups(tokenizers: dict[str, Any], forbidden: set[str]) -> list[list[str]]:
    buckets: dict[tuple[int, ...], list[str]] = defaultdict(list)
    for first in FIRST_NAMES:
        for surname in SURNAMES:
            value = f"{first} {surname}"
            if value not in forbidden:
                buckets[direct_lengths(tokenizers, value)].append(value)
    groups: list[list[str]] = []
    for key in sorted(buckets, key=lambda item: (-len(buckets[item]), item)):
        values = buckets[key]
        for start in range(0, len(values) - OBJECT_COUNT + 1, OBJECT_COUNT):
            groups.append(values[start : start + OBJECT_COUNT])
            if len(groups) >= WORLD_COUNT:
                return groups
    raise RuntimeError("insufficient fresh matched object groups")


def continuation_lengths_ok(tokenizers: dict[str, Any], candidates: list[str]) -> bool:
    dummy = "Registry record.\nWhich marker is assigned?\nAnswer:"
    for model_name, tokenizer in tokenizers.items():
        rendered = render_chat(tokenizer, dummy, model_name)
        for protocol in PROTOCOLS:
            content = []
            contract = []
            for candidate in candidates:
                _, ids = continuation_suffix(tokenizer, rendered, " " + candidate)
                content.append(len(ids))
                _, ids = continuation_suffix(tokenizer, rendered, contract_continuation(protocol, candidate))
                contract.append(len(ids))
            if len(set(content)) != 1 or len(set(contract)) != 1:
                return False
    return True


def build_candidate_groups(tokenizers: dict[str, Any], forbidden: set[str]) -> list[tuple[str, list[str]]]:
    used: set[str] = set()
    result: list[tuple[str, list[str]]] = []
    target_each = WORLD_COUNT // 2
    for topology in TOPOLOGIES:
        if topology == "same_left":
            outer, inner = LABEL_LEFT, LABEL_RIGHT
            make = lambda a, b: f"{a} {b}"
        else:
            outer, inner = LABEL_RIGHT, LABEL_LEFT
            make = lambda a, b: f"{b} {a}"
        count = 0
        for anchor in outer:
            available = [make(anchor, word) for word in inner]
            available = [value for value in available if value not in forbidden and value not in used]
            for offset in range(0, len(available) - CANDIDATE_COUNT + 1, CANDIDATE_COUNT):
                group = available[offset : offset + CANDIDATE_COUNT]
                if continuation_lengths_ok(tokenizers, group):
                    result.append((topology, group))
                    used.update(group)
                    count += 1
                    break
            if count >= target_each:
                break
        if count < target_each:
            raise RuntimeError(f"insufficient {topology} candidate groups: {count}")
    return result


def contract_continuation(protocol: str, candidate: str) -> str:
    if protocol == "bare":
        return " " + candidate
    if protocol == "sentence":
        return f" The recorded marker is {candidate}."
    if protocol == "natural":
        return f" The marker for the queried object is {candidate}."
    raise ValueError(protocol)


def expected_exact(protocol: str, candidate: str) -> str:
    return contract_continuation(protocol, candidate).strip()


def protocol_instruction(protocol: str) -> str:
    return {
        "bare": "Return only the exact two-word marker label.",
        "sentence": "Use exactly this sentence form: The recorded marker is [two-word marker].",
        "natural": "Answer the question directly in one concise sentence.",
    }[protocol]


def rotate_assignments(candidates: list[str], state: int) -> list[str]:
    assigned = candidates[:OBJECT_COUNT]
    if state == 0:
        return assigned
    return assigned[1:] + assigned[:1]


def build_material(tokenizers: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    forbidden = prior_terms()
    objects = build_object_groups(tokenizers, forbidden)
    candidate_groups = build_candidate_groups(tokenizers, forbidden)
    rows: list[dict[str, Any]] = []
    for world_index in range(WORLD_COUNT):
        partition = PARTITIONS[world_index // WORLDS_PER_PARTITION]
        topology, candidates = candidate_groups[world_index]
        world_objects = objects[world_index]
        template_index = world_index % len(RECORD_TEMPLATES)
        record_order = [int((index + world_index) % OBJECT_COUNT) for index in range(OBJECT_COUNT)]
        world_id = f"p1236-{partition}-w{world_index:03d}"
        for state in BINDING_STATES:
            assigned = rotate_assignments(candidates, state)
            mapping = dict(zip(world_objects, assigned))
            records = []
            for position, object_index in enumerate(record_order):
                object_name = world_objects[object_index]
                records.append(
                    RECORD_TEMPLATES[template_index].format(
                        object=object_name,
                        marker=mapping[object_name],
                        zone=ZONE_WORDS[object_index],
                        texture=TEXTURE_WORDS[object_index],
                        status=STATUS_WORDS[object_index],
                    )
                )
            prefix = " ".join(records)
            for query_index, query_object in enumerate(world_objects):
                query = QUERY_TEMPLATES[template_index].format(object=query_object)
                gold = mapping[query_object]
                for protocol in PROTOCOLS:
                    prompt = f"{prefix}\n{query}\n{protocol_instruction(protocol)}\nAnswer:"
                    pair_id = f"{world_id}|q{query_index}|{protocol}"
                    row: dict[str, Any] = {
                        "phase": PHASE,
                        "schema_version": "phase1236.response_world.v1",
                        "item_id": f"{pair_id}|s{state}",
                        "pair_id": pair_id,
                        "base_pair_id": f"{world_id}|q{query_index}",
                        "world_id": world_id,
                        "world_index": world_index,
                        "partition": partition,
                        "topology": topology,
                        "template_index": template_index,
                        "protocol": protocol,
                        "binding_state": state,
                        "query_index": query_index,
                        "query_object": query_object,
                        "objects": world_objects,
                        "candidates": candidates,
                        "assigned_markers": mapping,
                        "gold": gold,
                        "gold_slot": candidates.index(gold),
                        "unused_candidate": candidates[-1],
                        "prompt": prompt,
                        "expected_exact": expected_exact(protocol, gold),
                    }
                    row["row_digest"] = digest(row)
                    rows.append(row)
    rows.sort(key=lambda row: (row["world_index"], row["query_index"], row["protocol"], row["binding_state"]))
    if len(rows) != EXPECTED_ROWS or len({row["item_id"] for row in rows}) != EXPECTED_ROWS:
        raise RuntimeError("material cardinality failure")
    audit = {
        "forbidden_term_count": len(forbidden),
        "fresh_object_count": len({value for row in rows for value in row["objects"]}),
        "fresh_candidate_count": len({value for row in rows for value in row["candidates"]}),
        "prior_overlap": sorted(
            {value for row in rows for value in row["objects"] + row["candidates"]}.intersection(forbidden)
        ),
        "partition_world_counts": dict(Counter(row["partition"] for row in rows[:: QUERY_COUNT * 2 * len(PROTOCOLS)])),
        "topology_world_counts": dict(Counter(row["topology"] for row in rows[:: QUERY_COUNT * 2 * len(PROTOCOLS)])),
    }
    return rows, audit


def locate_last_span(tokenizer: Any, input_ids: list[int], text: str) -> int:
    variants = [text, " " + text]
    matches: list[tuple[int, int]] = []
    for variant in variants:
        needle = [int(value) for value in tokenizer.encode(variant, add_special_tokens=False)]
        if not needle:
            continue
        for start in range(len(input_ids) - len(needle) + 1):
            if input_ids[start : start + len(needle)] == needle:
                matches.append((start, start + len(needle) - 1))
    if not matches:
        raise RuntimeError(f"cannot locate span {text!r}")
    return max(matches)[1]


def build_manifest(rows: list[dict[str, Any]], tokenizer: Any, model_name: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    mismatched_lengths = 0
    max_input = 0
    for execution_index, row in enumerate(rows):
        rendered = render_chat(tokenizer, row["prompt"], model_name)
        input_ids = [int(value) for value in tokenizer.encode(rendered, add_special_tokens=False)]
        content_ids: dict[str, list[int]] = {}
        contract_ids: dict[str, list[int]] = {}
        for candidate in row["candidates"]:
            prefix, suffix = continuation_suffix(tokenizer, rendered, " " + candidate)
            if prefix != input_ids:
                raise RuntimeError("content prefix drift")
            content_ids[candidate] = suffix
            prefix, suffix = continuation_suffix(
                tokenizer, rendered, contract_continuation(row["protocol"], candidate)
            )
            if prefix != input_ids:
                raise RuntimeError("contract prefix drift")
            contract_ids[candidate] = suffix
        mismatched_lengths += len({len(value) for value in content_ids.values()}) != 1
        mismatched_lengths += len({len(value) for value in contract_ids.values()}) != 1
        positions = None
        if model_name == "qwen3":
            positions = {
                "generation_boundary": len(input_ids) - 1,
                "query_object_end": locate_last_span(tokenizer, input_ids, row["query_object"]),
                "source_gold_end": locate_last_span(tokenizer, input_ids, row["gold"]),
            }
        case: dict[str, Any] = {
            "phase": PHASE,
            "schema_version": "phase1236.model_manifest.v1",
            "model": model_name,
            "execution_index": execution_index,
            "item_id": row["item_id"],
            "pair_id": row["pair_id"],
            "base_pair_id": row["base_pair_id"],
            "row_digest": row["row_digest"],
            "world_id": row["world_id"],
            "partition": row["partition"],
            "protocol": row["protocol"],
            "binding_state": row["binding_state"],
            "query_index": row["query_index"],
            "gold": row["gold"],
            "candidates": row["candidates"],
            "input_ids": input_ids,
            "input_token_count": len(input_ids),
            "content_candidate_token_ids": content_ids,
            "contract_candidate_token_ids": contract_ids,
            "positions": positions,
        }
        case["manifest_row_digest"] = digest(case)
        manifest.append(case)
        max_input = max(max_input, len(input_ids))
    summary = {
        "model": model_name,
        "row_count": len(manifest),
        "manifest_digest": digest(manifest),
        "tokenizer_class": type(tokenizer).__name__,
        "candidate_length_mismatch_count": mismatched_lengths,
        "maximum_input_length": max_input,
        "gate": len(manifest) == EXPECTED_ROWS and mismatched_lengths == 0 and max_input <= 384,
    }
    return manifest, summary


def load_tokenizers() -> dict[str, Any]:
    from transformers import AutoTokenizer

    values: dict[str, Any] = {}
    for model_name in MODELS:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_CONFIGS[model_name]["path"], trust_remote_code=True, local_files_only=True, use_fast=False
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        values[model_name] = tokenizer
    return values


def source_hashes() -> dict[str, str]:
    return {"main": file_sha256(SCRIPT), "audit": file_sha256(AUDIT_SCRIPT)}


def preregister() -> None:
    if CONTRACT_PATH.exists():
        raise RuntimeError("Phase1236 is already preregistered")
    final, audit = verify_upstream()
    tokenizers = load_tokenizers()
    rows, material_audit = build_material(tokenizers)
    fixtures = parser_fixtures()
    if not all(row["pass"] for row in fixtures):
        raise RuntimeError("adversarial evaluator selftest failed")
    manifest_summaries = {}
    for model_name in MODELS:
        manifest, summary = build_manifest(rows, tokenizers[model_name], model_name)
        if not summary["gate"]:
            raise RuntimeError(f"{model_name} manifest gate failed")
        write_jsonl(model_manifest_path(model_name), manifest)
        manifest_summaries[model_name] = summary
    contract: dict[str, Any] = {
        "phase": PHASE,
        "schema_version": "phase1236.global_functional_structure_identification.v1",
        "created_at_utc": utc_now(),
        "objective": (
            "Separate pi/rho/kappa, identify a compact cross-protocol future-response law by sealed prediction, "
            "then test the selected law with one frozen Qwen3 residual interchange."
        ),
        "source_hashes": source_hashes(),
        "upstream": {
            "phase1235_final_digest": final["final_digest"],
            "phase1235_audit_digest": audit["audit_digest"],
            "phase1235_stop_respected": True,
            "new_user_authorized_stage": True,
        },
        "material": {
            "world_count": WORLD_COUNT,
            "worlds_per_partition": WORLDS_PER_PARTITION,
            "partitions": list(PARTITIONS),
            "protocols": list(PROTOCOLS),
            "topologies": list(TOPOLOGIES),
            "row_count": len(rows),
            "independent_unit": "world",
            "base_pair_count": EXPECTED_BASE_PAIRS,
            "material_digest": digest(rows),
            "material_audit": material_audit,
            "parser_fixture_digest": digest(fixtures),
        },
        "typed_observation_contract": {
            "pi": "bare, exact-sentence, or natural instruction in the model prompt",
            "rho_content_score": "external full-candidate continuation ranking using the same marker suffix under every pi",
            "rho_contract_score": "external complete-answer ranking matched to each pi",
            "rho_greedy": "unconstrained model greedy generation",
            "kappa_content": "one non-negated registered marker identity",
            "kappa_exact": "normalized string equality to the protocol-specific contract",
        },
        "behavior_authorization": {
            "qwen_hidden": "Qwen3 content-score and greedy-content gates only; exact formatting is separately reported",
            "cross_model": "at least two models independently pass the same content lane",
            "no_retroactive_phase1235_reclassification": True,
        },
        "response_tensor": {
            "model": "qwen3",
            "events": "all 36 residual writes, attention outputs, and MLP outputs",
            "roles": ["generation_boundary", "query_object_end", "source_gold_end"],
            "full_vector_scope": "generation-boundary residual differences only",
            "projection_dimension": PROJECTION_DIM,
            "projection_seed": PROJECTION_SEED,
            "difference": "binding_state_1 minus binding_state_0 within world/query/protocol",
        },
        "structure_competition": {
            "families": ["mean", "identity", "scalar", "affine_scalar", "krr_0.01", "krr_0.1", "krr_1.0"],
            "candidate_events": "generation-boundary residual depths 1..36",
            "ordered_protocol_pairs": [[source, target] for source in PROTOCOLS for target in PROTOCOLS if source != target],
            "fit_partition": "discovery",
            "selection_partition": "model_selection",
            "final_fit_partition": "discovery + model_selection",
            "sealed_partition": "sealed and never used for event/family selection",
            "matched_null": "deterministic within-partition source permutation",
            "selection_rule": "maximum selection score among candidates passing all model-selection thresholds",
        },
        "causal_interchange": {
            "authorization": "sealed structure gate only",
            "receiver": "binding_state_0 prompt in selected target protocol",
            "intervention": "selected full-vector predicted binding delta at the selected residual depth and generation boundary",
            "conditions": ["zero", "oracle_target", "mapped_source", "raw_source", "shuffled_mapped", "opposite_mapped", "random_norm"],
            "primary_readout": "state1-versus-state0 complete-candidate log-odds shift",
            "secondary_readout": "receiver-protocol greedy content and exact-format retention",
            "one_shot": True,
        },
        "thresholds": THRESHOLDS,
        "manifest_summaries": manifest_summaries,
        "execution": {
            "models_sequential": True,
            "precision": "float16",
            "quantization": "none",
            "cuda_required": True,
            "behavior_models": list(MODELS),
            "hidden_and_causal_model": "qwen3",
        },
        "stop_rules": [
            "No hidden capture unless the prospectively typed Qwen3 content lane passes.",
            "No sealed inspection before a family/event/protocol map is selected on model_selection.",
            "No causal interchange unless the frozen candidate passes sealed prediction thresholds.",
            "A failed candidate ends this response-law registry; do not reselect a new event from sealed data.",
            "A positive result is Qwen3-specific controlled sufficiency, not natural necessity or cross-model identity.",
        ],
        "claim_boundary": [
            "The materials are synthetic registries, not natural semantic knowledge.",
            "Behavioral content support does not establish a protocol-independent hidden content variable.",
            "A response map is a predictive functional relation, not a fixed semantic coordinate.",
            "Only the frozen sealed intervention can support a causal sufficiency claim within this protocol family.",
        ],
    }
    contract["contract_digest"] = digest(contract)
    write_jsonl(MATERIAL_PATH, rows)
    write_jsonl(PARSER_FIXTURES_PATH, fixtures)
    write_json(CONTRACT_PATH, contract)
    print(canonical_json({"status": "phase1236_preregistered", "rows": len(rows), "contract_digest": contract["contract_digest"]}))


def verify_frozen() -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    verify_upstream()
    contract = read_json(CONTRACT_PATH)
    if contract["contract_digest"] != digest(strip_digest(contract, "contract_digest")):
        raise RuntimeError("contract digest drift")
    if contract["source_hashes"] != source_hashes():
        raise RuntimeError("source hash drift")
    rows = read_jsonl(MATERIAL_PATH)
    if digest(rows) != contract["material"]["material_digest"]:
        raise RuntimeError("material drift")
    manifests = {}
    for model_name in MODELS:
        manifest = read_jsonl(model_manifest_path(model_name))
        if digest(manifest) != contract["manifest_summaries"][model_name]["manifest_digest"]:
            raise RuntimeError(f"{model_name} manifest drift")
        manifests[model_name] = manifest
    preaudit = read_json(PREAUDIT_PATH)
    if preaudit.get("all_checks_passed") is not True or preaudit.get("contract_digest") != contract["contract_digest"]:
        raise RuntimeError("independent preaudit missing or failed")
    return contract, rows, manifests


def grouped_batches(entries: list[dict[str, Any]], batch_rows: int, key: str) -> Iterable[list[dict[str, Any]]]:
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        groups[int(entry[key])].append(entry)
    for length in sorted(groups):
        values = groups[length]
        for start in range(0, len(values), batch_rows):
            yield values[start : start + batch_rows]


def model_forward(model: Any, **kwargs: Any) -> Any:
    try:
        return model(**kwargs)
    except TypeError as exc:
        if "logits_to_keep" not in str(exc):
            raise
        kwargs.pop("logits_to_keep", None)
        return model(**kwargs)


def direct_candidate_scores(
    model: Any,
    device: torch.device,
    manifest: list[dict[str, Any]],
    field: str,
    batch_rows: int,
) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, Any]]:
    entries = []
    for row in manifest:
        suffixes = row[field]
        continuation_length = len(next(iter(suffixes.values())))
        entries.append({**row, "total_length": len(row["input_ids"]) + continuation_length})
    result: dict[str, dict[str, dict[str, Any]]] = {}
    started = time.time()
    batches = 0
    all_finite = True
    for batch in grouped_batches(entries, batch_rows, "total_length"):
        sequences = []
        metadata = []
        for row in batch:
            for candidate in row["candidates"]:
                continuation = [int(value) for value in row[field][candidate]]
                sequences.append([int(value) for value in row["input_ids"]] + continuation)
                metadata.append((row, candidate, continuation))
        input_ids = torch.tensor(sequences, dtype=torch.long, device=device)
        continuation_length = len(metadata[0][2])
        with torch.inference_mode():
            output = model_forward(
                model,
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                use_cache=False,
                logits_to_keep=continuation_length + 1,
                return_dict=True,
            )
        output_start = input_ids.shape[1] - output.logits.shape[1]
        for index, (row, candidate, continuation) in enumerate(metadata):
            token_scores = []
            finite = True
            prompt_length = len(row["input_ids"])
            for offset, token_id in enumerate(continuation):
                logits = output.logits[index, prompt_length + offset - 1 - output_start].float()
                finite = finite and bool(torch.isfinite(logits).all().item())
                score = logits[int(token_id)] - torch.logsumexp(logits, dim=-1)
                token_scores.append(float(score.item()))
            result.setdefault(row["item_id"], {})[candidate] = {
                "sum_log_probability": float(sum(token_scores)),
                "mean_log_probability": float(sum(token_scores) / len(token_scores)),
                "token_count": len(token_scores),
                "all_vocab_logits_finite": finite and all(math.isfinite(value) for value in token_scores),
            }
            all_finite = all_finite and result[row["item_id"]][candidate]["all_vocab_logits_finite"]
        del output, input_ids
        batches += 1
        if batches % 50 == 0:
            print(f"[phase1236/{field}] batches={batches}", flush=True)
    return result, {"field": field, "batch_count": batches, "elapsed_seconds": time.time() - started, "all_finite": all_finite}


def greedy_generation(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    manifest: list[dict[str, Any]],
    batch_rows: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    started = time.time()
    results: dict[str, dict[str, Any]] = {}
    batches = 0
    eos = tokenizer.eos_token_id
    for batch in grouped_batches(manifest, batch_rows, "input_token_count"):
        input_ids = torch.tensor([row["input_ids"] for row in batch], dtype=torch.long, device=device)
        with torch.inference_mode():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                do_sample=False,
                num_beams=1,
                use_cache=True,
                max_new_tokens=GENERATION_BUDGET,
                eos_token_id=eos,
                pad_token_id=int(tokenizer.pad_token_id),
                return_dict_in_generate=True,
            )
        suffixes = generated.sequences[:, input_ids.shape[1] :].detach().cpu().tolist()
        for index, row in enumerate(batch):
            suffix = [int(value) for value in suffixes[index]]
            terminated = eos is not None and eos in suffix
            if terminated:
                suffix = suffix[: suffix.index(eos)]
            results[row["item_id"]] = {
                "generated_token_ids": suffix,
                "generated_text": tokenizer.decode(suffix, skip_special_tokens=True, clean_up_tokenization_spaces=False),
                "terminated_by_eos": terminated,
            }
        del generated, input_ids
        batches += 1
        if batches % 50 == 0:
            print(f"[phase1236/generation] batches={batches}", flush=True)
    return results, {"batch_count": batches, "elapsed_seconds": time.time() - started}


def unique_winner(scores: dict[str, dict[str, Any]]) -> str | None:
    ordered = sorted(scores, key=lambda key: scores[key]["sum_log_probability"], reverse=True)
    if len(ordered) > 1 and abs(scores[ordered[0]]["sum_log_probability"] - scores[ordered[1]]["sum_log_probability"]) <= TIE_TOLERANCE:
        return None
    return ordered[0]


def run_behavior(model_name: str) -> None:
    if behavior_raw_path(model_name).exists() or behavior_summary_path(model_name).exists():
        raise RuntimeError(f"{model_name} behavior output already exists")
    contract, material, manifests = verify_frozen()
    manifest = manifests[model_name]
    material_by_id = {row["item_id"]: row for row in material}
    batch_rows = {"qwen3": 8, "glm4": 2, "deepseek7b": 4}[model_name]
    generation_batch = {"qwen3": 16, "glm4": 2, "deepseek7b": 4}[model_name]
    started = time.time()
    model = None
    try:
        model, tokenizer, device, placement = load_fp16(model_name)
        precision = quantization_audit(model)
        if precision["has_quantized_modules"] or set(precision["parameter_dtypes"]) != {"float16"}:
            raise RuntimeError(f"{model_name} numerical contract failed")
        content_scores, content_runtime = direct_candidate_scores(
            model, device, manifest, "content_candidate_token_ids", batch_rows
        )
        contract_scores, contract_runtime = direct_candidate_scores(
            model, device, manifest, "contract_candidate_token_ids", batch_rows
        )
        generations, generation_runtime = greedy_generation(model, tokenizer, device, manifest, generation_batch)
        raw = []
        for manifest_row in manifest:
            item_id = manifest_row["item_id"]
            row = material_by_id[item_id]
            content_prediction = unique_winner(content_scores[item_id])
            contract_prediction = unique_winner(contract_scores[item_id])
            generation = generations[item_id]
            parsed = parse_output(
                generation["generated_text"], row["candidates"], row["expected_exact"], row["protocol"]
            )
            value: dict[str, Any] = {
                "phase": PHASE,
                "schema_version": "phase1236.behavior_row.v1",
                "model": model_name,
                "contract_digest": contract["contract_digest"],
                "item_id": item_id,
                "pair_id": row["pair_id"],
                "base_pair_id": row["base_pair_id"],
                "world_id": row["world_id"],
                "world_index": row["world_index"],
                "partition": row["partition"],
                "topology": row["topology"],
                "template_index": row["template_index"],
                "protocol": row["protocol"],
                "binding_state": row["binding_state"],
                "query_index": row["query_index"],
                "gold": row["gold"],
                "candidates": row["candidates"],
                "expected_exact": row["expected_exact"],
                "content_candidate_scores": content_scores[item_id],
                "contract_candidate_scores": contract_scores[item_id],
                "content_prediction": content_prediction,
                "contract_prediction": contract_prediction,
                "content_score_correct": content_prediction == row["gold"],
                "contract_score_correct": contract_prediction == row["gold"],
                "candidate_scores_finite": all(
                    candidate["all_vocab_logits_finite"]
                    for scores in (content_scores[item_id], contract_scores[item_id])
                    for candidate in scores.values()
                ),
                "generation": generation,
                "generation_parse": parsed,
                "generation_content_correct": parsed["prediction"] == row["gold"],
                "generation_exact": parsed["exact"],
                "generation_format_valid": parsed["format_valid"],
            }
            value["behavior_row_digest"] = digest(value)
            raw.append(value)
        raw.sort(key=lambda row: row["item_id"])
        summary: dict[str, Any] = {
            "phase": PHASE,
            "schema_version": "phase1236.behavior_summary.v1",
            "created_at_utc": utc_now(),
            "model": model_name,
            "contract_digest": contract["contract_digest"],
            "case_count": len(raw),
            "raw_digest": digest(raw),
            "runtimes": {"content_score": content_runtime, "contract_score": contract_runtime, "generation": generation_runtime},
            "precision_audit": precision,
            "placement": placement,
            "elapsed_seconds": time.time() - started,
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
        summary["summary_digest"] = digest(summary)
        write_jsonl(behavior_raw_path(model_name), raw)
        write_json(behavior_summary_path(model_name), summary)
        print(canonical_json({"status": "behavior_complete", "model": model_name, "rows": len(raw), "summary_digest": summary["summary_digest"]}))
    finally:
        if model is not None:
            release_fp16(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def rate(rows: list[dict[str, Any]], field: str) -> float:
    return float(sum(bool(row[field]) for row in rows) / len(rows)) if rows else float("nan")


def behavior_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cells = {}
    for partition in PARTITIONS:
        for protocol in PROTOCOLS:
            selected = [row for row in rows if row["partition"] == partition and row["protocol"] == protocol]
            cells[f"{partition}|{protocol}"] = {
                "n": len(selected),
                "finite": rate(selected, "candidate_scores_finite"),
                "content_score": rate(selected, "content_score_correct"),
                "contract_score": rate(selected, "contract_score_correct"),
                "generation_content": rate(selected, "generation_content_correct"),
                "generation_exact": rate(selected, "generation_exact"),
                "generation_format_valid": rate(selected, "generation_format_valid"),
            }
    by_protocol = {}
    for protocol in PROTOCOLS:
        selected = [row for row in rows if row["protocol"] == protocol]
        by_protocol[protocol] = {
            "n": len(selected),
            "finite": rate(selected, "candidate_scores_finite"),
            "content_score": rate(selected, "content_score_correct"),
            "contract_score": rate(selected, "contract_score_correct"),
            "generation_content": rate(selected, "generation_content_correct"),
            "generation_exact": rate(selected, "generation_exact"),
            "generation_format_valid": rate(selected, "generation_format_valid"),
        }
    finite = min(value["finite"] for value in cells.values())
    content_score_worst_cell = min(value["content_score"] for value in cells.values())
    content_score_worst_protocol = min(value["content_score"] for value in by_protocol.values())
    contract_score_worst_cell = min(value["contract_score"] for value in cells.values())
    generation_content_worst_cell = min(value["generation_content"] for value in cells.values())
    generation_content_worst_protocol = min(value["generation_content"] for value in by_protocol.values())
    format_exact_worst = min(by_protocol[protocol]["generation_format_valid"] for protocol in ("bare", "sentence"))
    content_score_gate = bool(
        finite >= THRESHOLDS["finite_rate"]
        and content_score_worst_cell >= THRESHOLDS["content_score_worst_partition_protocol"]
        and content_score_worst_protocol >= THRESHOLDS["content_score_worst_protocol"]
    )
    generation_content_gate = bool(
        generation_content_worst_cell >= THRESHOLDS["generation_content_worst_partition_protocol"]
        and generation_content_worst_protocol >= THRESHOLDS["generation_content_worst_protocol"]
    )
    return {
        "case_count": len(rows),
        "overall": {
            "finite": rate(rows, "candidate_scores_finite"),
            "content_score": rate(rows, "content_score_correct"),
            "contract_score": rate(rows, "contract_score_correct"),
            "generation_content": rate(rows, "generation_content_correct"),
            "generation_exact": rate(rows, "generation_exact"),
        },
        "by_partition_protocol": cells,
        "by_protocol": by_protocol,
        "worst": {
            "finite": finite,
            "content_score_partition_protocol": content_score_worst_cell,
            "content_score_protocol": content_score_worst_protocol,
            "contract_score_partition_protocol": contract_score_worst_cell,
            "generation_content_partition_protocol": generation_content_worst_cell,
            "generation_content_protocol": generation_content_worst_protocol,
            "format_valid_bare_sentence": format_exact_worst,
        },
        "gates": {
            "content_score": content_score_gate,
            "contract_score": contract_score_worst_cell >= THRESHOLDS["contract_score_worst_partition_protocol"],
            "generation_content": generation_content_gate,
            "format_valid": format_exact_worst >= THRESHOLDS["format_valid_worst_bare_sentence"],
            "hidden_content_lane": content_score_gate and generation_content_gate,
        },
    }


def adjudicate_behavior() -> None:
    if BEHAVIOR_ADJUDICATION_PATH.exists():
        raise RuntimeError("behavior adjudication already exists")
    contract, _material, _manifests = verify_frozen()
    models = {}
    for model_name in MODELS:
        if not behavior_raw_path(model_name).exists() or not behavior_audit_path(model_name).exists():
            raise RuntimeError(f"{model_name} behavior or independent audit missing")
        audit = read_json(behavior_audit_path(model_name))
        if audit.get("all_checks_passed") is not True:
            raise RuntimeError(f"{model_name} independent behavior audit failed")
        models[model_name] = behavior_metrics(read_jsonl(behavior_raw_path(model_name)))
    authorized = [name for name in MODELS if models[name]["gates"]["hidden_content_lane"]]
    value: dict[str, Any] = {
        "phase": PHASE,
        "schema_version": "phase1236.behavior_adjudication.v1",
        "created_at_utc": utc_now(),
        "contract_digest": contract["contract_digest"],
        "models": models,
        "authorized_content_models": authorized,
        "qwen_hidden_authorized": "qwen3" in authorized,
        "cross_model_behavior_authorized": len(authorized) >= 2,
        "exact_format_not_required_for_content_lane": True,
        "phase1235_not_reclassified": True,
    }
    value["adjudication_digest"] = digest(value)
    write_json(BEHAVIOR_ADJUDICATION_PATH, value)
    print(canonical_json({"status": "behavior_adjudicated", "authorized": authorized, "qwen_hidden": value["qwen_hidden_authorized"]}))


ROLES = ("generation_boundary", "query_object_end", "source_gold_end")


class EventCapture:
    def __init__(self, layers: list[Any]):
        self.layers = layers
        self.positions: torch.Tensor | None = None
        self.values: dict[str, torch.Tensor] = {}
        self.calls: dict[str, int] = defaultdict(int)
        self.handles: list[Any] = []

    def _hook(self, event_id: str):
        def hook(_module: Any, _args: Any, output: Any):
            value = output[0] if isinstance(output, tuple) else output
            if self.positions is None or not isinstance(value, torch.Tensor):
                raise RuntimeError(f"capture not initialized for {event_id}")
            positions = self.positions.to(value.device)
            batch = torch.arange(value.shape[0], device=value.device)[:, None]
            self.values[event_id] = value[batch, positions, :].detach()
            self.calls[event_id] += 1
            return output
        return hook

    def register(self) -> None:
        for depth, layer in enumerate(self.layers, 1):
            self.handles.append(layer.register_forward_hook(self._hook(f"residual_d{depth:02d}")))
            self.handles.append(layer.self_attn.register_forward_hook(self._hook(f"attention_d{depth:02d}")))
            self.handles.append(layer.mlp.register_forward_hook(self._hook(f"mlp_d{depth:02d}")))

    def begin(self, positions: torch.Tensor) -> None:
        self.positions = positions
        self.values = {}
        self.calls = defaultdict(int)

    def validate(self, event_ids: list[str]) -> None:
        if set(self.values) != set(event_ids):
            raise RuntimeError("capture event mismatch")
        if any(self.calls[event] != 1 for event in event_ids):
            raise RuntimeError("capture call mismatch")

    def close(self) -> None:
        for handle in reversed(self.handles):
            handle.remove()
        self.handles = []


def orthonormal_projection(hidden_size: int) -> np.ndarray:
    rng = np.random.default_rng(PROJECTION_SEED)
    matrix = rng.standard_normal((hidden_size, PROJECTION_DIM)).astype(np.float64)
    q, _r = np.linalg.qr(matrix, mode="reduced")
    return q.T.astype(np.float32)


def capture_qwen() -> None:
    if CAPTURE_ARRAY_PATH.exists() or CAPTURE_META_PATH.exists():
        raise RuntimeError("Qwen response tensor already exists")
    contract, material, manifests = verify_frozen()
    behavior = read_json(BEHAVIOR_ADJUDICATION_PATH)
    if behavior.get("qwen_hidden_authorized") is not True:
        denied = {
            "phase": PHASE,
            "schema_version": "phase1236.response_tensor_metadata.v1",
            "created_at_utc": utc_now(),
            "contract_digest": contract["contract_digest"],
            "status": "denied_by_behavior_gate",
            "capture_performed": False,
        }
        denied["metadata_digest"] = digest(denied)
        write_json(CAPTURE_META_PATH, denied)
        print(canonical_json({"status": "capture_denied"}))
        return
    manifest = manifests["qwen3"]
    manifest_by_id = {row["item_id"]: row for row in manifest}
    material_by_id = {row["item_id"]: row for row in material}
    base_pairs = sorted({row["base_pair_id"] for row in material})
    pair_index = {name: index for index, name in enumerate(base_pairs)}
    protocol_index = {name: index for index, name in enumerate(PROTOCOLS)}
    model = None
    capture = None
    started = time.time()
    try:
        model, _tokenizer, device, placement = load_fp16("qwen3")
        precision = quantization_audit(model)
        if precision["has_quantized_modules"] or set(precision["parameter_dtypes"]) != {"float16"}:
            raise RuntimeError("Qwen3 capture numerical gate failed")
        layers = get_layers(model)
        depth_count = len(layers)
        hidden_size = int(model.get_input_embeddings().weight.shape[1])
        event_ids = [f"{component}_d{depth:02d}" for depth in range(1, depth_count + 1) for component in ("residual", "attention", "mlp")]
        event_index = {name: index for index, name in enumerate(event_ids)}
        projection = orthonormal_projection(hidden_size)
        projection_t = torch.tensor(projection.T, dtype=torch.float32, device=device)
        full_residual = np.empty((len(base_pairs), len(PROTOCOLS), depth_count, hidden_size), dtype=np.float16)
        projected = np.empty((len(base_pairs), len(PROTOCOLS), len(event_ids), len(ROLES), PROJECTION_DIM), dtype=np.float16)
        pairs = []
        for base_pair_id in base_pairs:
            for protocol in PROTOCOLS:
                pair_id = f"{base_pair_id}|{protocol}"
                rows = [row for row in material if row["pair_id"] == pair_id]
                if len(rows) != 2:
                    raise RuntimeError("state pair cardinality drift")
                rows.sort(key=lambda row: row["binding_state"])
                cases = [manifest_by_id[row["item_id"]] for row in rows]
                if cases[0]["input_token_count"] != cases[1]["input_token_count"]:
                    raise RuntimeError("paired input length mismatch")
                pairs.append({
                    "base_pair_id": base_pair_id,
                    "protocol": protocol,
                    "cases": cases,
                    "input_token_count": cases[0]["input_token_count"],
                })
        capture = EventCapture(layers)
        capture.register()
        completed = 0
        with torch.inference_mode():
            for batch in grouped_batches(pairs, 4, "input_token_count"):
                cases = [case for pair in batch for case in pair["cases"]]
                input_ids = torch.tensor([case["input_ids"] for case in cases], dtype=torch.long, device=device)
                positions = torch.tensor(
                    [[int(case["positions"][role]) for role in ROLES] for case in cases], dtype=torch.long, device=device
                )
                capture.begin(positions)
                output = model(
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    use_cache=False,
                    logits_to_keep=1,
                    return_dict=True,
                )
                capture.validate(event_ids)
                for local_index, pair in enumerate(batch):
                    left = 2 * local_index
                    right = left + 1
                    pidx = pair_index[pair["base_pair_id"]]
                    tidx = protocol_index[pair["protocol"]]
                    for event_id in event_ids:
                        delta = capture.values[event_id][right].float() - capture.values[event_id][left].float()
                        projected[pidx, tidx, event_index[event_id]] = (delta @ projection_t).cpu().numpy().astype(np.float16)
                        if event_id.startswith("residual_"):
                            depth = int(event_id.rsplit("d", 1)[1]) - 1
                            full_residual[pidx, tidx, depth] = delta[0].cpu().numpy().astype(np.float16)
                    completed += 1
                del output, input_ids, positions
                if completed % 48 == 0:
                    print(f"[phase1236/capture] pairs={completed}/{len(pairs)}", flush=True)
        pair_metadata = []
        for base_pair_id in base_pairs:
            exemplar = next(row for row in material if row["base_pair_id"] == base_pair_id)
            pair_metadata.append({
                "base_pair_id": base_pair_id,
                "world_id": exemplar["world_id"],
                "world_index": exemplar["world_index"],
                "partition": exemplar["partition"],
                "topology": exemplar["topology"],
                "template_index": exemplar["template_index"],
                "query_index": exemplar["query_index"],
            })
        CAPTURE_ARRAY_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            CAPTURE_ARRAY_PATH,
            full_residual=full_residual,
            projected=projected,
            projection=projection,
        )
        metadata: dict[str, Any] = {
            "phase": PHASE,
            "schema_version": "phase1236.response_tensor_metadata.v1",
            "created_at_utc": utc_now(),
            "contract_digest": contract["contract_digest"],
            "status": "complete",
            "capture_performed": True,
            "base_pair_count": len(base_pairs),
            "protocols": list(PROTOCOLS),
            "depth_count": depth_count,
            "hidden_size": hidden_size,
            "event_ids": event_ids,
            "roles": list(ROLES),
            "pair_metadata": pair_metadata,
            "array_file_sha256": file_sha256(CAPTURE_ARRAY_PATH),
            "precision_audit": precision,
            "placement": placement,
            "elapsed_seconds": time.time() - started,
        }
        metadata["metadata_digest"] = digest(metadata)
        write_json(CAPTURE_META_PATH, metadata)
        print(canonical_json({"status": "capture_complete", "pairs": len(base_pairs), "array_sha256": metadata["array_file_sha256"]}))
    finally:
        if capture is not None:
            capture.close()
        if model is not None:
            release_fp16(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def cosine_rows(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    numerator = np.sum(prediction * target, axis=1)
    denominator = np.linalg.norm(prediction, axis=1) * np.linalg.norm(target, axis=1) + EPSILON
    return numerator / denominator


def fit_response_model(family: str, x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    if family == "mean":
        return {"family": family, "mean": y.mean(axis=0)}
    if family == "identity":
        return {"family": family}
    numerator = float(np.sum(x * y))
    denominator = float(np.sum(x * x)) + EPSILON
    if family == "scalar":
        return {"family": family, "scale": numerator / denominator}
    if family == "affine_scalar":
        x_mean = x.mean(axis=0)
        y_mean = y.mean(axis=0)
        centered_x = x - x_mean
        scale = float(np.sum(centered_x * (y - y_mean)) / (np.sum(centered_x * centered_x) + EPSILON))
        return {"family": family, "scale": scale, "x_mean": x_mean, "y_mean": y_mean}
    if family.startswith("krr_"):
        multiplier = float(family.split("_", 1)[1])
        kernel = x @ x.T
        scale = float(np.trace(kernel) / max(len(x), 1)) + EPSILON
        alpha = np.linalg.solve(kernel + multiplier * scale * np.eye(len(x)), y)
        return {"family": family, "train_x": x, "alpha": alpha}
    raise ValueError(family)


def predict_response(model: dict[str, Any], x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    family = model["family"]
    if family == "mean":
        return np.repeat(model["mean"][None, :], len(x), axis=0)
    if family == "identity":
        return x.copy()
    if family == "scalar":
        return x * model["scale"]
    if family == "affine_scalar":
        return (x - model["x_mean"]) * model["scale"] + model["y_mean"]
    if family.startswith("krr_"):
        return (x @ model["train_x"].T) @ model["alpha"]
    raise ValueError(family)


def response_metrics(prediction: np.ndarray, target: np.ndarray, train_mean: np.ndarray, shuffled_prediction: np.ndarray) -> dict[str, float]:
    mse = float(np.mean(np.sum((prediction - target) ** 2, axis=1)))
    zero_mse = float(np.mean(np.sum(target**2, axis=1))) + EPSILON
    mean_mse = float(np.mean(np.sum((target - train_mean[None, :]) ** 2, axis=1))) + EPSILON
    shuffled_mse = float(np.mean(np.sum((shuffled_prediction - target) ** 2, axis=1))) + EPSILON
    cosines = cosine_rows(prediction, target)
    return {
        "mse": mse,
        "zero_mse": zero_mse,
        "mean_mse": mean_mse,
        "shuffled_mse": shuffled_mse,
        "improvement_over_zero": float(1.0 - mse / zero_mse),
        "improvement_over_mean": float(1.0 - mse / mean_mse),
        "shuffled_advantage": float((shuffled_mse - mse) / zero_mse),
        "median_cosine": float(np.median(cosines)),
        "mean_cosine": float(np.mean(cosines)),
        "positive_cosine_fraction": float(np.mean(cosines > 0.0)),
        "prediction_target_norm_ratio": float(np.median(np.linalg.norm(prediction, axis=1) / (np.linalg.norm(target, axis=1) + EPSILON))),
    }


def candidate_pass(metrics: dict[str, float]) -> bool:
    return bool(
        metrics["improvement_over_mean"] >= THRESHOLDS["structure_improvement_over_mean"]
        and metrics["shuffled_advantage"] >= THRESHOLDS["structure_shuffled_advantage"]
        and metrics["median_cosine"] >= THRESHOLDS["structure_median_cosine"]
        and metrics["positive_cosine_fraction"] >= THRESHOLDS["structure_positive_cosine_fraction"]
        and 0.25 <= metrics["prediction_target_norm_ratio"] <= 4.0
    )


def deterministic_roll(values: np.ndarray) -> np.ndarray:
    return np.roll(values, shift=1, axis=0)


def fit_structures() -> None:
    if STRUCTURE_PATH.exists():
        raise RuntimeError("structure result already exists")
    contract, _material, _manifests = verify_frozen()
    metadata = read_json(CAPTURE_META_PATH)
    if metadata.get("capture_performed") is not True:
        value = {
            "phase": PHASE,
            "schema_version": "phase1236.structure_competition.v1",
            "created_at_utc": utc_now(),
            "contract_digest": contract["contract_digest"],
            "status": "denied_no_response_tensor",
            "structure_gate": False,
        }
        value["structure_digest"] = digest(value)
        write_json(STRUCTURE_PATH, value)
        return
    if file_sha256(CAPTURE_ARRAY_PATH) != metadata["array_file_sha256"]:
        raise RuntimeError("response tensor file drift")
    arrays = np.load(CAPTURE_ARRAY_PATH)
    residual = arrays["full_residual"].astype(np.float32)
    pair_meta = metadata["pair_metadata"]
    indices = {partition: np.array([index for index, row in enumerate(pair_meta) if row["partition"] == partition], dtype=np.int64) for partition in PARTITIONS}
    pindex = {name: index for index, name in enumerate(PROTOCOLS)}
    families = contract["structure_competition"]["families"]
    records = []
    for source in PROTOCOLS:
        for target in PROTOCOLS:
            if source == target:
                continue
            for depth in range(residual.shape[2]):
                x_train = residual[indices["discovery"], pindex[source], depth]
                y_train = residual[indices["discovery"], pindex[target], depth]
                x_select = residual[indices["model_selection"], pindex[source], depth]
                y_select = residual[indices["model_selection"], pindex[target], depth]
                for family in families:
                    fitted = fit_response_model(family, x_train, y_train)
                    prediction = predict_response(fitted, x_select)
                    shuffled_prediction = predict_response(fitted, deterministic_roll(x_select))
                    metrics = response_metrics(prediction, y_select, y_train.mean(axis=0), shuffled_prediction)
                    passed = candidate_pass(metrics)
                    score = (
                        metrics["improvement_over_mean"]
                        + metrics["shuffled_advantage"]
                        + metrics["median_cosine"]
                        + metrics["positive_cosine_fraction"]
                    )
                    records.append({
                        "source_protocol": source,
                        "target_protocol": target,
                        "depth": depth + 1,
                        "family": family,
                        "selection_metrics": metrics,
                        "selection_pass": passed,
                        "selection_score": float(score),
                    })
    passing = [row for row in records if row["selection_pass"]]
    winner = max(passing, key=lambda row: (row["selection_score"], -row["depth"], row["family"], row["source_protocol"], row["target_protocol"])) if passing else None
    sealed = None
    if winner is not None:
        fit_indices = np.concatenate((indices["discovery"], indices["model_selection"]))
        source = winner["source_protocol"]
        target = winner["target_protocol"]
        depth = int(winner["depth"]) - 1
        x_fit = residual[fit_indices, pindex[source], depth]
        y_fit = residual[fit_indices, pindex[target], depth]
        x_sealed = residual[indices["sealed"], pindex[source], depth]
        y_sealed = residual[indices["sealed"], pindex[target], depth]
        fitted = fit_response_model(winner["family"], x_fit, y_fit)
        prediction = predict_response(fitted, x_sealed)
        shuffled_prediction = predict_response(fitted, deterministic_roll(x_sealed))
        metrics = response_metrics(prediction, y_sealed, y_fit.mean(axis=0), shuffled_prediction)
        sealed = {"metrics": metrics, "gate": candidate_pass(metrics)}
    value: dict[str, Any] = {
        "phase": PHASE,
        "schema_version": "phase1236.structure_competition.v1",
        "created_at_utc": utc_now(),
        "contract_digest": contract["contract_digest"],
        "status": "complete",
        "candidate_count": len(records),
        "selection_passing_count": len(passing),
        "winner": winner,
        "sealed": sealed,
        "structure_gate": bool(sealed and sealed["gate"]),
        "selection_records": records,
        "sealed_inspected_once": winner is not None,
        "sealed_not_used_for_selection": True,
    }
    value["structure_digest"] = digest(value)
    write_json(STRUCTURE_PATH, value)
    print(canonical_json({"status": "structure_complete", "winner": winner, "sealed_gate": value["structure_gate"]}))


class ResidualPatch:
    def __init__(self, layer: Any, positions: torch.Tensor, deltas: torch.Tensor):
        self.layer = layer
        self.positions = positions
        self.deltas = deltas
        self.handle = None
        self.calls = 0

    def _hook(self, _module: Any, _args: Any, output: Any):
        value = output[0] if isinstance(output, tuple) else output
        if not isinstance(value, torch.Tensor):
            raise RuntimeError("patch layer returned non-tensor")
        positions = self.positions.to(value.device)
        deltas = self.deltas.to(value.device, dtype=value.dtype)
        batch = torch.arange(value.shape[0], device=value.device)
        patched = value.clone()
        patched[batch, positions, :] = value[batch, positions, :] + deltas
        self.calls += 1
        return (patched,) + output[1:] if isinstance(output, tuple) else patched

    def __enter__(self):
        self.handle = self.layer.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any):
        if self.handle is not None:
            self.handle.remove()
        self.handle = None


def random_norm_deltas(keys: list[str], norms: np.ndarray, hidden_size: int) -> np.ndarray:
    result = np.empty((len(keys), hidden_size), dtype=np.float64)
    for index, key in enumerate(keys):
        seed = int(hashlib.sha256(f"phase1236|{key}".encode("utf-8")).hexdigest()[:16], 16) % (2**32)
        rng = np.random.default_rng(seed)
        vector = rng.standard_normal(hidden_size)
        vector /= np.linalg.norm(vector) + EPSILON
        result[index] = vector * float(norms[index])
    return result


def score_patched_rows(
    model: Any,
    device: torch.device,
    layer: Any,
    rows: list[dict[str, Any]],
    deltas: np.ndarray,
    batch_rows: int = 4,
) -> dict[str, dict[str, dict[str, Any]]]:
    entries = []
    for index, row in enumerate(rows):
        suffixes = row["contract_candidate_token_ids"]
        continuation_length = len(next(iter(suffixes.values())))
        entries.append({**row, "delta_index": index, "total_length": len(row["input_ids"]) + continuation_length})
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for batch in grouped_batches(entries, batch_rows, "total_length"):
        sequences = []
        metadata = []
        patch_deltas = []
        positions = []
        for row in batch:
            for candidate in row["candidates"]:
                continuation = row["contract_candidate_token_ids"][candidate]
                sequences.append(row["input_ids"] + continuation)
                metadata.append((row, candidate, continuation))
                patch_deltas.append(deltas[int(row["delta_index"])])
                positions.append(len(row["input_ids"]) - 1)
        input_ids = torch.tensor(sequences, dtype=torch.long, device=device)
        position_t = torch.tensor(positions, dtype=torch.long, device=device)
        delta_t = torch.tensor(np.asarray(patch_deltas), dtype=torch.float32, device=device)
        continuation_length = len(metadata[0][2])
        with torch.inference_mode(), ResidualPatch(layer, position_t, delta_t) as patch:
            output = model_forward(
                model,
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                use_cache=False,
                logits_to_keep=continuation_length + 1,
                return_dict=True,
            )
        if patch.calls != 1:
            raise RuntimeError(f"patch call count drift: {patch.calls}")
        output_start = input_ids.shape[1] - output.logits.shape[1]
        for index, (row, candidate, continuation) in enumerate(metadata):
            scores = []
            finite = True
            prompt_length = len(row["input_ids"])
            for offset, token_id in enumerate(continuation):
                logits = output.logits[index, prompt_length + offset - 1 - output_start].float()
                finite = finite and bool(torch.isfinite(logits).all().item())
                score = logits[int(token_id)] - torch.logsumexp(logits, dim=-1)
                scores.append(float(score.item()))
            result.setdefault(row["item_id"], {})[candidate] = {
                "sum_log_probability": float(sum(scores)),
                "all_vocab_logits_finite": finite and all(math.isfinite(value) for value in scores),
            }
        del output, input_ids, position_t, delta_t
    return result


def condition_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    effects = np.array([row["log_odds_effect"] for row in records], dtype=np.float64)
    return {
        "n": len(records),
        "median_log_odds_effect": float(np.median(effects)),
        "mean_log_odds_effect": float(np.mean(effects)),
        "positive_effect_fraction": float(np.mean(effects > 0.0)),
        "state1_winner_fraction": float(np.mean([row["winner"] == row["gold1"] for row in records])),
        "state0_winner_fraction": float(np.mean([row["winner"] == row["gold0"] for row in records])),
        "all_finite_fraction": float(np.mean([row["finite"] for row in records])),
    }


def causal_qwen() -> None:
    if CAUSAL_PATH.exists():
        raise RuntimeError("causal result already exists")
    contract, material, manifests = verify_frozen()
    structure = read_json(STRUCTURE_PATH)
    structure_audit = read_json(STRUCTURE_AUDIT_PATH)
    if structure_audit.get("all_checks_passed") is not True or structure_audit.get("structure_digest") != structure.get("structure_digest"):
        raise RuntimeError("independent structure audit did not authorize causal adjudication")
    if structure.get("structure_gate") is not True:
        value: dict[str, Any] = {
            "phase": PHASE,
            "schema_version": "phase1236.cross_protocol_interchange.v1",
            "created_at_utc": utc_now(),
            "contract_digest": contract["contract_digest"],
            "status": "denied_by_sealed_structure_gate",
            "intervention_performed": False,
            "causal_gate": False,
        }
        value["causal_digest"] = digest(value)
        write_json(CAUSAL_PATH, value)
        print(canonical_json({"status": "causal_denied"}))
        return
    metadata = read_json(CAPTURE_META_PATH)
    arrays = np.load(CAPTURE_ARRAY_PATH)
    residual = arrays["full_residual"].astype(np.float32)
    pair_meta = metadata["pair_metadata"]
    winner = structure["winner"]
    source = winner["source_protocol"]
    target = winner["target_protocol"]
    depth = int(winner["depth"]) - 1
    family = winner["family"]
    pindex = {name: index for index, name in enumerate(PROTOCOLS)}
    fit_indices = np.array([index for index, row in enumerate(pair_meta) if row["partition"] in ("discovery", "model_selection")], dtype=np.int64)
    sealed_indices = np.array([index for index, row in enumerate(pair_meta) if row["partition"] == "sealed"], dtype=np.int64)
    fitted = fit_response_model(
        family,
        residual[fit_indices, pindex[source], depth],
        residual[fit_indices, pindex[target], depth],
    )
    source_delta = residual[sealed_indices, pindex[source], depth].astype(np.float64)
    target_delta = residual[sealed_indices, pindex[target], depth].astype(np.float64)
    mapped = predict_response(fitted, source_delta)
    shuffled_mapped = predict_response(fitted, deterministic_roll(source_delta))
    pair_keys = [pair_meta[index]["base_pair_id"] for index in sealed_indices]
    random_delta = random_norm_deltas(pair_keys, np.linalg.norm(mapped, axis=1), mapped.shape[1])
    conditions = {
        "zero": np.zeros_like(mapped),
        "oracle_target": target_delta,
        "mapped_source": mapped,
        "raw_source": source_delta,
        "shuffled_mapped": shuffled_mapped,
        "opposite_mapped": -mapped,
        "random_norm": random_delta,
    }
    manifest = manifests["qwen3"]
    manifest_by_id = {row["item_id"]: row for row in manifest}
    material_by_id = {row["item_id"]: row for row in material}
    receiver_rows = []
    gold_pairs = []
    for base_pair_id in pair_keys:
        pair_id = f"{base_pair_id}|{target}"
        material_pair = sorted(
            [row for row in material if row["pair_id"] == pair_id], key=lambda row: row["binding_state"]
        )
        if len(material_pair) != 2:
            raise RuntimeError("sealed causal pair drift")
        receiver_rows.append(manifest_by_id[material_pair[0]["item_id"]])
        gold_pairs.append((material_pair[0]["gold"], material_pair[1]["gold"]))
    model = None
    started = time.time()
    try:
        model, _tokenizer, device, placement = load_fp16("qwen3")
        precision = quantization_audit(model)
        layers = get_layers(model)
        if depth >= len(layers):
            raise RuntimeError("selected depth outside model")
        raw_records = []
        baseline_odds = None
        metrics = {}
        for condition, deltas in conditions.items():
            scores = score_patched_rows(model, device, layers[depth], receiver_rows, deltas)
            condition_records = []
            odds_values = []
            for index, row in enumerate(receiver_rows):
                gold0, gold1 = gold_pairs[index]
                values = scores[row["item_id"]]
                log_odds = values[gold1]["sum_log_probability"] - values[gold0]["sum_log_probability"]
                odds_values.append(log_odds)
                winner_label = unique_winner(values)
                condition_records.append({
                    "base_pair_id": pair_keys[index],
                    "condition": condition,
                    "gold0": gold0,
                    "gold1": gold1,
                    "log_odds_state1_vs_state0": float(log_odds),
                    "winner": winner_label,
                    "finite": all(value["all_vocab_logits_finite"] for value in values.values()),
                })
            odds_array = np.array(odds_values, dtype=np.float64)
            if condition == "zero":
                baseline_odds = odds_array
            if baseline_odds is None:
                raise RuntimeError("zero condition must execute first")
            for index, row in enumerate(condition_records):
                row["log_odds_effect"] = float(odds_array[index] - baseline_odds[index])
                row["row_digest"] = digest(row)
            raw_records.extend(condition_records)
            metrics[condition] = condition_metrics(condition_records)
            print(f"[phase1236/causal] {condition}: {metrics[condition]}", flush=True)
        oracle_gate = bool(
            metrics["oracle_target"]["median_log_odds_effect"] > 0.0
            and metrics["oracle_target"]["positive_effect_fraction"] >= THRESHOLDS["causal_oracle_positive_fraction"]
        )
        baseline_state1 = metrics["zero"]["state1_winner_fraction"]
        mapped_controls = [metrics[name]["median_log_odds_effect"] for name in ("shuffled_mapped", "opposite_mapped", "random_norm")]
        mapped_gate = bool(
            metrics["mapped_source"]["median_log_odds_effect"] > max(0.0, max(mapped_controls))
            and metrics["mapped_source"]["positive_effect_fraction"] >= THRESHOLDS["causal_mapped_positive_fraction"]
            and metrics["mapped_source"]["state1_winner_fraction"] - baseline_state1 >= THRESHOLDS["causal_mapped_state1_gain"]
        )
        value = {
            "phase": PHASE,
            "schema_version": "phase1236.cross_protocol_interchange.v1",
            "created_at_utc": utc_now(),
            "contract_digest": contract["contract_digest"],
            "structure_digest": structure["structure_digest"],
            "status": "complete",
            "intervention_performed": True,
            "selected": {"source_protocol": source, "target_protocol": target, "depth": depth + 1, "family": family},
            "sealed_pair_count": len(pair_keys),
            "condition_metrics": metrics,
            "raw_records": raw_records,
            "instrument_oracle_gate": oracle_gate,
            "mapped_interchange_gate": mapped_gate,
            "causal_gate": oracle_gate and mapped_gate,
            "precision_audit": precision,
            "placement": placement,
            "elapsed_seconds": time.time() - started,
            "claim_boundary": "Controlled Qwen3 residual sufficiency within the frozen synthetic protocol family only.",
        }
        value["causal_digest"] = digest(value)
        write_json(CAUSAL_PATH, value)
        print(canonical_json({"status": "causal_complete", "oracle_gate": oracle_gate, "mapped_gate": mapped_gate}))
    finally:
        if model is not None:
            release_fp16(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def finalize() -> None:
    if FINAL_PATH.exists():
        raise RuntimeError("Phase1236 final already exists")
    contract, _material, _manifests = verify_frozen()
    behavior = read_json(BEHAVIOR_ADJUDICATION_PATH)
    structure = read_json(STRUCTURE_PATH)
    causal = read_json(CAUSAL_PATH)
    behavior_audits = {name: read_json(behavior_audit_path(name)) for name in MODELS}
    all_behavior_audits = all(value.get("all_checks_passed") is True for value in behavior_audits.values())
    structure_gate = structure.get("structure_gate") is True
    causal_gate = causal.get("causal_gate") is True
    if causal.get("intervention_performed") is True and not structure_gate:
        raise RuntimeError("causal execution violated structure gate")
    if structure_gate and causal.get("intervention_performed") is not True:
        raise RuntimeError("authorized causal stage was not executed")
    if causal_gate:
        verdict = "qwen3_controlled_cross_protocol_response_law_supported"
        evidence_level = "E3-QWEN-CONTROLLED-SUFFICIENCY"
    elif structure_gate:
        verdict = "sealed_response_law_predictive_but_causal_interchange_failed"
        evidence_level = "E2-PREDICTIVE-NONCAUSAL"
    elif structure.get("winner") is not None:
        verdict = "selected_response_law_failed_sealed_prediction"
        evidence_level = "E3-NEGATIVE-BOUNDARY"
    elif behavior.get("qwen_hidden_authorized") is True:
        verdict = "no_compact_registered_cross_protocol_response_law"
        evidence_level = "E3-NEGATIVE-BOUNDARY"
    else:
        verdict = "behavior_gate_denied_internal_test"
        evidence_level = "E3-BEHAVIOR-BOUNDARY"
    auto_continue = False
    next_experiment = None
    if causal_gate:
        next_experiment = (
            "A separately preregistered natural-language external-validity stage may test the same response-law family; "
            "it is a new major stage and is not auto-authorized by this synthetic result."
        )
    value: dict[str, Any] = {
        "phase": PHASE,
        "schema_version": "phase1236.final.v1",
        "created_at_utc": utc_now(),
        "contract_digest": contract["contract_digest"],
        "behavior_adjudication_digest": behavior["adjudication_digest"],
        "structure_digest": structure["structure_digest"],
        "causal_digest": causal["causal_digest"],
        "all_behavior_audits_passed": all_behavior_audits,
        "typed_gates": {
            "qwen_hidden_authorized": behavior["qwen_hidden_authorized"],
            "cross_model_behavior_authorized": behavior["cross_model_behavior_authorized"],
            "sealed_structure_gate": structure_gate,
            "causal_interchange_gate": causal_gate,
        },
        "verdict": verdict,
        "evidence_level": evidence_level,
        "new_core_puzzle": "K211" if structure_gate or causal_gate else None,
        "auto_continue": auto_continue,
        "next_experiment": next_experiment,
        "registry_status": "closed_after_one_sealed_adjudication",
        "claims": {
            "pi_rho_kappa_separated_behaviorally": True,
            "protocol_independent_content_state_proven": causal_gate,
            "cross_model_physical_identity_proven": False,
            "natural_language_encoding_cracked": False,
            "new_mathematics_required": False,
        },
        "hard_limits": [
            "Synthetic registry materials and one operation family.",
            "Only Qwen3 internal states and interventions are tested.",
            "Residual interchange establishes sufficiency only if the oracle instrument and mapped controls pass.",
            "A predictive response map need not be unique, necessary, minimal, or naturally used.",
            "No neuron, head, or semantic module is localized in this phase.",
        ],
    }
    value["final_digest"] = digest(value)
    write_json(FINAL_PATH, value)
    print(canonical_json({"status": "phase1236_final", "verdict": verdict, "final_digest": value["final_digest"]}))


def selftest() -> None:
    fixtures = parser_fixtures()
    x = np.arange(96, dtype=np.float64).reshape(8, 12) / 10.0
    y = 1.7 * x + 0.3
    model = fit_response_model("affine_scalar", x[:4], y[:4])
    predicted = predict_response(model, x[4:])
    checks = {
        "parser_fixtures": all(row["pass"] for row in fixtures),
        "affine_scalar_recovery": bool(np.max(np.abs(predicted - y[4:])) < 1e-8),
        "digest_stable": digest({"b": 2, "a": 1}) == digest({"a": 1, "b": 2}),
        "partition_cardinality": WORLD_COUNT == WORLDS_PER_PARTITION * len(PARTITIONS),
        "expected_rows": EXPECTED_ROWS == 1152,
    }
    if not all(checks.values()):
        raise RuntimeError(f"selftest failed: {checks}")
    print(canonical_json({"status": "selftest_passed", "checks": checks}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        required=True,
        choices=("selftest", "preregister", "run-behavior", "adjudicate-behavior", "capture-qwen", "fit-structures", "causal-qwen", "finalize"),
    )
    parser.add_argument("--model", choices=MODELS)
    args = parser.parse_args()
    if args.stage == "selftest":
        selftest()
    elif args.stage == "preregister":
        preregister()
    elif args.stage == "run-behavior":
        if args.model is None:
            raise SystemExit("--model is required for run-behavior")
        run_behavior(args.model)
    elif args.stage == "adjudicate-behavior":
        adjudicate_behavior()
    elif args.stage == "capture-qwen":
        capture_qwen()
    elif args.stage == "fit-structures":
        fit_structures()
    elif args.stage == "causal-qwen":
        causal_qwen()
    elif args.stage == "finalize":
        finalize()


if __name__ == "__main__":
    main()
