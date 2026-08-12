#!/usr/bin/env python3
"""Freeze Phase1126 sense-inventory-free natural lexical-substitution protocol."""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
import urllib.request
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests" / "glm5"
sys.path.insert(0, str(TEST_ROOT))

from phase1018_language_pattern_protocol import tokenizer_for


PHASE = 1126
PROTOCOL_REVISION = 3
MODELS = ("qwen3", "glm4", "deepseek7b")
PRECISION = "fp16"
QUANTIZATION = "none"
OUT_ROOT = ROOT / "tests" / "glm5" / "result" / "phase1126_semeval_lexsub_natural_cloze"
SOURCE_ROOT = OUT_ROOT / "protocol" / "source"
PARTITIONS = ("discovery", "independent_confirmation", "hidden_holdout")
BEHAVIOR_PARTITIONS = PARTITIONS[:2]
POS = ("n", "a", "r")
PANELS_PER_POS = 10
PANELS_PER_PARTITION = 10
REPLICAS = (0, 1)
ROUTES = ("active", "matched_deranged")
SENSES = ("a", "b")
CANDIDATE_SIDES = ("a", "b")
MAX_SEQUENCE_TOKENS = 256
RIGHT_CONTEXT_TOKENS = 8
SELECTION_SEED = 1126001


SOURCE_SPECS = {
    "lexsub_test.xml": {
        "url": "https://ltdata1.informatik.uni-hamburg.de/lexsub2016/Tasks/SemEval2007/test/lexsub_test.xml",
        "sha256": "4ac652e8f58d35a63c516ab5d025581ad1e7e325af8cd7b3a1935fa2f92bd134",
        "bytes": 386624,
    },
    "gold.gold": {
        "url": "https://ltdata1.informatik.uni-hamburg.de/lexsub2016/Tasks/SemEval2007/test/gold.gold",
        "sha256": "4e27e4abac4e60b09547bbffeef10d0ba00c71118a2cbee76985aa604f06088b",
        "bytes": 100444,
    },
    "readme": {
        "url": "https://ltdata1.informatik.uni-hamburg.de/lexsub2016/Tasks/SemEval2007/test/readme",
        "sha256": "45deb2337b23f9692465ab9ebafe991ef22da3d8de18fcca3ddc2ccf78e21eba",
        "bytes": 1985,
    },
}

PAPER = {
    "title": "SemEval-2007 Task 10: English Lexical Substitution Task",
    "authors": "Diana McCarthy and Roberto Navigli",
    "url": "https://aclanthology.org/S07-1009/",
    "anthology_id": "S07-1009",
    "annotation_note": "The paper reports five native-English annotators and no predefined sense inventory.",
}

THRESHOLDS = {
    "finite_rate_min": 0.99,
    "active_positive_rate_min": 0.70,
    "active_median_min": 0.05,
    "matched_advantage_median_min": 0.05,
    "matched_advantage_positive_rate_min": 0.60,
    "lexical_zero_advantage_min": 0.05,
    "models_required": 2,
}

WORD_RE = re.compile(r"^[a-z]+$")


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def fetch_sources() -> dict[str, dict[str, Any]]:
    SOURCE_ROOT.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict[str, Any]] = {}
    for name, spec in SOURCE_SPECS.items():
        path = SOURCE_ROOT / name
        if not path.exists():
            raw = urllib.request.urlopen(spec["url"], timeout=120).read()
            path.write_bytes(raw)
        actual = {"sha256": file_sha256(path), "bytes": path.stat().st_size}
        if actual["sha256"] != spec["sha256"] or actual["bytes"] != spec["bytes"]:
            raise RuntimeError(f"source identity mismatch: {name}: {actual}")
        manifest[name] = {
            **spec,
            "path": str(path.relative_to(OUT_ROOT)).replace("\\", "/"),
        }
    return manifest


def parse_gold(path: Path) -> dict[int, dict[str, int]]:
    result: dict[int, dict[str, int]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        left, right = line.split(" :: ", 1)
        instance_id = int(left.rsplit(" ", 1)[1])
        counts: dict[str, int] = {}
        for item in right.split(";"):
            item = item.strip()
            if not item:
                continue
            value, count = item.rsplit(" ", 1)
            counts[value.lower()] = int(count)
        result[instance_id] = counts
    return result


def parse_xml(path: Path) -> dict[str, list[dict[str, Any]]]:
    root = ET.fromstring(path.read_bytes())
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for lexelt in root.findall("lexelt"):
        item = lexelt.attrib["item"]
        lemma, pos = item.rsplit(".", 1)
        for instance in lexelt.findall("instance"):
            context = instance.find("context")
            if context is None:
                continue
            head = context.find("head")
            if head is None or head.text is None:
                continue
            prefix = context.text or ""
            suffix = head.tail or ""
            result[item].append({
                "instance_id": int(instance.attrib["id"]),
                "lemma": lemma.lower(),
                "pos": pos,
                "head": head.text,
                "prefix": prefix,
                "suffix": suffix,
                "sentence": f"{prefix}{head.text}{suffix}",
            })
    return result


def stable_key(value: str) -> str:
    return hashlib.sha256(f"{SELECTION_SEED}:{value}".encode("utf-8")).hexdigest()


def char_trigrams(text: str) -> set[str]:
    normalized = re.sub(r"[^a-z]", "", text.lower())
    if len(normalized) < 3:
        return {normalized} if normalized else set()
    return {normalized[index:index + 3] for index in range(len(normalized) - 2)}


def lexical_overlap(candidate: str, context: str) -> float:
    candidate_grams = char_trigrams(candidate)
    context_grams = char_trigrams(context)
    if not candidate_grams:
        return 0.0
    return len(candidate_grams & context_grams) / len(candidate_grams)


def build_candidate_panels(
    instances: dict[str, list[dict[str, Any]]],
    gold: dict[int, dict[str, int]],
    tokenizers: dict[str, Any],
) -> list[dict[str, Any]]:
    def common_single_token(word: str) -> bool:
        if not WORD_RE.fullmatch(word):
            return False
        return all(len(tokenizer.encode(" " + word, add_special_tokens=False)) == 1 for tokenizer in tokenizers.values())

    candidates: list[dict[str, Any]] = []
    for item, rows in sorted(instances.items()):
        lemma, pos = item.rsplit(".", 1)
        if pos not in POS:
            continue
        clean_rows = []
        for row in rows:
            if row["head"].lower() != lemma.lower():
                continue
            word_count = len(re.findall(r"[A-Za-z]+", row["sentence"]))
            if not 8 <= word_count <= 100:
                continue
            clean_rows.append(row)
        values = sorted({
            value
            for row in clean_rows
            for value, count in gold.get(row["instance_id"], {}).items()
            if count >= 2 and value != lemma and common_single_token(value)
        })
        best = None
        for index, candidate_a in enumerate(values):
            for candidate_b in values[index + 1:]:
                context_pattern = re.compile(
                    rf"\b{re.escape(candidate_a)}\b|\b{re.escape(candidate_b)}\b",
                    re.IGNORECASE,
                )
                a_rows = [
                    row for row in clean_rows
                    if gold.get(row["instance_id"], {}).get(candidate_a, 0) >= 2
                    and gold.get(row["instance_id"], {}).get(candidate_b, 0) == 0
                    and not context_pattern.search(row["prefix"] + row["suffix"])
                ]
                b_rows = [
                    row for row in clean_rows
                    if gold.get(row["instance_id"], {}).get(candidate_b, 0) >= 2
                    and gold.get(row["instance_id"], {}).get(candidate_a, 0) == 0
                    and not context_pattern.search(row["prefix"] + row["suffix"])
                ]
                if len(a_rows) < 2 or len(b_rows) < 2:
                    continue
                a_rows.sort(key=lambda row: (-gold[row["instance_id"]][candidate_a], row["instance_id"]))
                b_rows.sort(key=lambda row: (-gold[row["instance_id"]][candidate_b], row["instance_id"]))
                strength = sum(gold[row["instance_id"]][candidate_a] for row in a_rows[:2])
                strength += sum(gold[row["instance_id"]][candidate_b] for row in b_rows[:2])
                key = (strength, min(len(a_rows), len(b_rows)), stable_key(f"{item}:{candidate_a}:{candidate_b}"))
                if best is None or key > best[0]:
                    best = (key, candidate_a, candidate_b, a_rows[:2], b_rows[:2])
        if best is None:
            continue
        _, candidate_a, candidate_b, a_rows, b_rows = best
        candidates.append({
            "item": item,
            "lemma": lemma,
            "pos": pos,
            "candidate_a": candidate_a,
            "candidate_b": candidate_b,
            "a_rows": a_rows,
            "b_rows": b_rows,
            "annotation_strength": best[0][0],
        })

    selected: list[dict[str, Any]] = []
    used_terms: set[str] = set()
    for pos in POS:
        panel = [row for row in candidates if row["pos"] == pos]
        panel.sort(key=lambda row: (-row["annotation_strength"], stable_key(row["item"])))
        chosen = []
        for row in panel:
            terms = {row["lemma"], row["candidate_a"], row["candidate_b"]}
            if terms & used_terms:
                continue
            chosen.append(row)
            used_terms |= terms
            if len(chosen) == PANELS_PER_POS:
                break
        if len(chosen) != PANELS_PER_POS:
            raise RuntimeError(f"insufficient disjoint panels for pos={pos}: {len(chosen)}")
        selected.extend(chosen)
    return selected


def assign_partitions(panels: list[dict[str, Any]]) -> None:
    counts = {
        "n": (4, 3, 3),
        "a": (3, 4, 3),
        "r": (3, 3, 4),
    }
    for pos in POS:
        rows = sorted((row for row in panels if row["pos"] == pos), key=lambda row: stable_key(row["item"]))
        cursor = 0
        for partition, count in zip(PARTITIONS, counts[pos]):
            for row in rows[cursor:cursor + count]:
                row["partition"] = partition
            cursor += count
    for partition in PARTITIONS:
        for pos in POS:
            group = sorted(
                (row for row in panels if row["partition"] == partition and row["pos"] == pos),
                key=lambda row: stable_key(row["item"]),
            )
            if len(group) < 2:
                raise RuntimeError(f"derangement group too small: {partition}/{pos}")
            for index, row in enumerate(group):
                row["deranged_item"] = group[(index + 1) % len(group)]["item"]


def tokenized_case(tokenizer: Any, prefix: str, candidate: str, suffix: str) -> dict[str, Any]:
    document_prefix = "\n"
    text = f"{document_prefix}{prefix}{candidate}{suffix}"
    start = len(document_prefix) + len(prefix)
    end = start + len(candidate)
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    input_ids = [int(value) for value in encoded["input_ids"]]
    offsets = [(int(a), int(b)) for a, b in encoded["offset_mapping"]]
    candidate_positions = [
        index for index, (left, right) in enumerate(offsets)
        if right > start and left < end and right > left
    ]
    if not candidate_positions or candidate_positions[0] == 0:
        raise ValueError(f"candidate span is not scoreable: {candidate!r}")
    suffix_positions = [
        index for index, (left, right) in enumerate(offsets)
        if left >= end and right > left
    ][:RIGHT_CONTEXT_TOKENS]
    if len(input_ids) > MAX_SEQUENCE_TOKENS:
        raise ValueError(f"sequence too long: {len(input_ids)}")
    return {
        "input_ids": input_ids,
        "candidate_positions": candidate_positions,
        "suffix_positions": suffix_positions,
        "sequence_tokens": len(input_ids),
    }


def main() -> None:
    manifest = fetch_sources()
    gold = parse_gold(SOURCE_ROOT / "gold.gold")
    instances = parse_xml(SOURCE_ROOT / "lexsub_test.xml")
    tokenizers = {model: tokenizer_for(model) for model in MODELS}
    panels = build_candidate_panels(instances, gold, tokenizers)
    assign_partitions(panels)
    panel_by_item = {panel["item"]: panel for panel in panels}

    material_rows = []
    cases_by_model: dict[str, list[dict[str, Any]]] = {model: [] for model in MODELS}
    tokenization_failures = []
    for panel_index, panel in enumerate(sorted(panels, key=lambda row: (PARTITIONS.index(row["partition"]), row["pos"], row["item"]))):
        panel["panel_index"] = panel_index
        deranged = panel_by_item[panel["deranged_item"]]
        material_rows.append({
            "panel_index": panel_index,
            "item": panel["item"],
            "lemma": panel["lemma"],
            "pos": panel["pos"],
            "partition": panel["partition"],
            "candidate_a": panel["candidate_a"],
            "candidate_b": panel["candidate_b"],
            "deranged_item": panel["deranged_item"],
            "deranged_candidate_a": deranged["candidate_a"],
            "deranged_candidate_b": deranged["candidate_b"],
            "annotation_strength": panel["annotation_strength"],
            "replicas": [
                {
                    "replica": replica,
                    "sense_a_instance": panel["a_rows"][replica]["instance_id"],
                    "sense_b_instance": panel["b_rows"][replica]["instance_id"],
                    "sense_a_gold": gold[panel["a_rows"][replica]["instance_id"]][panel["candidate_a"]],
                    "sense_b_gold": gold[panel["b_rows"][replica]["instance_id"]][panel["candidate_b"]],
                    "sense_a_context_sha256": digest(panel["a_rows"][replica]["sentence"]),
                    "sense_b_context_sha256": digest(panel["b_rows"][replica]["sentence"]),
                }
                for replica in REPLICAS
            ],
        })
        if panel["partition"] not in BEHAVIOR_PARTITIONS:
            continue
        for model_name, tokenizer in tokenizers.items():
            for replica in REPLICAS:
                source_rows = {"a": panel["a_rows"][replica], "b": panel["b_rows"][replica]}
                for route in ROUTES:
                    route_panel = panel if route == "active" else deranged
                    route_candidates = {"a": route_panel["candidate_a"], "b": route_panel["candidate_b"]}
                    for sense in SENSES:
                        source_row = source_rows[sense]
                        context_without_head = source_row["prefix"] + source_row["suffix"]
                        for candidate_side in CANDIDATE_SIDES:
                            candidate = route_candidates[candidate_side]
                            try:
                                encoded = tokenized_case(
                                    tokenizer,
                                    source_row["prefix"],
                                    candidate,
                                    source_row["suffix"],
                                )
                            except ValueError as exc:
                                tokenization_failures.append({
                                    "model": model_name,
                                    "item": panel["item"],
                                    "route": route,
                                    "candidate": candidate,
                                    "error": str(exc),
                                })
                                continue
                            cases_by_model[model_name].append({
                                "case_index": len(cases_by_model[model_name]),
                                "panel_index": panel_index,
                                "item": panel["item"],
                                "pos": panel["pos"],
                                "partition": panel["partition"],
                                "replica": replica,
                                "route": route,
                                "route_item": route_panel["item"],
                                "context_sense": sense,
                                "candidate_side": candidate_side,
                                "candidate": candidate,
                                "source_instance_id": source_row["instance_id"],
                                "lexical_overlap": lexical_overlap(candidate, context_without_head),
                                **encoded,
                            })

    expected_case_count = len(BEHAVIOR_PARTITIONS) * PANELS_PER_PARTITION * len(REPLICAS) * len(ROUTES) * len(SENSES) * len(CANDIDATE_SIDES)
    partition_counts = {
        partition: sum(row["partition"] == partition for row in material_rows)
        for partition in PARTITIONS
    }
    pos_counts = {pos: sum(row["pos"] == pos for row in material_rows) for pos in POS}
    source_history = ROOT / "tests" / "glm5" / "result" / "phase1121_wordnet_adjective_double_orthogonal" / "protocol" / "selected_concepts.json"
    historical_terms: set[str] = set()
    if source_history.exists():
        previous = read_json(source_history)
        for row in previous.get("selected", []):
            historical_terms.add(row["base"].lower())
            historical_terms.update(value.lower() for value in row.get("synonym_surfaces", []))
    current_terms = {
        value
        for row in material_rows
        for value in (row["lemma"], row["candidate_a"], row["candidate_b"])
    }

    preregistration = {
        "schema_version": "phase1126_semeval_lexsub_natural_cloze.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "objective": (
            "Test whether a public human-annotated, sense-inventory-free natural lexical-substitution "
            "interaction repeats beyond the Phase1121 Princeton WordNet material."
        ),
        "models": list(MODELS),
        "precision": PRECISION,
        "quantization": QUANTIZATION,
        "source": {"files": manifest, "paper": PAPER},
        "material": {
            "positions_of_speech": list(POS),
            "panels_per_pos": PANELS_PER_POS,
            "panels_per_partition": PANELS_PER_PARTITION,
            "partitions": list(PARTITIONS),
            "behavior_partitions": list(BEHAVIOR_PARTITIONS),
            "hidden_holdout_is_unscored": True,
            "gold_support_min": 2,
            "opposite_candidate_support_required": 0,
            "head_must_equal_lemma": True,
            "candidate_must_be_one_leading_space_token_in_all_models": True,
            "actual_context_candidate_span_may_contain_multiple_tokens": True,
            "global_target_and_candidate_terms_are_disjoint": True,
            "candidate_word_must_not_occur_elsewhere_in_selected_context": True,
            "selection_seed": SELECTION_SEED,
        },
        "score": {
            "candidate_log_probability": "sum log p(candidate span tokens | natural left context and earlier candidate tokens)",
            "suffix_mean_log_probability": f"mean log p of up to {RIGHT_CONTEXT_TOKENS} following tokens after substitution",
            "total": "candidate_log_probability + suffix_mean_log_probability",
            "interaction": "0.5 * ((S(A,a)-S(A,b)) + (S(B,b)-S(B,a)))",
            "candidate_prior_cancellation": "the same two candidates enter both context sides with opposite signs",
            "document_boundary": "one fixed leading newline makes sentence-initial target likelihood scoreable",
            "matched_null": "same natural contexts scored with a deterministic same-POS candidate pair from another target",
            "lexical_zero": "candidate-character-trigram coverage in context outside the target",
        },
        "thresholds": THRESHOLDS,
        "predictions": {
            "P1": "source, partition, tokenization, precision, and finite-score audits pass",
            "P2": "at least two models pass every behavior gate in discovery and independent confirmation",
            "P3": "P2 authorizes a separately frozen hidden-use protocol on hidden_holdout; it does not authorize causal claims",
            "P4": "failure of P2 stops hidden scanning and keeps K57 resource-bound",
        },
        "forbidden": [
            "no threshold changes after model scores",
            "no hidden_holdout behavior or hidden scan in Phase1126",
            "no same-material attention/head/SAE hotspot search",
            "no interpretation of natural-cloze interaction as a hidden mechanism",
            "no claim that the benchmark has no historical lexical-resource influence",
        ],
        "case_digests": {model: digest(rows) for model, rows in cases_by_model.items()},
        "material_digest": digest(material_rows),
    }
    preregistration["protocol_digest"] = digest(preregistration)

    checks = {
        "source_files_match_frozen_hashes": all(
            file_sha256(SOURCE_ROOT / name) == spec["sha256"]
            for name, spec in SOURCE_SPECS.items()
        ),
        "selected_panel_count_is_30": len(material_rows) == len(POS) * PANELS_PER_POS,
        "partition_counts_are_10_each": all(value == PANELS_PER_PARTITION for value in partition_counts.values()),
        "pos_counts_are_10_each": all(value == PANELS_PER_POS for value in pos_counts.values()),
        "no_tokenization_failures": not tokenization_failures,
        "case_counts_match": all(len(rows) == expected_case_count for rows in cases_by_model.values()),
        "case_indices_are_contiguous": all(
            [row["case_index"] for row in rows] == list(range(len(rows)))
            for rows in cases_by_model.values()
        ),
        "all_sequences_within_limit": all(
            row["sequence_tokens"] <= MAX_SEQUENCE_TOKENS
            for rows in cases_by_model.values() for row in rows
        ),
        "all_candidate_positions_are_scored": all(
            row["candidate_positions"] and min(row["candidate_positions"]) > 0
            for rows in cases_by_model.values() for row in rows
        ),
        "hidden_holdout_has_no_cases": all(
            row["partition"] in BEHAVIOR_PARTITIONS
            for rows in cases_by_model.values() for row in rows
        ),
        "terms_are_globally_disjoint": len(current_terms) == 3 * len(material_rows),
        "derangements_change_item": all(row["item"] != row["deranged_item"] for row in material_rows),
    }
    audit = {
        "schema_version": "phase1126_semeval_lexsub_natural_cloze_protocol_audit.v1",
        "phase": PHASE,
        "protocol_digest": preregistration["protocol_digest"],
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "partition_counts": partition_counts,
        "pos_counts": pos_counts,
        "expected_cases_per_model": expected_case_count,
        "actual_cases_per_model": {model: len(rows) for model, rows in cases_by_model.items()},
        "tokenization_failures": tokenization_failures,
        "history_overlap_terms": sorted(current_terms & historical_terms),
        "history_overlap_count": len(current_terms & historical_terms),
    }
    if not audit["all_checks_passed"]:
        raise RuntimeError(json.dumps(audit, indent=2, ensure_ascii=False))

    write_json(OUT_ROOT / "protocol" / "selected_panels.json", {
        "panels": material_rows,
        "material_digest": preregistration["material_digest"],
    })
    for model_name, rows in cases_by_model.items():
        write_jsonl(OUT_ROOT / "protocol" / f"cases.{model_name}.jsonl", rows)
    write_json(OUT_ROOT / "protocol" / "preregistration.json", preregistration)
    write_json(OUT_ROOT / "protocol" / "audit.json", audit)
    print(json.dumps({
        "protocol_digest": preregistration["protocol_digest"],
        "material_digest": preregistration["material_digest"],
        "audit": audit,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
