#!/usr/bin/env python3
"""Freeze a matched FoodOn/WordNet/evidence fruit-membership denominator."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import re
import sys
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from phase548_shared_attention_compute_protocol import tokenizer_for  # noqa: E402
import phase601_foodon_public_ontology_protocol as foodon  # noqa: E402
from phase602_download_wordnet import (  # noqa: E402
    ARCHIVE_PATH,
    DATA_NOUN_PATH,
    EXPECTED_SHA256 as WORDNET_SHA256,
    EXPECTED_SIZE as WORDNET_SIZE,
    LICENSE_PATH,
    SOURCE_URL as WORDNET_URL,
    download as download_wordnet,
)


PHASE = "Phase602"
SCHEMA_VERSION = "phase602_three_track_semantics.v1"
FROZEN_AT = "2026-07-22T13:00:00+00:00"
MODELS = ("qwen3", "glm4", "deepseek7b")
TRACKS = ("technical", "daily", "explicit_evidence")
SPLITS = ("discovery", "independent_confirmation", "heldout")
ENTITY_ROLES = ("raw_fruit", "seed_vegetable", "meat", "dairy", "seafood")
ROLE_QUOTAS = {"raw_fruit": 60, "seed_vegetable": 10, "meat": 25, "dairy": 21, "seafood": 4}
ROLE_SPLIT_QUOTAS = {
    "raw_fruit": (30, 15, 15),
    "seed_vegetable": (4, 3, 3),
    "meat": (13, 6, 6),
    "dairy": (11, 5, 5),
    "seafood": (2, 1, 1),
}
WORDNET_ROOTS = {
    "raw_fruit": ("07705931",),
    "nut": ("13136556",),
    "root_vegetable": ("07710283",),
    "seed_vegetable": ("07708798", "07770571"),
    "meat": ("07649854",),
    "seafood": ("07776866",),
    "egg": ("07840804",),
    "dairy": ("07843775",),
}
ROLE_TO_FOODON_FAMILY = {
    "raw_fruit": "fruit",
    "nut": "nut",
    "root_vegetable": "root_vegetable",
    "seed_vegetable": "seed_vegetable",
    "meat": "animal_food",
    "seafood": "animal_food",
    "egg": "animal_food",
    "dairy": "animal_food",
}
SURFACE_STEMS = (
    "Is {concept} a member of the edible-fruit category?",
    "Should {concept} be classified as an edible fruit?",
    "Does the category edible fruit include {concept}?",
    "In this classification, is {concept} an edible fruit?",
)
ANSWER_CONTINUATIONS = {"A": "A", "B": "B"}
FIXED_BATCH_SIZE = 32

# Frozen before any Phase602 model output is read. Each track qualifies independently.
TRACK_GATES = {
    "technical": {
        "overall_accuracy_min": 0.85,
        "split_accuracy_min": 0.80,
        "membership_accuracy_min": 0.80,
        "surface_accuracy_min": 0.80,
        "answer_order_gap_max": 0.08,
        "concept_unanimous_rate_min": 0.60,
        "direct_candidate_output_rate_min": 0.80,
        "direct_exact_accuracy_min": 0.70,
    },
    "daily": {
        "overall_accuracy_min": 0.85,
        "split_accuracy_min": 0.80,
        "membership_accuracy_min": 0.80,
        "surface_accuracy_min": 0.80,
        "answer_order_gap_max": 0.08,
        "concept_unanimous_rate_min": 0.60,
        "direct_candidate_output_rate_min": 0.80,
        "direct_exact_accuracy_min": 0.70,
    },
    "explicit_evidence": {
        "overall_accuracy_min": 0.95,
        "split_accuracy_min": 0.92,
        "membership_accuracy_min": 0.92,
        "surface_accuracy_min": 0.92,
        "answer_order_gap_max": 0.05,
        "concept_unanimous_rate_min": 0.85,
        "direct_candidate_output_rate_min": 0.90,
        "direct_exact_accuracy_min": 0.85,
    },
}

OUT_DIR = ROOT / "tests/gpt5/result/phase602_three_track_semantics"
CASES_PATH = OUT_DIR / "phase602_registered_cases.jsonl.gz"
PROTOCOL_PATH = OUT_DIR / "phase602_frozen_protocol.json"
AUDIT_PATH = OUT_DIR / "phase602_static_audit.json"
SOURCE_MANIFEST_PATH = OUT_DIR / "phase602_source_manifest.json"


def stable_hash(*parts: object) -> str:
    return hashlib.sha256("\x1f".join(str(part) for part in parts).encode()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def normalized_label(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def parse_wordnet(path: Path) -> tuple[dict[str, list[str]], dict[str, set[str]], int]:
    words: dict[str, list[str]] = {}
    children: dict[str, set[str]] = defaultdict(set)
    pointer_count = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line or line[0].isspace():
                continue
            fields = line.split("|", 1)[0].split()
            offset = fields[0]
            word_count = int(fields[3], 16)
            cursor = 4
            words[offset] = [fields[cursor + 2 * index].replace("_", " ") for index in range(word_count)]
            cursor += 2 * word_count
            relation_count = int(fields[cursor])
            cursor += 1
            for _index in range(relation_count):
                symbol, target, pos, _source_target = fields[cursor : cursor + 4]
                cursor += 4
                if symbol == "@" and pos == "n":
                    children[target].add(offset)
                    pointer_count += 1
    return words, children, pointer_count


def descendants(children: dict[str, set[str]], roots: tuple[str, ...]) -> set[str]:
    output = set(roots)
    queue = deque(roots)
    while queue:
        for child in children[queue.popleft()]:
            if child not in output:
                output.add(child)
                queue.append(child)
    return output


def previous_clusters() -> set[str]:
    clusters: set[str] = set()
    with gzip.open(foodon.CASES_PATH, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                clusters.add(json.loads(line)["cluster_key"])
    return clusters


def shortest_parent_path(parents: dict[str, set[str]], start: str, root: str) -> list[str]:
    queue = deque([(start, [start])])
    seen = {start}
    while queue:
        node, path = queue.popleft()
        if node == root:
            return path
        for parent in sorted(parents.get(node, set())):
            if parent not in seen:
                seen.add(parent)
                queue.append((parent, path + [parent]))
    raise RuntimeError(f"No named subclass path from {start} to {root}")


def candidate_inventory() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    labels, parents, foodon_class_count = foodon.parse_foodon(foodon.SOURCE_PATH)
    distances = {
        family: foodon.descendant_distances(parents, root)
        for family, root in foodon.ROOT_IDS.items()
    }
    foodon_sets = {family: set(values) for family, values in distances.items()}
    exclusive = {
        family: values - set().union(*(foodon_sets[other] for other in foodon_sets if other != family))
        for family, values in foodon_sets.items()
    }
    words, children, pointer_count = parse_wordnet(DATA_NOUN_PATH)
    wordnet_sets = {role: descendants(children, roots) for role, roots in WORDNET_ROOTS.items()}
    lemma_roles: dict[str, set[str]] = defaultdict(set)
    for role, synsets in wordnet_sets.items():
        for synset in synsets:
            for lemma in words[synset]:
                lemma_roles[normalized_label(lemma)].add(role)
    used_clusters = previous_clusters()
    output: list[dict[str, Any]] = []
    raw_counts: Counter[str] = Counter()
    novel_counts: Counter[str] = Counter()
    for concept_id, label in labels.items():
        technical = [family for family, values in exclusive.items() if concept_id in values]
        roles = lemma_roles.get(normalized_label(label), set())
        if len(technical) != 1 or len(roles) != 1:
            continue
        role = next(iter(roles))
        family = technical[0]
        if ROLE_TO_FOODON_FAMILY[role] != family:
            continue
        raw_counts[role] += 1
        cluster = foodon.lexical_cluster(label)
        if cluster in used_clusters:
            continue
        novel_counts[role] += 1
        root_id = foodon.ROOT_IDS[family]
        path_ids = shortest_parent_path(parents, concept_id, root_id)
        output.append({
            "concept_id": concept_id,
            "concept_label": label,
            "normalized_label": normalized_label(label),
            "cluster_key": cluster,
            "entity_role": role,
            "foodon_family": family,
            "foodon_root_id": root_id,
            "foodon_root_label": labels[root_id],
            "foodon_path_ids": path_ids,
            "foodon_path_labels": [labels.get(value, value) for value in path_ids],
            "fruit_member": role == "raw_fruit",
            "foodon_wordnet_binary_agreement": True,
            "phase601_cluster_novel": True,
        })
    audit = {
        "foodon_class_element_count": foodon_class_count,
        "wordnet_noun_synset_count": len(words),
        "wordnet_hypernym_pointer_count": pointer_count,
        "phase601_used_cluster_count": len(used_clusters),
        "exact_agreement_count_before_cluster_exclusion": dict(raw_counts),
        "novel_exact_agreement_count_by_role": dict(novel_counts),
        "full_five_family_matched_panel_feasible": False,
        "full_five_family_failure_reason": "Only 4 novel exact nut matches and 2 root-vegetable matches remain.",
    }
    return output, audit


def select_concepts(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for role in ENTITY_ROLES:
        values = sorted(
            (dict(row) for row in candidates if row["entity_role"] == role),
            key=lambda row: stable_hash("select", role, row["cluster_key"]),
        )
        quota = ROLE_QUOTAS[role]
        if len(values) < quota:
            raise RuntimeError(f"Not enough novel exact {role} concepts: {len(values)} < {quota}")
        values = values[:quota]
        cursor = 0
        for split, split_quota in zip(SPLITS, ROLE_SPLIT_QUOTAS[role]):
            for row in values[cursor : cursor + split_quota]:
                row["split"] = split
                selected.append(row)
            cursor += split_quota
    return selected


def options(target_yes: bool, surface_index: int) -> tuple[str, str, str]:
    target = "A" if surface_index % 2 == 0 else "B"
    true_text = "yes" if target_yes else "no"
    false_text = "no" if target_yes else "yes"
    return target, true_text if target == "A" else false_text, false_text if target == "A" else true_text


def prompt_for(track: str, concept: dict[str, Any], surface_index: int, a: str, b: str) -> str:
    question = SURFACE_STEMS[surface_index].format(concept=concept["concept_label"])
    if track == "technical":
        context = (
            "Use the FoodOn v2025-02-01 named-class hierarchy for this registered test, "
            "not ordinary everyday usage."
        )
    elif track == "daily":
        context = "Use ordinary English noun meaning as represented by the WordNet 3.0 noun hierarchy."
    else:
        path = " -> ".join(concept["foodon_path_labels"])
        context = (
            "Use only the following registered evidence. Every arrow means 'is a subclass of'. "
            "The five registered FoodOn roots are mutually exclusive for this item. "
            f"Named-class path: {path}."
        )
    return f"{context}\n{question}\nA. {a}\nB. {b}\nAnswer with A or B only."


def build_cases(selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for concept in selected:
        for track in TRACKS:
            for surface_index in range(len(SURFACE_STEMS)):
                target, a, b = options(concept["fruit_member"], surface_index)
                rows.append({
                    **concept,
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": PHASE,
                    "created_at": FROZEN_AT,
                    "case_id": f"phase602_{track}_{concept['concept_id']}_s{surface_index}",
                    "track": track,
                    "surface_id": f"surface_{surface_index}",
                    "surface_index": surface_index,
                    "option_a": a,
                    "option_b": b,
                    "target_letter": target,
                    "foil_letter": "B" if target == "A" else "A",
                    "raw_prompt": prompt_for(track, concept, surface_index, a, b),
                    "public_truth_sources": ["FoodOn v2025-02-01", "WordNet 3.0"],
                    "teacher_forced_internal_states_collected": False,
                    "causal_evidence": False,
                })
    return sorted(rows, key=lambda row: (SPLITS.index(row["split"]), row["case_id"]))


def freeze() -> dict[str, Any]:
    wordnet_source = download_wordnet()
    foodon.validate_source(foodon.SOURCE_PATH)
    candidates, inventory_audit = candidate_inventory()
    selected = select_concepts(candidates)
    cases = build_cases(selected)
    token_ledger: dict[str, dict[str, list[int]]] = {}
    for model in MODELS:
        tokenizer = tokenizer_for(model)
        token_ledger[model] = {
            letter: [int(value) for value in tokenizer.encode(text, add_special_tokens=False)]
            for letter, text in ANSWER_CONTINUATIONS.items()
        }
        if any(len(ids) != 1 for ids in token_ledger[model].values()):
            raise RuntimeError(f"Phase602 requires one-token A/B continuations for {model}")
    write_jsonl(CASES_PATH, cases)
    selected_counts = Counter((row["split"], row["entity_role"]) for row in selected)
    audit = {
        "schema_version": "phase602_static_audit.v1",
        "phase_id": PHASE,
        "created_at": FROZEN_AT,
        **inventory_audit,
        "selected_concept_count": len(selected),
        "registered_case_count": len(cases),
        "selected_concept_count_by_split_role": {
            f"{split}/{role}": selected_counts[(split, role)] for split in SPLITS for role in ENTITY_ROLES
        },
        "track_case_count": dict(Counter(row["track"] for row in cases)),
        "membership_case_count": dict(Counter(str(row["fruit_member"]) for row in cases)),
        "target_letter_count": dict(Counter(row["target_letter"] for row in cases)),
        "surface_count": dict(Counter(row["surface_id"] for row in cases)),
        "all_selected_clusters_novel_to_phase601": all(row["phase601_cluster_novel"] for row in selected),
        "selected_cluster_unique": len({row["cluster_key"] for row in selected}) == len(selected),
        "technical_daily_binary_truth_agreement": all(row["foodon_wordnet_binary_agreement"] for row in selected),
        "concept_split_disjoint": all(
            len({row["split"] for row in selected if row["concept_id"] == concept_id}) == 1
            for concept_id in {row["concept_id"] for row in selected}
        ),
        "answer_token_ledger_by_model": token_ledger,
    }
    write_json(AUDIT_PATH, audit)
    source_manifest = {
        "schema_version": "phase602_source_manifest.v1",
        "phase_id": PHASE,
        "created_at": FROZEN_AT,
        "foodon": {
            "url": foodon.SOURCE_URL,
            "size_bytes": foodon.EXPECTED_SIZE,
            "sha256": foodon.EXPECTED_SHA256,
            "license": "CC-BY-4.0",
        },
        "wordnet": {
            **wordnet_source,
            "official_project_url": "https://wordnet.princeton.edu/",
            "license_file_sha256": sha256_file(LICENSE_PATH),
        },
        "matching_rule": "Exact normalized FoodOn label equals an exclusive WordNet lemma role.",
        "phase601_exclusion_rule": "Reject every lexical cluster used by Phase601.",
    }
    write_json(SOURCE_MANIFEST_PATH, source_manifest)
    protocol = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE,
        "created_at": FROZEN_AT,
        "models": list(MODELS),
        "tracks": list(TRACKS),
        "splits": list(SPLITS),
        "entity_roles": list(ENTITY_ROLES),
        "registered_concept_count": len(selected),
        "registered_case_count": len(cases),
        "surface_count_per_concept_track": len(SURFACE_STEMS),
        "cases_sha256": sha256_file(CASES_PATH),
        "source_manifest_sha256": sha256_file(SOURCE_MANIFEST_PATH),
        "static_audit_sha256": sha256_file(AUDIT_PATH),
        "foodon_sha256": foodon.EXPECTED_SHA256,
        "wordnet_archive_size": WORDNET_SIZE,
        "wordnet_archive_sha256": WORDNET_SHA256,
        "answer_token_ledger_by_model": token_ledger,
        "fixed_batch_size": FIXED_BATCH_SIZE,
        "track_gates": TRACK_GATES,
        "constraints": {
            "same_concepts_across_tracks": True,
            "technical_daily_truth_agreement_required": True,
            "exact_lexical_match_only": True,
            "phase601_cluster_reuse_forbidden": True,
            "concept_split_isolation": True,
            "candidate_order_counterbalanced": True,
            "behavior_unread_before_freeze": True,
            "full_five_family_completion_claim_authorized": False,
            "internal_observation_authorized_only_per_qualified_track": True,
            "causal_intervention_authorized": False,
        },
    }
    write_json(PROTOCOL_PATH, protocol)
    return {"protocol": protocol, "audit": audit}


if __name__ == "__main__":
    print(json.dumps(freeze(), ensure_ascii=False, indent=2, sort_keys=True))
