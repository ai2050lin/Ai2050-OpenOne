#!/usr/bin/env python3
"""Freeze the Phase601 FoodOn taxonomy behavior denominator."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from phase548_shared_attention_compute_protocol import tokenizer_for  # noqa: E402
from phase601_download_foodon import (  # noqa: E402
    EXPECTED_SHA256,
    EXPECTED_SIZE,
    SOURCE_PATH,
    SOURCE_URL,
    validate as validate_source,
)


PHASE = "Phase601"
SCHEMA_VERSION = "phase601_foodon_public_ontology.v1"
FROZEN_AT = "2026-07-22T11:05:00+00:00"
MODELS = ("qwen3", "glm4", "deepseek7b")
FAMILIES = (
    "fruit",
    "nut",
    "root_vegetable",
    "seed_vegetable",
    "animal_food",
)
ROOT_IDS = {
    "fruit": "FOODON_00001057",
    "nut": "FOODON_00001172",
    "root_vegetable": "FOODON_00002150",
    "seed_vegetable": "FOODON_00002153",
    "animal_food": "FOODON_00004242",
}
SPLITS = ("discovery", "independent_confirmation", "heldout")
CONCEPTS_PER_FAMILY = 96
SPLIT_CONCEPT_QUOTAS = {
    "discovery": 48,
    "independent_confirmation": 24,
    "heldout": 24,
}
DEPTH_BUCKET_QUOTAS = {"direct": 6, "near": 78, "deep": 12}
SURFACES = (
    "In the FoodOn hierarchy, {child} belongs under which broader class?\nA. {a}\nB. {b}\nAnswer with A or B only.",
    "Choose the FoodOn ancestor of {child} from the two displayed classes.\nA. {a}\nB. {b}\nReturn only A or B.",
    "FoodOn classifies {child} as a descendant of one option. Which option is it?\nA. {a}\nB. {b}\nGive just the letter A or B.",
    "Which of these two FoodOn classes contains {child} in its subclass tree?\nA. {a}\nB. {b}\nReply using A or B only.",
)
ANSWER_CONTINUATIONS = {"A": "A", "B": "B"}
FIXED_BATCH_SIZE = 32

# Frozen before any model behavior is read.
GATES = {
    "split_forced_choice_accuracy_min": 0.80,
    "split_family_forced_choice_accuracy_min": 0.65,
    "family_forced_choice_accuracy_min": 0.70,
    "surface_forced_choice_accuracy_min": 0.70,
    "nonlexical_forced_choice_accuracy_min": 0.70,
    "answer_order_accuracy_gap_max": 0.08,
    "concept_unanimous_rate_min": 0.50,
    "direct_candidate_output_rate_min": 0.80,
    "direct_exact_accuracy_min": 0.70,
}

OUT_DIR = ROOT / "tests/gpt5/result/phase601_foodon_public_ontology"
CASES_PATH = OUT_DIR / "phase601_registered_cases.jsonl.gz"
PROTOCOL_PATH = OUT_DIR / "phase601_frozen_protocol.json"
AUDIT_PATH = OUT_DIR / "phase601_static_audit.json"
SOURCE_MANIFEST_PATH = OUT_DIR / "phase601_source_manifest.json"

RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
RDFS = "http://www.w3.org/2000/01/rdf-schema#"
OWL = "http://www.w3.org/2002/07/owl#"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_hash(*parts: object) -> str:
    return hashlib.sha256("\x1f".join(str(part) for part in parts).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8") as handle:
                for row in rows:
                    handle.write(
                        json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
                    )


def parse_foodon(path: Path) -> tuple[dict[str, str], dict[str, set[str]], int]:
    labels: dict[str, str] = {}
    parents: dict[str, set[str]] = defaultdict(set)
    class_count = 0
    for _event, element in ET.iterparse(path, events=("end",)):
        if element.tag != f"{{{OWL}}}Class":
            continue
        class_count += 1
        uri = element.get(f"{{{RDF}}}about")
        if uri:
            class_id = uri.rsplit("/", 1)[-1]
            label = element.find(f"{{{RDFS}}}label")
            if label is not None and label.text:
                labels[class_id] = " ".join(label.text.split())
            for relation in element.findall(f"{{{RDFS}}}subClassOf"):
                parent_uri = relation.get(f"{{{RDF}}}resource")
                if parent_uri:
                    parents[class_id].add(parent_uri.rsplit("/", 1)[-1])
        element.clear()
    return labels, parents, class_count


def descendant_distances(parents: dict[str, set[str]], root: str) -> dict[str, int]:
    children: dict[str, set[str]] = defaultdict(set)
    for child, parent_ids in parents.items():
        for parent in parent_ids:
            children[parent].add(child)
    distances: dict[str, int] = {}
    queue = deque((child, 1) for child in children[root])
    while queue:
        child, distance = queue.popleft()
        if child in distances and distances[child] <= distance:
            continue
        distances[child] = distance
        queue.extend((descendant, distance + 1) for descendant in children[child])
    return distances


def normalized_words(label: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", label.casefold())


def lexical_cluster(label: str) -> str:
    value = label.casefold()
    value = re.sub(r"\([^)]*\)", " ", value)
    value = re.sub(r"\b(?:food|plant|animal) product\b", " ", value)
    value = re.sub(
        r"\b(?:raw|dried|frozen|canned|cooked|roasted|peeled|whole|pieces|ground|"
        r"fresh|mature|immature|processed|dehydrated|ripe|unripe)\b",
        " ",
        value,
    )
    return " ".join(re.findall(r"[a-z0-9]+", value))


def valid_label(label: str) -> bool:
    folded = label.casefold()
    return bool(
        label
        and len(label) <= 80
        and "\n" not in label
        and not folded.startswith("obsolete")
        and "http://" not in folded
        and "https://" not in folded
        and re.search(r"[a-z]", folded)
    )


def depth_bucket(distance: int) -> str:
    if distance == 1:
        return "direct"
    if distance <= 3:
        return "near"
    return "deep"


def representative_candidates(
    labels: dict[str, str],
    distances_by_family: dict[str, dict[str, int]],
) -> dict[str, list[dict[str, Any]]]:
    descendant_sets = {family: set(values) for family, values in distances_by_family.items()}
    output: dict[str, list[dict[str, Any]]] = {}
    for family in FAMILIES:
        other_descendants = set().union(
            *(descendant_sets[other] for other in FAMILIES if other != family)
        )
        exclusive = descendant_sets[family] - other_descendants
        clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
        root_words = set(normalized_words(labels[ROOT_IDS[family]])) - {"food", "product", "plant"}
        for class_id in exclusive:
            label = labels.get(class_id, "")
            cluster = lexical_cluster(label)
            if not valid_label(label) or not cluster:
                continue
            words = set(normalized_words(label))
            clusters[cluster].append(
                {
                    "concept_id": class_id,
                    "concept_label": label,
                    "cluster_key": cluster,
                    "family": family,
                    "distance_to_family_root": distances_by_family[family][class_id],
                    "depth_bucket": depth_bucket(distances_by_family[family][class_id]),
                    "root_word_overlap": sorted(words & root_words),
                    "lexical_cue": bool(words & root_words),
                }
            )
        representatives = [
            min(values, key=lambda row: (len(row["concept_label"]), row["concept_label"], row["concept_id"]))
            for values in clusters.values()
        ]
        output[family] = representatives
    return output


def select_concepts(candidates: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for family in FAMILIES:
        family_selected: list[dict[str, Any]] = []
        for bucket, quota in DEPTH_BUCKET_QUOTAS.items():
            eligible = [row for row in candidates[family] if row["depth_bucket"] == bucket]
            eligible.sort(key=lambda row: stable_hash("select", family, bucket, row["cluster_key"]))
            if len(eligible) < quota:
                raise RuntimeError(f"Not enough {family}/{bucket} concepts: {len(eligible)} < {quota}")
            family_selected.extend(eligible[:quota])
        if len(family_selected) != CONCEPTS_PER_FAMILY:
            raise RuntimeError(f"Unexpected concept count for {family}")
        family_selected.sort(key=lambda row: stable_hash("split", family, row["cluster_key"]))
        cursor = 0
        for split in SPLITS:
            quota = SPLIT_CONCEPT_QUOTAS[split]
            for row in family_selected[cursor : cursor + quota]:
                row["split"] = split
                selected.append(row)
            cursor += quota
    return selected


def build_cases(selected: list[dict[str, Any]], labels: dict[str, str]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    created_at = FROZEN_AT
    for concept in selected:
        family = concept["family"]
        alternatives = [other for other in FAMILIES if other != family]
        false_family = alternatives[
            int(stable_hash("false-family", concept["cluster_key"]), 16) % len(alternatives)
        ]
        true_label = labels[ROOT_IDS[family]]
        false_label = labels[ROOT_IDS[false_family]]
        for surface_index, template in enumerate(SURFACES):
            target = "A" if surface_index % 2 == 0 else "B"
            a = true_label if target == "A" else false_label
            b = false_label if target == "A" else true_label
            raw_prompt = template.format(child=concept["concept_label"], a=a, b=b)
            cases.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": PHASE,
                    "created_at": created_at,
                    "case_id": f"phase601_{family}_{concept['concept_id']}_s{surface_index}",
                    "concept_id": concept["concept_id"],
                    "concept_label": concept["concept_label"],
                    "cluster_key": concept["cluster_key"],
                    "split": concept["split"],
                    "family": family,
                    "family_root_id": ROOT_IDS[family],
                    "family_root_label": true_label,
                    "false_family": false_family,
                    "false_root_id": ROOT_IDS[false_family],
                    "false_root_label": false_label,
                    "distance_to_family_root": concept["distance_to_family_root"],
                    "depth_bucket": concept["depth_bucket"],
                    "lexical_cue": concept["lexical_cue"],
                    "root_word_overlap": concept["root_word_overlap"],
                    "surface_id": f"surface_{surface_index}",
                    "surface_index": surface_index,
                    "option_a": a,
                    "option_b": b,
                    "target_letter": target,
                    "foil_letter": "B" if target == "A" else "A",
                    "raw_prompt": raw_prompt,
                    "public_source": "FoodOn v2025-02-01",
                    "public_source_sha256": EXPECTED_SHA256,
                    "generated_from_named_subclass_closure": True,
                    "exclusive_family_membership": True,
                    "causal_evidence": False,
                }
            )
    cases.sort(key=lambda row: (SPLITS.index(row["split"]), row["case_id"]))
    return cases


def freeze() -> dict[str, Any]:
    source = validate_source(SOURCE_PATH)
    labels, parents, owl_class_count = parse_foodon(SOURCE_PATH)
    distances_by_family = {
        family: descendant_distances(parents, root_id) for family, root_id in ROOT_IDS.items()
    }
    candidates = representative_candidates(labels, distances_by_family)
    selected = select_concepts(candidates)
    cases = build_cases(selected, labels)
    token_ledger: dict[str, dict[str, list[int]]] = {}
    for model in MODELS:
        tokenizer = tokenizer_for(model)
        token_ledger[model] = {
            letter: [int(value) for value in tokenizer.encode(text, add_special_tokens=False)]
            for letter, text in ANSWER_CONTINUATIONS.items()
        }
        if any(len(ids) != 1 for ids in token_ledger[model].values()):
            raise RuntimeError(f"Phase601 requires one-token A/B continuations for {model}")
        if token_ledger[model]["A"] == token_ledger[model]["B"]:
            raise RuntimeError(f"Phase601 A/B token collision for {model}")

    write_jsonl(CASES_PATH, cases)
    source_manifest = {
        "schema_version": "phase601_source_manifest.v1",
        "phase_id": PHASE,
        "created_at": FROZEN_AT,
        **source,
        "official_project_url": "https://foodon.org/",
        "repository_url": "https://github.com/FoodOntology/foodon",
        "license": "CC-BY-4.0",
        "source_role": "external_public_taxonomy_truth",
        "not_a_human_label_substitute": True,
    }
    write_json(SOURCE_MANIFEST_PATH, source_manifest)
    concept_counts = Counter((row["split"], row["family"]) for row in selected)
    case_counts = Counter((row["split"], row["family"]) for row in cases)
    audit = {
        "schema_version": "phase601_static_audit.v1",
        "phase_id": PHASE,
        "created_at": FROZEN_AT,
        "source_valid": True,
        "source_size_bytes": SOURCE_PATH.stat().st_size,
        "source_sha256": sha256_file(SOURCE_PATH),
        "owl_class_element_count": owl_class_count,
        "labeled_named_class_count": len(labels),
        "simple_named_subclass_edge_count": sum(len(values) for values in parents.values()),
        "descendant_count_by_family": {
            family: len(distances_by_family[family]) for family in FAMILIES
        },
        "representative_candidate_count_by_family": {
            family: len(candidates[family]) for family in FAMILIES
        },
        "selected_concept_count": len(selected),
        "case_count": len(cases),
        "concept_count_by_split_family": {
            f"{split}/{family}": concept_counts[(split, family)]
            for split in SPLITS for family in FAMILIES
        },
        "case_count_by_split_family": {
            f"{split}/{family}": case_counts[(split, family)]
            for split in SPLITS for family in FAMILIES
        },
        "depth_bucket_count": dict(Counter(row["depth_bucket"] for row in selected)),
        "lexical_cue_concept_count": sum(row["lexical_cue"] for row in selected),
        "nonlexical_concept_count": sum(not row["lexical_cue"] for row in selected),
        "target_letter_count": dict(Counter(row["target_letter"] for row in cases)),
        "surface_count": dict(Counter(row["surface_id"] for row in cases)),
        "cluster_key_unique": len({row["cluster_key"] for row in selected}) == len(selected),
        "concept_split_disjoint": all(
            len({row["split"] for row in selected if row["concept_id"] == concept_id}) == 1
            for concept_id in {row["concept_id"] for row in selected}
        ),
        "false_root_is_not_true_family_root": all(
            row["family_root_id"] != row["false_root_id"] for row in cases
        ),
        "false_root_is_not_an_ancestor": all(
            row["concept_id"] not in distances_by_family[row["false_family"]]
            for row in cases
        ),
        "true_false_family_pair_count": dict(Counter(
            f"{row['family']}->{row['false_family']}" for row in cases
        )),
        "answer_token_ledger_by_model": token_ledger,
    }
    write_json(AUDIT_PATH, audit)
    protocol = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE,
        "created_at": FROZEN_AT,
        "models": list(MODELS),
        "families": list(FAMILIES),
        "splits": list(SPLITS),
        "concepts_per_family": CONCEPTS_PER_FAMILY,
        "surface_count_per_concept": len(SURFACES),
        "registered_concept_count": len(selected),
        "registered_case_count": len(cases),
        "source_url": SOURCE_URL,
        "source_size_bytes": EXPECTED_SIZE,
        "source_sha256": EXPECTED_SHA256,
        "cases_sha256": sha256_file(CASES_PATH),
        "source_manifest_sha256": sha256_file(SOURCE_MANIFEST_PATH),
        "static_audit_sha256": sha256_file(AUDIT_PATH),
        "answer_continuations": ANSWER_CONTINUATIONS,
        "answer_token_ledger_by_model": token_ledger,
        "fixed_batch_size": FIXED_BATCH_SIZE,
        "gates": GATES,
        "protocol_constraints": {
            "concept_clusters_do_not_cross_splits": True,
            "true_family_membership_is_exclusive_across_registered_roots": True,
            "candidate_order_is_exactly_counterbalanced": True,
            "model_behavior_unread_before_freeze": True,
            "teacher_forced_internal_states_collected": False,
            "causal_intervention_authorized": False,
        },
    }
    write_json(PROTOCOL_PATH, protocol)
    return {"protocol": protocol, "audit": audit}


def main() -> None:
    print(json.dumps(freeze(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
