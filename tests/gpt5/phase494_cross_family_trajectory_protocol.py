#!/usr/bin/env python3
"""Freeze the Phase494 cross-family relation trajectory protocol.

The protocol removes the Phase490 lexical counterfactual artifact by keeping
the claim, vocabulary, rule count, and fact count fixed inside each paired
world. Only the evidence connectivity is exchanged. No model is loaded.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).resolve()
OUT_DIR = ROOT / "tests" / "gpt5" / "result" / "phase494_cross_family_trajectory_protocol"

TRAIN_FAMILIES = ("marker_inheritance", "signal_assignment")
UNSEEN_FAMILIES = (
    "symmetric_pair",
    "directed_mentor",
    "transitive_precedence",
    "direct_nontransitive",
)
ALL_FAMILIES = TRAIN_FAMILIES + UNSEEN_FAMILIES
TRACKS = ("identity", "native_plain_candidate")
POSITION_ROLES = (
    "rules_end",
    "target_evidence_end",
    "distractor_evidence_end",
    "claim_entity_end",
    "claim_relation_end",
    "claim_end",
    "final_instruction_end",
    "prompt_end",
)
LENGTHS = ("short", "medium", "long")
FACT_ORDERS = ("target_first", "distractor_first", "interleaved")
POLARITIES = ("positive", "negative")
PAIRS_PER_FAMILY = 72

SPLITS = {
    "behavior_qualification": {"index": 0, "prefix": "bq", "families": ALL_FAMILIES, "sealed": False},
    "formation_fit": {"index": 1, "prefix": "ff", "families": TRAIN_FAMILIES, "sealed": False},
    "family_prediction": {"index": 2, "prefix": "up", "families": UNSEEN_FAMILIES, "sealed": False},
    "sealed_cross_family": {"index": 3, "prefix": "sc", "families": ALL_FAMILIES, "sealed": True},
}

FAMILY_LABELS = {
    "marker_inheritance": "single-valued marker inheritance",
    "signal_assignment": "single-valued signal assignment",
    "symmetric_pair": "symmetric binary relation",
    "directed_mentor": "directed binary role relation",
    "transitive_precedence": "transitive relation",
    "direct_nontransitive": "direct non-transitive relation",
}


def stable_hash(*parts: object, n: int = 20) -> str:
    payload = "::".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:n]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def token_counter(texts: list[str]) -> Counter[str]:
    return Counter(re.findall(r"[A-Za-z0-9_]+", " ".join(texts).lower()))


def ident(prefix: str, family_index: int, pair_index: int, slot: str) -> str:
    return f"{prefix}{family_index}{pair_index:03d}{slot}"


def family_worlds(prefix: str, family: str, pair_index: int) -> dict[str, Any]:
    fi = ALL_FAMILIES.index(family)
    e = [ident(prefix, fi, pair_index, f"e{i}") for i in range(8)]
    a = [ident(prefix, fi, pair_index, f"a{i}") for i in range(6)]

    if family == "marker_inheritance":
        rules = [
            f"Every member of group {a[0]} carries marker {a[2]}.",
            f"Every member of group {a[1]} carries marker {a[3]}.",
            "Each listed entity belongs to exactly one listed group.",
        ]
        supporting = [f"{e[0]} belongs to group {a[0]}.", f"{e[1]} belongs to group {a[1]}."]
        opposing = [f"{e[0]} belongs to group {a[1]}.", f"{e[1]} belongs to group {a[0]}."]
        claim = f"{e[0]} carries marker {a[2]}."
        relation_anchor = "carries marker"
    elif family == "signal_assignment":
        rules = [
            f"Every station assigned channel {a[0]} emits signal {a[2]}.",
            f"Every station assigned channel {a[1]} emits signal {a[3]}.",
            "Each listed station is assigned exactly one listed channel.",
        ]
        supporting = [f"{e[0]} is assigned channel {a[0]}.", f"{e[1]} is assigned channel {a[1]}."]
        opposing = [f"{e[0]} is assigned channel {a[1]}.", f"{e[1]} is assigned channel {a[0]}."]
        claim = f"{e[0]} emits signal {a[2]}."
        relation_anchor = "emits signal"
    elif family == "symmetric_pair":
        rules = [
            "Pairing is mutual.",
            "Only explicitly listed pairs are paired.",
        ]
        supporting = [f"{e[0]} is paired with {e[1]}.", f"{e[2]} is paired with {e[3]}."]
        opposing = [f"{e[0]} is paired with {e[2]}.", f"{e[1]} is paired with {e[3]}."]
        claim = f"{e[0]} is paired with {e[1]}."
        relation_anchor = "paired with"
    elif family == "directed_mentor":
        rules = [
            "Mentoring is directed and is not automatically reciprocal.",
            "Only explicitly listed mentoring relationships hold.",
        ]
        supporting = [f"{e[0]} mentors {e[1]}.", f"{e[2]} mentors {e[3]}."]
        opposing = [f"{e[0]} mentors {e[2]}.", f"{e[1]} mentors {e[3]}."]
        claim = f"{e[1]} is mentored by {e[0]}."
        relation_anchor = "mentored by"
    elif family == "transitive_precedence":
        rules = [
            "Precedence is transitive: if one item precedes a second and the second precedes a third, the first precedes the third.",
            "Precedence is one-way and irreflexive.",
        ]
        supporting = [
            f"{e[0]} precedes {e[1]}.",
            f"{e[1]} precedes {e[2]}.",
            f"{e[3]} precedes {e[4]}.",
        ]
        opposing = [
            f"{e[2]} precedes {e[1]}.",
            f"{e[1]} precedes {e[0]}.",
            f"{e[4]} precedes {e[3]}.",
        ]
        claim = f"{e[0]} precedes {e[2]}."
        relation_anchor = "precedes"
    elif family == "direct_nontransitive":
        rules = [
            "Points-directly-to is directed and non-transitive.",
            "Only explicitly listed direct links hold.",
        ]
        supporting = [
            f"{e[0]} points directly to {e[1]}.",
            f"{e[1]} points directly to {e[2]}.",
            f"{e[0]} points directly to {e[2]}.",
            f"{e[3]} points directly to {e[4]}.",
        ]
        opposing = [
            f"{e[0]} points directly to {e[1]}.",
            f"{e[1]} points directly to {e[2]}.",
            f"{e[2]} points directly to {e[0]}.",
            f"{e[4]} points directly to {e[3]}.",
        ]
        claim = f"{e[0]} points directly to {e[2]}."
        relation_anchor = "points directly to"
    else:
        raise ValueError(family)

    return {
        "rules": rules,
        "supporting": supporting,
        "opposing": opposing,
        "positive_claim": claim,
        "claim_entity": e[0] if family != "directed_mentor" else e[1],
        "claim_relation": relation_anchor,
    }


def distractor_facts(prefix: str, family: str, pair_index: int, length: str) -> list[str]:
    count = {"short": 2, "medium": 4, "long": 8}[length]
    fi = ALL_FAMILIES.index(family)
    rows = []
    for index in range(count):
        item = ident(prefix, fi, pair_index, f"d{index}")
        box = ident(prefix, fi, pair_index, f"b{index}")
        rows.append(f"{item} is stored in container {box}.")
    return rows


def ordered_facts(target: list[str], distractors: list[str], mode: str) -> list[str]:
    if mode == "target_first":
        return target + distractors
    if mode == "distractor_first":
        return distractors + target
    if mode != "interleaved":
        raise ValueError(mode)
    out = []
    for target_fact, distractor_fact in itertools.zip_longest(target, distractors):
        if distractor_fact is not None:
            out.append(distractor_fact)
        if target_fact is not None:
            out.append(target_fact)
    return out


def render_surface(
    track: str,
    rules: list[str],
    facts: list[str],
    target_facts: list[str],
    distractors: list[str],
    claim: str,
    claim_entity: str,
    claim_relation: str,
) -> dict[str, Any]:
    rules_block = "\n".join(f"- {rule}" for rule in rules)
    facts_block = "\n".join(f"{index + 1}. {fact}" for index, fact in enumerate(facts))
    if track == "identity":
        instruction = "Based only on this closed world, is the claim true or false?"
        prompt = (
            "You are evaluating a closed synthetic world. Use only its rules and facts.\n"
            f"Rules:\n{rules_block}\n"
            f"Facts:\n{facts_block}\n"
            f"Claim:\n{claim}\n"
            f"Question: {instruction}\nAnswer:"
        )
    elif track == "native_plain_candidate":
        instruction = "Decide whether the proposition follows in this miniature world. Reply true or false."
        prompt = (
            "Consider only the miniature world described here.\n"
            f"World rules:\n{rules_block}\n"
            f"World evidence:\n{facts_block}\n"
            f"Proposition:\n{claim}\n"
            f"Task: {instruction}\nResponse:"
        )
    else:
        raise ValueError(track)

    claim_start = prompt.rfind(claim)
    if claim_start < 0:
        raise RuntimeError("Claim not found in rendered surface")
    entity_offset = claim.find(claim_entity)
    relation_offset = claim.find(claim_relation)
    if entity_offset < 0 or relation_offset < 0:
        raise RuntimeError("Claim role anchor missing")
    target_last = max(target_facts, key=lambda fact: prompt.rfind(fact))
    distractor_last = max(distractors, key=lambda fact: prompt.rfind(fact))
    instruction_start = prompt.rfind(instruction)
    role_char_ends = {
        "rules_end": prompt.find(rules_block) + len(rules_block),
        "target_evidence_end": prompt.rfind(target_last) + len(target_last),
        "distractor_evidence_end": prompt.rfind(distractor_last) + len(distractor_last),
        "claim_entity_end": claim_start + entity_offset + len(claim_entity),
        "claim_relation_end": claim_start + relation_offset + len(claim_relation),
        "claim_end": claim_start + len(claim),
        "final_instruction_end": instruction_start + len(instruction),
        "prompt_end": len(prompt),
    }
    if tuple(role_char_ends) != POSITION_ROLES or not all(0 < value <= len(prompt) for value in role_char_ends.values()):
        raise RuntimeError("Invalid role anchor ledger")
    return {
        "track": track,
        "track_class": "native_core",
        "semantic_prompt": prompt,
        "role_char_ends": role_char_ends,
    }


def control_grid() -> list[tuple[str, str, str, int]]:
    return list(itertools.product(LENGTHS, FACT_ORDERS, POLARITIES, range(4)))


def build_split(split: str) -> list[dict[str, Any]]:
    spec = SPLITS[split]
    rows = []
    controls = control_grid()
    if len(controls) != PAIRS_PER_FAMILY:
        raise RuntimeError("Control grid does not match the frozen pair count")
    for family in spec["families"]:
        for pair_index, (length, fact_order, polarity, replicate) in enumerate(controls):
            world = family_worlds(spec["prefix"], family, pair_index)
            distractors = distractor_facts(spec["prefix"], family, pair_index, length)
            source_pair_id = f"p494-{split}-{family}-{pair_index:03d}-{stable_hash(split, family, pair_index)}"
            positive_claim = world["positive_claim"]
            claim = (
                positive_claim
                if polarity == "positive"
                else f"It is not true that {positive_claim[:-1]}."
            )
            for world_role, positive_truth in (("supporting_world", True), ("opposing_world", False)):
                target = world["supporting"] if positive_truth else world["opposing"]
                facts = ordered_facts(target, distractors, fact_order)
                truth_value = positive_truth if polarity == "positive" else not positive_truth
                world_case_id = f"{source_pair_id}::{world_role}"
                variants = [
                    render_surface(
                        track,
                        world["rules"],
                        facts,
                        target,
                        distractors,
                        claim,
                        world["claim_entity"],
                        world["claim_relation"],
                    )
                    for track in TRACKS
                ]
                rows.append({
                    "schema_version": "phase494_cross_family_sample.v1",
                    "sample_id": f"p494-{stable_hash(world_case_id)}",
                    "world_case_id": world_case_id,
                    "source_pair_id": source_pair_id,
                    "split": split,
                    "sealed": bool(spec["sealed"]),
                    "family": family,
                    "family_class": FAMILY_LABELS[family],
                    "family_role": "fit" if family in TRAIN_FAMILIES else "unseen_prediction",
                    "pair_index": pair_index,
                    "replicate": replicate,
                    "world_role": world_role,
                    "positive_world_truth": positive_truth,
                    "truth_value": truth_value,
                    "claim_polarity": polarity,
                    "length_control": length,
                    "fact_order": fact_order,
                    "rules": world["rules"],
                    "target_facts": target,
                    "distractor_facts": distractors,
                    "facts": facts,
                    "positive_claim": positive_claim,
                    "claim": claim,
                    "claim_entity": world["claim_entity"],
                    "claim_relation": world["claim_relation"],
                    "surface_variants": variants,
                })
    return rows


def audit_split(split: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    spec = SPLITS[split]
    expected = len(spec["families"]) * PAIRS_PER_FAMILY * 2
    if len(rows) != expected:
        raise RuntimeError(f"{split}: expected {expected} rows, got {len(rows)}")
    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_pair[row["source_pair_id"]].append(row)
        if set(variant["track"] for variant in row["surface_variants"]) != set(TRACKS):
            raise RuntimeError("Track ledger mismatch")
        prompts = [variant["semantic_prompt"] for variant in row["surface_variants"]]
        if len(set(prompts)) != len(TRACKS):
            raise RuntimeError("Native surfaces are not genuinely distinct")
    for pair_id, pair in by_pair.items():
        if len(pair) != 2:
            raise RuntimeError(f"Incomplete paired world {pair_id}")
        left, right = pair
        if left["claim"] != right["claim"] or left["rules"] != right["rules"]:
            raise RuntimeError(f"Claim/rule drift inside pair {pair_id}")
        if len(left["facts"]) != len(right["facts"]):
            raise RuntimeError(f"Fact-count drift inside pair {pair_id}")
        if token_counter(left["facts"]) != token_counter(right["facts"]):
            raise RuntimeError(f"Evidence vocabulary drift inside pair {pair_id}")
        if {left["truth_value"], right["truth_value"]} != {False, True}:
            raise RuntimeError(f"Truth imbalance inside pair {pair_id}")

    family_counts = Counter(row["family"] for row in rows)
    truth_counts = Counter(str(row["truth_value"]).lower() for row in rows)
    length_counts = Counter(row["length_control"] for row in rows)
    order_counts = Counter(row["fact_order"] for row in rows)
    polarity_counts = Counter(row["claim_polarity"] for row in rows)
    return {
        "row_count": len(rows),
        "surface_variant_count": len(rows) * len(TRACKS),
        "paired_world_count": len(by_pair),
        "family_counts": dict(sorted(family_counts.items())),
        "truth_counts": dict(sorted(truth_counts.items())),
        "length_counts": dict(sorted(length_counts.items())),
        "fact_order_counts": dict(sorted(order_counts.items())),
        "polarity_counts": dict(sorted(polarity_counts.items())),
        "claim_fixed_within_pair": True,
        "fact_count_fixed_within_pair": True,
        "evidence_token_multiset_fixed_within_pair": True,
        "native_surfaces_distinct": True,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    split_rows = {split: build_split(split) for split in SPLITS}
    audits = {split: audit_split(split, rows) for split, rows in split_rows.items()}
    files = {}
    for split, rows in split_rows.items():
        path = OUT_DIR / f"phase494_{split}_samples.jsonl"
        write_jsonl(path, rows)
        files[split] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256_file(path),
            "rows": len(rows),
            "surface_variants": len(rows) * len(TRACKS),
            "sealed": SPLITS[split]["sealed"],
        }

    contract = {
        "schema_version": "phase494_cross_family_contract.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_before_model_run",
        "scientific_question": "Does a train-family relation observer predict four unseen relation structures, and where does that readability first appear?",
        "fit_families": list(TRAIN_FAMILIES),
        "unseen_families": list(UNSEEN_FAMILIES),
        "tracks": list(TRACKS),
        "position_roles": list(POSITION_ROLES),
        "projection": {
            "dimension": 64,
            "seeds": {"qwen3": 490031, "glm4": 490037, "deepseek7b": 494041},
            "changed_after_prediction_forbidden": True,
        },
        "primary_windows": {
            "qwen3": {"position_role": "prompt_end", "layer_with_embedding": 31, "source": "Phase491-492 frozen window"},
            "glm4": {"position_role": "prompt_end", "layer_with_embedding": 33, "source": "Phase491-492 frozen window"},
            "deepseek7b": {"position_role": "prompt_end", "normalized_depth": 0.85, "source": "predeclared architecture-normalized fallback"},
        },
        "behavior_gate": {
            "per_family_identity_lcb95_min": 0.80,
            "per_family_native_plain_lcb95_min": 0.80,
            "per_family_native_intersection_lcb95_min": 0.80,
            "per_family_paired_world_lcb95_min": 0.75,
            "overall_unseen_intersection_lcb95_min": 0.85,
        },
        "physical_gate": {
            "per_unseen_family_q_strictly_positive": True,
            "per_unseen_family_prediction_lcb95_min": 0.80,
            "overall_unseen_prediction_lcb95_min": 0.85,
            "stable_layer_count": 2,
        },
        "nonlinear_controls": ["fixed_tanh_random_features", "local_rbf_class_kernel"],
        "evidence_boundaries": {
            "sealed_split_read": False,
            "causal_intervention": False,
            "attention_head_channel_neuron_scan": False,
            "role_state_sequence_is_compute_transport": False,
        },
        "split_files": files,
        "source_path": str(SOURCE.relative_to(ROOT)),
        "source_sha256": sha256_file(SOURCE),
    }
    contract_path = OUT_DIR / "phase494_frozen_contract.json"
    contract_path.write_text(json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    audit = {
        "schema_version": "phase494_static_audit.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "static_pass_no_model_run",
        "model_loaded": False,
        "cuda_used": False,
        "splits": audits,
        "contract_sha256": sha256_file(contract_path),
        "sealed_split_created_but_not_read_by_downstream_protocol": True,
        "authorization": {"three_model_behavior_qualification": True},
    }
    audit_path = OUT_DIR / "phase494_static_audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(contract_path)
    print(audit_path)


if __name__ == "__main__":
    main()
