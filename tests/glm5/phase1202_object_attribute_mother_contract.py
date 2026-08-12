#!/usr/bin/env python3
"""Freeze and materialize the Phase1202 object-attribute mother-family contract.

This phase is deliberately zero-model.  It generates a controlled natural-
language package, audits local tokenizers without loading model weights, and
freezes the future observation/intervention registry.  It does not inspect
hidden states, run behavior scoring, or add an empirical K item.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests/glm5"
sys.path.insert(0, str(TEST_ROOT))

PHASE = 1202
SCRIPT = Path(__file__).resolve()
AUDIT_SCRIPT = TEST_ROOT / "phase1202_object_attribute_mother_contract_audit.py"
UPSTREAM_FINAL = (
    TEST_ROOT
    / "result/phase1201_registry_identifiability_abstention/analysis/final.json"
)
OUT_ROOT = TEST_ROOT / "result/phase1202_object_attribute_mother_contract"
CONTRACT_PATH = OUT_ROOT / "protocol/mother_family_contract.json"
PACKAGE_PATH = OUT_ROOT / "material/object_attribute_binding.jsonl"
TOKEN_AUDIT_PATH = OUT_ROOT / "audit/tokenizer_audit.json"
SUMMARY_PATH = OUT_ROOT / "analysis/readiness_summary.json"
AUDIT_PATH = OUT_ROOT / "audit/independent_audit.json"
FINAL_PATH = OUT_ROOT / "analysis/final.json"

PHASE1201_EXPECTED_DIGEST = (
    "0a0c5cee0f0ed305b35d959d5921f66c31474005b9fe04eee99ebcdbfb042b91"
)

FAMILY = "object_attribute_binding"
ATTRIBUTES = ("color", "material", "location", "size", "shape", "status")
PANELS = ("active", "matched_null", "surface_only", "semantic_neighbor")
TEMPLATES = ("profile_prose", "compact_ledger")
CANDIDATE_ORDERS = (0, 1, 2)
BINDING_STATES = (0, 1)
SPLITS = ("discovery", "confirmation", "unseen_composition")
EXPECTED_ROW_COUNT = 4 * 4 * 6 * 4 * 2 * 3 * 2
EXPECTED_SPLIT_COUNTS = {
    "discovery": 2304,
    "confirmation": 1152,
    "unseen_composition": 1152,
}

WORLD_SPECS: tuple[dict[str, Any], ...] = (
    {
        "world": "lexical_world_0",
        "entities": (
            "Aaron", "Adam", "Alex", "Allen", "Amy", "Anna",
            "Arthur", "Ben", "Bill", "Bob", "Brian", "Bruce",
        ),
        "values": {
            "color": ("red", "blue", "green"),
            "material": ("wood", "metal", "glass"),
            "location": ("north", "east", "west"),
            "size": ("small", "medium", "large"),
            "shape": ("round", "square", "flat"),
            "status": ("open", "closed", "ready"),
        },
    },
    {
        "world": "lexical_world_1",
        "entities": (
            "Carol", "Chris", "Cindy", "Daniel", "Dennis", "Edward",
            "Emma", "Eric", "Frank", "Fred", "George", "Henry",
        ),
        "values": {
            "color": ("black", "white", "yellow"),
            "material": ("steel", "stone", "paper"),
            "location": ("center", "left", "right"),
            "size": ("short", "normal", "tall"),
            "shape": ("oval", "curved", "straight"),
            "status": ("active", "idle", "locked"),
        },
    },
    {
        "world": "lexical_world_2",
        "entities": (
            "Jack", "James", "Jane", "Jason", "John", "Joseph",
            "Karen", "Kate", "Kelly", "Linda", "Lisa", "Mark",
        ),
        "values": {
            "color": ("orange", "purple", "gray"),
            "material": ("copper", "plastic", "cloth"),
            "location": ("upper", "lower", "middle"),
            "size": ("narrow", "regular", "wide"),
            "shape": ("sharp", "smooth", "solid"),
            "status": ("online", "offline", "paused"),
        },
    },
    {
        "world": "lexical_world_3",
        "entities": (
            "Martin", "Mary", "Michael", "Nancy", "Paul", "Rachel",
            "Rebecca", "Robert", "Roger", "Ryan", "Sarah", "Scott",
        ),
        "values": {
            "color": ("pink", "brown", "gold"),
            "material": ("iron", "rubber", "clay"),
            "location": ("front", "back", "side"),
            "size": ("thin", "average", "thick"),
            "shape": ("circular", "angular", "pointed"),
            "status": ("awake", "asleep", "waiting"),
        },
    },
)

PERMUTATIONS = tuple(itertools.permutations(range(3)))
TOKEN_PATTERN = re.compile(r"[A-Za-z]+|[0-9]+|[^\w\s]", re.UNICODE)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"line {line_number} is not an object")
            rows.append(value)
    return rows


def source_hashes() -> dict[str, str]:
    return {
        "generator": file_sha256(SCRIPT),
        "independent_audit": file_sha256(AUDIT_SCRIPT),
    }


def upstream_hashes() -> dict[str, str]:
    upstream = read_json(UPSTREAM_FINAL)
    if upstream.get("final_digest") != PHASE1201_EXPECTED_DIGEST:
        raise RuntimeError("Phase1201 final digest is not the frozen upstream")
    return {"phase1201_final_file": file_sha256(UPSTREAM_FINAL)}


def lexical_tokens(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


def token_multiset_digest(text: str) -> str:
    return digest(sorted(lexical_tokens(text)))


def split_for(profile_index: int, attribute_index: int) -> str:
    residue = (profile_index + attribute_index) % 4
    if residue in (0, 1):
        return "discovery"
    if residue == 2:
        return "confirmation"
    return "unseen_composition"


def base_assignments(world_index: int, profile_index: int) -> dict[str, dict[str, str]]:
    world = WORLD_SPECS[world_index]
    entities = world["entities"][profile_index * 3 : profile_index * 3 + 3]
    assignments = {entity: {} for entity in entities}
    for attribute_index, attribute in enumerate(ATTRIBUTES):
        values = world["values"][attribute]
        permutation = PERMUTATIONS[(world_index + 2 * profile_index + attribute_index) % 6]
        for entity_index, entity in enumerate(entities):
            assignments[entity][attribute] = values[permutation[entity_index]]
    return assignments


def deep_copy_assignments(assignments: dict[str, dict[str, str]]) -> dict[str, dict[str, str]]:
    return {entity: dict(values) for entity, values in assignments.items()}


def swap(assignments: dict[str, dict[str, str]], left: str, right: str, attribute: str) -> None:
    assignments[left][attribute], assignments[right][attribute] = (
        assignments[right][attribute],
        assignments[left][attribute],
    )


def rotate(values: tuple[str, str, str], offset: int) -> tuple[str, str, str]:
    return values[offset:] + values[:offset]


def render_record(entity: str, values: dict[str, str], template: str) -> str:
    if template == "profile_prose":
        return (
            f"The {entity} unit has color {values['color']}, material {values['material']}, "
            f"location {values['location']}, size {values['size']}, shape {values['shape']}, "
            f"and status {values['status']}."
        )
    fields = "; ".join(f"{attribute} {values[attribute]}" for attribute in ATTRIBUTES)
    return f"{entity} unit: {fields}."


def render_prompt(
    assignments: dict[str, dict[str, str]],
    record_order: tuple[str, str, str],
    attribute: str,
    target_value: str,
    candidates: tuple[str, str, str],
    template: str,
) -> tuple[str, list[str], str]:
    records = [render_record(entity, assignments[entity], template) for entity in record_order]
    if template == "profile_prose":
        query = (
            f"Based only on these profiles, which unit has {attribute} {target_value}? "
            f"Choose exactly one name from {', '.join(candidates)}."
        )
    else:
        query = (
            f"Select the unit whose {attribute} is {target_value}. "
            f"Allowed answers: {', '.join(candidates)}."
        )
    answer_prefix = "Answer:"
    return " ".join(records + [query, answer_prefix]), records, query


def build_row(
    world_index: int,
    profile_index: int,
    attribute_index: int,
    panel: str,
    template: str,
    candidate_order: int,
    binding_state: int,
) -> dict[str, Any]:
    world = WORLD_SPECS[world_index]
    attribute = ATTRIBUTES[attribute_index]
    neighbor_attribute = ATTRIBUTES[(attribute_index + 1) % len(ATTRIBUTES)]
    entities = tuple(world["entities"][profile_index * 3 : profile_index * 3 + 3])
    base = base_assignments(world_index, profile_index)
    assignments = deep_copy_assignments(base)
    focus_index = (profile_index + attribute_index) % 2
    target_value = base[entities[focus_index]][attribute]
    record_order = entities

    if panel in ("active", "matched_null"):
        if binding_state == 1:
            swap(assignments, entities[0], entities[1], attribute)
        if panel == "matched_null":
            target_value = base[entities[2]][attribute]
    elif panel == "surface_only":
        if binding_state == 1:
            record_order = (entities[1], entities[0], entities[2])
    elif panel == "semantic_neighbor":
        if binding_state == 1:
            swap(assignments, entities[0], entities[1], neighbor_attribute)
    else:
        raise ValueError(panel)

    matches = [entity for entity in entities if assignments[entity][attribute] == target_value]
    if len(matches) != 1:
        raise AssertionError("query must have exactly one correct entity")
    gold = matches[0]
    candidates = rotate(entities, candidate_order)
    prompt, rendered_records, query = render_prompt(
        assignments, record_order, attribute, target_value, candidates, template
    )
    split = split_for(profile_index, attribute_index)
    item_key = {
        "world": world["world"],
        "profile": profile_index,
        "attribute": attribute,
        "panel": panel,
        "template": template,
        "candidate_order": candidate_order,
        "binding_state": binding_state,
    }
    return {
        "schema_version": "phase1202.object_attribute.row.v1",
        "item_id": f"p1202-{digest(item_key)[:20]}",
        "family": FAMILY,
        "world": world["world"],
        "world_index": world_index,
        "profile_id": f"{world['world']}-profile-{profile_index}",
        "profile_index": profile_index,
        "combination_id": f"{world['world']}|{profile_index}|{attribute}",
        "split": split,
        "panel": panel,
        "template": template,
        "candidate_order": candidate_order,
        "binding_state": binding_state,
        "attribute": attribute,
        "neighbor_attribute": neighbor_attribute,
        "target_value": target_value,
        "entities": list(entities),
        "record_order": list(record_order),
        "candidates": list(candidates),
        "gold_candidate": gold,
        "gold_position": candidates.index(gold),
        "assignments": assignments,
        "prompt": prompt,
        "prompt_token_multiset_digest": token_multiset_digest(prompt),
        "rendered_records": rendered_records,
        "query": query,
        "answer_prefix": "Answer:",
        "probe_anchors": {
            "record_entity_names": list(record_order),
            "queried_attribute": attribute,
            "queried_value": target_value,
            "query_end": query[-32:],
            "answer_boundary": "Answer:",
        },
    }


def generate_rows() -> list[dict[str, Any]]:
    rows = [
        build_row(w, p, a, panel, template, order, state)
        for w in range(len(WORLD_SPECS))
        for p in range(4)
        for a in range(len(ATTRIBUTES))
        for panel in PANELS
        for template in TEMPLATES
        for order in CANDIDATE_ORDERS
        for state in BINDING_STATES
    ]
    if len(rows) != EXPECTED_ROW_COUNT:
        raise AssertionError((len(rows), EXPECTED_ROW_COUNT))
    return rows


def tokenizer_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    from model_utils import MODEL_CONFIGS
    from transformers import AutoTokenizer

    entity_atoms = sorted({entity for world in WORLD_SPECS for entity in world["entities"]})
    value_atoms = sorted(
        {
            value
            for world in WORLD_SPECS
            for values in world["values"].values()
            for value in values
        }
    )
    atoms = sorted(set(entity_atoms) | set(value_atoms) | set(ATTRIBUTES))
    models: dict[str, Any] = {}
    for model_name in ("qwen3", "glm4", "deepseek7b"):
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_CONFIGS[model_name]["path"],
            trust_remote_code=True,
            local_files_only=True,
            use_fast=False,
        )
        lengths = {
            atom: len(tokenizer.encode(" " + atom, add_special_tokens=False)) for atom in atoms
        }
        prompt_lengths = [
            len(tokenizer.encode(row["prompt"], add_special_tokens=False)) for row in rows
        ]
        candidate_prefix_free = True
        for row in rows:
            encoded = [
                tuple(tokenizer.encode(" " + candidate, add_special_tokens=False))
                for candidate in row["candidates"]
            ]
            if len(set(encoded)) != 3:
                candidate_prefix_free = False
                break
            for left, right in itertools.permutations(encoded, 2):
                if len(left) < len(right) and right[: len(left)] == left:
                    candidate_prefix_free = False
                    break
            if not candidate_prefix_free:
                break
        models[model_name] = {
            "tokenizer_path": str(MODEL_CONFIGS[model_name]["path"]),
            "atom_count": len(atoms),
            "single_token_atom_count": sum(length == 1 for length in lengths.values()),
            "all_atoms_single_token": all(length == 1 for length in lengths.values()),
            "non_single_token_atoms": {
                atom: length for atom, length in lengths.items() if length != 1
            },
            "candidate_sequences_unique_and_prefix_free": candidate_prefix_free,
            "minimum_prompt_tokens": min(prompt_lengths),
            "maximum_prompt_tokens": max(prompt_lengths),
            "mean_prompt_tokens": sum(prompt_lengths) / len(prompt_lengths),
            "prompt_within_512_tokens": max(prompt_lengths) <= 512,
        }
    overall = all(
        payload["all_atoms_single_token"]
        and payload["candidate_sequences_unique_and_prefix_free"]
        and payload["prompt_within_512_tokens"]
        for payload in models.values()
    )
    output = {
        "phase": PHASE,
        "kind": "local_tokenizer_only_audit",
        "model_weights_loaded": False,
        "models": models,
        "overall_pass": overall,
    }
    output["tokenizer_audit_digest"] = digest(output)
    return output


def factor_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "row_count": len(rows),
        "split_counts": dict(sorted(Counter(row["split"] for row in rows).items())),
        "panel_counts": dict(sorted(Counter(row["panel"] for row in rows).items())),
        "world_counts": dict(sorted(Counter(row["world"] for row in rows).items())),
        "attribute_counts": dict(sorted(Counter(row["attribute"] for row in rows).items())),
        "template_counts": dict(sorted(Counter(row["template"] for row in rows).items())),
        "gold_position_counts": dict(
            sorted(Counter(str(row["gold_position"]) for row in rows).items())
        ),
    }


def build_contract() -> dict[str, Any]:
    contract: dict[str, Any] = {
        "phase": PHASE,
        "schema_version": "phase1202.object_attribute.mother_contract.v1",
        "created_at": utc_now(),
        "purpose": (
            "freeze one controlled natural-language mother family before any model behavior, "
            "hidden-state, or causal scan"
        ),
        "upstream": {
            "phase1201_final_digest": PHASE1201_EXPECTED_DIGEST,
            "upstream_hashes": upstream_hashes(),
        },
        "source_hashes": source_hashes(),
        "family_definition": {
            "name": FAMILY,
            "operational_object": (
                "Given three directly named units with six declared attributes, select the unique "
                "unit bound to a queried attribute value."
            ),
            "not_claimed": [
                "a universal language primitive",
                "a fixed hidden direction",
                "a natural-corpus mechanism",
                "a model-independent neural coordinate",
            ],
        },
        "factor_design": {
            "worlds": [world["world"] for world in WORLD_SPECS],
            "profiles_per_world": 4,
            "entities_per_profile": 3,
            "attributes": list(ATTRIBUTES),
            "panels": list(PANELS),
            "templates": list(TEMPLATES),
            "candidate_orders": list(CANDIDATE_ORDERS),
            "binding_states": list(BINDING_STATES),
            "splits": list(SPLITS),
            "expected_rows": EXPECTED_ROW_COUNT,
            "expected_split_counts": EXPECTED_SPLIT_COUNTS,
            "split_rule": "(profile_index + attribute_index) mod 4; residues 0/1 discovery, 2 confirmation, 3 unseen_composition",
        },
        "panel_semantics": {
            "active": "The queried attribute swaps between entity0 and entity1; the correct entity must flip.",
            "matched_null": "The same queried-attribute swap occurs, but the target is the unchanged anchor entity; the answer must not flip.",
            "surface_only": "Only entity-record order changes; assignments and answer remain fixed.",
            "semantic_neighbor": "A nonqueried neighboring attribute swaps; queried binding and answer remain fixed.",
        },
        "probe_registry": {
            "P1_record_entity_entry": "direct entity-name spans inside each record",
            "P2_query_attribute_selector": "queried attribute span at query end",
            "P3_record_binding_write": "record-final attribute/value spans",
            "P4_query_value_load": "queried target-value span",
            "P5_answer_competition": "residual/attention/MLP state at Answer: boundary",
            "P6_matched_rescue": "same-family, same-role, correct-binding donor",
        },
        "future_intervention_registry": {
            "destroy": [
                "queried record body",
                "query attribute selector",
                "query target value",
                "answer-boundary candidate state",
            ],
            "rescue": ["correct same-family same-role donor"],
            "wrong_donor_controls": [
                "wrong attribute donor",
                "wrong entity donor",
                "wrong binding-state donor",
                "wrong role or wrong depth donor",
                "matched-null donor",
            ],
            "restriction": "No component, layer, head, or neuron may be selected before behavior and hidden specificity gates pass on discovery only.",
        },
        "evidence_gates": {
            "Z0_material": "schema AND truth uniqueness AND factorial balance AND split disjointness AND state multiset match AND tokenizer audit AND frozen registry",
            "B_behavior": "finite candidate scores >= 0.99 and worst-cell identity accuracy >= 0.85 on at least two numerically healthy models",
            "R_repeat": "discovery result repeats in confirmation without refitting the registry",
            "N_negative_controls": "active-minus-controls is positive and exceeds prefrozen margins",
            "G_unseen_composition": "held-out entity-profile x attribute combinations pass",
            "U_natural_use": "a separately sourced natural-use corpus passes; this package alone cannot satisfy U",
            "I_causal": "selective destroy plus matched rescue plus wrong-donor rejection passes",
            "X_cross_model": "functional event relation repeats in at least two models without asserting coordinate identity",
            "closure": "Z0 AND B AND R AND N AND G AND U AND I AND X",
        },
        "identifiability_policy": {
            "phase1201_registry_gate_required": True,
            "exact_collision_output": "UNIDENTIFIABLE",
            "local_equivalence_class_merging": "reserved_not_implemented",
            "near_collision_calibration": "reserved_not_implemented",
            "sample_level_ood": "reserved_not_implemented",
        },
        "execution_policy": {
            "this_phase_model_execution": False,
            "this_phase_new_data_type": "deterministic_contract_material_only",
            "this_phase_new_k_item": False,
            "model_order_after_separate_authorization": ["qwen3", "glm4", "deepseek7b"],
            "precision_after_separate_authorization": "FP16 CUDA, one model at a time",
            "hidden_scan_before_behavior_gate": False,
        },
        "claim_boundary": (
            "A passing Phase1202 audit establishes only that the object, controls, splits, token atoms, "
            "and future probe registry are executable and internally identifiable. It is not behavior or mechanism evidence."
        ),
    }
    contract["contract_digest"] = digest(contract)
    return contract


def verify_contract() -> dict[str, Any]:
    contract = read_json(CONTRACT_PATH)
    candidate = {key: value for key, value in contract.items() if key != "contract_digest"}
    if digest(candidate) != contract["contract_digest"]:
        raise RuntimeError("contract digest mismatch")
    if contract["source_hashes"] != source_hashes():
        raise RuntimeError("source changed after contract freeze")
    if contract["upstream"]["upstream_hashes"] != upstream_hashes():
        raise RuntimeError("upstream changed after contract freeze")
    return contract


def selftest() -> None:
    rows = generate_rows()
    split_counts = Counter(row["split"] for row in rows)
    assert dict(split_counts) == EXPECTED_SPLIT_COUNTS
    assert len({row["item_id"] for row in rows}) == len(rows)
    pair_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            row["combination_id"], row["panel"], row["template"], row["candidate_order"]
        )
        pair_groups.setdefault(key, []).append(row)
    assert all(len(group) == 2 for group in pair_groups.values())
    for key, group in pair_groups.items():
        group.sort(key=lambda row: row["binding_state"])
        assert group[0]["prompt_token_multiset_digest"] == group[1]["prompt_token_multiset_digest"]
        if key[1] == "active":
            assert group[0]["gold_candidate"] != group[1]["gold_candidate"]
        else:
            assert group[0]["gold_candidate"] == group[1]["gold_candidate"]
    print(canonical_json({"status": "selftest_pass", "rows": len(rows)}))


def preregister() -> None:
    if CONTRACT_PATH.exists() or PACKAGE_PATH.exists():
        raise RuntimeError("Phase1202 contract or package already exists")
    upstream = read_json(UPSTREAM_FINAL)
    if not upstream["authorized_next"]["theory_and_measurement_consolidation"]:
        raise RuntimeError("Phase1201 did not authorize contract consolidation")
    if upstream["authorized_next"]["natural_language_trace_scan"]:
        raise RuntimeError("Phase1201 unexpectedly authorized a trace scan")
    contract = build_contract()
    write_json(CONTRACT_PATH, contract)
    print(canonical_json({"contract_digest": contract["contract_digest"]}))


def build() -> None:
    contract = verify_contract()
    if PACKAGE_PATH.exists() or SUMMARY_PATH.exists() or TOKEN_AUDIT_PATH.exists():
        raise RuntimeError("Phase1202 material outputs already exist")
    rows = generate_rows()
    token_audit = tokenizer_audit(rows)
    summary = {
        "phase": PHASE,
        "created_at": utc_now(),
        "status": "zero_model_contract_materialized" if token_audit["overall_pass"] else "tokenizer_audit_failed",
        "contract_digest": contract["contract_digest"],
        "factor_summary": factor_summary(rows),
        "package_digest": digest(rows),
        "tokenizer_audit_digest": token_audit["tokenizer_audit_digest"],
        "model_weights_loaded": False,
        "model_behavior_cases_scored": 0,
        "new_k_item": None,
    }
    write_jsonl(PACKAGE_PATH, rows)
    write_json(TOKEN_AUDIT_PATH, token_audit)
    write_json(SUMMARY_PATH, summary)
    print(canonical_json({"status": summary["status"], "factor_summary": summary["factor_summary"]}))


def finalize() -> None:
    contract = verify_contract()
    summary = read_json(SUMMARY_PATH)
    token_audit = read_json(TOKEN_AUDIT_PATH)
    audit = read_json(AUDIT_PATH)
    if not audit.get("gate_pass", False):
        raise RuntimeError("independent audit did not pass")
    if not token_audit.get("overall_pass", False):
        raise RuntimeError("tokenizer audit did not pass")
    final = {
        "phase": PHASE,
        "created_at": utc_now(),
        "status": "mother_family_contract_ready",
        "contract_digest": contract["contract_digest"],
        "package_digest": summary["package_digest"],
        "tokenizer_audit_digest": token_audit["tokenizer_audit_digest"],
        "independent_audit_digest": audit["audit_digest"],
        "evidence_scope": {
            "kind": "zero-model measurement contract",
            "new_k_item": None,
            "canonical_k_range": "K1-K183",
            "behavior_evidence": False,
            "hidden_state_evidence": False,
            "causal_evidence": False,
            "natural_corpus_evidence": False,
        },
        "authorized_next": {
            "phase1203_behavior_protocol_preregistration": True,
            "automatic_model_execution": False,
            "hidden_state_scan": False,
            "causal_intervention": False,
            "new_mechanism_algebra": False,
        },
        "stop_reason": (
            "The zero-model object is ready, but model scoring requires a separately sealed behavior protocol; "
            "hidden and causal work remain gated by behavior, controls, confirmation, and unseen composition."
        ),
    }
    final["final_digest"] = digest(final)
    write_json(FINAL_PATH, final)
    print(canonical_json({"status": final["status"], "authorized_next": final["authorized_next"], "final_digest": final["final_digest"]}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("selftest", "preregister", "build", "finalize"))
    command = parser.parse_args().command
    {
        "selftest": selftest,
        "preregister": preregister,
        "build": build,
        "finalize": finalize,
    }[command]()


if __name__ == "__main__":
    main()
