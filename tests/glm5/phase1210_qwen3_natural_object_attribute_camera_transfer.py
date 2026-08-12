#!/usr/bin/env python3
"""Blind intervention-response transfer on fresh naturalized Qwen3 materials.

Phase1210 is deliberately narrower than a language-mechanism claim.  It uses
fresh entities, values, prose templates, and combinations, freezes four
residual events per attribute from discovery only, and asks whether the
Phase1208 camera predicts triple/all-event responses before they are run.
"""

from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import itertools
import json
import math
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests/glm5"
sys.path.insert(0, str(TEST_ROOT))

from model_utils import get_layers  # noqa: E402
from phase1023_fp16_utils import load_fp16, quantization_audit, release_fp16  # noqa: E402
import phase1203_object_attribute_behavior_protocol as phase1203  # noqa: E402
import phase1208_necessity_mediation_camera_calibration as camera  # noqa: E402


PHASE = 1210
MODEL = "qwen3"
MODEL_PATH = ROOT / "models/hf/qwen3-4b"
SOURCE1202 = TEST_ROOT / "result/phase1202_object_attribute_mother_contract/material/object_attribute_binding.jsonl"
SOURCE1208 = TEST_ROOT / "result/phase1208_necessity_mediation_camera_calibration"
SOURCE1209 = TEST_ROOT / "result/phase1209_free_transformer_necessity_camera_transfer"
OUT_ROOT = TEST_ROOT / "result/phase1210_qwen3_natural_object_attribute_camera_transfer"
SCRIPT = Path(__file__).resolve()
AUDIT_SCRIPT = SCRIPT.with_name("phase1210_qwen3_natural_object_attribute_camera_transfer_audit.py")

PROTOCOL_PATH = OUT_ROOT / "protocol/preregistration.json"
MATERIAL_PATH = OUT_ROOT / "material/sealed_groups.jsonl.gz"
PREAUDIT_PATH = OUT_ROOT / "audit/independent_preaudit.json"
FROZEN_SITES_PATH = OUT_ROOT / "analysis/frozen_sites.json"
FINAL_PATH = OUT_ROOT / "analysis/final.json"

EXPECTED_1209_FINAL = "65f655a6f55fd107759a67165af1ebaab792296bf68af4f088bb5866bee15fb3"
EXPECTED_1209_AUDIT = "c0d23608937793e6fb3b89c8d60180db5db42944ca60ef31062e143b650e4629"

ATTRIBUTES = ("color", "material", "location", "condition", "texture", "pattern")
NEIGHBOR = {
    "color": "material",
    "material": "location",
    "location": "condition",
    "condition": "texture",
    "texture": "pattern",
    "pattern": "color",
}
PANELS = ("active", "matched_null", "semantic_neighbor", "surface_only")
EVENT_ROLES = ("record_value0", "record_value1", "query_attribute", "query_value", "generation_boundary")
SCOUT_DEPTHS = (18, 24, 30, 33, 36)
TOP_EVENTS = 4
LAYER_COUNT = 36
HIDDEN_SIZE = 2560
BATCH_SIZE = 8
EPSILON = 1.0e-8
TIE_TOLERANCE = 1.0e-6
PREDICTION_KEYS = (
    "max_triple_ablation_damage",
    "all_hidden_ablation_damage",
    "all_hidden_donor_choice",
)

ENTITY_TRIPLES = {
    "discovery": (
        ("vase", "lantern", "cushion"),
        ("kettle", "mirror", "basket"),
        ("cabinet", "carpet", "pitcher"),
    ),
    "confirmation": (
        ("plaque", "curtain", "pedestal"),
        ("helmet", "folder", "tripod"),
        ("barrel", "canopy", "basin"),
    ),
}

VALUE_SETS = {
    "discovery": {
        "color": ("amber", "ivory", "beige"),
        "material": ("brass", "ceramic", "velvet"),
        "location": ("cellar", "hallway", "terrace"),
        "condition": ("polished", "sealed", "wrapped"),
        "texture": ("coarse", "silky", "rough"),
        "pattern": ("striped", "spotted", "plain"),
    },
    "confirmation": {
        "color": ("crimson", "teal", "silver"),
        "material": ("marble", "bronze", "linen"),
        "location": ("gallery", "foyer", "attic"),
        "condition": ("cracked", "restored", "intact"),
        "texture": ("sleek", "woven", "glossy"),
        "pattern": ("dotted", "floral", "checked"),
    },
}

TEMPLATES = {
    "discovery": ("restoration_hall", "curator_note"),
    "confirmation": ("estate_survey", "conservation_report"),
}

THRESHOLDS = {
    "finite_fraction_min": 1.0,
    "active_accuracy_min": 1.0,
    "control_accuracy_min": 0.98,
    "scout_active_donor_choice_min": 0.50,
    "scout_control_gap_min": 0.25,
    "nonabstain_attributes_min": 3,
    "matched_null_max_damage_max": 0.25,
    "carrier_max_damage_max": 0.25,
    "holdout_mae_max": 0.20,
    "holdout_max_abs_error_max": 0.60,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl_gz(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".pending")
    with gzip.open(temporary, "wt", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical(row) + "\n")
    os.replace(temporary, path)


def read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def validate_digest(value: dict[str, Any], field: str) -> None:
    clean = dict(value)
    stored = clean.pop(field)
    if digest(clean) != stored:
        raise RuntimeError(f"digest mismatch: {field}")


def source_hashes() -> dict[str, str]:
    return {
        "main": sha256_file(SCRIPT),
        "audit": sha256_file(AUDIT_SCRIPT),
        "phase1208": sha256_file(Path(camera.__file__).resolve()),
    }


def event_registry() -> list[dict[str, Any]]:
    return [
        {"event_id": f"residual_d{depth:02d}_{role}", "depth": depth, "role": role, "component": "residual"}
        for depth in SCOUT_DEPTHS
        for role in EVENT_ROLES
    ]


def base_assignments(split: str, profile_index: int) -> dict[str, dict[str, str]]:
    entities = ENTITY_TRIPLES[split][profile_index]
    assignments = {entity: {} for entity in entities}
    for attribute_index, attribute in enumerate(ATTRIBUTES):
        values = VALUE_SETS[split][attribute]
        shift = (profile_index + attribute_index) % 3
        for entity_index, entity in enumerate(entities):
            assignments[entity][attribute] = values[(entity_index + shift) % 3]
    return assignments


def swapped(assignments: dict[str, dict[str, str]], entities: tuple[str, str, str], attribute: str) -> dict[str, dict[str, str]]:
    output = {entity: dict(values) for entity, values in assignments.items()}
    output[entities[0]][attribute], output[entities[1]][attribute] = (
        output[entities[1]][attribute], output[entities[0]][attribute]
    )
    return output


def permuted(
    assignments: dict[str, dict[str, str]],
    entities: tuple[str, str, str],
    attribute: str,
    permutation: tuple[int, int, int],
) -> dict[str, dict[str, str]]:
    output = {entity: dict(values) for entity, values in assignments.items()}
    values = [assignments[entity][attribute] for entity in entities]
    for entity_index, entity in enumerate(entities):
        output[entity][attribute] = values[permutation[entity_index]]
    return output


def render_record(template: str, entity: str, values: dict[str, str]) -> str:
    if template == "restoration_hall":
        return (
            f"In the restoration hall, the {entity} is {values['color']} in color, made from {values['material']}, "
            f"kept in the {values['location']}, currently {values['condition']}, {values['texture']} to the touch, "
            f"and marked with a {values['pattern']} pattern."
        )
    if template == "curator_note":
        return (
            f"A curator's note records the {entity}: color {values['color']}; material {values['material']}; "
            f"location {values['location']}; condition {values['condition']}; texture {values['texture']}; "
            f"pattern {values['pattern']}."
        )
    if template == "estate_survey":
        return (
            f"During an evening estate survey, staff found the {entity} in the {values['location']}. Its surface "
            f"looked {values['color']}; it was fashioned from {values['material']}, remained {values['condition']}, "
            f"felt {values['texture']}, and showed a {values['pattern']} motif."
        )
    if template == "conservation_report":
        return (
            f"According to the conservation report, the {entity} was catalogued from the {values['location']}. "
            f"The report lists color {values['color']}, material {values['material']}, condition {values['condition']}, "
            f"texture {values['texture']}, and pattern {values['pattern']}."
        )
    raise KeyError(template)


def rendered_case(
    tokenizer: Any,
    split: str,
    group_id: str,
    panel: str,
    state_id: str,
    template: str,
    entities: tuple[str, str, str],
    candidate_order: tuple[int, int, int],
    record_order: tuple[int, int, int],
    assignments: dict[str, dict[str, str]],
    attribute: str,
    target_value: str,
) -> dict[str, Any]:
    records = [render_record(template, entities[index], assignments[entities[index]]) for index in record_order]
    candidates = [entities[index] for index in candidate_order]
    query = (
        f"Using only these descriptions, which object has the recorded {attribute} value {target_value}? "
        f"Reply with exactly one of: {', '.join(candidates)}. Answer:"
    )
    prompt = " ".join(records + [query])
    rendered = phase1203.render_native(tokenizer, MODEL, prompt)
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    input_ids = [int(value) for value in encoded["input_ids"]]
    offsets = [(int(left), int(right)) for left, right in encoded["offset_mapping"]]
    prompt_start = rendered.find(prompt)
    if prompt_start < 0 or rendered.find(prompt, prompt_start + 1) >= 0:
        raise RuntimeError("rendered prompt placement is not unique")

    record_bounds: dict[str, tuple[int, int]] = {}
    cursor = prompt_start
    for record_index in record_order:
        entity = entities[record_index]
        record = render_record(template, entity, assignments[entity])
        left = rendered.find(record, cursor)
        if left < 0:
            raise RuntimeError(f"record not found for {entity}")
        record_bounds[entity] = (left, left + len(record))
        cursor = left + len(record)
    query_left = rendered.find(query, cursor)
    if query_left < 0:
        raise RuntimeError("query not found")

    def token_at(left: int, right: int) -> int:
        indices = [index for index, (a, b) in enumerate(offsets) if b > left and a < right and b > a]
        if not indices:
            raise RuntimeError(f"no token for span {left}:{right}")
        return int(indices[-1])

    positions: dict[str, int] = {}
    for entity_index in (0, 1):
        entity = entities[entity_index]
        left, right = record_bounds[entity]
        scoped = rendered[left:right]
        entity_offset = scoped.find(entity)
        value = assignments[entity][attribute]
        value_offset = scoped.find(value)
        if entity_offset < 0 or value_offset < 0:
            raise RuntimeError("record role missing")
        positions[f"record_entity{entity_index}"] = token_at(left + entity_offset, left + entity_offset + len(entity))
        positions[f"record_value{entity_index}"] = token_at(left + value_offset, left + value_offset + len(value))
    query_scope = rendered[query_left : query_left + len(query)]
    attribute_offset = query_scope.find(attribute)
    value_offset = query_scope.find(target_value)
    if attribute_offset < 0 or value_offset < 0:
        raise RuntimeError("query role missing")
    positions["query_attribute"] = token_at(
        query_left + attribute_offset, query_left + attribute_offset + len(attribute)
    )
    positions["query_value"] = token_at(query_left + value_offset, query_left + value_offset + len(target_value))
    positions["generation_boundary"] = len(input_ids) - 1

    candidate_token_ids = {
        candidate: phase1203.continuation_ids(tokenizer, rendered, candidate) for candidate in candidates
    }
    if any(len(ids) != 1 for ids in candidate_token_ids.values()):
        raise RuntimeError(f"multi-token candidate in {group_id}: {candidate_token_ids}")
    if len({ids[0] for ids in candidate_token_ids.values()}) != 3:
        raise RuntimeError("candidate token collision")
    gold = next(entity for entity in entities if assignments[entity][attribute] == target_value)
    return {
        "case_id": digest([group_id, panel, state_id, prompt])[:24],
        "group_id": group_id,
        "split": split,
        "panel": panel,
        "state_id": state_id,
        "template": template,
        "template_context": TEMPLATES[split].index(template),
        "attribute": attribute,
        "entities": list(entities),
        "candidates": candidates,
        "candidate_token_ids": candidate_token_ids,
        "gold": gold,
        "input_ids": input_ids,
        "input_length": len(input_ids),
        "positions": positions,
        "prompt_digest": digest(prompt),
    }


def build_material(tokenizer: Any) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for split in ("discovery", "confirmation"):
        for profile_index, entities in enumerate(ENTITY_TRIPLES[split]):
            base = base_assignments(split, profile_index)
            for attribute in ATTRIBUTES:
                values = [base[entity][attribute] for entity in entities]
                for template in TEMPLATES[split]:
                    for order_index, candidate_order in enumerate(((0, 1, 2), (2, 0, 1))):
                        group_id = f"p1210:{split}:p{profile_index}:{attribute}:{template}:o{order_index}"
                        active0 = base
                        active1 = swapped(base, entities, attribute)
                        neighbor = NEIGHBOR[attribute]
                        neighbor1 = swapped(base, entities, neighbor)
                        panel_specs = {
                            "active": ((active0, (0, 1, 2), values[0]), (active1, (0, 1, 2), values[0])),
                            "matched_null": ((active0, (0, 1, 2), values[2]), (active1, (0, 1, 2), values[2])),
                            "semantic_neighbor": ((base, (0, 1, 2), values[0]), (neighbor1, (0, 1, 2), values[0])),
                            "surface_only": ((base, (0, 1, 2), values[0]), (base, (1, 0, 2), values[0])),
                        }
                        panels: dict[str, list[dict[str, Any]]] = {}
                        for panel, specifications in panel_specs.items():
                            panels[panel] = [
                                rendered_case(
                                    tokenizer, split, group_id, panel, f"state{state_index}", template, entities,
                                    candidate_order, record_order, assignments, attribute, target,
                                )
                                for state_index, (assignments, record_order, target) in enumerate(specifications)
                            ]
                        neutral: dict[str, list[dict[str, Any]]] = {}
                        for neutral_name, target in (("active", values[0]), ("matched_null", values[2])):
                            neutral[neutral_name] = [
                                rendered_case(
                                    tokenizer, split, group_id, f"neutral_{neutral_name}", f"perm{index}", template,
                                    entities, candidate_order, (0, 1, 2),
                                    permuted(base, entities, attribute, permutation), attribute, target,
                                )
                                for index, permutation in enumerate(itertools.permutations((0, 1, 2)))
                            ]
                        all_lengths = [row["input_length"] for rows in panels.values() for row in rows]
                        for rows in neutral.values():
                            all_lengths.extend(row["input_length"] for row in rows)
                        if len(set(all_lengths)) != 1:
                            raise RuntimeError(f"length imbalance in {group_id}: {sorted(set(all_lengths))}")
                        groups.append({
                            "group_id": group_id,
                            "split": split,
                            "profile_index": profile_index,
                            "attribute": attribute,
                            "template": template,
                            "template_context": TEMPLATES[split].index(template),
                            "entities": list(entities),
                            "panels": panels,
                            "neutral": neutral,
                        })
    return groups


def material_summary(groups: list[dict[str, Any]]) -> dict[str, Any]:
    old_rows = [json.loads(line) for line in SOURCE1202.read_text(encoding="utf-8").splitlines() if line.strip()]
    old_entities = {str(entity) for row in old_rows for entity in row["entities"]}
    old_values = {
        str(value) for row in old_rows for assignment in row["assignments"].values() for value in assignment.values()
    }
    new_entities = {entity for triples in ENTITY_TRIPLES.values() for triple in triples for entity in triple}
    new_values = {value for split in VALUE_SETS.values() for values in split.values() for value in values}
    split_groups = {split: [row for row in groups if row["split"] == split] for split in ("discovery", "confirmation")}
    case_rows = [case for group in groups for rows in group["panels"].values() for case in rows]
    neutral_rows = [case for group in groups for rows in group["neutral"].values() for case in rows]
    return {
        "group_count": len(groups),
        "groups_per_split": {split: len(rows) for split, rows in split_groups.items()},
        "groups_per_attribute_split": {
            split: {attribute: sum(row["attribute"] == attribute for row in rows) for attribute in ATTRIBUTES}
            for split, rows in split_groups.items()
        },
        "panel_case_count": len(case_rows),
        "neutral_case_count": len(neutral_rows),
        "all_candidates_single_token": all(
            len(ids) == 1 for row in case_rows + neutral_rows for ids in row["candidate_token_ids"].values()
        ),
        "all_group_lengths_exact": all(
            len({case["input_length"] for rows in group["panels"].values() for case in rows}
                | {case["input_length"] for rows in group["neutral"].values() for case in rows}) == 1
            for group in groups
        ),
        "discovery_confirmation_entities_disjoint": set(sum(ENTITY_TRIPLES["discovery"], ())).isdisjoint(
            set(sum(ENTITY_TRIPLES["confirmation"], ()))
        ),
        "discovery_confirmation_values_disjoint": set(sum(VALUE_SETS["discovery"].values(), ())).isdisjoint(
            set(sum(VALUE_SETS["confirmation"].values(), ()))
        ),
        "discovery_confirmation_templates_disjoint": set(TEMPLATES["discovery"]).isdisjoint(TEMPLATES["confirmation"]),
        "new_entities_exclude_phase1202": new_entities.isdisjoint(old_entities),
        "new_values_exclude_phase1202": new_values.isdisjoint(old_values),
        "material_digest": digest(groups),
    }


def protocol_payload(groups: list[dict[str, Any]], summary: dict[str, Any]) -> dict[str, Any]:
    final1209 = read_json(SOURCE1209 / "analysis/final.json")
    audit1209 = read_json(SOURCE1209 / "audit/independent_audit.json")
    validate_digest(final1209, "final_digest")
    validate_digest(audit1209, "audit_digest")
    checks = {
        "phase1209_final": final1209["final_digest"] == EXPECTED_1209_FINAL,
        "phase1209_audit": audit1209["audit_digest"] == EXPECTED_1209_AUDIT and audit1209["all_checks_passed"],
        "phase1209_transfer_passed": final1209["learned_micro_transformer_external_validity"] is True,
        "phase1209_auto_stopped": final1209["auto_continue"] is False,
        "material_groups": summary["group_count"] == 144,
        "material_balance": all(value == 12 for rows in summary["groups_per_attribute_split"].values() for value in rows.values()),
        "candidate_tokens": summary["all_candidates_single_token"],
        "lengths": summary["all_group_lengths_exact"],
        "split_entities": summary["discovery_confirmation_entities_disjoint"],
        "split_values": summary["discovery_confirmation_values_disjoint"],
        "split_templates": summary["discovery_confirmation_templates_disjoint"],
        "new_entities": summary["new_entities_exclude_phase1202"],
        "new_values": summary["new_values_exclude_phase1202"],
        "pretrained_scan_is_new_protocol": True,
        "cuda_required": True,
    }
    if not all(checks.values()):
        raise RuntimeError(f"protocol checks failed: {checks}")
    payload = {
        "phase": PHASE,
        "schema_version": "phase1210.qwen3_natural_camera_transfer.v1",
        "created_at_utc": utc_now(),
        "title": "Qwen3 fresh naturalized object-attribute blind intervention-response confirmation",
        "source_hashes": source_hashes(),
        "source_phase1209_final_digest": final1209["final_digest"],
        "source_phase1209_audit_digest": audit1209["audit_digest"],
        "model": {"name": MODEL, "path": str(MODEL_PATH.resolve()), "precision": "FP16", "quantization": "none", "placement": "full_cuda"},
        "material": summary,
        "material_file_sha256": sha256_file(MATERIAL_PATH),
        "attributes": list(ATTRIBUTES),
        "panels": list(PANELS),
        "event_registry": event_registry(),
        "top_events_per_attribute": TOP_EVENTS,
        "batch_size": BATCH_SIZE,
        "camera_thresholds": camera.CAMERA_THRESHOLDS,
        "thresholds": THRESHOLDS,
        "prediction_keys": list(PREDICTION_KEYS),
        "execution_order": [
            "preregister and independent zero-output preaudit",
            "run discovery behavior gate",
            "scout 25 frozen residual role-depth events and freeze four per attribute",
            "measure discovery low-order intervention table and seal quotient/holdout predictions",
            "measure discovery triple/all-event holdouts and score",
            "only if discovery passes: run confirmation behavior and frozen low-order table",
            "seal confirmation predictions before holdout",
            "measure confirmation holdout and score",
            "independent replay/result audit",
        ],
        "hard_stops": [
            "No Phase1202-1207 entity, value, template, or measured response may tune this protocol; the broad depth grid is not an inherited hotspot claim.",
            "The material is controlled naturalized prose, not an organic-corpus or all-language claim.",
            "If behavior fails, no hidden state is captured.",
            "Discovery alone selects four events per attribute; confirmation cannot reselect.",
            "The camera may abstain; no label is treated as latent ground truth.",
            "All high-order predictions are hashed before their interventions run.",
            "A pass is Qwen3-only naturalized task-family evidence, not cross-model or brain closure.",
            "No rescue, head, neuron, or Phase1211 search is automatic.",
        ],
        "checks": checks,
    }
    payload["protocol_digest"] = digest(payload)
    return payload


def tokenizer_instance() -> Any:
    return AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)


def dev_material() -> dict[str, Any]:
    tokenizer = tokenizer_instance()
    groups = build_material(tokenizer)
    value = material_summary(groups)
    print(json.dumps(value, ensure_ascii=False, indent=2))
    return value


def preregister() -> dict[str, Any]:
    if OUT_ROOT.exists():
        raise RuntimeError("refusing to overwrite Phase1210 artifacts")
    tokenizer = tokenizer_instance()
    groups = build_material(tokenizer)
    summary = material_summary(groups)
    write_jsonl_gz(MATERIAL_PATH, groups)
    payload = protocol_payload(groups, summary)
    write_json(PROTOCOL_PATH, payload)
    print(json.dumps({"protocol_digest": payload["protocol_digest"], "material": summary}, ensure_ascii=False, indent=2))
    return payload


def verify_protocol() -> dict[str, Any]:
    protocol = read_json(PROTOCOL_PATH)
    validate_digest(protocol, "protocol_digest")
    if protocol["source_hashes"] != source_hashes():
        raise RuntimeError("source hash drift")
    if protocol["material_file_sha256"] != sha256_file(MATERIAL_PATH):
        raise RuntimeError("material file drift")
    groups = read_jsonl_gz(MATERIAL_PATH)
    if protocol["material"]["material_digest"] != digest(groups):
        raise RuntimeError("material semantic drift")
    return protocol


def require_preaudit() -> dict[str, Any]:
    audit = read_json(PREAUDIT_PATH)
    validate_digest(audit, "audit_digest")
    if not audit["all_checks_passed"] or audit["protocol_digest"] != verify_protocol()["protocol_digest"]:
        raise RuntimeError("independent preaudit failed")
    return audit


def split_groups(split: str) -> list[dict[str, Any]]:
    return [row for row in read_jsonl_gz(MATERIAL_PATH) if row["split"] == split]


def flatten_panel(groups: list[dict[str, Any]], panel: str) -> list[dict[str, Any]]:
    return [case for group in groups for case in group["panels"][panel]]


def samples_for(groups: list[dict[str, Any]], panel: str) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for group in groups:
        states = group["panels"][panel]
        for receiver_index in (0, 1):
            receiver = states[receiver_index]
            donor = states[1 - receiver_index]
            samples.append({
                "group_id": group["group_id"],
                "attribute": group["attribute"],
                "context": group["template_context"],
                "receiver": receiver,
                "donor": donor,
                "neutral": group["neutral"]["active" if panel != "matched_null" else "matched_null"],
            })
    return samples


def bucket_indices(rows: list[dict[str, Any]]) -> list[list[int]]:
    by_length: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_length[int(row["input_length"])].append(index)
    batches: list[list[int]] = []
    for length in sorted(by_length):
        indices = by_length[length]
        for start in range(0, len(indices), BATCH_SIZE):
            batches.append(indices[start : start + BATCH_SIZE])
    return batches


def input_batch(rows: list[dict[str, Any]], indices: list[int], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    ids = torch.tensor([rows[index]["input_ids"] for index in indices], dtype=torch.long, device=device)
    mask = torch.ones_like(ids)
    return ids, mask


def candidate_scores(logits: torch.Tensor, rows: list[dict[str, Any]], indices: list[int]) -> np.ndarray:
    last = logits[:, -1].float()
    output = np.empty((len(indices), 3), dtype=np.float32)
    for local, index in enumerate(indices):
        row = rows[index]
        token_ids = [int(row["candidate_token_ids"][candidate][0]) for candidate in row["candidates"]]
        output[local] = last[local, token_ids].detach().cpu().numpy()
    return output


def run_plain_logits(model: Any, rows: list[dict[str, Any]], device: torch.device) -> np.ndarray:
    output = np.empty((len(rows), 3), dtype=np.float32)
    with torch.inference_mode():
        for indices in bucket_indices(rows):
            ids, mask = input_batch(rows, indices, device)
            result = model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True, logits_to_keep=1)
            output[indices] = candidate_scores(result.logits, rows, indices)
    return output


class EventCapture:
    def __init__(self, layers: list[Any], events: list[dict[str, Any]]):
        self.layers = layers
        self.events = events
        self.rows: list[dict[str, Any]] = []
        self.indices: list[int] = []
        self.values: dict[int, torch.Tensor] = {}
        self.calls: dict[int, int] = defaultdict(int)
        self.handles: list[Any] = []

    def _hook(self, depth: int):
        event_indices = [index for index, event in enumerate(self.events) if int(event["depth"]) == depth]
        def hook(module: Any, args: Any, output: Any):
            value = output[0] if isinstance(output, tuple) else output
            batch = torch.arange(value.shape[0], device=value.device)[:, None]
            positions = torch.tensor(
                [[int(self.rows[index]["positions"][self.events[event_index]["role"]]) for event_index in event_indices]
                 for index in self.indices],
                dtype=torch.long,
                device=value.device,
            )
            self.values[depth] = value[batch, positions, :].detach().float().cpu()
            self.calls[depth] += 1
            return output
        return hook

    def register(self) -> None:
        for depth in sorted({int(event["depth"]) for event in self.events}):
            self.handles.append(self.layers[depth - 1].register_forward_hook(self._hook(depth)))

    def begin(self, rows: list[dict[str, Any]], indices: list[int]) -> None:
        self.rows = rows
        self.indices = indices
        self.values = {}
        self.calls = defaultdict(int)

    def close(self) -> None:
        for handle in reversed(self.handles):
            handle.remove()
        self.handles = []


def capture_states(model: Any, layers: list[Any], rows: list[dict[str, Any]], events: list[dict[str, Any]], device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    states = np.empty((len(rows), len(events), HIDDEN_SIZE), dtype=np.float32)
    logits = np.empty((len(rows), 3), dtype=np.float32)
    capture = EventCapture(layers, events)
    capture.register()
    try:
        with torch.inference_mode():
            for indices in bucket_indices(rows):
                capture.begin(rows, indices)
                ids, mask = input_batch(rows, indices, device)
                result = model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True, logits_to_keep=1)
                logits[indices] = candidate_scores(result.logits, rows, indices)
                for depth in sorted({int(event["depth"]) for event in events}):
                    if capture.calls[depth] != 1:
                        raise RuntimeError(f"capture call mismatch at depth {depth}")
                    local_events = [index for index, event in enumerate(events) if int(event["depth"]) == depth]
                    states[np.ix_(indices, local_events, range(HIDDEN_SIZE))] = capture.values[depth].numpy()
    finally:
        capture.close()
    return logits, states


def captured_bundle(
    model: Any,
    layers: list[Any],
    samples: list[dict[str, Any]],
    events: list[dict[str, Any]],
    device: torch.device,
) -> dict[str, Any]:
    receiver_rows = [sample["receiver"] for sample in samples]
    donor_rows = [sample["donor"] for sample in samples]
    receiver_logits, receiver_states = capture_states(model, layers, receiver_rows, events, device)
    _donor_logits, donor_states = capture_states(model, layers, donor_rows, events, device)
    unique_groups: dict[str, list[dict[str, Any]]] = {}
    for sample in samples:
        unique_groups[sample["group_id"]] = sample["neutral"]
    neutral_rows: list[dict[str, Any]] = []
    neutral_slices: dict[str, tuple[int, int]] = {}
    for group_id in sorted(unique_groups):
        left = len(neutral_rows)
        neutral_rows.extend(unique_groups[group_id])
        neutral_slices[group_id] = (left, len(neutral_rows))
    _neutral_logits, neutral_flat = capture_states(model, layers, neutral_rows, events, device)
    neutral_by_group = {
        group_id: neutral_flat[left:right].mean(axis=0)
        for group_id, (left, right) in neutral_slices.items()
    }
    neutral_states = np.stack([neutral_by_group[sample["group_id"]] for sample in samples], axis=0)
    return {
        "rows": receiver_rows,
        "receiver_logits": receiver_logits,
        "receiver_states": receiver_states,
        "donor_states": donor_states,
        "neutral_states": neutral_states,
    }


def run_patched_logits(
    model: Any,
    layers: list[Any],
    captured: dict[str, Any],
    events: list[dict[str, Any]],
    operations: dict[int, str],
    device: torch.device,
) -> np.ndarray:
    rows = captured["rows"]
    result_logits = np.empty((len(rows), 3), dtype=np.float32)
    by_depth: dict[int, list[tuple[int, str]]] = defaultdict(list)
    for slot, mode in operations.items():
        by_depth[int(events[slot]["depth"])].append((slot, mode))
    for indices in bucket_indices(rows):
        calls: dict[int, int] = defaultdict(int)
        handles: list[Any] = []
        for depth, depth_operations in by_depth.items():
            def make_hook(d: int, ops: list[tuple[int, str]]):
                def hook(module: Any, args: Any, output: Any):
                    value = output[0] if isinstance(output, tuple) else output
                    patched = value.clone()
                    batch = torch.arange(value.shape[0], device=value.device)
                    for slot, mode in ops:
                        positions = torch.tensor(
                            [int(rows[index]["positions"][events[slot]["role"]]) for index in indices],
                            dtype=torch.long,
                            device=value.device,
                        )
                        source = captured[f"{mode}_states"][indices, slot]
                        source_tensor = torch.from_numpy(source).to(value.device, dtype=value.dtype)
                        patched[batch, positions, :] = source_tensor
                    calls[d] += 1
                    return (patched,) + output[1:] if isinstance(output, tuple) else patched
                return hook
            handles.append(layers[depth - 1].register_forward_hook(make_hook(depth, depth_operations)))
        try:
            with torch.inference_mode():
                ids, mask = input_batch(rows, indices, device)
                output = model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True, logits_to_keep=1)
                result_logits[indices] = candidate_scores(output.logits, rows, indices)
            if any(calls[depth] != 1 for depth in by_depth):
                raise RuntimeError(f"patch call mismatch: {dict(calls)}")
        finally:
            for handle in reversed(handles):
                handle.remove()
    return result_logits


def target_indices(samples: list[dict[str, Any]], donor: bool = False) -> np.ndarray:
    output = []
    for sample in samples:
        row = sample["receiver"]
        label = sample["donor"]["gold"] if donor else row["gold"]
        output.append(row["candidates"].index(label))
    return np.asarray(output, dtype=np.int64)


def accuracy(logits: np.ndarray, targets: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is None:
        mask = np.ones(len(targets), dtype=np.bool_)
    return float(np.mean(np.argmax(logits[mask], axis=1) == targets[mask]))


def median_margin(logits: np.ndarray, receiver: np.ndarray, donor: np.ndarray) -> float:
    rows = np.arange(len(logits))
    return float(np.median(logits[rows, receiver] - logits[rows, donor]))


def normalized_damage(base_accuracy: float, current_accuracy: float) -> float:
    return float((base_accuracy - current_accuracy) / max(base_accuracy - 1.0 / 3.0, EPSILON))


def behavior_rows(groups: list[dict[str, Any]], model: Any, device: torch.device) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw: list[dict[str, Any]] = []
    for panel in PANELS:
        rows = flatten_panel(groups, panel)
        logits = run_plain_logits(model, rows, device)
        for row, scores in zip(rows, logits):
            target = row["candidates"].index(row["gold"])
            predicted = int(np.argmax(scores))
            raw.append({
                "case_id": row["case_id"],
                "split": row["split"],
                "panel": panel,
                "attribute": row["attribute"],
                "finite": bool(np.isfinite(scores).all()),
                "prediction": row["candidates"][predicted],
                "gold": row["gold"],
                "correct": predicted == target,
                "candidate_scores": [float(value) for value in scores],
            })
    panel_accuracy = {
        panel: float(np.mean([row["correct"] for row in raw if row["panel"] == panel])) for panel in PANELS
    }
    summary = {
        "case_count": len(raw),
        "finite_fraction": float(np.mean([row["finite"] for row in raw])),
        "panel_accuracy": panel_accuracy,
    }
    summary["gate"] = bool(
        summary["finite_fraction"] >= THRESHOLDS["finite_fraction_min"]
        and panel_accuracy["active"] >= THRESHOLDS["active_accuracy_min"]
        and all(panel_accuracy[panel] >= THRESHOLDS["control_accuracy_min"] for panel in PANELS if panel != "active")
    )
    summary["summary_digest"] = digest(summary)
    return raw, summary


def scout_sites(
    model: Any, layers: list[Any], groups: list[dict[str, Any]], device: torch.device
) -> dict[str, Any]:
    candidates = event_registry()
    selected: dict[str, list[dict[str, Any]]] = {}
    profiles: dict[str, list[dict[str, Any]]] = {}
    checks: dict[str, bool] = {}
    for attribute in ATTRIBUTES:
        factor_groups = [group for group in groups if group["attribute"] == attribute]
        active_samples = samples_for(factor_groups, "active")
        null_samples = samples_for(factor_groups, "matched_null")
        active_bundle = captured_bundle(model, layers, active_samples, candidates, device)
        null_bundle = captured_bundle(model, layers, null_samples, candidates, device)
        active_donor = target_indices(active_samples, donor=True)
        null_receiver = target_indices(null_samples)
        null_base_accuracy = accuracy(null_bundle["receiver_logits"], null_receiver)
        rows: list[dict[str, Any]] = []
        for event_index, event in enumerate(candidates):
            active_changed = run_patched_logits(
                model, layers, active_bundle, candidates, {event_index: "donor"}, device
            )
            null_changed = run_patched_logits(
                model, layers, null_bundle, candidates, {event_index: "donor"}, device
            )
            active_choice = accuracy(active_changed, active_donor)
            null_damage = abs(normalized_damage(null_base_accuracy, accuracy(null_changed, null_receiver)))
            rows.append({
                **event,
                "candidate_index": event_index,
                "active_donor_choice": active_choice,
                "matched_null_damage": null_damage,
                "control_gap": active_choice - null_damage,
            })
        ranked = sorted(
            rows,
            key=lambda row: (-row["active_donor_choice"], row["matched_null_damage"], int(row["depth"]), str(row["role"])),
        )
        chosen = ranked[:TOP_EVENTS]
        selected[attribute] = chosen
        profiles[attribute] = rows
        checks[attribute] = bool(
            chosen[-1]["active_donor_choice"] >= THRESHOLDS["scout_active_donor_choice_min"]
            and chosen[-1]["control_gap"] >= THRESHOLDS["scout_control_gap_min"]
        )
        print(canonical({"scout_attribute": attribute, "selected": chosen, "pass": checks[attribute]}), flush=True)
    value = {
        "phase": PHASE,
        "selection_split": "discovery",
        "candidate_count": len(candidates),
        "top_events": TOP_EVENTS,
        "selected": selected,
        "profiles": profiles,
        "checks": checks,
        "measurement_authorized": all(checks.values()),
    }
    value["site_digest"] = digest(value)
    return value


def evaluate_bundle(
    model: Any,
    layers: list[Any],
    samples: list[dict[str, Any]],
    events: list[dict[str, Any]],
    device: torch.device,
) -> tuple[dict[str, Any], Any]:
    captured = captured_bundle(model, layers, samples, events, device)
    receiver = target_indices(samples)
    donor = target_indices(samples, donor=True)
    baseline = captured["receiver_logits"]
    base_accuracy = accuracy(baseline, receiver)
    base_margin = median_margin(baseline, receiver, donor)
    contexts = np.asarray([sample["context"] for sample in samples], dtype=np.int64)

    def evaluate(operations: dict[int, str]) -> dict[str, Any]:
        changed = run_patched_logits(model, layers, captured, events, operations, device)
        current_accuracy = accuracy(changed, receiver)
        return {
            "behavior_damage": normalized_damage(base_accuracy, current_accuracy),
            "margin_damage": float((base_margin - median_margin(changed, receiver, donor)) / max(abs(base_margin), EPSILON)),
            "donor_choice": accuracy(changed, donor),
            "context_behavior_damage": [
                normalized_damage(accuracy(baseline, receiver, contexts == context), accuracy(changed, receiver, contexts == context))
                for context in (0, 1)
            ],
            "context_donor_choice": [accuracy(changed, donor, contexts == context) for context in (0, 1)],
        }

    return {"captured": captured, "receiver": receiver, "donor": donor, "baseline_accuracy": base_accuracy, "baseline_margin": base_margin}, evaluate


def low_order_row(
    model: Any,
    layers: list[Any],
    split: str,
    attribute: str,
    groups: list[dict[str, Any]],
    events: list[dict[str, Any]],
    device: torch.device,
) -> dict[str, Any]:
    active_samples = samples_for(groups, "active")
    state, evaluate = evaluate_bundle(model, layers, active_samples, events, device)
    singles = [{"slot": slot, **evaluate({slot: "neutral"})} for slot in range(TOP_EVENTS)]
    donors = [{"slot": slot, **evaluate({slot: "donor"})} for slot in range(TOP_EVENTS)]
    pairs = [
        {"slots": [left, right], **evaluate({left: "neutral", right: "neutral"})}
        for left, right in itertools.combinations(range(TOP_EVENTS), 2)
    ]
    full_donor = evaluate({slot: "donor" for slot in range(TOP_EVENTS)})
    contrast = evaluate({0: "neutral", 1: "neutral"})

    energies = []
    captured = state["captured"]
    for slot in range(TOP_EVENTS):
        delta = captured["donor_states"][:, slot] - captured["receiver_states"][:, slot]
        energies.append(float(np.mean(np.sum(delta * delta, axis=1))))
    total_energy = max(sum(energies), EPSILON)

    null_samples = samples_for(groups, "matched_null")
    _null_state, null_evaluate = evaluate_bundle(model, layers, null_samples, events, device)
    null_responses = [null_evaluate({slot: "neutral"}) for slot in range(TOP_EVENTS)] + [
        null_evaluate({left: "neutral", right: "neutral"})
        for left, right in itertools.combinations(range(TOP_EVENTS), 2)
    ]
    surface_samples = samples_for(groups, "surface_only")
    _surface_state, surface_evaluate = evaluate_bundle(model, layers, surface_samples, events, device)
    surface_responses = [surface_evaluate({slot: "donor"}) for slot in range(TOP_EVENTS)] + [
        surface_evaluate({left: "donor", right: "donor"})
        for left, right in itertools.combinations(range(TOP_EVENTS), 2)
    ]

    return {
        "system_id": f"p1210:{split}:{attribute}",
        "model_id": "qwen3-4b-fp16",
        "split": split,
        "factor": attribute,
        "task_width": 3,
        "gauge": "naturalized_role_depth_response_quotient",
        "baseline_accuracy": state["baseline_accuracy"],
        "baseline_margin": state["baseline_margin"],
        "full_hidden_donor": full_donor,
        "phase1207_contrast": contrast,
        "single_ablation": singles,
        "single_donor": donors,
        "pair_ablation": pairs,
        "pair_donor": [],
        "contrast_single_rescue": [{"slot": slot, "recovery_fraction": 0.0} for slot in range(TOP_EVENTS)],
        "probe_energy_fraction": [value / total_energy for value in energies],
        "matched_null_max_drift": max(abs(row["behavior_damage"]) for row in null_responses),
        "carrier_control_max_drift": max(abs(row["behavior_damage"]) for row in surface_responses),
        "event_definitions": events,
        "sample_count": len(active_samples),
        "rescue_status": "untested_pending_high_order_necessity_gate",
    }


def holdout_row(
    model: Any,
    layers: list[Any],
    split: str,
    attribute: str,
    groups: list[dict[str, Any]],
    events: list[dict[str, Any]],
    device: torch.device,
) -> dict[str, Any]:
    samples = samples_for(groups, "active")
    _state, evaluate = evaluate_bundle(model, layers, samples, events, device)
    triple = [
        evaluate({slot: "neutral" for slot in slots})["behavior_damage"]
        for slots in itertools.combinations(range(TOP_EVENTS), 3)
    ]
    return {
        "system_id": f"p1210:{split}:{attribute}",
        "responses": {
            "max_triple_ablation_damage": float(max(triple)),
            "all_hidden_ablation_damage": evaluate({slot: "neutral" for slot in range(TOP_EVENTS)})["behavior_damage"],
            "all_hidden_donor_choice": evaluate({slot: "donor" for slot in range(TOP_EVENTS)})["donor_choice"],
        },
    }


def phase1208_prototypes() -> dict[str, dict[str, float]]:
    fit = read_json(SOURCE1208 / "analysis/fit.json")
    camera.validate_digest(fit, "fit_digest")
    return fit["holdout_prototypes"]


def camera_prediction(row: dict[str, Any]) -> dict[str, Any]:
    decision = camera.classify_camera(row)
    label = decision["predicted_quotient_label"]
    prototype = phase1208_prototypes()[label]
    return {
        "system_id": row["system_id"],
        "factor": row["factor"],
        "camera_decision": label,
        "abstain": label == "unidentifiable_equivalence",
        "predicted_holdout_responses": {key: float(prototype[key]) for key in PREDICTION_KEYS},
        "predicted_structure": {
            key: decision[key]
            for key in ("global_minimal_cut_sets", "context_minimal_cut_sets", "sufficient_single_slots", "rescue_slots")
        },
    }


def precision_gate(model: Any) -> dict[str, Any]:
    precision = quantization_audit(model)
    devices = sorted({str(parameter.device) for parameter in model.parameters()})
    value = {
        "precision": precision,
        "devices": devices,
        "full_cuda": devices == ["cuda:0"],
        "gate": bool(
            precision["has_fp16_parameters"] and not precision["has_bf16_parameters"]
            and not precision["has_quantized_modules"] and set(precision["parameter_dtypes"]) == {"float16"}
            and devices == ["cuda:0"]
        ),
    }
    if not value["gate"]:
        raise RuntimeError(f"precision gate failed: {value}")
    return value


def split_path(split: str, name: str) -> Path:
    return OUT_ROOT / "runs" / split / name


def prepare_split(split: str, device_name: str) -> dict[str, Any]:
    protocol = verify_protocol()
    require_preaudit()
    if split not in ("discovery", "confirmation"):
        raise ValueError(split)
    if split_path(split, "low_order_camera_inputs.jsonl.gz").exists():
        raise RuntimeError(f"{split} preparation already exists")
    if split == "confirmation":
        discovery_score = read_json(OUT_ROOT / "analysis/discovery_score.json")
        validate_digest(discovery_score, "score_digest")
        if not discovery_score["confirmation_authorized"]:
            raise RuntimeError("discovery denied confirmation")
    if not torch.cuda.is_available() or device_name != "cuda":
        raise RuntimeError("full CUDA is required")
    groups = split_groups(split)
    model, _tokenizer, device, _load = load_fp16(MODEL)
    layers = get_layers(model)
    if len(layers) != LAYER_COUNT:
        raise RuntimeError("layer count drift")
    precision = precision_gate(model)
    try:
        raw_behavior, behavior = behavior_rows(groups, model, device)
        write_jsonl_gz(split_path(split, "behavior_rows.jsonl.gz"), raw_behavior)
        behavior.update({"phase": PHASE, "split": split, "protocol_digest": protocol["protocol_digest"], "precision": precision})
        behavior["summary_digest"] = digest({key: value for key, value in behavior.items() if key != "summary_digest"})
        write_json(split_path(split, "behavior_summary.json"), behavior)
        if not behavior["gate"]:
            print(json.dumps({"split": split, "behavior_gate": False, "panel_accuracy": behavior["panel_accuracy"]}, indent=2))
            return behavior
        if split == "discovery":
            sites = scout_sites(model, layers, groups, device)
            write_json(FROZEN_SITES_PATH, sites)
            if not sites["measurement_authorized"]:
                print(json.dumps({"split": split, "site_gate": False, "checks": sites["checks"]}, indent=2))
                return sites
        else:
            sites = read_json(FROZEN_SITES_PATH)
            validate_digest(sites, "site_digest")
            if not sites["measurement_authorized"]:
                raise RuntimeError("frozen site gate failed")
        low_rows: list[dict[str, Any]] = []
        for attribute in ATTRIBUTES:
            factor_groups = [group for group in groups if group["attribute"] == attribute]
            events = [
                {key: value for key, value in row.items() if key in {"event_id", "depth", "role", "component"}}
                for row in sites["selected"][attribute]
            ]
            low_rows.append(low_order_row(model, layers, split, attribute, factor_groups, events, device))
            print(canonical({"split": split, "low_order": attribute, "completed": len(low_rows)}), flush=True)
        write_jsonl_gz(split_path(split, "low_order_camera_inputs.jsonl.gz"), low_rows)
        predictions = [camera_prediction(row) for row in low_rows]
        prediction_path = OUT_ROOT / "analysis" / f"{split}_predictions.jsonl.gz"
        write_jsonl_gz(prediction_path, predictions)
        holdout_path = split_path(split, "holdout_responses.jsonl.gz")
        manifest = {
            "phase": PHASE,
            "split": split,
            "protocol_digest": protocol["protocol_digest"],
            "site_digest": sites["site_digest"],
            "prediction_count": len(predictions),
            "prediction_digest": digest(predictions),
            "holdout_absent_at_prediction": not holdout_path.exists(),
            "created_at_utc": utc_now(),
        }
        manifest["manifest_digest"] = digest(manifest)
        write_json(OUT_ROOT / "analysis" / f"{split}_prediction_manifest.json", manifest)
        summary = {
            "phase": PHASE,
            "split": split,
            "unit_count": len(low_rows),
            "finite_fraction": float(np.mean([np.isfinite(list(camera.flatten_numeric(row))).mean() for row in low_rows])),
            "camera_distribution": dict(sorted(Counter(row["camera_decision"] for row in predictions).items())),
            "row_digest": digest(low_rows),
            "manifest_digest": manifest["manifest_digest"],
        }
        summary["summary_digest"] = digest(summary)
        write_json(split_path(split, "low_order_summary.json"), summary)
        return summary
    finally:
        release_fp16(model)
        gc.collect()


def measure_holdout(split: str, device_name: str) -> dict[str, Any]:
    verify_protocol()
    require_preaudit()
    manifest = read_json(OUT_ROOT / "analysis" / f"{split}_prediction_manifest.json")
    validate_digest(manifest, "manifest_digest")
    if not manifest["holdout_absent_at_prediction"]:
        raise RuntimeError("holdout existed before prediction")
    output_path = split_path(split, "holdout_responses.jsonl.gz")
    if output_path.exists():
        raise RuntimeError("holdout already exists")
    if not torch.cuda.is_available() or device_name != "cuda":
        raise RuntimeError("full CUDA is required")
    groups = split_groups(split)
    sites = read_json(FROZEN_SITES_PATH)
    validate_digest(sites, "site_digest")
    model, _tokenizer, device, _load = load_fp16(MODEL)
    layers = get_layers(model)
    precision_gate(model)
    try:
        rows: list[dict[str, Any]] = []
        for attribute in ATTRIBUTES:
            factor_groups = [group for group in groups if group["attribute"] == attribute]
            events = [
                {key: value for key, value in row.items() if key in {"event_id", "depth", "role", "component"}}
                for row in sites["selected"][attribute]
            ]
            rows.append(holdout_row(model, layers, split, attribute, factor_groups, events, device))
            print(canonical({"split": split, "holdout": attribute, "completed": len(rows)}), flush=True)
        write_jsonl_gz(output_path, rows)
        summary = {"phase": PHASE, "split": split, "unit_count": len(rows), "row_digest": digest(rows)}
        summary["summary_digest"] = digest(summary)
        write_json(split_path(split, "holdout_summary.json"), summary)
        return summary
    finally:
        release_fp16(model)
        gc.collect()


def score_split(split: str) -> dict[str, Any]:
    protocol = verify_protocol()
    manifest = read_json(OUT_ROOT / "analysis" / f"{split}_prediction_manifest.json")
    validate_digest(manifest, "manifest_digest")
    predictions = read_jsonl_gz(OUT_ROOT / "analysis" / f"{split}_predictions.jsonl.gz")
    if digest(predictions) != manifest["prediction_digest"]:
        raise RuntimeError("prediction drift")
    low_rows = read_jsonl_gz(split_path(split, "low_order_camera_inputs.jsonl.gz"))
    holdout = read_jsonl_gz(split_path(split, "holdout_responses.jsonl.gz"))
    behavior = read_json(split_path(split, "behavior_summary.json"))
    sites = read_json(FROZEN_SITES_PATH)
    validate_digest(sites, "site_digest")
    by_id = {row["system_id"]: row["responses"] for row in holdout}
    errors = [
        abs(float(prediction["predicted_holdout_responses"][key]) - float(by_id[prediction["system_id"]][key]))
        for prediction in predictions for key in PREDICTION_KEYS
    ]
    nonabstain = [row for row in predictions if not row["abstain"]]
    metrics = {
        "unit_count": len(predictions),
        "nonabstain_count": len(nonabstain),
        "nonabstain_fraction": len(nonabstain) / max(len(predictions), 1),
        "holdout_mae": float(np.mean(errors)),
        "holdout_max_abs_error": float(max(errors)),
        "camera_decision_distribution": dict(sorted(Counter(row["camera_decision"] for row in predictions).items())),
        "matched_null_max_damage": float(max(row["matched_null_max_drift"] for row in low_rows)),
        "carrier_max_damage": float(max(row["carrier_control_max_drift"] for row in low_rows)),
    }
    checks = {
        "behavior": behavior["gate"] is True,
        "sites": sites["measurement_authorized"] is True,
        "nonabstain_breadth": metrics["nonabstain_count"] >= THRESHOLDS["nonabstain_attributes_min"],
        "matched_null": metrics["matched_null_max_damage"] <= THRESHOLDS["matched_null_max_damage_max"],
        "carrier": metrics["carrier_max_damage"] <= THRESHOLDS["carrier_max_damage_max"],
        "holdout_mae": metrics["holdout_mae"] <= THRESHOLDS["holdout_mae_max"],
        "holdout_max": metrics["holdout_max_abs_error"] <= THRESHOLDS["holdout_max_abs_error_max"],
    }
    value = {
        "phase": PHASE,
        "split": split,
        "protocol_digest": protocol["protocol_digest"],
        "prediction_manifest_digest": manifest["manifest_digest"],
        "metrics": metrics,
        "checks": checks,
        "gate": all(checks.values()),
    }
    if split == "discovery":
        value["confirmation_authorized"] = value["gate"]
    value["score_digest"] = digest(value)
    write_json(OUT_ROOT / "analysis" / f"{split}_score.json", value)
    print(json.dumps(value, ensure_ascii=False, indent=2))
    return value


def finalize() -> dict[str, Any]:
    protocol = verify_protocol()
    discovery_behavior_path = split_path("discovery", "behavior_summary.json")
    if not discovery_behavior_path.exists():
        raise RuntimeError("discovery was not executed")
    discovery_behavior = read_json(discovery_behavior_path)
    discovery_score_path = OUT_ROOT / "analysis/discovery_score.json"
    if not discovery_behavior["gate"]:
        status = "discovery_behavior_gate_failed"
        passed = False
        discovery = None
        confirmation = None
    elif not discovery_score_path.exists():
        sites = read_json(FROZEN_SITES_PATH) if FROZEN_SITES_PATH.exists() else {"measurement_authorized": False}
        status = "discovery_event_or_measurement_gate_failed"
        passed = False
        discovery = None
        confirmation = None
    else:
        discovery = read_json(discovery_score_path)
        validate_digest(discovery, "score_digest")
        if not discovery["confirmation_authorized"]:
            status = "discovery_high_order_prediction_gate_failed"
            passed = False
            confirmation = None
        else:
            confirmation_path = OUT_ROOT / "analysis/confirmation_score.json"
            if not confirmation_path.exists():
                raise RuntimeError("confirmation was authorized but not scored")
            confirmation = read_json(confirmation_path)
            validate_digest(confirmation, "score_digest")
            passed = confirmation["gate"] is True
            status = "qwen3_naturalized_camera_transfer_confirmed" if passed else "confirmation_transfer_gate_failed"
    statement = (
        "The frozen Phase1208 intervention-response camera predicted held-out high-order interventions across fresh "
        "Qwen3 naturalized object-attribute entities, values, templates, and combinations."
        if passed else
        "The frozen Phase1208 intervention-response camera did not pass its one-shot Qwen3 naturalized object-attribute "
        "behavior/event/high-order prediction chain; the failure is an applicability boundary, not absence of mechanism."
    )
    value = {
        "phase": PHASE,
        "created_at_utc": utc_now(),
        "status": status,
        "protocol_digest": protocol["protocol_digest"],
        "naturalized_qwen3_external_validity": passed,
        "discovery": None if discovery is None else discovery["metrics"],
        "confirmation": None if confirmation is None else confirmation["metrics"],
        "claim_boundary": (
            "This is Qwen3-only evidence on controlled naturalized prose. It does not establish an organic-corpus, "
            "cross-model, all-language, unique physical implementation, brain, or AGI mechanism."
        ),
        "new_k_item": {"id": "K190", "level": "E3-Q", "statement": statement},
        "rescue_status": "untested; requires a separately preregistered nontrivial downstream rescue after a pass",
        "auto_continue": False,
        "authorized_next": (
            "A separately preregistered downstream rescue or second natural pattern family only if Phase1210 passes; "
            "otherwise productize the applicability boundary."
        ),
    }
    value["final_digest"] = digest(value)
    write_json(FINAL_PATH, value)
    print(json.dumps(value, ensure_ascii=False, indent=2))
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("dev-material")
    subparsers.add_parser("preregister")
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--split", choices=("discovery", "confirmation"), required=True)
    prepare.add_argument("--device", default="cuda")
    holdout = subparsers.add_parser("measure-holdout")
    holdout.add_argument("--split", choices=("discovery", "confirmation"), required=True)
    holdout.add_argument("--device", default="cuda")
    score = subparsers.add_parser("score")
    score.add_argument("--split", choices=("discovery", "confirmation"), required=True)
    subparsers.add_parser("finalize")
    args = parser.parse_args()
    if args.command == "dev-material":
        dev_material()
    elif args.command == "preregister":
        preregister()
    elif args.command == "prepare":
        prepare_split(args.split, args.device)
    elif args.command == "measure-holdout":
        measure_holdout(args.split, args.device)
    elif args.command == "score":
        score_split(args.split)
    elif args.command == "finalize":
        finalize()


if __name__ == "__main__":
    main()
