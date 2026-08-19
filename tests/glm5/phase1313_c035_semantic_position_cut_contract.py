#!/usr/bin/env python3
"""Phase1313: freeze C035 semantic-position cut camera, material, and gates."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
T = ROOT / "tests/glm5"
sys.path.insert(0, str(T))
import phase1292_c029_object_attribute_convergence_contract as scaffold  # noqa: E402
import phase1294_c030_grounded_lookup_contract as c030  # noqa: E402
import phase1297_c031_event_interval_contract as c031  # noqa: E402
import phase1304_c033_role_typed_causal_graph_contract as c033  # noqa: E402
from model_utils import MODEL_CONFIGS  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

PHASE = 1313
CAMPAIGN = "C035"
SCRIPT = Path(__file__).resolve()
AUDITOR = T / "phase1313_c035_semantic_position_cut_contract_audit.py"
PARENT = T / "result/phase1312_c034_upstream_selective_rescue"
OUT = T / "result/phase1313_c035_semantic_position_cut_contract"
PROTOCOL = OUT / "protocol/preregistration.json"
SOURCE = OUT / "material/frozen_new_world_cases.jsonl"
MATERIAL = OUT / "material/frozen_position_cut_pairs.jsonl"
NATURALNESS = OUT / "material/pre_model_semantic_naturalness_review.json"
MACHINE = OUT / "audit/tokenizer_semantic_program_audit.json"
CALIBRATION = OUT / "analysis/known_truth_position_cut_calibration.json"
AUDIT = OUT / "audit/independent_final_audit.json"
FINAL = OUT / "analysis/final.json"

SYSTEM = "Use only the supplied registry. Reply exactly as requested and do not explain."
PARTITIONS = ("discovery", "confirmation", "holdout")
ATTRS = ("temperature", "texture", "origin", "condition", "category", "priority")
SURFACES = ("registry_prose", "registry_ledger")
PANELS = ("active", "matched_null", "surface_only", "semantic_neighbor")
PROFILES = 6

VALUE_BANKS = {
    "discovery": {
        "temperature": ("warm", "cool", "mild"),
        "texture": ("smooth", "rough", "silky"),
        "origin": ("coastal", "desert", "inland"),
        "condition": ("stable", "fragile", "damaged"),
        "category": ("civic", "medical", "musical"),
        "priority": ("urgent", "routine", "elective"),
    },
    "confirmation": {
        "temperature": ("hot", "chilly", "frozen"),
        "texture": ("glossy", "matte", "coarse"),
        "origin": ("urban", "rural", "island"),
        "condition": ("pristine", "worn", "repaired"),
        "category": ("legal", "technical", "social"),
        "priority": ("critical", "normal", "optional"),
    },
    "holdout": {
        "temperature": ("cold", "brisk", "humid"),
        "texture": ("polished", "woven", "soft"),
        "origin": ("northern", "southern", "central"),
        "condition": ("sealed", "exposed", "restored"),
        "category": ("floral", "mineral", "textile"),
        "priority": ("immediate", "baseline", "secondary"),
    },
}

BEHAVIOR_TH = {
    "finite_fraction_min": 1.0,
    "candidate_accuracy_min": 0.98,
    "partition_accuracy_min": 0.97,
    "attribute_accuracy_min": 0.95,
    "surface_accuracy_min": 0.97,
    "active_pair_success_min": 0.95,
    "attribute_family_success_min": 0.90,
    "generation_coverage_min": 0.98,
    "generation_label_accuracy_min": 0.97,
    "generation_pair_success_min": 0.93,
}

CUT_TH = {
    "finite_fraction_min": 1.0,
    "baseline_accuracy_min": 0.98,
    "self_retention_min": 0.98,
    "full_cut_accuracy_max": 0.60,
    "full_cut_margin_drop_median_min": 0.50,
    "full_over_qend_drop_ratio_min": 1.25,
}

RESCUE_TH = {
    "finite_fraction_min": 1.0,
    "correct_rescue_accuracy_min": 0.75,
    "correct_recovery_fraction_median_min": 0.60,
    "own_attribute_win_fraction_min": 0.70,
    "wrong_attribute_exclusion_fraction_min": 0.65,
    "null_exclusion_fraction_min": 0.70,
    "self_retention_min": 0.98,
}


def canonical(v: Any) -> str:
    return json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(v: Any) -> str:
    return hashlib.sha256(canonical(v).encode()).hexdigest()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_rows(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for value in values:
            f.write(canonical(value) + "\n")


def prior_names() -> set[str]:
    result: set[str] = set()
    for path in (
        T / "result/phase1292_c029_object_attribute_convergence_contract/audit/tokenizer_semantic_program_audit.json",
        T / "result/phase1294_c030_grounded_lookup_contract/audit/tokenizer_semantic_program_audit.json",
        T / "result/phase1297_c031_event_interval_contract/audit/tokenizer_semantic_program_audit.json",
        T / "result/phase1304_c033_role_typed_causal_graph_contract/audit/tokenizer_semantic_program_audit.json",
    ):
        result.update(load(path)["token_audit"]["selected_names"])
    return result


def prior_values() -> set[str]:
    result: set[str] = set()
    for banks in (scaffold.VALUE_BANKS, c030.VALUE_BANKS, c031.VALUE_BANKS, c033.VALUE_BANKS):
        for partition in banks.values():
            for values in partition.values():
                result.update(values)
    return result


def record_clause(entity: str, fields: dict[str, str], surface: str) -> str:
    if surface == "registry_prose":
        return (
            f"The registry entry for {entity} lists temperature as {fields['temperature']}, "
            f"texture as {fields['texture']}, origin as {fields['origin']}, condition as {fields['condition']}, "
            f"category as {fields['category']}, and priority as {fields['priority']}."
        )
    return (
        f"{entity} | temperature: {fields['temperature']}; texture: {fields['texture']}; "
        f"origin: {fields['origin']}; condition: {fields['condition']}; "
        f"category: {fields['category']}; priority: {fields['priority']}."
    )


def query_clause(attribute: str, value: str, surface: str) -> str:
    if surface == "registry_ledger":
        return f"Which registry entry has {attribute}: {value}?"
    return f"According to the registry, which entry has {attribute} listed as {value}?"


def select_names(tokenizer: Any) -> tuple[str, ...]:
    used = prior_names()
    eligible = [
        name for name in dict.fromkeys(c031.NAME_CANDIDATES)
        if name not in used and len(tokenizer.encode(" " + name, add_special_tokens=False)) == 1
    ]
    needed = len(PARTITIONS) * PROFILES * 3
    if len(eligible) < needed:
        raise RuntimeError(f"only {len(eligible)} unused one-token names; need {needed}")
    return tuple(eligible[:needed])


def render(tokenizer: Any, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )


def token_hits(offsets: list[tuple[int, int]], left: int, right: int) -> list[int]:
    return [i for i, (a, b) in enumerate(offsets) if b > left and a < right and b > a]


def tokenized_state(tokenizer: Any, row: dict[str, Any]) -> dict[str, Any]:
    text = render(tokenizer, row["candidate_prompt"])
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = [int(x) for x in encoded["input_ids"]]
    offsets = [(int(a), int(b)) for a, b in encoded["offset_mapping"]]
    base = text.find(row["candidate_prompt"])
    if base < 0:
        raise RuntimeError("candidate prompt missing from rendered chat")
    query_left, query_right = row["typed_spans"]["query"][0]
    query_text = row["candidate_prompt"][query_left:query_right]
    attr_local = query_text.find(row["attribute"])
    value_local = query_text.rfind(row["target_value"])
    if min(attr_local, value_local) < 0:
        raise RuntimeError("query role text missing")
    query_tokens = token_hits(offsets, base + query_left, base + query_right)
    answer_span = row["typed_spans"]["answer_boundary"][0]
    role_positions = {
        "query_attribute": token_hits(offsets, base + query_left + attr_local,
                                      base + query_left + attr_local + len(row["attribute"])),
        "query_value": token_hits(offsets, base + query_left + value_local,
                                  base + query_left + value_local + len(row["target_value"])),
        "query_end": [query_tokens[-1]],
        "answer_boundary": [token_hits(offsets, base + answer_span[0], base + answer_span[1])[-1]],
        "record_entities": [],
        "record_queried_values": [],
    }
    for record in row["typed_spans"]["records"]:
        for left, right in record["entity_spans"]:
            role_positions["record_entities"].extend(token_hits(offsets, base + left, base + right))
        for left, right in record["queried_attribute_value_spans"]:
            role_positions["record_queried_values"].extend(token_hits(offsets, base + left, base + right))
    for key in role_positions:
        role_positions[key] = sorted(set(role_positions[key]))
        if not role_positions[key]:
            raise RuntimeError(f"empty role {key}")
    candidate_ids = []
    for name in row["candidates"]:
        full = tokenizer.encode(text + " " + name, add_special_tokens=False)
        if full[:len(ids)] != ids or len(full) != len(ids) + 1:
            raise RuntimeError("candidate token drift")
        candidate_ids.append(int(full[-1]))
    return {
        "case_id": row["case_id"], "ids": ids, "positions": role_positions,
        "candidate_ids": candidate_ids, "gold_position": int(row["gold_position"]),
    }


def build_material(tokenizer: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    names = select_names(tokenizer)
    original = (
        scaffold.ATTRIBUTES, scaffold.ATTRIBUTE_LEXEME, scaffold.PANELS, scaffold.SURFACES,
        scaffold.CANDIDATE_ORDERS, scaffold.PROFILES_PER_PARTITION, scaffold.VALUE_BANKS,
        scaffold.NAME_POOL, scaffold.record_clause, scaffold.query_clause,
    )
    try:
        scaffold.ATTRIBUTES = ATTRS
        scaffold.ATTRIBUTE_LEXEME = {a: a for a in ATTRS}
        scaffold.PANELS = PANELS
        scaffold.SURFACES = SURFACES
        scaffold.CANDIDATE_ORDERS = (0,)
        scaffold.PROFILES_PER_PARTITION = PROFILES
        scaffold.VALUE_BANKS = VALUE_BANKS
        scaffold.NAME_POOL = names
        scaffold.record_clause = record_clause
        scaffold.query_clause = query_clause
        source, token_audit = scaffold.build_cases(tokenizer)
    finally:
        (
            scaffold.ATTRIBUTES, scaffold.ATTRIBUTE_LEXEME, scaffold.PANELS, scaffold.SURFACES,
            scaffold.CANDIDATE_ORDERS, scaffold.PROFILES_PER_PARTITION, scaffold.VALUE_BANKS,
            scaffold.NAME_POOL, scaffold.record_clause, scaffold.query_clause,
        ) = original
    for row in source:
        row["schema_version"] = "phase1313.c035.source.v1"
        row["case_id"] = "c035-" + digest({"group": row["group_id"], "state": row["binding_state"],
                                            "prompt": row["candidate_prompt"]})[:20]

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in source:
        if row["panel"] in {"active", "matched_null"}:
            grouped[row["group_id"]].append(row)
    pairs = []
    site_counts: dict[str, list[int]] = defaultdict(list)
    same_shape = True
    for group_id, pair in sorted(grouped.items()):
        pair = sorted(pair, key=lambda x: x["binding_state"])
        if len(pair) != 2:
            raise RuntimeError("incomplete pair")
        states = [tokenized_state(tokenizer, row) for row in pair]
        same_shape &= len(states[0]["ids"]) == len(states[1]["ids"])
        same_shape &= states[0]["positions"] == states[1]["positions"]
        first = pair[0]
        for key, value in states[0]["positions"].items():
            site_counts[key].append(len(value))
        pairs.append({
            "pair_key": f"{first['partition']}|p{first['profile_index']:02d}|{first['attribute']}|{first['surface']}|{first['panel']}",
            "group_id": group_id, "partition": first["partition"], "profile_index": first["profile_index"],
            "attribute": first["attribute"], "surface": first["surface"], "panel": first["panel"],
            "candidates": first["candidates"],
            "identity_positions": [int(pair[0]["gold_position"]), int(pair[1]["gold_position"])],
            "states": states,
        })
    used = prior_names()
    flat_values = {v for partition in VALUE_BANKS.values() for values in partition.values() for v in values}
    token_audit.update({
        "selected_names": list(names), "prior_name_overlap": sorted(used & set(names)),
        "prior_value_overlap": sorted(prior_values() & flat_values),
        "all_values_single_token": all(len(tokenizer.encode(" " + v, add_special_tokens=False)) == 1 for v in flat_values),
        "same_shape_and_site_alignment_within_pairs": bool(same_shape),
        "site_count_ranges": {key: [min(values), max(values)] for key, values in site_counts.items()},
    })
    return source, pairs, token_audit


def minimal_sets(damaging: list[tuple[int, ...]]) -> list[tuple[int, ...]]:
    result = []
    for cut in sorted(damaging, key=lambda x: (len(x), x)):
        if not any(set(smaller).issubset(cut) for smaller in result):
            result.append(cut)
    return result


def known_truth_position_cut_calibration() -> dict[str, Any]:
    sites = tuple(range(6))
    cuts = [cut for size in range(1, 7) for cut in itertools.combinations(sites, size)]
    examples = []
    class_hits = Counter()
    class_totals = Counter()
    twin_abstain = []
    for split in ("discovery", "confirmation"):
        for morphology in ("single_required", "serial_required_pair", "redundant_pair", "readable_bypass", "response_twin"):
            for replicate in range(32):
                order = sorted(sites, key=lambda site: digest([split, morphology, replicate, site]))
                a, b = order[:2]
                if morphology == "single_required":
                    damaging = [cut for cut in cuts if a in cut]
                    expected = "single_required"
                elif morphology == "serial_required_pair":
                    damaging = [cut for cut in cuts if a in cut or b in cut]
                    expected = "serial_required_pair"
                elif morphology == "redundant_pair":
                    damaging = [cut for cut in cuts if a in cut and b in cut]
                    expected = "redundant_pair"
                else:
                    damaging = []
                    expected = "readable_nonessential_or_unregistered_bypass"
                mins = minimal_sets(damaging)
                if not mins:
                    prediction = "readable_nonessential_or_unregistered_bypass"
                elif len(mins) == 1 and len(mins[0]) == 1:
                    prediction = "single_required"
                elif all(len(x) == 1 for x in mins) and len(mins) > 1:
                    prediction = "serial_required_pair"
                elif len(mins) == 1 and len(mins[0]) > 1:
                    prediction = "redundant_pair"
                else:
                    prediction = "abstain_unregistered_morphology"
                if morphology == "response_twin":
                    twin_abstain.append(prediction == "readable_nonessential_or_unregistered_bypass")
                class_hits[expected] += int(prediction == expected)
                class_totals[expected] += 1
                if replicate < 2:
                    examples.append({"split": split, "morphology": morphology, "opaque_sites": [a, b],
                                     "minimal_damaging_cuts": [list(x) for x in mins], "prediction": prediction})

    typed_examples = []
    typed_hits = []
    for target in range(len(ATTRS)):
        signatures = {
            "generic": [1] * len(ATTRS),
            "typed": [int(i == target) for i in range(len(ATTRS))],
            "mixed": [int(i in {target, (target + 1) % len(ATTRS)}) for i in range(len(ATTRS))],
            "null": [0] * len(ATTRS),
        }
        for label, signature in signatures.items():
            total = sum(signature)
            prediction = "generic" if total == len(ATTRS) else "typed" if total == 1 and signature[target] else \
                "mixed" if total == 2 and signature[target] else "null"
            typed_hits.append(prediction == label)
            typed_examples.append({"target": ATTRS[target], "label": label, "signature": signature,
                                   "prediction": prediction})
    class_accuracy = {key: class_hits[key] / class_totals[key] for key in sorted(class_totals)}
    return {
        "schema_version": "phase1313.c035.position_cut_camera.v1",
        "system_count": sum(class_totals.values()), "cut_family_size": len(cuts),
        "class_accuracy": class_accuracy,
        "response_twin_origin_abstention_fraction": sum(twin_abstain) / len(twin_abstain),
        "typed_multi_readout_accuracy": sum(typed_hits) / len(typed_hits),
        "single_target_generic_typed_collision_fraction": 1.0,
        "claim_boundary": "The camera classifies registered intervention-response morphology, not latent generator identity or Transformer architecture.",
        "examples": examples, "typed_examples": typed_examples,
        "all_gates_passed": all(value == 1.0 for value in class_accuracy.values())
                            and all(twin_abstain) and all(typed_hits),
    }


def build(force: bool) -> None:
    if load(PARENT / "analysis/final.json").get("authorization") != "close_c034_at_upstream_rescue_boundary":
        raise RuntimeError("C034 terminal branch unavailable")
    if not load(PARENT / "audit/independent_final_audit.json").get("all_checks_passed"):
        raise RuntimeError("C034 terminal audit failed")
    if OUT.exists() and not force:
        raise RuntimeError(f"{OUT} already exists")
    if OUT.exists():
        shutil.rmtree(OUT)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIGS["qwen3"]["path"], trust_remote_code=True,
                                              local_files_only=True, use_fast=True)
    source, material, token_audit = build_material(tokenizer)
    write_rows(SOURCE, source)
    write_rows(MATERIAL, material)
    calibration = known_truth_position_cut_calibration()
    save(CALIBRATION, calibration)
    semantic_unique = all(
        sum(fields[row["attribute"]] == row["target_value"] for fields in row["assignments"].values()) == 1
        for row in source
    )
    grammar_checks = {
        "all_prompts_end_with_answer_cue": all(row["candidate_prompt"].endswith("Answer:") for row in source),
        "all_generation_prompts_end_with_answer_cue": all(row["generation_prompt"].endswith("Answer:") for row in source),
        "all_queries_have_one_question_mark": all(row["candidate_prompt"].count("?") == 1 for row in source),
        "all_records_end_with_period": all(row["candidate_prompt"].split(" Which ")[0].count(".") >= 3 for row in source),
        "all_surfaces_registered": set(row["surface"] for row in source) == set(SURFACES),
    }
    naturalness = {
        "phase": PHASE, "campaign": CAMPAIGN, "reviewed_before_model_weight_load": True,
        "deterministic_grammar_checks": grammar_checks, "semantic_uniqueness": semantic_unique,
        "lexical_role_interpretation": {
            "temperature": "descriptive thermal label", "texture": "surface-feel label",
            "origin": "provenance-region label", "condition": "current-state label",
            "category": "classification label", "priority": "handling-priority label",
        },
        "limitation": "Controlled registry English passed deterministic semantic and grammar review; no independent human naturalness panel was available.",
        "all_checks_passed": semantic_unique and all(grammar_checks.values()),
    }
    save(NATURALNESS, naturalness)
    program = scaffold.program_audit(source)
    machine = {
        "phase": PHASE, "campaign": CAMPAIGN, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "token_audit": token_audit, "program_audit": program,
        "source_case_count": len(source), "pair_count": len(material),
        "partition_pair_counts": dict(Counter(x["partition"] for x in material)),
        "panel_pair_counts": dict(Counter(x["panel"] for x in material)),
        "all_machine_checks_passed": bool(
            len(source) == 1728 and len(material) == 432 and semantic_unique
            and not token_audit["prior_name_overlap"] and not token_audit["prior_value_overlap"]
            and token_audit["all_candidates_single_token"] and token_audit["all_values_single_token"]
            and token_audit["same_shape_and_site_alignment_within_pairs"]
            and program["shortcut_ceiling"] <= 0.70
        ),
    }
    save(MACHINE, machine)
    timeless = {
        "phase": PHASE, "campaign": CAMPAIGN, "schema_version": "phase1313.c035.position_cut_contract.v1",
        "purpose": "adjudicate whether the C034 readable query-end state is a nonessential copy, a single path, or part of a redundant semantic-position path on independent worlds",
        "evidence_review": {
            "accepted": [
                "C034 found repeatable attribute-conditioned response geometry at query_end depth14",
                "C034 single-position block preserved 100 percent behavior and typed rescue failed",
                "Phase1158 and Phase1253 already calibrate redundancy and component minimal-cut logic in explicit known-truth systems",
            ],
            "corrected_overclaims": [
                "query_end depth14 is a repeatable readout site, not an identified attribute-information location",
                "single-position block failure does not establish distributed or redundant computation",
                "the upstream/late type-gap ratio mixes token role with depth and is not a pure compression law",
                "a registered layer14 test on new worlds is prospective confirmation of a historical candidate, not post-hoc selection",
                "a damaging multi-position replacement would establish dependence on the registered set, not minimality or semantic purity",
            ],
        },
        "known_truth_camera": {
            "registered_classes": ["single_required", "serial_required_pair", "redundant_pair",
                                   "readable_nonessential_or_unregistered_bypass"],
            "origin_twins": "must abstain from generator identity when intervention-response signatures coincide",
            "typed_readout_classes": ["generic", "typed", "mixed", "null"],
            "all_accuracy_thresholds": 1.0, "calibration_sha256": sha(CALIBRATION),
            "dependencies": {
                "phase1158": sha(T / "result/phase1158_redundancy_gate_calibration/analysis/final.json"),
                "phase1253": sha(T / "result/phase1253_c006_planted_mechanism_cut_library/analysis/final.json"),
            },
        },
        "material": {
            "independent_from_c033_c034": True, "partitions": list(PARTITIONS), "profiles_per_partition": PROFILES,
            "attributes": list(ATTRS), "surfaces": list(SURFACES), "panels": list(PANELS),
            "entity_count": 54, "source_case_count": len(source), "pair_count": len(material),
            "source_sha256": sha(SOURCE), "pair_sha256": sha(MATERIAL), "naturalness_sha256": sha(NATURALNESS),
            "human_naturalness_panel": False,
        },
        "model": {"model_id": "qwen3-4b-fp16-cuda-no-quantization", "compiler": "right_padding",
                  "formal_runs_per_model_phase": 1, "other_models_authorized": False},
        "behavior": {"thresholds": BEHAVIOR_TH, "hidden_states_read": False},
        "position_cut": {
            "depth": 14, "depth_source": "frozen out-of-sample confirmation of the C034 descriptive candidate",
            "sets": {
                "query_end_only": ["query_end"],
                "query_bundle": ["query_attribute", "query_value", "query_end"],
                "record_bundle": ["record_entities", "record_queried_values"],
                "full_registered": ["query_attribute", "query_value", "query_end", "record_entities", "record_queried_values"],
            },
            "operation": "at the input to layer14, replace active state1 role positions with the aligned active state0 residuals",
            "arms": ["baseline", "query_end_only", "query_bundle", "record_bundle", "full_registered", "self_retention"],
            "partitions": ["confirmation", "holdout"], "thresholds": CUT_TH,
            "claim_scope": "dependence on a frozen semantic-position set; not a minimal cut, semantic-pure edit, or distributed-path proof",
        },
        "typed_rescue": {
            "authorized_only_if": "full_registered position cut gate passes",
            "rescue_depth": 15,
            "readout_family": "six attribute-conditioned entity margins in the same world",
            "controls": ["correct_attribute_cross_surface", "all_five_wrong_attributes", "matched_null", "self_retention"],
            "thresholds": RESCUE_TH,
        },
        "branches": {
            "phase1313_fail": "close_c035_before_model", "phase1313_pass": "phase1314_qwen3_behavior_only",
            "phase1314_fail": "close_c035_without_hidden", "phase1314_pass": "phase1315_multisite_cut_only",
            "phase1315_fail": "close_c035_at_registered_cut_boundary", "phase1315_pass": "phase1316_typed_rescue_only",
            "phase1316_any_verdict": "close_c035",
        },
        "hard_stops": [
            "No model weights before Phase1313 independent audit passes",
            "No hidden state or intervention before Phase1314 behavior gate passes",
            "No typed rescue before Phase1315 registered-cut gate passes",
            "No post-unblinding material, role, position set, depth, arm, threshold, partition, or parser change",
            "No head, MLP, neuron, subspace, layer, or window search in C035",
            "C035 closes after Phase1316 or at the first failed gate",
        ],
        "dependencies": {"c034_final": sha(PARENT / "analysis/final.json"),
                         "c034_audit": sha(PARENT / "audit/independent_final_audit.json"),
                         "source": sha(SOURCE), "material": sha(MATERIAL), "calibration": sha(CALIBRATION)},
        "source_hashes": {"main": sha(SCRIPT), "auditor": sha(AUDITOR)}, "model_weights_loaded": False,
    }
    protocol = {**timeless, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "protocol_digest": digest(timeless)}
    save(PROTOCOL, protocol)
    passed = bool(machine["all_machine_checks_passed"] and naturalness["all_checks_passed"]
                  and calibration["all_gates_passed"])
    authorization = "phase1314_qwen3_behavior_only" if passed else "close_c035_before_model"
    save(FINAL, {"phase": PHASE, "campaign": CAMPAIGN,
                 "verdict": "contract_camera_material_qualified" if passed else "contract_camera_or_material_failed",
                 "all_gates_passed": passed, "authorization": authorization,
                 "protocol_digest": protocol["protocol_digest"], "model_weights_loaded": False})
    print(canonical({"source_cases": len(source), "pairs": len(material), "camera": calibration["all_gates_passed"],
                     "authorization": authorization, "digest": protocol["protocol_digest"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build",))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    build(args.force)
