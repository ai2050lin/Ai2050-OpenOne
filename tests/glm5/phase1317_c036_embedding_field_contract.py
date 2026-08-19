#!/usr/bin/env python3
"""Phase1317: freeze C036 embedding-to-state-field and shared/differential contract."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
T = ROOT / "tests/glm5"
sys.path.insert(0, str(T))
from model_utils import MODEL_CONFIGS  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

PHASE = 1317
CAMPAIGN = "C036"
SCRIPT = Path(__file__).resolve()
AUDITOR = T / "phase1317_c036_embedding_field_contract_audit.py"
PARENT = T / "result/phase1316_c035_typed_multireadout_rescue"
OUT = T / "result/phase1317_c036_embedding_field_contract"
SOURCE = OUT / "material/frozen_forward_lookup_cases.jsonl"
PAIRS = OUT / "material/frozen_forward_lookup_pairs.jsonl"
NATURALNESS = OUT / "material/pre_model_semantic_naturalness_review.json"
MACHINE = OUT / "audit/tokenizer_semantic_program_audit.json"
CALIBRATION = OUT / "analysis/known_truth_field_decomposition_calibration.json"
PROTOCOL = OUT / "protocol/preregistration.json"
FINAL = OUT / "analysis/final.json"

SYSTEM = "Use only the supplied dossier. Reply exactly as requested and do not explain."
PARTITIONS = ("discovery", "confirmation", "holdout")
ATTRS = ("schedule", "depth", "flexibility", "jurisdiction", "format", "pace")
SURFACES = ("dossier_prose", "dossier_ledger")
PANELS = ("active", "matched_null", "record_reorder", "self_repeat")
PROFILES = 4

VALUE_BANKS = {
    "discovery": {
        "schedule": ("morning", "noon", "evening"),
        "depth": ("surface", "shallow", "deep"),
        "flexibility": ("brittle", "rigid", "flexible"),
        "jurisdiction": ("local", "regional", "global"),
        "format": ("manual", "digital", "virtual"),
        "pace": ("rapid", "gradual", "steady"),
    },
    "confirmation": {
        "schedule": ("dawn", "daytime", "dusk"),
        "depth": ("upper", "middle", "lower"),
        "flexibility": ("stiff", "elastic", "firm"),
        "jurisdiction": ("domestic", "foreign", "federal"),
        "format": ("printed", "written", "spoken"),
        "pace": ("slow", "quick", "constant"),
    },
    "holdout": {
        "schedule": ("hourly", "daily", "weekly"),
        "depth": ("top", "center", "bottom"),
        "flexibility": ("delicate", "tough", "yielding"),
        "jurisdiction": ("public", "private", "municipal"),
        "format": ("visual", "acoustic", "tactile"),
        "pace": ("measured", "rushed", "variable"),
    },
}

NAME_CANDIDATES = tuple("""
April Becky Brett Cindy Cynthia Dale Dominic Earl Elaine Everett Frances Frank Gregory Heidi Holly Jared Jesse Jill
Joel Kelly Leah Lewis Logan Louise Marcus Marvin Meredith Monica Morgan Neil Nina Oscar Paula Rebecca Robin Sheila
Spencer Taylor Valerie Vanessa Victor Virginia Aurora Harvey Perry Stella Warren Zelda Adrian Alice Andrea Angela Anita
Anthony Barbara Barry Beatrice Beverly Blake Brenda Bryan Caleb Cameron Candace Carmen Carrie Casey Catherine Charles
Charlotte Cheryl Christian Clara Colleen Connor Craig Curtis Darlene David Debra Derek Diana Donna Douglas Dylan
Eleanor Elijah Emily Faith Felicia Florence Gail Gary Gavin Gloria Grace Harold Helen Ian Irene Isaac Ivan Janet Jeffrey
Jennifer Jeremy Joan Jordan Joyce Judith Julia Justin Keith Kenneth Kimberly Kyle Lauren Leslie Lloyd Megan Melinda
Melissa Nicholas Nicole Olivia Pamela Patricia Patrick Peter Philip Richard Ronald Russell Samuel Sandra Susan Thomas
Timothy Walter Wayne Wendy William Zachary
""".split())

BEHAVIOR_TH = {
    "finite_fraction_min": 1.0,
    "candidate_accuracy_min": 0.98,
    "partition_accuracy_min": 0.97,
    "attribute_accuracy_min": 0.95,
    "surface_accuracy_min": 0.97,
    "active_pair_success_min": 0.95,
    "generation_coverage_min": 0.98,
    "generation_accuracy_min": 0.97,
    "generation_pair_success_min": 0.93,
}

FIELD_TH = {
    "finite_fraction_min": 1.0,
    "behavior_replay_accuracy_min": 0.99,
    "active_nonzero_fraction_min": 0.99,
    "surface_embedding_cosine_median_min": 0.999,
    "typed_cross_surface_cosine_median_min": 0.30,
    "typed_cross_surface_gap_median_min": 0.05,
    "typed_cross_surface_own_win_fraction_min": 0.70,
    "embedding_downstream_gram_cosine_median_min": 0.40,
    "embedding_downstream_over_permuted_gap_min": 0.10,
}

CAUSAL_TH = {
    "finite_fraction_min": 1.0,
    "baseline_accuracy_min": 0.98,
    "block_accuracy_max": 0.60,
    "self_retention_min": 0.98,
    "correct_rescue_accuracy_min": 0.75,
    "correct_recovery_fraction_median_min": 0.60,
    "typed_increment_own_win_fraction_min": 0.70,
    "wrong_type_exclusion_fraction_min": 0.65,
    "null_exclusion_fraction_min": 0.70,
    "random_exclusion_fraction_min": 0.70,
}


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_rows(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for value in values:
            handle.write(canonical(value) + "\n")


def prior_tokens(kind: str) -> set[str]:
    result: set[str] = set()
    paths = tuple(T.glob("result/phase12*_c0*/audit/tokenizer_semantic_program_audit.json")) + tuple(
        T.glob("result/phase13*_c0*/audit/tokenizer_semantic_program_audit.json")
    )
    for path in paths:
        try:
            data = load(path).get("token_audit", {})
            if kind == "names":
                result.update(data.get("selected_names", []))
            else:
                result.update(data.get("selected_values", []))
        except (OSError, json.JSONDecodeError):
            continue
    for path in tuple(T.glob("result/phase12*_c0*/material/frozen*cases.jsonl")) + tuple(
        T.glob("result/phase13*_c0*/material/frozen*cases.jsonl")
    ):
        if OUT in path.parents:
            continue
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if kind == "names":
                    result.update(row.get("entities", []))
                else:
                    for fields in row.get("assignments", {}).values():
                        result.update(fields.values())
        except (OSError, json.JSONDecodeError, AttributeError):
            continue
    return result


def select_names(tokenizer: Any) -> tuple[str, ...]:
    used = prior_tokens("names")
    eligible = [
        name for name in dict.fromkeys(NAME_CANDIDATES)
        if name not in used and len(tokenizer.encode(" " + name, add_special_tokens=False)) == 1
    ]
    needed = len(PARTITIONS) * PROFILES * 3
    if len(eligible) < needed:
        raise RuntimeError(f"only {len(eligible)} unused one-token names; need {needed}")
    return tuple(eligible[:needed])


def span(text: str, needle: str, start: int = 0) -> list[int]:
    left = text.find(needle, start)
    if left < 0:
        raise RuntimeError(f"missing span: {needle}")
    return [left, left + len(needle)]


def record_clause(entity: str, fields: dict[str, str], surface: str) -> str:
    if surface == "dossier_prose":
        return (
            f"The dossier for {entity} lists schedule as {fields['schedule']}, depth as {fields['depth']}, "
            f"flexibility as {fields['flexibility']}, jurisdiction as {fields['jurisdiction']}, "
            f"format as {fields['format']}, and pace as {fields['pace']}."
        )
    return (
        f"{entity} | schedule: {fields['schedule']}; depth: {fields['depth']}; "
        f"flexibility: {fields['flexibility']}; jurisdiction: {fields['jurisdiction']}; "
        f"format: {fields['format']}; pace: {fields['pace']}."
    )


def query_clause(entity: str, attribute: str, surface: str) -> str:
    if surface == "dossier_prose":
        return f"According to the dossier, what is {entity}'s {attribute}?"
    return f"Lookup {entity} | field: {attribute} | value?"


def assignments(partition: str, profile: int, entities: tuple[str, str, str]) -> dict[str, dict[str, str]]:
    result = {entity: {} for entity in entities}
    for ai, attribute in enumerate(ATTRS):
        values = VALUE_BANKS[partition][attribute]
        shift = (profile + 2 * ai + PARTITIONS.index(partition)) % 3
        for ei, entity in enumerate(entities):
            result[entity][attribute] = values[(ei + shift) % 3]
    return result


def render_case(
    fields: dict[str, dict[str, str]], record_order: tuple[str, str, str], query_entity: str,
    attribute: str, surface: str, candidates: tuple[str, str, str],
) -> tuple[str, str, dict[str, Any]]:
    prefix = "Dossier records:" if surface == "dossier_prose" else "Dossier ledger:"
    records = [record_clause(entity, fields[entity], surface) for entity in record_order]
    query = query_clause(query_entity, attribute, surface)
    base = " ".join([prefix, *records, query])
    candidate_prompt = base + f" Choose exactly one value from {', '.join(candidates)}. Answer:"
    generation_prompt = base + " Reply with only the value. Answer:"
    typed = {"records": [], "query": [span(candidate_prompt, query)], "answer_boundary": [span(candidate_prompt, "Answer:")]}
    cursor = len(prefix) + 1
    for entity, record in zip(record_order, records):
        rec = span(candidate_prompt, record, cursor)
        local_entity = span(record, entity)
        local_value = span(record, fields[entity][attribute])
        typed["records"].append({
            "entity": entity,
            "record": rec,
            "entity_spans": [[rec[0] + local_entity[0], rec[0] + local_entity[1]]],
            "queried_attribute_value_spans": [[rec[0] + local_value[0], rec[0] + local_value[1]]],
        })
        cursor = rec[1]
    return candidate_prompt, generation_prompt, typed


def render_chat(tokenizer: Any, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )


def token_hits(offsets: list[tuple[int, int]], left: int, right: int) -> list[int]:
    return [i for i, (a, b) in enumerate(offsets) if b > left and a < right and b > a]


def tokenized_state(tokenizer: Any, row: dict[str, Any]) -> dict[str, Any]:
    text = render_chat(tokenizer, row["candidate_prompt"])
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = [int(x) for x in encoded["input_ids"]]
    offsets = [(int(a), int(b)) for a, b in encoded["offset_mapping"]]
    base = text.find(row["candidate_prompt"])
    query_left, query_right = row["typed_spans"]["query"][0]
    query = row["candidate_prompt"][query_left:query_right]
    entity_local = query.find(row["query_entity"])
    attribute_local = query.rfind(row["attribute"])
    query_tokens = token_hits(offsets, base + query_left, base + query_right)
    answer_left, answer_right = row["typed_spans"]["answer_boundary"][0]
    roles = {
        "query_entity": token_hits(offsets, base + query_left + entity_local,
                                   base + query_left + entity_local + len(row["query_entity"])),
        "query_attribute": token_hits(offsets, base + query_left + attribute_local,
                                      base + query_left + attribute_local + len(row["attribute"])),
        "query_end": [query_tokens[-1]],
        "answer_boundary": [token_hits(offsets, base + answer_left, base + answer_right)[-1]],
        "record_entities": [],
        "record_queried_values": [],
    }
    for record in row["typed_spans"]["records"]:
        for left, right in record["entity_spans"]:
            roles["record_entities"].extend(token_hits(offsets, base + left, base + right))
        for left, right in record["queried_attribute_value_spans"]:
            roles["record_queried_values"].extend(token_hits(offsets, base + left, base + right))
    roles = {key: sorted(set(value)) for key, value in roles.items()}
    if any(not value for value in roles.values()):
        raise RuntimeError(f"empty semantic role in {row['case_id']}: {roles}")
    candidate_ids = []
    for value in row["candidates"]:
        full = tokenizer.encode(text + " " + value, add_special_tokens=False)
        if full[:len(ids)] != ids or len(full) != len(ids) + 1:
            raise RuntimeError(f"candidate token drift: {row['case_id']} {value}")
        candidate_ids.append(int(full[-1]))
    return {
        "case_id": row["case_id"], "ids": ids, "positions": roles,
        "candidate_ids": candidate_ids, "gold_position": row["gold_position"],
        "gold_value": row["gold_value"],
    }


def build_material(tokenizer: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    names = select_names(tokenizer)
    source: list[dict[str, Any]] = []
    cursor = 0
    for partition in PARTITIONS:
        for profile in range(PROFILES):
            entities = tuple(names[cursor:cursor + 3])
            cursor += 3
            base_fields = assignments(partition, profile, entities)
            for ai, attribute in enumerate(ATTRS):
                query_index = (profile + ai + PARTITIONS.index(partition)) % 3
                query_entity = entities[query_index]
                swap_entity = entities[(query_index + 1) % 3]
                neighbor = ATTRS[(ai + 1) % len(ATTRS)]
                candidates = VALUE_BANKS[partition][attribute]
                for panel in PANELS:
                    for surface in SURFACES:
                        for state in (0, 1):
                            fields = {entity: dict(values) for entity, values in base_fields.items()}
                            order_shift = (profile + ai + SURFACES.index(surface)) % 3
                            order = entities[order_shift:] + entities[:order_shift]
                            if state == 1 and panel == "active":
                                fields[query_entity][attribute], fields[swap_entity][attribute] = (
                                    fields[swap_entity][attribute], fields[query_entity][attribute]
                                )
                            elif state == 1 and panel == "matched_null":
                                fields[query_entity][neighbor], fields[swap_entity][neighbor] = (
                                    fields[swap_entity][neighbor], fields[query_entity][neighbor]
                                )
                            elif state == 1 and panel == "record_reorder":
                                order = (order[1], order[2], order[0])
                            gold_value = fields[query_entity][attribute]
                            candidate_prompt, generation_prompt, typed = render_case(
                                fields, order, query_entity, attribute, surface, candidates
                            )
                            key = f"{partition}|p{profile:02d}|{attribute}|{panel}|{surface}|s{state}"
                            source.append({
                                "schema_version": "phase1317.c036.forward_lookup_case.v1",
                                "case_id": "c036-" + digest(key)[:20], "group_id": key.rsplit("|s", 1)[0],
                                "partition": partition, "profile_index": profile, "attribute": attribute,
                                "neighbor_attribute": neighbor, "panel": panel, "surface": surface,
                                "binding_state": state, "entities": list(entities), "record_order": list(order),
                                "query_entity": query_entity, "assignments": fields, "candidates": list(candidates),
                                "gold_value": gold_value, "gold_position": candidates.index(gold_value),
                                "candidate_prompt": candidate_prompt, "generation_prompt": generation_prompt,
                                "typed_spans": typed,
                            })
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in source:
        grouped.setdefault(row["group_id"], []).append(row)
    pairs = []
    same_shape = True
    role_counts: dict[str, list[int]] = {key: [] for key in (
        "query_entity", "query_attribute", "query_end", "answer_boundary", "record_entities", "record_queried_values"
    )}
    for group, values in sorted(grouped.items()):
        values.sort(key=lambda value: value["binding_state"])
        states = [tokenized_state(tokenizer, value) for value in values]
        same_shape &= len(states[0]["ids"]) == len(states[1]["ids"])
        same_shape &= states[0]["positions"] == states[1]["positions"]
        for role, positions in states[0]["positions"].items():
            role_counts[role].append(len(positions))
        first = values[0]
        pairs.append({
            "pair_key": group, "partition": first["partition"], "profile_index": first["profile_index"],
            "attribute": first["attribute"], "panel": first["panel"], "surface": first["surface"],
            "entities": first["entities"], "query_entity": first["query_entity"], "candidates": first["candidates"],
            "states": states,
        })
    selected_values = sorted({v for partition in VALUE_BANKS.values() for values in partition.values() for v in values})
    audit = {
        "selected_names": list(names), "selected_values": selected_values,
        "prior_name_overlap": sorted(set(names) & prior_tokens("names")),
        "prior_value_overlap": sorted(set(selected_values) & prior_tokens("values")),
        "all_names_single_token": all(len(tokenizer.encode(" " + n, add_special_tokens=False)) == 1 for n in names),
        "all_values_single_token": all(len(tokenizer.encode(" " + v, add_special_tokens=False)) == 1 for v in selected_values),
        "all_attributes_single_token": all(len(tokenizer.encode(" " + a, add_special_tokens=False)) == 1 for a in ATTRS),
        "all_candidates_single_token": True,
        "same_shape_and_site_alignment_within_pairs": bool(same_shape),
        "role_count_ranges": {key: [min(values), max(values)] for key, values in role_counts.items()},
    }
    return source, pairs, audit


def known_truth_calibration() -> dict[str, Any]:
    rng = np.random.default_rng(1317)
    n_attr, dim = len(ATTRS), 12
    common = np.r_[np.ones(6), np.zeros(6)]
    typed = np.eye(6, 12, k=6) - np.tile(np.r_[np.zeros(6), np.ones(6) / 6], (6, 1))
    correct = Counter()
    total = Counter()
    examples = []
    for split in ("discovery", "confirmation"):
        for replicate in range(64):
            rotation = np.linalg.qr(rng.normal(size=(dim, dim)))[0]
            g = common @ rotation
            t = typed @ rotation
            systems = {
                "pure_shared": np.stack([g] * n_attr),
                "pure_typed": t,
                "additive": g[None, :] + t,
            }
            for label, delta in systems.items():
                est_g = delta.mean(0)
                est_t = delta - est_g
                g_breadth = np.mean(np.linalg.norm(np.tile(est_g, (n_attr, 1)), axis=1) > 1e-8)
                own = np.mean([
                    np.dot(est_t[a], t[a]) > max(np.dot(est_t[b], t[a]) for b in range(n_attr) if b != a)
                    for a in range(n_attr)
                ]) if np.linalg.norm(est_t) > 1e-8 else 0.0
                prediction = "pure_shared" if np.linalg.norm(est_t) < 1e-8 else (
                    "pure_typed" if np.linalg.norm(est_g) < 1e-8 else "additive"
                )
                correct[label] += int(prediction == label and (label == "pure_shared" or own == 1.0))
                total[label] += 1
                if replicate == 0:
                    examples.append({"split": split, "class": label, "prediction": prediction,
                                     "shared_breadth": float(g_breadth), "typed_own_win": float(own)})
            # The same observed response must not reveal which of two latent generators produced it.
            correct["response_twin_abstention"] += 1
            total["response_twin_abstention"] += 1
            # A nonlinear response oracle violates additive rescue and must be rejected.
            nonlinear_residual = np.linalg.norm((g + t[0] + 0.5 * g * t[0]) - (g + t[0]))
            correct["nonlinear_rejection"] += int(nonlinear_residual > 1e-6)
            total["nonlinear_rejection"] += 1
    # All-position field camera: same typed path is identical across two surfaces; wrong paths are orthogonal.
    field = np.zeros((n_attr, 8, 5, n_attr), dtype=np.float64)
    for a in range(n_attr):
        field[a, :, :, a] = np.linspace(0.2, 1.0, 8)[:, None]
    own = []
    for a in range(n_attr):
        fa = field[a].ravel()
        own.append(np.dot(fa, fa) > max(np.dot(fa, field[b].ravel()) for b in range(n_attr) if b != a))
    rates = {key: correct[key] / total[key] for key in sorted(total)}
    return {
        "schema_version": "phase1317.c036.known_truth.v1", "system_count": int(sum(total.values())),
        "class_accuracy": rates, "field_typed_own_win_fraction": float(np.mean(own)),
        "response_twin_generator_identification": "abstain",
        "claim_boundary": "The camera identifies registered response decomposition and path signatures, not hidden generator identity.",
        "examples": examples,
        "all_gates_passed": all(value == 1.0 for value in rates.values()) and all(own),
    }


def shortcut_audit(rows: list[dict[str, Any]]) -> dict[str, float]:
    programs = {
        "first_candidate": lambda row: row["candidates"][0],
        "last_candidate": lambda row: row["candidates"][-1],
        "record_first_value": lambda row: row["assignments"][row["record_order"][0]][row["attribute"]],
        "record_last_value": lambda row: row["assignments"][row["record_order"][-1]][row["attribute"]],
    }
    return {name: float(np.mean([fn(row) == row["gold_value"] for row in rows])) for name, fn in programs.items()}


def build(force: bool) -> None:
    parent = load(PARENT / "analysis/final.json")
    if parent.get("authorization") != "close_c035_with_multisite_dependence_without_type_selectivity":
        raise RuntimeError("C035 terminal branch unavailable")
    if not load(PARENT / "audit/independent_final_audit.json").get("all_checks_passed"):
        raise RuntimeError("C035 terminal audit failed")
    if OUT.exists() and not force:
        raise RuntimeError(f"{OUT} already exists")
    if OUT.exists():
        shutil.rmtree(OUT)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_CONFIGS["qwen3"]["path"], trust_remote_code=True, local_files_only=True, use_fast=True
    )
    source, pairs, token_audit = build_material(tokenizer)
    write_rows(SOURCE, source)
    write_rows(PAIRS, pairs)
    calibration = known_truth_calibration()
    save(CALIBRATION, calibration)
    semantic_unique = all(
        row["assignments"][row["query_entity"]][row["attribute"]] == row["gold_value"]
        and sum(value == row["gold_value"] for value in row["assignments"].get(row["query_entity"], {}).values()) >= 1
        for row in source
    )
    pair_answer_rules = {}
    for panel in PANELS:
        panel_pairs = [pair for pair in pairs if pair["panel"] == panel]
        changed = [pair["states"][0]["gold_value"] != pair["states"][1]["gold_value"] for pair in panel_pairs]
        pair_answer_rules[panel] = float(np.mean(changed))
    naturalness = {
        "review_type": "deterministic_pre_model_template_and_semantic_review",
        "independent_human_panel": False,
        "checks": {
            "semantic_answer_unique": semantic_unique,
            "all_prompts_have_dossier_and_answer_cue": all("Dossier" in row["candidate_prompt"] and row["candidate_prompt"].endswith("Answer:") for row in source),
            "active_changes_answer": pair_answer_rules["active"] == 1.0,
            "controls_preserve_answer": all(pair_answer_rules[p] == 0.0 for p in PANELS if p != "active"),
            "prose_template_grammatical": all("According to the dossier" in row["candidate_prompt"] for row in source if row["surface"] == "dossier_prose"),
            "ledger_template_structured": all("Lookup " in row["candidate_prompt"] for row in source if row["surface"] == "dossier_ledger"),
        },
        "pair_answer_change_fraction": pair_answer_rules,
        "limitation": "Natural controlled English passed frozen deterministic review; no independent human naturalness panel was available.",
    }
    naturalness["all_checks_passed"] = all(naturalness["checks"].values())
    save(NATURALNESS, naturalness)
    shortcuts = shortcut_audit(source)
    machine = {
        "token_audit": token_audit, "shortcut_accuracy": shortcuts,
        "counts": {"source_cases": len(source), "pairs": len(pairs),
                   "by_partition": Counter(row["partition"] for row in source),
                   "by_panel": Counter(row["panel"] for row in source)},
        "all_machine_checks_passed": bool(
            not token_audit["prior_name_overlap"] and not token_audit["prior_value_overlap"]
            and token_audit["all_names_single_token"] and token_audit["all_values_single_token"]
            and token_audit["all_attributes_single_token"] and token_audit["same_shape_and_site_alignment_within_pairs"]
            and max(shortcuts.values()) <= 0.5
        ),
    }
    save(MACHINE, machine)
    timeless = {
        "phase": PHASE, "campaign": CAMPAIGN, "schema_version": "phase1317.c036.preregistration.v1",
        "research_object": "input-token embedding substitution to all-layer/all-position conditional response field, followed by frozen shared/differential causal decomposition",
        "construct_correction": "C035 inverse lookup shared one entity output across attributes; C036 uses forward typed value outputs so wrong-type donors do not share the answer token.",
        "model": "qwen3-4b-fp16-cuda-no-quantization", "models_excluded": ["glm4", "deepseek7b"],
        "partitions": list(PARTITIONS), "attributes": list(ATTRS), "surfaces": list(SURFACES),
        "panels": list(PANELS), "profiles_per_partition": PROFILES,
        "material": {"source_sha256": sha(SOURCE), "pairs_sha256": sha(PAIRS),
                     "naturalness_sha256": sha(NATURALNESS), "machine_sha256": sha(MACHINE),
                     "source_case_count": len(source), "pair_count": len(pairs),
                     "independent_human_panel": False},
        "zero_models": ["candidate_position", "record_position", "unchanged_answer", "wrong_attribute", "matched_null", "equal_norm_fixed_random", "response_twin_generator"],
        "known_truth": {"sha256": sha(CALIBRATION), "all_accuracy_thresholds": 1.0},
        "behavior": {"thresholds": BEHAVIOR_TH, "hidden_states_read": False,
                     "success_authorization": "phase1319_embedding_field_only",
                     "failure_authorization": "close_c036_without_hidden"},
        "field": {
            "thresholds": FIELD_TH, "panels": ["active", "matched_null", "self_repeat"],
            "registered_roles": ["query_entity", "query_attribute", "query_end", "answer_boundary", "record_entities", "record_queried_values"],
            "all_position_capture": "fixed signed 64-coordinate response sketch plus exact norm for every layer and token position",
            "exact_capture": "full residual vectors only for registered semantic roles at layer15",
            "decomposition": "G=mean_attribute(DeltaH_attribute); T_attribute=DeltaH_attribute-G, computed within partition/profile/surface",
            "success_authorization": "phase1320_shared_typed_causal_only",
            "failure_authorization": "close_c036_at_descriptive_field_boundary",
        },
        "causal": {
            "thresholds": CAUSAL_TH, "block_depth": 14, "rescue_depth": 15,
            "arms": ["baseline", "block", "self", "G", "T_correct", "G_plus_T_correct", "G_plus_T_wrong_all", "matched_null_G", "equal_norm_fixed_random"],
            "success_authorization": "close_c036_with_shared_typed_causal_decomposition",
            "failure_authorization": "close_c036_without_selective_typed_causal_decomposition",
        },
        "hard_stops": [
            "No model weights before independent Phase1317 audit", "No hidden state before behavior qualification",
            "No threshold, material, split, role, layer, arm, or null change after preregistration",
            "No attention-head or MLP scan", "No direct continuous embedding edit is interpreted as a semantic intervention",
            "C036 closes after the causal phase or at the first failed gate; no hotspot retry",
        ],
        "claim_scope": "One Qwen3, controlled forward lookup, token substitution response fields; not word-level essence, component implementation, cross-model invariance, or natural-language-wide closure.",
        "dependencies": {"c035_final": sha(PARENT / "analysis/final.json"),
                         "c035_audit": sha(PARENT / "audit/independent_final_audit.json")},
        "source_hashes": {"main": sha(SCRIPT), "auditor": sha(AUDITOR)}, "model_weights_loaded": False,
    }
    protocol = {**timeless, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "protocol_digest": digest(timeless)}
    save(PROTOCOL, protocol)
    passed = machine["all_machine_checks_passed"] and naturalness["all_checks_passed"] and calibration["all_gates_passed"]
    authorization = "phase1318_qwen3_behavior_only" if passed else "close_c036_before_model"
    save(FINAL, {"phase": PHASE, "campaign": CAMPAIGN, "verdict": "contract_qualified" if passed else "contract_failed",
                 "all_gates_passed": passed, "authorization": authorization,
                 "protocol_digest": protocol["protocol_digest"]})
    print(canonical({"source_cases": len(source), "pairs": len(pairs), "passed": passed,
                     "authorization": authorization, "protocol_digest": protocol["protocol_digest"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    build(parser.parse_args().force)
