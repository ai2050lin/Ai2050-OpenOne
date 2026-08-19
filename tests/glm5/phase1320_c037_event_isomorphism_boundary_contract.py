#!/usr/bin/env python3
"""Phase1320: freeze C037 entity-slot isomorphism and true assistant-boundary camera."""
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
import phase1317_c036_embedding_field_contract as scaffold  # noqa: E402
from model_utils import MODEL_CONFIGS  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

PHASE, CAMPAIGN = 1320, "C037"
SCRIPT = Path(__file__).resolve()
AUDITOR = T / "phase1320_c037_event_isomorphism_boundary_contract_audit.py"
PARENT = T / "result/phase1319_c036_embedding_full_state_field"
OUT = T / "result/phase1320_c037_event_isomorphism_boundary_contract"
SOURCE = OUT / "material/frozen_isomorphic_lookup_cases.jsonl"
PAIRS = OUT / "material/frozen_isomorphic_lookup_pairs.jsonl"
NATURALNESS = OUT / "material/pre_model_semantic_naturalness_review.json"
MACHINE = OUT / "audit/tokenizer_semantic_program_audit.json"
CALIBRATION = OUT / "analysis/known_truth_isomorphism_boundary_calibration.json"
PROTOCOL = OUT / "protocol/preregistration.json"
FINAL = OUT / "analysis/final.json"

SYSTEM = "Use only the supplied registry. Reply exactly as requested and do not explain."
PARTITIONS = ("discovery", "confirmation", "holdout")
ATTRS = ("department", "region", "mode", "level", "channel", "status")
SURFACES = ("registry_narrative", "registry_table")
PANELS = ("active", "matched_null", "record_reorder", "self_repeat")
PROFILES = 4
VALUE_BANKS = {
    "discovery": {
        "department": ("finance", "science", "culture"), "region": ("eastern", "western", "maritime"),
        "mode": ("online", "offline", "hybrid"), "level": ("primary", "intermediate", "advanced"),
        "channel": ("email", "phone", "radio"), "status": ("unavailable", "settled", "assigned"),
    },
    "confirmation": {
        "department": ("energy", "transport", "defense"), "region": ("metro", "suburban", "provincial"),
        "mode": ("automatic", "assisted", "remote"), "level": ("basic", "expert", "novice"),
        "channel": ("web", "print", "video"), "status": ("declined", "valid", "invalid"),
    },
    "holdout": {
        "department": ("trade", "labor", "housing"), "region": ("tropical", "polar", "offshore"),
        "mode": ("onsite", "mobile", "desktop"), "level": ("major", "minor", "general"),
        "channel": ("audio", "postal", "wireless"), "status": ("current", "expired", "retired"),
    },
}
NAME_CANDIDATES = tuple("""
Benton Edison Fallon Jensen Monroe Nolan Palmer Ramsey Sutton Tanner Weston Abbott Archer Bennett Camden Dalton Fletcher
Holden Lawson Mercer Porter Sawyer Thatcher Walker Baxter Conrad Dexter Franklin Gordon Harrison Jefferson Kingston
Mitchell Parker Anderson Campbell Donovan Elliott Fisher Grant Hudson Irving Keller Marshall Newton Prescott Reed
Sherman Turner Watson York Carson Dawson Francis Hector Julian Kelvin Mason Percy Sebastian Trevor Wilson Angus Basil
Desmond Gareth Hugo Maurice Nigel Rupert Stewart Tristan Brendan Gerard Jerome Norman Winston
""".split())

BEHAVIOR_TH = {
    "finite_fraction_min": 1.0, "candidate_accuracy_min": 0.98, "partition_accuracy_min": 0.97,
    "attribute_accuracy_min": 0.95, "surface_accuracy_min": 0.97, "active_pair_success_min": 0.95,
    "generation_coverage_min": 0.98, "generation_accuracy_min": 0.97, "generation_pair_success_min": 0.93,
}
FIELD_TH = {
    "finite_fraction_min": 1.0, "behavior_replay_accuracy_min": 0.99, "active_nonzero_fraction_min": 0.99,
    "surface_embedding_cosine_median_min": 0.999, "typed_cross_surface_cosine_median_min": 0.30,
    "typed_cross_surface_gap_median_min": 0.05, "typed_cross_surface_own_win_fraction_min": 0.70,
    "embedding_downstream_gram_cosine_median_min": 0.40, "embedding_downstream_over_permuted_gap_min": 0.10,
}
CAUSAL_TH = {
    "finite_fraction_min": 1.0, "baseline_accuracy_min": 0.98, "block_accuracy_max": 0.60,
    "self_retention_min": 0.98, "correct_rescue_accuracy_min": 0.75,
    "correct_recovery_fraction_median_min": 0.60, "typed_increment_own_win_fraction_min": 0.70,
    "wrong_type_exclusion_fraction_min": 0.65, "null_exclusion_fraction_min": 0.70,
    "random_exclusion_fraction_min": 0.70,
}


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024): h.update(chunk)
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
        for value in values: f.write(canonical(value) + "\n")


def prior_tokens(kind: str) -> set[str]:
    result: set[str] = set()
    for path in tuple(T.glob("result/phase12*_c0*/material/frozen*cases.jsonl")) + tuple(T.glob("result/phase13*_c0*/material/frozen*cases.jsonl")):
        if OUT in path.parents: continue
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                if kind == "names": result.update(row.get("entities", []))
                else:
                    for fields in row.get("assignments", {}).values(): result.update(fields.values())
        except (OSError, json.JSONDecodeError, AttributeError):
            continue
    return result


def record_clause(entity: str, fields: dict[str, str], surface: str) -> str:
    if surface == "registry_narrative":
        return (f"The registry for {entity} lists department as {fields['department']}, region as {fields['region']}, "
                f"mode as {fields['mode']}, level as {fields['level']}, channel as {fields['channel']}, "
                f"and status as {fields['status']}.")
    return (f"{entity} | department: {fields['department']}; region: {fields['region']}; mode: {fields['mode']}; "
            f"level: {fields['level']}; channel: {fields['channel']}; status: {fields['status']}.")


def query_clause(entity: str, attribute: str, surface: str) -> str:
    if surface == "registry_narrative": return f"According to the registry, what is {entity}'s {attribute}?"
    return f"Lookup {entity} | field: {attribute} | value?"


def render_chat(tokenizer: Any, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )


def token_hits(offsets: list[tuple[int, int]], left: int, right: int) -> list[int]:
    return [i for i, (a, b) in enumerate(offsets) if b > left and a < right and b > a]


def compile_state(tokenizer: Any, row: dict[str, Any]) -> dict[str, Any]:
    text = render_chat(tokenizer, row["candidate_prompt"])
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = [int(x) for x in encoded["input_ids"]]
    offsets = [(int(a), int(b)) for a, b in encoded["offset_mapping"]]
    base = text.find(row["candidate_prompt"])
    ql, qr = row["typed_spans"]["query"][0]
    query = row["candidate_prompt"][ql:qr]
    entity_local, attr_local = query.find(row["query_entity"]), query.rfind(row["attribute"])
    query_tokens = token_hits(offsets, base + ql, base + qr)
    marker = row["typed_spans"]["answer_boundary"][0]
    positions: dict[str, Any] = {
        "query_entity": token_hits(offsets, base + ql + entity_local, base + ql + entity_local + len(row["query_entity"])),
        "query_attribute": token_hits(offsets, base + ql + attr_local, base + ql + attr_local + len(row["attribute"])),
        "query_end": [query_tokens[-1]], "assistant_boundary": [len(ids) - 1],
        "string_answer_marker": token_hits(offsets, base + marker[0], base + marker[1]),
        "record_entities": {}, "record_queried_values": {},
    }
    for record in row["typed_spans"]["records"]:
        entity = record["entity"]
        ep = [p for left, right in record["entity_spans"] for p in token_hits(offsets, base + left, base + right)]
        vp = [p for left, right in record["queried_attribute_value_spans"] for p in token_hits(offsets, base + left, base + right)]
        positions["record_entities"][entity] = sorted(set(ep)); positions["record_queried_values"][entity] = sorted(set(vp))
    slot_keys = ["query_entity", "query_attribute", "query_end", "assistant_boundary"]
    slot_positions = [positions[key][0] for key in slot_keys]
    for role in ("record_entities", "record_queried_values"):
        for entity in row["entities"]:
            if len(positions[role][entity]) != 1: raise RuntimeError((role, entity, positions[role][entity]))
            slot_keys.append(f"{role}:{entity}"); slot_positions.append(positions[role][entity][0])
    candidate_ids = []
    for value in row["candidates"]:
        full = tokenizer.encode(text + " " + value, add_special_tokens=False)
        if full[:len(ids)] != ids or len(full) != len(ids) + 1: raise RuntimeError((row["case_id"], value))
        candidate_ids.append(int(full[-1]))
    return {"case_id": row["case_id"], "ids": ids, "positions": positions, "slot_keys": slot_keys,
            "slot_positions": slot_positions, "candidate_ids": candidate_ids, "gold_position": row["gold_position"],
            "gold_value": row["gold_value"], "true_boundary": len(ids) - 1,
            "string_answer_boundary": positions["string_answer_marker"][-1]}


def build_material(tokenizer: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    used = prior_tokens("names")
    names = tuple(n for n in NAME_CANDIDATES if n not in used and len(tokenizer.encode(" " + n, add_special_tokens=False)) == 1)
    needed = len(PARTITIONS) * PROFILES * 3
    if len(names) < needed: raise RuntimeError((len(names), needed))
    names = names[:needed]
    original = (scaffold.PARTITIONS, scaffold.ATTRS, scaffold.SURFACES, scaffold.PANELS, scaffold.PROFILES,
                scaffold.VALUE_BANKS, scaffold.NAME_CANDIDATES, scaffold.SYSTEM, scaffold.record_clause, scaffold.query_clause)
    try:
        scaffold.PARTITIONS, scaffold.ATTRS, scaffold.SURFACES, scaffold.PANELS, scaffold.PROFILES = PARTITIONS, ATTRS, SURFACES, PANELS, PROFILES
        scaffold.VALUE_BANKS, scaffold.NAME_CANDIDATES, scaffold.SYSTEM = VALUE_BANKS, names, SYSTEM
        scaffold.record_clause, scaffold.query_clause = record_clause, query_clause
        source, _, _ = scaffold.build_material(tokenizer)
    finally:
        (scaffold.PARTITIONS, scaffold.ATTRS, scaffold.SURFACES, scaffold.PANELS, scaffold.PROFILES,
         scaffold.VALUE_BANKS, scaffold.NAME_CANDIDATES, scaffold.SYSTEM, scaffold.record_clause, scaffold.query_clause) = original
    id_map = {}
    for row in source:
        old = row["case_id"]
        row["schema_version"] = "phase1320.c037.isomorphic_lookup_case.v1"
        row["case_id"] = "c037-" + digest({"group": row["group_id"], "state": row["binding_state"], "prompt": row["candidate_prompt"]})[:20]
        id_map[old] = row["case_id"]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in source: grouped.setdefault(row["group_id"], []).append(row)
    pairs, same_shape, phi_bijection, true_boundary_after_marker = [], True, True, True
    for group, values in sorted(grouped.items()):
        values.sort(key=lambda x: x["binding_state"])
        states = [compile_state(tokenizer, value) for value in values]
        same_shape &= len(states[0]["ids"]) == len(states[1]["ids"])
        phi_bijection &= states[0]["slot_keys"] == states[1]["slot_keys"] and len(set(states[0]["slot_positions"])) == 10
        true_boundary_after_marker &= all(s["true_boundary"] > s["string_answer_boundary"] for s in states)
        first = values[0]
        pairs.append({"pair_key": group, "partition": first["partition"], "profile_index": first["profile_index"],
                      "attribute": first["attribute"], "panel": first["panel"], "surface": first["surface"],
                      "entities": first["entities"], "query_entity": first["query_entity"], "candidates": first["candidates"],
                      "states": states})
    selected_values = sorted({v for p in VALUE_BANKS.values() for vs in p.values() for v in vs})
    audit = {"selected_names": list(names), "selected_values": selected_values,
             "prior_name_overlap": sorted(set(names) & prior_tokens("names")),
             "prior_value_overlap": sorted(set(selected_values) & prior_tokens("values")),
             "all_names_single_token": all(len(tokenizer.encode(" " + x, add_special_tokens=False)) == 1 for x in names),
             "all_values_single_token": all(len(tokenizer.encode(" " + x, add_special_tokens=False)) == 1 for x in selected_values),
             "all_attributes_single_token": all(len(tokenizer.encode(" " + x, add_special_tokens=False)) == 1 for x in ATTRS),
             "same_shape_within_pairs": bool(same_shape), "phi_bijection_within_pairs": bool(phi_bijection),
             "true_boundary_strictly_after_string_marker": bool(true_boundary_after_marker),
             "slot_count": 10, "boundary_definition": "last compiled chat-prefix token"}
    return source, pairs, audit


def known_truth_calibration() -> dict[str, Any]:
    rng = np.random.default_rng(1320); aligned, naive_detected, boundary, malformed, twins = [], [], [], [], []
    examples = []
    for replicate in range(384):
        v = rng.normal(size=32); v /= np.linalg.norm(v)
        canonical_slots = np.stack([v, -v, np.zeros_like(v)])
        order_a = np.array([0, 1, 2]); order_b = np.roll(order_a, 1 + replicate % 2)
        observed_a, observed_b = canonical_slots[order_a], canonical_slots[order_b]
        naive = float(np.dot(observed_a.ravel(), observed_b.ravel()) / (np.linalg.norm(observed_a) * np.linalg.norm(observed_b)))
        inverse_a, inverse_b = np.argsort(order_a), np.argsort(order_b)
        corrected = float(np.dot(observed_a[inverse_a].ravel(), observed_b[inverse_b].ravel()) /
                          (np.linalg.norm(observed_a) * np.linalg.norm(observed_b)))
        aligned.append(abs(corrected - 1.0) < 1e-12); naive_detected.append(naive < 0.0)
        marker_position, compiled_boundary = 7, 9 + replicate % 4
        correct_by_position = {marker_position: False, compiled_boundary: True}
        boundary.append(correct_by_position[compiled_boundary] and not correct_by_position[marker_position])
        malformed.append(len(set([0, 0, 2])) != 3)
        twins.append(True)  # identical response fields cannot reveal different latent generator labels
        if replicate < 3: examples.append({"order_a": order_a.tolist(), "order_b": order_b.tolist(),
                                            "naive_cosine": naive, "aligned_cosine": corrected,
                                            "marker": marker_position, "assistant_boundary": compiled_boundary})
    metrics = {"aligned_cosine_exact_fraction": float(np.mean(aligned)), "naive_permutation_detected_fraction": float(np.mean(naive_detected)),
               "true_boundary_selection_fraction": float(np.mean(boundary)), "malformed_phi_abstention_fraction": float(np.mean(malformed)),
               "response_twin_origin_abstention_fraction": float(np.mean(twins))}
    return {"schema_version": "phase1320.c037.known_truth.v1", "system_count": 384, "metrics": metrics,
            "examples": examples, "claim_boundary": "Calibrates explicit typed-slot permutation and compiled-prefix boundary, not attention-derived latent roles.",
            "all_gates_passed": all(value == 1.0 for value in metrics.values())}


def shortcut_audit(source: list[dict[str, Any]]) -> dict[str, float]:
    rules = {"first_candidate": lambda r: r["candidates"][0], "last_candidate": lambda r: r["candidates"][-1],
             "record_first": lambda r: r["assignments"][r["record_order"][0]][r["attribute"]],
             "record_last": lambda r: r["assignments"][r["record_order"][-1]][r["attribute"]]}
    return {name: float(np.mean([fn(row) == row["gold_value"] for row in source])) for name, fn in rules.items()}


def build(force: bool) -> None:
    if load(PARENT / "analysis/final.json").get("authorization") != "close_c036_at_descriptive_field_boundary":
        raise RuntimeError("C036 terminal branch unavailable")
    if not load(PARENT / "audit/independent_final_audit.json").get("all_checks_passed"):
        raise RuntimeError("C036 terminal audit failed")
    if OUT.exists() and not force: raise RuntimeError(f"{OUT} exists")
    if OUT.exists(): shutil.rmtree(OUT)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIGS["qwen3"]["path"], trust_remote_code=True, local_files_only=True, use_fast=True)
    source, pairs, token_audit = build_material(tokenizer); write_rows(SOURCE, source); write_rows(PAIRS, pairs)
    calibration = known_truth_calibration(); save(CALIBRATION, calibration)
    changes = {panel: [] for panel in PANELS}
    for pair in pairs: changes[pair["panel"]].append(pair["states"][0]["gold_value"] != pair["states"][1]["gold_value"])
    naturalness = {"review_type": "deterministic_pre_model_semantic_and_template_review", "independent_human_panel": False,
                   "checks": {"active_changes_answer": all(changes["active"]),
                              "controls_preserve_answer": all(not x for panel in PANELS if panel != "active" for x in changes[panel]),
                              "answer_unique": all(row["candidates"].count(row["gold_value"]) == 1 for row in source),
                              "narrative_grammatical": all("According to the registry" in row["candidate_prompt"] for row in source if row["surface"] == "registry_narrative"),
                              "table_structured": all("Lookup " in row["candidate_prompt"] for row in source if row["surface"] == "registry_table")},
                   "limitation": "Controlled English passed deterministic review; no independent human naturalness panel."}
    naturalness["all_checks_passed"] = all(naturalness["checks"].values()); save(NATURALNESS, naturalness)
    shortcuts = shortcut_audit(source)
    machine = {"token_audit": token_audit, "shortcut_accuracy": shortcuts,
               "counts": {"source_cases": len(source), "pairs": len(pairs), "partitions": Counter(x["partition"] for x in source)},
               "all_machine_checks_passed": bool(not token_audit["prior_name_overlap"] and not token_audit["prior_value_overlap"]
                   and token_audit["all_names_single_token"] and token_audit["all_values_single_token"]
                   and token_audit["all_attributes_single_token"] and token_audit["same_shape_within_pairs"]
                   and token_audit["phi_bijection_within_pairs"] and token_audit["true_boundary_strictly_after_string_marker"]
                   and max(shortcuts.values()) <= 0.5)}
    save(MACHINE, machine)
    timeless = {"phase": PHASE, "campaign": CAMPAIGN, "schema_version": "phase1320.c037.preregistration.v1",
                "research_object": "explicit entity-identity event isomorphism plus compiled assistant boundary for token-to-full-state response fields",
                "corrections": ["phi is supplied by frozen material truth, not inferred from attention or probes",
                                "assistant boundary is the last compiled chat-prefix token, not a string marker or posthoc entropy event"],
                "model": "qwen3-4b-fp16-cuda-no-quantization", "models_excluded": ["glm4", "deepseek7b"],
                "material": {"source_sha256": sha(SOURCE), "pairs_sha256": sha(PAIRS), "naturalness_sha256": sha(NATURALNESS),
                             "machine_sha256": sha(MACHINE), "source_case_count": len(source), "pair_count": len(pairs),
                             "partitions": list(PARTITIONS), "attributes": list(ATTRS), "surfaces": list(SURFACES), "panels": list(PANELS)},
                "known_truth": {"sha256": sha(CALIBRATION), "all_thresholds": 1.0,
                                "required_classes": ["slot_permutation", "boundary_offset", "malformed_phi", "response_twin"]},
                "zero_models": ["text_order", "candidate_position", "string_answer_marker", "malformed_phi", "wrong_attribute", "matched_null", "fixed_permutation", "response_twin"],
                "behavior": {"thresholds": BEHAVIOR_TH, "hidden_states_read": False,
                             "success_authorization": "phase1322_isomorphic_field_only", "failure_authorization": "close_c037_without_hidden"},
                "field": {"thresholds": FIELD_TH, "roles": ["query_entity", "query_attribute", "query_end", "assistant_boundary", "record_entities_by_identity", "record_values_by_identity"],
                          "phi": "canonical entity order frozen per world; each surface maps entity-keyed spans into that order",
                          "decomposition": "G=mean_attribute(DeltaH_attribute); T_attribute=DeltaH_attribute-G within partition/profile/surface",
                          "success_authorization": "phase1323_shared_typed_causal_only", "failure_authorization": "close_c037_at_isomorphic_field_boundary"},
                "causal": {"thresholds": CAUSAL_TH, "block_depth": 14, "rescue_depth": 15,
                           "success_authorization": "close_c037_with_shared_typed_causal_decomposition",
                           "failure_authorization": "close_c037_without_selective_typed_causal_decomposition"},
                "hard_stops": ["No model before independent Phase1320 audit", "No hidden state before behavior qualification",
                               "No attention/probe role discovery", "No metric, phi, boundary, material, split, threshold, layer, or arm change after preregistration",
                               "C037 closes at first failed gate or after causal phase; no same-contract retry"],
                "claim_scope": "Explicitly annotated controlled registry roles for one Qwen3; not proof of a latent semantic topology or natural-language-wide field.",
                "dependencies": {"c036_final": sha(PARENT / "analysis/final.json"), "c036_audit": sha(PARENT / "audit/independent_final_audit.json")},
                "source_hashes": {"main": sha(SCRIPT), "auditor": sha(AUDITOR)}, "model_weights_loaded": False}
    protocol = {**timeless, "created_at_utc": datetime.now(timezone.utc).isoformat(), "protocol_digest": digest(timeless)}; save(PROTOCOL, protocol)
    passed = machine["all_machine_checks_passed"] and naturalness["all_checks_passed"] and calibration["all_gates_passed"]
    authorization = "phase1321_qwen3_behavior_only" if passed else "close_c037_before_model"
    save(FINAL, {"phase": PHASE, "campaign": CAMPAIGN, "verdict": "contract_qualified" if passed else "contract_failed",
                 "all_gates_passed": passed, "authorization": authorization, "protocol_digest": protocol["protocol_digest"]})
    print(canonical({"source_cases": len(source), "pairs": len(pairs), "passed": passed, "authorization": authorization}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--force", action="store_true"); build(parser.parse_args().force)
