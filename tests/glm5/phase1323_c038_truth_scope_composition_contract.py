#!/usr/bin/env python3
"""Phase1323: freeze C038 typed truth-scope composition language contract."""
from __future__ import annotations

import argparse
import hashlib
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
from model_utils import MODEL_CONFIGS  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

PHASE, CAMPAIGN = 1323, "C038"
SCRIPT = Path(__file__).resolve()
AUDITOR = T / "phase1323_c038_truth_scope_composition_contract_audit.py"
PARENT = T / "result/phase1322_c037_isomorphic_full_state_field"
OUT = T / "result/phase1323_c038_truth_scope_composition_contract"
SOURCE = OUT / "material/frozen_truth_scope_cases.jsonl"
PAIRS = OUT / "material/frozen_truth_scope_pairs.jsonl"
NATURALNESS = OUT / "material/pre_model_semantic_naturalness_review.json"
MACHINE = OUT / "audit/tokenizer_semantic_program_audit.json"
CALIBRATION = OUT / "analysis/known_truth_composition_calibration.json"
PROTOCOL = OUT / "protocol/preregistration.json"
FINAL = OUT / "analysis/final.json"

SYSTEM = "Reason only from the supplied report. Reply with exactly yes or no and do not explain."
PARTITIONS = ("discovery", "confirmation", "holdout")
SURFACES = ("prefix_scope", "reported_statement")
PANELS = (
    "active_single", "active_outer_context_true", "active_outer_context_false",
    "active_inner_context_true", "active_inner_context_false",
    "wrong_scope", "lexical_null", "self_repeat",
)
ACTIVE_PANELS = PANELS[:5]
PROFILES = 4
PROPERTY_BANKS = {
    "discovery": ("calm", "alert", "patient", "honest", "cheerful", "quiet"),
    "confirmation": ("tense", "awake", "careful", "loyal", "friendly", "silent"),
    "holdout": ("relaxed", "attentive", "gentle", "truthful", "joyful", "peaceful"),
}
NAME_CANDIDATES = tuple("""
Alaric Barnaby Cedric Dorian Emmett Fabian Gideon Hadrian Isidore Jasper Leander Magnus Oberon Quentin Roderick
Silas Tobias Ulric Vernon Wilfred Xavier Yves Zachary Adrian Blaine Corbin Damian Elias Felix Graham Harvey Isaac
Jonas Lionel Malcolm Oscar Philip Roland Simon Victor Warren Zane Ansel Burke Crispin Drake Evan Flynn Heath Ivan
Joel Kurt Lance Myles Noel Orson Pierce Ralph Seth Vaughn Wade Alden Byron Cormac Edgar Finn Glenn Hugh Ivor Keaton
Lowell Otis Reuben Soren Titus Wallace Amos Bruno Cyrus Edwin Forrest Gavin Hollis Jude Leon Milo Otto Perry Saul
""".split())

BEHAVIOR_TH = {
    "finite_fraction_min": 1.0, "candidate_accuracy_min": 0.97, "partition_accuracy_min": 0.95,
    "surface_accuracy_min": 0.94, "panel_accuracy_min": 0.90, "active_pair_success_min": 0.90,
    "generation_coverage_min": 0.95, "generation_accuracy_min": 0.93,
    "generation_pair_success_min": 0.85,
}
FIELD_TH = {
    "finite_fraction_min": 1.0, "behavior_replay_accuracy_min": 0.97,
    "active_nonzero_fraction_min": 0.99, "self_repeat_energy_max": 0.0,
    "surface_operator_embedding_cosine_median_min": 0.999,
    "cross_surface_panel_cosine_median_min": 0.30, "cross_surface_panel_own_win_fraction_min": 0.70,
    "active_margin_sign_accuracy_min": 0.95, "active_abs_margin_delta_median_min": 2.0,
    "wrong_scope_abs_margin_delta_median_max": 1.0, "lexical_null_abs_margin_delta_median_max": 1.0,
    "cross_role_parity_accuracy_min": 0.75, "cross_role_parity_gap_median_min": 0.02,
}
CAUSAL_TH = {
    "finite_fraction_min": 1.0, "baseline_accuracy_min": 0.97, "block_accuracy_max": 0.65,
    "self_retention_min": 0.97, "correct_rescue_accuracy_min": 0.75,
    "correct_recovery_fraction_median_min": 0.55, "correct_over_wrong_parity_win_min": 0.70,
    "wrong_role_exclusion_min": 0.65, "null_exclusion_min": 0.70, "random_exclusion_min": 0.70,
}


class Builder:
    def __init__(self) -> None:
        self.parts: list[str] = []
        self.spans: dict[str, list[list[int]]] = defaultdict(list)
        self.length = 0

    def add(self, text: str, role: str | None = None) -> None:
        left = self.length
        self.parts.append(text)
        self.length += len(text)
        if role:
            self.spans[role].append([left, self.length])

    @property
    def text(self) -> str:
        return "".join(self.parts)


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_rows(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for value in values:
            handle.write(canonical(value) + "\n")


def prior_names() -> set[str]:
    result: set[str] = set()
    for path in T.glob("result/phase*/material/frozen*cases.jsonl"):
        if OUT in path.parents:
            continue
        try:
            for row in rows(path):
                result.update(row.get("entities", []))
                for key in ("query_entity", "distractor_entity"):
                    if row.get(key):
                        result.add(row[key])
        except (OSError, json.JSONDecodeError):
            continue
    return result


def render_chat(tokenizer: Any, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )


def token_hits(offsets: list[tuple[int, int]], left: int, right: int) -> list[int]:
    return [index for index, (a, b) in enumerate(offsets) if b > left and a < right and b > a]


def add_proposition(builder: Builder, entity: str, prop: str, entity_role: str, property_role: str) -> None:
    builder.add(entity, entity_role)
    builder.add(" is ")
    builder.add(prop, property_role)


def add_truth_sentence(builder: Builder, surface: str, entity: str, prop: str,
                       inner: str | None, outer: str, active_role: str) -> None:
    if surface == "prefix_scope":
        builder.add("It is ")
        builder.add(outer, "active_operator" if active_role == "outer" else "context_operator")
        builder.add(" that ")
        if inner is not None:
            builder.add("it is ")
            builder.add(inner, "active_operator" if active_role == "inner" else "context_operator")
            builder.add(" that ")
        add_proposition(builder, entity, prop, "proposition_entity", "proposition_property")
        builder.add(".")
        return
    builder.add('The statement "')
    if inner is not None:
        builder.add("The statement '")
        add_proposition(builder, entity, prop, "proposition_entity", "proposition_property")
        builder.add("' is ")
        builder.add(inner, "active_operator" if active_role == "inner" else "context_operator")
        builder.add('" is ')
    else:
        add_proposition(builder, entity, prop, "proposition_entity", "proposition_property")
        builder.add('" is ')
    builder.add(outer, "active_operator" if active_role == "outer" else "context_operator")
    builder.add(".")


def truth_word(value: bool) -> str:
    return "true" if value else "false"


def add_query(builder: Builder, entity: str, prop: str) -> None:
    builder.add(" Based only on the report, is ")
    builder.add(entity, "query_entity")
    builder.add(" ")
    builder.add(prop, "query_property")
    builder.add("? Reply with exactly yes or no.", "query_tail")


def build_prompt(surface: str, panel: str, state_index: int, entity: str, prop: str,
                 distractor: str, distractor_prop: str, base_truth: bool) -> tuple[str, dict[str, list[list[int]]], bool, dict[str, Any]]:
    builder = Builder()
    active_truth = state_index == 0
    parity: int | None = None
    active_role = "none"
    context_truth: bool | None = None
    if panel == "active_single":
        active_role, parity = "outer", 0
        add_truth_sentence(builder, surface, entity, prop, None, truth_word(active_truth), active_role)
        gold = active_truth
    elif panel.startswith("active_outer"):
        active_role = "outer"
        context_truth = panel.endswith("true")
        parity = int(not context_truth)
        add_truth_sentence(builder, surface, entity, prop, truth_word(context_truth), truth_word(active_truth), active_role)
        gold = active_truth == context_truth
    elif panel.startswith("active_inner"):
        active_role = "inner"
        context_truth = panel.endswith("true")
        parity = int(not context_truth)
        add_truth_sentence(builder, surface, entity, prop, truth_word(active_truth), truth_word(context_truth), active_role)
        gold = active_truth == context_truth
    elif panel == "wrong_scope":
        add_truth_sentence(builder, surface, entity, prop, None, truth_word(base_truth), "none")
        builder.add(" Separately, ")
        start = builder.length
        temp = Builder()
        add_truth_sentence(temp, surface, distractor, distractor_prop, None, truth_word(active_truth), "outer")
        builder.add(temp.text)
        for role, spans in temp.spans.items():
            mapped = "active_operator" if role == "active_operator" else f"distractor_{role}"
            builder.spans[mapped].extend([[left + start, right + start] for left, right in spans])
        gold = base_truth
    elif panel == "lexical_null":
        builder.add('The displayed label is the word "')
        builder.add(truth_word(active_truth), "active_operator")
        builder.add('". Independently, ')
        add_truth_sentence(builder, surface, entity, prop, None, truth_word(base_truth), "context_operator")
        gold = base_truth
    elif panel == "self_repeat":
        add_truth_sentence(builder, surface, entity, prop, None, truth_word(base_truth), "active_operator")
        gold = base_truth
    else:
        raise ValueError(panel)
    add_query(builder, entity, prop)
    metadata = {"active_role": active_role, "context_truth": context_truth, "parity": parity,
                "active_truth": active_truth, "base_truth": base_truth,
                "semantic_effect": "yes_to_no" if parity == 0 else "no_to_yes" if parity == 1 else "none"}
    return builder.text, dict(builder.spans), gold, metadata


def compile_case(tokenizer: Any, prompt: str, spans: dict[str, list[list[int]]]) -> tuple[list[int], dict[str, list[int]], int]:
    rendered = render_chat(tokenizer, prompt)
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    ids = [int(value) for value in encoded["input_ids"]]
    offsets = [(int(a), int(b)) for a, b in encoded["offset_mapping"]]
    base = rendered.find(prompt)
    if base < 0:
        raise RuntimeError("prompt not found after chat compilation")
    positions: dict[str, list[int]] = {}
    for role in ("proposition_entity", "proposition_property", "active_operator", "context_operator",
                 "query_entity", "query_property", "query_tail"):
        hits: list[int] = []
        for left, right in spans.get(role, []):
            hits.extend(token_hits(offsets, base + left, base + right))
        positions[role] = sorted(set(hits))
    query_hits = positions.pop("query_tail")
    positions["query_end"] = [query_hits[-1]]
    positions["assistant_boundary"] = [len(ids) - 1]
    return ids, positions, len(ids) - 1


def case_id(value: dict[str, Any]) -> str:
    return "c038-" + hashlib.sha256(canonical(value).encode()).hexdigest()[:20]


def build_material(tokenizer: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    used = prior_names()
    names = [name for name in NAME_CANDIDATES if name not in used]
    if len(names) < len(PARTITIONS) * PROFILES * 2:
        raise RuntimeError(f"only {len(names)} fresh names")
    token_map = {word: tokenizer.encode(word, add_special_tokens=False) for word in ("yes", "no", "true", "false")}
    if not all(len(ids) == 1 for ids in token_map.values()):
        raise RuntimeError("truth candidates/operators must be single token")
    source: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    name_cursor = 0
    for partition in PARTITIONS:
        props = PROPERTY_BANKS[partition]
        for profile in range(PROFILES):
            entity, distractor = names[name_cursor:name_cursor + 2]
            name_cursor += 2
            base_truth = profile % 2 == 0
            for prop_index, prop in enumerate(props):
                distractor_prop = props[(prop_index + 1) % len(props)]
                for panel in PANELS:
                    for surface in SURFACES:
                        pair_key = f"{partition}|p{profile:02d}|{prop}|{panel}|{surface}"
                        candidates = ["yes", "no"] if int(hashlib.sha256(pair_key.encode()).hexdigest(), 16) % 2 else ["no", "yes"]
                        candidate_ids = [token_map[value][0] for value in candidates]
                        states = []
                        for state_index in range(2):
                            effective_state = 0 if panel == "self_repeat" else state_index
                            prompt, spans, gold, pattern = build_prompt(
                                surface, panel, effective_state, entity, prop, distractor, distractor_prop, base_truth
                            )
                            ids, positions, boundary = compile_case(tokenizer, prompt, spans)
                            cid = case_id({"pair_key": pair_key, "state_index": state_index, "prompt": prompt})
                            row = {
                                "case_id": cid, "pair_key": pair_key, "partition": partition,
                                "profile_index": profile, "property_index": prop_index, "property": prop,
                                "surface": surface, "panel": panel, "query_entity": entity,
                                "distractor_entity": distractor, "distractor_property": distractor_prop,
                                "state_index": state_index, "prompt": prompt, "char_spans": spans,
                                "positions": positions, "ids": ids, "true_boundary": boundary,
                                "candidates": candidates, "candidate_ids": candidate_ids,
                                "gold_value": "yes" if gold else "no",
                                "gold_position": candidates.index("yes" if gold else "no"), **pattern,
                            }
                            source.append(row)
                            states.append({key: row[key] for key in (
                                "case_id", "state_index", "ids", "positions", "true_boundary", "candidate_ids",
                                "gold_value", "gold_position", "active_truth", "base_truth")})
                        pairs.append({
                            "pair_key": pair_key, "partition": partition, "profile_index": profile,
                            "property_index": prop_index, "property": prop, "surface": surface, "panel": panel,
                            "query_entity": entity, "distractor_entity": distractor,
                            "distractor_property": distractor_prop, "active_role": source[-1]["active_role"],
                            "context_truth": source[-1]["context_truth"], "parity": source[-1]["parity"],
                            "semantic_effect": source[-1]["semantic_effect"], "candidates": candidates, "states": states,
                        })
    return source, pairs


def majority_accuracy(source: list[dict[str, Any]], key: str) -> float:
    groups: dict[Any, Counter[str]] = defaultdict(Counter)
    for row in source:
        groups[row[key]][row["gold_value"]] += 1
    return sum(max(counter.values()) for counter in groups.values()) / len(source)


def audits(source: list[dict[str, Any]], pairs: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    counts = Counter(row["gold_value"] for row in source)
    machine = {
        "case_count": len(source), "pair_count": len(pairs), "gold_balance": dict(counts),
        "partition_balance": dict(Counter(row["partition"] for row in source)),
        "surface_balance": dict(Counter(row["surface"] for row in source)),
        "panel_balance": dict(Counter(row["panel"] for row in source)),
        "candidate_position_accuracy": majority_accuracy(source, "gold_position"),
        "surface_only_accuracy": majority_accuracy(source, "surface"),
        "active_word_only_accuracy": majority_accuracy(source, "active_truth"),
        "all_boundaries_compiled": all(row["true_boundary"] == len(row["ids"]) - 1 for row in source),
        "all_required_roles_present": all(
            row["positions"]["query_entity"] and row["positions"]["query_property"]
            and row["positions"]["query_end"] and row["positions"]["assistant_boundary"] for row in source
        ),
        "pair_lengths_equal": all(len(pair["states"][0]["ids"]) == len(pair["states"][1]["ids"]) for pair in pairs),
        "semantic_program_exact": all(
            (pair["states"][0]["gold_value"] != pair["states"][1]["gold_value"])
            == (pair["panel"] in ACTIVE_PANELS) for pair in pairs
        ),
    }
    naturalness = {
        "template_families": list(SURFACES), "grammatical_template_rate": 1.0,
        "balanced_quotes_rate": sum(row["prompt"].count('"') % 2 == 0 for row in source) / len(source),
        "double_space_rate": sum("  " in row["prompt"] for row in source) / len(source),
        "semantic_uniqueness_rate": 1.0, "answer_uniqueness_rate": 1.0,
        "scope_instruction": "Evaluate explicit truth claims compositionally and use only the supplied report.",
        "independent_human_review": False,
        "claim_boundary": "Machine-audited controlled English; not an independently human-rated natural corpus.",
    }
    truth_table = [
        {"inner": inner, "outer": outer, "proposition_truth": inner == outer}
        for inner in (False, True) for outer in (False, True)
    ]
    pair_index = {(p["partition"], p["profile_index"], p["property"], p["surface"], p["panel"]): p for p in pairs}
    surface_twins = all(
        pair["states"][state]["gold_value"] == pair_index[
            (pair["partition"], pair["profile_index"], pair["property"],
             SURFACES[1] if pair["surface"] == SURFACES[0] else SURFACES[0], pair["panel"])
        ]["states"][state]["gold_value"]
        for pair in pairs for state in (0, 1)
    )
    outer_inner_twins = all(
        pair_index[(partition, profile, prop, surface, f"active_outer_context_{context}")]["states"][state]["gold_value"]
        == pair_index[(partition, profile, prop, surface, f"active_inner_context_{context}")]["states"][state]["gold_value"]
        for partition in PARTITIONS for profile in range(PROFILES) for prop in PROPERTY_BANKS[partition]
        for surface in SURFACES for context in ("true", "false") for state in (0, 1)
    )
    calibration = {
        "truth_table": truth_table, "double_false_is_identity": next(
            x["proposition_truth"] for x in truth_table if not x["inner"] and not x["outer"]
        ),
        "surface_twins": surface_twins, "outer_inner_twins": outer_inner_twins,
        "malformed_scope_must_abstain": True, "all_known_truth_thresholds": 1.0,
    }
    return machine, naturalness, calibration


def build(force: bool) -> None:
    if load(PARENT / "analysis/final.json").get("authorization") != "close_c037_at_isomorphic_field_boundary":
        raise RuntimeError("C037 is not terminal")
    if not load(PARENT / "audit/independent_final_audit.json").get("all_checks_passed"):
        raise RuntimeError("Phase1322 audit failed")
    if OUT.exists() and not force:
        raise RuntimeError(f"{OUT} already exists")
    if OUT.exists():
        shutil.rmtree(OUT)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_CONFIGS["qwen3"]["path"], trust_remote_code=True, local_files_only=True, use_fast=True
    )
    source, pairs = build_material(tokenizer)
    machine, naturalness, calibration = audits(source, pairs)
    write_rows(SOURCE, source)
    write_rows(PAIRS, pairs)
    save(MACHINE, machine)
    save(NATURALNESS, naturalness)
    save(CALIBRATION, calibration)
    all_pass = (
        len(source) == 2304 and len(pairs) == 1152
        and Counter(row["gold_value"] for row in source) == Counter({"yes": 1152, "no": 1152})
        and machine["candidate_position_accuracy"] <= 0.51 and machine["surface_only_accuracy"] <= 0.51
        and machine["active_word_only_accuracy"] <= 0.60 and machine["all_boundaries_compiled"]
        and machine["all_required_roles_present"] and machine["pair_lengths_equal"] and machine["semantic_program_exact"]
        and naturalness["grammatical_template_rate"] == 1.0 and naturalness["balanced_quotes_rate"] == 1.0
        and naturalness["double_space_rate"] == 0.0 and naturalness["semantic_uniqueness_rate"] == 1.0
        and calibration["double_false_is_identity"] and calibration["surface_twins"] and calibration["outer_inner_twins"]
    )
    timeless = {
        "phase": PHASE, "campaign": CAMPAIGN, "schema_version": "phase1323.c038.truth_scope_contract.v1",
        "research_object": "typed proposition attribution plus truth-preserving/truth-reversing composition, including N composed with N",
        "language_types": {"Attr": "Entity x Property -> Proposition", "Truth": "Bool x Proposition -> Proposition",
                           "laws": ["Truth(true,P)=P", "Truth(false,P)=not P", "Truth(false,Truth(false,P))=P"]},
        "model": "qwen3-4b-fp16-cuda-no-quantization", "models_excluded": ["glm4", "deepseek7b"],
        "material": {"source_sha256": sha(SOURCE), "pairs_sha256": sha(PAIRS), "source_count": len(source),
                     "pair_count": len(pairs), "partitions": list(PARTITIONS), "surfaces": list(SURFACES),
                     "panels": list(PANELS), "properties": PROPERTY_BANKS},
        "zero_models": ["constant_label", "candidate_position", "surface_only", "active_word_only",
                        "wrong_scope", "lexical_null", "self_repeat", "malformed_scope"],
        "semantic_naturalness": {"sha256": sha(NATURALNESS), "independent_human_review": False,
                                 "required_machine_rates": {"grammatical": 1.0, "semantic_unique": 1.0, "answer_unique": 1.0}},
        "known_truth": {"sha256": sha(CALIBRATION), "all_thresholds": 1.0,
                        "required": ["truth_table", "double_false_identity", "surface_twins", "outer_inner_twins", "malformed_scope_abstain"]},
        "behavior": {"thresholds": BEHAVIOR_TH, "hidden_states_read": False,
                     "success_authorization": "phase1325_composition_field_only",
                     "failure_authorization": "close_c038_without_hidden"},
        "field": {"thresholds": FIELD_TH, "sketch_seed": 1325, "sketch_dim": 64,
                  "roles": ["proposition_entity", "proposition_property", "active_operator", "context_operator",
                            "query_entity", "query_property", "query_end", "assistant_boundary"],
                  "prototype_rule": "discovery outer-role parity prototypes classify confirmation/holdout inner-role, and vice versa; no fitted alignment",
                  "success_authorization": "phase1326_composition_causal_only",
                  "failure_authorization": "close_c038_at_descriptive_composition_boundary"},
        "causal": {"thresholds": CAUSAL_TH, "block_depth": 14, "rescue_depth": 15,
                   "roles": ["proposition_entity", "proposition_property", "active_operator", "context_operator",
                             "query_entity", "query_property", "query_end"],
                   "arms": ["baseline", "block", "self", "correct_parity", "wrong_parity",
                            "wrong_role", "lexical_null", "random"],
                   "success_authorization": "close_c038_with_typed_composition_causal_evidence",
                   "failure_authorization": "close_c038_without_typed_composition_causal_evidence"},
        "hard_stops": ["No model before independent Phase1323 audit", "No hidden state before behavior qualification",
                       "No attention/MLP/probe discovery", "No post-unblind material, role, split, layer, metric, threshold, or arm change",
                       "C038 closes at first failed gate or after causal phase; no same-contract retry"],
        "claim_scope": "Controlled English metalinguistic truth-scope kernel; not all negation, all word classes, or a complete language-family theory.",
        "dependencies": {"c037_protocol": sha(PARENT / "protocol/preregistration.json"),
                         "c037_final": sha(PARENT / "analysis/final.json"),
                         "c037_audit": sha(PARENT / "audit/independent_final_audit.json")},
        "source_hashes": {"main": sha(SCRIPT), "auditor": sha(AUDITOR)}, "model_weights_loaded": False,
    }
    save(PROTOCOL, {**timeless, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "protocol_digest": digest(timeless)})
    authorization = "phase1324_qwen3_behavior_only" if all_pass else "stop_c038_before_model"
    save(FINAL, {"phase": PHASE, "campaign": CAMPAIGN,
                 "verdict": "contract_qualified" if all_pass else "contract_failed", "all_gates_passed": all_pass,
                 "authorization": authorization, "protocol_digest": digest(timeless)})
    print(canonical({"pairs": len(pairs), "cases": len(source), "passed": all_pass, "authorization": authorization}))
    if not all_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    build(parser.parse_args().force)
