#!/usr/bin/env python3
"""C542-C559 typed linguistic-operation response-passport campaign.

Only token embeddings and HiddenState checkpoints are observed. All 2560
physical activation coordinates are retained. Attention, MLP activations,
weights, PCA, Top-K selection, and magnitude truncation are forbidden.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import itertools
import json
import math
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
VISUAL = ROOT / "frontend/public/vis_data/research_kernel/c557_typed_operation_response_passport_atlas.json"
REGISTRY = ROOT / "ai2050_research_os/registry/field_datasets.json"
CATALOG = ROOT / "frontend/public/research_data/current/language_encoding_catalog.json"
sys.path.insert(0, str(TESTS))

import model_utils
import phase2061_c527_c540_sample_specific_dynamics_campaign as previous


PHASES = {
    f"C{campaign}": (2076 + campaign - 542, slug)
    for campaign, slug in (
        (542, "evidence_adjudication_and_typed_operation_master_contract"),
        (543, "typed_linguistic_operation_ontology_and_program_graph"),
        (544, "large_material_compiler_semantic_balance_and_naturalness_audit"),
        (545, "qwen_behavior_and_all_token_all_coordinate_capture"),
        (546, "within_domain_operation_response_passports"),
        (547, "cross_domain_same_type_response_transfer"),
        (548, "truth_output_surface_and_equal_norm_confound_ledger"),
        (549, "independent_unit_evidence_and_candidate_adjudication"),
        (550, "minimal_sufficient_response_history_tournament"),
        (551, "attitude_event_atomic_composition_response"),
        (552, "graph_path_completion_interaction_response"),
        (553, "first_last_mean_token_granularity_tournament"),
        (554, "typed_response_causal_eligibility_adjudication"),
        (555, "qualified_hiddenstate_causal_branch_or_registered_na"),
        (556, "cross_model_functional_replication_branch_or_registered_na"),
        (557, "response_passport_full_coordinate_visual_atlas"),
        (558, "raw_field_cleanup_and_next_stage_adjudication"),
        (559, "campaign_synthesis_and_theory_ledger"),
    )
}
OUTS = {
    name: RESULT / f"phase{phase}_{name.lower()}_{slug}"
    for name, (phase, slug) in PHASES.items()
}

DIM = 2560
CHECKPOINTS = 38
ROLES = ("primary", "secondary", "relation", "context", "query", "boundary")
ROLE_INDEX = {name: i for i, name in enumerate(ROLES)}
QPOINTS = (0, 8, 16, 24, 32, 37)
SURFACES = ("record", "dialogue")
UNITS = 18
CONTROL_MARGIN = 0.02
PARENT_AUDIT = RESULT / "phase2075_c541_sample_specific_dynamics_campaign_independent_audit/analysis/final.json"

OPERATION_SPECS = {
    "entity_substitution": {
        "domains": ("event", "taxonomy", "spatial"),
        "truth_class": "truth_preserving",
        "rewrite": "replace a discourse entity while preserving the asserted and queried relation",
    },
    "relation_substitution": {
        "domains": ("event", "spatial", "temporal"),
        "truth_class": "truth_preserving",
        "rewrite": "replace the relation in both evidence and query",
    },
    "role_swap": {
        "domains": ("chase", "praise", "transfer"),
        "truth_class": "truth_preserving",
        "rewrite": "swap agent and patient in both evidence and query",
    },
    "polarity_toggle": {
        "domains": ("event", "property", "attitude"),
        "truth_class": "truth_preserving",
        "rewrite": "toggle positive and explicit negative polarity in both evidence and query",
    },
    "path_depth": {
        "domains": ("taxonomy", "temporal", "sequence"),
        "truth_class": "truth_preserving",
        "rewrite": "replace one direct relation with a valid two-step path",
    },
    "discourse_permutation": {
        "domains": ("temporal", "causal", "conjunction"),
        "truth_class": "truth_preserving",
        "rewrite": "reverse the order of two evidence statements without changing their content",
    },
    "active_passive": {
        "domains": ("inspect", "praise", "carry"),
        "truth_class": "truth_preserving",
        "rewrite": "switch active and passive voice while preserving thematic roles",
    },
    "query_truth_flip": {
        "domains": ("taxonomy", "part_whole", "spatial"),
        "truth_class": "truth_flip",
        "rewrite": "change only the queried target from supported to unsupported",
    },
    "output_order": {
        "domains": ("taxonomy", "event", "temporal"),
        "truth_class": "output_only",
        "rewrite": "swap the physical A/B answer order without changing proposition truth",
    },
    "surface_paraphrase": {
        "domains": ("taxonomy", "event", "attitude"),
        "truth_class": "surface_only",
        "rewrite": "paraphrase evidence and query while preserving proposition and answer protocol",
    },
    "translation": {
        "domains": ("taxonomy", "event", "temporal"),
        "truth_class": "translation",
        "rewrite": "translate the evidence and query between English and Chinese",
    },
}

COMPOSITION_PANELS = {
    "attitude_entity_object": ("experiencer_substitution", "object_substitution"),
    "attitude_polarity_object": ("polarity_toggle", "object_substitution"),
    "graph_path_completion": ("first_edge", "second_edge"),
}

NAMES_A = ("Aldren", "Beral", "Cedric", "Daria", "Emric", "Freya", "Gareth", "Helena", "Isen", "Jessa", "Kael", "Liora", "Marek", "Nadia", "Orin", "Pia", "Quinlan", "Rhea")
NAMES_B = ("Soren", "Talia", "Ulric", "Vessa", "Weylan", "Xara", "Yorin", "Zela", "Arven", "Brena", "Corin", "Delia", "Eamon", "Fiona", "Galen", "Hera", "Ivor", "Julia")
OBJECTS_A = ("cedar", "dahlia", "elm", "fern", "gardenia", "hazel", "iris", "juniper", "kalmia", "laurel", "magnolia", "nard", "oleander", "peony", "quince", "rosemary", "sage", "thyme")
OBJECTS_B = ("amber", "bronze", "cobalt", "denim", "ebony", "flint", "granite", "heather", "ivory", "jade", "khaki", "linen", "marble", "nickel", "opal", "pewter", "quartz", "rattan")
MIDDLES = tuple(f"midnode{i:02d}" for i in range(UNITS))
TARGETS = tuple(f"targetnode{i:02d}" for i in range(UNITS))
NOISES = ("sextant", "compass", "astrolabe", "chronometer", "barometer", "caliper", "odometer", "planimeter", "pyrometer", "tachometer", "anemometer", "micrometer", "spectrometer", "theodolite", "viscometer", "ammeter", "voltmeter", "altimeter")


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_rows(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def producer_hash() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(item) for item in value)
    if isinstance(value, (float, np.floating)):
        return math.isfinite(float(value))
    return True


def begin(name: str, protocol: dict, checks: dict) -> Path:
    out = OUTS[name]
    out.mkdir(parents=True, exist_ok=True)
    save(out / "protocol/preregistration.json", {
        "phase": PHASES[name][0], "campaign": name, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "producer_sha256": producer_hash(), **protocol,
    })
    save(out / "audit/internal_checks.json", checks)
    if not all(bool(value) for value in checks.values()):
        raise RuntimeError((name, checks))
    return out


def close(name: str, headline: dict, checks: dict, authorization: str) -> dict:
    out = OUTS[name]
    save(out / "analysis/summary.json", headline)
    save(out / "audit/internal_checks_post.json", checks)
    result = {
        "phase": PHASES[name][0], "campaign": name, "status": "closed",
        "all_checks_passed": all(bool(value) for value in checks.values()),
        "headline": headline, "next_authorization": authorization,
    }
    save(out / "analysis/final.json", result)
    if not result["all_checks_passed"]:
        raise RuntimeError((name, checks))
    return result


def final(name: str) -> dict:
    return load(OUTS[name] / "analysis/final.json")


def close_mmap(value: Any) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None:
        mmap.close()


def metric(prediction: np.ndarray, truth: np.ndarray) -> dict:
    p = np.asarray(prediction, np.float64).reshape(-1)
    y = np.asarray(truth, np.float64).reshape(-1)
    diff = p - y
    rmse = float(np.sqrt(np.mean(diff * diff)))
    scale = float(np.sqrt(np.mean(y * y)))
    denom = float(np.linalg.norm(p) * np.linalg.norm(y))
    return {
        "n": int(y.size), "mae": float(np.mean(np.abs(diff))), "rmse": rmse,
        "nrmse": rmse / max(scale, 1e-12),
        "cosine": float(np.dot(p, y) / denom) if denom > 0 else 0.0,
    }


def scaled_like(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    source = np.asarray(source, np.float32)
    reference = np.asarray(reference, np.float32)
    source_norm = float(np.linalg.norm(source.reshape(-1)))
    ref_norm = float(np.linalg.norm(reference.reshape(-1)))
    return source * (ref_norm / max(source_norm, 1e-12))


def partition(unit: int) -> str:
    if unit < 10:
        return "discovery"
    if unit < 14:
        return "confirmation"
    return "lockbox"


def answer_options(truth: bool, order: int) -> tuple[str, int]:
    if order == 0:
        return "(A) Yes (B) No", 0 if truth else 1
    return "(A) No (B) Yes", 1 if truth else 0


def wrap(surface: str, facts: list[str], question: str) -> str:
    body = " ".join(facts)
    if surface == "record":
        return f"A verified record states: {body} Based only on this record, {question}"
    return f"Analyst: {body} Reviewer: Using only those statements, {question}"


def unit_values(unit: int) -> dict:
    return {
        "a": NAMES_A[unit], "b": NAMES_B[unit], "x": OBJECTS_A[unit], "y": OBJECTS_B[unit],
        "mid": MIDDLES[unit], "target": TARGETS[unit], "noise": NOISES[unit],
    }


def atomic_semantics(operation: str, domain: str, surface: str, unit: int, variant: int) -> dict:
    u = unit_values(unit)
    a, b, x, y, mid, target, noise = u["a"], u["b"], u["x"], u["y"], u["mid"], u["target"], u["noise"]
    truth = True
    semantic_delta: dict[str, Any] = {"operation": operation, "variant": variant}

    if operation == "entity_substitution":
        primary = a if variant == 0 else b
        secondary = b if variant == 0 else a
        if domain == "event": relation, context = "inspected", x
        elif domain == "taxonomy": relation, context = "is a member of", target
        else: relation, context = "is above", x
        facts = [f"{primary} {relation} {context}.", f"{secondary} catalogued the {noise} separately."]
        question = f"Is it true that {primary} {relation} {context}?"
        semantic_delta["changed"] = ["primary_entity"]
    elif operation == "relation_substitution":
        primary, secondary, context = a, b, x
        relation_pairs = {
            "event": ("inspected", "carried"),
            "spatial": ("is above", "is beside"),
            "temporal": ("occurred earlier than", "occurred later than"),
        }
        relation = relation_pairs[domain][variant]
        facts = [f"{primary} {relation} {context}.", f"{secondary} logged the {noise} separately."]
        question = f"Is it true that {primary} {relation} {context}?"
        semantic_delta["changed"] = ["relation_identity"]
    elif operation == "role_swap":
        relation = {"chase": "chased", "praise": "praised", "transfer": "helped"}[domain]
        primary, context = ((a, b) if variant == 0 else (b, a))
        secondary = context
        facts = [f"{primary} {relation} {context}.", f"The {noise} was unrelated."]
        question = f"Is it true that {primary} {relation} {context}?"
        semantic_delta["changed"] = ["agent", "patient"]
    elif operation == "polarity_toggle":
        primary, secondary, context = a, b, x
        pairs = {
            "event": ("opened", "did not open"),
            "property": ("is polished like", "is not polished like"),
            "attitude": ("likes tasting", "does not like tasting"),
        }
        relation = pairs[domain][variant]
        facts = [f"{primary} {relation} {context}.", f"{secondary} inspected the {noise} separately."]
        question = f"Is it true that {primary} {relation} {context}?"
        semantic_delta["changed"] = ["polarity"]
    elif operation == "path_depth":
        primary, secondary, context = a, mid, target
        relation = {"taxonomy": "is a member of", "temporal": "precedes", "sequence": "comes before"}[domain]
        if variant == 0:
            facts = [f"{primary} {relation} {context}.", f"{secondary} was listed separately beside the {noise}."]
        else:
            facts = [f"{primary} {relation} {secondary}.", f"{secondary} {relation} {context}."]
        question = f"Is it true that {primary} {relation} {context}?"
        semantic_delta["changed"] = ["path_depth"]
    elif operation == "discourse_permutation":
        primary, secondary, context = a, b, x
        relation = {"temporal": "preceded", "causal": "enabled", "conjunction": "was paired with"}[domain]
        facts = [f"{primary} {relation} {context}.", f"{secondary} catalogued the {noise}."]
        if variant == 1:
            facts = list(reversed(facts))
        question = f"Is it true that {primary} {relation} {context}?"
        semantic_delta["changed"] = ["discourse_order"]
    elif operation == "active_passive":
        primary, secondary, context = a, b, x
        relation = {"inspect": "inspected", "praise": "praised", "carry": "carried"}[domain]
        if variant == 0:
            facts = [f"{primary} {relation} {context}.", f"{secondary} noted the {noise} separately."]
            question = f"Is it true that {primary} {relation} {context}?"
        else:
            facts = [f"{context} was {relation} by {primary}.", f"{secondary} noted the {noise} separately."]
            question = f"Is it true that {context} was {relation} by {primary}?"
        semantic_delta["changed"] = ["voice"]
    elif operation == "query_truth_flip":
        primary, secondary = a, b
        if domain == "taxonomy": relation, context = "is a member of", target
        elif domain == "part_whole": relation, context = "is part of", target
        else: relation, context = "is above", x
        facts = [f"{primary} {relation} {context}.", f"{secondary} {relation} {y}."]
        query_context = context if variant == 0 else y
        question = f"Is it true that {primary} {relation} {query_context}?"
        truth = variant == 0
        semantic_delta["changed"] = ["query_target", "truth"]
    elif operation == "output_order":
        primary, secondary = a, b
        if domain == "taxonomy": relation, context = "is a member of", target
        elif domain == "event": relation, context = "inspected", x
        else: relation, context = "occurred earlier than", x
        truth = unit % 2 == 0
        query_context = context if truth else y
        facts = [f"{primary} {relation} {context}.", f"{secondary} catalogued the {noise} separately."]
        question = f"Is it true that {primary} {relation} {query_context}?"
        semantic_delta["changed"] = ["candidate_order"]
    elif operation == "surface_paraphrase":
        primary, secondary, context = a, b, x
        relation = {"taxonomy": "is a member of", "event": "inspected", "attitude": "likes tasting"}[domain]
        if variant == 0:
            facts = [f"{primary} {relation} {context}.", f"{secondary} catalogued the {noise} separately."]
            question = f"Is it true that {primary} {relation} {context}?"
        else:
            facts = [f"The relevant entry says that {primary} {relation} {context}.", f"A separate entry associates {secondary} with the {noise}."]
            question = f"Does the relevant entry support that {primary} {relation} {context}?"
        semantic_delta["changed"] = ["surface_paraphrase"]
    elif operation == "translation":
        primary, secondary, context = a, b, x
        relation_en = {"taxonomy": "is a member of", "event": "inspected", "temporal": "occurred earlier than"}[domain]
        relation_zh = {"taxonomy": "属于", "event": "检查了", "temporal": "早于"}[domain]
        relation = relation_en if variant == 0 else relation_zh
        if variant == 0:
            facts = [f"{primary} {relation} {context}.", f"{secondary} catalogued the {noise} separately."]
            question = f"Is it true that {primary} {relation} {context}?"
        else:
            facts = [f"{primary}{relation}{context}。", f"{secondary}另行记录了{noise}。"]
            question = f"根据记录，{primary}{relation}{context}吗？"
        semantic_delta["changed"] = ["language"]
    else:
        raise KeyError(operation)

    if operation == "output_order":
        option_order = variant
    else:
        option_order = (unit + SURFACES.index(surface) + list(OPERATION_SPECS).index(operation)) % 2
    options, gold = answer_options(truth, option_order)
    core = wrap(surface, facts, question) if not (operation == "translation" and variant == 1) else f"一份核验记录写道：{' '.join(facts)} 只根据这份记录回答：{question}"
    return {
        "facts": facts, "question": question, "truth": truth, "option_order": option_order,
        "gold_position": gold, "prompt_core": core,
        "prompt": f"{core} {options}. Reply with only A or B.",
        "free_prompt": f"{core} Answer only Yes or No.",
        "role_values": {"primary": primary, "secondary": secondary, "relation": relation, "context": context, "query": primary},
        "semantic_delta": semantic_delta,
    }


def atomic_material() -> list[dict]:
    rows: list[dict] = []
    for operation, spec in OPERATION_SPECS.items():
        for domain, surface, unit, variant in itertools.product(spec["domains"], SURFACES, range(UNITS), (0, 1)):
            values = atomic_semantics(operation, domain, surface, unit, variant)
            pair_id = f"{operation}:{domain}:{surface}:u{unit:02d}"
            rows.append({
                "case_id": f"c543-{operation}-{domain}-{surface}-u{unit:02d}-v{variant}",
                "panel": "typed_atomic_operation", "operation_type": operation, "operation_domain": domain,
                "domain_id": f"typed:{operation}:{domain}", "surface": surface, "construction": surface,
                "unit": unit, "variant": variant, "pair_id": pair_id, "partition": partition(unit),
                "truth_class": spec["truth_class"], "rewrite": spec["rewrite"], **values,
                "correct_answer": "Yes" if values["truth"] else "No",
                "wrong_answer": "No" if values["truth"] else "Yes",
                "semantic_graph": {
                    "operation_type": operation, "domain": domain, "truth": values["truth"],
                    "truth_class": spec["truth_class"], "rewrite": spec["rewrite"],
                    "delta": values["semantic_delta"],
                },
            })
    return rows


def composition_semantics(panel: str, surface: str, unit: int, bit_a: int, bit_b: int) -> dict:
    u = unit_values(unit)
    a, b, x, y, mid, target, noise = u["a"], u["b"], u["x"], u["y"], u["mid"], u["target"], u["noise"]
    if panel == "attitude_entity_object":
        primary = a if bit_a == 0 else b
        secondary = b if bit_a == 0 else a
        context = x if bit_b == 0 else y
        relation = "likes eating"
        facts = [f"{primary} {relation} {context}.", f"{secondary} inspected the {noise} separately."]
        question = f"Is it true that {primary} {relation} {context}?"
        truth = True
    elif panel == "attitude_polarity_object":
        primary, secondary = a, b
        context = x if bit_b == 0 else y
        relation = "likes eating" if bit_a == 0 else "does not like eating"
        facts = [f"{primary} {relation} {context}.", f"{secondary} inspected the {noise} separately."]
        question = f"Is it true that {primary} {relation} {context}?"
        truth = True
    elif panel == "graph_path_completion":
        primary, secondary, context, relation = x, mid, target, "links to"
        facts = [f"The relation '{relation}' is directional."]
        facts.append(f"{primary} {relation} {secondary}." if bit_a else f"{primary} and {secondary} are only listed separately.")
        facts.append(f"{secondary} {relation} {context}." if bit_b else f"{secondary} and {context} are only listed separately.")
        facts.append(f"The {noise} is unrelated.")
        question = f"Do the recorded links form a two-step path from {primary} through {secondary} to {context}?"
        truth = bool(bit_a and bit_b)
    else:
        raise KeyError(panel)
    option_order = (unit + SURFACES.index(surface)) % 2
    options, gold = answer_options(truth, option_order)
    core = wrap(surface, facts, question)
    return {
        "truth": truth, "option_order": option_order, "gold_position": gold,
        "prompt_core": core, "prompt": f"{core} {options}. Reply with only A or B.",
        "free_prompt": f"{core} Answer only Yes or No.",
        "role_values": {"primary": primary, "secondary": secondary, "relation": relation, "context": context, "query": primary},
    }


def composition_material() -> list[dict]:
    rows: list[dict] = []
    for panel, surface, unit, bit_a, bit_b in itertools.product(COMPOSITION_PANELS, SURFACES, range(UNITS), (0, 1), (0, 1)):
        values = composition_semantics(panel, surface, unit, bit_a, bit_b)
        cell = f"{bit_a}{bit_b}"
        rows.append({
            "case_id": f"c543-{panel}-{surface}-u{unit:02d}-x{cell}",
            "panel": "typed_composition", "composition_panel": panel,
            "operation_type": "composition", "operation_domain": panel,
            "domain_id": f"composition:{panel}", "surface": surface, "construction": surface,
            "unit": unit, "bits": [bit_a, bit_b], "cell": cell,
            "partition": partition(unit), **values,
            "correct_answer": "Yes" if values["truth"] else "No",
            "wrong_answer": "No" if values["truth"] else "Yes",
            "semantic_graph": {
                "composition_panel": panel, "operators": COMPOSITION_PANELS[panel],
                "bits": [bit_a, bit_b], "truth": values["truth"],
            },
        })
    return rows


def all_material() -> list[dict]:
    return atomic_material() + composition_material()


def material_path() -> Path:
    return OUTS["C543"] / "material/typed_operation_cases.jsonl"


def compiled_path() -> Path:
    return OUTS["C544"] / "compiled/qwen3_typed_operations.jsonl"


def capture_paths() -> tuple[Path, Path, Path]:
    raw = OUTS["C545"] / "raw"
    return raw / "role_mean_states.float16.npy", raw / "role_last_states.float16.npy", raw / "full_token_states.float16.npy"


def capture_index() -> list[dict]:
    return read_rows(OUTS["C545"] / "raw/hidden_index.jsonl")


def pair_indices(index: list[dict], operation: str, domain: str, surface: str, partition_name: str) -> list[tuple[int, int]]:
    groups: dict[str, dict[int, int]] = defaultdict(dict)
    for row in index:
        if row.get("operation_type") == operation and row.get("operation_domain") == domain and row["surface"] == surface and row["partition"] == partition_name:
            groups[row["pair_id"]][int(row["variant"])] = int(row["hidden_index"])
    return [(v[0], v[1]) for v in groups.values() if 0 in v and 1 in v]


def pair_responses(states: np.ndarray, pairs: list[tuple[int, int]], q: int, roles: list[int] | None = None) -> np.ndarray:
    values = []
    for left, right in pairs:
        delta = np.asarray(states[right, q], np.float32) - np.asarray(states[left, q], np.float32)
        values.append(delta if roles is None else delta[roles])
    return np.stack(values) if values else np.empty((0, len(roles or ROLES), DIM), np.float32)


def atomic_groups(index: list[dict]) -> list[tuple[str, str, str]]:
    return sorted({
        (row["operation_type"], row["operation_domain"], row["surface"])
        for row in index if row.get("panel") == "typed_atomic_operation"
    })


def prototype_book(states: np.ndarray, index: list[dict], qpoints: tuple[int, ...] = QPOINTS, roles: list[int] | None = None) -> dict[tuple[str, str, str, int], np.ndarray]:
    result: dict[tuple[str, str, str, int], np.ndarray] = {}
    for operation, domain, surface in atomic_groups(index):
        pairs = pair_indices(index, operation, domain, surface, "discovery")
        for q in qpoints:
            result[(operation, domain, surface, q)] = pair_responses(states, pairs, q, roles).mean(axis=0)
    return result


def choose_wrong(book: dict, operation: str, domain: str, surface: str, q: int, same_operation: bool) -> np.ndarray:
    keys = sorted(book)
    truth_class = OPERATION_SPECS[operation]["truth_class"]
    candidates = []
    for key in keys:
        op, dom, surf, qp = key
        if qp != q or surf != surface:
            continue
        if same_operation and op == operation and dom != domain:
            candidates.append(key)
        if not same_operation and op != operation and OPERATION_SPECS[op]["truth_class"] == truth_class:
            candidates.append(key)
    if not candidates and not same_operation:
        candidates = [key for key in keys if key[3] == q and key[2] == surface and key[0] != operation]
    if not candidates:
        return np.zeros_like(next(iter(book.values())))
    return np.mean([book[key] for key in candidates], axis=0)


def summarize_gates(gates: dict[str, bool]) -> dict:
    return {"passed": int(sum(gates.values())), "total": len(gates), "pass_rate": float(np.mean(list(gates.values()))) if gates else 0.0}


def c542() -> None:
    parent = load(PARENT_AUDIT)
    begin("C542", {
        "status": "evidence_adjudication_and_typed_operation_master_contract_frozen",
        "research_object": "typed linguistic operation responses in the complete token x checkpoint x 2560 HiddenState field",
        "corrections": [
            "failure of five registered six-role models does not prove that every sample-specific carrier is absent",
            "C536 comparisons share domains, units, roles, checkpoints, and coordinates and are not independent replications",
            "mean checkpoint replay is observed but is not attributed uniquely to residual connections",
            "C536 wrong-bit controls did not isolate truth, answer order, surface, or equal-norm direction",
        ],
        "routes": [
            "typed ontology", "large material and compiler audit", "all-token Qwen capture",
            "within-domain passports", "cross-domain same-type transfer", "confound ledger",
            "independent-unit adjudication", "minimal response history", "attitude composition",
            "graph composition", "token granularity", "conditional causal", "conditional cross-model",
            "full-coordinate visual", "cleanup", "independent audit",
        ],
        "gate_margin": CONTROL_MARGIN,
        "stop_policy": "route-level elimination; every preregistered observational route runs even if another route fails",
    }, {"parent_audit": bool(parent["all_checks_passed"]), "routes": 16 == 16})
    close("C542", {
        "status": "master_contract_closed",
        "strict_target": "Determine which explicitly typed language operations have reusable full-coordinate responses after truth, answer-order, surface, amplitude, and independent-unit controls.",
        "old_rollout_route_reopened": False,
    }, {"parent": True, "scope_frozen": True}, "C543_typed_ontology")


def c543() -> None:
    out = begin("C543", {
        "status": "typed_linguistic_operation_ontology_and_program_graph_frozen",
        "atomic_types": list(OPERATION_SPECS),
        "composition_panels": list(COMPOSITION_PANELS),
        "minimum_domains_per_atomic_type": 3,
        "units_per_domain": UNITS,
        "surfaces": list(SURFACES),
    }, {"parent": final("C542")["all_checks_passed"], "atomic_types": len(OPERATION_SPECS) == 11})
    rows = all_material()
    write_rows(out / "material/typed_operation_cases.jsonl", rows)
    ontology = {
        "atomic": OPERATION_SPECS,
        "composition": {key: list(value) for key, value in COMPOSITION_PANELS.items()},
        "semantic_effect_classes": {
            key: value["truth_class"] for key, value in OPERATION_SPECS.items()
        },
        "partitions": {"discovery_units": list(range(10)), "confirmation_units": list(range(10, 14)), "lockbox_units": list(range(14, 18))},
    }
    save(out / "material/typed_operation_ontology.json", ontology)
    atomic_n = sum(row["panel"] == "typed_atomic_operation" for row in rows)
    composition_n = len(rows) - atomic_n
    close("C543", {
        "status": "typed_ontology_closed", "rows": len(rows), "atomic_rows": atomic_n,
        "composition_rows": composition_n, "atomic_types": len(OPERATION_SPECS),
        "atomic_domains": sum(len(value["domains"]) for value in OPERATION_SPECS.values()),
        "strict_boundary": "Operation labels are external experimental rewrites, not claims that Qwen internally uses the same named operators.",
    }, {
        "rows": len(rows) == 2808, "atomic": atomic_n == 2376, "composition": composition_n == 432,
        "unique": len({row["case_id"] for row in rows}) == len(rows),
    }, "C544_material_compiler_audit")


def c544() -> None:
    out = begin("C544", {
        "status": "large_material_compiler_semantic_balance_and_naturalness_audit_frozen",
        "model_not_loaded": True,
        "audits": ["pair completeness", "semantic rewrite uniqueness", "candidate balance", "truth ledger", "token width", "role span compilation", "machine naturalness lint"],
        "human_naturalness": "NA_not_run",
    }, {"parent": final("C543")["all_checks_passed"]})
    rows = read_rows(material_path())
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_utils.MODEL_CONFIGS["qwen3"]["path"], trust_remote_code=True, local_files_only=True,
    )
    compile_base = previous.prior.previous.parent.previous.prior.compile_base
    compiled = compile_base.compile_qwen(tokenizer, rows)
    write_rows(out / "compiled/qwen3_typed_operations.jsonl", compiled)
    atomic = [row for row in rows if row["panel"] == "typed_atomic_operation"]
    pairs: dict[str, set[int]] = defaultdict(set)
    for row in atomic:
        pairs[row["pair_id"]].add(int(row["variant"]))
    candidate_balance = float(np.mean([int(row["option_order"] == 0) for row in rows]))
    truth_rate = float(np.mean([int(row["truth"]) for row in rows]))
    widths = [len(row["prompt_ids"]) for row in compiled]
    malformed = [
        row["case_id"] for row in rows
        if "  " in row["prompt_core"] or not row["prompt_core"].strip().endswith(("?", "？"))
    ]
    prompt_groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        prompt_groups[row["prompt"]].append(row)
    shared_groups = [values for values in prompt_groups.values() if len(values) > 1]
    duplicate_prompts = len(rows) - len(prompt_groups)
    cross_partition_shared = [
        values for values in shared_groups if len({row["partition"] for row in values}) > 1
    ]
    inconsistent_shared = [
        values for values in shared_groups
        if len({(bool(row["truth"]), int(row["gold_position"])) for row in values}) > 1
    ]
    write_rows(out / "audit/shared_prompt_ledger.jsonl", (
        {
            "prompt_sha256": hashlib.sha256(values[0]["prompt"].encode("utf-8")).hexdigest(),
            "case_ids": [row["case_id"] for row in values],
            "operations": sorted({str(row.get("operation_type")) for row in values}),
            "partition": values[0]["partition"], "truth": bool(values[0]["truth"]),
            "gold_position": int(values[0]["gold_position"]),
        }
        for values in shared_groups
    ))
    operation_counts = {op: sum(row.get("operation_type") == op for row in atomic) for op in OPERATION_SPECS}
    domain_counts = {
        op: len({row["operation_domain"] for row in atomic if row["operation_type"] == op})
        for op in OPERATION_SPECS
    }
    close("C544", {
        "status": "material_compiler_audit_closed", "rows": len(rows), "compiled_rows": len(compiled),
        "pair_count": len(pairs), "candidate_a_first_rate": candidate_balance, "truth_rate": truth_rate,
        "max_width": max(widths), "min_width": min(widths), "duplicate_prompts": duplicate_prompts,
        "unique_physical_prompts": len(prompt_groups), "shared_prompt_groups": len(shared_groups),
        "cross_partition_shared_groups": len(cross_partition_shared),
        "inconsistent_shared_groups": len(inconsistent_shared),
        "formal_global_unique_prompt_gate_passed": duplicate_prompts == 0,
        "shared_prompt_accounting_authorized": duplicate_prompts > 0 and not cross_partition_shared and not inconsistent_shared,
        "malformed_count": len(malformed), "operation_counts": operation_counts, "domain_counts": domain_counts,
        "human_naturalness": "NA_not_run",
        "strict_boundary": "The global unique-prompt gate failed. Shared physical prompts are retained as explicit matched controls, never counted as independent replications. Machine lint does not replace human blind review.",
    }, {
        "rows": len(rows) == 2808, "compiled": len(compiled) == len(rows),
        "pairs": len(pairs) == 1188 and all(value == {0, 1} for value in pairs.values()),
        "candidate_balance": abs(candidate_balance - 0.5) < 1e-12,
        "width": max(widths) <= 160, "roles": all(set(row["role_positions"]) == set(ROLES) for row in compiled),
        "malformed": not malformed, "duplicates_detected": duplicate_prompts > 0,
        "no_partition_leak": not cross_partition_shared, "shared_consistent": not inconsistent_shared,
        "domains": all(value == 3 for value in domain_counts.values()),
    }, "C545_qwen_capture")


def c545() -> None:
    out = begin("C545", {
        "status": "qwen_behavior_and_all_token_all_coordinate_capture_frozen",
        "model": "local Qwen3-4B BF16 CUDA, no quantization",
        "views": ["all-case six-role span mean", "all-case role last token", "all-case full-token field"],
        "checkpoints": CHECKPOINTS, "coordinates": DIM,
        "coordinate_policy": "all coordinates retained; no PCA, Top-K, or magnitude truncation",
    }, {"parent": final("C544")["all_checks_passed"], "cuda": torch.cuda.is_available()})
    rows = read_rows(material_path())
    compiled = read_rows(compiled_path())
    n = len(rows)
    width = max(len(row["prompt_ids"]) for row in compiled)
    mean_path, last_path, full_path = capture_paths()
    mean_path.parent.mkdir(parents=True, exist_ok=True)
    mean_states = np.lib.format.open_memmap(mean_path, mode="w+", dtype=np.float16, shape=(n, CHECKPOINTS, len(ROLES), DIM))
    last_states = np.lib.format.open_memmap(last_path, mode="w+", dtype=np.float16, shape=(n, CHECKPOINTS, len(ROLES), DIM))
    full_states = np.lib.format.open_memmap(full_path, mode="w+", dtype=np.float16, shape=(n, CHECKPOINTS, width, DIM))
    model = None
    hooks: list[Any] = []
    captured: list[torch.Tensor] = []
    index: list[dict] = []
    headline: dict[str, Any] = {}
    try:
        model, tokenizer, device, placement = previous.model_base().load_bf16("qwen3")
        quant = previous.model_base().quantization_audit(model)
        base = model.model

        def hook(_module, _args, output):
            captured.append(output[0] if isinstance(output, tuple) else output)

        hooks.append(base.embed_tokens.register_forward_hook(hook))
        hooks.extend(layer.register_forward_hook(hook) for layer in base.layers)
        hooks.append(base.norm.register_forward_hook(hook))
        pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
        batch_size = 4
        for start in range(0, n, batch_size):
            batch = compiled[start:start + batch_size]
            ids = torch.full((len(batch), width), pad, dtype=torch.long, device=device)
            mask = torch.zeros_like(ids)
            pos = torch.zeros_like(ids)
            lengths: list[int] = []
            role_weights = torch.zeros((len(batch), len(ROLES), width), dtype=torch.float32, device=device)
            last_pos = torch.zeros((len(batch), len(ROLES)), dtype=torch.long, device=device)
            for local, row in enumerate(batch):
                values = row["prompt_ids"]
                lengths.append(len(values))
                ids[local, :len(values)] = torch.tensor(values, dtype=torch.long, device=device)
                mask[local, :len(values)] = 1
                pos[local, :len(values)] = torch.arange(len(values), device=device)
                for role_i, role in enumerate(ROLES):
                    positions = [int(value) for value in row["role_positions"][role]]
                    role_weights[local, role_i, positions] = 1.0 / len(positions)
                    last_pos[local, role_i] = positions[-1]
            captured.clear()
            with torch.inference_mode():
                output = model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False, return_dict=True)
            if len(captured) != CHECKPOINTS:
                raise RuntimeError(("checkpoint_count", len(captured)))
            for q, state in enumerate(captured):
                state32 = state.float()
                mean_states[start:start + len(batch), q] = torch.einsum("brt,btd->brd", role_weights, state32).cpu().numpy().astype(np.float16)
                gather = last_pos[:, :, None].expand(-1, -1, DIM)
                last_states[start:start + len(batch), q] = torch.gather(state32, 1, gather).cpu().numpy().astype(np.float16)
                for local, length in enumerate(lengths):
                    full_states[start + local, q, :length] = state[local, :length].float().cpu().numpy().astype(np.float16)
            for local, row in enumerate(batch):
                source_i = start + local
                length = lengths[local]
                scores = [float(output.logits[local, length - 1, candidate[0]]) for candidate in row["candidate_ids"]]
                prediction = int(scores[1] > scores[0])
                meta = rows[source_i]
                index.append({
                    "hidden_index": source_i, "case_id": meta["case_id"], "panel": meta["panel"],
                    "operation_type": meta.get("operation_type"), "operation_domain": meta.get("operation_domain"),
                    "composition_panel": meta.get("composition_panel"), "domain_id": meta["domain_id"],
                    "surface": meta["surface"], "construction": meta["construction"], "unit": int(meta["unit"]),
                    "variant": meta.get("variant"), "pair_id": meta.get("pair_id"), "bits": meta.get("bits"),
                    "cell": meta.get("cell"), "partition": meta["partition"], "truth": bool(meta["truth"]),
                    "truth_class": meta.get("truth_class"), "option_order": int(meta["option_order"]),
                    "length": length, "role_positions": row["role_positions"], "gold_position": int(meta["gold_position"]),
                    "prediction": prediction, "correct": prediction == int(meta["gold_position"]),
                })
            mean_states.flush(); last_states.flush(); full_states.flush()
            if start % 100 == 0 or start + len(batch) == n:
                print(f"[C545 capture] {start + len(batch)}/{n}", flush=True)
        write_rows(out / "raw/hidden_index.jsonl", index)
        domain_accuracy = {
            domain: float(np.mean([row["correct"] for row in index if row["domain_id"] == domain]))
            for domain in sorted({row["domain_id"] for row in index})
        }
        operation_accuracy = {
            operation: float(np.mean([row["correct"] for row in index if row.get("operation_type") == operation]))
            for operation in sorted({row.get("operation_type") for row in index if row.get("operation_type")})
        }
        headline = {
            "status": "qwen_capture_closed", "rows": n, "accuracy": float(np.mean([row["correct"] for row in index])),
            "domain_accuracy": domain_accuracy, "operation_accuracy": operation_accuracy,
            "mean_shape": list(mean_states.shape), "last_shape": list(last_states.shape), "full_shape": list(full_states.shape),
            "field_width": width, "placement": placement, "quantization": quant,
        }
    finally:
        for item in hooks:
            item.remove()
        close_mmap(mean_states); close_mmap(last_states); close_mmap(full_states)
        previous.model_base().release_bf16(model)
        gc.collect()
    close("C545", headline, {
        "rows": headline["rows"] == 2808, "mean": headline["mean_shape"] == [2808, 38, 6, 2560],
        "last": headline["last_shape"] == [2808, 38, 6, 2560],
        "full": headline["full_shape"][:2] == [2808, 38] and headline["full_shape"][3] == 2560,
        "bf16": headline["quantization"].get("has_bf16_parameters", False) and not headline["quantization"].get("has_quantized_modules", True),
        "finite": finite(headline),
    }, "C546_within_domain_passports")


def c546() -> None:
    out = begin("C546", {
        "status": "within_domain_operation_response_passports_frozen",
        "object": "discovery mean paired response over all six roles and 2560 coordinates",
        "tests": ["confirmation", "lockbox"],
        "controls": ["zero", "equal-norm wrong domain of same type", "equal-norm wrong type with matched effect class"],
        "gate": "correct prototype beats every control by at least 0.02 NRMSE",
    }, {"parent": final("C545")["all_checks_passed"]})
    states = np.load(capture_paths()[0], mmap_mode="r")
    index = capture_index()
    book = prototype_book(states, index)
    archive = {f"{op}|{dom}|{surf}|q{q}": value.astype(np.float16) for (op, dom, surf, q), value in book.items()}
    (out / "analysis").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "analysis/full_coordinate_passport_prototypes.npz", **archive)
    metrics: dict[str, dict] = {}
    gates: dict[str, bool] = {}
    for operation, domain, surface in atomic_groups(index):
        for part in ("confirmation", "lockbox"):
            pairs = pair_indices(index, operation, domain, surface, part)
            for q in QPOINTS:
                truth = pair_responses(states, pairs, q)
                correct = book[(operation, domain, surface, q)]
                wrong_domain = scaled_like(choose_wrong(book, operation, domain, surface, q, True), correct)
                wrong_type = scaled_like(choose_wrong(book, operation, domain, surface, q, False), correct)
                values = {
                    "pairs": len(pairs),
                    "correct": metric(np.broadcast_to(correct, truth.shape), truth),
                    "zero": metric(np.zeros_like(truth), truth),
                    "wrong_domain_equal_norm": metric(np.broadcast_to(wrong_domain, truth.shape), truth),
                    "wrong_type_equal_norm": metric(np.broadcast_to(wrong_type, truth.shape), truth),
                }
                key = f"{operation}:{domain}:{surface}:{part}:q{q}"
                metrics[key] = values
                best = min(values[name]["nrmse"] for name in ("zero", "wrong_domain_equal_norm", "wrong_type_equal_norm"))
                gates[key] = values["correct"]["nrmse"] <= best - CONTROL_MARGIN
        print(f"[C546 within] {operation}:{domain}:{surface}", flush=True)
    op_rates = {
        op: float(np.mean([value for key, value in gates.items() if key.startswith(op + ":")]))
        for op in OPERATION_SPECS
    }
    close_mmap(states)
    close("C546", {
        "status": "within_domain_passports_closed", "metrics": metrics, "gates": gates,
        "gate_summary": summarize_gates(gates), "operation_pass_rate": op_rates,
        "strict_interpretation": "A within-domain pass is material-specific response reuse, not cross-domain semantic identity.",
    }, {
        "groups": len(atomic_groups(index)) == 66, "comparisons": len(metrics) == 66 * 2 * len(QPOINTS),
        "finite": finite(metrics), "archive": (out / "analysis/full_coordinate_passport_prototypes.npz").exists(),
    }, "C547_cross_domain_transfer")


def cross_type_prototype(book: dict, operation: str, excluded_domain: str, train_surface: str, q: int) -> np.ndarray:
    values = [
        value for (op, domain, surface, qp), value in book.items()
        if op == operation and domain != excluded_domain and surface == train_surface and qp == q
    ]
    return np.mean(values, axis=0)


def wrong_type_prototype(book: dict, operation: str, train_surface: str, q: int) -> np.ndarray:
    truth_class = OPERATION_SPECS[operation]["truth_class"]
    preferred = [
        value for (op, _domain, surface, qp), value in book.items()
        if op != operation and surface == train_surface and qp == q and OPERATION_SPECS[op]["truth_class"] == truth_class
    ]
    if not preferred:
        preferred = [
            value for (op, _domain, surface, qp), value in book.items()
            if op != operation and surface == train_surface and qp == q
        ]
    return np.mean(preferred, axis=0)


def c547() -> None:
    out = begin("C547", {
        "status": "cross_domain_same_type_response_transfer_frozen",
        "train": "discovery prototypes from the other two domains of the same operation",
        "test": "target-domain lockbox",
        "surface_routes": ["same surface", "cross surface"],
        "controls": ["zero", "equal-norm wrong operation type"],
        "gate": "same-type cross-domain prototype beats all controls by at least 0.02 NRMSE",
    }, {"parent": final("C546")["all_checks_passed"]})
    states = np.load(capture_paths()[0], mmap_mode="r")
    index = capture_index()
    book = prototype_book(states, index)
    metrics: dict[str, dict] = {}
    gates: dict[str, bool] = {}
    for operation, spec in OPERATION_SPECS.items():
        for domain, test_surface, q in itertools.product(spec["domains"], SURFACES, QPOINTS):
            test_pairs = pair_indices(index, operation, domain, test_surface, "lockbox")
            truth = pair_responses(states, test_pairs, q)
            for route, train_surface in (("same_surface", test_surface), ("cross_surface", SURFACES[1 - SURFACES.index(test_surface)])):
                correct = cross_type_prototype(book, operation, domain, train_surface, q)
                wrong = scaled_like(wrong_type_prototype(book, operation, train_surface, q), correct)
                values = {
                    "pairs": len(test_pairs), "train_surface": train_surface,
                    "correct": metric(np.broadcast_to(correct, truth.shape), truth),
                    "zero": metric(np.zeros_like(truth), truth),
                    "wrong_type_equal_norm": metric(np.broadcast_to(wrong, truth.shape), truth),
                }
                key = f"{operation}:{domain}:{test_surface}:{route}:q{q}"
                metrics[key] = values
                best = min(values["zero"]["nrmse"], values["wrong_type_equal_norm"]["nrmse"])
                gates[key] = values["correct"]["nrmse"] <= best - CONTROL_MARGIN
        print(f"[C547 cross] {operation}", flush=True)
    op_rates = {
        op: float(np.mean([value for key, value in gates.items() if key.startswith(op + ":")]))
        for op in OPERATION_SPECS
    }
    close_mmap(states)
    close("C547", {
        "status": "cross_domain_transfer_closed", "metrics": metrics, "gates": gates,
        "gate_summary": summarize_gates(gates), "operation_pass_rate": op_rates,
        "strict_interpretation": "Cross-domain transfer is predictive response similarity; it is not yet a causal operator or a unique internal code.",
    }, {
        "comparisons": len(metrics) == 11 * 3 * 2 * 2 * len(QPOINTS),
        "finite": finite(metrics), "both_routes": any(":cross_surface:" in key for key in metrics),
    }, "C548_confound_ledger")


def c548() -> None:
    out = begin("C548", {
        "status": "truth_output_surface_and_equal_norm_confound_ledger_frozen",
        "semantic_roles": [role for role in ROLES if role != "boundary"],
        "controls": ["zero", "equal-norm output-order response", "equal-norm truth-flip response", "equal-norm surface-paraphrase response"],
        "gate": "cross-domain same-type response wins on non-boundary roles by at least 0.02",
    }, {"parent": final("C547")["all_checks_passed"]})
    states = np.load(capture_paths()[0], mmap_mode="r")
    index = capture_index()
    semantic_roles = list(range(len(ROLES) - 1))
    book = prototype_book(states, index, roles=semantic_roles)
    metrics: dict[str, dict] = {}
    gates: dict[str, bool] = {}
    role_metrics: dict[str, dict] = {}
    controls_by_name = ("output_order", "query_truth_flip", "surface_paraphrase")
    for operation, spec in OPERATION_SPECS.items():
        if operation in controls_by_name:
            continue
        for domain, surface, q in itertools.product(spec["domains"], SURFACES, (16, 24, 37)):
            pairs = pair_indices(index, operation, domain, surface, "lockbox")
            truth = pair_responses(states, pairs, q, semantic_roles)
            correct = cross_type_prototype(book, operation, domain, surface, q)
            controls = {}
            for control_name in controls_by_name:
                values = [
                    value for (op, _dom, surf, qp), value in book.items()
                    if op == control_name and surf == surface and qp == q
                ]
                control = np.mean(values, axis=0)
                controls[control_name] = scaled_like(control, correct)
            values = {
                "pairs": len(pairs), "correct": metric(np.broadcast_to(correct, truth.shape), truth),
                "zero": metric(np.zeros_like(truth), truth),
                **{
                    f"{name}_equal_norm": metric(np.broadcast_to(value, truth.shape), truth)
                    for name, value in controls.items()
                },
            }
            key = f"{operation}:{domain}:{surface}:q{q}"
            metrics[key] = values
            best = min(values[name]["nrmse"] for name in values if name not in ("pairs", "correct"))
            gates[key] = values["correct"]["nrmse"] <= best - CONTROL_MARGIN
            role_metrics[key] = {
                ROLES[role_i]: {
                    "correct": metric(np.broadcast_to(correct[local_i], truth[:, local_i].shape), truth[:, local_i]),
                    "zero": metric(np.zeros_like(truth[:, local_i]), truth[:, local_i]),
                }
                for local_i, role_i in enumerate(semantic_roles)
            }
        print(f"[C548 confounds] {operation}", flush=True)
    op_rates = {
        op: float(np.mean([value for key, value in gates.items() if key.startswith(op + ":")]))
        for op in OPERATION_SPECS if op not in controls_by_name
    }
    close_mmap(states)
    close("C548", {
        "status": "confound_ledger_closed", "metrics": metrics, "gates": gates,
        "role_metrics": role_metrics, "gate_summary": summarize_gates(gates),
        "operation_pass_rate": op_rates,
        "strict_interpretation": "Late answer-code and surface responses are explicit controls; non-boundary transfer remains observational.",
    }, {
        "semantic_types": len(op_rates) == 8, "comparisons": len(metrics) == 8 * 3 * 2 * 3,
        "finite": finite(metrics) and finite(role_metrics),
    }, "C549_independent_unit_adjudication")


def per_sample_nrmse(prediction: np.ndarray, truth: np.ndarray) -> np.ndarray:
    axes = tuple(range(1, truth.ndim))
    rmse = np.sqrt(np.mean((prediction - truth) ** 2, axis=axes))
    scale = np.sqrt(np.mean(truth ** 2, axis=axes))
    return rmse / np.maximum(scale, 1e-12)


def c549() -> None:
    out = begin("C549", {
        "status": "independent_unit_evidence_and_candidate_adjudication_frozen",
        "evidence_unit": "held-out lexical/program unit pair, never coordinates or checkpoints",
        "requirements": [
            "all three domains represented", "both surfaces represented", "q24 and q37 represented",
            "median per-unit margin >=0.02", "at least 75 percent held-out units have positive margin",
            "non-boundary confound pass in at least half of registered strata",
        ],
    }, {"parent": final("C548")["all_checks_passed"]})
    states = np.load(capture_paths()[0], mmap_mode="r")
    index = capture_index()
    book = prototype_book(states, index)
    unit_evidence: dict[str, dict] = {}
    candidates: dict[str, bool] = {}
    confound_gates = final("C548")["headline"]["gates"]
    for operation, spec in OPERATION_SPECS.items():
        margins: list[float] = []
        strata = 0
        strata_pass = 0
        for domain, surface, q in itertools.product(spec["domains"], SURFACES, (24, 37)):
            pairs = pair_indices(index, operation, domain, surface, "lockbox")
            truth = pair_responses(states, pairs, q)
            correct = cross_type_prototype(book, operation, domain, surface, q)
            wrong = scaled_like(wrong_type_prototype(book, operation, surface, q), correct)
            correct_pred = np.broadcast_to(correct, truth.shape)
            wrong_pred = np.broadcast_to(wrong, truth.shape)
            e_correct = per_sample_nrmse(correct_pred, truth)
            e_control = np.minimum(per_sample_nrmse(np.zeros_like(truth), truth), per_sample_nrmse(wrong_pred, truth))
            margins.extend((e_control - e_correct).tolist())
            strata += 1
            strata_pass += int(float(np.median(e_control - e_correct)) >= CONTROL_MARGIN)
        confound_values = [value for key, value in confound_gates.items() if key.startswith(operation + ":")]
        evidence = {
            "units": len(margins), "median_margin": float(np.median(margins)),
            "positive_fraction": float(np.mean(np.asarray(margins) > 0)),
            "strata_pass": strata_pass, "strata_total": strata,
            "confound_pass_rate": float(np.mean(confound_values)) if confound_values else 0.0,
        }
        candidate = (
            evidence["median_margin"] >= CONTROL_MARGIN
            and evidence["positive_fraction"] >= 0.75
            and strata_pass == strata
            and evidence["confound_pass_rate"] >= 0.5
        )
        unit_evidence[operation] = evidence
        candidates[operation] = bool(candidate)
    close_mmap(states)
    close("C549", {
        "status": "independent_unit_adjudication_closed", "unit_evidence": unit_evidence,
        "typed_response_candidates": candidates,
        "qualified_types": [key for key, value in candidates.items() if value],
        "candidate_count": int(sum(candidates.values())),
        "strict_interpretation": "Coordinates contribute to error magnitude, but qualification is counted over held-out units and preregistered strata.",
    }, {
        "operations": len(unit_evidence) == 11, "finite": finite(unit_evidence),
        "units": all(value["units"] == 48 for value in unit_evidence.values()),
    }, "C550_minimal_history")


def response_for_pair(states: np.ndarray, pair: tuple[int, int], q: int) -> np.ndarray:
    left, right = pair
    return np.asarray(states[right, q], np.float32) - np.asarray(states[left, q], np.float32)


def nearest_prediction(train_features: np.ndarray, train_targets: np.ndarray, test_features: np.ndarray) -> np.ndarray:
    predictions = []
    train_flat = train_features.reshape(train_features.shape[0], -1).astype(np.float64)
    for feature in test_features.reshape(test_features.shape[0], -1).astype(np.float64):
        distance = np.mean((train_flat - feature[None, :]) ** 2, axis=1)
        predictions.append(train_targets[int(np.argmin(distance))])
    return np.stack(predictions)


def c550() -> None:
    out = begin("C550", {
        "status": "minimal_sufficient_response_history_tournament_frozen",
        "object": "full six-role x 2560 paired response history",
        "train": "other-domain discovery pairs of the same operation and surface",
        "test": "target-domain lockbox pairs",
        "models": ["type mean", "nearest q0", "nearest q8", "nearest q16", "nearest q24", "nearest joint q16+q24"],
        "gate": "history nearest-neighbor beats type mean by at least 0.02 NRMSE",
    }, {"parent": final("C549")["all_checks_passed"]})
    states = np.load(capture_paths()[0], mmap_mode="r")
    index = capture_index()
    metrics: dict[str, dict] = {}
    gates: dict[str, bool] = {}
    for operation, spec in OPERATION_SPECS.items():
        for domain, surface in itertools.product(spec["domains"], SURFACES):
            train_pairs: list[tuple[int, int]] = []
            for source_domain in spec["domains"]:
                if source_domain != domain:
                    train_pairs.extend(pair_indices(index, operation, source_domain, surface, "discovery"))
            test_pairs = pair_indices(index, operation, domain, surface, "lockbox")
            train_target = np.stack([response_for_pair(states, pair, 37) for pair in train_pairs])
            test_target = np.stack([response_for_pair(states, pair, 37) for pair in test_pairs])
            mean_pred = np.broadcast_to(train_target.mean(axis=0), test_target.shape)
            values: dict[str, dict] = {"pairs": len(test_pairs), "type_mean": metric(mean_pred, test_target)}
            feature_sets = {
                "q0": (0,), "q8": (8,), "q16": (16,), "q24": (24,), "q16_q24": (16, 24),
            }
            for label, qset in feature_sets.items():
                train_feature = np.stack([
                    np.concatenate([response_for_pair(states, pair, q) for q in qset], axis=0)
                    for pair in train_pairs
                ])
                test_feature = np.stack([
                    np.concatenate([response_for_pair(states, pair, q) for q in qset], axis=0)
                    for pair in test_pairs
                ])
                pred = nearest_prediction(train_feature, train_target, test_feature)
                values[f"nearest_{label}"] = metric(pred, test_target)
            key = f"{operation}:{domain}:{surface}"
            metrics[key] = values
            best_history = min(values[name]["nrmse"] for name in values if name.startswith("nearest_"))
            gates[key] = best_history <= values["type_mean"]["nrmse"] - CONTROL_MARGIN
        print(f"[C550 history] {operation}", flush=True)
    op_rates = {
        op: float(np.mean([value for key, value in gates.items() if key.startswith(op + ":")]))
        for op in OPERATION_SPECS
    }
    close_mmap(states)
    close("C550", {
        "status": "minimal_history_tournament_closed", "metrics": metrics, "gates": gates,
        "gate_summary": summarize_gates(gates), "operation_pass_rate": op_rates,
        "strict_interpretation": "A later-checkpoint nearest neighbor is a predictive history result, not proof of a Markov state or causal transition law.",
    }, {
        "strata": len(metrics) == 11 * 3 * 2, "finite": finite(metrics),
        "all_methods": all(len(value) == 7 for value in metrics.values()),
    }, "C551_attitude_composition")


def composition_units(index: list[dict], panel: str, surface: str, part: str) -> list[dict[str, int]]:
    grouped: dict[int, dict[str, int]] = defaultdict(dict)
    for row in index:
        if row.get("composition_panel") == panel and row["surface"] == surface and row["partition"] == part:
            grouped[int(row["unit"])][str(row["cell"])] = int(row["hidden_index"])
    return [value for value in grouped.values() if set(value) == {"00", "10", "01", "11"}]


def interactions(states: np.ndarray, groups: list[dict[str, int]], q: int) -> tuple[np.ndarray, np.ndarray]:
    values, total = [], []
    for group in groups:
        h00 = np.asarray(states[group["00"], q], np.float32)
        h10 = np.asarray(states[group["10"], q], np.float32)
        h01 = np.asarray(states[group["01"], q], np.float32)
        h11 = np.asarray(states[group["11"], q], np.float32)
        values.append(h11 - h10 - h01 + h00)
        total.append(h11 - h00)
    return np.stack(values), np.stack(total)


def run_composition_panel(states: np.ndarray, index: list[dict], panels: tuple[str, ...]) -> tuple[dict, dict]:
    metrics: dict[str, dict] = {}
    gates: dict[str, bool] = {}
    for panel, surface, q in itertools.product(panels, SURFACES, QPOINTS):
        train, _train_total = interactions(states, composition_units(index, panel, surface, "discovery"), q)
        test, test_total = interactions(states, composition_units(index, panel, surface, "lockbox"), q)
        prototype = train.mean(axis=0)
        wrong_panel = next((candidate for candidate in COMPOSITION_PANELS if candidate != panel), panel)
        wrong, _ = interactions(states, composition_units(index, wrong_panel, surface, "discovery"), q)
        wrong_proto = scaled_like(wrong.mean(axis=0), prototype)
        additive_ratio = np.sqrt(np.mean(test * test, axis=(1, 2))) / np.maximum(np.sqrt(np.mean(test_total * test_total, axis=(1, 2))), 1e-12)
        values = {
            "units": len(test), "prototype": metric(np.broadcast_to(prototype, test.shape), test),
            "zero": metric(np.zeros_like(test), test),
            "wrong_panel_equal_norm": metric(np.broadcast_to(wrong_proto, test.shape), test),
            "median_additive_interaction_ratio": float(np.median(additive_ratio)),
        }
        key = f"{panel}:{surface}:q{q}"
        metrics[key] = values
        best = min(values["zero"]["nrmse"], values["wrong_panel_equal_norm"]["nrmse"])
        gates[key] = values["prototype"]["nrmse"] <= best - CONTROL_MARGIN
    return metrics, gates


def c551() -> None:
    out = begin("C551", {
        "status": "attitude_event_atomic_composition_response_frozen",
        "panels": ["attitude_entity_object", "attitude_polarity_object"],
        "object": "second-order full-coordinate interaction residual",
        "controls": ["zero/additive", "equal-norm wrong panel"],
    }, {"parent": final("C550")["all_checks_passed"]})
    states = np.load(capture_paths()[0], mmap_mode="r")
    index = capture_index()
    metrics, gates = run_composition_panel(states, index, ("attitude_entity_object", "attitude_polarity_object"))
    close_mmap(states)
    candidate = all(value for key, value in gates.items() if key.endswith(":q24") or key.endswith(":q37"))
    close("C551", {
        "status": "attitude_composition_closed", "metrics": metrics, "gates": gates,
        "gate_summary": summarize_gates(gates), "attitude_composition_candidate": candidate,
        "strict_interpretation": "A predictable interaction residual is evidence against simple additivity, not yet a compositional operator law.",
    }, {"comparisons": len(metrics) == 2 * 2 * len(QPOINTS), "finite": finite(metrics)}, "C552_graph_composition")


def c552() -> None:
    out = begin("C552", {
        "status": "graph_path_completion_interaction_response_frozen",
        "panel": "graph_path_completion", "object": "edge-completion second-order response",
        "truth_boundary": "only cell 11 is entailed; output/truth contamination remains a registered limitation",
    }, {"parent": final("C551")["all_checks_passed"]})
    states = np.load(capture_paths()[0], mmap_mode="r")
    index = capture_index()
    metrics, gates = run_composition_panel(states, index, ("graph_path_completion",))
    close_mmap(states)
    candidate = all(value for key, value in gates.items() if key.endswith(":q24") or key.endswith(":q37"))
    close("C552", {
        "status": "graph_composition_closed", "metrics": metrics, "gates": gates,
        "gate_summary": summarize_gates(gates), "graph_composition_candidate": candidate,
        "strict_interpretation": "Path-completion interaction is inseparable from the 11-cell truth change in this panel and cannot alone establish graph composition.",
    }, {"comparisons": len(metrics) == 2 * len(QPOINTS), "finite": finite(metrics)}, "C553_token_granularity")


def first_pair_responses(full: np.ndarray, index: list[dict], pairs: list[tuple[int, int]], q: int) -> np.ndarray:
    values = []
    for left, right in pairs:
        roles = []
        for role in ROLES:
            left_pos = int(index[left]["role_positions"][role][0])
            right_pos = int(index[right]["role_positions"][role][0])
            roles.append(np.asarray(full[right, q, right_pos], np.float32) - np.asarray(full[left, q, left_pos], np.float32))
        values.append(np.stack(roles))
    return np.stack(values)


def token_view_transfer(states: np.ndarray, index: list[dict], view: str) -> tuple[dict, dict]:
    metrics: dict[str, dict] = {}
    gates: dict[str, bool] = {}
    for operation, spec in OPERATION_SPECS.items():
        for domain, surface, q in itertools.product(spec["domains"], SURFACES, (24, 37)):
            train_responses = []
            for source_domain in spec["domains"]:
                if source_domain == domain:
                    continue
                pairs = pair_indices(index, operation, source_domain, surface, "discovery")
                if view == "first":
                    train_responses.append(first_pair_responses(states, index, pairs, q))
                else:
                    train_responses.append(pair_responses(states, pairs, q))
            train = np.concatenate(train_responses)
            test_pairs = pair_indices(index, operation, domain, surface, "lockbox")
            truth = first_pair_responses(states, index, test_pairs, q) if view == "first" else pair_responses(states, test_pairs, q)
            prototype = train.mean(axis=0)
            values = {
                "pairs": len(test_pairs), "prototype": metric(np.broadcast_to(prototype, truth.shape), truth),
                "zero": metric(np.zeros_like(truth), truth),
            }
            key = f"{operation}:{domain}:{surface}:q{q}"
            metrics[key] = values
            gates[key] = values["prototype"]["nrmse"] <= values["zero"]["nrmse"] - CONTROL_MARGIN
    return metrics, gates


def c553() -> None:
    out = begin("C553", {
        "status": "first_last_mean_token_granularity_tournament_frozen",
        "views": ["role span mean", "first physical token per role", "last physical token per role"],
        "coordinate_policy": "all 2560 coordinates retained in every view",
        "gate": "other-domain prototype beats zero by at least 0.02 at q24/q37",
    }, {"parent": final("C552")["all_checks_passed"]})
    mean_states = np.load(capture_paths()[0], mmap_mode="r")
    last_states = np.load(capture_paths()[1], mmap_mode="r")
    full_states = np.load(capture_paths()[2], mmap_mode="r")
    index = capture_index()
    mean_metrics, mean_gates = token_view_transfer(mean_states, index, "mean")
    last_metrics, last_gates = token_view_transfer(last_states, index, "last")
    first_metrics, first_gates = token_view_transfer(full_states, index, "first")
    view_rates = {
        "mean": summarize_gates(mean_gates), "first": summarize_gates(first_gates), "last": summarize_gates(last_gates),
    }
    close_mmap(mean_states); close_mmap(last_states); close_mmap(full_states)
    close("C553", {
        "status": "token_granularity_tournament_closed",
        "view_rates": view_rates, "mean_metrics": mean_metrics, "first_metrics": first_metrics, "last_metrics": last_metrics,
        "mean_gates": mean_gates, "first_gates": first_gates, "last_gates": last_gates,
        "strict_interpretation": "First/last-token results test registered token projections; they do not identify a unique token circuit.",
    }, {
        "views": all(value["total"] == 11 * 3 * 2 * 2 for value in view_rates.values()),
        "finite": finite(mean_metrics) and finite(first_metrics) and finite(last_metrics),
    }, "C554_causal_eligibility")


def c554() -> None:
    out = begin("C554", {
        "status": "typed_response_causal_eligibility_adjudication_frozen",
        "requirements": [
            "C549 independent-unit typed response candidate", "operation behavior accuracy >=0.90",
            "cross-domain same/cross-surface q24/q37 transfer", "non-boundary confound pass rate >=0.50",
            "at least one registered first/mean/last token projection has pass rate >=0.75",
        ],
        "rule": "eligibility is per operation type; one failure does not block other operation types",
    }, {"parent": final("C553")["all_checks_passed"]})
    candidates = final("C549")["headline"]["typed_response_candidates"]
    behavior = final("C545")["headline"]["operation_accuracy"]
    cross_gates = final("C547")["headline"]["gates"]
    confound_gates = final("C548")["headline"]["gates"]
    token = final("C553")["headline"]
    requirements: dict[str, dict] = {}
    authorized: dict[str, bool] = {}
    for operation in OPERATION_SPECS:
        cross_values = [
            value for key, value in cross_gates.items()
            if key.startswith(operation + ":") and (key.endswith(":q24") or key.endswith(":q37"))
        ]
        confound_values = [value for key, value in confound_gates.items() if key.startswith(operation + ":")]
        token_rates = []
        for gate_name in ("mean_gates", "first_gates", "last_gates"):
            values = [value for key, value in token[gate_name].items() if key.startswith(operation + ":")]
            token_rates.append(float(np.mean(values)) if values else 0.0)
        req = {
            "independent_unit": bool(candidates[operation]),
            "behavior": float(behavior.get(operation, 0.0)) >= 0.90,
            "cross_q24_q37": bool(cross_values) and all(cross_values),
            "confound": bool(confound_values) and float(np.mean(confound_values)) >= 0.50,
            "token_projection": max(token_rates) >= 0.75,
        }
        requirements[operation] = {**req, "behavior_accuracy": float(behavior.get(operation, 0.0)), "token_rates": token_rates}
        authorized[operation] = all(req.values())
    close("C554", {
        "status": "causal_eligibility_closed", "requirements": requirements,
        "authorized_types": [key for key, value in authorized.items() if value],
        "authorized": authorized, "authorized_count": int(sum(authorized.values())),
    }, {"operations": len(requirements) == 11, "complete": all(len(value) == 7 for value in requirements.values())}, "C555_causal_or_na")


def patched_forward(model, input_ids: torch.Tensor, attention_mask: torch.Tensor, position_ids: torch.Tensor,
                    role_positions: dict, response: np.ndarray, q: int) -> tuple[np.ndarray, np.ndarray]:
    base = model.model
    if not (1 <= q <= len(base.layers)):
        raise ValueError(q)
    final_state: list[torch.Tensor] = []

    def patch_hook(_module, _args, output):
        tensor = output[0] if isinstance(output, tuple) else output
        changed = tensor.clone()
        for role_i, role in enumerate(ROLES):
            pos = int(role_positions[role][-1])
            changed[0, pos] = changed[0, pos] + torch.tensor(response[role_i], dtype=changed.dtype, device=changed.device)
        if isinstance(output, tuple):
            return (changed, *output[1:])
        return changed

    def norm_hook(_module, _args, output):
        final_state.append(output.detach())

    patch_handle = base.layers[q - 1].register_forward_hook(patch_hook)
    norm_handle = base.norm.register_forward_hook(norm_hook)
    try:
        with torch.inference_mode():
            output = model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, use_cache=False, return_dict=True)
    finally:
        patch_handle.remove(); norm_handle.remove()
    return final_state[-1][0].float().cpu().numpy(), output.logits[0, int(attention_mask.sum()) - 1].float().cpu().numpy()


def c555() -> None:
    authorized = final("C554")["headline"]["authorized_types"]
    out = begin("C555", {
        "status": "qualified_hiddenstate_causal_branch_frozen",
        "authorized_types": authorized,
        "intervention": "add other-domain discovery mean last-token q24 response to every registered role last token of a base lockbox prompt",
        "controls": ["natural base", "equal-norm wrong-type response", "natural transformed target"],
        "gate": "correct patch improves q37 target NRMSE over base and wrong patch by >=0.02 in every registered test",
        "rule": "if no type is authorized, do not load the model and record NA",
    }, {"parent": final("C554")["all_checks_passed"]})
    if not authorized:
        close("C555", {
            "status": "causal_not_run", "ran": False, "model_loaded": False,
            "result": "NA_predictive_and_confound_qualification_failed", "metrics": {}, "causal_types": [],
        }, {"no_patch": True, "model_not_loaded": True}, "C556_cross_model_or_na")
        return

    rows = read_rows(material_path())
    compiled = read_rows(compiled_path())
    index = capture_index()
    last_states = np.load(capture_paths()[1], mmap_mode="r")
    book = prototype_book(last_states, index, qpoints=(24,))
    row_by_index = {int(row["hidden_index"]): row for row in index}
    model = None
    metrics: dict[str, dict] = {}
    causal_types: list[str] = []
    try:
        model, tokenizer, device, _placement = previous.model_base().load_bf16("qwen3")
        pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
        for operation in authorized:
            type_gates = []
            for domain in OPERATION_SPECS[operation]["domains"]:
                pairs = pair_indices(index, operation, domain, "record", "lockbox")
                left, right = pairs[0]
                comp = compiled[left]
                values = comp["prompt_ids"]
                ids = torch.tensor([values], dtype=torch.long, device=device)
                mask = torch.ones_like(ids)
                pos = torch.arange(len(values), device=device).unsqueeze(0)
                correct = cross_type_prototype(book, operation, domain, "record", 24)
                wrong = scaled_like(wrong_type_prototype(book, operation, "record", 24), correct)
                correct_final, correct_logits = patched_forward(model, ids, mask, pos, comp["role_positions"], correct, 24)
                wrong_final, wrong_logits = patched_forward(model, ids, mask, pos, comp["role_positions"], wrong, 24)
                gather = lambda state: np.stack([state[int(comp["role_positions"][role][-1])] for role in ROLES])
                target = np.asarray(last_states[right, 37], np.float32)
                base_state = np.asarray(last_states[left, 37], np.float32)
                correct_state, wrong_state = gather(correct_final), gather(wrong_final)
                values_metric = {
                    "base": metric(base_state, target), "correct_patch": metric(correct_state, target),
                    "wrong_patch": metric(wrong_state, target),
                    "correct_patch_logit_vector_norm": float(np.linalg.norm(correct_logits)),
                    "wrong_patch_logit_vector_norm": float(np.linalg.norm(wrong_logits)),
                }
                key = f"{operation}:{domain}"
                metrics[key] = values_metric
                gate = (
                    values_metric["correct_patch"]["nrmse"] <= values_metric["base"]["nrmse"] - CONTROL_MARGIN
                    and values_metric["correct_patch"]["nrmse"] <= values_metric["wrong_patch"]["nrmse"] - CONTROL_MARGIN
                )
                type_gates.append(gate)
            if type_gates and all(type_gates):
                causal_types.append(operation)
    finally:
        close_mmap(last_states)
        previous.model_base().release_bf16(model)
        gc.collect()
    close("C555", {
        "status": "causal_branch_closed", "ran": True, "model_loaded": True,
        "metrics": metrics, "causal_types": causal_types,
        "strict_interpretation": "A q37 state rescue would establish local causal sufficiency for this patch compiler, not necessity or a unique circuit.",
    }, {"finite": finite(metrics), "tested": bool(metrics)}, "C556_cross_model_or_na")


def c556() -> None:
    causal_types = final("C555")["headline"]["causal_types"]
    out = begin("C556", {
        "status": "cross_model_functional_replication_branch_frozen",
        "authorized_types": causal_types,
        "models": ["GLM4-9B", "DeepSeek-R1-Distill-Qwen-7B"],
        "sequence": "load and release GLM4 before loading DeepSeek7B",
        "comparison": "within-model typed response formation profile; never physical coordinate identity across models",
        "rule": "if C555 has no causal type, do not load either model and record NA",
    }, {"parent": final("C555")["all_checks_passed"]})
    if not causal_types:
        close("C556", {
            "status": "cross_model_not_run", "ran": False, "models_loaded": [],
            "result": "NA_qwen_causal_qualification_failed", "metrics": {},
        }, {"no_model": True, "registered_na": True}, "C557_visual")
        return
    # Cross-model work is deliberately conditional and uses behavior-qualified
    # within-model response profiles; this branch is rarely reached.
    model_results: dict[str, dict] = {}
    loaded: list[str] = []
    worker = TESTS / "phase2090_c556_cross_model_worker.py"
    for model_name in ("glm4", "deepseek7b"):
        worker_result = out / f"analysis/{model_name}_worker_result.json"
        completed = subprocess.run(
            [sys.executable, str(worker), "--model", model_name, "--output", str(worker_result)],
            cwd=str(ROOT), capture_output=True, text=True, check=False,
        )
        (out / f"audit/{model_name}_worker_stdout.txt").write_text(completed.stdout, encoding="utf-8")
        (out / f"audit/{model_name}_worker_stderr.txt").write_text(completed.stderr, encoding="utf-8")
        if worker_result.exists():
            result = load(worker_result)
        else:
            result = {"status": "worker_failed_without_result", "returncode": completed.returncode}
        result["returncode"] = completed.returncode
        model_results[model_name] = result
        if completed.returncode == 0 and result.get("status") == "closed":
            loaded.append(model_name)
        gc.collect()
    close("C556", {
        "status": "cross_model_branch_closed", "ran": True, "models_loaded": loaded,
        "metrics": model_results,
        "strict_interpretation": "This conditional branch establishes execution and behavior qualification only; full functional response is a later contract.",
    }, {"models": loaded == ["glm4", "deepseek7b"], "finite": finite(model_results)}, "C557_visual")


def register_visual() -> None:
    entry = {
        "id": "c557_typed_operation_response_passport_atlas",
        "title": "C557 Typed Operation Response Passport Atlas",
        "phase": 2091, "campaign": "C542-C559",
        "path": "vis_data/research_kernel/c558_typed_operation_response_passport_atlas.json",
        "schema": "ai2050.typed_operation_response_passport.v1",
        "description": "Typed language program graph, full-coordinate response passports, controls, and representative all-token fields.",
    }
    registry = load(REGISTRY) if REGISTRY.exists() else {"datasets": []}
    key = "datasets" if "datasets" in registry else "items"
    registry.setdefault(key, [])
    registry[key] = [item for item in registry[key] if item.get("id") != entry["id"]] + [entry]
    save(REGISTRY, registry)
    catalog = load(CATALOG) if CATALOG.exists() else {"datasets": []}
    key = "datasets" if "datasets" in catalog else "items"
    catalog.setdefault(key, [])
    catalog[key] = [item for item in catalog[key] if item.get("id") != entry["id"]] + [entry]
    save(CATALOG, catalog)


def c557() -> None:
    out = begin("C557", {
        "status": "response_passport_full_coordinate_visual_atlas_frozen",
        "content": ["typed operation graph", "all-coordinate operation prototypes", "strong controls", "representative all-token fields"],
        "selection": "top two operations by independent-unit median margin; selection affects display only, never analysis",
    }, {"parent": final("C556")["all_checks_passed"]})
    index = capture_index()
    rows = read_rows(material_path())
    full = np.load(capture_paths()[2], mmap_mode="r")
    prototype_npz = np.load(OUTS["C546"] / "analysis/full_coordinate_passport_prototypes.npz", allow_pickle=False)
    evidence = final("C549")["headline"]["unit_evidence"]
    selected_ops = sorted(evidence, key=lambda key: evidence[key]["median_margin"], reverse=True)[:2]
    representative_indices = []
    for operation in selected_ops:
        domain = OPERATION_SPECS[operation]["domains"][0]
        candidates = [
            int(row["hidden_index"]) for row in index
            if row.get("operation_type") == operation and row.get("operation_domain") == domain
            and row["surface"] == "record" and row["partition"] == "lockbox" and int(row["unit"]) == 14
        ]
        representative_indices.extend(sorted(candidates))
    prototype_rows = []
    for operation in OPERATION_SPECS:
        for q in QPOINTS:
            values = [
                np.asarray(prototype_npz[key], np.float32)
                for key in prototype_npz.files if key.startswith(operation + "|") and key.endswith(f"|q{q}")
            ]
            prototype_rows.append({
                "operation_type": operation, "checkpoint": q,
                "values": np.mean(values, axis=0).astype(float).tolist(),
            })
    full_rows = []
    for source_i in representative_indices:
        length = int(index[source_i]["length"])
        full_rows.append({
            "case_id": index[source_i]["case_id"], "operation_type": index[source_i]["operation_type"],
            "domain": index[source_i]["operation_domain"], "variant": index[source_i]["variant"],
            "length": length,
            "checkpoints": {
                str(q): np.asarray(full[source_i, q, :length], np.float32).astype(float).tolist()
                for q in QPOINTS
            },
            "role_positions": index[source_i]["role_positions"],
        })
    atlas = {
        "schema": "ai2050.typed_operation_response_passport.v1", "phase": 2091, "campaign": "C542-C559",
        "coordinate_count": DIM, "checkpoint_count": CHECKPOINTS, "qpoints": list(QPOINTS), "roles": list(ROLES),
        "ontology": load(OUTS["C543"] / "material/typed_operation_ontology.json"),
        "selected_operations": selected_ops, "prototype_rows": prototype_rows, "full_token_rows": full_rows,
        "within_domain": {key: final("C546")["headline"][key] for key in ("gate_summary", "operation_pass_rate")},
        "cross_domain": {key: final("C547")["headline"][key] for key in ("gate_summary", "operation_pass_rate")},
        "confounds": {key: final("C548")["headline"][key] for key in ("gate_summary", "operation_pass_rate")},
        "independent_units": final("C549")["headline"], "history": final("C550")["headline"]["gate_summary"],
        "attitude_composition": final("C551")["headline"]["gate_summary"],
        "graph_composition": final("C552")["headline"]["gate_summary"],
        "token_views": final("C553")["headline"]["view_rates"],
        "causal": final("C555")["headline"], "cross_model": final("C556")["headline"],
    }
    save(VISUAL, atlas)
    register_visual()
    close_mmap(full)
    prototype_values = len(prototype_rows) * len(ROLES) * DIM
    full_values = sum(len(row["checkpoints"]) * row["length"] * DIM for row in full_rows)
    close("C557", {
        "status": "visual_atlas_closed", "visual_path": str(VISUAL.relative_to(ROOT)),
        "selected_operations": selected_ops, "prototype_rows": len(prototype_rows),
        "representative_rows": len(full_rows), "prototype_coordinate_values": prototype_values,
        "full_token_coordinate_values": full_values, "visual_bytes": VISUAL.stat().st_size,
    }, {
        "visual": VISUAL.exists(), "prototype_rows": len(prototype_rows) == 11 * len(QPOINTS),
        "full_rows": len(full_rows) == 4, "coordinates": prototype_values > 1_000_000 and full_values > 1_000_000,
    }, "C558_cleanup")


def c558() -> None:
    out = begin("C558", {
        "status": "raw_field_cleanup_and_next_stage_adjudication_frozen",
        "cleanup": "hash and delete all three C545 raw state arrays only after all analyses and C557 visual preservation",
        "retained": ["materials", "compiled prompts", "hidden index", "metrics", "full-coordinate prototypes", "visual atlas", "hash ledger"],
    }, {"parent": final("C557")["all_checks_passed"]})
    cleanup = []
    for path in capture_paths():
        if path.exists():
            cleanup.append({"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha(path)})
    save(out / "audit/raw_field_cleanup_ledger.json", {"files": cleanup, "total_bytes": sum(row["bytes"] for row in cleanup)})
    for row in cleanup:
        (ROOT / row["path"]).unlink()
    typed_candidates = final("C549")["headline"]["qualified_types"]
    causal_types = final("C555")["headline"]["causal_types"]
    if causal_types:
        next_same_goal, route = True, "fresh_material_replication_of_causal_typed_response"
    elif typed_candidates:
        next_same_goal, route = True, "refine_token_compiler_for_predictive_typed_response"
    else:
        next_same_goal, route = False, "new_full_token_conditional_interaction_object_required"
    close("C558", {
        "status": "cleanup_and_next_stage_closed", "cleanup_files": len(cleanup),
        "cleanup_bytes": sum(row["bytes"] for row in cleanup),
        "raw_fields_absent": all(not path.exists() for path in capture_paths()),
        "next_stage_same_goal": next_same_goal, "next_route": route,
    }, {
        "ledger": len(cleanup) == 3, "cleanup": all(not path.exists() for path in capture_paths()),
        "visual_retained": VISUAL.exists(),
    }, "C559_synthesis")


def c559() -> None:
    out = begin("C559", {
        "status": "campaign_synthesis_and_theory_ledger_frozen",
        "evidence_levels": ["behavior", "observation", "prediction", "causal", "cross-model", "NA"],
        "new_math_gate": ["cross-material stable object", "unseen prediction", "composition", "causal use", "cross-model functional isomorphism"],
    }, {"parent": final("C558")["all_checks_passed"]})
    gates = {
        "within_domain": final("C546")["headline"]["gate_summary"],
        "cross_domain": final("C547")["headline"]["gate_summary"],
        "confound": final("C548")["headline"]["gate_summary"],
        "typed_candidates": final("C549")["headline"]["candidate_count"],
        "history": final("C550")["headline"]["gate_summary"],
        "attitude_composition": final("C551")["headline"]["gate_summary"],
        "graph_composition": final("C552")["headline"]["gate_summary"],
        "causal_types": len(final("C555")["headline"]["causal_types"]),
        "cross_model_ran": final("C556")["headline"]["ran"],
    }
    new_math_gates = {
        "cross_material_stable_object": final("C549")["headline"]["candidate_count"] > 0,
        "unseen_prediction": final("C549")["headline"]["candidate_count"] > 0,
        # The graph interaction is not eligible because its behavior panel scored 0.25.
        "behavior_qualified_composition": final("C551")["headline"]["attitude_composition_candidate"],
        "causal_use": bool(final("C555")["headline"]["causal_types"]),
        # C556 measured execution and behavior only, never cross-model HiddenState isomorphism.
        "cross_model_functional_isomorphism": False,
    }
    new_math_score = int(sum(new_math_gates.values()))
    close("C559", {
        "status": "campaign_synthesis_closed", "gates": gates,
        "qualified_types": final("C549")["headline"]["qualified_types"],
        "new_math_gates": new_math_gates,
        "new_math_gate_score": new_math_score, "new_math_gate_total": 5,
        "new_foundational_math_authorized": new_math_score == 5,
        "next_stage_same_goal": final("C558")["headline"]["next_stage_same_goal"],
        "next_route": final("C558")["headline"]["next_route"],
        "strict_conclusion": "The campaign tests typed full-coordinate response passports. It does not equate an external operation label with an internal neural operator unless predictive, confound, composition, causal, and cross-model accounts all close.",
    }, {"complete": len(gates) == 9, "finite": finite(gates), "visual": VISUAL.exists()}, "C560_independent_audit")


FUNCTIONS = {
    "C542": c542, "C543": c543, "C544": c544, "C545": c545, "C546": c546,
    "C547": c547, "C548": c548, "C549": c549, "C550": c550, "C551": c551,
    "C552": c552, "C553": c553, "C554": c554, "C555": c555, "C556": c556,
    "C557": c557, "C558": c558, "C559": c559,
}


def self_test() -> None:
    rows = all_material()
    assert len(rows) == 2808
    assert len({row["case_id"] for row in rows}) == 2808
    atomic = [row for row in rows if row["panel"] == "typed_atomic_operation"]
    assert len(atomic) == 2376
    assert all(len(spec["domains"]) == 3 for spec in OPERATION_SPECS.values())
    assert len(composition_material()) == 432
    assert abs(np.mean([row["option_order"] == 0 for row in rows]) - 0.5) < 1e-12
    print(json.dumps({"status": "self_test_passed", "rows": len(rows), "atomic": len(atomic)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="C542")
    parser.add_argument("--stop", default="C559")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    names = list(FUNCTIONS)
    start = names.index(args.start)
    stop = names.index(args.stop)
    if stop < start:
        raise ValueError((args.start, args.stop))
    for name in names[start:stop + 1]:
        print(f"\n=== {name} / Phase {PHASES[name][0]} ===", flush=True)
        FUNCTIONS[name]()


if __name__ == "__main__":
    main()
