#!/usr/bin/env python3
"""C369-C390: language-operation graph and full-coordinate response campaign.

The campaign observes embeddings and HiddenState checkpoints only. It never
reads attention maps, MLP activations, or model weights. Every measured model
axis is retained in full; summaries are derived views rather than substitutes
for the archived physical coordinates.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import itertools
import json
import math
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
sys.path.insert(0, str(TESTS))

import phase1844_c310_c335_dual_axis_common as common
import phase1797_c263_c272_state_operator_common as family_base
from model_utils import get_model_info

PHASES = {
    f"C{campaign}": (1903 + campaign - 369, slug)
    for campaign, slug in (
        (369, "evidence_and_language_operation_contract"),
        (370, "typed_language_operation_registry"),
        (371, "sixteen_family_material_compiler"),
        (372, "semantic_and_zero_model_audit"),
        (373, "qwen_language_family_behavior"),
        (374, "qwen_language_family_full_coordinate_field"),
        (375, "full_coordinate_operation_response_atlas"),
        (376, "cross_material_transfer_matrix"),
        (377, "conditional_second_order_interval_operator"),
        (378, "composition_order_response"),
        (379, "predictive_response_state_refinement"),
        (380, "ternary_graph_behavior_repair"),
        (381, "ternary_graph_full_coordinate_field"),
        (382, "recursive_graph_depth_forecast"),
        (383, "natural_language_external_panels"),
        (384, "known_truth_conditional_operator_calibration"),
        (385, "causal_eligibility_adjudication"),
        (386, "conditional_operator_deletion_rescue"),
        (387, "qwen_bilingual_operation_panel"),
        (388, "glm_bilingual_operation_panel"),
        (389, "deepseek_bilingual_operation_panel"),
        (390, "language_operation_campaign_synthesis"),
    )
}
OUTS = {
    campaign: RESULT / f"phase{phase}_{campaign.lower()}_{slug}"
    for campaign, (phase, slug) in PHASES.items()
}

ROLES = common.ROLES
DIM = 2560
CHECKPOINTS = 38
SURFACES = ("report", "hearing", "notes")
OLD_FAMILIES = tuple(common.FAMILIES)
NEW_FAMILIES = (
    "agent_patient_voice",
    "possession",
    "spatial_relation",
    "temporal_order",
    "causal_direction",
    "negation_scope",
    "modality",
    "coreference",
    "attribute_binding",
    "part_whole",
)
FAMILIES = OLD_FAMILIES + NEW_FAMILIES
OPS = ("A", "B", "I", "K")
CELLS = ("00", "10", "01", "11_ab", "11_ba")

NEW_UNITS = (
    {"primary": "Arlen", "secondary": "Bex", "observer": "Cira", "object": "quince", "other": "lantern", "node": "qavik", "middle": "pomelin", "parent": "produce", "wrong": "instrument"},
    {"primary": "Doran", "secondary": "Elya", "observer": "Fenn", "object": "turnip", "other": "compass", "node": "turelin", "middle": "rootava", "parent": "vegetable", "wrong": "device"},
    {"primary": "Garin", "secondary": "Hessa", "observer": "Ilan", "object": "papaya", "other": "violin", "node": "paporin", "middle": "fruitel", "parent": "food", "wrong": "music"},
    {"primary": "Jalen", "secondary": "Kira", "observer": "Leto", "object": "celeriac", "other": "tripod", "node": "celavik", "middle": "tuberin", "parent": "plant", "wrong": "tool"},
    {"primary": "Maren", "secondary": "Niko", "observer": "Orla", "object": "guava", "other": "abacus", "node": "guavor", "middle": "orchava", "parent": "organism", "wrong": "number"},
    {"primary": "Pavel", "secondary": "Rhea", "observer": "Soren", "object": "radicchio", "other": "hourglass", "node": "radelin", "middle": "leafora", "parent": "entity", "wrong": "time"},
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_rows(path: Path, values) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def producer_hash() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def begin(campaign: str, protocol: dict, checks: dict) -> Path:
    out = OUTS[campaign]
    if (out / "analysis/final.json").exists():
        return out
    if out.exists():
        raise RuntimeError(f"partial output exists: {out}")
    if not all(checks.values()):
        raise RuntimeError((campaign, checks))
    for sub in ("analysis", "audit", "compiled", "material", "protocol", "raw"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    save(out / "protocol/preregistration.json", {
        "phase": PHASES[campaign][0],
        "campaign": campaign,
        "created_at_utc": utc_now(),
        "producer_sha256": producer_hash(),
        **protocol,
    })
    save(out / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": True})
    return out


def close(campaign: str, headline: dict, checks: dict, next_authorization: str) -> dict:
    out = OUTS[campaign]
    if (out / "analysis/final.json").exists():
        return load(out / "analysis/final.json")
    save(out / "analysis/summary.json", headline)
    save(out / "audit/internal_analysis_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    final_checks = {
        "contract": load(out / "audit/internal_contract_audit.json")["all_checks_passed"],
        "analysis": all(checks.values()),
        "producer_hash": load(out / "protocol/preregistration.json")["producer_sha256"] == producer_hash(),
    }
    final = {
        "phase": PHASES[campaign][0],
        "campaign": campaign,
        "status": "closed",
        "checks": final_checks,
        "all_checks_passed": all(final_checks.values()),
        "headline": headline,
        "next_authorization": next_authorization,
    }
    save(out / "analysis/final.json", final)
    print(json.dumps(final, ensure_ascii=False), flush=True)
    return final


def final(campaign: str) -> dict:
    return load(OUTS[campaign] / "analysis/final.json")


def finite(value) -> bool:
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(v) for v in value)
    if isinstance(value, (float, np.floating)):
        return math.isfinite(float(value))
    return True


def operation_registry() -> list[dict]:
    specs = {
        "attitude_event": ("AttitudeState", "Event", "attitude polarity", "voice realization"),
        "type_graph": ("Entity", "Class", "path depth", "direct shortcut"),
        "contrast": ("Proposition", "Proposition", "connective form", "clause order"),
        "translation": ("Symbol", "Meaning", "path depth", "direct mapping"),
        "comparison": ("EntityPair", "Dimension", "comparison dimension", "inverse wording"),
        "nested_attitude": ("AttitudeState", "Event", "attitude wrapper", "patient specificity"),
        "agent_patient_voice": ("Event", "EntityPair", "active/passive voice", "patient replacement"),
        "possession": ("Possession", "EntityPair", "owns/belongs wording", "possessed item"),
        "spatial_relation": ("SpatialState", "EntityPair", "direct/inverse wording", "spatial axis"),
        "temporal_order": ("TemporalState", "EventPair", "before/after wording", "event type"),
        "causal_direction": ("CausalState", "EventPair", "active/passive causation", "outcome type"),
        "negation_scope": ("Proposition", "Scope", "inner/outer negation", "report wrapper"),
        "modality": ("ModalState", "Event", "must/required wording", "event patient"),
        "coreference": ("DiscourseState", "Entity", "name/first-person reference", "voice realization"),
        "attribute_binding": ("AttributeState", "Entity", "has/belongs wording", "attribute value"),
        "part_whole": ("PartWholeState", "EntityPair", "includes/part-of wording", "part identity"),
    }
    rows = []
    for family in FAMILIES:
        input_type, output_type, op_a, op_b = specs[family]
        rows.append({
            "family": family,
            "input_type": input_type,
            "output_type": output_type,
            "roles": list(ROLES),
            "operation_a": op_a,
            "operation_b": op_b,
            "composition_domain": "typed and context-indexed",
            "inverse_assumed": False,
            "commutative_assumed": False,
            "neural_family_assumed": False,
        })
    return rows


def wrap_surface(surface: str, fact1: str, fact2: str, question: str, reverse: bool = False) -> str:
    left, right = (fact2, fact1) if reverse else (fact1, fact2)
    if surface == "report":
        return f"A report records that {left} It separately records that {right} Based only on the report, {question}"
    if surface == "hearing":
        return f"During a hearing, a witness stated that {left} The witness then added that {right} Decide from these statements: {question}"
    if surface == "notes":
        return f"Notes: {left} Separate note: {right} Query: {question}"
    raise KeyError(surface)


def custom_case(family: str, unit: int, a: int, b: int, op_order: str) -> dict:
    u = NEW_UNITS[unit]
    p, s, o = u["primary"], u["secondary"], u["observer"]
    obj, other = u["object"], u["other"]
    node, middle, parent, wrong = u["node"], u["middle"], u["parent"], u["wrong"]
    if family == "agent_patient_voice":
        item = other if b else obj
        relation = "inspected"
        target = f"{p} inspected the {item}." if a == 0 else f"The {item} was inspected by {p}."
        noise = f"{s} catalogued the {obj if b else other}."
        question, correct, distractor = f"Who inspected the {item}?", p, s
        roles = {"primary": p, "secondary": s, "relation": relation, "context": item, "query": item}
    elif family == "possession":
        item = other if b else obj
        relation = "owns" if a == 0 else "belongs to"
        target = f"{p} owns the {item}." if a == 0 else f"The {item} belongs to {p}."
        noise = f"{s} examined the {obj if b else other}."
        question, correct, distractor = f"Who owns the {item}?", p, s
        roles = {"primary": p, "secondary": s, "relation": relation, "context": item, "query": item}
    elif family == "spatial_relation":
        direct, inverse = (("north of", "south of") if b == 0 else ("east of", "west of"))
        relation = direct if a == 0 else inverse
        target = f"{p} is {direct} {s}." if a == 0 else f"{s} is {inverse} {p}."
        noise = f"The {obj} is beside the {other}."
        question, correct, distractor = f"Who is {direct} the other person?", p, s
        roles = {"primary": p, "secondary": s, "relation": relation, "context": s, "query": direct}
    elif family == "temporal_order":
        event = "arrived" if b == 0 else "departed"
        relation = "before" if a == 0 else "after"
        target = f"{p} {event} before {s}." if a == 0 else f"{s} {event} after {p}."
        noise = f"{o} moved the {other}."
        question, correct, distractor = f"Who {event} first?", p, s
        roles = {"primary": p, "secondary": s, "relation": relation, "context": event, "query": event}
    elif family == "causal_direction":
        outcome = "alarm" if b == 0 else "signal"
        relation = "caused" if a == 0 else "was caused by"
        target = f"{p}'s action caused the {outcome}." if a == 0 else f"The {outcome} was caused by {p}'s action."
        noise = f"{s} carried the {other}."
        question, correct, distractor = f"Whose action caused the {outcome}?", p, s
        roles = {"primary": p, "secondary": s, "relation": relation, "context": outcome, "query": outcome}
    elif family == "negation_scope":
        relation = "did not claim" if a else "claimed"
        if a and b and op_order == "ba":
            target = f"{o} claimed that {p} did not inspect the {obj}."
            relation = "did not inspect"
        elif a:
            target = f"{o} did not claim that {p} inspected the {obj}."
        elif b:
            target = f"{o} claimed that {p} did not inspect the {obj}."
            relation = "did not inspect"
        else:
            target = f"{o} claimed that {p} inspected the {obj}."
        noise = f"{s} adjusted the {other}."
        question, correct, distractor = f"Who is mentioned as the possible inspector of the {obj}?", p, s
        roles = {"primary": p, "secondary": s, "relation": relation, "context": obj, "query": obj}
    elif family == "modality":
        item = other if b else obj
        relation = "must inspect" if a == 0 else "is required to inspect"
        target = f"{p} must inspect the {item}." if a == 0 else f"{p} is required to inspect the {item}."
        noise = f"{s} may carry the {obj if b else other}."
        question, correct, distractor = f"Who is required to inspect the {item}?", p, s
        roles = {"primary": p, "secondary": s, "relation": relation, "context": item, "query": item}
    elif family == "coreference":
        relation = "I inspected" if a else "inspected"
        if a == 0:
            target = f"{p} told {s} that {p} inspected the {obj}."
        else:
            target = f"{p} told {s}, 'I inspected the {obj}.'"
        if b:
            target += f" The {obj} was later checked again by {p}."
        noise = f"{s} stored the {other}."
        question, correct, distractor = f"Who inspected the {obj}?", p, s
        roles = {"primary": p, "secondary": s, "relation": relation, "context": obj, "query": obj}
    elif family == "attribute_binding":
        value = "red" if b == 0 else "blue"
        relation = "has" if a == 0 else "belongs to"
        target = f"{p} has the {value} marker." if a == 0 else f"The {value} marker belongs to {p}."
        noise = f"{s} has the green marker."
        question, correct, distractor = f"Whose marker is {value}?", p, s
        roles = {"primary": p, "secondary": s, "relation": relation, "context": value, "query": value}
    elif family == "part_whole":
        part = other if b == 0 else middle
        relation = "includes" if a == 0 else "is part of"
        target = f"The {obj} includes the {part}." if a == 0 else f"The {part} is part of the {obj}."
        noise = f"The {wrong} contains the {node}."
        question, correct, distractor = f"Which whole includes the {part}?", obj, wrong
        roles = {"primary": obj, "secondary": part, "relation": relation, "context": wrong, "query": part}
    else:
        raise KeyError(family)
    return {
        "target": target,
        "noise": noise,
        "question": question,
        "correct": correct,
        "wrong": distractor,
        "roles": roles,
        "semantic_graph": {"family": family, "factor_a": a, "factor_b": b, "composition_order": op_order},
    }


def build_case(family: str, surface: str, unit: int, a: int, b: int, op_order: str) -> dict:
    if family in OLD_FAMILIES:
        saved = family_base.UNITS
        family_base.UNITS = NEW_UNITS
        try:
            case = family_base.nested_case("dossier", unit, a, b) if family == "nested_attitude" else family_base.semantic_case(family, "dossier", unit, a, b)
        finally:
            family_base.UNITS = saved
        fact1, fact2, question = common.extract_dossier_parts(case["prompt_core"])
        target = {"target": fact1, "noise": fact2, "question": question, "correct": case["correct"], "wrong": case["wrong"], "roles": case["roles"], "semantic_graph": case["semantic_graph"]}
    else:
        target = custom_case(family, unit, a, b, op_order)
    reverse = op_order == "ba" and not (family == "negation_scope" and a == b == 1)
    prompt_core = wrap_surface(surface, target["target"], target["noise"], target["question"], reverse=reverse)
    return {**target, "prompt_core": prompt_core}


def language_material() -> list[dict]:
    rows = []
    for family, surface, unit, cell, answer_order in itertools.product(FAMILIES, SURFACES, range(len(NEW_UNITS)), CELLS, (1, -1)):
        a, b = (0, 0) if cell == "00" else (1, 0) if cell == "10" else (0, 1) if cell == "01" else (1, 1)
        op_order = "ba" if cell == "11_ba" else "ab"
        case = build_case(family, surface, unit, a, b, op_order)
        choices, gold = family_base.options(case["correct"], case["wrong"], answer_order)
        rows.append({
            "case_id": f"c371-{family}-{surface}-u{unit}-{cell}-{answer_order:+d}",
            "panel": "language_operation_graph",
            "family": family,
            "surface": surface,
            "unit": unit,
            "cell": cell,
            "factor_a": a,
            "factor_b": b,
            "composition_order": op_order,
            "order_semantics": "scope" if family == "negation_scope" and a == b == 1 else "realization",
            "order": answer_order,
            "partition": "discovery" if unit < 3 else "confirmation" if unit < 5 else "lockbox",
            "gold_position": gold,
            "correct_answer": case["correct"],
            "wrong_answer": case["wrong"],
            "prompt_core": case["prompt_core"],
            "prompt": f"{case['prompt_core']} {choices}. Reply with only A or B.",
            "free_prompt": f"{case['prompt_core']} Answer with only the answer word.",
            "role_values": case["roles"],
            "semantic_graph": case["semantic_graph"],
        })
    return rows


def compile_qwen_rows(tokenizer, rows: list[dict]) -> list[dict]:
    return family_base.compile_qwen(tokenizer, rows)


@torch.inference_mode()
def qwen_behavior(rows: list[dict], compiled: list[dict], out: Path, batch_size: int = 12) -> dict:
    model = None
    behavior = []
    try:
        model, _tokenizer, device, placement = common.model_base.load_bf16("qwen3")
        pad = int(_tokenizer.pad_token_id if _tokenizer.pad_token_id is not None else _tokenizer.eos_token_id)
        for start in range(0, len(compiled), batch_size):
            batch = compiled[start:start + batch_size]
            width = max(len(row["prompt_ids"]) for row in batch)
            ids = torch.full((len(batch), width), pad, dtype=torch.long, device=device)
            mask = torch.zeros_like(ids)
            positions = torch.zeros_like(ids)
            lengths = []
            for local, row in enumerate(batch):
                values = row["prompt_ids"]
                lengths.append(len(values))
                ids[local, :len(values)] = torch.tensor(values, dtype=torch.long, device=device)
                mask[local, :len(values)] = 1
                positions[local, :len(values)] = torch.arange(len(values), device=device)
            output = model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False, return_dict=True)
            for local, row in enumerate(batch):
                logits = [float(output.logits[local, lengths[local] - 1, candidate[0]]) for candidate in row["candidate_ids"]]
                prediction = int(logits[1] > logits[0])
                behavior.append({
                    "case_id": row["case_id"], "family": row["family"], "surface": row["surface"],
                    "unit": row["unit"], "cell": row["cell"], "partition": row["partition"],
                    "gold_position": row["gold_position"], "prediction": prediction,
                    "correct": prediction == row["gold_position"], "score0": logits[0], "score1": logits[1],
                })
            if start % 240 == 0 or start + len(batch) == len(compiled):
                print(f"[C373 behavior] {start + len(batch)}/{len(compiled)}", flush=True)
        write_rows(out / "raw/behavior.jsonl", behavior)
        return {"placement": placement, "rows": len(behavior), "accuracy": float(np.mean([row["correct"] for row in behavior]))}
    finally:
        common.model_base.release(model)
        gc.collect()


def material_lookup() -> tuple[list[dict], dict[str, dict]]:
    values = read_rows(OUTS["C371"] / "material/cases.jsonl")
    return values, {row["case_id"]: row for row in values}


def qualified_groups(states: np.ndarray, hidden_index: list[dict], material: dict[str, dict], family: str, partitions: tuple[str, ...] | None = None) -> list[dict]:
    index_by_case = {row["case_id"]: row for row in hidden_index if row["correct"]}
    groups = []
    for surface, unit, answer_order in itertools.product(SURFACES, range(len(NEW_UNITS)), (1, -1)):
        entries = {}
        for cell in CELLS:
            case_id = f"c371-{family}-{surface}-u{unit}-{cell}-{answer_order:+d}"
            if case_id in index_by_case:
                entries[cell] = index_by_case[case_id]["hidden_index"]
        partition = material[f"c371-{family}-{surface}-u{unit}-00-{answer_order:+d}"]["partition"]
        if len(entries) == len(CELLS) and (partitions is None or partition in partitions):
            h = {cell: np.asarray(states[i], np.float32) for cell, i in entries.items()}
            groups.append({
                "family": family, "surface": surface, "unit": unit, "order": answer_order,
                "partition": partition, "h00": h["00"], "h10": h["10"], "h01": h["01"],
                "h11_ab": h["11_ab"], "h11_ba": h["11_ba"],
                "A": h["10"] - h["00"], "B": h["01"] - h["00"],
                "I": h["11_ab"] - h["10"] - h["01"] + h["00"],
                "K": h["11_ab"] - h["11_ba"],
            })
    return groups


def relative_gain(truth: np.ndarray, prediction: np.ndarray, baseline: np.ndarray | float = 0.0) -> float:
    base = float(np.mean(np.abs(truth - baseline)))
    error = float(np.mean(np.abs(truth - prediction)))
    return (base - error) / max(base, 1e-12)


def c369() -> None:
    out = begin("C369", {
        "status": "evidence_and_language_operation_contract_frozen",
        "audited_range": "C336-C368 / Phase1870-1902",
        "corrections": [
            "C330 used identity role alignment; it did not optimize the observed role permutation",
            "negative I results apply to the tested estimators and materials, not every possible second-order mechanism",
            "C365 old-material breadth does not prove that surface overlap caused the gain",
            "a missing language-family framework is a plausible organizational bottleneck, not an established root cause",
        ],
        "research_order": "observe full field -> discover repeated structure -> prospective validation -> causal adjudication",
    }, {"parent": final_old_c368(), "phase_continuity": PHASES["C369"][0] == 1903})
    headline = {
        "status": "evidence_audit_closed",
        "accepted": ["material conditioning is unresolved", "single-sample I remains unqualified", "external and internal graphs must be frozen independently"],
        "rejected_overclaims": ["universal first-order operator established", "systematic impossibility of second-order interaction", "new mathematics required now"],
        "strict_interpretation": "The next campaign tests a language-operation framework; it does not assume linguistic classes equal neural mechanism classes.",
    }
    close("C369", headline, {"corrections": len(headline["rejected_overclaims"]) == 3}, "C370_typed_registry")


def final_old_c368() -> bool:
    path = RESULT / "phase1902_c368_campaign_adjudication/analysis/final.json"
    return path.exists() and load(path)["all_checks_passed"]


def c370() -> None:
    out = begin("C370", {
        "status": "typed_language_operation_registry_frozen",
        "object": "typed operation/construction families, not keyword sentence classes",
        "layers": ["content ecology", "semantic roles", "atomic operations", "composition and scope", "representation transforms", "output protocol"],
        "family_count": len(FAMILIES),
        "no_neural_equivalence_assumption": True,
    }, {"parent": final("C369")["all_checks_passed"], "six_old_plus_ten_new": len(FAMILIES) == 16})
    registry = operation_registry()
    write_rows(out / "material/operation_registry.jsonl", registry)
    save(out / "material/external_language_graph.json", {
        "schema": "typed-language-operation-graph.v1",
        "nodes": [row["family"] for row in registry],
        "typed_edges": [{"source": row["input_type"], "operation": row["family"], "target": row["output_type"]} for row in registry],
        "composition_cells": list(CELLS),
        "undefined_compositions_allowed": True,
    })
    close("C370", {
        "status": "typed_registry_closed", "families": len(registry),
        "strict_interpretation": "This is an external linguistic candidate graph. It is neither exhaustive language theory nor a neural taxonomy.",
    }, {"registry": len(registry) == 16, "types": all(row["input_type"] and row["output_type"] for row in registry)}, "C371_material")


def c371() -> None:
    out = begin("C371", {
        "status": "sixteen_family_material_frozen",
        "design": "16 families x 3 surfaces x 6 lexical units x 5 composition cells x 2 answer orders",
        "partitions": {"discovery": [0, 1, 2], "confirmation": [3, 4], "lockbox": [5]},
        "cells": list(CELLS),
        "claim_boundary": "Controlled English supports typed contrasts but has no independent human naturalness certification.",
    }, {"parent": final("C370")["all_checks_passed"], "fixed_families": len(FAMILIES) == 16})
    rows = language_material()
    write_rows(out / "material/cases.jsonl", rows)
    counts = {partition: sum(row["partition"] == partition for row in rows) for partition in ("discovery", "confirmation", "lockbox")}
    headline = {"status": "material_closed", "rows": len(rows), "partition_counts": counts, "family_count": len({r["family"] for r in rows}), "surface_count": len({r["surface"] for r in rows}), "strict_interpretation": "The material broadens operations and constructions; it does not sample all language."}
    close("C371", headline, {"rows": len(rows) == 2880, "balance": sum(r["gold_position"] == 0 for r in rows) == 1440, "cells": {r["cell"] for r in rows} == set(CELLS)}, "C372_semantic_audit")


def c372() -> None:
    out = begin("C372", {
        "status": "semantic_and_zero_model_audit_frozen",
        "zero_models": ["always first", "always second", "family majority", "surface majority", "cell majority", "answer-word identity"],
        "semantic_checks": ["registered roles occur", "candidate answers differ", "all cells exist", "scope-order status explicit"],
        "naturalness": "deterministic grammar audit only; independent human review missing by design",
        "gate": "every zero model <=0.51 and every structural audit exact",
    }, {"parent": final("C371")["all_checks_passed"]})
    rows = read_rows(OUTS["C371"] / "material/cases.jsonl")
    accuracies = {
        "always_first": float(np.mean([r["gold_position"] == 0 for r in rows])),
        "always_second": float(np.mean([r["gold_position"] == 1 for r in rows])),
    }
    for key in ("family", "surface", "cell"):
        correct = 0
        for value in sorted({r[key] for r in rows}):
            subset = [r for r in rows if r[key] == value]
            majority = int(np.mean([r["gold_position"] for r in subset]) >= 0.5)
            correct += sum(r["gold_position"] == majority for r in subset)
        accuracies[f"{key}_majority"] = correct / len(rows)
    role_occurrence = all(all(str(value) in row["prompt_core"] for value in row["role_values"].values()) for row in rows)
    answers_distinct = all(row["correct_answer"] != row["wrong_answer"] for row in rows)
    gate = max(accuracies.values()) <= 0.51 and role_occurrence and answers_distinct
    save(out / "analysis/zero_models.json", accuracies)
    close("C372", {"status": "semantic_zero_model_audit_closed", "zero_model_accuracies": accuracies, "role_occurrence": role_occurrence, "answers_distinct": answers_distinct, "material_eligible": gate, "human_naturalness_review": False, "strict_interpretation": "Shortcut audit passed, but machine structural checks cannot certify human naturalness."}, {"zero_models": max(accuracies.values()) <= 0.51, "roles": role_occurrence, "answers": answers_distinct}, "C373_behavior")


def c373() -> None:
    out = begin("C373", {
        "status": "qwen_behavior_frozen",
        "model": "Qwen3-4B bf16 CUDA",
        "hidden_state_policy": "no hidden states requested",
        "gates": {"overall_confirmation": 0.80, "per_family": 0.60, "per_surface": 0.70},
        "routing": "all families remain observable; only behavior-qualified groups enter mechanism claims",
    }, {"parent": final("C372")["all_checks_passed"], "material_eligible": final("C372")["headline"]["material_eligible"], "cuda": torch.cuda.is_available()})
    rows = read_rows(OUTS["C371"] / "material/cases.jsonl")
    model = None
    try:
        model, tokenizer, _device, _placement = common.model_base.load_bf16("qwen3")
        compiled = compile_qwen_rows(tokenizer, rows)
        write_rows(out / "compiled/qwen3.jsonl", compiled)
    finally:
        common.model_base.release(model)
        model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    metrics = qwen_behavior(rows, compiled, out)
    behavior = read_rows(out / "raw/behavior.jsonl")
    confirmation = [r for r in behavior if r["partition"] in ("confirmation", "lockbox")]
    by_family = {family: float(np.mean([r["correct"] for r in confirmation if r["family"] == family])) for family in FAMILIES}
    by_surface = {surface: float(np.mean([r["correct"] for r in confirmation if r["surface"] == surface])) for surface in SURFACES}
    eligible_families = [family for family, score in by_family.items() if score >= 0.60]
    overall = float(np.mean([r["correct"] for r in confirmation]))
    gate = overall >= 0.80 and min(by_surface.values()) >= 0.70 and len(eligible_families) >= 12
    headline = {"status": "qwen_behavior_closed", **metrics, "confirmation_accuracy": overall, "family_accuracy": by_family, "surface_accuracy": by_surface, "eligible_families": eligible_families, "hidden_state_eligible": gate, "strict_interpretation": "Behavior qualifies observation only; it does not establish common internal operators."}
    close("C373", headline, {"rows": len(behavior) == len(rows), "finite": finite(headline), "no_hidden": not (out / "raw/role_states.float16.npy").exists()}, "C374_full_field")


def c374() -> None:
    eligible = final("C373")["headline"]["hidden_state_eligible"]
    out = begin("C374", {
        "status": "qwen_full_coordinate_field_frozen",
        "archive": "all rows x embedding+36 blocks+final norm x six roles x all 2560 coordinates",
        "all_token_subset": "lockbox/report/answer-order+1/cells 00 and 11_ab across all families",
        "no_pca_topk_cosine_gate": True,
        "cleanup": "bulk fields may be removed after all downstream audits and visualization export",
    }, {"parent": final("C373")["all_checks_passed"], "behavior_eligible": eligible, "cuda": torch.cuda.is_available()})
    rows, _ = material_lookup()
    compiled = read_rows(OUTS["C373"] / "compiled/qwen3.jsonl")
    selector = lambda row: row["partition"] == "lockbox" and row["surface"] == "report" and row["order"] == 1 and row["cell"] in ("00", "11_ab")
    metrics = common.batch_capture_qwen(rows, compiled, out, full_selector=selector, batch_size=8, field_width=192)
    close("C374", {"status": "qwen_full_coordinate_field_closed", **metrics, "strict_interpretation": "The archive is an observational field. Role span means are aligned views; the all-token subset preserves physical token detail."}, {"rows": metrics["rows"] == 2880, "role_shape": metrics["role_shape"] == [2880, 38, 6, 2560], "full_rows": metrics["full_token_rows"] == 32, "full_coordinates": metrics["full_shape"][-1] == 2560}, "C375_response_atlas")


def c375() -> None:
    out = begin("C375", {
        "status": "operation_response_atlas_frozen",
        "operations": list(OPS),
        "object": "family x operation x checkpoint x role x all physical coordinates",
        "metrics": ["mean signed response", "mean absolute energy", "cross-sample sign agreement", "first differentiation checkpoint"],
        "no_prediction_claim": True,
    }, {"parent": final("C374")["all_checks_passed"]})
    states = np.load(OUTS["C374"] / "raw/role_states.float16.npy", mmap_mode="r")
    index = read_rows(OUTS["C374"] / "raw/hidden_index.jsonl")
    _rows, material = material_lookup()
    mean_response = np.lib.format.open_memmap(out / "analysis/family_operation_mean_response.float16.npy", mode="w+", dtype=np.float16, shape=(len(FAMILIES), len(OPS), CHECKPOINTS, len(ROLES), DIM))
    energy = np.zeros((len(FAMILIES), len(OPS), CHECKPOINTS, len(ROLES)), np.float32)
    sign_agreement = np.zeros_like(energy)
    first = {}
    group_counts = {}
    for fi, family in enumerate(FAMILIES):
        groups = qualified_groups(states, index, material, family)
        group_counts[family] = len(groups)
        for oi, op in enumerate(OPS):
            values = np.asarray([group[op] for group in groups], np.float32)
            mean = values.mean(axis=0)
            mean_response[fi, oi] = mean.astype(np.float16)
            energy[fi, oi] = np.mean(np.abs(values), axis=(0, 3))
            sign_agreement[fi, oi] = np.maximum(np.mean(values >= 0, axis=(0, 3)), np.mean(values < 0, axis=(0, 3)))
            threshold = float(np.quantile(energy[fi, oi], 0.75))
            crossing = np.where(energy[fi, oi].mean(axis=1) >= threshold)[0]
            first[f"{family}:{op}"] = int(crossing[0]) if crossing.size else None
        print(f"[C375] {family}: {len(groups)} complete groups", flush=True)
        del groups
        gc.collect()
    mean_response.flush()
    np.save(out / "analysis/family_operation_energy.float32.npy", energy)
    np.save(out / "analysis/family_operation_sign_agreement.float32.npy", sign_agreement)
    save(out / "analysis/first_differentiation.json", first)
    headline = {"status": "response_atlas_closed", "shape": list(mean_response.shape), "group_counts": group_counts, "median_sign_agreement": float(np.median(sign_agreement)), "strict_interpretation": "Repeated signed responses are observational candidates, not semantic coordinates or causal gears."}
    close("C375", headline, {"shape": list(mean_response.shape) == [16, 4, 38, 6, 2560], "groups": min(group_counts.values()) >= 18, "finite": bool(np.isfinite(energy).all())}, "C376_cross_material")


def c376() -> None:
    out = begin("C376", {
        "status": "cross_material_transfer_matrix_frozen",
        "training_material": "one surface, discovery lexical units 0-2",
        "test_material": "each surface, independent lexical units 3-5",
        "matrix": "16 families x 4 operations x 3 source surfaces x 3 target surfaces",
        "controls": ["zero response", "wrong family source", "coordinate roll"],
        "metric": "full-coordinate MAE relative gain; no cosine",
    }, {"parent": final("C375")["all_checks_passed"]})
    states = np.load(OUTS["C374"] / "raw/role_states.float16.npy", mmap_mode="r")
    index = read_rows(OUTS["C374"] / "raw/hidden_index.jsonl")
    _rows, material = material_lookup()
    detail = []
    gains = np.full((len(FAMILIES), len(OPS), len(SURFACES), len(SURFACES)), np.nan, np.float32)
    for fi, family in enumerate(FAMILIES):
        wrong_family = FAMILIES[(fi + 1) % len(FAMILIES)]
        family_groups = qualified_groups(states, index, material, family)
        wrong_groups = qualified_groups(states, index, material, wrong_family)
        for oi, op in enumerate(OPS):
            wrong_train_all = [g[op] for g in wrong_groups if g["partition"] == "discovery"]
            wrong_mean = np.mean(wrong_train_all, axis=0)
            for si, source in enumerate(SURFACES):
                train = [g[op] for g in family_groups if g["surface"] == source and g["partition"] == "discovery"]
                if not train:
                    continue
                prediction = np.mean(train, axis=0)
                for ti, target in enumerate(SURFACES):
                    truth = np.asarray([g[op] for g in family_groups if g["surface"] == target and g["partition"] != "discovery"], np.float32)
                    if not len(truth):
                        continue
                    gain = relative_gain(truth, prediction)
                    coord_roll_gain = relative_gain(truth, np.roll(prediction, 1, axis=-1))
                    wrong_gain = relative_gain(truth, wrong_mean)
                    gains[fi, oi, si, ti] = gain
                    detail.append({"family": family, "operation": op, "source": source, "target": target, "gain": gain, "coordinate_roll_gain": coord_roll_gain, "wrong_family_gain": wrong_gain, "control_advantage": gain - max(coord_roll_gain, wrong_gain)})
        del family_groups, wrong_groups
        gc.collect()
    np.save(out / "analysis/cross_material_gains.float32.npy", gains)
    write_rows(out / "analysis/transfer_cells.jsonl", detail)
    cross = [r for r in detail if r["source"] != r["target"]]
    passing = sorted({r["family"] for r in cross if r["gain"] > 0 and r["control_advantage"] > 0.01})
    headline = {"status": "cross_material_matrix_closed", "cells": len(detail), "mean_cross_surface_gain": float(np.mean([r["gain"] for r in cross])), "mean_cross_surface_control_advantage": float(np.mean([r["control_advantage"] for r in cross])), "families_with_any_qualified_cross_cell": passing, "universal_transfer": len(passing) == len(FAMILIES), "strict_interpretation": "A positive cell is source-surface transfer for one family/operation; it is not a universal language operator."}
    close("C376", headline, {"shape": list(gains.shape) == [16, 4, 3, 3], "finite_cells": len(detail) > 500, "finite": finite(headline)}, "C377_conditional_I")


def c377() -> None:
    out = begin("C377", {
        "status": "conditional_second_order_interval_operator_frozen",
        "condition": "per-coordinate sign x low/high absolute H00 interval",
        "training": "discovery lexical units across all surfaces",
        "testing": "confirmation+lockbox lexical units",
        "controls": ["family mean I", "coordinate roll", "test-label roll"],
        "gate": "positive gain over family mean and >0.01 advantage over both rolls",
        "no_kernel_or_projection": True,
    }, {"parent": final("C376")["all_checks_passed"]})
    states = np.load(OUTS["C374"] / "raw/role_states.float16.npy", mmap_mode="r")
    index = read_rows(OUTS["C374"] / "raw/hidden_index.jsonl")
    _rows, material = material_lookup()
    results = []
    for family in FAMILIES:
        groups = qualified_groups(states, index, material, family)
        train = [g for g in groups if g["partition"] == "discovery"]
        test = [g for g in groups if g["partition"] != "discovery"]
        h0 = np.asarray([g["h00"] for g in train], np.float32)
        y = np.asarray([g["I"] for g in train], np.float32)
        x_test = np.asarray([g["h00"] for g in test], np.float32)
        truth = np.asarray([g["I"] for g in test], np.float32)
        threshold = np.median(np.abs(h0), axis=0)
        train_code = (h0 >= 0).astype(np.uint8) + 2 * (np.abs(h0) > threshold).astype(np.uint8)
        test_code = (x_test >= 0).astype(np.uint8) + 2 * (np.abs(x_test) > threshold).astype(np.uint8)
        fallback = y.mean(axis=0)
        prototypes = np.empty((4, CHECKPOINTS, len(ROLES), DIM), np.float32)
        for code in range(4):
            mask = train_code == code
            count = mask.sum(axis=0)
            prototypes[code] = np.where(count > 0, (y * mask).sum(axis=0) / np.maximum(count, 1), fallback)
        prediction = np.take_along_axis(prototypes[None], test_code[:, None], axis=1)[:, 0]
        mean_gain = relative_gain(truth, prediction, fallback)
        coord_roll_gain = relative_gain(truth, np.roll(prediction, 1, axis=-1), fallback)
        label_roll_gain = relative_gain(truth, np.roll(prediction, 1, axis=0), fallback)
        advantage = mean_gain - max(coord_roll_gain, label_roll_gain)
        results.append({"family": family, "test_groups": len(test), "gain_over_family_mean": mean_gain, "coordinate_roll_gain": coord_roll_gain, "label_roll_gain": label_roll_gain, "control_advantage": advantage, "passed": mean_gain > 0 and advantage > 0.01})
        print(f"[C377] {family}: gain={mean_gain:.5f}, advantage={advantage:.5f}", flush=True)
        del groups, train, test, h0, y, x_test, truth, threshold, train_code, test_code, prototypes, prediction
        gc.collect()
    write_rows(out / "analysis/family_results.jsonl", results)
    passing = [r["family"] for r in results if r["passed"]]
    headline = {"status": "conditional_second_order_closed", "families_passed": passing, "passed_count": len(passing), "mean_gain": float(np.mean([r["gain_over_family_mean"] for r in results])), "mean_control_advantage": float(np.mean([r["control_advantage"] for r in results])), "causal_candidate_eligible": len(passing) >= 8, "strict_interpretation": "Interval conditioning is a basic state-dependent response test. Failure does not rule out other nonlinear state objects."}
    close("C377", headline, {"families": len(results) == 16, "finite": finite(results)}, "C378_order")


def c378() -> None:
    out = begin("C378", {
        "status": "composition_order_response_frozen",
        "object": "K = H11_ab - H11_ba",
        "split": "negation_scope is semantic scope order; other families are realization-order controls",
        "prediction": "discovery mean K tested on confirmation+lockbox",
        "controls": ["zero", "coordinate roll", "wrong family"],
    }, {"parent": final("C377")["all_checks_passed"]})
    states = np.load(OUTS["C374"] / "raw/role_states.float16.npy", mmap_mode="r")
    index = read_rows(OUTS["C374"] / "raw/hidden_index.jsonl")
    _rows, material = material_lookup()
    results = []
    for fi, family in enumerate(FAMILIES):
        family_groups = qualified_groups(states, index, material, family)
        wrong_groups = qualified_groups(states, index, material, FAMILIES[(fi + 1) % len(FAMILIES)])
        train = np.asarray([g["K"] for g in family_groups if g["partition"] == "discovery"], np.float32)
        truth = np.asarray([g["K"] for g in family_groups if g["partition"] != "discovery"], np.float32)
        mean = train.mean(axis=0)
        wrong = np.asarray([g["K"] for g in wrong_groups if g["partition"] == "discovery"], np.float32).mean(axis=0)
        gain = relative_gain(truth, mean)
        roll = relative_gain(truth, np.roll(mean, 1, axis=-1))
        wrong_gain = relative_gain(truth, wrong)
        results.append({"family": family, "order_type": "semantic_scope" if family == "negation_scope" else "realization", "gain": gain, "coordinate_roll_gain": roll, "wrong_family_gain": wrong_gain, "control_advantage": gain - max(roll, wrong_gain)})
        del family_groups, wrong_groups, train, truth, mean, wrong
        gc.collect()
    write_rows(out / "analysis/order_results.jsonl", results)
    semantic = next(r for r in results if r["family"] == "negation_scope")
    headline = {"status": "composition_order_closed", "semantic_scope_result": semantic, "realization_mean_gain": float(np.mean([r["gain"] for r in results if r["order_type"] == "realization"])), "strict_interpretation": "Only the negation-scope row represents a semantic order contrast; the other K rows measure realization order and must not be called noncommutative language algebra."}
    close("C378", headline, {"families": len(results) == 16, "semantic_split": semantic["order_type"] == "semantic_scope", "finite": finite(results)}, "C379_state_refinement")


def c379() -> None:
    out = begin("C379", {
        "status": "predictive_response_state_refinement_frozen",
        "node": "(family, operation, checkpoint, role, physical coordinate)",
        "signature": "signed mean-response bin plus cross-sample sign-agreement bin",
        "claim_boundary": "descriptive state refinement, not lockbox compression or semantic atom discovery",
    }, {"parent": final("C378")["all_checks_passed"]})
    mean = np.load(OUTS["C375"] / "analysis/family_operation_mean_response.float16.npy", mmap_mode="r")
    agreement = np.load(OUTS["C375"] / "analysis/family_operation_sign_agreement.float32.npy")
    signatures = {}
    per_family = {}
    for fi, family in enumerate(FAMILIES):
        values = np.asarray(mean[fi], np.float32)
        scale = np.median(np.abs(values), axis=-1, keepdims=True)
        magnitude = np.where(np.abs(values) <= scale, 0, 1)
        sign = (values >= 0).astype(np.uint8)
        stable = (agreement[fi, :, :, :, None] >= 0.75).astype(np.uint8)
        code = sign + 2 * magnitude.astype(np.uint8) + 4 * stable
        counts = np.bincount(code.reshape(-1), minlength=8)
        per_family[family] = counts.tolist()
        signatures[family] = int(np.sum(counts > 0))
    save(out / "analysis/family_signature_counts.json", per_family)
    headline = {"status": "predictive_state_refinement_closed", "signature_cardinality": signatures, "global_nonempty_codes": sorted({i for counts in per_family.values() for i, count in enumerate(counts) if count}), "strict_interpretation": "The bins preserve response distinctions but do not prove predictive sufficiency or a minimal state space."}
    close("C379", headline, {"families": len(signatures) == 16, "codes_bounded": max(signatures.values()) <= 8}, "C380_graph_repair")


GRAPH_UNITS = tuple(
    {
        "root": f"nex{chr(97 + i)}",
        "mid1": f"lev{chr(97 + i)}",
        "mid2": f"qor{chr(97 + i)}",
        "final": f"class{chr(97 + i)}",
        "wrong": f"other{chr(97 + i)}",
    }
    for i in range(8)
)


def graph_rows() -> list[dict]:
    rows = []
    labels = ("entailed", "contradicted", "unknown")
    permutations = tuple(itertools.permutations(labels))
    for unit_i, depth, mode, surface, label_order in itertools.product(range(8), (1, 2, 3), labels, ("registry", "briefing"), permutations):
        unit = GRAPH_UNITS[unit_i]
        nodes = [unit["root"], unit["mid1"], unit["mid2"], unit["final"]]
        path = nodes[:depth] + [unit["final"]] if depth < 3 else nodes
        facts = [f'"{path[i]}" belongs to "{path[i + 1]}".' for i in range(len(path) - 1)]
        if mode == "contradicted":
            facts = [f'The registry explicitly states that "{unit["root"]}" does not belong to "{unit["final"]}".']
        elif mode == "unknown":
            if len(facts) == 1:
                facts = [f'"{unit["root"]}" belongs to "{unit["mid1"]}".']
            else:
                facts[len(facts) // 2] = f'"{path[len(facts) // 2]}" belongs to "{unit["wrong"]}".'
        body = " ".join(facts)
        rule = "Use only these rules: membership is transitive; an explicit denial means contradicted; a missing link without denial means unknown."
        if surface == "registry":
            core = f"Registry entries: {body} {rule} Is \"{unit['root']}\" a member of \"{unit['final']}\"?"
        else:
            core = f"A briefing supplied these links: {body} {rule} Classify the claim that \"{unit['root']}\" belongs to \"{unit['final']}\"."
        option_text = " ".join(f"({chr(65 + i)}) {label}" for i, label in enumerate(label_order))
        gold = label_order.index(mode)
        relation_role = "does not belong to" if mode == "contradicted" else "belongs to"
        secondary_role = unit["final"] if mode == "contradicted" else path[1]
        rows.append({
            "case_id": f"c380-{surface}-u{unit_i}-d{depth}-{mode}-p{permutations.index(label_order)}",
            "panel": "ternary_graph", "family": "type_graph", "surface": surface, "unit": unit_i,
            "depth": depth, "mode": mode, "partition": "discovery" if unit_i < 4 else "confirmation" if unit_i < 6 else "lockbox",
            "gold_position": gold, "prompt_core": core, "prompt": f"{core} {option_text} Reply with only A, B, or C.",
            "role_values": {"primary": unit["root"], "secondary": secondary_role, "relation": relation_role, "context": unit["final"], "query": unit["root"]},
            "semantic_graph": {"depth": depth, "mode": mode, "candidate_order": list(label_order)},
        })
    return rows


def compile_qwen_multiclass(tokenizer, rows: list[dict]) -> list[dict]:
    candidates = [tokenizer.encode(f" {letter}", add_special_tokens=False) for letter in "ABC"]
    if any(len(value) != 1 for value in candidates):
        raise RuntimeError(candidates)
    system = "Answer only from the supplied rules and entries. Do not use outside knowledge."
    compiled = []
    for row in rows:
        ids = family_base.core.chat_ids(tokenizer, system, row["prompt"])
        positions = {}
        for role, value in row["role_values"].items():
            spans = common.graph_base.name_spans(tokenizer, ids, value)
            if not spans:
                raise RuntimeError((row["case_id"], role, value))
            positions[role] = spans[-1] if role == "query" else spans[0]
        positions["boundary"] = [len(ids) - 1]
        compiled.append({**row, "prompt_ids": ids, "candidate_ids": candidates, "role_positions": positions})
    return compiled


@torch.inference_mode()
def qwen_multiclass_behavior(compiled: list[dict], out: Path, batch_size: int = 12) -> dict:
    model = None
    behavior = []
    try:
        model, tokenizer, device, placement = common.model_base.load_bf16("qwen3")
        pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
        for start in range(0, len(compiled), batch_size):
            batch = compiled[start:start + batch_size]
            width = max(len(row["prompt_ids"]) for row in batch)
            ids = torch.full((len(batch), width), pad, dtype=torch.long, device=device)
            mask = torch.zeros_like(ids)
            positions = torch.zeros_like(ids)
            lengths = []
            for local, row in enumerate(batch):
                values = row["prompt_ids"]
                lengths.append(len(values))
                ids[local, :len(values)] = torch.tensor(values, dtype=torch.long, device=device)
                mask[local, :len(values)] = 1
                positions[local, :len(values)] = torch.arange(len(values), device=device)
            output = model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False, return_dict=True)
            for local, row in enumerate(batch):
                scores = [float(output.logits[local, lengths[local] - 1, candidate[0]]) for candidate in row["candidate_ids"]]
                prediction = int(np.argmax(scores))
                behavior.append({"case_id": row["case_id"], "mode": row["mode"], "depth": row["depth"], "surface": row["surface"], "partition": row["partition"], "gold_position": row["gold_position"], "prediction": prediction, "correct": prediction == row["gold_position"], "scores": scores})
            if start % 144 == 0 or start + len(batch) == len(compiled):
                print(f"[C380 behavior] {start + len(batch)}/{len(compiled)}", flush=True)
        write_rows(out / "raw/behavior.jsonl", behavior)
        return {"placement": placement, "rows": len(behavior), "accuracy": float(np.mean([row["correct"] for row in behavior]))}
    finally:
        common.model_base.release(model)
        gc.collect()


@torch.inference_mode()
def capture_qwen_multiclass(compiled: list[dict], out: Path, batch_size: int = 8) -> dict:
    model = None
    hooks = []
    caught = []
    n = len(compiled)
    states = np.lib.format.open_memmap(out / "raw/role_states.float16.npy", mode="w+", dtype=np.float16, shape=(n, 38, len(ROLES), DIM))
    behavior, hidden_index = [], []
    try:
        model, tokenizer, device, placement = common.model_base.load_bf16("qwen3")
        base = model.model
        def capture(_module, _args, value):
            caught.append(value[0] if isinstance(value, tuple) else value)
        hooks.append(base.embed_tokens.register_forward_hook(capture))
        hooks.extend(layer.register_forward_hook(capture) for layer in base.layers)
        hooks.append(base.norm.register_forward_hook(capture))
        pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
        for start in range(0, n, batch_size):
            batch = compiled[start:start + batch_size]
            width = max(len(row["prompt_ids"]) for row in batch)
            ids = torch.full((len(batch), width), pad, dtype=torch.long, device=device)
            mask = torch.zeros_like(ids)
            positions = torch.zeros_like(ids)
            lengths = []
            for local, row in enumerate(batch):
                values = row["prompt_ids"]
                lengths.append(len(values))
                ids[local, :len(values)] = torch.tensor(values, dtype=torch.long, device=device)
                mask[local, :len(values)] = 1
                positions[local, :len(values)] = torch.arange(len(values), device=device)
            caught.clear()
            output = model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False, return_dict=True)
            if len(caught) != 38:
                raise RuntimeError(("checkpoint_count", len(caught)))
            for local, row in enumerate(batch):
                i = start + local
                length = lengths[local]
                for q, state in enumerate(caught):
                    value = state[local, :length].float().cpu().numpy()
                    for ri, role in enumerate(ROLES):
                        states[i, q, ri] = value[row["role_positions"][role]].mean(axis=0).astype(np.float16)
                scores = [float(output.logits[local, length - 1, candidate[0]]) for candidate in row["candidate_ids"]]
                prediction = int(np.argmax(scores))
                correct = prediction == row["gold_position"]
                behavior.append({"case_id": row["case_id"], "prediction": prediction, "correct": correct, "scores": scores})
                hidden_index.append({"hidden_index": i, "case_id": row["case_id"], "unit": row["unit"], "depth": row["depth"], "mode": row["mode"], "surface": row["surface"], "partition": row["partition"], "correct": correct, "role_positions": row["role_positions"]})
            states.flush()
            if start % 128 == 0 or start + len(batch) == n:
                print(f"[C381 capture] {start + len(batch)}/{n}", flush=True)
        write_rows(out / "raw/behavior.jsonl", behavior)
        write_rows(out / "raw/hidden_index.jsonl", hidden_index)
        return {"placement": placement, "rows": n, "accuracy": float(np.mean([row["correct"] for row in behavior])), "role_shape": list(states.shape)}
    finally:
        for hook in hooks:
            hook.remove()
        common.model_base.release(model)
        gc.collect()


def c380() -> None:
    out = begin("C380", {
        "status": "ternary_graph_behavior_repair_frozen",
        "design": "8 graphs x depths1-3 x entailed/contradicted/unknown x two surfaces x all six label permutations",
        "explicit_rules": True,
        "hidden_state_policy": "no hidden states requested",
        "gates": {"overall_confirmation": 0.80, "each_mode": 0.75, "each_depth": 0.70},
    }, {"parent": final("C379")["all_checks_passed"], "cuda": torch.cuda.is_available()})
    rows = graph_rows()
    write_rows(out / "material/cases.jsonl", rows)
    model = None
    try:
        model, tokenizer, _device, _placement = common.model_base.load_bf16("qwen3")
        compiled = compile_qwen_multiclass(tokenizer, rows)
        write_rows(out / "compiled/qwen3.jsonl", compiled)
    finally:
        common.model_base.release(model)
        model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    metrics = qwen_multiclass_behavior(compiled, out)
    behavior = read_rows(out / "raw/behavior.jsonl")
    test = [r for r in behavior if r["partition"] != "discovery"]
    by_mode = {mode: float(np.mean([r["correct"] for r in test if r["mode"] == mode])) for mode in ("entailed", "contradicted", "unknown")}
    by_depth = {str(depth): float(np.mean([r["correct"] for r in test if r["depth"] == depth])) for depth in (1, 2, 3)}
    overall = float(np.mean([r["correct"] for r in test]))
    eligible = overall >= 0.80 and min(by_mode.values()) >= 0.75 and min(by_depth.values()) >= 0.70
    headline = {"status": "ternary_graph_behavior_closed", **metrics, "confirmation_accuracy": overall, "mode_accuracy": by_mode, "depth_accuracy": by_depth, "graph_hidden_eligible": eligible, "strict_interpretation": "A pass qualifies this explicit-rule symbolic panel; it does not establish natural taxonomy recursion."}
    close("C380", headline, {"rows": len(rows) == 864, "balanced_labels": all(sum(r["gold_position"] == i for r in rows) == 288 for i in range(3)), "finite": finite(headline)}, "C381_graph_field")


def c381() -> None:
    eligible = final("C380")["headline"]["graph_hidden_eligible"]
    out = begin("C381", {
        "status": "ternary_graph_full_coordinate_field_frozen",
        "eligible": eligible,
        "archive": "all qualified rows x every checkpoint x six roles x all native coordinates",
        "routing": "if behavior fails, close as behavior-ineligible while other campaign routes continue",
    }, {"parent": final("C380")["all_checks_passed"], "eligibility_consistent": isinstance(eligible, bool)})
    if not eligible:
        close("C381", {"status": "graph_field_not_run_behavior_ineligible", "role_archive_created": False, "strict_interpretation": "The repaired material did not qualify; this does not refute graph recursion in Qwen3."}, {"no_archive": not (out / "raw/role_states.float16.npy").exists()}, "C382_no_recursive_claim")
        return
    compiled = read_rows(OUTS["C380"] / "compiled/qwen3.jsonl")
    metrics = capture_qwen_multiclass(compiled, out)
    close("C381", {"status": "graph_field_closed", **metrics, "strict_interpretation": "The field is an explicit-rule graph response archive."}, {"rows": metrics["rows"] == 864, "shape": metrics["role_shape"] == [864, 38, 6, 2560]}, "C382_depth_forecast")


def c382() -> None:
    eligible = final("C381")["headline"]["status"] == "graph_field_closed"
    out = begin("C382", {
        "status": "recursive_graph_depth_forecast_frozen",
        "eligible": eligible,
        "forecast": "discovery depth1->2 response predicts independent depth2->3 response",
        "controls": ["zero", "coordinate roll", "wrong mode"],
        "claim_boundary": "depth-response forecast, not autonomous symbolic recursion",
    }, {"parent": final("C381")["all_checks_passed"]})
    if not eligible:
        close("C382", {"status": "recursive_forecast_not_run_ineligible", "recursive_operator_established": False, "strict_interpretation": "No graph HiddenState claim is made."}, {"no_claim": True}, "C383_natural_panel")
        return
    states = np.load(OUTS["C381"] / "raw/role_states.float16.npy", mmap_mode="r")
    index = read_rows(OUTS["C381"] / "raw/hidden_index.jsonl")
    lookup = {(r["unit"], r["surface"], r["mode"], r["depth"]): r["hidden_index"] for r in index if r["correct"]}
    train_steps, test_truth = [], []
    for unit, surface in itertools.product(range(8), ("registry", "briefing")):
        keys = [(unit, surface, "entailed", d) for d in (1, 2, 3)]
        if all(key in lookup for key in keys):
            h1, h2, h3 = (np.asarray(states[lookup[key]], np.float32) for key in keys)
            if unit < 4:
                train_steps.append(h2 - h1)
            else:
                test_truth.append(h3 - h2)
    prediction = np.mean(train_steps, axis=0)
    truth = np.asarray(test_truth, np.float32)
    gain = relative_gain(truth, prediction)
    roll = relative_gain(truth, np.roll(prediction, 1, axis=-1))
    gate = gain > 0 and gain - roll > 0.01
    headline = {"status": "recursive_depth_forecast_closed", "gain_over_zero": gain, "coordinate_roll_gain": roll, "control_advantage": gain - roll, "recursive_depth_candidate": gate, "autonomous_recursion_established": False, "strict_interpretation": "Even a positive depth forecast would be a repeated depth response, not proof that one learned operator recursively executes itself."}
    close("C382", headline, {"train": len(train_steps) >= 4, "test": len(test_truth) >= 4, "finite": finite(headline)}, "C383_natural_panel")


NATURAL_CHAINS = (
    ("apple", "fruit", "food", "physical object"),
    ("robin", "bird", "animal", "organism"),
    ("violin", "instrument", "artifact", "physical object"),
    ("oak", "tree", "plant", "organism"),
    ("salmon", "fish", "animal", "organism"),
    ("hammer", "tool", "artifact", "physical object"),
)


def natural_panel_rows() -> list[dict]:
    rows = []
    for unit, surface, cell, answer_order in itertools.product(range(6), ("report", "dialogue"), ("event", "attitude", "inner_negation", "outer_negation"), (1, -1)):
        u = NEW_UNITS[unit]
        p, s, o, obj = u["primary"], u["secondary"], u["observer"], u["object"]
        if cell == "event":
            fact, relation = f"{p} ate the {obj}.", "ate"
        elif cell == "attitude":
            fact, relation = f"{o} liked the fact that {p} ate the {obj}.", "liked"
        elif cell == "inner_negation":
            fact, relation = f"{o} liked the fact that {p} did not eat the {obj}.", "did not eat"
        else:
            fact, relation = f"{o} did not like the fact that {p} ate the {obj}.", "did not like"
        distractor_fact = f"Separately, {s} moved the {u['other']}."
        core = f"A report says: {fact} {distractor_fact} Who is the person mentioned as eating or possibly eating the {obj}?" if surface == "report" else f"In a dialogue, someone said, '{fact}' The speaker added, '{distractor_fact}' Identify the person mentioned as eating or possibly eating the {obj}."
        choices, gold = family_base.options(p, s, answer_order)
        rows.append({"case_id": f"c383-attitude-{surface}-u{unit}-{cell}-{answer_order:+d}", "panel": "natural_attitude_event", "family": "nested_attitude", "surface": surface, "unit": unit, "cell": cell, "factor_a": int(cell in ("attitude", "inner_negation", "outer_negation")), "factor_b": int("negation" in cell), "order": answer_order, "partition": "discovery" if unit < 3 else "confirmation", "gold_position": gold, "correct_answer": p, "wrong_answer": s, "prompt_core": core, "prompt": f"{core} {choices}. Reply with only A or B.", "free_prompt": f"{core} Answer with only the answer word.", "role_values": {"primary": p, "secondary": s, "relation": relation, "context": obj, "query": obj}, "semantic_graph": {"cell": cell}})
    for unit, depth, answer_order in itertools.product(range(6), (1, 2, 3), (1, -1)):
        chain = NATURAL_CHAINS[unit]
        facts = [f"A {chain[i]} is a {chain[i + 1]}." for i in range(depth)]
        core = f"Natural category facts: {' '.join(facts)} Based on these facts, what broad class contains the {chain[0]}?"
        correct, wrong = chain[depth], NATURAL_CHAINS[(unit + 1) % 6][depth]
        choices, gold = family_base.options(correct, wrong, answer_order)
        rows.append({"case_id": f"c383-taxonomy-u{unit}-d{depth}-{answer_order:+d}", "panel": "natural_taxonomy", "family": "type_graph", "surface": "natural_facts", "unit": unit, "cell": f"depth{depth}", "factor_a": depth, "factor_b": 0, "order": answer_order, "partition": "discovery" if unit < 3 else "confirmation", "gold_position": gold, "correct_answer": correct, "wrong_answer": wrong, "prompt_core": core, "prompt": f"{core} {choices}. Reply with only A or B.", "free_prompt": f"{core} Answer with only the answer word.", "role_values": {"primary": chain[0], "secondary": chain[1], "relation": "is a", "context": correct, "query": chain[0]}, "semantic_graph": {"depth": depth, "chain": list(chain)}})
    return rows


def c383() -> None:
    out = begin("C383", {
        "status": "natural_external_panels_frozen",
        "panels": ["attitude-event with inner/outer negation", "natural category chains depth1-3"],
        "procedure": "behavior first; capture all role coordinates only if overall>=0.80 and each panel>=0.75",
        "human_review": "ordinary English templates, but no independent blinded human rating",
    }, {"parent": final("C382")["all_checks_passed"], "cuda": torch.cuda.is_available()})
    rows = natural_panel_rows()
    write_rows(out / "material/cases.jsonl", rows)
    model = None
    try:
        model, tokenizer, _device, _placement = common.model_base.load_bf16("qwen3")
        compiled = compile_qwen_rows(tokenizer, rows)
        write_rows(out / "compiled/qwen3.jsonl", compiled)
    finally:
        common.model_base.release(model)
        model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    behavior_metrics = qwen_behavior(rows, compiled, out, batch_size=12)
    behavior = read_rows(out / "raw/behavior.jsonl")
    test = [r for r in behavior if next(x for x in rows if x["case_id"] == r["case_id"])["partition"] == "confirmation"]
    row_map = {r["case_id"]: r for r in rows}
    panel_acc = {panel: float(np.mean([r["correct"] for r in test if row_map[r["case_id"]]["panel"] == panel])) for panel in ("natural_attitude_event", "natural_taxonomy")}
    overall = float(np.mean([r["correct"] for r in test]))
    eligible = overall >= 0.80 and min(panel_acc.values()) >= 0.75
    capture_metrics = None
    if eligible:
        capture_metrics = common.batch_capture_qwen(rows, compiled, out, full_selector=None, batch_size=8, field_width=192)
    headline = {"status": "natural_external_panels_closed", "rows": len(rows), "behavior": behavior_metrics, "confirmation_accuracy": overall, "panel_accuracy": panel_acc, "hidden_state_eligible": eligible, "capture": capture_metrics, "strict_interpretation": "These panels improve external validity but remain small controlled natural-language slices."}
    checks = {"rows": len(rows) == 132, "finite": finite(headline), "capture_consistent": (capture_metrics is not None) == eligible}
    close("C383", headline, checks, "C384_known_truth")


def c384() -> None:
    out = begin("C384", {
        "status": "known_truth_conditional_operator_calibration_frozen",
        "system": "synthetic full-coordinate state with sign/magnitude-conditioned interaction",
        "dimensions": [38, 6, 2560],
        "samples": {"discovery": 24, "lockbox": 16},
        "controls": ["mean", "coordinate roll", "label roll"],
        "claim_boundary": "metric calibration only; not a Transformer mechanism result",
    }, {"parent": final("C383")["all_checks_passed"]})
    rng = np.random.default_rng(3841903)
    shape = (38, 6, 2560)
    x_train = rng.normal(size=(24, *shape)).astype(np.float32)
    x_test = rng.normal(size=(16, *shape)).astype(np.float32)
    def truth(x):
        return (0.07 * np.sign(x) + 0.03 * (np.abs(x) > 0.7) - 0.01 * (np.abs(x) < 0.2)).astype(np.float32)
    y_train, y_test = truth(x_train), truth(x_test)
    threshold = np.median(np.abs(x_train), axis=0)
    train_code = (x_train >= 0).astype(np.uint8) + 2 * (np.abs(x_train) > threshold)
    test_code = (x_test >= 0).astype(np.uint8) + 2 * (np.abs(x_test) > threshold)
    fallback = y_train.mean(axis=0)
    prototypes = np.empty((4, *shape), np.float32)
    for code in range(4):
        mask = train_code == code
        count = mask.sum(axis=0)
        prototypes[code] = np.where(count > 0, (y_train * mask).sum(axis=0) / np.maximum(count, 1), fallback)
    prediction = np.take_along_axis(prototypes[None], test_code[:, None], axis=1)[:, 0]
    gain = relative_gain(y_test, prediction, fallback)
    coord = relative_gain(y_test, np.roll(prediction, 1, axis=-1), fallback)
    label = relative_gain(y_test, np.roll(prediction, 1, axis=0), fallback)
    passed = gain > 0.5 and gain - max(coord, label) > 0.25
    headline = {"status": "known_truth_calibration_closed", "gain_over_mean": gain, "coordinate_roll_gain": coord, "label_roll_gain": label, "control_advantage": gain - max(coord, label), "calibration_passed": passed, "strict_interpretation": "The interval estimator can recover its own known-truth class; this does not validate that class for Qwen."}
    close("C384", headline, {"finite": finite(headline), "calibration": passed}, "C385_causal_eligibility")


def c385() -> None:
    out = begin("C385", {
        "status": "causal_eligibility_frozen",
        "requirements": ["at least eight C377 families pass independent controls", "known-truth calibration passes", "at least one cross-material I cell passes", "graph and natural routes reported independently"],
        "no_gate_relaxation": True,
    }, {"parent": final("C384")["all_checks_passed"]})
    conditional = final("C377")["headline"]
    transfer = read_rows(OUTS["C376"] / "analysis/transfer_cells.jsonl")
    cross_i = [r for r in transfer if r["operation"] == "I" and r["source"] != r["target"] and r["gain"] > 0 and r["control_advantage"] > 0.01]
    gates = {"conditional_breadth": conditional["passed_count"] >= 8, "known_truth": final("C384")["headline"]["calibration_passed"], "cross_material_i": len(cross_i) > 0}
    eligible = all(gates.values())
    headline = {"status": "causal_eligibility_closed", "gates": gates, "qualified_cross_i_cells": len(cross_i), "causal_eligible": eligible, "strict_interpretation": "Failure closes only this causal patch branch; all observational family results remain available."}
    close("C385", headline, {"gate_logic": eligible == all(gates.values())}, "C386_causal_if_eligible")


def c386() -> None:
    eligible = final("C385")["headline"]["causal_eligible"]
    out = begin("C386", {
        "status": "conditional_operator_deletion_rescue_frozen",
        "eligible": eligible,
        "conditions": ["natural", "predicted_source_delete", "correct_target_restore", "wrong_family_restore", "coordinate_roll_restore"],
        "routing": "not run unless every C385 requirement is met",
    }, {"parent": final("C385")["all_checks_passed"]})
    if not eligible:
        close("C386", {"status": "causal_deletion_rescue_not_run_ineligible", "causal_claim": False, "strict_interpretation": "No intervention was performed because the prospective conditional interaction failed qualification."}, {"no_model_intervention": True}, "C387_bilingual_qwen")
        return
    # The qualified object is a full response predictor, not a sparse coordinate set.
    # A causal write would require a separately calibrated full-state intervention.
    close("C386", {"status": "causal_deletion_rescue_withheld_type_mismatch", "causal_claim": False, "reason": "qualified predictor has no calibrated token-position write compiler", "strict_interpretation": "Eligibility of a predictor does not authorize an uncalibrated full-state patch."}, {"type_safety": True}, "C387_bilingual_qwen")


BILINGUAL_FAMILIES = ("possession", "spatial_relation", "temporal_order", "causal_direction", "modality", "attribute_binding")


def bilingual_case(family: str, language: str, unit: int, a: int, b: int) -> dict:
    u = NEW_UNITS[unit]
    p, s, obj, other = u["primary"], u["secondary"], u["object"], u["other"]
    if language == "en":
        case = custom_case(family, unit, a, b, "ab")
        core = f"Statement: {case['target']} Separate fact: {case['noise']} {case['question']}"
        return {**case, "prompt_core": core}
    if family == "possession":
        item = other if b else obj; relation = "拥有" if a == 0 else "属于"
        target = f"{p}拥有{item}。" if a == 0 else f"{item}属于{p}。"; question = f"谁拥有{item}？"; context = item
    elif family == "spatial_relation":
        direct, inverse = (("北面", "南面") if b == 0 else ("东面", "西面")); relation = direct if a == 0 else inverse
        target = f"{p}在{s}的{direct}。" if a == 0 else f"{s}在{p}的{inverse}。"; question = f"谁在另一人的{direct}？"; context = s
    elif family == "temporal_order":
        event = "到达" if b == 0 else "离开"; relation = "之前" if a == 0 else "之后"
        target = f"{p}在{s}之前{event}。" if a == 0 else f"{s}在{p}之后{event}。"; question = f"谁先{event}？"; context = event
    elif family == "causal_direction":
        outcome = "警报" if b == 0 else "信号"; relation = "导致" if a == 0 else "由"
        target = f"{p}的行动导致了{outcome}。" if a == 0 else f"{outcome}由{p}的行动引起。"; question = f"谁的行动导致了{outcome}？"; context = outcome
    elif family == "modality":
        item = other if b else obj; relation = "必须检查" if a == 0 else "被要求检查"
        target = f"{p}必须检查{item}。" if a == 0 else f"{p}被要求检查{item}。"; question = f"谁必须检查{item}？"; context = item
    elif family == "attribute_binding":
        value = "红色" if b == 0 else "蓝色"; relation = "有" if a == 0 else "属于"
        target = f"{p}有{value}标记。" if a == 0 else f"{value}标记属于{p}。"; question = f"谁的标记是{value}？"; context = value
    else:
        raise KeyError(family)
    noise = f"{s}记录了另一件事。"
    return {"target": target, "noise": noise, "question": question, "correct": p, "wrong": s, "roles": {"primary": p, "secondary": s, "relation": relation, "context": context, "query": p}, "semantic_graph": {"family": family, "language": language}, "prompt_core": f"陈述：{target} 另一事实：{noise} 问题：{question}"}


def bilingual_rows() -> list[dict]:
    rows = []
    for family, language, unit, a, b, answer_order in itertools.product(BILINGUAL_FAMILIES, ("en", "zh"), range(3), (0, 1), (0, 1), (1, -1)):
        case = bilingual_case(family, language, unit, a, b)
        choices, gold = family_base.options(case["correct"], case["wrong"], answer_order)
        rows.append({"case_id": f"c387-{family}-{language}-u{unit}-{a}{b}-{answer_order:+d}", "panel": "bilingual_operation", "family": family, "language": language, "surface": language, "unit": unit, "factor_a": a, "factor_b": b, "cell": f"{a}{b}", "order": answer_order, "partition": "discovery" if unit < 2 else "lockbox", "gold_position": gold, "correct_answer": case["correct"], "wrong_answer": case["wrong"], "prompt_core": case["prompt_core"], "prompt": f"{case['prompt_core']} {choices}. Reply with only A or B.", "free_prompt": f"{case['prompt_core']} Answer with only the answer word.", "role_values": case["roles"], "semantic_graph": case["semantic_graph"]})
    return rows


def compile_model_rows(tokenizer, rows: list[dict], interface: str = "strict_chat") -> list[dict]:
    compiled = []
    for row in rows:
        ids, candidates = common.render_interface(tokenizer, row, interface)
        positions = {}
        for role, value in row["role_values"].items():
            spans = common.graph_base.name_spans(tokenizer, ids, value)
            if not spans:
                raise RuntimeError((row["case_id"], role, value))
            positions[role] = spans[-1] if role == "query" else spans[0]
        positions["boundary"] = [len(ids) - 1]
        compiled.append({**row, "prompt_ids": ids, "candidate_ids": candidates, "role_positions": positions})
    return compiled


@torch.inference_mode()
def run_bilingual_model(campaign: str, model_name: str) -> None:
    out = begin(campaign, {
        "status": "single_model_bilingual_panel_frozen", "model": model_name,
        "rows": "6 families x English/Chinese x 3 lexical units x 4 cells x two answer orders",
        "archive": "all native checkpoints x six roles x all model-native coordinates",
        "interface": "strict_chat frozen before behavior",
        "claim_boundary": "model-native coordinates are not aligned across models",
    }, {"parent": final("C386" if campaign == "C387" else f"C{int(campaign[1:]) - 1}")["all_checks_passed"], "registered_model": model_name in common.MODELS, "cuda": torch.cuda.is_available()})
    rows = bilingual_rows()
    write_rows(out / "material/cases.jsonl", rows)
    model = None
    try:
        model, tokenizer, device, placement = common.model_base.load_bf16(model_name)
        compiled = compile_model_rows(tokenizer, rows)
        write_rows(out / "compiled/model_rows.jsonl", compiled)
        info = get_model_info(model, model_name)
        nq = info.n_layers + 1
        states = np.lib.format.open_memmap(out / "raw/role_states.float16.npy", mode="w+", dtype=np.float16, shape=(len(rows), nq, len(ROLES), info.d_model))
        behavior, index = [], []
        for i, row in enumerate(compiled):
            ids = torch.tensor([row["prompt_ids"]], dtype=torch.long, device=device)
            output = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, return_dict=True, output_hidden_states=True)
            if len(output.hidden_states) != nq:
                raise RuntimeError((model_name, len(output.hidden_states), nq))
            for q, state in enumerate(output.hidden_states):
                for ri, role in enumerate(ROLES):
                    states[i, q, ri] = state[0, row["role_positions"][role]].mean(0).float().cpu().numpy().astype(np.float16)
            if all(len(candidate) == 1 for candidate in row["candidate_ids"]):
                scores = [float(output.logits[0, ids.shape[1] - 1, candidate[0]]) for candidate in row["candidate_ids"]]
            else:
                scores = common.score_prompt_candidates(model, row["prompt_ids"], row["candidate_ids"], device, int(tokenizer.pad_token_id or tokenizer.eos_token_id)).tolist()
            prediction = int(np.argmax(scores)); correct = prediction == row["gold_position"]
            behavior.append({"case_id": row["case_id"], "family": row["family"], "language": row["language"], "partition": row["partition"], "prediction": prediction, "correct": correct, "scores": scores})
            index.append({"hidden_index": i, "case_id": row["case_id"], "family": row["family"], "language": row["language"], "unit": row["unit"], "factor_a": row["factor_a"], "factor_b": row["factor_b"], "order": row["order"], "partition": row["partition"], "correct": correct})
            states.flush()
            if i % 48 == 0 or i + 1 == len(rows):
                print(f"[{campaign}] {model_name} {i + 1}/{len(rows)}", flush=True)
        write_rows(out / "raw/behavior.jsonl", behavior)
        write_rows(out / "raw/hidden_index.jsonl", index)
        lockbox = [r for r in behavior if r["partition"] == "lockbox"]
        by_language = {lang: float(np.mean([r["correct"] for r in lockbox if r["language"] == lang])) for lang in ("en", "zh")}
        by_family = {family: float(np.mean([r["correct"] for r in lockbox if r["family"] == family])) for family in BILINGUAL_FAMILIES}
        eligible = min(by_language.values()) >= 0.60 and min(by_family.values()) >= 0.50
        headline = {"status": "single_model_bilingual_panel_closed", "model": model_name, "placement": placement, "rows": len(rows), "role_shape": list(states.shape), "lockbox_accuracy": float(np.mean([r["correct"] for r in lockbox])), "language_accuracy": by_language, "family_accuracy": by_family, "abstract_response_eligible": eligible, "strict_interpretation": "Eligibility permits only role/checkpoint response abstraction, never native coordinate identity."}
        close(campaign, headline, {"rows": len(rows) == 288, "shape": states.shape[0] == 288 and states.shape[2] == 6, "finite": finite(headline)}, f"C{int(campaign[1:]) + 1}")
    finally:
        common.model_base.release(model)
        gc.collect()


def c387() -> None:
    run_bilingual_model("C387", "qwen3")


def c388() -> None:
    run_bilingual_model("C388", "glm4")


def c389() -> None:
    run_bilingual_model("C389", "deepseek7b")


def model_response_abstraction(campaign: str) -> dict:
    states = np.load(OUTS[campaign] / "raw/role_states.float16.npy", mmap_mode="r")
    index = read_rows(OUTS[campaign] / "raw/hidden_index.jsonl")
    nq = states.shape[1]
    checkpoints = sorted({int(round(v * (nq - 1))) for v in (0.0, 0.25, 0.5, 0.75, 1.0)})
    lookup = {(r["family"], r["language"], r["unit"], r["order"], r["factor_a"], r["factor_b"]): r for r in index if r["correct"]}
    vectors = {}
    for family, language in itertools.product(BILINGUAL_FAMILIES, ("en", "zh")):
        op_values = {op: [] for op in ("A", "B", "I")}
        for unit, answer_order in itertools.product(range(3), (1, -1)):
            keys = {(a, b): (family, language, unit, answer_order, a, b) for a, b in itertools.product((0, 1), repeat=2)}
            if all(key in lookup for key in keys.values()):
                h = {cell: np.asarray(states[lookup[key]["hidden_index"]], np.float32) for cell, key in keys.items()}
                op_values["A"].append(h[(1, 0)] - h[(0, 0)])
                op_values["B"].append(h[(0, 1)] - h[(0, 0)])
                op_values["I"].append(h[(1, 1)] - h[(1, 0)] - h[(0, 1)] + h[(0, 0)])
        for op, values in op_values.items():
            if not values:
                continue
            energy = np.mean(np.abs(values), axis=(0, 3))[checkpoints]
            vector = energy.reshape(-1)
            vector = vector / max(float(vector.sum()), 1e-12)
            vectors[f"{family}:{language}:{op}"] = vector.tolist()
    return {"checkpoints": checkpoints, "vectors": vectors, "native_shape": list(states.shape)}


def total_variation(left, right) -> float:
    return 0.5 * float(np.abs(np.asarray(left, np.float64) - np.asarray(right, np.float64)).sum())


def file_manifest_and_remove(paths: list[Path], out: Path) -> list[dict]:
    manifest = []
    for path in paths:
        if not path.exists():
            continue
        shape = list(np.load(path, mmap_mode="r").shape) if path.suffix == ".npy" else None
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
        manifest.append({"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": digest.hexdigest(), "shape": shape, "removed_after_analysis": True})
        path.unlink()
    save(out / "audit/hidden_state_cleanup_manifest.json", manifest)
    return manifest


def c390() -> None:
    out = begin("C390", {
        "status": "language_operation_campaign_synthesis_frozen",
        "gates": ["cross-material", "conditional I", "semantic order", "graph depth", "natural external", "bilingual abstract response", "causal", "new mathematics"],
        "visualization": "all 2560 Qwen coordinates for 16 families x A/B/I/K at embedding and four HiddenState checkpoints, boundary role; plus one all-token example",
        "cleanup": "bulk HiddenState arrays not retained by visualization are checksummed and removed after all analyses",
    }, {"parent": final("C389")["all_checks_passed"]})
    abstractions = {campaign: model_response_abstraction(campaign) for campaign in ("C387", "C388", "C389")}
    save(out / "analysis/model_response_abstractions.json", abstractions)
    model_pairs = []
    for left, right in (("C387", "C388"), ("C387", "C389"), ("C388", "C389")):
        common_keys = sorted(set(abstractions[left]["vectors"]) & set(abstractions[right]["vectors"]))
        values = [total_variation(abstractions[left]["vectors"][key], abstractions[right]["vectors"][key]) for key in common_keys]
        model_pairs.append({"left": left, "right": right, "common_states": len(common_keys), "mean_total_variation": float(np.mean(values)) if values else None})
    write_rows(out / "analysis/cross_model_pairs.jsonl", model_pairs)
    bilingual = {}
    for campaign, abstraction in abstractions.items():
        values = []
        for family, op in itertools.product(BILINGUAL_FAMILIES, ("A", "B", "I")):
            en, zh = f"{family}:en:{op}", f"{family}:zh:{op}"
            if en in abstraction["vectors"] and zh in abstraction["vectors"]:
                values.append(total_variation(abstraction["vectors"][en], abstraction["vectors"][zh]))
        bilingual[campaign] = {"states": len(values), "mean_total_variation": float(np.mean(values)) if values else None}
    save(out / "analysis/bilingual_consistency.json", bilingual)

    mean = np.load(OUTS["C375"] / "analysis/family_operation_mean_response.float16.npy", mmap_mode="r")
    q_indices = (0, 12, 24, 36, 37)
    boundary_i = ROLES.index("boundary")
    heat_rows = []
    for fi, family in enumerate(FAMILIES):
        for oi, op in enumerate(OPS):
            for q in q_indices:
                heat_rows.append({"id": f"{family}:{op}:q{q}:boundary", "family": family, "operation": op, "checkpoint": q, "role": "boundary", "values": np.asarray(mean[fi, oi, q, boundary_i], np.float32).round(6).tolist()})
    full_path = OUTS["C374"] / "raw/full_fields_holdout.float16.npy"
    full = np.load(full_path, mmap_mode="r")
    field_map = load(OUTS["C374"] / "raw/full_field_row_map.json")
    hidden_index = read_rows(OUTS["C374"] / "raw/hidden_index.jsonl")
    source_i = field_map["source_indices"][0]
    length = hidden_index[source_i]["length"]
    token_rows = []
    for q in (0, 24, 37):
        for token in range(length):
            token_rows.append({"id": f"token:{token}:q{q}", "token": token, "checkpoint": q, "values": np.asarray(full[0, q, token], np.float32).round(6).tolist()})
    visual = {
        "schema": "c390.language_operation_full_coordinate.v1", "phase": 1924, "campaign": "C390", "model": "Qwen3-4B",
        "dimensions": list(range(2560)), "family_operation_rows": heat_rows, "all_token_rows": token_rows,
        "checkpoints": list(q_indices), "roles": ["boundary"],
        "claim_boundary": "Mean signed response rows and one complete token field are parameter-level observations, not causal semantic coordinates.",
    }
    visual_path = ROOT / "frontend/public/vis_data/research_kernel/c390_language_operation_full_coordinate.json"
    save(visual_path, visual)

    gates = {
        "cross_material_any": len(final("C376")["headline"]["families_with_any_qualified_cross_cell"]) > 0,
        "conditional_i_breadth": final("C377")["headline"]["passed_count"] >= 8,
        "semantic_order": final("C378")["headline"]["semantic_scope_result"]["gain"] > 0,
        "graph_depth": final("C382")["headline"].get("recursive_depth_candidate", False),
        "natural_external": final("C383")["headline"]["hidden_state_eligible"],
        "bilingual_all_models": all(final(c)["headline"]["abstract_response_eligible"] for c in ("C387", "C388", "C389")),
        "causal": final("C386")["headline"]["causal_claim"],
    }
    new_math = gates["conditional_i_breadth"] and gates["causal"] and gates["bilingual_all_models"]
    gates["new_math"] = new_math

    cleanup_paths = [
        OUTS["C374"] / "raw/role_states.float16.npy",
        OUTS["C374"] / "raw/full_fields_holdout.float16.npy",
        OUTS["C381"] / "raw/role_states.float16.npy",
        OUTS["C383"] / "raw/role_states.float16.npy",
        OUTS["C387"] / "raw/role_states.float16.npy",
        OUTS["C388"] / "raw/role_states.float16.npy",
        OUTS["C389"] / "raw/role_states.float16.npy",
    ]
    manifest = file_manifest_and_remove(cleanup_paths, out)
    removed_bytes = sum(item["bytes"] for item in manifest)
    headline = {
        "status": "language_operation_campaign_closed", "gates": gates,
        "cross_model_pairs": model_pairs, "bilingual_consistency": bilingual,
        "visual_rows": {"family_operation": len(heat_rows), "all_token": len(token_rows), "coordinates": 2560},
        "cleanup": {"files": len(manifest), "bytes": removed_bytes},
        "new_math_gate_passed": new_math,
        "strict_interpretation": "The campaign maps typed linguistic candidates to full-coordinate responses. Positive transfer is local evidence; no universal operator, causal language algebra, cross-model coordinate identity, or new mathematics is claimed without its separate gate.",
    }
    close("C390", headline, {"phases": all(final(f"C{c}")["all_checks_passed"] for c in range(369, 390)), "visual_coordinates": len(visual["dimensions"]) == 2560 and all(len(row["values"]) == 2560 for row in heat_rows + token_rows), "cleanup": all(not (ROOT / item["path"]).exists() for item in manifest), "finite": finite(headline)}, "independent_audit_and_next_cross_construction_lockbox")


FUNCTIONS = {f"C{campaign}": globals()[f"c{campaign}"] for campaign in range(369, 391)}


def validate_only() -> None:
    rows = language_material()
    graph = graph_rows()
    bilingual = bilingual_rows()
    natural = natural_panel_rows()
    checks = {
        "language_rows": len(rows) == 2880,
        "language_balance": sum(row["gold_position"] == 0 for row in rows) == 1440,
        "all_families": {row["family"] for row in rows} == set(FAMILIES),
        "all_cells": {row["cell"] for row in rows} == set(CELLS),
        "graph_rows": len(graph) == 864,
        "graph_balance": all(sum(row["gold_position"] == i for row in graph) == 288 for i in range(3)),
        "bilingual_rows": len(bilingual) == 288,
        "natural_rows": len(natural) == 132,
        "phases_continuous": [PHASES[f"C{c}"][0] for c in range(369, 391)] == list(range(1903, 1925)),
    }
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    if not all(checks.values()):
        raise SystemExit(1)


def parse_range(value: str) -> list[str]:
    if "-" not in value:
        return [value.upper()]
    left, right = value.upper().split("-", 1)
    start, stop = int(left.lstrip("C")), int(right.lstrip("C"))
    return [f"C{campaign}" for campaign in range(start, stop + 1)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--run", default="C369-C390")
    args = parser.parse_args()
    if args.validate_only:
        validate_only()
        return
    for campaign in parse_range(args.run):
        FUNCTIONS[campaign]()


if __name__ == "__main__":
    main()
