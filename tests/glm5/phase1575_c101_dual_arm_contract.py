#!/usr/bin/env python3
"""Phase1575 / C101: freeze fresh confirmation and language-breadth arms."""
from __future__ import annotations

import itertools
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
PARENT = RESULT / "phase1574_c100_graph_walsh_heatmap_client_integration"
C098 = RESULT / "phase1571_c098_observation_first_graph_campaign"
C099 = RESULT / "phase1572_c099_fixed_width_graph_field_campaign"
C100 = RESULT / "phase1573_c100_graph_field_analysis_adapter"
OUT = RESULT / "phase1575_c101_dual_arm"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core
import phase1571_c098_observation_first_graph_campaign as graph_base

PHASE = 1575
CAMPAIGN = "C101"
PARTITIONS = graph_base.PARTITIONS
WORLDS = graph_base.WORLDS
GRAPH_FAMILIES = graph_base.FAMILIES
BREADTH_FAMILIES = ("attribute_binding", "agent_patient", "negation_scope", "whole_part_exception")
BREADTH_FACTORS = ("truth", "surface", "distractor", "code")
CODEBOOKS = graph_base.CODEBOOKS
BREADTH_SYSTEM = (
    "Use only the local record. Treat every sentence as authoritative in its local world. "
    "Respect word order, negation, scope and explicit exceptions. Follow the decision code "
    "and reply with exactly yes or no."
)


FRESH_NATURAL_UNITS = {
    "taxonomy": [
        ("robin", "passerine", "avian", "wrench", "implement", "creature", "lifeform", "existent"),
        ("pear", "pome", "produce", "scalpel", "utensil", "crop", "biomass", "matter"),
        ("cedar", "conifer", "flora", "dolphin", "mammal", "woody species", "biota", "living entity"),
        ("trumpet", "brasswind", "soundmaker", "canyon", "terrain", "crafted item", "physical object", "thing"),
        ("orchid", "angiosperm", "vegetation", "quasar", "space object", "photosynthetic organism", "biological system", "natural entity"),
        ("tuna", "teleost", "fauna", "bench", "seating", "aquatic creature", "marine life", "animate being"),
    ],
    "containment": [
        ("button", "tin", "pantry", "receipt", "briefcase", "compartment", "suitcase", "storeroom"),
        ("medal", "satchel", "chest", "photo", "album", "lining", "rucksack", "garage"),
        ("chip", "socket", "panel", "manual", "archive", "circuit", "console", "depot"),
        ("bead", "canister", "armoire", "voucher", "billfold", "cavity", "toolbox", "hangar"),
        ("passport", "portfolio", "strongroom", "leaflet", "notebook", "recess", "bookcase", "repository"),
        ("capsule", "vial", "freezer", "tag", "catalogue", "chamber", "appliance", "laboratory"),
    ],
    "comparison": [
        ("mite", "hamster", "pony", "crumb", "watermelon", "lizard", "sheep", "giraffe"),
        ("thimble", "teacup", "kettle", "sand speck", "goblet", "saucer", "basin", "reservoir"),
        ("hut", "villa", "fortress", "kiosk", "terminal", "community hall", "capitol", "province"),
        ("atom", "molecule", "cell", "speck", "crystal", "organelle", "tissue", "organ"),
        ("raisin", "bun", "oven", "sesame", "platter", "plum", "basketful", "vanload"),
        ("dinghy", "yacht", "liner", "scooter", "truck", "barge", "harbor", "ocean"),
    ],
    "precedence": [
        ("checkin", "boarding", "takeoff", "reservation", "landing", "security", "gate call", "touchdown"),
        ("briefing", "launch", "orbit", "design", "splashdown", "inspection", "countdown", "reentry"),
        ("audition", "casting", "filming", "scriptwriting", "editing", "workshop", "premiere", "release"),
        ("nomination", "balloting", "announcement", "campaign", "ceremony", "filing", "debate", "inauguration"),
        ("germination", "budding", "fruiting", "composting", "preservation", "irrigation", "pollination", "dormancy"),
        ("triage", "imaging", "medication", "referral", "followup", "assessment", "therapy", "remission"),
    ],
}


ATTRIBUTE_UNITS = [
    ("lantern", "amber", "silver", "barrel", "bridge"), ("helmet", "bronze", "ivory", "tower", "tunnel"),
    ("banner", "scarlet", "navy", "ladder", "railing"), ("tablet", "matte", "glossy", "engine", "trailer"),
    ("vessel", "striped", "plain", "anchor", "mast"), ("marker", "violet", "teal", "furnace", "chimney"),
    ("parcel", "dotted", "smooth", "harvest", "granary"), ("folder", "crimson", "beige", "piston", "axle"),
    ("badge", "copper", "pearl", "harvestman", "webbing"), ("gadget", "opaque", "clear", "ramp", "platform"),
    ("token", "golden", "gray", "valley", "ridge"), ("module", "rough", "polished", "pier", "quay"),
]
AGENT_UNITS = [
    ("Mira", "Tobin", "Lena", "Pavel", "guided"), ("Kara", "Nolan", "Iris", "Damon", "followed"),
    ("Sena", "Rufus", "Milo", "Vera", "helped"), ("Jora", "Caleb", "Tara", "Oren", "called"),
    ("Nila", "Boris", "Rhea", "Simon", "warned"), ("Faye", "Galen", "Dina", "Hugo", "visited"),
    ("Arin", "Leona", "Kian", "Marta", "praised"), ("Tessa", "Roman", "Elsa", "Jonas", "tracked"),
    ("Nora", "Felix", "Asha", "Victor", "met"), ("Cora", "Elias", "Zara", "Bruno", "phoned"),
    ("Luca", "Mina", "Otis", "Sara", "backed"), ("Rina", "Theo", "Uma", "Wade", "notified"),
]
NEGATION_UNITS = [
    ("Luma", "warm", "Nexa", "open", "sealed"), ("Ravo", "bright", "Seli", "quiet", "locked"),
    ("Pira", "stable", "Daro", "empty", "marked"), ("Kelo", "active", "Vani", "clean", "ready"),
    ("Moro", "visible", "Teli", "safe", "aligned"), ("Savi", "heavy", "Rilo", "dry", "charged"),
    ("Demi", "smooth", "Kavi", "level", "enabled"), ("Faro", "silent", "Nemi", "valid", "closed"),
    ("Tova", "fragile", "Leri", "exact", "stored"), ("Bena", "mobile", "Sora", "fresh", "tested"),
    ("Javi", "narrow", "Pelo", "round", "sorted"), ("Vero", "dense", "Mali", "sharp", "packed"),
]
EXCEPTION_UNITS = [
    ("unitA", "unitB", "panel", "console", "cover"), ("unitC", "unitD", "seal", "pump", "valve"),
    ("unitE", "unitF", "wheel", "cart", "handle"), ("unitG", "unitH", "filter", "vent", "grille"),
    ("unitJ", "unitK", "sensor", "meter", "dial"), ("unitL", "unitM", "cap", "bottle", "label"),
    ("unitN", "unitP", "hinge", "gate", "latch"), ("unitQ", "unitR", "lens", "camera", "strap"),
    ("unitS", "unitT", "blade", "cutter", "guard"), ("unitU", "unitV", "screen", "terminal", "keypad"),
    ("unitW", "unitX", "nozzle", "sprayer", "hose"), ("unitY", "unitZ", "battery", "device", "switch"),
]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def partition_for(index: int, per_partition: int) -> str:
    return PARTITIONS[index // per_partition]


def fresh_artificial(family_index: int, unit_index: int) -> tuple[str, ...]:
    base = 2000 + family_index * 200 + unit_index * 10
    return tuple(f"Kavo{base + offset:04d}" for offset in range(8))


def build_confirmation() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    units: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    case_index = 0
    for family_index, family in enumerate(GRAPH_FAMILIES):
        for world in WORLDS:
            for unit_index, natural in enumerate(FRESH_NATURAL_UNITS[family]):
                nodes = natural if world == "natural" else fresh_artificial(family_index, unit_index)
                if world == "counterfactual":
                    nodes = graph_base.counterfactual_nodes(natural)
                unit_id = f"c101a-{family}-{world}-{unit_index:02d}"
                unit = {
                    "arm": "confirmation",
                    "unit_id": unit_id,
                    "unit_index": unit_index,
                    "family": family,
                    "world": world,
                    "partition": graph_base.partition_for(unit_index),
                    "surface": graph_base.surface_for(unit_index),
                    "nodes": list(nodes),
                }
                units.append(unit)
                for x, y, branch, code in itertools.product((1, -1), repeat=4):
                    prompt, edges = graph_base.build_prompt(family, nodes, x, y, branch, code, unit["surface"])
                    follows, path_count = graph_base.reachable(edges, nodes[0], nodes[2])
                    truth = x == y
                    if follows != truth or path_count != int(truth):
                        raise RuntimeError((unit_id, x, y, edges, follows, path_count))
                    output_yes = truth if code == 1 else not truth
                    cases.append({
                        **unit,
                        "case_id": f"c101a-{case_index:04d}",
                        "x": x,
                        "y": y,
                        "branch": branch,
                        "code": code,
                        "codebook": CODEBOOKS[code]["name"],
                        "truth": truth,
                        "output_yes": output_yes,
                        "gold_position": 0 if output_yes else 1,
                        "path_count": path_count,
                        "edges": [list(edge) for edge in edges],
                        "prompt": prompt,
                    })
                    case_index += 1
    return units, cases


def breadth_prompt(family: str, values: tuple[str, ...], truth: int, surface: int, distractor: int, code: int) -> tuple[str, str, str]:
    a, b, c, d, e = values
    if family == "attribute_binding":
        core_sentences = [f"{a} has the {b} marker", f"{c} has the {d} marker"] if truth == 1 else [f"{a} has the {d} marker", f"{c} has the {b} marker"]
        query, anchor = f"Does {a} have the {b} marker?", b
    elif family == "agent_patient":
        core_sentences = [f"{a} followed {b}", f"{c} observed {d}"] if truth == 1 else [f"{b} followed {a}", f"{c} observed {d}"]
        query, anchor = f"Did {a} follow {b}?", b
    elif family == "negation_scope":
        core_sentences = [f"The note says that {a} is not {b}", f"The note does not say that {c} is {d}"] if truth == 1 else [f"The note does not say that {a} is {b}", f"The note says that {c} is not {d}"]
        query, anchor = f"Does the note say that {a} is not {b}?", b
    elif family == "whole_part_exception":
        core_sentences = [f"The exception log says {a} retains its {c}", f"The exception log says {b} lacks its {c}"] if truth == 1 else [f"The exception log says {a} lacks its {c}", f"The exception log says {b} retains its {c}"]
        core_sentences.insert(0, f"The handbook says a {d} normally has a {c}")
        query, anchor = f"Does {a} currently retain its {c}?", c
    else:
        raise KeyError(family)
    if surface == -1:
        core_sentences = list(reversed(core_sentences))
    distract = "A spare label is listed beside a spare tag" if distractor == 1 else "A spare tag is listed beside a spare label"
    record = ". ".join([*core_sentences, distract]) + "."
    prompt = (
        f"Focus before record: {a}. Local record: {record} Focus after record: {a}. "
        f"Query: {query} Decision code: {CODEBOOKS[code]['instruction']} Reply exactly yes or no."
    )
    return prompt, a, anchor


def build_breadth() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    inventories = {
        "attribute_binding": ATTRIBUTE_UNITS,
        "agent_patient": AGENT_UNITS,
        "negation_scope": NEGATION_UNITS,
        "whole_part_exception": EXCEPTION_UNITS,
    }
    units: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    case_index = 0
    for family in BREADTH_FAMILIES:
        for unit_index, values in enumerate(inventories[family]):
            unit = {
                "arm": "breadth",
                "unit_id": f"c101b-{family}-{unit_index:02d}",
                "unit_index": unit_index,
                "family": family,
                "world": "controlled_natural",
                "partition": partition_for(unit_index, 4),
                "surface": "factorial",
                "values": list(values),
            }
            units.append(unit)
            for truth, surface, distractor, code in itertools.product((1, -1), repeat=4):
                prompt, focus, anchor = breadth_prompt(family, values, truth, surface, distractor, code)
                output_yes = (truth == 1) if code == 1 else (truth != 1)
                cases.append({
                    **unit,
                    "case_id": f"c101b-{case_index:04d}",
                    "truth_factor": truth,
                    "surface_factor": surface,
                    "distractor_factor": distractor,
                    "code": code,
                    "codebook": CODEBOOKS[code]["name"],
                    "truth": truth == 1,
                    "output_yes": output_yes,
                    "gold_position": 0 if output_yes else 1,
                    "focus": focus,
                    "anchor": anchor,
                    "prompt": prompt,
                })
                case_index += 1
    return units, cases


def find_spans(tok: Any, ids: list[int], text: str) -> list[list[int]]:
    return graph_base.name_spans(tok, ids, text)


def compile_breadth(tok: Any, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidate_ids = [[int(v) for v in tok.encode(" " + c, add_special_tokens=False)] for c in ("yes", "no")]
    if any(len(ids) != 1 for ids in candidate_ids):
        raise RuntimeError(("candidate singleton", candidate_ids))
    compiled = []
    for row in rows:
        ids = core.chat_ids(tok, BREADTH_SYSTEM, row["prompt"])
        focus_spans = find_spans(tok, ids, row["focus"])
        anchor_spans = find_spans(tok, ids, row["anchor"])
        if len(focus_spans) < 4 or len(anchor_spans) < 2:
            raise RuntimeError((row["case_id"], focus_spans, anchor_spans))
        instruction = CODEBOOKS[row["code"]]["instruction"]
        code_spans = find_spans(tok, ids, instruction)
        if not code_spans:
            raise RuntimeError(("code span", row["case_id"]))
        roles = {
            "focus_pre": focus_spans[0],
            "focus_record": focus_spans[1],
            "focus_post": focus_spans[-2],
            "query_focus": focus_spans[-1],
            "query_anchor": anchor_spans[-1],
            "code_instruction": code_spans[-1],
            "boundary": [len(ids) - 1],
        }
        if not (max(roles["focus_pre"]) < min(roles["focus_record"]) < min(roles["focus_post"]) < min(roles["code_instruction"]) < roles["boundary"][0]):
            raise RuntimeError(("role order", row["case_id"], roles))
        if max(roles["query_focus"]) >= min(roles["code_instruction"]) or max(roles["query_anchor"]) >= min(roles["code_instruction"]):
            raise RuntimeError(("query/code order", row["case_id"], roles))
        compiled.append({**row, "prompt_ids": ids, "candidate_ids": candidate_ids, "role_positions": roles})
    return compiled


def ba(gold: list[bool], pred: list[bool]) -> float:
    return float(sum(
        sum(p == g for p, g in zip(pred, gold, strict=True) if g == label) / sum(g == label for g in gold)
        for label in (False, True)
    ) / 2)


def zero_models(rows: list[dict[str, Any]], breadth: bool) -> dict[str, float]:
    gold = [row["output_yes"] for row in rows]
    values = {
        "always_yes": [True] * len(rows),
        "always_no": [False] * len(rows),
        "code_only": [row["code"] == 1 for row in rows],
        "truth_without_code": [row["truth"] for row in rows],
    }
    if breadth:
        values.update({
            "surface_only": [row["surface_factor"] == 1 for row in rows],
            "distractor_only": [row["distractor_factor"] == 1 for row in rows],
        })
    else:
        values.update({
            "x_only": [row["x"] == 1 for row in rows],
            "y_only": [row["y"] == 1 for row in rows],
            "branch_only": [row["branch"] == 1 for row in rows],
        })
    values["truth_x_code_oracle"] = [row["truth"] == (row["code"] == 1) for row in rows]
    return {name: ba(gold, pred) for name, pred in values.items()}


def prepare() -> None:
    if OUT.exists():
        raise RuntimeError(f"C101 already exists: {OUT}")
    parent = core.load(PARENT / "analysis/final.json")
    requirements = core.load(PARENT / "protocol/c101_requirements.json")
    c100_final = core.load(C100 / "analysis/final.json")
    old_nodes = {str(node).casefold() for row in core.rows(C098 / "material/frozen_graph_units.jsonl") for node in row["nodes"]}
    fresh_nodes = {str(node).casefold() for family in GRAPH_FAMILIES for unit in FRESH_NATURAL_UNITS[family] for node in unit}
    conf_units, conf_cases = build_confirmation()
    breadth_units, breadth_cases = build_breadth()
    tok = graph_base.tokenizer()
    conf_compiled = [{**row, "arm": "confirmation"} for row in graph_base.compile_rows(tok, conf_cases)]
    breadth_compiled = compile_breadth(tok, breadth_cases)
    all_compiled = [*conf_compiled, *breadth_compiled]
    max_width = max(len(row["prompt_ids"]) for row in all_compiled)
    conf_zero = zero_models(conf_cases, False)
    breadth_zero = zero_models(breadth_cases, True)
    semantic_grid = []
    for x, y in itertools.product((1, -1), repeat=2):
        edges = graph_base.graph_edges(tuple(FRESH_NATURAL_UNITS["taxonomy"][0]), x, y, 1)
        follows, count = graph_base.reachable(edges, FRESH_NATURAL_UNITS["taxonomy"][0][0], FRESH_NATURAL_UNITS["taxonomy"][0][2])
        semantic_grid.append({"x": x, "y": y, "edges": [list(v) for v in edges], "query_source": FRESH_NATURAL_UNITS["taxonomy"][0][0], "query_target": FRESH_NATURAL_UNITS["taxonomy"][0][2], "path_count": count, "truth": follows})
    checks = {
        "parent": parent["status"] == "graph_walsh_heatmap_client_integration_verified",
        "requirements": requirements["status"] == "requirements_frozen_not_started" and requirements["confirmation_arm"]["primary_state"] == 24,
        "c100": c100_final["all_checks_passed"],
        "confirmation_units": len(conf_units) == 72,
        "confirmation_cases": len(conf_cases) == 1152,
        "breadth_units": len(breadth_units) == 48,
        "breadth_cases": len(breadth_cases) == 768,
        "fresh_graph_lexicon": not (old_nodes & fresh_nodes),
        "graph_partitions": Counter(row["partition"] for row in conf_units) == {p: 24 for p in PARTITIONS},
        "breadth_partitions": Counter(row["partition"] for row in breadth_units) == {p: 16 for p in PARTITIONS},
        "graph_truth": all((row["path_count"] == 1) == row["truth"] for row in conf_cases),
        "graph_false_zero": all(row["path_count"] == 0 for row in conf_cases if not row["truth"]),
        "graph_factor_balance": all(Counter(row[f] for row in conf_cases) == {1: 576, -1: 576} for f in graph_base.FACTORS),
        "breadth_factor_balance": all(Counter(row[f] for row in breadth_cases) == {1: 384, -1: 384} for f in ("truth_factor", "surface_factor", "distractor_factor", "code")),
        "graph_output_balance": Counter(row["output_yes"] for row in conf_cases) == {True: 576, False: 576},
        "breadth_output_balance": Counter(row["output_yes"] for row in breadth_cases) == {True: 384, False: 384},
        "graph_zero_models": all(v == 0.5 for k, v in conf_zero.items() if k != "truth_x_code_oracle") and conf_zero["truth_x_code_oracle"] == 1.0,
        "breadth_zero_models": all(v == 0.5 for k, v in breadth_zero.items() if k != "truth_x_code_oracle") and breadth_zero["truth_x_code_oracle"] == 1.0,
        "compiled_count": len(all_compiled) == 1920,
        "fixed_width": max_width < 320,
        "semantic_grid": [row["truth"] for row in semantic_grid] == [True, False, False, True] and [row["path_count"] for row in semantic_grid] == [1, 0, 0, 1],
        "producer_pre_freeze": True,
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    protocol = {
        "schema": "c101.dual_arm_embedding_hidden_state_field.v1",
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "created_at_utc": now(),
        "model": "Qwen3-4B local BF16 CUDA nonquantized",
        "scientific_order": ["observe", "find_structure", "freeze_prediction", "validate", "intervene"],
        "adjudication": {
            "retained": ["C098/C099 numeric audit", "C100 upstream instability", "C100 post-hoc late-boundary convergence candidate", "K270-R1 design-null correction"],
            "corrected": [
                "C099 behavior used the physical padded tail because logits_to_keep=1; behavior must be recalibrated at each real boundary",
                "xy is an audited routing-XNOR path contrast, not an unaudited edge-presence AND",
                "Walsh coordinates are activation coordinates, not weight parameters",
                "state24 is post-hoc until this fresh confirmation arm is revealed",
            ],
        },
        "behavior_recalibration": {
            "source": "C099 frozen prompts and C099 raw final Hidden States",
            "old_method": "right padding to 210, logits_to_keep=1, output.logits[:, -1]",
            "correct_method": "lm_head(real answer-boundary final Hidden State)",
            "required_checks": ["fresh forward equals stored final boundary after float16 archival tolerance", "repeat exact", "old and corrected score difference reported"],
        },
        "confirmation": {
            "units": 72,
            "cases": 1152,
            "primary": {"state": 24, "role": "boundary", "effect": "xy", "threshold": 0.50, "required": "24/24 C100-discovery to C101 confirmation/lockbox full-vector cosines"},
            "secondary": [
                {"states": [31, 32], "role": "boundary", "effect": "xy", "median_cosine_threshold": 0.70},
                {"states": [24, 31, 32], "prediction": "minimum cross-world cosine exceeds minimum cross-family cosine"},
                {"states": [24, 31, 32], "prediction": "C100 discovery Top64 median sign agreement >= 0.90"},
                {"states": [24, 31, 32], "prediction": "boundary code norm exceeds boundary xy norm"},
            ],
            "rule": "each prediction is adjudicated separately; no five-way conjunctive project stop",
        },
        "breadth": {
            "families": list(BREADTH_FAMILIES),
            "units": 48,
            "cases": 768,
            "factors": list(BREADTH_FACTORS),
            "effect_of_interest": "truth",
            "states": "embedding plus all 36 Hidden States",
            "roles": ["focus_pre", "focus_record", "focus_post", "query_focus", "query_anchor", "code_instruction", "boundary"],
            "rule": "descriptive atlas with typed missingness; no universal pass gate",
        },
        "storage": {
            "raw": "every subtoken of every registered semantic role, all 37 states, all 2560 activation coordinates, float16 archive",
            "analysis": "float32 Walsh coefficients with float64 means/cosines",
            "role_reduction": "mean only for registered role-level comparisons; raw subtokens remain available",
            "fixed_global_sequence_length": max_width,
            "batch_size": 8,
        },
        "numeric_gates": {"repeat_hidden_max_abs": 1e-6, "repeat_logit_max_abs": 1e-6, "causal_prefix_max_abs": 1e-6, "code_previsible_max_abs": 1e-6},
        "claim_boundary": {
            "allowed": "single-Qwen activation-coordinate observations and preregistered fresh-material confirmation",
            "forbidden": ["weight-parameter mechanism", "semantic neurons", "attention/MLP mechanism", "cross-model law", "new mathematics"],
        },
        "human_naturalness": {"blind_review": False, "missingness": "M_HUMAN_NATURALNESS", "machine_and_manual_author_review": True},
        "authorization": "run_phase1576_c101_qwen_capture",
    }
    protocol["producer_sha256"] = core.sha(Path(__file__))
    protocol["material_digest"] = core.digest({"confirmation": conf_cases, "breadth": breadth_cases})
    core.write_rows(OUT / "material/confirmation_units.jsonl", conf_units)
    core.write_rows(OUT / "material/confirmation_cases.jsonl", conf_cases)
    core.write_rows(OUT / "material/breadth_units.jsonl", breadth_units)
    core.write_rows(OUT / "material/breadth_cases.jsonl", breadth_cases)
    core.write_rows(OUT / "compiled/qwen3_confirmation.jsonl", conf_compiled)
    core.write_rows(OUT / "compiled/qwen3_breadth.jsonl", breadth_compiled)
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "audit/pre_model_material_semantic_zero_audit.json", {
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "all_checks_passed": all(checks.values()),
        "semantic_grid": semantic_grid,
        "confirmation_zero_models": conf_zero,
        "breadth_zero_models": breadth_zero,
        "naturalness": protocol["human_naturalness"],
        "authorization": protocol["authorization"],
    })
    examples = [conf_cases[0]] + [next(row for row in breadth_cases if row["family"] == family and row["truth_factor"] == row["surface_factor"] == row["distractor_factor"] == row["code"] == 1) for family in BREADTH_FAMILIES]
    core.write_rows(OUT / "material/frozen_test_examples.jsonl", examples)
    print(json.dumps({"checks": checks, "fixed_width": max_width, "authorization": protocol["authorization"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    prepare()
