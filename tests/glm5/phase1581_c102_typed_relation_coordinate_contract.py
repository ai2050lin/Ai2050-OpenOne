#!/usr/bin/env python3
"""Phase1581 / C102: freeze the typed-relation all-token coordinate campaign."""
from __future__ import annotations

import itertools
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
C101 = RESULT / "phase1575_c101_dual_arm"
OUT = RESULT / "phase1581_c102_typed_relation_coordinate_campaign"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core
import phase1571_c098_observation_first_graph_campaign as graph_base
import phase1575_c101_dual_arm_contract as c101_contract

PHASE = 1581
CAMPAIGN = "C102"
PARTITIONS = graph_base.PARTITIONS
GRAPH_FAMILIES = graph_base.FAMILIES
BREADTH_FAMILIES = c101_contract.BREADTH_FAMILIES
WORLDS = graph_base.WORLDS
CODEBOOKS = graph_base.CODEBOOKS

GRAPH_BASES = {
    "taxonomy": [
        ("waxwing", "oscine", "vertebrate", "caliper", "gauge", "winged organism", "metazoan", "entity"),
        ("lentil", "legume", "edible seed", "chisel", "cutting tool", "cultivated plant", "biotic material", "substance"),
        ("viola", "flowering herb", "plant life", "satellite", "orbital body", "seed plant", "living system", "natural object"),
    ],
    "containment": [
        ("keycard", "document sleeve", "office cabinet", "invoice", "cash pouch", "inner pocket", "travel trunk", "supply room"),
        ("spool", "parts bin", "workbench drawer", "map sheet", "chart tube", "side compartment", "equipment case", "service bay"),
        ("sample", "specimen jar", "cold cabinet", "memo", "binder", "isolation chamber", "lab console", "research room"),
    ],
    "comparison": [
        ("flea", "ferret", "bison", "seed", "squash", "iguana", "donkey", "rhinoceros"),
        ("pin", "mug", "cauldron", "dust mote", "pitcher", "plate", "vat", "cistern"),
        ("shed", "townhouse", "citadel", "booth", "station", "library", "palace", "metropolis"),
    ],
    "precedence": [
        ("application", "screening", "appointment", "inquiry", "checkout", "verification", "interview", "completion"),
        ("drafting", "review", "publication", "brainstorm", "archiving", "outline", "revision", "circulation"),
        ("seeding", "sprouting", "flowering", "soil prep", "reaping", "watering", "fertilization", "storage"),
    ],
}

ATTRIBUTE_UNITS = [
    ("compass", "indigo", "chalk", "gantry", "alcove"), ("beacon", "maroon", "jade", "trellis", "courtyard"),
    ("amphora", "azure", "coral", "winch", "dock"), ("medallion", "umber", "lilac", "awning", "arcade"),
    ("flask", "ochre", "cyan", "crane", "foundry"), ("reticule", "magenta", "olive", "keel", "boathouse"),
    ("visor", "saffron", "platinum", "culvert", "embankment"), ("cartridge", "vermillion", "ebony", "turbine", "nacelle"),
    ("plaque", "cerulean", "charcoal", "scaffold", "atrium"),
]
AGENT_UNITS = [
    ("Alden", "Bianca", "Cyrus", "Delia", "escorted"), ("Emil", "Freya", "Gideon", "Helena", "consulted"),
    ("Ines", "Jasper", "Keira", "Lorcan", "signaled"), ("Maeve", "Nico", "Opal", "Percy", "advised"),
    ("Quinn", "Rosa", "Stellan", "Thalia", "summoned"), ("Ulric", "Viola", "Wyatt", "Xenia", "assisted"),
    ("Yara", "Zane", "Ansel", "Blythe", "questioned"), ("Clive", "Daria", "Eamon", "Ysolde", "greeted"),
    ("Greta", "Hamish", "Ilse", "Jules", "reminded"),
]
NEGATION_UNITS = [
    ("Arvo", "humid", "Bexa", "latched", "primed"), ("Ciro", "hollow", "Dova", "tinted", "coded"),
    ("Eris", "elastic", "Feno", "balanced", "staged"), ("Garo", "porous", "Hesta", "buffered", "logged"),
    ("Ivo", "rigid", "Jexa", "shaded", "indexed"), ("Koro", "magnetic", "Lysa", "fastened", "queued"),
    ("Mero", "soluble", "Niva", "bolted", "routed"), ("Olan", "thermal", "Pexa", "folded", "tagged"),
    ("Qiro", "vacant", "Ruva", "clamped", "mapped"),
]
EXCEPTION_UNITS = [
    ("rigA", "rigB", "gasket", "compressor", "housing"), ("rigC", "rigD", "bracket", "shelf", "rail"),
    ("rigE", "rigF", "fuse", "controller", "receptacle"), ("rigG", "rigH", "rotor", "motor", "casing"),
    ("rigJ", "rigK", "shutter", "projector", "mount"), ("rigL", "rigM", "clasp", "casework", "veneer"),
    ("rigN", "rigP", "float", "cistern", "gauge"), ("rigQ", "rigR", "reed", "instrument", "mouthpiece"),
    ("rigS", "rigT", "carriage", "printer", "tray"),
]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def artificial_nodes(family_index: int, partition_index: int) -> tuple[str, ...]:
    base = 7000 + family_index * 300 + partition_index * 20
    return tuple(f"Zerin{base + offset:04d}" for offset in range(8))


def build_graph() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    units: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    case_index = 0
    for family_index, family in enumerate(GRAPH_FAMILIES):
        for partition_index, partition in enumerate(PARTITIONS):
            natural = GRAPH_BASES[family][partition_index]
            for world in WORLDS:
                nodes = natural if world == "natural" else artificial_nodes(family_index, partition_index)
                if world == "counterfactual":
                    nodes = graph_base.counterfactual_nodes(natural)
                unit = {
                    "arm": "graph",
                    "unit_id": f"c102g-{family}-{partition}-{world}",
                    "family": family,
                    "world": world,
                    "partition": partition,
                    "surface": "forward_order" if partition_index % 2 == 0 else "reverse_order",
                    "nodes": list(nodes),
                }
                units.append(unit)
                for x, y, branch, code in itertools.product((1, -1), repeat=4):
                    prompt, edges = graph_base.build_prompt(family, nodes, x, y, branch, code, unit["surface"])
                    follows, path_count = graph_base.reachable(edges, nodes[0], nodes[2])
                    truth = x == y
                    if follows != truth or path_count != int(truth):
                        raise RuntimeError((unit["unit_id"], x, y, edges, follows, path_count))
                    output_yes = truth if code == 1 else not truth
                    cases.append({
                        **unit,
                        "case_id": f"c102g-{case_index:04d}",
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
            partition = PARTITIONS[unit_index // 3]
            unit = {
                "arm": "breadth",
                "unit_id": f"c102b-{family}-{unit_index:02d}",
                "family": family,
                "world": "controlled_natural",
                "partition": partition,
                "surface": "factorial",
                "values": list(values),
            }
            units.append(unit)
            for truth, surface, distractor, code in itertools.product((1, -1), repeat=4):
                prompt, focus, anchor = c101_contract.breadth_prompt(family, values, truth, surface, distractor, code)
                output_yes = (truth == 1) if code == 1 else (truth != 1)
                cases.append({
                    **unit,
                    "case_id": f"c102b-{case_index:04d}",
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


def design_rank() -> int:
    cells = list(itertools.product((1, -1), repeat=4))
    columns = []
    for cell in cells:
        columns.append([1, *[np.prod([cell[index] for index in subset]) for size in range(1, 5) for subset in itertools.combinations(range(4), size)]])
    return int(np.linalg.matrix_rank(np.asarray(columns, dtype=np.float64)))


def prepare() -> None:
    if OUT.exists():
        raise RuntimeError(f"C102 already exists: {OUT}")
    c101 = core.load(C101 / "analysis/final.json")
    graph_units, graph_cases = build_graph()
    breadth_units, breadth_cases = build_breadth()
    tok = graph_base.tokenizer()
    graph_compiled = [{**row, "arm": "graph"} for row in graph_base.compile_rows(tok, graph_cases)]
    breadth_compiled = [{**row, "arm": "breadth"} for row in c101_contract.compile_breadth(tok, breadth_cases)]
    compiled = [*graph_compiled, *breadth_compiled]
    old_words = {
        token.casefold()
        for path in (C101 / "material/confirmation_units.jsonl", C101 / "material/breadth_units.jsonl")
        for row in core.rows(path)
        for value in (row.get("nodes") or row.get("values") or [])
        for token in [str(value)]
    }
    new_words = {
        token.casefold()
        for row in [*graph_units, *breadth_units]
        for value in (row.get("nodes") or row.get("values") or [])
        for token in [str(value)]
    }
    max_width = max(len(row["prompt_ids"]) for row in compiled)
    total_valid_tokens = sum(len(row["prompt_ids"]) for row in compiled)
    expected_bytes = 37 * total_valid_tokens * 2560 * 2
    graph_zero = c101_contract.zero_models(graph_cases, False)
    breadth_zero = c101_contract.zero_models(breadth_cases, True)
    checks = {
        "c101_complete": c101["all_checks_passed"],
        "graph_units": len(graph_units) == 36,
        "graph_cases": len(graph_cases) == 576,
        "breadth_units": len(breadth_units) == 36,
        "breadth_cases": len(breadth_cases) == 576,
        "partitions_graph": Counter(row["partition"] for row in graph_units) == {partition: 12 for partition in PARTITIONS},
        "partitions_breadth": Counter(row["partition"] for row in breadth_units) == {partition: 12 for partition in PARTITIONS},
        "factor_rank": design_rank() == 16,
        "graph_balance": all(Counter(row[factor] for row in graph_cases) == {1: 288, -1: 288} for factor in graph_base.FACTORS),
        "breadth_balance": all(Counter(row[factor] for row in breadth_cases) == {1: 288, -1: 288} for factor in ("truth_factor", "surface_factor", "distractor_factor", "code")),
        "truth_paths": all(((row["path_count"] == 1) == row["truth"]) and (row["path_count"] in (0, 1)) for row in graph_cases),
        "output_balance": Counter(row["output_yes"] for row in compiled) == {True: 576, False: 576},
        "zero_graph": all(value == 0.5 for key, value in graph_zero.items() if key != "truth_x_code_oracle") and graph_zero["truth_x_code_oracle"] == 1.0,
        "zero_breadth": all(value == 0.5 for key, value in breadth_zero.items() if key != "truth_x_code_oracle") and breadth_zero["truth_x_code_oracle"] == 1.0,
        "fresh_lexicon": not (old_words & new_words),
        "compiled": len(compiled) == 1152,
        "width": max_width < 320,
        "storage": expected_bytes < 40 * 1024 ** 3,
    }
    if not all(checks.values()):
        raise RuntimeError({"checks": checks, "overlap": sorted(old_words & new_words)})
    protocol = {
        "schema": "c102.typed_relation_all_token_coordinate_field.v1",
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "created_at_utc": now(),
        "objective": "discover and prospectively validate per-coordinate relative responses, reuse-difference barcodes, and token-by-layer formation trajectories",
        "scientific_order": ["observe_existing_field", "freeze_barcode", "capture_fresh_all_token_field", "reveal_partitions", "conditional_intervention"],
        "model": "Qwen3-4B local CUDA BF16 nonquantized",
        "materials": {
            "graph": {"families": list(GRAPH_FAMILIES), "worlds": list(WORLDS), "units": 36, "cases": 576, "factors": list(graph_base.FACTORS), "primary_effect": "xy"},
            "breadth": {"families": list(BREADTH_FAMILIES), "units": 36, "cases": 576, "factors": ["truth", "surface", "distractor", "code"], "primary_effect": "truth"},
            "partitions": list(PARTITIONS),
            "human_naturalness": {"blind_review": False, "missingness": "M_HUMAN_NATURALNESS"},
        },
        "storage": {
            "scope": "all valid tokens, embedding plus all 36 Hidden States, all 2560 activation coordinates",
            "archive_dtype": "uint16 exact BF16 bits",
            "analysis_dtype": "float32 decode with float64 accumulation",
            "fixed_physical_width": max_width,
            "total_valid_tokens": total_valid_tokens,
            "expected_raw_bytes": expected_bytes,
            "batch_size": 8,
        },
        "discovery": {
            "source": "immutable C101 coefficient fields",
            "barcode": "coordinate-wise primary/code/primary-code effects with Rademacher sign-flip null dead zones",
            "formation": "layer finite differences of registered effect trajectories",
            "rule": "no preset coordinate percentage, dimensionality, cluster count, manifold, group, or fiber structure",
        },
        "validation": {
            "fresh_material": True,
            "reveal_order": list(PARTITIONS),
            "frozen_objects": ["family", "effect", "role", "state", "coordinate", "sign", "formation event"],
            "adjudication": "each family/effect prediction is typed separately; failures do not stop unrelated routes",
        },
        "numeric_gates": {"repeat_hidden_bitwise": True, "repeat_logit_max_abs": 1e-6, "causal_prefix_bitwise": True, "code_previsible_bitwise": True},
        "intervention_authorization": "only if at least one frozen coordinate barcode repeats in both fresh confirmation and lockbox beyond its design-null threshold",
        "claim_boundary": {
            "allowed": "single-Qwen activation-coordinate observations, prospective barcode validation, and conditionally authorized state-level intervention",
            "forbidden": ["weight parameter mechanism", "semantic neuron", "attention or MLP circuit", "cross-model coordinate identity", "new mathematics", "complete language mechanism"],
        },
        "analysis_correction": {
            "retained": ["all-coordinate preservation", "controlled factorials", "effect trajectories", "observation before freeze"],
            "corrected": ["activation coordinate is not a model parameter", "raw sign co-occurrence is not collaboration", "Walsh requires full-rank factorial cells", "reuse-difference averages are descriptive until intervention", "thresholds come from numeric and design nulls rather than arbitrary percentages"],
        },
        "authorization": "run_phase1582_c102_c101_field_discovery",
    }
    protocol["producer_sha256"] = core.sha(Path(__file__))
    protocol["material_digest"] = core.digest({"graph": graph_cases, "breadth": breadth_cases})
    core.write_rows(OUT / "material/graph_units.jsonl", graph_units)
    core.write_rows(OUT / "material/graph_cases.jsonl", graph_cases)
    core.write_rows(OUT / "material/breadth_units.jsonl", breadth_units)
    core.write_rows(OUT / "material/breadth_cases.jsonl", breadth_cases)
    core.write_rows(OUT / "compiled/qwen3_graph.jsonl", graph_compiled)
    core.write_rows(OUT / "compiled/qwen3_breadth.jsonl", breadth_compiled)
    core.write_rows(OUT / "material/frozen_examples.jsonl", [graph_cases[0], *[next(row for row in breadth_cases if row["family"] == family) for family in BREADTH_FAMILIES]])
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "audit/pre_model_material_semantic_audit.json", {
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "all_checks_passed": all(checks.values()),
        "factorial_rank": design_rank(),
        "zero_models": {"graph": graph_zero, "breadth": breadth_zero},
        "storage": protocol["storage"],
        "authorization": protocol["authorization"],
    })
    print(json.dumps({"checks": checks, "storage": protocol["storage"], "authorization": protocol["authorization"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    prepare()
