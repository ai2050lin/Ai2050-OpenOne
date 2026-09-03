#!/usr/bin/env python3
"""Phase1589 / C104: fresh breadth-material contract for frozen upstream candidates."""
from __future__ import annotations

import itertools
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
C101 = RESULT / "phase1575_c101_dual_arm"
C102 = RESULT / "phase1581_c102_typed_relation_coordinate_campaign"
C103 = RESULT / "phase1588_c103_code_residualized_role_state_atlas"
OUT = RESULT / "phase1589_c104_upstream_candidate_validation"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core
import phase1571_c098_observation_first_graph_campaign as graph_base
import phase1575_c101_dual_arm_contract as breadth_base

PHASE = 1589
CAMPAIGN = "C104"
PARTITIONS = graph_base.PARTITIONS
FAMILIES = breadth_base.BREADTH_FAMILIES

ATTRIBUTE_UNITS = [
    ("sextant", "cobalt", "nacre", "derrick", "wharfage"), ("stormlamp", "carmine", "eggshell", "parapet", "cloister"),
    ("tureen", "bluegreen", "wine-red", "capstan", "jetty"), ("brooch", "coppertoned", "mauve", "pergola", "promenade"),
    ("crucible", "mustard", "turquoise", "hoist", "smelter"), ("valise", "fuchsia", "khaki", "rudder", "shipyard"),
    ("goggles", "honeygold", "argent", "viaduct", "causeway"), ("cassette", "ruby", "obsidian", "propeller", "airshed"),
    ("slatepad", "midnightblue", "graphite", "truss", "vestibule"),
]
AGENT_UNITS = [
    ("Adair", "Briony", "Cedric", "Daphne", "accompanied"), ("Edgar", "Fiona", "Gareth", "Hollis", "briefed"),
    ("Isla", "Joric", "Kendra", "Leith", "alerted"), ("Mirel", "Noel", "Orla", "Piers", "coached"),
    ("Riona", "Silas", "Tamsin", "Uriah", "invited"), ("Viona", "Waldo", "Yvette", "Zelda", "supported"),
    ("Abram", "Bonnie", "Conrad", "Dinah", "interviewed"), ("Elton", "Frida", "Gavin", "Honor", "welcomed"),
    ("Irene", "Jorah", "Karla", "Leland", "messaged"),
]
NEGATION_UNITS = [
    ("Arox", "damp", "Bren", "hinged", "armed"), ("Cevan", "concave", "Drex", "painted", "etched"),
    ("Elda", "flexible", "Faris", "planar", "arrayed"), ("Gavinor", "permeable", "Hedra", "cached", "counted"),
    ("Iria", "brittle", "Jorin", "banded", "numbered"), ("Kelda", "conductive", "Lorin", "riveted", "scheduled"),
    ("Mavin", "fusible", "Neris", "soldered", "forwarded"), ("Orin", "radiant", "Prya", "creased", "stamped"),
    ("Quen", "occupied", "Rella", "gripped", "charted"),
]
EXCEPTION_UNITS = [
    ("unitU1", "unitU2", "packingring", "circulator", "shell"), ("unitU3", "unitU4", "cleat", "rack", "beam"),
    ("unitU5", "unitU6", "relay", "switchboard", "connectorblock"), ("unitU7", "unitU8", "impeller", "powerplant", "shroud"),
    ("unitU9", "unitU10", "aperture", "opticalunit", "stand"), ("unitU11", "unitU12", "buckle", "cabinetry", "laminate"),
    ("unitU13", "unitU14", "bobber", "tank", "indicatorwheel"), ("unitU15", "unitU16", "stopcock", "organ", "spouttip"),
    ("unitU17", "unitU18", "platen", "copier", "drawer"),
]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def design_rank() -> int:
    cells = list(itertools.product((1, -1), repeat=4))
    matrix = []
    for cell in cells:
        matrix.append([1, *[np.prod([cell[index] for index in subset]) for size in range(1, 5) for subset in itertools.combinations(range(4), size)]])
    return int(np.linalg.matrix_rank(np.asarray(matrix, dtype=np.float64)))


def build() -> tuple[list[dict], list[dict]]:
    inventories = {"attribute_binding": ATTRIBUTE_UNITS, "agent_patient": AGENT_UNITS, "negation_scope": NEGATION_UNITS, "whole_part_exception": EXCEPTION_UNITS}
    units, cases = [], []
    case_index = 0
    for family in FAMILIES:
        for unit_index, values in enumerate(inventories[family]):
            partition = PARTITIONS[unit_index // 3]
            unit = {"arm": "breadth", "unit_id": f"c104-{family}-{unit_index:02d}", "family": family, "world": "controlled_natural", "partition": partition, "surface": "factorial", "values": list(values)}
            units.append(unit)
            for truth, surface, distractor, code in itertools.product((1, -1), repeat=4):
                prompt, focus, anchor = breadth_base.breadth_prompt(family, values, truth, surface, distractor, code)
                output_yes = (truth == 1) if code == 1 else (truth != 1)
                cases.append({**unit, "case_id": f"c104-{case_index:04d}", "truth_factor": truth, "surface_factor": surface, "distractor_factor": distractor, "code": code, "codebook": graph_base.CODEBOOKS[code]["name"], "truth": truth == 1, "output_yes": output_yes, "gold_position": 0 if output_yes else 1, "focus": focus, "anchor": anchor, "prompt": prompt})
                case_index += 1
    return units, cases


def old_values() -> set[str]:
    paths = [C101 / "material/breadth_units.jsonl", C102 / "material/breadth_units.jsonl"]
    return {str(value).casefold() for path in paths for row in core.rows(path) for value in row["values"]}


def main() -> None:
    if OUT.exists():
        raise RuntimeError(f"C104 already exists: {OUT}")
    c103 = core.load(C103 / "analysis/final.json")
    c103_audit = core.load(C103 / "audit/independent_final_audit.json")
    if c103["authorization"] != "append_phase1588_c103_memo_and_preregister_fresh_validation_only_if_candidate_is_scientifically_useful" or not c103_audit["all_checks_passed"]:
        raise RuntimeError("C104 authorization missing")
    units, cases = build()
    tok = graph_base.tokenizer()
    compiled = breadth_base.compile_breadth(tok, cases)
    new_values = {str(value).casefold() for row in units for value in row["values"]}
    zero = breadth_base.zero_models(cases, True)
    max_width = max(len(row["prompt_ids"]) for row in compiled)
    total_tokens = sum(len(row["prompt_ids"]) for row in compiled)
    candidates = {row["family"]: row for row in c103["candidates"] if row["arm"] == "breadth"}
    source_breadth = np.load(C103 / "raw/breadth_residual_vectors.float32.npy", mmap_mode="r")
    barcodes = np.zeros((4, 2560), dtype=np.float32)
    prediction_rows = []
    for family_index, family in enumerate(FAMILIES):
        candidate = candidates[family]
        barcodes[family_index] = np.asarray(source_breadth[family_index, candidate["role_index"], candidate["state"]], dtype=np.float64).mean(axis=0).astype(np.float32)
        prediction_rows.append({"family": family, "role": candidate["role"], "role_index": candidate["role_index"], "state": candidate["state"], "source_minimum_cosine": candidate["minimum_residual_cosine"], "coordinates": "all_2560"})
    barcode_path = OUT / "raw/frozen_c103_upstream_barcodes.float32.npy"
    barcode_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(barcode_path, barcodes)
    checks = {
        "c103": c103_audit["all_checks_passed"],
        "units": len(units) == 36,
        "cases": len(cases) == 576,
        "partitions": Counter(row["partition"] for row in units) == {partition: 12 for partition in PARTITIONS},
        "rank": design_rank() == 16,
        "balance": Counter((row["truth_factor"], row["surface_factor"], row["distractor_factor"], row["code"]) for row in cases) == {cell: 36 for cell in itertools.product((1, -1), repeat=4)},
        "zero": all(abs(value - 0.5) < 1e-12 for key, value in zero.items() if key != "truth_x_code_oracle") and zero["truth_x_code_oracle"] == 1.0,
        "fresh": not (old_values() & new_values),
        "compiled": len(compiled) == 576 and all(len(row["candidate_ids"]) == 2 for row in compiled),
        "width": max_width < 224,
        "predictions": bool(len(prediction_rows) == 4 and barcodes.shape == (4, 2560) and np.isfinite(barcodes).all()),
    }
    if not all(checks.values()):
        raise RuntimeError({"checks": checks, "overlap": sorted(old_values() & new_values)})
    core.write_rows(OUT / "material/units.jsonl", units)
    core.write_rows(OUT / "material/cases.jsonl", cases)
    core.write_rows(OUT / "compiled/qwen3.jsonl", compiled)
    core.write_rows(OUT / "material/frozen_examples.jsonl", [cases[index] for index in (0, 144, 288, 432)])
    material_digest = core.digest([*units, *cases])
    protocol = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "created_at_utc": now(),
        "status": "fresh_upstream_candidate_validation_frozen",
        "model": "Qwen3-4B local BF16 CUDA nonquantized",
        "materials": {"families": list(FAMILIES), "units": 36, "cases": 576, "partitions": list(PARTITIONS), "factorial_rank": 16, "lexically_disjoint_from": ["C101", "C102"]},
        "predictions": prediction_rows,
        "barcode_path": str(barcode_path.relative_to(ROOT)).replace("\\", "/"),
        "barcode_sha256": core.sha(barcode_path),
        "validation": {"effect": "truth", "coordinates": 2560, "confirmation_and_lockbox": "cosine exceeds coordinate-permutation q99 with 2000 draws; no role/state reselection", "seeds": {"confirmation": 1591, "lockbox": 1592}},
        "storage": {"scope": "all valid tokens, all 37 states, all 2560 activation coordinates", "dtype": "uint16 exact BF16 bits", "fixed_width": 224, "batch_size": 8, "total_valid_tokens": total_tokens, "expected_bytes": 37 * total_tokens * 2560 * 2},
        "behavior": "reported separately; it does not stop internal observation",
        "numeric": {"repeat_bitwise": True, "causal_prefix_bitwise": True, "code_previsible_bitwise": True},
        "human_naturalness": "M_HUMAN_NATURALNESS",
        "claim_boundary": "fresh single-Qwen upstream activation-field validation only; not semantic neurons, weights, cross-model law or new mathematics",
        "material_digest": material_digest,
        "producer_sha256": core.sha(Path(__file__)),
        "authorization": "run_phase1590_c104_qwen_capture",
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    audit = {"phase": PHASE, "campaign": CAMPAIGN, "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "zero_models": zero, "max_width": max_width, "total_valid_tokens": total_tokens, "expected_bytes": protocol["storage"]["expected_bytes"], "authorization": protocol["authorization"]}
    core.save(OUT / "audit/pre_model_material_audit.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
