#!/usr/bin/env python3
"""Phase1600 / C108: freeze a fresh lexical test of the C106 discovery supports."""
from __future__ import annotations

import itertools
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
C101 = RESULT / "phase1575_c101_dual_arm"
C102 = RESULT / "phase1581_c102_typed_relation_coordinate_campaign"
C104 = RESULT / "phase1589_c104_upstream_candidate_validation"
C106 = RESULT / "phase1596_c106_minimal_coordinate_coalition"
C107 = RESULT / "phase1599_c107_code_aware_dual_readout_adjudication"
OUT = RESULT / "phase1600_c108_fresh_coordinate_causality"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core
import phase1571_c098_observation_first_graph_campaign as graph_base
import phase1575_c101_dual_arm_contract as breadth_base

PHASE = 1600
CAMPAIGN = "C108"
FAMILIES = ("attribute_binding", "agent_patient")
PARTITIONS = ("prospective_confirmation", "independent_lockbox")
FROZEN_K = {"attribute_binding": 256, "agent_patient": 128}

ATTRIBUTE_UNITS = [
    ("anemometer", "alizarin", "celadon", "belfry", "caravansary"),
    ("ophthalmoscope", "ecru", "gamboge", "cupola", "esplanade"),
    ("theodolite", "heliotrope", "isabelline", "gravingdock", "hangar"),
    ("hydrometer", "puce", "viridian", "orangery", "pavilion"),
    ("planimeter", "wenge", "xanthic", "quayhouse", "refectory"),
    ("spectroscope", "zaffre", "amaranth", "rotunda", "semaphoretower"),
    ("phonograph", "aubergine", "bistre", "tollhouse", "transept"),
    ("stereoscope", "coquelicot", "feldgrau", "trestlebridge", "turntablehouse"),
    ("galvanometer", "glaucous", "jonquil", "wharfhouse", "abbeygate"),
    ("periscope", "mikadoyellow", "nankeen", "enginehouse", "ferryhouse"),
    ("velocimeter", "paynesgrey", "smalt", "guardhouse", "maltings"),
    ("heliograph", "tyrianpurple", "verdigris", "pumpstation", "ropewalk"),
]

AGENT_UNITS = [
    ("Aeneas", "Berenice", "Cassian", "Dorothea", "corroborated"),
    ("Evander", "Fenella", "Gawain", "Hyacinth", "petitioned"),
    ("Isidore", "Jocasta", "Lysander", "Melisande", "debriefed"),
    ("Nicodemus", "Ophelia", "Peregrine", "Rowena", "chaperoned"),
    ("Septimus", "Thisbe", "Ulysses", "Valeria", "counseled"),
    ("Wolfgang", "Zenobia", "Ambrose", "Cordelia", "telephoned"),
    ("Erasmus", "Genevieve", "Horatio", "Isolde", "congratulated"),
    ("Jeremiah", "Kerensa", "Lorenzo", "Minerva", "apprised"),
    ("Nathaniel", "Octavia", "Percival", "Rosalind", "reprimanded"),
    ("Simeon", "Tabitha", "Valentine", "Wilhelmina", "commended"),
    ("Zachariah", "Anthea", "Barnabas", "Cressida", "befriended"),
    ("Demetrius", "Eudora", "Ferdinand", "Guinevere", "cautioned"),
]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build() -> tuple[list[dict], list[dict]]:
    inventories = {"attribute_binding": ATTRIBUTE_UNITS, "agent_patient": AGENT_UNITS}
    units, cases = [], []
    case_index = 0
    for family in FAMILIES:
        for unit_index, values in enumerate(inventories[family]):
            partition = PARTITIONS[unit_index // 6]
            unit = {
                "arm": "breadth", "unit_id": f"c108-{family}-{unit_index:02d}", "family": family,
                "world": "controlled_natural", "partition": partition, "surface": "factorial", "values": list(values),
            }
            units.append(unit)
            for truth, surface, distractor, code in itertools.product((1, -1), repeat=4):
                prompt, focus, anchor = breadth_base.breadth_prompt(family, values, truth, surface, distractor, code)
                output_yes = (truth == 1) if code == 1 else (truth != 1)
                cases.append({
                    **unit, "case_id": f"c108-{case_index:04d}", "truth_factor": truth,
                    "surface_factor": surface, "distractor_factor": distractor, "code": code,
                    "codebook": graph_base.CODEBOOKS[code]["name"], "truth": truth == 1,
                    "output_yes": output_yes, "gold_position": 0 if output_yes else 1,
                    "focus": focus, "anchor": anchor, "prompt": prompt,
                })
                case_index += 1
    return units, cases


def historical_values() -> set[str]:
    paths = [
        C101 / "material/breadth_units.jsonl",
        C102 / "material/breadth_units.jsonl",
        C104 / "material/units.jsonl",
    ]
    return {str(value).casefold() for path in paths for row in core.rows(path) for value in row["values"]}


def main() -> None:
    if OUT.exists():
        raise RuntimeError(f"C108 already exists: {OUT}")
    c107 = core.load(C107 / "analysis/final.json")
    c107_audit = core.load(C107 / "audit/independent_audit.json")
    c106 = core.load(C106 / "protocol/preregistration.json")
    if not c107_audit["all_checks_passed"] or not c107["next_authorization"].startswith("freeze the C106 supports"):
        raise RuntimeError("C108 authorization missing")
    units, cases = build()
    tok = graph_base.tokenizer()
    compiled = breadth_base.compile_breadth(tok, cases)
    zero = breadth_base.zero_models(cases, True)
    fresh_values = [str(value).casefold() for row in units for value in row["values"]]
    old_values = historical_values()
    predictions = {row["family"]: row for row in c106["predictions"]}
    rankings = {family: c106["rankings"][family] for family in FAMILIES}
    max_width = max(len(row["prompt_ids"]) for row in compiled)
    semantic_cells = Counter((row["family"], row["partition"], row["truth_factor"], row["surface_factor"], row["distractor_factor"], row["code"]) for row in cases)
    checks = {
        "authorization": c107_audit["all_checks_passed"],
        "units": len(units) == 24,
        "cases": len(cases) == 384,
        "partitions": Counter((row["family"], row["partition"]) for row in units) == {(family, partition): 6 for family in FAMILIES for partition in PARTITIONS},
        "factorial_balance": semantic_cells == {(family, partition, *cell): 6 for family in FAMILIES for partition in PARTITIONS for cell in itertools.product((1, -1), repeat=4)},
        "zero_models": all(abs(value - 0.5) < 1e-12 for key, value in zero.items() if key != "truth_x_code_oracle") and zero["truth_x_code_oracle"] == 1.0,
        "semantic_uniqueness": len({row["prompt"] for row in cases}) == len(cases) and all(row["output_yes"] == ((row["truth_factor"] == 1) == (row["code"] == 1)) for row in cases),
        "lexical_freshness": not (set(fresh_values) & old_values),
        "value_uniqueness": len(fresh_values) == len(set(fresh_values)),
        "compiled": len(compiled) == len(cases) and all(row["candidate_ids"] == [[9834], [902]] for row in compiled),
        "role_order": all(max(row["role_positions"]["query_anchor"]) < min(row["role_positions"]["code_instruction"]) < row["role_positions"]["boundary"][0] for row in compiled),
        "width": max_width < 224,
        "frozen_support": FROZEN_K == {"attribute_binding": 256, "agent_patient": 128} and all(sorted(rankings[family]) == list(range(2560)) for family in FAMILIES),
        "machine_naturalness": all(row["prompt"].count("Query:") == 1 and row["prompt"].endswith("Reply exactly yes or no.") for row in cases),
    }
    if not all(checks.values()):
        raise RuntimeError({"checks": checks, "overlap": sorted(set(fresh_values) & old_values)})
    core.write_rows(OUT / "material/units.jsonl", units)
    core.write_rows(OUT / "material/cases.jsonl", cases)
    core.write_rows(OUT / "compiled/qwen3.jsonl", compiled)
    protocol = {
        "phase": PHASE, "campaign": CAMPAIGN, "created_at_utc": now(),
        "status": "fresh_coordinate_causality_contract_frozen",
        "model": "Qwen3-4B local BF16 CUDA nonquantized",
        "object": "C106 discovery-ranked activation-coordinate supports in frozen upstream query_anchor@state19",
        "families": list(FAMILIES), "partitions": list(PARTITIONS), "units": len(units), "cases": len(cases),
        "predictions": [predictions[family] for family in FAMILIES],
        "frozen_k": FROZEN_K,
        "rankings": rankings,
        "coordinate_permutations": {family: c106["coordinate_permutations"][family] for family in FAMILIES},
        "candidate_order": ["yes", "no"],
        "readouts": {
            "raw_truth": "delta(Yes-minus-No)",
            "task_aligned": "code * delta(Yes-minus-No)",
            "truth_target": "patched Yes-minus-No > 0",
            "task_target": "code * patched Yes-minus-No > 0",
            "donor_recovery": "(patched-recipient)/(donor-recipient), descriptive and task-valid only when donor is task-correct",
        },
        "write_modes": ["frozen_support", "wrong_family_support", "sign_reversed", "same_truth", "coordinate_permuted", "whole_state"],
        "delete_modes": ["frozen_support", "wrong_family_support", "same_truth"],
        "adjudication": {
            "truth_direction_replication": "correct median raw gain > 0 and exceeds wrong-family, sign-reversed, same-truth, and coordinate-permuted controls in every partition x code cell",
            "task_aligned_replication": "correct median code-aligned gain > 0 and exceeds the same controls in every partition x code cell",
            "necessity_candidate": "correct median donor loss > 0 and exceeds wrong-family and same-truth deletion in every partition x code cell",
            "functional_metrics": "report exact target accuracy, flip rate, donor validity, and recovery ratio; no post-hoc pass threshold",
        },
        "typed_missingness": {"human_naturalness": "not independently blind-rated; machine naturalness and semantic compiler audit only"},
        "no_reselection": "no family, role, state, K, coordinate order, threshold, or control may change after model run",
        "claim_boundary": "fresh controlled-English Qwen activation-state test; coordinates are activations, not weights or neurons",
        "material_digest": core.digest([*units, *cases]),
        "producer_sha256": core.sha(Path(__file__)),
        "authorization": "execute_phase1601_c108_fresh_coordinate_interventions",
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    audit = {
        "phase": PHASE, "campaign": CAMPAIGN, "checks": checks, "passed": sum(checks.values()), "total": len(checks),
        "all_checks_passed": all(checks.values()), "zero_models": zero, "max_width": max_width,
        "material_digest": protocol["material_digest"], "authorization": protocol["authorization"],
    }
    core.save(OUT / "audit/pre_model_audit.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
