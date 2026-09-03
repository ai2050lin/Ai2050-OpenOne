#!/usr/bin/env python3
"""Phase1607 / C110: freeze a third-lexicon test of readout/control separation."""
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
C108 = RESULT / "phase1600_c108_fresh_coordinate_causality"
C109 = RESULT / "phase1603_c109_fresh_role_state_field_atlas"
OUT = RESULT / "phase1607_c110_fresh_readout_control_separation"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core
import phase1571_c098_observation_first_graph_campaign as graph_base
import phase1575_c101_dual_arm_contract as breadth_base

PHASE = 1607
CAMPAIGN = "C110"
FAMILIES = ("attribute_binding", "agent_patient")
PARTITIONS = ("fresh_confirmation", "fresh_lockbox")
ROLES = ("focus_pre", "focus_record", "focus_post", "query_focus", "query_anchor", "code_instruction", "boundary")
STATES = 37
DIM = 2560
WIDTH = 224
BATCH_SIZE = 8

ATTRIBUTE_UNITS = [
    ("actinometer", "caputmortuum", "atacamite", "bathhouse", "machicolation"),
    ("aerometer", "indigotine", "bleudefrance", "chapterhouse", "bartizan"),
    ("bolometer", "cinnabar", "brunswickgreen", "crenellation", "campanile"),
    ("cathetometer", "drab", "catawba", "dovecote", "drawbridge"),
    ("dynamometer", "fulvous", "davygrey", "forecourt", "embrasure"),
    ("ellipsometer", "livid", "falu", "gatehouse", "flyingbuttress"),
    ("gravimeter", "mazarine", "limerick", "hermitage", "saltbox"),
    ("interferometer", "murexide", "moonstoneblue", "icehouse", "hostelry"),
    ("magnetometer", "orpiment", "persiangreen", "lighthousekeep", "infirmary"),
    ("nephelometer", "sinopia", "reseda", "munimentroom", "sallyport"),
    ("pycnometer", "terreverte", "rufous", "navecrossing", "keepwall"),
    ("refractometer", "vermilion", "sangria", "vestry", "loggia"),
]

AGENT_UNITS = [
    ("Alaric", "Beatrix", "Cyprian", "Desdemona", "admonished"),
    ("Ephraim", "Fiora", "Godfrey", "Hermione", "forewarned"),
    ("Ignatius", "Jessamine", "Kilian", "Lavinia", "ushered"),
    ("Maximilian", "Nerissa", "Oberon", "Petronilla", "convoked"),
    ("Quentin", "Rhiannon", "Sylvester", "Theodora", "catechized"),
    ("Urban", "Vivienne", "Wystan", "Xanthe", "reproached"),
    ("Yorick", "Adelina", "Basilio", "Clarimond", "reconnoitered"),
    ("Dorian", "Eulalia", "Florian", "Griselda", "instructed"),
    ("Hector", "Ianthe", "Jareth", "Kallista", "observed"),
    ("Leander", "Morgana", "Nestor", "Oriana", "interrogated"),
    ("Prospero", "Regina", "Sebastian", "Tamaris", "importuned"),
    ("Vernon", "Winifred", "Xavier", "Yseult", "waylaid"),
]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build() -> tuple[list[dict], list[dict]]:
    inventories = {"attribute_binding": ATTRIBUTE_UNITS, "agent_patient": AGENT_UNITS}
    units, cases = [], []
    for family in FAMILIES:
        for unit_index, values in enumerate(inventories[family]):
            partition = PARTITIONS[unit_index // 6]
            unit = {"arm": "breadth", "unit_id": f"c110-{family}-{unit_index:02d}", "family": family, "world": "controlled_natural", "partition": partition, "surface": "factorial", "values": list(values)}
            units.append(unit)
            for truth, surface, distractor, code in itertools.product((1, -1), repeat=4):
                prompt, focus, anchor = breadth_base.breadth_prompt(family, values, truth, surface, distractor, code)
                output_yes = (truth == 1) == (code == 1)
                cases.append({
                    **unit,
                    "case_id": f"c110-{len(cases):04d}",
                    "truth_factor": truth,
                    "surface_factor": surface,
                    "distractor_factor": distractor,
                    "code": code,
                    "codebook": graph_base.CODEBOOKS[code]["name"],
                    "truth": truth == 1,
                    "output_yes": output_yes,
                    "gold_position": 0 if output_yes else 1,
                    "focus": focus,
                    "anchor": anchor,
                    "prompt": prompt,
                })
    return units, cases


def historical_values() -> set[str]:
    paths = [
        C101 / "material/breadth_units.jsonl",
        C102 / "material/breadth_units.jsonl",
        C104 / "material/units.jsonl",
        C108 / "material/units.jsonl",
    ]
    return {str(value).casefold() for path in paths for row in core.rows(path) for value in row["values"]}


def main() -> None:
    if OUT.exists():
        raise RuntimeError(f"C110 already exists: {OUT}")
    closure = core.load(C109 / "analysis/closure.json")
    source_audit = core.load(C109 / "audit/independent_closure_audit.json")
    c109_protocol = core.load(C109 / "protocol/preregistration.json")
    if not source_audit["all_checks_passed"] or not closure["next_authorization"].startswith("C110 fresh lexical"):
        raise RuntimeError("C110 authorization missing")
    units, cases = build()
    tok = graph_base.tokenizer()
    compiled = breadth_base.compile_breadth(tok, cases)
    zero = breadth_base.zero_models(cases, True)
    old = historical_values()
    fresh = [str(value).casefold() for row in units for value in row["values"]]
    occurrences = []
    disjoint = True
    for row_index, row in enumerate(compiled):
        occupied = []
        for role in ROLES:
            positions = [int(value) for value in row["role_positions"][role]]
            occupied.extend(positions)
            for subtoken, position in enumerate(positions):
                token_id = int(row["prompt_ids"][position])
                occurrences.append({
                    "occurrence_index": len(occurrences), "row_index": row_index, "case_id": row["case_id"], "unit_id": row["unit_id"],
                    "family": row["family"], "partition": row["partition"], "truth_factor": row["truth_factor"],
                    "surface_factor": row["surface_factor"], "distractor_factor": row["distractor_factor"], "code": row["code"],
                    "role": role, "subtoken": subtoken, "span_length": len(positions), "token_position": position,
                    "token_id": token_id, "token_text": tok.convert_ids_to_tokens([token_id])[0],
                })
        disjoint = disjoint and len(occupied) == len(set(occupied))
    semantic_cells = Counter((row["family"], row["partition"], row["truth_factor"], row["surface_factor"], row["distractor_factor"], row["code"]) for row in cases)
    supports = c109_protocol["supports"]
    checks = {
        "authorization": source_audit["all_checks_passed"],
        "units": len(units) == 24,
        "cases": len(cases) == 384,
        "partitions": Counter((row["family"], row["partition"]) for row in units) == {(family, partition): 6 for family in FAMILIES for partition in PARTITIONS},
        "factorial": semantic_cells == {(family, partition, *cell): 6 for family in FAMILIES for partition in PARTITIONS for cell in itertools.product((1, -1), repeat=4)},
        "zero_models": all(abs(value - 0.5) < 1e-12 for key, value in zero.items() if key != "truth_x_code_oracle") and zero["truth_x_code_oracle"] == 1.0,
        "semantic_uniqueness": len({row["prompt"] for row in cases}) == 384 and all(row["output_yes"] == ((row["truth_factor"] == 1) == (row["code"] == 1)) for row in cases),
        "freshness": not (set(fresh) & old),
        "value_uniqueness": len(fresh) == len(set(fresh)),
        "compiled": len(compiled) == 384 and all(row["candidate_ids"] == [[9834], [902]] for row in compiled),
        "roles": all(set(row["role_positions"]) == set(ROLES) for row in compiled) and disjoint,
        "width": max(len(row["prompt_ids"]) for row in compiled) < WIDTH,
        "supports": len(supports["attribute_binding_k256"]) == 256 and len(supports["agent_patient_k128"]) == 128,
        "machine_naturalness": all(row["prompt"].count("Query:") == 1 and row["prompt"].endswith("Reply exactly yes or no.") for row in cases),
    }
    if not all(checks.values()):
        raise RuntimeError({"checks": checks, "overlap": sorted(set(fresh) & old)})
    core.write_rows(OUT / "material/units.jsonl", units)
    core.write_rows(OUT / "material/cases.jsonl", cases)
    core.write_rows(OUT / "compiled/qwen3.jsonl", compiled)
    core.write_rows(OUT / "protocol/role_occurrence_manifest.jsonl", occurrences)
    protocol = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "created_at_utc": now(),
        "status": "fresh_readout_control_separation_contract_frozen",
        "model": "Qwen3-4B local BF16 CUDA nonquantized",
        "object": "prospective separation of full truth-field stability from fixed-support output leverage",
        "families": list(FAMILIES),
        "partitions": list(PARTITIONS),
        "units": 24,
        "cases": 384,
        "roles": list(ROLES),
        "states": STATES,
        "activation_coordinates": DIM,
        "occurrences": len(occurrences),
        "archive": {"path": "raw/qwen3_role_subtoken_all_states.uint16.npy", "shape": [STATES, len(occurrences), DIM], "dtype": "uint16 exact BF16 bit patterns", "fixed_width": WIDTH, "batch_size": BATCH_SIZE},
        "supports": supports,
        "frozen_field_prediction": {
            "role": "query_anchor",
            "state": 19,
            "cross_fresh_partition_cosine_min": 0.90,
            "each_fresh_partition_to_c109_reference_cosine_min": 0.85,
            "each_partition_frozen_support_topk_overlap_min": 0.50,
            "families": list(FAMILIES),
        },
        "frozen_leverage_prediction": {
            "attribute_binding": "median target gain per L2 exceeds same-K wrong support in all four partition-by-code cells",
            "agent_patient": "median target gain per L2 is below same-K wrong support in all four partition-by-code cells",
        },
        "write_modes": ["frozen_support", "wrong_same_k", "wrong_l2_matched", "coordinate_permuted", "whole_query_anchor", "whole_query_anchor_plus_focus_record"],
        "energy_match": "per pair, scale the wrong-support donor-recipient delta to exactly the target-support L2 movement; do not change support membership",
        "multi_role": {"state": 19, "roles": ["query_anchor", "focus_record"], "coordinates": "all_2560"},
        "behavior_policy": "stratify standard and reversed code; reversed failure does not stop upstream observation because code is causally later than registered pre-code roles",
        "completion_rule": "complete exact field capture, frozen field prediction adjudication, all frozen transport modes, and independent audits; route-level failures do not stop the campaign",
        "typed_missingness": {"human_naturalness": "no independent blind rating", "cross_model": "Qwen3 only"},
        "claim_boundary": "third controlled-English lexicon; activation coordinates are not weights, neurons, or universal semantic atoms",
        "source": {"c109_raw_sha256": core.load(C109 / "analysis/capture_summary.json")["raw_sha256"], "c109_mean_sha256": core.sha(C109 / "analysis/mean_truth_role_state.float32.npy")},
        "material_digest": core.digest([*units, *cases]),
        "producer_sha256": core.sha(Path(__file__)),
        "authorization": "execute_phase1608_c110_exact_field_capture",
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    audit = {"phase": PHASE, "campaign": CAMPAIGN, "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "zero_models": zero, "occurrences": len(occurrences), "max_width": max(len(row["prompt_ids"]) for row in compiled), "material_digest": protocol["material_digest"], "authorization": protocol["authorization"]}
    core.save(OUT / "audit/pre_model_audit.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
