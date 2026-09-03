#!/usr/bin/env python3
"""Phase1618 / C113: freeze a fourth-lexicon field and role-lattice replication."""
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
SOURCE = RESULT / "phase1607_c110_fresh_readout_control_separation"
C112 = RESULT / "phase1615_c112_value_identity_role_lattice"
OUT = RESULT / "phase1618_c113_fourth_lexicon_role_lattice_replication"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core
import phase1571_c098_observation_first_graph_campaign as graph_base
import phase1575_c101_dual_arm_contract as breadth_base

PHASE = 1618
CAMPAIGN = "C113"
FAMILIES = ("attribute_binding", "agent_patient")
PARTITIONS = ("fourth_confirmation", "fourth_lockbox")
ROLES = ("focus_pre", "focus_record", "focus_post", "query_focus", "query_anchor", "code_instruction", "boundary")
PATH_ROLES = ("focus_record", "focus_post", "query_focus", "query_anchor")
STATES = 37
DIM = 2560
WIDTH = 224
BATCH_SIZE = 8

ATTRIBUTE_UNITS = [
    ("velometer", "asterblue", "norvane", "birchmark", "northarch"),
    ("radiscope", "emberred", "peltrix", "cedarmark", "southarch"),
    ("lumograph", "fernjade", "quorlan", "elmmark", "eastarch"),
    ("densograph", "cloudgray", "selmire", "firmark", "westarch"),
    ("fleximeter", "sunamber", "torvex", "hazelmark", "upperarch"),
    ("tonoscope", "duskviolet", "ulmarin", "ivymark", "lowerarch"),
    ("aerograph", "frostwhite", "velsora", "junipermark", "innerarch"),
    ("fluxmeter", "mossgreen", "windrel", "larchmark", "outerarch"),
    ("chromascope", "plumrose", "xandrel", "mapremark", "riverarch"),
    ("graviscope", "lakecyan", "yarrowin", "oakmark", "valleyarch"),
    ("thermograph", "stoneochre", "zephoran", "pinemark", "ridgearch"),
    ("wavegauge", "ashblack", "caldrix", "rowanmark", "harborarch"),
]

AGENT_UNITS = [
    ("Avelor", "Brineth", "Caldor", "Doreva", "relayed"),
    ("Eldren", "Fiorael", "Galmir", "Helina", "signalwrote"),
    ("Iveron", "Jaselle", "Keldric", "Lorana", "informed"),
    ("Mavren", "Norelia", "Orvian", "Phaedra", "directed"),
    ("Quenlor", "Ravena", "Seldric", "Tavira", "alertwrote"),
    ("Ulrican", "Virelle", "Wendric", "Xavena", "advicewrote"),
    ("Yorven", "Zarela", "Ardric", "Belora", "contacted"),
    ("Cyrven", "Delara", "Evaric", "Fendria", "addressed"),
    ("Gorlan", "Havira", "Ilvren", "Jorelia", "consultedwith"),
    ("Korven", "Lysara", "Meridan", "Nivelle", "communicated"),
    ("Orren", "Pavira", "Quelric", "Romena", "coordinated"),
    ("Sarven", "Tirelia", "Uldric", "Varelle", "corresponded"),
]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build() -> tuple[list[dict], list[dict]]:
    inventories = {"attribute_binding": ATTRIBUTE_UNITS, "agent_patient": AGENT_UNITS}
    units, cases = [], []
    for family in FAMILIES:
        for unit_index, values in enumerate(inventories[family]):
            partition = PARTITIONS[unit_index // 6]
            unit = {
                "arm": "breadth", "unit_id": f"c113-{family}-{unit_index:02d}", "family": family,
                "world": "controlled_synthetic_lexicon", "partition": partition, "surface": "factorial", "values": list(values),
            }
            units.append(unit)
            for truth, surface, distractor, code in itertools.product((1, -1), repeat=4):
                prompt, focus, anchor = breadth_base.breadth_prompt(family, values, truth, surface, distractor, code)
                output_yes = (truth == 1) == (code == 1)
                cases.append({
                    **unit, "case_id": f"c113-{len(cases):04d}", "truth_factor": truth,
                    "surface_factor": surface, "distractor_factor": distractor, "code": code,
                    "codebook": graph_base.CODEBOOKS[code]["name"], "truth": truth == 1,
                    "output_yes": output_yes, "gold_position": 0 if output_yes else 1,
                    "focus": focus, "anchor": anchor, "prompt": prompt,
                })
    return units, cases


def historical_values() -> set[str]:
    paths = [
        RESULT / "phase1575_c101_dual_arm/material/breadth_units.jsonl",
        RESULT / "phase1581_c102_typed_relation_coordinate_campaign/material/breadth_units.jsonl",
        RESULT / "phase1589_c104_upstream_candidate_validation/material/units.jsonl",
        RESULT / "phase1600_c108_fresh_coordinate_causality/material/units.jsonl",
        SOURCE / "material/units.jsonl",
    ]
    return {str(value).casefold() for path in paths for row in core.rows(path) for value in row["values"]}


def main() -> None:
    if OUT.exists():
        raise RuntimeError(f"C113 already exists: {OUT}")
    closure = core.load(C112 / "analysis/closure.json")
    closure_audit = core.load(C112 / "audit/independent_closure_audit.json")
    if not closure_audit["all_checks_passed"] or not closure["next_authorization"].startswith("C113 fourth-lexicon"):
        raise RuntimeError("C113 authorization missing")
    source_protocol = core.load(SOURCE / "protocol/preregistration.json")
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
                    "occurrence_index": len(occurrences), "row_index": row_index, "case_id": row["case_id"],
                    "unit_id": row["unit_id"], "family": row["family"], "partition": row["partition"],
                    "truth_factor": row["truth_factor"], "surface_factor": row["surface_factor"],
                    "distractor_factor": row["distractor_factor"], "code": row["code"], "role": role,
                    "subtoken": subtoken, "span_length": len(positions), "token_position": position,
                    "token_id": token_id, "token_text": tok.convert_ids_to_tokens([token_id])[0],
                })
        disjoint = disjoint and len(occupied) == len(set(occupied))
    cells = Counter((row["family"], row["partition"], row["truth_factor"], row["surface_factor"], row["distractor_factor"], row["code"]) for row in cases)
    rng = np.random.default_rng(PHASE)
    movement_permutations = {
        family: [rng.permutation(k).astype(int).tolist() for _ in range(8)]
        for family, k in (("attribute_binding", 256), ("agent_patient", 128))
    }
    role_coalitions = {
        "record_to_query_path": list(PATH_ROLES),
        "path_plus_code": [*PATH_ROLES, "code_instruction"],
        "path_plus_code_boundary": [*PATH_ROLES, "code_instruction", "boundary"],
        "all_registered_roles": list(ROLES),
        **{f"path_without_{role}": [value for value in PATH_ROLES if value != role] for role in PATH_ROLES},
    }
    supports = source_protocol["supports"]
    checks = {
        "authorization": closure_audit["all_checks_passed"],
        "units": len(units) == 24,
        "cases": len(cases) == 384,
        "partitions": Counter((row["family"], row["partition"]) for row in units) == {(family, partition): 6 for family in FAMILIES for partition in PARTITIONS},
        "factorial": cells == {(family, partition, *cell): 6 for family in FAMILIES for partition in PARTITIONS for cell in itertools.product((1, -1), repeat=4)},
        "zero_models": all(abs(value - 0.5) < 1e-12 for key, value in zero.items() if key != "truth_x_code_oracle") and zero["truth_x_code_oracle"] == 1.0,
        "semantic_uniqueness": len({row["prompt"] for row in cases}) == 384,
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
        "phase": PHASE, "campaign": CAMPAIGN, "created_at_utc": now(),
        "status": "fourth_lexicon_role_lattice_contract_frozen",
        "model": "Qwen3-4B local BF16 CUDA nonquantized",
        "object": "prospective fourth-lexicon replication of full truth fields, coordinate assignment, and multi-position activation transport",
        "families": list(FAMILIES), "partitions": list(PARTITIONS), "units": 24, "cases": 384,
        "roles": list(ROLES), "states": STATES, "activation_coordinates": DIM, "occurrences": len(occurrences),
        "archive": {"path": "raw/qwen3_role_subtoken_all_states.uint16.npy", "shape": [STATES, len(occurrences), DIM], "dtype": "uint16 exact BF16 bit patterns", "fixed_width": WIDTH, "batch_size": BATCH_SIZE},
        "supports": supports,
        "movement_permutations": movement_permutations,
        "role_coalitions": role_coalitions,
        "modes": ["frozen_support"] + [f"movement_permutation_{index}" for index in range(8)] + [f"single_{role}" for role in ROLES] + [f"coalition_{name}" for name in role_coalitions],
        "frozen_field_prediction": {"role": "query_anchor", "state": 19, "cross_partition_cosine_min": 0.90, "each_partition_to_c110_reference_cosine_min": 0.85, "each_partition_frozen_support_topk_overlap_min": 0.50},
        "frozen_predictions": {
            "K292_attribute_coordinate_assignment": "frozen-support median raw truth gain exceeds every one of eight exact-energy within-support movement-permutation medians in each of four cells",
            "K294_agent_path_increment": "record_to_query_path median raw truth gain exceeds single query_anchor in each of four cells",
            "agent_all_role_increment": "all_registered_roles median raw truth gain exceeds record_to_query_path in each of four cells",
            "leave_query_anchor_candidate": "removing query_anchor from the four-role path lowers median raw truth gain in each of four agent cells",
            "leave_query_focus_candidate": "removing query_focus from the four-role path lowers median raw truth gain in each of four agent cells",
        },
        "observation_first": "capture and adjudicate embedding-to-state36 role fields before interventions; intervention candidates cannot rewrite field gates",
        "behavior_policy": "report standard and reversed code separately; reversed failure is typed missingness for output-task claims and does not erase upstream truth-field observations",
        "completion_rule": "finish every registered observation and intervention route; a failed route retires only that route",
        "numeric": {"movement_permutation_actual_l2_relative_tolerance": 0.02, "fixed_width": WIDTH, "batch_size": BATCH_SIZE},
        "typed_missingness": {"human_naturalness": "no independent blind rating", "cross_model": "Qwen3 only", "natural_route": "simultaneous role patching does not identify endogenous transport order"},
        "claim_boundary": "controlled-English activation-coordinate replication only; activation coordinates are not weights or independent semantic neurons; no attention/MLP, minimality, natural-route, or universal-language claim",
        "source_paths": {"c110_protocol": str(SOURCE / "protocol/preregistration.json"), "c110_mean_field": str(SOURCE / "analysis/mean_truth_role_state.float32.npy"), "c112_closure": str(C112 / "analysis/closure.json")},
        "source_hashes": {"c110_protocol": core.sha(SOURCE / "protocol/preregistration.json"), "c110_mean_field": core.sha(SOURCE / "analysis/mean_truth_role_state.float32.npy"), "c112_closure": core.sha(C112 / "analysis/closure.json")},
        "material_digest": core.digest([*units, *cases]), "producer_sha256": core.sha(Path(__file__)),
        "authorization": "execute_phase1619_c113_exact_field_capture",
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    audit = {"phase": PHASE, "campaign": CAMPAIGN, "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "zero_models": zero, "occurrences": len(occurrences), "max_width": max(len(row["prompt_ids"]) for row in compiled), "material_digest": protocol["material_digest"], "authorization": protocol["authorization"]}
    core.save(OUT / "audit/internal_pre_model_audit.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
