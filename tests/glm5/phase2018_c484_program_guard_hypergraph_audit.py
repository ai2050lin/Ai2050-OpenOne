#!/usr/bin/env python3
"""Independent audit and strict re-adjudication for Phase2005-2018."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/glm5/result"
PRODUCER = ROOT / "tests/glm5/phase2005_c471_c484_program_guard_hypergraph_campaign.py"
VISUAL = ROOT / "frontend/public/vis_data/research_kernel/c484_program_guard_hypergraph.json"

PHASES = {
    f"C{campaign}": (2005 + campaign - 471, slug)
    for campaign, slug in (
        (471, "evidence_adjudication_and_program_guard_master_contract"),
        (472, "eight_family_four_factor_language_program_material"),
        (473, "program_compiler_zero_model_and_naturalness_audit"),
        (474, "qwen_program_graph_behavior_qualification"),
        (475, "qualified_program_graph_full_coordinate_field"),
        (476, "complete_factorial_walsh_response_ledger"),
        (477, "observation_first_condition_and_coordinate_atlas"),
        (478, "shared_propagation_reconstruction"),
        (479, "family_blind_program_graph_guard"),
        (480, "coordinate_state_guarded_propagation"),
        (481, "cross_family_selector_lockbox_adjudication"),
        (482, "arbitrary_full_coordinate_coupling_tournament"),
        (483, "unseen_composition_and_conditional_writer"),
        (484, "campaign_synthesis_visual_cleanup_and_audit"),
    )
}


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def out(name: str) -> Path:
    phase, slug = PHASES[name]
    return RESULT / f"phase{phase}_{name.lower()}_{slug}"


def finite(value) -> bool:
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, list):
        return all(finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def main() -> None:
    digest = hashlib.sha256(PRODUCER.read_bytes()).hexdigest()
    checks = {}
    for name, (phase, _) in PHASES.items():
        protocol = load(out(name) / "protocol/preregistration.json")
        final = load(out(name) / "analysis/final.json")
        checks[f"{name}_closed"] = final["all_checks_passed"]
        checks[f"{name}_phase"] = protocol["phase"] == phase == final["phase"]
        checks[f"{name}_producer_hash"] = protocol["producer_sha256"] == digest

    material = rows(out("C472") / "material/cases.jsonl")
    compiled = rows(out("C474") / "compiled/qwen3.jsonl")
    checks["material_rows"] = len(material) == len(compiled) == 3840
    checks["material_programs"] = len(material) // 16 == 240
    checks["material_families"] = len({row["family"] for row in material}) == 8
    checks["material_balance"] = sum(row["gold_position"] == 0 for row in material) == 1920
    checks["factorial_cells"] = all(
        len({tuple(row["bits"]) for row in material if row["family"] == family and row["construction"] == construction and row["unit"] == unit}) == 16
        for family in {row["family"] for row in material}
        for construction in {row["construction"] for row in material}
        for unit in range(10)
    )
    checks["compiled_width"] = max(len(row["prompt_ids"]) for row in compiled) < 192
    checks["compiled_roles"] = all(
        positions and all(0 <= int(position) < len(row["prompt_ids"]) for position in positions)
        for row in compiled for positions in row["role_positions"].values()
    )

    behavior = rows(out("C474") / "raw/behavior.jsonl")
    c474 = load(out("C474") / "analysis/final.json")["headline"]
    checks["behavior_rows"] = len(behavior) == 3840
    checks["behavior_accuracy"] = abs(sum(row["correct"] for row in behavior) / len(behavior) - c474["accuracy"]) < 1e-12
    checks["all_families_eligible"] = len(c474["eligible_families"]) == 8 and c474["field_authorized"]

    effects = rows(out("C476") / "analysis/effect_index.jsonl")
    c476 = load(out("C476") / "analysis/final.json")["headline"]
    checks["effect_records"] = len(effects) == c476["effect_records"] == c476["complete_programs"] * 15 == 3165
    checks["effect_masks"] = all(1 <= row["effect_mask"] <= 15 and row["effect_order"] == int(row["effect_mask"]).bit_count() for row in effects)
    checks["temporal_sparse_registered"] = c476["family_program_counts"]["temporal_order"] == 1

    c477 = load(out("C477") / "analysis/final.json")["headline"]
    atlas = rows(out("C477") / "analysis/condition_atlas.jsonl")
    checks["atlas_rows"] = len(atlas) == c477["rows"] == 7200
    checks["atlas_fallback"] = c477["fallback_families"] == ["temporal_order"]
    checks["atlas_scope_labels"] = all(row["source_scope"] in ("discovery_ledger_brief", "fallback_all_complete_descriptive_only") for row in atlas)
    checks["atlas_finite"] = finite(c477) and finite(atlas)

    c478 = load(out("C478") / "analysis/final.json")["headline"]
    shared_gate = all(
        c478["metrics"][split]["shared"]["nrmse"] < min(
            c478["metrics"][split]["identity"]["nrmse"], c478["metrics"][split]["mean"]["nrmse"]
        ) for split in ("within", "family", "report")
    )
    checks["shared_gate_recomputed"] = c478["shared_candidate"] == shared_gate is False
    checks["five_family_training_registered"] = len(c478["registered_train_families"]) == 5 and c478["excluded_sparse_train_family"] == "temporal_order"

    c479 = load(out("C479") / "analysis/final.json")["headline"]
    c480 = load(out("C480") / "analysis/final.json")["headline"]
    c481 = load(out("C481") / "analysis/final.json")["headline"]
    program_gains = {
        split: c478["metrics"][split]["shared"]["nrmse"] - c479["metrics"][split]["program"]["nrmse"]
        for split in ("family", "report")
    }
    state_gains = {
        split: c479["metrics"][split]["program"]["nrmse"] - c480["metrics"][split]["state"]["nrmse"]
        for split in ("family", "report")
    }
    checks["program_gains_recomputed"] = all(abs(program_gains[key] - c481["program_gains"][key]) < 1e-12 for key in program_gains)
    checks["state_gains_recomputed"] = all(abs(state_gains[key] - c481["state_gains_over_program"][key]) < 1e-12 for key in state_gains)
    checks["selector_abstained"] = not c481["selector_candidate"] and c481["selected_model"] == "none"

    c482 = load(out("C482") / "analysis/final.json")["headline"]
    registered_gate = all(
        min(c482["gains"][split].values()) >= 0.02 for split in ("family", "report")
    )
    checks["registered_coupling_gate_recomputed"] = c482["full_coordinate_coupling_candidate"] == registered_gate
    checks["operator_passport"] = c482["operator_passport"]["shape"] == [2560, 2560] and c482["operator_passport"]["nonzero_fraction"] > 0.99

    c483 = load(out("C483") / "analysis/final.json")["headline"]
    composition_gate = c483["composition_gain"] >= 0.01 and c483["composition_mask_wins"] >= 3
    checks["composition_gate_recomputed"] = c483["composition_candidate"] == composition_gate is False
    checks["writer_abstention"] = not c483["writer_ran"] and not c483["specificity_passed"]

    c484 = load(out("C484") / "analysis/final.json")["headline"]
    visual = load(VISUAL)
    checks["visual_rows"] = len(visual["rows"]) == c484["visual_rows"] == 1088
    checks["visual_coordinates"] = all(len(row["values"]) == 2560 for row in visual["rows"])
    checks["visual_finite"] = finite(visual)
    cleanup = load(out("C484") / "audit/cleanup.json")
    checks["cleanup_count"] = len(cleanup) == c484["cleanup_files"] == 16
    checks["cleanup_hashes"] = all(row["deleted"] and len(row["sha256"]) == 64 for row in cleanup)
    checks["cleanup_absent"] = all(not (ROOT / row["path"]).exists() for row in cleanup)
    checks["cleanup_bytes"] = sum(row["bytes"] for row in cleanup) == c484["cleanup_bytes"]

    adjudication = {
        "retained": [
            "Eight controlled language families passed the fixed-codebook behavior interface, with temporal_order materially weaker than the other seven.",
            "Seven families supplied dense complete 16-cell programs; temporal_order supplied one descriptive confirmation program and no training program.",
            "The family-blind effect-mask guard and coordinate sign-amplitude guard failed whole-family and unseen-surface prediction.",
            "The registered full-coordinate ridge beat the selected-state fallback, shuffled-target, and coordinate-roll controls on family and report panels.",
            "The registered singleton-plus-pair residual composition formula failed strongly; causal writing was correctly not run.",
        ],
        "corrected_overclaims": [
            "C478 does not establish a shared propagation law because it lost to identity on the whole-family lockbox.",
            "C482 does not yet establish a general full-coordinate coupling law: the tournament omitted identity/shared baselines, and selector failure caused the registered comparator to fall back to the poor state model.",
            "The nearly dense ridge matrix is an underdetermined predictive fit, not evidence that almost every coordinate is a causal edge.",
            "C483 rejects only the frozen additive singleton-plus-pair residual formula, not all possible higher-order composition structures.",
        ],
        "strict_gates": {
            "shared_propagation": False,
            "family_blind_program_guard": False,
            "coordinate_state_guard": False,
            "registered_ridge_vs_listed_controls": True,
            "broad_full_coordinate_coupling": False,
            "unseen_composition": False,
            "causal_writer": False,
            "new_mathematics": False,
        },
        "next_stage_same_goal": False,
        "next_authorization": "design a distinct identity-controlled full-coordinate system-identification campaign; do not auto-continue this failed selector/composition route",
    }
    report = {
        "phase": 2018, "campaign": "C471-C484", "producer_sha256": digest,
        "checks": checks, "passed": sum(checks.values()), "total": len(checks),
        "all_checks_passed": all(checks.values()), "adjudication": adjudication,
    }
    destination = out("C484") / "audit/independent_audit.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    assert report["all_checks_passed"]


if __name__ == "__main__":
    main()
