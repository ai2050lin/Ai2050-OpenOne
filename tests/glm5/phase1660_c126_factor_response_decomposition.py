#!/usr/bin/env python3
"""C126 deterministic Walsh decomposition of the already captured C125 response field."""
from __future__ import annotations

import itertools
import json
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
OUT = RESULT / "phase1660_c126_factor_response_decomposition"
C123 = RESULT / "phase1657_c123_role_transition_atlas_discovery"
C125 = RESULT / "phase1659_c125_final_transition_decomposition"
PUBLIC = ROOT / "frontend/public/vis_data/research_kernel/c109_role_state_field_atlas.json"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core
import phase1657_c123_role_transition_atlas as c123

FAMILIES = ("attribute_binding", "agent_patient")
ROLES = ("focus_record", "boundary")
CHECKPOINTS = ("embedding", "pre_last_block", "post_last_block_pre_norm", "post_final_norm")
FACTORS = ("truth", "surface", "distractor", "code")
DIM = 2560
SUPPORT_K = 256


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return 0.0 if denominator <= 1e-12 else float(np.dot(left, right) / denominator)


def topk(values: np.ndarray, k: int = SUPPORT_K) -> set[int]:
    return {int(value) for value in np.argpartition(np.abs(values), -k)[-k:]}


def effect_specs() -> list[dict]:
    specs = []
    for mask in range(1, 1 << len(FACTORS)):
        members = [FACTORS[index] for index in range(len(FACTORS)) if mask & (1 << index)]
        specs.append({"mask": mask, "name": "_x_".join(members), "members": members})
    return specs


def source_paths() -> dict[str, Path]:
    return {
        "c125_raw": C125 / "raw/qwen3_role_checkpoint_states.float32.npy",
        "c125_cases": C125 / "compiled/qwen3.jsonl",
        "c125_units": C125 / "material/units.jsonl",
        "c125_closure": C125 / "analysis/closure.json",
        "c125_audit": C125 / "audit/independent_closure_audit.json",
        "c123_nomination": C123 / "protocol/frozen_discovery_nomination.json",
        "c123_increments": C123 / "analysis/discovery_selected_role_increments.float32.npy",
    }


def contract() -> None:
    if OUT.exists():
        raise RuntimeError(f"C126 already exists: {OUT}")
    paths = source_paths()
    missing = [name for name, path in paths.items() if not path.exists()]
    raw = np.load(paths["c125_raw"], mmap_mode="r")
    cases = core.rows(paths["c125_cases"])
    units = core.rows(paths["c125_units"])
    audit = core.load(paths["c125_audit"])
    closure = core.load(paths["c125_closure"])
    cells = Counter((row["unit_id"], row["truth_factor"], row["surface_factor"], row["distractor_factor"], row["code"]) for row in cases)
    expected = {(row["unit_id"], *cell): 1 for row in units for cell in itertools.product((1, -1), repeat=4)}
    checks = {
        "sources": not missing,
        "authorization": audit["all_checks_passed"] and closure["next_authorization"].startswith("C126 may use only"),
        "raw_shape": list(raw.shape) == [384, 2, 4, DIM],
        "rows": len(cases) == 384 and len(units) == 24,
        "factorial": cells == expected,
        "roles": core.load(C125 / "protocol/preregistration.json")["roles"] == list(ROLES),
        "finite": bool(np.isfinite(np.asarray(raw[:2], dtype=np.float32)).all()),
        "effects": len(effect_specs()) == 15 and effect_specs()[8]["name"] == "truth_x_code",
    }
    if not all(checks.values()):
        raise RuntimeError({"checks": checks, "missing": missing})
    protocol = {
        "phase": 1660,
        "campaign": "C126",
        "created_at_utc": now(),
        "status": "existing_data_factor_response_decomposition_frozen",
        "object": "complete four-factor Walsh response decomposition of the already captured C125 role/checkpoint activation field",
        "model_runs": 0,
        "source_cases": 384,
        "independent_units": 24,
        "families": list(FAMILIES),
        "roles": list(ROLES),
        "checkpoints": list(CHECKPOINTS),
        "factors": list(FACTORS),
        "effects": effect_specs(),
        "definition": "E_m=(1/16) sum_z chi_m(z) H(z), for every nonempty factor subset m",
        "registered_cells": [
            {"family": "attribute_binding", "role": "boundary"},
            {"family": "agent_patient", "role": "focus_record"},
            {"family": "agent_patient", "role": "boundary"},
        ],
        "analysis_policy": "descriptive complete decomposition with deterministic rankings; no post-hoc pass threshold and no scientific confirmation claim",
        "observation_policy": "full 2560 activation coordinates; no PCA/SVD, attention, MLP, or parameter-weight analysis",
        "typed_missingness": {
            "behavior": "C125 overall behavior failed because the reversed code arm failed",
            "fresh_confirmation": "none; C126 reuses only C125 arrays",
            "complete_tokens": "two registered roles only",
            "causality": "no intervention",
            "cross_model": "Qwen3 only",
        },
        "claim_boundary": "factorial response bookkeeping, not semantic operators, mechanisms, weights, neurons, complete-token paths, manifolds, topology, or new mathematics",
        "source_paths": {name: str(path) for name, path in paths.items()},
        "source_hashes": {name: core.sha(path) for name, path in paths.items()},
        "producer_sha256": core.sha(Path(__file__)),
        "authorization": "analyze_existing_c125_arrays_only",
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    report = {"phase": 1660, "campaign": "C126", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "authorization": protocol["authorization"]}
    core.save(OUT / "audit/internal_contract_audit.json", report)
    print(json.dumps(report, indent=2))


def frozen_vectors() -> dict[tuple[str, str], np.ndarray]:
    increments = np.load(C123 / "analysis/discovery_selected_role_increments.float32.npy", mmap_mode="r")
    discovery = c123.discovery_fields()
    return {
        ("attribute_binding", "boundary"): np.asarray(increments[0, 35], dtype=np.float32),
        ("agent_patient", "focus_record"): np.asarray(increments[1, 35], dtype=np.float32),
        ("agent_patient", "boundary"): np.mean(discovery["agent_patient"][:, 6, 36] - discovery["agent_patient"][:, 6, 35], axis=0, dtype=np.float32),
    }


def analyze() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    if protocol["authorization"] != "analyze_existing_c125_arrays_only" or not core.load(OUT / "audit/internal_contract_audit.json")["all_checks_passed"]:
        raise RuntimeError("C126 analysis authorization missing")
    for name, path in protocol["source_paths"].items():
        if core.sha(Path(path)) != protocol["source_hashes"][name]:
            raise RuntimeError(f"source drift: {name}")
    raw = np.load(C125 / "raw/qwen3_role_checkpoint_states.float32.npy", mmap_mode="r")
    cases = core.rows(C125 / "compiled/qwen3.jsonl")
    units = core.rows(C125 / "material/units.jsonl")
    unit_lookup = {row["unit_id"]: index for index, row in enumerate(units)}
    specs = effect_specs()
    means = np.zeros((len(units), len(ROLES), len(CHECKPOINTS), DIM), dtype=np.float32)
    effects = np.zeros((len(units), len(ROLES), len(specs), len(CHECKPOINTS), DIM), dtype=np.float32)
    counts = np.zeros(len(units), dtype=np.int32)
    factors_by_row = []
    for row_index, row in enumerate(cases):
        unit = unit_lookup[row["unit_id"]]
        factors = (row["truth_factor"], row["surface_factor"], row["distractor_factor"], row["code"])
        factors_by_row.append(factors)
        value = np.asarray(raw[row_index], dtype=np.float32)
        means[unit] += value / 16.0
        for effect_index, spec in enumerate(specs):
            sign = int(np.prod([factors[index] for index in range(len(FACTORS)) if spec["mask"] & (1 << index)]))
            effects[unit, :, effect_index] += sign * value / 16.0
        counts[unit] += 1
    if not np.all(counts == 16):
        raise RuntimeError(counts.tolist())
    reconstruction_max = 0.0
    for row_index, row in enumerate(cases):
        unit = unit_lookup[row["unit_id"]]
        factors = factors_by_row[row_index]
        predicted = means[unit].copy()
        for effect_index, spec in enumerate(specs):
            sign = int(np.prod([factors[index] for index in range(len(FACTORS)) if spec["mask"] & (1 << index)]))
            predicted += sign * effects[unit, :, effect_index]
        reconstruction_max = max(reconstruction_max, float(np.max(np.abs(predicted - np.asarray(raw[row_index], dtype=np.float32)))))
    (OUT / "analysis").mkdir(parents=True, exist_ok=True)
    np.save(OUT / "analysis/unit_role_checkpoint_means.float32.npy", means)
    np.save(OUT / "analysis/unit_role_effect_checkpoint.float32.npy", effects)

    frozen = frozen_vectors()
    comparison_rows = []
    visualization_rows = []
    registered = [(row["family"], row["role"]) for row in protocol["registered_cells"]]
    for family, role in registered:
        family_indices = [index for index, row in enumerate(units) if row["family"] == family]
        role_index = ROLES.index(role)
        family_effects = effects[family_indices, role_index]
        target = frozen[(family, role)]
        local_rows = []
        for effect_index, spec in enumerate(specs):
            mean = np.mean(family_effects[:, effect_index], axis=0, dtype=np.float32)
            left = np.mean(family_effects[:6, effect_index], axis=0, dtype=np.float32)
            right = np.mean(family_effects[6:, effect_index], axis=0, dtype=np.float32)
            component_vectors = {
                "embedding": mean[0],
                "pre_last_block": mean[1],
                "post_last_block_pre_norm": mean[2],
                "post_final_norm": mean[3],
                "final_block_increment": mean[2] - mean[1],
                "final_norm_increment": mean[3] - mean[2],
                "combined_increment": mean[3] - mean[1],
            }
            combined = component_vectors["combined_increment"]
            split_combined = cosine(left[3] - left[1], right[3] - right[1])
            row = {
                "family": family,
                "role": role,
                "effect": spec["name"],
                "mask": spec["mask"],
                "combined_norm": float(np.linalg.norm(combined)),
                "split_half_cosine": split_combined,
                "frozen_signed_cosine": cosine(target, combined),
                "frozen_absolute_cosine": abs(cosine(target, combined)),
                "frozen_top256_overlap": len(topk(target) & topk(combined)) / SUPPORT_K,
            }
            local_rows.append(row)
            for kind, values in component_vectors.items():
                visualization_rows.append({"family": family, "role": role, "effect": spec["name"], "kind": kind, "values": values.tolist()})
        signed_order = sorted(range(len(local_rows)), key=lambda index: (-local_rows[index]["frozen_signed_cosine"], local_rows[index]["effect"]))
        absolute_order = sorted(range(len(local_rows)), key=lambda index: (-local_rows[index]["frozen_absolute_cosine"], local_rows[index]["effect"]))
        for rank, index in enumerate(signed_order, 1):
            local_rows[index]["signed_rank"] = rank
        for rank, index in enumerate(absolute_order, 1):
            local_rows[index]["absolute_rank"] = rank
        comparison_rows.extend(local_rows)
    core.write_rows(OUT / "analysis/factor_comparisons.jsonl", comparison_rows)
    core.save(OUT / "analysis/visualization_effect_rows.json", visualization_rows)
    truth_rows = [row for row in comparison_rows if row["effect"] == "truth"]
    output_rows = [row for row in comparison_rows if row["effect"] == "truth_x_code"]
    checks = {
        "means_shape": list(means.shape) == [24, 2, 4, DIM],
        "effects_shape": list(effects.shape) == [24, 2, 15, 4, DIM],
        "finite": bool(np.isfinite(means).all() and np.isfinite(effects).all()),
        "reconstruction": reconstruction_max <= 2e-4,
        "comparisons": len(comparison_rows) == 45,
        "visualization": len(visualization_rows) == 315,
    }
    if not all(checks.values()):
        raise RuntimeError({"checks": checks, "reconstruction_max": reconstruction_max})
    summary = {
        "phase": 1660,
        "campaign": "C126",
        "created_at_utc": now(),
        "status": "complete_existing_data_factor_response_decomposition",
        "checks": checks,
        "reconstruction_max_abs": reconstruction_max,
        "truth_effects": truth_rows,
        "output_effects": output_rows,
        "top_signed_by_cell": {f"{family}::{role}": sorted([row for row in comparison_rows if row["family"] == family and row["role"] == role], key=lambda row: row["signed_rank"])[0] for family, role in registered},
        "top_absolute_by_cell": {f"{family}::{role}": sorted([row for row in comparison_rows if row["family"] == family and row["role"] == role], key=lambda row: row["absolute_rank"])[0] for family, role in registered},
        "claim_boundary": protocol["claim_boundary"],
        "authorization": "synthesize_c126_heatmap_and_close",
    }
    core.save(OUT / "analysis/adjudication.json", summary)
    core.save(OUT / "audit/internal_analysis_audit.json", {"phase": 1660, "campaign": "C126", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "authorization": summary["authorization"]})
    print(json.dumps(summary, indent=2))


def synthesize() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    report = core.load(OUT / "analysis/adjudication.json")
    if report["authorization"] != "synthesize_c126_heatmap_and_close" or not core.load(OUT / "audit/internal_analysis_audit.json")["all_checks_passed"]:
        raise RuntimeError("C126 synthesis authorization missing")
    drift = {
        name: {"frozen": digest, "current": core.sha(Path(protocol["source_paths"][name]))}
        for name, digest in protocol["source_hashes"].items()
        if core.sha(Path(protocol["source_paths"][name])) != digest
    }
    if drift and not set(drift).issubset({"c125_closure", "c125_audit"}):
        raise RuntimeError({"unexpected_source_drift": drift})
    if drift:
        core.save(OUT / "protocol/upstream_index_correction_amendment.json", {
            "created_at_utc": now(),
            "reason": "C126 exposed that C125 used the two-role local index for its frozen seven-role common-boundary comparator. C125 corrected that deterministic index after C126 analysis; its raw states, cases, units, C123 source, thresholds, and failed overall gate did not change.",
            "drift": drift,
            "unchanged_data_sources": ["c125_raw", "c125_cases", "c125_units", "c123_nomination", "c123_increments"],
            "effect_on_c126": "none; C126 independently used the correct C123 boundary index and the unchanged C125 arrays",
        })
    payload = core.load(PUBLIC)
    rows = core.load(OUT / "analysis/visualization_effect_rows.json")
    payload["c126_factor_response_batch"] = {"protocol": protocol, "adjudication": report, "effect_rows": rows}
    payload.update({
        "phase": 1660,
        "campaign": "C109-C117 + C123-C126",
        "title": "Role-State Atlas + Typed Transition and Factor Responses",
        "claim_boundary": "C126 shows full-coordinate factorial response components from existing C125 Qwen3 data. These are descriptive activation responses, not weights, independent neurons, semantic operators, attention/MLP mechanisms, or a complete language graph.",
        "created_at_utc": now(),
    })
    canonical = OUT / "visualization/c109_c126_factor_response_atlas.json"
    core.save(canonical, payload)
    shutil.copyfile(canonical, PUBLIC)
    truth = {f"{row['family']}::{row['role']}": row for row in report["truth_effects"]}
    output = {f"{row['family']}::{row['role']}": row for row in report["output_effects"]}
    closure = {
        "phase": 1660,
        "campaign": "C126",
        "created_at_utc": now(),
        "status": "factor_response_observation_closed",
        "headline": {"truth_effects": truth, "truth_x_code_effects": output, "top_signed_by_cell": report["top_signed_by_cell"], "top_absolute_by_cell": report["top_absolute_by_cell"]},
        "new_puzzles": {"K318": "C125's final transition can be decomposed exactly into all 15 nonempty effects of truth, surface, distractor, and code. The C123 match is therefore an effect-specific question, not a property of one undifferentiated trajectory."},
        "theory_update": "A candidate transformation graph must label edges by controlled input contrasts. Physical checkpoint increments alone conflate multiple orthogonal experimental effects.",
        "unified_formula": "E_m=(1/16) sum_z chi_m(z)H(z); DeltaE_m=E_m(post)-E_m(pre). The pair (checkpoint type, contrast m) is the minimum typed observation used here.",
        "problems": ["existing-data diagnosis only", "C125 behavior failed overall", "two controlled relation families", "two registered roles", "Qwen3 only", "no causal intervention", "no complete-token state"],
        "heatmap": {"path": str(PUBLIC.relative_to(ROOT)).replace("\\", "/"), "bytes": PUBLIC.stat().st_size, "sha256": core.sha(PUBLIC), "activation_coordinates": DIM, "effect_rows": len(rows)},
        "claim_boundary": payload["claim_boundary"],
        "next_authorization": "C127 may remain on the same observation-first objective, but must freeze a behavior-qualified fresh language family and uniformly typed HiddenState checkpoints before any new model run; C126 itself supplies diagnosis, not confirmation.",
    }
    core.save(OUT / "analysis/closure.json", closure)
    checks = {
        "analysis": core.load(OUT / "audit/internal_analysis_audit.json")["all_checks_passed"],
        "rows": len(rows) == 315 and all(len(row["values"]) == DIM for row in rows),
        "asset": core.sha(canonical) == core.sha(PUBLIC),
        "boundary": "not weights" in payload["claim_boundary"] and "attention/MLP" in payload["claim_boundary"],
        "next": closure["next_authorization"].startswith("C127 may remain"),
    }
    audit = {"phase": 1660, "campaign": "C126", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "asset_sha256": core.sha(PUBLIC), "authorization": "independent_audit_then_append_memo"}
    core.save(OUT / "audit/internal_closure_audit.json", audit)
    print(json.dumps({"audit": audit, "headline": closure["headline"], "next_authorization": closure["next_authorization"]}, indent=2))


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in {"contract", "analyze", "synthesize"}:
        raise SystemExit("usage: phase1660_c126_factor_response_decomposition.py {contract|analyze|synthesize}")
    {"contract": contract, "analyze": analyze, "synthesize": synthesize}[sys.argv[1]]()


if __name__ == "__main__":
    main()
