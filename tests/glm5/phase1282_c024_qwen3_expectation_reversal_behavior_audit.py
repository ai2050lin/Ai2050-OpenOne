#!/usr/bin/env python3
"""Independent pre/final audit for Phase1282."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "tests/glm5/result/phase1281_c024_expectation_reversal_contract"
OUT = ROOT / "tests/glm5/result/phase1282_c024_qwen3_expectation_reversal_behavior"
PROTOCOL = OUT / "protocol/preregistration.json"
RAW = OUT / "raw/full_continuation_scores.jsonl"
GENERATION = OUT / "raw/confirmation_generations.jsonl"
FINAL = OUT / "analysis/final.json"
PREAUDIT = OUT / "audit/independent_preaudit.json"
FINAL_AUDIT = OUT / "audit/independent_final_audit.json"
SURFACES = ("coordination", "adverbial", "expectation", "evaluation", "report")
PANELS = ("consistency", "contrast", "carrier_consistency", "lexical_consistency", "carrier_contrast", "lexical_contrast")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def file_sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def check(name: str, passed: bool, detail: Any = None) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def preaudit() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    parent = json.loads((INPUT / "analysis/final.json").read_text(encoding="utf-8"))
    parent_audit = json.loads((INPUT / "audit/independent_final_audit.json").read_text(encoding="utf-8"))
    dependencies = {
        "phase1281_protocol": INPUT / "protocol/preregistration.json",
        "phase1281_material": INPUT / "material/frozen_expectation_worlds.jsonl",
        "phase1281_final": INPUT / "analysis/final.json",
        "phase1281_audit": INPUT / "audit/independent_final_audit.json",
    }
    checks = [
        check("phase", protocol["phase"] == 1282),
        check("parent_authorization", parent["authorization"] == "phase1282_qwen3_multitoken_behavior_and_generation" and parent_audit["all_checks_passed"]),
        check("dependency_hashes", all(protocol["dependencies"][key] == file_sha256(path) for key, path in dependencies.items())),
        check("dimensions", protocol["row_count"] == 256 and protocol["context_count"] == 7680 and protocol["scored_sequence_count"] == 15360),
        check("surface_panel_registry", tuple(protocol["surfaces"]) == SURFACES and tuple(protocol["panels"]) == PANELS),
        check("full_continuation_primary", any("entire frozen continuation" in value for value in protocol["hard_stops"])),
        check("generation_frozen", protocol["generation"]["partition"] == "confirmation" and protocol["generation"]["do_sample"] is False),
        check("single_run", protocol["formal_run_budget"] == 1),
    ]
    result = {"phase": 1282, "audit_type": "independent_preaudit", "checks": checks, "passed_count": sum(row["passed"] for row in checks), "check_count": len(checks), "all_checks_passed": all(row["passed"] for row in checks)}
    atomic_json(PREAUDIT, result)
    print(canonical_json(result))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


def final_audit() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    final = json.loads(FINAL.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in RAW.read_text(encoding="utf-8").splitlines() if line.strip()]
    generations = [json.loads(line) for line in GENERATION.read_text(encoding="utf-8").splitlines() if line.strip()]
    behavior = final["behavior"]
    generation = final["generation"]
    thresholds = protocol["thresholds"]
    keys = {(row["row_id"], row["surface"], row["panel"]): row for row in rows}
    core_accuracy = []
    null_accuracy = []
    effect_positive = []
    effect_median = []
    lexical_ratio = []
    for partition in ("discovery", "selection", "confirmation"):
        ids = sorted({row["row_id"] for row in rows if row["partition"] == partition})
        for surface in SURFACES:
            for panel in PANELS:
                values = [keys[(row_id, surface, panel)]["D_opposite_minus_expected"] for row_id in ids]
                expected_negative = panel in ("consistency", "carrier_consistency", "lexical_consistency")
                accuracy = float(np.mean([value < 0 for value in values])) if expected_negative else float(np.mean([value > 0 for value in values]))
                (core_accuracy if panel in ("consistency", "contrast") else null_accuracy).append(accuracy)
            delta = np.asarray([keys[(row_id, surface, "contrast")]["D_opposite_minus_expected"] - keys[(row_id, surface, "consistency")]["D_opposite_minus_expected"] for row_id in ids])
            lc = np.asarray([keys[(row_id, surface, "lexical_consistency")]["D_opposite_minus_expected"] - keys[(row_id, surface, "carrier_consistency")]["D_opposite_minus_expected"] for row_id in ids])
            lr = np.asarray([keys[(row_id, surface, "lexical_contrast")]["D_opposite_minus_expected"] - keys[(row_id, surface, "carrier_contrast")]["D_opposite_minus_expected"] for row_id in ids])
            med = float(np.median(delta))
            effect_positive.append(float(np.mean(delta > 0)))
            effect_median.append(med)
            lexical_ratio.append(float(np.median(np.maximum(np.abs(lc), np.abs(lr))) / max(abs(med), 1.0e-12)))
    generation_cells = {}
    for surface in SURFACES:
        for panel in ("consistency", "contrast"):
            subset = [row for row in generations if row["surface"] == surface and row["panel"] == panel]
            coverage = float(np.mean([row["parsed"] for row in subset]))
            parsed = [row for row in subset if row["parsed"]]
            accuracy = float(np.mean([row["correct"] for row in parsed])) if parsed else 0.0
            generation_cells[f"{surface}.{panel}"] = {"coverage": coverage, "accuracy_given_parsed": accuracy, "n": len(subset)}
    behavior_gates = {
        "finite": all(row["finite"] for row in rows),
        "core_sign": min(core_accuracy) >= thresholds["core_sign_accuracy_min"],
        "null_sign": min(null_accuracy) >= thresholds["null_sign_accuracy_min"],
        "effect_positive": min(effect_positive) >= thresholds["effect_positive_fraction_min"],
        "effect_median": min(effect_median) >= thresholds["median_functional_effect_min"],
        "lexical_specificity": max(lexical_ratio) <= thresholds["lexical_specific_ratio_max"],
    }
    generation_gates = {
        "coverage": min(value["coverage"] for value in generation_cells.values()) >= thresholds["generation_parse_coverage_min"],
        "accuracy": min(value["accuracy_given_parsed"] for value in generation_cells.values()) >= thresholds["generation_sign_accuracy_min"],
    }
    expected_pass = all(behavior_gates.values()) and all(generation_gates.values())
    precision = final["precision_audit"]
    checks = [
        check("raw_context_count", len(rows) == 7680, len(rows)),
        check("generation_count", len(generations) == 1280, len(generations)),
        check("raw_keys_unique", len(keys) == len(rows)),
        check("all_finite", all(row["finite"] for row in rows)),
        check("behavior_gates", behavior_gates == behavior["gates"], {"recomputed": behavior_gates, "stored": behavior["gates"]}),
        check("behavior_scalars", min(core_accuracy) == behavior["core_sign_accuracy_min"] and min(null_accuracy) == behavior["null_sign_accuracy_min"] and min(effect_positive) == behavior["effect_positive_fraction_min"] and min(effect_median) == behavior["median_functional_effect_min"] and max(lexical_ratio) == behavior["lexical_specific_ratio_max"]),
        check("generation_cells", generation_cells == generation["cells"]),
        check("generation_gates", generation_gates == generation["gates"]),
        check("verdict", (final["authorization"] == "phase1283_qwen3_typed_multievent_causal_closure") == expected_pass),
        check("fp16", set(precision["parameter_dtypes"]) == {"float16"} and not precision["has_quantized_modules"] and not precision["has_bf16_parameters"]),
    ]
    result = {"phase": 1282, "audit_type": "independent_final_audit", "checks": checks, "passed_count": sum(row["passed"] for row in checks), "check_count": len(checks), "all_checks_passed": all(row["passed"] for row in checks), "scientific_gate_passed": expected_pass}
    atomic_json(FINAL_AUDIT, result)
    print(canonical_json(result))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("pre", "final"))
    args = parser.parse_args()
    preaudit() if args.action == "pre" else final_audit()
