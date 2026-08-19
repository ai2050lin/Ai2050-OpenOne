#!/usr/bin/env python3
"""Independent protocol and result audit for Phase1252."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "tests/glm5/phase1252_c005_answer_state_handoff.py"
AUDITOR = Path(__file__).resolve()
PHASE1251 = ROOT / "tests/glm5/phase1251_c004_causal_slice_competition.py"
MODEL = ROOT / "tests/glm5/phase1146_learned_composition_benchmark.py"
OUT = ROOT / "tests/glm5/result/phase1252_c005_answer_state_handoff"
PROTOCOL = OUT / "protocol/preregistration.json"
MATERIAL = OUT / "material/frozen_counterfactuals.jsonl"
ENVIRONMENT = OUT / "protocol/environment_snapshot.json"
PREAUDIT = OUT / "audit/independent_preaudit.json"
RAW = OUT / "raw/run_summary.json"
CURVES = OUT / "raw/handoff_curves.json"
COMPLETE = OUT / "raw/FORMAL_RUN_COMPLETE.json"
ANALYSIS = OUT / "analysis/handoff_adjudication.json"
FINAL = OUT / "analysis/final.json"
FINAL_AUDIT = OUT / "audit/independent_final_audit.json"

EXPECTED_DEPTHS = {"shallow4": 4, "middle6": 6, "deep8": 8}
EXPECTED_GROUPS = {
    "source": [4],
    "unqueried_source": [8],
    "mapping": list(range(11, 20)),
    "query": [20, 21],
    "answer": [22],
    "other": [0, 1, 2, 3, 5, 6, 7, 9, 10],
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    output = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            output.update(chunk)
    return output.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def valid_world(row: dict[str, Any]) -> bool:
    value = dict(row)
    stored = value.pop("row_digest", None)
    if stored != digest(value):
        return False
    names = ("base", "source", "mapping", "joint", "wrong_source", "wrong_mapping", "wrong_joint", "null")
    if any(len(row[f"{name}_ids"]) != 23 for name in names):
        return False
    if row["base_ids"][:4] != row["mapping_ids"][:4] or row["base_ids"][:4] != row["wrong_mapping_ids"][:4]:
        return False
    answers = row["answers"]
    if answers["base"] != answers["null"]:
        return False
    if len({answers["base"], answers["source"], answers["wrong_source"]}) != 3:
        return False
    if len({answers["base"], answers["mapping"], answers["wrong_mapping"]}) != 3:
        return False
    return True


def preaudit() -> None:
    protocol = read_json(PROTOCOL)
    rows = read_jsonl(MATERIAL)
    partition = sorted(position for values in protocol.get("position_partition", {}).values() for position in values)
    checks = {
        "schema_and_contract": protocol.get("phase") == 1252 and protocol.get("contract_id") == "EXP-C005-WP01-001",
        "source_hashes": protocol.get("source_hashes") == {
            "main": file_sha256(MAIN),
            "auditor": file_sha256(AUDITOR),
            "phase1251_dependency": file_sha256(PHASE1251),
            "model_dependency": file_sha256(MODEL),
        },
        "worlds_and_digests": len(rows) == 64 and len({row["group"] for row in rows}) == 64 and all(valid_world(row) for row in rows),
        "depth_breadth": {name: int(config["layers"]) for name, config in protocol.get("architectures", {}).items()} == EXPECTED_DEPTHS
        and protocol.get("replicates") == 2
        and len(set(protocol.get("model_seeds", {}).values())) == 6,
        "typed_position_partition": protocol.get("position_partition") == EXPECTED_GROUPS and partition == list(range(23)),
        "counterfactual_conditions": protocol.get("conditions") == ["source", "mapping", "joint"],
        "rescue_and_block_both_frozen": set(protocol.get("interventions", {})) >= {"rescue", "block", "origin", "wrong_identity"},
        "hard_stops_and_scope": len(protocol.get("hard_stops", [])) >= 6
        and any("not an attention edge" in item for item in protocol.get("hard_stops", []))
        and any("No Qwen3" in item for item in protocol.get("hard_stops", [])),
        "environment_recorded": ENVIRONMENT.exists() and read_json(ENVIRONMENT).get("cuda_available") is True,
    }
    report = {
        "phase": 1252,
        "audit_stage": "preaudit",
        "checks": checks,
        "passed": sum(bool(value) for value in checks.values()),
        "total": len(checks),
        "all_checks_passed": all(checks.values()),
    }
    report["audit_digest"] = digest(report)
    write_json(PREAUDIT, report)
    print(canonical_json(report))
    if not report["all_checks_passed"]:
        raise SystemExit(1)


def breadth(models: list[dict[str, Any]], field: str, minimum: int, per_depth_minimum: int) -> tuple[bool, dict[str, int]]:
    per_depth = {
        architecture: sum(bool(row.get(field)) for row in models if row["architecture"] == architecture)
        for architecture in EXPECTED_DEPTHS
    }
    return sum(bool(row.get(field)) for row in models) >= minimum and all(
        value >= per_depth_minimum for value in per_depth.values()
    ), per_depth


def final_audit() -> None:
    protocol = read_json(PROTOCOL)
    raw = read_json(RAW)
    curves = read_json(CURVES)
    complete = read_json(COMPLETE)
    analysis = read_json(ANALYSIS)
    final = read_json(FINAL)
    models = raw["models"]
    minimum = protocol["thresholds"]["breadth_models_min"]
    per_depth_minimum = protocol["thresholds"]["breadth_per_depth_min"]
    behavior, behavior_depth = breadth(models, "behavior_gate", minimum, per_depth_minimum)
    endpoint, endpoint_depth = breadth(models, "endpoint_instrument_gate", minimum, per_depth_minimum)
    specificity, specificity_depth = breadth(models, "specificity_gate", minimum, per_depth_minimum)
    identity, identity_depth = breadth(models, "identity_gate", minimum, per_depth_minimum)
    handoff, handoff_depth = breadth(models, "model_handoff_gate", minimum, per_depth_minimum)
    gates = {
        "G-BEHAVIOR-BREADTH": behavior,
        "G-ENDPOINT-INSTRUMENT": endpoint,
        "G-SPECIFICITY": specificity,
        "G-IDENTITY": identity,
        "G-HANDOFF-BREADTH": handoff,
    }
    per_depth = {
        "behavior": behavior_depth,
        "endpoint": endpoint_depth,
        "specificity": specificity_depth,
        "identity": identity_depth,
        "handoff": handoff_depth,
    }
    verdict = "known_truth_answer_state_handoff_confirmed" if all(gates.values()) else "known_truth_answer_state_handoff_not_confirmed"
    curve_structure = len(curves.get("models", [])) == sum(row.get("behavior_gate", False) for row in models)
    for model in curves.get("models", []):
        architecture = model["model_key"].rsplit("_r", 1)[0]
        layers = EXPECTED_DEPTHS[architecture]
        curve_structure = curve_structure and all(
            len(model["curves"][condition]) == layers + 1
            and all(set(layer["groups"]) == set(EXPECTED_GROUPS) | {"origin", "wrong_answer_identity"} for layer in model["curves"][condition])
            for condition in ("source", "mapping", "joint")
        )
    summary_structure = all(
        set(row.get("answer_write_onset", {})) == {"source", "mapping", "joint"}
        and set(row.get("answer_lock_in", {})) == {"source", "mapping", "joint"}
        and set(row.get("identity", {})) == {"source", "mapping", "joint"}
        for row in models if row.get("behavior_gate")
    )
    checks = {
        "one_shot_completion_marker": complete.get("status") == "formal_run_complete"
        and complete.get("run_digest") == raw.get("run_digest")
        and complete.get("raw_sha256") == file_sha256(RAW)
        and complete.get("curves_sha256") == file_sha256(CURVES),
        "protocol_and_curve_integrity": raw.get("protocol_digest") == protocol.get("protocol_digest")
        and raw.get("curves_sha256") == file_sha256(CURVES)
        and curves.get("protocol_digest") == protocol.get("protocol_digest"),
        "six_frozen_models_preserved": len(models) == 6
        and {row["architecture"] for row in models} == set(EXPECTED_DEPTHS)
        and all(sum(row["architecture"] == name for row in models) == 2 for name in EXPECTED_DEPTHS),
        "no_pretrained_model": raw.get("pretrained_model_loaded") is False,
        "curve_structure": curve_structure,
        "summary_structure": summary_structure,
        "gates_recomputed": analysis.get("gates") == gates and final.get("gates") == gates,
        "depth_breadth_recomputed": analysis.get("per_depth") == per_depth and final.get("per_depth") == per_depth,
        "verdict_recomputed": analysis.get("verdict") == verdict and final.get("verdict") == verdict,
        "authorization_scope": analysis.get("authorization", {}).get("semantic_mechanism_claim") is False
        and analysis.get("authorization", {}).get("attention_edge_claim") is False
        and analysis.get("authorization", {}).get("cross_model_claim") is False,
        "final_artifact_hashes": final.get("artifact_hashes", {}).get("raw") == file_sha256(RAW)
        and final.get("artifact_hashes", {}).get("curves") == file_sha256(CURVES)
        and final.get("artifact_hashes", {}).get("complete") == file_sha256(COMPLETE)
        and final.get("artifact_hashes", {}).get("analysis") == file_sha256(ANALYSIS),
    }
    report = {
        "phase": 1252,
        "audit_stage": "final",
        "checks": checks,
        "passed": sum(bool(value) for value in checks.values()),
        "total": len(checks),
        "all_checks_passed": all(checks.values()),
        "recomputed_verdict": verdict,
        "recomputed_gates": gates,
    }
    report["audit_digest"] = digest(report)
    write_json(FINAL_AUDIT, report)
    print(canonical_json(report))
    if not report["all_checks_passed"]:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("preaudit", "final"))
    args = parser.parse_args()
    if args.stage == "preaudit":
        preaudit()
    else:
        final_audit()


if __name__ == "__main__":
    main()
