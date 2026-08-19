#!/usr/bin/env python3
"""Independent audit for Phase1250."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/glm5/result/phase1250_c003_multievent_response_camera"
PROTOCOL = OUT / "protocol/preregistration.json"
MATERIAL = OUT / "material/frozen_worlds.jsonl"
PREAUDIT = OUT / "audit/independent_preaudit.json"
RAW = OUT / "raw/run_summary.json"
ARRAYS = OUT / "raw/camera_arrays.npz"
ANALYSIS = OUT / "analysis/camera_adjudication.json"
FINAL = OUT / "analysis/final.json"
FINAL_AUDIT = OUT / "audit/independent_final_audit.json"


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def rows() -> list[dict[str, Any]]:
    return [json.loads(line) for line in MATERIAL.read_text(encoding="utf-8").splitlines() if line.strip()]


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def add(checks: list[dict[str, Any]], name: str, passed: bool, detail: Any = None) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def preaudit() -> None:
    protocol = read(PROTOCOL); material = rows(); checks = []
    add(checks, "row_count", len(material) == 800, len(material))
    counts = {name: len({row["group"] for row in material if row["partition"] == name}) for name in ("discovery", "selection", "confirmation")}
    add(checks, "partition_counts", counts == protocol["partitions"], counts)
    add(checks, "camera_families_frozen", len(protocol["camera"]["families"]) == 5)
    add(checks, "equal_input_budget", protocol["camera"]["input_dimension_each"] == 80)
    add(checks, "confirmation_sealed", protocol["camera"]["confirmation"] == "sealed worlds, alpha=1")
    add(checks, "independent_model_seeds", len(set(protocol["model_seeds"].values())) == 4)
    add(checks, "two_architectures", len(protocol["architectures"]) == 2)
    add(checks, "source_collision_design", all(len({tuple(row["receiver_ids"][:5]) for row in material if row["group"] == group and row["representation"] == "code"}) == 1 for group in range(100)))
    add(checks, "hard_stop_scope", protocol["hard_stops"][2].startswith("A pass does not prove"))
    payload = {"phase": 1250, "stage": "preaudit", "checks": checks, "all_checks_passed": all(item["passed"] for item in checks)}
    write(PREAUDIT, payload); print(json.dumps({"checks": len(checks), "passed": payload["all_checks_passed"]}))


def final_audit() -> None:
    protocol = read(PROTOCOL); run = read(RAW); analysis = read(ANALYSIS); final = read(FINAL); checks = []
    add(checks, "protocol_link", run["protocol_digest"] == protocol["protocol_digest"] == analysis["protocol_digest"])
    add(checks, "array_hash", sha(ARRAYS) == run["array_sha256"])
    add(checks, "four_models", len(run["models"]) == 4)
    add(checks, "no_pretrained", run["pretrained_model_loaded"] is False)
    add(checks, "family_budget", all(len(row.get("camera_families", {})) == 5 for row in run["models"] if row.get("behavior_gate")))
    recomputed = {
        "G-BEHAVIOR": all(row.get("behavior_gate") for row in run["models"]),
        "G-NONIDENTIFIABILITY": all(row.get("source_collision", {}).get("exact_collision_fraction") == 1.0 and row.get("source_collision", {}).get("response_separated_fraction") == 1.0 for row in run["models"] if row.get("behavior_gate")),
        "G-MULTIEVENT": sum(row.get("model_gate", False) for row in run["models"]) >= protocol["thresholds"]["passing_models_min"] and all(sum(row.get("model_gate", False) for row in run["models"] if row["architecture"] == architecture) >= 1 for architecture in protocol["architectures"]),
    }
    add(checks, "gates_recomputed", recomputed == analysis["gates"], recomputed)
    expected_verdict = "known_truth_multievent_camera_confirmed" if all(recomputed.values()) else "known_truth_multievent_camera_not_confirmed"
    add(checks, "verdict_recomputed", expected_verdict == analysis["verdict"] == final["verdict"])
    add(checks, "authorization_typed", final["next_phase_authorized"] == all(recomputed.values()) and final["semantic_mechanism_claim_authorized"] is False)
    add(checks, "artifact_hashes", all(final["artifact_hashes"][name] == sha(path) for name, path in {"protocol": PROTOCOL, "material": MATERIAL, "raw": RAW, "arrays": ARRAYS, "analysis": ANALYSIS}.items()))
    payload = {"phase": 1250, "stage": "final_audit", "checks": checks, "all_checks_passed": all(item["passed"] for item in checks)}
    write(FINAL_AUDIT, payload); print(json.dumps({"checks": len(checks), "passed": payload["all_checks_passed"]}))


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("command", choices=("preaudit", "final")); args = parser.parse_args()
    preaudit() if args.command == "preaudit" else final_audit()


if __name__ == "__main__":
    main()
