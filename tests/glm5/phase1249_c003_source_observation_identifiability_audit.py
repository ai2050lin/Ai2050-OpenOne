#!/usr/bin/env python3
"""Independent audit for Phase1249 source-observation identifiability."""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "tests/glm5/phase1249_c003_source_observation_identifiability.py"
AUDITOR = Path(__file__).resolve()
UPSTREAM = ROOT / "tests/glm5/result/phase1248_c002_qwen_self_response_atlas"
MATERIAL = UPSTREAM / "material/frozen_worlds.jsonl"
ARRAYS = UPSTREAM / "raw/response_arrays.npz"
OUT = ROOT / "tests/glm5/result/phase1249_c003_source_observation_identifiability"
PROTOCOL = OUT / "protocol/preregistration.json"
PREAUDIT = OUT / "audit/independent_preaudit.json"
ANALYSIS = OUT / "analysis/observation_sufficiency.json"
FINAL = OUT / "analysis/final.json"
FINAL_AUDIT = OUT / "audit/independent_final_audit.json"
TOL = 1.0e-5
SEP = 1.0


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def rows() -> list[dict[str, Any]]:
    return [json.loads(line) for line in MATERIAL.read_text(encoding="utf-8").splitlines() if line.strip()]


def check(name: str, passed: bool, details: Any = None) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "details": details}


def preaudit() -> None:
    protocol = read(PROTOCOL)
    source = MAIN.read_text(encoding="utf-8")
    checks = [
        check("main_hash", protocol["source_hashes"]["main"] == sha(MAIN)),
        check("auditor_hash", protocol["source_hashes"]["auditor"] == sha(AUDITOR)),
        check("material_hash", protocol["source_hashes"]["material"] == sha(MATERIAL)),
        check("array_hash", protocol["source_hashes"]["arrays"] == sha(ARRAYS)),
        check("fixed_event", protocol["fixed_object"]["event_id"] == "residual_source_d06"),
        check("fixed_alpha", protocol["fixed_object"]["alpha"] == 1.0),
        check("no_model_import", "transformers" not in source and "load_model" not in source),
        check("no_fit", "ridge_fit" not in source and "LinearRegression" not in source),
        check("hard_stop", any("No model or GPU" in item for item in protocol["hard_stops"])),
    ]
    payload = {"phase": 1249, "mode": "preaudit", "check_count": len(checks), "checks": checks, "all_checks_passed": all(item["passed"] for item in checks)}
    write(PREAUDIT, payload)
    print(json.dumps({"status": "preaudit", "checks": len(checks), "passed": payload["all_checks_passed"]}, separators=(",", ":")))
    if not payload["all_checks_passed"]:
        raise SystemExit(1)


def final_audit() -> None:
    protocol = read(PROTOCOL)
    analysis = read(ANALYSIS)
    final = read(FINAL)
    data = rows()
    with np.load(ARRAYS) as arrays:
        x = arrays["target_projected"][:, 0] - arrays["receiver_projected"][:, 0]
        y = arrays["responses"][:, 0, 0, 3]
    groups: dict[tuple[str, str, int], list[int]] = {}
    for index, row in enumerate(data):
        if row["partition"] == "confirmation":
            groups.setdefault((row["world_id"], row["representation"], int(row["receiver_state"])), []).append(index)
    records = []
    prefix_ok = True
    for key, indices in sorted(groups.items()):
        xdist = [float(np.linalg.norm(x[a] - x[b])) for a, b in combinations(indices, 2)]
        ydist = [float(np.linalg.norm(y[a] - y[b])) for a, b in combinations(indices, 2)]
        mapping = []
        interface = []
        for a, b in combinations(indices, 2):
            if data[a]["interface"] == data[b]["interface"] and data[a]["mapping"] != data[b]["mapping"]:
                mapping.append(float(np.linalg.norm(y[a] - y[b])))
            if data[a]["mapping"] == data[b]["mapping"] and data[a]["interface"] != data[b]["interface"]:
                interface.append(float(np.linalg.norm(y[a] - y[b])))
        for variant in ("receiver", "target"):
            prefixes = []
            for index in indices:
                prompt = data[index]["variants"][variant]["prompt"]
                end = int(data[index]["variants"][variant]["source_span"][1])
                prefixes.append(prompt[:end])
            prefix_ok = prefix_ok and len(set(prefixes)) == 1
        records.append({
            "representation": key[1],
            "exact": max(xdist) <= TOL,
            "separated": max(xdist) <= TOL and max(ydist) >= SEP,
            "mapping_mean": float(np.mean(mapping)),
            "interface_mean": float(np.mean(interface)),
        })
    code = [item for item in records if item["representation"] == "code"]
    direct = [item for item in records if item["representation"] == "direct"]
    recomputed = {
        "G-ALPHA-IMPLEMENTATION": "train_x.append(float(alpha) * delta[discovery_index])" in (ROOT / "tests/glm5/phase1247_c002_hidden_response_imaging_camera.py").read_text(encoding="utf-8") and "return float(alpha) * (arrays[donor_key][indices, event] - arrays[\"receiver_projected\"][indices, event])" in (ROOT / "tests/glm5/phase1248_c002_qwen_self_response_atlas.py").read_text(encoding="utf-8"),
        "G-CAUSAL-PREFIX": prefix_ok,
        "G-EXACT-SOURCE-COLLISION": all(item["exact"] for item in records),
        "G-CODE-RESPONSE-SEPARATION": float(np.mean([item["mapping_mean"] for item in code])) >= SEP,
        "G-DIRECT-CODE-DIFFERENCE": float(np.mean([item["mapping_mean"] for item in code])) > float(np.mean([item["mapping_mean"] for item in direct])),
    }
    checks = [
        check("preaudit_passed", read(PREAUDIT)["all_checks_passed"]),
        check("upstream_hash", protocol["source_hashes"]["arrays"] == sha(ARRAYS)),
        check("group_count", len(records) == 64),
        check("gates_recomputed", analysis["gates"] == recomputed, recomputed),
        check("verdict", final["verdict"] == ("source_only_observation_nonidentifiable_across_later_context" if all(recomputed.values()) else "collision_hypothesis_not_confirmed")),
        check("no_qwen_authority", final["qwen_rerun_authorized"] is False),
        check("no_evidence_upgrade", analysis["authorization"]["phase1248_evidence_upgrade"] is False),
    ]
    payload = {"phase": 1249, "mode": "final", "check_count": len(checks), "checks": checks, "all_checks_passed": all(item["passed"] for item in checks), "recomputed_gates": recomputed}
    write(FINAL_AUDIT, payload)
    print(json.dumps({"status": "final_audit", "checks": len(checks), "passed": payload["all_checks_passed"]}, separators=(",", ":")))
    if not payload["all_checks_passed"]:
        raise SystemExit(1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("preaudit", "final"), required=True)
    args = parser.parse_args()
    preaudit() if args.mode == "preaudit" else final_audit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
