#!/usr/bin/env python3
"""Independent audit for Phase1364 C056 contract."""
from __future__ import annotations

import json
import py_compile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
T = ROOT / "tests/glm5"
OUT = T / "result/phase1364_c056_hidden_path_contract"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    protocol = load(OUT / "protocol/preregistration.json")
    pre = load(OUT / "audit/pre_model_semantic_naturalness_zero_model_audit.json")
    cases = rows(OUT / "material/path_cases.jsonl")
    compiled = {row["case_id"]: row for row in rows(OUT / "compiled/extended_rows.jsonl")}
    paths = protocol["paths"]
    checks = {
        "preaudit": pre.get("all_checks_passed") is True,
        "case_count": len(cases) == 96,
        "balanced": set(Counter((row["partition"], row["surface"]) for row in cases).values()) == {8},
        "unique_pairs": len({row["pair_id"] for row in cases}) == len(cases),
        "compiled_coverage": all(row[key] in compiled for row in cases for key in
                                 ("clean_true", "corrupt_false", "wrong_identity_true", "status_true")),
        "role_positions": all(set(row["role_positions"]) == {"target", "family", "query", "boundary"}
                              for row in compiled.values()),
        "finite_paths": len(paths) == 5,
        "causal_order": all(path["source"]["layer"] < checkpoint["layer"]
                            for path in paths.values() for checkpoint in path["checkpoints"]),
        "single_write": protocol["causal"]["single_write_only"] is True,
        "hidden_only": "attention weights or heads" in protocol["forbidden"]
                       and "MLP states or weights" in protocol["forbidden"],
        "all_paths_survive_observation": protocol["observation"]["observation_failure_does_not_cancel_camera_or_causal"] is True,
        "finite_finish": "close C056" in protocol["branching"]["finish"],
        "authorization": protocol["authorization"] == "run_phase1365_c056_known_truth_camera",
    }
    py_compile.compile(str(T / "phase1364_c056_hidden_path_contract.py"), doraise=True)
    checks["script_compiles"] = True
    result = {
        "phase": 1364, "campaign": "C056", "checks": checks,
        "passed": sum(checks.values()), "total": len(checks),
        "all_checks_passed": all(checks.values()),
    }
    path = OUT / "audit/independent_final_audit.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
