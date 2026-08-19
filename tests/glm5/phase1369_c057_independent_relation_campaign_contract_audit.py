#!/usr/bin/env python3
"""Independent audit for Phase1369 C057 contract."""
from __future__ import annotations

import json
import py_compile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1369_c057_independent_relation_campaign_contract"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    graph = load(OUT / "material/frozen_concept_graph.json")
    active = rows(OUT / "material/active_membership_cases.jsonl")
    status = rows(OUT / "material/status_cases.jsonl")
    pairs = rows(OUT / "material/candidate_pairs.jsonl")
    ca = rows(OUT / "compiled/qwen3_active.jsonl")
    cs = rows(OUT / "compiled/qwen3_status.jsonl")
    protocol = load(OUT / "protocol/preregistration.json")
    preaudit = load(OUT / "audit/pre_model_semantic_naturalness_zero_model_audit.json")
    final = load(OUT / "analysis/final.json")
    checks = {
        "concepts": len(graph["concepts"]) == 48 and len({r["word"] for r in graph["concepts"]}) == 48,
        "active": len(active) == len(ca) == 864 and Counter(r["truth"] for r in active) == {True: 432, False: 432},
        "status": len(status) == len(cs) == 288 and Counter(r["truth"] for r in status) == {True: 144, False: 144},
        "pairs": len(pairs) == 432 and len({r["pair_id"] for r in pairs}) == 432,
        "family_spans": all(len(r["tested_family_span"]) == 1 for r in ca + cs),
        "roles": all(set(r["role_positions"]) == {"target", "family", "query", "boundary"} for r in ca + cs),
        "preaudit": preaudit["all_checks_passed"] and preaudit["passed"] == preaudit["total"],
        "hashes": protocol["material"]["active_sha256"] == _sha(OUT / "material/active_membership_cases.jsonl")
                  and protocol["material"]["status_sha256"] == _sha(OUT / "material/status_cases.jsonl")
                  and protocol["material"]["pair_sha256"] == _sha(OUT / "material/candidate_pairs.jsonl"),
        "finite_routes": set(protocol["coordinate_groups"]["routes"]) == {"magnitude", "stable_sign", "family_min", "deterministic_random"}
                         and protocol["coordinate_groups"]["sizes"][-1] == 2560,
        "hidden_only": "attention" in protocol["forbidden"] and "MLP" in protocol["forbidden"]
                       and "PCA" in protocol["forbidden"],
        "authorization": final["authorization"] == "run_phase1370_c057_behavior_qualification",
    }
    py_compile.compile(str(TESTS / "phase1369_c057_independent_relation_campaign_contract.py"), doraise=True)
    py_compile.compile(str(TESTS / "phase1369_c057_independent_relation_campaign_contract_audit.py"), doraise=True)
    checks["scripts_compile"] = True
    result = {"phase": 1369, "campaign": "C057", "checks": checks,
              "passed": sum(checks.values()), "total": len(checks),
              "all_checks_passed": all(checks.values())}
    path = OUT / "audit/independent_final_audit.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


def _sha(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":
    main()
