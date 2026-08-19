#!/usr/bin/env python3
"""Independent audit for Phase1390."""
from pathlib import Path
import json, sys
ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"; sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core
OUT = TESTS / "result/phase1390_c062_route_factorized_field_campaign_contract"


def main() -> None:
    p = core.load(OUT / "protocol/preregistration.json"); f = core.load(OUT / "analysis/final.json")
    pre = core.load(OUT / "audit/pre_model_semantic_naturalness_zero_model_audit.json")
    graph = core.load(OUT / "material/frozen_concept_graph.json")
    checks = {
        "digest": core.digest({k: v for k, v in p.items() if k not in {"contract_sha256", "authorization"}}) == p["contract_sha256"],
        "authorization": f["authorization"] == "run_phase1391_c062_family_factorized_behavior",
        "preaudit": pre["all_checks_passed"], "families": len(graph["families"]) == 6,
        "route_factorized": p["behavior"]["route_rule"].startswith("qualify or eliminate each family independently"),
        "breadth": p["material"]["minimum_qualified_families"] == 4,
        "transfer_breadth": p["material"]["minimum_qualified_transfer_families"] == 1,
        "novel_breadth": p["material"]["minimum_qualified_novel_families"] == 2,
        "discovery_only": p["observation"]["candidate_source_is_discovery_only"],
        "scope": all(x in " ".join(p["forbidden"]).lower() for x in ("attention", "mlp", "gradient", "pca", "probe")),
        "ambiguity_disclosed": bool(pre["known_ambiguity_risk"]),
        "no_human_claim": not p["material"]["human_naturalness_lock"],
    }
    result = {"phase": 1390, "checks": checks, "passed": sum(checks.values()), "total": len(checks),
              "all_checks_passed": all(checks.values())}
    core.save(OUT / "audit/independent_final_audit.json", result); print(json.dumps(result, indent=2))
    if not result["all_checks_passed"]: raise SystemExit(1)


if __name__ == "__main__": main()
