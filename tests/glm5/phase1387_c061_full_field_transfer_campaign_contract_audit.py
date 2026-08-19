#!/usr/bin/env python3
"""Independent audit for Phase1387."""
from pathlib import Path
import json, sys

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core

OUT = TESTS / "result/phase1387_c061_full_field_transfer_campaign_contract"


def main() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    final = core.load(OUT / "analysis/final.json")
    pre = core.load(OUT / "audit/pre_model_semantic_naturalness_zero_model_audit.json")
    graph = core.load(OUT / "material/frozen_concept_graph.json")
    active = core.rows(OUT / "material/active_membership_cases.jsonl")
    status = core.rows(OUT / "material/status_cases.jsonl")
    pairs = core.rows(OUT / "material/candidate_pairs.jsonl")
    forbidden_text = " ".join(protocol["forbidden"]).lower()
    checks = {
        "contract_digest": core.digest({k: v for k, v in protocol.items()
                                         if k not in {"contract_sha256", "authorization"}}) == protocol["contract_sha256"],
        "authorization": final["authorization"] == "run_phase1388_c061_behavior_qualification",
        "preaudit": pre["all_checks_passed"] and pre["passed"] == pre["total"],
        "family_panels": len(graph["transfer_families"]) == len(graph["novel_families"]) == 4,
        "concepts": len(graph["concepts"]) == 96,
        "counts": (len(active), len(status), len(pairs)) == (1728, 576, 864),
        "hashes": core.sha(OUT / "material/active_membership_cases.jsonl") == protocol["material"]["active_sha256"]
                  and core.sha(OUT / "material/status_cases.jsonl") == protocol["material"]["status_sha256"]
                  and core.sha(OUT / "material/candidate_pairs.jsonl") == protocol["material"]["pair_sha256"],
        "partition_blind": set(protocol["coordinates"]["evaluation_partitions"]) == {"confirmation", "lockbox"},
        "field_discovery_only": protocol["observation"]["partition"] == "response_discovery",
        "all_states_positions": protocol["observation"]["all_hidden_state_indices"] == list(range(37))
                                and protocol["observation"]["all_physical_positions"],
        "fixed_primary": protocol["coordinates"]["primary_size"] == 512,
        "route_stop_not_campaign_stop": "only that frozen route" in protocol["stop_rule"],
        "scope": all(v in forbidden_text for v in ("attention", "mlp", "gradient", "pca", "probe")),
        "no_human_claim": not protocol["material"]["human_naturalness_lock"] and not pre["independent_human_blind_review"],
        "semantic_naturalness_disclosed": bool(pre["semantic_scope"] and pre["naturalness_scope"]),
    }
    result = {"phase": 1387, "checks": checks, "passed": sum(checks.values()), "total": len(checks),
              "all_checks_passed": all(checks.values())}
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
