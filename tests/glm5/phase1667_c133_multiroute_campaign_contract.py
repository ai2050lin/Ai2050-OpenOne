#!/usr/bin/env python3
"""C133 preregistration for the A-E observation-first language mechanism campaign."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1667_c133_multiroute_campaign_contract"
C132 = TESTS / "result/phase1666_c132_fixed_frame_composed_precedence"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    if OUT.exists():
        raise RuntimeError(f"C133 exists: {OUT}")
    parent = core.load(C132 / "analysis/closure.json")
    parent_audit = core.load(C132 / "audit/major_stage_audit.json")
    routes = {
        "A": {
            "object": "directed-link direct, two-hop, alternative-path, shortcut, reversal, and irrelevant-edge families",
            "behavior_first": True,
            "models": ["qwen3"],
            "observations": ["registered-role embedding and all post-block pre-final-norm HiddenStates", "final norm", "all 2560 activation coordinates"],
            "prediction": "freeze discovery direct-to-composed checkpointwise maps and predict untouched lexical confirmation trajectories",
            "failure_branch": "behavior failure closes A internal route only",
        },
        "B": {
            "object": "all-token full-coordinate atlas on a frozen anchor panel plus large-sample role field inherited from A",
            "anchor_limit": 12,
            "resource_reason": "full 256-case all-token x 38 x 2560 BF16 capture would exceed practical disk budget",
            "claim": "effective predictive dependency only; no unique causal circuit without intervention",
        },
        "C": {
            "object": "Chinese experiencer-like/eat/agent/patient/coreference/negation composition family centered on natural lexical variation",
            "behavior_first": True,
            "model": "qwen3",
            "claim": "pattern-composition observation, not isolated directions",
        },
        "D": {
            "object": "artificial is-a type graph with node renaming, edge direction, path depth, shortcut and distractors; natural apple panel external only",
            "behavior_first": True,
            "model": "qwen3",
            "claim": "response ecology over relations and queries, not an apple vector",
        },
        "E": {
            "object": "prospective fresh lexical/surface/depth replication and sequential qwen3/glm4/deepseek7b comparison",
            "models": ["qwen3", "glm4", "deepseek7b"],
            "comparison": ["relative layer depth", "role topology", "response-signature similarity after within-model normalization"],
            "forbidden": "same physical coordinate number across models",
        },
    }
    protocol = {
        "phase": 1667,
        "campaign": "C133",
        "created_at_utc": now(),
        "status": "five_route_observation_first_campaign_frozen",
        "parent_result": parent["status"],
        "priority": ["observe", "find repeatable structure", "predict unseen trajectories", "causal adjudication last"],
        "routes": routes,
        "shared_partitions": ["discovery", "confirmation", "lockbox_where_feasible"],
        "shared_measurement": {"checkpoint_types": ["embedding", "post_each_decoder_block_pre_final_norm", "post_final_norm"], "coordinates": "all physical activation coordinates", "forbidden": ["PCA", "SVD", "attention inspection", "MLP inspection", "weight interpretation"]},
        "route_policy": "failure closes only the named route; continue with all other preregistered routes",
        "causal_gate": {"required": ["behavior-qualified object", "discovery prediction frozen", "untouched new-vocabulary HiddenState trajectory predicted", "matched wrong-route control"], "interventions": ["delete candidate state support", "correct rescue", "wrong relation rescue", "wrong role/checkpoint rescue", "free generation", "unrelated behavior side effect"]},
        "mathematical_scope": "finite differences, coordinatewise arithmetic, exact counting, and conditional response fields; no new-mathematics claim before stable cross-family composition laws",
        "resource_limits": {"minimum_free_disk_gb": 40, "models_sequential": True, "full_token_anchor_max": 12, "large_sample_full_token_not_claimed": True},
        "claim_boundary": "a campaign plan, not model evidence or a discovered operator",
        "source_paths": {"c132_closure": str(C132 / "analysis/closure.json"), "c132_major_audit": str(C132 / "audit/major_stage_audit.json")},
        "source_hashes": {"c132_closure": core.sha(C132 / "analysis/closure.json"), "c132_major_audit": core.sha(C132 / "audit/major_stage_audit.json")},
        "producer_sha256": core.sha(Path(__file__)),
        "authorization": "start_route_A_C134",
    }
    checks = {
        "parent_closed": parent_audit["all_checks_passed"] and parent_audit["authorization"] == "composition_branch_closed_start_new_observation_contract",
        "five_routes": set(routes) == set("ABCDE"),
        "route_level_failure": protocol["route_policy"].startswith("failure closes only"),
        "causal_last": len(protocol["causal_gate"]["required"]) == 4,
        "typed_states": len(protocol["shared_measurement"]["checkpoint_types"]) == 3,
        "no_attention_mlp": "attention inspection" in protocol["shared_measurement"]["forbidden"] and "MLP inspection" in protocol["shared_measurement"]["forbidden"],
        "resource_bound": protocol["resource_limits"]["full_token_anchor_max"] == 12 and protocol["resource_limits"]["models_sequential"],
        "source_hashes": all(core.sha(Path(protocol["source_paths"][name])) == digest for name, digest in protocol["source_hashes"].items()),
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "audit/internal_contract_audit.json", {"phase": 1667, "campaign": "C133", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "authorization": protocol["authorization"]})
    print(json.dumps({"checks": checks, "authorization": protocol["authorization"]}, indent=2))


if __name__ == "__main__":
    main()
