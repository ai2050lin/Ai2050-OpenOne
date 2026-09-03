#!/usr/bin/env python3
"""Independent audit for C501-C516."""
from __future__ import annotations

import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
OUT = RESULT / "phase2051_c517_embedding_conditioned_state_campaign_independent_audit"
VISUAL = ROOT / "frontend/public/vis_data/research_kernel/c516_embedding_conditioned_state_atlas.json"
REGISTRY = ROOT / "ai2050_research_os/registry/field_datasets.json"
CATALOG = ROOT / "frontend/public/research_data/current/language_encoding_catalog.json"
MAIN = TESTS / "phase2035_c501_c516_embedding_conditioned_state_campaign.py"
sys.path.insert(0, str(TESTS))

import phase2035_c501_c516_embedding_conditioned_state_campaign as campaign


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def finite(value) -> bool:
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, list):
        return all(finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def main() -> None:
    if OUT.exists():
        raise RuntimeError(OUT)
    (OUT / "analysis").mkdir(parents=True)
    (OUT / "audit").mkdir(parents=True)
    (OUT / "protocol").mkdir(parents=True)
    checks = []

    def check(name: str, passed: bool, detail=None) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    main_hash = sha(MAIN)
    finals = {}
    for offset, number in enumerate(range(501, 517)):
        name = f"C{number}"
        final_path = campaign.OUTS[name] / "analysis/final.json"
        protocol_path = campaign.OUTS[name] / "protocol/preregistration.json"
        check(f"{name}.final_exists", final_path.exists())
        check(f"{name}.protocol_exists", protocol_path.exists())
        final = load(final_path)
        protocol = load(protocol_path)
        finals[name] = final
        check(f"{name}.phase", final["phase"] == 2035 + offset, final["phase"])
        check(f"{name}.closed", final["status"] == "closed")
        check(f"{name}.all_checks", final["all_checks_passed"])
        check(f"{name}.producer_hash", protocol["producer_sha256"] == main_hash)

    check("C502.rows", finals["C502"]["headline"]["rows"] == 1440)
    check("C502.family_counts", set(finals["C502"]["headline"]["family_counts"].values()) == {240})
    c503 = finals["C503"]["headline"]
    check("C503.balance", all(c503["checks"].values()))
    check("C503.width", c503["max_prompt_tokens"] <= 144, c503["max_prompt_tokens"])
    c504 = finals["C504"]["headline"]
    check("C504.behavior", c504["accuracy"] >= 0.80, c504["accuracy"])
    check("C504.family_eligible", len(c504["eligible_families"]) == 6)
    c505 = finals["C505"]["headline"]
    check("C505.rows", c505["rows"] == 6720)
    check("C505.shape", c505["role_shape"] == [6720, 38, 6, 2560], c505["role_shape"])
    check("C505.full_shape", c505["full_shape"] == [272, 10, 118, 2560], c505["full_shape"])
    c506 = finals["C506"]["headline"]
    check("C506.basis_equivalence", c506["basis_equivalent_within_tolerance"])
    check("C506.no_walsh_specificity", not c506["walsh_specificity_supported"])
    check("C506.numeric_tolerance", c506["max_nrmse_delta"] <= 0.001, c506["max_nrmse_delta"])
    c507 = finals["C507"]["headline"]
    check("C507.surface_embedding_exact", c507["surface_pair_embedding_max_abs_max"] == 0.0)
    check("C507.polysemy_embedding_exact", c507["polysemy_embedding_max_abs_max"] == 0.0)
    check("C507.polysemy_context_diff", c507["polysemy_q24_query_rms_difference_mean"] > 0.0)
    c508 = finals["C508"]["headline"]
    check("C508.embedding_only_rejected", not c508["embedding_only_candidate"])
    check("C508.all_gains_negative", all(value < 0 for value in c508["nrmse_gains_over_family_mean"].values()))
    c509 = finals["C509"]["headline"]
    check("C509.incremental_rejected", not c509["embedding_incremental_candidate"])
    check("C509.all_joint_gains_negative", all(value < 0 for value in c509["joint_nrmse_gains_over_state"].values()))
    c510 = finals["C510"]["headline"]
    check("C510.pairs", c510["pairs_edges"] == 600)
    check("C510.embedding_not_identifiable", not c510["embedding_alone_sense_identifiable"])
    check("C510.target_diff", c510["target_pair_rms_mean"] > 0.0)
    c511 = finals["C511"]["headline"]
    check("C511.aggregate_gate_strict", not c511["family_conditioned_candidate"])
    check("C511.gain_below_gate", c511["family_conditioned_nrmse_gain"] < 0.01, c511["family_conditioned_nrmse_gain"])
    check("C511.family_better_than_shared", c511["metrics"]["family"]["nrmse"] < c511["metrics"]["shared"]["nrmse"])
    check("C512.nested_negative", not finals["C512"]["headline"]["high_order_candidate"])
    check("C513.graph_positive", finals["C513"]["headline"]["high_order_candidate"])
    check("C514.temporal_positive", finals["C514"]["headline"]["high_order_candidate"])
    c515 = finals["C515"]["headline"]
    check("C515.single_sample_rejected", not c515["embedding_single_sample_candidate"])
    check("C515.panel_joint_rejected", not c515["complete_panel_candidate"])
    check("C515.no_causal_run", not c515["causal"]["ran"])

    check("visual.exists", VISUAL.exists())
    visual = load(VISUAL)
    check("visual.schema", visual["schema"] == "ai2050.embedding_conditioned_state_atlas.v1")
    check("visual.coordinates", visual["coordinate_count"] == 2560)
    check("visual.rows", len(visual["rows"]) == 102, len(visual["rows"]))
    check("visual.full_vectors", all(
        len(row["embedding_q0"]) == 2560 and len(row["state_q24"]) == 2560 and len(row["write_q24_q25"]) == 2560
        for row in visual["rows"]
    ))
    check("visual.finite", finite(visual))
    registry = load(REGISTRY)
    catalog = load(CATALOG)
    check("registry.entry", any(row.get("id") == "c516_embedding_conditioned_state_atlas" for row in registry.get("datasets", [])))
    check("catalog.entry", any(row.get("id") == "c516_embedding_conditioned_state_atlas" for row in catalog.get("field_datasets", [])))
    cleanup = load(campaign.OUTS["C516"] / "audit/raw_field_cleanup_ledger.json")
    check("cleanup.two_fields", len(cleanup["files"]) == 2)
    check("cleanup.bytes", cleanup["total_bytes"] > 0, cleanup["total_bytes"])
    for row in cleanup["files"]:
        check(f"cleanup.absent.{Path(row['path']).name}", not (ROOT / row["path"]).exists())
        check(f"cleanup.hash.{Path(row['path']).name}", len(row["sha256"]) == 64)

    passed = sum(row["passed"] for row in checks)
    audit = {
        "status": "passed" if passed == len(checks) else "failed",
        "passed": passed,
        "total": len(checks),
        "checks": checks,
        "evidence_adjudication": {
            "retained": [
                "Equal-capacity raw, orthonormal Walsh, and random orthogonal bases are numerically equivalent in the registered complete-channel test.",
                "q0 token embedding alone neither identifies explicit polysemous sense nor improves the registered local-write predictors.",
                "Family-conditioned channel-diagonal transition improves aggregate prediction by 0.00933 NRMSE, below the frozen 0.01 aggregate gate.",
                "Typed-graph and temporal high-order lockboxes pass locally; nested composition fails.",
            ],
            "forbidden_overclaims": [
                "Walsh is the model's natural semantic basis.",
                "Token embedding is a complete lexical routing key.",
                "The graph or temporal panel predictor is callable from one sentence.",
                "A unique coordinate circuit, causal mechanism, or new mathematics has been found.",
            ],
        },
    }
    save(OUT / "audit/independent_audit.json", audit)
    protocol = {
        "phase": 2051,
        "campaign": "C517",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "auditor_sha256": sha(Path(__file__)),
        "producer_sha256": main_hash,
        "scope": "independent artifact, metric, basis, visual, cleanup, and claim-boundary audit for C501-C516",
    }
    save(OUT / "protocol/preregistration.json", protocol)
    final = {
        "phase": 2051,
        "campaign": "C517",
        "status": "closed",
        "all_checks_passed": audit["status"] == "passed",
        "headline": {
            "status": audit["status"],
            "checks_passed": passed,
            "checks_total": len(checks),
            "basis_result": "complete_information_not_Walsh_specific",
            "embedding_result": "q0_embedding_not_incrementally_qualified",
            "conditioned_result": "aggregate_gate_failed_but_graph_and_temporal_high_order_local_candidates_retained",
            "causal": "NA_no_single_sample_predictor",
            "next_stage_same_goal": True,
            "next_stage": "fresh-vocabulary graph/temporal replication and single-sample context-conditioned callable-state search; nested route remains a negative control",
        },
        "next_authorization": "C518_fresh_callable_state_campaign",
    }
    save(OUT / "analysis/final.json", final)
    print(json.dumps(final, ensure_ascii=False, indent=2))
    if audit["status"] != "passed":
        failed = [row for row in checks if not row["passed"]]
        raise RuntimeError(failed)


if __name__ == "__main__":
    main()
