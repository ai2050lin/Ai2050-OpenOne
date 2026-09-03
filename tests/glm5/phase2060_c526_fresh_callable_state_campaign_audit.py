#!/usr/bin/env python3
"""Independent audit for C518-C525, preserving C519 missingness."""
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
OUT = RESULT / "phase2060_c526_fresh_callable_state_campaign_independent_audit"
VISUAL = ROOT / "frontend/public/vis_data/research_kernel/c525_fresh_callable_state_atlas.json"
REGISTRY = ROOT / "ai2050_research_os/registry/field_datasets.json"
CATALOG = ROOT / "frontend/public/research_data/current/language_encoding_catalog.json"
MAIN = TESTS / "phase2052_c518_c525_fresh_callable_state_campaign.py"
sys.path.insert(0, str(TESTS))

import phase2052_c518_c525_fresh_callable_state_campaign as campaign


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
    for sub in ("analysis", "audit", "protocol"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    checks = []

    def check(name, passed, detail=None):
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    main_hash = sha(MAIN)
    finals = {}
    for offset, number in enumerate(range(518, 526)):
        name = f"C{number}"
        fp = campaign.OUTS[name] / "analysis/final.json"
        pp = campaign.OUTS[name] / "protocol/preregistration.json"
        check(f"{name}.final", fp.exists())
        check(f"{name}.protocol", pp.exists())
        final = load(fp)
        protocol = load(pp)
        finals[name] = final
        check(f"{name}.phase", final["phase"] == 2052 + offset, final["phase"])
        check(f"{name}.closed", final["status"] == "closed")
        if name == "C519":
            check("C519.formal_failure_preserved", not final["all_checks_passed"])
            check("C519.analysis_false", not final["checks"]["analysis"])
        else:
            check(f"{name}.checks", final["all_checks_passed"])
        check(f"{name}.producer", protocol["producer_sha256"] == main_hash)

    check("C518.rows", finals["C518"]["headline"]["rows"] == 1440)
    c519 = finals["C519"]["headline"]
    check("C519.width_miss_exact", c519["max_prompt_tokens"] == 129, c519["max_prompt_tokens"])
    check("C519.repair_count", c519["premodel_role_repairs"] == 240)
    check("C519.behavior_complete", c519["rows"] == 1440)
    check("C519.behavior_after_width_hardness", c519["status"] == "audit_behavior_closed_after_premodel_metadata_repair")
    c520 = finals["C520"]["headline"]
    check("C520.missingness_propagated", not c520["upstream_C519_formal_pass"])
    check("C520.rows", c520["rows"] == 2880)
    check("C520.shape", c520["role_shape"] == [2880, 38, 6, 2560])
    c521 = finals["C521"]["headline"]
    check("C521.graph_not_replicated", not c521["family_candidates"]["typed_graph_path"])
    check("C521.temporal_not_replicated", not c521["family_candidates"]["temporal_composition"])
    check("C521.nested_frozen_gate", c521["family_candidates"]["nested_composition"])
    check("C521.nested_not_identity_better", c521["metrics"]["nested_composition"]["family"]["nrmse"] > c521["metrics"]["nested_composition"]["identity"]["nrmse"])
    c522 = finals["C522"]["headline"]
    check("C522.single_sample_rejected", not c522["shared_single_sample_candidate"])
    check("C522.core_rejected", not c522["q24_q25_graph_temporal_context_candidate"])
    check("C522.aggregate_worse_mean", c522["aggregate"]["shared_bundle"]["nrmse"] > c522["aggregate"]["mean"]["nrmse"])
    c523 = finals["C523"]["headline"]
    check("C523.weak_gate_pass", c523["autonomous_shared_candidate"])
    check("C523.all_persistence_gains", all(value >= 0.02 for value in c523["shared_final_gains_over_persistence"].values()))
    c524 = finals["C524"]["headline"]
    check("C524.no_causal", not c524["causal"]["ran"])
    check("C524.not_authorized", not c524["causal"]["authorized"])
    c525 = finals["C525"]["headline"]
    check("C525.raw_absent", c525["raw_fields_absent"])
    check("visual.exists", VISUAL.exists())
    visual = load(VISUAL)
    check("visual.schema", visual["schema"] == "ai2050.fresh_callable_state_atlas.v1")
    check("visual.coordinates", visual["coordinate_count"] == 2560)
    check("visual.rows", len(visual["rows"]) == 36)
    check("visual.full_vectors", all(len(row["embedding_q0"]) == 2560 and len(row["state_q24"]) == 2560 and len(row["write_q24_q25"]) == 2560 for row in visual["rows"]))
    check("visual.finite", finite(visual))
    registry = load(REGISTRY)
    catalog = load(CATALOG)
    check("registry.entry", any(row.get("id") == "c525_fresh_callable_state_atlas" for row in registry.get("datasets", [])))
    check("catalog.entry", any(row.get("id") == "c525_fresh_callable_state_atlas" for row in catalog.get("field_datasets", [])))
    cleanup = load(campaign.OUTS["C525"] / "audit/raw_field_cleanup_ledger.json")
    check("cleanup.files", len(cleanup["files"]) == 2)
    check("cleanup.bytes", cleanup["total_bytes"] > 0)
    for row in cleanup["files"]:
        check(f"cleanup.absent.{Path(row['path']).name}", not (ROOT / row["path"]).exists())
        check(f"cleanup.hash.{Path(row['path']).name}", len(row["sha256"]) == 64)

    passed = sum(row["passed"] for row in checks)
    audit = {
        "status": "passed" if passed == len(checks) else "failed",
        "passed": passed, "total": len(checks), "checks": checks,
        "evidence_adjudication": {
            "retained": [
                "C519 behavior is descriptively strong but formally failed its frozen width gate by one token, and inference ran after that audit should have stopped execution.",
                "C521 does not prospectively replicate graph or temporal panel gains on the paraphrased fresh vocabulary.",
                "C521 nested passes only the frozen shared-model comparison and remains worse than identity, so it is not a substantive positive.",
                "C522 rejects the registered single-sample role-bundle qualification.",
                "C523 shows a cross-vocabulary autonomous layer-depth rollout candidate against q0 persistence only.",
            ],
            "forbidden_overclaims": [
                "Graph/temporal high-order operators replicated.",
                "Nested composition is predicted better than a strong baseline.",
                "The rollout is a semantic program rather than generic layer-depth/scaffold dynamics.",
                "A causal writable role state or new mathematics has been found.",
            ],
        },
    }
    save(OUT / "audit/independent_audit.json", audit)
    save(OUT / "protocol/preregistration.json", {
        "phase": 2060, "campaign": "C526", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "auditor_sha256": sha(Path(__file__)), "producer_sha256": main_hash,
        "scope": "independent missingness, metric, visual, cleanup, and claim-boundary audit for C518-C525",
    })
    final = {
        "phase": 2060, "campaign": "C526", "status": "closed", "all_checks_passed": audit["status"] == "passed",
        "headline": {
            "status": audit["status"], "checks_passed": passed, "checks_total": len(checks),
            "panel_result": "graph_and_temporal_not_replicated; nested_frozen_gate_is_weaker_than_identity",
            "single_sample_result": "failed",
            "rollout_result": "candidate_against_q0_persistence_only",
            "causal": "NA",
            "next_stage_same_goal": False,
            "next_stage": "a separately frozen stronger-control rollout campaign is needed before any mechanism continuation; current automatic same-goal chain ends because its single-sample qualification failed",
        },
        "next_authorization": "complete",
    }
    save(OUT / "analysis/final.json", final)
    print(json.dumps(final, ensure_ascii=False, indent=2))
    if audit["status"] != "passed":
        raise RuntimeError([row for row in checks if not row["passed"]])


if __name__ == "__main__":
    main()
