#!/usr/bin/env python3
"""Independent audit for Phase1988-2004 / C454-C470."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/glm5/result"
PRODUCER = ROOT / "tests/glm5/phase1988_c454_c470_semantic_residual_campaign.py"
VISUAL = ROOT / "frontend/public/vis_data/research_kernel/c470_semantic_residual_graph.json"

PHASES = {
    f"C{campaign}": (1988 + campaign - 454, slug)
    for campaign, slug in (
        (454, "semantic_residual_evidence_adjudication_and_contract"),
        (455, "sixteen_family_surface_semantic_material"),
        (456, "material_zero_model_and_role_audit"),
        (457, "qwen_sixteen_family_behavior_qualification"),
        (458, "qualified_full_coordinate_semantic_surface_field"),
        (459, "typed_response_ledger"),
        (460, "identity_and_mean_propagation_baselines"),
        (461, "shared_checkpoint_role_propagation"),
        (462, "construction_conditioned_propagation"),
        (463, "operation_and_family_semantic_increment"),
        (464, "semantic_residual_lockbox_adjudication"),
        (465, "full_coordinate_neighbor_coupling_tournament"),
        (466, "autonomous_multistep_response_rollout"),
        (467, "fresh_graph_path_material_and_behavior"),
        (468, "graph_path_field_and_nonlinear_integration"),
        (469, "conditional_natural_semantic_writer"),
        (470, "campaign_synthesis_visual_cleanup_and_audit"),
    )
}


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def out(name: str) -> Path:
    phase, slug = PHASES[name]
    return RESULT / f"phase{phase}_{name.lower()}_{slug}"


def finite(value) -> bool:
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, list):
        return all(finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def main() -> None:
    digest = hashlib.sha256(PRODUCER.read_bytes()).hexdigest()
    checks = {}
    for name, (phase, _) in PHASES.items():
        preregistration = load(out(name) / "protocol/preregistration.json")
        final = load(out(name) / "analysis/final.json")
        checks[f"{name}_closed"] = final["all_checks_passed"]
        checks[f"{name}_phase"] = preregistration["phase"] == phase == final["phase"]
        checks[f"{name}_producer_hash"] = preregistration["producer_sha256"] == digest

    semantic = rows(out("C455") / "material/cases.jsonl")
    compiled = rows(out("C457") / "compiled/qwen3.jsonl")
    checks["semantic_rows"] = len(semantic) == len(compiled) == 3840
    checks["semantic_families"] = len({row["family"] for row in semantic}) == 16
    checks["semantic_global_balance"] = sum(row["gold_position"] == 0 for row in semantic) == 1920
    checks["semantic_stratum_balance"] = all(
        sum(row["gold_position"] == 0 for row in semantic if row[key] == value)
        * 2
        == sum(row[key] == value for row in semantic)
        for key in ("family", "construction", "style")
        for value in {row[key] for row in semantic}
    )
    checks["compiled_width"] = max(len(row["prompt_ids"]) for row in compiled) <= 192
    checks["compiled_roles"] = all(
        positions and all(0 <= int(position) < len(row["prompt_ids"]) for position in positions)
        for row in compiled
        for positions in row["role_positions"].values()
    )

    graph = rows(out("C467") / "material/cases.jsonl")
    graph_compiled = rows(out("C467") / "compiled/qwen3.jsonl")
    checks["graph_rows"] = len(graph) == len(graph_compiled) == 720
    checks["graph_balance"] = sum(row["gold_position"] == 0 for row in graph) == 360
    checks["graph_width"] = max(len(row["prompt_ids"]) for row in graph_compiled) <= 192

    ledger = rows(out("C459") / "analysis/response_records.jsonl")
    counts = {operation: sum(row["operation"] == operation for row in ledger) for operation in ("surface", "statement", "query")}
    checks["typed_ledger"] = len(ledger) == 3600 and counts == {"surface": 1800, "statement": 900, "query": 900}

    c460 = load(out("C460") / "analysis/final.json")["headline"]
    c461 = load(out("C461") / "analysis/final.json")["headline"]
    c464 = load(out("C464") / "analysis/final.json")["headline"]
    c465 = load(out("C465") / "analysis/final.json")["headline"]
    c466 = load(out("C466") / "analysis/final.json")["headline"]
    c468 = load(out("C468") / "analysis/final.json")["headline"]
    c469 = load(out("C469") / "analysis/final.json")["headline"]
    c470 = load(out("C470") / "analysis/final.json")["headline"]
    checks["shared_gate_recomputed"] = c461["shared_candidate"] == (
        c461["nrmse"] < min(c460["metrics"]["identity"]["nrmse"], c460["metrics"]["mean"]["nrmse"])
    )
    semantic_gate = (
        c464["semantic_gain"] > 0.02
        and c464["semantic_gain"] > c464["surface_gain"] + 0.01
        and c464["family_wins"] >= 10
        and c464["unseen_report_gain"] > 0.01
    )
    checks["semantic_gate_recomputed"] = c464["semantic_residual_candidate"] == semantic_gate is False
    checks["coupling_gate_recomputed"] = c465["neighbor_coupling_candidate"] == (c465["semantic_gain"] > 0.01)
    rollout_gate = all(value["m3"]["nrmse"] < value["identity"]["nrmse"] for value in c466["metrics"].values())
    checks["rollout_gate_recomputed"] = c466["multistep_candidate"] == rollout_gate
    graph_gate = c468["metrics"]["affine"]["nrmse"] < min(
        c468["metrics"][name]["nrmse"] for name in ("identity", "zero", "mean")
    )
    checks["graph_gate_recomputed"] = c468["depth_transition_candidate"] == graph_gate is False
    checks["writer_abstention"] = not c469["writer_ran"] and not c469["specificity_passed"]
    checks["new_math_abstention"] = not c470["new_math_gate_passed"] and not c470["gates"]["cross_model_composition"]

    visual = load(VISUAL)
    checks["visual_row_count"] = len(visual["rows"]) == c470["visual_rows"] == 1578
    checks["visual_full_coordinates"] = all(len(row["values"]) == 2560 for row in visual["rows"])
    checks["visual_finite"] = finite(visual)
    cleanup = load(out("C470") / "audit/cleanup.json")
    checks["cleanup_count"] = len(cleanup) == c470["cleanup_files"] == 13
    checks["cleanup_hashes"] = all(row["deleted"] and len(row["sha256"]) == 64 for row in cleanup)
    checks["cleanup_paths_absent"] = all(not (ROOT / row["path"]).exists() for row in cleanup)
    checks["cleanup_bytes"] = sum(row["bytes"] for row in cleanup) == c470["cleanup_bytes"]

    report = {
        "phase": 2004,
        "campaign": "C454-C470",
        "producer_sha256": digest,
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "all_checks_passed": all(checks.values()),
    }
    destination = out("C470") / "audit/independent_audit.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    assert report["all_checks_passed"]


if __name__ == "__main__":
    main()
