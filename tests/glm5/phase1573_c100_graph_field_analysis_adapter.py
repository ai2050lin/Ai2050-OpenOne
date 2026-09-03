#!/usr/bin/env python3
"""Phase1573 / C100: immutable C099 field analysis adapter and closure."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
C099 = RESULT / "phase1572_c099_fixed_width_graph_field_campaign"
OUT = RESULT / "phase1573_c100_graph_field_analysis_adapter"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core
import phase1571_c098_observation_first_graph_campaign as base

PHASE = 1573
CAMPAIGN = "C100"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def prepare() -> None:
    if (OUT / "protocol/preregistration.json").exists():
        raise RuntimeError("C100 already prepared")
    source_protocol = core.load(C099 / "protocol/preregistration.json")
    source_capture = core.load(C099 / "analysis/capture_summary.json")
    if source_capture["authorization"] != "run_phase1572_analysis" or not all(source_capture["checks"].values()):
        raise RuntimeError("C099 capture missing")
    adapter_failure = {
        "status": "analysis_not_started_authorization_literal_mismatch",
        "C099_authorization": source_capture["authorization"],
        "frozen_base_expected": "run_phase1571_analysis",
        "hidden_field_loaded_by_failed_call": False,
        "scientific_result": "none",
    }
    core.save(C099 / "analysis/analysis_adapter_failure.json", adapter_failure)
    core.save(C099 / "analysis/final.json", {
        "phase": 1572,
        "campaign": "C099",
        "status": "numeric_capture_passed_analysis_adapter_closed",
        "numeric_gate_passed": True,
        "hidden_structure_analyzed": False,
        "authorization": "run_C100_immutable_field_analysis_adapter",
    })
    for relative in (
        "material/frozen_graph_units.jsonl",
        "material/frozen_cases.jsonl",
        "material/frozen_test_examples.jsonl",
        "compiled/qwen3_active.jsonl",
        "raw/all_token_field_index.jsonl",
    ):
        destination = OUT / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(C099 / relative, destination)
    raw_source = C099 / "raw/all_token_all_state_field.float16.npy"
    raw_target = OUT / "raw/all_token_all_state_field.float16.npy"
    raw_target.parent.mkdir(parents=True, exist_ok=True)
    os.link(raw_source, raw_target)
    adapted_capture = json.loads(json.dumps(source_capture))
    adapted_capture.update({
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "authorization": "run_phase1571_analysis",
        "adapter_note": "literal compatibility only; all numeric values and raw bytes are inherited from C099",
    })
    core.save(OUT / "analysis/capture_summary.json", adapted_capture)
    protocol = json.loads(json.dumps(source_protocol))
    protocol.update({
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "schema": "c100.immutable_graph_field_analysis_adapter.v1",
        "single_changed_variable": "authorization literal adapter for frozen analysis engine",
        "source_raw_sha256": source_capture["raw_sha256"],
        "created_at_utc": now(),
        "authorization": "run_phase1573_offline_analysis",
    })
    protocol.pop("contract_sha256", None)
    protocol.pop("producer_sha256", None)
    protocol["contract_sha256"] = core.digest(protocol)
    protocol["producer_sha256"] = core.sha(Path(__file__))
    checks = {
        "C099_closed": core.load(C099 / "analysis/final.json")["status"] == "numeric_capture_passed_analysis_adapter_closed",
        "C099_numeric_pass": all(source_capture["checks"].values()),
        "failed_call_pre_hidden": not adapter_failure["hidden_field_loaded_by_failed_call"],
        "raw_hardlink_identity": os.path.samefile(raw_source, raw_target),
        "raw_hash_identity": core.sha(raw_target) == source_capture["raw_sha256"],
        "index_identity": core.sha(OUT / "raw/all_token_field_index.jsonl") == core.sha(C099 / "raw/all_token_field_index.jsonl"),
        "material_identity": all(
            core.sha(OUT / relative) == core.sha(C099 / relative)
            for relative in ("material/frozen_graph_units.jsonl", "material/frozen_cases.jsonl", "compiled/qwen3_active.jsonl")
        ),
        "threshold_identity": protocol["execution"] == source_protocol["execution"],
        "analysis_identity": protocol["analysis"] == source_protocol["analysis"],
        "adapter_only": protocol["single_changed_variable"] == "authorization literal adapter for frozen analysis engine",
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "audit/pre_analysis_adapter_audit.json", {
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "all_checks_passed": all(checks.values()),
        "source": adapter_failure,
    })
    print(json.dumps({"checks": checks, "contract_sha256": protocol["contract_sha256"], "authorization": protocol["authorization"]}, indent=2))


def analyze() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    audit = core.load(OUT / "audit/pre_analysis_adapter_audit.json")
    if protocol["authorization"] != "run_phase1573_offline_analysis" or not audit["all_checks_passed"]:
        raise RuntimeError("C100 analysis authorization missing")
    if protocol["producer_sha256"] != core.sha(Path(__file__)):
        raise RuntimeError("C100 producer changed after freeze")
    base.OUT = OUT
    base.PHASE = PHASE
    base.CAMPAIGN = CAMPAIGN
    base.analyze()


def export() -> None:
    base.OUT = OUT
    base.PHASE = PHASE
    base.CAMPAIGN = CAMPAIGN
    summary = core.load(OUT / "analysis/c098_graph_field_summary.json")
    if summary["authorization"] != "export_c098_graph_walsh_heatmap":
        core.save(OUT / "analysis/visualization_decision.json", {"important": False, "reason": "frozen importance threshold not reached"})
        return
    base.export_heatmap()
    old_canonical = OUT / "visualization/c098_graph_walsh_heatmap.json"
    old_client = ROOT / "frontend/public/vis_data/research_kernel/c098_graph_walsh_heatmap.json"
    asset = core.load(old_canonical)
    asset.update({"phase": PHASE, "campaign": CAMPAIGN, "title": "C100 Directed Graph Path Walsh Field"})
    canonical = OUT / "visualization/c100_graph_walsh_heatmap.json"
    client = ROOT / "frontend/public/vis_data/research_kernel/c100_graph_walsh_heatmap.json"
    core.save(canonical, asset)
    client.parent.mkdir(parents=True, exist_ok=True)
    client.write_bytes(canonical.read_bytes())
    old_canonical.unlink()
    if old_client.exists():
        old_client.unlink()
    decision = {
        "important": True,
        "asset": str(canonical.relative_to(ROOT)),
        "client": str(client.relative_to(ROOT)),
        "rows": len(asset["rows"]),
        "coordinates": len(asset["dimensions"]),
        "sha256": core.sha(canonical),
        "client_identity": core.sha(canonical) == core.sha(client),
    }
    core.save(OUT / "analysis/visualization_decision.json", decision)
    print(json.dumps(decision, indent=2))


def finalize() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    summary = core.load(OUT / "analysis/c098_graph_field_summary.json")
    visualization = core.load(OUT / "analysis/visualization_decision.json")
    checks = {
        "producer_frozen": protocol["producer_sha256"] == core.sha(Path(__file__)),
        "source_hardlink": os.path.samefile(C099 / "raw/all_token_all_state_field.float16.npy", OUT / "raw/all_token_all_state_field.float16.npy"),
        "source_hash": core.sha(OUT / "raw/all_token_all_state_field.float16.npy") == protocol["source_raw_sha256"],
        "walsh_hash": core.sha(OUT / "raw/focus_role_walsh_coefficients.float32.npy") == summary["walsh"]["sha256"],
        "support_count": len(core.rows(OUT / "analysis/discovery_top64_supports.jsonl")) == 3 * 4 * 37 * 4,
        "holdout_count": len(core.rows(OUT / "analysis/dual_holdout_xy_validation.jsonl")) == 3 * 4 * 37 * 4 * 2,
        "design_null": len(core.rows(OUT / "analysis/c097_shared_cell_design_null.jsonl")) == 12,
        "all_token_scan": summary["all_token_scan_rows"] > 0,
        "visualization": (not visualization["important"]) or visualization["client_identity"],
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    final = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "status": "graph_field_observation_major_stage_complete",
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "all_checks_passed": all(checks.values()),
        "result": summary,
        "visualization": visualization,
        "theory": {
            "name": "conditional output field closure theory",
            "principle": "reuse-difference-conditioning (RDC)",
            "formula": "H_{l+1,t}=T_l(H_{l,<=t};phi,eta); C_S=2^-4 sum_z chi_S(z)H(z)",
            "global_graph": "embedding identity -> directed local graph -> repeated-target/query code-invariant path response -> code-conditioned boundary response -> output competition",
            "math_status": "existing finite-difference and conditional-dynamics mathematics suffices for this observation; no new theory is established",
        },
        "next_authorization": "C101 observation-first breadth over non-transitive relation composition, using C100 as discovery geometry and fresh holdouts.",
        "finished_at_utc": now(),
    }
    core.save(OUT / "analysis/final.json", final)
    print(json.dumps(final, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "analyze", "export", "finalize", "all"))
    args = parser.parse_args()
    if args.stage in ("prepare", "all"):
        prepare()
    if args.stage in ("analyze", "all"):
        analyze()
    if args.stage in ("export", "all"):
        export()
    if args.stage in ("finalize", "all"):
        finalize()


if __name__ == "__main__":
    main()
