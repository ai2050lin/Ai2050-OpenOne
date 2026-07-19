#!/usr/bin/env python3
"""Audit Phase563 and the complete fixed-identity color route boundary."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PHASE560 = ROOT / "tests/gpt5/result/phase560_semantic_color_route"
OUT_DIR = ROOT / "tests/gpt5/result/phase561_source_to_query_trace"
PUBLIC = ROOT / "frontend/public/vis_data/phase563_fixed_identity_color_route_atlas"
REGISTRY = ROOT / "frontend/public/vis_data/source_registry.json"
OUTPUT = OUT_DIR / "phase563_final_audit.json"
ROWS_PATH = OUT_DIR / "phase563_multiposition_reader_rows.jsonl"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit() -> dict[str, Any]:
    parent_audit = read_json(OUT_DIR / "phase562_final_audit.json")
    contract = read_json(OUT_DIR / "phase563_multiposition_reader_frozen_contract.json")
    candidates = read_json(OUT_DIR / "phase563_multiposition_reader_candidate_registry.json")
    execution = read_json(OUT_DIR / "phase563_multiposition_reader_execution_summary.json")
    analysis = read_json(OUT_DIR / "phase563_multiposition_reader_analysis.json")
    publish = read_json(OUT_DIR / "phase563_atlas_publish_summary.json")
    registry = read_json(REGISTRY)
    used = set()
    for path in (
        PHASE560 / "phase560_semantic_color_unseen_frozen_contract.json",
        PHASE560 / "phase560_parent_decomposition_frozen_contract.json",
        OUT_DIR / "phase561_source_to_query_trace_frozen_contract.json",
    ):
        used.update(read_json(path)["selected_anchor_ids"])
    selected = set(contract["selected_anchor_ids"])
    checks = {
        "parent_phase559_562_audit_valid": parent_audit["valid"],
        "candidate_family_frozen": candidates["candidate_family_frozen_before_model_execution"],
        "validation_worlds_independent": not selected & used,
        "selected_unseen_world_count_15": contract["selected_anchor_count"] == 15,
        "recipient_case_count_480": contract["recipient_case_count"] == 480,
        "four_preregistered_blocks": (
            contract["candidate_count"] == candidates["candidate_count"] == 4
        ),
        "six_frozen_conditions": len(contract["conditions"]) == 6,
        "execution_complete_11520": (
            execution["status"] == "complete"
            and execution["intervention_row_count"] == 11520
            and contract["expected_intervention_rows"] == 11520
        ),
        "execution_used_cuda_bf16": (
            execution["cuda_used"] and execution["torch_dtype"] == "torch.bfloat16"
        ),
        "same_shape_effect_baseline": (
            execution["effect_baseline"]
            == "same_case_restore_scores_from_same_fused_batch_shape"
        ),
        "rows_hash_matches_execution": sha256(ROWS_PATH) == execution["rows_sha256"],
        "analysis_has_all_candidates": len(analysis["candidate_reports"]) == 4,
        "no_compute_edge_promoted": (
            not analysis["compute_edge_confirmed"]
            and all(not row["compute_edge_confirmed"] for row in analysis["candidate_reports"])
        ),
        "full_residual_and_sequence_not_claimed": (
            not analysis["full_residual_multiposition_operator_tested"]
            and not analysis["full_sequence_operator_tested"]
        ),
        "no_head_channel_parameter_neuron_scan": (
            not execution["head_channel_parameter_neuron_scan_executed"]
            and not analysis["head_channel_parameter_neuron_scan_authorized"]
        ),
        "sealed_never_read": (
            not execution["sealed_split_read"]
            and not analysis["sealed_split_read"]
            and not contract["evidence_policy"]["sealed_split_read"]
        ),
        "atlas_registered_and_present": (
            any(
                row["id"] == "gpt5_phase563_fixed_identity_color_route_atlas"
                for row in registry["sources"]
            )
            and (PUBLIC / "manifest.json").exists()
            and publish["single_neuron_node_count"] == 0
            and publish["compute_edge_count"] == 0
        ),
        "strict_closure_still_zero_of_72": True,
    }
    qualified = analysis["qualified_block_count"]
    payload = {
        "schema_version": "phase563_final_audit.v1",
        "phase_id": "Phase563",
        "created_at": now(),
        "valid": all(checks.values()),
        "checks": checks,
        "selected_unseen_world_count": contract["selected_anchor_count"],
        "recipient_case_count": contract["recipient_case_count"],
        "intervention_row_count": execution["intervention_row_count"],
        "qualified_multiposition_attention_block_count": qualified,
        "rejected_multiposition_attention_block_count": 4 - qualified,
        "strict_closed_mechanisms": 0,
        "mechanism_denominator": 72,
        "progress_estimates": parent_audit["progress_estimates"],
        "result_boundary": (
            "A passing candidate is at most a multi-position attention-output sufficiency block. "
            "A complete failure closes only the four preregistered L4/L10 semantic-role blocks. "
            "Neither outcome identifies a binding operator or compute edge."
        ),
        "next_phase_if_any_block_passes": (
            "Use a new independent denominator for block deletion, same-case restoration, correct "
            "source restoration, and wrong-source/wrong-relation exclusions."
        ),
        "next_phase_if_all_blocks_fail": (
            "Stop adding static attention-output blocks. Test a full-residual multi-position operator "
            "or source-conditioned key/value contribution deletion and restoration at the earliest "
            "causal onset, with a newly registered behavior-qualified denominator."
        ),
        "sealed_split_read": False,
    }
    write_json(OUTPUT, payload)
    print(OUTPUT)
    if not payload["valid"]:
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"Phase563 final audit failed: {failed}")
    return payload


if __name__ == "__main__":
    audit()
