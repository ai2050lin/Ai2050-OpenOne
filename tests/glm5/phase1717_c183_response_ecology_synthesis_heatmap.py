#!/usr/bin/env python3
"""C183: synthesize C173-C182 and export parameter-level Qwen state/response heatmaps."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
OUT = RESULT / "phase1717_c183_response_ecology_synthesis_heatmap"
PUBLIC = ROOT / "frontend/public/vis_data/research_kernel/c183_natural_response_ecology_heatmap.json"
C173 = RESULT / "phase1707_c173_role_specific_full_coordinate_response"
C174 = RESULT / "phase1708_c174_signed_target_edge_compression"
C175 = RESULT / "phase1709_c175_role_pair_hyperedge_response"
C177 = RESULT / "phase1711_c177_missing_aware_broad_family_atlas"
C180 = RESULT / "phase1714_c180_reachable_target_choice_ecology"
C181 = RESULT / "phase1715_c181_cross_model_functional_eligibility"
C182 = RESULT / "phase1716_c182_cross_model_hidden_topology_adjudication"
C167 = RESULT / "phase1701_c167_transport_component_decomposition"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core

PHASE, CAMPAIGN = 1717, "C183"
ROLES = ("primary", "secondary", "relation", "context", "query", "boundary")
CHECKPOINTS = (0, 8, 16, 24, 25, 32, 37)
CHECKPOINT_LABELS = {
    0: "embedding",
    8: "block_07_output",
    16: "block_15_output",
    24: "block_23_output_q24",
    25: "block_24_output_q25",
    32: "block_31_output",
    37: "final_norm",
}


def bf16_to_float(values: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(values, dtype=np.uint16)
    return torch.from_numpy(contiguous).view(torch.bfloat16).float().numpy()


def contract() -> None:
    if OUT.exists():
        raise RuntimeError(OUT)
    parents = {
        "c173": C173 / "audit/independent_final_audit.json",
        "c174": C174 / "audit/independent_final_audit.json",
        "c175": C175 / "audit/independent_final_audit.json",
        "c177": C177 / "audit/independent_final_audit.json",
        "c180": C180 / "audit/independent_final_audit.json",
        "c181": C181 / "audit/independent_final_audit.json",
        "c182": C182 / "audit/independent_final_audit.json",
    }
    checks = {name: core.load(path)["all_checks_passed"] for name, path in parents.items()}
    checks["authorization"] = "C183" in core.load(parents["c182"])["authorization"]
    checks["raw_shapes"] = (
        list(np.load(C180 / "raw/eligible_six_role_all_checkpoint.bf16.npy", mmap_mode="r").shape)[1:] == [6, 38, 2560]
        and list(np.load(C180 / "raw/anchor_role_response.float16.npy", mmap_mode="r").shape)[2:] == [64, 6, 2560]
    )
    if not all(checks.values()):
        raise RuntimeError(checks)
    OUT.mkdir(parents=True)
    protocol = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "existing_data_synthesis_contract_frozen",
        "model": "Qwen3-4B BF16 CUDA observations from C180; no new model run",
        "state_rows": "discovery anchor, seven eligible relation families, six semantic roles, seven frozen checkpoints, all 2560 activation coordinates",
        "response_rows": "discovery and fresh anchors, relation source, first four frozen source coordinates, six target roles, all 2560 q25 response coordinates",
        "checkpoints": list(CHECKPOINTS),
        "roles": list(ROLES),
        "coordinate_semantics": "physical activation axes; state0 is embedding; later states are HiddenState checkpoints",
        "forbidden": ["attention", "MLP", "weights", "PCA", "top-coordinate-only state display", "claiming a unique causal circuit"],
        "producer_sha256": core.sha(Path(__file__)),
        "authorization": "build_synthesis_and_public_heatmap",
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    print(json.dumps({"checks": checks}, indent=2))


def build() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    role_raw = np.load(C180 / "raw/eligible_six_role_all_checkpoint.bf16.npy", mmap_mode="r")
    response = np.load(C180 / "raw/anchor_role_response.float16.npy", mmap_mode="r")
    eligible = core.rows(C180 / "raw/eligible_row_index.jsonl")
    anchors = core.rows(C180 / "raw/anchor_index.jsonl")
    global_to_local = {row["row_index"]: row["local_index"] for row in eligible}
    anchor_lookup = {(row["partition"], row["family"]): row for row in anchors}
    families = core.load(C180 / "protocol/behavior_eligibility_lock.json")["eligible_families"]
    source_coordinates = core.load(C167 / "analysis/top_relation_source_coordinates.json")["coordinates"][:4]

    state_rows = []
    for family in families:
        anchor = anchor_lookup[("discovery", family)]
        local = global_to_local[anchor["row_index"]]
        for role_i, role in enumerate(ROLES):
            for checkpoint in CHECKPOINTS:
                values = bf16_to_float(role_raw[local, role_i, checkpoint])
                state_rows.append({
                    "kind": "anchor_state",
                    "dataset": "C180",
                    "family": family,
                    "partition": "discovery",
                    "role": role,
                    "checkpoint": checkpoint,
                    "checkpoint_label": CHECKPOINT_LABELS[checkpoint],
                    "label": f"{family} / {role} / {CHECKPOINT_LABELS[checkpoint]}",
                    "values": values.tolist(),
                })

    response_rows = []
    response_for_scale = []
    for partition in ("discovery", "fresh"):
        for family in families:
            anchor = anchor_lookup[(partition, family)]
            anchor_i = anchor["anchor_index"]
            for source_i, source_coordinate in enumerate(source_coordinates):
                for target_i, target_role in enumerate(ROLES):
                    values = np.asarray(response[2, anchor_i, source_i, target_i], dtype=np.float32)
                    response_for_scale.append(np.abs(values))
                    response_rows.append({
                        "kind": "local_response",
                        "dataset": "C180",
                        "family": family,
                        "partition": partition,
                        "source_role": "relation",
                        "source_coordinate": int(source_coordinate),
                        "source_checkpoint": 24,
                        "target_role": target_role,
                        "target_checkpoint": 25,
                        "label": f"{partition} {family} / relation src{source_coordinate} -> {target_role}",
                        "values": values.tolist(),
                    })
    mean_abs_response = np.mean(np.stack(response_for_scale), axis=0)
    default_coordinates = np.argsort(-mean_abs_response)[:64].astype(int).tolist()

    source_finals = {
        "c173": core.load(C173 / "analysis/final.json"),
        "c174": core.load(C174 / "analysis/final.json"),
        "c175": core.load(C175 / "analysis/final.json"),
        "c177": core.load(C177 / "analysis/final.json"),
        "c180": core.load(C180 / "analysis/final.json"),
        "c181": core.load(C181 / "analysis/final.json"),
        "c182": core.load(C182 / "analysis/final.json"),
    }
    atlas = core.load(C180 / "analysis/natural_ecology_atlas.json")
    synthesis = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "status": "response_ecology_synthesized",
        "families": families,
        "state_row_count": len(state_rows),
        "response_row_count": len(response_rows),
        "state_coordinates_per_row": 2560,
        "response_coordinates_per_row": 2560,
        "response_summary": atlas["response_summary"],
        "evidence_ledger": {
            "replicated": ["C173 query local response", "C180 query local response", "C180 relation local response"],
            "bounded_or_absent": ["C173 primary local response", "C174 fixed target-edge support", "C175 fixed coordinate-pair identity", "C181 cross-model common functional interface"],
            "measurement_correction": ["C176 zero-vector metrics invalid", "C177 missing-aware support accounting"],
        },
        "mechanism_candidate": "role-conditioned source-addressed response ecology with reconfigurable distributed target support",
        "not_established": ["fixed semantic vector", "compact fixed edge graph", "stable pairwise hyperedges", "minimal necessary circuit", "cross-model topology", "new mathematical theory"],
        "next_authorization": "run_C184_response_ecology_invariant_discovery_on_existing_full_coordinate_rows",
    }
    core.save(OUT / "analysis/synthesis.json", synthesis)

    payload = {
        "schema": "c183_natural_response_ecology_heatmap.v1",
        "result_type": "natural_response_ecology_heatmap",
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "model": "Qwen3-4B",
        "title": "C173-C183 Natural Relation State and Local Response Ecology",
        "dimensions": list(range(2560)),
        "default_coordinates": default_coordinates,
        "source_coordinates": source_coordinates,
        "checkpoints": [{"id": q, "label": CHECKPOINT_LABELS[q]} for q in CHECKPOINTS],
        "roles": list(ROLES),
        "families": families,
        "rows": state_rows + response_rows,
        "synthesis": synthesis,
        "source_finals": source_finals,
        "coordinate_semantics": "Every column is one Qwen3-4B physical activation coordinate. State rows include embedding and HiddenState checkpoints; response rows are signed q24-to-q25 finite-difference derivatives.",
        "claim_boundary": "The query/relation local response replicates across seven natural relation phrases and fresh vocabulary. Fixed sparse target edges and fixed coordinate-pair hyperedges do not. Coordinates are activations, not weights or named semantic neurons; this is not a unique causal circuit or complete language mechanism.",
    }
    PUBLIC.parent.mkdir(parents=True, exist_ok=True)
    PUBLIC.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    asset = {
        "path": str(PUBLIC.relative_to(ROOT)).replace("\\", "/"),
        "sha256": core.sha(PUBLIC),
        "bytes": PUBLIC.stat().st_size,
        "rows": len(payload["rows"]),
        "schema": payload["schema"],
    }
    core.save(OUT / "analysis/public_asset.json", asset)
    checks = {
        "state_rows": len(state_rows) == len(families) * len(ROLES) * len(CHECKPOINTS),
        "response_rows": len(response_rows) == 2 * len(families) * 4 * len(ROLES),
        "all_coordinates": all(len(row["values"]) == 2560 for row in payload["rows"]),
        "embedding_present": any(row.get("checkpoint") == 0 for row in state_rows),
        "hidden_present": any(row.get("checkpoint") == 37 for row in state_rows),
        "finite": bool(np.isfinite(mean_abs_response).all()),
    }
    core.save(OUT / "audit/internal_build_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    print(json.dumps({"synthesis": synthesis, "asset": asset, "checks": checks}, indent=2))


def close() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    build_audit = core.load(OUT / "audit/internal_build_audit.json")
    asset = core.load(OUT / "analysis/public_asset.json")
    checks = {
        "contract": core.load(OUT / "audit/internal_contract_audit.json")["all_checks_passed"],
        "build": build_audit["all_checks_passed"],
        "hash": core.sha(Path(__file__)) == protocol["producer_sha256"],
        "asset_hash": core.sha(PUBLIC) == asset["sha256"],
    }
    final = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "status": "closed",
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "headline": core.load(OUT / "analysis/synthesis.json"),
        "next_authorization": "run_C184_response_ecology_invariant_discovery_on_existing_full_coordinate_rows",
    }
    core.save(OUT / "analysis/final.json", final)
    print(json.dumps(final, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("contract", "build", "close"))
    args = parser.parse_args()
    {"contract": contract, "build": build, "close": close}[args.command]()


if __name__ == "__main__":
    main()
