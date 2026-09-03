#!/usr/bin/env python3
"""C169: build the C167-C168 full-coordinate relation residual heatmap."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
C167 = RESULT / "phase1701_c167_transport_component_decomposition"
C168 = RESULT / "phase1702_c168_fresh_relation_residual_confirmation"
OUT = RESULT / "phase1703_c169_relation_residual_coordinate_heatmap"
FRONTEND = ROOT / "frontend/public/vis_data/research_kernel/c167_c168_relation_residual_heatmap.json"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core

PHASE, CAMPAIGN, DIM = 1703, "C169", 2560
RELATIONS = ("is_a", "part_of", "located_in", "precedes")
ROLES = ("primary", "secondary", "relation", "context", "query", "boundary")
DISPLAY_ROLES = ("relation", "query", "boundary")


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cos_rows(a, b):
    af = a.reshape(len(a), -1).astype(np.float64, copy=False)
    bf = b.reshape(len(b), -1).astype(np.float64, copy=False)
    return np.sum(af * bf, axis=1) / np.maximum(np.linalg.norm(af, axis=1) * np.linalg.norm(bf, axis=1), 1e-12)


def contract():
    if OUT.exists():
        raise RuntimeError(OUT)
    audits = [core.load(path / "audit/independent_final_audit.json") for path in (C167, C168)]
    checks = {
        "parent_audits": all(a["all_checks_passed"] for a in audits),
        "scientific_parent": audits[1]["scientific_passed"],
        "frontend_parent": FRONTEND.parent.exists(),
        "old_tensor": list(np.load(C167 / "analysis/top_relation_component_fields.float16.npy", mmap_mode="r").shape) == [2, 4, 16, 6, DIM],
        "fresh_tensor": list(np.load(C168 / "analysis/fresh_relation_components.float16.npy", mmap_mode="r").shape) == [4, 64, 6, DIM],
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    OUT.mkdir(parents=True)
    protocol = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "created_at_utc": now(),
        "status": "relation_residual_heatmap_contract_frozen",
        "rows": "old reference and fresh relation components for 4 relations x 8 source coordinates x 3 roles, plus source-coordinate score rows",
        "dimensions": "all 2560 Qwen3-4B target activation coordinates",
        "coordinate_semantics": "source_coordinate is a q24 relation-role physical activation coordinate; each heatmap column is a q25 target-role physical activation coordinate",
        "asset": str(FRONTEND.relative_to(ROOT)),
        "forbidden": ["attention", "MLP", "weights", "PCA", "cross-model coordinate equivalence"],
        "source_hashes": {
            "C167": core.sha(C167 / "analysis/top_relation_component_fields.float16.npy"),
            "C168": core.sha(C168 / "analysis/fresh_relation_components.float16.npy"),
        },
        "producer_sha256": core.sha(Path(__file__)),
        "authorization": "build_asset",
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": True})
    print(json.dumps({"checks": checks, "protocol": protocol}, indent=2))


def build():
    old = np.asarray(np.load(C167 / "analysis/top_relation_component_fields.float16.npy", mmap_mode="r"), np.float32)
    fresh = np.asarray(np.load(C168 / "analysis/fresh_relation_components.float16.npy", mmap_mode="r"), np.float32)
    coordinates = core.load(C167 / "analysis/top_relation_source_coordinates.json")["coordinates"]
    source_ids = coordinates[:8]
    reference = old.mean(axis=0)[:, :8]
    rows = []
    matrix = []
    for ri, relation in enumerate(RELATIONS):
        for si, source in enumerate(source_ids):
            for role in DISPLAY_ROLES:
                role_i = ROLES.index(role)
                for split, values in (("old_reference", reference[ri, si, role_i]), ("fresh", fresh[ri, si, role_i])):
                    vector = np.asarray(values, np.float32)
                    rows.append({"dataset": "C167-C168", "kind": "relation_component", "label": f"{relation} src{source} {role} {split}", "relation": relation, "source_coordinate": int(source), "target_role": role, "split": split, "values": vector.tolist()})
                    matrix.append(vector)
    match = np.mean(np.stack([cos_rows(reference[ri], fresh[ri, :8]) for ri in range(4)]), axis=0)
    rank_vector = np.zeros(DIM, np.float32)
    match_vector = np.zeros(DIM, np.float32)
    measured = coordinates[:64]
    for rank, coordinate in enumerate(measured):
        rank_vector[int(coordinate)] = float(64 - rank) / 64.0
    reference64_old = old.mean(axis=0)
    match64 = np.mean(np.stack([cos_rows(reference64_old[ri], fresh[ri, :16]) for ri in range(4)]), axis=0)
    for local, coordinate in enumerate(coordinates[:16]):
        match_vector[int(coordinate)] = float(match64[local])
    rows.extend([
        {"dataset": "C167-C168", "kind": "source_coordinate_rank", "label": "C167 discovery relation-energy source rank", "source_coordinate": None, "target_role": "source_coordinate_axis", "split": "discovery_lock", "measured_coordinates": measured, "values": rank_vector.tolist()},
        {"dataset": "C167-C168", "kind": "source_coordinate_fresh_match", "label": "C168 fresh mean relation match (top16)", "source_coordinate": None, "target_role": "source_coordinate_axis", "split": "fresh", "measured_coordinates": coordinates[:16], "values": match_vector.tolist()},
    ])
    matrix = np.asarray(matrix, np.float32)
    default = np.argsort(-np.mean(np.abs(matrix), axis=0))[:64].astype(int).tolist()
    asset = {
        "schema": "c167_c168_relation_residual_heatmap.v1",
        "result_type": "relation_residual_coordinate_heatmap",
        "phase": PHASE,
        "campaign": "C167-C169",
        "model": "Qwen3-4B",
        "title": "Relation-Conditioned Local Transmission Coordinates",
        "dimensions": list(range(DIM)),
        "default_coordinates": default,
        "source_coordinates": coordinates[:64],
        "rows": rows,
        "c167": core.load(C167 / "analysis/final.json"),
        "c168": core.load(C168 / "analysis/final.json"),
        "coordinate_semantics": "For relation-component rows, the row source id is a q24 relation-role activation coordinate and each column is a q25 target-role activation coordinate. Source-score rows use the same 0..2559 axis and leave unmeasured coordinates at zero by contract.",
        "claim_boundary": "Fresh relation residuals pass prediction and source-coordinate controls at q24->q25. The atlas does not identify a minimal circuit, Attention/MLP mechanism, model weight, or whole-language causal closure.",
    }
    (OUT / "analysis").mkdir(parents=True, exist_ok=True)
    text = json.dumps(asset, separators=(",", ":"), ensure_ascii=True)
    (OUT / "analysis/heatmap.json").write_text(text, encoding="utf-8")
    FRONTEND.write_text(text, encoding="utf-8")
    report = {"phase": PHASE, "campaign": CAMPAIGN, "status": "asset_built", "row_count": len(rows), "field_rows": int(matrix.shape[0]), "dimensions": DIM, "source_coordinates": source_ids, "default_coordinates": default, "asset_bytes": FRONTEND.stat().st_size, "asset_sha256": sha(FRONTEND), "top8_mean_fresh_match": match.tolist(), "next_authorization": "frontend integration and independent audit"}
    core.save(OUT / "analysis/synthesis.json", report)
    checks = {"rows": len(rows) == 194, "field_rows": matrix.shape == (192, DIM), "finite": bool(np.isfinite(matrix).all()), "asset_equal": sha(FRONTEND) == sha(OUT / "analysis/heatmap.json"), "source_ids": len(set(source_ids)) == 8}
    core.save(OUT / "audit/internal_build_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    print(json.dumps(report, indent=2))


def close():
    report = core.load(OUT / "analysis/synthesis.json")
    checks = {"contract": core.load(OUT / "audit/internal_contract_audit.json")["all_checks_passed"], "build": core.load(OUT / "audit/internal_build_audit.json")["all_checks_passed"]}
    final = {"phase": PHASE, "campaign": CAMPAIGN, "status": "closed", "checks": checks, "all_checks_passed": all(checks.values()), "headline": report, "next_authorization": "broader relation-role-checkpoint campaign"}
    core.save(OUT / "analysis/final.json", final)
    print(json.dumps(final, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("contract", "build", "close"))
    args = parser.parse_args()
    {"contract": contract, "build": build, "close": close}[args.command]()


if __name__ == "__main__":
    main()
