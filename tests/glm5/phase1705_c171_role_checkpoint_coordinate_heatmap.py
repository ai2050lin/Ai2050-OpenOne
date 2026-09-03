#!/usr/bin/env python3
"""C171: parameter-level heatmap for the C170 role/checkpoint atlas."""
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
C170 = RESULT / "phase1704_c170_role_checkpoint_relation_transport_atlas"
OUT = RESULT / "phase1705_c171_role_checkpoint_coordinate_heatmap"
FRONTEND = ROOT / "frontend/public/vis_data/research_kernel/c170_role_checkpoint_coordinate_heatmap.json"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core

PHASE, CAMPAIGN, DIM = 1705, "C171", 2560
RELATIONS = ("is_a", "part_of", "located_in", "precedes")
SOURCE_ROLES = ("primary", "relation", "query")
SOURCE_QS = (23, 24, 25)
TARGET_ROLES = ("primary", "secondary", "relation", "context", "query", "boundary")
DISPLAY_TARGET_ROLES = ("relation", "query", "boundary")


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def contract():
    if OUT.exists():
        raise RuntimeError(OUT)
    audit = core.load(C170 / "audit/independent_final_audit.json")
    field = np.load(C170 / "analysis/fresh_relation_components.float16.npy", mmap_mode="r")
    checks = {"parent_audit": audit["all_checks_passed"], "field": list(field.shape) == [9, 4, 16, 6, DIM], "frontend": FRONTEND.parent.exists()}
    if not all(checks.values()):
        raise RuntimeError(checks)
    OUT.mkdir(parents=True)
    protocol = {"phase": PHASE, "campaign": CAMPAIGN, "created_at_utc": now(), "status": "role_checkpoint_heatmap_frozen", "rows": "9 source settings x 4 relations x 4 source coordinates x 3 target roles", "dimensions": DIM, "asset": str(FRONTEND.relative_to(ROOT)), "coordinate_semantics": "row source id is a physical activation coordinate at the named source checkpoint/role; columns are target checkpoint/role activation coordinates", "forbidden": ["attention", "MLP", "weights", "PCA"], "source_hash": core.sha(C170 / "analysis/fresh_relation_components.float16.npy"), "producer_sha256": core.sha(Path(__file__)), "authorization": "build_asset"}
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": True})
    print(json.dumps({"checks": checks, "protocol": protocol}, indent=2))


def build():
    field = np.asarray(np.load(C170 / "analysis/fresh_relation_components.float16.npy", mmap_mode="r"), np.float32)
    coordinates = core.load(C170 / "protocol/preregistration.json")["source_coordinates"]
    atlas = core.load(C170 / "analysis/atlas.json")
    setting_lookup = {(r["source_checkpoint"], r["source_role"]): r for r in atlas["settings"]}
    rows, matrix = [], []
    setting_i = 0
    for source_q in SOURCE_QS:
        for source_role in SOURCE_ROLES:
            label = setting_lookup[(source_q, source_role)]["label"]
            for relation_i, relation in enumerate(RELATIONS):
                for source_i, source_coordinate in enumerate(coordinates[:4]):
                    for target_role in DISPLAY_TARGET_ROLES:
                        target_role_i = TARGET_ROLES.index(target_role)
                        vector = field[setting_i, relation_i, source_i, target_role_i]
                        rows.append({"dataset": "C170", "kind": "fresh_relation_component", "label": f"q{source_q} {source_role} {label} {relation} src{source_coordinate} -> {target_role}", "source_checkpoint": source_q, "target_checkpoint": source_q + 1, "source_role": source_role, "setting_label": label, "relation": relation, "source_coordinate": int(source_coordinate), "target_role": target_role, "values": vector.tolist()})
                        matrix.append(vector)
            setting_i += 1
    matrix = np.asarray(matrix, np.float32)
    default = np.argsort(-np.mean(np.abs(matrix), axis=0))[:64].astype(int).tolist()
    asset = {"schema": "c170_role_checkpoint_coordinate_heatmap.v1", "result_type": "role_checkpoint_coordinate_heatmap", "phase": PHASE, "campaign": "C170-C171", "model": "Qwen3-4B", "title": "Role-Conditioned Relation Transport Across q23-q25", "dimensions": list(range(DIM)), "default_coordinates": default, "source_coordinates": coordinates, "rows": rows, "c170": core.load(C170 / "analysis/final.json"), "coordinate_semantics": "Each source id is one of 16 q24-relation discovery-ranked Qwen3 activation coordinates, reused at the named source role/checkpoint. Every column is a next-checkpoint target-role activation coordinate.", "claim_boundary": "Relation-source settings are stable, query-source settings partial, and primary-source settings absent for this frozen coordinate set. This does not prove primary/query lack separately optimized coordinates or establish a minimal natural-use circuit."}
    (OUT / "analysis").mkdir(parents=True, exist_ok=True)
    text = json.dumps(asset, separators=(",", ":"), ensure_ascii=True)
    (OUT / "analysis/heatmap.json").write_text(text, encoding="utf-8")
    FRONTEND.write_text(text, encoding="utf-8")
    report = {"phase": PHASE, "campaign": CAMPAIGN, "status": "asset_built", "rows": len(rows), "dimensions": DIM, "default_coordinates": default, "asset_bytes": FRONTEND.stat().st_size, "asset_sha256": sha(FRONTEND), "next_authorization": "frontend integration and audit"}
    core.save(OUT / "analysis/synthesis.json", report)
    checks = {"rows": len(rows) == 432, "shape": matrix.shape == (432, DIM), "finite": bool(np.isfinite(matrix).all()), "asset_equal": sha(FRONTEND) == sha(OUT / "analysis/heatmap.json")}
    core.save(OUT / "audit/internal_build_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    print(json.dumps(report, indent=2))


def close():
    report = core.load(OUT / "analysis/synthesis.json")
    checks = {"contract": core.load(OUT / "audit/internal_contract_audit.json")["all_checks_passed"], "build": core.load(OUT / "audit/internal_build_audit.json")["all_checks_passed"]}
    final = {"phase": PHASE, "campaign": CAMPAIGN, "status": "closed", "checks": checks, "all_checks_passed": all(checks.values()), "headline": report, "next_authorization": "stage complete; new coordinate-selection contract required"}
    core.save(OUT / "analysis/final.json", final)
    print(json.dumps(final, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("contract", "build", "close"))
    args = parser.parse_args()
    {"contract": contract, "build": build, "close": close}[args.command]()


if __name__ == "__main__":
    main()
