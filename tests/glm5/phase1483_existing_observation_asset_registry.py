#!/usr/bin/env python3
"""Phase1483: inventory legal full-state assets and predefined missingness."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
PARENT = RESULT / "phase1482_layered_observation_policy"
OUT = RESULT / "phase1483_existing_observation_asset_registry"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def asset(asset_id: str, campaign: str, path: Path, expected_sha: str, status: str, scope: str, selected: bool) -> dict:
    actual_sha = core.sha(path)
    return {
        "asset_id": asset_id,
        "campaign": campaign,
        "path": str(path.relative_to(ROOT)),
        "bytes": path.stat().st_size,
        "sha256": actual_sha,
        "expected_sha256": expected_sha,
        "hash_valid": actual_sha == expected_sha,
        "observation_status": status,
        "scope": scope,
        "selected_for_c084": selected,
    }


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1483 exists")
    parent = core.load(PARENT / "analysis/final.json")
    parent_audit = core.load(PARENT / "audit/independent_final_audit.json")
    policy = core.load(PARENT / "protocol/layered_observation_policy.json")
    c037_meta = core.load(RESULT / "phase1322_c037_isomorphic_full_state_field/raw/field_metadata.json")
    c039_meta = core.load(RESULT / "phase1326_c039_composition_field/raw/field_metadata.json")
    c079_d = core.load(RESULT / "phase1465_c079_discovery_full_field_capture/analysis/capture_metadata.json")
    c079_h = core.load(RESULT / "phase1467_c079_holdout_capture_and_validation/analysis/holdout_summary.json")
    c082 = core.load(RESULT / "phase1476_c082_coordinate_atlas/analysis/atlas_metadata.json")
    assets = [
        asset("C037_full_state_npz", "C037", RESULT / "phase1322_c037_isomorphic_full_state_field/raw/full_state_field_arrays.npz", c037_meta["arrays_sha256"], "M4", "already-open failed field-gate object; different task", False),
        asset("C039_composition_field_npz", "C039", RESULT / "phase1326_c039_composition_field/raw/full_state_composition_field.npz", c039_meta["arrays_sha256"], "M4", "already-open failed composition-field object; different task", False),
        asset("C079_discovery_full_field", "C079", RESULT / "phase1465_c079_discovery_full_field_capture/raw/discovery_role_field.float16.npy", c079_d["raw_sha256"], "M0", "behavior-qualified explicit-label full field", True),
        asset("C079_confirmation_lockbox_full_field", "C079", RESULT / "phase1467_c079_holdout_capture_and_validation/raw/holdout_role_field.float16.npy", c079_h["raw_sha256"], "M0", "behavior-qualified explicit-label full field; all splits now open", True),
        asset("C082_relation_mean_atlas", "C082", RESULT / "phase1476_c082_coordinate_atlas/atlas/mean_effect.float32.npy", c082["files"]["mean_effect.float32.npy"]["sha256"], "M4", "derived retrospective C079 relation-effect atlas", True),
        asset("C082_sign_atlas", "C082", RESULT / "phase1476_c082_coordinate_atlas/atlas/sign_consistency.float16.npy", c082["files"]["sign_consistency.float16.npy"]["sha256"], "M4", "derived retrospective C079 sign-consistency atlas", True),
    ]
    missing = [
        {"campaign": "C080", "code": "M1", "reason": "behavior gate denied Hidden State capture"},
        {"campaign": "C081", "code": "M1", "reason": "one-shot behavior rescue failed before Hidden State capture"},
        {"campaign": "C083", "code": "M1", "reason": "fresh behavior gate failed before Hidden State capture; P082 remains untested"},
    ]
    checks = {
        "parent": parent["authorization"] == "run_phase1483_existing_observation_asset_registry" and parent_audit["all_checks_passed"],
        "policy": policy["policy_sha256"] == parent["policy_sha256"],
        "hashes": all(row["hash_valid"] for row in assets),
        "selected_same_object": all(row["campaign"] in {"C079", "C082"} for row in assets if row["selected_for_c084"]),
        "historical_missingness": [row["campaign"] for row in missing] == ["C080", "C081", "C083"] and all(row["code"] == "M1" for row in missing),
        "no_model_run": True,
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    registry = {
        "phase": 1483,
        "schema": "glm5.existing_observation_asset_registry.v1",
        "policy_sha256": policy["policy_sha256"],
        "assets": assets,
        "predefined_missingness": missing,
        "selection_rule": "C084 uses only C079 immutable raw fields and their C082 deterministic derivatives; different-task assets remain registered but unpooled",
        "total_registered_bytes": sum(row["bytes"] for row in assets),
        "selected_bytes": sum(row["bytes"] for row in assets if row["selected_for_c084"]),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    registry["registry_sha256"] = core.digest(registry)
    core.save(OUT / "analysis/asset_registry.json", registry)
    core.save(OUT / "analysis/final.json", {
        "phase": 1483,
        "status": "legal_assets_and_missingness_registered",
        "checks": checks,
        "registry_sha256": registry["registry_sha256"],
        "model_run": False,
        "authorization": "preregister_c084_c079_batch_deep_mining",
    })
    print(json.dumps({"checks": checks, "asset_count": len(assets), "selected_bytes": registry["selected_bytes"], "registry_sha256": registry["registry_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
