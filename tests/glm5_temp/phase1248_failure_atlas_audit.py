#!/usr/bin/env python3
"""Independent recomputation audit for the Phase1248 descriptive failure atlas."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests/glm5"
TEMP_ROOT = ROOT / "tests/glm5_temp"
sys.path[:0] = [str(TEST_ROOT), str(TEMP_ROOT)]
import phase1248_c002_qwen_self_response_atlas as main  # noqa: E402
import phase1248_failure_atlas as atlas_main  # noqa: E402

OUT = main.OUT_ROOT / "audit/independent_failure_atlas_audit.json"


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def close(a: float, b: float) -> bool:
    return math.isclose(float(a), float(b), rel_tol=1e-8, abs_tol=1e-8)


def cli() -> None:
    value = json.loads(atlas_main.OUT.read_text(encoding="utf-8"))
    copy = dict(value)
    stored = copy.pop("atlas_digest")
    rows = main.read_jsonl(main.TOKEN_PATH)
    arrays = np.load(main.ARRAY_PATH)
    source = main.read_json(main.ATLAS_PATH)
    event = next(i for i, row in enumerate(main.EVENTS) if row["event_id"] == source["selected_event"]["event_id"])
    recomputed = atlas_main.grouped_camera(arrays, rows, event, ("representation",))
    observed = value["variants"]["by_representation"]
    checks = {
        "digest": stored == digest(copy),
        "source_digest": value["source_atlas_digest"] == source["atlas_digest"],
        "selected_event_frozen": value["selected_event_frozen"] == source["selected_event"]["event_id"],
        "no_reselection": value["event_reselection_performed"] is False,
        "representation_aggregate": all(close(recomputed["aggregate"][key], observed["aggregate"][key]) for key in recomputed["aggregate"]),
        "parameter_count": recomputed["camera_parameter_count"] == observed["camera_parameter_count"],
        "no_authorization": value["phase1249_authorized"] is False,
    }
    payload = {
        "phase": main.PHASE,
        "schema_version": "phase1248.descriptive_failure_atlas.audit.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "check_count": len(checks),
        "all_checks_passed": all(checks.values()),
        "claim_boundary": "This audit confirms arithmetic and no-reselection only; the post-hoc atlas is not upgraded evidence.",
    }
    payload["audit_digest"] = digest(payload)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(canonical({"status": "phase1248_failure_atlas_audit", "passed": payload["all_checks_passed"], "checks": len(checks)}))
    if not payload["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    cli()
