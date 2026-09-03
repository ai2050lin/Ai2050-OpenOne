#!/usr/bin/env python3
"""Reconcile the C566 primary worker artifact after the visual-only resample."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "tests/glm5/result/phase2100_c566_glm4_within_model_functional_response_replication"
FINAL = BASE / "analysis/final.json"
WORKER = BASE / "analysis/glm4_worker_result.json"
INCIDENT = BASE / "audit/glm4_visual_resample_serialization_incident.json"


def main() -> None:
    final = json.loads(FINAL.read_text(encoding="utf-8"))
    overwritten = json.loads(WORKER.read_text(encoding="utf-8"))
    if overwritten.get("status") != "worker_exception" or overwritten.get("exception_type") != "ValueError":
        raise RuntimeError("Expected the documented visual-resample path serialization exception")
    incident = {
        "status": "documented_engineering_incident",
        "scope": "second GLM4 run requested only to add parameter-level visual rows",
        "scientific_execution": "168/168 rows completed and raw arrays were written",
        "failure": overwritten,
        "effect_on_primary_c566_result": "none",
        "resolution": "worker path handling was fixed; visual rows were generated from the complete arrays; primary successful result restored below",
    }
    INCIDENT.write_text(json.dumps(incident, ensure_ascii=False, indent=2), encoding="utf-8")
    restored = dict(final["headline"]["glm4"])
    restored.pop("returncode", None)
    restored["artifact_role"] = "primary_successful_c566_worker_result"
    restored["visual_resample_incident"] = str(INCIDENT.relative_to(ROOT))
    WORKER.write_text(json.dumps(restored, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"restored_status": restored["status"], "incident": str(INCIDENT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
