#!/usr/bin/env python3
"""Independent result audit for Phase1355/C053."""
from __future__ import annotations

import json
import py_compile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
BEHAVIOR = TESTS / "result/phase1354_c053_behavior_route_competition"
OUT = TESTS / "result/phase1355_c053_full_width_route_fields"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    behavior = load(BEHAVIOR / "analysis/final.json")
    manifest = load(OUT / "protocol/execution_manifest.json")
    final = load(OUT / "analysis/final.json")
    checks = {
        "authorization": behavior["authorization"] == "run_phase1355_c053_fields",
        "fields": manifest["authorized_fields"] == behavior["authorized_fields"] == final["evaluated_fields"],
        "runtime": load(OUT / "analysis/runtime.json")["quantization"]["has_quantized_modules"] is False,
    }
    quartet = None
    if "quartet_interaction_field" in manifest["authorized_fields"]:
        quartet = load(OUT / "analysis/quartet_field_summary.json")
        raw = torch.load(OUT / "raw/qwen3_quartet_fields.pt", map_location="cpu", weights_only=False)
        checks["quartet_shape"] = quartet["shape_active"] == list(raw["active_interactions"].shape) \
            and quartet["shape_status"] == list(raw["status_interactions"].shape)
        checks["quartet_finite"] = torch.isfinite(raw["active_interactions"]).all().item() \
            and torch.isfinite(raw["status_interactions"]).all().item()
        checks["quartet_numeric"] = quartet["numeric_qualified"] and quartet["layer0_qualified"]
        checks["family_candidate"] = final["family_pair_candidate"] == quartet["family_pair_candidate"]
        checks["shared"] = final["shared_relation_qualified"] == quartet["shared_relation_qualified"]
    else:
        checks["no_quartet_claim"] = not final["family_pair_candidate"] and not final["shared_relation_qualified"]
    if "choice_order_invariance_field" in manifest["authorized_fields"]:
        choice = load(OUT / "analysis/choice_field_summary.json")
        raw = torch.load(OUT / "raw/qwen3_choice_fields.pt", map_location="cpu", weights_only=False)
        checks["choice_shape"] = choice["shape"] == list(raw["averaged_states"].shape)
        checks["choice_finite"] = torch.isfinite(raw["averaged_states"]).all().item()
        checks["choice"] = final["choice_field_qualified"] == choice["qualified"]
    else:
        checks["no_choice_claim"] = not final["choice_field_qualified"]
    expected = "run_phase1356_c053_typed_causal" if final["shared_relation_qualified"] else "close_c053_after_fields"
    checks["final_authorization"] = final["authorization"] == expected
    checks["script_compiles"] = True
    try:
        py_compile.compile(str(TESTS / "phase1355_c053_full_width_route_fields.py"), doraise=True)
    except Exception:
        checks["script_compiles"] = False
    result = {"phase": 1355, "campaign": "C053", "checks": checks,
              "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    (OUT / "audit").mkdir(parents=True, exist_ok=True)
    (OUT / "audit/independent_final_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
