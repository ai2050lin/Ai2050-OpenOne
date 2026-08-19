#!/usr/bin/env python3
"""Independent audit for Phase1346 C049 full interaction field."""
from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core

CONTRACT = TESTS / "result/phase1344_c049_disentangled_relation_contract"
BEHAVIOR = TESTS / "result/phase1345_c049_disentangled_behavior"
OUT = TESTS / "result/phase1346_c049_full_interaction_field"


def main():
    protocol = core.load(CONTRACT / "protocol/preregistration.json")
    behavior = core.load(BEHAVIOR / "analysis/final.json")
    final = core.load(OUT / "analysis/final.json")
    manifest = core.load(OUT / "protocol/execution_manifest.json")
    models = behavior["relation_interaction_qualified_models"]
    checks = {
        "contract": manifest["contract_sha256"] == protocol["contract_sha256"],
        "models": manifest["model_order"] == models == final["evaluated_models"],
        "authorization": final["authorization"]
        in ("run_phase1347_c049_same_label_causal_swaps", "close_c049_descriptive_field"),
        "full_dimensional": "no PCA" in manifest["storage"],
    }
    recomputed_qualified = []
    expected_classes = ["__".join(pair) for pair in combinations(protocol["material"]["families"], 2)]
    for model in models:
        bundle = torch.load(OUT / f"raw/{model}_full_interaction_field.pt", map_location="cpu", weights_only=False)
        summary = core.load(OUT / f"analysis/{model}_summary.json")
        vectors = bundle["interaction_vectors"]
        norms = bundle["relative_norms"]
        checks[f"{model}_shape"] = vectors.shape[0] == 432 and vectors.shape[2] == 3 and norms.shape == vectors.shape[:3]
        checks[f"{model}_finite"] = bool(torch.isfinite(vectors).all() and torch.isfinite(norms).all())
        checks[f"{model}_classes"] = bundle["classes"] == expected_classes
        checks[f"{model}_metadata"] = len(bundle["metadata"]) == 432 and len(
            {row["quartet_key"] for row in bundle["metadata"]}
        ) == 432
        checks[f"{model}_layer0"] = abs(float(norms[:, 0, 1].max()) - summary["layer0_max_relative_norm"]) <= 1e-12
        selected = summary["selected_layer"]
        if selected is None:
            selection_valid = not summary["discovery_passing_layers"]
        else:
            selection_valid = selected == min(summary["discovery_passing_layers"])
        checks[f"{model}_selection"] = selection_valid
        checks[f"{model}_numeric"] = summary["numeric_qualified"] == (
            summary["numeric"]["relative_l2_p95"] <= protocol["field_gate"]["numeric_relative_l2_p95_max"]
            and summary["numeric"]["relative_l2_max"] <= protocol["field_gate"]["numeric_relative_l2_max"]
        )
        recomputed_qualified.append(model) if summary["qualified"] else None
    checks["qualified_list"] = recomputed_qualified == final["field_qualified_models"]
    checks["cross_model"] = final["cross_model_field_repetition"] == (
        len(recomputed_qualified) >= protocol["field_gate"]["cross_model_minimum"]
    )
    result = {
        "phase": 1346,
        "campaign": "C049",
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "all_checks_passed": all(checks.values()),
    }
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
