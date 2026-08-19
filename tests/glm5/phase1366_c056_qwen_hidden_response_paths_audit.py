#!/usr/bin/env python3
"""Independent audit for Phase1366 C056 Qwen path observation."""
from __future__ import annotations

import json
import math
import py_compile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
T = ROOT / "tests/glm5"
OUT = T / "result/phase1366_c056_qwen_hidden_response_paths"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    summary = load(OUT / "analysis/qwen_path_observation.json")
    final = load(OUT / "analysis/final.json")
    bundle = torch.load(OUT / "raw/qwen3_hidden_response_paths.pt", map_location="cpu", weights_only=False)
    paths = summary["path_metrics"]
    recomputed = sorted(name for name, value in paths.items() if all(value["checks"].values()))
    checks = {
        "tensor_shape": list(bundle["clean_minus_corrupt"].shape) == [96, 37, 4, 2560],
        "relative_shape": list(bundle["relative_norm"].shape) == [96, 37, 4],
        "metadata": len(bundle["metadata"]) == 96,
        "roles": bundle["roles"] == ["target", "family", "query", "boundary"],
        "finite": bool(torch.isfinite(bundle["clean_minus_corrupt"]).all()
                       and torch.isfinite(bundle["relative_norm"]).all()),
        "numeric": summary["numeric_relative_l2_max"] <= 1e-6,
        "path_count": len(paths) == 5,
        "metrics_finite": all(math.isfinite(value["identity_gain_over_best_event"])
                              for value in paths.values()),
        "qualification_recomputed": recomputed == sorted(summary["qualified_paths"]),
        "observation_not_gate": final["authorization"] == "run_phase1367_c056_qwen_path_identity_camera",
    }
    py_compile.compile(str(T / "phase1366_c056_qwen_hidden_response_paths.py"), doraise=True)
    checks["script_compiles"] = True
    result = {
        "phase": 1366, "campaign": "C056", "checks": checks,
        "passed": sum(checks.values()), "total": len(checks),
        "all_checks_passed": all(checks.values()),
    }
    path = OUT / "audit/independent_final_audit.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
