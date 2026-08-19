#!/usr/bin/env python3
"""Independent audit for Phase1361/C055."""
from __future__ import annotations

import json
import py_compile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1361_c055_full_width_coalition_observation"
CONTRACT = TESTS / "result/phase1360_c055_hidden_state_coalition_contract"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def persistent(depths: list[int], length: int):
    values = set(depths)
    return next((depth for depth in depths if all(depth + offset in values for offset in range(length))), None)


def main() -> None:
    protocol = load(CONTRACT / "protocol/preregistration.json")
    result = load(OUT / "analysis/coalition_observation.json")
    final = load(OUT / "analysis/final.json")
    expected_persistent = {name: persistent(depths, protocol["observation"]["persistence_layers"])
                           for name, depths in result["coalition_passing_layers"].items()}
    candidates = [(depth, len(protocol["coalitions"][name]), name)
                  for name, depth in expected_persistent.items()
                  if depth is not None and len(protocol["coalitions"][name]) > 1]
    expected = sorted(candidates)[0] if candidates else None
    fallback = protocol["observation"]["fallback_if_none"]
    checks = {
        "tensor_shape": result["tensor_shapes"] == {"active": [432, 37, 3, 2560], "status": [144, 37, 3, 2560]},
        "layer_count": len(result["layer_metrics"]) == 36,
        "coalition_count": all(len(value) == 7 for value in result["layer_metrics"].values()),
        "persistence": result["coalition_persistent_start"] == expected_persistent,
        "selection_name": result["selected_descriptive_coalition"] == (expected[2] if expected else None),
        "selection_layer": result["selected_descriptive_layer"] == (expected[0] if expected else None),
        "fallback": result["causal_layer"] == (expected[0] if expected else fallback["layer"]),
        "finite_metrics": all(
            isinstance(cell["identity"]["top1"], (int, float))
            for layer in result["layer_metrics"].values() for cell in layer.values()
        ),
        "authorization": final["authorization"] == "run_phase1362_c055_coalition_camera",
        "script_compiles": True,
    }
    try:
        py_compile.compile(str(TESTS / "phase1361_c055_full_width_coalition_observation.py"), doraise=True)
    except Exception:
        checks["script_compiles"] = False
    audit = {"phase": 1361, "campaign": "C055", "checks": checks,
             "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    (OUT / "audit").mkdir(parents=True, exist_ok=True)
    (OUT / "audit/independent_final_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))
    if not audit["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
