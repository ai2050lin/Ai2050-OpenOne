#!/usr/bin/env python3
"""Independent audit for C231."""
from __future__ import annotations

import json
from pathlib import Path

import phase1757_c223_surface_transport_common as common

core = common.core
OUT = common.OUTS["C231"]


def main() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    final = core.load(OUT / "analysis/final.json")
    checks = {"final": final["all_checks_passed"], "three_models": set(final["headline"]["models"]) == {"qwen3", "glm4", "deepseek7b"}, "sequential": protocol["sequential_loading"], "fixed_interfaces": protocol["interfaces"] == {"qwen3": "strict_chat", "glm4": "strict_chat", "deepseek7b": "plain"}, "no_coordinates": all(len(row) == 6 for topology in final["headline"]["topologies"].values() for row in topology), "producer_hash": core.sha(Path(__file__).with_name("phase1765_c231_cross_model_functional_topology.py")) == protocol["producer_sha256"]}
    audit = {"phase": 1765, "campaign": "C231", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "authorization": final["next_authorization"]}
    core.save(OUT / "audit/independent_final_audit.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
