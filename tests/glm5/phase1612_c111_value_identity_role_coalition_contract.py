#!/usr/bin/env python3
"""Phase1612 / C111: freeze the read-only value-identity and role-coalition observation contract."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
C109 = TESTS / "result/phase1603_c109_fresh_role_state_field_atlas"
C110 = TESTS / "result/phase1607_c110_fresh_readout_control_separation"
OUT = TESTS / "result/phase1612_c111_value_identity_role_coalition_observation"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    sources = {
        "c109_protocol": C109 / "protocol/preregistration.json",
        "c109_mean_field": C109 / "analysis/mean_truth_role_state.float32.npy",
        "c110_protocol": C110 / "protocol/preregistration.json",
        "c110_compiled": C110 / "compiled/qwen3.jsonl",
        "c110_manifest": C110 / "protocol/role_occurrence_manifest.jsonl",
        "c110_raw_field": C110 / "raw/qwen3_role_subtoken_all_states.uint16.npy",
        "c110_mean_field": C110 / "analysis/mean_truth_role_state.float32.npy",
        "c110_transport_adapter": C110 / "protocol/transport_adapter.json",
        "c110_transport_rows": C110 / "analysis/fresh_transport_results.jsonl",
        "c110_closure": C110 / "analysis/closure.json",
    }
    missing = [name for name, path in sources.items() if not path.exists()]
    if missing:
        raise RuntimeError(missing)
    closure = core.load(sources["c110_closure"])
    if closure["next_authorization"] != "C111 observation-first value-identity and role-coalition atlas on the frozen C109-C110 archives; analyze value-preserving shuffles and role-conditioned response before any new model run":
        raise RuntimeError("C111 authorization mismatch")
    OUT.mkdir(parents=True, exist_ok=True)
    source_hashes = {name: core.sha(path) for name, path in sources.items()}
    contract = {
        "phase": 1612,
        "campaign": "C111",
        "created_at_utc": now(),
        "status": "read_only_observation_contract_frozen",
        "object": "separate support location, transported value identity, role-conditioned field geometry, and role-coalition output increment using frozen C109-C110 archives",
        "model_run": "forbidden; read-only archive analysis",
        "families": ["attribute_binding", "agent_patient"],
        "roles": core.load(sources["c110_protocol"])["roles"],
        "state": 19,
        "states": 37,
        "coordinates": 2560,
        "observations": [
            "pairwise BF16 target-vs-coordinate-permuted movement cosine and field alignment",
            "paired output-gain difference already measured in Phase1610",
            "C109-to-C110 full-vector role-state trajectory cosine without dimensional reduction",
            "within-C110 state19 role-to-role full-vector cosine matrix",
            "whole-query versus query-plus-focus-record paired margin increment and flip accounting",
        ],
        "adjudication": "descriptive observation only; no post-hoc mechanism pass gate",
        "basic_summaries": ["count", "median", "minimum", "maximum", "full-coordinate cosine"],
        "forbidden": ["PCA", "dimensional reduction", "attention decomposition", "MLP decomposition", "weight inspection", "new support selection", "new model execution"],
        "planned_missingness": [
            "only one frozen random donor-value permutation exists",
            "only query_anchor and query_anchor-plus-focus_record causal transports exist",
            "no single-role focus_record transport, all-role subset lattice, zero ablation, or natural necessity test exists",
            "no independent human naturalness audit or second model exists",
        ],
        "source_paths": {name: str(path) for name, path in sources.items()},
        "source_hashes": source_hashes,
        "authorization": "run_phase1613_c111_read_only_observation",
    }
    protocol = OUT / "protocol/preregistration.json"
    core.save(protocol, contract)
    checks = {
        "sources": len(source_hashes) == len(sources),
        "authorization": "C111 observation-first" in closure["next_authorization"],
        "read_only": contract["model_run"].startswith("forbidden"),
        "full_coordinates": contract["coordinates"] == 2560 and "PCA" in contract["forbidden"],
        "roles": len(contract["roles"]) == 7,
        "missingness": len(contract["planned_missingness"]) == 4,
        "no_gate": "no post-hoc mechanism pass gate" in contract["adjudication"],
    }
    report = {"phase": 1612, "campaign": "C111", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "producer_sha256": core.sha(Path(__file__)), "protocol_sha256": core.sha(protocol)}
    if not report["all_checks_passed"]:
        raise RuntimeError(report)
    core.save(OUT / "audit/internal_contract_audit.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
