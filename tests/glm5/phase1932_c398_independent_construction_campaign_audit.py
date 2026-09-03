#!/usr/bin/env python3
"""Independent audit for C391-C398 / Phase1925-1932."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))

import phase1925_c391_c398_independent_construction_lockbox as campaign


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    finals = {name: load(out / "analysis/final.json") for name, out in campaign.OUTS.items()}
    checks: dict[str, bool] = {}
    checks["phase_sequence"] = [finals[f"C{value}"]["phase"] for value in range(391, 399)] == list(range(1925, 1933))
    checks["all_closed"] = all(value["all_checks_passed"] and value["status"] == "closed" for value in finals.values())
    protocol_hashes = {name: load(out / "protocol/preregistration.json")["producer_sha256"] for name, out in campaign.OUTS.items()}
    checks["upstream_pre_fix_hash_consistent"] = len({protocol_hashes[f"C{value}"] for value in range(391, 395)}) == 1
    checks["downstream_current_hash_consistent"] = all(protocol_hashes[f"C{value}"] == sha(TESTS / "phase1925_c391_c398_independent_construction_lockbox.py") for value in range(395, 399))
    checks["superseded_index_attempt_preserved"] = (campaign.RESULT / "c395_superseded_pre_cell_index_parse_fix_20260824").exists()

    c392 = finals["C392"]["headline"]
    checks["material_960_balanced"] = c392["rows"] == 960 and max(c392["zero_model_accuracies"].values()) == 0.5
    checks["scope_output_sensitive"] = c392["scope_order_is_output_sensitive"] is True
    checks["naturalness_not_overclaimed"] = c392["human_naturalness_review"] is False

    c393 = finals["C393"]["headline"]
    checks["behavior_qualified"] = c393["heldout_accuracy"] > 0.99 and set(c393["eligible_families"]) == set(campaign.FAMILIES)
    checks["all_surfaces_qualified"] = min(c393["surface_accuracy"].values()) >= 0.975
    c394 = finals["C394"]["headline"]
    checks["full_coordinate_shape"] = c394["role_shape"] == [960, 38, 6, 2560] and c394["full_shape"] == [12, 38, 192, 2560]

    c395 = finals["C395"]["headline"]
    checks["conditional_three_of_three"] = set(c395["conditional_passed"]) == set(campaign.FAMILIES)
    checks["conditional_gains_are_small"] = all(0 < row["conditional_gain_over_mean"] < 0.02 for row in c395["results"])
    checks["old_transfer_only_negation"] = c395["old_transfer_passed"] == ["negation_scope"]
    checks["causal_and_attribute_old_transfer_fail"] = all(not row["old_transfer_passed"] for row in c395["results"] if row["family"] != "negation_scope")

    c396 = finals["C396"]["headline"]
    checks["fresh_output_sensitive_scope_passed"] = c396["fresh_scope_order_passed"] is True
    checks["old_scope_failed_to_transfer"] = c396["old_scope_order_transfer_passed"] is False
    checks["fresh_scope_beats_controls"] = c396["nrmse"]["discovery_mean"] < min(c396["nrmse"]["zero"], c396["nrmse"]["coordinate_roll"], c396["nrmse"]["wrong_family_k"])

    c397 = finals["C397"]["headline"]
    checks["ecology_decoding_descriptive"] = c397["family_accuracy"] == 1.0 and c397["descriptive_candidate"] is True
    checks["embedding_confound_disclosed"] = c397["first_energy_differentiation_checkpoint"] == 0 and "confounded" in c397["strict_interpretation"]

    c398 = finals["C398"]["headline"]
    checks["no_causal_claim"] = c398["gates"]["causal"] is False
    checks["new_math_closed"] = c398["new_math_gate_passed"] is False
    visual = load(ROOT / "frontend/public/vis_data/research_kernel/c398_independent_construction_lockbox.json")
    checks["visual_full_coordinates"] = visual["schema"] == "c398.independent_construction_lockbox.v1" and len(visual["rows"]) == 90 and all(len(row["values"]) == 2560 for row in visual["rows"])
    cleanup = load(campaign.OUTS["C398"] / "audit/cleanup.json")
    checks["cleanup_checksummed"] = len(cleanup) == 2 and all(item["sha256"] and item["removed"] for item in cleanup)
    checks["cleanup_paths_absent"] = all(not (ROOT / item["path"]).exists() for item in cleanup)

    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise AssertionError(failed)
    result = {
        "phase": 1932, "campaign": "C398", "audit": "independent",
        "checks": checks, "passed": sum(checks.values()), "total": len(checks),
        "all_checks_passed": True,
        "strict_conclusion": "Fresh within-material conditional response prediction is weakly positive for three candidates; only negation shows weak old-atlas I transfer, old scope-order K does not transfer, and perfect family decoding is embedding-confounded. No causal or new-mathematics claim is authorized.",
    }
    out = campaign.OUTS["C398"] / "audit/independent_audit.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
