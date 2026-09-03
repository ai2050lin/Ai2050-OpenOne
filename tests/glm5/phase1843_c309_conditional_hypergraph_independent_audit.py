#!/usr/bin/env python3
"""C309: independent integrity and claim-boundary audit for C293-C308."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import phase1827_c293_c309_conditional_hypergraph_common as common

core, OUT = common.core, common.OUTS["C309"]


def close(a: float, b: float, tol: float = 1e-8) -> bool:
    return abs(float(a) - float(b)) <= tol


def main() -> None:
    if (OUT / "analysis/final.json").exists() and core.load(OUT / "analysis/final.json").get("all_checks_passed"):
        raise RuntimeError(OUT)
    finals = {f"C{i}": core.load(common.OUTS[f"C{i}"] / "analysis/final.json") for i in range(293, 309)}
    for sub in ("analysis", "audit", "protocol"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    protocol = {"phase": 1843, "campaign": "C309", "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "independent_audit_frozen", "scope": "recompute phase continuity, corrected lockbox identity and hashes, registered gates, array shapes, key aggregates, heatmap completeness and claim boundaries without model execution", "producer_sha256": core.sha(Path(__file__))}
    core.save(OUT / "protocol/preregistration.json", protocol)

    compiled = core.rows(common.OUTS["C294"] / "compiled/qwen3.jsonl")
    c295_protocol = core.load(common.OUTS["C295"] / "protocol/preregistration.json")
    c300_protocol = core.load(common.OUTS["C300"] / "protocol/preregistration.json")
    c302_protocol = core.load(common.OUTS["C302"] / "protocol/preregistration.json")
    c303_protocol = core.load(common.OUTS["C303"] / "protocol/preregistration.json")
    c304_protocol = core.load(common.OUTS["C304"] / "protocol/preregistration.json")
    asset_manifest = core.load(common.OUTS["C308"] / "analysis/heatmap_manifest.json")
    asset_path = common.ROOT / asset_manifest["asset"]
    asset = core.load(asset_path)
    c306_samples = core.rows(common.OUTS["C306"] / "raw/sample_results.jsonl")
    c307 = finals["C307"]["headline"]
    c302_groups = core.rows(common.OUTS["C302"] / "raw/group_results.jsonl")
    c302_families = finals["C302"]["headline"]["families"]
    recomputed_family_gains = {}
    for family in common.FAMILIES:
        family_groups = [x for x in c302_groups if x["family"] == family]
        additive = float(np.mean([x["additive_mae"] for x in family_groups]))
        corrected = float(np.mean([x["corrected_mae"] for x in family_groups]))
        recomputed_family_gains[family] = (additive - corrected) / max(additive, 1e-12)
    phase_ok = all(finals[f"C{i}"]["phase"] == 1827 + i - 293 and finals[f"C{i}"]["campaign"] == f"C{i}" for i in range(293, 309))
    checks = {
        "01_all_16_parents_closed": len(finals) == 16 and all(x["all_checks_passed"] for x in finals.values()),
        "02_phase_campaign_continuity": phase_ok,
        "03_compiler_direct_base_v2": core.load(common.OUTS["C293"] / "protocol/preregistration.json").get("material_compiler_version") == "direct_base_v2",
        "04_sixth_identity_is_adira": compiled[0]["role_values"]["primary"] == "Adira",
        "05_fifth_identity_not_reused": compiled[0]["role_values"]["primary"] != "Anika",
        "06_compiled_rows_768": len(compiled) == 768,
        "07_compiled_hash_matches_c295": c295_protocol["compiled_sha256"] == core.sha(common.OUTS["C294"] / "compiled/qwen3.jsonl"),
        "08_c300_lockbox_hash_matches": c300_protocol["lockbox_sha256"] == core.sha(common.OUTS["C295"] / "analysis/final.json"),
        "09_c302_lockbox_hash_matches": c302_protocol["lockbox_sha256"] == c300_protocol["lockbox_sha256"],
        "10_c303_parent_hash_matches": c303_protocol["parent_sha256"] == core.sha(common.OUTS["C302"] / "analysis/final.json"),
        "11_c304_parent_hash_matches": c304_protocol["parent_sha256"] == core.sha(common.OUTS["C302"] / "analysis/final.json"),
        "12_c295_full_field_shape": list(np.load(common.OUTS["C295"] / "raw/full_fields.float16.npy", mmap_mode="r").shape) == [768, 38, 128, 2560],
        "13_c296_map_shape": list(np.load(common.OUTS["C296"] / "analysis/complete_transition_maps.int8.npy", mmap_mode="r").shape) == [6, 36, 6, 27, 2560],
        "14_c297_atlas_shape": list(np.load(common.OUTS["C297"] / "analysis/amplitude_coordinate_atlas.float32.npy", mmap_mode="r").shape) == [6, 4, 2560],
        "15_c300_atlas_shape": list(np.load(common.OUTS["C300"] / "analysis/lockbox_coordinate_score_atlas.float32.npy", mmap_mode="r").shape) == [6, 5, 2560],
        "16_c302_atlas_shape": list(np.load(common.OUTS["C302"] / "analysis/composition_coordinate_atlas.float32.npy", mmap_mode="r").shape) == [6, 4, 2560],
        "17_c302_all_six_family_gains_positive": all(float(x["relative_mae_gain"]) > 0.01 for x in c302_families),
        "18_c302_family_gains_recomputed": all(close(recomputed_family_gains[x["family"]], x["relative_mae_gain"], 1e-6) for x in c302_families),
        "19_c305_exactly_one_branch": finals["C305"]["headline"]["qualified_count"] == 1 and finals["C305"]["headline"]["qualified"][0]["model"] == "M4_all_token",
        "20_c306_sixteen_samples": len(c306_samples) == 16,
        "21_c306_gate_failure_preserved": not finals["C306"]["headline"]["branches"][0]["causal_gate_passed"],
        "22_c306_unquantized_bf16": finals["C306"]["headline"]["quantization"]["has_bf16_parameters"] and not finals["C306"]["headline"]["quantization"]["has_quantized_modules"],
        "23_c307_three_pairs_pass": len(c307["pairs"]) == 3 and all(x["pair_gate_passed"] for x in c307["pairs"]),
        "24_c307_exact_p_recomputed": all(close(x["exact_upper_p"], 2 / 721) for x in c307["pairs"]),
        "25_asset_hash": core.sha(asset_path) == asset_manifest["sha256"],
        "26_asset_all_coordinates": len(asset["dimensions"]) == 2560 and all(len(x["values"]) == 2560 for x in asset["rows"]),
        "27_asset_embedding_hiddenstate": any(x["checkpoint_type"] == "embedding" for x in asset["rows"]) and any(x["checkpoint_type"] == "hidden_state" for x in asset["rows"]),
        "28_asset_activation_semantics": "activation coordinates" in asset["coordinate_semantics"] and "not parameter indices" in asset["coordinate_semantics"],
        "29_causal_claim_not_promoted": asset["summary"]["causal_branches_passing"] == 0 and "not a unique causal" in asset["claim_boundary"],
        "30_new_math_gate_closed": not finals["C308"]["headline"]["theory"]["new_math_gate"]["gate_open"],
    }
    report = {"phase": 1843, "campaign": "C309", "status": "independent_audit_complete", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "strict_conclusion": "The corrected C293-C308 record is internally reproducible. The strongest result remains six-family held-out field composition; the registered causal coalition failed, and no new-mathematics claim is authorized.", "next_authorization": "major_stage_closed_future_campaign_requires_fresh_preregistration"}
    core.save(OUT / "analysis/summary.json", report)
    core.save(OUT / "audit/independent_audit.json", report)
    final_checks = {"audit": all(checks.values()), "producer_hash": core.sha(Path(__file__)) == protocol["producer_sha256"]}
    final = {"phase": 1843, "campaign": "C309", "status": "closed", "checks": final_checks, "all_checks_passed": all(final_checks.values()), "headline": report, "next_authorization": report["next_authorization"]}
    core.save(OUT / "analysis/final.json", final)
    print(json.dumps(final, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
