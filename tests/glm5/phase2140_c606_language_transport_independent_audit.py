#!/usr/bin/env python3
"""Independent artifact and claim-boundary audit for C600-C605."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
OUT = RESULT / "phase2140_c606_language_transport_independent_audit"
sys.path.insert(0, str(TESTS))

import phase2134_c600_c605_language_transport_campaign as campaign


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def check(name: str, condition: bool, detail) -> dict:
    return {"name": name, "passed": bool(condition), "detail": detail}


def finite(value) -> bool:
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, list):
        return all(finite(v) for v in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def main() -> None:
    checks = []
    finals = {}
    for name in campaign.PHASES:
        path = campaign.OUTS[name] / "analysis/final.json"
        checks.append(check(f"{name}_final_exists", path.exists(), str(path.relative_to(ROOT))))
        if path.exists():
            finals[name] = campaign.load(path)
            checks.append(check(f"{name}_closed", finals[name].get("status") == "closed", finals[name].get("status")))
            checks.append(check(f"{name}_finite", finite(finals[name]), "all numeric values finite"))

    rows = campaign.read_rows(campaign.material_path())
    compiled = campaign.read_rows(campaign.compiled_path())
    ids = [r["case_id"] for r in rows]
    checks.extend([
        check("material_rows_large", len(rows) >= 1500, len(rows)),
        check("material_case_ids_unique", len(ids) == len(set(ids)), len(set(ids))),
        check("compiled_alignment", [r["case_id"] for r in compiled] == ids, len(compiled)),
        check("unit_partitions_disjoint", all(r["partition"] == campaign.partition(r["unit"]) for r in rows),
              {p: sum(r["partition"] == p for r in rows) for p in ("discovery", "confirmation", "lockbox")}),
        check("program_breadth", len({r["family"] for r in rows}) >= 13, sorted({r["family"] for r in rows})),
        check("multi_token_registered", any(any(len(v) > 1 for v in r["candidate_ids"]) for r in compiled), "candidate token lengths"),
    ])

    behavior = campaign.read_rows(campaign.behavior_path())
    index = campaign.read_rows(campaign.capture_index_path())
    qualified = set(campaign.load(campaign.qualified_path())["qualified"])
    captured_slice_keys = {
        f"{r['panel']}|{r['family']}|{r['operation_domain']}" for r in index
    }
    checks.extend([
        check("behavior_complete", len(behavior) == len(rows), len(behavior)),
        check("capture_nonempty", bool(index), len(index)),
        check("capture_slices_qualified", captured_slice_keys <= qualified,
              {"captured": len(captured_slice_keys), "qualified": len(qualified)}),
        check("role_mean_retained", campaign.mean_path().exists(), str(campaign.mean_path().relative_to(ROOT))),
        check("role_last_retained", campaign.last_path().exists(), str(campaign.last_path().relative_to(ROOT))),
        check("bulk_qwen_shards_cleaned", not campaign.shard_dir().exists(), str(campaign.shard_dir().relative_to(ROOT))),
    ])
    if campaign.last_path().exists():
        states = np.load(campaign.last_path(), mmap_mode="r")
        checks.append(check("role_last_shape", list(states.shape) == [len(index), 38, 6, 2560], list(states.shape)))
        checks.append(check("role_last_finite_sample", bool(np.isfinite(np.asarray(states[:min(8, len(states))], np.float32)).all()), "first rows"))
        mmap = getattr(states, "_mmap", None)
        if mmap is not None:
            mmap.close()
        del states

    state_guard = finals.get("C602", {}).get("headline", {})
    composition = finals.get("C603", {}).get("headline", {})
    causal = finals.get("C604", {}).get("headline", {})
    checks.extend([
        check("state_guard_metrics_present", bool(state_guard.get("metrics")), len(state_guard.get("metrics", {}))),
        check("composition_four_ledgers", set(composition.get("results", {})) == {"factorial", "sequence", "attitude", "graph"},
              sorted(composition.get("results", {}))),
        check("causal_records_present", int(causal.get("records", 0)) > 0, causal.get("records")),
        check("causal_claims_separated", all(all(k in d for k in ("state_guidance", "candidate_output", "open_output", "generated_output", "necessity", "rescue"))
                                              for value in causal.get("summary", {}).values() for d in value.values()),
              "state/output/necessity/rescue"),
    ])

    atlas_path = campaign.VISUAL
    checks.append(check("atlas_exists", atlas_path.exists(), str(atlas_path.relative_to(ROOT))))
    if atlas_path.exists():
        atlas = campaign.load(atlas_path)
        representative = atlas.get("qwen3_4b", {}).get("representative", {})
        shape = representative.get("shape", [])
        checks.extend([
            check("atlas_schema", atlas.get("schema") == "ai2050.language_transport_output_atlas.v1", atlas.get("schema")),
            check("atlas_full_coordinates", atlas.get("qwen3_4b", {}).get("coordinates") == 2560, atlas.get("qwen3_4b", {}).get("coordinates")),
            check("atlas_exact_field_shape", len(shape) == 3 and shape[0] == 38 and shape[2] == 2560, shape),
            check("atlas_no_topk_policy", "no PCA" in atlas.get("coordinate_policy", "") and "Top-K" in atlas.get("coordinate_policy", ""), atlas.get("coordinate_policy")),
            check("atlas_cross_model_ledger", set(atlas.get("cross_model", {})) == {"glm4", "deepseek7b", "qwen3_14b"}, sorted(atlas.get("cross_model", {}))),
        ])
    catalog = campaign.load(campaign.CATALOG)
    artifacts = [v for v in catalog.get("artifacts", []) if v.get("id") == "c605_language_transport_output_atlas"]
    checks.append(check("catalog_entry", len(artifacts) == 1, artifacts))

    empirical = finals.get("C605", {}).get("headline", {}).get("empirical_gates", {})
    checks.extend([
        check("new_math_not_predeclared", empirical.get("new_math") is False, empirical.get("new_math")),
        check("human_naturalness_not_fabricated", empirical.get("human_naturalness") is False, empirical.get("human_naturalness")),
        check("theory_name_stable", finals.get("C605", {}).get("headline", {}).get("theory", {}).get("name") == "Conditional Output Field Closure Theory",
              finals.get("C605", {}).get("headline", {}).get("theory", {})),
    ])
    all_passed = all(v["passed"] for v in checks)
    next_object = "natural human-validated open generation and model-relative output compiler" if all_passed else "repair_failed_artifact_audit_only"
    same_exact_goal = False
    result = {
        "phase": 2140, "campaign": "C606", "status": "closed", "all_checks_passed": all_passed,
        "headline": {"status": "language_transport_campaign_independent_audit_closed",
                     "checks_passed": sum(v["passed"] for v in checks), "checks_total": len(checks),
                     "empirical_gates": empirical,
                     "route": {"same_exact_goal": same_exact_goal,
                               "completed_object": "broad language-program observation, state-guard tournament, sequential composition, output deletion/rescue and model-relative topology",
                               "next_object": next_object,
                               "why_not_automatic_same_goal": "The frozen C600-C605 object is exhausted. Human blind naturalness and natural open generation require a new external-data contract, not another reveal of this lockbox.",
                               "foundational_math_authorized": False,
                               "strict_boundary": "Conditional transport evidence is not a unique circuit, curvature, a fiber bundle, or proof that existing mathematics is insufficient."}},
        "checks": checks, "next_authorization": "new_external_validity_contract_freeze",
    }
    save(OUT / "analysis/final.json", result)
    save(OUT / "audit/checks.json", {"checks": checks, "all_checks_passed": all_passed})
    save(OUT / "protocol/preregistration.json", {"phase": 2140, "campaign": "C606",
                                                   "audit_only": True, "source_campaign": "C600-C605"})
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
