#!/usr/bin/env python3
"""Independent artifact and claim audit for C607-C611."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
OUT = RESULT / "phase2146_c612_natural_output_compiler_independent_audit"
sys.path.insert(0, str(TESTS))

import phase2141_c607_c611_natural_output_compiler_campaign as campaign


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def finite(value) -> bool:
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, list):
        return all(finite(v) for v in value)
    return not isinstance(value, float) or math.isfinite(value)


def check(name: str, passed: bool, detail) -> dict:
    return {"name": name, "passed": bool(passed), "detail": detail}


def main() -> None:
    (OUT / "protocol").mkdir(parents=True, exist_ok=True)
    (OUT / "analysis").mkdir(parents=True, exist_ok=True)
    save(OUT / "protocol/preregistration.json", {
        "phase": 2146, "campaign": "C612", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "object": "independent artifact, arithmetic, missingness, cleanup and claim-boundary audit",
        "scope": ["C607", "C608", "C609", "C610", "C611"],
        "rule": "No model rerun; recompute from frozen artifacts and do not convert pending human review into evidence.",
    })
    finals = {name: campaign.final(name) for name in campaign.PHASES}
    material = campaign.read_rows(campaign.material_path())
    compiled = campaign.read_rows(campaign.compiled_path())
    behavior = campaign.read_rows(campaign.behavior_path())
    qualified = campaign.load(campaign.qualified_path())
    ids = [r["case_id"] for r in material]
    prompts = [r["prompt"] for r in material]
    bmap = {r["case_id"]: r for r in behavior}
    candidate_accuracy = float(np.mean([r["candidate_correct"] for r in behavior]))
    generated_accuracy = float(np.mean([r["generated_correct"] for r in behavior]))
    review_path = campaign.OUTS["C607"] / "external/human_blind_review_template.csv"
    with review_path.open("r", encoding="utf-8-sig", newline="") as handle:
        review = list(csv.DictReader(handle))
    review_blank = all(not r["semantic_unique_0_1"] and not r["natural_1_5"] and not r["reviewer_id"] for r in review)

    hidden_index = campaign.read_rows(campaign.capture_index_path())
    last = np.load(campaign.last_path(), mmap_mode="r")
    mean = np.load(campaign.mean_path(), mmap_mode="r")
    role_shape = list(last.shape)
    mean_shape = list(mean.shape)
    role_dtype = str(last.dtype)
    mean_dtype = str(mean.dtype)
    for array in (last, mean):
        mmap = getattr(array, "_mmap", None)
        if mmap is not None:
            mmap.close()
    del last, mean

    causal = campaign.read_rows(campaign.OUTS["C610"] / "analysis/causal_records.jsonl")
    rescue_contract = all(
        ("rescue_ok" not in r["target_values"] and not r["target_values"]["rescue_eligible"])
        or r["target_values"]["rescue_eligible"] for r in causal
    )
    worker_files = {name: campaign.OUTS["C611"] / f"analysis/{name}_worker.json"
                    for name in ("glm4", "deepseek7b")}
    worker_files["qwen3_14b"] = campaign.OUTS["C611"] / "analysis/qwen14_worker.json"
    workers = {name: campaign.load(path) for name, path in worker_files.items()}
    worker_gating = all(
        (v.get("hiddenstate_ran") is True and v.get("candidate_accuracy", 0) >= campaign.BEHAVIOR_GATE
         and v.get("generated_accuracy", 0) >= campaign.BEHAVIOR_GATE)
        or (v.get("hiddenstate_ran") is False) for v in workers.values()
    )
    raw_cleanup = all(not (campaign.OUTS["C611"] / f"raw/{name}/role_last.float16.npy").exists()
                      for name in workers)
    visual = campaign.VISUAL
    catalog = campaign.load(campaign.CATALOG)
    visual_entry = [r for r in catalog.get("artifacts", []) if r.get("id") == "c611_natural_output_compiler_atlas"]
    with visual.open("r", encoding="utf-8") as handle:
        visual_text = handle.read(65536)
    exact_shape_token = '"shape": [' in visual_text and '"coordinates": 2560' in visual_text

    checks = [
        check("phase_finals_closed", all(v["status"] == "closed" and v["all_checks_passed"] for v in finals.values()),
              {k: v["all_checks_passed"] for k, v in finals.items()}),
        check("material_rows_and_uniqueness", len(material) == 784 and len(set(ids)) == len(set(prompts)) == 784,
              {"rows": len(material), "ids": len(set(ids)), "prompts": len(set(prompts))}),
        check("partitions_frozen", {p: sum(r["partition"] == p for r in material) for p in ("discovery", "confirmation", "lockbox")} ==
              {"discovery": 392, "confirmation": 196, "lockbox": 196}, "392/196/196"),
        check("behavior_complete", len(compiled) == len(behavior) == len(material) and set(ids) == set(bmap),
              {"compiled": len(compiled), "behavior": len(behavior)}),
        check("behavior_arithmetic", abs(candidate_accuracy - finals["C607"]["headline"]["candidate_accuracy"]) < 1e-12
              and abs(generated_accuracy - finals["C607"]["headline"]["generated_accuracy"]) < 1e-12,
              {"candidate": candidate_accuracy, "generated": generated_accuracy}),
        check("human_review_pending_not_fabricated", review_blank and finals["C607"]["headline"]["human_blind_naturalness"].startswith("NA"),
              {"rows": len(review), "blank": review_blank}),
        check("qualified_slice_ledger", len(qualified["slices"]) == finals["C607"]["headline"]["total_slices"],
              {"qualified": len(qualified["qualified"]), "total": len(qualified["slices"])}),
        check("retained_all_coordinate_role_fields", role_shape[1:] == [38, 6, 2560] and mean_shape == role_shape
              and role_dtype == mean_dtype == "float16", {"last": role_shape, "mean": mean_shape, "dtype": role_dtype}),
        check("capture_index_matches_tensor", role_shape[0] == len(hidden_index), {"rows": len(hidden_index), "shape": role_shape}),
        check("full_token_bulk_cleaned_after_visualization", not campaign.shard_dir().exists(), str(campaign.shard_dir())),
        check("tomography_bulk_cleaned", not (campaign.OUTS["C608"] / "raw/tomography_role_fields.float16.npy").exists(), "summary retained"),
        check("program_ledgers_present", finals["C609"]["headline"]["summary"]["graph"]["total"] > 0
              and finals["C609"]["headline"]["summary"]["sequence"]["total"] > 0
              and finals["C609"]["headline"]["summary"]["scope"]["registered_missing"] == 4,
              finals["C609"]["headline"]["summary"]),
        check("adaptive_rescue_contract", rescue_contract, {"records": len(causal)}),
        check("cross_model_behavior_first", worker_gating, {k: {x: v.get(x) for x in ("status", "candidate_accuracy", "generated_accuracy", "hiddenstate_ran")} for k, v in workers.items()}),
        check("cross_model_bulk_cleaned", raw_cleanup, {k: not (campaign.OUTS["C611"] / f"raw/{k}/role_last.float16.npy").exists() for k in workers}),
        check("visual_registered", visual.exists() and visual.stat().st_size > 0 and len(visual_entry) == 1 and exact_shape_token,
              {"path": str(visual.relative_to(ROOT)), "bytes": visual.stat().st_size, "catalog_entries": len(visual_entry)}),
        check("finite_finals", finite(finals), "all final values finite"),
        check("new_math_gate_closed", finals["C611"]["headline"]["theory"]["foundational_math_authorized"] is False,
              finals["C611"]["headline"]["empirical_gates"]),
    ]
    passed = sum(r["passed"] for r in checks)
    all_passed = passed == len(checks)
    manifest_paths = [campaign.material_path(), campaign.compiled_path(), campaign.behavior_path(),
                      campaign.qualified_path(), campaign.last_path(), campaign.mean_path(), visual,
                      *(campaign.OUTS[name] / "analysis/final.json" for name in campaign.PHASES), *worker_files.values()]
    manifest = [{"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha(path)}
                for path in manifest_paths if path.exists()]
    result = {
        "phase": 2146, "campaign": "C612", "status": "closed",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "all_checks_passed": all_passed, "checks_passed": passed, "checks_total": len(checks),
        "checks": checks, "manifest": manifest,
        "adjudication": {
            "supported": [
                "Broad Qwen behavior and qualified slices under the frozen machine contract",
                "Model-internal all-coordinate state-conditioned prediction candidates where registered gates pass",
                "Finite program-composition candidates where registered ledgers pass",
                "Generation-level necessity or rescue only for explicitly eligible records",
                "Model-relative cross-model topology only for behavior-qualified workers",
            ],
            "not_supported": [
                "Human-rated naturalness", "cross-model physical coordinate alignment",
                "a unique causal circuit", "global algebra, curvature, holonomy or category structure",
                "a new foundational mathematics",
            ],
            "same_exact_goal_next_stage": False,
            "reason": "The preregistered internal C607-C611 branches are exhausted. The next dependency is external human blind review plus a separately frozen multilingual corpus, not another result-contingent branch of this contract.",
        },
    }
    save(OUT / "analysis/final.json", result)
    print(json.dumps({"phase": 2146, "all_checks_passed": all_passed,
                      "checks_passed": passed, "checks_total": len(checks)}, ensure_ascii=False, indent=2))
    if not all_passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
