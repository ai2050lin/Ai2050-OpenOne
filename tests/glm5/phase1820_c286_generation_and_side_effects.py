#!/usr/bin/env python3
"""C286: test clean free generation; register causal side effects as no-test."""
from __future__ import annotations

import gc
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

import phase1811_c277_c289_joint_response_common as common

core, OUT = common.core, common.OUTS["C286"]
C277 = common.OUTS["C277"]
C285 = common.OUTS["C285"]


def normalize(text: str) -> str:
    return " ".join("".join(ch.lower() if ch.isalnum() else " " for ch in text).split())


@torch.inference_mode()
def main() -> None:
    if OUT.exists(): raise RuntimeError(OUT)
    parent = core.load(C285 / "analysis/final.json")
    rows = [row for row in core.rows(C277 / "compiled/qwen3.jsonl") if row["order"] == 1]
    checks = {"parent": parent["all_checks_passed"], "rows": len(rows) == 384, "all_families": {r["family"] for r in rows} == set(common.FAMILIES), "causal_side_effect_no_test_registered": not parent["headline"]["broad_gate_passed"], "cuda": torch.cuda.is_available()}
    if not all(checks.values()): raise RuntimeError(checks)
    OUT.mkdir(parents=True)
    for subdir in ("analysis", "audit", "protocol", "raw"): (OUT / subdir).mkdir()
    protocol = {
        "phase": 1820, "campaign": "C286", "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "generation_contract_frozen",
        "panel": "384 fifth-material semantic cases after removing candidate-order duplicates", "generation": "greedy, max_new_tokens=6",
        "success": "normalized output begins with the registered answer word or phrase",
        "family_gate": "success>=0.80", "broad_gate": "all six families pass",
        "side_effect_policy": "C285 had no eligible causal object; intervention side-effect and bidirectionality panels are registered no-test rather than simulated.",
        "claim_boundary": "This is constrained answer generation on controlled English, not unrestricted discourse or a causal rescue result.",
        "producer_sha256": core.sha(Path(__file__)), "authorization": "C287_cross_model_joint_state_capture",
    }
    core.save(OUT / "protocol/preregistration.json", protocol); core.save(OUT / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    model = None; results = []; started = time.time()
    try:
        model, tokenizer, device, placement = common.model_base.load_bf16("qwen3")
        quant = common.model_base.quantization_audit(model)
        for i, row in enumerate(rows):
            ids = torch.tensor([row["free_prompt_ids"]], dtype=torch.long, device=device)
            output = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=6, do_sample=False, use_cache=True)
            text = tokenizer.decode(output[0, ids.shape[1]:].tolist(), skip_special_tokens=True).strip()
            expected = normalize(row["correct_answer"]); observed = normalize(text)
            success = observed.startswith(expected)
            results.append({"case_id": row["case_id"], "family": row["family"], "surface": row["surface"], "unit": row["unit"], "factor_a": row["factor_a"], "factor_b": row["factor_b"], "expected": row["correct_answer"], "generated": text, "success": success})
            if i % 32 == 0 or i + 1 == len(rows): print(f"[C286] {i + 1}/{len(rows)}", flush=True)
        core.write_rows(OUT / "raw/generation_rows.jsonl", results)
        families = []
        for family in common.FAMILIES:
            selected = [r for r in results if r["family"] == family]
            rate = float(np.mean([r["success"] for r in selected]))
            families.append({"family": family, "support": len(selected), "success_rate": rate, "gate_passed": rate >= 0.80, "failed_examples": [r for r in selected if not r["success"]][:8]})
        broad = all(r["gate_passed"] for r in families)
        report = {
            "phase": 1820, "campaign": "C286", "status": "clean_generation_adjudicated", "families": families,
            "overall_success_rate": float(np.mean([r["success"] for r in results])), "generation_gate_passed": broad,
            "causal_generation_side_effect_status": "no_test_C285_local_eligibility_failed", "placement": placement, "quantization": quant, "elapsed_seconds": time.time() - started,
            "strict_interpretation": protocol["claim_boundary"], "next_authorization": "C287_cross_model_joint_state_capture",
        }
        core.save(OUT / "analysis/summary.json", report)
        ach = {"rows": len(results) == 384, "families": len(families) == 6, "support": sum(r["support"] for r in families) == 384, "finite": bool(np.isfinite([r["success_rate"] for r in families]).all())}
        core.save(OUT / "audit/internal_analysis_audit.json", {"checks": ach, "all_checks_passed": all(ach.values())})
        fch = {"contract": all(checks.values()), "analysis": all(ach.values()), "producer_hash": core.sha(Path(__file__)) == protocol["producer_sha256"]}
        final = {"phase": 1820, "campaign": "C286", "status": "closed", "checks": fch, "all_checks_passed": all(fch.values()), "headline": report, "next_authorization": report["next_authorization"]}; core.save(OUT / "analysis/final.json", final); print(json.dumps(final, ensure_ascii=False, indent=2))
    finally:
        common.model_base.release(model); gc.collect()


if __name__ == "__main__": main()

