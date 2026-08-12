#!/usr/bin/env python3
"""Post-decision exact audit of the Phase1170 finite-fraction instrument.

The frozen evaluator summarized a boolean tensor with float32 ``mean``.  For
all-finite tensors of some non-power-of-two sizes CUDA can return the adjacent
float below one.  This audit reloads every sealed checkpoint and uses boolean
``all`` plus integer counts.  It cannot change the preregistered breadth result
or authorize continuation; it only separates an instrument representation bug
from actual non-finite model values.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))

import phase1170_natural_rule_selection_breadth_confirmation as phase  # noqa: E402


SCRIPT = Path(__file__).resolve()


@torch.inference_mode()
def main() -> None:
    root = phase.OUT_ROOT
    seal = phase.base.read_json(root / "runs/training/seal.json")
    holdout_rows = phase.base.read_jsonl(root / "runs/holdout/holdout_metrics.jsonl")
    final = phase.base.read_json(root / "analysis/final.json")
    device = torch.device("cuda")
    rows = []
    for stored in holdout_rows:
        checkpoint_path = root / "runs/training/checkpoints" / f"{stored['checkpoint_id']}.pt"
        if phase.base.sha256_file(checkpoint_path) != seal["checkpoint_hashes"][stored["checkpoint_id"]]:
            raise RuntimeError(f"checkpoint hash mismatch: {stored['checkpoint_id']}")
        data = phase.base.make_data(stored["modulus"], stored["seed"] + 17)
        model = phase.load_checkpoint(checkpoint_path, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(data["holdout_x"].to(device)).float()
        finite = torch.isfinite(logits)
        exact_count = int(finite.sum(dtype=torch.int64).item())
        total_count = finite.numel()
        exact_all = bool(finite.all().item())
        exact_fraction = exact_count / total_count
        rows.append({
            "checkpoint_id": stored["checkpoint_id"],
            "task_name": stored["task_name"],
            "modulus": stored["modulus"],
            "replicate": stored["replicate"],
            "step": stored["step"],
            "stored_float32_mean": stored["holdout"]["finite_fraction"],
            "exact_finite_count": exact_count,
            "total_logit_count": total_count,
            "exact_fraction": exact_fraction,
            "exact_all_finite": exact_all,
        })
        del model, logits, finite
    stored_below_one = [row for row in rows if row["stored_float32_mean"] < 1.0]
    exact_failures = [row for row in rows if not row["exact_all_finite"]]
    report = {
        "phase": phase.PHASE,
        "audited_at_utc": phase.base.utc_now(),
        "script_sha256": phase.base.sha256_file(SCRIPT),
        "seal_digest": seal["seal_digest"],
        "checkpoint_count": len(rows),
        "stored_below_one_row_count": len(stored_below_one),
        "stored_below_one_moduli": sorted({row["modulus"] for row in stored_below_one}),
        "minimum_stored_float32_mean": min(row["stored_float32_mean"] for row in rows),
        "exact_nonfinite_row_count": len(exact_failures),
        "exact_all_checkpoints_finite": not exact_failures,
        "affected_rows_are_rounding_only": bool(stored_below_one) and not exact_failures,
        "primary_endpoint_remains_failed": final["decision"]["primary_endpoint_pass"] is False,
        "continuation_remains_denied": final["decision"]["auto_continue"] is False,
        "claim_scope": "Post-decision instrument audit only. It does not amend thresholds, trajectory labels, the primary endpoint, or continuation authorization.",
        "affected_rows": stored_below_one,
    }
    report["overall_pass"] = (
        report["exact_all_checkpoints_finite"]
        and report["affected_rows_are_rounding_only"]
        and report["primary_endpoint_remains_failed"]
        and report["continuation_remains_denied"]
    )
    report["report_digest"] = phase.base.digest(report)
    phase.base.write_json(root / "audit/finite_fraction_exact_recompute.json", report)
    print(json.dumps({
        "overall_pass": report["overall_pass"],
        "stored_below_one_row_count": report["stored_below_one_row_count"],
        "exact_nonfinite_row_count": report["exact_nonfinite_row_count"],
        "report_digest": report["report_digest"],
    }))
    if not report["overall_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
