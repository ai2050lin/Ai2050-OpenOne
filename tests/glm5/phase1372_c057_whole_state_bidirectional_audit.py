#!/usr/bin/env python3
"""Independent artifact audit for Phase1372."""
from __future__ import annotations

import json
import math
import py_compile
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core

OUT = TESTS / "result/phase1372_c057_whole_state_bidirectional"
SCRIPT = TESTS / "phase1372_c057_whole_state_bidirectional.py"


def close(a: float, b: float) -> bool:
    return abs(a - b) <= 1e-8


def main() -> None:
    manifest = core.load(OUT / "protocol/execution_manifest.json")
    summary = core.load(OUT / "analysis/qwen3_bidirectional_summary.json")
    final = core.load(OUT / "analysis/final.json")
    erratum = core.load(OUT / "audit/postprocessing_label_erratum.json")
    records = core.rows(OUT / "raw/qwen3_whole_state_bidirectional.jsonl")
    delta = torch.load(OUT / "raw/family3_source_deltas.pt", map_location="cpu", weights_only=True)
    gate = manifest["gate"]
    metric_ok, gate_ok = True, True
    for path_name, path in manifest["paths"].items():
        rows = [r for r in records if r["path"] == path_name]
        out = summary["path_metrics"][path_name]["output"]
        sc = [r["suff_output_gain"]["correct"] for r in rows]
        sw = [r["suff_output_gain"]["wrong"] for r in rows]
        ss = [r["suff_output_gain"]["status"] for r in rows]
        nc = [r["necessity_output_damage"]["corrupt"] for r in rows]
        ns = [r["necessity_output_damage"]["status"] for r in rows]
        metric_ok &= close(out["suff_correct_gain_median"], statistics.median(sc))
        metric_ok &= close(out["suff_advantage_median"], statistics.median(c - max(w, s) for c, w, s in zip(sc, sw, ss)))
        metric_ok &= close(out["necessity_corrupt_damage_median"], statistics.median(nc))
        metric_ok &= close(out["necessity_over_status_median"], statistics.median(c - s for c, s in zip(nc, ns)))
        recomputed_output = {
            "suff_gain": statistics.median(sc) >= gate["suff_output_gain_median_min"],
            "suff_advantage": statistics.median(c - max(w, s) for c, w, s in zip(sc, sw, ss)) >= gate["suff_output_advantage_median_min"],
            "suff_win": sum(c > max(w, s) for c, w, s in zip(sc, sw, ss)) / len(rows) >= gate["suff_output_win_min"],
            "necessity_damage": statistics.median(nc) >= gate["necessity_output_damage_median_min"],
            "necessity_direction": sum(v > 0 for v in nc) / len(rows) >= gate["necessity_direction_fraction_min"],
            "necessity_over_status": statistics.median(c - s for c, s in zip(nc, ns)) >= gate["necessity_over_status_median_min"],
            "necessity_over_status_win": sum(c > s for c, s in zip(nc, ns)) / len(rows) >= gate["necessity_over_status_win_min"],
            "self": max(max(abs(r["suff_output_gain"]["self"]), abs(r["necessity_output_damage"]["self"])) for r in rows) <= gate["self_output_max_abs_diff"],
        }
        gate_ok &= recomputed_output == summary["path_checks"][path_name]["output"]
        gate_ok &= summary["path_qualified"][path_name] == (
            all(recomputed_output.values()) and
            all(all(v.values()) for v in summary["path_checks"][path_name]["checkpoints"].values()))
    py_compile.compile(str(SCRIPT), doraise=True)
    py_compile.compile(__file__, doraise=True)
    checks = {
        "manifest_case_count": manifest["case_count"] == 288,
        "record_count": len(records) == 576,
        "pair_balance": len({r["pair_id"] for r in records}) == 288,
        "path_balance": all(sum(r["path"] == p for r in records) == 288 for p in manifest["paths"]),
        "finite": all(math.isfinite(v) for r in records for v in
                      list(r["suff_output_gain"].values()) + list(r["necessity_output_damage"].values())),
        "source_delta_shape": tuple(delta["family3_clean_minus_corrupt"].shape) == (288, 2560),
        "source_delta_meta": len(delta["metadata"]) == 288,
        "metrics_recomputed": metric_ok,
        "gates_recomputed": gate_ok,
        "final_consistent": final["path_qualified"] == summary["path_qualified"],
        "authorization_consistent": final["authorization"] ==
            ("run_phase1373_c057_early_path_mediation" if summary["path_qualified"]["family_early"]
             else "close_c057_without_early_bidirectional_qualification"),
        "erratum_numerically_inert": erratum["formal_model_run_repeated"] is False and
                                     erratum["raw_artifacts_rewritten"] is False and
                                     erratum["thresholds_or_routes_changed"] is False,
        "scripts_compile": True,
    }
    audit = {"phase": 1372, "campaign": "C057", "checks": checks,
             "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    core.save(OUT / "audit/independent_final_audit.json", audit)
    print(json.dumps(audit, indent=2))
    if not audit["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
