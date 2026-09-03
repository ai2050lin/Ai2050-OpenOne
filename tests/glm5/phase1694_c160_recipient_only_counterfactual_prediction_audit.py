#!/usr/bin/env python3
"""Independent split, selection and fresh-result audit for C160."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/glm5/result/phase1694_c160_recipient_only_counterfactual_prediction"
C159 = ROOT / "tests/glm5/result/phase1693_c159_natural_isomorphic_dual_graph_atlas"


def load(base, path):
    return json.loads((base / path).read_text(encoding="utf-8"))


def rows(base, path):
    return [json.loads(line) for line in (base / path).read_text(encoding="utf-8").splitlines()]


def main():
    protocol = load(OUT, "protocol/preregistration.json")
    selection = load(OUT, "protocol/confirmation_selection_lock.json")
    report = load(OUT, "analysis/prediction.json")
    pairs = rows(C159, "analysis/late_half_difference_index.jsonl")
    targets = np.load(C159 / "analysis/late_half_difference.float16.npy", mmap_mode="r")
    pred = np.load(OUT / "analysis/fresh_selected_predictions.float16.npy", mmap_mode="r")
    fresh = [row for row in pairs if row["partition"] == "fresh"]
    qi = 8
    y = np.asarray([targets[row["pair_index"], qi] for row in fresh], np.float32).reshape(256, -1)
    p = np.asarray(pred[:, qi], np.float32).reshape(256, -1)
    cos = np.sum(p * y, axis=1, dtype=np.float64) / np.maximum(np.linalg.norm(p, axis=1) * np.linalg.norm(y, axis=1), 1e-12)
    checks = {
        "contract": load(OUT, "audit/internal_contract_audit.json")["all_checks_passed"],
        "analysis": load(OUT, "audit/internal_analysis_audit.json")["all_checks_passed"],
        "split_counts": all(sum(row["partition"] == part for row in pairs) == 256 for part in ("discovery", "confirmation", "fresh")),
        "winner_locked": selection["selected_candidate"] == report["selected_candidate"] == "pooled_diagonal",
        "ranking": selection["ranking"][0]["candidate"] == "pooled_diagonal",
        "prediction_shape": list(pred.shape) == [256, 11, 6, 2560],
        "q32_cosine": abs(float(np.median(cos)) - report["fresh"]["checkpoint_rows"][qi]["median_cosine"]) < 2e-4,
        "fresh_pass": report["fresh"]["passed"] and all(report["fresh"]["gates"].values()),
        "no_test_donor": "test donor input" in protocol["forbidden"],
        "scope": "not a causal circuit" in protocol["claim_boundary"],
    }
    audit = {"phase": 1694, "campaign": "C160", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "scientific_fresh_passed": report["fresh"]["passed"], "authorization": "memo_then_C161"}
    (OUT / "audit/independent_final_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    if not audit["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
