#!/usr/bin/env python3
"""Rebuild omitted C615 per-cell metrics without changing any empirical gate."""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase2147_c613_c620_conditional_gear_campaign as c

OUT = TESTS / "result/phase2157_c623_guard_metric_artifact_recovery"

def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")

def main():
    save(OUT / "protocol/preregistration.json", {"phase": 2157, "campaign": "C623",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "object": "exact arithmetic recovery of omitted C615 per-cell metric artifact",
        "constraints": "same states, index, pairs, models, QPOINTS, margin and gates; no result-dependent change"})
    states = np.load(c.states_path(), mmap_mode="r"); index = c.read_rows(c.index_path())
    pairs = c.pair_records(index, dual_only=True); grouped = defaultdict(list)
    for pair in pairs: grouped[pair["operation"]].append(pair)
    results = {}
    for operation, values in sorted(grouped.items()):
        train = [x for x in values if x["partition"] == "discovery"]
        test = [x for x in values if x["partition"] == "lockbox"]
        if len(train) < 3 or len(test) < 2: continue
        wrong_pool = [x for op, vals in grouped.items() if op != operation for x in vals if x["partition"] == "discovery"]
        for q in c.QPOINTS:
            htr = np.stack([c.role_state(states, x["left"], q) for x in train])
            ytr = np.stack([c.role_state(states, x["right"], q) - c.role_state(states, x["left"], q) for x in train])
            hte = np.stack([c.role_state(states, x["left"], q) for x in test])
            truth = np.stack([c.role_state(states, x["right"], q) - c.role_state(states, x["left"], q) for x in test])
            mean = np.mean(ytr, axis=0)
            centered = htr - np.mean(htr, axis=0); ycenter = ytr - np.mean(ytr, axis=0)
            slope = np.sum(centered * ycenter, axis=0) / (np.sum(centered * centered, axis=0) + 1e-6)
            diagonal = np.mean(ytr, axis=0) + (hte - np.mean(htr, axis=0)) * slope
            pos_mean = np.sum(ytr * (htr >= 0), axis=0) / (np.sum(htr >= 0, axis=0) + 1e-6)
            neg_mean = np.sum(ytr * (htr < 0), axis=0) / (np.sum(htr < 0, axis=0) + 1e-6)
            sign_guard = np.where(hte >= 0, pos_mean, neg_mean)
            nearest, history = [], []
            for h in hte:
                d = np.mean((htr - h[None]) ** 2, axis=(1, 2)); nearest.append(ytr[int(np.argmin(d))])
            qprev = max(0, q - 1)
            htr_prev = np.stack([c.role_state(states, x["left"], qprev) for x in train])
            hte_prev = np.stack([c.role_state(states, x["left"], qprev) for x in test])
            for h, hp in zip(hte, hte_prev):
                d = np.mean((htr - h[None]) ** 2, axis=(1, 2)) + np.mean((htr_prev - hp[None]) ** 2, axis=(1, 2))
                history.append(ytr[int(np.argmin(d))])
            wrong = np.mean([c.role_state(states, x["right"], q) - c.role_state(states, x["left"], q)
                             for x in wrong_pool], axis=0) if wrong_pool else -mean
            preds = {"identity": np.zeros_like(truth), "mean": np.broadcast_to(mean, truth.shape),
                "diagonal": diagonal, "sign_guard": sign_guard, "nearest": np.stack(nearest),
                "history_nearest": np.stack(history), "wrong_operation": np.broadcast_to(wrong, truth.shape)}
            metrics = {name: c.metric(pred, truth) for name, pred in preds.items()}
            best = min(("diagonal", "sign_guard", "nearest", "history_nearest"), key=lambda x: metrics[x]["nrmse"])
            gate = all(metrics[best]["nrmse"] <= metrics[x]["nrmse"] - c.CONTROL_MARGIN for x in ("identity", "mean", "wrong_operation"))
            results[f"{operation}|q{q}"] = {"train": len(train), "test": len(test), "models": metrics,
                "best_conditional": best, "gate": gate}
    c.c607.passport.close_mmap(states); del states
    recovered = [k for k, v in results.items() if v["gate"]]
    frozen = c.final("C615")["headline"]["guard_candidates"]
    save(OUT / "analysis/recovered_guard_metrics.json", results)
    counts = defaultdict(int)
    for v in results.values(): counts[v["best_conditional"]] += 1
    headline = {"status": "guard_metric_artifact_recovered", "metric_cells": len(results),
        "recovered_candidates": len(recovered), "frozen_candidates": len(frozen),
        "candidate_keys_exact": recovered == frozen, "best_conditional_counts": dict(counts),
        "strict_interpretation": "Artifact recovery reproduces the frozen result; it is not a new empirical test."}
    result = {"phase": 2157, "campaign": "C623", "status": "closed",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "all_checks_passed": len(results) == 72 and recovered == frozen,
        "headline": headline, "checks": {"metric_cells": len(results) == 72, "candidate_keys_exact": recovered == frozen,
        "finite": all(math.isfinite(m["nrmse"]) for v in results.values() for m in v["models"].values())},
        "next_authorization": "C624_recovery_audit"}
    result["all_checks_passed"] = all(result["checks"].values()); save(OUT / "analysis/final.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__": main()
