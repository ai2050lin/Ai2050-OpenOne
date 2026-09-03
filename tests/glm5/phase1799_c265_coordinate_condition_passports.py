#!/usr/bin/env python3
"""C265: learn and prospectively test transparent per-coordinate state passports."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import phase1797_c263_c272_state_operator_common as common

core, OUT = common.core, common.OUTS["C265"]
TRAIN = common.prior.OUTS["C248"]
TEST = common.OUTS["C264"]
FAMILIES = common.FAMILIES + ("nested_attitude",)


def extract_training_role_states(path: Path) -> np.memmap:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return np.load(path, mmap_mode="r")
    fields = np.load(TRAIN / "raw/full_fields.float16.npy", mmap_mode="r")
    index = core.rows(TRAIN / "raw/hidden_index.jsonl")
    out = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=(768, 37, 6, 2560))
    for i, row in enumerate(index):
        for q in range(37):
            for ri, role in enumerate(common.ROLES):
                out[i, q, ri] = np.asarray(fields[i, q, row["role_positions"][role]], np.float32).mean(axis=0).astype(np.float16)
        if i % 64 == 0 or i == len(index) - 1:
            out.flush(); print(f"[C265] training role states {i + 1}/768", flush=True)
    out.flush()
    return np.load(path, mmap_mode="r")


def pair_ids(index: list[dict], family: str, panel: str):
    specs = common.pair_specs(index, family, "factor_a", panel)
    return np.asarray([x[0] for x in specs], int), np.asarray([x[1] for x in specs], int), [x[2] for x in specs]


def metrics(pred: np.ndarray, truth: np.ndarray) -> tuple[int, int, int, int, int]:
    union = (pred != 0) | (truth != 0)
    exact = (pred == truth) & union
    return int(exact.sum()), int(union.sum()), int(((pred == truth) & (pred != 0)).sum()), int((pred != 0).sum()), int((truth != 0).sum())


def main() -> None:
    if OUT.exists():
        raise RuntimeError(OUT)
    parent = core.load(TEST / "analysis/final.json")
    gates = core.load(common.OUTS["C263"] / "protocol/preregistration.json")["gates"]
    checks = {"parent": parent["all_checks_passed"], "train_archive": (TRAIN / "raw/full_fields.float16.npy").exists(), "all_coordinates": True, "no_topk": True, "frozen_support": gates["passport_support_min"] == 4, "frozen_agreement": gates["passport_agreement_min"] == 0.70}
    if not all(checks.values()):
        raise RuntimeError(checks)
    OUT.mkdir(parents=True)
    (OUT / "raw").mkdir(); (OUT / "analysis").mkdir(); (OUT / "audit").mkdir()
    protocol = {
        "phase": 1799, "campaign": "C265", "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "passport_rule_frozen",
        "training": "C248 third material", "prospective_test": "C264 fourth material", "families": list(FAMILIES),
        "key": "current response sign (-/+) x current baseline below/above the per-coordinate training median",
        "prediction": "dominant next-checkpoint event in {-1,0,+1}; support>=4 and agreement>=0.70",
        "magnitude_interval": "minimum and maximum observed absolute next response among sign-matching training examples",
        "baselines": ["same-coordinate persistence", "fixed tri-material core where available"],
        "primary_gate": "median family signed-Jaccard must exceed both available baselines by 0.01",
        "claim_boundary": "This is a transparent observational transition rule. It does not prove a unique causal edge, and span means remain role-level observations.",
        "producer_sha256": core.sha(Path(__file__)), "authorization": "run_C266_C268_regardless_of_gate; C269_uses_only_registered_rules",
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    train_states = extract_training_role_states(OUT / "raw/training_role_states.float16.npy")
    test_states = np.load(TEST / "raw/role_states.float16.npy", mmap_mode="r")
    train_index = core.rows(TRAIN / "raw/hidden_index.jsonl")
    test_index = core.rows(TEST / "raw/hidden_index.jsonl")
    thresholds = np.asarray(core.load(common.prior.OLD["C236"] / "protocol/frozen_event_thresholds.json")["thresholds"], np.float32)
    tri = np.load(common.prior.OUTS["C249"] / "analysis/tri_material_core.int8.npy", mmap_mode="r")
    shape = (len(FAMILIES), 36, 6, 4, 2560)
    pred_map = np.lib.format.open_memmap(OUT / "analysis/passport_pred_sign.int8.npy", mode="w+", dtype=np.int8, shape=shape)
    support_map = np.lib.format.open_memmap(OUT / "analysis/passport_support.uint8.npy", mode="w+", dtype=np.uint8, shape=shape)
    agreement_map = np.lib.format.open_memmap(OUT / "analysis/passport_agreement.float16.npy", mode="w+", dtype=np.float16, shape=shape)
    mag_lo = np.lib.format.open_memmap(OUT / "analysis/passport_magnitude_lo.float16.npy", mode="w+", dtype=np.float16, shape=shape)
    mag_hi = np.lib.format.open_memmap(OUT / "analysis/passport_magnitude_hi.float16.npy", mode="w+", dtype=np.float16, shape=shape)
    baseline_median = np.lib.format.open_memmap(OUT / "analysis/passport_baseline_median.float16.npy", mode="w+", dtype=np.float16, shape=(len(FAMILIES), 36, 6, 2560))
    summary_rows, selection_rows = [], []
    for fi, family in enumerate(FAMILIES):
        panel = "nested_composition" if family == "nested_attitude" else "core"
        tl, tr, _ = pair_ids(train_index, family, panel)
        vl, vr, vmeta = pair_ids(test_index, family, panel)
        if len(tl) < 4 or len(vl) < 1:
            raise RuntimeError((family, len(tl), len(vl)))
        totals = {name: np.zeros(5, np.int64) for name in ("state", "persistence", "fixed")}
        readiness = np.zeros((len(vl), 16), np.float64)
        for q in range(36):
            for ri in range(6):
                src = np.asarray(train_states[tl, q, ri], np.float32)
                delta = np.asarray(train_states[tr, q, ri], np.float32) - src
                nxt = np.asarray(train_states[tr, q + 1, ri], np.float32) - np.asarray(train_states[tl, q + 1, ri], np.float32)
                med = np.median(src, axis=0).astype(np.float32)
                baseline_median[fi, q, ri] = med.astype(np.float16)
                current = np.where(delta > thresholds[q], 1, np.where(delta < -thresholds[q], -1, 0)).astype(np.int8)
                outcome = np.where(nxt > thresholds[q + 1], 1, np.where(nxt < -thresholds[q + 1], -1, 0)).astype(np.int8)
                high = src >= med[None, :]
                keys = np.where(current < 0, high.astype(np.int8), np.where(current > 0, 2 + high.astype(np.int8), -1))
                for key in range(4):
                    member = keys == key
                    support = member.sum(axis=0)
                    counts = np.stack([((outcome == sign) & member).sum(axis=0) for sign in (-1, 0, 1)], axis=0)
                    choice = counts.argmax(axis=0)
                    pred = (choice - 1).astype(np.int8)
                    agree = counts.max(axis=0) / np.maximum(support, 1)
                    valid = (support >= gates["passport_support_min"]) & (agree >= gates["passport_agreement_min"])
                    pred = np.where(valid, pred, 0).astype(np.int8)
                    chosen = member & (outcome == pred[None, :]) & (pred[None, :] != 0)
                    magnitude = np.abs(nxt)
                    lo = np.where(chosen, magnitude, np.inf).min(axis=0)
                    hi = np.where(chosen, magnitude, -np.inf).max(axis=0)
                    pred_map[fi, q, ri, key] = pred
                    support_map[fi, q, ri, key] = np.minimum(support, 255).astype(np.uint8)
                    agreement_map[fi, q, ri, key] = agree.astype(np.float16)
                    mag_lo[fi, q, ri, key] = np.where(np.isfinite(lo), lo, 0).astype(np.float16)
                    mag_hi[fi, q, ri, key] = np.where(np.isfinite(hi), hi, 0).astype(np.float16)

                vsrc = np.asarray(test_states[vl, q, ri], np.float32)
                vdelta = np.asarray(test_states[vr, q, ri], np.float32) - vsrc
                vnxt = np.asarray(test_states[vr, q + 1, ri], np.float32) - np.asarray(test_states[vl, q + 1, ri], np.float32)
                vcur = np.where(vdelta > thresholds[q], 1, np.where(vdelta < -thresholds[q], -1, 0)).astype(np.int8)
                truth = np.where(vnxt > thresholds[q + 1], 1, np.where(vnxt < -thresholds[q + 1], -1, 0)).astype(np.int8)
                vhigh = vsrc >= med[None, :]
                vkeys = np.where(vcur < 0, vhigh.astype(np.int8), np.where(vcur > 0, 2 + vhigh.astype(np.int8), -1))
                predicted = np.zeros_like(vcur)
                for key in range(4):
                    predicted[vkeys == key] = np.broadcast_to(np.asarray(pred_map[fi, q, ri, key]), predicted.shape)[vkeys == key]
                persistence = vcur
                fixed = np.broadcast_to(np.asarray(tri[fi, 0, q + 1, ri]), truth.shape) if family in common.FAMILIES else np.zeros_like(truth)
                for name, value in (("state", predicted), ("persistence", persistence), ("fixed", fixed)):
                    totals[name] += np.asarray(metrics(value, truth), np.int64)
                if ri == common.ROLES.index("relation") and 1 <= q <= 16:
                    readiness[:, q - 1] = np.mean(predicted != 0, axis=1)
        pred_map.flush(); support_map.flush(); agreement_map.flush(); mag_lo.flush(); mag_hi.flush(); baseline_median.flush()
        values = {}
        for name, total in totals.items():
            exact, union, active_correct, pred_active, truth_active = total
            values[name] = {"signed_jaccard": float(exact / max(union, 1)), "signed_precision": float(active_correct / max(pred_active, 1)), "signed_recall": float(active_correct / max(truth_active, 1)), "union": int(union)}
        available = [values["persistence"]["signed_jaccard"]] + ([values["fixed"]["signed_jaccard"]] if family in common.FAMILIES else [])
        margin = values["state"]["signed_jaccard"] - max(available)
        summary_rows.append({"family": family, "training_pairs": len(tl), "test_pairs": len(vl), "metrics": values, "state_minus_best_baseline": margin, "family_gate_passed": margin >= gates["prediction_margin_min"]})
        for i, meta in enumerate(vmeta):
            qstar = int(np.argmax(readiness[i]) + 1)
            selection_rows.append({"family": family, **meta, "selected_checkpoint": qstar, "readiness": float(readiness[i, qstar - 1]), "readiness_curve_q1_q16": readiness[i].tolist()})
        print(f"[C265] {family}: state J={values['state']['signed_jaccard']:.4f}, margin={margin:.4f}", flush=True)
    core.write_rows(OUT / "analysis/family_prediction_summary.jsonl", summary_rows)
    core.write_rows(OUT / "analysis/state_selected_checkpoints.jsonl", selection_rows)
    passed = sum(r["family_gate_passed"] for r in summary_rows)
    report = {"phase": 1799, "campaign": "C265", "status": "prospective_passport_adjudicated", "family_results": summary_rows, "families_passing": passed, "families_total": len(summary_rows), "broad_prediction_gate_passed": passed >= 4, "strict_interpretation": "The passport is a state-indexed same-coordinate predictor. Passing would identify a reusable observational rule, not a unique natural transmission edge; failing leaves the full coordinate archive intact for other registered analyses.", "next_authorization": "C266_rolling_prediction_and_C267_C268_composition_regardless; C269_only_registered_rules"}
    core.save(OUT / "analysis/summary.json", report)
    analysis_checks = {"families": len(summary_rows) == 6, "passport_shape": list(pred_map.shape) == [6, 36, 6, 4, 2560], "selection_rows": len(selection_rows) >= 6 * 24, "finite": bool(np.isfinite([r["state_minus_best_baseline"] for r in summary_rows]).all())}
    core.save(OUT / "audit/internal_analysis_audit.json", {"checks": analysis_checks, "all_checks_passed": all(analysis_checks.values())})
    final_checks = {"contract": all(checks.values()), "analysis": all(analysis_checks.values()), "producer_hash": core.sha(Path(__file__)) == protocol["producer_sha256"]}
    final = {"phase": 1799, "campaign": "C265", "status": "closed", "checks": final_checks, "all_checks_passed": all(final_checks.values()), "headline": report, "next_authorization": report["next_authorization"]}
    core.save(OUT / "analysis/final.json", final)
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
