#!/usr/bin/env python3
"""C316: full-coordinate sign/amplitude coupling and interval calibration."""
from __future__ import annotations

import numpy as np

import phase1844_c310_c335_dual_axis_common as common


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    x = left.astype(np.float64).ravel()
    y = right.astype(np.float64).ravel()
    x -= x.mean()
    y -= y.mean()
    return float(np.dot(x, y) / max(np.sqrt(np.dot(x, x) * np.dot(y, y)), 1e-30))


def main() -> None:
    parent = common.core.load(common.OUTS["C315"] / "analysis/final.json")
    checks = {"parent": parent["all_checks_passed"], "all_coordinates": True, "coverage_and_width_jointly_reported": True}
    protocol = {
        "status": "amplitude_phase_coupling_frozen",
        "object": "third/fourth/fifth interaction residual distribution versus sixth interaction residual distribution at every checkpoint, role, and coordinate",
        "bins": "ten equal-count bins frozen from absolute training mean; bins are descriptive and do not discard coordinates",
        "interval": "training minimum to maximum per checkpoint-role-coordinate",
        "gate": "mean-sign agreement>=0.55, absolute-amplitude correlation>=0.10, and interval coverage>=0.50 in at least four families",
        "claim_boundary": "A pass describes repeatable sign/amplitude regimes. Min-max intervals are not calibrated probabilities and wide intervals are penalized only descriptively.",
    }
    out = common.prepare("C316", protocol, checks)
    sixth_states = np.load(common.SIXTH_STATES, mmap_mode="r")
    sixth_index = common.core.rows(common.SIXTH_INDEX)
    atlas = np.zeros((6, 6, 37, 6, 2560), np.float32)
    family_rows = []
    bin_rows = []
    for family_i, family in enumerate(common.FAMILIES):
        train_parts = []
        for state_path, index_path, _material in common.TRAIN_ROLE_SOURCES:
            arrays, _groups = common.factorial_arrays(np.load(state_path, mmap_mode="r"), common.core.rows(index_path), family)
            train_parts.append(arrays["interaction"])
        train = np.concatenate(train_parts, axis=0)
        test, groups = common.factorial_arrays(sixth_states, sixth_index, family)
        truth = test["interaction"]
        train_mean = train.mean(axis=0)
        test_mean = truth.mean(axis=0)
        train_min = train.min(axis=0)
        train_max = train.max(axis=0)
        width = train_max - train_min
        coverage = ((truth >= train_min[None, ...]) & (truth <= train_max[None, ...])).mean(axis=0)
        sign_same = (np.sign(train_mean) == np.sign(test_mean)).astype(np.float32)
        atlas[family_i] = np.stack((train_mean, test_mean, np.abs(train_mean), np.abs(test_mean), sign_same, width), axis=0)
        sign_agreement = float(sign_same.mean())
        amplitude_correlation = correlation(np.abs(train_mean), np.abs(test_mean))
        interval_coverage = float(coverage.mean())
        normalized_width = float(np.mean(width) / max(float(np.mean(np.abs(test_mean))), 1e-12))
        flat_amp = np.abs(train_mean).ravel()
        quantiles = np.quantile(flat_amp, np.linspace(0, 1, 11))
        flat_test = test_mean.ravel()
        flat_train = train_mean.ravel()
        for bin_i in range(10):
            lower, upper = quantiles[bin_i], quantiles[bin_i + 1]
            mask = (flat_amp >= lower) & ((flat_amp <= upper) if bin_i == 9 else (flat_amp < upper))
            bin_rows.append({
                "family": family,
                "amplitude_bin": bin_i,
                "coordinates": int(mask.sum()),
                "lower": float(lower),
                "upper": float(upper),
                "sign_agreement": float(np.mean(np.sign(flat_train[mask]) == np.sign(flat_test[mask]))) if mask.any() else None,
                "mean_train_amplitude": float(np.mean(np.abs(flat_train[mask]))) if mask.any() else None,
                "mean_test_amplitude": float(np.mean(np.abs(flat_test[mask]))) if mask.any() else None,
            })
        passed = sign_agreement >= 0.55 and amplitude_correlation >= 0.10 and interval_coverage >= 0.50
        family_rows.append({"family": family, "training_groups": len(train), "sixth_groups": len(groups), "mean_sign_agreement": sign_agreement, "absolute_amplitude_correlation": amplitude_correlation, "interval_coverage": interval_coverage, "normalized_interval_width": normalized_width, "family_gate_passed": passed})
        print(f"[C316] {family}: sign={sign_agreement:.4f} amp_r={amplitude_correlation:.4f} coverage={interval_coverage:.4f} width={normalized_width:.2f}", flush=True)
    np.save(out / "analysis/amplitude_phase_full_coordinate_atlas.float32.npy", atlas)
    common.core.write_rows(out / "analysis/family_results.jsonl", family_rows)
    common.core.write_rows(out / "analysis/amplitude_bins.jsonl", bin_rows)
    passing = [row["family"] for row in family_rows if row["family_gate_passed"]]
    headline = {"status": "amplitude_phase_adjudicated", "families": family_rows, "families_passing": passing, "breadth_gate_passed": len(passing) >= 4, "strict_interpretation": protocol["claim_boundary"]}
    common.close("C316", headline, {"six_families": len(family_rows) == 6, "sixty_bins": len(bin_rows) == 60, "atlas_shape": list(atlas.shape) == [6, 6, 37, 6, 2560], "finite": bool(np.isfinite(atlas).all())}, "C317_multisource_response_grammar")


if __name__ == "__main__":
    main()
