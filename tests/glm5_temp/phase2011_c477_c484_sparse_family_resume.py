#!/usr/bin/env python3
"""Resume C477-C484 while registering the sparse temporal-family stratum."""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))

import phase2005_c471_c484_program_guard_hypergraph_campaign as c


_material_lookup = c.material_lookup


def compatible_material_lookup() -> tuple[list[dict], dict[str, dict]]:
    rows, _ = _material_lookup()
    rows = [{**row, "cell": "".join(str(bit) for bit in row["bits"])} for row in rows]
    return rows, {row["case_id"]: row for row in rows}


c.material_lookup = compatible_material_lookup


def c477_sparse_aware() -> None:
    out = c.begin("C477", {
        "status": "observation_first_coordinate_atlas_sparse_strata_registered",
        "scope": "family x effect mask x registered checkpoint x role x every coordinate",
        "primary_estimator": "discovery units0-4 and ledger/brief complete programs",
        "registered_missingness_fallback": "if a family has no primary complete program, use all of that family's complete programs for description only and label the source scope",
        "statistics": ["RMS", "positive fraction", "nonzero fraction", "centroid sign agreement"],
        "no_predictive_gate": True,
    }, {"parent": c.final("C476")["all_checks_passed"]})
    states = np.load(c.OUTS["C476"] / "raw/walsh_effects.float16.npy", mmap_mode="r")
    records = c.read_rows(c.OUTS["C476"] / "analysis/effect_index.jsonl")
    centroids = np.lib.format.open_memmap(
        out / "analysis/family_effect_centroids.float16.npy", mode="w+", dtype=np.float16,
        shape=(len(c.FAMILIES), 15, len(c.QPOINTS), 6, c.DIM),
    )
    atlas = []
    fallback_families = set()
    for fi, family in enumerate(c.FAMILIES):
        for effect_mask in c.MASKS:
            primary = [
                row for row in records
                if row["family"] == family and row["effect_mask"] == effect_mask
                and row["unit"] < 5 and row["construction"] in c.CONSTRUCTIONS[:2]
            ]
            subset = primary
            source_scope = "discovery_ledger_brief"
            if not subset:
                subset = [row for row in records if row["family"] == family and row["effect_mask"] == effect_mask]
                source_scope = "fallback_all_complete_descriptive_only"
                fallback_families.add(family)
            indices = [row["effect_index"] for row in subset]
            if not indices:
                raise RuntimeError((family, effect_mask, "no complete program in any stratum"))
            values = np.asarray(states[indices], dtype=np.float32)
            mean = values.mean(0)
            centroids[fi, effect_mask - 1] = mean.astype(np.float16)
            for qi, checkpoint in enumerate(c.QPOINTS):
                for role in range(6):
                    block = values[:, qi, role]
                    centroid = mean[qi, role]
                    atlas.append({
                        "family": family, "effect_mask": effect_mask,
                        "effect_order": c.mask_order(effect_mask), "checkpoint": checkpoint,
                        "role": c.ROLES[role], "samples": len(indices), "source_scope": source_scope,
                        "rms": float(np.sqrt(np.mean(block * block))),
                        "positive_fraction": float(np.mean(block > 0)),
                        "nonzero_fraction": float(np.mean(block != 0)),
                        "centroid_sign_agreement": float(np.mean(np.sign(block) == np.sign(centroid[None, :]))),
                    })
        centroids.flush()
    c.close_mmap(states)
    c.close_mmap(centroids)
    c.write_rows(out / "analysis/condition_atlas.jsonl", atlas)
    headline = {
        "status": "observation_atlas_closed", "ran": True, "rows": len(atlas),
        "fallback_families": sorted(fallback_families),
        "fallback_reason": "temporal_order had one complete 16-cell program, in confirmation only",
        "mean_sign_agreement": float(np.mean([row["centroid_sign_agreement"] for row in atlas])),
        "mean_nonzero_fraction": float(np.mean([row["nonzero_fraction"] for row in atlas])),
        "strict_interpretation": "Fallback rows are sparse descriptive observations and are forbidden from training or confirmation claims.",
    }
    c.close("C477", headline, {
        "rows": len(atlas) == len(c.FAMILIES) * 15 * len(c.QPOINTS) * 6,
        "finite": c.finite(headline), "fallback_registered": fallback_families == {"temporal_order"},
    }, "C478_shared")


def c478_sparse_aware() -> None:
    records = c.read_rows(c.OUTS["C476"] / "analysis/effect_index.jsonl")
    splits = c.effect_splits(records)
    observed_train_families = sorted({row["family"] for row in splits["train"]})
    out = c.begin("C478", {
        "status": "shared_propagation_reconstruction_sparse_strata_registered",
        "train": "behavior-complete discovery ledger/brief programs from the five available training families",
        "registered_train_families": observed_train_families,
        "excluded_sparse_family": "temporal_order; no complete discovery program",
        "evaluation": ["within-family", "whole-family lockbox", "unseen report", "order3-4 composition"],
        "edges": [[q, q + 1] for q in c.Q_STARTS], "controls": ["identity", "training mean"],
    }, {"parent": c.final("C477")["all_checks_passed"]})
    states = np.load(c.OUTS["C476"] / "raw/walsh_effects.float16.npy", mmap_mode="r")
    slope = np.lib.format.open_memmap(out / "analysis/slope.float16.npy", mode="w+", dtype=np.float16, shape=(len(c.Q_STARTS), 6, c.DIM))
    intercept = np.lib.format.open_memmap(out / "analysis/intercept.float16.npy", mode="w+", dtype=np.float16, shape=(len(c.Q_STARTS), 6, c.DIM))
    means = np.lib.format.open_memmap(out / "analysis/mean.float16.npy", mode="w+", dtype=np.float16, shape=(len(c.Q_STARTS), 6, c.DIM))
    acc = {split: {name: c.metric() for name in ("shared", "identity", "mean")} for split in splits if split != "train"}
    for edge in range(len(c.Q_STARTS)):
        for role in range(6):
            xt, yt = c.effect_arrays(states, splits["train"], edge * 2, role)
            a, b = c.fit_diagonal(xt, yt)
            slope[edge, role] = a.astype(np.float16)
            intercept[edge, role] = b.astype(np.float16)
            means[edge, role] = yt.mean(0).astype(np.float16)
            for split, rows in splits.items():
                if split == "train" or not rows:
                    continue
                x, y = c.effect_arrays(states, rows, edge * 2, role)
                c.add_metric(acc[split]["shared"], a * x + b, y)
                c.add_metric(acc[split]["identity"], x, y)
                c.add_metric(acc[split]["mean"], np.broadcast_to(yt.mean(0), y.shape), y)
        slope.flush(); intercept.flush(); means.flush()
        print(f"[C478 sparse-aware] edge={c.Q_STARTS[edge]}", flush=True)
    for value in (states, slope, intercept, means):
        c.close_mmap(value)
    metrics = {split: {name: c.finish_metric(value) for name, value in models.items()} for split, models in acc.items()}
    c.save(out / "analysis/metrics.json", metrics)
    candidate = all(
        metrics[split]["shared"]["nrmse"] < min(metrics[split]["identity"]["nrmse"], metrics[split]["mean"]["nrmse"])
        for split in ("within", "family", "report")
    )
    headline = {
        "status": "shared_reconstruction_closed", "metrics": metrics,
        "shared_candidate": candidate, "registered_train_families": observed_train_families,
        "excluded_sparse_train_family": "temporal_order", "train_records": len(splits["train"]),
        "split_records": {key: len(value) for key, value in splits.items()},
        "strict_interpretation": "This is a family-blind diagonal baseline trained on five families; it makes no six-family training claim.",
    }
    c.close("C478", headline, {
        "finite": c.finite(headline), "train": len(splits["train"]) >= 200,
        "train_family_count": len(observed_train_families) == 5,
        "temporal_excluded": "temporal_order" not in observed_train_families,
    }, "C479_program_guard")


c.RUNNERS["C477"] = c477_sparse_aware
c.RUNNERS["C478"] = c478_sparse_aware

for name in [f"C{value}" for value in range(477, 485)]:
    c.RUNNERS[name]()
