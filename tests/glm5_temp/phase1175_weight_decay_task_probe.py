#!/usr/bin/env python3
"""Zero-training material probe for the Phase1175 causal fork."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))

import phase1169_natural_training_trajectory_bifurcation as base  # noqa: E402
import phase1172_cross_quotient_event_time_prediction as p1172  # noqa: E402
import phase1174_training_inferred_relation_event_prediction as p1174  # noqa: E402


P = 61
OUT = ROOT / "tests/glm5_temp/phase1175_weight_decay_task_probe.json"


def candidate_functions():
    return {
        "add_power1": lambda a, b: (a + b) % P,
        "add_power6": lambda a, b: (a + pow(b, 6, P)) % P,
        "add_power20": lambda a, b: (a + pow(b, 20, P)) % P,
        "add_poly2": lambda a, b: (a + b * b + b) % P,
        "add_poly3": lambda a, b: (a + pow(b, 3, P) + 2 * b) % P,
        "add_poly4": lambda a, b: (a + pow(b, 4, P) + 3 * b) % P,
        "add_poly5": lambda a, b: (a + pow(b, 5, P) + 2 * b * b) % P,
        "add_factor3": lambda a, b: (a + b * (b - 1) * (b - 2)) % P,
    }


def table(function) -> np.ndarray:
    return np.asarray([[function(a, b) for b in range(P)] for a in range(P)], dtype=np.int64)


def signature(values: np.ndarray) -> dict:
    payload = p1172.quotient_invariant_payload(values)
    return {
        "digest": base.digest(payload),
        "table_digest": base.digest(values.tolist()),
        "distinct_rows": payload["distinct_row_count"],
        "distinct_columns": payload["distinct_column_count"],
        "row_output_range": [min(payload["row_distinct_output_counts"]), max(payload["row_distinct_output_counts"])],
        "column_output_range": [min(payload["column_distinct_output_counts"]), max(payload["column_distinct_output_counts"])],
    }


def data_for(values: np.ndarray, seed: int) -> dict:
    pairs = np.asarray([(a, b) for a in range(P) for b in range(P)], dtype=np.int64)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(pairs))
    mask = np.zeros(len(pairs), dtype=bool)
    mask[order[: int(round(len(pairs) * p1174.TRAIN_FRACTION))]] = True
    backgrounds = rng.permutation(P)
    import torch

    x = torch.tensor(pairs, dtype=torch.long)
    y = torch.tensor(values.reshape(-1), dtype=torch.long)
    mask_t = torch.tensor(mask, dtype=torch.bool)
    return {
        "train_x": x[mask_t],
        "train_y": y[mask_t],
        "train_mask": mask.reshape(P, P),
        "contexts": {
            "key": backgrounds[: p1174.KEY_CONTEXT_COUNT],
            "fit": backgrounds[p1174.KEY_CONTEXT_COUNT : p1174.KEY_CONTEXT_COUNT + p1174.FIT_CONTEXT_COUNT],
            "test": backgrounds[p1174.KEY_CONTEXT_COUNT + p1174.FIT_CONTEXT_COUNT :],
        },
    }


def main() -> None:
    old = {
        f"1172:{task.name}": p1172.quotient_signature(task.name)["digest"]
        for task in p1172.TASK_SPECS
    }
    old.update({
        f"1174:{task.name}": p1174.quotient_signature(task.name)["digest"]
        for task in p1174.TASK_SPECS
    })
    old_digests = set(old.values())
    rows = []
    for index, (name, function) in enumerate(candidate_functions().items()):
        values = table(function)
        sig = signature(values)
        keys = [p1174.infer_relation_key(data_for(values, 11_750_000 + index * 1009 + rep * 97)) for rep in range(8)]
        rows.append({
            "name": name,
            "signature": sig,
            "old_collision": sig["digest"] in old_digests,
            "eligible_counts": [key["eligible_count"] for key in keys],
            "all_keys_identified": all(key["eligible_count"] == 3 for key in keys),
        })
    payload = {
        "scope": "zero-training material probe; candidate generation only",
        "old_signature_count": len(old_digests),
        "rows": rows,
        "new_signature_unique": len({row["signature"]["digest"] for row in rows}) == len(rows),
        "eligible_no_collision_names": [
            row["name"] for row in rows if row["all_keys_identified"] and not row["old_collision"]
        ],
    }
    payload["digest"] = base.digest(payload)
    base.write_json(OUT, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
