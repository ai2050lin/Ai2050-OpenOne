#!/usr/bin/env python3
"""Post-hoc descriptive factor decomposition for the failed Phase1248 camera.

This script never reselects an event and never upgrades evidence.  It asks only
which registered interface factors account for the frozen selected-event error.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests/glm5"
sys.path.insert(0, str(TEST_ROOT))
import phase1248_c002_qwen_self_response_atlas as main  # noqa: E402

OUT = main.OUT_ROOT / "analysis/descriptive_failure_atlas.json"


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def fit_group(arrays: Any, rows: list[dict[str, Any]], event: int, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for donor in main.DONORS:
        for alpha in (0.25, 0.5):
            xs.append(main.feature_delta(arrays, donor, event, indices, alpha))
            ys.append(main.response_values(arrays, donor, event, indices, alpha))
    return main.ridge_fit(np.concatenate(xs), np.concatenate(ys))


def grouped_camera(
    arrays: Any,
    rows: list[dict[str, Any]],
    event: int,
    fields: tuple[str, ...],
) -> dict[str, Any]:
    discovery = np.asarray([i for i, row in enumerate(rows) if row["partition"] == "discovery"])
    confirmation = np.asarray([i for i, row in enumerate(rows) if row["partition"] == "confirmation"])
    groups = sorted({tuple(str(rows[i][field]) for field in fields) for i in discovery})
    all_actual, all_predicted = [], []
    detail: dict[str, Any] = {}
    parameter_count = 0
    for group in groups:
        train = np.asarray([i for i in discovery if tuple(str(rows[i][field]) for field in fields) == group])
        test = np.asarray([i for i in confirmation if tuple(str(rows[i][field]) for field in fields) == group])
        camera = fit_group(arrays, rows, event, train)
        x = main.feature_delta(arrays, "target", event, test, 1.0)
        actual = main.response_values(arrays, "target", event, test, 1.0)
        predicted = main.predict(x, camera)
        key = "|".join(group)
        detail[key] = main.metrics(actual, predicted)
        detail[key]["discovery_rows"] = int(len(train))
        parameter_count += int(camera[0].size + camera[1].size)
        all_actual.append(actual)
        all_predicted.append(predicted)
    aggregate = main.metrics(np.concatenate(all_actual), np.concatenate(all_predicted))
    return {
        "group_fields": list(fields),
        "group_count": len(groups),
        "camera_parameter_count": parameter_count,
        "aggregate": aggregate,
        "groups": detail,
    }


def cross_representation(arrays: Any, rows: list[dict[str, Any]], event: int) -> dict[str, Any]:
    discovery = np.asarray([i for i, row in enumerate(rows) if row["partition"] == "discovery"])
    confirmation = np.asarray([i for i, row in enumerate(rows) if row["partition"] == "confirmation"])
    output: dict[str, Any] = {}
    for source in main.REPRESENTATIONS:
        train = np.asarray([i for i in discovery if rows[i]["representation"] == source])
        camera = fit_group(arrays, rows, event, train)
        for target in main.REPRESENTATIONS:
            test = np.asarray([i for i in confirmation if rows[i]["representation"] == target])
            actual = main.response_values(arrays, "target", event, test, 1.0)
            predicted = main.predict(main.feature_delta(arrays, "target", event, test, 1.0), camera)
            output[f"{source}_to_{target}"] = main.metrics(actual, predicted)
    return output


def main_cli() -> None:
    atlas = main.read_json(main.ATLAS_PATH)
    rows = main.read_jsonl(main.TOKEN_PATH)
    arrays = np.load(main.ARRAY_PATH)
    selected_id = atlas["selected_event"]["event_id"]
    event = next(i for i, row in enumerate(main.EVENTS) if row["event_id"] == selected_id)
    variants = {
        "global_frozen": {
            "group_fields": [],
            "group_count": 1,
            "camera_parameter_count": main.PROJECTION_DIM * len(main.LABELS) + len(main.LABELS),
            "aggregate": atlas["confirmation"]["camera"],
        },
        "by_representation": grouped_camera(arrays, rows, event, ("representation",)),
        "by_mapping": grouped_camera(arrays, rows, event, ("mapping",)),
        "by_interface": grouped_camera(arrays, rows, event, ("interface",)),
        "by_representation_mapping": grouped_camera(arrays, rows, event, ("representation", "mapping")),
        "by_full_condition": grouped_camera(arrays, rows, event, ("representation", "mapping", "interface")),
    }
    payload = {
        "phase": main.PHASE,
        "schema_version": "phase1248.descriptive_failure_atlas.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "posthoc_descriptive_only",
        "source_atlas_digest": atlas["atlas_digest"],
        "selected_event_frozen": selected_id,
        "event_reselection_performed": False,
        "variants": variants,
        "cross_representation": cross_representation(arrays, rows, event),
        "interpretation_boundary": [
            "Conditional cameras were not preregistered and cannot rescue Phase1248.",
            "Extra maps increase parameter count and may overfit discovery strata.",
            "The atlas may motivate a new mathematical object only after a new contract and all-new confirmation data.",
        ],
        "phase1249_authorized": False,
    }
    payload["atlas_digest"] = digest(payload)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(canonical({name: value["aggregate"] for name, value in variants.items()}))


if __name__ == "__main__":
    main_cli()
