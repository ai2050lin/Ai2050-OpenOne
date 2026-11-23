"""Copy one existing Phase 2933 row into the offline workbench preview."""

import base64
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent / "result" / "b36_transition14_20261114_v1"
RECORD = "native/confirm-000/record.json"
ARRAYS = "native/confirm-000/native.npz"


def read(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def verify(name, expected):
    with (ROOT / name).open("rb") as file:
        if hashlib.file_digest(file, "sha256").hexdigest() != expected:
            raise ValueError(f"Phase 2933 archive checksum mismatch: {name}")


def pack(array):
    if array.dtype.kind != "u" or array.dtype.itemsize != 2:
        raise ValueError("Expected raw BF16 words")
    return base64.b64encode(array.astype("<u2", copy=False).tobytes()).decode("ascii")


def export():
    execution = read("native-execution.json")
    record_sha = next(row["sha256"] for row in execution["records"] if row["path"] == RECORD)
    verify(RECORD, record_sha)
    record = read(RECORD)
    card = read("native-feature-card.json")
    plan = read("native-plan.json")
    snapshot = card["model_snapshot"]
    if snapshot != plan["model_snapshot"] or snapshot != record["model_snapshot"]:
        raise ValueError("Phase 2933 artifacts use different model snapshots")

    example = card["representative_contexts"][0]
    index = record["row_ids"].index(example["id"])
    row = next(item for item in plan["rows"] if item["id"] == example["id"])
    if example["generated_prefix"] != row["generated_prefix"] or len(row["sites"]) != 4:
        raise ValueError("Unexpected Phase 2933 sample layout")
    baseline = next(item for item in record["effects"] if item["id"] == row["id"] and item["arm"] == "mask0")
    verify(ARRAYS, record["arrays_sha256"])
    with np.load(ROOT / ARRAYS, allow_pickle=False) as archive:
        hidden = archive["hidden"][index]
        mlp = archive["B36_A"][index]
        if hidden.shape != (41, 4, 5120) or mlp.shape != (4, 17408):
            raise ValueError("Saved activation arrays do not match the display layout")
        return {
            "phase": 2933,
            "snapshot": snapshot,
            "run": record["run_id"],
            "caseId": row["id"],
            "prefix": row["generated_prefix"],
            "positions": [
                {"index": position, "tokenId": row["tokens"][position], "site": site}
                for site, position in zip(row["site_names"], row["sites"], strict=True)
            ],
            "units": card["unit_ids"],
            "block": card["block_index"],
            "featureId": card["feature_id"],
            "comparisons": [item for item in card["causal_results"] if item["arm"] in ("mask0", "mask7")],
            "tokens": row["choice_strings"],
            "scores": baseline["scores"],
            "hiddenShape": hidden.shape,
            "mlpShape": mlp.shape,
            "hidden": pack(hidden),
            "mlp": pack(mlp),
            "source": {"file": ARRAYS, "sha256": record["arrays_sha256"], "recordSha256": record_sha},
        }


if __name__ == "__main__":
    print(json.dumps(export(), separators=(",", ":"), allow_nan=False))
