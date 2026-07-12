#!/usr/bin/env python3
"""Extract all label-free pairwise physical profiles from Phase379 ledgers."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import torch


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase379_global_reuse_difference_layout"
BLIND = OUT / "phase379_blind_case_registry.jsonl"
MODELS = ("qwen3", "glm4", "deepseek7b")
SPLITS = ("fresh_discovery", "fresh_calibration")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cosine_rows(values: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    dot = torch.einsum("...h,h->...", values, target)
    denominator = torch.linalg.vector_norm(values, dim=-1) * torch.linalg.vector_norm(
        target
    )
    return torch.where(denominator > 1e-12, dot / denominator, torch.zeros_like(dot))


def process(split: str) -> dict[str, Any]:
    if split == "fresh_calibration":
        freeze = OUT / "phase379_discovery_mapping_freeze.json"
        if not freeze.is_file() or not read_json(freeze)["authorization"][
            "open_calibration_blind_extraction"
        ]:
            raise RuntimeError("Calibration blind extraction is not authorized")
    blind_rows = [row for row in read_jsonl(BLIND) if row["phase379_split"] == split]
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in blind_rows:
        groups[(row["anonymous_model_id"], row["anonymous_group_id"])].append(row)
    if any(len(rows) != 4 for rows in groups.values()):
        raise RuntimeError("Every blind group must contain four conditions")
    event_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    model_group_counts = Counter()
    for (_anonymous_model, anonymous_group), rows in sorted(groups.items()):
        rows.sort(key=lambda row: row["anonymous_condition_slot"])
        model = next(
            name
            for name in MODELS
            if (OUT / split / "models" / name / "phase379_trace_rows.jsonl").is_file()
            and any(
                item["blind_case_id"] == rows[0]["blind_case_id"]
                for item in read_jsonl(
                    OUT / split / "models" / name / "phase379_trace_rows.jsonl"
                )
            )
        )
        model_group_counts[model] += 1
        payloads = {}
        for row in rows:
            path = (
                OUT
                / split
                / "private/models"
                / model
                / "cases"
                / f"{row['blind_case_id']}.pt"
            )
            payloads[row["blind_case_id"]] = torch.load(
                path, map_location="cpu", weights_only=True
            )
        for left_index, right_index in itertools.combinations(range(4), 2):
            left = rows[left_index]
            right = rows[right_index]
            left_payload = payloads[left["blind_case_id"]]
            right_payload = payloads[right["blind_case_id"]]
            if left_payload["component_names"] != right_payload["component_names"]:
                raise RuntimeError("Component ledger mismatch")
            vectors = (
                left_payload["vectors"].float() - right_payload["vectors"].float()
            )
            terminal = vectors[-1, -1, -1]
            terminal_norm = float(torch.linalg.vector_norm(terminal).item())
            norms = torch.linalg.vector_norm(vectors, dim=-1)
            cosines = cosine_rows(vectors, terminal)
            inner = torch.einsum("lcrh,h->lcr", vectors, terminal) / max(
                terminal_norm * terminal_norm, 1e-12
            )
            vocab_delta = (
                left_payload["full_vocabulary_logits"]
                - right_payload["full_vocabulary_logits"]
            ).float()
            pair_id = hashlib.sha256(
                f"{anonymous_group}:{left['anonymous_condition_slot']}:{right['anonymous_condition_slot']}".encode()
            ).hexdigest()[:20]
            layer_count = int(vectors.shape[0])
            for layer in range(layer_count):
                relative_depth = layer / max(layer_count - 1, 1)
                for component_index, component in enumerate(
                    left_payload["component_names"]
                ):
                    for role_index, role in enumerate(left_payload["role_names"]):
                        norm = float(norms[layer, component_index, role_index].item())
                        event_rows.append(
                            {
                                "schema_version": "52.2.0",
                                "phase_id": "Phase379-BlindLayout",
                                "model": model,
                                "phase379_split": split,
                                "anonymous_group_id": anonymous_group,
                                "anonymous_pair_id": pair_id,
                                "anonymous_slot_left": left[
                                    "anonymous_condition_slot"
                                ],
                                "anonymous_slot_right": right[
                                    "anonymous_condition_slot"
                                ],
                                "left_case_id": left["blind_case_id"],
                                "right_case_id": right["blind_case_id"],
                                "layer": layer,
                                "relative_depth": relative_depth,
                                "component_type": component,
                                "position_role": role,
                                "exact_difference_norm": norm,
                                "terminal_difference_norm": terminal_norm,
                                "norm_ratio_to_terminal": norm
                                / max(terminal_norm, 1e-12),
                                "cosine_to_terminal_difference": float(
                                    cosines[layer, component_index, role_index].item()
                                ),
                                "terminal_inner_product_share": float(
                                    inner[layer, component_index, role_index].item()
                                ),
                                "semantic_labels_available": False,
                                "target_tokens_available": False,
                                "candidate_selected": False,
                            }
                        )
            pair_rows.append(
                {
                    "schema_version": "52.2.0",
                    "phase_id": "Phase379-BlindLayout",
                    "model": model,
                    "phase379_split": split,
                    "anonymous_group_id": anonymous_group,
                    "anonymous_pair_id": pair_id,
                    "anonymous_slot_left": left["anonymous_condition_slot"],
                    "anonymous_slot_right": right["anonymous_condition_slot"],
                    "left_case_id": left["blind_case_id"],
                    "right_case_id": right["blind_case_id"],
                    "layer_count": layer_count,
                    "event_count": layer_count
                    * len(left_payload["component_names"])
                    * len(left_payload["role_names"]),
                    "terminal_difference_norm": terminal_norm,
                    "full_vocabulary_difference_norm": float(
                        torch.linalg.vector_norm(vocab_delta).item()
                    ),
                    "semantic_labels_available": False,
                    "candidate_selected": False,
                }
            )
    private = OUT / split / "private"
    private.mkdir(parents=True, exist_ok=True)
    event_path = private / "phase379_blind_event_rows.parquet"
    pair_path = private / "phase379_blind_pair_rows.parquet"
    pq.write_table(pa.Table.from_pylist(event_rows), event_path, compression="zstd")
    pq.write_table(pa.Table.from_pylist(pair_rows), pair_path, compression="zstd")
    summary = {
        "schema_version": "52.2.0",
        "phase_id": "Phase379-BlindLayout",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "split": split,
        "denominator": {
            "model_group_count": len(groups),
            "unordered_pair_count": len(pair_rows),
            "event_row_count": len(event_rows),
            "model_group_counts": dict(model_group_counts),
        },
        "quality": {
            "all_six_pairs_retained": len(pair_rows) == len(groups) * 6,
            "semantic_labels_available": False,
            "target_tokens_available": False,
            "top_k_used": False,
            "candidate_selected": False,
            "event_rows_sha256": sha256(event_path),
            "pair_rows_sha256": sha256(pair_path),
        },
        "authorization": {
            "open_semantic_mapping": split == "fresh_discovery",
            "open_calibration_semantic_mapping": split == "fresh_calibration",
            "run_causal_intervention": False,
        },
    }
    write_json(OUT / split / "phase379_blind_layout_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=SPLITS, required=True)
    args = parser.parse_args()
    process(args.split)
