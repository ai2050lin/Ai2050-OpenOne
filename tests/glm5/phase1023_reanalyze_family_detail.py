#!/usr/bin/env python3
"""Recompute Phase1023 residual/head metrics with per-family detail.

No model is loaded and no candidate selection is changed.  This script only
expands already frozen observations into eight-family reporting.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1023_ecological_niche_protocol as protocol
import phase1023_fp16_ecology_scan as scan


def main() -> None:
    summaries = {}
    for model in protocol.MODELS:
        out_dir = protocol.OUT_ROOT / "ecology" / model
        summary = protocol.read_json(out_dir / "summary.json")
        cases = protocol.read_jsonl(
            protocol.OUT_ROOT / "protocol" / f"atlas.{model}.jsonl"
        )
        residual = np.load(
            out_dir / "residual_states.fp16.npy",
            mmap_mode="r",
        )
        residual_rows = []
        for role_index, role in enumerate(scan.ROLES):
            for depth in range(residual.shape[2]):
                residual_rows.append({
                    "schema_version": "phase1023_residual_metric.v1",
                    "role": role,
                    "depth": depth,
                    "relative_depth": (
                        depth / max(residual.shape[2] - 1, 1)
                    ),
                    "metrics": scan.all_metrics(
                        np.asarray(residual[:, role_index, depth, :]),
                        cases,
                    ),
                })
        protocol.write_jsonl(
            out_dir / "residual_metrics.jsonl",
            residual_rows,
        )

        selected = summary["selected_layers"]
        selected_union = sorted({
            depth for values in selected.values() for depth in values
        })
        selected_index = {
            depth: index for index, depth in enumerate(selected_union)
        }
        heads = np.load(
            out_dir / "attention_heads.fp16.npy",
            mmap_mode="r",
        )
        head_rows = []
        for role in scan.ROLES:
            role_index = scan.ROLE_INDEX[role]
            for depth in selected[role]:
                layer_index = selected_index[depth]
                for head in range(heads.shape[3]):
                    head_rows.append({
                        "schema_version": (
                            "phase1023_attention_head_metric.v1"
                        ),
                        "role": role,
                        "depth": depth,
                        "relative_depth": (
                            depth / max(residual.shape[2] - 1, 1)
                        ),
                        "head": head,
                        "metrics": scan.all_metrics(
                            np.asarray(
                                heads[
                                    :,
                                    role_index,
                                    layer_index,
                                    head,
                                    :,
                                ]
                            ),
                            cases,
                        ),
                    })
        protocol.write_jsonl(
            out_dir / "attention_head_metrics.jsonl",
            head_rows,
        )
        summaries[model] = {
            "residual_metric_count": len(residual_rows),
            "attention_head_metric_count": len(head_rows),
            "candidate_selection_changed": False,
        }
        print(f"[family-detail] {model} complete", flush=True)
    protocol.write_json(
        protocol.OUT_ROOT / "final" / "family_detail_reanalysis.json",
        {
            "schema_version": "phase1023_family_detail_reanalysis.v1",
            "phase": protocol.PHASE,
            "models": summaries,
            "claim_limit": (
                "descriptive per-family expansion of frozen observations"
            ),
        },
    )
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
