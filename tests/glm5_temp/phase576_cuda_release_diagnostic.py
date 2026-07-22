#!/usr/bin/env python3
"""Synthetic-only CUDA allocator diagnostic for the Phase576 qwen3 failure."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT / "tests/glm5", ROOT / "tests/gpt5"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import phase576_gpt5_fruit_engineering_qualification as engineering  # noqa: E402
from phase983_cross_model_engine import (  # noqa: E402
    load_model_adapter,
    release_model_adapter,
)


def memory() -> dict[str, Any]:
    return {
        "allocated": int(torch.cuda.memory_allocated()),
        "reserved": int(torch.cuda.memory_reserved()),
        "max_allocated": int(torch.cuda.max_memory_allocated()),
    }


def active_snapshot() -> dict[str, Any]:
    active = []
    for segment in torch.cuda.memory_snapshot():
        for block in segment.get("blocks", []):
            if str(block.get("state", "")).startswith("active"):
                active.append({
                    "size": int(block.get("size", 0)),
                    "requested_size": int(block.get("requested_size", 0)),
                    "state": block.get("state"),
                })
    counts: dict[str, int] = {}
    for row in active:
        key = f"{row['requested_size']}|{row['size']}|{row['state']}"
        counts[key] = counts.get(key, 0) + 1
    return {
        "active_block_count": len(active),
        "active_requested_bytes": sum(row["requested_size"] for row in active),
        "active_size_bytes": sum(row["size"] for row in active),
        "block_multiplicity": dict(sorted(counts.items())),
    }


def live_cuda_tensors() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for value in gc.get_objects():
        try:
            if isinstance(value, torch.Tensor) and value.is_cuda:
                rows.append({
                    "python_type": f"{type(value).__module__}.{type(value).__qualname__}",
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "nbytes": int(value.numel() * value.element_size()),
                    "requires_grad": bool(value.requires_grad),
                    "data_ptr": int(value.data_ptr()),
                })
        except (ReferenceError, RuntimeError):
            continue
    return sorted(rows, key=lambda row: (row["nbytes"], row["data_ptr"]), reverse=True)


def cleanup(adapter: Any, report: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    report = None
    release_model_adapter(adapter)
    adapter = None
    gc.collect()
    torch.cuda.synchronize()
    clear_cublas = getattr(torch._C, "_cuda_clearCublasWorkspaces", None)
    if clear_cublas is not None:
        clear_cublas()
    torch.cuda.empty_cache()
    try:
        torch.cuda.ipc_collect()
    except RuntimeError:
        pass
    gc.collect()
    snapshot = active_snapshot()
    snapshot["live_cuda_tensors"] = live_cuda_tensors()
    return memory(), snapshot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=("load-only", "qualification"))
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.memory_allocated() != 0:
        raise RuntimeError(f"dirty diagnostic CUDA baseline: {memory()}")
    torch.cuda.reset_peak_memory_stats()
    adapter = load_model_adapter("qwen3")
    after_load = memory()
    report = None
    if args.mode == "qualification":
        report = engineering.collect_repeat_forward_report(adapter, "qwen3")
    before_release = memory()
    after_release, snapshot = cleanup(adapter, report)
    print(json.dumps({
        "mode": args.mode,
        "after_load": after_load,
        "before_release": before_release,
        "after_release": after_release,
        "active_snapshot_after_release": snapshot,
        "research_case_content_read": False,
        "synthetic_prompt_sha256": engineering.sha256_bytes(
            engineering.SYNTHETIC_PROMPT.encode("utf-8")
        ),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
