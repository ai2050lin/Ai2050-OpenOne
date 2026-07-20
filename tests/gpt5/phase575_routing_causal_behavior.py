#!/usr/bin/env python3
"""Qualify the frozen Phase575 causal worlds before internal intervention."""

from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))
os.environ.setdefault("PROBE_TORCH_DTYPE", "bfloat16")

from hf_probe_env import load_probe_model, release_loaded  # noqa: E402
from phase573_natural_transition_behavior import (  # noqa: E402
    balanced_worlds,
    generate_batch,
    stable_expected,
)
import phase575_routing_causal_protocol as causal_protocol  # noqa: E402
import phase575_source_competition_protocol as protocol  # noqa: E402


MODEL = "qwen3"
OUT_DIR = protocol.OUT_DIR
ROWS_PATH = OUT_DIR / "phase575_qwen3_routing_causal_behavior_rows.jsonl.gz"
REGISTRY_PATH = OUT_DIR / "phase575_qwen3_routing_causal_behavior_registry.json"
SUMMARY_PATH = OUT_DIR / "phase575_qwen3_routing_causal_behavior_summary.json"
CONTRACT_PATH = OUT_DIR / "phase575_qwen3_routing_causal_behavior_contract.json"
CONTROL_VARIANTS = ("object_swap", "relation_object_swap", "order_swap")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)
                + "\n"
            )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def run_rows(
    loaded: Any,
    rows: list[dict[str, Any]],
    output: list[dict[str, Any]],
    stage: str,
    max_new_tokens: int,
    batch_size: int,
) -> None:
    for repeat in ("noop1", "noop2"):
        for start in range(0, len(rows), batch_size):
            output.extend(
                generate_batch(
                    loaded,
                    MODEL,
                    rows[start : start + batch_size],
                    repeat,
                    max_new_tokens,
                )
            )
        print(
            f"[{time.strftime('%H:%M:%S')}] {MODEL} Phase575 {stage}/{repeat} "
            f"{len(rows)}/{len(rows)}",
            flush=True,
        )


def stable_variants(
    by_repeat: dict[tuple[str, str], dict[str, Any]],
    base_id: str,
    variants: tuple[str, ...],
) -> bool:
    return all(
        stable_expected(by_repeat, f"{base_id}_{variant}")
        for variant in variants
    )


def run(restart: bool) -> Path:
    frozen = read_json(causal_protocol.CAUSAL_PROTOCOL)
    behavior = frozen["causal_behavior"]
    if restart:
        for path in (ROWS_PATH, REGISTRY_PATH, SUMMARY_PATH, CONTRACT_PATH):
            path.unlink(missing_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("Phase575 causal behavior qualification requires CUDA")

    contract = {
        "schema_version": "phase575_routing_causal_behavior_contract.v1",
        "phase_id": protocol.PHASE,
        "created_at": now(),
        "model": MODEL,
        "causal_protocol_sha256": sha256_file(causal_protocol.CAUSAL_PROTOCOL),
        "open_cases_sha256": sha256_file(protocol.OPEN_CASES_PATH),
        "splits": list(protocol.CAUSAL_SPLITS),
        "two_noop_repeats": True,
        "do_sample": False,
        "torch_dtype_requested": "torch.bfloat16",
        "cuda_required": True,
        "causal_splits_read": True,
        "sealed_split_read": False,
    }
    if CONTRACT_PATH.exists():
        existing = read_json(CONTRACT_PATH)
        for key, value in contract.items():
            if key != "created_at" and existing[key] != value:
                raise RuntimeError(f"Phase575 causal behavior contract drift: {key}")
    else:
        write_json(CONTRACT_PATH, contract)

    cases = [
        row
        for row in iter_jsonl(protocol.OPEN_CASES_PATH)
        if row["split"] in protocol.CAUSAL_SPLITS
    ]
    expected = len(protocol.CAUSAL_SPLITS) * 1024 * len(protocol.VARIANTS)
    if len(cases) != expected or any(row["sealed"] for row in cases):
        raise RuntimeError(f"Phase575 causal behavior denominator drift: {len(cases)}")
    by_world_variant = {
        (row["base_case_id"], row["variant"]): row for row in cases
    }
    base_rows = {
        row["base_case_id"]: row for row in cases if row["variant"] == "base"
    }

    loaded = None
    output: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {}
    selected: dict[str, list[str]] = {}
    started = time.monotonic()
    try:
        loaded = load_probe_model(MODEL)
        if loaded.input_device.type != "cuda":
            raise RuntimeError("Phase575 causal behavior model is not on CUDA")
        run_dtype = str(next(loaded.model.parameters()).dtype)
        if run_dtype != "torch.bfloat16":
            raise RuntimeError(f"Phase575 causal behavior requires BF16, got {run_dtype}")
        loaded.tokenizer.padding_side = "left"
        for split in protocol.CAUSAL_SPLITS:
            relation_rows = sorted(
                [
                    row
                    for row in cases
                    if row["split"] == split
                    and row["variant"] in ("base", "relation_swap")
                ],
                key=lambda row: row["case_id"],
            )
            run_rows(
                loaded,
                relation_rows,
                output,
                f"{split}_relation",
                int(behavior["max_new_tokens"]),
                int(behavior["batch_size"]),
            )
            by_repeat = {
                (row["case_id"], row["execution_repeat"]): row for row in output
            }
            world_ids = sorted({row["base_case_id"] for row in relation_rows})
            relation_eligible = [
                base_id
                for base_id in world_ids
                if stable_variants(
                    by_repeat, base_id, ("base", "relation_swap")
                )
            ]
            if len(relation_eligible) < int(
                behavior["minimum_relation_qualified_each_split"]
            ):
                raise RuntimeError(
                    f"Phase575 causal relation behavior gate failed: "
                    f"{split}/{len(relation_eligible)}"
                )
            control_ids = balanced_worlds(
                base_rows,
                relation_eligible,
                int(behavior["control_screen_cap_each_split"]),
            )
            controls = sorted(
                [
                    by_world_variant[(base_id, variant)]
                    for base_id in control_ids
                    for variant in CONTROL_VARIANTS
                ],
                key=lambda row: row["case_id"],
            )
            run_rows(
                loaded,
                controls,
                output,
                f"{split}_controls",
                int(behavior["max_new_tokens"]),
                int(behavior["batch_size"]),
            )
            by_repeat = {
                (row["case_id"], row["execution_repeat"]): row for row in output
            }
            five_variant = [
                base_id
                for base_id in control_ids
                if stable_variants(by_repeat, base_id, CONTROL_VARIANTS)
            ]
            selected[split] = balanced_worlds(
                base_rows,
                five_variant,
                int(behavior["selected_five_variant_worlds_each_split"]),
            )
            if len(selected[split]) != int(
                behavior["selected_five_variant_worlds_each_split"]
            ):
                raise RuntimeError(
                    f"Phase575 causal five-variant behavior gate failed: "
                    f"{split}/{len(selected[split])}"
                )
            diagnostics[split] = {
                "relation_qualified_world_count": len(relation_eligible),
                "control_screen_world_count": len(control_ids),
                "five_variant_qualified_world_count": len(five_variant),
                "selected_world_count": len(selected[split]),
            }

        write_jsonl(ROWS_PATH, output)
        repeats: dict[str, list[dict[str, Any]]] = {}
        for row in output:
            repeats.setdefault(row["case_id"], []).append(row)
        exact_mismatch = sum(
            len(rows) == 2
            and rows[0]["normalized_generated"] != rows[1]["normalized_generated"]
            for rows in repeats.values()
        )
        semantic_mismatch = sum(
            len(rows) == 2
            and rows[0]["semantic_event"] != rows[1]["semantic_event"]
            for rows in repeats.values()
        )
        registry = {
            "schema_version": "phase575_routing_causal_behavior_registry.v1",
            "phase_id": protocol.PHASE,
            "created_at": now(),
            "model": MODEL,
            "diagnostics_by_split": diagnostics,
            "selected_base_case_ids_by_split": selected,
            "selected_count_by_split": {
                split: len(ids) for split, ids in selected.items()
            },
            "authorized_for_routing_causal_test": (
                exact_mismatch == 0 and semantic_mismatch == 0
            ),
            "selection_uses_internal_state": False,
            "causal_splits_read": True,
            "sealed_split_read": False,
        }
        write_json(REGISTRY_PATH, registry)
        summary = {
            "schema_version": "phase575_routing_causal_behavior_summary.v1",
            "phase_id": protocol.PHASE,
            "created_at": now(),
            "status": "complete",
            "model": MODEL,
            "device_type": loaded.input_device.type,
            "torch_dtype": run_dtype,
            "executed_behavior_row_count": len(output),
            "executed_unique_case_count": len(repeats),
            "diagnostics_by_split": diagnostics,
            "noop_exact_text_mismatch_count": exact_mismatch,
            "noop_semantic_event_mismatch_count": semantic_mismatch,
            "authorized_for_routing_causal_test": registry[
                "authorized_for_routing_causal_test"
            ],
            "runtime_seconds": time.monotonic() - started,
            "rows_sha256": sha256_file(ROWS_PATH),
            "registry_sha256": sha256_file(REGISTRY_PATH),
            "causal_splits_read": True,
            "sealed_split_read": False,
        }
        write_json(SUMMARY_PATH, summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return SUMMARY_PATH
    finally:
        release_loaded(loaded)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()
    run(args.restart)


if __name__ == "__main__":
    main()
