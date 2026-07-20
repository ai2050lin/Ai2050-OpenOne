#!/usr/bin/env python3
"""Run staged three-model Phase574 behavior qualification."""

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
import phase574_query_source_protocol as protocol  # noqa: E402


OUT_DIR = protocol.OUT_DIR


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


def rows_path(model: str) -> Path:
    return OUT_DIR / f"phase574_{model}_behavior_rows.jsonl.gz"


def summary_path(model: str) -> Path:
    return OUT_DIR / f"phase574_{model}_behavior_summary.json"


def registry_path(model: str) -> Path:
    return OUT_DIR / f"phase574_{model}_behavior_registry.json"


def contract_path(model: str) -> Path:
    return OUT_DIR / f"phase574_{model}_behavior_contract.json"


def run_rows(
    loaded: Any,
    model: str,
    rows: list[dict[str, Any]],
    output: list[dict[str, Any]],
    stage: str,
    max_new_tokens: int,
) -> None:
    for repeat in ("noop1", "noop2"):
        for start in range(0, len(rows), protocol.FIXED_BATCH_SIZE):
            batch = rows[start:start + protocol.FIXED_BATCH_SIZE]
            output.extend(generate_batch(
                loaded, model, batch, repeat, max_new_tokens
            ))
        print(
            f"[{time.strftime('%H:%M:%S')}] {model} Phase574 {stage}/{repeat} "
            f"{len(rows)}/{len(rows)}",
            flush=True,
        )


def run(model: str, max_new_tokens: int, restart: bool) -> Path:
    frozen = read_json(protocol.PROTOCOL_PATH)
    audit = read_json(protocol.AUDIT_PATH)
    if not audit["valid"] or audit["open_cases_sha256"] != sha256_file(
        protocol.OPEN_CASES_PATH
    ):
        raise RuntimeError("Phase574 frozen protocol or case drift")
    if restart:
        for path in (
            rows_path(model), summary_path(model), registry_path(model),
            contract_path(model),
        ):
            path.unlink(missing_ok=True)
    contract = {
        "schema_version": "phase574_behavior_contract.v1",
        "phase_id": protocol.PHASE,
        "created_at": now(),
        "model": model,
        "open_cases_sha256": sha256_file(protocol.OPEN_CASES_PATH),
        "protocol_sha256": sha256_file(protocol.PROTOCOL_PATH),
        "fixed_batch_size": protocol.FIXED_BATCH_SIZE,
        "noop_repeats": protocol.NOOP_REPEATS,
        "do_sample": False,
        "torch_dtype_requested": "torch.bfloat16",
        "causal_splits_read": False,
        "sealed_split_read": False,
    }
    if contract_path(model).exists():
        existing = read_json(contract_path(model))
        for key in (
            "model", "open_cases_sha256", "protocol_sha256", "fixed_batch_size",
            "noop_repeats", "do_sample", "torch_dtype_requested",
            "causal_splits_read", "sealed_split_read",
        ):
            if existing[key] != contract[key]:
                raise RuntimeError(f"Phase574 behavior contract drift: {model}/{key}")
    else:
        write_json(contract_path(model), contract)

    cases = [
        row for row in iter_jsonl(protocol.OPEN_CASES_PATH)
        if row["split"] in protocol.STRUCTURE_SPLITS
    ]
    case_bank = {row["case_id"]: row for row in cases}
    by_world_variant = {
        (row["base_case_id"], row["variant"]): row for row in cases
    }
    base_rows = {
        row["base_case_id"]: row for row in cases if row["variant"] == "base"
    }
    loaded = None
    output_rows: list[dict[str, Any]] = []
    relation_counts: dict[str, int] = {}
    all_axis_counts: dict[str, int] = {}
    selected_for_controls: dict[str, list[str]] = {}
    final_selected: dict[str, list[str]] = {}
    started = time.monotonic()
    try:
        loaded = load_probe_model(model)
        loaded.tokenizer.padding_side = "left"
        run_dtype = str(next(loaded.model.parameters()).dtype)
        if run_dtype != "torch.bfloat16":
            raise RuntimeError(f"Phase574 requires BF16, got {run_dtype}")

        structure_pass = True
        for split in ("structure_discovery", "structure_confirmation"):
            relation_rows = sorted(
                [
                    row for row in cases
                    if row["split"] == split
                    and row["variant"] in ("base", "relation_swap")
                ],
                key=lambda row: row["case_id"],
            )
            run_rows(
                loaded, model, relation_rows, output_rows,
                f"{split}_relation", max_new_tokens,
            )
            by_repeat = {
                (row["case_id"], row["execution_repeat"]): row
                for row in output_rows
            }
            world_ids = sorted({row["base_case_id"] for row in relation_rows})
            eligible = [
                base_id for base_id in world_ids
                if stable_expected(by_repeat, f"{base_id}_base")
                and stable_expected(by_repeat, f"{base_id}_relation_swap")
            ]
            relation_counts[split] = len(eligible)
            selected_for_controls[split] = balanced_worlds(
                base_rows, eligible, protocol.CONTROL_SCREEN_CAP_PER_SPLIT_MODEL
            )
            structure_pass = structure_pass and len(eligible) >= frozen[
                "behavior_gate"
            ]["minimum_relation_qualified_worlds_each_structure_split"]

        if structure_pass:
            for split in ("structure_discovery", "structure_confirmation"):
                controls = sorted(
                    [
                        by_world_variant[(base_id, variant)]
                        for base_id in selected_for_controls[split]
                        for variant in ("object_swap", "order_swap")
                    ],
                    key=lambda row: row["case_id"],
                )
                run_rows(
                    loaded, model, controls, output_rows,
                    f"{split}_controls", max_new_tokens,
                )
                by_repeat = {
                    (row["case_id"], row["execution_repeat"]): row
                    for row in output_rows
                }
                eligible = [
                    base_id for base_id in selected_for_controls[split]
                    if stable_expected(by_repeat, f"{base_id}_object_swap")
                    and stable_expected(by_repeat, f"{base_id}_order_swap")
                ]
                all_axis_counts[split] = len(eligible)
                final_selected[split] = balanced_worlds(
                    base_rows, eligible, protocol.FINAL_WORLDS_PER_SPLIT_MODEL
                )
                structure_pass = structure_pass and len(final_selected[split]) >= frozen[
                    "behavior_gate"
                ]["minimum_all_axis_qualified_worlds_each_structure_split"]

        heldout_pass = False
        if structure_pass:
            split = "heldout_recombination"
            relation_rows = sorted(
                [
                    row for row in cases
                    if row["split"] == split
                    and row["variant"] in ("base", "relation_swap")
                ],
                key=lambda row: row["case_id"],
            )
            run_rows(
                loaded, model, relation_rows, output_rows,
                f"{split}_relation", max_new_tokens,
            )
            by_repeat = {
                (row["case_id"], row["execution_repeat"]): row
                for row in output_rows
            }
            world_ids = sorted({row["base_case_id"] for row in relation_rows})
            eligible = [
                base_id for base_id in world_ids
                if stable_expected(by_repeat, f"{base_id}_base")
                and stable_expected(by_repeat, f"{base_id}_relation_swap")
            ]
            relation_counts[split] = len(eligible)
            selected_for_controls[split] = balanced_worlds(
                base_rows, eligible, protocol.CONTROL_SCREEN_CAP_PER_SPLIT_MODEL
            )
            controls = sorted(
                [
                    by_world_variant[(base_id, variant)]
                    for base_id in selected_for_controls[split]
                    for variant in ("object_swap", "order_swap")
                ],
                key=lambda row: row["case_id"],
            )
            run_rows(
                loaded, model, controls, output_rows,
                f"{split}_controls", max_new_tokens,
            )
            by_repeat = {
                (row["case_id"], row["execution_repeat"]): row
                for row in output_rows
            }
            eligible = [
                base_id for base_id in selected_for_controls[split]
                if stable_expected(by_repeat, f"{base_id}_object_swap")
                and stable_expected(by_repeat, f"{base_id}_order_swap")
            ]
            all_axis_counts[split] = len(eligible)
            final_selected[split] = balanced_worlds(
                base_rows, eligible, protocol.FINAL_WORLDS_PER_SPLIT_MODEL
            )
            heldout_pass = len(final_selected[split]) >= frozen[
                "behavior_gate"
            ]["minimum_all_axis_qualified_worlds_heldout"]

        write_jsonl(rows_path(model), output_rows)
        repeats: dict[str, list[dict[str, Any]]] = {}
        for row in output_rows:
            repeats.setdefault(row["case_id"], []).append(row)
        exact_mismatch = sum(
            len(values) == 2
            and values[0]["normalized_generated"] != values[1]["normalized_generated"]
            for values in repeats.values()
        )
        semantic_mismatch = sum(
            len(values) == 2
            and values[0]["semantic_event"] != values[1]["semantic_event"]
            for values in repeats.values()
        )
        authorized = bool(structure_pass and heldout_pass)
        registry = {
            "schema_version": "phase574_behavior_registry.v1",
            "phase_id": protocol.PHASE,
            "created_at": now(),
            "model": model,
            "relation_qualified_counts_by_split": relation_counts,
            "all_axis_qualified_counts_by_split": all_axis_counts,
            "selected_base_case_ids_by_split": final_selected,
            "selected_count_per_split": {
                split: len(ids) for split, ids in final_selected.items()
            },
            "authorized_for_query_source_trace": authorized,
            "selection_uses_internal_state": False,
            "causal_splits_read": False,
            "sealed_split_read": False,
        }
        write_json(registry_path(model), registry)
        summary = {
            "schema_version": "phase574_behavior_summary.v1",
            "phase_id": protocol.PHASE,
            "created_at": now(),
            "status": "complete",
            "model": model,
            "torch_dtype": run_dtype,
            "executed_behavior_row_count": len(output_rows),
            "executed_unique_case_count": len(repeats),
            "relation_qualified_counts_by_split": relation_counts,
            "all_axis_qualified_counts_by_split": all_axis_counts,
            "selected_count_per_split": registry["selected_count_per_split"],
            "noop_exact_text_mismatch_count": exact_mismatch,
            "noop_semantic_event_mismatch_count": semantic_mismatch,
            "structure_behavior_gate_pass": structure_pass,
            "heldout_behavior_gate_pass": heldout_pass,
            "authorized_for_query_source_trace": authorized,
            "runtime_seconds": time.monotonic() - started,
            "rows_sha256": sha256_file(rows_path(model)),
            "registry_sha256": sha256_file(registry_path(model)),
            "causal_splits_read": False,
            "sealed_split_read": False,
        }
        write_json(summary_path(model), summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return summary_path(model)
    finally:
        release_loaded(loaded)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=protocol.MODELS)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()
    run(args.model, args.max_new_tokens, args.restart)


if __name__ == "__main__":
    main()
