#!/usr/bin/env python3
"""Run Phase571 independent phenotype and fixed-batch stability screens."""

from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import json
import os
import sys
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))
os.environ.setdefault("PROBE_TORCH_DTYPE", "bfloat16")

from hf_probe_env import load_probe_model, release_loaded  # noqa: E402
from phase548_shared_attention_compute_protocol import render_chat  # noqa: E402
from phase569_relation_competition_behavior import classify  # noqa: E402
import phase571_relation_block_protocol as protocol  # noqa: E402


MODELS = protocol.MODELS
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
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def rows_path(model: str) -> Path:
    return OUT_DIR / f"phase571_{model}_behavior_rows.jsonl.gz"


def summary_path(model: str) -> Path:
    return OUT_DIR / f"phase571_{model}_behavior_summary.json"


def contract_path(model: str) -> Path:
    return OUT_DIR / f"phase571_{model}_behavior_contract.json"


def load_cases(model: str) -> list[dict[str, Any]]:
    rows = [row for row in iter_jsonl(protocol.OPEN_CASES_PATH) if row["model"] == model]
    expected = (
        protocol.MIXED_CELLS_PER_MODEL
        * protocol.CANDIDATE_WORLDS_PER_CELL
        * len(protocol.OPEN_POOLS)
    )
    if len(rows) != expected or any(row["sealed"] for row in rows):
        raise RuntimeError(f"Phase571 open denominator drift for {model}: {len(rows)}")
    return sorted(rows, key=lambda row: (row["pool"], row["case_id"]))


def prepare(model: str, restart: bool) -> None:
    frozen = read_json(protocol.PROTOCOL_PATH)
    audit = read_json(protocol.AUDIT_PATH)
    if not audit["valid"] or audit["open_cases_sha256"] != sha256_file(protocol.OPEN_CASES_PATH):
        raise RuntimeError("Phase571 frozen protocol failed or drifted")
    payload = {
        "schema_version": "phase571_behavior_contract.v1",
        "phase_id": protocol.PHASE,
        "created_at": now(),
        "model": model,
        "open_cases_sha256": sha256_file(protocol.OPEN_CASES_PATH),
        "protocol_sha256": sha256_file(protocol.PROTOCOL_PATH),
        "fixed_batch_size": frozen["fixed_execution_batch_size"],
        "do_sample": False,
        "torch_dtype_requested": "torch.bfloat16",
        "use_8bit": False,
        "sealed_split_read": False,
    }
    if restart:
        for path in (rows_path(model), summary_path(model), contract_path(model)):
            path.unlink(missing_ok=True)
    if contract_path(model).exists():
        existing = read_json(contract_path(model))
        for key in (
            "model", "open_cases_sha256", "protocol_sha256", "fixed_batch_size",
            "do_sample", "torch_dtype_requested", "use_8bit", "sealed_split_read",
        ):
            if existing[key] != payload[key]:
                raise RuntimeError(f"Phase571 behavior contract drift: {model}/{key}")
    else:
        write_json(contract_path(model), payload)


def phenotype_matches(row: dict[str, Any], phenotype: str) -> bool:
    if phenotype == "stable_correct":
        return bool(row["semantic_correct"])
    if phenotype == "stable_relation_confusion":
        return bool(row["relation_confusion"])
    raise ValueError(phenotype)


def balanced(rows: list[dict[str, Any]], cap: int) -> list[dict[str, Any]]:
    strata: dict[tuple[str, str, str], deque[dict[str, Any]]] = defaultdict(deque)
    for row in sorted(rows, key=lambda item: item["case_id"]):
        strata[(
            row["source_factorial_cell"], row["target"], row["other_relation_target"]
        )].append(row)
    keys = sorted(strata)
    selected: list[dict[str, Any]] = []
    while keys and len(selected) < cap:
        remaining = []
        for key in keys:
            if len(selected) >= cap:
                break
            if strata[key]:
                selected.append(strata[key].popleft())
            if strata[key]:
                remaining.append(key)
        keys = remaining
    return selected


def generate_batch(
    loaded: Any,
    model: str,
    rows: list[dict[str, Any]],
    repeat: str,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    prompts = [render_chat(loaded.tokenizer, model, row["raw_prompt"]) for row in rows]
    encoded = loaded.tokenizer(prompts, return_tensors="pt", padding=True, truncation=False)
    prompt_width = int(encoded["input_ids"].shape[1])
    encoded = {key: value.to(loaded.input_device) for key, value in encoded.items()}
    with torch.inference_mode():
        generated = loaded.model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=loaded.tokenizer.pad_token_id,
            eos_token_id=loaded.tokenizer.eos_token_id,
        )
    output = []
    for index, row in enumerate(rows):
        text = loaded.tokenizer.decode(
            generated[index, prompt_width:], skip_special_tokens=True
        )
        output.append({
            **row,
            **classify(row, text),
            "execution_repeat": repeat,
            "observer_only": True,
            "causal": False,
        })
    del encoded, generated
    return output


def run(model: str, batch_size: int, max_new_tokens: int, restart: bool) -> Path:
    if batch_size != 8:
        raise ValueError("Phase571 freezes batch size at 8")
    prepare(model, restart)
    cases = load_cases(model)
    loaded = None
    started = time.monotonic()
    try:
        loaded = load_probe_model(model)
        loaded.tokenizer.padding_side = "left"
        run_dtype = str(next(loaded.model.parameters()).dtype)
        if run_dtype != "torch.bfloat16":
            raise RuntimeError(f"Phase571 requires BF16, got {run_dtype}")
        output_rows: list[dict[str, Any]] = []
        for pool in protocol.OPEN_POOLS:
            pool_rows = [row for row in cases if row["pool"] == pool]
            repeats = ("baseline",) if pool != "block_causal" else ("noop1", "noop2")
            for repeat in repeats:
                for start in range(0, len(pool_rows), batch_size):
                    batch = pool_rows[start:start + batch_size]
                    output_rows.extend(
                        generate_batch(loaded, model, batch, repeat, max_new_tokens)
                    )
                    done = min(start + batch_size, len(pool_rows))
                    if start == 0 or done == len(pool_rows) or (start // batch_size) % 16 == 15:
                        print(
                            f"[{time.strftime('%H:%M:%S')}] {model} Phase571 "
                            f"{pool}/{repeat} {done}/{len(pool_rows)}",
                            flush=True,
                        )
        write_jsonl(rows_path(model), output_rows)

        by_case_repeat = {
            (row["case_id"], row["execution_repeat"]): row for row in output_rows
        }
        selected_ids: dict[str, dict[str, list[str]]] = {}
        eligible_counts: dict[str, dict[str, int]] = {}
        pair_counts: dict[str, dict[str, int]] = {}
        for pool in protocol.OPEN_POOLS:
            selected_ids[pool] = {}
            eligible_counts[pool] = {}
            pair_counts[pool] = {}
            for phenotype in protocol.PHENOTYPES:
                candidates = [
                    row for row in cases
                    if row["pool"] == pool
                ]
                eligible: list[dict[str, Any]] = []
                for case in candidates:
                    if pool == "block_causal":
                        first = by_case_repeat[(case["case_id"], "noop1")]
                        second = by_case_repeat[(case["case_id"], "noop2")]
                        valid = (
                            phenotype_matches(first, phenotype)
                            and phenotype_matches(second, phenotype)
                            and first["semantic_event"] == second["semantic_event"]
                        )
                    else:
                        first = by_case_repeat[(case["case_id"], "baseline")]
                        valid = phenotype_matches(first, phenotype)
                    if valid:
                        eligible.append(first)
                cap = (
                    protocol.CAUSAL_SELECTION_PER_PHENOTYPE
                    if pool == "block_causal"
                    else protocol.TRACE_SELECTION_PER_PHENOTYPE
                )
                chosen = balanced(eligible, cap)
                eligible_counts[pool][phenotype] = len(eligible)
                selected_ids[pool][phenotype] = [row["case_id"] for row in chosen]
                pair_counts[pool][phenotype] = len({
                    (row["target"], row["other_relation_target"]) for row in chosen
                })

        causal_cases = [row for row in cases if row["pool"] == "block_causal"]
        exact_mismatch = 0
        semantic_mismatch = 0
        for case in causal_cases:
            first = by_case_repeat[(case["case_id"], "noop1")]
            second = by_case_repeat[(case["case_id"], "noop2")]
            exact_mismatch += int(first["normalized_generated"] != second["normalized_generated"])
            semantic_mismatch += int(first["semantic_event"] != second["semantic_event"])

        trace_qualified = all(
            len(selected_ids[pool][phenotype]) >= protocol.MINIMUM_CASES_PER_PHENOTYPE
            and pair_counts[pool][phenotype] >= 8
            for pool in ("block_discovery", "block_confirmation")
            for phenotype in protocol.PHENOTYPES
        )
        causal_qualified = all(
            len(selected_ids["block_causal"][phenotype])
            >= protocol.MINIMUM_CASES_PER_PHENOTYPE
            and pair_counts["block_causal"][phenotype] >= 8
            for phenotype in protocol.PHENOTYPES
        )
        summary = {
            "schema_version": "phase571_behavior_summary.v2",
            "phase_id": protocol.PHASE,
            "created_at": now(),
            "status": "complete",
            "model": model,
            "torch_dtype": run_dtype,
            "registered_open_case_count": len(cases),
            "behavior_row_count": len(output_rows),
            "fixed_batch_size": batch_size,
            "eligible_counts_by_pool_phenotype": eligible_counts,
            "selected_case_ids_by_pool_phenotype": selected_ids,
            "selected_target_other_pair_counts": pair_counts,
            "causal_noop_exact_text_mismatch_count": exact_mismatch,
            "causal_noop_exact_text_mismatch_rate": exact_mismatch / len(causal_cases),
            "causal_noop_semantic_event_mismatch_count": semantic_mismatch,
            "causal_noop_semantic_event_mismatch_rate": semantic_mismatch / len(causal_cases),
            "qualified_for_signed_write_trace": trace_qualified,
            "qualified_for_coarse_block_causal": causal_qualified,
            "runtime_seconds": time.monotonic() - started,
            "rows_sha256": sha256_file(rows_path(model)),
            "sealed_split_read": False,
        }
        write_json(summary_path(model), summary)
        print(json.dumps({
            "model": model,
            "eligible": eligible_counts,
            "selected_pair_counts": pair_counts,
            "noop_exact_mismatch": exact_mismatch,
            "noop_semantic_mismatch": semantic_mismatch,
            "trace_qualified": trace_qualified,
            "causal_qualified": causal_qualified,
        }, ensure_ascii=False, indent=2), flush=True)
        return summary_path(model)
    finally:
        release_loaded(loaded)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()
    run(args.model, args.batch_size, args.max_new_tokens, args.restart)


if __name__ == "__main__":
    main()
