#!/usr/bin/env python3
"""Phase 998 larger disjoint confirmation of the frozen causal candidates."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_utils import get_layers, get_model_info, load_model, release_model
from phase998_minimal_causal_thread_behavior import eos_ids
from phase998_minimal_causal_thread_causal import (
    ROLE_NAMES,
    THRESHOLDS,
    directional_pairs,
    final_gate,
    natural_summary,
    run_candidate_and_restoration,
    run_natural_holdout,
    summarize_conditions,
    summarize_mediation,
    write_rows,
)
from phase998_minimal_causal_thread_protocol import (
    COLORS,
    MODEL,
    OUT_ROOT,
    PHASE,
    write_json,
    write_jsonl,
)


PER_STRATUM = 4


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def pair_hash(pair_id: str) -> str:
    return hashlib.sha256(("phase998-confirm:" + pair_id).encode("utf-8")).hexdigest()


def select_confirmation(
    cases: list[dict[str, Any]], already_selected: set[str]
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in cases:
        groups[row["pair_id"]].append(row)
    strata: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for pair_id, rows in groups.items():
        rows = sorted(rows, key=lambda row: row["arm"])
        if (
            pair_id in already_selected
            or rows[0]["split"] != "holdout"
            or rows[0]["template"] == 3
        ):
            continue
        contrast = f"{rows[0]['gold']}->{rows[1]['gold']}"
        key = (
            rows[0]["template"],
            rows[0]["order"],
            rows[0]["query_role"],
            contrast,
        )
        strata[key].append(
            {
                "pair_id": pair_id,
                "partition": "confirmation",
                "template": rows[0]["template"],
                "order": rows[0]["order"],
                "query_role": rows[0]["query_role"],
                "contrast": contrast,
                "arm0_record_id": rows[0]["record_id"],
                "arm1_record_id": rows[1]["record_id"],
            }
        )
    selected = []
    for key, rows in sorted(strata.items(), key=lambda item: str(item[0])):
        ordered = sorted(rows, key=lambda row: pair_hash(row["pair_id"]))
        if len(ordered) < PER_STRATUM:
            raise RuntimeError(f"confirmation stratum underfilled: {key}/{len(ordered)}")
        selected.extend(ordered[:PER_STRATUM])
    if len(selected) != 288:
        raise RuntimeError(f"confirmation count drift: {len(selected)}")
    return selected


def run(batch_size: int, natural_budget: int) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("Phase 998 confirmation requires CUDA")
    cases = read_jsonl(OUT_ROOT / "protocol" / "cases.jsonl")
    trace_selected = read_jsonl(OUT_ROOT / "trace" / "selected_pairs.jsonl")
    trace_summary = json.loads(
        (OUT_ROOT / "trace" / "summary.json").read_text(encoding="utf-8")
    )
    channel_sets = json.loads(
        (OUT_ROOT / "trace" / "channel_sets.json").read_text(encoding="utf-8")
    )
    selected = select_confirmation(
        cases, {row["pair_id"] for row in trace_selected}
    )
    output_root = OUT_ROOT / "confirmation"
    output_root.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_root / "selected_pairs.jsonl", selected)
    case_by_record = {row["record_id"]: row for row in cases}
    chain = trace_summary["selected_chain"]
    event_specs = {}
    for role in ROLE_NAMES:
        event = chain[role]
        metric = trace_summary["selected_event_metrics"][role]
        event_specs[role] = {
            "event": event,
            "depth": int(metric["depth"]),
            "position_role": metric["role"],
            "channels": channel_sets[event],
        }
    directional = directional_pairs(selected, case_by_record, "confirmation", True)
    natural_pairs = directional_pairs(
        selected, case_by_record, "confirmation", False
    )
    protocol = json.loads(
        (OUT_ROOT / "protocol" / "protocol.json").read_text(encoding="utf-8")
    )
    candidate_ids = {
        color: int(protocol["candidate_token_ids"][color]) for color in COLORS
    }
    model = tokenizer = None
    started = time.time()
    try:
        model, tokenizer, device = load_model(MODEL, dtype=torch.bfloat16, use_8bit=False)
        tokenizer.padding_side = "left"
        layers = get_layers(model)
        info = get_model_info(model, MODEL)
        effective_eos = eos_ids(model, tokenizer)
        causal_rows, restoration_rows = run_candidate_and_restoration(
            model,
            layers,
            device,
            directional,
            event_specs,
            candidate_ids,
            batch_size,
        )
        candidate = summarize_conditions(causal_rows)
        mediation = summarize_mediation(causal_rows)
        natural_rows = run_natural_holdout(
            model,
            layers,
            tokenizer,
            device,
            natural_pairs,
            event_specs,
            candidate_ids,
            effective_eos,
            batch_size,
            natural_budget,
        )
        natural = natural_summary(natural_rows)
        checks, metrics = final_gate(
            candidate, natural, restoration_rows, mediation
        )
        summary = {
            "schema_version": "phase998_confirmation_summary.v1",
            "phase": PHASE,
            "model": MODEL,
            "selected_pair_count": len(selected),
            "candidate_direction_count": len(directional),
            "natural_pair_count": len(natural_pairs),
            "selection_disjoint_from_trace": True,
            "selection_uses_frozen_chain_and_channels": True,
            "selected_chain": chain,
            "candidate_condition_summary": candidate,
            "natural_condition_summary": natural,
            "mediation_summary": mediation,
            "restoration_summary": {
                "n": len(restoration_rows),
                "median_recovery_fraction": float(
                    __import__("numpy").median(
                        [row["recovery_fraction"] for row in restoration_rows]
                    )
                ),
                "restored_to_source_rate": float(
                    __import__("numpy").mean(
                        [row["restored_to_source"] for row in restoration_rows]
                    )
                ),
            },
            "thresholds": THRESHOLDS,
            "gate_metrics": metrics,
            "gate_checks": checks,
            "causal_thread_gate_pass": all(checks.values()),
            "n_layers": info.n_layers,
            "d_model": info.d_model,
            "elapsed_seconds": time.time() - started,
        }
        write_rows(output_root / "causal_rows.jsonl", causal_rows)
        write_rows(output_root / "restoration_rows.jsonl", restoration_rows)
        write_rows(output_root / "natural_rows.jsonl", natural_rows)
        write_json(output_root / "summary.json", summary)
        return summary
    finally:
        if model is not None:
            release_model(model)
        model = tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--natural-max-new-tokens", type=int, default=8)
    args = parser.parse_args()
    result = run(args.batch_size, args.natural_max_new_tokens)
    print(
        json.dumps(
            {
                "passed": result["causal_thread_gate_pass"],
                "selected_pair_count": result["selected_pair_count"],
                "gate_metrics": result["gate_metrics"],
                "gate_checks": result["gate_checks"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
