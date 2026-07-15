#!/usr/bin/env python3
"""Execute one Phase425 condition without touching formal checkpoints."""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from hf_probe_env import get_layers, load_probe_model, release_loaded  # noqa: E402
from phase358_multiresolution_component_conservation import install_hooks  # noqa: E402
from phase371b_anchor_qk_collection import capture_actual_qkv  # noqa: E402
from phase424_global_physical_collect import collect_condition  # noqa: E402
from phase425_role_exchange_collect import add_transport_diagnostics, read_jsonl  # noqa: E402
from phase425_role_exchange_protocol import OUT  # noqa: E402


def main() -> None:
    row = next(
        value
        for value in read_jsonl(OUT / "phase425_registered_conditions_open.jsonl")
        if value["model"] == "qwen3"
    )
    loaded = None
    handles = []
    captures = {}
    try:
        loaded = load_probe_model("qwen3")
        layers = get_layers(loaded.model)
        handles = install_hooks(layers, captures)
        with capture_actual_qkv(loaded.key, tuple(range(len(layers))), captures):
            scalars, _, summary = collect_condition(
                loaded,
                layers,
                captures,
                {**row, "pair_identity": row["condition_key"]},
            )
            add_transport_diagnostics(layers, captures, row, scalars)
        if not all(
            torch.isfinite(torch.tensor(value["source_write_coherence"]))
            for value in scalars
        ):
            raise RuntimeError("non-finite transport diagnostic")
        payload = {
            "valid": True,
            "model": "qwen3",
            "layer_count": len(layers),
            "condition_id": row["condition_id"],
            "branch_correct": summary["branch_correct"],
            "final_target_branch_margin": summary["final_target_branch_margin"],
            "max_component_ledger_relative_error": summary[
                "max_component_ledger_relative_error"
            ],
            "max_source_write_coherence": max(
                value["source_write_coherence"] for value in scalars
            ),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    finally:
        for handle in handles:
            handle.remove()
        release_loaded(loaded)
        captures.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
