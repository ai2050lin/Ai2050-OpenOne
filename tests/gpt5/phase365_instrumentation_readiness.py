#!/usr/bin/env python3
"""Validate synthetic adapters and real checkpoint references before any model execution."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from transformers.models.glm.configuration_glm import GlmConfig
from transformers.models.glm.modeling_glm import GlmMLP
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from model_registry import get_model_spec  # noqa: E402
from phase365_dynamic_flow_instrumentation import (  # noqa: E402
    decompose_mlp_input, direct_mlp_output, relative_error,
    replay_mlp_from_neuron_writes, schema_payload, validate_blind_bundle,
)


OUT = ROOT / "tests/gpt5/result/phase365_dynamic_flow_instrumentation/schema_and_adapter_gate"
MODELS = ("qwen3", "glm4", "deepseek7b")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def sample_bundle() -> dict[str, Any]:
    vector_ref = {
        "relative_path": "private/sample.pt", "sha256": "0" * 64,
        "dtype": "float16", "shape": [1, 1, 8], "slice": [0, 1],
    }
    return {
        "schema_version": "42.0.0", "bundle_id": "sample", "anonymous_case_id": "sample_case",
        "anonymous_model_id": "sample_model", "anonymous_condition_slot": "slot_0",
        "split": "blind_discovery",
        "events": [{
            "event_id": "e0", "event_type": "mlp_neuron_write", "generation_time": 0,
            "layer_index": 0, "receiver_role": "query", "channel_id": 0,
            "vector_ref": vector_ref, "raw_event_retained": True,
        }],
        "edges": [],
    }


def main() -> None:
    torch.manual_seed(365)
    hidden = torch.randn(2, 3, 8)
    modules = {
        "qwen3": Qwen3MLP(Qwen3Config(hidden_size=8, intermediate_size=12, hidden_act="silu")),
        "glm4": GlmMLP(GlmConfig(hidden_size=8, intermediate_size=12, hidden_act="silu")),
        "deepseek7b": Qwen2MLP(Qwen2Config(hidden_size=8, intermediate_size=12, hidden_act="silu")),
    }
    adapter_rows = []
    for model, mlp in modules.items():
        parts = decompose_mlp_input(model, mlp, hidden)
        actual = mlp(hidden)
        direct = direct_mlp_output(parts)
        replayed = replay_mlp_from_neuron_writes(parts, chunk_size=5)
        adapter_rows.append({
            "model": model,
            "adapter_kind": parts.adapter_kind,
            "intermediate_size": int(parts.product.shape[-1]),
            "direct_relative_error": relative_error(actual, direct),
            "neuron_write_relative_error": relative_error(actual, replayed),
            "gate_pass": bool(
                relative_error(actual, direct) <= 1e-6
                and relative_error(actual, replayed) <= 1e-6
            ),
        })

    checkpoint_rows = []
    for model in MODELS:
        spec = get_model_spec(model)
        config = json.loads((spec.local_dir / "config.json").read_text(encoding="utf-8"))
        index_path = spec.local_dir / "model.safetensors.index.json"
        index = json.loads(index_path.read_text(encoding="utf-8"))
        down_keys = sorted(key for key in index["weight_map"] if key.endswith("mlp.down_proj.weight"))
        files = sorted({index["weight_map"][key] for key in down_keys})
        checkpoint_rows.append({
            "model": model,
            "private_model_type": config["model_type"],
            "expected_layer_count": int(config["num_hidden_layers"]),
            "down_projection_reference_count": len(down_keys),
            "all_layer_references_present": len(down_keys) == int(config["num_hidden_layers"]),
            "checkpoint_index_relative_path": str(index_path.relative_to(ROOT)),
            "checkpoint_index_sha256": sha256_file(index_path),
            "checkpoint_shard_files": files,
            "all_checkpoint_shards_exist": all((spec.local_dir / value).exists() for value in files),
            "first_parameter_path": down_keys[0],
        })

    schema = schema_payload()
    schema_errors = validate_blind_bundle(sample_bundle())
    summary = {
        "schema_version": "42.1.0", "phase_id": "Phase365", "created_at": now(),
        "denominator": {
            "model_adapter_count": len(adapter_rows),
            "real_checkpoint_reference_model_count": len(checkpoint_rows),
            "dynamic_bundle_schema_count": 1,
            "new_model_execution_count": 0,
        },
        "results": {
            "synthetic_adapter_gate_model_count": sum(row["gate_pass"] for row in adapter_rows),
            "real_checkpoint_down_reference_gate_model_count": sum(
                row["all_layer_references_present"] and row["all_checkpoint_shards_exist"] for row in checkpoint_rows
            ),
            "blind_bundle_schema_gate_pass": not schema_errors,
            "blind_bundle_schema_errors": schema_errors,
            "target_specific_competition_blocked_during_discovery": not schema["blind_discovery"]["target_specific_competition_in_event"],
            "typed_alignment_required_before_contrast": schema["condition_contrast"]["typed_event_alignment_required"],
            "raw_event_retention_required": schema["public_backbone"]["raw_events_must_be_retained"],
            "repeat_noise_floor_available": False,
        },
        "authorization": {
            "full_96_case_engineering_run_authorized": False,
            "six_run_repeat_noise_format_gate_authorized": True,
            "repeat_noise_format_gate": {
                "model_order": list(MODELS),
                "fixed_case_count_per_model": 1,
                "repeat_count_per_model": 2,
                "total_forward_run_count": 6,
                "selected_layer_roles": ["early", "middle", "late"],
                "causal_intervention": False,
            },
        },
        "claim_boundary": {
            "synthetic_adapter_test_is_real_model_execution": False,
            "real_checkpoint_weights_loaded": False,
            "language_mechanism_tested": False,
            "physical_confirmation_opened": False,
        },
        "next_decision": "run_six_repeat_noise_format_forwards_sequentially_qwen3_glm4_deepseek7b",
    }
    write_jsonl(OUT / "phase365_synthetic_adapter_rows.jsonl", adapter_rows)
    write_jsonl(OUT / "phase365_checkpoint_reference_rows.jsonl", checkpoint_rows)
    write_json(OUT / "phase365_instrumentation_readiness_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
