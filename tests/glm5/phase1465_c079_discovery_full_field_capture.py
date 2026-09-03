#!/usr/bin/env python3
"""Phase1465: raw discovery full-field capture for C079."""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core
from phase1332_bf16_utils import load_bf16, quantization_audit, release_bf16
from phase1392_c062_full_field_camera import make_batch

CONTRACT = TESTS / "result/phase1463_c079_aggregate_observation_contract"
BEHAVIOR = TESTS / "result/phase1464_c079_behavior"
OUT = TESTS / "result/phase1465_c079_discovery_full_field_capture"
BATCH = 12


def ordered_cases(protocol: dict, eligible: list[dict], compiled: dict[str, dict]) -> list[dict]:
    result = []
    for group in eligible:
        if group["partition"] != protocol["capture"]["discovery_partition"]:
            continue
        for surface in protocol["surfaces"]:
            for cell in protocol["cells"]:
                result.append(compiled[group[f"{surface}_{cell}"]])
    return result


@torch.inference_mode()
def capture(cases: list[dict], protocol: dict) -> tuple[list[dict], dict]:
    model = None
    raw_path = OUT / "raw/discovery_role_field.float16.npy"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        model, tok, device, placement = load_bf16("qwen3")
        quant = quantization_audit(model)
        hidden_dim = int(model.config.hidden_size)
        shape = (len(cases), protocol["capture"]["state_count"], protocol["capture"]["role_slot_count"], hidden_dim)
        field = np.lib.format.open_memmap(raw_path, mode="w+", dtype=np.float16, shape=shape)
        pad = int(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
        roles = protocol["role_slots"]
        index = []
        finite = True
        for start in range(0, len(cases), BATCH):
            batch = cases[start:start + BATCH]
            ids, mask, positions, offsets = make_batch(batch, pad, device)
            output = model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False, output_hidden_states=True, return_dict=True)
            if len(output.hidden_states) != protocol["capture"]["state_count"]:
                raise RuntimeError(("state_count", len(output.hidden_states)))
            role_index = torch.tensor([[offsets[row_index] + batch[row_index]["role_positions"][role][0] for role in roles] for row_index in range(len(batch))], dtype=torch.long, device=device)
            batch_index = torch.arange(len(batch), device=device)[:, None]
            block = np.empty((len(batch), len(output.hidden_states), len(roles), hidden_dim), dtype=np.float16)
            for state_index, hidden in enumerate(output.hidden_states):
                gathered = hidden[batch_index, role_index]
                if not bool(torch.isfinite(gathered).all()):
                    finite = False
                block[:, state_index] = gathered.to(dtype=torch.float16, device="cpu").numpy()
            field[start:start + len(batch)] = block
            logits = output.logits[:, -1].float()
            for local, row in enumerate(batch):
                candidate_ids = [values[0] for values in row["candidate_ids"]]
                scores = [float(logits[local, token_id]) for token_id in candidate_ids]
                index.append({
                    "row_index": start + local,
                    "case_id": row["case_id"],
                    "partition": row["partition"],
                    "family": row["family"],
                    "index": row["index"],
                    "surface": row["surface"],
                    "cell": row["cell"],
                    "record_relation_id": row["record_relation_id"],
                    "query_relation_id": row["query_relation_id"],
                    "entity_match": row["entity_match"],
                    "object_match": row["object_match"],
                    "relation_match": row["relation_match"],
                    "truth": row["truth"],
                    "gold_position": row["gold_position"],
                    "capture_scores": scores,
                    "capture_prediction": int(np.argmax(scores)),
                    "role_positions": row["role_positions"],
                })
            del output, ids, mask, positions, role_index, batch_index, block
        field.flush()
        del field
        runtime = {"placement": placement, "quantization": quant, "hidden_dim": hidden_dim, "shape": list(shape), "finite_during_capture": finite}
        return index, runtime
    finally:
        if model is not None:
            release_bf16(model)


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1465 exists")
    behavior = core.load(BEHAVIOR / "analysis/final.json")
    behavior_audit = core.load(BEHAVIOR / "audit/independent_final_audit.json")
    protocol = core.load(CONTRACT / "protocol/preregistration.json")
    if behavior["authorization"] != "run_phase1465_c079_discovery_full_field_capture" or not behavior_audit["all_checks_passed"]:
        raise RuntimeError("Phase1464 did not authorize capture")
    eligible = core.rows(BEHAVIOR / "material/eligible_composition_sets.jsonl")
    compiled = {row["case_id"]: row for row in core.rows(CONTRACT / "compiled/qwen3_active.jsonl")}
    cases = ordered_cases(protocol, eligible, compiled)
    expected = sum(row["partition"] == protocol["capture"]["discovery_partition"] for row in eligible) * len(protocol["surfaces"]) * len(protocol["cells"])
    if len(cases) != expected or expected != 1104:
        raise RuntimeError((len(cases), expected))
    index, runtime = capture(cases, protocol)
    raw_path = OUT / "raw/discovery_role_field.float16.npy"
    core.write_rows(OUT / "raw/discovery_role_field_index.jsonl", index)
    raw_sha = core.sha(raw_path)
    index_sha = core.sha(OUT / "raw/discovery_role_field_index.jsonl")
    behavior_rows = {row["case_id"]: row for row in core.rows(BEHAVIOR / "raw/active_behavior.jsonl")}
    checks = {
        "count": len(index) == expected == runtime["shape"][0],
        "shape": runtime["shape"][1:] == [37, 9, runtime["hidden_dim"]],
        "dtype": np.load(raw_path, mmap_mode="r").dtype == np.float16,
        "discovery_only": all(row["partition"] == "response_discovery" for row in index),
        "eligible_behavior": all(behavior_rows[row["case_id"]]["correct"] for row in index),
        "capture_prediction": all(row["capture_prediction"] == row["gold_position"] for row in index),
        "finite": runtime["finite_during_capture"] and all(math.isfinite(value) for row in index for value in row["capture_scores"]),
        "bf16": runtime["quantization"]["has_bf16_parameters"],
        "not_quantized": not runtime["quantization"]["has_quantized_modules"],
        "no_holdout": not any(row["partition"] in protocol["validation"]["partitions"] for row in index),
    }
    if not all(checks.values()):
        raise RuntimeError({key: value for key, value in checks.items() if not value})
    metadata = {
        "phase": 1465,
        "campaign": "C079",
        "raw_path": str(raw_path.relative_to(ROOT)),
        "index_path": str((OUT / "raw/discovery_role_field_index.jsonl").relative_to(ROOT)),
        "shape": runtime["shape"],
        "dtype": "float16",
        "file_size_bytes": raw_path.stat().st_size,
        "raw_sha256": raw_sha,
        "index_sha256": index_sha,
        "roles": protocol["role_slots"],
        "states": "state0 embedding output plus state1-state36 transformer block outputs",
        "checks": checks,
        "runtime": runtime,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    core.save(OUT / "analysis/capture_metadata.json", metadata)
    core.save(OUT / "analysis/final.json", {"phase": 1465, "campaign": "C079", "capture_qualified": True, "raw_sha256": raw_sha, "authorization": "run_phase1466_c079_discovery_basic_observation_and_freeze"})
    print(json.dumps({key: value for key, value in metadata.items() if key != "runtime"}, indent=2))


if __name__ == "__main__":
    main()
