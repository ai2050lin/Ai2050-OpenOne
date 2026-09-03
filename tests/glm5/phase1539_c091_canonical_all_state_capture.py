#!/usr/bin/env python3
"""Phase1539: canonical all-state capture after C091 behavior qualification."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
CONTRACT = RESULT / "phase1536_c091_human_validated_chinese_relation_contract"
BEHAVIOR = RESULT / "phase1537_c091_behavior_only_qualification"
PARENT = RESULT / "phase1538_c091_behavior_gate_adjudication"
OUT = RESULT / "phase1539_c091_canonical_all_state_capture"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core
from phase1332_bf16_utils import load_bf16, quantization_audit, release_bf16
from phase1537_c091_behavior_only_qualification import make_fixed_batch

ROLES = ("source_word", "target_word", "relation_anchor", "boundary")


@torch.inference_mode()
def run_hidden_batch(model, rows: list[dict], pad: int, device, fixed_length: int):
    ids, mask, position_ids, lengths = make_fixed_batch(rows, pad, device, fixed_length)
    out = model(
        input_ids=ids,
        attention_mask=mask,
        position_ids=position_ids,
        use_cache=False,
        output_hidden_states=True,
        return_dict=True,
    )
    states = len(out.hidden_states)
    hidden_size = int(model.config.hidden_size)
    pooled = np.empty((len(rows), states, len(ROLES), hidden_size), dtype=np.float32)
    values = np.empty((len(rows), 2), dtype=np.float32)
    for state, hidden in enumerate(out.hidden_states):
        for i, row in enumerate(rows):
            for role_index, role in enumerate(ROLES):
                points = torch.tensor(row["role_positions"][role], dtype=torch.long, device=device)
                pooled[i, state, role_index] = hidden[i, points].float().mean(dim=0).cpu().numpy()
    for i, row in enumerate(rows):
        for j, candidate in enumerate(row["candidate_ids"]):
            values[i, j] = float(out.logits[i, lengths[i] - 1, candidate[0]].float().cpu())
    return pooled, values


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1539 exists")
    parent = core.load(PARENT / "analysis/final.json")
    parent_audit = core.load(PARENT / "audit/independent_final_audit.json")
    protocol = core.load(CONTRACT / "protocol/preregistration.json")
    scope = core.load(PARENT / "protocol/frozen_behavior_routes_and_hidden_scope.json")
    if parent["authorization"] != "run_phase1539_c091_canonical_all_state_capture" or not parent_audit["all_checks_passed"]:
        raise RuntimeError("Phase1538 authorization missing")
    compiled = core.rows(CONTRACT / "compiled/qwen3_active.jsonl")
    pairs = core.rows(CONTRACT / "material/frozen_pairs.jsonl")
    behavior_rows = {row["case_id"]: row for row in core.rows(BEHAVIOR / "raw/behavior_logits.jsonl")}
    row_index = {row["case_id"]: i for i, row in enumerate(compiled)}
    grouped = []
    for pair in pairs:
        rows = [row for row in compiled if row["pair_id"] == pair["pair_id"]]
        rows.sort(key=lambda row: (protocol["surfaces"].index(row["surface"]), protocol["families"].index(row["query_family"])))
        grouped.append(rows)
    field_path = OUT / "raw/canonical_all_role_field.float16.npy"
    field_path.parent.mkdir(parents=True, exist_ok=True)
    field = np.lib.format.open_memmap(field_path, mode="w+", dtype=np.float16, shape=(540, 37, 4, 2560))
    scores = {}
    repeat_blocks = []
    model = None
    try:
        model, tok, device, placement = load_bf16("qwen3")
        quant = quantization_audit(model)
        pad = int(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
        fixed_length = int(protocol["execution"]["fixed_global_sequence_length"])
        for batch_index, rows in enumerate(grouped):
            pooled, values = run_hidden_batch(model, rows, pad, device, fixed_length)
            for i, row in enumerate(rows):
                field[row_index[row["case_id"]]] = pooled[i].astype(np.float16)
                scores[row["case_id"]] = values[i].tolist()
            if batch_index < 3:
                repeat_blocks.append((rows, pooled, values))
            if (batch_index + 1) % 15 == 0:
                print(f"[phase1539] completed {batch_index + 1}/{len(grouped)} pair batches", flush=True)
        field.flush()
        repeat_hidden_max_abs = 0.0
        repeat_logit_max_abs = 0.0
        for rows, pooled, values in repeat_blocks:
            again_hidden, again_logits = run_hidden_batch(model, rows, pad, device, fixed_length)
            repeat_hidden_max_abs = max(repeat_hidden_max_abs, float(np.max(np.abs(pooled - again_hidden))))
            repeat_logit_max_abs = max(repeat_logit_max_abs, float(np.max(np.abs(values - again_logits))))
    finally:
        if model is not None:
            release_bf16(model)
    del field
    field = np.load(field_path, mmap_mode="r")

    index = []
    behavior_replay_max_abs = 0.0
    for i, row in enumerate(compiled):
        values = scores[row["case_id"]]
        old = behavior_rows[row["case_id"]]["candidate_logits"]
        behavior_replay_max_abs = max(behavior_replay_max_abs, float(np.max(np.abs(np.asarray(values) - np.asarray(old)))))
        index.append({
            "row_index": i,
            "case_id": row["case_id"],
            "pair_id": row["pair_id"],
            "pair_family": row["pair_family"],
            "query_family": row["query_family"],
            "partition": row["partition"],
            "partition_rank": row["partition_rank"],
            "concreteness": row["concreteness"],
            "surface": row["surface"],
            "source": row["source"],
            "target": row["target"],
            "gold_label": row["gold_label"],
            "candidate_logits": values,
            "role_positions": row["role_positions"],
        })

    postquery_causal_max_abs = 0.0
    source_target_roles = (ROLES.index("source_word"), ROLES.index("target_word"))
    for pair in pairs:
        rows = [row for row in index if row["pair_id"] == pair["pair_id"] and row["surface"] == "postquery"]
        base = rows[0]
        for row in rows[1:]:
            for role_index in source_target_roles:
                left = np.asarray(field[base["row_index"], :, role_index], dtype=np.float32)
                right = np.asarray(field[row["row_index"], :, role_index], dtype=np.float32)
                postquery_causal_max_abs = max(postquery_causal_max_abs, float(np.max(np.abs(left - right))))

    prequery_anchor_max_abs = 0.0
    anchor_role = ROLES.index("relation_anchor")
    for query_family in protocol["families"]:
        rows = [row for row in index if row["surface"] == "prequery" and row["query_family"] == query_family]
        base = rows[0]
        left = np.asarray(field[base["row_index"], :, anchor_role], dtype=np.float32)
        for row in rows[1:]:
            right = np.asarray(field[row["row_index"], :, anchor_role], dtype=np.float32)
            prequery_anchor_max_abs = max(prequery_anchor_max_abs, float(np.max(np.abs(left - right))))

    finite = True
    for start in range(0, len(field), 30):
        if not np.isfinite(np.asarray(field[start:start + 30])).all():
            finite = False
            break
    index_path = OUT / "raw/canonical_all_role_field_index.jsonl"
    core.write_rows(index_path, index)
    numeric_gate = protocol["numeric_gate_before_hidden_use"]
    checks = {
        "shape": list(field.shape) == [540, 37, 4, 2560],
        "finite": finite,
        "coverage": len(index) == 540 and len(scores) == 540,
        "repeat_hidden": repeat_hidden_max_abs <= numeric_gate["repeat_hidden_max_abs"],
        "repeat_logits": repeat_logit_max_abs <= numeric_gate["repeat_logit_max_abs"],
        "behavior_logit_replay": behavior_replay_max_abs <= numeric_gate["behavior_logit_replay_max_abs"],
        "postquery_causal_identity": postquery_causal_max_abs <= numeric_gate["postquery_source_target_causal_max_abs"],
        "prequery_causal_identity": prequery_anchor_max_abs <= numeric_gate["prequery_relation_anchor_causal_max_abs"],
        "bf16": quant["has_bf16_parameters"] and not quant["has_quantized_modules"],
        "semantic_scope": scope["qualified_families"] == ["whole_part"],
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    report = {
        "phase": 1539,
        "campaign": "C091",
        "status": "canonical_all_state_capture_numeric_gate_pass",
        "qualified_families": scope["qualified_families"],
        "field_shape": list(field.shape),
        "repeat_hidden_max_abs": repeat_hidden_max_abs,
        "repeat_logit_max_abs": repeat_logit_max_abs,
        "behavior_logit_replay_max_abs": behavior_replay_max_abs,
        "postquery_source_target_causal_max_abs": postquery_causal_max_abs,
        "prequery_relation_anchor_causal_max_abs": prequery_anchor_max_abs,
        "runtime": {"placement": placement, "quantization": quant},
        "checks": checks,
        "files": {
            "field": {"sha256": core.sha(field_path), "bytes": field_path.stat().st_size},
            "index": {"sha256": core.sha(index_path), "rows": len(index)},
        },
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    core.save(OUT / "analysis/canonical_capture_summary.json", report)
    core.save(OUT / "analysis/final.json", {
        "phase": 1539,
        "campaign": "C091",
        "status": report["status"],
        "authorization": "run_phase1540_c091_discovery_timing_atlas",
    })
    print(json.dumps({key: value for key, value in report.items() if key != "runtime"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
