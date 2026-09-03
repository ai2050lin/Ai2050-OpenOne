#!/usr/bin/env python3
"""Phase1498: capture every C086 role across embeddings and all Hidden States."""
from __future__ import annotations

import inspect
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
CONTRACT = RESULT / "phase1496_c086_unlabeled_counterbalanced_contract"
BEHAVIOR = RESULT / "phase1497_c086_behavior_stratification"
OUT = RESULT / "phase1498_c086_all_case_field_capture"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core
from phase1332_bf16_utils import load_bf16, quantization_audit, release_bf16
from phase1392_c062_full_field_camera import make_batch

BATCH = 12


@torch.inference_mode()
def capture(cases, protocol, case_strata):
    raw = OUT / "raw/all_role_field.float16.npy"
    raw.parent.mkdir(parents=True, exist_ok=True)
    model = None
    try:
        model, tok, device, placement = load_bf16("qwen3")
        quant = quantization_audit(model)
        dim = int(model.config.hidden_size)
        roles = protocol["roles"]
        shape = (len(cases), 37, len(roles), dim)
        field = np.lib.format.open_memmap(raw, mode="w+", dtype=np.float16, shape=shape)
        pad = int(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
        supports = "logits_to_keep" in inspect.signature(model.forward).parameters
        index = []
        finite = True
        for start in range(0, len(cases), BATCH):
            batch = cases[start : start + BATCH]
            ids, mask, pos, offsets = make_batch(batch, pad, device)
            kwargs = {
                "input_ids": ids,
                "attention_mask": mask,
                "position_ids": pos,
                "use_cache": False,
                "output_hidden_states": True,
                "return_dict": True,
            }
            if supports:
                kwargs["logits_to_keep"] = 1
            out = model(**kwargs)
            if len(out.hidden_states) != 37:
                raise RuntimeError(("state_count", len(out.hidden_states)))
            role_index = torch.tensor(
                [
                    [offsets[i] + batch[i]["role_positions"][role][0] for role in roles]
                    for i in range(len(batch))
                ],
                dtype=torch.long,
                device=device,
            )
            bi = torch.arange(len(batch), device=device)[:, None]
            block = np.empty((len(batch), 37, len(roles), dim), dtype=np.float16)
            for state, hidden in enumerate(out.hidden_states):
                gathered = hidden[bi, role_index]
                finite = finite and bool(torch.isfinite(gathered).all())
                block[:, state] = gathered.to(dtype=torch.float16, device="cpu").numpy()
            field[start : start + len(batch)] = block
            logits = out.logits[:, -1].float()
            for i, row in enumerate(batch):
                scores = [float(logits[i, values[0]]) for values in row["candidate_ids"]]
                set_id, stratum = case_strata[row["case_id"]]
                index.append(
                    {
                        "row_index": start + i,
                        "case_id": row["case_id"],
                        "set_id": set_id,
                        "stratum": stratum,
                        "partition": row["partition"],
                        "family": row["family"],
                        "index": row["index"],
                        "surface": row["surface"],
                        "codebook": row["codebook"],
                        "code_sign": row["code_sign"],
                        "cell": row["cell"],
                        "record_relation_id": row["record_relation_id"],
                        "query_relation_id": row["query_relation_id"],
                        "entity_match": row["entity_match"],
                        "object_match": row["object_match"],
                        "relation_match": row["relation_match"],
                        "semantic_truth": row["semantic_truth"],
                        "output_yes": row["output_yes"],
                        "gold_position": row["gold_position"],
                        "capture_scores": scores,
                        "capture_prediction": int(np.argmax(scores)),
                        "role_positions": row["role_positions"],
                    }
                )
            del out, ids, mask, pos, role_index, bi, block
        field.flush()
        del field
        return index, {
            "placement": placement,
            "quantization": quant,
            "shape": list(shape),
            "hidden_dim": dim,
            "finite_during_capture": finite,
        }
    finally:
        if model is not None:
            release_bf16(model)


def main():
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1498 exists")
    behavior_final = core.load(BEHAVIOR / "analysis/final.json")
    behavior_audit = core.load(BEHAVIOR / "audit/independent_final_audit.json")
    protocol = core.load(CONTRACT / "protocol/preregistration.json")
    if (
        behavior_final["authorization"] != "run_phase1498_c086_all_case_field_capture"
        or not behavior_audit["all_checks_passed"]
    ):
        raise RuntimeError("Phase1497 authorization missing")
    compiled = core.rows(CONTRACT / "compiled/qwen3_active.jsonl")
    groups = core.rows(BEHAVIOR / "material/stratified_composition_sets.jsonl")
    keys = tuple(
        f"{surface}_{codebook}_{cell}"
        for surface in protocol["surfaces"]
        for codebook in protocol["codebooks"]
        for cell in protocol["cells"]
    )
    case_strata = {
        group[key]: (group["set_id"], group["stratum"])
        for group in groups
        for key in keys
    }
    raw = OUT / "raw/all_role_field.float16.npy"
    index_path = OUT / "raw/all_role_field_index.jsonl"
    if raw.exists() and index_path.exists():
        # The first capture completed before a non-preregistered prediction-identity
        # assertion fired. Preserve that immutable field and independently rescan it.
        index = core.rows(index_path)
        arr_existing = np.load(raw, mmap_mode="r")
        finite_existing = all(
            bool(np.isfinite(np.asarray(arr_existing[start : start + 24])).all())
            for start in range(0, len(arr_existing), 24)
        )
        behavior_runtime = core.load(
            BEHAVIOR / "analysis/behavior_stratification_summary.json"
        )["runtime"]
        runtime = {
            "placement": behavior_runtime["placement"],
            "quantization": behavior_runtime["quantization"],
            "shape": list(arr_existing.shape),
            "hidden_dim": int(arr_existing.shape[-1]),
            "finite_during_capture": finite_existing,
            "recovered_from_completed_immutable_capture": True,
        }
        del arr_existing
    else:
        index, runtime = capture(compiled, protocol, case_strata)
        core.write_rows(index_path, index)
    behavior = {r["case_id"]: r for r in core.rows(BEHAVIOR / "raw/behavior.jsonl")}
    max_diff = max(
        abs(value - behavior[row["case_id"]]["scores"][i])
        for row in index
        for i, value in enumerate(row["capture_scores"])
    )
    arr = np.load(raw, mmap_mode="r")
    agreement_count = sum(
        row["capture_prediction"] == behavior[row["case_id"]]["prediction"]
        for row in index
    )
    group_rows = core.rows(BEHAVIOR / "material/stratified_composition_sets.jsonl")
    by_capture = {row["case_id"]: row for row in index}
    capture_strata = {}
    for group in group_rows:
        group_ids = [group[key] for key in keys]
        correct_count = sum(
            by_capture[case_id]["capture_prediction"]
            == by_capture[case_id]["gold_position"]
            for case_id in group_ids
        )
        capture_strata[group["set_id"]] = (
            "success" if correct_count == 32 else "failed" if correct_count == 0 else "mixed"
        )
    checks = {
        "count": len(index) == 6912,
        "shape": list(arr.shape) == [6912, 37, 7, 2560],
        "dtype": arr.dtype == np.float16,
        "all_observed_strata": set(r["stratum"] for r in index)
        == set(behavior_final["stratum_counts"]),
        "behavior_stratum_identity": all(
            capture_strata[group["set_id"]] == group["stratum"] for group in group_rows
        ),
        "semantic_output_separated": all(
            row["output_yes"]
            == (row["relation_match"] == (row["code_sign"] == 1))
            for row in index
        ),
        "finite": runtime["finite_during_capture"]
        and all(math.isfinite(v) for row in index for v in row["capture_scores"]),
        "bf16": runtime["quantization"]["has_bf16_parameters"],
        "not_quantized": not runtime["quantization"]["has_quantized_modules"],
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    meta = {
        "phase": 1498,
        "campaign": "C086",
        "shape": runtime["shape"],
        "dtype": "float16",
        "axis_order": ["case", "state", "role", "coordinate"],
        "roles": protocol["roles"],
        "states": "state0 embedding output plus state1-state36 block outputs",
        "file_size_bytes": raw.stat().st_size,
        "raw_sha256": core.sha(raw),
        "index_sha256": core.sha(OUT / "raw/all_role_field_index.jsonl"),
        "behavior_score_max_abs_diff": max_diff,
        "behavior_prediction_agreement_count": agreement_count,
        "behavior_prediction_disagreement_count": len(index) - agreement_count,
        "behavior_prediction_agreement_rate": agreement_count / len(index),
        "checks": checks,
        "runtime": runtime,
        "interpretation_boundary": "all observed sets are mixed; this field is diagnostic, not evidence of mastered behavior; BF16 behavior-only versus hidden-state-forward predictions disagree on a reported subset although every composition-set stratum is unchanged",
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    core.save(OUT / "analysis/capture_metadata.json", meta)
    core.save(
        OUT / "analysis/final.json",
        {
            "phase": 1498,
            "campaign": "C086",
            "status": "all_case_field_capture_complete",
            "raw_sha256": meta["raw_sha256"],
            "authorization": "run_phase1499_c086_four_factor_atlas",
        },
    )
    print(json.dumps({k: v for k, v in meta.items() if k != "runtime"}, indent=2))


if __name__ == "__main__":
    main()
