#!/usr/bin/env python3
"""Phase1506: capture C087 embeddings and all role-aligned Hidden States."""
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
CONTRACT = RESULT / "phase1504_c087_cross_root_semeval_contract"
BEHAVIOR = RESULT / "phase1505_c087_behavior_stratification"
OUT = RESULT / "phase1506_c087_all_case_field_capture"
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
        roles = protocol["roles"]
        dim = int(model.config.hidden_size)
        shape = (len(cases), 37, len(roles), dim)
        field = np.lib.format.open_memmap(raw, mode="w+", dtype=np.float16, shape=shape)
        pad = int(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
        supports = "logits_to_keep" in inspect.signature(model.forward).parameters
        index, finite = [], True
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
                scores = [float(logits[i, candidate[0]]) for candidate in row["candidate_ids"]]
                set_id, stratum = case_strata[row["case_id"]]
                index.append(
                    {
                        "row_index": start + i,
                        "case_id": row["case_id"],
                        "set_id": set_id,
                        "stratum": stratum,
                        "partition": row["partition"],
                        "item": row["item"],
                        "lemma": row["lemma"],
                        "source_instance_id": row["source_instance_id"],
                        "surface": row["surface"],
                        "semantic_match": row["semantic_match"],
                        "semantic_label": row["semantic_label"],
                        "candidate": row["candidate"],
                        "human_votes_here": row["human_votes_here"],
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
        raise RuntimeError("Phase1506 exists")
    behavior_final = core.load(BEHAVIOR / "analysis/final.json")
    behavior_audit = core.load(BEHAVIOR / "audit/independent_final_audit.json")
    protocol = core.load(CONTRACT / "protocol/preregistration.json")
    if (
        behavior_final["authorization"] != "run_phase1506_c087_all_case_field_capture"
        or not behavior_audit["all_checks_passed"]
    ):
        raise RuntimeError("Phase1505 authorization missing")
    compiled = core.rows(CONTRACT / "compiled/qwen3_active.jsonl")
    groups = core.rows(BEHAVIOR / "material/stratified_composition_sets.jsonl")
    keys = tuple(f"{surface}_{label}" for surface in protocol["surfaces"] for label in ("same", "different"))
    case_strata = {
        group[key]: (group["set_id"], group["stratum"])
        for group in groups
        for key in keys
    }
    index, runtime = capture(compiled, protocol, case_strata)
    raw = OUT / "raw/all_role_field.float16.npy"
    index_path = OUT / "raw/all_role_field_index.jsonl"
    core.write_rows(index_path, index)
    behavior = {row["case_id"]: row for row in core.rows(BEHAVIOR / "raw/behavior.jsonl")}
    agreement_count = sum(
        row["capture_prediction"] == behavior[row["case_id"]]["prediction"] for row in index
    )
    max_diff = max(
        abs(value - behavior[row["case_id"]]["scores"][i])
        for row in index
        for i, value in enumerate(row["capture_scores"])
    )
    by_capture = {row["case_id"]: row for row in index}
    stratum_identity = True
    for group in groups:
        n = sum(
            by_capture[group[key]]["capture_prediction"] == by_capture[group[key]]["gold_position"]
            for key in keys
        )
        observed = "success" if n == 4 else "failed" if n == 0 else "mixed"
        stratum_identity = stratum_identity and observed == group["stratum"]
    arr = np.load(raw, mmap_mode="r")
    checks = {
        "count": len(index) == 864,
        "shape": list(arr.shape) == [864, 37, 3, 2560],
        "dtype": arr.dtype == np.float16,
        "stratum_identity": stratum_identity,
        "finite": runtime["finite_during_capture"] and all(math.isfinite(v) for row in index for v in row["capture_scores"]),
        "bf16": runtime["quantization"]["has_bf16_parameters"],
        "not_quantized": not runtime["quantization"]["has_quantized_modules"],
    }
    acquisition_checks = {key: value for key, value in checks.items() if key != "stratum_identity"}
    acquisition_complete = all(acquisition_checks.values())
    execution_identity_gate_passed = checks["stratum_identity"]
    meta = {
        "phase": 1506,
        "campaign": "C087",
        "shape": runtime["shape"],
        "dtype": "float16",
        "axis_order": ["case", "state", "role", "coordinate"],
        "roles": protocol["roles"],
        "states": "state0 embedding output plus state1-state36 block outputs",
        "file_size_bytes": raw.stat().st_size,
        "raw_sha256": core.sha(raw),
        "index_sha256": core.sha(index_path),
        "behavior_score_max_abs_diff": max_diff,
        "behavior_prediction_agreement_count": agreement_count,
        "behavior_prediction_disagreement_count": len(index) - agreement_count,
        "behavior_prediction_agreement_rate": agreement_count / len(index),
        "checks": checks,
        "acquisition_complete": acquisition_complete,
        "execution_identity_gate_passed": execution_identity_gate_passed,
        "evidence_scope": (
            "confirmatory" if execution_identity_gate_passed else
            "descriptive_only_due_to_cross_execution_stratum_mismatch"
        ),
        "runtime": runtime,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    core.save(OUT / "analysis/capture_metadata.json", meta)
    core.save(
        OUT / "analysis/final.json",
        {
            "phase": 1506,
            "campaign": "C087",
            "status": (
                "all_case_field_capture_complete" if execution_identity_gate_passed else
                "capture_complete_execution_identity_gate_failed"
            ),
            "raw_sha256": meta["raw_sha256"],
            "authorization": (
                "run_phase1507_c087_semantic_contrast_atlas" if execution_identity_gate_passed else
                "run_phase1507_c087_descriptive_semantic_contrast_atlas"
            ),
        },
    )
    print(json.dumps({key: value for key, value in meta.items() if key != "runtime"}, indent=2))


if __name__ == "__main__":
    main()
