#!/usr/bin/env python3
"""C155: checkpoint transfer curve for the frozen C154 predicted response field."""
from __future__ import annotations

import gc
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
OUT = RESULT / "phase1689_c155_checkpoint_transfer_curve"
C153 = RESULT / "phase1687_c153_type_graph_conditional_pool_confirmation"
C154 = RESULT / "phase1688_c154_type_graph_hiddenstate_causal_adjudication"
PUBLIC = ROOT / "frontend/public/vis_data/research_kernel/c109_role_state_field_atlas.json"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core
from phase1332_bf16_utils import load_bf16, quantization_audit, release_bf16
import phase1572_c099_fixed_width_graph_field_campaign as fixed_base
import phase1675_c141_multifamily_full_coordinate_atlas as c141

PHASE, CAMPAIGN = 1689, "C155"
CHECKPOINTS = tuple(range(24, 35))
DIM, WIDTH, BATCH = 2560, 224, 4
ROLES = c141.ROLES


def now():
    return datetime.now(timezone.utc).isoformat()


def contract():
    if OUT.exists():
        raise RuntimeError(OUT)
    parent = core.load(C154 / "audit/independent_closure_audit.json")
    index = core.rows(C154 / "material/intervention_index.jsonl")
    vectors = np.load(C154 / "material/predicted_delta.float32.npy", mmap_mode="r")
    checks = {
        "authorization": parent["all_checks_passed"] and parent["authorization"] == "memo_and_campaign_synthesis",
        "interventions": len(index) == 128,
        "vector_shape": list(vectors.shape) == [128, 6, DIM],
        "checkpoints": CHECKPOINTS == tuple(range(24, 35)),
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    OUT.mkdir(parents=True)
    protocol = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "created_at_utc": now(),
        "status": "checkpoint_transfer_contract_frozen",
        "vector": "unchanged C154 predicted q32 six-role field",
        "checkpoints": list(CHECKPOINTS),
        "readouts": ["mean matched-donor margin gain", "positive-gain rate", "matched-donor choice rate"],
        "classification": {
            "local_peak": "post-block-32 has the largest mean gain",
            "broad_portable": "at least 6 checkpoints have positive mean gain and donor-choice increase >= 0.10",
            "early_max": "largest mean gain occurs before post-block-32",
        },
        "claim_boundary": "checkpoint response map for one frozen field; no new direction search and no unique-circuit claim",
        "forbidden": ["attention", "MLP", "weights", "PCA", "checkpoint-dependent rescaling"],
        "source_hashes": {"C154_vector": core.sha(C154 / "material/predicted_delta.float32.npy"), "C154_index": core.sha(C154 / "material/intervention_index.jsonl")},
        "producer_sha256": core.sha(Path(__file__)),
        "authorization": "run_C155_qwen",
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": all(checks.values()), "authorization": protocol["authorization"]})
    print(json.dumps(protocol, indent=2))


def tensor(value):
    return value[0] if isinstance(value, tuple) else value


@torch.inference_mode()
def run():
    all_rows = core.rows(C153 / "compiled/qwen3.jsonl")
    index = core.rows(C154 / "material/intervention_index.jsonl")
    rows = [all_rows[r["base_row_index"]] for r in index]
    vectors = np.load(C154 / "material/predicted_delta.float32.npy", mmap_mode="r")
    modes = ("baseline",) + tuple(f"state_{q}" for q in CHECKPOINTS)
    scores = np.zeros((len(modes), 128, 2), np.float32)
    model = None
    try:
        model, tokenizer, device, _placement = load_bf16("qwen3")
        quant = quantization_audit(model)
        layers = model.model.layers
        pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)

        def forward(batch, batch_indices, state):
            ids, mask, pos, lengths = fixed_base.fixed_batch(batch, pad, device, WIDTH)
            handle = None
            if state is not None:
                values = torch.from_numpy(np.asarray(vectors[batch_indices])).to(device=device, dtype=torch.float32)

                def patch(_module, _args, output):
                    hidden = tensor(output)
                    patched = hidden.clone()
                    for local, row in enumerate(batch):
                        for role_index, role in enumerate(ROLES):
                            delta = values[local, role_index].to(dtype=patched.dtype)
                            for position in row["role_positions"][role]:
                                patched[local, position] += delta
                    return (patched,) + output[1:] if isinstance(output, tuple) else patched

                handle = layers[state - 1].register_forward_hook(patch)
            try:
                output = model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False, return_dict=True)
            finally:
                if handle is not None:
                    handle.remove()
            return np.asarray([[float(output.logits[i, lengths[i] - 1, candidate[0]]) for candidate in row["candidate_ids"]] for i, row in enumerate(batch)], np.float32)

        for mode_index, mode in enumerate(modes):
            state = None if mode == "baseline" else int(mode.rsplit("_", 1)[1])
            for start in range(0, 128, BATCH):
                ids = np.arange(start, min(start + BATCH, 128))
                scores[mode_index, ids] = forward(rows[start:start + BATCH], ids, state)
            print(f"[C155] {mode_index + 1}/{len(modes)} {mode}", flush=True)
    finally:
        if model is not None:
            release_bf16(model)
        gc.collect(); torch.cuda.empty_cache()
    (OUT / "raw").mkdir(parents=True, exist_ok=True)
    np.save(OUT / "raw/checkpoint_candidate_logits.float32.npy", scores)
    donor_positions = np.asarray([r["donor_gold_position"] for r in index], np.int64)
    margins = np.asarray([[score[i, donor_positions[i]] - score[i, 1 - donor_positions[i]] for i in range(128)] for score in scores], np.float32)
    gains = margins - margins[0]
    baseline_choice = np.argmax(scores[0], axis=1)
    rows_out = []
    for offset, state in enumerate(CHECKPOINTS, start=1):
        rows_out.append({
            "state": state,
            "mean_gain": float(np.mean(gains[offset])),
            "positive_gain_rate": float(np.mean(gains[offset] > 0)),
            "donor_choice_rate": float(np.mean(np.argmax(scores[offset], axis=1) == donor_positions)),
            "donor_choice_increase": float(np.mean(np.argmax(scores[offset], axis=1) == donor_positions) - np.mean(baseline_choice == donor_positions)),
            "stratum_mean_gain": {f"f2={f2},f3={f3}": float(np.mean(gains[offset, [i for i, row in enumerate(index) if row["f2"] == f2 and row["f3"] == f3]])) for f2 in (1, -1) for f3 in (1, -1)},
        })
    best = max(rows_out, key=lambda row: row["mean_gain"])["state"]
    broad_count = sum(row["mean_gain"] > 0 and row["donor_choice_increase"] >= 0.10 for row in rows_out)
    classifications = {"local_peak": best == 32, "broad_portable": broad_count >= 6, "early_max": best < 32}
    report = {"phase": PHASE, "campaign": CAMPAIGN, "created_at_utc": now(), "status": "checkpoint_transfer_curve_adjudicated", "baseline_donor_choice_rate": float(np.mean(baseline_choice == donor_positions)), "checkpoint_rows": rows_out, "best_state": best, "broad_checkpoint_count": broad_count, "classifications": classifications, "numeric_checks": {"shape": list(scores.shape) == [12, 128, 2], "finite": bool(np.isfinite(scores).all()), "bf16": quant["has_bf16_parameters"] and not quant["has_quantized_modules"]}, "claim_boundary": core.load(OUT / "protocol/preregistration.json")["claim_boundary"], "authorization": "close_C155"}
    core.save(OUT / "analysis/transfer_curve.json", report)
    core.save(OUT / "audit/internal_run_audit.json", {"checks": {**report["numeric_checks"], "rows": len(rows_out) == 11, "single_best": sum(row["state"] == best for row in rows_out) == 1}, "all_checks_passed": all(report["numeric_checks"].values()) and len(rows_out) == 11, "authorization": report["authorization"]})
    print(json.dumps({"best_state": best, "broad_count": broad_count, "classifications": classifications, "rows": rows_out}, indent=2))


def close():
    report = core.load(OUT / "analysis/transfer_curve.json")
    payload = core.load(PUBLIC)
    payload["c155_checkpoint_transfer"] = report
    payload.update({"phase": PHASE, "campaign": "C109-C155", "title": "Role-State Atlas + Type-Graph Causal Checkpoint Transfer", "created_at_utc": now()})
    canonical = OUT / "analysis/c109_c155_atlas.json"
    core.save(canonical, payload)
    shutil.copyfile(canonical, PUBLIC)
    checks = {"contract": core.load(OUT / "audit/internal_contract_audit.json")["all_checks_passed"], "run": core.load(OUT / "audit/internal_run_audit.json")["all_checks_passed"], "asset": core.sha(canonical) == core.sha(PUBLIC)}
    closure = {"phase": PHASE, "campaign": CAMPAIGN, "status": "checkpoint_transfer_closed", "best_state": report["best_state"], "broad_checkpoint_count": report["broad_checkpoint_count"], "classifications": report["classifications"], "claim_boundary": report["claim_boundary"], "next_authorization": "campaign_synthesis_and_external_natural_graph_contract"}
    core.save(OUT / "analysis/closure.json", closure)
    core.save(OUT / "audit/internal_closure_audit.json", {"checks": checks, "all_checks_passed": all(checks.values()), "asset_sha256": core.sha(PUBLIC), "authorization": "independent_final_and_memo"})
    print(json.dumps(closure, indent=2))


def main():
    modes = {"contract": contract, "run": run, "close": close}
    if len(sys.argv) != 2 or sys.argv[1] not in modes:
        raise SystemExit("contract|run|close")
    modes[sys.argv[1]]()


if __name__ == "__main__":
    main()
