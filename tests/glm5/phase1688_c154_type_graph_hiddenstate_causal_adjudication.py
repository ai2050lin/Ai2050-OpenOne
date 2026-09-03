#!/usr/bin/env python3
"""C154: typed HiddenState causal adjudication for the C153 transition object."""
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
OUT = RESULT / "phase1688_c154_type_graph_hiddenstate_causal_adjudication"
C152 = RESULT / "phase1686_c152_type_graph_transition_object_discovery"
C153 = RESULT / "phase1687_c153_type_graph_conditional_pool_confirmation"
PUBLIC = ROOT / "frontend/public/vis_data/research_kernel/c109_role_state_field_atlas.json"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core
from phase1332_bf16_utils import load_bf16, quantization_audit, release_bf16
import phase1572_c099_fixed_width_graph_field_campaign as fixed_base
import phase1661_c127_typed_transition_language_family as c127
import phase1675_c141_multifamily_full_coordinate_atlas as c141
import phase1677_c143_transition_model_competition as c143

PHASE, CAMPAIGN = 1688, "C154"
Q, TARGET_STATE = 31, 32
DIM, WIDTH, BATCH = 2560, 224, 4
ROLES = c141.ROLES
MODES = ("baseline", "predicted", "observed_mean", "matched_donor", "reverse", "wrong_role", "wrong_coordinate", "wrong_checkpoint")


def now():
    return datetime.now(timezone.utc).isoformat()


def prepare_vectors():
    rows = core.rows(C153 / "compiled/qwen3.jsonl")
    raw = np.load(C153 / "raw/qwen3_window_role_field.bf16.npy", mmap_mode="r")
    trajectories = np.load(C153 / "analysis/fresh_conditional_trajectories.float32.npy", mmap_mode="r")
    keys = core.rows(C153 / "analysis/fresh_conditional_index.jsonl")
    train = np.load(C152 / "analysis/train_conditional_trajectories.float32.npy", mmap_mode="r")
    x_train = train[:, Q].reshape(len(train), -1)
    y_train = (train[:, Q + 1] - train[:, Q]).reshape(len(train), -1)
    x = trajectories[:, Q - 24].reshape(len(trajectories), -1)
    pred_y = c143.fit_predict("linear_kernel", x_train, y_train, x, 0.01)
    predicted_delta = 2.0 * (x + pred_y).reshape(len(trajectories), 6, DIM)
    observed_delta = 2.0 * trajectories[:, TARGET_STATE - 24]
    key_lookup = {(k["unit"], k["surface"], k["code"], k["f2"], k["f3"]): i for i, k in enumerate(keys)}
    row_lookup = {}
    for i, row in enumerate(rows):
        f = row["factors"]
        unit = int(row["unit_id"].rsplit("-", 1)[1])
        row_lookup[(unit, row["surface_factor"], row["codebook_factor"], f["f1"], f["f2"], f["f3"])] = i
    interventions, exact = [], []
    state_index = TARGET_STATE - 24
    for base_index, row in enumerate(rows):
        f = row["factors"]
        if f["f1"] != -1:
            continue
        unit = int(row["unit_id"].rsplit("-", 1)[1])
        group = key_lookup[(unit, row["surface_factor"], row["codebook_factor"], f["f2"], f["f3"])]
        donor_index = row_lookup[(unit, row["surface_factor"], row["codebook_factor"], 1, f["f2"], f["f3"])]
        donor = rows[donor_index]
        exact.append(c127.decode(raw[donor_index, :, state_index]) - c127.decode(raw[base_index, :, state_index]))
        interventions.append({
            "intervention_index": len(interventions),
            "base_row_index": base_index,
            "donor_row_index": donor_index,
            "trajectory_index": group,
            "case_id": row["case_id"],
            "unit_id": row["unit_id"],
            "f2": f["f2"],
            "f3": f["f3"],
            "surface": row["surface_factor"],
            "code": row["codebook_factor"],
            "donor_gold_position": donor["gold_position"],
        })
    pred = np.asarray([predicted_delta[r["trajectory_index"]] for r in interventions], np.float32)
    observed = np.asarray([
        observed_delta[[i for i, key in enumerate(keys) if key["f2"] == row["f2"] and key["f3"] == row["f3"]]].mean(0)
        for row in interventions
    ], np.float32)
    exact = np.asarray(exact, np.float32)
    return interventions, pred, observed, exact


def contract():
    refreeze = OUT.exists()
    if refreeze and ((OUT / "raw").exists() or (OUT / "analysis").exists()):
        raise RuntimeError("C154 cannot be refrozen after model execution or analysis")
    parent = core.load(C153 / "audit/independent_closure_audit.json")
    confirmation = core.load(C153 / "analysis/confirmation.json")
    interventions, predicted, observed, exact = prepare_vectors()
    checks = {
        "authorization": parent["all_checks_passed"] and parent["scientific_gate_passed"],
        "best_transition": max(confirmation["transition_rows"], key=lambda r: r["target"]["cosine"])["q"] == Q,
        "interventions": len(interventions) == 128,
        "shapes": predicted.shape == observed.shape == exact.shape == (128, 6, DIM),
        "finite": bool(np.isfinite(predicted).all() and np.isfinite(observed).all() and np.isfinite(exact).all()),
        "positive_controls_distinct": not np.array_equal(observed, exact),
        "modes": len(MODES) == 8,
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    OUT.mkdir(parents=True, exist_ok=True)
    if refreeze and (OUT / "protocol/preregistration.json").is_file():
        core.save(OUT / "audit/pre_run_contract_correction.json", {"reason": "fully conditioned observed mean was identical to matched donor; replaced by cross-unit f2/f3 structural mean", "old_protocol": core.load(OUT / "protocol/preregistration.json"), "corrected_before_model_run": True, "created_at_utc": now()})
    core.write_rows(OUT / "material/intervention_index.jsonl", interventions)
    np.save(OUT / "material/predicted_delta.float32.npy", predicted)
    np.save(OUT / "material/observed_mean_delta.float32.npy", observed)
    np.save(OUT / "material/matched_donor_delta.float32.npy", exact)
    protocol = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "created_at_utc": now(),
        "status": "typed_hiddenstate_causal_contract_refrozen_before_run" if refreeze else "typed_hiddenstate_causal_contract_frozen",
        "checkpoint": "post_block_31_pre_final_norm to post_block_32_pre_final_norm",
        "patch_state": TARGET_STATE,
        "wrong_checkpoint_state": 24,
        "roles": list(ROLES),
        "modes": list(MODES),
        "interventions": 128,
        "readout": "change in matched f1=+1 donor candidate logit margin from the f1=-1 base",
        "gates": {
            "predicted_mean_gain_min": 0.0,
            "predicted_positive_gain_rate_min": 0.60,
            "predicted_donor_choice_increase_min": 0.10,
            "each_stratum_mean_gain_min": 0.0,
            "paired_win_rate_over_each_wrong_control_min": 0.60,
            "observed_and_donor_positive_control_mean_gain_min": 0.0,
        },
        "claim_boundary": "causal use of a six-role mean HiddenState response field at one checkpoint; not minimal, unique, or parameter-level",
        "forbidden": ["attention", "MLP", "weights", "PCA", "post-unblind alpha search"],
        "source_hashes": {"C153": core.sha(C153 / "analysis/confirmation.json"), "predicted": core.sha(OUT / "material/predicted_delta.float32.npy")},
        "producer_sha256": core.sha(Path(__file__)),
        "authorization": "run_C154_qwen_causal",
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": all(checks.values()), "authorization": protocol["authorization"]})
    print(json.dumps({"checks": checks, "vector_norms": {"predicted": float(np.linalg.norm(predicted)), "observed": float(np.linalg.norm(observed)), "matched_donor": float(np.linalg.norm(exact))}}, indent=2))


def tensor(value):
    return value[0] if isinstance(value, tuple) else value


@torch.inference_mode()
def run():
    protocol = core.load(OUT / "protocol/preregistration.json")
    all_rows = core.rows(C153 / "compiled/qwen3.jsonl")
    index = core.rows(OUT / "material/intervention_index.jsonl")
    rows = [all_rows[r["base_row_index"]] for r in index]
    (OUT / "raw").mkdir(parents=True, exist_ok=True)
    vectors = {
        "predicted": np.load(OUT / "material/predicted_delta.float32.npy", mmap_mode="r"),
        "observed_mean": np.load(OUT / "material/observed_mean_delta.float32.npy", mmap_mode="r"),
        "matched_donor": np.load(OUT / "material/matched_donor_delta.float32.npy", mmap_mode="r"),
    }
    scores = np.zeros((len(MODES), 128, 2), np.float32)
    model = None
    try:
        model, tokenizer, device, _placement = load_bf16("qwen3")
        quant = quantization_audit(model)
        layers = model.model.layers
        pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)

        def forward(batch, batch_indices, mode):
            ids, mask, pos, lengths = fixed_base.fixed_batch(batch, pad, device, WIDTH)
            handle = None
            if mode != "baseline":
                source_mode = mode
                if mode in ("reverse", "wrong_role", "wrong_coordinate", "wrong_checkpoint"):
                    source_mode = "predicted"
                values = torch.from_numpy(np.asarray(vectors[source_mode][batch_indices])).to(device=device, dtype=torch.float32)
                if mode == "reverse":
                    values = -values
                elif mode == "wrong_role":
                    values = torch.roll(values, 1, dims=1)
                elif mode == "wrong_coordinate":
                    values = torch.roll(values, 1, dims=2)
                target_state = 24 if mode == "wrong_checkpoint" else TARGET_STATE

                def patch(_module, _args, output):
                    hidden = tensor(output)
                    patched = hidden.clone()
                    for local, row in enumerate(batch):
                        for role_index, role in enumerate(ROLES):
                            delta = values[local, role_index].to(dtype=patched.dtype)
                            for position in row["role_positions"][role]:
                                patched[local, position] += delta
                    return (patched,) + output[1:] if isinstance(output, tuple) else patched

                handle = layers[target_state - 1].register_forward_hook(patch)
            try:
                output = model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False, return_dict=True)
            finally:
                if handle is not None:
                    handle.remove()
            return np.asarray([[float(output.logits[i, lengths[i] - 1, candidate[0]]) for candidate in row["candidate_ids"]] for i, row in enumerate(batch)], np.float32)

        for mode_index, mode in enumerate(MODES):
            for start in range(0, 128, BATCH):
                ids = np.arange(start, min(start + BATCH, 128))
                scores[mode_index, ids] = forward(rows[start:start + BATCH], ids, mode)
            print(f"[C154] {mode_index + 1}/{len(MODES)} {mode}", flush=True)
    finally:
        if model is not None:
            release_bf16(model)
        gc.collect(); torch.cuda.empty_cache()
    np.save(OUT / "raw/intervention_candidate_logits.float32.npy", scores)
    baseline = scores[0]
    donor_positions = np.asarray([r["donor_gold_position"] for r in index], np.int64)
    margins = np.asarray([[score[i, donor_positions[i]] - score[i, 1 - donor_positions[i]] for i in range(128)] for score in scores], np.float32)
    gains = margins - margins[0]
    baseline_choice = np.argmax(baseline, axis=1)
    mode_reports = {}
    for mode_index, mode in enumerate(MODES):
        per_stratum = {f"f2={f2},f3={f3}": float(np.mean(gains[mode_index, [i for i, row in enumerate(index) if row["f2"] == f2 and row["f3"] == f3]])) for f2 in (1, -1) for f3 in (1, -1)}
        mode_reports[mode] = {
            "mean_margin": float(np.mean(margins[mode_index])),
            "mean_gain": float(np.mean(gains[mode_index])),
            "positive_gain_rate": float(np.mean(gains[mode_index] > 0)),
            "donor_choice_rate": float(np.mean(np.argmax(scores[mode_index], axis=1) == donor_positions)),
            "donor_choice_increase": float(np.mean(np.argmax(scores[mode_index], axis=1) == donor_positions) - np.mean(baseline_choice == donor_positions)),
            "stratum_mean_gain": per_stratum,
        }
    controls = ("reverse", "wrong_role", "wrong_coordinate", "wrong_checkpoint")
    paired = {mode: float(np.mean(gains[MODES.index("predicted")] > gains[MODES.index(mode)])) for mode in controls}
    g = protocol["gates"]
    predicted = mode_reports["predicted"]
    gates = {
        "positive_gain": predicted["mean_gain"] > g["predicted_mean_gain_min"],
        "positive_rate": predicted["positive_gain_rate"] >= g["predicted_positive_gain_rate_min"],
        "choice_increase": predicted["donor_choice_increase"] >= g["predicted_donor_choice_increase_min"],
        "stratum_breadth": all(value > g["each_stratum_mean_gain_min"] for value in predicted["stratum_mean_gain"].values()),
        "wrong_controls": all(value >= g["paired_win_rate_over_each_wrong_control_min"] for value in paired.values()),
        "positive_controls": mode_reports["observed_mean"]["mean_gain"] > g["observed_and_donor_positive_control_mean_gain_min"] and mode_reports["matched_donor"]["mean_gain"] > g["observed_and_donor_positive_control_mean_gain_min"],
    }
    report = {"phase": PHASE, "campaign": CAMPAIGN, "created_at_utc": now(), "status": "typed_hiddenstate_causal_adjudicated", "numeric_checks": {"shape": list(scores.shape) == [8, 128, 2], "finite": bool(np.isfinite(scores).all()), "bf16": quant["has_bf16_parameters"] and not quant["has_quantized_modules"]}, "mode_reports": mode_reports, "paired_win_rates": paired, "gates": gates, "causal_gate_passed": all(gates.values()), "claim_boundary": protocol["claim_boundary"], "authorization": "close_C154"}
    core.save(OUT / "analysis/causal.json", report)
    checks = {**report["numeric_checks"], "modes": set(mode_reports) == set(MODES), "strata": all(len(r["stratum_mean_gain"]) == 4 for r in mode_reports.values())}
    core.save(OUT / "audit/internal_run_audit.json", {"checks": checks, "all_checks_passed": all(checks.values()), "scientific_gate_passed": all(gates.values()), "authorization": report["authorization"]})
    print(json.dumps({"mode_reports": mode_reports, "paired": paired, "gates": gates}, indent=2))


def close():
    report = core.load(OUT / "analysis/causal.json")
    predicted = np.load(OUT / "material/predicted_delta.float32.npy", mmap_mode="r")
    observed = np.load(OUT / "material/observed_mean_delta.float32.npy", mmap_mode="r")
    donor = np.load(OUT / "material/matched_donor_delta.float32.npy", mmap_mode="r")
    index = core.rows(OUT / "material/intervention_index.jsonl")
    coordinate_rows = []
    for f2 in (1, -1):
        for f3 in (1, -1):
            ids = np.asarray([i for i, row in enumerate(index) if row["f2"] == f2 and row["f3"] == f3])
            name = f"f2={f2},f3={f3}"
            for kind, values in (("predicted_causal_delta", predicted), ("observed_mean_causal_delta", observed), ("matched_donor_causal_delta", donor)):
                coordinate_rows.append({"dataset": "C154", "kind": kind, "role": "boundary", "stratum": name, "checkpoint": "post_block_32_pre_final_norm", "values": values[ids, 5].mean(0).astype(np.float32).tolist()})
    core.save(OUT / "analysis/coordinate_rows.json", coordinate_rows)
    payload = core.load(PUBLIC)
    payload["c154_type_graph_causal"] = {"causal": report, "coordinate_rows": coordinate_rows}
    payload.update({"phase": PHASE, "campaign": "C109-C154", "title": "Role-State Atlas + Type-Graph Predictive and Causal Field", "created_at_utc": now()})
    canonical = OUT / "analysis/c109_c154_atlas.json"
    core.save(canonical, payload)
    shutil.copyfile(canonical, PUBLIC)
    checks = {"contract": core.load(OUT / "audit/internal_contract_audit.json")["all_checks_passed"], "run": core.load(OUT / "audit/internal_run_audit.json")["all_checks_passed"], "coordinates": len(coordinate_rows) == 12 and all(len(r["values"]) == DIM for r in coordinate_rows), "asset": core.sha(canonical) == core.sha(PUBLIC)}
    closure = {"phase": PHASE, "campaign": CAMPAIGN, "status": "typed_hiddenstate_causal_closed", "causal_gate_passed": report["causal_gate_passed"], "headline": {"predicted": report["mode_reports"]["predicted"], "paired_win_rates": report["paired_win_rates"], "positive_controls": {m: report["mode_reports"][m] for m in ("observed_mean", "matched_donor")}}, "claim_boundary": report["claim_boundary"], "next_authorization": "campaign_synthesis_and_natural_graph_extension"}
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
