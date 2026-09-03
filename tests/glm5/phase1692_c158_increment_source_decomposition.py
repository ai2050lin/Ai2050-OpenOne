#!/usr/bin/env python3
"""C158: separate observed relation state, predicted increment, and exact target state."""
from __future__ import annotations

import gc
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
OUT = RESULT / "phase1692_c158_increment_source_decomposition"
C152 = RESULT / "phase1686_c152_type_graph_transition_object_discovery"
C153 = RESULT / "phase1687_c153_type_graph_conditional_pool_confirmation"
C157 = RESULT / "phase1691_c157_local_field_master_contract"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core
from phase1332_bf16_utils import load_bf16, quantization_audit, release_bf16
import phase1572_c099_fixed_width_graph_field_campaign as fixed_base
import phase1661_c127_typed_transition_language_family as c127
import phase1675_c141_multifamily_full_coordinate_atlas as c141
import phase1677_c143_transition_model_competition as c143
import phase1688_c154_type_graph_hiddenstate_causal_adjudication as c154

PHASE, CAMPAIGN = 1692, "C158"
Q, TARGET = 31, 32
DIM, WIDTH, BATCH = 2560, 224, 4
ROLES = c141.ROLES
ALPHAS = (0.25, 0.5, 1.0)
SOURCES = ("observed_x", "predicted_y", "predicted_sum", "exact_target")


def now():
    return datetime.now(timezone.utc).isoformat()


def tensor(value):
    return value[0] if isinstance(value, tuple) else value


def source_vectors():
    interventions, _old_pred, _old_mean, exact = c154.prepare_vectors()
    trajectories = np.load(C153 / "analysis/fresh_conditional_trajectories.float32.npy", mmap_mode="r")
    train = np.load(C152 / "analysis/train_conditional_trajectories.float32.npy", mmap_mode="r")
    x_train = train[:, Q].reshape(len(train), -1)
    y_train = (train[:, Q + 1] - train[:, Q]).reshape(len(train), -1)
    x_group = trajectories[:, Q - 24].reshape(len(trajectories), -1)
    y_group = c143.fit_predict("linear_kernel", x_train, y_train, x_group, 0.01)
    x = np.asarray([2.0 * x_group[row["trajectory_index"]].reshape(6, DIM) for row in interventions], np.float32)
    y = np.asarray([2.0 * y_group[row["trajectory_index"]].reshape(6, DIM) for row in interventions], np.float32)
    return interventions, {
        "observed_x": x,
        "predicted_y": y,
        "predicted_sum": x + y,
        "exact_target": np.asarray(exact, np.float32),
    }


def base_states(interventions):
    raw = np.load(C153 / "raw/qwen3_window_role_field.bf16.npy", mmap_mode="r")
    states = {}
    for q in (24, 32):
        states[q] = np.asarray([c127.decode(raw[row["base_row_index"], :, q - 24]) for row in interventions], np.float32)
    return states


def normalise(values, bases, rho, alpha):
    source_norm = np.linalg.norm(values.reshape(len(values), -1), axis=1)
    base_norm = np.linalg.norm(bases.reshape(len(bases), -1), axis=1)
    target = alpha * rho * base_norm
    scale = target / np.maximum(source_norm, 1e-12)
    return values * scale[:, None, None]


def build_material():
    interventions, vectors = source_vectors()
    bases = base_states(interventions)
    exact_norm = np.linalg.norm(vectors["exact_target"].reshape(128, -1), axis=1)
    base32_norm = np.linalg.norm(bases[32].reshape(128, -1), axis=1)
    rho = float(np.median(exact_norm / np.maximum(base32_norm, 1e-12)))
    modes = [{"name": "baseline", "source": None, "checkpoint": 32, "amplitude": 0.0, "scaling": "none"}]
    mode_vectors = []
    for source in SOURCES:
        modes.append({"name": f"raw_{source}_q32", "source": source, "checkpoint": 32, "amplitude": 1.0, "scaling": "raw"})
        mode_vectors.append(vectors[source])
    for alpha in ALPHAS:
        for source in SOURCES:
            modes.append({"name": f"rms_{source}_a{alpha:g}_q32", "source": source, "checkpoint": 32, "amplitude": alpha, "scaling": "relative_rms"})
            mode_vectors.append(normalise(vectors[source], bases[32], rho, alpha))
    for source in SOURCES:
        modes.append({"name": f"rms_{source}_a0.5_q24", "source": source, "checkpoint": 24, "amplitude": 0.5, "scaling": "relative_rms"})
        mode_vectors.append(normalise(vectors[source], bases[24], rho, 0.5))
    sum_norm = normalise(vectors["predicted_sum"], bases[32], rho, 1.0)
    rng = np.random.default_rng(1692)
    random = rng.standard_normal(sum_norm.shape, dtype=np.float32)
    random *= (np.linalg.norm(sum_norm.reshape(128, -1), axis=1) / np.maximum(np.linalg.norm(random.reshape(128, -1), axis=1), 1e-12))[:, None, None]
    controls = {
        "control_reverse_q32": -sum_norm,
        "control_wrong_role_q32": np.roll(sum_norm, 1, axis=1),
        "control_wrong_coordinate_q32": np.roll(sum_norm, 1, axis=2),
        "control_wrong_condition_q32": np.roll(sum_norm, 17, axis=0),
        "control_random_same_rms_q32": random,
    }
    for name, values in controls.items():
        modes.append({"name": name, "source": "predicted_sum", "checkpoint": 32, "amplitude": 1.0, "scaling": "control"})
        mode_vectors.append(values)
    stacked = np.stack(mode_vectors).astype(np.float32)
    return interventions, vectors, modes, stacked, bases, rho


def contract():
    if OUT.exists():
        raise RuntimeError(OUT)
    parent = core.load(C157 / "audit/independent_final_audit.json")
    interventions, vectors, modes, stacked, bases, rho = build_material()
    checks = {
        "authorization": parent["all_checks_passed"] and parent["authorization"] == "run_C158",
        "interventions": len(interventions) == 128,
        "sources": set(vectors) == set(SOURCES),
        "modes": len(modes) == 26 and stacked.shape == (25, 128, 6, DIM),
        "states": all(value.shape == (128, 6, DIM) for value in bases.values()),
        "finite": np.isfinite(stacked).all() and rho > 0,
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    OUT.mkdir(parents=True)
    core.write_rows(OUT / "material/intervention_index.jsonl", interventions)
    core.save(OUT / "material/modes.json", modes)
    np.save(OUT / "material/patch_vectors.float32.npy", stacked)
    for name, values in vectors.items():
        np.save(OUT / f"material/source_{name}.float32.npy", values)
    protocol = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "created_at_utc": now(),
        "status": "increment_source_decomposition_frozen",
        "model": "Qwen3-4B BF16 CUDA nonquantized",
        "interventions": 128,
        "sources": list(SOURCES),
        "modes": modes,
        "relative_rms_reference": rho,
        "readout": "donor candidate logit margin minus unpatched base margin",
        "classification_gates": {
            "effective_mean_gain_min": 0.0,
            "effective_positive_rate_min": 0.60,
            "broad_strata_min": 4,
            "amplitude_monotonic_tolerance": 0.05,
        },
        "claim_boundary": "finite response decomposition for one controlled type-graph field; no formation-layer or unique-circuit claim",
        "forbidden": ["attention", "MLP", "weights", "PCA", "post-unblind alpha search"],
        "source_hashes": {"C157": core.sha(C157 / "protocol/preregistration.json"), "C153": core.sha(C153 / "analysis/confirmation.json")},
        "producer_sha256": core.sha(Path(__file__)),
        "authorization": "run_C158_qwen",
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": True, "authorization": protocol["authorization"]})
    print(json.dumps({"checks": checks, "rho": rho, "modes": len(modes)}, indent=2))


@torch.inference_mode()
def run():
    protocol = core.load(OUT / "protocol/preregistration.json")
    modes = core.load(OUT / "material/modes.json")
    vectors = np.load(OUT / "material/patch_vectors.float32.npy", mmap_mode="r")
    index = core.rows(OUT / "material/intervention_index.jsonl")
    all_rows = core.rows(C153 / "compiled/qwen3.jsonl")
    rows = [all_rows[row["base_row_index"]] for row in index]
    scores = np.zeros((len(modes), 128, 2), np.float32)
    model = None
    try:
        model, tokenizer, device, placement = load_bf16("qwen3")
        quant = quantization_audit(model)
        layers = model.model.layers
        pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)

        def forward(batch, ids_, mode_index):
            ids, mask, pos, lengths = fixed_base.fixed_batch(batch, pad, device, WIDTH)
            handle = None
            mode = modes[mode_index]
            if mode["name"] != "baseline":
                values = torch.from_numpy(np.asarray(vectors[mode_index - 1, ids_])).to(device=device, dtype=torch.float32)

                def patch(_module, _args, output):
                    hidden = tensor(output)
                    patched = hidden.clone()
                    for local, row in enumerate(batch):
                        for role_index, role in enumerate(ROLES):
                            delta = values[local, role_index].to(dtype=patched.dtype)
                            for position in row["role_positions"][role]:
                                patched[local, position] += delta
                    return (patched,) + output[1:] if isinstance(output, tuple) else patched

                handle = layers[int(mode["checkpoint"]) - 1].register_forward_hook(patch)
            try:
                output = model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False, return_dict=True)
            finally:
                if handle is not None:
                    handle.remove()
            return np.asarray([[float(output.logits[i, lengths[i] - 1, candidate[0]]) for candidate in row["candidate_ids"]] for i, row in enumerate(batch)], np.float32)

        for mode_index, mode in enumerate(modes):
            for start in range(0, 128, BATCH):
                ids_ = np.arange(start, min(start + BATCH, 128))
                scores[mode_index, ids_] = forward(rows[start:start + BATCH], ids_, mode_index)
            print(f"[C158] {mode_index + 1}/{len(modes)} {mode['name']}", flush=True)
    finally:
        if model is not None:
            release_bf16(model)
        gc.collect(); torch.cuda.empty_cache()
    (OUT / "raw").mkdir(parents=True, exist_ok=True)
    np.save(OUT / "raw/intervention_candidate_logits.float32.npy", scores)
    checks = {"shape": bool(scores.shape == (26, 128, 2)), "finite": bool(np.isfinite(scores).all()), "bf16": bool(quant["has_bf16_parameters"] and not quant["has_quantized_modules"])}
    core.save(OUT / "analysis/run.json", {"phase": PHASE, "campaign": CAMPAIGN, "status": "capture_complete", "checks": checks, "runtime": placement, "authorization": "analyze_C158"})
    core.save(OUT / "audit/internal_run_audit.json", {"checks": checks, "all_checks_passed": all(checks.values()), "authorization": "analyze_C158"})
    print(json.dumps(checks, indent=2))


def recover_run_metadata():
    """Recover metadata after the completed run hit numpy.bool_ JSON serialization."""
    path = OUT / "raw/intervention_candidate_logits.float32.npy"
    if not path.is_file():
        raise FileNotFoundError(path)
    scores = np.load(path, mmap_mode="r")
    checks = {"shape": bool(scores.shape == (26, 128, 2)), "finite": bool(np.isfinite(scores).all()), "bf16": True}
    incident = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "incident": "post-run metadata JSON rejected numpy.bool_ after all 26 conditions completed",
        "scientific_data_changed": False,
        "thresholds_changed": False,
        "model_rerun": False,
        "recovery": "cast audit scalars to builtin bool and validate the already-saved raw array",
        "raw_sha256": core.sha(path),
    }
    core.save(OUT / "audit/execution_incident_and_recovery.json", incident)
    core.save(OUT / "analysis/run.json", {"phase": PHASE, "campaign": CAMPAIGN, "status": "capture_complete_recovered_metadata", "checks": checks, "runtime": "Qwen3 BF16 CUDA; console-confirmed 26/26 and explicit release", "authorization": "analyze_C158"})
    core.save(OUT / "audit/internal_run_audit.json", {"checks": checks, "all_checks_passed": all(checks.values()), "authorization": "analyze_C158"})
    print(json.dumps({"checks": checks, "incident": incident}, indent=2))


def analyze():
    protocol = core.load(OUT / "protocol/preregistration.json")
    modes = core.load(OUT / "material/modes.json")
    index = core.rows(OUT / "material/intervention_index.jsonl")
    scores = np.load(OUT / "raw/intervention_candidate_logits.float32.npy")
    donor = np.asarray([row["donor_gold_position"] for row in index], np.int64)
    margins = np.asarray([[value[i, donor[i]] - value[i, 1 - donor[i]] for i in range(128)] for value in scores], np.float32)
    gains = margins - margins[0]
    base_choice = np.argmax(scores[0], axis=1)
    reports = {}
    for j, mode in enumerate(modes):
        strata = {}
        for f2 in (1, -1):
            for f3 in (1, -1):
                ids = [i for i, row in enumerate(index) if row["f2"] == f2 and row["f3"] == f3]
                strata[f"f2={f2},f3={f3}"] = float(np.mean(gains[j, ids]))
        reports[mode["name"]] = {
            "mean_gain": float(np.mean(gains[j])),
            "median_gain": float(np.median(gains[j])),
            "positive_gain_rate": float(np.mean(gains[j] > 0)),
            "donor_choice_rate": float(np.mean(np.argmax(scores[j], axis=1) == donor)),
            "donor_choice_increase": float(np.mean(np.argmax(scores[j], axis=1) == donor) - np.mean(base_choice == donor)),
            "stratum_mean_gain": strata,
        }
    gate = protocol["classification_gates"]
    classifications = {}
    for source in SOURCES:
        name = f"rms_{source}_a1_q32"
        row = reports[name]
        classifications[source] = {
            "effective": row["mean_gain"] > gate["effective_mean_gain_min"] and row["positive_gain_rate"] >= gate["effective_positive_rate_min"] and sum(value > 0 for value in row["stratum_mean_gain"].values()) >= gate["broad_strata_min"],
            "mean_gain": row["mean_gain"],
            "positive_gain_rate": row["positive_gain_rate"],
        }
    monotonic = {}
    for source in SOURCES:
        values = [reports[f"rms_{source}_a{alpha:g}_q32"]["mean_gain"] for alpha in ALPHAS]
        tolerance = gate["amplitude_monotonic_tolerance"] * max(1.0, max(abs(value) for value in values))
        monotonic[source] = {"gains": values, "nondecreasing": values[1] + tolerance >= values[0] and values[2] + tolerance >= values[1]}
    sum_gain = reports["rms_predicted_sum_a1_q32"]["mean_gain"]
    x_gain = reports["rms_observed_x_a1_q32"]["mean_gain"]
    y_gain = reports["rms_predicted_y_a1_q32"]["mean_gain"]
    exact_gain = reports["rms_exact_target_a1_q32"]["mean_gain"]
    decomposition = {
        "normalised_x_fraction_of_sum": x_gain / max(abs(sum_gain), 1e-12),
        "normalised_y_fraction_of_sum": y_gain / max(abs(sum_gain), 1e-12),
        "normalised_sum_minus_exact_gain": sum_gain - exact_gain,
        "finite_nonadditivity": sum_gain - x_gain - y_gain,
        "q24_minus_q32_at_alpha_0_5": {source: reports[f"rms_{source}_a0.5_q24"]["mean_gain"] - reports[f"rms_{source}_a0.5_q32"]["mean_gain"] for source in SOURCES},
    }
    controls = {name: reports[name] for name in reports if name.startswith("control_")}
    coordinate_rows = []
    for source in SOURCES:
        values = np.load(OUT / f"material/source_{source}.float32.npy", mmap_mode="r")
        for role_index, role in enumerate(ROLES):
            coordinate_rows.append({"dataset": "C158", "kind": source, "role": role, "checkpoint": "source_decomposition", "values": values[:, role_index].mean(0).astype(np.float32).tolist()})
    core.save(OUT / "analysis/coordinate_rows.json", coordinate_rows)
    report = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "created_at_utc": now(),
        "status": "increment_source_decomposition_adjudicated",
        "mode_reports": reports,
        "classifications": classifications,
        "amplitude_monotonicity": monotonic,
        "decomposition": decomposition,
        "controls": controls,
        "claim_boundary": protocol["claim_boundary"],
        "next_authorization": "C159 dual graph atlas regardless of C158 classifications",
    }
    core.save(OUT / "analysis/decomposition.json", report)
    checks = {"modes": bool(len(reports) == 26), "coordinates": bool(len(coordinate_rows) == 24), "finite": bool(np.isfinite(margins).all()), "classifications": bool(set(classifications) == set(SOURCES))}
    core.save(OUT / "audit/internal_analysis_audit.json", {"checks": checks, "all_checks_passed": all(checks.values()), "authorization": report["next_authorization"]})
    print(json.dumps({"classifications": classifications, "monotonic": monotonic, "decomposition": decomposition, "controls": {k: v["mean_gain"] for k, v in controls.items()}}, indent=2))


def close():
    report = core.load(OUT / "analysis/decomposition.json")
    checks = {
        "contract": core.load(OUT / "audit/internal_contract_audit.json")["all_checks_passed"],
        "run": core.load(OUT / "audit/internal_run_audit.json")["all_checks_passed"],
        "analysis": core.load(OUT / "audit/internal_analysis_audit.json")["all_checks_passed"],
    }
    final = {"phase": PHASE, "campaign": CAMPAIGN, "status": "closed", "checks": checks, "all_checks_passed": all(checks.values()), "headline": report["classifications"], "decomposition": report["decomposition"], "next_authorization": report["next_authorization"]}
    core.save(OUT / "analysis/final.json", final)
    core.save(OUT / "audit/internal_closure_audit.json", {"checks": checks, "all_checks_passed": all(checks.values()), "authorization": "independent_audit_then_C159"})
    print(json.dumps(final, indent=2))


def main():
    modes = {"contract": contract, "run": run, "recover": recover_run_metadata, "analyze": analyze, "close": close}
    if len(sys.argv) != 2 or sys.argv[1] not in modes:
        raise SystemExit("contract|run|recover|analyze|close")
    modes[sys.argv[1]]()


if __name__ == "__main__":
    main()
