#!/usr/bin/env python3
"""C153: prospective confirmation of the frozen type-graph conditional-pooled transition object."""
from __future__ import annotations

import gc
import itertools
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
OUT = RESULT / "phase1687_c153_type_graph_conditional_pool_confirmation"
C152 = RESULT / "phase1686_c152_type_graph_transition_object_discovery"
PUBLIC = ROOT / "frontend/public/vis_data/research_kernel/c109_role_state_field_atlas.json"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core
from phase1332_bf16_utils import load_bf16, quantization_audit, release_bf16
import phase1571_c098_observation_first_graph_campaign as graph_base
import phase1572_c099_fixed_width_graph_field_campaign as fixed_base
import phase1661_c127_typed_transition_language_family as c127
import phase1675_c141_multifamily_full_coordinate_atlas as c141
import phase1677_c143_transition_model_competition as c143
import phase1686_c152_type_graph_transition_object_discovery as c152

PHASE, CAMPAIGN = 1687, "C153"
WINDOW = tuple(range(24, 34))
STATES = tuple(range(24, 35))
ROLES = c141.ROLES
DIM, WIDTH, BATCH = 2560, 224, 4


def now():
    return datetime.now(timezone.utc).isoformat()


def material():
    units, cases = [], []
    for local in range(8):
        source_unit = 80 + local
        unit_id = f"c153-type_graph-{local:02d}"
        units.append({"unit_id": unit_id, "source_unit": source_unit, "arm": "type_graph"})
        for f1, f2, f3, surface, code in itertools.product((1, -1), repeat=5):
            row = c141.make_case("type_graph", source_unit, f1, f2, f3, surface, code)
            row.update({"case_id": f"c153-{len(cases):05d}", "unit_id": unit_id, "source_unit": source_unit, "partition": "fresh_confirmation"})
            cases.append(row)
    return units, cases


def contract():
    if OUT.exists():
        raise RuntimeError(OUT)
    parent = core.load(C152 / "audit/independent_closure_audit.json")
    discovery = core.load(C152 / "analysis/discovery.json")
    units, cases = material()
    compiled = c141.compile_rows(graph_base.tokenizer(), cases)
    cells = {(r["unit_id"], r["factors"]["f1"], r["factors"]["f2"], r["factors"]["f3"], r["surface_factor"], r["codebook_factor"]) for r in cases}
    checks = {
        "authorization": parent["all_checks_passed"] and parent["scientific_candidate_stable"],
        "winner": discovery["selected_candidate"] == "conditional_pooled" and discovery["stable_candidate"],
        "units": len(units) == 8,
        "cases": len(cases) == 256 and len(cells) == 256,
        "unique": len({r["prompt"] for r in cases}) == 256,
        "balance": sum(r["gold_position"] == 0 for r in cases) == 128,
        "roles": all(set(r["role_positions"]) == set(ROLES) for r in compiled),
        "width": max(len(r["prompt_ids"]) for r in compiled) < WIDTH,
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    OUT.mkdir(parents=True)
    core.write_rows(OUT / "material/units.jsonl", units)
    core.write_rows(OUT / "material/cases.jsonl", cases)
    core.write_rows(OUT / "compiled/qwen3.jsonl", compiled)
    protocol = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "created_at_utc": now(),
        "status": "fresh_type_graph_confirmation_contract_frozen",
        "execution_model": "Qwen3-4B BF16 CUDA nonquantized",
        "predictor": "conditional_pooled linear_kernel lambda=0.01",
        "training": str(C152 / "analysis/train_conditional_trajectories.float32.npy"),
        "cases": 256,
        "window": list(WINDOW),
        "gates": {
            "median_cosine_min": 0.65,
            "median_relative_error_max": 0.78,
            "each_transition_cosine_min": 0.50,
            "each_transition_relative_error_max": 0.90,
            "each_stratum_median_cosine_min": 0.50,
            "wrong_control_median_margin_min": 0.05,
            "rollout_final_ratio_vs_identity_max": 0.90,
        },
        "behavior_policy": "descriptive; errors retained and do not stop HiddenState observation",
        "claim_boundary": "prospective effective transition prediction in one controlled type-graph family; not a unique circuit or natural-language ontology",
        "forbidden": ["attention", "MLP", "weights", "PCA", "post-unblind model competition"],
        "source_hashes": {"C152": core.sha(C152 / "analysis/discovery.json"), "training": core.sha(C152 / "analysis/train_conditional_trajectories.float32.npy")},
        "producer_sha256": core.sha(Path(__file__)),
        "authorization": "run_C153_qwen",
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": all(checks.values()), "authorization": protocol["authorization"]})
    print(json.dumps({"checks": checks, "max_width": max(len(r["prompt_ids"]) for r in compiled)}, indent=2))


def tensor(value):
    return value[0] if isinstance(value, tuple) else value


@torch.inference_mode()
def run():
    rows = core.rows(OUT / "compiled/qwen3.jsonl")
    path = OUT / "raw/qwen3_window_role_field.bf16.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint16, shape=(256, 6, 11, DIM))
    logits = np.lib.format.open_memmap(OUT / "raw/qwen3_candidate_logits.float32.npy", mode="w+", dtype=np.float32, shape=(256, 2))
    result, model, repeat = [], None, 0.0
    try:
        model, tokenizer, device, _placement = load_bf16("qwen3")
        quant = quantization_audit(model)
        base = model.model
        embed, layers, norm = base.embed_tokens, base.layers, base.norm
        pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)

        def batch_run(batch):
            captured = {}
            hooks = [embed.register_forward_hook(lambda _m, _a, output: captured.__setitem__(0, tensor(output).detach()))]
            hooks += [layer.register_forward_hook(lambda _m, _a, output, q=i + 1: captured.__setitem__(q, tensor(output).detach())) for i, layer in enumerate(layers)]
            hooks += [norm.register_forward_hook(lambda _m, _a, output: captured.__setitem__(37, tensor(output).detach()))]
            try:
                ids, mask, pos, lens = fixed_base.fixed_batch(batch, pad, device, WIDTH)
                output = model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False, return_dict=True)
            finally:
                for hook in hooks:
                    hook.remove()
            return captured, output, ids, mask, pos, lens

        for start in range(0, 256, BATCH):
            batch = rows[start:start + BATCH]
            cap, output, ids, mask, pos, lens = batch_run(batch)
            scores = np.asarray([[float(output.logits[i, lens[i] - 1, candidate[0]]) for candidate in row["candidate_ids"]] for i, row in enumerate(batch)], np.float32)
            logits[start:start + len(batch)] = scores
            for i, row in enumerate(batch):
                prediction = int(scores[i, 1] > scores[i, 0])
                result.append({"row_index": start + i, "case_id": row["case_id"], "unit_id": row["unit_id"], "factors": row["factors"], "surface_factor": row["surface_factor"], "codebook_factor": row["codebook_factor"], "gold_position": row["gold_position"], "prediction": prediction, "correct": prediction == row["gold_position"]})
                for role_index, role in enumerate(ROLES):
                    for state_index, q in enumerate(STATES):
                        raw[start + i, role_index, state_index] = cap[q][i, row["role_positions"][role]].mean(0).contiguous().view(torch.uint16).cpu().numpy()
            if (start // BATCH + 1) % 32 == 0:
                raw.flush(); logits.flush(); print(f"[C153] {start + len(batch)}/256", flush=True)
            del cap, output, ids, mask, pos
        raw.flush(); logits.flush()
        cap, output, ids, mask, pos, lens = batch_run(rows[:BATCH])
        check = np.asarray([[float(output.logits[i, lens[i] - 1, candidate[0]]) for candidate in row["candidate_ids"]] for i, row in enumerate(rows[:BATCH])], np.float32)
        repeat = float(np.max(np.abs(check - np.asarray(logits[:BATCH]))))
    finally:
        raw.flush(); logits.flush()
        if model is not None:
            release_bf16(model)
        gc.collect(); torch.cuda.empty_cache()
    core.write_rows(OUT / "raw/qwen3_behavior_index.jsonl", result)
    behavior = {"global": float(np.mean([r["correct"] for r in result])), "stratum": {f"f2={f2},f3={f3}": float(np.mean([r["correct"] for r in result if r["factors"]["f2"] == f2 and r["factors"]["f3"] == f3])) for f2 in (1, -1) for f3 in (1, -1)}}
    checks = {"rows": len(result) == 256, "shape": list(raw.shape) == [256, 6, 11, DIM], "finite": bool(np.isfinite(logits).all()), "repeat": repeat == 0.0, "bf16": quant["has_bf16_parameters"] and not quant["has_quantized_modules"]}
    report = {"phase": PHASE, "campaign": CAMPAIGN, "status": "fresh_type_graph_capture_complete", "behavior": behavior, "checks": checks, "repeat_logits_max_abs": repeat, "role_sha256": core.sha(path), "authorization": "analyze_C153"}
    core.save(OUT / "analysis/capture.json", report)
    core.save(OUT / "audit/internal_capture_audit.json", {"checks": checks, "all_checks_passed": all(checks.values()), "authorization": report["authorization"]})
    print(json.dumps(report, indent=2))


def analyze():
    protocol = core.load(OUT / "protocol/preregistration.json")
    rows = core.rows(OUT / "compiled/qwen3.jsonl")
    raw = np.load(OUT / "raw/qwen3_window_role_field.bf16.npy", mmap_mode="r")
    trajectories, keys = c152.conditional_trajectories(rows, raw, range(8), STATES)
    train = np.load(C152 / "analysis/train_conditional_trajectories.float32.npy", mmap_mode="r")
    transition_rows, coordinate_rows = [], []
    predictions = []
    for j, q in enumerate(WINDOW):
        x_train = train[:, q].reshape(len(train), -1)
        y_train = (train[:, q + 1] - train[:, q]).reshape(len(train), -1)
        x = trajectories[:, j].reshape(len(trajectories), -1)
        y = (trajectories[:, j + 1] - trajectories[:, j]).reshape(len(trajectories), -1)
        pred = c143.fit_predict("linear_kernel", x_train, y_train, x, 0.01)
        predictions.append(pred)
        strata = {}
        for f2 in (1, -1):
            for f3 in (1, -1):
                name = f"f2={f2},f3={f3}"
                ids = np.asarray([i for i, key in enumerate(keys) if key["f2"] == f2 and key["f3"] == f3])
                strata[name] = c143.metrics(pred[ids], y[ids])
                for kind, values in (("predicted_increment", pred), ("target_increment", y)):
                    coordinate_rows.append({"dataset": "C153", "kind": kind, "role": "boundary", "stratum": name, "transition_index": q, "checkpoint": f"{q}->{q + 1}", "values": values.reshape(len(values), 6, DIM)[ids, 5].mean(0).astype(np.float32).tolist()})
        transition_rows.append({"q": q, "target": c143.metrics(pred, y), "wrong_role": c143.metrics(np.roll(pred.reshape(len(pred), 6, DIM), 1, axis=1).reshape(len(pred), -1), y), "wrong_coordinate": c143.metrics(np.roll(pred.reshape(len(pred), 6, DIM), 1, axis=2).reshape(len(pred), -1), y), "strata": strata})
    current = trajectories[:, 0].reshape(len(trajectories), -1).copy()
    rollout = []
    for j, q in enumerate(WINDOW):
        x_train = train[:, q].reshape(len(train), -1)
        y_train = (train[:, q + 1] - train[:, q]).reshape(len(train), -1)
        current += c143.fit_predict("linear_kernel", x_train, y_train, current, 0.01)
        target = trajectories[:, j + 1].reshape(len(trajectories), -1)
        identity = trajectories[:, 0].reshape(len(trajectories), -1)
        metric, baseline = c143.metrics(current, target), c143.metrics(identity, target)
        rollout.append({"q": q, "rollout": metric, "identity": baseline, "ratio": metric["relative_error"] / max(baseline["relative_error"], 1e-12)})
    median_cos = float(np.median([r["target"]["cosine"] for r in transition_rows]))
    median_error = float(np.median([r["target"]["relative_error"] for r in transition_rows]))
    stratum_medians = {name: float(np.median([r["strata"][name]["cosine"] for r in transition_rows])) for name in transition_rows[0]["strata"]}
    role_margin = float(np.median([r["wrong_role"]["relative_error"] - r["target"]["relative_error"] for r in transition_rows]))
    coord_margin = float(np.median([r["wrong_coordinate"]["relative_error"] - r["target"]["relative_error"] for r in transition_rows]))
    g = protocol["gates"]
    gates = {
        "aggregate": median_cos >= g["median_cosine_min"] and median_error <= g["median_relative_error_max"],
        "each_transition": all(r["target"]["cosine"] >= g["each_transition_cosine_min"] and r["target"]["relative_error"] <= g["each_transition_relative_error_max"] for r in transition_rows),
        "strata": all(v >= g["each_stratum_median_cosine_min"] for v in stratum_medians.values()),
        "wrong_role": role_margin >= g["wrong_control_median_margin_min"],
        "wrong_coordinate": coord_margin >= g["wrong_control_median_margin_min"],
        "rollout": rollout[-1]["ratio"] <= g["rollout_final_ratio_vs_identity_max"],
    }
    report = {"phase": PHASE, "campaign": CAMPAIGN, "created_at_utc": now(), "status": "fresh_type_graph_confirmation_adjudicated", "median_cosine": median_cos, "median_relative_error": median_error, "stratum_median_cosine": stratum_medians, "control_margins": {"wrong_role": role_margin, "wrong_coordinate": coord_margin}, "transition_rows": transition_rows, "rollout_rows": rollout, "gates": gates, "confirmation_gate_passed": all(gates.values()), "claim_boundary": protocol["claim_boundary"], "authorization": "close_C153"}
    core.save(OUT / "analysis/confirmation.json", report)
    core.save(OUT / "analysis/coordinate_rows.json", coordinate_rows)
    np.save(OUT / "analysis/fresh_conditional_trajectories.float32.npy", trajectories)
    core.write_rows(OUT / "analysis/fresh_conditional_index.jsonl", keys)
    checks = {"shape": list(trajectories.shape) == [128, 11, 6, DIM], "transitions": len(transition_rows) == 10, "coordinates": len(coordinate_rows) == 80 and all(len(r["values"]) == DIM for r in coordinate_rows), "finite": bool(np.isfinite(trajectories).all())}
    core.save(OUT / "audit/internal_analysis_audit.json", {"checks": checks, "all_checks_passed": all(checks.values()), "scientific_gate_passed": all(gates.values()), "authorization": report["authorization"]})
    print(json.dumps({"behavior": core.load(OUT / "analysis/capture.json")["behavior"], "median_cosine": median_cos, "median_error": median_error, "strata": stratum_medians, "margins": report["control_margins"], "final_rollout_ratio": rollout[-1]["ratio"], "gates": gates}, indent=2))


def close():
    report = core.load(OUT / "analysis/confirmation.json")
    payload = core.load(PUBLIC)
    coordinate_rows = core.load(OUT / "analysis/coordinate_rows.json")
    payload["c153_type_graph_confirmation"] = {"capture": core.load(OUT / "analysis/capture.json"), "confirmation": report, "coordinate_rows": coordinate_rows}
    payload.update({"phase": PHASE, "campaign": "C109-C153", "title": "Role-State Atlas + Type-Graph Conditional Transition Confirmation", "created_at_utc": now()})
    canonical = OUT / "analysis/c109_c153_atlas.json"
    core.save(canonical, payload)
    shutil.copyfile(canonical, PUBLIC)
    checks = {"contract": core.load(OUT / "audit/internal_contract_audit.json")["all_checks_passed"], "capture": core.load(OUT / "audit/internal_capture_audit.json")["all_checks_passed"], "analysis": core.load(OUT / "audit/internal_analysis_audit.json")["all_checks_passed"], "asset": core.sha(canonical) == core.sha(PUBLIC)}
    closure = {"phase": PHASE, "campaign": CAMPAIGN, "status": "fresh_type_graph_confirmation_closed", "gate_passed": report["confirmation_gate_passed"], "headline": {"behavior": core.load(OUT / "analysis/capture.json")["behavior"], "median_cosine": report["median_cosine"], "median_relative_error": report["median_relative_error"], "strata": report["stratum_median_cosine"], "final_rollout_ratio": report["rollout_rows"][-1]["ratio"]}, "claim_boundary": report["claim_boundary"], "next_authorization": "freeze_local_HiddenState_causal_contract" if report["confirmation_gate_passed"] else "continue_observational_type_graph_object_search"}
    core.save(OUT / "analysis/closure.json", closure)
    core.save(OUT / "audit/internal_closure_audit.json", {"checks": checks, "all_checks_passed": all(checks.values()), "asset_sha256": core.sha(PUBLIC), "authorization": "independent_final_and_memo"})
    print(json.dumps(closure, indent=2))


def main():
    modes = {"contract": contract, "run": run, "analyze": analyze, "close": close}
    if len(sys.argv) != 2 or sys.argv[1] not in modes:
        raise SystemExit("contract|run|analyze|close")
    modes[sys.argv[1]]()


if __name__ == "__main__":
    main()
