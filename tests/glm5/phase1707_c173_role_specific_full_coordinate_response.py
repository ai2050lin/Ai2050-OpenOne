#!/usr/bin/env python3
"""C173: role-specific exhaustive q24->q25 coordinate response discovery and validation."""
from __future__ import annotations

import argparse
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
OUT = RESULT / "phase1707_c173_role_specific_full_coordinate_response"
C159 = RESULT / "phase1693_c159_natural_isomorphic_dual_graph_atlas"
C167 = RESULT / "phase1701_c167_transport_component_decomposition"
C172 = RESULT / "phase1706_c172_typed_response_graph_master_contract"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core
from phase1332_bf16_utils import load_bf16, quantization_audit, release_bf16
import phase1572_c099_fixed_width_graph_field_campaign as fixed_base
import phase1661_c127_typed_transition_language_family as c127
import phase1693_c159_natural_isomorphic_dual_graph_atlas as c159

PHASE, CAMPAIGN = 1707, "C173"
DIM, WIDTH, BATCH = 2560, 256, 16
SOURCE_Q, TARGET_Q = 24, 25
SOURCE_ROLES = ("primary", "query")
TARGET_ROLES = c159.ROLES
PANELS = ("natural_lexical", "isomorphic_nonce")
RELATIONS = ("is_a", "part_of", "located_in", "precedes")
PARTITIONS = ("discovery", "confirmation", "fresh")
ALLIANCE_SIZE = 64


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def tensor(value):
    return value[0] if isinstance(value, tuple) else value


def anchors():
    pairs = core.rows(C159 / "analysis/late_half_difference_index.jsonl")
    selected = []
    for part in PARTITIONS:
        for panel in PANELS:
            for relation in RELATIONS:
                choices = [r for r in pairs if r["partition"] == part and r["panel"] == panel and r["relation_family"] == relation and r["path"] == -1 and r["interference"] == 1 and r["direction_form"] == 1 and r["surface"] == 1 and r["code"] == 1]
                if len(choices) != 1:
                    raise RuntimeError((part, panel, relation, len(choices)))
                selected.append(dict(choices[0], anchor_index=len(selected)))
    return selected


def decompose(x):
    shared = x.mean(axis=(0, 1))
    panel = x.mean(axis=1) - shared[None]
    relation = x.mean(axis=0) - shared[None]
    interaction = x - shared[None, None] - panel[:, None] - relation[None]
    return shared, panel, relation, interaction


def contract():
    if OUT.exists():
        raise RuntimeError(OUT)
    parent = core.load(C172 / "audit/independent_final_audit.json")
    selected = anchors()
    compiled = core.rows(C159 / "compiled/qwen3.jsonl")
    logits = np.load(C159 / "raw/qwen3_candidate_logits.float32.npy", mmap_mode="r")
    anchor_rows = [r[k] for r in selected for k in ("minus_row", "plus_row")]
    behavior = [int(np.argmax(logits[i]) == compiled[i]["gold_position"]) for i in anchor_rows]
    relation_coordinates = core.load(C167 / "analysis/top_relation_source_coordinates.json")["coordinates"][:ALLIANCE_SIZE]
    checks = {
        "authorization": parent["all_checks_passed"] and parent["authorization"] == "run_C173_role_specific_full_coordinate_campaign",
        "anchors": len(selected) == 24,
        "balanced": len({(r["partition"], r["panel"], r["relation_family"]) for r in selected}) == 24,
        "behavior": all(behavior),
        "roles": len(SOURCE_ROLES) == 2,
        "relation_control": len(relation_coordinates) == ALLIANCE_SIZE and len(set(relation_coordinates)) == ALLIANCE_SIZE,
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    OUT.mkdir(parents=True)
    core.write_rows(OUT / "material/anchors.jsonl", selected)
    protocol = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "created_at_utc": now(),
        "status": "role_specific_full_coordinate_contract_frozen",
        "model": "Qwen3-4B BF16 CUDA nonquantized",
        "source": {"checkpoint": SOURCE_Q, "roles": list(SOURCE_ROLES), "coordinates": DIM},
        "target": {"checkpoint": TARGET_Q, "roles": list(TARGET_ROLES), "coordinates_per_role": DIM},
        "discovery": "8 anchors = panel x relation; exhaustive 2560 source-coordinate scan for each source role",
        "selection": "rank by cross-panel signed relation-centered response stability; lock 64 per role",
        "validation": "confirmation and fresh; own-role, wrong-role, and relation-selected alliances",
        "perturbation": "symmetric plus/minus, epsilon=0.5 times source-role state RMS",
        "primary_metrics": ["signed_nrmse", "signed_explained_energy", "discovery-frozen active-edge sign agreement", "source-permutation advantage"],
        "secondary_metric": "cosine",
        "descriptive_replication": {"sign_agreement_min": 0.55, "source_permutation_nrmse_advantage_min": 0.02, "required_partitions": 2},
        "campaign_policy": "evaluate all roles and alliances regardless of labels",
        "claim_boundary": "role-specific local effective response field at q24->q25; not a unique circuit, role capacity norm, or natural necessity result",
        "forbidden": ["attention", "MLP", "weights", "PCA", "holdout-informed coordinate selection"],
        "relation_control_coordinates": relation_coordinates,
        "source_hashes": {"C159_states": core.sha(C159 / "raw/qwen3_six_role_all_checkpoint.bf16.npy"), "C167_lock": core.sha(C167 / "analysis/top_relation_source_coordinates.json")},
        "producer_sha256": core.sha(Path(__file__)),
        "authorization": "run_C173_discovery",
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    print(json.dumps({"checks": checks, "discovery_shape": [2, 8, DIM, 6, DIM], "estimated_discovery_bytes": 2 * 8 * DIM * 6 * DIM * 2}, indent=2))


def source_epsilon(states, row_index, source_role):
    role_i = TARGET_ROLES.index(source_role)
    source = c127.decode(states[row_index, role_i, SOURCE_Q])
    return 0.5 * float(np.sqrt(np.mean(np.square(source), dtype=np.float64)))


@torch.inference_mode()
def run_response(partitions, coordinate_sets, output_path, output_shape, label):
    selected = [r for r in core.rows(OUT / "material/anchors.jsonl") if r["partition"] in partitions]
    compiled = core.rows(C159 / "compiled/qwen3.jsonl")
    states = np.load(C159 / "raw/qwen3_six_role_all_checkpoint.bf16.npy", mmap_mode="r")
    output = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.float16, shape=output_shape)
    epsilons = np.zeros(output_shape[:3] if label == "validate" else output_shape[:2], np.float32)
    model = None
    try:
        model, tokenizer, device, placement = load_bf16("qwen3")
        quant = quantization_audit(model)
        layers = model.model.layers
        pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)

        def perturb(row, source_role, coordinates, sign, epsilon):
            batch = [row] * len(coordinates)
            ids, mask, pos, _lengths = fixed_base.fixed_batch(batch, pad, device, WIDTH)
            captured = {}

            def patch(_module, _args, value):
                hidden = tensor(value)
                patched = hidden.clone()
                for local, coordinate in enumerate(coordinates):
                    for position in row["role_positions"][source_role]:
                        patched[local, position, int(coordinate)] += sign * epsilon
                return (patched,) + value[1:] if isinstance(value, tuple) else patched

            h1 = layers[SOURCE_Q - 1].register_forward_hook(patch)
            h2 = layers[TARGET_Q - 1].register_forward_hook(lambda _m, _a, value: captured.__setitem__("state", tensor(value).detach()))
            try:
                model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False, return_dict=True)
            finally:
                h1.remove(); h2.remove()
            field = np.zeros((len(coordinates), 6, DIM), np.float32)
            for local in range(len(coordinates)):
                for role_i, role in enumerate(TARGET_ROLES):
                    field[local, role_i] = captured["state"][local, row["role_positions"][role]].mean(0).float().cpu().numpy()
            return field

        if label == "discovery":
            for role_i, source_role in enumerate(SOURCE_ROLES):
                for anchor_i, anchor in enumerate(selected):
                    row = compiled[anchor["minus_row"]]
                    epsilon = source_epsilon(states, anchor["minus_row"], source_role)
                    epsilons[role_i, anchor_i] = epsilon
                    for start in range(0, DIM, BATCH):
                        coordinates = np.arange(start, min(start + BATCH, DIM))
                        plus = perturb(row, source_role, coordinates, 1.0, epsilon)
                        minus = perturb(row, source_role, coordinates, -1.0, epsilon)
                        output[role_i, anchor_i, coordinates] = ((plus - minus) / (2 * epsilon)).astype(np.float16)
                    output.flush()
                    print(f"[C173-discovery] role {role_i + 1}/2 {source_role} anchor {anchor_i + 1}/8", flush=True)
        else:
            alliance_names = ("own", "wrong_role", "relation")
            for role_i, source_role in enumerate(SOURCE_ROLES):
                for alliance_i, alliance_name in enumerate(alliance_names):
                    coordinates = np.asarray(coordinate_sets[source_role][alliance_name], int)
                    for anchor_i, anchor in enumerate(selected):
                        row = compiled[anchor["minus_row"]]
                        epsilon = source_epsilon(states, anchor["minus_row"], source_role)
                        epsilons[role_i, alliance_i, anchor_i] = epsilon
                        for start in range(0, ALLIANCE_SIZE, BATCH):
                            local_coordinates = coordinates[start:start + BATCH]
                            plus = perturb(row, source_role, local_coordinates, 1.0, epsilon)
                            minus = perturb(row, source_role, local_coordinates, -1.0, epsilon)
                            output[role_i, alliance_i, anchor_i, start:start + len(local_coordinates)] = ((plus - minus) / (2 * epsilon)).astype(np.float16)
                    output.flush()
                    print(f"[C173-validation] {source_role} {alliance_name}", flush=True)
    finally:
        output.flush()
        if model is not None:
            release_bf16(model)
        gc.collect(); torch.cuda.empty_cache()
    return epsilons, placement, quant


def run_discovery():
    (OUT / "raw").mkdir(parents=True, exist_ok=True)
    eps, placement, quant = run_response(("discovery",), None, OUT / "raw/discovery_full_response.float16.npy", (2, 8, DIM, 6, DIM), "discovery")
    np.save(OUT / "raw/discovery_epsilons.float32.npy", eps)
    raw = np.load(OUT / "raw/discovery_full_response.float16.npy", mmap_mode="r")
    checks = {"shape": list(raw.shape) == [2, 8, DIM, 6, DIM], "finite": bool(np.isfinite(raw).all()), "epsilon": bool(np.all(eps > 0)), "bf16": bool(quant["has_bf16_parameters"] and not quant["has_quantized_modules"])}
    core.save(OUT / "analysis/discovery_run.json", {"checks": checks, "runtime": placement})
    core.save(OUT / "audit/internal_discovery_run_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    print(json.dumps(checks, indent=2))


def select():
    raw = np.load(OUT / "raw/discovery_full_response.float16.npy", mmap_mode="r")
    locks = {}
    for role_i, source_role in enumerate(SOURCE_ROLES):
        strength = np.zeros(DIM, np.float64)
        cross_panel_cosine = np.zeros(DIM, np.float64)
        for start in range(0, DIM, 16):
            stop = min(start + 16, DIM)
            x = np.asarray(raw[role_i, :, start:stop], np.float32).reshape(2, 4, stop - start, 6, DIM)
            centered = x - x.mean(axis=1, keepdims=True)
            a = centered[0].transpose(1, 0, 2, 3).reshape(stop - start, -1)
            b = centered[1].transpose(1, 0, 2, 3).reshape(stop - start, -1)
            an = np.linalg.norm(a, axis=1)
            bn = np.linalg.norm(b, axis=1)
            cross_panel_cosine[start:stop] = np.sum(a * b, axis=1) / np.maximum(an * bn, 1e-12)
            strength[start:stop] = np.sqrt(an * bn)
        stable_score = strength * np.maximum(cross_panel_cosine, 0.0)
        order = np.lexsort((-strength, -stable_score))
        coordinates = order[:ALLIANCE_SIZE].tolist()
        locks[source_role] = {
            "coordinates": coordinates,
            "stable_score": stable_score[coordinates].tolist(),
            "cross_panel_cosine": cross_panel_cosine[coordinates].tolist(),
            "strength": strength[coordinates].tolist(),
        }
    lock = {"phase": PHASE, "campaign": CAMPAIGN, "created_at_utc": now(), "selection_source": "discovery only", "alliance_size": ALLIANCE_SIZE, "roles": locks, "confirmation_and_fresh_unread": True, "authorization": "run_C173_validation"}
    core.save(OUT / "protocol/role_specific_coordinate_lock.json", lock)
    checks = {"roles": set(locks) == set(SOURCE_ROLES), "size": all(len(v["coordinates"]) == ALLIANCE_SIZE for v in locks.values()), "unique": all(len(set(v["coordinates"])) == ALLIANCE_SIZE for v in locks.values()), "finite": all(np.isfinite(v["stable_score"]).all() for v in locks.values())}
    core.save(OUT / "audit/internal_selection_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    print(json.dumps({"checks": checks, "head": {k: v["coordinates"][:8] for k, v in locks.items()}}, indent=2))


def run_validation():
    protocol = core.load(OUT / "protocol/preregistration.json")
    lock = core.load(OUT / "protocol/role_specific_coordinate_lock.json")
    own = {r: lock["roles"][r]["coordinates"] for r in SOURCE_ROLES}
    coordinate_sets = {
        "primary": {"own": own["primary"], "wrong_role": own["query"], "relation": protocol["relation_control_coordinates"]},
        "query": {"own": own["query"], "wrong_role": own["primary"], "relation": protocol["relation_control_coordinates"]},
    }
    eps, placement, quant = run_response(("confirmation", "fresh"), coordinate_sets, OUT / "raw/validation_response.float16.npy", (2, 3, 16, ALLIANCE_SIZE, 6, DIM), "validate")
    np.save(OUT / "raw/validation_epsilons.float32.npy", eps)
    raw = np.load(OUT / "raw/validation_response.float16.npy", mmap_mode="r")
    checks = {"shape": list(raw.shape) == [2, 3, 16, ALLIANCE_SIZE, 6, DIM], "finite": bool(np.isfinite(raw).all()), "epsilon": bool(np.all(eps > 0)), "bf16": bool(quant["has_bf16_parameters"] and not quant["has_quantized_modules"])}
    core.save(OUT / "analysis/validation_run.json", {"checks": checks, "runtime": placement})
    core.save(OUT / "audit/internal_validation_run_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    print(json.dumps(checks, indent=2))


def field_metrics(pred, actual):
    p = pred.reshape(len(pred), -1).astype(np.float64, copy=False)
    a = actual.reshape(len(actual), -1).astype(np.float64, copy=False)
    err2 = np.sum((a - p) ** 2, axis=1)
    an2 = np.sum(a * a, axis=1)
    nrmse = np.sqrt(err2 / np.maximum(an2, 1e-12))
    explained = 1.0 - err2 / np.maximum(an2, 1e-12)
    cosine = np.sum(a * p, axis=1) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(p, axis=1), 1e-12)
    sign = []
    for ai in range(len(a)):
        threshold = np.quantile(np.abs(p[ai]), 0.95)
        active = np.abs(p[ai]) >= threshold
        sign.append(float(np.mean(np.sign(a[ai, active]) == np.sign(p[ai, active]))))
    return nrmse, explained, cosine, np.asarray(sign)


def analyze():
    protocol = core.load(OUT / "protocol/preregistration.json")
    lock = core.load(OUT / "protocol/role_specific_coordinate_lock.json")
    discovery = np.load(OUT / "raw/discovery_full_response.float16.npy", mmap_mode="r")
    validation = np.load(OUT / "raw/validation_response.float16.npy", mmap_mode="r")
    relation_coordinates = protocol["relation_control_coordinates"]
    alliance_names = ("own", "wrong_role", "relation")
    rows = []
    frozen_components = np.zeros((2, 3, 4, ALLIANCE_SIZE, 6, DIM), np.float16)
    validation_components = np.zeros((2, 3, 2, 4, ALLIANCE_SIZE, 6, DIM), np.float16)
    for role_i, source_role in enumerate(SOURCE_ROLES):
        own = lock["roles"][source_role]["coordinates"]
        wrong = lock["roles"][SOURCE_ROLES[1 - role_i]]["coordinates"]
        sets = (own, wrong, relation_coordinates)
        for alliance_i, (alliance_name, coordinates) in enumerate(zip(alliance_names, sets)):
            d = np.asarray(discovery[role_i, :, coordinates], np.float32).transpose(1, 0, 2, 3).reshape(2, 4, ALLIANCE_SIZE, 6, DIM)
            reference = decompose(d)[2]
            frozen_components[role_i, alliance_i] = reference.astype(np.float16)
            for part_i, partition in enumerate(("confirmation", "fresh")):
                start = part_i * 8
                x = np.asarray(validation[role_i, alliance_i, start:start + 8], np.float32).reshape(2, 4, ALLIANCE_SIZE, 6, DIM)
                actual = decompose(x)[2]
                validation_components[role_i, alliance_i, part_i] = actual.astype(np.float16)
                pred = reference.reshape(4 * ALLIANCE_SIZE, 6, DIM)
                act = actual.reshape(4 * ALLIANCE_SIZE, 6, DIM)
                nrmse, explained, cosine, sign = field_metrics(pred, act)
                perm = np.roll(pred, 1, axis=0)
                perm_nrmse = field_metrics(perm, act)[0]
                wrong_relation = np.roll(reference, 1, axis=0).reshape(4 * ALLIANCE_SIZE, 6, DIM)
                wrong_relation_nrmse = field_metrics(wrong_relation, act)[0]
                metrics = {
                    "median_signed_nrmse": float(np.median(nrmse)),
                    "median_signed_explained_energy": float(np.median(explained)),
                    "median_active_sign_agreement": float(np.median(sign)),
                    "median_cosine_secondary": float(np.median(cosine)),
                    "source_permutation_nrmse": float(np.median(perm_nrmse)),
                    "source_permutation_advantage": float(np.median(perm_nrmse - nrmse)),
                    "wrong_relation_nrmse": float(np.median(wrong_relation_nrmse)),
                    "wrong_relation_advantage": float(np.median(wrong_relation_nrmse - nrmse)),
                }
                rows.append({"source_role": source_role, "alliance": alliance_name, "partition": partition, "metrics": metrics})
    np.save(OUT / "analysis/discovery_relation_components.float16.npy", frozen_components)
    np.save(OUT / "analysis/validation_relation_components.float16.npy", validation_components)
    criteria = protocol["descriptive_replication"]
    labels = {}
    for source_role in SOURCE_ROLES:
        own_rows = [r for r in rows if r["source_role"] == source_role and r["alliance"] == "own"]
        passes = [r["metrics"]["median_active_sign_agreement"] >= criteria["sign_agreement_min"] and r["metrics"]["source_permutation_advantage"] >= criteria["source_permutation_nrmse_advantage_min"] for r in own_rows]
        labels[source_role] = "replicated" if sum(passes) == criteria["required_partitions"] else ("partial" if any(passes) else "absent")
    report = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "created_at_utc": now(),
        "status": "role_specific_response_adjudicated",
        "labels": labels,
        "rows": rows,
        "coordinate_overlap": {"primary_query": len(set(lock["roles"]["primary"]["coordinates"]) & set(lock["roles"]["query"]["coordinates"])), "primary_relation": len(set(lock["roles"]["primary"]["coordinates"]) & set(relation_coordinates)), "query_relation": len(set(lock["roles"]["query"]["coordinates"]) & set(relation_coordinates))},
        "claim_boundary": "Signed field prediction tests role-specific effective response alliances. It neither identifies a unique circuit nor measures all information available in a role.",
        "next_authorization": "run_C174_target_edge_compression_and_C175_hyperedges_regardless_of_role_labels",
    }
    core.save(OUT / "analysis/role_specific_atlas.json", report)
    checks = {"rows": len(rows) == 12, "labels": set(labels) == set(SOURCE_ROLES), "finite": all(np.isfinite(list(r["metrics"].values())).all() for r in rows), "tensors": list(frozen_components.shape) == [2, 3, 4, 64, 6, 2560] and list(validation_components.shape) == [2, 3, 2, 4, 64, 6, 2560]}
    core.save(OUT / "audit/internal_analysis_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    print(json.dumps(report, indent=2))


def close():
    report = core.load(OUT / "analysis/role_specific_atlas.json")
    checks = {name: core.load(OUT / f"audit/{name}")["all_checks_passed"] for name in ("internal_contract_audit.json", "internal_discovery_run_audit.json", "internal_selection_audit.json", "internal_validation_run_audit.json", "internal_analysis_audit.json")}
    final = {"phase": PHASE, "campaign": CAMPAIGN, "status": "closed", "checks": checks, "all_checks_passed": all(checks.values()), "headline": {"labels": report["labels"], "coordinate_overlap": report["coordinate_overlap"], "own_rows": [r for r in report["rows"] if r["alliance"] == "own"]}, "next_authorization": report["next_authorization"]}
    core.save(OUT / "analysis/final.json", final)
    print(json.dumps(final, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("contract", "run-discovery", "select", "run-validation", "analyze", "close"))
    args = parser.parse_args()
    {"contract": contract, "run-discovery": run_discovery, "select": select, "run-validation": run_validation, "analyze": analyze, "close": close}[args.command]()


if __name__ == "__main__":
    main()
