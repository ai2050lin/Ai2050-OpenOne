#!/usr/bin/env python3
"""C175: pairwise non-additive response of locked role-specific coordinate pairs."""
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
OUT = RESULT / "phase1709_c175_role_pair_hyperedge_response"
C159 = RESULT / "phase1693_c159_natural_isomorphic_dual_graph_atlas"
C173 = RESULT / "phase1707_c173_role_specific_full_coordinate_response"
C174 = RESULT / "phase1708_c174_signed_target_edge_compression"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core
from phase1332_bf16_utils import load_bf16, quantization_audit, release_bf16
import phase1572_c099_fixed_width_graph_field_campaign as fixed_base
import phase1661_c127_typed_transition_language_family as c127
import phase1693_c159_natural_isomorphic_dual_graph_atlas as c159

PHASE, CAMPAIGN = 1709, "C175"
DIM, WIDTH, SOURCE_Q, TARGET_Q = 2560, 256, 24, 25
SOURCE_ROLES = ("primary", "query")
TARGET_ROLES = c159.ROLES
PANELS = ("natural_lexical", "isomorphic_nonce")
RELATIONS = ("is_a", "part_of", "located_in", "precedes")
PARTITIONS = ("discovery", "fresh")
PAIR_COUNT = 8


def now():
    return datetime.now(timezone.utc).isoformat()


def tensor(value):
    return value[0] if isinstance(value, tuple) else value


def anchors():
    rows = core.rows(C159 / "analysis/late_half_difference_index.jsonl")
    selected = []
    for part in PARTITIONS:
        for panel in PANELS:
            for relation in RELATIONS:
                choices = [r for r in rows if r["partition"] == part and r["panel"] == panel and r["relation_family"] == relation and r["path"] == -1 and r["interference"] == 1 and r["direction_form"] == 1 and r["surface"] == 1 and r["code"] == 1]
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
    parent = core.load(C174 / "audit/independent_final_audit.json")
    lock = core.load(C173 / "protocol/role_specific_coordinate_lock.json")
    selected = anchors()
    pairs = {role: [[int(coords[2 * i]), int(coords[2 * i + 1])] for i in range(PAIR_COUNT)] for role in SOURCE_ROLES for coords in [lock["roles"][role]["coordinates"]]}
    checks = {
        "authorization": parent["all_checks_passed"] and "C175" in parent["authorization"],
        "anchors": len(selected) == 16,
        "balanced": len({(r["partition"], r["panel"], r["relation_family"]) for r in selected}) == 16,
        "pairs": all(len(v) == PAIR_COUNT and len({i for pair in v for i in pair}) == PAIR_COUNT * 2 for v in pairs.values()),
        "selection_blind": lock["confirmation_and_fresh_unread"],
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    OUT.mkdir(parents=True)
    core.write_rows(OUT / "material/anchors.jsonl", selected)
    protocol = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "created_at_utc": now(),
        "status": "role_pair_hyperedge_contract_frozen",
        "model": "Qwen3-4B BF16 CUDA nonquantized",
        "source_checkpoint": SOURCE_Q,
        "target_checkpoint": TARGET_Q,
        "pairs": pairs,
        "partitions": list(PARTITIONS),
        "interaction": "K_ab=(H_ab-H_a-H_b+H_0)/epsilon^2",
        "primary_metrics": ["finite-step nonadditivity ratio", "fresh signed NRMSE", "fresh active-edge sign agreement", "pair-permutation NRMSE advantage"],
        "descriptive_hyperedge": {"nonadditivity_ratio_min": 0.05, "fresh_sign_agreement_min": 0.55, "pair_permutation_advantage_min": 0.02},
        "claim_boundary": "pairwise local curvature at locked coordinates; not a unique higher-order causal circuit",
        "producer_sha256": core.sha(Path(__file__)),
        "authorization": "run_C175_all_roles_partitions",
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    print(json.dumps({"checks": checks, "pairs": pairs}, indent=2))


@torch.inference_mode()
def run():
    protocol = core.load(OUT / "protocol/preregistration.json")
    selected = core.rows(OUT / "material/anchors.jsonl")
    compiled = core.rows(C159 / "compiled/qwen3.jsonl")
    states = np.load(C159 / "raw/qwen3_six_role_all_checkpoint.bf16.npy", mmap_mode="r")
    (OUT / "raw").mkdir(parents=True, exist_ok=True)
    interaction = np.lib.format.open_memmap(OUT / "raw/pair_interaction.float16.npy", mode="w+", dtype=np.float16, shape=(2, 2, 8, PAIR_COUNT, 6, DIM))
    epsilons = np.zeros((2, 2, 8), np.float32)
    model = None
    try:
        model, tokenizer, device, placement = load_bf16("qwen3")
        quant = quantization_audit(model)
        layers = model.model.layers
        pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)

        def condition_fields(row, source_role, pairs, epsilon):
            conditions = [tuple()]
            for a, b in pairs:
                conditions.extend(((a,), (b,), (a, b)))
            batch = [row] * len(conditions)
            ids, mask, pos, _lengths = fixed_base.fixed_batch(batch, pad, device, WIDTH)
            captured = {}

            def patch(_module, _args, value):
                hidden = tensor(value)
                patched = hidden.clone()
                for local, coordinates in enumerate(conditions):
                    for coordinate in coordinates:
                        for position in row["role_positions"][source_role]:
                            patched[local, position, int(coordinate)] += epsilon
                return (patched,) + value[1:] if isinstance(value, tuple) else patched

            h1 = layers[SOURCE_Q - 1].register_forward_hook(patch)
            h2 = layers[TARGET_Q - 1].register_forward_hook(lambda _m, _a, value: captured.__setitem__("state", tensor(value).detach()))
            try:
                model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False, return_dict=True)
            finally:
                h1.remove(); h2.remove()
            fields = np.zeros((len(conditions), 6, DIM), np.float32)
            for local in range(len(conditions)):
                for role_i, role in enumerate(TARGET_ROLES):
                    fields[local, role_i] = captured["state"][local, row["role_positions"][role]].mean(0).float().cpu().numpy()
            return fields

        for role_i, source_role in enumerate(SOURCE_ROLES):
            pairs = protocol["pairs"][source_role]
            role_index = TARGET_ROLES.index(source_role)
            for anchor_i, anchor in enumerate(selected):
                part_i = PARTITIONS.index(anchor["partition"])
                within = anchor_i - part_i * 8
                row = compiled[anchor["minus_row"]]
                source = c127.decode(states[anchor["minus_row"], role_index, SOURCE_Q])
                epsilon = 0.5 * float(np.sqrt(np.mean(np.square(source), dtype=np.float64)))
                epsilons[role_i, part_i, within] = epsilon
                fields = condition_fields(row, source_role, pairs, epsilon)
                base = fields[0]
                for pair_i in range(PAIR_COUNT):
                    a, b, ab = fields[1 + 3 * pair_i:1 + 3 * pair_i + 3]
                    interaction[role_i, part_i, within, pair_i] = ((ab - a - b + base) / (epsilon * epsilon)).astype(np.float16)
                interaction.flush()
                print(f"[C175] {source_role} {anchor['partition']} {within + 1}/8", flush=True)
    finally:
        interaction.flush()
        if model is not None:
            release_bf16(model)
        gc.collect(); torch.cuda.empty_cache()
    np.save(OUT / "raw/epsilons.float32.npy", epsilons)
    checks = {"shape": list(interaction.shape) == [2, 2, 8, 8, 6, 2560], "finite": bool(np.isfinite(interaction).all()), "epsilon": bool(np.all(epsilons > 0)), "bf16": bool(quant["has_bf16_parameters"] and not quant["has_quantized_modules"])}
    core.save(OUT / "analysis/run.json", {"checks": checks, "runtime": placement})
    core.save(OUT / "audit/internal_run_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    print(json.dumps(checks, indent=2))


def metrics(pred, actual):
    p = pred.reshape(len(pred), -1).astype(np.float64)
    a = actual.reshape(len(actual), -1).astype(np.float64)
    nrmse = np.linalg.norm(a - p, axis=1) / np.maximum(np.linalg.norm(a, axis=1), 1e-12)
    threshold = np.quantile(np.abs(p), 0.95, axis=1)
    sign = [float(np.mean(np.sign(a[i, np.abs(p[i]) >= threshold[i]]) == np.sign(p[i, np.abs(p[i]) >= threshold[i]]))) for i in range(len(p))]
    return nrmse, np.asarray(sign)


def analyze():
    protocol = core.load(OUT / "protocol/preregistration.json")
    raw = np.load(OUT / "raw/pair_interaction.float16.npy", mmap_mode="r")
    eps = np.load(OUT / "raw/epsilons.float32.npy")
    c173_discovery = np.load(C173 / "raw/discovery_full_response.float16.npy", mmap_mode="r")
    c173_validation = np.load(C173 / "raw/validation_response.float16.npy", mmap_mode="r")
    lock = core.load(C173 / "protocol/role_specific_coordinate_lock.json")
    rows = []
    for role_i, source_role in enumerate(SOURCE_ROLES):
        coords = lock["roles"][source_role]["coordinates"]
        pair_local = [(2 * i, 2 * i + 1) for i in range(PAIR_COUNT)]
        relation_components = []
        for part_i, partition in enumerate(PARTITIONS):
            x = np.asarray(raw[role_i, part_i], np.float32).reshape(2, 4, PAIR_COUNT, 6, DIM)
            relation_components.append(decompose(x)[2])
        pred = relation_components[0].reshape(4 * PAIR_COUNT, 6, DIM)
        actual = relation_components[1].reshape(4 * PAIR_COUNT, 6, DIM)
        nrmse, sign = metrics(pred, actual)
        perm_nrmse = metrics(np.roll(pred, 1, axis=0), actual)[0]

        ratios = {}
        for part_i, partition in enumerate(PARTITIONS):
            values = []
            for anchor_i in range(8):
                epsilon = float(eps[role_i, part_i, anchor_i])
                if partition == "discovery":
                    first = np.asarray(c173_discovery[role_i, anchor_i, coords], np.float32)
                else:
                    first = np.asarray(c173_validation[role_i, 0, 8 + anchor_i], np.float32)
                for pair_i, (a_i, b_i) in enumerate(pair_local):
                    nonlinear = np.asarray(raw[role_i, part_i, anchor_i, pair_i], np.float32) * epsilon * epsilon
                    linear = (first[a_i] + first[b_i]) * epsilon
                    values.append(float(np.linalg.norm(nonlinear) / max(np.linalg.norm(linear), 1e-12)))
            ratios[partition] = float(np.median(values))
        aggregate = {
            "discovery_nonadditivity_ratio": ratios["discovery"],
            "fresh_nonadditivity_ratio": ratios["fresh"],
            "fresh_signed_nrmse": float(np.median(nrmse)),
            "fresh_active_sign_agreement": float(np.median(sign)),
            "pair_permutation_nrmse": float(np.median(perm_nrmse)),
            "pair_permutation_advantage": float(np.median(perm_nrmse - nrmse)),
        }
        criteria = protocol["descriptive_hyperedge"]
        passes = {"nonadditive": aggregate["fresh_nonadditivity_ratio"] >= criteria["nonadditivity_ratio_min"], "sign": aggregate["fresh_active_sign_agreement"] >= criteria["fresh_sign_agreement_min"], "pair_identity": aggregate["pair_permutation_advantage"] >= criteria["pair_permutation_advantage_min"]}
        label = "replicated_hyperedge_candidate" if all(passes.values()) else ("nonadditive_not_replicated" if passes["nonadditive"] else "approximately_additive")
        rows.append({"source_role": source_role, "aggregate": aggregate, "criteria": passes, "label": label})
    report = {"phase": PHASE, "campaign": CAMPAIGN, "created_at_utc": now(), "status": "pairwise_hyperedge_adjudicated", "rows": rows, "claim_boundary": protocol["claim_boundary"], "next_authorization": "run_C176_broad_family_reuse_then_continue_natural_and_cross_model_branches"}
    core.save(OUT / "analysis/hyperedge_atlas.json", report)
    checks = {"roles": len(rows) == 2, "finite": all(np.isfinite(list(r["aggregate"].values())).all() for r in rows)}
    core.save(OUT / "audit/internal_analysis_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    print(json.dumps(report, indent=2))


def close():
    report = core.load(OUT / "analysis/hyperedge_atlas.json")
    checks = {"contract": core.load(OUT / "audit/internal_contract_audit.json")["all_checks_passed"], "run": core.load(OUT / "audit/internal_run_audit.json")["all_checks_passed"], "analysis": core.load(OUT / "audit/internal_analysis_audit.json")["all_checks_passed"]}
    final = {"phase": PHASE, "campaign": CAMPAIGN, "status": "closed", "checks": checks, "all_checks_passed": all(checks.values()), "headline": {"rows": report["rows"]}, "next_authorization": report["next_authorization"]}
    core.save(OUT / "analysis/final.json", final)
    print(json.dumps(final, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("contract", "run", "analyze", "close"))
    args = parser.parse_args()
    {"contract": contract, "run": run, "analyze": analyze, "close": close}[args.command]()


if __name__ == "__main__":
    main()
