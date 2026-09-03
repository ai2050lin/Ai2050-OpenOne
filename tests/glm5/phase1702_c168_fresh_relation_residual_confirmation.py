#!/usr/bin/env python3
"""C168: prospective fresh confirmation of C167 relation-residual transport."""
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
C159 = RESULT / "phase1693_c159_natural_isomorphic_dual_graph_atlas"
C161 = RESULT / "phase1695_c161_full_coordinate_local_transmission"
C167 = RESULT / "phase1701_c167_transport_component_decomposition"
OUT = RESULT / "phase1702_c168_fresh_relation_residual_confirmation"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core
from phase1332_bf16_utils import load_bf16, quantization_audit, release_bf16
import phase1572_c099_fixed_width_graph_field_campaign as fixed_base
import phase1661_c127_typed_transition_language_family as c127
import phase1693_c159_natural_isomorphic_dual_graph_atlas as c159

PHASE, CAMPAIGN = 1702, "C168"
DIM, WIDTH, BATCH, SOURCE_Q, TARGET_Q = 2560, 256, 16, 24, 25
ROLES = c159.ROLES
PANELS = ("natural_lexical", "isomorphic_nonce")
RELATIONS = ("is_a", "part_of", "located_in", "precedes")


def now():
    return datetime.now(timezone.utc).isoformat()


def tensor(value):
    return value[0] if isinstance(value, tuple) else value


def fresh_anchors():
    pairs = core.rows(C159 / "analysis/late_half_difference_index.jsonl")
    selected = []
    for panel in PANELS:
        for relation in RELATIONS:
            choices = [r for r in pairs if r["partition"] == "fresh" and r["panel"] == panel and r["relation_family"] == relation and r["path"] == -1 and r["interference"] == 1 and r["direction_form"] == 1 and r["surface"] == 1 and r["code"] == 1]
            if len(choices) != 1:
                raise RuntimeError((panel, relation, len(choices)))
            selected.append(dict(choices[0], anchor_index=len(selected)))
    return selected


def decompose(x):
    shared = x.mean(axis=(0, 1))
    panel = x.mean(axis=1) - shared[None]
    relation = x.mean(axis=0) - shared[None]
    interaction = x - shared[None, None] - panel[:, None] - relation[None]
    return shared, panel, relation, interaction


def cos_rows(a, b, source_ids=None, remove_identity=False):
    af = a.reshape(len(a), -1).astype(np.float64, copy=False)
    bf = b.reshape(len(b), -1).astype(np.float64, copy=False)
    dot = np.sum(af * bf, axis=1)
    an2 = np.sum(af * af, axis=1)
    bn2 = np.sum(bf * bf, axis=1)
    if remove_identity:
        local = np.arange(len(a))
        role = ROLES.index("relation")
        av = a[local, role, source_ids].astype(np.float64)
        bv = b[local, role, source_ids].astype(np.float64)
        dot -= av * bv
        an2 -= av * av
        bn2 -= bv * bv
    return dot / np.maximum(np.sqrt(np.maximum(an2, 0) * np.maximum(bn2, 0)), 1e-12)


def contract():
    if OUT.exists():
        raise RuntimeError(OUT)
    audits = [core.load(path / "audit/independent_final_audit.json") for path in (C159, C161, C167)]
    anchors = fresh_anchors()
    compiled = core.rows(C159 / "compiled/qwen3.jsonl")
    logits = np.load(C159 / "raw/qwen3_candidate_logits.float32.npy", mmap_mode="r")
    behavior = [int(np.argmax(logits[i]) == row["gold_position"]) for i, row in enumerate(compiled)]
    anchor_behavior = [behavior[r[key]] for r in anchors for key in ("minus_row", "plus_row")]
    coordinates = core.load(C167 / "analysis/top_relation_source_coordinates.json")["coordinates"][:64]
    checks = {
        "parent_audits": all(a["all_checks_passed"] for a in audits),
        "anchors": len(anchors) == 8,
        "balanced": len({(r["panel"], r["relation_family"]) for r in anchors}) == 8,
        "anchor_behavior": all(anchor_behavior),
        "coordinates": len(coordinates) == 64 and len(set(coordinates)) == 64,
        "discovery_selected": core.load(C167 / "protocol/preregistration.json")["epistemic_status"].startswith("retrospective"),
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    OUT.mkdir(parents=True)
    core.write_rows(OUT / "material/fresh_anchors.jsonl", anchors)
    protocol = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "created_at_utc": now(),
        "status": "fresh_relation_residual_confirmation_frozen",
        "model": "Qwen3-4B BF16 CUDA nonquantized",
        "source": {"checkpoint": SOURCE_Q, "role": "relation", "coordinates": coordinates},
        "target": {"checkpoint": TARGET_Q, "roles": list(ROLES), "coordinates_per_role": DIM},
        "fresh_material": "eight previously unperturbed fresh anchors: panel x relation",
        "reference": "mean of C161 discovery and confirmation balanced relation components; source coordinates ranked by C167 discovery relation energy",
        "perturbation": "symmetric plus/minus, epsilon=0.5 source-state RMS",
        "gates": {
            "matched_median_cosine_min": 0.30,
            "matched_wrong_relation_margin_min": 0.20,
            "identity_removed_margin_min": 0.20,
            "positive_margin_rate_min": 0.75,
            "source_permutation_advantage_min": 0.05,
        },
        "forbidden": ["attention", "MLP", "weights", "PCA", "fresh-informed coordinate selection"],
        "source_hashes": {
            "C159_states": core.sha(C159 / "raw/qwen3_six_role_all_checkpoint.bf16.npy"),
            "C161_response": core.sha(C161 / "raw/q24_relation_to_q25_six_role_response.float16.npy"),
            "C167_lock": core.sha(C167 / "analysis/top_relation_source_coordinates.json"),
        },
        "producer_sha256": core.sha(Path(__file__)),
        "authorization": "run_fresh_responses",
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": True})
    print(json.dumps({"checks": checks, "protocol": protocol}, indent=2))


@torch.inference_mode()
def run():
    protocol = core.load(OUT / "protocol/preregistration.json")
    anchors = core.rows(OUT / "material/fresh_anchors.jsonl")
    compiled = core.rows(C159 / "compiled/qwen3.jsonl")
    states = np.load(C159 / "raw/qwen3_six_role_all_checkpoint.bf16.npy", mmap_mode="r")
    coordinates = protocol["source"]["coordinates"]
    (OUT / "raw").mkdir(parents=True, exist_ok=True)
    response = np.lib.format.open_memmap(OUT / "raw/fresh_q24_q25_response.float16.npy", mode="w+", dtype=np.float16, shape=(8, 64, 6, DIM))
    epsilons = np.zeros(8, np.float32)
    model = None
    try:
        model, tokenizer, device, placement = load_bf16("qwen3")
        quant = quantization_audit(model)
        layers = model.model.layers
        pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
        relation_i = ROLES.index("relation")

        def perturb(row, coordinate_ids, sign, epsilon):
            batch = [row] * len(coordinate_ids)
            ids, mask, pos, _lengths = fixed_base.fixed_batch(batch, pad, device, WIDTH)
            captured = {}

            def patch(_module, _args, output):
                hidden = tensor(output)
                patched = hidden.clone()
                for local, coordinate in enumerate(coordinate_ids):
                    for position in row["role_positions"]["relation"]:
                        patched[local, position, int(coordinate)] += sign * epsilon
                return (patched,) + output[1:] if isinstance(output, tuple) else patched

            h1 = layers[SOURCE_Q - 1].register_forward_hook(patch)
            h2 = layers[TARGET_Q - 1].register_forward_hook(lambda _m, _a, o: captured.__setitem__("state", tensor(o).detach()))
            try:
                model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False, return_dict=True)
            finally:
                h1.remove(); h2.remove()
            field = np.zeros((len(coordinate_ids), 6, DIM), np.float32)
            for local in range(len(coordinate_ids)):
                for role_i, role in enumerate(ROLES):
                    field[local, role_i] = captured["state"][local, row["role_positions"][role]].mean(0).float().cpu().numpy()
            return field

        for ai, anchor in enumerate(anchors):
            row = compiled[anchor["minus_row"]]
            source = c127.decode(states[anchor["minus_row"], relation_i, SOURCE_Q])
            epsilon = 0.5 * float(np.sqrt(np.mean(np.square(source), dtype=np.float64)))
            epsilons[ai] = epsilon
            for start in range(0, 64, BATCH):
                ids_ = coordinates[start:start + BATCH]
                plus = perturb(row, ids_, 1.0, epsilon)
                minus = perturb(row, ids_, -1.0, epsilon)
                response[ai, start:start + len(ids_)] = ((plus - minus) / (2.0 * epsilon)).astype(np.float16)
            response.flush()
            print(f"[C168] {ai + 1}/8 {anchor['panel']} {anchor['relation_family']}", flush=True)
    finally:
        response.flush()
        if model is not None:
            release_bf16(model)
        gc.collect(); torch.cuda.empty_cache()
    np.save(OUT / "raw/epsilons.float32.npy", epsilons)
    checks = {
        "shape": list(response.shape) == [8, 64, 6, DIM],
        "finite": bool(np.isfinite(response).all()),
        "epsilon": bool(np.all(epsilons > 0)),
        "bf16": bool(quant["has_bf16_parameters"] and not quant["has_quantized_modules"]),
    }
    core.save(OUT / "analysis/run.json", {"checks": checks, "runtime": placement})
    core.save(OUT / "audit/internal_run_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    print(json.dumps(checks, indent=2))


def analyze():
    protocol = core.load(OUT / "protocol/preregistration.json")
    train_anchors = core.rows(C161 / "material/anchors.jsonl")
    train_raw = np.load(C161 / "raw/q24_relation_to_q25_six_role_response.float16.npy", mmap_mode="r")
    fresh_raw = np.load(OUT / "raw/fresh_q24_q25_response.float16.npy", mmap_mode="r")
    coordinates = np.asarray(protocol["source"]["coordinates"], int)
    train_index = {(r["partition"], r["panel"], r["relation_family"]): r["anchor_index"] for r in train_anchors}
    train_parts = []
    for part in ("discovery", "confirmation"):
        x = np.stack([np.stack([np.asarray(train_raw[train_index[(part, panel, relation)], coordinates], np.float32) for relation in RELATIONS]) for panel in PANELS])
        train_parts.append(decompose(x))
    reference_relation = np.mean(np.stack([v[2] for v in train_parts]), axis=0)
    reference_conditional = np.mean(np.stack([v[2][None] + v[3] for v in train_parts]), axis=0)
    fresh = np.asarray(fresh_raw, np.float32).reshape(2, 4, 64, 6, DIM)
    _shared, _panel, fresh_relation, fresh_interaction = decompose(fresh)
    fresh_conditional = fresh_relation[None] + fresh_interaction
    relation_rows, conditional_rows = [], []
    all_relation_margin, all_relation_no, all_relation_match, all_permuted = [], [], [], []
    all_conditional_margin = []
    permutation = np.roll(np.arange(64), 1)
    for ri, relation in enumerate(RELATIONS):
        matched = cos_rows(reference_relation[ri], fresh_relation[ri])
        wrong = np.median(np.stack([cos_rows(reference_relation[ri], fresh_relation[wj]) for wj in range(4) if wj != ri]), axis=0)
        matched_no = cos_rows(reference_relation[ri], fresh_relation[ri], coordinates, True)
        wrong_no = np.median(np.stack([cos_rows(reference_relation[ri], fresh_relation[wj], coordinates, True) for wj in range(4) if wj != ri]), axis=0)
        permuted = cos_rows(reference_relation[ri][permutation], fresh_relation[ri])
        all_relation_match.extend(matched.tolist()); all_relation_margin.extend((matched - wrong).tolist()); all_relation_no.extend((matched_no - wrong_no).tolist()); all_permuted.extend(permuted.tolist())
        relation_rows.append({"relation": relation, "matched_median_cosine": float(np.median(matched)), "wrong_median_cosine": float(np.median(wrong)), "margin": float(np.median(matched - wrong)), "identity_removed_margin": float(np.median(matched_no - wrong_no)), "source_permuted_median_cosine": float(np.median(permuted))})
        for pi, panel in enumerate(PANELS):
            matched_c = cos_rows(reference_conditional[pi, ri], fresh_conditional[pi, ri])
            wrong_c = np.median(np.stack([cos_rows(reference_conditional[pi, ri], fresh_conditional[pi, wj]) for wj in range(4) if wj != ri]), axis=0)
            all_conditional_margin.extend((matched_c - wrong_c).tolist())
            conditional_rows.append({"panel": panel, "relation": relation, "matched_median_cosine": float(np.median(matched_c)), "wrong_median_cosine": float(np.median(wrong_c)), "margin": float(np.median(matched_c - wrong_c))})
    aggregate = {
        "matched_median_cosine": float(np.median(all_relation_match)),
        "relation_margin": float(np.median(all_relation_margin)),
        "relation_positive_margin_rate": float(np.mean(np.asarray(all_relation_margin) > 0)),
        "identity_removed_relation_margin": float(np.median(all_relation_no)),
        "identity_removed_positive_margin_rate": float(np.mean(np.asarray(all_relation_no) > 0)),
        "source_permuted_median_cosine": float(np.median(all_permuted)),
        "source_permutation_advantage": float(np.median(all_relation_match) - np.median(all_permuted)),
        "panel_conditioned_margin": float(np.median(all_conditional_margin)),
    }
    gates = {
        "cosine": aggregate["matched_median_cosine"] >= protocol["gates"]["matched_median_cosine_min"],
        "relation_margin": aggregate["relation_margin"] >= protocol["gates"]["matched_wrong_relation_margin_min"],
        "identity_removed_margin": aggregate["identity_removed_relation_margin"] >= protocol["gates"]["identity_removed_margin_min"],
        "positive_rate": aggregate["relation_positive_margin_rate"] >= protocol["gates"]["positive_margin_rate_min"],
        "source_control": aggregate["source_permutation_advantage"] >= protocol["gates"]["source_permutation_advantage_min"],
    }
    (OUT / "analysis").mkdir(parents=True, exist_ok=True)
    np.save(OUT / "analysis/fresh_relation_components.float16.npy", fresh_relation.astype(np.float16))
    report = {"phase": PHASE, "campaign": CAMPAIGN, "created_at_utc": now(), "status": "fresh_relation_residual_adjudicated", "aggregate": aggregate, "relation_rows": relation_rows, "panel_conditioned_rows": conditional_rows, "gates": gates, "passed": all(gates.values()), "claim_boundary": "Fresh finite-response confirmation for 64 discovery-ranked source coordinates at q24 relation role to q25 six-role field; not whole-network or natural free-generation closure.", "next_authorization": "C169 coordinate heatmap and broader relation/role/checkpoint replication regardless of gate"}
    core.save(OUT / "analysis/confirmation.json", report)
    checks = {"relations": len(relation_rows) == 4, "conditional": len(conditional_rows) == 8, "finite": bool(np.isfinite(list(aggregate.values())).all()), "gates": len(gates) == 5}
    core.save(OUT / "audit/internal_analysis_audit.json", {"checks": checks, "all_checks_passed": all(checks.values()), "scientific_passed": all(gates.values())})
    print(json.dumps(report, indent=2))


def close():
    report = core.load(OUT / "analysis/confirmation.json")
    checks = {"contract": core.load(OUT / "audit/internal_contract_audit.json")["all_checks_passed"], "run": core.load(OUT / "audit/internal_run_audit.json")["all_checks_passed"], "analysis": core.load(OUT / "audit/internal_analysis_audit.json")["all_checks_passed"]}
    final = {"phase": PHASE, "campaign": CAMPAIGN, "status": "closed", "checks": checks, "all_checks_passed": all(checks.values()), "headline": {"aggregate": report["aggregate"], "gates": report["gates"], "passed": report["passed"]}, "next_authorization": report["next_authorization"]}
    core.save(OUT / "analysis/final.json", final)
    print(json.dumps(final, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("contract", "run", "analyze", "close"))
    args = parser.parse_args()
    {"contract": contract, "run": run, "analyze": analyze, "close": close}[args.command]()


if __name__ == "__main__":
    main()
