#!/usr/bin/env python3
"""C170: broad source-role/checkpoint atlas for relation-conditioned local transport."""
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
C167 = RESULT / "phase1701_c167_transport_component_decomposition"
C168 = RESULT / "phase1702_c168_fresh_relation_residual_confirmation"
OUT = RESULT / "phase1704_c170_role_checkpoint_relation_transport_atlas"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core
from phase1332_bf16_utils import load_bf16, quantization_audit, release_bf16
import phase1572_c099_fixed_width_graph_field_campaign as fixed_base
import phase1661_c127_typed_transition_language_family as c127
import phase1693_c159_natural_isomorphic_dual_graph_atlas as c159

PHASE, CAMPAIGN = 1704, "C170"
DIM, WIDTH, BATCH = 2560, 256, 16
ROLES = c159.ROLES
PANELS = ("natural_lexical", "isomorphic_nonce")
RELATIONS = ("is_a", "part_of", "located_in", "precedes")
PARTITIONS = ("discovery", "confirmation", "fresh")
SOURCE_ROLES = ("primary", "relation", "query")
SOURCE_QS = (23, 24, 25)
SETTINGS = tuple((q, role) for q in SOURCE_QS for role in SOURCE_ROLES)


def now():
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


def cos_rows(a, b, source_ids=None, source_role=None, remove_identity=False):
    af = a.reshape(len(a), -1).astype(np.float64, copy=False)
    bf = b.reshape(len(b), -1).astype(np.float64, copy=False)
    dot = np.sum(af * bf, axis=1)
    an2 = np.sum(af * af, axis=1)
    bn2 = np.sum(bf * bf, axis=1)
    if remove_identity:
        local = np.arange(len(a))
        role_i = ROLES.index(source_role)
        av = a[local, role_i, source_ids].astype(np.float64)
        bv = b[local, role_i, source_ids].astype(np.float64)
        dot -= av * bv
        an2 -= av * av
        bn2 -= bv * bv
    return dot / np.maximum(np.sqrt(np.maximum(an2, 0) * np.maximum(bn2, 0)), 1e-12)


def contract():
    if OUT.exists():
        raise RuntimeError(OUT)
    audits = [core.load(path / "audit/independent_final_audit.json") for path in (C159, C167, C168)]
    selected = anchors()
    compiled = core.rows(C159 / "compiled/qwen3.jsonl")
    logits = np.load(C159 / "raw/qwen3_candidate_logits.float32.npy", mmap_mode="r")
    behavior = [int(np.argmax(logits[i]) == row["gold_position"]) for i, row in enumerate(compiled)]
    anchor_behavior = [behavior[r[key]] for r in selected for key in ("minus_row", "plus_row")]
    coordinates = core.load(C167 / "analysis/top_relation_source_coordinates.json")["coordinates"][:16]
    checks = {
        "parent_audits": all(a["all_checks_passed"] for a in audits),
        "anchors": len(selected) == 24,
        "balanced": len({(r["partition"], r["panel"], r["relation_family"]) for r in selected}) == 24,
        "anchor_behavior": all(anchor_behavior),
        "settings": len(SETTINGS) == 9,
        "coordinates": len(coordinates) == 16 and len(set(coordinates)) == 16,
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    OUT.mkdir(parents=True)
    core.write_rows(OUT / "material/anchors.jsonl", selected)
    protocol = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "created_at_utc": now(),
        "status": "role_checkpoint_relation_transport_atlas_frozen",
        "model": "Qwen3-4B BF16 CUDA nonquantized",
        "source_settings": [{"checkpoint": q, "role": role, "target_checkpoint": q + 1} for q, role in SETTINGS],
        "source_coordinates": coordinates,
        "material": "24 behavior-correct anchors: partition x panel x relation",
        "response": "symmetric finite response at next checkpoint, all six roles x 2560 coordinates",
        "setting_labels": {"stable": "all four descriptive criteria pass", "partial": "at least two pass", "absent": "fewer than two pass"},
        "descriptive_criteria": {"matched_cosine": 0.30, "relation_margin": 0.20, "positive_rate": 0.75, "source_advantage": 0.05},
        "campaign_policy": "evaluate all nine settings regardless of individual labels",
        "forbidden": ["attention", "MLP", "weights", "PCA", "single-setting campaign stop"],
        "source_hashes": {"C159": core.sha(C159 / "raw/qwen3_six_role_all_checkpoint.bf16.npy"), "coordinate_lock": core.sha(C167 / "analysis/top_relation_source_coordinates.json")},
        "producer_sha256": core.sha(Path(__file__)),
        "authorization": "run_all_nine_settings",
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": True})
    print(json.dumps({"checks": checks, "estimated_raw_bytes": len(SETTINGS) * 24 * 16 * 6 * DIM * 2}, indent=2))


@torch.inference_mode()
def run():
    protocol = core.load(OUT / "protocol/preregistration.json")
    selected = core.rows(OUT / "material/anchors.jsonl")
    compiled = core.rows(C159 / "compiled/qwen3.jsonl")
    states = np.load(C159 / "raw/qwen3_six_role_all_checkpoint.bf16.npy", mmap_mode="r")
    coordinates = protocol["source_coordinates"]
    (OUT / "raw").mkdir(parents=True, exist_ok=True)
    response = np.lib.format.open_memmap(OUT / "raw/role_checkpoint_response.float16.npy", mode="w+", dtype=np.float16, shape=(9, 24, 16, 6, DIM))
    epsilons = np.zeros((9, 24), np.float32)
    model = None
    try:
        model, tokenizer, device, placement = load_bf16("qwen3")
        quant = quantization_audit(model)
        layers = model.model.layers
        pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)

        def perturb(row, source_q, target_q, source_role, sign, epsilon):
            batch = [row] * len(coordinates)
            ids, mask, pos, _lengths = fixed_base.fixed_batch(batch, pad, device, WIDTH)
            captured = {}

            def patch(_module, _args, output):
                hidden = tensor(output)
                patched = hidden.clone()
                for local, coordinate in enumerate(coordinates):
                    for position in row["role_positions"][source_role]:
                        patched[local, position, int(coordinate)] += sign * epsilon
                return (patched,) + output[1:] if isinstance(output, tuple) else patched

            h1 = layers[source_q - 1].register_forward_hook(patch)
            h2 = layers[target_q - 1].register_forward_hook(lambda _m, _a, o: captured.__setitem__("state", tensor(o).detach()))
            try:
                model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False, return_dict=True)
            finally:
                h1.remove(); h2.remove()
            field = np.zeros((16, 6, DIM), np.float32)
            for local in range(16):
                for role_i, role in enumerate(ROLES):
                    field[local, role_i] = captured["state"][local, row["role_positions"][role]].mean(0).float().cpu().numpy()
            return field

        for setting_i, (source_q, source_role) in enumerate(SETTINGS):
            role_i = ROLES.index(source_role)
            for anchor_i, anchor in enumerate(selected):
                row = compiled[anchor["minus_row"]]
                source = c127.decode(states[anchor["minus_row"], role_i, source_q])
                epsilon = 0.5 * float(np.sqrt(np.mean(np.square(source), dtype=np.float64)))
                epsilons[setting_i, anchor_i] = epsilon
                plus = perturb(row, source_q, source_q + 1, source_role, 1.0, epsilon)
                minus = perturb(row, source_q, source_q + 1, source_role, -1.0, epsilon)
                response[setting_i, anchor_i] = ((plus - minus) / (2.0 * epsilon)).astype(np.float16)
            response.flush()
            print(f"[C170] setting {setting_i + 1}/9 q{source_q} {source_role}", flush=True)
    finally:
        response.flush()
        if model is not None:
            release_bf16(model)
        gc.collect(); torch.cuda.empty_cache()
    np.save(OUT / "raw/epsilons.float32.npy", epsilons)
    checks = {"shape": list(response.shape) == [9, 24, 16, 6, DIM], "finite": bool(np.isfinite(response).all()), "epsilon": bool(np.all(epsilons > 0)), "bf16": bool(quant["has_bf16_parameters"] and not quant["has_quantized_modules"])}
    core.save(OUT / "analysis/run.json", {"checks": checks, "runtime": placement})
    core.save(OUT / "audit/internal_run_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    print(json.dumps(checks, indent=2))


def analyze():
    protocol = core.load(OUT / "protocol/preregistration.json")
    raw = np.load(OUT / "raw/role_checkpoint_response.float16.npy", mmap_mode="r")
    coordinates = np.asarray(protocol["source_coordinates"], int)
    criteria = protocol["descriptive_criteria"]
    rows = []
    fresh_components = np.zeros((9, 4, 16, 6, DIM), np.float16)
    for setting_i, (source_q, source_role) in enumerate(SETTINGS):
        x = np.asarray(raw[setting_i], np.float32).reshape(3, 2, 4, 16, 6, DIM)
        parts = [decompose(x[pi]) for pi in range(3)]
        reference = np.mean(np.stack([parts[0][2], parts[1][2]]), axis=0)
        fresh_relation = parts[2][2]
        fresh_components[setting_i] = fresh_relation.astype(np.float16)
        matched_all, margin_all, no_all, permuted_all = [], [], [], []
        permutation = np.roll(np.arange(16), 1)
        relation_rows = []
        for ri, relation in enumerate(RELATIONS):
            matched = cos_rows(reference[ri], fresh_relation[ri])
            wrong = np.median(np.stack([cos_rows(reference[ri], fresh_relation[wj]) for wj in range(4) if wj != ri]), axis=0)
            matched_no = cos_rows(reference[ri], fresh_relation[ri], coordinates, source_role, True)
            wrong_no = np.median(np.stack([cos_rows(reference[ri], fresh_relation[wj], coordinates, source_role, True) for wj in range(4) if wj != ri]), axis=0)
            permuted = cos_rows(reference[ri][permutation], fresh_relation[ri])
            matched_all.extend(matched.tolist()); margin_all.extend((matched - wrong).tolist()); no_all.extend((matched_no - wrong_no).tolist()); permuted_all.extend(permuted.tolist())
            relation_rows.append({"relation": relation, "matched_median_cosine": float(np.median(matched)), "margin": float(np.median(matched - wrong)), "identity_removed_margin": float(np.median(matched_no - wrong_no))})
        aggregate = {"matched_median_cosine": float(np.median(matched_all)), "relation_margin": float(np.median(margin_all)), "positive_margin_rate": float(np.mean(np.asarray(margin_all) > 0)), "identity_removed_margin": float(np.median(no_all)), "source_permuted_median_cosine": float(np.median(permuted_all)), "source_advantage": float(np.median(matched_all) - np.median(permuted_all))}
        passes = {"cosine": aggregate["matched_median_cosine"] >= criteria["matched_cosine"], "margin": aggregate["relation_margin"] >= criteria["relation_margin"], "positive_rate": aggregate["positive_margin_rate"] >= criteria["positive_rate"], "source_advantage": aggregate["source_advantage"] >= criteria["source_advantage"]}
        count = sum(passes.values())
        label = "stable" if count == 4 else ("partial" if count >= 2 else "absent")
        rows.append({"source_checkpoint": source_q, "target_checkpoint": source_q + 1, "source_role": source_role, "aggregate": aggregate, "criteria": passes, "label": label, "relation_rows": relation_rows})
    (OUT / "analysis").mkdir(parents=True, exist_ok=True)
    np.save(OUT / "analysis/fresh_relation_components.float16.npy", fresh_components)
    labels = {label: sum(row["label"] == label for row in rows) for label in ("stable", "partial", "absent")}
    report = {"phase": PHASE, "campaign": CAMPAIGN, "created_at_utc": now(), "status": "role_checkpoint_atlas_adjudicated", "settings": rows, "label_counts": labels, "campaign_complete": len(rows) == 9, "claim_boundary": "Broad descriptive atlas over nine settings with prospective fresh split, but coordinates were selected at q24 relation role and are not guaranteed optimal elsewhere; no natural-use necessity or minimal circuit claim.", "next_authorization": "synthesize role/checkpoint topology and design new coordinate selection for settings that are partial or absent"}
    core.save(OUT / "analysis/atlas.json", report)
    checks = {"settings": len(rows) == 9, "all_labels": sum(labels.values()) == 9, "tensor": list(fresh_components.shape) == [9, 4, 16, 6, DIM], "finite": bool(np.isfinite(fresh_components).all()), "campaign": report["campaign_complete"]}
    core.save(OUT / "audit/internal_analysis_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    print(json.dumps(report, indent=2))


def close():
    report = core.load(OUT / "analysis/atlas.json")
    checks = {"contract": core.load(OUT / "audit/internal_contract_audit.json")["all_checks_passed"], "run": core.load(OUT / "audit/internal_run_audit.json")["all_checks_passed"], "analysis": core.load(OUT / "audit/internal_analysis_audit.json")["all_checks_passed"]}
    final = {"phase": PHASE, "campaign": CAMPAIGN, "status": "closed", "checks": checks, "all_checks_passed": all(checks.values()), "headline": {"label_counts": report["label_counts"], "settings": [{"q": r["source_checkpoint"], "role": r["source_role"], "label": r["label"], **r["aggregate"]} for r in report["settings"]]}, "next_authorization": report["next_authorization"]}
    core.save(OUT / "analysis/final.json", final)
    print(json.dumps(final, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("contract", "run", "analyze", "close"))
    args = parser.parse_args()
    {"contract": contract, "run": run, "analyze": analyze, "close": close}[args.command]()


if __name__ == "__main__":
    main()
