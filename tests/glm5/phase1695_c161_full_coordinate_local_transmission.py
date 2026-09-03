#!/usr/bin/env python3
"""C161: exhaustive q24 relation-coordinate to q25 six-role finite-response graph."""
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
OUT = RESULT / "phase1695_c161_full_coordinate_local_transmission"
C159 = RESULT / "phase1693_c159_natural_isomorphic_dual_graph_atlas"
C160 = RESULT / "phase1694_c160_recipient_only_counterfactual_prediction"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core
from phase1332_bf16_utils import load_bf16, quantization_audit, release_bf16
import phase1572_c099_fixed_width_graph_field_campaign as fixed_base
import phase1661_c127_typed_transition_language_family as c127
import phase1693_c159_natural_isomorphic_dual_graph_atlas as c159

PHASE, CAMPAIGN = 1695, "C161"
DIM, WIDTH, BATCH = 2560, 256, 16
SOURCE_Q, TARGET_Q = 24, 25
ROLES = c159.ROLES


def now():
    return datetime.now(timezone.utc).isoformat()


def tensor(value):
    return value[0] if isinstance(value, tuple) else value


def anchors():
    pairs = core.rows(C159 / "analysis/late_half_difference_index.jsonl")
    selected = []
    for part in ("discovery", "confirmation"):
        for panel in c159.PANELS:
            for relation in ("is_a", "part_of", "located_in", "precedes"):
                choices = [row for row in pairs if row["partition"] == part and row["panel"] == panel and row["relation_family"] == relation and row["path"] == -1 and row["interference"] == 1 and row["direction_form"] == 1 and row["surface"] == 1 and row["code"] == 1]
                if len(choices) != 1:
                    raise RuntimeError((part, panel, relation, len(choices)))
                row = dict(choices[0])
                row["anchor_index"] = len(selected)
                selected.append(row)
    return selected


def contract():
    if OUT.exists():
        raise RuntimeError(OUT)
    parent = core.load(C160 / "audit/independent_final_audit.json")
    selected = anchors()
    checks = {
        "authorization": parent["all_checks_passed"],
        "recipient_prediction_pass": parent["scientific_fresh_passed"],
        "anchors": len(selected) == 16,
        "balanced": all(sum(row["partition"] == part for row in selected) == 8 for part in ("discovery", "confirmation")),
        "panels_relations": len({(row["partition"], row["panel"], row["relation_family"]) for row in selected}) == 16,
        "all_source_coordinates": DIM == 2560,
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    OUT.mkdir(parents=True)
    core.write_rows(OUT / "material/anchors.jsonl", selected)
    protocol = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "created_at_utc": now(),
        "status": "full_coordinate_local_transmission_contract_frozen",
        "model": "Qwen3-4B BF16 CUDA nonquantized",
        "source": {"checkpoint": SOURCE_Q, "role": "relation", "coordinates": DIM},
        "target": {"checkpoint": TARGET_Q, "roles": list(ROLES), "coordinates_per_role": DIM},
        "anchors": "one frozen path-length-3 clean anchor per partition x panel x relation family",
        "perturbation": "symmetric plus/minus, epsilon=0.5 times source-role state RMS",
        "response": "(H25_plus-H25_minus)/(2 epsilon)",
        "discovery": "all 2560 outgoing response norms select 16 source coordinates and 8 disjoint pairs",
        "confirmation": "same full graph plus one-sided second-order interaction for the eight locked pairs",
        "gates": {"median_discovery_confirmation_coordinate_cosine_min": 0.15, "stable_coordinate_count_min": 64, "stable_coordinate_cosine_min": 0.30},
        "claim_boundary": "exhaustive source-coordinate finite-response slice at one role/checkpoint; not a unique circuit or Attention/MLP attribution",
        "forbidden": ["attention", "MLP", "weights", "PCA", "confirmation-informed coordinate selection"],
        "source_hashes": {"C159": core.sha(C159 / "raw/qwen3_six_role_all_checkpoint.bf16.npy"), "C160": core.sha(C160 / "analysis/fresh_selected_predictions.float16.npy")},
        "producer_sha256": core.sha(Path(__file__)),
        "authorization": "run_C161_full_first_order",
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": True, "authorization": protocol["authorization"]})
    print(json.dumps({"checks": checks, "response_shape": [16, 2560, 6, 2560], "estimated_bytes_float16": 16 * 2560 * 6 * 2560 * 2}, indent=2))


@torch.inference_mode()
def run_first_order():
    anchors_ = core.rows(OUT / "material/anchors.jsonl")
    all_rows = core.rows(C159 / "compiled/qwen3.jsonl")
    raw = np.load(C159 / "raw/qwen3_six_role_all_checkpoint.bf16.npy", mmap_mode="r")
    (OUT / "raw").mkdir(parents=True, exist_ok=True)
    response = np.lib.format.open_memmap(OUT / "raw/q24_relation_to_q25_six_role_response.float16.npy", mode="w+", dtype=np.float16, shape=(16, DIM, 6, DIM))
    logit_response = np.lib.format.open_memmap(OUT / "raw/q24_relation_to_output_margin_response.float32.npy", mode="w+", dtype=np.float32, shape=(16, DIM))
    epsilons = np.zeros(16, np.float32)
    model = None
    try:
        model, tokenizer, device, placement = load_bf16("qwen3")
        quant = quantization_audit(model)
        layers = model.model.layers
        pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
        relation_index = ROLES.index("relation")

        def perturb(row, coordinate_ids, sign, epsilon):
            batch = [row] * len(coordinate_ids)
            ids, mask, pos, lengths = fixed_base.fixed_batch(batch, pad, device, WIDTH)
            captured = {}

            def patch(_module, _args, output):
                hidden = tensor(output)
                patched = hidden.clone()
                for local, coordinate in enumerate(coordinate_ids):
                    for position in row["role_positions"]["relation"]:
                        patched[local, position, int(coordinate)] += sign * epsilon
                return (patched,) + output[1:] if isinstance(output, tuple) else patched

            def capture(_module, _args, output):
                captured["state"] = tensor(output).detach()

            h1 = layers[SOURCE_Q - 1].register_forward_hook(patch)
            h2 = layers[TARGET_Q - 1].register_forward_hook(capture)
            try:
                output = model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False, return_dict=True)
            finally:
                h1.remove(); h2.remove()
            field = np.zeros((len(coordinate_ids), 6, DIM), np.float32)
            for local in range(len(coordinate_ids)):
                for role_i, role in enumerate(ROLES):
                    field[local, role_i] = captured["state"][local, row["role_positions"][role]].mean(0).float().cpu().numpy()
            score = np.asarray([[float(output.logits[i, lengths[i] - 1, candidate[0]]) for candidate in row["candidate_ids"]] for i in range(len(coordinate_ids))], np.float32)
            gold = int(row["gold_position"])
            margin = score[:, gold] - score[:, 1 - gold]
            return field, margin

        for anchor_i, anchor in enumerate(anchors_):
            row = all_rows[anchor["minus_row"]]
            source = c127.decode(raw[anchor["minus_row"], relation_index, SOURCE_Q])
            epsilon = 0.5 * float(np.sqrt(np.mean(np.square(source), dtype=np.float64)))
            epsilons[anchor_i] = epsilon
            for start in range(0, DIM, BATCH):
                coordinate_ids = np.arange(start, min(start + BATCH, DIM))
                plus, plus_margin = perturb(row, coordinate_ids, 1.0, epsilon)
                minus, minus_margin = perturb(row, coordinate_ids, -1.0, epsilon)
                response[anchor_i, coordinate_ids] = ((plus - minus) / (2.0 * epsilon)).astype(np.float16)
                logit_response[anchor_i, coordinate_ids] = (plus_margin - minus_margin) / (2.0 * epsilon)
            response.flush(); logit_response.flush(); print(f"[C161-first] anchor {anchor_i + 1}/16 {anchor['partition']} {anchor['panel']} {anchor['relation_family']}", flush=True)
    finally:
        response.flush(); logit_response.flush()
        if model is not None:
            release_bf16(model)
        gc.collect(); torch.cuda.empty_cache()
    np.save(OUT / "raw/anchor_epsilons.float32.npy", epsilons)
    checks = {"shape": list(response.shape) == [16, DIM, 6, DIM], "logit_shape": list(logit_response.shape) == [16, DIM], "finite": bool(np.isfinite(response).all() and np.isfinite(logit_response).all()), "epsilon": bool(np.all(epsilons > 0)), "bf16": bool(quant["has_bf16_parameters"] and not quant["has_quantized_modules"])}
    core.save(OUT / "analysis/first_order_run.json", {"phase": PHASE, "campaign": CAMPAIGN, "status": "first_order_complete", "checks": checks, "runtime": placement, "authorization": "select_C161_pairs_from_discovery_only"})
    core.save(OUT / "audit/internal_first_order_run_audit.json", {"checks": checks, "all_checks_passed": all(checks.values()), "authorization": "select_C161_pairs_from_discovery_only"})
    print(json.dumps(checks, indent=2))


def select_pairs():
    anchors_ = core.rows(OUT / "material/anchors.jsonl")
    response = np.load(OUT / "raw/q24_relation_to_q25_six_role_response.float16.npy", mmap_mode="r")
    discovery = [row["anchor_index"] for row in anchors_ if row["partition"] == "discovery"]
    norm = np.mean([np.linalg.norm(np.asarray(response[i], np.float32).reshape(DIM, -1), axis=1) for i in discovery], axis=0)
    top = np.argsort(norm)[-16:][::-1]
    pairs = [[int(top[2 * i]), int(top[2 * i + 1])] for i in range(8)]
    lock = {"phase": PHASE, "campaign": CAMPAIGN, "created_at_utc": now(), "selection_source": "discovery anchors only", "top_coordinates": top.tolist(), "outgoing_norms": norm[top].tolist(), "pairs": pairs, "confirmation_interactions_unread": True, "authorization": "run_C161_confirmation_second_order"}
    core.save(OUT / "protocol/discovery_pair_selection_lock.json", lock)
    print(json.dumps(lock, indent=2))


@torch.inference_mode()
def run_second_order():
    lock = core.load(OUT / "protocol/discovery_pair_selection_lock.json")
    anchors_ = [row for row in core.rows(OUT / "material/anchors.jsonl") if row["partition"] == "confirmation"]
    all_rows = core.rows(C159 / "compiled/qwen3.jsonl")
    raw = np.load(C159 / "raw/qwen3_six_role_all_checkpoint.bf16.npy", mmap_mode="r")
    epsilons = np.load(OUT / "raw/anchor_epsilons.float32.npy")
    interaction = np.lib.format.open_memmap(OUT / "raw/confirmation_second_order_interactions.float16.npy", mode="w+", dtype=np.float16, shape=(8, 8, 6, DIM))
    model = None
    try:
        model, tokenizer, device, placement = load_bf16("qwen3")
        quant = quantization_audit(model)
        layers = model.model.layers
        pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)

        def state(row, coordinates, epsilon):
            batch = [row]
            ids, mask, pos, _lengths = fixed_base.fixed_batch(batch, pad, device, WIDTH)
            captured = {}

            def patch(_module, _args, output):
                hidden = tensor(output)
                patched = hidden.clone()
                for coordinate in coordinates:
                    for position in row["role_positions"]["relation"]:
                        patched[0, position, int(coordinate)] += epsilon
                return (patched,) + output[1:] if isinstance(output, tuple) else patched

            h1 = layers[SOURCE_Q - 1].register_forward_hook(patch) if coordinates else None
            h2 = layers[TARGET_Q - 1].register_forward_hook(lambda _m, _a, o: captured.__setitem__("state", tensor(o).detach()))
            try:
                model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False, return_dict=True)
            finally:
                if h1 is not None: h1.remove()
                h2.remove()
            out = np.zeros((6, DIM), np.float32)
            for role_i, role in enumerate(ROLES):
                out[role_i] = captured["state"][0, row["role_positions"][role]].mean(0).float().cpu().numpy()
            return out

        for ai, anchor in enumerate(anchors_):
            row = all_rows[anchor["minus_row"]]
            epsilon = float(epsilons[anchor["anchor_index"]])
            base = state(row, [], epsilon)
            for pi, pair in enumerate(lock["pairs"]):
                a = state(row, [pair[0]], epsilon)
                b = state(row, [pair[1]], epsilon)
                ab = state(row, pair, epsilon)
                interaction[ai, pi] = ((ab - a - b + base) / (epsilon * epsilon)).astype(np.float16)
            interaction.flush(); print(f"[C161-second] anchor {ai + 1}/8", flush=True)
    finally:
        interaction.flush()
        if model is not None:
            release_bf16(model)
        gc.collect(); torch.cuda.empty_cache()
    checks = {"shape": list(interaction.shape) == [8, 8, 6, DIM], "finite": bool(np.isfinite(interaction).all()), "bf16": bool(quant["has_bf16_parameters"] and not quant["has_quantized_modules"])}
    core.save(OUT / "analysis/second_order_run.json", {"phase": PHASE, "campaign": CAMPAIGN, "status": "second_order_complete", "checks": checks, "runtime": placement, "authorization": "analyze_C161"})
    core.save(OUT / "audit/internal_second_order_run_audit.json", {"checks": checks, "all_checks_passed": all(checks.values()), "authorization": "analyze_C161"})
    print(json.dumps(checks, indent=2))


def cosine_rows(a, b):
    dot = np.sum(a * b, axis=1, dtype=np.float64)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return dot / np.maximum(den, 1e-12)


def coordinate_cosines_with_identity_diagnostic(a, b):
    """Return full and same-coordinate-edge-removed cosine for every source coordinate."""
    dot = np.sum(a * b, axis=1, dtype=np.float64)
    an2 = np.sum(a * a, axis=1, dtype=np.float64)
    bn2 = np.sum(b * b, axis=1, dtype=np.float64)
    full = dot / np.maximum(np.sqrt(an2 * bn2), 1e-12)
    relation_role = ROLES.index("relation")
    ids = np.arange(DIM)
    av = a.reshape(DIM, 6, DIM)[ids, relation_role, ids].astype(np.float64)
    bv = b.reshape(DIM, 6, DIM)[ids, relation_role, ids].astype(np.float64)
    removed = (dot - av * bv) / np.maximum(np.sqrt(np.maximum(an2 - av * av, 0) * np.maximum(bn2 - bv * bv, 0)), 1e-12)
    identity_fraction_a = av * av / np.maximum(an2, 1e-12)
    identity_fraction_b = bv * bv / np.maximum(bn2, 1e-12)
    return full, removed, identity_fraction_a, identity_fraction_b


def analyze():
    protocol = core.load(OUT / "protocol/preregistration.json")
    anchors_ = core.rows(OUT / "material/anchors.jsonl")
    response = np.load(OUT / "raw/q24_relation_to_q25_six_role_response.float16.npy", mmap_mode="r")
    logit_response = np.load(OUT / "raw/q24_relation_to_output_margin_response.float32.npy", mmap_mode="r")
    comparisons = []
    stable_union = set()
    for panel in c159.PANELS:
        for relation in ("is_a", "part_of", "located_in", "precedes"):
            d = next(row["anchor_index"] for row in anchors_ if row["partition"] == "discovery" and row["panel"] == panel and row["relation_family"] == relation)
            c = next(row["anchor_index"] for row in anchors_ if row["partition"] == "confirmation" and row["panel"] == panel and row["relation_family"] == relation)
            a = np.asarray(response[d], np.float32).reshape(DIM, -1)
            b = np.asarray(response[c], np.float32).reshape(DIM, -1)
            cos, removed, identity_a, identity_b = coordinate_cosines_with_identity_diagnostic(a, b)
            an, bn = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
            active = (an >= np.median(an)) & (bn >= np.median(bn))
            stable = np.where(active & (cos >= protocol["gates"]["stable_coordinate_cosine_min"]))[0]
            stable_union.update(stable.tolist())
            comparisons.append({"panel": panel, "relation_family": relation, "median_coordinate_cosine": float(np.median(cos)), "identity_removed_median_cosine": float(np.median(removed)), "median_same_coordinate_energy_fraction": float(np.median(np.concatenate((identity_a, identity_b)))), "active_median_cosine": float(np.median(cos[active])), "stable_count": int(len(stable)), "top_stable_coordinates": stable[np.argsort((an[stable] + bn[stable]))[-32:][::-1]].tolist() if len(stable) else []})
    overall_median = float(np.median([row["median_coordinate_cosine"] for row in comparisons]))
    stable_count = len(stable_union)
    gates = {"cosine": overall_median >= protocol["gates"]["median_discovery_confirmation_coordinate_cosine_min"], "stable_count": stable_count >= protocol["gates"]["stable_coordinate_count_min"]}
    relation_specificity = []
    for panel in c159.PANELS:
        for relation in ("is_a", "part_of", "located_in", "precedes"):
            d = next(row["anchor_index"] for row in anchors_ if row["partition"] == "discovery" and row["panel"] == panel and row["relation_family"] == relation)
            c_match = next(row["anchor_index"] for row in anchors_ if row["partition"] == "confirmation" and row["panel"] == panel and row["relation_family"] == relation)
            source = np.asarray(response[d], np.float32).reshape(DIM, -1)
            match = np.asarray(response[c_match], np.float32).reshape(DIM, -1)
            matched = float(np.median(cosine_rows(source, match)))
            wrong = []
            for other in ("is_a", "part_of", "located_in", "precedes"):
                if other == relation:
                    continue
                c_wrong = next(row["anchor_index"] for row in anchors_ if row["partition"] == "confirmation" and row["panel"] == panel and row["relation_family"] == other)
                wrong_target = np.asarray(response[c_wrong], np.float32).reshape(DIM, -1)
                wrong.append(float(np.median(cosine_rows(source, wrong_target))))
            relation_specificity.append({"panel": panel, "relation_family": relation, "matched_median_cosine": matched, "wrong_relation_median_cosine": float(np.median(wrong)), "margin": matched - float(np.median(wrong))})
    specificity_margin = float(np.median([row["margin"] for row in relation_specificity]))
    identity_removed_median = float(np.median([row["identity_removed_median_cosine"] for row in comparisons]))
    identity_energy_fraction = float(np.median([row["median_same_coordinate_energy_fraction"] for row in comparisons]))
    lock = core.load(OUT / "protocol/discovery_pair_selection_lock.json")
    interaction = np.load(OUT / "raw/confirmation_second_order_interactions.float16.npy", mmap_mode="r")
    interaction_norm = np.linalg.norm(np.asarray(interaction, np.float32).reshape(8, 8, -1), axis=2)
    first_norms = []
    confirmation_ids = [row["anchor_index"] for row in anchors_ if row["partition"] == "confirmation"]
    for ai, anchor_i in enumerate(confirmation_ids):
        flat = np.asarray(response[anchor_i], np.float32).reshape(DIM, -1)
        first_norms.append([float(np.linalg.norm(flat[a]) + np.linalg.norm(flat[b])) for a, b in lock["pairs"]])
    first_norms = np.asarray(first_norms)
    second_ratio = interaction_norm / np.maximum(first_norms, 1e-12)
    top_edges = []
    mean_abs = np.mean(np.abs(np.asarray(response, np.float32)), axis=0)
    outgoing = np.linalg.norm(mean_abs.reshape(DIM, -1), axis=1)
    for source in np.argsort(outgoing)[-64:][::-1]:
        target_flat = mean_abs[source].reshape(-1)
        target_ids = np.argsort(target_flat)[-16:][::-1]
        top_edges.append({"source_coordinate": int(source), "outgoing_norm": float(outgoing[source]), "targets": [{"role": ROLES[int(value // DIM)], "coordinate": int(value % DIM), "mean_abs_response": float(target_flat[value])} for value in target_ids]})
    core.save(OUT / "analysis/top_coordinate_edges.json", top_edges)
    report = {"phase": PHASE, "campaign": CAMPAIGN, "created_at_utc": now(), "status": "full_coordinate_local_transmission_adjudicated", "comparisons": comparisons, "overall_median_coordinate_cosine": overall_median, "stable_coordinate_union_count": stable_count, "gates": gates, "first_order_replication_passed": all(gates.values()), "generic_transport_diagnostic": {"identity_removed_median_cosine": identity_removed_median, "median_same_coordinate_energy_fraction": identity_energy_fraction, "relation_specificity_rows": relation_specificity, "median_matched_minus_wrong_relation_margin": specificity_margin, "relation_specific_transport_supported": specificity_margin > 0.05}, "second_order": {"pairs": lock["pairs"], "median_interaction_norm": float(np.median(interaction_norm)), "median_ratio_to_first_order_sum": float(np.median(second_ratio)), "max_ratio": float(np.max(second_ratio))}, "logit_response": {"median_abs": float(np.median(np.abs(logit_response))), "top_coordinates": np.argsort(np.mean(np.abs(logit_response), axis=0))[-32:][::-1].tolist()}, "top_edges": len(top_edges), "claim_boundary": protocol["claim_boundary"], "next_authorization": "C162 broad language program regardless of C161 gate"}
    core.save(OUT / "analysis/transmission.json", report)
    checks = {"comparisons": len(comparisons) == 8, "coordinates": all(row["stable_count"] <= DIM for row in comparisons), "specificity_controls": len(relation_specificity) == 8, "edges": len(top_edges) == 64, "second_order": list(interaction.shape) == [8, 8, 6, DIM], "finite": bool(np.isfinite(second_ratio).all())}
    core.save(OUT / "audit/internal_analysis_audit.json", {"checks": checks, "all_checks_passed": all(checks.values()), "scientific_first_order_passed": all(gates.values()), "authorization": report["next_authorization"]})
    print(json.dumps({key: value for key, value in report.items() if key not in ("comparisons",)}, indent=2))


def close():
    report = core.load(OUT / "analysis/transmission.json")
    checks = {"contract": core.load(OUT / "audit/internal_contract_audit.json")["all_checks_passed"], "first": core.load(OUT / "audit/internal_first_order_run_audit.json")["all_checks_passed"], "second": core.load(OUT / "audit/internal_second_order_run_audit.json")["all_checks_passed"], "analysis": core.load(OUT / "audit/internal_analysis_audit.json")["all_checks_passed"]}
    final = {"phase": PHASE, "campaign": CAMPAIGN, "status": "closed", "checks": checks, "all_checks_passed": all(checks.values()), "headline": {"median_coordinate_cosine": report["overall_median_coordinate_cosine"], "stable_coordinate_count": report["stable_coordinate_union_count"], "gates": report["gates"], "second_order": report["second_order"]}, "next_authorization": report["next_authorization"]}
    core.save(OUT / "analysis/final.json", final)
    core.save(OUT / "audit/internal_closure_audit.json", {"checks": checks, "all_checks_passed": all(checks.values()), "authorization": "independent_audit_then_C162"})
    print(json.dumps(final, indent=2))


def main():
    modes = {"contract": contract, "run_first": run_first_order, "select": select_pairs, "run_second": run_second_order, "analyze": analyze, "close": close}
    if len(sys.argv) != 2 or sys.argv[1] not in modes:
        raise SystemExit("contract|run_first|select|run_second|analyze|close")
    modes[sys.argv[1]]()


if __name__ == "__main__":
    main()
