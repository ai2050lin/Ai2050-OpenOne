#!/usr/bin/env python3
"""Phase1355: frozen full-width fields for behavior-qualified C053 routes."""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core
from phase1332_bf16_utils import load_bf16, quantization_audit, release_bf16

PHASE, CAMPAIGN = 1355, "C053"
CONTRACT = TESTS / "result/phase1353_c053_route_portfolio_contract"
BEHAVIOR = TESTS / "result/phase1354_c053_behavior_route_competition"
OUT = TESTS / "result/phase1355_c053_full_width_route_fields"
MODEL = "qwen3"
QUARTET_ROLES = ("target_span_mean", "tested_family_span_mean", "answer_boundary")
CHOICE_ROLES = ("target_span_mean", "candidate_pair_mean", "answer_boundary")


def parents():
    final = core.load(BEHAVIOR / "analysis/final.json")
    audit = core.load(BEHAVIOR / "audit/independent_final_audit.json")
    if final.get("authorization") != "run_phase1355_c053_fields" or not audit.get("all_checks_passed"):
        raise RuntimeError("Phase1354 did not authorize fields")
    return core.load(CONTRACT / "protocol/preregistration.json"), final


def prepare():
    protocol, behavior = parents()
    path = OUT / "protocol/execution_manifest.json"
    if path.exists():
        raise RuntimeError("Phase1355 manifest already exists")
    manifest = {
        "phase": PHASE, "campaign": CAMPAIGN, "contract_sha256": protocol["contract_sha256"],
        "behavior_parent_sha256": core.sha(BEHAVIOR / "analysis/final.json"), "model": MODEL,
        "precision": "bfloat16-no-quantization", "batch_size": 4,
        "authorized_fields": behavior["authorized_fields"], "gate": protocol["field_gate"],
        "quartet_roles": list(QUARTET_ROLES), "choice_roles": list(CHOICE_ROLES),
        "numeric_sentinel_groups": 8, "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    core.save(path, manifest)
    print(json.dumps(manifest, indent=2))


def tensors(batch, width, pad, device):
    ids = torch.full((len(batch), width), int(pad), dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    lengths = []
    for i, row in enumerate(batch):
        value = torch.tensor(row["prompt_ids"], dtype=torch.long, device=device)
        ids[i, :len(value)] = value
        mask[i, :len(value)] = 1
        lengths.append(len(value))
    positions = mask.cumsum(-1) - 1
    positions.masked_fill_(mask == 0, 0)
    return ids, mask, positions, lengths


@torch.inference_mode()
def capture(model, device, batch, width, pad, route):
    ids, mask, positions, lengths = tensors(batch, width, pad, device)
    output = model(input_ids=ids, attention_mask=mask, position_ids=positions,
                   use_cache=False, output_hidden_states=True, return_dict=True)
    result = []
    for sample_index, row in enumerate(batch):
        if route == "quartet":
            spans = (row["target_span"], row["tested_family_span"], [lengths[sample_index] - 1])
        else:
            pair_span = sorted(set(row["candidate_a_span"] + row["candidate_b_span"]))
            spans = (row["target_span"], pair_span, [lengths[sample_index] - 1])
        per_depth = []
        for hidden in output.hidden_states:
            state = hidden[sample_index].float()
            per_depth.append(torch.stack([state[s].mean(0) for s in spans]).cpu())
        result.append(torch.stack(per_depth))
    del ids, mask, positions, output
    return result


def interaction(states):
    return states[0] - states[1] - states[2] + states[3]


def norm_rows(value):
    return F.normalize(value.float(), dim=-1, eps=1e-12)


def identity_metrics(vectors, metadata, depth, role):
    classes = sorted({x["family_pair"] for x in metadata if x["partition"] == "prototype_discovery"})
    prototypes = torch.stack([
        vectors[[i for i, x in enumerate(metadata)
                 if x["partition"] == "prototype_discovery" and x["family_pair"] == cls], depth, role].mean(0)
        for cls in classes
    ])
    prototypes = norm_rows(prototypes)
    indexes = [i for i, x in enumerate(metadata) if x["partition"] == "clock_selection"]
    queries = norm_rows(vectors[indexes, depth, role])
    scores = queries @ prototypes.T
    correct = torch.tensor([classes.index(metadata[i]["family_pair"]) for i in indexes])
    predictions = scores.argmax(-1)
    good = scores[torch.arange(len(indexes)), correct]
    wrong = scores.clone()
    wrong[torch.arange(len(indexes)), correct] = -float("inf")
    gaps = good - wrong.max(-1).values
    surface = {}
    for name in sorted({metadata[i]["surface"] for i in indexes}):
        mask = torch.tensor([metadata[i]["surface"] == name for i in indexes])
        surface[name] = float((predictions[mask] == correct[mask]).float().mean())
    return {"count": len(indexes), "top1": float((predictions == correct).float().mean()),
            "surface_top1": surface, "median_gap": float(gaps.median())}


def shared_metrics(active, status, active_meta, status_meta, depth, role):
    discovery_a = [i for i, x in enumerate(active_meta) if x["partition"] == "prototype_discovery"]
    discovery_s = [i for i, x in enumerate(status_meta) if x["partition"] == "prototype_discovery"]
    ga = norm_rows(active[discovery_a, depth, role].mean(0, keepdim=True))[0]
    gs = norm_rows(status[discovery_s, depth, role].mean(0, keepdim=True))[0]
    result = {"active_status_direction_cosine": float(ga @ gs)}
    for partition in ("clock_selection", "confirmation", "lockbox"):
        ia = [i for i, x in enumerate(active_meta) if x["partition"] == partition]
        is_ = [i for i, x in enumerate(status_meta) if x["partition"] == partition]
        qa = norm_rows(active[ia, depth, role])
        qs = norm_rows(status[is_, depth, role])
        active_cos = qa @ ga
        status_to_active = qs @ ga
        active_gap = (qa @ ga) - (qa @ gs)
        result[partition] = {
            "active_median_cosine": float(active_cos.median()),
            "status_median_cosine_to_active": float(status_to_active.median()),
            "active_over_status_median_gap": float(active_gap.median()),
            "active_over_status_win_fraction": float((active_gap > 0).float().mean()),
        }
    return result


def persistent_start(passing, length):
    for depth in passing:
        if all(depth + offset in passing for offset in range(length)):
            return depth
    return None


def run_quartet_field(model, device, tok, manifest):
    active_source = core.rows(CONTRACT / "material/b1_binary_cases.jsonl")
    active_compiled = core.rows(CONTRACT / "compiled/qwen3_B1_binary.jsonl")
    status_source = core.rows(CONTRACT / "material/status_null_cases.jsonl")
    status_compiled = core.rows(CONTRACT / "compiled/qwen3_N_status.jsonl")
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    bundles = {}
    numeric = []
    for label, source, compiled in (("active", active_source, active_compiled),
                                    ("status", status_source, status_compiled)):
        width = max(len(x["prompt_ids"]) for x in compiled)
        groups = [(source[i:i + 4], compiled[i:i + 4]) for i in range(0, len(source), 4)]
        vectors = relative = None
        metadata = []
        sentinels = {}
        for group_index, (source_rows, compiled_rows) in enumerate(groups):
            states = torch.stack(capture(model, device, compiled_rows, width, pad, "quartet"))
            value = interaction(states)
            scale = states.norm(dim=-1).mean(0)
            rel = value.norm(dim=-1) / (scale + 1e-12)
            if vectors is None:
                vectors = torch.empty((len(groups),) + tuple(value.shape), dtype=torch.float32)
                relative = torch.empty((len(groups),) + tuple(rel.shape), dtype=torch.float32)
            vectors[group_index], relative[group_index] = value, rel
            metadata.append({key: source_rows[0][key] for key in
                             ("quartet_key", "partition", "family_scope", "family_pair", "surface")})
            if group_index < manifest["numeric_sentinel_groups"]:
                sentinels[group_index] = value.clone()
        for group_index in range(manifest["numeric_sentinel_groups"]):
            source_rows, compiled_rows = groups[group_index]
            order = (3, 2, 1, 0)
            captured = capture(model, device, [compiled_rows[i] for i in order], width, pad, "quartet")
            by_id = {compiled_rows[i]["case_id"]: state for i, state in zip(order, captured)}
            canonical = torch.stack([by_id[x["case_id"]] for x in compiled_rows])
            repeated = interaction(canonical)
            reference = sentinels[group_index]
            err = (reference - repeated).norm(dim=-1) / (reference.norm(dim=-1) + 1e-12)
            numeric.extend(float(x) for x in err.flatten())
        bundles[label] = (vectors, relative, metadata)
    active, active_rel, active_meta = bundles["active"]
    status, status_rel, status_meta = bundles["status"]
    gate = manifest["gate"]
    layer_metrics = {}
    family_pass, shared_pass = [], []
    for depth in range(active.shape[1]):
        identity = identity_metrics(active, active_meta, depth, 1)
        shared = shared_metrics(active, status, active_meta, status_meta, depth, 1)
        layer_metrics[str(depth)] = {"family_pair": identity, "shared_relation": shared,
                                     "median_relative_norm": float(active_rel[:, depth, 1].median())}
        if depth > 0 and identity["top1"] >= gate["family_pair_selection_top1_min"] \
                and min(identity["surface_top1"].values()) >= gate["family_pair_surface_min"] \
                and identity["median_gap"] >= gate["family_pair_gap_min"]:
            family_pass.append(depth)
        s = shared["clock_selection"]
        if depth > 0 and s["active_median_cosine"] >= gate["shared_selection_cosine_min"] \
                and s["active_over_status_median_gap"] >= gate["active_over_status_gap_min"] \
                and s["active_over_status_win_fraction"] >= gate["active_over_status_win_min"] \
                and shared["active_status_direction_cosine"] <= gate["status_direction_cosine_max"]:
            shared_pass.append(depth)
    family_start = persistent_start(family_pass, gate["persistence_layers"])
    shared_start = persistent_start(shared_pass, gate["persistence_layers"])
    selected = shared_start if shared_start is not None else family_start
    transfer = None
    shared_qualified = False
    if shared_start is not None:
        value = layer_metrics[str(shared_start)]["shared_relation"]
        transfer = {}
        for partition in ("confirmation", "lockbox"):
            part = value[partition]
            transfer[partition] = {
                "cosine": part["active_median_cosine"] >= gate["shared_transfer_cosine_min"],
                "gap": part["active_over_status_median_gap"] >= gate["active_over_status_gap_min"],
                "win": part["active_over_status_win_fraction"] >= gate["active_over_status_win_min"],
            }
        shared_qualified = all(v for x in transfer.values() for v in x.values())
    numeric_max = max(numeric)
    layer0 = max(float(active_rel[:, 0].max()), float(status_rel[:, 0].max()))
    numeric_ok = numeric_max <= gate["numeric_relative_l2_max"]
    layer0_ok = layer0 <= gate["layer0_relative_norm_max"]
    shared_qualified = shared_qualified and numeric_ok and layer0_ok
    OUT.joinpath("raw").mkdir(parents=True, exist_ok=True)
    torch.save({"roles": list(QUARTET_ROLES), "active_metadata": active_meta,
                "status_metadata": status_meta, "active_interactions": active,
                "status_interactions": status, "active_relative_norms": active_rel,
                "status_relative_norms": status_rel}, OUT / "raw/qwen3_quartet_fields.pt")
    summary = {
        "shape_active": list(active.shape), "shape_status": list(status.shape),
        "numeric_relative_l2_max": numeric_max, "numeric_qualified": numeric_ok,
        "layer0_max_relative_norm": layer0, "layer0_qualified": layer0_ok,
        "family_pair_passing_layers": family_pass, "family_pair_persistent_start": family_start,
        "shared_relation_passing_layers": shared_pass, "shared_relation_selected_layer": shared_start,
        "selected_layer_for_causal": selected, "transfer_checks": transfer,
        "family_pair_candidate": family_start is not None and numeric_ok and layer0_ok,
        "shared_relation_qualified": shared_qualified, "layer_metrics": layer_metrics,
        "claim_boundary": "descriptive full-dimensional response objects; no component or parameter identity",
    }
    core.save(OUT / "analysis/quartet_field_summary.json", summary)
    return summary


def run_choice_field(model, device, tok, manifest):
    source = core.rows(CONTRACT / "material/b3_choice_cases.jsonl")
    compiled = core.rows(CONTRACT / "compiled/qwen3_B3_choice.jsonl")
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    width = max(len(x["prompt_ids"]) for x in compiled)
    groups = [(source[i:i + 2], compiled[i:i + 2]) for i in range(0, len(source), 2)]
    averaged = None
    metadata, order_cosines = [], []
    for index, (source_rows, compiled_rows) in enumerate(groups):
        states = torch.stack(capture(model, device, compiled_rows, width, pad, "choice"))
        avg = states.mean(0)
        cosine = F.cosine_similarity(states[0], states[1], dim=-1)
        if averaged is None:
            averaged = torch.empty((len(groups),) + tuple(avg.shape), dtype=torch.float32)
            order_cosines = torch.empty((len(groups),) + tuple(cosine.shape), dtype=torch.float32)
        averaged[index], order_cosines[index] = avg, cosine
        metadata.append({key: source_rows[0][key] for key in
                         ("choice_group", "partition", "family_scope", "family_pair", "surface",
                          "target_side", "target", "target_family")})
    layer_metrics = {}
    gate = manifest["gate"]
    passing = []
    for depth in range(averaged.shape[1]):
        part = {}
        for partition in ("clock_selection", "confirmation", "lockbox"):
            indexes = [i for i, x in enumerate(metadata) if x["partition"] == partition]
            median_cos = float(order_cosines[indexes, depth, 2].median())
            wins = []
            by_base = defaultdict(dict)
            for i in indexes:
                base = metadata[i]["choice_group"].rsplit(":", 1)[0]
                by_base[base][metadata[i]["target_side"]] = i
            for pair in by_base.values():
                if set(pair) != {"a", "b"}:
                    continue
                for side, other in (("a", "b"), ("b", "a")):
                    i, j = pair[side], pair[other]
                    own = float(order_cosines[i, depth, 2])
                    cross = float(F.cosine_similarity(averaged[i, depth, 2], averaged[j, depth, 2], dim=0))
                    wins.append(own > cross)
            part[partition] = {"median_order_cosine": median_cos,
                               "retrieval_win_fraction": sum(wins) / len(wins)}
        layer_metrics[str(depth)] = part
        value = part["clock_selection"]
        if depth > 0 and value["median_order_cosine"] >= gate["choice_order_cosine_min"] \
                and value["retrieval_win_fraction"] >= gate["choice_retrieval_win_min"]:
            passing.append(depth)
    selected = persistent_start(passing, gate["persistence_layers"])
    transfer = None
    qualified = False
    if selected is not None:
        value = layer_metrics[str(selected)]
        transfer = {partition: value[partition]["median_order_cosine"] >= gate["choice_order_cosine_min"]
                    and value[partition]["retrieval_win_fraction"] >= gate["choice_retrieval_win_min"]
                    for partition in ("confirmation", "lockbox")}
        qualified = all(transfer.values())
    OUT.joinpath("raw").mkdir(parents=True, exist_ok=True)
    torch.save({"roles": list(CHOICE_ROLES), "metadata": metadata, "averaged_states": averaged,
                "order_cosines": order_cosines}, OUT / "raw/qwen3_choice_fields.pt")
    summary = {"shape": list(averaged.shape), "passing_layers": passing,
               "selected_layer": selected, "transfer_checks": transfer, "qualified": qualified,
               "layer_metrics": layer_metrics,
               "claim_boundary": "choice-order invariance and target retrieval only; not an abstract relation operator"}
    core.save(OUT / "analysis/choice_field_summary.json", summary)
    return summary


def run():
    protocol, behavior = parents()
    manifest = core.load(OUT / "protocol/execution_manifest.json")
    model = None
    try:
        model, tok, device, placement = load_bf16(MODEL)
        quant = quantization_audit(model)
        summaries = {}
        if "quartet_interaction_field" in manifest["authorized_fields"]:
            summaries["quartet_interaction_field"] = run_quartet_field(model, device, tok, manifest)
        if "choice_order_invariance_field" in manifest["authorized_fields"]:
            summaries["choice_order_invariance_field"] = run_choice_field(model, device, tok, manifest)
        runtime = {"placement": placement, "quantization": quant,
                   "finished_at_utc": datetime.now(timezone.utc).isoformat()}
        core.save(OUT / "analysis/runtime.json", runtime)
        compact = {key: {k: v for k, v in value.items() if k != "layer_metrics"}
                   for key, value in summaries.items()}
        print(json.dumps(compact, indent=2))
    finally:
        if model is not None:
            release_bf16(model)


def finalize():
    _, behavior = parents()
    fields = behavior["authorized_fields"]
    quartet = core.load(OUT / "analysis/quartet_field_summary.json") if "quartet_interaction_field" in fields else None
    choice = core.load(OUT / "analysis/choice_field_summary.json") if "choice_order_invariance_field" in fields else None
    shared = bool(quartet and quartet["shared_relation_qualified"])
    final = {
        "phase": PHASE, "campaign": CAMPAIGN, "evaluated_fields": fields,
        "family_pair_candidate": bool(quartet and quartet["family_pair_candidate"]),
        "shared_relation_qualified": shared,
        "choice_field_qualified": bool(choice and choice["qualified"]),
        "authorization": "run_phase1356_c053_typed_causal" if shared else "close_c053_after_fields",
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    core.save(OUT / "analysis/final.json", final)
    print(json.dumps(final, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "run", "finalize"))
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "run":
        run()
    else:
        finalize()


if __name__ == "__main__":
    main()
