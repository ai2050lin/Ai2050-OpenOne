#!/usr/bin/env python3
"""Phase1346: full-dimensional all-layer C049 interaction field."""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from statistics import median

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core
from phase1332_bf16_utils import load_bf16, quantization_audit, release_bf16

PHASE = 1346
CAMPAIGN = "C049"
CONTRACT = TESTS / "result/phase1344_c049_disentangled_relation_contract"
BEHAVIOR = TESTS / "result/phase1345_c049_disentangled_behavior"
OUT = TESTS / "result/phase1346_c049_full_interaction_field"
ROLES = ("target_span_mean", "tested_family_span_mean", "answer_boundary")


def parents():
    behavior = core.load(BEHAVIOR / "analysis/final.json")
    audit = core.load(BEHAVIOR / "audit/independent_final_audit.json")
    if behavior.get("authorization") != "run_phase1346_c049_full_interaction_field" or not audit.get(
        "all_checks_passed"
    ):
        raise RuntimeError("Phase1345 parent not authorized")
    return core.load(CONTRACT / "protocol/preregistration.json"), behavior[
        "relation_interaction_qualified_models"
    ]


def prepare():
    protocol, models = parents()
    if (OUT / "protocol/execution_manifest.json").exists():
        raise RuntimeError("Phase1346 manifest already exists")
    manifest = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "contract_sha256": protocol["contract_sha256"],
        "behavior_parent_sha256": core.sha(BEHAVIOR / "analysis/final.json"),
        "model_order": models,
        "precision": "bfloat16-no-quantization",
        "batch_size": 4,
        "roles": list(ROLES),
        "primary_role_index": 1,
        "object": protocol["field_gate"]["object"],
        "storage": protocol["field_gate"]["storage"],
        "gate": protocol["field_gate"],
        "numeric_sentinel_quartets": 8,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    core.save(OUT / "protocol/execution_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


def tensors(batch, width, pad, device):
    ids = torch.full((len(batch), width), int(pad), dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    lengths = []
    for index, row in enumerate(batch):
        value = torch.tensor(row["prompt_ids"], dtype=torch.long, device=device)
        ids[index, : len(value)] = value
        mask[index, : len(value)] = 1
        lengths.append(len(value))
    positions = mask.cumsum(-1) - 1
    positions.masked_fill_(mask == 0, 0)
    return ids, mask, positions, lengths


@torch.inference_mode()
def capture(model, device, batch, width, pad):
    ids, mask, positions, lengths = tensors(batch, width, pad, device)
    output = model(
        input_ids=ids,
        attention_mask=mask,
        position_ids=positions,
        use_cache=False,
        output_hidden_states=True,
        return_dict=True,
    )
    result = []
    for sample_index, row in enumerate(batch):
        per_depth = []
        spans = (row["target_span"], row["tested_family_span"], [lengths[sample_index] - 1])
        for hidden in output.hidden_states:
            state = hidden[sample_index].float()
            per_depth.append(torch.stack([state[span].mean(0) for span in spans]).cpu())
        result.append(torch.stack(per_depth))
    del ids, mask, positions, output
    return result


def interaction_from_states(states):
    return states[0] - states[1] - states[2] + states[3]


def normalized_rows(tensor):
    return F.normalize(tensor.float(), dim=-1, eps=1e-12)


def identity_metrics(vectors, metadata, depth, role, prototype_selector, query_selector, classes):
    prototypes = []
    for family_pair in classes:
        indexes = [
            index
            for index, row in enumerate(metadata)
            if row["family_pair"] == family_pair and prototype_selector(row)
        ]
        prototypes.append(vectors[indexes, depth, role].mean(0))
    prototypes = normalized_rows(torch.stack(prototypes))
    query_indexes = [index for index, row in enumerate(metadata) if query_selector(row)]
    queries = normalized_rows(vectors[query_indexes, depth, role])
    scores = queries @ prototypes.T
    correct = torch.tensor([classes.index(metadata[index]["family_pair"]) for index in query_indexes])
    predictions = scores.argmax(-1)
    correct_scores = scores[torch.arange(len(query_indexes)), correct]
    wrong_scores = scores.clone()
    wrong_scores[torch.arange(len(query_indexes)), correct] = -float("inf")
    gaps = correct_scores - wrong_scores.max(-1).values
    permuted = (correct + 1) % len(classes)
    return {
        "count": len(query_indexes),
        "top1": float((predictions == correct).float().mean()),
        "median_gap": float(gaps.median()),
        "permuted_label_top1": float((predictions == permuted).float().mean()),
    }


def metrics_for_bundle(vectors, relative_norms, metadata, classes):
    result = {}
    for depth in range(vectors.shape[1]):
        discovery = identity_metrics(
            vectors,
            metadata,
            depth,
            1,
            lambda row: row["partition"] == "discovery" and row["surface"] == "ordinary",
            lambda row: row["partition"] == "discovery" and row["surface"] in ("dictionary", "claim"),
            classes,
        )
        confirmation = identity_metrics(
            vectors,
            metadata,
            depth,
            1,
            lambda row: row["partition"] == "discovery",
            lambda row: row["partition"] == "confirmation",
            classes,
        )
        holdout = identity_metrics(
            vectors,
            metadata,
            depth,
            1,
            lambda row: row["partition"] == "discovery",
            lambda row: row["partition"] == "holdout",
            classes,
        )
        result[str(depth)] = {
            "discovery": discovery,
            "confirmation": confirmation,
            "holdout": holdout,
            "discovery_median_relative_norm": float(
                relative_norms[
                    [index for index, row in enumerate(metadata) if row["partition"] == "discovery"], depth, 1
                ].median()
            ),
        }
    return result


def run_model(model_name):
    protocol, models = parents()
    if model_name not in models:
        raise RuntimeError(f"{model_name} was not behavior-qualified")
    manifest = core.load(OUT / "protocol/execution_manifest.json")
    source = core.rows(CONTRACT / "material/frozen_factorial_cases.jsonl")
    compiled = core.rows(CONTRACT / f"compiled/{model_name}_factorial.jsonl")
    quartets = [(source[i : i + 4], compiled[i : i + 4]) for i in range(0, len(source), 4)]
    width = max(len(row["prompt_ids"]) for row in compiled)
    model = None
    try:
        model, tokenizer, device, placement = load_bf16(model_name)
        quant = quantization_audit(model)
        pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        vectors, relative_norms, metadata = None, None, []
        sentinel_saved = {}
        for quartet_index, (source_rows, compiled_rows) in enumerate(quartets):
            states = torch.stack(capture(model, device, compiled_rows, width, pad))
            interaction = interaction_from_states(states)
            scale = states.norm(dim=-1).mean(dim=0)
            relative = interaction.norm(dim=-1) / (scale + 1e-12)
            if vectors is None:
                vectors = torch.empty((len(quartets),) + tuple(interaction.shape), dtype=torch.float32)
                relative_norms = torch.empty((len(quartets),) + tuple(relative.shape), dtype=torch.float32)
            vectors[quartet_index] = interaction
            relative_norms[quartet_index] = relative
            metadata.append(
                {
                    key: source_rows[0][key]
                    for key in ("quartet_key", "partition", "family_pair", "pair_index", "pair_offset", "surface")
                }
            )
            if quartet_index < manifest["numeric_sentinel_quartets"]:
                sentinel_saved[quartet_index] = interaction.clone()
            del states, interaction, scale, relative

        numeric_relative = []
        for quartet_index in range(manifest["numeric_sentinel_quartets"]):
            source_rows, compiled_rows = quartets[quartet_index]
            order = (3, 2, 1, 0)
            permuted_rows = [compiled_rows[index] for index in order]
            captured = capture(model, device, permuted_rows, width, pad)
            by_case = {row["case_id"]: state for row, state in zip(permuted_rows, captured)}
            canonical = torch.stack([by_case[row["case_id"]] for row in compiled_rows])
            repeated = interaction_from_states(canonical)
            reference = sentinel_saved[quartet_index]
            relative = (reference - repeated).norm(dim=-1) / (reference.norm(dim=-1) + 1e-12)
            numeric_relative.extend(float(value) for value in relative.flatten())
        numeric_relative.sort()
        numeric = {
            "relative_l2_p95": numeric_relative[math.ceil(0.95 * len(numeric_relative)) - 1],
            "relative_l2_max": max(numeric_relative),
            "comparison_count": len(numeric_relative),
        }

        classes = ["__".join(pair) for pair in combinations(protocol["material"]["families"], 2)]
        layer_metrics = metrics_for_bundle(vectors, relative_norms, metadata, classes)
        gate = manifest["gate"]
        candidates = []
        for depth in range(1, vectors.shape[1]):
            value = layer_metrics[str(depth)]
            if (
                value["discovery"]["top1"] >= gate["discovery_family_pair_top1_min"]
                and value["discovery"]["median_gap"] >= gate["discovery_median_gap_min"]
                and value["discovery_median_relative_norm"] >= gate["discovery_relative_norm_min"]
            ):
                candidates.append(depth)
        selected_layer = candidates[0] if candidates else None
        numeric_qualified = (
            numeric["relative_l2_p95"] <= gate["numeric_relative_l2_p95_max"]
            and numeric["relative_l2_max"] <= gate["numeric_relative_l2_max"]
        )
        layer0_max = float(relative_norms[:, 0, 1].max())
        layer0_qualified = layer0_max <= gate["layer0_relative_norm_max"]
        selected_gates = None
        if selected_layer is not None:
            selected = layer_metrics[str(selected_layer)]
            selected_gates = {
                "confirmation_top1": selected["confirmation"]["top1"]
                >= gate["confirmation_family_pair_top1_min"],
                "holdout_top1": selected["holdout"]["top1"] >= gate["holdout_family_pair_top1_min"],
                "confirmation_gap": selected["confirmation"]["median_gap"] >= gate["transfer_median_gap_min"],
                "holdout_gap": selected["holdout"]["median_gap"] >= gate["transfer_median_gap_min"],
                "confirmation_permuted": selected["confirmation"]["permuted_label_top1"]
                <= gate["permuted_label_top1_max"],
                "holdout_permuted": selected["holdout"]["permuted_label_top1"]
                <= gate["permuted_label_top1_max"],
            }
        qualified = numeric_qualified and layer0_qualified and selected_layer is not None and all(
            selected_gates.values() if selected_gates else []
        )
        OUT.joinpath("raw").mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": model_name,
                "roles": list(ROLES),
                "classes": classes,
                "metadata": metadata,
                "interaction_vectors": vectors,
                "relative_norms": relative_norms,
            },
            OUT / f"raw/{model_name}_full_interaction_field.pt",
        )
        summary = {
            "model": model_name,
            "shape": list(vectors.shape),
            "numeric": numeric,
            "numeric_qualified": numeric_qualified,
            "layer0_max_relative_norm": layer0_max,
            "layer0_qualified": layer0_qualified,
            "layer_metrics": layer_metrics,
            "discovery_passing_layers": candidates,
            "selected_layer": selected_layer,
            "selected_transfer_gates": selected_gates,
            "qualified": qualified,
            "claim_boundary": "full-dimensional model-specific family-pair interaction identity; no semantic ontology or component localization",
            "runtime": {
                "placement": placement,
                "quantization_audit": quant,
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        }
        core.save(OUT / f"analysis/{model_name}_summary.json", summary)
        compact = dict(summary)
        compact.pop("layer_metrics")
        print(json.dumps(compact, indent=2))
    finally:
        if model is not None:
            release_bf16(model)


def finalize():
    protocol, models = parents()
    summaries = {model: core.load(OUT / f"analysis/{model}_summary.json") for model in models}
    qualified = [model for model in models if summaries[model]["qualified"]]
    authorization = "run_phase1347_c049_same_label_causal_swaps" if qualified else "close_c049_descriptive_field"
    final = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "evaluated_models": models,
        "field_qualified_models": qualified,
        "cross_model_field_repetition": len(qualified) >= protocol["field_gate"]["cross_model_minimum"],
        "all_gates_passed": bool(qualified),
        "authorization": authorization,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    core.save(OUT / "analysis/final.json", final)
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    _, authorized_models = parents()
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", action="store_true")
    group.add_argument("--model", choices=authorized_models)
    group.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    prepare() if args.prepare else run_model(args.model) if args.model else finalize()
