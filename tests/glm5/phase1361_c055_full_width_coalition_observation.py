#!/usr/bin/env python3
"""Phase1361: compare frozen full-width hidden-state role coalitions without fitting."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core

PHASE, CAMPAIGN = 1361, "C055"
CONTRACT = TESTS / "result/phase1360_c055_hidden_state_coalition_contract"
FIELD = TESTS / "result/phase1355_c053_full_width_route_fields"
OUT = TESTS / "result/phase1361_c055_full_width_coalition_observation"
TENSOR = FIELD / "raw/qwen3_quartet_fields.pt"
ROLE_INDEX = {"target": 0, "family": 1, "boundary": 2}


def parent() -> dict:
    final = core.load(CONTRACT / "analysis/final.json")
    audit = core.load(CONTRACT / "audit/independent_final_audit.json")
    if final.get("authorization") != "run_phase1361_c055_hidden_state_observation" or not audit.get("all_checks_passed"):
        raise RuntimeError("Phase1360 did not authorize observation")
    return core.load(CONTRACT / "protocol/preregistration.json")


def coalition(tensor: torch.Tensor, depth: int, roles: list[str]) -> torch.Tensor:
    indexes = [ROLE_INDEX[role] for role in roles]
    value = tensor[:, depth, indexes, :].reshape(tensor.shape[0], -1).float()
    return F.normalize(value, dim=-1, eps=1e-12)


def identity_metrics(vectors: torch.Tensor, metadata: list[dict]) -> dict:
    classes = sorted({row["family_pair"] for row in metadata if row["partition"] == "prototype_discovery"})
    prototypes = []
    for family_pair in classes:
        indexes = [i for i, row in enumerate(metadata)
                   if row["partition"] == "prototype_discovery" and row["family_pair"] == family_pair]
        prototypes.append(vectors[indexes].mean(0))
    prototypes = F.normalize(torch.stack(prototypes), dim=-1, eps=1e-12)
    indexes = [i for i, row in enumerate(metadata) if row["partition"] == "clock_selection"]
    queries = vectors[indexes]
    scores = queries @ prototypes.T
    correct = torch.tensor([classes.index(metadata[i]["family_pair"]) for i in indexes])
    prediction = scores.argmax(-1)
    good = scores[torch.arange(len(indexes)), correct]
    wrong = scores.clone()
    wrong[torch.arange(len(indexes)), correct] = -float("inf")
    gaps = good - wrong.max(-1).values
    surface = {}
    for name in sorted({metadata[i]["surface"] for i in indexes}):
        mask = torch.tensor([metadata[i]["surface"] == name for i in indexes])
        surface[name] = float((prediction[mask] == correct[mask]).float().mean())
    return {"count": len(indexes), "top1": float((prediction == correct).float().mean()),
            "surface_top1": surface, "median_gap": float(gaps.median())}


def shared_metrics(active: torch.Tensor, status: torch.Tensor,
                   active_meta: list[dict], status_meta: list[dict]) -> dict:
    discovery_a = [i for i, row in enumerate(active_meta) if row["partition"] == "prototype_discovery"]
    discovery_s = [i for i, row in enumerate(status_meta) if row["partition"] == "prototype_discovery"]
    ga = F.normalize(active[discovery_a].mean(0), dim=0, eps=1e-12)
    gs = F.normalize(status[discovery_s].mean(0), dim=0, eps=1e-12)
    result = {"active_status_direction_cosine": float(ga @ gs)}
    for partition in ("clock_selection", "confirmation", "lockbox"):
        ia = [i for i, row in enumerate(active_meta) if row["partition"] == partition]
        is_ = [i for i, row in enumerate(status_meta) if row["partition"] == partition]
        qa, qs = active[ia], status[is_]
        active_cos = qa @ ga
        status_cos = qs @ ga
        gap = (qa @ ga) - (qa @ gs)
        result[partition] = {
            "active_median_cosine": float(active_cos.median()),
            "status_median_cosine_to_active": float(status_cos.median()),
            "active_over_status_median_gap": float(gap.median()),
            "active_over_status_win_fraction": float((gap > 0).float().mean()),
        }
    return result


def persistent_start(passing: list[int], length: int) -> int | None:
    found = set(passing)
    for depth in passing:
        if all(depth + offset in found for offset in range(length)):
            return depth
    return None


def main() -> None:
    protocol = parent()
    bundle = torch.load(TENSOR, map_location="cpu", weights_only=False)
    active = bundle["active_interactions"].float()
    status = bundle["status_interactions"].float()
    active_meta, status_meta = bundle["active_metadata"], bundle["status_metadata"]
    gate = protocol["observation"]
    coalitions = protocol["coalitions"]
    layer_metrics = {}
    passing = {name: [] for name in coalitions}
    for depth in gate["depths"]:
        layer_metrics[str(depth)] = {}
        vectors = {}
        for name, roles in coalitions.items():
            av = coalition(active, depth, roles)
            sv = coalition(status, depth, roles)
            vectors[name] = av
            layer_metrics[str(depth)][name] = {
                "identity": identity_metrics(av, active_meta),
                "shared": shared_metrics(av, sv, active_meta, status_meta),
            }
        best_singleton_top1 = max(layer_metrics[str(depth)][name]["identity"]["top1"]
                                  for name in ("target", "family", "boundary"))
        for name in ("target_family", "target_boundary", "family_boundary", "all_roles"):
            value = layer_metrics[str(depth)][name]
            identity, shared = value["identity"], value["shared"]
            value["identity_gain_over_best_singleton"] = identity["top1"] - best_singleton_top1
            checks = {
                "identity": identity["top1"] >= gate["multi_identity_top1_min"],
                "surface": min(identity["surface_top1"].values()) >= gate["multi_surface_top1_min"],
                "synergy": value["identity_gain_over_best_singleton"] >= gate["identity_gain_over_best_singleton_min"],
                "selection_cosine": shared["clock_selection"]["active_median_cosine"] >= gate["selection_active_cosine_min"],
                "selection_gap": shared["clock_selection"]["active_over_status_median_gap"] >= gate["active_over_status_gap_min"],
                "selection_win": shared["clock_selection"]["active_over_status_win_fraction"] >= gate["active_over_status_win_min"],
                "status_direction": shared["active_status_direction_cosine"] <= gate["status_direction_cosine_max"],
                "confirmation": shared["confirmation"]["active_median_cosine"] >= gate["held_active_cosine_min"]
                                and shared["confirmation"]["active_over_status_median_gap"] >= gate["held_gap_min"]
                                and shared["confirmation"]["active_over_status_win_fraction"] >= gate["held_win_min"],
                "lockbox": shared["lockbox"]["active_median_cosine"] >= gate["held_active_cosine_min"]
                           and shared["lockbox"]["active_over_status_median_gap"] >= gate["held_gap_min"]
                           and shared["lockbox"]["active_over_status_win_fraction"] >= gate["held_win_min"],
            }
            value["checks"] = checks
            value["qualified_at_layer"] = all(checks.values())
            if value["qualified_at_layer"]:
                passing[name].append(depth)

    persistent = {name: persistent_start(depths, gate["persistence_layers"])
                  for name, depths in passing.items()}
    candidates = [
        (depth, len(coalitions[name]), name)
        for name, depth in persistent.items() if depth is not None and name not in ("target", "family", "boundary")
    ]
    selected_name = selected_layer = None
    if candidates:
        selected_layer, _size, selected_name = sorted(candidates)[0]
    fallback = gate["fallback_if_none"]
    causal_name = selected_name if selected_name is not None else fallback["coalition"]
    causal_layer = selected_layer if selected_layer is not None else fallback["layer"]
    result = {
        "phase": PHASE, "campaign": CAMPAIGN,
        "tensor_shapes": {"active": list(active.shape), "status": list(status.shape)},
        "coalition_passing_layers": passing, "coalition_persistent_start": persistent,
        "descriptive_multi_coalition_qualified": selected_name is not None,
        "selected_descriptive_coalition": selected_name, "selected_descriptive_layer": selected_layer,
        "causal_reference_coalition": causal_name, "causal_layer": causal_layer,
        "used_fallback": selected_name is None, "layer_metrics": layer_metrics,
        "claim_boundary": "ordered full-width role concatenation; no fitted/projection model and no causal claim",
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    core.save(OUT / "analysis/coalition_observation.json", result)
    core.save(OUT / "analysis/final.json", {
        "phase": PHASE, "campaign": CAMPAIGN,
        "descriptive_multi_coalition_qualified": result["descriptive_multi_coalition_qualified"],
        "causal_layer": causal_layer, "authorization": "run_phase1362_c055_coalition_camera",
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    print(json.dumps({key: value for key, value in result.items() if key != "layer_metrics"}, indent=2))


if __name__ == "__main__":
    main()
