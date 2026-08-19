#!/usr/bin/env python3
"""Phase1366: observe every frozen C056 Qwen Hidden-State response path."""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
T = ROOT / "tests/glm5"
sys.path.insert(0, str(T))
import phase1331_relational_measurement_core as core
from phase1332_bf16_utils import load_bf16, quantization_audit, release_bf16

PHASE, CAMPAIGN = 1366, "C056"
CONTRACT = T / "result/phase1364_c056_hidden_path_contract"
PARENT = T / "result/phase1365_c056_planted_hidden_path_camera"
OUT = T / "result/phase1366_c056_qwen_hidden_response_paths"
ROLES = ("target", "family", "query", "boundary")
ROLE_INDEX = {role: index for index, role in enumerate(ROLES)}


def make_batch(rows: list[dict], pad: int, device: torch.device):
    width = max(len(row["prompt_ids"]) for row in rows)
    ids = torch.full((len(rows), width), pad, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    offsets = []
    for index, row in enumerate(rows):
        values = torch.tensor(row["prompt_ids"], dtype=torch.long, device=device)
        offset = width - len(values)
        offsets.append(offset)
        ids[index, offset:] = values
        mask[index, offset:] = 1
    positions = mask.cumsum(-1) - 1
    positions.masked_fill_(mask == 0, 0)
    return ids, mask, positions, offsets


@torch.inference_mode()
def capture(model, rows: list[dict], pad: int, device: torch.device) -> torch.Tensor:
    ids, mask, positions, offsets = make_batch(rows, pad, device)
    output = model(input_ids=ids, attention_mask=mask, position_ids=positions,
                   use_cache=False, output_hidden_states=True, return_dict=True)
    result = []
    for sample, row in enumerate(rows):
        depths = []
        for hidden in output.hidden_states:
            state = hidden[sample].float()
            role_values = []
            for role in ROLES:
                points = [offsets[sample] + value for value in row["role_positions"][role]]
                role_values.append(state[points].mean(0).cpu())
            depths.append(torch.stack(role_values))
        result.append(torch.stack(depths))
    del ids, mask, positions, output
    return torch.stack(result)


def path_events(path: dict) -> list[tuple[int, str]]:
    return [(path["source"]["layer"], path["source"]["role"])] + [
        (point["layer"], point["role"]) for point in path["checkpoints"]
    ]


def vectors(delta: torch.Tensor, path: dict) -> torch.Tensor:
    values = [delta[:, layer, ROLE_INDEX[role], :] for layer, role in path_events(path)]
    return F.normalize(torch.cat(values, dim=-1), dim=-1, eps=1e-12)


def event_vectors(delta: torch.Tensor, event: tuple[int, str]) -> torch.Tensor:
    return F.normalize(delta[:, event[0], ROLE_INDEX[event[1]], :], dim=-1, eps=1e-12)


def identity(v: torch.Tensor, metadata: list[dict]) -> dict:
    discovery_classes = sorted({row["family_pair"] for row in metadata if row["partition"] == "prototype_discovery"})
    clock_classes = {row["family_pair"] for row in metadata if row["partition"] == "clock_selection"}
    classes = [name for name in discovery_classes if name in clock_classes]
    prototypes = []
    for name in classes:
        indexes = [i for i, row in enumerate(metadata)
                   if row["partition"] == "prototype_discovery" and row["family_pair"] == name]
        prototypes.append(v[indexes].mean(0))
    prototypes = F.normalize(torch.stack(prototypes), dim=-1, eps=1e-12)
    indexes = [i for i, row in enumerate(metadata)
               if row["partition"] == "clock_selection" and row["family_pair"] in classes]
    scores = v[indexes] @ prototypes.T
    correct = torch.tensor([classes.index(metadata[i]["family_pair"]) for i in indexes])
    prediction = scores.argmax(-1)
    surface = {}
    for name in sorted({metadata[i]["surface"] for i in indexes}):
        mask = torch.tensor([metadata[i]["surface"] == name for i in indexes])
        surface[name] = float((prediction[mask] == correct[mask]).float().mean())
    good = scores[torch.arange(len(indexes)), correct]
    wrong = scores.clone()
    wrong[torch.arange(len(indexes)), correct] = -float("inf")
    return {
        "classes": classes, "count": len(indexes),
        "top1": float((prediction == correct).float().mean()),
        "surface_top1": surface,
        "median_gap": float((good - wrong.max(-1).values).median()),
    }


def main() -> None:
    parent = core.load(PARENT / "analysis/final.json")
    audit = core.load(PARENT / "audit/independent_final_audit.json")
    protocol = core.load(CONTRACT / "protocol/preregistration.json")
    if parent.get("authorization") != "run_phase1366_c056_qwen_path_observation" or not audit.get("all_checks_passed"):
        raise RuntimeError("Phase1365 did not authorize observation")
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1366 already exists")
    cases = core.rows(CONTRACT / "material/path_cases.jsonl")
    compiled = {row["case_id"]: row for row in core.rows(CONTRACT / "compiled/extended_rows.jsonl")}
    model = None
    try:
        model, tok, device, placement = load_bf16("qwen3")
        quant = quantization_audit(model)
        pad = int(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
        delta = torch.empty((len(cases), 37, len(ROLES), 2560), dtype=torch.float32)
        relative = torch.empty((len(cases), 37, len(ROLES)), dtype=torch.float32)
        sentinels = {}
        for start in range(0, len(cases), 4):
            group = cases[start:start + 4]
            rows = []
            for case in group:
                rows.extend([compiled[case["clean_true"]], compiled[case["corrupt_false"]]])
            states = capture(model, rows, pad, device)
            for local in range(len(group)):
                value = states[2 * local] - states[2 * local + 1]
                scale = 0.5 * (states[2 * local].norm(dim=-1) + states[2 * local + 1].norm(dim=-1))
                index = start + local
                delta[index] = value
                relative[index] = value.norm(dim=-1) / (scale + 1e-12)
                if index < 4:
                    sentinels[index] = value.clone()
            if (start // 4 + 1) % 6 == 0:
                print(json.dumps({"pairs": min(start + 4, len(cases)), "total": len(cases)}), flush=True)
        numeric_errors = []
        for index in range(4):
            case = cases[index]
            repeated = capture(model, [compiled[case["corrupt_false"]], compiled[case["clean_true"]]], pad, device)
            value = repeated[1] - repeated[0]
            reference = sentinels[index]
            error = (value - reference).norm(dim=-1) / (reference.norm(dim=-1) + 1e-12)
            numeric_errors.extend(float(x) for x in error.flatten())
        metadata = [{key: row[key] for key in
                     ("pair_id", "partition", "surface", "family_pair", "direction", "target")}
                    for row in cases]
        path_metrics = {}
        gate = protocol["observation"]
        for name, path in protocol["paths"].items():
            descriptor_identity = identity(vectors(delta, path), metadata)
            event_identity = {f"{role}@{layer}": identity(event_vectors(delta, (layer, role)), metadata)
                              for layer, role in path_events(path)}
            best_event = max(value["top1"] for value in event_identity.values())
            gain = descriptor_identity["top1"] - best_event
            checks = {
                "identity": descriptor_identity["top1"] >= gate["family_pair_top1_min"],
                "surface": min(descriptor_identity["surface_top1"].values()) >= gate["surface_top1_min"],
                "synergy": gain >= gate["gain_over_best_event_min"],
            }
            medians = {f"{role}@{layer}": float(relative[:, layer, ROLE_INDEX[role]].median())
                       for layer, role in path_events(path)}
            path_metrics[name] = {
                "descriptor_identity": descriptor_identity, "event_identity": event_identity,
                "identity_gain_over_best_event": gain, "median_relative_norm": medians,
                "checks": checks, "qualified": all(checks.values()),
            }
        numeric_max = max(numeric_errors)
        bundle = {"roles": list(ROLES), "metadata": metadata,
                  "clean_minus_corrupt": delta, "relative_norm": relative}
        (OUT / "raw").mkdir(parents=True, exist_ok=True)
        torch.save(bundle, OUT / "raw/qwen3_hidden_response_paths.pt")
        summary = {
            "phase": PHASE, "campaign": CAMPAIGN,
            "tensor_shape": list(delta.shape), "numeric_relative_l2_max": numeric_max,
            "numeric_qualified": numeric_max <= 1e-6,
            "path_metrics": path_metrics,
            "qualified_paths": [name for name, value in path_metrics.items() if value["qualified"]],
            "runtime": {"placement": placement, "quantization": quant,
                        "all_finite": bool(torch.isfinite(delta).all() and torch.isfinite(relative).all()),
                        "finished_at_utc": datetime.now(timezone.utc).isoformat()},
            "claim_boundary": "descriptive full-width clean-minus-corrupt response paths only",
        }
        core.save(OUT / "analysis/qwen_path_observation.json", summary)
        core.save(OUT / "analysis/final.json", {
            "phase": PHASE, "campaign": CAMPAIGN,
            "qualified_paths": summary["qualified_paths"],
            "authorization": "run_phase1367_c056_qwen_path_identity_camera",
        })
        print(json.dumps({"shape": summary["tensor_shape"], "numeric": numeric_max,
                          "qualified_paths": summary["qualified_paths"], "path_metrics": path_metrics}, indent=2))
    finally:
        if model is not None:
            release_bf16(model)


if __name__ == "__main__":
    main()
